from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import tempfile
import asyncio
import logging
from pathlib import Path
from typing import Optional
import aiofiles
import uuid
import json
import re
import openai
from threading import Lock

from video_processor import VideoProcessor
from transcriber import Transcriber
from summarizer import Summarizer
from translator import Translator
from metadata_generator import MetadataGenerator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI视频转录器", version="1.0.0")

# CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")

# 创建临时目录
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)


def _get_whisper_model_name() -> str:
    """Resolve Whisper model from supported environment variables."""
    return (
        os.getenv("WHISPER_MODEL")
        or os.getenv("WHISPER_MODEL_SIZE")
        or "medium"
    )


TRANSCRIPTION_MODELS = {
    "fast": os.getenv("WHISPER_FAST_MODEL", "base"),
    "accurate": os.getenv("WHISPER_ACCURATE_MODEL") or _get_whisper_model_name(),
}
transcriber_lock = Lock()
transcriber_cache: dict[str, Transcriber] = {}


def _normalize_transcription_mode(mode: Optional[str]) -> str:
    value = (mode or "accurate").strip().lower()
    return value if value in TRANSCRIPTION_MODELS else "fast"


# Per-mode transcription parameters
TRANSCRIPTION_PARAMS = {
    "fast": {"beam_size": 1, "condition_on_previous_text": False},
    "accurate": {"beam_size": 5, "condition_on_previous_text": True},
}


def _get_transcriber(mode: Optional[str]) -> Transcriber:
    normalized_mode = _normalize_transcription_mode(mode)
    model_name = TRANSCRIPTION_MODELS[normalized_mode]
    params = TRANSCRIPTION_PARAMS[normalized_mode]
    cache_key = f"{model_name}_{normalized_mode}"

    with transcriber_lock:
        transcriber = transcriber_cache.get(cache_key)
        if transcriber is None:
            transcriber = Transcriber(model_size=model_name, **params)
            transcriber_cache[cache_key] = transcriber
        return transcriber

# 初始化处理器
video_processor = VideoProcessor()
transcriber = _get_transcriber("accurate")
summarizer = Summarizer()
translator = Translator()

# 存储任务状态 - 使用文件持久化
import json
import threading

TASKS_FILE = TEMP_DIR / "tasks.json"
tasks_lock = threading.Lock()

def load_tasks():
    """加载任务状态"""
    try:
        if TASKS_FILE.exists():
            with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return {}

def save_tasks(tasks_data):
    """保存任务状态"""
    try:
        with tasks_lock:
            with open(TASKS_FILE, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存任务状态失败: {e}")

async def broadcast_task_update(task_id: str, task_data: dict):
    """向所有连接的SSE客户端广播任务状态更新"""
    logger.info(f"广播任务更新: {task_id}, 状态: {task_data.get('status')}, 连接数: {len(sse_connections.get(task_id, []))}")
    if task_id in sse_connections:
        connections_to_remove = []
        for queue in sse_connections[task_id]:
            try:
                await queue.put(json.dumps(task_data, ensure_ascii=False))
                logger.debug(f"消息已发送到队列: {task_id}")
            except Exception as e:
                logger.warning(f"发送消息到队列失败: {e}")
                connections_to_remove.append(queue)
        
        # 移除断开的连接
        for queue in connections_to_remove:
            sse_connections[task_id].remove(queue)
        
        # 如果没有连接了，清理该任务的连接列表
        if not sse_connections[task_id]:
            del sse_connections[task_id]

# 启动时加载任务状态
tasks = load_tasks()
# 存储正在处理的URL，防止重复处理
processing_urls = set()
# 存储活跃的任务对象，用于控制和取消
active_tasks = {}
# 存储SSE连接，用于实时推送状态更新
sse_connections = {}

def _sanitize_title_for_filename(title: str) -> str:
    """将视频标题清洗为安全的文件名片段。"""
    if not title:
        return "untitled"
    # 仅保留字母数字、下划线、连字符与空格
    safe = re.sub(r"[^\w\-\s]", "", title)
    # 压缩空白并转为下划线
    safe = re.sub(r"\s+", "_", safe).strip("._-")
    # 最长限制，避免过长文件名问题
    return safe[:80] or "untitled"

@app.get("/")
async def read_root():
    """返回前端页面"""
    return FileResponse(str(PROJECT_ROOT / "static" / "index.html"))

@app.post("/api/models")
async def list_models(
    base_url: str = Form(default=""),
    api_key:  str = Form(default=""),
):
    """Proxy: fetch model list from any OpenAI-compatible API."""
    effective_key = api_key or os.getenv("OPENAI_API_KEY", "")
    effective_url = base_url.rstrip("/") or os.getenv("OPENAI_BASE_URL") or None

    if not effective_key:
        raise HTTPException(status_code=400, detail="API key is required")

    try:
        client = openai.OpenAI(api_key=effective_key, base_url=effective_url)
        resp   = await asyncio.to_thread(client.models.list)
        models = [{"id": m.id, "name": getattr(m, "name", m.id)} for m in resp.data]
        # Sort by id for readability
        models.sort(key=lambda x: x["id"])
        return {"data": models}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/process-video")
async def process_video(
    url: str = Form(default=""),
    summary_language: str = Form(default="zh"),
    transcription_mode: str = Form(default="accurate"),
    api_key:       str = Form(default=""),
    model_base_url: str = Form(default=""),
    model_id:      str = Form(default=""),
    verbatim_mode: str = Form(default=""),
    file: UploadFile = File(None),
):
    """
    处理视频链接或上传文件，返回任务ID
    """
    try:
        # Must have either a URL or a file
        has_file = file is not None and file.filename
        if not url and not has_file:
            raise HTTPException(status_code=400, detail="请提供视频链接或上传文件")

        uploaded_path = None
        if has_file:
            # Save uploaded file to temp dir
            unique_id = str(uuid.uuid4())[:8]
            # Sanitize filename
            safe_name = re.sub(r"[^\w\-.]", "_", file.filename)
            uploaded_filename = f"upload_{unique_id}_{safe_name}"
            uploaded_path = str(TEMP_DIR / uploaded_filename)
            async with aiofiles.open(uploaded_path, "wb") as out:
                content = await file.read()
                await out.write(content)
            logger.info(f"文件已保存: {uploaded_path} ({len(content)} bytes)")

        # 检查是否已经在处理相同的URL
        dedup_key = url if url else uploaded_path
        if url and url in processing_urls:
            # 查找现有任务
            for tid, task in tasks.items():
                if task.get("url") == url:
                    return {"task_id": tid, "message": "该视频正在处理中，请等待..."}
            
        # 生成唯一任务ID
        task_id = str(uuid.uuid4())
        
        # 标记URL为正在处理
        if url:
            processing_urls.add(url)
        
        # 初始化任务状态
        tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "开始处理视频...",
            "script": None,
            "summary": None,
            "error": None,
            "url": url or file.filename  # 保存URL/文件名用于去重
        }
        save_tasks(tasks)
        
        # 创建并跟踪异步任务
        is_verbatim = verbatim_mode.lower() in ("true", "1", "yes", "on") if verbatim_mode else False
        normalized_mode = _normalize_transcription_mode(transcription_mode)
        task = asyncio.create_task(process_video_task(
            task_id, url, summary_language, normalized_mode, api_key, model_base_url, model_id, is_verbatim,
            uploaded_file_path=uploaded_path,
            uploaded_file_name=file.filename if has_file else None,
        ))
        active_tasks[task_id] = task
        
        return {"task_id": task_id, "message": "任务已创建，正在处理中..."}
        
    except Exception as e:
        logger.error(f"处理视频时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

async def process_video_task(
    task_id: str,
    url: str,
    summary_language: str,
    transcription_mode: str = "accurate",
    api_key: str = "",
    model_base_url: str = "",
    model_id: str = "",
    verbatim_mode: bool = False,
    uploaded_file_path: str = None,
    uploaded_file_name: str = None,
):
    """
    异步处理视频任务
    """
    try:
        request_transcriber = _get_transcriber(transcription_mode)

        # ── 阶段一：优先尝试获取平台字幕（快速路径） ──────────────────────
        tasks[task_id].update({
            "status": "processing",
            "progress": 10,
            "message": "正在检测视频字幕...",
            "transcription_mode": transcription_mode,
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])
        await asyncio.sleep(0.1)

        # 如果前端传入了 API 凭据，创建专用 Summarizer（线程安全，覆盖全局实例）
        if api_key:
            effective_url = model_base_url.rstrip("/") or None
            request_summarizer = Summarizer(
                api_key=api_key,
                base_url=effective_url,
                model=model_id or None,
            )
            request_translator = Translator(
                api_key=api_key,
                base_url=effective_url,
                model=model_id or None,
            )
            logger.info(f"使用前端提供的 API Key，base_url={effective_url}, model={model_id or 'default'}")
        else:
            request_summarizer = summarizer  # 全局实例（使用环境变量）
            request_translator = translator  # 全局实例（使用环境变量）

        subtitle_text, sub_title, sub_lang = None, None, None

        if uploaded_file_path:
            # ── File upload path: extract audio from local file, skip subtitle fetch ──
            video_title = Path(uploaded_file_name or uploaded_file_path).stem

            tasks[task_id].update({
                "progress": 15,
                "message": "正在从文件提取音频..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            audio_path = await video_processor.extract_audio_from_file(uploaded_file_path, TEMP_DIR)

            tasks[task_id].update({
                "progress": 40,
                "message": "正在转录音频（Whisper）..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            raw_script = await request_transcriber.transcribe(audio_path)
        else:
            # ── URL path: try subtitles first, then download ──────────────────────
            subtitle_text, sub_title, sub_lang = await video_processor.fetch_subtitles(url, TEMP_DIR)

            if subtitle_text:
                # ── 快速路径：有字幕，跳过音频下载和 Whisper ──────────────────
                video_title = sub_title
                raw_script = subtitle_text
                # 把语言写入 transcriber，保持下游逻辑一致
                request_transcriber.last_detected_language = sub_lang

                tasks[task_id].update({
                    "progress": 40,
                    "message": f"字幕获取成功（{sub_lang}），正在处理文本..."
                })
                save_tasks(tasks)
                await broadcast_task_update(task_id, tasks[task_id])
            else:
                # ── 慢速路径：无字幕，下载音频 → Whisper 转录 ─────────────────
                tasks[task_id].update({
                    "progress": 15,
                    "message": "未找到字幕，正在下载视频音频..."
                })
                save_tasks(tasks)
                await broadcast_task_update(task_id, tasks[task_id])

                audio_path, video_title = await video_processor.download_and_convert(url, TEMP_DIR)

                tasks[task_id].update({
                    "progress": 35,
                    "message": "音频下载完成，准备转录..."
                })
                save_tasks(tasks)
                await broadcast_task_update(task_id, tasks[task_id])

                tasks[task_id].update({
                    "progress": 40,
                    "message": "正在转录音频（Whisper）..."
                })
                save_tasks(tasks)
                await broadcast_task_update(task_id, tasks[task_id])

                raw_script = await request_transcriber.transcribe(audio_path)

        # 将Whisper原始转录保存为Markdown文件，供下载/归档
        try:
            short_id = task_id.replace("-", "")[:6]
            safe_title = _sanitize_title_for_filename(video_title)
            raw_md_filename = f"raw_{safe_title}_{short_id}.md"
            raw_md_path = TEMP_DIR / raw_md_filename
            with open(raw_md_path, "w", encoding="utf-8") as f:
                source_label = url or uploaded_file_name or "file upload"
                content_raw = (raw_script or "") + f"\n\nsource: {source_label}\n"
                f.write(content_raw)

            # 记录原始转录文件路径（仅保存文件名，实际路径位于TEMP_DIR）
            tasks[task_id].update({
                "raw_script_file": raw_md_filename
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])
        except Exception as e:
            logger.error(f"保存原始转录Markdown失败: {e}")
        
        source_label = url or uploaded_file_name or "file upload"

        # ── Verbatim mode: skip AI optimization/summary, output raw transcript ──
        if verbatim_mode:
            logger.info(f"逐字转录模式，跳过AI优化和摘要")
            # Format the raw transcript into clean verbatim output
            verbatim_script = request_transcriber.format_verbatim(raw_script, video_title)
            
            tasks[task_id].update({
                "progress": 80,
                "message": "正在保存逐字转录..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])
            
            script_with_title = verbatim_script + f"\n\nsource: {source_label}\n"
            
            # Save transcript file
            script_filename = f"transcript_{safe_title}_{short_id}.md"
            script_path = TEMP_DIR / script_filename
            async with aiofiles.open(script_path, "w", encoding="utf-8") as f:
                await f.write(script_with_title)
            
            # Generate a minimal summary (just video info, no AI)
            detected_language = request_transcriber.get_detected_language(raw_script)
            summary_text = f"# {video_title}\n\n**Mode:** Verbatim Transcription\n**Language:** {detected_language or 'unknown'}\n\nFull verbatim transcript available in the Transcript tab.\n"
            summary_with_source = summary_text + f"\n\nsource: {source_label}\n"
            
            summary_filename = f"summary_{safe_title}_{short_id}.md"
            summary_path = TEMP_DIR / summary_filename
            async with aiofiles.open(summary_path, "w", encoding="utf-8") as f:
                await f.write(summary_with_source)
            
            task_result = {
                "status": "completed",
                "progress": 100,
                "message": "处理完成！",
                "video_title": video_title,
                "script": script_with_title,
                "summary": summary_with_source,
                "script_path": script_filename,
                "summary_path": summary_filename,
                "short_id": short_id,
                "safe_title": safe_title,
                "detected_language": detected_language,
                "summary_language": summary_language,
                "transcription_mode": transcription_mode,
            }
            
            # ── Generate structured YAML metadata in verbatim mode too ──────
            metadata_filename = None
            try:
                tasks[task_id].update({"progress": 90, "message": "Generating structured metadata (YAML)…"})
                save_tasks(tasks)
                await broadcast_task_update(task_id, tasks[task_id])

                meta_gen = MetadataGenerator(
                    client=request_summarizer.client,
                    fast_model=request_summarizer.fast_model,
                )

                # Duration from last timestamp in raw transcript
                duration_sec = 0.0
                try:
                    import re as _re
                    ts_matches = _re.findall(
                        r"\*\*\[(?:\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)\]\*\*",
                        raw_script or "",
                    )
                    if ts_matches:
                        last_ts = ts_matches[-1]
                        parts = last_ts.split(":")
                        if len(parts) == 3:
                            duration_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                        elif len(parts) == 2:
                            duration_sec = int(parts[0]) * 60 + float(parts[1])
                except Exception:
                    pass

                yaml_doc = await meta_gen.generate(
                    raw_transcript=raw_script or "",
                    video_title=video_title,
                    source_url=url or "",
                    detected_language=detected_language or "en",
                    duration_seconds=duration_sec,
                    task_id=task_id,
                )
                yaml_content = MetadataGenerator.to_yaml(yaml_doc)

                metadata_filename = f"metadata_{safe_title}_{short_id}.yaml"
                metadata_path_file = TEMP_DIR / metadata_filename
                async with aiofiles.open(metadata_path_file, "w", encoding="utf-8") as f:
                    await f.write(yaml_content)
                logger.info(f"YAML metadata saved (verbatim): {metadata_filename}")
            except Exception as e:
                logger.error(f"YAML metadata generation failed in verbatim mode (non-fatal): {e}")

            if metadata_filename:
                task_result["metadata_path"] = metadata_filename

            tasks[task_id].update(task_result)
            save_tasks(tasks)
            logger.info(f"逐字转录任务完成: {task_id}")
            await broadcast_task_update(task_id, tasks[task_id])
            
            processing_urls.discard(url)
            if task_id in active_tasks:
                del active_tasks[task_id]
            return

        # 更新状态：优化转录文本
        tasks[task_id].update({
            "progress": 55,
            "message": "正在优化转录文本..."
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])
        
        # 优化转录文本：修正错别字，按含义分段
        script = await request_summarizer.optimize_transcript(raw_script)
        
        # 为转录文本添加标题，并在结尾添加来源链接
        script_with_title = f"# {video_title}\n\n{script}\n\nsource: {source_label}\n"
        
        # 检查是否需要翻译
        detected_language = request_transcriber.get_detected_language(raw_script)
        logger.info(f"检测到的语言: {detected_language}, 摘要语言: {summary_language}")
        
        translation_content = None
        translation_filename = None
        translation_path = None
        
        if detected_language and request_translator.should_translate(detected_language, summary_language):
            logger.info(f"需要翻译: {detected_language} -> {summary_language}")
            # 更新状态：生成翻译
            tasks[task_id].update({
                "progress": 70,
                "message": "正在生成翻译..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])
            
            # 翻译转录文本
            translation_content = await request_translator.translate_text(script, summary_language, detected_language)
            translation_with_title = f"# {video_title}\n\n{translation_content}\n\nsource: {source_label}\n"
            
            # 保存翻译到文件
            translation_filename = f"translation_{safe_title}_{short_id}.md"
            translation_path = TEMP_DIR / translation_filename
            async with aiofiles.open(translation_path, "w", encoding="utf-8") as f:
                await f.write(translation_with_title)
        else:
            logger.info(f"不需要翻译: detected_language={detected_language}, summary_language={summary_language}, should_translate={request_translator.should_translate(detected_language, summary_language) if detected_language else 'N/A'}")
        
        # 更新状态：生成摘要
        tasks[task_id].update({
            "progress": 80,
            "message": "正在生成摘要..."
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])
        
        # 生成摘要
        summary = await request_summarizer.summarize(script, summary_language, video_title)
        summary_with_source = summary + f"\n\nsource: {source_label}\n"
        
        # 保存优化后的转录文本到文件
        script_filename = f"transcript_{task_id}.md"
        script_path = TEMP_DIR / script_filename
        async with aiofiles.open(script_path, "w", encoding="utf-8") as f:
            await f.write(script_with_title)
        
        # 重命名为新规则：transcript_标题_短ID.md
        new_script_filename = f"transcript_{safe_title}_{short_id}.md"
        new_script_path = TEMP_DIR / new_script_filename
        try:
            if script_path.exists():
                script_path.rename(new_script_path)
                script_path = new_script_path
        except Exception as _:
            # 如重命名失败，继续使用原路径
            pass

        # 保存摘要到文件（summary_标题_短ID.md）
        summary_filename = f"summary_{safe_title}_{short_id}.md"
        summary_path = TEMP_DIR / summary_filename
        async with aiofiles.open(summary_path, "w", encoding="utf-8") as f:
            await f.write(summary_with_source)

        # ── Generate structured YAML metadata ─────────────────────────────
        metadata_filename = None
        try:
            tasks[task_id].update({
                "progress": 93,
                "message": "Generating structured metadata (YAML)…",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            meta_gen = MetadataGenerator(
                client=request_summarizer.client,
                fast_model=request_summarizer.fast_model,
            )

            # Duration: try to parse from raw transcript or fall back to 0
            duration_sec = 0.0
            try:
                import re as _re
                ts_matches = _re.findall(
                    r"\*\*\[(?:\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)\]\*\*",
                    raw_script or "",
                )
                if ts_matches:
                    last_ts = ts_matches[-1]
                    parts = last_ts.split(":")
                    if len(parts) == 3:
                        duration_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                    elif len(parts) == 2:
                        duration_sec = int(parts[0]) * 60 + float(parts[1])
            except Exception:
                pass

            yaml_doc = await meta_gen.generate(
                raw_transcript=raw_script or "",
                video_title=video_title,
                source_url=url or "",
                detected_language=detected_language or "en",
                duration_seconds=duration_sec,
                task_id=task_id,
            )
            yaml_content = MetadataGenerator.to_yaml(yaml_doc)

            metadata_filename = f"metadata_{safe_title}_{short_id}.yaml"
            metadata_path = TEMP_DIR / metadata_filename
            async with aiofiles.open(metadata_path, "w", encoding="utf-8") as f:
                await f.write(yaml_content)
            logger.info(f"YAML metadata saved: {metadata_filename}")
        except Exception as e:
            logger.error(f"YAML metadata generation failed (non-fatal): {e}")

        # 更新状态：完成
        task_result = {
            "status": "completed",
            "progress": 100,
            "message": "处理完成！",
            "video_title": video_title,
            "script": script_with_title,
            "summary": summary_with_source,
            "script_path": script_path.name,
            "summary_path": summary_path.name,
            "short_id": short_id,
            "safe_title": safe_title,
            "detected_language": detected_language,
            "summary_language": summary_language,
            "transcription_mode": transcription_mode,
        }

        if metadata_filename:
            task_result["metadata_path"] = metadata_filename

        # 如果有翻译，添加翻译信息
        if translation_content and translation_path:
            task_result.update({
                "translation": translation_with_title,
                "translation_path": translation_path.name,
                "translation_filename": translation_filename
            })
        
        tasks[task_id].update(task_result)
        save_tasks(tasks)
        logger.info(f"任务完成，准备广播最终状态: {task_id}")
        await broadcast_task_update(task_id, tasks[task_id])
        logger.info(f"最终状态已广播: {task_id}")
        
        # 从处理列表中移除URL
        processing_urls.discard(url)
        
        # 从活跃任务列表中移除
        if task_id in active_tasks:
            del active_tasks[task_id]
        
        # 不要立即删除临时文件！保留给用户下载
        # 文件会在一定时间后自动清理或用户手动清理
            
    except Exception as e:
        logger.error(f"任务 {task_id} 处理失败: {str(e)}")
        # 从处理列表中移除URL
        processing_urls.discard(url)
        
        # 从活跃任务列表中移除
        if task_id in active_tasks:
            del active_tasks[task_id]
            
        tasks[task_id].update({
            "status": "error",
            "error": str(e),
            "message": f"处理失败: {str(e)}"
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

@app.get("/api/task-status/{task_id}")
async def get_task_status(task_id: str):
    """
    获取任务状态
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return tasks[task_id]

@app.get("/api/task-stream/{task_id}")
async def task_stream(task_id: str):
    """
    SSE实时任务状态流
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    async def event_generator():
        # 创建任务专用的队列
        queue = asyncio.Queue()
        
        # 将队列添加到连接列表
        if task_id not in sse_connections:
            sse_connections[task_id] = []
        sse_connections[task_id].append(queue)
        
        try:
            # 立即发送当前状态
            current_task = tasks.get(task_id, {})
            yield f"data: {json.dumps(current_task, ensure_ascii=False)}\n\n"
            
            # 持续监听状态更新
            while True:
                try:
                    # 等待状态更新，超时时间30秒发送心跳
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {data}\n\n"
                    
                    # 如果任务完成或失败，结束流
                    task_data = json.loads(data)
                    if task_data.get("status") in ["completed", "error"]:
                        break
                        
                except asyncio.TimeoutError:
                    # 发送心跳保持连接
                    yield f"data: {json.dumps({'type': 'heartbeat'}, ensure_ascii=False)}\n\n"
                    
        except asyncio.CancelledError:
            logger.info(f"SSE连接被取消: {task_id}")
        except Exception as e:
            logger.error(f"SSE流异常: {e}")
        finally:
            # 清理连接
            if task_id in sse_connections and queue in sse_connections[task_id]:
                sse_connections[task_id].remove(queue)
                if not sse_connections[task_id]:
                    del sse_connections[task_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """
    直接从temp目录下载文件（简化方案）
    """
    try:
        # 检查文件扩展名安全性
        if not (filename.endswith('.md') or filename.endswith('.yaml')):
            raise HTTPException(status_code=400, detail="仅支持下载.md和.yaml文件")
        
        # 检查文件名格式（防止路径遍历攻击）
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="文件名格式无效")
            
        file_path = TEMP_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")

        media_type = "application/x-yaml" if filename.endswith('.yaml') else "text/markdown"
        return FileResponse(
            file_path,
            filename=filename,
            media_type=media_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")


@app.get("/api/download-pdf/{task_id}")
async def download_pdf(task_id: str):
    """Generate and download a PDF of the transcript for a completed task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task_data = tasks[task_id]
    if task_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="任务未完成")

    script = task_data.get("script") or ""
    video_title = task_data.get("video_title") or "Transcript"
    safe_title = task_data.get("safe_title") or "transcript"
    short_id = task_data.get("short_id") or task_id[:6]
    metadata_path = task_data.get("metadata_path")  # New: pass metadata

    try:
        pdf_path = await asyncio.to_thread(
            _generate_pdf, script, video_title, safe_title, short_id, metadata_path
        )
        pdf_filename = pdf_path.name
        return FileResponse(
            str(pdf_path),
            filename=pdf_filename,
            media_type="application/pdf",
        )
    except Exception as e:
        logger.error(f"PDF生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@app.get("/api/download-srt/{task_id}")
async def download_srt(task_id: str):
    """Generate and download an SRT subtitle file for a completed task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task_data = tasks[task_id]
    if task_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="任务未完成")

    script = task_data.get("script") or ""
    safe_title = task_data.get("safe_title") or "transcript"
    short_id = task_data.get("short_id") or task_id[:6]

    try:
        srt_content = transcriber.generate_srt(script)
        if not srt_content.strip():
            raise HTTPException(
                status_code=400,
                detail="No timestamped segments found in transcript",
            )

        srt_filename = f"subtitle_{safe_title}_{short_id}.srt"
        srt_path = TEMP_DIR / srt_filename
        srt_path.write_text(srt_content, encoding="utf-8")
        logger.info(f"SRT generated: {srt_path}")

        return FileResponse(
            str(srt_path),
            filename=srt_filename,
            media_type="application/x-subrip",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SRT生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"SRT generation failed: {str(e)}")


@app.get("/api/download-docx/{task_id}")
async def download_docx(task_id: str):
    """Generate and download a Word (.docx) document of the transcript."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task_data = tasks[task_id]
    if task_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="任务未完成")

    script = task_data.get("script") or ""
    video_title = task_data.get("video_title") or "Transcript"
    safe_title = task_data.get("safe_title") or "transcript"
    short_id = task_data.get("short_id") or task_id[:6]
    metadata_path = task_data.get("metadata_path")  # New: pass metadata

    try:
        docx_path = await asyncio.to_thread(
            _generate_docx, script, video_title, safe_title, short_id, metadata_path
        )
        return FileResponse(
            str(docx_path),
            filename=docx_path.name,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except Exception as e:
        logger.error(f"DOCX生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"DOCX generation failed: {str(e)}")


def _generate_docx(markdown_text: str, title: str, safe_title: str, short_id: str, metadata_path: Optional[str] = None) -> Path:
    """Render markdown transcript text into a styled Word (.docx) file."""
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    # ── Add YAML metadata section if available ──────────────────────────
    if metadata_path:
        yaml_path = TEMP_DIR / metadata_path
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f)
                
                # Add metadata header
                p = doc.add_heading("Lesson Metadata", level=1)
                p.paragraph_format.space_after = Pt(8)
                
                # Key info fields (all metadata fields)
                fields_to_show = [
                    "document_id", "course_name", "module_name", "lesson_name",
                    "lesson_type", "source_type", "instructor", "language",
                    "duration_seconds", "source_video_url"
                ]
                for key in fields_to_show:
                    val = meta.get(key, "")
                    if val or key in ["course_name", "module_name"]:  # Show even if empty
                        p = doc.add_paragraph()
                        run_key = p.add_run(f"{key.replace('_', ' ').title()}: ")
                        run_key.bold = True
                        run_key.font.size = Pt(10)
                        run_val = p.add_run(str(val) if val else "")
                        run_val.font.size = Pt(10)
                
                # Timestamps covered
                ts_covered = meta.get("timestamps_covered", {})
                if ts_covered:
                    p = doc.add_paragraph()
                    run_key = p.add_run("Timestamps Covered: ")
                    run_key.bold = True
                    run_key.font.size = Pt(10)
                    start = ts_covered.get("start", "")
                    end = ts_covered.get("end", "")
                    run_val = p.add_run(f"{start} to {end}")
                    run_val.font.size = Pt(10)
                
                # Topics
                topics = meta.get("topics", [])
                if topics:
                    p = doc.add_paragraph()
                    run_key = p.add_run("Topics: ")
                    run_key.bold = True
                    run_key.font.size = Pt(10)
                    run_val = p.add_run(", ".join(str(t) for t in topics))
                    run_val.font.size = Pt(10)
                
                # Pillars
                pillars = meta.get("pillars", [])
                if pillars:
                    p = doc.add_paragraph()
                    run_key = p.add_run("Pillars: ")
                    run_key.bold = True
                    run_key.font.size = Pt(10)
                    run_val = p.add_run(", ".join(str(p) for p in pillars))
                    run_val.font.size = Pt(10)
                
                doc.add_paragraph()  # spacing
                
                # Summary
                summary = meta.get("summary", "")
                if summary:
                    p = doc.add_heading("Summary", level=2)
                    p.paragraph_format.space_after = Pt(4)
                    p = doc.add_paragraph(summary)
                    p.paragraph_format.space_after = Pt(8)
                
                # Key Takeaways
                takeaways = meta.get("key_takeaways", [])
                if takeaways:
                    p = doc.add_heading("Key Takeaways", level=2)
                    p.paragraph_format.space_after = Pt(4)
                    for item in takeaways:
                        p = doc.add_paragraph(str(item), style='List Bullet')
                        p.paragraph_format.space_after = Pt(2)
                    doc.add_paragraph()  # spacing
                
                # Definitions
                definitions = meta.get("definitions", [])
                if definitions:
                    p = doc.add_heading("Definitions", level=2)
                    p.paragraph_format.space_after = Pt(4)
                    for defn in definitions:
                        term = defn.get("term", "")
                        meaning = defn.get("meaning", "")
                        if term:
                            p = doc.add_paragraph()
                            run_term = p.add_run(f"{term}: ")
                            run_term.bold = True
                            run_term.font.size = Pt(10)
                            run_meaning = p.add_run(meaning)
                            run_meaning.font.size = Pt(10)
                    doc.add_paragraph()  # spacing
                
                # Visual Notes
                visual_notes = meta.get("visual_notes", [])
                if visual_notes:
                    p = doc.add_heading("Visual Notes", level=2)
                    p.paragraph_format.space_after = Pt(4)
                    for vn in visual_notes:
                        ts = vn.get("timestamp", "")
                        vtype = vn.get("type", "")
                        content = vn.get("content", "")
                        p = doc.add_paragraph()
                        run_ts = p.add_run(f"[{ts}] ")
                        run_ts.bold = True
                        run_ts.font.size = Pt(9)
                        run_ts.font.color.rgb = RGBColor(140, 90, 30)
                        run_content = p.add_run(f"{vtype}: {content}")
                        run_content.font.size = Pt(9)
                    doc.add_paragraph()  # spacing
                
                # Transcript Blocks (Full YAML format)
                transcript_blocks = meta.get("transcript_blocks", [])
                if transcript_blocks:
                    p = doc.add_heading("Transcript Blocks", level=2)
                    p.paragraph_format.space_after = Pt(4)
                    for block in transcript_blocks:
                        block_id = block.get("block_id", "")
                        start = block.get("start", "")
                        end = block.get("end", "")
                        speaker = block.get("speaker", "")
                        text = block.get("text", "")
                        retrieval_tags = block.get("retrieval_tags", [])
                        
                        # Block header
                        p = doc.add_paragraph()
                        run_header = p.add_run(f"Block: {block_id}")
                        run_header.bold = True
                        run_header.font.size = Pt(10)
                        run_header.font.color.rgb = RGBColor(40, 100, 180)
                        
                        # Time and speaker
                        p = doc.add_paragraph()
                        run_meta = p.add_run(f"[{start} - {end}] | Speaker: {speaker}")
                        run_meta.italic = True
                        run_meta.font.size = Pt(9)
                        run_meta.font.color.rgb = RGBColor(100, 100, 100)
                        
                        # Text
                        p = doc.add_paragraph(text[:500] + "..." if len(text) > 500 else text)
                        p.paragraph_format.space_after = Pt(2)
                        
                        # Tags
                        if retrieval_tags:
                            p = doc.add_paragraph()
                            run_tags = p.add_run(f"Tags: {', '.join(retrieval_tags)}")
                            run_tags.italic = True
                            run_tags.font.size = Pt(8)
                            run_tags.font.color.rgb = RGBColor(120, 120, 120)
                        
                        p.paragraph_format.space_after = Pt(6)
                    doc.add_paragraph()  # spacing
                
                # FAQ Candidates
                faq = meta.get("faq_candidates", [])
                if faq:
                    p = doc.add_heading("FAQ Candidates", level=2)
                    p.paragraph_format.space_after = Pt(4)
                    for item in faq:
                        q = item.get("question", "")
                        a = item.get("answer", "")
                        if q:
                            p = doc.add_paragraph()
                            run_q = p.add_run(f"Q: {q}")
                            run_q.bold = True
                            run_q.font.size = Pt(10)
                            p = doc.add_paragraph(f"A: {a}")
                            p.paragraph_format.left_indent = Pt(12)
                            p.paragraph_format.space_after = Pt(4)
                    doc.add_paragraph()  # spacing
                
                # Searchable Keywords
                keywords = meta.get("searchable_keywords", [])
                if keywords:
                    p = doc.add_heading("Searchable Keywords", level=2)
                    p.paragraph_format.space_after = Pt(4)
                    p = doc.add_paragraph(", ".join(str(k) for k in keywords))
                    p.paragraph_format.space_after = Pt(8)
                
                # Add separator
                doc.add_paragraph("_" * 70)
                doc.add_paragraph()
            except Exception as e:
                logger.warning(f"Failed to load metadata for DOCX: {e}")

    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(30, 30, 30)

    lines = markdown_text.split("\n")
    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.lower().startswith("source:"):
            continue

        # Heading: # Title
        if stripped.startswith("# "):
            text = stripped.lstrip("# ").strip()
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            p = doc.add_heading(text, level=1)
            continue

        # Sub-heading: ## ...
        if stripped.startswith("## "):
            text = stripped.lstrip("# ").strip()
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            doc.add_heading(text, level=2)
            continue

        # Timestamp line: **[00:00 - 00:05]**
        ts_match = re.match(r"\*\*\[(.+?)\]\*\*", stripped)
        if ts_match:
            p = doc.add_paragraph()
            run = p.add_run(f"[{ts_match.group(1)}]")
            run.bold = True
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(140, 90, 30)
            continue

        # Verbatim timestamp: [HH:MM:SS] text
        verb_match = re.match(r"(\[\d{2}:\d{2}:\d{2}\])\s*(.*)", stripped)
        if verb_match:
            p = doc.add_paragraph()
            ts_run = p.add_run(verb_match.group(1) + "  ")
            ts_run.bold = True
            ts_run.font.size = Pt(9)
            ts_run.font.color.rgb = RGBColor(140, 90, 30)
            txt_run = p.add_run(verb_match.group(2))
            txt_run.font.size = Pt(11)
            continue

        # Metadata lines (bold key-value)
        meta_match = re.match(r"\*\*(.+?):\*\*\s*(.*)", stripped)
        if meta_match:
            p = doc.add_paragraph()
            run = p.add_run(f"{meta_match.group(1)}: {meta_match.group(2)}")
            run.bold = True
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(80, 80, 80)
            continue

        # Regular paragraph text
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        doc.add_paragraph(text)

    docx_filename = f"transcript_{safe_title}_{short_id}.docx"
    docx_path = TEMP_DIR / docx_filename
    doc.save(str(docx_path))
    logger.info(f"DOCX generated: {docx_path}")
    return docx_path


def _pdf_safe(text: str) -> str:
    """Replace Unicode characters that fpdf's built-in Helvetica font cannot encode."""
    if not text:
        return text
    replacements = {
        "\u2018": "'", "\u2019": "'",   # left/right single quotes
        "\u201c": '"', "\u201d": '"',   # left/right double quotes
        "\u2013": "-", "\u2014": "-",   # en-dash, em-dash
        "\u2026": "...",                 # ellipsis
        "\u00b7": "-",                  # middle dot
        "\u2022": "-",                  # bullet
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Final safety: drop anything still outside Latin-1 range
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _generate_pdf(markdown_text: str, title: str, safe_title: str, short_id: str, metadata_path: Optional[str] = None) -> Path:
    """Render markdown transcript text into a styled PDF file."""
    from fpdf import FPDF

    class TranscriptPDF(FPDF):
        def __init__(self, doc_title: str):
            super().__init__()
            self.doc_title = doc_title

        def header(self):
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, self.doc_title, align="L")
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(160, 160, 160)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    pdf = TranscriptPDF(title)
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Add YAML metadata section if available ──────────────────────────
    if metadata_path:
        yaml_path = TEMP_DIR / metadata_path
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f)
                
                # Metadata header
                pdf.set_font("Helvetica", "B", 16)
                pdf.set_text_color(30, 30, 30)
                pdf.cell(0, 10, "Lesson Metadata", ln=True)
                pdf.ln(4)
                
                # Key info fields (all metadata fields)
                pdf.set_font("Helvetica", "", 10)
                fields_to_show = [
                    "document_id", "course_name", "module_name", "lesson_name", 
                    "lesson_type", "source_type", "instructor", "language", 
                    "duration_seconds", "source_video_url"
                ]
                for key in fields_to_show:
                    val = meta.get(key, "")
                    if val or key in ["course_name", "module_name"]:  # Show even if empty for course/module
                        pdf.set_x(pdf.l_margin)
                        pdf.set_font("Helvetica", "B", 10)
                        pdf.cell(55, 6, f"{key.replace('_', ' ').title()}: ", ln=False)
                        pdf.set_font("Helvetica", "", 10)
                        pdf.multi_cell(0, 6, _pdf_safe(str(val) if val else ""))
                        pdf.set_x(pdf.l_margin)
                
                # Timestamps covered
                ts_covered = meta.get("timestamps_covered", {})
                if ts_covered:
                    pdf.set_x(pdf.l_margin)
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.cell(55, 6, "Timestamps Covered: ", ln=False)
                    pdf.set_font("Helvetica", "", 10)
                    start = ts_covered.get("start", "")
                    end = ts_covered.get("end", "")
                    pdf.cell(0, 6, _pdf_safe(f"{start} to {end}"), ln=True)
                    pdf.set_x(pdf.l_margin)
                
                # Topics
                topics = meta.get("topics", [])
                if topics:
                    pdf.set_x(pdf.l_margin)
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.cell(55, 6, "Topics: ", ln=False)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.multi_cell(0, 6, _pdf_safe(", ".join(str(t) for t in topics)))
                    pdf.set_x(pdf.l_margin)
                
                # Pillars
                pillars = meta.get("pillars", [])
                if pillars:
                    pdf.set_x(pdf.l_margin)
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.cell(55, 6, "Pillars: ", ln=False)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.multi_cell(0, 6, _pdf_safe(", ".join(str(p) for p in pillars)))
                    pdf.set_x(pdf.l_margin)
                
                pdf.ln(4)
                
                # Summary
                summary = meta.get("summary", "")
                if summary:
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.cell(0, 8, "Summary", ln=True)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.multi_cell(0, 5, _pdf_safe(summary))
                    pdf.ln(4)
                
                # Key Takeaways
                takeaways = meta.get("key_takeaways", [])
                if takeaways:
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.cell(0, 8, "Key Takeaways", ln=True)
                    pdf.set_font("Helvetica", "", 10)
                    for item in takeaways:
                        pdf.set_x(pdf.l_margin)
                        # Calculate proper indent and bullet position
                        x_start = pdf.get_x()
                        y_start = pdf.get_y()
                        pdf.cell(5, 5, "-", ln=False)
                        pdf.set_xy(x_start + 5, y_start)
                        pdf.multi_cell(0, 5, _pdf_safe(str(item)))
                        pdf.set_x(pdf.l_margin)
                    pdf.ln(4)
                
                # Definitions
                definitions = meta.get("definitions", [])
                if definitions:
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.cell(0, 8, "Definitions", ln=True)
                    pdf.set_font("Helvetica", "", 10)
                    for defn in definitions:
                        term = defn.get("term", "")
                        meaning = defn.get("meaning", "")
                        if term:
                            pdf.set_font("Helvetica", "B", 10)
                            pdf.cell(0, 5, _pdf_safe(term), ln=True)
                            pdf.set_font("Helvetica", "", 10)
                            pdf.multi_cell(0, 5, _pdf_safe(meaning))
                            pdf.ln(2)
                    pdf.ln(2)
                
                # Visual Notes
                visual_notes = meta.get("visual_notes", [])
                if visual_notes:
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.cell(0, 8, "Visual Notes", ln=True)
                    for vn in visual_notes:
                        ts = vn.get("timestamp", "")
                        vtype = vn.get("type", "")
                        content = vn.get("content", "")
                        pdf.set_x(pdf.l_margin)
                        pdf.set_font("Helvetica", "B", 9)
                        pdf.set_text_color(140, 90, 30)
                        pdf.cell(25, 5, _pdf_safe(f"[{ts}]"), ln=False)
                        pdf.set_font("Helvetica", "", 9)
                        pdf.set_text_color(30, 30, 30)
                        pdf.multi_cell(0, 5, _pdf_safe(f"{vtype}: {content}"))
                        pdf.set_x(pdf.l_margin)
                    pdf.ln(4)
                
                # Transcript Blocks (Full YAML format)
                transcript_blocks = meta.get("transcript_blocks", [])
                if transcript_blocks:
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.cell(0, 8, "Transcript Blocks", ln=True)
                    pdf.ln(2)
                    for block in transcript_blocks:
                        block_id = block.get("block_id", "")
                        start = block.get("start", "")
                        end = block.get("end", "")
                        speaker = block.get("speaker", "")
                        text = block.get("text", "")
                        cleaned_text = block.get("cleaned_text", "")
                        retrieval_tags = block.get("retrieval_tags", [])
                        
                        # Block header
                        pdf.set_x(pdf.l_margin)
                        pdf.set_font("Helvetica", "B", 10)
                        pdf.set_text_color(40, 100, 180)
                        pdf.cell(0, 6, _pdf_safe(f"Block: {block_id}"), ln=True)
                        pdf.set_x(pdf.l_margin)
                        
                        # Time range and speaker
                        pdf.set_font("Helvetica", "I", 9)
                        pdf.set_text_color(100, 100, 100)
                        pdf.cell(0, 5, _pdf_safe(f"[{start} - {end}] | Speaker: {speaker}"), ln=True)
                        pdf.set_x(pdf.l_margin)
                        
                        # Text
                        pdf.set_font("Helvetica", "", 9)
                        pdf.set_text_color(30, 30, 30)
                        block_text = text[:500] + "..." if len(text) > 500 else text
                        pdf.multi_cell(0, 4, _pdf_safe(block_text))
                        pdf.set_x(pdf.l_margin)
                        
                        # Retrieval tags
                        if retrieval_tags:
                            pdf.set_font("Helvetica", "I", 8)
                            pdf.set_text_color(120, 120, 120)
                            pdf.cell(0, 4, _pdf_safe(f"Tags: {', '.join(retrieval_tags)}"), ln=True)
                            pdf.set_x(pdf.l_margin)
                        
                        pdf.ln(3)
                    pdf.ln(2)
                
                # FAQ Candidates
                faq = meta.get("faq_candidates", [])
                if faq:
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.cell(0, 8, "FAQ Candidates", ln=True)
                    for item in faq:
                        q = item.get("question", "")
                        a = item.get("answer", "")
                        if q:
                            pdf.set_x(pdf.l_margin)
                            pdf.set_font("Helvetica", "B", 10)
                            pdf.multi_cell(0, 5, _pdf_safe(f"Q: {q}"))
                            pdf.set_x(pdf.l_margin)
                            pdf.set_font("Helvetica", "", 10)
                            pdf.set_x(15)  # indent answer
                            pdf.multi_cell(0, 5, _pdf_safe(f"A: {a}"))
                            pdf.set_x(pdf.l_margin)
                    pdf.ln(2)
                
                # Searchable Keywords
                keywords = meta.get("searchable_keywords", [])
                if keywords:
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.cell(0, 8, "Searchable Keywords", ln=True)
                    pdf.set_font("Helvetica", "", 9)
                    pdf.set_text_color(80, 80, 80)
                    pdf.multi_cell(0, 5, _pdf_safe(", ".join(str(k) for k in keywords)))
                    pdf.ln(4)
                
                # Add separator
                pdf.set_draw_color(200, 200, 200)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(8)
            except Exception as e:
                logger.warning(f"Failed to load metadata for PDF: {e}")

    # Strip markdown formatting for clean PDF text
    lines = markdown_text.split("\n")
    for line in lines:
        stripped = line.strip()

        # Skip empty lines → small spacing
        if not stripped:
            pdf.ln(4)
            continue

        # Remove trailing "source: ..." line
        if stripped.lower().startswith("source:"):
            continue

        # Heading: # Title
        if stripped.startswith("# "):
            text = stripped.lstrip("# ").strip()
            # Strip bold markdown
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 9, _pdf_safe(text))
            pdf.ln(4)
            continue

        # Sub-heading: ## ...
        if stripped.startswith("## "):
            text = stripped.lstrip("# ").strip()
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 8, _pdf_safe(text))
            pdf.ln(3)
            continue

        # Timestamp line: **[00:00 - 00:05]** or [HH:MM:SS] text
        ts_match = re.match(r"\*\*\[(.+?)\]\*\*", stripped)
        if ts_match:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(140, 90, 30)
            pdf.cell(0, 6, _pdf_safe(f"[{ts_match.group(1)}]"))
            pdf.ln(5)
            continue

        # Verbatim timestamp: [HH:MM:SS] text
        verb_match = re.match(r"(\[\d{2}:\d{2}:\d{2}\])\s*(.*)", stripped)
        if verb_match:
            ts_part = verb_match.group(1)
            text_part = verb_match.group(2)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(140, 90, 30)
            ts_w = pdf.get_string_width(ts_part + "  ") + 2
            pdf.cell(ts_w, 6, ts_part)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 6, _pdf_safe(text_part))
            pdf.ln(2)
            continue

        # Metadata lines (bold key-value)
        meta_match = re.match(r"\*\*(.+?):\*\*\s*(.*)", stripped)
        if meta_match:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 6, _pdf_safe(f"{meta_match.group(1)}: {meta_match.group(2)}"))
            pdf.ln(5)
            continue

        # Regular paragraph text — strip remaining bold markers
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 6, _pdf_safe(text))
        pdf.ln(2)

    pdf_filename = f"transcript_{safe_title}_{short_id}.pdf"
    pdf_path = TEMP_DIR / pdf_filename
    pdf.output(str(pdf_path))
    logger.info(f"PDF generated: {pdf_path}")
    return pdf_path


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """
    取消并删除任务
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 如果任务还在运行，先取消它
    if task_id in active_tasks:
        task = active_tasks[task_id]
        if not task.done():
            task.cancel()
            logger.info(f"任务 {task_id} 已被取消")
        del active_tasks[task_id]
    
    # 从处理URL列表中移除
    task_url = tasks[task_id].get("url")
    if task_url:
        processing_urls.discard(task_url)
    
    # 删除任务记录
    del tasks[task_id]
    return {"message": "任务已取消并删除"}

@app.get("/api/tasks/active")
async def get_active_tasks():
    """
    获取当前活跃任务列表（用于调试）
    """
    active_count = len(active_tasks)
    processing_count = len(processing_urls)
    return {
        "active_tasks": active_count,
        "processing_urls": processing_count,
        "task_ids": list(active_tasks.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
