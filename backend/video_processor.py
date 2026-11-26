import os
import re
import shutil
import subprocess
import uuid
import yt_dlp
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class VideoProcessor:
    """视频处理器，使用yt-dlp下载和转换视频"""
    
    def __init__(self):
        self.ffmpeg_path, self.ffprobe_path = self._resolve_ffmpeg_binaries()
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': '%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                # 直接在提取阶段转换为单声道 16k（空间小且稳定）
                'preferredcodec': 'wav',
                'preferredquality': '0'
            }],
            # 全局FFmpeg参数：单声道 + 16k 采样率
            'postprocessor_args': ['-ac', '1', '-ar', '16000'],
            'prefer_ffmpeg': True,
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,  # 强制只下载单个视频，不下载播放列表
        }
        if self.ffmpeg_path:
            self.ydl_opts['ffmpeg_location'] = str(Path(self.ffmpeg_path).parent)
            logger.info(f"使用FFmpeg路径: {self.ffmpeg_path}")

    def _resolve_ffmpeg_binaries(self) -> tuple[Optional[str], Optional[str]]:
        """解析 ffmpeg/ffprobe 可执行文件路径，尽量避免依赖启动时 PATH。"""
        env_ffmpeg = os.getenv("FFMPEG_BINARY") or os.getenv("FFMPEG_PATH")
        env_ffprobe = os.getenv("FFPROBE_BINARY") or os.getenv("FFPROBE_PATH")

        ffmpeg_path = self._find_binary("ffmpeg", env_ffmpeg)
        ffprobe_path = self._find_binary("ffprobe", env_ffprobe)

        if ffmpeg_path and ffprobe_path:
            return ffmpeg_path, ffprobe_path

        for base_dir in self._candidate_ffmpeg_dirs():
            candidate_ffmpeg = base_dir / "ffmpeg.exe"
            candidate_ffprobe = base_dir / "ffprobe.exe"
            if not ffmpeg_path and candidate_ffmpeg.exists():
                ffmpeg_path = str(candidate_ffmpeg)
            if not ffprobe_path and candidate_ffprobe.exists():
                ffprobe_path = str(candidate_ffprobe)
            if ffmpeg_path and ffprobe_path:
                return ffmpeg_path, ffprobe_path

        return ffmpeg_path, ffprobe_path

    def _find_binary(self, command: str, env_path: Optional[str] = None) -> Optional[str]:
        """优先使用环境变量指定路径，其次使用 PATH 查找。"""
        if env_path:
            candidate = Path(env_path)
            if candidate.is_dir():
                candidate = candidate / f"{command}.exe"
            if candidate.exists():
                return str(candidate)

        resolved = shutil.which(command)
        return resolved if resolved else None

    def _candidate_ffmpeg_dirs(self) -> list[Path]:
        """搜索 Windows 常见的 FFmpeg 安装位置。"""
        candidates: list[Path] = []
        for env_name in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)"):
            root = os.getenv(env_name)
            if not root:
                continue

            root_path = Path(root)
            if env_name == "LOCALAPPDATA":
                patterns = [
                    Path("Microsoft/WinGet/Packages/**/bin"),
                    Path("Microsoft/WinGet/Links"),
                ]
            else:
                patterns = [
                    Path("ffmpeg/**/bin"),
                    Path("FFmpeg/**/bin"),
                ]

            for pattern in patterns:
                candidates.extend(path for path in root_path.glob(str(pattern)) if path.is_dir())

        unique_candidates = []
        seen = set()
        for path in candidates:
            normalized = str(path).lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_candidates.append(path)
        return unique_candidates
    
    async def fetch_subtitles(self, url: str, output_dir: Path) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        先尝试从平台获取字幕文本，比下载音频快得多。

        Returns:
            (subtitle_markdown, video_title, language_code)
            subtitle_markdown 为 None 表示无可用字幕。
        """
        import asyncio

        output_dir.mkdir(exist_ok=True)
        unique_id = str(uuid.uuid4())[:8]
        sub_dir = output_dir / f"subs_{unique_id}"

        try:
            # 1. 快速探测：获取视频信息和字幕可用性，不下载任何内容
            check_opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
            with yt_dlp.YoutubeDL(check_opts) as ydl:
                info = await asyncio.to_thread(ydl.extract_info, url, False)

            video_title = info.get("title", "unknown")
            manual_subs: dict = info.get("subtitles") or {}
            auto_caps: dict = info.get("automatic_captions") or {}

            # 过滤掉 live_chat 等非语音轨道
            manual_langs = [k for k in manual_subs if not k.startswith("live_chat")]
            auto_langs = [k for k in auto_caps if not k.startswith("live_chat")]

            if not manual_langs and not auto_langs:
                logger.info(f"视频无可用字幕: {url}")
                return None, video_title, None

            # Prefer subtitles matching the video's source language when known.
            source_language = info.get("language")
            manual_source = self._match_subtitle_language(manual_langs, source_language)
            auto_source = self._match_subtitle_language(auto_langs, source_language)
            if manual_source:
                prefer_manual, prefer_lang = True, manual_source
            elif auto_source:
                prefer_manual, prefer_lang = False, auto_source
            else:
                prefer_manual = bool(manual_langs)
                candidate_langs = manual_langs if prefer_manual else auto_langs
                prefer_lang = next(
                    (lang for lang in candidate_langs if lang.lower().endswith("-orig")),
                    candidate_langs[0],
                )
            candidate_langs = manual_langs if prefer_manual else auto_langs
            logger.info(
                f"发现{'手动' if prefer_manual else '自动'}字幕，选用语言: {prefer_lang}"
                f"（候选 {len(candidate_langs)} 种）"
            )

            # 2. 仅下载字幕，跳过音视频
            sub_dir.mkdir(exist_ok=True)
            dl_opts = {
                "writesubtitles": prefer_manual,
                "writeautomaticsub": not prefer_manual,
                "subtitlesformat": "vtt/srt/best",
                "subtitleslangs": [prefer_lang],
                "skip_download": True,
                "outtmpl": str(sub_dir / "sub.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
            }
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [url])

            # 3. 查找下载的字幕文件
            sub_files = list(sub_dir.glob("*.vtt")) + list(sub_dir.glob("*.srt"))
            if not sub_files:
                logger.warning("字幕下载后未找到文件，回退音频模式")
                return None, video_title, None

            sub_file = sub_files[0]

            # 从文件名提取语言代码 (e.g. sub.en.vtt → en)
            stem_parts = sub_file.stem.split(".")
            file_lang = stem_parts[-1] if len(stem_parts) > 1 else prefer_lang

            # 4. 解析字幕文件
            if sub_file.suffix == ".vtt":
                entries = self._parse_vtt(str(sub_file))
            else:
                entries = self._parse_srt(str(sub_file))

            if not entries:
                logger.warning("字幕解析结果为空，回退音频模式")
                return None, video_title, None

            # 5. 格式化为与 Whisper 输出兼容的 Markdown
            formatted = self._format_subtitle_entries(entries, file_lang)
            logger.info(f"字幕获取成功: lang={file_lang}, {len(entries)} 条目")
            return formatted, video_title, file_lang

        except Exception as e:
            logger.warning(f"字幕获取失败（将回退至音频下载）: {e}")
            return None, None, None
        finally:
            if sub_dir.exists():
                try:
                    shutil.rmtree(str(sub_dir))
                except Exception:
                    pass

    @staticmethod
    def _match_subtitle_language(
        languages: list[str], source_language: Optional[str]
    ) -> Optional[str]:
        """Find a subtitle track matching the video's reported source language."""
        if not source_language:
            return None

        source = source_language.lower().replace("_", "-")
        exact_match = next(
            (lang for lang in languages if lang.lower().replace("_", "-") == source),
            None,
        )
        if exact_match:
            return exact_match

        source_base = source.split("-", 1)[0]
        return next(
            (
                lang
                for lang in languages
                if lang.lower().replace("_", "-").split("-", 1)[0] == source_base
            ),
            None,
        )

    # ------------------------------------------------------------------
    # 字幕解析辅助方法
    # ------------------------------------------------------------------

    def _parse_vtt(self, filepath: str) -> list:
        """解析 WebVTT，并过滤滚动字幕中被后续重叠条目替代的片段。

        特别处理 YouTube 自动字幕的「滚动追加」格式：
        同一句话会被分成多个重叠 cue 逐步追加，只保留最终版本。
        """
        raw_entries = []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"读取 VTT 文件失败: {e}")
            return []

        # 移除 WEBVTT 文件头，按空行分割 cue 块
        content = re.sub(r"^WEBVTT[^\n]*\n", "", content)
        blocks = re.split(r"\n{2,}", content.strip())

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = block.split("\n")
            timing_idx = next((i for i, l in enumerate(lines) if "-->" in l), -1)
            if timing_idx < 0:
                continue

            timing_line = lines[timing_idx]
            text_lines = lines[timing_idx + 1:]

            match = re.match(
                r"(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s*-->\s*"
                r"(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)",
                timing_line,
            )
            if not match:
                continue

            start_str = self._normalize_time(match.group(1))
            end_str = self._normalize_time(match.group(2))

            raw_text = " ".join(text_lines)
            # 去除 HTML / VTT 内联标签（包括 YouTube 逐字时间码标签）
            text = re.sub(r"<[^>]+>", "", raw_text)
            text = (
                text.replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&nbsp;", " ")
                    .replace("&#39;", "'")
                    .replace("&quot;", '"')
                    .strip()
            )
            # 合并行内多余空白
            text = re.sub(r"\s+", " ", text).strip()

            if not text:
                continue

            raw_entries.append({"start": start_str, "end": end_str, "text": text})

        # Filter overlapping YouTube rolling-caption updates without removing
        # the same words when they are spoken again later.
        if not raw_entries:
            return []

        entries = []
        for i, entry in enumerate(raw_entries):
            text = entry["text"]
            is_intermediate = False
            if i + 1 < len(raw_entries):
                next_entry = raw_entries[i + 1]
                start_seconds = self._subtitle_time_to_seconds(next_entry["start"])
                end_seconds = self._subtitle_time_to_seconds(entry["end"])
                next_text = next_entry["text"]
                is_intermediate = (
                    start_seconds < end_seconds
                    and next_text.startswith(text)
                    and len(next_text) > len(text)
                )
            if not is_intermediate:
                entries.append(entry)

        return entries

    def _parse_srt(self, filepath: str) -> list:
        """解析 SRT 字幕文件，保留时间上重复出现的内容。"""
        entries = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"读取 SRT 文件失败: {e}")
            return []

        blocks = re.split(r"\n{2,}", content.strip())

        for block in blocks:
            lines = block.strip().split("\n")
            timing_idx = next((i for i, l in enumerate(lines) if "-->" in l), -1)
            if timing_idx < 0:
                continue

            timing_line = lines[timing_idx]
            text_lines = lines[timing_idx + 1:]

            match = re.match(
                r"(\d{1,2}:\d{2}:\d{2}[.,]\d+)\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d+)",
                timing_line,
            )
            if not match:
                continue

            start_str = self._normalize_time(match.group(1))
            end_str = self._normalize_time(match.group(2))

            text = " ".join(text_lines)
            text = re.sub(r"<[^>]+>", "", text).strip()

            if not text:
                continue

            entries.append({"start": start_str, "end": end_str, "text": text})

        return entries

    def _subtitle_time_to_seconds(self, time_str: str) -> int:
        """Convert normalized MM:SS subtitle times to seconds."""
        minutes, seconds = (int(part) for part in time_str.split(":"))
        return minutes * 60 + seconds

    def _normalize_time(self, time_str: str) -> str:
        """将 HH:MM:SS.mmm 或 MM:SS.mmm 统一转为 MM:SS 格式。"""
        time_str = re.sub(r"[.,]\d+$", "", time_str)
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{h * 60 + m:02d}:{s:02d}"
        elif len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
            return f"{m:02d}:{s:02d}"
        return time_str

    def _format_subtitle_entries(self, entries: list, language: str) -> str:
        """将字幕条目格式化为与 Whisper 输出兼容的 Markdown，供下游管道直接使用。"""
        lines = [
            "# Video Transcription",
            "",
            f"**Detected Language:** {language}",
            "**Language Probability:** 1.00",
            "",
            "## Transcription Content",
            "",
        ]
        for entry in entries:
            lines.append(f"**[{entry['start']} - {entry['end']}]**")
            lines.append("")
            lines.append(entry["text"])
            lines.append("")
        return "\n".join(lines)

    async def download_and_convert(self, url: str, output_dir: Path) -> tuple[str, str]:
        """
        下载视频并转换为m4a格式
        
        Args:
            url: 视频链接
            output_dir: 输出目录
            
        Returns:
            转换后的音频文件路径
        """
        try:
            # 创建输出目录
            output_dir.mkdir(exist_ok=True)
            
            # 生成唯一的文件名
            unique_id = str(uuid.uuid4())[:8]
            output_template = str(output_dir / f"audio_{unique_id}.%(ext)s")
            
            # 更新yt-dlp选项
            ydl_opts = self.ydl_opts.copy()
            ydl_opts['outtmpl'] = output_template
            
            logger.info(f"开始下载视频: {url}")
            
            # 直接同步执行，不使用线程池
            # 在FastAPI中，IO密集型操作可以直接await
            import asyncio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # 获取视频信息（放到线程池避免阻塞事件循环）
                info = await asyncio.to_thread(ydl.extract_info, url, False)
                video_title = info.get('title', 'unknown')
                expected_duration = info.get('duration') or 0
                logger.info(f"视频标题: {video_title}")
                
                # 下载视频（放到线程池避免阻塞事件循环）
                await asyncio.to_thread(ydl.download, [url])
            
            # 查找生成的wav文件
            audio_file = str(output_dir / f"audio_{unique_id}.wav")
            
            if not os.path.exists(audio_file):
                # 如果wav文件不存在，查找其他音频格式
                for ext in ['m4a', 'webm', 'mp4', 'mp3']:
                    potential_file = str(output_dir / f"audio_{unique_id}.{ext}")
                    if os.path.exists(potential_file):
                        audio_file = potential_file
                        break
                else:
                    raise Exception("未找到下载的音频文件")
            
            # 校验时长，如果和源视频差异较大，尝试一次ffmpeg规范化重封装
            try:
                actual_duration = self._probe_duration(audio_file)
            except Exception:
                actual_duration = 0.0
            
            if expected_duration and actual_duration and abs(actual_duration - expected_duration) / expected_duration > 0.1:
                logger.warning(
                    f"音频时长异常，期望{expected_duration}s，实际{actual_duration}s，尝试重封装修复…"
                )
                try:
                    fixed_path = str(output_dir / f"audio_{unique_id}_fixed.wav")
                    self._remux_audio(audio_file, fixed_path)
                    # 用修复后的文件替换
                    audio_file = fixed_path
                    # 重新探测
                    actual_duration2 = self._probe_duration(audio_file)
                    logger.info(f"重封装完成，新时长≈{actual_duration2:.2f}s")
                except Exception as e:
                    logger.error(f"重封装失败：{e}")
            
            logger.info(f"音频文件已保存: {audio_file}")
            return audio_file, video_title
            
        except Exception as e:
            logger.error(f"下载视频失败: {str(e)}")
            raise Exception(f"下载视频失败: {str(e)}")
    
    async def extract_audio_from_file(self, file_path: str, output_dir: Path) -> str:
        """
        Extract audio from a local video/audio file and convert to WAV for Whisper.
        Optimized for speed on large files.

        Args:
            file_path: Path to the uploaded file
            output_dir: Output directory for the converted audio

        Returns:
            Path to the converted wav audio file
        """
        import asyncio

        if not self.ffmpeg_path:
            raise RuntimeError("ffmpeg is not available")

        output_dir.mkdir(exist_ok=True)
        unique_id = str(uuid.uuid4())[:8]
        output_file = str(output_dir / f"audio_{unique_id}.wav")

        cmd = [
            self.ffmpeg_path,
            '-y',
            '-hide_banner',
            '-loglevel', 'error',
            '-threads', str(os.cpu_count() or 4),
            '-i', file_path,
            '-vn',              # no video
            '-ac', '1',         # mono
            '-ar', '16000',     # 16kHz
            '-c:a', 'pcm_s16le',
            '-f', 'wav',
            output_file,
        ]

        logger.info(f"Extracting audio from uploaded file: {file_path}")
        await asyncio.to_thread(subprocess.check_call, cmd)
        logger.info(f"Audio extracted to: {output_file}")
        return output_file

    def get_video_info(self, url: str) -> dict:
        """
        获取视频信息
        
        Args:
            url: 视频链接
            
        Returns:
            视频信息字典
        """
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', ''),
                    'upload_date': info.get('upload_date', ''),
                    'description': info.get('description', ''),
                    'view_count': info.get('view_count', 0),
                }
        except Exception as e:
            logger.error(f"获取视频信息失败: {str(e)}")
            raise Exception(f"获取视频信息失败: {str(e)}")

    def _probe_duration(self, audio_file: str) -> float:
        """使用 ffprobe 获取音频时长。"""
        if not self.ffprobe_path:
            raise RuntimeError("ffprobe 不可用")

        result = subprocess.check_output(
            [
                self.ffprobe_path,
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_file,
            ],
            text=True,
        ).strip()
        return float(result) if result else 0.0

    def _remux_audio(self, audio_file: str, fixed_path: str) -> None:
        """使用 ffmpeg 重新封装音频文件。"""
        if not self.ffmpeg_path:
            raise RuntimeError("ffmpeg 不可用")

        subprocess.check_call(
            [
                self.ffmpeg_path,
                '-y',
                '-i', audio_file,
                '-vn',
                '-c:a', 'aac',
                '-b:a', '160k',
                '-movflags', '+faststart',
                fixed_path,
            ]
        )
