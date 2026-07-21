import os
import subprocess
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from faster_whisper import WhisperModel
import logging
from typing import Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Shared thread pool for parallel chunk transcription
_transcribe_pool = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) // 2))

def _detect_device():
    """Auto-detect best available device and compute type."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"

def _find_ffmpeg() -> Optional[str]:
    """Find ffmpeg binary."""
    import shutil
    return shutil.which("ffmpeg")

def _find_ffprobe() -> Optional[str]:
    """Find ffprobe binary."""
    import shutil
    return shutil.which("ffprobe")

class Transcriber:
    """音频转录器，使用Faster-Whisper进行语音转文字"""
    
    # Chunk duration in seconds for parallel processing
    CHUNK_SECONDS = 600  # 10 minutes per chunk to preserve more context
    # Minimum file duration (seconds) before splitting into chunks
    CHUNK_THRESHOLD = 900  # Only chunk if > 15 minutes
    
    def __init__(self, model_size: str = "small", preload: bool = True,
                 beam_size: int = 5, condition_on_previous_text: bool = True):
        """
        初始化转录器
        
        Args:
            model_size: Whisper模型大小 (tiny, base, small, medium, large)
            preload: 是否在初始化时预加载模型
            beam_size: Beam search width (1 = greedy/fast, 5 = thorough/slow)
            condition_on_previous_text: Use previous text as context (better accuracy, slower)
        """
        self.model_size = model_size
        self.model = None
        self.last_detected_language = None
        self.beam_size = beam_size
        self.condition_on_previous_text = condition_on_previous_text
        self._device, self._compute_type = _detect_device()
        self._ffmpeg = _find_ffmpeg()
        self._ffprobe = _find_ffprobe()
        if preload:
            self._load_model()
        
    def _load_model(self):
        """加载模型"""
        if self.model is None:
            logger.info(f"正在加载Whisper模型: {self.model_size} (device={self._device}, compute={self._compute_type})")
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                    num_workers=2,
                    cpu_threads=os.cpu_count() or 4,
                )
                logger.info("模型加载完成")
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                raise Exception(f"模型加载失败: {str(e)}")

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds using ffprobe."""
        if not self._ffprobe:
            return 0.0
        try:
            result = subprocess.check_output(
                [self._ffprobe, '-v', 'error',
                 '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1',
                 audio_path],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
            return float(result) if result else 0.0
        except Exception:
            return 0.0

    def _split_audio_chunks(self, audio_path: str, duration: float) -> List[Tuple[str, float]]:
        """Split audio into chunks using FFmpeg. Returns list of (chunk_path, start_offset)."""
        if not self._ffmpeg:
            return [(audio_path, 0.0)]

        num_chunks = max(1, math.ceil(duration / self.CHUNK_SECONDS))
        if num_chunks <= 1:
            return [(audio_path, 0.0)]

        chunks = []
        base_dir = Path(audio_path).parent
        base_name = Path(audio_path).stem

        for i in range(num_chunks):
            start = i * self.CHUNK_SECONDS
            chunk_path = str(base_dir / f"{base_name}_chunk{i}.wav")
            cmd = [
                self._ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(start),
                '-i', audio_path,
                '-t', str(self.CHUNK_SECONDS),
                '-ac', '1', '-ar', '16000',
                '-c:a', 'pcm_s16le', '-f', 'wav',
                chunk_path,
            ]
            try:
                subprocess.check_call(cmd, stderr=subprocess.DEVNULL)
                chunks.append((chunk_path, start))
            except Exception as e:
                logger.warning(f"Failed to create chunk {i}: {e}")

        logger.info(f"Split audio into {len(chunks)} chunks of {self.CHUNK_SECONDS}s each")
        return chunks if chunks else [(audio_path, 0.0)]

    def _detect_language(self, audio_path: str) -> str:
        """Detect the language of an audio file using Whisper. Returns language code."""
        _, info = self.model.transcribe(
            audio_path,
            language=None,
            beam_size=1,
            temperature=0.0,
            word_timestamps=False,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        detected = info.language if info and info.language else None
        prob = getattr(info, 'language_probability', None)
        logger.info(f"Auto-detected language: {detected} (probability: {prob})")
        return detected

    def _transcribe_chunk(self, chunk_path: str, time_offset: float, language: Optional[str] = None) -> List[dict]:
        """Transcribe a single chunk (runs in thread pool). Returns list of segment dicts."""
        segments, info = self.model.transcribe(
            chunk_path,
            language=language,
            beam_size=self.beam_size,
            temperature=0.0,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 200,
            },
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            condition_on_previous_text=self.condition_on_previous_text,
        )
        results = []
        for seg in segments:
            results.append({
                "start": seg.start + time_offset,
                "end": seg.end + time_offset,
                "text": seg.text.strip(),
            })
        return results

    async def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        转录音频文件 — 自动将长音频拆分为多个块并行转录
        
        Args:
            audio_path: 音频文件路径
            language: 指定语言（可选，如果不指定则默认英语）
            
        Returns:
            转录文本（Markdown格式）
        """
        try:
            if not os.path.exists(audio_path):
                raise Exception(f"音频文件不存在: {audio_path}")
            
            self._load_model()
            effective_language = language  # None means auto-detect
            
            # Determine duration and decide whether to chunk
            duration = self._get_audio_duration(audio_path)
            logger.info(f"开始转录音频: {audio_path} (duration={duration:.1f}s, language={effective_language or 'auto'})")

            import asyncio
            chunk_files = []
            detected_probability = None

            if duration > self.CHUNK_THRESHOLD and self._ffmpeg:
                # ── Parallel chunked transcription ──
                chunks = await asyncio.to_thread(
                    self._split_audio_chunks, audio_path, duration
                )

                all_segments: List[dict] = []

                # Detect language from first chunk if not specified
                if not effective_language and chunks:
                    logger.info("No language specified, auto-detecting from first chunk...")
                    effective_language = await asyncio.to_thread(self._detect_language, chunks[0][0])
                    logger.info(f"Will transcribe all chunks in: {effective_language}")

                if len(chunks) > 1:
                    # Transcribe chunks in parallel threads
                    loop = asyncio.get_event_loop()
                    futures = []
                    for chunk_path, offset in chunks:
                        chunk_files.append(chunk_path)
                        fut = loop.run_in_executor(
                            _transcribe_pool,
                            self._transcribe_chunk, chunk_path, offset, effective_language,
                        )
                        futures.append(fut)

                    results = await asyncio.gather(*futures)
                    for segs in results:
                        all_segments.extend(segs)

                    # Sort by start time (chunks processed in parallel may finish out of order)
                    all_segments.sort(key=lambda s: s["start"])
                else:
                    # Single chunk, just transcribe directly
                    segs = await asyncio.to_thread(
                        self._transcribe_chunk, chunks[0][0], 0.0, effective_language
                    )
                    all_segments = segs
            else:
                # ── Short audio: single-pass transcription ──
                # Pass language=None to let Whisper auto-detect AND transcribe in one shot
                def _do_transcribe():
                    segments, info = self.model.transcribe(
                        audio_path,
                        language=effective_language,  # None = auto-detect
                        beam_size=self.beam_size,
                        temperature=0.0,
                        word_timestamps=False,
                        vad_filter=True,
                        vad_parameters={
                            "min_silence_duration_ms": 300,
                            "speech_pad_ms": 200,
                        },
                        no_speech_threshold=0.6,
                        compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0,
                        condition_on_previous_text=self.condition_on_previous_text,
                    )
                    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments], info

                result, info = await asyncio.to_thread(_do_transcribe)
                all_segments = result
                # Capture the actual detected language from Whisper
                if info and info.language:
                    effective_language = info.language
                    detected_probability = getattr(info, 'language_probability', None)
                    logger.info(f"Whisper detected language: {effective_language} (prob={detected_probability})")

            if not effective_language:
                effective_language = "unknown"

            self.last_detected_language = effective_language
            
            # Build markdown output
            transcript_lines = [
                "# Video Transcription",
                "",
                f"**Detected Language:** {effective_language}",
                f"**Language Probability:** {detected_probability or 'N/A'}",
                "",
                "## Transcription Content",
                "",
            ]
            
            for seg in all_segments:
                start_time = self._format_time(seg["start"])
                end_time = self._format_time(seg["end"])
                text = seg["text"]
                if text:
                    transcript_lines.append(f"**[{start_time} - {end_time}]**")
                    transcript_lines.append("")
                    transcript_lines.append(text)
                    transcript_lines.append("")
            
            transcript_text = "\n".join(transcript_lines)
            logger.info(f"转录完成: {len(all_segments)} segments")

            # Cleanup chunk files
            for cf in chunk_files:
                try:
                    if cf != audio_path and os.path.exists(cf):
                        os.remove(cf)
                except Exception:
                    pass
            
            return transcript_text
            
        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            raise Exception(f"转录失败: {str(e)}")
    
    def _format_time(self, seconds: float) -> str:
        """
        将秒数转换为时分秒格式
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化的时间字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def get_supported_languages(self) -> list:
        """
        获取支持的语言列表
        """
        return [
            "zh", "en", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
            "ar", "hi", "th", "vi", "tr", "pl", "nl", "sv", "da", "no"
        ]
    
    def get_detected_language(self, transcript_text: Optional[str] = None) -> Optional[str]:
        """
        获取检测到的语言
        
        Args:
            transcript_text: 转录文本（可选，用于从文本中提取语言信息）
            
        Returns:
            检测到的语言代码
        """
        # 如果有保存的语言，直接返回
        if self.last_detected_language:
            return self.last_detected_language
        
        # 如果提供了转录文本，尝试从中提取语言信息
        if transcript_text and "**Detected Language:**" in transcript_text:
            lines = transcript_text.split('\n')
            for line in lines:
                if "**Detected Language:**" in line:
                    lang = line.split(":")[-1].replace("*", "").strip()
                    return lang
        
        return None

    def format_verbatim(self, raw_transcript: str, video_title: str = None) -> str:
        """
        Format a raw transcript (from Whisper or subtitles) into the same
        markdown structure used by the app, while refining only the spoken
        text for readability.
        """
        import re
        from metadata_generator import _refine_tagalog_english_text

        lines = raw_transcript.split('\n')
        output_lines = []

        title_match = re.search(r'^#\s+(.+)$', raw_transcript, re.MULTILINE)
        if title_match:
            output_lines.append(f"# {title_match.group(1).strip()}")
            output_lines.append("")
        elif video_title:
            output_lines.append(f"# {video_title}")
            output_lines.append("")

        # Preserve the original metadata block format.
        detected_lang = self.get_detected_language(raw_transcript)
        if detected_lang:
            output_lines.append(f"**Detected Language:** {detected_lang}")
        prob_match = re.search(r'\*\*Language Probability:\*\*\s*(.+)', raw_transcript)
        if prob_match:
            output_lines.append(f"**Language Probability:** {prob_match.group(1).strip()}")
        if detected_lang or prob_match:
            output_lines.append("")

        output_lines.append("## Transcription Content")
        output_lines.append("")

        # Parse timestamped segments from the Whisper markdown format
        # Pattern: **[MM:SS - MM:SS]** followed by text on the next non-empty line
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Match timestamp lines like **[00:00 - 00:05]** or **[00:00:00 - 00:00:05]**
            ts_match = re.match(
                r'\*\*\[(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)\]\*\*',
                line
            )
            if ts_match:
                start_ts = self._normalize_timestamp(ts_match.group(1))
                end_ts = self._normalize_timestamp(ts_match.group(2))
                # Collect text lines until next timestamp or end
                text_parts = []
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line:
                        i += 1
                        continue
                    if re.match(r'\*\*\[', next_line):
                        break
                    # Skip metadata lines
                    if next_line.startswith('#') or next_line.startswith('**Detected') or next_line.startswith('**Language'):
                        i += 1
                        continue
                    text_parts.append(next_line)
                    i += 1
                if text_parts:
                    text = ' '.join(text_parts)
                    text = _refine_tagalog_english_text(text)
                    output_lines.append(f"**[{start_ts} - {end_ts}]**")
                    output_lines.append("")
                    output_lines.append(text)
                    output_lines.append("")
                continue

            # Skip markdown headers and metadata from Whisper output
            if (line.startswith('#') or
                line.startswith('**Detected') or
                line.startswith('**Language') or
                line.startswith('## Transcription Content') or
                not line):
                i += 1
                continue

            # Plain text line without timestamp (e.g. from subtitle path)
            output_lines.append(_refine_tagalog_english_text(line))
            i += 1

        return '\n'.join(output_lines)

    def _normalize_timestamp(self, ts: str) -> str:
        """Convert MM:SS to HH:MM:SS format."""
        parts = ts.split(':')
        if len(parts) == 2:
            return f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
        elif len(parts) == 3:
            return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
        return ts

    def generate_srt(self, transcript_text: str) -> str:
        """
        Convert a Whisper/verbatim transcript (with timestamps) into a
        properly formatted SRT subtitle file.

        Supports two input formats:
          1. Whisper markdown: **[MM:SS - MM:SS]** followed by text
          2. Verbatim lines:   [HH:MM:SS] text
        """
        import re

        blocks: list[tuple[str, str, str]] = []  # (start, end, text)
        lines = transcript_text.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Format 1: **[00:12 - 00:18]** (Whisper markdown)
            ts_match = re.match(
                r'\*\*\[(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)\]\*\*',
                line,
            )
            if ts_match:
                start = self._normalize_timestamp(ts_match.group(1))
                end = self._normalize_timestamp(ts_match.group(2))
                text_parts = []
                i += 1
                while i < len(lines):
                    nl = lines[i].strip()
                    if not nl:
                        i += 1
                        continue
                    if re.match(r'\*\*\[', nl) or re.match(r'\[\d{2}:\d{2}:\d{2}\]', nl):
                        break
                    if nl.startswith('#') or nl.startswith('**Detected') or nl.startswith('**Language'):
                        i += 1
                        continue
                    text_parts.append(nl)
                    i += 1
                if text_parts:
                    blocks.append((start, end, ' '.join(text_parts)))
                continue

            # Format 2: [HH:MM:SS] text (verbatim)
            verb_match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*(.*)', line)
            if verb_match:
                start = verb_match.group(1)
                text = verb_match.group(2).strip()
                # End time = start of next block, or start + 5s as fallback
                i += 1
                if text:
                    blocks.append((start, None, text))  # end filled below
                continue

            i += 1

        # Fill in missing end times
        for idx in range(len(blocks)):
            if blocks[idx][1] is None:
                if idx + 1 < len(blocks):
                    blocks[idx] = (blocks[idx][0], blocks[idx + 1][0], blocks[idx][2])
                else:
                    # Last block: add 5 seconds
                    blocks[idx] = (blocks[idx][0], self._add_seconds_srt(blocks[idx][0], 5), blocks[idx][2])

        # Build SRT output
        srt_lines = []
        for idx, (start, end, text) in enumerate(blocks, 1):
            srt_start = self._to_srt_timestamp(start)
            srt_end = self._to_srt_timestamp(end)
            srt_lines.append(str(idx))
            srt_lines.append(f"{srt_start} --> {srt_end}")
            srt_lines.append(text)
            srt_lines.append('')

        return '\n'.join(srt_lines)

    def _to_srt_timestamp(self, hms: str) -> str:
        """Convert HH:MM:SS to SRT format HH:MM:SS,000."""
        parts = hms.split(':')
        if len(parts) == 3:
            return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)},000"
        if len(parts) == 2:
            return f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)},000"
        return f"00:00:{hms.zfill(2)},000"

    def _add_seconds_srt(self, hms: str, sec: int) -> str:
        """Add seconds to an HH:MM:SS timestamp and return HH:MM:SS."""
        parts = hms.split(':')
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        total = h * 3600 + m * 60 + s + sec
        nh = total // 3600
        nm = (total % 3600) // 60
        ns = total % 60
        return f"{nh:02d}:{nm:02d}:{ns:02d}"
