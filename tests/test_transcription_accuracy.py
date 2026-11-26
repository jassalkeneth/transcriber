import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.transcriber import Transcriber
from backend.video_processor import VideoProcessor


class SubtitleParsingTests(unittest.TestCase):
    def setUp(self):
        self.processor = object.__new__(VideoProcessor)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def _write_subtitle(self, name, content):
        path = Path(self.temp_dir.name) / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    def test_vtt_keeps_repeated_words_spoken_again_later(self):
        path = self._write_subtitle(
            "repeated.vtt",
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.000\nHello there.\n\n"
            "00:00:05.000 --> 00:00:06.000\nHello there.\n",
        )

        entries = self.processor._parse_vtt(path)

        self.assertEqual([entry["text"] for entry in entries], ["Hello there.", "Hello there."])

    def test_vtt_removes_overlapping_rolling_caption_intermediate(self):
        path = self._write_subtitle(
            "rolling.vtt",
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:03.000\nI think\n\n"
            "00:00:02.000 --> 00:00:04.000\nI think we should go.\n",
        )

        entries = self.processor._parse_vtt(path)

        self.assertEqual([entry["text"] for entry in entries], ["I think we should go."])

    def test_srt_keeps_repeated_words_spoken_again_later(self):
        path = self._write_subtitle(
            "repeated.srt",
            "1\n00:00:01,000 --> 00:00:02,000\nYes.\n\n"
            "2\n00:00:05,000 --> 00:00:06,000\nYes.\n",
        )

        entries = self.processor._parse_srt(path)

        self.assertEqual([entry["text"] for entry in entries], ["Yes.", "Yes."])

    def test_subtitle_language_selection_matches_source_metadata(self):
        language = self.processor._match_subtitle_language(
            ["en", "zh-Hans", "es-MX"], "zh-CN"
        )

        self.assertEqual(language, "zh-Hans")


class VerbatimTranscriptionTests(unittest.TestCase):
    def test_verbatim_format_does_not_replace_recognized_words(self):
        transcriber = Transcriber(preload=False)
        transcript = (
            "# Video Transcription\n\n"
            "**Detected Language:** tl\n\n"
            "## Transcription Content\n\n"
            "**[00:00 - 00:05]**\n\n"
            "mangyare\n"
        )

        formatted = transcriber.format_verbatim(transcript)

        self.assertIn("mangyare", formatted)
        self.assertNotIn("mangyari", formatted)


class AudioSourceQualityTests(unittest.TestCase):
    def test_download_prefers_best_available_audio(self):
        with patch.object(VideoProcessor, "_resolve_ffmpeg_binaries", return_value=(None, None)):
            processor = VideoProcessor()

        self.assertEqual(processor.ydl_opts["format"], "bestaudio/best")


if __name__ == "__main__":
    unittest.main()
