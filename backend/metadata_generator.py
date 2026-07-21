"""
metadata_generator.py
Generates structured YAML metadata for every transcript in the format
the AI brain needs to read and index the content automatically.
"""

import json
import logging
import re
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_str_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS or MM:SS string to float seconds."""
    parts = ts.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return 0.0


def _normalize_ts(ts: str) -> str:
    """Ensure timestamp is HH:MM:SS."""
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    if len(parts) == 3:
        return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
    return "00:00:00"


def _slug(text: str, max_len: int = 60) -> str:
    """Convert a title to a slug suitable for document_id."""
    s = re.sub(r"[^\w\s-]", "", (text or "document").lower())
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return s[:max_len] or "document"


def _refine_tagalog_english_text(text: str) -> str:
    """
    Clean up and improve grammar for mixed Tagalog-English transcriptions.
    Makes text accurate, concise, and precise for AI recognition.
    """
    if not text or not text.strip():
        return text

    refined = text

    # ── Step 1: Fix specific Whisper transcription errors ─────────────
    # Format: (regex_pattern, replacement) — order matters!
    phrase_corrections = [
        # Common multi-word transcription errors
        (r'\bKrabis obrang lopit mo\b', 'Grabe, sobrang lupit mo'),
        (r'\balam mo kung bakit\b', 'alam mo kung bakit'),
        (r'\bdalawang classy ng tao\b', 'dalawang klase ng tao'),
        (r'\btao nagusto\b', 'taong gusto'),
        (r'\btao yung nag-take action\b', 'taong nag-take action'),
        (r'\byung mamahan\b', 'yung mangyari'),
        (r'\bmagbaga buhay ko\b', 'magbago ang buhay ko'),
        (r'\bparti ka ng\b', 'parte ka ng'),
        (r'\bsmall percentage dito sa mundo\b', 'maliit na porsiyento sa mundong ito'),
        (r'\bmodule overview mo na tayo\b', 'mag-module overview muna tayo'),
        (r'\beto pinaka-favorit natin lahat\b', 'ito ang paborito nating lahat'),
        (r'\bmay makikita ka ng mga worksheets\b', 'may makikita kang mga worksheet'),
        (r'\bmeron tayong pag-quiz dyan\b', 'may quiz tayo diyan'),
        (r'\bsagotan yung mga quizzes\b', 'sagutan ang mga quiz'),
        (r'\bsa babaan ng modules\b', 'sa baba ng modules'),
        (r'\bEnong kapala\b', 'Eto pa pala'),
        (r'\bremind kulit sa yun\b', 'paalala ulit sa iyo'),
        (r'\bpangasokoy sa mga groups\b', 'o pumapasok kayo sa mga groups'),
        (r'\bmga scammers pangat\b', 'mga scammer pa nga'),
        (r'\bmag-PPM talaga sa inyo\b', 'talagang magpi-PM sa inyo'),
        (r'\bpag-usaka\b', 'pag-uusap'),
        (r'\bGanaba ka-importante\b', 'Gaano ka-importante'),
        (r'\bbigang kung\b', 'bibigyan kita ng'),
        (r'\batas na building\b', 'mataas na building'),
        (r'\bSa yan\b', 'Sa loob niyan'),
        (r'\bbintanan\b', 'bintana'),
        (r'\bHindi pa lo\b', 'Hindi pa lahat ng'),
        (r'\bAyan\.\s*Tapos\b', 'Tapos'),
        (r'\bkatastrofis\b', 'katastrope'),
        (r'\bnabah\b', 'di ba'),
        (r'\bKonting hipan\b', 'Kahit konting hipan'),
        (r'\bna-build\b', 'naitayo'),
        (r'\bpinaka-pina ka ayon\b', 'pinaka ayaw'),
        (r'\bpinaka-pina kawag\b', 'pinaka ayaw'),
        (r'\bayon natin mangyare\b', 'ayaw nating mangyari'),
        (r'\bmangyare\b', 'mangyari'),
        (r'\bnabah\b', 'di ba'),
        (r'\bnag-madalika\b', 'nagmamadali'),
        (r'\bgumita\b', 'kumita'),
        (r'\bcompound mo toh\b', 'i-compound mo tapos maging'),
        (r'\bginul mo\b', 'goal mo'),
        (r'\bkakatapas\b', 'katatapos'),
        (r'\bnag-madalik ang ginawam mo\b', 'nagmadali ang ginawa mo'),
        (r'\btumamba\b', 'tumama'),
        (r'\bdoblimo\b', 'doble mo'),
        (r'\btalagin\b', 'talagang'),
        (r'\bnabay mga\b', 'ang mga'),
        (r'\bnandyan\b', 'nandiyan'),
        (r'\bsyempre\b', 'siyempre'),
        (r'\bNababasan yung mabasa ko\b', ''),
        (r'\bsinulot\b', 'sinabi'),
        (r'\bpalalim\b', 'palaliman'),
        (r'\bpala-limin\b', 'palalimin'),
        (r'\bmamamay\b', 'mamaya'),
        (r'\batuturo\b', 'ituturo'),
        (r'\bmagintayin\b', 'maghihintay'),
        (r'\bbinibild\b', 'binubuo'),
        (r'\bmagkakaron\b', 'magkakaroon'),
        (r'\banta nung\b', 'ngayon'),
        (r'\buhugot\b', 'huhugot'),
        (r'\bmababao\b', 'mababaw'),
        (r'\bnandiyong\b', 'nandoon ang'),
        (r'\bbumayan\b', 'bumili'),
        (r'\bpinakang\b', 'pinaka'),
        (r'\bnagdadrive\b', 'nagda-drive'),
        (r'\bnilagawin\b', 'gawin'),
        (r'\bpag-ugutan\b', 'paghuhugutan'),
        (r'\bayong muna\b', 'ayaw mo nang'),
        (r'\bbienan\b', 'biyenan'),
        (r'\bipun lang\b', 'ipon lang'),
        (r'\bkailang\b', 'kailangan'),
        (r'\bAsi\b', 'Kasi'),
        (r'\bnantama\b', 'naman'),
        (r'\bAyakaw\b', 'Ayoko'),
        (r'\bbumiling\b', 'bumili ng'),
        (r'\bkainin na siyang\b', 'kaya na siyang'),
        (r'\bAyong ko\b', 'Ayoko'),
        (r'\bganin\b', 'ganyan'),
        (r'\bgamiti\b', 'gamitin'),
        (r'\bmagmo forward\b', 'mag-move forward'),
        (r'\bmagwit na kaya\b', 'mag-quit na kaya'),
        (r'\bmakuwan\b', 'makuha'),
        (r'\bnakuwam\b', 'nakuha'),
        (r'\bismeron\b', 'meron'),
        (r'\bnaglakat\b', 'naglakad'),
        (r'\bmasyempre\b', 'masyado'),
        (r'\bkilangan\b', 'kailangan'),
        (r'\byamanin\b', 'mayaman'),
        (r'\bmalibre\b', 'maialay'),
        (r'\bgit nanon\b', 'gitna noon'),
        (r'\bnakailang balik\b', 'nakailang balik'),
        (r'\bhugot talagay\b', 'hugot talaga'),
        (r'\bemotions talagang\b', 'emotions, talagang'),
        (r'\bTaplion\b', 'Tatlo'),
        (r'\bnagkompangta\b', 'nag-compute'),
        (r'\bnagkompang\b', 'nag-compute'),
        (r'\bmigita\b', 'kumita'),
        (r'\bnakomuni\b', 'nakuha mo na'),
        (r'\bkapag-shopping\b', 'kapag mag-shopping'),
        (r'\bitura nyan\b', 'kalidad niyan'),
        (r'\bIto talaga yung\b', 'Ito ang'),
        (r'\bito talagin\b', 'ito talagang'),
        (r'\bmga yari sa inyo\b', 'mangyayari sa inyo'),
        (r'\bnabagsabi\b', 'nabanggit'),
        (r'\blipenda\b', 'depende'),
        (r'\btututo\b', 'matuto'),
        (r'\bayong\b', 'ayaw'),
        (r'\bAyong\b', 'Ayaw'),
        (r'\bnasa babaya\b', 'nasa baba'),
        (r'\bnasa worksheet natin ko\b', 'nasa worksheet natin'),
        (r'\bbaba\.?$', 'baba.'),
        (r'\bnilolok\b', 'iniisip'),
        (r'\bnamag\b', 'na mag-'),
        (r'\bnando palad\b', 'naroon palad'),
        (r'\bpa ron\b', 'na sa'),
        (r'\bnag-boo forward\b', 'nag-move forward'),
        (r'\bpinakaway\b', 'pinakaraw'),
        (r'\bganun\b', 'ganoon'),
        (r'\bgano\'n\b', 'ganoon'),
        (r'\blabu-labu\b', 'malabo'),
        (r'\bnapipaint\b', 'napipinturahan'),
        (r'\bnag-resonit\b', 'nag-resonate'),
        (r'\bmaukuwam\b', 'makukuha'),
        (r'\bnakasports\b', 'may sports car'),
        (r'\bnasa babaya\b', 'nasa baba ng'),
        (r'\bAlat na naging\b', 'Lahat ng naging'),
        (r'\bnatatalo\b', 'natatalo'),
        (r'\bnitong\b', 'nito'),
        (r'\bsa toh\b', 'na ito'),
        (r'\bmalay\b', 'malaki'),
    ]

    for pattern, replacement in phrase_corrections:
        refined = re.sub(pattern, replacement, refined, flags=re.IGNORECASE)

    # ── Step 2: Single-word corrections ───────────────────────────────
    word_corrections = {
        r'\bkong\b': 'kung',
        r'\bnang\b': 'ng',
        r'\bnung\b': 'noong',
        r'\bdi ba\b': 'di ba',
        r'\bdiba\b': 'di ba',
        r'\bsyempre\b': 'siyempre',
        r'\bpwede\b': 'puwede',
        r'\bmeron\b': 'mayroon',
        r'\bmerong\b': 'mayroong',
        r'\bfavorit\b': 'paborito',
        r'\bkapala\b': 'pa pala',
        r'\brecruiters\b': 'recruiters',
    }

    for pattern, replacement in word_corrections.items():
        refined = re.sub(pattern, replacement, refined, flags=re.IGNORECASE)

    # ── Step 3: Remove triple+ word repetitions ───────────────────────
    refined = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', refined, flags=re.IGNORECASE)
    # Also remove double duplicates like "darating at darating at darating"
    refined = re.sub(r'\b(\w+ \w+)( \1){1,}\b', r'\1', refined, flags=re.IGNORECASE)

    # ── Step 4: Remove excessive filler interjections ─────────────────
    # Multiple "Okay?" or "Okay." in a row
    refined = re.sub(r'(\bOkay[.?,!])\s*\1+', r'\1', refined, flags=re.IGNORECASE)
    # Remove single "Pa?" or "Pa." filler
    refined = re.sub(r'\bPa[?.]\s+', '', refined)
    # Remove "Yo-yo!" intros
    refined = re.sub(r'^Yo-yo!\s*', '', refined, flags=re.IGNORECASE)

    # ── Step 5: Fix punctuation and spacing ───────────────────────────
    refined = re.sub(r'\s+', ' ', refined)
    refined = re.sub(r'\s+([.,!?;:])', r'\1', refined)
    refined = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', refined)
    refined = re.sub(r'\.\.+', '.', refined)
    refined = re.sub(r'\?\.', '?', refined)
    refined = re.sub(r'!\.', '!', refined)
    refined = re.sub(r',\s*,', ',', refined)

    # ── Step 6: Fix common grammar patterns ───────────────────────────
    refined = re.sub(r'\bmga yung\b', 'ang mga', refined, flags=re.IGNORECASE)
    refined = re.sub(r'\byung mga\b', 'ang mga', refined, flags=re.IGNORECASE)
    refined = re.sub(r'\bna mga\b', 'ng mga', refined, flags=re.IGNORECASE)

    # ── Step 7: Remove trailing ellipsis and fragments ────────────────
    refined = re.sub(r'\s*\.\.\.+\s*', ' ', refined)
    refined = re.sub(r'\s+', ' ', refined)

    # ── Step 8: Capitalize sentences ──────────────────────────────────
    refined = re.sub(
        r'(^|[.!?]\s+)([a-z])',
        lambda m: m.group(1) + m.group(2).upper(),
        refined,
    )

    # ── Step 9: Final cleanup ─────────────────────────────────────────
    refined = ' '.join(refined.split()).strip()
    if refined and refined[-1] not in '.!?':
        refined += '.'

    return refined


def _condense_text(text: str, max_sentences: int = 3) -> str:
    """
    Create a concise cleaned version of text by keeping only the most
    meaningful sentences (skips filler-heavy fragments).
    """
    if not text:
        return text

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out very short or filler-heavy sentences
    filler_pattern = re.compile(
        r'^(okay|so|yan|ayan|alright|right|well|hmm|ah|eh|oo|hindi|pa|ba|po|kasi)[.,!?\s]*$',
        re.IGNORECASE,
    )
    meaningful = [s.strip() for s in sentences if len(s.strip()) > 30 and not filler_pattern.match(s.strip())]
    if not meaningful:
        meaningful = [s.strip() for s in sentences if s.strip()]

    return ' '.join(meaningful[:max_sentences])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MetadataGenerator:
    """
    Parses a Whisper transcript into structured blocks and enriches it with
    AI-generated metadata (topics, key_takeaways, definitions, FAQs, etc.)
    to produce the canonical YAML document the AI can index.
    """

    # Group raw Whisper segments into blocks of at most this many seconds
    BLOCK_SECONDS = 90

    def __init__(self, client=None, fast_model: str = "gpt-4o-mini"):
        self.client = client
        self.fast_model = fast_model

    # ------------------------------------------------------------------
    # Step 1 – Parse raw Whisper markdown into segment list
    # ------------------------------------------------------------------

    def _parse_whisper_segments(self, raw_transcript: str) -> list:
        """
        Parse the Whisper markdown output:
            **[MM:SS - MM:SS]**
            text…
        Returns a list of dicts: {start, end, text}
        """
        segments = []
        lines = raw_transcript.split("\n")
        i = 0
        ts_re = re.compile(
            r"\*\*\[(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)\]\*\*"
        )
        while i < len(lines):
            m = ts_re.match(lines[i].strip())
            if m:
                start_ts = m.group(1)
                end_ts = m.group(2)
                text_parts = []
                i += 1
                while i < len(lines):
                    stripped = lines[i].strip()
                    # Stop at the next timestamp line
                    if ts_re.match(stripped):
                        break
                    if (
                        stripped
                        and not stripped.startswith("#")
                        and not stripped.startswith("**Detected")
                        and not stripped.startswith("**Language")
                        and not stripped.startswith("**检测语言")
                        and not stripped.startswith("**语言")
                    ):
                        text_parts.append(stripped)
                    i += 1
                text = " ".join(text_parts).strip()
                if text:
                    segments.append(
                        {
                            "start": start_ts,
                            "end": end_ts,
                            "text": text,
                        }
                    )
            else:
                i += 1
        return segments

    # ------------------------------------------------------------------
    # Step 2 – Group segments into ~BLOCK_SECONDS blocks
    # ------------------------------------------------------------------

    def _group_into_blocks(self, segments: list) -> list:
        """Group raw segments into logical blocks of ~BLOCK_SECONDS."""
        if not segments:
            return []

        blocks = []
        current_segs = []
        block_start_sec = _time_str_to_seconds(segments[0]["start"])

        for seg in segments:
            seg_end_sec = _time_str_to_seconds(seg["end"])
            current_segs.append(seg)
            if seg_end_sec - block_start_sec >= self.BLOCK_SECONDS:
                blocks.append(
                    {
                        "start": current_segs[0]["start"],
                        "end": current_segs[-1]["end"],
                        "text": " ".join(s["text"] for s in current_segs),
                    }
                )
                current_segs = []
                if len(segments) > segments.index(seg) + 1:
                    block_start_sec = _time_str_to_seconds(
                        segments[segments.index(seg) + 1]["start"]
                    )

        if current_segs:
            blocks.append(
                {
                    "start": current_segs[0]["start"],
                    "end": current_segs[-1]["end"],
                    "text": " ".join(s["text"] for s in current_segs),
                }
            )
        return blocks

    # ------------------------------------------------------------------
    # Step 3 – AI metadata generation (single call)
    # ------------------------------------------------------------------

    async def _generate_ai_fields(
        self,
        video_title: str,
        detected_language: str,
        transcript_excerpt: str,
        blocks: list,
    ) -> dict:
        """
        Single OpenAI call to enrich all metadata fields at once.
        Returns a dict with keys matching the YAML schema.
        """
        if not self.client:
            return self._fallback_fields(blocks)

        # Build a compact blocks summary to keep the prompt small
        blocks_summary = "\n".join(
            f"Block {i+1} [{b['start']} - {b['end']}]: {b['text'][:300]}"
            for i, b in enumerate(blocks[:25])  # cap at 25 blocks
        )

        system_prompt = (
            "You are a content metadata analyst. "
            "Analyze the provided transcript and return ONLY a single valid JSON object — no markdown, no explanation."
        )

        user_prompt = f"""Video title: {video_title or 'Unknown'}
Language: {detected_language or 'en'}

--- Transcript excerpt (first 4000 chars) ---
{transcript_excerpt[:4000]}

--- Blocks summary ---
{blocks_summary}

Generate a JSON object with EXACTLY these keys:
{{
  "instructor": "Name if clearly stated in transcript, otherwise null",
  "course_name": "Course name if mentioned, otherwise empty string",
  "module_name": "Module/section name if mentioned, otherwise empty string",
  "lesson_type": "educational_video",
  "topics": ["topic1", "topic2"],
  "pillars": ["pillar1"],
  "summary": "2-3 sentence plain-text summary of the whole lesson",
  "key_takeaways": ["takeaway1", "takeaway2"],
  "definitions": [
    {{"term": "Term", "meaning": "Definition"}}
  ],
  "visual_notes": [
    {{"timestamp": "MM:SS", "type": "slide|formula|worked_example|chart_annotation", "content": "brief description"}}
  ],
  "faq_candidates": [
    {{"question": "Q?", "answer": "A."}}
  ],
  "searchable_keywords": ["kw1", "kw2"],
  "block_metadata": [
    {{"block_index": 0, "cleaned_text": "concise cleaned text for block 0", "retrieval_tags": ["tag1"]}}
  ]
}}

Rules:
- topics: 3-8 items
- pillars: 1-3 high-level categories
- key_takeaways: 3-8 items
- definitions: 0-8 items (only include if the transcript defines terms)
- visual_notes: 0-6 items (infer from transcript content like formulas, examples, charts)
- faq_candidates: 3-6 items
- searchable_keywords: 5-12 items
- block_metadata: one entry per block in the blocks summary above (same count, same order)
- retrieval_tags per block: 2-4 short tags
- Return ONLY the JSON object, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model=self.fast_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=4000,
                temperature=0.1,
            )
            raw = response.choices[0].message.content or "{}"
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw.strip())
            return json.loads(raw)
        except Exception as e:
            logger.error(f"AI metadata generation failed: {e}")
            return self._fallback_fields(blocks)

    # ------------------------------------------------------------------
    # Stopwords used by the local fallback (English + Tagalog fillers)
    # ------------------------------------------------------------------
    _STOPWORDS = {
        # ── English ──
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "on",
        "at", "by", "for", "with", "about", "and", "or", "but", "if", "then",
        "that", "this", "it", "its", "i", "you", "we", "they", "he", "she",
        "your", "my", "our", "so", "just", "like", "also", "very", "what",
        "how", "when", "where", "which", "who", "all", "not", "no", "as",
        "from", "up", "out", "into", "than", "more", "some", "any", "there",
        "their", "them", "these", "those", "because", "here", "make", "made",
        "know", "going", "gonna", "okay", "yeah", "right", "now", "come",
        "actually", "really", "let", "get", "got", "want", "need", "one",
        "two", "three", "four", "five", "much", "many", "even", "well",
        "still", "back", "again", "thing", "things", "stuff", "good", "bad",
        "way", "ways", "time", "times", "day", "days", "year", "years",
        "first", "next", "last", "every", "another", "other", "same",
        "only", "own", "say", "said", "see", "say", "think", "thought",
        # ── Tagalog/Filipino fillers and particles ──
        "yung", "ang", "ng", "sa", "na", "ay", "at", "o", "po", "opo",
        "kasi", "para", "mga", "lang", "ito", "iyan", "iyon", "nito",
        "dito", "diyan", "doon", "natin", "tayo", "kami", "kayo", "sila",
        "niya", "nila", "ko", "mo", "ka", "ako", "ikaw", "siya", "tayo",
        "namin", "ninyo", "kanila", "kanya", "akin", "iyo", "atin",
        "nung", "noong", "kung", "pag", "kapag", "habang",
        "talaga", "naman", "din", "rin", "pala", "nga", "lamang", "muna",
        "pa", "ba", "daw", "raw", "yata", "siguro", "baka", "marahil",
        "gusto", "ayaw", "kailangan", "puwede", "pwede", "may", "meron",
        "merong", "mayroon", "mayroong", "wala", "walang", "hindi", "oo", "yes", "tama", "mali",
        "pero", "ngunit", "subalit", "datapwa", "kaya", "dahil", "upang",
        "tapos", "saka", "tsaka", "diba", "ano", "anong", "bakit", "paano",
        "sino", "kanino", "nasaan", "saan", "kailan", "alin",
        "isang", "isa", "dalawa", "tatlo", "apat", "lima",
        "ganito", "ganyan", "ganoon", "gano'n", "ganun",
        "sobrang", "masyadong", "talagang", "tunay", "totoo",
        "kong", "nang", "yan", "yon", "ayan", "ayon", "ayun",
        "okay", "alright", "well", "ok",
    }

    def _fallback_fields(self, blocks: list) -> dict:
        """
        Extract useful metadata directly from transcript text when AI is unavailable.
        Uses frequency analysis and sentence heuristics instead of an API call.
        """
        full_text = " ".join(b["text"] for b in blocks)

        # ── Sentence splitting ────────────────────────────────────────────
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", full_text)
            if len(s.strip()) > 25
        ]

        # ── Summary: first 3 meaningful sentences ────────────────────────
        summary = " ".join(sentences[:3]) if sentences else full_text[:400]

        # ── Key takeaways: sentences that sound instructional ─────────────
        important_re = re.compile(
            r"\b(important|key|remember|note|make sure|always|never|must|"
            r"should|rule|first|second|third|step|number|how to|learn|tip)\b",
            re.IGNORECASE,
        )
        takeaways = [s for s in sentences if important_re.search(s)][:5]
        if not takeaways:
            takeaways = sentences[:4]

        # ── Keyword frequency ─────────────────────────────────────────────
        words = re.findall(r"\b[a-zA-Z]{4,}\b", full_text.lower())
        freq: dict = {}
        for w in words:
            if w not in self._STOPWORDS:
                freq[w] = freq.get(w, 0) + 1
        top_keywords = sorted(freq, key=freq.get, reverse=True)  # type: ignore[arg-type]

        # ── Per-block metadata ────────────────────────────────────────────
        block_metadata = []
        for i, b in enumerate(blocks):
            bwords = re.findall(r"\b[a-zA-Z]{4,}\b", b["text"].lower())
            bfreq: dict = {}
            for w in bwords:
                if w not in self._STOPWORDS:
                    bfreq[w] = bfreq.get(w, 0) + 1
            tags = sorted(bfreq, key=bfreq.get, reverse=True)[:4]  # type: ignore[arg-type]
            # Generate a real condensed/cleaned version (not just truncated)
            cleaned = _condense_text(b["text"], max_sentences=3)
            block_metadata.append(
                {
                    "block_index": i,
                    "cleaned_text": cleaned,
                    "retrieval_tags": tags,
                }
            )

        return {
            "instructor": None,
            "course_name": "",
            "module_name": "",
            "lesson_type": "educational_video",
            "topics": top_keywords[:6],
            "pillars": [],
            "summary": summary,
            "key_takeaways": takeaways,
            "definitions": [],
            "visual_notes": [],
            "faq_candidates": [],
            "searchable_keywords": top_keywords[:12],
            "block_metadata": block_metadata,
        }

    # ------------------------------------------------------------------
    # Step 4 – Assemble the full YAML document
    # ------------------------------------------------------------------

    def _build_document(
        self,
        video_title: str,
        source_url: str,
        detected_language: str,
        duration_seconds: float,
        blocks: list,
        ai: dict,
        task_id: str,
    ) -> dict:
        doc_id = _slug(video_title or task_id)

        # Build transcript_blocks list
        block_metas = ai.get("block_metadata", [])
        transcript_blocks = []
        for i, block in enumerate(blocks):
            meta = block_metas[i] if i < len(block_metas) else {}
            
            # Refine the text for better AI recognition (Tagalog-English)
            refined_text = _refine_tagalog_english_text(block["text"])
            refined_cleaned = _refine_tagalog_english_text(
                meta.get("cleaned_text", block["text"][:300])
            )
            
            transcript_blocks.append(
                {
                    "block_id": f"{doc_id}-b{str(i + 1).zfill(2)}",
                    "start": _normalize_ts(block["start"]),
                    "end": _normalize_ts(block["end"]),
                    "speaker": ai.get("instructor") or "Instructor",
                    "text": refined_text,
                    "cleaned_text": refined_cleaned,
                    "retrieval_tags": meta.get("retrieval_tags", []),
                }
            )

        first_start = _normalize_ts(blocks[0]["start"]) if blocks else "00:00:00"
        last_end = _normalize_ts(blocks[-1]["end"]) if blocks else "00:00:00"

        # Build visual_notes list, normalise timestamps
        visual_notes = []
        for vn in ai.get("visual_notes", []):
            visual_notes.append(
                {
                    "timestamp": vn.get("timestamp", "00:00:00"),
                    "type": vn.get("type", "slide"),
                    "content": vn.get("content", ""),
                }
            )

        document = {
            "document_id": doc_id,
            "course_name": ai.get("course_name", ""),
            "module_name": ai.get("module_name", ""),
            "lesson_name": video_title or "",
            "lesson_type": ai.get("lesson_type", "educational_video"),
            "source_type": "video_module" if source_url else "audio_upload",
            "instructor": ai.get("instructor") or "Unknown",
            "language": detected_language or "en",
            "topics": ai.get("topics", []),
            "pillars": ai.get("pillars", []),
            "duration_seconds": int(duration_seconds) if duration_seconds else 0,
            "source_video_url": source_url or "",
            "timestamps_covered": {
                "start": first_start,
                "end": last_end,
            },
            "summary": _refine_tagalog_english_text(ai.get("summary", "")),
            "key_takeaways": [
                _refine_tagalog_english_text(item) for item in ai.get("key_takeaways", [])
            ],
            "definitions": [
                {"term": d.get("term", ""), "meaning": d.get("meaning", "")}
                for d in ai.get("definitions", [])
            ],
            "visual_notes": visual_notes,
            "transcript_blocks": transcript_blocks,
            "faq_candidates": [
                {"question": f.get("question", ""), "answer": f.get("answer", "")}
                for f in ai.get("faq_candidates", [])
            ],
            "searchable_keywords": ai.get("searchable_keywords", []),
        }
        return document

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def generate(
        self,
        raw_transcript: str,
        video_title: str,
        source_url: str,
        detected_language: str,
        duration_seconds: float,
        task_id: str,
    ) -> dict:
        """
        Full pipeline: parse → block → AI enrich → assemble.
        Returns a dict that can be serialised to YAML.
        """
        # 1. Parse raw transcript into segments
        segments = self._parse_whisper_segments(raw_transcript)
        logger.info(f"Parsed {len(segments)} Whisper segments")

        # 2. Group into blocks
        blocks = self._group_into_blocks(segments)
        logger.info(f"Grouped into {len(blocks)} blocks")

        if not blocks:
            # Fallback: treat the entire optimised transcript as one block
            cleaned = re.sub(r"\*\*\[.*?\]\*\*", "", raw_transcript).strip()
            blocks = [{"start": "00:00:00", "end": "00:00:00", "text": cleaned}]

        # 2b. Pre-refine block text so all downstream analysis works on
        #     grammatically corrected text (better topics/keywords/cleaned)
        for b in blocks:
            b["text"] = _refine_tagalog_english_text(b["text"])

        # 3. Generate AI metadata
        # Use the plain (cleaned) transcript for the AI prompt
        plain_text = re.sub(r"\*\*\[.*?\]\*\*", "", raw_transcript)
        plain_text = re.sub(r"^#{1,6}\s.*$", "", plain_text, flags=re.MULTILINE)
        plain_text = re.sub(r"\n{3,}", "\n\n", plain_text).strip()

        ai_fields = await self._generate_ai_fields(
            video_title=video_title,
            detected_language=detected_language,
            transcript_excerpt=plain_text,
            blocks=blocks,
        )

        # 4. Assemble
        document = self._build_document(
            video_title=video_title,
            source_url=source_url,
            detected_language=detected_language,
            duration_seconds=duration_seconds,
            blocks=blocks,
            ai=ai_fields,
            task_id=task_id,
        )

        return document

    # ------------------------------------------------------------------
    # Serialise to YAML string
    # ------------------------------------------------------------------

    @staticmethod
    def to_yaml(document: dict) -> str:
        """
        Serialise to YAML matching the canonical message.txt format exactly:
          - timestamps always double-quoted  ("00:00:00")
          - prose fields (summary, text, cleaned_text) use folded scalar (>)
          - strings containing ': ' are auto-quoted
          - blank lines between every major section and within each block
        """
        import copy
        import re

        class _Quoted(str):
            """Always rendered with double-quotes."""

        class _Folded(str):
            """Always rendered as folded block scalar (>)."""

        class _Dumper(yaml.Dumper):
            def increase_indent(self, flow=False, indentless=False):
                return super().increase_indent(flow, False)

        def _quoted_rep(dumper, data):
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')

        def _folded_rep(dumper, data):
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=">")

        def _str_rep(dumper, data):
            if ": " in data or data.startswith(":"):
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        _Dumper.add_representer(_Quoted, _quoted_rep)
        _Dumper.add_representer(_Folded, _folded_rep)
        _Dumper.add_representer(str, _str_rep)

        doc = copy.deepcopy(document)

        # timestamps_covered → quoted
        tc = doc.get("timestamps_covered", {})
        if "start" in tc:
            tc["start"] = _Quoted(tc["start"])
        if "end" in tc:
            tc["end"] = _Quoted(tc["end"])

        # summary → folded block scalar (> style)
        if doc.get("summary"):
            doc["summary"] = _Folded(doc["summary"].strip() + "\n")

        # visual_notes timestamps → quoted
        for vn in doc.get("visual_notes", []):
            if "timestamp" in vn:
                vn["timestamp"] = _Quoted(vn["timestamp"])

        # transcript_blocks: quoted timestamps, folded text fields
        for block in doc.get("transcript_blocks", []):
            if "start" in block:
                block["start"] = _Quoted(block["start"])
            if "end" in block:
                block["end"] = _Quoted(block["end"])
            if block.get("text"):
                block["text"] = _Folded(block["text"].strip() + "\n")
            if block.get("cleaned_text"):
                block["cleaned_text"] = _Folded(block["cleaned_text"].strip() + "\n")

        raw = yaml.dump(
            doc,
            Dumper=_Dumper,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )

        # ── Post-process: insert blank lines to match message.txt exactly ──

        # 1. Blank line before each major top-level section
        for key in [
            "summary:", "key_takeaways:", "definitions:", "visual_notes:",
            "transcript_blocks:", "faq_candidates:", "searchable_keywords:",
        ]:
            raw = re.sub(r"\n(" + re.escape(key) + r")", r"\n\n\1", raw)

        # 2. Within each transcript block: blank line before cleaned_text and retrieval_tags
        raw = re.sub(r"\n(    cleaned_text:)", r"\n\n\1", raw)
        raw = re.sub(r"\n(    retrieval_tags:)", r"\n\n\1", raw)

        # 3. Blank line between blocks (before every '  - block_id:')
        raw = re.sub(r"\n(  - block_id:)", r"\n\n\1", raw)
        # Remove the double blank that appears right after 'transcript_blocks:' (first block only)
        raw = re.sub(r"(transcript_blocks:)\n\n(  - block_id:)", r"\1\n\2", raw)

        # 4. Collapse 3+ consecutive newlines to max 2
        raw = re.sub(r"\n{3,}", "\n\n", raw)

        return raw
