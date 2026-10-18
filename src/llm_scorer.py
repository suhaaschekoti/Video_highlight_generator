"""
Stage 2c: LLM semantic scoring (primary signal), reel title/summary, and
chapter generation — all via Groq's free tier with JSON mode and backoff.
"""

import json
import re
import time

from groq import Groq

from src.config import get_api_key
from src.transcribe import Segment

SCORING_PROMPT = """You are assisting with automatic video highlight generation.
You will be given consecutive transcript segments from a {domain} video, each with an id and text.

For EACH segment give an "interestingness" score from 0.0 to 1.0: how likely it is to belong in a highlight reel.
Consider: key statements, conclusions, definitions, turning points; emotionally charged or exciting language;
strong claims and memorable quotes; clear topic shifts that start something important.
Score filler, transitions, repetition and housekeeping near 0. Highlights are rare — use the full range and
do not cluster everything around 0.5. Give a reason of at most 12 words.

Respond ONLY with JSON: {{"results": [{{"id": 0, "score": 0.12, "reason": "routine transition"}}, ...]}}"""

SUMMARY_PROMPT = """These are the transcript texts of clips chosen for a highlight reel of a {domain} video.
Respond ONLY with JSON: {{"title": "<specific title, max 10 words>", "summary": "<two plain sentences on what the reel covers>"}}"""

CHAPTER_PROMPT = """You are given a timestamped transcript of a {domain} video as lines "[id @ seconds] text".
Split it into 3-12 coherent chapters. Each chapter starts at the id of its first segment.
Respond ONLY with JSON: {{"chapters": [{{"start_id": 0, "title": "<max 8 words>"}}, ...]}}"""


def _client():
    return Groq(api_key=get_api_key())


def _parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON in LLM response: {raw[:200]}")
        return json.loads(m.group(0))


def chat_json(system: str, user: str, model: str, max_retries: int = 5, max_tokens: int = 2500) -> dict:
    client = _client()
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=0.2,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return _parse(r.choices[0].message.content)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt if "429" in str(e) or "rate" in str(e).lower() else 1.5)
    return {}


def score_segments_by_semantics(segments: list[Segment], model: str, batch_size=20, domain="general",
                                max_retries=5, progress=None) -> dict[int, dict]:
    system = SCORING_PROMPT.format(domain=domain)
    results: dict[int, dict] = {}
    batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]
    for b, batch in enumerate(batches, 1):
        user = json.dumps([{"id": s.id, "text": s.text} for s in batch], ensure_ascii=False)
        parsed = chat_json(system, user, model, max_retries)
        for item in parsed.get("results", []):
            try:
                results[int(item["id"])] = {"score": min(1.0, max(0.0, float(item["score"]))),
                                            "reason": str(item.get("reason", ""))[:160]}
            except (KeyError, ValueError, TypeError):
                continue
        if progress:
            progress(b, len(batches))
    for s in segments:
        results.setdefault(s.id, {"score": 0.1, "reason": "no LLM score returned"})
    return results


def summarize_reel(clip_texts: list[str], model: str, domain="general") -> dict:
    joined = "\n".join(f"- {t}" for t in clip_texts)[:6000]
    try:
        out = chat_json(SUMMARY_PROMPT.format(domain=domain), joined, model, 3, 400)
        return {"title": str(out.get("title") or "Highlights"), "summary": str(out.get("summary") or "")}
    except Exception:
        return {"title": "Highlights", "summary": ""}


def generate_chapters(segments: list[Segment], model: str, domain="general") -> list[dict]:
    """YouTube-style chapters: [{start, title}]. Long transcripts are thinned to fit."""
    if len(segments) < 6:
        return []
    step = max(1, len(segments) // 250)  # keep the prompt bounded on long videos
    lines = [f"[{s.id} @ {int(s.start)}] {s.text[:120]}" for s in segments[::step]]
    try:
        out = chat_json(CHAPTER_PROMPT.format(domain=domain), "\n".join(lines), model, 3, 800)
    except Exception:
        return []
    by_id = {s.id: s for s in segments}
    chapters = []
    for c in out.get("chapters", []):
        try:
            seg = by_id[int(c["start_id"])]
            chapters.append({"start": seg.start, "title": str(c.get("title", "")).strip()[:80]})
        except (KeyError, ValueError, TypeError):
            continue
    chapters.sort(key=lambda c: c["start"])
    if chapters and chapters[0]["start"] > 0:
        chapters[0]["start"] = 0.0
    return chapters
