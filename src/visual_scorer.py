"""
Stage 2d: visual semantics with a Groq-hosted vision-language model (free tier).

Scoring every segment's frame would cost hundreds of requests, so we only
score the top-k candidates from a preliminary fusion of the other signals
("visual re-ranking"). Each candidate gets one mid-segment JPEG, captioned
and scored 0–1 for visual interest. Non-candidates score 0 and the fusion
weight for `visual` is modest, so this refines rather than dominates.
"""

import base64
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from groq import Groq

from src.config import get_api_key
from src.transcribe import Segment

PROMPT = ("You are scoring a single frame from a {domain} video for a highlight reel. "
          "Reply ONLY with JSON: {{\"caption\": \"<one short sentence>\", \"score\": <0.0-1.0>}}. "
          "Score high for visually eventful, emotional, or informative frames (action, reactions, "
          "key slides, demonstrations); low for static, blank, or transitional frames.")


def grab_frame(video_path: str, t: float, out_path: str, width: int = 512) -> bool:
    r = subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", video_path, "-frames:v", "1",
                        "-vf", f"scale={width}:-1", "-q:v", "6", out_path], capture_output=True)
    return r.returncode == 0 and Path(out_path).exists()


def _parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else {}


def score_frames(video_path: str, segments: list[Segment], candidate_ids: list[int],
                 model: str, domain: str = "general", progress=None) -> dict[int, dict]:
    client = Groq(api_key=get_api_key())
    by_id = {s.id: s for s in segments}
    out: dict[int, dict] = {}

    with tempfile.TemporaryDirectory() as tmp:
        for n, sid in enumerate(candidate_ids, 1):
            seg = by_id[sid]
            frame = str(Path(tmp) / f"{sid}.jpg")
            if not grab_frame(video_path, (seg.start + seg.end) / 2, frame):
                continue
            b64 = base64.b64encode(Path(frame).read_bytes()).decode()
            for attempt in range(4):
                try:
                    resp = client.chat.completions.create(
                        model=model, max_tokens=120, temperature=0.2,
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": PROMPT.format(domain=domain)},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ]}])
                    d = _parse(resp.choices[0].message.content or "")
                    out[sid] = {"score": min(1.0, max(0.0, float(d.get("score", 0.0)))),
                                "caption": str(d.get("caption", ""))[:160]}
                    break
                except Exception as e:
                    if attempt == 3:
                        out[sid] = {"score": 0.0, "caption": ""}
                    else:
                        time.sleep(2 ** attempt if "429" in str(e) else 1)
            if progress:
                progress(n, len(candidate_ids))
    return out
