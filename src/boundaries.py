"""
Clip boundary refinement.

A clip that starts on Whisper's segment boundary can begin a hair before a
word or cut into the previous one. We snap to the nearest of:
  1. the first/last word edges (from word timestamps), then
  2. the nearest silence gap (librosa.effects.split) within a small window,
so cuts land on natural pauses. Falls back gracefully when data is missing.
"""

import numpy as np
import librosa

from src.fusion import Clip
from src.transcribe import Segment


def silence_gaps(y: np.ndarray, sr: int, top_db: int = 32) -> list[tuple[float, float]]:
    """Return (start, end) of non-silent regions; gaps between them are silences."""
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256)
    return [(s / sr, e / sr) for s, e in intervals]


def _nearest_gap_edge(t: float, voiced: list[tuple[float, float]], window: float, want: str) -> float:
    """Move t to the nearest silence edge within ±window. want='start' → move to a gap end
    (speech begins), want='end' → move to a gap start (speech stops)."""
    best, best_d = t, window
    for i in range(1, len(voiced)):
        gap_start, gap_end = voiced[i - 1][1], voiced[i][0]
        cand = gap_end if want == "start" else gap_start
        d = abs(cand - t)
        if d < best_d:
            best, best_d = cand, d
    return best


def refine(clips: list[Clip], segments: list[Segment], y: np.ndarray, sr: int,
           lead_in: float, lead_out: float, video_duration: float) -> list[Clip]:
    by_id = {s.id: s for s in segments}
    voiced = silence_gaps(y, sr)
    out = []
    for c in clips:
        segs = [by_id[i] for i in c.segment_ids if i in by_id]
        start, end = c.start, c.end
        words = [w for s in segs for w in s.words]
        if words:
            start = min(start, words[0].start - lead_in)
            end = max(end, words[-1].end + lead_out)
        if voiced:
            start = _nearest_gap_edge(start, voiced, 0.6, "start") - 0.15
            end = _nearest_gap_edge(end, voiced, 0.8, "end") + 0.15
        start = max(0.0, start)
        end = min(video_duration, end)
        if end - start < 1.0:
            start, end = c.start, c.end
        out.append(Clip(start, end, c.text, c.score, c.reason, c.segment_ids))
    return out
