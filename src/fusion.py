"""
Stage 3: fusion + selection.

final = Σ w_k · signal_k  over {semantic, audio, scene, visual}
Manual overrides: force_include ids are always chosen first; force_exclude
ids are never chosen. Selection is greedy under a duration budget, then
padded, ordered chronologically, and merged.
"""

from dataclasses import dataclass, asdict

from src.transcribe import Segment
from src.config import PipelineConfig, SIGNALS


@dataclass
class ScoredSegment:
    id: int
    start: float
    end: float
    text: str
    semantic: float
    audio: float
    scene: float
    visual: float
    final: float
    reason: str
    caption: str = ""


@dataclass
class Clip:
    start: float
    end: float
    text: str
    score: float
    reason: str
    segment_ids: list[int]


def fuse(segments: list[Segment], sem: dict, aud: dict, scn: dict, vis: dict,
         weights: dict[str, float]) -> list[ScoredSegment]:
    out = []
    for s in segments:
        se = sem.get(s.id, {"score": 0.0, "reason": ""})
        v = vis.get(s.id, {"score": 0.0, "caption": ""})
        vals = {"semantic": se["score"], "audio": aud.get(s.id, 0.0),
                "scene": scn.get(s.id, 0.0), "visual": v["score"]}
        final = sum(weights.get(k, 0.0) * vals[k] for k in SIGNALS)
        out.append(ScoredSegment(s.id, s.start, s.end, s.text, vals["semantic"], vals["audio"],
                                 vals["scene"], vals["visual"], final, se["reason"], v.get("caption", "")))
    return out


def refuse(scored: list[ScoredSegment], weights: dict[str, float]) -> list[ScoredSegment]:
    """Recompute `final` with new weights without touching the raw signals (used by re-cut)."""
    for s in scored:
        s.final = (weights["semantic"] * s.semantic + weights["audio"] * s.audio
                   + weights["scene"] * s.scene + weights["visual"] * s.visual)
    return scored


def select_clips(scored: list[ScoredSegment], config: PipelineConfig, video_duration: float,
                 force_include: set[int] | None = None, force_exclude: set[int] | None = None) -> list[Clip]:
    force_include = force_include or set()
    force_exclude = force_exclude or set()
    ranked = sorted(scored, key=lambda s: (s.id in force_include, s.final), reverse=True)
    chosen, total = [], 0.0
    for s in ranked:
        if s.id in force_exclude:
            continue
        if total >= config.target_highlight_duration_s and s.id not in force_include:
            break
        seg_end = max(s.end, s.start + config.min_clip_duration_s)
        start = max(0.0, s.start - config.lead_in_padding_s)
        end = min(seg_end + config.lead_out_padding_s, s.start + config.max_clip_duration_s, video_duration)
        if end - start <= 0.5:
            continue
        chosen.append(Clip(start, end, s.text, s.final, s.reason, [s.id]))
        total += end - start
    chosen.sort(key=lambda c: c.start)
    return merge_close(chosen, config.merge_gap_s)


def merge_close(clips: list[Clip], gap: float) -> list[Clip]:
    if not clips:
        return clips
    merged = [clips[0]]
    for c in clips[1:]:
        last = merged[-1]
        if c.start <= last.end + gap:
            merged[-1] = Clip(last.start, max(last.end, c.end), (last.text + " " + c.text).strip(),
                              max(last.score, c.score),
                              last.reason if last.score >= c.score else c.reason,
                              last.segment_ids + c.segment_ids)
        else:
            merged.append(c)
    return merged


def to_dicts(scored: list[ScoredSegment]) -> list[dict]:
    return [asdict(s) for s in scored]


def from_dicts(rows: list[dict]) -> list[ScoredSegment]:
    return [ScoredSegment(**r) for r in rows]
