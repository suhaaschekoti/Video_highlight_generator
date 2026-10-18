"""
Orchestrator, split in two so the UI can re-cut instantly:

  analyse(video, cfg)  -> Analysis  (transcribe, score all four signals, chapters) — slow, cached to data/jobs/
  render(analysis, cfg, ...) -> reel + manifests                                    — seconds

CLI:  python -m src.pipeline --input v.mp4 --domain sports --duration 120 [--asr local] [--no-visual]
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from dotenv import load_dotenv

from src.config import PipelineConfig, DOMAIN_PRESETS, normalise
from src.transcribe import transcribe, get_duration_seconds, has_video_stream, Segment
from src.llm_scorer import score_segments_by_semantics, summarize_reel, generate_chapters
from src.audio_features import load_audio, score_segments_by_audio_energy
from src.scene_detect import detect_scene_boundaries, score_segments_by_scene_proximity
from src.visual_scorer import score_frames
from src.fusion import fuse, refuse, select_clips, to_dicts, from_dicts, Clip
from src.boundaries import refine
from src.clip_extractor import build_reel, write_srt, write_chapters, write_manifests, export_truth


def _print(stage, pct, msg):
    print(f"[{pct:5.1f}%] {stage}: {msg}")


@dataclass
class Analysis:
    video_path: str
    name: str
    domain: str
    source_duration_s: float
    has_video: bool
    segments: list[dict]            # ScoredSegment dicts (raw signals + final)
    chapters: list[dict] = field(default_factory=list)
    n_scene_cuts: int = 0
    analysis_seconds: float = 0.0

    def save(self, jobs_dir: str) -> str:
        Path(jobs_dir).mkdir(parents=True, exist_ok=True)
        p = Path(jobs_dir) / f"{self.name}.analysis.json"
        p.write_text(json.dumps(asdict(self), ensure_ascii=False), encoding="utf-8")
        return str(p)

    @staticmethod
    def load(path: str) -> "Analysis":
        return Analysis(**json.loads(Path(path).read_text(encoding="utf-8")))


def analyse(video_path: str, cfg: PipelineConfig, progress=_print) -> Analysis:
    cfg.validate()
    t0 = time.time()
    name = Path(video_path).stem
    duration = get_duration_seconds(video_path)
    has_vid = has_video_stream(video_path)

    progress("transcribe", 2, f"transcribing ({cfg.asr_backend})")
    segments = transcribe(video_path, cfg.asr_backend, cfg.groq_asr_model, cfg.local_asr_model,
                          cfg.asr_language, cfg.asr_chunk_seconds, cfg.transcript_cache_dir)
    if not segments:
        raise RuntimeError("No speech found — this pipeline needs spoken audio.")
    progress("transcribe", 22, f"{len(segments)} segments")

    progress("semantic", 24, "LLM scoring")
    sem = score_segments_by_semantics(segments, cfg.llm_model, cfg.llm_batch_size, cfg.domain,
                                      cfg.llm_max_retries,
                                      lambda b, n: progress("semantic", 24 + 24 * b / n, f"batch {b}/{n}"))

    progress("audio", 50, "loudness, pitch, flux, speech gate")
    y, sr = load_audio(video_path, cache_dir=cfg.transcript_cache_dir)
    aud = score_segments_by_audio_energy(y, sr, segments, cfg.use_speech_gate)

    progress("scene", 58, "scene cuts")
    cuts = detect_scene_boundaries(video_path, cfg.scene_detector_threshold) if has_vid else []
    scn = score_segments_by_scene_proximity(segments, cuts)

    vis: dict[int, dict] = {}
    if cfg.use_visual and has_vid and cfg.weights.get("visual", 0) > 0:
        # preliminary fusion without visual → pick candidates → score their frames
        prelim_w = normalise({k: v for k, v in cfg.weights.items() if k != "visual"})
        prelim = fuse(segments, sem, aud, scn, {}, prelim_w)
        cand = [s.id for s in sorted(prelim, key=lambda s: s.final, reverse=True)[:cfg.visual_top_k]]
        progress("visual", 62, f"VLM scoring {len(cand)} candidate frames")
        vis = score_frames(video_path, segments, cand, cfg.vision_model, cfg.domain,
                           lambda i, n: progress("visual", 62 + 18 * i / n, f"frame {i}/{n}"))

    scored = fuse(segments, sem, aud, scn, vis, cfg.weights)

    chapters = []
    if cfg.generate_chapters:
        progress("chapters", 82, "chapter titles")
        chapters = generate_chapters(segments, cfg.llm_model, cfg.domain)

    a = Analysis(video_path, name, cfg.domain, duration, has_vid, to_dicts(scored), chapters,
                 len(cuts), round(time.time() - t0, 1))
    a.save(cfg.jobs_dir)
    progress("analysed", 85, "analysis complete")
    return a


def render(a: Analysis, cfg: PipelineConfig, force_include=None, force_exclude=None,
           progress=_print, out_dir: str | None = None) -> dict:
    t0 = time.time()
    out = Path(out_dir or cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored = refuse(from_dicts(a.segments), cfg.weights)
    clips = select_clips(scored, cfg, a.source_duration_s, set(force_include or []), set(force_exclude or []))
    if not clips:
        raise RuntimeError("No clips selected — increase the target length.")

    if cfg.refine_boundaries:
        progress("refine", 87, "snapping cuts to words and pauses")
        seg_objs = [Segment.from_dict(s) for s in json.loads(
            (Path(cfg.transcript_cache_dir) / f"{a.name}.json").read_text(encoding="utf-8"))]
        y, sr = load_audio(a.video_path, cache_dir=cfg.transcript_cache_dir)
        clips = refine(clips, seg_objs, y, sr, cfg.lead_in_padding_s, cfg.lead_out_padding_s, a.source_duration_s)
        from src.fusion import merge_close
        clips = merge_close(clips, cfg.merge_gap_s)

    meta = {"title": "Highlights", "summary": ""}
    if cfg.generate_summary:
        progress("summary", 90, "title and summary")
        meta = summarize_reel([c.text for c in clips], cfg.llm_model, cfg.domain)

    video_out = str(out / f"{a.name}_highlight.mp4")
    srt_out = str(out / f"{a.name}_highlight.srt")
    chap_out = str(out / f"{a.name}_chapters.txt")
    json_out = str(out / f"{a.name}_manifest.json")
    txt_out = str(out / f"{a.name}_manifest.txt")
    truth_out = str(out / f"{a.name}_labels.json")

    write_srt(clips, srt_out)
    write_chapters(a.chapters, chap_out)
    progress("render", 92, f"cutting and stitching {len(clips)} clips")
    build_reel(a.video_path, clips, video_out, a.has_video, cfg.burn_subtitles, srt_out)

    reel = sum(c.end - c.start for c in clips)
    meta.update({"source": Path(a.video_path).name, "domain": cfg.domain, "weights": cfg.weights,
                 "source_duration_s": a.source_duration_s, "reel_duration_s": reel,
                 "compression_ratio": a.source_duration_s / reel if reel else 0,
                 "n_segments": len(a.segments), "n_clips": len(clips), "n_scene_cuts": a.n_scene_cuts,
                 "analysis_seconds": a.analysis_seconds, "render_seconds": round(time.time() - t0, 1),
                 "force_include": sorted(force_include or []), "force_exclude": sorted(force_exclude or [])})
    write_manifests(clips, to_dicts(scored), meta, a.chapters, json_out, txt_out)
    export_truth(clips, truth_out)
    progress("done", 100, f"reel {reel:.0f}s from {a.source_duration_s:.0f}s")
    return {"video": video_out, "srt": srt_out, "chapters": chap_out, "manifest_json": json_out,
            "manifest_txt": txt_out, "labels": truth_out, "meta": meta, "clips": clips}


def run_pipeline(video_path: str, cfg: PipelineConfig, progress=_print) -> dict:
    return render(analyse(video_path, cfg, progress), cfg, progress=progress)


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--domain", default="general", choices=list(DOMAIN_PRESETS))
    ap.add_argument("--duration", type=int, default=120)
    ap.add_argument("--asr", default="groq", choices=["groq", "local"])
    ap.add_argument("--no-visual", action="store_true")
    ap.add_argument("--burn-subtitles", action="store_true")
    args = ap.parse_args()
    cfg = PipelineConfig(domain=args.domain, target_highlight_duration_s=args.duration, asr_backend=args.asr,
                         use_visual=not args.no_visual, burn_subtitles=args.burn_subtitles)
    r = run_pipeline(args.input, cfg)
    print(f"\n{r['meta']['title']}\n{r['meta']['summary']}\n\nReel: {r['video']}\nManifest: {r['manifest_txt']}")


if __name__ == "__main__":
    main()
