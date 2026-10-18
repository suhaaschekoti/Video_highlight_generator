"""
Ablation runner: re-fuse a saved analysis with one signal removed at a time
(and single-signal variants), re-select, and evaluate against truth. No
transcription or API calls — takes seconds.

  python -m src.ablate --analysis data/jobs/x.analysis.json --truth data/truth/x.json --duration 120
"""

import argparse
from dotenv import load_dotenv

from src.config import PipelineConfig, SIGNALS, normalise
from src.pipeline import Analysis
from src.fusion import refuse, from_dicts, select_clips
from src.evaluate import evaluate, load_truth


def variants(base: dict[str, float]) -> dict[str, dict[str, float]]:
    v = {"full": base}
    for k in SIGNALS:
        if base.get(k, 0) > 0:
            v[f"no_{k}"] = normalise({kk: (0 if kk == k else vv) for kk, vv in base.items()})
            v[f"only_{k}"] = normalise({kk: (1 if kk == k else 0) for kk in SIGNALS})
    return v


def run(analysis_path: str, truth_path: str, duration: int) -> list[dict]:
    a = Analysis.load(analysis_path)
    truth = load_truth(truth_path)
    rows = []
    for name, w in variants(PipelineConfig(domain=a.domain).weights).items():
        cfg = PipelineConfig(domain=a.domain, target_highlight_duration_s=duration, weights=w)
        clips = select_clips(refuse(from_dicts(a.segments), cfg.weights), cfg, a.source_duration_s)
        m = evaluate([(c.start, c.end) for c in clips], truth)
        rows.append({"variant": name, **m, "weights": cfg.weights})
    return rows


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--duration", type=int, default=120)
    args = ap.parse_args()
    rows = run(args.analysis, args.truth, args.duration)
    print(f"{'variant':14s} {'P':>6s} {'R':>6s} {'F1':>6s} {'hit':>6s}")
    for r in rows:
        print(f"{r['variant']:14s} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f} {r['hit_rate']:6.3f}")


if __name__ == "__main__":
    main()
