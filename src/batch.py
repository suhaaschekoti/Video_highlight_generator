"""
Batch mode: process every video in a folder, evaluate where a truth file
exists, and write results.csv — the results table for the report in one command.

  python -m src.batch --input data/input --domain lecture --duration 120 --truth data/truth
"""

import argparse
import csv
import time
from pathlib import Path

from dotenv import load_dotenv

from src.config import PipelineConfig, DOMAIN_PRESETS
from src.pipeline import analyse, render
from src.evaluate import evaluate, load_truth

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/input")
    ap.add_argument("--domain", default="general", choices=list(DOMAIN_PRESETS))
    ap.add_argument("--duration", type=int, default=120)
    ap.add_argument("--truth", default="data/truth")
    ap.add_argument("--out", default="data/output/results.csv")
    ap.add_argument("--no-visual", action="store_true")
    args = ap.parse_args()

    rows = []
    for v in sorted(Path(args.input).iterdir()):
        if v.suffix.lower() not in VIDEO_EXT:
            continue
        cfg = PipelineConfig(domain=args.domain, target_highlight_duration_s=args.duration,
                             use_visual=not args.no_visual)
        t0 = time.time()
        try:
            a = analyse(str(v), cfg, progress=lambda *_: None)
            r = render(a, cfg, progress=lambda *_: None)
            row = {"video": v.name, "domain": args.domain, "source_s": round(a.source_duration_s, 1),
                   "reel_s": round(r["meta"]["reel_duration_s"], 1), "clips": r["meta"]["n_clips"],
                   "seconds_per_source_min": round((time.time() - t0) / (a.source_duration_s / 60), 1)}
            t = Path(args.truth) / f"{v.stem}.json"
            if t.exists():
                row.update(evaluate([(c.start, c.end) for c in r["clips"]], load_truth(str(t))))
            print(row)
        except Exception as e:
            row = {"video": v.name, "error": str(e)[:200]}
            print(row)
        rows.append(row)

    keys = sorted({k for r in rows for k in r})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
