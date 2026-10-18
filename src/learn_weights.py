"""
Learned fusion: fit per-domain weights from labelled videos.

For every saved analysis with a matching truth file, each segment becomes a
training row (semantic, audio, scene, visual) → label = fraction of the
segment inside a labelled highlight. A non-negative least-squares fit (simple
projected gradient descent, numpy only) gives weights that are then
normalised and written to data/learned_weights.json, which config.py picks up.

  python -m src.learn_weights --jobs data/jobs --truth data/truth
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import SIGNALS, LEARNED_WEIGHTS_PATH, normalise
from src.pipeline import Analysis
from src.evaluate import load_truth


def rows_for(a: Analysis, truth: list[tuple[float, float]]):
    X, y = [], []
    for s in a.segments:
        dur = max(1e-6, s["end"] - s["start"])
        inside = sum(max(0.0, min(s["end"], e) - max(s["start"], b)) for b, e in truth)
        X.append([s[k] for k in SIGNALS])
        y.append(inside / dur)
    return np.array(X), np.array(y)


def fit_nonneg(X: np.ndarray, y: np.ndarray, iters: int = 3000, lr: float = 0.05) -> np.ndarray:
    w = np.full(X.shape[1], 0.25)
    for _ in range(iters):
        grad = X.T @ (X @ w - y) / len(y)
        w = np.clip(w - lr * grad, 0.0, None)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="data/jobs")
    ap.add_argument("--truth", default="data/truth")
    args = ap.parse_args()

    by_domain: dict[str, list] = {}
    for p in Path(args.jobs).glob("*.analysis.json"):
        a = Analysis.load(str(p))
        t = Path(args.truth) / f"{a.name}.json"
        if not t.exists():
            continue
        X, y = rows_for(a, load_truth(str(t)))
        by_domain.setdefault(a.domain, []).append((X, y))

    if not by_domain:
        print("No (analysis, truth) pairs found — label some videos first.")
        return

    learned = json.loads(LEARNED_WEIGHTS_PATH.read_text()) if LEARNED_WEIGHTS_PATH.exists() else {}
    for domain, parts in by_domain.items():
        X = np.vstack([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        w = normalise(dict(zip(SIGNALS, fit_nonneg(X, y))))
        learned[domain] = w
        print(f"{domain:8s} n={len(y):5d}  " + "  ".join(f"{k}={v:.2f}" for k, v in w.items()))
    LEARNED_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEARNED_WEIGHTS_PATH.write_text(json.dumps(learned, indent=1))
    print(f"wrote {LEARNED_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
