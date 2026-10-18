"""
Time-overlap evaluation against hand-labelled highlights.

truth file: {"highlights": [{"start": 62.0, "end": 78.5}, ...]}

  python -m src.evaluate --manifest data/output/x_manifest.json --truth data/truth/x.json
"""

import argparse
import json


def overlap(a, b) -> float:
    return sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)


def evaluate(pred: list[tuple[float, float]], truth: list[tuple[float, float]]) -> dict:
    pl = sum(e - s for s, e in pred)
    tl = sum(e - s for s, e in truth)
    ov = overlap(pred, truth)
    p = ov / pl if pl else 0.0
    r = ov / tl if tl else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    hits = sum(1 for s, e in truth if any(min(e, ce) - max(s, cs) > 0 for cs, ce in pred))
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "hit_rate": round(hits / len(truth), 3) if truth else 0.0,
            "overlap_s": round(ov, 1), "pred_s": round(pl, 1), "truth_s": round(tl, 1)}


def load_truth(path: str) -> list[tuple[float, float]]:
    with open(path) as f:
        return [(h["start"], h["end"]) for h in json.load(f)["highlights"]]


def load_pred(manifest_path: str) -> list[tuple[float, float]]:
    with open(manifest_path) as f:
        return [(c["start"], c["end"]) for c in json.load(f)["clips"]]


def agreement(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Jaccard overlap between two annotators' label sets (inter-annotator agreement)."""
    inter = overlap(a, b)
    union = sum(e - s for s, e in a) + sum(e - s for s, e in b) - inter
    return round(inter / union, 3) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--truth2", help="second annotator's file → also prints agreement")
    args = ap.parse_args()
    truth = load_truth(args.truth)
    for k, v in evaluate(load_pred(args.manifest), truth).items():
        print(f"{k:12s} {v}")
    if args.truth2:
        print(f"{'agreement':12s} {agreement(truth, load_truth(args.truth2))}")


if __name__ == "__main__":
    main()
