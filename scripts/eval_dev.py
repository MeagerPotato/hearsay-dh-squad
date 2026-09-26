"""Offline leaderboard: per-detector and fused minDCF on test-matched dev data, broken down by spoof source.

  python scripts/eval_dev.py --sets la_dev_m diffssd_m say_m lj_m --models w2v2_aasist xlsr_sls aasist19
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hearsay.fusion import Standardiser, cv_mindcf  # noqa: E402
from hearsay.metrics import evaluate  # noqa: E402


def load_set(name, models):
    rows = list(csv.DictReader(open(ROOT / f"data/proc/{name}.csv")))
    scores = {}
    for m in models:
        p = ROOT / f"work/scores/{name}/{m}.json"
        scores[m] = json.load(open(p))[m] if p.exists() else {}
    out = []
    for r in rows:
        # detector JSONs are keyed by file name ("<utt>.wav"); embedding heads by utterance id
        v = [scores[m].get(r["utt"], scores[m].get(r["utt"] + ".wav", np.nan)) for m in models]
        out.append({"utt": r["utt"], "set": name, "y": int(r["is_spoof"]),
                    "src": r.get("attack", "bonafide") if r["is_spoof"] == "1" else "bonafide",
                    "group": r.get("speaker") or r.get("voice") or r.get("attack") or name, "x": v})
    return out


def fmt(d):
    return f"minDCF {d['minDCF_intended']:.4f} (flipped {d['minDCF_shipped_flipped']:.4f})  EER {100 * d['EER']:.2f}%  [{d['n_bona']} bona / {d['n_spoof']} spoof]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    data = [r for s in a.sets for r in load_set(s, a.models)]
    X = np.array([r["x"] for r in data], float)
    y = np.array([r["y"] for r in data])
    keep = ~np.isnan(X).any(1)
    print(f"{keep.sum()} of {len(X)} clips have all model scores")
    data = [r for r, k in zip(data, keep) if k]
    X, y = X[keep], y[keep]
    bona = y == 0
    sources = sorted({r["src"] for r in data if r["y"] == 1})
    src = np.array([r["src"] for r in data])
    for j, m in enumerate(a.models):
        print(f"\n== {m}: ALL  {fmt(evaluate(X[:, j], y))}")
        for s in sources:
            sel = bona | (src == s)
            print(f"   {s:>14s}: {fmt(evaluate(X[sel, j], y[sel]))}")
    Z = Standardiser().fit(X).transform(X)
    groups = np.array([r["group"] for r in data])
    print("\n== equal-weight z-mean fusion:", fmt(evaluate(Z.mean(1), y)))
    for C in (0.01, 0.1, 1.0):
        res, _ = cv_mindcf(Z, y, groups, a.models, C=C)
        print(f"== LR fusion (grouped 5-fold CV, C={C}): {fmt(res)}")


if __name__ == "__main__":
    main()
