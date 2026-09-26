"""Score a set of files with one deep detector and save {model: {basename: spoof_log_odds}} as JSON.

  python scripts/score_detector.py w2v2_aasist test          -> work/scores/test/w2v2_aasist.json
  python scripts/score_detector.py w2v2_aasist la_dev --limit 4000
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hearsay import deep  # noqa: E402


def file_set(name, limit=None, seed=0):
    if name == "test":
        return sorted(glob.glob(str(ROOT / "data/HackGTHearsayTesting/*.wav")))
    csv_path = ROOT / f"data/proc/{name}.csv"
    rows = list(csv.DictReader(open(csv_path)))
    if limit and limit < len(rows):
        # keep every bona fide clip (the scarce class) and a random spoof subset balanced across attacks
        rng = np.random.default_rng(seed)
        bona = [r for r in rows if r["is_spoof"] == "0"]
        spoof = [r for r in rows if r["is_spoof"] == "1"]
        take = max(0, limit - len(bona))
        spoof = [spoof[i] for i in sorted(rng.choice(len(spoof), size=min(take, len(spoof)), replace=False))]
        rows = bona + spoof
    return [str(ROOT / f"data/proc/{name}/{r['utt']}.wav") for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dataset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    files = file_set(a.dataset, a.limit)
    det = deep.build(a.model, threads=a.threads)
    mean, first = deep.score_files(det, files, max_windows=a.windows, return_first=True)
    out = ROOT / f"work/scores/{a.dataset}/{a.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({a.model: mean, f"{a.model}_w1": first}, open(out, "w"))
    v = np.array(list(mean.values()))
    print(f"{a.model} on {a.dataset}: n={len(v)} frac>0={np.mean(v > 0):.3f} -> {out}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
