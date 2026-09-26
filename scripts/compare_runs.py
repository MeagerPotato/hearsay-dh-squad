"""Compare two `hearsay predict` runs (e.g. the Docker image vs the run that produced our submission).

  python scripts/compare_runs.py out/dh_squad_predictions_scores.json \
      submissions/final_pipeline/dh_squad_predictions_scores.json

Reports, per detector and for the final cm-score: files in common, largest absolute difference, and the
Spearman rank correlation. The two runs need not cover the same files; only common files are compared.
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr


def load(path):
    return json.load(open(path))["files"]


def main(a_path, b_path):
    a, b = load(a_path), load(b_path)
    common = sorted(set(a) & set(b))
    if not common:
        sys.exit("no files in common")
    models = sorted(set(a[common[0]]["scores"]) & set(b[common[0]]["scores"]))
    print(f"{len(common)} files in common")
    rows = [(m, [a[f]["scores"][m] for f in common], [b[f]["scores"][m] for f in common]) for m in models]
    rows.append(("cm_score", [a[f]["cm_score"] for f in common], [b[f]["cm_score"] for f in common]))
    for name, x, y in rows:
        x, y = np.array(x, float), np.array(y, float)
        rho = spearmanr(x, y)[0] if len(x) > 2 else float("nan")
        print(f"  {name:12s} max |diff| {np.abs(x - y).max():.2e}   Spearman {rho:.6f}")
    ca = np.array([a[f]["cm_score"] for f in common])
    cb = np.array([b[f]["cm_score"] for f in common])
    flips = int(np.sum((ca > 0.5) != (cb > 0.5)))
    print(f"  files whose verdict (cm-score > 0.5) differs: {flips}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
