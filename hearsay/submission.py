"""Write and validate the HEARSAY prediction TSV (filename<TAB>cm-score, 1.0 = synthetic)."""
import csv
from pathlib import Path

import numpy as np

TEAM = "dh_squad"


def read_template(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    assert rows[0] == ["filename", "cm-score"], rows[0]
    return [r[0] for r in rows[1:]]


def to_probability(raw, temperature=None):
    """Map raw spoof log-odds to (0, 1) without changing rank order or creating ties.

    minDCF depends only on rank, but the organizers require probabilities, and saturated or rounded
    values can create ties, which never help. We z-score, apply a gentle sigmoid, and break any
    remaining exact ties with a tiny deterministic epsilon.
    """
    raw = np.asarray(raw, float)
    z = (raw - raw.mean()) / (raw.std() + 1e-12)
    t = temperature or 2.0
    p = 1.0 / (1.0 + np.exp(-z / t))
    order = np.argsort(np.argsort(raw, kind="mergesort"), kind="mergesort")
    p = p + order * 1e-12  # strict monotone tie-break, far below any meaningful difference
    return np.clip(p, 1e-6, 1 - 1e-6)


def write(scores_by_name, template_path, out_path):
    names = read_template(template_path)
    missing = [n for n in names if n not in scores_by_name]
    assert not missing, f"{len(missing)} files unscored, e.g. {missing[:3]}"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        f.write("filename\tcm-score\n")
        for n in names:
            f.write(f"{n}\t{scores_by_name[n]:.12f}\n")
    validate(out_path, template_path)
    return out_path


def validate(path, template_path):
    names = read_template(template_path)
    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    assert rows[0] == ["filename", "cm-score"], "bad header"
    got = [r[0] for r in rows[1:]]
    assert got == names, "filenames/order differ from template"
    v = np.array([float(r[1]) for r in rows[1:]])
    assert np.all(np.isfinite(v)) and v.min() >= 0 and v.max() <= 1, "scores must be finite in [0,1]"
    assert len(np.unique(v)) == len(v), "tied scores present"
    return True
