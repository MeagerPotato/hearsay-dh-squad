"""Run the interpretable forensic experts over a dataset and save one feature row per clip.

  python scripts/extract_forensic.py test la_dev_m la_train_m:3000 say_m diffssd_m lj_m libri_m:900
  (name:N keeps every real clip plus a random subset so the file has at most N rows)
Output: work/forensic/<name>.npz with names, feature names and a float32 matrix (NaN = not measurable).
"""
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKIP = {"transcript", "repeated_script", "speaker_consistency"}  # slow or batch-level; not used for training


def _features(path):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from hearsay import orchestrator as orch
    from hearsay.evidence import REGISTRY
    orch.ensure_experts_loaded()
    try:
        ctx = orch.triage(path)
    except Exception as e:  # unreadable file
        return path, {}, repr(e)
    feats = {}
    for name, spec in REGISTRY.items():
        if name in SKIP:
            continue
        ev = orch._run_expert(spec, ctx)
        feats.update(orch._clean_features(ev.features))
    return path, feats, None


def files_for(spec, seed=0):
    name, _, limit = spec.partition(":")
    if name == "test":
        fs = sorted(glob.glob(str(ROOT / "data/HackGTHearsayTesting/*.wav")))
        return name, [Path(f).name for f in fs], fs
    rows = list(csv.DictReader(open(ROOT / f"data/proc/{name}.csv")))
    if limit and int(limit) < len(rows):
        rng = np.random.default_rng(seed)
        bona = [r for r in rows if r["is_spoof"] == "0"]
        spoof = [r for r in rows if r["is_spoof"] == "1"]
        n_b = min(len(bona), int(limit) // 2) if spoof else int(limit)
        bona = [bona[i] for i in sorted(rng.choice(len(bona), n_b, replace=False))]
        n_s = min(len(spoof), int(limit) - len(bona))
        spoof = [spoof[i] for i in sorted(rng.choice(len(spoof), n_s, replace=False))] if n_s > 0 else []
        rows = bona + spoof
    return name, [r["utt"] for r in rows], [str(ROOT / f"data/proc/{name}/{r['utt']}.wav") for r in rows]


def main(specs, workers=2):
    out_dir = ROOT / "work/forensic"
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        name, ids, files = files_for(spec)
        out = out_dir / f"{name}.npz"
        if out.exists():
            print("skip", name)
            continue
        rows = {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (path, feats, err) in enumerate(ex.map(_features, files, chunksize=16)):
                rows[path] = feats
                if i % 500 == 0:
                    print(f"{name}: {i}/{len(files)}", flush=True)
        cols = sorted({k for f in rows.values() for k in f})
        M = np.array([[rows[p].get(c, np.nan) for c in cols] for p in files], np.float32)
        np.savez_compressed(out, names=np.array(ids), cols=np.array(cols), X=M)
        print(f"{name}: {M.shape} -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
