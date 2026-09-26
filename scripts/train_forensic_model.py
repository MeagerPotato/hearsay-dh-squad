"""Interpretable spoof model on the forensic experts' named features (the "glass-box" branch).

Feature hygiene before training:
  1. drop features that are constant on the test set (e.g. container metadata, laundered by ffmpeg);
  2. drop features the expert authors flagged as corpus shortcuts;
  3. adversarial validation: drop features that tell our matched dev data apart from the test set too
     easily (|AUC - 0.5| > 0.3), because those measure dataset differences, not synthesis.
Then fit LightGBM (with real clips weighted 4x) and report minDCF plus feature importances.

  python scripts/train_forensic_model.py --train la_train_m say_m@a diffssd_m@a libri_m@a \
      --eval la_dev_m say_m@b diffssd_m@b lj_m libri_m@b --apply test
"""
import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hearsay.metrics import evaluate  # noqa: E402

SHORTCUT = {"codec_cutoff_depth_db", "codec_cutoff_steep_db_per_khz", "double_compression_hf_flatness"}


def _group(r):
    return r.get("voice") or r.get("speaker") or r.get("attack") or ""


def load_set(spec):
    name, _, half = spec.partition("@")
    d = np.load(ROOT / f"work/forensic/{name}.npz", allow_pickle=False)
    ids, cols, X = [str(i) for i in d["names"]], [str(c) for c in d["cols"]], d["X"].astype(np.float64)
    if name == "test":
        return ids, cols, X, None
    rows = {r["utt"]: r for r in csv.DictReader(open(ROOT / f"data/proc/{name}.csv"))}
    keep = np.ones(len(ids), bool)
    if half:
        keep = np.array([(zlib.crc32(_group(rows[i]).encode()) % 2 == 0) == (half == "a") for i in ids])
    ids = [i for i, k in zip(ids, keep) if k]
    y = np.array([int(rows[i]["is_spoof"]) for i in ids])
    return ids, cols, X[keep], y


def align(cols_src, X, cols):
    idx = {c: j for j, c in enumerate(cols_src)}
    out = np.full((len(X), len(cols)), np.nan)
    for k, c in enumerate(cols):
        if c in idx:
            out[:, k] = X[:, idx[c]]
    return out


def auc(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        return 0.5
    r = np.argsort(np.argsort(np.r_[a, b]))
    return (r[: len(a)].sum() - len(a) * (len(a) - 1) / 2) / (len(a) * len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--apply", nargs="*", default=["test"])
    a = ap.parse_args()
    import lightgbm as lgb

    t_ids, t_cols, Xt_raw, _ = load_set("test")
    cols = [c for c in t_cols if c not in SHORTCUT and not c.startswith("container_")]
    Xt = align(t_cols, Xt_raw, cols)
    # 1) constant on test
    std = np.nanstd(Xt, 0)
    keep = std > 1e-9
    # 3) adversarial validation against the matched dev set
    d_ids, d_cols, Xd_raw, yd = load_set("la_dev_m")
    Xd = align(d_cols, Xd_raw, cols)
    shift = np.array([abs(auc(Xd[:, j], Xt[:, j]) - 0.5) for j in range(len(cols))])
    keep &= shift <= 0.3
    dropped_shift = [c for c, s, k in zip(cols, shift, std > 1e-9) if k and s > 0.3]
    cols = [c for c, k in zip(cols, keep) if k]
    print(f"features kept: {len(cols)}; dropped for train/test shift: {len(dropped_shift)} e.g. {dropped_shift[:8]}")

    def stack(specs):
        Xs, ys = [], []
        for s in specs:
            ids, cs, X, y = load_set(s)
            Xs.append(align(cs, X, cols))
            ys.append(y)
        return np.concatenate(Xs), np.concatenate(ys)

    Xtr, ytr = stack(a.train)
    w = np.where(ytr == 0, 4.0, 1.0)
    params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2, seed=0)
    model = lgb.train(params, lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=400)
    report = {"n_features": len(cols), "dropped_shift": dropped_shift}
    for s in a.eval:
        ids, cs, X, y = load_set(s)
        p = model.predict(align(cs, X, cols), raw_score=True)
        r = evaluate(p, y.astype(bool)) if len(set(y)) > 1 else {"frac_flagged": float((p > 0).mean())}
        print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        report[s] = r
        name = s.partition("@")[0]
        dst = ROOT / f"work/scores/{name}/forensic_gbm{'_' + s.partition('@')[2] if '@' in s else ''}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"forensic_gbm": {i + ".wav": float(v) for i, v in zip(ids, p)}}, open(dst, "w"))
    for s in a.apply:
        ids, cs, X, _ = load_set(s)
        p = model.predict(align(cs, X, cols), raw_score=True)
        dst = ROOT / f"work/scores/{s}/forensic_gbm.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"forensic_gbm": dict(zip(ids, p.astype(float).tolist()))}, open(dst, "w"))
        print(f"applied to {s}: frac>0 {np.mean(p > 0):.3f}")
    imp = sorted(zip(cols, model.feature_importance("gain")), key=lambda t: -t[1])
    report["importance_top30"] = [(c, float(g)) for c, g in imp[:30]]
    print("top features by gain:", [c for c, _ in imp[:15]])
    model.save_model(str(ROOT / "work/models/forensic_gbm.txt"))
    json.dump({"cols": cols}, open(ROOT / "work/models/forensic_gbm_cols.json", "w"))
    json.dump(report, open(ROOT / "work/forensic_model_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
