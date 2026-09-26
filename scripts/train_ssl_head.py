"""Train a light spoof classifier on frozen WavLM-Large layer embeddings.

1. Layer probe: one logistic regression per layer (mean+std pooled), scored by minDCF on held-out
   data. This shows where in the network the spoofing evidence lives (reported in docs).
2. Head: logistic regression on the concatenation of the best K layers (standardised, L2-regularised,
   real clips weighted 4x to mirror the false-alarm cost).

  python scripts/train_ssl_head.py --train la_train_m say_m --eval la_dev_m diffssd_m lj_m --apply test
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hearsay.metrics import evaluate  # noqa: E402


def _group(r):
    return r.get("voice") or r.get("speaker") or r.get("attack") or ""


def dataset(spec):
    """spec = 'name' or 'name@a' / 'name@b': a deterministic half of the *groups* (generator, TTS voice or
    speaker), so held-out halves contain unseen generators/voices/speakers, not just unseen clips."""
    name, _, half = spec.partition("@")
    d = np.load(str(ROOT / f"work/ssl/{name}.npz"))
    ids = list(d["names"])
    F = d["feats"][:, :, : d["feats"].shape[2] // 2]  # keep only per-layer means, still float16 (memory-safe)
    if name == "test":
        return ids, F, None, None
    rows = {r["utt"]: r for r in csv.DictReader(open(ROOT / f"data/proc/{name}.csv"))}
    if half:
        import zlib
        groups = sorted({_group(rows[i]) for i in ids})
        in_a = {g for g in groups if zlib.crc32(g.encode()) % 2 == 0}
        keep = np.array([(_group(rows[i]) in in_a) == (half == "a") for i in ids])
        ids, F = [i for i, k in zip(ids, keep) if k], F[keep]
    y = np.array([int(rows[i]["is_spoof"]) for i in ids])
    src = np.array([rows[i].get("attack", "bonafide") if y_ else "bonafide" for i, y_ in zip(ids, y)])
    return ids, F, y, src


def stack(names):
    parts = [dataset(n) for n in names]
    return (sum([p[0] for p in parts], []), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]))


def select(F, layers):
    """F holds per-layer mean embeddings (float16); return the chosen layers as a float32 matrix."""
    return F[:, layers, :].astype(np.float32).reshape(len(F), -1)


def fit(X, y, C):
    sc = StandardScaler().fit(X)
    w = np.where(y == 0, 4.0, 1.0)
    lr = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X), y, sample_weight=w)
    return sc, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--apply", nargs="*", default=[])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--layers", default=None, help="fixed layer range like 3-12 (skips the probe-based choice)")
    a = ap.parse_args()
    _, Ftr, ytr, _ = stack(a.train)
    ev = {n: dataset(n) for n in a.eval}
    # combined eval: every eval set pooled (bona from all, spoof from all)
    Fev = np.concatenate([ev[n][1] for n in a.eval])
    yev = np.concatenate([ev[n][2] for n in a.eval])
    L = Ftr.shape[1]
    probe = []
    for layer in (range(L) if not a.layers else []):
        sc, lr = fit(select(Ftr, [layer]), ytr, a.C)
        s = lr.decision_function(sc.transform(select(Fev, [layer])))
        probe.append(evaluate(s, yev.astype(bool))["minDCF_intended"])
        print(f"layer {layer:2d}: pooled-eval minDCF {probe[-1]:.4f}", flush=True)
    best = sorted(np.argsort(probe)[: a.k].tolist())
    if a.layers:
        lo, hi = map(int, a.layers.split("-"))
        best = list(range(lo, hi + 1))
    print("best layers:", best)
    sc, lr = fit(select(Ftr, best), ytr, a.C)
    report = {"layers": best, "probe_minDCF": probe}
    for n in a.eval:
        ids, F, y, src = ev[n]
        s = lr.decision_function(sc.transform(select(F, best)))
        r = evaluate(s, y.astype(bool)) if len(set(y)) > 1 else {"frac_flagged_at_0": float((s > 0).mean())}
        print(n, r)
        report[n] = r
        (ROOT / f"work/scores/{n}").mkdir(parents=True, exist_ok=True)
        json.dump({"wavlm_head": dict(zip(ids, s.astype(float).tolist()))}, open(ROOT / f"work/scores/{n}/wavlm_head.json", "w"))
    s = lr.decision_function(sc.transform(select(Fev, best)))
    report["pooled"] = evaluate(s, yev.astype(bool))
    print("pooled eval:", report["pooled"])
    for n in a.apply:
        ids, F, _, _ = dataset(n)
        s = lr.decision_function(sc.transform(select(F, best)))
        out = ROOT / f"work/scores/{n}/wavlm_head.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"wavlm_head": dict(zip(ids, s.astype(float).tolist()))}, open(out, "w"))
        print(f"applied to {n}: frac>0 {np.mean(s > 0):.3f}")
    json.dump(report, open(ROOT / "work/wavlm_head_report.json", "w"), indent=1)
    # save the head so inference (and the Docker image) can reproduce these scores exactly
    np.savez(ROOT / "work/models/wavlm_head.npz", layers=np.array(best), mean=sc.mean_, scale=sc.scale_,
             coef=lr.coef_[0], intercept=np.array([lr.intercept_[0]]))
    # base training set (selected layers, float16) for transductive self-training at inference time
    Xb = select(Ftr, best).astype(np.float16)
    np.savez_compressed(ROOT / "work/models/selftrain_base.npz", X=Xb, y=ytr.astype(np.int8), layers=np.array(best))
    print(f"saved head + self-training base ({Xb.shape}, {Xb.nbytes / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
