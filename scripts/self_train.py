"""Transductive self-training of the WavLM head on an unlabelled target set (semi-supervised; no labels used).

Idea: the test set contains attacks our training data never saw. The deep detectors still rank most of
them correctly, so their most confident calls are reliable pseudo-labels. Adding those clips to the
head's training data lets it learn the *test's own* attack signatures, then we rescore everything.

  simulate: python scripts/self_train.py --simulate   (pseudo-test = VCTK dev reals + held-out DiffSSD gens;
                                                       true labels only used to *measure*, never to train)
  apply:    python scripts/self_train.py --apply test --anchor-models w2v2_aasist xlsr_sls
"""
import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hearsay.metrics import evaluate  # noqa: E402

# The WavLM head's own layers (work/models/wavlm_head.npz, written by train_ssl_head.py together with
# selftrain_base.npz), so these scores match what hearsay/pipeline.py computes at inference. Fallback: the
# first head's layers.
_HEAD = ROOT / "work/models/wavlm_head.npz"
LAYERS = [int(v) for v in np.load(_HEAD)["layers"]] if _HEAD.exists() else [6, 7, 8, 9]


def emb(name):
    d = np.load(ROOT / f"work/ssl/{name}.npz")
    F = d["feats"][:, LAYERS, : d["feats"].shape[2] // 2].astype(np.float32).reshape(len(d["names"]), -1)
    return [str(n)[:-4] if str(n).endswith(".wav") else str(n) for n in d["names"]], F


def labels(name):
    return {r["utt"]: r for r in csv.DictReader(open(ROOT / f"data/proc/{name}.csv"))}


def half_mask(ids, rows, half):
    g = lambda r: r.get("voice") or r.get("speaker") or r.get("attack") or ""  # noqa: E731
    return np.array([(zlib.crc32(g(rows[i]).encode()) % 2 == 0) == (half == "a") for i in ids])


def score_map(dataset, model):
    d = json.load(open(ROOT / f"work/scores/{dataset}/{model}.json"))[model]
    return {k[:-4] if k.endswith(".wav") else k: v for k, v in d.items()}


def base_training():
    Xs, ys = [], []
    for spec in ["la_train_m", "libri_m@a", "diffssd_m@a", "say_m@a"]:
        name, _, half = spec.partition("@")
        ids, F = emb(name)
        rows = labels(name)
        keep = half_mask(ids, rows, half) if half else np.ones(len(ids), bool)
        Xs.append(F[keep])
        ys.append(np.array([int(rows[i]["is_spoof"]) for i, k in zip(ids, keep) if k]))
    return np.concatenate(Xs), np.concatenate(ys)


def fit(X, y, w=None, C=0.01):
    sc = StandardScaler().fit(X)
    ww = np.where(y == 0, 4.0, 1.0) * (w if w is not None else 1.0)
    return sc, LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X), y, sample_weight=ww)


def self_train(Xb, yb, Xt, anchor, top=0.12, bottom=0.40, rounds=2, pseudo_weight=1.0):
    """anchor: ensemble score on target clips (higher = more synthetic). Returns head scores on target."""
    sc, lr = fit(Xb, yb)
    head = lr.decision_function(sc.transform(Xt))
    for r in range(rounds):
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)  # noqa: E731
        s = z(anchor) + z(head)
        order = np.argsort(s)
        n = len(s)
        pos = order[-int(top * n):]
        neg = order[: int(bottom * n)]
        X = np.concatenate([Xb, Xt[pos], Xt[neg]])
        y = np.concatenate([yb, np.ones(len(pos), int), np.zeros(len(neg), int)])
        w = np.concatenate([np.ones(len(yb)), np.full(len(pos) + len(neg), pseudo_weight)])
        sc, lr = fit(X, y, w)
        head = lr.decision_function(sc.transform(Xt))
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--apply", default=None)
    ap.add_argument("--anchor-models", nargs="+", default=["w2v2_aasist", "xlsr_sls"])
    ap.add_argument("--top", type=float, default=0.15)
    ap.add_argument("--bottom", type=float, default=0.50)
    a = ap.parse_args()
    Xb, yb = base_training()
    print(f"base training: {len(yb)} clips")
    if a.simulate:
        # pseudo-test: half of the VCTK dev speakers (real) + held-out DiffSSD generators (unseen fakes)
        ids_d, Fd = emb("la_dev_m")
        rd = labels("la_dev_m")
        bona = np.array([rd[i]["is_spoof"] == "0" for i in ids_d]) & half_mask(ids_d, rd, "b")
        ids_s, Fs = emb("diffssd_m")
        rs = labels("diffssd_m")
        spoof = half_mask(ids_s, rs, "b")
        Xt = np.concatenate([Fd[bona], Fs[spoof]])
        yt = np.r_[np.zeros(bona.sum()), np.ones(spoof.sum())].astype(bool)
        tid = [i for i, k in zip(ids_d, bona) if k] + [i for i, k in zip(ids_s, spoof) if k]
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)  # noqa: E731
        maps = {m: {**score_map("la_dev_m", m), **score_map("diffssd_m", m)} for m in a.anchor_models}
        anchor = sum(z(np.array([maps[m][i] for i in tid])) for m in a.anchor_models) / len(a.anchor_models)
        sc, lr = fit(Xb, yb)
        head0 = lr.decision_function(sc.transform(Xt))
        print(f"pseudo-test: {int((~yt).sum())} real, {int(yt.sum())} unseen-generator fakes")
        print("anchor ensemble        :", evaluate(anchor, yt))
        print("head (no self-train)   :", evaluate(head0, yt))
        print("anchor + head fused    :", evaluate(z(anchor) + z(head0), yt))
        for top, bottom in [(0.10, 0.40), (0.15, 0.40), (0.20, 0.50)]:
            head = self_train(Xb, yb, Xt, anchor, top=top, bottom=bottom)
            print(f"self-trained head t{top}/b{bottom}:", evaluate(head, yt), "| fused:", evaluate(z(anchor) + z(head), yt))
    if a.apply:
        # the SAME procedure runs on each target set (dev as a rehearsal, then test), so fusion weights learned
        # on dev transfer to the test-set version of this score
        specs = a.apply.split(",")
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)  # noqa: E731
        ids_all, X_all, anc_all, owner = [], [], [], []
        for spec in specs:
            name, _, half = spec.partition("@")
            ids_t, Xt = emb(name)
            if half:
                keep = half_mask(ids_t, labels(name), half)
                ids_t, Xt = [i for i, k in zip(ids_t, keep) if k], Xt[keep]
            maps = {m: score_map(spec if (ROOT / f"work/scores/{spec}/{m}.json").exists() else name, m) for m in a.anchor_models}
            anc = np.mean([z(np.array([maps[m][i] for i in ids_t])) for m in a.anchor_models], 0)
            ids_all += ids_t
            X_all.append(Xt)
            anc_all.append(anc)
            owner += [spec] * len(ids_t)
        head = self_train(Xb, yb, np.concatenate(X_all), np.concatenate(anc_all), top=a.top, bottom=a.bottom)
        owner = np.array(owner)
        for spec in specs:
            sel = owner == spec
            out = ROOT / f"work/scores/{spec}/wavlm_selftrain.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            json.dump({"wavlm_selftrain": {i + ".wav": float(v) for i, v, k in zip(ids_all, head, sel) if k}}, open(out, "w"))
            print(f"wrote {out}; frac>0 {np.mean(head[sel] > 0):.3f}")

if __name__ == "__main__":
    main()
