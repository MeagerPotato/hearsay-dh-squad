"""Fit the final fusion on the test-matched dev set, apply it to the test set and write the TSV.

  python scripts/make_submission.py --models w2v2_aasist xlsr_sls wavlm_head tak_ft --version v2

Fusion: logistic regression on detector log-odds (real clips weighted 4x, like the false-alarm cost),
regularisation chosen by grouped cross-validated minDCF on dev. Output probabilities are calibrated
(Platt scaling on dev, then shifted to the organizers' stated ~30% synthetic prior); calibration never
changes the ranking, which is all minDCF looks at.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hearsay import submission  # noqa: E402
from hearsay.fusion import cv_mindcf  # noqa: E402
from hearsay.metrics import evaluate  # noqa: E402

TEST_PRIOR = 0.30  # organizers: "approximately 70% of the files are real"


def scores(dataset, model):
    """Scores for a dataset spec; halves ('x@b') fall back to the full-set file when no half file exists."""
    p = ROOT / f"work/scores/{dataset}/{model}.json"
    if not p.exists():
        p = ROOT / f"work/scores/{dataset.partition('@')[0]}/{model}.json"
    d = json.load(open(p))[model]
    return {k[:-4] if k.endswith(".wav") else k: v for k, v in d.items()}


def dev_rows(spec):
    """Rows of a dev spec; 'name@b' keeps the held-out half of groups (generators / voices / speakers)."""
    import zlib
    name, _, half = spec.partition("@")
    rows = list(csv.DictReader(open(ROOT / f"data/proc/{name}.csv")))
    grp = lambda r: r.get("voice") or r.get("speaker") or r.get("attack") or ""  # noqa: E731
    if half:
        rows = [r for r in rows if (zlib.crc32(grp(r).encode()) % 2 == 0) == (half == "a")]
    for r in rows:
        r["_group"] = r.get("speaker") if r["is_spoof"] == "0" else (r.get("attack") or r.get("voice") or name)
        r["_group"] = f"{name}:{r['_group']}"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", default=["la_dev_m", "diffssd_m@b", "say_m@b"],
                    help="VCTK reals + seen attacks (la_dev_m) plus held-out unseen attack families")
    ap.add_argument("--version", required=True)
    ap.add_argument("--cond", action="store_true",
                    help="condition-aware fusion: add a quiet-clip indicator and model-by-quiet interactions (noise floor < -55 dBFS)")
    a = ap.parse_args()
    ids, ys, gs, X, NF = [], [], [], [], []
    for spec in a.dev:
        sc = {m: scores(spec, m) for m in a.models}
        for r in dev_rows(spec):
            if all(r["utt"] in sc[m] for m in a.models):
                ids.append(r["utt"])
                ys.append(int(r["is_spoof"]))
                gs.append(r["_group"])
                X.append([sc[m][r["utt"]] for m in a.models])
                NF.append(float(r.get("nf_after", "nan")))
    y, groups, X = np.array(ys), np.array(gs), np.array(X, float)
    nf_dev = np.array(NF, float)
    print(f"fusion dev: {len(y)} clips ({int((y == 0).sum())} real, {int(y.sum())} synthetic) from {a.dev}")
    mu, sd = X.mean(0), X.std(0) + 1e-9

    def design(Zm, nf):
        if not a.cond:
            return Zm
        quiet = (np.nan_to_num(nf, nan=-45.0) < -55).astype(float)[:, None]
        return np.hstack([Zm, Zm * quiet, quiet])

    Z = design((X - mu) / sd, nf_dev)
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        r, _ = cv_mindcf(Z, y, groups, a.models, C=C)
        print(f"C={C}: CV minDCF {r['minDCF_intended']:.4f} (flipped {r['minDCF_shipped_flipped']:.4f}) EER {100 * r['EER']:.2f}%")
        if best is None or r["minDCF_intended"] < best[1]["minDCF_intended"]:
            best = (C, r)
    C = best[0]
    lr = LogisticRegression(C=C, max_iter=2000).fit(Z, y, sample_weight=np.where(y == 0, 4.0, 1.0))
    dev_fused = lr.decision_function(Z)
    # Platt calibration on dev (unweighted), then prior shift to the test prior
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(dev_fused[:, None], y)
    dev_prior = y.mean()
    shift = np.log(TEST_PRIOR / (1 - TEST_PRIOR)) - np.log(dev_prior / (1 - dev_prior))

    test = {m: scores("test", m) for m in a.models}
    names = sorted(test[a.models[0]])
    Xt = np.array([[test[m][n] for m in a.models] for n in names])
    from hearsay.augment import noise_floor_db
    import soundfile as sf
    nf_test = np.array([noise_floor_db(sf.read(ROOT / "data/HackGTHearsayTesting" / (n if n.endswith(".wav") else n + ".wav"), dtype="float32")[0]) for n in names])
    fused = lr.decision_function(design((Xt - mu) / sd, nf_test))
    logit = platt.coef_[0, 0] * fused + platt.intercept_[0] + shift
    p = 1 / (1 + np.exp(-logit))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    # strictly unique values without changing order (ties can only hurt minDCF)
    order = np.argsort(np.argsort(fused, kind="mergesort"), kind="mergesort")
    p = np.clip(p + order * 1e-12, 1e-6, 1 - 1e-6)
    out = submission.write({n + ".wav" if not n.endswith(".wav") else n: v for n, v in zip(names, p)},
                           ROOT / "data/HackGTHearsayTesting/HGT_Hearsay_score_template.csv",
                           ROOT / f"submissions/{a.version}/{submission.TEAM}_predictions.tsv")
    meta = {"models": a.models, "cond": bool(a.cond), "C": C, "cv_dev": best[1], "weights": dict(zip(a.models, lr.coef_[0][:len(a.models)].round(4).tolist())), "coef_all": lr.coef_[0].round(5).tolist(), "coef_full": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]), "z_mean": mu.tolist(), "z_std": sd.tolist(),
            "platt": [float(platt.coef_[0, 0]), float(platt.intercept_[0])], "prior_shift": float(shift),
            "test_frac_p_gt_0.5": float((p > 0.5).mean()), "dev_in_sample": evaluate(dev_fused, y.astype(bool))}
    json.dump(meta, open(ROOT / f"submissions/{a.version}/fusion_meta.json", "w"), indent=1)
    print(f"wrote {out}\n weights {meta['weights']}\n test share with p>0.5: {meta['test_frac_p_gt_0.5']:.3f}")


if __name__ == "__main__":
    main()
