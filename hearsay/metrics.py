"""minDCF / EER exactly as the organizers' modified ASVspoof5 Track-1 scorer computes them.

The organizers evaluate with the ASVspoof 5 Track-1 package, with Pspoof = 0.3 and Cfa = 4 in
calculate_metrics.py (upstream defaults: 0.05 and 10). They confirmed 0.3 on Discord on 2026-09-26; the copy
of calculate_metrics.py they shipped to teams still said 0.5. HEARSAY_PSPOOF overrides it for comparisons.

Their code treats the *target* class as the one with higher scores, while our submission convention is
1.0 = synthetic, so we report both readings of the scorer:

  * "intended": spoof is the target class and our raw score is used. The Cfa-weighted error is a real clip
    flagged as synthetic (the slides' definition of a false alarm). Normalised, the cost is
    P(miss) + 1.71 * P(false alarm) at Pspoof = 0.3.
  * "shipped_flipped": bona fide is the target class and 1 - score is used. The Cfa-weighted error is then a
    missed fake: P(false alarm) + 1.71 * P(miss).

"mean" averages the two, which is what we optimise while the organizers' score direction is unconfirmed.
"""
import os

import numpy as np

PSPOOF = float(os.environ.get("HEARSAY_PSPOOF", "0.3"))
CMISS, CFA = 1.0, 4.0


def _det_curve(target, nontarget):
    # Mirrors calculate_modules.compute_det_curve (stable mergesort, target concatenated first).
    n = target.size + nontarget.size
    scores = np.concatenate((target, nontarget))
    labels = np.concatenate((np.ones(target.size), np.zeros(nontarget.size)))
    idx = np.argsort(scores, kind="mergesort")
    labels = labels[idx]
    tar_sums = np.cumsum(labels)
    non_sums = nontarget.size - (np.arange(1, n + 1) - tar_sums)
    frr = np.concatenate(([0.0], tar_sums / target.size))
    far = np.concatenate(([1.0], non_sums / nontarget.size))
    return frr, far


def _mindcf_eer(target, nontarget, pspoof):
    frr, far = _det_curve(np.asarray(target, float), np.asarray(nontarget, float))
    p_target = 1 - pspoof
    cdet = CMISS * frr * p_target + CFA * far * (1 - p_target)
    mindcf = cdet.min() / min(CMISS * p_target, CFA * (1 - p_target))
    i = np.argmin(np.abs(frr - far))
    return float(mindcf), float((frr[i] + far[i]) / 2)


def evaluate(p_synth, is_spoof, pspoof=None):
    """p_synth: higher = more synthetic. is_spoof: bool array. Returns dict of metrics."""
    pspoof = PSPOOF if pspoof is None else pspoof
    s = np.asarray(p_synth, float)
    y = np.asarray(is_spoof, bool)
    md_int, eer = _mindcf_eer(s[y], s[~y], pspoof)
    md_flip, _ = _mindcf_eer(-s[~y], -s[y], pspoof)
    return {"minDCF_intended": md_int, "minDCF_shipped_flipped": md_flip, "minDCF_mean": (md_int + md_flip) / 2,
            "EER": eer, "n_spoof": int(y.sum()), "n_bona": int((~y).sum())}
