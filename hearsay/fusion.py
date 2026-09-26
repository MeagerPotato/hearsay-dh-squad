"""Score fusion tuned for the challenge metric.

minDCF depends only on how files are ranked, and a false alarm (real flagged as synthetic) costs 4x a
miss. So we (1) put every detector on a common scale, (2) fit a prior-weighted logistic regression on a
labelled dev set that mimics the test pipeline, and (3) choose between candidate fusions by minDCF on
held-out folds, not by accuracy or EER.

Inputs are spoof log-odds from hearsay.deep plus optional interpretable features from the experts.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .metrics import evaluate


def rank_normalise(v):
    """Map scores to (0,1) by rank. Robust to per-model scale and domain shift."""
    v = np.asarray(v, float)
    r = np.argsort(np.argsort(v, kind="mergesort"), kind="mergesort")
    return (r + 0.5) / len(v)


@dataclass
class Standardiser:
    """z-score each column with statistics from the *target* set (transductive test-time normalisation)."""
    mean: np.ndarray = None
    std: np.ndarray = None

    def fit(self, X):
        self.mean = np.nanmean(X, 0)
        self.std = np.nanstd(X, 0) + 1e-9
        return self

    def transform(self, X):
        Z = (X - self.mean) / self.std
        return np.nan_to_num(Z, nan=0.0, posinf=6.0, neginf=-6.0).clip(-6, 6)


@dataclass
class Fusion:
    columns: list
    C: float = 0.1
    fa_weight: float = 4.0      # weight real clips 4x so the fit prefers not to flag them
    model: LogisticRegression = None
    coef_: dict = field(default_factory=dict)

    def fit(self, X, y):
        w = np.where(y == 0, self.fa_weight, 1.0)
        self.model = LogisticRegression(C=self.C, max_iter=2000, class_weight=None)
        self.model.fit(X, y, sample_weight=w)
        self.coef_ = dict(zip(self.columns, self.model.coef_[0].round(4).tolist()))
        return self

    def decision(self, X):
        return self.model.decision_function(X)

    def save(self, path, standardiser=None):
        json.dump({"columns": self.columns, "coef": self.model.coef_[0].tolist(),
                   "intercept": float(self.model.intercept_[0]), "C": self.C, "fa_weight": self.fa_weight},
                  open(path, "w"), indent=1)


def cv_mindcf(X, y, groups, columns, C=0.1, fa_weight=4.0, folds=5):
    """Grouped cross-validated minDCF (group = speaker or attack) for one fusion configuration."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        f = Fusion(columns, C=C, fa_weight=fa_weight).fit(X[tr], y[tr])
        oof[te] = f.decision(X[te])
    return evaluate(oof, y.astype(bool)), oof


def load_scores(paths):
    """Merge {model: {basename: score}} JSON files into one dict."""
    out = {}
    for p in paths:
        out.update(json.load(open(p)))
    return out


def matrix(score_dict, names, columns):
    return np.array([[score_dict[c].get(n, np.nan) for c in columns] for n in names], float)
