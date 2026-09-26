"""End-to-end HEARSAY inference: audio files -> detector scores -> fusion -> submission TSV (+ reports).

    python -m hearsay.cli predict data/HackGTHearsayTesting --out submissions/final.tsv \
        --meta submissions/<version>/fusion_meta.json \
        --template data/HackGTHearsayTesting/HGT_Hearsay_score_template.csv [--reports reports/]

The fusion config drives everything. ``fusion_meta.json`` is written by scripts/make_submission.py, and
only the detectors it lists in ``meta["models"]`` are run. Their scores are fused with the stored dev-set
standardisation, weights, Platt calibration and prior shift, so a new fusion needs no code change.

1. Audio. Every file ffmpeg can read is decoded to 16 kHz mono float32 (hearsay.audio.load). A clip is
   used exactly as it is unless its peak is below 0.5 or above 1.0; only then is it peak-normalised to
   0.998, the level of the organizers' test clips. Already-processed 16 kHz clips are never trimmed or
   resampled, so the test-set scores reproduce the ones we computed during development.
2. Detectors (see DETECTORS). Only one model is in memory at a time, and it is freed before the next one
   loads. Scores are cached per model in ``cache_dir/<model>.json``, in the same layout as
   work/scores/<set>/<model>.json ({model: {filename: spoof log-odds}}) plus fingerprints of every audio
   file and of the model files. A re-run only computes what changed. A plain copy of
   work/scores/test/<model>.json (no fingerprints) is accepted as a trusted seed.
3. Fusion, exactly as scripts/make_submission.py does it: z-score each detector with the dev statistics,
   apply the linear fusion (with the quiet-clip interaction terms when ``meta["cond"]`` is set), then Platt
   scaling and the prior shift. A rank-based 1e-12 tie-break makes every cm-score unique. Outputs are the
   TSV and ``<out>_scores.json``, which holds every detector's log-odds, the fused log-odds and the
   probability for each file.
4. Optional reports. The agentic orchestrator runs on every file. The ensemble appears as one "deep"
   evidence item, and a fusion hook returns the submitted probability, so each HTML report's verdict is
   exactly the value in the TSV.

Device: "auto" means CUDA if available, else Apple MPS, else CPU, and the pipeline runs end to end on
CPU (the Docker image is CPU-only). The ONNX detectors always run on CPU. On a Mac ("auto" = MPS) every
detector reproduces our development scores: the WavLM features are bit-identical, and every score is
within 2e-6. On CPU, the ONNX, fine-tuned XLS-R and forensic scores stay within 1.3e-4. WavLM, however,
runs in fp32 on CPU instead of the fp16 used on MPS for the development features, which moves the
WavLM-head log-odds by up to about 0.006 (8-file check). The transductive self-training amplifies that
(about 0.05 log-odds in simulation), so a CPU run ranks files almost, but not exactly, like the Mac run.
"""
from __future__ import annotations

import datetime as _dt
import gc
import hashlib
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from hearsay import audio as haudio
from hearsay.paths import models_dir

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = models_dir()   # $HEARSAY_MODELS, else <repo>/work/models: where hearsay.deep and hearsay.tak_aasist load from
DEFAULT_CACHE = "work/cache_predict"

PEAK_LOW, PEAK_HIGH, PEAK_TARGET = 0.5, 1.0, 0.998
# Batch-level or slow experts that scripts/extract_forensic.py leaves out of the forensic feature row.
FORENSIC_SKIP = {"transcript", "repeated_script", "speaker_consistency"}
# Transductive self-training of the WavLM head (scripts/self_train.py); meta["self_train"] overrides these.
SELF_TRAIN_DEFAULTS = {"anchor_models": ["w2v2_aasist", "xlsr_sls"], "top": 0.15, "bottom": 0.50,
                       "rounds": 2, "C": 0.01, "real_weight": 4.0, "pseudo_weight": 1.0, "min_files": 50}
WAVLM_BATCH, WAVLM_MAX_SEC = 8, 6.0      # as scripts/extract_ssl.py (padding is per batch of 8, in file order)
ONNX_BATCH, TAK_BATCH, MAX_WINDOWS = 8, 16, 3  # as scripts/score_detector.py and scripts/score_tak_ft.py


@dataclass(frozen=True)
class Detector:
    key: str
    title: str
    what: str                     # one plain-English sentence, shown in the reports
    artifacts: Tuple[str, ...]    # model files (relative to MODEL_DIR) that define its scores


DETECTORS: Dict[str, Detector] = {d.key: d for d in [
    Detector("w2v2_aasist", "XLS-R + AASIST",
             "Tak et al. (2022): the self-supervised XLS-R 300M front end with an AASIST graph-attention "
             "back end, trained on ASVspoof 2019 LA (published weights, run through ONNX)",
             ("w2v2-aasist/w2v2-aasist.onnx",)),
    Detector("xlsr_sls", "XLS-R + SLS",
             "Zhang et al. (2024): XLS-R 300M with a sensitive-layer-selection head that pools evidence from "
             "several transformer layers, trained on ASVspoof 2019 LA (published weights, run through ONNX)",
             ("xlsr-sls/xlsr-sls.onnx",)),
    Detector("tak_ft", "XLS-R + AASIST, fine-tuned",
             "the Tak et al. model fine-tuned by us on test-matched audio (replica of the organizers' "
             "preprocessing plus noise matched to the test set's noise floors)",
             ("tak_ft/best.pt", "xls-r-300m/config.json")),
    Detector("tak_head", "XLS-R + AASIST, back end re-trained",
             "the Tak et al. model with XLS-R frozen and only its AASIST back end re-trained by us on the same "
             "test-matched audio (scripts/finetune_tak.py --train-top 0)",
             ("tak_head/best.pt", "xls-r-300m/config.json")),
    Detector("wavlm_head", "WavLM head",
             "a logistic regression on frozen WavLM-Large middle-layer embeddings, trained on noise-matched "
             "ASVspoof 2019 LA audio plus extra real voices",
             ("wavlm-large/pytorch_model.bin", "wavlm-large/config.json", "wavlm_head.npz")),
    Detector("wavlm_selftrain", "WavLM head, self-trained",
             "the WavLM head re-fitted on this batch's own most confident clips, labelled by the ensemble "
             "(semi-supervised; no ground truth is used)",
             ("wavlm-large/pytorch_model.bin", "wavlm-large/config.json", "selftrain_base.npz")),
    Detector("forensic_gbm", "Forensic model",
             "a LightGBM model over about 200 named forensic measurements (pitch, pauses, spectral dynamics, "
             "reverberation, vocoder traces) produced by our expert modules",
             ("forensic_gbm.txt", "forensic_gbm_cols.json")),
]}
COMPUTE_ORDER = ["w2v2_aasist", "xlsr_sls", "tak_ft", "tak_head", "wavlm_head", "wavlm_selftrain", "forensic_gbm"]
TAK_KEYS = ("tak_ft", "tak_head")   # our XLS-R + AASIST checkpoints: <key>/best.pt, scored like scripts/score_tak_ft.py


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------

def _logger(quiet: bool):
    def say(msg: str) -> None:
        if not quiet:
            print(f"[predict] {msg}", file=sys.stderr, flush=True)
    return say


def _strip_wav(name: str) -> str:
    return name[:-4] if name.endswith(".wav") else name


def default_threads() -> int:
    env = os.environ.get("OMP_NUM_THREADS", "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(8, os.cpu_count() or 1))


def resolve_device(device: str = "auto") -> str:
    """'auto' -> cuda if available, else mps, else cpu. An unavailable explicit choice falls back to cpu."""
    import torch
    device = (device or "auto").lower()
    has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"
    if (device.startswith("cuda") and not torch.cuda.is_available()) or (device == "mps" and not has_mps):
        print(f"[predict] device '{device}' is not available; using cpu", file=sys.stderr, flush=True)
        return "cpu"
    return device


def _release_torch() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def file_fingerprint(path: str) -> str:
    """Size + SHA-1 prefix of the file's bytes (identifies the audio independently of name and mtime)."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"{os.path.getsize(path)}:{h.hexdigest()[:16]}"


def artifact_fingerprint(paths: Sequence[Union[str, Path]], extra: str = "") -> str:
    """Fingerprint of model files (content hash for small files, size + mtime for multi-GB weights)."""
    h = hashlib.sha1(extra.encode())
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"model file not found: {p}")
        st = p.stat()
        h.update(f"|{p.name}:{st.st_size}".encode())
        if st.st_size <= (64 << 20):
            h.update(p.read_bytes())
        else:
            h.update(f":{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def _expert_code_fingerprint() -> str:
    """Fingerprint of the code that produces forensic features (experts + triage), for the feature cache."""
    from hearsay import orchestrator as orch
    h = hashlib.sha1()
    for p in sorted((ROOT / "hearsay" / "experts").glob("*.py")) + [ROOT / "hearsay" / "audio.py",
                                                                    ROOT / "hearsay" / "evidence.py"]:
        h.update(p.read_bytes())
    for fn in (orch.triage, orch._native_info, orch.effective_bandwidth, orch._run_expert, orch._clean_features):
        h.update(inspect.getsource(fn).encode())
    h.update(repr(sorted(FORENSIC_SKIP)).encode())
    return h.hexdigest()[:16]


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------------------------------

def load_audio(path: str) -> Tuple[np.ndarray, float]:
    """16 kHz mono float32 and the gain applied (1.0 = untouched).

    Clips already at the organizers' level are returned exactly as decoded (bit-exact for PCM WAV). Only
    clips whose peak is below 0.5 or above 1.0 are peak-normalised to 0.998, as in hearsay.preprocess.
    """
    x, _ = haudio.load(str(path))
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        raise ValueError("no audio samples could be decoded (corrupt, empty or unsupported file)")
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(x)))
    if peak > 0 and (peak < PEAK_LOW or peak > PEAK_HIGH):
        return (x / peak * PEAK_TARGET).astype(np.float32), PEAK_TARGET / peak
    return x, 1.0


class AudioStore:
    """Decode each file once and keep it in memory up to a byte budget (the test set needs ~0.4 GB)."""

    def __init__(self, budget_bytes: int = int(1.5e9)):
        self._mem: Dict[str, np.ndarray] = {}
        self._used, self.budget = 0, budget_bytes
        self.gain: Dict[str, float] = {}

    def get(self, path: str) -> np.ndarray:
        x = self._mem.get(path)
        if x is None:
            x, self.gain[path] = load_audio(path)
            if self._used + x.nbytes <= self.budget:
                self._mem[path] = x
                self._used += x.nbytes
        return x


# --------------------------------------------------------------------------------------------------
# caches
# --------------------------------------------------------------------------------------------------

class ScoreCache:
    """cache_dir/<model>.json = {model: {filename: score}, "_files": {filename: fingerprint}, "_artifact": ...}.

    An entry is reused when the model files are unchanged ("_artifact") and the audio file has the same
    fingerprint. Entries without a fingerprint (e.g. a plain copy of work/scores/test/<model>.json) are
    trusted by file name.
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]]):
        self.dir = Path(cache_dir) if cache_dir else None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Optional[Path]:
        return self.dir / name if self.dir is not None else None

    def _read(self, model: str) -> Dict[str, Any]:
        p = self.path(f"{model}.json")
        if p is None or not p.exists():
            return {}
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception:
            return {}

    def load(self, model: str, artifact: str, fps: Dict[str, str]) -> Tuple[Dict[str, float], str]:
        d = self._read(model)
        if not d:
            return {}, "no cache"
        if d.get("_artifact") not in (None, artifact):
            return {}, "cache is stale (model files changed)"
        scores, files = d.get(model) or {}, d.get("_files") or {}
        out, seeded = {}, 0
        for name, fp in fps.items():
            if name not in scores or scores[name] is None:
                continue
            saved = files.get(name)
            if saved is None or saved == fp:
                v = float(scores[name])
                if math.isfinite(v):
                    out[name] = v
                    seeded += saved is None
        return out, f"{len(out)} cached" + (f" ({seeded} trusted by name from a seed)" if seeded else "")

    def save(self, model: str, artifact: str, scores: Dict[str, float], fps: Dict[str, str]) -> None:
        p = self.path(f"{model}.json")
        if p is None or not scores:
            return
        d = self._read(model)
        if d.get("_artifact") not in (None, artifact):
            d = {}
        sc, files = d.get(model) or {}, d.get("_files") or {}
        for n, v in scores.items():
            if n in fps:
                sc[n], files[n] = float(v), fps[n]
        d.update({model: sc, "_files": files, "_artifact": artifact})
        _atomic_write(p, json.dumps(d))


# --------------------------------------------------------------------------------------------------
# fusion (identical arithmetic to scripts/make_submission.py)
# --------------------------------------------------------------------------------------------------

ROUNDED_COEF_KEYS = ("coef_all", "weights")


def fusion_coefficients(meta: Dict[str, Any], n_design: int) -> Tuple[np.ndarray, str]:
    """Fusion weights in design-matrix order. Prefers full precision ('coef_full' / 'coef') over the rounded
    'coef_all' (5 decimals) and 'weights' (4 decimals) that make_submission.py records."""
    for key in ("coef_full", "coef", "coef_all"):
        v = meta.get(key)
        if isinstance(v, list) and len(v) == n_design:
            return np.asarray(v, dtype=float), key
    w = meta.get("weights")
    models = list(meta["models"])
    if isinstance(w, dict) and n_design == len(models) and all(m in w for m in models):
        return np.array([w[m] for m in models], dtype=float), "weights"
    raise ValueError(f"fusion meta has no coefficient vector of length {n_design} "
                     f"(keys: {sorted(meta)}); re-run scripts/make_submission.py")


def design_matrix(meta: Dict[str, Any], Z: np.ndarray, noise_floor: Optional[np.ndarray]) -> np.ndarray:
    """make_submission.py's design(): Z, or [Z, Z*quiet, quiet] for the condition-aware fusion."""
    if not meta.get("cond"):
        return Z
    if noise_floor is None:
        raise ValueError("condition-aware fusion (meta['cond']) needs per-file noise floors")
    quiet = (np.nan_to_num(np.asarray(noise_floor, float), nan=-45.0) < -55).astype(float)[:, None]
    return np.hstack([Z, Z * quiet, quiet])


def fuse(meta: Dict[str, Any], X: np.ndarray, noise_floor: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Rows of X = files, columns = meta['models'] (raw spoof log-odds). Missing scores (NaN) get z = 0.

    Returns z-scores, design matrix, fused log-odds, calibrated logit and the final unique probabilities,
    computed with the same operations (and order of operations) as scripts/make_submission.py.
    """
    X = np.asarray(X, dtype=float)
    mu, sd = np.asarray(meta["z_mean"], dtype=float), np.asarray(meta["z_std"], dtype=float)
    Z = (X - mu) / sd
    if not np.all(np.isfinite(Z)):
        Z = np.where(np.isfinite(Z), Z, 0.0)   # an unscorable file counts as the dev-set mean for that model
    D = design_matrix(meta, Z, noise_floor)
    coef, source = fusion_coefficients(meta, D.shape[1])
    fused = (D @ coef[None, :].T + np.array([float(meta["intercept"])])).reshape(-1)  # = LR.decision_function
    a, c = (float(v) for v in meta["platt"])
    logit = a * fused + c + float(meta.get("prior_shift", 0.0))
    p = 1 / (1 + np.exp(-logit))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    # strictly unique values without changing the order (ties can only hurt minDCF)
    order = np.argsort(np.argsort(fused, kind="mergesort"), kind="mergesort")
    p = np.clip(p + order * 1e-12, 1e-6, 1 - 1e-6)
    return {"z": Z, "design": D, "coef": coef, "coef_source": source, "fused": fused, "logit": logit, "p": p}


# --------------------------------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------------------------------

def self_train_head(Xb: np.ndarray, yb: np.ndarray, Xt: np.ndarray, anchor: np.ndarray, top: float = 0.15,
                    bottom: float = 0.50, rounds: int = 2, C: float = 0.01, real_weight: float = 4.0,
                    pseudo_weight: float = 1.0, **_ignored) -> np.ndarray:
    """Transductive self-training; the same procedure as scripts/self_train.py (fit + self_train).

    Fit the head on the base data, then `rounds` times: rank the target clips by z(anchor) + z(head),
    add the top `top` share as pseudo-synthetic and the bottom `bottom` share as pseudo-real, refit.
    Real clips are weighted `real_weight` (4x, like the false-alarm cost). Returns head log-odds on Xt.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fit(X, y, w=None):
        sc = StandardScaler().fit(X)
        ww = np.where(y == 0, real_weight, 1.0) * (w if w is not None else 1.0)
        return sc, LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X), y, sample_weight=ww)

    sc, lr = fit(Xb, yb)
    head = lr.decision_function(sc.transform(Xt))
    for _ in range(rounds):
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)  # noqa: E731
        s = z(anchor) + z(head)
        order = np.argsort(s)
        n = len(s)
        pos = order[-max(1, int(top * n)):]   # max(1, .): order[-0:] would select every clip
        neg = order[: int(bottom * n)]
        X = np.concatenate([Xb, Xt[pos], Xt[neg]])
        y = np.concatenate([yb, np.ones(len(pos), int), np.zeros(len(neg), int)])
        w = np.concatenate([np.ones(len(yb)), np.full(len(pos) + len(neg), pseudo_weight)])
        sc, lr = fit(X, y, w)
        head = lr.decision_function(sc.transform(Xt))
    return head


def _forensic_worker(path: str):
    """One file's forensic feature row, computed like scripts/extract_forensic.py (_features): triage, then
    every registered expert except FORENSIC_SKIP; later experts overwrite duplicate keys."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from hearsay import orchestrator as orch
    from hearsay.evidence import REGISTRY
    orch.ensure_experts_loaded()
    try:
        ctx = orch.triage(path)
    except Exception as e:  # unreadable file
        return path, {}, repr(e)
    if ctx.audio.size:
        peak = float(np.max(np.abs(ctx.audio)))
        if peak > 0 and (peak < PEAK_LOW or peak > PEAK_HIGH):   # same level rule as load_audio
            ctx.audio = (ctx.audio / peak * PEAK_TARGET).astype(np.float32)
    feats: Dict[str, float] = {}
    for name, spec in REGISTRY.items():
        if name in FORENSIC_SKIP:
            continue
        ev = orch._run_expert(spec, ctx)
        feats.update(orch._clean_features(ev.features))
    return path, feats, None


class _Runner:
    """Computes (or loads from cache) every requested detector's scores for one batch of files."""

    def __init__(self, files: List[str], audio: AudioStore, bad: Dict[str, str], cache: ScoreCache,
                 device: str, threads: int, workers: int, st_cfg: Dict[str, Any], say):
        self.files, self.audio, self.bad, self.cache = files, audio, bad, cache
        self.names = [os.path.basename(f) for f in files]
        self.good = [f for f in files if f not in bad]
        self.fps = {os.path.basename(f): file_fingerprint(f) for f in files}
        self.good_fps = {os.path.basename(f): self.fps[os.path.basename(f)] for f in self.good}
        self.device_req, self.device_used = device, None
        self.threads, self.workers, self.st_cfg, self.say = threads, workers, st_cfg, say
        self.results: Dict[str, Dict[str, float]] = {}
        self.timing: Dict[str, Dict[str, Any]] = {}
        self.notes: List[str] = []
        self._feat: Dict[str, np.ndarray] = {}     # WavLM features (float16) by file name

    # ---- plumbing ----------------------------------------------------------------------------------
    def device(self) -> str:
        if self.device_used is None:
            self.device_used = resolve_device(self.device_req)
            self.say(f"torch device: {self.device_used}")
        return self.device_used

    def _artifact(self, key: str) -> str:
        spec = DETECTORS[key]
        paths = [MODEL_DIR / a for a in spec.artifacts]
        if key in ("w2v2_aasist", "xlsr_sls") + TAK_KEYS:
            return artifact_fingerprint(paths, f"{key}|windows={MAX_WINDOWS}")
        if key == "wavlm_head":
            return artifact_fingerprint(paths, f"{key}|sec={WAVLM_MAX_SEC}|bs={WAVLM_BATCH}")
        if key == "forensic_gbm":
            return artifact_fingerprint(paths, key)
        if key == "wavlm_selftrain":
            # transductive: valid only for this exact batch, these anchor scores and these settings
            cfg = {k: v for k, v in self.st_cfg.items()}
            anchors = [[round(self.results[m][n], 9) for n in sorted(self.good_fps)] for m in cfg["anchor_models"]]
            batch = json.dumps([sorted(self.good_fps.items()), anchors, cfg], sort_keys=True)
            return artifact_fingerprint(paths, f"{key}|{hashlib.sha1(batch.encode()).hexdigest()}")
        raise KeyError(key)

    RUNNERS = {"w2v2_aasist": "_run_onnx", "xlsr_sls": "_run_onnx", "tak_ft": "_run_tak_ft",
               "tak_head": "_run_tak_ft", "wavlm_head": "_run_wavlm_head", "wavlm_selftrain": "_run_wavlm_selftrain",
               "forensic_gbm": "_run_forensic_gbm"}

    def score(self, key: str) -> Dict[str, float]:
        if key in self.results:
            return self.results[key]
        if key not in DETECTORS:
            raise KeyError(f"unknown detector '{key}'; known: {', '.join(DETECTORS)}")
        if key == "wavlm_selftrain":
            for m in self.st_cfg["anchor_models"]:   # pseudo-labels come from these detectors' scores
                self.score(m)
        artifact = self._artifact(key)
        have, how = self.cache.load(key, artifact, self.good_fps)
        todo = [f for f in self.good if os.path.basename(f) not in have]
        if key == "wavlm_selftrain" and todo:
            have, todo = {}, list(self.good)   # transductive: always the whole batch at once
        t = {"n_files": len(self.good), "n_cached": len(have), "n_computed": len(todo), "load_s": 0.0,
             "score_s": 0.0}
        self.say(f"{key}: {how}; computing {len(todo)} file(s)")
        if todo:
            new = getattr(self, self.RUNNERS[key])(key, todo, t)
            missing = [os.path.basename(f) for f in todo if os.path.basename(f) not in new]
            if missing:
                self.notes.append(f"{key}: no score for {len(missing)} file(s), e.g. {missing[:3]}")
            self.cache.save(key, artifact, new, self.good_fps)
            have.update(new)
        if t["n_computed"]:
            t["s_per_file"] = round(t["score_s"] / t["n_computed"], 4)
        self.results[key] = {n: have.get(n, float("nan")) for n in self.names}
        self.timing[key] = t
        return self.results[key]

    # ---- ONNX detectors (w2v2_aasist, xlsr_sls) ------------------------------------------------------
    def _run_onnx(self, key: str, files: List[str], t: Dict[str, Any]) -> Dict[str, float]:
        from hearsay import deep
        t0 = time.perf_counter()
        det = deep.build(key, threads=self.threads)
        t["load_s"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        try:
            out = deep.score_files(det, files, batch_size=ONNX_BATCH, max_windows=MAX_WINDOWS, load=self.audio.get,
                                   progress=len(files) > 200)
        finally:
            del det
            gc.collect()
        t["score_s"] = round(time.perf_counter() - t0, 2)
        return out

    # ---- our fine-tuned XLS-R + AASIST checkpoints (tak_ft, tak_head) --------------------------------------
    def _run_tak_ft(self, key: str, files: List[str], t: Dict[str, Any]) -> Dict[str, float]:
        import torch
        from hearsay.deep import windows
        from hearsay.tak_aasist import Model
        device = self.device()
        torch.set_num_threads(self.threads)
        t0 = time.perf_counter()
        ckpt = MODEL_DIR / key / "best.pt"
        net = Model(None, device)   # architecture only (config from work/models/xls-r-300m); weights from ckpt
        try:
            sd = torch.load(str(ckpt), map_location="cpu", mmap=True, weights_only=True)
        except Exception:
            sd = torch.load(str(ckpt), map_location="cpu")
        net.load_state_dict(sd)
        del sd
        net = net.to(device).eval()
        t["load_s"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        out: Dict[str, List[float]] = {}
        buf, owners = [], []

        def flush():
            if not buf:
                return
            o = net(torch.tensor(np.stack(buf)).to(device)).float().cpu().numpy()
            for (n, _k), v in zip(owners, o[:, 0] - o[:, 1]):   # index 1 = bona fide
                out.setdefault(n, []).append(float(v))
            buf.clear()
            owners.clear()

        try:
            with torch.no_grad():
                for i, f in enumerate(files):
                    for k, w in enumerate(windows(self.audio.get(f), max_windows=MAX_WINDOWS)):
                        buf.append(w)
                        owners.append((os.path.basename(f), k))
                        if len(buf) >= TAK_BATCH:
                            flush()
                    if i and i % 200 == 0:
                        self.say(f"  {key}: {i}/{len(files)}")
                flush()
        finally:
            del net
            _release_torch()
        t["score_s"] = round(time.perf_counter() - t0, 2)
        return {n: float(np.mean(v)) for n, v in out.items()}

    # ---- WavLM features, head and self-trained head -----------------------------------------------------
    def _wavlm_layers(self) -> List[int]:
        src = MODEL_DIR / ("wavlm_head.npz" if (MODEL_DIR / "wavlm_head.npz").exists() else "selftrain_base.npz")
        return [int(v) for v in np.load(src)["layers"]]

    def _wavlm_features(self, files: List[str], t: Dict[str, Any]) -> np.ndarray:
        """Per-layer mean embeddings of the head's layers, concatenated and rounded to float16 exactly as
        scripts/extract_ssl.py stored them (the heads were trained and applied on those values)."""
        layers = self._wavlm_layers()
        art = artifact_fingerprint([MODEL_DIR / "wavlm-large/pytorch_model.bin", MODEL_DIR / "wavlm-large/config.json"],
                                   f"layers={layers}|sec={WAVLM_MAX_SEC}|bs={WAVLM_BATCH}")
        path = self.cache.path("wavlm_features.npz")
        if path is not None and path.exists() and not self._feat:
            try:
                d = np.load(path, allow_pickle=False)
                if str(d["artifact"]) == art:
                    for n, fp, v in zip(d["names"], d["fps"], d["feats"]):
                        if self.good_fps.get(str(n)) == str(fp):
                            self._feat[str(n)] = v
            except Exception:
                pass
        todo = [f for f in files if os.path.basename(f) not in self._feat]
        if todo:
            import torch
            from hearsay.ssl_features import SSLEmbedder
            device = self.device()
            t0 = time.perf_counter()
            emb = SSLEmbedder(str(MODEL_DIR / "wavlm-large"), device=device, layers=layers, max_sec=WAVLM_MAX_SEC,
                              threads=self.threads)
            t["load_s"] = round(t["load_s"] + time.perf_counter() - t0, 2)
            t0 = time.perf_counter()
            try:
                for i in range(0, len(todo), WAVLM_BATCH):
                    chunk = todo[i:i + WAVLM_BATCH]
                    f = emb.embed_batch([self.audio.get(p) for p in chunk])      # (B, L, mean+std)
                    means = f[:, :, : f.shape[2] // 2].reshape(len(chunk), -1).astype(np.float16)
                    for p, v in zip(chunk, means):
                        self._feat[os.path.basename(p)] = v
                    if i and i % 400 == 0:
                        self.say(f"  wavlm: {i}/{len(todo)}")
            finally:
                del emb
                _release_torch()
            t["score_s"] = round(t["score_s"] + time.perf_counter() - t0, 2)
            t["wavlm_files_embedded"] = len(todo)
            if path is not None:
                names = sorted(n for n in self._feat if n in self.good_fps)
                tmp = path.with_name("wavlm_features.tmp.npz")
                np.savez(tmp, names=np.array(names), fps=np.array([self.good_fps[n] for n in names]),
                         feats=np.stack([self._feat[n] for n in names]), artifact=np.array(art))
                os.replace(tmp, path)
        return np.stack([self._feat[os.path.basename(f)] for f in files])

    def _run_wavlm_head(self, key: str, files: List[str], t: Dict[str, Any]) -> Dict[str, float]:
        F = self._wavlm_features(files, t)
        t0 = time.perf_counter()
        h = np.load(MODEL_DIR / "wavlm_head.npz")
        X = F.astype(np.float32)
        X -= h["mean"]            # in place, like StandardScaler.transform (stays float32)
        X /= h["scale"]
        s = (X @ h["coef"][None, :].T + h["intercept"]).reshape(-1)   # = LogisticRegression.decision_function
        t["score_s"] = round(t["score_s"] + time.perf_counter() - t0, 2)
        return {os.path.basename(f): float(v) for f, v in zip(files, s)}

    def _run_wavlm_selftrain(self, key: str, files: List[str], t: Dict[str, Any]) -> Dict[str, float]:
        cfg = self.st_cfg
        if len(files) < int(cfg.get("min_files", 50)):
            self.notes.append(f"wavlm_selftrain: only {len(files)} files (< {cfg.get('min_files', 50)}); "
                              "self-training needs a batch, so the un-adapted WavLM head is used instead")
            self.say(self.notes[-1])
            return self._run_wavlm_head(key, files, t)
        base = np.load(MODEL_DIR / "selftrain_base.npz")
        layers = [int(v) for v in base["layers"]]
        if layers != self._wavlm_layers():
            raise ValueError(f"selftrain_base.npz layers {layers} differ from the WavLM head's {self._wavlm_layers()}")
        Xt = self._wavlm_features(files, t).astype(np.float32)
        t0 = time.perf_counter()
        names = [os.path.basename(f) for f in files]
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)  # noqa: E731
        anchor = sum(z(np.array([self.results[m][n] for n in names])) for m in cfg["anchor_models"]) \
            / len(cfg["anchor_models"])
        Xb, yb = base["X"].astype(np.float32), base["y"].astype(int)
        head = self_train_head(Xb, yb, Xt, anchor, **{k: cfg[k] for k in ("top", "bottom", "rounds", "C",
                                                                           "real_weight", "pseudo_weight")})
        t["score_s"] = round(t["score_s"] + time.perf_counter() - t0, 2)
        return {n: float(v) for n, v in zip(names, head)}

    # ---- forensic LightGBM ------------------------------------------------------------------------------
    def _forensic_features(self, files: List[str], t: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        art = _expert_code_fingerprint()
        path = self.cache.path("forensic_features.json")
        rows: Dict[str, Dict[str, float]] = {}
        stored: Dict[str, Any] = {}
        if path is not None and path.exists():
            try:
                stored = json.loads(path.read_text())
                if stored.get("_artifact") != art:
                    stored = {}
            except Exception:
                stored = {}
        for f in files:
            n = os.path.basename(f)
            if (stored.get("_files") or {}).get(n) == self.good_fps.get(n) and n in (stored.get("features") or {}):
                rows[n] = stored["features"][n]
        todo = [f for f in files if os.path.basename(f) not in rows]
        t["forensic_cached"] = len(files) - len(todo)
        if todo:
            t0 = time.perf_counter()
            workers = max(1, min(self.workers, len(todo)))
            errors = {}
            if workers == 1:
                results = map(_forensic_worker, todo)
            else:
                from concurrent.futures import ProcessPoolExecutor
                ex = ProcessPoolExecutor(max_workers=workers)
                results = ex.map(_forensic_worker, todo, chunksize=max(1, min(16, len(todo) // (4 * workers))))
            try:
                for i, (p, feats, err) in enumerate(results, 1):
                    rows[os.path.basename(p)] = feats
                    if err:
                        errors[os.path.basename(p)] = err
                    if i % 200 == 0:
                        self.say(f"  forensic features: {i}/{len(todo)}")
            finally:
                if workers > 1:
                    ex.shutdown()
            if errors:
                self.notes.append(f"forensic features failed for {len(errors)} file(s): {list(errors.items())[:2]}")
            t["score_s"] = round(t["score_s"] + time.perf_counter() - t0, 2)
            if path is not None:
                files_fp = dict((stored.get("_files") or {}))
                feats_all = dict((stored.get("features") or {}))
                for n in rows:
                    files_fp[n], feats_all[n] = self.good_fps[n], rows[n]
                _atomic_write(path, json.dumps({"_artifact": art, "_files": files_fp, "features": feats_all}))
        return rows

    def _run_forensic_gbm(self, key: str, files: List[str], t: Dict[str, Any]) -> Dict[str, float]:
        import lightgbm as lgb
        rows = self._forensic_features(files, t)
        t0 = time.perf_counter()
        cols = json.loads((MODEL_DIR / "forensic_gbm_cols.json").read_text())["cols"]
        names = [os.path.basename(f) for f in files]
        # float32 first, like work/forensic/*.npz, which the model was trained and applied on
        M = np.array([[rows[n].get(c, np.nan) for c in cols] for n in names], np.float32).astype(np.float64)
        booster = lgb.Booster(model_file=str(MODEL_DIR / "forensic_gbm.txt"))
        raw = booster.predict(M, raw_score=True, num_threads=self.threads)
        t["score_s"] = round(t["score_s"] + time.perf_counter() - t0, 2)
        return {n: float(v) for n, v in zip(names, raw)}


# --------------------------------------------------------------------------------------------------
# presenting the ensemble to the agentic orchestrator
# --------------------------------------------------------------------------------------------------

class EnsembleScorer:
    """deep_scorer(ctx) for hearsay.orchestrator: the fused detector ensemble as one "deep" Evidence item.

    Findings list every detector's log-odds, what the detector is, and how much it moved the fused score.
    Evidence.score is the fused log-odds; `threshold`/`scale` map it onto the calibrated log-odds (used by
    the router only if the fusion hook were missing).
    """
    name = "ensemble"

    def __init__(self, table: Dict[str, Dict[str, Any]], meta: Dict[str, Any]):
        self.table, self.meta = table, meta
        a, c = (float(v) for v in meta["platt"])
        shift = float(meta.get("prior_shift", 0.0))
        self.threshold = -(c + shift) / a if a else 0.0
        self.scale = 1.0 / a if a else 1.0

    def __call__(self, ctx):
        from hearsay.evidence import Evidence, Finding
        row = self.table.get(os.path.basename(ctx.path))
        if row is None:
            return None
        findings = [Finding(
            f"Detector ensemble ({len(row['models'])} models, fused by a logistic regression fitted on test-matched "
            f"dev data): fused log-odds {row['fused']:+.2f}; after Platt calibration and the shift to the ~30% "
            f"synthetic prior, P(synthetic) = {row['p']:.3f}. This is the submitted cm-score.",
            1 if row["logit"] > 0.5 else -1 if row["logit"] < -0.5 else 0, float(min(1.0, abs(row["logit"]) / 4.0)))]
        for m in row["models"]:
            d, v, zv, c = DETECTORS[m], row["scores"][m], row["z"][m], row["contrib"][m]
            if not math.isfinite(v):
                findings.append(Finding(f"{d.title} [{m}], {d.what}: no score for this file (it could not be "
                                        "decoded), so the fusion treats it as the dev-set mean.", 0, 0.0))
                continue
            side = "synthetic" if v > 0 else "genuine"
            findings.append(Finding(
                f"{d.title} [{m}], {d.what}: spoof log-odds {v:+.2f} ({side} side of its own decision boundary; "
                f"{zv:+.2f} SD from the fusion dev-set mean). It moves the fused log-odds by {c:+.2f}.",
                1 if c > 0.25 else -1 if c < -0.25 else 0, float(min(1.0, abs(c) / 3.0))))
        if row.get("quiet") is not None:
            q = row["quiet"]
            findings.append(Finding(
                f"Noise floor {row['noise_floor_db']:.1f} dBFS: {'a quiet clip (below -55 dBFS), so' if q else 'not a quiet clip, so'}"
                f" the condition-aware fusion {'applies' if q else 'does not apply'} its quiet-clip terms "
                f"({row['quiet_contrib']:+.2f} log-odds).",
                1 if row["quiet_contrib"] > 0.25 else -1 if row["quiet_contrib"] < -0.25 else 0,
                float(min(1.0, abs(row["quiet_contrib"]) / 3.0))))
        feats = {f"{m}_logodds": row["scores"][m] for m in row["models"]}
        feats.update({"ensemble_fused_logodds": row["fused"], "ensemble_calibrated_logodds": row["logit"],
                      "ensemble_cm_score": row["p"]})
        return Evidence(expert=self.name, category="deep", features=feats, findings=findings, score=row["fused"])


def submitted_probability(features: Dict[str, float], evidence) -> float:
    """Fusion hook for the orchestrator: the probability predict() wrote to the TSV for this file."""
    v = features.get("ensemble_cm_score")
    return float(v) if v is not None else float("nan")


# --------------------------------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------------------------------

def list_inputs(input_dir_or_files) -> List[str]:
    """Audio files in a folder (recursive, sorted, hidden files skipped), one file, or an explicit list."""
    from hearsay import orchestrator as orch
    files = orch.list_audio_files(input_dir_or_files)
    return [f for f in files if not os.path.basename(f).startswith(".")]


def predict(input_dir_or_files, out_tsv, fusion_meta_path, template=None, reports_dir=None,
            self_train: Optional[bool] = None, device: str = "auto", cache_dir=DEFAULT_CACHE,
            threads: Optional[int] = None, workers: Optional[int] = None, transcripts=None,
            skip: Optional[Sequence[str]] = None, asr: bool = False, limit: Optional[int] = None,
            scores_json=None, quiet: bool = False) -> Dict[str, Any]:
    """Score every audio file with the detectors named in the fusion config, fuse, write the TSV.

    self_train: None = as the config says (on when 'wavlm_selftrain' is one of its models). False with a
    config that uses 'wavlm_selftrain' substitutes the un-adapted WavLM head for it (and says so).
    cache_dir=None disables the score cache. Returns paths, probabilities and per-model timing.
    """
    from hearsay import orchestrator as orch
    from hearsay import submission

    t_start = time.perf_counter()
    say = _logger(quiet)
    meta_path = Path(fusion_meta_path)
    meta = json.loads(meta_path.read_text())
    models = list(meta["models"])
    unknown = [m for m in models if m not in DETECTORS]
    if unknown:
        raise KeyError(f"fusion config uses unknown detector(s) {unknown}; known: {', '.join(DETECTORS)}")
    st_cfg = {**SELF_TRAIN_DEFAULTS, **(meta.get("self_train") or {})}
    use_st = ("wavlm_selftrain" in models) if self_train is None else bool(self_train)
    threads = int(threads or default_threads())
    workers = int(workers or max(1, min(4, (os.cpu_count() or 2) // 2)))

    # ---- 1. inputs ---------------------------------------------------------------------------------------
    files = list_inputs(input_dir_or_files)
    if limit:
        files = files[:limit]
    if not files:
        raise ValueError(f"no audio files found in {input_dir_or_files}")
    names = [os.path.basename(f) for f in files]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate file names in the input (the TSV is keyed by name): {dup[:5]}")
    tmpl = submission.read_template(template) if template else None
    if tmpl is not None:
        have = set(names)
        missing = [n for n in tmpl if n not in have]
        if missing:
            raise ValueError(f"{len(missing)} template file(s) are not in the input, e.g. {missing[:3]}")
    say(f"{len(files)} files; fusion config {meta_path} uses {models}"
        + (" (condition-aware)" if meta.get("cond") else ""))
    audio, bad = AudioStore(), {}
    for f in files:
        try:
            audio.get(f)
        except Exception as e:
            bad[f] = f"{type(e).__name__}: {e}"
    if bad:
        say(f"warning: {len(bad)} file(s) could not be decoded and count as the dev-set mean for every detector: "
            f"{[os.path.basename(b) for b in list(bad)[:3]]}")
    normalised = {os.path.basename(f): g for f, g in audio.gain.items() if g != 1.0}
    if normalised:
        say(f"{len(normalised)} file(s) were peak-normalised to {PEAK_TARGET} (peak outside [{PEAK_LOW}, {PEAK_HIGH}])")

    # ---- 2. detectors ------------------------------------------------------------------------------------
    need = set(models)
    if "wavlm_selftrain" in need and not use_st:
        need.discard("wavlm_selftrain")
        need.add("wavlm_head")
    if use_st:
        need |= {"wavlm_selftrain", *st_cfg["anchor_models"]}
    runner = _Runner(files, audio, bad, ScoreCache(cache_dir), device, threads, workers, st_cfg, say)
    for key in [k for k in COMPUTE_ORDER if k in need] + sorted(need - set(COMPUTE_ORDER)):
        runner.score(key)
    cols = {}
    for m in models:
        if m == "wavlm_selftrain" and not use_st:
            runner.notes.append("self-training disabled: the un-adapted WavLM head stands in for 'wavlm_selftrain'")
            cols[m] = runner.results["wavlm_head"]
        else:
            cols[m] = runner.results[m]
    for note in runner.notes:
        say(f"note: {note}")

    # ---- 3. fusion (make_submission.py order: file names sorted without '.wav') --------------------------
    path_of = dict(zip(names, files))
    order = sorted(names, key=_strip_wav)
    X = np.array([[cols[m][n] for m in models] for n in order], dtype=float)
    nf = None
    if meta.get("cond"):
        from hearsay.augment import noise_floor_db
        nf = np.array([noise_floor_db(audio.get(path_of[n])) if path_of[n] not in bad else float("nan")
                       for n in order])
    fz = fuse(meta, X, nf)
    if fz["coef_source"] in ROUNDED_COEF_KEYS:
        say(f"note: the fusion config stores rounded coefficients ('{fz['coef_source']}'); scores match "
            "make_submission.py to ~1e-5, not bit-exactly (store full-precision 'coef_full' to fix)")
    probs = {n: float(v) for n, v in zip(order, fz["p"])}
    k = len(models)
    table: Dict[str, Dict[str, Any]] = {}
    for i, n in enumerate(order):
        contrib = {m: float(fz["coef"][j] * fz["z"][i, j]) for j, m in enumerate(models)}
        row = {"models": models, "scores": {m: float(X[i, j]) for j, m in enumerate(models)},
               "z": {m: float(fz["z"][i, j]) for j, m in enumerate(models)}, "contrib": contrib,
               "fused": float(fz["fused"][i]), "logit": float(fz["logit"][i]), "p": probs[n]}
        if nf is not None:
            q = float(fz["design"][i, -1])
            for j, m in enumerate(models):
                contrib[m] += float(fz["coef"][k + j] * fz["design"][i, k + j])
            row.update({"quiet": bool(q), "noise_floor_db": float(nf[i]),
                        "quiet_contrib": float(fz["coef"][-1] * q) + 0.0})   # + 0.0: no "-0.00" in reports
        table[n] = row

    # ---- outputs -----------------------------------------------------------------------------------------
    out_tsv = Path(out_tsv)
    if tmpl is not None:
        submission.write({n: probs[n] for n in tmpl}, template, out_tsv)
        say(f"wrote {out_tsv} ({len(tmpl)} files, validated against the template)")
    else:
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_tsv, "w", newline="") as fh:
            fh.write("filename\tcm-score\n")
            for n in sorted(names):
                fh.write(f"{n}\t{probs[n]:.12f}\n")
        say(f"wrote {out_tsv} ({len(names)} files)")
    scores_json = Path(scores_json) if scores_json else out_tsv.with_name(out_tsv.stem + "_scores.json")
    doc = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "fusion_meta": str(meta_path), "models": models, "cond": bool(meta.get("cond")),
        "coefficients": fz["coef"].tolist(), "coefficients_source": fz["coef_source"],
        "self_train": use_st, "self_train_config": st_cfg if use_st else None,
        "device": runner.device_used, "threads": threads, "forensic_workers": workers, "n_files": len(names),
        "undecodable": {os.path.basename(f): e for f, e in bad.items()}, "peak_normalised_gain": normalised,
        "notes": runner.notes, "timing": runner.timing, "wall_s": round(time.perf_counter() - t_start, 2),
        "files": {n: {"path": path_of[n],
                      "scores": {m: runner.results[m].get(n) for m in runner.results},
                      **({"noise_floor_db": table[n]["noise_floor_db"]} if nf is not None else {}),
                      "fused_logodds": table[n]["fused"], "calibrated_logodds": table[n]["logit"],
                      "cm_score": probs[n]} for n in sorted(names)},
    }
    scores_json.parent.mkdir(parents=True, exist_ok=True)
    scores_json.write_text(json.dumps(orch._jsonable(doc), indent=1))
    say(f"wrote {scores_json}; share with cm-score > 0.5: {np.mean(fz['p'] > 0.5):.3f}")
    result = {"tsv": str(out_tsv), "scores_json": str(scores_json), "probabilities": probs, "table": table,
              "timing": runner.timing, "notes": runner.notes, "coef_source": fz["coef_source"],
              "device": runner.device_used, "reports_index": None}

    # ---- 4. optional agentic reports ---------------------------------------------------------------------
    if reports_dir:
        result.update(write_reports(files, table, meta, Path(reports_dir), transcripts=transcripts, skip=skip,
                                    asr=asr, quiet=quiet, extra={"fusion_meta": str(meta_path), "models": models,
                                                                 "submission_tsv": str(out_tsv)}))
    result["wall_s"] = round(time.perf_counter() - t_start, 2)
    return result


def write_reports(files: List[str], table: Dict[str, Dict[str, Any]], meta: Dict[str, Any], reports_dir: Path,
                  transcripts=None, skip: Optional[Sequence[str]] = None, asr: bool = False, quiet: bool = False,
                  extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the agentic orchestrator per file with the ensemble as its deep scorer and write HTML reports."""
    from hearsay import orchestrator as orch
    from hearsay.report import make_report_writer, render_index
    say = _logger(quiet)
    if transcripts:
        trans = orch.load_transcripts(str(transcripts))
    else:
        try:
            from hearsay.experts.semantic import TRANSCRIPTS_JSON   # cached Whisper transcripts, if shipped
            trans = orch.load_transcripts(str(TRANSCRIPTS_JSON)) if Path(TRANSCRIPTS_JSON).exists() else {}
        except Exception:
            trans = {}
    disabled = set(skip or ())
    lacking = [f for f in files if os.path.basename(f) not in trans]
    if lacking and not asr:
        disabled.add("transcript")
        say(f"reports: {len(lacking)} file(s) have no cached transcript, so the Whisper transcript expert is "
            "off (pass --asr to allow it)")
    cfg = orch.RouterConfig(disabled_experts=tuple(sorted(disabled)))
    reports_dir.mkdir(parents=True, exist_ok=True)
    results = orch.run_batch(files, deep_scorer=EnsembleScorer(table, meta), fusion=submitted_probability,
                             mode="agentic", config=cfg, transcripts=trans,
                             on_result=make_report_writer(str(reports_dir), cfg), progress=not quiet)
    mismatched = [r.name for r in results if table.get(r.name) is None or r.cm_score != table[r.name]["p"]]
    acc = orch.compute_accounting(results)
    orch.export_json(results, str(reports_dir / "evidence.json"), extra=extra)
    index = render_index(results, str(reports_dir), acc, config=cfg,
                         title="HEARSAY forensic dashboard (submitted detector ensemble)")
    say(f"wrote {index} and {len(results)} file reports; verdicts equal to the TSV: "
        f"{len(results) - len(mismatched)}/{len(results)}")
    return {"reports_index": index, "reports_mismatched": mismatched, "accounting": acc}
