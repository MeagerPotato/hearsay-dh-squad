"""Agentic orchestration: triage a file, decide which forensic experts to run, score, attribute, explain.

Instead of brute-forcing every detector on every file, the orchestrator works like an analyst:

  1. TRIAGE     probe the container (codec, sample rate, lossy?, extension mismatch), decode, and take
                a quick look at the signal (level, clipping, silence, effective bandwidth).
  2. CHEAP      always run the cheap tripwires (container, codec cut-off, spectral stats, splices,
                ENF, noise floor, repeated script).
  3. DEEP       run the injected deep-learning detector, if the caller supplied one.
  4. CONDITIONAL add compression-history analysis only when the file is lossy, band-limited or
                its extension lies about its contents.
  5. MEDIUM     escalate to prosody / voice-quality / speaker / vocoder / replay experts only when the
                provisional verdict is uncertain, a cheap expert raised a strong flag, cheap experts
                contradict each other, or the clip is long enough to hide a local edit.
  6. EXPENSIVE  if the verdict is *still* uncertain, run the expensive experts (transcript).

The rules live in the declarative POLICY below; every rule evaluation (fired or not, and why) is
appended to a routing trace that the HTML report renders. `mode="full"` fires every rule, which is
what the compute-accounting comparison ("agentic mode runs X% of the compute of full mode") uses.

Scoring: cm_score = fusion(features, evidence) when a learned fusion is supplied; otherwise the
sigmoid of the deep detector's calibrated score; otherwise a transparent weighted vote of the expert
findings (see `vote_logodds`). The manipulation-type label is a documented rule list
(`ATTRIBUTION_RULES`) over the evidence, not a classifier.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from hearsay import audio as haudio
from hearsay.evidence import CATEGORIES, REGISTRY, Context, Evidence, ExpertSpec, Finding

ROUTER_VERSION = "1.0"

DeepScorer = Callable[[Context], Optional[Evidence]]
Fusion = Callable[[Dict[str, float], List[Evidence]], float]

AUDIO_EXTENSIONS = {".wav", ".wave", ".flac", ".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".oga", ".opus",
                    ".webm", ".mkv", ".aif", ".aiff", ".aifc", ".wma", ".amr", ".caf", ".3gp"}

# Codecs whose decoded samples are not bit-identical to the input (perceptual or ADPCM-style coding).
LOSSY_CODEC_PREFIXES = ("mp3", "mp2", "mp1", "aac", "vorbis", "opus", "wmav", "wmapro", "wmavoice", "amr",
                        "ac3", "eac3", "dts", "gsm", "g723", "g729", "adpcm", "speex", "qcelp", "evrc",
                        "atrac", "cook", "ra_", "nellymoser", "ilbc", "sbc", "aptx",
                        "pcm_mulaw", "pcm_alaw")   # 8-bit companded telephony PCM loses precision too
LOSSLESS_CODEC_PREFIXES = ("pcm_", "flac", "alac", "wavpack", "ape", "tta", "mlp", "truehd", "shorten")
# Which ffprobe format names are consistent with each file extension (anything else is a mismatch).
EXTENSION_FORMATS = {
    ".wav": {"wav", "w64", "rf64"}, ".wave": {"wav", "w64", "rf64"}, ".flac": {"flac"}, ".mp3": {"mp3"},
    ".m4a": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}, ".mp4": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
    ".3gp": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}, ".aac": {"aac"}, ".ogg": {"ogg"}, ".oga": {"ogg"},
    ".opus": {"ogg"}, ".webm": {"matroska", "webm"}, ".mkv": {"matroska", "webm"}, ".aif": {"aiff"},
    ".aiff": {"aiff"}, ".aifc": {"aiff"}, ".wma": {"asf"}, ".amr": {"amr"}, ".caf": {"caf"},
}


# --------------------------------------------------------------------------------------------------
# Configuration and the declarative routing policy
# --------------------------------------------------------------------------------------------------

@dataclass
class RouterConfig:
    """Tunable thresholds of the router, the fallback vote and the attribution rules."""
    deep_threshold: float = 0.0        # decision threshold on the deep score (log-odds); scorer may override
    deep_scale: float = 1.0            # z = (deep score - threshold) / scale; scorer may override
    uncertainty_band: float = 1.0      # |provisional z| below this => "uncertain" => escalate
    long_clip_s: float = 6.0           # clips longer than this can hide a local edit => escalate
    strong_flag: float = 0.6           # a finding at least this strong counts as a "strong flag"
    band_limited_ratio: float = 0.85   # effective bandwidth below ratio * Nyquist => band-limited
    silence_dbfs: float = -60.0        # RMS below this => no usable signal
    fusion_threshold: float = 0.5      # fusion probability treated as the decision threshold
    vote_gain: float = 2.0             # log-odds contributed by one full-strength finding (weight 1.0)
    vote_clip: float = 6.0             # cap on the fallback vote's |log-odds|
    vote_weight_with_deep: float = 0.5 # how much expert findings nudge the *provisional* deep score
    disabled_experts: Tuple[str, ...] = ()  # never run these (e.g. "transcript" on a laptop), both modes
    synthetic_above: float = 0.65      # cm_score >= this => synthetic-type labels
    bona_fide_below: float = 0.35      # cm_score <= this (and no strong synthetic flag) => "bona fide"
    # Hand-set (not learned) reliability of each evidence family for the fallback vote. Compression and
    # container evidence describe the processing chain, not the talker, so they count for less.
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "deep": 1.0, "spectral": 1.0, "splice": 0.8, "prosody": 0.8, "speaker": 0.8,
        "environment": 0.7, "compression": 0.4, "container": 0.4, "semantic": 0.5,
    })


# Each rule: `when` is a list of predicate names combined with OR, `requires` a list combined with AND.
# `experts` are explicit expert names (a name claimed by a rule runs ONLY when that rule fires);
# `cost` additionally sweeps every registered, unclaimed expert of that cost tier, so experts written
# after this policy are still routed sensibly. `deep: True` calls the injected deep scorer.
POLICY: List[Dict[str, Any]] = [
    {"id": "cheap_baseline", "tier": "cheap", "when": ["always"], "requires": [],
     "experts": ["container", "codec_cutoff", "spectral_stats", "discontinuity", "enf", "noise_floor",
                 "pauses_breaths", "repeated_script"], "cost": "cheap",
     "why": "Cheap container and signal tripwires run on every file; their flags steer escalation."},
    {"id": "deep_detector", "tier": "deep", "when": ["deep_scorer_available"], "requires": [], "deep": True,
     "why": "The deep anti-spoofing detector gives the primary provisional score."},
    {"id": "compression_history", "tier": "conditional", "when": ["lossy_or_band_limited"], "requires": [],
     "experts": ["double_compression"],
     "why": "Re-encoding history only matters when the file is lossy, band-limited or mislabelled."},
    {"id": "escalate_medium", "tier": "medium", "when": ["uncertain", "strong_flag", "conflict", "long_clip"],
     "requires": ["has_signal"],
     "experts": ["pitch", "voice_quality", "speaker_consistency", "vocoder_artifacts", "reverb_replay"],
     "cost": "medium",
     "why": "Slower talker-level analyses are spent only where the cheap evidence is not decisive."},
    {"id": "escalate_expensive", "tier": "expensive", "when": ["uncertain"], "requires": ["has_signal"],
     "experts": ["transcript"], "cost": "expensive",
     "why": "The most expensive analyses run only if the verdict is still uncertain after escalation."},
]

# Experts that are meaningless for some inputs are suppressed (and the suppression is logged).
SUPPRESS_RULES: List[Dict[str, Any]] = [
    {"expert": "vocoder_artifacts", "when": "narrowband",
     "why": "vocoder traces live above ~4 kHz and this recording has no energy there"},
    {"expert": "speaker_consistency", "when": "very_short",
     "why": "fewer than two 1-second windows, so there is nothing to compare"},
    {"expert": "enf", "when": "very_short",
     "why": "too short to resolve a 50/60 Hz mains-hum frequency track"},
]

TIER_ORDER = ["triage", "cheap", "deep", "conditional", "medium", "expensive"]


# --------------------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------------------

@dataclass
class FileResult:
    path: str
    cm_score: float                                   # 0..1, 1.0 = synthetic
    evidence: List[Evidence] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    manipulation_type: str = "uncertain"
    summary: str = ""
    score_source: str = ""                            # "fusion" | "deep" | "vote" | "none"
    mode: str = "agentic"
    native: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    tiers: List[str] = field(default_factory=list)    # tiers that actually ran something
    attribution_reasons: List[str] = field(default_factory=list)
    provisional: Optional[float] = None               # last provisional z used for routing
    runtime_ms: float = 0.0
    error: Optional[str] = None
    ctx: Optional[Context] = field(default=None, repr=False, compare=False)  # dropped in batch runs

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict (artifacts and audio are omitted; artifact keys are listed)."""
        return {
            "path": self.path, "name": self.name, "cm_score": _jsonable(self.cm_score),
            "manipulation_type": self.manipulation_type, "summary": self.summary,
            "score_source": self.score_source, "mode": self.mode, "tiers": self.tiers,
            "attribution_reasons": self.attribution_reasons, "provisional": _jsonable(self.provisional),
            "runtime_ms": round(self.runtime_ms, 2), "error": self.error, "native": _jsonable(self.native),
            "features": _jsonable(self.features), "trace": _jsonable(self.trace),
            "evidence": [evidence_to_dict(e) for e in self.evidence],
        }


def evidence_to_dict(ev: Evidence) -> Dict[str, Any]:
    return {
        "expert": ev.expert, "category": ev.category, "score": _jsonable(ev.score),
        "applicable": ev.applicable, "runtime_ms": round(float(ev.runtime_ms or 0.0), 2), "error": ev.error,
        "partial_errors": list((ev.artifacts or {}).get("partial_errors") or []),
        "features": _jsonable(ev.features),
        "findings": [{"text": f.text, "direction": int(f.direction), "strength": _jsonable(f.strength)}
                     for f in ev.findings],
        "artifact_keys": sorted(ev.artifacts.keys()) if ev.artifacts else [],
    }


def _jsonable(x):
    """Recursively convert numpy scalars/arrays and NaN/inf into JSON-friendly values."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist()) if x.size <= 64 else f"<array shape={x.shape}>"
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


# --------------------------------------------------------------------------------------------------
# Registry helpers
# --------------------------------------------------------------------------------------------------

_EXPERTS_LOADED = False


def ensure_experts_loaded() -> Dict[str, str]:
    """Import hearsay.experts once (registering every available expert); return its import errors."""
    global _EXPERTS_LOADED
    try:
        import hearsay.experts as _ex  # noqa: F401  (import side effect: registration)
        _EXPERTS_LOADED = True
        return dict(getattr(_ex, "IMPORT_ERRORS", {}))
    except Exception as e:  # pragma: no cover - defensive
        return {"hearsay.experts": repr(e)}


def _claimed_names(policy: Sequence[Dict[str, Any]]) -> set:
    return {n for rule in policy for n in rule.get("experts", [])}


def _select_experts(rule: Dict[str, Any], registry: Dict[str, ExpertSpec], claimed: set,
                    already: set) -> Tuple[List[str], List[str]]:
    """Resolve a rule to (registered expert names to run, explicitly named experts that are missing)."""
    names: List[str] = []
    missing: List[str] = []
    for n in rule.get("experts", []):
        if n in registry:
            names.append(n)
        else:
            missing.append(n)
    cost = rule.get("cost")
    if cost:
        for n, spec in registry.items():
            spec_cost = spec.cost if spec.cost in ("cheap", "medium", "expensive") else "medium"
            if spec_cost == cost and n not in claimed and n not in names:
                names.append(n)
    return [n for n in names if n not in already], missing


def _run_expert(spec: ExpertSpec, ctx: Context) -> Evidence:
    """Run one registered expert; guarantees an Evidence object even for buggy experts."""
    t = time.perf_counter()
    try:
        ev = spec.fn(ctx)
    except Exception as e:  # the @expert wrapper already catches, this is belt and braces
        ev = Evidence(expert=spec.name, category=spec.category, applicable=False, error=f"{type(e).__name__}: {e}")
    if not isinstance(ev, Evidence):
        ev = Evidence(expert=spec.name, category=spec.category, applicable=False,
                      error=f"expert returned {type(ev).__name__}, not Evidence")
    ev.expert = ev.expert or spec.name
    if ev.category not in CATEGORIES:
        ev.category = spec.category
    if not ev.runtime_ms:
        ev.runtime_ms = (time.perf_counter() - t) * 1000
    if ev.error and ev.applicable:
        # Partial failure: a sub-analysis failed but the remaining features/findings are valid. `error` is
        # reserved for a failed expert (whose evidence is excluded from the vote, attribution and report), so
        # keep this evidence and record the problem as a warning instead.
        ev.artifacts = dict(ev.artifacts or {})
        ev.artifacts.setdefault("partial_errors", []).append(ev.error)
        ev.error = None
    ev.features = _clean_features(ev.features)
    return ev


def _clean_features(feats: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k, v in (feats or {}).items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------------------------------
# Triage
# --------------------------------------------------------------------------------------------------

def effective_bandwidth(x: np.ndarray, sr: int, drop_db: float = 20.0) -> float:
    """Highest frequency (Hz) above which the long-term spectrum sits >= drop_db below the band just under it.

    A codec low-pass or resampler shows up as a cliff in the average spectrum; natural speech decays
    gradually. Returns Nyquist when no cliff is found and NaN for empty/silent input.
    """
    from scipy.signal import welch

    x = np.asarray(x, np.float64)
    if x.size < 512 or not np.any(x):
        return float("nan")
    f, p = welch(x, fs=sr, nperseg=min(2048, x.size))
    s = 10 * np.log10(p + 1e-20)
    df = f[1] - f[0]
    nyq = sr / 2.0
    win = max(2, int(round(500 / df)))               # 500 Hz comparison bands
    step = max(1, int(round(50 / df)))
    top = int(np.searchsorted(f, nyq - 150.0))       # keep >= 150 Hz of "above" band
    bottom = int(np.searchsorted(f, 1500.0))
    for i in range(top, bottom, -step):               # scan downwards: first hit is the highest cliff
        below = np.median(s[max(0, i - win):i])
        near = np.median(s[i:i + win])                # the band right above the candidate edge ...
        rest = np.percentile(s[i:], 95)               # ... and everything above it must stay low
        if below - near >= drop_db and below - rest >= drop_db - 6.0:
            return float(f[i])
    return float(nyq)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _native_info(path: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten ffprobe output into the chain-of-custody facts the router and report use."""
    fmt = info.get("format", {}) or {}
    streams = info.get("streams", []) or []
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})
    ext = Path(path).suffix.lower()
    fmt_names = set((fmt.get("format_name") or "").split(",")) - {""}
    codec = (a.get("codec_name") or "").lower()
    lossy = bool(codec) and codec.startswith(LOSSY_CODEC_PREFIXES)
    lossy = lossy or (bool(codec) and not codec.startswith(LOSSLESS_CODEC_PREFIXES))  # unknown codec: assume lossy
    allowed = EXTENSION_FORMATS.get(ext)
    tags = {}
    tags.update({f"format.{k}": v for k, v in (fmt.get("tags") or {}).items()})
    tags.update({f"stream.{k}": v for k, v in (a.get("tags") or {}).items()})

    def _num(v, typ=float):
        try:
            return typ(v)
        except (TypeError, ValueError):
            return None

    st = os.stat(path)
    return {
        "filename": os.path.basename(path),
        "extension": ext,
        "size_bytes": st.st_size,
        "modified_utc": _dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc).isoformat(timespec="seconds"),
        "sha256": _sha256(path),
        "container": fmt.get("format_name"),
        "container_long": fmt.get("format_long_name"),
        "codec": codec or None,
        "codec_long": a.get("codec_long_name"),
        "native_sr": _num(a.get("sample_rate"), int),
        "channels": _num(a.get("channels"), int),
        "channel_layout": a.get("channel_layout"),
        "bits_per_sample": _num(a.get("bits_per_sample"), int) or _num(a.get("bits_per_raw_sample"), int),
        "sample_fmt": a.get("sample_fmt"),
        "bitrate": _num(a.get("bit_rate")) or _num(fmt.get("bit_rate")),
        "container_duration_s": _num(fmt.get("duration")),
        "encoder": (fmt.get("tags") or {}).get("encoder") or (a.get("tags") or {}).get("encoder"),
        "tags": tags,
        "n_streams": len(streams),
        "is_lossy": lossy,
        "extension_mismatch": bool(allowed) and bool(fmt_names) and not (fmt_names & allowed),
        "probe_ok": bool(info),
    }


def triage(path: str, cfg: Optional[RouterConfig] = None) -> Context:
    """Probe + decode one file into a Context whose `native` dict describes the container and signal."""
    cfg = cfg or RouterConfig()
    path = str(path)
    probe = haudio.probe(path)
    native = _native_info(path, probe)
    t = time.perf_counter()
    try:
        x, native_sr = haudio.load(path)
    except Exception as e:
        x, native_sr = np.zeros(0, np.float32), native.get("native_sr") or haudio.ANALYSIS_SR
        native["decode_error"] = f"{type(e).__name__}: {e}"
    x = np.nan_to_num(np.asarray(x, np.float32))
    if x.size == 0 and "decode_error" not in native:
        native["decode_error"] = "no audio samples could be decoded (corrupt or empty file)"
    native["native_sr"] = native.get("native_sr") or native_sr
    native["decode_ms"] = (time.perf_counter() - t) * 1000

    ctx = Context(path=path, audio=x, sr=haudio.ANALYSIS_SR, native=native)
    ctx.cache["ffprobe"] = probe   # raw ffprobe JSON, reused by the container/compression experts (one probe per file)
    sr = ctx.sr
    native["duration_s"] = x.size / sr
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
    native["peak"] = peak
    native["rms_dbfs"] = 20 * math.log10(rms + 1e-12)
    native["clip_fraction"] = float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0
    native["is_silent"] = x.size == 0 or native["rms_dbfs"] < cfg.silence_dbfs

    # Effective bandwidth at the native rate when we can read it cheaply (PCM/FLAC/OGG via libsndfile);
    # otherwise at the 16 kHz analysis rate, which can only reveal cut-offs below 8 kHz.
    bw_sr, bw_x = sr, x
    try:
        if (native.get("native_sr") or sr) > sr:
            import soundfile as sf
            y, ysr = sf.read(path, dtype="float32", always_2d=True)
            bw_x, bw_sr = y.mean(axis=1), ysr
    except Exception:
        pass
    bw = effective_bandwidth(bw_x, bw_sr) if not native["is_silent"] else float("nan")
    native["effective_bandwidth_hz"] = bw
    native["bandwidth_measured_at_hz"] = bw_sr
    nyq = (native.get("native_sr") or bw_sr) / 2.0
    native["band_limited"] = bool(np.isfinite(bw) and bw < cfg.band_limited_ratio * min(nyq, bw_sr / 2.0))
    native["narrowband"] = bool(np.isfinite(bw) and bw < 4500.0)
    return ctx


# --------------------------------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))


def _logit(p: float) -> float:
    p = min(1 - 1e-9, max(1e-9, float(p)))
    return math.log(p / (1 - p))


def vote_logodds(evidence: Iterable[Evidence], cfg: RouterConfig, exclude: Iterable[Evidence] = ()) -> float:
    """Transparent fallback: sum of direction x strength x category weight x gain, clipped.

    One full-strength synthetic finding in a weight-1 category contributes +2 log-odds (p ~ 0.88);
    authenticity findings subtract symmetrically. This is a hand-set prior, not a learned model, and is
    only used when no learned fusion is supplied.
    """
    skip = {id(e) for e in exclude}
    total = 0.0
    for ev in evidence:
        if id(ev) in skip or ev.error:
            continue
        w = cfg.category_weights.get(ev.category, 0.5)
        for f in ev.findings:
            s = float(np.clip(_finite(f.strength, 0.0), 0.0, 1.0))
            total += int(np.sign(f.direction)) * s * w * cfg.vote_gain
    return float(np.clip(total, -cfg.vote_clip, cfg.vote_clip))


def _finite(v, default=float("nan")) -> float:
    try:
        v = float(v)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def merged_features(evidence: Iterable[Evidence]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for ev in evidence:
        feats.update(ev.features)
    return feats


@dataclass
class _State:
    """Mutable per-file routing state shared by the predicates."""
    ctx: Context
    cfg: RouterConfig
    mode: str
    evidence: List[Evidence] = field(default_factory=list)
    deep_ev: Optional[Evidence] = None
    deep_threshold: float = 0.0
    deep_scale: float = 1.0
    fusion: Optional[Fusion] = None
    deep_scorer: Optional[DeepScorer] = None
    provisional: Optional[float] = None
    provisional_source: str = "none"

    def deep_z(self) -> Optional[float]:
        if self.deep_ev is None:
            return None
        s = _finite(self.deep_ev.score)
        if not math.isfinite(s):
            return None
        return (s - self.deep_threshold) / (self.deep_scale or 1.0)

    def update_provisional(self) -> Tuple[float, str]:
        """Provisional log-odds relative to the decision threshold, from the best available source."""
        if self.fusion is not None:
            try:
                p = float(self.fusion(merged_features(self.evidence), self.evidence))
                if math.isfinite(p):
                    self.provisional = _logit(p) - _logit(self.cfg.fusion_threshold)
                    self.provisional_source = "learned fusion"
                    return self.provisional, self.provisional_source
            except Exception:
                pass  # fall through to the deep score / vote
        z = self.deep_z()
        vote = vote_logodds(self.evidence, self.cfg, exclude=[self.deep_ev] if self.deep_ev else [])
        if z is not None:
            self.provisional = z + self.cfg.vote_weight_with_deep * vote
            self.provisional_source = "deep detector (+ expert vote nudge)"
        else:
            self.provisional = vote
            self.provisional_source = "expert-finding vote"
        return self.provisional, self.provisional_source


# --------------------------------------------------------------------------------------------------
# Predicates: name -> fn(state) -> (bool, human-readable reason)
# --------------------------------------------------------------------------------------------------

def _p_always(st: _State):
    return True, "runs on every file"


def _p_deep_available(st: _State):
    return (st.deep_scorer is not None,
            "a deep detector was supplied" if st.deep_scorer is not None else "no deep detector supplied")


def _p_lossy_or_band_limited(st: _State):
    n = st.ctx.native
    reasons = []
    if n.get("is_lossy"):
        reasons.append(f"lossy codec '{n.get('codec')}'")
    if n.get("extension_mismatch"):
        reasons.append(f"extension {n.get('extension')} does not match container '{n.get('container')}'")
    bw = _finite(n.get("effective_bandwidth_hz"))
    if n.get("band_limited"):
        reasons.append(f"effective bandwidth {bw / 1000:.1f} kHz is well below Nyquist")
    else:
        # Cheap experts may have measured a codec cut-off more carefully than triage did. Each cut-off is judged
        # against the Nyquist of the rate it was measured at (codec_cutoff reports it as codec_cutoff_native_sr;
        # the other experts analyse the 16 kHz signal), otherwise every 44.1/48 kHz file would look band-limited
        # at the 8 kHz analysis Nyquist.
        nyq = (n.get("native_sr") or st.ctx.sr) / 2.0
        feats = merged_features(st.evidence)
        for k, v in feats.items():
            if not k.endswith("cutoff_hz") or not math.isfinite(v):
                continue
            meas_nyq = min(nyq, _finite(feats.get(k[:-3] + "_native_sr"), float(st.ctx.sr)) / 2.0)
            if 0 < v < st.cfg.band_limited_ratio * meas_nyq:
                reasons.append(f"expert feature {k} = {v:.0f} Hz indicates band-limiting")
                break
    if reasons:
        return True, "; ".join(reasons)
    if not n.get("codec"):
        return False, "no codec information (nothing to re-encode)"
    return False, f"lossless '{n.get('codec')}' at full bandwidth ({bw / 1000:.1f} kHz)" if math.isfinite(bw) \
        else f"lossless '{n.get('codec')}'"


def _p_uncertain(st: _State):
    z, src = st.update_provisional()
    band = st.cfg.uncertainty_band
    if abs(z) < band:
        return True, f"provisional score {z:+.2f} is inside the +/-{band:g} uncertainty band ({src})"
    side = "synthetic" if z > 0 else "genuine"
    return False, f"provisional score {z:+.2f} is confidently {side} ({src})"


def _strong(st: _State, direction: int) -> List[Tuple[str, Finding]]:
    out = []
    for ev in st.evidence:
        if ev is st.deep_ev or ev.error:
            continue
        for f in ev.findings:
            if int(np.sign(f.direction)) == direction and _finite(f.strength, 0.0) >= st.cfg.strong_flag:
                out.append((ev.expert, f))
    return out


def _p_strong_flag(st: _State):
    flags = _strong(st, +1)
    if flags:
        expert, f = max(flags, key=lambda t: t[1].strength)
        return True, f"strong synthetic flag from '{expert}' (strength {f.strength:.2f}): {_short(f.text, 110)}"
    return False, "no expert raised a strong synthetic flag"


def _p_conflict(st: _State):
    pos, neg = _strong(st, +1), _strong(st, -1)
    if pos and neg:
        return True, f"experts disagree strongly ('{pos[0][0]}' says synthetic, '{neg[0][0]}' says genuine)"
    z = st.deep_z()
    if z is not None and abs(z) >= st.cfg.uncertainty_band:
        against = neg if z > 0 else pos
        if against:
            return True, f"'{against[0][0]}' strongly contradicts the deep detector"
    return False, "no strong contradictions"


def _p_long_clip(st: _State):
    d = st.ctx.native.get("duration_s", 0.0)
    return (d > st.cfg.long_clip_s,
            f"clip is {d:.1f} s (> {st.cfg.long_clip_s:g} s, room for a local edit)" if d > st.cfg.long_clip_s
            else f"clip is {d:.1f} s")


def _p_has_signal(st: _State):
    n = st.ctx.native
    if n.get("decode_error"):
        return False, f"audio could not be decoded ({n['decode_error']})"
    if n.get("is_silent"):
        return False, f"no usable signal (RMS {n.get('rms_dbfs', -120):.0f} dBFS)"
    return True, "signal present"


def _p_narrowband(st: _State):
    return bool(st.ctx.native.get("narrowband")), "narrowband"


def _p_very_short(st: _State):
    return st.ctx.native.get("duration_s", 0.0) < 2.0, "shorter than 2 s"


PREDICATES: Dict[str, Callable[[_State], Tuple[bool, str]]] = {
    "always": _p_always, "deep_scorer_available": _p_deep_available,
    "lossy_or_band_limited": _p_lossy_or_band_limited, "uncertain": _p_uncertain,
    "strong_flag": _p_strong_flag, "conflict": _p_conflict, "long_clip": _p_long_clip,
    "has_signal": _p_has_signal, "narrowband": _p_narrowband, "very_short": _p_very_short,
}


def _evaluate(rule: Dict[str, Any], st: _State) -> Tuple[bool, str]:
    """OR over `when`, AND over `requires`. Returns (fires, reason)."""
    for req in rule.get("requires", []):
        ok, why = PREDICATES[req](st)
        if not ok:
            return False, f"requirement '{req}' not met: {why}"
    reasons_no = []
    for name in rule.get("when", []):
        ok, why = PREDICATES[name](st)
        if ok:
            return True, why
        reasons_no.append(why)
    return False, "; ".join(reasons_no) or "no trigger"


# --------------------------------------------------------------------------------------------------
# Manipulation-type attribution
# --------------------------------------------------------------------------------------------------

MANIPULATION_TYPES = ("bona fide", "TTS / neural vocoder", "voice conversion", "replay",
                      "laundered (transcoded/band-limited/noise)", "spliced/edited", "uncertain")

# Ordered rules, first match wins. Documented here so the report can print them verbatim.
ATTRIBUTION_RULES = [
    ("uncertain", "no decodable signal or no evidence at all"),
    ("spliced/edited", "a splice/discontinuity expert reports a seam with strength >= 0.6 "
                       "(a local edit is the most specific explanation, even inside a synthetic clip)"),
    ("replay", "an environment/replay finding (reverb, loudspeaker, re-recording) is synthetic-leaning "
               "with strength >= 0.6"),
    ("voice conversion", "cm-score >= synthetic threshold AND (speaker-consistency flags OR prosody evidence "
                         "leans natural while spectral/deep evidence leans synthetic: VC keeps the source "
                         "talker's prosody but replaces the timbre)"),
    ("TTS / neural vocoder", "cm-score >= synthetic threshold otherwise (spectral, vocoder, deep or "
                             "prosody evidence of generated speech)"),
    ("laundered (transcoded/band-limited/noise)", "compression-history evidence >= 0.5, or a lossy / "
                                                  "band-limited file, while the score is below the synthetic "
                                                  "threshold (processing may be hiding artefacts)"),
    ("bona fide", "cm-score <= bona fide threshold and no synthetic finding reaches the strong-flag level"),
    ("uncertain", "anything else"),
]


def _category_signals(evidence: Iterable[Evidence]) -> Dict[str, Dict[str, float]]:
    sig = {c: {"syn": 0.0, "real": 0.0, "net": 0.0} for c in CATEGORIES}
    for ev in evidence:
        if ev.error:
            continue
        c = sig.setdefault(ev.category, {"syn": 0.0, "real": 0.0, "net": 0.0})
        for f in ev.findings:
            s = float(np.clip(_finite(f.strength, 0.0), 0, 1))
            d = int(np.sign(f.direction))
            if d > 0:
                c["syn"] = max(c["syn"], s)
            elif d < 0:
                c["real"] = max(c["real"], s)
            c["net"] += d * s
    return sig


def _keyword_strength(evidence: Iterable[Evidence], words: Sequence[str], categories: Sequence[str] = ()) -> float:
    best = 0.0
    for ev in evidence:
        if ev.error:
            continue
        for f in ev.findings:
            if f.direction <= 0:
                continue
            hay = f"{ev.expert} {f.text}".lower()
            if any(w in hay for w in words) or (categories and ev.category in categories and
                                                any(w in ev.expert.lower() for w in words)):
                best = max(best, _finite(f.strength, 0.0))
    return best


def attribute(evidence: List[Evidence], cm: float, native: Dict[str, Any], cfg: RouterConfig) -> Tuple[str, List[str]]:
    """Apply ATTRIBUTION_RULES; returns (label, reasons incl. secondary tags)."""
    sig = _category_signals(evidence)
    reasons: List[str] = []
    usable = [e for e in evidence if not e.error]
    replay = _keyword_strength(usable, ("replay", "loudspeaker", "re-record", "rerecord", "reverb"), ("environment",))
    vc_kw = _keyword_strength(usable, ("voice conversion", "converted voice"))
    laundering = max(sig["compression"]["syn"], 0.5 if (native.get("is_lossy") or native.get("band_limited")) else 0.0)
    max_syn = max((v["syn"] for k, v in sig.items() if k != "deep"), default=0.0)

    # Secondary tags are reported regardless of the primary label.
    if sig["splice"]["syn"] >= 0.4:
        reasons.append(f"splice evidence (strength {sig['splice']['syn']:.2f})")
    if laundering >= 0.5:
        reasons.append("processing chain shows lossy/band-limited/transcoded audio")
    if replay >= 0.4:
        reasons.append(f"replay/reverb cues (strength {replay:.2f})")

    if native.get("decode_error") or native.get("is_silent") or not usable:
        return "uncertain", ["no decodable signal or no usable evidence"] + reasons
    if sig["splice"]["syn"] >= 0.6:
        return "spliced/edited", [f"splice expert strength {sig['splice']['syn']:.2f} >= 0.6"] + reasons
    if replay >= 0.6:
        return "replay", [f"replay/reverb cue strength {replay:.2f} >= 0.6"] + reasons
    if cm >= cfg.synthetic_above:
        # High score + natural-sounding prosody => the timbre, not the delivery, was generated (VC).
        if vc_kw >= 0.5 or sig["speaker"]["syn"] >= 0.5 or sig["prosody"]["net"] <= -0.3:
            why = ("speaker-consistency flag" if sig["speaker"]["syn"] >= 0.5 else
                   "explicit voice-conversion cue" if vc_kw >= 0.5 else
                   "natural prosody with synthetic timbre")
            return "voice conversion", [f"cm {cm:.2f} >= {cfg.synthetic_above}; {why}"] + reasons
        return "TTS / neural vocoder", [f"cm {cm:.2f} >= {cfg.synthetic_above}"] + reasons
    if laundering >= 0.5:
        return "laundered (transcoded/band-limited/noise)", [f"cm {cm:.2f} below synthetic threshold"] + reasons
    if cm <= cfg.bona_fide_below and max_syn < cfg.strong_flag:
        return "bona fide", [f"cm {cm:.2f} <= {cfg.bona_fide_below} and no strong synthetic flag"] + reasons
    return "uncertain", [f"cm {cm:.2f} between thresholds or conflicting flags"] + reasons


# --------------------------------------------------------------------------------------------------
# Summary text
# --------------------------------------------------------------------------------------------------

def _short(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "..."


def _describe_file(n: Dict[str, Any]) -> str:
    bits = []
    d = n.get("duration_s")
    if d:
        bits.append(f"{d:.1f} s")
    if n.get("native_sr"):
        bits.append(f"{n['native_sr'] / 1000:g} kHz")
    ch = n.get("channels")
    if ch:
        bits.append("mono" if ch == 1 else f"{ch}-channel")
    if n.get("codec"):
        bits.append(n["codec"])
    if n.get("container"):
        bits.append(f"in a {n['container'].split(',')[0].upper()} container")
    s = "The file is " + " ".join(bits) if bits else "The file could not be probed"
    if n.get("encoder"):
        s += f", written by {n['encoder']}"
    return s + "."


def summarize(res: FileResult, cfg: RouterConfig, n_available: int) -> str:
    cm = res.cm_score
    verdict = ("Likely SYNTHETIC" if cm >= cfg.synthetic_above else
               "Likely GENUINE" if cm <= cfg.bona_fide_below else "INCONCLUSIVE")
    ran = [e for e in res.evidence if not e.error]
    parts = [f"{verdict} (cm-score {cm:.2f}; attributed type: {res.manipulation_type}).",
             _describe_file(res.native)]
    esc = [t for t in res.trace if t.get("fired") and t.get("tier") in ("medium", "expensive")]
    tiers = ", ".join(t for t in res.tiers if t != "triage") or "none"
    route = f"The {res.mode} router ran {len(ran)} of {n_available} available analyses (tiers: {tiers})"
    if res.mode == "agentic":
        if esc:
            route += f"; it escalated because {esc[0]['reason']}"
        else:
            skipped = next((t for t in res.trace if t.get("tier") == "medium" and not t.get("fired")), None)
            if skipped:
                route += f"; no escalation was needed ({skipped['reason']})"
    parts.append(route + ".")
    ranked = sorted(((e, f) for e in ran for f in e.findings), key=lambda t: -_finite(t[1].strength, 0.0))
    syn = [f"{_short(f.text, 140)} [{e.expert}]" for e, f in ranked if f.direction > 0][:2]
    real = [f"{_short(f.text, 140)} [{e.expert}]" for e, f in ranked if f.direction < 0][:2]
    if syn:
        parts.append("Strongest signs of synthesis: " + " | ".join(syn) + ".")
    if real:
        parts.append("Strongest signs of authenticity: " + " | ".join(real) + ".")
    if not syn and not real:
        parts.append("No expert produced a directional finding.")
    src = {"fusion": "the learned fusion model", "deep": "the deep detector's calibrated score",
           "vote": "a transparent weighted vote of expert findings (no fusion model or deep detector supplied)",
           "none": "no usable evidence"}.get(res.score_source, res.score_source)
    parts.append(f"The score comes from {src}.")
    return " ".join(parts)


# --------------------------------------------------------------------------------------------------
# Main entry points
# --------------------------------------------------------------------------------------------------

def _call_deep(deep_scorer: DeepScorer, ctx: Context) -> Tuple[Optional[Evidence], float, Optional[str]]:
    t = time.perf_counter()
    try:
        ev = deep_scorer(ctx)
        err = None
    except Exception as e:
        ev, err = None, f"{type(e).__name__}: {e}"
    ms = (time.perf_counter() - t) * 1000
    if ev is not None and not isinstance(ev, Evidence):
        ev, err = None, f"deep scorer returned {type(ev).__name__}, not Evidence"
    if ev is not None:
        ev.category = "deep"
        ev.expert = ev.expert or getattr(deep_scorer, "name", "deep")
        ev.runtime_ms = ev.runtime_ms or ms
        ev.features = _clean_features(ev.features)
        if ev.score is not None and not ev.features:
            ev.features = {f"{ev.expert}_score": float(ev.score)}
    return ev, ms, err


def run(path: str, deep_scorer: Optional[DeepScorer] = None, fusion: Optional[Fusion] = None,
        mode: str = "agentic", config: Optional[RouterConfig] = None, corpus: Optional[Dict[str, Any]] = None,
        transcript: Optional[str] = None, policy: Optional[List[Dict[str, Any]]] = None) -> FileResult:
    """Analyse one file. mode="agentic" follows POLICY; mode="full" runs every available analysis.

    deep_scorer(ctx) -> Evidence (category "deep", .score = spoof log-odds) or None. It may expose
    `threshold` and `scale` attributes to calibrate the uncertainty band. fusion(features, evidence)
    -> probability of synthetic; in agentic mode it receives only the features that were computed,
    so it must tolerate missing keys (e.g. LightGBM with NaN) or be run in full mode.
    """
    assert mode in ("agentic", "full"), mode
    cfg = config or RouterConfig()
    policy = policy or POLICY
    t0 = time.perf_counter()
    import_errors = ensure_experts_loaded()
    registry = dict(REGISTRY)
    trace: List[Dict[str, Any]] = []

    # ---- 1. triage ------------------------------------------------------------------------------
    t = time.perf_counter()
    try:
        ctx = triage(path, cfg)
    except Exception as e:
        return FileResult(path=str(path), cm_score=0.5, manipulation_type="uncertain", mode=mode,
                          summary=f"INCONCLUSIVE: the file could not be opened ({type(e).__name__}: {e}).",
                          error=f"triage failed: {type(e).__name__}: {e}", score_source="none",
                          trace=[{"step": 0, "tier": "triage", "rule": "triage", "fired": True,
                                  "reason": f"failed: {e}", "experts_run": [], "ms": 0.0}])
    ctx.corpus = corpus if corpus is not None else {}
    if transcript:
        ctx.cache["transcript"] = transcript
        ctx.cache["transcript_source"] = "transcript supplied with the batch"
    n = ctx.native
    trace.append({"step": 0, "tier": "triage", "rule": "triage", "fired": True,
                  "reason": (f"{n.get('codec')} / {n.get('container')} @ {n.get('native_sr')} Hz, "
                             f"{n.get('duration_s', 0):.2f} s, lossy={n.get('is_lossy')}, "
                             f"bandwidth={_finite(n.get('effective_bandwidth_hz')) / 1000:.2f} kHz, "
                             f"extension_mismatch={n.get('extension_mismatch')}"
                             + (f", DECODE ERROR {n['decode_error']}" if n.get("decode_error") else "")),
                  "experts_run": [], "experts_missing": [], "ms": (time.perf_counter() - t) * 1000})
    if import_errors:
        trace.append({"step": 0, "tier": "triage", "rule": "expert_imports", "fired": False,
                      "reason": "some expert modules failed to import: " + ", ".join(sorted(import_errors)),
                      "experts_run": [], "experts_missing": sorted(import_errors), "ms": 0.0})

    st = _State(ctx=ctx, cfg=cfg, mode=mode, fusion=fusion, deep_scorer=deep_scorer,
                deep_threshold=float(getattr(deep_scorer, "threshold", cfg.deep_threshold)),
                deep_scale=float(getattr(deep_scorer, "scale", cfg.deep_scale)))
    claimed = _claimed_names(policy)
    done: set = set()
    tiers_run = ["triage"]

    # Suppressions are decided once, from triage facts only.
    suppressed: Dict[str, str] = {n_: "disabled by configuration" for n_ in cfg.disabled_experts}
    if mode == "agentic":
        for sr_ in SUPPRESS_RULES:
            ok, _ = PREDICATES[sr_["when"]](st)
            if ok:
                suppressed.setdefault(sr_["expert"], sr_["why"])

    # ---- 2..6. policy rules ---------------------------------------------------------------------
    for step, rule in enumerate(policy, start=1):
        t = time.perf_counter()
        if mode == "full":
            fires, reason = True, "full mode: every analysis runs"
            if rule.get("deep") and deep_scorer is None:
                fires, reason = False, "full mode, but no deep detector supplied"
        else:
            fires, reason = _evaluate(rule, st)
        entry = {"step": step, "tier": rule["tier"], "rule": rule["id"], "fired": fires, "reason": reason,
                 "experts_run": [], "experts_missing": [], "experts_suppressed": [], "ms": 0.0,
                 "provisional": st.provisional}
        if fires and rule.get("deep"):
            ev, ms, err = _call_deep(deep_scorer, ctx)
            if ev is not None:
                st.deep_ev = ev
                st.evidence.append(ev)
                entry["experts_run"].append(ev.expert)
                done.add(ev.expert)
            else:
                entry["reason"] += f"; deep scorer produced no score ({err or 'file not covered'})"
        elif fires:
            names, missing = _select_experts(rule, registry, claimed, done)
            entry["experts_missing"] = missing
            for name in names:
                if name in suppressed:
                    entry["experts_suppressed"].append(f"{name}: {suppressed[name]}")
                    continue
                ev = _run_expert(registry[name], ctx)
                st.evidence.append(ev)
                done.add(name)
                entry["experts_run"].append(name)
        entry["ms"] = (time.perf_counter() - t) * 1000
        if entry["experts_run"] and rule["tier"] not in tiers_run:
            tiers_run.append(rule["tier"])
        trace.append(entry)

    # Full mode also sweeps any registered expert that no rule covered (e.g. unusual cost labels).
    if mode == "full":
        rest = [n_ for n_ in registry if n_ not in done and n_ not in suppressed]
        if rest:
            t = time.perf_counter()
            for name in rest:
                st.evidence.append(_run_expert(registry[name], ctx))
                done.add(name)
            trace.append({"step": len(policy) + 1, "tier": "expensive", "rule": "full_sweep", "fired": True,
                          "reason": "full mode: remaining registered experts", "experts_run": rest,
                          "experts_missing": [], "experts_suppressed": [], "ms": (time.perf_counter() - t) * 1000})

    # ---- final score ----------------------------------------------------------------------------
    feats = merged_features(st.evidence)
    cm, source = None, "none"
    if fusion is not None:
        try:
            p = float(fusion(feats, st.evidence))
            if math.isfinite(p):
                cm, source = float(np.clip(p, 0.0, 1.0)), "fusion"
        except Exception as e:
            trace.append({"step": len(trace), "tier": "fusion", "rule": "fusion", "fired": False,
                          "reason": f"fusion failed ({type(e).__name__}: {e}); falling back",
                          "experts_run": [], "ms": 0.0})
    if cm is None and st.deep_z() is not None:
        cm, source = _sigmoid(st.deep_z()), "deep"
    if cm is None:
        usable = [e for e in st.evidence if not e.error and e.findings]
        cm, source = (_sigmoid(vote_logodds(st.evidence, cfg)), "vote") if usable else (0.5, "none")
    st.update_provisional()

    res = FileResult(path=str(path), cm_score=float(cm), evidence=st.evidence, trace=trace, mode=mode,
                     score_source=source, native=n, features=feats, tiers=tiers_run,
                     provisional=st.provisional, ctx=ctx)
    res.manipulation_type, res.attribution_reasons = attribute(st.evidence, res.cm_score, n, cfg)
    res.summary = summarize(res, cfg, len(registry) + (1 if deep_scorer is not None else 0))
    res.runtime_ms = (time.perf_counter() - t0) * 1000
    return res


# --------------------------------------------------------------------------------------------------
# Batch mode
# --------------------------------------------------------------------------------------------------

class BatchResults(list):
    """list[FileResult] plus batch-level metadata (corpus-step log, wall time, mode)."""
    corpus_log: List[Dict[str, Any]]
    mode: str = "agentic"
    wall_s: float = 0.0


def list_audio_files(dir_or_files) -> List[str]:
    if isinstance(dir_or_files, (str, os.PathLike)):
        p = Path(dir_or_files)
        if p.is_dir():
            return sorted(str(q) for q in p.rglob("*") if q.suffix.lower() in AUDIO_EXTENSIONS and q.is_file())
        return [str(p)]
    return [str(q) for q in dir_or_files]


def load_transcripts(src) -> Dict[str, str]:
    """Accept a dict or a JSON file mapping filename (basename) -> transcript text."""
    if not src:
        return {}
    if isinstance(src, dict):
        return {os.path.basename(k): v for k, v in src.items()}
    with open(src) as fh:
        data = json.load(fh)
    return {os.path.basename(k): (v if isinstance(v, str) else (v or {}).get("text", "")) for k, v in data.items()}


def build_corpus(files: Sequence[str], transcripts: Dict[str, str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Batch-level context: calls `build_corpus_stats` on every expert module that defines one.

    Arguments are matched by parameter name: transcripts/texts -> the transcript dict, files/paths ->
    the file list. E.g. semantic.build_corpus_stats counts repeated sentences across the batch, and a
    speaker module may cluster speakers. Missing modules or failures are logged, never raised.
    """
    ensure_experts_loaded()
    corpus: Dict[str, Any] = {"n_files": len(files), "transcripts": transcripts}
    log: List[Dict[str, Any]] = []
    for modname, mod in sorted(sys.modules.items()):
        if not modname.startswith("hearsay.experts.") or mod is None:
            continue
        fn = getattr(mod, "build_corpus_stats", None)
        if not callable(fn):
            continue
        t = time.perf_counter()
        try:
            params = list(inspect.signature(fn).parameters.values())
            kwargs, args = {}, []
            for prm in params:
                key = prm.name.lower()
                val = (transcripts if key in ("transcripts", "texts", "transcript_map") else
                       list(files) if key in ("files", "paths", "file_list") else inspect.Parameter.empty)
                if val is inspect.Parameter.empty:
                    if prm.default is inspect.Parameter.empty and prm.kind in (prm.POSITIONAL_ONLY,
                                                                               prm.POSITIONAL_OR_KEYWORD):
                        val = transcripts  # best guess for an unnamed first argument
                    else:
                        continue
                if prm.kind == prm.POSITIONAL_ONLY:
                    args.append(val)
                else:
                    kwargs[prm.name] = val
            if any(k in kwargs for k in ("transcripts", "texts", "transcript_map")) and not transcripts:
                log.append({"module": modname, "ok": False, "reason": "no transcripts available", "ms": 0.0})
                continue
            stats = fn(*args, **kwargs)
            if isinstance(stats, dict):
                corpus.update(stats)
            corpus.setdefault("_sources", []).append(modname)
            log.append({"module": modname, "ok": True, "reason": f"added {len(stats or {})} corpus keys",
                        "ms": (time.perf_counter() - t) * 1000})
        except Exception as e:
            log.append({"module": modname, "ok": False, "reason": f"{type(e).__name__}: {e}",
                        "ms": (time.perf_counter() - t) * 1000})
    if not log:
        log.append({"module": "-", "ok": False, "reason": "no expert module defines build_corpus_stats", "ms": 0.0})
    return corpus, log


def run_batch(dir_or_files, deep_scorer: Optional[DeepScorer] = None, fusion: Optional[Fusion] = None,
              mode: str = "agentic", config: Optional[RouterConfig] = None, transcripts=None,
              on_result: Optional[Callable[[FileResult], None]] = None, json_out: Optional[str] = None,
              progress: bool = True, keep_context: bool = False, keep_artifacts: bool = False,
              limit: Optional[int] = None) -> BatchResults:
    """Analyse many files. Runs the corpus step first, then `run` per file.

    on_result(result) is called while the result still holds its Context and artifacts (use it to
    write per-file HTML reports); afterwards audio/artifacts are dropped unless keep_* is set, so a
    1671-file batch stays small in memory.
    """
    files = list_audio_files(dir_or_files)
    if limit:
        files = files[:limit]
    trans = load_transcripts(transcripts)
    t0 = time.perf_counter()
    corpus, corpus_log = build_corpus(files, trans)
    out = BatchResults()
    out.corpus_log, out.mode = corpus_log, mode
    embeddings: Dict[str, Any] = {}
    for i, f in enumerate(files, 1):
        res = run(f, deep_scorer=deep_scorer, fusion=fusion, mode=mode, config=config, corpus=corpus,
                  transcript=trans.get(os.path.basename(f)))
        if res.ctx is not None and res.ctx.cache.get("spk_emb") is not None:
            embeddings[res.name] = np.asarray(res.ctx.cache["spk_emb"], np.float32)
        if on_result is not None:
            try:
                on_result(res)
            except Exception as e:
                res.error = (res.error + "; " if res.error else "") + f"on_result failed: {type(e).__name__}: {e}"
        if not keep_context:
            res.ctx = None
        if not keep_artifacts:
            for ev in res.evidence:
                ev.artifacts = {}
        out.append(res)
        if progress:
            print(f"[{i:>5}/{len(files)}] {res.name:<28} cm={res.cm_score:.3f} {res.manipulation_type:<24} "
                  f"tiers={'+'.join(t for t in res.tiers if t != 'triage') or '-':<30} {res.runtime_ms:7.0f} ms",
                  file=sys.stderr, flush=True)
    out.corpus_log.append(speaker_cluster_pass(out, embeddings, corpus))
    out.wall_s = time.perf_counter() - t0
    if json_out:
        export_json(out, json_out)
    return out


def speaker_cluster_pass(results: Sequence[FileResult], embeddings: Dict[str, Any],
                         corpus: Dict[str, Any], min_files: int = 3) -> Dict[str, Any]:
    """Post-batch corpus step: cluster the voiceprints the speaker expert cached, then annotate results.

    Uses `cluster_speakers` from any expert module that provides it (the speaker expert does). Only
    files that reached the speaker expert have a voiceprint, so in agentic mode the clusters cover the
    escalated subset. Per-file HTML written during the batch predates this step; the JSON export and
    dashboard include it.
    """
    t = time.perf_counter()
    mod = next((m for k, m in sorted(sys.modules.items()) if k.startswith("hearsay.experts.") and m is not None
                and callable(getattr(m, "cluster_speakers", None))), None)
    if mod is None:
        return {"module": "speaker clustering", "ok": False, "reason": "no expert module provides cluster_speakers",
                "ms": 0.0}
    if len(embeddings) < min_files:
        return {"module": f"{mod.__name__}.cluster_speakers", "ok": False,
                "reason": f"only {len(embeddings)} voiceprints (need >= {min_files})", "ms": 0.0}
    try:
        stats = mod.cluster_speakers(embeddings)
        corpus.update(stats)
        annotate = getattr(mod, "_add_typicality", None)
        n_annot = 0
        for r in results:
            if r.name not in embeddings or not callable(annotate):
                continue
            for ev in r.evidence:
                if ev.category == "speaker" and not ev.error and not any(k.endswith("_typicality") for k in ev.features):
                    stub = Context(path=r.path, audio=np.zeros(0, np.float32), sr=haudio.ANALYSIS_SR, corpus=corpus)
                    annotate(stub, ev, embeddings[r.name])
                    ev.features = _clean_features(ev.features)
                    r.features.update(ev.features)
                    n_annot += 1
        n_clusters = len(set((stats.get("speaker_cluster") or {}).values()))
        return {"module": f"{mod.__name__}.cluster_speakers", "ok": True,
                "reason": f"{len(embeddings)} voiceprints -> {n_clusters} speaker clusters; annotated {n_annot} files",
                "ms": (time.perf_counter() - t) * 1000}
    except Exception as e:
        return {"module": f"{mod.__name__}.cluster_speakers", "ok": False, "reason": f"{type(e).__name__}: {e}",
                "ms": (time.perf_counter() - t) * 1000}


def export_json(results: Sequence[FileResult], path: str, extra: Optional[Dict[str, Any]] = None) -> str:
    """Write every file's evidence, findings, trace and scores to one JSON document."""
    doc = {
        "router_version": ROUTER_VERSION,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "mode": getattr(results, "mode", None),
        "corpus_log": _jsonable(getattr(results, "corpus_log", [])),
        "accounting": compute_accounting(results),
        "policy": _jsonable(POLICY),
        "files": [r.to_dict() for r in results],
    }
    if extra:
        doc.update(_jsonable(extra))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
    return str(path)


# --------------------------------------------------------------------------------------------------
# Compute accounting
# --------------------------------------------------------------------------------------------------

def compute_accounting(results: Sequence[FileResult], reference_ms: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """How many files reached each tier, and where the milliseconds went.

    `full_estimate_ms` prices a full run with the mean per-call cost observed for each expert (or
    `reference_ms`, e.g. measured by a full-mode run on a sample); experts never observed are listed
    in `unpriced_experts` and excluded, so the agentic fraction is then an upper bound.
    """
    n = len(results)
    tier_files = {t: 0 for t in TIER_ORDER}
    expert_ms: Dict[str, float] = {}
    expert_runs: Dict[str, int] = {}
    triage_ms = 0.0
    total_ms = 0.0
    for r in results:
        total_ms += r.runtime_ms
        for t in set(r.tiers):
            tier_files[t] = tier_files.get(t, 0) + 1
        triage_ms += sum(e.get("ms", 0.0) for e in r.trace if e.get("tier") == "triage")
        for ev in r.evidence:
            expert_ms[ev.expert] = expert_ms.get(ev.expert, 0.0) + float(ev.runtime_ms or 0.0)
            expert_runs[ev.expert] = expert_runs.get(ev.expert, 0) + 1
    mean_ms = {k: expert_ms[k] / expert_runs[k] for k in expert_ms}
    if reference_ms:
        mean_ms.update(reference_ms)
    names = set(REGISTRY) | set(expert_ms)
    unpriced = sorted(k for k in names if k not in mean_ms)
    analysed_ms = sum(expert_ms.values())
    full_est = sum(mean_ms.get(k, 0.0) for k in names if k in mean_ms) * n
    return {
        "n_files": n,
        "files_per_tier": tier_files,
        "expert_total_ms": dict(sorted(expert_ms.items(), key=lambda kv: -kv[1])),
        "expert_runs": expert_runs,
        "expert_mean_ms": {k: round(v, 2) for k, v in mean_ms.items()},
        "triage_total_ms": triage_ms,
        "analysis_total_ms": analysed_ms,
        "wall_total_ms": total_ms,
        "full_estimate_analysis_ms": full_est,
        "agentic_fraction_of_full": (analysed_ms / full_est) if full_est > 0 else None,
        "unpriced_experts": unpriced,
    }


def describe_policy() -> str:
    """Plain-text rendering of the routing policy (for the CLI and README)."""
    lines = [f"HEARSAY routing policy v{ROUTER_VERSION}", ""]
    for i, r in enumerate(POLICY, 1):
        when = " OR ".join(r.get("when", []))
        req = (" AND requires " + " AND ".join(r["requires"])) if r.get("requires") else ""
        what = "deep detector" if r.get("deep") else ", ".join(r.get("experts", []))
        if r.get("cost"):
            what += f" (+ any other '{r['cost']}' expert)"
        lines.append(f"{i}. [{r['tier']}] {r['id']}: when {when}{req}")
        lines.append(f"     runs: {what}")
        lines.append(f"     why : {r['why']}")
    lines.append("")
    lines.append("Suppressions:")
    for s in SUPPRESS_RULES:
        lines.append(f"  - skip {s['expert']} when {s['when']}: {s['why']}")
    lines.append("")
    lines.append("Attribution rules (first match wins):")
    for label, rule in ATTRIBUTION_RULES:
        lines.append(f"  - {label}: {rule}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# Reference deep_scorer: cached AASIST scores
# --------------------------------------------------------------------------------------------------

class CachedDeepScorer:
    """Reference implementation of the deep_scorer interface backed by precomputed scores.

    Reads a JSON {model_key: {filename: spoof log-odds}} (e.g. work/aasist_scores.json produced by
    AASIST models trained on ASVspoof 2019 LA), standardises each model over the batch, and averages.
    `threshold` is the z value at which the models' own log-odds cross 0 (their decision boundary),
    so the router's uncertainty band is centred on what the detectors themselves call the boundary.
    Files not in the cache return None and the router logs that the deep tier had no score.
    """

    def __init__(self, json_path: str, keys: Sequence[str] = ("aasist19", "aasist19L"), name: str = "aasist"):
        with open(json_path) as fh:
            data = json.load(fh)
        self.name = name
        self.keys = [k for k in keys if k in data] or list(data)[:1]
        self.data = {k: data[k] for k in self.keys}
        self.stats = {}
        for k in self.keys:
            v = np.array(list(self.data[k].values()), float)
            self.stats[k] = (float(v.mean()), float(v.std() + 1e-9))
        self.threshold = float(np.mean([(0.0 - m) / s for m, s in self.stats.values()]))
        self.scale = 1.0
        common = set.intersection(*(set(self.data[k]) for k in self.keys))
        self._z_sorted = np.sort([self._z({k: self.data[k][f] for k in self.keys}) for f in common])

    def _z(self, raws: Dict[str, float]) -> float:
        return float(np.mean([(raws[k] - self.stats[k][0]) / self.stats[k][1] for k in raws]))

    def __call__(self, ctx: Context) -> Optional[Evidence]:
        fname = os.path.basename(ctx.path)
        raws = {k: float(self.data[k][fname]) for k in self.keys if fname in self.data[k]}
        if not raws:
            return None
        z = self._z(raws)
        pct = float(np.searchsorted(self._z_sorted, z) / max(1, len(self._z_sorted))) * 100
        margin = z - self.threshold
        spoof_side = sum(v > 0 for v in raws.values())
        raw_txt = ", ".join(f"{k} {v:+.1f}" for k, v in raws.items())
        text = (f"Deep anti-spoofing detectors (AASIST, trained on ASVspoof 2019 LA) score this clip "
                f"z = {z:+.2f} ({pct:.0f}th percentile of the batch); {spoof_side} of {len(raws)} models put it on "
                f"the spoof side of their own decision boundary (log-odds {raw_txt}). These networks learn "
                f"vocoder and phase artefacts that are hard to hear.")
        direction = 1 if margin >= 0.5 else (-1 if margin <= -0.5 else 0)
        feats = {f"{self.name}_{k}_logodds": v for k, v in raws.items()}
        feats[f"{self.name}_z"] = z
        return Evidence(expert=self.name, category="deep", features=feats, score=z,
                        findings=[Finding(text, direction, float(min(1.0, abs(margin) / 2.5)))])


def load_callable(spec: str):
    """Import 'package.module:attr' (used by the CLI for --deep and --fusion)."""
    import importlib

    mod, _, attr = spec.partition(":")
    obj = importlib.import_module(mod)
    for part in (attr or "").split("."):
        if part:
            obj = getattr(obj, part)
    return obj
