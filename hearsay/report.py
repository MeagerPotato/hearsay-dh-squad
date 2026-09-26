"""Self-contained HTML forensic reports: one page per file plus a batch dashboard (index.html).

Everything is inline (matplotlib Agg PNGs as base64, CSS in <style>), so reports open offline and
can be e-mailed as single files. Templates are jinja2 with autoescaping; expert text is never
trusted as HTML.

Artifact conventions an expert can use to get a plot (all optional, read from Evidence.artifacts):
  ctx.cache["stft_mag"] linear-frequency magnitude STFT [freq, time] (preferred); else a spectrogram_db artifact:
                        2-D dB array over 0..sr/2, or uint8 + spectrogram_meta (db_top, db_range, n_mels) as
                        written by spectral_stats; else computed here
  seams_s (+seam_types) seam times in seconds -> red dashed lines on the waveform and spectrogram
  zero_runs             [(start_s, duration_ms, region), ...] digital-silence runs -> shaded spans
  f0 / ctx.cache["f0"]  dict(t=..., f0=...) or 1-D array (Hz, 0/NaN = unvoiced) -> pitch contour
  speaker_drift         dict(t=..., sim=..., drift_threshold=...) -> speaker-identity drift curve
  <name>_hz + <name>_db same-length 1-D arrays -> a spectrum-style plot (e.g. ENF, long-term spectrum)
  plot_<name>           dict(x=..., y=..., title=, xlabel=, ylabel=, hline=) -> generic line plot
"""
from __future__ import annotations

import base64
import datetime as _dt
import io
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from jinja2 import Environment, BaseLoader, select_autoescape  # noqa: E402

from hearsay.evidence import CATEGORIES, REGISTRY, Evidence  # noqa: E402
from hearsay import orchestrator as orch  # noqa: E402

# ---- palette (reference data-viz palette; blue = genuine pole, red = synthetic pole) ---------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8986"
GRID = "#e4e3df"
BLUE = "#2a78d6"      # evidence of authenticity
RED = "#e34948"       # evidence of synthesis
ORANGE = "#eb6834"    # measured cut-off frequency
NEUTRAL = "#b9b8b3"
SPEC_CMAP = LinearSegmentedColormap.from_list(
    "hearsay_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

CATEGORY_LABELS = {
    "container": ("Container & metadata", "How the file was packaged: codec, encoder tags, header consistency."),
    "spectral": ("Spectral", "Frequency-domain fingerprints: band-limiting, vocoder harmonics, unnatural smoothness."),
    "prosody": ("Prosody & phonetics", "Pitch contour, jitter/shimmer, pauses and breaths: how the voice is delivered."),
    "environment": ("Acoustic environment", "Mains hum (ENF), noise floor and reverberation: the room the voice was in."),
    "compression": ("Compression history", "Traces of lossy encoding and re-encoding."),
    "speaker": ("Speaker consistency", "Whether the same voice identity persists through the clip."),
    "deep": ("Deep-learning detector", "Neural anti-spoofing networks trained on known deepfake systems."),
    "splice": ("Splices & discontinuities", "Seams, clicks and digital-silence gaps left by editing."),
    "semantic": ("Content & transcript", "What is said, and whether the same script recurs across files."),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlecolor": INK,
})


# ------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------

def _png(fig, dpi: int = 96) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def fmt(v: Any) -> str:
    """Compact human formatting for feature values."""
    if v is None:
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "-"
    a = abs(f)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{f:,.0f}"
    if a >= 10:
        return f"{f:.1f}"
    if a >= 0.01:
        return f"{f:.3g}"
    return f"{f:.2e}"


def _as_array(v) -> Optional[np.ndarray]:
    try:
        a = np.asarray(v, dtype=float)
        return a if a.ndim >= 1 and a.size else None
    except (TypeError, ValueError):
        return None


def _all_artifacts(result) -> List[Tuple[str, str, Any]]:
    out = []
    for ev in result.evidence:
        for k, v in (ev.artifacts or {}).items():
            out.append((ev.expert, k, v))
    return out


def _find_artifact(result, *keys):
    for ev in result.evidence:
        for k in keys:
            if ev.artifacts and k in ev.artifacts and ev.artifacts[k] is not None:
                return ev.expert, ev.artifacts[k]
    return None, None


def _load_audio(result) -> Tuple[np.ndarray, int]:
    if result.ctx is not None and result.ctx.audio is not None:
        return np.asarray(result.ctx.audio, np.float32), result.ctx.sr
    try:
        from hearsay import audio as haudio
        x, _ = haudio.load(result.path)
        return x, haudio.ANALYSIS_SR
    except Exception:
        return np.zeros(0, np.float32), 16000


def _curve_from(value, duration: float) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """Parse the flexible curve formats experts use: dict(t/x, f0/sim/y), (t, y) tuple, or 1-D array."""
    meta: Dict[str, Any] = {}
    if isinstance(value, dict):
        x = _as_array(value.get("t", value.get("x", value.get("times"))))
        y = None
        for k in ("f0", "sim", "y", "values", "curve"):
            if k in value:
                y = _as_array(value[k])
                break
        meta = {k: v for k, v in value.items() if isinstance(v, (int, float, str)) or v is None}
    elif isinstance(value, (tuple, list)) and len(value) in (2, 3) and _as_array(value[0]) is not None \
            and np.ndim(value[0]) == 1 and len(value[0]) > 3:
        a0, a1 = _as_array(value[0]), _as_array(value[1])
        if a1 is not None and a1.dtype != bool and a0.size == a1.size and np.all(np.diff(a0) >= 0):
            x, y = a0, a1                       # (times, values)
        else:
            x, y = None, a0                     # e.g. librosa.pyin -> (f0, voiced_flag, voiced_prob)
    else:
        x, y = None, _as_array(value)
    if y is None or y.ndim != 1 or y.size < 2:
        return None
    if x is None or x.size != y.size:
        x = np.linspace(0, duration, y.size)
    return x, y, meta


# ------------------------------------------------------------------------------------------------
# per-file figures
# ------------------------------------------------------------------------------------------------

def _cutoff_hz(result) -> Tuple[Optional[float], str]:
    for k, v in result.features.items():
        if k.endswith("cutoff_hz") and v is not None and math.isfinite(v) and v > 0:
            return float(v), k
    bw = result.native.get("effective_bandwidth_hz")
    if bw is not None and math.isfinite(bw):
        return float(bw), "triage effective bandwidth"
    return None, ""


def _spectrogram(result, x: np.ndarray, sr: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(spectrogram in dB [freq, time], row frequencies in Hz or None for rows spread evenly over 0..sr/2).

    Prefers the linear-frequency STFT the spectral expert caches (the cut-off overlay is in linear Hz); then a
    `spectrogram_db` artifact, decoding the spectral expert's compact form (uint8 over `db_range` dB, mel rows,
    described by `spectrogram_meta`); else computes an STFT here.
    """
    mag = _as_array(result.ctx.cache.get("stft_mag")) if result.ctx is not None else None
    if mag is not None and mag.ndim == 2:
        return 20 * np.log10(np.abs(mag) + 1e-10), None
    exp, raw = _find_artifact(result, "spectrogram_db")
    spec = _as_array(raw)
    if spec is not None and spec.ndim == 2:
        ev = next((e for e in result.evidence if e.expert == exp), None)
        meta = (ev.artifacts.get("spectrogram_meta") if ev is not None else None) or {}
        if np.asarray(raw).dtype == np.uint8 and meta.get("db_range"):
            top, rng = float(meta.get("db_top", 0.0)), float(meta["db_range"])
            spec = spec / 255.0 * rng + (top - rng)
        freqs = None
        if meta.get("n_mels") and int(meta["n_mels"]) == spec.shape[0]:
            import librosa
            freqs = librosa.mel_frequencies(n_mels=spec.shape[0] + 2, fmin=float(meta.get("fmin", 0.0)),
                                            fmax=float(meta.get("fmax", sr / 2)))[1:-1]
        return spec, freqs
    from scipy.signal import stft
    _, _, Z = stft(x, fs=sr, nperseg=512, noverlap=384, boundary=None)
    return 20 * np.log10(np.abs(Z) + 1e-10), None


def fig_signal(result, x: np.ndarray, sr: int) -> Optional[str]:
    """Waveform (top) and spectrogram (bottom) with seam, digital-silence and cut-off overlays."""
    if x.size < 64:
        return None
    dur = x.size / sr
    spec, spec_hz = _spectrogram(result, x, sr)
    vmax = float(np.nanmax(spec)) if np.isfinite(spec).any() else 0.0
    # seams from every expert that reports them (splice seams, ENF hum discontinuities, ...)
    seams = []
    for ev in result.evidence:
        for s in (ev.artifacts or {}).get("seams_s") or []:
            try:
                s = float(s)
            except (TypeError, ValueError):
                continue
            if math.isfinite(s):
                seams.append(s)
    seams = sorted(set(seams))
    # digital-silence runs: splice expert uses (start_s, duration_ms, region); zero_runs_s is (start_s, end_s)
    spans = []
    for key, to_end in (("zero_runs", lambda a, b: a + b / 1000.0), ("zero_runs_s", lambda a, b: b)):
        _, runs = _find_artifact(result, key)
        for r in runs or []:
            try:
                start = float(r[0])
                spans.append((start, max(to_end(start, float(r[1])), start + 0.002)))
            except (TypeError, ValueError, IndexError):
                continue

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.08})
    # waveform as a min/max envelope (fast for long files)
    n_b = min(2000, x.size)
    edges = np.linspace(0, x.size, n_b + 1).astype(int)
    lo = np.array([x[a:b].min() if b > a else 0 for a, b in zip(edges[:-1], edges[1:])])
    hi = np.array([x[a:b].max() if b > a else 0 for a, b in zip(edges[:-1], edges[1:])])
    tt = edges[:-1] / sr
    ax1.fill_between(tt, lo, hi, color=INK2, linewidth=0, step="post")
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_ylabel("amplitude")
    ax1.set_title("Waveform and spectrogram")
    if spec_hz is None:
        ax2.imshow(spec, origin="lower", aspect="auto", cmap=SPEC_CMAP, vmin=vmax - 80, vmax=vmax,
                   extent=[0, dur, 0, sr / 2 / 1000], interpolation="nearest")
    else:  # non-uniform rows (mel bands) drawn at their true frequencies, so overlays in Hz line up
        ax2.pcolormesh(np.linspace(0, dur, spec.shape[1]), spec_hz / 1000, spec, shading="nearest",
                       cmap=SPEC_CMAP, vmin=vmax - 80, vmax=vmax)
        ax2.set_ylim(0, sr / 2 / 1000)
    ax2.grid(False)
    ax2.set_ylabel("frequency (kHz)")
    ax2.set_xlabel("time (s)")
    for s0, s1 in spans:
        for ax in (ax1, ax2):
            ax.axvspan(s0, s1, color=RED, alpha=0.18, linewidth=0)
    for i, s in enumerate(seams):
        for ax in (ax1, ax2):
            ax.axvline(s, color=RED, linestyle="--", linewidth=1.2, label="seam" if (i == 0 and ax is ax2) else None)
    cut, cut_src = _cutoff_hz(result)
    if cut is not None and cut < sr / 2 * 0.995:
        ax2.axhline(cut / 1000, color=ORANGE, linestyle="--", linewidth=1.2)
        ax2.text(dur * 0.995, cut / 1000, f"  cut-off {cut / 1000:.2f} kHz", color=ORANGE, ha="right",
                 va="bottom", fontsize=8, fontweight="bold")
    if seams:
        ax2.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white", edgecolor=GRID)
    ax2.set_xlim(0, dur)
    return _png(fig)


def fig_f0(result, duration: float) -> Optional[str]:
    exp, v = _find_artifact(result, "f0", "f0_hz", "f0_contour")
    if v is not None and not isinstance(v, dict):
        # pair a bare F0 array with the same expert's frame times when it has them (pitch: f0_t + f0_hz)
        ev = next((e for e in result.evidence if e.expert == exp), None)
        t, a = (_as_array(ev.artifacts.get("f0_t")) if ev is not None else None), _as_array(v)
        if t is not None and a is not None and t.shape == a.shape:
            v = {"t": t, "f0": a}
    if v is None and result.ctx is not None:
        v = result.ctx.cache.get("f0")
    parsed = _curve_from(v, duration) if v is not None else None
    if parsed is None:
        return None
    t, f0, _ = parsed
    f0 = f0.astype(float).copy()
    f0[~np.isfinite(f0) | (f0 <= 0)] = np.nan
    if np.all(np.isnan(f0)):
        return None
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.plot(t, f0, color=BLUE, linewidth=2)
    ax.set_title("Pitch (F0) contour: gaps are unvoiced; unnaturally flat or jumpy contours are suspicious")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_xlim(0, max(duration, float(np.nanmax(t)) if t.size else duration))
    return _png(fig)


def fig_speaker_drift(result, duration: float) -> Optional[str]:
    _, v = _find_artifact(result, "speaker_drift", "drift_curve", "speaker_similarity")
    parsed = _curve_from(v, duration) if v is not None else None
    if parsed is None:
        return None
    t, sim, meta = parsed
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.plot(t, sim, color=BLUE, linewidth=2, marker="o", markersize=4)
    thr = meta.get("drift_threshold")
    if isinstance(thr, (int, float)):
        ax.axhline(thr, color=RED, linestyle="--", linewidth=1)
        ax.text(t[0], thr, " drift threshold", color=RED, va="bottom", fontsize=8)
    ax.set_title("Speaker-identity drift: similarity of each window's voiceprint to the clip's average voice")
    ax.set_xlabel("window centre (s)")
    ax.set_ylabel("cosine similarity")
    ax.set_xlim(0, max(duration, float(t.max())))
    return _png(fig)


def fig_extra(result, duration: float) -> List[Tuple[str, str]]:
    """Generic expert plots: <name>_hz/<name>_db pairs and plot_<name> dicts."""
    figs = []
    for expert, key, v in _all_artifacts(result):
        try:
            if key.startswith("plot_") and isinstance(v, dict):
                x, y = _as_array(v.get("x")), _as_array(v.get("y"))
                if x is None or y is None or x.shape != y.shape or x.ndim != 1:
                    continue
                title = v.get("title") or f"{expert}: {key[5:]}"
                fig, ax = plt.subplots(figsize=(10, 2.2))
                ax.plot(x, y, color=BLUE, linewidth=2)
                if isinstance(v.get("hline"), (int, float)):
                    ax.axhline(v["hline"], color=RED, linestyle="--", linewidth=1)
                ax.set_xlabel(v.get("xlabel", ""))
                ax.set_ylabel(v.get("ylabel", ""))
                ax.set_title(title)
                figs.append((title, _png(fig)))
            elif key.endswith("_db"):
                base = key[:-3]
                ev = next(e for e in result.evidence if e.expert == expert)
                x, y = _as_array(ev.artifacts.get(base + "_hz")), _as_array(v)
                if x is None or y is None or x.ndim != 1 or x.shape != y.shape or x.size < 4:
                    continue
                title = f"{expert}: {base.replace('_', ' ')}"
                fig, ax = plt.subplots(figsize=(10, 2.2))
                ax.plot(x, y, color=BLUE, linewidth=1.6)
                ax.set_xlabel("frequency (Hz)")
                ax.set_ylabel("level (dB)")
                ax.set_title(title)
                figs.append((title, _png(fig)))
        except Exception:
            plt.close("all")
            continue
    return figs


def fig_balance(result) -> Optional[str]:
    """Diverging bars: summed synthetic-leaning vs authentic-leaning strength per evidence category."""
    rows = []
    for c in CATEGORIES:
        evs = [e for e in result.evidence if e.category == c and not e.error]
        if not evs:
            continue
        syn = sum(f.strength for e in evs for f in e.findings if f.direction > 0)
        real = sum(f.strength for e in evs for f in e.findings if f.direction < 0)
        rows.append((CATEGORY_LABELS.get(c, (c,))[0], syn, real))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(rows) + 0.9))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=RED, height=0.6, label="suggests synthetic")
    ax.barh(y, [-r[2] for r in rows], color=BLUE, height=0.6, label="suggests genuine")
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y, [r[0] for r in rows])
    lim = max(1.0, max(max(r[1], r[2]) for r in rows) * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("summed finding strength  (genuine <-  |  -> synthetic)")
    ax.set_title("Evidence balance by category")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    return _png(fig)


# ------------------------------------------------------------------------------------------------
# per-file report
# ------------------------------------------------------------------------------------------------

def _short_feature(name: str, ev: Evidence) -> str:
    """Drop the redundant expert/category prefix (e.g. 'spectral_cutoff_hz' -> 'cutoff_hz')."""
    for prefix in (ev.expert + "_", ev.category + "_", ev.expert.split("_")[0] + "_"):
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix):]
    return name


def _key_values(ev: Evidence, k: int = 4) -> List[Tuple[str, str]]:
    return [(_short_feature(n, ev), fmt(v)) for n, v in list(ev.features.items())[:k]]


def _why_not_run(name: str, result) -> str:
    spec = REGISTRY.get(name)
    for entry in result.trace:
        for s in entry.get("experts_suppressed", []) or []:
            if s.startswith(name + ":"):
                return "suppressed: " + s.split(":", 1)[1].strip()
    for rule in orch.POLICY:
        claims = name in rule.get("experts", [])
        sweeps = spec is not None and rule.get("cost") == spec.cost and name not in orch._claimed_names(orch.POLICY)
        if claims or sweeps:
            entry = next((e for e in result.trace if e.get("rule") == rule["id"]), None)
            if entry is None:
                return f"rule '{rule['id']}' was not evaluated"
            if not entry.get("fired"):
                return f"rule '{rule['id']}' did not fire: {entry.get('reason', '')}"
            return f"rule '{rule['id']}' fired but the expert was not run"
    return "not covered by any routing rule"


def _context(result, cfg: Optional[orch.RouterConfig] = None) -> Dict[str, Any]:
    cfg = cfg or orch.RouterConfig()
    x, sr = _load_audio(result)
    dur = x.size / sr if x.size else float(result.native.get("duration_s") or 0.0)
    plots = {}
    for key, fn in (("signal", lambda: fig_signal(result, x, sr)), ("f0", lambda: fig_f0(result, dur)),
                    ("drift", lambda: fig_speaker_drift(result, dur)), ("balance", lambda: fig_balance(result))):
        try:
            plots[key] = fn()
        except Exception as e:  # a plot must never break the report
            plt.close("all")
            plots[key] = None
            plots.setdefault("errors", []).append(f"{key}: {type(e).__name__}: {e}")
    extra = fig_extra(result, dur)

    groups = []
    for c in list(CATEGORIES) + sorted({e.category for e in result.evidence} - set(CATEGORIES)):
        evs = [e for e in result.evidence if e.category == c]
        if not evs:
            continue
        rows = []
        for ev in evs:
            spec = REGISTRY.get(ev.expert)
            findings = [f for f in ev.findings]
            base = {"expert": ev.expert, "cost": spec.cost if spec else "injected",
                    "description": (spec.description if spec else "") or "", "key_values": _key_values(ev),
                    "all_features": [(_short_feature(k, ev), fmt(v)) for k, v in ev.features.items()],
                    "runtime": ev.runtime_ms, "error": ev.error, "applicable": ev.applicable, "n": max(1, len(findings))}
            if ev.error:
                rows.append(dict(base, text=f"Expert failed: {ev.error}", direction=0, strength=0.0, first=True))
                continue
            if not findings:
                rows.append(dict(base, text="No finding reported.", direction=0, strength=0.0, first=True))
                continue
            for i, f in enumerate(sorted(findings, key=lambda f: -float(f.strength or 0))):
                rows.append(dict(base, text=f.text, direction=int(np.sign(f.direction)),
                                 strength=float(np.clip(f.strength or 0.0, 0, 1)), first=(i == 0)))
        label, blurb = CATEGORY_LABELS.get(c, (c, ""))
        groups.append({"category": c, "label": label, "blurb": blurb, "rows": rows})

    ran = {e.expert for e in result.evidence}
    neutral = [(e.expert, f.text) for e in result.evidence if not e.error for f in e.findings
               if int(np.sign(f.direction)) == 0 or float(f.strength or 0) < 0.05]
    silent = [e.expert for e in result.evidence if not e.error and not e.findings]
    failed = [(e.expert, e.error) for e in result.evidence if e.error]
    partial = [(e.expert, "; ".join(map(str, e.artifacts["partial_errors"]))) for e in result.evidence
               if not e.error and e.artifacts and e.artifacts.get("partial_errors")]
    not_run = [(n, _why_not_run(n, result)) for n in REGISTRY if n not in ran]
    missing = sorted({m for t in result.trace for m in (t.get("experts_missing") or [])})

    n = result.native
    custody = [
        ("File", n.get("filename")), ("SHA-256", n.get("sha256")), ("Size", f"{n.get('size_bytes', 0):,} bytes"),
        ("Modified (UTC)", n.get("modified_utc")), ("Container", n.get("container_long") or n.get("container")),
        ("Codec", n.get("codec_long") or n.get("codec")), ("Lossy codec", fmt(n.get("is_lossy"))),
        ("Native sample rate", f"{n.get('native_sr')} Hz" if n.get("native_sr") else "-"),
        ("Channels", n.get("channel_layout") or n.get("channels")),
        ("Bit depth / format", f"{n.get('bits_per_sample') or '-'} / {n.get('sample_fmt') or '-'}"),
        ("Bit rate", f"{fmt((n.get('bitrate') or 0) / 1000)} kb/s" if n.get("bitrate") else "-"),
        ("Duration", f"{fmt(n.get('duration_s'))} s"), ("Encoder tag", n.get("encoder") or "none"),
        ("Extension matches container", "NO - mislabelled" if n.get("extension_mismatch") else "yes"),
        ("Effective bandwidth", f"{fmt(n.get('effective_bandwidth_hz'))} Hz (measured at {n.get('bandwidth_measured_at_hz')} Hz)"),
        ("Band-limited", fmt(n.get("band_limited"))), ("Peak / RMS", f"{fmt(n.get('peak'))} / {fmt(n.get('rms_dbfs'))} dBFS"),
        ("Clipped samples", f"{100 * (n.get('clip_fraction') or 0):.2f}%"),
    ]
    other_tags = [(k, v) for k, v in (n.get("tags") or {}).items() if not k.endswith(".encoder")]
    if n.get("decode_error"):
        custody.append(("DECODE ERROR", n["decode_error"]))

    cm = float(result.cm_score)
    verdict = ("Likely synthetic" if cm >= cfg.synthetic_above else
               "Likely genuine" if cm <= cfg.bona_fide_below else "Inconclusive")
    verdict_cls = "syn" if cm >= cfg.synthetic_above else "real" if cm <= cfg.bona_fide_below else "unc"
    return {
        "r": result, "cm": cm, "verdict": verdict, "verdict_cls": verdict_cls, "plots": plots, "extra": extra,
        "groups": groups, "neutral": neutral, "silent": silent, "failed": failed, "partial": partial,
        "not_run": not_run,
        "missing": missing, "custody": custody, "other_tags": other_tags, "cfg": cfg,
        "attribution_rules": orch.ATTRIBUTION_RULES, "features": sorted(result.features.items()),
        "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "version": orch.ROUTER_VERSION,
        "n_registered": len(REGISTRY), "fmt": fmt,
    }


_ENV = Environment(loader=BaseLoader(), autoescape=select_autoescape(default=True, default_for_string=True),
                   trim_blocks=True, lstrip_blocks=True)
_ENV.filters["fmt"] = fmt

_CSS = """
:root { --surface:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8986; --line:#e4e3df;
        --blue:#2a78d6; --red:#e34948; --orange:#eb6834; --neutral:#b9b8b3; --chip:#f0efec; }
* { box-sizing:border-box; }
body { margin:0; background:var(--surface); color:var(--ink); font:14px/1.5 -apple-system, BlinkMacSystemFont,
       "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
main { max-width:1120px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:28px 0 10px; } h3 { font-size:14px; margin:0 0 6px; }
.sub { color:var(--ink2); font-size:13px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin:12px 0; }
.grid2 { display:grid; grid-template-columns: 1fr 1fr; gap:12px; } @media (max-width:820px){ .grid2 { grid-template-columns:1fr; } }
.verdict { display:flex; gap:22px; align-items:center; flex-wrap:wrap; }
.big { font-size:40px; font-weight:700; font-variant-numeric:tabular-nums; line-height:1; }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600; background:var(--chip); color:var(--ink); }
.syn { color:var(--red); } .real { color:var(--blue); } .unc { color:var(--ink2); }
.gauge { position:relative; height:14px; border-radius:7px; min-width:260px; flex:1;
         background:linear-gradient(90deg, var(--blue) 0%, #f0efec 50%, var(--red) 100%); }
.gauge .mark { position:absolute; top:-5px; width:4px; height:24px; background:var(--ink); border-radius:2px; transform:translateX(-2px); }
.gauge-l { display:flex; justify-content:space-between; font-size:11px; color:var(--ink2); margin-top:4px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--ink2); font-weight:600; font-size:12px; background:#f7f7f5; }
td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.kv td:first-child { color:var(--ink2); width:42%; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; word-break:break-all; }
.cat { background:#f7f7f5; font-weight:600; } .cat span { font-weight:400; color:var(--ink2); font-size:12px; }
.dir { white-space:nowrap; font-weight:600; font-size:12px; }
.bar { height:8px; background:var(--chip); border-radius:4px; width:90px; overflow:hidden; }
.bar > div { height:8px; border-radius:4px; }
.expert { font-weight:600; } .desc { color:var(--muted); font-size:11.5px; }
.chips span { display:inline-block; background:var(--chip); border-radius:6px; padding:1px 6px; margin:1px 2px 1px 0; font-size:11.5px; }
.chips b { font-weight:600; }
img.plot { width:100%; height:auto; display:block; }
.fired { color:var(--red); font-weight:600; } .notfired { color:var(--muted); }
details summary { cursor:pointer; color:var(--ink2); font-size:13px; }
ul.tight { margin:4px 0; padding-left:20px; } ul.tight li { margin:2px 0; }
.err { color:var(--red); }
footer { color:var(--muted); font-size:12px; margin-top:32px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:12px; }
.tile .v { font-size:26px; font-weight:700; font-variant-numeric:tabular-nums; } .tile .l { color:var(--ink2); font-size:12px; }
th.sortable { cursor:pointer; } th.sortable:after { content:" \\2195"; color:var(--muted); }
"""

_FILE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HEARSAY report - {{ r.name }}</title><style>{{ css | safe }}</style></head>
<body><main>
<h1>Forensic audio report: {{ r.name }}</h1>
<div class="sub">HEARSAY router v{{ version }} &middot; {{ r.mode }} mode &middot; generated {{ generated }} &middot; total analysis {{ r.runtime_ms | fmt }} ms</div>

<div class="card">
  <div class="verdict">
    <div><div class="big {{ verdict_cls }}">{{ '%.2f' % cm }}</div><div class="sub">cm-score (1.0 = synthetic)</div></div>
    <div style="flex:1; min-width:260px">
      <div class="gauge"><div class="mark" style="left: {{ (cm * 100) | round(1) }}%"></div></div>
      <div class="gauge-l"><span>0 genuine</span><span>{{ cfg.bona_fide_below }} | {{ cfg.synthetic_above }}</span><span>1 synthetic</span></div>
    </div>
    <div><div class="{{ verdict_cls }}" style="font-size:18px; font-weight:700">{{ verdict }}</div>
      <span class="pill">{{ r.manipulation_type }}</span></div>
  </div>
  <p style="margin:14px 0 6px">{{ r.summary }}</p>
  {% if r.attribution_reasons %}<div class="sub">Why this manipulation type: {{ r.attribution_reasons | join('; ') }}.</div>{% endif %}
  {% if r.error %}<p class="err">Pipeline note: {{ r.error }}</p>{% endif %}
</div>

<div class="grid2">
  <div class="card"><h3>Chain of custody</h3>
    <table class="kv">{% for k, v in custody %}<tr><td>{{ k }}</td><td{% if k == 'SHA-256' %} class="mono"{% endif %}>{{ v if v is not none else '-' }}</td></tr>{% endfor %}
    {% for k, v in other_tags %}<tr><td>tag {{ k }}</td><td class="mono">{{ v }}</td></tr>{% endfor %}</table>
  </div>
  <div class="card">{% if plots.balance %}<img class="plot" alt="Evidence balance by category" src="{{ plots.balance }}">{% else %}<h3>Evidence balance</h3><p class="sub">No directional findings.</p>{% endif %}
    <p class="sub">Each bar sums the strength of findings in that family. Red pushes toward synthetic, blue toward genuine. Score source: <b>{{ r.score_source }}</b>.</p>
  </div>
</div>

<h2>Signal</h2>
<div class="card">
  {% if plots.signal %}<img class="plot" alt="Waveform and spectrogram" src="{{ plots.signal }}">
  <p class="sub">Red dashed lines mark seams reported by splice/environment experts; red shading marks digital-silence runs; the orange line is the measured frequency cut-off (a codec or resampler low-pass). Genuine microphone speech usually shows smooth, noisy energy up to the cut-off; neural vocoders can leave over-regular harmonics, band gaps or an unnaturally clean floor.</p>
  {% else %}<p class="sub">No decodable audio to plot.</p>{% endif %}
  {% if plots.f0 %}<img class="plot" alt="F0 contour" src="{{ plots.f0 }}">{% endif %}
  {% if plots.drift %}<img class="plot" alt="Speaker drift" src="{{ plots.drift }}">{% endif %}
  {% for title, src in extra %}<img class="plot" alt="{{ title }}" src="{{ src }}">{% endfor %}
  {% if plots.errors %}<p class="err">Plot errors: {{ plots.errors | join('; ') }}</p>{% endif %}
</div>

<h2>Evidence by category</h2>
<div class="card" style="padding:0; overflow-x:auto">
<table>
<tr><th>Expert</th><th>Key measurements</th><th>Finding</th><th>Direction</th><th>Strength</th><th class="num">ms</th></tr>
{% for g in groups %}
<tr><td colspan="6" class="cat">{{ g.label }} <span>&middot; {{ g.blurb }}</span></td></tr>
{% for row in g.rows %}
<tr>
  {% if row.first %}<td rowspan="{{ row.n }}"><div class="expert">{{ row.expert }}</div><div class="desc">{{ row.cost }}{% if not row.applicable %} &middot; not applicable{% endif %}</div></td>
  <td rowspan="{{ row.n }}" class="chips">{% for k, v in row.key_values %}<span>{{ k }} <b>{{ v }}</b></span>{% endfor %}
    {% if row.all_features|length > row.key_values|length %}<details><summary>all {{ row.all_features|length }}</summary>{% for k, v in row.all_features %}<span>{{ k }} <b>{{ v }}</b></span>{% endfor %}</details>{% endif %}</td>{% endif %}
  <td{% if row.error %} class="err"{% endif %}>{{ row.text }}</td>
  <td class="dir">{% if row.direction > 0 %}<span class="syn">&#9650; synthetic</span>{% elif row.direction < 0 %}<span class="real">&#9660; genuine</span>{% else %}<span class="unc">&#9679; neutral</span>{% endif %}</td>
  <td><div class="bar" title="{{ '%.2f' % row.strength }}"><div style="width: {{ (row.strength * 100) | round(0) }}%; background: {{ 'var(--red)' if row.direction > 0 else ('var(--blue)' if row.direction < 0 else 'var(--neutral)') }}"></div></div></td>
  {% if row.first %}<td rowspan="{{ row.n }}" class="num">{{ row.runtime | fmt }}</td>{% endif %}
</tr>
{% endfor %}{% endfor %}
</table></div>

<h2>Routing trace</h2>
<div class="card" style="padding:0; overflow-x:auto">
<table>
<tr><th>#</th><th>Tier</th><th>Rule</th><th>Decision</th><th>Reason</th><th>Experts run</th><th class="num">ms</th></tr>
{% for t in r.trace %}
<tr><td>{{ t.step }}</td><td>{{ t.tier }}</td><td>{{ t.rule }}</td>
<td>{% if t.fired and (t.experts_run or t.tier == 'triage') %}<span class="fired">ran</span>{% elif t.fired %}<span class="notfired">fired, nothing to run</span>{% else %}<span class="notfired">skipped</span>{% endif %}</td>
<td>{{ t.reason }}{% if t.experts_missing %}<div class="desc">not installed: {{ t.experts_missing | join(', ') }}</div>{% endif %}{% if t.experts_suppressed %}<div class="desc">suppressed: {{ t.experts_suppressed | join('; ') }}</div>{% endif %}</td>
<td>{{ (t.experts_run or []) | join(', ') or '-' }}</td><td class="num">{{ t.ms | fmt }}</td></tr>
{% endfor %}
</table></div>

<h2>Evidence that had no effect</h2>
<div class="card">
  {% if neutral %}<h3>Neutral findings</h3><ul class="tight">{% for e, t in neutral %}<li><b>{{ e }}</b>: {{ t }}</li>{% endfor %}</ul>{% endif %}
  {% if silent %}<h3>Ran without a finding</h3><p>{{ silent | join(', ') }}</p>{% endif %}
  {% if failed %}<h3>Failed (excluded from the verdict)</h3><ul class="tight">{% for e, t in failed %}<li><b>{{ e }}</b>: <span class="err">{{ t }}</span></li>{% endfor %}</ul>{% endif %}
  {% if partial %}<h3>Partly failed (the other measurements of these experts were kept)</h3><ul class="tight">{% for e, t in partial %}<li><b>{{ e }}</b>: <span class="err">{{ t }}</span></li>{% endfor %}</ul>{% endif %}
  {% if not_run %}<h3>Available but not run</h3><ul class="tight">{% for e, t in not_run %}<li><b>{{ e }}</b>: {{ t }}</li>{% endfor %}</ul>{% endif %}
  {% if missing %}<h3>Named in the routing policy but not installed</h3><p>{{ missing | join(', ') }}</p>{% endif %}
  {% if not (neutral or silent or failed or partial or not_run or missing) %}<p class="sub">Every analysis that ran contributed a directional finding.</p>{% endif %}
</div>

<details class="card"><summary>How the manipulation type is chosen (rules, first match wins)</summary>
<ol>{% for label, rule in attribution_rules %}<li><b>{{ label }}</b>: {{ rule }}</li>{% endfor %}</ol></details>
<details class="card"><summary>All {{ features|length }} numeric features passed to fusion</summary>
<table>{% for k, v in features %}<tr><td class="mono">{{ k }}</td><td class="num">{{ v | fmt }}</td></tr>{% endfor %}</table></details>

<footer>cm-score semantics follow the HEARSAY submission format (1.0 = synthetic). Findings are written by individual
experts; their direction and strength are the expert's own assessment. Registered experts: {{ n_registered }}.</footer>
</main></body></html>
"""


def render_file_report(result, out_path: Optional[str] = None, config: Optional[orch.RouterConfig] = None) -> str:
    """Render one FileResult to a self-contained HTML string (and write it when out_path is given)."""
    ctx = _context(result, config)
    page = _ENV.from_string(_FILE_TEMPLATE).render(css=_CSS, **ctx)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(page, encoding="utf-8")
    return page


def report_filename(result) -> str:
    stem = Path(result.path).stem
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)
    return f"{safe}.html"


def make_report_writer(out_dir: str, config: Optional[orch.RouterConfig] = None):
    """on_result callback for orchestrator.run_batch: writes <out_dir>/files/<stem>.html per file."""
    out = Path(out_dir) / "files"
    out.mkdir(parents=True, exist_ok=True)

    def _write(result):
        render_file_report(result, str(out / report_filename(result)), config)

    return _write


# ------------------------------------------------------------------------------------------------
# batch dashboard
# ------------------------------------------------------------------------------------------------

def fig_histogram(scores: Sequence[float], cfg: orch.RouterConfig) -> Optional[str]:
    s = np.asarray([v for v in scores if v is not None and math.isfinite(v)], float)
    if s.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.hist(s, bins=np.linspace(0, 1, 41), color=BLUE, edgecolor="white", linewidth=1.5)
    for thr in (cfg.bona_fide_below, cfg.synthetic_above):
        ax.axvline(thr, color=INK2, linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("cm-score (1.0 = synthetic); dashed = genuine / synthetic thresholds")
    ax.set_ylabel("files")
    ax.set_title("Score distribution")
    ax.grid(axis="x", visible=False)
    return _png(fig)


def fig_category_flags(results) -> Optional[str]:
    rows = []
    for c in CATEGORIES:
        syn = sum(any(e.category == c and not e.error and f.direction > 0 and f.strength >= 0.5
                      for e in r.evidence for f in e.findings) for r in results)
        real = sum(any(e.category == c and not e.error and f.direction < 0 and f.strength >= 0.5
                       for e in r.evidence for f in e.findings) for r in results)
        ran = sum(any(e.category == c for e in r.evidence) for r in results)
        if ran:
            rows.append((CATEGORY_LABELS.get(c, (c,))[0], syn, real))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6, 0.42 * len(rows) + 0.9))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=RED, height=0.6, label="files with a synthetic flag (>= 0.5)")
    ax.barh(y, [-r[2] for r in rows], color=BLUE, height=0.6, label="files with a genuine flag (>= 0.5)")
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y, [r[0] for r in rows])
    lim = max(1, max(max(r[1], r[2]) for r in rows)) * 1.15
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    ax.set_title("Strong flags per evidence category")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    return _png(fig)


def fig_expert_ms(acc: Dict[str, Any]) -> Optional[str]:
    items = list(acc.get("expert_total_ms", {}).items())
    if not items:
        return None
    items = sorted(items, key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(items) + 0.9))
    ax.barh(np.arange(len(items)), [v / 1000 for _, v in items], color=BLUE, height=0.6)
    ax.set_yticks(np.arange(len(items)), [k for k, _ in items])
    runs = acc.get("expert_runs", {})
    for i, (k, v) in enumerate(items):
        n_runs = runs.get(k, 0)
        ax.text(v / 1000, i, f"  {n_runs} run{'s' if n_runs != 1 else ''}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlabel("total seconds in this batch")
    ax.set_title("Where the compute went")
    ax.grid(axis="y", visible=False)
    return _png(fig)


_INDEX_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HEARSAY batch dashboard</title><style>{{ css | safe }}</style></head>
<body><main>
<h1>{{ title }}</h1>
<div class="sub">{{ n }} files &middot; {{ mode }} mode &middot; router v{{ version }} &middot; generated {{ generated }}{% if wall_s %} &middot; wall time {{ wall_s | fmt }} s{% endif %}</div>

<div class="tiles" style="margin-top:14px">
  <div class="card tile"><div class="v">{{ n }}</div><div class="l">files analysed</div></div>
  <div class="card tile"><div class="v syn">{{ n_syn }}</div><div class="l">likely synthetic (cm &ge; {{ cfg.synthetic_above }})</div></div>
  <div class="card tile"><div class="v real">{{ n_real }}</div><div class="l">likely genuine (cm &le; {{ cfg.bona_fide_below }})</div></div>
  <div class="card tile"><div class="v">{{ n_unc }}</div><div class="l">inconclusive</div></div>
  <div class="card tile"><div class="v">{{ median_ms | fmt }}</div><div class="l">median ms per file</div></div>
  <div class="card tile"><div class="v">{% if acc.agentic_fraction_of_full is not none %}{{ (acc.agentic_fraction_of_full * 100) | round(0) | int }}%{% else %}-{% endif %}</div><div class="l">analysis compute vs. running every expert on every file{% if mode == 'full' %} (full mode){% endif %}</div></div>
</div>

<div class="grid2">
  <div class="card">{% if hist %}<img class="plot" alt="Score histogram" src="{{ hist }}">{% endif %}</div>
  <div class="card">{% if flags %}<img class="plot" alt="Flags per category" src="{{ flags }}">{% endif %}</div>
</div>

<div class="grid2">
  <div class="card"><h3>Attributed manipulation type</h3>
    <table>{% for label, count in types %}<tr><td>{{ label }}</td><td class="num">{{ count }}</td><td class="num">{{ (100 * count / n) | round(1) }}%</td></tr>{% endfor %}</table></div>
  <div class="card"><h3>Agentic routing: files reaching each tier</h3>
    <table>{% for tier, count in tiers %}<tr><td>{{ tier }}</td><td class="num">{{ count }}</td><td class="num">{{ (100 * count / n) | round(1) }}%</td></tr>{% endfor %}</table>
    <p class="sub">Analysis time {{ (acc.analysis_total_ms / 1000) | fmt }} s; running every expert on every file is estimated at
    {{ (acc.full_estimate_analysis_ms / 1000) | fmt }} s (mean measured cost per expert x files).
    {% if acc.unpriced_experts %}Never-run experts are not priced: {{ acc.unpriced_experts | join(', ') }}.{% endif %}</p></div>
</div>

<div class="card">{% if ms %}<img class="plot" alt="Compute per expert" src="{{ ms }}" style="max-width:640px">{% endif %}</div>

{% if corpus_log %}<div class="card"><h3>Batch-level corpus steps</h3><table>
{% for c in corpus_log %}<tr><td>{{ c.module }}</td><td>{{ 'ok' if c.ok else 'skipped' }}</td><td>{{ c.reason }}</td><td class="num">{{ c.ms | fmt }} ms</td></tr>{% endfor %}
</table></div>{% endif %}

<h2>Files</h2>
<div class="card" style="padding:0; overflow-x:auto">
<table id="files">
<thead><tr><th class="sortable" data-k="s">File</th><th class="sortable" data-k="n">cm-score</th><th class="sortable" data-k="s">Type</th><th class="sortable" data-k="s">Tiers</th><th>Top finding</th><th class="sortable num" data-k="n">ms</th></tr></thead>
<tbody>
{% for f in files %}<tr>
<td data-v="{{ f.name }}">{% if f.link %}<a href="{{ f.link }}">{{ f.name }}</a>{% else %}{{ f.name }}{% endif %}</td>
<td data-v="{{ f.cm }}" style="white-space:nowrap"><span class="{{ f.cls }}" style="font-variant-numeric:tabular-nums; font-weight:600">{{ '%.3f' % f.cm }}</span>
  <span class="bar" style="display:inline-block; vertical-align:middle; margin-left:6px"><div style="width:{{ (f.cm*100)|round(0) }}%; background:{{ 'var(--red)' if f.cm >= cfg.synthetic_above else ('var(--blue)' if f.cm <= cfg.bona_fide_below else 'var(--neutral)') }}"></div></span></td>
<td data-v="{{ f.type }}">{{ f.type }}</td><td data-v="{{ f.tiers }}">{{ f.tiers }}</td>
<td style="font-size:12px">{{ f.top }}</td><td class="num" data-v="{{ f.ms }}">{{ f.ms | fmt }}</td></tr>
{% endfor %}</tbody></table></div>

<footer>Each file links to a self-contained forensic report. Click a column header to sort.</footer>
</main>
<script>
document.querySelectorAll('#files th.sortable').forEach(function (th, idx) {
  var asc = true;
  th.addEventListener('click', function () {
    var col = Array.prototype.indexOf.call(th.parentNode.children, th);
    var body = document.querySelector('#files tbody');
    var rows = Array.prototype.slice.call(body.rows);
    var num = th.getAttribute('data-k') === 'n';
    rows.sort(function (a, b) {
      var x = a.cells[col].getAttribute('data-v'), y = b.cells[col].getAttribute('data-v');
      if (num) { x = parseFloat(x); y = parseFloat(y); return asc ? x - y : y - x; }
      return asc ? x.localeCompare(y) : y.localeCompare(x);
    });
    asc = !asc;
    rows.forEach(function (r) { body.appendChild(r); });
  });
});
</script>
</body></html>
"""


def render_index(results, out_dir: str, accounting: Optional[Dict[str, Any]] = None,
                 title: str = "HEARSAY batch forensic dashboard", config: Optional[orch.RouterConfig] = None,
                 link_reports: bool = True) -> str:
    """Write <out_dir>/index.html summarising a batch (scores, flags, routing, compute, file table)."""
    cfg = config or orch.RouterConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    acc = accounting or orch.compute_accounting(results)
    scores = [r.cm_score for r in results]
    types: Dict[str, int] = {}
    for r in results:
        types[r.manipulation_type] = types.get(r.manipulation_type, 0) + 1
    files = []
    for r in results:
        ranked = sorted(((e, f) for e in r.evidence if not e.error for f in e.findings if f.direction != 0),
                        key=lambda t: -float(t[1].strength or 0))
        top = f"{ranked[0][0].expert}: {orch._short(ranked[0][1].text, 120)}" if ranked else "-"
        link = f"files/{report_filename(r)}"
        cls = "syn" if r.cm_score >= cfg.synthetic_above else "real" if r.cm_score <= cfg.bona_fide_below else "unc"
        files.append({"name": r.name, "cm": float(r.cm_score), "type": r.manipulation_type, "cls": cls,
                      "tiers": "+".join(t for t in r.tiers if t != "triage") or "-", "top": top,
                      "ms": r.runtime_ms, "link": link if link_reports and (out / link).exists() else None})
    ctx = {
        "css": _CSS, "title": title, "n": len(results), "mode": getattr(results, "mode", None) or
        (results[0].mode if results else "-"), "version": orch.ROUTER_VERSION,
        "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "wall_s": getattr(results, "wall_s", None),
        "cfg": cfg, "acc": acc,
        "n_syn": sum(s >= cfg.synthetic_above for s in scores), "n_real": sum(s <= cfg.bona_fide_below for s in scores),
        "n_unc": sum(cfg.bona_fide_below < s < cfg.synthetic_above for s in scores),
        "median_ms": float(np.median([r.runtime_ms for r in results])) if results else 0.0,
        "hist": fig_histogram(scores, cfg), "flags": fig_category_flags(results), "ms": fig_expert_ms(acc),
        "types": sorted(types.items(), key=lambda kv: -kv[1]),
        "tiers": [(t, acc["files_per_tier"].get(t, 0)) for t in orch.TIER_ORDER],
        "corpus_log": getattr(results, "corpus_log", []), "files": sorted(files, key=lambda f: -f["cm"]),
    }
    page = _ENV.from_string(_INDEX_TEMPLATE).render(**ctx) if results else "<p>No files.</p>"
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    return str(path)
