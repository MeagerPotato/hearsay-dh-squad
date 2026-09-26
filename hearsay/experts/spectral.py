"""Spectral forensics: frequency-domain traces that separate recorded speech from synthesised speech.

Three experts share one set of per-file intermediates (computed once, cached in ``ctx.cache``):

* ``spectral_stats``    (cheap)  long-term spectral shape - band-energy balance, effective bandwidth,
  per-band flatness, centroid / roll-off, spectral flux and how smoothly sub-band energies evolve
  over time. Also renders the log-mel spectrogram used by the HTML report.
* ``lfcc_stats``        (cheap)  20 linear-frequency cepstral coefficients (+ deltas) summarised as
  mean/std: the ASVspoof LFCC-GMM baseline front end, kept as a vector for a downstream model.
* ``vocoder_artifacts`` (medium) traces that neural / statistical vocoders leave behind: harmonic-to-
  noise regularity, harmonic phase coherence and group-delay irregularity, narrow spectral lines
  from upsampling layers, frame-rate lines in the modulation spectrum, and low-frequency harmonic
  sharpness measured on a constant-Q transform.

Cache keys. This module relies only on its own namespaced keys (``spectral/base``, ``spectral/f0``,
``spectral/voiced``) so that another expert storing a same-named intermediate with different
parameters can never corrupt these features. For other experts it *publishes*, only when absent:

  ``stft_mag``  float32 [n_fft/2+1, frames] magnitude STFT (Hann, 25 ms window, 10 ms hop, centred)
  ``stft_meta`` dict(sr, n_fft, hop, win, window, center) describing ``stft_mag``
  ``mel_db``    float32 [80, frames] log-mel power spectrogram (dB) on the same frame grid
  ``lfcc``      float32 [frames, 60] LFCC + delta + delta-delta (non-silent frames only)

Feature conventions: every feature is a float prefixed by its expert name (``spectral_``, ``lfcc_``,
``vocoder_``); the key set is identical for every file, and a quantity that cannot be measured
(no speech, no voiced frames, sub-analysis failure) is NaN rather than a made-up default.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sps
from scipy.fft import dct
from scipy.ndimage import median_filter, uniform_filter1d

from ..evidence import Context, Evidence, Finding, expert

NAN = float("nan")
MAX_VOICED_FRAMES = 600    # cap on voiced frames analysed in detail (keeps long files bounded)
MAX_CQT_SECONDS = 12.0     # CQT is the costliest step; analyse at most this much audio
MAX_SPEC_FRAMES = 2000     # cap on spectrogram columns saved for the report

# ----------------------------------------------------------------------------------------------
# Reference values for NATURAL speech, used only to decide when a finding is worth reporting (the
# fusion model consumes the raw features). Calibrated on 16 kHz organiser-pipeline audio: 57
# LJSpeech clips (real) and 60 clips from the lower half of the challenge test set by AASIST score
# (scripts in work/expert_tests/spectral/). (lo, hi) ~ the 0-2nd .. 98-100th percentile of those
# real-ish clips. Because false alarms on real audio cost 4x, synthetic-leaning findings fire only
# beyond that range, or when several independent cues agree (the *_RULE tables below).
# ----------------------------------------------------------------------------------------------
REF = {
    "hf_ratio_db": (-33.0, -3.0),         # 4-8 kHz energy relative to 0-4 kHz
    "lf_ratio_db": (-53.0, -9.0),         # 20-80 Hz energy relative to 80-4000 Hz
    "harm_hi": (0.13, 0.42),              # spectral-ACF harmonicity of the 4-7 kHz band
    "tonal_db": (6.0, 18.0),              # strongest narrow spectral line above its +-20 Hz surroundings
    "modline_db": (6.0, 12.5),            # strongest line in the 1-7 kHz envelope modulation spectrum
    "cqt_contrast_db": (19.0, 36.0),      # low-harmonic peak-to-valley contrast on the CQT
    "rps_coh_lo": (0.58, 0.92),           # frame-to-frame stability of harmonic phase (h2-h4)
}
# "Over-smoothed" spectral dynamics: value below threshold counts as one vote; >= 3 votes -> finding.
# Hit rate on the calibration data: real 0-2 %, AASIST-flagged test clips 12 %, DiffSSD 3 %.
SMOOTH_RULE = {"spectral_flux_mean_db": 7.45, "spectral_roughness_lo_db": 3.6, "spectral_roughness_mid_db": 3.6,
               "spectral_roughness_hi_db": 3.6, "spectral_tvar_hi_db": 11.5}
SMOOTH_MIN_VOTES = 3
# "Over-regular" voice quality: >= 2 votes -> finding. Real 0-2 %, AASIST-flagged 19 %, DiffSSD 0 %.
REGULAR_RULE = {"vocoder_cpp_std_db": 3.4, "vocoder_cqt_contrast_std_db": 5.5, "vocoder_hnr_jump_db": 1.75,
                "vocoder_harm_hi_std": 0.105}
REGULAR_MIN_VOTES = 2
# Naturally variable voice quality (all three above threshold) -> weak real-leaning finding.
NATURAL_RULE = {"vocoder_cpp_std_db": 5.0, "vocoder_cqt_contrast_std_db": 9.0, "vocoder_hnr_jump_db": 2.4}

# ----------------------------------------------------------------------------------------------
# Feature name lists (fixed column sets for fusion)
# ----------------------------------------------------------------------------------------------
STATS_KEYS = [
    "spectral_hf_ratio_db", "spectral_b4_6k_db", "spectral_b6_8k_db", "spectral_lf_ratio_db",
    "spectral_cutoff_hz", "spectral_cutoff_drop_db",
    "spectral_flatness_lo_db", "spectral_flatness_mid_db", "spectral_flatness_hi_db",
    "spectral_centroid_mean_hz", "spectral_centroid_std_hz", "spectral_rolloff_mean_hz", "spectral_rolloff_std_hz",
    "spectral_flux_mean_db", "spectral_flux_cv",
    "spectral_roughness_lo_db", "spectral_roughness_mid_db", "spectral_roughness_hi_db",
    "spectral_tvar_hi_db", "spectral_hf_voiced_ratio_db", "spectral_speech_frac",
]
N_LFCC = 20
LFCC_KEYS = ([f"lfcc_mean_{i}" for i in range(N_LFCC)] + [f"lfcc_std_{i}" for i in range(N_LFCC)]
             + [f"lfcc_dstd_{i}" for i in range(N_LFCC)] + [f"lfcc_ddstd_{i}" for i in range(N_LFCC)])
VOCODER_KEYS = [
    "vocoder_voiced_frac",
    # (1) harmonic-to-noise structure and its regularity
    "vocoder_hnr_mean_db", "vocoder_hnr_std_db", "vocoder_hnr_jump_db",
    "vocoder_cpp_mean_db", "vocoder_cpp_std_db",
    "vocoder_harm_lo", "vocoder_harm_mid", "vocoder_harm_hi", "vocoder_harm_hi_std",
    # (2) phase coherence / group delay
    "vocoder_rps_coh_lo", "vocoder_rps_coh_hi", "vocoder_gd_spike_rate", "vocoder_mgd_std", "vocoder_mgd_tcorr",
    # (3) upsampling / frame-rate artefacts
    "vocoder_tonal_max_db", "vocoder_tonal_hz", "vocoder_tonal_count", "vocoder_grid_excess_db",
    "vocoder_modline_db", "vocoder_modline_hz",
    # (4) CQT low-frequency harmonic sharpness
    "vocoder_cqt_contrast_db", "vocoder_cqt_contrast_std_db", "vocoder_cqt_contrast_slope_db", "vocoder_cqt_peakiness_db",
]


def _nan_features(keys: List[str]) -> Dict[str, float]:
    return {k: NAN for k in keys}


def _f(x) -> float:
    """Coerce to a plain Python float (NaN for None / non-finite)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v if math.isfinite(v) else NAN


def _db(x, floor: float = 1e-20):
    return 10.0 * np.log10(np.maximum(x, floor))


def _band(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freqs >= lo) & (freqs < hi)


def _native_sr(ctx: Context) -> Optional[int]:
    """Best-effort native sample rate from ctx.native (orchestrator keys or raw ffprobe JSON)."""
    n = ctx.native or {}
    for k in ("native_sr", "sample_rate", "sr"):
        try:
            if n.get(k):
                return int(float(n[k]))
        except (TypeError, ValueError):
            pass
    for s in n.get("streams", []) or []:
        try:
            if s.get("codec_type") == "audio" and s.get("sample_rate"):
                return int(float(s["sample_rate"]))
        except (TypeError, ValueError, AttributeError):
            pass
    return None


# ==============================================================================================
# Shared intermediates
# ==============================================================================================
def _base(ctx: Context) -> dict:
    """STFT, frame levels, speech mask, long-term spectra and log-mel, computed once per file."""
    b = ctx.cache.get("spectral/base")
    if b is not None:
        return b
    import librosa

    sr = int(ctx.sr)
    y = np.nan_to_num(np.asarray(ctx.audio, dtype=np.float32).ravel())
    hop = max(1, int(round(0.010 * sr)))                  # 10 ms
    win = max(16, int(round(0.025 * sr)))                 # 25 ms
    n_fft = 1 << int(math.ceil(math.log2(win)))           # 512 at 16 kHz
    yy = y if len(y) >= n_fft else np.pad(y, (0, n_fft - len(y)))
    mag = np.abs(librosa.stft(yy, n_fft=n_fft, hop_length=hop, win_length=win, window="hann", center=True))
    mag = mag.astype(np.float32)
    P = mag.astype(np.float64) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    w = sps.get_window("hann", win, fftbins=True)

    # Per-frame mean-square level in dBFS (Parseval on the one-sided spectrum).
    frame_ms = (2.0 / n_fft) * P.sum(axis=0) / np.sum(w ** 2)
    frame_db = _db(frame_ms, 1e-12)
    valid = frame_db > -100.0                             # excludes digital-zero frames
    speech = np.zeros_like(valid)
    if valid.sum() >= 3:
        ref = np.percentile(frame_db[valid], 95)
        floor = np.percentile(frame_db[valid], 10)
        # Speech = within 40 dB of the loud frames and clearly above the noise floor, but never
        # discard frames within 15 dB of the loud ones (clips with no pauses have no "floor").
        thr = max(ref - 40.0, min(floor + 6.0, ref - 15.0))
        speech = valid & (frame_db > thr)
    has_speech = bool(y.size and speech.sum() >= 10 and np.abs(y).max() > 1e-4)

    # Log-mel spectrogram (report + other experts).
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=80, fmin=0.0, fmax=sr / 2.0)
    mel_db = _db(mel_fb @ P, 1e-10).astype(np.float32)

    # High-resolution long-term average spectrum (Welch, ~4 Hz bins at 16 kHz) over the whole clip.
    nper = 1 << int(round(math.log2(0.256 * sr)))
    nper = max(256, min(nper, 1 << int(math.floor(math.log2(max(len(y), 256))))))
    yw = y if len(y) >= nper else np.pad(y, (0, nper - len(y)))
    f_w, P_w = sps.welch(yw, fs=sr, window="hann", nperseg=nper, noverlap=3 * nper // 4)

    b = dict(sr=sr, y=y, hop=hop, win=win, n_fft=n_fft, P=P, freqs=freqs, frame_db=frame_db,
             valid=valid, speech=speech, has_speech=has_speech, mel_db=mel_db,
             ltas_w_hz=f_w, ltas_w_db=_db(P_w, 1e-20), times=np.arange(P.shape[1]) * hop / sr)
    ctx.cache["spectral/base"] = b
    # Publish for other experts, never overwriting what someone else stored.
    ctx.cache.setdefault("stft_mag", mag)
    ctx.cache.setdefault("stft_meta", dict(sr=sr, n_fft=n_fft, hop=hop, win=win, window="hann", center=True))
    ctx.cache.setdefault("mel_db", mel_db)
    return b


def _f0_frames(ctx: Context, b: dict) -> np.ndarray:
    """Praat autocorrelation pitch (Hz, 0 = unvoiced) resampled onto the STFT frame grid."""
    f0 = ctx.cache.get("spectral/f0")
    if f0 is not None:
        return f0
    n_frames = b["P"].shape[1]
    f0 = np.zeros(n_frames)
    try:
        import parselmouth

        step = b["hop"] / b["sr"]
        snd = parselmouth.Sound(b["y"].astype(np.float64), sampling_frequency=b["sr"])
        pitch = snd.to_pitch_ac(time_step=step, pitch_floor=60.0, pitch_ceiling=500.0)
        pt = np.asarray(pitch.xs())
        pf = np.nan_to_num(np.asarray(pitch.selected_array["frequency"], dtype=float))
        if len(pt):
            idx = np.clip(np.round((b["times"] - pt[0]) / step).astype(int), 0, len(pt) - 1)
            near = np.abs(pt[idx] - b["times"]) <= step
            f0 = np.where(near, pf[idx], 0.0)
    except Exception:
        pass  # too short for the pitch window, or parselmouth missing: everything stays unvoiced
    f0[~b["speech"]] = 0.0
    ctx.cache["spectral/f0"] = f0
    return f0


def _voiced_frames(ctx: Context, b: dict) -> dict:
    """Long-window (40 ms, zero-padded) spectra of voiced frames for harmonic / phase analysis.

    Returns frame indices, f0, complex spectra X and the spectra of the time-weighted signal
    Y = FFT((n - centre) * x[n]), from which group delay follows as Re(conj(X) Y) / |X|^2.
    """
    v = ctx.cache.get("spectral/voiced")
    if v is not None:
        return v
    sr, y, hop = b["sr"], b["y"], b["hop"]
    f0 = _f0_frames(ctx, b)
    idx = np.flatnonzero(f0 > 0)
    if len(idx) > MAX_VOICED_FRAMES:
        # Keep every k-th block of 20 voiced frames, so frame-to-frame measures still see neighbours.
        k = int(math.ceil(len(idx) / MAX_VOICED_FRAMES))
        idx = idx[(np.arange(len(idx)) // 20) % k == 0]
    M = int(round(0.020 * sr))
    n_fft = 1 << int(math.ceil(math.log2(3 * (2 * M + 1))))   # 2048 at 16 kHz (~7.8 Hz bins)
    v = dict(idx=idx, f0=f0[idx], M=M, n_fft=n_fft, freqs=np.fft.rfftfreq(n_fft, 1.0 / sr), X=None, Yc=None)
    if len(idx):
        ypad = np.pad(y.astype(np.float64), (M, M))
        seg = ypad[(idx * hop)[:, None] + np.arange(2 * M + 1)[None, :]]
        seg = seg / (np.sqrt(np.mean(seg ** 2, axis=1, keepdims=True)) + 1e-9)   # level-invariant
        w = np.hanning(2 * M + 1)
        n = np.arange(-M, M + 1)
        v["X"] = np.fft.rfft(seg * w, n_fft, axis=1)
        v["Yc"] = np.fft.rfft(seg * w * n, n_fft, axis=1)
    ctx.cache["spectral/voiced"] = v
    return v


# ==============================================================================================
# Expert 1: spectral_stats
# ==============================================================================================
def _spectrogram_artifact(b: dict) -> dict:
    """uint8-quantised log-mel spectrogram (80 dB range) for the HTML report."""
    mel = b["mel_db"]
    stride = max(1, int(math.ceil(mel.shape[1] / MAX_SPEC_FRAMES)))
    mel = mel[:, ::stride]
    top = float(np.max(mel)) if mel.size else 0.0
    q = np.round((np.clip(mel, top - 80.0, top) - (top - 80.0)) / 80.0 * 255.0).astype(np.uint8)
    meta = dict(sr=b["sr"], hop=b["hop"] * stride, n_fft=b["n_fft"], n_mels=int(mel.shape[0]), fmin=0.0,
                fmax=b["sr"] / 2.0, db_top=top, db_range=80.0,
                layout="[n_mels, frames]; row 0 = lowest mel band; 0..255 maps linearly to db_top-80..db_top dB")
    return {"spectrogram_db": q, "spectrogram_meta": meta}


def _cutoff(f: np.ndarray, L: np.ndarray, nyq: float):
    """Effective bandwidth: the frequency of the sharpest roll-off above which nothing substantial
    remains. Returns (cutoff_hz, drop_db over +-200 Hz); cutoff = Nyquist when no roll-off found."""
    df = f[1] - f[0]
    Ls = uniform_filter1d(L, max(1, int(round(60.0 / df))))
    k = max(1, int(round(200.0 / df)))
    rev_max = np.maximum.accumulate(Ls[::-1])[::-1]           # max of Ls at or above each bin
    best_i, best_drop, max_drop = None, 0.0, 0.0
    for i in range(k, len(f) - k):
        if f[i] < 1500.0 or f[i] > nyq - 200.0:
            continue
        drop = Ls[i - k] - Ls[i + k]
        max_drop = max(max_drop, drop)
        # A roll-off: > 12 dB fall within 400 Hz AND nothing above it comes back up.
        if drop > 12.0 and rev_max[i + k] < Ls[i - k] - 12.0 and drop > best_drop:
            best_i, best_drop = i, drop
    if best_i is None:
        return float(nyq), float(max_drop)
    # Refine: the -6 dB point of the roll-off relative to the level just below it.
    ref = Ls[best_i - k]
    j = best_i - k
    while j < len(Ls) - 1 and Ls[j] > ref - 6.0:
        j += 1
    return float(f[j]), float(best_drop)


@expert("spectral_stats", "spectral", cost="cheap")
def spectral_stats(ctx: Context) -> Evidence:
    """Long-term spectral shape of the speech: high-band (4-8 kHz) vs speech-band energy, effective
    bandwidth, per-band flatness, centroid/roll-off, spectral flux and the frame-to-frame roughness
    of sub-band energies. Neural vocoders and acoustic models tend to over-smooth the spectrum
    (dull top band, overly smooth energy trajectories). Also renders the report spectrogram."""
    ev = Evidence("spectral_stats", "spectral", features=_nan_features(STATS_KEYS))
    b = _base(ctx)
    ev.artifacts.update(_spectrogram_artifact(b))
    ev.artifacts["ltas_hz"] = b["ltas_w_hz"].astype(np.float32)
    ev.artifacts["ltas_db"] = b["ltas_w_db"].astype(np.float32)
    if not b["has_speech"]:
        ev.applicable = False
        ev.findings.append(Finding("No speech-level signal was found, so spectral statistics were not computed.", 0, 0.0))
        return ev

    feats = ev.features
    sr, P, freqs, sp = b["sr"], b["P"], b["freqs"], b["speech"]
    nyq = sr / 2.0
    Ps = P[:, sp]
    ltas = Ps.mean(axis=1)
    feats["spectral_speech_frac"] = _f(sp.mean())

    def band_e(lo, hi):
        m = _band(freqs, lo, min(hi, nyq + 1))
        return float(ltas[m].sum()) if m.any() else 0.0

    e_lo = band_e(0, 4000)
    feats["spectral_hf_ratio_db"] = _f(_db(band_e(4000, 8000.1) / (e_lo + 1e-20)))
    feats["spectral_b4_6k_db"] = _f(_db(band_e(4000, 6000) / (e_lo + 1e-20)))
    feats["spectral_b6_8k_db"] = _f(_db(band_e(6000, 8000.1) / (e_lo + 1e-20)))
    feats["spectral_lf_ratio_db"] = _f(_db(band_e(20, 80) / (band_e(80, 4000) + 1e-20)))

    cut, drop = _cutoff(b["ltas_w_hz"], b["ltas_w_db"], nyq)
    ctx.cache["spectral/cutoff_hz"] = cut
    feats["spectral_cutoff_hz"], feats["spectral_cutoff_drop_db"] = _f(cut), _f(drop)

    # Per-band spectral flatness (geometric / arithmetic mean, dB; 0 dB = white noise).
    eps = 1e-10 * float(Ps.max() + 1e-30)
    for name, lo, hi in (("lo", 100, 1000), ("mid", 1000, 4000), ("hi", 4000, min(7000, 0.875 * nyq))):
        m = _band(freqs, lo, hi)
        if m.sum() >= 4:
            pb = Ps[m] + eps
            flat = np.exp(np.mean(np.log(pb), axis=0)) / np.mean(pb, axis=0)
            feats[f"spectral_flatness_{name}_db"] = _f(np.median(_db(flat)))

    # Centroid and 85 % roll-off per speech frame.
    tot = Ps.sum(axis=0) + 1e-30
    cent = (freqs[:, None] * Ps).sum(axis=0) / tot
    roll = freqs[np.argmax(np.cumsum(Ps, axis=0) >= 0.85 * tot[None, :], axis=0)]
    feats["spectral_centroid_mean_hz"], feats["spectral_centroid_std_hz"] = _f(cent.mean()), _f(cent.std())
    feats["spectral_rolloff_mean_hz"], feats["spectral_rolloff_std_hz"] = _f(roll.mean()), _f(roll.std())

    # Spectral flux and sub-band trajectories over consecutive speech-frame pairs.
    pair = sp[1:] & sp[:-1]
    Ldb = _db(P, 1e-20)
    Ldb = np.maximum(Ldb, Ldb.max(axis=0, keepdims=True) - 80.0)          # 80 dB floor per frame
    if pair.sum() >= 5:
        flux = np.sqrt(np.mean(np.diff(Ldb, axis=1)[:, pair] ** 2, axis=0))
        feats["spectral_flux_mean_db"] = _f(flux.mean())
        feats["spectral_flux_cv"] = _f(flux.std() / (flux.mean() + 1e-9))
        sub = {}
        for name, lo, hi in (("lo", 80, 1000), ("mid", 1000, 4000), ("hi", 4000, nyq + 1)):
            m = _band(freqs, lo, hi)
            if m.any():
                e = _db(P[m].sum(axis=0), 1e-20)
                sub[name] = e
                feats[f"spectral_roughness_{name}_db"] = _f(np.std(np.diff(e)[pair]))
        if "hi" in sub:
            feats["spectral_tvar_hi_db"] = _f(np.std(sub["hi"][sp]))

    # High-band balance restricted to voiced frames (vowels: harmonics should fade above 4 kHz).
    f0 = _f0_frames(ctx, b)
    vo = f0 > 0
    if vo.sum() >= 5:
        lt_v = P[:, vo].mean(axis=1)
        feats["spectral_hf_voiced_ratio_db"] = _f(_db(lt_v[_band(freqs, 4000, 8000.1)].sum()
                                                      / (lt_v[_band(freqs, 0, 4000)].sum() + 1e-20)))

    ev.findings = _stats_findings(ctx, feats)
    return ev


def _votes(f: Dict[str, float], rule: Dict[str, float], below: bool = True) -> List[str]:
    """Names of the features in `rule` that fall below (or above) their threshold."""
    return [k for k, thr in rule.items() if math.isfinite(f.get(k, NAN)) and ((f[k] < thr) if below else (f[k] > thr))]


def _stats_findings(ctx: Context, f: Dict[str, float]) -> List[Finding]:
    out: List[Finding] = []
    nsr = _native_sr(ctx)
    nyq = ctx.sr / 2.0
    hf, cut = f["spectral_hf_ratio_db"], f["spectral_cutoff_hz"]
    lo, hi = REF["hf_ratio_db"]
    low_rate_file = nsr is not None and nsr < 0.75 * ctx.sr    # e.g. an 8 kHz file analysed at 16 kHz
    band_limited = math.isfinite(cut) and cut < min(6000.0, 0.8 * nyq)

    if band_limited:
        why = (f"expected, because the file itself is sampled at only {nsr / 1000:g} kHz" if low_rate_file else
               "so the audio passed through a lower-bandwidth stage: a TTS model or vocoder running at a low sample "
               "rate, a telephone channel, or a low-bitrate codec. This is a provenance clue, not proof of synthesis")
        out.append(Finding(
            f"The spectrum is band-limited at about {cut / 1000:.1f} kHz (a {f['spectral_cutoff_drop_db']:.0f} dB "
            f"roll-off), well below the {nyq / 1000:.0f} kHz analysis bandwidth; this is {why}.",
            0, 0.1 if low_rate_file else 0.3))
    if math.isfinite(hf) and not band_limited:
        if hf < lo - 5:
            s = min(0.6, 0.25 + (lo - 5 - hf) / 30)
            out.append(Finding(
                f"Energy above 4 kHz is {-hf:.0f} dB below the 0-4 kHz speech band (natural wideband speech: "
                f"{-hi:.0f} to {-lo:.0f} dB). A top band this dull is typical of vocoder over-smoothing, though "
                "heavy low-pass processing or a muffled microphone can do the same.", +1, round(s, 2)))
        elif hf > hi:
            out.append(Finding(
                f"Energy above 4 kHz is unusually strong, only {-hf:.0f} dB below the speech band (typical: {-hi:.0f} "
                f"to {-lo:.0f} dB): broadband hiss or noise in the top band. Some vocoders produce noisy high "
                "frequencies, but so do noisy recordings, so this is reported as neutral.", 0, 0.15))

    votes = _votes(f, SMOOTH_RULE)
    if len(votes) >= SMOOTH_MIN_VOTES:
        s = min(0.5, 0.2 + 0.1 * (len(votes) - SMOOTH_MIN_VOTES))
        out.append(Finding(
            f"The spectrum evolves unusually smoothly over time: spectral flux {f['spectral_flux_mean_db']:.1f} dB "
            f"per frame, and 10 ms energy changes of {f['spectral_roughness_lo_db']:.1f} / "
            f"{f['spectral_roughness_mid_db']:.1f} / {f['spectral_roughness_hi_db']:.1f} dB in the low / mid / high "
            f"bands ({len(votes)} of {len(SMOOTH_RULE)} smoothness measures are below the natural-speech range). "
            "Over-smoothed spectral trajectories are a classic side-effect of statistical acoustic models and "
            "vocoders; in our calibration they appeared in 12 % of AASIST-flagged clips and at most 2 % of real ones.",
            +1, round(s, 2)))

    lf = f["spectral_lf_ratio_db"]
    if math.isfinite(lf) and lf > REF["lf_ratio_db"][1]:
        out.append(Finding(
            f"Strong energy below 80 Hz ({lf:.0f} dB relative to the speech band): room rumble, handling noise, "
            "wind or mains hum. This is common in real recordings but we also measured it in synthetic corpora, "
            "so it is reported as neutral.", 0, 0.1))

    if not out:
        out.append(Finding(
            "Spectral balance, bandwidth and frame-to-frame spectral dynamics are within the range of natural "
            "speech; no spectral anomaly was detected (this alone does not establish authenticity).", 0, 0.0))
    return out[:4]


# ==============================================================================================
# Expert 2: lfcc_stats
# ==============================================================================================
def _linear_filterbank(sr: int, n_fft: int, n_filters: int = N_LFCC) -> np.ndarray:
    """Triangular filters equally spaced on a LINEAR frequency axis from 0 to Nyquist."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = np.linspace(0.0, sr / 2.0, n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)))
    for i in range(n_filters):
        l, c, r = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.clip(np.minimum((freqs - l) / (c - l), (r - freqs) / (r - c)), 0.0, None)
    return fb


def _delta(x: np.ndarray, width: int = 2) -> np.ndarray:
    """Regression deltas along axis 0 (frames), edge-padded (HTK formula)."""
    if len(x) == 0:
        return x
    xp = np.pad(x, ((width, width), (0, 0)), mode="edge")
    num = sum(n * (xp[width + n: width + n + len(x)] - xp[width - n: width - n + len(x)]) for n in range(1, width + 1))
    return num / (2.0 * sum(n * n for n in range(1, width + 1)))


@expert("lfcc_stats", "spectral", cost="cheap")
def lfcc_stats(ctx: Context) -> Evidence:
    """Linear-frequency cepstral coefficients (20 LFCC + deltas, the ASVspoof baseline front end)
    summarised as per-coefficient mean and standard deviation. Unlike MFCCs, LFCCs keep the same
    resolution at high frequencies, where vocoder artefacts live. Intended for a downstream
    statistical model (GMM / logistic regression); frame-level LFCCs are kept in artifacts."""
    ev = Evidence("lfcc_stats", "spectral", features=_nan_features(LFCC_KEYS))
    b = _base(ctx)
    frames = b["valid"]
    if frames.sum() < 5 or not b["has_speech"]:
        ev.applicable = False
        ev.findings.append(Finding("Too little non-silent audio to compute cepstral statistics.", 0, 0.0))
        return ev
    fb = _linear_filterbank(b["sr"], b["n_fft"])
    E = fb @ b["P"][:, frames]
    logE = np.log(E + 1e-10 * float(E.max() + 1e-30))
    C = dct(logE, type=2, axis=0, norm="ortho")[:N_LFCC].T          # [frames, 20]
    D = _delta(C)
    DD = _delta(D)
    feats = ev.features
    for i in range(N_LFCC):
        feats[f"lfcc_mean_{i}"] = _f(C[:, i].mean())
        feats[f"lfcc_std_{i}"] = _f(C[:, i].std())
        feats[f"lfcc_dstd_{i}"] = _f(D[:, i].std())
        feats[f"lfcc_ddstd_{i}"] = _f(DD[:, i].std())
    full = np.concatenate([C, D, DD], axis=1).astype(np.float32)
    ev.artifacts["lfcc_frames"] = full
    ctx.cache.setdefault("lfcc", full)
    ev.findings.append(Finding(
        "Linear-frequency cepstral statistics were extracted as a numeric fingerprint for the learned "
        "fusion model; they carry no stand-alone verdict.", 0, 0.0))
    return ev


# ==============================================================================================
# Expert 3: vocoder_artifacts
# ==============================================================================================
def _hnr(b: dict) -> Tuple[Dict[str, float], np.ndarray]:
    """Praat cross-correlation harmonicity (HNR, dB) over voiced frames and its frame-to-frame jumps."""
    out = {}
    import parselmouth

    snd = parselmouth.Sound(b["y"].astype(np.float64), sampling_frequency=b["sr"])
    h = snd.to_harmonicity_cc(time_step=b["hop"] / b["sr"], minimum_pitch=75.0)
    vals = np.asarray(h.values).ravel()
    ok = vals > -199.0                                        # Praat marks unvoiced frames as -200
    if ok.sum() >= 5:
        v = vals[ok]
        out["vocoder_hnr_mean_db"], out["vocoder_hnr_std_db"] = _f(v.mean()), _f(v.std())
        both = ok[1:] & ok[:-1]
        if both.sum() >= 3:
            out["vocoder_hnr_jump_db"] = _f(np.median(np.abs(np.diff(vals))[both]))
    return out, vals


def _cpp(v: dict, sr: int) -> Dict[str, float]:
    """Cepstral peak prominence (Hillenbrand-style) per voiced frame: harmonic clarity in dB."""
    P = np.abs(v["X"]) ** 2
    L = _db(P, 1e-12)
    ceps = np.fft.irfft(L, n=v["n_fft"], axis=1)[:, : v["n_fft"] // 2]
    cdb = _db(ceps ** 2, 1e-20)
    q = np.arange(cdb.shape[1]) / sr
    fit = (q >= 0.001) & (q <= 0.020)
    rng = (q >= 1 / 400.0) & (q <= 1 / 60.0)
    A = np.vstack([q[fit], np.ones(fit.sum())]).T
    coef = np.linalg.lstsq(A, cdb[:, fit].T, rcond=None)[0]      # [2, frames]
    ri = np.flatnonzero(rng)
    pk = ri[np.argmax(cdb[:, ri], axis=1)]
    cpp = cdb[np.arange(len(pk)), pk] - (coef[0] * q[pk] + coef[1])
    return {"vocoder_cpp_mean_db": _f(cpp.mean()), "vocoder_cpp_std_db": _f(cpp.std())}


def _band_harmonicity(v: dict) -> Dict[str, float]:
    """Harmonicity per band = normalised autocorrelation of the (detrended) log spectrum at a lag of
    one f0. 1 = perfectly regular harmonic comb, ~0 = noise. Robust to small f0 errors."""
    freqs = v["freqs"]
    df = freqs[1] - freqs[0]
    L = _db(np.abs(v["X"]) ** 2, 1e-12)
    bands = {"lo": (300.0, 2000.0), "mid": (2000.0, 4000.0), "hi": (4000.0, 7000.0)}
    res = {k: [] for k in bands}
    for i, f0 in enumerate(v["f0"]):
        lag0 = f0 / df
        if lag0 < 4:
            continue
        lags = np.arange(int(math.floor(lag0 * 0.92)), int(math.ceil(lag0 * 1.08)) + 1)
        for name, (lo, hi) in bands.items():
            m = _band(freqs, lo, min(hi, freqs[-1]))
            x = L[i, m]
            if len(x) < 3 * lags[-1]:
                continue
            x = x - uniform_filter1d(x, max(3, int(round(lag0))), mode="nearest")
            den = np.dot(x, x) + 1e-12
            r = max(np.dot(x[:-l], x[l:]) / den for l in lags)
            res[name].append(r)
    out = {}
    for name, vals in res.items():
        if len(vals) >= 5:
            out[f"vocoder_harm_{name}"] = _f(np.median(vals))
            if name == "hi":
                out["vocoder_harm_hi_std"] = _f(np.std(vals))
    return out


def _phase(v: dict, b: dict) -> Dict[str, float]:
    """(a) Relative phase shift (RPS) of harmonics: RPS_k = phi_k - k*phi_1 removes the linear-phase
    term, leaving the glottal-pulse phase shape. We measure how stable it is between consecutive
    voiced frames (circular mean resultant length). (b) Group delay of the voiced spectrum: the rate of
    'impossible' group-delay spikes and the spread / temporal consistency of the modified group delay."""
    out = {}
    sr, hop, y = b["sr"], b["hop"], b["y"].astype(np.float64)
    idx, f0s = v["idx"], v["f0"]
    K = 10
    rps = np.full((len(idx), K + 1), np.nan)
    for i, (t, f0) in enumerate(zip(idx, f0s)):
        half = int(round(1.5 * sr / f0))                     # 3-period Hann window, centred on the frame
        c = t * hop
        if c - half < 0 or c + half + 1 > len(y):
            continue
        seg = y[c - half: c + half + 1] * np.hanning(2 * half + 1)
        kmax = int(min(K, (0.45 * sr) // f0, 4000.0 // f0))
        if kmax < 2:
            continue
        n = np.arange(-half, half + 1)
        k = np.arange(1, kmax + 1)
        Xk = np.exp(-2j * np.pi * np.outer(k, n) * f0 / sr) @ seg
        ph = np.angle(Xk)
        rps[i, 2: kmax + 1] = np.angle(np.exp(1j * (ph[1:] - k[1:] * ph[0])))
    consec = (np.diff(idx) == 1) & (np.abs(np.diff(np.log(f0s + 1e-9))) < 0.1)
    if consec.sum() >= 5:
        d = np.angle(np.exp(1j * (rps[1:] - rps[:-1])))[consec]
        coh = np.array([np.abs(np.nanmean(np.exp(1j * d[:, k]))) if np.isfinite(d[:, k]).sum() >= 5 else np.nan
                        for k in range(K + 1)])
        out["vocoder_rps_coh_lo"] = _f(np.nanmean(coh[2:5]))
        if np.isfinite(coh[5:]).any():
            out["vocoder_rps_coh_hi"] = _f(np.nanmean(coh[5:]))

    X, Yc, freqs, M = v["X"], v["Yc"], v["freqs"], v["M"]
    m = _band(freqs, 100.0, 4000.0)
    Xm, Ym = X[:, m], Yc[:, m]
    num = np.real(np.conj(Xm) * Ym)
    tau = num / (np.abs(Xm) ** 2 + 1e-12)                       # samples, relative to window centre
    out["vocoder_gd_spike_rate"] = _f(np.mean(np.abs(tau) > M))
    # Modified group delay (Hegde/Murthy): denominator = cepstrally smoothed spectrum.
    logmag = np.log(np.abs(X) + 1e-9)
    cep = np.fft.irfft(logmag, n=v["n_fft"], axis=1)
    cep[:, 30:-30] = 0.0
    S = np.exp(np.real(np.fft.rfft(cep, n=v["n_fft"], axis=1)))[:, m]
    tp = num / (S ** (2 * 0.9) + 1e-12)
    mgd = np.sign(tp) * np.abs(tp) ** 0.4
    out["vocoder_mgd_std"] = _f(np.median(mgd.std(axis=1)))
    if len(idx) >= 6:
        adj = np.flatnonzero(np.diff(idx) == 1)
        if len(adj) >= 5:
            a, c2 = mgd[adj], mgd[adj + 1]
            a = a - a.mean(axis=1, keepdims=True)
            c2 = c2 - c2.mean(axis=1, keepdims=True)
            r = (a * c2).sum(axis=1) / (np.sqrt((a ** 2).sum(axis=1) * (c2 ** 2).sum(axis=1)) + 1e-12)
            out["vocoder_mgd_tcorr"] = _f(np.median(r))
    return out


def _upsampling_grid(lo: float, hi: float) -> np.ndarray:
    """Frequencies where transposed-convolution upsampling layers leave tones: multiples of
    base_rate / k for common vocoder output rates (16, 22.05, 24 kHz) and upsampling products k."""
    g = set()
    for base in (16000.0, 22050.0, 24000.0):
        for k in (2, 4, 8, 16, 32, 64):
            step = base / k
            m = np.arange(1, int(hi // step) + 1) * step
            g.update(np.round(m[(m >= lo) & (m <= hi)], 2).tolist())
    return np.array(sorted(g))


def _tonal(b: dict, cutoff_hz: float) -> Tuple[Dict[str, float], dict]:
    """Narrow spectral lines in the high-resolution long-term spectrum (upsampling 'checkerboard')."""
    f, L = b["ltas_w_hz"], b["ltas_w_db"]
    df = f[1] - f[0]
    k = max(2, int(round(20.0 / df)))                          # compare with the level +-20 Hz away
    hi = min(7500.0, (cutoff_hz if math.isfinite(cutoff_hz) else b["sr"] / 2) - 150.0)
    prom = np.full_like(L, -np.inf)
    prom[k:-k] = L[k:-k] - 0.5 * (L[:-2 * k] + L[2 * k:])
    locmax = np.zeros_like(L, dtype=bool)
    locmax[1:-1] = (L[1:-1] >= L[:-2]) & (L[1:-1] >= L[2:])
    region = (f >= 300.0) & (f <= hi) & locmax
    out = {}
    art = {}
    if region.sum() >= 5:
        pr = prom[region]
        j = int(np.argmax(pr))
        out["vocoder_tonal_max_db"] = _f(pr[j])
        out["vocoder_tonal_hz"] = _f(f[region][j])
        out["vocoder_tonal_count"] = _f(np.sum(pr > 10.0))
        grid = _upsampling_grid(300.0, hi)
        if len(grid):
            gi = np.round(grid / df).astype(int)
            gp = []
            for g in gi:
                w = prom[max(0, g - 1): g + 2]
                w = w[np.isfinite(w)]
                if len(w):
                    gp.append(w.max())
            if gp:
                top = np.sort(gp)[-3:]
                out["vocoder_grid_excess_db"] = _f(np.mean(top) - np.percentile(pr, 90))
        art = {"tonal_hz": f[(f >= 300.0) & (f <= hi)].astype(np.float32),
               "tonal_prominence_db": np.clip(prom[(f >= 300.0) & (f <= hi)], -40, 40).astype(np.float32)}
    return out, art


def _modulation_lines(b: dict) -> Tuple[Dict[str, float], dict]:
    """Lines in the modulation spectrum of the 1-7 kHz envelope. A vocoder that assembles audio frame
    by frame (e.g. 80 frames/s for a 200-sample hop) can imprint a constant-rate envelope modulation
    that shows up as a narrow line, unlike the pitch-driven modulation of natural speech."""
    sr, y = b["sr"], b["y"].astype(np.float64)
    if len(y) < sr:                                          # need >= 1 s for a meaningful line
        return {}, {}
    hi = min(7000.0, 0.45 * sr)
    sos = sps.butter(4, [1000.0, hi], btype="bandpass", fs=sr, output="sos")
    env = sps.sosfiltfilt(sos, y) ** 2
    sos2 = sps.butter(4, 480.0, btype="lowpass", fs=sr, output="sos")
    env = sps.sosfiltfilt(sos2, env)
    dec = max(1, int(sr // 1000))
    env = env[::dec]
    fs_e = sr / dec
    env = (env - env.mean()) * np.hanning(len(env))
    nfft = 1 << int(math.ceil(math.log2(len(env) * 4)))
    Mdb = _db(np.abs(np.fft.rfft(env, nfft)) ** 2, 1e-30)
    fm = np.fft.rfftfreq(nfft, 1.0 / fs_e)
    width = max(3, int(round(10.0 / (fm[1] - fm[0]))) | 1)
    resid = Mdb - median_filter(Mdb, size=width, mode="nearest")
    reg = (fm >= 30.0) & (fm <= 450.0)
    if reg.sum() < 5:
        return {}, {}
    j = int(np.argmax(resid[reg]))
    return ({"vocoder_modline_db": _f(resid[reg][j]), "vocoder_modline_hz": _f(fm[reg][j])},
            {"modspec_hz": fm[reg].astype(np.float32), "modspec_resid_db": resid[reg].astype(np.float32)})


def _cqt_sharpness(b: dict, f0_frames: np.ndarray) -> Tuple[Dict[str, float], dict]:
    """Constant-Q (36 bins/octave, 60-1920 Hz) sharpness of the first ~8 harmonics: peak-to-valley
    contrast at the known f0, plus an f0-free 'peakiness' of the low-frequency spectrum."""
    import librosa

    sr, y = b["sr"], b["y"]
    max_len = int(MAX_CQT_SECONDS * sr)
    off = 0
    if len(y) > max_len:                                     # centre crop long files
        off = (len(y) - max_len) // 2
        y = y[off: off + max_len]
    bpo, fmin, n_oct = 36, 60.0, 5
    hop_c = 1 << int(round(math.log2(0.008 * sr)))            # 128 at 16 kHz
    C = np.abs(librosa.cqt(y.astype(np.float32), sr=sr, hop_length=hop_c, fmin=fmin, n_bins=bpo * n_oct,
                           bins_per_octave=bpo))
    Cdb = _db(C ** 2, 1e-20)
    Cdb = np.maximum(Cdb, Cdb.max() - 100.0)
    tc = (np.arange(C.shape[1]) * hop_c + off) / sr
    fi = np.clip(np.round(tc / (b["hop"] / sr)).astype(int), 0, len(f0_frames) - 1)
    f0c = f0_frames[fi]
    voiced = np.flatnonzero((f0c >= 60.0) & (f0c <= 400.0))
    out, art = {}, {}
    if len(voiced) < 5:
        return out, art

    def binof(f):
        return bpo * np.log2(f / fmin)

    per_frame, slope = [], []
    nb = Cdb.shape[0]
    for t in voiced:
        f0 = f0c[t]
        col = Cdb[:, t]
        cons = []
        for k in range(1, 9):
            if (k + 1) * f0 > fmin * 2 ** n_oct * 0.97:
                break
            pk1 = int(round(binof(k * f0)))
            pk2 = int(round(binof((k + 1) * f0)))
            va_lo = int(math.floor(binof((k + 0.3) * f0)))
            va_hi = int(math.ceil(binof((k + 0.7) * f0)))
            if pk1 < 1 or pk2 + 2 > nb or va_hi <= va_lo:
                continue
            peak = 0.5 * (col[pk1 - 1: pk1 + 2].max() + col[pk2 - 1: pk2 + 2].max())
            cons.append(peak - col[va_lo: va_hi + 1].min())
        if len(cons) >= 3:
            per_frame.append(np.mean(cons))
            if len(cons) >= 6:
                slope.append(np.mean(cons[4:7]) - np.mean(cons[:3]))
    if len(per_frame) >= 5:
        out["vocoder_cqt_contrast_db"] = _f(np.median(per_frame))
        out["vocoder_cqt_contrast_std_db"] = _f(np.std(per_frame))
    if len(slope) >= 5:
        out["vocoder_cqt_contrast_slope_db"] = _f(np.median(slope))
    Cv = Cdb[:, voiced]
    out["vocoder_cqt_peakiness_db"] = _f(np.median(np.mean(np.abs(Cv - uniform_filter1d(Cv, 12, axis=0)), axis=0)))
    art = {"cqt_contrast_track_db": np.asarray(per_frame, np.float32)}
    return out, art


@expert("vocoder_artifacts", "spectral", cost="medium")
def vocoder_artifacts(ctx: Context) -> Evidence:
    """Fine-structure traces of neural / statistical vocoders in voiced speech: (1) harmonic-to-noise
    level and its frame-to-frame regularity (HNR, cepstral peak prominence, per-band harmonicity);
    (2) harmonic phase coherence (relative phase shift) and group-delay irregularity; (3) narrow
    spectral lines at upsampling-layer frequencies and frame-rate lines in the modulation spectrum;
    (4) constant-Q sharpness of the low harmonics. Real voices are harmonically rich but irregular;
    vocoders tend to be too regular, too noisy in the top band, or to leave periodic artefacts."""
    ev = Evidence("vocoder_artifacts", "spectral", features=_nan_features(VOCODER_KEYS))
    b = _base(ctx)
    if not b["has_speech"]:
        ev.applicable = False
        ev.findings.append(Finding("No speech-level signal was found, so vocoder analysis was skipped.", 0, 0.0))
        return ev
    feats, errors = ev.features, []
    f0 = _f0_frames(ctx, b)
    feats["vocoder_voiced_frac"] = _f((f0 > 0).sum() / max(1, b["speech"].sum()))

    def run(name, fn):
        """Run one sub-analysis; a failure only blanks its own features."""
        try:
            res = fn()
            if isinstance(res, tuple):
                res, art = res
                ev.artifacts.update(art)
            feats.update({k: _f(v) for k, v in res.items() if k in feats})
        except Exception as e:  # keep the other sub-analyses
            errors.append(f"{name}: {type(e).__name__}: {e}")

    def hnr():
        res, vals = _hnr(b)
        return res, {"hnr_track_db": np.where(vals > -199, vals, np.nan).astype(np.float32)}

    run("hnr", hnr)
    v = _voiced_frames(ctx, b)
    if v["X"] is not None and len(v["idx"]) >= 5:
        run("cpp", lambda: _cpp(v, b["sr"]))
        run("harmonicity", lambda: _band_harmonicity(v))
        run("phase", lambda: _phase(v, b))
        run("cqt", lambda: _cqt_sharpness(b, f0))
    cutoff = ctx.cache.get("spectral/cutoff_hz")
    if cutoff is None:
        cutoff, _ = _cutoff(b["ltas_w_hz"], b["ltas_w_db"], b["sr"] / 2.0)
    run("tonal", lambda: _tonal(b, cutoff))
    run("modulation", lambda: _modulation_lines(b))
    if errors:
        ev.error = "; ".join(errors)
    ev.findings = _vocoder_findings(feats)
    return ev


def _vocoder_findings(f: Dict[str, float]) -> List[Finding]:
    """Up to 4 findings, strongest first. Synthetic-leaning ones fire only beyond the natural range
    (or when several regularity cues agree); otherwise a neutral or weakly real-leaning summary."""
    cands = []   # (strength, Finding)

    votes = _votes(f, REGULAR_RULE)
    if len(votes) >= REGULAR_MIN_VOTES:
        s = min(0.55, 0.3 + 0.1 * (len(votes) - REGULAR_MIN_VOTES))
        cands.append((s, Finding(
            "Voice quality is unusually uniform across the utterance: the frame-to-frame variation of harmonic "
            f"clarity (cepstral peak prominence spread {f['vocoder_cpp_std_db']:.1f} dB), of low-harmonic sharpness "
            f"({f['vocoder_cqt_contrast_std_db']:.1f} dB) and of the harmonic-to-noise ratio (median change "
            f"{f['vocoder_hnr_jump_db']:.2f} dB per 10 ms) is low ({len(votes)} of {len(REGULAR_RULE)} measures below "
            "the natural range). Real vocal folds and breathing add irregularity that vocoders tend to smooth away; "
            "in our calibration this pattern appeared in 19 % of AASIST-flagged clips and at most 2 % of real ones.",
            +1, round(s, 2))))

    tonal, t_hz, grid = f["vocoder_tonal_max_db"], f["vocoder_tonal_hz"], f["vocoder_grid_excess_db"]
    if math.isfinite(tonal) and tonal > REF["tonal_db"][1]:
        on_grid = math.isfinite(grid) and grid > 3.0
        shown = f"{tonal:.0f} dB" if tonal < 60 else "more than 60 dB"
        cands.append((0.3 if on_grid else 0.15, Finding(
            f"A narrow spectral line at {t_hz:.0f} Hz stands {shown} above the neighbouring spectrum"
            + (", on a frequency where upsampling layers of neural vocoders typically leave tones ('checkerboard' "
               "artefacts). Electrical interference in a real recording can look similar."
               if on_grid else ". Its origin is unclear (electrical interference, a tonal background noise, or a "
               "synthesis artefact), so it is reported as neutral."),
            +1 if on_grid else 0, 0.3 if on_grid else 0.15)))

    ml, ml_hz = f["vocoder_modline_db"], f["vocoder_modline_hz"]
    if math.isfinite(ml) and ml > REF["modline_db"][1]:
        s = min(0.3, 0.15 + (ml - REF["modline_db"][1]) / 20)
        cands.append((s, Finding(
            f"The loudness envelope of the 1-7 kHz band pulses at a constant {ml_hz:.1f} Hz ({ml:.0f} dB above its "
            "surroundings in the modulation spectrum). A fixed-rate pulse suggests audio assembled frame by frame "
            "by a vocoder; natural speech modulates at its (varying) pitch rate.", +1, round(s, 2))))

    hh = f["vocoder_harm_hi"]
    if math.isfinite(hh) and hh > REF["harm_hi"][1] + 0.1:
        s = min(0.35, 0.15 + (hh - REF["harm_hi"][1]))
        cands.append((s, Finding(
            f"Voiced frames show a clean, regular harmonic comb even at 4-7 kHz (harmonicity {hh:.2f}; natural "
            f"speech stays below about {REF['harm_hi'][1]:.2f} because breath noise dominates there). Overly periodic "
            "high-frequency harmonics are a known trait of some neural vocoders.", +1, round(s, 2))))

    cc = f["vocoder_cqt_contrast_db"]
    if math.isfinite(cc) and cc > REF["cqt_contrast_db"][1] + 3:
        s = min(0.3, 0.15 + (cc - REF["cqt_contrast_db"][1]) / 20)
        cands.append((s, Finding(
            f"The low harmonics are unusually sharp and deep-valleyed on a high-resolution constant-Q analysis "
            f"({cc:.0f} dB peak-to-valley; natural speech: {REF['cqt_contrast_db'][0]:.0f}-"
            f"{REF['cqt_contrast_db'][1]:.0f} dB). This points to an unnaturally steady pitch or a noise-free "
            "excitation signal.", +1, round(s, 2))))

    rc = f["vocoder_rps_coh_lo"]
    if math.isfinite(rc) and rc > REF["rps_coh_lo"][1] + 0.05:
        s = min(0.3, 0.1 + (rc - REF["rps_coh_lo"][1]))
        cands.append((s, Finding(
            f"The phase relationship between the first harmonics is almost frozen from frame to frame "
            f"(coherence {rc:.2f}; natural speech: below {REF['rps_coh_lo'][1]:.2f}). A fixed excitation phase shape "
            "is typical of vocoders that impose a deterministic pulse shape.", +1, round(s, 2))))

    cands.sort(key=lambda c: -c[0])
    out = [c[1] for c in cands[:4]]
    if not out:
        if len(_votes(f, NATURAL_RULE, below=False)) == len(NATURAL_RULE):
            out.append(Finding(
                "Voice quality varies naturally from frame to frame (cepstral peak prominence spread "
                f"{f['vocoder_cpp_std_db']:.1f} dB, harmonic-sharpness spread {f['vocoder_cqt_contrast_std_db']:.1f} dB, "
                f"HNR change {f['vocoder_hnr_jump_db']:.2f} dB per 10 ms), as expected from a human voice, and no "
                "vocoder fingerprints stood out.", -1, 0.15))
        else:
            out.append(Finding(
                "No vocoder fingerprints stood out: harmonic regularity, harmonic phase, group delay, spectral lines "
                "and low-harmonic sharpness are all within the range of natural speech (not proof of authenticity).",
                0, 0.0))
    return out
