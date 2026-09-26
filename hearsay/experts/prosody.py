"""Prosody & phonetics experts: how the voice *moves* (pitch), *breathes and pauses*, and *sounds* (voice quality).

Why this category exists
------------------------
Neural TTS and voice-conversion systems model the short-term spectrum very well, but the slower,
behavioural layer of speech still carries habits of the generator:
  * pitch (F0) contours that are too smooth (acoustic models predict a smoothed F0 and vocoders
    interpolate it), flat intonation, or abrupt octave errors where the F0 predictor fails;
  * no inhalation noise between phrases (training data are usually breath-trimmed and most systems
    never learned to produce breaths) and very regular pause / syllable timing;
  * unnaturally regular phonation: low jitter/shimmer, high harmonics-to-noise ratio, little creak
    (vocal fry) at phrase ends, over-smooth formant transitions.
None of these cues is proof on its own - careful studio speakers, noise gates and heavy compression
can mimic several - so findings are phrased as "consistent with" and carry modest strengths. The
learned fusion decides how much weight each numeric feature deserves.

Experts registered here
-----------------------
  pitch           Praat F0 track, intonation statistics, micro-prosody smoothness, jitter/shimmer/HNR
  pauses_breaths  webrtcvad pauses, speech rate (de Jong & Wempe syllable nuclei), breath detector
  voice_quality   formants + bandwidths + formant-velocity smoothness, H1-H2, CPP, spectral tilt, creak

Shared cache entries (ctx.cache) written here and reusable by other experts:
  "praat_sound"     parselmouth.Sound of the analysis signal
  "praat_pitch"     parselmouth.Pitch (autocorrelation method, 75-500 Hz, 10 ms hop)
  "f0"              dict(t=frame centre times [s], f0=Hz with 0 for unvoiced, hop_s, floor, ceiling, method,
                    praat=the parselmouth.Pitch object)
  "vad"             dict(frame_s, speech=bool array per frame, segments=[(t0, t1), ...] in s, method)
  "prosody_frames"  dict of 10 ms frame-level arrays: t, db, centroid_hz, flatness, hf_ratio_db, voiced

Every feature is always present (NaN when it cannot be measured, e.g. no voicing), so the fusion sees
a fixed column set. All experts are defensive: silence, clipping, 0.1-60 s clips and any sample rate
are handled without raising.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np

from ..evidence import Context, Evidence, Finding, expert

CATEGORY = "prosody"
NAN = float("nan")

# Analysis constants --------------------------------------------------------------------------
F0_FLOOR, F0_CEILING = 75.0, 500.0   # Praat pitch search range (Hz); covers adult male..child speech
HOP_S = 0.010                        # every track in this module uses a 10 ms hop
WIN_S = 0.025                        # frame length for level / spectral-shape features
MAX_ANALYSIS_S = 60.0                # longer user files are analysed on their first minute
MIN_SIGNAL_S = 0.25                  # below this nothing prosodic can be measured
MIN_VOICED_FRAMES = 20               # 0.2 s of voicing needed for contour statistics
MIN_RUN_FRAMES = 5                   # voiced runs shorter than 50 ms are ignored for micro-prosody
OCTAVE_JUMP_ST = 8.0                 # |dF0| between adjacent 10 ms frames that counts as an octave-type jump
PAUSE_MIN_S = 0.15                   # internal silence >= 150 ms counts as a pause
BREATH_MIN_S, BREATH_MAX_S = 0.15, 0.60
VAD_AGGRESSIVENESS = 2

# Reference ranges for natural read speech, used only to phrase findings (the fusion learns its own
# weights from the raw features). Each is the 10th-90th percentile measured with this module on the
# AASIST-lowest half of the challenge test set (200 clips, likely bona fide VCTK-style speech); real
# LJSpeech clips passed through the organiser pipeline agree within ~15 % except where noted in
# work/expert_tests/prosody/. Jitter/shimmer sit far above clinical norms (<1 % / <4 %) because these are
# 3-4 s peak-normalised, band-limited connected-speech clips, not sustained vowels.
REF = {
    "delta_std_st": (0.32, 0.62),      # frame-to-frame F0 change within voiced stretches (semitones / 10 ms)
    "residual_st": (0.24, 0.52),       # F0 residual after 70 ms smoothing (semitones RMS)
    "f0_range_st": (4.8, 18.0),        # 5-95 % F0 range (semitones)
    "jitter_local_pct": (1.6, 3.2),
    "shimmer_local_pct": (7.0, 11.5),
    "hnr_db": (11.3, 17.7),
    "articulation_rate": (3.1, 4.7),   # syllable nuclei per second of speech
    "syllable_interval_cv": (0.23, 0.53),
    "creak_fraction": (0.05, 0.24),
    "f2_jerk": (0.9, 1.2),             # RMS(d2 F2) / RMS(d F2); ~1.7 = white noise, ~0 = perfectly smooth
    "nonspeech_level_rel_db": (-44.0, -28.0),  # non-speech (room tone / pauses) level re speech
    "cpp_db": (20.5, 23.8),            # cepstral peak prominence of voiced frames
}


# =============================================================================================
# Shared intermediates (cached)
# =============================================================================================
def _signal(ctx: Context) -> Tuple[np.ndarray, int]:
    """Mono float64 analysis signal (finite, DC removed, capped at MAX_ANALYSIS_S) and its rate."""
    if "prosody_signal" not in ctx.cache:
        sr = int(ctx.sr) if ctx.sr else 16000
        x = np.asarray(ctx.audio, dtype=np.float64)
        if x.ndim > 1:                        # tolerate (n, ch) / (ch, n) input: average the channels
            x = x.mean(axis=int(np.argmin(x.shape)))
        x = x.reshape(-1)
        x = np.nan_to_num(x[: int(MAX_ANALYSIS_S * sr)], nan=0.0, posinf=0.0, neginf=0.0)
        if x.size:
            x = x - x.mean()
        ctx.cache["prosody_signal"] = (x, sr)
    return ctx.cache["prosody_signal"]


def _praat_sound(ctx: Context):
    if "praat_sound" not in ctx.cache:
        import parselmouth
        x, sr = _signal(ctx)
        ctx.cache["praat_sound"] = parselmouth.Sound(x, sampling_frequency=sr)
    return ctx.cache["praat_sound"]


def _f0(ctx: Context) -> Dict[str, np.ndarray]:
    """Praat autocorrelation F0 track on a 10 ms grid. Cached as ctx.cache['f0'] (and 'praat_pitch')."""
    own = ctx.cache.get("prosody_f0")
    if own is not None:
        return own
    snd = _praat_sound(ctx)
    pitch = snd.to_pitch_ac(time_step=HOP_S, pitch_floor=F0_FLOOR, pitch_ceiling=F0_CEILING)
    f0 = np.nan_to_num(np.asarray(pitch.selected_array["frequency"], dtype=float), nan=0.0)
    track = {"t": np.asarray(pitch.xs(), dtype=float), "f0": f0, "hop_s": HOP_S,
             "floor": F0_FLOOR, "ceiling": F0_CEILING, "method": "praat_ac", "praat": pitch}
    ctx.cache["prosody_f0"] = track
    ctx.cache.setdefault("praat_pitch", pitch)
    ctx.cache.setdefault("f0", track)      # never clobber a track another expert already published
    return track


def _voicing_at(track: Dict[str, np.ndarray], times: np.ndarray) -> np.ndarray:
    """F0 (Hz, 0 = unvoiced) of the nearest pitch frame for arbitrary times."""
    t, f0 = track["t"], track["f0"]
    if t.size == 0:
        return np.zeros_like(times)
    idx = np.clip(np.round((times - t[0]) / track["hop_s"]).astype(int), 0, t.size - 1)
    return f0[idx]


def _frames(ctx: Context) -> Dict[str, np.ndarray]:
    """10 ms frame-level level (dB), spectral centroid, flatness, high-band ratio and voicing."""
    if "prosody_frames" in ctx.cache:
        return ctx.cache["prosody_frames"]
    x, sr = _signal(ctx)
    win, hop = max(16, int(round(WIN_S * sr))), max(1, int(round(HOP_S * sr)))
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    fr = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    t = (np.arange(fr.shape[0]) * hop + win / 2) / sr
    db = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    n_fft = 1 << int(np.ceil(np.log2(win)))
    P = np.abs(np.fft.rfft(fr * np.hanning(win), n=n_fft, axis=1)) ** 2 + 1e-18
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (f >= 100) & (f <= min(7000.0, sr / 2))
    centroid = (P[:, band] * f[band]).sum(1) / P[:, band].sum(1)
    flat = np.exp(np.mean(np.log(P[:, band]), axis=1)) / np.mean(P[:, band], axis=1)
    lo = (f >= 300) & (f < 4000)
    hi = (f >= 4000) & (f <= min(8000.0, sr / 2))
    hf_ratio = 10 * np.log10(P[:, hi].sum(1) / P[:, lo].sum(1)) if hi.any() else np.full(t.size, NAN)
    out = {"t": t, "db": db, "centroid_hz": centroid, "flatness": flat, "hf_ratio_db": hf_ratio,
           "voiced": _voicing_at(_f0(ctx), t) > 0}
    ctx.cache["prosody_frames"] = out
    return out


def _levels(fr: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """(noise floor, speech level) in dB: 5th percentile of non-digital-silence frames, 90th pct of voiced."""
    db = fr["db"]
    live = db[db > -100]                       # exclude exact digital zeros
    if live.size == 0:
        return -120.0, -120.0
    noise = float(np.percentile(live, 5))
    ref = db[fr["voiced"]] if fr["voiced"].sum() >= 5 else live
    return noise, float(np.percentile(ref, 90))


def _vad(ctx: Context) -> Dict[str, object]:
    """webrtcvad (aggressiveness 2, 10 ms frames) speech mask + segments; energy VAD fallback."""
    if "prosody_vad" in ctx.cache:
        return ctx.cache["prosody_vad"]
    x, sr = _signal(ctx)
    y, vsr = x, sr
    if vsr not in (8000, 16000, 32000, 48000):
        from scipy.signal import resample_poly
        y, vsr = resample_poly(x, 16000, sr), 16000
    peak = np.max(np.abs(y)) if y.size else 0.0
    y = y / peak * 0.9 if peak > 0 else y        # webrtcvad has absolute thresholds: normalise level
    flen = int(vsr * HOP_S)
    n = y.size // flen
    method = "webrtcvad"
    try:
        import webrtcvad
        v = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        pcm = (np.clip(y[: n * flen], -1, 1) * 32767).astype("<i2")
        speech = np.fromiter((v.is_speech(pcm[i * flen:(i + 1) * flen].tobytes(), vsr) for i in range(n)),
                             dtype=bool, count=n)
    except Exception:                         # missing module or odd input -> energy gate
        method = "energy"
        fr = _frames(ctx)
        noise, level = _levels(fr)
        speech = fr["db"][:n] > max(noise + 10, level - 35)
    speech = _fill_runs(speech, value=True, max_len=2)     # drop <=20 ms speech blips (clicks)
    speech = _fill_runs(speech, value=False, max_len=3)    # bridge <=30 ms gaps inside words
    # webrtcvad happily labels breaths, hum and background chatter as speech. A "speech" segment that
    # contains no voicing at all (< 30 ms of Praat-voiced frames) is re-labelled as non-speech.
    voiced = _voicing_at(_f0(ctx), (np.arange(speech.size) + 0.5) * HOP_S) > 0
    for a, b in _runs(speech):
        if voiced[a:b].sum() < 3:
            speech[a:b] = False
    segs = [(a * HOP_S, b * HOP_S) for a, b in _runs(speech)]
    out = {"frame_s": HOP_S, "speech": speech, "segments": segs, "method": method,
           "aggressiveness": VAD_AGGRESSIVENESS}
    ctx.cache["prosody_vad"] = out
    ctx.cache.setdefault("vad", out)
    return out


# =============================================================================================
# Small numeric helpers
# =============================================================================================
def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[(start, end_exclusive)] of consecutive True values."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _fill_runs(mask: np.ndarray, value: bool, max_len: int) -> np.ndarray:
    """Flip interior runs equal to `value` that are at most `max_len` frames long."""
    m = mask.copy()
    target = m if value else ~m
    for a, b in _runs(target):
        if b - a <= max_len and a > 0 and b < m.size:
            m[a:b] = not value
    return m


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if x.size < k:
        return np.full_like(x, x.mean() if x.size else 0.0)
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def _f(v) -> float:
    """Coerce to a plain finite-or-NaN float."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return NAN
    return v if np.isfinite(v) else NAN


def _prefixed(prefix: str, names: Tuple[str, ...], values: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{n}": _f(values.get(n, NAN)) for n in names}


def _net_score(findings: List[Finding]) -> float:
    """Signed sum of finding strengths (+ = synthetic-leaning), clipped to [-1, 1]; a quick routing cue for
    the orchestrator. The fusion should rely on the numeric features, not on this."""
    return float(np.clip(sum(f.direction * f.strength for f in findings), -1.0, 1.0))


# =============================================================================================
# Expert 1: pitch
# =============================================================================================
PITCH_FEATURES = (
    "voiced_fraction", "voiced_s", "f0_median_hz", "f0_std_st", "f0_range_st", "slope_st_per_s",
    "declination_st", "octave_jumps", "octave_jump_rate", "delta_std_st", "residual_st", "jerk_ratio",
    "flat_fraction", "step_rate", "voiced_runs_per_s", "voiced_run_mean_s",
    "jitter_local_pct", "jitter_rap_pct", "shimmer_local_pct", "shimmer_apq3_pct", "hnr_db",
)


@expert("pitch", CATEGORY, cost="medium")
def pitch(ctx: Context) -> Evidence:
    """Praat F0 contour (75-500 Hz): intonation range and declination, micro-prosody smoothness
    (frame-to-frame F0 variation and residual after smoothing - TTS contours are often too smooth or
    show unnatural steps/octave errors), plus Praat jitter, shimmer and harmonics-to-noise ratio."""
    ev = Evidence(expert="pitch", category=CATEGORY)
    x, sr = _signal(ctx)
    v: Dict[str, float] = {}
    if x.size < MIN_SIGNAL_S * sr or not np.any(x):
        ev.applicable = False
        ev.features = _prefixed("pitch", PITCH_FEATURES, v)
        return ev

    tr = _f0(ctx)
    t, f0 = tr["t"], tr["f0"]
    voiced = f0 > 0
    dur = x.size / sr
    v["voiced_fraction"] = voiced.mean() if voiced.size else 0.0
    v["voiced_s"] = voiced.sum() * HOP_S
    runs = _runs(voiced)
    v["voiced_runs_per_s"] = len(runs) / dur
    v["voiced_run_mean_s"] = np.mean([b - a for a, b in runs]) * HOP_S if runs else NAN

    smooth_hz = np.zeros_like(f0)
    if voiced.sum() >= MIN_VOICED_FRAMES:
        med = float(np.median(f0[voiced]))
        st = np.zeros_like(f0)
        st[voiced] = 12 * np.log2(f0[voiced] / med)          # semitones re the speaker's median
        v["f0_median_hz"] = med
        v["f0_std_st"] = np.std(st[voiced])
        p5, p95 = np.percentile(st[voiced], [5, 95])
        v["f0_range_st"] = p95 - p5
        tv, sv = t[voiced], st[voiced]
        if np.ptp(tv) > 0.3:
            v["slope_st_per_s"] = np.polyfit(tv, sv, 1)[0]
        third = max(1, sv.size // 3)
        v["declination_st"] = np.median(sv[-third:]) - np.median(sv[:third])

        # Micro-prosody inside continuous voiced stretches: never across unvoiced gaps, and octave-size
        # jumps split a stretch (they are counted separately, not mixed into the smoothness measures).
        segs, n_jumps = [], 0
        for a, b in runs:
            if b - a < 2:
                continue
            cut = np.flatnonzero(np.abs(np.diff(st[a:b])) > OCTAVE_JUMP_ST)
            n_jumps += cut.size
            bounds = [a, *(a + cut + 1), b]
            segs += [(s, e) for s, e in zip(bounds[:-1], bounds[1:]) if e - s >= MIN_RUN_FRAMES]
        deltas, resid, d2 = [], [], []
        for a, b in segs:
            s = st[a:b]
            d = np.diff(s)
            deltas.append(d)
            d2.append(np.diff(d))
            if b - a >= 7:                                    # 70 ms moving-average "intended" contour
                sm = _moving_average(s, 7)
                smooth_hz[a:b] = med * 2 ** (sm / 12)
                resid.append(s - sm)
        v["octave_jumps"] = n_jumps
        v["octave_jump_rate"] = n_jumps / max(v["voiced_s"], 1e-3)
        d_all = np.concatenate(deltas) if deltas else np.array([])
        if d_all.size >= 10:
            core = d_all[np.abs(d_all) <= 4.0]                # micro-variation, excluding steps
            v["delta_std_st"] = np.std(core) if core.size >= 10 else NAN
            v["flat_fraction"] = np.mean(np.abs(d_all) < 0.03)
            v["step_rate"] = np.sum(np.abs(d_all) > 1.5) / max(v["voiced_s"], 1e-3)
            dd = np.concatenate(d2)
            if dd.size >= 10 and np.any(d_all):
                v["jerk_ratio"] = np.sqrt(np.mean(dd ** 2)) / np.sqrt(np.mean(d_all ** 2))
        if resid:
            r = np.concatenate(resid)
            v["residual_st"] = np.sqrt(np.mean(r ** 2))

        # Cycle-level perturbation (Praat voice report defaults).
        try:
            from parselmouth.praat import call
            snd = _praat_sound(ctx)
            pp = call([snd, tr["praat"]], "To PointProcess (cc)")
            v["jitter_local_pct"] = 100 * call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            v["jitter_rap_pct"] = 100 * call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            v["shimmer_local_pct"] = 100 * call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            v["shimmer_apq3_pct"] = 100 * call([snd, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            harm = snd.to_harmonicity_cc(time_step=HOP_S, minimum_pitch=F0_FLOOR,
                                         silence_threshold=0.1, periods_per_window=1.0)
            hv = np.asarray(harm.values[0], dtype=float)
            hnr_at = _voicing_at({"t": np.asarray(harm.xs()), "f0": hv, "hop_s": HOP_S}, t[voiced])
            hnr_at = hnr_at[hnr_at > -100]                    # Praat marks silent frames -200 dB
            v["hnr_db"] = np.mean(hnr_at) if hnr_at.size else NAN
        except Exception as e:                                # perturbation measures are optional
            ev.error = f"praat perturbation: {type(e).__name__}: {e}"
    else:
        ev.applicable = False                                 # < 0.2 s of voicing: nothing to interpret

    ev.features = _prefixed("pitch", PITCH_FEATURES, v)
    ev.artifacts = {"f0_t": t.astype(np.float32), "f0_hz": f0.astype(np.float32),
                    "f0_smooth_hz": smooth_hz.astype(np.float32)}
    ev.findings = _pitch_findings(ev.features)
    ev.score = _net_score(ev.findings)
    return ev


def _pitch_findings(F: Dict[str, float]) -> List[Finding]:
    out: List[Finding] = []
    vs = F["pitch_voiced_s"]
    if not np.isfinite(F["pitch_f0_median_hz"]):
        if np.isfinite(vs):
            out.append(Finding(f"Too little voiced speech for pitch analysis ({vs:.2f} s voiced).", 0, 0.0))
        return out
    dstd, rng = F["pitch_delta_std_st"], F["pitch_f0_range_st"]
    lo, hi = REF["delta_std_st"]
    if np.isfinite(dstd) and dstd < 0.7 * lo:
        out.append(Finding(
            f"Pitch contour is unusually smooth: frame-to-frame F0 variation {dstd:.2f} semitones vs about "
            f"{lo:.2f}-{hi:.2f} in natural read speech. Human voices carry small involuntary pitch fluctuations; "
            f"acoustic models that predict a smoothed F0 track remove them - consistent with TTS.",
            +1, float(np.clip(0.2 + (0.7 * lo - dstd) / lo, 0.2, 0.45))))
    jumps, rate = F["pitch_octave_jumps"], F["pitch_octave_jump_rate"]
    if np.isfinite(jumps) and jumps >= 4 and rate >= 2.0:
        out.append(Finding(
            f"{int(jumps)} abrupt octave-size pitch jumps ({rate:.1f} per voiced second). Sudden F0 halving/doubling "
            f"inside a voiced stretch is typical of vocoder or F0-predictor errors, although creaky voice can "
            f"cause occasional jumps in real speech.", +1, 0.1))
    if np.isfinite(rng) and rng < 0.6 * REF["f0_range_st"][0]:
        out.append(Finding(
            f"Flat intonation: F0 range only {rng:.1f} semitones (typical read speech {REF['f0_range_st'][0]:.0f}-"
            f"{REF['f0_range_st'][1]:.0f}). Monotone delivery is common in low-expressivity TTS, but also in "
            f"tired or bored human speakers.", +1, 0.15))
    jit, shim, hnr = F["pitch_jitter_local_pct"], F["pitch_shimmer_local_pct"], F["pitch_hnr_db"]
    jlo, shlo, hlo = REF["jitter_local_pct"][0], REF["shimmer_local_pct"][0], REF["hnr_db"][0]
    if np.isfinite(jit) and np.isfinite(shim) and jit < 0.6 * jlo and shim < 0.6 * shlo:
        out.append(Finding(
            f"Very regular vocal-fold cycles: jitter {jit:.2f}% and shimmer {shim:.1f}% (natural clips here "
            f"{jlo:.1f}%+ and {shlo:.0f}%+). Overly periodic phonation is consistent with a neural vocoder.", +1, 0.2))
    if np.isfinite(hnr) and hnr < 0.8 * hlo:
        out.append(Finding(
            f"Rough, noisy voicing: harmonics-to-noise ratio {hnr:.1f} dB (natural clips here "
            f"{REF['hnr_db'][0]:.0f}-{REF['hnr_db'][1]:.0f} dB). Buzzy or noisy excitation is typical of older "
            f"vocoders, but a noisy or heavily compressed channel lowers HNR too.", +1, 0.15))
    if not out:
        out.append(Finding(
            f"Pitch behaviour within the natural range: median F0 {F['pitch_f0_median_hz']:.0f} Hz, range "
            f"{rng:.1f} semitones, micro-variation {dstd:.2f} semitones/10 ms, jitter {jit:.2f}%, shimmer "
            f"{shim:.1f}%, HNR {hnr:.1f} dB.", 0, 0.0))
    return sorted(out, key=lambda f: -f.strength)[:4]


# =============================================================================================
# Expert 2: pauses & breaths
# =============================================================================================
PAUSE_FEATURES = (
    "speech_s", "speech_fraction", "lead_silence_s", "trail_silence_s", "pause_count", "pause_rate",
    "pause_mean_s", "pause_max_s", "pause_total_s", "pause_level_rel_db", "nonspeech_level_rel_db",
    "noise_floor_rel_db", "syllables", "speech_rate", "articulation_rate", "syllable_interval_cv",
    "intensity_range_db", "breath_count", "breath_total_s", "breath_rel_db", "transcript_syl_rate",
)


def _syllable_nuclei(ctx: Context, track) -> np.ndarray:
    """de Jong & Wempe (2009) style syllable nuclei: voiced intensity peaks with >= 2 dB dips."""
    from scipy.signal import find_peaks
    snd = _praat_sound(ctx)
    it = snd.to_intensity(minimum_pitch=50.0, time_step=HOP_S, subtract_mean=False)
    iv, ti = np.asarray(it.values[0], dtype=float), np.asarray(it.xs(), dtype=float)
    if iv.size < 3:
        return np.array([])
    thr = max(np.percentile(iv, 99) - 25.0, np.median(iv) - 5.0)
    pk, _ = find_peaks(iv, height=thr, prominence=2.0, distance=5)     # >= 50 ms apart
    pk_t = ti[pk]
    f0_near = np.maximum.reduce([_voicing_at(track, pk_t + dt) for dt in (-0.02, 0.0, 0.02)])
    return pk_t[f0_near > 0]


def _count_text_syllables(text: str) -> int:
    """Rough English syllable count (vowel groups, silent final e); good to about +-10 %."""
    n = 0
    for w in re.findall(r"[a-z']+", text.lower()):
        w = w.replace("'", "")
        groups = re.findall(r"[aeiouy]+", w)
        k = len(groups)
        if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and k > 1:
            k -= 1
        n += max(1, k)
    return n


def _detect_breaths(fr, speech: np.ndarray, noise: float, level: float) -> List[Tuple[float, float, float]]:
    """Inhalation-like events: 150-600 ms of unvoiced, broadband (spectrally flat, mid/high centroid),
    low-level noise that rises above the noise floor, is not a loud sibilant, and sits in (or directly
    against) a pause or the clip edge. Returns [(t0, t1, peak level re speech dB)].

    Breaths look like soft white-ish noise 15-45 dB below the vowels; room tone is steadier and (in studio
    recordings) low-frequency dominated; fricatives /s/ /sh/ are louder and embedded between voiced sounds.
    """
    n = min(fr["t"].size, speech.size)
    db, cen, flat, hfr = fr["db"][:n], fr["centroid_hz"][:n], fr["flatness"][:n], fr["hf_ratio_db"][:n]
    voiced = fr["voiced"][:n]
    near_voiced = np.convolve(voiced.astype(float), np.ones(5), mode="same") > 0   # +-20 ms guard
    rel = db - level
    sibilant = (hfr > 6.0) & (rel > -25.0) if np.isfinite(hfr).any() else np.zeros(n, bool)
    cand = ((~near_voiced) & (rel >= -50.0) & (rel <= -12.0) & (db >= noise + 2.0)
            & (cen >= 800) & (cen <= 5000) & (flat >= 0.03) & ~sibilant)
    cand = _fill_runs(cand, value=False, max_len=3)
    quiet_or_pause = (db < noise + 4.0) | ~speech[:n]
    events = []
    for a, b in _runs(cand):
        d = (b - a) * HOP_S
        if not (BREATH_MIN_S <= d <= BREATH_MAX_S):
            continue
        seg = db[a:b]
        if seg.max() < noise + 6.0 or np.median(flat[a:b]) < 0.05:     # must stand out and be noise-like
            continue
        if np.median(cen[a:b]) > 3500 or sibilant[max(0, a - 5):a].any() or sibilant[b:b + 5].any():
            continue                                                    # decaying /s/ tail, not a breath
        in_pause = 1.0 - speech[a:b].mean()
        left = a <= 5 or quiet_or_pause[max(0, a - 8):a].mean() > 0.5
        right = b >= n - 5 or quiet_or_pause[b:b + 8].mean() > 0.5
        if in_pause >= 0.5 or left or right:
            events.append((a * HOP_S, b * HOP_S, float(seg.max() - level)))
    return events


@expert("pauses_breaths", CATEGORY, cost="cheap")
def pauses_breaths(ctx: Context) -> Evidence:
    """Pause structure (webrtcvad), speech and articulation rate from syllable nuclei (de Jong & Wempe),
    rhythm regularity, and a breath detector (150-600 ms low-level, broadband, aperiodic inhalation noise
    next to pauses). Real speakers breathe and pause irregularly; most TTS never synthesises breaths."""
    ev = Evidence(expert="pauses_breaths", category=CATEGORY)
    x, sr = _signal(ctx)
    v: Dict[str, float] = {}
    if x.size < MIN_SIGNAL_S * sr or not np.any(x):
        ev.applicable = False
        ev.features = _prefixed("pauses_breaths", PAUSE_FEATURES, v)
        return ev

    dur = x.size / sr
    vad = _vad(ctx)
    speech = vad["speech"]
    fr = _frames(ctx)
    noise, level = _levels(fr)
    track = _f0(ctx)

    v["speech_s"] = speech.sum() * HOP_S
    v["speech_fraction"] = speech.mean() if speech.size else 0.0
    v["noise_floor_rel_db"] = noise - level
    n = min(speech.size, fr["db"].size)
    if (~speech[:n]).sum() >= 5:          # all non-speech (lead-in, pauses, tail): room tone vs. clean gaps
        v["nonspeech_level_rel_db"] = np.median(fr["db"][:n][~speech[:n]]) - level
    runs = _runs(speech)
    pauses: List[Tuple[float, float]] = []
    if runs:
        first, last = runs[0][0], runs[-1][1]
        v["lead_silence_s"] = first * HOP_S
        v["trail_silence_s"] = (speech.size - last) * HOP_S
        gaps = [(b0, a1) for (_, b0), (a1, _) in zip(runs[:-1], runs[1:])]
        pauses = [(a * HOP_S, b * HOP_S) for a, b in gaps if (b - a) * HOP_S >= PAUSE_MIN_S]
        # Level of internal non-speech frames relative to the speech level (room tone vs. clean gaps).
        inner = np.zeros(speech.size, bool)
        inner[first:last] = ~speech[first:last]
        if inner[:n].sum() >= 5:
            v["pause_level_rel_db"] = np.median(fr["db"][:n][inner[:n]]) - level
    else:
        v["lead_silence_s"], v["trail_silence_s"] = dur, 0.0
    pl = np.array([b - a for a, b in pauses])
    v["pause_count"] = len(pauses)
    v["pause_rate"] = len(pauses) / dur
    v["pause_total_s"] = pl.sum() if pl.size else 0.0
    v["pause_mean_s"] = pl.mean() if pl.size else NAN
    v["pause_max_s"] = pl.max() if pl.size else 0.0

    # Speech rate / rhythm.
    nuclei = _syllable_nuclei(ctx, track)
    v["syllables"] = nuclei.size
    v["speech_rate"] = nuclei.size / dur
    if v["speech_s"] > 0.3:
        v["articulation_rate"] = nuclei.size / v["speech_s"]
    if nuclei.size >= 4:
        iv = np.diff(nuclei)
        iv = iv[iv < 0.6]                                         # intervals across pauses are not rhythm
        if iv.size >= 3:
            v["syllable_interval_cv"] = np.std(iv) / np.mean(iv)
    sp_db = fr["db"][:n][speech[:n]]
    if sp_db.size >= 10:
        p5, p95 = np.percentile(sp_db, [5, 95])
        v["intensity_range_db"] = p95 - p5

    # Optional transcript (set by the semantic expert): syllables per second of speech from text.
    tx = ctx.cache.get("transcript")
    text = tx.get("text") if isinstance(tx, dict) else tx
    if isinstance(text, str) and text.strip() and v["speech_s"] > 0.3:
        v["transcript_syl_rate"] = _count_text_syllables(text) / v["speech_s"]

    # Breaths.
    breaths = _detect_breaths(fr, speech, noise, level)
    v["breath_count"] = len(breaths)
    v["breath_total_s"] = sum(b - a for a, b, _ in breaths)
    v["breath_rel_db"] = max((r for _, _, r in breaths), default=NAN)

    ev.features = _prefixed("pauses_breaths", PAUSE_FEATURES, v)
    ev.artifacts = {"vad_t": (np.arange(speech.size) * HOP_S).astype(np.float32), "vad_speech": speech,
                    "pauses": pauses, "breaths": breaths, "nuclei_t": nuclei.astype(np.float32)}
    if v["speech_s"] < 0.2:                       # noise, tones, music: nothing prosodic to interpret
        ev.applicable = False
        ev.findings = [Finding(f"No voiced speech found by the {vad['method']} voice-activity detector "
                               f"({v['speech_s']:.2f} s); pause/breath analysis not applicable.", 0, 0.0)]
        return ev
    ev.findings = _pause_findings(ev.features, pauses, breaths, vad["method"])
    ev.score = _net_score(ev.findings)
    return ev


def _pause_findings(F, pauses, breaths, method) -> List[Finding]:
    out: List[Finding] = []
    long_gaps = [p for p in pauses if p[1] - p[0] >= 0.3]
    long_total = sum(b - a for a, b in long_gaps)
    if breaths:
        where = ", ".join(f"{a:.2f}-{b:.2f} s" for a, b, _ in breaths[:3])
        rel = max(r for _, _, r in breaths)
        out.append(Finding(
            f"Breath-like inhalation noise detected ({len(breaths)} event(s) at {where}, peak {-rel:.0f} dB below "
            f"the vowels). Real speakers inhale between phrases; most TTS systems do not synthesise breaths "
            f"(some modern commercial voices do).", -1, 0.25))
    elif long_total >= 0.5:
        out.append(Finding(
            f"No breath sounds in {len(long_gaps)} pause(s) totalling {long_total:.2f} s. A speaker pausing this long "
            f"often inhales audibly; breathless gaps are mildly consistent with synthesised or edited speech.",
            +1, 0.1))
    ns = F["pauses_breaths_nonspeech_level_rel_db"]
    lo, hi = REF["nonspeech_level_rel_db"]
    if np.isfinite(ns) and ns < lo - 6:
        out.append(Finding(
            f"The gaps around and between words are unusually clean: {-ns:.0f} dB below the speech (natural "
            f"recordings here typically {-hi:.0f}-{-lo:.0f} dB below) - no room tone or breath noise. Consistent "
            f"with audio generated in the digital domain, or with aggressive noise gating/denoising of a real "
            f"recording.", +1, float(np.clip(0.15 + (lo - 6 - ns) / 100, 0.15, 0.3))))
    elif np.isfinite(ns) and lo + 2 <= ns <= hi + 6 and F["pauses_breaths_speech_fraction"] < 0.95:
        out.append(Finding(
            f"Steady room tone in the gaps ({-ns:.0f} dB below the speech), as expected from a microphone "
            f"recording; weak evidence, since TTS trained on noisy recordings can reproduce background noise.",
            -1, 0.1))
    ar = F["pauses_breaths_articulation_rate"]
    lo, hi = REF["articulation_rate"]
    if np.isfinite(ar) and (ar < 0.75 * lo or ar > 1.3 * hi):
        kind = "slow" if ar < lo else "fast"
        out.append(Finding(
            f"Unusually {kind} articulation: {ar:.1f} syllables per second of speech (natural read speech about "
            f"{lo:.1f}-{hi:.1f}).", +1, 0.1))
    cv = F["pauses_breaths_syllable_interval_cv"]
    if np.isfinite(cv) and cv < 0.15:
        out.append(Finding(
            f"Very regular syllable timing (interval variability {cv:.2f}; natural read speech about "
            f"{REF['syllable_interval_cv'][0]:.2f}-{REF['syllable_interval_cv'][1]:.2f}). Metronomic rhythm is a "
            f"known TTS trait.", +1, 0.15))
    if not out:
        out.append(Finding(
            f"{int(F['pauses_breaths_pause_count'])} pause(s) >= {PAUSE_MIN_S * 1000:.0f} ms, speech rate "
            f"{F['pauses_breaths_speech_rate']:.1f} syllables/s ({method} VAD); no breathing or timing anomalies.",
            0, 0.0))
    return sorted(out, key=lambda f: -f.strength)[:4]


# =============================================================================================
# Expert 3: voice quality
# =============================================================================================
VQ_FEATURES = (
    "f1_mean_hz", "f2_mean_hz", "f3_mean_hz", "f1_std_hz", "f2_std_hz", "b1_median_hz", "b2_median_hz",
    "b3_median_hz", "f1_velocity_hz", "f2_velocity_hz", "f2_jerk", "formant_dropout", "h1h2_db",
    "h1h2_std_db", "cpp_db", "ltas_slope_db_khz", "creak_fraction", "low_f0_fraction", "aperiodic_sonorant_fraction",
)


def _harmonic_measures(x, sr, times, f0s, max_frames=200):
    """Per voiced frame: H1-H2 (dB, uncorrected) and cepstral peak prominence (dB)."""
    if times.size == 0:
        return np.array([]), np.array([])
    if times.size > max_frames:                                  # subsample long files evenly
        keep = np.linspace(0, times.size - 1, max_frames).astype(int)
        times, f0s = times[keep], f0s[keep]
    win = int(0.05 * sr)
    n_fft = 1 << int(np.ceil(np.log2(4 * win)))
    xp = np.pad(x, (win, win))
    starts = np.clip((times * sr).astype(int) + win - win // 2, 0, xp.size - win)
    fr = np.stack([xp[s:s + win] for s in starts]) * np.hanning(win)
    mag = np.abs(np.fft.rfft(fr, n=n_fft, axis=1)) + 1e-12
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    db = 20 * np.log10(mag)
    h1h2 = np.full(times.size, NAN)
    for i, f in enumerate(f0s):
        b1 = (freqs >= 0.85 * f) & (freqs <= 1.15 * f)
        b2 = (freqs >= 1.85 * f) & (freqs <= 2.15 * f)
        if b1.any() and b2.any():
            h1h2[i] = db[i, b1].max() - db[i, b2].max()
    # Cepstral peak prominence (Hillenbrand): power cepstrum of the dB spectrum, expressed in dB; CPP is
    # the height of the rahmonic peak (60-330 Hz quefrency range) above a straight line fitted to the
    # cepstrum over 1-20 ms. Scale-invariant, so the irfft normalisation does not matter.
    ceps = np.fft.irfft(db, axis=1)[:, : n_fft // 2]
    cdb = 20 * np.log10(np.abs(ceps) + 1e-12)
    q = np.arange(cdb.shape[1]) / sr
    fit = (q >= 0.001) & (q <= 0.020)
    qm = (q >= 1 / 330) & (q <= 1 / 60)
    cpp = np.full(times.size, NAN)
    if qm.sum() > 3 and fit.sum() > 3:
        A = np.vstack([q[fit], np.ones(fit.sum())]).T
        coef = np.linalg.lstsq(A, cdb[:, fit].T, rcond=None)[0]   # (2, n_frames)
        cq, qq = cdb[:, qm], q[qm]
        k = np.argmax(cq, axis=1)
        cpp = cq[np.arange(cq.shape[0]), k] - (coef[0] * qq[k] + coef[1])
    return h1h2, cpp


@expert("voice_quality", CATEGORY, cost="medium")
def voice_quality(ctx: Context) -> Evidence:
    """Phonation and articulation: Praat Burg formants F1-F3 (means, bandwidths) and how smoothly they
    move, H1-H2 spectral tilt and cepstral peak prominence (breathy vs pressed voice), long-term spectral
    slope, and the fraction of creaky / irregular phonation (vocal fry) - behaviours vocoders smooth over."""
    ev = Evidence(expert="voice_quality", category=CATEGORY)
    x, sr = _signal(ctx)
    v: Dict[str, float] = {}
    track = _f0(ctx) if x.size >= MIN_SIGNAL_S * sr and np.any(x) else None
    if track is None or np.sum(track["f0"] > 0) < MIN_VOICED_FRAMES:
        ev.applicable = False
        ev.features = _prefixed("voice_quality", VQ_FEATURES, v)
        return ev

    t, f0 = track["t"], track["f0"]
    voiced = f0 > 0
    med_f0 = float(np.median(f0[voiced]))

    # Formants: the Burg ceiling follows the usual Praat advice (5000 Hz male-range voices, 5500 Hz otherwise),
    # limited by Nyquist for low-rate files.
    snd = _praat_sound(ctx)
    ceiling = min(5000.0 if med_f0 < 160 else 5500.0, 0.5 * sr - 50)
    fm = snd.to_formant_burg(time_step=HOP_S, max_number_of_formants=5, maximum_formant=ceiling,
                             window_length=WIN_S, pre_emphasis_from=50.0)
    tv = t[voiced]
    F = np.array([[fm.get_value_at_time(k, ti) for ti in tv] for k in (1, 2, 3)], dtype=float)
    B = np.array([[fm.get_bandwidth_at_time(k, ti) for ti in tv] for k in (1, 2, 3)], dtype=float)
    ok = np.all(np.isfinite(F), axis=0)
    v["formant_dropout"] = 1.0 - ok.mean()
    if ok.sum() >= 10:
        for k in range(3):
            v[f"f{k + 1}_mean_hz"] = np.mean(F[k, ok])
            v[f"b{k + 1}_median_hz"] = np.median(B[k, ok & np.isfinite(B[k])])
        v["f1_std_hz"], v["f2_std_hz"] = np.std(F[0, ok]), np.std(F[1, ok])
        # Formant velocity within continuous voiced runs (index gaps split runs).
        vidx = np.flatnonzero(voiced)
        d1, d2, dd2 = [], [], []
        for a, b in _runs(voiced):
            sel = (vidx >= a) & (vidx < b)
            f1r, f2r = F[0, sel], F[1, sel]
            if f2r.size >= MIN_RUN_FRAMES and np.all(np.isfinite(f2r)) and np.all(np.isfinite(f1r)):
                d1.append(np.diff(f1r))
                d2.append(np.diff(f2r))
                dd2.append(np.diff(f2r, 2))
        if d2:
            D1, D2, DD2 = np.concatenate(d1), np.concatenate(d2), np.concatenate(dd2)
            v["f1_velocity_hz"] = np.median(np.abs(D1))
            v["f2_velocity_hz"] = np.median(np.abs(D2))
            if DD2.size >= 10 and np.any(D2):
                v["f2_jerk"] = np.sqrt(np.mean(DD2 ** 2)) / np.sqrt(np.mean(D2 ** 2))

    # Harmonic-level voice quality on voiced frames.
    h1h2, cpp = _harmonic_measures(x, sr, tv, f0[voiced])
    if np.isfinite(h1h2).sum() >= 5:
        v["h1h2_db"] = np.nanmedian(h1h2)
        v["h1h2_std_db"] = np.nanstd(h1h2)
    if np.isfinite(cpp).sum() >= 5:
        v["cpp_db"] = np.nanmean(cpp)

    # Long-term average spectrum slope of voiced frames, 100 Hz - min(5 kHz, Nyquist).
    fr = _frames(ctx)
    nfr = min(fr["t"].size, fr["voiced"].size)
    win, hop = max(16, int(round(WIN_S * sr))), max(1, int(round(HOP_S * sr)))
    vi = np.flatnonzero(fr["voiced"][:nfr])
    if vi.size >= 10:
        starts = np.clip(vi * hop, 0, max(0, x.size - win))
        seg = np.stack([x[s:s + win] if s + win <= x.size else np.pad(x[s:], (0, s + win - x.size)) for s in starts])
        n_fft = 1 << int(np.ceil(np.log2(win)))
        P = np.mean(np.abs(np.fft.rfft(seg * np.hanning(win), n=n_fft, axis=1)) ** 2, axis=0) + 1e-18
        f = np.fft.rfftfreq(n_fft, 1 / sr)
        m = (f >= 100) & (f <= min(5000.0, sr / 2 - 100))
        if m.sum() > 5:
            v["ltas_slope_db_khz"] = np.polyfit(f[m] / 1000.0, 10 * np.log10(P[m]), 1)[0]

    # Creak / irregular phonation: among sonorant frames (loud, low-frequency dominated), those Praat
    # cannot track (aperiodic) or that sit far below the speaker's median F0 (period doubling / fry).
    noise, level = _levels(fr)
    son = (fr["db"][:nfr] >= level - 20) & (fr["centroid_hz"][:nfr] < 1500)
    f0_fr = _voicing_at(track, fr["t"][:nfr])
    low = (f0_fr > 0) & (f0_fr < 0.6 * med_f0)
    aper = f0_fr <= 0
    if son.sum() >= 10:
        v["low_f0_fraction"] = (son & low).sum() / son.sum()
        v["aperiodic_sonorant_fraction"] = (son & aper).sum() / son.sum()
        v["creak_fraction"] = (son & (low | aper)).sum() / son.sum()

    ev.features = _prefixed("voice_quality", VQ_FEATURES, v)
    ev.artifacts = {"formant_t": tv.astype(np.float32), "f1_hz": F[0].astype(np.float32),
                    "f2_hz": F[1].astype(np.float32), "f3_hz": F[2].astype(np.float32)}
    ev.findings = _vq_findings(ev.features)
    ev.score = _net_score(ev.findings)
    return ev


def _vq_findings(F: Dict[str, float]) -> List[Finding]:
    out: List[Finding] = []
    jerk = F["voice_quality_f2_jerk"]
    lo, hi = REF["f2_jerk"]
    if np.isfinite(jerk) and jerk < 0.75 * lo:
        out.append(Finding(
            f"Formant (vocal-tract resonance) trajectories are unusually smooth: roughness index {jerk:.2f} vs "
            f"about {lo:.1f}-{hi:.1f} in natural speech. Real articulators move with small irregularities that "
            f"frame-predicting acoustic models average away.", +1, 0.15))
    creak = F["voice_quality_creak_fraction"]
    cpp = F["voice_quality_cpp_db"]
    if np.isfinite(cpp) and cpp < REF["cpp_db"][0] - 2.5:
        out.append(Finding(
            f"Weak harmonic structure: cepstral peak prominence {cpp:.1f} dB (natural clips here "
            f"{REF['cpp_db'][0]:.0f}-{REF['cpp_db'][1]:.0f} dB). A blurred harmonic comb is typical of older "
            f"vocoders (phase/excitation artefacts); a noisy or reverberant channel can also lower CPP.", +1, 0.15))
    if not out:
        out.append(Finding(
            f"Voice quality within the natural range: cepstral peak prominence {cpp:.1f} dB (higher = more "
            f"periodic), H1-H2 {F['voice_quality_h1h2_db']:.1f} dB (higher = breathier), creaky/irregular "
            f"phonation {100 * creak:.0f}% of loud voiced frames, F1/F2/F3 means "
            f"{F['voice_quality_f1_mean_hz']:.0f}/{F['voice_quality_f2_mean_hz']:.0f}/"
            f"{F['voice_quality_f3_mean_hz']:.0f} Hz.", 0, 0.0))
    return sorted(out, key=lambda f: -f.strength)[:4]
