"""Acoustic-environment forensics: mains hum (ENF), background noise floor, reverberation / replay cues.

A real recording is made *somewhere*: a microphone in a room, powered from the electric grid, with a
continuous background (room tone, preamp hiss, hum). A speech synthesizer has no room, no grid and no
microphone, so it either produces no environment at all (digitally clean pauses, no hum, abrupt offsets)
or an environment learned from its training data that is not physically consistent (background that
changes between pauses, noise under speech that differs from noise in the pauses). The three experts
here measure those physical traces:

  enf            electric-network-frequency lines at 50/60 Hz and harmonics, and their stability over time
  noise_floor    level, stationarity and colour of the background in the pauses, and whether the
                 background under the speech matches the background in the pauses
  reverb_replay  blind reverberation estimates (offset decay rate, SRMR-like modulation ratio, tail
                 drop) and loudspeaker-like band-pass shape (replay attacks)

All three share one framing of the signal (25 ms Hann frames, 10 ms hop), computed once by
`frame_analysis()` and cached in ctx.cache; the splice expert reuses it too.
"""
from __future__ import annotations

from functools import lru_cache
from math import gcd
from typing import Dict, List

import numpy as np
from scipy import signal

from ..evidence import Context, Evidence, Finding, expert

# Samples with |x| below half an LSB of 16-bit PCM are "digital silence": they are exactly zero in any
# 16-bit file and indistinguishable from zero after quantisation for any other source.
ZERO_EPS = 0.5 / 32768
FRAME_KEY = "frames@25ms/10ms"   # ctx.cache key of the shared frame analysis (parameters in the name)
N_BANDS = 16                     # log-spaced analysis bands used for noise spectra
BAND_LO_HZ = 100.0
BAND_HI_HZ = 7000.0              # the test set is low-passed near 7.5 kHz; stay below it


# --------------------------------------------------------------------------------------------------
# Shared helpers (also imported by hearsay.experts.splice)
# --------------------------------------------------------------------------------------------------

def mono(ctx: Context) -> np.ndarray:
    """ctx.audio as a finite float64 mono vector (defensive against 2-D or NaN input)."""
    x = np.asarray(ctx.audio, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=int(np.argmin(x.shape)))
    return np.nan_to_num(x.ravel(), nan=0.0, posinf=0.0, neginf=0.0)


def runs(mask: np.ndarray, min_len: int = 1) -> List[tuple]:
    """(start, end) index pairs (end exclusive) of True-runs in a boolean vector, at least min_len long."""
    m = np.asarray(mask, dtype=np.int8)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate(([0], m, [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


def finalize(ev: Evidence, defaults: Dict[str, float]) -> Evidence:
    """Guarantee a fixed, finite, float feature vector and a findings-based lean score.

    Fusion needs the same feature names for every file, so any feature an expert could not compute
    falls back to its neutral default. Findings are capped at 4 (strongest first).
    """
    feats = {}
    for k, default in defaults.items():
        v = ev.features.get(k, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = float(default)
        feats[k] = v if np.isfinite(v) else float(default)
    ev.features = feats
    ev.findings = sorted(ev.findings, key=lambda f: -f.strength)[:4]
    if ev.findings:
        ev.score = float(np.clip(sum(f.direction * f.strength for f in ev.findings), -1.0, 1.0))
    return ev


def frame_analysis(ctx: Context) -> dict:
    """Frame-level level/spectrum analysis shared by the environment and splice experts (cached).

    Returns a dict with per-frame RMS level (dBFS), digital-silence / pause / speech masks, power
    spectra, 16 log-spaced band powers, the noise floor (10th percentile of non-silent frame levels),
    the speech level (95th percentile), the pause segments (>= 50 ms) and the long pauses (>= 80 ms,
    with a robust background level each).
    """
    fa = ctx.cache.get(FRAME_KEY)
    if fa is not None:
        return fa
    x, sr = mono(ctx), int(ctx.sr)
    win, hop = max(16, int(round(0.025 * sr))), max(1, int(round(0.010 * sr)))
    n_fft = 1 << int(np.ceil(np.log2(win)))
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    silent = np.max(np.abs(frames), axis=1) < ZERO_EPS
    # Levels and spectra use mean-removed frames: a DC offset is not acoustic energy and must not turn
    # pauses into "speech" (DC offsets and steps are analysed separately by the splice expert).
    ac = frames - frames.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(ac ** 2, axis=1))
    rms_db = 20 * np.log10(np.maximum(rms, 1e-6))          # floor at -120 dBFS

    w = np.hanning(win)
    spec = np.abs(np.fft.rfft(ac * w, n=n_fft, axis=1)) ** 2 / (np.sum(w ** 2) * sr)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # log-spaced bands between 100 Hz and min(7 kHz, 0.45 sr); every band gets at least one bin
    hi = min(BAND_HI_HZ, 0.45 * sr)
    edges = np.geomspace(BAND_LO_HZ, hi, N_BANDS + 1)
    bmat = np.zeros((len(freqs), N_BANDS))
    for b in range(N_BANDS):
        sel = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if not sel.any():
            sel = np.abs(freqs - np.sqrt(edges[b] * edges[b + 1])) == np.min(np.abs(freqs - np.sqrt(edges[b] * edges[b + 1])))
        bmat[sel, b] = 1.0 / sel.sum()
    band_pow = spec @ bmat + 1e-20

    active = ~silent
    lv = rms_db[active]
    floor_db = float(np.percentile(lv, 10)) if lv.size else -120.0
    speech_db = float(np.percentile(lv, 95)) if lv.size else -120.0
    dyn = speech_db - floor_db
    pause_thr = floor_db + float(np.clip(0.25 * dyn, 3.0, 10.0))
    pause = active & (rms_db <= pause_thr)
    speech = active & (rms_db >= max(pause_thr + 3.0, speech_db - 25.0))

    fa = dict(
        sr=sr, win=win, hop=hop, n_frames=len(rms_db),
        times=(np.arange(len(rms_db)) * hop + win / 2) / sr,
        rms_db=rms_db, silent=silent, pause=pause, speech=speech,
        floor_db=floor_db, speech_db=speech_db, pause_thr_db=pause_thr,
        spec=spec, freqs=freqs, band_edges=edges, band_pow=band_pow,
        pause_segments=runs(pause, min_len=5),      # >= 50 ms of consecutive pause frames
        # Pauses long enough (>= 80 ms) to show the background rather than a word's decaying tail, with their
        # level taken as the 20th percentile of frame levels (robust to residual speech inside the pause).
        long_pauses=[(s, e, float(np.percentile(rms_db[s:e], 20))) for s, e in runs(pause, min_len=8)],
    )
    ctx.cache[FRAME_KEY] = fa
    return fa


def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    g = gcd(int(sr), int(target))
    return signal.resample_poly(x, int(target) // g, int(sr) // g)


def _parabolic_peak(p_db: np.ndarray, i: int) -> float:
    """Sub-bin offset of a spectral peak from a parabola through the log-power of bins i-1, i, i+1."""
    if i <= 0 or i >= len(p_db) - 1:
        return 0.0
    a, b, c = p_db[i - 1], p_db[i], p_db[i + 1]
    den = a - 2 * b + c
    return float(0.5 * (a - c) / den) if den < 0 else 0.0


# --------------------------------------------------------------------------------------------------
# ENF / mains hum
# --------------------------------------------------------------------------------------------------

ENF_FS = 1000                 # analysis rate after decimation (Hz)
ENF_NFFT = 16384              # zero-padded FFT of 1 s windows -> 0.06 Hz grid
ENF_HARMONICS = (1, 2, 3)
ENF_FUND_DB = 12.0            # fundamental line this far above its neighbours => hum present
ENF_HARM_DB = 10.0            # ...or at least two harmonics this far above their neighbours
ENF_SHARP_DB = 8.0            # a pure tone falls ~15 dB within 1.5 Hz (1 s Hann window); broad bumps do not
ENF_TRACK_DB = 20.0           # line SNR above which the 1 s-window track is precise enough to judge jumps
ENF_JUMP_HZ = 0.1             # frequency jump (fundamental-equivalent) that no real grid makes in < 1 s
ENF_MIN_DBFS = -85.0          # lines weaker than this are at 16-bit resolution: "SNR" against near-silence

ENF_DEFAULTS = {
    "enf_hum50_snr_db": 0.0, "enf_hum60_snr_db": 0.0, "enf_fund_snr_db": 0.0, "enf_harmonics_detected": 0.0,
    "enf_present": 0.0, "enf_nominal_hz": 0.0, "enf_level_dbfs": -120.0, "enf_freq_dev_hz": 0.0,
    "enf_track_step_hz": 0.0, "enf_track_std_hz": 0.0, "enf_track_max_jump_hz": 0.0,
    "enf_level_std_db": 0.0, "enf_level_max_jump_db": 0.0,
}


def _line(P: np.ndarray, f: np.ndarray, fc: float, tol: float):
    """Strongest candidate line within +-tol of fc: (snr_db, is_line, index).

    snr_db is the peak power relative to the median power 3-10 Hz away (outside the Hann main lobe).
    is_line requires the peak to be a genuine narrow line: not on the edge of the search band (that is the
    flank of something outside it) and at least ENF_SHARP_DB above the spectrum 1.5 Hz either side.
    """
    band = np.flatnonzero(np.abs(f - fc) <= tol)
    ref = (np.abs(f - fc) >= 3.0) & (np.abs(f - fc) <= 10.0)
    if band.size < 3 or not ref.any():
        return 0.0, False, int(np.argmin(np.abs(f - fc)))
    i = int(band[np.argmax(P[band])])
    snr = float(10 * np.log10(P[i] / (np.median(P[ref]) + 1e-30) + 1e-30))
    d = int(round(1.5 / (f[1] - f[0])))
    side = max(P[max(i - d, 0)], P[min(i + d, len(P) - 1)])
    sharp = 10 * np.log10(P[i] / (side + 1e-30) + 1e-30)
    return snr, bool(sharp >= ENF_SHARP_DB and band[0] < i < band[-1]), i


def _step_limit(snr_db: float) -> float:
    """Largest median window-to-window frequency step a real hum can show at this line SNR. The 1 s-window
    estimate is noise-limited (roughly 0.1-0.2 Hz at 10 dB in speech-dominated windows, shrinking 10x per
    20 dB); a spurious line jumps around the whole +-0.5 Hz search band."""
    return max(0.06, 0.4 * 10 ** (-(snr_db - 10.0) / 20.0))


def _track(S: np.ndarray, f: np.ndarray, fc: float, tol: float, k: int):
    """Per-window peak frequency (interpolated, divided by the harmonic number k) and level near fc."""
    band = np.flatnonzero(np.abs(f - fc) <= tol)
    df = f[1] - f[0]
    track, level = [], []
    for s in S:
        i = int(band[np.argmax(s[band])])
        s_db = 10 * np.log10(s + 1e-30)
        track.append((f[i] + _parabolic_peak(s_db, i) * df) / k)
        level.append(s_db[i])
    return np.array(track), np.array(level)


def _split_change(track: np.ndarray, level: np.ndarray, skip: int):
    """Largest change of the hum between the windows before and after any split point.

    `skip` windows straddle a seam (each 1 s window overlaps it), so they are excluded; each side needs at least
    two windows. Returns (|frequency change| Hz, |level change| dB, split position in window units).
    """
    best = (0.0, 0.0, 0.0)
    n = len(track)
    for j in range(2, n - skip - 1):
        a, b = slice(0, j), slice(j + skip, n)
        if n - (j + skip) < 2:
            break
        dfq = abs(float(np.median(track[a]) - np.median(track[b])))
        dlv = abs(float(np.median(level[a]) - np.median(level[b])))
        if dfq + dlv / 100.0 > best[0] + best[1] / 100.0:
            best = (dfq, dlv, j + skip / 2.0)
    return best


@expert("enf", "environment", cost="cheap")
def enf(ctx: Context) -> Evidence:
    """Electric network frequency (mains hum): narrowband lines at 50/60 Hz and harmonics (100/120, 150/180 Hz)
    relative to neighbouring frequencies; when present, the hum frequency and level are tracked over time
    (1 s windows, 0.25 s hop) and checked for stability. Recordings made on grid-powered equipment often carry
    a hum whose frequency drifts only by millihertz; synthetic speech carries none, and a spliced or fabricated
    background can show a jump in the hum track."""
    ev = Evidence("enf", "environment")
    x, sr = mono(ctx), int(ctx.sr)
    dur = len(x) / max(sr, 1)
    if dur < 1.0 or sr < 500:
        ev.applicable = False
        ev.findings.append(Finding(f"Clip too short ({dur:.1f} s) for mains-hum analysis.", 0, 0.0))
        return finalize(ev, ENF_DEFAULTS)

    y = _resample(x, sr, ENF_FS)
    L = int(min(1.0, len(y) / ENF_FS) * ENF_FS)
    hop = ENF_FS // 4
    frames = np.lib.stride_tricks.sliding_window_view(y, L)[::hop]
    w = np.hanning(L)
    S = np.abs(np.fft.rfft((frames - frames.mean(axis=1, keepdims=True)) * w, n=ENF_NFFT, axis=1)) ** 2
    f = np.fft.rfftfreq(ENF_NFFT, 1.0 / ENF_FS)
    # The median over windows keeps a line that is present all the time and suppresses speech harmonics,
    # which wander with the pitch and are only present in some windows.
    P = np.median(S, axis=0) if len(S) >= 3 else S.mean(axis=0)
    keep = f <= 200.0
    ev.artifacts["enf_spectrum_hz"] = f[keep][::2]
    ev.artifacts["enf_spectrum_db"] = (10 * np.log10(P[keep] + 1e-30))[::2]

    feats = ev.features
    fam = {}
    for nominal in (50.0, 60.0):
        lines = [_line(P, f, k * nominal, 0.5 * k) for k in ENF_HARMONICS]
        snr = np.array([s if ok else min(s, 0.0) for s, ok, _ in lines])  # non-lines count as no evidence
        hit = bool(snr[0] >= ENF_FUND_DB or int((snr >= ENF_HARM_DB).sum()) >= 2)
        # level of the strongest harmonic as an equivalent sine re full scale (Hann coherent gain sum(w)/2)
        kb = int(np.argmax(snr))
        level_dbfs = float(10 * np.log10(4 * P[lines[kb][2]] / np.sum(w) ** 2 + 1e-30))
        too_weak = hit and level_dbfs < ENF_MIN_DBFS
        hit = hit and not too_weak
        fam[nominal] = dict(snr=snr, idx=[i for _, _, i in lines], hit=hit, present=False, step=0.0,
                            level_dbfs=level_dbfs, too_weak=too_weak)
        if hit:
            # Track the strongest harmonic window by window (frequencies expressed at the fundamental). A
            # genuine hum gives nearly the same frequency in consecutive windows; a spurious line scatters.
            # The median step (not the spread) is used so that a splice -- two coherent pieces and one
            # jump -- still counts as hum.
            k = ENF_HARMONICS[kb]
            track, level = _track(S, f, k * nominal, 0.5 * k, k)
            step = float(np.median(np.abs(np.diff(track)))) if len(track) > 1 else 0.0
            fam[nominal].update(kb=kb, k=k, track=track, level=level, step=step,
                                present=bool(len(track) < 3 or step <= _step_limit(snr[kb])))
    feats["enf_hum50_snr_db"] = float(fam[50.0]["snr"].max())
    feats["enf_hum60_snr_db"] = float(fam[60.0]["snr"].max())
    # prefer a family that tracks coherently, then one that passed the SNR test, then the stronger one
    nominal = max(fam, key=lambda n: (fam[n]["present"], fam[n]["hit"], np.sort(fam[n]["snr"])[-2:].sum()))
    F = fam[nominal]
    snrs, idxs, hit, present, step = F["snr"], F["idx"], F["hit"], F["present"], F["step"]
    feats.update(enf_fund_snr_db=float(snrs[0]), enf_harmonics_detected=float((snrs >= ENF_HARM_DB).sum()),
                 enf_present=float(present))

    if not present:
        best = max(feats["enf_hum50_snr_db"], feats["enf_hum60_snr_db"])
        weak = [n for n in fam if fam[n]["too_weak"]]
        if hit:
            why = (f"a candidate line {best:.1f} dB above its neighbours wanders by a median {step:.2f} Hz between "
                   f"1-s windows, so it is noise or speech rather than a mains hum")
        elif weak:
            why = (f"the only candidate line ({best:.1f} dB above its neighbours) has a level of "
                   f"{fam[weak[0]]['level_dbfs']:.0f} dBFS, at the resolution limit of 16-bit audio, so it is a "
                   f"quantisation artefact of near-silent passages rather than a hum")
        elif best > 0:
            why = (f"the strongest narrow line at 50/60 Hz or a harmonic is only {best:.1f} dB above the neighbouring "
                   f"spectrum (a real hum typically stands 10+ dB above it)")
        else:
            why = "there is no narrow spectral line at 50/60 Hz or their harmonics (100/120, 150/180 Hz)"
        ev.findings.append(Finding(
            f"No mains hum (electric network frequency) detected: {why}. Studio recordings and synthesized speech "
            f"both usually lack hum, so its absence is weak evidence either way.", 0, 0.05))
        return finalize(ev, ENF_DEFAULTS)

    kb, k, track, level = F["kb"], F["k"], F["track"], F["level"]
    t_track = (np.arange(len(track)) * hop + L / 2) / ENF_FS
    amp_dbfs = F["level_dbfs"]
    df_split, dl_split, t_split = _split_change(track, level, L // hop - 1)
    t_seam = float(t_track[0] + t_split * hop / ENF_FS) if df_split > 0 else 0.0
    feats.update(
        enf_nominal_hz=nominal, enf_level_dbfs=float(amp_dbfs), enf_freq_dev_hz=float(np.median(track) - nominal),
        enf_track_step_hz=step, enf_track_std_hz=float(np.std(track)), enf_track_max_jump_hz=df_split,
        enf_level_std_db=float(np.std(level)), enf_level_max_jump_db=dl_split,
    )
    ev.artifacts.update(enf_track_t_s=t_track, enf_track_hz=track, enf_track_level_db=level)

    ev.findings.append(Finding(
        f"Mains hum detected: {nominal:.0f} Hz family, strongest at {k * nominal:.0f} Hz, {snrs[kb]:.1f} dB above "
        f"the neighbouring spectrum (level {amp_dbfs:.0f} dBFS, measured {np.median(track):.3f} Hz). Hum is picked "
        f"up by grid-powered recording equipment; speech synthesizers do not produce it unless it is deliberately "
        f"added, so a hum present throughout the clip points to a real recording chain.", -1, 0.3))
    if len(track) >= 3 and snrs[kb] >= ENF_TRACK_DB:
        if df_split > max(ENF_JUMP_HZ, 4 * step) or dl_split > 10.0:
            ev.artifacts["seams_s"] = [t_seam]
            ev.findings.append(Finding(
                f"The hum is not continuous: its frequency shifts by {df_split:.3f} Hz and/or its level by "
                f"{dl_split:.1f} dB around t = {t_seam:.2f} s. A real grid drifts by only millihertz within a "
                f"second, so this suggests audio from different recordings was joined, or the hum was pasted in.",
                +1, 0.4))
        else:
            ev.findings.append(Finding(
                f"The hum is stable over the clip (frequency spread {np.std(track) * 1000:.0f} mHz, level spread "
                f"{np.std(level):.1f} dB), consistent with one continuous recording.", -1, 0.15))
    return finalize(ev, ENF_DEFAULTS)


# --------------------------------------------------------------------------------------------------
# Background noise floor
# --------------------------------------------------------------------------------------------------

NOISE_DEFAULTS = {
    "noise_floor_dbfs": -120.0, "noise_floor_speech_level_dbfs": -120.0, "noise_floor_snr_db": 0.0,
    "noise_floor_pause_frac": 0.0, "noise_floor_n_pauses": 0.0,
    "noise_floor_pause_level_range_db": 0.0, "noise_floor_pause_level_std_db": 0.0,
    "noise_floor_pause_spec_dev_db": 0.0, "noise_floor_flatness": 0.0, "noise_floor_tilt_db_oct": 0.0,
    "noise_floor_under_speech_diff_db": 0.0, "noise_floor_under_speech_corr": 0.0,
    "noise_floor_digital_silence_frac": 0.0,
}
CLEAN_FLOOR_DBFS = -70.0      # pauses cleaner than this are unusual for a microphone recording
FLOOR_RANGE_DB = 12.0         # background level change between long pauses that deserves a finding


@expert("noise_floor", "environment", cost="cheap")
def noise_floor(ctx: Context) -> Evidence:
    """Background noise in the pauses: noise-floor level (10th-percentile frame RMS, dBFS), speech-to-noise ratio,
    stationarity of the background across separate pauses (level and spectral shape), colour of the noise
    (spectral flatness and tilt: white hiss vs rumble) and whether the noise under the speech (minimum statistics
    over speech frames) matches the noise in the pauses. A single real recording has one continuous background;
    synthesized audio often has digitally clean pauses, and edited audio can show backgrounds that change
    between pauses or differ between speech and pauses."""
    ev = Evidence("noise_floor", "environment")
    fa = frame_analysis(ctx)
    feats = ev.features
    rms_db, pause, speech, silent = fa["rms_db"], fa["pause"], fa["speech"], fa["silent"]
    n = max(fa["n_frames"], 1)
    feats.update(
        noise_floor_dbfs=fa["floor_db"], noise_floor_speech_level_dbfs=fa["speech_db"],
        noise_floor_snr_db=fa["speech_db"] - fa["floor_db"], noise_floor_pause_frac=pause.sum() / n,
        noise_floor_digital_silence_frac=silent.sum() / n,
    )
    if silent.all():
        ev.applicable = False
        ev.findings.append(Finding("The clip is entirely digital silence; there is no background to analyse.", 0, 0.0))
        return finalize(ev, NOISE_DEFAULTS)

    freqs, spec, band_pow = fa["freqs"], fa["spec"], fa["band_pow"]
    edges = fa["band_edges"]
    centers = np.sqrt(edges[:-1] * edges[1:])
    in_band = (freqs >= edges[0]) & (freqs <= edges[-1])

    # --- colour of the pause noise (all pause frames pooled) ---
    if pause.sum() >= 3:
        P = spec[pause].mean(axis=0)[in_band] + 1e-30
        feats["noise_floor_flatness"] = float(np.exp(np.mean(np.log(P))) / np.mean(P))
        pause_band_db = 10 * np.log10(band_pow[pause].mean(axis=0))
        feats["noise_floor_tilt_db_oct"] = float(np.polyfit(np.log2(centers), pause_band_db, 1)[0])
        ev.artifacts["noise_pause_band_db"] = pause_band_db
    else:
        pause_band_db = None

    # --- stationarity across separate (long) pauses ---
    segs = [(s, e) for s, e, _ in fa["long_pauses"]]
    seg_levels = [lv for _, _, lv in fa["long_pauses"]]
    feats["noise_floor_n_pauses"] = float(len(segs))
    seg_spec_dev = []
    for s, e in segs:
        if pause_band_db is not None:
            seg_db = 10 * np.log10(band_pow[s:e].mean(axis=0))
            seg_spec_dev.append(float(np.sqrt(np.mean((seg_db - pause_band_db) ** 2))))
    if len(seg_levels) >= 2:
        feats["noise_floor_pause_level_range_db"] = float(np.ptp(seg_levels))
        feats["noise_floor_pause_level_std_db"] = float(np.std(seg_levels))
        feats["noise_floor_pause_spec_dev_db"] = float(np.mean(seg_spec_dev)) if seg_spec_dev else 0.0
    ev.artifacts["noise_pause_segments_s"] = [(float(fa["times"][s]), float(fa["times"][e - 1])) for s, e in segs]
    ev.artifacts["noise_pause_levels_db"] = seg_levels

    # --- background under speech (minimum statistics) vs background in pauses ---
    if pause_band_db is not None and speech.sum() >= 10:
        under_db = 10 * np.log10(np.percentile(band_pow[speech], 10, axis=0))
        diff = under_db - pause_band_db
        feats["noise_floor_under_speech_diff_db"] = float(np.median(diff))
        a, b = under_db - under_db.mean(), pause_band_db - pause_band_db.mean()
        feats["noise_floor_under_speech_corr"] = float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12))
        ev.artifacts["noise_under_speech_band_db"] = under_db
    ev.artifacts["noise_band_centers_hz"] = centers

    # --- findings ---
    fl, snr = feats["noise_floor_dbfs"], feats["noise_floor_snr_db"]
    sil_frac = feats["noise_floor_digital_silence_frac"]
    if fl < CLEAN_FLOOR_DBFS:
        ev.findings.append(Finding(
            f"The pauses are almost perfectly clean: noise floor {fl:.0f} dBFS (speech at {fa['speech_db']:.0f} dBFS, "
            f"SNR {snr:.0f} dB). Real microphones and preamps nearly always leave a background above about "
            f"{CLEAN_FLOOR_DBFS:.0f} dBFS; such clean pauses are typical of synthesized speech or of aggressive "
            f"noise gating/denoising.", +1, 0.25))
    else:
        flat, tilt = feats.get("noise_floor_flatness", 0.0), feats.get("noise_floor_tilt_db_oct", 0.0)
        colour = ("white-like hiss" if flat > 0.5 else
                  "low-frequency-heavy (rumble/hum-like)" if tilt < -4 else "coloured")
        ev.findings.append(Finding(
            f"Background noise floor {fl:.0f} dBFS in the pauses (speech at {fa['speech_db']:.0f} dBFS, SNR "
            f"{snr:.0f} dB); the background is {colour} (spectral flatness {flat:.2f}, tilt {tilt:.1f} "
            f"dB/octave).", 0, 0.05))
    if sil_frac > 0.02:
        ev.findings.append(Finding(
            f"{sil_frac * 100:.0f}% of the clip is exact digital silence (no background at all), which a live "
            f"microphone does not produce by itself; it points to generated audio or to digital editing/padding "
            f"(see the splice expert for where it occurs).", +1, 0.15))
    if len(seg_levels) >= 2:
        rng = feats["noise_floor_pause_level_range_db"]
        if rng >= FLOOR_RANGE_DB:
            i_lo, i_hi = int(np.argmin(seg_levels)), int(np.argmax(seg_levels))
            ev.findings.append(Finding(
                f"The background level changes by {rng:.0f} dB between pauses (quietest pause at "
                f"{fa['times'][segs[i_lo][0]]:.2f} s, loudest at {fa['times'][segs[i_hi][0]]:.2f} s). One continuous "
                f"recording keeps a steady background; large changes suggest synthesis or concatenated/edited audio, "
                f"though a noise gate, a denoiser or a changing environment can also cause them.", +1, 0.15))
        elif len(seg_levels) >= 3 and rng < 4.0:
            ev.findings.append(Finding(
                f"The background is steady across {len(seg_levels)} pauses (level range {rng:.1f} dB), consistent "
                f"with one continuous recording.", -1, 0.1))
    return finalize(ev, NOISE_DEFAULTS)


# --------------------------------------------------------------------------------------------------
# Reverberation and replay (loudspeaker) cues
# --------------------------------------------------------------------------------------------------

REVERB_DEFAULTS = {
    "reverb_replay_n_offsets": 0.0, "reverb_replay_decay_db_s": 0.0, "reverb_replay_decay_fast_db_s": 0.0,
    "reverb_replay_t60_s": 0.0, "reverb_replay_tail_drop_50ms_db": 0.0, "reverb_replay_srmr_db": 0.0,
    "reverb_replay_lowcut_hz": 0.0, "reverb_replay_lf_ratio_db": 0.0, "reverb_replay_sub80_db": 0.0,
    "reverb_replay_ltas_ripple_db": 0.0,
}
REPLAY_LOWCUT_HZ = 250.0      # low-frequency roll-off above this is loudspeaker-like
REPLAY_LF_RATIO_DB = -25.0    # 50-150 Hz energy this far below 300-1000 Hz


def _offset_decays(fa: dict):
    """Energy decays after speech offsets: (slope in dB/s, drop after 50 ms in dB, start time) per offset.

    A decay segment starts at a frame within 20 dB of the speech level and continues while the level keeps
    falling (bumps < 0.5 dB tolerated). The slope is fitted between 3 dB below the start (skipping the direct
    sound) and 5 dB above the noise floor, where the room's reverberant tail dominates.
    """
    L, t, floor = fa["rms_db"].copy(), fa["times"], fa["floor_db"]
    L = np.maximum(L, floor - 10.0)          # digital-silence frames: clamp instead of -120 dB cliffs
    start_thr = fa["speech_db"] - 20.0
    out, i, n = [], 0, len(L)
    while i < n - 1:
        if L[i] >= start_thr and L[i + 1] < L[i]:
            j = i
            while j + 1 < n and L[j + 1] <= L[j] + 0.5:
                j += 1
            if j - i >= 4 and L[i] - L[j] >= 15.0:
                sel = np.arange(i, j + 1)
                sel = sel[(L[sel] <= L[i] - 3.0) & (L[sel] >= floor + 5.0)]
                if len(sel) >= 3:
                    slope = np.polyfit(t[sel], L[sel], 1)[0]
                    drop50 = L[i] - L[min(i + 5, n - 1)]
                    out.append((float(-slope), float(drop50), float(t[i])))
            i = max(j, i + 1)
        else:
            i += 1
    return out


@lru_cache(maxsize=8)
def _mel_bank(sr: int, n_fft: int) -> np.ndarray:
    import librosa
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=23, fmin=125.0, fmax=min(BAND_HI_HZ, 0.45 * sr))


def _srmr_like(x: np.ndarray, sr: int) -> float:
    """Speech-to-reverberation modulation energy ratio (simplified SRMR, dB).

    Band envelopes (23 mel bands, 4 ms hop) are Fourier-analysed over the clip; the ratio of modulation energy
    at 3-22 Hz (syllable rate, the speech itself) to 22-125 Hz (fast fluctuations that reverberation and
    noise fill in) drops as a room gets more reverberant.
    """
    n_fft, hop = 512, max(1, sr // 250)
    if len(x) < n_fft + 8 * hop:
        return 0.0
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    S = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    env = np.sqrt(S @ _mel_bank(sr, n_fft).T + 1e-20)     # (frames, bands) amplitude envelopes
    env = env - env.mean(axis=0, keepdims=True)
    env *= np.hanning(len(env))[:, None]
    fs_env = sr / hop
    M = np.abs(np.fft.rfft(env, axis=0)) ** 2
    fm = np.fft.rfftfreq(len(env), 1.0 / fs_env)
    lo = M[(fm >= 3) & (fm < 22)].sum()
    hi = M[(fm >= 22) & (fm < min(125, fs_env / 2))].sum()
    return float(10 * np.log10((lo + 1e-20) / (hi + 1e-20)))


@expert("reverb_replay", "environment", cost="medium")
def reverb_replay(ctx: Context) -> Evidence:
    """Blind room-acoustics and replay analysis: energy decay rate after speech offsets (dB/s, and an implied
    reverberation time T60), the level drop 50 ms after an offset (a direct-to-reverberant proxy), an SRMR-like
    speech-to-reverberation modulation ratio, and the long-term spectrum's low-frequency roll-off and ripple
    (small loudspeakers used in replay attacks cut the bass below ~200-300 Hz and add resonances). Real
    recordings carry the decay of a physical room; synthetic speech often stops abruptly or has decays
    inconsistent with its apparent room."""
    ev = Evidence("reverb_replay", "environment")
    fa = frame_analysis(ctx)
    x, sr = mono(ctx), int(ctx.sr)
    feats = ev.features
    if fa["silent"].all() or len(x) < 0.5 * sr:
        ev.applicable = False
        ev.findings.append(Finding("Clip is silent or too short for room-acoustic analysis.", 0, 0.0))
        return finalize(ev, REVERB_DEFAULTS)

    # --- decay after speech offsets ---
    dec = _offset_decays(fa)
    feats["reverb_replay_n_offsets"] = float(len(dec))
    if dec:
        rates = np.array([d[0] for d in dec])
        feats["reverb_replay_decay_db_s"] = float(np.median(rates))
        # the room limits how fast energy can fall, so the fastest decays measure the room, not the talker
        fast = float(np.percentile(rates, 80))
        feats["reverb_replay_decay_fast_db_s"] = fast
        feats["reverb_replay_t60_s"] = float(np.clip(60.0 / max(fast, 1e-3), 0.0, 5.0))
        feats["reverb_replay_tail_drop_50ms_db"] = float(np.median([d[1] for d in dec]))
        ev.artifacts["reverb_offsets_s"] = [d[2] for d in dec]
        ev.artifacts["reverb_offset_rates_db_s"] = rates

    feats["reverb_replay_srmr_db"] = _srmr_like(x, sr)

    # --- long-term average spectrum: low-frequency roll-off and ripple ---
    nper = int(min(len(x), 2048 * sr // 16000))
    f, ltas = signal.welch(x, fs=sr, nperseg=nper, noverlap=nper // 2)
    ltas_db = 10 * np.log10(ltas + 1e-30)
    sm = np.convolve(ltas_db, np.ones(3) / 3, mode="same")
    mid = (f >= 300) & (f <= 1000)
    if mid.any():
        mid_db = float(np.median(sm[mid]))
        # low cut = lower edge of the continuous pass band: walk down from 300 Hz until the LTAS first falls
        # 20 dB below the 300-1000 Hz level
        lowcut = 20.0
        for i in np.flatnonzero((f >= 20) & (f < 300))[::-1]:
            if sm[i] < mid_db - 20.0:
                lowcut = float(f[min(i + 1, len(f) - 1)])
                break
        feats["reverb_replay_lowcut_hz"] = lowcut

        def band_db(a, b):
            s = (f >= a) & (f < b)
            return 10 * np.log10(ltas[s].mean() + 1e-30) if s.any() else -300.0

        ref = band_db(300, 1000)
        feats["reverb_replay_lf_ratio_db"] = float(band_db(50, 150) - ref)
        feats["reverb_replay_sub80_db"] = float(band_db(20, 80) - ref)
        # ripple: deviation from a 1/3-octave-smoothed LTAS between 300 Hz and 5 kHz
        rip = (f >= 300) & (f <= min(5000, 0.45 * sr))
        if rip.sum() > 5:
            smooth = np.array([np.mean(ltas_db[(f >= fi / 2 ** (1 / 6)) & (f <= fi * 2 ** (1 / 6))]) for fi in f[rip]])
            feats["reverb_replay_ltas_ripple_db"] = float(np.std(ltas_db[rip] - smooth))
    keep = f <= 8000
    ev.artifacts["reverb_ltas_hz"], ev.artifacts["reverb_ltas_db"] = f[keep], ltas_db[keep]

    # --- findings ---
    if dec:
        t60, fast = feats["reverb_replay_t60_s"], feats["reverb_replay_decay_fast_db_s"]
        room = ("very dry (close-miked studio or synthetic)" if t60 < 0.15 else
                "moderately damped room" if t60 < 0.5 else "reverberant room")
        ev.findings.append(Finding(
            f"Energy after speech offsets falls at up to {fast:.0f} dB/s ({len(dec)} offsets analysed), implying a "
            f"reverberation time of about {t60:.2f} s: a {room}. The level drops {feats['reverb_replay_tail_drop_50ms_db']:.0f} dB "
            f"within 50 ms of an offset (larger = drier/closer microphone).", 0, 0.05))
    else:
        ev.findings.append(Finding(
            "No clean speech offsets (15+ dB decays into a pause) were found, so reverberation could not be "
            "estimated.", 0, 0.0))
    lc, lf = feats["reverb_replay_lowcut_hz"], feats["reverb_replay_lf_ratio_db"]
    if lc >= REPLAY_LOWCUT_HZ and lf <= REPLAY_LF_RATIO_DB:
        ev.findings.append(Finding(
            f"The long-term spectrum rolls off below about {lc:.0f} Hz (50-150 Hz band {lf:.0f} dB below the "
            f"300-1000 Hz band). This band-pass shape is typical of speech played back through a small loudspeaker "
            f"and re-recorded (a replay attack), though telephone channels and some voices produce it too.", +1, 0.25))
    return finalize(ev, REVERB_DEFAULTS)
