"""Compression forensics: effective bandwidth (codec / resampler cut-off) and traces of earlier lossy coding.

Two experts live here:

``codec_cutoff`` (cheap)
    Every digital audio chain leaves a band edge: a telephone codec stops near 3.4-4 kHz, a 16 kHz
    file near 7-8 kHz, LJSpeech-style TTS vocoders (22.05 kHz) near 11 kHz, 24 kHz TTS / neural codecs
    near 12 kHz, MP3 encoders at the low-pass LAME picks for the bitrate. We estimate that edge from the
    long-term average spectrum with a step detector, measure how deep and how steep it is (a brick wall
    comes from a digital filter or codec, a gentle slope from microphones and rooms), and flag a
    *hidden* band limit: a file that is stored at a higher sample rate than its content uses, i.e. it
    was upsampled or transcoded from a narrower source.

``double_compression`` (medium)
    Lossy codecs quantise transform coefficients; at low bitrates whole coefficients become exactly
    zero, which leaves (a) spectral holes, near-silent time-frequency cells inside the passband,
    concentrated in the upper band, (b) loss of noise-likeness (spectral flatness) in the upper
    bands, and (c) a block grid: the codec works on fixed frames (MP3 granules of 576 samples, AAC
    frames of 1024), so re-analysing the decoded audio with a matching MDCT at the right offset
    reproduces those zeros. We scan every offset and look for a sharp sparsity peak that shows up in
    both halves of the clip. For an MP3 file we also predict where its *own* grid must sit (from the
    decoder's gapless trim), so a grid at the wrong phase, or a second grid, reveals double MP3
    coding; a 576-grid inside an AAC/Opus file reveals an MP3 -> AAC/Opus transcode.

Validated by encoding 10 test clips to MP3 at 32/64/128 kbps and AAC 64 kbps with ffmpeg and decoding
back to WAV (work/expert_tests/compression/validate_mp3.py): the grid fires on 10/10 MP3-32k, 1/10
MP3-64k, 0/10 MP3-128k and 0/10 originals; double MP3 coding (32k -> shifted -> 128k) was flagged
4/4. The organisers' 16 -> 22.05 -> 16 kHz round trip preserves a 16 kHz MP3 grid (z unchanged on the
32k transcodes), so the scan remains meaningful on the HEARSAY test set.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter

from ..audio import FFMPEG, probe
from ..evidence import Context, Evidence, Finding, expert

PCM16_FLOOR_DB = 10 * np.log10((2.0 ** -15) ** 2 / 12)   # per-bin power of 16-bit quantisation noise (~-101 dB)
LOSSY_DECODE = {"mp3", "mp2", "aac", "opus", "vorbis", "wmav2", "ac3", "eac3", "amr_nb", "amr_wb"}
MAX_SECONDS = 30.0                                         # analyse at most this much audio (cost bound)

# Known band edges (Hz). The label explains what usually produces that edge.
CUTOFF_CLASSES = [
    (2900, 4050, "narrowband telephony (8 kHz sampling: G.711 / GSM / AMR-NB)"),
    (4050, 5700, "11-12 kHz sampling or a very low-bitrate codec"),
    (5700, 6900, "low-bitrate lossy coding (e.g. MP3 at 24-40 kbps) or a 14-16 kHz chain with a conservative filter"),
    (6900, 8050, "16 kHz wideband chain (resampler anti-alias filter, AMR-WB / G.722 / Opus wideband)"),
    (8050, 9500, "Opus/MP3 mid-band low-pass or an 18-20 kHz chain"),
    (9500, 11100, "22.05 kHz chain (typical of LJSpeech-style neural TTS vocoders) or MP3 at ~56-64 kbps"),
    (11100, 12100, "24 kHz chain (XTTS, Bark, neural codecs, TTS APIs) or Opus super-wideband"),
    (12100, 16500, "MP3/AAC encoder low-pass at roughly 80-128 kbps"),
    (16500, 20000, "MP3/AAC encoder low-pass at 128 kbps or more"),
    (20000, 1e9, "full-band audio (no encoder low-pass)"),
]
TTS_RATE_CLASSES = ("22.05 kHz chain", "24 kHz chain")


def classify_cutoff(fc: float) -> str:
    return next((lab for lo, hi, lab in CUTOFF_CLASSES if lo <= fc < hi), "unusually narrow band (< 2.9 kHz)")


# --------------------------------------------------------------------------------------------------
# Shared intermediates
# --------------------------------------------------------------------------------------------------


def _container_info(ctx: Context) -> Dict[str, Any]:
    """Native codec / rate / bitrate, from the container expert's cache if it ran, else from ffprobe."""
    if "container" in ctx.cache:
        return ctx.cache["container"]
    pr = ctx.cache.get("ffprobe")
    if pr is None:
        pr = ctx.native.get("ffprobe") if isinstance(ctx.native.get("ffprobe"), dict) else \
            (ctx.native if "streams" in ctx.native else (probe(ctx.path) if ctx.path and os.path.isfile(ctx.path) else {}))
        ctx.cache["ffprobe"] = pr or {}
    st = next((s for s in (pr or {}).get("streams", []) if s.get("codec_type") == "audio"), {})
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
    return {"codec": st.get("codec_name", ""), "native_sr": int(num(st.get("sample_rate"))),
            "bitrate": num(st.get("bit_rate")) or num((pr or {}).get("format", {}).get("bit_rate")),
            "family": (pr or {}).get("format", {}).get("format_name", "")}


def _decoder_skip(ctx: Context, sr: int) -> int:
    """Samples FFmpeg trims from the start of a gapless MP3 (encoder delay + 529 decoder delay, from the LAME/Info tag).
    ffprobe reports it as the stream start_time; we need it to predict where the file's *own* MP3 grid must appear."""
    st = ctx.cache.get("ffprobe") or {}
    a = next((s for s in st.get("streams", []) if s.get("codec_type") == "audio"), {})
    try:
        return int(round(float(a.get("start_time", 0) or 0) * sr))
    except (TypeError, ValueError):
        return 0


def native_audio(ctx: Context) -> Tuple[np.ndarray, int]:
    """Mono float32 audio at the file's *native* sample rate (bandwidth above 8 kHz is invisible at 16 kHz).

    Falls back to ctx.audio when the native rate equals the analysis rate, is unknown, or decoding fails.
    """
    if "native_audio" in ctx.cache:
        return ctx.cache["native_audio"]
    x, sr = ctx.audio, ctx.sr
    info = _container_info(ctx)
    nsr = int(info.get("native_sr") or ctx.native.get("sr") or 0)
    lossy = (info.get("codec") or "") in LOSSY_DECODE
    if nsr and (nsr != ctx.sr or lossy) and ctx.path and os.path.isfile(ctx.path):
        y = None
        if not lossy:  # PCM/FLAC: soundfile is bit-exact; lossy: FFmpeg, whose gapless trimming fixes the grid offset
            try:
                import soundfile as sf
                y, s = sf.read(ctx.path, dtype="float32", always_2d=True, frames=int(MAX_SECONDS * nsr))
                y = y.mean(axis=1) if s == nsr else None
            except Exception:
                y = None
        if y is None:
            try:
                out = subprocess.run([FFMPEG, "-v", "error", "-t", str(MAX_SECONDS), "-i", ctx.path, "-f", "f32le",
                                      "-ac", "1", "pipe:1"], capture_output=True, timeout=60)
                y = np.frombuffer(out.stdout, dtype=np.float32).copy()
            except Exception:
                y = None
        if y is not None and len(y) > nsr // 4:
            x, sr = y, nsr
    x = np.nan_to_num(np.asarray(x, dtype=np.float32)[: int(MAX_SECONDS * sr)])
    ctx.cache["native_audio"] = (x, sr)
    return x, sr


def power_spectrogram(ctx: Context) -> Optional[Dict[str, Any]]:
    """Hann-windowed power spectrogram at the native rate (per-bin power normalised so white noise of
    variance s^2 reads 10*log10(s^2) dB), plus the speech-active frame mask. Cached for both experts."""
    if "compression_spec" in ctx.cache:
        return ctx.cache["compression_spec"]
    x, sr = native_audio(ctx)
    n = 1024 if sr <= 24000 else 2048 if sr <= 48000 else 4096
    spec = None
    if len(x) >= 2 * n:
        w = np.hanning(n).astype(np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(x, n)[:: n // 4] * w
        P = (np.abs(np.fft.rfft(frames, axis=1)) ** 2 / np.sum(w ** 2)).astype(np.float32)
        e = P.sum(axis=1)
        if e.max() > 0:
            active = e > max(np.percentile(e, 40), e.max() * 1e-5)   # speech-bearing frames (loudest 60%)
            spec = {"P": P, "freqs": np.fft.rfftfreq(n, 1.0 / sr), "active": active, "sr": sr, "n_fft": n}
    ctx.cache["compression_spec"] = spec
    return spec


# --------------------------------------------------------------------------------------------------
# Cut-off estimation
# --------------------------------------------------------------------------------------------------


def estimate_cutoff(freqs: np.ndarray, S: np.ndarray, nyq: float) -> Dict[str, float]:
    """Find the band edge in a long-term spectrum S (dB).

    Step detector: for every candidate frequency compare the median level in a ~300 Hz window below
    with the median level in a window above; the largest drop is the edge. The cut-off is then placed
    where the spectrum crosses half-way (in dB) between the passband level just below the edge and the
    stop-band floor above it; the 90%->10% crossing distance gives the transition width / steepness.
    """
    df = freqs[1] - freqs[0]
    S = median_filter(S, size=max(3, int(round(60 / df)) | 1), mode="nearest")
    w = max(4, int(round(300 / df)))
    lo = max(w, int(1000 / df))
    hi = len(S) - max(2, w // 2)                       # need at least half a window of stop band
    res = {"cutoff_hz": nyq, "depth_db": 0.0, "steep_db_per_khz": 0.0, "width_hz": 0.0, "drop_db": 0.0,
           "floor_db": float(np.median(S[-max(3, w // 2):])), "pass_db": float(np.median(S[lo:hi])) if hi > lo else 0.0,
           "found": 0.0}
    if hi <= lo:
        return res
    drops = np.array([np.median(S[i - w:i]) - np.median(S[i:min(len(S), i + w)]) for i in range(lo, hi)])
    k = int(np.argmax(drops)) + lo
    drop = float(drops[k - lo])
    res["drop_db"] = drop
    stop = S[k + 1:] if k + 1 < len(S) else S[-1:]
    floor = float(np.median(stop[: max(1, len(stop) - 2)]))
    band = S[max(0, k - int(1500 / df)):k]
    passlvl = float(np.percentile(band, 75)) if len(band) else float(S[k])
    depth = passlvl - floor
    res.update(floor_db=floor, pass_db=passlvl)
    if drop < 10 or depth < 15:                         # no convincing edge: bandwidth reaches Nyquist
        return res
    half = floor + 0.5 * depth
    # highest frequency (searching down from the edge region) still above the half-way level
    above = np.where(S[: min(len(S), k + w)] > half)[0]
    ic = int(above[-1]) if len(above) else k
    i90 = np.where(S[: ic + 1] >= floor + 0.9 * depth)[0]
    i90 = int(i90[-1]) if len(i90) else ic
    i10 = np.where(S[ic:] <= floor + 0.1 * depth)[0]
    i10 = ic + int(i10[0]) if len(i10) else len(S) - 1
    width = max(df, (i10 - i90) * df)
    res.update(cutoff_hz=float(freqs[ic]), depth_db=float(depth), width_hz=float(width),
               steep_db_per_khz=float(0.8 * depth / (width / 1000.0)), found=1.0)
    return res


def _cutoff_result(ctx: Context) -> Optional[Dict[str, Any]]:
    if "compression_cutoff" in ctx.cache:
        return ctx.cache["compression_cutoff"]
    spec = power_spectrogram(ctx)
    out = None
    if spec is not None and spec["active"].sum() >= 3:
        P = spec["P"][spec["active"]]
        S = 10 * np.log10(P.mean(axis=0) + 1e-20)
        nyq = spec["sr"] / 2.0
        out = estimate_cutoff(spec["freqs"], S, nyq)
        out.update(S=S, freqs=spec["freqs"], nyq=nyq, sr=spec["sr"])
    ctx.cache["compression_cutoff"] = out
    return out


def _decimate_for_plot(freqs: np.ndarray, S: np.ndarray, n: int = 400) -> Dict[str, list]:
    idx = np.linspace(0, len(S) - 1, min(n, len(S))).astype(int)
    return {"freq_hz": freqs[idx].round(1).tolist(), "level_db": S[idx].round(2).tolist()}


# --------------------------------------------------------------------------------------------------
# Expert 1: codec / bandwidth cut-off
# --------------------------------------------------------------------------------------------------

CUTOFF_DEFAULTS = {"codec_cutoff_hz": 0.0, "codec_cutoff_rel_nyquist": 1.0, "codec_cutoff_depth_db": 0.0,
                   "codec_cutoff_steep_db_per_khz": 0.0, "codec_cutoff_width_hz": 0.0, "codec_cutoff_floor_dbfs": 0.0,
                   "codec_cutoff_floor_above_pcm16_db": 0.0, "codec_cutoff_found": 0.0, "codec_cutoff_hidden": 0.0,
                   "codec_cutoff_tts_rate_class": 0.0, "codec_cutoff_telephony": 0.0, "codec_cutoff_bitrate_gap_hz": 0.0,
                   "codec_cutoff_native_sr": 0.0}


@expert("codec_cutoff", "compression", cost="cheap")
def analyze_cutoff(ctx: Context) -> Evidence:
    """Effective bandwidth: locates the low-pass edge of the long-term spectrum at the native sample rate (step detector),
    measures its depth and steepness (brick wall = digital filter/codec), names the codec/sample-rate chain that usually
    produces such an edge, and flags a hidden band limit (content narrower than the stored sample rate allows)."""
    ev = Evidence(expert="codec_cutoff", category="compression", features=dict(CUTOFF_DEFAULTS))
    r = _cutoff_result(ctx)
    if r is None:
        ev.applicable = False
        ev.findings.append(Finding("Too little non-silent audio to measure the signal bandwidth.", 0, 0.0))
        return ev
    info = _container_info(ctx)
    fc, nyq, sr = r["cutoff_hz"], r["nyq"], r["sr"]
    rel = fc / nyq
    label = classify_cutoff(fc)
    codec = info.get("codec", "") or ""
    native_lossy = codec in LOSSY_DECODE
    hidden = bool(r["found"] and rel < 0.85 and r["depth_db"] >= 20)
    telephony = bool((r["found"] and fc < 4050) or sr <= 8000)
    # "Fake quality" check: an MP3's measured bandwidth far below what LAME would keep at the declared bitrate.
    gap = 0.0
    if codec == "mp3" and info.get("bitrate"):
        from .container import lame_expected_lowpass   # local import: keeps this module usable on its own
        exp_lp = lame_expected_lowpass(info["bitrate"] / 1000.0)
        if exp_lp:
            gap = max(0.0, min(exp_lp, nyq) - fc)
    # A lossy file's own encoder low-pass explains a hidden band limit, unless (MP3) it is far below the expected one.
    own_codec_edge = hidden and native_lossy and not (codec == "mp3" and gap > 3000)
    tts_rate = hidden and not own_codec_edge and label.startswith(TTS_RATE_CLASSES)
    f = ev.features
    f.update({"codec_cutoff_hz": fc, "codec_cutoff_rel_nyquist": rel, "codec_cutoff_depth_db": r["depth_db"],
              "codec_cutoff_steep_db_per_khz": min(r["steep_db_per_khz"], 1000.0), "codec_cutoff_width_hz": r["width_hz"],
              "codec_cutoff_floor_dbfs": r["floor_db"], "codec_cutoff_floor_above_pcm16_db": r["floor_db"] - PCM16_FLOOR_DB,
              "codec_cutoff_found": r["found"], "codec_cutoff_hidden": float(hidden), "codec_cutoff_tts_rate_class": float(tts_rate),
              "codec_cutoff_telephony": float(telephony), "codec_cutoff_bitrate_gap_hz": gap, "codec_cutoff_native_sr": float(sr)})

    # ---- findings -------------------------------------------------------------------------------------
    brick = r["found"] and r["steep_db_per_khz"] >= 60
    shape = (f"a brick-wall edge ({r['steep_db_per_khz']:.0f} dB/kHz over {r['width_hz']:.0f} Hz, {r['depth_db']:.0f} dB deep), "
             "which only a digital filter or codec produces - microphones and rooms roll off gradually"
             if brick else f"a {r['steep_db_per_khz']:.0f} dB/kHz roll-off, {r['depth_db']:.0f} dB deep")
    floor_note = (" Above the edge the spectrum sits at the 16-bit quantisation floor (digital silence)."
                  if r["found"] and r["floor_db"] - PCM16_FLOOR_DB < 8 else "")
    if not r["found"]:
        ev.findings.append(Finding(f"Bandwidth: content extends to the Nyquist limit of the {sr / 1000:g} kHz sample rate "
                                   f"({nyq / 1000:.1f} kHz) with no clear low-pass edge; the file uses all the bandwidth it "
                                   "can store, so there is no sign of an earlier narrower stage.", 0, 0.0))
    elif own_codec_edge:
        ev.findings.append(Finding(f"Bandwidth: content stops at {fc / 1000:.2f} kHz although the {sr / 1000:g} kHz sample rate allows "
                                   f"{nyq / 1000:.1f} kHz. The file is {codec.upper()}-coded and lossy encoders low-pass the signal "
                                   f"to save bits, so this edge ({shape}) is most likely the encoder's own and says nothing "
                                   "about the voice's origin.", 0, 0.0))
    elif hidden:
        why = f"The spectrum shows {shape} (typical source: {label})."
        if tts_rate:
            why += (" Neural TTS vocoders commonly generate at 22.05 or 24 kHz, so an upsampled file of this kind is "
                    "consistent with (not proof of) a synthetic origin.")
        ev.findings.append(Finding(f"Hidden band limit: the file is stored at {sr / 1000:g} kHz (room for {nyq / 1000:.1f} kHz of "
                                   f"audio) but its content stops at {fc / 1000:.2f} kHz. It was upsampled or transcoded from a "
                                   f"narrower source. {why}{floor_note}", +1 if tts_rate else 0, 0.3 if tts_rate else 0.1))
    else:
        ev.findings.append(Finding(f"Bandwidth: content reaches {fc / 1000:.2f} kHz, {100 * rel:.0f}% of the {nyq / 1000:.1f} kHz "
                                   f"Nyquist limit, ending in {shape}. That edge is the file's own {sr / 1000:g} kHz "
                                   f"anti-alias/resampling filter ({label}); no narrower earlier stage is visible.{floor_note}",
                                   0, 0.0))
    if telephony:
        ev.findings.append(Finding(f"Telephone bandwidth: nothing above {min(fc, nyq) / 1000:.2f} kHz. The audio went through an 8 kHz "
                                   "telephony codec (real calls, voicemail, or a TTS clip deliberately degraded to hide "
                                   "artefacts). Many spectral detectors lose power in this band, so weigh their output "
                                   "cautiously.", 0, 0.1))
    if gap > 3000:
        ev.findings.append(Finding(f"Bitrate/bandwidth mismatch: the MP3 declares {info['bitrate'] / 1000:.0f} kbps, for which the "
                                   f"LAME encoder keeps ~{(fc + gap) / 1000:.1f} kHz, yet content stops at {fc / 1000:.1f} kHz. The "
                                   "audio was already band-limited before this encode (transcoded from a lower-quality or "
                                   "lower-rate source).", +1, 0.2))
    ev.score = None
    ev.artifacts = {"ltas": _decimate_for_plot(r["freqs"], r["S"]), "cutoff_hz": fc, "nyquist_hz": nyq,
                    "class": label, "pass_db": r["pass_db"], "floor_db": r["floor_db"]}
    return ev


# --------------------------------------------------------------------------------------------------
# Expert 2: traces of (double) lossy compression
# --------------------------------------------------------------------------------------------------


def _mdct_basis(N: int) -> np.ndarray:
    """MDCT analysis matrix (2N x N) with a sine window, as used by MP3's long blocks and AAC."""
    n = np.arange(2 * N)[:, None]
    k = np.arange(N)[None, :]
    w = np.sin(np.pi * (n + 0.5) / (2 * N))
    return (w * np.cos(np.pi / N * (n + 0.5 + N / 2) * (k + 0.5))).astype(np.float32)


def _robust_z(c: np.ndarray) -> np.ndarray:
    med = np.median(c)
    mad = np.median(np.abs(c - med)) * 1.4826 + 1e-12
    return (c - med) / mad


def grid_scan(x: np.ndarray, sr: int, N: int, fmax: float, n_frames: int = 80) -> Dict[str, Any]:
    """Codec block-grid detector.

    For each offset 0..N-1, analyse up to `n_frames` speech frames with an N-band MDCT whose blocks start at that
    offset and measure coefficient sparsity (share of coefficients at the 16-bit noise floor, and share 60 dB below
    the frame peak) in the passband below fmax. Decoded MP3/AAC audio shows a sharp, single-sample-wide sparsity
    peak at the codec's own frame alignment, because the quantiser zeroed exactly those coefficients.

    Frames: the loudest sixth is skipped (fewest zeroed coefficients) and the next `n_frames` are used. They are
    split into two interleaved halves and scanned separately; `zmin` is the weaker half's z at the common peak, so
    a genuine grid must show up in both halves. A second, distinct peak suggests two encodes with different
    alignment (double compression).
    """
    out = {"z": 0.0, "zmin": 0.0, "offset": -1.0, "second_z": 0.0, "second_offset": -1.0}
    if len(x) < 6 * N or fmax <= 0 or n_frames < 8:
        return out
    kmax = max(8, int(N * min(fmax, sr / 2) / (sr / 2)))
    B = _mdct_basis(N)[:, 2:kmax]                              # only the passband coefficients are needed
    q = (2.0 ** -15) / np.sqrt(12) * np.sqrt(N / 2)          # PCM16 noise per MDCT coefficient
    starts = np.arange(0, len(x) - 3 * N, N)
    energy = np.array([np.dot(x[s:s + 2 * N], x[s:s + 2 * N]) for s in starts])
    order = np.argsort(energy)[::-1]
    order = order[energy[order] > energy.max() * 1e-4]        # speech frames only (within 40 dB of the loudest)
    order = order[len(order) // 6:][:n_frames]
    if len(order) < 8:
        return out
    top = starts[order]
    curves = []
    for half in (top[0::2], top[1::2]):
        idx = half[:, None] + np.arange(2 * N)[None, :]
        at_floor, below_peak = np.empty(N), np.empty(N)
        for o in range(N):
            a = np.abs(x[idx + o] @ B)
            at_floor[o] = np.mean(a < 4 * q)
            below_peak[o] = np.mean(a < a.max(axis=1, keepdims=True) * 1e-3)
        curves.append(_robust_z(_robust_z(at_floor) + _robust_z(below_peak)))
    c = _robust_z(curves[0] + curves[1])
    k = int(np.argmax(c))
    masked = c.copy()
    masked[[(k + d) % N for d in range(-8, 9)]] = -np.inf     # the main peak has side-lobes a few samples wide
    k2 = int(np.argmax(masked))
    out.update(z=float(c[k]), zmin=float(min(curves[0][k], curves[1][k])), offset=float(k),
               second_z=float(masked[k2]), second_offset=float(k2), curve=c.astype(np.float32))
    return out


DC_DEFAULTS = {"double_compression_hole_frac": 0.0, "double_compression_hf_hole_frac": 0.0, "double_compression_nz_frac_pass": 0.0, "double_compression_nz_frac_hf": 0.0,
               "double_compression_nz_frac_stop": 0.0, "double_compression_hf_flatness": 0.0, "double_compression_flatness_step": 0.0, "double_compression_grid576_z": 0.0,
               "double_compression_grid576_zmin": 0.0, "double_compression_grid576_second_z": 0.0, "double_compression_grid1024_z": 0.0,
               "double_compression_grid1024_zmin": 0.0, "double_compression_grid_detected": 0.0, "double_compression_double_grid": 0.0, "double_compression_lossy_trace": 0.0,
               "double_compression_native_lossy": 0.0}

# Calibrated on 10 test clips transcoded with ffmpeg (work/expert_tests/compression/test_grid_frames2.py): originals
# peak at z <= 3.7 (weaker half <= 2.3); MP3 32 kbps @16 kHz median z 11.6, AAC 64 kbps @44.1 kHz median z 12.7
# (weaker half >= 5.8); MP3 64 kbps is only sometimes caught and 128 kbps never.
GRID_Z_STRONG = 5.0
GRID_ZMIN = 3.0
AAC_RATES = (22050, 24000, 32000, 44100, 48000)   # AAC scan only where AAC is common (saves ~0.5 s on 16 kHz PCM)
LOSSY_NATIVE = {"mp3": 576, "mp2": 576, "aac": 1024}
# Where the sine-window MDCT(576) sparsity peak of an MP3 lands in untrimmed decoder output (hybrid filter-bank delay).
# Measured: ElevenLabs MP3 without Info tag -> 240; LAME files (1105 samples trimmed) -> (240 - 1105) mod 576 = 287.
MP3_GRID_PHASE = 240


@expert("double_compression", "compression", cost="medium")
def analyze_double_compression(ctx: Context) -> Evidence:
    """Traces of earlier lossy coding / transcoding: spectral holes and near-zero time-frequency cells in the passband
    (quantised-to-zero coefficients), loss of spectral flatness in the upper bands, and an MDCT block-grid scan
    (576-sample MP3 granules, 1024-sample AAC frames) that finds a codec's frame alignment in decoded audio; two
    distinct grids, or a grid of a different codec than the container declares, indicate double compression."""
    ev = Evidence(expert="double_compression", category="compression", features=dict(DC_DEFAULTS))
    spec = power_spectrogram(ctx)
    cut = _cutoff_result(ctx)
    if spec is None or cut is None or spec["active"].sum() < 3:
        ev.applicable = False
        ev.findings.append(Finding("Too little non-silent audio to look for compression traces.", 0, 0.0))
        return ev
    info = _container_info(ctx)
    codec = info.get("codec", "") or ""
    native_lossy = codec in ("mp3", "mp2", "aac", "opus", "vorbis", "wmav2", "ac3", "amr_nb", "amr_wb")
    x, sr = native_audio(ctx)
    freqs, P = spec["freqs"], spec["P"][spec["active"]]
    fc = cut["cutoff_hz"]
    L = 10 * np.log10(P + 1e-20)
    f = ev.features

    # (a) spectral holes: cells > 25 dB below their frame's local spectral envelope (median over 17 bins), and cells at
    #     or below the 16-bit quantisation floor (exact zeros left by a decoder, or digital silence).
    env = median_filter(L, size=(1, 17), mode="nearest")
    holes = L < env - 25
    near_zero = L < PCM16_FLOOR_DB + 6
    pas = (freqs > 1000) & (freqs < 0.95 * fc)
    hf = (freqs > 0.6 * fc) & (freqs < 0.95 * fc)
    stop = freqs > fc + 150
    if pas.sum() >= 4:
        f["double_compression_hole_frac"] = float(holes[:, pas].mean())
        f["double_compression_nz_frac_pass"] = float(near_zero[:, pas].mean())
    if hf.sum() >= 4:
        f["double_compression_hf_hole_frac"] = float(holes[:, hf].mean())
        f["double_compression_nz_frac_hf"] = float(near_zero[:, hf].mean())
    if stop.sum() >= 2:
        f["double_compression_nz_frac_stop"] = float(near_zero[:, stop].mean())

    # (b) band-wise spectral flatness (geometric / arithmetic mean power) in 12 bands from 1 kHz to the edge; coarse
    #     quantisation turns noise-like upper bands into sparse "tonal islands" (lower flatness, abrupt band steps).
    edges = np.linspace(1000, 0.95 * fc, 13)
    flat = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        if m.sum() >= 2:
            p = P[:, m] + 1e-20
            flat.append(float(np.mean(np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1))))
    if len(flat) >= 4:
        flat_a = np.array(flat)
        f["double_compression_hf_flatness"] = float(flat_a[-4:].mean())
        f["double_compression_flatness_step"] = float(np.max(np.abs(np.diff(np.log(flat_a + 1e-9)))))

    # (c) codec block grids at the native rate.
    g576 = grid_scan(x, sr, 576, fmax=0.95 * fc)
    g1024 = grid_scan(x, sr, 1024, fmax=0.95 * fc) if (codec == "aac" or sr in AAC_RATES) else grid_scan(x, sr, 1024, 0, 0)
    strong = lambda g: bool(g["z"] >= GRID_Z_STRONG and g["zmin"] >= GRID_ZMIN)
    f.update({"double_compression_grid576_z": g576["z"], "double_compression_grid576_zmin": g576["zmin"], "double_compression_grid576_second_z": g576["second_z"],
              "double_compression_grid1024_z": g1024["z"], "double_compression_grid1024_zmin": g1024["zmin"]})
    grids = [(nm, n, g) for nm, n, g in (("MP3", 576, g576), ("AAC", 1024, g1024)) if strong(g)]
    own_n = LOSSY_NATIVE.get(codec)
    exp_off = (MP3_GRID_PHASE - _decoder_skip(ctx, sr)) % 576 if own_n == 576 else None
    near = lambda o, e: min((o - e) % 576, (e - o) % 576) <= 4
    own, foreign = None, []
    for nm, n, g in grids:
        if n == own_n and (exp_off is None or near(g["offset"], exp_off)):
            own = (nm, g)
        else:                                  # other codec's grid, or an MP3 grid at the wrong phase for this file
            foreign.append((nm, n, g))
    second_mp3 = bool(own_n == 576 and own and g576["second_z"] >= GRID_Z_STRONG + 1.0)
    f["double_compression_grid_detected"] = float(bool(grids))
    f["double_compression_double_grid"] = float(bool(native_lossy and foreign) or second_mp3)
    f["double_compression_native_lossy"] = float(native_lossy)

    # Combined 0..1 "earlier lossy coding" indicator (thresholds from the MP3 transcode experiment: originals show
    # ~0.5% near-zero / ~0.4% hole cells in the upper band, 64 kbps ~1.3% / 0.5%, 32 kbps ~9% / 1.9%).
    sparse = max(np.clip((f["double_compression_nz_frac_hf"] - 0.01) / 0.05, 0, 1), np.clip((f["double_compression_hf_hole_frac"] - 0.006) / 0.015, 0, 1))
    f["double_compression_lossy_trace"] = float(max(sparse, 1.0 if foreign else 0.0))

    # ---- findings -------------------------------------------------------------------------------------
    best = max(g576["z"], g1024["z"])
    if native_lossy:
        if own:
            ev.findings.append(Finding(
                f"Own-codec check: the {own[0]} frame grid is visible at offset {int(own[1]['offset'])} (robust z = {own[1]['z']:.1f})"
                f"{', exactly where this file' + chr(39) + 's own framing puts it' if exp_off is not None else ''}, as expected for a "
                f"file stored as {codec.upper()}.", 0, 0.0))
        if foreign or second_mp3:
            if foreign:
                nm, n, g = foreign[0]
                what = (f"an MP3 frame grid at offset {int(g['offset'])} (z = {g['z']:.1f}) although this file's own MP3 framing "
                        f"would put it at offset {exp_off}" if n == 576 and own_n == 576 else
                        f"an {nm} block grid (z = {g['z']:.1f}) that does not belong to the file's own {codec.upper()} coding")
            else:
                what = (f"two distinct MP3 frame grids (offsets {int(g576['offset'])} and {int(g576['second_offset'])}, "
                        f"z = {g576['z']:.1f} / {g576['second_z']:.1f})")
            ev.findings.append(Finding(
                f"Double compression: the decoded audio shows {what}. The content was lossy-encoded and decoded before this "
                "encode, so the file is at least a second-generation copy. That breaks chain of custody but does not by "
                "itself indicate synthesis.", 0, 0.25))
        elif sparse >= 0.5:
            ev.findings.append(Finding(
                f"Sparse upper band ({100 * f['double_compression_nz_frac_hf']:.1f}% of cells at the digital floor): consistent with the file's own "
                f"{codec.upper()} coding at {info.get('bitrate', 0) / 1000:.0f} kbps; an earlier encode cannot be separated from it.",
                0, 0.0))
        return _finish(ev, flat, g576, g1024, fc)

    if grids:
        nm, n, g = max(grids, key=lambda t: t[2]["z"])
        ev.findings.append(Finding(
            f"Codec block grid: although stored as uncompressed PCM, the audio re-analysed with a {n}-sample MDCT "
            f"shows a sharp coefficient-sparsity peak at offset {int(g['offset'])} (robust z = {g['z']:.1f}; z = {g['zmin']:.1f} "
            f"even in the weaker half of the clip). That is the fingerprint of an {nm} encoder's quantiser: the audio was lossy-compressed and "
            "decoded at least once before it was saved as PCM. Transcoding does not by itself mean the voice is synthetic, "
            "but the file is not an original recording.", 0, 0.2))
    elif sparse >= 0.5:
        ev.findings.append(Finding(
            f"Spectral holes: {100 * f['double_compression_hf_hole_frac']:.1f}% of upper-band time-frequency cells are near-silent pockets and "
            f"{100 * f['double_compression_nz_frac_hf']:.1f}% sit at the digital noise floor (clean uncompressed speech: ~0.4% and ~0.5%). "
            "Low-bitrate codecs zero weak coefficients in exactly this way, so the audio probably passed through lossy "
            "compression, or a generator that leaves the high band similarly empty.", 0, 0.15))
    else:
        ev.findings.append(Finding(
            f"No trace of earlier lossy coding: upper-band hole rate {100 * f['double_compression_hf_hole_frac']:.2f}%, near-zero cells "
            f"{100 * f['double_compression_nz_frac_hf']:.2f}%, and no MP3/AAC block grid (best z = {best:.1f}, threshold {GRID_Z_STRONG}). "
            "Caveat: resampling, high bitrates (>= 128 kbps) and added noise erase these traces.", 0, 0.0))
    return _finish(ev, flat, g576, g1024, fc)


def _finish(ev: Evidence, flat: list, g576: Dict[str, Any], g1024: Dict[str, Any], fc: float) -> Evidence:
    ev.score = None
    ev.artifacts = {"band_flatness": flat, "cutoff_hz": fc,
                    "grid576": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in g576.items()},
                    "grid1024": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in g1024.items()}}
    return ev
