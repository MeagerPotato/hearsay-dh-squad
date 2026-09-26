"""Replicate the organizers' test-set preprocessing so training/dev audio looks like the test audio.

Reverse-engineered from the 1671 test files (see docs/FINDINGS.md):
  * 94% of test lengths are exact multiples of 512 samples at 22,050 Hz, i.e. librosa.load() at its
    default 22.05 kHz followed by librosa.effects.trim() (frame 2048, hop 512, top_db=60).
  * Every clip is >= 3.0 s after trimming (minimum is 130 hops = 3.019 s).
  * Output is 16 kHz mono PCM16, peak-normalised to ~0 dBFS (median peak 0.998), written by ffmpeg.
"""
import numpy as np
import librosa
import soundfile as sf

LOAD_SR = 22050
OUT_SR = 16000
HOP = 512
MIN_SEC = 3.0

# Empirical test-set duration distribution (seconds), used when cropping long sources.
TEST_DUR_QUANTILES = np.array([3.02, 3.10, 3.18, 3.27, 3.35, 3.41, 3.50, 3.62, 3.80, 4.14, 13.58])


def sample_test_like_duration(rng):
    """Draw a duration from the test set's (piecewise-linear) empirical distribution."""
    u = rng.random() * (len(TEST_DUR_QUANTILES) - 1)
    i = int(u)
    return float(TEST_DUR_QUANTILES[i] + (u - i) * (TEST_DUR_QUANTILES[i + 1] - TEST_DUR_QUANTILES[i]))


def organizer_pipeline(path_or_audio, sr=None, rng=None, crop_long=True, top_db=60):
    """Return 16 kHz float32 audio processed like the test set, or None if shorter than 3 s.

    crop_long: sources much longer than the test clips (e.g. DiffSSD, 6-10 s) are cropped to a
    test-like duration on the 512-hop grid, so clip length cannot act as a label shortcut.
    """
    rng = rng or np.random.default_rng()
    if isinstance(path_or_audio, (str, bytes)) or hasattr(path_or_audio, "__fspath__"):
        y, _ = librosa.load(path_or_audio, sr=LOAD_SR, mono=True)
    else:
        y = librosa.resample(np.asarray(path_or_audio, np.float32), orig_sr=sr, target_sr=LOAD_SR) if sr != LOAD_SR \
            else np.asarray(path_or_audio, np.float32)
    y, _ = librosa.effects.trim(y, top_db=top_db, frame_length=2048, hop_length=HOP)
    if len(y) < MIN_SEC * LOAD_SR:
        return None
    if crop_long and len(y) > 4.5 * LOAD_SR:
        n = int(round(sample_test_like_duration(rng) * LOAD_SR / HOP)) * HOP
        if n < len(y):
            start = int(rng.integers(0, (len(y) - n) // HOP + 1)) * HOP
            y = y[start:start + n]
    y = librosa.resample(y, orig_sr=LOAD_SR, target_sr=OUT_SR)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 0.998
    return y.astype(np.float32)


def write_pcm16(path, y):
    sf.write(path, np.clip(y, -1, 1), OUT_SR, subtype="PCM_16")
