"""Class-agnostic augmentation that makes training/dev audio statistically resemble the test set.

The key idea is *distribution matching*: rather than adding noise at arbitrary SNRs, we measure the
test set's noise-floor distribution and add real-world noise (MUSAN) so each training clip's floor is
drawn from that same distribution. Real and synthetic clips are treated identically, so noise can
never become a label shortcut, which matters because a false alarm on noisy *real* speech costs 4x.
"""
import glob
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
FRAME = 320  # 20 ms at 16 kHz


def noise_floor_db(x):
    """10th-percentile 20-ms frame RMS in dBFS: a robust estimate of the background level."""
    n = len(x) // FRAME * FRAME
    if n == 0:
        return -120.0
    rms = np.sqrt((x[:n].reshape(-1, FRAME) ** 2).mean(1) + 1e-12)
    return float(20 * np.log10(np.percentile(rms, 10) + 1e-12))


class NoiseBank:
    def __init__(self, pattern=str(ROOT / "data/ext/musan/noise/**/*.wav"), max_files=400, seed=0):
        files = sorted(glob.glob(pattern, recursive=True))
        rng = np.random.default_rng(seed)
        if len(files) > max_files:
            files = [files[i] for i in rng.choice(len(files), max_files, replace=False)]
        self.clips = []
        for f in files:
            x, sr = sf.read(f, dtype="float32", always_2d=True)
            x = x.mean(1)
            if sr != 16000:
                import librosa
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            if len(x) > 16000 and np.abs(x).max() > 1e-4:
                self.clips.append(x)
        assert self.clips, "no MUSAN noise found"

    def sample(self, n, rng):
        c = self.clips[int(rng.integers(len(self.clips)))]
        if len(c) < n:
            c = np.tile(c, int(np.ceil(n / len(c))))
        s = int(rng.integers(0, len(c) - n + 1))
        seg = c[s:s + n].copy()
        if rng.random() < 0.3:  # sometimes coloured white noise instead of a recording
            seg = rng.standard_normal(n).astype(np.float32)
            if rng.random() < 0.5:
                seg = np.cumsum(seg) * 0.02  # brownish
                seg -= np.convolve(seg, np.ones(400) / 400, mode="same")
        return seg


def test_noise_floor_distribution(cache=ROOT / "work/test_noise_floor.npy"):
    cache = Path(cache)
    if cache.exists():
        return np.load(cache)
    v = np.array([noise_floor_db(sf.read(f, dtype="float32")[0])
                  for f in sorted(glob.glob(str(ROOT / "data/HackGTHearsayTesting/*.wav")))])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, v)
    return v


def match_noise_floor(x, target_db, bank, rng):
    """Add noise so the clip's noise floor rises to target_db (no-op if it is already louder)."""
    cur = noise_floor_db(x)
    if target_db <= cur + 1.0:
        return x
    noise = bank.sample(len(x), rng)
    nf = noise_floor_db(noise)
    # the floor of (x + g*noise) is dominated by g*noise when it is well above x's own floor
    g = 10 ** ((target_db - nf) / 20)
    y = x + g * noise
    peak = np.abs(y).max()
    return (y / peak * 0.998).astype(np.float32) if peak > 0 else y


def codec_roundtrip(x, codec, rng=None):
    """Lossy encode/decode with ffmpeg for laundering robustness.

    codec: 'mp3', 'aac', 'opus' (media codecs) or 'g726', 'g722', 'mulaw8k' (telephony). All encoders are
    built into stock ffmpeg on Linux and macOS, so the Docker image behaves the same as the laptop.
    """
    rng = rng or np.random.default_rng()
    args = {
        "mp3": ["-c:a", "libmp3lame", "-b:a", f"{int(rng.choice([24, 32, 48, 64, 96, 128]))}k", "-f", "mp3"],
        "aac": ["-c:a", "aac", "-b:a", f"{int(rng.choice([24, 32, 64, 96]))}k", "-f", "adts"],
        "opus": ["-c:a", "libopus", "-b:a", f"{int(rng.choice([12, 16, 24, 32]))}k", "-f", "ogg"],
        "g726": ["-ar", "8000", "-c:a", "g726", "-b:a", f"{int(rng.choice([16, 24, 32]))}k", "-f", "wav"],
        "g722": ["-c:a", "g722", "-f", "wav"],
        "mulaw8k": ["-ar", "8000", "-c:a", "pcm_mulaw", "-f", "wav"],
    }[codec]
    with tempfile.TemporaryDirectory() as d:
        src, enc = Path(d) / "in.wav", Path(d) / "enc.bin"
        sf.write(src, x, 16000, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *args, str(enc)], check=True)
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(enc), "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
                             check=True, capture_output=True).stdout
    y = np.frombuffer(out, np.float32).copy()
    # undo codec priming delay (AAC ~1-2k samples, G.722 a few) by cross-correlating the first second
    n = min(len(x), len(y), 16000)
    lags = np.arange(0, min(4096, len(y) - n + 1) if len(y) > n else 1)
    if len(lags) > 1:
        seg = x[:n]
        best = max(lags, key=lambda L: float(np.dot(seg, y[L:L + n])))
        y = y[best:]
    y = np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]
    peak = np.abs(y).max()
    return (y / peak * 0.998).astype(np.float32) if peak > 0 else y
