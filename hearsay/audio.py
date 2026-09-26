"""Format-agnostic audio loading: any container ffmpeg can read -> mono float32 at 16 kHz."""
import json
import shutil
import subprocess

import numpy as np
import soundfile as sf

ANALYSIS_SR = 16000
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def probe(path):
    """ffprobe JSON (format + streams). Returns {} if ffprobe is unavailable or fails."""
    try:
        out = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                             capture_output=True, text=True, timeout=30)
        return json.loads(out.stdout or "{}")
    except Exception:
        return {}


def load(path, sr=ANALYSIS_SR):
    """Decode with soundfile when possible (bit-exact for PCM WAV), otherwise via ffmpeg."""
    try:
        x, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if native_sr != sr:
            import librosa
            x = librosa.resample(x, orig_sr=native_sr, target_sr=sr)
        return x.astype(np.float32), native_sr
    except Exception:
        out = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
                             capture_output=True, timeout=120)
        x = np.frombuffer(out.stdout, dtype=np.float32).copy()
        info = probe(path)
        native_sr = next((int(s.get("sample_rate", sr)) for s in info.get("streams", []) if s.get("codec_type") == "audio"), sr)
        return x, native_sr
