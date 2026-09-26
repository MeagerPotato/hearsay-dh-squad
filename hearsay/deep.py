"""Deep-learning anti-spoofing detectors behind one interface.

Every detector returns a *spoof log-odds* per clip (higher = more synthetic), regardless of each
model's native output convention, so fusion never has to know about polarity.

Models (all trained on ASVspoof 2019 LA *train* only unless noted in docs/MODELS.md):
  * aasist19 / aasist19L : clovaai AASIST / AASIST-L (raw-waveform graph attention)
  * w2v2_aasist          : Tak et al. 2022, XLS-R 300M + AASIST (ONNX export)
  * xlsr_sls             : Zhang et al. 2024, XLS-R 300M + sensitive-layer-selection head (ONNX)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from hearsay.paths import models_dir

ROOT = Path(__file__).resolve().parents[1]
WIN = 64600  # ~4.04 s at 16 kHz, the fixed window every ASVspoof-2019 model was evaluated with


def tile_pad(x, n=WIN):
    """ASVspoof-style padding: repeat short clips, truncate long ones."""
    if len(x) >= n:
        return x[:n]
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


def windows(x, n=WIN, max_windows=3):
    """First window (the canonical protocol) plus evenly spaced extra windows for long clips."""
    if len(x) <= n or max_windows <= 1:
        return [tile_pad(x, n)]
    starts = np.linspace(0, len(x) - n, num=min(max_windows, 1 + (len(x) - n) // (n // 2))).astype(int)
    return [x[s:s + n] for s in starts]


class OnnxDetector:
    """XLS-R based detectors exported to ONNX (fixed 64600-sample input, logits[:, 1] = bona fide)."""

    def __init__(self, name, path, threads=4, bona_index=1):
        import onnxruntime as ort
        self.name, self.bona_index = name, bona_index
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        if os.environ.get("HEARSAY_COREML", "0") == "1":  # CoreML fails on the XLS-R graphs; CPU is ~0.2 s/clip
            providers.insert(0, ("CoreMLExecutionProvider", {"MLComputeUnits": "ALL"}))
        try:
            self.sess = ort.InferenceSession(str(path), so, providers=providers)
        except Exception:
            self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, batch):
        logits = self.sess.run(None, {self.inp: np.asarray(batch, np.float32)})[0]
        return logits[:, 1 - self.bona_index] - logits[:, self.bona_index]


class AasistDetector:
    """clovaai AASIST checkpoints (logits index 1 = bona fide)."""

    def __init__(self, name, conf, weights, device="cpu", threads=4):
        import torch
        torch.set_num_threads(threads)
        sys.path.insert(0, str(ROOT / "third_party/aasist"))
        from models.AASIST import Model
        self.torch, self.name, self.device = torch, name, device
        self.net = Model(json.load(open(conf))["model_config"]).to(device).eval()
        self.net.load_state_dict(torch.load(weights, map_location=device))

    def __call__(self, batch):
        with self.torch.no_grad():
            _, out = self.net(self.torch.tensor(np.asarray(batch, np.float32)).to(self.device))
        out = out.cpu().numpy()
        return out[:, 0] - out[:, 1]


def build(name, **kw):
    m = models_dir()  # $HEARSAY_MODELS (Docker: /app/models), default <repo>/work/models
    if name == "w2v2_aasist":
        return OnnxDetector(name, m / "w2v2-aasist/w2v2-aasist.onnx", **kw)
    if name == "xlsr_sls":
        return OnnxDetector(name, m / "xlsr-sls/xlsr-sls.onnx", **kw)
    if name == "aasist19":
        return AasistDetector(name, ROOT / "third_party/aasist/config/AASIST.conf",
                              ROOT / "third_party/aasist/models/weights/AASIST.pth", **kw)
    if name == "aasist19L":
        return AasistDetector(name, ROOT / "third_party/aasist/config/AASIST-L.conf",
                              ROOT / "third_party/aasist/models/weights/AASIST-L.pth", **kw)
    raise KeyError(name)


def score_files(detector, files, batch_size=8, max_windows=1, load=None, progress=True, return_first=False):
    """Return {basename: spoof_log_odds}. Long clips can be scored on several windows and averaged.

    With return_first=True, also return the canonical first-window score (the ASVspoof protocol).
    """
    import soundfile as sf
    load = load or (lambda f: sf.read(f, dtype="float32")[0])
    out, buf, owners = {}, [], []

    def flush():
        if not buf:
            return
        s = detector(np.stack(buf))
        for (name, k), v in zip(owners, s):
            out.setdefault(name, []).append(float(v))
        buf.clear()
        owners.clear()

    for i, f in enumerate(files):
        x = load(f)
        for k, w in enumerate(windows(x, max_windows=max_windows)):
            buf.append(w)
            owners.append((os.path.basename(f), k))
            if len(buf) >= batch_size:
                flush()
        if progress and i % 200 == 0:
            print(f"  {detector.name}: {i}/{len(files)}", flush=True)
    flush()
    mean = {k: float(np.mean(v)) for k, v in out.items()}
    if return_first:
        return mean, {k: float(v[0]) for k, v in out.items()}
    return mean
