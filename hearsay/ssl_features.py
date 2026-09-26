"""Frozen self-supervised speech embeddings for a light, interpretable-by-layer spoof classifier.

We run a pretrained WavLM / XLS-R model once, pool every transformer layer's hidden states (mean and
std over time), and train small heads on top. Lower and middle layers carry most anti-spoofing
information (vocoder artefacts live below the phonetic level), which the per-layer weights of the
head let us show in the documentation.
"""
import os

import numpy as np
import soundfile as sf
import torch

from hearsay.paths import model_path

DEFAULT_MODEL = "microsoft/wavlm-large"  # Hub id, used only when there is no local copy


def default_model():
    """Local WavLM-Large folder ($HEARSAY_MODELS/wavlm-large) when present, else the Hub id."""
    local = model_path("wavlm-large")
    return str(local) if (local / "config.json").is_file() else DEFAULT_MODEL


class SSLEmbedder:
    def __init__(self, model_name=None, device=None, layers=None, max_sec=6.0, threads=4):
        from transformers import AutoFeatureExtractor, AutoModel
        model_name = model_name or default_model()
        torch.set_num_threads(threads)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.fe = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.device).eval()
        if self.device == "mps":
            self.model = self.model.half()
        self.layers = layers  # None = all hidden states (embedding output + each transformer layer)
        self.max_len = int(max_sec * 16000)

    @torch.no_grad()
    def embed_batch(self, wavs):
        """wavs: list of 1-D float32 arrays at 16 kHz -> (B, L, 2*H) mean+std pooled per layer."""
        wavs = [w[: self.max_len] for w in wavs]
        inp = self.fe(wavs, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
        x = inp["input_values"].to(self.device)
        mask = inp["attention_mask"].to(self.device)
        if self.device == "mps":
            x = x.half()
        out = self.model(x, attention_mask=mask)
        # pool in float32: summing hundreds of fp16 frames overflows fp16's 65,504 limit
        hs = torch.stack(out.hidden_states, 1).float()  # (B, L, T, H)
        if self.layers is not None:
            hs = hs[:, self.layers]
        # frame-level mask from sample-level mask
        T = hs.shape[2]
        lengths = mask.sum(1).float() / mask.shape[1] * T
        fmask = (torch.arange(T, device=hs.device)[None, :] < lengths[:, None]).to(hs.dtype)[:, None, :, None]
        n = fmask.sum(2).clamp(min=1)
        mean = (hs * fmask).sum(2) / n
        std = (((hs - mean[:, :, None]) ** 2 * fmask).sum(2) / n).clamp(min=1e-6).sqrt()
        return torch.cat([mean, std], -1).float().cpu().numpy()

    def embed_files(self, files, batch_size=8, load=None, progress_every=500):
        load = load or (lambda f: sf.read(f, dtype="float32")[0])
        feats = []
        for i in range(0, len(files), batch_size):
            feats.append(self.embed_batch([load(f) for f in files[i:i + batch_size]]))
            if progress_every and (i // batch_size) % max(1, progress_every // batch_size) == 0:
                print(f"  ssl {i}/{len(files)}", flush=True)
        return np.concatenate(feats, 0)


def save(path, names, feats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, names=np.array(names), feats=feats.astype(np.float16))


def load(path):
    d = np.load(path, allow_pickle=False)
    return list(d["names"]), d["feats"].astype(np.float32)
