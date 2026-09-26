"""Fine-tune XLS-R + AASIST (Tak et al.) on test-matched audio, on the Mac GPU (MPS).

Why: on clean ASVspoof 2019 LA this model has 0.22% EER, but once audio goes through the organizers'
pipeline (trim + added background noise) it degrades to ~6% EER even on attacks it was trained on.
We adapt it with *fresh* distribution-matched noise every epoch, so it learns noise robustness rather
than memorising one noisy copy.

Frozen: CNN feature encoder + bottom transformer layers (memory-safe on a 16 GB laptop).
Trained: top K transformer layers + the AASIST graph back-end.
Model selection: minDCF (false alarms cost 4x) on the held-out test-matched dev set.
"""
import argparse
import csv
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hearsay.augment import NoiseBank, codec_roundtrip, match_noise_floor, test_noise_floor_distribution  # noqa: E402
from hearsay.metrics import evaluate  # noqa: E402
from hearsay.tak_aasist import load_la_model  # noqa: E402

WIN = 64600


def rows_of(name, half=None, group_key=None):
    rows = list(csv.DictReader(open(ROOT / f"data/proc/{name}.csv")))
    if half:
        def grp(r):
            return r.get(group_key) or r.get("voice") or r.get("speaker") or r.get("attack") or ""
        rows = [r for r in rows if (zlib.crc32(grp(r).encode()) % 2 == 0) == (half == "a")]
    return [(str(ROOT / f"data/proc/{name}/{r['utt']}.wav"), int(r["is_spoof"])) for r in rows]


def crop(x, rng):
    if len(x) >= WIN:
        s = int(rng.integers(0, len(x) - WIN + 1))
        return x[s:s + WIN]
    return np.tile(x, int(np.ceil(WIN / len(x))))[:WIN]


class Sampler:
    """Draw training clips from several pools with fixed mixing weights, augmenting on the fly."""

    def __init__(self, pools, weights, seed=0, codec_frac=0.05, noise_files=120):
        self.pools, self.w = pools, np.array(weights) / np.sum(weights)
        self.rng = np.random.default_rng(seed)
        self.targets = test_noise_floor_distribution()
        self.bank = NoiseBank(seed=seed, max_files=noise_files)  # small bank: each file is held in RAM
        self.codec_frac = codec_frac

    def batch(self, n):
        xs, ys = [], []
        for _ in range(n):
            pool = self.pools[int(self.rng.choice(len(self.pools), p=self.w))]
            path, y = pool[int(self.rng.integers(len(pool)))]
            x, _ = sf.read(path, dtype="float32")
            x = match_noise_floor(x, float(self.rng.choice(self.targets)), self.bank, self.rng)
            if self.rng.random() < self.codec_frac:
                x = codec_roundtrip(x, str(self.rng.choice(["mp3", "aac", "opus", "g726", "g722", "mulaw8k"])), self.rng)
            xs.append(crop(x, self.rng))
            ys.append(y)
        return torch.tensor(np.stack(xs)), torch.tensor(ys)


@torch.no_grad()
def score(net, files, device, bs=4):
    net.eval()
    if device == "mps":
        torch.mps.empty_cache()
    out = []
    for i in range(0, len(files), bs):
        xs = []
        for f in files[i:i + bs]:
            x, _ = sf.read(f, dtype="float32")
            xs.append(x[:WIN] if len(x) >= WIN else np.tile(x, int(np.ceil(WIN / len(x))))[:WIN])
        o = net(torch.tensor(np.stack(xs)).to(device)).float().cpu().numpy()
        out.extend((o[:, 0] - o[:, 1]).tolist())  # index 1 = bona fide -> spoof log-odds
    if device == "mps":
        torch.mps.empty_cache()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--train-top", type=int, default=8)
    ap.add_argument("--lr-ssl", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--dev-limit", type=int, default=1200)
    ap.add_argument("--out", default=str(ROOT / "work/models/tak_ft"))
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--init", default=None, help="resume from a fine-tuned state_dict (e.g. work/models/tak_ft/best.pt)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cw-spoof", type=float, default=0.1, help="class weight for spoof (bona fide gets 1 - this)")
    a = ap.parse_args()
    torch.set_num_threads(3)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = load_la_model(device=device)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=device))
        print("initialised from", a.init)

    # 0) verify the port reproduces the ONNX export before touching anything
    files = sorted(Path(ROOT / "data/HackGTHearsayTesting").glob("*.wav"))[:8]
    ref = json.load(open(ROOT / "work/scores/test/w2v2_aasist.json"))["w2v2_aasist_w1"]
    got = score(net, [str(f) for f in files], device)
    want = np.array([ref[f.name] for f in files])
    print("port check (torch vs onnx):", np.round(got, 2), np.round(want, 2), "max diff", float(np.abs(got - want).max()))
    if a.verify_only or a.init:
        pass
    if a.verify_only:
        return

    # 1) freeze all but the top transformer layers and the back-end
    for p in net.ssl_model.parameters():
        p.requires_grad = False
    layers = net.ssl_model.model.encoder.layers
    for layer in layers[len(layers) - a.train_top:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in net.ssl_model.model.encoder.layer_norm.parameters():
        p.requires_grad = True
    if a.train_top > 0:
        net.ssl_model.model.gradient_checkpointing_enable()  # recompute activations in backward: less memory
    ssl_params = [p for p in net.ssl_model.parameters() if p.requires_grad]
    head_params = [p for n, p in net.named_parameters() if not n.startswith("ssl_model.")]
    opt = torch.optim.AdamW([{"params": ssl_params, "lr": a.lr_ssl}, {"params": head_params, "lr": a.lr_head}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_ssl, a.lr_head], total_steps=a.steps, pct_start=0.1)
    print(f"trainable: ssl {sum(p.numel() for p in ssl_params) / 1e6:.1f}M, head {sum(p.numel() for p in head_params) / 1e6:.2f}M")

    # 2) data pools (clean processed audio; noise/codec added on the fly)
    la = rows_of("la_train")
    pools = [
        [r for r in la if r[1] == 0],                       # VCTK real
        rows_of("libri", "a", "speaker"),                   # other real voices (half of speakers)
        [r for r in la if r[1] == 1],                       # ASVspoof 2019 LA attacks A01-A06
        rows_of("diffssd", "a", "attack") + rows_of("say", "a", "voice"),  # modern + macOS TTS (held-in halves)
    ]
    weights = [0.40, 0.10, 0.38, 0.12]
    print("pool sizes:", [len(p) for p in pools])
    samp = Sampler(pools, weights, seed=a.seed)
    dev = list(csv.DictReader(open(ROOT / "data/proc/la_dev_m.csv")))
    rng = np.random.default_rng(1)
    dev = [dev[i] for i in sorted(rng.choice(len(dev), min(a.dev_limit, len(dev)), replace=False))]
    dev_files = [str(ROOT / f"data/proc/la_dev_m/{r['utt']}.wav") for r in dev]
    dev_y = np.array([int(r["is_spoof"]) for r in dev], bool)
    base = evaluate(score(net, dev_files, device), dev_y)
    print("before fine-tuning, dev:", base, flush=True)
    best = base["minDCF_intended"]
    os.makedirs(a.out, exist_ok=True)
    # class weights: index 0 = spoof, 1 = bona fide (Tak's convention); penalise missing real speech more
    cw = torch.tensor([a.cw_spoof, 1.0 - a.cw_spoof], device=device)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        net.train()
        x, y = samp.batch(a.bs)
        target = (1 - y).to(device)  # label 1 = bona fide
        loss = F.cross_entropy(net(x.to(device)), target, weight=cw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ssl_params + head_params, 5.0)
        opt.step()
        sched.step()
        if step % 25 == 0:
            print(f"step {step} loss {loss.item():.4f} ({(time.time() - t0) / step:.2f}s/step)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(score(net, dev_files, device), dev_y)
            print(f"== step {step} dev: {r}", flush=True)
            if r["minDCF_intended"] < best:
                best = r["minDCF_intended"]
                torch.save(net.state_dict(), f"{a.out}/best.pt")
                json.dump({"step": step, **r}, open(f"{a.out}/best.json", "w"))
                print("   saved new best", flush=True)
    print("done; best dev minDCF", best)


if __name__ == "__main__":
    main()
