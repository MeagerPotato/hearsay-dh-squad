"""Score datasets with the fine-tuned XLS-R + AASIST checkpoint (work/models/tak_ft/best.pt) on MPS.

  python scripts/score_tak_ft.py test la_dev_m say_m diffssd_m lj_m libri_m
Writes work/scores/<set>/tak_ft.json ({"tak_ft": {name: spoof_log_odds}, "tak_ft_w1": first-window}).
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hearsay.deep import windows  # noqa: E402
from hearsay.tak_aasist import load_la_model  # noqa: E402


def files_for(name):
    if name == "test":
        fs = sorted(glob.glob(str(ROOT / "data/HackGTHearsayTesting/*.wav")))
        return [Path(f).name for f in fs], fs
    rows = list(csv.DictReader(open(ROOT / f"data/proc/{name}.csv")))
    return [r["utt"] + ".wav" for r in rows], [str(ROOT / f"data/proc/{name}/{r['utt']}.wav") for r in rows]


@torch.no_grad()
def main(names, ckpt=None, bs=16):
    import os
    ckpt = ckpt or os.environ.get("TAK_CKPT", str(ROOT / "work/models/tak_ft/best.pt"))
    key = os.environ.get("TAK_KEY", "tak_ft")
    torch.set_num_threads(2)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = load_la_model(device=device)
    net.load_state_dict(torch.load(str(ckpt), map_location=device))
    net.eval()
    for name in names:
        ids, files = files_for(name)
        buf, owners, out = [], [], {}

        def flush():
            if not buf:
                return
            o = net(torch.tensor(np.stack(buf)).to(device)).float().cpu().numpy()
            for (n, k), v in zip(owners, o[:, 0] - o[:, 1]):
                out.setdefault(n, []).append(float(v))
            buf.clear()
            owners.clear()

        for n, f in zip(ids, files):
            x, _ = sf.read(f, dtype="float32")
            for k, w in enumerate(windows(x, max_windows=3)):
                buf.append(w)
                owners.append((n, k))
                if len(buf) >= bs:
                    flush()
        flush()
        dst = ROOT / f"work/scores/{name}/{key}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        json.dump({key: {k: float(np.mean(v)) for k, v in out.items()},
                   f"{key}_w1": {k: float(v[0]) for k, v in out.items()}}, open(dst, "w"))
        v = np.array([np.mean(x) for x in out.values()])
        print(f"{key} on {name}: n={len(v)} frac>0={np.mean(v > 0):.3f}", flush=True)
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
