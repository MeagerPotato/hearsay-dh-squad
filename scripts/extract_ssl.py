"""Extract frozen WavLM-Large layer embeddings for a dataset (runs on the Mac GPU via MPS, fp16).

  python scripts/extract_ssl.py test
  python scripts/extract_ssl.py la_train_m la_dev_m say_m diffssd_m lj_m
Output: work/ssl/<dataset>.npz with names + (N, 25 layers, 2048) float16 features.
"""
import csv
import glob
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hearsay.ssl_features import SSLEmbedder, save  # noqa: E402


def files_for(name):
    if name == "test":
        fs = sorted(glob.glob(str(ROOT / "data/HackGTHearsayTesting/*.wav")))
        return [Path(f).name for f in fs], fs
    rows = list(csv.DictReader(open(ROOT / f"data/proc/{name}.csv")))
    return [r["utt"] for r in rows], [str(ROOT / f"data/proc/{name}/{r['utt']}.wav") for r in rows]


def main(names):
    emb = SSLEmbedder(str(ROOT / "work/models/wavlm-large"), threads=2)
    for name in names:
        out = ROOT / f"work/ssl/{name}.npz"
        if out.exists():
            print("skip", name)
            continue
        ids, files = files_for(name)
        t = time.time()
        feats = emb.embed_files(files, batch_size=8)
        save(str(out), ids, feats)
        print(f"{name}: {feats.shape} in {time.time() - t:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
