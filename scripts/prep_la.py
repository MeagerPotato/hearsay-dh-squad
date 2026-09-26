"""Extract ASVspoof 2019 LA train/dev ONLY and pass every clip through the organizers' pipeline.

The eval partition and its keys are deliberately never extracted: the HEARSAY test set appears to be
drawn from the same VCTK/ASVspoof-2019 family, so eval audio or labels would leak test answers.

Output: data/proc/la_{train,dev}/<utt>.wav (16 kHz PCM16, trimmed, >=3 s, peak-normalised)
        data/proc/la_{train,dev}.csv  (utt, speaker, attack, label, is_spoof, dur)
"""
import csv
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hearsay.preprocess import organizer_pipeline, write_pcm16  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data/ext/LA.zip"
RAW = ROOT / "data/ext"
PROC = ROOT / "data/proc"
SPLITS = {"train": "ASVspoof2019.LA.cm.train.trn.txt", "dev": "ASVspoof2019.LA.cm.dev.trl.txt"}


def extract():
    with zipfile.ZipFile(ZIP) as z:
        members = [m for m in z.namelist()
                   if ("ASVspoof2019_LA_train/" in m or "ASVspoof2019_LA_dev/" in m
                       or any(m.endswith(p) for p in SPLITS.values()))]
        assert not any("_eval" in m or ".eval." in m for m in members)
        todo = [m for m in members if not (RAW / m).exists()]
        print(f"extracting {len(todo)} of {len(members)} train/dev members (eval skipped)")
        for m in todo:
            z.extract(m, RAW)


def _one(args):
    src, dst, seed = args
    if os.path.exists(dst):
        return dst, None
    y = organizer_pipeline(src, rng=np.random.default_rng(seed))
    if y is None:
        return dst, "short"
    write_pcm16(dst, y)
    return dst, len(y) / 16000


def process(split, workers=3):
    proto = next(RAW.rglob(SPLITS[split]))
    flac_dir = next(RAW.rglob(f"ASVspoof2019_LA_{split}/flac"))
    out_dir = PROC / f"la_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [line.split() for line in open(proto)]  # speaker utt - attack label
    jobs = [(str(flac_dir / f"{r[1]}.flac"), str(out_dir / f"{r[1]}.wav"), i) for i, r in enumerate(rows)]
    durs = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (dst, d) in enumerate(ex.map(_one, jobs, chunksize=64)):
            durs[Path(dst).stem] = d
            if i % 5000 == 0:
                print(split, i, "/", len(jobs), flush=True)
    with open(PROC / f"la_{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt", "speaker", "attack", "label", "is_spoof", "dur"])
        kept = 0
        for spk, utt, _, attack, label in rows:
            d = durs.get(utt)
            if d is None or d == "short":
                if os.path.exists(out_dir / f"{utt}.wav"):
                    import soundfile as sf
                    d = sf.info(out_dir / f"{utt}.wav").duration
                else:
                    continue
            w.writerow([utt, spk, attack, label, int(label == "spoof"), f"{d:.3f}"])
            kept += 1
    print(f"{split}: kept {kept} of {len(rows)} (>= 3 s after trim)")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    extract()
    for s in ("train", "dev"):
        process(s)
