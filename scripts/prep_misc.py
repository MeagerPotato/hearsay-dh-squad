"""Process out-of-family validation audio through the organizer pipeline.

  diffssd : N clips per DiffSSD generator (unseen modern TTS; cropped to test-like durations)
  lj      : the 242 provided LJSpeech real clips (an unfamiliar *real* voice: guards against false alarms)
"""
import csv
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hearsay.preprocess import organizer_pipeline, write_pcm16  # noqa: E402


def run(name, items, seed=0):
    rng = np.random.default_rng(seed)
    out = ROOT / f"data/proc/{name}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for utt, src, is_spoof, attack in items:
        # speaker id for grouped CV: LibriSpeech "<spk>-<chapter>-<utt>", otherwise the attack/generator
        spk = utt.split("_", 1)[1].split("-")[0] if utt.startswith("libri_") else attack
        dst = out / f"{utt}.wav"
        if not dst.exists():
            y = organizer_pipeline(src, rng=rng)
            if y is None:
                continue
            write_pcm16(dst, y)
        rows.append([utt, attack, is_spoof, spk])
    with open(ROOT / f"data/proc/{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt", "attack", "is_spoof", "speaker"])
        w.writerows(rows)
    print(name, len(rows))


def libri(seed=0):
    """LibriSpeech dev-clean: non-VCTK real voices (audiobooks), so 'unfamiliar voice' != 'synthetic'."""
    files = sorted(glob.glob(str(ROOT / "data/ext/LibriSpeech/dev-clean/*/*/*.flac")))
    run("libri", [(f"libri_{Path(f).stem}", f, 0, "bonafide") for f in files], seed)


def main(per_gen=100, seed=0):
    if len(sys.argv) > 1 and sys.argv[1] == "libri":
        return libri(seed)
    rng = np.random.default_rng(seed)
    items = []
    for gdir in sorted(glob.glob(str(ROOT / "data/DiffSSD/generated_speech/*"))):
        g = Path(gdir).name
        files = sorted(f for f in glob.glob(f"{gdir}/**/*", recursive=True) if f.endswith((".wav", ".mp3")))
        for i in rng.choice(len(files), min(per_gen, len(files)), replace=False):
            items.append((f"dssd_{g}_{i}", files[i], 1, g))
    run("diffssd", items, seed)
    lj = sorted(glob.glob(str(ROOT / "data/resampled/*.wav")))
    run("lj", [(f"lj_{Path(f).stem}", f, 0, "bonafide") for f in lj], seed)


if __name__ == "__main__":
    main()
