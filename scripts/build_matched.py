"""Build test-matched copies of processed datasets (noise floor + light laundering), class-agnostically.

  python scripts/build_matched.py la_dev   --bona all --spoof-per-attack 350 --out la_dev_m
  python scripts/build_matched.py la_train --bona all --spoof-per-attack 1000 --out la_train_m
  python scripts/build_matched.py say      --out say_m

Every output clip keeps its source label; the augmentation choice is independent of the label.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hearsay.augment import NoiseBank, codec_roundtrip, match_noise_floor, noise_floor_db, test_noise_floor_distribution  # noqa: E402


def select(rows, bona, spoof_per_attack, rng):
    b = [r for r in rows if r["is_spoof"] == "0"]
    s = [r for r in rows if r["is_spoof"] == "1"]
    if bona != "all":
        b = [b[i] for i in rng.choice(len(b), min(int(bona), len(b)), replace=False)]
    if spoof_per_attack and "attack" in (rows[0] if rows else {}):
        keep = []
        for att in sorted({r["attack"] for r in s}):
            grp = [r for r in s if r["attack"] == att]
            keep += [grp[i] for i in rng.choice(len(grp), min(spoof_per_attack, len(grp)), replace=False)]
        s = keep
    return b + s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bona", default="all")
    ap.add_argument("--spoof-per-attack", type=int, default=0)
    ap.add_argument("--codec-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = list(csv.DictReader(open(ROOT / f"data/proc/{a.src}.csv")))
    rows = select(rows, a.bona, a.spoof_per_attack, rng)
    targets = test_noise_floor_distribution()
    bank = NoiseBank(seed=a.seed)
    out_dir = ROOT / f"data/proc/{a.out}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) + ["aug", "nf_before", "nf_after"]
    with open(ROOT / f"data/proc/{a.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            x, _ = sf.read(ROOT / f"data/proc/{a.src}/{r['utt']}.wav", dtype="float32")
            nf0 = noise_floor_db(x)
            aug = "noise"
            y = match_noise_floor(x, float(rng.choice(targets)), bank, rng)
            u = rng.random()
            if u < a.codec_frac:
                aug = str(rng.choice(["mp3", "aac", "opus", "g726", "g722", "mulaw8k"]))
                y = codec_roundtrip(y, aug, rng)
            sf.write(out_dir / f"{r['utt']}.wav", y, 16000, subtype="PCM_16")
            w.writerow({**r, "aug": aug, "nf_before": f"{nf0:.1f}", "nf_after": f"{noise_floor_db(y):.1f}"})
            if i % 1000 == 0:
                print(f"{a.out}: {i}/{len(rows)}", flush=True)
    print(f"wrote {len(rows)} clips to {out_dir}")


if __name__ == "__main__":
    main()
