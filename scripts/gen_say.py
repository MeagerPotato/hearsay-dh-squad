"""Generate our own synthetic speech with the macOS `say` TTS engine ("test yourself" in the NSA brief).

Text comes from passages that every VCTK speaker also reads (Rainbow Passage, "Please call Stella"),
plus neutral news-style sentences, so real and synthetic training audio share vocabulary and the
detector cannot learn a text shortcut. Output goes through the organizer pipeline replica.
"""
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hearsay.preprocess import organizer_pipeline, write_pcm16  # noqa: E402

RAINBOW = [
    "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.",
    "The rainbow is a division of white light into many beautiful colors.",
    "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon.",
    "There is, according to legend, a boiling pot of gold at one end.",
    "People look, but no one ever finds it.",
    "When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow.",
    "Throughout the centuries people have explained the rainbow in various ways.",
    "Some have accepted it as a miracle without physical explanation.",
    "To the Hebrews it was a token that there would be no more universal floods.",
    "The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
]
STELLA = [
    "Please call Stella.",
    "Ask her to bring these things with her from the store.",
    "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
    "We also need a small plastic snake and a big toy frog for the kids.",
    "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.",
]
NEWS = [
    "The council will meet again next week to discuss the budget.",
    "Officials said the new bridge would open early next year.",
    "Supporters gathered outside the hall before the evening vote.",
    "The company reported a sharp rise in profits over the summer.",
    "Local schools are preparing for a busy start to the term.",
    "The minister declined to comment on the latest figures.",
    "Heavy rain is expected across the region on Thursday.",
    "The team will travel to the capital for the final match.",
    "Residents have raised concerns about traffic near the station.",
    "The report was published after months of careful review.",
    "Talks between the two sides are expected to continue tomorrow.",
    "The museum will extend its opening hours during the festival.",
    "Engineers are still investigating the cause of the power cut.",
    "Nobody expected the result to be quite so close.",
    "The new rules will apply to all businesses from April.",
    "It was a difficult decision, but we believe it is the right one.",
    "The hospital has appealed for more volunteers this winter.",
    "Farmers say the dry weather has damaged this year's crop.",
    "The singer thanked her fans at the end of a long tour.",
    "Police have asked anyone with information to come forward.",
]
TEXTS = RAINBOW + STELLA + NEWS
NOVELTY = {"Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Wobble", "Good News", "Jester", "Organ", "Superstar"}


def voices():
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout.splitlines()
    vs = []
    for line in out:
        name, rest = line.split("#")[0].rsplit(None, 1)[0].strip(), line.split("#")[0].split()
        loc = [t for t in rest if t.startswith("en_")]
        if loc and name not in NOVELTY:
            vs.append(name)
    return vs


def main(out_dir=ROOT / "data/proc/say", seed=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for v in voices():
            slug = v.replace(" ", "_").replace("(", "").replace(")", "")
            for i, t in enumerate(TEXTS):
                dst = out_dir / f"say_{slug}_{i:03d}.wav"
                if not dst.exists():
                    aiff = os.path.join(tmp, "x.aiff")
                    rate = int(rng.integers(160, 220))  # vary speaking rate
                    subprocess.run(["say", "-v", v, "-r", str(rate), "-o", aiff, t], check=True)
                    y = organizer_pipeline(aiff, rng=rng)
                    if y is None:
                        continue
                    write_pcm16(dst, y)
                rows.append([dst.stem, v, i, 1])
    with open(ROOT / "data/proc/say.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt", "voice", "text_id", "is_spoof"])
        w.writerows(rows)
    print(f"wrote {len(rows)} synthetic clips from {len(set(r[1] for r in rows))} voices")


if __name__ == "__main__":
    main()
