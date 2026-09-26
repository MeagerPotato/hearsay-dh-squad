"""v0 safety-net submission: z-score average of the two ASVspoof2019-LA-train AASIST models."""
import json, sys, numpy as np
sys.path.insert(0, ".")
from hearsay import submission
S = json.load(open("work/aasist_scores.json"))
names = sorted(S["aasist19"])
z = lambda v: (v - v.mean()) / v.std()
raw = (z(np.array([S["aasist19"][n] for n in names])) + z(np.array([S["aasist19L"][n] for n in names]))) / 2
p = submission.to_probability(raw)
out = submission.write(dict(zip(names, p)), "data/HackGTHearsayTesting/HGT_Hearsay_score_template.csv",
                       f"submissions/v0/{submission.TEAM}_predictions.tsv")
print("wrote", out, "| frac p>0.5:", round(float((p > 0.5).mean()), 3), "| min/max", p.min(), p.max())
