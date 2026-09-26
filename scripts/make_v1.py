"""v1: equal-weight z-score fusion of the two in-domain XLS-R detectors (XLSR-AASIST + XLSR-SLS)."""
import json, sys, numpy as np
sys.path.insert(0, ".")
from hearsay import submission
W = json.load(open("work/scores/test/w2v2_aasist.json"))["w2v2_aasist"]
S = json.load(open("work/scores/test/xlsr_sls.json"))["xlsr_sls"]
names = sorted(W)
z = lambda v: (v - v.mean()) / v.std()
raw = (z(np.array([W[n] for n in names])) + z(np.array([S[n] for n in names]))) / 2
p = submission.to_probability(raw)
out = submission.write(dict(zip(names, p)), "data/HackGTHearsayTesting/HGT_Hearsay_score_template.csv",
                       f"submissions/v1/{submission.TEAM}_predictions.tsv")
print("wrote", out, "| frac p>0.5:", round(float((p > 0.5).mean()), 3))
