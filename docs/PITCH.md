# Expo talk track (about 2 minutes) + demo + likely questions

> For the team's expo table. Say it in your own words. Every number here is in README.md or FINDINGS.md.

## The pitch

1. **The problem, in the judges' terms.** A cloned voice takes seconds to make. An analyst's time is the
   scarce resource, so the challenge penalises a false alarm on real audio four times more than a miss. We
   built HEARSAY to rank clips by how likely they are to be synthetic, keep false alarms rare, and explain
   every call.

2. **We studied the test set before training anything.** Transcribing all 1,671 clips showed they are VCTK
   sentences, the corpus behind ASVspoof 2019, not the DiffSSD data provided for training. From the clip lengths we
   reverse-engineered the organizers' preprocessing and rebuilt it in code. Comparing the two showed the
   organizers had also added background noise, about 17 dB more than the source audio.

3. **That changed everything.** The best published detector scores minDCF 0.020 on clean audio but 0.140
   once that noise is added, 7 times worse. Almost all its errors are on very quiet clips. So we trained on audio whose noise matches
   the test set's distribution, and made the fusion condition-aware: on quiet clips it shifts trust to the
   detector that handles them best.

4. **Result.** Four detectors fused. Cross-validated minDCF is 0.076 on test-matched audio with seen attacks
   and 0.153 on synthesis families no component was trained on. The strongest published detector gets 0.140
   and 0.179 under the same conditions (the organizers' settings, P_spoof = 0.3).

5. **Explainable and efficient.** Sixteen forensic experts (pitch, pauses, breaths, spectral and vocoder
   artefacts, compression history, mains hum, splices, speaker consistency, metadata) sit behind an
   agentic router. It runs the cheap checks on every file and the slow ones only when the evidence is unclear, which is 11%
   of the compute of running everything, on the full test set. Every file gets an HTML report an analyst can read.

6. **Honest about what didn't work.** Denoising before detection made things worse. Full fine-tuning forgot
   unseen attacks. Self-training helped in simulation but not on the full mix. Our interpretable model didn't
   improve the score, so we use it to explain calls rather than claim it helps the number.

## Demo path (3 clicks)

1. `docs/sample_reports/dashboard_full_test_set.html`: counts, attributed manipulation types, routing
   tiers, and the 11% compute figure.
2. `docs/sample_reports/confident_synthetic_HGT8852055.html`: verdict gauge, chain of custody (SHA-256,
   timestamps, codec), evidence by category, then scroll to the spectrogram. There's a suspected seam at 4.2 s,
   and the 7.3 kHz cut-off line and pitch contour are visible.
3. `docs/sample_reports/inconclusive_escalated_HGT8649435.html`: a borderline file. Show the routing
   trace: the router escalated it to the medium and expensive tiers, and says why.

Live alternative on any file: `python -m hearsay.cli explain <file> --html out.html`

## Likely questions

- **"Why not just use the best public detector?"** The top public models (NII AntiDeepfake) were trained on
  ASVspoof *evaluation* data, which likely overlaps this test set. A good score from them would say nothing
  about generalisation, so we excluded them, and we never extracted the ASVspoof 2019 eval partition.
- **"How do you know your dev numbers transfer?"** Our dev set goes through the same preprocessing replica
  and the same noise distribution as the test. It holds out whole speakers, voices and synthesis
  generators, not just clips. The fusion weights were chosen on seen *and* unseen attack families.
- **"What's the scorer direction issue?"** The shipped ASVspoof 5 scorer treats higher scores as bona fide,
  but the spec says 1.0 = synthetic. Run unmodified, a perfect detector gets the worst minDCF. We followed the
  spec, flagged it to the organizers, and our metric code reports both readings.
- **"What's agentic about it?"** A declarative policy picks analyses per file from triage facts (codec,
  lossy or not, bandwidth, duration) and from interim evidence (uncertain score, conflicting or strong
  flags). Every decision is logged with its reason in the report's routing trace.
- **"Can we run it?"** Yes. `docker build -t hearsay-dhsquad .` from a fresh clone, then one `docker run`.
  We verified that path on the whole test set: 55 minutes on the laptop, rank correlation 0.999999 with our
  submitted scores and no verdict changed (see docs/DOCKER.md).
- **"What would you do next?"** Train with more modern commercial voices while keeping the pretrained
  representation frozen, so noise robustness doesn't cost generalisation. Also add per-condition calibration
  (telephony, codecs, reverberant rooms) and frame-level detection of spliced audio.
