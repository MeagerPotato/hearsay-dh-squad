# Recon findings: what the data told us before we trained anything

This page records what we learned by treating the challenge data itself as forensic evidence, before
building any detector. Several of these findings changed our whole strategy.

## 1. The test set is not what the training data suggests

The organizers provided DiffSSD (70,000 synthetic clips from 10 modern TTS systems) and 242 real
LJSpeech clips for training. The natural assumption is that the test set looks like them. It does not.

| Evidence | Test set | DiffSSD / LJ |
|---|---|---|
| Spoken content (Whisper transcripts of all 1,671 clips) | VCTK prompts: Glasgow Herald news sentences, the Rainbow Passage, the "Please call Stella" paragraph | ChatGPT-written sentences (DiffSSD), 19th-century non-fiction (LJ) |
| Clip length | 3.0–13.6 s, 81% between 3 and 4 s | medians 6–10 s |
| Script matches | 0 of the sampled test clips match any of DiffSSD's 5,000 scripts | — |

VCTK is the source corpus of the ASVspoof 2019 LA benchmark. We therefore moved our main training and
validation domain to the public ASVspoof 2019 LA **train/dev** partitions, which the challenge PDF lists as a
recommended dataset. DiffSSD became a secondary "modern TTS variety" source.

## 2. We reverse-engineered the organizers' preprocessing

* 94% of test clip lengths are an exact whole number of 512-sample hops at 22,050 Hz. That is the
  signature of `librosa.load()` (default 22.05 kHz) followed by `librosa.effects.trim()` (hop 512).
* The shortest clip is exactly 130 hops (3.019 s), so clips shorter than 3 s after trimming were dropped.
* Every file is 16 kHz mono PCM16, peak-normalised (median peak 0.998), and carries the identical
  `LIST/INFO/ISFT = Lavf58.29.100` header: all container metadata was rewritten by one ffmpeg run.
* File timestamps span 28 minutes, and signal statistics are uniform across that window, so processing
  order does not leak labels.

`hearsay/preprocess.py` replicates this pipeline, and every training and validation clip goes through it.
This removes clip length, loudness, sample rate and container as possible "shortcut" cues that would work
on the training data but not on the test set.

### Confirmation, and a discovery: the organizers added noise

Running the public ASVspoof 2019 LA **dev** partition through our pipeline replica reproduces the test
set's statistics closely, with one exception:

| | Median duration | Share 3–4 s | Spectral roll-off (−20 / −40 dB) | Median noise floor |
|---|---|---|---|---|
| ASVspoof 2019 LA dev, our pipeline | 3.58 s | 74–82% | 7,671 / 7,765 Hz | −61 dBFS |
| HEARSAY test set | 3.41 s | 81% | 7,609 / 7,718 Hz | **−44 dBFS** |

The test audio is about 17 dB noisier. The organizers added background noise to most clips, which matches
the NSA brief ("added noise to real/synthetic is still real/synthetic"). We therefore built
**distribution-matched augmentation** (`hearsay/augment.py`). We measure the test set's noise-floor
distribution, draw a target floor from it for every training clip, and mix in real-world MUSAN noise
until the clip's floor reaches that target. This is done identically for real and synthetic clips, so
noise can never become a label shortcut. About 5% of clips are also passed through a lossy codec (MP3,
AAC, Opus, G.726, G.722, µ-law), mirroring the ~5% of band-limited, telephony-like test clips.

## 3. What does *not* work on this test set (and why)

* **Detectors trained only on VCTK treat every unfamiliar real voice as fake.** The strongest in-domain
  detector (XLS-R + AASIST, trained on ASVspoof 2019 LA) catches 99.4% of our own macOS-TTS fakes and
  97.2% of DiffSSD's modern TTS, which is excellent generalisation to unseen generators. But it also flags
  **97.5% of genuine LJSpeech recordings** as synthetic. It learned "real = VCTK studio audio". On a
  VCTK-based test set this mostly works; on real-world audio it would bury analysts in false alarms. We
  counter it by training our own classifier with additional real voices, and we report it here because
  the NSA brief asks what did not work.

* **Container / metadata forensics.** Every test header is identical (laundered by ffmpeg), so metadata
  carries no class information here. The module still runs and matters for real-world files: it catches,
  for example, DiffSSD PlayHT files named `.wav` that are really MP3 inside.
* **Models trained only on the provided data.** DiffSSD's fakes and LJ's reals differ in speaker, text,
  sample rate, loudness, length and mains hum, so a model can separate them without learning anything
  about synthesis artefacts. We found these shortcuts by measurement and removed them with the pipeline
  replica above.
* **The provided ASVspoof 5 AASIST baseline.** It flags only 0.6% of test clips as spoof-leaning (the
  expected share is about 30%). It is out of domain, because ASVspoof 5 is built from LibriVox audiobooks,
  not VCTK.

* **Noise, not the attacks, is what breaks off-the-shelf detectors here.** On the same 3,128 ASVspoof 2019
  LA dev clips, XLS-R + AASIST scores minDCF 0.020 (EER 0.8%) on clean pipeline output but 0.140
  (EER 6.4%) once test-matched noise is added (organizers' setting, P_spoof = 0.3). The added noise alone
  makes it 7× worse.
* **Denoising before detection does not help; it hurts.** Spectral-gating noise reduction (stationary and
  non-stationary) before scoring made minDCF worse (0.13–0.17 vs 0.11 on the same clips, measured at the development setting P_spoof = 0.5). Removing noise
  also removes or alters the fine detail the detectors rely on. Training *with* matched noise is the
  fix that works (see below).
* **Classic forensic cues are weak on laundered, noisy audio.** Our interpretable model on 216 named
  features (pitch, pauses, breaths, spectral dynamics, reverberation, vocoder harmonics) reaches minDCF
  0.38 on its own, far behind the deep models. It still adds a small, independent signal in fusion,
  and it supplies the plain-English explanations. Its most useful features were spectral-dynamics
  variability (LFCC deltas), speech/pause fraction, leading silence, reverberation (SRMR), vocoder
  harmonic structure, speaking rate, pitch variability, shimmer and formant bandwidths.
* **Mains hum (ENF) and container metadata contributed nothing on the test set.** Hum is absent, and
  every header was rewritten. Both still work on real-world files and appear in the per-file reports.

## 4. What worked

* **Training on distribution-matched noise.** A plain logistic regression on frozen WavLM-Large
  embeddings, trained on noise-matched ASVspoof train audio, beats both off-the-shelf detectors on
  matched dev (minDCF 0.079 vs 0.116 / 0.140 at P_spoof = 0.3, with the final layers 3–12). It also flags only ~8% of genuine LJSpeech clips, versus 97.5%.
* **Where the evidence lives.** A per-layer probe of WavLM-Large shows spoofing evidence peaks in the
  middle layers (6–10). The bottom layers (raw acoustics) and the top layers (words and meaning) are
  both worse. Our first head used layers 6–9; the final head uses the wider band 3–12 (see below).
* **Fusion across different model families** (the two published XLS-R detectors, our re-trained AASIST
  back end and the WavLM head). Weights are fitted on a dev set that mixes seen attacks with held-out *unseen*
  generator families, so they are not tuned only to attacks the models already know.
* **Transductive self-training** (semi-supervised, no labels). The ensemble's most confident test clips
  are used as pseudo-labels to adapt the WavLM head to the test's own attacks. In a labelled
  simulation (VCTK reals + unseen DiffSSD generators) it improved fused minDCF from 0.269 to 0.229–0.251 (P_spoof = 0.5),
  though not in the full fusion (see below).

### Choosing the final ensemble (and what we dropped)

All numbers below are cross-validated (out-of-fold) minDCF on a dev mix, at the development setting
P_spoof = 0.5 (the value in the scorer shipped to us). At the organizers' confirmed P_spoof = 0.3 the submitted
fusion scores 0.119 mixed (0.076 seen, 0.153 on all unseen families pooled). Re-running the sweep at 0.3, over
model sets, real-clip weights of 1 to 4 and regularisation, found nothing better than it by more than 0.001. Real speech is ASVspoof 2019 LA
dev bona fide (VCTK), matched to the test's noise. Fakes are drawn from three families: seen ASVspoof
attacks A01–A06, held-out DiffSSD generators (unseen modern TTS) and held-out macOS voices.

| Fusion (condition-aware) | Mixed | Seen ASVspoof | Unseen DiffSSD | macOS TTS |
|---|---|---|---|---|
| XLS-R+AASIST, XLSR-SLS, WavLM head | 0.149 | 0.101 | 0.267 | 0.042 |
| … + glass-box forensic model | 0.154 | 0.108 | 0.262 | 0.042 |
| … + noise-adapted AASIST back-end (**submitted**) | **0.146** | **0.098** | 0.280 | 0.044 |

* **Full fine-tuning of XLS-R + AASIST on noise-matched data: better on noise, worse on unseen attacks.**
  On matched dev it improved minDCF from 0.140 to 0.123 (P_spoof = 0.3), and it stopped flagging unfamiliar real voices
  (LJSpeech false alarms fell from 97.5% to 5%). But it caught only 33% of unseen DiffSSD generators,
  down from 97%. In fusion it received a *negative* weight, so we dropped it. Fine-tuning the transformer on six attack types
  made it forget what it knew about other synthesis families.
* **Adapting only the AASIST back-end (XLS-R frozen) kept most of the generalisation.** It caught 64% of
  unseen DiffSSD and flagged 38% of LJSpeech. It earns a small positive weight and improves the fusion,
  most clearly under the alternative ("flipped") reading of the scorer (0.137 vs 0.153 on seen attacks).
  It is in the submitted ensemble.
* **Condition-aware fusion.** On dev, detectors fail mostly on *quiet* clips (noise floor below −55 dBFS: EER 27–31%
  vs 1–3% on noisier clips). Giving the fusion a quiet-clip indicator lets it lean on the WavLM head there
  (weight 3.3 → 4.2) and discount the XLS-R detectors. That improved mixed minDCF from 0.178 to 0.156.
* **Transductive self-training was not used.** It helped in a simulation on unseen DiffSSD generators, but
  on the full dev mix it made the fusion worse (0.179 vs 0.156): its pseudo-labels inherit the XLS-R
  detectors' mistakes on quiet clips.
* **The glass-box forensic model is not in the score.** It is useful for explanations, and it gets a small
  positive fusion weight, but it did not lower minDCF (0.154 with it vs 0.149 without). We keep it for the
  per-file reports and say so here rather than claim it helps the number.
* **A wider band of WavLM layers helps.** Using layers 3–12 instead of 6–9 lowered the head's dev minDCF
  from 0.131 to 0.111. Using all 24 layers was worse (0.140).

## 5. Signals that do line up with an in-domain detector

These were measured against an AASIST model trained on ASVspoof 2019 LA train. It is only a noisy pseudo-label, so
each signal is a lead to validate, not a proven feature.

| Signal | Clips | Share scored spoof-leaning by AASIST |
|---|---|---|
| Baseline (all clips) | 1,671 | 16% |
| Sentence spoken in 5 or more other clips ("repeated script") | 86 | ~40% |
| Interior runs of exact digital silence (≥ 5 ms) | 24 | 50% |
| Very clean noise floor (< −55 dBFS) | ~160 | strongly elevated |
| Very noisy (> −35 dBFS) | ~70 | elevated: a likely false-alarm risk on noise-augmented real speech |

## 6. The scorer's direction problem

The organizers' scoring code is the ASVspoof 5 Track-1 evaluation package with two parameters changed
(`Pspoof` 0.05→0.5 in the copy shipped to teams, and 0.3 in the organizers' actual scoring, which they
confirmed on Discord on 26 September; `Cfa` 10→4; verified by diffing against the upstream repository). That code treats a
*higher* score as *bona fide*, while the submission format asks for 1.0 = synthetic. Run unmodified on
correctly oriented scores, it gives a perfect detector minDCF = 1.0, the worst possible value. We follow
the written specification (1.0 = synthetic; C_fa = 4, so with P_spoof = 0.3 a false alarm on real audio
weighs about 1.7× a miss after normalisation) and flagged the mismatch
to the organizers. Because the metric depends only on ranking, calibration does not matter, but ties and
saturated scores can only hurt, so our writer guarantees strictly unique, non-saturated values.
