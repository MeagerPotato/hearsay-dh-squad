# HEARSAY architecture, component by component

This is the detailed companion to the [README](../README.md). The README says what each part is for; this page
says exactly how each part works, which file implements it, and which numbers we measured. Measurements about
the data itself are in [FINDINGS.md](FINDINGS.md).

Contents

1. [One run, two outputs](#1-one-run-two-outputs)
2. [Reading audio](#2-reading-audio)
3. [Triage](#3-triage)
4. [Deep detectors](#4-deep-detectors)
5. [Forensic experts](#5-forensic-experts)
6. [The glass-box model](#6-the-glass-box-model)
7. [Fusion and calibration](#7-fusion-and-calibration)
8. [The agentic router](#8-the-agentic-router)
9. [Reports](#9-reports)
10. [Metric and submission writer](#10-metric-and-submission-writer)
11. [Test-matched training and validation data](#11-test-matched-training-and-validation-data)
12. [Compute, memory and caching](#12-compute-memory-and-caching)
13. [Configuration and model files](#13-configuration-and-model-files)

---

## 1. One run, two outputs

`python -m hearsay.cli predict` ([`hearsay/pipeline.py`](../hearsay/pipeline.py)) produces the submission and,
optionally, the reports, in one run:

1. **Scores.** Every detector named in `fusion_meta.json` scores every file. The forensic feature row for the
   glass-box model is computed for every file with the same 13 experts that produced its training data
   (every registered expert except `transcript`, `repeated_script` and `speaker_consistency`).
2. **Fusion.** The detector scores are fused and calibrated with the stored dev-set statistics and weights,
   exactly as `scripts/make_submission.py` does it, and written to the TSV and to `<out>_scores.json`.
3. **Reports (optional, `--reports DIR`).** The agentic orchestrator runs on every file. The detector ensemble
   is handed to it as one "deep" evidence item, and a fusion hook returns the submitted probability, so each
   report's verdict is exactly the value in the TSV. The router then decides which further analyses each file
   gets for its explanation.

Because the score only uses evidence that is computed for every file, routing changes the explanation and the
compute, never the submitted score.

## 2. Reading audio

- [`hearsay/audio.py`](../hearsay/audio.py) decodes with soundfile when it can (bit-exact for PCM WAV) and with
  ffmpeg otherwise, then mixes to mono and resamples to 16 kHz. Any container ffmpeg reads is accepted.
- The predictor peak-normalises a clip to 0.998 only when its peak is below 0.5 or above 1.0 (0.998 is the
  level of the organizers' clips). Already-processed 16 kHz test clips are never trimmed or resampled, so
  test-set scores reproduce the ones we computed during development.
- The XLS-R detectors take a fixed 64,600-sample input (4.04 s, the ASVspoof 2019 protocol). Shorter clips are
  tiled (repeated) to that length; longer clips are scored on up to three evenly spaced windows and the
  spoof log-odds are averaged ([`hearsay/deep.py`](../hearsay/deep.py), `windows()`).
- WavLM sees the first 6 s of a clip (81% of test clips are 3–4 s long).

## 3. Triage

`triage()` in [`hearsay/orchestrator.py`](../hearsay/orchestrator.py) runs once per file, before any expert:

- **Container facts** from ffprobe: format, codec, lossy or lossless, native sample rate, channels, bit depth,
  bitrate, encoder tag, all metadata tags, and whether the extension matches the real container.
- **Chain of custody:** size, modification time and a SHA-256 of the exact bytes.
- **Signal facts:** duration, peak, RMS level, share of clipped samples, silence (RMS below −60 dBFS), and the
  effective bandwidth. The bandwidth estimator looks for a cliff in the long-term spectrum (a band at least
  20 dB below the band just under it, with everything above staying low). A file is band-limited below 0.85 of
  Nyquist and narrowband below 4.5 kHz.

Cost: about 30–40 ms per file, mostly decoding and hashing.

## 4. Deep detectors

Every detector returns a **spoof log-odds** per clip (higher = more synthetic), whatever its native output
convention, so fusion never has to think about polarity. The published models output two logits with index
1 = bona fide; we use `logit[0] - logit[1]`.

### 4.1 Published XLS-R detectors (`w2v2_aasist`, `xlsr_sls`)

| | XLS-R + AASIST (`w2v2_aasist`) | XLSR-SLS (`xlsr_sls`) |
|---|---|---|
| Paper | Tak et al., Odyssey 2022 | Zhang, Wen and Hu, ACM MM 2024 |
| Front end | wav2vec 2.0 XLS-R 300M, fine-tuned end to end | XLS-R 300M |
| Back end | AASIST spectro-temporal graph attention | sensitive-layer-selection head over several hidden layers |
| Training data | ASVspoof 2019 LA (RawBoost augmentation) | ASVspoof 2019 LA |
| Our copy | ONNX from SpeechAntiSpoofingBenchmarks/W2V2-AASIST (pinned commit) | ONNX from SpeechAntiSpoofingBenchmarks/XLSR-SLS (pinned commit) |
| Runtime | onnxruntime on CPU, about 0.2 s per clip (CoreML fails on these graphs) | same |
| Size | 1.26 GB | 1.36 GB |

Measured on our data (all at the model's own boundary, log-odds above 0, unless it says minDCF; minDCF at the
organizers' P_spoof = 0.3):

| | XLS-R + AASIST | XLSR-SLS |
|---|---|---|
| Replica of LA dev, clean (no added noise), minDCF (EER) | 0.020 (0.8%) | – |
| Noise-matched LA dev, minDCF (EER) | 0.140 (6.4%) | 0.116 (6.1%) |
| VCTK real clips flagged (noise-matched dev) | 13.6% | 15.6% |
| VCTK real + unseen generators, minDCF (EER) | 0.179 (6.8%) | 0.214 (9.3%) |
| All 10 DiffSSD generators flagged | 97.2% | 91.0% |
| 7 held-out DiffSSD generators flagged | 97.4% | 92.1% |
| All 32 macOS voices flagged | 99.4% | 99.0% |
| 17 held-out macOS voices flagged | 99.3% | 99.1% |
| LJ Speech (genuine) flagged | 97.5% | 55.8% |
| Test clips flagged (sanity check; the brief says about 30% are synthetic) | 29.4% | 27.2% |

Both generalise well to generators they never saw, and both raise false alarms on real voices that do not
sound like VCTK. The added noise makes XLS-R + AASIST 5.5 times worse (we did not score XLSR-SLS on the clean
copy).

### 4.2 XLS-R + AASIST fine-tuned on test-matched audio (`tak_ft`)

**Port.** The official checkpoint depends on fairseq. We rebuilt the network on Hugging Face transformers
([`hearsay/tak_aasist.py`](../hearsay/tak_aasist.py)): the AASIST back end is copied verbatim from
SSL_Anti-spoofing (MIT), the fairseq wav2vec 2.0 wrapper is replaced by `Wav2Vec2Model`, and a small converter
maps fairseq weight names onto transformers names. Before training, `scripts/finetune_tak.py` checks the port
against the ONNX export on 8 test clips: the largest difference in log-odds was 3.2e-5.

**Recipe** ([`scripts/finetune_tak.py`](../scripts/finetune_tak.py), run with
`--steps 2000 --bs 4 --train-top 6 --eval-every 400 --dev-limit 800`):

| Setting | Value |
|---|---|
| Frozen | CNN feature encoder and the bottom 18 of 24 transformer layers |
| Trained | top 6 transformer layers and the final layer norm (75.6 M parameters), AASIST back end (0.45 M) |
| Batch mix (sampling weights) | VCTK real 40%, LibriSpeech real (25 held-in speakers) 10%, ASVspoof 2019 LA attacks A01–A06 38%, held-in DiffSSD generators and macOS voices 12% |
| Augmentation | fresh noise to a noise floor drawn from the test distribution on every draw; 5% lossy codec round trip; random 4.04 s crop |
| Loss | cross-entropy with class weights 0.1 (spoof) and 0.9 (bona fide), the weighting of the original training code; missing real speech costs 9 times more |
| Optimiser | AdamW, weight decay 1e-4; peak learning rate 1e-5 for XLS-R layers and 1e-4 for the back end; one-cycle schedule with 10% warm-up; gradient clipping at 5 |
| Memory | batch 4, gradient checkpointing, Apple MPS |
| Speed | about 1.1 s per step, so about 40 minutes for 2,000 steps |
| Model selection | minDCF on 800 random noise-matched dev clips (407 real, 393 fake) every 400 steps; the best checkpoint is kept |

Results on those 800 clips (minDCF at P_spoof = 0.5, the setting we trained with): 0.172 (EER 5.4%) before fine-tuning, 0.160 at step 400, 0.153 at step 800, 0.148
at step 1,200 and 0.145 (EER 4.6%) at step 2,000, the checkpoint we kept. On the full evaluation sets
(`scripts/score_tak_ft.py`), compared with the published model (minDCF at P_spoof = 0.3):

| | Published XLS-R + AASIST | Fine-tuned (`tak_ft`) |
|---|---|---|
| Noise-matched LA dev (3,128 clips), minDCF (EER) | 0.140 (6.4%) | 0.123 (4.9%) |
| VCTK real + unseen generators, minDCF (EER) | 0.179 (6.8%) | 0.331 (12.4%) |
| VCTK real clips flagged | 13.6% | 4.0% |
| 7 held-out DiffSSD generators / 17 held-out macOS voices flagged | 97.4% / 99.3% | 36.5% / 99.1% |
| LJ Speech / held-out LibriSpeech genuine clips flagged | 97.5% / – | 5.0% / 4.7% |
| Test clips flagged (sanity check) | 29.4% | 7.7% |

Fine-tuning moved the model's boundary toward "real" and removed most false alarms on unfamiliar voices, but it
also lost most of its ability to catch generators it never saw. In the fusion it received a negative weight and
the cross-validated minDCF got worse (v2e 0.164 vs v2d 0.156 at P_spoof = 0.5).

**Back-end-only variant (`tak_head`).** Same script, data and noise, but XLS-R stays frozen and only the AASIST
back end (0.45 M parameters) is trained (1,500 steps at about 0.7 s, model selection every 300 steps; best at
step 600: 0.172 to 0.149 on the 800 selection clips at P_spoof = 0.5). Saved as `work/models/tak_head/best.pt`, scored with
`TAK_CKPT=work/models/tak_head/best.pt TAK_KEY=tak_head python scripts/score_tak_ft.py ...` (minDCF at P_spoof = 0.3):

| | Back end re-trained (`tak_head`) |
|---|---|
| Noise-matched LA dev (3,128 clips), minDCF (EER) | 0.117 (5.2%) |
| VCTK real + unseen generators, minDCF (EER) | 0.197 (8.7%) |
| VCTK real clips flagged | 3.5% |
| 7 held-out DiffSSD generators / 17 held-out macOS voices flagged | 67.7% / 98.4% |
| LJ Speech / held-out LibriSpeech genuine clips flagged | 38.4% / 16.0% |
| Test clips flagged (sanity check) | 11.3% |

It keeps far more of the published model's reach to unseen generators than the full fine-tune, and it helps the
fusion (v2h 0.154, v2i 0.151, v3 0.146 at the development setting P_spoof = 0.5; v3 is 0.119 at 0.3).

**Scoring:** the score is the mean over up to three 4.04 s windows.
[`scripts/score_tak_ft.py`](../scripts/score_tak_ft.py) builds the network from the official checkpoint and then
replaces every weight with ours; the predictor builds it from the XLS-R config alone and loads our checkpoint,
which gives the same scores (within 5e-7 on the Mac GPU) without the 1.27 GB official file.

### 4.3 WavLM-Large head (`wavlm_head`)

**Features** ([`hearsay/ssl_features.py`](../hearsay/ssl_features.py),
[`scripts/extract_ssl.py`](../scripts/extract_ssl.py)). Frozen `microsoft/wavlm-large`, first 6 s of each clip,
batches of 8. We keep all 25 hidden states (the input embeddings plus 24 transformer layers) and average each
over time. On the Mac GPU the model runs in fp16, but pooling is done in float32: our first run pooled in fp16
and overflowed (summing hundreds of frames exceeds fp16's 65,504 limit), so we recomputed everything. On CPU
the model runs in fp32, so CPU scores can differ slightly from the ones computed on the Mac.

**Layer probe** ([`scripts/train_ssl_head.py`](../scripts/train_ssl_head.py)). One logistic regression per
layer, trained on the same data as the head, scored by minDCF (at P_spoof = 0.5) on a pooled evaluation set (noise-matched LA
dev, held-out DiffSSD generators and macOS voices, LJ Speech, held-out LibriSpeech speakers: 2,719 real,
2,645 fake):

| Layer | 0 | 2 | 4 | 5 | **6** | **7** | **8** | **9** | 10 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| minDCF | 0.766 | 0.488 | 0.320 | 0.305 | **0.242** | **0.240** | **0.251** | **0.265** | 0.265 | 0.293 | 0.310 | 0.312 | 0.380 |

The evidence peaks in the middle of the network. The bottom layers are close to raw acoustics and the top
layers encode words and meaning; synthesis traces live in between. Our first head (v2a–v2g) used the best four
layers, 6–9 (probe saved in `work/heads/headB_report.json`). The current head (v2h on) uses the wider band
3–12 (`--layers 3-12`), which did better on every evaluation below. Both choices were made on this evaluation
set, which makes the head's own evaluation numbers slightly optimistic.

**Head.** The chosen layer means are concatenated (10 x 1,024 = 10,240 numbers for the current head; 4,096 for
the first), standardised, and fed to an L2-regularised logistic regression (C = 0.01, real clips weighted 4x).
Training data: 9,694 clips (3,021 real, 6,673 fake): noise-matched LA train, the held-in LibriSpeech speakers,
the 3 held-in DiffSSD generators and the 15 held-in macOS voices. Saved as `wavlm_head.npz` (layers, mean,
scale, coefficients, intercept).

We compared the two heads during development, at P_spoof = 0.5. The first head's scores were overwritten when
it was replaced, so only the current head can be re-scored at the organizers' 0.3: 0.079 on seen attacks,
0.252 on unseen generators and 0.182 on the pooled set (same EERs as below).

| Evaluation | First head, layers 6–9 | Current head, layers 3–12 |
|---|---|---|
| Noise-matched LA dev (seen attacks), minDCF (EER) | 0.131 (3.7%) | 0.111 (3.5%) |
| VCTK real + unseen generators, minDCF (EER) | 0.354 (10.6%) | 0.344 (10.4%) |
| Pooled evaluation set above, minDCF (EER) | 0.237 (7.6%) | 0.222 (7.6%) |
| VCTK real clips flagged | 4.7% | 3.4% |
| Held-out DiffSSD / macOS fakes flagged | 68.3% / 98.9% | 63.3% / 98.4% |
| LJ Speech / held-out LibriSpeech genuine clips flagged | 7.9% / 5.7% | 7.4% / 4.0% |
| Test clips flagged (sanity check) | 17.4% | 15.1% |

### 4.4 Transductive self-training (`wavlm_selftrain`, optional)

The test set contains attacks our training data never saw. The ensemble still ranks most of them correctly,
so its most confident calls are reliable enough to use as pseudo-labels. No ground truth is used at any point.

1. Fit the WavLM head on the base training data (`selftrain_base.npz`: the 9,694 training rows above, float16).
2. Rank the test clips by z(anchor) + z(head), where the anchor is the mean z-score of the published XLS-R
   detectors.
3. Label the top 15% synthetic and the bottom 50% real, add them to the base data (pseudo-label weight 1, real
   clips still weighted 4x), and refit. Repeat once more (two rounds in total).

These are the predictor's defaults (`SELF_TRAIN_DEFAULTS` in `hearsay/pipeline.py`; a `self_train` block in
`fusion_meta.json` overrides them). Self-training needs a batch: with fewer than 50 files the predictor uses
the un-adapted WavLM head instead and says so.

In a labelled simulation (half of the VCTK dev speakers as real, held-out DiffSSD generators as fakes; labels
used only to measure), the fused anchor + head minDCF improved from 0.269 to 0.229–0.251 (P_spoof = 0.5), depending on the
pseudo-label fractions. Caveat: a file's score now depends on the rest of the batch.
The submitted fusion (v3) does **not** use self-training: on the full dev mix it made the fusion worse (0.179 vs 0.156 at P_spoof = 0.5).

## 5. Forensic experts

### The contract

[`hearsay/evidence.py`](../hearsay/evidence.py) defines what every expert returns:

- `Evidence`: expert name, category (one of 9: container, spectral, prosody, environment, compression,
  speaker, deep, splice, semantic), numeric `features` for the glass-box model, human-readable `findings`,
  an optional score, runtime, plot artifacts, and an error field.
- `Finding`: a sentence, a direction (+1 synthetic, −1 genuine, 0 neutral) and a strength from 0 to 1.
- `@expert(name, category, cost)`: registers the function and wraps it so an exception becomes an
  "expert failed" record instead of a crash. [`hearsay/experts/__init__.py`](../hearsay/experts/__init__.py)
  imports each module defensively, so one missing dependency never takes the pipeline down.

Conventions: every feature is prefixed with its expert's name; each expert returns the same set of keys for
every file, and a quantity that cannot be measured is NaN, never a made-up default. Heavy intermediates (STFT,
log-mel spectrogram, LFCCs, F0 track, voice activity, 25 ms frame analysis) are computed once and shared through
`ctx.cache`.

### The experts

| Module | Experts | Notes |
|---|---|---|
| [`container.py`](../hearsay/experts/container.py) | `container` | Never looks at samples. Magic bytes vs extension, RIFF chunk walk (INFO, bext, iXML, size fields), ID3/Xing/LAME/MP4/Vorbis tags via mutagen, exiftool fallback, encoder signatures of transcoders and TTS services, embedded vs filesystem times, SHA-256. Honesty rule: when a transcoder has rewritten the file (every test clip), it says the evidence is uninformative. |
| [`compression.py`](../hearsay/experts/compression.py) | `codec_cutoff`, `double_compression` | The cut-off expert finds the low-pass edge at the native rate and names the chain that usually produces it (telephone codec near 3.4–4 kHz, 16 kHz audio near 7–8 kHz, 22.05 kHz TTS vocoders near 11 kHz). The double-compression expert scans MDCT block grids (576-sample MP3 granules, 1,024-sample AAC frames) in decoded audio; for MP3 it predicts where the file's own grid must sit, so a grid at the wrong phase or a second grid reveals re-encoding. The organizers' 16 to 22.05 to 16 kHz round trip preserves a 16 kHz MP3 grid, so the scan stays meaningful on the test set. |
| [`spectral.py`](../hearsay/experts/spectral.py) | `spectral_stats`, `lfcc_stats`, `vocoder_artifacts` | Long-term spectral shape and sub-band dynamics; 20 LFCCs + deltas (the ASVspoof baseline front end, which keeps full resolution at high frequencies where vocoder artefacts live); harmonic regularity, phase coherence, group delay, upsampling lines, frame-rate lines in the modulation spectrum, constant-Q sharpness of low harmonics. |
| [`prosody.py`](../hearsay/experts/prosody.py) | `pitch`, `pauses_breaths`, `voice_quality` | Praat (parselmouth) F0 track, jitter, shimmer, HNR; webrtcvad pauses, syllable-nuclei speaking rate (de Jong and Wempe), a breath detector (150–600 ms of broadband, aperiodic, low-level noise next to pauses); formants and bandwidths, H1–H2, cepstral peak prominence, creaky voice. Findings are worded "consistent with", with modest strengths, because careful speakers, noise gates and compression can mimic several of these cues. |
| [`environment.py`](../hearsay/experts/environment.py) | `enf`, `noise_floor`, `reverb_replay` | A real recording happens in a room, on grid power, through a microphone; a synthesiser has none of these. Mains hum at 50/60 Hz and harmonics tracked over time; background level, stationarity and colour in the pauses, and whether the noise under speech matches the noise in pauses; decay after speech offsets, a modulation-based reverberation ratio, and loudspeaker-like band-pass shape for replay. |
| [`splice.py`](../hearsay/experts/splice.py) | `discontinuity` | Five continuity checks: exact digital-zero runs of 5 ms or more (split into leading, interior and trailing), DC-offset steps, noise-floor change points between pauses, sample-level clicks, and amplitude-histogram combs from re-quantisation. Returns seam times for the report overlay. |
| [`speaker.py`](../hearsay/experts/speaker.py) | `speaker_consistency` | SpeechBrain ECAPA-TDNN voiceprints on 1.5 s windows (hop 0.5 s) compared with the clip's average voice, plus a split scan comparing the speech before and after every candidate cut. Thresholds were set on 110 test clips, 20 LJ, 20 DiffSSD and 20 synthetic two-talker splices: 16 of 20 splices and 0 of 20 LJ clips are flagged. Caches the voiceprint so the batch step can cluster speakers. |
| [`semantic.py`](../hearsay/experts/semantic.py) | `transcript`, `repeated_script` | Whisper small.en transcript (cached for the test set): speaking rate, fillers, word length, recogniser loops, confidence. Batch-level script statistics: how many other clips speak the same sentence (fuzzy match), how far this reading deviates from the others. |

Costs and what we observed on the challenge data are in the README's expert table. Two batch-level steps run
around the per-file loop: `build_corpus_stats` (repeated-script counts) before it, and `cluster_speakers`
(voiceprint clustering, used to score how typical a clip is of its speaker) after it.

## 6. The glass-box model

[`scripts/extract_forensic.py`](../scripts/extract_forensic.py) computes one row of 293 raw features per clip
with the 13 per-file experts (training: 3,000 noise-matched LA train clips, half real; 900 LibriSpeech clips;
all DiffSSD and macOS clips; plus the evaluation sets). [`scripts/train_forensic_model.py`](../scripts/train_forensic_model.py)
then cleans the feature set before training:

1. drop features that are constant on the test set (all `container_*` features: every test header is identical);
2. drop features an expert's author flagged as corpus shortcuts (codec cut-off depth and steepness,
   high-band flatness of the compression expert);
3. **adversarial validation:** drop any feature whose AUC for "our noise-matched dev set vs the test set" is
   further than 0.3 from 0.5. Such a feature measures a dataset difference, not synthesis. This removed 42
   features: all but one of the mains-hum features, the codec cut-off frequency and bandwidth features, the
   interior and trailing digital-silence run lengths, and several LFCC means and spreads.

216 features remain. The model is LightGBM (binary objective, learning rate 0.03, 15 leaves, at least 40 clips
per leaf, 70% feature and 80% row subsampling, L2 1.0, 400 rounds), with real clips weighted 4x, trained on
noise-matched LA train plus the held-in halves of LibriSpeech, DiffSSD and our macOS voices.

| Evaluation | Result |
|---|---|
| Noise-matched LA dev, minDCF at P_spoof = 0.3 (EER) | 0.384 (15.6%) |
| VCTK real + unseen generators, minDCF at P_spoof = 0.3 (EER) | 0.382 (15.5%) |
| Held-out DiffSSD / macOS fakes flagged | 48.8% / 90.2% |
| LJ Speech / held-out LibriSpeech genuine clips flagged | 20.7% / 12.9% |

Most useful features (LightGBM gain):

| Rank | Feature | Plain meaning |
|---|---|---|
| 1 | `lfcc_dstd_1` | how much the spectral tilt changes from frame to frame |
| 2 | `pauses_breaths_speech_fraction` | share of the clip that is speech rather than pause |
| 3 | `lfcc_ddstd_1` | how abruptly those tilt changes themselves change |
| 4 | `reverb_replay_srmr_db` | speech-to-reverberation modulation ratio (how "roomy" the voice is) |
| 5 | `vocoder_harm_lo` | harmonic structure in the low frequencies |
| 6 | `pauses_breaths_lead_silence_s` | silence before the first word |
| 7 | `pauses_breaths_speech_rate` | syllables per second |
| 8 | `pitch_f0_std_st` | pitch variability, in semitones |
| 9 | `voice_quality_b3_median_hz` | bandwidth of the third formant |
| 10 | `pitch_shimmer_apq3_pct` | cycle-to-cycle loudness jitter (shimmer) |

Caveat: leading silence and speech fraction may partly describe how ASVspoof 2019 was recorded and generated,
not synthesis in general.

## 7. Fusion and calibration

Written by [`scripts/make_submission.py`](../scripts/make_submission.py) into
`submissions/<version>/fusion_meta.json`, applied by `fuse()` in [`hearsay/pipeline.py`](../hearsay/pipeline.py)
with the same arithmetic in the same order.

For detector i with raw spoof log-odds x_i:

```
z_i   = (x_i - z_mean_i) / z_std_i            dev-set mean and standard deviation; a missing score gets z_i = 0
fused = sum_i w_i * z_i + b                   logistic regression, real clips weighted 4x
p     = sigmoid(a * fused + c + s)            Platt scaling (a, c) fitted on dev; prior shift s
p     = clip(p + rank(fused) * 1e-12, 1e-6, 1 - 1e-6)   every value unique, order unchanged
```

- **Dev set:** noise-matched LA dev (1,628 real, 1,500 fakes from the 6 seen attacks) plus the held-out
  generators (694 DiffSSD and 451 macOS clips): 1,628 real and 2,645 fake. Mixing in unseen generators keeps the
  weights from favouring detectors that only know the seen attacks.
- **Regularisation:** C is chosen from {0.01, 0.03, 0.1, 0.3, 1.0} by grouped 5-fold cross-validated minDCF.
  Groups are speakers for real clips and attacks or voices for fakes, so a speaker or generator is never on both
  sides of a fold. The reported fusion numbers are these out-of-fold numbers.
- **Prior shift:** s = logit(0.30) − logit(dev share of fakes), moving the calibrated probability to the brief's
  "about 70% real". It changes no ranking.
- **Condition-aware variant (`--cond`):** adds a quiet-clip indicator q (noise floor below −55 dBFS) and
  model-by-quiet interaction terms: fused = sum_i w_i z_i + sum_i v_i z_i q + u q + b. It follows the finding
  that very clean test clips are strongly spoof-leaning. The submitted fusion (v3) uses these condition-aware terms.

Interim example (v2c, four detectors, C = 0.3): weights on the z-scores are 1.57 (XLS-R + AASIST), 0.94
(XLSR-SLS), 3.36 (WavLM head) and 0.78 (glass-box model); Platt a = 1.06, c = 1.43; prior shift −1.33; grouped
CV minDCF 0.178 (EER 5.2%) at P_spoof = 0.5. The WavLM head gets the largest weight although it is not the best single model on
this dev set: it disagrees most usefully with the XLS-R models. Weights of correlated inputs should not be read
as importance on their own. Submitted fusion (v3, `submissions/v3/fusion_meta.json`), weights on z-scored log-odds; the quiet column adds
the model-by-quiet interaction (noise floor below -55 dBFS):

| Detector | Normal clips | Quiet clips |
|---|---|---|
| `w2v2_aasist` | +2.30 | +1.04 |
| `xlsr_sls` | +1.08 | +0.45 |
| `wavlm_head` | +3.34 | +4.16 |
| `tak_head` | +0.53 | -0.47 |

Quiet-clip offset -0.71; regularisation C = 0.3; Platt scaling a = 1.122, c = 1.459; prior shift to 30% synthetic = -1.333.

## 8. The agentic router

### Predicates

The policy is written in terms of named predicates ([`hearsay/orchestrator.py`](../hearsay/orchestrator.py),
`PREDICATES`); every evaluation is logged with a human-readable reason.

| Predicate | True when |
|---|---|
| `always` | every file |
| `deep_scorer_available` | a deep scorer (the detector ensemble) was supplied |
| `lossy_or_band_limited` | lossy codec, extension that does not match the container, effective bandwidth below 0.85 of Nyquist, or an expert's cut-off measurement below that |
| `uncertain` | the provisional log-odds is within 1.0 of the decision threshold |
| `strong_flag` | a non-deep finding points to synthetic with strength 0.6 or more |
| `conflict` | strong findings point both ways, or a strong finding contradicts a confident deep score |
| `long_clip` | longer than 6 s (room for a local edit) |
| `has_signal` | decoded, and RMS at or above −60 dBFS |
| `narrowband` | effective bandwidth below 4.5 kHz |
| `very_short` | shorter than 2 s |

### Policy

Rules combine predicates with OR over `when` and AND over `requires`. A rule can name experts explicitly and
can also sweep in any other registered expert of its cost tier, so a new expert is routed sensibly without
editing the policy.

1. `cheap_baseline` (always): container, codec_cutoff, spectral_stats, discontinuity, enf, noise_floor,
   pauses_breaths, repeated_script, plus any other cheap expert (lfcc_stats).
2. `deep_detector` (when a deep scorer is available): the provisional score.
3. `compression_history` (when lossy or band-limited): double_compression.
4. `escalate_medium` (when uncertain, strong flag, conflict or long clip; requires signal): pitch,
   voice_quality, speaker_consistency, vocoder_artifacts, reverb_replay, plus any other medium expert.
5. `escalate_expensive` (when still uncertain; requires signal): transcript, plus any other expensive expert.

Suppressions, decided from triage facts and logged: `vocoder_artifacts` for narrowband audio,
`speaker_consistency` and `enf` for clips under 2 s. `--skip NAME` disables an expert in both modes. `--mode
full` fires every rule, which is what the compute comparison uses.

### Provisional score

The provisional score is re-evaluated each time a rule asks whether the file is uncertain:

- with a fusion hook (the predictor supplies one): logit(p) − logit(0.5) of the submitted probability;
- otherwise, the deep detector's standardised score plus half of the expert vote;
- otherwise, the expert vote alone.

The vote is a transparent, hand-set prior used only when nothing learned is available: the sum over findings of
direction x strength x category weight x 2, clipped to ±6 log-odds. Category weights: deep and spectral 1.0;
splice, prosody and speaker 0.8; environment 0.7; semantic 0.5; compression and container 0.4 (they describe
the processing chain, not the talker).

### Attribution (first match wins)

1. **uncertain** if there is no decodable signal or no usable evidence;
2. **spliced/edited** if a splice expert reports a seam with strength 0.6 or more;
3. **replay** if an environment or replay finding (reverb, loudspeaker, re-recording) is synthetic-leaning with
   strength 0.6 or more;
4. **voice conversion** if the cm-score is at least 0.65 and the speaker expert flags a change, or prosody leans
   natural while spectral and deep evidence lean synthetic (conversion keeps the source talker's delivery but
   replaces the timbre);
5. **TTS / neural vocoder** if the cm-score is at least 0.65 otherwise;
6. **laundered** (transcoded, band-limited or noisy) if compression evidence reaches 0.5, or the file is lossy
   or band-limited, while the score is below the synthetic threshold;
7. **bona fide** if the cm-score is at most 0.35 and no synthetic finding reaches 0.6;
8. **uncertain** otherwise.

Secondary tags (splice evidence, a lossy processing chain, replay cues) are reported whatever the label. These
are rules, not a trained classifier; the organizers state the test set has no partial fakes, so labels 2–4 are
for real-world files.

### Compute accounting

`compute_accounting()` records, per batch, how many files reached each tier and the time spent per expert, and
prices a full run with the mean measured cost of each expert. `python -m hearsay.cli compute` runs both modes on
a sample after an untimed warm-up (imports, JIT compilation, model loads), so one-off costs do not bias the ratio.

Measured on 20 random test clips (deep scores and transcripts from cache):

| | Full mode | Agentic mode |
|---|---|---|
| Expert analysis time | 14.8 s | 5.7 s (39%) |
| Files reaching the medium tier | 20 | 8 |
| Files reaching the expensive tier | 20 | 7 |
| Files needing compression history | 20 | 0 |

Mean cost per call in full mode: speaker_consistency 390 ms, double_compression 202 ms, vocoder_artifacts 67 ms,
voice_quality 25 ms, pitch 14 ms, spectral_stats 11 ms, pauses_breaths 7.5 ms, codec_cutoff 6.8 ms,
reverb_replay 6.3 ms, discontinuity 4.0 ms, enf 4.0 ms, lfcc_stats 0.8 ms, container 0.5 ms, noise_floor 0.4 ms,
transcript 0.5 ms (cached text; live Whisper costs about 0.5 s per clip after a 2 s model load), repeated_script 0.03 ms.
On the full test set with the submitted fusion (v3, `predict --reports`), 1,671 files: every file got triage,
the cheap tier and the detector ensemble; 3 files (0.2%) reached the conditional compression tier, 145 (8.7%)
the medium tier and 109 (6.5%) the expensive tier. Analysis time was 136.5 s against an estimated 1,265 s for
running every expert on every file (11%), with a median of 58.5 ms per file.

## 9. Reports

[`hearsay/report.py`](../hearsay/report.py) writes self-contained HTML (plots are embedded PNGs, CSS is inline;
templates are autoescaped, so expert text is never trusted as HTML).

**Per file:** verdict (cm-score, gauge, likely synthetic / likely genuine / inconclusive, manipulation type and
why); plain-English summary; chain of custody (all triage facts and tags); an evidence-balance chart per family
(red toward synthetic, blue toward genuine); waveform and spectrogram with seams (red dashed), digital-silence
runs (shaded) and the measured cut-off (orange); pitch contour and speaker-drift curve when those experts ran;
extra plots experts attach; a table of every finding by family with key measurements, direction, strength and
runtime; the routing trace (each rule, fired or skipped, and why, with suppressed or missing experts); the
evidence that had no effect (neutral findings, experts that ran without a finding, failed or partly failed
experts, experts that were available but not run); the attribution rules; and every numeric feature.

**Batch dashboard (`index.html`):** counts of likely synthetic, likely genuine and inconclusive files, median
time per file, the agentic compute fraction, a score histogram, flags per family, manipulation-type counts, files
reaching each tier, compute per expert, batch-level steps, and a sortable table linking every file report.

**`evidence.json`:** every file's evidence, findings, features, trace and scores, plus the policy, the batch
steps and the compute accounting.

## 10. Metric and submission writer

[`hearsay/metrics.py`](../hearsay/metrics.py) mirrors the organizers' modified ASVspoof 5 code (the same DET
computation with a stable mergesort, C_miss = 1, C_fa = 4 and P_spoof = 0.3, the value the organizers confirmed on
26 September; the copy shipped to teams said 0.5, which we used during development) and reports two readings: *intended*
(spoof is the target class and 1.0 = synthetic, as the written specification says) and *shipped flipped* (what the
unmodified code computes on our scores). Every number in our documentation is the intended reading.

[`hearsay/submission.py`](../hearsay/submission.py) writes `filename<TAB>cm-score` in the template's order and
validates the result: same header, same names in the same order, every value finite and inside [0, 1], and no
two values equal.

## 11. Test-matched training and validation data

- **Pipeline replica** ([`hearsay/preprocess.py`](../hearsay/preprocess.py)): `librosa.load` at 22,050 Hz,
  `librosa.effects.trim` (top_db 60, frame 2,048, hop 512), drop clips under 3 s, crop sources longer than
  4.5 s to a duration drawn from the test set's duration quantiles (on the 512-sample grid), resample to 16 kHz,
  peak-normalise to 0.998, write PCM16.
- **Noise-floor matching** ([`hearsay/augment.py`](../hearsay/augment.py)): the noise floor is the 10th percentile
  of 20 ms frame RMS in dBFS. The test set's 1,671 floors are measured once and cached. For each clip we draw a
  target from them; if it is at least 1 dB above the clip's own floor, we add noise scaled so the floor reaches
  it, then re-normalise the peak. Noise is a random excerpt of one of 400 MUSAN noise recordings, or, 30% of the
  time, white noise, half of which is turned into brown-ish noise.
- **Codecs:** 5% of clips go through ffmpeg round trips: MP3 at 24–128 kbps, AAC at 24–96 kbps, Opus at 12–32
  kbps, G.726 at 16–32 kbps (8 kHz), G.722, or µ-law at 8 kHz. The codec delay is removed by cross-correlation so
  the clip stays aligned.
- **Class-agnostic:** the augmentation choice never depends on the label.
- **Group splits:** `name@a` / `name@b` are two halves of a dataset by a CRC32 hash of the group name
  (generator for DiffSSD, voice for macOS TTS, speaker for LibriSpeech). Held-in halves (`@a`) train; held-out
  halves (`@b`) evaluate and fit the fusion.

| Set | Real | Fake | Notes |
|---|---|---|---|
| `la_train_m` | 1,580 | 6,000 | 1,000 per attack A01–A06 |
| `la_dev_m` | 1,628 | 1,500 | 250 per attack; `la_dev_c` is the same clips without added noise |
| `diffssd_m` | – | 991 | `@a`: DiffGAN-TTS, ElevenLabs, OpenVoice V2 (297); `@b`: Grad-TTS, PlayHT, ProDiff, UnitSpeech, WaveGrad 2, XTTS v2, YourTTS (694) |
| `say_m` | – | 827 | 32 macOS voices; `@a` 15 voices (376), `@b` 17 voices (451) |
| `libri_m` | 2,290 | – | 40 LibriSpeech dev-clean speakers; `@a` 25 (1,441), `@b` 15 (849) |
| `lj_m` | 242 | – | evaluation only |

## 12. Compute, memory and caching

We developed on a 16 GB Apple-silicon MacBook, so memory shaped the design:

- The predictor loads **one model at a time** and frees it before loading the next (each large model is about
  1.3 GB on disk and needs more than that in memory while it runs).
- **Score cache:** each detector's scores go to `work/cache_predict/<model>.json` together with fingerprints of
  every audio file and of the model files (and, for the forensic model, of the expert code). A re-run computes
  only what changed; a changed model or file is rescored.
- **Devices:** `--device auto` picks CUDA, then Apple MPS, then CPU. The ONNX detectors always run on CPU. The
  Docker image is CPU-only.
- **Training jobs** ran one at a time; [`scripts/mem_watchdog.sh`](../scripts/mem_watchdog.sh) stops a job if
  free system memory drops below a threshold.

Measured per clip on 8 test clips (laptop CPU, 2 threads): XLS-R + AASIST 0.43 s, XLSR-SLS 0.49 s, fine-tuned
XLS-R + AASIST 0.32 s, WavLM-Large 0.75 s, forensic features 0.33 s (2 processes); on the Mac GPU the fine-tuned
XLS-R takes 0.11 s and WavLM 0.065 s. Time for the full test set in Docker (6 CPU threads, 8 GB; from `predict`'s timing
printout): XLS-R + AASIST 750 s, XLSR-SLS 764 s, re-trained XLS-R + AASIST 792 s, WavLM-Large 1,002 s and 11 s
of model loading, 3,322 s in total (55 minutes, 2.0 s per clip). On the laptop we scored each detector
separately with the development scripts and cached the results; writing all 1,671 reports from those cached
scores took 589 s, and the router's analyses used 137 s of the estimated 1,265 s that running every expert
on every file would take (11%).

## 13. Configuration and model files

`fusion_meta.json` is the configuration: the predictor runs exactly the detectors in its `models` list and uses
its statistics and weights (`z_mean`, `z_std`, the coefficient vector, `intercept`, `platt`, `prior_shift`,
`cond`, and an optional `self_train` block). A new fusion needs no code change.

Model files live under one folder, `$HEARSAY_MODELS` (default `work/models/`; the Docker image uses
`/app/models`). `python -m hearsay.paths --meta <fusion_meta.json>` checks that every file the fusion needs is
there. `scripts/fetch_models.sh` downloads the public weights, pinned to exact Hugging Face commits with sizes and
SHA-256 hashes (`scripts/public_models.tsv`); our trained artifacts come from an archive made by
`scripts/export_models.sh` and published as the GitHub release asset
[`hearsay-models-v3.tar.gz`](https://github.com/MeagerPotato/hearsay-dh-squad/releases/tag/v1.0) (588.6 MB,
SHA-256 `3565772ef137b1091aa944fbc461348aac4327176899f128ee9e7e0fdfaabc40`). The Dockerfile downloads it by default.

| Detector key | Files under `$HEARSAY_MODELS` | Source |
|---|---|---|
| `w2v2_aasist` | `w2v2-aasist/w2v2-aasist.onnx` (1.26 GB) | SpeechAntiSpoofingBenchmarks/W2V2-AASIST |
| `xlsr_sls` | `xlsr-sls/xlsr-sls.onnx` (1.36 GB) | SpeechAntiSpoofingBenchmarks/XLSR-SLS |
| `tak_ft` | `tak_ft/best.pt` (1.26 GB, ours; 0.63 GB as float16 in the Docker image), `xls-r-300m/config.json` (architecture) | ours; facebook/wav2vec2-xls-r-300m |
| `tak_head` | `tak_head/best.pt` (same network, only the AASIST back end re-trained), `xls-r-300m/config.json` | ours; facebook/wav2vec2-xls-r-300m |
| `wavlm_head` | `wavlm-large/` (config, preprocessor config, 1.26 GB weights), `wavlm_head.npz` (206 KB, ours) | microsoft/wavlm-large; ours |
| `wavlm_selftrain` | `wavlm-large/`, `selftrain_base.npz` (184 MB, ours) | microsoft/wavlm-large; ours |
| `forensic_gbm` | `forensic_gbm.txt` (0.7 MB), `forensic_gbm_cols.json` (ours) | ours |
| reports only | `spkrec-ecapa-voxceleb/` (about 90 MB) | speechbrain/spkrec-ecapa-voxceleb |
| reports only | Whisper small.en (Hugging Face cache; only with `--asr`, otherwise cached test transcripts) | openai/whisper-small.en |
