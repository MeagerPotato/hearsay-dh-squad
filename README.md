# HEARSAY

A detector for synthetic speech that writes a forensic report explaining each call. Built by team dh squad for the NSA HEARSAY challenge at HackGT 13 (September 2026).

Give HEARSAY an audio clip and it returns the probability that the speech is synthetic (1.0 means synthetic), plus a self-contained HTML report showing the evidence behind the number. On audio matched to the challenge's test conditions, the fused score reaches a cross-validated minDCF of 0.076, against 0.116 for the best published detector we tried. Details are in [Results](#results).

```
audio file
  -> triage (container, codec, level, bandwidth, SHA-256)
  -> four deep detectors, every file -> condition-aware fusion -> score in (0, 1)
  -> router -> 16 forensic experts, only where needed ---------> HTML report
```

## The problem

The challenge scored one number per clip: how likely is it synthetic? Judging was 60% minDCF, 20% technique, 20% documentation. The cost function made a false alarm four times as expensive as a miss, and assumed 30% of the 1,671 test clips were synthetic. That pushes a detector to leave real recordings alone, because an analyst chasing a real call has wasted the time the tool was meant to save.

The cost function reduces to this, where t is the decision threshold:

```
minDCF = min over t of [ P_miss(t) + 1.71 * P_fa(t) ]
```

Flagging nothing scores 1.0. A perfect ranking scores 0.

## What the data showed us

We measured the test set before training anything.

- The test clips are VCTK sentences, the corpus behind ASVspoof 2019. The organizers' training material (DiffSSD and LJ Speech) is a different domain, so a model trained only on it learns speaker, text and sample-rate shortcuts.
- 94% of clip lengths are a whole number of 512-sample hops at 22,050 Hz, the trace of `librosa.load()` plus `librosa.effects.trim()`. We rebuilt that preprocessing in [`hearsay/preprocess.py`](hearsay/preprocess.py) and ran every training clip through it.
- The organizers added background noise. The test set's median noise floor is -44 dBFS against -61 dBFS for the same source audio, about 17 dB noisier.

The noise matters most. The best-known published detector (XLS-R + AASIST) scores minDCF 0.020 on the clean replica and 0.140 once test-matched noise is added. It also flags 97.5% of genuine LJ Speech clips as fake, because it learned that real speech sounds like VCTK studio audio.

## Approach

Training data comes from ASVspoof 2019 LA (train and dev partitions only, never evaluation), three DiffSSD generators, LibriSpeech voices, and 827 clips we generated with macOS voices. Each clip passes through the preprocessing replica, then gets MUSAN noise added until its noise floor follows the test set's distribution. Real and fake clips get the same treatment, so noise cannot become a label. About 5% of clips also go through a lossy codec round trip.

Four detectors score every file:

| Detector | Origin |
|---|---|
| XLS-R + AASIST | published (Tak et al.) |
| XLSR-SLS | published (Zhang et al., ACM MM 2024) |
| XLS-R + AASIST, back end re-trained | ours, XLS-R frozen, noise-matched audio |
| WavLM-Large layers 3-12 + logistic head | ours |

A logistic regression fuses them. Real clips are weighted 4x, regularisation is chosen by grouped cross-validated minDCF, and the weights change on very quiet clips (noise floor below -55 dBFS), where the published models fail most. Platt scaling then turns the output into a probability at the 30% prior.

Next to the score sits a forensic layer: 16 interpretable experts covering pitch, pauses and breaths, spectral dynamics, reverberation, compression traces, splices, speaker consistency, metadata and the spoken text. A rule-based router runs the cheap checks on every file and the slow ones only when the evidence is uncertain, conflicting or strongly flagged, and logs why. On the full test set it used 11% of the analysis time of running every expert, with identical scores. The report puts each detector's contribution beside the plain-English findings.

## Results

All numbers are minDCF at P_spoof = 0.3 and C_fa = 4, the organizers' confirmed settings. Seen attacks are noise-matched ASVspoof 2019 LA dev clips from speakers not in training. Unseen generators are 1,145 fakes from synthesis systems no component trained on (7 DiffSSD generators, 17 macOS voices).

| Model | Seen attacks | Unseen generators | Real LJ Speech flagged |
|---|---|---|---|
| XLS-R + AASIST, published | 0.140 | 0.179 | 97.5% |
| XLSR-SLS, published | 0.116 | 0.214 | 55.8% |
| WavLM-Large head, ours | 0.079 | 0.252 | 7.4% |
| Fusion, submitted (out of fold) | 0.076 | 0.153 | 37.6% |

The fusion's EER is 3.7% on seen attacks and 6.0% on unseen generators. The WavLM head is the best single model on seen attacks and the published XLS-R model is the best on unseen ones, which is why fusing them helps.

These are development numbers. The same dev data chose the WavLM layers, the fine-tuning checkpoint and the fusion weights (the fusion figures are cross-validated), so they run slightly optimistic.

What did not work, all measured:

- Fine-tuning the whole XLS-R + AASIST model cut false alarms on LJ Speech from 97.5% to 5%, but unseen-generator minDCF went from 0.179 to 0.331. It forgot what made the published model generalise.
- Denoising before detection made things worse (minDCF 0.13 to 0.17 against 0.11 on the same clips at P_spoof = 0.5). Training with matched noise worked instead.
- Transductive self-training helped in a labelled simulation and hurt on the full dev mix.
- Adding the interpretable LightGBM model to the fusion did not improve minDCF, so it only drives the explanations.

Separately, the organizers' scorer treats a higher score as bona fide while the spec asks for 1.0 = synthetic. Run unmodified, a perfect detector scores 1.0. We followed the spec, reported the mismatch to the organizers, and [`hearsay/metrics.py`](hearsay/metrics.py) computes both readings.

Full measurements are in [docs/FINDINGS.md](docs/FINDINGS.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Demo

Open these in a browser (each is one self-contained HTML file):

1. [`docs/sample_reports/dashboard_full_test_set.html`](docs/sample_reports/dashboard_full_test_set.html) for the batch view: score histogram, manipulation types, routing tiers and the 11% compute figure.
2. [`docs/sample_reports/confident_synthetic_HGT8852055.html`](docs/sample_reports/confident_synthetic_HGT8852055.html) for one file end to end: verdict gauge, chain of custody, spectrogram with a suspected seam at 4.2 s, codec cut-off and pitch contour.
3. [`docs/sample_reports/inconclusive_escalated_HGT8649435.html`](docs/sample_reports/inconclusive_escalated_HGT8649435.html) for a borderline file, where the routing trace shows the router escalating and why.

## Run it

Python 3.12 or 3.13, with `ffmpeg` and `ffprobe` on the PATH.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
scripts/fetch_models.sh
scripts/fetch_models.sh --own https://github.com/MeagerPotato/hearsay-dh-squad/releases/download/v1.0/hearsay-models-v3.tar.gz --own-only

# score a folder and write the submission TSV (add --reports reports/ for HTML reports)
python -m hearsay.cli predict data/HackGTHearsayTesting \
    --meta submissions/v3/fusion_meta.json \
    --template data/HackGTHearsayTesting/HGT_Hearsay_score_template.csv \
    --out submissions/final/dh_squad_predictions.tsv

# explain one file
python -m hearsay.cli explain data/HackGTHearsayTesting/HGT1013455.wav --html reports/HGT1013455.html
```

Model files are not in git (about 1.3 GB each). With Docker, `docker build -t hearsay-dhsquad .` followed by one `docker run` scores the test folder on CPU; see [docs/DOCKER.md](docs/DOCKER.md). On our laptop the full test set took 55 minutes and reproduced the submitted scores with rank correlation 0.999999 and no verdict changed.

## Repository layout

```
hearsay/       the package: preprocessing, detectors, fusion, 16 experts, router, reports, CLI
scripts/       data preparation, training, scoring, fusion
submissions/   submitted TSVs and the fusion_meta.json behind each version
docs/          findings, architecture, Docker notes, sample reports
```

The original long-form README, with the full expert table, the 12-step training recipe and the submission history, is kept as [docs/FULL_REPORT.md](docs/FULL_REPORT.md).

## Credits and licences

We built on other people's data and models.

| Resource | How we use it | Licence |
|---|---|---|
| [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336) (Yamagishi et al.) | training and validation (train and dev partitions only) | ODC-By 1.0 |
| [DiffSSD](https://huggingface.co/datasets/purdueviperlab/diffssd) (Purdue, provided by the organizers) | 3 generators for training, 7 held out | CC BY-NC-ND 4.0 for the synthetic audio; we do not redistribute any of it |
| [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) (provided subset) | validation only | public domain |
| [LibriSpeech](https://www.openslr.org/12/) dev-clean | extra real voices | CC BY 4.0 |
| [MUSAN](https://www.openslr.org/17/) | noise for augmentation | CC BY 4.0 |
| [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) (Tak et al., 2022): XLS-R + AASIST code and LA checkpoint, via [SpeechAntiSpoofingBenchmarks/W2V2-AASIST](https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST) | published detector; starting point of our fine-tuned model (the AASIST back end in `hearsay/tak_aasist.py` is copied from it) | MIT |
| XLSR-SLS (Zhang, Wen and Hu, ACM MM 2024), via [SpeechAntiSpoofingBenchmarks/XLSR-SLS](https://huggingface.co/SpeechAntiSpoofingBenchmarks/XLSR-SLS) | published detector | MIT (per the model card) |
| [XLS-R 300M](https://huggingface.co/facebook/wav2vec2-xls-r-300m) (Meta) | architecture config for the fine-tuned model | Apache-2.0 |
| [WavLM-Large](https://huggingface.co/microsoft/wavlm-large) (Microsoft) | frozen embedding model under our head | the card points to the [UniSpeech licence](https://github.com/microsoft/UniSpeech/blob/main/LICENSE), which is CC BY-SA 3.0 (we checked; it is not MIT) |
| [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | speaker-consistency expert | Apache-2.0 |
| [Whisper small.en](https://huggingface.co/openai/whisper-small.en) (OpenAI) | transcript expert; transcripts for our data analysis | Apache-2.0 (Hugging Face card) |
| [AASIST](https://github.com/clovaai/aasist) (NAVER / clovaai) | our first baseline (v0) and pseudo-labels for data analysis | MIT |
| ASVspoof 5 evaluation package, as modified by the organizers | reference for `hearsay/metrics.py` | no licence file in the package; we reimplemented the calculation and only compare against it |

Python dependencies keep their own licences, and `praat-parselmouth` (Praat) is GPL-3.0. Our macOS TTS clips were generated locally with the built-in `say` command and are not redistributed. Our code is MIT licensed (see [LICENSE](LICENSE)).

Team dh squad. We used AI coding assistants during development, as the organizers' tools slide suggests. Every number here comes from our own runs.
