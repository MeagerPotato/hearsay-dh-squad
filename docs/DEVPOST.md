# Devpost draft: HEARSAY by dh squad

> Paste each section into the matching Devpost field. Edit freely. The judges asked for documentation "in your
> own words", so rewrite anything that doesn't sound like you. Numbers match README.md and docs/FINDINGS.md.

## Project name
HEARSAY: a synthetic-speech detector that shows its evidence

## Tagline
Ranks audio clips by how likely they are to be synthetic, and writes a forensic report that says why, tuned so
analysts are rarely sent chasing real recordings.

## Inspiration
A cloned voice needs only a few seconds of reference audio, and the NSA brief framed the cost clearly: every false
alarm on a real recording wastes an analyst's time and erodes trust, so it is penalised four times more than a
miss. We wanted a system that makes few false alarms, holds up on audio that has been re-encoded and buried in
noise, and can explain each call to a human.

## What it does
- Takes any audio file ffmpeg can read and returns a probability that the speech is synthetic (1.0 = synthetic).
- Fuses four deep anti-spoofing detectors: two published XLS-R models, the same XLS-R + AASIST model with its
  back end re-trained by us on noise-matched audio, and our own classifier on WavLM-Large features. The fusion is
  *condition-aware*: it measures each clip's background noise floor and trusts different detectors on very quiet
  clips, where the published models fail most.
- Runs up to 16 interpretable forensic experts in 9 categories: container and metadata (including
  modified/accessed/created timestamps), spectral and vocoder artefacts, prosody (pitch, pauses, breaths, speaking
  rate), acoustic environment and mains hum (ENF), compression history, speaker-embedding consistency, splices, and
  the transcript. An agentic router chooses which to run per file (cheap checks always; slower ones only when the
  verdict is uncertain, the evidence conflicts, or the file is lossy or long), so it uses ~39% of the compute of
  running everything, with identical scores.
- Writes a self-contained HTML report per file: verdict, chain of custody, spectrogram with suspected seams and
  codec cut-offs, pitch contour, evidence table with each finding's direction and strength, the routing trace, and
  "evidence that had no effect".

## How we built it
1. **We studied the test set first.** Whisper transcripts showed the test clips are VCTK sentences (the corpus
   behind ASVspoof 2019), not the DiffSSD/LJSpeech material provided for training. From the clip lengths (exact
   multiples of 512 samples at 22.05 kHz) we reverse-engineered the organizers' preprocessing (librosa load → silence
   trim → ≥3 s → 16 kHz → peak-normalise → ffmpeg) and rebuilt it in code. Comparing its output with the test set
   showed the organizers had also **added background noise** (median noise floor −44 vs −61 dBFS).
2. **We built test-matched training data.** ASVspoof 2019 LA *train/dev only* (we never extracted the eval
   partition or its keys), DiffSSD, LibriSpeech real voices, and 827 fakes we generated ourselves with macOS voices,
   all passed through our pipeline replica. We then added real-world MUSAN noise so each clip's noise floor follows
   the test set's distribution, applied the same way to real and fake clips, plus ~5% codec laundering (MP3, AAC,
   Opus, G.726, G.722, µ-law).
3. **We measured what hurts.** The best published detector scores minDCF 0.020 on clean pipeline audio but 0.140
   once test-matched noise is added (organizers' scoring settings, P_spoof = 0.3). Almost all errors come from quiet clips.
4. **We trained and fused.** We trained a WavLM-Large layer probe and head, fine-tuned the XLS-R + AASIST model on
   the Mac GPU, and fitted a logistic-regression fusion with false alarms weighted 4×, on a dev set that includes
   generator families no component was trained on.
5. **We built the forensic layer and orchestrator** as a registry of experts sharing one evidence contract, plus a
   declarative routing policy and HTML reports. Docker and a single `hearsay predict` command reproduce the
   submission byte for byte.

## Challenges we ran into
- The provided training data looked nothing like the test set. A model trained on it mostly learns shortcuts
  (speaker, text, sample rate, mains hum).
- The official scorer treats higher scores as *bona fide*, while the submission format asks for 1.0 = synthetic.
  Run unmodified, a perfect detector scores the worst possible minDCF. We flagged it to the organizers.
- Everything ran on a 16 GB laptop. One fine-tuning run pushed the machine into heavy swap, so we added hard GPU
  memory caps and a watchdog that stops any job if free memory drops too low.
- The best public detectors turned out to be trained on ASVspoof *evaluation* data, which likely overlaps this test
  set. We chose not to use them.

## Accomplishments that we're proud of
- Cross-validated minDCF of **0.076** on test-matched audio with seen attack types and **0.153** on synthesis
  families no component was trained on, versus 0.140 and 0.179 for the strongest published detector under the same
  conditions (organizers' scoring settings: P_spoof = 0.3, C_fa = 4). **[Add the organizers' test-set feedback score here if you receive it.]**
- A clear, honest account of what did *not* work: denoising before detection (made it worse), full fine-tuning
  (forgot unseen attacks), transductive self-training (helped in simulation, hurt on the full mix), and putting the
  glass-box model into the score (no gain, so we kept it for explanations only).
- Reports an analyst can actually read, and a router that spends compute only where the evidence is unclear.

## What we learned
- Look at the test data before training anything: the biggest gains came from understanding its preprocessing and
  noise, not from bigger models.
- Anti-spoofing models trained on one studio corpus treat every unfamiliar real voice as fake (97.5% of genuine
  LJSpeech flagged). Robustness to *real* variety matters as much as coverage of attacks.
- For a rank-only metric with asymmetric costs, you have to measure at the operating point that matters (few false
  alarms), not just accuracy or EER.

## What's next
- Fine-tune with a broader set of modern synthesis systems (e.g., ElevenLabs, Gemini, Qwen-TTS) while keeping the
  pretrained representation frozen or regularised, so noise robustness does not cost generalisation.
- Calibrate per acoustic condition (telephony, codecs, reverberant rooms) with dedicated dev sets.
- Add partial-spoof (spliced) detection with frame-level scores.

## Built with
python, pytorch, onnxruntime, transformers (XLS-R, WavLM, Whisper), speechbrain (ECAPA-TDNN), librosa, praat-parselmouth,
scikit-learn, lightgbm, ffmpeg, exiftool, sox, typer, jinja2, matplotlib, docker

## Links
- GitHub: https://github.com/MeagerPotato/hearsay-dh-squad
- Submission file: `dh_squad_predictions.tsv` (sent to @PuzzleMaster on Discord)
