# Running HEARSAY in Docker

The image runs our complete inference pipeline (`python -m hearsay.cli predict`) on CPU. It reads a folder of
audio files and writes the submission TSV: `filename<TAB>cm-score`, where 1.0 means synthetic. Every model
file is inside the image, so the container needs no network access and no configuration.

## Build and run (judges start here)

```bash
docker build -t hearsay-dhsquad .
mkdir -p out
docker run --rm -v /path/to/HackGTHearsayTesting:/data -v "$PWD/out:/out" hearsay-dhsquad
```

`/path/to/HackGTHearsayTesting` is the organizers' test folder: the 1,671 WAVs plus
`HGT_Hearsay_score_template.csv`. The container can mount it read-only (`-v ...:/data:ro`). The run writes:

| File | Contents |
|---|---|
| `out/dh_squad_predictions.tsv` | the submission: rows in template order, every cm-score unique and in [0, 1]; the writer checks names, order, range and uniqueness against the template |
| `out/dh_squad_predictions_scores.json` | every detector's log-odds for every file, the fused and calibrated log-odds, the probability, plus per-model timing |

The default command is:

```bash
python -m hearsay.cli predict /data --out /out/dh_squad_predictions.tsv \
    --meta /app/models/fusion_meta.json --template /data/HGT_Hearsay_score_template.csv
```

To score a folder that has no template, write out the command without `--template`. Rows then come out
sorted by file name:

```bash
docker run --rm -v /path/to/audio:/data:ro -v "$PWD/out:/out" hearsay-dhsquad \
    predict /data --out /out/predictions.tsv --meta /app/models/fusion_meta.json
```

`fusion_meta.json` is the only configuration. It lists which detectors to run, and it holds the dev-set
z-scoring statistics, fusion weights, Platt calibration and prior shift. The pipeline runs exactly the detectors
it names, so a new fusion needs no code change and no image change beyond that file.

## Where the weights come from (nothing to do for judges)

Two kinds of weights go into the image:

1. **Public pretrained weights (3.9 GB).** `scripts/fetch_models.sh` downloads them during the build, pinned
   to exact Hugging Face commits. It checks each file's byte size and SHA-256 against
   `scripts/public_models.tsv`. We hashed the exact files our submission was produced with, so the image gets
   byte-identical weights.

   | Under `/app/models/` | Size | Source (pinned commit) | Used by |
   |---|---|---|---|
   | `w2v2-aasist/w2v2-aasist.onnx` | 1.26 GB | SpeechAntiSpoofingBenchmarks/W2V2-AASIST @196128e | `w2v2_aasist` |
   | `xlsr-sls/xlsr-sls.onnx` | 1.36 GB | SpeechAntiSpoofingBenchmarks/XLSR-SLS @6b99ebc | `xlsr_sls` |
   | `wavlm-large/` (config, preprocessor config, weights) | 1.26 GB | microsoft/wavlm-large @c1423ed | `wavlm_head`, `wavlm_selftrain` |
   | `xls-r-300m/config.json` | 2 KB | facebook/wav2vec2-xls-r-300m @1a640f3 | `tak_ft` (architecture) |
   | `w2v2-aasist/LA_model.pth` | 1.27 GB | SpeechAntiSpoofingBenchmarks/W2V2-AASIST @196128e | fine-tuning scripts only; not downloaded by default (`--build-arg FETCH_LA=1` adds it) |
   | `spkrec-ecapa-voxceleb/` | 90 MB | speechbrain/spkrec-ecapa-voxceleb @0f99f2d | speaker-consistency expert, reports only |

2. **Our own trained artifacts (589 MB archive).** They come from our GitHub release
   [`v1.0`](https://github.com/MeagerPotato/hearsay-dh-squad/releases/tag/v1.0)
   (`hearsay-models-v3.tar.gz`, SHA-256 `3565772ef137b1091aa944fbc461348aac4327176899f128ee9e7e0fdfaabc40`).
   The Dockerfile's `OWN_MODELS_URL` / `OWN_MODELS_SHA256` defaults point at it, so a plain `docker build`
   downloads it, checks the SHA-256 and unpacks it into `/app/models`. On the machine that trained the models,
   `scripts/export_models.sh` fills `models/` directly and the build uses that copy instead:

   ```bash
   scripts/export_models.sh --version v3      # the fusion we submit
   ```

   This produces `fusion_meta.json`, `wavlm_head.npz`, `forensic_gbm.txt` and `forensic_gbm_cols.json`, and
   `test_transcripts.json` (report text only). When the fusion uses them, it also produces `tak_ft/best.pt` or
   `tak_head/best.pt` (converted to float16: 1.26 GB becomes 0.63 GB) and `selftrain_base.npz`. The fusion must
   have been fitted on the model files that are exported: v2a–v2g were fitted on our first WavLM head, which
   was replaced when the head was retrained for v2h, so export v2h or a later version. It writes `MANIFEST.json`
   (provenance) and `SHA256SUMS`, and the build verifies every file against `SHA256SUMS`. Re-running the script
   is safe. It copies only what changed, and it removes artifacts that a different fusion no longer needs. It
   refuses to export `tak_ft` while `scripts/finetune_tak.py` is still writing the checkpoint.

How the archive was made (team step, already done for v3): `scripts/export_models.sh --version v3 --tar`
writes `work/dist/hearsay-models-v3.tar.gz`, which we uploaded as the release asset above. To build from a
different archive, override both build arguments:

```bash
docker build --build-arg OWN_MODELS_URL=https://github.com/<owner>/<repo>/releases/download/<tag>/hearsay-models-<v>.tar.gz \
             --build-arg OWN_MODELS_SHA256=<sha256 printed by export_models.sh> -t hearsay-dhsquad .
```

If `models/` is incomplete, the build fails at its last step with a message naming the missing files. The build
runs `python -m hearsay.paths --meta /app/models/fusion_meta.json` for this check, and you can run the same
command locally.

## Other ways to use the image

**One arbitrary file, quick forensic explanation (seconds).** This takes any format ffmpeg reads (WAV, MP3,
M4A, FLAC, ...). The agentic orchestrator runs the forensic experts, prints the findings and routing trace,
and writes an HTML report. It uses no deep detector, so the verdict comes from the transparent expert vote.
`--skip transcript` avoids a Whisper error for files that have no cached transcript:

```bash
docker run --rm -v /path/to/folder:/in:ro -v "$PWD/out:/out" hearsay-dhsquad \
    explain /in/suspect.mp3 --html /out/suspect.html --skip transcript
```

**One file with the full submitted ensemble and an HTML report.** `predict` also accepts a single file. The
report's verdict is exactly the TSV value:

```bash
docker run --rm -v /path/to/folder:/in:ro -v "$PWD/out:/out" hearsay-dhsquad \
    predict /in/suspect.mp3 --out /out/suspect.tsv --meta /app/models/fusion_meta.json --reports /out/suspect_report
```

**HTML reports for the whole test set.** Add `--reports /out/reports` to the default command, then open
`out/reports/index.html`. The orchestrator runs on every file with the ensemble as its deep evidence, which adds
roughly 10 to 25 minutes on CPU:

```bash
docker run --rm -v /path/to/HackGTHearsayTesting:/data:ro -v "$PWD/out:/out" hearsay-dhsquad \
    predict /data --out /out/dh_squad_predictions.tsv --meta /app/models/fusion_meta.json \
    --template /data/HGT_Hearsay_score_template.csv --reports /out/reports
```

**Quicker repeat runs.** Keep the score cache outside the container with `--cache-dir /out/cache`. The cache
is keyed by fingerprints of the audio and model files, so a second run only computes what changed.

**Look around.** These commands list the forensic experts, print the routing policy, or open a shell:

```bash
docker run --rm hearsay-dhsquad experts
docker run --rm hearsay-dhsquad policy
docker run --rm -it --entrypoint bash hearsay-dhsquad
```

## Time, memory and disk

Measured on our laptop (Apple M5, Docker Desktop limited to 6 CPUs and 8 GB), building from a fresh clone of
this repository with a plain `docker build`, native arm64:

| | Measured |
|---|---|
| Build from a fresh clone | **11.5 min**: public weights 2.5 min (3.9 GB at about 26 MB/s), Python packages 6.5 min (includes compiling `praat-parselmouth`, which has no Linux arm64 wheel), our release archive plus checksum checks 0.5 min |
| Build, x86-64 | not measured; expect less, because every wheel exists prebuilt for x86-64 |
| Scoring, submitted fusion v3 (4 detectors), CPU | **2.0 s per clip** with 6 threads over the whole test set (XLS-R + AASIST 0.45 s, XLSR-SLS 0.46 s, re-trained XLS-R + AASIST 0.47 s, WavLM-Large 0.60 s) plus about 11 s of model loading |
| Scoring the full test set (1,671 clips) | **55.4 min** from `docker run` to exit (3,324 s) |
| Memory | one large model is loaded at a time; 8 GB for Docker was enough (4.2 GB in use when we checked during WavLM-Large scoring) |
| Disk | 6.7 GB of files inside the image (4.3 GB model weights, 1.8 GB Python packages, 0.6 GB system); Docker Desktop's image store reports 11.7 GB because it also counts the compressed layers. Keep about 20 GB free for the build |

`predict` prints the real per-model time (load, scoring, seconds per file) at the end of every run, and
`*_scores.json` stores it.

## Does the container reproduce our submitted scores?

Everything that goes into the image is pinned:

- **Python packages:** `requirements.txt` pins our direct dependencies, and `constraints.txt` pins every
  transitive one at the version we developed with. We ran pip's resolver for Python 3.12 on linux x86_64
  and aarch64: it installs 80 packages, all at exactly these versions, with CPU-only torch and no CUDA
  packages.
- **Public weights:** exact Hugging Face commits plus a SHA-256 for every file.
- **Our artifacts:** `SHA256SUMS`, which the build verifies.
- **The base image:** a tag by default. Pass `BASE_IMAGE=python:3.12-slim@sha256:...` to pin its digest.

It runs the same code and the same fusion file on hash-verified copies of the same weights, so the ranking,
which is all minDCF uses, should be essentially identical. The scores are not guaranteed to match bit for
bit, for these known reasons:

- WavLM-Large ran in float16 on the Mac GPU when we developed the head, and it runs in float32 on CPU in
  Docker, so `wavlm_head` log-odds differ slightly (self-training, which v3 does not use, would amplify this).
- `tak_ft` and `tak_head` are stored as float16 (they load back into the float32 network). This rounds each weight by at most
  about 0.05%. Use `scripts/export_models.sh --keep-fp32` if bit-exact weights matter more than 0.63 GB.
- CPU math libraries differ between x86 and ARM, which changes results around the fifth significant digit.

The fusion arithmetic itself is exact: fed our development scores instead of live models (a copy of
`work/scores/test/*.json` passed as `--cache-dir`), `predict` reproduces `submissions/v2h/dh_squad_predictions.tsv`
byte for byte. That needs the full-precision coefficients (`coef_full`) that `make_submission.py` records from
v2h on; with only the rounded ones, scores differ by up to about 4e-6.

**Measured.** We built the image from a fresh clone of this repository and scored 30 test clips chosen across
the whole score range (confident fakes, confident reals, borderline and very quiet clips), then compared with
the run that produced our submission (`python scripts/compare_runs.py out/..._scores.json
submissions/final_pipeline/dh_squad_predictions_scores.json`):

| | Largest difference | Spearman rank correlation |
|---|---|---|
| `w2v2_aasist`, `xlsr_sls` (ONNX) | 1.2e-4 log-odds | 1.000 |
| `tak_head` (float16-stored weights, CPU vs Mac GPU) | 0.057 log-odds | 1.000 |
| `wavlm_head` (float32 CPU vs float16 Mac GPU) | 0.089 log-odds | 1.000 |
| **cm-score** | **0.0089** | **0.9996** |

No clip changed its verdict (cm-score above or below 0.5).

**Measured on the whole test set.** We then ran the same image on all 1,671 test clips (the `docker run` command
above, 55 minutes) and compared the same way:

| | Largest difference | Spearman rank correlation |
|---|---|---|
| `w2v2_aasist` (ONNX) | 2.0e-3 log-odds | 1.000000 |
| `xlsr_sls` (ONNX) | 6.6e-4 log-odds | 1.000000 |
| `tak_head` (float16-stored weights, CPU vs Mac GPU) | 0.25 log-odds | 0.99989 |
| `wavlm_head` (float32 CPU vs float16 Mac GPU) | 0.057 log-odds | 0.999997 |
| **cm-score** | **0.0097** | **0.999999** |

No clip changed its verdict, all 1,671 scores are distinct, and the 501 clips ranked most synthetic (the top
30%, the expected share of fakes) are the same set in both runs. No clip moved more than 9 places in the
ranking of 1,671 (Kendall's tau 0.9995). minDCF depends only on that ranking, so the image reproduces our
submission's score for practical purposes. That run's TSV, per-model scores and log are in
[`submissions/docker_full_run/`](../submissions/docker_full_run/).

To check after a Docker run, compare the result with the TSV we submitted:

```bash
python - <<'EOF'
import csv
import numpy as np
read = lambda p: {r[0]: float(r[1]) for r in list(csv.reader(open(p), delimiter="\t"))[1:]}
a, b = read("submissions/v3/dh_squad_predictions.tsv"), read("out/dh_squad_predictions.tsv")
names = sorted(a)
x, y = np.array([a[n] for n in names]), np.array([b[n] for n in names])
rank = lambda v: np.argsort(np.argsort(v))
print("max |difference|", np.abs(x - y).max(), " rank correlation", np.corrcoef(rank(x), rank(y))[0, 1])
EOF
```

## Build options and settings

| Build argument | Default | Effect |
|---|---|---|
| `OWN_MODELS_URL`, `OWN_MODELS_SHA256` | our v1.0 release archive and its SHA-256 | where our trained artifacts come from when `models/` has no complete export |
| `FETCH_LA` | 0 | 1 adds the official LA checkpoint (1.27 GB); inference does not read it, only `scripts/score_tak_ft.py` does |
| `FETCH_ECAPA` | 1 | 0 leaves out the speaker model (90 MB); reports then lack speaker-consistency evidence |
| `WITH_WHISPER` | 0 | 1 adds Whisper small.en (1 GB) for `predict --reports --asr` on files without a cached transcript |
| `HF_ENDPOINT` | https://huggingface.co | a Hugging Face mirror |
| `TORCH_INDEX_URL` | https://download.pytorch.org/whl/cpu | a mirror of PyTorch's CPU wheel index |
| `BASE_IMAGE` | python:3.12-slim | e.g. `python:3.12-slim@sha256:<digest>` to pin the base image exactly |

Inside the image, `HEARSAY_MODELS=/app/models` is the one folder the code reads model files from (locally it
defaults to `work/models/`). `/app/work/models` is a link to it, and HF/Transformers offline mode is on.
`predict` also takes `--threads N` (CPU threads per model; default `OMP_NUM_THREADS`, else all cores up to 8),
`--workers N` (processes for the forensic features), `--limit N` (first N files, for a smoke test) and
`--no-cache`.

## Troubleshooting

- **Apple silicon.** A plain `docker build` makes a native linux/arm64 image. It compiles `praat-parselmouth`
  once, and it runs fastest. To build and test the linux/amd64 image that x86 judges get, use
  `docker build --platform linux/amd64 -t hearsay-dhsquad .` and `docker run --platform linux/amd64 ...`.
  This image runs under emulation. Turn on "Use Rosetta for x86_64/amd64 emulation" in Docker Desktop's
  settings, and expect it to be several times slower.
- **`exec format error`, or a warning that the image platform does not match the host.** The image was
  built for the other architecture. Rebuild it, or pass the same `--platform` to `docker run`.
- **Exit code 137, or `Killed`.** The container ran out of memory. Raise Docker Desktop's memory limit to
  8 GB or more, or lower the load with `--threads 2 --workers 1`.
- **`Permission denied` writing `/out` on Linux.** Create the folder first (`mkdir -p out`). The container
  runs as root, so the output files are owned by root. Add `--user "$(id -u):$(id -g)"` to get your own
  ownership; the image supports that.
- **`template file(s) are not in the input`.** The template lists files that are missing from `/data`.
  Mount the complete test folder, or leave out `--template`.
- **`no audio files found`.** Check the `-v` path. It must be absolute (on Linux and macOS, `"$PWD/..."`
  works).
- **The build stops at `python -m hearsay.paths`.** `models/` is missing files that the fusion needs. Run
  `scripts/export_models.sh`, or pass `OWN_MODELS_URL`.
- **The build fails while downloading.** The script retries with back-off and resumes partial files. Behind a
  proxy, pass `--build-arg HTTPS_PROXY=...`. If Hugging Face is slow or blocked, use
  `--build-arg HF_ENDPOINT=<mirror>`.
- **Legacy builder on arm64.** The compiler step needs BuildKit, which is the default in Docker Desktop and
  Docker Engine 23 and later. Otherwise build with `DOCKER_BUILDKIT=1`.
- **`praat-parselmouth` fails to compile (native arm64 only).** Build on the older Debian base with
  `--build-arg BASE_IMAGE=python:3.12-slim-bookworm`, or build the linux/amd64 image, which uses a prebuilt
  wheel.
- **A different fusion.** Re-run `scripts/export_models.sh --version <v>` and rebuild. Only the small
  `models/` layer and the layers after it are rebuilt; the 5 GB weight layer stays cached.
