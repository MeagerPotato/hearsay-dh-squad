# syntax=docker/dockerfile:1
#
# HEARSAY (team dh squad): scores audio clips for synthetic speech. CPU-only, and fully offline at run time.
#
#   docker build -t hearsay-dhsquad .           # fetches public weights + our release archive automatically
#   (on the machine that trained the models, scripts/export_models.sh --version v3 fills models/ instead)
#   docker run --rm -v /path/to/HackGTHearsayTesting:/data -v "$PWD/out:/out" hearsay-dhsquad
#
# The default command writes /out/dh_squad_predictions.tsv (plus *_scores.json). Details, runtimes and
# troubleshooting: docs/DOCKER.md.

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# Set by BuildKit (amd64 or arm64). Only used to add a compiler on arm64, see step 2.
ARG TARGETARCH

LABEL org.opencontainers.image.title="hearsay-dhsquad" \
      org.opencontainers.image.description="HEARSAY audio-deepfake detection (NSA challenge, HackGT 13, team dh squad)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HEARSAY_MODELS=/app/models \
    HEARSAY_ECAPA_DIR=/app/models/spkrec-ecapa-voxceleb \
    HEARSAY_TRANSCRIPTS=/app/models/test_transcripts.json \
    HF_HOME=/app/models/hf \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    XDG_CACHE_HOME=/tmp/cache

# ffmpeg: decodes any audio container; libsndfile1: soundfile; exiftool: container forensics;
# libgomp1: LightGBM's OpenMP runtime; curl: downloads the public weights during the build.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libimage-exiftool-perl libgomp1 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Public pretrained weights (3.9 GB), pinned to Hugging Face commits, byte size and SHA-256 verified
#    (scripts/public_models.tsv). This is its own early layer, so code changes never download them again.
#    FETCH_LA=0 (the default) leaves out the official LA checkpoint (1.27 GB), which inference does not read;
#    FETCH_ECAPA=0 leaves out the speaker model (90 MB) used only for report evidence.
ARG HF_ENDPOINT=https://huggingface.co
ARG FETCH_LA=0
ARG FETCH_ECAPA=1
COPY scripts/fetch_models.sh scripts/public_models.tsv /app/scripts/
RUN HF_ENDPOINT="${HF_ENDPOINT}" FETCH_LA="${FETCH_LA}" FETCH_ECAPA="${FETCH_ECAPA}" \
        bash /app/scripts/fetch_models.sh /app/models

# 2) Python packages, versions from requirements.txt. torch and torchaudio come from PyTorch's CPU-only
#    index, because the default Linux wheel pulls in several GB of CUDA libraries. constraints.txt pins the
#    transitive dependencies to the versions we developed with. praat-parselmouth has no Linux arm64
#    wheel, so a native arm64 build compiles it and then removes the compiler again (-fpermissive: GCC 14
#    turns some old-C warnings in Praat's sources into errors).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY requirements.txt constraints.txt /app/
RUN set -eux; \
    if [ "${TARGETARCH:-amd64}" = "arm64" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends build-essential cmake; \
        export CFLAGS="-fpermissive"; \
    fi; \
    torch_pin="$(grep -E '^torch==' requirements.txt)"; \
    torchaudio_pin="$(grep -E '^torchaudio==' requirements.txt)"; \
    pip install --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple \
        -c constraints.txt "${torch_pin}+cpu" "${torchaudio_pin}+cpu"; \
    grep -vE '^(torch|torchaudio)==' requirements.txt > /tmp/requirements-rest.txt; \
    pip install -r /tmp/requirements-rest.txt -c constraints.txt; \
    rm -f /tmp/requirements-rest.txt; \
    if [ "${TARGETARCH:-amd64}" = "arm64" ]; then \
        apt-get purge -y --auto-remove build-essential cmake; \
        rm -rf /var/lib/apt/lists/*; \
    fi; \
    python -c "import torch; assert torch.version.cuda is None, 'a CUDA build of torch was installed'; print('torch', torch.__version__)"; \
    python -c "import numpy, scipy, sklearn, soundfile, librosa, onnxruntime, transformers, lightgbm, parselmouth, typer, jinja2, matplotlib, webrtcvad, mutagen"; \
    python -c "import torchaudio, speechbrain" \
        || echo "WARNING: torchaudio/speechbrain do not import; only the speaker-consistency expert is affected"

# 3) Optional: Whisper small.en (1 GB), so the transcript expert can transcribe files it has no cached
#    transcript for (predict --reports --asr). Off by default: no cm-score uses it.
ARG WITH_WHISPER=0
RUN if [ "${WITH_WHISPER}" = "1" ]; then \
        python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/whisper-small.en', revision='e8727524f962ee844a7319d92be39ac1bd25655a', allow_patterns=['*.json', '*.txt', 'model.safetensors'])" \
     && mkdir -p /app/models/hf/hub/models--openai--whisper-small.en/refs \
     && printf '%s' e8727524f962ee844a7319d92be39ac1bd25655a > /app/models/hf/hub/models--openai--whisper-small.en/refs/main; \
    fi

# 4) Our own trained artifacts and the chosen fusion, exported into models/ by scripts/export_models.sh.
#    Or build from a packed archive (scripts/export_models.sh --tar), e.g. a GitHub release asset:
#      docker build --build-arg OWN_MODELS_URL=https://.../hearsay-models-<version>.tar.gz \
#                   --build-arg OWN_MODELS_SHA256=<sha256> -t hearsay-dhsquad .
#    A fresh clone only has models/README.md and models/fusion_meta.json, so by default the build downloads
#    our release archive (SHA-256 checked). If models/ already holds a complete, checksum-valid export (the
#    machine that trained the models), the download is skipped.
COPY models/ /app/models/
ARG OWN_MODELS_URL="https://github.com/MeagerPotato/hearsay-dh-squad/releases/download/v1.0/hearsay-models-v3.tar.gz"
ARG OWN_MODELS_SHA256="3565772ef137b1091aa944fbc461348aac4327176899f128ee9e7e0fdfaabc40"
RUN if [ -f /app/models/SHA256SUMS ] && (cd /app/models && sha256sum --check --quiet SHA256SUMS); then \
        echo "trained artifacts found in the build context (models/); skipping the release download"; \
    elif [ -n "${OWN_MODELS_URL}" ]; then \
        OWN_MODELS_SHA256="${OWN_MODELS_SHA256}" bash /app/scripts/fetch_models.sh --own-only --own "${OWN_MODELS_URL}" /app/models; \
    else \
        echo "ERROR: no trained artifacts: run scripts/export_models.sh or pass --build-arg OWN_MODELS_URL=..." >&2; exit 1; \
    fi

# 5) Code last: editing it rebuilds only these small layers.
COPY hearsay/ /app/hearsay/
COPY scripts/ /app/scripts/
COPY docs/ /app/docs/

# Code that looks in <repo>/work/models finds the same files. /app/work (default score cache) and
# /app/models/hf are world-writable, so `docker run --user ...` works as well.
RUN mkdir -p /app/work /app/models/hf /data /out \
 && ln -sfn /app/models /app/work/models \
 && chmod 1777 /app/work /app/models/hf /out

# Nothing is downloaded at run time.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Fail here, not an hour into scoring, if models/ is incomplete or corrupted or the CLI does not load.
RUN if [ -f /app/models/SHA256SUMS ]; then \
        cd /app/models && sha256sum --check --quiet SHA256SUMS; \
    else \
        echo "WARNING: models/SHA256SUMS is missing, so our trained artifacts were not verified"; \
    fi \
 && python -m hearsay.paths --meta /app/models/fusion_meta.json \
 && python -m hearsay.cli predict --help > /dev/null \
 && python -m hearsay.cli experts > /dev/null

ENTRYPOINT ["python", "-m", "hearsay.cli"]
CMD ["predict", "/data", "--out", "/out/dh_squad_predictions.tsv", "--meta", "/app/models/fusion_meta.json", "--template", "/data/HGT_Hearsay_score_template.csv"]
