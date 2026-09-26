"""Where HEARSAY finds its model files: one folder, one environment variable.

Every model file is looked up under a single folder that has the same layout as ``work/models/`` on the
laptop the models were trained on:

    HEARSAY_MODELS=/some/folder     # the Docker image sets /app/models
    (unset)                         # <repo>/work/models

    from hearsay.paths import model_path
    model_path("xlsr-sls", "xlsr-sls.onnx")     # -> $HEARSAY_MODELS/xlsr-sls/xlsr-sls.onnx

``python -m hearsay.paths [--meta fusion_meta.json]`` prints the folder in use and exits with status 1
when a file that the fusion needs is missing. The Docker build runs it, so an incomplete ``models/``
folder stops the build with a clear message instead of failing an hour into scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_VAR = "HEARSAY_MODELS"

_WAVLM = ("wavlm-large/config.json", "wavlm-large/preprocessor_config.json", "wavlm-large/pytorch_model.bin")

# Files each fusion member needs at inference, relative to models_dir() (the artifacts listed in
# hearsay.pipeline.DETECTORS, plus WavLM's preprocessor config). Public weights are downloaded by
# scripts/fetch_models.sh; our own trained artifacts are copied by scripts/export_models.sh.
REQUIRED = {
    "w2v2_aasist": ("w2v2-aasist/w2v2-aasist.onnx",),
    "xlsr_sls": ("xlsr-sls/xlsr-sls.onnx",),
    # Fine-tuned XLS-R + AASIST: the network is built from the XLS-R config and every weight comes from our
    # checkpoint, so the official LA_model.pth is not needed here (only load_la_model() reads it).
    "tak_ft": ("tak_ft/best.pt", "xls-r-300m/config.json"),
    "tak_head": ("tak_head/best.pt", "xls-r-300m/config.json"),   # same network, only the back end re-trained
    "wavlm_head": _WAVLM + ("wavlm_head.npz",),
    "wavlm_selftrain": _WAVLM + ("selftrain_base.npz",),
    "forensic_gbm": ("forensic_gbm.txt", "forensic_gbm_cols.json"),
}


def models_dir() -> Path:
    """$HEARSAY_MODELS if set, else <repo>/work/models. Read on every call, so the variable can also be
    set after import."""
    env = os.environ.get(ENV_VAR, "").strip()
    return Path(env).expanduser() if env else ROOT / "work" / "models"


def model_path(*parts: str) -> Path:
    """Path of a model file or folder under models_dir(), e.g. model_path("tak_ft", "best.pt")."""
    return models_dir().joinpath(*parts)


def missing_files(models) -> dict:
    """{model key: [missing paths relative to models_dir()]} for the given fusion members.

    Keys without an entry in REQUIRED are not checked (they are reported by main())."""
    root = models_dir()
    out = {}
    for m in models:
        gone = [f for f in REQUIRED.get(m, ()) if not (root / f).is_file()]
        if gone:
            out[m] = gone
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hearsay.paths",
                                 description="Show the model folder in use and check that the files a fusion needs exist.")
    ap.add_argument("--meta", help="fusion_meta.json whose 'models' list should be checked")
    ap.add_argument("--models", nargs="+", help="model keys to check (default: the --meta list, else every known key)")
    a = ap.parse_args(argv)
    root = models_dir()
    print(f"{ENV_VAR} = {root}" + ("" if os.environ.get(ENV_VAR, "").strip() else "  (default; set HEARSAY_MODELS to change)"))
    models = a.models
    if models is None and a.meta:
        meta = Path(a.meta)
        if not meta.is_file():
            print(f"MISSING  fusion meta {meta}: run scripts/export_models.sh (see docs/DOCKER.md)", file=sys.stderr)
            return 1
        models = list(json.loads(meta.read_text())["models"])
    if models is None:
        models = list(REQUIRED)
    missing = missing_files(models)
    for m in models:
        if m not in REQUIRED:
            print(f"  ?        {m}: no file list in hearsay/paths.py, not checked")
        elif m in missing:
            print(f"  MISSING  {m}: {', '.join(missing[m])}")
        else:
            print(f"  ok       {m}")
    if missing:
        print("Public weights come from scripts/fetch_models.sh, our trained artifacts from "
              "scripts/export_models.sh (see docs/DOCKER.md).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
