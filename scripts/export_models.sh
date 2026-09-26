#!/usr/bin/env bash
# Put our own trained artifacts and the chosen fusion into models/, the folder the Docker image copies
# to /app/models. Public pretrained weights are NOT copied: the image downloads those itself
# (scripts/fetch_models.sh), which keeps models/ small enough to ship as one archive.
#
#   scripts/export_models.sh --version v2c [options]
#
# Sources: work/models/ (trained artifacts) and submissions/<version>/fusion_meta.json (the fusion).
# models/ gets the same layout as work/models/, so the code finds everything through HEARSAY_MODELS:
#   fusion_meta.json        which models are fused, their weights, z-scoring and calibration
#   tak_ft/best.pt          fine-tuned XLS-R + AASIST, stored as float16 (1.26 GB -> 0.63 GB); it loads back
#                           into the float32 network (--keep-fp32 copies it bit-exact instead)
#   tak_ft/best.json        its dev-set metrics (provenance)
#   tak_head/...            the same for the back-end-only fine-tune (key tak_head), when a fusion uses it
#   wavlm_head.npz          WavLM-Large head: chosen layers, scaler, logistic regression
#   selftrain_base.npz      base features for transductive self-training (only if the fusion uses it)
#   forensic_gbm.txt, forensic_gbm_cols.json   glass-box LightGBM over the forensic expert features
#   test_transcripts.json   cached Whisper transcripts of the test set (report text only)
#   MANIFEST.json, SHA256SUMS   provenance and checksums (the Docker build verifies SHA256SUMS)
#
# Options
#   --version V        use submissions/V/fusion_meta.json (default: the newest one, with a warning)
#   --all              also export trained artifacts that the chosen fusion does not use
#   --keep-fp32        copy tak_ft/best.pt unchanged (bit-exact scores, image +0.63 GB)
#   --no-transcripts   leave out test_transcripts.json
#   --tar              also pack models/ into work/dist/hearsay-models-V.tar.gz, e.g. for a GitHub release;
#                      `docker build --build-arg OWN_MODELS_URL=<url of that file>` then fetches it
#   --dry-run          show what would happen; change nothing
#   --force            export tak_ft even while scripts/finetune_tak.py is still running
#
# Idempotent: unchanged files are not copied again, the float16 conversion is redone only when the source
# checkpoint changes, and artifacts of an earlier export that are no longer needed are removed from models/.
# PYTHON (default .venv/bin/python, else python3) needs torch for the float16 step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$ROOT/work/models"
OUT="$ROOT/models"
TAK_KEYS="tak_ft tak_head"   # our XLS-R + AASIST checkpoints, work/models/<key>/best.pt
KNOWN="fusion_meta.json tak_ft/best.pt tak_ft/best.pt.source tak_ft/best.json tak_head/best.pt tak_head/best.pt.source tak_head/best.json wavlm_head.npz selftrain_base.npz forensic_gbm.txt forensic_gbm_cols.json test_transcripts.json"

log() { printf '[export_models] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }
filesize() { wc -c < "$1" | tr -d '[:space:]'; }
sha256_of() {
  if command -v sha256sum > /dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
human() { awk -v b="$1" 'BEGIN { split("B KB MB GB TB", u, " "); i = 1; while (b >= 1000 && i < 5) { b /= 1000; i++ } printf("%.1f %s", b, u[i]) }'; }
rel_to_root() { printf '%s' "${1#"$ROOT"/}"; }

version=""
export_all=0
keep_fp32=0
transcripts=1
make_tar=0
dry=0
force=0
while [ $# -gt 0 ]; do
  case "$1" in
    --version) [ $# -ge 2 ] || die "--version needs a value"; version="$2"; shift ;;
    --version=*) version="${1#--version=}" ;;
    --all) export_all=1 ;;
    --keep-fp32) keep_fp32=1 ;;
    --no-transcripts) transcripts=0 ;;
    --tar) make_tar=1 ;;
    --dry-run) dry=1 ;;
    --force) force=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
  shift
done

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="python3"; fi
fi

# ---- which fusion -----------------------------------------------------------------------------------
if [ -n "$version" ]; then
  meta="$ROOT/submissions/$version/fusion_meta.json"
  [ -f "$meta" ] || die "not found: $(rel_to_root "$meta")"
else
  meta=""
  for f in "$ROOT"/submissions/*/fusion_meta.json; do
    [ -f "$f" ] || continue
    if [ -z "$meta" ] || [ "$f" -nt "$meta" ]; then meta="$f"; fi
  done
  [ -n "$meta" ] || die "no submissions/*/fusion_meta.json found; run scripts/make_submission.py first"
  version="$(basename "$(dirname "$meta")")"
  log "WARNING: no --version given, using the newest fusion ($version); pass --version to be explicit"
fi
models="$("$PY" -c 'import json, sys; print(" ".join(json.load(open(sys.argv[1]))["models"]))' "$meta")" \
  || die "cannot read the models list from $(rel_to_root "$meta")"
log "fusion $version: $models"
[ "$dry" = 0 ] || log "dry run: nothing is written"

fused() { case " $models " in *" $1 "*) return 0 ;; esac; return 1; }
wanted() { fused "$1" || [ "$export_all" = 1 ]; }

exported=""
note() { exported="$exported $1"; }

# copy_art SRC REL REQUIRED: copy SRC to models/REL unless an identical copy is already there.
copy_art() {
  local src="$1" rel="$2" required="$3" dst="$OUT/$2"
  if [ ! -f "$src" ]; then
    if [ "$required" = 1 ]; then die "missing $(rel_to_root "$src") (fusion $version needs it)"; fi
    log "skip     $rel (no $(rel_to_root "$src"))"
    return 0
  fi
  note "$rel"
  if [ -f "$dst" ] && cmp -s "$src" "$dst"; then log "same     $rel"; return 0; fi
  log "copy     $rel <- $(rel_to_root "$src") ($(human "$(filesize "$src")"))"
  if [ "$dry" = 1 ]; then return 0; fi
  mkdir -p "$(dirname "$dst")"
  cp -p "$src" "$dst.tmp"
  mv -f "$dst.tmp" "$dst"
}

# export_tak KEY REQUIRED: fine-tuned checkpoint work/models/KEY/best.pt, float16 unless --keep-fp32.
export_tak() {
  local key="$1" required="$2"
  local src="$SRC/$key/best.pt" dst="$OUT/$key/best.pt" stamp="$OUT/$key/best.pt.source"
  local precision=fp16 want
  if [ ! -f "$src" ]; then
    if [ "$required" = 1 ]; then die "missing $(rel_to_root "$src") (fusion $version needs it)"; fi
    return 0
  fi
  if command -v pgrep > /dev/null 2>&1 && pgrep -f finetune_tak.py > /dev/null 2>&1; then
    if [ "$force" = 1 ] || [ "$dry" = 1 ]; then
      log "WARNING: scripts/finetune_tak.py is running and may still overwrite $(rel_to_root "$src")"
    else
      die "scripts/finetune_tak.py is still running and may overwrite $(rel_to_root "$src"); wait for it or pass --force"
    fi
  fi
  if [ "$keep_fp32" = 1 ]; then precision=fp32; fi
  copy_art "$SRC/$key/best.json" "$key/best.json" 0
  note "$key/best.pt"
  note "$key/best.pt.source"
  want="$(sha256_of "$src") $precision"
  if [ -f "$dst" ] && [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ]; then
    log "same     $key/best.pt ($precision)"
    return 0
  fi
  log "$precision     $key/best.pt <- $(rel_to_root "$src") ($(human "$(filesize "$src")"))"
  if [ "$dry" = 1 ]; then return 0; fi
  mkdir -p "$OUT/$key"
  if [ "$precision" = fp32 ]; then
    cp -p "$src" "$dst.tmp"
  else
    "$PY" -c 'import torch' 2> /dev/null || die "$PY cannot import torch; set PYTHON=<python with torch> or use --keep-fp32"
    log "         converting (loads the 1.26 GB checkpoint: needs about 2.5 GB of free memory)"
    "$PY" - "$src" "$dst.tmp" << 'PYEOF'
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu")
if not isinstance(sd, dict):
    sys.exit(f"{src}: expected a state_dict (dict of tensors), got {type(sd).__name__}")
out, kept, n16, worst = type(sd)(), [], 0, 0.0
for k, v in sd.items():
    if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.float16:
        # float16 holds |x| <= 65504; anything outside (or non-finite) keeps its original precision
        if v.numel() == 0 or (bool(torch.isfinite(v).all()) and float(v.abs().max()) <= 65504.0):
            h = v.to(torch.float16)
            if v.numel():
                worst = max(worst, float((h.float() - v.float()).abs().max() / v.float().abs().max().clamp(min=1e-12)))
            out[k] = h
            n16 += 1
            continue
        kept.append(k)
    out[k] = v  # integer buffers (e.g. BatchNorm num_batches_tracked) stay as they are
meta = getattr(sd, "_metadata", None)
if meta is not None:
    out._metadata = meta  # module versions, as torch's own state_dict() stores them
torch.save(out, dst)
print(f"[export_models]   {n16} tensors -> float16, {len(kept)} kept at full precision {kept[:3]}; "
      f"largest rounding error {worst:.1e} of a tensor's max magnitude", file=sys.stderr)
PYEOF
  fi
  mv -f "$dst.tmp" "$dst"
  printf '%s\n' "$want" > "$stamp"
  log "         $key/best.pt is $(human "$(filesize "$dst")")"
}

# ---- export -----------------------------------------------------------------------------------------
if [ "$dry" = 0 ]; then mkdir -p "$OUT"; fi
copy_art "$meta" fusion_meta.json 1

for k in $TAK_KEYS; do
  if wanted "$k"; then
    if fused "$k"; then export_tak "$k" 1; else export_tak "$k" 0; fi
  fi
done

# the WavLM head is tiny: always shipped when it exists, required when fused
if fused wavlm_head; then copy_art "$SRC/wavlm_head.npz" wavlm_head.npz 1; else copy_art "$SRC/wavlm_head.npz" wavlm_head.npz 0; fi

if fused wavlm_selftrain; then
  copy_art "$SRC/selftrain_base.npz" selftrain_base.npz 1
elif [ "$export_all" = 1 ]; then
  copy_art "$SRC/selftrain_base.npz" selftrain_base.npz 0
fi

if fused forensic_gbm; then r=1; else r=0; fi
copy_art "$SRC/forensic_gbm.txt" forensic_gbm.txt "$r"
copy_art "$SRC/forensic_gbm_cols.json" forensic_gbm_cols.json "$r"

if [ "$transcripts" = 1 ]; then copy_art "$ROOT/work/test_transcripts.json" test_transcripts.json 0; fi

for m in $models; do
  case " w2v2_aasist xlsr_sls $TAK_KEYS wavlm_head wavlm_selftrain forensic_gbm " in
    *" $m "*) ;;
    *) log "WARNING: fusion model '$m' is not known to this script; copy its files into models/ yourself" ;;
  esac
done

# artifacts of an earlier export that this fusion does not use
for rel in $KNOWN; do
  case " $exported " in *" $rel "*) continue ;; esac
  if [ -e "$OUT/$rel" ]; then
    log "remove   $rel (not part of this export)"
    if [ "$dry" = 0 ]; then rm -f "$OUT/$rel"; fi
  fi
done
for k in $TAK_KEYS; do
  if [ "$dry" = 0 ] && [ -d "$OUT/$k" ]; then rmdir "$OUT/$k" 2> /dev/null || true; fi
done

if [ "$dry" = 1 ]; then
  log "dry run finished"
  exit 0
fi

# ---- provenance + checksums -------------------------------------------------------------------------
# shellcheck disable=SC2086  # $exported is a list of relative paths without spaces
"$PY" - "$OUT" "$(rel_to_root "$meta")" "$version" $exported << 'PYEOF'
import datetime
import hashlib
import json
import os
import sys

out_dir, meta_rel, version, *files = sys.argv[1:]
files = sorted(set(files))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


meta = json.load(open(os.path.join(out_dir, "fusion_meta.json")))
entries = {rel: {"bytes": os.path.getsize(os.path.join(out_dir, rel)), "sha256": sha256(os.path.join(out_dir, rel))}
           for rel in files}
manifest = {
    "team": "dh_squad",
    "fusion_version": version,
    "fusion_meta_source": meta_rel,
    "fusion_models": meta.get("models"),
    "exported_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "files": entries,
    "public_weights": "not in this folder: downloaded at build time by scripts/fetch_models.sh "
                      "(pinned commits and SHA-256 in scripts/public_models.tsv)",
}
for key in ("tak_ft", "tak_head"):
    if f"{key}/best.pt.source" in entries:
        src_sha, precision = open(os.path.join(out_dir, f"{key}/best.pt.source")).read().split()
        manifest[key] = {"source": f"work/models/{key}/best.pt", "source_sha256": src_sha, "stored_as": precision}
with open(os.path.join(out_dir, "MANIFEST.json"), "w") as fh:
    json.dump(manifest, fh, indent=1)
    fh.write("\n")
with open(os.path.join(out_dir, "SHA256SUMS"), "w") as fh:
    for rel, e in entries.items():
        fh.write(f"{e['sha256']}  {rel}\n")
    fh.write(f"{sha256(os.path.join(out_dir, 'MANIFEST.json'))}  MANIFEST.json\n")
total = sum(e["bytes"] for e in entries.values())
print(f"[export_models] wrote MANIFEST.json and SHA256SUMS ({len(entries)} files, {total / 1e6:.0f} MB)", file=sys.stderr)
PYEOF

if [ "$make_tar" = 1 ]; then
  tarball="$ROOT/work/dist/hearsay-models-$version.tar.gz"
  mkdir -p "$(dirname "$tarball")"
  extra=""
  if [ -f "$OUT/README.md" ]; then extra="README.md"; fi
  # COPYFILE_DISABLE: stop macOS tar from adding ._* resource-fork files
  # shellcheck disable=SC2086  # word-splitting of the path lists is intended
  COPYFILE_DISABLE=1 tar -czf "$tarball.tmp" -C "$OUT" $exported MANIFEST.json SHA256SUMS $extra
  mv -f "$tarball.tmp" "$tarball"
  log "packed   $(rel_to_root "$tarball") ($(human "$(filesize "$tarball")"), sha256 $(sha256_of "$tarball"))"
  log "         upload it (e.g. as a GitHub release asset), then build with"
  log "         --build-arg OWN_MODELS_URL=<its URL> --build-arg OWN_MODELS_SHA256=<the sha256 above>"
fi
log "done. Next: docker build -t hearsay-dhsquad .   (see docs/DOCKER.md)"
