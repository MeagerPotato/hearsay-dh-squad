#!/usr/bin/env bash
# Download the public pretrained weights HEARSAY builds on, pinned to exact Hugging Face commits.
#
#   scripts/fetch_models.sh [options] [DEST]
#
# DEST defaults to $HEARSAY_MODELS, else <repo>/work/models (the Docker build uses /app/models).
# The file list, commits, byte sizes and SHA-256 hashes live in scripts/public_models.tsv. Every file's
# size is checked and every new download is hashed. Idempotent: files already present with the right
# size are skipped, and an interrupted download resumes from its .part file on the next run.
#
# Options
#   --dry-run     print what would be downloaded and from where; download nothing
#   --check       only verify the files already in DEST (exit 1 if one is missing or wrong)
#   --no-la       skip the official LA checkpoint (1.27 GB; inference does not need it, only
#                 fine-tuning and other scripts that call hearsay.tak_aasist.load_la_model)
#   --no-ecapa    skip the SpeechBrain ECAPA speaker model (~90 MB; report evidence only)
#   --own URL     also fetch our trained artifacts as packed by `scripts/export_models.sh --tar`
#                 (a .tar.gz, e.g. a GitHub release asset), unpack them into DEST, verify SHA256SUMS
#   --own-only    with --own: skip the public weights
#
# Environment
#   HF_ENDPOINT=https://huggingface.co   Hub endpoint or mirror
#   FETCH_LA=0                           same as --no-la
#   FETCH_ECAPA=0                        same as --no-ecapa
#   VERIFY_SHA256=0                      do not hash new downloads (sizes are still checked)
#   VERIFY_EXISTING=1                    also re-hash files that are already present
#   OWN_MODELS_SHA256=<hex>              expected SHA-256 of the --own archive
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
MANIFEST="$HERE/public_models.tsv"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_ENDPOINT="${HF_ENDPOINT%/}"
VERIFY_SHA256="${VERIFY_SHA256:-1}"
VERIFY_EXISTING="${VERIFY_EXISTING:-0}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"

log() { printf '[fetch_models] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

filesize() { wc -c < "$1" | tr -d '[:space:]'; }
sha256_of() {
  if command -v sha256sum > /dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
human() { awk -v b="$1" 'BEGIN { split("B KB MB GB TB", u, " "); i = 1; while (b >= 1000 && i < 5) { b /= 1000; i++ } printf("%.1f %s", b, u[i]) }'; }

mode=fetch
with_la="${FETCH_LA:-1}"
with_ecapa="${FETCH_ECAPA:-1}"
own_url=""
own_only=0
dest=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) mode=dry ;;
    --check) mode=check ;;
    --no-la) with_la=0 ;;
    --no-ecapa) with_ecapa=0 ;;
    --own) [ $# -ge 2 ] || die "--own needs a URL"; own_url="$2"; shift ;;
    --own=*) own_url="${1#--own=}" ;;
    --own-only) own_only=1 ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1 (see --help)" ;;
    *) [ -z "$dest" ] || die "only one DEST folder can be given"; dest="$1" ;;
  esac
  shift
done
if [ "$own_only" = 1 ] && [ -z "$own_url" ]; then die "--own-only needs --own URL"; fi
dest="${dest:-${HEARSAY_MODELS:-$ROOT/work/models}}"
[ -f "$MANIFEST" ] || die "manifest not found: $MANIFEST"
if [ "$mode" = fetch ] && ! command -v curl > /dev/null 2>&1; then die "curl is required"; fi
progress="--silent"
if [ -t 2 ]; then progress="--progress-bar"; fi

# fail_fast RC URL: stop on curl errors that no retry can fix.
fail_fast() {
  case "$1" in
    1|2|3) die "curl cannot use this URL (exit code $1; spaces must be written as %20): $2" ;;
    23) die "cannot write the download (disk full?), curl exit code 23" ;;
  esac
}

# download URL OUT SIZE: fetch URL into OUT, resuming a partial OUT, with retries and back-off.
download() {
  local url="$1" out="$2" size="$3" attempt=1 have rc
  while :; do
    have=0
    if [ -f "$out" ]; then have="$(filesize "$out")"; fi
    if [ "$have" -gt "$size" ]; then rm -f "$out"; have=0; fi
    if [ "$have" -eq "$size" ]; then return 0; fi
    [ "$attempt" -le "$MAX_ATTEMPTS" ] || die "giving up after $MAX_ATTEMPTS attempts: $url"
    if [ "$have" -gt 0 ]; then log "  resuming at $(human "$have")"; fi
    rc=0
    curl "$progress" --show-error --fail --location --connect-timeout 30 \
         --speed-limit 20000 --speed-time 60 \
         --continue-at - --output "$out" "$url" || rc=$?
    if [ "$rc" -ne 0 ]; then
      fail_fast "$rc" "$url"
      log "  attempt $attempt failed (curl exit code $rc)"
      # 33 = server refused the byte range, 36 = bad resume offset: start this file again from zero
      if [ "$rc" -eq 33 ] || [ "$rc" -eq 36 ]; then rm -f "$out"; fi
      sleep $((attempt * 5))
    fi
    attempt=$((attempt + 1))
  done
}

missing=0
fetched=0
need_bytes=0

# fetch_file REPO REV FILE REL SIZE SHA256: make DEST/REL a verified copy of REPO@REV:FILE.
fetch_file() {
  local repo="$1" rev="$2" file="$3" rel="$4" size="$5" sha="$6"
  local out="$dest/$rel" url="$HF_ENDPOINT/$repo/resolve/$rev/$file" got
  if [ -f "$out" ] && [ "$(filesize "$out")" = "$size" ]; then
    if [ "$VERIFY_EXISTING" = 1 ] && [ "$sha" != "-" ]; then
      got="$(sha256_of "$out")"
      if [ "$got" != "$sha" ]; then
        [ "$mode" = check ] || die "$rel has the right size but SHA-256 $got (expected $sha); delete it and re-run"
        log "BAD      $rel (SHA-256 $got, expected $sha)"
        missing=$((missing + 1))
        return 0
      fi
    fi
    log "ok       $rel ($(human "$size"))"
    return 0
  fi
  case "$mode" in
    dry)
      log "get      $rel ($(human "$size")) <- $url"
      return 0 ;;
    check)
      if [ -e "$out" ]; then log "BAD      $rel (size $(filesize "$out"), expected $size)"; else log "MISSING  $rel"; fi
      missing=$((missing + 1))
      return 0 ;;
  esac
  log "get      $rel ($(human "$size")) <- $url"
  mkdir -p "$(dirname "$out")"
  download "$url" "$out.part" "$size"
  if [ "$VERIFY_SHA256" = 1 ] && [ "$sha" != "-" ]; then
    got="$(sha256_of "$out.part")"
    if [ "$got" != "$sha" ]; then
      rm -f "$out.part"
      die "$rel: SHA-256 mismatch (got $got, expected $sha); the download was deleted, re-run to retry"
    fi
  fi
  mv -f "$out.part" "$out"
  fetched=$((fetched + 1))
}

count_need() {
  local out="$dest/$4"
  if [ ! -f "$out" ] || [ "$(filesize "$out")" != "$5" ]; then need_bytes=$((need_bytes + $5)); fi
}

# each_public FUNCTION: call FUNCTION REPO REV FILE REL SIZE SHA256 for every selected manifest row.
each_public() {
  local fn="$1" group repo rev file rel size sha
  while IFS=$'\t' read -r group repo rev file rel size sha <&3; do
    case "$group" in ''|'#'*) continue ;; esac
    if [ "$group" = la ] && [ "$with_la" != 1 ]; then continue; fi
    if [ "$group" = ecapa ] && [ "$with_ecapa" != 1 ]; then continue; fi
    case "$size" in ''|*[!0-9]*) die "malformed row in $MANIFEST: $group $repo $file" ;; esac
    "$fn" "$repo" "$rev" "$file" "$rel" "$size" "${sha:--}"
  done 3< "$MANIFEST"
}

# fetch_own URL: download our packed artifacts and unpack them into DEST.
fetch_own() {
  local url="$1" tmp got attempt=1 rc
  case "$mode" in
    dry) log "get      trained artifacts <- $url"; return 0 ;;
    check) return 0 ;;  # --check verifies DEST/SHA256SUMS instead
  esac
  mkdir -p "$dest"
  tmp="$dest/.own-models.tar.gz.part"
  rm -f "$tmp"
  log "get      trained artifacts <- $url"
  while :; do
    rc=0
    curl "$progress" --show-error --fail --location --connect-timeout 30 \
         --speed-limit 20000 --speed-time 60 --output "$tmp" "$url" || rc=$?
    if [ "$rc" -eq 0 ]; then break; fi
    fail_fast "$rc" "$url"
    [ "$attempt" -lt "$MAX_ATTEMPTS" ] || die "could not download $url (curl exit code $rc)"
    log "  attempt $attempt failed (curl exit code $rc); retrying"
    sleep $((attempt * 5))
    attempt=$((attempt + 1))
  done
  if [ -n "${OWN_MODELS_SHA256:-}" ]; then
    got="$(sha256_of "$tmp")"
    if [ "$got" != "$OWN_MODELS_SHA256" ]; then rm -f "$tmp"; die "archive SHA-256 $got != OWN_MODELS_SHA256"; fi
  fi
  tar -xzf "$tmp" -C "$dest" --no-same-owner
  rm -f "$tmp"
  log "unpacked trained artifacts into $dest"
}

# verify_sums: check DEST/SHA256SUMS (written by scripts/export_models.sh), if there is one.
verify_sums() {
  if [ ! -f "$dest/SHA256SUMS" ]; then log "no SHA256SUMS in $dest, trained artifacts not verified"; return 0; fi
  log "verifying $dest/SHA256SUMS"
  if command -v sha256sum > /dev/null 2>&1; then
    (cd "$dest" && sha256sum --check SHA256SUMS >&2) || die "trained artifacts in $dest fail their SHA256SUMS"
  else
    (cd "$dest" && shasum -a 256 --check SHA256SUMS >&2) || die "trained artifacts in $dest fail their SHA256SUMS"
  fi
}

log "destination: $dest ($mode)"
if [ "$own_only" = 0 ]; then
  if [ "$mode" = fetch ]; then
    mkdir -p "$dest"
    each_public count_need
    avail_kb="$(df -Pk "$dest" | awk 'NR == 2 { print $4 }')"
    if [ "$need_bytes" -gt 0 ] && [ -n "$avail_kb" ] && [ $((avail_kb * 1024)) -lt "$need_bytes" ]; then
      log "WARNING: $(human "$need_bytes") to download but only $(human $((avail_kb * 1024))) free in $dest"
    fi
  fi
  each_public fetch_file
fi
if [ -n "$own_url" ]; then
  fetch_own "$own_url"
  if [ "$mode" = fetch ]; then verify_sums; fi
fi
if [ "$mode" = check ]; then
  verify_sums
  if [ "$missing" -gt 0 ]; then die "$missing file(s) missing or wrong in $dest; run scripts/fetch_models.sh $dest"; fi
  log "all files present"
elif [ "$mode" = fetch ]; then
  log "done ($fetched public weight file(s) downloaded into $dest)"
fi
