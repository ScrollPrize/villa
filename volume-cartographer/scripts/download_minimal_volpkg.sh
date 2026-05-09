#!/usr/bin/env bash
# Download a minimal local copy of s3://volpkgs/s1_ds2.volpkg/ for vc3d testing.
# Zarr volumes are streamed via VC3D's remote-cache code path; only metadata
# files, the normalgrids_2um_ds folder, and a handful of tifxyz directories
# from paths_2um_ds2/ and traces/ are pulled.
#
# Usage:
#   AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_SESSION_TOKEN=... \
#   AWS_DEFAULT_REGION=us-east-1 \
#   scripts/download_minimal_volpkg.sh [--dest PATH] [--dry-run] [--keep N]
#
#   --dest PATH   Local destination (default: ./test-data/s1_ds2.volpkg)
#   --dry-run     List what would be pulled and exit.
#   --keep N      Keep N smallest tifxyz dirs from each of paths_2um_ds2/
#                 and traces/ (default: 2).
set -euo pipefail

SRC_BUCKET="volpkgs"
SRC_KEY="s1_ds2.volpkg"
SRC=":s3:${SRC_BUCKET}/${SRC_KEY}"

DEST="./test-data/s1_ds2.volpkg"
DRY_RUN=0
KEEP=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)    DEST="${2:?missing value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --keep)    KEEP="${2:?missing value}"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

# --- Credentials check -------------------------------------------------------
for v in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION; do
  if [[ -z "${!v:-}" ]]; then
    echo "ERROR: $v is not set in the environment." >&2
    echo "Source your STS creds (e.g. via 'aws configure export-credentials')." >&2
    exit 3
  fi
done
# AWS_SESSION_TOKEN is required for STS but not for IAM-user creds.
if [[ -z "${AWS_SESSION_TOKEN:-}" ]]; then
  echo "warning: AWS_SESSION_TOKEN is unset (assuming non-STS creds)." >&2
fi

# --- Tool selection ----------------------------------------------------------
USE_RCLONE=1
if ! command -v rclone >/dev/null 2>&1; then
  USE_RCLONE=0
  if ! command -v aws >/dev/null 2>&1; then
    echo "ERROR: need either rclone or aws CLI." >&2
    exit 4
  fi
fi

rclone_copy() {
  # rclone_copy <subpath> <local_subdest> [extra rclone args...]
  local sub="$1"; shift
  local dst="$1"; shift
  local cmd=(rclone copy
             "${SRC}/${sub}" "${dst}"
             --s3-provider AWS --s3-env-auth
             --transfers 64 --buffer-size 2M --size-only --fast-list)
  cmd+=("$@")
  if (( DRY_RUN )); then
    echo "DRY: ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
}

aws_sync() {
  local sub="$1"; shift
  local dst="$1"; shift
  local cmd=(aws s3 sync
             "s3://${SRC_BUCKET}/${SRC_KEY}/${sub}" "${dst}")
  cmd+=("$@")
  if (( DRY_RUN )); then
    echo "DRY: ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
}

pull_dir() {
  local sub="$1"; shift
  local dst="$DEST/${sub%/}"
  mkdir -p "$dst"
  if (( USE_RCLONE )); then rclone_copy "$sub" "$dst" "$@"
  else                      aws_sync    "$sub" "$dst" "$@"
  fi
}

# --- 1) Top-level volpkg metadata -------------------------------------------
echo "== top-level metadata =="
mkdir -p "$DEST"
if (( USE_RCLONE )); then
  # Single-level top dir; --max-depth 1 keeps it light.
  rclone copy "${SRC}" "$DEST" \
    --s3-provider AWS --s3-env-auth \
    --transfers 8 --size-only --fast-list \
    --include "*.json" --max-depth 1 \
    $( (( DRY_RUN )) && echo --dry-run )
else
  aws_sync "" "$DEST" --exclude "*" --include "*.json" --no-progress
fi

# --- 2) normalgrids_2um_ds (full) -------------------------------------------
echo "== normalgrids_2um_ds (full) =="
pull_dir "normalgrids_2um_ds/"

# --- 3) volumes/<vol>/ metadata only ----------------------------------------
echo "== volumes (metadata only, chunks excluded) =="
if (( USE_RCLONE )); then
  # List immediate children of volumes/ to discover individual zarr roots.
  vols=$(rclone lsf --dirs-only --max-depth 1 "${SRC}/volumes/" \
         --s3-provider AWS --s3-env-auth 2>/dev/null || true)
  for v in $vols; do
    v="${v%/}"
    [[ -z "$v" ]] && continue
    echo "  -> volumes/$v"
    rclone_copy "volumes/${v}/" "$DEST/volumes/${v}/" \
      --include "*.json" --include "*.zarray" --include "*.zgroup" --include "*.zattrs" \
      --exclude "*"
  done
else
  aws_sync "volumes/" "$DEST/volumes/" \
    --exclude "*" \
    --include "*.json" --include "*.zarray" --include "*.zgroup" --include "*.zattrs" \
    --no-progress
fi

# --- 4) Smallest N tifxyz dirs from paths_2um_ds2/ and traces/ --------------
pull_smallest_tifxyz() {
  local sub="$1"
  echo "== ${sub} (${KEEP} smallest tifxyz dirs) =="
  if ! (( USE_RCLONE )); then
    # aws CLI cannot easily sort by size; fall back to alphabetical first N.
    local names
    names=$(aws s3 ls "s3://${SRC_BUCKET}/${SRC_KEY}/${sub}/" \
            | awk '$1=="PRE"{print $2}' | sort | head -n "$KEEP")
    for n in $names; do
      n="${n%/}"
      pull_dir "${sub}/${n}/"
    done
    return
  fi
  # rclone path: enumerate dir sizes and take the smallest N.
  local sizes
  sizes=$(rclone size --json "${SRC}/${sub}/" \
          --s3-provider AWS --s3-env-auth 2>/dev/null || true)
  local children
  children=$(rclone lsf --dirs-only --max-depth 1 "${SRC}/${sub}/" \
             --s3-provider AWS --s3-env-auth 2>/dev/null || true)
  if [[ -z "$children" ]]; then
    echo "  (no children under ${sub}/, skipping)"
    return
  fi
  # For each child, query its size; sort ascending; take N.
  local picks=()
  while read -r name; do
    name="${name%/}"
    [[ -z "$name" ]] && continue
    local b
    b=$(rclone size "${SRC}/${sub}/${name}/" \
        --s3-provider AWS --s3-env-auth --json 2>/dev/null \
        | sed -n 's/.*"bytes":\([0-9]*\).*/\1/p' | head -n1)
    [[ -z "$b" ]] && b=0
    picks+=("$b $name")
  done <<<"$children"
  printf '  candidates: %d\n' "${#picks[@]}"
  local chosen
  chosen=$(printf '%s\n' "${picks[@]}" | sort -n | head -n "$KEEP" | awk '{print $2}')
  for n in $chosen; do
    echo "  -> ${sub}/${n}"
    pull_dir "${sub}/${n}/"
  done
}
pull_smallest_tifxyz "paths_2um_ds2"
pull_smallest_tifxyz "traces"

# --- 5) Verifier -------------------------------------------------------------
if (( DRY_RUN )); then
  echo "(dry-run: skipping verifier)"
  exit 0
fi

echo "== verifier =="
fail=0
warn() { echo "WARN: $*"; }
check() { [[ -e "$1" ]] || { echo "MISS: $1"; fail=1; }; }

# Top-level metadata: at least one *.json (the volpkg config).
shopt -s nullglob
top_jsons=("$DEST"/*.json)
shopt -u nullglob
(( ${#top_jsons[@]} >= 1 )) || { echo "MISS: no top-level *.json under $DEST"; fail=1; }

# normalgrids_2um_ds populated.
ng_count=$(find "$DEST/normalgrids_2um_ds" -type f 2>/dev/null | wc -l | tr -d ' ')
(( ng_count >= 1 )) || { echo "MISS: normalgrids_2um_ds empty"; fail=1; }

# At least one .zarray under volumes/ (some Zarr volume metadata).
zarray_count=$(find "$DEST/volumes" -name '.zarray' -o -name 'zarr.json' 2>/dev/null | wc -l | tr -d ' ')
(( zarray_count >= 1 )) || warn "no .zarray / zarr.json found under $DEST/volumes (volumes may be missing)"

# At least 2 tifxyz dirs across paths_2um_ds2/ + traces/ with x.tif/y.tif/z.tif.
tifxyz_ok=0
for d in "$DEST"/paths_2um_ds2/*/ "$DEST"/traces/*/; do
  [[ -d "$d" ]] || continue
  if [[ -f "$d/x.tif" && -f "$d/y.tif" && -f "$d/z.tif" ]]; then
    tifxyz_ok=$((tifxyz_ok + 1))
  fi
done
(( tifxyz_ok >= 2 )) || { echo "MISS: fewer than 2 complete tifxyz dirs (have $tifxyz_ok)"; fail=1; }

total=$(du -sh "$DEST" 2>/dev/null | awk '{print $1}')
total_bytes=$(du -sk "$DEST" 2>/dev/null | awk '{print $1}')
echo "Total size: ${total:-?}"
if [[ -n "${total_bytes:-}" ]] && (( total_bytes > 2 * 1024 * 1024 )); then
  warn "downloaded set is >2 GB; consider --keep 1"
fi

if (( fail )); then
  echo "Verification FAILED."
  exit 5
fi
echo "Verification OK: ${tifxyz_ok} tifxyz dirs, ${ng_count} normalgrid files, ${zarray_count} zarr metadata files."
