#!/bin/bash
# render_band.sh LIST VOLUME: render a list of tifxyz surfaces to zarrs with vc_render_tifxyz.
#   LIST lines: "SURFACE_DIR OUT_ZARR" (OUT_ZARR must end in .zarr; GPU mode deletes it first).
#   Paths must not contain spaces.
#   VOLUME: local OME-Zarr root (level 0 is VOLUME/0).
#   Flags passed to every render: -g 0 --scale 1 -n $RENDER_N [--flip-normals] --cache-gb 4.
#
# RENDER_GPU=0 (default): the unchanged CPU vc_render_tifxyz, one surface at a time.
# RENDER_GPU=1: vc_render_tifxyz runs with render_shim.so preloaded, which keeps the C++ geometry
#   but dumps each tile's base points and step directions instead of sampling; one render_gpu.py
#   process samples every surface on the GPU with a volume-chunk pool shared across the list.
#   A remote (http/s3) VOLUME, or no PyTorch GPU, falls back to the CPU path.
#
# env: VC_RENDER_TIFXYZ (default: vc_render_tifxyz on PATH), PYTHON (default: python3),
#      RENDER_N=31, RENDER_FLIP=1, RGPU_DEVICE=cuda:0, RGPU_POOL_GB=10 (GPU memory for volume chunks)
set -u -o pipefail
if [ $# -ne 2 ]; then
  echo "usage: $0 LIST VOLUME" >&2
  exit 2
fi
LIST=$1
V=$2
N=${RENDER_N:-31}
FLIP=()
[ "${RENDER_FLIP:-1}" = 1 ] && FLIP=(--flip-normals)
BIN=${VC_RENDER_TIFXYZ:-vc_render_tifxyz}
PY=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
elapsed() { awk -v a="$1" -v b="$(date +%s.%N)" 'BEGIN { printf "%.2f", b - a }'; }

# refuse to delete anything that is not a non-empty *.zarr path
check_out() {
  case "$1" in
    *?.zarr) ;;
    *) echo "render_band: OUT_ZARR must end in .zarr: '$1'" >&2; exit 2 ;;
  esac
}
while read -r S O; do
  [ -z "$S" ] && continue
  check_out "${O:-}"
done < "$LIST"

render() {  # SURFACE OUT
  "$BIN" -s "$1" -v "$V" -g 0 --scale 1 -n "$N" "${FLIP[@]}" --zarr-output "$2" --cache-gb 4 > "$2.log" 2>&1
}

s0=$(date +%s.%N)
GPU=${RENDER_GPU:-0}
if [ "$GPU" = 1 ] && [[ "$V" == http* || "$V" == s3* ]]; then
  echo "render_band: remote volume, RENDER_GPU falls back to CPU" >&2
  GPU=0
fi
if [ "$GPU" = 1 ] && ! "$PY" -c "import sys, torch; sys.exit(not torch.cuda.is_available())" 2>/dev/null; then
  echo "render_band: no PyTorch GPU, RENDER_GPU falls back to CPU" >&2
  GPU=0
fi
if [ "$GPU" != 1 ]; then
  fails=0
  while read -r S O; do
    [ -z "$S" ] && continue
    s=$(date +%s.%N)
    render "$S" "$O"
    rc=$?
    [ $rc = 0 ] || fails=$((fails + 1))
    echo "CPU $(basename "$S") rc=$rc $(elapsed "$s")s"
  done < "$LIST"
  echo "render_band CPU $(grep -c . "$LIST") surfaces, $fails failed, $(elapsed "$s0")s"
  [ $fails = 0 ]
  exit
fi

[ -f "$HERE/render_shim.so" ] || { echo "build render_shim.so first (README.md)" >&2; exit 1; }
GL=$(mktemp)
producer=
trap '[ -n "$producer" ] && kill "$producer" 2>/dev/null; rm -f "$GL"' EXIT
while read -r S O; do
  [ -n "$S" ] && echo "$O.rgpu_dump $O" >> "$GL"
done < "$LIST"
(
  while read -r S O; do
    [ -z "$S" ] && continue
    D=$O.rgpu_dump
    s=$(date +%s.%N)
    rm -rf -- "$O" "$D"
    mkdir -p "$D"
    LD_PRELOAD=$HERE/render_shim.so RENDER_SHIM_DIR=$D RENDER_SHIM_MODE=dump render "$S" "$O"
    rc=$?
    if [ $rc = 0 ]; then touch "$D/.done"; else touch "$D/.fail"; fi
    echo "DUMP $(basename "$S") rc=$rc $(elapsed "$s")s"
  done < "$LIST"
) &
producer=$!
"$PY" "$HERE/render_gpu.py" "@$GL" "$V/0" --device "${RGPU_DEVICE:-cuda:0}" \
  --pool-gb "${RGPU_POOL_GB:-10}" --wait --producer-pid "$producer" --rm-dump
rc=$?
[ $rc = 0 ] || kill "$producer" 2>/dev/null
wait "$producer"
producer=
if [ $rc != 0 ]; then  # --rm-dump only removes dumps of surfaces that were written
  while read -r D O; do rm -rf -- "$D"; done < "$GL"
fi
echo "render_band GPU $(grep -c . "$LIST") surfaces rc=$rc $(elapsed "$s0")s"
exit $rc
