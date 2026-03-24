#!/usr/bin/env bash
# Debug tool for predict3d normal visualization.
# Runs 3D UNet inference on a small crop, extracts normal slices as JPGs,
# and optionally compares against a reference zarr.
#
# Required env vars:
#   INPUT_ZARR    - Input volume zarr path
#   CHECKPOINT    - 3D UNet checkpoint .pt
#
# Optional env vars (with defaults):
#   OUTPUT_DIR    - Output directory (default: ./debug_normals_out)
#   PX,PY,PZ     - Center point in fullres coords (default: 2568, 3695, 8577)
#   WHD           - Crop size in fullres (default: 1000)
#   SCALEDOWN     - Downsample factor (default: 4)
#   TILE_SIZE     - Tile size for inference (default: 192)
#   OVERLAP       - Tile overlap (default: 48)
#   BORDER        - Tile border discard (default: 16)
#   CHECKPOINT_2D - Optional 2D UNet checkpoint for comparison
#   REF_ZARR      - Optional reference zarr for comparison

set -euo pipefail

# --- Edit these paths directly ---
INPUT_ZARR=$VOL/volumes_webcache/s5_rechunked.zarr/0/
# CHECKPOINT=../train3d/runs/unet3d/20260319_214354_3d/model_best.pt #original long train
CHECKPOINT=../train3d/runs/unet3d/20260324_165356_3d/model_best.pt #trilin upsample, bf16, nonorms
OUTPUT_DIR=./debug_normals_out
REF_ZARR=../s5/PHerc172.volpkg/volumes_lasagna/cos3_3d.zarr
CHECKPOINT_2D=../../exps_2d/logs/20251205_125909_test_newgrad_lr1e-3/unet_current.pt

: "${INPUT_ZARR:?INPUT_ZARR is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"

OUTPUT_DIR="${OUTPUT_DIR:-./debug_normals_out}"
PX="${PX:-2568}"
PY="${PY:-3695}"
PZ="${PZ:-8577}"
WH="${WH:-1000}"
SCALEDOWN="${SCALEDOWN:-4}"
TILE_SIZE="${TILE_SIZE:-192}"
OVERLAP="${OVERLAP:-0}"
BORDER="${BORDER:-0}"
REF_ZARR="${REF_ZARR:-}"
CHECKPOINT_2D="${CHECKPOINT_2D:-}"

# Z crop depth = one tile (no blending in Z)
ZDEPTH="${ZDEPTH:-$TILE_SIZE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRED_ZARR="${OUTPUT_DIR}/pred.zarr"

# Compute crop origin (top-left corner)
HALF_WH=$((WH / 2))
HALF_Z=$((ZDEPTH / 2))
CX=$((PX - HALF_WH))
CY=$((PY - HALF_WH))
CZ=$((PZ - HALF_Z))

mkdir -p "$OUTPUT_DIR"

echo "=== debug_normals ==="
echo "Center: ($PX, $PY, $PZ)  WH: $WH  Zdepth: $ZDEPTH  Scaledown: $SCALEDOWN"
echo "Crop origin: ($CX, $CY, $CZ)  Overlap: $OVERLAP  Border: $BORDER"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: Run predict3d inference (delete old zarr to avoid resume logic)
if [ -e "$PRED_ZARR" ]; then
    echo "Removing old $PRED_ZARR"
    rm -rf "$PRED_ZARR"
fi
echo "--- Step 1: predict3d inference ---"
python "$SCRIPT_DIR/preprocess_cos_omezarr.py" predict3d \
    --input "$INPUT_ZARR" \
    --output "$PRED_ZARR" \
    --unet-checkpoint "$CHECKPOINT" \
    --crop $CX $CY $CZ $WH $WH $ZDEPTH \
    --scaledown "$SCALEDOWN" \
    --tile-size "$TILE_SIZE" \
    --overlap "$OVERLAP" \
    --border "$BORDER" \
    --calibrate-norm
echo ""

# Step 2: Extract slices
echo "--- Step 2: extract slices ---"
EXTRACT_ARGS=(
    --pred-zarr "$PRED_ZARR"
    --point "$PX" "$PY" "$PZ"
    --wh "$WH"
    --zdepth "$ZDEPTH"
    --output-dir "$OUTPUT_DIR"
)
if [ -n "$REF_ZARR" ]; then
    EXTRACT_ARGS+=(--ref-zarr "$REF_ZARR")
fi
if [ -n "$CHECKPOINT_2D" ]; then
    EXTRACT_ARGS+=(--checkpoint-2d "$CHECKPOINT_2D" --input-zarr "$INPUT_ZARR")
fi
python "$SCRIPT_DIR/debug_normals_extract.py" "${EXTRACT_ARGS[@]}"

echo ""
echo "=== done ==="
