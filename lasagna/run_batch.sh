#!/usr/bin/env bash
# run_batch.sh — Run the full lasagna pipeline on all matching label TIFFs.
#
# Usage:
#   ./run_batch.sh <output_root> <input_tifs...>
#
# Example:
#   ./run_batch.sh ./batch_output ../kaggle_dataset/labels/sample_*_surface.tif
#
# Each sample gets its own output directory:
#   <output_root>/sample_00033/
#     ├── winding.zarr
#     ├── normals.zarr
#     ├── work/           (intermediate files from prep2)
#     ├── fit_output/     (model.pt, snapshots)
#     ├── vis/            (OBJ/MTL/PNG visualization + stats.json)
#     └── logs/
#         ├── prep1.log
#         ├── prep2.log
#         ├── fit.log
#         └── analyze.log

set -euo pipefail

# --- Configuration -----------------------------------------------------------
MAX_JOBS=16
OMP_THREADS=2          # threads per vc_ngrids/vc_gen_normalgrids call
SRC="${SRC:-$(cd "$(dirname "$0")/.." && pwd)}"

# --- Args ---------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <output_root> <input_tifs...>"
    echo "  e.g. $0 ./batch_output ../kaggle_dataset/labels/sample_*_surface.tif"
    exit 1
fi

OUTPUT_ROOT="$1"; shift
FILES=("$@")

# Ensure vc tools are on PATH
export PATH="$PATH:${SRC}/volume-cartographer/build/bin/"
export OMP_NUM_THREADS="$OMP_THREADS"

echo "Found ${#FILES[@]} samples to process (max $MAX_JOBS parallel)."

# --- Per-sample pipeline function ---------------------------------------------
run_sample() {
    local input_tif="$1"
    local output_root="$2"

    # Extract sample id from filename: sample_00033_surface.tif -> sample_00033
    local base
    base="$(basename "$input_tif" .tif)"        # sample_00033_surface
    local label="${base%_surface}"               # sample_00033

    local outdir="${output_root}/${label}"
    local logdir="${outdir}/logs"
    mkdir -p "$logdir" "${outdir}/work"

    echo "[${label}] Starting pipeline..."

    # Step 1: prep1 — labels_to_winding_volume
    echo "[${label}] Step 1/4: winding volume"
    python "${SRC}/lasagna/labels_to_winding_volume.py" \
        --input "$input_tif" \
        --output "${outdir}/winding.zarr" \
        > "${logdir}/prep1.log" 2>&1 || {
            echo "[${label}] FAILED at prep1"; return 1
        }

    # Step 2: prep2 — labels_to_lasagna_normals
    echo "[${label}] Step 2/4: lasagna normals"
    python "${SRC}/lasagna/labels_to_lasagna_normals.py" \
        --input "$input_tif" \
        --work-dir "${outdir}/work" \
        --output "${outdir}/normals.zarr" \
        > "${logdir}/prep2.log" 2>&1 || {
            echo "[${label}] FAILED at prep2"; return 1
        }

    # Step 3: fit
    echo "[${label}] Step 3/4: fit"
    python "${SRC}/lasagna/fit.py" \
        "${SRC}/lasagna/vc3d_configs/vc3d_labels_3d_straight.json" \
        --input "${outdir}/normals.zarr" \
        --seed 150 150 150 \
        --model-w 1000 --model-h 1000 \
        --windings 20 \
        --out-dir "${outdir}/fit_output" \
        --model-output "${outdir}/fit_output/model.pt" \
        --winding-volume "${outdir}/winding.zarr" \
        --normal-mask-zero 1 \
        --erode-valid-mask 1 \
        > "${logdir}/fit.log" 2>&1 || {
            echo "[${label}] FAILED at fit"; return 1
        }

    # Step 4: analyze (vis + stats)
    echo "[${label}] Step 4/4: analyze"
    python "${SRC}/lasagna/export_vis_obj.py" \
        --model "${outdir}/fit_output/model.pt" \
        --input "${outdir}/normals.zarr" \
        --output-dir "${outdir}/vis" \
        --stats-json "${outdir}/stats.json" \
        --winding-volume "${outdir}/winding.zarr" \
        > "${logdir}/analyze.log" 2>&1 || {
            echo "[${label}] FAILED at analyze"; return 1
        }

    echo "[${label}] Done."
}
export -f run_sample
export SRC OMP_NUM_THREADS

# --- Run in parallel ----------------------------------------------------------
mkdir -p "$OUTPUT_ROOT"
printf '%s\n' "${FILES[@]}" | xargs -P "$MAX_JOBS" -I{} bash -c 'run_sample "$@"' _ {} "$OUTPUT_ROOT"

echo "All done. Results in: $OUTPUT_ROOT"
