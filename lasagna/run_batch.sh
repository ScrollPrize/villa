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
#     ├── fitted.zarr     (dense normal/winding/validity/density volumes)
#     ├── fitted_tif/     (same as fitted.zarr but as multi-layer TIFs)
#     ├── unet_labels/    (UNet training labels: cos, grad_mag, dir_{z,y,x}, validity)
#     └── logs/
#         ├── prep1.log
#         ├── prep2.log
#         ├── fit.log
#         ├── analyze.log
#         ├── fit_data.log
#         └── unet_labels.log

set -euo pipefail

# --- Configuration -----------------------------------------------------------
MAX_JOBS=4
OMP_THREADS=1          # threads per vc_ngrids/vc_gen_normalgrids call
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

    # Skip logic: three-tier
    if [[ -f "${outdir}/unet_labels/validity.tif" ]]; then
        echo "[${label}] already complete (unet_labels done), skipping."
        return 0
    fi

    if [[ -f "${outdir}/fitted_tif/winding.tif" ]]; then
        echo "[${label}] steps 1-5 done, running unet_labels only..."
        # Jump straight to step 6
        echo "[${label}] Step 6/6: unet_labels"
        python "${SRC}/lasagna/fitted_to_unet_labels.py" \
            --input "${outdir}/fitted.zarr" \
            --output-dir "${outdir}/unet_labels" \
            > "${logdir}/unet_labels.log" 2>&1 || {
                echo "[${label}] FAILED at unet_labels"; return 1
            }
        echo "[${label}] Done."
        return 0
    fi

    if [[ -f "${outdir}/stats.json" ]]; then
        echo "[${label}] steps 1-4 done, running fit_data + unet_labels..."
        # Jump straight to step 5
        echo "[${label}] Step 5/6: fit_data"
        python "${SRC}/lasagna/lasagna_fit_data.py" \
            --model "${outdir}/fit_output/model_final.pt" \
            --input "${outdir}/normals.zarr" \
            --output "${outdir}/fitted.zarr" \
            --tif-output "${outdir}/fitted_tif" \
            --normal-vis-dir "${outdir}/normal_vis" \
            --labels "$input_tif" \
            --stats-json "${outdir}/fit_data_stats.json" \
            > "${logdir}/fit_data.log" 2>&1 || {
                echo "[${label}] FAILED at fit_data"; return 1
            }
        echo "[${label}] Step 6/6: unet_labels"
        python "${SRC}/lasagna/fitted_to_unet_labels.py" \
            --input "${outdir}/fitted.zarr" \
            --output-dir "${outdir}/unet_labels" \
            > "${logdir}/unet_labels.log" 2>&1 || {
                echo "[${label}] FAILED at unet_labels"; return 1
            }
        echo "[${label}] Done."
        return 0
    fi

    echo "[${label}] Starting pipeline..."

    # Step 1: prep1 — labels_to_winding_volume
    echo "[${label}] Step 1/6: winding volume"
    python "${SRC}/lasagna/labels_to_winding_volume.py" \
        --input "$input_tif" \
        --output "${outdir}/winding.zarr" \
        > "${logdir}/prep1.log" 2>&1 || {
            echo "[${label}] FAILED at prep1"; return 1
        }

    # Step 2: prep2 — labels_to_lasagna_normals
    echo "[${label}] Step 2/6: lasagna normals"
    python "${SRC}/lasagna/labels_to_lasagna_normals.py" \
        --input "$input_tif" \
        --work-dir "${outdir}/work" \
        --output "${outdir}/normals.zarr" \
        > "${logdir}/prep2.log" 2>&1 || {
            echo "[${label}] FAILED at prep2"; return 1
        }

    # Step 3: fit
    echo "[${label}] Step 3/6: fit"
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
        --erode-valid-mask 4 \
        --no-pyramid-d \
        > "${logdir}/fit.log" 2>&1 || {
            echo "[${label}] FAILED at fit"; return 1
        }

    # Step 4: analyze (vis + stats)
    echo "[${label}] Step 4/6: analyze"
    python "${SRC}/lasagna/lasagna_analyze.py" \
        --model "${outdir}/fit_output/model.pt" \
        --input "${outdir}/normals.zarr" \
        --output-dir "${outdir}/vis" \
        --stats-json "${outdir}/stats.json" \
        --winding-volume "${outdir}/winding.zarr" \
        --erode-valid-mask 4 \
        > "${logdir}/analyze.log" 2>&1 || {
            echo "[${label}] FAILED at analyze"; return 1
        }

    # Step 5: fit_data — dense volumes + TIFs + normal vis
    echo "[${label}] Step 5/6: fit_data"
    python "${SRC}/lasagna/lasagna_fit_data.py" \
        --model "${outdir}/fit_output/model_final.pt" \
        --input "${outdir}/normals.zarr" \
        --output "${outdir}/fitted.zarr" \
        --tif-output "${outdir}/fitted_tif" \
        --normal-vis-dir "${outdir}/normal_vis" \
        --labels "$input_tif" \
        --stats-json "${outdir}/fit_data_stats.json" \
        > "${logdir}/fit_data.log" 2>&1 || {
            echo "[${label}] FAILED at fit_data"; return 1
        }

    # Step 6: unet_labels — convert fitted.zarr to UNet training labels
    echo "[${label}] Step 6/6: unet_labels"
    python "${SRC}/lasagna/fitted_to_unet_labels.py" \
        --input "${outdir}/fitted.zarr" \
        --output-dir "${outdir}/unet_labels" \
        > "${logdir}/unet_labels.log" 2>&1 || {
            echo "[${label}] FAILED at unet_labels"; return 1
        }

    echo "[${label}] Done."
}
export -f run_sample
export SRC OMP_NUM_THREADS

# --- Run in parallel ----------------------------------------------------------
mkdir -p "$OUTPUT_ROOT"
printf '%s\n' "${FILES[@]}" | xargs -P "$MAX_JOBS" -I{} bash -c 'run_sample "$@"' _ {} "$OUTPUT_ROOT"

echo "All done. Results in: $OUTPUT_ROOT"
