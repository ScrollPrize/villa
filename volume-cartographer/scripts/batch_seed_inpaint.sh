#!/bin/bash

# Batch run vc_grow_seg_from_seed in resume/inpaint mode on tifxyz segments.
# Usage:
#   batch_seed_inpaint.sh <zarr_volume> <input_folder> <params_json> [output_folder] [extra vc_grow_seg_from_seed args...]
#
# - <zarr_volume>:     Path to the OME-Zarr volume (scale 0) used for tracing.
# - <input_folder>:    Directory whose immediate subfolders are tifxyz surfaces (x.tif, y.tif, z.tif).
# - <params_json>:     JSON file with base parameters for vc_grow_seg_from_seed.
# - [output_folder]:   Optional destination root. If omitted, each segment writes to <segment>_inpaint.
# - [extra args...]:   Optional additional flags passed through to vc_grow_seg_from_seed
#                      (e.g. --skip-overlap-check).
#
# The script forces the generated parameter JSON to use generations=0 so the tracer
# performs only the inpainting pass without expanding further generations.

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <zarr_volume> <input_folder> <params_json> [output_folder] [extra vc_grow_seg_from_seed args...]"
    exit 1
fi

VOLUME_PATH="$1"
INPUT_FOLDER="$2"
PARAMS_JSON="$3"
shift 3

OUTPUT_ROOT=""
EXTRA_ARGS=()

if [ "$#" -gt 0 ]; then
    if [[ ! "$1" =~ ^- ]]; then
        OUTPUT_ROOT="$1"
        shift
    fi
    if [ "$#" -gt 0 ]; then
        EXTRA_ARGS=("$@")
    fi
fi

if [ ! -d "$VOLUME_PATH" ]; then
    echo "Error: Volume path is not a directory: $VOLUME_PATH"
    exit 1
fi

if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder does not exist: $INPUT_FOLDER"
    exit 1
fi

if [ ! -f "$PARAMS_JSON" ]; then
    echo "Error: Params JSON not found: $PARAMS_JSON"
    exit 1
fi

if [ -n "$OUTPUT_ROOT" ]; then
    mkdir -p "$OUTPUT_ROOT"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VC_GSFS=""

if [ -x "$PROJECT_ROOT/build/bin/vc_grow_seg_from_seed" ]; then
    VC_GSFS="$PROJECT_ROOT/build/bin/vc_grow_seg_from_seed"
elif [ -x "$PROJECT_ROOT/cmake-build-debug/bin/vc_grow_seg_from_seed" ]; then
    VC_GSFS="$PROJECT_ROOT/cmake-build-debug/bin/vc_grow_seg_from_seed"
elif command -v vc_grow_seg_from_seed &> /dev/null; then
    VC_GSFS="vc_grow_seg_from_seed"
else
    echo "Error: vc_grow_seg_from_seed executable not found."
    echo "Build the project or ensure vc_grow_seg_from_seed is on PATH."
    exit 1
fi

TMP_PARAMS="$(mktemp)"
trap 'rm -f "$TMP_PARAMS"' EXIT

python3 - <<PY
import json, pathlib, sys
params_path = pathlib.Path("$PARAMS_JSON")
with params_path.open() as f:
    params = json.load(f)
params["generations"] = 0
with open("$TMP_PARAMS", "w") as f:
    json.dump(params, f, indent=2)
PY

echo "Using vc_grow_seg_from_seed: $VC_GSFS"
echo "Volume: $VOLUME_PATH"
echo "Input folder: $INPUT_FOLDER"
echo "Params JSON (with generations forced to 0): $TMP_PARAMS"
if [ -n "$OUTPUT_ROOT" ]; then
    echo "Output root: $OUTPUT_ROOT"
else
    echo "Output root: <segment_dir>_inpaint"
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    echo "Extra arguments: ${EXTRA_ARGS[*]}"
fi
echo ""

count=0
processed=0
failed=0
skipped=0
skipped_holes=0

while IFS= read -r -d '' dir; do
    if [ -f "$dir/x.tif" ] && [ -f "$dir/y.tif" ] && [ -f "$dir/z.tif" ]; then
        count=$((count + 1))
        folder_name=$(basename "$dir")

        if [ -n "$OUTPUT_ROOT" ]; then
            target_dir="$OUTPUT_ROOT/$folder_name"
        else
            target_dir="${dir}_inpaint"
        fi

        # Skip if output already exists
        if [ -f "$target_dir/x.tif" ] && [ -f "$target_dir/y.tif" ] && [ -f "$target_dir/z.tif" ]; then
            echo "[$count] Skipping (output exists): $dir"
            skipped=$((skipped + 1))
            continue
        fi

        mkdir -p "$target_dir"

        # Get segment name from target directory basename
        segment_name=$(basename "$target_dir")

        echo "[$count] Inpainting: $dir"
        echo "    Target dir: $target_dir"
        echo "    Segment name: $segment_name"

        # Run with output monitoring to detect too many holes relative to points
        status_file=$(mktemp)
        echo "OK" > "$status_file"
        num_points=0

        # Start the process and monitor output (disable pipefail for this section)
        set +o pipefail
        "$VC_GSFS" \
            --volume "$VOLUME_PATH" \
            --target-dir "$target_dir" \
            --params "$TMP_PARAMS" \
            --resume "$dir" \
            --inpaint \
            --segment-name "$segment_name" \
            "${EXTRA_ARGS[@]}" 2>&1 | while IFS= read -r line; do
            echo "$line"
            # Parse: "Resuming from generation X with Y points."
            if [[ "$line" =~ "with "([0-9]+)" points" ]]; then
                echo "${BASH_REMATCH[1]}" > "${status_file}.points"
            fi
            # Parse: "performing inpaint on Z potential holes"
            if [[ "$line" =~ "performing inpaint on "([0-9]+)" potential holes" ]]; then
                num_holes="${BASH_REMATCH[1]}"
                if [ -f "${status_file}.points" ]; then
                    num_points=$(cat "${status_file}.points")
                    threshold=$((num_points / 2))
                    if [ "$num_holes" -ge "$threshold" ]; then
                        echo "    ⚠ Too many holes ($num_holes) relative to points ($num_points), skipping..."
                        pkill -f "vc_grow_seg_from_seed.*--target-dir $target_dir" 2>/dev/null || true
                        echo "KILLED" > "$status_file"
                        break
                    fi
                fi
            fi
        done
        set -o pipefail

        final_status=$(cat "$status_file")
        rm -f "$status_file" "${status_file}.points"

        if [ "$final_status" = "KILLED" ]; then
            skipped_holes=$((skipped_holes + 1))
            # Clean up partial output
            rm -rf "$target_dir"
            echo "    ⚠ Skipped (too many holes)"
        elif [ -f "$target_dir/x.tif" ] && [ -f "$target_dir/y.tif" ] && [ -f "$target_dir/z.tif" ]; then
            processed=$((processed + 1))
            # Move original to inpaint_completed folder
            completed_dir="$INPUT_FOLDER/inpaint_completed"
            mkdir -p "$completed_dir"
            mv "$dir" "$completed_dir/"
            echo "    ✓ Success (moved original to inpaint_completed)"
        else
            failed=$((failed + 1))
            echo "    ✗ Failed"
        fi
        echo ""
    fi
done < <(find "$INPUT_FOLDER" -mindepth 1 -maxdepth 1 -type d ! -name '*_inpaint' ! -name 'inpaint_completed' -print0)

echo "======================================"
echo "vc_grow_seg_from_seed inpaint batch complete"
echo "Total candidates: $count"
echo "Skipped (already done): $skipped"
echo "Skipped (too many holes): $skipped_holes"
echo "Succeeded: $processed"
echo "Failed: $failed"
echo "======================================"

if [ "$count" -eq 0 ]; then
    echo "Warning: No tifxyz directories found in $INPUT_FOLDER"
    exit 1
fi

if [ "$failed" -gt 0 ]; then
    exit 1
fi

exit 0
