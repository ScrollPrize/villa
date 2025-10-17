#!/bin/bash

# Batch alpha-comp refinement for tifxyz surfaces.
# Usage: batch_objrefine.sh <zarr_volume> <input_folder> <params_json> [output_folder]
#
# For every immediate subdirectory in <input_folder> that looks like a tifxyz
# (contains x.tif, y.tif, z.tif), this script runs:
#   vc_objrefine <zarr_volume> <segment_dir> <output_path> <params_json>
# The output path is <segment_dir>_refined by default, or
# <output_folder>/<segment_name>_refined when an explicit output root is given.

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <zarr_volume> <input_folder> <params_json> [output_folder]"
    echo ""
    echo "  <zarr_volume>  : Path to the OME-Zarr volume used for refinement."
    echo "  <input_folder> : Directory containing tifxyz segment folders (one level deep)."
    echo "  <params_json>  : JSON file with vc_objrefine parameters."
    echo "  [output_folder]: Optional destination root; defaults to <segment_dir>_refined."
    echo ""
    echo "Example:"
    echo "  $0 /data/volumes/PHerc.volpkg/zarr/1.zarr /data/segments params.json /data/refined"
    exit 1
fi

VOLUME_PATH="$1"
INPUT_FOLDER="$2"
PARAMS_JSON="$3"
OUTPUT_ROOT="${4:-}"

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
VC_OBJREFINE=""

if [ -x "$PROJECT_ROOT/build/bin/vc_objrefine" ]; then
    VC_OBJREFINE="$PROJECT_ROOT/build/bin/vc_objrefine"
elif [ -x "$PROJECT_ROOT/cmake-build-debug/bin/vc_objrefine" ]; then
    VC_OBJREFINE="$PROJECT_ROOT/cmake-build-debug/bin/vc_objrefine"
elif command -v vc_objrefine &> /dev/null; then
    VC_OBJREFINE="vc_objrefine"
else
    echo "Error: vc_objrefine executable not found."
    echo "Build the project or ensure vc_objrefine is on PATH."
    exit 1
fi

echo "Using vc_objrefine: $VC_OBJREFINE"
echo "Volume: $VOLUME_PATH"
echo "Input folder: $INPUT_FOLDER"
echo "Params JSON: $PARAMS_JSON"
if [ -n "$OUTPUT_ROOT" ]; then
    echo "Output root: $OUTPUT_ROOT"
else
    echo "Output root: <segment_dir>_refined"
fi
echo ""

count=0
processed=0
failed=0

while IFS= read -r -d '' dir; do
    if [ -f "$dir/x.tif" ] && [ -f "$dir/y.tif" ] && [ -f "$dir/z.tif" ]; then
        count=$((count + 1))
        folder_name=$(basename "$dir")

        if [ -n "$OUTPUT_ROOT" ]; then
            out_path="$OUTPUT_ROOT/${folder_name}_refined"
        else
            out_path="${dir}_refined"
        fi

        echo "[$count] Refining: $dir"
        echo "    Output: $out_path"

        if "$VC_OBJREFINE" "$VOLUME_PATH" "$dir" "$out_path" "$PARAMS_JSON"; then
            processed=$((processed + 1))
            echo "    ✓ Success"
        else
            failed=$((failed + 1))
            echo "    ✗ Failed"
        fi
        echo ""
    fi
done < <(find "$INPUT_FOLDER" -mindepth 1 -maxdepth 1 -type d -print0)

echo "======================================"
echo "vc_objrefine batch complete"
echo "Total candidates: $count"
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
