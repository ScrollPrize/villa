#!/bin/bash

# Batch alpha-comp refinement for tifxyz surfaces or OBJ meshes.
# Usage: batch_objrefine.sh <zarr_volume> <input_folder> <params_json> [output_folder] [extra vc_objrefine args...]
#
# For every immediate subdirectory in <input_folder> that looks like a tifxyz
# (contains x.tif, y.tif, z.tif) or every OBJ file found directly under the folder,
# this script runs:
#   vc_objrefine <zarr_volume> <segment_dir> <output_path> <params_json>
# The output path defaults to <segment_dir>_refined (for tifxyz) or
# <segment>_refined.obj (for OBJ). When an explicit output root is supplied, the
# refined data is written beneath it.

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <zarr_volume> <input_folder> <params_json> [output_folder] [extra vc_objrefine args...]"
    echo ""
    echo "  <zarr_volume>  : Path to the OME-Zarr volume used for refinement."
    echo "  <input_folder> : Directory containing tifxyz folders or OBJ files (one level deep)."
    echo "  <params_json>  : JSON file with vc_objrefine parameters."
    echo "  [output_folder]: Optional destination root; defaults to <segment>_refined(_refined.obj)."
    echo "  [extra args...]: Optional extra flags passed through to vc_objrefine."
    echo ""
    echo "Example:"
    echo "  $0 /data/volumes/vol.zarr /data/segments params.json /data/refined"
    echo "  $0 /data/volumes/vol.zarr /data/meshes params.json --cache_gb 2"
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
    echo "Output root: <segment>_refined (folder or OBJ)"
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    echo "Extra arguments: ${EXTRA_ARGS[*]}"
fi
echo ""

count=0
processed=0
failed=0

while IFS= read -r -d '' path; do
    if [ -d "$path" ] && [ -f "$path/x.tif" ] && [ -f "$path/y.tif" ] && [ -f "$path/z.tif" ]; then
        count=$((count + 1))
        folder_name=$(basename "$path")
        if [ -n "$OUTPUT_ROOT" ]; then
            out_path="$OUTPUT_ROOT/${folder_name}_refined"
        else
            out_path="${path}_refined"
        fi

        echo "[$count] Refining (tifxyz): $path"
        echo "    Output: $out_path"

        if "$VC_OBJREFINE" "$VOLUME_PATH" "$path" "$out_path" "$PARAMS_JSON" "${EXTRA_ARGS[@]}"; then
            processed=$((processed + 1))
            echo "    ✓ Success"
        else
            failed=$((failed + 1))
            echo "    ✗ Failed"
        fi
        echo ""
    elif [ -f "$path" ] && ([[ "$path" == *.obj ]] || [[ "$path" == *.OBJ ]]); then
        count=$((count + 1))
        file_name=$(basename "$path")
        base_name="${file_name%.*}"
        if [ -n "$OUTPUT_ROOT" ]; then
            mkdir -p "$OUTPUT_ROOT"
            out_path="$OUTPUT_ROOT/${base_name}_refined.obj"
        else
            out_path="$(dirname "$path")/${base_name}_refined.obj"
        fi

        echo "[$count] Refining (OBJ): $path"
        echo "    Output: $out_path"

        if "$VC_OBJREFINE" "$VOLUME_PATH" "$path" "$out_path" "$PARAMS_JSON" "${EXTRA_ARGS[@]}"; then
            processed=$((processed + 1))
            echo "    ✓ Success"
        else
            failed=$((failed + 1))
            echo "    ✗ Failed"
        fi
        echo ""
    fi
done < <(find "$INPUT_FOLDER" -mindepth 1 -maxdepth 1 \( -type d -o -type f \) -print0)

echo "======================================"
echo "vc_objrefine batch complete"
echo "Total candidates: $count"
echo "Succeeded: $processed"
echo "Failed: $failed"
echo "======================================"

if [ "$count" -eq 0 ]; then
    echo "Warning: No tifxyz directories or OBJ files found in $INPUT_FOLDER"
    exit 1
fi

if [ "$failed" -gt 0 ]; then
    exit 1
fi

exit 0
