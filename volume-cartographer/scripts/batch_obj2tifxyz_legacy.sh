#!/bin/bash

# Batch convert OBJ meshes to tifxyz folders using vc_obj2tifxyz_legacy.
# Usage: batch_obj2tifxyz_legacy.sh <input_folder> <output_folder> [mesh_units] [uv_pixels_per_unit]

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder> [mesh_units] [uv_pixels_per_unit]"
    echo ""
    echo "  <input_folder> : Directory containing OBJ files (one level deep)."
    echo "  <output_folder>: Destination root where tifxyz subfolders will be created."
    echo "  mesh_units         : Mesh coordinate units in micrometers (default: 1.0)."
    echo "  uv_pixels_per_unit : UV pixel density (default: 20)."
    echo ""
    echo "Notes:"
    echo "  - vc_obj2tifxyz_legacy expects the per-mesh output directory to not already exist."
    echo "  - Optional parameters are positional; omit later values to use tool defaults."
    echo ""
    echo "Example:"
    echo "  $0 /data/objs /data/tifxyz 1.0 20"
    exit 1
fi

if [ "$#" -gt 4 ]; then
    echo "Error: Too many arguments. Only [mesh_units] [uv_pixels_per_unit] are optional."
    exit 1
fi

INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"

MESH_UNITS=""
UV_PIXELS=""

if [ "$#" -ge 3 ]; then
    MESH_UNITS="$3"
fi
if [ "$#" -ge 4 ]; then
    UV_PIXELS="$4"
fi

if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder does not exist: $INPUT_FOLDER"
    exit 1
fi

mkdir -p "$OUTPUT_FOLDER"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VC_OBJ2TIFXYZ=""

if [ -x "$PROJECT_ROOT/build/bin/vc_obj2tifxyz_legacy" ]; then
    VC_OBJ2TIFXYZ="$PROJECT_ROOT/build/bin/vc_obj2tifxyz_legacy"
elif [ -x "$PROJECT_ROOT/cmake-build-debug/bin/vc_obj2tifxyz_legacy" ]; then
    VC_OBJ2TIFXYZ="$PROJECT_ROOT/cmake-build-debug/bin/vc_obj2tifxyz_legacy"
elif command -v vc_obj2tifxyz_legacy &> /dev/null; then
    VC_OBJ2TIFXYZ="vc_obj2tifxyz_legacy"
else
    echo "Error: vc_obj2tifxyz_legacy executable not found."
    echo "Please build the project or ensure vc_obj2tifxyz_legacy is on your PATH."
    exit 1
fi

CONVERSION_ARGS=()
if [ -n "$MESH_UNITS" ]; then
    CONVERSION_ARGS+=("$MESH_UNITS")
fi
if [ -n "$UV_PIXELS" ]; then
    CONVERSION_ARGS+=("$UV_PIXELS")
fi

echo "Using vc_obj2tifxyz_legacy: $VC_OBJ2TIFXYZ"
echo "Input folder: $INPUT_FOLDER"
echo "Output root: $OUTPUT_FOLDER"
if [ "${#CONVERSION_ARGS[@]}" -gt 0 ]; then
    echo "Optional args (positional): ${CONVERSION_ARGS[*]}"
else
    echo "Optional args: defaults from vc_obj2tifxyz_legacy"
fi
echo ""

count=0
processed=0
failed=0
skipped_existing=0

while IFS= read -r -d '' obj_file; do
    count=$((count + 1))
    file_name="$(basename "$obj_file")"
    base_name="${file_name%.*}"
    output_dir="$OUTPUT_FOLDER/${base_name}_tifxyz"

    echo "[$count] Converting: $obj_file"
    echo "    Output dir: $output_dir"

    if [ -e "$output_dir" ]; then
        echo "    Skipping (output already exists)"
        skipped_existing=$((skipped_existing + 1))
        echo ""
        continue
    fi

    if "$VC_OBJ2TIFXYZ" "$obj_file" "$output_dir" "${CONVERSION_ARGS[@]}"; then
        processed=$((processed + 1))
        echo "    ✓ Success"
    else
        failed=$((failed + 1))
        echo "    ✗ Failed"
    fi
    echo ""
done < <(find "$INPUT_FOLDER" -mindepth 1 -maxdepth 1 -type f -iname "*.obj" -print0)

echo "======================================"
echo "vc_obj2tifxyz_legacy batch complete"
echo "Total OBJ files: $count"
echo "Succeeded: $processed"
echo "Failed: $failed"
echo "Skipped (existing output): $skipped_existing"
echo "======================================"

if [ "$count" -eq 0 ]; then
    echo "Warning: No OBJ files found in $INPUT_FOLDER"
    exit 1
fi

if [ "$failed" -gt 0 ]; then
    exit 1
fi

exit 0
