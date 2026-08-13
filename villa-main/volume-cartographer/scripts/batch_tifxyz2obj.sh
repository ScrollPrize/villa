#!/bin/bash

# Script to batch process tifxyz folders to OBJ files
# Usage: batch_tifxyz2obj.sh <input_folder> <output_folder> [additional args]

set -e

# Check for minimum required arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder> [additional args for vc_tifxyz2obj]"
    echo ""
    echo "This script will find all *tifxyz folders in the input folder and convert them to OBJ files."
    echo ""
    echo "Additional arguments (optional) will be passed to vc_tifxyz2obj:"
    echo "  --normalize-uv : Normalize UVs to [0,1] range"
    echo "  --align-grid   : Align grid Z only (flatten Z per row)"
    echo "  --decimate [n] : Reduce points by ~90% per iteration (default n=1)"
    echo "  --clean [K]    : Remove outlier points; K is sigma multiplier (default 5.0)"
    echo ""
    echo "Example: $0 /path/to/input /path/to/output --normalize-uv --decimate 1"
    exit 1
fi

INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"
shift 2
EXTRA_ARGS="$@"

# Validate input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder does not exist: $INPUT_FOLDER"
    exit 1
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Find the vc_tifxyz2obj executable
# Check common locations
VC_TIFXYZ2OBJ=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -x "$PROJECT_ROOT/build/bin/vc_tifxyz2obj" ]; then
    VC_TIFXYZ2OBJ="$PROJECT_ROOT/build/bin/vc_tifxyz2obj"
elif [ -x "$PROJECT_ROOT/cmake-build-debug/bin/vc_tifxyz2obj" ]; then
    VC_TIFXYZ2OBJ="$PROJECT_ROOT/cmake-build-debug/bin/vc_tifxyz2obj"
elif command -v vc_tifxyz2obj &> /dev/null; then
    VC_TIFXYZ2OBJ="vc_tifxyz2obj"
else
    echo "Error: vc_tifxyz2obj executable not found."
    echo "Please build the project first or ensure vc_tifxyz2obj is in your PATH."
    exit 1
fi

echo "Using vc_tifxyz2obj: $VC_TIFXYZ2OBJ"
echo "Input folder: $INPUT_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra arguments: $EXTRA_ARGS"
fi
echo ""

# Find all immediate subdirectories containing x.tif, y.tif, and z.tif (tifxyz folders)
count=0
processed=0
failed=0

while IFS= read -r -d '' dir; do
    # Check if this directory contains x.tif, y.tif, and z.tif
    if [ -f "$dir/x.tif" ] && [ -f "$dir/y.tif" ] && [ -f "$dir/z.tif" ]; then
        count=$((count + 1))

        # Get the base name of the folder
        folder_name=$(basename "$dir")

        # Construct output OBJ file path
        output_obj="$OUTPUT_FOLDER/${folder_name}.obj"

        echo "[$count] Processing: $dir"
        echo "    Output: $output_obj"

        # Run vc_tifxyz2obj
        if $VC_TIFXYZ2OBJ "$dir" "$output_obj" $EXTRA_ARGS; then
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
echo "Batch processing complete!"
echo "Total found: $count"
echo "Successfully processed: $processed"
echo "Failed: $failed"
echo "======================================"

if [ "$count" -eq 0 ]; then
    echo "Warning: No tifxyz folders found in $INPUT_FOLDER"
    exit 1
fi

if [ "$failed" -gt 0 ]; then
    exit 1
fi

exit 0
