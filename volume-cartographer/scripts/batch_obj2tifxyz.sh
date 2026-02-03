#!/bin/bash

# Script to batch convert OBJ files to tifxyz format
# Usage: batch_obj2tifxyz.sh <input_folder> <output_folder> [step_size]

set -e

# Check for minimum required arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder> [step_size]"
    echo ""
    echo "This script will find all .obj files in the input folder and convert them to tifxyz format."
    echo ""
    echo "Arguments:"
    echo "  input_folder  : Folder containing .obj files"
    echo "  output_folder : Folder where tifxyz subfolders will be created"
    echo "  step_size     : UV units per grid cell (default: 20)"
    echo ""
    echo "Example: $0 /path/to/objs /path/to/output 20"
    exit 1
fi

INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"
STEP_SIZE="${3:-}"

# Validate input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder does not exist: $INPUT_FOLDER"
    exit 1
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Find the vc_obj2tifxyz_legacy executable
# Check common locations
VC_OBJ2TIFXYZ=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -x "$PROJECT_ROOT/build/bin/vc_obj2tifxyz_legacy" ]; then
    VC_OBJ2TIFXYZ="$PROJECT_ROOT/build/bin/vc_obj2tifxyz_legacy"
elif [ -x "$PROJECT_ROOT/cmake-build-debug/bin/vc_obj2tifxyz_legacy" ]; then
    VC_OBJ2TIFXYZ="$PROJECT_ROOT/cmake-build-debug/bin/vc_obj2tifxyz_legacy"
elif command -v vc_obj2tifxyz_legacy &> /dev/null; then
    VC_OBJ2TIFXYZ="vc_obj2tifxyz_legacy"
else
    echo "Error: vc_obj2tifxyz_legacy executable not found."
    echo "Please build the project first or ensure vc_obj2tifxyz_legacy is in your PATH."
    exit 1
fi

echo "Using vc_obj2tifxyz_legacy: $VC_OBJ2TIFXYZ"
echo "Input folder: $INPUT_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
if [ -n "$STEP_SIZE" ]; then
    echo "Step size: $STEP_SIZE"
fi
echo ""

# Find all .obj files in the input folder (immediate children only)
count=0
processed=0
failed=0

while IFS= read -r -d '' obj_file; do
    count=$((count + 1))

    # Get the base name without extension
    base_name=$(basename "$obj_file" .obj)

    # Construct output folder path
    output_dir="$OUTPUT_FOLDER/$base_name"

    echo "[$count] Processing: $obj_file"
    echo "    Output: $output_dir"

    # Build command arguments
    CMD_ARGS=("$obj_file" "$output_dir")
    if [ -n "$STEP_SIZE" ]; then
        CMD_ARGS+=("$STEP_SIZE")
    fi

    # Run vc_obj2tifxyz_legacy
    if "$VC_OBJ2TIFXYZ" "${CMD_ARGS[@]}"; then
        processed=$((processed + 1))
        echo "    ✓ Success"
    else
        failed=$((failed + 1))
        echo "    ✗ Failed"
    fi
    echo ""
done < <(find "$INPUT_FOLDER" -mindepth 1 -maxdepth 1 -type f -name "*.obj" -print0)

echo "======================================"
echo "Batch processing complete!"
echo "Total found: $count"
echo "Successfully processed: $processed"
echo "Failed: $failed"
echo "======================================"

if [ "$count" -eq 0 ]; then
    echo "Warning: No .obj files found in $INPUT_FOLDER"
    exit 1
fi

if [ "$failed" -gt 0 ]; then
    exit 1
fi

exit 0
