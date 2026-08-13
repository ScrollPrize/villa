#!/bin/bash

# Script to batch process tifxyz folders to add overlap metadata
# Usage: batch_seg_add_overlap.sh <segments_to_process> <segments_to_check_against>

set -e

# Check for minimum required arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <segments_to_process> <segments_to_check_against>"
    echo ""
    echo "This script will find all tifxyz folders in <segments_to_process> and check"
    echo "each one for overlaps against all tifxyz folders in <segments_to_check_against>."
    echo ""
    echo "Example: $0 /path/to/hold /path/to/paths"
    echo "  - Creates overlapping.json for each segment in 'hold'"
    echo "  - By checking against all segments in 'paths'"
    exit 1
fi

INPUT_FOLDER="$1"
TARGET_FOLDER="$2"

# Validate folders exist
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder does not exist: $INPUT_FOLDER"
    exit 1
fi

if [ ! -d "$TARGET_FOLDER" ]; then
    echo "Error: Target folder does not exist: $TARGET_FOLDER"
    exit 1
fi

# Find the vc_seg_add_overlap executable
VC_SEG_ADD_OVERLAP=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -x "$PROJECT_ROOT/build/bin/vc_seg_add_overlap" ]; then
    VC_SEG_ADD_OVERLAP="$PROJECT_ROOT/build/bin/vc_seg_add_overlap"
elif [ -x "$PROJECT_ROOT/cmake-build-debug/bin/vc_seg_add_overlap" ]; then
    VC_SEG_ADD_OVERLAP="$PROJECT_ROOT/cmake-build-debug/bin/vc_seg_add_overlap"
elif command -v vc_seg_add_overlap &> /dev/null; then
    VC_SEG_ADD_OVERLAP="vc_seg_add_overlap"
else
    echo "Error: vc_seg_add_overlap executable not found."
    echo "Please build the project first or ensure vc_seg_add_overlap is in your PATH."
    exit 1
fi

echo "Using vc_seg_add_overlap: $VC_SEG_ADD_OVERLAP"
echo "Segments to process: $INPUT_FOLDER"
echo "Check against: $TARGET_FOLDER"
echo ""

# Find all immediate subdirectories containing x.tif, y.tif, and z.tif (tifxyz folders)
count=0
processed=0
failed=0

while IFS= read -r -d '' dir; do
    # Check if this directory contains x.tif, y.tif, and z.tif
    if [ -f "$dir/x.tif" ] && [ -f "$dir/y.tif" ] && [ -f "$dir/z.tif" ]; then
        count=$((count + 1))

        folder_name=$(basename "$dir")

        echo "[$count] Processing: $folder_name"

        # Run vc_seg_add_overlap: check this segment against all in target folder
        if $VC_SEG_ADD_OVERLAP "$TARGET_FOLDER" "$dir"; then
            processed=$((processed + 1))
            echo "    Done"
        else
            failed=$((failed + 1))
            echo "    Failed"
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
