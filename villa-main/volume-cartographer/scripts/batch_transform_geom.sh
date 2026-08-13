#!/bin/bash

# Batch transform geometry for tifxyz surfaces.
# Usage: batch_transform_geom.sh <input_folder> <output_folder> --scale-segmentation <factor> [--affine <json>] [--invert]
#
# For every immediate subdirectory in <input_folder> that looks like a tifxyz
# (contains x.tif, y.tif, z.tif), this script runs:
#   vc_transform_geom -i <input_dir> -o <output_dir> --scale-segmentation <factor> [--affine <json>] [--invert]

set -euo pipefail

usage() {
    echo "Usage: $0 <input_folder> <output_folder> --scale-segmentation <factor> [--affine <json>] [--invert]"
    echo ""
    echo "  <input_folder>            : Directory containing tifxyz folders (one level deep)."
    echo "  <output_folder>           : Destination root for transformed outputs."
    echo "  --scale-segmentation <n>  : Scale factor to apply (e.g., 4.0 for level 2 -> level 0)."
    echo "  --affine <json>           : Optional path to affine transform JSON file."
    echo "  --invert                  : Optional flag to invert the affine transform."
    echo ""
    echo "Each input '<name>/' becomes '<output_folder>/<name>/'."
    echo ""
    echo "Examples:"
    echo "  $0 /data/segments /data/scaled --scale-segmentation 4.0"
    echo "  $0 /data/segments /data/transformed --scale-segmentation 2.0 --affine transform.json"
    echo "  $0 /data/segments /data/transformed --scale-segmentation 1.0 --affine transform.json --invert"
    exit 1
}

# Check for minimum required arguments
if [ "$#" -lt 4 ]; then
    usage
fi

INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"
shift 2

# Parse remaining arguments
SCALE=""
AFFINE=""
INVERT=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --scale-segmentation)
            if [ "$#" -lt 2 ]; then
                echo "Error: --scale-segmentation requires a value"
                exit 1
            fi
            SCALE="$2"
            shift 2
            ;;
        --affine)
            if [ "$#" -lt 2 ]; then
                echo "Error: --affine requires a value"
                exit 1
            fi
            AFFINE="$2"
            shift 2
            ;;
        --invert)
            INVERT="--invert"
            shift
            ;;
        *)
            echo "Error: Unknown argument: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$SCALE" ]; then
    echo "Error: --scale-segmentation is required"
    usage
fi

# Validate input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder does not exist: $INPUT_FOLDER"
    exit 1
fi

# Validate affine file if provided
if [ -n "$AFFINE" ] && [ ! -f "$AFFINE" ]; then
    echo "Error: Affine JSON file not found: $AFFINE"
    exit 1
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Find the vc_transform_geom executable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VC_TRANSFORM_GEOM=""

if [ -x "$PROJECT_ROOT/build/bin/vc_transform_geom" ]; then
    VC_TRANSFORM_GEOM="$PROJECT_ROOT/build/bin/vc_transform_geom"
elif [ -x "$PROJECT_ROOT/cmake-build-debug/bin/vc_transform_geom" ]; then
    VC_TRANSFORM_GEOM="$PROJECT_ROOT/cmake-build-debug/bin/vc_transform_geom"
elif command -v vc_transform_geom &> /dev/null; then
    VC_TRANSFORM_GEOM="vc_transform_geom"
else
    echo "Error: vc_transform_geom executable not found."
    echo "Build the project or ensure vc_transform_geom is on PATH."
    exit 1
fi

echo "Using vc_transform_geom: $VC_TRANSFORM_GEOM"
echo "Input folder: $INPUT_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "Scale segmentation: $SCALE"
if [ -n "$AFFINE" ]; then
    echo "Affine JSON: $AFFINE"
fi
if [ -n "$INVERT" ]; then
    echo "Invert: yes"
fi
echo ""

count=0
processed=0
failed=0

while IFS= read -r -d '' path; do
    if [ -d "$path" ] && [ -f "$path/x.tif" ] && [ -f "$path/y.tif" ] && [ -f "$path/z.tif" ]; then
        count=$((count + 1))
        folder_name=$(basename "$path")
        out_path="$OUTPUT_FOLDER/$folder_name"

        echo "[$count] Transforming: $path"
        echo "    Output: $out_path"

        # Build command arguments
        CMD_ARGS=("-i" "$path" "-o" "$out_path" "--scale-segmentation" "$SCALE")
        if [ -n "$AFFINE" ]; then
            CMD_ARGS+=("--affine" "$AFFINE")
        fi
        if [ -n "$INVERT" ]; then
            CMD_ARGS+=("$INVERT")
        fi

        if "$VC_TRANSFORM_GEOM" "${CMD_ARGS[@]}"; then
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
echo "vc_transform_geom batch complete"
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
