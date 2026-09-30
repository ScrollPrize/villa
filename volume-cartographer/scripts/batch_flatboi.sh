#!/bin/bash

# Batch SLIM-flatten (flatboi) a folder of tifxyz patches.
#
# Mirrors VC3D's "SLIM-flatten" action with Keep = 100% (no decimation, so no
# coarse mesh / UV lift step):
#   vc_tifxyz2obj <patch> <work>/<name>.obj
#   flatboi <work>/<name>.obj <iters> <energy> [--tol=<x>]
#   vc_obj2tifxyz <work>/<name>_flatboi.obj <out>/<name> --tifxyz-source=<patch>
#   meta.json "scale" := [0.05, 0.05]
# vc_obj2tifxyz recomputes area_vx2/area_cm2 for the flattened grid; pass
# --voxel-size if the source meta.json has no area_cm2 to recover it from.
# Intermediate OBJs go to a temp dir, so input patches are never modified.
#
# Usage: batch_flatboi.sh <input_folder> <output_folder> [options]

set -e

ITERS=50
ENERGY="symmetric_dirichlet"
TOL=0
INPAINT=0
OVERWRITE=0
KEEP_WORK=0
VOXEL_SIZE=""

usage() {
    echo "Usage: $0 <input_folder> <output_folder> [options]"
    echo ""
    echo "Flattens every tifxyz folder directly under <input_folder> with flatboi"
    echo "at full resolution and writes <output_folder>/<patch_name> for each one."
    echo ""
    echo "Options:"
    echo "  --iters N       SLIM iterations (default $ITERS)"
    echo "  --energy E      symmetric_dirichlet (default) or conformal"
    echo "  --tol X         relative-energy early stop (default 0 = run all iters)"
    echo "  --inpaint       fill interior holes before building the OBJ"
    echo "  --voxel-size U  micrometers per voxel for area_cm2 (default: recovered"
    echo "                  from each source meta.json's area_cm2/area_vx2)"
    echo "  --overwrite     replace existing outputs (default: skip them)"
    echo "  --keep-work     keep intermediate OBJs/heatmaps in <output_folder>/_work"
    echo ""
    echo "Environment: PASTIX_NUM_THREADS / OMP_NUM_THREADS are passed through."
}

if [ "$#" -lt 2 ]; then
    usage
    exit 1
fi

INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"
shift 2

while [ "$#" -gt 0 ]; do
    case "$1" in
        --iters)     ITERS="$2"; shift 2 ;;
        --energy)    ENERGY="$2"; shift 2 ;;
        --tol)       TOL="$2"; shift 2 ;;
        --inpaint)   INPAINT=1; shift ;;
        --voxel-size) VOXEL_SIZE="$2"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --keep-work) KEEP_WORK=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "Error: unknown option: $1"; usage; exit 1 ;;
    esac
done

if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder does not exist: $INPUT_FOLDER"
    exit 1
fi

mkdir -p "$OUTPUT_FOLDER"
INPUT_FOLDER="$(cd "$INPUT_FOLDER" && pwd)"
OUTPUT_FOLDER="$(cd "$OUTPUT_FOLDER" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

find_tool() {
    local name="$1"
    if [ -x "$PROJECT_ROOT/build/bin/$name" ]; then
        echo "$PROJECT_ROOT/build/bin/$name"
    elif [ -x "$PROJECT_ROOT/cmake-build-debug/bin/$name" ]; then
        echo "$PROJECT_ROOT/cmake-build-debug/bin/$name"
    elif command -v "$name" &> /dev/null; then
        command -v "$name"
    else
        echo "Error: $name executable not found." >&2
        echo "Please build the project first or ensure $name is in your PATH." >&2
        exit 1
    fi
}

FLATBOI="${FLATBOI:-$(find_tool flatboi)}"
VC_TIFXYZ2OBJ="$(find_tool vc_tifxyz2obj)"
VC_OBJ2TIFXYZ="$(find_tool vc_obj2tifxyz)"

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required to update meta.json."
    exit 1
fi

if [ "$KEEP_WORK" -eq 1 ]; then
    WORK_ROOT="$OUTPUT_FOLDER/_work"
    mkdir -p "$WORK_ROOT"
else
    WORK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/batch_flatboi.XXXXXX")"
    trap 'rm -rf "$WORK_ROOT"' EXIT
fi

FLATBOI_ARGS=("$ITERS" "$ENERGY")
if [ "$TOL" != "0" ]; then
    FLATBOI_ARGS+=("--tol=$TOL")
fi
OBJ2TIFXYZ_ARGS=()
if [ -n "$VOXEL_SIZE" ]; then
    OBJ2TIFXYZ_ARGS+=("--voxel-size=$VOXEL_SIZE")
fi
TIFXYZ2OBJ_ARGS=()
if [ "$INPAINT" -eq 1 ]; then
    TIFXYZ2OBJ_ARGS+=("--inpaint")
fi

echo "Using flatboi: $FLATBOI"
echo "Using vc_tifxyz2obj: $VC_TIFXYZ2OBJ"
echo "Using vc_obj2tifxyz: $VC_OBJ2TIFXYZ"
echo "Input folder: $INPUT_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "Iterations: $ITERS  energy: $ENERGY  tol: $TOL  inpaint: $INPAINT  voxel size: ${VOXEL_SIZE:-<from source meta>}"
echo ""

# Same meta.json fixup VC3D does after vc_obj2tifxyz (overwriteMetaScale_).
set_meta_scale() {
    python3 - "$1" <<'EOF'
import json, os, sys, uuid
d = sys.argv[1]
p = os.path.join(d, "meta.json")
root = {}
if os.path.exists(p):
    try:
        with open(p) as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            root = loaded
    except ValueError:
        pass
root["scale"] = [0.05, 0.05]
root.setdefault("type", "seg")
root.setdefault("uuid", str(uuid.uuid4()))  # new surface: random, never the dir name
root.setdefault("format", "tifxyz")
with open(p, "w") as f:
    json.dump(root, f, indent=4)
EOF
}

count=0
processed=0
skipped=0
failed=0
failed_names=()

flatten_one() {
    local dir="$1" name="$2" out="$3"
    local work="$WORK_ROOT/$name"
    rm -rf "$work"
    mkdir -p "$work"

    "$VC_TIFXYZ2OBJ" "$dir" "$work/$name.obj" "${TIFXYZ2OBJ_ARGS[@]}" || return 1
    [ -f "$work/$name.obj" ] || return 1

    "$FLATBOI" "$work/$name.obj" "${FLATBOI_ARGS[@]}" || return 1
    [ -f "$work/${name}_flatboi.obj" ] || return 1

    # vc_obj2tifxyz requires the target directory not to exist.
    rm -rf "$out"
    "$VC_OBJ2TIFXYZ" "$work/${name}_flatboi.obj" "$out" "--tifxyz-source=$dir" "${OBJ2TIFXYZ_ARGS[@]}" || return 1
    [ -d "$out" ] || return 1

    set_meta_scale "$out" || return 1

    if [ "$KEEP_WORK" -ne 1 ]; then
        rm -rf "$work"
    fi
}

while IFS= read -r -d '' dir; do
    if [ -f "$dir/x.tif" ] && [ -f "$dir/y.tif" ] && [ -f "$dir/z.tif" ]; then
        count=$((count + 1))
        name="$(basename "$dir")"
        out="$OUTPUT_FOLDER/$name"

        echo "[$count] Processing: $dir"
        echo "    Output: $out"

        if [ -e "$out" ] && [ "$OVERWRITE" -ne 1 ]; then
            skipped=$((skipped + 1))
            echo "    - Skipped (output exists; use --overwrite)"
            echo ""
            continue
        fi

        if flatten_one "$dir" "$name" "$out"; then
            processed=$((processed + 1))
            echo "    ✓ Success"
        else
            failed=$((failed + 1))
            failed_names+=("$name")
            rm -rf "$out"
            echo "    ✗ Failed"
        fi
        echo ""
    fi
done < <(find "$INPUT_FOLDER" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

echo "======================================"
echo "Batch flatboi complete!"
echo "Total found: $count"
echo "Successfully processed: $processed"
echo "Skipped (existing): $skipped"
echo "Failed: $failed"
for n in "${failed_names[@]}"; do
    echo "  - $n"
done
echo "======================================"

if [ "$count" -eq 0 ]; then
    echo "Warning: No tifxyz folders found in $INPUT_FOLDER"
    exit 1
fi

if [ "$failed" -gt 0 ]; then
    exit 1
fi

exit 0
