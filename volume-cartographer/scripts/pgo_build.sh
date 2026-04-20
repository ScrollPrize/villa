#!/usr/bin/env bash
# PGO (Profile-Guided Optimization) build script for VC3D.
#
# Usage:
#   ./scripts/pgo_build.sh <volume_path_or_url>
#
# This performs a 3-step build:
#   1. Build with profiling instrumentation (-fprofile-generate)
#   2. Run vc_render_bench to collect real-world profile data
#   3. Rebuild with profile-guided optimization (-fprofile-use)
#
# The final binaries will be in build-pgo/bin/.

set -euo pipefail

VOLUME="${1:?Usage: $0 <volume_path_or_url>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${SRC_DIR}/build-pgo"
PGO_DIR="/tmp/pgo"
NPROC="$(nproc)"

echo "=== PGO Build ==="
echo "  Source:  $SRC_DIR"
echo "  Build:   $BUILD_DIR"
echo "  Volume:  $VOLUME"
echo "  Cores:   $NPROC"
echo ""

# ---------- Step 1: Instrumented build ----------------------------------------
echo ">>> Step 1/3: Building with profiling instrumentation..."
rm -rf "$PGO_DIR"
mkdir -p "$PGO_DIR"

cmake -G Ninja -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVC_PGO=generate \
    -DVC_PGO_DIR="$PGO_DIR"

cmake --build "$BUILD_DIR" -j "$NPROC" --target vc_render_bench

# ---------- Step 2: Collect profile data --------------------------------------
echo ""
echo ">>> Step 2/3: Running vc_render_bench to collect profile data..."
"$BUILD_DIR/bin/vc_render_bench" "$VOLUME"

# Merge raw profiles (Clang produces .profraw files that must be merged)
if command -v llvm-profdata &>/dev/null; then
    echo "  Merging Clang profile data..."
    llvm-profdata merge -output="$PGO_DIR/merged.profdata" "$PGO_DIR"/*.profraw
elif ls "$PGO_DIR"/*.profraw &>/dev/null 2>&1; then
    # Try to find llvm-profdata with a version suffix
    PROFDATA="$(compgen -c llvm-profdata- 2>/dev/null | head -1 || true)"
    if [[ -n "$PROFDATA" ]]; then
        echo "  Merging Clang profile data with $PROFDATA..."
        "$PROFDATA" merge -output="$PGO_DIR/merged.profdata" "$PGO_DIR"/*.profraw
    else
        echo "ERROR: Found .profraw files but no llvm-profdata to merge them."
        echo "Install llvm or set PATH to include the LLVM bin directory."
        exit 1
    fi
fi

echo "  Profile data collected in $PGO_DIR"

# ---------- Step 3: Optimized rebuild -----------------------------------------
echo ""
echo ">>> Step 3/3: Rebuilding with profile-guided optimization..."
cmake -G Ninja -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVC_PGO=use \
    -DVC_PGO_DIR="$PGO_DIR"

cmake --build "$BUILD_DIR" -j "$NPROC"

echo ""
echo "=== PGO build complete ==="
echo "  Binaries: $BUILD_DIR/bin/"
echo ""
echo "  Verify with:"
echo "    $BUILD_DIR/bin/vc_render_bench $VOLUME"
