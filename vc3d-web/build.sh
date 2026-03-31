#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# Check for Emscripten
if ! command -v emcmake &>/dev/null; then
    echo "ERROR: emcmake not found."
    echo ""
    echo "Install Emscripten SDK:"
    echo "  git clone https://github.com/emscripten-core/emsdk.git"
    echo "  cd emsdk && ./emsdk install latest && ./emsdk activate latest"
    echo "  source emsdk_env.sh"
    exit 1
fi

echo "=== Building vc3d-web with Emscripten ==="

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

emcmake cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release
emmake make -j$(nproc 2>/dev/null || echo 4)

echo ""
echo "=== Build complete ==="
echo "Output files:"
echo "  $BUILD_DIR/vc3d-web.js"
echo "  $BUILD_DIR/vc3d-web.wasm"
echo ""
echo "To run locally:"
echo "  cp $SCRIPT_DIR/index.html $BUILD_DIR/"
echo "  cd $BUILD_DIR && python3 -m http.server 8080"
echo "  Open http://localhost:8080"
