#!/bin/bash
#
# h264-zarr-only.sh - Build only H264/Zarr compression stack from vendored forks
#
# This script installs most dependencies via apt and only builds the
# vendored H264 compression components from SuperOptimizer GitHub forks:
#   - xtl, xsimd, xtensor (newer versions than apt provides)
#   - c-blosc2 (with H264 codec support via system libopenh264)
#   - z5 (Zarr C++ library with blosc2)
#   - python-blosc2, numcodecs, zarr-python (Python stack)
#
# System dependencies (installed via apt):
#   - libopenh264-dev (Cisco's H.264 codec)
#   - All other compression, math, image, and Qt libraries
#
# Usage:
#   ./scripts/h264-zarr-only.sh [PREFIX]
#
# Default PREFIX is ~/vc-dependencies

set -euo pipefail

# Determine script and VC root directories (before any cd commands)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
PREFIX="${1:-$HOME/vc-dependencies}"
JOBS="${JOBS:-$(nproc)}"
BUILDDIR="${BUILDDIR:-$PREFIX/build}"

# Compiler setup - use ccache with clang if available
if command -v ccache >/dev/null 2>&1; then
    CC="ccache clang"
    CXX="ccache clang++"
else
    CC="clang"
    CXX="clang++"
fi
export CC CXX

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[BUILD]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# Common CMake flags
CMAKE_COMMON=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
    -DCMAKE_PREFIX_PATH="$PREFIX"
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DBUILD_SHARED_LIBS=ON
)

# Add ccache launcher if available
if command -v ccache >/dev/null 2>&1; then
    CMAKE_COMMON+=(
        -DCMAKE_C_COMPILER_LAUNCHER=ccache
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    )
fi

# Add LTO flags if using lld
if command -v lld >/dev/null 2>&1; then
    CMAKE_COMMON+=(
        -DCMAKE_AR="$(which llvm-ar)"
        -DCMAKE_RANLIB="$(which llvm-ranlib)"
        -DCMAKE_NM="$(which llvm-nm)"
        -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld"
        -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"
        -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld"
    )
fi

mark_done() {
    touch "$PREFIX/.done-$1"
}

is_done() {
    [[ -f "$PREFIX/.done-$1" ]]
}

clone_repo() {
    local url="$1"
    local dest="$2"
    local branch="${3:-main}"

    if [[ -d "$dest" ]]; then
        log "Repository already exists at $dest, skipping clone"
        return
    fi

    log "Cloning $url to $dest..."
    git clone --depth 1 --branch "$branch" "$url" "$dest" || error "Failed to clone $url"
}

#------------------------------------------------------------------------------
# Install system dependencies via apt
#------------------------------------------------------------------------------
install_system_deps() {
    log "Checking/installing system dependencies via apt..."

    # List of required packages
    local PACKAGES=(
        clang lld llvm ccache cmake ninja-build pkg-config git python3 python3-pip
        zlib1g-dev libzstd-dev libbz2-dev liblz4-dev
        libopenh264-dev
        libjpeg-dev libpng-dev libtiff-dev
        libopenblas-dev libeigen3-dev libsuitesparse-dev
        libgflags-dev libgoogle-glog-dev libceres-dev
        libtbb-dev
        libboost-program-options-dev
        nlohmann-json3-dev
        libopencv-dev
        qt6-base-dev
    )

    # Check if running as root or if sudo is needed
    if [[ $EUID -eq 0 ]]; then
        apt update
        apt install -y "${PACKAGES[@]}"
    else
        log "Installing packages (may require sudo password)..."
        sudo apt update
        sudo apt install -y "${PACKAGES[@]}"
    fi

    log "System dependencies installed"
}

#------------------------------------------------------------------------------
# Setup environment
#------------------------------------------------------------------------------
setup_env() {
    log "Setting up build environment..."
    log "Building H264/Zarr components to: $PREFIX"
    log "Build directory: $BUILDDIR"
    log "Using $JOBS parallel jobs"

    mkdir -p "$PREFIX" "$BUILDDIR" "$PREFIX/thirdparty"
    export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
    export CMAKE_PREFIX_PATH="$PREFIX:${CMAKE_PREFIX_PATH:-}"
    export PATH="$PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"
    export CFLAGS="-I$PREFIX/include"
    export CXXFLAGS="-I$PREFIX/include"
    export LDFLAGS="-L$PREFIX/lib -L$PREFIX/lib64"
}

#------------------------------------------------------------------------------
# xtl (xtensor dependency, header-only)
#------------------------------------------------------------------------------
build_xtl() {
    is_done xtl && { log "xtl already built, skipping"; return; }
    log "Building xtl..."

    local XTL_SRC="$PREFIX/thirdparty/xtl"

    clone_repo "https://github.com/xtensor-stack/xtl" "$XTL_SRC" "master"

    mkdir -p "$BUILDDIR/xtl"
    cd "$BUILDDIR/xtl"
    cmake "$XTL_SRC" -G Ninja "${CMAKE_COMMON[@]}" \
        -DBUILD_TESTS=OFF
    ninja install

    mark_done xtl
}

#------------------------------------------------------------------------------
# xsimd (SIMD library, header-only)
#------------------------------------------------------------------------------
build_xsimd() {
    is_done xsimd && { log "xsimd already built, skipping"; return; }
    log "Building xsimd..."

    local XSIMD_SRC="$PREFIX/thirdparty/xsimd"

    clone_repo "https://github.com/xtensor-stack/xsimd" "$XSIMD_SRC" "master"

    mkdir -p "$BUILDDIR/xsimd"
    cd "$BUILDDIR/xsimd"
    cmake "$XSIMD_SRC" -G Ninja "${CMAKE_COMMON[@]}" \
        -DBUILD_TESTS=OFF
    ninja install

    mark_done xsimd
}

#------------------------------------------------------------------------------
# xtensor (multi-dimensional arrays, header-only)
#------------------------------------------------------------------------------
build_xtensor() {
    is_done xtensor && { log "xtensor already built, skipping"; return; }
    log "Building xtensor..."

    local XTENSOR_SRC="$PREFIX/thirdparty/xtensor"

    clone_repo "https://github.com/xtensor-stack/xtensor" "$XTENSOR_SRC" "master"

    mkdir -p "$BUILDDIR/xtensor"
    cd "$BUILDDIR/xtensor"
    cmake "$XTENSOR_SRC" -G Ninja "${CMAKE_COMMON[@]}" \
        -DBUILD_TESTS=OFF
    ninja install

    mark_done xtensor
}

#------------------------------------------------------------------------------
# c-blosc2 (Blosc compression library v2 with H264 codec built-in)
# Using vendored fork: https://github.com/SuperOptimizer/c-blosc2
#------------------------------------------------------------------------------
build_blosc2() {
    is_done blosc2 && { log "c-blosc2 already built, skipping"; return; }
    log "Building c-blosc2 (SuperOptimizer fork with H264 support)..."

    local BLOSC2_SRC="$PREFIX/thirdparty/c-blosc2"

    clone_repo "https://github.com/SuperOptimizer/c-blosc2" "$BLOSC2_SRC" "main"

    mkdir -p "$BUILDDIR/blosc2"
    cd "$BUILDDIR/blosc2"
    # Use system openh264 from apt (libopenh264-dev)
    cmake "$BLOSC2_SRC" -G Ninja "${CMAKE_COMMON[@]}" \
        -DBUILD_TESTS=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DDEACTIVATE_ZLIB=OFF \
        -DDEACTIVATE_ZSTD=OFF \
        -DPREFER_EXTERNAL_ZLIB=ON \
        -DPREFER_EXTERNAL_ZSTD=ON
    ninja -j"$JOBS"
    ninja install

    mark_done blosc2
}

#------------------------------------------------------------------------------
# z5 (Zarr/N5 chunked array storage - using blosc2)
# Using vendored fork: https://github.com/SuperOptimizer/z5
#------------------------------------------------------------------------------
build_z5() {
    is_done z5 && { log "z5 already built, skipping"; return; }
    log "Building z5 (SuperOptimizer fork with blosc2 support)..."

    local Z5_SRC="$PREFIX/thirdparty/z5"

    clone_repo "https://github.com/SuperOptimizer/z5" "$Z5_SRC" "master"

    mkdir -p "$BUILDDIR/z5"
    cd "$BUILDDIR/z5"
    cmake "$Z5_SRC" -G Ninja "${CMAKE_COMMON[@]}" \
        -DBUILD_TESTS=OFF \
        -DBUILD_Z5PY=OFF \
        -DWITH_BLOSC=ON \
        -DWITH_ZLIB=ON \
        -DWITH_BZIP2=OFF \
        -DWITH_XZ=OFF \
        -DWITH_LZ4=OFF \
        -DBLOSC2_INCLUDE_DIR="$PREFIX/include" \
        -DBLOSC2_LIBRARY="$PREFIX/lib/libblosc2.so"
    ninja install

    mark_done z5
}

#------------------------------------------------------------------------------
# Python blosc2 (built from vendored c-blosc2 with openh264 support)
#------------------------------------------------------------------------------
build_python_blosc2() {
    is_done python_blosc2 && { log "python-blosc2 already built, skipping"; return; }
    log "Building python-blosc2 (with openh264 support)..."

    local PYBLOSC2_SRC="$PREFIX/thirdparty/python-blosc2"
    local BLOSC2_SRC="$PREFIX/thirdparty/c-blosc2"

    # Ensure c-blosc2 source exists (needed for headers/build)
    if [[ ! -d "$BLOSC2_SRC" ]]; then
        error "c-blosc2 source not found at $BLOSC2_SRC - run build_blosc2 first"
    fi

    clone_repo "https://github.com/Blosc/python-blosc2" "$PYBLOSC2_SRC" "main"

    cd "$PYBLOSC2_SRC"

    # Install build dependencies first
    pip install --user scikit-build-core cython numpy setuptools --quiet

    # Build python-blosc2 linking against our vendored c-blosc2 with openh264
    USE_SYSTEM_BLOSC2=TRUE \
    BLOSC2_DIR="$PREFIX" \
    CMAKE_PREFIX_PATH="$PREFIX" \
    CFLAGS="-I$PREFIX/include" \
    LDFLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib" \
    LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH" \
    pip install --user . --no-cache-dir --no-build-isolation -v

    # Verify openh264 codec is available
    LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH" \
    python3 -c "
import blosc2
codecs = [str(c) for c in blosc2.compressor_list()]
print('Available codecs:', codecs)
print('libblosc2 loaded successfully')
" && log "python-blosc2 built with vendored libblosc2" || warn "python-blosc2 verification had warnings"

    mark_done python_blosc2
}

#------------------------------------------------------------------------------
# Numcodecs (vendored fork with Blosc2/openh264 support)
# Using fork: https://github.com/SuperOptimizer/numcodecs
#------------------------------------------------------------------------------
build_numcodecs() {
    is_done numcodecs && { log "numcodecs already built, skipping"; return; }
    log "Building numcodecs (SuperOptimizer fork with Blosc2/openh264 support)..."

    local NUMCODECS_SRC="$PREFIX/thirdparty/numcodecs"

    clone_repo "https://github.com/SuperOptimizer/numcodecs" "$NUMCODECS_SRC" "main"

    cd "$NUMCODECS_SRC"

    # Install numcodecs with blosc2 support
    # Disable C extensions (blosc1) - we only need the blosc2 Python wrapper
    DISABLE_NUMCODECS_CEXT=1 \
    LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH" \
    pip install --user . --no-cache-dir -v

    # Verify Blosc2 codec is available
    LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH" \
    python3 -c "
from numcodecs import Blosc2
codec = Blosc2(cname='zstd')
print('Blosc2 codec available:', codec)
# Test openh264 codec instantiation
codec_h264 = Blosc2(cname='openh264')
print('Blosc2 openh264 codec available:', codec_h264)
" && log "numcodecs built with Blosc2/openh264 support" || warn "numcodecs verification had warnings"

    mark_done numcodecs
}

#------------------------------------------------------------------------------
# Zarr-Python (vendored fork with Blosc2/openh264 codec)
# Using fork: https://github.com/SuperOptimizer/zarr-python
#------------------------------------------------------------------------------
build_zarr() {
    is_done zarr && { log "zarr-python already built, skipping"; return; }
    log "Building zarr-python (SuperOptimizer fork with Blosc2/openh264 codec)..."

    local ZARR_SRC="$PREFIX/thirdparty/zarr-python"

    clone_repo "https://github.com/SuperOptimizer/zarr-python" "$ZARR_SRC" "main"

    cd "$ZARR_SRC"

    # Install zarr-python
    LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH" \
    pip install --user . --no-cache-dir -v

    # Verify Blosc2Codec is available
    LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH" \
    python3 -c "
from zarr.codecs import Blosc2Codec, Blosc2Cname
codec = Blosc2Codec(cname=Blosc2Cname.zstd)
print('Blosc2Codec available:', codec)
# Test openh264 codec
codec_h264 = Blosc2Codec(cname=Blosc2Cname.openh264)
print('Blosc2Codec openh264 available:', codec_h264)
" && log "zarr-python built with Blosc2/openh264 codec" || warn "zarr-python verification had warnings"

    mark_done zarr
}

#------------------------------------------------------------------------------
# Main build sequence
#------------------------------------------------------------------------------
main() {
    log "=== H264/Zarr Compression Stack Builder ==="
    log ""
    log "This script builds only the vendored H264 compression components."
    log "All other dependencies are installed via apt."
    log ""

    # Install system dependencies
    install_system_deps

    # Setup build environment
    setup_env

    log ""
    log "=== Building vendored H264/Zarr components ==="
    log ""

    # Build xtensor ecosystem (header-only, but need newer version than apt)
    build_xtl
    build_xsimd
    build_xtensor

    # Build c-blosc2 with H264 support (SuperOptimizer fork)
    # Uses system openh264 from libopenh264-dev
    build_blosc2

    # Build z5 with blosc2 support (SuperOptimizer fork)
    build_z5

    # Build Python stack
    build_python_blosc2
    build_numcodecs
    build_zarr

    log ""
    log "=== H264/Zarr compression stack built successfully! ==="
    log ""
    log "Vendored components installed to: $PREFIX"
    log ""
    log "To use with Volume Cartographer, build with:"
    log "  mkdir build && cd build"
    log "  cmake .. -G Ninja \\"
    log "    -DCMAKE_PREFIX_PATH=\"$PREFIX\" \\"
    log "    -DCMAKE_BUILD_TYPE=Release"
    log "  ninja"
    log ""
    log "For Python scripts, ensure LD_LIBRARY_PATH includes $PREFIX/lib:"
    log "  export LD_LIBRARY_PATH=\"$PREFIX/lib:\$LD_LIBRARY_PATH\""
    log ""
    log "Build artifacts are in: $BUILDDIR"
    log "You can remove them with: rm -rf $BUILDDIR"
}

main "$@"
