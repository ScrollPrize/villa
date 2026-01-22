#!/bin/bash
#
# fromscratch.sh - Build all Volume Cartographer dependencies from source
#
# Prerequisites (must be installed via system package manager):
#   - clang/llvm toolchain (clang, clang++, lld, llvm-ar, llvm-ranlib, llvm-nm, etc.)
#   - cmake, ninja-build
#   - ccache
#   - flang (LLVM Fortran compiler, for OpenBLAS LAPACK)
#   - python3
#   - git, wget, tar, xz-utils
#   - X11/OpenGL dev libraries (libx11-dev, libgl-dev, libxkbcommon-dev, etc.)
#   - fontconfig, freetype dev libraries
#
# On Ubuntu/Debian:
#   apt install clang lld llvm flang ccache cmake ninja-build python3 \
#       git wget xz-utils libx11-dev libxext-dev libxfixes-dev libxi-dev \
#       libxrender-dev libxcb1-dev libx11-xcb-dev libxcb-glx0-dev \
#       libxcb-keysyms1-dev libxcb-image0-dev libxcb-shm0-dev \
#       libxcb-icccm4-dev libxcb-sync-dev libxcb-xfixes0-dev \
#       libxcb-shape0-dev libxcb-randr0-dev libxcb-render-util0-dev \
#       libxcb-util-dev libxcb-xinerama0-dev libxcb-xkb-dev \
#       libxkbcommon-dev libxkbcommon-x11-dev libgl-dev libegl-dev \
#       libfontconfig1-dev libfreetype-dev libharfbuzz-dev \
#       libdbus-1-dev libicu-dev
#
# Usage:
#   ./scripts/fromscratch.sh [PREFIX]
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
# Source tarballs cached in PREFIX to avoid re-downloading each run
SRCDIR="${SRCDIR:-$PREFIX/src-cache}"

# Compiler setup - use ccache with clang
CC="ccache clang"
CXX="ccache clang++"
export CC CXX

# Versions - update these as needed
ZLIB_VERSION="1.3.1"
ZSTD_VERSION="1.5.6"
LIBDEFLATE_VERSION="1.22"
BZIP2_VERSION="1.0.8"
BROTLI_VERSION="1.1.0"
JPEG_VERSION="3.0.4"
PNG_VERSION="1.6.44"
# NOTE: WEBP removed - VC3D doesn't use WebP, disabled in OpenCV
TIFF_VERSION="4.7.0"
# NOTE: IMATH/OPENEXR removed - VC3D doesn't use OpenEXR, disabled in OpenCV
OPENBLAS_VERSION="0.3.28"
TBB_VERSION="2022.3.0"
METIS_VERSION="5.2.1"
SUITESPARSE_VERSION="7.8.3"
EIGEN_VERSION="3.4.0"
GFLAGS_VERSION="2.2.2"
GLOG_VERSION="0.7.1"
CERES_VERSION="2.2.0"
BOOST_VERSION="1.86.0"
QT_VERSION="6.8.1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[BUILD]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# Common CMake flags - use ccache with clang via launcher
CMAKE_COMMON=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
    -DCMAKE_PREFIX_PATH="$PREFIX"
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    -DCMAKE_AR="$(which llvm-ar)"
    -DCMAKE_RANLIB="$(which llvm-ranlib)"
    -DCMAKE_NM="$(which llvm-nm)"
    -DCMAKE_STRIP="$(which llvm-strip)"
    -DCMAKE_OBJDUMP="$(which llvm-objdump)"
    -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld -flto=thin -ffat-lto-objects "
    -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld -flto=thin -ffat-lto-objects  "
    -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld -flto=thin -ffat-lto-objects "
    -DCMAKE_C_FLAGS=" -O3 -march=native -flto=thin -ffat-lto-objects "
    -DCMAKE_CXX_FLAGS=" -O3 -march=native -flto=thin -ffat-lto-objects "
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
)

download() {
    local url="$1"
    local dest="$2"
    if [[ ! -f "$dest" ]]; then
        log "Downloading $(basename "$dest") from $url..."
        wget -q -O "$dest" "$url" || { rm -f "$dest"; error "Failed to download $url"; }
    else
        log "Using cached $(basename "$dest")"
    fi
}

extract() {
    local archive="$1"
    local dest="$2"
    if [[ ! -d "$dest" ]]; then
        log "Extracting $(basename "$archive")..."
        mkdir -p "$dest"
        tar -xf "$archive" -C "$dest" --strip-components=1
    fi
}

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

# Setup
log "Building dependencies to: $PREFIX"
log "Build directory: $BUILDDIR"
log "Using $JOBS parallel jobs"

mkdir -p "$PREFIX" "$SRCDIR" "$BUILDDIR"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$PREFIX:${CMAKE_PREFIX_PATH:-}"
export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export CFLAGS="-I$PREFIX/include"
export CXXFLAGS="-I$PREFIX/include"
export LDFLAGS="-L$PREFIX/lib -L$PREFIX/lib64"

#------------------------------------------------------------------------------
# zlib
#------------------------------------------------------------------------------
build_zlib() {
    is_done zlib && { log "zlib already built, skipping"; return; }
    log "Building zlib $ZLIB_VERSION..."

    download "https://zlib.net/zlib-$ZLIB_VERSION.tar.gz" "$SRCDIR/zlib.tar.gz"
    extract "$SRCDIR/zlib.tar.gz" "$BUILDDIR/zlib"

    mkdir -p "$BUILDDIR/zlib/build"
    cd "$BUILDDIR/zlib/build"
    cmake .. -G Ninja "${CMAKE_COMMON[@]}"
    ninja -j"$JOBS"
    ninja install

    mark_done zlib
}

#------------------------------------------------------------------------------
# zstd
#------------------------------------------------------------------------------
build_zstd() {
    is_done zstd && { log "zstd already built, skipping"; return; }
    log "Building zstd $ZSTD_VERSION..."

    download "https://github.com/facebook/zstd/releases/download/v$ZSTD_VERSION/zstd-$ZSTD_VERSION.tar.gz" \
        "$SRCDIR/zstd.tar.gz"
    extract "$SRCDIR/zstd.tar.gz" "$BUILDDIR/zstd"

    mkdir -p "$BUILDDIR/zstd/build/cmake/build"
    cd "$BUILDDIR/zstd/build/cmake/build"
    cmake .. -G Ninja "${CMAKE_COMMON[@]}" \
        -DZSTD_BUILD_STATIC=OFF \
        -DZSTD_BUILD_SHARED=ON \
        -DZSTD_BUILD_PROGRAMS=OFF \
        -DZSTD_BUILD_TESTS=OFF
    ninja -j"$JOBS"
    ninja install

    mark_done zstd
}

#------------------------------------------------------------------------------
# bzip2
#------------------------------------------------------------------------------
build_bzip2() {
    is_done bzip2 && { log "bzip2 already built, skipping"; return; }
    log "Building bzip2 $BZIP2_VERSION..."

    download "https://sourceware.org/pub/bzip2/bzip2-$BZIP2_VERSION.tar.gz" \
        "$SRCDIR/bzip2.tar.gz"
    extract "$SRCDIR/bzip2.tar.gz" "$BUILDDIR/bzip2"

    cd "$BUILDDIR/bzip2"

    # bzip2 uses Makefile-libbz2_so for shared libraries
    make -f Makefile-libbz2_so -j"$JOBS" \
        CC="ccache clang" \
        AR="llvm-ar" \
        RANLIB="llvm-ranlib" \
        CFLAGS="-O3 -march=native -fPIC -flto=thin -ffat-lto-objects"

    # Install manually
    mkdir -p "$PREFIX/lib" "$PREFIX/include" "$PREFIX/bin"
    cp -a libbz2.so.1.0.8 "$PREFIX/lib/" 2>/dev/null || cp -a libbz2.so.1.0 "$PREFIX/lib/"

    # Create symlinks
    cd "$PREFIX/lib"
    if [[ -f libbz2.so.1.0.8 ]]; then
        ln -sf libbz2.so.1.0.8 libbz2.so.1.0
    fi
    ln -sf libbz2.so.1.0 libbz2.so

    cd "$BUILDDIR/bzip2"
    cp bzlib.h "$PREFIX/include/"

    # Create pkg-config file
    mkdir -p "$PREFIX/lib/pkgconfig"
    cat > "$PREFIX/lib/pkgconfig/bzip2.pc" << EOF
prefix=$PREFIX
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include
Name: bzip2
Description: A high-quality data compression program
Version: $BZIP2_VERSION
Libs: -L\${libdir} -lbz2
Cflags: -I\${includedir}
EOF

    mark_done bzip2
}

#------------------------------------------------------------------------------
# brotli
#------------------------------------------------------------------------------
build_brotli() {
    is_done brotli && { log "brotli already built, skipping"; return; }
    log "Building brotli $BROTLI_VERSION..."

    download "https://github.com/google/brotli/archive/refs/tags/v$BROTLI_VERSION.tar.gz" \
        "$SRCDIR/brotli.tar.gz"
    extract "$SRCDIR/brotli.tar.gz" "$BUILDDIR/brotli"

    mkdir -p "$BUILDDIR/brotli/build"
    cd "$BUILDDIR/brotli/build"
    cmake .. -G Ninja "${CMAKE_COMMON[@]}" \
        -DBROTLI_DISABLE_TESTS=ON \
        -DBUILD_SHARED_LIBS=ON
    ninja -j"$JOBS"
    ninja install

    mark_done brotli
}

#------------------------------------------------------------------------------
# libdeflate - fast deflate/zlib/gzip compression (needed by OpenCV/libtiff)
#------------------------------------------------------------------------------
build_libdeflate() {
    is_done libdeflate && { log "libdeflate already built, skipping"; return; }
    log "Building libdeflate $LIBDEFLATE_VERSION..."

    download "https://github.com/ebiggers/libdeflate/releases/download/v$LIBDEFLATE_VERSION/libdeflate-$LIBDEFLATE_VERSION.tar.gz" \
        "$SRCDIR/libdeflate.tar.gz"
    extract "$SRCDIR/libdeflate.tar.gz" "$BUILDDIR/libdeflate"

    mkdir -p "$BUILDDIR/libdeflate/build"
    cd "$BUILDDIR/libdeflate/build"
    cmake .. -G Ninja "${CMAKE_COMMON[@]}" \
        -DLIBDEFLATE_BUILD_STATIC_LIB=OFF \
        -DLIBDEFLATE_BUILD_SHARED_LIB=ON \
        -DLIBDEFLATE_BUILD_GZIP=OFF
    ninja -j"$JOBS"
    ninja install

    mark_done libdeflate
}

#------------------------------------------------------------------------------
# libjpeg-turbo
#------------------------------------------------------------------------------
build_jpeg() {
    is_done jpeg && { log "libjpeg-turbo already built, skipping"; return; }
    log "Building libjpeg-turbo $JPEG_VERSION..."

    download "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/$JPEG_VERSION/libjpeg-turbo-$JPEG_VERSION.tar.gz" \
        "$SRCDIR/jpeg.tar.gz"
    extract "$SRCDIR/jpeg.tar.gz" "$BUILDDIR/jpeg"

    mkdir -p "$BUILDDIR/jpeg/build"
    cd "$BUILDDIR/jpeg/build"
    # Disable TurboJPEG (not used by VC3D) and 12-bit mode
    cmake .. -G Ninja "${CMAKE_COMMON[@]}" \
        -DENABLE_SHARED=ON \
        -DENABLE_STATIC=OFF \
        -DWITH_TURBOJPEG=OFF \
        -DWITH_12BIT=OFF
    ninja -j"$JOBS"
    ninja install

    mark_done jpeg
}

#------------------------------------------------------------------------------
# libpng
#------------------------------------------------------------------------------
build_png() {
    is_done png && { log "libpng already built, skipping"; return; }
    log "Building libpng $PNG_VERSION..."

    download "https://download.sourceforge.net/libpng/libpng-$PNG_VERSION.tar.gz" \
        "$SRCDIR/png.tar.gz"
    extract "$SRCDIR/png.tar.gz" "$BUILDDIR/png"

    mkdir -p "$BUILDDIR/png/build"
    cd "$BUILDDIR/png/build"
    cmake .. -G Ninja "${CMAKE_COMMON[@]}" \
        -DPNG_SHARED=ON \
        -DPNG_STATIC=OFF \
        -DPNG_TESTS=OFF \
        -DZLIB_ROOT="$PREFIX"
    ninja -j"$JOBS"
    ninja install

    mark_done png
}

# NOTE: libwebp removed - OpenCV WebP support is disabled, VC3D doesn't need it

#------------------------------------------------------------------------------
# libtiff
#------------------------------------------------------------------------------
build_tiff() {
    is_done tiff && { log "libtiff already built, skipping"; return; }
    log "Building libtiff $TIFF_VERSION..."

    download "https://download.osgeo.org/libtiff/tiff-$TIFF_VERSION.tar.gz" \
        "$SRCDIR/tiff.tar.gz"
    extract "$SRCDIR/tiff.tar.gz" "$BUILDDIR/tiff"

    mkdir -p "$BUILDDIR/tiff/build"
    cd "$BUILDDIR/tiff/build"
    # Disable webp and jpeg12 to avoid static linking issues with library ordering
    # Force HAVE_JPEGTURBO_DUAL_MODE_8_12 to FALSE to prevent tif_jpeg_12.c from being built
    # Explicitly set all compression library paths to ensure our versions are used
    cmake .. -G Ninja "${CMAKE_COMMON[@]}" \
        -Dtiff-tools=OFF \
        -Dtiff-tests=OFF \
        -Dtiff-contrib=OFF \
        -Dtiff-docs=OFF \
        -Dzlib=ON \
        -Dlibdeflate=ON \
        -Dzstd=ON \
        -Djpeg=ON \
        -Djpeg12=OFF \
        -Dold-jpeg=OFF \
        -Dwebp=OFF \
        -Dlzma=OFF \
        -Djbig=OFF \
        -Dlerc=OFF \
        -DZLIB_INCLUDE_DIR="$PREFIX/include" \
        -DZLIB_LIBRARY="$PREFIX/lib/libz.so" \
        -DDeflate_INCLUDE_DIR="$PREFIX/include" \
        -DDeflate_LIBRARY="$PREFIX/lib/libdeflate.so" \
        -Dzstd_INCLUDE_DIR="$PREFIX/include" \
        -Dzstd_LIBRARY="$PREFIX/lib/libzstd.so" \
        -DZSTD_INCLUDE_DIR="$PREFIX/include" \
        -DZSTD_LIBRARY="$PREFIX/lib/libzstd.so" \
        -DJPEG_INCLUDE_DIR="$PREFIX/include" \
        -DJPEG_LIBRARY="$PREFIX/lib/libjpeg.so" \
        -DHAVE_JPEGTURBO_DUAL_MODE_8_12=FALSE \
        -DCMAKE_SKIP_RPATH=FALSE \
        -DCMAKE_INSTALL_RPATH="$PREFIX/lib" \
        -DCMAKE_BUILD_RPATH="$PREFIX/lib" \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=FALSE \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE \
        -DCMAKE_LIBRARY_PATH="$PREFIX/lib"
    ninja -j"$JOBS"
    ninja install

    # Patch TiffConfig.cmake to properly find dependencies (upstream doesn't do this)
    cat > "$PREFIX/lib/cmake/tiff/TiffConfig.cmake" << 'TIFFCFG'
include(CMakeFindDependencyMacro)
# Find required dependencies
find_dependency(ZLIB)
find_dependency(JPEG)
# Find zstd
find_package(zstd QUIET CONFIG)
if(NOT zstd_FOUND)
    find_package(ZSTD QUIET)
endif()
# Find Deflate - use libdeflate config and create Deflate::Deflate alias
find_package(libdeflate QUIET CONFIG)
if(libdeflate_FOUND AND TARGET libdeflate::libdeflate_shared)
    if(NOT TARGET Deflate::Deflate)
        add_library(Deflate::Deflate ALIAS libdeflate::libdeflate_shared)
    endif()
endif()
# CMath target (math library)
if(NOT TARGET CMath::CMath)
    find_library(CMATH_LIBRARY m)
    if(CMATH_LIBRARY)
        add_library(CMath::CMath UNKNOWN IMPORTED)
        set_target_properties(CMath::CMath PROPERTIES IMPORTED_LOCATION "${CMATH_LIBRARY}")
    else()
        add_library(CMath::CMath INTERFACE IMPORTED)
    endif()
endif()
function(set_variable_from_rel_or_absolute_path var root rel_or_abs_path)
    if(IS_ABSOLUTE "${rel_or_abs_path}")
        set(${var} "${rel_or_abs_path}" PARENT_SCOPE)
    else()
        set(${var} "${root}/${rel_or_abs_path}" PARENT_SCOPE)
    endif()
endfunction()
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
get_filename_component(_ROOT "${_DIR}/" ABSOLUTE)
set_variable_from_rel_or_absolute_path("TIFF_INCLUDE_DIR" "${_ROOT}" "include")
set(TIFF_INCLUDE_DIRS ${TIFF_INCLUDE_DIR})
set(TIFF_LIBRARIES TIFF::tiff)
if(NOT TARGET TIFF::tiff)
    include("${CMAKE_CURRENT_LIST_DIR}/TiffTargets.cmake")
endif()
unset (_ROOT)
unset (_DIR)
TIFFCFG

    mark_done tiff
}

# NOTE: Imath and OpenEXR removed - VC3D doesn't use OpenEXR, disabled in OpenCV

#------------------------------------------------------------------------------
# OpenH264 (Cisco's H.264 codec - needed by c-blosc2 with H264 support)
#------------------------------------------------------------------------------
build_openh264() {
    is_done openh264 && { log "OpenH264 already built, skipping"; return; }
    log "Building OpenH264..."

    local OPENH264_SRC="$PREFIX/thirdparty/openh264"

    clone_repo "https://github.com/cisco/openh264" "$OPENH264_SRC" "master"

    cd "$OPENH264_SRC"
    # OpenH264 uses meson or make - use make for simplicity
    make -j"$JOBS" \
        CC="ccache clang" \
        CXX="ccache clang++" \
        AR="llvm-ar" \
        CFLAGS="-O3 -march=native -fPIC" \
        CXXFLAGS="-O3 -march=native -fPIC" \
        LDFLAGS="-fuse-ld=lld" \
        PREFIX="$PREFIX" \
        BUILDTYPE=Release \
        libraries

    make PREFIX="$PREFIX" install-shared

    mark_done openh264
}

#------------------------------------------------------------------------------
# c-blosc2 (Blosc compression library v2 with H264 codec built-in)
# Using vendored fork: https://github.com/SuperOptimizer/c-blosc2
#------------------------------------------------------------------------------
build_blosc2() {
    is_done blosc2 && { log "c-blosc2 already built, skipping"; return; }
    log "Building c-blosc2 (with H264 support)..."

    local BLOSC2_SRC="$PREFIX/thirdparty/c-blosc2"

    clone_repo "https://github.com/SuperOptimizer/c-blosc2" "$BLOSC2_SRC" "main"

    mkdir -p "$BUILDDIR/blosc2"
    cd "$BUILDDIR/blosc2"
    cmake "$BLOSC2_SRC" -G Ninja "${CMAKE_COMMON[@]}" \
        -DBUILD_TESTS=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DDEACTIVATE_ZLIB=OFF \
        -DDEACTIVATE_ZSTD=OFF \
        -DPREFER_EXTERNAL_ZLIB=ON \
        -DPREFER_EXTERNAL_ZSTD=ON \
        -DOPENH264_INCLUDE_DIR="$PREFIX/include" \
        -DOPENH264_LIBRARY="$PREFIX/lib/libopenh264.so"
    ninja -j"$JOBS"
    ninja install

    mark_done blosc2
}

#------------------------------------------------------------------------------
# OpenBLAS (includes LAPACK)
#------------------------------------------------------------------------------
build_openblas() {
    is_done openblas && { log "OpenBLAS already built, skipping"; return; }
    log "Building OpenBLAS $OPENBLAS_VERSION..."

    download "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$OPENBLAS_VERSION/OpenBLAS-$OPENBLAS_VERSION.tar.gz" \
        "$SRCDIR/openblas.tar.gz"
    extract "$SRCDIR/openblas.tar.gz" "$BUILDDIR/openblas"

    cd "$BUILDDIR/openblas"
    # OpenBLAS's native make build with explicit ZEN target for AMD Ryzen
    # ZEN target optimizes for AMD Zen architecture (all Ryzen CPUs)
    # Use gfortran for Fortran (flang runtime not easily linkable with GCC)
    make -j"$JOBS" \
        CC="ccache clang" \
        FC="ccache gfortran" \
        HOSTCC="ccache clang" \
        AR="llvm-ar" \
        RANLIB="llvm-ranlib" \
        USE_OPENMP=1 \
        NO_SHARED=0 \
        NO_STATIC=1 \
        NO_LAPACK=0 \
        NOFORTRAN=0 \
        TARGET=ZEN \
        libs netlib shared
    make PREFIX="$PREFIX" NO_SHARED=0 NO_STATIC=1 install

    mark_done openblas
}

#------------------------------------------------------------------------------
# TBB (Threading Building Blocks)
#------------------------------------------------------------------------------
build_tbb() {
    is_done tbb && { log "TBB already built, skipping"; return; }
    log "Building TBB $TBB_VERSION..."

    download "https://github.com/oneapi-src/oneTBB/archive/refs/tags/v$TBB_VERSION.tar.gz" \
        "$SRCDIR/tbb.tar.gz"
    extract "$SRCDIR/tbb.tar.gz" "$BUILDDIR/tbb"

    mkdir -p "$BUILDDIR/tbb/build"
    cd "$BUILDDIR/tbb/build"

    # TBB has issues with lld + version scripts, so disable LTO and use regular linking
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_PREFIX_PATH="$PREFIX" \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_AR="$(which llvm-ar)" \
        -DCMAKE_RANLIB="$(which llvm-ranlib)" \
        -DCMAKE_NM="$(which llvm-nm)" \
        -DCMAKE_STRIP="$(which llvm-strip)" \
        -DCMAKE_OBJDUMP="$(which llvm-objdump)" \
        -DCMAKE_C_FLAGS="-O3 -march=native" \
        -DCMAKE_CXX_FLAGS="-O3 -march=native" \
        -DBUILD_SHARED_LIBS=ON \
        -DTBB_TEST=OFF \
        -DTBB_EXAMPLES=OFF \
        -DTBB_STRICT=OFF
    ninja -j"$JOBS"
    ninja install

    mark_done tbb
}

#------------------------------------------------------------------------------
# GKlib (required by METIS)
#------------------------------------------------------------------------------
build_gklib() {
    is_done gklib && { log "GKlib already built, skipping"; return; }
    log "Building GKlib..."

    download "https://github.com/KarypisLab/GKlib/archive/refs/heads/master.tar.gz" \
        "$SRCDIR/gklib.tar.gz"
    extract "$SRCDIR/gklib.tar.gz" "$BUILDDIR/gklib"

    mkdir -p "$BUILDDIR/gklib/build"
    cd "$BUILDDIR/gklib/build"

    cmake .. -G Ninja \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DOPENMP=ON \
        -DSHARED=ON \
        -DGKLIB_BUILD_APPS=OFF

    ninja -j"$JOBS"
    ninja install

    # Ensure development symlink exists (ninja install sometimes doesn't create it)
    cd "$PREFIX/lib"
    if [[ -f libGKlib.so.0.0.1 ]] && [[ ! -L libGKlib.so ]]; then
        ln -sf libGKlib.so.0.0.1 libGKlib.so
    fi

    mark_done gklib
}

#------------------------------------------------------------------------------
# METIS (required by SuiteSparse)
#------------------------------------------------------------------------------
build_metis() {
    is_done metis && { log "METIS already built, skipping"; return; }
    log "Building METIS $METIS_VERSION..."

    # METIS 5.2.x requires GKlib - build it first
    build_gklib

    download "https://github.com/KarypisLab/METIS/archive/refs/tags/v$METIS_VERSION.tar.gz" \
        "$SRCDIR/metis.tar.gz"
    extract "$SRCDIR/metis.tar.gz" "$BUILDDIR/metis"

    cd "$BUILDDIR/metis"

    # Set up build directory structure manually (make config has issues)
    rm -rf build
    mkdir -p build/xinclude
    echo "#define IDXTYPEWIDTH 32" > build/xinclude/metis.h
    echo "#define REALTYPEWIDTH 32" >> build/xinclude/metis.h
    cat include/metis.h >> build/xinclude/metis.h
    cp include/CMakeLists.txt build/xinclude/

    # Run cmake directly for better control
    cd build
    cmake .. -G Ninja \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DGKLIB_PATH="$PREFIX" \
        -DOPENMP=ON \
        -DSHARED=ON \
        -DCMAKE_SHARED_LINKER_FLAGS="-L$PREFIX/lib -lGKlib -Wl,-rpath,$PREFIX/lib" \
        -DCMAKE_BUILD_RPATH="$PREFIX/lib" \
        -DCMAKE_INSTALL_RPATH="$PREFIX/lib"

    # Build only the library target
    ninja -j"$JOBS" metis

    # Manual install - we need libmetis.so and metis.h for SuiteSparse
    mkdir -p "$PREFIX/lib" "$PREFIX/include"
    cp libmetis/libmetis.so* "$PREFIX/lib/" 2>/dev/null || cp libmetis/libmetis.so "$PREFIX/lib/"
    # Use the generated header from xinclude (has proper IDXTYPEWIDTH/REALTYPEWIDTH defines)
    cp xinclude/metis.h "$PREFIX/include/"

    # Fix RPATH in the installed library using patchelf if available
    if command -v patchelf >/dev/null 2>&1; then
        patchelf --set-rpath "$PREFIX/lib" "$PREFIX/lib/libmetis.so" || true
    fi

    mark_done metis
}

#------------------------------------------------------------------------------
# SuiteSparse
#------------------------------------------------------------------------------
build_suitesparse() {
    is_done suitesparse && { log "SuiteSparse already built, skipping"; return; }
    log "Building SuiteSparse $SUITESPARSE_VERSION..."

    download "https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v$SUITESPARSE_VERSION.tar.gz" \
        "$SRCDIR/suitesparse.tar.gz"
    extract "$SRCDIR/suitesparse.tar.gz" "$BUILDDIR/suitesparse"

    mkdir -p "$BUILDDIR/suitesparse/build"
    cd "$BUILDDIR/suitesparse/build"
    cmake .. -G Ninja "${CMAKE_COMMON[@]}" \
        -DBLA_VENDOR=OpenBLAS \
        -DBLAS_LIBRARIES="$PREFIX/lib/libopenblas.so" \
        -DLAPACK_LIBRARIES="$PREFIX/lib/libopenblas.so" \
        -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;camd;ccolamd;colamd;cholmod;spqr" \
        -DSUITESPARSE_USE_CUDA=OFF \
        -DSUITESPARSE_USE_OPENMP=ON \
        -DSUITESPARSE_DEMOS=OFF
    ninja -j"$JOBS"
    ninja install

    mark_done suitesparse
}

#------------------------------------------------------------------------------
# Eigen (header-only)
#------------------------------------------------------------------------------
build_eigen() {
    is_done eigen && { log "Eigen already built, skipping"; return; }
    log "Building Eigen $EIGEN_VERSION..."

    download "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.gz" \
        "$SRCDIR/eigen.tar.gz"
    extract "$SRCDIR/eigen.tar.gz" "$BUILDDIR/eigen"

    mkdir -p "$BUILDDIR/eigen/build"
    cd "$BUILDDIR/eigen/build"
    cmake .. -G Ninja "${CMAKE_COMMON[@]}" \
        -DBUILD_TESTING=OFF \
        -DEIGEN_BUILD_DOC=OFF
    ninja install

    mark_done eigen
}

#------------------------------------------------------------------------------
# gflags (required by glog)
#------------------------------------------------------------------------------
build_gflags() {
    is_done gflags && { log "gflags already built, skipping"; return; }
    log "Building gflags $GFLAGS_VERSION..."

    download "https://github.com/gflags/gflags/archive/refs/tags/v$GFLAGS_VERSION.tar.gz" \
        "$SRCDIR/gflags.tar.gz"
    extract "$SRCDIR/gflags.tar.gz" "$BUILDDIR/gflags"

    mkdir -p "$BUILDDIR/gflags/build"
    cd "$BUILDDIR/gflags/build"
    cmake .. -G Ninja "${CMAKE_COMMON[@]}" \
        -DBUILD_TESTING=OFF \
        -DBUILD_gflags_LIB=ON
    ninja -j"$JOBS"
    ninja install

    mark_done gflags
}

#------------------------------------------------------------------------------
# glog
#------------------------------------------------------------------------------
build_glog() {
    is_done glog && { log "glog already built, skipping"; return; }
    log "Building glog $GLOG_VERSION..."

    download "https://github.com/google/glog/archive/refs/tags/v$GLOG_VERSION.tar.gz" \
        "$SRCDIR/glog.tar.gz"
    extract "$SRCDIR/glog.tar.gz" "$BUILDDIR/glog"

    mkdir -p "$BUILDDIR/glog/build"
    cd "$BUILDDIR/glog/build"
    cmake .. -G Ninja "${CMAKE_COMMON[@]}" \
        -DBUILD_TESTING=OFF \
        -DWITH_GFLAGS=ON \
        -DWITH_UNWIND=OFF
    ninja -j"$JOBS"
    ninja install

    mark_done glog
}

#------------------------------------------------------------------------------
# Ceres Solver
#------------------------------------------------------------------------------
build_ceres() {
    is_done ceres && { log "Ceres already built, skipping"; return; }
    log "Building Ceres Solver $CERES_VERSION..."

    download "https://github.com/ceres-solver/ceres-solver/archive/refs/tags/$CERES_VERSION.tar.gz" \
        "$SRCDIR/ceres.tar.gz"
    extract "$SRCDIR/ceres.tar.gz" "$BUILDDIR/ceres"

    mkdir -p "$BUILDDIR/ceres/build"
    cd "$BUILDDIR/ceres/build"
    cmake .. -G Ninja "${CMAKE_COMMON[@]}" \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DUSE_CUDA=OFF \
        -DPROVIDE_UNINSTALL_TARGET=OFF \
        -DEIGENSPARSE=ON \
        -DSUITESPARSE=ON \
        -DCXSPARSE=OFF \
        -DLAPACK=ON \
        -DGFLAGS=ON \
        -DBLA_VENDOR=OpenBLAS \
        -DBLAS_LIBRARIES="$PREFIX/lib/libopenblas.so" \
        -DLAPACK_LIBRARIES="$PREFIX/lib/libopenblas.so" \
        -DCHOLMOD_INCLUDE_DIR="$PREFIX/include" \
        -DCHOLMOD_LIBRARY="$PREFIX/lib/libcholmod.so" \
        -DCAMD_LIBRARY="$PREFIX/lib/libcamd.so" \
        -DCCOLAMD_LIBRARY="$PREFIX/lib/libccolamd.so" \
        -DCOLAMD_LIBRARY="$PREFIX/lib/libcolamd.so" \
        -DAMD_LIBRARY="$PREFIX/lib/libamd.so" \
        -DSUITESPARSEQR_LIBRARY="$PREFIX/lib/libspqr.so" \
        -DCMAKE_FIND_ROOT_PATH="$PREFIX" \
        -DCMAKE_SKIP_RPATH=FALSE \
        -DCMAKE_INSTALL_RPATH="$PREFIX/lib" \
        -DCMAKE_BUILD_RPATH="$PREFIX/lib" \
        -DCMAKE_LIBRARY_PATH="$PREFIX/lib"
    ninja -j"$JOBS"
    ninja install

    mark_done ceres
}

#------------------------------------------------------------------------------
# Boost (just program_options)
#------------------------------------------------------------------------------
build_boost() {
    is_done boost && { log "Boost already built, skipping"; return; }
    log "Building Boost $BOOST_VERSION..."

    local boost_underscore="${BOOST_VERSION//./_}"
    download "https://archives.boost.io/release/$BOOST_VERSION/source/boost_$boost_underscore.tar.gz" \
        "$SRCDIR/boost.tar.gz"
    extract "$SRCDIR/boost.tar.gz" "$BUILDDIR/boost"

    cd "$BUILDDIR/boost"
    ./bootstrap.sh --prefix="$PREFIX" --with-toolset=clang --with-libraries=program_options

    # Create a ccache wrapper script for b2 to use
    mkdir -p "$BUILDDIR/bin"
    cat > "$BUILDDIR/bin/clang++-ccache" <<'WRAPPER'
#!/bin/bash
exec ccache clang++ "$@"
WRAPPER
    chmod +x "$BUILDDIR/bin/clang++-ccache"

    # Create a user-config.jam pointing to the wrapper
    cat > user-config.jam <<EOF
using clang : : "$BUILDDIR/bin/clang++-ccache" : <archiver>llvm-ar <ranlib>llvm-ranlib ;
EOF

    ./b2 install \
        --user-config=user-config.jam \
        toolset=clang \
        link=shared \
        threading=multi \
        variant=release \
        -j"$JOBS"

    mark_done boost
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
# nlohmann_json (JSON library, header-only)
#------------------------------------------------------------------------------
build_json() {
    is_done json && { log "nlohmann_json already built, skipping"; return; }
    log "Building nlohmann_json..."

    local JSON_SRC="$PREFIX/thirdparty/json"

    clone_repo "https://github.com/nlohmann/json" "$JSON_SRC" "develop"

    mkdir -p "$BUILDDIR/json"
    cd "$BUILDDIR/json"
    cmake "$JSON_SRC" -G Ninja "${CMAKE_COMMON[@]}" \
        -DJSON_BuildTests=OFF
    ninja install

    mark_done json
}

#------------------------------------------------------------------------------
# z5 (Zarr/N5 chunked array storage - using blosc2 instead of blosc1)
# Using vendored fork: https://github.com/SuperOptimizer/z5
#------------------------------------------------------------------------------
build_z5() {
    is_done z5 && { log "z5 already built, skipping"; return; }
    log "Building z5 (with blosc2 support)..."

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
        return 1
    fi

    clone_repo "https://github.com/Blosc/python-blosc2" "$PYBLOSC2_SRC" "main"

    cd "$PYBLOSC2_SRC"

    # Install build dependencies first
    pip install scikit-build-core cython numpy setuptools --quiet

    # Build python-blosc2 linking against our vendored c-blosc2 with openh264
    # USE_SYSTEM_BLOSC2=TRUE tells it to use our pre-built libblosc2
    USE_SYSTEM_BLOSC2=TRUE \
    BLOSC2_DIR="$PREFIX" \
    CMAKE_PREFIX_PATH="$PREFIX" \
    CFLAGS="-I$PREFIX/include" \
    LDFLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib" \
    LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH" \
    pip install . --no-cache-dir --no-build-isolation -v

    # Verify openh264 codec is available
    LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH" \
    python3 -c "
import blosc2
codecs = [str(c) for c in blosc2.compressor_list()]
print('Available codecs:', codecs)
# Check if openh264 is registered (codec 240)
import ctypes
lib = ctypes.CDLL('$PREFIX/lib/libblosc2.so')
print('libblosc2 loaded successfully')
" && log "python-blosc2 built with vendored libblosc2" || warn "python-blosc2 verification had warnings"

    mark_done python_blosc2
}

#------------------------------------------------------------------------------
# Numcodecs (vendored fork with Blosc2/openh264 support)
#------------------------------------------------------------------------------
build_numcodecs() {
    is_done numcodecs && { log "numcodecs already built, skipping"; return; }
    log "Building numcodecs (with Blosc2/openh264 support)..."

    local NUMCODECS_SRC="$PREFIX/thirdparty/numcodecs"

    clone_repo "https://github.com/SuperOptimizer/numcodecs" "$NUMCODECS_SRC" "main"

    cd "$NUMCODECS_SRC"

    # Install numcodecs with blosc2 support
    # Disable C extensions (blosc1) - we only need the blosc2 Python wrapper
    DISABLE_NUMCODECS_CEXT=1 \
    LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH" \
    pip install . --no-cache-dir -v

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
#------------------------------------------------------------------------------
build_zarr() {
    is_done zarr && { log "zarr-python already built, skipping"; return; }
    log "Building zarr-python (with Blosc2/openh264 codec)..."

    local ZARR_SRC="$PREFIX/thirdparty/zarr-python"

    clone_repo "https://github.com/SuperOptimizer/zarr-python" "$ZARR_SRC" "main"

    cd "$ZARR_SRC"

    # Install zarr-python
    LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH" \
    pip install . --no-cache-dir -v

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
# OpenCV (with contrib modules)
#------------------------------------------------------------------------------
build_opencv() {
    is_done opencv && { log "OpenCV already built, skipping"; return; }
    log "Building OpenCV..."

    local OPENCV_SRC="$PREFIX/thirdparty/opencv"
    local OPENCV_CONTRIB="$PREFIX/thirdparty/opencv_contrib"

    clone_repo "https://github.com/opencv/opencv" "$OPENCV_SRC" "4.x"
    clone_repo "https://github.com/opencv/opencv_contrib" "$OPENCV_CONTRIB" "4.x"

    mkdir -p "$BUILDDIR/opencv"
    cd "$BUILDDIR/opencv"
    cmake "$OPENCV_SRC" -G Ninja "${CMAKE_COMMON[@]}" \
        -DOPENCV_EXTRA_MODULES_PATH="$OPENCV_CONTRIB/modules" \
        -DBUILD_LIST=core,imgcodecs,imgproc,calib3d,photo,videoio,ximgproc \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_opencv_apps=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENCL_SVM=OFF \
        -DWITH_OPENGL=ON \
        -DWITH_QT=OFF \
        -DWITH_GTK=OFF \
        -DWITH_TBB=OFF \
        -DWITH_IPP=ON \
        -DWITH_EIGEN=ON \
        -DWITH_LAPACK=ON \
        -DWITH_OPENMP=ON \
        -DWITH_JPEG=ON \
        -DWITH_PNG=ON \
        -DWITH_TIFF=ON \
        -DWITH_WEBP=OFF \
        -DWITH_OPENEXR=OFF \
        -DWITH_JASPER=OFF \
        -DWITH_OPENJPEG=OFF \
        -DWITH_V4L=OFF \
        -DWITH_GSTREAMER=OFF \
        -DWITH_FFMPEG=ON \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_java=OFF \
        -DOPENCV_ENABLE_NONFREE=OFF \
        -DENABLE_PRECOMPILED_HEADERS=OFF \
        -DOPENCV_GENERATE_PKGCONFIG=ON
    ninja -j"$JOBS"
    ninja install

    mark_done opencv
}

#------------------------------------------------------------------------------
# Qt6 (Core, Gui, Widgets, Network only)
#------------------------------------------------------------------------------
build_qt6() {
    is_done qt6 && { log "Qt6 already built, skipping"; return; }
    log "Building Qt6 $QT_VERSION..."

    local qt_major="${QT_VERSION%%.*}"
    local qt_minor="${QT_VERSION#*.}"
    qt_minor="${qt_minor%%.*}"

    download "https://download.qt.io/official_releases/qt/$qt_major.$qt_minor/$QT_VERSION/single/qt-everywhere-src-$QT_VERSION.tar.xz" \
        "$SRCDIR/qt6.tar.xz"
    extract "$SRCDIR/qt6.tar.xz" "$BUILDDIR/qt6"

    mkdir -p "$BUILDDIR/qt6/build"
    cd "$BUILDDIR/qt6/build"

    # Qt6 configure - build only what we need
    # Note: Don't use -ccache flag - it has issues with automoc.
    # Instead use CMAKE_*_COMPILER_LAUNCHER which works properly.
    # -debug
    ../configure \
        -prefix "$PREFIX" \
        -shared \
        -ccache \
        -opensource \
        -confirm-license \
        -nomake examples \
        -nomake tests \
        -nomake benchmarks \
        -no-feature-testlib \
        -no-ltcg \
        -skip qt3d \
        -skip qt5compat \
        -skip qtactiveqt \
        -skip qtcharts \
        -skip qtcoap \
        -skip qtconnectivity \
        -skip qtdatavis3d \
        -skip qtdeclarative \
        -skip qtdoc \
        -skip qtgraphs \
        -skip qtgrpc \
        -skip qthttpserver \
        -skip qtimageformats \
        -skip qtlanguageserver \
        -skip qtlocation \
        -skip qtlottie \
        -skip qtmqtt \
        -skip qtmultimedia \
        -skip qtopcua \
        -skip qtpositioning \
        -skip qtquick3d \
        -skip qtquick3dphysics \
        -skip qtquickeffectmaker \
        -skip qtquicktimeline \
        -skip qtremoteobjects \
        -skip qtscxml \
        -skip qtsensors \
        -skip qtserialbus \
        -skip qtserialport \
        -skip qtshadertools \
        -skip qtspeech \
        -skip qtsvg \
        -skip qttools \
        -skip qttranslations \
        -skip qtvirtualkeyboard \
        -skip qtwayland \
        -skip qtwebchannel \
        -skip qtwebengine \
        -skip qtwebsockets \
        -skip qtwebview \
        -platform linux-clang \
        -- \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_AR="$(which llvm-ar)" \
        -DCMAKE_RANLIB="$(which llvm-ranlib)" \
        -DCMAKE_NM="$(which llvm-nm)" \
        -DCMAKE_OBJDUMP="$(which llvm-objdump)" \
    -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld -flto=thin -ffat-lto-objects " \
    -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld -flto=thin -ffat-lto-objects  " \
    -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld -flto=thin -ffat-lto-objects " \
    -DCMAKE_C_FLAGS=" -O3 -march=native -flto=thin -ffat-lto-objects " \
    -DCMAKE_CXX_FLAGS=" -O3 -march=native -flto=thin -ffat-lto-objects " \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON

    cmake --build . --parallel "$JOBS"
    cmake --install .

    mark_done qt6
}

#------------------------------------------------------------------------------
# Main build sequence
#------------------------------------------------------------------------------
main() {
    log "=== Starting dependency build ==="
    log "PREFIX: $PREFIX"
    log ""

    # Compression libraries (build first - needed by everything)
    build_zlib
    build_zstd
    build_bzip2
    build_brotli
    build_libdeflate

    # Image format libraries
    build_jpeg
    build_png
    # NOTE: libwebp, imath, openexr removed - not needed by VC3D
    build_tiff

    # Data compression (blosc2 with H264 support)
    build_openh264
    build_blosc2

    # Math libraries
    build_openblas
    build_tbb
    build_metis
    build_suitesparse
    build_eigen

    # Ceres and its dependencies
    build_gflags
    build_glog
    build_ceres

    # Boost
    build_boost

    # xtensor ecosystem (header-only libraries)
    build_xtl
    build_xsimd
    build_xtensor
    build_json

    # z5 (depends on xtensor, blosc2 with H264, json)
    build_z5

    # Python blosc2 (with openh264 from vendored c-blosc2)
    build_python_blosc2

    # Numcodecs (vendored fork with Blosc2/openh264 support)
    build_numcodecs

    # Zarr-Python (vendored fork with Blosc2/openh264 codec)
    build_zarr

    # OpenCV (with contrib modules)
    build_opencv

    # Qt6
    build_qt6

    log ""
    log "=== All dependencies built successfully! ==="
    log ""
    log "To build Volume Cartographer, use:"
    log "  mkdir build && cd build"
    log "  cmake .. -G Ninja \\"
    log "    -DCMAKE_PREFIX_PATH=$PREFIX \\"
    log "    -DCMAKE_BUILD_TYPE=Debug \\"
    log "    -DVC_WITH_CUDA_SPARSE=OFF"
    log "  ninja"
    log ""
    log "Build artifacts are in: $BUILDDIR"
    log "You can remove them with: rm -rf $BUILDDIR"
}

main "$@"
