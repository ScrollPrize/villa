#!/usr/bin/env bash
# build_dependencies.sh - Mirrors the Dockerfile flow on a dev machine
# - Builds VC3D WITHOUT PaStiX
# - Builds Scotch + PaStiX ONLY for Flatboi
# - Fetches libigl from GitHub at a pinned commit and overlays libs/libigl_changes

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
export CC="ccache clang"
export CXX="ccache clang++"
export INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/vc-dependencies}"      # 3rd-party prefix for VC3D deps
export BUILD_DIR="${BUILD_DIR:-$HOME/vc-dependencies-build}"          # scratch build tree
export COMMON_FLAGS="-march=native -w"
export COMMON_LDFLAGS="-fuse-ld=lld"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBS_DIR="$REPO_ROOT/libs"

# libigl pin (latest on main as of 2025-10-02)
LIBIGL_COMMIT="ae8f959ea26d7059abad4c698aba8d6b7c3205e8"
LIBIGL_DIR="$LIBS_DIR/libigl"
LIBIGL_CHANGES_DIR="$LIBS_DIR/libigl_changes"

# Determine parallelism
if command -v nproc >/dev/null 2>&1; then
  JOBS="$(nproc)"
else
  JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi
export JOBS

log() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

# ---------------------------------------------------------------------------
# Architecture / GPU detection
#   - CUDA + cudss are only available for x86_64 with an NVIDIA GPU.
#   - Otherwise we fall back to apt libceres-dev and disable CUDA sparse.
# ---------------------------------------------------------------------------
ARCH="$(uname -m)"
USE_CUDA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
  case "$ARCH" in
    x86_64|aarch64) USE_CUDA=1 ;;
  esac
fi
if [[ "${VC_FORCE_NO_CUDA:-0}" == "1" ]]; then USE_CUDA=0; fi
case "$ARCH" in
  x86_64)  CUDSS_DEB_ARCH=amd64; CUDSS_LIBDIR=/usr/lib/x86_64-linux-gnu ;;
  aarch64) CUDSS_DEB_ARCH=arm64; CUDSS_LIBDIR=/usr/lib/aarch64-linux-gnu ;;
  *)       CUDSS_DEB_ARCH=""; CUDSS_LIBDIR="" ;;
esac
log "Arch: $ARCH   USE_CUDA=$USE_CUDA"

# ---------------------------------------------------------------------------
# OS prerequisites (Ubuntu/Noble-like flow)
# ---------------------------------------------------------------------------
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script mirrors the Ubuntu Dockerfile flow. For macOS, adapt packages/paths." >&2
  exit 1
fi

log "Installing toolchain and libraries"
sudo apt-get update
sudo ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime
sudo apt-get install -y --no-install-recommends tzdata
sudo dpkg-reconfigure -f noninteractive tzdata

sudo apt-get install -y \
  build-essential git clang llvm ccache ninja-build lld cmake pkg-config \
  qt6-base-dev libboost-system-dev libboost-program-options-dev \
  libcgal-dev libsuitesparse-dev \
  libopencv-dev libxsimd-dev libblosc-dev libspdlog-dev libgsl-dev libsdl2-dev \
  libavahi-client-dev libde265-dev libx265-dev rclone nlohmann-json3-dev liblz4-dev \
  libcurl4-openssl-dev file curl unzip ca-certificates bzip2 wget fuse jq gimp \
  desktop-file-utils flex bison zlib1g-dev gfortran libopenblas-dev liblapack-dev \
  libscotch-dev libhwloc-dev libomp-dev

# Pin xtl + xtensor to match the Dockerfile
log "Pinning xtl-dev 0.7.7 and libxtensor-dev 0.25.0"
tmpd="$(mktemp -d)"; pushd "$tmpd" >/dev/null
wget -q http://archive.ubuntu.com/ubuntu/pool/universe/x/xtl/xtl-dev_0.7.7-1_all.deb
wget -q http://archive.ubuntu.com/ubuntu/pool/universe/x/xtensor/libxtensor-dev_0.25.0-2ubuntu1_all.deb
sudo apt-get install -y --no-install-recommends ./xtl-dev_0.7.7-1_all.deb ./libxtensor-dev_0.25.0-2ubuntu1_all.deb
sudo apt-mark hold xtl-dev libxtensor-dev
popd >/dev/null
rm -rf "$tmpd"

# ---------------------------------------------------------------------------
# Fresh build roots
# ---------------------------------------------------------------------------
rm -rf "$BUILD_DIR" "$INSTALL_PREFIX"
mkdir -p "$BUILD_DIR" "$INSTALL_PREFIX"

# ---------------------------------------------------------------------------
# Optional: CUDA + cuDSS + Ceres-from-source (x86_64 w/ NVIDIA GPU only)
# On other hosts we use the apt libceres-dev installed above.
# ---------------------------------------------------------------------------
CERES_PREFIX=""
if [[ "$USE_CUDA" == "1" ]]; then
  log "Installing CUDA toolkit + cuDSS (x86_64 GPU host)"
  if ! command -v nvcc >/dev/null 2>&1; then
    sudo apt-get install -y nvidia-cuda-toolkit
  fi
  if ! dpkg -l | grep -q '^ii  cudss'; then
    tmpc="$(mktemp -d)"; pushd "$tmpc" >/dev/null
    CUDSS_DEB="cudss-local-repo-ubuntu2404-0.4.0_0.4.0-1_${CUDSS_DEB_ARCH}.deb"
    wget -q "https://developer.download.nvidia.com/compute/cudss/0.4.0/local_installers/${CUDSS_DEB}"
    sudo dpkg -i "$CUDSS_DEB"
    sudo cp /var/cudss-local-repo-ubuntu2404-0.4.0/cudss-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get install -y cudss
    popd >/dev/null; rm -rf "$tmpc"
  fi

  log "Building Ceres from source with CUDA sparse"
  CERES_PREFIX="$INSTALL_PREFIX/ceres-cuda"
  CERES_SRC="$BUILD_DIR/ceres-solver"
  rm -rf "$CERES_SRC" "$CERES_PREFIX"
  git clone https://github.com/ceres-solver/ceres-solver.git "$CERES_SRC"
  pushd "$CERES_SRC" >/dev/null
  git submodule update --init
  mkdir build && cd build
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -Dcudss_DIR="${CUDSS_LIBDIR}/libcudss/12/cmake/cudss" \
    -DSUITESPARSE=ON -DEIGENSPARSE=ON \
    -DUSE_CUDA=ON -DCUDA_SPARSE=ON -DACCELERATING_SPARSE_SCHUR=ON \
    -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF \
    -DCMAKE_CXX_FLAGS="-march=native" \
    -DCMAKE_INSTALL_PREFIX="$CERES_PREFIX"
  cmake --build . -j"$JOBS"
  cmake --install .
  sudo ldconfig
  popd >/dev/null
else
  log "Using distro Ceres (libceres-dev); CUDA sparse disabled"
  sudo apt-get install -y libceres-dev
fi

# ---------------------------------------------------------------------------
# z5 (pinned) → $INSTALL_PREFIX
# ---------------------------------------------------------------------------
log "Building z5 (pinned) into $INSTALL_PREFIX"
pushd "$BUILD_DIR" >/dev/null
rm -rf z5
git clone https://github.com/constantinpape/z5.git z5
pushd z5 >/dev/null
Z5_COMMIT=ee2081bb974fe0d0d702538400c31c38b09f1629
git fetch origin "$Z5_COMMIT" --depth 1
git checkout --detach "$Z5_COMMIT"
# Align z5 with xtensor 0.25’s header layout
sed -i 's|xtensor/containers/xadapt.hpp|xtensor/xadapt.hpp|' \
  include/z5/multiarray/xtensor_util.hxx || true
popd >/dev/null

cmake -S z5 -B z5/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DWITH_BLOSC=ON -DWITH_ZLIB=ON -DBUILD_Z5PY=OFF -DBUILD_TESTS=OFF
cmake --build z5/build -j"$JOBS"
cmake --install z5/build
popd >/dev/null

# ---------------------------------------------------------------------------
# Build Scotch + PaStiX ONLY for Flatboi (like Dockerfile)
# Scotch -> /usr/local/scotch, PaStiX -> /usr/local/pastix
# ---------------------------------------------------------------------------
[[ -f "$LIBS_DIR/scotch_6.0.4.tar.gz" ]] || { echo "Missing $LIBS_DIR/scotch_6.0.4.tar.gz"; exit 1; }
[[ -f "$LIBS_DIR/pastix_5.2.3.tar.bz2" ]] || { echo "Missing $LIBS_DIR/pastix_5.2.3.tar.bz2"; exit 1; }
[[ -f "$LIBS_DIR/config.in" ]] || { echo "Missing $LIBS_DIR/config.in"; exit 1; }

log "Building Scotch 6.0.4 into /usr/local/scotch"
SCOTCH_SRC="$BUILD_DIR/scotch"
sudo mkdir -p /usr/local/scotch
rm -rf "$SCOTCH_SRC"; mkdir -p "$SCOTCH_SRC"
tar -xzf "$LIBS_DIR/scotch_6.0.4.tar.gz" -C "$SCOTCH_SRC" --strip-components=1
pushd "$SCOTCH_SRC/src" >/dev/null
case "$ARCH" in
  aarch64) cp ./Make.inc/Makefile.inc.aarch64_pc_linux2 Makefile.inc 2>/dev/null \
           || cp ./Make.inc/Makefile.inc.x86-64_pc_linux2 Makefile.inc ;;
  *)       cp ./Make.inc/Makefile.inc.x86-64_pc_linux2 Makefile.inc ;;
esac
make -j"$JOBS" scotch
sudo mkdir -p /usr/local/scotch/{bin,include,lib,share/man/man1}
sudo make prefix=/usr/local/scotch install
# Don't prefer static Scotch; force link against shared libs.
sudo find /usr/local/scotch/lib -name "libscotch*.a" -delete || true
popd >/dev/null

log "Building PaStiX 5.2.3 into /usr/local/pastix (linked to /usr/local/scotch)"
PASTIX_SRC="$BUILD_DIR/pastix"
sudo mkdir -p /usr/local/pastix
rm -rf "$PASTIX_SRC"; mkdir -p "$PASTIX_SRC"
tar -xjf "$LIBS_DIR/pastix_5.2.3.tar.bz2" -C "$PASTIX_SRC" --strip-components=1
pushd "$PASTIX_SRC/src" >/dev/null
cp "$LIBS_DIR/config.in" config.in
# Build PaStiX against the Scotch we just built into /usr/local/scotch
sed -i -E "s|^SCOTCH_HOME[[:space:]]*=.*$|SCOTCH_HOME = /usr/local/scotch|" config.in
make SCOTCH_HOME=/usr/local/scotch
sudo make install SCOTCH_HOME=/usr/local/scotch
popd >/dev/null

# ---------------------------------------------------------------------------
# libigl (Git clone + overlay) + Flatboi
#   - Clone libigl to libs/libigl and pin to LIBIGL_COMMIT
#   - Overlay libs/libigl_changes/* into libs/libigl/
#   - Build from libs/flatboi if present; else fallback to tutorial/999_Flatboi
# ---------------------------------------------------------------------------
log "Cloning libigl at pinned commit into $LIBIGL_DIR"
rm -rf "$LIBIGL_DIR"
git clone https://github.com/libigl/libigl.git "$LIBIGL_DIR"
pushd "$LIBIGL_DIR" >/dev/null
git fetch origin "$LIBIGL_COMMIT" --depth 1
git checkout --detach "$LIBIGL_COMMIT"
git submodule update --init --recursive
popd >/dev/null

if [[ -d "$LIBIGL_CHANGES_DIR" ]]; then
  log "Overlaying custom changes from $LIBIGL_CHANGES_DIR into $LIBIGL_DIR"
  cp -a "$LIBIGL_CHANGES_DIR/." "$LIBIGL_DIR/"
fi

# Prefer libs/flatboi (matches Dockerfile); fallback to tutorial/999_Flatboi
FLATBOI_DIR="$LIBS_DIR/flatboi"
FLATBOI_CMAKE="$FLATBOI_DIR/CMakeLists.txt"
if [[ -f "$FLATBOI_CMAKE" ]]; then
  # Patch any hard-coded /src/libs/libigl subdirectory call
  sed -i -E \
    's|^([[:space:]]*)add_subdirectory\([[:space:]]*/src/libs/libigl[^)]*\)|\1add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../libigl ${CMAKE_BINARY_DIR}/libigl-build)|' \
    "$FLATBOI_CMAKE" || true
  # If something still uses /src/libs/libigl, provide a symlink
  if grep -q "/src/libs/libigl" "$FLATBOI_CMAKE"; then
    sudo mkdir -p /src/libs
    sudo ln -sfn "$LIBIGL_DIR" /src/libs/libigl
  fi
fi

log "Configuring and building Flatboi at $FLATBOI_DIR"
mkdir -p "$FLATBOI_DIR/build"
pushd "$FLATBOI_DIR/build" >/dev/null
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIBIGL_WITH_PASTIX=ON \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="/usr/local/pastix;/usr/local/scotch" \
  -DPASTIX_ROOT="/usr/local/pastix" \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_FLAGS="-DSLIM_CACHED"
cmake --build . -j"$JOBS"
install -d "$INSTALL_PREFIX/bin"
install -m 0755 ./flatboi "$INSTALL_PREFIX/bin/flatboi"
echo "==> flatboi installed at: $INSTALL_PREFIX/bin/flatboi"
popd >/dev/null

# ---------------------------------------------------------------------------
# Build main project (VC3D) WITHOUT PaStiX
# ---------------------------------------------------------------------------
log "Configuring & building VC3D (no PaStiX)"
VC3D_PREFIX_PATH="$INSTALL_PREFIX"
[[ -n "$CERES_PREFIX" ]] && VC3D_PREFIX_PATH="$CERES_PREFIX;$INSTALL_PREFIX"
rm -rf "$REPO_ROOT/build"
mkdir -p "$REPO_ROOT/build"
pushd "$REPO_ROOT/build" >/dev/null
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_PREFIX_PATH="$VC3D_PREFIX_PATH" \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DVC_WITH_PASTIX=OFF \
  -DVC_WITH_CUDA_SPARSE=$([[ "$USE_CUDA" == "1" ]] && echo ON || echo OFF)
cmake --build . -j"$JOBS"
popd >/dev/null
log "VC3D built successfully."

log "All done."
