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
  qt6-base-dev libboost-system-dev libboost-program-options-dev libceres-dev \
  libopencv-dev libxsimd-dev libblosc-dev libspdlog-dev libgsl-dev libsdl2-dev \
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
cp ./Make.inc/Makefile.inc.x86-64_pc_linux2 Makefile.inc
make -j"$JOBS" scotch
sudo mkdir -p /usr/local/scotch/{bin,include,lib,share/man/man1}
make prefix=/usr/local/scotch install || true
sudo cp -f ../lib/*scotch*.a /usr/local/scotch/lib/ || true
sudo cp -f ../include/*scotch*.h /usr/local/scotch/include/ || true
popd >/dev/null

log "Building PaStiX 5.2.3 into /usr/local/pastix (linked to /usr/local/scotch)"
PASTIX_SRC="$BUILD_DIR/pastix"
sudo mkdir -p /usr/local/pastix
rm -rf "$PASTIX_SRC"; mkdir -p "$PASTIX_SRC"
tar -xjf "$LIBS_DIR/pastix_5.2.3.tar.bz2" -C "$PASTIX_SRC" --strip-components=1
pushd "$PASTIX_SRC/src" >/dev/null
cp "$LIBS_DIR/config.in" config.in
sed -i -E "s|^ROOT[[:space:]]*=.*$|ROOT = /usr/local/pastix|" config.in
sed -i -E "s|^SCOTCH_HOME[[:space:]]*=.*$|SCOTCH_HOME = /usr/local/scotch|" config.in
make -j"$JOBS" SCOTCH_HOME=/usr/local/scotch
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
    's|^([[:space:]]*)add_subdirectory\([[:space:]]*/src/libs/libigl[^)]*\)|\1add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../.. ${CMAKE_BINARY_DIR}/libigl-build)|' \
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
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build . -j"$JOBS"
install -d "$INSTALL_PREFIX/bin"
install -m 0755 ./flatboi "$INSTALL_PREFIX/bin/flatboi"
echo "==> flatboi installed at: $INSTALL_PREFIX/bin/flatboi"
popd >/dev/null

# ---------------------------------------------------------------------------
# Build main project (VC3D) WITHOUT PaStiX
# ---------------------------------------------------------------------------
log "Configuring & building VC3D (no PaStiX)"
mkdir -p "$REPO_ROOT/build"
pushd "$REPO_ROOT/build" >/dev/null
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DVC_BUILD_Z5=OFF \
  -DVC_BUILD_JSON=OFF \
  -DVC_WITH_PASTIX=OFF \
  -DVC_WITH_CUDA_SPARSE=OFF
cmake --build . -j"$JOBS"
popd >/dev/null
log "VC3D built successfully (without PaStiX)."

log "All done."
