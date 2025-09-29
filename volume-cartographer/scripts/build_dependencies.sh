#!/usr/bin/env bash
# build_dependencies.sh - Drop-in replacement mirroring your Dockerfile flow
# - Builds VC3D WITHOUT PaStiX
# - Builds Scotch + PaStiX ONLY for libigl's 999_Flatboi
# - Fetches libigl from your tarball (not from GitHub)

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

# libigl tarball (your repo). If local tarball exists, use it; otherwise use URL.
LIBIGL_TARBALL_LOCAL="${LIBS_DIR}/libigl.tar.bz2"
LIBIGL_TARBALL_URL="${LIBIGL_TARBALL_URL:-https://dl.ash2txt.org/other/dev/libigl.tar.bz2}"

# Determine parallelism
if command -v nproc >/dev/null 2>&1; then
  JOBS="$(nproc)"
else
  JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi
export JOBS

log() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

# ---------------------------------------------------------------------------
# OS prerequisites (Ubuntu/Noble flow, like the Dockerfile)
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
  build-essential git clang llvm ccache ninja-build lld cmake \
  qt6-base-dev libboost-system-dev libboost-program-options-dev libceres-dev \
  libopencv-dev libxsimd-dev libblosc-dev libspdlog-dev libgsl-dev libsdl2-dev \
  libcurl4-openssl-dev file curl unzip ca-certificates bzip2 wget fuse jq gimp \
  desktop-file-utils flex bison zlib1g-dev gfortran libopenblas-dev liblapack-dev \
  libscotch-dev libhwloc-dev libomp-dev pkg-config

# Pin xtl + xtensor to match your Dockerfile exactly
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
# z5 (pinned to the commit your CMake expects) → $INSTALL_PREFIX
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
# Build your main project (VC3D) WITHOUT PaStiX
# - Use external z5 (VC_BUILD_Z5=OFF)
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

# ---------------------------------------------------------------------------
# Build Scotch + PaStiX ONLY for libigl/999_Flatboi, like your Dockerfile
# Scotch -> /usr/local/scotch
# PaStiX -> /usr/local/pastix
# ---------------------------------------------------------------------------
# Sanity check: required tarballs present
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
# Installer sometimes assumes dirs exist; create them first
sudo mkdir -p /usr/local/scotch/{bin,include,lib,share/man/man1}
make prefix=/usr/local/scotch install || true
# Ensure headers/libs are present even if install targets were picky
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
# libigl + 999_Flatboi (from your tarball)
#   - Patch the hard-coded /src/libs/libigl path
#   - Point CMake to /usr/local/pastix (and scotch) so PaStiX is found
# ---------------------------------------------------------------------------
log "Fetching libigl and building 999_Flatboi"

LIBIGL_DIR="$BUILD_DIR/libigl"
FLATBOI_DIR="$LIBIGL_DIR/tutorial/999_Flatboi"
FLATBOI_CMAKE="$FLATBOI_DIR/CMakeLists.txt"

rm -rf "$LIBIGL_DIR" "$BUILD_DIR/libigl.tar.bz2"
mkdir -p "$BUILD_DIR" "$LIBIGL_DIR"

if [[ -f "$LIBIGL_TARBALL_LOCAL" ]]; then
  cp -f "$LIBIGL_TARBALL_LOCAL" "$BUILD_DIR/libigl.tar.bz2"
else
  curl -fL "$LIBIGL_TARBALL_URL" -o "$BUILD_DIR/libigl.tar.bz2"
fi

tar -xjf "$BUILD_DIR/libigl.tar.bz2" -C "$LIBIGL_DIR" --strip-components=1

# Patch hard-coded absolute path in 999_Flatboi/CMakeLists.txt
if [[ -f "$FLATBOI_CMAKE" ]]; then
  sed -i -E \
    's|^([[:space:]]*)add_subdirectory\([[:space:]]*/src/libs/libigl[^)]*\)|\1add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../.. ${CMAKE_BINARY_DIR}/libigl-build)|' \
    "$FLATBOI_CMAKE"
fi

# Safety net: if anything still references /src/libs/libigl, provide a symlink
if grep -q "/src/libs/libigl" "$FLATBOI_CMAKE"; then
  sudo mkdir -p /src/libs
  sudo ln -sfn "$LIBIGL_DIR" /src/libs/libigl
fi

log "Configuring and building 999_Flatboi"
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

# Install flatboi into your local prefix (no sudo required)
install -d "$INSTALL_PREFIX/bin"
install -m 0755 ./flatboi "$INSTALL_PREFIX/bin/flatboi"
echo "==> flatboi installed at: $INSTALL_PREFIX/bin/flatboi"

popd >/dev/null

log "All done."
