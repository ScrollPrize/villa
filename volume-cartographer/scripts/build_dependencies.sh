#!/usr/bin/env bash
# build_dependencies.sh - one-shot setup for a fresh Ubuntu 24.04 host.
#
# Usage:
#   sudo bash scripts/build_dependencies.sh   # ROOT  pass: apt + system deps
#        bash scripts/build_dependencies.sh   # USER  pass: build z5/Flatboi/VC3D
#
# Detects arch (x86_64 / aarch64) and NVIDIA GPU.
#   GPU present → installs cuDSS, builds Ceres w/ CUDA sparse, enables NVENC
#                 in xpra, configures VC3D with -DVC_WITH_CUDA_SPARSE=ON.
#   GPU absent  → uses distro libceres-dev and disables CUDA sparse.

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBS_DIR="$REPO_ROOT/libs"

# -----------------------------------------------------------------------------
# Mode detection: root pass vs user pass
# -----------------------------------------------------------------------------
if [[ $(id -u) -eq 0 ]]; then
  MODE=root
  # When invoked via sudo, target the invoking user's home for build/install trees
  TARGET_USER="${SUDO_USER:-root}"
  TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
else
  MODE=user
  TARGET_USER="$USER"
  TARGET_HOME="$HOME"
fi

INSTALL_PREFIX="${INSTALL_PREFIX:-$TARGET_HOME/vc-dependencies}"
BUILD_DIR="${BUILD_DIR:-$TARGET_HOME/vc-dependencies-build}"
CERES_PREFIX_SYSTEM="/usr/local/ceres-cuda"

if command -v nproc >/dev/null 2>&1; then JOBS="$(nproc)"; else JOBS=4; fi

log() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

# -----------------------------------------------------------------------------
# Arch / GPU detection (both modes)
# -----------------------------------------------------------------------------
ARCH="$(uname -m)"
USE_CUDA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
  case "$ARCH" in x86_64|aarch64) USE_CUDA=1 ;; esac
fi
[[ "${VC_FORCE_NO_CUDA:-0}" == "1" ]] && USE_CUDA=0
case "$ARCH" in
  x86_64)  CUDSS_DEB_ARCH=amd64; CUDSS_LIBDIR=/usr/lib/x86_64-linux-gnu ;;
  aarch64) CUDSS_DEB_ARCH=arm64; CUDSS_LIBDIR=/usr/lib/aarch64-linux-gnu ;;
  *)       CUDSS_DEB_ARCH="";    CUDSS_LIBDIR="" ;;
esac

log "Mode: $MODE   Arch: $ARCH   USE_CUDA=$USE_CUDA   prefix=$INSTALL_PREFIX"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Linux only." >&2; exit 1
fi

# =============================================================================
# ROOT PASS
# =============================================================================
if [[ "$MODE" == "root" ]]; then
  log "apt: toolchain + libraries"
  apt-get update
  ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime
  apt-get install -y --no-install-recommends tzdata
  dpkg-reconfigure -f noninteractive tzdata

  apt-get install -y \
    build-essential git clang llvm ccache ninja-build lld cmake pkg-config \
    qt6-base-dev libboost-system-dev libboost-program-options-dev \
    libcgal-dev libsuitesparse-dev \
    libopencv-dev libxsimd-dev libblosc-dev libspdlog-dev libgsl-dev libsdl2-dev \
    libavahi-client-dev libde265-dev libx265-dev rclone nlohmann-json3-dev liblz4-dev \
    libcurl4-openssl-dev file curl unzip ca-certificates bzip2 wget fuse jq gimp \
    desktop-file-utils flex bison zlib1g-dev gfortran libopenblas-dev liblapack-dev \
    libscotch-dev libhwloc-dev libomp-dev \
    mdadm nvme-cli

  # ---- Ephemeral NVMe: RAID0 + mount at /ephemeral ------------------------
  # EC2 instance-store NVMe drives show up with model "Amazon EC2 NVMe Instance
  # Storage". Detect them, RAID0 into /dev/md0 (if >=2), format ext4, mount
  # at /ephemeral (owned by $TARGET_USER). Idempotent — skipped if already
  # mounted. Data here is lost on stop/terminate, never use for state we need.
  if ! mountpoint -q /ephemeral; then
    mapfile -t NVMES < <(lsblk -dno NAME,MODEL | awk '/Instance Storage/ {print "/dev/"$1}')
    if [[ ${#NVMES[@]} -eq 0 ]]; then
      log "Ephemeral: no local NVMe instance storage detected; skipping"
    else
      log "Ephemeral: found ${#NVMES[@]} NVMe devices: ${NVMES[*]}"
      for d in "${NVMES[@]}"; do umount "$d" 2>/dev/null || true; wipefs -a "$d" || true; done
      if [[ ${#NVMES[@]} -ge 2 ]]; then
        EPHEMERAL_DEV=/dev/md0
        mdadm --stop /dev/md0 2>/dev/null || true
        mdadm --create --verbose /dev/md0 --level=0 --raid-devices=${#NVMES[@]} \
              --force --run "${NVMES[@]}"
      else
        EPHEMERAL_DEV="${NVMES[0]}"
      fi
      mkfs.ext4 -F -E nodiscard -L ephemeral "$EPHEMERAL_DEV"
      mkdir -p /ephemeral
      mount -o noatime,nodiratime "$EPHEMERAL_DEV" /ephemeral
      chown "$TARGET_USER:$TARGET_USER" /ephemeral
      chmod 0755 /ephemeral
      log "Ephemeral: mounted $EPHEMERAL_DEV at /ephemeral ($(df -h /ephemeral | awk 'NR==2{print $2}'))"
    fi
  else
    log "Ephemeral: /ephemeral already mounted; skipping"
  fi

  # ---- CUDA path: cuDSS + Ceres-from-source --------------------------------
  if [[ "$USE_CUDA" == "1" ]]; then
    log "apt: CUDA toolkit"
    command -v nvcc >/dev/null 2>&1 || apt-get install -y nvidia-cuda-toolkit

    if ! dpkg -l | grep -q '^ii  cudss'; then
      log "cuDSS: install local repo deb"
      tmpc="$(mktemp -d)"; pushd "$tmpc" >/dev/null
      CUDSS_DEB="cudss-local-repo-ubuntu2404-0.4.0_0.4.0-1_${CUDSS_DEB_ARCH}.deb"
      wget -q "https://developer.download.nvidia.com/compute/cudss/0.4.0/local_installers/${CUDSS_DEB}"
      dpkg -i "$CUDSS_DEB"
      cp /var/cudss-local-repo-ubuntu2404-0.4.0/cudss-*-keyring.gpg /usr/share/keyrings/
      apt-get update
      apt-get install -y cudss
      popd >/dev/null; rm -rf "$tmpc"
    fi

    log "Ceres: build from source with CUDA → $CERES_PREFIX_SYSTEM"
    CERES_SRC="/tmp/ceres-solver"
    rm -rf "$CERES_SRC" "$CERES_PREFIX_SYSTEM"
    git clone https://github.com/ceres-solver/ceres-solver.git "$CERES_SRC"
    pushd "$CERES_SRC" >/dev/null
    git submodule update --init
    mkdir build && cd build
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -Dcudss_DIR="${CUDSS_LIBDIR}/libcudss/12/cmake/cudss" \
      -DSUITESPARSE=ON -DEIGENSPARSE=ON \
      -DUSE_CUDA=ON -DCUDA_SPARSE=ON -DACCELERATING_SPARSE_SCHUR=ON \
      -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF \
      -DCMAKE_CXX_FLAGS="-march=native" \
      -DCMAKE_INSTALL_PREFIX="$CERES_PREFIX_SYSTEM"
    cmake --build . -j"$JOBS"
    cmake --install .
    ldconfig
    popd >/dev/null
  else
    log "Ceres: distro libceres-dev"
    apt-get install -y libceres-dev
  fi

  # ---- Scotch + PaStiX (needed by Flatboi) ---------------------------------
  [[ -f "$LIBS_DIR/scotch_6.0.4.tar.gz" ]] || { echo "Missing $LIBS_DIR/scotch_6.0.4.tar.gz"; exit 1; }
  [[ -f "$LIBS_DIR/pastix_5.2.3.tar.bz2" ]] || { echo "Missing $LIBS_DIR/pastix_5.2.3.tar.bz2"; exit 1; }
  [[ -f "$LIBS_DIR/config.in" ]]              || { echo "Missing $LIBS_DIR/config.in";              exit 1; }

  log "Scotch 6.0.4 → /usr/local/scotch"
  SCOTCH_SRC="/tmp/scotch"
  rm -rf /usr/local/scotch "$SCOTCH_SRC"
  mkdir -p /usr/local/scotch "$SCOTCH_SRC"
  tar -xzf "$LIBS_DIR/scotch_6.0.4.tar.gz" -C "$SCOTCH_SRC" --strip-components=1
  pushd "$SCOTCH_SRC/src" >/dev/null
  cp ./Make.inc/Makefile.inc.x86-64_pc_linux2 Makefile.inc
  sed -i -E 's|^(CFLAGS[[:space:]]*=.*)$|\1 -std=gnu89 -fpermissive|' Makefile.inc
  make -j"$JOBS" scotch
  mkdir -p /usr/local/scotch/{bin,include,lib,share/man/man1}
  make prefix=/usr/local/scotch install
  find /usr/local/scotch/lib -name "libscotch*.a" -delete || true
  popd >/dev/null

  log "PaStiX 5.2.3 → /usr/local/pastix"
  PASTIX_SRC="/tmp/pastix"
  rm -rf /usr/local/pastix "$PASTIX_SRC"
  mkdir -p /usr/local/pastix "$PASTIX_SRC"
  tar -xjf "$LIBS_DIR/pastix_5.2.3.tar.bz2" -C "$PASTIX_SRC" --strip-components=1
  pushd "$PASTIX_SRC/src" >/dev/null
  cp "$LIBS_DIR/config.in" config.in
  sed -i -E "s|^SCOTCH_HOME[[:space:]]*=.*$|SCOTCH_HOME = /usr/local/scotch|" config.in
  sed -i -E 's|^(CCFOPT[[:space:]]*=[[:space:]]*-O3.*)$|\1 -std=gnu89 -fpermissive -Wno-error=implicit-function-declaration -Wno-error=incompatible-pointer-types -Wno-error=int-conversion -Wno-error=implicit-int|' config.in
  make SCOTCH_HOME=/usr/local/scotch
  make install SCOTCH_HOME=/usr/local/scotch
  popd >/dev/null

  # ---- xpra (build from source — works on any distro/python) -------------
  log "xpra: install build deps + build from source"
  apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3-setuptools python3-wheel cython3 \
    libcairo2-dev libgirepository-2.0-dev \
    libgtk-3-dev gir1.2-gtk-3.0 \
    python3-gi python3-cairo python3-gi-cairo python3-pil \
    libxres-dev libxtst-dev libxkbfile-dev \
    libxcb1-dev libxcb-render0-dev libxcb-randr0-dev libxcb-shape0-dev \
    libxcb-composite0-dev libxcb-damage0-dev libxcb-keysyms1-dev \
    libxcb-cursor-dev libxcb-image0-dev libxcb-xfixes0-dev \
    libavcodec-dev libavformat-dev libswscale-dev libavutil-dev libavfilter-dev \
    libvpx-dev libwebp-dev libturbojpeg0-dev libyuv-dev \
    libpulse-dev gstreamer1.0-plugins-base python3-gst-1.0 \
    libdbus-1-dev libdbus-glib-1-dev python3-dbus \
    libpam0g-dev libsystemd-dev \
    xvfb

  XPRA_SRC=/tmp/xpra
  rm -rf "$XPRA_SRC"
  git clone --depth 1 https://github.com/Xpra-org/xpra.git "$XPRA_SRC"
  pushd "$XPRA_SRC" >/dev/null
  python3 -m pip install --break-system-packages .
  popd >/dev/null
  log "xpra installed: $(command -v xpra)"

  log "ROOT pass complete. Next: run WITHOUT sudo to build VC3D."
  exit 0
fi

# =============================================================================
# USER PASS
# =============================================================================
export CC="ccache gcc"
export CXX="ccache g++"

# libigl pin
LIBIGL_COMMIT="ae8f959ea26d7059abad4c698aba8d6b7c3205e8"
LIBIGL_DIR="$LIBS_DIR/libigl"
LIBIGL_CHANGES_DIR="$LIBS_DIR/libigl_changes"

# Sanity: confirm root pass has run
for p in /usr/local/scotch/lib /usr/local/pastix/lib /usr/include/nlohmann; do
  [[ -d "$p" ]] || { echo "Missing $p — did you run 'sudo bash $0' first?"; exit 1; }
done
if [[ "$USE_CUDA" == "1" ]]; then
  [[ -d "$CERES_PREFIX_SYSTEM" ]] || { echo "Missing $CERES_PREFIX_SYSTEM — rerun root pass"; exit 1; }
fi

rm -rf "$BUILD_DIR" "$INSTALL_PREFIX"
mkdir -p "$BUILD_DIR" "$INSTALL_PREFIX"

# ---- libigl + Flatboi ------------------------------------------------------
log "libigl clone (pinned)"
rm -rf "$LIBIGL_DIR"
git clone https://github.com/libigl/libigl.git "$LIBIGL_DIR"
pushd "$LIBIGL_DIR" >/dev/null
git fetch origin "$LIBIGL_COMMIT" --depth 1
git checkout --detach "$LIBIGL_COMMIT"
git submodule update --init --recursive
popd >/dev/null
[[ -d "$LIBIGL_CHANGES_DIR" ]] && cp -a "$LIBIGL_CHANGES_DIR/." "$LIBIGL_DIR/"

FLATBOI_DIR="$LIBS_DIR/flatboi"
FLATBOI_CMAKE="$FLATBOI_DIR/CMakeLists.txt"
if [[ -f "$FLATBOI_CMAKE" ]]; then
  sed -i -E \
    's|^([[:space:]]*)add_subdirectory\([[:space:]]*/src/libs/libigl[^)]*\)|\1add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../libigl ${CMAKE_BINARY_DIR}/libigl-build)|' \
    "$FLATBOI_CMAKE" || true
fi

log "Flatboi build"
mkdir -p "$FLATBOI_DIR/build"
pushd "$FLATBOI_DIR/build" >/dev/null
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIBIGL_WITH_PASTIX=ON -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="/usr/local/pastix;/usr/local/scotch" \
  -DPASTIX_ROOT="/usr/local/pastix" \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_FLAGS="-DSLIM_CACHED"
cmake --build . -j"$JOBS"
install -d "$INSTALL_PREFIX/bin"
install -m 0755 ./flatboi "$INSTALL_PREFIX/bin/flatboi"
popd >/dev/null

# ---- VC3D ------------------------------------------------------------------
log "VC3D configure + build"
VC3D_PREFIX_PATH="$INSTALL_PREFIX"
[[ "$USE_CUDA" == "1" ]] && VC3D_PREFIX_PATH="$CERES_PREFIX_SYSTEM;$INSTALL_PREFIX"
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

log "Done. VC3D binaries in $REPO_ROOT/build/bin/"
