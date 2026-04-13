#!/bin/bash
set -euo pipefail

# ============================================================================
# vc3d_aws_setup.sh — One-shot setup for VC3D remote development on AWS EC2
#
# Usage:
#   ./vc3d_aws_setup.sh --key ~/.ssh/my.pem --host ubuntu@ec2-1-2-3-4.compute-1.amazonaws.com
#   ./vc3d_aws_setup.sh --key ~/.ssh/my.pem --host ubuntu@ec2-1-2-3-4.compute-1.amazonaws.com --branch main
#   ./vc3d_aws_setup.sh --host-only    # Only set up the local machine
#   ./vc3d_aws_setup.sh --remote-only --key ~/.ssh/my.pem --host ubuntu@...  # Only set up remote
#
# Supports: x86_64 and aarch64 on both host and target
#           GPU and non-GPU instances
#           Linux host (macOS/Windows planned)
# ============================================================================

# --- Defaults ---
SSH_KEY=""
REMOTE_HOST=""
BRANCH="tiled_renderer"
REPO_URL="https://github.com/SuperOptimizer/villa.git"
HOST_ONLY=false
REMOTE_ONLY=false
SKIP_LOCAL_XPRA_REBUILD=false

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}===${NC} $*"; }
warn() { echo -e "${YELLOW}WARNING:${NC} $*"; }
err()  { echo -e "${RED}ERROR:${NC} $*" >&2; }
step() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; log "$@"; echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --key)        SSH_KEY="$2"; shift 2 ;;
        --host)       REMOTE_HOST="$2"; shift 2 ;;
        --branch)     BRANCH="$2"; shift 2 ;;
        --repo)       REPO_URL="$2"; shift 2 ;;
        --host-only)  HOST_ONLY=true; shift ;;
        --remote-only) REMOTE_ONLY=true; shift ;;
        --skip-local-xpra-rebuild) SKIP_LOCAL_XPRA_REBUILD=true; shift ;;
        -h|--help)
            echo "Usage: $0 --key <ssh-key.pem> --host <user@ec2-host>"
            echo ""
            echo "Options:"
            echo "  --key <path>       SSH private key for EC2 (required unless --host-only)"
            echo "  --host <user@host> EC2 hostname (required unless --host-only)"
            echo "  --branch <name>    Git branch to build (default: tiled_renderer)"
            echo "  --repo <url>       Git repo URL (default: $REPO_URL)"
            echo "  --host-only        Only set up the local machine"
            echo "  --remote-only      Only set up the remote machine"
            echo "  --skip-local-xpra-rebuild  Skip rebuilding xpra locally from source"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Validate ---
if [[ "$HOST_ONLY" == false ]]; then
    if [[ -z "$SSH_KEY" ]]; then
        err "Missing --key <ssh-key.pem>"
        exit 1
    fi
    if [[ -z "$REMOTE_HOST" ]]; then
        err "Missing --host <user@ec2-host>"
        exit 1
    fi
    if [[ ! -f "$SSH_KEY" ]]; then
        err "SSH key not found: $SSH_KEY"
        exit 1
    fi
    # Fix key permissions if needed
    KEY_PERMS=$(stat -c %a "$SSH_KEY" 2>/dev/null || stat -f %Lp "$SSH_KEY" 2>/dev/null)
    if [[ "$KEY_PERMS" != "600" && "$KEY_PERMS" != "400" ]]; then
        warn "Fixing SSH key permissions (was $KEY_PERMS, setting to 600)"
        chmod 600 "$SSH_KEY"
    fi
fi

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
ssh_cmd() { ssh -i "$SSH_KEY" $SSH_OPTS "$REMOTE_HOST" "$@"; }
ARCH=$(uname -m)

# ============================================================================
# COMMON PACKAGE LIST (shared between host and target)
# ============================================================================

# Xpra build + runtime dependencies
XPRA_APT_PACKAGES=(
    git pkg-config bc
    # X11 libs
    libx11-dev libxtst-dev libxcomposite-dev libxdamage-dev libxres-dev
    libxkbfile-dev libsystemd-dev liblz4-dev pandoc
    libgtk-3-dev libxxhash-dev
    libturbojpeg0-dev libwebp-dev
    xvfb xserver-xorg-video-dummy keyboard-configuration
    libpam0g-dev gobject-introspection
    # Python
    python3-dev python3-pip python3-setuptools cython3
    python3-cairo python3-gi python3-gi-cairo python3-opengl python3-pil
    python3-dbus python3-cryptography python3-netifaces python3-yaml python3-paramiko
    python3-lz4 python3-rencode
    python3-zeroconf python3-ifaddr python3-setproctitle
    python3-pyinotify
    # GStreamer (video/audio streaming)
    python3-gst-1.0 gstreamer1.0-tools
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
    gstreamer1.0-libav
    # Native video encoder libraries (compiled into xpra C extensions)
    libx264-dev libvpx-dev libopenh264-dev
)

# System prerequisites for fromscratch.sh (builds all VC3D deps from source)
# NOTE: clang/lld/llvm/flang are installed from apt.llvm.org (see install_llvm)
FROMSCRATCH_APT_PACKAGES=(
    # Build toolchain (clang/lld/llvm from apt.llvm.org, not distro)
    build-essential ccache ninja-build pkg-config
    git wget tar xz-utils ca-certificates software-properties-common
    # CMake (will be upgraded to 4.x if too old)
    cmake
    # Fortran (for OpenBLAS LAPACK)
    gfortran
    # X11/OpenGL dev libraries (needed by Qt6)
    libx11-dev libxext-dev libxfixes-dev libxi-dev
    libxrender-dev libxcb1-dev libx11-xcb-dev libxcb-glx0-dev
    libxcb-keysyms1-dev libxcb-image0-dev libxcb-shm0-dev
    libxcb-icccm4-dev libxcb-sync-dev libxcb-xfixes0-dev
    libxcb-shape0-dev libxcb-randr0-dev libxcb-render-util0-dev
    libxcb-util-dev libxcb-xinerama0-dev libxcb-xkb-dev
    libxcb-cursor-dev libxcb-composite0-dev libxcb-damage0-dev
    libxcb-dpms0-dev libxcb-dri2-0-dev libxcb-dri3-dev
    libxcb-present-dev libxcb-record0-dev libxcb-res0-dev
    libxcb-xinput-dev
    libxkbcommon-dev libxkbcommon-x11-dev libgl-dev libegl-dev
    # Font/text rendering (needed by Qt6)
    libfontconfig1-dev libfreetype-dev libharfbuzz-dev
    # D-Bus and ICU (needed by Qt6)
    libdbus-1-dev libicu-dev
    # Misc (needed by VC3D directly, not built by fromscratch.sh)
    libcurl4-openssl-dev libspdlog-dev libsdl2-dev libavahi-client-dev
    libblosc-dev libgsl-dev
    python3
)

# Latest stable LLVM version from apt.llvm.org
LLVM_VERSION=22

# ============================================================================
# FUNCTIONS
# ============================================================================

install_llvm() {
    # $1 = "local" or "remote"
    # Installs latest LLVM/clang from apt.llvm.org (supports amd64 + arm64)
    local where="$1"
    log "Installing LLVM $LLVM_VERSION toolchain from apt.llvm.org ($where)..."

    local script='
set -e
LLVM_VERSION='"$LLVM_VERSION"'

# Add LLVM apt key and repository
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null

# Detect Ubuntu codename
CODENAME=$(lsb_release -cs 2>/dev/null || grep VERSION_CODENAME /etc/os-release | cut -d= -f2)
echo "deb http://apt.llvm.org/$CODENAME/ llvm-toolchain-$CODENAME-$LLVM_VERSION main" | \
    sudo tee /etc/apt/sources.list.d/llvm.list >/dev/null

sudo apt-get update -qq
sudo apt-get install -y -qq \
    clang-$LLVM_VERSION lld-$LLVM_VERSION llvm-$LLVM_VERSION \
    clang-tools-$LLVM_VERSION \
    libomp-$LLVM_VERSION-dev \
    flang-$LLVM_VERSION 2>/dev/null || true

# Set up alternatives so "clang" / "lld" / "llvm-ar" etc. point to version $LLVM_VERSION
for tool in clang clang++ clang-cpp; do
    sudo update-alternatives --install /usr/bin/$tool $tool /usr/bin/$tool-$LLVM_VERSION 100 2>/dev/null || true
done
for tool in lld ld.lld llvm-ar llvm-ranlib llvm-nm llvm-strip llvm-objdump llvm-objcopy llvm-readelf llvm-symbolizer flang-new; do
    if [ -f /usr/bin/$tool-$LLVM_VERSION ]; then
        sudo update-alternatives --install /usr/bin/$tool $tool /usr/bin/$tool-$LLVM_VERSION 100 2>/dev/null || true
    fi
done

# Also create flang symlink if flang-new exists
if [ -f /usr/bin/flang-new-$LLVM_VERSION ] && [ ! -f /usr/bin/flang ]; then
    sudo ln -sf /usr/bin/flang-new-$LLVM_VERSION /usr/bin/flang
fi

echo "LLVM $LLVM_VERSION installed: $(clang --version | head -1)"
'

    if [[ "$where" == "remote" ]]; then
        ssh_cmd bash -c "$script"
    else
        bash -c "$script"
    fi
}

install_xpra_deps() {
    # $1 = "local" or "remote"
    local where="$1"
    log "Installing xpra dependencies ($where)..."

    local run=""
    if [[ "$where" == "remote" ]]; then
        run="ssh_cmd"
    fi

    # Install packages, tolerating missing ones
    $run sudo apt-get update -qq
    $run sudo apt-get install -y -qq "${XPRA_APT_PACKAGES[@]}" 2>/dev/null || true

    # Packages whose names vary by Ubuntu version
    $run sudo apt-get install -y -qq libgirepository-2.0-dev 2>/dev/null || \
      $run sudo apt-get install -y -qq libgirepository1.0-dev 2>/dev/null || true
    $run sudo apt-get install -y -qq python-gi-dev 2>/dev/null || \
      $run sudo apt-get install -y -qq python3-gi-dev 2>/dev/null || true
    $run sudo apt-get install -y -qq python3-cairo-dev 2>/dev/null || true
}

upgrade_cython() {
    # $1 = "local" or "remote"
    local where="$1"
    log "Checking Cython version ($where)..."

    local cmd_prefix=""
    if [[ "$where" == "remote" ]]; then
        cmd_prefix="ssh_cmd"
    fi

    local cython_ver
    cython_ver=$($cmd_prefix cython3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1 || echo "0.0")
    if command -v bc &>/dev/null && [[ "$(echo "$cython_ver < 3.1" | bc)" == "1" ]]; then
        log "Upgrading Cython ($cython_ver -> >= 3.1) — 3.0.x has PyMemoryView ABI bug"
        $cmd_prefix sudo pip3 install --break-system-packages 'cython>=3.1' 2>/dev/null || true
    else
        log "Cython $cython_ver is OK"
    fi
}

install_pyopengl_accelerate() {
    # $1 = "local" or "remote"
    local where="$1"
    log "Installing PyOpenGL-accelerate ($where)..."

    if [[ "$where" == "remote" ]]; then
        ssh_cmd sudo pip3 install --break-system-packages PyOpenGL-accelerate 2>/dev/null || true
    else
        sudo pip3 install --break-system-packages PyOpenGL-accelerate 2>/dev/null || \
          pip3 install --break-system-packages PyOpenGL-accelerate 2>/dev/null || true
    fi
}

build_xpra_from_source() {
    # $1 = "local" or "remote"
    local where="$1"
    step "Building xpra from source ($where)"

    if [[ "$where" == "remote" ]]; then
        ssh_cmd bash <<'REMOTE_XPRA_BUILD'
set -e
cd /tmp
sudo rm -rf xpra-build
git clone --depth 1 https://github.com/Xpra-org/xpra.git xpra-build
cd xpra-build

echo "Building xpra..."
python3 setup.py build 2>&1

echo "Installing xpra..."
sudo python3 setup.py install --prefix=/usr --install-layout=deb 2>&1 || \
  sudo python3 setup.py install --prefix=/usr 2>&1
sudo cp fs/bin/* /usr/bin/ 2>/dev/null || true

if [ ! -d /usr/share/xpra/css ]; then
    sudo mkdir -p /usr/share/xpra
    sudo cp -r fs/share/xpra/* /usr/share/xpra/ 2>/dev/null || true
fi

echo "=== Verifying xpra ==="
xpra --version
cd /tmp
python3 -c "from xpra.codecs.vpx import encoder; print('vpx encoder: OK')" 2>&1 || echo "vpx: not compiled"
python3 -c "from xpra.codecs.x264 import encoder; print('x264 encoder: OK')" 2>&1 || echo "x264: not compiled"
python3 -c "from xpra.codecs.openh264 import encoder; print('openh264 encoder: OK')" 2>&1 || echo "openh264: not compiled"
python3 -c "from xpra.net.lz4.lz4 import compress; print('lz4: OK')" 2>&1 || echo "lz4: not compiled"
REMOTE_XPRA_BUILD
    else
        cd /tmp
        sudo rm -rf xpra-build
        git clone --depth 1 https://github.com/Xpra-org/xpra.git xpra-build
        cd xpra-build

        log "Building xpra..."
        python3 setup.py build 2>&1

        log "Installing xpra..."
        sudo python3 setup.py install --prefix=/usr --install-layout=deb 2>&1 || \
          sudo python3 setup.py install --prefix=/usr 2>&1
        sudo cp fs/bin/* /usr/bin/ 2>/dev/null || true

        if [ ! -d /usr/share/xpra/css ]; then
            sudo mkdir -p /usr/share/xpra
            sudo cp -r fs/share/xpra/* /usr/share/xpra/ 2>/dev/null || true
        fi

        log "Verifying xpra..."
        xpra --version
        cd /tmp
        python3 -c "from xpra.codecs.vpx import encoder; print('vpx encoder: OK')" 2>&1 || warn "vpx: not compiled"
        python3 -c "from xpra.codecs.x264 import encoder; print('x264 encoder: OK')" 2>&1 || warn "x264: not compiled"
        python3 -c "from xpra.codecs.openh264 import encoder; print('openh264 encoder: OK')" 2>&1 || warn "openh264: not compiled"
        python3 -c "from xpra.net.lz4.lz4 import compress; print('lz4: OK')" 2>&1 || warn "lz4: not compiled"
    fi
}

# ============================================================================
# HOST (LOCAL) SETUP
# ============================================================================

setup_host() {
    step "Setting up local host ($ARCH)"

    install_llvm "local"
    install_xpra_deps "local"
    upgrade_cython "local"
    install_pyopengl_accelerate "local"

    # Remove distro xpra if present (we build from source)
    sudo apt-get remove -y xpra 2>/dev/null || true
    sudo rm -rf /usr/lib/python3/dist-packages/xpra* 2>/dev/null || true

    if [[ "$SKIP_LOCAL_XPRA_REBUILD" == false ]]; then
        build_xpra_from_source "local"
    else
        log "Skipping local xpra rebuild (--skip-local-xpra-rebuild)"
    fi

    log "Local host setup complete"
}

# ============================================================================
# REMOTE SETUP
# ============================================================================

setup_remote() {
    step "Setting up remote: $REMOTE_HOST"

    # Test SSH connectivity
    log "Testing SSH connection..."
    if ! ssh_cmd "echo 'SSH OK — $(uname -m) — $(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY | cut -d= -f2)'"; then
        err "Cannot SSH to $REMOTE_HOST with key $SSH_KEY"
        exit 1
    fi

    REMOTE_ARCH=$(ssh_cmd "uname -m")
    log "Remote architecture: $REMOTE_ARCH"

    # --- NVMe setup ---
    step "Setting up NVMe storage (remote)"
    ssh_cmd bash <<'NVME_SETUP'
set -e
ROOT_DISK=$(findmnt -n -o SOURCE / | sed 's/p[0-9]*$//' | sed 's/[0-9]*$//')
MOUNTED_NVME=false
for NVME in /dev/nvme*n1; do
    [ -b "$NVME" ] || continue
    if [ "$NVME" = "$ROOT_DISK" ]; then
        echo "  Skipping $NVME (root disk)"
        continue
    fi
    if mount | grep -q "^$NVME"; then
        MOUNT=$(mount | grep "^$NVME" | awk '{print $3}')
        echo "  $NVME already mounted at $MOUNT"
        MOUNTED_NVME=true
        continue
    fi
    FSTYPE=$(blkid -o value -s TYPE "$NVME" 2>/dev/null || true)
    if [ -z "$FSTYPE" ]; then
        echo "  Formatting $NVME as ext4..."
        sudo mkfs.ext4 -q "$NVME"
    fi
    MOUNT_DIR="/mnt/nvme-cache"
    sudo mkdir -p "$MOUNT_DIR"
    sudo mount "$NVME" "$MOUNT_DIR"
    sudo chown "$(whoami):$(whoami)" "$MOUNT_DIR"
    echo "  Mounted $NVME at $MOUNT_DIR ($(lsblk -n -o SIZE "$NVME" | head -1))"
    MOUNTED_NVME=true
done
if [ "$MOUNTED_NVME" = true ] && [ -d /mnt/nvme-cache ]; then
    mkdir -p /mnt/nvme-cache/vc3d-cache
    echo "  Cache dir: /mnt/nvme-cache/vc3d-cache"
else
    echo "  No extra NVMe drives found"
fi
NVME_SETUP

    # --- LLVM toolchain ---
    install_llvm "remote"

    # --- Xpra dependencies + build ---
    step "Installing xpra dependencies (remote)"
    ssh_cmd bash <<'REMOTE_XPRA_DEPS'
set -e
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git pkg-config bc \
    libx11-dev libxtst-dev libxcomposite-dev libxdamage-dev libxres-dev \
    libxkbfile-dev libsystemd-dev liblz4-dev pandoc \
    libgtk-3-dev libxxhash-dev \
    libturbojpeg0-dev libwebp-dev \
    xvfb xserver-xorg-video-dummy keyboard-configuration \
    libpam0g-dev gobject-introspection \
    python3-dev python3-pip python3-setuptools cython3 \
    python3-cairo python3-gi python3-gi-cairo python3-opengl python3-pil \
    python3-dbus python3-cryptography python3-netifaces python3-yaml python3-paramiko \
    python3-lz4 python3-rencode \
    python3-zeroconf python3-ifaddr python3-setproctitle \
    python3-pyinotify \
    python3-gst-1.0 gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libx264-dev libvpx-dev libopenh264-dev \
    2>/dev/null || true

sudo apt-get install -y -qq libgirepository-2.0-dev 2>/dev/null || \
  sudo apt-get install -y -qq libgirepository1.0-dev 2>/dev/null || true
sudo apt-get install -y -qq python-gi-dev 2>/dev/null || \
  sudo apt-get install -y -qq python3-gi-dev 2>/dev/null || true
sudo apt-get install -y -qq python3-cairo-dev 2>/dev/null || true

# Upgrade Cython if needed
CYTHON_VER=$(cython3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1 || echo "0.0")
if [ "$(echo "$CYTHON_VER < 3.1" | bc)" = "1" ] 2>/dev/null; then
    echo "Upgrading Cython ($CYTHON_VER -> >= 3.1)"
    sudo pip3 install --break-system-packages 'cython>=3.1' 2>/dev/null || true
fi

# PyOpenGL-accelerate
sudo pip3 install --break-system-packages PyOpenGL-accelerate 2>/dev/null || true

# Remove old xpra
sudo apt-get remove -y xpra 2>/dev/null || true
sudo rm -rf /usr/lib/python3/dist-packages/xpra* 2>/dev/null || true
REMOTE_XPRA_DEPS

    build_xpra_from_source "remote"

    # --- System prerequisites for fromscratch.sh ---
    step "Installing system prerequisites (remote)"
    ssh_cmd bash <<'REMOTE_PREREQS'
set -e
sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y universe 2>/dev/null || true
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends \
    build-essential ccache ninja-build pkg-config \
    git wget tar xz-utils ca-certificates \
    cmake gfortran python3 \
    libx11-dev libxext-dev libxfixes-dev libxi-dev \
    libxrender-dev libxcb1-dev libx11-xcb-dev libxcb-glx0-dev \
    libxcb-keysyms1-dev libxcb-image0-dev libxcb-shm0-dev \
    libxcb-icccm4-dev libxcb-sync-dev libxcb-xfixes0-dev \
    libxcb-shape0-dev libxcb-randr0-dev libxcb-render-util0-dev \
    libxcb-util-dev libxcb-xinerama0-dev libxcb-xkb-dev \
    libxkbcommon-dev libxkbcommon-x11-dev libgl-dev libegl-dev \
    libfontconfig1-dev libfreetype-dev libharfbuzz-dev \
    libdbus-1-dev libicu-dev \
    libcurl4-openssl-dev libspdlog-dev libsdl2-dev libavahi-client-dev \
    libblosc-dev libgsl-dev
REMOTE_PREREQS

    # --- CMake ---
    step "Ensuring CMake >= 3.22 (remote)"
    ssh_cmd bash <<REMOTE_CMAKE
set -e
ARCH=\$(uname -m)
if ! cmake --version 2>/dev/null | grep -qE '(3\.2[2-9]|3\.[3-9]|[4-9]\.)'; then
    echo "Installing CMake 4.1..."
    if [ "\$ARCH" = "x86_64" ]; then
        CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v4.1.0/cmake-4.1.0-linux-x86_64.sh"
    elif [ "\$ARCH" = "aarch64" ]; then
        CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v4.1.0/cmake-4.1.0-linux-aarch64.sh"
    else
        echo "Unsupported architecture: \$ARCH" && exit 1
    fi
    wget -q "\$CMAKE_URL" -O /tmp/cmake-installer.sh
    sudo bash /tmp/cmake-installer.sh --skip-license --prefix=/usr/local --exclude-subdir
    rm /tmp/cmake-installer.sh
fi
echo "CMake: \$(cmake --version | head -1)"
REMOTE_CMAKE

    # --- Clone repo ---
    step "Cloning repository (remote, branch: $BRANCH)"
    ssh_cmd bash <<REMOTE_CLONE
set -e
cd /home/ubuntu
if [ -d villa ]; then
    echo "villa/ exists, fetching..."
    cd villa
    git fetch --all
    git checkout "$BRANCH"
    git pull --ff-only || true
else
    git clone "$REPO_URL"
    cd villa
    git checkout "$BRANCH"
fi
REMOTE_CLONE

    # --- Build all dependencies from source ---
    step "Building all dependencies from source (fromscratch.sh)"
    ssh_cmd bash <<'REMOTE_FROMSCRATCH'
set -e
cd /home/ubuntu/villa/volume-cartographer
echo "Running scripts/fromscratch.sh..."
echo "This builds Qt6, OpenCV, Boost, Ceres, Eigen, etc. from source."
echo "First run may take 30-60+ minutes. Subsequent runs skip already-built deps."
echo ""
bash scripts/fromscratch.sh ~/vc-dependencies
REMOTE_FROMSCRATCH

    # --- Build VC3D ---
    step "Building VC3D (remote, branch: $BRANCH)"
    ssh_cmd bash <<REMOTE_BUILD
set -e
PREFIX=\$HOME/vc-dependencies
export PATH="\$PREFIX/bin:\$PATH"
export LD_LIBRARY_PATH="\$PREFIX/lib:\$PREFIX/lib64:\${LD_LIBRARY_PATH:-}"

cd /home/ubuntu/villa/volume-cartographer
mkdir -p build
cd build

cmake -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=\$PREFIX \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DVC_WITH_CUDA_SPARSE=off \
    ..

ninja -j\$(nproc) VC3D 2>&1

echo ""
ls -lh bin/VC3D
echo "Branch: \$(git -C .. branch --show-current)"
REMOTE_BUILD

    log "Remote setup complete"
}

# ============================================================================
# LAUNCH — Start xpra server + VC3D on remote, attach from local
# ============================================================================

launch() {
    step "Launching VC3D via xpra"

    # Start xpra server on remote
    log "Starting xpra server on remote..."
    ssh_cmd bash <<'REMOTE_LAUNCH'
set -e
# Stop existing
xpra stop :100 2>/dev/null || true
sleep 1

# Kill any lingering VC3D
killall VC3D 2>/dev/null || true
sleep 0.5

# Start xpra display
xpra start :100 --resize-display=yes 2>&1

sleep 2

# Verify encoders
echo "=== Server video encoders ==="
xpra info :100 2>/dev/null | strings | grep "video-encoder" | head -10

# Launch VC3D (with LD_LIBRARY_PATH for from-source deps)
BINARY="/home/ubuntu/villa/volume-cartographer/build/bin/VC3D"
PREFIX="$HOME/vc-dependencies"
if [ -x "$BINARY" ]; then
    export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"
    DISPLAY=:100 nohup "$BINARY" > /tmp/vc3d.log 2>&1 &
    sleep 2
    echo "VC3D launched (PID $(pgrep -f VC3D | head -1))"
else
    echo "WARNING: VC3D binary not found at $BINARY"
fi
REMOTE_LAUNCH

    # Attach from local
    log "Attaching xpra client..."
    local REMOTE_DISPLAY="${REMOTE_HOST##*@}"
    xpra attach "ssh://${REMOTE_HOST}:22/100" \
        --ssh="ssh -i $SSH_KEY $SSH_OPTS" \
        --encoding=stream \
        --speed=100 \
        --quality=90 \
        --min-quality=50 \
        --min-speed=80 \
        --compress=0 \
        --opengl=yes \
        &

    local XPRA_PID=$!
    sleep 3

    if kill -0 "$XPRA_PID" 2>/dev/null; then
        log "xpra client attached (PID $XPRA_PID)"
        log "Press Ctrl-C to detach"
        echo ""
        echo "  To reattach later:"
        echo "    xpra attach \"ssh://${REMOTE_HOST}:22/100\" --ssh=\"ssh -i $SSH_KEY\" --encoding=stream --speed=100 --quality=90 --opengl=yes"
        echo ""
        wait "$XPRA_PID" 2>/dev/null || true
    else
        warn "xpra client may have failed to attach"
        warn "Try manually: xpra attach \"ssh://${REMOTE_HOST}:22/100\" --ssh=\"ssh -i $SSH_KEY\""
    fi
}

# ============================================================================
# MAIN
# ============================================================================

echo ""
echo "  VC3D AWS Setup"
echo "  ────────────────────────────────"
echo "  Host arch:  $ARCH"
echo "  OS:         $(lsb_release -ds 2>/dev/null || cat /etc/os-release 2>/dev/null | grep PRETTY | cut -d= -f2 || echo 'unknown')"
if [[ -n "$REMOTE_HOST" ]]; then
    echo "  Remote:     $REMOTE_HOST"
    echo "  SSH key:    $SSH_KEY"
fi
echo "  Branch:     $BRANCH"
echo ""

if [[ "$REMOTE_ONLY" == true ]]; then
    setup_remote
    launch
elif [[ "$HOST_ONLY" == true ]]; then
    setup_host
else
    setup_host
    setup_remote
    launch
fi

step "Done!"
