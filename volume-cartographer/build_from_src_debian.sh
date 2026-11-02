#!/usr/bin/env bash

# Build and install VC3D and its command-line tools from this checkout.
# Supported platform: recent Debian-family distributions using APT.

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

if [[ "${AGENTS_AGENT_MODE:-0}" == 1 && "${AGENTS_ALLOW_INSTALL:-0}" != 1 ]]; then
    echo "Installation is disabled in agent mode. Set AGENTS_ALLOW_INSTALL=1 to allow it." >&2
    exit 1
fi

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${BUILD_DIR:-$source_dir/build/from-source}"
prefix="${PREFIX:-/usr/local}"
jobs="${JOBS:-$(nproc)}"

if [[ "$(uname -s)" != Linux ]] || [[ ! -r /etc/os-release ]]; then
    echo "build_from_src_debian.sh requires a Debian-family Linux distribution." >&2
    exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != debian && "${ID:-}" != ubuntu ]]; then
    echo "build_from_src_debian.sh requires a Debian-family distribution (found ${ID:-unknown})." >&2
    exit 1
fi
if ! command -v apt-get >/dev/null 2>&1 || ! command -v dpkg >/dev/null 2>&1; then
    echo "build_from_src_debian.sh requires apt-get and dpkg." >&2
    exit 1
fi

if (( EUID == 0 )); then
    sudo=()
elif command -v sudo >/dev/null 2>&1; then
    sudo=(sudo)
else
    echo "sudo is required to install system packages and files under $prefix." >&2
    exit 1
fi

"$source_dir/scripts/install_build_deps.sh" minimal-build

cmake_version="$(cmake --version 2>/dev/null | sed -n '1s/^cmake version //p' || true)"
if [[ -z "$cmake_version" ]] || ! dpkg --compare-versions "$cmake_version" ge 3.28; then
    echo "VC3D requires CMake 3.28 or newer; this system provides ${cmake_version:-unknown}." >&2
    exit 1
fi

echo "Configuring VC3D..."
cmake \
    -S "$source_dir" \
    -B "$build_dir" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DBLA_VENDOR=OpenBLAS \
    -DVC_BUILD_APPS=ON \
    -DVC_BUILD_FLATBOI=ON \
    -DVC_BUILD_PYTHON=OFF \
    -DVC_TESTING=OFF

echo "Building VC3D with $jobs parallel jobs..."
cmake --build "$build_dir" --parallel "$jobs"

echo "Installing VC3D under $prefix..."
"${sudo[@]}" cmake --install "$build_dir" --component vc_runtime

for program in VC3D vc_grow_seg_from_seed flatboi; do
    if [[ ! -x "$prefix/bin/$program" ]]; then
        echo "Installation failed: $prefix/bin/$program was not installed." >&2
        exit 1
    fi
done

if [[ ":$PATH:" != *":$prefix/bin:"* ]]; then
    echo "VC3D installed successfully, but $prefix/bin is not currently on PATH." >&2
    echo "Add this line to your shell profile:" >&2
    echo "  export PATH=\"$prefix/bin:\$PATH\"" >&2
fi

hash -r
echo "VC3D installed successfully. Try: VC3D"
