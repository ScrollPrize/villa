#!/usr/bin/env bash
# Shared Debian/Ubuntu toolchain. No provisioning occurs before all guards pass.
set -euo pipefail

usage() {
    echo "Usage: $0 {minimal-build|ci-tooling}" >&2
}
profile="${1:-}"
if [[ $# != 1 || ( "$profile" != minimal-build && "$profile" != ci-tooling ) ]]; then
    usage
    exit 2
fi
if [[ "${AGENTS_AGENT_MODE:-0}" == 1 && "${AGENTS_ALLOW_INSTALL:-0}" != 1 ]]; then
    echo "Installation is disabled in agent mode. Set AGENTS_ALLOW_INSTALL=1 to allow it." >&2
    exit 1
fi
if [[ "$(uname -s)" != Linux || ! -r "${VC_OS_RELEASE_FILE:-/etc/os-release}" ]]; then
    echo "Build dependencies require Debian or Ubuntu Linux." >&2
    exit 1
fi
# The override allows distribution checks to be tested without provisioning.
# shellcheck disable=SC1090
source "${VC_OS_RELEASE_FILE:-/etc/os-release}"
case "${ID:-}" in
    debian|ubuntu) ;;
    *) echo "Unsupported distribution: ${ID:-unknown} (expected debian or ubuntu)." >&2; exit 1 ;;
esac
if (( EUID == 0 )); then
    elevate=()
elif command -v sudo >/dev/null 2>&1; then
    elevate=(sudo)
else
    echo "Root or sudo is required to install build dependencies." >&2
    exit 1
fi
export DEBIAN_FRONTEND=noninteractive
"${elevate[@]}" apt-get update
if [[ "$ID" == ubuntu ]]; then
    "${elevate[@]}" apt-get install -y --no-install-recommends software-properties-common
    "${elevate[@]}" add-apt-repository -y universe
    "${elevate[@]}" apt-get update
fi
packages=(
    build-essential ca-certificates cmake gfortran git ninja-build pkg-config
    qt6-base-dev libboost-program-options-dev libceres-dev libsuitesparse-dev
    libopencv-dev libopencv-contrib-dev libblosc-dev libzstd-dev libcurl4-openssl-dev
    nlohmann-json3-dev libavahi-client-dev libsqlite3-dev libssl-dev
    liblz4-dev libtiff-dev libpng-dev libjpeg-dev zlib1g-dev
    libopenblas-dev liblapack-dev liblapacke-dev libomp-dev
    libscotch-dev libscotchmetis-dev libhwloc-dev python3
)
if [[ "$profile" == ci-tooling ]]; then
    packages+=(clang lld llvm flang-21 libclang-rt-21-dev mold ccache
               file bzip2 wget jq valgrind python3-venv curl unzip)
fi
"${elevate[@]}" apt-get install -y --no-install-recommends "${packages[@]}"
if [[ "$profile" == ci-tooling ]]; then
    "${elevate[@]}" ln -sf /usr/bin/flang-21 /usr/local/bin/flang
    arch="$(uname -m)"
    case "$arch" in
        x86_64|aarch64) ;;
        *) echo "AWS CLI installer does not support $arch." >&2; exit 1 ;;
    esac
    aws_tmp="$(mktemp -d)"
    trap 'rm -rf "$aws_tmp"' EXIT
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${arch}.zip" -o "$aws_tmp/awscli.zip"
    unzip -q "$aws_tmp/awscli.zip" -d "$aws_tmp"
    "${elevate[@]}" "$aws_tmp/aws/install" --update
fi
