#!/usr/bin/env bash
# install_build_deps.sh — VC3D build toolchain. Shared by the CI Dockerfile
# and scripts/ec2_setup.sh. Runs apt-get directly; caller must be root.

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -y
apt-get install -y --no-install-recommends software-properties-common ca-certificates curl unzip gnupg
add-apt-repository -y universe
apt-get update -y

# flang-21 / libclang-rt-21-dev ship in Ubuntu 26.04's default archives, but
# not in older releases (e.g. 22.04 jammy, used by local dev machines, Colab,
# and any ec2_setup.sh host that isn't on 26.04 yet). Fall back to the
# official LLVM apt repo for this codename so the install below still
# resolves; this is a no-op wherever the packages are already native.
if ! apt-cache show flang-21 >/dev/null 2>&1; then
    codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"
    curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg
    echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/${codename}/ llvm-toolchain-${codename}-21 main" \
        > /etc/apt/sources.list.d/llvm-21.list
    apt-get update -y
fi

apt-get install -y --no-install-recommends \
    build-essential clang lld llvm flang-21 libclang-rt-21-dev mold git cmake ninja-build ccache pkg-config \
    qt6-base-dev \
    libboost-system-dev libboost-program-options-dev \
    libceres-dev libsuitesparse-dev \
    libopencv-dev libopencv-contrib-dev \
    libcgal-dev libmpfr-dev libgmp-dev \
    libblosc-dev libzstd-dev libcurl4-openssl-dev \
    nlohmann-json3-dev libavahi-client-dev \
    liblz4-dev libtiff-dev \
    zlib1g-dev gfortran libopenblas-dev liblapack-dev liblapacke-dev libomp-dev \
    libscotch-dev libscotchmetis-dev libhwloc-dev \
    file bzip2 wget jq valgrind \
    python3 python3-venv

ln -sf /usr/bin/flang-21 /usr/local/bin/flang

# AWS CLI v2 (architecture-aware official installer).
arch="$(uname -m)"
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${arch}.zip" -o /tmp/awscli.zip
unzip -q /tmp/awscli.zip -d /tmp
/tmp/aws/install --update
rm -rf /tmp/awscli.zip /tmp/aws
