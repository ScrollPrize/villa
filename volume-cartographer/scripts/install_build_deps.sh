#!/usr/bin/env bash
# install_build_deps.sh — VC3D build toolchain. Shared by the CI Dockerfile
# and scripts/ec2_setup.sh. Runs apt-get directly; caller must be root.

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -y
apt-get install -y --no-install-recommends software-properties-common ca-certificates curl unzip
add-apt-repository -y universe
apt-get update -y
apt-get install -y --no-install-recommends \
    build-essential clang lld llvm flang-21 libclang-rt-21-dev mold git cmake ninja-build pkg-config \
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
    file bzip2 wget jq \
    python3 python3-venv

ln -sf /usr/bin/flang-21 /usr/local/bin/flang

# AWS CLI v2 (architecture-aware official installer).
arch="$(uname -m)"
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${arch}.zip" -o /tmp/awscli.zip
unzip -q /tmp/awscli.zip -d /tmp
/tmp/aws/install --update
rm -rf /tmp/awscli.zip /tmp/aws
