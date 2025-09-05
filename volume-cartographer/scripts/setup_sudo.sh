#!/bin/bash
set -e

# Update and install system packages
sudo apt -y update
sudo apt -y install software-properties-common
sudo add-apt-repository -y universe
sudo apt -y update
sudo apt -y upgrade
sudo apt -y full-upgrade
sudo apt -y install build-essential git qt6-base-dev libboost-system-dev \
    libboost-program-options-dev libgsl-dev libsdl2-dev \
    libcurl4-openssl-dev file curl unzip ca-certificates bzip2 wget \
    fuse jq gimp desktop-file-utils ninja-build libomp-dev libgomp1 \
    ccache lld clang llvm libgmp-dev libmpfr-dev libsuitesparse-dev \
    libeigen3-dev zlib1g-dev liblapack-dev libblas-dev libmetis-dev \
    libavcodec-dev libavformat-dev libswscale-dev libavutil-dev \
    libv4l-dev v4l-utils libtbb-dev

# Install CMake from official release
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v4.1.0/cmake-4.1.0-linux-x86_64.sh"
elif [ "$ARCH" = "aarch64" ]; then
    CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v4.1.0/cmake-4.1.0-linux-aarch64.sh"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

echo "Downloading CMake for $ARCH..."
wget -q "$CMAKE_URL" -O cmake-installer.sh
chmod +x cmake-installer.sh
echo "Installing CMake to /usr/local..."
sudo bash cmake-installer.sh --skip-license --prefix=/usr/local --exclude-subdir
rm cmake-installer.sh

# Install AWS CLI
if [ "$ARCH" = "x86_64" ]; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
elif [ "$ARCH" = "aarch64" ]; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

unzip awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws

# Install GL/GLib packages for Open3D (x86_64 only)
if [ "$(uname -m)" = "x86_64" ]; then
    sudo apt -y install libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 libegl1
fi

# Clean up
sudo apt -y autoremove

echo "System setup complete!"