#!/bin/bash

set -e  # Exit on error
set -x  # Print commands as they execute

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root (use sudo)"
    exit 1
fi

print_status "Starting AWS GPU instance system provisioning..."

print_status "Updating and upgrading system packages..."
apt update
apt upgrade -y

print_status "Installing build essentials and development tools..."
apt install -y \
    build-essential \
    gcc \
    g++ \
    clang \
    make \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    tree \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    linux-headers-$(uname -r)

print_status "Installing NVIDIA drivers..."
apt remove --purge -y nvidia-* cuda-* 2>/dev/null || true
apt autoremove -y

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

apt update
apt install -y nvidia-driver-570
apt install -y cuda-toolkit-12-8
apt install -y cudnn9-cuda-12

print_status "Configuring NVMe drives..."

apt install -y btrfs-progs
NVME_DRIVES=($(ls /dev/nvme*n1 2>/dev/null | grep -v p))
NVME_COUNT=${#NVME_DRIVES[@]}

if [ $NVME_COUNT -eq 0 ]; then
    print_warning "No NVMe drives found"
elif [ $NVME_COUNT -eq 1 ]; then
    print_status "Found 1 NVMe drive: ${NVME_DRIVES[0]}"
    mkfs.btrfs -f ${NVME_DRIVES[0]}
    mkdir -p /vesuvius
    mount ${NVME_DRIVES[0]} /vesuvius
    UUID=$(blkid -s UUID -o value ${NVME_DRIVES[0]})
    echo "UUID=$UUID /vesuvius btrfs defaults 0 0" >> /etc/fstab

    print_status "Single NVMe drive formatted and mounted at /vesuvius"
else
    print_status "Found $NVME_COUNT NVMe drives: ${NVME_DRIVES[@]}"
    mkfs.btrfs -f -d raid0 ${NVME_DRIVES[@]}
    mkdir -p /vesuvius
    mount ${NVME_DRIVES[0]} /vesuvius
    UUID=$(blkid -s UUID -o value ${NVME_DRIVES[0]})
    echo "UUID=$UUID /vesuvius btrfs defaults 0 0" >> /etc/fstab

    print_status "RAID0 array created with $NVME_COUNT drives and mounted at /vesuvius"
fi

if [ -d /vesuvius ]; then
    chmod 755 /vesuvius
    chown root:root /vesuvius
fi

print_status "Verifying installations..."

if nvidia-smi &>/dev/null; then
    print_status "NVIDIA driver installed successfully"
    nvidia-smi
else
    print_error "NVIDIA driver installation failed"
fi

if mountpoint -q /vesuvius; then
    print_status "/vesuvius is mounted"
    df -h /vesuvius
    btrfs filesystem show /vesuvius
else
    print_warning "/vesuvius is not mounted (no NVMe drives found or mount failed)"
fi

print_status "System provisioning complete!"
print_status "Note: You may need to reboot for all changes to take effect"
print_status "Next: Run provision-python.sh as a regular user to set up Python environment"