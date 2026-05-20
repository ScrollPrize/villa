#!/usr/bin/env bash
# ec2_setup.sh — bootstrap a fresh Ubuntu EC2 host for VC3D development.
# Run once as root. After this completes, build VC3D with:
#   cmake --preset ci-release-gcc
#   cmake --build --preset ci-release-gcc

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

if [[ $(id -u) -ne 0 ]]; then
  echo "Run as root: sudo bash $0" >&2
  exit 1
fi
TARGET_USER="${SUDO_USER:-root}"

log() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

log "apt: toolchain + libraries"
apt-get update
ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime
apt-get install -y --no-install-recommends tzdata
dpkg-reconfigure -f noninteractive tzdata

apt-get install -y \
    build-essential git clang llvm ninja-build lld cmake pkg-config \
    qt6-base-dev libboost-system-dev libboost-program-options-dev \
    libceres-dev libsuitesparse-dev \
    libcgal-dev libmpfr-dev libgmp-dev \
    libopencv-dev libopencv-contrib-dev \
    libblosc-dev libcurl4-openssl-dev \
    libavahi-client-dev nlohmann-json3-dev \
    liblz4-dev libtiff-dev \
    zlib1g-dev gfortran libopenblas-dev liblapack-dev liblapacke-dev libomp-dev \
    libscotch-dev libscotchmetis-dev libhwloc-dev \
    file curl unzip ca-certificates bzip2 wget jq rclone fuse gimp \
    desktop-file-utils \
    mdadm nvme-cli

# ---- Ephemeral NVMe: RAID0 + mount at /ephemeral ----------------------------
# EC2 instance-store NVMes show up with model "Amazon EC2 NVMe Instance Storage".
# Detect them, RAID0 into /dev/md0 (if >=2), format ext4, mount at /ephemeral.
# Idempotent — skipped if already mounted. Data here is lost on stop/terminate.
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

# ---- xpra (from xpra.org apt repo — matches distro Python) ------------------
# We don't build from source because xpra master uses Python APIs newer than
# what noble's Python 3.12 ships. xpra.org publishes prebuilt packages per
# distro/codename, so derive the codename from the running system instead of
# hardcoding one.
log "xpra: add xpra.org apt repo and install"
install -d -m 0755 /usr/share/keyrings
wget -qO- https://xpra.org/xpra.asc | gpg --dearmor > /usr/share/keyrings/xpra.gpg
codename="$(lsb_release -cs)"
echo "deb [signed-by=/usr/share/keyrings/xpra.gpg] https://xpra.org/ ${codename} main" \
  > /etc/apt/sources.list.d/xpra.list
apt-get update
apt-get install -y -o Dpkg::Options::="--force-confnew" xpra xpra-codecs xvfb

log "Done. Next: cmake --preset ci-release-gcc && cmake --build --preset ci-release-gcc"
