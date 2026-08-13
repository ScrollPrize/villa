#!/usr/bin/env bash
# ec2_setup.sh — bootstrap a fresh Ubuntu EC2 host for VC3D development.
# Run once as root. After this completes, build VC3D with:
#   cmake --preset ci-release-gcc
#   cmake --build --preset ci-release-gcc
#
# The core build toolchain is installed by scripts/install_build_deps.sh
# (shared with the CI Dockerfile). This script runs that, then layers the
# EC2-specific extras on top: timezone, ephemeral NVMe RAID, and GUI/sync
# tooling.

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

if [[ $(id -u) -ne 0 ]]; then
  echo "Run as root: sudo bash $0" >&2
  exit 1
fi
TARGET_USER="${SUDO_USER:-root}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

log "tzdata: set UTC"
apt-get update
ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime
apt-get install -y --no-install-recommends tzdata
dpkg-reconfigure -f noninteractive tzdata

log "Core build toolchain (scripts/install_build_deps.sh)"
bash "$SCRIPT_DIR/install_build_deps.sh"

log "EC2 extras: sync + GUI + hardware tooling"
apt-get install -y --no-install-recommends \
    rclone fuse gimp desktop-file-utils \
    mdadm nvme-cli lsb-release

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

log "Done. Next: cmake --preset ci-release-gcc && cmake --build --preset ci-release-gcc"
