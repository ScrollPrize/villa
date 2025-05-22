#!/usr/bin/env bash
set -euo pipefail

# --- CONFIGURE BELOW IF NEEDED ---
CKPT_FILENAME="march2025.ckpt"
RENAMED_CKPT="last-epoch.ckpt"
REMOTE_PATH=":http:/other/dev/mask3d_checkpoints/${CKPT_FILENAME}"
HTTP_URL="http://dl.ash2txt.org/"
# -------------------------------

# Determine repo root (this script’s parent dir)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# Navigate into mask3d
MASK3D_DIR="${REPO_ROOT}/ThaumatoAnakalyptor/mask3d"
if [[ ! -d "$MASK3D_DIR" ]]; then
  echo "ERROR: directory not found: $MASK3D_DIR" >&2
  exit 1
fi
cd "$MASK3D_DIR"

# Make sure saved/train exists
mkdir -p saved/train
cd saved/train

# Download via rclone’s http backend
echo "→ Downloading ${CKPT_FILENAME} from ${HTTP_URL}${REMOTE_PATH#*:http:}"
rclone copy "${REMOTE_PATH}" . \
    --http-url "${HTTP_URL}" \
    --progress \
    --transfers=64 \
    --size-only \
    --fast-list

# Rename to last-epoch.ckpt
if [[ -f "$CKPT_FILENAME" ]]; then
  mv -v "$CKPT_FILENAME" "$RENAMED_CKPT"
  echo "✔ Renamed to $RENAMED_CKPT"
else
  echo "ERROR: download failed, $CKPT_FILENAME not found" >&2
  exit 1
fi

echo "All done in $(pwd)"
