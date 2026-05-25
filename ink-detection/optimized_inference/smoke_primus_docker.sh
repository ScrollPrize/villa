#!/usr/bin/env bash
set -euo pipefail

if [[ "${AGENTS_AGENT_MODE:-0}" == "1" && "${AGENTS_ALLOW_INSTALL:-0}" != "1" ]]; then
  echo "INFO: smoke_primus_docker.sh is disabled by default in agent mode."
  echo "Set AGENTS_ALLOW_INSTALL=1 to run this script."
  echo "Example: AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 ./smoke_primus_docker.sh"
  exit 0
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-ink-detection-optimized-inference:gpu-primus-smoke}"
VILLA_REPO="${VILLA_REPO:-https://github.com/ScrollPrize/villa.git}"
VILLA_REF="${VILLA_REF:-main}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:---gpus all}"

docker build --target gpu \
  --build-arg INSTALL_PRIMUS_DEPS=1 \
  --build-arg VILLA_REPO="$VILLA_REPO" \
  --build-arg VILLA_REF="$VILLA_REF" \
  -t "$IMAGE" \
  "$DIR"

cat <<'PY' | docker run --rm -i $DOCKER_GPU_ARGS \
  -e MODEL=dummy \
  -e MODEL_TYPE=primus \
  -e START_LAYER=0 \
  -e END_LAYER=16 \
  --entrypoint python \
  "$IMAGE" -
from pathlib import Path
import tempfile

import torch

import model_primus
from vesuvius.models.build.build_network_from_config import NetworkFromConfig

model_config = {
    "architecture_type": "primus_s",
    "primus_variant": "S",
    "patch_embed_size": [8, 8, 8],
    "input_shape": [16, 16, 16],
    "in_channels": 1,
    "targets": {"ink": {"out_channels": 1, "activation": "none"}},
    "decoder_head_channels": 4,
    "drop_path_rate": 0.0,
    "patch_drop_rate": 0.0,
    "proj_drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "num_register_tokens": 0,
}

source_model = NetworkFromConfig(model_primus._PrimusConfigShim(model_config))
checkpoint = {
    "model_config": model_config,
    "state_dict": source_model.state_dict(),
    "target_key": "ink",
}

with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_path = Path(tmpdir) / "primus-smoke.ckpt"
    torch.save(checkpoint, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = model_primus.load_model(str(checkpoint_path), device, num_frames=16)
    x = torch.randn(1, 1, 16, 16, 16, device=device)
    with torch.inference_mode():
        y = wrapper.forward(x)

assert y.shape == (1, 1, 16, 16, 16), y.shape
assert torch.isfinite(y).all()
print({"device": str(device), "shape": tuple(y.shape)})
PY
