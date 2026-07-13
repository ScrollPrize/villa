#!/usr/bin/env python3
"""Pinned, bounded ResNet152/3D-decoder ink-model inference adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

EXPECTED_SHA256 = "36dd0de84b7b7aa6590184192c7415466cd8a1ba7c1e59f42c6373846373c3e0"
MAX_PIXELS = 4_194_304


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid(length: int, tile: int, stride: int) -> list[int]:
    positions = list(range(0, max(1, length - tile + 1), stride))
    end = max(0, length - tile)
    if not positions or positions[-1] != end:
        positions.append(end)
    return positions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    request = json.loads(args.request.read_text())
    if request.get("model_profile") != "resnet152-3d-decoder-62":
        raise ValueError("only resnet152-3d-decoder-62 is supported")
    if sha256(args.checkpoint) != EXPECTED_SHA256:
        raise ValueError("ResNet152 checkpoint SHA-256 mismatch")
    expected_commit = request["repository_commit"]
    actual_commit = subprocess.check_output(
        ["git", "-C", str(args.repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != expected_commit:
        raise ValueError(f"ink-model repository commit mismatch: {actual_commit}")

    import torch
    import torch.nn.functional as F
    import tifffile
    import zarr
    from PIL import Image

    sys.path.insert(0, str(args.repository / "optimized_inference"))
    from model_resnet3d_3d_decoder import load_model

    manifest = json.loads((args.artifact / "manifest.json").read_text())
    if manifest.get("profile") != "resnet152-3d-decoder-62" or manifest.get("channels") != 62:
        raise ValueError("surface-volume artifact is not a ResNet152 62-layer stack")
    stack = zarr.open(str(args.artifact / "surface-volume.zarr"), mode="r")
    if len(stack.shape) != 3 or stack.shape[2] != 62 or stack.dtype != np.uint8:
        raise ValueError(f"expected uint8 HxWx62 Zarr, got {stack.shape} {stack.dtype}")
    height, width, _ = stack.shape
    if height * width > MAX_PIXELS:
        raise ValueError("ResNet152 inference is limited to 4,194,304 surface pixels")

    tile = int(request.get("tile_size", 64))
    stride = int(request.get("stride", max(1, tile // 2)))
    if tile not in (64, 128, 256) or stride < 1 or stride > tile:
        raise ValueError("tile_size must be 64/128/256 and 1 <= stride <= tile_size")
    device_name = request.get("device", "mps")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device_name not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be cpu, mps, or cuda")
    device = torch.device(device_name)
    model = load_model(str(args.checkpoint), device, num_frames=62)

    prediction = np.zeros((height, width), np.float32)
    counts = np.zeros((height, width), np.float32)
    window = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    window /= max(float(window.sum()), 1.0)
    reverse = bool(request.get("reverse_layers", False))
    positions = [(y, x) for y in grid(height, tile, stride) for x in grid(width, tile, stride)]
    for index, (y, x) in enumerate(positions, 1):
        y2, x2 = min(height, y + tile), min(width, x + tile)
        block = np.asarray(stack[y:y2, x:x2, :], dtype=np.uint8)
        padded = np.zeros((tile, tile, 62), np.uint8)
        padded[: y2 - y, : x2 - x] = block
        valid = np.any(padded != 0, axis=-1).astype(np.float32)
        if reverse:
            padded = padded[..., ::-1].copy()
        padded = np.clip(padded, 0, 200).astype(np.float32) / 200.0
        tensor = torch.from_numpy(np.moveaxis(padded, -1, 0)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model.forward(tensor)
            logits = F.interpolate(logits, size=(tile, tile), mode="bilinear", align_corners=False)
            probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        weighted = probability * window * valid
        prediction[y:y2, x:x2] += weighted[: y2 - y, : x2 - x]
        counts[y:y2, x:x2] += (window * valid)[: y2 - y, : x2 - x]
        print(f"progress {index}/{len(positions)}", flush=True)
    supported = counts > 0
    prediction = np.divide(prediction, counts, out=np.zeros_like(prediction), where=supported)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "ink-model-score.npy", prediction, allow_pickle=False)
    np.save(args.output / "ink-valid.npy", supported.astype(np.uint8), allow_pickle=False)
    tifffile.imwrite(args.output / "ink-model-score.tif", prediction)
    Image.fromarray(np.clip(np.rint(prediction * 255), 0, 255).astype(np.uint8)).save(
        args.output / "ink-model-score.png"
    )
    output_manifest = {
        "kind": "resnet152_ink_model_score_v1",
        "score_semantics": "uncalibrated_ink_model_score_not_probability_or_proof_of_ink",
        "score_display_name": "ResNet152 ink-model score",
        "model_profile": "resnet152-3d-decoder-62",
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": EXPECTED_SHA256,
        "repository_commit": actual_commit,
        "surface_volume_artifact": request["surface_volume"],
        "source_surface_artifact": manifest.get("source_artifact"),
        "input_shape_hwc": list(stack.shape),
        "output_shape_hw": list(prediction.shape),
        "device": device_name,
        "tile_size": tile,
        "stride": stride,
        "reverse_layers": reverse,
        "preprocessing": {"clip": [0, 200], "scale_divisor": 200.0},
        "blending": "normalized 2D Hann overlap-add with raw nonzero validity",
        "minimum_score": float(prediction.min()),
        "maximum_score": float(prediction.max()),
        "mean_score": float(prediction.mean()),
        "supported_pixels": int(supported.sum()),
        "artifacts": ["ink-model-score.npy", "ink-valid.npy", "ink-model-score.tif", "ink-model-score.png"],
    }
    (args.output / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(json.dumps(output_manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
