#!/usr/bin/env python
"""
Convert a neural_tracing checkpoint trained with Vesuvius3dUnetModel
to the NetworkFromConfig-based NeuralTracingNet.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vesuvius.neural_tracing.models import NeuralTracingNet  # noqa: E402


def _parse_spacing(spacing_str: str) -> Tuple[float, float, float]:
    parts = spacing_str.split(",")
    if len(parts) != 3:
        raise ValueError("Spacing must be three comma-separated floats, e.g. '8,8,8'")
    return tuple(float(p) for p in parts)


def _remap_key(key: str) -> str | None:
    """
    Map old vesuvius_unet3d parameter names to NetworkFromConfig.
    Returns None for keys with no analogue (e.g., final_transpconv).
    """
    if key.startswith("encoder."):
        return "shared_encoder." + key[len("encoder.") :]
    if key.startswith("decoder."):
        rest = key[len("decoder.") :]
        if rest.startswith("final_transpconv."):
            return None  # No matching module
        if rest.startswith("final_seg_layer."):
            tail = rest[len("final_seg_layer.") :]
            return f"task_decoders.uv.seg_layers.3.{tail}"
        return "task_decoders.uv." + rest
    return None


def _map_state_dict(old_sd: Dict[str, torch.Tensor], new_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mapped: Dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    aliased: list[Tuple[str, str]] = []

    for old_key, tensor in old_sd.items():
        new_key = _remap_key(old_key)
        if new_key is None:
            skipped.append(old_key)
            continue

        def _try_set(k: str) -> bool:
            if k in new_sd and new_sd[k].shape == tensor.shape:
                mapped[k] = tensor
                return True
            return False

        if _try_set(new_key):
            continue

        aliases = []
        if ".conv." in new_key:
            aliases.append(new_key.replace(".conv.", ".all_modules.0."))
        if ".norm." in new_key:
            aliases.append(new_key.replace(".norm.", ".all_modules.1."))

        if any(_try_set(alias) for alias in aliases):
            aliased.append((old_key, next(k for k in aliases if k in mapped)))
            continue

        skipped.append(old_key)

    if skipped:
        print(f"Skipped {len(skipped)} old keys (no match or shape mismatch).")
    if aliased:
        print(f"Aliased {len(aliased)} keys to all_modules.* tensors.")
    return mapped


def convert(checkpoint_path: Path, config_path: Path, output_path: Path, spacing_override: Tuple[float, float, float] | None):
    cfg = json.loads(config_path.read_text())
    if spacing_override is not None:
        cfg = dict(cfg)
        cfg["spacing"] = spacing_override

    model = NeuralTracingNet(cfg)
    new_sd = model.state_dict()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    old_sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt

    mapped = _map_state_dict(old_sd, new_sd)

    updated_sd = dict(new_sd)
    for k, v in mapped.items():
        updated_sd[k] = v

    missing_before = set(new_sd.keys()) - set(mapped.keys())
    load_msg = model.load_state_dict(updated_sd, strict=False)
    missing_after = load_msg.missing_keys
    unexpected_after = load_msg.unexpected_keys

    print(f"Loaded with strict=False. Missing after load: {len(missing_after)}; unexpected: {len(unexpected_after)}")
    if missing_after:
        print("Missing keys (first 10):", missing_after[:10])
    if unexpected_after:
        print("Unexpected keys:", unexpected_after)

    output = {
        "model": model.state_dict(),
        "config": cfg,
    }
    torch.save(output, output_path)
    print(f"Converted checkpoint written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert neural_tracing UNet checkpoints to NetworkFromConfig format.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to old checkpoint (.pth)")
    parser.add_argument("--config", required=True, type=Path, help="Path to neural_tracing config.json used for training")
    parser.add_argument("--output", required=True, type=Path, help="Where to save the converted checkpoint")
    parser.add_argument("--spacing", type=str, help="Override spacing as comma-separated floats, e.g. '8,8,8'")
    args = parser.parse_args()

    spacing_override = _parse_spacing(args.spacing) if args.spacing else None
    convert(args.checkpoint, args.config, args.output, spacing_override)


if __name__ == "__main__":
    main()
