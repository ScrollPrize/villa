#!/usr/bin/env python3
"""Add flow refinement heads to a rowcol_cond dense displacement checkpoint.

This is a one-off conversion utility for finetuning a known-good
displacement-only checkpoint with the auxiliary flow_dir/flow_dist heads.
Existing compatible weights are copied exactly; new head weights are left at
the model's deterministic random initialization.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from vesuvius.neural_tracing.nets.models import make_model, resolve_checkpoint_path, strip_state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a rowcol_cond dense displacement checkpoint into a checkpoint "
            "with additional flow_dir and flow_dist heads."
        )
    )
    parser.add_argument("input_checkpoint", type=str, help="Input .pth checkpoint or checkpoint directory.")
    parser.add_argument("output_checkpoint", type=str, help="Output .pth checkpoint path.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used only for deterministic initialization of newly added heads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output checkpoint if it already exists.",
    )
    parser.add_argument(
        "--keep-training-state",
        action="store_true",
        help=(
            "Keep optimizer/lr_scheduler state from the source checkpoint. By default these "
            "are removed because they do not contain state for the new flow heads."
        ),
    )
    return parser.parse_args()


def _flow_config_from_source(source_config: dict) -> dict:
    config = copy.deepcopy(source_config)
    if bool(config.get("use_triplet_wrap_displacement", False)):
        raise ValueError(
            "Flow-refinement conversion currently supports regular single-wrap dense checkpoints only, "
            "not triplet-wrap checkpoints."
        )

    config["use_dense_displacement"] = True
    config["use_flow_refinement_targets"] = True
    config.setdefault("lambda_flow_dir", 0.1)
    config.setdefault("lambda_flow_dist", 0.1)
    config.setdefault("flow_dist_huber_beta", config.get("displacement_huber_beta", 5.0))

    targets = copy.deepcopy(config.get("targets") or {})
    displacement_channels = int(targets.get("displacement", {}).get("out_channels", 3))
    if displacement_channels != 3:
        raise ValueError(
            "Flow-refinement conversion expected a 3-channel single-wrap displacement head; "
            f"got displacement out_channels={displacement_channels}."
        )

    targets["displacement"] = {
        **targets.get("displacement", {}),
        "out_channels": displacement_channels,
        "activation": targets.get("displacement", {}).get("activation", "none"),
    }
    targets["flow_dir"] = {"out_channels": displacement_channels, "activation": "none"}
    targets["flow_dist"] = {"out_channels": 1, "activation": "none"}
    config["targets"] = targets
    return config


def _copy_compatible_weights(new_model: torch.nn.Module, source_state: dict[str, torch.Tensor]) -> tuple[dict, list[str], list[str]]:
    clean_source = strip_state(source_state)
    new_state = new_model.state_dict()
    copied = []
    skipped = []

    for key, value in clean_source.items():
        target_value = new_state.get(key)
        if target_value is None:
            skipped.append(key)
            continue
        if tuple(target_value.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        new_state[key] = value.detach().to(dtype=target_value.dtype, device=target_value.device)
        copied.append(key)

    new_model.load_state_dict(new_state, strict=True)
    missing_from_source = [key for key in new_state.keys() if key not in copied]
    return new_model.state_dict(), copied, missing_from_source + skipped


def main() -> None:
    args = _parse_args()
    input_path = resolve_checkpoint_path(args.input_checkpoint)
    output_path = Path(args.output_checkpoint)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output checkpoint already exists: {output_path} (use --overwrite)")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint or "model" not in checkpoint:
        raise RuntimeError(f"Checkpoint must contain 'config' and 'model': {input_path}")

    new_config = _flow_config_from_source(checkpoint["config"])
    torch.manual_seed(int(args.seed))
    new_model = make_model(new_config)
    new_state, copied, not_copied = _copy_compatible_weights(new_model, checkpoint["model"])

    converted = copy.deepcopy(checkpoint)
    converted["config"] = new_config
    converted["model"] = new_state
    converted["source_checkpoint"] = str(input_path)
    converted["flow_refinement_conversion"] = {
        "seed": int(args.seed),
        "copied_tensor_count": len(copied),
        "not_copied_tensor_count": len(not_copied),
        "new_heads": ["flow_dir", "flow_dist"],
    }
    if not args.keep_training_state:
        converted.pop("optimizer", None)
        converted.pop("lr_scheduler", None)
        converted["step"] = 0

    torch.save(converted, output_path)
    print(f"Saved converted checkpoint: {output_path}")
    print(f"Copied tensors: {len(copied)}")
    print(f"Initialized/new or skipped tensors: {len(not_copied)}")
    print("For finetuning, set load_weights_only=true when loading this checkpoint.")


if __name__ == "__main__":
    main()
