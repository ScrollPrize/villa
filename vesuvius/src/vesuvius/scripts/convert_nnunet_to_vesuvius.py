#!/usr/bin/env python3
"""
Convert an nnUNet checkpoint to Vesuvius train.py checkpoint format.

This script takes an nnUNet model trained with nnUNetTrainer (or variants like
nnUNetTrainerMedialSurfaceRecall) and converts it to a format that can be loaded
by the Vesuvius training framework (vesuvius/models/training/train.py).

Key differences between formats:
- nnUNet uses 'network_weights' key; Vesuvius uses 'model' key
- nnUNet encoder keys: encoder.*; Vesuvius: shared_encoder.*
- nnUNet decoder keys: decoder.*; Vesuvius: task_decoders.{task_name}.*

Usage:
    python convert_nnunet_to_vesuvius.py \
        --nnunet-dir /path/to/nnunet/model/dir \
        --output /path/to/output.pth \
        --task-name segmentation \
        --fold 0 \
        --checkpoint best
"""

import argparse
import json
import torch
from pathlib import Path


def load_nnunet_checkpoint(nnunet_dir: str, fold: int = 0, checkpoint: str = "best") -> dict:
    """Load nnUNet checkpoint and associated JSON files."""
    nnunet_dir = Path(nnunet_dir)

    # Load plans.json
    plans_path = nnunet_dir / "plans.json"
    if not plans_path.exists():
        raise FileNotFoundError(f"plans.json not found at {plans_path}")
    with open(plans_path, "r") as f:
        plans = json.load(f)

    # Load dataset.json
    dataset_path = nnunet_dir / "dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset.json not found at {dataset_path}")
    with open(dataset_path, "r") as f:
        dataset_json = json.load(f)

    # Load checkpoint
    fold_dir = nnunet_dir / f"fold_{fold}"
    checkpoint_name = f"checkpoint_{checkpoint}.pth"
    checkpoint_path = fold_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    return {
        "checkpoint": ckpt,
        "plans": plans,
        "dataset_json": dataset_json,
        "checkpoint_path": str(checkpoint_path)
    }


def convert_state_dict(nnunet_weights: dict, task_name: str = "segmentation") -> dict:
    """
    Convert nnUNet state dict keys to Vesuvius format.

    Mapping:
    - encoder.* -> shared_encoder.*
    - decoder.encoder.* -> task_decoders.{task_name}.encoder.*
    - decoder.stages.* -> task_decoders.{task_name}.stages.*
    - decoder.transpconvs.* -> task_decoders.{task_name}.transpconvs.*
    - decoder.seg_layers.* -> task_decoders.{task_name}.seg_layers.*
    """
    vesuvius_weights = {}

    for key, value in nnunet_weights.items():
        if key.startswith("encoder."):
            # Map encoder.* -> shared_encoder.*
            new_key = "shared_encoder." + key[len("encoder."):]
            vesuvius_weights[new_key] = value
        elif key.startswith("decoder."):
            # Map decoder.* -> task_decoders.{task_name}.*
            decoder_suffix = key[len("decoder."):]
            new_key = f"task_decoders.{task_name}." + decoder_suffix
            vesuvius_weights[new_key] = value
        else:
            # Keep other keys as-is (shouldn't happen in standard nnUNet)
            print(f"Warning: Unknown key pattern '{key}', keeping as-is")
            vesuvius_weights[key] = value

    return vesuvius_weights


def build_model_config(plans: dict, dataset_json: dict, task_name: str = "segmentation") -> dict:
    """Build Vesuvius model_config from nnUNet plans."""

    # Use 3d_fullres configuration
    config_name = "3d_fullres"
    if config_name not in plans["configurations"]:
        # Fallback to first available 3D config
        for k in plans["configurations"].keys():
            if "3d" in k:
                config_name = k
                break

    config = plans["configurations"][config_name]
    arch = config["architecture"]
    arch_kwargs = arch["arch_kwargs"]

    # Determine output channels from labels
    labels = dataset_json.get("labels", {})
    # Count non-background labels (label 0 is usually background)
    num_classes = len([k for k in labels.keys() if k != "background" and labels[k] != 0])
    if num_classes == 0:
        num_classes = len(labels)
    # For segmentation, output channels = num_classes (including background typically for softmax)
    out_channels = len(labels)
    if "ignore" in labels:
        out_channels -= 1  # Don't count ignore label

    # Get intensity properties for normalization
    intensity_props = plans.get("foreground_intensity_properties_per_channel", {})

    model_config = {
        "model_name": f"nnunet_{plans.get('dataset_name', 'converted')}",
        "basic_encoder_block": "BasicBlockD",
        "basic_decoder_block": "ConvBlock",
        "bottleneck_block": "BasicBlockD",
        "features_per_stage": arch_kwargs["features_per_stage"],
        "num_stages": arch_kwargs["n_stages"],
        "n_stages": arch_kwargs["n_stages"],
        "n_blocks_per_stage": arch_kwargs["n_blocks_per_stage"],
        "n_conv_per_stage_decoder": arch_kwargs["n_conv_per_stage_decoder"],
        "kernel_sizes": arch_kwargs["kernel_sizes"],
        "strides": arch_kwargs["strides"],
        "conv_op": "Conv3d",
        "conv_bias": arch_kwargs.get("conv_bias", True),
        "norm_op": "InstanceNorm3d",
        "norm_op_kwargs": arch_kwargs.get("norm_op_kwargs", {"affine": True, "eps": 1e-5}),
        "dropout_op": arch_kwargs.get("dropout_op"),
        "dropout_op_kwargs": arch_kwargs.get("dropout_op_kwargs"),
        "nonlin": "LeakyReLU",
        "nonlin_kwargs": arch_kwargs.get("nonlin_kwargs", {"inplace": True}),
        "return_skips": True,
        "do_stem": True,
        "stem_channels": arch_kwargs["features_per_stage"][0],
        "bottleneck_channels": None,
        "stochastic_depth_p": 0.0,
        "squeeze_excitation": False,
        "squeeze_excitation_reduction_ratio": 0.0625,
        "pool_type": "conv",
        "op_dims": 3,
        "patch_size": tuple(config["patch_size"]),
        "batch_size": config["batch_size"],
        "in_channels": len(dataset_json.get("channel_names", {"0": "default"})),
        "autoconfigure": False,
        "targets": {
            task_name: {
                "out_channels": out_channels,
                "activation": "softmax"  # nnUNet uses softmax for multi-class
            }
        },
        "separate_decoders": True,
        "num_pool_per_axis": None,
        "must_be_divisible_by": None
    }

    return model_config, intensity_props


def convert_nnunet_to_vesuvius(
    nnunet_dir: str,
    output_path: str,
    task_name: str = "segmentation",
    fold: int = 0,
    checkpoint: str = "best"
) -> str:
    """
    Convert nnUNet checkpoint to Vesuvius format.

    Args:
        nnunet_dir: Path to nnUNet model directory (contains plans.json, dataset.json, fold_X/)
        output_path: Path to save the converted checkpoint
        task_name: Name for the segmentation task in Vesuvius format
        fold: Which fold to load (default: 0)
        checkpoint: Which checkpoint to load ('best', 'final', or 'latest')

    Returns:
        Path to the saved checkpoint
    """
    # Load nnUNet checkpoint and metadata
    data = load_nnunet_checkpoint(nnunet_dir, fold, checkpoint)
    ckpt = data["checkpoint"]
    plans = data["plans"]
    dataset_json = data["dataset_json"]

    print(f"nnUNet trainer: {ckpt.get('trainer_name', 'unknown')}")
    print(f"nnUNet epoch: {ckpt.get('current_epoch', 'unknown')}")

    # Convert state dict
    print("Converting state dict keys...")
    nnunet_weights = ckpt["network_weights"]
    vesuvius_weights = convert_state_dict(nnunet_weights, task_name)

    print(f"  Original keys: {len(nnunet_weights)}")
    print(f"  Converted keys: {len(vesuvius_weights)}")

    # Build model config
    print("Building model config...")
    model_config, intensity_props = build_model_config(plans, dataset_json, task_name)

    # Get normalization scheme
    norm_schemes = plans["configurations"].get("3d_fullres", {}).get("normalization_schemes", ["ZScoreNormalization"])
    normalization_scheme = norm_schemes[0] if norm_schemes else "ZScoreNormalization"

    # Format intensity properties for Vesuvius
    formatted_intensity_props = {}
    if intensity_props and "0" in intensity_props:
        props = intensity_props["0"]
        formatted_intensity_props = {
            "mean": props.get("mean", 0.0),
            "std": props.get("std", 1.0),
            "percentile_00_5": props.get("percentile_00_5", 0.0),
            "percentile_99_5": props.get("percentile_99_5", 255.0),
            "min": props.get("min", 0.0),
            "max": props.get("max", 255.0)
        }

    # Build Vesuvius checkpoint
    vesuvius_checkpoint = {
        "model": vesuvius_weights,
        "optimizer": None,  # Will be reinitialized on load
        "scheduler": None,  # Will be reinitialized on load
        "epoch": ckpt.get("current_epoch", 0),
        "model_config": model_config,
        "normalization_scheme": normalization_scheme,
        "intensity_properties": formatted_intensity_props,
        # Store original nnUNet metadata for reference
        "_nnunet_metadata": {
            "trainer_name": ckpt.get("trainer_name"),
            "original_epoch": ckpt.get("current_epoch"),
            "plans_name": plans.get("plans_name"),
            "dataset_name": plans.get("dataset_name"),
            "source_checkpoint": data["checkpoint_path"]
        }
    }

    # Save checkpoint
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vesuvius_checkpoint, output_path)

    print(f"\nCheckpoint saved to: {output_path}")
    print(f"\nModel configuration:")
    print(f"  Patch size: {model_config['patch_size']}")
    print(f"  Features per stage: {model_config['features_per_stage']}")
    print(f"  Num stages: {model_config['num_stages']}")
    print(f"  Input channels: {model_config['in_channels']}")
    print(f"  Output channels: {model_config['targets'][task_name]['out_channels']}")
    print(f"  Normalization: {normalization_scheme}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert nnUNet checkpoint to Vesuvius train.py format"
    )
    parser.add_argument(
        "--nnunet-dir",
        type=str,
        required=True,
        help="Path to nnUNet model directory (contains plans.json, dataset.json, fold_X/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the converted checkpoint"
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="segmentation",
        help="Name for the segmentation task (default: segmentation)"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Which fold to load (default: 0)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        choices=["best", "final", "latest"],
        help="Which checkpoint to load (default: best)"
    )

    args = parser.parse_args()

    convert_nnunet_to_vesuvius(
        nnunet_dir=args.nnunet_dir,
        output_path=args.output,
        task_name=args.task_name,
        fold=args.fold,
        checkpoint=args.checkpoint
    )


if __name__ == "__main__":
    main()
