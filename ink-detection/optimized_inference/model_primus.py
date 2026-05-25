"""Primus (LeJEPA fine-tune) loader for villa optimized_inference.

Adds support for `MODEL_TYPE=primus` to villa/ink-detection/optimized_inference,
allowing the container to run a Primus-encoder fine-tuned ink-detection model
that was produced via `TrainFineTuneLEJEPA`. Mirrors the structure of
`model_timesformer.py` / `model_resnet3d.py`.

The Primus architecture is config-driven (a NetworkFromConfig hybrid encoder
plus per-task heads), so unlike the TimeSformer / ResNet3D loaders we do not
hard-code a class here — we reconstruct the model from the saved model_config
in the checkpoint envelope. The checkpoint envelope is expected to be the one
produced by `scripts/export_for_production.py` with `architecture=primus_lejepa`,
which embeds `model_state_dict`, `config` (containing `model_config` or a
flat patch_size + targets pair), and `metadata`.

Container dependencies: this loader imports from the `vesuvius` Python package
(`vesuvius.models.build.build_network_from_config`). The optimized inference
image must install the local `villa/vesuvius` package for `MODEL_TYPE=primus`
to be runnable.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _strip_state_dict_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop DDP / torch.compile wrapper prefixes that may appear in checkpoints."""
    prefixes = ("module.", "_orig_mod.")

    def strip_key(k: str) -> str:
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        return k

    return {strip_key(k): v for k, v in state_dict.items()}


def _normalize_model_config(checkpoint_data: Dict[str, Any], num_frames: int) -> Dict[str, Any]:
    """Recover a NetworkFromConfig-compatible model_config from the envelope.

    Accepts both the optimized_inference exported envelope (`config.model_config`
    or flattened `config`) and the train_py envelope (`model_config` top-level).
    """
    if isinstance(checkpoint_data.get("model_config"), dict):
        model_config = dict(checkpoint_data["model_config"])
    else:
        config = checkpoint_data.get("config") or {}
        if not isinstance(config, dict):
            raise ValueError("checkpoint has no usable config / model_config")
        model_config = dict(config.get("model_config") or {})
        if not model_config:
            # Flattened envelope: lift the model-relevant fields.
            for key in (
                "patch_size",
                "train_patch_size",
                "patch_embed_size",
                "in_channels",
                "targets",
                "enable_deep_supervision",
            ):
                if key in config:
                    model_config[key] = config[key]
        else:
            for key in ("patch_size", "train_patch_size", "in_channels", "enable_deep_supervision"):
                if key in config and key not in model_config:
                    model_config[key] = config[key]
    model_config.setdefault("in_channels", num_frames)
    model_config.setdefault("enable_deep_supervision", False)
    if "patch_size" in model_config and "train_patch_size" not in model_config:
        model_config["train_patch_size"] = model_config["patch_size"]
    if "targets" not in model_config:
        config = checkpoint_data.get("config") or {}
        if isinstance(config, dict):
            targets = config.get("targets")
            dataset_config = config.get("dataset_config")
            if targets is None and isinstance(dataset_config, dict):
                targets = dataset_config.get("targets")
            if isinstance(targets, dict):
                model_config["targets"] = targets
    return model_config


class _PrimusConfigShim:
    """Small ConfigManager-compatible surface required by NetworkFromConfig."""

    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.model_name = "optimized_inference_primus"
        self.targets = model_config.get("targets") or {"ink": {"out_channels": 1, "activation": "none"}}
        self.train_patch_size = model_config.get("train_patch_size") or model_config.get("patch_size")
        if self.train_patch_size is None:
            raise ValueError("Primus checkpoint config must include patch_size or train_patch_size")
        self.train_batch_size = int(model_config.get("train_batch_size", 1))
        self.in_channels = int(model_config.get("in_channels", 1))
        self.enable_deep_supervision = bool(model_config.get("enable_deep_supervision", False))
        self.autoconfigure = bool(model_config.get("autoconfigure", False))
        self.op_dims = int(model_config.get("op_dims", len(self.train_patch_size)))
        for key, value in model_config.items():
            setattr(self, key, value)


class PrimusWrapper:
    """Adapts the Primus NetworkFromConfig model to the InferenceModel protocol."""

    def __init__(self, model: nn.Module, device: torch.device, target_key: Optional[str] = None):
        self.model = model
        self.device = device
        self.target_key = target_key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inference container delivers (B, 1, D, H, W), which matches
        # NetworkFromConfig's single-channel 3D volume contract.
        if x.ndim == 4:
            x = x[:, None]
        out = self.model(x)
        if isinstance(out, dict):
            # Multi-task model: pick the configured target or the first available.
            if self.target_key and self.target_key in out:
                out = out[self.target_key]
            else:
                # Prefer 'ink' if present; otherwise first value.
                out = out.get("ink") or next(iter(out.values()))
        return out

    def get_output_scale_factor(self) -> int:
        # Primus 3D UNet predicts at the input spatial resolution.
        return 1

    def eval(self):
        self.model.eval()

    def to(self, device: torch.device):
        self.model.to(device)
        self.device = device


def load_model(model_path: str, device: torch.device, num_frames: int = 26) -> PrimusWrapper:
    """Load a Primus fine-tune checkpoint and return an InferenceModel wrapper.

    Args:
        model_path: Path to the production_model.pt produced by
            scripts/export_for_production.py with architecture=primus_lejepa.
        device: torch device.
        num_frames: Number of input layers (depth axis). Defaults to 26 to match
            the optimized_inference container's existing convention.
    """
    try:
        from vesuvius.models.build.build_network_from_config import NetworkFromConfig
    except ImportError as exc:  # pragma: no cover - exercised at container runtime
        raise ImportError(
            "model_primus requires the vesuvius package. Install villa/vesuvius/src "
            "into this container (or vendor build_network_from_config) before "
            "loading MODEL_TYPE=primus."
        ) from exc

    logger.info(f"Loading Primus model from: {model_path} with {num_frames} frames")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{model_path}: expected a dict checkpoint envelope")

    model_config = _normalize_model_config(checkpoint, num_frames=num_frames)

    shim_mgr = _PrimusConfigShim(model_config)
    try:
        model = NetworkFromConfig(shim_mgr)
    except TypeError:
        # Fallback for variants that take the model_config dict directly.
        model = NetworkFromConfig(model_config)  # type: ignore[arg-type]

    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"{model_path}: no usable state_dict found in checkpoint envelope")
    state_dict = _strip_state_dict_prefixes(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Primus loader: %d missing keys (first 5: %s)", len(missing), missing[:5])
    if unexpected:
        logger.warning("Primus loader: %d unexpected keys (first 5: %s)", len(unexpected), unexpected[:5])

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Primus model wrapped with DataParallel for {torch.cuda.device_count()} GPUs")

    model.to(device)
    model.eval()

    target_key = None
    config = checkpoint.get("config") or {}
    if isinstance(config, dict):
        targets = (config.get("targets") or {})
        if isinstance(targets, dict) and targets:
            target_key = "ink" if "ink" in targets else next(iter(targets.keys()))

    return PrimusWrapper(model, device, target_key=target_key)
