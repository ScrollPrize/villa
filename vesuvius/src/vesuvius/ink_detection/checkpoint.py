"""Loading and restoration of ink training checkpoint payloads."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from vesuvius.ink_detection.config import InkConfig


def resolve_checkpoint_path(
    checkpoint_path: str | Path | None,
    config_path: str | Path | None = None,
) -> Path | None:
    """Resolve a configured checkpoint relative to its JSON file."""

    if checkpoint_path in (None, ""):
        return None
    expanded = Path(os.path.expanduser(str(checkpoint_path)))
    if config_path is not None and not expanded.is_absolute():
        config_directory = Path(
            os.path.dirname(os.path.abspath(str(config_path)))
        )
        expanded = config_directory / expanded
    return expanded


def load_checkpoint(
    checkpoint_path: str | Path,
) -> Any:
    """Load a checkpoint on CPU without interpreting its payload."""

    resolved = resolve_checkpoint_path(checkpoint_path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    try:
        return torch.load(str(resolved), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(resolved), map_location="cpu")


def load_checkpoint_from_config(
    config: InkConfig,
    config_path: str | Path,
) -> tuple[Path | None, Any | None, bool]:
    """Resolve and optionally load the checkpoint selected by an InkConfig."""

    checkpoint_path = resolve_checkpoint_path(config.checkpoint.path, config_path)
    payload = None
    if checkpoint_path is not None:
        payload = load_checkpoint(checkpoint_path)
    return checkpoint_path, payload, config.checkpoint.weights_only


def load_model_state(
    model: nn.Module,
    model_state: Mapping[str, Any],
) -> None:
    """Load state after trying direct, DDP-stripped, and DDP-prefixed keys."""

    try:
        model.load_state_dict(model_state)
        return
    except RuntimeError as direct_error:
        stripped_state = {
            (
                str(key)[len("module.") :]
                if str(key).startswith("module.")
                else str(key)
            ): value
            for key, value in model_state.items()
        }
        if any(str(key).startswith("module.") for key in model_state):
            stripped_error = None
            try:
                model.load_state_dict(stripped_state)
            except RuntimeError as error:
                stripped_error = error
            if stripped_error is None:
                return

        prefixed_state = {
            (
                str(key)
                if str(key).startswith("module.")
                else f"module.{key}"
            ): value
            for key, value in model_state.items()
        }
        if any(not str(key).startswith("module.") for key in model_state):
            model.load_state_dict(prefixed_state)
            return
        raise direct_error


def restore_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    checkpoint: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    load_weights_only: bool = False,
    ema_model: nn.Module | None = None,
) -> tuple[int, int]:
    """Restore training state and return the next step and EMA optimizer step."""

    model_state = checkpoint.get("model")
    if model_state is None:
        raise ValueError(f"Checkpoint {str(checkpoint_path)!r} is missing 'model'")
    load_model_state(model, model_state)
    if load_weights_only:
        return 0, 0

    if "optimizer" not in checkpoint:
        raise KeyError(f"Checkpoint missing optimizer state: {checkpoint_path}")
    if "lr_scheduler" not in checkpoint:
        raise KeyError(f"Checkpoint missing lr_scheduler state: {checkpoint_path}")
    if "step" not in checkpoint:
        raise KeyError(f"Checkpoint missing step: {checkpoint_path}")

    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    optimizer_step = 0
    if ema_model is not None:
        ema_model_state = checkpoint.get("ema_model")
        if ema_model_state is not None:
            ema_model.load_state_dict(ema_model_state)
            optimizer_step = int(checkpoint.get("ema_optimizer_step", 0))
    return int(checkpoint["step"]) + 1, optimizer_step


def select_inference_weights(
    checkpoint: Mapping[str, Any],
    *,
    prefer_ema: bool = True,
    source: str | Path = "<memory>",
) -> tuple[str, Mapping[str, Any]]:
    """Select EMA weights when requested and available, otherwise model weights."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint {str(source)!r} must contain a mapping")
    if prefer_ema and isinstance(checkpoint.get("ema_model"), Mapping):
        return "ema_model", checkpoint["ema_model"]
    model_state = checkpoint.get("model")
    if not isinstance(model_state, Mapping):
        raise ValueError(f"Checkpoint {str(source)!r} is missing model weights")
    return "model", model_state


def config_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    source: str | Path = "<memory>",
) -> InkConfig:
    """Reconstruct the typed configuration when a checkpoint stores one."""

    if "config" not in checkpoint:
        raise KeyError(
            f"Checkpoint {str(source)!r} is missing required field 'config'"
        )
    return InkConfig.from_mapping(checkpoint["config"])
