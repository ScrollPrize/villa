"""Loading and restoration of ink training checkpoint payloads."""

from __future__ import annotations

from copy import deepcopy
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
    """Select one supported inference state container by compatibility priority."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint {str(source)!r} must contain a mapping")
    candidates = (
        ("ema_model", "state_dict", "model_state_dict", "model")
        if prefer_ema
        else ("state_dict", "model_state_dict", "model")
    )
    for name in candidates:
        model_state = checkpoint.get(name)
        if isinstance(model_state, Mapping):
            return name, model_state
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return "<root>", checkpoint
    raise ValueError(
        f"Checkpoint {str(source)!r} has no supported model state; expected "
        + ", ".join(candidates)
    )


def _backbone_is_checkpoint(value: Any) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    text = str(value).strip()
    if not text:
        return False
    return (
        text.startswith(("~", ".", "/"))
        or "/" in text
        or "\\" in text
        or Path(text).suffix.lower() in {".pt", ".pth"}
    )


def _checkpoint_reference_path(
    reference: str | Path,
    containing_checkpoint: str | Path,
) -> Path:
    path = Path(os.path.expanduser(str(reference)))
    if path.is_absolute():
        return path.resolve(strict=False)
    containing_path = Path(
        os.path.abspath(os.path.expanduser(str(containing_checkpoint)))
    )
    containing_directory = (
        containing_path.parent if containing_path.suffix else containing_path
    )
    candidates = (containing_directory / path, path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve(strict=False)


def resolve_pretrained_backbone_config(
    authored: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    seen_paths: set[Path] | None = None,
) -> dict[str, Any]:
    """Resolve checkpoint-backed backbone names without mutating saved config."""

    resolved = deepcopy(dict(authored))
    config = InkConfig.from_mapping(resolved)
    backbone = config.model.pretrained_backbone
    if not _backbone_is_checkpoint(backbone):
        return resolved

    backbone_path = _checkpoint_reference_path(backbone, checkpoint_path)
    visited = set() if seen_paths is None else set(seen_paths)
    containing_path = Path(os.path.abspath(str(checkpoint_path))).resolve(
        strict=False
    )
    visited.add(containing_path)
    if backbone_path in visited:
        chain = " -> ".join(str(path) for path in (*visited, backbone_path))
        raise ValueError(
            f"Detected recursive pretrained_backbone checkpoint chain: {chain}"
        )
    payload = load_checkpoint(backbone_path)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("config"), Mapping
    ):
        raise ValueError(
            "Checkpoint-backed pretrained_backbone requires a checkpoint "
            f"config mapping: {backbone_path}"
        )
    nested = resolve_pretrained_backbone_config(
        payload["config"],
        checkpoint_path=backbone_path,
        seen_paths=visited | {backbone_path},
    )
    concrete = InkConfig.from_mapping(nested).model.pretrained_backbone
    if not concrete or _backbone_is_checkpoint(concrete):
        raise ValueError(
            "Checkpoint-backed pretrained_backbone did not resolve to a "
            f"concrete backbone name: {backbone_path}"
        )
    updated_model = config.model.model_settings_mapping()
    updated_model["pretrained_backbone"] = concrete
    resolved["model_config"] = updated_model
    return resolved


def config_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    source: str | Path = "<memory>",
) -> InkConfig:
    """Reconstruct the typed configuration when a checkpoint stores one."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint {str(source)!r} must contain a mapping")
    if "config" not in checkpoint:
        raise KeyError(
            f"Checkpoint {str(source)!r} is missing required field 'config'"
        )
    return InkConfig.from_mapping(checkpoint["config"])
