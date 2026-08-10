"""Device, AMP, DataParallel, and compilation policy shared by inference commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn


LOGGER = logging.getLogger(__name__)


def parse_gpu_ids(value: str | None) -> tuple[int, ...]:
    """Parse a unique comma-separated CUDA device list."""

    if value is None or not str(value).strip():
        return ()
    parsed: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            raise ValueError("--gpus must be a comma-separated list such as 0,1")
        try:
            device_id = int(item)
        except ValueError as exc:
            raise ValueError(f"--gpus entries must be integers, got {item!r}") from exc
        if device_id < 0:
            raise ValueError(
                f"--gpus entries must be nonnegative, got {device_id}"
            )
        if device_id in parsed:
            raise ValueError(f"--gpus contains duplicate device id {device_id}")
        parsed.append(device_id)
    return tuple(parsed)


def checkpoint_amp_dtype(
    checkpoint: Any,
    source: str | Path = "<memory>",
) -> torch.dtype | None:
    """Read the optional training mixed-precision dtype from checkpoint config."""

    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("config"), Mapping
    ):
        return None
    value = checkpoint["config"].get("mixed_precision")
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"no", "none", "false", "off", "disabled"}:
        return None
    LOGGER.warning(
        "Checkpoint %s has unknown mixed_precision=%r; using default autocast dtype",
        source,
        value,
    )
    return None


def resolve_amp_dtype(
    requested: str,
    checkpoint: Any,
    source: str | Path = "<memory>",
) -> torch.dtype | None:
    """Resolve a CLI AMP request, with auto delegated to checkpoint config."""

    normalized = str(requested).strip().lower()
    if normalized == "default":
        return None
    if normalized == "fp16":
        return torch.float16
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "auto":
        return checkpoint_amp_dtype(checkpoint, source)
    raise ValueError(f"Unsupported --amp-dtype value {requested!r}")


def maybe_compile_model(
    model: nn.Module,
    *,
    enabled: bool,
    mode: str,
) -> tuple[nn.Module, bool]:
    """Return the model and whether compilation completed successfully."""

    if not enabled:
        return model, False
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        LOGGER.warning("torch.compile is unavailable; continuing eagerly")
        return model, False
    try:
        return (
            compile_fn(model, mode=str(mode), fullgraph=False, dynamic=False),
            True,
        )
    except Exception as exc:
        LOGGER.warning("torch.compile failed (%s); continuing eagerly", exc)
        return model, False


def prepare_model_for_inference(
    model: nn.Module,
    *,
    gpu_ids: Sequence[int],
    compile_model: bool,
    compile_mode: str,
) -> tuple[nn.Module, torch.device, bool]:
    """Move, optionally wrap, and optionally compile an inference model."""

    requested = tuple(int(device_id) for device_id in gpu_ids)
    compile_enabled = bool(compile_model)
    if requested:
        if not torch.cuda.is_available():
            raise ValueError("--gpus was provided, but CUDA is unavailable")
        visible = int(torch.cuda.device_count())
        invalid = [device_id for device_id in requested if device_id >= visible]
        if invalid:
            raise ValueError(
                f"Requested CUDA device ids {invalid!r} are unavailable; "
                f"visible device count is {visible}"
            )
        device = torch.device(f"cuda:{requested[0]}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    if len(requested) > 1:
        compile_enabled = False
        model = nn.DataParallel(
            model,
            device_ids=list(requested),
            output_device=requested[0],
        )
    model, compile_enabled = maybe_compile_model(
        model,
        enabled=compile_enabled,
        mode=compile_mode,
    )
    return model, device, compile_enabled
