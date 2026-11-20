"""Optional inkrs engine for flat ink inference.

The default engine is the PyTorch path in ``infer.py``. With ``--engine inkrs``
villa still selects the layers, schedules the tiles and writes the TIFF, and
inkrs normalizes, runs the model and blends on the GPU. When inkrs cannot run
a command or an input, villa logs why and runs it with PyTorch, unless
``--engine-fallback error`` is given.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vesuvius.ink_detection.models.checkpoint import resolve_checkpoint_path

LOGGER = logging.getLogger(__name__)

ENGINE_CHOICES = ("torch", "inkrs")
FALLBACK_CHOICES = ("torch", "error")
# inkrs's macOS build runs on Metal, which villa reaches as the MPS device
INKRS_DEVICE_TYPES = ("cuda", "mps")


class EngineUnavailable(RuntimeError):
    """inkrs cannot run this command or this input."""


def add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the opt-in engine selection flags to the flat inference CLI."""

    parser.add_argument(
        "--engine",
        choices=ENGINE_CHOICES,
        default="torch",
        help=(
            "Inference engine. 'torch' (default) is the PyTorch path. 'inkrs' "
            "uses the optional inkrs package (the inkrs extra) for supported "
            "models on supported NVIDIA GPUs or Apple silicon, and writes the "
            "same maps to within 4/255."
        ),
    )
    parser.add_argument(
        "--engine-fallback",
        choices=FALLBACK_CHOICES,
        default="torch",
        help=(
            "What to do when --engine cannot run a command or an input: "
            "continue with PyTorch and log why (default), or stop with an error."
        ),
    )


def fall_back(reason: str, *, fallback: str) -> None:
    """Log that PyTorch runs instead, or raise under ``--engine-fallback error``."""

    if fallback == "error":
        raise EngineUnavailable(reason)
    LOGGER.warning("%s. Running it with PyTorch.", reason)


def command_reasons(
    args: argparse.Namespace,
    configured_model: Any,
    device: torch.device,
    stride: int,
    blend_mode: str,
) -> list[str]:
    """Why inkrs cannot run this command, from villa options inkrs never sees."""

    reasons = []
    # inkrs exports a .pth itself and reads any other path as its own export
    checkpoint = Path(args.checkpoint)
    if checkpoint.suffix != ".pth":
        reasons.append(f"it loads .pth checkpoints only, got {checkpoint.name}")
    if device.type not in INKRS_DEVICE_TYPES:
        reasons.append(f"it needs a CUDA device, or MPS on macOS, got {device.type!r}")
    if len(args.gpu_ids) > 1:
        reasons.append("it runs on one GPU; drop --gpus or pass one id")
    if args.ensemble_checkpoint:
        reasons.append("it does not run --ensemble-checkpoint")
    if args.tta_mirror:
        reasons.append("it does not run --tta-mirror")
    if args.roi is not None:
        reasons.append("it does not crop to --roi")
    if args.mask_path is not None:
        reasons.append("it does not apply --mask-path")
    # inkrs runs the precision the checkpoint stores and ignores --amp-dtype,
    # so both must be fp16 (no dtype means autocast's CUDA default, fp16)
    if configured_model.stored_amp_dtype != torch.float16:
        reasons.append(f"it needs a checkpoint that stores mixed_precision fp16, which {checkpoint.name} does not")
    elif (configured_model.amp_dtype or torch.float16) != torch.float16:
        reasons.append(f"it reproduces fp16 autocast only, got {configured_model.amp_dtype}")
    if blend_mode != "hann":
        reasons.append(f"it blends with Hann only, got {blend_mode!r}")
    if not 1 <= stride <= configured_model.patch_size:
        reasons.append(f"it needs a stride from 1 to the patch size {configured_model.patch_size}, got {stride}")
    return reasons


class InkrsEngine:
    """Run flat inference through the optional ``inkrs`` package.

    ``inkrs`` is imported only when this engine is selected, so villa neither
    depends on it nor imports it by default.
    """

    def __init__(self, module: Any, handle: Any) -> None:
        self._module = module
        self._handle = handle
        # the last source-layer span read: the reverse run of an input selects
        # the forward run's layers in the other order, so it reuses this read
        self._span: tuple[tuple, np.ndarray] | None = None

    @classmethod
    def load(cls, checkpoint: str | Path, device: torch.device) -> "InkrsEngine":
        """Import inkrs and load the checkpoint into it on ``device``."""

        try:
            module = importlib.import_module("inkrs")
        except ImportError as exc:
            raise EngineUnavailable(
                f"the inkrs package is not installed ({exc}); install villa's "
                "inkrs extra (`uv sync --extra models --extra inkrs`)"
            ) from exc

        # an index-less CUDA device is PyTorch's current one, which a fallback also uses
        index = device.index if device.index is not None else (
            torch.cuda.current_device() if device.type == "cuda" else 0
        )
        # resolve the path as the PyTorch loader does, so a quoted ~ is expanded
        path = resolve_checkpoint_path(checkpoint)
        try:
            handle = module.Engine(path, device=int(index))
        except module.UnsupportedError as exc:
            raise EngineUnavailable(f"inkrs cannot load {path.name} on device {index}: {exc}") from exc

        LOGGER.info("Using the inkrs engine %s on device %d", getattr(module, "__version__", "?"), index)
        return cls(module, handle)

    def run(
        self,
        volume: Any,
        depth_axis_first: bool,
        layer_indices: np.ndarray,
        occupancy: np.ndarray | None,
        stride: int,
        span_key: tuple,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the uint8 map and its coverage for the selected layers.

        Raises EngineUnavailable when inkrs declines the input, before any GPU
        work.
        """

        started = time.perf_counter()
        indices = np.asarray(layer_indices, dtype=np.int64)
        low, high = int(indices.min()), int(indices.max()) + 1
        key = (*span_key, depth_axis_first, low, high)
        if self._span is None or self._span[0] != key:
            # drop the kept span first, so two are never held at once
            self._span = None
            self._span = (key, _read_span(volume, depth_axis_first, low, high))
        span = self._span[1]
        window = span if np.array_equal(indices, np.arange(low, high)) else np.ascontiguousarray(span[indices - low])
        read_s = time.perf_counter() - started

        # the window is in model order, so it runs forward from layer 0; villa's
        # occupancy mask keeps villa's tile schedule
        try:
            [(probability, coverage)] = self._handle.run(
                window, [(0, 1)], stride=stride, occupancy=occupancy, coverage=True
            )
        except self._module.UnsupportedError as exc:
            raise EngineUnavailable(f"inkrs does not support this input: {exc}") from exc

        LOGGER.info("inkrs ran in %.2f s (window read %.2f s)", time.perf_counter() - started, read_s)
        return np.asarray(probability, dtype=np.uint8), np.asarray(coverage, dtype=bool)


def _read_span(volume: Any, depth_axis_first: bool, low: int, high: int) -> np.ndarray:
    # layers low:high as a C-contiguous (Z, H, W) array
    if depth_axis_first:
        return np.ascontiguousarray(volume[low:high])
    return np.ascontiguousarray(np.moveaxis(np.asarray(volume[:, :, low:high]), -1, 0))


def load_engine(
    args: argparse.Namespace,
    configured_model: Any,
    device: torch.device,
    stride: int,
    blend_mode: str,
) -> InkrsEngine | None:
    """Return the inkrs engine for ``--engine inkrs``, or None when PyTorch runs instead."""

    try:
        reasons = command_reasons(args, configured_model, device, stride, blend_mode)
        if reasons:
            raise EngineUnavailable("; ".join(reasons))
        return InkrsEngine.load(args.checkpoint, device)
    except EngineUnavailable as exc:
        fall_back(f"The inkrs engine cannot run this command: {exc}", fallback=args.engine_fallback)
        return None
