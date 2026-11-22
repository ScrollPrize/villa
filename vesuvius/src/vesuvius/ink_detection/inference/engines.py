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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
        self._span: _Span | None = None

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

    def begin_input(self, input_key: tuple) -> None:
        """Release a span read for another input.

        Only the two directions of one input share a span, so a new input
        releases it before anything else runs, including an input that never
        reaches inkrs.
        """

        if self._span is not None and self._span.input_key != input_key:
            self._span = None

    def run(
        self,
        volume: Any,
        depth_axis_first: bool,
        layer_indices: np.ndarray,
        occupancy: np.ndarray | None,
        stride: int,
        input_key: tuple,
    ) -> np.ndarray:
        """Return the uint8 map for the selected layers.

        Raises EngineUnavailable when inkrs declines the input, before any GPU
        work. inkrs checks the span's dtype and size before it reads it, so
        an input it declines on those is never read.
        """

        self.begin_input(input_key)
        started = time.perf_counter()
        indices = np.asarray(layer_indices, dtype=np.int64)
        low, high = int(indices.min()), int(indices.max()) + 1
        # villa selects a contiguous run of layers, in either order, so inkrs
        # walks the span itself and the reverse run needs no reordered copy
        if np.array_equal(indices, np.arange(low, high)):
            window = (0, 1)
        elif np.array_equal(indices, np.arange(high - 1, low - 1, -1)):
            window = (high - 1 - low, -1)
        else:
            raise EngineUnavailable(f"it runs a contiguous run of layers, got {indices.tolist()}")
        depth_axis = 0 if depth_axis_first else 2
        height, width = (int(n) for i, n in enumerate(volume.shape) if i != depth_axis)
        span = _LazyArray(
            (high - low, height, width),
            volume.dtype,
            lambda: self._read(volume, depth_axis_first, low, high, input_key),
        )

        # villa's occupancy mask keeps villa's tile schedule
        try:
            [probability] = self._handle.run(span, [window], stride=stride, occupancy=occupancy)
        except self._module.UnsupportedError as exc:
            # PyTorch reads this input itself, so the span would only be held twice
            self._span = None
            raise EngineUnavailable(f"inkrs does not support this input: {exc}") from exc

        LOGGER.info("inkrs ran in %.2f s (span read %.2f s)", time.perf_counter() - started, span.read_s)
        return np.asarray(probability, dtype=np.uint8)

    def _read(self, volume: Any, depth_axis_first: bool, low: int, high: int, input_key: tuple) -> np.ndarray:
        # source layers low:high as a C-contiguous (Z, H, W) array, kept for
        # the input's other direction
        span = self._span
        if span is None or (span.low, span.high) != (low, high):
            # drop the kept span first, so two are never held at once
            self._span = None
            span = self._span = _Span(input_key, low, high, _read_span(volume, depth_axis_first, low, high))
        return span.layers


@dataclass(frozen=True)
class _Span:
    """Source layers ``low:high`` of one input, as a (Z, H, W) array."""

    input_key: tuple
    low: int
    high: int
    layers: np.ndarray


class _LazyArray:
    """A (Z, H, W) array read only when inkrs converts it.

    inkrs checks a lazy array's dtype and shape before ``numpy.asarray``
    reads it, so a span it declines on those is never read.
    """

    def __init__(self, shape: tuple[int, int, int], dtype: Any, read: Callable[[], np.ndarray]) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.read_s = 0.0
        self._read = read

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        started = time.perf_counter()
        layers = self._read()
        self.read_s += time.perf_counter() - started
        return layers if dtype is None else layers.astype(dtype, copy=False)


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
