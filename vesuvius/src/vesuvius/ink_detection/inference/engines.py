"""Optional alternative engines for flat ink inference.

The default engine is the PyTorch path in ``infer.py``. An alternative engine
receives the same per-direction request that the PyTorch path would run (the
source layers in model order and villa's tile schedule after the low-resolution
occupancy skip) and returns the same uint8 map and its coverage. villa keeps
scheduling, TIFF writing and folder handling, so an engine only replaces the
forward, normalization and blending work.

Engines are opt-in with ``--engine``. When the requested engine cannot run, the
command logs why and continues with PyTorch, unless ``--engine-fallback error``
is given.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch

from vesuvius.ink_detection.models.checkpoint import resolve_checkpoint_path

LOGGER = logging.getLogger(__name__)

ENGINE_CHOICES = ("torch", "inkrs")
FALLBACK_CHOICES = ("torch", "error")

# inkrs judges the checkpoint, the volume and the GPU itself and raises
# inkrs.UnsupportedError for what it does not implement; villa checks its own
# options, which inkrs never sees, and the window limits before reading one
INKRS_BLEND_MODE = "hann"
INKRS_MAX_WINDOW_BYTES = 2**32 - 1


class EngineUnavailable(RuntimeError):
    """The requested engine cannot run this command or this input.

    ``FlatEngine.predict`` raises it to decline an input before doing any
    work; villa then runs that input with PyTorch, or stops under
    ``--engine-fallback error``.
    """


@dataclass(frozen=True)
class FlatEngineRequest:
    """One direction of one flat input, scheduled exactly as PyTorch runs it.

    ``blocks`` are villa's tiles after the low-resolution occupancy skip, in
    villa's order, and ``occupancy`` is the 2D mask that skip used (None when
    the input has no pyramid to scan); its scale to the volume is
    ``round(height / mask_height)``. The engine applies the second skip
    itself: a tile whose selected source layers are all zero contributes
    nothing.
    """

    input_zarr: str | Path
    resolution: str
    volume: Any
    depth_axis_first: bool
    height: int
    width: int
    layer_indices: np.ndarray
    input_depth: int
    patch_size: int
    stride: int
    blend_mode: str
    preprocessing: str
    blocks: tuple[Any, ...]
    occupancy: np.ndarray | None = None
    # shared by the requests of one run, so the directions of an input read
    # their common source layers once; None reads every request afresh
    window_cache: WindowCache | None = None


@dataclass(frozen=True)
class FlatEngineResult:
    """A finished flat map in villa's output contract.

    ``probability`` is ``trunc(255 * clip(sum(p * w) / sum(w), 0, 1))`` as
    uint8, 0 where the weight sum is at most 1e-6. ``coverage`` is true where
    the weight sum is above 1e-6.
    """

    probability: np.ndarray
    coverage: np.ndarray


class FlatEngine(Protocol):
    """An alternative to the PyTorch forward, normalization and blend."""

    name: str

    def unsupported_reason(self, request: FlatEngineRequest) -> str | None:
        """Return why this request cannot run, or None when it can."""

    def predict(self, request: FlatEngineRequest) -> FlatEngineResult:
        """Run one request and return its finished map.

        Raises EngineUnavailable to decline the request before any work.
        """


EngineFactory = Callable[[argparse.Namespace, Any, torch.device], FlatEngine]


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


SpanKey = tuple[str, str, bool, int, int]


class WindowCache:
    """The last source-layer span read, kept for the next request that needs it.

    The forward and reverse runs of one input select the same layers in
    opposite orders, so the second takes its window from the first's read.
    A span can also be read ahead on a thread (``prefetch``), so the read
    overlaps the engine's load. One span is kept at a time; a request for
    another span replaces it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entry: tuple[SpanKey, Future] | None = None
        self._threads: list[threading.Thread] = []

    def prefetch(self, key: SpanKey, read: Callable[[], np.ndarray]) -> None:
        """Start ``read`` on a thread; a later ``span(key, ...)`` waits for it."""

        future: Future = Future()
        with self._lock:
            self._entry = (key, future)

        def work() -> None:
            try:
                future.set_result(_shared(read()))
            except BaseException as exc:
                future.set_exception(exc)

        thread = threading.Thread(target=work, name="engine-window-prefetch", daemon=True)
        with self._lock:
            self._threads.append(thread)
        thread.start()

    def close(self) -> None:
        """Wait for any read ahead to finish, then drop the kept span.

        A zarr read can't be interrupted, so this bounds its life to the run
        that started it rather than leaving it to hold memory afterwards.
        """

        with self._lock:
            threads, self._threads = self._threads, []
            self._entry = None
        for thread in threads:
            thread.join()

    def span(self, key: SpanKey, read: Callable[[], np.ndarray]) -> np.ndarray:
        """Return the span for ``key``, calling ``read`` unless it is cached."""

        with self._lock:
            entry = self._entry
        if entry is not None and entry[0] == key:
            try:
                return entry[1].result()
            except Exception:
                # a failed read ahead is read again here, so any error is this call's own
                pass
        span = _shared(read())
        future: Future = Future()
        future.set_result(span)
        with self._lock:
            self._entry = (key, future)
        return span


def _shared(span: np.ndarray) -> np.ndarray:
    # later requests share this array, so no engine may change it
    span.flags.writeable = False
    return span


def span_key(
    input_zarr: str | Path, resolution: str, depth_axis_first: bool, layer_indices: Any
) -> tuple[SpanKey, int, int]:
    """The cache key of the span holding ``layer_indices``, and its bounds."""

    indices = np.asarray(layer_indices, dtype=np.int64)
    low, high = int(indices.min()), int(indices.max()) + 1
    return (str(input_zarr), str(resolution), bool(depth_axis_first), low, high), low, high


def read_span(volume: Any, depth_axis_first: bool, low: int, high: int) -> np.ndarray:
    """Read layers ``low:high`` as a C-contiguous (Z, H, W) array."""

    if depth_axis_first:
        return np.ascontiguousarray(volume[low:high])
    return np.ascontiguousarray(np.moveaxis(np.asarray(volume[:, :, low:high]), -1, 0))


def read_window(request: FlatEngineRequest) -> np.ndarray:
    """Read the selected source layers as a contiguous (Z, H, W) array.

    Layers are in model order, so a reverse run is already reversed. With a
    ``window_cache``, a span an earlier request read is not read again.
    """

    indices = np.asarray(request.layer_indices, dtype=np.int64)
    key, low, high = span_key(
        request.input_zarr, request.resolution, request.depth_axis_first, indices
    )

    def read() -> np.ndarray:
        return read_span(request.volume, request.depth_axis_first, low, high)

    cache = request.window_cache
    span = read() if cache is None else cache.span(key, read)
    if np.array_equal(indices, np.arange(low, high)):
        # forward order is the span itself; the engine only reads it
        return span
    return np.ascontiguousarray(span[indices - low])


class InkrsEngine:
    """Run flat inference through the optional ``inkrs`` package.

    ``inkrs`` is imported only when this engine is selected, so villa neither
    depends on it nor imports it by default.
    """

    name = "inkrs"

    def __init__(self, module: Any, handle: Any, *, device_index: int) -> None:
        self._module = module
        self._handle = handle
        self.device_index = int(device_index)

    @classmethod
    def load(
        cls,
        args: argparse.Namespace,
        configured_model: Any,
        device: torch.device,
    ) -> "InkrsEngine":
        """Check the command, import inkrs and load the checkpoint into it."""

        reasons = inkrs_command_reasons(args, configured_model, device)
        if reasons:
            raise EngineUnavailable("; ".join(reasons))

        try:
            module = importlib.import_module("inkrs")
        except ImportError as exc:
            raise EngineUnavailable(
                f"the inkrs package is not installed ({exc}); install villa's "
                "inkrs extra (`uv sync --extra models --extra inkrs`)"
            ) from exc

        device_index = _engine_device_index(device)
        # resolve the path as the PyTorch loader does, so a quoted ~ reaches inkrs
        # expanded instead of failing inkrs's own file check
        checkpoint = resolve_checkpoint_path(args.checkpoint)
        handle = _inkrs_open(module, checkpoint, device_index)
        LOGGER.info(
            "Using the inkrs engine %s on device %d",
            getattr(module, "__version__", "(unknown version)"),
            device_index,
        )
        return cls(module, handle, device_index=device_index)

    def unsupported_reason(self, request: FlatEngineRequest) -> str | None:
        reason = inkrs_tiling_reason(request.blend_mode, request.stride, request.patch_size)
        if reason is not None:
            return reason
        # villa zero-pads a shallow input to the model depth; inkrs has no padding
        if len(request.layer_indices) != request.input_depth:
            return (
                f"it needs exactly {request.input_depth} source layers, got "
                f"{len(request.layer_indices)} (the input is shallower than the model)"
            )
        return inkrs_window_reason(
            request.volume, request.input_depth, request.height, request.width, request.patch_size
        )

    def predict(self, request: FlatEngineRequest) -> FlatEngineResult:
        started = time.perf_counter()
        window = read_window(request)
        read_s = time.perf_counter() - started
        probability, coverage = _inkrs_infer(
            self._module, self._handle, window, request.occupancy, stride=request.stride
        )
        LOGGER.info(
            "inkrs ran %d scheduled tiles in %.2f s (window read %.2f s)",
            len(request.blocks),
            time.perf_counter() - started,
            read_s,
        )
        return FlatEngineResult(
            probability=np.asarray(probability, dtype=np.uint8),
            coverage=np.asarray(coverage, dtype=bool),
        )


def inkrs_tiling_reason(blend_mode: str, stride: int, patch_size: int) -> str | None:
    """Why inkrs cannot tile with this blend mode and stride, or None when it can."""

    if blend_mode != INKRS_BLEND_MODE:
        return f"it blends with Hann only, got {blend_mode!r}"
    # inkrs tiles at the reader's own patch size (128, or 384 for reader-v3)
    if not 1 <= stride <= patch_size:
        return f"it needs a stride from 1 to the patch size {patch_size}, got {stride}"
    return None


def inkrs_window_reason(
    volume: Any, depth: int, height: int, width: int, patch_size: int
) -> str | None:
    """Why inkrs cannot take a ``depth``-layer window of ``volume``, or None when it can.

    It judges from the volume's metadata alone, so villa can decline a window
    before reading it: inkrs makes the same checks, but only once it has the
    window in memory, which for an oversized input may not fit.
    """

    # inkrs's documented input limits: uint8 layers, each side at least one
    # tile, and a window under 4 GiB, since its kernels index it with u32
    dtype = getattr(volume, "dtype", None)
    if dtype is not None and np.dtype(dtype) != np.uint8:
        return f"it reads uint8 volumes only, got {np.dtype(dtype)}"
    if min(height, width) < patch_size:
        return f"the input's {height}x{width} layers are smaller than one {patch_size} px tile"
    window_bytes = int(depth) * int(height) * int(width)
    if window_bytes >= INKRS_MAX_WINDOW_BYTES:
        return f"its {depth}-layer window of {window_bytes} bytes is 4 GiB or more"
    return None


def inkrs_command_reasons(
    args: argparse.Namespace, configured_model: Any, device: torch.device
) -> list[str]:
    """Why ``--engine inkrs`` cannot run this command at all, from villa's own options.

    Empty when it may run; inkrs then judges the checkpoint, GPU and inputs.
    ``InkrsEngine.load`` checks these before it imports inkrs.
    """

    reasons = []
    # inkrs exports a .pth itself and reads any other path as its own
    # safetensors export, which no checkpoint villa loads is
    checkpoint = Path(args.checkpoint)
    if checkpoint.suffix != ".pth":
        reasons.append(f"it loads .pth checkpoints only, got {checkpoint.name}")
    # inkrs's macOS build runs on Metal, which villa reaches as the MPS device
    if device.type not in ("cuda", "mps"):
        reasons.append(f"it needs a CUDA device, or MPS on macOS, got {device.type!r}")
    if len(getattr(args, "gpu_ids", ())) > 1:
        reasons.append("it runs on one GPU; drop --gpus or pass one id")
    if getattr(args, "ensemble_checkpoint", None):
        reasons.append("it does not run --ensemble-checkpoint")
    if getattr(args, "tta_mirror", False):
        reasons.append("it does not run --tta-mirror")
    # inkrs's export reads the precision the checkpoint stores, never
    # --amp-dtype, and refuses anything but fp16; --amp-dtype still sets
    # PyTorch's own dtype, so the effective dtype must be fp16 as well (no
    # dtype means torch.autocast's CUDA default, fp16 unless changed)
    amp_dtype = configured_model.amp_dtype or torch.get_autocast_dtype("cuda")
    if configured_model.stored_amp_dtype != torch.float16:
        stored = configured_model.stored_mixed_precision
        reasons.append(
            "it runs the precision the checkpoint stores and needs mixed_precision "
            f"fp16, but {checkpoint.name} stores "
            + ("none" if stored is None else repr(stored))
        )
    elif amp_dtype != torch.float16:
        reasons.append(f"it reproduces fp16 autocast only, got {amp_dtype}")
    # an ROI, a mask and the tiling hold for every input, so a command inkrs
    # would decline for them never loads it
    if getattr(args, "roi", None) is not None:
        reasons.append("it does not crop to --roi")
    if getattr(args, "mask_path", None) is not None:
        reasons.append("it does not apply --mask-path")
    tiling = _command_tiling(args, configured_model.patch_size)
    if tiling is not None:
        stride, blend_mode = tiling
        reason = inkrs_tiling_reason(blend_mode, stride, configured_model.patch_size)
        if reason is not None:
            reasons.append(reason)
    return reasons


def _command_tiling(args: argparse.Namespace, patch_size: int) -> tuple[int, str] | None:
    # infer imports this module, so its tiling rule is imported when called
    from vesuvius.ink_detection.inference.infer import resolve_tiling

    try:
        return resolve_tiling(args, patch_size)
    except ValueError:
        # the run reports a bad stride or overlap itself, before any input
        return None


def _engine_device_index(device: torch.device) -> int:
    """The device index an engine uses for villa's ``device``.

    An index-less CUDA device is PyTorch's current CUDA device, which
    ``torch.cuda.set_device`` may have moved off GPU 0, so the engine and the
    PyTorch path (including its per-input fallback) run on the same GPU. MPS
    has a single device, index 0.
    """

    if device.index is not None:
        return int(device.index)
    if device.type == "cuda":
        return int(torch.cuda.current_device())
    return 0


def _inkrs_open(module: Any, checkpoint: Path, device_index: int) -> Any:
    """Load a checkpoint into inkrs, mapping its refusals to EngineUnavailable.

    inkrs raises ``inkrs.UnsupportedError`` for a model or GPU it does not
    implement, or a ``.pth`` it cannot export, such as one whose keys or
    tensor shapes do not match the model (villa's ``strict=False`` load runs
    one with missing or unexpected keys); for a ``.pth`` it finds or makes its
    safetensors export itself. Any other ``inkrs.InkrsError``, such as missing
    CUDA libraries, is a real failure and is raised.
    """

    try:
        return module.Engine(checkpoint, device=device_index)
    except module.UnsupportedError as exc:
        raise EngineUnavailable(
            f"inkrs cannot load {checkpoint.name} on device {device_index}: {exc}"
        ) from exc


def _inkrs_infer(
    module: Any,
    handle: Any,
    window: np.ndarray,
    occupancy: np.ndarray | None,
    *,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    # the window is already in model order, so it runs forward from layer 0;
    # villa's own occupancy mask keeps its tile schedule
    try:
        [(probability, coverage)] = handle.run(
            window, [(0, 1)], stride=stride, occupancy=occupancy, coverage=True
        )
    except module.UnsupportedError as exc:
        # inkrs checks the volume before any GPU work, so nothing ran
        raise EngineUnavailable(f"inkrs does not support this input: {exc}") from exc
    return probability, coverage


ENGINE_FACTORIES: dict[str, EngineFactory] = {"inkrs": InkrsEngine.load}


def select_engine(
    args: argparse.Namespace,
    configured_model: Any,
    device: torch.device,
    *,
    factory: EngineFactory | None = None,
) -> FlatEngine | None:
    """Return the requested engine, or None for the PyTorch path.

    ``factory`` replaces the factory that ``--engine`` names, for tools that
    compare engines.
    """

    name = str(getattr(args, "engine", "torch"))
    if factory is None and name == "torch":
        return None
    factory = factory or ENGINE_FACTORIES[name]
    try:
        return factory(args, configured_model, device)
    except EngineUnavailable as exc:
        if getattr(args, "engine_fallback", "torch") == "error":
            raise
        LOGGER.warning(
            "The %s engine cannot run this command: %s. Continuing with PyTorch.",
            name,
            exc,
        )
        return None


def decline_request(
    engine: FlatEngine, request: FlatEngineRequest, reason: str, *, fallback: str
) -> None:
    """Log that PyTorch runs this request instead, or raise under fallback=error."""

    message = f"The {engine.name} engine cannot run {request.input_zarr}: {reason}"
    if fallback == "error":
        raise EngineUnavailable(message)
    LOGGER.warning("%s. Running this input with PyTorch.", message)
