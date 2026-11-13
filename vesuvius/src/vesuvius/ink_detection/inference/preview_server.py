"""Serve flat ink inference for the part of a segment a viewer is showing.

The server speaks "ink preview api 1", which VC3D's Ink Detection dock uses for
its live preview. The client samples the layer stack under the visible canvas
rectangle itself and posts it; the server runs the checkpoint on the patches
that overlap the view and returns their blended probabilities. Every numeric
step is villa's flat inference (``infer.py``): checkpoint loading, per-patch
normalization, the patch grid with its edge-flush last position, layer
selection and reversal, the forward call and its autocast, the sigmoid, the
blend window and the uint8 truncation. A preview pixel therefore equals the
pixel ``infer.py --roi`` writes for the same input array and settings.

Run it with::

    python -m vesuvius.ink_detection.inference.preview_server --checkpoint CKPT.pth

Protocol (all coordinates are canvas pixels; the canvas is the flat image the
uploaded stack is sampled on, and the patch grid starts at its pixel (0, 0)):

``GET /health``
    JSON with ``service``, ``api``, ``backend`` ("torch:<device>"), ``model``,
    ``patch``, ``stride``, ``depth``, ``voxel_um`` (null when unknown),
    ``max_view_px``, ``blend_mode`` and ``cache`` statistics.

``POST /preview?key=K&cw=CW&ch=CH&x=X&y=Y&w=W&h=H&variant=f&format=u8``
    Optional body: ``depth * SH * SW`` voxels, layers first, C order,
    little-endian, described by ``sx``, ``sy``, ``sw``, ``sh`` and ``dtype``
    (``uint8`` or ``uint16``). ``key`` names everything that decides the
    voxels under the canvas; the client changes it when any of that changes.
    ``variant`` is ``f`` or ``r`` (the uploaded layers reversed, as
    ``--direction reverse`` does). ``format`` is ``u8`` (floor(255 p)),
    ``f32`` (float32 little-endian) or ``png`` (8-bit gray).

    200: the view, with headers ``X-Shape: h,w``, ``X-Backend`` and
    ``X-Tiles: computed,cached``. 409: JSON ``{"need": {x, y, w, h},
    "depth": D, "missing": N}``, the canvas rectangle the client must upload
    next; nothing is computed. 400, 404, 413 and 500 carry JSON ``error``.

The server listens on 127.0.0.1 only. Its output is model output.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch
import vc_reader
from torch.utils.data import DataLoader

from vesuvius.ink_detection.inference.inference_runtime import (
    DEVICE_CHOICES,
    prepare_model_for_inference,
)
from vesuvius.ink_detection.inference.infer import (
    DEFAULT_OVERLAP,
    Block,
    ChunkAccumulator,
    ConfiguredModel,
    FlatBlockDataset,
    FlatPatchReader,
    clip_block_to_roi,
    clip_tile_to_roi,
    compute_chunk_contribution_counts,
    compute_importance_map_2d,
    configure_model,
    iter_blocks,
    normalize_probability_sums,
    probabilities_to_uint8,
    resolve_blend_mode,
    resolve_patch_stride,
    run_block_inference,
    select_layer_indices,
)
from vesuvius.ink_detection.models.checkpoint import load_checkpoint
from vesuvius.utils.cli import HyphenUnderscoreParser


LOGGER = logging.getLogger(__name__)

API_VERSION = 1
HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_CACHE_MB = 256.0
DEFAULT_MAX_VIEW_PX = 4096 * 4096
MAX_KEY_CHARS = 200
VARIANTS = {"f": "forward", "r": "reverse"}
FORMATS = ("u8", "f32", "png")
UPLOAD_DTYPES = {"uint8": np.dtype("<u1"), "uint16": np.dtype("<u2")}
_ENTRY_BYTES = 128  # bookkeeping charged per cache entry, so empty patches count too
_DISCARD_LIMIT = 1 << 28  # unused request bodies up to this size are read before replying


class PreviewError(Exception):
    """A reply other than 200, with its JSON payload."""

    def __init__(self, status: int, payload: Mapping[str, Any]) -> None:
        super().__init__(payload)
        self.status = int(status)
        self.payload = dict(payload)


def _bad_request(text: str) -> PreviewError:
    return PreviewError(400, {"error": text})


class TileCache:
    """Patch probabilities, least recently used first out, under a byte cap.

    Keys are ``(key, variant, row position, column position)``. A value is the
    patch's valid ``(rows, columns)`` and its float32 probabilities, or None for
    a patch whose raw input was all zero.
    """

    def __init__(self, cap_bytes: int) -> None:
        self.cap_bytes = max(0, int(cap_bytes))
        self.hits = 0
        self.misses = 0
        self._bytes = 0
        self._entries: OrderedDict[tuple, tuple[tuple[int, int], np.ndarray | None]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _size(tile: np.ndarray | None) -> int:
        return _ENTRY_BYTES + (0 if tile is None else int(tile.nbytes))

    def get(self, key: tuple, shape: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
        """Return (found, tile); an entry of another shape is not a hit."""

        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry[0] == shape:
                self._entries.move_to_end(key)
                self.hits += 1
                return True, entry[1]
            self.misses += 1
            return False, None

    def put(self, key: tuple, shape: tuple[int, int], tile: np.ndarray | None) -> None:
        size = self._size(tile)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= self._size(old[1])
            if size > self.cap_bytes:
                return
            self._entries[key] = (shape, tile)
            self._bytes += size
            while self._bytes > self.cap_bytes:
                _, (_, evicted) = self._entries.popitem(last=False)
                self._bytes -= self._size(evicted)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tiles": len(self._entries),
                "mb": round(self._bytes / 2**20, 3),
                "cap_mb": round(self.cap_bytes / 2**20, 3),
                "hits": self.hits,
                "misses": self.misses,
            }


@dataclass(frozen=True)
class PreviewRequest:
    """One parsed ``POST /preview``; rectangles are (y0, y1, x0, x1), half open."""

    key: str
    canvas_hw: tuple[int, int]
    view: tuple[int, int, int, int]
    variant: str
    fmt: str
    upload: tuple[int, int, int, int] | None = None
    dtype: np.dtype = UPLOAD_DTYPES["uint8"]
    stack: np.ndarray | None = None


@dataclass(frozen=True)
class PreviewReply:
    """A 200 reply."""

    body: bytes
    content_type: str
    headers: dict[str, str]


class _TileSink:
    """Receives each patch's probabilities from ``run_block_inference``.

    It takes the place of the output accumulator: the server keeps every
    patch whole, so a cached patch can serve any later view.
    """

    def __init__(self, origin_yx: tuple[int, int]) -> None:
        self.origin_y, self.origin_x = origin_yx
        self.tiles: dict[tuple[int, int], np.ndarray | None] = {}

    def add_tile(self, *, y0: int, x0: int, tile: np.ndarray, tile_weights: np.ndarray) -> None:
        position = (y0 + self.origin_y, x0 + self.origin_x)
        self.tiles[position] = np.array(tile, dtype=np.float32, copy=True)

    def skip_tile(self, *, y0: int, x0: int, valid_h: int, valid_w: int) -> None:
        self.tiles[(y0 + self.origin_y, x0 + self.origin_x)] = None


def _single(query: Mapping[str, Sequence[str]], name: str, default: str | None = None) -> str | None:
    values = query.get(name)
    if not values:
        return default
    if len(values) > 1:
        raise _bad_request(f"parameter {name!r} was given more than once")
    return values[0]


def _integer(query: Mapping[str, Sequence[str]], name: str, *, minimum: int) -> int:
    text = _single(query, name)
    if text is None:
        raise _bad_request(f"missing parameter {name!r}")
    try:
        value = int(text, 10)
    except ValueError:
        raise _bad_request(f"parameter {name!r} must be an integer, got {text!r}") from None
    if value < minimum:
        raise _bad_request(f"parameter {name!r} must be at least {minimum}, got {value}")
    return value


def _inside(rect: tuple[int, int, int, int], canvas_hw: tuple[int, int]) -> bool:
    y0, y1, x0, x1 = rect
    return 0 <= y0 < y1 <= canvas_hw[0] and 0 <= x0 < x1 <= canvas_hw[1]


def _covers(rect: tuple[int, int, int, int], block: Block) -> bool:
    y0, y1, x0, x1 = rect
    return (
        y0 <= block.y0
        and block.y0 + block.valid_h <= y1
        and x0 <= block.x0
        and block.x0 + block.valid_w <= x1
    )


class PreviewEngine:
    """Flat inference over canvas views, with villa's grid and a patch cache."""

    def __init__(
        self,
        configured: ConfiguredModel,
        *,
        device: torch.device = torch.device("cpu"),
        overlap: float = DEFAULT_OVERLAP,
        stride: int | None = None,
        blend_mode: str = "auto",
        batch_size: int = 1,
        cache_mb: float = DEFAULT_CACHE_MB,
        max_view_px: int = DEFAULT_MAX_VIEW_PX,
        model_name: str = "model",
        voxel_um: float | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_view_px <= 0:
            raise ValueError("max_view_px must be positive")
        if cache_mb < 0:
            raise ValueError("cache_mb must be nonnegative")
        self.configured = configured
        self.device = torch.device(device)
        self.patch = int(configured.patch_size)
        self.depth = int(configured.input_depth)
        self.stride = resolve_patch_stride(
            patch_size=self.patch, overlap=overlap, explicit_stride=stride
        )
        self.blend_mode = resolve_blend_mode(
            blend_mode, stride=self.stride, patch_size=self.patch
        )
        self.weight_map = compute_importance_map_2d(
            patch_size=(self.patch, self.patch), mode=self.blend_mode
        ).numpy()
        self.batch_size = int(batch_size)
        self.max_view_px = int(max_view_px)
        self.model_name = str(model_name)
        self.voxel_um = None if voxel_um is None else float(voxel_um)
        self.backend = f"torch:{self.device}"
        self.cache = TileCache(int(float(cache_mb) * 2**20))
        self._lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "service": "ink-preview",
            "api": API_VERSION,
            "backend": self.backend,
            "model": self.model_name,
            "patch": self.patch,
            "stride": self.stride,
            "depth": self.depth,
            "voxel_um": self.voxel_um,
            "max_view_px": self.max_view_px,
            "blend_mode": self.blend_mode,
            "cache": self.cache.stats(),
        }

    def parse_query(self, query: Mapping[str, Sequence[str]]) -> PreviewRequest:
        """Validate the query of ``POST /preview``; the body is attached later."""

        key = _single(query, "key")
        if not key:
            raise _bad_request("missing parameter 'key'")
        if len(key) > MAX_KEY_CHARS:
            raise _bad_request(f"key is longer than {MAX_KEY_CHARS} characters")
        cw = _integer(query, "cw", minimum=1)
        ch = _integer(query, "ch", minimum=1)
        x = _integer(query, "x", minimum=0)
        y = _integer(query, "y", minimum=0)
        w = _integer(query, "w", minimum=1)
        h = _integer(query, "h", minimum=1)
        view = (y, y + h, x, x + w)
        if not _inside(view, (ch, cw)):
            raise _bad_request(f"view {w}x{h} at ({x}, {y}) is not inside the {cw}x{ch} canvas")
        variant = _single(query, "variant", "f")
        if variant not in VARIANTS:
            raise _bad_request(f"variant must be 'f' or 'r', got {variant!r}")
        fmt = _single(query, "format", "u8")
        if fmt not in FORMATS:
            raise _bad_request(f"format must be one of {', '.join(FORMATS)}, got {fmt!r}")
        if w * h > self.max_view_px:
            raise PreviewError(413, {"error": f"view of {w * h} pixels exceeds max_view_px {self.max_view_px}"})
        names = ("sx", "sy", "sw", "sh")
        given = [name for name in names if _single(query, name) is not None]
        upload = None
        dtype = UPLOAD_DTYPES["uint8"]
        if given:
            if len(given) != len(names):
                raise _bad_request("an upload needs all of sx, sy, sw and sh")
            sx = _integer(query, "sx", minimum=0)
            sy = _integer(query, "sy", minimum=0)
            sw = _integer(query, "sw", minimum=1)
            sh = _integer(query, "sh", minimum=1)
            upload = (sy, sy + sh, sx, sx + sw)
            if not _inside(upload, (ch, cw)):
                raise _bad_request(f"upload {sw}x{sh} at ({sx}, {sy}) is not inside the {cw}x{ch} canvas")
            # Every patch that touches the view lies within one patch of it.
            if sw * sh > (w + 2 * self.patch) * (h + 2 * self.patch):
                raise PreviewError(413, {"error": "upload is larger than the view plus one patch on each side"})
            dtype_name = _single(query, "dtype", "uint8")
            if dtype_name not in UPLOAD_DTYPES:
                raise _bad_request(f"dtype must be uint8 or uint16, got {dtype_name!r}")
            dtype = UPLOAD_DTYPES[dtype_name]
            if self.configured.preprocessing == "divide_255" and dtype_name != "uint8":
                raise _bad_request("this checkpoint divides by 255 and needs uint8 layers")
        return PreviewRequest(
            key=key, canvas_hw=(ch, cw), view=view, variant=variant, fmt=fmt, upload=upload, dtype=dtype
        )

    def body_bytes(self, request: PreviewRequest) -> int:
        """Exact body length the request's upload rectangle needs (0 without one)."""

        if request.upload is None:
            return 0
        y0, y1, x0, x1 = request.upload
        return self.depth * (y1 - y0) * (x1 - x0) * request.dtype.itemsize

    def check_body_length(self, request: PreviewRequest, length: int) -> None:
        if request.upload is None:
            if length:
                raise _bad_request("a body needs sx, sy, sw and sh")
            return
        expected = self.body_bytes(request)
        if length != expected:
            raise _bad_request(
                f"body has {length} bytes; the upload rectangle needs {expected} "
                f"({self.depth} layers of {request.dtype.name})"
            )

    def attach_body(self, request: PreviewRequest, body: bytes) -> PreviewRequest:
        self.check_body_length(request, len(body))
        if request.upload is None:
            return request
        y0, y1, x0, x1 = request.upload
        stack = np.frombuffer(body, dtype=request.dtype).reshape(self.depth, y1 - y0, x1 - x0)
        return replace(request, stack=stack)

    def preview(self, request: PreviewRequest) -> PreviewReply:
        """Answer one parsed request; raises PreviewError for 409 and errors."""

        canvas_hw = request.canvas_hw
        blocks = iter_blocks(canvas_hw, self.patch, self.stride, roi_bounds=request.view)
        with self._lock:
            tiles: dict[tuple[int, int], np.ndarray | None] = {}
            to_compute: list[Block] = []
            missing: list[Block] = []
            for block in blocks:
                position = (block.y0, block.x0)
                found, tile = self.cache.get(
                    (request.key, request.variant, *position), (block.valid_h, block.valid_w)
                )
                if found:
                    tiles[position] = tile
                elif request.stack is not None and _covers(request.upload, block):
                    to_compute.append(block)
                else:
                    missing.append(block)
            if missing:
                top = min(block.y0 for block in missing)
                left = min(block.x0 for block in missing)
                bottom = max(block.y0 + block.valid_h for block in missing)
                right = max(block.x0 + block.valid_w for block in missing)
                raise PreviewError(
                    409,
                    {
                        "need": {"x": left, "y": top, "w": right - left, "h": bottom - top},
                        "depth": self.depth,
                        "missing": len(missing),
                    },
                )
            cached = len(tiles)
            if to_compute:
                computed = self._compute(request, to_compute)
                for block in to_compute:
                    position = (block.y0, block.x0)
                    self.cache.put(
                        (request.key, request.variant, *position),
                        (block.valid_h, block.valid_w),
                        computed[position],
                    )
                    tiles[position] = computed[position]
            probability = self._blend(blocks, tiles, request.view)
        y0, y1, x0, x1 = request.view
        headers = {
            "X-Shape": f"{y1 - y0},{x1 - x0}",
            "X-Backend": self.backend,
            "X-Tiles": f"{len(to_compute)},{cached}",
        }
        if request.fmt == "f32":
            return PreviewReply(probability.astype("<f4", copy=False).tobytes(), "application/octet-stream", headers)
        gray = probabilities_to_uint8(probability)
        if request.fmt == "png":
            return PreviewReply(bytes(vc_reader.encode_png(np.ascontiguousarray(gray))), "image/png", headers)
        return PreviewReply(gray.tobytes(), "application/octet-stream", headers)

    def _compute(self, request: PreviewRequest, blocks: Sequence[Block]) -> dict[tuple[int, int], np.ndarray | None]:
        """Run villa's flat patch pipeline on blocks inside the uploaded rectangle."""

        sy0, sy1, sx0, sx1 = request.upload
        # The reader works in the uploaded rectangle's frame. Every block here
        # lies inside it, so its reads stay in bounds and the reader pads only
        # past the canvas edge, as villa pads a volume smaller than a patch.
        reader = FlatPatchReader(
            input_path="<uploaded layer stack>",
            resolution="0",
            depth_axis_first=True,
            height=sy1 - sy0,
            width=sx1 - sx0,
            layer_indices=select_layer_indices(
                self.depth,
                layer_start=None,
                layer_end=None,
                output_depth=self.depth,
                direction=VARIANTS[request.variant],
            ),
            output_depth=self.depth,
            preprocessing=self.configured.preprocessing,
            array=request.stack,
        )
        dataset = FlatBlockDataset(
            reader=reader,
            blocks=[Block(b.y0 - sy0, b.x0 - sx0, b.valid_h, b.valid_w) for b in blocks],
            patch_size=self.patch,
            preprocessing=self.configured.preprocessing,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )
        sink = _TileSink((sy0, sx0))
        run_block_inference(
            loader=loader,
            model=self.configured.model,
            accumulator=sink,
            weight_map=self.weight_map,
            mask=None,
            device=self.device,
            amp_dtype=self.configured.amp_dtype,
            tta_axes=(),
            tta_batch_size=None,
        )
        return sink.tiles

    def _blend(
        self,
        blocks: Sequence[Block],
        tiles: Mapping[tuple[int, int], np.ndarray | None],
        view: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Blend patches into the view in schedule order, as the ROI path does."""

        shape = (view[1] - view[0], view[3] - view[2])
        probability_sum = np.zeros(shape, dtype=np.float32)
        weight_sum = np.zeros(shape, dtype=np.float32)
        accumulator = ChunkAccumulator(
            shape=shape,
            chunk_shape=shape,
            prob_sum_store=probability_sum,
            weight_sum_store=weight_sum,
            contribution_counts=compute_chunk_contribution_counts(
                [clip_block_to_roi(block, view) for block in blocks], chunk_shape=shape
            ),
        )
        for block in blocks:
            tile = tiles[(block.y0, block.x0)]
            if tile is None:
                clipped = clip_block_to_roi(block, view)
                accumulator.skip_tile(
                    y0=clipped.y0, x0=clipped.x0, valid_h=clipped.valid_h, valid_w=clipped.valid_w
                )
                continue
            clipped, part, part_weights = clip_tile_to_roi(
                block, tile, self.weight_map[: block.valid_h, : block.valid_w], view
            )
            accumulator.add_tile(y0=clipped.y0, x0=clipped.x0, tile=part, tile_weights=part_weights)
        accumulator.flush_remaining()
        return normalize_probability_sums(probability_sum, weight_sum)


class PreviewRequestHandler(BaseHTTPRequestHandler):
    """HTTP front end of a PreviewEngine (``self.server.engine``)."""

    server_version = "vesuvius-ink-preview/1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        LOGGER.debug("%s %s", self.address_string(), format % args)

    def _send(
        self, status: int, body: bytes, content_type: str, headers: Mapping[str, str] | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/health":
            self._send_json(200, self.server.engine.health())
        else:
            self._send_json(404, {"error": f"unknown path {path!r}"})

    def _discard_body(self, length: int) -> None:
        """Read an unused body so closing the socket does not reset the reply."""

        if length > _DISCARD_LIMIT:
            self.close_connection = True
            return
        while length > 0:
            chunk = self.rfile.read(min(length, 1 << 20))
            if not chunk:
                break
            length -= len(chunk)

    def do_POST(self) -> None:  # noqa: N802
        engine: PreviewEngine = self.server.engine
        parts = urlsplit(self.path)
        length_text = self.headers.get("Content-Length", "0")
        try:
            length = int(length_text)
        except ValueError:
            length = -1
        body_read = False
        try:
            if parts.path != "/preview":
                raise PreviewError(404, {"error": f"unknown path {parts.path!r}"})
            if length < 0:
                self.close_connection = True
                raise _bad_request(f"bad Content-Length {length_text!r}")
            request = engine.parse_query(parse_qs(parts.query, keep_blank_values=True))
            engine.check_body_length(request, length)
            body = self.rfile.read(length)
            body_read = True
            reply = engine.preview(engine.attach_body(request, body))
        except Exception as exc:
            if not body_read:
                self._discard_body(length)
            if isinstance(exc, PreviewError):
                self._send_json(exc.status, exc.payload)
            else:  # the server stays up and reports the failure
                LOGGER.exception("Preview request failed")
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send(200, reply.body, reply.content_type, reply.headers)


def make_server(engine: PreviewEngine, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Bind a threaded HTTP server for ``engine`` to 127.0.0.1 only."""

    server = ThreadingHTTPServer((HOST, int(port)), PreviewRequestHandler)
    server.daemon_threads = True
    server.engine = engine
    return server


def checkpoint_voxel_um(checkpoint: str | Path) -> float | None:
    """The training voxel size the checkpoint config states, else None.

    Flat checkpoints carry no voxel size field today; a positive number under
    ``voxel_size_um`` in the embedded config is reported.
    """

    payload = load_checkpoint(Path(checkpoint))
    config = payload.get("config") if isinstance(payload, Mapping) else None
    value = config.get("voxel_size_um") if isinstance(config, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
        return float(value)
    return None


def load_engine(args: argparse.Namespace) -> PreviewEngine:
    """Load the checkpoint through flat inference's own model setup."""

    configured = configure_model(
        argparse.Namespace(
            checkpoint=Path(args.checkpoint),
            ensemble_checkpoint=[],
            amp_dtype=args.amp_dtype,
        )
    )
    model, device = prepare_model_for_inference(
        configured.model,
        gpu_ids=(),
        device_type=args.device,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
    )
    voxel_um = args.voxel_um if args.voxel_um is not None else checkpoint_voxel_um(args.checkpoint)
    return PreviewEngine(
        replace(configured, model=model),
        device=device,
        overlap=args.overlap,
        stride=args.stride,
        blend_mode=args.blend_mode,
        batch_size=args.batch_size,
        cache_mb=args.cache_mb,
        max_view_px=args.max_view_px,
        model_name=args.name or Path(args.checkpoint).stem,
        voxel_um=voxel_um,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = HyphenUnderscoreParser(
        description="Serve flat ink inference for the canvas view a viewer is showing (127.0.0.1 only)"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                        help="auto picks CUDA, then MPS, then CPU.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--cache-mb", type=float, default=DEFAULT_CACHE_MB,
                        help="Memory cap of the patch cache in MiB (0 disables it).")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP,
                        help="Sliding-window overlap fraction, as in infer.py. Default: 0.5.")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--blend-mode", choices=("auto", "constant", "gaussian", "hann"), default="auto")
    parser.add_argument("--amp-dtype", choices=("auto", "default", "fp16", "bf16"), default="auto",
                        help="CUDA autocast dtype, as in infer.py; auto uses the checkpoint's.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-view-px", type=int, default=DEFAULT_MAX_VIEW_PX,
                        help="Largest view (w * h) answered; larger views get 413.")
    parser.add_argument("--name", help="Model name for /health (default: the checkpoint file stem).")
    parser.add_argument("--voxel-um", type=float,
                        help="Training voxel size for /health when the checkpoint does not state one.")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap the model in torch.compile, as infer.py does by default.")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.cache_mb < 0:
        parser.error("--cache-mb must be nonnegative")
    if args.max_view_px <= 0:
        parser.error("--max-view-px must be positive")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    engine = load_engine(args)
    server = make_server(engine, args.port)
    host, port = server.server_address[:2]
    LOGGER.info(
        "Ink preview api %d on http://%s:%d: model=%s backend=%s patch=%d stride=%d depth=%d blend=%s",
        API_VERSION, host, port, engine.model_name, engine.backend,
        engine.patch, engine.stride, engine.depth, engine.blend_mode,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
