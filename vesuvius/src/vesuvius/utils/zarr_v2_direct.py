"""Direct, thread-scalable reads of local Zarr v2 arrays.

``zarr-python`` 3 funnels every synchronous read through one event loop, so
reading an array from many threads does not scale (measured ~400 MB/s decoded
regardless of thread count on a 128**3-chunk uint8 volume). For the common
case in this repo (a local Zarr v2 array with a numcodecs compressor, no
filters, C order) the chunk files can instead be read and decoded directly in
the calling thread, which scales with the thread pool.

Anything outside that envelope raises ``UnsupportedZarrArray`` at construction
so callers can fall back to ``zarr.open``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numcodecs import get_codec


class UnsupportedZarrArray(ValueError):
    """The array cannot be read by :class:`DirectZarrV2Reader`."""


def _parse_fill_value(raw: Any, dtype: np.dtype) -> Any:
    if raw is None:
        return np.zeros((), dtype=dtype)[()]
    if isinstance(raw, str) and np.issubdtype(dtype, np.floating):
        special = {"NaN": np.nan, "Infinity": np.inf, "-Infinity": -np.inf}
        if raw in special:
            return dtype.type(special[raw])
    return np.array(raw, dtype=dtype)[()]


@dataclass(frozen=True, slots=True)
class DirectZarrV2Reader:
    """Read rectangular regions of a local Zarr v2 array without zarr-python.

    Chunk keys, compression, fill value and edge-chunk padding follow the v2
    spec: every stored chunk is a full ``chunks``-shaped block, missing chunk
    files mean ``fill_value``.
    """

    array_dir: Path
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype
    fill_value: Any
    separator: str
    compressor_config: dict[str, Any] | None

    @classmethod
    def open(cls, array_dir: str | Path) -> DirectZarrV2Reader:
        array_dir = Path(array_dir)
        meta_path = array_dir / ".zarray"
        if not meta_path.is_file():
            raise UnsupportedZarrArray(f"No .zarray in {array_dir}")
        meta = json.loads(meta_path.read_text())
        if meta.get("zarr_format") != 2:
            raise UnsupportedZarrArray(f"Not a Zarr v2 array: {array_dir}")
        if meta.get("order", "C") != "C":
            raise UnsupportedZarrArray(f"Only C-order arrays are supported: {array_dir}")
        if meta.get("filters"):
            raise UnsupportedZarrArray(f"Arrays with filters are not supported: {array_dir}")
        dtype = np.dtype(meta["dtype"])
        if dtype.kind not in "biuf":
            raise UnsupportedZarrArray(f"Unsupported dtype {dtype} in {array_dir}")
        compressor = meta.get("compressor")
        if compressor is not None:
            try:
                get_codec(dict(compressor))
            except Exception as exc:  # pragma: no cover - depends on installed codecs
                raise UnsupportedZarrArray(f"Unsupported compressor {compressor!r}: {exc}") from exc
        shape = tuple(int(v) for v in meta["shape"])
        chunks = tuple(int(v) for v in meta["chunks"])
        if len(shape) != len(chunks) or any(c <= 0 for c in chunks):
            raise UnsupportedZarrArray(f"Invalid chunks {chunks} for shape {shape} in {array_dir}")
        return cls(
            array_dir=array_dir,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            fill_value=_parse_fill_value(meta.get("fill_value"), dtype),
            separator=str(meta.get("dimension_separator") or "."),
            compressor_config=None if compressor is None else dict(compressor),
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def _chunk_path(self, index: Sequence[int]) -> Path:
        return self.array_dir / self.separator.join(str(i) for i in index)

    def _decode_chunk(self, raw: bytes, codec: Any) -> np.ndarray:
        payload = raw if codec is None else codec.decode(raw)
        chunk = np.frombuffer(payload, dtype=self.dtype)
        expected = math.prod(self.chunks)
        if chunk.size != expected:
            raise ValueError(
                f"Chunk in {self.array_dir} decoded to {chunk.size} elements, expected {expected}"
            )
        return chunk.reshape(self.chunks)

    def read(self, ranges: Sequence[tuple[int, int]]) -> np.ndarray:
        """Return ``array[r0[0]:r0[1], r1[0]:r1[1], ...]`` as a C-contiguous array.

        Ranges must lie inside ``shape``; empty ranges yield empty output.
        """

        if len(ranges) != self.ndim:
            raise ValueError(f"Expected {self.ndim} ranges, got {len(ranges)}")
        starts = []
        stops = []
        for dim, (start, stop) in enumerate(ranges):
            if not (0 <= start <= stop <= self.shape[dim]):
                raise ValueError(f"Range {(start, stop)} out of bounds for axis {dim} of shape {self.shape}")
            starts.append(int(start))
            stops.append(int(stop))

        out_shape = tuple(stop - start for start, stop in zip(starts, stops))
        out = np.empty(out_shape, dtype=self.dtype)
        if out.size == 0:
            return out
        out.fill(self.fill_value)

        # numcodecs codecs are not documented as thread-safe; instantiate per call.
        codec = None if self.compressor_config is None else get_codec(self.compressor_config)

        chunk_ranges = [
            range(start // size, -(-stop // size))
            for start, stop, size in zip(starts, stops, self.chunks)
        ]
        for chunk_index in np.ndindex(*(len(r) for r in chunk_ranges)):
            index = tuple(r[i] for r, i in zip(chunk_ranges, chunk_index))
            path = self._chunk_path(index)
            try:
                raw = path.read_bytes()
            except FileNotFoundError:
                continue
            chunk = self._decode_chunk(raw, codec)

            src: list[slice] = []
            dst: list[slice] = []
            for dim, ci in enumerate(index):
                chunk_start = ci * self.chunks[dim]
                lo = max(starts[dim], chunk_start)
                hi = min(stops[dim], chunk_start + self.chunks[dim])
                src.append(slice(lo - chunk_start, hi - chunk_start))
                dst.append(slice(lo - starts[dim], hi - starts[dim]))
            out[tuple(dst)] = chunk[tuple(src)]

        return out
