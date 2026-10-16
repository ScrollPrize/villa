"""Zarr v2 array metadata (``.zarray``) and chunk decoding."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ArrayMeta:
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype
    fill_value: Any
    order: str = "C"
    separator: str = "."
    compressor: Mapping[str, Any] | None = None
    filters: tuple[Mapping[str, Any], ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_zarray(cls, d: Mapping[str, Any]) -> ArrayMeta:
        if d.get("zarr_format") != 2:
            raise ValueError(f"only zarr v2 arrays are supported, got zarr_format={d.get('zarr_format')!r}")
        dtype = np.dtype(d["dtype"])
        if dtype.hasobject:
            raise ValueError(f"object dtype {dtype} is not supported")
        fill = d.get("fill_value")
        return cls(
            shape=tuple(int(n) for n in d["shape"]),
            chunks=tuple(int(n) for n in d["chunks"]),
            dtype=dtype,
            fill_value=0 if fill is None else np.asarray(fill, dtype=dtype).item(),
            order=d.get("order") or "C",
            separator=d.get("dimension_separator") or ".",
            compressor=d.get("compressor"),
            filters=tuple(d.get("filters") or ()),
            raw=d,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ArrayMeta:
        return cls.from_zarray(json.loads(data))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def is_raw(self) -> bool:
        """Chunks are stored as plain C-order bytes: no compressor, no filters."""
        return self.compressor is None and not self.filters and self.order == "C"

    @property
    def chunk_nbytes(self) -> int:
        return math.prod(self.chunks) * self.dtype.itemsize

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(-(-s // c) for s, c in zip(self.shape, self.chunks))

    def chunk_key(self, idx: tuple[int, ...]) -> str:
        return self.separator.join(map(str, idx))

    def decoded_zarray(self) -> dict[str, Any]:
        """``.zarray`` of the same array stored uncompressed (the disk-cache layout)."""
        return dict(self.raw, compressor=None, filters=None, order="C")


class ChunkDecoder:
    """Decodes stored chunk bytes into a full chunk ndarray (writable, C order)."""

    def __init__(self, meta: ArrayMeta):
        self.meta = meta
        self._codec = None
        self._filters: list[Any] = []
        if meta.compressor is not None or meta.filters:
            import numcodecs

            from .codecs import ensure_registered

            for config in (meta.compressor, *meta.filters):
                if config is not None:
                    ensure_registered(config.get("id"))
            if meta.compressor is not None:
                self._codec = numcodecs.get_codec(dict(meta.compressor))
            self._filters = [numcodecs.get_codec(dict(f)) for f in meta.filters]

    def decode(self, data: Any) -> np.ndarray:
        m = self.meta
        if self._codec is None and not self._filters:
            buf = data
        else:
            buf = self._codec.decode(data) if self._codec is not None else data
            for f in reversed(self._filters):
                buf = f.decode(buf)
        arr = np.frombuffer(buf, dtype=m.dtype)
        if arr.size != math.prod(m.chunks):
            raise ValueError(f"chunk decoded to {arr.size} elements, expected {math.prod(m.chunks)}")
        arr = arr.reshape(m.chunks, order=m.order)
        if m.order != "C" or not arr.flags.writeable:
            arr = np.ascontiguousarray(arr) if m.order != "C" else arr.copy()
        return arr
