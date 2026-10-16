"""Zarr v2 array metadata (``.zarray``)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

#: Compressors the native reader decodes, by numcodecs id (None: uncompressed).
SUPPORTED_COMPRESSORS = {None: "", "blosc": "blosc", "zstd": "zstd", "lzw": "lzw",
                         "imagecodecs_lzw": "lzw", "vc-delta3d": "vc-delta3d"}


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

    @property
    def codec_id(self) -> str:
        """The native reader's id for this array's compressor; ValueError if it cannot read the chunks."""
        cid = self.compressor.get("id") if self.compressor else None
        if cid not in SUPPORTED_COMPRESSORS:
            raise ValueError(f"vc_reader cannot read arrays compressed with {cid!r} "
                             f"(supported: uncompressed, {', '.join(c for c in SUPPORTED_COMPRESSORS if c)})")
        if self.filters:
            raise ValueError("vc_reader cannot read arrays with numcodecs filters")
        if self.order != "C":
            raise ValueError("vc_reader cannot read arrays stored in Fortran order")
        return SUPPORTED_COMPRESSORS[cid]

    def chunk_key(self, idx: tuple[int, ...]) -> str:
        return self.separator.join(map(str, idx))

    def decoded_zarray(self) -> dict[str, Any]:
        """``.zarray`` of the same array stored uncompressed (the decoded-cache layout)."""
        return dict(self.raw, compressor=None, filters=None, order="C")
