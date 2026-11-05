"""Zarr array metadata: NumPy's view of what the native reader parsed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .. import _reader


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
    _native: Any = field(default=None, compare=False, repr=False)

    @classmethod
    def from_native(cls, native) -> ArrayMeta:
        dtype = np.dtype(native.dtype)
        return cls(tuple(native.shape), tuple(native.chunks), dtype, np.frombuffer(native.fill, dtype=dtype)[0].item(),
                   native.order, native.separator, native.compressor, tuple(native.filters or ()), native.raw, native)

    @classmethod
    def from_zarray(cls, d: Mapping[str, Any]) -> ArrayMeta:
        return cls.from_native(_reader.parse_array_meta(dict(d)))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def chunk_nbytes(self) -> int:
        return self._native.chunk_nbytes

    @property
    def codec_id(self) -> str:
        """The native reader's id for this array's compressor; ValueError if it cannot read the chunks."""
        return self._native.codec()

    def decoded_zarray(self) -> dict[str, Any]:
        """``.zarray`` of the same array stored uncompressed (the decoded-cache layout)."""
        return self._native.decoded_zarray()
