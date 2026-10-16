"""A process-local native cache whose shared budget survives worker startup."""

from __future__ import annotations

import operator
import os

from .. import _native

# Never destroy a cache inherited while another thread might have held its lock.
_INHERITED: list[object] = []


class MemoryCache:
    """One decoded-chunk budget shared by readers, independently in each process.

    Pickling preserves sharing through Python's normal object memo. Forked
    workers lazily replace the native cache once, without touching its locks.
    Contents are never copied between processes.
    """

    def __init__(self, max_bytes: int):
        self._max_bytes = operator.index(max_bytes)
        if self._max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self._pid = os.getpid()
        self._cache = _native.MemoryCache(self._max_bytes)

    @property
    def _native(self):
        if self._pid != os.getpid():
            _INHERITED.append(self._cache)
            self._cache = _native.MemoryCache(self._max_bytes)
            self._pid = os.getpid()
        return self._cache

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def nbytes(self) -> int:
        return self._native.nbytes

    @property
    def hits(self) -> int:
        return self._native.hits

    @property
    def misses(self) -> int:
        return self._native.misses

    def __len__(self) -> int:
        return len(self._native)

    def clear(self) -> None:
        self._native.clear()

    def __getstate__(self):
        return self._max_bytes

    def __setstate__(self, max_bytes):
        self.__init__(max_bytes)
