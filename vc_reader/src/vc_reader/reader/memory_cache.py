"""A process-local C++ cache whose shared budget survives worker startup."""

from __future__ import annotations

import operator
import os

from .. import _reader

# Never destroy a cache inherited while another thread might have held its lock.
_INHERITED: list[object] = []


class MemoryCache:
    """One decoded-chunk budget shared by readers, independently in each process.

    Pickling preserves sharing through Python's normal object memo. Forked
    workers lazily replace the C++ cache once, without touching its locks.
    Contents are never copied between processes.
    """

    def __init__(self, max_bytes: int | str, *, fraction: float = 0.5, smoothing_seconds: float = 10):
        self._max_bytes = "auto" if max_bytes == "auto" else operator.index(max_bytes)
        if self._max_bytes != "auto" and self._max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self._fraction = fraction
        self._smoothing_seconds = smoothing_seconds
        self._pid = os.getpid()
        self._cache = self._new_engine()

    def _new_engine(self):
        if self._max_bytes == "auto":
            return _reader.MemoryCache.automatic(self._fraction, self._smoothing_seconds)
        return _reader.MemoryCache(self._max_bytes)

    @property
    def _engine(self):
        if self._pid != os.getpid():
            _INHERITED.append(self._cache)
            self._cache = self._new_engine()
            self._pid = os.getpid()
        return self._cache

    @property
    def max_bytes(self) -> int:
        return self._engine.max_bytes

    @property
    def is_auto(self) -> bool:
        return self._engine.is_auto

    @property
    def fraction(self) -> float:
        return self._engine.fraction

    @property
    def smoothing_seconds(self) -> float:
        return self._engine.smoothing_seconds

    def refresh_budget(self) -> None:
        self._engine.refresh_budget()

    @property
    def nbytes(self) -> int:
        return self._engine.nbytes

    @property
    def hits(self) -> int:
        return self._engine.hits

    @property
    def misses(self) -> int:
        return self._engine.misses

    def __len__(self) -> int:
        return len(self._engine)

    def clear(self) -> None:
        self._engine.clear()

    def __getstate__(self):
        return self._max_bytes, self._fraction, self._smoothing_seconds

    def __setstate__(self, state):
        self.__init__(state[0], fraction=state[1], smoothing_seconds=state[2])
