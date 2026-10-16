"""In-memory LRU cache of decoded chunks, bounded by bytes, shareable between readers."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Hashable

import numpy as np

_MISSING_COST = 64  # bookkeeping cost charged for a cached "chunk is not stored"
_ABSENT = object()


class MemoryChunkCache:
    """Thread-safe LRU of decoded chunks.

    Values are read-only ndarrays, or None for chunks known to be missing.
    Keys are ``(source identity, chunk index)`` so one cache can serve many
    arrays. ``max_bytes=0`` disables caching.
    """

    def __init__(self, max_bytes: int = 2 << 30):
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[Hashable, np.ndarray | None] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __getstate__(self):
        # Worker processes start with an empty cache of the same size.
        return {"max_bytes": self.max_bytes}

    def __setstate__(self, state):
        self.__init__(state["max_bytes"])

    def __len__(self) -> int:
        return len(self._items)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def get(self, key: Hashable):
        """The cached value, or the module sentinel ``ABSENT`` on a miss."""
        with self._lock:
            value = self._items.get(key, _ABSENT)
            if value is _ABSENT:
                self.misses += 1
                return _ABSENT
            self._items.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: Hashable, value: np.ndarray | None) -> None:
        if self.max_bytes <= 0:
            return
        cost = _MISSING_COST if value is None else value.nbytes
        if cost > self.max_bytes:
            return
        if value is not None and value.flags.writeable:
            value.flags.writeable = False
        with self._lock:
            old = self._items.pop(key, _ABSENT)
            if old is not _ABSENT:
                self._bytes -= _MISSING_COST if old is None else old.nbytes
            self._items[key] = value
            self._bytes += cost
            while self._bytes > self.max_bytes:
                _, evicted = self._items.popitem(last=False)
                self._bytes -= _MISSING_COST if evicted is None else evicted.nbytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0


ABSENT = _ABSENT
