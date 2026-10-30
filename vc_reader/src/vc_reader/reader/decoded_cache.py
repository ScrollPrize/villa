"""Explicitly shared, process-local image cache and decode working budget."""

from collections import OrderedDict
from contextlib import contextmanager
import operator
import os
import threading

DEFAULT_DECODE_BYTES = 256 << 20


def _budget(value, name):
    value = operator.index(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


class DecodedCache:
    """One LRU and decode budget shared by all readers referencing this object.

    Decode operations on an owner are serialized, including conversion scratch.
    Independent owners may run concurrently. Fork and pickle start empty caches
    with fresh locks. Returned crops never alias these cached arrays.
    """

    def __init__(self, max_bytes=DEFAULT_DECODE_BYTES, max_decode_bytes=DEFAULT_DECODE_BYTES):
        self._max_bytes = _budget(max_bytes, "max_bytes")
        self._max_decode_bytes = _budget(max_decode_bytes, "max_decode_bytes")
        self._reset()

    def _reset(self):
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._entries = OrderedDict()
        self._nbytes = self._hits = self._misses = self._peak_working = self._working = 0

    def _process(self):
        if self._pid != os.getpid():
            self._reset()

    @property
    def max_bytes(self):
        return self._max_bytes

    @property
    def max_decode_bytes(self):
        return self._max_decode_bytes

    def stats(self):
        self._process()
        with self._lock:
            return dict(nbytes=self._nbytes, entries=len(self._entries), hits=self._hits,
                        misses=self._misses, peak_working_bytes=self._peak_working)

    def clear(self):
        self._process()
        with self._lock:
            self._entries.clear()
            self._nbytes = 0

    @contextmanager
    def use(self, key, *, decode_bytes, scratch_bytes, load):
        """Borrow an immutable decoded object while reserving its working set.

        ``decode_bytes`` includes decoder output and estimated codec workspace;
        ``scratch_bytes`` includes all conversion buffers needed by the caller.
        Caller-owned final outputs and retained cache entries are separate.
        """
        self._process()
        with self._lock:
            array = self._entries.get(key)
            working = scratch_bytes + (decode_bytes if array is None else 0)
            if working + self._working > self.max_decode_bytes:
                raise MemoryError(
                    f"decode working set needs {working} bytes; max_decode_bytes="
                    f"{self.max_decode_bytes}. Increase the limit or use a chunked source."
                )
            self._peak_working = max(self._peak_working, working + self._working)
            if array is None:
                self._misses += 1
                array = load()
                if array.nbytes > decode_bytes:
                    raise ValueError("decoder returned more bytes than its declared working set")
                array.flags.writeable = False
                if 0 < array.nbytes <= self.max_bytes:
                    while self._entries and self._nbytes + array.nbytes > self.max_bytes:
                        _, evicted = self._entries.popitem(last=False)
                        self._nbytes -= evicted.nbytes
                    self._entries[key] = array
                    self._nbytes += array.nbytes
            else:
                self._hits += 1
                self._entries.move_to_end(key)
            self._working += working
            try:
                yield array
            finally:
                self._working -= working

    @contextmanager
    def working(self, nbytes):
        """Reserve scratch for a memory-mapped read without retaining a page."""
        self._process()
        with self._lock:
            if nbytes + self._working > self.max_decode_bytes:
                raise MemoryError(f"read scratch needs {nbytes} bytes; max_decode_bytes="
                                  f"{self.max_decode_bytes}. Increase the limit or read a smaller region.")
            self._peak_working = max(self._peak_working, nbytes + self._working)
            self._working += nbytes
            try:
                yield
            finally:
                self._working -= nbytes

    def __getstate__(self):
        return self.max_bytes, self.max_decode_bytes

    def __setstate__(self, state):
        self.__init__(*state)
