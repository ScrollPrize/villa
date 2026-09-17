"""Reusable chunk-local hot/spill storage for demand inference accumulators."""

from __future__ import annotations

from collections import OrderedDict
import heapq
import mmap
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


class ChunkAccumulatorArena:
    """Owner-local fixed RAM cache with anonymous regular-file spill segments.

    This class is intentionally unaware of volume coordinates and accumulator
    dtypes.  Every key owns one opaque, fixed-size record.  Callers serialize
    access (the demand accumulator owner queue does this in production).
    """

    def __init__(self, scratch_dir, *, record_bytes: int, hot_slots: int,
                 spill_segment_slots: int = 256, hot_buffer: np.ndarray | None = None,
                 spill_fd: int | None = None, spill_slot_acquire=None,
                 spill_slot_release=None, spill_segment_stride: int | None = None):
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.record_bytes = int(record_bytes)
        self.hot_slots = int(hot_slots)
        self.spill_segment_slots = int(spill_segment_slots)
        if self.record_bytes < 1 or self.hot_slots < 1 or self.spill_segment_slots < 1:
            raise ValueError("positive record_bytes, hot_slots and spill_segment_slots required")
        self._hot_mapping = None
        self._hot_stride = self.record_bytes
        if hot_buffer is None:
            # Anonymous mappings start as logical zero pages without committing
            # physical RAM.  Keep records page-aligned so Linux can return one
            # completed/evicted record with MADV_DONTNEED; its next use again
            # observes zeros without a full-record memset or retained RSS.
            page = int(mmap.PAGESIZE)
            self._hot_stride = ((self.record_bytes + page - 1) // page) * page
            size = self.hot_slots * self._hot_stride
            if os.name == "posix":
                anonymous = getattr(mmap, "MAP_ANONYMOUS", None)
                if anonymous is None:  # pragma: no cover - legacy POSIX spelling
                    anonymous = mmap.MAP_ANON
                self._hot_mapping = mmap.mmap(
                    -1, size, flags=mmap.MAP_PRIVATE | anonymous,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE)
            else:  # pragma: no cover - exercised by platform CI
                self._hot_mapping = mmap.mmap(-1, size, access=mmap.ACCESS_WRITE)
            self._hot = np.ndarray(
                (self.hot_slots, self.record_bytes), dtype=np.uint8,
                buffer=self._hot_mapping, strides=(self._hot_stride, 1))
            self._hot_zero = [True] * self.hot_slots
        else:
            if hot_buffer.dtype != np.uint8 or hot_buffer.shape != (self.hot_slots, self.record_bytes):
                raise ValueError("hot_buffer has incompatible shape or dtype")
            self._hot = hot_buffer
            self._hot_zero = [False] * self.hot_slots
        self._hot_free = list(range(self.hot_slots))
        heapq.heapify(self._hot_free)
        self._spill_free: list[int] = []
        self._segments: dict[int, mmap.mmap] = {}
        self._spill_fd = spill_fd
        self._owns_spill_fd = spill_fd is None
        self._spill_slot_acquire = spill_slot_acquire
        self._spill_slot_release = spill_slot_release
        self._external_spill = spill_slot_acquire is not None
        if self._external_spill != (spill_slot_release is not None) or (
                self._external_spill and spill_fd is None):
            raise ValueError("external spill requires fd plus acquire/release callbacks")
        minimum_stride = self.record_bytes * self.spill_segment_slots
        self._spill_segment_stride = int(spill_segment_stride or minimum_stride)
        if self._spill_segment_stride < minimum_stride:
            raise ValueError("spill segment stride is smaller than its records")
        self._segment_live: dict[int, int] = {}
        self._local_spill_capacity = 0
        self._locations: dict[object, tuple[str, int]] = {}
        self._lru: OrderedDict[object, None] = OrderedDict()
        self.stats = {
            "hot_high_water": 0, "spill_high_water": 0, "spill_capacity": 0,
            "evictions": 0, "promotions": 0, "spill_bytes_written": 0,
            "spill_bytes_read": 0, "allocations": 0, "frees": 0,
            "hot_discard_bytes": 0,
        }

    @property
    def live(self) -> int:
        return len(self._locations)

    @property
    def hot_occupancy(self) -> int:
        return self.hot_slots - len(self._hot_free)

    @property
    def spill_occupancy(self) -> int:
        return sum(self._segment_live.values())

    def _open_local_spill(self) -> None:
        if self._spill_fd is not None:
            return
        fd = None
        if hasattr(os, "O_TMPFILE"):
            try:
                fd = os.open(self.scratch_dir, os.O_RDWR | os.O_TMPFILE, 0o600)
            except OSError:
                fd = None
        if fd is None:
            fd, name = tempfile.mkstemp(prefix="arena-", suffix=".mmap", dir=self.scratch_dir)
            os.unlink(name)
        self._spill_fd = fd

    def _create_segment(self) -> None:
        self._open_local_spill()
        segment = len(self._segments)
        end = (segment + 1) * self._spill_segment_stride
        os.ftruncate(self._spill_fd, end)
        mapping = mmap.mmap(self._spill_fd, self._spill_segment_stride,
                            access=mmap.ACCESS_WRITE,
                            offset=segment * self._spill_segment_stride)
        self._segments[segment] = mapping
        self._segment_live[segment] = 0
        start = self._local_spill_capacity
        self._local_spill_capacity += self.spill_segment_slots
        self.stats["spill_capacity"] = self._local_spill_capacity
        self._spill_free.extend(range(start, start + self.spill_segment_slots))
        heapq.heapify(self._spill_free)

    def _ensure_external_segment(self, slot: int) -> int:
        segment = int(slot) // self.spill_segment_slots
        if segment not in self._segments:
            self._segments[segment] = mmap.mmap(
                self._spill_fd, self._spill_segment_stride, access=mmap.ACCESS_WRITE,
                offset=segment * self._spill_segment_stride)
            self._segment_live[segment] = 0
        return segment

    def _spill_view(self, slot: int) -> memoryview:
        segment, local = divmod(int(slot), self.spill_segment_slots)
        if segment not in self._segments:
            raise ValueError("invalid spill slot")
        start = local * self.record_bytes
        return memoryview(self._segments[segment])[start:start + self.record_bytes]

    def _allocate_spill(self) -> int:
        if self._external_spill:
            slot = int(self._spill_slot_acquire())
            segment = self._ensure_external_segment(slot)
            self.stats["spill_capacity"] = max(
                self.stats["spill_capacity"], self.spill_occupancy + 1)
        else:
            if not self._spill_free:
                self._create_segment()
            slot = heapq.heappop(self._spill_free)
            segment = slot // self.spill_segment_slots
        self._segment_live[segment] += 1
        self.stats["spill_high_water"] = max(self.stats["spill_high_water"], self.spill_occupancy)
        return slot

    def _release_spill(self, slot: int) -> None:
        segment = int(slot) // self.spill_segment_slots
        self._segment_live[segment] -= 1
        if self._external_spill:
            self._spill_slot_release(int(slot))
            if not self._segment_live[segment]:
                self._segments.pop(segment).close()
                del self._segment_live[segment]
        else:
            heapq.heappush(self._spill_free, int(slot))

    def _touch_hot(self, key) -> None:
        self._lru.pop(key, None)
        self._lru[key] = None

    def _evict_one(self) -> int:
        if not self._lru:
            raise RuntimeError("hot arena has no evictable record")
        key, _ = self._lru.popitem(last=False)
        kind, hot_slot = self._locations[key]
        if kind != "hot":
            raise RuntimeError("arena LRU contains a non-hot record")
        spill_slot = self._allocate_spill()
        view = self._spill_view(spill_slot)
        view[:] = self._hot[hot_slot]
        view.release()
        self._locations[key] = ("spill", spill_slot)
        self.stats["evictions"] += 1
        self.stats["spill_bytes_written"] += self.record_bytes
        return hot_slot

    def _acquire_hot_slot(self) -> int:
        return heapq.heappop(self._hot_free) if self._hot_free else self._evict_one()

    def _zero_hot_slot(self, slot: int) -> None:
        if self._hot_zero[slot]:
            return
        if (self._hot_mapping is not None and sys.platform.startswith("linux")
                and hasattr(self._hot_mapping, "madvise")
                and hasattr(mmap, "MADV_DONTNEED")):
            self._hot_mapping.madvise(
                mmap.MADV_DONTNEED, slot * self._hot_stride, self._hot_stride)
            self.stats["hot_discard_bytes"] += self._hot_stride
        else:
            self._hot[slot].fill(0)
        self._hot_zero[slot] = True

    def mutable_record(self, key, *, create: bool = False) -> np.ndarray:
        location = self._locations.get(key)
        if location is None:
            if not create:
                raise KeyError(key)
            slot = self._acquire_hot_slot()
            self._zero_hot_slot(slot)
            self._hot_zero[slot] = False
            self._locations[key] = ("hot", slot)
            self._touch_hot(key)
            self.stats["allocations"] += 1
            self.stats["hot_high_water"] = max(self.stats["hot_high_water"], self.hot_occupancy)
            return self._hot[slot]
        kind, slot = location
        if kind == "hot":
            self._touch_hot(key)
            return self._hot[slot]
        hot_slot = self._acquire_hot_slot()
        view = self._spill_view(slot)
        self._hot[hot_slot] = np.frombuffer(view, dtype=np.uint8, count=self.record_bytes)
        view.release()
        self._release_spill(slot)
        self._locations[key] = ("hot", hot_slot)
        self._touch_hot(key)
        self._hot_zero[hot_slot] = False
        self.stats["promotions"] += 1
        self.stats["spill_bytes_read"] += self.record_bytes
        return self._hot[hot_slot]

    def snapshot_and_release(self, key, destination: np.ndarray) -> str:
        if destination.dtype != np.uint8 or destination.size != self.record_bytes:
            raise ValueError("publication destination has incompatible record size")
        kind, slot = self._locations.pop(key)
        if kind == "hot":
            destination.reshape(-1)[:] = self._hot[slot]
            self._lru.pop(key, None)
            self._zero_hot_slot(slot)
            heapq.heappush(self._hot_free, slot)
        else:
            view = self._spill_view(slot)
            destination.reshape(-1)[:] = np.frombuffer(
                view, dtype=np.uint8, count=self.record_bytes)
            view.release()
            self._release_spill(slot)
            self.stats["spill_bytes_read"] += self.record_bytes
        self.stats["frees"] += 1
        return kind

    def telemetry(self) -> dict[str, int]:
        return {**self.stats, "hot_occupancy": self.hot_occupancy,
                "spill_occupancy": self.spill_occupancy, "live": self.live}

    def close(self) -> None:
        if self._external_spill:
            for kind, slot in self._locations.values():
                if kind == "spill":
                    self._spill_slot_release(int(slot))
            self._locations.clear()
        for mapping in self._segments.values():
            mapping.close()
        self._segments.clear()
        self._segment_live.clear()
        if self._owns_spill_fd and self._spill_fd is not None:
            os.close(self._spill_fd)
        self._spill_fd = None
        if self._hot_mapping is not None:
            # Drop the ndarray export before closing its backing mapping.
            self._hot = None
            self._hot_mapping.close()
            self._hot_mapping = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
