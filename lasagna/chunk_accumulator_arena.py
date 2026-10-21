"""Bounded hot-RAM storage with reusable direct-I/O spill records."""

from __future__ import annotations

from collections import OrderedDict
import heapq
import errno
import math
import mmap
import os
from pathlib import Path
import sys
import tempfile
import threading

import numpy as np


class ChunkAccumulatorArena:
    """Owner-local fixed-size records with asynchronous LRU spill.

    The demand accumulator owner serializes public method calls, while cleanup
    threads evict unpinned records. Across 128 production owners, two cleanup
    threads per owner permit up to 256 writes in flight. Cold records are
    promoted before mutation, so each spill operation is one immutable,
    full-record write followed eventually by one full-record read.
    """

    def __init__(self, scratch_dir, *, record_bytes: int, hot_slots: int,
                 spill_segment_slots: int = 256,
                 hot_buffer: np.ndarray | None = None,
                 spill_slot_acquire=None, spill_slot_release=None,
                 spill_segment_stride: int | None = None,
                 background_eviction_workers: int = 2,
                 pressure_fraction: float = 0.90):
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.record_bytes = int(record_bytes)
        self.hot_slots = int(hot_slots)
        self.spill_segment_slots = int(spill_segment_slots)
        if self.record_bytes < 1 or self.hot_slots < 1 or self.spill_segment_slots < 1:
            raise ValueError("positive record_bytes, hot_slots and spill_segment_slots required")
        if not (0.5 <= float(pressure_fraction) < 1.0):
            raise ValueError("pressure_fraction must be in [0.5, 1.0)")
        if int(background_eviction_workers) < 1:
            raise ValueError("at least one background eviction worker required")

        page = int(mmap.PAGESIZE)
        self._hot_stride = ((self.record_bytes + page - 1) // page) * page
        self._hot_mapping = None
        if hot_buffer is None:
            size = self.hot_slots * self._hot_stride
            if os.name == "posix":
                anonymous = getattr(mmap, "MAP_ANONYMOUS", getattr(mmap, "MAP_ANON", None))
                self._hot_mapping = mmap.mmap(
                    -1, size, flags=mmap.MAP_PRIVATE | anonymous,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE)
            else:  # pragma: no cover - platform CI
                self._hot_mapping = mmap.mmap(-1, size, access=mmap.ACCESS_WRITE)
            self._hot = np.ndarray(
                (self.hot_slots, self.record_bytes), np.uint8,
                buffer=self._hot_mapping, strides=(self._hot_stride, 1))
            self._hot_zero = [True] * self.hot_slots
        else:
            if hot_buffer.dtype != np.uint8 or hot_buffer.shape != (self.hot_slots, self.record_bytes):
                raise ValueError("hot_buffer has incompatible shape or dtype")
            self._hot = hot_buffer
            self._hot_stride = int(hot_buffer.strides[0])
            self._hot_zero = [False] * self.hot_slots

        minimum_stride = self.record_bytes * self.spill_segment_slots
        self._spill_segment_stride = int(spill_segment_stride or minimum_stride)
        if self._spill_segment_stride < minimum_stride:
            raise ValueError("spill segment stride is smaller than its records")
        self._spill_slot_acquire = spill_slot_acquire
        self._spill_slot_release = spill_slot_release
        self._external_spill = spill_slot_acquire is not None
        if self._external_spill != (spill_slot_release is not None):
            raise ValueError("external spill requires acquire and release callbacks")

        self._hot_free = list(range(self.hot_slots))
        heapq.heapify(self._hot_free)
        # location values: (state, slot, path_or_none, pins)
        # state is hot, spill, evicting, promoting, or publishing.
        self._locations: dict[object, tuple[str, int, str | None, int]] = {}
        self._lru: OrderedDict[object, None] = OrderedDict()
        self._condition = threading.Condition(threading.RLock())
        self._stop = False
        self._cleanup_error: BaseException | None = None
        self._pressure_slots = max(1, min(
            self.hot_slots - 1, int(math.ceil(self.hot_slots * float(pressure_fraction)))))
        self._target_free = max(1, self.hot_slots - self._pressure_slots + 1)

        self._spill_lock = threading.RLock()
        self._segment_fds: dict[str, tuple[int, bool]] = {}
        self._segment_live: dict[str, int] = {}
        self._local_paths: dict[int, str] = {}
        self._local_fds: dict[int, int] = {}
        self._local_named: list[Path] = []
        self._spill_free: list[int] = []
        self._local_next = 0
        self._io_alignment = max(page, 4096)

        self.stats = {
            "hot_high_water": 0, "spill_high_water": 0,
            "spill_capacity": 0, "evictions": 0, "promotions": 0,
            "spill_bytes_written": 0, "spill_bytes_read": 0,
            "allocations": 0, "frees": 0, "hot_discard_bytes": 0,
            "evictions_in_flight": 0, "evictions_in_flight_high_water": 0,
            "promotion_waits": 0, "allocation_waits": 0,
            "direct_segments": 0, "buffered_segments": 0,
            "cleanup_workers": int(background_eviction_workers),
            "pressure_slots": self._pressure_slots,
            "target_free_slots": self._target_free,
        }
        self._cleanup_threads = [threading.Thread(
            target=self._cleanup_loop, name=f"accumulator-evict-{i}", daemon=True)
            for i in range(int(background_eviction_workers))]
        for thread in self._cleanup_threads:
            thread.start()

    @property
    def live(self) -> int:
        with self._condition:
            return len(self._locations)

    @property
    def hot_occupancy(self) -> int:
        with self._condition:
            return self.hot_slots - len(self._hot_free)

    @property
    def spill_occupancy(self) -> int:
        with self._condition:
            return self._spill_occupancy_locked()

    def _spill_occupancy_locked(self) -> int:
        return sum(state in ("spill", "evicting", "promoting", "publishing")
                   for state, _, _, _ in self._locations.values())

    def _touch_hot_locked(self, key) -> None:
        self._lru.pop(key, None)
        self._lru[key] = None

    def _signal_pressure_locked(self) -> None:
        if self.hot_slots - len(self._hot_free) >= self._pressure_slots:
            self._condition.notify_all()

    def _find_evictable_locked(self):
        for key in tuple(self._lru):
            state, slot, _, pins = self._locations.get(key, (None, 0, None, 0))
            if state == "hot" and pins == 0:
                self._lru.pop(key, None)
                self._locations[key] = ("evicting", slot, None, 0)
                self.stats["evictions_in_flight"] += 1
                self.stats["evictions_in_flight_high_water"] = max(
                    self.stats["evictions_in_flight_high_water"],
                    self.stats["evictions_in_flight"])
                return key, slot
        return None

    def _cleanup_loop(self) -> None:
        while True:
            with self._condition:
                while True:
                    if self._stop:
                        return
                    occupied = self.hot_slots - len(self._hot_free)
                    if occupied >= self._pressure_slots:
                        candidate = self._find_evictable_locked()
                        if candidate is not None:
                            break
                    self._condition.wait()
            key, hot_slot = candidate
            spill = None
            try:
                spill = self._allocate_spill()
                self._write_spill(spill, self._hot[hot_slot])
                with self._condition:
                    current = self._locations.get(key)
                    if current is None or current[0] != "evicting" or current[1] != hot_slot:
                        raise RuntimeError("eviction state changed unexpectedly")
                    self._locations[key] = ("spill", spill[0], spill[1], 0)
                    self.stats["evictions"] += 1
                    self.stats["spill_bytes_written"] += self.record_bytes
                    self.stats["evictions_in_flight"] -= 1
                    self._zero_hot_slot_locked(hot_slot)
                    heapq.heappush(self._hot_free, hot_slot)
                    self.stats["spill_high_water"] = max(
                        self.stats["spill_high_water"], self._spill_occupancy_locked())
                    self._condition.notify_all()
            except BaseException as exc:
                if spill is not None:
                    self._release_spill(spill)
                with self._condition:
                    current = self._locations.get(key)
                    if current is not None and current[0] == "evicting":
                        self._locations[key] = ("hot", hot_slot, None, 0)
                        self._touch_hot_locked(key)
                    self.stats["evictions_in_flight"] -= 1
                    self._cleanup_error = exc
                    self._condition.notify_all()
                # Preserve the hot record rather than losing data. Foreground
                # allocation remains blocked, making the failure observable.
                return

    def _create_local_segment_locked(self, segment: int) -> str:
        fd = None
        path = None
        if hasattr(os, "O_TMPFILE"):
            try:
                fd = os.open(self.scratch_dir, os.O_RDWR | os.O_TMPFILE, 0o600)
                path = f"/proc/{os.getpid()}/fd/{fd}"
            except OSError:
                fd = None
        if fd is None:
            fd, name = tempfile.mkstemp(
                prefix="spill-segment-", suffix=".bin", dir=self.scratch_dir)
            path = name
            self._local_named.append(Path(name))
        os.ftruncate(fd, self._spill_segment_stride)
        if hasattr(os, "posix_fallocate"):
            try:
                os.posix_fallocate(fd, 0, self._spill_segment_stride)
            except OSError as exc:
                if exc.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                    raise
        self._local_fds[segment] = fd
        self._local_paths[segment] = path
        start = segment * self.spill_segment_slots
        self._spill_free.extend(range(start, start + self.spill_segment_slots))
        heapq.heapify(self._spill_free)
        self.stats["spill_capacity"] += self.spill_segment_slots
        return path

    def _allocate_spill(self) -> tuple[int, str]:
        if self._external_spill:
            slot, path = self._spill_slot_acquire()
            slot, path = int(slot), str(path)
        else:
            with self._spill_lock:
                if not self._spill_free:
                    segment = self._local_next
                    self._local_next += 1
                    self._create_local_segment_locked(segment)
                slot = heapq.heappop(self._spill_free)
                path = self._local_paths[slot // self.spill_segment_slots]
        with self._spill_lock:
            self._segment_live[path] = self._segment_live.get(path, 0) + 1
        return slot, path

    def _release_spill(self, spill: tuple[int, str]) -> None:
        slot, path = int(spill[0]), str(spill[1])
        with self._spill_lock:
            live = self._segment_live[path] - 1
            if live:
                self._segment_live[path] = live
            else:
                self._segment_live.pop(path, None)
                opened = self._segment_fds.pop(path, None)
                if opened is not None:
                    os.close(opened[0])
            if not self._external_spill:
                heapq.heappush(self._spill_free, slot)
        if self._external_spill:
            self._spill_slot_release(slot)

    def _open_segment(self, path: str) -> tuple[int, bool]:
        with self._spill_lock:
            opened = self._segment_fds.get(path)
            if opened is not None:
                return opened
            direct = bool(
                sys.platform.startswith("linux") and hasattr(os, "O_DIRECT")
                and self.record_bytes % self._io_alignment == 0)
            flags = os.O_RDWR | (os.O_DIRECT if direct else 0)
            try:
                fd = os.open(path, flags)
            except OSError:
                if not direct:
                    raise
                direct = False
                fd = os.open(path, os.O_RDWR)
            self._segment_fds[path] = (fd, direct)
            self.stats["direct_segments" if direct else "buffered_segments"] += 1
            return fd, direct

    def _spill_offset(self, slot: int) -> int:
        return (int(slot) % self.spill_segment_slots) * self.record_bytes

    @staticmethod
    def _write_exact(fd: int, view: memoryview, offset: int) -> None:
        written = os.pwrite(fd, view, offset)
        if written != len(view):
            raise OSError(f"short spill write: {written} of {len(view)} bytes")

    @staticmethod
    def _read_exact(fd: int, view: memoryview, offset: int) -> None:
        read = os.preadv(fd, [view], offset)
        if read != len(view):
            raise OSError(f"short spill read: {read} of {len(view)} bytes")

    def _write_spill(self, spill: tuple[int, str], source: np.ndarray) -> None:
        slot, path = spill
        fd, _ = self._open_segment(path)
        self._write_exact(fd, memoryview(source).cast("B"), self._spill_offset(slot))

    def _read_spill(self, spill: tuple[int, str], destination: np.ndarray) -> None:
        slot, path = spill
        fd, _ = self._open_segment(path)
        self._read_exact(fd, memoryview(destination).cast("B"), self._spill_offset(slot))

    def _zero_hot_slot_locked(self, slot: int) -> None:
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

    def _acquire_hot_locked(self, *, promotion: bool) -> int:
        while not self._hot_free:
            if self._cleanup_error is not None:
                raise RuntimeError("background accumulator eviction failed") from self._cleanup_error
            self.stats["promotion_waits" if promotion else "allocation_waits"] += 1
            self._condition.notify_all()
            self._condition.wait()
        slot = heapq.heappop(self._hot_free)
        self._signal_pressure_locked()
        return slot

    def mutable_record(self, key, *, create: bool = False) -> np.ndarray:
        with self._condition:
            while True:
                location = self._locations.get(key)
                if location is None:
                    if not create:
                        raise KeyError(key)
                    slot = self._acquire_hot_locked(promotion=False)
                    self._zero_hot_slot_locked(slot)
                    self._hot_zero[slot] = False
                    self._locations[key] = ("hot", slot, None, 1)
                    self._touch_hot_locked(key)
                    self.stats["allocations"] += 1
                    self.stats["hot_high_water"] = max(
                        self.stats["hot_high_water"], self.hot_slots - len(self._hot_free))
                    self._signal_pressure_locked()
                    return self._hot[slot]
                state, slot, path, pins = location
                if state == "hot":
                    self._locations[key] = (state, slot, path, pins + 1)
                    self._touch_hot_locked(key)
                    return self._hot[slot]
                if state == "spill":
                    hot_slot = self._acquire_hot_locked(promotion=True)
                    self._locations[key] = ("promoting", slot, path, 0)
                    break
                self._condition.wait()

        spill = (slot, path)
        try:
            self._read_spill(spill, self._hot[hot_slot])
        except BaseException:
            with self._condition:
                self._locations[key] = ("spill", slot, path, 0)
                heapq.heappush(self._hot_free, hot_slot)
                self._condition.notify_all()
            raise
        self._release_spill(spill)
        with self._condition:
            self._hot_zero[hot_slot] = False
            self._locations[key] = ("hot", hot_slot, None, 1)
            self._touch_hot_locked(key)
            self.stats["promotions"] += 1
            self.stats["spill_bytes_read"] += self.record_bytes
            self._signal_pressure_locked()
            return self._hot[hot_slot]

    def record_mutated(self, key) -> None:
        """Unpin a record after the caller finishes its in-place update."""
        with self._condition:
            state, slot, path, pins = self._locations[key]
            if state != "hot" or pins < 1:
                raise RuntimeError("record_mutated without a pinned hot record")
            self._locations[key] = (state, slot, path, pins - 1)
            self._touch_hot_locked(key)
            self._signal_pressure_locked()
            self._condition.notify_all()

    def snapshot_and_release(self, key, destination: np.ndarray) -> str:
        if destination.dtype != np.uint8 or destination.size != self.record_bytes:
            raise ValueError("publication destination has incompatible record size")
        with self._condition:
            while True:
                state, slot, path, pins = self._locations[key]
                if state == "hot" and pins == 0:
                    self._locations[key] = ("publishing", slot, None, 0)
                    self._lru.pop(key, None)
                    source_kind = "hot"
                    break
                if state == "spill":
                    self._locations[key] = ("publishing", slot, path, 0)
                    source_kind = "spill"
                    break
                self._condition.wait()

        if source_kind == "hot":
            destination.reshape(-1)[:] = self._hot[slot]
            with self._condition:
                self._locations.pop(key)
                self._zero_hot_slot_locked(slot)
                heapq.heappush(self._hot_free, slot)
                self.stats["frees"] += 1
                self._condition.notify_all()
        else:
            spill = (slot, path)
            self._read_spill(spill, destination.reshape(-1))
            self._release_spill(spill)
            with self._condition:
                self._locations.pop(key)
                self.stats["spill_bytes_read"] += self.record_bytes
                self.stats["frees"] += 1
                self._condition.notify_all()
        return source_kind

    def telemetry(self) -> dict[str, int]:
        with self._condition:
            return {**self.stats,
                    "hot_occupancy": self.hot_slots - len(self._hot_free),
                    "hot_free": len(self._hot_free),
                    "spill_occupancy": self._spill_occupancy_locked(),
                    "live": len(self._locations)}

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        for thread in self._cleanup_threads:
            thread.join()
        releases = []
        with self._condition:
            for state, slot, path, _ in self._locations.values():
                if state == "spill":
                    releases.append((slot, path))
            self._locations.clear()
            self._lru.clear()
        for spill in releases:
            self._release_spill(spill)
        with self._spill_lock:
            for fd, _ in self._segment_fds.values():
                os.close(fd)
            self._segment_fds.clear()
            for fd in self._local_fds.values():
                os.close(fd)
            self._local_fds.clear()
            for path in self._local_named:
                path.unlink(missing_ok=True)
            self._local_named.clear()
        if self._hot_mapping is not None:
            self._hot = None
            self._hot_mapping.close()
            self._hot_mapping = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
