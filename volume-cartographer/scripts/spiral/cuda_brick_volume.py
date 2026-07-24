"""Bounded, lazy CUDA residency for exact uint8 Zarr volume bricks.

The cache is intentionally independent of SDT/Lasagna encoding and sampling
semantics.  Callers provide one or more aligned three-dimensional uint8 arrays
and gather ROI-local integer coordinates from whichever channels they need.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import torch


class _PreparedGather:
    """One-shot resident gather whose slots stay pinned until consumption."""

    def __init__(
            self, store, flat, slots, local, original_shape, pinned_slots,
            ready_event):
        self._store = store
        self.flat = flat
        self.slots = slots
        self.local = local
        self.original_shape = original_shape
        self.pinned_slots = pinned_slots
        self.ready_event = ready_event
        self.consumed = False

    def release(self):
        """Release an unused request without launching its gather."""
        self._store._release_prepared(self)

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class CudaBrickVolume:
    """Lazy fixed-size brick cache shared by aligned uint8 volume channels.

    ``indices_zyx`` passed to :meth:`gather` are relative to the requested
    z-ROI; ``z_origin`` maps them back to the source arrays.  Slot zero is a
    permanent no-data brick.  All other slots are bounded by ``capacity_bytes``
    and evicted by least-recently-used epoch when necessary.
    """

    def __init__(
            self, arrays, *, z_origin, roi_shape, capacity_bytes,
            brick_size=64, no_data_value=0, device='cuda', workers=None):
        self.arrays = tuple(arrays)
        if not self.arrays:
            raise ValueError('CudaBrickVolume requires at least one channel')
        self.channel_count = len(self.arrays)
        self.source_shape = tuple(int(v) for v in self.arrays[0].shape)
        self.source_chunks = tuple(int(v) for v in self.arrays[0].chunks)
        self.roi_shape = tuple(int(v) for v in roi_shape)
        self.z_origin = int(z_origin)
        self.brick_size = int(brick_size)
        self.no_data_value = int(no_data_value)
        self.device = torch.device(device)
        if self.brick_size <= 0:
            raise ValueError('brick_size must be positive')
        if len(self.source_shape) != 3 or len(self.source_chunks) != 3:
            raise ValueError('CudaBrickVolume supports three-dimensional arrays')
        for array in self.arrays:
            if tuple(array.shape) != self.source_shape:
                raise ValueError('all brick-volume channels must have the same shape')
            if tuple(array.chunks) != self.source_chunks:
                raise ValueError('all brick-volume channels must have the same chunks')
            if np.dtype(array.dtype) != np.dtype(np.uint8):
                raise ValueError('CudaBrickVolume currently supports uint8 channels')
        if any(
                max(chunk, self.brick_size) % min(chunk, self.brick_size)
                for chunk in self.source_chunks):
            raise ValueError(
                f'source chunks {self.source_chunks} and brick size '
                f'{self.brick_size} must be integer multiples on every axis')
        # Read the smallest source-aligned region which contains whole cache
        # bricks.  A 128^3 SDT chunk is read once and split into eight 64^3
        # bricks; a 64^3 Lasagna brick is assembled by Zarr from its eight
        # underlying 32^3 chunks.
        self.read_shape = tuple(
            max(chunk, self.brick_size) for chunk in self.source_chunks)

        self.brick_grid_shape = tuple(
            -(-size // self.brick_size) for size in self.source_shape)
        self.brick_count = int(np.prod(self.brick_grid_shape, dtype=np.int64))
        self.brick_voxels = self.brick_size ** 3
        bytes_per_slot = self.channel_count * self.brick_voxels
        self.slot_count = int(capacity_bytes) // bytes_per_slot
        if self.slot_count < 2:
            raise ValueError(
                f'capacity_bytes must hold a no-data brick and at least one '
                f'{self.channel_count}-channel {self.brick_size}^3 brick')

        self.pool = torch.empty(
            (self.channel_count, self.slot_count, self.brick_voxels),
            dtype=torch.uint8, device=self.device)
        self.pool[:, 0].fill_(self.no_data_value)
        self._table_cpu = np.full(self.brick_count, -1, dtype=np.int32)
        self.table = torch.full(
            (self.brick_count,), -1, dtype=torch.int32, device=self.device)
        self._slot_to_brick = np.full(self.slot_count, -1, dtype=np.int64)
        self._last_used = np.zeros(self.slot_count, dtype=np.int64)
        self._slot_pins = np.zeros(self.slot_count, dtype=np.int32)
        self._slot_readers = [[] for _index in range(self.slot_count)]
        self._free_slots = list(range(self.slot_count - 1, 0, -1))
        self._epoch = 0
        cpu_count = max(1, int(__import__('os').cpu_count() or 1))
        self.worker_count = max(
            1, min(int(workers or min(16, cpu_count)), cpu_count, 32))
        self._pool = ThreadPoolExecutor(
            max_workers=self.worker_count, thread_name_prefix='cuda-bricks')
        self._lock = threading.RLock()
        self._last_timings = {}
        self.total_loaded_bricks = 0
        self.total_evicted_bricks = 0
        self._table_ready_event = self._record_event()
        self._pending_timing = None

    @property
    def resident_bricks(self):
        return int((self._slot_to_brick >= 0).sum())

    @property
    def resident_bytes(self):
        return self.resident_bricks * self.channel_count * self.brick_voxels

    @property
    def capacity(self):
        return self.slot_count - 1

    @property
    def last_timings(self):
        with self._lock:
            self._collect_timing()
            return self._last_timings

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)

    def _current_stream(self):
        if self.device.type != 'cuda':
            return None
        return torch.cuda.current_stream(self.device)

    def _record_event(self, *, timing=False):
        stream = self._current_stream()
        if stream is None:
            return None
        event = torch.cuda.Event(enable_timing=timing)
        event.record(stream)
        return event

    def _wait_for_table(self):
        stream = self._current_stream()
        if stream is not None and self._table_ready_event is not None:
            stream.wait_event(self._table_ready_event)

    def _wait_for_slot_readers(self, slots):
        stream = self._current_stream()
        if stream is None:
            return
        for slot in np.asarray(slots, dtype=np.int64):
            readers = self._slot_readers[int(slot)]
            for event in readers:
                stream.wait_event(event)
            readers.clear()

    def _record_slot_readers(self, slots, event):
        if event is None:
            return
        for slot in np.asarray(slots, dtype=np.int64):
            readers = self._slot_readers[int(slot)]
            readers[:] = [reader for reader in readers if not reader.query()]
            readers.append(event)

    def _collect_timing(self):
        pending = self._pending_timing
        if pending is None:
            return
        start, end = pending
        if end.query():
            self._last_timings['gather_seconds'] = \
                start.elapsed_time(end) / 1000.0
            self._pending_timing = None

    def _decode_brick_ids(self, brick_ids):
        nz, ny, nx = self.brick_grid_shape
        bx = brick_ids % nx
        quotient = brick_ids // nx
        by = quotient % ny
        bz = quotient // ny
        return bz, by, bx

    def _brick_ids(self, flat_indices):
        global_z = flat_indices[:, 0] + self.z_origin
        nz, ny, nx = self.brick_grid_shape
        del nz
        return ((global_z // self.brick_size) * ny
                + flat_indices[:, 1] // self.brick_size) * nx \
            + flat_indices[:, 2] // self.brick_size

    def _read_parent(self, item):
        parent, children = item
        starts = tuple(
            parent[axis] * self.read_shape[axis] for axis in range(3))
        stops = tuple(
            min(starts[axis] + self.read_shape[axis],
                self.source_shape[axis]) for axis in range(3))
        selection = tuple(slice(starts[axis], stops[axis]) for axis in range(3))
        channel_parents = []
        for array in self.arrays:
            data = np.asarray(array[selection], dtype=np.uint8)
            padding = tuple(
                (0, self.read_shape[axis] - data.shape[axis])
                for axis in range(3))
            if any(after for _before, after in padding):
                data = np.pad(
                    data, padding, constant_values=self.no_data_value)
            channel_parents.append(data)

        bricks = np.empty(
            (self.channel_count, len(children), self.brick_voxels),
            dtype=np.uint8)
        brick_ids = np.empty(len(children), dtype=np.int64)
        for child_index, (brick_id, local) in enumerate(children):
            slices = tuple(
                slice(local[axis] * self.brick_size,
                      (local[axis] + 1) * self.brick_size)
                for axis in range(3))
            brick_ids[child_index] = brick_id
            for channel, parent_data in enumerate(channel_parents):
                bricks[channel, child_index] = np.ascontiguousarray(
                    parent_data[slices]).reshape(-1)
        return brick_ids, bricks

    def _reserve_slots(self, count, pinned_slots):
        free_count = min(count, len(self._free_slots))
        slots = [
            self._free_slots.pop() for _index in range(free_count)]
        remaining = count - free_count
        if remaining:
            candidates = np.flatnonzero(
                (self._slot_to_brick >= 0) & (self._slot_pins == 0))
            if pinned_slots:
                pinned = np.fromiter(pinned_slots, dtype=np.int64)
                candidates = candidates[~np.isin(candidates, pinned)]
            if len(candidates) < remaining:
                raise RuntimeError(
                    'one gather requires more nonzero bricks than the '
                    f'configured CUDA cache capacity ({self.capacity} bricks)')
            if remaining < len(candidates):
                selected = np.argpartition(
                    self._last_used[candidates], remaining - 1)[:remaining]
                victims = candidates[selected]
            else:
                victims = candidates
            # Queue reuse after every outstanding reader of these slots.  This
            # is a stream dependency, not a host or device-wide synchronize.
            self._wait_for_slot_readers(victims)
            evicted = self._slot_to_brick[victims].copy()
            self._table_cpu[evicted] = -1
            self.table[torch.from_numpy(evicted).to(self.device)] = -1
            self._slot_to_brick[victims] = -1
            self.total_evicted_bricks += len(victims)
            slots.extend(victims.tolist())
        return np.asarray(slots, dtype=np.int64)

    def _load_missing(self, missing_ids, requested_ids):
        bz, by, bx = self._decode_brick_ids(missing_ids)
        ratios = tuple(
            size // self.brick_size for size in self.read_shape)
        by_parent = defaultdict(list)
        for brick_id, z, y, x in zip(
                missing_ids.tolist(), bz.tolist(), by.tolist(), bx.tolist()):
            parent = (z // ratios[0], y // ratios[1], x // ratios[2])
            local = (z % ratios[0], y % ratios[1], x % ratios[2])
            by_parent[parent].append((int(brick_id), local))

        current_slots = self._table_cpu[requested_ids]
        pinned_slots = set(
            int(slot) for slot in current_slots if int(slot) > 0)
        loaded = zero_bricks = 0
        items = sorted(by_parent.items())
        for batch_start in range(0, len(items), self.worker_count * 2):
            results = list(self._pool.map(
                self._read_parent,
                items[batch_start:batch_start + self.worker_count * 2]))
            brick_ids = np.concatenate([result[0] for result in results])
            bricks = np.concatenate([result[1] for result in results], axis=1)
            nonzero = np.any(bricks != self.no_data_value, axis=(0, 2))
            zero_ids = brick_ids[~nonzero]
            if len(zero_ids):
                self._table_cpu[zero_ids] = 0
                self.table[torch.from_numpy(zero_ids).to(self.device)] = 0
                zero_bricks += len(zero_ids)
            live_ids = brick_ids[nonzero]
            if not len(live_ids):
                continue
            slots = self._reserve_slots(len(live_ids), pinned_slots)
            pinned_slots.update(slots.tolist())
            live_bricks = np.ascontiguousarray(bricks[:, nonzero])
            slot_device = torch.from_numpy(slots).to(self.device)
            self.pool.index_copy_(
                1, slot_device, torch.from_numpy(live_bricks).to(self.device))
            self._table_cpu[live_ids] = slots.astype(np.int32)
            self.table[torch.from_numpy(live_ids).to(self.device)] = \
                slot_device.to(torch.int32)
            self._slot_to_brick[slots] = live_ids
            self._last_used[slots] = self._epoch
            loaded += len(live_ids)
        self.total_loaded_bricks += loaded
        return loaded, zero_bricks

    def prepare(self, indices_zyx):
        """Resolve residency and return a pinned, one-shot gather request.

        Preparation may block while GPU-produced indices are inspected and
        missing Zarr bricks are read.  :meth:`gather_prepared` performs only
        CUDA work and returns without synchronizing the host.
        """
        started = time.perf_counter()
        original_shape = tuple(indices_zyx.shape[:-1])
        flat = indices_zyx.detach().reshape(-1, 3).to(
            device=self.device, dtype=torch.int64)
        if not flat.numel():
            return _PreparedGather(
                self, flat, torch.empty(
                    0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
                original_shape, np.empty(0, dtype=np.int64), None)

        with self._lock:
            self._collect_timing()
            self._epoch += 1
            self._wait_for_table()
            brick_ids = self._brick_ids(flat)
            # The host owns cache residency and Zarr loading, so preparation
            # explicitly transfers only the deduplicated brick IDs.
            requested = torch.unique(brick_ids).cpu().numpy()
            after_lookup = time.perf_counter()
            missing = requested[self._table_cpu[requested] < 0]
            loaded = zero_bricks = 0
            unique_count = len(requested) if len(missing) else 0
            if len(missing):
                loaded, zero_bricks = self._load_missing(missing, requested)
                self._table_ready_event = self._record_event()
            resident_slots = self._table_cpu[requested]
            if bool((resident_slots < 0).any()):
                raise RuntimeError('CUDA brick table still has misses after load')
            pinned_slots = np.unique(resident_slots[resident_slots > 0])
            self._slot_pins[pinned_slots] += 1
            self._last_used[pinned_slots] = self._epoch
            slots = self.table[brick_ids].long()
            global_z = flat[:, 0] + self.z_origin
            local = (((global_z % self.brick_size) * self.brick_size
                      + flat[:, 1] % self.brick_size) * self.brick_size
                     + flat[:, 2] % self.brick_size)
            ready_event = self._record_event()
            finished = time.perf_counter()
            self._last_timings = {
                'lookup_seconds': after_lookup - started,
                'load_seconds': finished - after_lookup,
                'loaded_bricks': loaded,
                'zero_bricks': zero_bricks,
                'requested_unique_bricks': unique_count,
                'resident_bricks': self.resident_bricks,
                'resident_gb': self.resident_bytes / 1e9,
            }
            return _PreparedGather(
                self, flat, slots, local, original_shape, pinned_slots,
                ready_event)

    def _release_prepared(self, prepared):
        with self._lock:
            if prepared._store is not self or prepared.consumed:
                return
            self._slot_pins[prepared.pinned_slots] -= 1
            prepared.consumed = True

    def gather_prepared(self, prepared, channels=None):
        """Enqueue a prepared gather and return without CUDA synchronization."""
        if prepared._store is not self:
            raise ValueError('prepared gather belongs to another store')
        if channels is None:
            channels = tuple(range(self.channel_count))
        channels = tuple(int(channel) for channel in channels)
        if any(channel < 0 or channel >= self.channel_count
               for channel in channels):
            raise IndexError('brick-volume channel is out of range')

        with self._lock:
            if prepared.consumed:
                raise RuntimeError('prepared gather has already been consumed')
            stream = self._current_stream()
            if stream is not None and prepared.ready_event is not None:
                stream.wait_event(prepared.ready_event)
            if not prepared.flat.numel():
                prepared.consumed = True
                return torch.empty(
                    (*prepared.original_shape, len(channels)),
                    dtype=torch.uint8, device=self.device)

            started = time.perf_counter()
            timing_start = self._record_event(timing=True)
            channel_index = torch.as_tensor(
                channels, dtype=torch.long, device=self.device)
            values = self.pool[
                channel_index[:, None], prepared.slots[None, :],
                prepared.local[None, :]].transpose(0, 1)
            timing_end = self._record_event(timing=True)
            read_event = self._record_event()
            self._record_slot_readers(prepared.pinned_slots, read_event)
            self._slot_pins[prepared.pinned_slots] -= 1
            prepared.consumed = True
            self._last_timings['gather_enqueue_seconds'] = \
                time.perf_counter() - started
            if timing_start is not None:
                self._pending_timing = (timing_start, timing_end)
            else:
                self._last_timings['gather_seconds'] = \
                    self._last_timings['gather_enqueue_seconds']
        return values.reshape(*prepared.original_shape, len(channels))

    def gather(self, indices_zyx, channel=0):
        """Gather one channel at ROI-local integer coordinates.

        The output shape is ``indices_zyx.shape[:-1]``.  Indices must already
        be clamped to the ROI bounds by the caller.
        """
        return self.gather_channels(indices_zyx, (channel,))[..., 0]

    def gather_channels(self, indices_zyx, channels=None):
        """Prepare, enqueue, and return selected channels in request order."""
        return self.gather_prepared(self.prepare(indices_zyx), channels)
