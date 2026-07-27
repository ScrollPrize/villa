"""Bounded LRU cache for sparse integer gathers from local uint8 Zarr arrays.

The loss code asks for integer SDT corners and nearest-neighbour normal
samples.  Keeping that contract here preserves the existing SDT no-data
renormalisation while moving the random fetch itself entirely onto the GPU
after a chunk is resident.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

CHUNK_SIZE = 32
CHUNK_VOXELS = CHUNK_SIZE ** 3


def _env_gib(name: str, default: float) -> float:
    raw = os.environ.get(name)
    value = default if raw is None else float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def cache_budget_bytes(kind: str, device: torch.device | str) -> int:
    """Return the bounded device-pool budget for one logical field."""
    defaults = {"sdt": 16.0, "normals": 6.0, "grad_mag": 2.0}
    fractions = {"sdt": 0.30, "normals": 0.20, "grad_mag": 0.10}
    names = {
        "sdt": "FIT_SPIRAL_SPARSE_SDT_CACHE_GB",
        "normals": "FIT_SPIRAL_SPARSE_NORMAL_CACHE_GB",
        "grad_mag": "FIT_SPIRAL_SPARSE_GRAD_CACHE_GB",
    }
    requested = int(_env_gib(names[kind], defaults[kind]) * 1024 ** 3)
    device = torch.device(device)
    if device.type != "cuda":
        return requested
    free, _total = torch.cuda.mem_get_info(device)
    # Loaders are created sequentially. Capping each against currently free
    # memory leaves room for the model, optimizer, and transient loss graphs on
    # smaller cards while retaining the measured H100 defaults.
    return max(CHUNK_VOXELS, min(requested, int(free * fractions[kind])))


class BoundedSparseCudaCache:
    """Fixed-size device chunk pool backed by TensorStore and LRU eviction.

    ``source_paths`` are Zarr array directories, one per output channel.
    Gather indices are local to the configured ROI; ``origin_zyx`` maps them
    back to the full source arrays.
    """

    def __init__(
        self,
        *,
        source_paths: list[str],
        shape_zyx: tuple[int, int, int],
        origin_zyx: tuple[int, int, int] = (0, 0, 0),
        budget_bytes: int,
        device: torch.device | str = "cuda",
        label: str,
        io_batch_chunks: int = 256,
        tensorstore_cache_bytes: int | None = None,
    ) -> None:
        if not source_paths:
            raise ValueError("source_paths must contain at least one Zarr array")
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        self.source_paths = [str(Path(path)) for path in source_paths]
        self.shape_zyx = tuple(int(v) for v in shape_zyx)
        self.origin_zyx = tuple(int(v) for v in origin_zyx)
        self.device = torch.device(device)
        self.label = str(label)
        self.channels = len(self.source_paths)
        self.io_batch_chunks = max(1, int(io_batch_chunks))

        z, y, x = self.shape_zyx
        self.chunk_grid_zyx = (
            (z + CHUNK_SIZE - 1) // CHUNK_SIZE,
            (y + CHUNK_SIZE - 1) // CHUNK_SIZE,
            (x + CHUNK_SIZE - 1) // CHUNK_SIZE,
        )
        total_chunks = int(np.prod(self.chunk_grid_zyx, dtype=np.int64))
        bytes_per_slot = self.channels * CHUNK_VOXELS
        self.capacity = min(total_chunks, max(1, int(budget_bytes) // bytes_per_slot))
        self.pool_bytes = self.capacity * bytes_per_slot

        try:
            self.pool = torch.empty(
                (self.channels, self.capacity, CHUNK_VOXELS),
                dtype=torch.uint8,
                device=self.device,
            )
            self.chunk_table = torch.full(
                self.chunk_grid_zyx,
                -1,
                dtype=torch.int32,
                device=self.device,
            )
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                f"Could not allocate {self.label} sparse cache "
                f"({self.pool_bytes / 1024**3:.2f} GiB, {self.capacity} slots)"
            ) from exc

        self._lru: OrderedDict[int, int] = OrderedDict()
        self._free_slots = list(range(self.capacity - 1, -1, -1))
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._gathers = 0
        self._read_seconds = 0.0
        self._gather_seconds = 0.0
        self.last_timings: dict[str, float | int] = {}

        import tensorstore as ts

        if tensorstore_cache_bytes is None:
            tensorstore_cache_bytes = int(
                _env_gib("FIT_SPIRAL_TENSORSTORE_CACHE_GB", 2.0) * 1024 ** 3
            )
        self._ts = ts
        self._context = ts.Context({
            "cache_pool": {"total_bytes_limit": int(tensorstore_cache_bytes)},
            "file_io_concurrency": {
                "limit": int(os.environ.get("FIT_SPIRAL_SPARSE_IO_THREADS", "16"))
            },
            "data_copy_concurrency": {
                "limit": int(os.environ.get("FIT_SPIRAL_SPARSE_COPY_THREADS", "8"))
            },
        })
        self._stores = [
            ts.open(
                {
                    "driver": "zarr",
                    "kvstore": {"driver": "file", "path": path},
                },
                context=self._context,
                open=True,
                read=True,
                recheck_cached_data="open",
            ).result()
            for path in self.source_paths
        ]
        print(
            f"{self.label}: sparse CUDA LRU cache "
            f"{self.capacity}x{CHUNK_SIZE}^3x{self.channels} "
            f"({self.pool_bytes / 1024**3:.2f} GiB), "
            f"grid={self.chunk_grid_zyx}",
            flush=True,
        )

    def _key_to_coord(self, key: int) -> tuple[int, int, int]:
        _cz, cy_count, cx_count = self.chunk_grid_zyx
        cz, rem = divmod(int(key), cy_count * cx_count)
        cy, cx = divmod(rem, cx_count)
        return cz, cy, cx

    def _plan_slots(
        self, requested_keys: list[int]
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Return ``(loads, evictions)`` as ``(chunk_key, slot)`` pairs."""
        protected = set(requested_keys)
        missing = []
        for key in requested_keys:
            slot = self._lru.get(key)
            if slot is None:
                missing.append(key)
                self._misses += 1
            else:
                self._hits += 1
                self._lru.move_to_end(key)
        if len(requested_keys) > self.capacity:
            required_gib = (
                len(requested_keys) * self.channels * CHUNK_VOXELS / 1024 ** 3
            )
            raise RuntimeError(
                f"{self.label} gather touches {len(requested_keys)} chunks "
                f"({required_gib:.2f} GiB), exceeding its {self.capacity}-chunk "
                f"LRU capacity; increase the corresponding FIT_SPIRAL_SPARSE_*_CACHE_GB"
            )

        loads: list[tuple[int, int]] = []
        evictions: list[tuple[int, int]] = []
        for key in missing:
            if self._free_slots:
                slot = self._free_slots.pop()
            else:
                victim = None
                for candidate, candidate_slot in self._lru.items():
                    if candidate not in protected:
                        victim = (candidate, candidate_slot)
                        break
                if victim is None:
                    raise RuntimeError(
                        f"{self.label} could not find an evictable cache slot"
                    )
                victim_key, slot = victim
                del self._lru[victim_key]
                evictions.append((victim_key, slot))
                self._evictions += 1
            self._lru[key] = slot
            loads.append((key, slot))
        return loads, evictions

    def _chunk_bounds(self, key: int):
        cz, cy, cx = self._key_to_coord(key)
        z0, y0, x0 = cz * CHUNK_SIZE, cy * CHUNK_SIZE, cx * CHUNK_SIZE
        z1 = min(z0 + CHUNK_SIZE, self.shape_zyx[0])
        y1 = min(y0 + CHUNK_SIZE, self.shape_zyx[1])
        x1 = min(x0 + CHUNK_SIZE, self.shape_zyx[2])
        return (cz, cy, cx), (slice(z0, z1), slice(y0, y1), slice(x0, x1))

    def _load_chunks(self, loads: list[tuple[int, int]]) -> None:
        if not loads:
            return
        started = time.perf_counter()
        for batch_lo in range(0, len(loads), self.io_batch_chunks):
            batch_loads = loads[batch_lo:batch_lo + self.io_batch_chunks]
            batch = self._ts.Batch()
            pending = []
            bounds = []
            for key, _slot in batch_loads:
                coord, slices = self._chunk_bounds(key)
                bounds.append((coord, slices))
                pending.append([
                    store[slices].read(order="C", batch=batch)
                    for store in self._stores
                ])
            batch.submit()

            host = np.zeros(
                (len(batch_loads), self.channels, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE),
                dtype=np.uint8,
            )
            for row, futures in enumerate(pending):
                _coord, slices = bounds[row]
                dz = slices[0].stop - slices[0].start
                dy = slices[1].stop - slices[1].start
                dx = slices[2].stop - slices[2].start
                for channel, future in enumerate(futures):
                    host[row, channel, :dz, :dy, :dx] = np.asarray(
                        future.result(), dtype=np.uint8
                    )

            values = torch.from_numpy(host.reshape(
                len(batch_loads), self.channels, CHUNK_VOXELS
            ))
            if self.device.type == "cuda":
                values = values.pin_memory().to(self.device, non_blocking=True)
            else:
                values = values.to(self.device)
            slots = torch.tensor(
                [slot for _key, slot in batch_loads],
                dtype=torch.long,
                device=self.device,
            )
            self.pool[:, slots, :] = values.permute(1, 0, 2)
            coords = torch.tensor(
                [coord for coord, _slices in bounds],
                dtype=torch.long,
                device=self.device,
            )
            self.chunk_table[coords[:, 0], coords[:, 1], coords[:, 2]] = (
                slots.to(torch.int32)
            )
        self._read_seconds += time.perf_counter() - started

    def gather(self, indices_zyx: torch.Tensor) -> torch.Tensor:
        """Gather local ROI indices, returning ``(..., channels)`` uint8."""
        started = time.perf_counter()
        original_shape = tuple(indices_zyx.shape[:-1])
        flat = indices_zyx.detach().reshape(-1, 3).to(
            device=self.device, dtype=torch.long
        )
        origin = torch.tensor(
            self.origin_zyx, dtype=torch.long, device=self.device
        )
        source = flat + origin
        shape = torch.tensor(
            self.shape_zyx, dtype=torch.long, device=self.device
        )
        if bool(((source < 0) | (source >= shape)).any()):
            raise IndexError(f"{self.label} gather received an out-of-bounds index")

        chunk = torch.div(source, CHUNK_SIZE, rounding_mode="floor")
        _cz_count, cy_count, cx_count = self.chunk_grid_zyx
        keys = (chunk[:, 0] * cy_count + chunk[:, 1]) * cx_count + chunk[:, 2]
        unique_keys = torch.unique(keys)
        key_list = [int(v) for v in unique_keys.cpu().tolist()]
        loads, evictions = self._plan_slots(key_list)
        if evictions:
            evicted_coords = torch.tensor(
                [self._key_to_coord(key) for key, _slot in evictions],
                dtype=torch.long,
                device=self.device,
            )
            self.chunk_table[
                evicted_coords[:, 0], evicted_coords[:, 1], evicted_coords[:, 2]
            ] = -1
        self._load_chunks(loads)

        slots = self.chunk_table[
            chunk[:, 0], chunk[:, 1], chunk[:, 2]
        ].to(torch.long)
        if bool((slots < 0).any()):
            raise RuntimeError(f"{self.label} cache miss remained after loading")
        local = source - chunk * CHUNK_SIZE
        linear = (local[:, 0] * CHUNK_SIZE + local[:, 1]) * CHUNK_SIZE + local[:, 2]
        values = self.pool[:, slots, linear].transpose(0, 1)

        elapsed = time.perf_counter() - started
        self._gather_seconds += elapsed
        self._gathers += 1
        self.last_timings = {
            "gather_seconds": elapsed,
            "requested_chunks": len(key_list),
            "loaded_chunks": len(loads),
            "resident_chunks": len(self._lru),
            "evictions": len(evictions),
            "resident_mib": len(self._lru) * self.channels * CHUNK_VOXELS / 1024 ** 2,
        }
        return values.reshape(*original_shape, self.channels)

    def stats(self) -> dict[str, float | int]:
        total = self._hits + self._misses
        return {
            "capacity_chunks": self.capacity,
            "resident_chunks": len(self._lru),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
            "evictions": self._evictions,
            "gathers": self._gathers,
            "read_seconds": self._read_seconds,
            "gather_seconds": self._gather_seconds,
            "pool_bytes": self.pool_bytes,
        }

    def close(self) -> None:
        # Drop the owning references even when the surrounding volume dict
        # remains alive (notably in an interactive session reload).
        self._stores.clear()
        self._lru.clear()
        self._free_slots.clear()
        self.pool = None
        self.chunk_table = None
        self._context = None


class SparseLasagnaStore:
    def __init__(
        self,
        *,
        normal_cache: BoundedSparseCudaCache | None,
        grad_cache: BoundedSparseCudaCache | None,
    ) -> None:
        self.normal_cache = normal_cache
        self.grad_cache = grad_cache
        self.last_timings: dict[str, float | int] = {}

    def gather_pair(self, normal_zyx, grad_zyx, device):
        if normal_zyx.numel():
            if self.normal_cache is None:
                raise RuntimeError("normal cache is not configured")
            normals = self.normal_cache.gather(normal_zyx)
        else:
            normals = torch.empty(
                (*normal_zyx.shape[:-1], 2), dtype=torch.uint8, device=device
            )
        if grad_zyx.numel():
            if self.grad_cache is None:
                raise RuntimeError("gradient-magnitude cache is not configured")
            gradient = self.grad_cache.gather(grad_zyx)[..., 0]
        else:
            gradient = torch.empty(
                grad_zyx.shape[:-1], dtype=torch.uint8, device=device
            )
        self.last_timings = {}
        if self.normal_cache is not None:
            self.last_timings.update({
                f"normal_{key}": value
                for key, value in self.normal_cache.last_timings.items()
            })
        if self.grad_cache is not None:
            self.last_timings.update({
                f"grad_{key}": value
                for key, value in self.grad_cache.last_timings.items()
            })
        return normals, gradient

    def close(self):
        if self.normal_cache is not None:
            self.normal_cache.close()
        if self.grad_cache is not None:
            self.grad_cache.close()


class SparseScalarStore:
    def __init__(self, cache: BoundedSparseCudaCache) -> None:
        self.cache = cache
        self.last_timings: dict[str, float | int] = {}

    def gather(self, indices_zyx, device):
        values = self.cache.gather(indices_zyx)[..., 0]
        self.last_timings = dict(self.cache.last_timings)
        return values

    def close(self):
        self.cache.close()
