"""Fully-resident sparse brick pools for integer gathers from packed sidecars.

The loss code asks for integer SDT corners and nearest-neighbour normal
samples at positions scattered uniformly over the whole ROI, so any bounded
cache thrashes; instead the entire occupied brick set of each field is loaded
once from a ``pack_resident_pools.py`` sidecar (a single sequential read per
channel) and every gather afterwards is pure device indexing with no I/O, no
eviction bookkeeping, and no host synchronisation.

Bricks absent from the sidecar (never written, or zeroed by its CT mask) map
to the reserved all-zero row 0, which reads as the encoded no-data value the
sampling contract already handles.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from pack_resident_pools import open_pool


class ResidentBrickPool:
    """Device-resident brick pool loaded from a resident-pool sidecar.

    Gather indices are local to the configured ROI; ``origin_zyx`` maps them
    back to the full source arrays. ``z_roi`` (full-array grid voxels,
    half-open) restricts which bricks are uploaded; gathers must stay inside
    the ROI the surrounding volume dict advertises, as everywhere else.
    """

    def __init__(
        self,
        sidecar_dir: str,
        *,
        origin_zyx: tuple[int, int, int] = (0, 0, 0),
        z_roi: tuple[int, int] | None = None,
        device: torch.device | str = "cuda",
        label: str,
        expected_channels: int | None = None,
        expected_shape_zyx: tuple[int, int, int] | None = None,
    ) -> None:
        started = time.perf_counter()
        self.sidecar_dir = str(sidecar_dir)
        self.device = torch.device(device)
        self.label = str(label)

        meta, table_np, coords_np, memmaps = open_pool(sidecar_dir)
        self.meta = meta
        self.shape_zyx = tuple(int(v) for v in meta["array_shape"])
        if (expected_shape_zyx is not None
                and tuple(int(v) for v in expected_shape_zyx) != self.shape_zyx):
            raise ValueError(
                f"{label}: sidecar {sidecar_dir} covers array shape "
                f"{self.shape_zyx}, expected {tuple(expected_shape_zyx)}")
        self.channels = len(memmaps)
        if expected_channels is not None and self.channels != expected_channels:
            raise ValueError(
                f"{label}: sidecar {sidecar_dir} has {self.channels} "
                f"channel(s), expected {expected_channels}")
        self.origin_zyx = tuple(int(v) for v in origin_zyx)
        brick = tuple(int(v) for v in meta["brick_shape"])
        brick_voxels = int(np.prod(brick))
        rows = int(meta["rows"])

        keep = np.ones(rows, dtype=bool)
        if z_roi is not None:
            z_lo, z_hi = int(z_roi[0]), int(z_roi[1])
            brick_z = coords_np[:, 0].astype(np.int64)
            keep = (brick_z * brick[0] < z_hi) & ((brick_z + 1) * brick[0] > z_lo)
            keep[0] = True  # the reserved all-zero brick
        kept_ids = np.flatnonzero(keep)
        remap = np.zeros(rows, dtype=np.int32)
        remap[kept_ids] = np.arange(len(kept_ids), dtype=np.int32)

        self.pool_bytes = self.channels * len(kept_ids) * brick_voxels
        try:
            self.pool = torch.empty(
                (self.channels, len(kept_ids), brick_voxels),
                dtype=torch.uint8,
                device=self.device,
            )
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                f"Could not allocate the {self.label} resident pool "
                f"({self.pool_bytes / 1024**3:.2f} GiB, {len(kept_ids)} bricks)"
            ) from exc
        slab = max(1, (256 << 20) // brick_voxels)
        for channel, mm in enumerate(memmaps):
            for lo in range(0, len(kept_ids), slab):
                ids = kept_ids[lo:lo + slab]
                self.pool[channel, lo:lo + len(ids)] = torch.from_numpy(
                    np.ascontiguousarray(mm[ids])).to(self.device)
        self.table = torch.from_numpy(
            np.ascontiguousarray(remap[table_np])).to(self.device)

        self._brick = torch.tensor(brick, dtype=torch.long, device=self.device)
        self._origin = torch.tensor(
            self.origin_zyx, dtype=torch.long, device=self.device)
        self._shape = torch.tensor(
            self.shape_zyx, dtype=torch.long, device=self.device)
        self._stride_z = brick[1] * brick[2]
        self._stride_y = brick[2]
        self._bounds_check = (
            os.environ.get("FIT_SPIRAL_RESIDENT_BOUNDS_CHECK") == "1")
        self.resident_bricks = len(kept_ids)
        self.total_bricks = rows
        self._gathers = 0
        self._gather_seconds = 0.0
        self.load_seconds = time.perf_counter() - started
        self.last_timings: dict[str, float | int] = {}
        print(
            f"{self.label}: resident pool {self.resident_bricks:,}/{rows:,} "
            f"bricks of {brick} x{self.channels} channel(s) "
            f"({self.pool_bytes / 1024**3:.2f} GiB) loaded in "
            f"{self.load_seconds:.1f}s from {Path(sidecar_dir).name}",
            flush=True,
        )

    def gather(self, indices_zyx: torch.Tensor) -> torch.Tensor:
        """Gather local ROI indices, returning ``(..., channels)`` uint8."""
        started = time.perf_counter()
        original_shape = tuple(indices_zyx.shape[:-1])
        flat = indices_zyx.detach().reshape(-1, 3)
        if flat.shape[0] == 0:
            return torch.empty(
                (*original_shape, self.channels),
                dtype=torch.uint8, device=self.device)
        flat = flat.to(device=self.device, dtype=torch.long)
        source = flat + self._origin
        if self._bounds_check and bool(
                ((source < 0) | (source >= self._shape)).any()):
            raise IndexError(
                f"{self.label} gather received an out-of-bounds index")
        brick_idx = torch.div(source, self._brick, rounding_mode="floor")
        slots = self.table[
            brick_idx[:, 0], brick_idx[:, 1], brick_idx[:, 2]
        ].to(torch.long)
        local = source - brick_idx * self._brick
        linear = (local[:, 0] * self._stride_z
                  + local[:, 1] * self._stride_y + local[:, 2])
        values = self.pool[:, slots, linear].transpose(0, 1)

        # Host-side timer only (kernels may still be in flight); never sync.
        elapsed = time.perf_counter() - started
        self._gather_seconds += elapsed
        self._gathers += 1
        self.last_timings = {
            "gather_seconds": elapsed,
            "resident_bricks": self.resident_bricks,
            "resident_mib": self.pool_bytes / 1024 ** 2,
        }
        return values.reshape(*original_shape, self.channels)

    def stats(self) -> dict[str, float | int]:
        return {
            "resident_bricks": self.resident_bricks,
            "total_bricks": self.total_bricks,
            "gathers": self._gathers,
            "gather_seconds": self._gather_seconds,
            "load_seconds": self.load_seconds,
            "pool_bytes": self.pool_bytes,
        }

    def close(self) -> None:
        # Drop the owning references even when the surrounding volume dict
        # remains alive (notably in an interactive session reload).
        self.pool = None
        self.table = None


class SparseLasagnaStore:
    def __init__(
        self,
        *,
        normal_cache: ResidentBrickPool | None,
        grad_cache: ResidentBrickPool | None,
    ) -> None:
        self.normal_cache = normal_cache
        self.grad_cache = grad_cache
        self.last_timings: dict[str, float | int] = {}

    def gather_pair(self, normal_zyx, grad_zyx, device):
        if normal_zyx.numel():
            if self.normal_cache is None:
                raise RuntimeError("normal cache is not configured")
            normals = self.normal_cache.gather(normal_zyx).to(device)
        else:
            normals = torch.empty(
                (*normal_zyx.shape[:-1], 2), dtype=torch.uint8, device=device
            )
        if grad_zyx.numel():
            if self.grad_cache is None:
                raise RuntimeError("gradient-magnitude cache is not configured")
            gradient = self.grad_cache.gather(grad_zyx)[..., 0].to(device)
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
    def __init__(self, cache: ResidentBrickPool) -> None:
        self.cache = cache
        self.last_timings: dict[str, float | int] = {}

    def gather(self, indices_zyx, device):
        values = self.cache.gather(indices_zyx)[..., 0].to(device)
        self.last_timings = dict(self.cache.last_timings)
        return values

    def close(self):
        self.cache.close()
