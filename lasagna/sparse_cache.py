"""Sparse GPU chunk cache for streaming zarr volumes."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import TYPE_CHECKING

import numpy as np
import torch
import zarr

if TYPE_CHECKING:
    pass

_CHUNK_SIZE = 32
_PADDED = _CHUNK_SIZE + 2  # 34: 1 voxel margin each side


def _dilate26(t: torch.Tensor) -> torch.Tensor:
    """26-connected dilation of a 3D bool tensor (full 3x3x3 neighborhood)."""
    d = t.clone()
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                sz = slice(max(0, dz), t.shape[0] + min(0, dz))
                sy = slice(max(0, dy), t.shape[1] + min(0, dy))
                sx = slice(max(0, dx), t.shape[2] + min(0, dx))
                tz = slice(max(0, -dz), t.shape[0] + min(0, -dz))
                ty = slice(max(0, -dy), t.shape[1] + min(0, -dy))
                tx = slice(max(0, -dx), t.shape[2] + min(0, -dx))
                d[sz, sy, sx] |= t[tz, ty, tx]
    return d


class SparseChunkGroupCache:
    """Sparse GPU chunk cache for one zarr channel group.

    chunk_table: int64 (cZ, cY, cX) on GPU — each entry is a device pointer
    to a uint8[C, 34, 34, 34] padded chunk, or 0 if not loaded.

    Chunks are allocated in batches on GPU as needed. No fixed capacity limit —
    memory grows until GPU OOM.
    """

    def __init__(
        self,
        *,
        channels: list[str],
        zarr_path: str,
        vol_shape_zyx: tuple[int, int, int],
        channel_indices: dict[str, int],
        is_3d_zarr: bool,
        device: torch.device,
        n_workers: int = 8,
        prefetch_backend: str = "cuda",
    ) -> None:
        self.channels = channels
        self.zarr_path = zarr_path
        self.n_channels = len(channels)
        self.vol_shape_zyx = vol_shape_zyx
        self.channel_indices = channel_indices  # channel_name -> index in zarr C dim
        self.is_3d_zarr = is_3d_zarr
        self.device = device
        self.prefetch_backend = str(prefetch_backend)
        if self.prefetch_backend not in {"cuda", "python"}:
            raise ValueError(
                f"unknown sparse prefetch backend {self.prefetch_backend!r}; "
                "expected 'cuda' or 'python'")

        Z, Y, X = vol_shape_zyx
        self.chunk_grid = (
            (Z + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
            (Y + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
            (X + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
        )
        cZ, cY, cX = self.chunk_grid

        # GPU storage: chunk_table stores device pointers (0 = empty)
        self.chunk_table = torch.zeros(cZ, cY, cX, dtype=torch.int64, device=device)
        self._batches: list[torch.Tensor] = []  # keep alive to prevent GC

        # Zarr handle kept open
        self._zarr = zarr.open(zarr_path, mode="r")

        # Background loader
        self._executor = ThreadPoolExecutor(max_workers=n_workers)
        self._pending: list[Future] = []
        self._transfer_stream = torch.cuda.Stream(device=device)

        # Stats: accumulated over entire optimization
        self._iter_count: int = 0
        self._total_new_chunks: int = 0
        self._total_fetch_ms: float = 0.0
        self._last_sync_new: int = 0  # chunks from most recent sync

        table_mib = cZ * cY * cX * 8 / 1024**2
        print(f"[sparse_cache] {','.join(channels)}: chunk_grid={cZ}x{cY}x{cX} "
              f"vol={Z}x{Y}x{X} table={table_mib:.1f}MiB "
              f"prefetch={self.prefetch_backend}", flush=True)

    def prefetch(self, xyz_fullres: torch.Tensor, origin: tuple[float, float, float],
                 spacing: tuple[float, float, float]) -> None:
        """Start async loading of chunks needed for given sample positions.

        xyz_fullres: (..., 3) float32 GPU — sample positions in fullres coords.
        """
        coords = self._missing_chunk_coords(xyz_fullres, origin, spacing)
        if coords.numel() == 0:
            return
        n_missing = coords.shape[0]

        # Submit reads to thread pool
        for i in range(n_missing):
            cz, cy, cx = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
            fut = self._executor.submit(self._load_chunk, cz, cy, cx)
            self._pending.append(fut)

    def _missing_chunk_coords(self, xyz_fullres: torch.Tensor,
                              origin: tuple[float, float, float],
                              spacing: tuple[float, float, float]) -> torch.Tensor:
        """Return CPU int64 (N, 3) missing chunk coordinates as (cz, cy, cx)."""
        cZ, cY, cX = self.chunk_grid
        dev = xyz_fullres.device
        origin_t = torch.tensor(origin, dtype=torch.float32, device=dev)
        spacing_t = torch.tensor(spacing, dtype=torch.float32, device=dev)

        with torch.no_grad():
            if self.prefetch_backend == "cuda":
                from sparse_prefetch_chunks import missing_chunks
                return missing_chunks(
                    xyz_fullres, self.chunk_table, origin_t, spacing_t).cpu()

            flat = xyz_fullres.reshape(-1, 3)
            local = (flat - origin_t) / spacing_t
            ci_x = (local[:, 0] / _CHUNK_SIZE).long().clamp(0, cX - 1)
            ci_y = (local[:, 1] / _CHUNK_SIZE).long().clamp(0, cY - 1)
            ci_z = (local[:, 2] / _CHUNK_SIZE).long().clamp(0, cZ - 1)

            needed = torch.zeros(cZ, cY, cX, dtype=torch.bool, device=dev)
            needed[ci_z, ci_y, ci_x] = True
            needed = _dilate26(needed)
            missing = needed & (self.chunk_table == 0)
            return missing.nonzero().cpu()

    def _load_chunk(self, cz: int, cy: int, cx: int) -> tuple[int, int, int, np.ndarray]:
        """Read a 34³ padded chunk from zarr. Runs in thread pool."""
        Z, Y, X = self.vol_shape_zyx
        C = self.n_channels

        # Global voxel range with 1-voxel margin
        gz0 = cz * _CHUNK_SIZE - 1
        gy0 = cy * _CHUNK_SIZE - 1
        gx0 = cx * _CHUNK_SIZE - 1

        # Clamp to volume bounds
        rz0 = max(0, gz0)
        ry0 = max(0, gy0)
        rx0 = max(0, gx0)
        rz1 = min(Z, gz0 + _PADDED)
        ry1 = min(Y, gy0 + _PADDED)
        rx1 = min(X, gx0 + _PADDED)

        buf = np.zeros((C, _PADDED, _PADDED, _PADDED), dtype=np.uint8)

        if rz1 > rz0 and ry1 > ry0 and rx1 > rx0:
            # Destination slice in buf
            dz0 = rz0 - gz0
            dy0 = ry0 - gy0
            dx0 = rx0 - gx0
            dz1 = dz0 + (rz1 - rz0)
            dy1 = dy0 + (ry1 - ry0)
            dx1 = dx0 + (rx1 - rx0)

            if self.is_3d_zarr:
                # 3D zarr: (Z, Y, X) — one channel at index 0
                data = np.asarray(self._zarr[rz0:rz1, ry0:ry1, rx0:rx1])
                buf[0, dz0:dz1, dy0:dy1, dx0:dx1] = data
            else:
                # 4D zarr: (C_zarr, Z, Y, X) — read specific channels
                for ch_name, ch_idx in self.channel_indices.items():
                    i = self.channels.index(ch_name)
                    data = np.asarray(self._zarr[ch_idx, rz0:rz1, ry0:ry1, rx0:rx1])
                    buf[i, dz0:dz1, dy0:dy1, dx0:dx1] = data

        return (cz, cy, cx, buf)

    def sync(self) -> None:
        """Wait for pending chunk loads, batch-transfer to GPU, update chunk_table."""
        if not self._pending:
            self._last_sync_new = 0
            return

        t0 = time.perf_counter()
        results = [f.result() for f in self._pending]
        self._pending.clear()

        n = len(results)
        C = self.n_channels

        # Stack into pinned CPU tensor for efficient transfer
        cpu_batch = torch.empty(n, C, _PADDED, _PADDED, _PADDED, dtype=torch.uint8,
                                pin_memory=True)
        coords_list = []
        for i, (cz, cy, cx, buf) in enumerate(results):
            cpu_batch[i] = torch.from_numpy(buf)
            coords_list.append((cz, cy, cx))

        # Batch transfer to GPU
        with torch.cuda.stream(self._transfer_stream):
            gpu_batch = cpu_batch.to(self.device, non_blocking=True)

        self._transfer_stream.synchronize()

        # Store batch to prevent GC
        self._batches.append(gpu_batch)

        # Update chunk_table with device pointers
        chunk_bytes = C * _PADDED * _PADDED * _PADDED  # bytes per chunk
        base_ptr = gpu_batch.data_ptr()
        for i, (cz, cy, cx) in enumerate(coords_list):
            ptr = base_ptr + i * chunk_bytes
            self.chunk_table[cz, cy, cx] = ptr

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._last_sync_new = n
        self._total_new_chunks += n
        self._total_fetch_ms += dt_ms

    def end_iteration(self) -> None:
        """Call once per optimizer iteration to accumulate stats."""
        self._iter_count += 1

    def print_summary(self) -> None:
        """Print accumulated cache stats summary. Call after optimization."""
        n = self._total_new_chunks
        ms = self._total_fetch_ms
        its = self._iter_count
        ms_per_it = ms / its if its > 0 else 0.0
        ms_per_chunk = ms / n if n > 0 else 0.0
        total = self.loaded_chunks()
        total_mib = self.loaded_mib()
        print(f"[sparse_cache] {','.join(self.channels)}: "
              f"{n} chunks in {its}it ({ms_per_it:.1f}ms/it, {ms_per_chunk:.1f}ms/chunk) "
              f"total={total} ({total_mib:.1f}MiB)", flush=True)

    def grid_sample(self, xyz_fullres: torch.Tensor, origin: torch.Tensor,
                    inv_scale: torch.Tensor, *, diff: bool = False,
                    context: str = "") -> torch.Tensor:
        """Sample from sparse chunk cache.

        Returns (C, D, H, W) — uint8 for non-diff, float32 for diff.
        """
        check_enabled = os.environ.get("LASAGNA_CHECK_SPARSE_CACHE", "0") != "0"
        if check_enabled:
            self._check_sample_chunks_loaded(
                xyz_fullres,
                origin,
                inv_scale,
                context=context,
            )
        if diff:
            from sparse_grid_sample_3d_u8_diff import sparse_grid_sample_3d_u8_diff
            out = sparse_grid_sample_3d_u8_diff(
                self.chunk_table, self.n_channels, xyz_fullres, origin, inv_scale,
            )
        else:
            from sparse_grid_sample_3d_u8 import sparse_grid_sample_3d_u8
            out = sparse_grid_sample_3d_u8(
                self.chunk_table, self.n_channels, xyz_fullres, origin, inv_scale,
            )
        if check_enabled:
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError as exc:
                shape = tuple(int(v) for v in xyz_fullres.shape)
                ctx = f" context={context}" if context else ""
                raise RuntimeError(
                    "sparse CUDA sample failed after cache coverage check: "
                    f"channels={','.join(self.channels)}{ctx} diff={diff} "
                    f"sample_shape={shape} loaded_chunks={self.loaded_chunks()} "
                    f"chunk_grid={self.chunk_grid}"
                ) from exc
        return out

    def _check_sample_chunks_loaded(
        self,
        xyz_fullres: torch.Tensor,
        origin: torch.Tensor,
        inv_scale: torch.Tensor,
        *,
        context: str = "",
    ) -> None:
        """Fail before CUDA sampling when in-volume sample chunks were not prefetched."""
        cZ, cY, cX = self.chunk_grid
        with torch.no_grad():
            flat = xyz_fullres.reshape(-1, 3)
            local = (flat - origin.view(1, 3)) * inv_scale.view(1, 3)
            finite = torch.isfinite(local).all(dim=1)
            ci_x = torch.floor(local[:, 0] / float(_CHUNK_SIZE)).to(torch.long)
            ci_y = torch.floor(local[:, 1] / float(_CHUNK_SIZE)).to(torch.long)
            ci_z = torch.floor(local[:, 2] / float(_CHUNK_SIZE)).to(torch.long)
            in_bounds = (
                finite &
                (ci_x >= 0) & (ci_x < cX) &
                (ci_y >= 0) & (ci_y < cY) &
                (ci_z >= 0) & (ci_z < cZ)
            )
            if not bool(in_bounds.any().detach().cpu()):
                return
            idx = in_bounds.nonzero(as_tuple=False).flatten()
            loaded = self.chunk_table[ci_z[idx], ci_y[idx], ci_x[idx]] != 0
            if bool(loaded.all().detach().cpu()):
                return
            missing_idx = idx[~loaded]
            first = missing_idx[:8]
            first_local = local[first].detach().cpu().numpy()
            first_full = flat[first].detach().cpu().numpy()
            first_chunks = torch.stack(
                [ci_z[first], ci_y[first], ci_x[first]], dim=1
            ).detach().cpu().numpy()
            total = int(flat.shape[0])
            n_in = int(idx.numel())
            n_missing = int(missing_idx.numel())
            loaded_total = int((self.chunk_table != 0).sum().detach().cpu())
            ctx = f" context={context}" if context else ""
            raise RuntimeError(
                "sparse chunk cache miss before CUDA sample: "
                f"channels={','.join(self.channels)}{ctx} "
                f"missing={n_missing}/{n_in} in-volume samples "
                f"total_samples={total} loaded_chunks={loaded_total} "
                f"chunk_grid={cZ}x{cY}x{cX} "
                f"first_chunks_zyx={first_chunks.tolist()} "
                f"first_local_xyz={first_local.tolist()} "
                f"first_full_xyz={first_full.tolist()}. "
                "Unset LASAGNA_CHECK_SPARSE_CACHE or set it to 0 to disable this debug guard."
            )

    def loaded_chunks(self) -> int:
        """Number of chunks currently loaded on GPU."""
        return int((self.chunk_table != 0).sum())

    def loaded_mib(self) -> float:
        """MiB of chunk data currently on GPU."""
        n = self.loaded_chunks()
        return n * self.n_channels * _PADDED**3 / 1024**2
