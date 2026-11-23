"""Sparse GPU chunk cache for streaming zarr volumes.

On CUDA the chunk table holds raw device pointers and the samplers are the JIT-built CUDA kernels. On every other
device (MPS, CPU) the table holds int32 pool slots (slot + 1, 0 = not loaded) into one growable
``(S, C, 34, 34, 34)`` uint8 pool, sampled by the Metal kernels on MPS (``u8_sample_metal``) or by the plain
PyTorch reference (``u8_sample_torch``) elsewhere.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

_CHUNK_SIZE = 32
_PADDED = _CHUNK_SIZE + 2  # 34: 1 voxel margin each side
READER_MEMORY_CACHE_BYTES = 8 << 30


def _metal_enabled() -> bool:
    """Native Metal kernels on MPS unless LASAGNA_METAL=0 selects the PyTorch reference."""
    return os.environ.get("LASAGNA_METAL", "1") != "0"


def _fmt_mib(value: int | float) -> str:
    return f"{float(value) / 1024.0 ** 2:.1f}MiB"


class SparseChunkGroupCache:
    """Sparse GPU chunk cache for one zarr channel group.

    chunk_table: (cZ, cY, cX) on the device. On CUDA, int64 device pointers to uint8[C, 34, 34, 34] padded
    chunks; elsewhere, int32 slot + 1 into ``pool``. 0 means not loaded in both layouts.

    prefetch() records the chunks a sampling step will need; sync() reads them
    with one batched call to the opened ZarrReader (in parallel, sharing the
    store chunks they overlap) and uploads them. Chunks are allocated in
    batches on GPU as needed. No fixed capacity limit — memory grows until GPU OOM.
    """

    def __init__(
        self,
        *,
        channels: list[str],
        reader,
        vol_shape_zyx: tuple[int, int, int],
        channel_indices: dict[str, int],
        is_3d_zarr: bool,
        device: torch.device,
    ) -> None:
        if reader.dtype != np.uint8:
            raise ValueError(f"sparse sampling needs a uint8 array, {reader.location} is {reader.dtype}")
        self.channels = channels
        self.reader = reader
        self.n_channels = len(channels)
        self.vol_shape_zyx = tuple(int(v) for v in vol_shape_zyx)
        self.channel_indices = channel_indices  # channel_name -> index in zarr C dim
        self.is_3d_zarr = bool(is_3d_zarr)
        self.device = torch.device(device)
        Z, Y, X = self.vol_shape_zyx
        self.chunk_grid = (
            (Z + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
            (Y + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
            (X + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
        )
        cZ, cY, cX = self.chunk_grid

        # raw device pointers need CUDA; other devices index one pool by slot
        self.slot_mode = torch.device(device).type != "cuda"
        self._batches: list[torch.Tensor] = []  # keep alive to prevent GC
        self._pending: dict[tuple[int, int, int], None] = {}  # in request order
        self._collector = None  # Metal prefetch rounds (u8_sample_metal.MissingChunkCollector), made on first use
        if self.slot_mode:
            self.chunk_table = torch.zeros(cZ, cY, cX, dtype=torch.int32, device=device)
            self.pool = torch.empty(0, self.n_channels, _PADDED, _PADDED, _PADDED, dtype=torch.uint8, device=device)
            self.n_slots = 0
            self._transfer_stream = None
        else:
            self.chunk_table = torch.zeros(cZ, cY, cX, dtype=torch.int64, device=device)
            self._transfer_stream = torch.cuda.Stream(device=device)

        # Stats: accumulated over entire optimization
        self._iter_count: int = 0
        self._total_new_chunks: int = 0
        self._total_fetch_ms: float = 0.0
        self._last_sync_new: int = 0  # chunks from most recent sync

        table_mib = cZ * cY * cX * 8 / 1024**2
        print(f"[sparse_cache] {','.join(channels)}: chunk_grid={cZ}x{cY}x{cX} "
              f"vol={Z}x{Y}x{X} table={table_mib:.1f}MiB", flush=True)

    def prefetch(self, xyz_fullres: torch.Tensor, origin: tuple[float, float, float],
                 spacing: tuple[float, float, float]) -> None:
        """Record the chunks needed for given sample positions; sync() loads them.

        xyz_fullres: (..., 3) float32 GPU — sample positions in fullres coords.
        """
        dev = xyz_fullres.device
        if self.slot_mode and dev.type == "mps" and _metal_enabled():
            # nothing is read back here: on MPS each readback waits for the queue (about 0.5-1 ms), so the
            # collector gathers every prefetch call of this round and sync() reads the list once
            if self._collector is None:
                from u8_sample_metal import MissingChunkCollector
                self._collector = MissingChunkCollector(self.chunk_grid, dev)
            self._collector.add(xyz_fullres, self.chunk_table, origin, spacing)
            return

        if self.slot_mode:
            from u8_sample_torch import missing_chunks
        else:
            from sparse_prefetch_chunks import missing_chunks
        with torch.no_grad():
            origin_t = torch.tensor(origin, dtype=torch.float32, device=dev)
            spacing_t = torch.tensor(spacing, dtype=torch.float32, device=dev)
            coords = missing_chunks(xyz_fullres, self.chunk_table, origin_t, spacing_t).cpu()
        self._pending.update(dict.fromkeys(tuple(int(v) for v in row) for row in coords.tolist()))

    def _regions(self, coord: tuple[int, int, int], chunk: np.ndarray) -> list[tuple]:
        """(start, size, destination) reads filling one (C, 34, 34, 34) padded chunk."""
        start = tuple(c * _CHUNK_SIZE - 1 for c in coord)
        if self.is_3d_zarr:
            # 3D zarr: (Z, Y, X) — one channel at index 0
            return [(start, (_PADDED,) * 3, chunk[0])]
        # 4D zarr: (C_zarr, Z, Y, X) — read specific channels
        return [
            ((int(self.channel_indices[name]), *start), (1, *(_PADDED,) * 3), chunk[i:i + 1])
            for i, name in enumerate(self.channels)
        ]

    def sync(self) -> None:
        """Read pending chunks, batch-transfer to GPU, update chunk_table."""
        if self._collector is not None:
            self._pending.update(dict.fromkeys(tuple(int(v) for v in row) for row in self._collector.take().tolist()))
        if not self._pending:
            self._last_sync_new = 0
            return

        t0 = time.perf_counter()
        coords = list(self._pending)
        self._pending = {}
        n = len(coords)
        C = self.n_channels
        chunk_bytes = C * _PADDED**3  # bytes per chunk
        if self.slot_mode:
            self._sync_slots(coords, t0)
            return
        try:
            # Read straight into a pinned CPU tensor for efficient transfer
            cpu_batch = torch.empty(n, C, _PADDED, _PADDED, _PADDED, dtype=torch.uint8,
                                    pin_memory=True)
            buffers = cpu_batch.numpy()
            reads = [read for i, coord in enumerate(coords) for read in self._regions(coord, buffers[i])]
            self.reader.read_regions([read[:2] for read in reads], out=[read[2] for read in reads])

            # Batch transfer to GPU
            with torch.cuda.stream(self._transfer_stream):
                gpu_batch = cpu_batch.to(self.device, non_blocking=True)
            self._transfer_stream.synchronize()
        except torch.OutOfMemoryError:
            print(f"[sparse_cache_oom] sync: pending_chunks={n} pending={_fmt_mib(n * chunk_bytes)} "
                  f"{self.cache_diagnostics()}", flush=True)
            raise

        # Store batch to prevent GC
        self._batches.append(gpu_batch)

        # Update chunk_table with device pointers
        base_ptr = gpu_batch.data_ptr()
        for i, (cz, cy, cx) in enumerate(coords):
            self.chunk_table[cz, cy, cx] = base_ptr + i * chunk_bytes

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._last_sync_new = n
        self._total_new_chunks += n
        self._total_fetch_ms += dt_ms

    def _sync_slots(self, coords: list[tuple[int, int, int]], t0: float) -> None:
        """Slot-mode sync: read the chunks into the pool's next slots and point the table at them."""
        n = len(coords)
        C = self.n_channels
        cpu_batch = torch.empty(n, C, _PADDED, _PADDED, _PADDED, dtype=torch.uint8)
        buffers = cpu_batch.numpy()
        reads = [read for i, coord in enumerate(coords) for read in self._regions(coord, buffers[i])]
        self.reader.read_regions([read[:2] for read in reads], out=[read[2] for read in reads])

        first = self.n_slots
        need = first + n
        if need > int(self.pool.shape[0]):
            # grow by doubling so repeated small syncs copy the pool O(log n) times
            capacity = max(need, 2 * int(self.pool.shape[0]), 64)
            grown = torch.empty(capacity, C, _PADDED, _PADDED, _PADDED, dtype=torch.uint8, device=self.device)
            if first:
                grown[:first].copy_(self.pool[:first])
            self.pool = grown
        # blocking copy: a non_blocking CPU to MPS copy can read the source after it is freed (villa #1990)
        self.pool[first:need].copy_(cpu_batch)
        self.n_slots = need

        idx = torch.tensor(coords, dtype=torch.long).t().to(self.device)
        slots = torch.arange(first + 1, need + 1, dtype=torch.int32, device=self.device)
        self.chunk_table[idx[0], idx[1], idx[2]] = slots

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

    def cache_diagnostics(self) -> str:
        """One line about GPU memory held by this cache, for out-of-memory reports."""
        table_bytes = self.chunk_table.numel() * self.chunk_table.element_size()
        batch_bytes = sum(t.numel() * t.element_size() for t in self._batches)
        if self.slot_mode:
            batch_bytes = self.pool.numel() * self.pool.element_size()
        return (
            f"channels={','.join(self.channels)} device={self.device} "
            f"chunk_grid={self.chunk_grid[0]}x{self.chunk_grid[1]}x{self.chunk_grid[2]} "
            f"loaded_chunks_tracked={self._total_new_chunks} "
            f"last_sync_new={self._last_sync_new} pending_chunks={len(self._pending)} "
            f"batches={len(self._batches)} table={_fmt_mib(table_bytes)} "
            f"batch_gpu={_fmt_mib(batch_bytes)}"
        )

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
        if self.slot_mode:
            out = self._grid_sample_slots(xyz_fullres, origin, inv_scale, diff=diff)
        elif diff:
            from sparse_grid_sample_3d_u8_diff import sparse_grid_sample_3d_u8_diff
            out = sparse_grid_sample_3d_u8_diff(
                self.chunk_table, self.n_channels, xyz_fullres, origin, inv_scale,
            )
        else:
            from sparse_grid_sample_3d_u8 import sparse_grid_sample_3d_u8
            out = sparse_grid_sample_3d_u8(
                self.chunk_table, self.n_channels, xyz_fullres, origin, inv_scale,
            )
        if check_enabled and not self.slot_mode:
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

    def _grid_sample_slots(self, xyz_fullres: torch.Tensor, origin: torch.Tensor,
                           inv_scale: torch.Tensor, *, diff: bool) -> torch.Tensor:
        pool = self.pool[:max(self.n_slots, 1)] if self.n_slots else self._empty_pool()
        if xyz_fullres.device.type == "mps" and _metal_enabled():
            from u8_sample_metal import sparse_sample
        else:
            from u8_sample_torch import sparse_sample
        return sparse_sample(self.chunk_table, pool, xyz_fullres, origin, inv_scale, diff=diff)

    def _empty_pool(self) -> torch.Tensor:
        # one zero brick so an all-empty table still has a valid buffer to bind; no slot points at it
        return torch.zeros(1, self.n_channels, _PADDED, _PADDED, _PADDED, dtype=torch.uint8, device=self.device)

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
            if not bool(finite.all().detach().cpu()):
                bad_idx = (~finite).nonzero(as_tuple=False).flatten()
                first = bad_idx[:8]
                ctx = f" context={context}" if context else ""
                raise RuntimeError(
                    "non-finite sparse sample coordinates before CUDA sample: "
                    f"channels={','.join(self.channels)}{ctx} "
                    f"bad={int(bad_idx.numel())}/{int(flat.shape[0])} "
                    f"chunk_grid={cZ}x{cY}x{cX} "
                    f"first_local_xyz={local[first].detach().cpu().tolist()} "
                    f"first_full_xyz={flat[first].detach().cpu().tolist()}"
                )
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
