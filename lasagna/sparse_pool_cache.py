"""Sparse chunk cache for streaming zarr volumes on CPU and MPS.

The CUDA cache (sparse_cache.SparseChunkGroupCache) uploads chunk batches on a
CUDA stream and keeps a table of their raw device pointers for its .cu
kernels. PooledChunkGroupCache keeps the same chunks, read the same way, in one
pooled tensor on the device with an int32 slot table (layout in
sparse_pool_ops), and samples them with the Metal kernels on MPS
(sparse_pool_metal; see metal_mode for LASAGNA_METAL) or with sparse_pool_ops'
torch ops. fit_data.load_3d_streaming picks this class for
every device type except CUDA (which includes ROCm builds of PyTorch).
"""
from __future__ import annotations

import importlib
import mmap
import os
import time

import numpy as np
import torch

import sparse_pool_ops as _ops
from sparse_cache import _CHUNK_SIZE, _PADDED, _fmt_mib

_MIN_SLOTS = 64
_HEADROOM_SLOTS = 256
# Reader decoded-chunk cache per group off CUDA (CUDA keeps sparse_cache.READER_MEMORY_CACHE_BYTES = 8 GiB):
# the pool holds every chunk read, so the reader needs only recent store chunks for halo re-reads.
POOLED_READER_MEMORY_CACHE_BYTES = 256 << 20
_metal_module = None
_metal_error: str | None = None


def metal_mode() -> str:
    """LASAGNA_METAL: unset or empty = "auto" (Metal on MPS when it imports, compiles and passes its
    self-check, else torch ops), "0" = "off" (torch ops), "1" = "required" (fail rather than fall back)."""
    raw = os.environ.get("LASAGNA_METAL", "")
    if raw == "":
        return "auto"
    if raw == "0":
        return "off"
    if raw == "1":
        return "required"
    raise ValueError(
        f"LASAGNA_METAL={raw!r}: use 0 (torch ops), 1 (require the Metal kernels) "
        "or leave it unset (Metal kernels on MPS when available)")


def _metal():
    """sparse_pool_metal if it imports and its kernels compile, else None (reason in _metal_error)."""
    global _metal_module, _metal_error
    if _metal_module is None and _metal_error is None:
        try:
            module = importlib.import_module("sparse_pool_metal")
            module.compile()
            _metal_module = module
        except Exception as exc:  # missing module, or a shader that does not compile or self-check here
            _metal_error = f"{type(exc).__name__}: {exc}"
    return _metal_module


def resolve_backend(device: torch.device):
    """(module, label) sampling chunk pools on `device`: sparse_pool_metal or sparse_pool_ops.

    The label says what runs and, when MPS falls back to torch ops, why. Raises when
    LASAGNA_METAL=1 cannot be honoured (no Metal kernels, or a device that is not MPS).
    """
    mode = metal_mode()
    device = torch.device(device)
    if mode == "off":
        return _ops, "torch (LASAGNA_METAL=0)"
    if device.type != "mps":
        if mode == "required":
            raise RuntimeError(f"LASAGNA_METAL=1 needs an MPS device, the sparse cache is on {device}")
        return _ops, "torch"
    module = _metal()
    if module is not None:
        contraction = getattr(module, "CONTRACTION", "off")
        # Bitwise equality with the torch ops needs contraction off; say so when the pragma was rejected.
        return module, "metal" if contraction == "off" else f"metal (contraction: {contraction})"
    if mode == "required":
        raise RuntimeError(f"LASAGNA_METAL=1 but the Lasagna Metal kernels are unavailable: {_metal_error}")
    return _ops, f"torch (metal unavailable: {_metal_error})"


class _PoolSampleDiff(torch.autograd.Function):
    """Differentiable pooled sample; like the CUDA autograd function, the
    backward reads the cache's table and pool as they are at backward time."""

    @staticmethod
    def forward(ctx, cache, grid, offset, inv_scale):
        out = cache.backend.sample_f32(cache.pool, cache.chunk_table, grid, offset, inv_scale)
        ctx.save_for_backward(grid)
        ctx.cache = cache
        ctx.offset = offset
        ctx.inv_scale = inv_scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grid, = ctx.saved_tensors
        cache = ctx.cache
        grad_grid = cache.backend.sample_grad(
            cache.pool, cache.chunk_table, grid, ctx.offset, ctx.inv_scale, grad_output.contiguous())
        return None, grad_grid, None, None


def _host_buffer(nbytes: int) -> torch.Tensor:
    """uint8 CPU tensor on its own anonymous memory map. Freed, it is unmapped at once: a plain
    torch.empty block of this size stays in the process footprint on macOS after it is freed."""
    if nbytes == 0:
        return torch.empty(0, dtype=torch.uint8)
    try:
        buffer = mmap.mmap(-1, nbytes)
    except OSError as exc:  # ENOMEM and the like: report it as the out-of-memory it is
        raise MemoryError(f"cannot map {nbytes} bytes for the CPU chunk pool: {exc}") from exc
    return torch.frombuffer(buffer, dtype=torch.uint8)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class PooledChunkGroupCache:
    """Sparse chunk cache for one zarr channel group, in one pooled device tensor.

    pool: uint8 (capacity, C, 34, 34, 34) on the device; slots [0, used) hold
    loaded chunks. chunk_table: int32 (cZ, cY, cX) on the device, the slot of
    each chunk or -1 if not loaded.

    Same interface and reads as the CUDA cache: prefetch() records the chunks a
    sampling step will need, sync() reads them with one batched call to the
    opened ZarrReader and copies them into the pool. Chunks are never evicted;
    when the pool is full, sync() reallocates it with 25 % headroom and copies
    the loaded slots over (slot numbers are kept), so memory grows with the
    chunks touched, as on CUDA.
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
        self.backend, self.backend_label = resolve_backend(self.device)
        Z, Y, X = self.vol_shape_zyx
        self.chunk_grid = (
            (Z + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
            (Y + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
            (X + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
        )
        cZ, cY, cX = self.chunk_grid
        self.chunk_table = torch.full((cZ, cY, cX), _ops.EMPTY, dtype=torch.int32, device=self.device)
        # Host copy of the table. The torch-ops prefetch runs on it off the CPU device, and sync()
        # updates it and then copies the whole table to the device. On MPS that keeps every device op
        # at a fixed shape: data-dependent shapes there compile and cache a new graph each time.
        self._table_host = (self.chunk_table if self.device.type == "cpu"
                            else torch.full((cZ, cY, cX), _ops.EMPTY, dtype=torch.int32))
        self.pool = torch.empty((0, self.n_channels, _PADDED, _PADDED, _PADDED), dtype=torch.uint8,
                                device=self.device)
        self._used = 0
        self._pending: dict[tuple[int, int, int], None] = {}  # in request order

        # Stats: accumulated over entire optimization
        self._iter_count: int = 0
        self._total_new_chunks: int = 0
        self._total_fetch_ms: float = 0.0
        self._last_sync_new: int = 0  # chunks from most recent sync
        self._pool_grows: int = 0

        table_mib = cZ * cY * cX * 4 / 1024**2
        print(f"[sparse_cache] {','.join(channels)}: chunk_grid={cZ}x{cY}x{cX} "
              f"vol={Z}x{Y}x{X} table={table_mib:.1f}MiB device={self.device} "
              f"sampler={self.backend_label}", flush=True)

    def prefetch(self, xyz_fullres: torch.Tensor, origin: tuple[float, float, float],
                 spacing: tuple[float, float, float]) -> None:
        """Record the chunks needed for given sample positions; sync() loads them.

        xyz_fullres: (..., 3) float32 on the cache's device — sample positions in fullres coords.
        """
        with torch.no_grad():
            if self.backend is _ops:
                # Torch ops: compute on the host (bitwise the same results; no MPS graphs per shape).
                origin_t = torch.tensor(origin, dtype=torch.float32)
                spacing_t = torch.tensor(spacing, dtype=torch.float32)
                coords = _ops.missing_chunks(xyz_fullres.detach().cpu(), self._table_host, origin_t, spacing_t)
            else:
                dev = xyz_fullres.device
                origin_t = torch.tensor(origin, dtype=torch.float32, device=dev)
                spacing_t = torch.tensor(spacing, dtype=torch.float32, device=dev)
                coords = self.backend.missing_chunks(
                    xyz_fullres, self.chunk_table, origin_t, spacing_t).cpu()
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

    def _reserve(self, slots: int) -> None:
        """Make the pool hold at least `slots` slots, keeping the loaded ones.

        Growing copies the pool, so old and new briefly coexist. Each allocation keeps 25 % + _HEADROOM_SLOTS
        of headroom: the trickle of chunks after a big first sync then fits without another copy, and slack
        stays at most a quarter of the pool.
        """
        capacity = int(self.pool.shape[0])
        if slots <= capacity:
            return
        new_capacity = max(slots + slots // 4 + _HEADROOM_SLOTS, capacity + capacity // 4, _MIN_SLOTS)
        shape = (new_capacity, *self.pool.shape[1:])
        if self.device.type == "cpu":
            pool = _host_buffer(new_capacity * self.n_channels * _PADDED**3).view(shape)
        else:
            pool = torch.empty(shape, dtype=torch.uint8, device=self.device)
        if self._used:
            pool[:self._used].copy_(self.pool[:self._used])
        self.pool = pool
        self._pool_grows += 1
        if self.device.type == "mps":
            torch.mps.empty_cache()  # hand the old pool's block back instead of keeping it cached

    def sync(self) -> None:
        """Read pending chunks into the pool and point the chunk table at them."""
        if not self._pending:
            self._last_sync_new = 0
            return

        t0 = time.perf_counter()
        coords = list(self._pending)
        self._pending = {}
        cZ, cY, cX = self.chunk_grid
        outside = [c for c in coords if not (0 <= c[0] < cZ and 0 <= c[1] < cY and 0 <= c[2] < cX)]
        if outside:
            # Never index_put outside the table (MPS does not bounds-check it).
            raise RuntimeError(
                f"sparse cache {','.join(self.channels)}: {len(outside)} of {len(coords)} pending chunks "
                f"lie outside the chunk grid {self.chunk_grid} (first {outside[:4]}); the prefetch "
                f"backend ({self.backend_label}) returned invalid coordinates")
        n = len(coords)
        C = self.n_channels
        chunk_bytes = C * _PADDED**3  # bytes per chunk
        start = self._used
        try:
            self._reserve(start + n)
            if self.device.type == "cpu":
                host = self.pool[start:start + n]  # read straight into the pool
            else:
                host = torch.empty(n, C, _PADDED, _PADDED, _PADDED, dtype=torch.uint8)
            buffers = host.numpy()
            reads = [read for i, coord in enumerate(coords) for read in self._regions(coord, buffers[i])]
            self.reader.read_regions([read[:2] for read in reads], out=[read[2] for read in reads])
            if self.device.type != "cpu":
                self.pool[start:start + n].copy_(host)
        except (torch.OutOfMemoryError, MemoryError):
            print(f"[sparse_cache_oom] sync: pending_chunks={n} pending={_fmt_mib(n * chunk_bytes)} "
                  f"{self.cache_diagnostics()}", flush=True)
            raise

        idx = torch.tensor(coords, dtype=torch.long)
        self._table_host[idx[:, 0], idx[:, 1], idx[:, 2]] = torch.arange(start, start + n, dtype=torch.int32)
        if self._table_host is not self.chunk_table:
            self.chunk_table.copy_(self._table_host)  # one fixed-shape copy, whatever n is
        self._used = start + n

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
              f"total={total} ({total_mib:.1f}MiB) pool_slots={int(self.pool.shape[0])} "
              f"pool_grows={self._pool_grows}", flush=True)

    def cache_diagnostics(self) -> str:
        """One line about device memory held by this cache, for out-of-memory reports."""
        table_bytes = self.chunk_table.numel() * self.chunk_table.element_size()
        pool_bytes = self.pool.numel() * self.pool.element_size()
        return (
            f"channels={','.join(self.channels)} device={self.device} "
            f"chunk_grid={self.chunk_grid[0]}x{self.chunk_grid[1]}x{self.chunk_grid[2]} "
            f"loaded_chunks_tracked={self._total_new_chunks} "
            f"last_sync_new={self._last_sync_new} pending_chunks={len(self._pending)} "
            f"pool_slots={int(self.pool.shape[0])} used_slots={self._used} table={_fmt_mib(table_bytes)} "
            f"pool={_fmt_mib(pool_bytes)}"
        )

    def grid_sample(self, xyz_fullres: torch.Tensor, origin: torch.Tensor,
                    inv_scale: torch.Tensor, *, diff: bool = False,
                    context: str = "") -> torch.Tensor:
        """Sample from the chunk pool.

        Returns (C, D, H, W) — uint8 for non-diff, float32 for diff.
        """
        check_enabled = os.environ.get("LASAGNA_CHECK_SPARSE_CACHE", "0") != "0"
        if check_enabled:
            self._check_sample_chunks_loaded(xyz_fullres, origin, inv_scale, context=context)
        if diff:
            out = _PoolSampleDiff.apply(self, xyz_fullres, origin, inv_scale)
        else:
            out = self.backend.sample_u8(self.pool, self.chunk_table, xyz_fullres, origin, inv_scale)
        if check_enabled:
            try:
                _synchronize(self.device)
            except RuntimeError as exc:
                shape = tuple(int(v) for v in xyz_fullres.shape)
                ctx = f" context={context}" if context else ""
                raise RuntimeError(
                    f"sparse {self.device.type} sample failed after cache coverage check: "
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
        """Fail before sampling when in-volume sample chunks were not prefetched."""
        cZ, cY, cX = self.chunk_grid
        with torch.no_grad():
            # On the CPU: this debug guard uses nonzero, which is unreliable on MPS (torch 2.11).
            flat = xyz_fullres.detach().reshape(-1, 3).cpu()
            local = (flat - origin.detach().cpu().view(1, 3)) * inv_scale.detach().cpu().view(1, 3)
            finite = torch.isfinite(local).all(dim=1)
            if not bool(finite.all().detach().cpu()):
                bad_idx = (~finite).nonzero(as_tuple=False).flatten()
                first = bad_idx[:8]
                ctx = f" context={context}" if context else ""
                raise RuntimeError(
                    f"non-finite sparse sample coordinates before {self.device.type} sample: "
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
            loaded = self._table_host[ci_z[idx], ci_y[idx], ci_x[idx]] >= 0
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
            ctx = f" context={context}" if context else ""
            raise RuntimeError(
                f"sparse chunk cache miss before {self.device.type} sample: "
                f"channels={','.join(self.channels)}{ctx} "
                f"missing={n_missing}/{n_in} in-volume samples "
                f"total_samples={total} loaded_chunks={self.loaded_chunks()} "
                f"chunk_grid={cZ}x{cY}x{cX} "
                f"first_chunks_zyx={first_chunks.tolist()} "
                f"first_local_xyz={first_local.tolist()} "
                f"first_full_xyz={first_full.tolist()}. "
                "Unset LASAGNA_CHECK_SPARSE_CACHE or set it to 0 to disable this debug guard."
            )

    def loaded_chunks(self) -> int:
        """Number of chunks currently loaded."""
        return int((self._table_host >= 0).sum())

    def loaded_mib(self) -> float:
        """MiB of chunk data currently loaded."""
        n = self.loaded_chunks()
        return n * self.n_channels * _PADDED**3 / 1024**2
