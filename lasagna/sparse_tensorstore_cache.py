"""Sparse GPU chunk cache backed by TensorStore's Python API."""
from __future__ import annotations

import os
import time

import numpy as np
import tensorstore as ts
import torch

_CHUNK_SIZE = 32
_PADDED = _CHUNK_SIZE + 2


class TensorStoreSparseChunkGroupCache:
    """PyTorch-facing sparse cache using TensorStore for parallel chunk reads."""

    def __init__(
        self,
        *,
        channels: list[str],
        zarr_path: str,
        vol_shape_zyx: tuple[int, int, int],
        channel_indices: dict[str, int],
        is_3d_zarr: bool,
        device: torch.device,
        cache_pool_bytes: int = 8 << 30,
        file_io_threads: int = 16,
        data_copy_threads: int = 8,
    ) -> None:
        self.channels = channels
        self.zarr_path = zarr_path
        self.n_channels = len(channels)
        self.vol_shape_zyx = tuple(int(v) for v in vol_shape_zyx)
        self.channel_indices = channel_indices
        self.is_3d_zarr = bool(is_3d_zarr)
        self.device = device

        Z, Y, X = self.vol_shape_zyx
        self.chunk_grid = (
            (Z + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
            (Y + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
            (X + _CHUNK_SIZE - 1) // _CHUNK_SIZE,
        )
        cZ, cY, cX = self.chunk_grid

        self.chunk_table = torch.zeros(cZ, cY, cX, dtype=torch.int64, device=device)
        self._batches: list[torch.Tensor] = []
        self._pending: list[tuple[int, int, int, list[object]]] = []
        self._pending_keys: set[tuple[int, int, int]] = set()
        self._transfer_stream = torch.cuda.Stream(device=device)

        self._context = ts.Context({
            "cache_pool": {"total_bytes_limit": int(cache_pool_bytes)},
            "file_io_concurrency": {"limit": int(file_io_threads)},
            "data_copy_concurrency": {"limit": int(data_copy_threads)},
        })
        self._store = ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": str(zarr_path)},
            },
            context=self._context,
            open=True,
            read=True,
            recheck_cached_data="open",
        ).result()

        self._iter_count = 0
        self._total_new_chunks = 0
        self._total_fetch_ms = 0.0
        self._last_sync_new = 0

        table_mib = cZ * cY * cX * 8 / 1024**2
        print(f"[sparse_cache_ts] {','.join(channels)}: chunk_grid={cZ}x{cY}x{cX} "
              f"vol={Z}x{Y}x{X} table={table_mib:.1f}MiB "
              f"cache_pool={cache_pool_bytes / 1024**2:.0f}MiB "
              f"file_io={file_io_threads} data_copy={data_copy_threads} "
              f"recheck_cached_data=open", flush=True)

    def prefetch(self, xyz_fullres: torch.Tensor, origin: tuple[float, float, float],
                 spacing: tuple[float, float, float]) -> None:
        dev = xyz_fullres.device
        origin_t = torch.tensor(origin, dtype=torch.float32, device=dev)
        spacing_t = torch.tensor(spacing, dtype=torch.float32, device=dev)

        from sparse_prefetch_chunks import missing_chunks
        coords = missing_chunks(xyz_fullres, self.chunk_table, origin_t, spacing_t)
        if coords.numel() == 0:
            return

        coords_cpu = coords.detach().cpu().contiguous()
        batch = ts.Batch()
        submitted = 0
        for i in range(coords_cpu.shape[0]):
            cz, cy, cx = (int(coords_cpu[i, 0]), int(coords_cpu[i, 1]), int(coords_cpu[i, 2]))
            key = (cz, cy, cx)
            if key in self._pending_keys:
                continue
            futures = self._submit_reads(cz, cy, cx, batch)
            self._pending.append((cz, cy, cx, futures))
            self._pending_keys.add(key)
            submitted += 1
        if submitted == 0:
            return
        batch.submit()

    def sync(self) -> None:
        if not self._pending:
            self._last_sync_new = 0
            return

        t0 = time.perf_counter()
        pending = self._pending
        self._pending = []
        self._pending_keys.clear()

        n = len(pending)
        C = self.n_channels
        cpu_batch = torch.empty(n, C, _PADDED, _PADDED, _PADDED, dtype=torch.uint8,
                                pin_memory=True)
        coords_list: list[tuple[int, int, int]] = []
        for i, (cz, cy, cx, futures) in enumerate(pending):
            cpu_batch[i].zero_()
            self._finish_chunk(cz, cy, cx, futures, cpu_batch[i].numpy())
            coords_list.append((cz, cy, cx))

        with torch.cuda.stream(self._transfer_stream):
            gpu_batch = cpu_batch.to(self.device, non_blocking=True)
        self._transfer_stream.synchronize()
        self._batches.append(gpu_batch)

        chunk_bytes = C * _PADDED * _PADDED * _PADDED
        base_ptr = gpu_batch.data_ptr()
        for i, (cz, cy, cx) in enumerate(coords_list):
            self.chunk_table[cz, cy, cx] = base_ptr + i * chunk_bytes

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._last_sync_new = n
        self._total_new_chunks += n
        self._total_fetch_ms += dt_ms

    def end_iteration(self) -> None:
        self._iter_count += 1

    def print_summary(self) -> None:
        n = self._total_new_chunks
        ms = self._total_fetch_ms
        its = self._iter_count
        ms_per_it = ms / its if its > 0 else 0.0
        ms_per_chunk = ms / n if n > 0 else 0.0
        total = self.loaded_chunks()
        total_mib = self.loaded_mib()
        print(f"[sparse_cache_ts] {','.join(self.channels)}: "
              f"{n} chunks in {its}it ({ms_per_it:.1f}ms/it, {ms_per_chunk:.1f}ms/chunk) "
              f"total={total} ({total_mib:.1f}MiB)", flush=True)

    def grid_sample(self, xyz_fullres: torch.Tensor, origin: torch.Tensor,
                    inv_scale: torch.Tensor, *, diff: bool = False,
                    context: str = "") -> torch.Tensor:
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

    def loaded_chunks(self) -> int:
        return int((self.chunk_table != 0).sum())

    def loaded_mib(self) -> float:
        return self.loaded_chunks() * self.n_channels * _PADDED**3 / 1024**2

    def _chunk_bounds(self, cz: int, cy: int, cx: int) -> tuple[slice, slice, slice, slice, slice, slice]:
        Z, Y, X = self.vol_shape_zyx
        gz0 = cz * _CHUNK_SIZE - 1
        gy0 = cy * _CHUNK_SIZE - 1
        gx0 = cx * _CHUNK_SIZE - 1
        rz0 = max(0, gz0)
        ry0 = max(0, gy0)
        rx0 = max(0, gx0)
        rz1 = min(Z, gz0 + _PADDED)
        ry1 = min(Y, gy0 + _PADDED)
        rx1 = min(X, gx0 + _PADDED)
        dz0 = rz0 - gz0
        dy0 = ry0 - gy0
        dx0 = rx0 - gx0
        src_z = slice(rz0, rz1)
        src_y = slice(ry0, ry1)
        src_x = slice(rx0, rx1)
        dst_z = slice(dz0, dz0 + (rz1 - rz0))
        dst_y = slice(dy0, dy0 + (ry1 - ry0))
        dst_x = slice(dx0, dx0 + (rx1 - rx0))
        return src_z, src_y, src_x, dst_z, dst_y, dst_x

    def _submit_reads(self, cz: int, cy: int, cx: int, batch: ts.Batch) -> list[object]:
        src_z, src_y, src_x, _, _, _ = self._chunk_bounds(cz, cy, cx)
        if src_z.stop <= src_z.start or src_y.stop <= src_y.start or src_x.stop <= src_x.start:
            return []
        if self.is_3d_zarr:
            return [self._store[src_z, src_y, src_x].read(order="C", batch=batch)]
        futures = []
        for ch in self.channels:
            ch_idx = int(self.channel_indices[ch])
            futures.append(self._store[ch_idx, src_z, src_y, src_x].read(order="C", batch=batch))
        return futures

    def _finish_chunk(
        self,
        cz: int,
        cy: int,
        cx: int,
        futures: list[object],
        dst: np.ndarray,
    ) -> None:
        _, _, _, dst_z, dst_y, dst_x = self._chunk_bounds(cz, cy, cx)
        if not futures:
            return
        if self.is_3d_zarr:
            dst[0, dst_z, dst_y, dst_x] = np.asarray(futures[0].result(), dtype=np.uint8)
            return
        for ch_i, fut in enumerate(futures):
            dst[ch_i, dst_z, dst_y, dst_x] = np.asarray(fut.result(), dtype=np.uint8)

    def _check_sample_chunks_loaded(
        self,
        xyz_fullres: torch.Tensor,
        origin: torch.Tensor,
        inv_scale: torch.Tensor,
        *,
        context: str = "",
    ) -> None:
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
