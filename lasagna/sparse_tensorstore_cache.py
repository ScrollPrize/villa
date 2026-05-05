"""TensorStore C++ sparse chunk cache wrapper."""
from __future__ import annotations

import os
import shlex
from pathlib import Path

import torch
from torch.utils.cpp_extension import load as _load_ext

_module = None


def _split_env(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    if not raw:
        return []
    return [p for p in raw.split(os.pathsep) if p]


def _get_module():
    global _module
    if _module is not None:
        return _module

    src = str(Path(__file__).with_name("sparse_tensorstore_cache_ext.cpp"))
    include_dirs = _split_env("TENSORSTORE_INCLUDE_DIRS")
    library_dirs = _split_env("TENSORSTORE_LIBRARY_DIRS")
    libraries = os.environ.get("TENSORSTORE_LIBRARIES", "tensorstore")

    extra_cflags = ["-DGLOG_USE_GLOG_EXPORT"]
    extra_cflags += shlex.split(os.environ.get("TENSORSTORE_EXTRA_CFLAGS", ""))
    extra_ldflags = []
    for d in library_dirs:
        extra_ldflags.extend([f"-L{d}", f"-Wl,-rpath,{d}"])
    for lib in shlex.split(libraries):
        extra_ldflags.append(lib if lib.startswith("-l") else f"-l{lib}")
    extra_ldflags += shlex.split(os.environ.get("TENSORSTORE_EXTRA_LDFLAGS", ""))

    try:
        _module = _load_ext(
            name="sparse_tensorstore_cache_ext",
            sources=[src],
            extra_include_paths=include_dirs,
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to build sparse TensorStore C++ cache extension. "
            "Install TensorStore C++ development headers/libraries or set "
            "TENSORSTORE_INCLUDE_DIRS, TENSORSTORE_LIBRARY_DIRS, and "
            "TENSORSTORE_LIBRARIES. Use --sparse-prefetch-backend python for "
            "the old zarr-python fallback."
        ) from exc
    return _module


class TensorStoreSparseChunkGroupCache:
    """PyTorch-facing sparse cache using TensorStore C++ for chunk reads."""

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
        self.n_channels = len(channels)
        self.device = device
        self.channel_indices = channel_indices
        self.is_3d_zarr = bool(is_3d_zarr)
        self.vol_shape_zyx = tuple(int(v) for v in vol_shape_zyx)

        channel_index_list = [int(channel_indices[ch]) for ch in channels]
        dev_index = device.index
        if dev_index is None:
            dev_index = torch.cuda.current_device() if device.type == "cuda" else -1

        mod = _get_module()
        self._cache = mod.TensorStoreSparseChunkGroupCache(
            channels,
            str(zarr_path),
            list(self.vol_shape_zyx),
            channel_index_list,
            self.is_3d_zarr,
            int(dev_index),
            int(cache_pool_bytes),
            int(file_io_threads),
            int(data_copy_threads),
        )
        self.chunk_table = self._cache.chunk_table()

    def prefetch(self, xyz_fullres: torch.Tensor, origin: tuple[float, float, float],
                 spacing: tuple[float, float, float]) -> None:
        dev = xyz_fullres.device
        origin_t = torch.tensor(origin, dtype=torch.float32, device=dev)
        spacing_t = torch.tensor(spacing, dtype=torch.float32, device=dev)
        from sparse_prefetch_chunks import missing_chunks
        coords = missing_chunks(xyz_fullres, self.chunk_table, origin_t, spacing_t)
        if coords.numel() == 0:
            return
        self._cache.prefetch_coords(coords.cpu())

    def sync(self) -> None:
        self._cache.sync()
        self.chunk_table = self._cache.chunk_table()

    def end_iteration(self) -> None:
        self._cache.end_iteration()

    def print_summary(self) -> None:
        self._cache.print_summary()

    def grid_sample(self, xyz_fullres: torch.Tensor, origin: torch.Tensor,
                    inv_scale: torch.Tensor, *, diff: bool = False) -> torch.Tensor:
        if diff:
            from sparse_grid_sample_3d_u8_diff import sparse_grid_sample_3d_u8_diff
            return sparse_grid_sample_3d_u8_diff(
                self.chunk_table, self.n_channels, xyz_fullres, origin, inv_scale,
            )
        from sparse_grid_sample_3d_u8 import sparse_grid_sample_3d_u8
        return sparse_grid_sample_3d_u8(
            self.chunk_table, self.n_channels, xyz_fullres, origin, inv_scale,
        )

    def loaded_chunks(self) -> int:
        return int((self.chunk_table != 0).sum())

    def loaded_mib(self) -> float:
        return self._cache.loaded_mib()
