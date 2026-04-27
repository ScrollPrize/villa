"""Sparse chunk cache 3D grid_sample for uint8 volumes (non-differentiable)."""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load as _load_ext

_module = None


def _get_module():
    global _module
    if _module is None:
        src = str(Path(__file__).with_name("sparse_grid_sample_3d_u8_kernel.cu"))
        _module = _load_ext(
            name="sparse_grid_sample_3d_u8_ext",
            sources=[src],
            extra_cflags=["-DGLOG_USE_GLOG_EXPORT"],
            extra_cuda_cflags=["-DGLOG_USE_GLOG_EXPORT"],
            verbose=False,
        )
    return _module


def sparse_grid_sample_3d_u8(
    chunk_table: torch.Tensor,
    C: int,
    grid: torch.Tensor,
    offset: torch.Tensor,
    inv_scale: torch.Tensor,
) -> torch.Tensor:
    """Trilinear 3D grid_sample from sparse chunk cache.

    Args:
        chunk_table: (cZ, cY, cX) int64 CUDA — device pointers (0 = empty)
        C: number of channels per chunk
        grid: (D, H, W, 3) float32 CUDA — fullres coordinates (x, y, z)
        offset: (3,) float32 CUDA — origin in fullres coords
        inv_scale: (3,) float32 CUDA — 1.0 / spacing per axis

    Returns:
        (C, D, H, W) uint8 CUDA
    """
    return _get_module().sparse_grid_sample_3d_u8(chunk_table, C, grid, offset, inv_scale)
