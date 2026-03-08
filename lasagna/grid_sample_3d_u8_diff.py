"""Differentiable CUDA uint8 3D grid_sample with gradients w.r.t. sample positions."""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load as _load_ext

_module = None


def _get_module():
    global _module
    if _module is None:
        src = str(Path(__file__).with_name("grid_sample_3d_u8_diff_kernel.cu"))
        _module = _load_ext(
            name="grid_sample_3d_u8_diff_ext",
            sources=[src],
            extra_cflags=["-DGLOG_USE_GLOG_EXPORT"],
            extra_cuda_cflags=["-DGLOG_USE_GLOG_EXPORT"],
            verbose=False,
        )
    return _module


class _GridSample3DU8Diff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, grid, offset, inv_scale):
        mod = _get_module()
        out = mod.grid_sample_3d_u8_diff_fwd(volume, grid, offset, inv_scale)
        ctx.save_for_backward(grid)
        ctx.volume = volume
        ctx.offset = offset
        ctx.inv_scale = inv_scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grid, = ctx.saved_tensors
        mod = _get_module()
        grad_grid = mod.grid_sample_3d_u8_diff_bwd(
            ctx.volume, grid, ctx.offset, ctx.inv_scale,
            grad_output.contiguous()
        )
        return None, grad_grid, None, None


def grid_sample_3d_u8_diff(
    volume: torch.Tensor,
    grid: torch.Tensor,
    offset: torch.Tensor,
    inv_scale: torch.Tensor,
) -> torch.Tensor:
    """Trilinear 3D grid_sample on uint8 volume, differentiable w.r.t. grid positions.

    Args:
        volume: (C, Z, Y, X) uint8 CUDA
        grid: (D, H, W, 3) float32 CUDA — fullres coordinates (x, y, z)
        offset: (3,) float32 CUDA — origin in fullres coords
        inv_scale: (3,) float32 CUDA — 1.0 / spacing per axis

    Returns:
        (C, D, H, W) float32 CUDA — raw interpolated values (not decoded)
    """
    return _GridSample3DU8Diff.apply(volume, grid, offset, inv_scale)
