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
    if volume.device.type != "cuda":
        return _grid_sample_3d_u8_diff_cpu(volume, grid, offset, inv_scale)
    return _GridSample3DU8Diff.apply(volume, grid, offset, inv_scale)


def _grid_sample_3d_u8_diff_cpu(
    volume: torch.Tensor,
    grid: torch.Tensor,
    offset: torch.Tensor,
    inv_scale: torch.Tensor,
) -> torch.Tensor:
    """CPU fallback: convert uint8 to float32 and use F.grid_sample."""
    import torch.nn.functional as F
    C, Z, Y, X = volume.shape
    vol_f = volume.float().unsqueeze(0)  # (1, C, Z, Y, X)
    # Convert fullres coords to grid_sample [-1, 1] range
    g = grid.clone()  # (D, H, W, 3) — (x, y, z)
    g[..., 0] = (g[..., 0] - offset[0]) * inv_scale[0] / max(1, X - 1) * 2 - 1
    g[..., 1] = (g[..., 1] - offset[1]) * inv_scale[1] / max(1, Y - 1) * 2 - 1
    g[..., 2] = (g[..., 2] - offset[2]) * inv_scale[2] / max(1, Z - 1) * 2 - 1
    # F.grid_sample 5D: grid[...,0]→W, grid[...,1]→H, grid[...,2]→D
    # Our (x,y,z) already maps to (W=X, H=Y, D=Z) — no reorder needed
    out = F.grid_sample(
        vol_f, g.unsqueeze(0),
        mode='bilinear', padding_mode='zeros', align_corners=True,
    )
    # (1, C, D, H, W) → (C, D, H, W)
    return out.squeeze(0)
