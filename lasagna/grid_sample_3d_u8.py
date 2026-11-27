"""uint8 3D grid_sample with built-in coordinate transform.

CUDA runs the JIT-built kernel; MPS the Metal port (``u8_sample_metal``); other devices the PyTorch reference.
"""

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load as _load_ext

_module = None


def _get_module():
    global _module
    if _module is None:
        src = str(Path(__file__).with_name("grid_sample_3d_u8_kernel.cu"))
        _module = _load_ext(
            name="grid_sample_3d_u8_ext",
            sources=[src],
            extra_cflags=["-DGLOG_USE_GLOG_EXPORT"],
            extra_cuda_cflags=["-DGLOG_USE_GLOG_EXPORT"],
            verbose=False,
        )
    return _module


def grid_sample_3d_u8(
    volume: torch.Tensor,
    grid: torch.Tensor,
    offset: torch.Tensor,
    inv_scale: torch.Tensor,
) -> torch.Tensor:
    """Trilinear 3D grid_sample on uint8 volume.

    Args:
        volume: (C, Z, Y, X) uint8 on CUDA, MPS or the CPU
        grid: (D, H, W, 3) float32, same device — fullres coordinates (x, y, z)
        offset: (3,) float32 — origin in fullres coords
        inv_scale: (3,) float32 — 1.0 / spacing per axis

    Returns:
        (C, D, H, W) uint8
    """
    if volume.device.type == "cuda":
        return _get_module().grid_sample_3d_u8(volume, grid, offset, inv_scale)
    if volume.device.type == "mps" and os.environ.get("LASAGNA_METAL", "1") != "0":
        from u8_sample_metal import dense_sample
    else:
        from u8_sample_torch import dense_sample
    return dense_sample(volume, grid, offset, inv_scale, diff=False)
