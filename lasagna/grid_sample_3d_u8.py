"""Custom CUDA uint8 3D grid_sample with built-in coordinate transform (Metal on MPS, torch on CPU)."""

import importlib
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
        volume: (C, Z, Y, X) uint8 CUDA
        grid: (D, H, W, 3) float32 CUDA — fullres coordinates (x, y, z)
        offset: (3,) float32 CUDA — origin in fullres coords
        inv_scale: (3,) float32 CUDA — 1.0 / spacing per axis

    Returns:
        (C, D, H, W) uint8 CUDA
    """
    if volume.device.type == "mps":
        # Metal kernels, or torch ops under LASAGNA_METAL=0 (F.grid_sample has no 3D backward on MPS).
        return _ops().mps_dense_backend()[0].grid_sample_3d_u8(volume, grid, offset, inv_scale)
    if volume.device.type != "cuda":
        # Devices without a kernel (CPU): the torch ops, which follow the kernel term by term
        # (corners outside the volume read 0, half away from zero, non-finite positions read 0).
        return _ops().sample_u8(volume, grid, offset, inv_scale)
    return _get_module().grid_sample_3d_u8(volume, grid, offset, inv_scale)


def _ops():
    """grid_sample_3d_u8_ops (the dense torch ops and the MPS backend choice), imported by name: fit_data
    executes this file afresh on every call, and one module keeps one backend choice and one compiled Metal
    library per process."""
    return importlib.import_module("grid_sample_3d_u8_ops")
