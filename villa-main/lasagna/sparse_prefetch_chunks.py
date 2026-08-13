"""CUDA helper for sparse-cache missing chunk discovery."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load as _load_ext

_module = None


def _get_module():
    global _module
    if _module is None:
        src = str(Path(__file__).with_name("sparse_prefetch_chunks_kernel.cu"))
        _module = _load_ext(
            name="sparse_prefetch_chunks_ext",
            sources=[src],
            extra_cflags=["-DGLOG_USE_GLOG_EXPORT"],
            extra_cuda_cflags=["-DGLOG_USE_GLOG_EXPORT"],
            verbose=False,
        )
    return _module


def missing_chunks(
    xyz_fullres: torch.Tensor,
    chunk_table: torch.Tensor,
    origin: torch.Tensor,
    spacing: torch.Tensor,
) -> torch.Tensor:
    """Return GPU int64 (N, 3) missing chunk coordinates as (cz, cy, cx)."""
    return _get_module().missing_chunks(xyz_fullres, chunk_table, origin, spacing)
