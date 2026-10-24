"""Availability of the optional CUDA transform backend.

Use ``transforms.Pipeline(operations, backend="cuda")`` to require CUDA, or
``backend="auto"`` to use CUDA when available and CPU otherwise. DeviceArray
inputs and outputs remain on device; NumPy and CPU Torch inputs return to host.
"""
from .. import _transforms


def is_available() -> bool:
    """Whether this build has CUDA support and a visible CUDA device."""
    return _transforms.cuda_available()


def pool_stats(device: int = 0) -> dict:
    """Bytes reserved and used in the transform scratch pool, currently and at their peak."""
    return dict(_transforms._pool_stats(int(device)))


__all__ = ["is_available", "pool_stats"]
