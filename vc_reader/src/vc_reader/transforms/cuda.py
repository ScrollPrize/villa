"""Availability of the optional CUDA transform backend.

Use ``transforms.Pipeline(operations, backend="cuda")`` to require CUDA, or
``backend="auto"`` to use CUDA when available and CPU otherwise. DeviceArray
inputs and outputs remain on device; NumPy and CPU Torch inputs return to host.
"""
from .. import _transforms


def is_available() -> bool:
    """Whether this build has CUDA support and a visible CUDA device."""
    return _transforms.cuda_available()


__all__ = ["is_available"]
