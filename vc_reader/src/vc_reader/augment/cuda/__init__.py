"""Availability of the optional CUDA augmentation backend.

Use ``augment.Pipeline(transforms, backend="cuda")`` to require CUDA, or
``backend="auto"`` to use CUDA when available and CPU otherwise. Inputs and
outputs retain the public NumPy / CPU Torch tensor API.
"""
from ... import _augment


def is_available() -> bool:
    """Whether this build has CUDA support and a visible CUDA device."""
    return _augment.cuda_available()


__all__ = ["is_available"]
