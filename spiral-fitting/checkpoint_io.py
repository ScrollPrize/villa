"""Memory-conscious checkpoint loading helpers for Spiral fits."""

from __future__ import annotations

import torch


def load_checkpoint_cpu(path):
    """Load a modern checkpoint with lazily mapped CPU tensor storages."""
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
