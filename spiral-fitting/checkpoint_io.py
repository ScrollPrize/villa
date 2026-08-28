"""Memory-conscious checkpoint loading helpers for Spiral fits."""

from __future__ import annotations

import torch

from checkpoint_migrations import migrate_legacy_gap_parameterization


def load_checkpoint_cpu(path):
    """Load a modern checkpoint with lazily mapped CPU tensor storages."""
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    return migrate_legacy_gap_parameterization(checkpoint)
