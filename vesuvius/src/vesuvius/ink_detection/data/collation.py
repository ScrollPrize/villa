"""Collation for flat ink training after per-sample augmentation."""

from __future__ import annotations

import torch
from torch.utils.data import default_collate


def collate_flat_ink_samples(samples):
    """Keep only the depth maxima consumed by the flat loss for both labels.

    Image and auxiliary fields retain their original shapes. Dataset samples
    remain CZYX; projection happens after augmentation and before pinning or
    transferring a batch to the accelerator.
    """
    projected = []
    for sample in samples:
        values = dict(sample)
        for key in ("inklabels", "supervision_mask"):
            labels = values[key]
            if labels.ndim != 4:
                raise ValueError("Flat ink labels must have shape CZYX")
            values[key] = torch.amax(labels, dim=1, keepdim=True)
        projected.append(values)
    return default_collate(projected)


def ink_collate_fn(mode: str):
    """Leave native 3D collation on the default PyTorch path."""
    return collate_flat_ink_samples if mode == "flat" else None
