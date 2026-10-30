"""MPS host-to-device copy safety for the ink inference loops.

On torch 2.11 and 2.12, a non_blocking host-to-MPS copy can read its CPU source
after the source has been freed (pytorch/pytorch#189690). Both inference loops
drop their host batch right after the copy, so each model below first fills
same-size host tensors with a sentinel and then checks that its MPS input still
holds the batch it was given.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vesuvius.ink_detection.inference import infer_full3d_tifxyz
from vesuvius.ink_detection.inference.infer import run_block_inference


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)

SENTINEL = -7777.0


class _IndexedPatches(Dataset):
    """Patch i is filled with i + 1, so any foreign value is detectable."""

    def __init__(self, count, shape, metadata):
        self.count = count
        self.shape = shape
        self.metadata = metadata

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return torch.full(self.shape, float(index + 1)), self.metadata(index)


class _HostOverwritingModel(nn.Module):
    """Overwrite freed host memory with SENTINEL, then record the input range."""

    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.ranges = []

    def forward(self, images):
        scratch = [torch.empty(images.shape).fill_(SENTINEL) for _ in range(4)]
        dims = tuple(range(1, images.ndim))
        ranges = torch.stack([images.amin(dim=dims), images.amax(dim=dims)], dim=1)
        self.ranges.extend(ranges.cpu().tolist())
        del scratch
        return torch.zeros(
            (images.shape[0], *self.output_shape), device=images.device
        )


def _assert_inputs_intact(ranges, count):
    assert len(ranges) == count
    corrupted = [
        index
        for index, (low, high) in enumerate(ranges)
        if low != index + 1 or high != index + 1
    ]
    assert corrupted == [], (
        f"{len(corrupted)} of {count} MPS inputs differ from their host batch"
    )


def test_flat_block_inference_keeps_mps_inputs_intact():
    count, patch = 32, 128
    dataset = _IndexedPatches(
        count,
        (1, 17, patch, patch),
        lambda index: torch.tensor([0, 0, patch, patch, 1]),
    )
    model = _HostOverwritingModel((1, patch, patch))
    run_block_inference(
        loader=DataLoader(dataset, batch_size=1),
        model=model,
        accumulator=SimpleNamespace(add_tile=lambda **kwargs: None),
        weight_map=np.ones((patch, patch), dtype=np.float32),
        mask=None,
        device=torch.device("mps"),
        amp_dtype=None,
        tta_axes=(),
        tta_batch_size=None,
    )
    _assert_inputs_intact(model.ranges, count)


def test_native_full3d_inference_keeps_mps_inputs_intact(monkeypatch):
    count, patch_zyx = 32, (32, 64, 64)
    monkeypatch.setattr(
        infer_full3d_tifxyz,
        "ChunkAccumulator3D",
        lambda **kwargs: SimpleNamespace(
            add_patch=lambda **kw: None, flush_remaining=lambda: None
        ),
    )
    dataset = _IndexedPatches(
        count,
        (1, *patch_zyx),
        lambda index: torch.zeros(3, dtype=torch.int64),
    )
    model = _HostOverwritingModel((1, *patch_zyx))
    infer_full3d_tifxyz.run_native_inference(
        args=SimpleNamespace(
            batch_size=1,
            gpu_ids=(),
            num_workers=0,
            prefetch_factor=2,
            tta=False,
            tta_batch_size=None,
        ),
        bundle=SimpleNamespace(
            model=model, device=torch.device("mps"), amp_dtype=None
        ),
        dataset=dataset,
        plan=SimpleNamespace(
            target_chunks=frozenset(),
            contribution_counts={},
            chunk_shape_zyx=patch_zyx,
            patch_size_zyx=patch_zyx,
        ),
        output=None,
        importance_map=np.ones(patch_zyx, dtype=np.float32),
    )
    _assert_inputs_intact(model.ranges, count)
