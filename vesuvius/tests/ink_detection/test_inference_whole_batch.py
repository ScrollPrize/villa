"""Whole flat batches go to the device as the loader gave them.

`run_block_inference` drops empty blocks with a boolean index. Indexing copies the
batch into new pageable memory: the DataLoader's pinned buffer is lost, so the copy to
the GPU blocks, and the index runs as a parallel CPU op once per batch. Block
selection already drops empty patches, so batches are usually whole; those skip the
index.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from vesuvius.ink_detection.inference.infer import run_block_inference

PATCH = 16
SENTINEL = -7777.0
# GPU spin queued before each host-to-device copy, so the copy is still pending when
# the loop drops the host batch (about 10 ms on a laptop GPU).
SPIN_CYCLES = 20_000_000


def _batch(occupancy, *, first_value=0.0, pin=False):
    """Block i is filled with first_value + i; it sits at x0 = i * PATCH."""

    count = len(occupancy)
    values = first_value + torch.arange(count, dtype=torch.float32)
    images = values.view(count, 1, 1, 1, 1).expand(count, 1, 2, PATCH, PATCH).contiguous()
    metadata = torch.tensor(
        [[0, i * PATCH, PATCH, PATCH, int(occupied)] for i, occupied in enumerate(occupancy)]
    )
    return (images.pin_memory() if pin else images), metadata


class _RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, images):
        self.inputs.append(images)
        return torch.zeros((images.shape[0], 1, PATCH, PATCH), device=images.device)


class _RecordingAccumulator:
    def __init__(self):
        self.calls = []

    def add_tile(self, *, y0, x0, tile, tile_weights):
        self.calls.append(("add", y0, x0))

    def skip_tile(self, *, y0, x0, valid_h, valid_w):
        self.calls.append(("skip", y0, x0))


def _run(loader, model, accumulator, device):
    run_block_inference(
        loader=loader,
        model=model,
        accumulator=accumulator,
        weight_map=np.ones((PATCH, PATCH), dtype=np.float32),
        mask=None,
        device=device,
        amp_dtype=None,
        tta_axes=(),
        tta_batch_size=None,
    )


def test_whole_batch_reaches_the_model_without_a_host_copy():
    images, metadata = _batch([1, 1, 1])
    model, accumulator = _RecordingModel(), _RecordingAccumulator()
    _run([(images, metadata)], model, accumulator, torch.device("cpu"))
    assert len(model.inputs) == 1
    assert model.inputs[0].data_ptr() == images.data_ptr()
    assert accumulator.calls == [("add", 0, 0), ("add", 0, PATCH), ("add", 0, 2 * PATCH)]


def test_partial_batch_forwards_only_occupied_blocks_in_order():
    images, metadata = _batch([1, 0, 1])
    model, accumulator = _RecordingModel(), _RecordingAccumulator()
    _run([(images, metadata)], model, accumulator, torch.device("cpu"))
    assert torch.equal(model.inputs[0][:, 0, 0, 0, 0], torch.tensor([0.0, 2.0]))
    assert accumulator.calls == [("skip", 0, PATCH), ("add", 0, 0), ("add", 0, 2 * PATCH)]


class _HostOverwritingModel(_RecordingModel):
    """Grab freed pinned host memory, fill it with SENTINEL, record the input range."""

    def __init__(self):
        super().__init__()
        self.ranges = []
        self.copy_pending = []

    def forward(self, images):
        scratch = [torch.empty(images.shape).pin_memory().fill_(SENTINEL) for _ in range(4)]
        self.copy_pending.append(not torch.cuda.current_stream().query())
        dims = tuple(range(1, images.ndim))
        self.ranges.extend(
            torch.stack([images.amin(dim=dims), images.amax(dim=dims)], dim=1).cpu().tolist()
        )
        del scratch
        return super().forward(images)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_pinned_whole_batches_reach_cuda_intact():
    """The loop drops the pinned host batch right after its non_blocking copy."""

    batches, size = 32, 4

    def loader():
        for index in range(batches):
            torch.cuda._sleep(SPIN_CYCLES)
            yield _batch([1] * size, first_value=float(index * size), pin=True)

    model = _HostOverwritingModel()
    _run(loader(), model, _RecordingAccumulator(), torch.device("cuda"))
    assert model.ranges == [[float(i), float(i)] for i in range(batches * size)]
    if not any(model.copy_pending):
        pytest.skip("inconclusive: every copy landed before the host memory was reused")
