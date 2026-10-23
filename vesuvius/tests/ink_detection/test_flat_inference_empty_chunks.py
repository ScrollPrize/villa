"""Raw-empty patches complete output chunks without contributing blend weight."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vesuvius.ink_detection.inference.infer import (
    ChunkAccumulator,
    compute_chunk_contribution_counts,
    iter_blocks,
    run_block_inference,
)


def _accumulator(shape, chunk_shape):
    probability = np.zeros(shape, dtype=np.float32)
    weight = np.zeros(shape, dtype=np.float32)
    blocks = iter_blocks(shape, 2, 1)
    accumulator = ChunkAccumulator(
        shape=shape,
        chunk_shape=chunk_shape,
        prob_sum_store=probability,
        weight_sum_store=weight,
        contribution_counts=compute_chunk_contribution_counts(
            blocks, chunk_shape=chunk_shape
        ),
    )
    return accumulator, probability, weight, blocks


@pytest.mark.parametrize("skip_first", [False, True])
def test_raw_empty_tile_releases_chunk_in_either_order(skip_first):
    accumulator, probability, weight, _ = _accumulator((2, 3), (2, 3))

    def add():
        accumulator.add_tile(
            y0=0, x0=0,
            tile=np.full((2, 2), 0.25, dtype=np.float32),
            tile_weights=np.ones((2, 2), dtype=np.float32),
        )

    def skip():
        accumulator.skip_tile(y0=0, x0=1, valid_h=2, valid_w=2)

    if skip_first:
        skip()
        add()
    else:
        add()
        skip()

    assert not accumulator.buffers
    assert not accumulator.seen_counts
    np.testing.assert_array_equal(probability, [[0.25, 0.25, 0], [0.25, 0.25, 0]])
    np.testing.assert_array_equal(weight, [[1, 1, 0], [1, 1, 0]])


def test_only_raw_empty_tiles_do_not_allocate_or_write_chunks():
    accumulator, probability, weight, blocks = _accumulator((3, 5), (2, 2))

    class RejectWrites:
        def __setitem__(self, key, values):
            raise AssertionError("raw-empty chunks should keep implicit fill")

    accumulator.prob_sum_store = RejectWrites()
    accumulator.weight_sum_store = RejectWrites()
    for block in blocks:
        accumulator.skip_tile(
            y0=block.y0, x0=block.x0,
            valid_h=block.valid_h, valid_w=block.valid_w,
        )
        assert not accumulator.buffers
    accumulator.flush_remaining()
    assert not accumulator.seen_counts
    assert not probability.any()
    assert not weight.any()


class _ZeroLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.images_seen = 0

    def forward(self, images):
        self.images_seen += len(images)
        return torch.zeros(
            (len(images), 1, *images.shape[-2:]), device=images.device
        )


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("nonempty_index", [0, 1])
def test_inference_completes_skipped_tiles_without_adding_weights(
    batch_size, nonempty_index
):
    accumulator, probability, weight, _ = _accumulator((2, 3), (2, 3))
    # A raw-nonempty robust-normalized patch can itself be all zero.
    images = torch.zeros((2, 1, 1, 2, 2))
    metadata = torch.tensor([[0, 0, 2, 2, 0], [0, 1, 2, 2, 0]])
    metadata[nonempty_index, 4] = 1
    model = _ZeroLogits()
    run_block_inference(
        loader=DataLoader(TensorDataset(images, metadata), batch_size=batch_size),
        model=model, accumulator=accumulator,
        weight_map=np.ones((2, 2), dtype=np.float32), mask=None,
        device=torch.device("cpu"), amp_dtype=None, tta_axes=(),
        tta_batch_size=None,
    )
    assert model.images_seen == 1
    assert not accumulator.buffers
    assert not accumulator.seen_counts
    expected = np.zeros((2, 3), dtype=np.float32)
    expected[:, nonempty_index : nonempty_index + 2] = 1
    np.testing.assert_array_equal(weight, expected)
    np.testing.assert_array_equal(probability, expected * 0.5)


def test_all_raw_empty_batch_finishes_without_model_forward():
    accumulator, probability, weight, _ = _accumulator((2, 3), (2, 3))
    images = torch.ones((2, 1, 1, 2, 2))
    metadata = torch.tensor([[0, 0, 2, 2, 0], [0, 1, 2, 2, 0]])
    model = _ZeroLogits()
    run_block_inference(
        loader=DataLoader(TensorDataset(images, metadata), batch_size=2),
        model=model, accumulator=accumulator,
        weight_map=np.ones((2, 2), dtype=np.float32), mask=None,
        device=torch.device("cpu"), amp_dtype=None, tta_axes=(),
        tta_batch_size=None,
    )
    assert model.images_seen == 0
    assert not accumulator.buffers
    assert not accumulator.seen_counts
    assert not probability.any()
    assert not weight.any()
