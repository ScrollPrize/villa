from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from vesuvius.ink_detection.inference import infer


class SavedLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer('logits', torch.from_numpy(logits)[None, None])

    def forward(self, images):
        return self.logits.expand(images.shape[0], -1, -1, -1).clone()


@pytest.mark.parametrize('threads', [0, 2])
@pytest.mark.parametrize('roi', [None, (8, 120, 8, 120)])
@pytest.mark.parametrize('masked', [False, True])
def test_real_segment_edge_has_no_predictions_without_ct(threads, roi, masked):
    fixture = Path(__file__).parent / 'fixtures/ink_empty_pixels.npz'
    with np.load(fixture) as saved:
        ct, logits = saved['ct'], saved['logits']
    valid = ct.any(axis=0)
    assert valid.any() and not valid.all()
    reader = infer.FlatPatchReader(
        input_path='fixture', resolution='0', depth_axis_first=True,
        height=128, width=128, layer_indices=np.arange(17), output_depth=17,
        preprocessing='tifxyz_robust',
    )
    reader._array = ct
    blocks = infer.iter_blocks((128, 128), 128, 64, roi_bounds=roi)
    dataset = infer.FlatBlockDataset(
        reader=reader, blocks=blocks, patch_size=128,
        preprocessing='tifxyz_robust',
    )
    loader = (
        infer.ThreadedBatchLoader(dataset, batch_size=1, num_threads=threads, prefetch_batches=2)
        if threads else DataLoader(dataset, batch_size=1)
    )
    bounds = roi or (0, 128, 0, 128)
    rows, cols = slice(*bounds[:2]), slice(*bounds[2:])
    shape = (bounds[1] - bounds[0], bounds[3] - bounds[2])
    probability = np.zeros(shape, dtype=np.float32)
    weight = np.zeros_like(probability)
    clipped = [infer.clip_block_to_roi(block, roi) for block in blocks] if roi else blocks
    accumulator = infer.ChunkAccumulator(
        shape=shape, chunk_shape=(32, 32), prob_sum_store=probability,
        weight_sum_store=weight,
        contribution_counts=infer.compute_chunk_contribution_counts(clipped, chunk_shape=(32, 32)),
    )
    mask = None
    if masked:
        mask = np.ones((128, 128), dtype=bool)
        mask[32:64, 32:64] = False
        valid &= mask
    infer.run_block_inference(
        loader=loader, model=SavedLogits(logits), accumulator=accumulator,
        weight_map=np.ones((128, 128), dtype=np.float32), mask=mask,
        device=torch.device('cpu'), amp_dtype=None, tta_axes=(),
        tta_batch_size=None, roi_bounds=roi,
    )
    accumulator.flush_remaining()
    expected = torch.from_numpy(logits).sigmoid().numpy()
    expected[~valid] = 0
    np.testing.assert_array_equal(probability, expected[rows, cols])
    np.testing.assert_array_equal(weight, valid[rows, cols])
    assert not accumulator.buffers
