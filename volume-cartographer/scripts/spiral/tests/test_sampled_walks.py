from types import SimpleNamespace

import numpy as np
import pytest
import torch

import losses
from native_spiral import load_native_spiral_sampling


native = load_native_spiral_sampling()


class _Atlas:
    device = torch.device('cpu')

    def __init__(self, masks):
        if native is None:
            raise RuntimeError('vc.spiral_sampling is required by this test')
        self.sampling_atlas = native.PatchSamplingAtlas(masks)
        self.node_maps = []
        start = 0
        for mask in masks:
            node_map = np.full(mask.shape, -1, dtype=np.int64)
            node_map[mask] = start + np.arange(mask.sum())
            self.node_maps.append(node_map)
            start += int(mask.sum())

    def theta_node_ids(self, patch_indices, ijs):
        patch_indices = np.asarray(patch_indices)
        cells = np.floor(np.asarray(ijs)).astype(np.int64)
        out = np.empty(cells.shape[:-1], dtype=np.int64)
        for patch_idx in np.unique(patch_indices):
            selected = patch_indices == patch_idx
            ij = cells[selected]
            out[selected] = self.node_maps[int(patch_idx)][ij[:, 0], ij[:, 1]]
        return out

    def lookup(self, patch_indices, ijs):
        return torch.cat([
            torch.zeros((*ijs.shape[:-1], 1)), ijs.to(torch.float32)], dim=-1)


def _patch(mask):
    return SimpleNamespace(
        _sampling_valid_quad_mask_np=mask,
        _sampling_valid_quad_indices_np=np.argwhere(mask).astype(np.int64),
    )


def test_patch_sampler_requires_current_native_binding():
    with pytest.raises(RuntimeError, match='sample_patch_points'):
        losses._sample_patch_points(
            np.array([0]), 8, np.random.RandomState(7),
            SimpleNamespace(sampling_atlas=None))


def test_patch_batch_returns_node_ids_and_explicit_padding_mask():
    masks = [np.ones((2, 2), dtype=bool), np.ones((5, 5), dtype=bool)]
    patches = [_patch(mask) for mask in masks]
    atlas = _Atlas(masks)
    np.random.seed(3)
    batch = losses._sample_patch_batch(
        'uniform_2d', patches, np.array([1.0, 0.0]), 2, 7, {},
        patch_atlas=atlas, crossing_map=object())
    ijs, patch_indices, zyxs, node_ids, sample_mask = batch
    assert ijs.shape == (2, 7, 2)
    assert zyxs.shape == (2, 7, 3)
    assert node_ids.shape == (2, 7)
    assert sample_mask.shape == (2, 7)
    assert sample_mask.sum(dim=-1).tolist() == [4, 4]
    assert patch_indices.tolist() == [0, 0]
    expected = atlas.theta_node_ids(
        np.zeros((2, 7), dtype=np.int64), ijs.numpy())
    np.testing.assert_array_equal(node_ids.numpy(), expected)
