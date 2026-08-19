from types import SimpleNamespace

import numpy as np
import torch

import losses


class _Atlas:
    sampling_atlas = None
    device = torch.device('cpu')

    def __init__(self, masks):
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


def test_python_sampler_is_seeded_uniform_without_replacement():
    mask = np.ones((7, 9), dtype=bool)
    mask[2:5, 3:7] = False
    patches = [_patch(mask)]
    first = losses._sample_patch_points_python(
        patches, [0, 0], 25, np.random.RandomState(42))
    second = losses._sample_patch_points_python(
        patches, [0, 0], 25, np.random.RandomState(42))
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], [25, 25])
    cells = np.floor(first[0]).astype(np.int64)
    assert mask[cells[..., 0], cells[..., 1]].all()
    assert all(len(np.unique(row, axis=0)) == 25 for row in cells)


def test_python_sampler_uses_every_small_cell_and_masks_padding():
    mask = np.zeros((4, 6), dtype=bool)
    mask[[0, 1, 3], [4, 2, 5]] = True
    ijs, counts = losses._sample_patch_points_python(
        [_patch(mask)], [0], 8, np.random.RandomState(7))
    assert counts.tolist() == [3]
    cells = np.floor(ijs[0]).astype(np.int64)
    assert len(np.unique(cells[:3], axis=0)) == 3
    assert mask[cells[:, 0], cells[:, 1]].all()
    np.testing.assert_array_equal(
        ijs[0, 3:], np.repeat(ijs[0, :1], 5, axis=0))


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
