from types import SimpleNamespace

import numpy as np
import pytest
import torch

import losses
from losses import PatchWalk, SampledWalk
from theta_crossing_map import ThetaCrossingMap


class _NativeStraightSampler:
    def sample_patch_walks(self, patch_indices, num_points, seed):
        assert len(patch_indices) == 1
        horizontal = np.array([[1, 0], [1, 1], [1, 2]], dtype=np.int64)
        vertical = np.array([[0, 2], [1, 2], [2, 2]], dtype=np.int64)
        paths = np.concatenate([horizontal, vertical])
        picks = np.resize(np.array([0, 1, 2], dtype=np.int64), (2, 1, num_points))
        ijs = np.empty((2, 1, num_points, 2), dtype=np.float32)
        ijs[0, 0] = horizontal[picks[0, 0]] + 0.25
        ijs[1, 0] = vertical[picks[1, 0]] + 0.25
        return {
            'ijs': ijs,
            'path_ijs': paths,
            'path_offsets': np.array([0, 3, 6], dtype=np.int64),
            'pick_positions': picks,
        }


class _NativeLShapeSampler:
    def sample_l_shapes(self, patch_indices, anchors, num_points, seed):
        waypoints = np.array([[[
            [1, 1], [1, 2], [2, 2],
        ]] * 4], dtype=np.int64)
        picks = np.resize(
            np.array([0, 1, 2], dtype=np.int64), (1, 4, num_points))
        path = np.array([[1, 1], [1, 2], [2, 2]], dtype=np.float32)
        ijs = np.empty((1, 4, num_points, 2), dtype=np.float32)
        for shape in range(4):
            ijs[0, shape] = path[picks[0, shape]] + 0.25
        return {
            'ijs': ijs,
            'pick_positions': picks,
            'waypoints': waypoints,
            'valid': np.array([True]),
        }


class _Atlas:
    id_to_idx = {'p': 0}
    device = torch.device('cpu')

    def __init__(self, native=None):
        self.sampling_atlas = native
        self.node_map = np.arange(16, dtype=np.int64).reshape(4, 4)

    def theta_node_ids(self, patch_indices, ijs):
        cells = np.floor(np.asarray(ijs)).astype(np.int64)
        return self.node_map[cells[..., 0], cells[..., 1]]

    def lookup(self, patch_indices, ijs):
        return torch.zeros((*ijs.shape[:-1], 3), dtype=torch.float32)


def _crossing_map():
    crossing_map = ThetaCrossingMap('cpu')
    points = torch.zeros((16, 3), dtype=torch.float32)
    crossing_map.register_nodes(16, lambda indices: points[indices])
    node_map = np.arange(16, dtype=np.int64).reshape(4, 4)
    edges = []
    for di, dj in ((0, 1), (1, -1), (1, 0), (1, 1)):
        a = node_map[:4 - di, max(0, -dj):4 - max(0, dj)]
        b = node_map[di:, max(0, dj):4 - max(0, -dj)]
        edges.append(np.stack([a.reshape(-1), b.reshape(-1)], axis=1))
    crossing_map.register_edges(np.concatenate(edges))
    return crossing_map


def _straight_patch():
    return SimpleNamespace(
        _sampling_2d_path=None,
        _sampling_valid_quad_rows=np.arange(4, dtype=np.int64),
        _sampling_valid_quad_cols=np.arange(4, dtype=np.int64),
        _h_runs_los=[np.array([0])] * 4,
        _h_runs_his=[np.array([4])] * 4,
        _h_runs_cum=[np.array([4])] * 4,
        _v_runs_los=[np.array([0])] * 4,
        _v_runs_his=[np.array([4])] * 4,
        _v_runs_cum=[np.array([4])] * 4,
    )


@pytest.mark.parametrize('mode', ['straight', 'dijkstra', 'serpentine', 'native'])
def test_patch_sampler_normalizes_every_row_to_global_node_ids(
    mode, monkeypatch,
):
    atlas = _Atlas(_NativeStraightSampler() if mode == 'native' else None)
    patch = _straight_patch()
    cfg_mode = 'straight'
    if mode == 'dijkstra':
        cfg_mode = 'dijkstra'
        path = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=np.int64)
        patch._strip_path_pool = [path]
        monkeypatch.setattr(
            losses.strip_path_pools, 'ensure_patch_path_pools', lambda patches: None)
        monkeypatch.setattr(
            losses.strip_path_pools, 'submit_patch_pool_refresh', lambda patch: None)
    elif mode == 'serpentine':
        patch._sampling_2d_path = np.array([
            [0, 0], [0, 1], [1, 1], [1, 0],
        ], dtype=np.int64)

    ijs, _, _, packed = losses._sample_patch_batch(
        f'normalize_{mode}', [patch], np.array([1.0]), 1, 3,
        {'patch_strip_sampling': cfg_mode},
        patch_atlas=atlas, crossing_map=_crossing_map())
    expected_nodes = atlas.theta_node_ids(
        np.zeros((2, 3), dtype=np.int64), ijs.numpy().reshape(2, 3, 2))
    np.testing.assert_array_equal(
        packed.correction_node_ids.numpy(), expected_nodes)
    assert packed.correction_node_ids.shape == (2, 3)


@pytest.mark.parametrize('native', [False, True])
def test_l_shape_sampler_keeps_ijs_but_normalizes_topology(native):
    atlas = _Atlas(_NativeLShapeSampler() if native else None)
    patch = SimpleNamespace(_sampling_valid_quad_mask_np=np.ones((4, 4), dtype=bool))
    np.random.seed(4)
    shapes = losses._sample_l_shapes_batch(
        {'p': patch}, atlas, [('p', 1, 1)], 3,
        {'patch_strip_sampling': 'straight'})[0]

    assert len(shapes) == 4
    assert all(isinstance(shape, PatchWalk) for shape in shapes)
    assert all(isinstance(shape.walk, SampledWalk) for shape in shapes)
    assert all(shape.walk.connect_fractional_picks for shape in shapes)
    for shape in shapes:
        picked_nodes = shape.walk.node_ids[shape.walk.pick_positions]
        expected = atlas.theta_node_ids(np.zeros(3, dtype=np.int64), shape.ijs)
        np.testing.assert_array_equal(picked_nodes, expected)
        assert not hasattr(shape, 'path')
        assert not hasattr(shape, 'waypoints')
