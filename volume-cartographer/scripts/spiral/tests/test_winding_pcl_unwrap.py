import math
from types import SimpleNamespace

import numpy as np
import torch

import losses
from spiral_sampling import load_spiral_sampling
from theta_crossing_map import ThetaCrossingMap


DR = 12.0
spiral_sampling = load_spiral_sampling()


class _IdentityTransform:
    def __call__(self, points):
        return points


class _Atlas:
    device = torch.device('cpu')

    def __init__(self, patch_points, node_maps):
        if spiral_sampling is None:
            raise RuntimeError('vc.spiral_sampling is required by this test')
        self.patch_points = [torch.tensor(p, dtype=torch.float32)
                             for p in patch_points]
        self.node_maps = node_maps
        self.id_to_idx = {f'p{idx + 1}': idx
                          for idx in range(len(patch_points))}
        self.sampling_atlas = spiral_sampling.PatchSamplingAtlas([
            np.ones((1, len(points)), dtype=bool)
            for points in patch_points
        ])

    def lookup(self, patch_indices, ijs):
        patch_indices = patch_indices.to(torch.int64)
        columns = torch.floor(ijs[..., 1]).to(torch.int64)
        result = torch.empty((*ijs.shape[:-1], 3), dtype=torch.float32)
        for patch_idx in torch.unique(patch_indices):
            selected = patch_indices == patch_idx
            result[selected] = self.patch_points[int(patch_idx)][columns[selected]]
        return result

    def theta_node_ids(self, patch_indices, ijs):
        patch_indices = np.asarray(patch_indices)
        columns = np.floor(np.asarray(ijs)[..., 1]).astype(np.int64)
        result = np.empty(columns.shape, dtype=np.int64)
        for patch_idx in np.unique(patch_indices):
            selected = patch_indices == patch_idx
            result[selected] = self.node_maps[int(patch_idx)][columns[selected]]
        return result


class _Chain:
    def __init__(self, points):
        self.points = points

    def points_between(self, _first, _second):
        return self.points


def _spiral_point(theta, raw_winding):
    radius = DR * (raw_winding + theta / (2 * math.pi))
    return [0.0, math.sin(theta) * radius, math.cos(theta) * radius]


def _setup(ref_points, patch_points, pcl_edges=()):
    crossing_map = ThetaCrossingMap('cpu')
    refs = torch.tensor(ref_points, dtype=torch.float32)
    crossing_map.register_nodes(len(refs), lambda lo, hi: refs[lo:hi])
    if pcl_edges:
        crossing_map.register_edges(pcl_edges)
    node_maps = []
    for points in patch_points:
        values = torch.tensor(points, dtype=torch.float32)
        start = crossing_map.register_nodes(
            len(values), lambda lo, hi, values=values: values[lo:hi])
        nodes = start + np.arange(len(values), dtype=np.int64)
        crossing_map.register_unwrap_tree(
            nodes, np.arange(-1, len(nodes) - 1, dtype=np.int64))
        node_maps.append(nodes)
    crossing_map.force_refresh(_IdentityTransform())
    atlas = _Atlas(patch_points, node_maps)
    patches = {
        pid: SimpleNamespace(
            _sampling_valid_quad_mask_np=np.ones((1, 2), dtype=bool),
            _sampling_valid_quad_indices_np=np.array([[0, 0], [0, 1]],
                                                     dtype=np.int64))
        for pid in atlas.id_to_idx
    }
    return crossing_map, atlas, patches


def _cfg():
    return {
        'sample_count_points_per_patch': 4,
        'sample_count_relative_winding_pcls': 1,
        'sample_count_relative_winding_patch_pairs_per_pcl': 1,
        'sample_count_absolute_winding_pcls': 1,
        'sample_count_absolute_winding_points_per_pcl': 1,
        'pcl_rel_winding_adjacent_patches_only': True,
        'pcl_sampling_weights': None,
        'pcl_stratified_pcl_sampling': False,
        'patch_loss_z_margin': 0.0,
    }


def test_absolute_winding_keeps_exact_annotation_frame_with_padding():
    ref = _spiral_point(6.0, 4)
    patch_points = [[_spiral_point(0.1, 5), _spiral_point(0.2, 5)]]
    crossing_map, atlas, patches = _setup([ref], patch_points)
    point = {
        '_theta_node_id': 0,
        'winding_annotation': 4.0,
        'on_patch': {'id': 'p1', 'ij': [0.0, 0.0]},
    }
    pcl = {'metadata': {'winding_is_absolute': True},
           'points_by_patch': {'p1': [point]}}
    loss = losses.get_patch_abs_winding_loss(
        _IdentityTransform(), torch.tensor(DR), patches, atlas, [pcl],
        crossing_map=crossing_map, cfg=_cfg(), z_begin=-1, z_end=1)
    assert loss.item() < 1e-5


def test_relative_winding_keeps_both_exact_annotation_frames():
    refs = [_spiral_point(6.0, 4), _spiral_point(5.0, 6)]
    patch_points = [
        [_spiral_point(0.1, 5), _spiral_point(0.2, 5)],
        [_spiral_point(5.2, 6), _spiral_point(5.3, 6)],
    ]
    crossing_map, atlas, patches = _setup(refs, patch_points, [[0, 1]])
    p1 = {'_theta_node_id': 0, 'winding_annotation': 4.0,
          'on_patch': {'id': 'p1', 'ij': [0.0, 0.0]}}
    p2 = {'_theta_node_id': 1, 'winding_annotation': 6.0,
          'on_patch': {'id': 'p2', 'ij': [0.0, 0.0]}}
    pcl = {'metadata': {}, 'points_by_patch': {'p1': [p1], 'p2': [p2]},
           'chain': _Chain([p1, p2])}
    cfg = _cfg()
    strata = losses.build_pcl_sampling_strata(['relative'], cfg)
    loss = losses.get_patch_rel_winding_loss(
        _IdentityTransform(), torch.tensor(DR), patches, atlas, [pcl], strata,
        crossing_map=crossing_map, cfg=cfg, z_begin=-1, z_end=1)
    assert loss.item() < 1e-5


def test_annotation_outside_retained_mask_is_skipped():
    ref = _spiral_point(1.0, 3)
    patch_points = [[_spiral_point(1.1, 3), _spiral_point(1.2, 3)]]
    crossing_map, atlas, patches = _setup([ref], patch_points)
    patches['p1']._sampling_valid_quad_mask_np[0, 0] = False
    point = {'_theta_node_id': 0, 'winding_annotation': 3.0,
             'on_patch': {'id': 'p1', 'ij': [0.0, 0.0]}}
    pcl = {'metadata': {'winding_is_absolute': True},
           'points_by_patch': {'p1': [point]}}
    loss = losses.get_patch_abs_winding_loss(
        _IdentityTransform(), torch.tensor(DR), patches, atlas, [pcl],
        crossing_map=crossing_map, cfg=_cfg(), z_begin=-1, z_end=1)
    assert loss.item() == 0.0
