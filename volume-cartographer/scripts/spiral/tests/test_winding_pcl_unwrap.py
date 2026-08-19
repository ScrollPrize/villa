import math

import numpy as np
import torch

import losses
from losses import PatchWalk, SampledWalk
from theta_crossing_map import ThetaCrossingMap


DR = 12.0


class _IdentityTransform:
    def __call__(self, points):
        return points


class _Atlas:
    device = torch.device('cpu')
    id_to_idx = {'p1': 0, 'p2': 1}

    def __init__(self, picked_points):
        self.picked_points = torch.stack(picked_points)

    def lookup(self, patch_indices, ijs):
        sample_indices = torch.floor(ijs[..., 1]).to(torch.int64)
        return self.picked_points[patch_indices.to(torch.int64), sample_indices]


class _Chain:
    def __init__(self, points):
        self.points = points

    def points_between(self, _first, _second):
        return self.points


def _spiral_point(theta, raw_winding):
    radius = DR * (raw_winding + theta / (2 * math.pi))
    return [0.0, math.sin(theta) * radius, math.cos(theta) * radius]


def _crossing_map(points, edges):
    points = torch.tensor(points, dtype=torch.float32)
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(len(points), lambda indices: points[indices])
    crossing_map.register_edges(edges)
    crossing_map.force_refresh(_IdentityTransform())
    return crossing_map


def _sampled_shapes(monkeypatch, nodes_by_patch):
    ijs = np.array([[0.25, 0.25], [0.25, 1.25]], dtype=np.float32)

    def sample(_patches, _atlas, requests, _num_points, _cfg):
        result = []
        for pid, _i, _j in requests:
            nodes = np.asarray(nodes_by_patch[pid], dtype=np.int64)
            result.append([
                PatchWalk(
                    ijs=ijs.copy(),
                    walk=SampledWalk(
                        node_ids=nodes.copy(),
                        pick_positions=np.array([1, 2], dtype=np.int64),
                        connect_fractional_picks=True,
                    ),
                )
                for _ in range(4)
            ])
        return result

    monkeypatch.setattr(losses, '_sample_l_shapes_batch', sample)


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


def test_absolute_winding_loss_keeps_annotation_frame_across_seam(monkeypatch):
    # Node 0 is the exact absolute annotation. The L-walk begins at node 1,
    # crosses theta=0, and only samples nodes 2/3. Those raw samples are in
    # winding 5, but must unwrap back to the annotation's winding 4.
    points = [
        _spiral_point(6.0, 4),
        _spiral_point(6.1, 4),
        _spiral_point(0.1, 5),
        _spiral_point(0.2, 5),
    ]
    crossing_map = _crossing_map(points, [[1, 2], [2, 3]])
    _sampled_shapes(monkeypatch, {'p1': [1, 2, 3]})
    atlas = _Atlas([
        torch.tensor([points[2], points[3]], dtype=torch.float32),
        torch.tensor([points[2], points[3]], dtype=torch.float32),
    ])
    point = {
        '_theta_node_id': 0,
        'winding_annotation': 4.0,
        'on_patch': {'id': 'p1', 'ij': [0.0, 0.0]},
    }
    pcl = {
        'metadata': {'winding_is_absolute': True},
        'points_by_patch': {'p1': [point]},
    }

    loss = losses.get_patch_abs_winding_loss(
        _IdentityTransform(), torch.tensor(DR), {'p1': object()}, atlas,
        [pcl], crossing_map=crossing_map, cfg=_cfg(), z_begin=-1, z_end=1)
    assert loss.item() < 1e-5


def test_relative_winding_loss_keeps_both_annotation_frames(monkeypatch):
    # The first patch crosses theta=0 before its first pick; the second does
    # not. After anchor-relative transport their samples remain exactly two
    # windings apart, matching the PCL annotations.
    points = [
        _spiral_point(6.0, 4),
        _spiral_point(6.1, 4),
        _spiral_point(0.1, 5),
        _spiral_point(0.2, 5),
        _spiral_point(5.0, 6),
        _spiral_point(5.1, 6),
        _spiral_point(5.2, 6),
        _spiral_point(5.3, 6),
    ]
    crossing_map = _crossing_map(
        points, [[0, 4], [1, 2], [2, 3], [5, 6], [6, 7]])
    _sampled_shapes(monkeypatch, {'p1': [1, 2, 3], 'p2': [5, 6, 7]})
    atlas = _Atlas([
        torch.tensor([points[2], points[3]], dtype=torch.float32),
        torch.tensor([points[6], points[7]], dtype=torch.float32),
    ])
    p1 = {
        '_theta_node_id': 0,
        'winding_annotation': 4.0,
        'on_patch': {'id': 'p1', 'ij': [0.0, 0.0]},
    }
    p2 = {
        '_theta_node_id': 4,
        'winding_annotation': 6.0,
        'on_patch': {'id': 'p2', 'ij': [0.0, 0.0]},
    }
    pcl = {
        'metadata': {},
        'points_by_patch': {'p1': [p1], 'p2': [p2]},
        'chain': _Chain([p1, p2]),
    }
    cfg = _cfg()
    strata = losses.build_pcl_sampling_strata(['relative'], cfg)

    loss = losses.get_patch_rel_winding_loss(
        _IdentityTransform(), torch.tensor(DR),
        {'p1': object(), 'p2': object()}, atlas, [pcl], strata,
        crossing_map=crossing_map, cfg=cfg, z_begin=-1, z_end=1)
    assert loss.item() < 1e-5
