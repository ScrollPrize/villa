"""find_inconsistent_windings counts the theta=0 step between an annotated point
and its attachment on the patch, where the within-patch strips start and end."""

import math
from types import SimpleNamespace

import numpy as np
import torch

import connect_overlapping_patches as cop
import find_inconsistent_windings as fiw
from tifxyz import Patch

DR = torch.tensor(12.0)
RADIUS = 800.0
TWO_PI = 2 * math.pi


class IdentityTransform:
    def __call__(self, zyxs):
        return zyxs


def _polar_zyx(theta, z=1.0):
    return np.array([z, math.sin(theta) * RADIUS, math.cos(theta) * RADIUS], dtype=np.float32)


def _arc_patch(theta_lo, theta_hi, cols):
    thetas = np.linspace(theta_lo, theta_hi, cols)
    grid = np.stack([np.stack([_polar_zyx(t, z=float(i)) for t in thetas]) for i in range(3)])
    return Patch(zyxs=torch.from_numpy(grid), scale=torch.tensor([1.0, 1.0]),
                 overlapping_ids=None, winding=None)


def _point(pid, theta, patch_id, ij, winding=0.0):
    zyx = _polar_zyx(theta)
    return {'id': pid, 'p': zyx[::-1].tolist(), 'zyx': zyx, 'winding_annotation': winding,
            'on_patch': {'id': patch_id, 'ij': list(ij), 'distance': 1.0}}


def _relative_pcl(points):
    return {'name': 'rel', 'source_file': 'relative_windings.json', 'metadata': {},
            'has_winding_annotations': True, 'points': {p['id']: p for p in points},
            'chain': SimpleNamespace(iter_chain=lambda: iter(points))}


def _patch_graph(patch):
    graph = cop.build_patch_graph(patch, 4.0)
    graph._dij_cache = {}
    return graph


def test_relative_edge_is_expressed_between_attachments():
    patches = {'P': _arc_patch(-0.002, 0.002, 3), 'R': _arc_patch(0.998, 1.002, 3)}
    b = _point(2, 1.0, 'R', (1.0, 1.0))

    across_the_ray = _point(1, TWO_PI - 0.001, 'P', (1.0, 1.5))
    adjacency = fiw.build_rel_adjacency(
        {0: _relative_pcl([across_the_ray, b])}, patches, IdentityTransform(), DR)
    assert adjacency['P'][0]['winding_delta'] == 0
    assert adjacency['R'][0]['winding_delta'] == 0

    same_side = _point(1, TWO_PI - 0.001, 'P', (1.0, 0.5))
    adjacency = fiw.build_rel_adjacency(
        {0: _relative_pcl([same_side, b])}, patches, IdentityTransform(), DR)
    assert adjacency['P'][0]['winding_delta'] == 1
    assert adjacency['R'][0]['winding_delta'] == -1


def test_consistent_two_patch_loop_closes_when_a_point_straddles_the_ray():
    patches = {'P': _arc_patch(-0.002, 0.302, 39), 'R': _arc_patch(0.998, 1.302, 39)}
    graphs = {pid: _patch_graph(patch) for pid, patch in patches.items()}

    def strip(pid, from_ij, to_ij):
        return fiw.strip_winding_delta(
            IdentityTransform(), DR, graphs[pid], from_ij, to_ij, 1.0)['delta_windings']

    a1 = _point(1, TWO_PI - 0.001, 'P', (1.0, 0.75))
    b1 = _point(2, 1.0, 'R', (1.0, 0.5))
    a2 = _point(3, 0.2, 'P', (1.0, 25.25))
    b2 = _point(4, 1.2, 'R', (1.0, 25.5))
    adjacency = fiw.build_rel_adjacency(
        {0: _relative_pcl([a1, b1]), 1: _relative_pcl([a2, b2])},
        patches, IdentityTransform(), DR)
    tree_edge, closing_edge = adjacency['P']
    entry_P, entry_R = tree_edge['from_ij'], tree_edge['to_ij']

    acc_R = 0 - strip('P', entry_P, tree_edge['from_ij']) - tree_edge['winding_delta']
    via_edge = 0 - strip('P', entry_P, closing_edge['from_ij']) - closing_edge['winding_delta']
    via_tree = acc_R - strip('R', entry_R, closing_edge['to_ij'])
    assert via_edge - via_tree == 0


def test_absolute_anchor_carries_the_step_to_its_attachment():
    patches = {'P': _arc_patch(-0.002, 0.002, 3)}
    across_the_ray = _point(1, TWO_PI - 0.001, 'P', (1.0, 1.5), winding=7.0)
    same_side = _point(2, TWO_PI - 0.001, 'P', (1.0, 0.5), winding=7.0)
    pcl = {'name': 'abs', 'source_file': 'abs_winding.json',
           'metadata': {'winding_is_absolute': True},
           'points_by_patch': {'P': [across_the_ray, same_side]}}
    anchors = fiw.build_abs_anchors_by_patch({0: pcl}, patches, IdentityTransform(), DR)
    assert [a['winding'] for a in anchors['P']] == [7, 7]
    assert [a['attachment_branch_delta'] for a in anchors['P']] == [1, 0]
