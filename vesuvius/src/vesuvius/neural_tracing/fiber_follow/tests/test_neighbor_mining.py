"""Regressions for false-negative rejection, two-point seeds and trace units."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.regression.neighbor_mining import (
    MiningConfig, dense_line, exact_nearest, seed_pairs, trace_controls, traversed_voxels, validate_path,
)
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength


def fixture():
    target = np.c_[np.arange(65.), np.full(65, 20.), np.full(65, 20.)]
    path = np.array([[24., 25., 20.], [40., 25., 20.]])
    presence = np.full((64,)*3, 255, np.uint8)
    directions = np.zeros((*presence.shape, 3))
    directions[..., 0] = 1
    labels = np.ones(presence.shape, int)
    cfg = MiningConfig(seed_spacing=8, extrapolation=4, block_size=64)
    return target, path, presence, directions, labels, cfg


def check(path, target, presence, directions, labels, cfg):
    return validate_path(path, target, arclength(target), np.zeros(3), presence, directions, labels, 1, cfg)


def test_true_parallel_neighbor_passes_with_certified_separation():
    target, path, presence, directions, labels, cfg = fixture()
    ok, info = check(path, target, presence, directions, labels, cfg)
    assert ok, info
    assert 4.8 < info['min_distance_lower_bound'] <= 5


def test_sparse_annotation_segment_interiors_are_excluded():
    target, path, presence, directions, labels, cfg = fixture()
    path[:, 1] = 21
    # Vertex-only distances would report >24 voxels of clearance.
    target = target[[0, -1]]
    ok, info = check(path, target, presence, directions, labels, cfg)
    assert not ok and info['reason'] == 'target_exclusion'


def test_subvoxel_presence_gap_and_component_change_are_rejected():
    target, path, presence, directions, labels, cfg = fixture()
    labels[20, 25, 32] = 0
    ok, info = check(path, target, presence, directions, labels, cfg)
    assert not ok and info['reason'] == 'left_presence_component'


def test_wrong_direction_and_censored_annotation_end_are_rejected():
    target, path, presence, directions, labels, cfg = fixture()
    directions[..., 0], directions[..., 1] = 0, 1
    assert not check(path, target, presence, directions, labels, cfg)[0]
    directions[..., 0], directions[..., 1] = 1, 0
    ok, info = check(path, target[:42], presence, directions, labels, cfg)
    assert not ok and info['reason'] == 'annotation_boundary'


def test_continuous_separation_margin_rejects_near_threshold():
    target, path, presence, directions, labels, cfg = fixture()
    path[:, 1] = 20+cfg.exclusion+.05
    assert not check(path, target, presence, directions, labels, cfg)[0]


def test_densification_preserves_corners_and_final_short_segment():
    line = np.array([[0., 0, 0], [.7, 0, 0], [.7, .1, 0]])
    dense = dense_line(line, .25)
    np.testing.assert_array_equal(dense[-1], line[-1])
    assert (dense == line[1]).all(-1).any()
    assert np.linalg.norm(np.diff(dense, axis=0), axis=1).max() <= .25
    distance, nearest, _, _ = exact_nearest(np.array([[.4, .1, 0]]), line)
    assert distance[0] == pytest.approx(.1)
    np.testing.assert_allclose(nearest[0], [.4, 0, 0])


def test_voxel_traversal_catches_brief_visits_between_regular_samples():
    line = np.array([[0., 0, 0], [2.1, 2.11, 0]])
    cells, _ = traversed_voxels(line)
    # This cell is crossed for only ~0.0035 voxels of travel.
    assert (cells == [0, 1, 0]).all(-1).any()
    regular = np.round(dense_line(line, .25)).astype(int)
    assert not (regular == [0, 1, 0]).all(-1).any()


def test_seed_points_share_uncut_component_and_are_well_separated():
    target, _, presence, directions, _, cfg = fixture()
    presence[:] = 0
    # Two independent parallel tubes: the target itself and a neighbor.
    presence[19:22, 19:22, :] = 255
    presence[19:22, 25:28, :] = 255
    pairs, _ = seed_pairs(presence, directions, np.zeros(3), np.array([32., 20, 20]), np.array([1., 0, 0]), target, cfg)
    assert len(pairs) == 1
    assert np.linalg.norm(pairs[0][1][1]-pairs[0][1][0]) >= cfg.seed_spacing-2
    assert (pairs[0][1][:, 1] >= 25).all()
    # Joining the high-threshold components must cause abstention, not create
    # a fake foreign component by first erasing the annotation's exclusion tube.
    presence[20, 20:28, 32] = 255
    pairs, _ = seed_pairs(presence, directions, np.zeros(3), np.array([32., 20, 20]), np.array([1., 0, 0]), target, cfg)
    assert not pairs


def test_native_adapter_uses_scale_and_both_outward_tangents():
    calls = []
    def segment(field, controls, a, b, config):
        calls.append(('segment', controls.copy(), a, b))
        return SimpleNamespace(accepted=True, fused_line=controls[[a, b]], meeting_error_trace_voxels=0.)
    def extrapolate(field, start, direction, distance, config):
        calls.append(('tail', np.array(start), direction, distance))
        return SimpleNamespace(reached_trace_length=True, points=np.stack([start, start+direction*distance]))
    native = SimpleNamespace(trace_segment=segment, trace_extrapolation=extrapolate)
    controls = np.array([[10., 0, 0], [20., 0, 0], [30., 0, 0]])
    field = SimpleNamespace(trace_to_base_scale=2.)
    path, _ = trace_controls(native, field, controls, None, MiningConfig())
    np.testing.assert_array_equal(calls[0][1], controls*4)
    np.testing.assert_allclose(path[[0, -1]], [[0., 0, 0], [40., 0, 0]])
    assert calls[-1][-1] == 40
    with pytest.raises(ValueError, match='at least two'):
        trace_controls(native, field, controls[:1], None, MiningConfig())


def test_parameters_fail_closed():
    with pytest.raises(ValueError):
        replace(MiningConfig(), path_presence=.9, seed_presence=.8)
    with pytest.raises(ValueError):
        replace(MiningConfig(), extrapolation=float('nan'))
