"""Fiber supervision: classification, patch-side linking and radial offsets."""

import unittest
from unittest import mock

import numpy as np
import pytest
import torch

import fit_spiral
import point_collection
from config import Config, FitConfig
from dt_targets import compute_strip_dt_target_cache
from fit_spiral import (
    FitContext,
    _build_strip_flat_bundle,
    accumulate_radial_offset_bake_scale,
    inward_winding_direction,
    materialize_fiber_fit_inputs,
)
from losses import get_radial_normal_in_scroll_space
from point_collection import SIDE_BEHIND, SIDE_FRONT
from sample_spiral import get_radial_covector_in_scroll_space, get_radial_normal_stretch
from spiral_helpers import classify_fiber_hv, fiber_collection_hv_tag
from tifxyz import Patch

import test_live_patch_relink as relink
from test_revisioned_geometry import context


DR = 12.0
WINDING = 40


def _vertical_fiber(theta_of_z, z_len=400.0, spacing=20.0, winding=WINDING):
    # Points on the continuous sheet r = DR * (winding + theta / 2pi) at
    # theta(z), z stepped at `spacing`. theta is deliberately not wrapped:
    # a fiber just past the seam sits on the next winding's radius.
    zs = np.arange(0.0, z_len + 1e-6, spacing)
    pts = []
    for z in zs:
        theta = float(theta_of_z(z))
        r = DR * (winding + theta / (2 * np.pi))
        pts.append([z, np.sin(theta) * r, np.cos(theta) * r])
    return np.asarray(pts, dtype=np.float32)


def _horizontal_fiber(wraps=0.5, theta0=0.3, spacing=20.0, winding=WINDING):
    pts = []
    theta = theta0
    while theta < theta0 + wraps * 2 * np.pi:
        r = DR * (winding + theta / (2 * np.pi))
        pts.append([0.0, np.sin(theta) * r, np.cos(theta) * r])
        theta += spacing / r
    return np.asarray(pts, dtype=np.float32)


def test_manual_tag_wins_over_geometry():
    horizontal = _horizontal_fiber()
    assert classify_fiber_hv(
        {'manual_tag': 'V', 'automatic_tag': 'H', 'automatic_certainty': 1.0},
        horizontal, min_z_fraction=0.8, min_auto_certainty=0.5) == 'V'
    assert classify_fiber_hv(
        {'manual_tag': 'h'}, _vertical_fiber(lambda z: 1.0),
        min_z_fraction=0.8, min_auto_certainty=0.5) == 'H'


def test_automatic_tag_requires_certainty_then_geometry_decides():
    vertical = _vertical_fiber(lambda z: 1.0)
    confident = {'automatic_tag': 'H', 'automatic_certainty': 0.9}
    unsure = {'automatic_tag': 'H', 'automatic_certainty': 0.2}
    assert classify_fiber_hv(
        confident, vertical, min_z_fraction=0.8, min_auto_certainty=0.5) == 'H'
    assert classify_fiber_hv(
        unsure, vertical, min_z_fraction=0.8, min_auto_certainty=0.5) == 'V'


def test_geometric_fallback_splits_vertical_horizontal_and_diagonal():
    assert classify_fiber_hv(
        None, _vertical_fiber(lambda z: 1.0),
        min_z_fraction=0.8, min_auto_certainty=0.5) == 'V'
    assert classify_fiber_hv(
        {}, _horizontal_fiber(),
        min_z_fraction=0.8, min_auto_certainty=0.5) == 'H'
    diagonal = np.asarray(
        [[0, 0, 0], [10, 10, 0], [20, 20, 0]], dtype=np.float32)
    assert classify_fiber_hv(
        {}, diagonal, min_z_fraction=0.8, min_auto_certainty=0.5) is None
    assert classify_fiber_hv(
        {}, diagonal[:1], min_z_fraction=0.8, min_auto_certainty=0.5) is None


def test_materialize_stamps_hv_tag_and_is_vertical_on_fiber_strips():
    from fit_spiral import materialize_fiber_fit_inputs

    def member(cid, logical_id, zyxs, hv=None):
        points = {
            i: {"id": i, "collectionId": cid,
                "p": [z[2], z[1], z[0]],
                "zyx": np.asarray(z, dtype=np.float32),
                "winding_annotation": float("nan")}
            for i, z in enumerate(zyxs)
        }
        return {
            "id": cid, "file_basename": f"{logical_id}.json",
            "sampling_group": "fibers",
            "metadata": {"logical_input_id": logical_id,
                         "logical_input_kind": "fiber",
                         "winding_is_absolute": False,
                         "hv_classification": hv or {}},
            "points": points,
            "kept_orig_indices": np.arange(len(zyxs)),
            "control_line_indices": np.arange(len(zyxs)),
            "branches": [],
        }

    vertical = [[float(z), 0.0, 100.0] for z in range(0, 100, 10)]
    horizontal = [[0.0, 0.0, 100.0 + x] for x in range(0, 100, 10)]
    catalog = {
        "v": member(1, "v", vertical),
        "h": member(2, "h", horizontal),
        "forced": member(3, "forced", horizontal, {"manual_tag": "V"}),
    }
    _, strips, *_ = materialize_fiber_fit_inputs(
        catalog, {}, z_begin=-10, z_end=200, z_margin=0,
        min_point_spacing=0, use_links=False,
        vertical_min_z_fraction=0.8, vertical_min_auto_certainty=0.5,
        vertical_radial_offset=4.0)
    by_id = {strip['id']: strip for strip in strips}
    assert by_id[1]['hv_tag'] == 'V' and by_id[1]['is_vertical']
    assert by_id[2]['hv_tag'] == 'H' and not by_id[2]['is_vertical']
    assert by_id[3]['hv_tag'] == 'V' and by_id[3]['is_vertical']
    # Vertical strips carry the radial target offset, horizontals never do.
    assert np.all(by_id[1]['radial_offsets'] == 4.0)
    assert np.all(by_id[3]['radial_offsets'] == 4.0)
    assert np.all(by_id[2]['radial_offsets'] == 0.0)
    _, strips_off, *_ = materialize_fiber_fit_inputs(
        catalog, {}, z_begin=-10, z_end=200, z_margin=0,
        min_point_spacing=0, use_links=False,
        vertical_min_z_fraction=0.8, vertical_min_auto_certainty=0.5)
    assert all(np.all(strip['radial_offsets'] == 0.0) for strip in strips_off)


# --- live Run-boundary offset changes ---------------------------------------------

def test_apply_config_refills_radial_offsets_and_drops_cached_bundle():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from fit_spiral import (
        FitContext, _UnattachedPclStripList, get_or_build_unattached_pcl_flat)

    def strip(cid, is_vertical, n=5):
        return {
            'id': cid,
            'zyxs': np.zeros((n, 3), dtype=np.float32),
            'windings': np.zeros(n, dtype=np.float32),
            'radial_offsets': np.zeros(n, dtype=np.float32),
            'is_vertical': is_vertical,
        }

    strips = _UnattachedPclStripList([
        strip(1, True), strip(2, False),
        # A regular (non-fiber) unattached pcl carries no hv tag at all.
        {k: v for k, v in strip(3, False).items() if k != 'is_vertical'},
    ])
    context = FitContext.__new__(FitContext)
    context.config = {
        'pcl_vertical_fiber_radial_offset_enabled': False,
        'pcl_vertical_fiber_radial_offset_voxels': 4.0,
        'theta_crossing_map_update_interval': 100,
        'dt_target_update_interval': 100,
        'track_max_tortuosity': 0,
        'track_exclusion_radius': 0,
    }
    context.unattached_pcl_strips = strips
    context.shell_map = None
    context.shell_outer_winding_idx = None
    context.shell_valid_zyxs_gpu = None
    context.tracks = []
    context.prepared_main_tracks = None
    context.verified_patches_list = []
    context.unverified_patches = None
    context.unverified_patches_list = []
    context.unverified_patch_sampling_probabilities = None
    context.unverified_patch_atlas = None
    context.dt_target_cache_manager = SimpleNamespace(
        update_interval=100, reset=Mock())
    context.theta_crossing_map = SimpleNamespace(invalidate=Mock())

    stale = get_or_build_unattached_pcl_flat(strips, torch.device('cpu'))
    assert strips.flat is stale
    assert torch.all(stale['radial_offsets'] == 0.0)

    context.apply_config(
        {'pcl_vertical_fiber_radial_offset_enabled': True},
        current_iteration=0)
    assert np.all(strips[0]['radial_offsets'] == 4.0)
    assert np.all(strips[1]['radial_offsets'] == 0.0)
    assert np.all(strips[2]['radial_offsets'] == 0.0)
    assert strips.flat is None
    context.dt_target_cache_manager.reset.assert_called()
    fresh = get_or_build_unattached_pcl_flat(strips, torch.device('cpu'))
    assert torch.all(fresh['radial_offsets'][:5] == 4.0)
    assert torch.all(fresh['radial_offsets'][5:] == 0.0)

    context.apply_config(
        {'pcl_vertical_fiber_radial_offset_voxels': 7.0},
        current_iteration=0)
    assert np.all(strips[0]['radial_offsets'] == 7.0)
    assert strips.flat is None

    context.apply_config(
        {'pcl_vertical_fiber_radial_offset_enabled': False},
        current_iteration=0)
    assert all(np.all(s['radial_offsets'] == 0.0) for s in strips)
    assert strips.flat is None


def _fiber(cid, zyxs, hv=None, kind='fiber'):
    return {
        'id': cid, 'name': f'fiber{cid}',
        'metadata': {'logical_input_kind': kind, 'logical_input_id': str(cid),
                     'hv_classification': hv or {}},
        'points': {
            i: {'id': i, 'collectionId': cid, 'p': [x, y, z],
                'zyx': np.asarray([z, y, x], dtype=np.float32),
                'winding_annotation': float('nan')}
            for i, (z, y, x) in enumerate(zyxs)
        },
    }


def _y_plane(y, z0, x0, size=5, spacing=10.0):
    grid = torch.zeros((size, size, 3), dtype=torch.float32)
    for i in range(size):
        for j in range(size):
            grid[i, j] = torch.tensor([z0 + i * spacing, y, x0 + j * spacing])
    return Patch(grid, torch.ones(2), None, None)


class HvTagTests(unittest.TestCase):
    THRESHOLDS = dict(min_z_fraction=0.8, min_auto_certainty=0.5)

    def test_manual_tag_wins_and_non_fibers_are_untagged(self):
        vertical = _fiber(1, [(0, 0, 0), (0, 10, 0)], hv={'manual_tag': 'V'})
        self.assertEqual(fiber_collection_hv_tag(vertical, **self.THRESHOLDS), 'V')
        regular = _fiber(2, [(0, 0, 0), (100, 0, 0)], kind=None)
        self.assertIsNone(fiber_collection_hv_tag(regular, **self.THRESHOLDS))

    def test_geometric_fallback_reads_the_id_ordered_points(self):
        # Points inserted out of id order; in id order the path is a straight
        # column (or row), in insertion order it doubles back.
        fiber = _fiber(3, [(0, 0, 0), (100, 0, 0), (50, 0, 0)])
        fiber['points'] = {2: fiber['points'][1], 0: fiber['points'][0], 1: fiber['points'][2]}
        self.assertEqual(fiber_collection_hv_tag(fiber, **self.THRESHOLDS), 'V')
        flat = _fiber(4, [(0, 0, 0), (0, 0, 100), (0, 0, 50)])
        flat['points'] = {2: flat['points'][1], 0: flat['points'][0], 1: flat['points'][2]}
        self.assertEqual(fiber_collection_hv_tag(flat, **self.THRESHOLDS), 'H')


class InwardWindingDirectionTests(unittest.TestCase):
    def test_identity_transform_points_at_the_spiral_axis(self):
        # With scroll == spiral space the winding gradient is the outward
        # radial direction, so the inward direction points at the yx origin.
        direction = inward_winding_direction(lambda zyx: zyx, [[0.0, 10.0, 0.0], [5.0, 0.0, -4.0]])
        np.testing.assert_allclose(direction, [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], atol=1e-6)


class _StubContext:
    @staticmethod
    def make(**overrides):
        context = FitContext.__new__(FitContext)
        context.config = FitConfig(Config({
            'pcl_fiber_link_side_filter': True,
            'pcl_fiber_link_model_direction_step': 10,
            **overrides}).as_dict())
        context.fiber_catalog = {'f': _fiber(1, [(0, 0, 0), (100, 0, 0)], hv={'manual_tag': 'V'})}
        context._fiber_link_direction_source = 'umbilicus'
        context._relink_fibers_to_patches = mock.Mock()
        return context


class DirectionSwitchHookTests(unittest.TestCase):
    def test_switches_to_the_model_once_at_the_threshold(self):
        context = _StubContext.make()
        context._maybe_relink_fibers_for_direction(9)
        context._relink_fibers_to_patches.assert_not_called()
        context._maybe_relink_fibers_for_direction(10)
        context._relink_fibers_to_patches.assert_called_once_with('model', iteration=10)
        # The relink records the switch; a later step does nothing.
        context._fiber_link_direction_source = 'model'
        context._relink_fibers_to_patches.reset_mock()
        context._maybe_relink_fibers_for_direction(11)
        context._relink_fibers_to_patches.assert_not_called()

    def test_a_checkpoint_before_the_threshold_returns_to_the_umbilicus(self):
        context = _StubContext.make()
        context._fiber_link_direction_source = 'model'
        context._maybe_relink_fibers_for_direction(3)
        context._relink_fibers_to_patches.assert_called_once_with('umbilicus', iteration=3)

    def test_no_op_without_the_filter_or_without_fibers(self):
        context = _StubContext.make(pcl_fiber_link_side_filter=False)
        context._maybe_relink_fibers_for_direction(10)
        context._relink_fibers_to_patches.assert_not_called()
        context = _StubContext.make()
        context.fiber_catalog = {}
        context._maybe_relink_fibers_for_direction(10)
        context._relink_fibers_to_patches.assert_not_called()


class RelinkFibersTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(
            point_collection, 'can_use_surface_index_backend', return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _context(self):
        context = FitContext.__new__(FitContext)
        context.config = FitConfig(Config({
            'pcl_fiber_link_side_filter': True,
            'pcl_fiber_link_side_margin_voxels': 0.0,
        }).as_dict())
        # Vertical fiber at y = 21 between a patch in front (y = 20) and one
        # behind (y = 22); horizontal fiber at y = 21 likewise.
        vertical = _fiber(1, [(60.0, 21.0, 0.0), (61.0, 21.0, 1.0), (62.0, 21.0, 2.0)],
                          hv={'manual_tag': 'V'})
        horizontal = _fiber(2, [(60.0, 21.0, 5.0), (60.0, 21.0, 6.0), (60.0, 21.0, 7.0)],
                            hv={'manual_tag': 'H'})
        for pcl in (vertical, horizontal):
            for point in pcl['points'].values():
                point['on_patch'] = {'id': 'stale', 'distance': 0.0, 'ij': [0, 0]}
        context.fiber_catalog = {'1': vertical, '2': horizontal}
        context.verified_patches = {
            'front': _y_plane(20.0, 40.0, -20.0), 'behind': _y_plane(22.0, 40.0, -20.0)}
        context.link_distance_tolerance = 2.0
        context._baked_voxel_scale = mock.Mock(return_value=1.0)
        context._rematerialize_fiber_views = mock.Mock()
        context.dist = mock.Mock(is_main_process=True)
        return context, vertical, horizontal

    def test_relink_applies_mirrored_side_rules_and_records_the_source(self):
        context, vertical, horizontal = self._context()
        inward = mock.Mock(side_effect=lambda zyxs: np.tile([0.0, -1.0, 0.0], (len(zyxs), 1)))
        with mock.patch.object(
                context, '_fiber_link_inward_direction', return_value=inward) as direction:
            context._relink_fibers_to_patches('model', iteration=10)
        direction.assert_called_once_with('model', umbilicus_z_to_yx=None)
        self.assertEqual(
            [p['on_patch']['id'] for _, p in sorted(vertical['points'].items())], ['front'] * 3)
        self.assertEqual(
            [p['on_patch']['id'] for _, p in sorted(horizontal['points'].items())], ['behind'] * 3)
        self.assertEqual(context._fiber_link_direction_source, 'model')
        context._rematerialize_fiber_views.assert_called_once_with()

    def test_side_rules_follow_the_hv_tag(self):
        context, vertical, horizontal = self._context()
        rules = context._fiber_link_side_rules({1: vertical, 2: horizontal, 3: _fiber(3, [(0, 0, 0)], kind=None)})
        self.assertEqual(rules, {1: SIDE_FRONT, 2: SIDE_BEHIND})
        context.config = FitConfig(Config({'pcl_fiber_link_side_filter': False}).as_dict())
        self.assertEqual(context._fiber_link_side_rules({1: vertical}), {})


class YxScale:
    """Scroll -> spiral map scaling y by `sy` and x by `sx` about the axis."""

    def __init__(self, sy, sx):
        self.sy, self.sx = float(sy), float(sx)

    def __call__(self, zyxs):
        out = zyxs.clone()
        out[..., 1] = out[..., 1] * self.sy
        out[..., 2] = out[..., 2] * self.sx
        return out


def _points_on_axes():
    # Points on the +x, +y axes and a diagonal, at several z.
    return torch.tensor([
        [10.0, 0.0, 50.0],
        [20.0, 50.0, 0.0],
        [30.0, 30.0, 40.0],
        [40.0, 0.0, -70.0],
        [50.0, -20.0, 0.0],
    ])


def test_stretch_is_unity_under_identity_and_the_scale_under_uniform_scaling():
    points = _points_on_axes()
    identity = torch.distributions.transforms.identity_transform
    torch.testing.assert_close(
        get_radial_normal_stretch(identity, points), torch.ones(5),
        rtol=0, atol=1e-5)
    torch.testing.assert_close(
        get_radial_normal_stretch(YxScale(2.5, 2.5), points),
        torch.full((5,), 2.5), rtol=0, atol=1e-4)


def test_stretch_follows_the_normal_direction_under_anisotropic_scaling():
    # Scaling x alone: the sheet normal at a point on the x axis is along x
    # (stretch 2), at a point on the y axis along y (stretch 1). Spiral-space
    # normals of the transformed points are still radial, so on the diagonal
    # the pulled-back covector is (0, y, 2x) normalised over the image.
    points = _points_on_axes()
    transform = YxScale(1.0, 2.0)
    stretch = get_radial_normal_stretch(transform, points)
    torch.testing.assert_close(stretch[[0, 3]], torch.full((2,), 2.0), rtol=0, atol=1e-4)
    torch.testing.assert_close(stretch[[1, 4]], torch.full((2,), 1.0), rtol=0, atol=1e-4)
    image = transform(points[2:3])
    n = torch.nn.functional.normalize(image[0, 1:], dim=0)
    expected = torch.sqrt((n[0] * 1.0) ** 2 + (n[1] * 2.0) ** 2)
    torch.testing.assert_close(stretch[2], expected, rtol=0, atol=1e-4)



def test_accumulate_bake_scale_multiplies_successive_epochs():
    zyxs = _points_on_axes().numpy()
    first = accumulate_radial_offset_bake_scale(YxScale(2.0, 2.0), zyxs)
    np.testing.assert_allclose(first, 2.0, atol=1e-4)
    second = accumulate_radial_offset_bake_scale(
        YxScale(1.5, 1.5), zyxs * 2.0, first)
    np.testing.assert_allclose(second, 3.0, atol=1e-4)
    assert second.dtype == np.float32


def test_flat_bundle_applies_the_bake_scale_and_flags_nonzero_offsets():
    zyxs = np.zeros((4, 3), dtype=np.float32)
    windings = np.zeros(4, dtype=np.float32)
    plain = _build_strip_flat_bundle(
        [(zyxs, windings, np.full(4, 4.0, np.float32)),
         (zyxs, windings, None)], torch.device('cpu'))
    assert plain['has_radial_offsets'] is True
    torch.testing.assert_close(
        plain['radial_offsets'], torch.tensor([4.0] * 4 + [0.0] * 4))
    scaled = _build_strip_flat_bundle(
        [(zyxs, windings, np.full(4, 4.0, np.float32),
          np.array([1.0, 1.5, 2.0, 0.5], np.float32)),
         (zyxs, windings, np.full(4, 4.0, np.float32), None)],
        torch.device('cpu'))
    torch.testing.assert_close(
        scaled['radial_offsets'],
        torch.tensor([4.0, 6.0, 8.0, 2.0] + [4.0] * 4))
    zeros = _build_strip_flat_bundle(
        [(zyxs, windings, np.zeros(4, np.float32),
          np.full(4, 3.0, np.float32))], torch.device('cpu'))
    assert zeros['has_radial_offsets'] is False


def test_materialize_carries_the_catalog_points_bake_scale_through_decimation():
    vertical = [[float(z), 0.0, 100.0] for z in range(0, 100, 10)]
    points = {
        i: {"id": i, "collectionId": 1, "p": [z[2], z[1], z[0]],
            "zyx": np.asarray(z, dtype=np.float32),
            "winding_annotation": float("nan"),
            "radial_offset_bake_scale": 1.0 + 0.1 * i}
        for i, z in enumerate(vertical)
    }
    del points[4]["radial_offset_bake_scale"]  # an unbaked point: scale 1
    catalog = {"v": {
        "id": 1, "file_basename": "v.json", "sampling_group": "fibers",
        "metadata": {"logical_input_id": "v", "logical_input_kind": "fiber",
                     "winding_is_absolute": False, "hv_classification": {}},
        "points": points, "kept_orig_indices": np.arange(10),
        "control_line_indices": np.arange(10), "branches": [],
    }}
    _, strips, *_ = materialize_fiber_fit_inputs(
        catalog, {}, z_begin=-10, z_end=200, z_margin=0,
        min_point_spacing=25.0, use_links=False,
        vertical_min_z_fraction=0.8, vertical_min_auto_certainty=0.5,
        vertical_radial_offset=4.0)
    strip, = strips
    assert strip['is_vertical']
    kept = strip['zyxs'][:, 0].round().astype(int) // 10
    expected = np.array(
        [1.0 if i == 4 else 1.0 + 0.1 * i for i in kept], dtype=np.float32)
    assert len(kept) < 10  # decimation happened
    np.testing.assert_allclose(strip['radial_offset_bake_scale'], expected, rtol=1e-6)
    assert strip['radial_offset_bake_scale'].dtype == np.float32




def test_dt_target_cache_converts_offsets_through_the_stretch():
    # One on-sheet strip and one back-face strip 4 scroll voxels outside it,
    # under a 2x radial stretch: with the physical offset both strips share a
    # target; with the offset read as a spiral constant they would not.
    dr = torch.tensor(12.0)
    transform = YxScale(2.0, 2.0)
    theta = torch.linspace(0.1, 2.0, 40)
    radius_spiral = dr * (5 + theta / (2 * np.pi))
    sheet_scroll = torch.stack([
        torch.full_like(theta, 10.0),
        torch.sin(theta) * radius_spiral / 2.0,
        torch.cos(theta) * radius_spiral / 2.0], dim=-1)
    unit = torch.nn.functional.normalize(sheet_scroll[:, 1:], dim=-1)
    back = sheet_scroll.clone()
    back[:, 1:] += 4.0 * unit
    zyxs = torch.cat([sheet_scroll, back])
    starts = torch.tensor([0, 40, 80])
    windings = torch.zeros(80)

    physical = torch.cat([torch.zeros(40), torch.full((40,), 4.0)])
    cache = compute_strip_dt_target_cache(
        transform, dr, zyxs, starts, windings=windings, radial_offsets=physical)
    # Both strips normalise onto the same sheet radius: the whole-object
    # target is identical and finite for both.
    assert bool(cache['valid'].all())
    torch.testing.assert_close(
        cache['target_relative'][0], cache['target_relative'][1],
        rtol=0, atol=1e-3)


@pytest.mark.parametrize('iteration, direction', [(0, 'umbilicus'), (10000, 'model')])
def test_fiber_revision_preserves_link_direction_and_radial_offset(
        context, monkeypatch, iteration, direction):
    pcl = relink._regular_pcl(5, [[50, 80, 80], [100, 80, 80]])
    pcl['metadata'].update(
        logical_input_kind='fiber', logical_input_id='startup',
        hv_classification={'manual_tag': 'V'})
    pcl['sampling_group'] = 'fibers'
    context._source_point_collections = {5: pcl}
    context.config.update({
        'pcl_fiber_link_side_filter': True,
        'pcl_vertical_fiber_radial_offset_enabled': True,
        'pcl_vertical_fiber_radial_offset_voxels': 7.0,
        'pcl_link_distance_tolerance': 0.75,
    })
    directions = []

    def inward(self, source, **kwargs):
        directions.append(source)
        return lambda zyxs: np.broadcast_to([0., 0., -1.], zyxs.shape)

    monkeypatch.setattr(FitContext, '_fiber_link_inward_direction', inward)
    candidate = context.prepare_input_changes([], current_iteration=iteration)
    assert directions == [direction]
    assert candidate._fiber_link_direction_source == direction
    assert candidate.link_distance_tolerance == 0.75
    strip, = candidate.unattached_pcl_strips
    assert strip['is_vertical']
    np.testing.assert_array_equal(strip['radial_offsets'], 7.0)
    assert context._fiber_link_direction_source == 'umbilicus'
