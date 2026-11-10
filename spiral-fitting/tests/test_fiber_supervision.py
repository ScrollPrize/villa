"""Fiber supervision: classification and radial offsets."""

import numpy as np
import torch

from dt_targets import compute_strip_dt_target_cache
from fit_spiral import (
    _build_strip_flat_bundle,
    accumulate_radial_offset_bake_scale,
)
from pcl_views import fiber_training_views, prepare_fiber_catalog
from sample_spiral import get_radial_normal_stretch
from spiral_helpers import classify_fiber_hv


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


def _fiber_strips(catalog, **kwargs):
    """Strips of resident fibers ``catalog`` with no links or patches."""
    _, strips, _, _ = fiber_training_views(
        prepare_fiber_catalog(catalog), [], {}, z_begin=-10, z_end=200,
        z_margin=0, vertical_min_z_fraction=0.8,
        vertical_min_auto_certainty=0.5, **kwargs)
    return strips


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
    def member(cid, logical_id, zyxs, hv=None):
        points = {
            i: {"id": i, "collectionId": cid,
                "p": [z[2], z[1], z[0]],
                "zyx": np.asarray(z, dtype=np.float32),
                "winding_annotation": float("nan")}
            for i, z in enumerate(zyxs)
        }
        return {
            "id": cid, "sampling_group": "fibers",
            "metadata": {"hv_classification": hv or {}},
            "points": points,
        }

    vertical = [[float(z), 0.0, 100.0] for z in range(0, 100, 10)]
    horizontal = [[0.0, 0.0, 100.0 + x] for x in range(0, 100, 10)]
    catalog = {
        "v": member(1, "v", vertical),
        "h": member(2, "h", horizontal),
        "forced": member(3, "forced", horizontal, {"manual_tag": "V"}),
    }
    strips = _fiber_strips(catalog, min_point_spacing=0,
                           vertical_radial_offset=4.0)
    by_id = {strip['id']: strip for strip in strips}
    assert by_id[1]['hv_tag'] == 'V' and by_id[1]['is_vertical']
    assert by_id[2]['hv_tag'] == 'H' and not by_id[2]['is_vertical']
    assert by_id[3]['hv_tag'] == 'V' and by_id[3]['is_vertical']
    # Vertical strips carry the radial target offset, horizontals never do.
    assert np.all(by_id[1]['radial_offsets'] == 4.0)
    assert np.all(by_id[3]['radial_offsets'] == 4.0)
    assert np.all(by_id[2]['radial_offsets'] == 0.0)
    strips_off = _fiber_strips(catalog, min_point_spacing=0)
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
    context.tracks = []
    context.prepared_main_tracks = None
    context.verified_patches_list = []
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
        "id": 1, "sampling_group": "fibers",
        "metadata": {"hv_classification": {}},
        "points": points,
    }}
    strip, = _fiber_strips(catalog, min_point_spacing=25.0,
                           vertical_radial_offset=4.0)
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
