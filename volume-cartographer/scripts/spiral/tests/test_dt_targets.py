import numpy as np
import torch
from types import SimpleNamespace

from dt_targets import (
    DtTargetCacheManager,
    compute_strip_dt_target_cache,
    prepare_patch_dt_target_samples,
    patch_dt_target_in_sample_frame,
    strip_dt_target_in_sample_frame,
    select_whole_object_target,
    snap_patch_dt_target,
    snap_strip_dt_target,
)


def test_dt_target_cache_manager_refreshes_each_kind_independently():
    updates = []
    manager = DtTargetCacheManager(3, lambda kind, cache: updates.append((kind, cache)))
    calls = {'patch': 0, 'track': 0}

    def compute(kind):
        calls[kind] += 1
        return calls[kind]

    assert manager.get('patch', 10, lambda: compute('patch')) == 1
    assert manager.get('patch', 12, lambda: compute('patch')) == 1
    assert manager.get('patch', 13, lambda: compute('patch')) == 2
    assert manager.get('track', 13, lambda: compute('track')) == 1
    assert updates == [('patch', 1), ('track', 1)]


def _cache(ijs, theta, adjustment, target, valid=None, anchor_dist_sq_limit=1.e6):
    if valid is None:
        valid = [True] * len(theta)
    return {
        'ijs': torch.tensor([ijs], dtype=torch.float32),
        'theta': torch.tensor([theta], dtype=torch.float32),
        'relative_adjustment': torch.tensor([adjustment], dtype=torch.float32),
        'point_valid': torch.tensor([valid]),
        'target_relative': torch.tensor([target], dtype=torch.float32),
        'valid': torch.tensor([True]),
        'anchor_dist_sq_limit': torch.tensor([anchor_dist_sq_limit], dtype=torch.float32),
    }


def test_patch_target_transfers_through_uv_anchor():
    dr = torch.tensor(10.0)
    cache = _cache([[0.0, 0.0]], [1.0], [0.0], target=2.0)
    target, valid = snap_patch_dt_target(
        sample_ijs=torch.tensor([[[0.1, 0.1], [0.2, 0.2]]]),
        sample_theta=torch.tensor([[1.1, 1.2]]),
        sample_adjustments=torch.zeros(1, 2),
        dr_per_winding=dr,
        cache=cache,
        cache_idx=torch.tensor([0]),
    )
    assert bool(valid.all())
    assert torch.equal(target, torch.tensor([[20.0]]))


def test_patch_target_uses_nearest_valid_uv_adjustment():
    dr = torch.tensor(10.0)
    cache = _cache(
        [[0.0, 0.0], [10.0, 10.0]],
        [1.0, 2.0],
        [7.0, 2.0],
        target=5.0,
        valid=[False, True],
    )
    target, valid = snap_patch_dt_target(
        sample_ijs=torch.tensor([[[0.0, 0.0], [9.9, 10.0]]]),
        sample_theta=torch.tensor([[1.0, 2.1]]),
        sample_adjustments=torch.tensor([[0.0, -10.0]]),
        dr_per_winding=dr,
        cache=cache,
        cache_idx=torch.tensor([0]),
    )
    # target_relative 5 + sample adjustment -1 - cached relative adjustment 2
    assert bool(valid.all())
    assert torch.equal(target, torch.tensor([[20.0]]))


def test_patch_target_applies_local_theta_crossing():
    dr = torch.tensor(10.0)
    cache = _cache([[0.0, 0.0]], [6.2], [0.0], target=3.0)
    target, valid = snap_patch_dt_target(
        sample_ijs=torch.tensor([[[0.1, 0.0]]]),
        sample_theta=torch.tensor([[0.1]]),
        sample_adjustments=torch.zeros(1, 1),
        dr_per_winding=dr,
        cache=cache,
        cache_idx=torch.tensor([0]),
    )
    # Moving from theta 6.2 to 0.1 contributes cache-frame adjustment -1,
    # so target winding 3 in the cache frame is winding 4 in the sample frame.
    assert bool(valid.all())
    assert torch.equal(target, torch.tensor([[40.0]]))

def test_patch_target_rejects_anchor_beyond_distance_limit():
    dr = torch.tensor(10.0)
    # Nearest valid cache point is 10 UV cells away but the limit only allows 2:
    # the strip must be reported invalid (median fallback), not transferred through
    # an anchor that may violate the |dtheta| < pi assumption.
    cache = _cache([[10.0, 0.0]], [1.0], [0.0], target=2.0, anchor_dist_sq_limit=4.0)
    target, valid = snap_patch_dt_target(
        sample_ijs=torch.tensor([[[0.0, 0.0]]]),
        sample_theta=torch.tensor([[1.0]]),
        sample_adjustments=torch.zeros(1, 1),
        dr_per_winding=dr,
        cache=cache,
        cache_idx=torch.tensor([0]),
    )
    assert torch.equal(valid, torch.tensor([False]))

    # The wrapper then falls back to the snapped sample median.
    radii = torch.tensor([[31.0, 32.0, 33.0]])
    target = patch_dt_target_in_sample_frame(
        radii, torch.zeros(1, 3, 2), torch.full((1, 3), 1.0), torch.zeros(1, 3),
        dr, cache, np.array([0]),
    )
    assert torch.equal(target, torch.tensor([[30.0]]))


def _strip_cache(keys, key_scale, theta, adjustment, target_relative, valid):
    return {
        'keys': torch.tensor(keys, dtype=torch.int64),
        'key_scale': key_scale,
        'theta': torch.tensor(theta, dtype=torch.float32),
        'adjustment': torch.tensor(adjustment, dtype=torch.float32),
        'target_relative': torch.tensor(target_relative, dtype=torch.float32),
        'valid': torch.tensor(valid),
        'num_points': len(keys),
    }


def test_strip_target_transfers_through_index_anchor():
    dr = torch.tensor(10.0)
    cache = _strip_cache([0, 10, 20], 100, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [5.0], [True])
    target, valid = snap_strip_dt_target(
        sample_local_idx=torch.tensor([[9, 11]]),
        sample_theta=torch.tensor([[2.05, 2.1]]),
        sample_adjustments=torch.zeros(1, 2),
        dr_per_winding=dr,
        cache=cache,
        cache_idx=torch.tensor([0]),
    )
    assert bool(valid.all())
    assert torch.equal(target, torch.tensor([[50.0]]))


def test_strip_target_anchors_within_own_strip_with_adjustments_and_crossing():
    dr = torch.tensor(10.0)
    # Two strips; strip 1's cached point at local index 50 carries adjustment -1 and
    # sits just before theta 0, while the sample has crossed (theta 0.1, adjustment +1).
    cache = _strip_cache(
        [0, 50, 100, 150], 100,
        [1.0, 2.0, 6.2, 6.2],
        [0.0, 0.0, 0.0, -1.0],
        [2.0, 3.0],
        [True, True],
    )
    target, valid = snap_strip_dt_target(
        sample_local_idx=torch.tensor([[49]]),
        sample_theta=torch.tensor([[0.1]]),
        sample_adjustments=torch.tensor([[10.0]]),
        dr_per_winding=dr,
        cache=cache,
        cache_idx=torch.tensor([1]),
    )
    # target 3 + sample adjustment 1 - cache adjustment -1 - crossing -1 = 6
    assert bool(valid.all())
    assert torch.equal(target, torch.tensor([[60.0]]))

    # A sample at strip 1's local index 0 must anchor to strip 1's own first point
    # (gap 0), never to strip 0's nearer-in-key last point.
    target, valid = snap_strip_dt_target(
        sample_local_idx=torch.tensor([[0]]),
        sample_theta=torch.tensor([[6.2]]),
        sample_adjustments=torch.zeros(1, 1),
        dr_per_winding=dr,
        cache=cache,
        cache_idx=torch.tensor([1]),
    )
    assert bool(valid.all())
    assert torch.equal(target, torch.tensor([[30.0]]))


def test_strip_target_reports_invalid_cache_entries():
    dr = torch.tensor(10.0)
    cache = _strip_cache([0], 100, [1.0], [0.0], [7.0], [False])
    _, valid = snap_strip_dt_target(
        sample_local_idx=torch.tensor([[0]]),
        sample_theta=torch.tensor([[1.0]]),
        sample_adjustments=torch.zeros(1, 1),
        dr_per_winding=dr,
        cache=cache,
        cache_idx=torch.tensor([0]),
    )
    assert torch.equal(valid, torch.tensor([False]))

    # An entirely empty cache reports every strip invalid.
    cache = _strip_cache([], 100, [], [], [7.0], [True])
    target, valid = snap_strip_dt_target(
        sample_local_idx=torch.tensor([[0]]),
        sample_theta=torch.tensor([[1.0]]),
        sample_adjustments=torch.zeros(1, 1),
        dr_per_winding=dr,
        cache=cache,
        cache_idx=torch.tensor([0]),
    )
    assert target.shape == (1, 1)
    assert torch.equal(valid, torch.tensor([False]))


def test_strip_dt_target_in_sample_frame_dispatches_and_falls_back():
    dr = torch.tensor(10.0)
    radii = torch.tensor([[31.0, 32.0, 33.0]])  # snapped sample median = winding 3
    local_idx = np.array([[0, 1, 2]])
    theta = torch.full((1, 3), 1.0)
    adjustments = torch.zeros(1, 3)
    cache_idx = np.array([0])

    # No cache (legacy strip-median mode): snapped sample median.
    target = strip_dt_target_in_sample_frame(radii, local_idx, theta, adjustments, dr, None, cache_idx)
    assert torch.equal(target, torch.tensor([[30.0]]))

    # Valid cache entry: the whole-strip target wins over the sample median.
    cache = _strip_cache([0, 10], 100, [1.0, 1.0], [0.0, 0.0], [5.0], [True])
    target = strip_dt_target_in_sample_frame(radii, local_idx, theta, adjustments, dr, cache, cache_idx)
    assert torch.equal(target, torch.tensor([[50.0]]))

    # Invalid cache entry: falls back to the snapped sample median.
    cache = _strip_cache([0, 10], 100, [1.0, 1.0], [0.0, 0.0], [5.0], [False])
    target = strip_dt_target_in_sample_frame(radii, local_idx, theta, adjustments, dr, cache, cache_idx)
    assert torch.equal(target, torch.tensor([[30.0]]))


def test_patch_dt_target_in_sample_frame_broadcasts_patch_indices():
    dr = torch.tensor(10.0)
    radii = torch.full((2, 1, 2), 41.0)  # snapped sample median = winding 4
    sample_ijs = torch.zeros(2, 1, 2, 2)
    theta = torch.full((2, 1, 2), 1.0)
    adjustments = torch.zeros(2, 1, 2)
    patch_indices = np.array([0])  # broadcast over the leading row/column axis

    cache = _cache([[0.0, 0.0]], [1.0], [0.0], target=2.0)
    target = patch_dt_target_in_sample_frame(radii, sample_ijs, theta, adjustments, dr, cache, patch_indices)
    assert torch.equal(target, torch.full((2, 1, 1), 20.0))

    target = patch_dt_target_in_sample_frame(radii, sample_ijs, theta, adjustments, dr, None, patch_indices)
    assert torch.equal(target, torch.full((2, 1, 1), 40.0))


def _bimodal_strip_zyxs(dr):
    # 40 ordered points at increasing theta 0.5 -> 2.5 (no theta=0 crossing): the first
    # 15 sit exactly on winding 3, the remaining 25 exactly on winding 4.
    t = torch.linspace(0.5, 2.5, 40)
    winding = torch.where(torch.arange(40) < 15, 3.0, 4.0)
    radius = (winding + t / (2 * np.pi)) * dr
    return torch.stack([torch.zeros(40), torch.sin(t) * radius, torch.cos(t) * radius], dim=-1)


def test_strip_cache_whole_object_target_transfers_to_minority_mode_sample():
    dr = torch.tensor(10.0)
    zyxs = _bimodal_strip_zyxs(dr)
    for max_points in (None, 10):
        cache = compute_strip_dt_target_cache(
            lambda x: x, dr,
            zyxs, torch.tensor([0, 40]),
            windings=None, floating_threshold=0.25, num_points_per_strip=max_points,
        )
        assert bool(cache['valid'].all())
        assert torch.equal(cache['target_relative'], torch.tensor([4.0]))

        # Sample only from the winding-3 minority: the legacy median target would be
        # 30, but the anchored whole-strip target must come out as 40.
        sample_local_idx = torch.arange(5)[None, :]
        from sample_spiral import get_theta_and_radii, unwrap_shifted_radii
        theta, _, shifted = get_theta_and_radii(zyxs[sample_local_idx][..., 1:], dr)
        shifted, adjustments = unwrap_shifted_radii(theta, shifted, dr)
        strip_median = shifted.median(dim=-1, keepdim=True).values
        assert torch.allclose(strip_median, torch.tensor([[30.0]]))
        target, valid = snap_strip_dt_target(
            sample_local_idx, theta, adjustments, dr, cache, torch.tensor([0]),
        )
        assert bool(valid.all())
        assert torch.allclose(target, torch.tensor([[40.0]]))


def test_strip_cache_handles_theta_crossings_between_cache_and_sample_frames():
    dr = torch.tensor(10.0)
    # One strip spiralling outward across two theta=0 crossings: continuous angle
    # 0.5 -> 0.5 + 4*pi on a perfect spiral starting at winding 3.
    t = torch.linspace(0.5, 0.5 + 4 * np.pi, 41)
    radius = (3.0 + t / (2 * np.pi)) * dr
    zyxs = torch.stack([torch.zeros(41), torch.sin(t) * radius, torch.cos(t) * radius], dim=-1)
    cache = compute_strip_dt_target_cache(
        lambda x: x, dr,
        zyxs, torch.tensor([0, 41]),
        windings=None, floating_threshold=0.25, num_points_per_strip=15,
    )
    assert torch.equal(cache['target_relative'], torch.tensor([3.0]))

    # Sample the last two points (past both crossings): in their own unwrap frame the
    # strip lies at winding 5, and the transferred target must agree.
    sample_local_idx = torch.tensor([[39, 40]])
    from sample_spiral import get_theta_and_radii, unwrap_shifted_radii
    theta, _, shifted = get_theta_and_radii(zyxs[sample_local_idx][..., 1:], dr)
    shifted, adjustments = unwrap_shifted_radii(theta, shifted, dr)
    target, valid = snap_strip_dt_target(
        sample_local_idx, theta, adjustments, dr, cache, torch.tensor([0]),
    )
    assert bool(valid.all())
    assert torch.allclose(target, torch.tensor([[50.0]]))


def test_strip_cache_max_stride_adds_samples_and_keeps_endpoints():
    dr = torch.tensor(10.0)
    n = 1000
    theta = torch.linspace(0.1, 1.0, n)
    radius = (3.0 + theta / (2 * np.pi)) * dr
    zyxs = torch.stack([
        torch.zeros(n), torch.sin(theta) * radius, torch.cos(theta) * radius,
    ], dim=-1)
    cache = compute_strip_dt_target_cache(
        lambda x: x, dr,
        zyxs, torch.tensor([0, n]),
        floating_threshold=0.25, num_points_per_strip=3, max_stride=96,
    )
    local_indices = cache['keys']
    assert local_indices[0] == 0
    assert local_indices[-1] == n - 1
    assert int(torch.diff(local_indices).max()) <= 96
    assert cache['num_points'] > 3


def test_patch_max_stride_is_converted_from_voxels_using_scale():
    patch = SimpleNamespace(
        _sampling_valid_quad_mask_np=np.ones((100, 100), dtype=bool),
        scale=torch.tensor([0.5, 0.25]),
    )
    prepare_patch_dt_target_samples([patch], num_points=1, max_stride_voxels=128)
    # 128 voxels correspond to at most 64 row cells and 32 column cells.
    assert patch._dt_target_block_shape == (2, 4)
    # Anchor limit = (2 block diagonals)^2 with 50x25-cell blocks.
    assert patch._dt_target_anchor_max_dist_sq == 4.0 * (50.0 ** 2 + 25.0 ** 2)


def test_whole_object_target_does_not_let_outer_tail_override_attached_majority():
    dr = torch.tensor(10.0)
    values = torch.tensor([30.0] * 7 + [39.0] * 3)
    assert select_whole_object_target(values, dr, 0.25) == 3


def test_whole_object_target_grabs_outer_sheet_when_majority_is_floating():
    dr = torch.tensor(10.0)
    values = torch.tensor([34.0, 34.5, 35.0, 35.5, 36.0])
    assert select_whole_object_target(values, dr, 0.25) == 4
