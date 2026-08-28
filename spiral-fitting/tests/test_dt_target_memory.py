import numpy as np
import pytest
import torch

from dt_targets import (
    _compact_integer_tensor,
    compute_strip_dt_target_cache,
    endpoint_strip_dt_target_in_sample_frame,
    patch_dt_target_in_sample_frame,
    strip_dt_target_in_sample_frame,
)
from tracks import iter_track_losses, prepare_main_phase_tracks


class _IdentityTransform:
    def __call__(self, points):
        return points

    @property
    def inv(self):
        return self


def _endpoint_cache(**overrides):
    cache = {
        'frame': 'strip_endpoints',
        'anchor_theta': torch.tensor([[0.0, 3.0]]),
        'anchor_adjustment': torch.tensor([[0, 5]], dtype=torch.int16),
        'target_relative': torch.tensor([7], dtype=torch.int16),
        'valid': torch.tensor([True]),
        'num_points': 2,
    }
    cache.update(overrides)
    return cache


def test_patch_theta_potential_cache_is_used_directly():
    cache = {
        'frame': 'theta_potential',
        'target_relative': torch.tensor([2, 7], dtype=torch.int16),
        'valid': torch.tensor([True, True]),
    }
    target = patch_dt_target_in_sample_frame(
        torch.tensor([[71.0, 72.0], [21.0, 22.0]]),
        torch.zeros((2, 2, 2)),
        torch.zeros((2, 2)),
        torch.zeros((2, 2)),
        torch.tensor(10.0),
        cache,
        torch.tensor([1, 0]),
    )
    torch.testing.assert_close(target, torch.tensor([[70.0], [20.0]]))


def test_compact_integer_storage_widens_instead_of_overflowing():
    assert _compact_integer_tensor(
        torch.tensor([0, 32767]), 'test').dtype == torch.int16
    assert _compact_integer_tensor(
        torch.tensor([0, 32768]), 'test').dtype == torch.int32
    assert _compact_integer_tensor(
        torch.tensor([0, 2**31]), 'test').dtype == torch.int64
    with pytest.raises(ValueError, match='not integer-valued'):
        _compact_integer_tensor(torch.tensor([1.25]), 'test')


def test_strip_cache_has_one_two_endpoint_representation():
    zyxs = torch.tensor([
        [0.0, 0.0, 30.0], [0.0, 0.0, 30.0], [0.0, 0.0, 30.0],
        [0.0, 0.0, 40.0], [0.0, 0.0, 40.0], [0.0, 0.0, 40.0],
    ])
    cache = compute_strip_dt_target_cache(
        _IdentityTransform(), torch.tensor(10.0), zyxs,
        torch.tensor([0, 3, 6]))

    assert set(cache) == {
        'frame', 'anchor_theta', 'anchor_adjustment', 'target_relative',
        'valid', 'num_points',
    }
    assert cache['frame'] == 'strip_endpoints'
    assert cache['anchor_theta'].shape == (2, 2)
    assert cache['anchor_adjustment'].shape == (2, 2)
    assert cache['anchor_theta'].device == zyxs.device
    assert cache['target_relative'].dtype == torch.int16
    torch.testing.assert_close(
        cache['target_relative'].to(torch.int64), torch.tensor([3, 4]))


def test_empty_strip_cache_uses_the_same_representation():
    cache = compute_strip_dt_target_cache(
        _IdentityTransform(), torch.tensor(10.0), torch.empty((0, 3)),
        torch.tensor([0, 0, 0]))
    assert cache['frame'] == 'strip_endpoints'
    assert cache['anchor_theta'].shape == (2, 2)
    assert not cache['valid'].any()


def test_track_uses_first_endpoint_of_shared_cache():
    target = strip_dt_target_in_sample_frame(
        torch.tensor([[72.0, 73.0]]),
        torch.tensor([[0, 9]]),
        torch.tensor([[-3.0, 0.0]]),
        torch.zeros((1, 2)),
        torch.tensor(10.0),
        _endpoint_cache(
            anchor_theta=torch.tensor([[3.0, 0.0]]),
            anchor_adjustment=torch.tensor([[0, 4]], dtype=torch.int16)),
        torch.tensor([0]),
    )
    # Endpoint zero crossed from +3 to -3, so the target gains one winding.
    torch.testing.assert_close(target, torch.tensor([[80.0]]))


def test_track_falls_back_when_first_endpoint_is_not_sampled():
    target = strip_dt_target_in_sample_frame(
        torch.tensor([[72.0, 73.0]]),
        torch.tensor([[4, 9]]),
        torch.zeros((1, 2)),
        torch.zeros((1, 2)),
        torch.tensor(10.0),
        _endpoint_cache(target_relative=torch.tensor([9], dtype=torch.int16)),
        torch.tensor([0]),
    )
    torch.testing.assert_close(target, torch.tensor([[70.0]]))


def test_pcl_target_uses_selected_endpoint_adjustment():
    target = endpoint_strip_dt_target_in_sample_frame(
        torch.tensor([[72.0, 73.0]]), torch.tensor(10.0),
        _endpoint_cache(), torch.tensor([0]),
        sample_anchor_theta=torch.tensor([-3.0]),
        sample_anchor_adjustment=torch.tensor([20.0]),
        anchor_at_end=torch.tensor([True]),
    )
    # 7 + current adjustment 2 - cached end adjustment 5, plus one winding
    # because endpoint one crossed from +3 to -3.
    torch.testing.assert_close(target, torch.tensor([[50.0]]))


def test_chunked_strip_cache_keeps_the_same_device_representation():
    zyxs = torch.tensor([
        [0.0, 0.0, 30.0], [0.0, 0.0, 30.0],
        [0.0, 0.0, 40.0], [0.0, 0.0, 40.0],
        [0.0, 0.0, 50.0], [0.0, 0.0, 50.0],
    ])
    cache = compute_strip_dt_target_cache(
        _IdentityTransform(), torch.tensor(10.0), zyxs,
        torch.tensor([0, 2, 4, 6]), max_total_points=3)
    assert cache['frame'] == 'strip_endpoints'
    assert cache['anchor_theta'].device == zyxs.device
    torch.testing.assert_close(
        cache['target_relative'].to(torch.int64), torch.tensor([3, 4, 5]))


def test_track_loss_consumes_shared_endpoint_cache():
    tracks = [
        np.asarray([[z, 0, 30] for z in range(4)], dtype=np.float32),
        np.asarray([[z, 0, 40] for z in range(4)], dtype=np.float32),
    ]
    prepared = prepare_main_phase_tracks(
        tracks, None, 0.0, 'cpu',
        sampling_config={'track_crossing_mode': 'count'})
    transform = _IdentityTransform()
    dr = torch.tensor(10.0)
    cache = compute_strip_dt_target_cache(
        transform, dr, prepared['flat_zyx_cpu'], prepared['offsets'])
    cfg = {
        'sample_count_tracks_per_step': 2,
        'sample_count_track_points_per_step': 4,
        'track_min_sample_spacing': 20.0,
        'track_max_sample_spacing': 60.0,
        'track_max_track_crossing_per_step': 0,
        'track_radius_loss_margin': 0.025,
        'track_radius_target': 'mean',
        'track_radius_within_norm_p': 3.0,
        'track_dt_loss_margin': 0.025,
        'track_dt_within_track_norm_p': 3.0,
        'track_dt_norm_p': 0.5,
    }
    torch.manual_seed(3)
    values = list(iter_track_losses(
        transform, dr, prepared, cfg, compute_dt=True,
        dt_target_cache=cache))
    assert [name for name, _value in values] == ['track_radius', 'track_dt']
    assert all(torch.isfinite(value) for _name, value in values)
