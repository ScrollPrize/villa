import pytest
import torch

import dt_targets
from dt_targets import (
    _compact_integer_tensor,
    _compact_strip_cache_to_host,
    compute_strip_dt_target_cache,
    patch_dt_target_in_sample_frame,
    strip_dt_target_in_sample_frame,
)


class _IdentityTransform:
    def __call__(self, points):
        return points


def test_patch_theta_potential_cache_stages_only_selected_host_rows(
        monkeypatch):
    dr = torch.tensor(10.0)
    cache = {
        'frame': 'theta_potential',
        'target_relative': torch.arange(100, dtype=torch.int16),
        'valid': torch.ones(100, dtype=torch.bool),
    }
    staged_indices = []
    real_stage = dt_targets._stage_host_tensor

    def capture_stage(values, indices, device, **kwargs):
        staged_indices.append(torch.as_tensor(indices).clone())
        return real_stage(values, indices, device, **kwargs)

    monkeypatch.setattr(dt_targets, '_stage_host_tensor', capture_stage)
    target = patch_dt_target_in_sample_frame(
        torch.tensor([[71.0, 72.0], [21.0, 22.0]]),
        torch.zeros((2, 2, 2)),
        torch.zeros((2, 2)),
        torch.zeros((2, 2)),
        dr,
        cache,
        torch.tensor([7, 2]),
    )
    torch.testing.assert_close(target, torch.tensor([[70.0], [20.0]]))
    assert len(staged_indices) == 2
    for indices in staged_indices:
        torch.testing.assert_close(indices, torch.tensor([7, 2]))


def test_compact_host_strip_cache_stages_only_selected_strip():
    construction = {
        'keys': torch.tensor([0, 4, 102, 110]),
        'key_scale': 100,
        'theta': torch.zeros(4),
        'adjustment': torch.tensor([0.0, 1.0, 2.0, 5.0]),
        'target_relative': torch.tensor([3.0, 7.0]),
        'valid': torch.tensor([True, True]),
        'num_points': 4,
    }
    cache = _compact_strip_cache_to_host(construction)
    assert cache['storage'] == 'host_compact'
    assert cache['offsets'].device.type == 'cpu'
    assert cache['offsets'].dtype == torch.int32
    assert cache['local_idx'].dtype == torch.int16
    assert cache['adjustment'].dtype == torch.int16
    assert cache['target_relative'].dtype == torch.int16
    assert cache['theta'].dtype == torch.float32

    target = strip_dt_target_in_sample_frame(
        torch.tensor([[70.0]]),
        torch.tensor([[8]]),
        torch.zeros((1, 1)),
        torch.zeros((1, 1)),
        torch.tensor(10.0),
        cache,
        torch.tensor([1]),
    )
    # Nearest cached local point is 10: target 7 - cached adjustment 5.
    torch.testing.assert_close(target, torch.tensor([[20.0]]))


def test_compact_integer_storage_widens_instead_of_overflowing():
    assert _compact_integer_tensor(
        torch.tensor([0, 32767]), 'test').dtype == torch.int16
    assert _compact_integer_tensor(
        torch.tensor([0, 32768]), 'test').dtype == torch.int32
    assert _compact_integer_tensor(
        torch.tensor([0, 2**31]), 'test').dtype == torch.int64
    with pytest.raises(ValueError, match='not integer-valued'):
        _compact_integer_tensor(torch.tensor([1.25]), 'test')


def test_chunked_strip_cache_moves_each_chunk_to_host_before_merge():
    # Three simple theta=0 strips force the raw-point chunking path.
    zyxs = torch.tensor([
        [0.0, 0.0, 30.0], [0.0, 0.0, 30.0],
        [0.0, 0.0, 40.0], [0.0, 0.0, 40.0],
        [0.0, 0.0, 50.0], [0.0, 0.0, 50.0],
    ])
    cache = compute_strip_dt_target_cache(
        _IdentityTransform(), torch.tensor(10.0), zyxs,
        torch.tensor([0, 2, 4, 6]), max_total_points=3)
    assert cache['storage'] == 'host_compact'
    assert cache['theta'].device.type == 'cpu'
    torch.testing.assert_close(
        cache['offsets'].to(torch.int64), torch.tensor([0, 2, 4, 6]))
    torch.testing.assert_close(
        cache['target_relative'].to(torch.int64), torch.tensor([3, 4, 5]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_compact_host_cache_stages_selected_strip_to_cuda():
    try:
        torch.empty(1, device='cuda')
    except torch.AcceleratorError as exc:
        if 'out of memory' in str(exc).lower():
            pytest.skip('CUDA device is occupied')
        raise
    construction = {
        'keys': torch.tensor([0, 4, 102, 110], device='cuda'),
        'key_scale': 100,
        'theta': torch.zeros(4, device='cuda'),
        'adjustment': torch.tensor([0, 1, 2, 5], device='cuda'),
        'target_relative': torch.tensor([3, 7], device='cuda'),
        'valid': torch.tensor([True, True], device='cuda'),
        'num_points': 4,
    }
    cache = _compact_strip_cache_to_host(construction)
    target = strip_dt_target_in_sample_frame(
        torch.tensor([[70.0]], device='cuda'),
        torch.tensor([[8]], device='cuda'),
        torch.zeros((1, 1), device='cuda'),
        torch.zeros((1, 1), device='cuda'),
        torch.tensor(10.0, device='cuda'),
        cache,
        torch.tensor([1], device='cuda'),
    )
    torch.testing.assert_close(target.cpu(), torch.tensor([[20.0]]))
