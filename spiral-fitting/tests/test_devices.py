import pytest
import torch

import devices
from spiral_helpers import _segmented_median_per_strip


def test_device_order_is_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.delenv('FIT_SPIRAL_DEVICE', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert devices.fit_device() == torch.device('cuda')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    assert devices.fit_device() == torch.device('mps')
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    assert devices.fit_device() == torch.device('cpu')


def test_device_override(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'cpu')
    assert devices.fit_device() == torch.device('cpu')
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'tpu')
    with pytest.raises(ValueError, match='unsupported device'):
        devices.fit_device()
    assert devices.fit_device('cpu') == torch.device('cpu')
    with pytest.raises(RuntimeError, match='CUDA is not available'):
        devices.fit_device('cuda')


def test_float_hi_is_float64_except_on_mps(monkeypatch):
    monkeypatch.delenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', raising=False)
    assert devices.float_hi(torch.device('cuda')) == torch.float64
    assert devices.float_hi('cpu') == torch.float64
    assert devices.float_hi() == torch.float64
    assert devices.float_hi('mps') == torch.float32
    monkeypatch.setenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', '32')
    assert devices.float_hi('cuda') == torch.float32
    monkeypatch.setenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', '16')
    with pytest.raises(ValueError):
        devices.float_hi('cpu')


def _strip_context(device):
    generator = torch.Generator().manual_seed(0)
    lengths = torch.randint(1, 40, (3000,), generator=generator)
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    strip_id = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    # Offsets of whole windings plus sub-voxel jitter: values the float32
    # composite key (strip_id * range + value) could no longer tell apart.
    values = (torch.randint(0, 60, (int(starts[-1]),), generator=generator).float()
              + torch.rand(int(starts[-1]), generator=generator) * 1e-3)
    return {
        'normalised_radii': values.to(device), 'strip_id': strip_id.to(device),
        'starts': starts.to(device), 'lengths': lengths.to(device),
        'S': len(lengths), 'device': torch.device(device),
    }


def _reference_medians(ctx):
    medians = []
    for start, length in zip(ctx['starts'][:-1].tolist(), ctx['lengths'].tolist()):
        strip = ctx['normalised_radii'][start:start + length].cpu().sort().values
        medians.append(strip[(length - 1) // 2])
    return torch.stack(medians)


def test_segmented_median_without_float64_matches_per_strip_sort(monkeypatch):
    ctx = _strip_context('cpu')
    expected = _reference_medians(ctx)
    monkeypatch.delenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', raising=False)
    assert torch.equal(_segmented_median_per_strip(ctx), expected)
    monkeypatch.setenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', '32')
    assert torch.equal(_segmented_median_per_strip(ctx), expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs MPS')
def test_segmented_median_on_mps(monkeypatch):
    monkeypatch.delenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', raising=False)
    ctx = _strip_context('mps')
    assert torch.equal(_segmented_median_per_strip(ctx).cpu(), _reference_medians(ctx))
