"""get_spiral_yxs against the same spiral built in float64."""
import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import devices
from sample_spiral import get_spiral_yxs


def worst_node_offset(device='cpu', dr_per_winding=16.0):
    # 256 windings at 16 voxels reach a radius of 4096, a full scroll cross section.
    windings, dr, spacing = 256, 16.0, 20.0
    shipped = get_spiral_yxs(windings, dr_per_winding, spacing, group_by_winding=True, device=device)
    worst = 0.0
    for k, got in enumerate(shipped):
        assert got.dtype == torch.get_default_dtype()
        assert got.device.type == torch.device(device).type
        step = spacing / (k + 0.5) / dr
        theta = k * 2 * np.pi + torch.arange(0, 2 * np.pi, step=step, dtype=torch.float64)
        want = torch.stack([theta.sin(), theta.cos()], dim=-1) * (dr * theta / (2 * np.pi))[:, None]
        assert got.shape == want.shape
        worst = max(worst, float((got.cpu().double() - want).norm(dim=-1).max()))
    return worst


def test_outer_winding_nodes_match_a_float64_spiral(monkeypatch):
    monkeypatch.delenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', raising=False)
    devices.max_precision_bits.cache_clear()
    try:
        worst = worst_node_offset()
    finally:
        devices.max_precision_bits.cache_clear()
    # With the angles in float32 the worst node is 0.4852 voxels off, at winding 250.
    assert worst < 1e-3, f'worst node {worst:.4f} voxels from the float64 spiral'


def test_float32_cap_builds_the_angles_on_the_cpu(monkeypatch):
    # The precision MPS has, forced on the CPU, takes the path MPS takes: angles in float64 on the CPU. So the
    # nodes keep the 1e-3 bound (float32 angles miss it at 0.4852 voxels) and match the uncapped spiral exactly.
    # dr_per_winding is a tensor, as the model's is.
    monkeypatch.delenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', raising=False)
    devices.max_precision_bits.cache_clear()
    uncapped = get_spiral_yxs(256, torch.tensor(16.0), 20.0, device='cpu')
    monkeypatch.setenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', '32')
    devices.max_precision_bits.cache_clear()
    try:
        assert devices.float_hi('cpu') == torch.float32
        worst = worst_node_offset(dr_per_winding=torch.tensor(16.0))
        capped = get_spiral_yxs(256, torch.tensor(16.0), 20.0, device='cpu')
    finally:
        devices.max_precision_bits.cache_clear()
    assert worst < 1e-3, f'worst node {worst:.4f} voxels from the float64 spiral'
    assert torch.equal(capped, uncapped)


class _Float64OnDevice(TorchDispatchMode):
    # Names every op that leaves a float64 tensor on the device under test.
    def __init__(self, device):
        super().__init__()
        self.device, self.seen = torch.device(device), []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if torch.is_tensor(t) and t.device == self.device and t.dtype == torch.float64:
                self.seen.append(str(func))
        return out


def test_float32_cap_puts_no_float64_on_the_device(monkeypatch):
    # MPS has no float64. The meta device stands in for it, so CI without MPS still fails if the
    # float64 angles are ever built on the device instead of the CPU.
    monkeypatch.setenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', '32')
    devices.max_precision_bits.cache_clear()
    try:
        with _Float64OnDevice('meta') as mode:
            yxs = get_spiral_yxs(32, 16.0, 20.0, group_by_winding=True, device='meta')
    finally:
        devices.max_precision_bits.cache_clear()
    assert not mode.seen, f'float64 tensors on the device from {sorted(set(mode.seen))}'
    assert all(y.device.type == 'meta' and y.dtype == torch.get_default_dtype() for y in yxs)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs MPS')
def test_mps_nodes_match_a_float64_spiral(monkeypatch):
    # dr_per_winding on the device, as save_mesh and save_combined_preview pass it.
    monkeypatch.delenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', raising=False)
    devices.max_precision_bits.cache_clear()
    try:
        dr = torch.tensor(16.0, device='mps')
        worst = worst_node_offset('mps', dr)
        on_mps = get_spiral_yxs(256, dr, 20.0, device='mps')
        on_cpu = get_spiral_yxs(256, dr.cpu(), 20.0, device='cpu')
    finally:
        devices.max_precision_bits.cache_clear()
    assert worst < 1e-3, f'worst node {worst:.4f} voxels from the float64 spiral'
    # The angles are built on the CPU, so the MPS spiral is the CPU one, copied.
    assert on_mps.device.type == 'mps'
    assert torch.equal(on_mps.cpu(), on_cpu)
