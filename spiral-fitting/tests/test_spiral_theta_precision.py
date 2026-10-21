"""get_spiral_yxs against the same spiral built in float64."""
import numpy as np
import torch

import devices
from sample_spiral import get_spiral_yxs


def worst_node_offset():
    # 256 windings at 16 voxels reach a radius of 4096, a full scroll cross section.
    windings, dr, spacing = 256, 16.0, 20.0
    shipped = get_spiral_yxs(windings, dr, spacing, group_by_winding=True, device='cpu')
    worst = 0.0
    for k, got in enumerate(shipped):
        assert got.dtype == torch.get_default_dtype()
        step = spacing / (k + 0.5) / dr
        theta = k * 2 * np.pi + torch.arange(0, 2 * np.pi, step=step, dtype=torch.float64)
        want = torch.stack([theta.sin(), theta.cos()], dim=-1) * (dr * theta / (2 * np.pi))[:, None]
        worst = max(worst, float((got.double() - want).norm(dim=-1).max()))
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


def test_float32_cap_still_builds_the_spiral(monkeypatch):
    # The precision MPS has, forced on the CPU: the angles are float32 (worse than the 1e-3 bound above,
    # so the cap reached them) and the nodes still within a voxel.
    monkeypatch.setenv('FIT_SPIRAL_MAX_PRECISION_FLOAT', '32')
    devices.max_precision_bits.cache_clear()
    try:
        worst = worst_node_offset()
    finally:
        devices.max_precision_bits.cache_clear()
    assert 1e-3 < worst < 1.0, f'worst node {worst:.4f} voxels from the float64 spiral'
