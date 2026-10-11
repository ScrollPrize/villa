"""get_spiral_yxs against the same spiral built in float64."""
import numpy as np
import torch

from sample_spiral import get_spiral_yxs


def test_outer_winding_nodes_match_a_float64_spiral():
    # 256 windings at 16 voxels reach a radius of 4096, a full scroll cross section.
    windings, dr, spacing = 256, 16.0, 20.0
    shipped = get_spiral_yxs(windings, dr, spacing, group_by_winding=True, device='cpu')
    worst = 0.0
    for k, got in enumerate(shipped):
        step = spacing / (k + 0.5) / dr
        theta = k * 2 * np.pi + torch.arange(0, 2 * np.pi, step=step, dtype=torch.float64)
        want = torch.stack([theta.sin(), theta.cos()], dim=-1) * (dr * theta / (2 * np.pi))[:, None]
        worst = max(worst, float((got.double() - want).norm(dim=-1).max()))
    # With the angles in float32 the worst node is 0.4852 voxels off, at winding 250.
    assert worst < 1e-3, f'worst node {worst:.4f} voxels from the float64 spiral'
