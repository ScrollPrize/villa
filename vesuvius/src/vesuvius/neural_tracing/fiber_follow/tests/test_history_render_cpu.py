"""The fused CPU path must preserve the original tensor renderer bit for bit."""
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.shared.geometry import render_history, crop_local_grid
from vesuvius.neural_tracing.fiber_follow.regression.history_slabs import SLAB


@pytest.mark.parametrize('points', [0, 1, 19])
def test_segment_renderer_exact(points):
    rng = torch.Generator().manual_seed(73)
    hist = torch.randn(3, points, 3, generator=rng)*4
    mask = torch.randint(0, 2, (3, points), generator=rng).float()
    mask[0] = 0
    if points > 1:
        hist[:, 3] = hist[:, 2]  # zero-length segment
        mask[1, :4] = 1
    grid = torch.from_numpy(crop_local_grid(SLAB)).float()
    # Gradients deliberately keep the existing Torch implementation as oracle.
    reference = render_history(hist.clone().requires_grad_(), mask, grid, .7, 'segments').detach()
    actual = render_history(hist, mask, grid, .7, 'segments')
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
