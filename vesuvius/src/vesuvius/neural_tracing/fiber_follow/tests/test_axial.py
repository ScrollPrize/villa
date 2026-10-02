"""Receptive field of the patch encoder's axial attention block."""
import torch

from vesuvius.neural_tracing.fiber_follow.models.model import AxialBlock


def test_one_patch_axial_block_connects_distant_positions_in_both_directions():
    torch.manual_seed(47)
    block = AxialBlock(12, 2, ffn=16, local_convolution=False, rotary=True)
    x = torch.randn(1, 5, 6, 7, 12, requires_grad=True)
    out = block(x)
    g = torch.autograd.grad(out[0, 0, 0, 0, 0], x, retain_graph=True)[0]
    assert g[0, -1, -1, -1].abs().sum() > 0
    g = torch.autograd.grad(out[0, -1, -1, -1, 0], x)[0]
    assert g[0, 0, 0, 0].abs().sum() > 0
