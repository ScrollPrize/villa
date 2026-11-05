"""Regression tests for exact border coordinates on a valid TIFXYZ patch."""
import torch
from tifxyz import Patch


def _patch():
    zyx = torch.tensor([
        [[0., 0., 0.], [0., 0., 1.]],
        [[0., 1., 0.], [0., 1., 1.]],
    ])
    return Patch(zyxs=zyx, scale=torch.ones(3), overlapping_ids=None, winding=None)


def test_ij_to_zyx_closed_border_and_outside():
    patch = _patch()
    coords = torch.tensor([
        [0., 0.], [0.5, 0.5], [1., 0.], [0., 1.], [1., 1.],
        [-0.1, 0.5], [1.1, 0.5], [0.5, 1.1],
        [float("nan"), 0.5], [float("inf"), 0.5],
    ])
    result, valid = patch.ij_to_zyx(coords)
    assert valid.tolist() == [True] * 5 + [False] * 5
    expected = torch.tensor([
        [0., 0., 0.], [0., 0.5, 0.5], [0., 1., 0.],
        [0., 0., 1.], [0., 1., 1.],
    ])
    torch.testing.assert_close(result[:5], expected)
    assert torch.all(result[5:] == -1)


def test_ij_to_zyx_rejects_invalid_quad_on_border():
    patch = _patch()
    patch.valid_quad_mask[0, 0] = False
    result, valid = patch.ij_to_zyx(torch.tensor([[1., 1.]]))
    assert not valid.item()
    assert torch.all(result == -1)
