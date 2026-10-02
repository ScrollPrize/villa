"""3D rotary geometry, axial equivalence, mixed precision."""
import math

import torch

from vesuvius.models.build.pretrained_backbones.rope import apply_rotary_embedding
from vesuvius.neural_tracing.fiber_follow.regression.rope import AxialRoPE3D


def test_axial_rope_matches_full_3d_grid_scores_and_gradients():
    torch.manual_seed(33)
    shape, head_dim = (3, 4, 5), 64  # Production head width: 60 rotary channels plus a remainder.
    rope = AxialRoPE3D(head_dim).double()
    for axis in range(3):
        length = shape[axis]
        lines = math.prod(shape)//length
        q = torch.randn(lines, 2, length, head_dim, dtype=torch.float64, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        qr, kr = rope(q, k, shape, axis)
        tables = tuple(table.reshape(*shape, rope.rotary_dim).movedim(axis, -2)
                       .reshape(lines, 1, length, rope.rotary_dim)
                       for table in rope.embedding.get_embed(shape))
        def full(value):
            return torch.cat((apply_rotary_embedding(value[..., :rope.rotary_dim], tables),
                              value[..., rope.rotary_dim:]), -1)
        actual = qr @ kr.transpose(-1, -2)
        expected = full(q) @ full(k).transpose(-1, -2)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        probe = torch.randn_like(actual)
        a = torch.autograd.grad((actual*probe).sum(), (q, k), retain_graph=True)
        b = torch.autograd.grad((expected*probe).sum(), (q, k))
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(qr[..., rope.rotary_dim:], q[..., rope.rotary_dim:], atol=0, rtol=0)
        torch.testing.assert_close(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-12, rtol=1e-12)


def test_each_axis_has_relative_position_signal_without_training_randomness():
    rope = AxialRoPE3D(32)
    q = torch.ones(1, 4, 5, 32)
    for axis in range(3):
        rope.train()
        state = torch.get_rng_state().clone()
        a, b = rope(q, q, (5, 5, 5), axis)
        assert torch.equal(torch.get_rng_state(), state)
        scores = a @ b.transpose(-1, -2)
        assert scores[0, 0, 0, 0] > scores[0, 0, 0, 1]
        torch.testing.assert_close(scores[..., 0, 1], scores[..., 1, 2])
        rope.eval()
        c, d = rope(q, q, (5, 5, 5), axis)
        torch.testing.assert_close(a, c, atol=0, rtol=0)
        torch.testing.assert_close(b, d, atol=0, rtol=0)


def test_rotary_bfloat16_outputs_and_gradients_are_finite():
    rope = AxialRoPE3D(32)
    q = torch.randn(2, 4, 5, 32, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    a, b = rope(q, k, (5, 5, 5), 2)
    assert a.dtype == b.dtype == torch.bfloat16
    assert rope.embedding.periods.dtype == torch.float32
    (a.float().square().mean()+b.float().square().mean()).backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
