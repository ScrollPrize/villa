from __future__ import annotations

import unittest

import torch

from dinovol_2.model.rope import apply_rotary_embedding, rope_apply


def concatenated_rotary_embedding(x, rope, prefix_tokens):
    """The previous formulation: rotate the suffix, then concatenate the prefix."""
    sin, cos = rope
    suffix = rope_apply(x[..., prefix_tokens:, :].to(sin.dtype), sin, cos).to(x.dtype)
    return torch.cat((x[..., :prefix_tokens, :], suffix), dim=-2)


class RotaryEmbeddingTests(unittest.TestCase):
    def test_matches_concatenated_formulation_exactly(self):
        torch.manual_seed(5)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for prefix_tokens in (0, 5):
                with self.subTest(dtype=dtype, prefix_tokens=prefix_tokens):
                    x = torch.randn(2, 3, 11 + prefix_tokens, 8, dtype=dtype, requires_grad=True)
                    angles = torch.randn(3, 11, 4).tile(2)
                    sin, cos = angles.sin().requires_grad_(), angles.cos().requires_grad_()
                    expected = concatenated_rotary_embedding(x, (sin, cos), prefix_tokens)
                    actual = apply_rotary_embedding(x, (sin, cos), prefix_tokens=prefix_tokens)
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    grad = torch.randn_like(expected)
                    for got, want in zip(torch.autograd.grad(actual, (x, sin, cos), grad),
                                         torch.autograd.grad(expected, (x, sin, cos), grad)):
                        torch.testing.assert_close(got, want, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
