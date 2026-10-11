from __future__ import annotations

import copy
import itertools
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from dinovol_2.model.dinov2_eva import EvaAttention, EvaAxialAttention, EvaBlock
from dinovol_2.model.model import DinoVitStudentTeacher, _materialize_backbone_config
from dinovol_2.model.rope import RopePositionEmbedding


class AxialAttentionTests(unittest.TestCase):
    def test_axis_matches_dense_connectivity_reference_and_gradients(self):
        # An independent dense mask specifies the graph, including prefix tokens.
        # Rectangular grids and multiple samples catch incorrect line/batch ordering.
        torch.manual_seed(41)
        for shape, prefix, fused in itertools.product(((2, 3), (2, 3, 4), (1, 3, 2)), (0, 3), (False, True)):
            ndim = len(shape)
            coords = torch.tensor(list(itertools.product(*(range(size) for size in shape))))
            count = math.prod(shape) + prefix
            rope = RopePositionEmbedding(12, ndim=ndim).eval().get_embed(shape)
            for axis in range(ndim):
                with self.subTest(shape=shape, prefix=prefix, fused=fused, axis=axis):
                    kwargs = dict(dim=24, num_heads=2, num_prefix_tokens=prefix, qkv_fused=fused)
                    axial = EvaAxialAttention(**kwargs, axis=axis, ndim=ndim).double()
                    dense = EvaAttention(**kwargs).double()
                    dense.load_state_dict(axial.state_dict(), strict=True)
                    allowed = torch.ones(count, count, dtype=torch.bool)
                    other_axes = [i for i in range(ndim) if i != axis]
                    allowed[prefix:, prefix:] = (
                        coords[:, None, other_axes] == coords[None, :, other_axes]
                    ).all(dim=-1)
                    x = torch.randn(2, count, 24, dtype=torch.float64, requires_grad=True)
                    ref_x = x.detach().clone().requires_grad_()
                    # Keep RoPE arithmetic in double for tight gradient comparisons.
                    double_rope = tuple(component.double() for component in rope)
                    actual = axial(x, rope=double_rope, rope_shape=shape)
                    expected = dense(ref_x, rope=double_rope, attn_mask=allowed)
                    torch.testing.assert_close(actual, expected)
                    weights = torch.randn_like(actual)
                    (actual * weights).sum().backward()
                    (expected * weights).sum().backward()
                    torch.testing.assert_close(x.grad, ref_x.grad)
                    for parameter, reference in zip(axial.parameters(), dense.parameters()):
                        torch.testing.assert_close(parameter.grad, reference.grad)

    def test_attention_calls_use_lines_and_global_prefix_queries(self):
        module = EvaAxialAttention(24, num_heads=2, num_prefix_tokens=3, axis=1, ndim=3)
        with patch('dinovol_2.model.dinov2_eva.F.scaled_dot_product_attention',
                   wraps=F.scaled_dot_product_attention) as attention:
            module(torch.randn(2, 27, 24), rope_shape=(2, 3, 4))
        shapes = [(call.args[0].shape, call.args[1].shape) for call in attention.call_args_list]
        self.assertEqual(shapes, [((16, 2, 3, 12), (16, 2, 6, 12)),
                                  ((2, 2, 3, 12), (2, 2, 27, 12))])

    def test_sequential_axes_connect_opposite_corners(self):
        torch.manual_seed(5)
        block = EvaBlock(24, 2, ndim=3, attention_type='axial', num_prefix_tokens=0)
        x = torch.randn(1, 24, 24, requires_grad=True)
        block(x, rope_shape=(2, 3, 4))[0, 0, 0].backward()
        self.assertGreater(x.grad[0, -1].abs().sum().item(), 0)

    def test_checkpointing_mixed_rope_and_variable_grids(self):
        for rope_type, chunks in itertools.product(('axial', 'mixed'), (0, 1)):
            with self.subTest(rope_type=rope_type, chunks=chunks):
                torch.manual_seed(17)
                config = dict(model_type='v3', attention_type='axial', embed_dim=36,
                              num_heads=3, depth=2, block_chunks=chunks, grad_checkpointing=True,
                              patch_size=[4]*3, global_crops_size=[8, 12, 16],
                              local_crops_size=[4, 8, 12], rope_type=rope_type, drop_path_rate=0.0)
                model = DinoVitStudentTeacher._build_backbone(config)
                reference = copy.deepcopy(model)
                reference.grad_checkpointing = False
                for shape, kind in (((8, 12, 16), 'global'), ((4, 8, 12), 'local')):
                    model.zero_grad(set_to_none=True)
                    reference.zero_grad(set_to_none=True)
                    x = torch.randn(2, 1, *shape)
                    masks = torch.rand(2, math.prod(size // 4 for size in shape)) > 0.5
                    rng = torch.get_rng_state()
                    actual = model(x, masks=masks, view_kind=kind)['x_prenorm']
                    torch.set_rng_state(rng)
                    expected = reference(x, masks=masks, view_kind=kind)['x_prenorm']
                    torch.testing.assert_close(actual, expected)
                    weights = torch.randn_like(actual)
                    (actual * weights).sum().backward()
                    (expected * weights).sum().backward()
                    for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
                        if ref_parameter.grad is None:
                            self.assertIsNone(parameter.grad)
                        else:
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                            torch.testing.assert_close(parameter.grad, ref_parameter.grad)

    def test_validation_and_default(self):
        self.assertEqual(_materialize_backbone_config({'model_type': 'v3'})['attention_type'], 'full')
        with self.assertRaisesRegex(ValueError, 'Unknown attention_type'):
            EvaBlock(24, 2, ndim=3, attention_type='typo')
        module = EvaAxialAttention(24, num_heads=2, axis=0, ndim=3)
        x = torch.randn(1, 9, 24)
        for shape in (None, (2, 2), (2, 2, 3)):
            with self.assertRaisesRegex(ValueError, 'matching the patch token grid'):
                module(x, rope_shape=shape)
        with self.assertRaisesRegex(ValueError, 'explicit attention mask'):
            module(x, rope_shape=(2, 2, 2), attn_mask=torch.ones(9, 9, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
