from __future__ import annotations

import copy
import itertools
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from dinovol_2.model.dinov2_eva import EvaAttention
from dinovol_2.model.model import DinoVitStudentTeacher, _materialize_backbone_config
from dinovol_2.model.rope import RopePositionEmbedding, apply_rotary_embedding


def reference_attention(module, x, rope, normalize):
    batch, count, _ = x.shape
    if module.qkv is not None:
        bias = torch.cat((module.q_bias, module.k_bias, module.v_bias))
        projected = F.linear(x, module.qkv.weight, bias).chunk(3, dim=-1)
    else:
        projected = (module.q_proj(x), module.k_proj(x), module.v_proj(x))
    q, k, v = [t.reshape(batch, count, module.num_heads, -1).transpose(1, 2) for t in projected]
    if normalize:
        # Independent formula, reducing only channels within each head/token.
        q = q / (q.square().mean(-1, keepdim=True) + 1e-6).sqrt() * module.q_norm.weight
        k = k / (k.square().mean(-1, keepdim=True) + 1e-6).sqrt() * module.k_norm.weight
    q = apply_rotary_embedding(q, rope, prefix_tokens=module.num_prefix_tokens)
    k = apply_rotary_embedding(k, rope, prefix_tokens=module.num_prefix_tokens)
    out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, count, -1)
    return module.proj(out)


class QKNormTests(unittest.TestCase):
    def test_attention_matches_reference_and_gradients(self):
        torch.manual_seed(41)
        for fused, enabled, head_dim in itertools.product((False, True), (False, True), (12, 18)):
            with self.subTest(fused=fused, enabled=enabled, head_dim=head_dim):
                module = EvaAttention(24, num_heads=2, attn_head_dim=head_dim,
                                      qkv_fused=fused, qk_norm=enabled,
                                      num_prefix_tokens=3, attention_backend="math").double()
                if enabled:
                    with torch.no_grad():
                        # Nonuniform scales make normalization after RoPE observably wrong.
                        module.q_norm.weight.copy_(torch.linspace(.5, 1.5, head_dim))
                        module.k_norm.weight.copy_(torch.linspace(1.8, .8, head_dim))
                reference = copy.deepcopy(module)
                rope = tuple(t.double() for t in RopePositionEmbedding(head_dim, ndim=3).get_embed((2, 2, 2)))
                x = torch.randn(2, 11, 24, dtype=torch.float64, requires_grad=True)
                ref_x = x.detach().clone().requires_grad_()
                actual = module(x, rope=rope)
                expected = reference_attention(reference, ref_x, rope, enabled)
                torch.testing.assert_close(actual, expected)
                dy = torch.randn_like(actual)
                actual.backward(dy)
                expected.backward(dy)
                torch.testing.assert_close(x.grad, ref_x.grad)
                for (name, param), ref_param in zip(module.named_parameters(), reference.parameters()):
                    torch.testing.assert_close(param.grad, ref_param.grad, msg=name)

    def test_disabled_preserves_checkpoint_layout_and_defaults(self):
        for model_type in (None, "v2", "v3"):
            self.assertFalse(_materialize_backbone_config({"model_type": model_type})["qk_norm"])
        legacy = EvaAttention(24, num_heads=2)
        self.assertIsInstance(legacy.q_norm, torch.nn.Identity)
        self.assertIsInstance(legacy.k_norm, torch.nn.Identity)
        self.assertFalse(any("norm" in key for key in legacy.state_dict()))
        disabled = EvaAttention(24, num_heads=2, qk_norm=False)
        disabled.load_state_dict(legacy.state_dict(), strict=True)
        x = torch.randn(2, 9, 24)
        torch.testing.assert_close(disabled(x), legacy(x), rtol=0, atol=0)

    def test_student_teacher_checkpointing_and_roundtrip(self):
        for attention_type, chunks in itertools.product(("full", "axial"), (0, 1)):
            with self.subTest(attention_type=attention_type, chunks=chunks):
                config = dict(model_type="v3", qk_norm=True, attention_type=attention_type,
                              attention_backend="math", embed_dim=24, num_heads=2, depth=2,
                              global_crops_size=[8]*3, local_crops_size=[4]*3, patch_size=[4]*3,
                              rope_type="mixed", grad_checkpointing=True, block_chunks=chunks,
                              drop_path_rate=0., init_values=None, dino_out_dim=8, ibot_out_dim=8,
                              dino_head_hidden_dim=24, dino_head_bottleneck_dim=12,
                              ibot_head_hidden_dim=24, ibot_head_bottleneck_dim=12)
                model = DinoVitStudentTeacher(config)
                for branch in (model.student, model.teacher):
                    attentions = [m for m in branch.backbone.modules() if isinstance(m, EvaAttention)]
                    self.assertEqual(len(attentions), 2 * (3 if attention_type == "axial" else 1))
                    for attn in attentions:
                        self.assertIsInstance(attn.q_norm, torch.nn.RMSNorm)
                        self.assertEqual(attn.q_norm.normalized_shape, (12,))
                        torch.testing.assert_close(attn.q_norm.weight, torch.ones(12))
                for size, kind in ((8, "global"), (4, "local")):
                    out = model.student.backbone(torch.randn(2, 1, size, size, size), view_kind=kind)["x_prenorm"]
                    (out * torch.randn_like(out)).sum().backward()
                for name, param in model.student.backbone.named_parameters():
                    if "q_norm" in name or "k_norm" in name:
                        self.assertTrue(torch.isfinite(param.grad).all(), name)
                        self.assertGreater(param.grad.abs().sum().item(), 0, name)
                self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
                restored = DinoVitStudentTeacher._build_backbone(_materialize_backbone_config(config))
                restored.load_state_dict(model.student.backbone.state_dict(), strict=True)
                restored.eval()
                model.student.backbone.eval()
                x = torch.randn(2, 1, 8, 8, 8)
                torch.testing.assert_close(restored(x)["x_prenorm"], model.student.backbone(x)["x_prenorm"])

    def test_autocast_keeps_attention_dtype_with_and_without_rope(self):
        for fused, use_rope in itertools.product((False, True), (False, True)):
            with self.subTest(fused=fused, use_rope=use_rope):
                module = EvaAttention(24, num_heads=2, qkv_fused=fused, qk_norm=True,
                                      attention_backend="math")
                rope = RopePositionEmbedding(12, ndim=3).get_embed((2, 2, 2)) if use_rope else None
                with patch("dinovol_2.model.dinov2_eva.F.scaled_dot_product_attention",
                           wraps=F.scaled_dot_product_attention) as attention:
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        out = module(torch.randn(2, 9, 24), rope=rope)
                for tensor in attention.call_args.args[:3]:
                    self.assertEqual(tensor.dtype, torch.bfloat16)
                out.float().square().mean().backward()
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(torch.isfinite(module.q_norm.weight.grad).all())
                self.assertTrue(torch.isfinite(module.k_norm.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
