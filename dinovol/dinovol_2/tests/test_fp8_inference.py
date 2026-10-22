import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from dinovol_2.eval.embed_volume import parse_args, validate_resume
from dinovol_2.ops.fp8_inference import (
    FP8Linear, enable_fp8_transformer, fp8_attention, require_supported_device, triton,
)


class PrecisionConfigurationTests(unittest.TestCase):
    def test_default_and_legacy_resume_stay_bf16(self):
        args = parse_args(['--checkpoint', 'c', '--volume', 'v', '--output', 'o'])
        self.assertEqual(args.transformer_precision, 'bf16')
        saved = {'outputs': {}, 'slabs': [], 'completed_slabs': {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / '.zgroup').write_text('{"zarr_format":2}')
            validate_resume(saved, dict(saved, transformer_precision='bf16'), path)
            with self.assertRaisesRegex(ValueError, 'transformer_precision'):
                validate_resume(saved, dict(saved, transformer_precision='fp8'), path)

    def test_unsupported_device_and_dimensions_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'NVIDIA'):
            require_supported_device('cpu')
        with self.assertRaisesRegex(ValueError, 'multiples of 16'):
            FP8Linear(torch.nn.Linear(15, 32))


@unittest.skipUnless(triton is not None and torch.cuda.is_available()
                     and torch.cuda.get_device_capability() >= (8, 9), 'FP8 NVIDIA GPU required')
class FP8KernelTests(unittest.TestCase):
    def test_conversion_is_explicit_and_rejects_unsupported_model_before_mutation(self):
        from dinovol_2.model.dinov2_eva import EvaAttention
        backbone = torch.nn.Module()
        block = torch.nn.Module()
        block.attn = EvaAttention(dim=64, num_heads=2, qkv_bias=True)
        backbone.blocks = torch.nn.ModuleList([block])
        backbone.cuda().eval().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, 'separate QKV biases'):
            enable_fp8_transformer(backbone)
        self.assertIsInstance(block.attn.proj, torch.nn.Linear)
        block.attn = EvaAttention(dim=64, num_heads=2, qkv_bias=False).cuda().eval().requires_grad_(False)
        self.assertEqual(enable_fp8_transformer(backbone), 2)
        self.assertIsInstance(block.attn.qkv, FP8Linear)
        self.assertEqual(block.attn.inference_precision, 'fp8')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = block.attn(torch.randn(1, 17, 64, device='cuda'))
        self.assertEqual(result.shape, (1, 17, 64))
        self.assertTrue(torch.isfinite(result).all())

    def test_linear_scales_zeros_and_compilation(self):
        torch.manual_seed(42)
        original = torch.nn.Linear(64, 96, device='cuda').eval().requires_grad_(False)
        converted = FP8Linear(original)
        compiled = torch.compile(converted, dynamic=False)
        inputs = torch.randn(2, 37, 64, device='cuda', dtype=torch.bfloat16)
        inputs[:, :1] = 0
        inputs[:, 1:2] *= 0.0001
        inputs[:, 2:3] *= 1000
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            expected, actual = original(inputs), compiled(inputs)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual[:, 0], expected[:, 0], rtol=0, atol=0)
        # Normalize error per vector so high-magnitude rows cannot hide errors in small rows.
        relative = (actual.float() - expected.float()).norm(dim=-1) / expected.float().norm(dim=-1).clamp_min(1e-6)
        self.assertLess(relative.max().item(), 0.075)
        self.assertFalse(list(converted.parameters()))
        converted.train()
        with self.assertRaisesRegex(RuntimeError, 'inference-only'):
            converted(inputs)

    def test_attention_strides_partial_tiles_and_zero_values(self):
        torch.manual_seed(123)
        for n, d in [(17, 16), (137, 48), (513, 64), (65, 128)]:
            with self.subTest(n=n, d=d), torch.inference_mode():
                # Strided Q/K/V like the fused projection in the real backbone.
                q, k, v = torch.randn(2, n, 3, 3, d, device='cuda', dtype=torch.bfloat16).permute(2, 0, 3, 1, 4)
                v[0, 0] = 0
                expected = F.scaled_dot_product_attention(q, k, v)
                actual = fp8_attention(q, k, v)
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(actual[0, 0].count_nonzero().item(), 0)
                error = (actual.float() - expected.float()).norm() / expected.float().norm()
                self.assertLess(error.item(), 0.075)
                cosine = F.cosine_similarity(actual[1].float().flatten(), expected[1].float().flatten(), dim=0)
                self.assertGreater(cosine.item(), 0.995)

    def test_attention_compiled_dispatch_and_uniform_scores(self):
        compiled = torch.compile(fp8_attention, dynamic=False)
        with torch.inference_mode():
            q = torch.zeros(1, 2, 131, 48, device='cuda', dtype=torch.bfloat16)
            v = torch.ones_like(q)
            actual = compiled(q, q, v)
            torch.testing.assert_close(actual, v, rtol=0.01, atol=0.01)
            with self.assertRaisesRegex(ValueError, 'matching nonempty'):
                fp8_attention(q, q[:, :, :-1], v)


if __name__ == '__main__':
    unittest.main()
