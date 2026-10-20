import unittest

import torch
from torch.utils.checkpoint import checkpoint

from dinovol_2.model.patch_encode_decode import LayerNormNd, PatchEmbedDeeper
from dinovol_2.ops import channel_norm


class ChannelNormTests(unittest.TestCase):
    def test_fallback_and_state_dict(self):
        for shape in ((2, 7, 9), (2, 7, 5, 3), (2, 7, 3, 5, 9)):
            x = torch.randn(shape, requires_grad=True)
            reference = LayerNormNd(7, backend="torch")
            actual = LayerNormNd(7, backend="auto")
            actual.load_state_dict(reference.state_dict(), strict=True)
            residual = torch.randn_like(x, requires_grad=True)
            for relu in (False, True):
                expected = reference(x) + residual
                if relu:
                    expected = expected.relu()
                output = actual(x, residual, relu)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                weight = torch.randn_like(output)
                grads = torch.autograd.grad(output, (x, residual, *actual.parameters()), weight)
                expected_grads = torch.autograd.grad(expected, (x, residual, *reference.parameters()), weight)
                for got, want in zip(grads, expected_grads):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            LayerNormNd(7, backend="typo")

    @unittest.skipUnless(torch.cuda.is_available() and channel_norm.triton is not None, "CUDA/Triton required")
    def test_cuda_outputs_and_all_gradients(self):
        torch.manual_seed(83)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for c in (7, 32, 64, 128, 256):
                for layout in ("channels_last", "strided"):
                    with self.subTest(dtype=dtype, channels=c, layout=layout):
                        x = torch.randn(2, c, 3, 5, 18, device="cuda", dtype=dtype)
                        x = (x.contiguous(memory_format=torch.channels_last_3d)
                             if layout == "channels_last" else x[..., ::2])
                        x.requires_grad_()
                        residual = torch.randn_like(x, requires_grad=True)
                        w = torch.randn(c, device="cuda", requires_grad=True)
                        b = torch.randn(c, device="cuda", requires_grad=True)
                        for add, relu in ((False, False), (False, True), (True, True), (True, False)):
                            r = residual if add else None
                            got = channel_norm.channel_norm(x, w, b, 1e-5, r, relu)
                            want = channel_norm.reference(x, w, b, 1e-5, r, relu)
                            # Random, noncontiguous upstream gradients exercise stride handling.
                            dy = torch.randn(*got.shape[:-1], got.shape[-1] * 2,
                                             device="cuda", dtype=dtype)[..., ::2]
                            inputs = (x, w, b, residual) if add else (x, w, b)
                            got_grads = torch.autograd.grad(got, inputs, dy)
                            want_grads = torch.autograd.grad(want, inputs, dy)
                            tol = 2e-5 if dtype == torch.float32 else (0.004 if dtype == torch.float16 else 0.03)
                            torch.testing.assert_close(got, want, rtol=tol, atol=tol)
                            for a, e in zip(got_grads, want_grads):
                                torch.testing.assert_close(a, e, rtol=tol, atol=tol)
                            self.assertTrue(got.is_contiguous(memory_format=torch.channels_last_3d))

    @unittest.skipUnless(torch.cuda.is_available() and channel_norm.triton is not None, "CUDA/Triton required")
    def test_constant_input_and_relu_rounding(self):
        # The ReLU mask must be based on the rounded norm output AND rounded residual sum.
        x = torch.full((1, 32, 2, 3, 5), 4., device="cuda", dtype=torch.float16, requires_grad=True)
        w = torch.ones(32, device="cuda", requires_grad=True)
        b = torch.linspace(-1e-7, 1e-7, 32, device="cuda", requires_grad=True)
        r = torch.zeros_like(x, requires_grad=True)
        a = channel_norm.channel_norm(x, w, b, 1e-5, r, True)
        e = channel_norm.reference(x, w, b, 1e-5, r, True)
        torch.testing.assert_close(a, e, rtol=0, atol=0)
        for ag, eg in zip(torch.autograd.grad(a.sum(), (x, w, b, r)),
                          torch.autograd.grad(e.sum(), (x, w, b, r))):
            torch.testing.assert_close(ag, eg, rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available() and channel_norm.triton is not None, "CUDA/Triton required")
    def test_persistent_reduction_tail(self):
        # More than 2048 * 128 rows exercises multiple iterations per partition,
        # including a partial last tile, rather than only single-tile reductions.
        torch.manual_seed(97)
        x = torch.randn(1, 32, 65, 65, 65, device="cuda", dtype=torch.float16)
        x = x.contiguous(memory_format=torch.channels_last_3d).requires_grad_()
        w = torch.randn(32, device="cuda", requires_grad=True)
        b = torch.randn(32, device="cuda", requires_grad=True)
        r = torch.randn_like(x, requires_grad=True)
        got = channel_norm.channel_norm(x, w, b, 1e-5, r, True)
        want = channel_norm.reference(x, w, b, 1e-5, r, True)
        dy = torch.randn_like(x)
        torch.testing.assert_close(got, want, rtol=.003, atol=.003)
        for a, e in zip(torch.autograd.grad(got, (x, w, b, r), dy),
                        torch.autograd.grad(want, (x, w, b, r), dy)):
            relative_l2 = (a.float()-e.float()).norm() / e.float().norm().clamp_min(1e-6)
            self.assertLess(relative_l2.item(), .003)

    @unittest.skipUnless(torch.cuda.is_available() and channel_norm.triton is not None, "CUDA/Triton required")
    def test_custom_op_registration(self):
        x = torch.randn(2, 32, 3, 5, 7, device="cuda", requires_grad=True)
        w = torch.randn(32, device="cuda", requires_grad=True)
        b = torch.randn(32, device="cuda", requires_grad=True)
        torch.library.opcheck(channel_norm.channel_norm, (x, w, b, 1e-5, None, True))

    @unittest.skipUnless(torch.cuda.device_count() > 1 and channel_norm.triton is not None, "Two CUDA devices required")
    def test_noncurrent_cuda_device(self):
        with torch.cuda.device(0):
            x = torch.randn(2, 32, 3, 5, 7, device="cuda:1", requires_grad=True)
            w = torch.randn(32, device=x.device, requires_grad=True)
            b = torch.randn(32, device=x.device, requires_grad=True)
            got = channel_norm.channel_norm(x, w, b, 1e-5, None, True)
            want = channel_norm.reference(x, w, b, 1e-5, None, True)
            self.assertEqual(torch.cuda.current_device(), 0)
            torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)
            dy = torch.randn_like(x)
            for a, e in zip(torch.autograd.grad(got, (x, w, b), dy),
                            torch.autograd.grad(want, (x, w, b), dy)):
                torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-5)

    @unittest.skipUnless(torch.cuda.is_available() and channel_norm.triton is not None, "CUDA/Triton required")
    def test_compiled_checkpointed_cnn_matches_reference(self):
        torch.manual_seed(51)
        kwargs = dict(patch_size=(2, 2, 2), input_channels=1, embed_dim=32,
                      stage_channels=(32, 64), stage_strides=(1, 2), depth_per_level=(1, 1),
                      stem_channels=32, stem_type="conv", channels_last=True)
        reference = PatchEmbedDeeper(**kwargs, norm_backend="torch").cuda().to(memory_format=torch.channels_last_3d)
        model = PatchEmbedDeeper(**kwargs).cuda().to(memory_format=torch.channels_last_3d)
        model.load_state_dict(reference.state_dict(), strict=True)
        from dinovol_2.model.dinov2_eva import _save_convolutions_context
        def run(x):
            return checkpoint(model, x, use_reentrant=False, context_fn=_save_convolutions_context)
        compiled = torch.compile(run, fullgraph=True)
        x = torch.randn(2, 1, 8, 8, 8, device="cuda", requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.float16):
            got = compiled(x)
            want = reference(xr)
        dy = torch.randn_like(got)
        got.backward(dy)
        want.backward(dy)
        torch.testing.assert_close(got, want, rtol=.02, atol=.01)
        for name, a, e in [("input", x.grad, xr.grad), *[
            (n, p.grad, dict(reference.named_parameters())[n].grad) for n, p in model.named_parameters()
        ]]:
            self.assertTrue(torch.isfinite(a).all(), name)
            # Compare full gradients; max-element relative error is ill-defined near zero.
            rel = (a.float() - e.float()).norm() / e.float().norm().clamp_min(1e-6)
            self.assertLess(rel.item(), .02, name)


if __name__ == "__main__":
    unittest.main()
