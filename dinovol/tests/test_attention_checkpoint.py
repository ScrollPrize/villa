import copy
import unittest

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from torch.utils._python_dispatch import TorchDispatchMode

from dinovol_2.model.dinov2_eva import _save_attention_context
from dinovol_2.model.model import DinoVitStudentTeacher


class _CountAttention(TorchDispatchMode):
    def __init__(self):
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if "scaled_dot_product" in str(func) and "backward" not in str(func):
            self.calls += 1
        return func(*args, **(kwargs or {}))


class AttentionCheckpointTests(unittest.TestCase):
    def test_fused_attention_is_not_recomputed(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                torch.manual_seed(43)
                dtype = torch.float32 if device == "cpu" else torch.float16
                x = torch.randn(2, 3, 31, 16, device=device, dtype=dtype, requires_grad=True)
                xr = x.detach().clone().requires_grad_()
                def block(x):
                    q, k, v = x.sin(), x.cos(), x * 2
                    return F.scaled_dot_product_attention(q, k, v).square()
                counter = _CountAttention()
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    with counter:
                        got = checkpoint(block, x, use_reentrant=False, context_fn=_save_attention_context)
                        got.sum().backward()
                    want = block(xr)
                    want.sum().backward()
                self.assertEqual(counter.calls, 1)
                torch.testing.assert_close(got, want)
                torch.testing.assert_close(x.grad, xr.grad)

    def test_chunked_and_unchunked_policy_matches_original(self):
        for chunks in (0, 2):
            torch.manual_seed(13)
            model = DinoVitStudentTeacher._build_backbone({
                "model_type": "v3", "embed_dim": 36, "num_heads": 3, "depth": 4,
                "global_crops_size": [8]*3, "local_crops_size": [8]*3, "patch_size": [4]*3,
                "grad_checkpointing": True, "block_chunks": chunks, "drop_path_rate": .2,
            })
            reference = copy.deepcopy(model)
            reference.checkpoint_save_attention = False
            x = torch.randn(2, 1, 8, 8, 8)
            state = torch.get_rng_state()
            got = model(x)["x_norm_patchtokens"]
            torch.set_rng_state(state)
            want = reference(x)["x_norm_patchtokens"]
            dy = torch.randn_like(got)
            got.backward(dy)
            want.backward(dy)
            torch.testing.assert_close(got, want, rtol=0, atol=0)
            for (name, p), pr in zip(model.named_parameters(), reference.parameters()):
                if pr.grad is None:
                    self.assertIsNone(p.grad, name)
                else:
                    torch.testing.assert_close(p.grad, pr.grad, msg=name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_attention_dropout_preserves_downstream_rng(self):
        torch.manual_seed(71)
        x = torch.randn(2, 3, 32, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        def block(x):
            out = F.scaled_dot_product_attention(x.sin(), x.cos(), x * 2, dropout_p=.2)
            return F.dropout(out, p=.3, training=True).square()
        state = torch.cuda.get_rng_state()
        counter = _CountAttention()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            with counter:
                got = checkpoint(block, x, use_reentrant=False, context_fn=_save_attention_context)
                got.sum().backward()
            torch.cuda.set_rng_state(state)
            want = block(xr)
            want.sum().backward()
        self.assertEqual(counter.calls, 2)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, xr.grad)


if __name__ == "__main__":
    unittest.main()
