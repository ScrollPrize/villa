"""Batched decoder attention: key masking, gradients and compiled graph reuse."""
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from vesuvius.neural_tracing.fiber_follow.regression.model import PathDecoderLayer


@pytest.mark.parametrize('device,compiled', [('cpu', False), ('cuda', True)])
def test_batched_attention_masks_keys_and_preserves_gradients(device, compiled):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from torch.nn.attention import _cur_sdpa_kernel_backends
    from torch._dynamo.utils import counters
    torch.manual_seed(217)
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    layer = PathDecoderLayer(128, 4, dropout=0., batch_first=True)
    inputs = [torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
              for shape in ((2, 4, 16, 32), (2, 4, 1024, 32), (2, 4, 1024, 32))]
    probe = torch.randn(2, 4, 16, 32, device=device)
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=False):
        attend = torch.compile(layer.attend_memory, fullgraph=True, dynamic=False) if compiled else layer.attend_memory
        graphs = None
        for stride in (3, 5):
            padding = torch.zeros(2, 1024, device=device, dtype=torch.bool)
            padding[0, ::stride] = True
            padding[1, 100:100+stride*37] = True
            q, k, v = inputs
            with sdpa_kernel(SDPBackend.MATH):
                preferences = _cur_sdpa_kernel_backends(with_priority=True)
                actual = attend(q, (k, v), padding)
                assert _cur_sdpa_kernel_backends(with_priority=True) == preferences
                reference = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                    attn_mask=(~padding)[:, None, None])
            actual_grads = torch.autograd.grad((actual.float()*probe).sum(), inputs)
            reference_grads = torch.autograd.grad((reference*probe).sum(), inputs)
            rtol, atol = (.03, .002) if device == 'cuda' else (3e-4, 2e-6)
            torch.testing.assert_close(actual.float(), reference, rtol=rtol, atol=atol)
            for a, b in zip(actual_grads, reference_grads):
                torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
                assert (a.float()-b.float()).norm() <= .015*b.float().norm()
            excluded = padding[:, None, :, None].expand_as(k)
            assert actual_grads[1][excluded].eq(0).all()
            assert actual_grads[2][excluded].eq(0).all()
            if compiled:
                current = counters['stats']['unique_graphs']
                if graphs is not None:
                    assert current == graphs  # Different valid key counts reuse the graph.
                graphs = current
