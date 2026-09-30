"""Layout/backend changes must preserve the convolution and masked attention."""
import pytest
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from vesuvius.neural_tracing.fiber_follow.regression.model import DepthwiseConv3d, PathDecoderLayer


@pytest.mark.parametrize('channels_last', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_depthwise_layout_preserves_values_and_all_gradients(channels_last, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(72)
    dtype = torch.bfloat16 if device == 'cuda' else torch.float64
    original = nn.Conv3d(8, 8, 3, padding=1, groups=8).to(device=device, dtype=dtype)
    optimized = DepthwiseConv3d(8, 8, 3, padding=1, groups=8).to(device=device, dtype=dtype)
    optimized.load_state_dict(original.state_dict())
    value = torch.randn(2, 8, 5, 7, 9, device=device, dtype=dtype)
    if channels_last:
        value = value.to(memory_format=torch.channels_last_3d)
        original.to(memory_format=torch.channels_last_3d)
        optimized.to(memory_format=torch.channels_last_3d)
    inputs = [value.clone().requires_grad_() for _ in range(2)]
    results = [model(x) for model, x in zip((original, optimized), inputs)]
    for result in results:
        result.square().mean().backward()
    pairs = [(results[0], results[1]), (inputs[0].grad, inputs[1].grad)]
    pairs += [(a.grad, b.grad) for a, b in zip(original.parameters(), optimized.parameters())]
    for a, b in pairs:
        if device == 'cuda':
            # Different cuDNN reductions round differently in BF16, particularly
            # near zero. Bound both elementwise error and overall relative error.
            scale = a.detach().float().abs().max().item()
            torch.testing.assert_close(a, b, rtol=.02, atol=.002*scale)
            assert (a.float()-b.float()).norm() <= .01*a.float().norm()
        else:
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('mask_kind', ['none', 'all_references', 'mixed'])
def test_path_decoder_preserves_mask_outputs_and_gradients(device, mask_kind):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(31)
    dtype = torch.bfloat16 if device == 'cuda' else torch.float64
    original = nn.TransformerDecoderLayer(32, 4, 64, dropout=0., batch_first=True,
                                         norm_first=True).to(device=device, dtype=dtype)
    optimized = PathDecoderLayer(32, 4, 64, dropout=0., batch_first=True,
                                 norm_first=True).to(device=device, dtype=dtype)
    optimized.load_state_dict(original.state_dict())
    query = torch.randn(2, 16, 32, device=device, dtype=dtype)
    memory = torch.randn(2, 192, 32, device=device, dtype=dtype)
    mask = torch.zeros(2, 192, device=device, dtype=torch.bool)
    if mask_kind == 'all_references':
        mask[:, 128:] = True
    elif mask_kind == 'mixed':
        mask[0, 130::2] = True
        mask[1, 128:155] = True
    outputs, grads = [], []
    for model in (original, optimized):
        q, m = query.clone().requires_grad_(), memory.clone().requires_grad_()
        # Compare against the original backend, regardless of global preferences.
        backend = SDPBackend.EFFICIENT_ATTENTION if device == 'cuda' else SDPBackend.MATH
        with sdpa_kernel(backend):
            result = model(q, m, memory_key_padding_mask=mask)
        result.float().square().mean().backward()
        outputs.append(result)
        grads.append((q.grad, m.grad, *(p.grad for p in model.parameters())))
        assert m.grad[mask].eq(0).all()
    atol, rtol = (2e-2, 2e-2) if device == 'cuda' else (1e-10, 1e-9)
    torch.testing.assert_close(outputs[0], outputs[1], atol=atol, rtol=rtol)
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, atol=atol / 100, rtol=rtol)
    # Masked values must remain irrelevant even when large, finite values change.
    with torch.no_grad():
        changed = memory.clone()
        changed[mask] = 100.
        a = optimized(query, memory, memory_key_padding_mask=mask)
        b = optimized(query, changed, memory_key_padding_mask=mask)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_path_decoder_restores_attention_backend_preferences(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from torch.nn.attention import _cur_sdpa_kernel_backends
    layer = PathDecoderLayer(32, 4, dropout=0., batch_first=True).to(device)
    with sdpa_kernel(SDPBackend.MATH):
        before = _cur_sdpa_kernel_backends(with_priority=True)
        layer(torch.randn(1, 2, 32, device=device), torch.randn(1, 5, 32, device=device))
        assert _cur_sdpa_kernel_backends(with_priority=True) == before


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cached_attention_removes_only_masked_keys_and_preserves_gradients(device, compiled):
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
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
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
