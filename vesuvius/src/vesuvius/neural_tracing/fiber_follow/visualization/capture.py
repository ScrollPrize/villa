"""Temporary, output-preserving instrumentation of the current slab model."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F


def array(tensor):
    return tensor.detach().cpu().numpy().copy()


def attention(q, k, allowed):
    logits = q.float() @ k.float().transpose(-1, -2) / q.shape[-1]**.5
    # Long image banks (~20k keys) can accumulate >1e-4 normalization error
    # in CPU FP32 softmax. Stabilize only this diagnostic reduction, retaining
    # the FP32 Q/K logits and archive format; model SDPA is never replaced.
    return logits.masked_fill(~allowed, -torch.inf).softmax(-1, dtype=torch.float64).mean(1).float()


def capabilities(model):
    return dict(patch_features=hasattr(getattr(model, 'encoder', None), 'patch_projection'),
                history=hasattr(model, 'history_encoder'),
                attention=all(hasattr(model, name) for name in ('decoder', 'confidence_scorer', 'history_attention')),
                solver=hasattr(model, 'velocity_field'))


@contextmanager
def capture(model):
    """Record real features and Q/K attention; retain the original forward kernels."""
    arrays = {}
    if not all(capabilities(model)[key] for key in ('patch_features', 'history', 'attention')):
        yield arrays
        return
    counters = {}

    def append(key, value):
        index = counters.get(key, 0)
        counters[key] = index + 1
        arrays[f'{key}_{index}'] = array(value[0])

    with ExitStack() as stack:
        def hook(module, fn, pre=False):
            handle = (module.register_forward_pre_hook(fn) if pre
                      else module.register_forward_hook(fn))
            stack.callback(handle.remove)

        hook(model.encoder.patch_projection,
             lambda m, i, o: arrays.update(patch=array(o[0].permute(1, 2, 3, 0))))
        hook(model.encoder.blocks[0],
             lambda m, i: arrays.update(encoder_input=array(i[0][0])), pre=True)
        for j, block in enumerate(model.encoder.blocks):
            hook(block, lambda m, i, o, j=j: arrays.update({f'axial_{j}': array(o[0])}))
        hook(model.encoder.norm, lambda m, i, o: arrays.update(encoded=array(o[0])))
        hook(model.history_encoder, lambda m, i, o: arrays.update(
            history_tokens=array(o[0][0]), history_padding=array(o[1][0])))
        hook(model.query, lambda m, i, o: append('query', o))
        hook(model.decoder.norm, lambda m, i, o: append('decoded', o))

        families = [('generator', model.decoder.layers, model.history_attention),
                    ('scorer', model.confidence_scorer.layers, model.confidence_scorer.history_attention)]
        for family, layers, history in families:
            for j, layer in enumerate(layers):
                original = layer.attend_memory

                def attend(q, kv, padding, original=original, family=family, j=j):
                    result = original(q, kv, padding)
                    append(f'{family}_attention_{j}', attention(q, kv[0], ~padding[:, None, None]))
                    arrays['context_padding'] = array(padding[0])
                    return result

                stack.enter_context(patch.object(layer, 'attend_memory', attend))
            original = history.forward_cached

            def read(query, k, v, allowed, empty, original=original, history=history, family=family):
                result = original(query, k, v, allowed, empty)
                attn = history.attention
                h, heads = attn.embed_dim, attn.num_heads
                q = F.linear(history.norm(query), attn.in_proj_weight[:h], attn.in_proj_bias[:h])
                q = q.reshape(len(query), -1, heads, h // heads).transpose(1, 2)
                weights = attention(q, k, allowed)
                # An empty history contributes exactly zero, including output bias.
                weights = torch.where(empty[:, None, None], 0., weights)
                append(f'{family}_history_attention', weights)
                append(f'{family}_history_update', result - query)
                return result

            stack.enter_context(patch.object(history, 'forward_cached', read))
        yield arrays


def validate_attention(arrays):
    for key, weights in arrays.items():
        if '_attention_' not in key:
            continue
        history = '_history_' in key
        padding = arrays['history_padding' if history else 'context_padding'].astype(bool)
        expected = 0 if history and padding.all() else 1
        np.testing.assert_allclose(weights.sum(-1, dtype=np.float64), expected,
                                   rtol=0, atol=2e-6, err_msg=key)
        assert (weights[:, padding] == 0).all(), key
