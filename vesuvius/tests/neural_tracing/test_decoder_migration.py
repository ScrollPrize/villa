from copy import deepcopy

import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
from vesuvius.neural_tracing.fiber_trace_3d.decoder_migration import migrate_pixelshuffle
from vesuvius.neural_tracing.fiber_trace_3d.ema import ModelEMA
from vesuvius.neural_tracing.fiber_trace_3d.training_state import model_optimizer_payload, restore_model_optimizer
from vesuvius.neural_tracing.nets.vesuvius_unet3d import Trilinear3dUpsample


def config(mode):
    return {'patch_shape_zyx': [16]*3, 'model_3d': {'unet_base_channels': 4,
        'unet_depth': 3, 'normalization': 'group', 'decoder_upsample_mode': mode}}


def test_trilinear_resize_exact_shape_constant_and_backward():
    module = Trilinear3dUpsample(2, 3, [2, 2, 1], conv_op=torch.nn.Conv3d, bias=True)
    x = torch.ones(1, 2, 3, 4, 5, requires_grad=True)
    y = module(x, size=(7, 9, 5))
    assert y.shape == (1, 3, 7, 9, 5)
    torch.testing.assert_close(y, y[:, :, :1, :1, :1].expand_as(y))
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_migration_preserves_weights_moments_and_ema_then_adam_step():
    before_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        old = build_fiber_trace_3d_model(config('pixelshuffle'))
        optimizer = torch.optim.AdamW(old.parameters())
        sum(p.square().sum() for p in old.parameters()).backward()
        optimizer.step()
        ema = ModelEMA(old, [.99])
        ema.tracks['ema_0p99']['best_metric'] = .1
        payload = deepcopy(model_optimizer_payload(old, optimizer, ema=ema))
        payload['config'] = config('pixelshuffle')
        target = build_fiber_trace_3d_model(config('trilinear'))
        fresh = deepcopy(target.state_dict())
        opt = torch.optim.AdamW(target.parameters())
        migrated = migrate_pixelshuffle(deepcopy(payload), target, opt)
        changed = migrated['decoder_migration']['reinitialized']
        assert changed and all('transpconvs' in k for k in changed)
        for key, value in migrated['model'].items():
            torch.testing.assert_close(value, fresh[key] if key in changed else payload['model'][key])
            shadow = migrated['ema']['tracks']['ema_0p99']['model'][key]
            torch.testing.assert_close(shadow, value)
        assert migrated['ema']['tracks']['ema_0p99']['best_metric'] == float('inf')
        restore_model_optimizer(migrated, model=target, optimizer=opt)
        for (name, parameter), old_parameter in zip(target.named_parameters(), old.parameters(), strict=True):
            if name in changed:
                assert parameter not in opt.state
            else:
                torch.testing.assert_close(opt.state[parameter]['exp_avg'], optimizer.state[old_parameter]['exp_avg'])
        out = target(torch.randn(1, 1, 16, 16, 16))
        assert out.shape[-3:] == (16, 16, 16)
        out.square().mean().backward()
        opt.step()
        assert all(torch.isfinite(p).all() for p in target.parameters())
        current = model_optimizer_payload(target, opt)
        assert current['decoder_migration'] == migrated['decoder_migration']
        current['config'] = config('trilinear')
        assert migrate_pixelshuffle(current, target, opt) is current
        bad = deepcopy(payload)
        bad['model']['net.decoder.final_seg_layer.weight'] = torch.zeros(1)
        with pytest.raises(ValueError, match='non-upsampler mismatch'):
            migrate_pixelshuffle(bad, target, opt)
        bad = deepcopy(payload)
        bad['config']['model_3d']['decoder_upsample_mode'] = 'transpose'
        with pytest.raises(ValueError, match='pixelshuffle source'):
            migrate_pixelshuffle(bad, target, opt)
        reversed_opt = torch.optim.AdamW(list(target.parameters())[::-1])
        with pytest.raises(ValueError, match='parameter order'):
            migrate_pixelshuffle(deepcopy(payload), target, reversed_opt)
    finally:
        torch.set_num_threads(before_threads)
