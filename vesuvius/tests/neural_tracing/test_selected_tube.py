from copy import deepcopy

import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.coarse_supervision import CoarseTubeConfig, selected_tube_statistics
from vesuvius.neural_tracing.fiber_trace_3d.decoder_migration import migrate_output_stride
from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
from vesuvius.neural_tracing.fiber_trace_3d.training_state import model_optimizer_payload, restore_model_optimizer
from vesuvius.neural_tracing.fiber_trace_3d.ema import ModelEMA


@pytest.mark.parametrize('stride', [1, 2])
@pytest.mark.parametrize('phase', [0., .5])
def test_tube_gradient_and_detached_geometry(stride, phase):
    logits = torch.zeros((25, 25, 25), requires_grad=True)
    center = torch.tensor([[12. + phase] * 3], requires_grad=True) * stride
    jac = (torch.eye(3)[None] / 2).requires_grad_()
    normal = torch.tensor([[1., 0, 0]], requires_grad=True)
    width = torch.tensor([[0., 1, 0]], requires_grad=True)
    pos, neg = selected_tube_statistics(logits, center, jac, normal, width,
        torch.tensor([True]), stride=stride, config=CoarseTubeConfig(weight=1))
    assert (pos + neg).item() == pytest.approx(torch.log(torch.tensor(2.)).item())
    (pos + neg).backward()
    assert logits.grad[12, 12, 12] < 0
    assert logits.grad.max() > 0
    assert jac.grad is None and normal.grad is None and width.grad is None


@pytest.mark.parametrize('supported', [True, False])
def test_boundary_no_padding_supervision(supported):
    logits = torch.zeros((9, 9, 9), requires_grad=True)
    pos, neg = selected_tube_statistics(logits, torch.tensor([[0., 0., 4.]]),
        torch.eye(3)[None], torch.tensor([[1., 0, 0]]), torch.tensor([[0., 1, 0]]),
        torch.tensor([supported]), stride=1, config=CoarseTubeConfig(weight=1))
    assert (pos + neg).item() == pytest.approx(.69314718 if supported else 0)
    (pos + neg).backward()
    assert torch.isfinite(logits.grad).all()


def test_stride_migration_named_adam_ema_and_step():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = {'model_3d': {'unet_base_channels': 4, 'unet_depth': 3,
                              'normalization': 'group', 'decoder_upsample_mode': 'trilinear',
                              'output_stride': 1}}
        source = build_fiber_trace_3d_model(config)
        opt = torch.optim.AdamW(source.parameters())
        sum(p.square().sum() for p in source.parameters()).backward()
        opt.step()
        payload = deepcopy(model_optimizer_payload(source, opt, ema=ModelEMA(source, [.99])))
        payload.update(config=config, step=123, samples_seen=15744)
        target_config = deepcopy(config)
        target_config['model_3d']['output_stride'] = 2
        target = build_fiber_trace_3d_model(target_config)
        fresh = deepcopy(target.state_dict())
        target_opt = torch.optim.AdamW(target.parameters())
        moved = migrate_output_stride(deepcopy(payload), target, target_opt)
        report = moved['decoder_migration']
        assert report['reinitialized'] and report['removed']
        assert moved['step'] == 123 and moved['samples_seen'] == 15744
        restore_model_optimizer(moved, model=target, optimizer=target_opt)
        source_params = dict(source.named_parameters())
        for name, p in target.named_parameters():
            if name in report['retained']:
                torch.testing.assert_close(p, payload['model'][name])
                torch.testing.assert_close(target_opt.state[p]['exp_avg'], opt.state[source_params[name]]['exp_avg'])
            else:
                torch.testing.assert_close(p, fresh[name])
                assert p not in target_opt.state
            torch.testing.assert_close(moved['ema']['tracks']['ema_0p99']['model'][name], p)
        result = target(torch.randn(1, 1, 16, 16, 16))
        assert result.shape[-3:] == (8, 8, 8)
        result.square().mean().backward()
        target_opt.step()
        assert all(torch.isfinite(p).all() for p in target.parameters())
    finally:
        torch.set_num_threads(threads)
