"""Regression follower (models/crop_transformer.py): forward contract, losses, optimizer updates and checkpoints."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.models.model import build_model, config_from_checkpoint
from vesuvius.neural_tracing.fiber_follow.train.supervision import geometry_mask, loss_terms
from vesuvius.neural_tracing.fiber_follow.train import train
from vesuvius.neural_tracing.fiber_follow.train.train import clip_training_gradients, initialize_training_optimizer, load_checkpoint, optimizer_update, prepare_training, save_checkpoint
from vesuvius.neural_tracing.fiber_follow.data.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.train.runloop import read_checkpoint, training_rng_state
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec
from label_fixtures import set_terminal
from model_fixtures import coordinate_batch, coordinate_config, batch as label_batch, forward, proposal_output


def config(**kwargs):
    """Small regression model; one refinement unless refinement is the subject."""
    return coordinate_config(**{'recurrent_refinement_steps': 1, **kwargs})


def batch(cfg, b=2):
    return coordinate_batch(cfg, b)


def take(value, sl):
    return {k: take(v, sl) for k, v in value.items()} if isinstance(value, dict) else value[sl]


def test_regression_forward_is_deterministic_causal_and_trainable():
    torch.manual_seed(20)
    m = build_model(config())
    b = batch(m.cfg)
    b['hmask'][0] = 0
    b['x']['seed'][:] = torch.tensor([100., 100., 100.])  # outside the crop
    out = forward(m, b)
    torch.manual_seed(100)
    torch.testing.assert_close(out['points'], forward(m, b)['points'], rtol=0, atol=0)
    torch.testing.assert_close(out['points'][..., 2], torch.arange(1, 5).float().expand(2, -1))
    assert (out['confidence'][:, 1:] <= out['confidence'][:, :-1]).all()
    # Masked or invisible inputs and annotations cannot influence any output.
    hidden = copy.deepcopy(b)
    x = hidden['x']
    hidden['hist'][0] = float('nan')
    hidden['hist'][:, 10:] = float('nan')  # behind the crop
    hidden['gt_history'] = torch.full_like(hidden['hist'], float('nan'))
    for key in ('seed', 'seed_tangent', 'seed_age'):
        x[key] = torch.full_like(x[key], float('nan'))
    x['path_geometry'][~x['path_geometry_valid'].bool()] = float('nan')
    other = forward(m, hidden)
    for key in out:
        assert torch.isfinite(other[key]).all(), key
        torch.testing.assert_close(out[key], other[key], rtol=0, atol=0)
    loss_terms(out, b, m.cfg)['geometry_per_state'].mean().backward()
    for module in (m.coordinates, m.cnn.stages[0].blocks[0].conv1.conv, m.cell_token, m.reference_token[0]):
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.path_geometry.parameters())


def test_departures_unknown_ends_and_crop_censoring():
    cfg = config()
    b = label_batch(cfg, 3)
    set_terminal(b, 0)
    b['dense_mask'][1] = 0
    b['dense_ab'][1] = float('nan')
    b['dense_ab'][2, 4, 0] = 100
    mask = geometry_mask(b, cfg)
    assert not mask[:2].any() and mask[2, :4].all() and not mask[2, 4:].any()
    points = torch.zeros(3, 1, 4, 3)
    points[..., 2] = torch.arange(1, 5)
    terms = loss_terms(proposal_output(points, torch.zeros(3, 1, 4)), b, cfg)
    assert torch.equal(terms['geometry_per_state'][:2], torch.zeros(2))
    assert terms['confidence_per_state'][0] > 0
    assert terms['confidence_per_state'][1] == 0
    assert torch.isfinite(terms['error_sum'])


def test_microbatches_give_the_same_loss_and_update():
    """Splitting a batch into microbatches keeps the objective and the update; metrics don't change training or RNG."""
    torch.manual_seed(3)
    cfg = config()
    a = build_model(cfg)
    bmodel = copy.deepcopy(a)
    data = batch(cfg, 2)
    data['dense_mask'][0, 5:] = 0
    set_terminal(data, 1)
    averages = [copy.deepcopy(m) for m in (a, bmodel)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=.001) for m in (a, bmodel)]
    for model in (a, bmodel):
        prepare_training(model, batch_size=2, backend='eager')
    rng = torch.get_rng_state()
    results = [optimizer_update(a, averages[0], optimizers[0], [data], 1, .001, n_commit=2, compute_metrics=False)]
    results.append(optimizer_update(bmodel, averages[1], optimizers[1], [take(data, slice(0, 1)), take(data, slice(1, 2))],
                                    1, .001, n_commit=2, compute_metrics=True))
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert results[0]['loss'] == pytest.approx(results[1]['loss'], rel=2e-6)
    for key in ('positive_confidence_targets', 'negative_confidence_targets', 'confidence_terminal_states'):
        assert results[0][key] == results[1][key]
    for p, q in zip(a.parameters(), bmodel.parameters()):
        torch.testing.assert_close(p, q, rtol=2e-5, atol=2e-7)


def test_gradient_clipping_and_guards():
    def model(gradient):
        module = torch.nn.Module()
        module.a, module.b = torch.nn.Linear(2, 1, bias=False), torch.nn.Linear(2, 1, bias=False)
        module.a.weight.grad, module.b.weight.grad = torch.tensor([gradient[:2]]), torch.tensor([gradient[2:]])
        return module
    m = model([30., 0., 0., 40.])
    metrics = clip_training_gradients(m, 5.)
    torch.testing.assert_close(m.a.weight.grad, torch.tensor([[3., 0.]]))
    torch.testing.assert_close(m.b.weight.grad, torch.tensor([[0., 4.]]))
    assert metrics['grad_norm'] == 50. and metrics['grad_clip_scale'] == pytest.approx(.1)
    m = model([3., 0., 0., 4.])
    assert clip_training_gradients(m, 20.)['grad_clip_scale'] == 1.
    torch.testing.assert_close(m.b.weight.grad, torch.tensor([[0., 4.]]), rtol=0, atol=0)
    m = model([30., 0., 0., 40.])
    assert clip_training_gradients(m, 0.)['grad_norm'] == 50. and m.b.weight.grad.norm() == 40.  # 0 disables
    m = model([100., 100., 100., 100.])
    m.a.weight.grad.fill_(float('inf'))
    with pytest.raises(RuntimeError, match='non-finite'):
        clip_training_gradients(m, 5.)
    assert (m.b.weight.grad == 100.).all()
    for bad in (-1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite and nonnegative'):
            clip_training_gradients(torch.nn.Linear(2, 1), bad)


def test_nonfinite_microbatch_loss_raises_before_any_update():
    """Loss sums are read once per update; a nonfinite microbatch still blocks the step."""
    torch.manual_seed(4)
    cfg = config()
    model = build_model(cfg)
    data = batch(cfg, 2)
    poisoned = take(data, slice(1, 2))
    poisoned['x']['fine'] = torch.full_like(poisoned['x']['fine'], float('nan'))
    before = copy.deepcopy(model.state_dict())
    ema = copy.deepcopy(model)
    prepare_training(model, backend='eager')
    with pytest.raises(FloatingPointError, match='Nonfinite loss at step 7'):
        optimizer_update(model, ema, torch.optim.SGD(model.parameters(), lr=.1),
                         [take(data, slice(0, 1)), poisoned], 7, .1)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_training_compilation_emulates_eager_bf16_rounding(monkeypatch):
    import torch._inductor.config
    monkeypatch.setattr(torch._inductor.config, 'emulate_precision_casts', False)
    compiled = []
    monkeypatch.setattr(torch, 'compile', lambda module, **kwargs: compiled.append((module, kwargs)) or module)
    model = build_model(config())
    parameters, keys = list(model.parameters()), list(model.state_dict())
    assert prepare_training(model) is model
    assert not hasattr(model, '_orig_mod')
    assert [fn.__name__ for fn, _ in compiled] == ['training_forward', 'loss_terms']
    assert all(options == dict(dynamic=False, fullgraph=True, options=dict(emulate_precision_casts=True))
               for _, options in compiled)
    assert list(model.parameters()) == parameters
    assert list(model.state_dict()) == keys
    assert prepare_training(model) is model
    assert len(compiled) == 2  # Setup is idempotent.
    assert not torch._inductor.config.emulate_precision_casts


def test_checkpoint_resume_and_init_weights(tmp_path):
    resume_lr = .0005  # a resume may change the LR
    torch.manual_seed(7)
    cfg = config()
    args = SimpleNamespace(reset_optimizer=False, lr=.001)
    m = build_model(cfg)
    ema = copy.deepcopy(m)
    opt, done, origin = initialize_training_optimizer(m, ema, args)
    assert done == origin == 0
    data = batch(cfg)
    prepare_training(m, backend='eager')
    optimizer_update(m, ema, opt, [data], 1, .001, compute_metrics=False)
    spec = FiberVolumeSpec('unused', ct_zarr='unused', ct_level=0, ct_grid_scale=4., inputs='ct')
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history)
    path = tmp_path/'last.pt'
    save_checkpoint(path, m, ema, spec, sample, dict(step=1, lr_restart_step=0, optimizer=opt.state_dict(),
                                                     rng=training_rng_state()))
    expected_rng = torch.rand(3)
    loaded, crop, nh, loaded_spec, ck = load_checkpoint(path, 'cpu')
    torch.testing.assert_close(forward(loaded, data)['points'], forward(ema, data)['points'], rtol=0, atol=0)
    assert crop == cfg.fine and nh == cfg.n_history and loaded_spec == spec
    # Resume restores optimizer, RNG stream and schedule origin: the next update is identical.
    restored = build_model(cfg)
    restored_ema = copy.deepcopy(restored)
    args.lr = resume_lr
    restored_opt, done, origin = initialize_training_optimizer(restored, restored_ema, args, ck)
    assert (done, origin) == (1, 0)
    torch.testing.assert_close(torch.rand(3), expected_rng, rtol=0, atol=0)
    prepare_training(restored, backend='eager')
    optimizer_update(m, ema, opt, [data], 2, resume_lr, compute_metrics=False)
    optimizer_update(restored, restored_ema, restored_opt, [data], 2, resume_lr, compute_metrics=False)
    for p, q in zip(m.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    # init_weights: model and EMA tensors load by name and shape; the run's own architecture is built.
    initial = read_checkpoint(path, 'cpu')
    assert config_from_checkpoint(initial) == cfg
    fresh, fresh_ema = train.initialize_model_weights(config_from_checkpoint(initial), 'cpu', initial)
    assert fresh.initialization_report == dict(fresh=[], unexpected=[], mismatched=[])
    for key, module in (('model', fresh), ('ema', fresh_ema)):
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, initial[key][name], rtol=0, atol=0)
    wider, _ = train.initialize_model_weights(replace(cfg, ffn=64), 'cpu', initial, exclude=('hazard',))
    report = wider.initialization_report
    assert report['mismatched'] and all('.ffn.' in name for name in report['mismatched'])
    assert {'hazard.weight', 'hazard.bias'} <= set(report['fresh']) and not report['unexpected']
    torch.testing.assert_close(wider.cell_token.weight, initial['model']['cell_token.weight'], rtol=0, atol=0)


def test_loss_supervises_every_proposal():
    import torch.nn.functional as F
    # Commit window and auxiliary weighting of the initial proposal.
    cfg = config()
    b = label_batch(cfg, 1)
    b['dense_mask'][:, -2:] = 0
    points = torch.tensor([[[1., 0., 1.], [1., 0., 2.], [3., 0., 3.], [3., 0., 4.]]], requires_grad=True)
    initial = (points.detach()*torch.tensor([2., 1., 1.])).requires_grad_()
    logits = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
    out = proposal_output(torch.stack((initial, points), 1), logits[:, None].expand(-1, 2, -1))
    result = loss_terms(out, b, cfg, n_commit=2)
    def expected(curve):
        dense = F.interpolate(curve[..., :2].transpose(1, 2), size=13, mode='linear', align_corners=True).transpose(1, 2)
        errors = F.smooth_l1_loss(dense, b['dense_ab'], reduction='none').mean(-1)
        return .5*errors[:, :5].mean()+.5*errors[:, :11].mean()
    torch.testing.assert_close(result['geometry_per_state'][0], .75*expected(points)+.25*expected(initial))
    expected_conf = (F.softplus(logits[0, :2]).sum()+F.softplus(-logits[0, 2])+F.softplus(-logits[0, 0]))/2
    torch.testing.assert_close(result['confidence_per_state'][0], expected_conf)
    result['geometry_per_state'].sum().backward()
    assert points.grad.abs().sum() > 0 and initial.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='Commit window'):
        loss_terms(out, b, cfg, n_commit=5)
    # Selecting an earlier path must not deprive later proposals of geometry loss.
    cfg = config(recurrent_refinement_steps=2)
    b = label_batch(cfg, 1)
    curves = [torch.tensor([[[error, 0., float(z)] for z in range(1, 5)]], requires_grad=True) for error in (3., 2., 1.)]
    terms = loss_terms(proposal_output(torch.stack(curves, 1), torch.zeros(1, 3, 4), selected=0), b, cfg)
    # Smooth L1 averaged over x,y; first two stages share the 25% auxiliary term.
    torch.testing.assert_close(terms['geometry_per_state'], torch.tensor([.75*.25+.25*((3-.5)/2+(2-.5)/2)/2]))
    terms['geometry_per_state'].sum().backward()
    assert all(curve.grad[..., 0].abs().sum() > 0 for curve in curves)
    # Every generated attempt gets confidence supervision from its own first failure.
    curves = torch.zeros(1, 3, 4, 3)
    curves[..., 2] = torch.arange(1, 5)
    curves[:, 1, :, 0] = 3.  # Immediate failure.
    curves[:, 2, 2:, 0] = 3.  # Failure in the third segment.
    hazards = torch.zeros(1, 3, 4, requires_grad=True)
    loss_terms(proposal_output(curves, hazards, selected=0), b, cfg)['confidence_per_state'].sum().backward()
    assert (hazards.grad[0, 0] > 0).all()
    assert hazards.grad[0, 1, 0] < 0 and hazards.grad[0, 1, 1:].eq(0).all()
    assert (hazards.grad[0, 2, :2] > 0).all()
    assert hazards.grad[0, 2, 2] < 0 and hazards.grad[0, 2, 3] == 0
