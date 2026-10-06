"""Flow matching (models/flow.py) on the crop transformer: the common trainer, proposals, selection and losses."""
import copy
from types import SimpleNamespace

import pytest
import torch

from model_fixtures import coordinate_batch, flow_config
from vesuvius.neural_tracing.fiber_follow.models.model import build_model, config_from_checkpoint
from vesuvius.neural_tracing.fiber_follow.models.flow import fit_flow_sigma, flow_targets
from vesuvius.neural_tracing.fiber_follow.train.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.train.train import (
    prepare_training, optimizer_update, save_checkpoint, load_checkpoint,
    initialize_training_optimizer, initialize_model_weights,
)
from vesuvius.neural_tracing.fiber_follow.train.runloop import training_rng_state
from vesuvius.neural_tracing.fiber_follow.data.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec


def config(**options):
    return flow_config(**dict(dict(flow_draws=2), **options))


def take(value, index):
    return {k: take(v, index) for k, v in value.items()} if isinstance(value, dict) else value[index]


def test_flow_draws_and_trace_rows_are_independent_and_unknown_targets_are_censored():
    cfg = config(); model = build_model(cfg).eval(); b = coordinate_batch(cfg)
    with torch.no_grad():
        out = model(b['x'], b['hist'], b['hmask'])
        for row in range(2):
            one = take(b, slice(row, row+1))
            result = model(one['x'], one['hist'], one['hmask'])
            torch.testing.assert_close(result['points'], out['points'][row:row+1], atol=2e-5, rtol=2e-5)
        ctx = model.context(b['x'], b['hist'], b['hmask']); model.prepare_prediction(ctx, b['hist'])
        y, t = torch.randn(2, 2, 4, 2), torch.rand(2, 2)
        both = model.velocity_field(ctx, y, t)
        solo = model.velocity_field(ctx, y[:, :1], t[:, :1])
        torch.testing.assert_close(both[:, :1], solo, atol=1e-6, rtol=1e-5)
        b['geometry_valid'].zero_(); b['dense_ab'].fill_(float('nan'))
        target, mask = flow_targets(b, cfg)
        assert not mask.any() and not target.any()
        result = model.training_forward(b['x'], b['hist'], b['hmask'], .5, b)
        assert torch.isfinite(result['flow_per_state']).all() and not result['flow_per_state'].any()


def test_flow_optimizer_ema_checkpoint_and_resume(tmp_path):
    torch.manual_seed(32)
    cfg = config(); model, ema = initialize_model_weights(cfg, 'cpu')
    args = SimpleNamespace(lr=.001, reset_optimizer=False)
    opt, _, _ = initialize_training_optimizer(model, ema, args)
    prepare_training(model, backend='eager'); b = coordinate_batch(cfg)
    metrics = optimizer_update(model, ema, opt, [b], 1, .001)
    assert metrics['prediction_loss_type'] == 'flow' and metrics['flow'] > 0
    path = tmp_path/'last.pt'
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future)
    save_checkpoint(path, model, ema, FiberVolumeSpec('unused', inputs='ct'), sample,
                    dict(step=1, optimizer=opt.state_dict(), rng=training_rng_state()))
    loaded, _, _, _, ck = load_checkpoint(path, 'cpu')
    assert loaded.cfg == cfg
    restored, restored_ema = initialize_model_weights(cfg, 'cpu')
    restored_opt, done, _ = initialize_training_optimizer(restored, restored_ema, args, ck)
    assert done == 1 and restored_opt.state
    prepare_training(restored, backend='eager')
    # Fixed velocity draws isolate update/accumulation semantics from RNG order.
    b['flow_noise'] = torch.randn(2, cfg.flow_draws, cfg.n_future, 2)
    b['flow_times'] = torch.rand(2, cfg.flow_draws)
    optimizer_update(model, ema, opt, [b], 2, .001, compute_metrics=False)
    optimizer_update(restored, restored_ema, restored_opt, [b],
                     2, .001, compute_metrics=False)
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_flow_loss_and_gradients_are_weighted_per_state_across_microbatches():
    torch.manual_seed(53)
    cfg = config(); whole = build_model(cfg); split = copy.deepcopy(whole)
    b = coordinate_batch(cfg)
    b['dense_mask'][1, 5:] = 0  # Unequal numbers of known planes must not reweight states.
    b['flow_noise'] = torch.randn(2, cfg.flow_draws, cfg.n_future, 2)
    b['flow_times'] = torch.tensor([[.1, .6], [.2, .8]])
    def objective(model, data):
        out = model.select_prediction(model.training_forward(data['x'], data['hist'], data['hmask'], .5, data))
        terms = loss_terms(out, data, cfg)
        return (terms['geometry_per_state']+.5*terms['confidence_per_state']).sum()/2
    expected = objective(whole, b); expected.backward()
    actual = 0.
    for index in range(2):
        loss = objective(split, take(b, slice(index, index+1)))
        loss.backward(); actual += loss.detach()
    torch.testing.assert_close(actual, expected.detach(), atol=1e-6, rtol=1e-5)
    for (name, p), q in zip(whole.named_parameters(), split.parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_inference_samples_retry_after_the_zero_start_path_only_when_it_is_not_accepted():
    torch.manual_seed(17)
    model = build_model(config()).eval()
    sampled = build_model(config(flow_samples=3, flow_sample_scale=1.5)).eval()
    sampled.load_state_dict(model.state_dict())
    b = coordinate_batch(model.cfg)
    with torch.no_grad():
        for threshold in (0., .5, 1.):
            plain = model(b['x'], b['hist'], b['hmask'], threshold)
            out = sampled(b['x'], b['hist'], b['hmask'], threshold)
            assert out['refinement_points'].shape[1] == 4
            torch.testing.assert_close(out['refinement_points'][:, 0], plain['points'])
            torch.testing.assert_close(out['refinement_confidence'][:, 0], plain['confidence'])
            assert not torch.equal(out['refinement_points'][:, 1], out['refinement_points'][:, 0])
            index = torch.arange(len(out['points']))
            torch.testing.assert_close(out['points'], out['refinement_points'][index, out['selected_refinement']])
            if threshold == 0.:  # the zero-start path is accepted, so no sample is used
                assert (out['selected_refinement'] == 0).all()
                torch.testing.assert_close(out['points'], plain['points'])


def test_sampled_proposals_are_scored_in_training_and_the_zero_start_path_is_unchanged():
    torch.manual_seed(23)
    model = build_model(config())
    sampled = build_model(config(flow_samples=2, flow_sample_scale=2.))
    sampled.load_state_dict(model.state_dict())
    b = coordinate_batch(model.cfg)
    out = sampled.training_forward(b['x'], b['hist'], b['hmask'], .5, b)
    plain = model.training_forward(b['x'], b['hist'], b['hmask'], .5, b)
    assert out['refinement_points'].shape[1] == 3 and out['refinement_mask'].all()
    torch.testing.assert_close(out['refinement_points'][:, 0], plain['refinement_points'][:, 0])
    terms = loss_terms(sampled.select_prediction(out), b, sampled.cfg)
    assert int(terms['refinement_attempts_sum']) == 2*3
    terms['confidence_per_state'].sum().backward()
    assert all(float(m.weight.grad.abs().sum()) > 0 for m in (sampled.score[0], sampled.hazard))


def test_unknown_planes_are_attended_only_with_own_path_and_the_scale_floor_bounds_the_prior():
    torch.manual_seed(29)
    b = coordinate_batch(config())
    b['dense_mask'][:, b['dense_mask'].shape[1]//2:] = 0  # the trailing planes have no target
    for mode, attended in (('padded', False), ('own_path', True)):
        model = build_model(config(flow_unknown_planes=mode)).eval()
        _, known = flow_targets(b, model.cfg)
        assert known.any() and not known.all()
        losses = []
        for seed in (0, 1):
            noise = torch.randn(2, model.cfg.flow_draws, model.cfg.n_future, 2, generator=torch.Generator().manual_seed(7))
            # Change the Gaussian draw at target-less planes only.
            noise[~known[:, None].expand(-1, model.cfg.flow_draws, -1)] = torch.randn(
                int((~known).sum())*model.cfg.flow_draws, 2, generator=torch.Generator().manual_seed(seed))
            with torch.no_grad():
                out = model.training_forward(b['x'], b['hist'], b['hmask'], .5,
                                             dict(b, flow_noise=noise, flow_times=torch.tensor([[.3, .8]]*2)))
            losses.append(out['flow_per_state'])
        assert torch.isfinite(losses[0]).all()
        assert (not torch.equal(losses[0], losses[1])) == attended
    cfg = config(flow_sigma_floor=3., flow_sigma=((3., 3.),)*4)
    data = coordinate_batch(cfg)
    data['dense_ab'][0, :, 0] = -2.; data['dense_ab'][1, :, 0] = 2.
    assert fit_flow_sigma(iter([data]), cfg, 2) == ((3., 3.),)*cfg.n_future
    with pytest.raises(ValueError, match='scale floor'):
        config(flow_sigma_floor=3.)  # the fixture's unit scales lie below the floor


def test_pseudo_huber_flow_loss_is_half_the_squared_distance_for_small_residuals_and_linear_beyond_c():
    torch.manual_seed(37)
    b = coordinate_batch(config())
    noise = torch.randn(2, 2, 4, 2)*3
    target, known = flow_targets(b, config())
    # A zero velocity makes the residual (noise - target) per known path point (unit scales).
    distance = (noise-target[:, None]).norm(dim=-1)[known[:, None].expand(-1, 2, -1)].reshape(2, -1)
    for loss, c, expected in (('mse', 1., .5*distance.square()),
                              ('pseudo_huber', 1., ((1+distance.square()).sqrt()-1)),
                              ('pseudo_huber', .5, .25*((1+(distance/.5).square()).sqrt()-1))):
        model = build_model(config(flow_loss=loss, flow_huber_c=c))
        model.velocity_field = lambda ctx, y, t, known=None: torch.zeros_like(y)
        out = model.training_forward(b['x'], b['hist'], b['hmask'], .5, dict(b, flow_noise=noise))
        torch.testing.assert_close(out['flow_per_state'], expected.mean(-1).float(), rtol=1e-5, atol=1e-6)
    huber = ((1+distance.square()).sqrt()-1)
    assert (huber <= .5*distance.square()+1e-6).all()  # never above the squared loss; linear for large residuals
    with pytest.raises(ValueError, match='Flow loss'):
        config(flow_loss='l1')


def test_flow_geometry_weight_trains_the_integrated_zero_start_path_and_nothing_else_sees_its_gradient():
    torch.manual_seed(41)
    plain = build_model(config(flow_samples=2))
    model = build_model(config(flow_samples=2, flow_geometry_weight=.5))
    model.load_state_dict(plain.state_dict())
    b = coordinate_batch(model.cfg)
    b['flow_noise'] = torch.randn(2, 2, 4, 2)
    torch.manual_seed(5); base = plain.training_forward(b['x'], b['hist'], b['hmask'], .5, b)
    torch.manual_seed(5); out = model.training_forward(b['x'], b['hist'], b['hmask'], .5, b)
    assert 'flow_geometry_points' not in base and out['flow_geometry_points'].requires_grad
    # Same proposals, scores and flow loss; the proposals reach the scorer detached.
    for key in ('refinement_points', 'refinement_confidence', 'flow_per_state'):
        torch.testing.assert_close(out[key], base[key], atol=2e-5, rtol=1e-5)
    assert not out['refinement_points'].requires_grad
    torch.testing.assert_close(out['flow_geometry_points'], out['refinement_points'][:, 0], atol=2e-5, rtol=1e-5)
    terms = loss_terms(model.select_prediction(out), b, model.cfg)
    torch.testing.assert_close(terms['geometry_per_state'],
                               out['flow_per_state']+.5*terms['flow_path_geometry_per_state'])
    assert 'flow_path_geometry_per_state' not in loss_terms(plain.select_prediction(base), b, plain.cfg)
    terms['flow_path_geometry_per_state'].sum().backward()
    assert model.velocity.weight.grad.abs().sum() > 0
    assert all(p.grad is None or not p.grad.any() for m in (model.score, model.hazard) for p in m.parameters())
    model.eval()
    with torch.no_grad():
        assert 'flow_geometry_points' not in model.training_forward(b['x'], b['hist'], b['hmask'], .5)
    with pytest.raises(ValueError, match='zero-start path'):
        config(flow_samples=2, flow_zero_start=False, flow_geometry_weight=.5)


def test_qk_norm_and_modulation_bound_hold_path_attention_under_runaway_time_modulation_and_old_checkpoints_load(monkeypatch):
    torch.manual_seed(43)
    sdpa, logits, modulation = torch.nn.functional.scaled_dot_product_attention, [], []

    def record(q, k, v, attn_mask=None, **kw):
        logits.append(float((q@k.transpose(-1, -2)).abs().max())/q.shape[-1]**.5)
        return sdpa(q, k, v, attn_mask=attn_mask, **kw)

    def runaway(model):
        """Largest attention logit and |shift|, |scale|, |gate| applied with the drift seen in training."""
        b = coordinate_batch(model.cfg)
        for layer in model.layers:
            queries = layer.queries
            monkeypatch.setattr(layer, 'queries', lambda x, pair, mask, m, queries=queries, **kw: (
                modulation.append(m.abs().amax((0, 1, 2, 4))), queries(x, pair, mask, m, **kw))[1])
        with torch.no_grad():
            ctx = model.context(b['x'], b['hist'], b['hmask'])
            for m in model.time_modulation:  # modulation of tens, as in the diverged run
                torch.nn.init.normal_(m.weight, std=30.)
            logits.clear(); modulation.clear()
            monkeypatch.setattr(torch.nn.functional, 'scaled_dot_product_attention', record)
            velocity = model.velocity_field(ctx, torch.randn(2, 2, 4, 2), torch.rand(2, 2))
            monkeypatch.setattr(torch.nn.functional, 'scaled_dot_product_attention', sdpa)
        assert torch.isfinite(velocity).all()
        return max(logits), torch.stack(modulation).amax(0).tolist()

    bounded = build_model(config(flow_time_conditioning='adaln_zero')).eval()
    assert bounded.cfg.qk_norm and bounded.cfg.flow_modulation_bound == 4. and all(hasattr(l, 'q_norm') for l in bounded.layers)
    head = bounded.cfg.hidden//bounded.cfg.heads
    logit, (shift, scale, gate) = runaway(bounded)
    assert logit <= head**.5+1e-3 and max(shift, scale) <= 4. and gate <= 1.  # |q|=|k|=sqrt(head) at unit gain
    plain = build_model(config(flow_time_conditioning='adaln_zero', qk_norm=False, flow_modulation_bound=0.)).eval()
    logit, (shift, scale, gate) = runaway(plain)
    assert logit > 20*head**.5 and min(shift, scale, gate) > 10.
    # A flow checkpoint recorded before these fields existed was trained without them and loads as such.
    recorded = {k: v for k, v in plain.cfg.to_dict().items() if k not in ('qk_norm', 'flow_modulation_bound')}
    cfg = config_from_checkpoint(dict(model_type='flow', model_cfg=recorded))
    assert not cfg.qk_norm and not cfg.flow_modulation_bound
    build_model(cfg).load_state_dict(plain.state_dict())
