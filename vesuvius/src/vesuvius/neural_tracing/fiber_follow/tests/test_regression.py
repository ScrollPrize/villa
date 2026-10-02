"""Aligned direct follower: forward contracts, optimizer updates, checkpoints, CT crops and run diagnostics."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.models.model import DirectFollower, build_model, feature_grid
from vesuvius.neural_tracing.fiber_follow.train.supervision import geometry_mask, loss_terms
from vesuvius.neural_tracing.fiber_follow.train import train
from vesuvius.neural_tracing.fiber_follow.train.train import ARCHITECTURES, checkpoint_config, clip_training_gradients, initialize_training_optimizer, load_checkpoint, optimizer_update, prepare_training, save_checkpoint
from vesuvius.neural_tracing.fiber_follow.data.observations import image_crop
from vesuvius.neural_tracing.fiber_follow.data.data import SampleConfig, TracedFiber, fiber_identities, fiber_manifest
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, crop_local_grid, sample_oriented_fast
from vesuvius.neural_tracing.fiber_follow.train.runloop import read_checkpoint, training_rng_state
from vesuvius.neural_tracing.fiber_follow.train.training_log import DirectTrainingInterval, SamplingLedger, format_training_log
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec
from label_fixtures import set_terminal
from model_fixtures import aligned_batch, aligned_config, batch as label_batch, forward, proposal_output


def config(**kwargs):
    """Small aligned model; one refinement unless refinement is the subject."""
    return aligned_config(**{'recurrent_refinement_steps': 1, **kwargs})


def batch(cfg, b=2):
    return aligned_batch(cfg, b)


def take(value, sl):
    return {k: take(v, sl) for k, v in value.items()} if isinstance(value, dict) else value[sl]


def test_aligned_forward_is_deterministic_causal_and_trainable():
    torch.manual_seed(20)
    m = DirectFollower(config())
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
    x['history_slabs'][~x['history_valid']] = float('nan')
    x['history_path_points'][~x['history_path_valid']] = float('nan')
    x['path_geometry'][~x['path_geometry_valid'].bool()] = float('nan')
    other = forward(m, hidden)
    for key in out:
        assert torch.isfinite(other[key]).all(), key
        torch.testing.assert_close(out[key], other[key], rtol=0, atol=0)
    loss_terms(out, b, m.cfg)['geometry_per_state'].mean().backward()
    for module in (m.coordinates, m.encoder.patch_projection, m.reference_token[0]):
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0
    for module in (m.encoder.stem, m.history_encoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


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


def test_feature_coordinates_respect_even_sized_strided_lattice():
    crop = CropSpec(depth=16, width=12, behind=7, spacing=.5)
    # Deep convolution lattice index (x,y,z) = (2,1,3), stride four.
    p = torch.tensor([[[8*.5-5.5*.5, 4*.5-5.5*.5, 12*.5-7*.5]]])
    grid = feature_grid(p, crop, (4, 3, 3), stride=4)
    torch.testing.assert_close(grid, torch.tensor([[[1., 0., 1.]]]))


def test_optimizer_update_partition_metrics_and_sampling_ledger():
    """Microbatches keep the objective and update; metrics and the ledger don't change training or RNG."""
    torch.manual_seed(3)
    cfg = config()
    a = DirectFollower(cfg)
    bmodel = copy.deepcopy(a)
    data = batch(cfg, 2)
    data['dense_mask'][0, 5:] = 0
    set_terminal(data, 1)
    data.update(source=torch.tensor([0, 2]), task_requested=torch.tensor([0, 6]), task_delivered=torch.tensor([0, 0]),
                task_fallback=torch.tensor([0, 1]), ct_frame_rejected_batches=torch.tensor([3, 0]))
    data['x'].update(ct_frame_source=torch.tensor([0, 2]),
                     ct_frame_energy=torch.tensor([.2, 0.]), ct_frame_gap=torch.tensor([.8, 0.]),
                     history_frame_source=torch.tensor([[0, 1, -1], [2, -1, -1]]),
                     history_frame_energy=torch.tensor([[.1, 0., 0.], [0., 0., 0.]]),
                     history_frame_gap=torch.tensor([[.6, 0., 0.], [0., 0., 0.]]))
    averages = [copy.deepcopy(m) for m in (a, bmodel)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=.001) for m in (a, bmodel)]
    for model in (a, bmodel):
        prepare_training(model, batch_size=2, backend='eager')
    ledger = SamplingLedger()
    rng = torch.get_rng_state()
    results = [optimizer_update(a, averages[0], optimizers[0], [data], 1, .001, n_commit=2, compute_metrics=False,
                                ledger=ledger)]
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    results.append(optimizer_update(bmodel, averages[1], optimizers[1], [take(data, slice(0, 1)), take(data, slice(1, 2))],
                                    1, .001, n_commit=2, compute_metrics=True))
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert results[0]['loss'] == pytest.approx(results[1]['loss'], rel=2e-6)
    for key in ('positive_confidence_targets', 'negative_confidence_targets', 'confidence_terminal_states'):
        assert results[0][key] == results[1][key]
    for p, q in zip(a.parameters(), bmodel.parameters()):
        torch.testing.assert_close(p, q, rtol=2e-5, atol=2e-7)
    metrics = results[0]
    assert 'decisions' not in metrics and results[1]['decisions']['n_commit'] == 2
    assert sum(metrics[k] for k in ('point_correct_count', 'point_wrong_count', 'point_unknown_count')) == 8
    assert metrics['refinement_attempts_sum'] >= 2
    assert metrics['ct_frame_rejected_batches'] == 3
    assert (metrics['ct_frame_count'], metrics['ct_frame_transported'], metrics['ct_frame_deterministic']) == (2, 0, 1)
    assert (metrics['history_frame_count'], metrics['history_frame_transported'],
            metrics['history_frame_deterministic']) == (3, 1, 1)
    sampling = ledger.summary()['0']
    assert sampling['requested_share'] == dict(fresh=.5, dagger_ordinary=.5)
    assert sampling['delivered_share'] == dict(fresh=1.)
    assert sampling['fallbacks'] == {'dagger_ordinary->fresh': 1}
    assert sampling['sources'] == dict(fresh=1, replay=1)
    assert sampling['positive_targets']+sampling['negative_targets'] == (
        metrics['positive_confidence_targets']+metrics['negative_confidence_targets'])
    interval = DirectTrainingInterval()
    interval.add(metrics)
    interval.add(metrics)
    summary = interval.summary()
    assert summary['ct_frame_count'] == 4 and summary['history_frame_count'] == 6
    printed = format_training_log(dict(step=50, geometry=1., loss=1., lr=.001, interval=summary,
        n_future=4, tolerance=1.5, sampling=ledger.summary(), interval_update_seconds=1.,
        interval_data_seconds=.1, interval_samples_per_second=4.))
    assert 'source 0: requested dagger_ordinary 50%, fresh 50% | delivered fresh 100%' in printed
    assert "fallbacks {'dagger_ordinary->fresh': 1}" in printed
    assert 'current 2/4 fallbacks (0 transported, 2 deterministic); mean gap 0.400' in printed
    assert 'history 4/6 fallbacks (2 transported, 2 deterministic); mean gap 0.200' in printed


def test_gradient_clipping_groups_and_guards():
    def model(history, rest):
        module = torch.nn.Module()
        module.history_encoder = torch.nn.Linear(2, 1, bias=False)
        module.encoder = torch.nn.Linear(2, 1, bias=False)
        module.history_encoder.weight.grad = torch.tensor([history])
        module.encoder.weight.grad = torch.tensor([rest])
        return module
    m = model([3e6, 4e6], [3., 4.])
    metrics = clip_training_gradients(m, 5., 20.)
    torch.testing.assert_close(m.history_encoder.weight.grad, torch.tensor([[3., 4.]]))
    torch.testing.assert_close(m.encoder.weight.grad, torch.tensor([[3., 4.]]), rtol=0, atol=0)
    assert metrics['history_grad_norm'] == pytest.approx(5e6) and metrics['rest_grad_norm'] == 5.
    assert metrics['history_grad_clip_scale'] == pytest.approx(1e-6) and metrics['rest_grad_clip_scale'] == 1.
    for history_cap, rest_cap in [(0., 20.), (5., 0.)]:
        m = model([30., 40.], [30., 40.])
        result = clip_training_gradients(m, history_cap, rest_cap)
        assert m.history_encoder.weight.grad.norm().item() == pytest.approx(history_cap or 50.)
        assert m.encoder.weight.grad.norm().item() == pytest.approx(rest_cap or 50.)
        assert result['history_grad_norm'] == result['rest_grad_norm'] == 50.
    for bad in ('history_encoder', 'encoder'):
        m = model([100., 100.], [100., 100.])
        getattr(m, bad).weight.grad.fill_(float('inf'))
        with pytest.raises(RuntimeError, match='non-finite'):
            clip_training_gradients(m, 5., 5.)
        good = 'encoder' if bad == 'history_encoder' else 'history_encoder'
        assert (getattr(m, good).weight.grad == 100.).all()
    for bad in (-1., float('nan'), float('inf')):
        for limits in ((bad, 20.), (5., bad)):
            with pytest.raises(ValueError, match='finite and nonnegative'):
                clip_training_gradients(torch.nn.Linear(2, 1), *limits)


def test_nonfinite_microbatch_loss_raises_before_any_update():
    """Loss sums are read once per update; a nonfinite microbatch still blocks the step."""
    torch.manual_seed(4)
    cfg = config()
    model = DirectFollower(cfg)
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
    model = DirectFollower(config())
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


def test_checkpoint_resume_init_weights_and_architecture_contract(tmp_path):
    torch.manual_seed(7)
    cfg = config()
    args = SimpleNamespace(reset_optimizer=False, lr=.001)
    m = DirectFollower(cfg)
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
    restored = DirectFollower(cfg)
    restored_ema = copy.deepcopy(restored)
    restored_opt, done, origin = initialize_training_optimizer(restored, restored_ema, args, ck)
    assert (done, origin) == (1, 0)
    torch.testing.assert_close(torch.rand(3), expected_rng, rtol=0, atol=0)
    prepare_training(restored, backend='eager')
    optimizer_update(m, ema, opt, [data], 2, .001, compute_metrics=False)
    optimizer_update(restored, restored_ema, restored_opt, [data], 2, .001, compute_metrics=False)
    for p, q in zip(m.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    # --init-weights: the launcher's architecture flags must match; model and EMA load strictly.
    initial = read_checkpoint(path, ARCHITECTURES, 'cpu')
    assert checkpoint_config(initial) == cfg
    launcher = dict(encoder=('patch4', 'conv'), token_only=(True, False), history_encoder=('fine', 'legacy'),
                    history_path_tokens=(True, False), path_geometry_tokens=(True, False))
    for name, (requested, wrong) in launcher.items():
        resolve = getattr(train, 'resolve_'+name)
        assert resolve(requested, initial) == requested
        with pytest.raises(ValueError, match='match'):
            resolve(wrong, initial)
    for key in ('model', 'ema'):
        build_model(checkpoint_config(initial)).load_state_dict(initial[key], strict=True)
    with pytest.raises(ValueError, match='Unsupported'):
        checkpoint_config(dict(initial, architecture='axial_patch4_fiber_slabs_v10'))
    # The init fiber check compares identities: repaired geometry passes, a changed source does not.
    arc = np.arange(50.)
    fiber = TracedFiber('f', np.c_[arc*0, arc*0, arc], arc, '', source_hash='abc')
    repaired = replace(fiber, points=fiber.points+[.5, 0., 0.])
    assert fiber_manifest([fiber]) != fiber_manifest([repaired])
    assert fiber_identities(fiber_manifest([fiber])) == fiber_identities(fiber_manifest([repaired]))
    assert fiber_identities(fiber_manifest([fiber])) != fiber_identities(fiber_manifest([replace(fiber, source_hash='abd')]))


def test_inference_checkpoint_keeps_checkpoint_storage_on_cpu(monkeypatch):
    cfg = config()
    spec = FiberVolumeSpec('unused', ct_zarr='unused')
    ck = dict(ema={}, vol_spec=spec.to_dict())
    calls = []
    monkeypatch.setattr(train, 'read_checkpoint', lambda path, arch, device: calls.append(('read', device)) or ck)
    monkeypatch.setattr(train, 'checkpoint_config', lambda checkpoint: cfg)
    class Model:
        def to(self, device, **kwargs):
            calls.append(('model', device))
            return self
        def load_state_dict(self, weights):
            assert weights is ck['ema']
        def eval(self):
            return self
    monkeypatch.setattr(train, 'build_model', lambda config: Model())
    train.load_checkpoint('unused', 'cuda')
    assert calls == [('read', 'cpu'), ('model', 'cuda')]


def test_ct_crops_match_reference_sampling_at_edges_and_native_resolution():
    """Per-item tight blocks give the reference values, zero fill beyond the array, on the 2x CT grid."""
    from vesuvius.neural_tracing.fiber_follow.data.data import _grid_flat, read_blocks
    from vesuvius.neural_tracing.fiber_follow.shared.fast_sample import sample_crop
    from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import normalize_ct
    rng = np.random.default_rng(11)
    data = rng.integers(1, 256, (70, 60, 64), dtype=np.uint8)

    def read(start, size):
        out = np.zeros(tuple(size), np.uint8)
        lo, hi = np.maximum(start, 0), np.minimum(start+size, data.shape)
        if np.all(hi > lo):
            out[tuple(slice(a-s, b-s) for a, b, s in zip(lo, hi, start))] = data[tuple(map(slice, lo, hi))]
        return out
    # Avoid a hard-mask boundary at an exactly representable intensity: backends differ in rounding.
    calibration = dict(threshold=62.123, noise=4.)
    vol = SimpleNamespace(presence=None, input_scale=2., raw_block=lambda s, z: read(s, z)[None],
                          spec=SimpleNamespace(ct_normalization=calibration))
    crop = CropSpec(depth=14, width=10, behind=5, spacing=.5)
    items = []
    for _ in range(24):
        # Interior, straddling each face, and fully outside; arbitrary orientation.
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        items.append(dict(pos=rng.uniform(-6, 38, 3), frame=q*np.sign(np.linalg.det(q))))
    image = image_crop(items, vol, crop, input_mode='ct')
    assert image.shape == (24, 1, crop.depth, crop.width, crop.width)
    raw, starts = read_blocks(items, vol, crop)
    grid, empty, mask = _grid_flat(crop), np.empty((0, 3), np.float32), np.empty(0, np.float32)
    expected = np.stack([sample_crop(raw[j], starts[j], item['pos']*2., item['frame']*2., grid, empty, mask, 2, 1.,
                                     'points')[0].reshape(crop.depth, crop.width, crop.width)
                         for j, item in enumerate(items)]).astype(np.float32)
    reference = torch.from_numpy(expected.copy())
    for row in expected:
        normalize_ct(row, calibration)
    assert torch.equal(image[:, 0], torch.from_numpy(expected))
    assert 0 < float((reference != 0).float().mean()) < 1
    # The fast sampler agrees with the oriented torch reference.
    oriented = torch.cat([sample_oriented_fast(torch.from_numpy(raw[j:j+1]).reshape(1, 1, *raw[j].shape[-3:]),
                                               torch.from_numpy(starts[j:j+1]), torch.from_numpy(item['pos'][None]*2.).float(),
                                               torch.from_numpy(item['frame'][None]*2.).float(),
                                               torch.from_numpy(crop_local_grid(crop)).float())
                          for j, item in enumerate(items)])
    torch.testing.assert_close(oriented[:, 0], reference, atol=3e-4, rtol=1e-3)


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


def test_decision_metrics_score_actual_commits_censor_unknowns_and_pool_counts():
    import json
    from vesuvius.neural_tracing.fiber_follow.evaluation.diagnostics import decision_rows, summarize_decisions
    cfg = config()
    b = label_batch(cfg, 5)
    b['match_distance'] = torch.tensor([.5, 1.25, 1.75, 2.5, 7.])
    set_terminal(b, 4)
    b['dense_mask'][3] = 0
    points = torch.zeros(5, 4, 3)
    points[..., 2] = torch.arange(1, 5)
    points[1, 2:, 0] = 3.  # four-point prefix fails, but only first two accepted
    points[2, :, 0] = 2.   # knowingly wrong accepted prefix
    conf = torch.ones(5, 4)*.9
    conf[0] = .1           # false stop on a correct path
    conf[1, 2:] = .1
    out = proposal_output(points[:, None], torch.zeros(5, 1, 4))
    out.update(initial_points=points+torch.tensor([.2, 0., 0.]), confidence=conf,
               refinement_confidence=conf[:, None])
    stats = summarize_decisions(decision_rows(out, b, cfg, n_commit=4), 4)
    all_stats = stats['by_state']['all']
    assert all_stats['first_known'] == 4 and all_stats['first_correct'] == 2
    assert all_stats['commit_correct'] == 1
    assert all_stats['gate_0.5'] == dict(false_stops=1, accepted_known=3, accepted_wrong=2,
                                      accepted_unknown=1, terminal_continues=1)
    assert stats['by_state']['terminal']['states'] == 1
    assert stats['by_displacement']['d<=3']['states'] == 4 and stats['by_displacement']['d>6']['states'] == 1
    assert stats['by_displacement']['d>6']['final_error_mean'] is None
    assert all_stats['final_error_mean'] == pytest.approx(all_stats['final_error_sum']/all_stats['final_error_count'])
    json.dumps(stats, allow_nan=False)
    points[0, 0, 0] = float('nan')
    bad = summarize_decisions(decision_rows(out, b, cfg, n_commit=4), 4)['by_state']['all']
    assert bad['recovery_blocked'] == 1 and bad['final_nonfinite_count'] > 0
    json.dumps(bad, allow_nan=False)
    # History groups measure what the model actually received.
    cfg = config(n_history=64)
    b = label_batch(cfg, 4)
    for row, length in zip(b['hmask'], (0, 4, 16, 64)):
        row[length:] = 0
    stats = summarize_decisions(decision_rows(proposal_output(points[:4, None].nan_to_num(), torch.zeros(4, 1, 4)),
                                              b, cfg), 4)
    assert {k: v['states'] for k, v in stats['by_history'].items()} == {'0': 1, '1-8': 1, '9-32': 1, '>32': 1}
    assert all(0 <= v['first_confidence_mean'] <= 1 for v in stats['by_history'].values())


def test_monitor_fixtures_are_fixed_private_rng_and_exclude_other_splits(tmp_path, monkeypatch):
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.evaluation.recovery.FiberVolume', lambda *a, **kw: None)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.ct_tensor',
                        lambda vol, pos: np.outer(np.array([1., 0., 0.]), np.array([1., 0., 0.])))
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.oriented_seed_heading',
                        lambda vol, pos, family, direction: np.asarray(direction))
    from vesuvius.neural_tracing.fiber_follow.evaluation.recovery import monitor_fixture
    from vesuvius.neural_tracing.fiber_follow.evaluation.recovery_fixtures import FIXTURE_STRATA
    arc = np.arange(300, dtype=float)
    fibers = [TracedFiber(str(i), np.c_[arc*0+i*10, arc*0, arc], arc, '') for i in range(3)]
    manifest = dict(sha256='frozen', monitor_fibers=[0], calibration_fibers=[1], final_fibers=[2],
                    monitor=[dict(fiber=0, t=150., sign=1)])
    cfg = config()
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, recent_history_points=cfg.n_history,
                          n_future=cfg.n_future)
    spec = FiberVolumeSpec('unused', ct_zarr='unused', inputs='ct')
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
    path = tmp_path/'monitor.npz'
    a, digest = monitor_fixture(path, fibers, manifest, sample, spec, 1)
    b, other = monitor_fixture(path, fibers, manifest, sample, spec, 1)
    assert digest == other and len(a) == 4 and set(a.fiber_idx) == {0}
    np.testing.assert_array_equal(a.hist, b.hist)
    assert all(lo <= d < hi for d, (lo, hi) in zip(a.match_distance, FIXTURE_STRATA))
    torch.testing.assert_close(torch_state, torch.get_rng_state())
    np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
    with pytest.raises(ValueError, match='settings changed'):
        monitor_fixture(path, fibers, manifest, replace(sample, excursion_amplitude=(3., 5.)), spec, 1)
    with pytest.raises(ValueError, match='monitor seeds'):
        monitor_fixture(path, fibers, dict(manifest, monitor=[dict(fiber=1, t=150., sign=1)]), sample, spec, 1)
