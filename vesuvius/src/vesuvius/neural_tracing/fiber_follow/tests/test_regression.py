"""Direct predictor: geometry gradients, identity recovery, observation and run contracts."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig, DirectFollower, feature_grid
from vesuvius.neural_tracing.fiber_follow.regression.supervision import geometry_mask, loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    optimizer_update, save_checkpoint, load_checkpoint, validate_volume_source, clip_training_gradients, prepare_training,
)
from vesuvius.neural_tracing.fiber_follow.regression.data import image_crop
from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset, SampleConfig, ZBand, training_state_allowed
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, crop_local_grid, sample_oriented_fast
from vesuvius.neural_tracing.fiber_follow.shared.runloop import training_rng_state, resume_training
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec


def config():
    return DirectConfig(fine=CropSpec(depth=16, width=12, behind=7),
                        channels=4, hidden=16, heads=2, layers=1, decoder_layers=1, activation_checkpointing=False, n_future=4, n_history=32, recurrent_refinement_steps=0)


def batch(cfg, b=2):
    hist = torch.zeros(b, cfg.n_history, 3)
    hist[..., 2] = -torch.arange(1, cfg.n_history+1)
    q = 4*(cfg.n_future-1)+1
    from slab_fixtures import slab_inputs
    return dict(x=slab_inputs(b) | {name: torch.rand(b, 2, crop.depth, crop.width, crop.width)
                   for name, crop in (('fine', cfg.fine),)},
                hist=hist, hmask=torch.ones(b, cfg.n_history), dense_ab=torch.ones(b, q, 2),
                dense_mask=torch.ones(b, q), offtrack=torch.zeros(b), endpoint_known=torch.zeros(b),
                end_local=torch.zeros(b, 3), source=torch.zeros(b))


def forward(m, b):
    return m(b['x'], b['hist'], b['hmask'])


def proposal_output(curves, hazards, selected=-1):
    """Build the current all-proposal output contract for loss/policy fixtures."""
    from vesuvius.neural_tracing.fiber_follow.regression.survival_confidence import survival_predictions
    logits, confidence = survival_predictions(hazards)
    return dict(points=curves[:, selected], initial_points=curves[:, 0],
                hazard_logits=hazards[:, selected], confidence_logits=logits[:, selected],
                confidence=confidence[:, selected], refinement_points=curves,
                refinement_hazard_logits=hazards, refinement_confidence_logits=logits,
                refinement_confidence=confidence,
                refinement_mask=torch.ones(curves.shape[:2], device=curves.device, dtype=torch.bool),
                selected_refinement=torch.full((len(curves),), selected % curves.shape[1], device=curves.device))


def test_prediction_is_deterministic_and_geometry_trains_actual_coordinates():
    torch.manual_seed(20)
    m = DirectFollower(config())
    b = batch(m.cfg)
    out = forward(m, b)
    assert out['points'].requires_grad
    torch.testing.assert_close(out['points'][..., 2], torch.arange(1, 5).float().expand(2, -1))
    torch.manual_seed(100)
    torch.testing.assert_close(out['points'], forward(m, b)['points'], rtol=0, atol=0)
    assert (out['confidence'][:, 1:] <= out['confidence'][:, :-1]).all()
    terms = loss_terms(out, b, m.cfg)
    terms['geometry_per_state'].mean().backward()
    for module in (m.coordinates, m.encoder.stem[0], m.encoder.compress, m.reference_token[0]):
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0


def test_masked_history_nan_and_no_history_are_safe_and_gt_is_not_an_input():
    m = DirectFollower(config())
    b = batch(m.cfg)
    b['hmask'][0] = 0
    b['hmask'][1, 12:] = 0
    first = forward(m, b)
    b['hist'][b['hmask'] == 0] = float('nan')
    b['gt_history'] = torch.full_like(b['hist'], float('nan'))
    other = forward(m, b)
    for key in first:
        assert torch.isfinite(other[key]).all()
        torch.testing.assert_close(first[key], other[key], rtol=0, atol=0)


def test_departures_unknown_ends_and_crop_censoring():
    m = DirectFollower(config())
    b = batch(m.cfg, 3)
    b['offtrack'][0] = 1
    b['dense_mask'][1] = 0
    b['dense_ab'][1] = float('nan')
    b['dense_ab'][2, 4, 0] = 100
    mask = geometry_mask(b, m.cfg)
    assert not mask[:2].any() and mask[2, :4].all() and not mask[2, 4:].any()
    terms = loss_terms(forward(m, b), b, m.cfg)
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




def test_microbatch_partition_keeps_objective_and_update():
    torch.manual_seed(3)
    cfg = config()
    a = DirectFollower(cfg)
    bmodel = copy.deepcopy(a)
    data = batch(cfg, 3)
    data['dense_mask'][0, 5:] = 0
    data['offtrack'][1] = 1
    data['source'] = torch.tensor([3, 0, 3])
    data['bank_tail_length'] = torch.tensor([16., 0., 128.])
    def take(value, sl):
        return {k: take(v, sl) for k, v in value.items()} if isinstance(value, dict) else value[sl]
    averages = [copy.deepcopy(m) for m in (a, bmodel)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=.001) for m in (a, bmodel)]
    prepare_training(a, batch_size=3, backend='eager')
    prepare_training(bmodel, batch_size=2, backend='eager')
    results = [optimizer_update(a, averages[0], optimizers[0], [data], 1, .001),
               optimizer_update(bmodel, averages[1], optimizers[1],
                                [take(data, slice(0, 1)), take(data, slice(1, 3))], 1, .001)]
    assert results[0]['loss'] == pytest.approx(results[1]['loss'], rel=2e-6)
    for result in results:
        assert result['bank_wrong_continuation_fraction'] == pytest.approx(2/3)
        assert result['bank_wrong_continuation_tail_mean'] == 72.
        assert result['bank_wrong_continuation_tail_min'] == 16.
        assert result['bank_wrong_continuation_tail_max'] == 128.
    for p, q in zip(a.parameters(), bmodel.parameters()):
        torch.testing.assert_close(p, q, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize('compiled', [False, True])
def test_memory_spike_cannot_scale_other_gradients(compiled):
    model = torch.nn.Module()
    model.history_encoder = torch.nn.Linear(2, 1, bias=False)
    model.encoder = torch.nn.Linear(2, 1, bias=False)
    memory, rest = model.history_encoder.weight, model.encoder.weight
    memory.grad = torch.tensor([[3e6, 4e6]])
    rest.grad = torch.tensor([[3., 4.]])
    if compiled:
        model = torch.compile(model, backend='eager')
    metrics = clip_training_gradients(model, 5., 20.)
    torch.testing.assert_close(memory.grad, torch.tensor([[3., 4.]]))
    torch.testing.assert_close(rest.grad, torch.tensor([[3., 4.]]), rtol=0, atol=0)
    assert metrics['history_grad_norm'] == pytest.approx(5e6)
    assert metrics['rest_grad_norm'] == 5.
    assert metrics['history_grad_clip_scale'] == pytest.approx(1e-6)
    assert metrics['rest_grad_clip_scale'] == 1.


def test_clipping_can_be_disabled_independently():
    model = torch.nn.Module()
    model.history_encoder = torch.nn.Linear(2, 1, bias=False)
    model.encoder = torch.nn.Linear(2, 1, bias=False)
    for memory_cap, rest_cap in [(0., 20.), (5., 0.)]:
        model.history_encoder.weight.grad = torch.tensor([[30., 40.]])
        model.encoder.weight.grad = torch.tensor([[30., 40.]])
        result = clip_training_gradients(model, memory_cap, rest_cap)
        assert model.history_encoder.weight.grad.norm().item() == pytest.approx(memory_cap or 50.)
        assert model.encoder.weight.grad.norm().item() == pytest.approx(rest_cap or 50.)
        assert result['history_grad_norm'] == result['rest_grad_norm'] == 50.


@pytest.mark.parametrize('bad_group', ['history_encoder', 'encoder'])
@pytest.mark.parametrize('cap', [0., 5.])
def test_nonfinite_gradients_raise_before_clipping_either_group(bad_group, cap):
    model = torch.nn.Module()
    model.history_encoder = torch.nn.Linear(2, 1, bias=False)
    model.encoder = torch.nn.Linear(2, 1, bias=False)
    for p in model.parameters():
        p.grad = torch.full_like(p, 100.)
    getattr(model, bad_group).weight.grad.fill_(float('inf'))
    with pytest.raises(RuntimeError, match='non-finite'):
        clip_training_gradients(model, cap, cap)
    for name in ('history_encoder', 'encoder'):
        if name != bad_group:
            assert (getattr(model, name).weight.grad == 100.).all()


@pytest.mark.parametrize('bad', [-1., float('nan'), float('inf')])
def test_invalid_gradient_clip_limits_are_rejected(bad):
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match='finite and nonnegative'):
        clip_training_gradients(model, bad, 20.)
    with pytest.raises(ValueError, match='finite and nonnegative'):
        clip_training_gradients(model, 5., bad)


def test_nonfinite_microbatch_loss_raises_before_any_update():
    """Loss sums are read once per update; a nonfinite microbatch still blocks the step."""
    torch.manual_seed(4)
    cfg = config()
    model = DirectFollower(cfg)
    data = batch(cfg, 2)
    def take(value, sl):
        return {k: take(v, sl) for k, v in value.items()} if isinstance(value, dict) else value[sl]
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
    from vesuvius.neural_tracing.fiber_follow.regression.train import prepare_training
    monkeypatch.setattr(torch._inductor.config, 'emulate_precision_casts', False)
    compiled = []
    monkeypatch.setattr(torch, 'compile', lambda module, **kwargs: compiled.append((module, kwargs)) or module)
    model = DirectFollower(config())
    parameters, keys = list(model.parameters()), list(model.state_dict())
    assert prepare_training(model) is model
    assert not hasattr(model, '_orig_mod')
    assert [fn.__name__ for fn, _ in compiled] == [
        'training_forward', 'score_candidates', 'loss_terms']
    assert all(options == dict(dynamic=False, fullgraph=True, options=dict(emulate_precision_casts=True))
               for _, options in compiled)
    assert list(model.parameters()) == parameters
    assert list(model.state_dict()) == keys
    assert prepare_training(model) is model
    assert len(compiled) == 3  # Setup is idempotent.
    assert not torch._inductor.config.emulate_precision_casts


@pytest.mark.parametrize('encoder,token_only', [('conv',False), ('patch4',False), ('patch4',True)])
def test_checkpoint_roundtrip_and_resume_optimizer_rng(tmp_path, encoder, token_only):
    torch.manual_seed(7)
    cfg = replace(config(), encoder=encoder, token_only=token_only)
    m = DirectFollower(cfg)
    ema = copy.deepcopy(m)
    opt = torch.optim.AdamW(m.parameters(), lr=.001)
    data = batch(cfg)
    prepare_training(m, backend='eager')
    optimizer_update(m, ema, opt, [data], 1, .001)
    spec = FiberVolumeSpec('unused', ct_zarr='unused', ct_level=0, ct_grid_scale=4., inputs='ct+presence')
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history)
    path = tmp_path/'last.pt'
    save_checkpoint(path, m, ema, spec, sample, dict(step=1, optimizer=opt.state_dict(), rng=training_rng_state()))
    expected_rng = torch.rand(3)
    loaded, crop, nh, loaded_spec, ck = load_checkpoint(path, 'cpu')
    torch.testing.assert_close(forward(loaded, data)['points'], forward(ema, data)['points'], rtol=0, atol=0)
    assert crop == cfg.fine and nh == cfg.n_history and loaded_spec == spec
    restored = DirectFollower(cfg)
    restored_ema = copy.deepcopy(restored)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=.001)
    assert resume_training(ck, restored, restored_ema, restored_opt)[0] == 1
    prepare_training(restored, backend='eager')
    torch.testing.assert_close(torch.rand(3), expected_rng, rtol=0, atol=0)
    optimizer_update(m, ema, opt, [data], 2, .001)
    optimizer_update(restored, restored_ema, restored_opt, [data], 2, .001)
    for p, q in zip(m.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)




def test_image_sampler_matches_reference_at_fine_and_coarse_resolution(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.shared import crop_sampling as module
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 256, (1, 1, 40, 40, 40), dtype=np.uint8)
    starts = np.zeros((1, 3), np.int64)
    monkeypatch.setattr(module, 'read_tight_blocks', lambda *a, **kw: (raw, starts))
    items = [dict(pos=np.array([8., 8., 8.]), frame=np.eye(3))]
    crop = CropSpec(depth=12, width=10, behind=4, spacing=.5)
    from vesuvius.neural_tracing.fiber_follow.shared.ct_normalization import normalize_ct
    # Avoid a hard-mask boundary at exactly representable native intensity 62:
    # the two interpolation backends differ slightly in floating-point rounding.
    calibration = dict(threshold=62.123, noise=4.)
    vol = SimpleNamespace(input_scale=2.,presence=object(), spec=SimpleNamespace(ct_normalization=calibration))
    image = image_crop(items, vol, crop)
    grid = torch.from_numpy(crop_local_grid(crop)).float()
    for channel, scale in enumerate((2., 1.)):
        expected = sample_oriented_fast(torch.from_numpy(raw), torch.from_numpy(starts),
            torch.tensor([[8., 8., 8.]])*scale, torch.eye(3)[None]*scale, grid)
        if channel == 0:
            normalize_ct(expected.numpy()[0, 0], calibration)
        torch.testing.assert_close(image[:, channel:channel+1], expected, atol=3e-4, rtol=1e-3)


def test_manifest_allows_resolution_change_but_not_source_change():
    original = FiberVolumeSpec('fiber', ct_zarr='ct', inputs='ct+presence')
    manifest = dict(volume=original.to_dict())
    fine = replace(original, ct_level=0, ct_grid_scale=4.)
    validate_volume_source(fine, manifest)
    with pytest.raises(ValueError, match='ct_zarr'):
        validate_volume_source(replace(fine, ct_zarr='different'), manifest)


@pytest.mark.slow
def test_parallel_fibers_learn_recovery_from_older_observed_history():
    torch.manual_seed(91)
    cfg = config()
    m = DirectFollower(cfg)
    b = batch(cfg)
    sign = torch.tensor([-1., 1.])
    for name, crop in (('fine', cfg.fine),):
        axis = torch.tensor(crop.lateral_coords)
        image = torch.exp(-((axis-2)/.6)**2)+torch.exp(-((axis+2)/.6)**2)
        b['x'][name] = image.float()[None, None, None, None].expand(2, 2, crop.depth, crop.width, -1).clone()
    b['hist'][..., 0] = sign[:, None]*2*torch.linspace(0, 1, cfg.n_history)[None]
    b['dense_ab'].zero_()
    b['dense_ab'][..., 0] = sign[:, None]*2
    opt = torch.optim.AdamW(m.parameters(), lr=.003)
    with torch.no_grad():
        before = (forward(m, b)['points'][..., 0]-sign[:, None]*2).square().mean().item()
    for _ in range(140):
        opt.zero_grad(set_to_none=True)
        terms = loss_terms(forward(m, b), b, cfg)
        terms['geometry_per_state'].mean().backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
        opt.step()
    with torch.no_grad():
        prediction = forward(m, b)['points']
        error = (prediction[..., :2]-torch.stack((sign*2, torch.zeros(2)), -1)[:, None]).square().mean().item()
    assert error < .15 and error < before*.1, (before, error)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_bf16_geometry_gradients():
    m = DirectFollower(config()).cuda()
    from vesuvius.neural_tracing.fiber_follow.regression.train import move_batch
    b = move_batch(batch(m.cfg), 'cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        terms = loss_terms(forward(m, b), b, m.cfg)
        loss = terms['geometry_per_state'].mean()+terms['confidence_per_state'].mean()
    loss.backward()
    assert torch.isfinite(loss) and m.coordinates.weight.grad.abs().sum() > 0




def test_commit_window_loss_and_auxiliary_proposal_supervision():
    import torch.nn.functional as F
    cfg = replace(config(), recurrent_refinement_steps=1)
    b = batch(cfg, 1)
    b['dense_ab'].zero_()
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


def test_every_refinement_receives_auxiliary_geometry_supervision():
    cfg = replace(config(), recurrent_refinement_steps=2)
    b = batch(cfg, 1)
    b['dense_ab'].zero_()
    curves = []
    for error in (3., 2., 1.):
        curve = torch.tensor([[[error, 0., float(z)] for z in range(1, 5)]], requires_grad=True)
        curves.append(curve)
    # Selecting an earlier path must not deprive the final proposal of its loss.
    out = proposal_output(torch.stack(curves, 1), torch.zeros(1, 3, 4), selected=0)
    terms = loss_terms(out, b, cfg)
    # Smooth L1 averaged over x,y; first two stages share the 25% auxiliary term.
    expected = .75*.25+.25*((3-.5)/2+(2-.5)/2)/2
    torch.testing.assert_close(terms['geometry_per_state'], torch.tensor([expected]))
    terms['geometry_per_state'].sum().backward()
    assert all(curve.grad[..., 0].abs().sum() > 0 for curve in curves)


def test_startup_sampling_and_history_diagnostics():
    from vesuvius.neural_tracing.fiber_follow.shared.data import TracedFiber, make_sample
    from vesuvius.neural_tracing.fiber_follow.regression.diagnostics import decision_rows, summarize_decisions
    arc = np.arange(400, dtype=float)
    fiber = TracedFiber('line', np.c_[arc*0, arc*0, arc], arc, '')
    sample = SampleConfig(no_history_prob=0., short_history_prob=1.)
    rng = np.random.default_rng(17)
    counts = [int(make_sample(fiber, 200., False, sample, rng)['hmask'].sum()) for _ in range(200)]
    assert all(1 <= n <= 32 for n in counts)
    assert 70 < sum(n <= 8 for n in counts) < 130
    assert make_sample(fiber, 200., False, replace(sample, no_history_prob=1.), rng)['hmask'].sum() == 0
    # Grouping measures what the model actually receives, including replay.
    cfg = replace(config(), n_history=64)
    b = batch(cfg, 4)
    for row, length in zip(b['hmask'], (0, 4, 16, 64)):
        row[length:] = 0
    m = DirectFollower(cfg)
    stats = summarize_decisions(decision_rows(forward(m, b), b, cfg), 4)
    assert {k: v['states'] for k, v in stats['by_history'].items()} == {'0': 1, '1-8': 1, '9-32': 1, '>32': 1}
    assert all(0 <= v['first_confidence_mean'] <= 1 for v in stats['by_history'].values())




def test_decision_metrics_score_actual_commits_censor_unknowns_and_pool_counts():
    import json
    from vesuvius.neural_tracing.fiber_follow.regression.diagnostics import decision_rows, summarize_decisions
    cfg = config()
    b = batch(cfg, 5)
    b['dense_ab'].zero_()
    b['gt_history'] = torch.zeros(5, 1, 3)
    b['gt_history'][:, 0, 0] = torch.tensor([.5, 1.25, 1.75, 2.5, 0.])
    b['gt_history_mask'] = torch.ones(5, 1)
    b['offtrack'][4] = 1
    b['gt_history_mask'][4] = 0
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
    rows = decision_rows(out, b, cfg, n_commit=4)
    stats = summarize_decisions(rows, 4)
    all_stats = stats['by_drift']['all']
    assert all_stats['first_known'] == 4 and all_stats['first_correct'] == 2
    assert all_stats['commit_correct'] == 1
    assert all_stats['gate_0.5'] == dict(false_stops=1, accepted_known=3, accepted_wrong=2,
                                      accepted_unknown=1, departed_continues=1)
    assert stats['by_drift']['1-1.5']['gate_0.5']['accepted_wrong'] == 0
    assert stats['by_drift']['>=3.5']['final_error_mean'] is None
    assert all_stats['final_error_mean'] == pytest.approx(all_stats['final_error_sum']/all_stats['final_error_count'])
    json.dumps(stats, allow_nan=False)
    points[0, 0, 0] = float('nan')
    bad = summarize_decisions(decision_rows(out, b, cfg, n_commit=4), 4)['by_drift']['all']
    assert bad['recovery_blocked'] == 1 and bad['final_nonfinite_count'] > 0
    json.dumps(bad, allow_nan=False)


def test_monitor_fixtures_are_fixed_private_rng_and_exclude_other_splits(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.shared.data import TracedFiber
    from vesuvius.neural_tracing.fiber_follow.regression.recovery import monitor_fixture
    arc = np.arange(300, dtype=float)
    fibers = [TracedFiber(str(i), np.c_[arc*0+i*10, arc*0, arc], arc, '') for i in range(3)]
    manifest = dict(sha256='frozen', monitor_fibers=[0], calibration_fibers=[1], final_fibers=[2],
                    monitor=[dict(fiber=0, t=150., sign=1)])
    cfg = config()
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, recent_history_points=cfg.n_history,
                          n_future=cfg.n_future)
    spec = FiberVolumeSpec('unused', ct_zarr='unused', inputs='ct+presence')
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
    path = tmp_path/'monitor.npz'
    a, digest = monitor_fixture(path, fibers, manifest, sample, spec, 1)
    b, other = monitor_fixture(path, fibers, manifest, sample, spec, 1)
    assert digest == other and len(a) == 4 and set(a.fiber_idx) == {0}
    np.testing.assert_array_equal(a.hist, b.hist)
    assert all(lo <= d < hi for d, (lo, hi) in zip(a.drift, [(0, 1), (1, 1.5), (1.5, 2), (2, 3.5)]))
    torch.testing.assert_close(torch_state, torch.get_rng_state())
    np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
    with pytest.raises(ValueError, match='settings changed'):
        monitor_fixture(path, fibers, manifest, replace(sample, history_drift=3.), spec, 1)
    with pytest.raises(ValueError, match='monitor seeds'):
        monitor_fixture(path, fibers, dict(manifest, monitor=[dict(fiber=1, t=150., sign=1)]), sample, spec, 1)


def test_diagnostic_logging_preserves_training_update_and_rng():
    torch.manual_seed(2)
    a = DirectFollower(config())
    bmodel = copy.deepcopy(a)
    data = batch(a.cfg)
    results = []
    rng = torch.get_rng_state()
    for model, enabled in ((a, False), (bmodel, True)):
        torch.set_rng_state(rng)
        opt = torch.optim.SGD(model.parameters(), lr=.001)
        ema = copy.deepcopy(model)
        prepare_training(model, backend='eager')
        results.append(optimizer_update(model, ema, opt, [data], 1, .001,
                                        n_commit=2, compute_metrics=enabled))
        torch.testing.assert_close(torch.get_rng_state(), rng)
    assert 'decisions' not in results[0] and results[1]['decisions']['n_commit'] == 2
    assert results[0]['loss'] == results[1]['loss']
    for p, q in zip(a.parameters(), bmodel.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_slab_recovery_evaluator_preserves_float_inputs_and_observed_states():
    from vesuvius.neural_tracing.fiber_follow.shared.data import TracedFiber
    from vesuvius.neural_tracing.fiber_follow.shared.recovery import make_recovery_states, evaluate_recovery_states
    cfg = config()
    arc = np.arange(300, dtype=float)
    fiber = TracedFiber('line', np.c_[arc*0, arc*0, arc], arc, '')
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, recent_history_points=cfg.n_history,
                          n_future=cfg.n_future)
    states = make_recovery_states([fiber], [dict(fiber=0, t=150., sign=1)], sample, dict(split='monitor'))
    inputs = batch(cfg, 1)
    inputs['x'] = {k: v.half() if v.is_floating_point() else v for k, v in inputs['x'].items()}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
        def forward(self, x, hist, hmask):
            def check(value):
                if isinstance(value, dict):
                    for v in value.values():
                        check(v)
                else:
                    assert value.dtype == (torch.float32 if value.is_floating_point() else torch.bool)
            check(x)
            assert x['history_valid'].dtype == torch.bool
            assert x['history_slabs'].dtype == torch.float32
            p = torch.zeros(1, cfg.n_future, 3)
            p[..., 2] = torch.arange(1, cfg.n_future+1)
            return dict(points=p, confidence=torch.ones(1, cfg.n_future))
    traced = []
    closed = []
    class Tracer:
        def __init__(self, *args, **kwargs):
            pass
        def trace(self, pos, heading, initial_states):
            traced.append(initial_states[0])
            return [np.stack((pos[0], pos[0]+[0, 0, 1]))], ['max_len']
        def close(self):
            closed.append(True)
    audited = []
    rows, predictions = evaluate_recovery_states(Model(), None, states, [fiber], sample,
        batch_builder=lambda items, vol: inputs, tracer_class=Tracer, n_commit=4, limit=1,
        thresholds=(.5,), on_prediction=lambda out, b: audited.append(out))
    assert predictions.shape == (1, 4, 3) and len(rows) == len(audited) == len(closed) == 1
    for key in ('hist', 'hmask', 'frame'):
        np.testing.assert_array_equal(traced[0][key], getattr(states, key)[0])
    np.testing.assert_array_equal(traced[0]['observed_path'], states.observed_prefix(0))


def test_tight_blocks_match_rotation_invariant_blocks_at_array_edges():
    """Per-item minimal blocks read the same values, including zero fill beyond the array."""
    from vesuvius.neural_tracing.fiber_follow.shared.data import _grid_flat, read_blocks
    from vesuvius.neural_tracing.fiber_follow.shared.fast_sample import sample_crop
    rng = np.random.default_rng(11)

    class Array:
        def __init__(self, shape):
            self.data = rng.integers(1, 256, shape, dtype=np.uint8)

        def read(self, start, size):
            out = np.zeros(tuple(size), np.uint8)
            lo, hi = np.maximum(start, 0), np.minimum(start+size, self.data.shape)
            if np.all(hi > lo):
                out[tuple(slice(a-s, b-s) for a, b, s in zip(lo, hi, start))] = self.data[tuple(map(slice, lo, hi))]
            return out

    ct, presence = Array((70, 60, 64)), Array((35, 30, 32))
    from vesuvius.neural_tracing.fiber_follow.shared.ct_normalization import normalize_ct
    calibration = dict(threshold=62., noise=4.)
    vol = SimpleNamespace(presence=presence, input_scale=2., raw_block=lambda s, z: ct.read(s, z)[None],
                          spec=SimpleNamespace(ct_normalization=calibration))
    crop = CropSpec(depth=14, width=10, behind=5, spacing=.5)

    def reference(items):
        grid, empty, mask = _grid_flat(crop), np.empty((0, 3), np.float32), np.empty(0, np.float32)
        out = np.empty((len(items), 2, crop.depth, crop.width, crop.width), np.float32)
        for use_presence in (False, True):
            raw, starts = read_blocks(items, vol, crop, presence=use_presence)
            scale = 1. if use_presence else vol.input_scale
            for j, item in enumerate(items):
                out[j, int(use_presence)] = sample_crop(raw[j], starts[j], item['pos']*scale, item['frame']*scale, grid,
                    empty, mask, 2, 1., 'points')[0].reshape(crop.depth, crop.width, crop.width)
        return torch.from_numpy(out)

    items = []
    for _ in range(24):
        # Interior, straddling each face, and fully outside; arbitrary orientation.
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        pos = rng.uniform(-6, 38, 3)
        items.append(dict(pos=pos, frame=q*np.sign(np.linalg.det(q))))
    image = image_crop(items, vol, crop)
    expected = reference(items)
    for row in expected:
        normalize_ct(row[0].numpy(), calibration)
    assert torch.equal(image, expected)
    assert 0 < float((expected != 0).float().mean()) < 1


def test_fresh_optimizer_resume_preserves_weights_and_future_resume_schedule():
    from vesuvius.neural_tracing.fiber_follow.regression.train import initialize_training_optimizer
    from vesuvius.neural_tracing.fiber_follow.shared.runloop import lr_at
    args = SimpleNamespace(reset_optimizer=False, lr=.0003)
    model = DirectFollower(config())
    ema = copy.deepcopy(model)
    opt, _, _ = initialize_training_optimizer(model, ema, args)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    ck = dict(model=copy.deepcopy(model.state_dict()), ema=copy.deepcopy(ema.state_dict()),
              optimizer=opt.state_dict(), rng=training_rng_state(), step=3000)
    assert len(ck['optimizer']['param_groups']) == 1 and ck['optimizer']['state']
    expected_rng = torch.rand(3)
    reset_model = DirectFollower(config()).requires_grad_(False)
    reset_ema = copy.deepcopy(reset_model)
    args.reset_optimizer = True
    fresh, done, origin = initialize_training_optimizer(reset_model, reset_ema, args, ck)
    assert done == origin == 3000
    assert len(fresh.param_groups) == 1 and not fresh.state
    assert all(p.requires_grad for p in reset_model.parameters())
    torch.testing.assert_close(torch.rand(3), expected_rng, rtol=0, atol=0)
    for actual, saved in ((reset_model, ck['model']), (reset_ema, ck['ema'])):
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    fresh.param_groups[0]['lr'] = lr_at(done+1-origin, args.lr, 500, 100000-origin)
    assert fresh.param_groups[0]['lr'] == pytest.approx(.0003/500)
    for p in reset_model.parameters():
        p.grad = torch.ones_like(p)
    fresh.step()
    after = dict(model=reset_model.state_dict(), ema=reset_ema.state_dict(),
                 optimizer=fresh.state_dict(), rng=training_rng_state(), step=3001,
                 lr_restart_step=origin)
    args.reset_optimizer = False
    restored = DirectFollower(config())
    restored_opt, done, origin = initialize_training_optimizer(restored, copy.deepcopy(restored), args, after)
    assert done == 3001 and origin == 3000 and len(restored_opt.param_groups) == 1
    assert len(restored_opt.state) == len(fresh.state)
    assert lr_at(done+1-origin, args.lr, 500, 100000-origin) == pytest.approx(.0003*2/500)






@pytest.mark.parametrize('option', [['--memory-version', '5'], ['--proposal-step', '.5'],
                                  ['--proposal-warmup-steps', '500']])
def test_removed_model_cli_options_rejected(option):
    from vesuvius.neural_tracing.fiber_follow.regression.train import build_parser
    required = ['--name', 'test', '--fiber-zarrs', '/tmp/presence', '--fibers', '/tmp/fibers',
                '--ct', '/tmp/ct', '--manifest', '/tmp/seeds.json']
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(required+option)
    assert error.value.code == 2
