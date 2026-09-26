"""Direct predictor: geometry gradients, identity recovery, observation and run contracts."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.direct.model import DirectConfig, DirectFollower, feature_grid
from vesuvius.neural_tracing.fiber_follow.direct.supervision import geometry_mask, loss_terms
from vesuvius.neural_tracing.fiber_follow.direct.train import (
    optimizer_update, save_checkpoint, load_checkpoint, validate_volume_source,
)
from vesuvius.neural_tracing.fiber_follow.direct.data import image_crop, DirectTracer
from vesuvius.neural_tracing.fiber_follow.data import FollowDataset, SampleConfig, ZBand, training_state_allowed
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec, crop_local_grid, sample_oriented_fast
from vesuvius.neural_tracing.fiber_follow.runloop import training_rng_state, resume_training
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolumeSpec


def config():
    return DirectConfig(fine=CropSpec(depth=16, width=12, behind=7),
                        coarse=CropSpec(depth=24, width=12, behind=18, spacing=2.),
                        channels=4, hidden=16, heads=2, layers=1, n_future=4, n_history=32)


def batch(cfg, b=2):
    hist = torch.zeros(b, cfg.n_history, 3)
    hist[..., 2] = -torch.arange(1, cfg.n_history+1)
    q = 4*(cfg.n_future-1)+1
    return dict(x={name: torch.rand(b, 2, crop.depth, crop.width, crop.width)
                   for name, crop in (('fine', cfg.fine), ('coarse', cfg.coarse))},
                hist=hist, hmask=torch.ones(b, cfg.n_history), dense_ab=torch.ones(b, q, 2),
                dense_mask=torch.ones(b, q), offtrack=torch.zeros(b), endpoint_known=torch.zeros(b),
                end_local=torch.zeros(b, 3), source=torch.zeros(b))


def forward(m, b):
    return m(b['x'], b['hist'], b['hmask'])


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
    for module in (m.coordinates, m.fine_encoder.local[0], m.coarse_encoder.local[0], m.history_token[0]):
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0
    assert m.confidence_head[-1].weight.grad is None
    m.zero_grad(set_to_none=True)
    loss_terms(forward(m, b), b, m.cfg)['confidence_per_state'].mean().backward()
    assert m.confidence_head[-1].weight.grad.abs().sum() > 0
    assert m.coordinates.weight.grad is None


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
    assert torch.isfinite(terms['geometry_sum'])


def test_feature_coordinates_respect_even_sized_strided_lattice():
    crop = CropSpec(depth=16, width=12, behind=7, spacing=.5)
    # Deep convolution lattice index (x,y,z) = (2,1,3), stride four.
    p = torch.tensor([[[8*.5-5.5*.5, 4*.5-5.5*.5, 12*.5-7*.5]]])
    grid = feature_grid(p, crop, (4, 3, 3), stride=4)
    torch.testing.assert_close(grid, torch.tensor([[[1., 0., 1.]]]))


def test_path_evidence_samples_longitudinal_slices_and_both_deep_lattices():
    cfg = config()
    model = DirectFollower(cfg)
    def z_features(crop, channels, stride):
        shape = tuple((size+stride-1)//stride for size in (crop.depth, crop.width, crop.width))
        z = (torch.arange(shape[0])*stride-crop.behind)*crop.spacing
        return z[None, None, :, None, None].expand(1, channels, *shape).contiguous()
    fine = z_features(cfg.fine, cfg.channels, 1)
    deep = z_features(cfg.fine, 4*cfg.channels, 4)
    coarse = z_features(cfg.coarse, 4*cfg.channels, 4)
    points = torch.tensor([[[0., 0., 2.], [100., 0., 2.]]])
    features = model.path_features(fine, deep, coarse, points)
    local_width = 27*(cfg.channels+1)
    local = features[0, 0, :local_width].reshape(3, 9, cfg.channels+1)
    torch.testing.assert_close(local[..., :-1], torch.tensor([1., 2., 3.])[:, None, None].expand(3, 9, cfg.channels))
    assert local[..., -1].eq(1).all()
    for chunk in features[0, 0, local_width:].reshape(2, -1):
        torch.testing.assert_close(chunk[:-1], torch.full((4*cfg.channels,), 2.))
        assert chunk[-1] == 1
    assert features[0, 1].eq(0).all()  # Outside both crops, including support flags.


def test_microbatch_partition_keeps_objective_and_update():
    torch.manual_seed(3)
    cfg = config()
    a = DirectFollower(cfg)
    bmodel = copy.deepcopy(a)
    data = batch(cfg, 3)
    data['dense_mask'][0, 5:] = 0
    data['offtrack'][1] = 1
    def take(value, sl):
        return {k: take(v, sl) for k, v in value.items()} if isinstance(value, dict) else value[sl]
    averages = [copy.deepcopy(m) for m in (a, bmodel)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=.001) for m in (a, bmodel)]
    results = [optimizer_update(a, averages[0], optimizers[0], [data], 1, .001),
               optimizer_update(bmodel, averages[1], optimizers[1],
                                [take(data, slice(0, 1)), take(data, slice(1, 3))], 1, .001)]
    assert results[0]['loss'] == pytest.approx(results[1]['loss'], rel=2e-6)
    for p, q in zip(a.parameters(), bmodel.parameters()):
        torch.testing.assert_close(p, q, rtol=2e-5, atol=2e-7)


def test_checkpoint_roundtrip_and_resume_optimizer_rng(tmp_path):
    torch.manual_seed(7)
    cfg = config()
    m = DirectFollower(cfg)
    ema = copy.deepcopy(m)
    opt = torch.optim.AdamW(m.parameters(), lr=.001)
    data = batch(cfg)
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
    torch.testing.assert_close(torch.rand(3), expected_rng, rtol=0, atol=0)
    optimizer_update(m, ema, opt, [data], 2, .001)
    optimizer_update(restored, restored_ema, restored_opt, [data], 2, .001)
    for p, q in zip(m.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_both_crops_are_excluded_from_holdout():
    cfg = DirectConfig()
    ds = object.__new__(FollowDataset)
    ds.cfg = SampleConfig(crop=cfg.fine)
    ds.additional_crops = (cfg.coarse,)
    ds.exclude = ZBand(100., 120.)
    item = dict(pos=np.array([0., 0., 49.]), frame=np.diag([1., -1., -1.]))
    assert training_state_allowed(item, cfg.fine, ds.exclude)
    assert not ds.state_allowed(item)


def test_image_sampler_matches_reference_at_fine_and_coarse_resolution(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.direct import data as module
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 256, (1, 1, 40, 40, 40), dtype=np.uint8)
    starts = np.zeros((1, 3), np.int64)
    monkeypatch.setattr(module, 'read_tight_blocks', lambda *a, **kw: (raw, starts))
    items = [dict(pos=np.array([8., 8., 8.]), frame=np.eye(3))]
    crop = CropSpec(depth=12, width=10, behind=4, spacing=.5)
    vol = SimpleNamespace(input_scale=2.)
    image = image_crop(items, vol, crop)
    grid = torch.from_numpy(crop_local_grid(crop)).float()
    for channel, scale in enumerate((2., 1.)):
        expected = sample_oriented_fast(torch.from_numpy(raw), torch.from_numpy(starts),
            torch.tensor([[8., 8., 8.]])*scale, torch.eye(3)[None]*scale, grid)
        torch.testing.assert_close(image[:, channel:channel+1], expected, atol=3e-4, rtol=1e-3)


def test_manifest_allows_resolution_change_but_not_source_change():
    original = FiberVolumeSpec('fiber', ct_zarr='ct', inputs='ct+presence')
    manifest = dict(volume=original.to_dict())
    fine = replace(original, ct_level=0, ct_grid_scale=4.)
    validate_volume_source(fine, manifest)
    with pytest.raises(ValueError, match='ct_zarr'):
        validate_volume_source(replace(fine, ct_zarr='different'), manifest)


def test_parallel_fibers_learn_recovery_from_older_observed_history():
    torch.manual_seed(91)
    cfg = config()
    m = DirectFollower(cfg)
    b = batch(cfg)
    sign = torch.tensor([-1., 1.])
    for name, crop in (('fine', cfg.fine), ('coarse', cfg.coarse)):
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
    from vesuvius.neural_tracing.fiber_follow.direct.train import move_batch
    b = move_batch(batch(m.cfg), 'cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        terms = loss_terms(forward(m, b), b, m.cfg)
        loss = terms['geometry_per_state'].mean()+terms['confidence_per_state'].mean()
    loss.backward()
    assert torch.isfinite(loss) and m.coordinates.weight.grad.abs().sum() > 0


def test_local_correction_is_bounded_and_confidence_cannot_train_its_coordinates():
    torch.manual_seed(123)
    cfg = replace(config(), correction_limit=.4)
    m = DirectFollower(cfg)
    b = batch(cfg)
    seen = []
    original = m.path_features
    def patches(fine, fine_deep, coarse_deep, points):
        seen.append(points.detach().clone())
        return original(fine, fine_deep, coarse_deep, points)
    m.path_features = patches
    encodings = []
    handles = [encoder.register_forward_hook(lambda *args: encodings.append(1))
               for encoder in (m.fine_encoder, m.coarse_encoder)]
    out = forward(m, b)
    for handle in handles:
        handle.remove()
    assert len(encodings) == 2  # Each crop encoded once despite two corrections.
    assert len(seen) == cfg.correction_steps+1
    assert out['refinement_points'].shape == (2, 3, 4, 3)
    for i, coordinates in enumerate(seen):
        torch.testing.assert_close(coordinates, out['refinement_points'][:, i])
    delta = out['refinement_points'][:, 1:]-out['refinement_points'][:, :-1]
    assert delta[..., :2].norm(dim=-1).max() <= cfg.correction_limit+1e-6
    assert delta[..., 2].count_nonzero() == 0
    assert out['points'][..., :2].abs().max() <= cfg.lateral_limit
    loss_terms(out, b, cfg, n_commit=2)['geometry_per_state'].mean().backward()
    assert m.correction_head[-1].weight.grad.abs().sum() > 0
    assert m.coordinates.weight.grad.abs().sum() > 0
    assert m.fine_encoder.local[0].weight.grad.abs().sum() > 0
    m.zero_grad(set_to_none=True)
    loss_terms(forward(m, b), b, cfg, n_commit=2)['confidence_per_state'].mean().backward()
    assert m.correction_head[-1].weight.grad is None
    assert m.coordinates.weight.grad is None


def test_commit_window_loss_and_auxiliary_proposal_supervision():
    import torch.nn.functional as F
    cfg = config()
    b = batch(cfg, 1)
    b['dense_ab'].zero_()
    b['dense_mask'][:, -2:] = 0
    points = torch.tensor([[[1., 0., 1.], [1., 0., 2.], [3., 0., 3.], [3., 0., 4.]]], requires_grad=True)
    initial = (points.detach()*torch.tensor([2., 1., 1.])).requires_grad_()
    logits = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
    out = dict(points=points, initial_points=initial, confidence_logits=logits)
    result = loss_terms(out, b, cfg, n_commit=2)
    def expected(curve):
        dense = F.interpolate(curve[..., :2].transpose(1, 2), size=13, mode='linear', align_corners=True).transpose(1, 2)
        errors = F.smooth_l1_loss(dense, b['dense_ab'], reduction='none').mean(-1)
        return .5*errors[:, :5].mean()+.5*errors[:, :11].mean()
    torch.testing.assert_close(result['geometry_per_state'][0], .75*expected(points)+.25*expected(initial))
    from vesuvius.neural_tracing.fiber_follow.supervision import prefix_labels
    labels, known, _ = prefix_labels(points, b)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    expected_conf = .5*(bce[:, :2]*known[:, :2]).sum()/known[:, :2].sum()+.5*(bce*known).sum()/known.sum()
    torch.testing.assert_close(result['confidence_per_state'][0], expected_conf)
    result['geometry_per_state'].sum().backward()
    assert points.grad.abs().sum() > 0 and initial.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='Commit window'):
        loss_terms(out, b, cfg, n_commit=5)


def test_every_refinement_receives_auxiliary_geometry_supervision():
    cfg = config()
    b = batch(cfg, 1)
    b['dense_ab'].zero_()
    curves = []
    for error in (3., 2., 1.):
        curve = torch.tensor([[[error, 0., float(z)] for z in range(1, 5)]], requires_grad=True)
        curves.append(curve)
    out = dict(points=curves[-1], initial_points=curves[0], refinement_points=torch.stack(curves, 1),
               confidence_logits=torch.zeros(1, 4))
    terms = loss_terms(out, b, cfg)
    # Smooth L1 averaged over x,y; first two stages share the 25% auxiliary term.
    expected = .75*.25+.25*((3-.5)/2+(2-.5)/2)/2
    torch.testing.assert_close(terms['geometry_per_state'], torch.tensor([expected]))
    terms['geometry_per_state'].sum().backward()
    assert all(curve.grad[..., 0].abs().sum() > 0 for curve in curves)


def test_startup_sampling_and_history_diagnostics():
    from vesuvius.neural_tracing.fiber_follow.data import TracedFiber, make_sample
    from vesuvius.neural_tracing.fiber_follow.direct.diagnostics import decision_rows, summarize_decisions
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


@pytest.mark.parametrize('correction', [False, True])
def test_legacy_checkpoint_inference(tmp_path, correction):
    cfg = replace(config(), correction=correction, correction_steps=1, rich_path_context=False)
    m = DirectFollower(cfg).eval()
    b = batch(cfg)
    assert hasattr(m, 'correction_head') == correction
    out = forward(m, b)
    if not correction:
        torch.testing.assert_close(out['initial_points'], out['points'], rtol=0, atol=0)
    spec = FiberVolumeSpec('unused', ct_zarr='unused', ct_level=0, ct_grid_scale=4., inputs='ct+presence')
    path = tmp_path/'legacy.pt'
    save_checkpoint(path, m, m, spec, SampleConfig(crop=cfg.fine, n_history=cfg.n_history))
    ck = torch.load(path, weights_only=False)
    del ck['model_cfg']['correction_steps'], ck['model_cfg']['rich_path_context']
    if not correction:
        del ck['model_cfg']['correction'], ck['model_cfg']['correction_limit']
    torch.save(ck, path)
    loaded = load_checkpoint(path, 'cpu')[0]
    assert loaded.cfg.correction == correction and not loaded.cfg.rich_path_context
    assert loaded.cfg.correction_steps == 1
    for key, value in forward(loaded, b).items():
        torch.testing.assert_close(value, out[key], rtol=0, atol=0)


def test_decision_metrics_score_actual_commits_censor_unknowns_and_pool_counts():
    import json
    from vesuvius.neural_tracing.fiber_follow.direct.diagnostics import decision_rows, summarize_decisions
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
    out = dict(points=points, initial_points=points+torch.tensor([.2, 0., 0.]), confidence=conf)
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
    from vesuvius.neural_tracing.fiber_follow.data import TracedFiber
    from vesuvius.neural_tracing.fiber_follow.direct.recovery import monitor_fixture
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
        results.append(optimizer_update(model, copy.deepcopy(model), opt, [data], 1, .001,
                                        n_commit=2, compute_metrics=enabled))
        torch.testing.assert_close(torch.get_rng_state(), rng)
    assert 'decisions' not in results[0] and results[1]['decisions']['n_commit'] == 2
    assert results[0]['loss'] == results[1]['loss']
    for p, q in zip(a.parameters(), bmodel.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


@pytest.mark.parametrize('nested_images', [False, True])
def test_shared_recovery_evaluator_preserves_float_inputs_and_observed_states(nested_images):
    from vesuvius.neural_tracing.fiber_follow.data import TracedFiber
    from vesuvius.neural_tracing.fiber_follow.recovery import make_recovery_states, evaluate_recovery_states
    cfg = config()
    arc = np.arange(300, dtype=float)
    fiber = TracedFiber('line', np.c_[arc*0, arc*0, arc], arc, '')
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, recent_history_points=cfg.n_history,
                          n_future=cfg.n_future)
    states = make_recovery_states([fiber], [dict(fiber=0, t=150., sign=1)], sample, dict(split='monitor'))
    inputs = batch(cfg, 1)
    inputs['x'] = ({k: v.half() for k, v in inputs['x'].items()} if nested_images
                   else torch.zeros(1, 3, 4, 4, 4, dtype=torch.float16))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
        def forward(self, x, hist, hmask):
            tensors = x.values() if isinstance(x, dict) else [x]
            assert all(t.dtype == torch.float32 for t in tensors)
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


def test_tight_blocks_match_rotation_invariant_blocks_at_array_edges():
    """Per-item minimal blocks read the same values, including zero fill beyond the array."""
    from vesuvius.neural_tracing.fiber_follow.data import _grid_flat, read_blocks
    from vesuvius.neural_tracing.fiber_follow.fast_sample import sample_crop
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
    vol = SimpleNamespace(presence=presence, input_scale=2., raw_block=lambda s, z: ct.read(s, z)[None])
    crop = CropSpec(depth=14, width=10, behind=5, spacing=.5)

    def reference(items):
        grid, empty, mask = _grid_flat(crop), np.empty((0, 3), np.float32), np.empty(0, np.float32)
        out = np.empty((len(items), 2, crop.depth, crop.width, crop.width), np.float32)
        for use_presence in (False, True):
            raw, starts = read_blocks(items, vol, crop, presence=use_presence)
            scale = 1. if use_presence else vol.input_scale
            for j, item in enumerate(items):
                out[j, int(use_presence)] = sample_crop(raw[j], starts[j], item['pos']*scale, item['frame']*scale, grid,
                    False, empty, mask, 2, 1., 'points')[0].reshape(crop.depth, crop.width, crop.width)
        return torch.from_numpy(out)

    items = []
    for _ in range(24):
        # Interior, straddling each face, and fully outside; arbitrary orientation.
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        pos = rng.uniform(-6, 38, 3)
        items.append(dict(pos=pos, frame=q*np.sign(np.linalg.det(q))))
    image = image_crop(items, vol, crop)
    expected = reference(items)
    assert torch.equal(image, expected)
    assert 0 < float((expected != 0).float().mean()) < 1
