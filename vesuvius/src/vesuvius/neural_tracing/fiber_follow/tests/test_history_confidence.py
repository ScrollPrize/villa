"""History confidence and native CT/presence contracts for the next run."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow import data as D
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec, arclength, crop_local_grid, frame_from_heading
from vesuvius.neural_tracing.fiber_follow.history_audit import HistoryAudit
from vesuvius.neural_tracing.fiber_follow.history_metrics import cleaning_measurements
from vesuvius.neural_tracing.fiber_follow.model import FollowNet, FollowNetConfig, history_tangent
from vesuvius.neural_tracing.fiber_follow.supervision import loss_fn, teacher_candidates
from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer, TraceParams, field_axis
from vesuvius.neural_tracing.fiber_follow.train import load_checkpoint, save_checkpoint
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume, FiberVolumeSpec


def config():
    return FollowNetConfig(in_channels=3, depth=16, width=9, behind=6, spacing=.5,
                           widths=(8, 16), hidden=16, n_future=4, future_step=1.,
                           hist_points=4, hist_stride=2, clean_points=4, n_candidates=2,
                           flow_layers=1, flow_heads=2, flow_steps=2, flow_samples=4, flow_draws=2, norm='group')


def array_at(path, values):
    path.mkdir(parents=True)
    (path / '.zarray').write_text(json.dumps(dict(shape=list(values.shape), chunks=list(values.shape),
                                                dtype='|u1', fill_value=0, order='C', filters=None,
                                                compressor=None, zarr_format=2)))
    (path / '0.0.0').write_bytes(values.astype(np.uint8).tobytes())


def volume(tmp_path):
    z, y, x = np.indices((16, 16, 16))
    array_at(tmp_path / 'fields/test_presence.ome.zarr/3', 20 + x + 2*y + 3*z)
    z, y, x = np.indices((32, 32, 32))
    array_at(tmp_path / 'ct/0', x + 2*y + 3*z)
    spec = FiberVolumeSpec(str(tmp_path / 'fields'), ct_zarr=str(tmp_path / 'ct'),
                           ct_level=0, ct_grid_scale=4, inputs='ct+presence')
    return FiberVolume(spec, cache_bytes=1 << 20)


@pytest.mark.parametrize('fused', [False, True])
def test_native_ct_and_presence_sample_the_same_world_positions(tmp_path, monkeypatch, fused):
    vol = volume(tmp_path)
    assert vol.nx is None and vol.ny is None
    assert vol.input_scale == 2 and vol.channels == 2
    crop = CropSpec(depth=8, width=7, behind=3, spacing=.5, history_render='segments')
    frame = frame_from_heading(np.array([1., 2., 3.]))
    item = dict(pos=np.array([7.25, 7.5, 7.75]), frame=frame,
                hist_local=np.zeros((8, 3)), hmask=np.zeros(8))
    grid = torch.from_numpy(crop_local_grid(crop)).float()
    world = item['pos'] + grid.numpy() @ frame.T
    ramp = world[..., 0] + 2*world[..., 1] + 3*world[..., 2]
    monkeypatch.setattr(D, 'FUSED_SAMPLER', fused)
    batch = D.collate_with_volume([item], vol, crop, grid)
    assert batch['x'].shape == (1, 3, 8, 7, 7)
    np.testing.assert_allclose(batch['x'][0, 0], 2*ramp/255, atol=3e-4)
    np.testing.assert_allclose(batch['x'][0, 1], (20+ramp)/255, atol=3e-4)
    assert not batch['x'][0, 2].any()
    assert np.isfinite(field_axis(vol, item['pos'])[0]).all()
    assert vol.sample_image_nearest(np.array([[4., 4., 4.]]))[0] == 48


def test_presence_zero_padding_preserves_ct_and_history(tmp_path):
    vol = volume(tmp_path)
    crop = CropSpec(depth=3, width=3, behind=1)
    grid = torch.from_numpy(crop_local_grid(crop)).float()
    items = [dict(pos=np.array([-10., -10., -10.]), frame=np.eye(3))]
    x = torch.rand(1, 2, 3, 3, 3)
    result = D.add_presence_input(x, items, vol, crop, grid)
    torch.testing.assert_close(result[:, 0], x[:, 0])
    torch.testing.assert_close(result[:, 2], x[:, 1])
    assert not result[:, 1].any()


def score_inputs():
    cfg = config()
    features = torch.randn(1, cfg.widths[0], cfg.depth, cfg.width, cfg.width)
    hist = torch.zeros(1, 8, 3)
    hist[0, :, 2] = -torch.arange(1, 9)
    mask = torch.ones(1, 8)
    clean = torch.cat([torch.zeros(1, 1, 3), hist[:, :cfg.clean_points]], 1)
    candidates = torch.zeros(1, 2, cfg.n_future, 3)
    candidates[..., 2] = torch.arange(1, cfg.n_future + 1)
    return features, candidates, clean, hist, mask


def test_ranking_and_confidence_use_both_histories_and_reset_between_calls():
    torch.manual_seed(12)
    model = FollowNet(config()).eval()
    features, candidates, clean, hist, mask = score_inputs()
    ranks, confidence = model.score_candidates(features, candidates, clean, hist, mask)
    changed_hist = hist.clone()
    changed_hist[:, 1:4, 0] += 1.5
    changed_clean = clean.clone()
    changed_clean[:, 1:4, 0] -= 1.5
    r1, observed_change = model.score_candidates(features, candidates, clean, changed_hist, mask)
    r2, cleaned_change = model.score_candidates(features, candidates, changed_clean, hist, mask)
    assert (observed_change - confidence).abs().min() > 1e-8
    assert (cleaned_change - confidence).abs().min() > 1e-8
    assert (ranks - r1).abs().min() > 1e-8
    assert (ranks - r2).abs().min() > 1e-8
    torch.testing.assert_close(confidence, model.score_candidates(features, candidates, clean, hist, mask)[1])
    confidence.sum().backward()
    assert model.history_tokens[0].weight.grad.abs().sum() > 0
    assert model.prefix_context.weight_hh_l0.grad.abs().sum() > 0


def test_masked_history_coordinates_cannot_change_confidence():
    torch.manual_seed(17)
    model = FollowNet(config()).eval()
    features, candidates, clean, hist, mask = score_inputs()
    mask[:, 1:4] = 0
    first = model.score_candidates(features, candidates, clean, hist, mask)[1]
    hist[:, 1:4] = 10000
    clean[:, 2:5] = -10000
    torch.testing.assert_close(first, model.score_candidates(features, candidates, clean, hist, mask)[1])
    mask.zero_()
    assert torch.isfinite(model.score_candidates(features, candidates, clean, hist, mask)[1]).all()


def test_full_continuation_changes_first_confidence_without_cross_candidate_leakage():
    torch.manual_seed(23)
    cfg = replace(config(), depth=208, behind=64, n_future=64,
                  hist_points=32, hist_stride=4, clean_points=32)
    model = FollowNet(cfg).eval()
    features = torch.randn(1, cfg.widths[0], cfg.depth, cfg.width, cfg.width)
    hist = torch.zeros(1, 128, 3)
    hist[..., 2] = -torch.arange(1, 129)
    mask = torch.ones(1, 128)
    clean = torch.cat([torch.zeros(1, 1, 3), hist[:, :32]], 1)
    candidates = torch.zeros(1, 2, 64, 3)
    candidates[..., 2] = torch.arange(1, 65)
    candidates[:, 1, :, 0] = .5
    ranks, confidence = model.score_candidates(features, candidates, clean, hist, mask)
    changed = candidates.clone()
    changed[:, 0, 48:, 0] = 1.5
    r1, c1 = model.score_candidates(features, changed, clean, hist, mask)
    assert (r1[:, 0] - ranks[:, 0]).abs().item() > 1e-7
    assert (c1[:, 0, 0] - confidence[:, 0, 0]).abs().item() > 1e-7
    # The reverse recurrent path carries the distant suffix into the first
    # decision. Inspect gradients as well: a purely local path cannot do this.
    distant = candidates.clone().requires_grad_()
    _, logits = model.score_candidates(features, distant, clean, hist, mask)
    logits[0, 0, 0].backward()
    assert distant.grad[0, 0, -1].abs().sum() > 0
    torch.testing.assert_close(r1[:, 1], ranks[:, 1])
    torch.testing.assert_close(c1[:, 1], confidence[:, 1])
    # Candidate ordering has no meaning, including teacher candidates.
    rp, cp = model.score_candidates(features, changed.flip(1), clean, hist, mask)
    torch.testing.assert_close(rp.flip(1), r1)
    torch.testing.assert_close(cp.flip(1), c1)
    assert torch.isfinite(c1).all()


def test_ranking_compares_each_candidate_with_history_beyond_eight_points():
    torch.manual_seed(31)
    cfg = replace(config(), depth=80, behind=64, hist_points=32, hist_stride=4, clean_points=32)
    model = FollowNet(cfg).eval()
    features = torch.randn(1, cfg.widths[0], cfg.depth, cfg.width, cfg.width)
    hist = torch.zeros(1, 128, 3)
    hist[..., 2] = -torch.arange(1, 129)
    hist.requires_grad_()
    mask = torch.ones(1, 128)
    clean = torch.cat([torch.zeros(1, 1, 3), hist.detach()[:, :32]], 1).requires_grad_()
    candidates = torch.zeros(1, 2, 4, 3)
    candidates[..., 2] = torch.arange(1, 5)
    candidates[:, 1, :, 0] = 1.
    ranks, _ = model.score_candidates(features, candidates, clean, hist, mask)
    # A shared additive history bias would cancel here and cannot select a path.
    (ranks[0, 0] - ranks[0, 1]).backward()
    assert hist.grad[:, 15:32].abs().sum() > 0
    assert clean.grad[:, 16:33].abs().sum() > 0
    assert model.continuation_fusion[0].weight.grad.abs().sum() > 0
    changed = hist.detach().clone()
    changed[:, 15:32, 0] += 1.
    r1, _ = model.score_candidates(features, candidates, clean.detach(), changed, mask)
    assert ((r1[0, 0]-r1[0, 1])-(ranks[0, 0]-ranks[0, 1])).abs() > 1e-8


def test_long_horizon_labels_keep_correct_prefixes_and_censor_unknown_tail():
    from vesuvius.neural_tracing.fiber_follow.supervision import candidate_labels
    cfg = D.SampleConfig(n_future=64, future_step=1., clean_points=32)
    points = np.array([[float(x), 0., 0.] for x in range(201)])
    fiber = D.TracedFiber('line', points, arclength(points), '')
    frame = frame_from_heading(np.array([1., 0., 0.]))
    targets = D.continuation_targets(fiber, 100., False, np.array([100., 0., 0.]), frame, cfg)
    batch = {k: torch.as_tensor(v)[None] for k, v in targets.items()}
    candidates = torch.zeros(1, 2, 64, 3)
    candidates[..., 2] = torch.arange(1, 65)
    candidates[:, 1, 32:, 0] = 5.
    labels, mask, quality, _ = candidate_labels(candidates, batch)
    assert labels[0, 0].all() and mask.all()
    assert labels[0, 1, :4].all() and not labels[0, 1, -1]
    assert quality[0, 0] > quality[0, 1]
    targets = D.continuation_targets(fiber, 180., False, np.array([180., 0., 0.]), frame, cfg)
    batch = {k: torch.as_tensor(v)[None] for k, v in targets.items()}
    _, mask, _, _ = candidate_labels(candidates, batch)
    assert mask[0, 0, :4].all() and not mask[0, 0, -1]


def test_history_context_is_independent_across_batch_and_supports_bfloat16():
    torch.manual_seed(11)
    model = FollowNet(config()).eval()
    args = score_inputs()
    batch = [a.expand(2, *a.shape[1:]).clone() for a in args]
    batch[3][1, :, 0] = 1.5
    together = model.score_candidates(*batch)[1]
    for i in range(2):
        separate = model.score_candidates(*[a[i:i+1] for a in batch])[1]
        torch.testing.assert_close(together[i:i+1], separate, atol=1e-6, rtol=1e-5)
    # Some CPU oneDNN builds support BF16 forward but not backward. Exercise
    # the dtype contract with native kernels, independent of that CPU feature.
    with torch.backends.mkldnn.flags(enabled=False):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits = model.score_candidates(*batch)[1]
        logits.float().sum().backward()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(model.history_tokens[0].weight.grad).all()


def test_tangent_fit_uses_several_points_and_ignores_missing_tail():
    p = torch.zeros(1, 6, 3)
    p[0, :, 2] = -torch.arange(6)
    p[0, 1, 0] = 1
    tangent2, _ = history_tangent(p, torch.ones(1, 6), 2)
    tangent6, valid = history_tangent(p, torch.ones(1, 6), 6)
    assert valid.item() and abs(tangent6[0, 0]) < abs(tangent2[0, 0])
    mask = torch.tensor([[1., 1., 0., 1., 1., 1.]])
    p[:, 2:] = 1000
    torch.testing.assert_close(history_tangent(p, mask, 6)[0], tangent2)
    tangent, valid = history_tangent(p, torch.tensor([[1., 0., 0., 0., 0., 0.]]), 6)
    assert not valid.item()
    torch.testing.assert_close(tangent, torch.tensor([[0., 0., 1.]]))


def test_cleaning_measurements_use_common_masks_and_do_not_score_departed_geometry():
    _, _, target, hist, mask = score_inputs()
    hist[:, :4, 0] = 1
    target_mask = torch.ones(1, 5)
    result = cleaning_measurements(target, hist, mask, target, target_mask, 4)
    assert result['clean_history_error'][0].item() == 0
    assert result['observed_history_error'][0].item() == pytest.approx(.8)
    assert result['clean_position_improved'][0].item() == 1
    result = cleaning_measurements(target, hist, mask, target, target_mask*0, 4)
    assert not result['clean_history_error'][1].item()
    assert not result['clean_tangent_error_deg'][1].item()
    assert result['clean_correction_size'][1].item()


def test_training_rollout_checkpoint_and_audit_share_inputs(tmp_path):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    cfg = config()
    vol = volume(tmp_path)
    crop = CropSpec(depth=cfg.depth, width=cfg.width, behind=cfg.behind, spacing=cfg.spacing,
                    history_render='segments', history_sigma=.35)
    sample = D.SampleConfig(crop=crop, n_history=8, clean_points=cfg.clean_points, n_future=4,
                            future_step=1., n_candidates=2)
    p = np.array([[x, 8., 8.] for x in np.linspace(0, 15, 31)])
    fiber = D.TracedFiber('synthetic', p, arclength(p), '')
    pos, heading = np.array([8., 8., 8.]), np.array([1., 0., 0.])
    history = np.array([[x, 8., 8.] for x in range(8)])
    item = D.label_state(fiber, pos, frame_from_heading(heading), history[::-1], np.ones(8), sample,
                         t=8., reverse=False)
    batch = D.collate_with_volume([item], vol, crop, torch.from_numpy(crop_local_grid(crop)).float())
    model = FollowNet(cfg)
    out = model(batch['x'].float(), batch['hist'], batch['hmask'], teacher_candidates(batch, cfg), targets=batch)
    loss, metrics = loss_fn(out, batch, cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert 'observed_tangent_error_deg' in metrics
    for parameter in (model.history_tokens[0].weight, model.flow.velocity.weight,
                      model.prefix_context.weight_ih_l0, model.encoders[0][0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    path = tmp_path / 'new.pt'
    save_checkpoint(path, model, vol.spec, sample, dict(tolerance=1.5))
    loaded, loaded_crop, nh, spec, checkpoint = load_checkpoint(path, 'cpu')
    assert checkpoint['architecture'] == 'joint_flow_v8'
    assert loaded.cfg == cfg and loaded_crop == crop and nh == 8 and spec == vol.spec
    captured = []
    handle = loaded.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
    tracer = ModelTracer(loaded, vol, crop, nh, TraceParams(max_len=1, confidence=.99), device='cpu')
    audit = HistoryAudit(tracer)
    audit.start_batch([fiber], [dict(fiber=0, t=8., sign=1.)])
    try:
        paths, reasons = tracer.trace(pos[None], heading[None], histories=[history], on_decision=audit)
    finally:
        tracer.close()
        handle.remove()
    torch.testing.assert_close(captured[0], batch['x'].float(), atol=3e-4, rtol=0)
    assert audit.summary()['groups']['all']['states'] >= 1
    assert audit.summary()['groups']['ontrack']['metrics']['clean_current_error']['count'] >= 1
    assert len(paths) == len(reasons) == 1


def test_training_entrypoint_writes_presence_history_run(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_follow import train
    vol = volume(tmp_path)
    points = np.array([[float(x), 8., 8.] for x in range(16)])
    fiber = D.TracedFiber('line', points, arclength(points), '')
    monkeypatch.setattr(train, 'load_fibers', lambda *a, **kw: [fiber])
    path = train.main([
        '--fiber-zarrs', vol.spec.fiber_zarr_dir, '--fibers', str(tmp_path), '--ct', vol.spec.ct_zarr,
        '--ct-level', '0', '--ct-grid-scale', '4', '--inputs', 'ct+presence',
        '--out-root', str(tmp_path / 'runs'), '--name', 'smoke', '--steps', '2', '--batch', '2',
        '--device', 'cpu', '--workers', '0', '--dagger-every', '0', '--diag-every', '0',
        '--log-every', '1', '--ckpt-every', '2', '--crop-depth', '16', '--crop-width', '9',
        '--crop-behind', '6', '--crop-spacing', '.5', '--n-history', '8', '--hist-points', '4',
        '--hist-stride', '2', '--clean-points', '4', '--flow-layers', '1', '--flow-heads', '2',
        '--flow-steps', '2', '--flow-samples', '4', '--flow-draws', '2',
        '--n-future', '4', '--future-step', '1',
        '--n-candidates', '2', '--widths', '8', '16', '--hidden', '16',
        '--norm', 'group', '--history-render', 'segments', '--history-jitter', '0'])
    model, _, _, spec, ck = load_checkpoint(path, 'cpu')
    assert spec.mode == 'ct+presence' and model.cfg.in_channels == 3 and ck['step'] == 2
    records = [json.loads(line) for line in (tmp_path / 'runs/smoke/log.jsonl').read_text().splitlines()]
    training = [r for r in records if 'loss' in r]
    assert len(training) == 2 and all(np.isfinite(r['loss']) for r in training)
    assert all('clean_tangent_error_deg_count' in r for r in training)


def test_diagnostic_batching_preserves_all_seeds_and_metrics(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.diag import rollout_diag
    points = np.array([[float(x), 0., 0.] for x in range(21)])
    fiber = D.TracedFiber('line', points, arclength(points), '')
    seeds = [dict(fiber=0, t=float(t), sign=1., pos=np.array([t, 0., 0.]),
                  heading=np.array([1., 0., 0.])) for t in range(4, 9)]
    calls = []
    def trace(pos, heading):
        calls.append(len(pos))
        paths = [p[None] + np.arange(3)[:, None]*h[None] for p, h in zip(pos, heading)]
        return paths, ['max_len']*len(pos)
    tracer = SimpleNamespace(p=SimpleNamespace(max_len=12.), trace=trace,
                             vol=SimpleNamespace(sample_image_nearest=lambda q: np.zeros(q.shape[:-1])))
    small = rollout_diag(tracer, [fiber], seeds, tmp_path/'small.png', max_len=4., half=2, batch=2)
    assert calls == [2, 2, 1] and tracer.p.max_len == 12.
    large = rollout_diag(tracer, [fiber], seeds, tmp_path/'large.png', max_len=4., half=2, batch=5)
    assert small == large


@pytest.mark.parametrize('case', ['false_stop_first', 'false_go_first', 'offtrack', 'unknown_end'])
def test_audit_classifies_gate_errors_and_censors_unknown_end(case):
    cfg = config()
    crop = CropSpec(depth=16, width=9, behind=6, spacing=.5)
    tracer = SimpleNamespace(model=SimpleNamespace(cfg=cfg), crop=crop, n_history=8,
                             p=TraceParams(confidence=.7))
    points = np.array([[float(x), 0., 0.] for x in range(21)])
    fiber = D.TracedFiber('line', points, arclength(points), '')
    audit = HistoryAudit(tracer)
    t = 18. if case == 'unknown_end' else 8.
    audit.start_batch([fiber], [dict(fiber=0, t=t, sign=1.)])
    pos = np.array([t, 0., 0.])
    frame = frame_from_heading(np.array([1., 0., 0.]))
    hist = pos - np.arange(1, 9)[:, None]*frame[:, 2]
    candidates = np.zeros((2, 4, 3))
    candidates[..., 2] = np.arange(1, 5)
    clean = np.zeros((5, 3))
    clean[:, 2] = -np.arange(5)
    if case == 'false_go_first':
        candidates[..., 0] = 5
    state = dict(pos=pos, frame=frame, hist=hist, hmask=np.ones(8), candidates=candidates,
                 confidence=np.full((2, 4), .2 if case == 'false_stop_first' else .9), rank_scores=np.ones(2),
                 chosen=0, n_commit=0 if case == 'false_stop_first' else 4, would_stop=case == 'false_stop_first',
                 exploratory=False, travelled=0., last_segment=pos[None], clean_history=clean)
    if case == 'offtrack':
        state['pos'] = pos + np.array([0., 5., 0.])
        state['last_segment'] = np.stack([pos, state['pos']])
        state['travelled'] = 5.
    if case == 'unknown_end':
        state['pos'] = np.array([21., 0., 0.])
        state['last_segment'] = np.stack([pos, state['pos']])
        state['travelled'] = 3.
    assert audit(0, state) is None  # never asks the tracer to stop
    summary = audit.summary()
    if case == 'unknown_end':
        assert summary['censored_traces'] == 1 and not summary['groups']
    else:
        assert summary['groups'][case]['states'] == 1
        if case == 'offtrack':
            assert 'clean_current_error' not in summary['groups'][case]['metrics']
            assert 'clean_correction_size' in summary['groups'][case]['metrics']
        if case == 'false_go_first':
            assert summary['groups']['no_correct_candidate_first']['states'] == 1
