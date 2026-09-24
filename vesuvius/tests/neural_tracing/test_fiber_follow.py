"""Controlled-span supervision, holdout isolation, and trace length accounting."""
from dataclasses import asdict, replace
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from vc3d_fiber_format import legacy_lasagna_segments
from vesuvius.neural_tracing.fiber_follow import data as D
from vesuvius.neural_tracing.fiber_follow.collect import DecisionCollector
from vesuvius.neural_tracing.fiber_follow.evaluate import load_or_make_seeds, score_trace, summarize
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec, arclength
from vesuvius.neural_tracing.fiber_follow.train import loss_fn, load_checkpoint, save_checkpoint
from vesuvius.neural_tracing.fiber_follow.model import FollowNet, FollowNetConfig
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolumeSpec


def fiber(length=100, z=0, endpoints=(False, False)):
    p = np.c_[np.arange(length + 1), np.zeros(length + 1), np.full(length + 1, z)].astype(float)
    return D.TracedFiber('test.json', p, arclength(p), 'H', endpoint_stop=endpoints)


def write_fiber(path, *, reviewed=False, reverse_line=False, single_control=False, version=1):
    line = [[float(x), 0., 0.] for x in range(41)]
    controls = [line[5], line[17], line[33]]
    if single_control:
        controls = controls[:1]
    raw = dict(type='vc3d_fiber', version=version, line_points=line[::-1] if reverse_line else line,
               control_points=controls, tags=['reviewed'] if reviewed else [])
    if version == 3:
        segment = asdict(legacy_lasagna_segments(2)[0])
        segment.pop('outcome')
        segment.update(optimizer='native_fiber_trace3d', metadata_version=3, tracer_version=2,
                       interp_mode='cspline', interp_goal='cspline', msg='cspline')
        positive = ['step_voxels', 'cone_angle_degrees', 'cone_angle_step_degrees', 'cone_grid_size',
                    'beam_width', 'beam_prune_distance_voxels', 'max_step_factor', 'endpoint_accept_threshold_base_voxels']
        nonnegative = ['beam_lookahead_steps', 'smoothness_weight', 'smoothness_normal_weight',
                       'smoothness_tangent_weight', 'smoothness_free_angle_degrees',
                       'cumulative_smoothness_steps', 'cumulative_smoothness_tangent_weight',
                       'initial_free_angle_degrees', 'meeting_accept_max_error_ratio']
        segment['config'] = {**dict.fromkeys(positive, 1), **dict.fromkeys(nonnegative, 0)}
        raw['optimization_mode'] = 'native_fiber_trace3d'
        raw['control_points'] = [dict(position=p, **({'segment_to_next': segment} if i < len(controls)-1 else {}))
                                 for i, p in enumerate(controls)]
        raw['control_points'][0]['tags'] = ['kollesis_termination']
    path.write_text(json.dumps(raw))
    return raw


@pytest.mark.parametrize('reviewed', [False, True])
@pytest.mark.parametrize('reverse_line', [False, True])
def test_loader_trims_even_reviewed_tails_and_retains_controls(tmp_path, reviewed, reverse_line):
    write_fiber(tmp_path / 'f.json', reviewed=reviewed, reverse_line=reverse_line)
    f, = D.load_fibers(str(tmp_path), grid_scale=1, spacing=2.5)
    np.testing.assert_array_equal(f.points[[0, -1], 0], [5, 33])
    assert 17 in f.points[:, 0]
    assert f.excluded_tail_length == 12
    assert f.endpoint_stop == (False, False)
    assert [(s.start, s.end) for s in f.spans] == [(0, 12), (12, 28)]
    assert all(s.provenance.interp_mode == 'lasagna' for s in f.spans)


def test_loader_keeps_v3_provenance_and_explicit_termination(tmp_path):
    write_fiber(tmp_path / 'f.json', version=3)
    f, = D.load_fibers(str(tmp_path), grid_scale=1)
    assert f.endpoint_stop == (True, False)
    assert f.spans[0].provenance.interp_goal == 'cspline'
    assert f.spans[0].provenance.config['step_voxels'] == 1


def test_seed_only_fiber_is_not_supervision(tmp_path):
    write_fiber(tmp_path / 'f.json', single_control=True)
    assert D.load_fibers(str(tmp_path), grid_scale=1) == []


def test_unanchored_controls_fail_instead_of_using_nearest_winding(tmp_path):
    raw = write_fiber(tmp_path / 'f.json')
    raw['control_points'][1] = [17., 1., 0.]
    (tmp_path / 'f.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='anchored'):
        D.load_fibers(str(tmp_path), grid_scale=1)


def test_split_uses_controlled_geometry_centroid():
    f = fiber(z=150)
    assert D.split_fibers([f], D.ZBand(100, 200)) == ([], [f])










def safe_item(z=0):
    return dict(pos=np.array([0., 0., z]), frame=np.eye(3),
                hist_local=np.zeros((1, 3)), hmask=np.zeros(1),
                fut_local=np.zeros((1, 3)), fmask=np.zeros(1))


def test_holdout_filter_checks_perturbed_position_and_rotated_read_block():
    band = D.ZBand(100, 200)
    assert D.training_state_allowed(safe_item(0), CropSpec(), band)
    assert not D.training_state_allowed(safe_item(60), CropSpec(), band)
    # Position is outside the old fixed guard, but a long crop reaches the holdout.
    assert not D.training_state_allowed(safe_item(0), CropSpec(depth=150, behind=8), band)


@pytest.mark.parametrize('key,mask', [('hist_local', 'hmask'), ('fut_local', 'fmask')])
def test_holdout_filter_checks_supervision_and_long_history(key, mask):
    item = safe_item(0)
    item[key] = np.array([[0., 0., 150.]])
    item[mask] = np.ones(1)
    assert not D.training_state_allowed(item, CropSpec(), D.ZBand(100, 200))


def test_holdout_filter_checks_heatmap_targets():
    item = safe_item(0)
    item.update(plane_ab=np.zeros((1, 2)), plane_mask=np.ones(1), planes=np.array([150.]))
    assert not D.training_state_allowed(item, CropSpec(), D.ZBand(100, 200))






def test_onpolicy_cache_requires_identity(tmp_path):
    path = tmp_path / 'old.npz'
    np.savez(path, pos=np.zeros((1, 3)))
    with pytest.raises(KeyError, match='__metadata__'):
        D.OnPolicyStates.load(path)
    assert not (tmp_path / 'old_mmap_v3').exists()


def test_onpolicy_identity_and_mmap_roundtrip(tmp_path):
    f = fiber()
    op = make_states(f)
    path = tmp_path / 'states.npz'
    op.save(path)
    loaded = D.OnPolicyStates.load(path)
    loaded.validate_fibers([f])
    assert isinstance(loaded.pos, np.memmap)
    pickle.loads(pickle.dumps(loaded)).validate_fibers([f])
    moved = replace(f, points=f.points + 1)
    with pytest.raises(ValueError, match='incompatible'):
        loaded.validate_fibers([moved])
    # Replacing an NPZ also refreshes the mmap, even if its manifest is unchanged.
    op.pos[0, 0] = 42
    op.save(path)
    assert D.OnPolicyStates.load(path).pos[0, 0] == 42






def test_presence_cache_identity_checks_geometry_not_only_length(tmp_path):
    class Presence:
        def __init__(self): self.calls = 0
        def sample_nearest(self, q):
            self.calls += 1
            return np.full(q.shape[:-1], 255)
    vol = SimpleNamespace(spec=FiberVolumeSpec('unused'), presence=Presence())
    f = fiber()
    path = tmp_path / 'presence.npz'
    D.gt_presence([f], vol, path)
    D.gt_presence([f], vol, path)
    assert vol.presence.calls == 1
    D.gt_presence([replace(f, points=f.points + 1)], vol, path)
    assert vol.presence.calls == 2


@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('known', [False, True])
def test_endpoint_crossing_partitions_segment_exactly(sign, known):
    f = fiber(endpoints=(known, known))
    x = np.arange(50, 112, 2.5) if sign > 0 else np.arange(50, -12, -2.5)
    p = np.c_[x, np.zeros(len(x)), np.zeros(len(x))]
    m = score_trace(p, f, 50, sign)
    assert m['correct'] == 50
    assert m['followed'] == 50
    assert m['unknown'] == (0 if known else 10)
    assert m['offtrack'] == (10 if known else 0)
    assert m['endpoint_overrun'] == (10 if known else 0)
    assert not m['diverged']
    assert m['correct'] + m['offtrack'] + m['unknown'] == m['length']


def test_crossing_inside_long_segment_keeps_fractional_length():
    f = fiber()
    p = np.array([[90., 0, 0], [99., 0, 0], [103., 0, 0], [107., 0, 0], [110., 0, 0]])
    m = score_trace(p, f, 90, 1)
    assert m['correct'] == 10 and m['unknown'] == 10


def test_early_departure_then_return_is_not_endpoint_censored():
    f = fiber()
    x = np.arange(50, 112, dtype=float)
    p = np.c_[x, np.zeros(len(x)), np.zeros(len(x))]
    p[5:10, 1] = 5
    m = score_trace(p, f, 50, 1)
    assert m['diverged'] and m['unknown'] == 0
    assert m['correct'] == 4
    assert m['correct'] + m['offtrack'] == m['length']


def test_departure_near_endpoint_is_not_excused_by_reached_end_tolerance():
    f = fiber()
    p = np.array([[95., 0, 0], [98., 0, 0], [98., 4, 0], [98., 5, 0], [98., 6, 0]])
    m = score_trace(p, f, 95, 1)
    assert m['reached_end'] and m['diverged']
    assert m['unknown'] == 0 and m['offtrack'] == 6


def test_nearby_earlier_winding_does_not_censor_at_endpoint_plane():
    p = np.array([[x, 1, 0] for x in range(11)] +
                 [[10, y, 0] for y in range(2, 11)] +
                 [[x, 10, 0] for x in range(9, -1, -1)] +
                 [[0, y, 0] for y in range(9, -1, -1)] +
                 [[x, 0, 0] for x in range(1, 6)], dtype=float)
    f = D.TracedFiber('winding', p, arclength(p), 'H')
    m = score_trace(p[:11], f, 0, 1)
    assert m['unknown'] == 0
    assert not m['crossed_endpoint']
    assert m['followed'] == 10


def test_summary_accounts_for_all_length_and_unknown_is_not_verified():
    f = fiber()
    x = np.arange(50., 111.)
    row = score_trace(np.c_[x, x*0, x*0], f, 50, 1)
    s = summarize([row])
    assert s['length_precision'] == 1
    assert s['verified_length_fraction'] == pytest.approx(50/60)
    assert s['unknown_length_fraction'] == pytest.approx(10/60)
    assert s['correct_length'] + s['wrong_length'] + s['unknown_length'] == s['total_length']






def test_seed_cache_requires_matching_identity(tmp_path, monkeypatch):
    import vesuvius.neural_tracing.fiber_follow.evaluate as E
    f = fiber()
    vol = SimpleNamespace(spec=FiberVolumeSpec('unused'))
    path = tmp_path / 'seeds.pkl'
    path.write_bytes(pickle.dumps(dict(seeds=[], val=['test.json'])))
    with pytest.raises(ValueError, match='Incompatible seed cache'):
        load_or_make_seeds(path, [f], vol, D.ZBand(0, 1))
    monkeypatch.setattr(E, 'make_seeds', lambda *a, **kw: [dict(t=20)])
    assert load_or_make_seeds(path, [f], vol, D.ZBand(0, 1), rebuild=True) == [dict(t=20)]
    assert load_or_make_seeds(path, [f], vol, D.ZBand(0, 1)) == [dict(t=20)]
    with pytest.raises(ValueError, match='Incompatible seed cache'):
        load_or_make_seeds(path, [replace(f, points=f.points+1)], vol, D.ZBand(0, 1))

from vesuvius.neural_tracing.fiber_follow.geometry import frame_from_heading
from vesuvius.neural_tracing.fiber_follow.supervision import candidate_labels, teacher_candidates
from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.online import publish_replay, OnlineCollector


def small_config():
    return FollowNetConfig(depth=12, width=9, behind=2, n_future=4, widths=(8, 16), hidden=16,
                           hist_points=2, hist_stride=2, heat_bins=7, n_candidates=2, norm='group')


def sample_config():
    return D.SampleConfig(crop=CropSpec(depth=12, width=9, behind=2), n_history=4, n_future=4, n_candidates=2)


def target_batch(f=None, t=50, reverse=False, offtrack=False):
    f = f or fiber()
    cfg = sample_config()
    frame = frame_from_heading(np.array([-1 if reverse else 1., 0, 0]))
    traversal_t = f.length-t if reverse else t
    item = D.continuation_targets(f, traversal_t, reverse, f.points[t], frame, cfg, offtrack)
    return {key: torch.as_tensor(value, dtype=torch.float32)[None] for key, value in item.items()}


def straight_candidates(batch=1, count=2, k=4):
    points = torch.zeros(batch, count, k, 3)
    points[..., 2] = torch.arange(1, k+1)*2
    return points


def make_states(f, z=0, offtrack=False):
    cfg = sample_config()
    return D.OnPolicyStates(manifest=D.fiber_manifest([f]),
        fiber_idx=[0], t=[50.], reverse=[False], pos=[[50., 0., z]],
        frame=[frame_from_heading(np.array([1., 0, 0]))], hist=np.zeros((1, 4, 3)), hmask=np.zeros((1, 4)),
        offtrack=[offtrack], hard=[offtrack], exploratory=[False], candidates=straight_candidates().numpy(),
        chosen=[0], confidence=np.ones((1, 2, 4)), rank_scores=np.zeros((1, 2)))


@pytest.mark.parametrize('reverse', [False, True])
def test_dense_supervision_includes_all_low_presence_points(reverse):
    f = fiber()
    f.brk = np.ones(len(f.points), bool)
    cfg = sample_config()
    cfg.lateral_sigmas = cfg.angle_sigmas_deg = (0., 0., 0.)
    item = D.make_sample(f, 50, reverse, cfg, np.random.default_rng(5))
    assert item['plane_mask'].sum() == cfg.n_future
    assert item['dense_mask'].all()
    assert 'stop' not in item
    batch = target_batch(f, reverse=reverse)
    targets, mask, _, _ = candidate_labels(straight_candidates(), batch)
    assert targets.all() and mask.all()


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('physical', [False, True])
def test_candidate_prefix_end_censoring(reverse, physical):
    f = fiber(endpoints=(physical, physical))
    b = target_batch(f, t=3 if reverse else 97, reverse=reverse)
    y, mask, _, _ = candidate_labels(straight_candidates(), b)
    assert y[0, 0, 0] == mask[0, 0, 0] == 1
    if physical:
        assert mask.all() and not y[..., 1:].any()
    else:
        assert not mask[..., 1:].any()


def test_prefix_supervision_detects_between_point_errors_and_persists_failure():
    b = target_batch()
    b['dense_ab'][0, 2, 0] = 4  # GT bends between the two predicted points
    y, mask, _, _ = candidate_labels(straight_candidates(), b)
    assert mask.all() and y[..., 0].all()
    assert not y[..., 1:].any()
    b['dense_mask'][0, 5:] = 0
    y, mask, _, _ = candidate_labels(straight_candidates(), b)
    assert mask.all() and not y[..., 1:].any()  # known failure survives an unknown tail


def test_unannotated_prefix_has_no_confidence_gradient():
    cfg = small_config()
    b = target_batch(t=99)
    logits = torch.zeros(1, 2, 4, requires_grad=True)
    output = dict(candidates=straight_candidates(), heatmap=torch.zeros(1, 4, 7, 7, requires_grad=True),
                  ranks=torch.zeros(1, 2, requires_grad=True), confidence_logits=logits)
    loss, _ = loss_fn(output, b, cfg)
    loss.backward()
    assert not logits.grad.any()


def test_departed_state_supplies_confidence_negatives_only():
    b = target_batch(offtrack=True)
    y, mask, _, _ = candidate_labels(straight_candidates(), b)
    assert not y.any() and mask.all()
    assert not b['plane_mask'].any()


def test_spatial_network_and_all_three_supervised_heads_backpropagate():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    cfg = small_config()
    model = FollowNet(cfg)
    b = target_batch()
    extras = teacher_candidates(b, cfg)
    out = model(torch.randn(1, 8, 12, 9, 9), torch.zeros(1, 4, 3), torch.zeros(1, 4), extras)
    assert out['candidates'].shape == (1, 9, 4, 3)
    assert torch.all(out['confidence'][..., 1:] <= out['confidence'][..., :-1])
    loss, metrics = loss_fn(out, b, cfg)
    assert torch.isfinite(loss)
    loss.backward()
    for layer in (model.heat_head, model.rank_head, model.confidence_head, model.encoders[0][0]):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
    assert set(('oracle_error','selected_error','oracle_recall')) <= metrics.keys()


def test_decoder_retains_distinct_coherent_modes():
    from vesuvius.neural_tracing.fiber_follow.model import decode_candidates
    cfg = small_config()
    cfg.heat_bins = 15
    logits = torch.full((1, 4, 15, 15), -20.)
    logits[:, :, 7, 3] = 5
    logits[:, :, 7, 11] = 5
    paths = decode_candidates(logits, cfg)
    assert (paths[0, :, :, 0].diff(dim=-1).abs() < .1).all()
    assert abs(paths[0, 0, 0, 0]-paths[0, 1, 0, 0]) > 6


def test_current_checkpoint_roundtrip(tmp_path):
    model = FollowNet(small_config())
    cfg = sample_config()
    spec = FiberVolumeSpec('unused')
    path = tmp_path/'model.pt'
    save_checkpoint(path, model, spec, cfg)
    loaded, crop, history, loaded_spec, ck = load_checkpoint(path, device='cpu')
    assert ck['data_policy'] == D.DATA_POLICY
    assert crop == cfg.crop and history == cfg.n_history and loaded_spec == spec
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key])


def test_cache_history_must_match_training_configuration():
    f = fiber()
    with pytest.raises(ValueError, match='history length'):
        D.FollowDataset([f], FiberVolumeSpec('unused'), D.SampleConfig(n_history=128), None,
                        onpolicy=[make_states(f)])


def test_live_replay_refresh_keeps_iterator_and_relabels_exact_frames(tmp_path, monkeypatch):
    f = fiber(length=300)
    op = make_states(f)
    path = tmp_path/'states.npz'
    op.save(path)
    cache = D.OnPolicyStates.load(path)
    index = tmp_path/'index.json'
    publish_replay(index, [])
    monkeypatch.setattr(D, 'FiberVolume', lambda *a, **kw: object())
    monkeypatch.setattr(D, 'collate_with_volume', lambda items, *args: items)
    ds = D.FollowDataset([f], FiberVolumeSpec('unused'), sample_config(), None, chunk=1,
                         replay_index=str(index), refresh_chunks=1, onpolicy_prob=1, hard_prob=0)
    iterator = iter(ds)
    next(iterator)  # starts fresh before collection completes
    publish_replay(index, [cache._dir])
    item, = next(iterator)  # same iterator now sees the new states
    np.testing.assert_array_equal(item['frame'], op.frame[0])
    np.testing.assert_array_equal(item['pos'], op.pos[0])


def decision_at(x, travelled, previous=None, would_stop=False):
    pos = np.array([x, 0., 0.])
    return dict(pos=pos, frame=frame_from_heading(np.array([1., 0, 0])), hist=np.zeros((4, 3)),
                hmask=np.zeros(4), candidates=straight_candidates()[0].numpy(), chosen=0,
                confidence=np.ones((2, 4)), rank_scores=np.zeros(2), exploratory=False,
                would_stop=would_stop, travelled=travelled,
                last_segment=np.array([pos]) if previous is None else np.array([previous, pos]))


def test_collector_censors_unknown_boundary_and_marks_pre_stop_window():
    c = DecisionCollector(fiber(), 0, 50, 1, sample_config(), stride=16)
    assert c(decision_at(50, 0))
    assert c(decision_at(58, 8, [50., 0, 0], would_stop=True))
    assert all(row['hard'] for row in c.rows)
    assert c(decision_at(96, 46, [58., 0, 0]))
    assert c(decision_at(102, 52, [96., 0, 0])) is False
    assert not any(r['offtrack'] for r in c.finish())


def test_collector_retains_predeparture_and_short_failure_suffix():
    c = DecisionCollector(fiber(length=300), 0, 50, 1, sample_config(), stride=16)
    assert c(decision_at(50, 0))
    d = decision_at(54, 6, [50., 0, 0])
    d['pos'][1] = 5
    d['last_segment'][-1, 1] = 5
    assert c(d)
    assert c.rows[0]['hard'] and c.rows[1]['offtrack']
    d['travelled'] = 40
    assert c(d) is False


class FixedPolicy(torch.nn.Module):
    def __init__(self, confidence):
        super().__init__()
        self.cfg = SimpleNamespace(future_step=2.)
        self.conf = confidence
    def forward(self, x, hist, hmask):
        n = len(x)
        candidates = straight_candidates(n)
        return dict(candidates=candidates, ranks=torch.tensor([[1.,0.]]).expand(n,-1),
                    confidence=torch.tensor(self.conf).reshape(1,1,4).expand(n,2,-1),
                    clean_history=torch.zeros(n, 9, 3))


def fake_tracer(confidence, params, monkeypatch):
    import vesuvius.neural_tracing.fiber_follow.trace as T
    monkeypatch.setattr(T, 'build_inputs', lambda raw, *a, **kw: torch.zeros(len(raw),8,12,9,9))
    vol = SimpleNamespace(spec=SimpleNamespace(mode='fiber'), shape=(1000,1000,1000), raw_block=lambda st, size: np.zeros((1,1,1,1), np.uint8))
    return ModelTracer(FixedPolicy(confidence), vol, sample_config().crop, 4, params, device='cpu')


def test_tracer_stops_before_unsafe_commit_and_captures_exact_decision(monkeypatch):
    tracer = fake_tracer([.2]*4, TraceParams(), monkeypatch)
    seen = []
    try:
        paths, reasons = tracer.trace(np.array([[50.,50,50]]), np.array([[1.,0,0]]),
                                      on_decision=lambda i,s: seen.append(s))
    finally:
        tracer.close()
    assert reasons == ['confidence'] and len(paths[0]) == 1
    assert seen[0]['would_stop'] and not seen[0]['hmask'].any()
    np.testing.assert_allclose(seen[0]['frame'][:,2], [1,0,0])


def test_adaptive_commit_and_bounded_exploration(monkeypatch):
    tracer = fake_tracer([.9,.85,.2,.1], TraceParams(max_len=8), monkeypatch)
    seen = []
    try:
        paths, reasons = tracer.trace(np.array([[50.,50,50]]), np.array([[1.,0,0]]),
                                      on_decision=lambda i,s: seen.append(s))
    finally:
        tracer.close()
    assert len(seen) == 2 and all(s['n_commit'] == 2 for s in seen)
    assert arclength(paths[0])[-1] == 8 and reasons == ['max_len']
    tracer = fake_tracer([.2]*4, TraceParams(explore_calls=3), monkeypatch)
    seen = []
    try:
        paths, reasons = tracer.trace(np.array([[50.,50,50]]), np.array([[1.,0,0]]),
                                      on_decision=lambda i,s: seen.append(s))
    finally:
        tracer.close()
    assert len(seen) == 4 and all(s['exploratory'] for s in seen)
    assert arclength(paths[0])[-1] == 6 and reasons == ['exploration_limit']


def test_every_interior_line_vertex_is_retained_and_review_tag_has_no_effect(tmp_path):
    raw = write_fiber(tmp_path/'f.json')
    # A bend that uniform resampling could skip; it is not a control point.
    raw['line_points'][8] = [8.15, .35, 0.]
    (tmp_path/'f.json').write_text(json.dumps(raw))
    f, = D.load_fibers(str(tmp_path), grid_scale=1, spacing=2.5)
    interior = np.asarray(raw['line_points'][5:34])
    for point in interior:
        assert np.any(np.linalg.norm(f.points-point, axis=-1) < 1e-12)
    before = D.make_sample(f, 8, False, sample_config(), np.random.default_rng(5))
    raw['tags'] = ['reviewed']
    (tmp_path/'f.json').write_text(json.dumps(raw))
    tagged, = D.load_fibers(str(tmp_path), grid_scale=1, spacing=2.5)
    after = D.make_sample(tagged, 8, False, sample_config(), np.random.default_rng(5))
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_spatial_split_is_invariant_to_dense_annotation_sampling():
    f = fiber(length=300)
    f.points = f.points[:, [1, 2, 0]]
    coarse = replace(f, points=f.points[::10], s=f.s[::10])
    assert bool(D.split_fibers([f], D.ZBand(149,151))[1])
    assert bool(D.split_fibers([coarse], D.ZBand(149,151))[1])


@pytest.mark.parametrize('hard', [False, True])
def test_replay_outside_training_holdout_is_rejected(monkeypatch, hard):
    f = fiber(length=300)
    op = make_states(f, z=550, offtrack=hard)
    monkeypatch.setattr(D, 'FiberVolume', lambda *a, **kw: object())
    monkeypatch.setattr(D, 'collate_with_volume', lambda items, *args: items)
    ds = D.FollowDataset([f], FiberVolumeSpec('unused'), sample_config(), D.ZBand(500,600),
                         chunk=1, onpolicy=[op], onpolicy_prob=0 if hard else 1, hard_prob=1 if hard else 0)
    items = next(iter(ds))
    assert all(D.training_state_allowed(it, ds.cfg.crop, ds.exclude) for it in items)
    assert all(it['pos'][2] < 100 for it in items)


def test_collector_stops_on_holdout_and_preserves_original_frame():
    f = fiber(length=600)
    f.points = f.points[:, [1,2,0]]
    c = DecisionCollector(f, 0, 0, 1, sample_config(), D.ZBand(200,300))
    d = decision_at(0,0)
    d['frame'] = frame_from_heading(np.array([0.,0,1]))
    assert c(d)
    np.testing.assert_array_equal(c.rows[0]['frame'], d['frame'])
    d['pos'] = np.array([0.,0,180.])
    d['last_segment'] = np.c_[np.zeros(181), np.zeros(181), np.arange(181)]
    d['travelled'] = 180
    assert c(d) is False


def test_online_collection_does_not_wait_and_publishes_only_complete_caches(tmp_path, monkeypatch):
    import vesuvius.neural_tracing.fiber_follow.online as online
    class Process:
        returncode = None
        def poll(self): return self.returncode
    process = Process()
    commands = []
    monkeypatch.setattr(online.subprocess, 'Popen', lambda cmd, **kw: commands.append(cmd) or process)
    manager = OnlineCollector(tmp_path/'online', 'fibers', [100,200], 'cpu', every=2, replay_keep=2)
    saved = []
    assert not manager.launch(1, saved.append)
    assert manager.launch(2, saved.append)
    assert manager.poll() is None
    assert not manager.launch(4, saved.append)  # collector busy; trainer keeps its optimizer
    assert len(saved) == 1 and json.loads(manager.index.read_text()) == []
    states = make_states(fiber())
    states.provenance['step'] = 2
    states.save(manager.output)
    process.returncode = 0
    event = manager.poll()
    assert event['dagger_source_step'] == 2 and event['dagger_states'] == 1
    assert len(json.loads(manager.index.read_text())) == 1
    assert '--checkpoint' in commands[0]
    manager.close()


def test_uniform_spatial_sampling_uses_metric_coordinates():
    model = FollowNet(small_config())
    grid = torch.from_numpy(D.crop_local_grid(model.crop)).float()
    # A coordinate ramp has a known value at every integer forward plane.
    volume = grid.permute(3,0,1,2)[None]
    points = torch.tensor([[[[[1.,2.,2.]], [[-2.,1.,6.]]]]])
    sampled = torch.nn.functional.grid_sample(volume, model.sampling_grid(points), align_corners=True)
    actual = sampled.permute(0,2,3,4,1)
    torch.testing.assert_close(actual, points, atol=1e-5, rtol=1e-5)


def test_plane_teacher_intersects_original_segments_without_smoothing_vertices():
    points = np.array([[0.,0,0], [1.,1,.1], [0.,0,.2], [0.,0,2.]])
    s = arclength(points)
    ab, mask = D.plane_targets(points, s, 0., s[-1], np.zeros(3), np.eye(3), np.array([.1, 2.]))
    assert mask.all()
    np.testing.assert_allclose(ab[0], [1.,1.])


def test_confidence_head_sees_early_mistakes_at_every_later_horizon():
    """Two locally identical suffixes still need opposite prefix confidence labels."""
    torch.manual_seed(19)
    cfg = replace(small_config(), depth=40, n_future=16)
    model = FollowNet(cfg)
    model.eval()
    candidates = straight_candidates(k=16)
    candidates[0, 1, 0, 0] = 4.0  # leaves GT at the first point, then returns
    candidates.requires_grad_()
    features = torch.randn(1, cfg.widths[0], cfg.depth, cfg.width, cfg.width)
    batch = dict(dense_ab=torch.zeros(1, 61, 2), dense_mask=torch.ones(1, 61),
                 end_local=torch.tensor([[0., 0., 100.]]), endpoint_known=torch.zeros(1), offtrack=torch.zeros(1))
    labels, mask, *_ = candidate_labels(candidates.detach(), batch)
    assert labels[0, 0].all() and not labels[0, 1].any() and mask.all()
    _, logits = model.score_candidates(features, candidates)
    # The old local-only head produced bit-identical predictions from horizon 5 onward.
    assert (logits[0, 0, 4:]-logits[0, 1, 4:]).abs().min().item() > 1e-8
    gradient = torch.autograd.grad(logits[0, 1, -1], candidates)[0]
    assert gradient[0, 1, 0, :2].abs().sum().item() > 1e-10
    # Candidate states are independent; no recurrence leaks between paths.
    assert gradient[0, 0].abs().sum().item() == 0


def test_prefix_context_gets_supervised_gradients_and_resets_between_calls():
    torch.manual_seed(20)
    cfg = small_config()
    model = FollowNet(cfg)
    features = torch.randn(1, cfg.widths[0], cfg.depth, cfg.width, cfg.width)
    candidates = straight_candidates()
    candidates[0, 1, 0, 0] = 4.
    b = target_batch()
    target, mask, *_ = candidate_labels(candidates, b)
    _, first = model.score_candidates(features, candidates)
    _, second = model.score_candidates(features, candidates)
    torch.testing.assert_close(first, second)
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(first, target, reduction='none')*mask).mean()
    loss.backward()
    for p in model.prefix_context.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_confidence_threshold_default_is_shared_by_rollout_and_collection(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.trace import DEFAULT_CONFIDENCE
    assert TraceParams().confidence == DEFAULT_CONFIDENCE == .7
    collector = OnlineCollector(tmp_path/'collector', 'fibers', [100, 200], 'cpu')
    assert collector.confidence == DEFAULT_CONFIDENCE
    collector.close()


def test_default_cube_has_uniform_spacing_and_encloses_proposal_stencil():
    cfg = FollowNetConfig()
    model = FollowNet(cfg)
    crop = D.SampleConfig().crop
    assert model.crop == crop == CropSpec()
    assert (crop.depth, crop.width, crop.behind) == (64, 64, 16)
    grid = D.crop_local_grid(crop)
    for axis, component in [(0, 2), (1, 1), (2, 0)]:
        np.testing.assert_allclose(np.diff(grid, axis=axis)[..., component], 1.)
    np.testing.assert_allclose(grid[0, 0, 0], [-31.5, -31.5, -16])
    np.testing.assert_allclose(grid[-1, -1, -1], [31.5, 31.5, 47])
    assert model.plane_grid.shape == (16, 61, 61, 3)
    assert model.plane_grid[..., :2].abs().max() == 30
    assert model.sampling_grid(model.plane_grid[..., None, :]+model.stencil).abs().max() <= 1
    # The enclosing read block must contain all cube corners in any orientation.
    for heading in [np.array([1., 0, 0]), np.array([1., 2, 3])]:
        frame = frame_from_heading(heading)
        pos = np.array([12.3, 24.4, 36.5])
        corners = grid[np.ix_([0, 63], [0, 63], [0, 63])].reshape(-1, 3)
        start = D.block_start(pos, frame, crop)[::-1]
        indices = corners @ frame.T + pos-start
        assert indices.min() >= 0 and indices.max() < crop.block_size-1


def test_proposals_and_loss_cover_visible_targets_beyond_previous_range():
    from vesuvius.neural_tracing.fiber_follow.model import decode_candidates
    from vesuvius.neural_tracing.fiber_follow.supervision import heatmap_loss
    cfg = replace(FollowNetConfig(), n_candidates=1, peaks_per_plane=1)
    logits = torch.full((1, 16, 61, 61), -20.)
    logits[:, :, 30, 52] = 10.  # x = +22 voxels, well outside the former ±12
    paths = decode_candidates(logits, cfg)
    torch.testing.assert_close(paths[0, 0, :, 0], torch.full((16,), 22.))
    target = torch.tensor([22., 0.]).expand(1, 16, 2)
    uniform = torch.zeros_like(logits, requires_grad=True)
    loss = heatmap_loss(uniform, target, torch.ones(1, 16), cfg)
    loss.backward()
    assert loss.item() > 0 and uniform.grad[0, 0, 30, 52] < 0
    # Missing annotation and genuinely unrepresentable targets add no position loss.
    assert heatmap_loss(uniform, target, torch.zeros(1, 16), cfg) == 0
    assert heatmap_loss(uniform, target+100, torch.ones(1, 16), cfg) == 0


def test_proposal_support_cannot_exceed_input_crop():
    with pytest.raises(ValueError, match='support must fit'):
        FollowNet(replace(small_config(), heat_bins=11))


def test_batch_diagnostic_keeps_crop_bounds_and_marks_outside_gt(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_follow import diag
    crop = CropSpec(depth=12, width=9, behind=2)
    x = torch.zeros(1, 8, 12, 9, 9)
    gt = torch.tensor([[[0., 0., 2.], [20., 0., 4.]]])
    pred = torch.tensor([[[0., 0., 2.], [1., 0., 4.]]])
    captured = []
    close = diag.plt.close
    monkeypatch.setattr(diag.plt, 'close', lambda fig: captured.append(fig))
    diag.plot_batch(x, pred, gt, torch.ones(1, 2), crop, tmp_path/'batch.png', heat_half=np.array([3.,3.]))
    fig = captured[-1]
    for ax in fig.axes:
        assert ax.get_xlim() == (-.5, 8.5) and ax.get_ylim() == (-.5, 11.5)
        assert any(line.get_color() == 'magenta' and len(line.get_xdata()) for line in ax.lines)
    assert 'GT outside' in fig.axes[0].get_title()
    close(fig)


def test_channels_last_layout_preserves_predictions_and_supervised_gradients():
    import copy
    from vesuvius.neural_tracing.fiber_follow.model import prepare_model
    torch.manual_seed(452)
    cfg = small_config()
    reference = prepare_model(FollowNet(cfg), 'cpu').eval()
    alternate = copy.deepcopy(reference)
    torch.nn.utils.convert_conv3d_weight_memory_format(alternate, torch.channels_last_3d)
    x = torch.randn(1, 8, cfg.depth, cfg.width, cfg.width)
    hist = torch.zeros(1, 4, 3)
    mask = torch.ones(1, 4)
    batch = target_batch()
    extra = teacher_candidates(batch, cfg)
    outputs = [m(x, hist, mask, extra) for m in (reference, alternate)]
    for key in ('heatmap', 'candidates', 'ranks', 'confidence_logits'):
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=2e-5, atol=2e-6)
    for model, output in zip((reference, alternate), outputs):
        loss, _ = loss_fn(output, batch, cfg)
        loss.backward()
    for a, b in zip(reference.parameters(), alternate.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6)



def test_replay_preserves_endpoint_arc_precision_and_rebuilds_rounded_cache(tmp_path):
    f = fiber()
    f.points *= 1.00000006
    f.s = arclength(f.points)
    assert float(np.float32(f.length)) > f.length
    op = make_states(f)
    op.t = np.array([f.length], dtype=np.float64)
    path = tmp_path / 'endpoint.npz'
    op.save(path)
    loaded = D.OnPolicyStates.load(path)
    loaded.validate_fibers([f])
    assert loaded.t.dtype == np.float64
    assert loaded.t[0] == f.length
    assert loaded.pos.dtype == np.float32

    # Simulate the old derived cache while retaining the intact source archive.
    directory = tmp_path / 'endpoint_mmap_v3'
    del loaded
    np.save(directory / 't.npy', op.t.astype(np.float32))
    metadata_path = directory / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.pop('mmap_t_dtype')
    metadata_path.write_text(json.dumps(metadata))
    rebuilt = D.OnPolicyStates.load(path)
    rebuilt.validate_fibers([f])
    assert rebuilt.t[0] == f.length
    D.OnPolicyStates.load(directory).validate_fibers([f])
    pickle.loads(pickle.dumps(rebuilt)).validate_fibers([f])


@pytest.mark.parametrize('arc', [-1., np.nan, np.inf, np.nextafter(100., np.inf)])
def test_replay_still_rejects_invalid_arc_positions_after_roundtrip(tmp_path, arc):
    f = fiber()
    op = make_states(f)
    op.t = np.array([arc], dtype=np.float64)
    path = tmp_path / 'invalid.npz'
    op.save(path)
    loaded = D.OnPolicyStates.load(path)
    with pytest.raises(ValueError, match='arc positions outside controlled spans'):
        loaded.validate_fibers([f])


def test_stop_patience_and_commit_floor_delay_the_confidence_stop(monkeypatch):
    # Every call is a would-stop (0.4 < 0.7). Patience 3 commits one point on
    # the first two calls and stops on the third; the floor vetoes that grace.
    tracer = fake_tracer([.4]*4, TraceParams(stop_patience=3), monkeypatch)
    seen = []
    try:
        paths, reasons = tracer.trace(np.array([[50.,50,50]]), np.array([[1.,0,0]]),
                                      on_decision=lambda i,s: seen.append(s))
    finally:
        tracer.close()
    assert reasons == ['confidence'] and len(seen) == 3 and all(s['would_stop'] for s in seen)
    assert arclength(paths[0])[-1] == 4  # two single-point (2-voxel) commits
    tracer = fake_tracer([.4]*4, TraceParams(stop_patience=3, commit_floor=.5), monkeypatch)
    try:
        paths, reasons = tracer.trace(np.array([[50.,50,50]]), np.array([[1.,0,0]]))
    finally:
        tracer.close()
    assert reasons == ['confidence'] and len(paths[0]) == 1
    with pytest.raises(ValueError):
        TraceParams(stop_patience=0)
