"""Training fiber loading, holdout isolation, replay storage and collection."""
from dataclasses import asdict, replace
import json
import pickle

import numpy as np
import pytest

from vc3d_fiber_format import legacy_lasagna_segments
from vesuvius.neural_tracing.fiber_follow.shared import data as D
from vesuvius.neural_tracing.fiber_follow.shared.collect import DecisionCollector
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength, frame_from_heading
from vesuvius.neural_tracing.fiber_follow.shared.online import OnlineCollector
from vesuvius.neural_tracing.fiber_follow.shared.trace import TraceParams
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec


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


def safe_item(z=0):
    return dict(pos=np.array([0., 0., z]), frame=np.eye(3),
                hist_local=np.zeros((1, 3)), hmask=np.zeros(1),
                fut_local=np.zeros((1, 3)), fmask=np.zeros(1))


def sample_config():
    return D.SampleConfig(crop=CropSpec(depth=12, width=9, behind=2), n_history=4, n_future=4)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('t', [0., 2.5, 60.])
def test_clean_gt_preserves_position_heading_and_all_available_history(reverse, t):
    from vesuvius.neural_tracing.fiber_follow.shared.geometry import interp_at, tangent_at
    u = np.linspace(0, 100, 401)
    points = np.c_[u, 4*np.sin(u/12), 2*np.cos(u/9)]
    f = D.TracedFiber('clean', points, arclength(points), 'H')
    cfg = replace(sample_config(), full_observed_history=True, no_history_prob=1.,
                  short_history_prob=1., history_jitter=20., history_drift=20., history_wobble=20.)
    p = points[::-1] if reverse else points
    s = f.s[-1]-f.s[::-1] if reverse else f.s
    for seed in [1, 19]:
        item = D.make_sample(f, t, reverse, cfg, np.random.default_rng(seed), perturb=False)
        np.testing.assert_allclose(item['pos'], interp_at(p, s, np.array([t]))[0], atol=1e-12)
        np.testing.assert_allclose(item['frame'][:, 2], tangent_at(p, s, t), atol=1e-12)
        back = t-np.arange(int(t/cfg.history_step), 0, -1)*cfg.history_step
        expected = interp_at(p, s, np.r_[back, t])
        np.testing.assert_allclose(item['observed_path'], expected, atol=1e-12)
        valid = item['hmask'].astype(bool)
        history = item['hist_local'][valid] @ item['frame'].T+item['pos']
        np.testing.assert_allclose(history, expected[:-1][::-1][:cfg.n_history], atol=1e-12)
        np.testing.assert_allclose(item['seed_pos'], expected[0], atol=1e-12)


def make_states(f, z=0, offtrack=False):
    return D.OnPolicyStates(manifest=D.fiber_manifest([f]),
        fiber_idx=[0], t=[50.], reverse=[False], pos=[[50., 0., z]],
        frame=[frame_from_heading(np.array([1., 0, 0]))], hist=np.zeros((1, 4, 3)), hmask=np.zeros((1, 4)),
        offtrack=[offtrack], hard=[offtrack], exploratory=[False])


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('t', [0., 2.5, 60.])
def test_light_gt_is_bounded_keeps_seed_full_history_and_gt_targets(reverse, t):
    f = fiber()
    cfg = replace(sample_config(), full_observed_history=True, no_history_prob=1.,
                  short_history_prob=1., history_jitter=20., history_drift=20., history_wobble=20.)
    clean = D.make_sample(f, t, reverse, cfg, np.random.default_rng(0), perturb=False)
    for seed in range(20):
        item = D.make_sample(f, t, reverse, cfg, np.random.default_rng(seed),
                             perturb=False, light_perturbation=(.5, 2.))
        np.testing.assert_array_equal(item['hmask'], clean['hmask'])
        np.testing.assert_allclose(item['seed_pos'], clean['seed_pos'], atol=1e-12, rtol=0)
        delta = item['observed_path']-clean['observed_path']
        assert np.linalg.norm(delta, axis=1).max() <= .5+1e-12
        assert np.linalg.norm(item['pos']-clean['pos']) <= .5+1e-12
        assert np.dot(item['frame'][:, 2], clean['frame'][:, 2]) >= np.cos(np.deg2rad(2.))-1e-12
        # Old observations, including the seed, are untouched; the tip moves smoothly.
        old = np.linalg.norm(clean['observed_path']-clean['pos'], axis=1) >= 32.
        np.testing.assert_array_equal(delta[old], np.zeros_like(delta[old]))
        target = item['fut_local'] @ item['frame'].T+item['pos']
        expected = clean['fut_local'] @ clean['frame'].T+clean['pos']
        np.testing.assert_allclose(target, expected, atol=1e-6)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('replay_options', [dict(correct_replay_only=True), dict(replay_continuation_fraction=.8)])
def test_correct_replay_uses_committed_prefix_and_supervises_only_beyond_tip(reverse, replay_options):
    from types import SimpleNamespace
    f, cfg = fiber(), sample_config()
    sign = -1 if reverse else 1
    prefix = np.c_[50+sign*np.arange(-4, 1), np.full(5, .2), np.zeros(5)]
    frame = frame_from_heading(np.array([sign, 0., 0.]))
    op = D.OnPolicyStates(manifest=D.fiber_manifest([f]), fiber_idx=[0], t=[50.], reverse=[reverse],
        pos=prefix[-1:], frame=frame[None], hist=prefix[-2::-1][None], hmask=np.ones((1, 4)),
        offtrack=[False], hard=[False], exploratory=[False], drift=[.2], travelled=[4.],
        seed_pos=prefix[:1], seed_valid=[True], seq_start=[0], seq_end=[5],
        # The archive can contain later points, but the row must never expose them.
        track_pos=np.concatenate((prefix, [[99., 7., 0.]])))

    def reject_replacement(*args):
        raise AssertionError('Correct replay must never be synthetically replaced')

    ds = D.FollowDataset([f], None, cfg, None, **replay_options,
        batch_builder=SimpleNamespace(replace_replay=reject_replacement))
    item = ds.replay_item((2, 0, op, 0), np.random.default_rng(1))
    np.testing.assert_array_equal(item['observed_path'], prefix)
    np.testing.assert_array_equal(item['seed_pos'], prefix[0])
    assert item['seed_valid']
    assert item['replay_correct_continuation']
    np.testing.assert_allclose(item['hist_local'] @ frame.T+item['pos'], prefix[-2::-1])
    target = item['fut_local'] @ frame.T+item['pos']
    np.testing.assert_allclose(target[:, 0], 50+sign*cfg.future_s)
    np.testing.assert_allclose(target[:, 1:], 0.)
    assert item['fmask'].all() and not item['offtrack']


def decision_at(x, travelled, previous=None, would_stop=False):
    pos = np.array([x, 0., 0.])
    return dict(pos=pos, frame=frame_from_heading(np.array([1., 0, 0])), hist=np.zeros((4, 3)),
                hmask=np.zeros(4), exploratory=False, would_stop=would_stop, travelled=travelled,
                last_segment=np.array([pos]) if previous is None else np.array([previous, pos]))


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


def test_holdout_filter_checks_plane_targets():
    item = safe_item(0)
    item.update(plane_ab=np.zeros((1, 2)), plane_mask=np.ones(1), planes=np.array([150.]))
    assert not D.training_state_allowed(item, CropSpec(), D.ZBand(100, 200))


def test_plane_teacher_intersects_original_segments_without_smoothing_vertices():
    points = np.array([[0.,0,0], [1.,1,.1], [0.,0,.2], [0.,0,2.]])
    s = arclength(points)
    ab, mask = D.plane_targets(points, s, 0., s[-1], np.zeros(3), np.eye(3), np.array([.1, 2.]))
    assert mask.all()
    np.testing.assert_allclose(ab[0], [1.,1.])


def test_onpolicy_cache_requires_identity(tmp_path):
    path = tmp_path / 'old.npz'
    np.savez(path, pos=np.zeros((1, 3)))
    with pytest.raises(KeyError, match='__metadata__'):
        D.OnPolicyStates.load(path)
    assert not (tmp_path / 'old_mmap_v5').exists()


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


def test_replay_preserves_endpoint_arc_precision(tmp_path):
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
    assert loaded.pos.dtype == np.float64
    pickle.loads(pickle.dumps(loaded)).validate_fibers([f])


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


def test_cache_history_must_match_training_configuration():
    f = fiber()
    with pytest.raises(ValueError, match='history length'):
        D.FollowDataset([f], FiberVolumeSpec('unused'), D.SampleConfig(n_history=128), None,
                        onpolicy=[make_states(f)])


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
    import vesuvius.neural_tracing.fiber_follow.shared.online as online
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


def test_confidence_threshold_default_is_shared_by_rollout_and_collection(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.shared.trace import DEFAULT_CONFIDENCE
    assert TraceParams().confidence == DEFAULT_CONFIDENCE == .5
    collector = OnlineCollector(tmp_path/'collector', 'fibers', [100, 200], 'cpu')
    assert collector.confidence == DEFAULT_CONFIDENCE
    collector.close()
