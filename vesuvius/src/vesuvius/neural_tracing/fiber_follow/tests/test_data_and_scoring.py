"""Training fiber loading, holdout isolation, replay storage and collection."""
from dataclasses import asdict, replace
import json
import pickle

import numpy as np
import pytest

from vc3d_fiber_format import legacy_lasagna_segments
from vesuvius.neural_tracing.fiber_follow.data import data as D
from vesuvius.neural_tracing.fiber_follow.tracing.collection import DecisionCollector
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength, frame_from_heading
from vesuvius.neural_tracing.fiber_follow.train.online import OnlineCollector
from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec


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


def make_states(f, z=0):
    from replay_fixtures import replay_states
    return replay_states([f], [dict(t=50., pos=np.array([50., 0., z]), frame=frame_from_heading(np.array([1., 0, 0])),
                                    hist=np.zeros((4, 3)), hmask=np.zeros(4))])


def test_replay_uses_committed_prefix_and_supervises_only_beyond_tip():
    from replay_fixtures import replay_states
    f, cfg = fiber(), sample_config()
    reverse, sign = True, -1
    prefix = np.c_[50+sign*np.arange(-4, 1), np.full(5, .2), np.zeros(5)]
    frame = frame_from_heading(np.array([sign, 0., 0.]))
    op = replay_states([f], [dict(t=50., reverse=reverse, pos=prefix[-1], frame=frame, hist=prefix[-2::-1],
                                  hmask=np.ones(4), match_distance=.2, window_distance=.2, travelled=4.,
                                  seed_pos=prefix[0], seed_valid=True, seq_start=0, seq_end=5)],
                       # The archive can contain later points, but the row must never expose them.
                       track=np.concatenate((prefix, [[99., 7., 0.]])))
    ds = D.FollowDataset([f], None, cfg, None)
    item = ds.replay_item(op, 0, np.random.default_rng(1))
    np.testing.assert_array_equal(item['observed_path'], prefix)
    np.testing.assert_array_equal(item['seed_pos'], prefix[0])
    assert item['seed_valid'] and item['source'] == D.SOURCE['replay'] and item['source_step'] == 1000
    np.testing.assert_allclose(item['hist_local'] @ frame.T+item['pos'], prefix[-2::-1])
    target = item['fut_local'] @ frame.T+item['pos']
    np.testing.assert_allclose(target[:, 0], 50+sign*cfg.future_s)
    np.testing.assert_allclose(target[:, 1:], 0.)
    assert item['fmask'].all() and item['geometry_valid'] and not item['terminal']
    assert item['labeler_state']['t'] == 50. and item['labeler_state']['last_travelled'] == 4.


def decision_at(x, travelled, previous=None, would_stop=False, y=0.):
    pos = np.array([x, y, 0.])
    segment = np.array([pos]) if previous is None else np.array([previous, pos])
    return dict(pos=pos, frame=frame_from_heading(np.array([1., 0, 0])), hist=np.zeros((4, 3)),
                hmask=np.zeros(4), would_stop=would_stop, n_commit=0 if would_stop else 4,
                points=np.zeros((4, 3)), confidence=np.ones(4), heading_start=0, travelled=travelled,
                seed_pos=segment[0], seed_tangent=np.array([1., 0, 0]), seed_age=travelled, seed_valid=False,
                last_segment=segment)


def test_loader_trims_even_reviewed_tails_and_retains_controls(tmp_path):
    write_fiber(tmp_path / 'f.json', reviewed=True, reverse_line=True)
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


def test_loader_rejects_seed_only_and_unanchored_fibers(tmp_path):
    write_fiber(tmp_path / 'f.json', single_control=True)
    assert D.load_fibers(str(tmp_path), grid_scale=1) == []  # a seed alone is not supervision
    # Unanchored controls fail instead of snapping to the nearest winding.
    raw = write_fiber(tmp_path / 'f.json')
    raw['control_points'][1] = [17., 1., 0.]
    (tmp_path / 'f.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='anchored'):
        D.load_fibers(str(tmp_path), grid_scale=1)


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
        if not key.startswith('_'):  # private references to the loaded fiber object
            np.testing.assert_array_equal(before[key], after[key])


def test_spatial_split_uses_controlled_geometry_independent_of_sampling():
    centred = fiber(z=150)
    assert D.split_fibers([centred], D.ZBand(100, 200)) == ([], [centred])
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
    # Long history, supervision and plane targets inside the band are rejected too.
    for key, mask in (('hist_local', 'hmask'), ('fut_local', 'fmask')):
        item = safe_item(0)
        item[key], item[mask] = np.array([[0., 0., 150.]]), np.ones(1)
        assert not D.training_state_allowed(item, CropSpec(), band)
    item = safe_item(0)
    item.update(plane_ab=np.zeros((1, 2)), plane_mask=np.ones(1), planes=np.array([150.]))
    assert not D.training_state_allowed(item, CropSpec(), band)


def test_plane_teacher_intersects_original_segments_without_smoothing_vertices():
    points = np.array([[0.,0,0], [1.,1,.1], [0.,0,.2], [0.,0,2.]])
    s = arclength(points)
    ab, mask = D.plane_targets(points, s, 0., s[-1], np.zeros(3), np.eye(3), np.array([.1, 2.]))
    assert mask.all()
    np.testing.assert_allclose(ab[0], [1.,1.])


def test_onpolicy_cache_requires_identity_and_the_whole_schema(tmp_path):
    path = tmp_path / 'old.npz'
    np.savez(path, pos=np.zeros((1, 3)))
    with pytest.raises(KeyError, match='__metadata__'):
        D.OnPolicyStates.load(path)
    assert not (tmp_path / 'old_mmap').exists()
    # A previous-pipeline cache (sticky offtrack, no supervision fields) is rejected outright.
    op = make_states(fiber())
    arrays = {k: getattr(op, k) for k in (*D.OnPolicyStates.FIELDS, *D.OnPolicyStates.TRACK)
              if k not in ('supervision', 'confidence_valid')}
    np.savez(path, __metadata__=json.dumps(dict(fibers=op.manifest, provenance=op.provenance)),
             offtrack=np.zeros(1, bool), **arrays)
    with pytest.raises(ValueError, match='lacks schema fields'):
        D.OnPolicyStates.load(path)
    with pytest.raises(ValueError, match='differ from the schema'):
        D.OnPolicyStates(manifest=op.manifest, provenance=op.provenance, offtrack=np.zeros(1, bool),
                         **{k: getattr(op, k) for k in (*D.OnPolicyStates.FIELDS, *D.OnPolicyStates.TRACK)})
    # Cache history must match the training configuration.
    with pytest.raises(ValueError, match='history length'):
        D.FollowDataset([fiber()], FiberVolumeSpec('unused'), D.SampleConfig(n_history=128), None, onpolicy=[op])


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


def test_collector_censors_unknown_boundary_and_keeps_the_stop_decision():
    from vesuvius.neural_tracing.fiber_follow.data.state_labels import FOLLOWING, REASON, REPLAY_CLASS, TERMINAL
    c = DecisionCollector(fiber(), 0, 50, 1, sample_config(), stride=16)
    assert c(decision_at(50, 0))
    assert c(decision_at(58, 8, [50., 0, 0], would_stop=True))
    stop = c.rows[-1]
    assert stop['hard'] and stop['would_stop'] and stop['event_id'] >= 0
    assert c(decision_at(96, 46, [58., 0, 0]))
    assert c(decision_at(102, 52, [96., 0, 0])) is False and c.censored == 'unannotated'
    kept = c.finish()
    assert [r['replay_class'] for r in kept] == [REPLAY_CLASS['ordinary'], REPLAY_CLASS['premature_stop'],
                                                REPLAY_CLASS['ordinary']]
    # A physical endpoint is terminal instead, distinct from the premature stop before it.
    c = DecisionCollector(fiber(endpoints=(True, True)), 0, 50, 1, sample_config(), stride=16)
    assert c(decision_at(50, 0, would_stop=True))
    assert c.rows[-1]['supervision'] == FOLLOWING and c.rows[-1]['geometry_valid']
    assert c(decision_at(102, 52, [50., 0, 0]))
    assert c.rows[-1]['supervision'] == TERMINAL and c.rows[-1]['supervision_reason'] == REASON['endpoint']
    assert [r['replay_class'] for r in c.finish()] == [REPLAY_CLASS['premature_stop'], REPLAY_CLASS['terminal']]


def test_departure_is_the_evaluation_rule_and_a_return_restores_supervision():
    # Three consecutive committed points beyond 3 voxels (score_trace tolerance and
    # patience), dated at the run's first point, including runs spanning commits.
    from vesuvius.neural_tracing.fiber_follow.data.state_labels import FOLLOWING, RECOVERABLE, REPLAY_CLASS
    c = DecisionCollector(fiber(length=300), 0, 50, 1, sample_config(), stride=16)
    assert c(decision_at(50, 0))
    # Isolated off-track points are not departures.
    probe = DecisionCollector(fiber(length=300), 0, 50, 1, sample_config(), stride=16)
    assert probe(decision_at(50, 0))
    path = np.array([[50., 0, 0], [51, 3.5, 0], [52, 3.5, 0], [53, 0, 0], [54, 3.5, 0], [55, 0, 0]])
    d = decision_at(55, float(arclength(path)[-1]))
    d['last_segment'] = path
    assert probe(d) and probe.labeler.departure_distance is None
    assert all(np.isnan(r['departure_distance']) for r in probe.rows)
    first = np.array([[50., 0, 0], [51, 0, 0], [52, 0, 0], [53, 3.5, 0], [54, 3.5, 0]])
    d = decision_at(54, float(arclength(first)[-1]), y=3.5)
    d['last_segment'] = first
    assert c(d) and not c.rows[-1]['bad_run'] >= 3 and np.isnan(c.rows[-1]['departure_distance'])
    assert c.rows[-1]['supervision'] == RECOVERABLE  # displaced now, but not yet a departure event
    second = np.array([[54., 3.5, 0], [55, 3.5, 0], [56, 3.5, 0]])
    d = decision_at(56, d['travelled']+2., y=3.5)
    d['last_segment'] = second
    assert c(d) and c.rows[-1]['departure_distance'] == pytest.approx(arclength(first[:4])[-1])
    assert c.rows[-1]['supervision'] == RECOVERABLE and c.rows[-1]['geometry_valid']
    assert c.rows[0]['pre_excursion'] and c.rows[0]['hard']
    # Back on the fiber: following again; the historical departure is retained.
    back = np.array([[56., 3.5, 0], [60, 0, 0], [70, 0, 0]])
    d = decision_at(70, d['travelled']+float(arclength(back)[-1]))
    d['last_segment'] = back
    assert c(d) and c.rows[-1]['supervision'] == FOLLOWING and c.rows[-1]['geometry_valid']
    assert np.isfinite(c.rows[-1]['departure_distance']) and c.rows[-1]['event_id'] == -1
    classes = [r['replay_class'] for r in c.finish()]
    assert classes == [REPLAY_CLASS['pre_excursion'], REPLAY_CLASS['recoverable'], REPLAY_CLASS['recoverable'],
                       REPLAY_CLASS['ordinary']]


def test_correspondence_progress_is_bounded_in_both_directions():
    """A head far from the fiber cannot jump along it: the window bounds each update."""
    from vesuvius.neural_tracing.fiber_follow.data.state_labels import MATCH_AHEAD, MATCH_BEHIND, TERMINAL
    c = DecisionCollector(fiber(length=300), 0, 50, 1, sample_config(), stride=16)
    assert c(decision_at(50, 0))
    # Travel 10 voxels but appear 200 voxels ahead and far off: progress grows by at most 10+32.
    far = decision_at(250, 10., [50., 0, 0], y=40.)
    assert c(far)
    assert 50. < c.labeler.t <= 50.+10.+MATCH_AHEAD
    assert c.rows[-1]['supervision'] == TERMINAL  # unreachable from 40 voxels away
    # Reversing direction moves progress back by at most MATCH_BEHIND.
    t = c.labeler.t
    back = decision_at(0, 20., [250., 40, 0], y=40.)
    assert c(back)
    assert t-MATCH_BEHIND <= c.labeler.t <= t


def test_collector_stops_on_holdout_and_preserves_original_frame():
    f = fiber(length=600)
    f.points = f.points[:, [1,2,0]]
    c = DecisionCollector(f, 0, 0, 1, sample_config(), D.ZBand(200,300))
    d = decision_at(0,0)
    d['pos'], d['last_segment'] = np.zeros(3), np.zeros((1, 3))
    d['frame'] = frame_from_heading(np.array([0.,0,1]))
    assert c(d)
    np.testing.assert_array_equal(c.rows[0]['frame'], d['frame'])
    d['pos'] = np.array([0.,0,180.])
    d['last_segment'] = np.c_[np.zeros(181), np.zeros(181), np.arange(181)]
    d['travelled'] = 180
    assert c(d) is False and c.censored == 'holdout'


def test_online_collection_does_not_wait_and_publishes_only_complete_caches(tmp_path, monkeypatch):
    import vesuvius.neural_tracing.fiber_follow.train.online as online
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
    states.provenance.update(step=2, supply={}, coverage={}, operating_policy={})
    states.save(manager.output)
    manager.output.with_suffix('.coverage.json').write_text('{}')
    process.returncode = 0
    event = manager.poll()
    assert event['dagger_source_step'] == 2 and event['dagger_states'] == 1
    assert len(json.loads(manager.index.read_text())) == 1
    assert '--checkpoint' in commands[0] and '--explore-calls' not in commands[0]
    assert commands[0][commands[0].index('--fibers-per-collection')+1] == '64'
    # Collection takes the checkpoint's operating policy (default confidence shared with rollout), no exploration.
    from vesuvius.neural_tracing.fiber_follow.tracing.trace import DEFAULT_CONFIDENCE, DEFAULT_GATE, DEFAULT_N_COMMIT
    assert (TraceParams().confidence, TraceParams().gate, TraceParams().n_commit) == (DEFAULT_CONFIDENCE, DEFAULT_GATE,
                                                                                      DEFAULT_N_COMMIT) == (.4, 'full', 8)
    assert manager.confidence is None and manager.settings()['exploration'] == 'none'
    manager.close()
