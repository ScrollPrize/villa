"""Acceptance checks for the aligned training/tracing contract.

State labels, the first connection, augmentation footprints, collection coverage, the
task budget with shared replay limits, export status records and the evaluation protocol.
"""
from dataclasses import replace
import json
import multiprocessing as mp
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from replay_fixtures import replay_states
from vesuvius.neural_tracing.fiber_follow.data import data as D
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength, frame_from_heading
from vesuvius.neural_tracing.fiber_follow.tracing.policy import commit_prefix, recovery_allowed
from vesuvius.neural_tracing.fiber_follow.data.state_labels import FOLLOWING, REASON, RECOVERABLE, REPLAY_CLASS, TERMINAL, UNKNOWN, TraceLabeler, classify, facts


def line_fiber(length=400, endpoints=(False, False), x=0.):
    z = np.arange(length+1.)
    p = np.c_[np.full(len(z), x), np.zeros(len(z)), z]
    return D.TracedFiber(f'line{x}', p, arclength(p), 'V', endpoint_stop=endpoints)


def targets(first, *, plane=1., endpoint_known=False, end=100., annotated=True, continues=True):
    planes = np.arange(1., 17.)
    planes[0] = plane
    ab = np.zeros((16, 2))
    ab[0] = first
    return dict(plane_ab=ab, plane_mask=np.full(16, float(annotated)), planes=planes, fmask=np.full(16, float(continues)),
                endpoint_known=float(endpoint_known), end_local=np.array([0., 0., end]))


def contract(distance=0., **flags):
    return facts(match_distance=distance, tolerance=1.5, max_recovery_distance=6., **flags)


# --------------------------------------------------------------------------- state labels

def test_reachability_and_certified_connection_boundaries_match_the_commit_limit():
    reach = np.sqrt(36-1)
    # A first point at the reach is exactly what the shared commit operation still permits.
    proposal = torch.tensor([[[reach-1e-6, 0., 1.]] + [[0., 0., float(k)] for k in range(2, 17)]])
    assert recovery_allowed(proposal, 6.).item()
    assert not recovery_allowed(proposal+torch.tensor([2e-3, 0., 0.]), 6.).item()
    counts, allowed = commit_prefix(proposal, torch.ones(1, 16), .5, 4, 6.)
    assert allowed.item() and counts.item() == 4
    certified = classify(targets([5.9, 0.]), contract(5.9))
    assert certified['supervision'] == RECOVERABLE and certified['geometry_valid'] and certified['confidence_valid']
    # Reachable within tolerance, but the annotated connection itself exceeds the limit:
    # proposals are still scored, no geometry is taught, and the state is not terminal.
    partial = classify(targets([reach+1.5-1e-3, 0.]), contract(5.))
    assert partial['supervision'] == UNKNOWN and partial['supervision_reason'] == REASON['unsupported_connection']
    assert not partial['geometry_valid'] and partial['confidence_valid'] and not partial['terminal']
    unreachable = classify(targets([reach+1.5+1e-3, 0.]), contract(5.))
    assert unreachable['supervision'] == TERMINAL and unreachable['supervision_reason'] == REASON['unreachable']
    # Distance alone is descriptive: d > 6 with a reachable crossing is not terminal.
    assert classify(targets([5., 0.]), contract(6.5))['supervision'] == RECOVERABLE


def test_switch_endpoint_unannotated_ambiguous_and_identity_semantics():
    assert classify(targets([0., 0.]), contract(0., switched=True))['supervision'] == TERMINAL
    assert classify(targets([0., 0.]), contract(0., beyond_end=True))['supervision_reason'] == REASON['unannotated']
    tagged = classify(targets([0., 0.], endpoint_known=True), contract(20., window_distance=20., beyond_end=True))
    assert tagged['supervision'] == TERMINAL and tagged['supervision_reason'] == REASON['endpoint']
    # No annotated continuation ahead: censored; an uncrossed plane with annotation ahead is unsupported.
    ended = classify(targets([0., 0.], annotated=False, continues=False), contract(0.))
    assert ended['supervision_reason'] == REASON['unannotated'] and not ended['confidence_valid']
    uncrossed = classify(targets([0., 0.], annotated=False), contract(0.))
    assert uncrossed['supervision_reason'] == REASON['unsupported_connection'] and not uncrossed['confidence_valid']
    unknown = classify(targets([4., 0.]), contract(4., match_ambiguous=True))
    assert unknown['supervision'] == UNKNOWN and not unknown['confidence_valid']
    # A displaced head without visible original-fiber evidence gets no supervision either way.
    hidden = classify(targets([4., 0.]), contract(4., identity_observable=False))
    assert hidden['supervision_reason'] == REASON['identity'] and not (hidden['geometry_valid'] or hidden['confidence_valid'])
    near = classify(targets([1., 0.]), contract(1., identity_observable=False))
    assert near['supervision'] == FOLLOWING


def test_certified_switch_stays_terminal_after_a_geometric_return():
    fiber = line_fiber()
    detector = SimpleNamespace(first_contact=lambda fi, t, segment: dict(distance=1., pos=segment[-1], bank_path='p',
                                                                        bank_run='r') if segment[-1][0] > 2 else None)
    labeler = TraceLabeler(fiber, 50., 1, tolerance=1.5, max_recovery_distance=6., fiber_idx=0, bank_detector=detector)
    labeler.observe(np.array([[0., 0., 50.]]), 0.)
    out = labeler.observe(np.array([[0., 0., 50.], [3., 0., 54.]]), 5.)
    assert out['switched']
    back = labeler.observe(np.array([[3., 0., 54.], [0., 0., 60.]]), 12.)
    assert back['switched'] and back['match_distance'] < 1
    assert classify(targets([0., 0.]), back)['supervision'] == TERMINAL


# --------------------------------------------------------------------------- first connection

def connector_batch(cfg, foreign_at):
    from label_fixtures import state_labels
    q = 4*(cfg.n_future-1)+1
    foreign = torch.zeros(1, cfg.fine.depth, cfg.fine.width, cfg.fine.width, dtype=torch.uint8)
    for a, b, c in foreign_at:
        centre = (cfg.fine.width-1)/2
        foreign[0, int(c+cfg.fine.behind), round(b+centre), round(a+centre)] = 1
    dense = torch.zeros(1, q, 2)
    dense[..., 0] = 4.
    return dict(dense_ab=dense, dense_mask=torch.ones(1, q), endpoint_known=torch.zeros(1),
                end_local=torch.zeros(1, 3), foreign=foreign, **state_labels(1))


def test_foreign_contact_between_origin_and_first_plane_rejects_a_correct_endpoint():
    from model_fixtures import config
    from vesuvius.neural_tracing.fiber_follow.train.supervision import connector_failures, geometry_mask, proposal_labels
    cfg = config()
    points = torch.zeros(1, cfg.n_future, 3)
    points[..., 0] = 4.
    points[..., 2] = torch.arange(1, cfg.n_future+1.)
    # The neighbor cell sits halfway along the connection, nowhere on the predicted path.
    crossing = connector_batch(cfg, [(2, 0, 0), (2, 0, 1)])
    clear = connector_batch(cfg, [(-6, 0, 0), (-6, 0, 1)])
    assert connector_failures(points[:, 0], crossing, cfg).item() and not connector_failures(points[:, 0], clear, cfg).item()
    labels, known, _, _ = proposal_labels(points, crossing, cfg, 1.5)
    assert known[0, 0] == 1 and labels[0].eq(0).all()
    labels, known, _, _ = proposal_labels(points, clear, cfg, 1.5)
    assert labels[0].eq(1).all()
    # The annotated target's own connection crosses the same neighbor: no geometry is taught.
    assert not geometry_mask(crossing, cfg).any() and geometry_mask(clear, cfg).any()
    # A foreign component under the third point, within tolerance of the annotation, ends the prefix there.
    under = connector_batch(cfg, [])
    near = points.clone()
    near[..., 0] = 4.5
    c, a = int(round(3+cfg.fine.behind)), int(round(4.5+(cfg.fine.width-1)/2))
    under['foreign'][0, c, :, a-1:a+2] = 1
    labels, known, _, _ = proposal_labels(near, under, cfg, 1.5)
    assert labels[0].tolist() == [1., 1., 0., 0.] and known[0].all()
    assert geometry_mask(under, cfg).any()


def test_displaced_origin_recovery_is_a_positive_continuation():
    fiber = line_fiber()
    cfg = D.SampleConfig(crop=CropSpec(depth=24, width=17, behind=8), n_history=32, n_future=16)
    pos = np.array([5., 0., 100.])
    item = D.label_state(fiber, pos, frame_from_heading(np.array([0., 0., 1.])), np.repeat(pos[None], 32, 0),
                         np.zeros(32), cfg, t=100., reverse=False, trace=contract(5.))
    assert item['supervision'] == RECOVERABLE and item['geometry_valid']
    batch = {k: torch.as_tensor(np.asarray(item[k]), dtype=torch.float32)[None] for k in
             ('dense_ab', 'dense_mask', 'endpoint_known', 'end_local', 'terminal', 'confidence_valid')}
    from vesuvius.neural_tracing.fiber_follow.data.labels import prefix_labels
    # Return straight to the annotated crossings from the displaced origin.
    back = torch.as_tensor(np.c_[item['plane_ab'], item['planes']], dtype=torch.float32)[None]
    assert np.linalg.norm(item['plane_ab'][0]) == pytest.approx(5.)
    labels, known, _ = prefix_labels(back, batch, 1.5, 6.)
    assert labels.eq(1).all() and known.eq(1).all()


# --------------------------------------------------------------------------- augmentation footprints

def test_recorded_frame_takes_its_roll_before_the_read_is_planned(monkeypatch):
    from model_fixtures import config
    from vesuvius.neural_tracing.fiber_follow.data.observations import IdentityObservationBuilder, IdentitySampling
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import FRAME_POLICY
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.oriented_seed_heading',
                        lambda vol, pos, family, direction: np.asarray(direction))
    cfg = config(fine=CropSpec(depth=120, width=104, behind=48, spacing=.5))
    fiber = line_fiber(800)
    builder = IdentityObservationBuilder(cfg, [fiber], IdentitySampling(), augment=True)
    sample = D.SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future,
                            startup_shares=(0., 0., 0., 1.), excursion_probability=0.)
    vol = SimpleNamespace(input_scale=2.)
    world = lambda i: dict(hist=i['hist_local'] @ i['frame'].T+i['pos'], fut=i['fut_local'] @ i['frame'].T+i['pos'],
                           plane=np.c_[i['plane_ab'], i['planes']] @ i['frame'].T+i['pos'])
    angles, moved = [], False
    for seed in range(40):
        item = D.make_sample(fiber, 400., False, sample, np.random.default_rng(seed))
        item.pop('_pending_seed_heading', None)
        item.update(fiber_ref=(0, 400., False), frame_policy=FRAME_POLICY)
        before, heading, unrolled = world(item), item['frame'][:, 2].copy(), item['frame'].copy()
        # A recorded frame takes its roll during preparation, before any footprint is planned.
        item = builder.prepare(item, fiber, np.random.default_rng(100+seed))
        assert 'roll_augmentation' not in item
        angles.append(item.get('roll_augmented', 0.))
        planned = next(iter(builder.prefetch_bounds(item, vol)))
        final = D.tight_block(item['pos'], item['frame'], cfg.fine, 2.)
        np.testing.assert_array_equal(planned[0], final[0])
        np.testing.assert_array_equal(planned[1], final[1])
        moved |= not np.array_equal(D.tight_block(item['pos'], unrolled, cfg.fine, 2.)[1], final[1])
        frame = item['frame'].copy()
        builder.finalize_frames([item], None)
        np.testing.assert_array_equal(item['frame'], frame)
        # Every local quantity is already expressed in the final frame.
        after = world(item)
        for key in before:
            np.testing.assert_allclose(after[key], before[key], atol=1e-9)
        np.testing.assert_allclose(item['frame'][:, 2], heading, atol=1e-12)
        np.testing.assert_allclose(after['fut'][:, :2], 0., atol=1e-9)
    assert moved  # an unrolled plan would not cover the final crop
    flipped = np.abs(np.angle(np.exp(1j*np.asarray(angles)))) > np.pi/2
    jitter = np.rad2deg(np.angle(np.exp(1j*(np.asarray(angles)+np.pi*flipped))))
    assert 8 < flipped.sum() < 32 and np.abs(jitter).max() <= 15+1e-9 and np.abs(jitter).std() > 1


# --------------------------------------------------------------------------- collection

def test_collection_takes_one_directed_episode_per_distinct_fiber_and_records_skips(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.tracing.collection import CoverageCursor, select_seeds
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import SeedHeadingError
    fibers = [line_fiber(200, x=10.*i) for i in range(6)]
    def heading(vol, pos, family, direction):
        if pos[0] == 30.:
            raise SeedHeadingError('CT seed context has no identifiable sheet normal')
        return np.asarray(direction)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.evaluation.seeds.oriented_seed_heading', heading)
    cursor = CoverageCursor(fibers, seed=4)
    rng = np.random.default_rng(0)
    first, skipped = select_seeds(fibers, None, cursor, 3, rng)
    second, more = select_seeds(fibers, None, cursor, 3, rng)
    for collection in (first, second):  # one directed episode per distinct fiber
        assert len({s['fiber'] for s in collection}) == len(collection) == 3
    assert {s['reason'] for s in skipped+more} == {'ct_heading'} and len(skipped+more) == 1
    taken = [s['fiber'] for s in first+second]
    assert set(taken[:5]) == {0, 1, 2, 4, 5}  # every eligible fiber before any repeat
    assert all(s['sign'] in (-1., 1.) and np.isclose(abs(s['heading'][2]), 1.) for s in first+second)
    # Every fiber once per epoch regardless of length; the next epoch traces each the other way.
    cursor = CoverageCursor([line_fiber(5000 if i == 9 else 350, x=10.*i) for i in range(10)], seed=3)
    epoch = [cursor.take(rng) for _ in range(10)]
    assert sorted(fi for fi, _ in epoch) == list(range(10))
    signs = dict(epoch)
    assert all(signs[fi] == -s for fi, s in (cursor.take(rng) for _ in range(10)))


def test_length_weighted_coverage_favors_long_fibers_but_still_visits_each_once_per_epoch(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.tracing.collection import CoverageCursor
    fibers = [line_fiber(3000 if i < 5 else 300, x=10.*i) for i in range(50)]
    first = [CoverageCursor(fibers, seed=s, length_power=3.).state['order'][:5] for s in range(40)]
    assert np.mean([fi < 5 for order in first for fi in order]) > .8
    uniform = [CoverageCursor(fibers, seed=s).state['order'][:5] for s in range(40)]
    assert np.mean([fi < 5 for order in uniform for fi in order]) < .3
    cursor = CoverageCursor(fibers, seed=1, length_power=3.)
    rng = np.random.default_rng(0)
    assert sorted(cursor.take(rng)[0] for _ in range(50)) == list(range(50))
    cursor.save(tmp_path/'coverage.json')
    assert CoverageCursor.load(fibers, tmp_path/'coverage.json', 1, 3.).state == cursor.state
    with pytest.raises(ValueError, match='length weighting'):
        CoverageCursor.load(fibers, tmp_path/'coverage.json', 1, 0.)


def follow(x, travelled, previous=None, y=0., stop=False):
    pos = np.array([y, 0., x])
    segment = np.array([pos]) if previous is None else np.array([previous, pos])
    return dict(pos=pos, frame=frame_from_heading(np.array([0., 0., 1.])), hist=np.zeros((32, 3)),
                hmask=np.zeros(32), would_stop=stop, n_commit=0 if stop else 4, points=np.zeros((16, 3)),
                confidence=np.ones(16), heading_start=0, travelled=travelled, seed_pos=segment[0],
                seed_tangent=np.array([0., 0., 1.]), seed_age=travelled, seed_valid=False, last_segment=segment)


def test_event_windows_survive_thinning_and_failures_keep_a_bounded_suffix():
    from vesuvius.neural_tracing.fiber_follow.tracing.collection import DecisionCollector
    cfg = D.SampleConfig(crop=CropSpec(depth=24, width=17, behind=8), n_history=32, n_future=16)
    c = DecisionCollector(line_fiber(600), 0, 50., 1, cfg, before=48., after=24., stride=16.)
    previous = None
    for k in range(40):  # ordinary following every 4 voxels
        z = 50.+4*k
        assert c(follow(z, 4.*k, previous))
        previous = np.array([0., 0., z])
    for k in range(40, 52):  # 10 voxels off: unreachable, terminal
        z = 50.+4*k
        row = follow(z, 4.*k, previous, y=10.)
        if not c(row):
            break
        previous = np.array([10., 0., z])
    assert c.censored == 'after_failure'
    kept = c.finish()
    terminal = [r for r in kept if r['supervision'] == TERMINAL]
    assert terminal and max(r['travelled'] for r in terminal)-terminal[0]['travelled'] <= 24.
    window = [r for r in kept if r['replay_class'] == REPLAY_CLASS['pre_excursion']]
    onset = terminal[0]['travelled']
    assert {r['travelled'] for r in window} == {t for t in 4.*np.arange(40) if onset-48 <= t < onset}
    ordinary = [r['travelled'] for r in kept if r['replay_class'] == REPLAY_CLASS['ordinary']]
    assert np.all(np.diff(ordinary) >= 16.)


# --------------------------------------------------------------------------- task budget and replay

def classed_cache(fiber, provenance=None):
    rows = []
    for episode, name in enumerate(('terminal', 'premature_stop', 'recoverable', 'pre_excursion', 'ordinary')):
        for k in range(3):
            rows.append(dict(t=50.+k, pos=np.array([0., 0., 50.+k]), hist=np.zeros((32, 3)), hmask=np.zeros(32),
                             episode=episode, event_id=0 if name != 'ordinary' else -1,
                             replay_class=REPLAY_CLASS[name], travelled=10.*k,
                             supervision=TERMINAL if name == 'terminal' else RECOVERABLE if name == 'recoverable' else FOLLOWING,
                             geometry_valid=name != 'terminal', would_stop=name == 'premature_stop'))
    return replay_states([fiber], rows, provenance=provenance)


def budget_dataset(fiber, budget, caches=()):
    cfg = D.SampleConfig(crop=CropSpec(depth=24, width=17, behind=8), n_history=32, n_future=16)
    ds = D.FollowDataset([fiber], SimpleNamespace(grid_scale=8., ct_grid_scale=8.), cfg, None, chunk=4,
                         onpolicy=list(caches), budget=budget)
    ds.set_step(1000)
    return ds


def test_replay_classes_fill_their_own_slots_and_missing_classes_fall_back_explicitly():
    fiber = line_fiber()
    shares = ['fresh=.2', 'live=0', 'dagger_pre_excursion=.2', 'dagger_recoverable=.2', 'dagger_terminal=.2',
              'dagger_premature_stop=.1', 'dagger_ordinary=.1', 'synthetic_terminal=0']
    ds = budget_dataset(fiber, D.TaskBudget.parse(shares, replay_event_cap=10**6), [classed_cache(fiber)])
    rng = np.random.default_rng(0)
    items = [ds.task_item(int(task), rng, []) for task in rng.choice(len(D.TASKS), size=600, p=ds.budget.shares)]
    for item in items:
        name = D.TASKS[item['task_requested']]
        if name.startswith('dagger_'):
            assert item['source'] == D.SOURCE['replay'] and item['task_delivered'] == item['task_requested']
            assert item['replay_class'] == REPLAY_CLASS[name[len('dagger_'):]]
    delivered = np.bincount([i['task_delivered'] for i in items], minlength=len(D.TASKS))/len(items)
    np.testing.assert_allclose(delivered, ds.budget.shares, atol=.05)
    # Without replay every DAgger slot is an explicit fresh fallback; terminal has no synthetic source here.
    empty = budget_dataset(fiber, ds.budget)
    item = empty.task_item(D.TASK['dagger_terminal'], rng, [])
    assert item['task_delivered'] == D.TASK['fresh'] and D.FALLBACKS[item['task_fallback']] == 'fresh'
    recovery = empty.task_item(D.TASK['dagger_recoverable'], rng, [])
    assert recovery['startup'] == D.STARTUP_CATEGORIES.index('established') and recovery['excursion']


def test_synthetic_terminal_fallback_respects_its_cap():
    fiber = line_fiber()
    ds = budget_dataset(fiber, D.TaskBudget(terminal_fallback_cap=.25))
    calls = []
    ds.batch_builder = SimpleNamespace(synthetic_terminal=lambda cfg, rng: calls.append(1) or dict(
        D.make_sample(fiber, 100., False, ds.cfg, rng), fiber_ref=(0, 100., False), identity_evidence=True))
    rng = np.random.default_rng(1)
    items = [ds.task_item(D.TASK['dagger_terminal'], rng, []) for _ in range(40)]
    synthetic = sum(D.FALLBACKS[i['task_fallback']] == 'synthetic' for i in items)
    assert 0 < synthetic <= .25*40 and synthetic == len(calls)
    assert all(D.FALLBACKS[i['task_fallback']] in ('synthetic', 'fresh') for i in items)


def test_age_ceiling_and_fiber_first_sampling():
    fiber = line_fiber()
    old = classed_cache(fiber, provenance=dict(step=0))
    ds = budget_dataset(fiber, D.TaskBudget(replay_max_age=500), [old])
    assert ds.replay_draw('terminal', np.random.default_rng(0)) is None  # 1000 updates old
    ds.set_step(400)
    assert ds.replay_draw('terminal', np.random.default_rng(0)) is not None
    # Fibers are drawn first: one dense event cannot outweigh another fiber's single row.
    other = line_fiber(x=30.)
    rows = [dict(fiber_idx=0, t=50.+k, pos=np.array([0., 0., 50.+k]), hist=np.zeros((32, 3)), hmask=np.zeros(32),
                 episode=0, event_id=0, replay_class=REPLAY_CLASS['terminal'], supervision=TERMINAL) for k in range(50)]
    rows.append(dict(fiber_idx=1, t=50., pos=np.array([30., 0., 50.]), hist=np.zeros((32, 3)), hmask=np.zeros(32),
                     episode=1, event_id=0, replay_class=REPLAY_CLASS['terminal'], supervision=TERMINAL))
    index = D.ReplayIndex([replay_states([fiber, other], rows)])
    rng = np.random.default_rng(3)
    fibers = [int(index.caches[0].fiber_idx[index.draw('terminal', rng, {0})[3][0]]) for _ in range(400)]
    assert .4 < np.mean(fibers) < .6


def _claim_in_child(ds, key, result):
    result.value = int(ds.claim(key))


def test_event_cap_is_shared_by_loader_workers():
    fiber = line_fiber()
    ds = budget_dataset(fiber, D.TaskBudget(replay_event_cap=2))
    context = mp.get_context('fork')
    assert ds.claim(17)
    result = context.Value('i', -1)
    worker = context.Process(target=_claim_in_child, args=(ds, 17, result))
    worker.start()
    worker.join(timeout=30)
    assert result.value == 1
    assert not ds.claim(17) and ds.max_event_reuse() == 2


def test_failed_ct_seed_is_redrawn_within_its_task(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import SeedHeadingError
    fiber = line_fiber()
    ds = budget_dataset(fiber, D.TaskBudget())
    calls = []
    def heading(vol, pos, family, direction):
        calls.append(1)
        if len(calls) == 1:
            raise SeedHeadingError('CT seed context has no identifiable sheet normal')
        return np.asarray(direction)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.oriented_seed_heading', heading)
    rng = np.random.default_rng(5)
    items = [ds.task_item(D.TASK['fresh'], rng, [])]
    first = items[0]
    assert ds.resolve_seeds(items, None, rng, []) == 1
    assert items[0] is not first and items[0]['task_requested'] == D.TASK['fresh']
    assert 'seed_heading_family' not in items[0] and len(calls) == 2
    # Missing CT orientation raises; there is no annotation fallback.
    monkeypatch.undo()
    def unavailable(vol, pos, family):
        raise SeedHeadingError('CT seed context has no identifiable sheet normal')
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.ct_seed_heading', unavailable)
    seed_only = D.make_sample(fiber, 50., False, replace(ds.cfg, startup_shares=(1., 0., 0., 0.)), rng)
    with pytest.raises(SeedHeadingError):
        D.resolve_trace_seed(seed_only, vol=object())
    assert 'seed_heading_fallback' not in seed_only


# --------------------------------------------------------------------------- export and evaluation

def test_export_keeps_short_valid_traces_and_records_every_seed(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.tracing import inference as infer
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import SeedHeadingError
    def heading(vol, seed, family):
        if seed[0] > 50:
            raise SeedHeadingError('H/V is ambiguous for this sheet orientation')
        return np.array([0., 0., 1.])
    monkeypatch.setattr(infer, 'ct_seed_heading', heading)
    class Tracer:
        def trace(self, seeds, headings):
            paths = [np.array([s]) if s[0] < 5 else np.array([s, s+np.sign(h[2])*np.array([0., 0., 2.])])
                     for s, h in zip(seeds, headings)]
            return paths, ['confidence']*len(seeds)
    seeds = [np.array([0., 0., 0.]), np.array([10., 0., 0.]), np.array([60., 0., 0.])]
    written, statuses = infer.export_seeds(Tracer(), None, seeds, ['V']*3, tmp_path, grid_scale=1., batch=8,
                                           min_length=0., dedupe=0., cp_every=800., provenance=dict(policy='p'))
    assert [s['status'] for s in statuses] == ['seed_only', 'exported', 'unavailable']
    assert statuses[2]['reason'].startswith('H/V is ambiguous') and statuses[0]['stop_reasons'] == ['confidence']*2
    assert len(written) == 1 and json.loads((tmp_path/'seeds.json').read_text())['seeds'] == statuses
    _, filtered = infer.export_seeds(Tracer(), None, seeds[1:2], ['V'], tmp_path/'filtered', grid_scale=1., batch=8,
                                     min_length=10., dedupe=0., cp_every=800., provenance={})
    assert filtered[0]['status'] == 'filtered_min_length'


def test_geometric_outcomes_count_returns_without_erasing_the_first_departure():
    from vesuvius.neural_tracing.fiber_follow.evaluation.seeds import geometric_outcomes, score_trace
    fiber = line_fiber(600)
    z = np.arange(100., 400.)
    lateral = np.where((z > 150) & (z < 180), 4., 0.)+np.where(z > 300, 8., 0.)
    path = np.c_[lateral, np.zeros(len(z)), z]
    outcome = geometric_outcomes(path, fiber, 100., 1.)
    assert outcome['excursions'] == 2 and outcome['excursion_returns'] == 1
    assert outcome['distance_events'] == 1 and outcome['distance_event_returns'] == 0
    assert outcome['geometric_agreement'] == pytest.approx(299.-29.-99., abs=4.)
    first = score_trace(path, fiber, 100., 1.)
    assert first['diverged'] and first['correct'] < 60  # the strict first-departure metric stands


def test_departure_patience_is_path_length_not_point_count():
    from vesuvius.neural_tracing.fiber_follow.evaluation.seeds import score_trace
    fiber = line_fiber(600)

    def path(step, excursion):
        z = np.arange(100., 400.+1e-9, step)
        lateral = np.where((z >= 200) & (z < 200+excursion), 4., 0.)
        return np.c_[lateral, np.zeros(len(z)), z]
    for step in (1., .25):
        # 2 voxels beyond the tolerance is a brief excursion (8 points at 0.25-voxel spacing), not a departure.
        assert not score_trace(path(step, 2.), fiber, 100., 1.)['diverged']
        # 5 voxels is a departure at either sampling density, dated at the excursion's first point.
        long = score_trace(path(step, 5.), fiber, 100., 1.)
        assert long['diverged'] and long['correct'] == pytest.approx(100., abs=1.)
    assert score_trace(path(1., 5.), fiber, 100., 1., patience=10.)['diverged'] is False


def spiral_fiber(windings=3., radius=60., pitch=8.):
    """A fiber wound around the scroll axis: each winding passes ``pitch`` voxels outside the previous one."""
    theta = np.linspace(0., 2*np.pi*windings, 200000)
    r = radius+pitch*theta/(2*np.pi)
    dense = np.c_[r*np.cos(theta)+200., r*np.sin(theta)+200., np.full_like(theta, 50.)]
    arc = arclength(dense)
    s = np.arange(0., arc[-1], 1.)
    points = np.stack([np.interp(s, arc, dense[:, k]) for k in range(3)], -1)
    return D.TracedFiber('spiral', points, s, ''), r, theta, arc


def test_a_jump_onto_the_same_fibers_next_winding_is_a_departure():
    from vesuvius.neural_tracing.fiber_follow.evaluation.seeds import score_trace
    fiber, r, theta, arc = spiral_fiber()
    t0, hop = 100., 150.
    faithful = fiber.points[(fiber.s >= t0) & (fiber.s <= t0+400.)]
    full = score_trace(faithful, fiber, t0, 1.)
    assert not full['diverged'] and full['correct'] == pytest.approx(400., abs=1.)
    # Follow the fiber to arc 150, step 8 voxels outward onto its next winding and follow that winding onward.
    angle = np.interp(hop, arc, theta)
    outer = np.interp(angle+2*np.pi, theta, arc)  # the next winding at the same angle, about one turn later
    assert outer-hop > 300.
    on_next = fiber.points[(fiber.s >= outer) & (fiber.s <= outer+200.)]
    path = np.concatenate((fiber.points[(fiber.s >= t0) & (fiber.s <= hop)], on_next))
    jumped = score_trace(path, fiber, t0, 1.)
    assert jumped['diverged'] and jumped['correct'] == pytest.approx(hop-t0, abs=2.)
    # Progress (coverage) never jumps by a winding.
    assert jumped['followed'] <= hop-t0+2.


def test_history_on_the_same_fibers_other_winding_is_not_the_original_fiber():
    from vesuvius.neural_tracing.fiber_follow.data.observations import memory_distance
    fiber, r, theta, arc = spiral_fiber()
    t, age = 500., 40.
    behind = fiber.points[int(t-age)]
    assert memory_distance(fiber, t, False, behind, age) < .5
    # The same angle one winding out lies 8 voxels away in space but about one turn further along the fiber.
    angle = np.interp(t-age, arc, theta)
    outer = fiber.points[int(np.interp(angle+2*np.pi, theta, arc))]
    assert np.linalg.norm(outer-behind) < 9. and memory_distance(fiber, t, False, outer, age) > 7.
    inner = fiber.points[int(np.interp(angle-2*np.pi, theta, arc))]  # the previous winding
    assert memory_distance(fiber, t, False, inner, age) > 7.


def protocol_sources():
    return [dict(name='src', fibers=[line_fiber()], volume=None, detector=None,
                 manifest=dict(calibration=[dict(fiber=0, t=50., sign=1., pos=np.array([0., 0., 50.]),
                                                 heading=np.array([0., 0., 1.]))],
                               final=[dict(fiber=0, t=60., sign=1., pos=np.array([0., 0., 60.]),
                                           heading=np.array([0., 0., 1.]))]))]


def test_calibration_requires_95_percent_precision_and_final_needs_a_locked_policy(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.evaluation import evaluation
    cfg = SimpleNamespace(n_future=4, max_recovery_distance=6., recurrent_refinement_steps=0,
                          recent_history_points=4, future_step=1.)
    def loader(path, device):
        return SimpleNamespace(cfg=cfg), CropSpec(depth=8, width=5, behind=2), 4, None, dict(step=7, n_commit=4)
    seen = []
    class Tracer:
        def __init__(self, model, vol, crop, n_history, params, device):
            self.p = params
            seen.append(params.confidence)
            self.model, self.crop, self.n_history = SimpleNamespace(cfg=cfg), crop, n_history
        def trace(self, seeds, headings, on_decision=None):
            # Low thresholds go wrong after 20 voxels; high thresholds stay on the fiber.
            wrong = self.p.confidence < .8
            paths = [np.c_[np.where(np.arange(60.) > 20, 6., 0.) if wrong else np.zeros(60), np.zeros(60),
                           s[2]+np.arange(60.)] for s in seeds]
            return paths, ['max_len']*len(seeds)
        def close(self):
            pass
    args = evaluation.parser().parse_args(['calibrate', '--checkpoint', str(tmp_path/'ck.pt'), '--out', str(tmp_path/'cal'),
                                           '--thresholds', '.5', '.9', '--max-len', '100', '--device', 'cpu'])
    (tmp_path/'ck.pt').write_bytes(b'checkpoint')
    outcome = evaluation.calibrate(args, protocol_sources(), checkpoint_loader=loader, tracer_class=Tracer)
    assert outcome['selected'] == .9 and outcome['operating_policy']['confidence'] == .9
    assert all(r['length_precision'] < .95 for r in outcome['sweep'] if r['threshold'] == .5)
    with pytest.raises(ValueError, match='explicit --confidence or a calibration --policy'):
        evaluation.run(evaluation.parser().parse_args(['run', '--checkpoint', str(tmp_path/'ck.pt'),
                                                       '--out', str(tmp_path/'final.json')]),
                       protocol_sources(), checkpoint_loader=loader, tracer_class=Tracer)
    final = evaluation.run(evaluation.parser().parse_args(['run', '--checkpoint', str(tmp_path/'ck.pt'), '--device', 'cpu',
                                                           '--out', str(tmp_path/'final.json'), '--max-len', '100',
                                                           '--policy', str(tmp_path/'cal'/'selection.json')]),
                           protocol_sources(), checkpoint_loader=loader, tracer_class=Tracer)
    assert final['operating_policy']['confidence'] == .9 and {r['split'] for r in final['rows']} == {'final'}
    # Nothing qualifies: say so and lock no selection.
    args = evaluation.parser().parse_args(['calibrate', '--checkpoint', str(tmp_path/'ck.pt'), '--out', str(tmp_path/'none'),
                                           '--thresholds', '.5', '--max-len', '100', '--device', 'cpu'])
    assert evaluation.calibrate(args, protocol_sources(), checkpoint_loader=loader, tracer_class=Tracer) is None
    assert not (tmp_path/'none'/'selection.json').exists()
    report = evaluation.paired_report(dict(final, rows=final['rows']), final, repeats=20)
    assert report['all']['correct']['delta'] == 0 and report['all']['traces'] == 1
