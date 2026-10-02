"""Training decisions are built like tracing decisions: simulated traces, CT seed headings, roll augmentation."""
from dataclasses import replace

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.shared import data as D
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength, frame_from_heading, interp_at, normalize
from vesuvius.neural_tracing.fiber_follow.shared.heading import FRAME_POLICY, heading_free_bounds, trace_heading
from vesuvius.neural_tracing.fiber_follow.shared.trace import trace_history


def curved_fiber():
    u = np.linspace(0, 100, 401)
    points = np.c_[u, 4*np.sin(u/12), 2*np.cos(u/9)]
    return D.TracedFiber('curved', points, arclength(points), 'H')


def straight_fiber(length):
    p = np.c_[np.arange(length+1.), np.zeros(length+1), np.zeros(length+1)]
    return D.TracedFiber('straight', p, arclength(p), 'H')


def sample_config(**kwargs):
    return replace(D.SampleConfig(crop=CropSpec(depth=12, width=9, behind=2), n_history=32, n_future=4), **kwargs)


def lateral_tangents(p, s, arcs):
    tangent = interp_at(p, s, np.clip(arcs+3., 0., s[-1]))-interp_at(p, s, np.clip(arcs-3., 0., s[-1]))
    return tangent/np.linalg.norm(tangent, axis=1, keepdims=True)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('t', [0., 2.5, 60.])
def test_simulated_trace_is_the_tracer_state_of_its_observed_path(reverse, t):
    f, cfg = curved_fiber(), sample_config(startup_shares=(.2, .15, .15, .5), excursion_probability=0.)
    p, s = D.traversal_curve(f, reverse)
    for seed in range(12):
        item = D.make_sample(f, t, reverse, cfg, np.random.default_rng(seed))
        path = item['observed_path']
        np.testing.assert_array_equal(item['pos'], path[-1])
        np.testing.assert_array_equal(item['seed_pos'], path[0])
        assert item['seed_valid'] and item['seed_age'] == pytest.approx(arclength(path)[-1])
        # The trace starts on the annotation; its error is lateral to GT and smooth.
        arcs = t-np.arange(len(path)-1, -1, -1)*cfg.history_step
        residual = path-interp_at(p, s, arcs)
        np.testing.assert_allclose(residual[0], 0., atol=1e-12)
        np.testing.assert_allclose((residual*lateral_tangents(p, s, arcs)).sum(-1), 0., atol=1e-9)
        # Crop heading and history are the tracer's own functions of this path.
        np.testing.assert_allclose(item['frame'][:, 2], trace_heading(path, 0, item['seed_tangent']), atol=1e-12)
        hist, mask = trace_history(list(path), cfg.n_history)
        np.testing.assert_array_equal(item['hmask'], mask)
        valid = mask > 0
        np.testing.assert_allclose(item['hist_local'][valid] @ item['frame'].T+item['pos'], hist[valid], atol=1e-9)
        assert ('_pending_seed_heading' in item) == (arclength(path)[-1] < 12-1e-8)
        assert item['seed_heading_family'] == 'H'
        # Labels remain the GT continuation, now from the offset head.
        future = interp_at(p, s, np.clip(t+cfg.future_s, 0, s[-1]))
        np.testing.assert_allclose(item['fut_local'] @ item['frame'].T+item['pos'], future, atol=1e-9)


def test_trace_starts_follow_the_configured_shares():
    f = straight_fiber(400)
    seed_only = D.make_sample(f, 300., False, sample_config(startup_shares=(1., 0., 0., 0.)), np.random.default_rng(0))
    assert len(seed_only['observed_path']) == 1 and not seed_only['hmask'].any() and seed_only['seed_age'] == 0
    assert '_pending_seed_heading' in seed_only and seed_only['startup'] == 0
    lengths = [D.make_sample(f, 300., False, sample_config(startup_shares=(0., .5, .5, 0.)),
                             np.random.default_rng(seed))['trace_prefix_length'] for seed in range(200)]
    assert 1 <= min(lengths) and max(lengths) <= 32 and len(set(lengths)) > 20
    # Default draws: 15% seed-only, 17% 1-8, 17% 9-32 requested voxels, 51% uniform available history.
    rng = np.random.default_rng(1)
    categories = np.bincount([D.make_sample(f, 300., False, sample_config(), rng)['startup'] for _ in range(4000)],
                             minlength=4)/4000
    np.testing.assert_allclose(categories, (.15, .17, .17, .51), atol=.025)
    # Requests are clipped to the available annotation; realized ages are reported separately.
    near_seed = D.make_sample(f, 5., False, sample_config(startup_shares=(0., 0., 1., 0.)), rng)
    assert near_seed['trace_prefix_length'] <= 5. and near_seed['startup'] == 2
    assert D.seed_age_stratum(near_seed['seed_age']) == 1


def test_trace_noise_matches_held_out_rollout_residuals():
    # On-track 81k rollouts: lateral residual p50 0.47 / p90 1.11 voxels, correlation 0.46 at 32 voxels.
    f, cfg = straight_fiber(1200), sample_config(startup_shares=(0., 0., 0., 1.), excursion_probability=0.)
    magnitudes, lagged = [], []
    for seed in range(300):
        item = D.make_sample(f, 1200., False, cfg, np.random.default_rng(seed))
        lateral = item['observed_path'][200:, 1:]  # stationary part; GT runs along x
        if len(lateral) > 64:
            magnitudes.append(np.linalg.norm(lateral, axis=1))
            lagged.append(((lateral[:-32]*lateral[32:]).sum(), .5*((lateral[:-32]**2).sum()+(lateral[32:]**2).sum())))
    magnitudes = np.concatenate(magnitudes)
    p50, p90 = np.quantile(magnitudes, [.5, .9])
    assert .37 < p50 < .58 and .9 < p90 < 1.5
    num, den = np.sum(lagged, axis=0)
    assert .35 < num/den < .58


def test_trace_start_takes_ct_seed_heading_and_relabels(monkeypatch):
    f, cfg = curved_fiber(), sample_config(startup_shares=(0., .5, .5, 0.))
    item = next(i for i in (D.make_sample(f, 50., False, cfg, np.random.default_rng(k)) for k in range(50))
                if '_pending_seed_heading' in i and len(i['observed_path']) > 3)
    travel = item['seed_tangent']
    axis = normalize(np.array([1., .35, -.2]))
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.shared.heading.ct_seed_heading', lambda vol, pos, family: -axis)
    world_hist = item['hist_local'] @ item['frame'].T+item['pos']
    D.resolve_trace_seed(item, vol=object())
    assert 'seed_heading_family' not in item and '_pending_seed_heading' not in item
    # Signed along travel, like the tracer's seeds; held as the crop heading.
    expected = axis if axis @ travel > 0 else -axis
    np.testing.assert_allclose(item['seed_tangent'], expected)
    np.testing.assert_allclose(item['frame'][:, 2], expected, atol=1e-12)
    np.testing.assert_allclose(item['hist_local'] @ item['frame'].T+item['pos'], world_hist, atol=1e-9)
    labels = D.continuation_targets(f, 50., False, item['pos'], item['frame'], cfg)
    for key in ('plane_ab', 'plane_mask', 'dense_ab', 'dense_mask', 'fut_local'):
        np.testing.assert_allclose(item[key], labels[key], atol=1e-12)


def test_long_trace_only_takes_ct_seed_tangent(monkeypatch):
    f, cfg = straight_fiber(400), sample_config(startup_shares=(0., 0., 0., 1.))
    item = next(i for i in (D.make_sample(f, 300., False, cfg, np.random.default_rng(k)) for k in range(20))
                if arclength(i['observed_path'])[-1] > 20)
    frame = item['frame'].copy()
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.shared.heading.ct_seed_heading',
                        lambda vol, pos, family: np.array([-.6, .8, 0.]))
    D.resolve_trace_seed(item, vol=object())
    np.testing.assert_allclose(item['seed_tangent'], [.6, -.8, 0.])  # signed along travel (+x)
    np.testing.assert_array_equal(item['frame'], frame)


def test_heading_free_bounds_cover_any_crop_orientation():
    crop = CropSpec(depth=120, width=104, behind=48, spacing=.5)
    pos = np.array([200.3, 180.7, 300.1])
    lo, size = heading_free_bounds(pos, crop, 2.)
    rng = np.random.default_rng(3)
    for _ in range(64):
        frame = frame_from_heading(rng.normal(size=3), rng.normal(size=3))
        block, extent = D.tight_block(pos, frame, crop, 2.)
        assert (block >= lo).all() and (block+extent <= lo+size).all()


def roll_builder():
    from model_fixtures import config
    from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, IdentitySampling
    cfg = config()
    return cfg, IdentityObservationBuilder(cfg, [straight_fiber(400)], IdentitySampling(), augment=True)


def test_roll_augmentation_keeps_world_geometry_and_is_final_before_planning(monkeypatch):
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.shared.heading.oriented_seed_heading',
                        lambda vol, pos, family, direction: np.asarray(direction))
    cfg, builder = roll_builder()
    f = builder.fibers[0]
    sample = D.SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future,
                            startup_shares=(0., 0., 0., 1.), excursion_probability=0.)
    angles = []
    for seed in range(40):
        item = D.make_sample(f, 300., False, sample, np.random.default_rng(seed))
        # Recorded frame (replay or a resolved live state): no CT roll or seed heading pending.
        item.pop('_pending_seed_heading', None)
        item.update(fiber_ref=(0, 300., False), frame_policy=FRAME_POLICY)
        world = lambda i: dict(hist=i['hist_local'] @ i['frame'].T+i['pos'], fut=i['fut_local'] @ i['frame'].T+i['pos'],
                               plane=np.c_[i['plane_ab'], i['planes']] @ i['frame'].T+i['pos'])
        before, heading = world(item), item['frame'][:, 2].copy()
        # A recorded frame takes its roll during preparation, before any footprint is planned.
        item = builder.prepare(item, f, np.random.default_rng(seed))
        assert 'roll_augmentation' not in item
        angles.append(item.get('roll_augmented', 0.))
        planned = item['frame'].copy()
        builder.finalize_frames([item], None)
        np.testing.assert_array_equal(item['frame'], planned)
        after = world(item)
        for key in before:
            np.testing.assert_allclose(after[key], before[key], atol=1e-9)
        np.testing.assert_allclose(item['frame'][:, 2], heading, atol=1e-12)
    flipped = np.abs(np.angle(np.exp(1j*np.asarray(angles)))) > np.pi/2
    jitter = np.rad2deg(np.angle(np.exp(1j*(np.asarray(angles)+np.pi*flipped))))
    assert 8 < flipped.sum() < 32 and np.abs(jitter).max() <= 15+1e-9 and np.abs(jitter).std() > 1


def test_excursions_displace_established_heads_on_departure_and_return():
    f = straight_fiber(1200)
    cfg = sample_config(startup_shares=(0., 0., 0., 1.), excursion_probability=1., trace_noise_sigma=(0., 0.))
    rng = np.random.default_rng(5)
    items = [D.make_sample(f, 800., False, cfg, rng) for _ in range(400)]
    excursions = [i for i in items if i['excursion']]
    assert len(excursions) > 350
    offsets = np.array([i['excursion_head_offset'] for i in excursions])
    phases = np.array([i['excursion_phase'] for i in excursions])
    assert ((phases < 1) & (offsets > 1)).any() and ((phases > 1) & (offsets > 1)).any()  # departing and returning
    assert offsets.max() <= 6.+1e-9 and (offsets > 3).mean() > .2
    for item in excursions:
        np.testing.assert_allclose(item['seed_pos'][1:], 0., atol=1e-12)  # seed stays on the annotation
        assert item['match_distance'] == pytest.approx(item['excursion_head_offset'], abs=1e-6)
        from vesuvius.neural_tracing.fiber_follow.shared.state_labels import (
            FOLLOWING, RECOVERABLE, REASON, UNKNOWN, connection_certified)
        certified = connection_certified(item['plane_ab'][0], item['planes'][0], cfg.max_recovery_distance)
        if certified:
            assert item['supervision'] == (RECOVERABLE if item['match_distance'] > 3 else FOLLOWING)
            assert item['geometry_valid']
        else:
            # A safe proposal may exist, but the annotated connection exceeds the commit limit.
            assert item['supervision'] == UNKNOWN and item['supervision_reason'] == REASON['unsupported_connection']
            assert not item['geometry_valid'] and item['confidence_valid']
    # Startup draws are never given excursions.
    early = D.make_sample(f, 800., False, sample_config(startup_shares=(0., 0., 1., 0.), excursion_probability=1.), rng)
    assert not early['excursion']


def test_ct_heading_failure_rejects_the_seed_without_annotation_fallback(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.shared.heading import SeedHeadingError
    f = curved_fiber()
    item = D.make_sample(f, 50., False, sample_config(startup_shares=(1., 0., 0., 0.)), np.random.default_rng(0))
    def unavailable(vol, pos, family):
        raise SeedHeadingError('CT seed context has no identifiable sheet normal')
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.shared.heading.ct_seed_heading', unavailable)
    with pytest.raises(SeedHeadingError):
        D.resolve_trace_seed(item, vol=object())
    assert 'seed_heading_fallback' not in item
