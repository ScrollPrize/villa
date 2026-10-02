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
    f, cfg = curved_fiber(), sample_config(no_history_prob=.2, short_history_prob=.3)
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
    seed_only = D.make_sample(f, 300., False, sample_config(no_history_prob=1.), np.random.default_rng(0))
    assert len(seed_only['observed_path']) == 1 and not seed_only['hmask'].any() and seed_only['seed_age'] == 0
    assert '_pending_seed_heading' in seed_only
    lengths = [D.make_sample(f, 300., False, sample_config(no_history_prob=0., short_history_prob=1.),
                             np.random.default_rng(seed))['trace_prefix_length'] for seed in range(200)]
    assert 1 <= min(lengths) and max(lengths) <= 32 and len(set(lengths)) > 20


def test_trace_noise_matches_held_out_rollout_residuals():
    # On-track 81k rollouts: lateral residual p50 0.47 / p90 1.11 voxels, correlation 0.46 at 32 voxels.
    f, cfg = straight_fiber(1200), sample_config(no_history_prob=0., short_history_prob=0.)
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
    f, cfg = curved_fiber(), sample_config(no_history_prob=0., short_history_prob=1.)
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
    f, cfg = straight_fiber(400), sample_config(no_history_prob=0., short_history_prob=0.)
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
    from test_identity import config
    from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, IdentitySampling
    cfg = config()
    return cfg, IdentityObservationBuilder(cfg, [straight_fiber(400)], IdentitySampling(), augment=True)


def test_roll_augmentation_keeps_world_geometry_and_pairs_share_it():
    cfg, builder = roll_builder()
    f = builder.fibers[0]
    sample = D.SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future,
                            no_history_prob=0., short_history_prob=0.)
    angles = []
    for seed in range(40):
        item = D.make_sample(f, 300., False, sample, np.random.default_rng(seed))
        item.update(fiber_ref=(0, 300., False), frame_policy=FRAME_POLICY)  # recorded frame: no CT roll needed
        item = builder.prepare(item, f, np.random.default_rng(seed))
        world = lambda i: dict(hist=i['hist_local'] @ i['frame'].T+i['pos'], fut=i['fut_local'] @ i['frame'].T+i['pos'],
                               plane=np.c_[i['plane_ab'], i['planes']] @ i['frame'].T+i['pos'])
        before, heading = world(item), item['frame'][:, 2].copy()
        builder.finalize_frames([item], None)
        angles.append(item.get('roll_augmented', 0.))
        after = world(item)
        for key in before:
            np.testing.assert_allclose(after[key], before[key], atol=1e-9)
        np.testing.assert_allclose(item['frame'][:, 2], heading, atol=1e-12)
    flipped = np.abs(np.angle(np.exp(1j*np.asarray(angles)))) > np.pi/2
    jitter = np.rad2deg(np.angle(np.exp(1j*(np.asarray(angles)+np.pi*flipped))))
    assert 8 < flipped.sum() < 32 and np.abs(jitter).max() <= 15+1e-9 and np.abs(jitter).std() > 1
    # Matched pair rows draw from the same pair RNG, so their crops stay identical.
    rows = [dict(D.make_sample(f, 300., False, sample, np.random.default_rng(1)), fiber_ref=(0, 300., False),
                 pair_observation_seed=77) for _ in range(2)]
    rows = [builder.prepare(row, f, np.random.default_rng(k)) for k, row in enumerate(rows)]
    assert rows[0]['roll_augmentation'] == rows[1]['roll_augmentation']


def test_decision_pairs_are_tracer_states_with_identical_local_inputs(tmp_path):
    from test_mixed_datasets import afv_fixture
    from test_identity import config
    from test_neighbor_following import clean_sample
    from vesuvius.neural_tracing.fiber_follow.regression.identity_decisions import decision_pair
    from vesuvius.neural_tracing.fiber_follow.regression.data import visible_points
    from vesuvius.neural_tracing.fiber_follow.shared.afv import AFVFibers
    from vesuvius.neural_tracing.fiber_follow.regression.datasets import AFVBank
    p = tmp_path/'test.afv'
    afv_fixture(p, length=800, neighbor_x=36)
    bank = AFVBank(AFVFibers(p))
    cfg = config(input_mode='ct', fine=replace(config().fine, depth=48, behind=24))
    sample, rng = clean_sample(cfg), np.random.default_rng(13)
    for choice in (True, False):
        rows = next(r for r in (decision_pair(bank, sample, cfg, rng, choice=choice) for _ in range(40)) if r)
        assert rows[0]['decision_tail'] >= cfg.fine.behind*cfg.fine.spacing+4
        np.testing.assert_array_equal(rows[0]['pos'], rows[1]['pos'])
        np.testing.assert_array_equal(rows[0]['frame'], rows[1]['frame'])
        tail = int(rows[0]['decision_tail'])
        for row in rows:
            path = row['observed_path']
            hist, mask = trace_history(list(path), sample.n_history)
            np.testing.assert_array_equal(row['hmask'], mask)
            np.testing.assert_allclose(row['hist_local'][mask > 0] @ row['frame'].T+row['pos'], hist[mask > 0], atol=1e-9)
            np.testing.assert_allclose(row['frame'][:, 2], trace_heading(path[-tail-1:], 0, row['frame'][:, 2]), atol=1e-9)
            assert row['seed_heading_family'] == 'H'
        # Crop-visible history (the shared tail) is identical in both rows.
        visible = [visible_points(r['hist_local'], cfg.fine) & (r['hmask'] > 0) for r in rows]
        np.testing.assert_array_equal(visible[0], visible[1])
        np.testing.assert_allclose(rows[0]['hist_local'][visible[0]], rows[1]['hist_local'][visible[1]], atol=1e-9)


def test_sampling_fork_preserves_weights_resets_optimizer_and_retires_light_gt_options(tmp_path):
    import copy
    import torch
    from types import SimpleNamespace
    from test_path_geometry import source_checkpoint
    from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
    from vesuvius.neural_tracing.fiber_follow.regression.sampling_fork import RETIRED_OPTIONS, trace_sampling_checkpoint
    from vesuvius.neural_tracing.fiber_follow.regression.train import checkpoint_config, initialize_training_optimizer
    original = source_checkpoint(tmp_path)
    original['training_options'].update(gt_perturb_max_offset=.5, gt_perturb_max_angle_deg=2.,
                                        no_history_prob=.15, short_history_prob=.4)  # the 81k run's values
    before = copy.deepcopy(original['training_options'])
    changed = trace_sampling_checkpoint(original, steps=40000, warmup=2000, no_history_prob=.05, short_history_prob=.1)
    assert original['training_options'] == before  # the source checkpoint is not modified
    assert changed['lr_restart_step'] == changed['step'] == 22000 and changed['optimizer']['state'] == {}
    for section in ('model', 'ema'):
        for name, value in original[section].items():
            torch.testing.assert_close(changed[section][name], value, rtol=0, atol=0)
    new = build_model(checkpoint_config(changed))
    new.load_state_dict(changed['model'], strict=True)
    opt, done, origin = initialize_training_optimizer(new, copy.deepcopy(new), SimpleNamespace(lr=.0001, reset_optimizer=False), changed)
    assert (done, origin) == (22000, 22000) and not opt.state
    options = changed['training_options']
    assert not set(RETIRED_OPTIONS) & set(options)
    assert {k for k in options if options[k] != before[k]} == {'steps', 'warmup', 'no_history_prob', 'short_history_prob'}
    with pytest.raises(ValueError, match='beyond its source step'):
        trace_sampling_checkpoint(original, steps=22000, warmup=0, no_history_prob=0., short_history_prob=0.)
