"""Long memory sequences: burn-in, per-write probe, observed tracks and switch paths."""
import copy

import numpy as np
import pytest
import torch

from test_identity import config, line_fiber
from test_memory_data import memory_config
from test_memory_training import scene
from test_neighbor_bank import make_bank
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, IdentitySampling, LOCATION_SOURCES
from vesuvius.neural_tracing.fiber_follow.regression.memory_data import memory_images, memory_layout, memory_targets
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectFollower
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_continuations import wrong_continuation
from vesuvius.neural_tracing.fiber_follow.regression.supervision import memory_probe_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import optimizer_update
from vesuvius.neural_tracing.fiber_follow.shared.collect import DecisionCollector, append_traces, track_arrays
from vesuvius.neural_tracing.fiber_follow.shared.data import OnPolicyStates, SampleConfig, fiber_manifest






def test_probe_terms_average_labeled_writes_per_state():
    probe = torch.zeros(2, 3, 4)
    probe[0, :, 0] = torch.tensor([2., -2., 2.])
    probe[1, :, 1:] = 1.
    batch = dict(memory_target_identity=torch.tensor([[1., 0., 0.], [1., 1., 1.]]),
                 memory_target_identity_mask=torch.tensor([[True, True, True], [False, False, True]]),
                 memory_target_offset=torch.zeros(2, 3, 3),
                 memory_target_offset_mask=torch.tensor([[False, False, False], [False, True, True]]))
    terms = memory_probe_terms(dict(memory_probe=probe), batch)
    assert terms['memory_identity_count'] == 4 and terms['memory_identity_correct'] == 2
    assert terms['memory_departed_count'] == 2 and terms['memory_departed_correct'] == 1
    bce = torch.nn.functional.softplus(-torch.tensor(2.))
    torch.testing.assert_close(terms['memory_identity_per_state'][0], (bce+bce+torch.nn.functional.softplus(torch.tensor(2.)))/3)
    torch.testing.assert_close(terms['memory_offset_per_state'], torch.tensor([0., 1.5]))
    torch.testing.assert_close(terms['memory_offset_error_sum'], torch.tensor(2*3**.5))


def track_item(cfg, count=10):
    z = 300.+4*np.arange(count)
    track = dict(pos=np.c_[np.full(count, 100.), np.full(count, 100.), z],
                 frame=np.repeat(np.eye(3)[None], count, 0),
                 offtrack=np.r_[np.zeros(count-3), np.nan, 1., 1.].astype(np.float32),
                 offset=np.r_[np.tile([[.5, 0., 0.]], (count-3, 1)), np.full((3, 3), np.nan)])
    hist = np.c_[np.zeros(cfg.n_history), np.zeros(cfg.n_history), -np.arange(1, cfg.n_history+1)]
    return dict(pos=np.array([100., 100., 300.+4*count]), frame=np.eye(3),
                hist_local=hist, hmask=np.ones(cfg.n_history), seed_pos=np.array([100., 100., 290.]),
                seed_tangent=np.array([0., 0., 1.]), seed_age=50., seed_valid=True, memory_track=track,
                offtrack=1., gt_history=np.zeros((cfg.n_history+1, 3), np.float32),
                gt_history_mask=np.zeros(cfg.n_history+1, np.float32))


def test_tracks_supply_observations_and_aligned_targets_with_burn_in():
    cfg = memory_config(memory_steps=6, memory_grad_steps=4)
    item = track_item(cfg)
    observations, seed = memory_layout(item, cfg)
    np.testing.assert_array_equal([o['pos'][2] for o in observations], [316, 320, 324, 328, 332, 336, 340])
    assert [o.get('track') for o in observations] == [4, 5, 6, 7, 8, 9, None] and seed['pos'][2] == 290
    reads = []
    def sample(items, vol, crop, pool=None):
        reads.extend(items)
        return torch.stack([torch.full((2, crop.depth, crop.width, crop.width), float(i['pos'][2])) for i in items])
    x = memory_images([item], None, cfg, sample)
    torch.testing.assert_close(x['history_crops'][0, :, 0, 0, 0, 0], torch.tensor([316., 320, 324, 328, 332, 336]))
    targets = memory_targets([item], cfg)
    # Two burn-in writes, an unlabeled bridge write, then departed writes and head.
    np.testing.assert_array_equal(targets['memory_target_identity_mask'][0], [False, False, True, False, True, True, True])
    np.testing.assert_array_equal(targets['memory_target_identity'][0], [0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(targets['memory_target_offset_mask'][0], [False, False, True, False, False, False, False])
    expected = np.array([.5, 0., 0.]) @ observations[2]['frame']
    torch.testing.assert_close(targets['memory_target_offset'][0, 2], torch.tensor(expected, dtype=torch.float32))
    item['identity_observable'] = False
    assert not memory_targets([item], cfg)['memory_target_identity_mask'].any()


def test_reconstructed_histories_label_only_the_head_in_its_observation_frame():
    cfg = memory_config()
    item = track_item(cfg)
    del item['memory_track']
    item.update(offtrack=0., gt_history_mask=np.ones(cfg.n_history+1, np.float32))
    item['gt_history'][0] = [1., 2., 0.]
    item['frame'] = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    targets = memory_targets([item], cfg)
    np.testing.assert_array_equal(targets['memory_target_identity_mask'][0], [False]*cfg.memory_steps+[True])
    head = memory_layout(item, cfg)[0][-1]
    world = item['frame'] @ item['gt_history'][0]
    torch.testing.assert_close(targets['memory_target_offset'][0, -1], torch.tensor(world @ head['frame'], dtype=torch.float32))
    item['gt_history'][0] = [7., 0., 0.]  # beyond recovery distance: identity only
    targets = memory_targets([item], cfg)
    assert targets['memory_target_identity_mask'][0, -1] and not targets['memory_target_offset_mask'].any()


def test_collector_tracks_every_decision_and_replay_rows_reference_prior_heads(tmp_path):
    fiber = line_fiber()
    sample = SampleConfig(crop=config().fine, n_history=32, recent_history_points=32, n_future=4)
    collectors, heads = [], []
    for start in (300., 350.):
        collector = DecisionCollector(fiber, 0, start-200., 1, sample, stride=8.)
        pos = np.array([100., 100., start])
        segment = pos[None]
        for step in range(6):
            hist = np.c_[np.full(32, 100.), np.full(32, 100.), pos[2]-np.arange(1, 33)]
            state = dict(pos=pos.copy(), frame=np.eye(3), hist=hist, hmask=np.ones(32), exploratory=False,
                         would_stop=False, last_segment=segment, travelled=4.*step)
            assert collector(state)
            heads.append(pos.copy())
            segment = np.array([pos, pos+[0, 0, 4]])
            pos = pos+[.5, 0., 4.]
        collectors.append(collector)
    rows, track = [], []
    append_traces(collectors, rows, track)
    assert len(track) == 12 and len(rows) < 12  # thinning keeps the whole track
    states = OnPolicyStates(manifest=fiber_manifest([fiber]),
                            **{k: np.asarray([r[k] for r in rows]) for k in
                               OnPolicyStates.FIELDS+tuple(OnPolicyStates.OPTIONAL)+tuple(OnPolicyStates.ROW_TRACK)},
                            **track_arrays(track))
    states.save(tmp_path/'decisions.npz')
    loaded = OnPolicyStates.load(tmp_path/'decisions.npz')
    for j in range(len(loaded)):
        previous = loaded.track(j)
        trace = 0 if loaded.seq_start[j] == 0 else 6
        np.testing.assert_allclose(previous['pos'], np.reshape(heads[trace:trace+len(previous['pos'])], (-1, 3)), atol=1e-4)
        assert len(previous['pos']) == loaded.source_row[j]
    drifted = loaded.track(int(np.argmax(loaded.seq_end)))
    assert (drifted['offtrack'] == 0).all()
    np.testing.assert_allclose(drifted['offset'][:, 0], -.5*np.arange(len(drifted['offset'])), atol=1e-4)
    # Archives without tracks load unchanged, without mirroring new files.
    plain = OnPolicyStates(manifest=fiber_manifest([fiber]),
                           **{k: np.asarray([r[k] for r in rows]) for k in OnPolicyStates.FIELDS+tuple(OnPolicyStates.OPTIONAL)})
    np.savez(tmp_path/'plain.npz', __metadata__=np.load(tmp_path/'decisions.npz')['__metadata__'],
             **{k: getattr(plain, k) for k in OnPolicyStates.FIELDS+tuple(OnPolicyStates.OPTIONAL)})
    old = OnPolicyStates.load(tmp_path/'plain.npz')
    assert old.track(0) is None and len(old.track_pos) == 0
    assert not (tmp_path/'plain_mmap_v5'/'seq_end.npy').exists()
    with pytest.raises(ValueError, match='outside'):
        OnPolicyStates(manifest=[], **{k: np.asarray([r[k] for r in rows]) for k in
                       OnPolicyStates.FIELDS+tuple(OnPolicyStates.OPTIONAL)+tuple(OnPolicyStates.ROW_TRACK)})


def test_switch_sequences_observe_original_prefix_bridge_and_neighbor_tail(tmp_path):
    bank, fiber = make_bank(tmp_path, with_path=True)
    cfg = memory_config(memory_steps=24, n_history=32)
    sample = SampleConfig(crop=cfg.fine, n_history=32, recent_history_points=32, n_future=4)
    builder = IdentityObservationBuilder(cfg, [fiber], negative_bank=bank,
        sampling=IdentitySampling(memory_switch_probability=1., bank_following_probability=0.,
                                  bank_coverage_probability=0., memory_switch_tail=(4., 8.)))
    for seed in range(4):
        item = builder.replace_fresh(sample, np.random.default_rng(seed))
        assert item is not None and item['offtrack'] == 1 and item['location_source'] == LOCATION_SOURCES.index('memory_switch')
        track = item['memory_track']
        x = np.abs(track['pos'][:, 0])
        assert np.all(np.diff(track['pos'][:, 2]*(1 if item['fiber_ref'][2] is False else -1)) > 0)
        np.testing.assert_allclose(x[track['offtrack'] == 0], 0., atol=1e-6)
        np.testing.assert_allclose(x[track['offtrack'] == 1], 6., atol=1e-6)
        assert np.isnan(track['offtrack']).any() and (track['offtrack'] == 0).sum() > 32//cfg.memory_stride
        assert track['offset'].shape == track['pos'].shape
        assert (track['offset'][track['offtrack'] == 0] == 0).all() and np.isnan(track['offset'][track['offtrack'] != 0]).all()
        assert item['seed_age'] > 70  # the seed moved back beyond the default n_history+4 prefix
        builder.prepare(item, fiber, np.random.default_rng(seed))
        assert item['identity_observable']
    # Tracks consume no randomness; defaults keep the established sampler.
    plain = wrong_continuation(bank, sample, np.random.default_rng(3), tail_length_range=(4., 8.))
    tracked = wrong_continuation(bank, sample, np.random.default_rng(3), tail_length_range=(4., 8.), track_stride=4)
    assert 'memory_track' not in plain and len(tracked['memory_track']['pos'])
    for key in ('pos', 'frame', 'hist_local', 'seed_pos'):
        np.testing.assert_array_equal(plain[key], tracked[key])


def test_optimizer_adds_weighted_probe_loss_and_logs_it():
    torch.manual_seed(8)
    cfg,data = scene(b=2,steps=4,grad=4)
    count = cfg.memory_steps+1
    data.update(memory_target_identity=torch.ones(2, count), memory_target_identity_mask=torch.ones(2, count, dtype=torch.bool),
                memory_target_offset=torch.zeros(2, count, 3), memory_target_offset_mask=torch.ones(2, count, dtype=torch.bool))
    losses = []
    for weight in (0., 1.):
        model = DirectFollower(cfg)
        torch.manual_seed(9)
        model.load_state_dict(DirectFollower(cfg).state_dict())
        metrics = optimizer_update(model, copy.deepcopy(model), torch.optim.SGD(model.parameters(), lr=0.),
                                   [data], 1, 0., memory_probe_weight=weight)
        losses.append(metrics['loss'])
    memory = metrics['memory']
    assert losses[1] == pytest.approx(losses[0]+memory['probe_identity_loss']+memory['probe_offset_loss'], rel=1e-5)
    assert memory['identity_count'] == 2*count and memory['labeled_writes_per_state'] == count
