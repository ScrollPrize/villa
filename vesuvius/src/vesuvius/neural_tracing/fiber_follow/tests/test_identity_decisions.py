"""Ownership decisions train the deployed classifier and retain observed references."""
import copy
from types import SimpleNamespace

import numpy as np
import torch

from test_identity import config, batch
from test_neighbor_bank import make_bank, add_shard, publish
from test_neighbor_following import clean_sample
from vesuvius.neural_tracing.fiber_follow.regression.identity_decisions import decision_pair
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, reference_layout
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig, DirectFollower
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import optimizer_update, prepare_training
from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset, OnPolicyStates, fiber_manifest
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS, observed_seed
from vesuvius.neural_tracing.fiber_follow.shared.trace import ModelTracer, TraceParams




def candidate_batch(cfg):
    data = batch(cfg)
    curves = torch.zeros(2, 2, cfg.n_future, 3)
    curves[..., 2] = torch.arange(1, cfg.n_future+1)
    curves[:, 1, :, 0] = 2.
    labels = torch.zeros(2, 2, cfg.n_future)
    labels[0, 0], labels[1, 1] = 1., 1.
    data.update(candidate_points=curves, candidate_labels=labels, candidate_mask=torch.ones_like(labels),
                source=torch.full((2,), 5), decision_kind=torch.ones(2), decision_tail=torch.full((2,), 8.))
    return data




def test_candidate_update_independent_of_microbatch_boundaries():
    torch.manual_seed(4)
    a = DirectFollower(config())
    b = copy.deepcopy(a)
    data = candidate_batch(a.cfg)
    data['candidate_mask'][1] = 0  # unknown states contribute zero, not a new denominator
    def take(value, sl):
        return {k: take(v, sl) for k, v in value.items()} if isinstance(value, dict) else value[sl]
    results = []
    for model, batches in ((a, [data]), (b, [take(data, slice(0, 1)), take(data, slice(1, 2))])):
        ema = copy.deepcopy(model)
        prepare_training(model, backend='eager')
        results.append(optimizer_update(model, ema, torch.optim.SGD(model.parameters(), lr=.001),
                                        batches, 1, .001, compute_metrics=True))
    np.testing.assert_allclose(results[0]['loss'], results[1]['loss'], rtol=2e-5)
    assert results[0]['identity']['candidate_states'] == results[1]['identity']['candidate_states'] == 1
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=2e-5, atol=1e-7)


def test_observed_seed_at_start_reaches_reference_tokens():
    cfg = config()
    state = dict(pos=np.array([20., 30., 40.]), frame=np.eye(3),
                 hist_local=np.zeros((cfg.n_history, 3)), hmask=np.zeros(cfg.n_history))
    state.update(observed_seed(**state))
    reference_layout(state, cfg)
    assert state['reference_mask'].sum() == 1
    np.testing.assert_array_equal(state['reference_points'][cfg.n_history], 0.)
    model = DirectFollower(cfg)
    data = batch(cfg, 1)
    data['hmask'].zero_()
    out = model(data['x'], data['hist'], data['hmask'])
    _, mask = model.references(data['x'], data['hist'], data['hmask'])
    assert mask.sum() == 1 and mask[0, -1]
    out['hazard_logits'].sum().backward()
    assert model.reference_token[0].weight.grad.abs().sum() > 0


def test_replay_preserves_seed_but_distant_seed_is_not_observable(tmp_path):
    bank, parent = make_bank(tmp_path/'bank')
    cfg = DirectConfig()
    arrays = dict(fiber_idx=[0], t=[100.], reverse=[False], pos=[[0., 0., 100.]],
                  frame=[np.eye(3)], hist=np.zeros((1, cfg.n_history, 3)), hmask=np.zeros((1, cfg.n_history)),
                  offtrack=[False], hard=[False], exploratory=[False])
    ds = object.__new__(FollowDataset)
    ds.correct_replay_only, ds.replay_continuation_fraction = False, None
    ds.vol_spec = SimpleNamespace(grid_scale=8., ct_grid_scale=4.)
    ds.fibers, ds.cfg, ds.exclude, ds.additional_crops = [parent], clean_sample(cfg), bank.band, ()
    ds.batch_builder = IdentityObservationBuilder(cfg, [parent], negative_bank=bank)
    for valid in (False, True):
        seed = dict(seed_pos=[[0., 0., 20.]], seed_tangent=[[0., 0., 1.]], seed_age=[80.], seed_valid=[True]) if valid else {}
        prefix = np.array([[0.,0.,20.],[0.,0.,100.]]) if valid else np.array([[0.,0.,100.]])
        states = OnPolicyStates(manifest=fiber_manifest([parent]), **arrays, **seed,
                                seq_start=[0], seq_end=[len(prefix)], track_pos=prefix)
        path = tmp_path/f'{valid}.npz'
        states.save(path)
        loaded = OnPolicyStates.load(path)
        row = ds.replay_item((1, 0, loaded, 0), np.random.default_rng(0))
        assert bool(row['seed_valid']) == valid
        assert row['reference_mask'][cfg.n_history:].sum() == 0  # distant seed is preserved in replay, masked in the crop
        for field in SEED_FIELDS:
            np.testing.assert_array_equal(getattr(loaded, field), getattr(states, field))


def test_trace_keeps_seed_from_first_decision_through_recovery(monkeypatch):
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.shared.heading.ct_tensor',
                        lambda vol,pos: np.outer(np.array([1., 0., 0.]), np.array([1., 0., 0.])))
    class Model(torch.nn.Module):
        cfg = SimpleNamespace(max_recovery_distance=4.)
        def forward(self, x, hist, hmask):
            points = torch.zeros(len(hist), 2, 3)
            points[..., 2] = torch.tensor([1., 2.])
            return dict(points=points, confidence=torch.ones(len(hist), 2))
    class Tracer(ModelTracer):
        path_context = True
        def build_inputs(self, pos, fr, hist, hm, **context):
            return {}
    tracer = object.__new__(Tracer)
    tracer.model, tracer.device, tracer.n_history = Model(), 'cpu', 8
    tracer.vol = SimpleNamespace(shape=(1000, 1000, 1000))
    tracer.p = TraceParams(n_commit=2, max_len=6.)
    rows = []
    tracer._trace(np.array([[10., 20., 30.]]), np.array([[0., 0., 1.]]), None, None,
                  lambda i, state: rows.append(state))
    assert len(rows) == 3 and [r['seed_age'] for r in rows] == [0., 2., 4.]
    for row in rows:
        np.testing.assert_array_equal(row['seed_pos'], [10., 20., 30.])
        assert row['seed_valid']
    recovered = dict(rows[1])
    rows.clear()
    tracer._trace(recovered['pos'][None], np.array([[0., 0., 1.]]), None, None,
                  lambda i, state: rows.append(state), initial_states=[recovered])
    np.testing.assert_array_equal(rows[-1]['seed_pos'], recovered['seed_pos'])
    assert rows[-1]['seed_age'] == 6.
