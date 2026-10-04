"""Decision memory: selection/pruning, light stem, entry parity and tracer cache parity."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model_fixtures import coordinate_config, coordinate_batch
from vesuvius.neural_tracing.fiber_follow.models.model import build_model, CoordinateRegressionConfig
from vesuvius.neural_tracing.fiber_follow.models.decision_memory import (
    SLOTS, ENTRY_FEATURES, TOKENS_PER_ENTRY, select_decisions, prune_decisions)
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, crop_local_grid, frame_from_heading


def memory_config(**kwargs):
    return coordinate_config(**dict(dict(stem='stride2', stem_blocks=1, memory='decisions', layers=2), **kwargs))


def test_defaults_keep_existing_model():
    cfg = CoordinateRegressionConfig()
    assert (cfg.stem, cfg.memory) == ('residual', 'slabs')
    with pytest.raises(ValueError):
        coordinate_config(memory='tube')


def test_selection_reads_seed_older_lattice_and_spaced_recent_decisions():
    t = np.arange(0., 2000., 16.)
    chosen = t[select_decisions(t, 2000.)]
    np.testing.assert_allclose(chosen, [0, 512, 1024, 1536, 1904, 1936, 1968])
    assert select_decisions([], 10.) == [] and select_decisions([0.], 10.) == [0]
    assert list(t[select_decisions(t[:4], 64.)]) == [0., 32.]
    with pytest.raises(ValueError):
        select_decisions([0., 20.], 10.)


def test_selection_on_pruned_records_matches_full_history():
    rng = np.random.default_rng(0)
    for _ in range(60):
        full, kept, length = [0.], [0.], 0.
        for _ in range(int(rng.integers(5, 300))):
            length += float(rng.uniform(1, 16))
            assert [full[i] for i in select_decisions(full, length)] == [kept[i] for i in select_decisions(kept, length)]
            full.append(length)
            kept.append(length)
            kept = [kept[i] for i in prune_decisions(kept, length)]
            assert len(kept) <= 40


def test_light_stem_outputs_the_token_grid():
    c = memory_config()
    model = build_model(c)
    image = torch.rand(2, 1, c.fine.depth, c.fine.width, c.fine.width)
    assert model.encoder.stem(image).shape == (2, c.hidden, *c.token_shape)
    assert not any(isinstance(m, torch.nn.Conv3d) and m.stride == (1, 1, 1) and m.kernel_size == (3, 3, 3)
                   for m in model.encoder.stem.input.modules())


def decision_inputs(c, b, crops=0):
    z = lambda *shape, **kw: torch.zeros(b, *shape, **kw)
    return dict(history_valid=z(SLOTS, dtype=torch.bool), history_encode=z(SLOTS, dtype=torch.bool),
                history_keys=torch.full((b, SLOTS), -1), history_chain=torch.full((b,), -1),
                history_pose=z(SLOTS, 14), history_ages=z(SLOTS), history_path_points=z(SLOTS, 3, 3),
                history_path_tangents=z(SLOTS, 3, 3), history_path_valid=z(SLOTS, 3, dtype=torch.bool),
                history_crops=torch.zeros(crops, 1, c.fine.depth, c.fine.width, c.fine.width),
                history_references=torch.zeros(crops, c.n_history+1, 3),
                history_reference_mask=torch.zeros(crops, c.n_history+1))


def test_decision_entry_equals_the_entry_encoded_from_its_crop():
    torch.manual_seed(3)
    c = memory_config()
    model = build_model(c).eval()
    batch = coordinate_batch(c, 1)
    x = {k: v for k, v in batch['x'].items() if not k.startswith('history_')}
    x.update(decision_inputs(c, 1))
    hist, hmask = batch['hist'].clone(), batch['hmask'].clone()
    hist[..., 0] = .3*torch.sin(torch.arange(c.n_history)/5.)
    with torch.no_grad():
        entry = model(x, hist, hmask)['memory_entry']
        references, mask = model.references(x, hist, hmask)
        later = {k: v for k, v in x.items() if not k.startswith('history_')}
        later.update(decision_inputs(c, 1, crops=1))
        later['history_valid'][0, 0] = later['history_encode'][0, 0] = True
        later['history_crops'][:] = x['fine']
        later['history_references'][:], later['history_reference_mask'][:] = references, mask.float()
        later['history_path_points'][0, 0] = torch.stack((hist[0, 1], hist[0, 0], torch.zeros(3)))
        later['history_path_valid'][0, 0] = True
        encoded = model.memory_features(later)
    assert entry.shape == (1, ENTRY_FEATURES, c.hidden) and entry.dtype == torch.bfloat16
    torch.testing.assert_close(encoded[0, 0], entry[0], rtol=0, atol=0)
    assert not encoded[0, 1:].any()
    tokens, padding = model.encode_history(later)
    assert tokens.shape == (1, SLOTS*TOKENS_PER_ENTRY, c.hidden)
    assert not padding[0, :TOKENS_PER_ENTRY].any() and padding[0, TOKENS_PER_ENTRY:].all()


def test_invalid_memory_slots_are_inert():
    torch.manual_seed(5)
    c = memory_config()
    model = build_model(c).eval()
    batch = coordinate_batch(c, 2)
    x = {k: v for k, v in batch['x'].items() if not k.startswith('history_')}
    x.update(decision_inputs(c, 2))
    x['history_valid'][:, :2] = True
    x['history_path_valid'][:, :2] = True
    x['history_features'] = torch.randn(2, SLOTS, ENTRY_FEATURES, c.hidden).to(torch.bfloat16)
    first = model(x, batch['hist'], batch['hmask'])
    x['history_features'][:, 2:] = float('nan')
    x['history_pose'][:, 2:] = float('nan')
    x['history_path_points'][:, 2:] = float('nan')
    other = model(x, batch['hist'], batch['hmask'])
    for key in ('points', 'hazard_logits', 'refinement_points'):
        torch.testing.assert_close(first[key], other[key], rtol=0, atol=0)


def world_ct(items, vol, crop, pool=None, *, presence=False, out=None, **kwargs):
    """Deterministic CT of world position, sampled in each item's crop frame."""
    grid = crop_local_grid(crop)
    values = []
    for item in items:
        world = grid @ np.asarray(item['frame']).T+np.asarray(item['pos'])
        values.append(np.sin(.37*world[..., 0])+np.cos(.23*world[..., 1]-.11*world[..., 2]))
    result = torch.from_numpy(np.stack(values)[:, None].astype(np.float32))
    if out is not None:
        out[:] = result.numpy()
        return torch.from_numpy(out)
    return result


class StraightModel(torch.nn.Module):
    """The real model's memory, with a fixed straight path so both runs trace identically."""
    def __init__(self, model):
        super().__init__()
        self.inner, self.cfg = model, model.cfg
        self.raw = []

    def encode_memory_crops(self, x, rows):
        return self.inner.encode_memory_crops(x, rows)

    def forward(self, x, hist, hmask, **kwargs):
        out = self.inner(x, hist, hmask, **kwargs)
        self.raw.append((out['points'].clone(), out['confidence'].clone(), x['history_valid'].clone()))
        planes = torch.arange(1, self.cfg.n_future+1, dtype=torch.float32)*self.cfg.future_step
        out['points'] = torch.stack((.05*torch.ones_like(planes), torch.zeros_like(planes), planes), -1).expand(len(hist), -1, -1).clone()
        out['confidence'] = torch.ones(len(hist), self.cfg.n_future)
        return out


def run_tracer(model, monkeypatch, recorded=True):
    from vesuvius.neural_tracing.fiber_follow.data.observations import ObservationBuilder, FiberTracer
    from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams, ModelTracer
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.data.crop_sampling.scalar_crops', world_ct)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.data.observations.image_crop',
                        lambda items, vol, crop, pool=None: world_ct(items, vol, crop))
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.ct_tensor',
                        lambda vol, pos: np.outer([0., 1., 0.], [0., 1., 0.]))
    tracer = FiberTracer.__new__(FiberTracer)
    tracer.model, tracer.device, tracer.n_history = model, 'cpu', model.cfg.n_history
    tracer.vol, tracer.pool, tracer.observations = SimpleNamespace(shape=(1000, 1000, 1000)), None, ObservationBuilder(model.cfg)
    tracer.p = TraceParams(n_commit=model.cfg.n_future, max_len=90., confidence=.5, loop_radius=.01)
    if not recorded:
        # Never reuse a decision's own entry: every read entry is encoded from its crop.
        def record(entries, idx, pos, frames, paths, memory):
            for j, i in enumerate(idx):
                length = float(arclength(np.asarray(paths[i]))[-1])
                memory[i]['decisions'].append(dict(travelled=length, pos=pos[j].copy(), frame=frames[j].copy(), key=None))
        monkeypatch.setattr(tracer, 'memory_record', record)
    with torch.no_grad():
        paths, _ = tracer._trace(np.array([[100., 100., 100.]]), np.array([[0., 0., 1.]]), None, None, None)
    return paths


def test_tracer_reuses_decision_entries_exactly_as_encoding_their_crops(monkeypatch):
    torch.manual_seed(11)
    model = build_model(memory_config(n_future=4)).eval()
    cached, fresh = StraightModel(model), StraightModel(model)
    first = run_tracer(cached, monkeypatch)
    second = run_tracer(fresh, monkeypatch, recorded=False)
    np.testing.assert_allclose(first[0], second[0])
    assert len(cached.raw) == len(fresh.raw) > 10
    assert max(int(v.sum()) for _, _, v in cached.raw) >= 3
    for (p0, c0, v0), (p1, c1, v1) in zip(cached.raw, fresh.raw):
        torch.testing.assert_close(v0, v1, rtol=0, atol=0)
        torch.testing.assert_close(p0, p1, rtol=0, atol=1e-5)
        torch.testing.assert_close(c0, c1, rtol=0, atol=1e-5)


def test_data_side_entry_matches_the_decision_observation():
    from vesuvius.neural_tracing.fiber_follow.data.decision_memory import memory_layout, entry_path_samples, entry_item
    from vesuvius.neural_tracing.fiber_follow.data.observations import reference_layout
    from vesuvius.neural_tracing.fiber_follow.tracing.trace import trace_history
    c = memory_config(n_history=32)
    t = np.arange(0., 121.)
    path = np.c_[3*np.sin(t/17.), np.zeros_like(t), t]
    arc = arclength(path)
    at = float(arc[40])
    prefix = path[:41]
    frame = frame_from_heading([.1, 0., 1.])
    world, mask = trace_history(list(prefix), c.n_history)
    decision = dict(pos=prefix[-1], frame=frame, hist_local=(world-prefix[-1]) @ frame, hmask=mask,
                    seed_pos=path[0], seed_tangent=np.array([0., 0., 1.]), seed_age=at, seed_valid=True)
    record = dict(travelled=at, pos=prefix[-1].copy(), frame=frame, key=None)
    item = dict(pos=path[-1], frame=np.eye(3), observed_path=path, seed_pos=path[0], seed_tangent=np.array([0., 0., 1.]),
                seed_age=float(arc[-1]), seed_valid=True, memory_decisions=[record])
    entries, p, a, _ = memory_layout(item)
    assert len(entries) == 1 and entries[0]['encode'] and entries[0]['seed'] is False
    points, _, valid = entry_path_samples(p, a, entries[0])
    np.testing.assert_allclose(points, np.stack((decision['hist_local'][1], decision['hist_local'][0], np.zeros(3))), atol=1e-9)
    assert valid.all()
    expected = reference_layout(dict(decision), c)
    observed = entry_item(item, entries[0], c)
    np.testing.assert_allclose(observed['reference_points'], expected['reference_points'], atol=1e-9)
    np.testing.assert_array_equal(observed['reference_mask'], expected['reference_mask'])


def test_chain_records_and_store_keep_exactly_what_later_states_read():
    from vesuvius.neural_tracing.fiber_follow.train.live_continuation import chain_memory, DecisionStore
    from vesuvius.neural_tracing.fiber_follow.data.decision_memory import memory_layout
    store = DecisionStore(max_age=10)
    path = np.c_[np.zeros(1), np.zeros(1), np.zeros(1)]
    item = dict(pos=path[-1], frame=np.eye(3), observed_path=path, seed_pos=path[0],
                seed_tangent=np.array([0., 0., 1.]), seed_age=0., seed_valid=True)
    chain = None
    for depth in range(40):
        chain_id, decisions = chain_memory(dict(item, **({} if chain is None else dict(live_chain_id=chain))), depth)
        assert chain is None or chain_id == chain
        chain = chain_id
        assert decisions[-1]['key'] == depth and len(decisions) <= 40
        store.put(chain, depth, torch.full((2,), float(depth)), {d['key'] for d in decisions}, step=depth)
        # The next state reads only recorded decisions that the store still holds.
        step = np.c_[np.zeros(9), np.zeros(9), item['pos'][2]+np.arange(1., 10.)]
        item = dict(item, pos=step[-1], observed_path=np.concatenate((item['observed_path'], step)),
                    memory_decisions=decisions)
        entries = memory_layout(item)[0]
        keys = torch.tensor([[e['record']['key'] for e in entries]+[-1]*(SLOTS-len(entries))])
        features, found = store.gather(torch.tensor([chain]), keys, (2,), 'cpu')
        assert found[0, :len(entries)].all() and not found[0, len(entries):].any()
        assert not any(e['encode'] for e in entries)
        torch.testing.assert_close(features[0, :len(entries), 0], keys[0, :len(entries)].float().to(torch.bfloat16))
    store.evict(40+11)
    assert len(store) == 0


def test_identity_loss_trains_projection_and_current_crop_only():
    from vesuvius.neural_tracing.fiber_follow.models.decision_memory import IDENTITY_NEGATIVES
    torch.manual_seed(21)
    c = memory_config(identity_dim=8)
    model = build_model(c).train()
    batch = coordinate_batch(c, 2)
    x = {k: v for k, v in batch['x'].items() if not k.startswith('history_')}
    x.update(decision_inputs(c, 2))
    ctx = model.context(x, batch['hist'], batch['hmask'])
    points = torch.zeros(2, 1+IDENTITY_NEGATIVES, 3)
    points[:, 0, 2], points[:, 1, :] = 1., torch.tensor([1.5, 0., 1.])
    point_mask = torch.zeros(2, 1+IDENTITY_NEGATIVES, dtype=torch.bool)
    point_mask[:, :2] = True
    anchor_mask = torch.zeros(2, SLOTS, dtype=torch.bool)
    anchor_mask[0, :2] = True  # row 1 has no anchor: inert
    anchors = torch.randn(2, SLOTS, c.hidden, requires_grad=True)
    terms = model.identity_terms(ctx, dict(x, identity_points=points, identity_point_mask=point_mask,
                                          identity_anchor_mask=anchor_mask, identity_anchor_features=anchors))
    assert terms['identity_pairs'].sum() == 2 and terms['identity_loss_per_state'][1] == 0
    assert torch.isfinite(terms['identity_loss_per_state']).all()
    terms['identity_loss_per_state'].sum().backward()
    assert model.identity_projection.weight.grad.abs().sum() > 0
    assert model.encoder.patch_projection.weight.grad.abs().sum() > 0
    # Masked negatives and anchors contribute nothing (finite masking, no NaN gradients).
    assert torch.isfinite(model.identity_projection.weight.grad).all()
    assert not anchors.grad[1].any() and not anchors.grad[0, 2:].any()


def test_identity_targets_use_each_neighbor_path_crossing_of_the_positive_plane_and_on_fiber_old_anchors():
    from types import SimpleNamespace as NS
    from vesuvius.neural_tracing.fiber_follow.data.observations import IdentityObservationBuilder
    from vesuvius.neural_tracing.fiber_follow.models.decision_memory import IDENTITY_NEGATIVES
    builder = IdentityObservationBuilder.__new__(IdentityObservationBuilder)
    builder.fibers = [NS(points=np.c_[np.zeros(300), np.zeros(300), np.arange(300.)])]
    builder.sampling = NS(rule=NS(own_radius=1.5), on_fiber_tolerance=1.5)
    planes = np.arange(1., 17.)
    item = dict(planes=planes, plane_mask=np.ones(16), plane_ab=np.zeros((16, 2)), fiber_ref=(0, 100., False),
                trace_facts=dict(departure_distance=10.), travelled=50.,
                _memory_entries=[dict(age=200., pos=np.array([0., 0., 50.])), dict(age=40., pos=np.array([0., 0., 150.])),
                                 dict(age=150., pos=np.array([4., 0., 60.]))])
    found = dict(local=np.array([[3., 0., 1.5], [3., 0., 2.5],      # path 7 crosses z=2 at x=3
                                 [6., 0., 1.], [6., 0., 3.],        # path 8 at x=6
                                 [.5, 0., 1.5], [.5, 0., 2.5],      # path 9: inside the own radius
                                 [5., 0., 8.], [5., 0., 9.],        # path 10: never reaches the plane
                                 [8., 0., 1.], [8., 0., 5.]]),      # path 11: a filtered gap, not a crossing
                 path_ids=np.array([7, 7, 8, 8, 9, 9, 10, 10, 11, 11]))
    out = dict(identity_points=np.zeros((1, 1+IDENTITY_NEGATIVES, 3), np.float32),
               identity_point_mask=np.zeros((1, 1+IDENTITY_NEGATIVES), bool), identity_anchor_mask=np.zeros((1, SLOTS), bool),
               identity_departure_age=np.full(1, np.nan, np.float32))
    builder.identity_targets(item, found, out, 0)
    np.testing.assert_allclose(out['identity_points'][0, 0], [0., 0., 2.])
    np.testing.assert_allclose(out['identity_points'][0, 1:3], [[3., 0., 2.], [6., 0., 2.]])
    assert out['identity_point_mask'][0].tolist() == [True, True, True]+[False]*(IDENTITY_NEGATIVES-2)
    assert out['identity_anchor_mask'][0].tolist() == [True, False, False]+[False]*(SLOTS-3)
    assert out['identity_departure_age'][0] == 40.


def test_identity_anchor_side_excludes_same_fiber_and_padded_rows_and_control_uses_other_anchors():
    from vesuvius.neural_tracing.fiber_follow.models.decision_memory import IDENTITY_NEGATIVES
    torch.manual_seed(4)
    c = memory_config(identity_dim=8)
    model = build_model(c).eval()
    batch = coordinate_batch(c, 4)
    x = {k: v for k, v in batch['x'].items() if not k.startswith('history_')}
    x.update(decision_inputs(c, 4))
    with torch.no_grad():
        ctx = model.context(x, batch['hist'], batch['hmask'])
    points = torch.zeros(4, 1+IDENTITY_NEGATIVES, 3)
    points[:, 0, 2], points[:, 1] = 1., torch.tensor([1.5, 0., 1.])
    point_mask = torch.zeros(4, 1+IDENTITY_NEGATIVES, dtype=torch.bool)
    point_mask[:, :2] = True
    anchor_mask = torch.zeros(4, SLOTS, dtype=torch.bool)
    anchor_mask[:, 0] = True
    inputs = dict(x, identity_points=points, identity_point_mask=point_mask, identity_anchor_mask=anchor_mask,
                  identity_anchor_features=torch.randn(4, SLOTS, c.hidden),
                  identity_fiber=torch.tensor([5, 5, 6, 5]), identity_row=torch.tensor([True, True, True, False]))
    with torch.no_grad():
        terms = model.identity_terms(ctx, inputs)
    # Rows 0/1 share fiber 5, so each has only row 2 as another fiber; row 3 is padding.
    assert terms['identity_anchor_valid'].tolist() == [True, True, True, False]
    assert terms['identity_pairs'][:, 0].tolist() == [True, True, True, False]
    # The control scores row r with row r-1's anchors only when that row is real and another fiber.
    assert terms['identity_control_pairs'][:, 0].tolist() == [False, False, True, False]
    assert torch.isfinite(terms['identity_loss_per_state']).all() and terms['identity_loss_per_state'][3] == 0
