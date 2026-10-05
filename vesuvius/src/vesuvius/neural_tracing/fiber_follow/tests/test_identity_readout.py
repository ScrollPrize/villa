"""Path-blind readout identity memory: entry layout, warm start, path blindness, gradients and tracer parity."""
import pytest
import torch

from model_fixtures import coordinate_batch
from test_decision_memory import memory_config, decision_inputs, StraightModel, run_tracer
from vesuvius.neural_tracing.fiber_follow.models.model import build_model, CoordinateRegressionConfig
from vesuvius.neural_tracing.fiber_follow.models.decision_memory import SLOTS, ENTRY_FEATURES
from vesuvius.neural_tracing.fiber_follow.models.identity_readout import (
    KEY_POINTS, KEY_DIM, key_rows, pack_keys, unpack_keys)
from vesuvius.neural_tracing.fiber_follow.models.identity_verifier import (
    IDENTITY_PLANES, IDENTITY_CANDIDATES, IDENTITY_SAMPLES)


def readout_config(**kwargs):
    return memory_config(**dict(dict(identity_objective='readout', identity_map=True, identity_feedback=True), **kwargs))


def inputs(c, b, valid=2):
    batch = coordinate_batch(c, b)
    x = {k: v for k, v in batch['x'].items() if not k.startswith('history_')}
    x.update(decision_inputs(c, b))
    x['history_valid'][:, :valid] = True
    x['history_path_valid'][:, :valid] = True
    x['history_features'] = torch.randn(b, SLOTS, *c.memory_entry_shape).to(torch.bfloat16)
    return x, batch['hist'], batch['hmask']


def queries(b):
    return dict(identity_candidates=torch.randn(b, len(IDENTITY_PLANES), IDENTITY_CANDIDATES, 3),
                identity_samples=torch.randn(b, IDENTITY_SAMPLES, 3)*3)


def test_config_entry_layout_and_key_packing():
    assert CoordinateRegressionConfig().memory_entry_shape == (ENTRY_FEATURES, 256)
    c = readout_config()
    assert c.identity_mode == 'verify' and c.memory_entry_shape == (ENTRY_FEATURES+key_rows(c.hidden), c.hidden)
    with pytest.raises(ValueError):
        memory_config(identity_objective='readout', memory='slabs')
    keys = torch.randn(3, KEY_POINTS, KEY_DIM)
    torch.testing.assert_close(unpack_keys(pack_keys(keys, c.hidden)), keys, rtol=0, atol=0)


def test_warm_start_is_exact_and_entries_equal_entries_encoded_from_their_crops():
    torch.manual_seed(7)
    base = build_model(memory_config()).eval()
    c = readout_config()
    model = build_model(c).eval()
    missing = model.load_state_dict(base.state_dict(), strict=False).missing_keys
    assert {k.split('.')[0] for k in missing} == {'identity_readout', 'identity_readout_embedding',
                                                   'identity_feedback', 'confidence_scorer'}
    x, hist, hmask = inputs(c, 2)
    with torch.no_grad():
        before = base(dict(x, history_features=x['history_features'][:, :, :ENTRY_FEATURES]), hist, hmask)
        after = model(x, hist, hmask)
    for key in ('points', 'confidence'):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    assert after['memory_entry'].shape == (2, *c.memory_entry_shape)
    torch.testing.assert_close(after['memory_entry'][:, :ENTRY_FEATURES], before['memory_entry'], rtol=0, atol=0)
    # The same decision's entry, encoded later from its crop, is identical (encoder rows and keys).
    one = {k: v[:1] for k, v in x.items()}
    with torch.no_grad():
        entry = model(one, hist[:1], hmask[:1])['memory_entry']
        references, mask = model.references(one, hist[:1], hmask[:1])
        later = {k: v for k, v in one.items() if not k.startswith('history_')}
        later.update(decision_inputs(c, 1, crops=1))
        later['history_valid'][0, 0] = later['history_encode'][0, 0] = True
        later['history_crops'][:] = one['fine']
        later['history_references'][:], later['history_reference_mask'][:] = references, mask.float()
        later['history_path_points'][0, 0] = torch.stack((hist[0, 1], hist[0, 0], torch.zeros(3)))
        later['history_path_valid'][0, 0] = True
        encoded = model.memory_features(later)
    torch.testing.assert_close(encoded[0, 0], entry[0], rtol=0, atol=0)
    assert unpack_keys(entry[:, ENTRY_FEATURES:]).abs().sum(-1).gt(0).any()  # some keys lie inside the crop


def test_keys_and_field_are_blind_to_the_observed_path_and_invalid_slots_are_inert():
    torch.manual_seed(9)
    c = readout_config()
    model = build_model(c).eval()
    x, hist, hmask = inputs(c, 2)
    shifted = hist.clone()
    shifted[..., 0] += 2.
    with torch.no_grad():
        first = model.context(x, hist, hmask)
        moved = model.context(x, shifted, hmask)
    torch.testing.assert_close(first['identity_keys'], moved['identity_keys'], rtol=0, atol=0)
    torch.testing.assert_close(first['identity_field'], moved['identity_field'], rtol=0, atol=0)
    assert not torch.equal(first['deep'], moved['deep'])  # the path-conditioned lattice does move
    # Invalid slots never enter: NaN features and pose there change nothing.
    with torch.no_grad():
        reference = model(x, hist, hmask)
        x['history_features'][:, 2:] = float('nan')
        x['history_pose'][:, 2:] = float('nan')
        x['history_path_points'][:, 2:] = float('nan')
        other = model(x, hist, hmask)
    for key in ('points', 'hazard_logits', 'refinement_points', 'memory_entry'):
        torch.testing.assert_close(reference[key], other[key], rtol=0, atol=0)


def test_identity_loss_trains_keys_stem_and_readout_and_decoder_reaches_the_map():
    torch.manual_seed(11)
    c = readout_config()
    model = build_model(c).train()
    x, hist, hmask = inputs(c, 2)
    x.update(queries(2))
    out = model.training_forward(x, hist, hmask, torch.tensor(.5))
    assert out['identity_logits'].shape == (2, len(IDENTITY_PLANES)*IDENTITY_CANDIDATES+IDENTITY_SAMPLES+c.n_future)
    for key in ('identity_shuffled_logits', 'identity_empty_logits'):
        assert not out[key].requires_grad and torch.isfinite(out[key]).all()
    out['identity_logits'].sum().backward()
    grad = lambda m: sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
    for module in (model.identity_readout.key_head, model.identity_readout.key, model.encoder.stem,
                   model.encoder.patch_projection):
        assert grad(module) > 0
    # The identity loss never sees the path-conditioned lattice or the decision-memory tokens.
    for module in (model.encoder.blocks, model.encoder.condition, model.encoder.position, model.history_encoder):
        assert grad(module) == 0
    model.zero_grad(set_to_none=True)
    out = model(x, hist, hmask)
    (out['points'].sum()+out['confidence'].sum()).backward()
    assert grad(model.identity_readout_embedding) > 0


def test_training_prediction_passes_the_readout_memory_with_the_history_tokens():
    from vesuvius.neural_tracing.fiber_follow.train.train import prepare_training, training_prediction
    torch.manual_seed(13)
    c = readout_config()
    model = build_model(c).train()
    x, hist, hmask = inputs(c, 2)
    targets = dict(queries(2), fiber_id=torch.tensor([3, 4]))
    reference = model.training_forward(dict(x, **{k: targets[k] for k in ('identity_candidates', 'identity_samples')}),
                                       hist, hmask, torch.tensor(.5))
    prepare_training(model, 2, backend='eager')  # full-graph capture of the readout
    out = training_prediction(model, x, hist, hmask, targets=targets)
    for key in ('memory_entry', 'identity_logits', 'identity_shuffled_logits', 'identity_empty_logits'):
        torch.testing.assert_close(out[key], reference[key], rtol=0, atol=1e-5)
    torch.testing.assert_close(out['refinement_points'], reference['refinement_points'], rtol=0, atol=1e-6)
    assert out['identity_shuffled_valid'].all()  # two real rows of different fibers


def test_tracer_reuses_readout_entries_exactly_as_encoding_their_crops(monkeypatch):
    torch.manual_seed(17)
    model = build_model(readout_config(n_future=4)).eval()
    with torch.no_grad():  # a non-zero field embedding, so memory content reaches the outputs
        model.identity_readout_embedding[-1].weight.normal_(std=.1)
    cached, fresh = StraightModel(model), StraightModel(model)
    run_tracer(cached, monkeypatch)
    run_tracer(fresh, monkeypatch, recorded=False)
    assert len(cached.raw) == len(fresh.raw) > 10
    for (p0, c0, v0), (p1, c1, v1) in zip(cached.raw, fresh.raw):
        torch.testing.assert_close(v0, v1, rtol=0, atol=0)
        torch.testing.assert_close(p0, p1, rtol=0, atol=1e-5)
        torch.testing.assert_close(c0, c1, rtol=0, atol=1e-5)
