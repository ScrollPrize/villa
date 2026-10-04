"""Memory-conditioned identity verification: targets, balance, warm start, gradients and memory augmentation."""
from types import SimpleNamespace as NS

import numpy as np
import torch

from model_fixtures import coordinate_batch
from test_decision_memory import memory_config, decision_inputs
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.models.decision_memory import SLOTS
from vesuvius.neural_tracing.fiber_follow.models.identity_verifier import (
    IDENTITY_PLANES, IDENTITY_CANDIDATES, IDENTITY_SAMPLES, balance_weights, update_balance, verification_loss)


def verify_builder(cfg):
    from vesuvius.neural_tracing.fiber_follow.data.observations import IdentityObservationBuilder
    builder = IdentityObservationBuilder.__new__(IdentityObservationBuilder)
    builder.cfg = cfg
    builder.fibers = [NS(points=np.c_[np.zeros(300), np.zeros(300), np.arange(300.)])]
    builder.sampling = NS(rule=NS(own_radius=1.5), on_fiber_tolerance=1.5)
    return builder


def test_verification_queries_are_dense_and_labels_come_from_the_original_fiber_curve():
    from vesuvius.neural_tracing.fiber_follow.data.observations import SOURCE
    from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
    from vesuvius.neural_tracing.fiber_follow.models.identity_verifier import (
        IDENTITY_SAMPLES, SAMPLE_MIX, verification_buffers, curve_labels)
    builder = verify_builder(NS(fine=CropSpec(depth=48, width=24, behind=8, spacing=1.)))
    # The original fiber runs along x=-5 through the crop; the trace (head axis, history) runs along x=0.
    z = np.arange(-60., 120., .25)
    curve = np.c_[np.full(len(z), -5.), np.zeros(len(z)), z]
    ab = np.tile([-5., 0.], (16, 1))
    neighbor = np.c_[np.full(40, 4.), np.zeros(40), np.arange(-8., 32.)]
    found = dict(local=np.concatenate((neighbor, np.c_[np.zeros(40), np.zeros(40), np.arange(-8., 32.)])),
                 path_ids=np.repeat([3, 4], 40))
    history = np.c_[np.zeros(32), np.zeros(32), -np.arange(1., 33.)]
    item = dict(planes=np.arange(1., 17.), plane_mask=np.ones(16), plane_ab=ab, fiber_ref=(0, 100., False),
                source=SOURCE['synthetic'], _leave_arc=10., _constructed_arc=np.array([0., 90.]), identity_seed=11,
                identity_curve=curve, hist_local=history, hmask=np.ones(32),
                _memory_entries=[dict(age=80., pos=np.array([0., 0., 20.])), dict(age=40., pos=np.array([9., 0., 60.]))])
    out = verification_buffers(1)
    builder.verification_targets(item, found, out, 0)
    assert out['identity_synthetic'][0] and out['identity_departure_age'][0] == 80.
    assert out['identity_anchor_mask'][0].tolist() == [True]+[False]*(SLOTS-1)
    assert not out['identity_curve_ends'][0].any()  # the annotation continues past both crop faces
    # Crossings: original, head axis, then neighbors nearest first (the trace's own path x=0, then x=4).
    np.testing.assert_allclose(out['identity_candidates'][0, 0, :4], [[-5., 0., 2.], [0., 0., 2.], [0., 0., 2.], [4., 0., 2.]])
    assert out['identity_mask'][0, :, :4].all() and not out['identity_mask'][0, :, 4:].any()
    samples = out['identity_samples'][0][out['identity_sample_mask'][0]]
    assert len(samples) > .85*IDENTITY_SAMPLES
    t = lambda value: torch.as_tensor(np.asarray(value))[None]
    label, known = curve_labels(t(samples), t(out['identity_curve'][0]), t(out['identity_curve_mask'][0]), t(out['identity_curve_ends'][0]))
    # Dense positives along the original fiber (its share of the mix, jittered inside the positive tube).
    assert label.sum() >= .9*SAMPLE_MIX['original'] and known.float().mean() > .8
    again = verification_buffers(1)
    builder.verification_targets(item, found, again, 0)
    np.testing.assert_array_equal(out['identity_samples'], again['identity_samples'])  # seeded per state
    # No memory entry on the original fiber: memory cannot describe it, so the row is inert.
    out = verification_buffers(1)
    builder.verification_targets(dict(item, _memory_entries=[dict(age=80., pos=np.array([9., 0., 20.]))]), found, out, 0)
    assert not out['identity_mask'].any() and not out['identity_sample_mask'].any()


def test_curve_labels_leave_the_band_and_annotation_ends_unknown_and_path_strata_follow_the_trace():
    from vesuvius.neural_tracing.fiber_follow.models.identity_verifier import curve_labels, path_stratum
    curve = torch.stack((torch.zeros(21), torch.zeros(21), torch.arange(21.)), -1)[None]
    points = torch.tensor([[[1., 0., 5.], [2., 0., 5.], [4., 0., 5.], [0., 0., 25.], [5., 0., 25.], [0., 0., 20.5]]])
    for ends, expected_known in ((torch.tensor([[False, False]]), [True, False, True, True, True, True]),
                                 (torch.tensor([[False, True]]), [True, False, True, False, False, True])):
        label, known = curve_labels(points, curve, torch.ones(1, 21, dtype=torch.bool), ends)
        assert label[0].tolist() == [True, False, False, False, False, True]
        assert known[0].tolist() == expected_known
    hist = torch.tensor([[[0., 0., -1.], [0., 0., -2.]]])
    predicted = torch.tensor([[[3., 0., 1.], [3., 0., 2.]]])
    on = path_stratum(torch.tensor([[[0., 0., -1.5], [3., 0., 1.5], [0., 0., 2.]]]), hist, torch.ones(1, 2), predicted)
    assert on[0].tolist() == [True, True, False]


def test_balance_gives_each_label_half_of_its_stratum_and_the_loss_is_finite_with_masked_queries():
    labels = torch.tensor([[True, True, False, False]])
    on_path = torch.tensor([[False, True, True, False]])
    valid = torch.tensor([[True, True, True, False]])
    balance = torch.ones(2, 2)
    for _ in range(400):  # on-path stratum: one positive per negative; off-path: positives only
        balance = update_balance(balance, labels, on_path, valid)
    weights = balance_weights(labels, on_path, valid, balance)
    torch.testing.assert_close(weights[0, 1], weights[0, 2])
    assert weights[0, 3] == 0 and weights[0, 0] < 1
    # A heavily positive on-path stratum is reweighted so both labels carry equal mass.
    skewed = torch.tensor([[0., 0.], [1., 9.]])
    w = balance_weights(torch.tensor([True, False]), torch.tensor([True, True]), torch.tensor([True, True]), skewed)
    torch.testing.assert_close(w[0]*.9, w[1]*.1)
    # Candidates (1 plane x 4) followed by two dense samples.
    label = torch.tensor([[True, False, False, True, True, False]])
    known = torch.tensor([[True, True, True, False, True, True]])
    targets = dict(label=label, known=known, train=known, on_path=torch.zeros_like(label), candidate_label=label[:, :4].reshape(1, 1, 4),
                   candidate_known=known[:, :4].reshape(1, 1, 4))
    logits = torch.randn(1, 6, requires_grad=True)
    loss, planes = verification_loss(logits, targets, balance)
    loss.sum().backward()
    assert planes.all() and torch.isfinite(logits.grad).all() and logits.grad[0, 3] == 0


def test_verifier_warm_start_is_exact_and_its_loss_trains_memory_tokens_but_never_the_field():
    torch.manual_seed(5)
    base = build_model(memory_config()).eval()
    cfg = memory_config(identity_objective='verify', identity_map=True, identity_feedback=True)
    model = build_model(cfg).eval()
    missing = model.load_state_dict(base.state_dict(), strict=False).missing_keys
    assert {k.split('.')[0] for k in missing} == {'identity_verifier', 'identity_embedding', 'identity_feedback',
                                                   'confidence_scorer'}
    batch = coordinate_batch(cfg, 2)
    x = {k: v for k, v in batch['x'].items() if not k.startswith('history_')}
    x.update(decision_inputs(cfg, 2))
    x['history_valid'][:, :2] = True
    with torch.no_grad():
        before, after = base(x, batch['hist'], batch['hmask']), model(x, batch['hist'], batch['hmask'])
    for key in ('points', 'confidence'):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    model.train()
    # Geometry/confidence outputs reach the field embedding but never the verifier.
    out = model(x, batch['hist'], batch['hmask'])
    (out['points'].sum()+out['confidence'].sum()).backward()
    assert all(p.grad is None or not p.grad.any() for p in model.identity_verifier.parameters())
    model.zero_grad(set_to_none=True)
    from vesuvius.neural_tracing.fiber_follow.models.identity_verifier import verification_targets
    candidates = torch.zeros(2, len(IDENTITY_PLANES), IDENTITY_CANDIDATES, 3)
    candidates[..., 2], candidates[..., 2:, 0] = 2., 4.
    samples = (torch.rand(2, IDENTITY_SAMPLES, 3)-.5)*torch.tensor([10., 10., 14.])
    predicted = torch.zeros(2, cfg.n_future, 3)
    predicted[..., 2] = torch.arange(1., cfg.n_future+1)
    ctx = model.context(x, batch['hist'], batch['hmask'])
    terms = model.verification_terms(ctx, dict(x, identity_candidates=candidates, identity_samples=samples,
                                                identity_fiber=torch.tensor([4, 4]), identity_row=torch.tensor([True, True])),
                                     predicted)
    assert not terms['identity_shuffled_valid'].any()  # the other row is the same fiber
    mask = torch.zeros(2, len(IDENTITY_PLANES), IDENTITY_CANDIDATES, dtype=torch.bool)
    mask[..., :3] = True
    curve = torch.stack((torch.zeros(40), torch.zeros(40), torch.linspace(-10., 10., 40)), -1).expand(2, -1, -1)
    inputs = dict(
        identity_mask=mask, identity_sample_mask=torch.ones(2, IDENTITY_SAMPLES, dtype=torch.bool), identity_curve=curve,
        identity_curve_mask=torch.ones(2, 40, dtype=torch.bool), identity_curve_ends=torch.zeros(2, 2, dtype=torch.bool),
        hist=batch['hist'], hmask=batch['hmask'], identity_synthetic=torch.tensor([True, False]))
    targets = verification_targets(terms, inputs)
    # Excluded synthetic states keep their labels for the metrics but leave the loss.
    excluded = verification_targets(terms, inputs, exclude_synthetic=True)
    assert torch.equal(excluded['known'], targets['known']) and torch.equal(excluded['train'][1], targets['known'][1])
    assert targets['known'][0].any() and not excluded['train'][0].any()
    # The original crossing and head axis lie on the curve; the crossing at x=4 is off it; predictions are on it.
    assert targets['candidate_label'][..., :3].tolist() == [[[True, True, False]]*3]*2
    assert targets['label'][:, targets['predicted']].all() and targets['on_path'][:, targets['predicted']].all()
    loss, planes = verification_loss(terms['identity_logits'], targets, torch.ones(2, 2))
    assert planes.all()
    loss.sum().backward()
    for module in (model.identity_verifier, model.history_encoder, model.encoder):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.identity_embedding.parameters())


def test_independent_memory_augmentation_draws_per_slot_and_is_reproducible():
    from vesuvius.neural_tracing.fiber_follow.data.observations import IdentitySampling, IdentityObservationBuilder
    builder = IdentityObservationBuilder.__new__(IdentityObservationBuilder)
    encode = torch.zeros(2, SLOTS, dtype=torch.bool)
    encode[0, [1, 3]] = encode[1, 2] = True
    item = dict(identity_seed=7, photometric=(1.2, .05, 0.), blur_sigma=0.)
    def augmented(mode, row=0):
        builder.sampling = IdentitySampling(memory_augmentation=mode, blur_probability=0.)
        x = dict(history_encode=encode, history_crops=torch.ones(3, 1, 4, 4, 4)*torch.arange(4.))
        builder.augment_memory_crops(x, row, item, np.random.default_rng(8))
        return x['history_crops']
    shared, independent = augmented('shared'), augmented('independent')
    # Shared: both of row 0's crops get the current crop's contrast/brightness; row 1's crop is untouched.
    torch.testing.assert_close(shared[0], shared[1])
    assert torch.equal(shared[2], torch.ones(1, 4, 4, 4)*torch.arange(4.))
    assert not torch.allclose(independent[0], independent[1])
    torch.testing.assert_close(independent, augmented('independent'))
