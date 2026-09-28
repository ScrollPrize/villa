"""The deployed classifier must have a direct, trainable ownership comparison."""
import torch

from test_identity import config
from test_identity_decisions import candidate_batch
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectFollower, sample_features
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms


def scene():
    model = DirectFollower(config(embedding=4)).eval()
    cfg = model.cfg
    with torch.no_grad():
        model.embedding.weight.copy_(torch.eye(4))
        model.embedding.bias.zero_()
    fine = torch.zeros(1, 4, cfg.fine.depth, cfg.fine.width, cfg.fine.width)
    fine[:, 0, :, :, :cfg.fine.width//2] = 1.
    fine[:, 1, :, :, cfg.fine.width//2:] = 1.
    refs = torch.zeros(1, cfg.n_history+1, 4)
    refs[:, :2, 1] = 1.  # recent history is on B
    refs[:, -1, 0] = 1.  # observed seed is on A
    mask = torch.zeros(1, cfg.n_history+1, dtype=torch.bool)
    mask[:, :2] = True; mask[:, -1] = True
    ctx = dict(fine=fine.requires_grad_(), reference_embedding=refs.requires_grad_(), reference_mask=mask)
    points = torch.zeros(1, cfg.n_future, 3)
    points[..., 0] = -2.; points[..., 2] = torch.arange(1, cfg.n_future+1)
    return model, ctx, points


def compare(model, ctx, points):
    return model.identity_evidence(ctx, *sample_features(ctx['fine'], points, model.cfg.fine))


def test_seed_comparison_survives_contaminated_history_and_changes_ownership():
    model, ctx, points = scene()
    before = compare(model, ctx, points)
    expected = torch.tensor([1., 0., 1., 0., 1., 0., 1., 1.]).expand_as(before)
    torch.testing.assert_close(before, expected)
    switched = dict(ctx, reference_embedding=ctx['reference_embedding'].detach().clone())
    switched['reference_embedding'][:, -1] = torch.tensor([0., 1., 0., 0.])
    after = compare(model, switched, points)
    torch.testing.assert_close(after[..., :6], torch.zeros_like(after[..., :6]))
    torch.testing.assert_close(before[..., [1, 3, 5, 7]], after[..., [1, 3, 5, 7]])
    # Reversing candidate order does not change a candidate's evidence.
    other = points.clone(); other[..., 0] = 2.
    torch.testing.assert_close(compare(model, ctx, other)[..., :2],
                               torch.tensor([0., 1.]).expand(1, model.cfg.n_future, 2))


def test_missing_references_and_unsupported_queries_are_not_false_mismatches():
    model, ctx, points = scene()
    absent = dict(ctx, reference_mask=torch.zeros_like(ctx['reference_mask']))
    evidence = compare(model, absent, points)
    assert evidence.eq(0).all()
    evidence.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert ctx['reference_embedding'].grad.eq(0).all()
    # An unsupported sample does not lower a valid prefix's mean/minimum;
    # the coverage explicitly reports that only half the prefix was observed.
    points[:, 1, 0] = 100.
    evidence = compare(model, ctx, points)
    torch.testing.assert_close(evidence[0, 1], torch.tensor([0., 0., 1., 0., 1., 0., .5, .5]))


def test_prefix_summary_exposes_departure_even_after_candidate_returns():
    model, ctx, points = scene()
    points[:, 1:3, 0] = 2.
    evidence = compare(model, ctx, points)
    # A, B, B, A: current point matches the seed again, but the prefix departed.
    torch.testing.assert_close(evidence[0, -1], torch.tensor([1., 0., .5, .5, 0., 0., 1., 1.]))


def test_confidence_can_use_seed_directly_with_identical_decoded_features():
    model, ctx, points = scene()
    # Isolate the new connection from all pre-existing visual/decoder paths.
    with torch.no_grad():
        for parameter in model.confidence_head.parameters():
            parameter.zero_()
        model.confidence_head[0].weight[0, 2*model.cfg.hidden+2] = 1.  # seed prefix mean
        model.confidence_head[-1].weight[0, 0] = 1.
    # Deep/path evidence is unchanged for either seed.
    ctx['deep'] = torch.zeros(1, model.cfg.hidden, *model.cfg.token_shape)
    decoded = torch.zeros(1, model.cfg.n_future, model.cfg.hidden)
    with torch.no_grad():
        ctx['reference_embedding'][:, -1] = torch.tensor([.8, .6, 0., 0.])
    a = model.confidence_logits(ctx, decoded, points)
    switched = dict(ctx, reference_embedding=ctx['reference_embedding'].detach().clone())
    switched['reference_embedding'][:, -1] = torch.tensor([0., 1., 0., 0.])
    b = model.confidence_logits(switched, decoded, points)
    assert (a > b+.5).all()
    a.sum().backward()
    assert ctx['fine'].grad.abs().sum() > 0
    assert model.embedding.weight.grad.abs().sum() > 0


def test_candidate_bce_trains_new_connection_and_shared_projection():
    torch.manual_seed(17)
    model = DirectFollower(config()).eval()
    data = candidate_batch(model.cfg)
    out = model(data['x'], data['hist'], data['hmask'], candidates=data['candidate_points'])
    loss_terms(out, data, model.cfg)['candidate_per_state'].sum().backward()
    assert model.confidence_head[0].weight.grad[:, 2*model.cfg.hidden:].abs().sum() > 0
    # Once the neutral connection has learned a nonzero weight, decision loss
    # must also train the shared embedding; coordinate targets stay detached.
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.confidence_head[0].weight[:, 2*model.cfg.hidden:].normal_(std=.02)
    candidates = data['candidate_points'].clone().requires_grad_()
    out = model(data['x'], data['hist'], data['hmask'], candidates=candidates)
    loss_terms(out, data, model.cfg)['candidate_per_state'].sum().backward()
    assert model.embedding.weight.grad.abs().sum() > 0
    assert candidates.grad is None


def test_reused_stencil_centre_matches_direct_point_sample_and_gradients():
    model, ctx, points = scene()
    ctx['deep'] = torch.zeros(1, model.cfg.hidden, *model.cfg.token_shape)
    points = points+.125  # exercise interpolation, not just voxel centres
    spatial = model.evidence(ctx, points, 'confidence')
    c = model.cfg.channels
    centre = (len(model.path_stencil)//2)*(c+1)
    shared = model.identity_evidence(ctx, spatial[..., centre:centre+c], spatial[..., centre+c].bool())
    direct = compare(model, ctx, points)
    torch.testing.assert_close(shared, direct, rtol=0, atol=0)
    a = torch.autograd.grad(shared.sum(), ctx['fine'], retain_graph=True)[0]
    b = torch.autograd.grad(direct.sum(), ctx['fine'])[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
