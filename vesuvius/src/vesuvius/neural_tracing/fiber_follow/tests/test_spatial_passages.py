"""Contracts for CT-only spatial proposals, identity labels and short commits."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.data import plane_targets, SampleConfig, TracedFiber, ZBand
from vesuvius.neural_tracing.fiber_follow.direct.spatial_model import (
    SpatialConfig, SpatialFollower, choose_passage, extract_passages, SPATIAL_ARCHITECTURE)
from vesuvius.neural_tracing.fiber_follow.direct.spatial_supervision import (
    passage_labels, teaching_candidates, spatial_loss_terms)
from vesuvius.neural_tracing.fiber_follow.direct.train import optimizer_update, save_checkpoint, load_checkpoint
from vesuvius.neural_tracing.fiber_follow.direct.contacts import ContactIndex, SpatialObservationBuilder
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume, FiberVolumeSpec


def config():
    return SpatialConfig(fine=CropSpec(depth=24, width=20, behind=6, spacing=.5),
        coarse=CropSpec(depth=24, width=12, behind=12, spacing=2.),
        n_future=6, n_history=16, channels=4, hidden=16, heads=2, layers=1, candidates=3, peak_count=5)


def batch(cfg=None, b=2):
    cfg = cfg or config()
    h = torch.zeros(b, cfg.n_history, 3)
    h[..., 2] = -torch.arange(1, cfg.n_history+1)
    n, q = cfg.n_future, 4*(cfg.n_future-1)+1
    x = {name: torch.rand(b, 1, c.depth, c.width, c.width)
         for name, c in [('fine', cfg.fine), ('coarse', cfg.coarse), ('seed_ct', cfg.seed_crop)]}
    x.update(frontier=torch.zeros(b, 3), frontier_direction=torch.tensor([0., 0., 1.]).expand(b, -1),
             seed_metadata=torch.tensor([0., 0., -1., 0., 0., 1., 1.]).expand(b, -1))
    return dict(x=x, hist=h, hmask=torch.ones(b, cfg.n_history), plane_ab=torch.zeros(b, n, 2),
                plane_mask=torch.ones(b, n), dense_ab=torch.zeros(b, q, 2), dense_mask=torch.ones(b, q),
                offtrack=torch.zeros(b), endpoint_known=torch.zeros(b), end_local=torch.zeros(b, 3),
                neighbor_ab=torch.zeros(b, 2, q, 2), neighbor_mask=torch.zeros(b, 2, q), source=torch.zeros(b))


def curves(data, cfg, k=1):
    z = torch.arange(1, cfg.n_future+1).float()*cfg.future_step
    p = torch.zeros(len(data['hist']), k, cfg.n_future, 3)
    p[..., 2] = z
    return p


def test_spatial_gradients_and_training_candidates_never_condition_heatmaps():
    torch.manual_seed(7)
    cfg, data = config(), batch()
    model = SpatialFollower(cfg)
    teacher = teaching_candidates(data, cfg)
    first = model(data['x'], data['hist'], data['hmask'], teacher)
    other = model(data['x'], data['hist'], data['hmask'], teacher+1)
    torch.testing.assert_close(first['heatmap_logits'], other['heatmap_logits'], rtol=0, atol=0)
    torch.testing.assert_close(first['candidate_points'], other['candidate_points'], rtol=0, atol=0)
    terms = spatial_loss_terms(first, data, cfg)
    (terms['geometry_per_state'].mean()+terms['confidence_per_state'].mean()).backward()
    for weight in (model.fine_encoder.local[0].weight, model.coarse_encoder.local[0].weight,
                   model.history_token[0].weight, model.spatial_key[0].weight, model.passage_score[-1].weight,
                   model.seed_projection.weight):
        assert weight.grad is not None and torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0


def test_neighbor_identity_overrides_loose_localization_tolerance_and_return():
    cfg, data = config(), batch(b=1)
    data['neighbor_ab'][:, 0, :, 0] = 1.
    data['neighbor_mask'][:, 0] = 1
    p = curves(data, cfg, 2)
    p[:, 1, 2:4, 0] = 1.
    y, known = passage_labels(p, data, cfg, tolerance=1.5)
    assert y[0, 0].eq(1).all() and known.all()
    assert y[0, 1, :2].eq(1).all() and y[0, 1, 2:].eq(0).all()


def test_unknown_suffix_censoring_invalid_sets_and_physical_endpoint():
    cfg, data = config(), batch(b=3)
    data['dense_mask'][0] = 0
    data['offtrack'][1] = 1
    data['dense_mask'][2, 9:] = 0
    data['endpoint_known'][2] = 1
    data['end_local'][2, 2] = 3.
    y, known = passage_labels(curves(data, cfg), data, cfg)
    assert not known[0].any()
    assert known[1].all() and not y[1].any()
    assert y[2, 0, :3].eq(1).all() and y[2, 0, 3:].eq(0).all() and known[2].all()


def test_safe_prefix_can_advance_before_unresolved_distant_fork():
    cfg, data = config(), batch(b=1)
    p = curves(data, cfg, 2)
    p[:, 1, 3:, 0] = 2.
    logits = torch.full((1, 2, cfg.n_future), 3.)
    logits[:, 0, -1], logits[:, 1, -1] = -2., -2.2
    curve, confidence, chosen = choose_passage(p, logits, torch.ones(1, 2, dtype=torch.bool), cfg)
    assert confidence[0, :3].gt(.9).all()
    assert confidence[0, 3:].lt(.5).all()
    assert chosen.item() == 0
    torch.testing.assert_close(curve, p[:, 0])


def test_full_passage_ranking_cannot_ignore_an_earlier_rejection():
    cfg, data = config(), batch(b=1)
    p = curves(data, cfg, 2)
    logits = torch.full((1, 2, cfg.n_future), 2.)
    logits[:, 0, 0], logits[:, 0, -1] = -4., 5.
    _, confidence, chosen = choose_passage(p, logits, torch.ones(1, 2, dtype=torch.bool), cfg)
    assert chosen.item() == 1 and confidence.min() > .8


def test_equally_valid_complete_variants_do_not_require_arbitrary_ranking():
    cfg, data = config(), batch(b=1)
    p = curves(data, cfg, 2)
    p[:, 1, :, 0] = 2.
    _, confidence, _ = choose_passage(p, torch.full((1, 2, cfg.n_future), 4.),
                                      torch.ones(1, 2, dtype=torch.bool), cfg)
    assert confidence.min() > .95


def test_contact_fraction_applies_to_samples_at_larger_microbatches(monkeypatch):
    builder = SpatialObservationBuilder.__new__(SpatialObservationBuilder)
    builder.contacts = SimpleNamespace(episodes=[1])
    builder.contact_fraction = 1.
    monkeypatch.setattr(builder, '_contact_pair', lambda rng, count: [dict(contact=1.) for _ in range(count)])
    assert len(builder.contact_batch(np.random.default_rng(0), 8)) == 8


def test_seed_ct_and_provenance_are_immutable_across_tracing_calls(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.direct.data import DirectTracer
    import vesuvius.neural_tracing.fiber_follow.direct.data as direct_data
    tracer = DirectTracer.__new__(DirectTracer)
    tracer.model, tracer.device, tracer.vol = SimpleNamespace(cfg=config()), 'cpu', None
    tracer._spatial_seeds = [None]
    reads = []
    def sample(items, vol, crop):
        reads.append(items[0]['pos'].copy())
        return torch.ones(1, 1, crop.depth, crop.width, crop.width)
    monkeypatch.setattr(direct_data, 'scalar_crops', sample)
    first = tracer.condition_inputs({}, np.zeros((1, 3)), np.eye(3)[None], [0])
    next_input = tracer.condition_inputs({}, np.array([[0., 0., 20.]]), np.eye(3)[None], [0])
    assert len(reads) == 1
    np.testing.assert_array_equal(tracer._spatial_seeds[0]['pos'], np.zeros(3))
    torch.testing.assert_close(first['seed_ct'], next_input['seed_ct'])
    assert next_input['seed_metadata'][0, 2] == -20/128


def test_search_preserves_distinct_exits_and_bounds_geometry():
    cfg = config()
    w = cfg.output_width
    axis = torch.linspace(-cfg.lateral_limit, cfg.lateral_limit, w)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    grid = torch.stack((xx, yy), -1).reshape(-1, 2)
    logits = torch.full((1, cfg.n_future, w, w), -20.)
    logits[:, :, w//2, 1] = 5
    logits[:, :, w//2, w-2] = 5
    candidates, valid, pool, pool_valid = extract_passages(logits, grid, torch.zeros(1, 3), cfg)
    exits = candidates[0, valid[0], -1, 0]
    assert exits.min() < -1 and exits.max() > 1
    assert (torch.diff(candidates[0, valid[0], :, :2], dim=1).norm(dim=-1) <= cfg.lateral_step).all()


def test_multiple_crossings_are_explicitly_unknown():
    p = np.array([[0., 0., 0.], [0., 0., 3.], [1., 0., 1.], [1., 0., 5.]])
    s = np.r_[0., np.linalg.norm(np.diff(p, axis=0), axis=-1).cumsum()]
    _, legacy = plane_targets(p, s, 0., s[-1], np.zeros(3), np.eye(3), np.array([2., 4.]))
    _, strict = plane_targets(p, s, 0., s[-1], np.zeros(3), np.eye(3), np.array([2., 4.]), True)
    assert legacy.tolist() == [1., 1.] and strict.tolist() == [0., 1.]


def test_ct_only_volume_never_opens_presence_and_derives_bounds(monkeypatch):
    import vesuvius.neural_tracing.fiber_follow.volume as volume
    opened = []
    class Array:
        def __init__(self, path, cache_bytes):
            opened.append(path)
            self.shape, self.dtype, self.cache_bytes = (100, 120, 140), np.dtype('uint8'), cache_bytes
    monkeypatch.setattr(volume, 'ChunkedArray', Array)
    monkeypatch.setattr(volume, '_find_channel_zarr', lambda *args: (_ for _ in ()).throw(AssertionError('presence opened')))
    v = FiberVolume(FiberVolumeSpec('/missing', ct_zarr='/ct', inputs='ct', ct_level=0, ct_grid_scale=4, load_presence=False))
    assert v.shape == (50, 60, 70) and v.presence is None and opened == ['/ct/0']


def test_spatial_checkpoint_and_optimizer_roundtrip(tmp_path):
    cfg, data = config(), batch()
    m = SpatialFollower(cfg)
    ema = copy.deepcopy(m)
    opt = torch.optim.AdamW(m.parameters(), lr=.001)
    report = optimizer_update(m, ema, opt, [data], 1, .001, n_commit=4)
    assert np.isfinite(report['loss']) and report['scorer_known_prefixes'] > 0
    sample = SampleConfig(crop=cfg.fine, n_future=cfg.n_future, n_history=cfg.n_history, unique_crossings=True)
    spec = FiberVolumeSpec('/none', ct_zarr='/ct', inputs='ct', load_presence=False)
    path = tmp_path/'spatial.pt'
    save_checkpoint(path, m, ema, spec, sample)
    loaded, _, _, source, ck = load_checkpoint(path, 'cpu')
    assert ck['architecture'] == SPATIAL_ARCHITECTURE and not source.load_presence
    ema.eval()
    torch.testing.assert_close(loaded(data['x'], data['hist'], data['hmask'])['heatmap_logits'],
                               ema(data['x'], data['hist'], data['hmask'])['heatmap_logits'])


def test_prompt_swap_fits_two_targets_in_identical_ct():
    torch.manual_seed(4)
    cfg, data = config(), batch()
    for key in ('fine', 'coarse', 'seed_ct'):
        data['x'][key][1] = data['x'][key][0]
    for i, offset in enumerate((-2., 2.)):
        data['x']['frontier'][i, 0] = offset
        data['hist'][i, :, 0] = offset
        data['plane_ab'][i, :, 0] = offset
        data['dense_ab'][i, :, 0] = offset
    m = SpatialFollower(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=.008)
    for _ in range(45):
        opt.zero_grad(set_to_none=True)
        out = m(data['x'], data['hist'], data['hmask'])
        terms = spatial_loss_terms(out, data, cfg, compute_metrics=False)
        terms['geometry_per_state'].mean().backward()
        opt.step()
    peak = out['heatmap_logits'].flatten(2).argmax(-1)
    x = m.lateral_grid[peak][..., 0]
    assert x[0].mean() < -1 and x[1].mean() > 1
    assert torch.equal(data['x']['fine'][0], data['x']['fine'][1])
