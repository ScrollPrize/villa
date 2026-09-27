"""Visual identity: path patches, lateral-component negatives, augmentation, holdout and gradients."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.regression.data import (
    IdentityObservationBuilder, IdentitySampling, path_anchor, patch_layout,
)
from vesuvius.neural_tracing.fiber_follow.regression.model import (
    AppearanceEncoder, DirectConfig, DirectFollower, IdentityConfig, IdentityFollower,
)
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import load_checkpoint, optimizer_update, save_checkpoint
from vesuvius.neural_tracing.fiber_follow.shared.components import ComponentRule, lateral_components, sample_pairs
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig, TracedFiber, ZBand, crop_corners
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, crop_local_grid, sample_oriented_fast
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec


def config(**kwargs):
    return IdentityConfig(fine=CropSpec(depth=24, width=16, behind=8), coarse=CropSpec(depth=24, width=12, behind=18, spacing=2.),
                          channels=4, hidden=16, heads=2, layers=1, n_future=4, n_history=32, patch_span=32,
                          appearance_channels=4, embedding=8, **kwargs)


def batch(cfg, b=2):
    hist = torch.zeros(b, cfg.n_history, 3)
    hist[..., 2] = -torch.arange(1, cfg.n_history+1)
    q = 4*(cfg.n_future-1)+1
    K, M, P = 2, 3, cfg.n_patches
    x = {name: torch.rand(b, 2, crop.depth, crop.width, crop.width) for name, crop in (('fine', cfg.fine), ('coarse', cfg.coarse))}
    pc = cfg.patch_crop
    x.update(patches=torch.rand(b, P, pc.depth, pc.width, pc.width), patch_geometry=torch.randn(b, P, 11), patch_mask=torch.ones(b, P))
    points = torch.zeros(b, K*(1+M), 3)
    points[:, :K, 2] = torch.tensor([2., 3.])
    points[:, K:, 0], points[:, K:, 2] = 3., 2.
    return dict(x=x, hist=hist, hmask=torch.ones(b, cfg.n_history), dense_ab=torch.zeros(b, q, 2),
                dense_mask=torch.ones(b, q), offtrack=torch.zeros(b), endpoint_known=torch.zeros(b),
                end_local=torch.zeros(b, 3), source=torch.zeros(b), identity_points=points,
                positive_mask=torch.ones(b, K), negative_mask=torch.ones(b, K, M), patch_on_fiber=torch.ones(b, P),
                foreign=torch.zeros(b, cfg.fine.depth, cfg.fine.width, cfg.fine.width, dtype=torch.uint8))


def forward(m, b):
    return m(b['x'], b['hist'], b['hmask'], queries=b['identity_points'])


@pytest.mark.parametrize('version, expected_shape', [(1, (7, 33)), (2, (9, 41))])
def test_patch_embedding_equals_dense_map_at_the_same_place(version, expected_shape):
    """Receptive field equals the patch and the norm is per voxel, so both paths agree exactly."""
    torch.manual_seed(0)
    encoder = AppearanceEncoder(6, 5, version).double()
    from vesuvius.neural_tracing.fiber_follow.regression.model import appearance_receptive_field
    along, lateral = appearance_receptive_field(version)
    assert (along, lateral) == expected_shape
    a, l = along//2, lateral//2
    crop = torch.rand(1, 1, 2*a+14, 2*l+18, 2*l+18, dtype=torch.float64)
    dense = encoder(crop)
    for z, y, x in ((a+3, l+2, l+8), (a+9, l+12, l+12)):
        patch = crop[:, :, z-a:z+a+1, y-l:y+l+1, x-l:x+l+1]
        torch.testing.assert_close(encoder(patch, same=False)[0, :, 0, 0, 0].double(), dense[0, :, z, y, x].double())


def test_default_fine_crop_fits_negative_patches_at_the_lateral_limit():
    cfg, rule = IdentityConfig(), ComponentRule()
    curve = np.c_[np.zeros(21), np.zeros(21), np.arange(21.)]
    foreign = np.array([[-rule.lateral_max, 0., 10.], [rule.lateral_max, 0., 10.]])
    presence = np.ones((cfg.fine.depth, cfg.fine.width, cfg.fine.width), np.float32)
    _, positive_mask, negative, negative_mask = sample_pairs(
        curve, presence, cfg.fine, foreign, np.full(2, 10), np.random.default_rng(0),
        positives=1, negatives=2, forward=(10., 10.),
        margin=(cfg.patch_crop.width-1)*cfg.patch_crop.spacing/2, rule=rule)
    assert positive_mask.all() and negative_mask.all()
    corners = negative[0, :, None]+crop_corners(cfg.patch_crop)[None]
    bounds = crop_corners(cfg.fine)
    assert (corners >= bounds.min(0)).all() and (corners <= bounds.max(0)).all()


def test_path_frame_patches_follow_the_curved_path_and_sample_its_own_frame():
    cfg = replace(config(), anchor_patches=0)
    # History bends in the u-c plane: patch frames follow the local tangent.
    arc = np.arange(1, cfg.n_history+1, dtype=float)
    hist = np.stack((.02*arc**2, 0*arc, -arc), -1)
    item = patch_layout(dict(pos=np.zeros(3), frame=np.eye(3), hist_local=hist, hmask=np.ones(cfg.n_history)), cfg)
    k = np.arange(cfg.recent_patches)*cfg.patch_every+cfg.patch_every-1
    expected = np.stack((-.04*(k+1), 0*k, np.ones_like(k, dtype=float)), -1)
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(item['patch_frames'][:, :, 2], expected, atol=.03)
    np.testing.assert_allclose(item['patch_frames'][:, 1], np.tile([0., 1., 0.], (len(k), 1)), atol=1e-9)  # v stays v
    np.testing.assert_array_equal(item['patch_centers'], hist[k])
    # Reading uses each patch's world frame; compare against the reference oriented sampler.
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 256, (1, 1, 64, 64, 64), dtype=np.uint8)
    vol = SimpleNamespace(input_scale=1., raw_block=lambda start, size: np.zeros((1, *size), np.uint8))
    from vesuvius.neural_tracing.fiber_follow.shared import crop_sampling
    pos = np.array([32., 32., 44.])
    item = patch_layout(dict(pos=pos, frame=np.eye(3), hist_local=hist, hmask=np.ones(cfg.n_history)), cfg)
    builder = IdentityObservationBuilder(cfg)
    original = crop_sampling.read_tight_blocks
    crop_sampling.read_tight_blocks = lambda items, *a, **kw: ([raw[0]]*len(items), [np.zeros(3, np.int64)]*len(items))
    try:
        patches = builder.patch_inputs([item], vol)['patches'][0]
    finally:
        crop_sampling.read_tight_blocks = original
    grid = torch.from_numpy(crop_local_grid(cfg.patch_crop)).float()
    p = 5
    reference = sample_oriented_fast(torch.from_numpy(raw), torch.zeros(1, 3),
                                     torch.from_numpy(pos+item['patch_centers'][p])[None].float(),
                                     torch.from_numpy(item['patch_frames'][p])[None].float(), grid)
    torch.testing.assert_close(patches[p], reference[0, 0], atol=3e-4, rtol=1e-3)


def tube(shape, crop, center, radius=.9, along=None):
    grid = crop_local_grid(crop)
    d = np.linalg.norm(grid[..., :2]-np.asarray(center), axis=-1)
    inside = d <= radius
    if along is not None:
        inside &= (grid[..., 2] >= along[0]) & (grid[..., 2] <= along[1])
    return inside


def test_lateral_rule_gap_touching_neighbor_and_disconnected_neighbor():
    crop = CropSpec(depth=48, width=32, behind=8, spacing=.5)
    shape = (crop.depth, crop.width, crop.width)
    curve = np.stack((np.zeros(121), np.zeros(121), np.linspace(-4, 12, 121)), -1)  # annotation ends at c=12
    presence = np.zeros(shape, np.float32)
    presence[tube(shape, crop, (0, 0), along=(-4, 3))] = 1  # traced fiber ...
    presence[tube(shape, crop, (0, 0), along=(4.5, 16))] = 1  # ... gap, then continuation past the annotation
    presence[tube(shape, crop, (2, 0), along=(-4, 12))] = 1  # touching neighbor merges with the traced fiber
    presence[tube(shape, crop, (-4, 0), along=(-4, 12))] = 1  # disconnected lateral neighbor
    presence[tube(shape, crop, (0, -4), along=(14, 19))] = 1  # disconnected piece only ahead of the annotation end
    found = lateral_components(presence, crop, curve, ComponentRule(lateral_max=6.))
    foreign = found['foreign']
    grid = crop_local_grid(crop)
    assert foreign.any()
    lateral = grid[foreign]
    np.testing.assert_allclose(lateral[:, 0].mean(), -4, atol=.2)  # only the disconnected neighbor
    assert not foreign[tube(shape, crop, (0, 0), along=(4.5, 16))].any()  # gap continuation is never negative
    assert not foreign[tube(shape, crop, (2, 0))].any()
    assert not foreign[tube(shape, crop, (0, -4), along=(14, 19))].any()  # ahead of the end is excluded
    assert lateral[:, 2].max() < 12.01
    rng = np.random.default_rng(0)
    pos, pos_mask, neg, neg_mask = sample_pairs(curve, presence, crop, found['local'], found['nearest'], rng,
                                                positives=3, negatives=4, margin=2.)
    assert pos_mask.all() and neg_mask.all()
    np.testing.assert_allclose(neg[..., 0], -4, atol=1.)
    assert np.abs(neg[..., 2]-pos[:, None, 2]).max() <= 2.+crop.spacing/2  # same place along the fiber (voxel rounding)


def real_like_builder(cfg, fibers, augment, **kwargs):
    return IdentityObservationBuilder(cfg, fibers, IdentitySampling(**kwargs), augment=augment)


def fake_images(builder, items):
    cfg = builder.cfg
    rng = np.random.default_rng(3)
    x = {name: torch.from_numpy(rng.random((len(items), 2, c.depth, c.width, c.width), np.float32))
         for name, c in (('fine', cfg.fine), ('coarse', cfg.coarse))}
    for item in items:
        patch_layout(item, cfg)
    pc = cfg.patch_crop
    x.update(patches=torch.from_numpy(rng.random((len(items), cfg.n_patches, pc.depth, pc.width, pc.width), np.float32)),
             patch_geometry=torch.from_numpy(np.stack([i['patch_geometry'] for i in items])),
             patch_mask=torch.from_numpy(np.stack([i['patch_mask'] for i in items])))
    return x


def line_fiber(length=600.):
    arc = np.arange(length)
    return TracedFiber('line', np.c_[arc*0+100, arc*0+100, arc+200], arc, '')


def prepared(builder, fiber, rng, t=400.):
    from vesuvius.neural_tracing.fiber_follow.shared.data import make_sample
    cfg = builder.cfg
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future, recent_history_points=cfg.n_history,
                          no_history_prob=0., short_history_prob=0.)
    item = make_sample(fiber, t, False, sample, rng)
    item.update(fiber_ref=(0, t, False), source=0, source_step=-1, stratum=-1)
    return builder.prepare(item, fiber, rng)


def test_photometric_augmentation_is_independent_and_absent_at_inference(monkeypatch):
    cfg = config()
    fiber = line_fiber()
    for augment in (False, True):
        builder = real_like_builder(cfg, [fiber], augment, presence_dropout=0.)
        rng = np.random.default_rng(5)
        items = [prepared(builder, fiber, rng) for _ in range(2)]
        images = fake_images(builder, items)
        raw = {k: v.clone() for k, v in images.items()}
        monkeypatch.setattr(IdentityObservationBuilder, 'images', lambda self, items, vol, pool=None: images)
        out = builder(items, None)['x']
        if not augment:
            assert 'appearance' not in out and torch.equal(out['patches'], raw['patches'])
            continue
        (crop_params, patch_params) = items[0]['photometric']
        assert crop_params != patch_params
        assert not torch.equal(out['appearance'], raw['fine'][:, :1]) and not torch.equal(out['patches'], raw['patches'])
        torch.testing.assert_close(out['fine'], raw['fine'])  # localization input is untouched
        # Contrast is applied separately: crop and patch residuals scale differently.
        a = (out['appearance'][0, 0]-out['appearance'][0, 0].mean()).std()/(raw['fine'][0, 0]-raw['fine'][0, 0].mean()).std()
        valid = raw['patch_mask'][0] > 0
        b = out['patches'][0, valid].std()/raw['patches'][0, valid].std()
        assert abs(float(a)-float(b)) > 1e-3


def test_presence_dropout_after_targets(monkeypatch):
    cfg = config()
    fiber = line_fiber()
    builder = real_like_builder(cfg, [fiber], True, presence_dropout=1.)
    items = [prepared(builder, fiber, np.random.default_rng(2))]
    images = fake_images(builder, items)
    monkeypatch.setattr(IdentityObservationBuilder, 'images', lambda self, items, vol, pool=None: images)
    seen = []
    original = IdentityObservationBuilder.identity_targets
    def targets(self, items, x):
        seen.append(float(x['fine'][:, 1].abs().sum()))
        return original(self, items, x)
    monkeypatch.setattr(IdentityObservationBuilder, 'identity_targets', targets)
    out = builder(items, None)
    assert seen[0] > 0 and out['x']['fine'][:, 1].abs().sum() == 0 and out['x']['coarse'][:, 1].abs().sum() == 0
    assert out['presence_dropped'].tolist() == [1.]


def test_holdout_excludes_patch_footprints_beyond_the_crops():
    cfg = config()
    fiber = line_fiber(900.)
    builder = real_like_builder(cfg, [fiber], False, anchor_prob=1.)
    item = prepared(builder, fiber, np.random.default_rng(0), t=800.)
    assert 'anchor_world' in item
    z = item['anchor_world'][:, 2]
    # A band far from the head but covering the seed-segment anchors.
    band = ZBand(float(z.min()-1), float(z.max()+1))
    from vesuvius.neural_tracing.fiber_follow.shared.data import training_state_allowed
    assert all(training_state_allowed(item, crop, band) for crop in (cfg.fine, cfg.coarse))
    assert not builder.footprint_allowed(item, band)
    assert builder.footprint_allowed(item, ZBand(-100., -50.))


def test_seed_anchor_only_after_leaving_the_recent_span():
    cfg = config()
    segment = np.c_[np.zeros(40), np.zeros(40), np.arange(40.)]
    assert path_anchor(segment, cfg.anchor_offset-1, cfg) == {}
    anchor = path_anchor(segment, 500., cfg)
    np.testing.assert_allclose(anchor['anchor_world'][:, 2], np.arange(cfg.anchor_patches)*cfg.anchor_every)
    np.testing.assert_allclose(anchor['anchor_age'], 500-np.arange(cfg.anchor_patches)*cfg.anchor_every)


def test_gradients_reach_the_appearance_encoder_from_every_objective():
    torch.manual_seed(4)
    m = IdentityFollower(config())
    b = batch(m.cfg)
    out = forward(m, b)
    terms = loss_terms(out, b, m.cfg)
    assert terms['identity_count'] == 4
    for key in ('identity_per_state', 'geometry_per_state', 'confidence_per_state'):
        m.zero_grad(set_to_none=True)
        terms = loss_terms(forward(m, b), b, m.cfg)
        terms[key].sum().backward()
        grad = m.appearance.stem.conv.weight.grad
        assert grad is not None and grad.abs().sum() > 0, key
    # Masking all history patches keeps outputs finite (null attention token).
    b['x']['patch_mask'].zero_()
    assert all(torch.isfinite(v).all() for v in forward(m, b).values())


def test_identity_aware_labels_reject_foreign_points_within_tolerance():
    m = IdentityFollower(config())
    b = batch(m.cfg, 1)
    points = torch.zeros(1, 4, 3)
    points[..., 2] = torch.arange(1, 5)
    points[..., 0] = .5  # within tolerance of the annotation
    out = dict(forward(m, b), points=points)
    before = loss_terms(out, b, m.cfg)
    assert before['identity_correct_count'] == before['correct_count'] == 4
    crop = m.cfg.fine
    c = int(round(3/crop.spacing+crop.behind))
    a = int(round(.5/crop.spacing+(crop.width-1)/2))
    b['foreign'][0, c, :, a-1:a+2] = 1  # a lateral component under the third point
    after = loss_terms(out, b, m.cfg)
    assert after['correct_count'] == 4 and after['identity_correct_count'] == 2 and after['identity_flipped_count'] == 2


@pytest.mark.parametrize('appearance_version', [1, 2])
def test_identity_microbatches_and_checkpoint_roundtrip(tmp_path, appearance_version):
    torch.manual_seed(3)
    cfg = config(appearance_version=appearance_version)
    a = IdentityFollower(cfg)
    c = copy.deepcopy(a)
    data = batch(cfg, 3)
    def take(value, sl):
        return {k: take(v, sl) for k, v in value.items()} if isinstance(value, dict) else value[sl]
    opts = [torch.optim.SGD(m.parameters(), lr=.001) for m in (a, c)]
    results = [optimizer_update(a, copy.deepcopy(a), opts[0], [data], 1, .001),
               optimizer_update(c, copy.deepcopy(c), opts[1], [take(data, slice(0, 1)), take(data, slice(1, 3))], 1, .001)]
    assert results[0]['loss'] == pytest.approx(results[1]['loss'], rel=2e-5)
    assert results[0]['identity']['identity_count'] == results[1]['identity']['identity_count']
    spec = FiberVolumeSpec('unused', ct_zarr='unused', ct_level=0, ct_grid_scale=4., inputs='ct+presence')
    path = tmp_path/'identity.pt'
    save_checkpoint(path, a, a, spec, SampleConfig(crop=cfg.fine, n_history=cfg.n_history))
    if appearance_version == 1:
        ck = torch.load(path, weights_only=False)
        del ck['model_cfg']['appearance_version']  # legacy checkpoints predate the version field
        torch.save(ck, path)
    loaded, _, _, _, ck = load_checkpoint(path, 'cpu')
    assert ck['architecture'] == 'direct_identity_v1' and isinstance(loaded.cfg, IdentityConfig)
    assert loaded.cfg.appearance_version == appearance_version and loaded.cfg.patch_crop == cfg.patch_crop
    for key, value in forward(loaded, data).items():
        torch.testing.assert_close(value, forward(a.eval(), data)[key], rtol=0, atol=0)


def test_retired_rich_path_context_checkpoints_still_load(tmp_path):
    cfg = replace(DirectConfig(), fine=CropSpec(depth=16, width=12, behind=7), coarse=CropSpec(depth=24, width=12, behind=18, spacing=2.),
                  channels=4, hidden=16, heads=2, layers=1, n_future=4, n_history=32)
    m = DirectFollower(cfg)
    spec = FiberVolumeSpec('unused', ct_zarr='unused', ct_level=0, ct_grid_scale=4., inputs='ct+presence')
    path = tmp_path/'old.pt'
    save_checkpoint(path, m, m, spec, SampleConfig(crop=cfg.fine, n_history=cfg.n_history))
    ck = torch.load(path, weights_only=False)
    ck['model_cfg']['rich_path_context'] = True
    torch.save(ck, path)
    assert type(load_checkpoint(path, 'cpu')[0]) is DirectFollower
    ck['model_cfg']['rich_path_context'] = False
    torch.save(ck, path)
    with pytest.raises(ValueError, match='rich path context'):
        load_checkpoint(path, 'cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_bf16_identity_gradients():
    from vesuvius.neural_tracing.fiber_follow.regression.train import move_batch
    m = IdentityFollower(config()).cuda()
    b = move_batch(batch(m.cfg), 'cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        terms = loss_terms(forward(m, b), b, m.cfg)
        loss = sum(terms[k].mean() for k in ('geometry_per_state', 'confidence_per_state', 'identity_per_state'))
    loss.backward()
    assert torch.isfinite(loss) and m.appearance.stem.conv.weight.grad.abs().sum() > 0
