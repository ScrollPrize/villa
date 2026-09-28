"""Visual identity: path patches, centerline negatives, augmentation, holdout and gradients."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.regression.data import (
    IdentityObservationBuilder, IdentitySampling, reference_layout,
)
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig, DirectFollower

from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import load_checkpoint, optimizer_update, save_checkpoint
from vesuvius.neural_tracing.fiber_follow.shared.components import ComponentRule, sample_pairs
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig, TracedFiber, ZBand, crop_corners
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, crop_local_grid, sample_oriented_fast
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec


def config(**kwargs):
    options=dict(fine=CropSpec(depth=24,width=17,behind=8),channels=4,hidden=16,
        heads=2,layers=1,decoder_layers=1,n_future=4,n_history=32,embedding=8,activation_checkpointing=False)
    options.update(kwargs)
    return DirectConfig(**options)


def batch(cfg,b=2):
    hist=torch.zeros(b,cfg.n_history,3)
    hist[...,2]=-torch.arange(1,cfg.n_history+1)
    q=4*(cfg.n_future-1)+1
    K,M=2,3
    x=dict(fine=torch.rand(b,2,cfg.fine.depth,cfg.fine.width,cfg.fine.width),
        seed=torch.zeros(b,1,3),seed_mask=torch.ones(b,1),seed_tangent=torch.tensor([0.,0.,1.]).expand(b,-1),
        seed_age=torch.zeros(b))
    points=torch.zeros(b,K*(1+M),3)
    points[:,:K,2]=torch.tensor([2.,3.])
    points[:,K:,0],points[:,K:,2]=3.,2.
    return dict(x=x,hist=hist,hmask=torch.ones(b,cfg.n_history),dense_ab=torch.zeros(b,q,2),
        dense_mask=torch.ones(b,q),offtrack=torch.zeros(b),endpoint_known=torch.zeros(b),
        end_local=torch.zeros(b,3),source=torch.zeros(b),identity_points=points,
        positive_mask=torch.ones(b,K),negative_mask=torch.ones(b,K,M),reference_on_fiber=torch.ones(b,cfg.n_history+1),
        foreign=torch.zeros(b,cfg.fine.depth,cfg.fine.width,cfg.fine.width,dtype=torch.uint8))


def forward(m, b):
    return m(b['x'], b['hist'], b['hmask'], queries=b['identity_points'])






def test_bank_radius_keeps_supported_diagonals_and_rejects_crop_edge_patches():
    cfg, rule = DirectConfig(), ComponentRule()
    from vesuvius.neural_tracing.fiber_follow.regression.neighbor_mining import MiningConfig
    assert rule.lateral_max == MiningConfig().max_distance
    curve = np.c_[np.zeros(21), np.zeros(21), np.arange(21.)]
    diagonal = rule.lateral_max/np.sqrt(2.)
    foreign = np.array([[-diagonal, -diagonal, 10.], [diagonal, diagonal, 10.],
                        [rule.lateral_max, 0., 10.], [9., 9., 10.]])
    presence = np.ones((cfg.fine.depth, cfg.fine.width, cfg.fine.width), np.float32)
    _, positive_mask, negative, negative_mask = sample_pairs(
        curve, presence, cfg.fine, foreign, np.full(len(foreign), 10), np.random.default_rng(0),
        positives=1, negatives=4, forward=(10., 10.),
        margin=cfg.patch_radius, rule=rule)
    assert positive_mask.all() and negative_mask.sum() == 3
    selected = negative[0, negative_mask[0] > 0]
    np.testing.assert_allclose(np.linalg.norm(selected[:,:2],axis=1),12.,atol=1e-6)
    corners = selected[:, None]+np.array([[-1,-1,-1],[1,1,1]])[None]*cfg.patch_radius
    bounds = crop_corners(cfg.fine)
    assert (corners >= bounds.min(0)).all() and (corners <= bounds.max(0)).all()




def tube(shape, crop, center, radius=.9, along=None):
    grid = crop_local_grid(crop)
    d = np.linalg.norm(grid[..., :2]-np.asarray(center), axis=-1)
    inside = d <= radius
    if along is not None:
        inside &= (grid[..., 2] >= along[0]) & (grid[..., 2] <= along[1])
    return inside


def real_like_builder(cfg, fibers, augment, **kwargs):
    class EmptyBank:
        shard_count = 0
        def candidates(self,item,crop,presence,rule,**kwargs):
            return dict(foreign=np.zeros_like(presence,bool),local=np.empty((0,3)),nearest=np.empty(0,int),
                        path_ids=np.empty(0,int),
                        counts=dict(foreign_components=0))
    return IdentityObservationBuilder(cfg, fibers, IdentitySampling(**kwargs), augment=augment,negative_bank=EmptyBank())


def fake_images(builder,items):
    cfg=builder.cfg
    for item in items: reference_layout(item,cfg)
    stack=lambda key: torch.from_numpy(np.stack([i[key] for i in items]).astype(np.float32))
    return dict(fine=torch.from_numpy(np.random.default_rng(3).random((len(items),2,cfg.fine.depth,cfg.fine.width,cfg.fine.width),np.float32)),
        seed=stack('visible_seed'),seed_mask=stack('visible_seed_mask'),seed_tangent=stack('visible_seed_tangent'),seed_age=stack('visible_seed_age'))


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
    assert seen[0] > 0 and out['x']['fine'][:, 1].abs().sum() == 0
    assert out['presence_dropped'].tolist() == [1.]


def test_monitor_observations_need_no_negative_bank_or_identity_labels(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.shared.data import make_sample
    cfg = config()
    builder = IdentityObservationBuilder(cfg)
    sample = SampleConfig(crop=cfg.fine,n_history=cfg.n_history,n_future=cfg.n_future)
    items = [make_sample(line_fiber(),400.,False,sample,np.random.default_rng(0))]
    images = fake_images(builder,items)
    monkeypatch.setattr(IdentityObservationBuilder,'images',lambda *a,**kw:images)
    result = builder(items,None)
    assert set(result['x']) == {'fine','seed','seed_mask','seed_age','seed_tangent'} and 'dense_mask' in result
    assert 'identity_points' not in result and 'negative_mask' not in result






def test_gradients_reach_the_shared_encoder_from_every_objective():
    torch.manual_seed(4)
    m = DirectFollower(config())
    b = batch(m.cfg)
    out = forward(m, b)
    terms = loss_terms(out, b, m.cfg)
    assert terms['identity_count'] == 4
    for key in ('identity_per_state', 'geometry_per_state', 'confidence_per_state'):
        m.zero_grad(set_to_none=True)
        terms = loss_terms(forward(m, b), b, m.cfg)
        terms[key].sum().backward()
        grad = m.encoder.stem[0].weight.grad
        assert grad is not None and grad.abs().sum() > 0, key
    # Masking all history patches keeps outputs finite (null attention token).
    b['hmask'].zero_(); b['x']['seed_mask'].zero_()
    assert all(torch.isfinite(v).all() for v in forward(m, b).values())


def test_identity_aware_labels_reject_foreign_points_within_tolerance():
    m = DirectFollower(config())
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






@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_bf16_identity_gradients():
    from vesuvius.neural_tracing.fiber_follow.regression.train import move_batch
    m = DirectFollower(config()).cuda()
    b = move_batch(batch(m.cfg), 'cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        terms = loss_terms(forward(m, b), b, m.cfg)
        loss = sum(terms[k].mean() for k in ('geometry_per_state', 'confidence_per_state', 'identity_per_state'))
    loss.backward()
    assert torch.isfinite(loss) and m.encoder.stem[0].weight.grad.abs().sum() > 0
