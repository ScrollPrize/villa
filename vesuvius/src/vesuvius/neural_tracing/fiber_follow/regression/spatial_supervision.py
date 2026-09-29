"""Annotation-only targets for memory-conditioned localization.

An off-track head can still teach where its original fiber is, without being
given permission to connect to it. Missing crossings remain unknown.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .spatial_model import route_lattice
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig, continuation_targets


def prepare_route_targets(item, fiber, cfg):
    _, t, reverse = item['fiber_ref']
    sample = SampleConfig(crop=cfg.fine,n_history=cfg.n_history,n_future=cfg.n_future,
                          future_step=cfg.future_step,recent_history_points=cfg.n_history)
    target = continuation_targets(fiber,t,reverse,item['pos'],item['frame'],sample,offtrack=False)
    ab = np.asarray(target['plane_ab'],np.float32)
    valid = (target['plane_mask'] > 0) & np.isfinite(ab).all(-1) & (np.abs(ab).max(-1) <= cfg.lateral_limit)
    valid &= item.get('identity_observable',True)
    item['route_ab'] = np.where(valid[:,None],ab,0.)
    item['route_mask'] = valid


def route_loss_terms(output, batch, cfg):
    logits = output['route_logits'].float()
    # The fallback supports existing synthetic batches; real data uses explicit
    # targets, including localization-only targets at departed states.
    target = batch.get('route_ab',batch.get('plane_ab'))
    mask = batch.get('route_mask',batch.get('plane_mask'))
    if target is None:
        target = F.interpolate(batch['dense_ab'].transpose(1,2),size=cfg.n_future,mode='linear',align_corners=True).transpose(1,2)
        mask = F.interpolate(batch['dense_mask'][:,None].float(),size=cfg.n_future,mode='nearest')[:,0]
    mask = mask.bool() & torch.isfinite(target).all(-1) & (target.abs().amax(-1) <= cfg.lateral_limit)
    if 'route_mask' not in batch:
        mask &= ~batch['offtrack'][:,None].bool()
    if 'identity_observable' in batch:
        mask &= batch['identity_observable'][:,None].bool()
    target = torch.where(mask[...,None],target.float(),0.)
    xy, n = route_lattice(cfg)
    xy = xy.to(logits)
    spacing = 2*cfg.lateral_limit/(n-1)
    # Bilinear target weights avoid teaching a jump at cell boundaries. Decoding
    # remains max-sum, so separate fiber modes are never averaged at inference.
    distance = (target[:,:,None]-xy).abs()/spacing
    weights = (1-distance).clamp_min(0.).prod(-1)
    weights = weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
    ce = -(weights*logits.log_softmax(-1)).sum(-1)
    per_state = torch.where(mask,ce,0.).sum(-1)/mask.sum(-1).clamp_min(1)
    chosen = xy[logits.argmax(-1)]
    return dict(route_per_state=per_state,route_count=mask.sum(),
                route_error_sum=torch.where(mask,(chosen-target).norm(dim=-1),0.).sum())
