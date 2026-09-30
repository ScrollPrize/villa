"""Annotation-only objectives for v5; no teacher candidates enter model inputs.

Coverage counts planes whose shortlist holds a node within one proposal spacing
of the annotated crossing. It is a diagnostic only; there is no shortlist loss.
"""
import torch
import torch.nn.functional as F


def proposal_loss_terms(output, batch, cfg):
    logits = output['proposal_logits'].float()
    target = batch.get('route_ab', batch.get('plane_ab'))
    mask = batch.get('route_mask', batch.get('plane_mask'))
    if target is None:
        target = F.interpolate(batch['dense_ab'].transpose(1, 2), size=cfg.n_future,
                               mode='linear', align_corners=True).transpose(1, 2)
        mask = F.interpolate(batch['dense_mask'][:, None].float(), size=cfg.n_future, mode='nearest')[:, 0]
    mask = mask.bool() & torch.isfinite(target).all(-1) & (target.abs().amax(-1) <= cfg.lateral_limit)
    if 'route_mask' not in batch:
        mask &= ~batch['offtrack'][:, None].bool()
    if 'identity_observable' in batch:
        mask &= batch['identity_observable'][:, None].bool()
    target = torch.where(mask[..., None], target.float(), 0.)
    grid = output['proposal_grid'][..., :2]
    n = round(grid.shape[1]**.5)
    spacing = 2*cfg.lateral_limit/(n-1)
    difference = target[:, :, None]-grid
    weights = (1-difference.abs()/spacing).clamp_min(0.).prod(-1)
    weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-8)
    classification = -(weights*logits.log_softmax(-1)).sum(-1)
    # Only sites whose own half-cell contains the target learn its offset.
    # Boundary ties are averaged; no clipped target pulls unrelated sites.
    assigned = (difference.abs().amax(-1) <= spacing/2+1e-6) & mask[..., None]
    errors = F.smooth_l1_loss(output['proposal_offsets'], difference, beta=.25,
                             reduction='none').mean(-1)
    offset = torch.where(assigned, errors, 0.).sum(-1)/assigned.sum(-1).clamp_min(1)
    locations = output['proposal_positions'].detach()
    distances = (locations[..., :2]-target[:, :, None]).norm(dim=-1)
    available = output['proposal_valid'].bool()
    covered = (available & (distances <= spacing)).any(-1)

    def average(values, known):
        return torch.where(known, values, 0.).sum(-1)/known.sum(-1).clamp_min(1)

    proposal = average(classification, mask)
    offset = average(offset, mask)
    initial = (output['initial_points'][..., :2]-target).norm(dim=-1)
    final = (output['points'][..., :2]-target).norm(dim=-1)
    nearest = distances.masked_fill(~available, 1e4).amin(-1)
    terms = dict(proposal_per_state=cfg.proposal_loss_weight*proposal+cfg.proposal_offset_weight*offset,
                 proposal_classification_per_state=proposal, proposal_offset_per_state=offset,
                 proposal_known_count=mask.sum(), proposal_covered_count=(covered & mask).sum(),
                 proposal_nearest_error_sum=torch.where(mask, nearest, 0.).sum(),
                 proposal_initial_error_sum=torch.where(mask, initial, 0.).sum(),
                 proposal_final_error_sum=torch.where(mask, final, 0.).sum(),
                 proposal_disconnected_count=(~output['proposal_connected']).sum())
    # Plane-level error/coverage by annotated tangent change; all states, no confidence filtering.
    if cfg.n_future >= 3:
        xyz = torch.cat((target, output['points'][..., 2:].detach()), -1)
        tangent = F.normalize(xyz[:, 1:]-xyz[:, :-1], dim=-1)
        angle = (tangent[:, 1:]*tangent[:, :-1]).sum(-1).clamp(-1, 1).acos()*180/torch.pi
        known = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]
        for name, low, high in (('under10', 0., 10.), ('10to30', 10., 30.), ('over30', 30., 181.)):
            chosen = known & (angle >= low) & (angle < high)
            terms['proposal_bend_'+name+'_count'] = chosen.sum()
            for key, values in (('initial', initial), ('final', final), ('nearest', nearest)):
                terms['proposal_bend_'+name+'_'+key+'_error_sum'] = torch.where(chosen, values[:, 1:-1], 0.).sum()
    return terms
