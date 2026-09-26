"""Categorical localization and identity-aware passage/prefix validity targets."""
import torch
import torch.nn.functional as F


def plane_mask(batch, cfg):
    ab = batch['plane_ab']
    known = batch['plane_mask'].bool() & ~batch['offtrack'][:, None].bool()
    known &= torch.isfinite(ab).all(-1) & (ab.abs().amax(-1) <= cfg.lateral_limit)
    frontier = batch['x'].get('frontier', ab.new_zeros(len(ab), 3))
    planes = frontier[:, 2, None]+torch.arange(1, cfg.n_future+1, device=ab.device)*cfg.future_step
    known &= (planes >= -cfg.fine.behind*cfg.fine.spacing) & (planes <= (cfg.fine.depth-cfg.fine.behind-1)*cfg.fine.spacing)
    return known


@torch.no_grad()
def passage_labels(points, batch, cfg, tolerance=1.5):
    """Known wrong anywhere stays wrong after return; missing coverage is unknown.

    In addition to localization tolerance, a candidate loses the seeded identity
    if it follows a separately validated neighbor more closely than the target.
    Exact/indistinguishable overlaps supply no invented pairwise ordering.
    """
    b, k, n, _ = points.shape
    q = batch['dense_ab'].shape[1]
    dense = F.interpolate(points[..., :2].reshape(b*k, n, 2).transpose(1, 2), size=q,
                          mode='linear', align_corners=True).transpose(1, 2).reshape(b, k, q, 2)
    target = batch['dense_ab'][:, None]
    known = batch['dense_mask'].bool() & torch.isfinite(batch['dense_ab']).all(-1)
    known &= batch['dense_ab'].abs().amax(-1) <= cfg.lateral_limit
    error = (dense-target).norm(dim=-1)
    frontier = batch['x'].get('frontier', points.new_zeros(b, 3))
    forward = frontier[:, 2, None]+torch.linspace(cfg.future_step, cfg.future_step*n, q, device=points.device)
    known &= (forward <= (cfg.fine.depth-cfg.fine.behind-1)*cfg.fine.spacing)
    beyond = batch['endpoint_known'].bool()[:, None] & ~known & (forward > batch['end_local'][:, 2, None]+1e-4)
    beyond &= batch['end_local'][:, 2, None] >= frontier[:, 2, None]
    failure = known[:, None] & ((error > tolerance) | ~torch.isfinite(error))
    if 'neighbor_ab' in batch:
        neighbors = batch['neighbor_ab'][:, None]
        distance = (dense[:, :, None]-neighbors).norm(dim=-1)
        separation = (neighbors-target[:, :, None]).norm(dim=-1)
        neighbor_known = batch['neighbor_mask'].bool()[:, None] & known[:, None, None]
        switched = neighbor_known & (separation > .5) & (distance+.25 < error[:, :, None]) & (distance <= tolerance)
        failure |= switched.any(2)
    connection = points[:, :, 0]-frontier[:, None]
    bad_connection = (~torch.isfinite(connection).all(-1) | (connection.norm(dim=-1) > cfg.max_recovery_distance)
                      | (connection[..., 2] <= 0))
    lateral_step = torch.diff(points[..., :2], dim=2).norm(dim=-1)
    invalid_step = F.pad(lateral_step > cfg.lateral_step+1e-4, (1, 0))
    step_bad = F.interpolate(invalid_step.float().reshape(b*k, 1, n), size=q, mode='nearest').reshape(b, k, q).bool()
    failure |= beyond[:, None] | batch['offtrack'][:, None, None].bool() | bad_connection[..., None] | step_bad
    failed = failure.cumsum(-1) > 0
    supported = (known | beyond).int().cummin(-1).values.bool()[:, None].expand(-1, k, -1)
    indices = torch.linspace(0, q-1, n, device=points.device).round().long()
    labels = (~failed[..., indices]).float()
    masks = (failed | supported)[..., indices]
    return labels, masks


@torch.no_grad()
def teaching_candidates(batch, cfg):
    """Annotation proposals bootstrap scoring only; never condition the predictor."""
    target = torch.nan_to_num(batch['plane_ab']).clone()
    b, n, _ = target.shape
    device = target.device
    frontier = batch['x'].get('frontier', target.new_zeros(b, 3))
    z = frontier[:, 2, None]+torch.arange(1, n+1, device=device)*cfg.future_step
    # Both positive and negative examples have smooth perturbations, preventing
    # an exact-GT versus noisy-generated geometry shortcut.
    noise = torch.randn(b, 2, 2, device=device)
    noise = F.interpolate(noise, size=n, mode='linear', align_corners=True).transpose(1, 2)
    good = target+.15*noise
    shifted = target+noise*2.
    paths = [good, shifted]
    neighbor = target.clone()
    known_neighbor = torch.zeros(b, n, dtype=torch.bool, device=device)
    if 'neighbor_ab' in batch:
        neighbor = F.interpolate(batch['neighbor_ab'][:, 0].transpose(1, 2), size=n,
                                 mode='linear', align_corners=True).transpose(1, 2)
        known_neighbor = F.interpolate(batch['neighbor_mask'][:, :1], size=n, mode='nearest')[:, 0] > 0
        neighbor = torch.where(known_neighbor[..., None], neighbor, shifted)
    separation = (neighbor-target).norm(dim=-1).masked_fill(~known_neighbor, torch.inf)
    switch = separation.argmin(-1).clamp(min=1, max=max(1, n-4))
    age = torch.arange(n, device=device)[None]-switch[:, None]
    weight = ((age+2)/4).clamp(0, 1)
    weight = weight.square()*(3-2*weight)
    paths.append(target*(1-weight[..., None])+neighbor*weight[..., None]+.15*noise)
    returning = weight*(1-((age-5)/4).clamp(0, 1))
    paths.append(target*(1-returning[..., None])+neighbor*returning[..., None]+.15*noise)
    return torch.cat((torch.stack(paths, 1), z[:, None, :, None].expand(-1, 4, -1, -1)), -1)


def spatial_loss_terms(output, batch, cfg, tolerance=1.5, n_commit=4, compute_metrics=True):
    logits = output['heatmap_logits'].float()
    axis = torch.linspace(-cfg.lateral_limit, cfg.lateral_limit, cfg.output_width, device=logits.device)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    grid = torch.stack((xx, yy), -1)
    mask = plane_mask(batch, cfg)
    target = torch.where(mask[..., None], batch['plane_ab'], 0.)
    distance = (grid[None, None]-target[:, :, None, None]).square().sum(-1)
    distribution = (-distance/(2*cfg.target_sigma**2)).flatten(2).softmax(-1)
    ce = -(distribution*logits.flatten(2).log_softmax(-1)).sum(-1)
    geometry = (ce*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    points = output['candidate_points']
    labels, known = passage_labels(points, batch, cfg, tolerance)
    known &= output['candidate_valid'][..., None]
    all_logits, all_labels, all_known = [output['candidate_logits']], [labels], [known]
    present = [output['candidate_valid'][..., None].expand_as(known)]
    if 'teacher_logits' in output:
        teacher_labels, teacher_known = passage_labels(output['teacher_points'], batch, cfg, tolerance)
        all_logits.append(output['teacher_logits']); all_labels.append(teacher_labels); all_known.append(teacher_known)
        present.append(torch.ones_like(teacher_known))
    all_logits, all_labels, all_known = [torch.cat(values, 1) for values in (all_logits, all_labels, all_known)]
    bce = F.binary_cross_entropy_with_logits(all_logits.float(), all_labels, reduction='none')
    # Equal state weight; half on immediate prefixes, half on all evaluated extents.
    def mean(part):
        return (bce[..., part]*all_known[..., part]).sum((1, 2))/all_known[..., part].sum((1, 2)).clamp_min(1)
    confidence = .5*mean(slice(None, n_commit))+.5*mean(slice(None))
    selected = output['selected_candidate']
    idx = torch.arange(len(selected), device=selected.device)
    selected_labels, selected_known = labels[idx, selected], known[idx, selected]
    error = (output['points'][..., :2]-batch['plane_ab']).norm(dim=-1)
    result = dict(geometry_per_state=geometry, confidence_per_state=confidence,
                  error_sum=torch.where(mask, error, 0.).sum(), geometry_count=mask.sum(),
                  correct_count=(selected_labels*selected_known).sum(), confidence_count=selected_known.sum())
    if compute_metrics:
        pool_labels, pool_known = passage_labels(output['pool_points'], batch, cfg, tolerance)
        pool_known &= output['pool_valid'][..., None]
        end = cfg.n_future-1
        supported_state = mask.all(-1) & ~batch['offtrack'].bool()
        oracle = (labels[..., end].bool() & known[..., end]).any(1)
        before = (pool_labels[..., end].bool() & pool_known[..., end]).any(1)
        # Invalid padded proposals do not make an otherwise known pool unknown.
        all_invalid = ((known[..., end] | ~output['candidate_valid']).all(1)
                       & ~(labels[..., end].bool() & known[..., end]).any(1) & output['candidate_valid'].any(1))
        chosen_ok = selected_labels[:, end].bool() & selected_known[:, end]
        metrics = dict(passage_states=supported_state.sum(), proposal_recall_before_count=(before & supported_state).sum(),
                       proposal_recall_after_count=(oracle & supported_state).sum(), selection_opportunities=oracle.sum(),
                       selected_valid_count=(chosen_ok & oracle).sum(), all_invalid_sets=all_invalid.sum(),
                       all_invalid_rejected=(all_invalid & (output['confidence'][:, end] < .5)).sum(),
                       unknown_candidate_prefixes=(~all_known & torch.cat(present, 1)).sum(),
                       padded_candidate_prefixes=(~torch.cat(present, 1)).sum(), scorer_known_prefixes=all_known.sum(),
                       scorer_positive_prefixes=(all_labels*all_known).sum(),
                       contact_states=batch.get('contact', torch.zeros_like(selected)).sum(),
                       plane_censored=(~mask).sum())
        result['spatial_metrics'] = {k: v.detach() for k, v in metrics.items()}
    return result
