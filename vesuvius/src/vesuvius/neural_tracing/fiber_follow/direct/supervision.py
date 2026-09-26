"""Direct dense curve regression plus correctness of the produced prefix."""
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.supervision import prefix_labels


def geometry_mask(batch, cfg):
    annotated = batch['dense_mask'].bool() & ~batch['offtrack'][:, None].bool()
    target = torch.where(annotated[..., None], batch['dense_ab'], 0.)
    observable = (target.abs().amax(-1) <= cfg.lateral_limit).int().cummin(-1).values.bool()
    return annotated & observable


def commit_window(cfg, n_commit):
    window = min(4, cfg.n_future) if n_commit is None else n_commit
    if not 1 <= window <= cfg.n_future:
        raise ValueError('Commit window must fit forecast')
    return window


def dense_commit_mask(count, cfg, n_commit, device):
    planes = torch.linspace(cfg.future_step, cfg.n_future*cfg.future_step, count, device=device)
    return planes <= n_commit*cfg.future_step+1e-6


def window_mean(values, mask, near):
    """Half commit-window, half full-horizon; empty windows contribute zero."""
    def mean(selected):
        return torch.where(selected, values, 0.).sum(-1)/selected.sum(-1).clamp_min(1)
    return .5*mean(mask & near)+.5*mean(mask)


def loss_terms(output, batch, cfg, tolerance=1.5, *, n_commit=None):
    """Return numerators/counts so effective-batch means are independent of microbatch.

    Unknown/crop-censored targets and departed states do not teach localization.
    Confirmed departures still teach rejection. Prefix correctness uses the
    same annotation semantics as the established tracer evaluation.
    """
    mask = geometry_mask(batch, cfg)
    window = commit_window(cfg, n_commit)
    near = dense_commit_mask(mask.shape[1], cfg, window, mask.device)
    predicted = F.interpolate(output['points'][..., :2].transpose(1, 2),
                              size=mask.shape[1], mode='linear', align_corners=True).transpose(1, 2)
    target = torch.where(mask[..., None], batch['dense_ab'], 0.)
    error = F.smooth_l1_loss(predicted, target, beta=1., reduction='none').mean(-1)
    geometry = window_mean(error, mask, near)
    initial_geometry = geometry
    if cfg.correction:
        initial = F.interpolate(output['initial_points'][..., :2].transpose(1, 2),
                                size=mask.shape[1], mode='linear', align_corners=True).transpose(1, 2)
        initial_error = F.smooth_l1_loss(initial, target, beta=1., reduction='none').mean(-1)
        initial_geometry = window_mean(initial_error, mask, near)
        # Supervise every earlier proposal without increasing the total loss
        # weight as refinement steps are added. The final curve gets 75%.
        auxiliary = initial_geometry
        if 'refinement_points' in output:
            earlier = output['refinement_points'][:, :-1]
            losses = []
            for curve in earlier.unbind(1):
                dense = F.interpolate(curve[..., :2].transpose(1, 2),
                    size=mask.shape[1], mode='linear', align_corners=True).transpose(1, 2)
                errors = F.smooth_l1_loss(dense, target, beta=1., reduction='none').mean(-1)
                losses.append(window_mean(errors, mask, near))
            auxiliary = torch.stack(losses).mean(0)
        geometry = .75*geometry+.25*auxiliary
    labels, known, _ = prefix_labels(output['points'], batch, tolerance, cfg.max_recovery_distance)
    bce = F.binary_cross_entropy_with_logits(output['confidence_logits'], labels, reduction='none')
    return dict(geometry_per_state=geometry, initial_geometry_per_state=initial_geometry,
                confidence_per_state=window_mean(bce, known.bool(), torch.arange(cfg.n_future, device=mask.device)<window),
                geometry_sum=torch.where(mask, error, 0.).sum(), geometry_count=mask.sum(),
                confidence_sum=(bce*known).sum(), confidence_count=known.sum(),
                error_sum=torch.where(mask, (predicted-target).norm(dim=-1), 0.).sum(),
                correct_count=(labels*known).sum())
