"""Per-step supervision from the GT patch window, and the matching rollout metrics.

Confidence answers whether a candidate point in this front column lies on the seed's sheet, i.e. within
``loss.sheet_tolerance`` (about half the sheet spacing) of the GT sheet. It does not grade accuracy, which the
regression terms handle. Kept points of a column are then all on one sheet, as the rows should be.

Position supervision only applies to front points currently on the seed's sheet. A point on another sheet cannot be
told where "its" sheet continues, so it only learns confidence 0, and the spline (where it now has the weight
floor) carries it along with its neighbours, back onto the sheet when they are right. Each on-sheet point's
target is the GT point one ``step`` ahead (in the patch's ``u``) of its current foot point, so a drifted point
learns to correct as well as advance. Errors are split along the target's GT frame; the along-front component
is weighted lightly because re-spacing redistributes front points anyway. The curve term pulls dense samples between
points onto the sheet wherever both ends of the gap are on the seed's sheet, so a segment bowing over to a
neighbouring sheet between two good points is penalised. Nodes outside the patch, invalid nodes and
projections pinned to the window border carry no supervision (a patch edge is not a sheet edge).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.tracers.sheets.geometry import grid_eval, huber, project, sheet_distance, surface_frame


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(x.dtype)
    return (x*mask).sum()/mask.sum().clamp_min(1)


@torch.no_grad()
def landing(points: torch.Tensor, gt, valid, tolerance: float):
    """Foot points of ``points`` on the GT sheet, their distance, and whether they are on the same sheet."""
    uv, S, frame, ok = project(gt, valid, points)
    dist = sheet_distance(points, S, frame).abs()
    return uv, dist, ok & (dist < tolerance), ok


def gap_mask(per_point: torch.Tensor, samples: int, per_gap: int) -> torch.Tensor:
    """(B, N) point flags -> (B, samples) flags where both ends of a sample's gap are flagged."""
    gap = (torch.arange(samples, device=per_point.device)//per_gap).clamp(max=per_point.shape[1]-2)
    return per_point[:, gap] & per_point[:, gap+1]


@torch.no_grad()
def targets(pos: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, step_cells: float, tolerance: float,
            columns: torch.Tensor):
    """GT point one step ahead of each on-sheet point's foot point, at the points' seed grid columns ``columns``
    (N,), its frame, and where it applies."""
    uv, _, same, _ = landing(pos, gt, valid, tolerance)
    target_uv = torch.stack([uv[..., 0]+step_cells, columns.expand_as(uv[..., 1])], -1)
    S, Su, Sv, ok_t = grid_eval(gt, valid, target_uv)
    return S, surface_frame(Su, Sv), same & ok_t


def step_losses(out, front, gt, valid, cfg):
    d, lc = cfg.data, cfg.loss
    tol = lc.sheet_tolerance
    columns = (torch.arange(front.pos.shape[1], device=gt.device)*d.point_spacing+d.gt_margin)/d.gt_resolution
    S, frame, pos_mask = targets(front.pos, gt, valid, d.step/d.gt_resolution, tol, columns)
    err = ((out.pred-S)[..., :, None]*frame).sum(-2)  # components along (t, a, n)
    point = lc.tangent*huber(err[..., 0], lc.huber)+lc.advance*huber(err[..., 1], lc.huber)
    point = point+lc.normal*huber(err[..., 2], lc.huber)
    loss_point = masked_mean(point, pos_mask)

    with torch.no_grad():
        _, Sd, frame_d, ok_d = project(gt, valid, out.dense)
        _, _, ends_same, _ = landing(out.points.detach(), gt, valid, tol)
        curve_mask = ok_d & gap_mask(ends_same, out.dense.shape[1], cfg.trace.spline_samples)
    dist_d = sheet_distance(out.dense, Sd, frame_d)
    loss_spline = masked_mean(huber(dist_d, lc.huber), curve_mask)

    _, dist, same, ok_p = landing(out.pred.detach(), gt, valid, tol)
    bce = F.binary_cross_entropy_with_logits(out.conf_logit, same.float(), reduction='none')
    # Class-balanced: the mean of the on- and off-sheet means over the classes present, without a host sync: an
    # absent class's masked mean is 0 and its flag 0, so this equals averaging only the present ones (0 if neither is).
    on, off = ok_p & same, ok_p & ~same
    present = torch.stack([on.any(), off.any()]).to(bce.dtype)
    parts = torch.stack([masked_mean(bce, on), masked_mean(bce, off)])
    loss_conf = (parts*present).sum()/present.sum().clamp_min(1)

    loss = loss_point+lc.spline*loss_spline+lc.confidence*loss_conf
    # Front length: the (fixed) front points should neither stretch nor shrink along the sheet. Points are
    # re-spaced evenly every step, so the arc length of the curve, per gap, is the spacing that results.
    points = front.pos.shape[1]
    arc = (out.dense[:, 1:]-out.dense[:, :-1]).norm(dim=-1).sum(1)
    excess = (arc-(points-1)*d.point_spacing)/(points-1)  # voxels per gap
    if lc.front_length:
        loss = loss+lc.front_length*huber(excess, lc.huber).mean()
    with torch.no_grad():
        conf = torch.sigmoid(out.conf_logit) > cfg.trace.confidence_threshold
        stats = dict(loss=loss.detach(), point=loss_point.detach(), spline=loss_spline.detach(),
                     conf_bce=loss_conf.detach(), supervised=ok_p.float().mean(),
                     err_normal=masked_mean(err[..., 2].abs().detach(), pos_mask),
                     dist=masked_mean(dist, same), same_sheet=masked_mean(same.float(), ok_p),
                     confident=masked_mean(conf.float(), ok_p),
                     conf_precision=masked_mean(same.float(), ok_p & conf),
                     off_caught=masked_mean((~conf).float(), ok_p & ~same),
                     curve_same_sheet=masked_mean((dist_d.detach().abs() < tol).float(), curve_mask),
                     gap_excess=excess.detach().abs().mean())
    return loss, stats


@torch.no_grad()
def front_metrics(pos, conf, seed_u: float, steps_done: int, gt, valid, cfg):
    """Rollout metrics of a committed row; ``seed_u`` is the seed row's grid ``u`` index, ``steps_done`` >= 1."""
    d, lc = cfg.data, cfg.loss
    uv, dist, same, ok = landing(pos, gt, valid, lc.sheet_tolerance)
    advance = (uv[..., 0]-(seed_u+steps_done*d.step/d.gt_resolution))*d.gt_resolution
    return dict(dist=dist, advance=advance, same=same, ok=ok, confident=conf > cfg.trace.confidence_threshold)
