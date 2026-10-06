"""The supervised-path contract: which target planes carry geometry, and contacts with validated neighboring paths."""
import math

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.tracers.models.model import feature_grid


CONNECTOR_SPACING = .25  # same dense spatial resolution as the supervised path


def foreign_hits(points, batch, cfg):
    """Which (B, N, 3) crop-local points fall in cells of a validated neighboring path."""
    foreign = batch['foreign']
    grid = feature_grid(points.detach().float(), cfg.fine, foreign.shape[-3:])
    values = F.grid_sample(foreign[:, None].float(), grid[:, :, None, None], mode='nearest', align_corners=True)
    return values[:, 0, :, 0, 0] > .5


def foreign_failures(points, batch, cfg, count):
    """Dense predictions in cells occupied by a validated neighboring path."""
    dense = F.interpolate(points.detach().float().transpose(1, 2), size=count, mode='linear',
                          align_corners=True).transpose(1, 2)
    return foreign_hits(dense, batch, cfg)


def connector_failures(first, batch, cfg):
    """Foreign contact on the origin-to-first-point segment, excluding the origin itself.

    ``first`` is (B, 3). A safe endpoint does not make a connector safe if it crosses a
    known foreign fiber. Samples are ``CONNECTOR_SPACING`` apart along the longest
    allowed connection, so every committed connection is checked as densely as the path.
    """
    count = math.ceil(cfg.max_recovery_distance/CONNECTOR_SPACING)
    fraction = torch.arange(1, count+1, device=first.device, dtype=torch.float32)/count
    samples = first.detach().float()[:, None]*fraction[None, :, None]
    return foreign_hits(samples, batch, cfg).any(-1)


def geometry_mask(batch, cfg):
    """Certified original-fiber continuation inside the observable crop.

    Only states whose annotated first connection satisfies the commit limit carry
    geometry (``geometry_valid``); with a neighbor raster, a target connection that
    crosses a known foreign fiber is censored as well.
    """
    annotated = batch['dense_mask'].bool() & batch['geometry_valid'][:, None].bool()
    target = torch.where(annotated[..., None], batch['dense_ab'], 0.)
    observable = (target.abs().amax(-1) <= cfg.lateral_limit).int().cummin(-1).values.bool()
    mask = annotated & observable
    if 'foreign' in batch:
        mask = mask & ~target_connector_foreign(batch, cfg)[:, None]
    return mask


def target_connector_foreign(batch, cfg):
    first = torch.cat((batch['dense_ab'][:, 0], batch['dense_ab'].new_full((len(batch['dense_ab']), 1), cfg.future_step)), -1)
    return connector_failures(first, batch, cfg) & batch['geometry_valid'].bool()
