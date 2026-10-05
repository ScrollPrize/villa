"""Whole-crop path and Gaussian-tube heads (``path_planes='crop'``, ``tube_head``).

The path head predicts the original fiber's lateral position at every crop plane, behind and ahead of the head
(``shared.geometry.crop_path_planes``); planes 1..n_future are the proposal the commit, scorer and metrics use.
The tube head predicts, at every crop sample, exp(-d^2 / 2 sigma^2) with d the distance to the original fiber;
neighbouring fibers are background, and samples near an untagged annotation end inside the crop are unknown.
Both targets follow the original fiber wherever it is in the crop, however far the head is from it.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

TUBE_SUBSTEPS = 4  # curve densification between 1-voxel samples (max splat error 1/8 voxel)
OPEN_END_RADIUS = 8.  # voxels around an untagged annotation end that the tube loss ignores
TUBE_LEVEL = .05  # target value above which a sample counts as inside the tube (second loss average)


class TubeHead(nn.Module):
    """Per-token linear readout of a 4x4x4 block of crop samples (the token stride), pixel-shuffled to full res."""
    def __init__(self, cfg):
        super().__init__()
        self.stride = cfg.token_stride
        self.readout = nn.Linear(cfg.hidden, math.prod(self.stride))

    def forward(self, deep):
        b, _, d, h, w = deep.shape
        sd, sh, sw = self.stride
        values = self.readout(deep.permute(0, 2, 3, 4, 1)).reshape(b, d, h, w, sd, sh, sw)
        return values.permute(0, 1, 4, 2, 5, 3, 6).reshape(b, d*sd, h*sh, w*sw)


def crop_index(points, crop):
    """Crop-local xyz (trace voxels) -> fractional (depth, height, width) sample indices."""
    half = (crop.width-1)/2
    return torch.stack((points[..., 2]/crop.spacing+crop.behind, points[..., 1]/crop.spacing+half,
                        points[..., 0]/crop.spacing+half), -1)


@torch.no_grad()
def tube_targets(curve, curve_mask, open_ends, crop, sigma):
    """Gaussian tube target (B, D, H, W) and known mask from the original fiber's curve (B, N, 3).

    The curve is densified and splatted with an amax scatter over a (2R+1)^3 neighbourhood (R = 3 sigma), which
    equals the tube of the polyline's distance up to the densification step."""
    b, n = curve_mask.shape
    device = curve.device
    shape = (crop.depth, crop.width, crop.width)
    a, c = curve[:, :-1], curve[:, 1:]
    pair = curve_mask[:, :-1] & curve_mask[:, 1:]
    steps = torch.arange(TUBE_SUBSTEPS, device=device, dtype=curve.dtype)/TUBE_SUBSTEPS
    dense = (a[:, :, None]+steps[None, None, :, None]*(c-a)[:, :, None]).reshape(b, -1, 3)
    dense_ok = pair[:, :, None].expand(-1, -1, TUBE_SUBSTEPS).reshape(b, -1)
    index = crop_index(dense.float(), crop)
    radius = int(math.ceil(3*sigma/crop.spacing))
    offsets = torch.stack(torch.meshgrid(*(torch.arange(-radius, radius+1, device=device),)*3, indexing='ij'), -1).reshape(-1, 3)
    target = torch.zeros(b, math.prod(shape), device=device)
    voxel = index.round()[:, :, None]+offsets[None, None]  # (B, M, K, 3)
    distance = ((voxel-index[:, :, None])*crop.spacing).norm(dim=-1)
    value = torch.exp(-.5*(distance/sigma)**2)
    limit = torch.tensor(shape, device=device)
    inside = dense_ok[:, :, None] & (voxel >= 0).all(-1) & (voxel < limit).all(-1)
    flat = ((voxel[..., 0]*shape[1]+voxel[..., 1])*shape[2]+voxel[..., 2]).long()
    value = torch.where(inside, value, 0.)
    flat = torch.where(inside, flat, 0)
    target.scatter_reduce_(1, flat.reshape(b, -1), value.reshape(b, -1), reduce='amax', include_self=True)
    known = torch.ones(b, math.prod(shape), dtype=torch.bool, device=device)
    # An untagged annotation end inside the crop: the fiber may continue unannotated there.
    count = curve_mask.sum(-1)
    first = curve_mask.float().argmax(-1)
    last = (count-1+first).clamp_min(0)
    ends = torch.stack((curve.gather(1, first[:, None, None].expand(-1, 1, 3))[:, 0],
                        curve.gather(1, last[:, None, None].expand(-1, 1, 3))[:, 0]), 1)  # (B, 2, 3)
    grid = crop_index_grid(crop, device)
    for k in range(2):
        open_end = open_ends[:, k] & (count > 0)
        if open_end.any():
            near = (grid[None]-crop_index(ends[:, k].float(), crop)[:, None]).mul(crop.spacing).norm(dim=-1) <= OPEN_END_RADIUS
            known &= ~(near & open_end[:, None])
    return target.reshape(b, *shape), known.reshape(b, *shape) & (count > 0)[:, None, None, None]


def crop_index_grid(crop, device):
    d, h, w = torch.meshgrid(torch.arange(crop.depth, device=device), torch.arange(crop.width, device=device),
                             torch.arange(crop.width, device=device), indexing='ij')
    return torch.stack((d, h, w), -1).reshape(-1, 3).float()


def tube_loss(logits, batch, crop, sigma):
    """Per-state soft-target BCE: mean over known samples plus mean over known samples inside the tube."""
    target, known = tube_targets(batch['crop_curve'], batch['crop_curve_mask'].bool(),
                                 batch['crop_curve_open_ends'].bool(), crop, sigma)
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none')
    tube = known & (target > TUBE_LEVEL)
    dims = (1, 2, 3)
    everywhere = torch.where(known, bce, 0.).sum(dims)/known.sum(dims).clamp_min(1)
    inside = torch.where(tube, bce, 0.).sum(dims)/tube.sum(dims).clamp_min(1)
    return everywhere+inside, dict(tube_known_samples=known.sum(), tube_samples=tube.sum())


def crop_path_loss(curve, batch):
    """Per-state smooth-L1 between a whole-crop path (B, P, 3) and the original fiber's crossings, mean over known planes."""
    mask = batch['crop_mask'].bool()
    target = torch.where(mask[..., None], batch['crop_ab'], 0.)
    error = F.smooth_l1_loss(curve[..., :2].float(), target.float(), beta=1., reduction='none').mean(-1)
    return torch.where(mask, error, 0.).sum(-1)/mask.sum(-1).clamp_min(1)
