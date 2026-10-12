"""A neighbouring trace's edge as a second crop channel (``model.neighbor_channel``).

The edge is the column of an already traced neighbour at the end point a continuation starts on: one point per
traced row, ``row_step`` apart, and which of them were kept. It is drawn as a Gaussian tube (``SIGMA`` voxels) into
a small volume at pyramid level ``LEVEL`` (2 vx voxels) around the edge, scaled so the line reads about 1; samples
outside the volume read 0. The crops sample it at the same points as the CT (``Tracer.step``), unnormalised.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LEVEL = 1  # level-1 voxels (2 vx): the channel is smooth, a coarse volume keeps it small
SIGMA = 1.  # Gaussian blur (level voxels)
MARGIN = 4  # voxels beyond the surface's bounding box


def _kernel(device) -> torch.Tensor:
    r = math.ceil(3*SIGMA)
    x = torch.arange(-r, r+1, device=device, dtype=torch.float32)
    k = torch.exp(-.5*(x/SIGMA)**2)
    return k/k.sum()


def render(edge: torch.Tensor, kept: torch.Tensor, row_step: float):
    """Volume (1, D, H, W) and its level-``LEVEL`` origin (zyx, long) of a Gaussian tube along a neighbour's edge:
    ``edge`` (R, 3) level-0 xyz points of its traced rows, ``row_step`` apart, ``kept`` (R,) bool. None if fewer than
    two consecutive rows are kept."""
    device = edge.device
    if not (kept[1:] & kept[:-1]).any():
        return None
    # Densify to about 1 vx between points, so every level-LEVEL voxel the edge passes through gets one; a point
    # counts when both rows it lies between are kept.
    up = max(1, math.ceil(row_step))
    R = edge.shape[0]
    size = (R-1)*up+1
    dense = F.interpolate(edge.T[None].float(), size=size, mode='linear', align_corners=True)[0].T
    keep = F.interpolate(kept[None, None].float(), size=size, mode='linear', align_corners=True)[0, 0]
    points = dense[keep > 1-1e-4]
    scale = float(1 << LEVEL)
    idx = torch.round((points+.5)/scale-.5).long().flip(-1)  # zyx level voxels
    lo = idx.min(0).values-MARGIN-math.ceil(3*SIGMA)
    shape = (idx.max(0).values+MARGIN+math.ceil(3*SIGMA)-lo+1).tolist()
    volume = torch.zeros(shape, device=device)
    rel = idx-lo
    volume[rel[:, 0], rel[:, 1], rel[:, 2]] = 1.
    k = _kernel(device)
    v = volume[None, None]
    for axis in range(3):  # separable blur
        view = [1, 1, 1, 1, 1]
        view[2+axis] = len(k)
        pad = [0]*6
        pad[2*(2-axis)] = pad[2*(2-axis)+1] = len(k)//2
        v = F.conv3d(F.pad(v, pad), k.view(view))
    return (v[0]/k[len(k)//2]**2).clamp(max=1.), lo  # a line through voxel centres reads about 1


def volumes(edges: list, device):
    """``Tracer.step``'s ``neighbor`` input from per-episode edges (``render`` outputs, or None for none): the
    volumes (an empty one where there is none: every sample reads 0) and their level-``LEVEL`` origins (B, 3)."""
    empty = torch.zeros(1, 2, 2, 2, device=device)  # far away: reads 0 (a single voxel would divide by size - 1 = 0)
    vols = [e[0] if e is not None else empty for e in edges]
    origins = torch.stack([e[1] if e is not None else torch.full((3,), -(1 << 30), device=device, dtype=torch.long)
                           for e in edges])
    return dict(volumes=vols, origins=origins, level=LEVEL)
