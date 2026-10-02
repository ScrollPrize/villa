"""Observed-path geometry beyond the crop, in current-frame coordinates.

Local history references are crop-supported, so only the newest ~24 voxels of the
committed path reach the decoder and scorer directly. These tokens sample the same
observed polyline (never annotation geometry) at fixed arclengths behind the head,
plus its first point, and express each sample in the current crop frame without
crop masking. Positions use multi-scale Fourier features so voxel-scale lateral
offsets of distant samples remain distinguishable.
"""
import math

import numpy as np
import torch
from torch import nn

from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at
from vesuvius.neural_tracing.fiber_follow.shared.reference import observed_path

# Arclength behind the head, then the observed path's first point (the seed).
OFFSETS = (1., 2., 3., 4., 6., 8., 12., 16., 20., 24., 28., 32., 40., 48., 56., 64.,
           80., 96., 112., 128., 160., 192., 224., 256., 320., 384., 448., 512.)
COUNT = len(OFFSETS)+1
# Local position (3), local unit tangent (3), arclength behind the head, first-point role.
FEATURES = 8
SCALES = tuple(2.**k for k in range(1, 11))  # 2..1024 tracing voxels
MAX_BACK = 1024.


def path_geometry_samples(item):
    """Samples of the committed polyline; unavailable arclengths are masked."""
    path = observed_path(item)
    arc = arclength(path)
    total = float(arc[-1])
    back = np.r_[OFFSETS, total]
    at = total-back
    valid = at >= -1e-8
    valid[-1] = total >= 1.  # A seed-only state has no earlier observation.
    at = np.clip(at, 0., total)
    points = interp_at(path, arc, at)
    tangent = interp_at(path, arc, np.clip(at+.5, 0., total))-interp_at(path, arc, np.clip(at-.5, 0., total))
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    valid &= norm[:, 0] > 1e-8
    frame, pos = np.asarray(item['frame']), np.asarray(item['pos'])
    features = np.zeros((COUNT, FEATURES), np.float32)
    features[:, :3] = (points-pos) @ frame
    features[:, 3:6] = (tangent/np.maximum(norm, 1e-8)) @ frame
    features[:, 6] = np.minimum(back, MAX_BACK)
    features[-1, 7] = 1.
    features[~valid] = 0.
    return features, valid


def path_geometry_inputs(items):
    samples = [path_geometry_samples(item) for item in items]
    return dict(path_geometry=torch.from_numpy(np.stack([s[0] for s in samples])),
                path_geometry_valid=torch.from_numpy(np.stack([s[1] for s in samples])))


class PathGeometryTokens(nn.Module):
    """Memory tokens for decoder and scorer cross-attention; zero at initialization."""
    def __init__(self, cfg):
        super().__init__()
        self.register_buffer('frequencies', 2*math.pi/torch.tensor(SCALES), persistent=False)
        width = 3*2*len(SCALES)+FEATURES
        self.embed = nn.Sequential(nn.Linear(width, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, cfg.hidden))
        nn.init.zeros_(self.embed[-1].weight)
        nn.init.zeros_(self.embed[-1].bias)

    def forward(self, geometry, valid):
        geometry = torch.where(valid[..., None], geometry.float(), 0.)
        pos, tangent = geometry[..., :3], geometry[..., 3:6]
        back, seed = geometry[..., 6:7], geometry[..., 7:8]
        phase = pos[..., None]*self.frequencies
        fourier = torch.cat((phase.sin(), phase.cos()), -1).flatten(-2)
        features = torch.cat((fourier, pos/64., tangent, torch.log1p(back)/math.log1p(MAX_BACK), seed), -1)
        return torch.where(valid[..., None], self.embed(features), 0.)
