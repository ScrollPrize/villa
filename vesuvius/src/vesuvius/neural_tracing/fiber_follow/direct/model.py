"""Direct curve regression and bounded local correction from CT and observed history.

All coordinates are trace-grid voxels, regardless of image sampling resolution.
There is no noise process, ODE, candidate ranking, or inherited flow backbone.
"""
from dataclasses import asdict, dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec

ARCHITECTURE = 'direct_curve_v1'


@dataclass
class DirectConfig:
    fine: CropSpec = field(default_factory=lambda: CropSpec(depth=80, width=48, behind=32, spacing=.5))
    coarse: CropSpec = field(default_factory=lambda: CropSpec(depth=88, width=32, behind=64, spacing=2.))
    channels: int = 16
    hidden: int = 128
    heads: int = 4
    layers: int = 2
    n_future: int = 16
    future_step: float = 1.
    n_history: int = 128
    history_stride: int = 4
    max_recovery_distance: float = 6.
    patch_radius: float = 1.
    correction: bool = True
    correction_limit: float = 1.  # maximum lateral Euclidean adjustment, trace voxels

    def __post_init__(self):
        for name in ('fine', 'coarse'):
            if isinstance(getattr(self, name), dict):
                setattr(self, name, CropSpec(**getattr(self, name)))
            crop = getattr(self, name)
            if min(crop.depth, crop.width) < 8 or not 0 <= crop.behind < crop.depth:
                raise ValueError('Image crops must be at least eight samples wide/deep with an interior origin')
            if not math.isfinite(crop.spacing) or crop.spacing <= 0:
                raise ValueError('Crop spacing must be positive')
        if min(self.channels, self.hidden, self.heads, self.layers, self.n_future,
               self.n_history, self.history_stride) < 1 or self.hidden % self.heads:
            raise ValueError('Positive dimensions required; hidden must divide by heads')
        if not 0 < self.future_step <= self.max_recovery_distance or not math.isfinite(self.max_recovery_distance):
            raise ValueError('Invalid forward spacing or connection limit')
        if not 0 < self.patch_radius < (self.fine.width-1)*self.fine.spacing/2:
            raise ValueError('Local observation patch must fit fine crop')
        if not math.isfinite(self.correction_limit) or self.correction_limit <= 0:
            raise ValueError('Correction limit must be finite and positive')
        if self.n_future*self.future_step > (self.fine.depth-self.fine.behind-1)*self.fine.spacing:
            raise ValueError('Future horizon exceeds fine image')

    @property
    def recent_history_points(self):
        return self.n_history

    @property
    def lateral_limit(self):
        return (self.fine.width-1)*self.fine.spacing/2 - self.patch_radius

    def to_dict(self):
        return asdict(self)


def feature_grid(points, crop, shape, stride=1):
    """Map physical coordinates to a strided-convolution lattice, exactly.

    Padding-one, kernel-three convolutions put element i at input i*stride;
    feature endpoints need not coincide with crop endpoints for even sizes.
    """
    size = points.new_tensor(tuple(reversed(shape)))
    origin = points.new_tensor((-(crop.width-1)*crop.spacing/2,
                                -(crop.width-1)*crop.spacing/2, -crop.behind*crop.spacing))
    return 2*(points-origin)/(crop.spacing*stride*(size-1))-1


def sample_features(features, points, crop):
    grid = feature_grid(points.float(), crop, features.shape[-3:])
    supported = (grid.abs() <= 1).all(-1) & torch.isfinite(grid).all(-1)
    grid = torch.where(supported[..., None], grid, 0.)
    values = F.grid_sample(features.float(), grid[:, :, None, None], align_corners=True)
    values = values[:, :, :, 0, 0].transpose(1, 2)
    return torch.where(supported[..., None], values, 0.), supported


class ImageEncoder(nn.Module):
    """Small image pyramid; no full-resolution decoder or history modulation."""
    def __init__(self, channels):
        super().__init__()
        def stage(a, b, stride):
            return nn.Sequential(nn.Conv3d(a, b, 3, stride=stride, padding=1, bias=False),
                                 nn.GroupNorm(math.gcd(8, b), b), nn.SiLU(),
                                 nn.Conv3d(b, b, 3, padding=1, bias=False),
                                 nn.GroupNorm(math.gcd(8, b), b), nn.SiLU())
        self.local = stage(2, channels, 1)
        self.down = nn.Sequential(stage(channels, 2*channels, 2), stage(2*channels, 4*channels, 2))

    def forward(self, x):
        local = self.local(x)
        return local, self.down(local)


class DirectFollower(nn.Module):
    def __init__(self, cfg: DirectConfig):
        super().__init__()
        self.cfg = cfg
        c, h = cfg.channels, cfg.hidden
        self.fine_encoder = ImageEncoder(c)
        self.coarse_encoder = ImageEncoder(c)
        self.image_token = nn.Linear(4*c+4, h)  # features, physical xyz, scale flag
        self.history_token = nn.Sequential(nn.Linear(2*c+6, h), nn.SiLU(), nn.Linear(h, h))
        self.query = nn.Sequential(nn.Linear(9*c+1, h), nn.SiLU(), nn.Linear(h, h))
        layer = nn.TransformerDecoderLayer(h, cfg.heads, 2*h, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.layers, norm=nn.LayerNorm(h))
        self.coordinates = nn.Linear(h, 2)
        nn.init.normal_(self.coordinates.weight, std=.005)
        nn.init.zeros_(self.coordinates.bias)
        if cfg.correction:
            self.correction_head = nn.Sequential(nn.Linear(h+9*c+3, h), nn.SiLU(), nn.Linear(h, 2))
            nn.init.normal_(self.correction_head[-1].weight, std=.001)
            nn.init.zeros_(self.correction_head[-1].bias)
        self.path_evidence = nn.Sequential(nn.Linear(h+9*c+3, h), nn.SiLU())
        self.confidence_head = nn.Sequential(nn.Linear(2*h, h), nn.SiLU(), nn.Linear(h, 1))
        stencil = torch.tensor([[a, b, 0.] for a in (-1., 0., 1.) for b in (-1., 0., 1.)])
        self.register_buffer('stencil', stencil*cfg.patch_radius, persistent=False)
        self.register_buffer('planes', torch.arange(1, cfg.n_future+1).float()*cfg.future_step, persistent=False)

    def image_tokens(self, deep, crop, scale):
        # Pool coordinates with exactly the same cells as features.
        d, y, x = deep.shape[-3:]
        zyx = torch.meshgrid(*(torch.arange(n, device=deep.device).float() for n in (d, y, x)), indexing='ij')
        xyz = torch.stack((zyx[2], zyx[1], zyx[0]), 0)*4*crop.spacing
        xyz -= xyz.new_tensor(((crop.width-1)*crop.spacing/2,
                              (crop.width-1)*crop.spacing/2, crop.behind*crop.spacing))[:, None, None, None]
        xyz = xyz[None].expand(len(deep), -1, -1, -1, -1)/128
        pooled = F.avg_pool3d(torch.cat((deep.float(), xyz), 1), 2)
        tokens = pooled.flatten(2).transpose(1, 2)
        return self.image_token(torch.cat((tokens, torch.full_like(tokens[..., :1], scale)), -1))

    def patches(self, features, points):
        b, k, _ = points.shape
        values, _ = sample_features(features, (points[:, :, None]+self.stencil).reshape(b, k*9, 3), self.cfg.fine)
        return values.reshape(b, k, -1)

    def forward(self, x, hist, hmask):
        cfg = self.cfg
        fine, fine_deep = self.fine_encoder(x['fine'])
        coarse, coarse_deep = self.coarse_encoder(x['coarse'])
        valid = hmask.bool()
        observed = torch.where(valid[..., None], hist.float(), 0.)
        # Keep eight most recent observations, then one every four voxels.
        indices = torch.arange(cfg.n_history, device=hist.device)
        indices = indices[(indices < 8) | (indices % cfg.history_stride == 0)]
        observed, valid = observed[:, indices], valid[:, indices]
        flocal, fsupport = sample_features(fine, observed, cfg.fine)
        clocal, csupport = sample_features(coarse, observed, cfg.coarse)
        age = (indices.float()+1)/cfg.n_history
        ht = self.history_token(torch.cat((flocal, clocal, observed/128,
            age[None, :, None].expand(len(hist), -1, -1), fsupport[..., None].float(),
            csupport[..., None].float()), -1))
        image = torch.cat((self.image_tokens(fine_deep, cfg.fine, 0.),
                           self.image_tokens(coarse_deep, cfg.coarse, 1.)), 1)
        memory = torch.cat((image, ht), 1)
        padding = torch.cat((torch.zeros(image.shape[:2], dtype=torch.bool, device=hist.device), ~valid), 1)
        initial = hist.new_zeros(len(hist), cfg.n_future, 3)
        initial[..., 2] = self.planes
        query = self.query(torch.cat((self.patches(fine, initial),
                                      initial[..., 2:]/(cfg.n_future*cfg.future_step)), -1))
        decoded = self.decoder(query, memory, memory_key_padding_mask=padding)
        lateral = cfg.lateral_limit*torch.tanh(self.coordinates(decoded).float())
        points = torch.cat((lateral, initial[..., 2:]), -1)
        initial_points = points
        if cfg.correction:
            # Decoder features retain history/identity conditioning. The small
            # residual reads fine evidence at the proposal, without re-encoding.
            correction = self.correction_head(torch.cat((decoded, self.patches(fine, points), points/16), -1))
            correction = correction.float().tanh()*(cfg.correction_limit/math.sqrt(2))
            lateral = (lateral+correction).clamp(-cfg.lateral_limit, cfg.lateral_limit)
            points = torch.cat((lateral, initial[..., 2:]), -1)
        # Confidence inspects the actual prediction. Its labels and coordinate
        # sampling are detached; coordinate regression retains its own gradient.
        evidence = self.path_evidence(torch.cat((decoded, self.patches(fine, points.detach()),
                                                 points.detach()/16), -1))
        count = torch.arange(1, cfg.n_future+1, device=hist.device)[None, :, None]
        prefix_mean = evidence.cumsum(1)/count
        prefix_max = evidence.cummax(1).values
        logits = self.confidence_head(torch.cat((prefix_mean, prefix_max), -1)).squeeze(-1).float()
        return dict(points=points, initial_points=initial_points,
                    confidence_logits=logits, confidence=logits.sigmoid().cummin(-1).values)
