"""Shared spatial encoder."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec, crop_local_grid

def prepare_model(model, device):
    """Apply the model's measured CUDA layout without changing precision."""
    model = model.to(device)
    if torch.device(device).type == 'cuda':
        layout = getattr(model, 'cuda_memory_format', torch.channels_last_3d)
        nn.utils.convert_conv3d_weight_memory_format(model, layout)
    return model


def block(cin, cout, norm, stride=1):
    def normalization():
        if norm == 'batch':
            return nn.BatchNorm3d(cout)
        if norm == 'group':
            return nn.GroupNorm(math.gcd(8, cout), cout)
        raise ValueError('norm must be batch or group')
    return nn.Sequential(nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         normalization(), nn.SiLU(), nn.Conv3d(cout, cout, 3, padding=1, bias=False),
                         normalization(), nn.SiLU())


class SpatialEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.crop = CropSpec(depth=cfg.depth, width=cfg.width, behind=cfg.behind,
                             spacing=cfg.spacing)
        grid = torch.from_numpy(crop_local_grid(self.crop)).float()
        self.register_buffer('coordinates', grid.permute(3, 0, 1, 2)[None] / 32, persistent=False)
        w = cfg.widths
        self.encoders = nn.ModuleList([block(cfg.in_channels+3, w[0], cfg.norm)] +
                                     [block(a, b, cfg.norm, 2) for a, b in zip(w[:-1], w[1:])])
        self.decoders = nn.ModuleList([block(w[i+1]+w[i], w[i], cfg.norm) for i in range(len(w)-2, -1, -1)])
        self.history = nn.Sequential(nn.Linear(cfg.hist_points*4, cfg.hidden), nn.SiLU())
        self.condition = nn.ModuleList([nn.Linear(cfg.hidden, 2*c) for c in w])

    def sampling_grid(self, local):
        cfg = self.cfg
        half = (cfg.width-1)*cfg.spacing/2
        return torch.stack([local[..., 0]/half, local[..., 1]/half,
                            2*(local[..., 2]/cfg.spacing+cfg.behind)/(cfg.depth-1)-1], -1)

    def encode(self, x, hist, hmask, *, return_deep=False):
        """Spatial features and the geometric-history context vector."""
        cfg = self.cfg
        if self.encoders[0][0].weight.is_contiguous(memory_format=torch.channels_last_3d):
            x = x.contiguous(memory_format=torch.channels_last_3d)
        h = hist[:, cfg.hist_stride-1::cfg.hist_stride][:, :cfg.hist_points]
        m = hmask[:, cfg.hist_stride-1::cfg.hist_stride][:, :cfg.hist_points]
        if h.shape[1] != cfg.hist_points:
            raise ValueError('History must cover hist_points * hist_stride')
        h = torch.where(m[..., None] > 0, h, 0.)
        context = self.history(torch.cat([h/32, m[..., None]], -1).flatten(1).to(x.dtype))
        features = torch.cat([x, self.coordinates.expand(len(x), -1, -1, -1, -1).to(x.dtype)], 1)
        skips = []
        for encoder, condition in zip(self.encoders, self.condition):
            features = encoder(features)
            gain, bias = condition(context).chunk(2, -1)
            features = features * (1 + .1*gain[..., None, None, None]) + bias[..., None, None, None]
            skips.append(features)
        deep = features
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            features = decoder(torch.cat([F.interpolate(features, size=skip.shape[-3:], mode='trilinear', align_corners=True), skip], 1))
        return (features, context, deep) if return_deep else (features, context)

