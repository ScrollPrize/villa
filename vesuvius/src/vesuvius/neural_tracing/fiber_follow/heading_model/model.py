"""Crop-heading network and inference: CT patch + observed path -> crop heading that keeps the fiber in crop.

Inputs are expressed in a frame whose +z is the tracer's prior heading (any roll; training uses random roll, so
no CT-derived roll or structure tensor is needed):
- a CT patch around the head extending forward, sampled with the tracer's crop sampler and CT normalization;
- the last ``window`` voxels of the observed path at unit arclength relative to the head, with validity (none at
  a seed).
The output is a residual on the prior (+z), so an untrained network returns the prior.
"""
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch
from torch import nn

from vesuvius.neural_tracing.fiber_follow.models.model import CoordinateRegressionConfig
from vesuvius.neural_tracing.fiber_follow.data.crop_sampling import scalar_crops
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength, frame_from_heading, interp_at

ARCHITECTURE = 'crop_heading_ct_path_v1'


def main_crop_forward(crop=None):
    """Forward extent of the follower's crop (trace voxels): the span whose fiber the heading should keep in crop."""
    crop = crop or CoordinateRegressionConfig().fine
    return (crop.depth-1-crop.behind)*crop.spacing


@dataclass
class HeadingConfig:
    width: int = 8
    patch: CropSpec = field(default_factory=lambda: CropSpec(depth=32, width=32, behind=4, spacing=1.25))
    window: int = 32  # observed-path voxels behind the head
    forward: float = field(default_factory=main_crop_forward)  # target span ahead of the head
    # CT pyramid levels above the follower's CT level for the patch (each a 2x block mean); 0 reads the follower's CT.
    ct_downsample_levels: int = 0

    def __post_init__(self):
        if isinstance(self.patch, dict):
            self.patch = CropSpec(**self.patch)
        if self.width < 1 or self.window < 1 or not self.forward > 0:
            raise ValueError('Heading model width, window and forward span must be positive')
        if not (isinstance(self.ct_downsample_levels, int) and self.ct_downsample_levels >= 0):
            raise ValueError('ct_downsample_levels must be a nonnegative integer')

    def to_dict(self):
        return asdict(self)


class HeadingNet(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, cfg: HeadingConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.features = nn.Sequential(
            nn.Conv3d(1, w, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(w, 2*w, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(2*w, 4*w, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(4*w, 4*w, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool3d(2), nn.Flatten())
        self.path = nn.Sequential(nn.Linear(4*cfg.window, 64), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(4*w*8+64, 64), nn.SiLU(), nn.Linear(64, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)  # starts as the prior heading

    def forward(self, patch, path):
        hidden = torch.cat((self.features(patch), self.path(path)), -1)
        return nn.functional.normalize(self.head(hidden)+patch.new_tensor([0., 0., 1.]), dim=-1)


def prior_frames(priors, rng=None):
    """+z along each prior heading; deterministic roll at inference, random roll when ``rng`` is given."""
    frames = []
    for h in priors:
        frame = frame_from_heading(np.asarray(h, np.float64))
        if rng is not None:
            a = rng.uniform(0, 2*np.pi)
            c, s = np.cos(a), np.sin(a)
            frame = frame @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        frames.append(frame)
    return frames


def path_features(path, pos, frame, window):
    """Last ``window`` voxels behind the head (points and validity), newest first, in ``frame``."""
    out = np.zeros((window, 4), np.float32)
    path = np.asarray(path, np.float64)
    if len(path) > 1:
        arc = arclength(path)
        back = arc[-1]-np.arange(1, window+1.)
        valid = back >= 0
        if valid.any():
            out[valid, :3] = (interp_at(path, arc, back[valid])-np.asarray(pos)) @ frame
            out[valid, 3] = 1.
    return out.ravel()


def patch_volume_spec(spec, levels):
    """The follower's volume spec at ``levels`` coarser CT pyramid levels, for the heading patch.

    Each pyramid level is a 2x block mean, so its voxels are 2x larger in trace units. Per-crop z-score is the
    only normalization that carries over; a bound record is rebound to the coarser array.
    """
    if not levels:
        return spec
    from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import ZSCORE_EPSILON, ZSCORE_METHOD, volume_key
    record = spec.ct_normalization
    if record is not None and record.get('method') != ZSCORE_METHOD:
        raise ValueError('Downsampled heading patches need per-crop z-score CT normalization')
    coarse = replace(spec, ct_level=spec.ct_level+levels, ct_grid_scale=spec.ct_grid_scale*2**levels,
                     ct_normalization=None)
    if record is not None:
        coarse.ct_normalization = dict(method=ZSCORE_METHOD, volume=volume_key(coarse), epsilon=ZSCORE_EPSILON)
    return coarse


def ct_shift(cfg, vol):
    """Trace-voxel offset of the patch volume's grid from the follower's CT grid.

    A coarser voxel j averages follower voxels [f*j, f*j+f) (f = 2**levels), so its center lies (f-1)/2 follower
    voxels past f*j. Sampling at position - shift puts the patch exactly where the follower's CT would.
    """
    f = 2**cfg.ct_downsample_levels
    return (f-1)/(2*f*vol.input_scale)


def model_inputs(vol, cfg, positions, frames, paths, pool=None):
    """CT patches (tracer sampler and normalization) and path features for each head.

    ``vol`` is the patch volume: the follower's, or its ``patch_volume_spec`` level for a downsampled patch.
    """
    shift = ct_shift(cfg, vol)
    items = [dict(pos=np.asarray(p, np.float64)-shift, frame=np.asarray(f, np.float64)) for p, f in zip(positions, frames)]
    patch = scalar_crops(items, vol, cfg.patch, pool)
    path = torch.from_numpy(np.stack([path_features(q, p, f, cfg.window) for q, p, f in zip(paths, positions, frames)]))
    return patch, path


def save_heading_model(path, model, **extra):
    torch.save(dict(architecture=ARCHITECTURE, config=model.cfg.to_dict(), state=model.state_dict(), **extra), path)


def load_heading_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture') != ARCHITECTURE:
        raise ValueError(f'Not a {ARCHITECTURE} checkpoint: {path}')
    model = HeadingNet(HeadingConfig(**checkpoint['config']))
    model.load_state_dict(checkpoint['state'])
    return model.to(device).eval(), checkpoint


class HeadingPredictor:
    """Crop headings for tracer heads from (position, prior heading, observed path)."""
    def __init__(self, model, device='cpu'):
        self.model, self.device = model.to(device).eval(), torch.device(device)
        self._volumes = {}

    @classmethod
    def load(cls, path, device='cpu'):
        return cls(load_heading_model(path, device)[0], device)

    def patch_volume(self, vol):
        """The tracer's volume, or its coarser CT pyramid level for a downsampled-patch model (opened once)."""
        levels = self.model.cfg.ct_downsample_levels
        if not levels:
            return vol
        key = (vol.spec.ct_zarr, vol.spec.ct_level, vol.spec.cache_dir)
        if key not in self._volumes:
            from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume
            self._volumes[key] = FiberVolume(patch_volume_spec(vol.spec, levels), cache_bytes=256 << 20)
        return self._volumes[key]

    @torch.no_grad()
    def predict(self, vol, positions, priors, paths, pool=None):
        """World headings, each signed along its prior. ``vol`` is the tracer's (follower CT) volume."""
        if not len(positions):
            return []
        frames = prior_frames(priors)
        patch, path = model_inputs(self.patch_volume(vol), self.model.cfg, positions, frames, paths, pool)
        local = self.model(patch.to(self.device), path.to(self.device)).double().cpu().numpy()
        headings = [f @ d for f, d in zip(frames, local)]
        return [h if h @ np.asarray(p) >= 0 else -h for h, p in zip(headings, priors)]
