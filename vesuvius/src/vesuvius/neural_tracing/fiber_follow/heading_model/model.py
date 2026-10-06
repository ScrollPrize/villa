"""Crop-heading network and inference: CT patch + observed path -> crop heading that keeps the fiber in crop.

Inputs are expressed in a frame whose +z is the tracer's prior heading (any roll; training uses random roll, so
no CT-derived roll or structure tensor is needed):
- a CT patch around the head extending forward, sampled with the tracer's crop sampler and CT normalization;
- the last ``window`` voxels of the observed path at unit arclength relative to the head, with validity (none at
  a seed).
The output is a residual on the prior (+z), so an untrained network returns the prior.

The encoder halves the patch four times with stride-2 convolutions whose kernels are 4 wide (padding 1): output o
covers inputs 2o-1..2o+2, so every grid stays centered on the patch, each input feeds exactly two outputs per axis,
and the final 2x2x2 cells are mirror-image octants. v1 used 3-wide kernels, which centered outputs on even inputs:
on each axis the low final cell sat on the patch edge and the high one saw nearly the whole patch.
(scripts/convert_heading_v1.py converts a v1 checkpoint exactly: a 3-wide kernel is a 4-wide one with a zero tap.)
"""
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch
from torch import nn

from vesuvius.neural_tracing.fiber_follow.data.crop_sampling import scalar_crops
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength, frame_from_heading, interp_at

ARCHITECTURE = 'crop_heading_ct_path_v2'
NORMAL_ARCHITECTURE = 'crop_heading_normal_ct_path_v3'
FRAME_ARCHITECTURE = 'crop_frame_ct_path_family_v4'
# The follower crop the heading model was designed for (the earlier coordinate model's default).
FOLLOWER_CROP = CropSpec(depth=120, width=104, behind=48, spacing=.5)
DOWNSAMPLINGS = 4  # stride-2 encoder layers; patch sizes must be divisible by 2**DOWNSAMPLINGS


def main_crop_forward(crop=None):
    """Forward extent of the follower's crop (trace voxels): the span whose fiber the heading should keep in crop."""
    crop = crop or FOLLOWER_CROP
    return (crop.depth-1-crop.behind)*crop.spacing


@dataclass
class HeadingConfig:
    width: int = 8
    patch: CropSpec = field(default_factory=lambda: CropSpec(depth=32, width=32, behind=4, spacing=1.25))
    window: int = 32  # observed-path voxels behind the head
    forward: float = field(default_factory=main_crop_forward)  # target span ahead of the head
    # CT pyramid levels above the follower's CT level for the patch (each a 2x block mean); 0 reads the follower's CT.
    ct_downsample_levels: int = 0
    predict_normals: bool = False
    predict_frames: bool = False

    def __post_init__(self):
        if isinstance(self.patch, dict):
            self.patch = CropSpec(**self.patch)
        if self.width < 1 or self.window < 1 or not self.forward > 0:
            raise ValueError('Heading model width, window and forward span must be positive')
        if not (isinstance(self.ct_downsample_levels, int) and self.ct_downsample_levels >= 0):
            raise ValueError('ct_downsample_levels must be a nonnegative integer')
        if self.predict_frames and not self.predict_normals:
            raise ValueError('Frame prediction requires predict_normals')

    def to_dict(self):
        return asdict(self)


class HeadingNet(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, cfg: HeadingConfig):
        super().__init__()
        self.cfg = cfg
        self.architecture = FRAME_ARCHITECTURE if cfg.predict_frames else (NORMAL_ARCHITECTURE if cfg.predict_normals else ARCHITECTURE)
        if cfg.patch.depth % 2**DOWNSAMPLINGS or cfg.patch.width % 2**DOWNSAMPLINGS:
            raise ValueError(f'Heading patch depth and width must be divisible by {2**DOWNSAMPLINGS} for a centered encoder grid')
        w = cfg.width
        # 4-wide stride-2 kernels keep every grid centered on the patch (see module docstring).
        self.features = nn.Sequential(
            nn.Conv3d(1, w, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(w, 2*w, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(2*w, 4*w, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(4*w, 4*w, 4, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool3d(2), nn.Flatten())
        self.path = nn.Sequential(nn.Linear(4*cfg.window, 64), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(4*w*8+64, 64), nn.SiLU(), nn.Linear(64, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)  # starts as the prior heading
        if cfg.predict_normals:
            self.normal_head = nn.Sequential(nn.Linear(4*w*8+64, 64), nn.SiLU(), nn.Linear(64, 3))
        if cfg.predict_frames:
            self.family_embedding = nn.Embedding(2, 64)
            nn.init.zeros_(self.family_embedding.weight)

    def forward(self, patch, path, family=None):
        return self.forward_outputs(patch, path, family)['heading']

    def forward_outputs(self, patch, path, family=None):
        path_hidden = self.path(path)
        if self.cfg.predict_frames:
            if family is None or family.shape != (len(patch),) or family.dtype not in (torch.int32, torch.int64):
                raise ValueError('Frame model requires a batch of family IDs (H=0, V=1)')
            path_hidden = path_hidden+self.family_embedding(family)
        hidden = torch.cat((self.features(patch), path_hidden), -1)
        out = dict(heading=nn.functional.normalize(self.head(hidden)+patch.new_tensor([0., 0., 1.]), dim=-1))
        if self.cfg.predict_normals:
            out['normal'] = nn.functional.normalize(self.normal_head(hidden), dim=-1)
        if self.cfg.predict_frames:
            from vesuvius.neural_tracing.fiber_follow.heading_model.frames import orthonormal_frame
            out['frame'] = orthonormal_frame(out['heading'], out['normal'])
        return out


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


def model_inputs(vol, cfg, positions, frames, paths, pool=None, *, normal_targets=False):
    """CT patches (tracer sampler and normalization) and path features for each head.

    ``vol`` is the patch volume: the follower's, or its ``patch_volume_spec`` level for a downsampled patch.
    """
    shift = ct_shift(cfg, vol)
    items = [dict(pos=np.asarray(p, np.float64)-shift, frame=np.asarray(f, np.float64)) for p, f in zip(positions, frames)]
    if normal_targets:
        from vesuvius.neural_tracing.fiber_follow.heading_model.normals import sample_training_crops
        patch, normals, weights = sample_training_crops(items, vol, cfg, pool)
    else:
        patch = scalar_crops(items, vol, cfg.patch, pool)
    path = torch.from_numpy(np.stack([path_features(q, p, f, cfg.window) for q, p, f in zip(paths, positions, frames)]))
    return (patch, path, normals, weights) if normal_targets else (patch, path)


def save_heading_model(path, model, **extra):
    torch.save(dict(architecture=model.architecture, config=model.cfg.to_dict(), state=model.state_dict(), **extra), path)


def load_heading_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture') == 'crop_heading_ct_path_v1':
        raise ValueError(f'{path} uses the v1 (off-center) encoder; convert it with scripts/convert_heading_v1.py')
    if checkpoint.get('architecture') not in (ARCHITECTURE, NORMAL_ARCHITECTURE, FRAME_ARCHITECTURE):
        raise ValueError(f'Not a {ARCHITECTURE} checkpoint: {path}')
    model = HeadingNet(HeadingConfig(**checkpoint['config']))
    if model.architecture != checkpoint['architecture']:
        raise ValueError('Heading architecture and config disagree')
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
    def predict_with_normals(self, vol, positions, priors, paths, pool=None, *, families=None):
        """World headings and unsigned world sheet-normal axes, using only the inference patch."""
        if not self.model.cfg.predict_normals:
            raise ValueError('This checkpoint has no normal head')
        if not len(positions):
            return [], []
        frames = prior_frames(priors)
        patch, path = model_inputs(self.patch_volume(vol), self.model.cfg, positions, frames, paths, pool)
        out = self.model.forward_outputs(patch.to(self.device), path.to(self.device), self._families(families))
        heads, normals = (out[k].double().cpu().numpy() for k in ('heading', 'normal'))
        heads = [f @ h for f, h in zip(frames, heads)]
        return ([h if h @ np.asarray(p) >= 0 else -h for h, p in zip(heads, priors)],
                [f @ n for f, n in zip(frames, normals)])

    def _families(self, families):
        from vesuvius.neural_tracing.fiber_follow.heading_model.frames import family_ids
        if not self.model.cfg.predict_frames:
            return None
        if families is None:
            raise ValueError('Supply H/V families for the frame model')
        return family_ids(families, self.device)

    @torch.no_grad()
    def predict_frames(self, vol, positions, priors, paths, families, previous=None, pool=None):
        """World crop frames (u,v,forward); H/V supplied explicitly, optional previous frames prevent roll flips."""
        from vesuvius.neural_tracing.fiber_follow.heading_model.frames import orthonormal_frame
        if not self.model.cfg.predict_frames:
            raise ValueError('This checkpoint is not a family-conditioned frame model')
        if not len(positions):
            return []
        if previous is not None and len(previous) != len(positions):
            raise ValueError('Previous frames must match the position count')
        heads, normals = self.predict_with_normals(vol, positions, priors, paths, pool, families=families)
        anchors = np.stack([np.zeros(3) if f is None else np.asarray(f)[:, 0]
                            for f in (previous if previous is not None else [None]*len(heads))])
        frames = orthonormal_frame(torch.from_numpy(np.stack(heads)), torch.from_numpy(np.stack(normals)),
                                   torch.from_numpy(anchors))
        return list(frames.numpy())
