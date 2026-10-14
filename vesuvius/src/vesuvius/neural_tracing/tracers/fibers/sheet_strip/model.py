"""The sheet-strip follower: the regression crop transformer (models/crop_transformer.py) shown its fiber as a thin
Gaussian tube in a second input channel, predicting the papyrus sheet the fiber lies on across a strip of
+-``strip_half_width`` around it, and still the fiber's continuation.

The strip is a height field in the crop frame. The frame's ``u`` axis is the frame model's sheet normal, ``v`` the
in-sheet direction across the fiber and ``f`` the crop axis along it, so the sheet near the fiber is close to the
plane ``u = 0``. On a regular grid of ``(f, v)`` (``strip_grid``: every crop slab of ``patch_size`` samples split into
``strip_f_points`` rows, ``v`` every ``strip_v_spacing`` in [-strip_half_width, strip_half_width]) the model predicts
the sheet's ``u``. One strip token per slab and ``v`` column reads out its rows' heights: strip tokens are a set of
path tokens (CropTransformer.run_paths), so they attend to the crop context and to each other, and cost only their
own tokens. A token starts from the CNN evidence at its anchor (u = 0, its v, the slab center) and its position;
each refinement pass adds the evidence sampled at the heights it predicted. Heights start at zero (the frame
model's plane).

Sizes are crop-local units (trace voxels) as in models/follower.py; the tube sigma is in crop samples.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from vesuvius.neural_tracing.tracers.models.blocks import MLP
from vesuvius.neural_tracing.tracers.models.crop_transformer import CELL, RegressionConfig, RegressionFollower
from vesuvius.neural_tracing.tracers.models.follower import crop_support, sample_features
from vesuvius.neural_tracing.tracers.models.patch_tokens import _local, slab_depths
from vesuvius.neural_tracing.tracers.shared.policy import DEFAULT_CONFIDENCE


@dataclass
class SheetStripConfig(RegressionConfig):
    model_type: str = 'sheet_strip'
    tube_channel: bool = True  # the fiber as a Gaussian tube in a second input channel
    tube_sigma: float = 1.  # crop samples
    # Lateral bound of the path outputs (models/follower.future_points); None: the crop's. A follower fine-tuned with a
    # wider crop keeps its checkpoint's bound, so its path readout means what it did.
    path_lateral_limit: float | None = None
    strip_half_width: float = 30.  # +-60 voxels of a 0.5-unit crop spacing
    strip_v_spacing: float = 2.5
    strip_f_points: int = 2  # height rows per crop slab (patch_size samples)
    strip_scale: float = 8.  # readout units -> height
    strip_refinement_steps: int = 1
    # Multiscale models (multiscale_fine_width): the strip's evidence reads the fine CNN run over the whole crop
    # cross-section, not only its central multiscale_fine_width samples (the context tokens are unchanged). Without it a
    # strip point beyond the fine box reads only the pooled CNN.
    strip_full_fine: bool = False
    # > 0: every refinement pass also reads the crop's raw input channels (CT and tube) along u at every crop sample
    # within +-strip_profile_reach (crop units) of each strip point, through its own branch (zero at the start).
    strip_profile_reach: float = 0.
    # > 0 (needs strip_full_fine): sheet-band context tokens beside the polar ones, read from the full fine lattice:
    # per crop slab, strip_band_cell x strip_band_cell sample cells tiling |u| <= strip_band_half_height (samples)
    # across the whole v range, where the sheet near the fiber lies; the polar sectors there are tens of samples wide.
    strip_band_half_height: float = 0.
    strip_band_cell: int = 8
    # The strip tokens also read the crop's orientation against the scroll (sheet_strip/data.crop_orientation: which
    # crop directions point outward, the axes' z components; input 'orientation'), e.g. which face of its sheet a
    # fiber of either family lies on.
    strip_orientation: bool = False
    # A per-point error scale (log, crop units) of the last pass's heights, read from its tokens with their gradient
    # stopped (a probe: the heights train as without it); trained by a Laplace likelihood of the heights' error.
    strip_uncertainty: bool = False

    def __post_init__(self):
        super().__post_init__()
        c = self.fine
        columns = 2*self.strip_half_width/self.strip_v_spacing
        if not 0 < self.strip_half_width <= (c.width-1)*c.spacing/2 or abs(columns-round(columns)) > 1e-6:
            raise ValueError('The strip must fit the crop cross-section, its width a multiple of the column spacing')
        if self.strip_f_points < 1 or c.depth % self.patch_size or self.strip_scale <= 0 or self.tube_sigma <= 0:
            raise ValueError('Strip rows, slabs dividing the crop depth, a positive height scale and tube sigma needed')
        if not isinstance(self.strip_refinement_steps, int) or self.strip_refinement_steps < 0:
            raise ValueError('Strip refinement steps must be a nonnegative integer')
        if self.strip_profile_reach < 0 or (self.strip_profile_reach and not self.strip_refinement_steps):
            raise ValueError('A strip profile reach must be nonnegative and needs refinement passes')
        if self.strip_band_half_height and not (self.strip_full_fine and self.multiscale_fine_width is not None
                                                and self.strip_band_cell > 0 and self.strip_band_half_height > 0):
            raise ValueError('Sheet-band tokens need strip_full_fine on a multiscale model and positive cell sizes')
        if self.path_lateral_limit is not None and not 0 < self.path_lateral_limit <= (c.width-1)*c.spacing/2:
            raise ValueError('The path lateral limit must lie inside the crop')

    @property
    def input_channels(self):
        return 2 if self.tube_channel else 1

    @property
    def lateral_limit(self):
        if self.path_lateral_limit is not None:
            return self.path_lateral_limit
        return (self.fine.width-1)*self.fine.spacing/2-self.patch_radius


def extend_strip_stages(state, cfg):
    """A sheet-strip state dict for ``cfg``: refinement stages beyond the checkpoint's start as copies of its last."""
    stage = state.get('strip_stage.weight')
    if stage is None or cfg.strip_refinement_steps <= len(stage):
        return state
    extra = stage[-1:].expand(cfg.strip_refinement_steps-len(stage), -1)
    return dict(state, **{'strip_stage.weight': torch.cat([stage, extra])})


def band_layout(cfg):
    """Sheet-band tokens (strip_band_half_height): per slab (slab-major), cells of strip_band_cell samples across u
    and v tiling |u| <= strip_band_half_height over the whole v range; each cell's patch_points points per axis.
    Points (T, patch_points**3, 3) and centers (T, 3), crop-local."""
    c, n, cell = cfg.fine, cfg.patch_points, cfg.strip_band_cell
    depth, depth_center = slab_depths(cfg)
    rows = max(1, int(round(2*cfg.strip_band_half_height/cell)))
    columns = c.width//cell
    frac = (torch.arange(n, dtype=torch.float64)+.5)/n*cell
    u = (c.width-1)/2-rows*cell/2+torch.arange(rows, dtype=torch.float64)[:, None]*cell+frac  # (R, n) samples
    v = (c.width-columns*cell)/2+torch.arange(columns, dtype=torch.float64)[:, None]*cell+frac  # (V, n)
    s, shape = len(depth), (len(depth), rows, columns, n, n, n)
    points = _local(cfg, u[None, :, None, None, :, None].expand(shape), v[None, None, :, None, None, :].expand(shape),
                    depth[:, None, None, :, None, None].expand(shape)).reshape(-1, n**3, 3)
    centers = _local(cfg, u.mean(-1)[None, :, None].expand(s, rows, columns),
                     v.mean(-1)[None, None, :].expand(s, rows, columns),
                     depth_center[:, None, None].expand(s, rows, columns)).reshape(-1, 3)
    return points.float(), centers.float()


def strip_grid(cfg):
    """The strip's crop-axis rows ``f`` (Nf,) and columns ``v`` (Nv,), crop-local units: the rows split every slab of
    ``patch_size`` samples into ``strip_f_points`` equal parts (at their centers)."""
    c, n = cfg.fine, cfg.strip_f_points
    rows = (torch.arange(c.depth//cfg.patch_size*n, dtype=torch.float64)+.5)*cfg.patch_size/n
    f = (rows-.5-c.behind)*c.spacing  # sample index -> crop-local (rows centered between the slab's samples)
    columns = int(round(2*cfg.strip_half_width/cfg.strip_v_spacing))+1
    v = torch.linspace(-cfg.strip_half_width, cfg.strip_half_width, columns, dtype=torch.float64)
    return f.float(), v.float()


def tube_offsets(sigma, reach=3.):
    """Integer (d, h, w) sample offsets within ``reach`` sigma (plus half a sample) of a point."""
    r = int(math.ceil(reach*sigma))
    grid = torch.stack(torch.meshgrid(*(torch.arange(-r, r+1),)*3, indexing='ij'), -1).reshape(-1, 3)
    return grid[(grid.float()**2).sum(-1) <= (reach*sigma+.5)**2]


@torch.no_grad()
def render_tube(points, valid, crop, sigma, out=None):
    """A Gaussian tube (B, 1, D, H, W) of peak 1 and ``sigma`` samples around crop-local (a, b, c) curve points
    (B, P, 3) spaced at most half a sample apart: each sample takes its largest Gaussian weight over the points.
    ``valid`` (B, P) marks the points."""
    b = len(points)
    height = crop.height or crop.width
    shape = (crop.depth, height, crop.width)
    index = torch.stack((points[..., 2]/crop.spacing+crop.behind, points[..., 1]/crop.spacing+(height-1)/2,
                         points[..., 0]/crop.spacing+(crop.width-1)/2), -1).float()  # (B, P, 3) d, h, w
    offsets = tube_offsets(sigma).to(points.device)
    out = points.new_zeros((b, 1, *shape), dtype=torch.float32) if out is None else out.zero_()
    flat = out.view(-1)
    limit = torch.tensor(shape, device=points.device)
    for row in range(b):  # bounded memory: one crop's points x offsets at a time
        centre = index[row][valid[row]]
        if not len(centre):
            continue
        sample = centre.round().long()[:, None]+offsets  # (P, K, 3)
        weight = torch.exp(-((sample.float()-centre[:, None])**2).sum(-1)/(2*sigma*sigma))
        inside = ((sample >= 0) & (sample < limit)).all(-1)
        sample, weight = sample[inside], weight[inside]
        linear = ((row*shape[0]+sample[:, 0])*shape[1]+sample[:, 1])*shape[2]+sample[:, 2]
        flat.scatter_reduce_(0, linear, weight, 'amax')
    return out


class SheetStripFollower(RegressionFollower):
    """The regression follower plus strip tokens (module docstring); ``training_forward`` returns the path proposals
    and ``strip_heights`` (B, passes, Nf, Nv)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        h, cells, n = cfg.hidden, self.cell_channels, cfg.strip_f_points
        f, v = strip_grid(cfg)
        self.register_buffer('strip_f', f, persistent=False)
        self.register_buffer('strip_v', v, persistent=False)
        slabs = len(f)//n
        centers = f.reshape(slabs, n).mean(-1)
        anchors = torch.zeros(slabs, len(v), 3)
        anchors[..., 1] = v[None]
        anchors[..., 2] = centers[:, None]
        self.register_buffer('strip_anchor', anchors.reshape(-1, 3), persistent=False)  # slab-major tokens
        # CNN evidence and support at the anchor, its (v, f).
        self.strip_query = MLP(cells+1+2, h)
        self.strip_kind = nn.Parameter(torch.zeros(h))
        nn.init.normal_(self.strip_kind, std=.02)
        self.strip_height = nn.Linear(h, n)
        nn.init.normal_(self.strip_height.weight, std=.005)
        nn.init.zeros_(self.strip_height.bias)
        if cfg.strip_refinement_steps:
            # CNN evidence and support at the token's predicted points, their heights.
            self.strip_refinement = MLP(n*(cells+1)+n, h)
            self.strip_stage = nn.Embedding(cfg.strip_refinement_steps, h)
            nn.init.zeros_(self.strip_stage.weight)
        if cfg.strip_band_half_height:
            points, centers = band_layout(cfg)
            self.register_buffer('band_points', points, persistent=False)
            self.register_buffer('band_centers', centers, persistent=False)
            self.strip_band = nn.Linear(points.shape[1]*(cfg.cnn_channels[-1]+1), h)
            self.strip_band_kind = nn.Parameter(torch.zeros(h))
        if cfg.strip_orientation:
            self.strip_orient = MLP(5, h, zero_last=True)
        if cfg.strip_uncertainty:
            self.strip_log_scale = MLP(h, n)
        if cfg.strip_profile_reach:
            k = int(round(cfg.strip_profile_reach/cfg.fine.spacing))
            self.register_buffer('strip_profile_offsets', torch.arange(-k, k+1)*cfg.fine.spacing, persistent=False)
            self.strip_profile = MLP(n*(2*k+1)*cfg.input_channels, h, zero_last=True)

    def extra_cell_tokens(self, ctx, x):
        """Keeps what the strip reads (the full fine lattice, the input crop) in ``ctx``; the sheet-band tokens."""
        if self.cfg.strip_full_fine and len(self.levels) > 1:
            ctx['strip_fine'] = self.cnn(x['fine'])
        if self.cfg.strip_profile_reach:
            ctx['image'] = x['fine']
        if self.cfg.strip_orientation:
            ctx['orientation'] = x['orientation']
        if not self.cfg.strip_band_half_height:
            return None
        b, (t, p, _) = len(x['fine']), self.band_points.shape
        points = self.band_points.reshape(1, -1, 3).expand(b, -1, -1)
        values, support = sample_features(ctx['strip_fine'].float(), points, self.cfg.fine, self.levels[0][1])
        values = torch.cat((values, support[..., None].to(values.dtype)), -1).reshape(b, t, -1)
        tokens = self.strip_band(values)
        return tokens+(self.cell_position(self.band_centers/16.)+self.kind[CELL]+self.strip_band_kind).to(tokens.dtype)

    def strip_profiles(self, ctx, points):
        """The crop's input channels along u around strip points (B, T, n, 3): (B, T, n*K*C), trilinear."""
        c = self.cfg.fine
        b, t, n, _ = points.shape
        image = ctx['image'].float()
        depth, height, width = image.shape[2:]
        p = points.float()[..., None, :].expand(-1, -1, -1, len(self.strip_profile_offsets), -1).clone()
        p[..., 0] = p[..., 0]+self.strip_profile_offsets
        grid = torch.stack(((p[..., 0]/c.spacing+(width-1)/2)/(width-1),
                            (p[..., 1]/c.spacing+(height-1)/2)/(height-1),
                            (p[..., 2]/c.spacing+c.behind)/(depth-1)), -1)*2-1
        values = F.grid_sample(image, grid.reshape(b, -1, 1, 1, 3), align_corners=True)  # (B, C, M, 1, 1)
        return values.reshape(b, -1, t, n*len(self.strip_profile_offsets)).permute(0, 2, 3, 1).reshape(b, t, -1)

    def strip_evidence(self, ctx, points):
        """The evidence (models/crop_transformer.evidence) of strip points, the fine level over the whole crop with
        ``strip_full_fine``."""
        if 'strip_fine' not in ctx:
            return self.evidence(ctx, points)
        points = points.float()
        values = [sample_features(ctx['strip_fine'].float(), points, self.cfg.fine, self.levels[0][1])[0]]
        values += [sample_features(features.float(), points, crop, stride)[0]
                   for features, (crop, stride) in zip(ctx['levels'][1:], self.levels[1:])]
        support = crop_support(points, self.cfg.fine)
        return torch.cat((*values, support[..., None].to(values[0].dtype)), -1)

    def strip_points(self, heights):
        """Crop-local points (B, T, n, 3) of token heights (B, T, n)."""
        b, t, n = heights.shape
        f = self.strip_f.reshape(-1, n)[:, None].expand(-1, len(self.strip_v), -1).reshape(t, n)
        v = self.strip_anchor[:, 1, None].expand(t, n)
        return torch.stack((heights, v[None].expand(b, -1, -1), f[None].expand(b, -1, -1)), -1)

    def strip_layout(self, heights):
        """Token heights (B, T, n) -> the strip grid (B, Nf, Nv)."""
        b, _, n = heights.shape
        return heights.reshape(b, -1, len(self.strip_v), n).transpose(2, 3).reshape(b, -1, len(self.strip_v))

    def strip(self, ctx):
        """Strip heights (B, 1+strip_refinement_steps, Nf, Nv): the first pass, then each refinement."""
        return self.strip_outputs(ctx)[0]

    def strip_outputs(self, ctx):
        """The strip heights (``strip``) and, with ``strip_uncertainty``, the last pass's log error scale (B, Nf, Nv),
        else None."""
        cfg, b = self.cfg, len(ctx['cells'])
        anchors = self.strip_anchor[None].expand(b, -1, -1)
        tokens = self.strip_query(torch.cat((self.strip_evidence(ctx, anchors), anchors[..., 1:]/16.), -1))
        if cfg.strip_orientation:
            tokens = tokens+self.strip_orient(ctx['orientation'].to(tokens.dtype))[:, None]
        decoded = self.run_paths(ctx, (tokens+self.strip_kind.to(tokens.dtype))[:, None])[:, 0]
        heights = [cfg.strip_scale*self.strip_height(decoded).float()]
        for stage in range(cfg.strip_refinement_steps):
            last = heights[-1].detach()
            points = self.strip_points(last)
            evidence = self.strip_evidence(ctx, points.flatten(1, 2)).reshape(b, last.shape[1], -1)
            refreshed = self.strip_refinement(torch.cat((evidence, last/16.), -1))
            tokens = decoded+refreshed.to(decoded.dtype)+self.strip_stage.weight[stage].to(decoded.dtype)
            if self.cfg.strip_profile_reach:
                tokens = tokens+self.strip_profile(self.strip_profiles(ctx, points)).to(decoded.dtype)
            decoded = self.run_paths(ctx, tokens[:, None])[:, 0]
            heights.append(cfg.strip_scale*self.strip_height(decoded).float())
        scale = self.strip_layout(self.strip_log_scale(decoded.detach()).float()) if cfg.strip_uncertainty else None
        return torch.stack([self.strip_layout(value) for value in heights], 1), scale

    def training_forward(self, x, hist, hmask, threshold, targets=None):
        ctx = self.context(x, hist, hmask)
        heights, scale = self.strip_outputs(ctx)
        out = dict(self.predict(ctx, threshold), strip_heights=heights)
        if scale is not None:
            out['strip_log_scale'] = scale
        return out

    def forward_strip(self, x, hist, hmask):
        """Strip heights (B, Nf, Nv) of the last pass, without the path."""
        return self.strip(self.context(x, hist, hmask))[:, -1]

    def forward(self, x, hist, hmask, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        output = self.training_forward(x, hist, hmask, confidence_threshold)
        return self.select_prediction(output, confidence_threshold, n_commit)
