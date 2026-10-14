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

from vesuvius.neural_tracing.tracers.models.blocks import MLP
from vesuvius.neural_tracing.tracers.models.crop_transformer import RegressionConfig, RegressionFollower
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
        cfg, b = self.cfg, len(ctx['cells'])
        anchors = self.strip_anchor[None].expand(b, -1, -1)
        tokens = self.strip_query(torch.cat((self.evidence(ctx, anchors), anchors[..., 1:]/16.), -1))
        decoded = self.run_paths(ctx, (tokens+self.strip_kind.to(tokens.dtype))[:, None])[:, 0]
        heights = [cfg.strip_scale*self.strip_height(decoded).float()]
        for stage in range(cfg.strip_refinement_steps):
            last = heights[-1].detach()
            points = self.strip_points(last)
            evidence = self.evidence(ctx, points.flatten(1, 2)).reshape(b, last.shape[1], -1)
            refreshed = self.strip_refinement(torch.cat((evidence, last/16.), -1))
            tokens = decoded+refreshed.to(decoded.dtype)+self.strip_stage.weight[stage].to(decoded.dtype)
            decoded = self.run_paths(ctx, tokens[:, None])[:, 0]
            heights.append(cfg.strip_scale*self.strip_height(decoded).float())
        return torch.stack([self.strip_layout(value) for value in heights], 1)

    def training_forward(self, x, hist, hmask, threshold, targets=None):
        ctx = self.context(x, hist, hmask)
        return dict(self.predict(ctx, threshold), strip_heights=self.strip(ctx))

    def forward_strip(self, x, hist, hmask):
        """Strip heights (B, Nf, Nv) of the last pass, without the path."""
        return self.strip(self.context(x, hist, hmask))[:, -1]

    def forward(self, x, hist, hmask, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        output = self.training_forward(x, hist, hmask, confidence_threshold)
        return self.select_prediction(output, confidence_threshold, n_commit)
