"""Whole-trace sequence follower (model_type 'sequence'): one transformer reads the trace and predicts the path.

Every decision encodes one head crop (default 80 deep x 64 x 64 at 0.5 trace voxels, 16 samples behind the head:
8 voxels behind, 31.5 ahead, +-16 across) with a CNN: a full-resolution 16-channel residual block, then 32/64/128
channels with two residual blocks per stage at strides 2/4/8. Its stride-8 cells become the crop tokens
(10 x 8 x 8 = 640). One transformer (12 layers, width 256, 8 heads, FFN 2048) reads, per decision:

  * history tokens, one per earlier committed step of the trace: that step's CNN features pooled at its committed
    points (mean and max), its committed displacement and its arc length. They are inputs, not model outputs, so a
    training episode runs in parallel. In every layer they attend causally to each other (the same weights);
  * the crop tokens and one query token per forward plane (31): these attend to each other and to every earlier
    history token, each offset by that step's pose in this decision's crop frame (head position and age).

The query tokens' final states give each plane's lateral position and hazard logit (survival confidence), so the
same model that reads the whole trace predicts the path and its confidence; acceptance (gate plane, default 16),
commits and losses are the shared ones. Training (train/sequence.py) runs whole episodes; tracing keeps each trace's
per-layer history states and extends them by one token per commit, computing exactly the training states.
"""
from dataclasses import dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from vesuvius.models.build.resblocks import BasicBlockD, StackedResidualBlocks

from vesuvius.neural_tracing.fiber_follow.models.model import (
    CoordinateRegressionConfig, device_vector, future_points, proposal_output, sample_features, select_refinement)
from vesuvius.neural_tracing.fiber_follow.models.survival_confidence import survival_predictions
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.tracing.policy import DEFAULT_CONFIDENCE

POSITION_FREQUENCIES = 8  # sinusoidal arc length, periods 16 .. 2048 trace voxels


@dataclass
class SequenceConfig(CoordinateRegressionConfig):
    """The shared crop/horizon/label contract (CoordinateRegressionConfig fields read by data, losses and tracing)
    plus this model's own sizes. ``layers``/``heads`` size the transformer; refinement is not used."""
    model_type: str = 'sequence'
    fine: CropSpec = field(default_factory=lambda: CropSpec(depth=80, width=64, behind=16, spacing=.5))
    n_future: int = 31
    gate_plane: int | None = 16
    layers: int = 12
    heads: int = 8
    ffn: int = 2048
    cnn_channels: tuple = (16, 32, 64, 128)  # full-resolution block, then stages at strides 2, 4, 8
    cnn_blocks: int = 2
    history_limit: int = 512  # most recent history tokens a decision reads
    memory: str = 'none'  # no memory inputs for the data builders: the history is the trace's own steps
    recurrent_refinement_steps: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.cnn_channels = tuple(int(c) for c in self.cnn_channels)
        if len(self.cnn_channels) != 4 or min(self.cnn_channels) < 1 or self.cnn_blocks < 1 or self.ffn < 1:
            raise ValueError('The CNN needs four positive channel widths and positive blocks; the FFN a positive width')
        if self.recurrent_refinement_steps or self.memory != 'none' or self.path_planes != 'future' or self.tube_head:
            raise ValueError('The sequence follower has no refinement, memory inputs, whole-crop planes or tube head')
        if self.history_limit < 1:
            raise ValueError('History limit must be positive')
        c = self.fine
        if c.depth % 8 or c.width % 8:
            raise ValueError('Sequence crops must be multiples of eight (stride-8 tokens)')

    @property
    def token_stride(self):
        return (8, 8, 8)

    @property
    def token_offset(self):
        return (0., 0., 0.)  # stride-2 3x3x3 convolutions with padding 1 centre cell i on input sample 2i


def arc_position(travelled):
    periods = 16.*2.**torch.arange(POSITION_FREQUENCIES, device=travelled.device, dtype=torch.float32)
    angle = 2*math.pi*travelled.float()[..., None]/periods
    return torch.cat((angle.sin(), angle.cos()), -1)


def relative_pose(past_pos, past_travelled, pos, frame, travelled):
    """Past heads (..., N, 3) in a decision's crop frame (..., 3, 3) at (..., 3), and their ages (..., N)."""
    local = torch.einsum('...ni,...ij->...nj', (past_pos-pos[..., None, :]).double(), frame.double())
    return local.float(), (travelled[..., None]-past_travelled).float()


class CropCNN(nn.Module):
    """Residual CNN stages (channels, initial stride, blocks per stage) over a one-channel crop; the first stage's
    convolutions carry biases."""
    def __init__(self, channels, strides, blocks, checkpointing=False):
        super().__init__()
        options = dict(conv_op=nn.Conv3d, kernel_size=3, norm_op=nn.InstanceNorm3d,
                       norm_op_kwargs=dict(eps=1e-5, affine=True), nonlin=nn.ReLU, nonlin_kwargs=dict(inplace=True),
                       block=BasicBlockD)
        inputs = (1, *channels[:-1])
        self.stages = nn.ModuleList(
            StackedResidualBlocks(n_blocks=n, input_channels=i, output_channels=c, initial_stride=s, conv_bias=not index,
                                  **options)
            for index, (i, c, s, n) in enumerate(zip(inputs, channels, strides, blocks)))
        self.checkpointing = checkpointing

    def forward(self, image):
        x = image
        for stage in self.stages:
            x = checkpoint(stage, x, use_reentrant=False) if self.checkpointing and torch.is_grad_enabled() else stage(x)
        return x


class SequenceLayer(nn.Module):
    """One pre-norm transformer layer shared by the history stream and the decision tokens."""
    def __init__(self, width, heads, ffn):
        super().__init__()
        if width % heads:
            raise ValueError('Transformer width must divide by its heads')
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.out = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width))

    def split(self, x):
        b, n, width = x.shape
        shape = lambda t: t.reshape(b, n, self.heads, width//self.heads).transpose(1, 2)
        return tuple(map(shape, self.qkv(x).chunk(3, -1)))

    def merge(self, value):
        b, heads, n, d = value.shape
        return self.out(value.transpose(1, 2).reshape(b, n, heads*d))

    def history(self, s):
        """Causal history stream: (B, T, W) -> (B, T, W)."""
        q, k, v = self.split(self.norm1(s))
        s = s+self.merge(F.scaled_dot_product_attention(q, k, v, is_causal=True))
        return s+self.ffn(self.norm2(s))

    def history_step(self, s, past):
        """One new history token (B, 1, W) after its earlier tokens' inputs to this layer (B, n, W)."""
        q, k, v = self.split(self.norm1(torch.cat((past, s), 1)))
        s = s+self.merge(F.scaled_dot_product_attention(q[:, :, -1:], k, v))
        return s+self.ffn(self.norm2(s))

    def decision(self, x, history, padding):
        """Decision tokens (R, M, W) attend to themselves and to earlier history (R, N, W; ``padding`` True = absent)."""
        _, hk, hv = self.split(self.norm1(history))
        q, k, v = self.split(self.norm1(x))
        k, v = torch.cat((hk, k), 2), torch.cat((hv, v), 2)
        visible = torch.cat((~padding, padding.new_ones(len(x), x.shape[1])), 1)
        x = x+self.merge(F.scaled_dot_product_attention(q, k, v, attn_mask=visible[:, None, None, :]))
        return x+self.ffn(self.norm2(x))


class SequenceFollower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.model_type = cfg, cfg.model_type
        h, cells = cfg.hidden, cfg.cnn_channels[-1]
        # A full-resolution block, then stages at strides 2, 4, 8.
        self.cnn = CropCNN(cfg.cnn_channels, (1, 2, 2, 2), (1,)+(cfg.cnn_blocks,)*3, cfg.activation_checkpointing)
        self.cell_token = nn.Linear(cells, h)
        self.cell_position = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h))
        width = 2*cells+4+2*POSITION_FREQUENCIES
        self.history_token = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, h), nn.SiLU(), nn.Linear(h, h))
        self.history_pose = nn.Sequential(nn.Linear(4, h), nn.SiLU(), nn.Linear(h, h))
        self.query_token = nn.Sequential(nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h))
        self.kind = nn.Parameter(torch.zeros(3, h))  # history, crop cell, plane query
        nn.init.normal_(self.kind, std=.02)
        self.layers = nn.ModuleList(SequenceLayer(h, cfg.heads, cfg.ffn) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(h)
        self.coordinates = nn.Linear(h, 2)
        nn.init.normal_(self.coordinates.weight, std=.005)
        nn.init.zeros_(self.coordinates.bias)
        self.hazard = nn.Linear(h, 1)
        planes = torch.arange(1, cfg.n_future+1).float()*cfg.future_step
        self.register_buffer('planes', planes, persistent=False)
        self.register_buffer('cell_xyz', self.cell_coordinates(cfg), persistent=False)

    @staticmethod
    def cell_coordinates(cfg):
        """Crop-local xyz of the stride-8 cells, flattened in (d, h, w) order."""
        c, stride = cfg.fine, cfg.token_stride[0]
        d, y, x = torch.meshgrid(*(torch.arange(n//stride).float()*stride for n in (c.depth, c.width, c.width)), indexing='ij')
        return torch.stack((x-(c.width-1)/2, y-(c.width-1)/2, d-c.behind), -1).reshape(-1, 3)*c.spacing

    # ------------------------------------------------------------------ tokens
    def encode(self, image):
        """Stride-8 CNN cells (B, C, D/8, W/8, W/8) of head crops (B, 1, D, W, W)."""
        return self.cnn(image)

    def crop_tokens(self, cells):
        tokens = self.cell_token(cells.flatten(2).transpose(1, 2))
        return tokens+self.cell_position(self.cell_xyz/16.).to(tokens.dtype)+self.kind[1].to(tokens.dtype)

    def query_tokens(self, count, like):
        tokens = self.query_token((self.planes/16.)[:, None].to(like.dtype))+self.kind[2].to(like.dtype)
        return tokens[None].expand(count, -1, -1)

    def step_features(self, cells, points, valid):
        """(B, 2C) CNN cells pooled (mean, max) at a step's committed crop-local points (B, P, 3)."""
        values, support = sample_features(cells.float(), points.float(), self.cfg.fine, self.cfg.token_stride,
                                          self.cfg.token_offset)
        valid = valid.bool() & support
        count = valid.sum(1, keepdim=True)
        mean = torch.where(valid[..., None], values, 0.).sum(1)/count.clamp_min(1)
        peak = torch.where(count > 0, torch.where(valid[..., None], values, -torch.inf).amax(1), 0.)
        return torch.cat((mean, peak), -1)

    def history_input(self, features, displacement, travelled):
        """Layer-0 history tokens (..., W) of committed steps."""
        geometry = torch.cat((displacement/16., displacement.norm(dim=-1, keepdim=True)/16.), -1)
        token = self.history_token(torch.cat((features.float(), geometry.float(), arc_position(travelled)), -1))
        return token+self.kind[0].to(token.dtype)

    def pose(self, relative, age):
        features = torch.cat((relative/32., torch.log1p(age.clamp_min(0)[..., None])/math.log(4097.)), -1)
        return self.history_pose(features)

    # ------------------------------------------------------------------ prediction
    def outputs(self, queries, confidence_threshold, n_commit):
        """Plane points and survival confidence from the final query states, in the shared output contract."""
        cfg = self.cfg
        decoded = self.norm(queries)
        points = future_points(self.coordinates(decoded), self.planes, cfg)
        hazards = self.hazard(decoded).squeeze(-1).float()
        out = proposal_output(points, [points], [(hazards, *survival_predictions(hazards))],
                              [torch.ones(len(points), dtype=torch.bool, device=points.device)])
        return select_refinement(out, cfg, confidence_threshold, n_commit)

    def decide(self, cells, history, pose, padding, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        """Decisions from their crop cells and earlier history: ``history`` is a list (one per layer) of (R, N, W)
        layer inputs of the history tokens, ``pose`` (R, N, W) their pose embedding, ``padding`` (R, N)."""
        x = self.crop_tokens(cells)
        x = torch.cat((x, self.query_tokens(len(x), x)), 1)
        for layer, states in zip(self.layers, history):
            x = layer.decision(x, states+pose.to(states.dtype), padding)
        return self.outputs(x[:, -self.cfg.n_future:], confidence_threshold, n_commit)

    def history_states(self, tokens):
        """Per-layer inputs (list of L tensors (B, T, W)) of whole history sequences of layer-0 tokens (B, T, W)."""
        states = [tokens]
        for layer in self.layers[:-1]:
            states.append(layer.history(states[-1]))
        return states

    def extend_history(self, token, past):
        """Per-layer inputs (list of L (B, W)) of one new history token (B, W) after its trace's earlier tokens'
        per-layer inputs ``past`` (list of L (B, n, W), n may be 0): exactly ``history_states`` incrementally."""
        states, s = [token], token[:, None]
        for layer, previous in zip(self.layers[:-1], past):
            s = layer.history_step(s, previous)
            states.append(s[:, 0])
        return states

    def forward(self, x, hist, hmask, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        """One decision per row; earlier history from ``x['sequence_states']`` (R, L, N, W), ``sequence_relative``
        (R, N, 3), ``sequence_age`` (R, N) and ``sequence_padding`` (R, N), or none. Also returns the CNN cells
        (``sequence_cells``) for the tracer's next history token."""
        cells = self.encode(x['fine'])
        rows, width = len(cells), self.cfg.hidden
        if 'sequence_states' in x:
            history = list(x['sequence_states'].unbind(1))
            pose, padding = self.pose(x['sequence_relative'], x['sequence_age']), x['sequence_padding'].bool()
        else:
            history = [cells.new_zeros(rows, 1, width)]*len(self.layers)
            pose, padding = cells.new_zeros(rows, 1, width), torch.ones(rows, 1, dtype=torch.bool, device=cells.device)
        out = self.decide(cells, history, pose, padding, confidence_threshold, n_commit)
        out['sequence_cells'] = cells
        return out
