"""Crop transformer: a residual CNN tokenizes the crop and one transformer reads every token and predicts the path.

The crop (by default 144 deep x 104 x 104 at 0.5 trace voxels, 72 samples behind the head) goes through residual CNN
stages, by default 32/64/128 channels at strides 2/4/8 with 1/2/2 blocks. Each stride-8 cell becomes one token
(18 x 13 x 13 = 3042), with no other encoder. One transformer (by default 12 layers, width 256, 8 heads, FFN 2048)
reads, in every layer:

  * context tokens: the crop cells, the recent observed path and seed (reference points, with CNN features sampled at
    each point) and the observed-path geometry tokens (models/path_geometry.py). They attend to each other (key
    padding for absent references);
  * path tokens, one per forward plane (planes 1..n_future): these attend to every context token and to the path
    tokens of their own set, through the same layer weights.

Context tokens never attend to path tokens, so a decision's context keys/values are computed once and every further
set of path tokens (refinement passes, flow draws and solver steps, scoring passes) costs only its own tokens. The
two model types share this backbone and differ only in what their path tokens carry and read out:

  * 'regression': the path tokens start at the centerline and read out each plane's lateral position and hazard
    logit (survival confidence) together; optional refinement passes feed the proposal, its evidence and its detached
    confidence back as new path tokens;
  * 'flow' (flow matching, models/flow.py): path tokens carry a noisy path and the flow time and read out the
    velocity; a scoring set of path tokens carries a finished proposal and reads out its hazard logits. The time also
    modulates every layer's two branches and the final norm of the velocity path tokens (DiT's adaLN-Zero: each
    branch is gated by a zero-initialized alpha, so every velocity-token block starts as the identity). Context and
    scoring tokens have no time and are never modulated.

Both keep the shared output contract, losses, acceptance (gate plane), commits and trainer (train/train.py).
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.models.blocks import CropCNN, TransformerLayer
from vesuvius.neural_tracing.fiber_follow.models.flow import FlowMatching, FlowOptions, time_embedding
from vesuvius.neural_tracing.fiber_follow.models.model import (
    REFERENCE_METADATA, FollowerConfig, path_readout, plane_coordinates, proposal_output, read_path,
    reference_metadata, reference_points, sample_features, select_refinement, token_coordinates)
from vesuvius.neural_tracing.fiber_follow.models.path_geometry import PathGeometryTokens
from vesuvius.neural_tracing.fiber_follow.tracing.policy import DEFAULT_CONFIDENCE, commit_prefix

CELL, REFERENCE, GEOMETRY, PATH, SCORED = range(5)  # token kinds


@dataclass
class CropTransformerArchitecture:
    """Sizes of the crop transformer backbone, mixed into the regression and flow configurations (``layers``/``heads``
    size the transformer)."""
    layers: int = 12
    heads: int = 8
    ffn: int = 2048
    cnn_channels: tuple = (32, 64, 128)  # one stage per entry, each at stride 2
    cnn_blocks: tuple = (1, 2, 2)
    # Per-head RMS normalization of attention queries and keys (models/blocks.py). Unbounded, the flow model's adaLN
    # time modulation of the path tokens grew their q/k norms until path attention saturated and training diverged.
    qk_norm: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.cnn_channels = tuple(int(c) for c in self.cnn_channels)
        self.cnn_blocks = tuple(int(b) for b in self.cnn_blocks)
        if (not self.cnn_channels or len(self.cnn_blocks) != len(self.cnn_channels)
                or min(self.cnn_channels+self.cnn_blocks) < 1 or type(self.ffn) is not int or self.ffn < 1):
            raise ValueError('The crop CNN needs positive channels and blocks per stage; the FFN a positive width')
        if min(self.layers, self.heads) < 1 or self.hidden % self.heads:
            raise ValueError('Positive layers and heads required; hidden must divide by heads')
        stride = self.token_stride[0]
        if self.fine.depth % stride or self.fine.width % stride:
            raise ValueError(f'Crop transformer crops must be multiples of the token stride ({stride})')

    @property
    def token_stride(self):
        stride = 2**len(self.cnn_channels)
        return (stride,)*3


@dataclass
class RegressionConfig(CropTransformerArchitecture, FollowerConfig):
    model_type: str = 'regression'
    n_future: int = 35
    gate_plane: int | None = 16
    recurrent_refinement_steps: int = 3


@dataclass
class FlowConfig(FlowOptions, CropTransformerArchitecture, FollowerConfig):
    model_type: str = 'flow'
    n_future: int = 16
    gate_plane: int | None = None
    qk_norm: bool = True


class CropTransformerLayer(TransformerLayer):
    """One pre-norm transformer layer: context tokens attend to each other; path tokens attend to the context and
    to the path tokens their mask allows."""

    def context(self, x, padding, last=False):
        """Context tokens (B, N, W), ``padding`` (B, N) True = absent: the next states (None after the last layer,
        whose context output nothing reads) and this layer's context keys/values."""
        q, k, v = self.split(self.norm1(x))
        if last:
            return None, (k, v)
        x = x+self.merge(F.scaled_dot_product_attention(q, k, v, attn_mask=~padding[:, None, None, :]))
        return x+self.ffn(self.norm2(x)), (k, v)

    def queries(self, x, context, mask, modulation=None):
        """Path tokens (B, M, W) after this layer; ``context`` its context keys/values, ``mask`` (B, M, N+M) the
        context and path keys each path token reads. ``modulation`` (B, M, 2, 3, W) conditions the attention and
        FFN branches per token (adaLN-Zero): normalized input * (1+scale) + shift, branch output * gate, so zero
        modulation is the identity."""
        if modulation is None:
            branch_input = lambda value, norm, i: norm(value)
            residual = lambda value, branch, i: value+branch
        else:
            shift, scale, gate = modulation.unbind(-2)  # each (B, M, 2, W)
            branch_input = lambda value, norm, i: norm(value)*(1+scale[..., i, :])+shift[..., i, :]
            residual = lambda value, branch, i: value+gate[..., i, :]*branch
        q, k, v = self.split(branch_input(x, self.norm1, 0))
        k, v = torch.cat((context[0], k), 2), torch.cat((context[1], v), 2)
        x = residual(x, self.merge(F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])), 0)
        return residual(x, self.ffn(branch_input(x, self.norm2, 1)), 1)


class CropTransformer(nn.Module):
    """CNN tokens, context tokens and the transformer shared by the regression and flow models."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.model_type = cfg, cfg.model_type
        h, cells = cfg.hidden, cfg.cnn_channels[-1]
        self.cnn = CropCNN(cfg.cnn_channels, (2,)*len(cfg.cnn_channels), cfg.cnn_blocks)
        self.cell_token = nn.Linear(cells, h)
        self.cell_position = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h))
        self.reference_token = nn.Sequential(nn.Linear(cells+REFERENCE_METADATA, h), nn.SiLU(), nn.Linear(h, h))
        self.path_geometry = PathGeometryTokens(cfg)
        self.kind = nn.Parameter(torch.zeros(5, h))  # CELL, REFERENCE, GEOMETRY, PATH, SCORED
        nn.init.normal_(self.kind, std=.02)
        self.layers = nn.ModuleList(CropTransformerLayer(h, cfg.heads, cfg.ffn, cfg.qk_norm) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(h)
        self.register_buffer('planes', plane_coordinates(cfg), persistent=False)
        self.plane_scale = cfg.n_future*cfg.future_step  # forward-distance normalization of path tokens
        self.register_buffer('cell_xyz', token_coordinates(cfg).reshape(-1, 3), persistent=False)

    def sample_cells(self, ctx, points):
        """CNN cell features (B, M, C) and crop support (B, M) at crop-local points (B, M, 3)."""
        return sample_features(ctx['cells'].float(), points.float(), self.cfg.fine, self.cfg.token_stride)

    def evidence(self, ctx, points):
        values, support = self.sample_cells(ctx, points)
        return torch.cat((values, support[..., None].to(values.dtype)), -1)

    def context(self, x, hist, hmask):
        """Every layer's context keys/values for one decision per row, and its CNN cells."""
        cfg = self.cfg
        cells = self.cnn(x['fine'])
        tokens = self.cell_token(cells.flatten(2).transpose(1, 2))
        dtype = tokens.dtype
        tokens = tokens+self.cell_position(self.cell_xyz/16.).to(dtype)+self.kind[CELL].to(dtype)
        ctx = dict(cells=cells)
        references, mask = reference_points(x, hist, hmask, cfg.fine)
        local, _ = self.sample_cells(ctx, references)
        references = self.reference_token(torch.cat((local, reference_metadata(x, hist, references, mask)), -1))
        valid = x['path_geometry_valid'].bool()
        geometry = self.path_geometry(x['path_geometry'], valid)
        sequence = torch.cat((tokens, (references+self.kind[REFERENCE]).to(dtype),
                              (geometry+self.kind[GEOMETRY]).to(dtype)), 1)
        padding = torch.cat((torch.zeros(tokens.shape[:2], device=tokens.device, dtype=torch.bool), ~mask, ~valid), 1)
        pairs = []
        for index, layer in enumerate(self.layers):
            sequence, pair = layer.context(sequence, padding, last=index == len(self.layers)-1)
            pairs.append(pair)
        ctx.update(context=pairs, padding=padding)
        return ctx

    def run_paths(self, ctx, tokens, modulation=None):
        """Final states (B, G, P, W) of G independent sets of path tokens (B, G, P, W) over one decision's context.
        ``modulation`` (per-layer list of (B, G, 2, 3, W), final-norm (B, G, 2, W) shift/scale) conditions each set's
        tokens (adaLN-Zero)."""
        b, g, p, w = tokens.shape
        group = torch.arange(g, device=tokens.device).repeat_interleave(p)
        own = (group[:, None] == group[None, :])[None]
        mask = torch.cat(((~ctx['padding'])[:, None].expand(b, g*p, -1), own.expand(b, g*p, g*p)), -1)
        x = tokens.reshape(b, g*p, w)
        per_token = lambda m: m[:, :, None].expand(b, g, p, *m.shape[2:]).reshape(b, g*p, *m.shape[2:]).to(x.dtype)
        layers = [None]*len(self.layers) if modulation is None else modulation[0]
        for layer, pair, condition in zip(self.layers, ctx['context'], layers):
            x = layer.queries(x, pair, mask, None if condition is None else per_token(condition))
        x = self.norm(x).reshape(b, g, p, w)
        if modulation is not None:
            shift, scale = modulation[1][:, :, None].to(x.dtype).unbind(-2)
            x = x*(1+scale)+shift
        return x

    def centerline(self, like):
        """Planes 1..n_future on the crop axis (B, P, 3)."""
        reference = like.new_zeros(len(like), len(self.planes), 3)
        reference[..., 2] = self.planes
        return reference

    def proposal_output(self, initial, refinements, scores, valid):
        return proposal_output(initial, refinements, scores, valid)


class RegressionFollower(CropTransformer):
    """Coordinate regression: plane positions and survival confidence from one set of path tokens."""

    def __init__(self, cfg):
        super().__init__(cfg)
        h, cells = cfg.hidden, cfg.cnn_channels[-1]
        # CNN evidence and support at the centerline, forward distance.
        self.query = nn.Sequential(nn.Linear(cells+2, h), nn.SiLU(), nn.Linear(h, h))
        self.coordinates, self.hazard = path_readout(h)
        if cfg.recurrent_refinement_steps:
            # CNN evidence and support at the proposal, its coordinates, detached failure/survival.
            self.refinement_fusion = nn.Sequential(nn.Linear(cells+1+3+2, h), nn.SiLU(), nn.Linear(h, h))
            self.refinement_stage = nn.Embedding(cfg.recurrent_refinement_steps, h)
            nn.init.zeros_(self.refinement_stage.weight)

    def decode(self, ctx, tokens):
        decoded = self.run_paths(ctx, tokens[:, None])[:, 0]
        return (decoded, *read_path(decoded, self.coordinates, self.hazard, self.planes, self.cfg))

    def predict(self, ctx, threshold):
        """Fixed proposal slots: refinement passes run for every row and an accepted row keeps its last attempt (path
        tokens are cheap, so tracing does not compact accepted rows either)."""
        cfg = self.cfg
        reference = self.centerline(ctx['cells'])
        query = self.query(torch.cat((self.evidence(ctx, reference), reference[..., 2:]/self.plane_scale), -1))
        decoded, points, score = self.decode(ctx, query+self.kind[PATH].to(query.dtype))
        refinements, scores = [points], [score]
        active = torch.ones(len(points), device=points.device, dtype=torch.bool)
        valid = [active]
        for stage in range(cfg.recurrent_refinement_steps):
            gate = cfg.gate_horizon
            counts, _ = commit_prefix(points, score[2][..., :gate].detach(), threshold, gate, cfg.max_recovery_distance)
            active = active & (counts < gate)
            # Geometry cannot teach the confidence to manufacture convenient feedback.
            feedback = torch.stack((score[0].sigmoid(), score[2]), -1).detach()
            refreshed = self.refinement_fusion(torch.cat((self.evidence(ctx, points), points/16., feedback), -1))
            next_decoded, next_points, next_score = self.decode(
                ctx, decoded+refreshed.to(decoded.dtype)+self.refinement_stage.weight[stage].to(decoded.dtype))
            decoded = torch.where(active[:, None, None], next_decoded, decoded)
            points = torch.where(active[:, None, None], next_points, points)
            score = tuple(torch.where(active[:, None], new, old) for new, old in zip(next_score, score))
            refinements.append(points)
            scores.append(score)
            valid.append(active)
        return proposal_output(refinements[0], refinements, scores, valid)

    def training_forward(self, x, hist, hmask, threshold, targets=None):
        return self.predict(self.context(x, hist, hmask), threshold)

    def select_prediction(self, output, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        return select_refinement(output, self.cfg, confidence_threshold, n_commit)

    def forward(self, x, hist, hmask, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        return self.select_prediction(self.predict(self.context(x, hist, hmask), confidence_threshold),
                                      confidence_threshold, n_commit)


class FlowFollower(FlowMatching, CropTransformer):
    """Flow matching (models/flow.py) on the crop transformer: velocity and proposal scoring are path-token sets."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.init_flow(cfg)
        h, cells = cfg.hidden, cfg.cnn_channels[-1]
        # CNN evidence and support at the noisy path, its residual-scale coordinates, forward distance.
        self.query = nn.Sequential(nn.Linear(cells+1+2+1, h), nn.SiLU(), nn.Linear(h, h))
        self.time = nn.Sequential(nn.Linear(64, h), nn.SiLU(), nn.Linear(h, h))
        # Per layer: shift/scale/gate of the attention and FFN branches; then the final norm's shift/scale.
        self.time_modulation = nn.ModuleList(nn.Linear(h, 6*h) for _ in range(cfg.layers))
        self.output_modulation = nn.Linear(h, 2*h)
        for linear in (*self.time_modulation, self.output_modulation):
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)
        self.velocity = nn.Linear(h, 2)
        nn.init.normal_(self.velocity.weight, std=.01)
        nn.init.zeros_(self.velocity.bias)
        # CNN evidence and support at a finished proposal, its coordinates.
        self.score = nn.Sequential(nn.Linear(cells+1+3, h), nn.SiLU(), nn.Linear(h, h))
        self.hazard = nn.Linear(h, 1)

    def velocity_field(self, ctx, y, t):
        """Velocity (B, D, P, 2) of D draws per decision."""
        b, draws, planes, _ = y.shape
        points = self.to_points(y)
        evidence = self.evidence(ctx, points.reshape(b, draws*planes, 3)).reshape(b, draws, planes, -1)
        tokens = self.query(torch.cat((evidence, y.to(evidence.dtype), points[..., 2:]/self.plane_scale), -1))
        time = self.time(time_embedding(t))
        tokens = tokens+time[:, :, None].to(tokens.dtype)+self.kind[PATH].to(tokens.dtype)
        condition = F.silu(time)  # (B, D, W): one modulation per draw, shared by its planes
        layers = [m(condition).unflatten(-1, (2, 3, self.cfg.hidden)) for m in self.time_modulation]
        output = self.output_modulation(condition).unflatten(-1, (2, self.cfg.hidden))
        if bound := self.cfg.flow_modulation_bound:  # shift/scale M*tanh(x/M), gates tanh
            soft = lambda value: bound*torch.tanh(value/bound)
            layers = [torch.cat((soft(m[..., :2, :]), m[..., 2:, :].tanh()), -2) for m in layers]
            output = soft(output)
        return self.velocity(self.run_paths(ctx, tokens, (layers, output))).float()

    def hazard_logits(self, ctx, points):
        # No generator state enters a proposal's score.
        points = points.detach()
        tokens = self.score(torch.cat((self.evidence(ctx, points), points/16.), -1))
        decoded = self.run_paths(ctx, (tokens+self.kind[SCORED].to(tokens.dtype))[:, None])[:, 0]
        return self.hazard(decoded).squeeze(-1).float()
