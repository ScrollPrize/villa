"""Frozen pieces shared by the pre-cleanup patch-encoder followers (legacy/flow_v5.py, legacy/coordinate_regression.py).

The model removed in the model cleanup, reduced to what memory-free, future-plane checkpoints use: the stride-two
patch encoder with axial RoPE attention and observed-path conditioning; the image, reference and path-geometry
context tokens; the pre-norm decoder layer reading precomputed image K/V; the causal segment survival scorer; and the
proposal selection. Parameter and buffer names are unchanged, so checkpoint EMA weights load strictly. Inference
only; nothing outside legacy/ imports this, and nothing here uses the current models' construction.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from vesuvius.models.build.pretrained_backbones.rope import RopePositionEmbedding, apply_rotary_embedding
from vesuvius.models.build.resblocks import BasicBlockD, StackedResidualBlocks
from vesuvius.neural_tracing.fiber_follow.models.path_geometry import PathGeometryTokens
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.tracing.policy import commit_prefix

# Pre-cleanup model fields every supported checkpoint holds at these values (the code implements no other).
REQUIRED = dict(memory='none', identity_dim=0, identity_objective='infonce', identity_map=False, identity_feedback=False,
                path_planes='future', tube_head=False, stem='stride2')


@dataclass
class LegacyPatchConfig:
    """Fields the shared pieces read; each follower's configuration adds its own."""
    fine: CropSpec = field(default_factory=lambda: CropSpec(depth=120, width=104, behind=48, spacing=.5))
    hidden: int = 256
    encoder_ffn: int = 256
    decoder_ffn: int = 2048
    heads: int = 4
    layers: int = 4
    decoder_layers: int = 6
    scorer_layers: int = 4
    n_future: int = 16
    future_step: float = 1.
    gate_plane: int | None = None
    n_history: int = 128
    max_recovery_distance: float = 6.
    stem_channels: int = 16
    stem_blocks: int = 1
    frame_checkpoint: str | None = None
    frame_checkpoint_sha256: str | None = None

    def __post_init__(self):
        if isinstance(self.fine, dict):
            self.fine = CropSpec(**self.fine)

    @property
    def gate_horizon(self):
        return self.n_future if self.gate_plane is None else self.gate_plane

    @property
    def recent_history_points(self):
        return self.n_history

    @property
    def token_stride(self):
        return (4, 4, 4)

    @property
    def token_offset(self):
        return (1.5, 1.5, 1.5)

    @property
    def token_shape(self):
        return tuple(math.ceil(n/s) for n, s in zip((self.fine.depth, self.fine.width, self.fine.width), self.token_stride))

    def to_dict(self):
        return asdict(self)


def check_recorded(recorded, required, kind):
    """Refuse a pre-cleanup model_cfg that uses a feature this code does not implement."""
    for key, value in required.items():
        if recorded.get(key, value) != value:
            raise ValueError(f'Not a supported legacy {kind} checkpoint: {key}={recorded.get(key)!r} (needs {value!r})')


def load_legacy(path, device, build):
    """(EMA model in eval mode, crop, n_history, volume spec, checkpoint) from ``build(checkpoint)``: the tracing CLI's
    checkpoint loader contract."""
    from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec
    from vesuvius.neural_tracing.fiber_follow.shared.paths import relocate_checkpoint
    ck = relocate_checkpoint(torch.load(path, map_location='cpu', weights_only=False))
    model = build(ck)
    model.load_state_dict(ck['ema'])
    model.to(device).eval()
    return model, model.cfg.fine, model.cfg.n_history, FiberVolumeSpec.from_dict(ck['vol_spec']), ck


# ---------------------------------------------------------------------------------------------------- geometry
def device_vector(like, values):
    return torch.stack([like.new_full((), v) for v in values])


def feature_grid(points, crop, shape, stride, offset):
    size = device_vector(points, tuple(reversed(shape)))
    scale = device_vector(points, tuple(reversed(stride)))*crop.spacing
    origin = device_vector(points, (-(crop.width-1)*crop.spacing/2, -(crop.width-1)*crop.spacing/2, -crop.behind*crop.spacing))
    origin = origin+device_vector(points, tuple(reversed(offset)))*crop.spacing
    return 2*(points-origin)/(scale*(size-1).clamp_min(1))-1


def crop_support(points, crop):
    lo = device_vector(points, (-(crop.width-1)*crop.spacing/2,)*2+(-crop.behind*crop.spacing,))
    hi = device_vector(points, ((crop.width-1)*crop.spacing/2,)*2+((crop.depth-1-crop.behind)*crop.spacing,))
    return torch.isfinite(points).all(-1) & (points >= lo).all(-1) & (points <= hi).all(-1)


def sample_features(features, points, crop, stride, offset):
    """Trilinear FP32 samples of a token lattice at crop-local points, zero (and unsupported) outside the crop."""
    supported = crop_support(points, crop)
    points = torch.where(supported[..., None], points.float(), 0.)
    grid = feature_grid(points, crop, features.shape[-3:], stride, offset)
    values = F.grid_sample(features.float(), grid[:, :, None, None], padding_mode='border', align_corners=True)
    values = values[:, :, :, 0, 0].transpose(1, 2)
    return torch.where(supported[..., None], values, 0.), supported


def token_coordinates(cfg):
    d, y, x = torch.meshgrid(*(torch.arange(n).float() for n in cfg.token_shape), indexing='ij')
    xyz = torch.stack((x, y, d), -1)
    xyz = (xyz*xyz.new_tensor(tuple(reversed(cfg.token_stride)))+xyz.new_tensor(tuple(reversed(cfg.token_offset))))*cfg.fine.spacing
    return xyz-xyz.new_tensor(((cfg.fine.width-1)*cfg.fine.spacing/2,)*2+(cfg.fine.behind*cfg.fine.spacing,))


def select_refinement(output, cfg, confidence_threshold, n_commit=None, *, retry=True):
    """Longest acceptable prefix, then confidence at its end, then earlier proposal (later ones only as retries)."""
    window = cfg.n_future if n_commit is None else n_commit
    curves, confidence = output['refinement_points'], output['refinement_confidence'].detach()
    counts, allowed = commit_prefix(curves, confidence, confidence_threshold, window, cfg.max_recovery_distance)
    accepted = (confidence[..., cfg.gate_horizon-1] >= confidence_threshold) & allowed & output['refinement_mask']
    prior_accept = torch.cat((torch.zeros_like(accepted[:, :1]), accepted[:, :-1]), 1).long().cumsum(1) > 0
    valid = output['refinement_mask'] & ~prior_accept if retry else output['refinement_mask'].bool()
    longest = counts.masked_fill(~valid, -1).max(-1, keepdim=True).values
    score = confidence.gather(-1, (counts-1).clamp_min(0)[..., None]).squeeze(-1)
    score = score.nan_to_num(nan=-torch.inf).masked_fill((counts != longest) | ~allowed | ~valid, -torch.inf)
    selected = score.argmax(-1)
    index = torch.arange(len(curves), device=curves.device)
    result = dict(output, selected_refinement=selected)
    for name in ('points', 'hazard_logits', 'confidence_logits', 'confidence'):
        result[name] = output['refinement_'+name][index, selected]
    return result


# ---------------------------------------------------------------------------------------------------- encoder
class AxialRoPE3D(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rotary_dim = 6*(head_dim//6)
        self.embedding = RopePositionEmbedding(self.rotary_dim, ndim=3, base=100., normalize_coords='max',
                                               shift_coords=None, jitter_coords=None, rescale_coords=None,
                                               dtype=torch.float32)

    def forward(self, q, k, spatial_shape, axis):
        coordinate = 2*(torch.arange(spatial_shape[axis], device=q.device, dtype=self.embedding.periods.dtype)+.5)/max(spatial_shape)-1
        zero = torch.zeros_like(coordinate)
        embedding = self.embedding.get_embed_from_coords(torch.stack([coordinate if d == axis else zero for d in range(3)], -1))
        rotated = [torch.cat((apply_rotary_embedding(v[..., :self.rotary_dim], embedding), v[..., self.rotary_dim:]), -1)
                   for v in (q, k)]
        return rotated[0], rotated[1]


class AxisAttention(nn.Module):
    def __init__(self, width, heads, axis):
        super().__init__()
        self.heads, self.axis = heads, axis
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.projection = nn.Linear(width, width)
        self.rotary = AxialRoPE3D(width//heads)

    def forward(self, x):
        moved = x.movedim(self.axis, -2)
        shape = moved.shape
        seq = self.norm(moved).reshape(-1, shape[-2], shape[-1])
        q, k, v = self.qkv(seq).reshape(seq.shape[0], seq.shape[1], 3, self.heads, shape[-1]//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rotary(q, k, x.shape[1:4], self.axis-1)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False).transpose(1, 2).reshape_as(seq)
        return x+self.projection(attended).reshape(shape).movedim(-2, self.axis)


class AxialBlock(nn.Module):
    def __init__(self, width, heads, ffn):
        super().__init__()
        self.axes = nn.ModuleList(AxisAttention(width, heads, a) for a in (3, 2, 1))
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width))

    def forward(self, x):
        for axis in self.axes:
            x = axis(x)
        return x+self.mlp(self.norm(x))


class LightPatchStem(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.stem_channels
        self.input = nn.Sequential(nn.Conv3d(1, c, 3, stride=2, padding=1), nn.InstanceNorm3d(c, eps=1e-5, affine=True),
                                   nn.ReLU(inplace=True))
        self.blocks = StackedResidualBlocks(n_blocks=cfg.stem_blocks, input_channels=c, output_channels=2*c,
            initial_stride=2, conv_bias=False, conv_op=nn.Conv3d, kernel_size=3, norm_op=nn.InstanceNorm3d,
            norm_op_kwargs=dict(eps=1e-5, affine=True), nonlin=nn.ReLU, nonlin_kwargs=dict(inplace=True), block=BasicBlockD)
        self.projection = nn.Conv3d(2*c, cfg.hidden, 1)

    def forward(self, image):
        return self.projection(self.blocks(self.input(image)))


class PatchEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_projection = nn.Conv3d(1, cfg.hidden, 6, stride=4, padding=1)
        self.stem = LightPatchStem(cfg)
        self.position = nn.Linear(3, cfg.hidden)
        self.condition = nn.Linear(3, cfg.hidden, bias=False)
        self.blocks = nn.ModuleList(AxialBlock(cfg.hidden, cfg.heads, cfg.encoder_ffn) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.hidden)
        self.register_buffer('token_xyz', token_coordinates(cfg).reshape(-1, 3), persistent=False)

    def forward(self, image, references, mask):
        tokens = (self.patch_projection(image)+self.stem(image)).permute(0, 2, 3, 4, 1)
        tokens = tokens+self.position(self.token_xyz/16).reshape(*self.cfg.token_shape, self.cfg.hidden).to(tokens.dtype)
        tokens = tokens+self.condition(self.conditioning(references, mask)).to(tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens).permute(0, 4, 1, 2, 3)

    def conditioning(self, references, mask):
        cfg = self.cfg
        points = torch.where(mask[..., None], references, 0.).float()
        origin = device_vector(points, (-(cfg.fine.width-1)*cfg.fine.spacing/2,)*2+(-cfg.fine.behind*cfg.fine.spacing,))
        offset = device_vector(points, tuple(reversed(cfg.token_offset)))*cfg.fine.spacing
        index = torch.round((points-origin-offset)/(device_vector(points, tuple(reversed(cfg.token_stride)))*cfg.fine.spacing)).long()
        d, y, x = cfg.token_shape
        index = torch.stack((index[..., 0].clamp(0, x-1), index[..., 1].clamp(0, y-1), index[..., 2].clamp(0, d-1)), -1)
        flat = index[..., 0]+x*(index[..., 1]+y*index[..., 2])
        ages = torch.arange(1, cfg.n_history+2, device=points.device).float()/cfg.n_history
        history = mask.clone()
        history[:, -1] = False
        seed = mask & ~history
        values = torch.stack((history.float(), history*ages[None], seed.float()), -1)
        rendered = points.new_zeros(len(points), d*y*x, 3).scatter_add(1, flat[..., None].expand(-1, -1, 3), values)
        count = rendered[..., :1]
        rendered = torch.cat((count.clamp_max(1), rendered[..., 1:2]/count.clamp_min(1), rendered[..., 2:].clamp_max(1)), -1)
        return rendered.reshape(len(points), d, y, x, 3)


# ---------------------------------------------------------------------------------------------------- decoder
class PathDecoderLayer(nn.TransformerDecoderLayer):
    """Pre-norm decoder layer reading already projected image K/V (no memory attention)."""
    def attention_backend(self, tensor):
        return (sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                             SDPBackend.MATH], set_priority=True) if tensor.is_cuda else nullcontext())

    def project_memory(self, memory):
        attn = self.multihead_attn
        width, heads = attn.embed_dim, attn.num_heads
        kv = F.linear(memory, attn.in_proj_weight[width:], attn.in_proj_bias[width:])
        return tuple(v.reshape(len(memory), -1, heads, width//heads).transpose(1, 2).contiguous() for v in kv.chunk(2, dim=-1))

    def attend_memory(self, q, kv, padding):
        mask = torch.zeros(padding.shape, device=q.device, dtype=q.dtype).masked_fill(padding, -torch.inf)
        with self.attention_backend(q):
            return F.scaled_dot_product_attention(q, *kv, attn_mask=mask[:, None, None, :])

    def cross_attention_value(self, normed, kv, padding):
        attn = self.multihead_attn
        h, heads = attn.embed_dim, attn.num_heads
        q = F.linear(normed, attn.in_proj_weight[:h], attn.in_proj_bias[:h]).reshape(len(normed), -1, heads, h//heads).transpose(1, 2).contiguous()
        value = self.attend_memory(q, kv, padding).transpose(1, 2).contiguous().reshape(len(normed), -1, h)
        return attn.out_proj(value)

    def forward_cached(self, x, kv, padding, causal_mask=None):
        x = x+self._sa_block(self.norm1(x), causal_mask, None, is_causal=causal_mask is not None)
        x = x+self.dropout2(self.cross_attention_value(self.norm2(x), kv, padding))
        return x+self._ff_block(self.norm3(x))

    def forward_draws(self, query, kv, padding, self_padding, modulation=None):
        """Independent self-attention draws (B, D, P, W) sharing one decision's image K/V; ``modulation``
        (B, D, 4, 3, W) adaLN shift/scale/gate per branch (the memory branch, index 2, is absent)."""
        b, draws, planes, width = query.shape
        if modulation is None:
            x = query.reshape(b*draws, planes, width)
            x = x+self._sa_block(self.norm1(x), None, self_padding, is_causal=False)
            x = x.reshape(b, draws*planes, width)
            x = x+self.dropout2(self.cross_attention_value(self.norm2(x), kv, padding))
            x = x+self._ff_block(self.norm3(x))
            return x.reshape(b, draws, planes, width)
        shift, scale, gate = modulation[:, :, None].unbind(-2)
        branch_input = lambda x, norm, i: norm(x)*(1+scale[..., i, :])+shift[..., i, :]
        residual = lambda x, value, i: x+(1+gate[..., i, :])*value.reshape(b, draws, planes, width)
        x = residual(query, self._sa_block(branch_input(query, self.norm1, 0).reshape(b*draws, planes, width),
                                           None, self_padding, is_causal=False), 0)
        x = residual(x, self.dropout2(self.cross_attention_value(
            branch_input(x, self.norm2, 1).reshape(b, draws*planes, width), kv, padding)), 1)
        return residual(x, self._ff_block(branch_input(x, self.norm3, 3)), 3)


def decoder_layer(cfg, ffn):
    return PathDecoderLayer(cfg.hidden, cfg.heads, ffn, dropout=0., activation='gelu', batch_first=True, norm_first=True)


class SegmentSurvivalScorer(nn.Module):
    samples_per_segment = 4

    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        width = self.samples_per_segment*(h+1)+10
        self.query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, h), nn.SiLU())
        self.layers = nn.ModuleList(decoder_layer(cfg, 1024) for _ in range(cfg.scorer_layers))
        self.norm = nn.LayerNorm(h)
        self.failure = nn.Linear(h, 1)

    def segment_samples(self, points):
        start = torch.cat((torch.zeros_like(points[:, :1]), points[:, :-1]), 1)
        fraction = torch.arange(1, self.samples_per_segment+1, device=points.device, dtype=points.dtype)/self.samples_per_segment
        return start[:, :, None]+fraction[None, None, :, None]*(points-start)[:, :, None]

    def forward(self, spatial, points, projected, padding):
        start = torch.cat((torch.zeros_like(points[:, :1]), points[:, :-1]), 1)
        delta = points-start
        geometry = torch.cat((start, points, delta, delta.norm(dim=-1, keepdim=True)), -1)/16.
        query = self.query(torch.cat((spatial.flatten(2), geometry), -1))
        k = points.shape[1]
        causal = torch.ones(k, k, device=points.device, dtype=torch.bool).triu(1)
        for layer, kv in zip(self.layers, projected):
            query = layer.forward_cached(query, kv, padding, causal)
        with torch.autocast(query.device.type, enabled=False):
            return self.failure(self.norm(query.float())).squeeze(-1)


# ---------------------------------------------------------------------------------------------------- follower
class LegacyPatchFollower(nn.Module):
    """Observation context and segment scoring shared by the legacy followers; subclasses add their generator."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg, h = cfg, cfg.hidden
        self.model_type = cfg.model_type
        self.encoder = PatchEncoder(cfg)
        self.reference_token = nn.Sequential(nn.Linear(h+8, h), nn.SiLU(), nn.Linear(h, h))
        self.confidence_scorer = SegmentSurvivalScorer(cfg)
        self.path_geometry = PathGeometryTokens(cfg)

    def sample_local(self, features, points):
        return sample_features(features, points, self.cfg.fine, self.cfg.token_stride, self.cfg.token_offset)

    def context(self, x, hist, hmask):
        """Encoder tokens, the decoder/scorer memory (image, reference and path-geometry tokens) with its padding, and
        the scorer's projected K/V."""
        cfg = self.cfg
        seed = x.get('seed', hist.new_zeros(len(hist), 1, 3))
        seed_mask = x.get('seed_mask', hmask.new_zeros(len(hist), 1)).bool()
        references = torch.cat((hist, seed), 1)
        mask = torch.cat((hmask.bool(), seed_mask), 1) & crop_support(references, cfg.fine)
        references = torch.where(mask[..., None], references.float(), 0.)
        deep = self.encoder(x['fine'], references, mask)
        image_tokens = deep.flatten(2).transpose(1, 2)
        local = self.sample_local(deep, references)[0].to(deep.dtype)
        ages = torch.arange(1, cfg.n_history+2, device=hist.device).float()[None].expand(len(hist), -1).clone()
        ages[:, -1] = x.get('seed_age', hist.new_zeros(len(hist))).reshape(-1)
        anchor = torch.zeros_like(ages)
        anchor[:, -1] = 1.
        tangent = torch.zeros_like(references)
        tangent[:, -1] = x.get('seed_tangent', hist.new_zeros(len(hist), 3))
        metadata = torch.cat((references/16, tangent, torch.log1p(ages.clamp(0, 2048))[..., None]/math.log(2049), anchor[..., None]), -1)
        metadata = torch.where(mask[..., None], metadata, 0.)
        ref_tokens = self.reference_token(torch.cat((local, metadata), -1))
        valid = x['path_geometry_valid'].bool()
        memory = torch.cat((image_tokens, ref_tokens.to(image_tokens.dtype),
                            self.path_geometry(x['path_geometry'], valid).to(image_tokens.dtype)), 1)
        padding = torch.cat((torch.zeros(image_tokens.shape[:2], device=hist.device, dtype=torch.bool), ~mask, ~valid), 1)
        return dict(deep=deep, memory=memory, padding=padding,
                    confidence_projected=[layer.project_memory(memory) for layer in self.confidence_scorer.layers])

    def evidence(self, ctx, points):
        """Sampled encoder tokens and crop support at crop-local points (B, M, hidden+1)."""
        values, support = self.sample_local(ctx['deep'], points)
        return torch.cat((values.to(ctx['deep'].dtype), support[..., None]), -1)

    def hazard_logits(self, ctx, points):
        points = points.detach()
        samples = self.confidence_scorer.segment_samples(points)
        spatial = self.evidence(ctx, samples.flatten(1, 2)).reshape(*samples.shape[:3], -1)
        return self.confidence_scorer(spatial, points, ctx['confidence_projected'], ctx['padding'])
