"""Historical-slab fiber regression with continuous paths and causal survival scoring."""
from dataclasses import asdict, dataclass, field
from contextlib import nullcontext
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.tracing.policy import DEFAULT_CONFIDENCE, commit_prefix



@dataclass
class CoordinateRegressionConfig:
    model_type: str = "coordinate_regression"
    fine: CropSpec = field(default_factory=lambda: CropSpec(depth=120, width=104, behind=48, spacing=.5))
    hidden: int = 256
    encoder_ffn: int = 256
    decoder_ffn: int = 2048
    heads: int = 4
    layers: int = 2
    decoder_layers: int = 6
    scorer_layers: int = 4
    activation_checkpointing: bool = False
    n_future: int = 16
    future_step: float = 1.
    gate_plane: int | None = None  # plane whose confidence accepts a proposal (full gate, retries); None: the last plane
    query_scale: float | None = None  # forward-distance normalization of decoder queries; None: n_future*future_step
    n_history: int = 128
    max_recovery_distance: float = 6.
    patch_radius: float = 1.
    recurrent_refinement_steps: int = 3
    stem_channels: int = 32
    stem_blocks: int = 2
    stem: str = 'residual'  # 'residual' full-resolution stem, or 'stride2' light stem
    # 'slabs' separate historical-slab CNN, 'decisions' encoder features of past decisions, or 'none' (no memory:
    # no memory tokens, memory attention or memory inputs; the current crop and observed path only)
    memory: str = 'slabs'
    identity_dim: int = 0  # >0 (decision memory): linear projection for the memory-identity InfoNCE loss
    identity_temperature: float = .1
    # 'infonce' (with identity_dim), 'verify': a memory-conditioned on-fiber verifier (identity_verifier.py),
    # or 'readout': path-blind appearance-key memory read out per location (identity_readout.py).
    identity_objective: str = 'infonce'
    identity_map: bool = False  # verify/readout: the dense identity field enters the decoder/scorer image tokens
    identity_feedback: bool = False  # verify/readout: field samples enter scorer segment queries and retry fusion
    # 'future': the path is planes 1..n_future; 'crop': one point per crop plane behind and ahead of the head
    # (whole_crop.py), supervised on the original fiber; planes 1..n_future remain the committed proposal.
    path_planes: str = 'future'
    tube_head: bool = False  # dense Gaussian-tube head around the original fiber (whole_crop.py)
    frame_checkpoint: str | None = None  # frozen heading/normal model used by crop builders
    frame_checkpoint_sha256: str | None = None

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in
               (self.encoder_ffn, self.decoder_ffn)):
            raise ValueError('Feed-forward widths must be positive integers')
        if type(self.identity_dim) is not int or self.identity_dim < 0 or not self.identity_temperature > 0:
            raise ValueError('Identity projection width must be a nonnegative integer and its temperature positive')
        if self.identity_dim and self.memory != 'decisions':
            raise ValueError('The memory-identity loss requires decision memory')
        if self.identity_objective not in ('infonce', 'verify', 'readout'):
            raise ValueError("Identity objective must be 'infonce', 'verify' or 'readout'")
        if self.identity_objective in ('verify', 'readout') and (self.memory != 'decisions' or self.identity_dim):
            raise ValueError('Identity verification requires decision memory and no InfoNCE projection')
        if (self.identity_map or self.identity_feedback) and self.identity_objective not in ('verify', 'readout'):
            raise ValueError('The identity field requires the verification or readout objective')
        if self.stem not in ('residual', 'stride2') or self.memory not in ('slabs', 'decisions', 'none'):
            raise ValueError("Stem must be 'residual' or 'stride2'; memory must be 'slabs', 'decisions' or 'none'")
        if type(self.stem_channels) is not int or self.stem_channels < 1 or type(self.stem_blocks) is not int or self.stem_blocks < 1:
            raise ValueError('Stem channels must be a positive integer and stem blocks a positive integer')
        if isinstance(self.fine, dict):
            self.fine = CropSpec(**self.fine)
        c = self.fine
        if c.depth % 4 or c.width % 4:
            raise ValueError('Patch4 crop dimensions must be multiples of four')
        if min(c.depth, c.width) < 8 or not 0 <= c.behind < c.depth or not math.isfinite(c.spacing) or c.spacing <= 0:
            raise ValueError('Invalid fine crop')
        if type(self.scorer_layers) is not int or self.scorer_layers < 1:
            raise ValueError('Scorer layers must be a positive integer')
        if min(self.hidden, self.heads, self.layers, self.decoder_layers,
               self.n_future, self.n_history) < 1 or self.hidden % self.heads:
            raise ValueError('Positive dimensions required; hidden must divide by heads')
        if not 0 < self.future_step <= self.max_recovery_distance or not math.isfinite(self.max_recovery_distance):
            raise ValueError('Invalid forward spacing or connection limit')
        if not 0 < self.patch_radius < (c.width-1)*c.spacing/2:
            raise ValueError('Local observation patch must fit fine crop')
        if self.n_future*self.future_step > (c.depth-c.behind-1)*c.spacing:
            raise ValueError('Future horizon exceeds fine image')
        if self.gate_plane is not None and (type(self.gate_plane) is not int or not 1 <= self.gate_plane <= self.n_future):
            raise ValueError('Gate plane must be an integer in [1, n_future]')
        if self.query_scale is not None and not (math.isfinite(self.query_scale) and self.query_scale > 0):
            raise ValueError('Query scale must be finite and positive')
        if not isinstance(self.recurrent_refinement_steps, int) or self.recurrent_refinement_steps < 0:
            raise ValueError('Recurrent refinement steps must be a nonnegative integer')
        if self.path_planes not in ('future', 'crop'):
            raise ValueError("Path planes must be 'future' or 'crop'")
        if self.tube_head and self.model_type != 'coordinate_regression':
            raise ValueError('The tube head is implemented for coordinate regression only')

    @property
    def gate_horizon(self):
        """Planes whose confidence decides acceptance (full gate) and retries; the rest are predicted and supervised."""
        return self.n_future if self.gate_plane is None else self.gate_plane

    @property
    def path_plane_values(self):
        """Forward coordinates of the decoder's path planes."""
        import numpy as np
        if self.path_planes == 'future':
            return self.future_step*np.arange(1, self.n_future+1, dtype=np.float64)
        from vesuvius.neural_tracing.fiber_follow.shared.geometry import crop_path_planes
        return crop_path_planes(self.fine, self.future_step)

    @property
    def proposal_slice(self):
        """Path-plane indices of the committed proposal (planes 1..n_future)."""
        start = int(round(float(-self.path_plane_values[0])/self.future_step))+1 if self.path_planes == 'crop' else 0
        return slice(start, start+self.n_future)

    @property
    def input_channels(self):
        return 1

    @property
    def recent_history_points(self):
        return self.n_history

    @property
    def lateral_limit(self):
        return (self.fine.width-1)*self.fine.spacing/2-self.patch_radius

    @property
    def path_evidence_width(self):
        return self.hidden+1

    @property
    def token_stride(self):
        return (4, 4, 4)

    @property
    def token_offset(self):
        return (1.5, 1.5, 1.5)

    @property
    def token_shape(self):
        return tuple(math.ceil(n/s) for n,s in zip((self.fine.depth,self.fine.width,self.fine.width),self.token_stride))

    @property
    def identity_mode(self):
        """Which memory-identity training targets this model consumes, if any (readout uses the verifier's)."""
        if self.identity_objective in ('verify', 'readout'):
            return 'verify'
        return 'infonce' if self.identity_dim else None

    @property
    def memory_entry_shape(self):
        """(rows, width) of one decision-memory entry: encoder samples, then packed readout keys."""
        from .decision_memory import ENTRY_FEATURES
        rows = ENTRY_FEATURES
        if self.identity_objective == 'readout':
            from .identity_readout import key_rows
            rows += key_rows(self.hidden)
        return rows, self.hidden

    def to_dict(self):
        return asdict(self)


# Readout identity memory passed alongside precomputed history tokens (training, see train.training_prediction).
READOUT_MEMORY = ('identity_memory_keys', 'identity_memory_padding', 'identity_memory_role')


def build_model(cfg):
    if cfg.model_type == 'flow_matching':
        from .flow import FlowFollower
        return FlowFollower(cfg)
    if cfg.model_type == 'sequence':
        from .sequence import SequenceFollower
        return SequenceFollower(cfg)
    if cfg.model_type != 'coordinate_regression':
        raise ValueError(f'Unsupported model type: {cfg.model_type}')
    return CoordinateRegressionFollower(cfg)


def select_refinement(output, cfg, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None, *, retry=True):
    """Longest acceptable prefix, then confidence at its end, then earlier pass.

    If every proposal stops, select the best first-point confidence among valid
    connections; the unchanged commit gate still stops. Selection never splices
    paths or transfers one proposal's confidence to another proposal.
    With ``retry`` (default) proposals after the first accepted one are ignored, so later
    proposals act only as retries; without it the same ranking runs over all proposals.
    """
    if not 0 <= confidence_threshold <= 1:
        raise ValueError('Confidence threshold must lie in [0, 1]')
    window = cfg.n_future if n_commit is None else n_commit
    curves, confidence = output['refinement_points'], output['refinement_confidence'].detach()
    counts, allowed = commit_prefix(curves, confidence, confidence_threshold, window, cfg.max_recovery_distance)
    gate = getattr(cfg, 'gate_horizon', confidence.shape[-1])  # acceptance through the gate plane, as in tracing
    accepted = (confidence[..., gate-1] >= confidence_threshold) & allowed & output['refinement_mask']
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
    if 'refinement_crop_points' in output:
        result['crop_points'] = output['refinement_crop_points'][index, selected]
    return result


def select_history(projected, keep):
    """Active rows of projected memory K/V (None without memory)."""
    return None if projected is None else tuple(v[keep] for v in projected)


def device_vector(like, values):
    """``like.new_tensor(values)`` without a host-to-device copy, which would
    synchronize with the queued GPU work (these run in eager losses and tracing)."""
    return torch.stack([like.new_full((), v) for v in values])


def feature_grid(points, crop, shape, stride=1, offset=(0,0,0)):
    """Exact physical coordinates for a feature lattice (stride/offset in zyx)."""
    if isinstance(stride, (int, float)):
        stride = (stride,)*3
    size = device_vector(points, tuple(reversed(shape)))
    scale = device_vector(points, tuple(reversed(stride)))*crop.spacing
    origin = device_vector(points, (-(crop.width-1)*crop.spacing/2,
                                    -(crop.width-1)*crop.spacing/2, -crop.behind*crop.spacing))
    origin = origin+device_vector(points, tuple(reversed(offset)))*crop.spacing
    return 2*(points-origin)/(scale*(size-1).clamp_min(1))-1


def crop_support(points, crop):
    """Input support, independent of the coarser contextual-feature lattice."""
    lo = device_vector(points, (-(crop.width-1)*crop.spacing/2,)*2+(-crop.behind*crop.spacing,))
    hi = device_vector(points, ((crop.width-1)*crop.spacing/2,)*2+((crop.depth-1-crop.behind)*crop.spacing,))
    return torch.isfinite(points).all(-1) & (points >= lo).all(-1) & (points <= hi).all(-1)


def sample_features(features, points, crop, stride=1, offset=(0,0,0)):
    supported = crop_support(points, crop)
    points = torch.where(supported[...,None], points.float(), 0.)
    grid = feature_grid(points, crop, features.shape[-3:], stride, offset)
    # Border extrapolation covers the half-token margins of the input crop.
    values = F.grid_sample(features.float(), grid[:,:,None,None], padding_mode='border', align_corners=True)
    values = values[:,:,:,0,0].transpose(1,2)
    return torch.where(supported[...,None],values,0.), supported




class AxisAttention(nn.Module):
    """Unmasked attention along one whole spatial axis of a BDHWC tensor."""
    def __init__(self, width, heads, axis, *, rotary=False):
        super().__init__()
        self.heads, self.axis = heads, axis
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width,3*width)
        self.projection = nn.Linear(width,width)
        from vesuvius.neural_tracing.fiber_follow.models.rope import AxialRoPE3D
        self.rotary = AxialRoPE3D(width//heads) if rotary else None

    def forward(self, x):
        moved = x.movedim(self.axis,-2)
        shape = moved.shape
        seq = self.norm(moved).reshape(-1,shape[-2],shape[-1])
        qkv = self.qkv(seq).reshape(seq.shape[0],seq.shape[1],3,self.heads,shape[-1]//self.heads)
        q,k,v = qkv.permute(2,0,3,1,4).unbind(0)
        if self.rotary is not None:
            q, k = self.rotary(q, k, x.shape[1:4], self.axis-1)
        attended = F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False)
        attended = attended.transpose(1,2).reshape_as(seq)
        return x+self.projection(attended).reshape(shape).movedim(-2,self.axis)




def project_attention_memory(attn, memory):
    """Differentiable K/V with the parameter layout of MultiheadAttention."""
    width, heads = attn.embed_dim, attn.num_heads
    kv = F.linear(memory, attn.in_proj_weight[width:], attn.in_proj_bias[width:])
    return tuple(value.reshape(len(memory), -1, heads, width//heads).transpose(1, 2).contiguous()
                 for value in kv.chunk(2, dim=-1))


class PathDecoderLayer(nn.TransformerDecoderLayer):
    """Prefer cuDNN for short queries over long, padding-masked image memory."""
    def attention_backend(self, tensor):
        return (sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                             SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True)
                if tensor.is_cuda else nullcontext())

    def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal=False):
        with self.attention_backend(x):
            return super()._mha_block(x, mem, attn_mask, key_padding_mask, is_causal)

    def project_memory(self, memory):
        """Differentiable per-decision K/V; never retained across model forwards."""
        return project_attention_memory(self.multihead_attn, memory)

    @staticmethod
    def select_memory(projected, keep):
        """Select active inference rows from per-decision K/V."""
        return [tuple(v[keep] for v in kv) for kv in projected]

    def attend_memory(self, q, kv, padding):
        """Batched SDPA with a broadcast key mask and fixed-size K/V.

        cuDNN handles the short-query/long-memory case without per-row gathers
        or data-dependent key counts. Keep the mask compact, never BxHxQxK.
        """
        mask = torch.zeros(padding.shape, device=q.device, dtype=q.dtype).masked_fill(padding, -torch.inf)
        with self.attention_backend(q):
            return F.scaled_dot_product_attention(q, *kv, attn_mask=mask[:, None, None, :])

    def forward_cached(self, x, kv, padding, *, causal_mask=None, history=None, history_attention=None):
        """The ordinary pre-norm decoder computation with already projected K/V."""
        if not self.norm_first or self.multihead_attn.dropout:
            raise ValueError('Cached trajectory decoder requires pre-norm and zero attention dropout')
        x = x+self._sa_block(self.norm1(x), causal_mask, None, is_causal=causal_mask is not None)
        x = self.cross_attention(x, kv, padding)
        if history is not None:
            x = history_attention.forward_cached(x, *history)
        return x+self._ff_block(self.norm3(x))

    def cross_attention(self, x, kv, padding):
        return x+self.dropout2(self.cross_attention_value(self.norm2(x), kv, padding))

    def cross_attention_value(self, normed, kv, padding):
        """Image attention output for already normalized queries (the residual branch)."""
        attn = self.multihead_attn
        h, heads = attn.embed_dim, attn.num_heads
        q = F.linear(normed, attn.in_proj_weight[:h], attn.in_proj_bias[:h])
        q = q.reshape(len(normed), -1, heads, h//heads).transpose(1, 2).contiguous()
        value = self.attend_memory(q, kv, padding)
        value = value.transpose(1, 2).contiguous().reshape(len(normed), -1, h)
        return attn.out_proj(value)

    def forward_draws(self, query, kv, padding, self_padding, *, history, history_attention, modulation=None):
        """Independent self-attention draws sharing differentiable observation K/V.

        ``modulation`` (B, D, 4, 3, H) conditions each pre-norm branch (self-attention, image attention,
        history attention, FFN) per draw: normalized input * (1+scale) + shift, branch output * (1+gate).
        Zero modulation is exactly the unmodulated layer.
        """
        b, draws, planes, width = query.shape
        if modulation is None:
            x = query.reshape(b*draws, planes, width)
            x = x+self._sa_block(self.norm1(x), None, self_padding, is_causal=False)
            x = x.reshape(b, draws*planes, width)
            x = self.cross_attention(x, kv, padding)
            if history is not None:
                x = history_attention.forward_cached(x, *history)
            x = x+self._ff_block(self.norm3(x))
            return x.reshape(b, draws, planes, width)
        shift, scale, gate = modulation[:, :, None].unbind(-2)  # each (B, D, 1, 4, H)
        branch_input = lambda x, norm, i: norm(x)*(1+scale[..., i, :])+shift[..., i, :]
        residual = lambda x, value, i: x+(1+gate[..., i, :])*value.reshape(b, draws, planes, width)
        x = query
        x = residual(x, self._sa_block(branch_input(x, self.norm1, 0).reshape(b*draws, planes, width),
                                       None, self_padding, is_causal=False), 0)
        x = residual(x, self.dropout2(self.cross_attention_value(
            branch_input(x, self.norm2, 1).reshape(b, draws*planes, width), kv, padding)), 1)
        if history is not None:
            x = residual(x, history_attention.attend(
                branch_input(x, history_attention.norm, 2).reshape(b, draws*planes, width), *history), 2)
        return residual(x, self._ff_block(branch_input(x, self.norm3, 3)), 3)


class AxialBlock(nn.Module):
    def __init__(self, width, heads, *, ffn=1024, rotary=True):
        super().__init__()
        self.axes = nn.ModuleList(AxisAttention(width,heads,a,rotary=rotary) for a in (3,2,1))
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width,ffn),nn.GELU(),nn.Linear(ffn,width))


    def forward(self, x):
        for axis in self.axes:
            x = axis(x)
        x = x+self.mlp(self.norm(x))
        return x


def token_coordinates(cfg):
    """Token centers in crop-local XYZ."""
    d,y,x = torch.meshgrid(*(torch.arange(n).float() for n in cfg.token_shape),indexing='ij')
    xyz = torch.stack((x,y,d),-1)
    xyz = (xyz*xyz.new_tensor(tuple(reversed(cfg.token_stride)))
           +xyz.new_tensor(tuple(reversed(cfg.token_offset))))*cfg.fine.spacing
    return xyz-xyz.new_tensor(((cfg.fine.width-1)*cfg.fine.spacing/2,)*2+(cfg.fine.behind*cfg.fine.spacing,))






class ObservationFollower(nn.Module):
    """Shared observations, history, spatial evidence and survival scoring."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_type = cfg.model_type
        from .patch_encoder import PatchEncoder
        from .history_slabs import HistoryEncoder, HistoryAttention
        from .survival_confidence import SegmentSurvivalScorer
        from .path_geometry import PathGeometryTokens
        self.encoder = PatchEncoder(cfg)
        self.reference_token = nn.Sequential(nn.Linear(cfg.hidden+8,cfg.hidden),nn.SiLU(),nn.Linear(cfg.hidden,cfg.hidden))
        if cfg.memory == 'decisions':
            from .decision_memory import DecisionMemory
            self.history_encoder = DecisionMemory(cfg)
            # Training-only head of the memory-identity loss; tracing never uses it.
            self.identity_projection = nn.Linear(cfg.hidden, cfg.identity_dim) if cfg.identity_dim else None
            if cfg.identity_objective == 'verify':
                from .identity_verifier import IdentityVerifier
                self.identity_verifier = IdentityVerifier(cfg)
            if cfg.identity_objective == 'readout':
                from .identity_readout import IdentityReadout, FEATURE_DIM
                self.identity_readout = IdentityReadout(cfg)
                if cfg.identity_map:
                    # Readout features and logit per token; zero output keeps a warm start exact.
                    self.identity_readout_embedding = nn.Sequential(nn.LayerNorm(FEATURE_DIM+1),
                                                                    nn.Linear(FEATURE_DIM+1, cfg.hidden))
                    nn.init.zeros_(self.identity_readout_embedding[-1].weight)
                    nn.init.zeros_(self.identity_readout_embedding[-1].bias)
            elif cfg.identity_map:
                # Zero output: a warm-started model initially computes what it did without the field.
                self.identity_embedding = nn.Sequential(nn.Linear(1, cfg.hidden), nn.SiLU(),
                                                        nn.Linear(cfg.hidden, cfg.hidden))
                nn.init.zeros_(self.identity_embedding[-1].weight)
                nn.init.zeros_(self.identity_embedding[-1].bias)
        elif cfg.memory == 'slabs':
            self.history_encoder = HistoryEncoder(cfg)
        if cfg.memory != 'none':
            self.history_attention = HistoryAttention(cfg.hidden, cfg.heads)
        self.confidence_scorer = SegmentSurvivalScorer(cfg)
        self.path_geometry = PathGeometryTokens(cfg)
        self.register_buffer('planes', torch.arange(1,cfg.n_future+1).float()*cfg.future_step, persistent=False)

    def references(self, x, hist, hmask):
        cfg = self.cfg
        seed = x.get('seed',hist.new_zeros(len(hist),1,3))
        seed_mask = x.get('seed_mask',hmask.new_zeros(len(hist),1)).bool()
        references = torch.cat((hist,seed),1)
        mask = torch.cat((hmask.bool(),seed_mask),1) & crop_support(references,cfg.fine)
        references = torch.where(mask[...,None],references.float(),0.)
        return references, mask

    def context_from_features(self, x, hist, references, mask, dense, deep, identity=None, identity_tokens=None):
        cfg = self.cfg
        image_tokens = deep.flatten(2).transpose(1,2)
        if identity_tokens is not None:
            image_tokens = image_tokens+identity_tokens.to(image_tokens.dtype)
        elif identity is not None and cfg.identity_map:
            embedding = self.identity_embedding(identity.flatten(2).transpose(1, 2).to(image_tokens.dtype))
            image_tokens = image_tokens+embedding.to(image_tokens.dtype)
        sampling_dense = dense.float() if cfg.recurrent_refinement_steps else dense
        local,_ = self.sample_local(sampling_dense, references)
        local = local.to(dense.dtype)
        ages = torch.arange(1,cfg.n_history+2,device=hist.device).float()[None].expand(len(hist),-1).clone()
        ages[:,-1] = x.get('seed_age',hist.new_zeros(len(hist))).reshape(-1)
        anchor = torch.zeros_like(ages)
        anchor[:,-1] = 1.
        tangent = torch.zeros_like(references)
        tangent[:,-1] = x.get('seed_tangent',hist.new_zeros(len(hist),3))
        metadata = torch.cat((references/16,tangent,torch.log1p(ages.clamp(0,2048))[...,None]/math.log(2049),anchor[...,None]),-1)
        metadata = torch.where(mask[...,None],metadata,0.)
        ref_tokens = self.reference_token(torch.cat((local,metadata),-1))
        memory = torch.cat((image_tokens,ref_tokens.to(image_tokens.dtype)),1)
        padding = torch.cat((torch.zeros(image_tokens.shape[:2],device=hist.device,dtype=torch.bool),~mask),1)
        valid = x['path_geometry_valid'].bool()
        geometry = self.path_geometry(x['path_geometry'], valid)
        memory = torch.cat((memory,geometry.to(memory.dtype)),1)
        padding = torch.cat((padding,~valid),1)
        ctx = dict(fine=dense,deep=deep,memory=memory,padding=padding,reference_mask=mask)
        if cfg.recurrent_refinement_steps:
            ctx.update(fine_fp32=sampling_dense, deep_fp32=sampling_dense)
        if identity is not None:
            ctx['identity_field'] = identity
        return ctx

    def identity_samples(self, ctx, points):
        """Identity-field probability at crop-local points (zero outside the crop)."""
        return self.sample_local(ctx['identity_field'], points)[0][..., 0]

    def sample_local(self, features, points):
        cfg = self.cfg
        return sample_features(features, points, cfg.fine,
                               cfg.token_stride,
                               cfg.token_offset)

    def patches(self, fine, points):
        return self.sample_local(fine, points)[0]

    def query_features(self, ctx, initial):
        patches = self.patches(ctx.get('fine_fp32', ctx['fine']),initial).to(ctx['fine'].dtype)
        scale = getattr(self, 'query_scale', self.cfg.n_future*self.cfg.future_step)
        return torch.cat((patches,initial[...,2:]/scale),-1)

    def evidence(self, ctx, points):
        values, support = self.sample_local(ctx.get('deep_fp32', ctx['deep']), points)
        return torch.cat((values.to(ctx['deep'].dtype), support[..., None]), -1)

    def encode_history(self, x, return_anchors=False, return_identity=False):
        """Memory tokens and padding; with ``return_anchors`` also each entry's feature at its own head,
        with ``return_identity`` also the readout identity memory (a dict, empty for other objectives)."""
        if self.cfg.memory == 'none':
            out = (None, None, None) if return_anchors else (None, None)
            return (*out, {}) if return_identity else out
        extra = {key: x['history_'+key] for key in ('path_points', 'path_tangents', 'path_valid')}
        if self.cfg.memory == 'slabs':
            out = self.history_encoder(x['history_slabs'], x['history_valid'], x['history_pose'], **extra)
            return (*out, {}) if return_identity else out
        from .decision_memory import ANCHOR_FEATURE, ENTRY_FEATURES
        features = self.memory_features(x)
        tokens, padding = self.history_encoder(features[:, :, :ENTRY_FEATURES], x['history_valid'], x['history_pose'], **extra)
        out = (tokens, padding, features[:, :, ANCHOR_FEATURE]) if return_anchors else (tokens, padding)
        if return_identity:
            out = (*out, self.identity_memory(features, x['history_valid'], x['history_pose']))
        return out

    def identity_memory(self, features, valid, pose):
        """Readout memory (keys, padding, slot roles) from entry rows past the encoder samples."""
        if self.cfg.identity_objective != 'readout':
            return {}
        from .decision_memory import ENTRY_FEATURES
        from .identity_readout import unpack_keys
        return self.identity_readout.memory(unpack_keys(features[:, :, ENTRY_FEATURES:]), valid.bool(), pose)

    def verification_terms(self, ctx, x, predicted=None):
        """Verifier logits at every identity query, with gradient into the current crop, the memory
        tokens and the verifier: the crossing candidates (flattened), the dense samples and this
        decision's own (detached) predicted path. Two gradient-free controls re-score the same queries
        with the previous row's memory (another state) and with no memory at all."""
        readout = getattr(self, 'identity_readout', None)
        if (getattr(self, 'identity_verifier', None) is None and readout is None) or 'identity_candidates' not in x:
            return {}
        b = len(x['identity_candidates'])
        predicted = (x['identity_candidates'].new_zeros(b, 0, 3) if predicted is None
                     else predicted.detach().to(x['identity_candidates'].dtype))
        points = torch.cat((x['identity_candidates'].reshape(b, -1, 3), x['identity_samples'], predicted), 1)
        if readout is not None:
            # Current keys at the query points; no position of the point enters.
            values, support = self.sample_local(ctx['identity_keys'].float(), points)
            memory = ctx['identity_memory']
            logits = readout(values, readout.project(memory))[1]
            with torch.no_grad():
                rolled = {key: value.roll(1, 0) for key, value in memory.items()}
                shuffled = readout(values, readout.project(rolled))[1]
                blank = dict(memory, identity_memory_padding=torch.ones_like(memory['identity_memory_padding']))
                empty = readout(values, readout.project(blank))[1]
        else:
            verifier = self.identity_verifier
            values, support = self.sample_local(ctx.get('deep_fp32', ctx['deep']).float(), points)
            tokens, padding = ctx['history_tokens'], ctx['history_padding']
            logits = verifier(values, points, verifier.project(tokens, padding))
            with torch.no_grad():
                shuffled = verifier(values, points, verifier.project(tokens.roll(1, 0), padding.roll(1, 0)))
                empty = verifier(values, points, verifier.project(tokens, torch.ones_like(padding)))
        # The shuffled control is meaningful only with a real row of another fiber (padding repeats row 0).
        rows = x.get('identity_row', torch.ones(b, dtype=torch.bool, device=points.device)).bool()
        fiber = x.get('identity_fiber', torch.arange(b, device=points.device))
        return dict(identity_points=points, identity_predicted_points=predicted, identity_logits=logits,
                    identity_shuffled_logits=shuffled, identity_empty_logits=empty, identity_support=support,
                    identity_shuffled_valid=rows & rows.roll(1, 0) & (fiber.roll(1, 0) != fiber))

    def identity_terms(self, ctx, x):
        """Memory-identity InfoNCE in a linear projection, contrasted in both directions.

        Query side: an old on-fiber memory entry (detached anchor) must be closer to the own fiber
        in the current crop (point 0 of ``identity_points``) than to every annotated neighbor there.
        Anchor side: the own-fiber point must be closer to this state's anchors than to the anchors
        of the other states in the batch (other fibers). Every anchor is a feature at the centre of
        its own past crop, so crop position cannot separate them and a solution that ignores the
        anchor fails this side. A control re-scores the query side with another state's anchors.
        Gradient reaches the encoder only through the current crop; incomplete rows are inert.
        """
        if getattr(self, 'identity_projection', None) is None or 'identity_points' not in x:
            return {}
        values, support = self.sample_local(ctx.get('deep_fp32', ctx['deep']).float(), x['identity_points'])
        rows = x.get('identity_row', torch.ones_like(support[:, 0])).bool()
        fiber = x.get('identity_fiber', torch.arange(len(rows), device=rows.device))
        points_ok = x['identity_point_mask'].bool() & support & rows[:, None]
        anchor_ok = x['identity_anchor_mask'].bool() & rows[:, None]
        query = F.normalize(self.identity_projection(values.float()), dim=-1)
        anchor = F.normalize(self.identity_projection(x['identity_anchor_features'].float()), dim=-1)
        temperature = self.cfg.identity_temperature

        def query_side(anchors, usable):
            similarity = torch.einsum('bsd,bkd->bsk', anchors, query)/temperature
            negatives = points_ok[:, 1:]
            valid = usable & points_ok[:, :1] & negatives.any(-1, keepdim=True)
            # Finite masking: no -inf/NaN can enter the gradient of an excluded anchor.
            logits = similarity.masked_fill(~points_ok[:, None, :], -1e4)
            loss = torch.where(valid, torch.logsumexp(logits, -1)-logits[..., 0], 0.)
            best = similarity[..., 1:].masked_fill(~negatives[:, None, :], -1e4).amax(-1)
            return loss, valid, valid & (similarity[..., 0] > best)

        loss, valid, correct = query_side(anchor, anchor_ok)
        b, slots = anchor_ok.shape
        cross = query[:, 0] @ anchor.flatten(0, 1).T/temperature  # (B, B*S)
        same_state = torch.eye(b, dtype=torch.bool, device=rows.device).repeat_interleave(slots, 1)
        same_fiber = fiber[:, None] == fiber.repeat_interleave(slots)[None]
        own = same_state & anchor_ok.flatten()[None]
        other = ~same_fiber & anchor_ok.flatten()[None]
        anchor_valid = points_ok[:, 0] & own.any(1) & other.any(1)
        own_logits = cross.masked_fill(~own, -1e4)
        anchor_loss = torch.where(anchor_valid, torch.logsumexp(cross.masked_fill(~(own | other), -1e4), 1)
                                  - torch.logsumexp(own_logits, 1), 0.)
        anchor_correct = anchor_valid & (own_logits.amax(1) > cross.masked_fill(~other, -1e4).amax(1))
        # Control: the same query side, scored with the previous row's anchors (another fiber).
        shifted = anchor_ok.roll(1, 0) & rows[:, None] & (fiber.roll(1, 0) != fiber)[:, None]
        _, control_valid, control_correct = query_side(anchor.roll(1, 0), shifted)
        return dict(identity_loss_per_state=loss.sum(-1)/valid.sum(-1).clamp_min(1)+anchor_loss,
                    identity_pairs=valid, identity_correct=correct,
                    identity_anchor_valid=anchor_valid, identity_anchor_correct=anchor_correct,
                    identity_control_pairs=control_valid, identity_control_correct=control_correct)

    def memory_features(self, x):
        """(B, SLOTS, *cfg.memory_entry_shape): recorded entries, plus entries encoded here from crops.

        ``history_features`` holds entries recorded at earlier decisions (zeros elsewhere).
        Slots flagged in ``history_encode`` have no recorded decision; their crops are encoded
        by this model's own encoder without gradient, exactly as at a decision.
        """
        valid = x['history_valid']
        features = x.get('history_features')
        if features is None:
            features = valid.new_zeros((*valid.shape, *self.cfg.memory_entry_shape), dtype=torch.bfloat16)
        encode = x.get('history_encode')
        if encode is not None and encode.any():
            rows = encode.flatten().nonzero().flatten()
            fresh = self.encode_memory_crops(x, rows)
            features = features.flatten(0, 1).index_copy(0, rows, fresh.to(features.dtype)).reshape(features.shape)
        return features

    def encode_memory_crops(self, x, rows):
        from .decision_memory import MEMORY_CHUNK
        crops, references = x['history_crops'], x['history_references']
        mask = x['history_reference_mask'].bool()
        if len(crops) != len(rows):
            raise ValueError('One memory crop is required per flagged slot')
        points = x['history_path_points'].flatten(0, 1)[rows]
        valid = x['history_path_valid'].flatten(0, 1)[rows]
        out = []
        with torch.no_grad():
            for start in range(0, len(crops), MEMORY_CHUNK):
                part = slice(start, start+MEMORY_CHUNK)
                if self.cfg.identity_objective == 'readout':
                    _, deep, stem = self.encoder.encode(crops[part], references[part], mask[part], return_stem=True)
                    keys = self.identity_readout.key_head(stem)
                else:
                    _, deep, _ = self.encoder(crops[part], references[part], mask[part])
                    keys = None
                out.append(self.memory_entry(deep, keys, points[part], valid[part]))
        return torch.cat(out)

    def memory_entry(self, deep, keys, path_points, path_valid):
        """One decision's entry rows: encoder samples, then (readout) its packed keys around its head."""
        entry = self.history_encoder.entry_features(deep, path_points, path_valid)
        if keys is None:
            return entry
        from .identity_readout import pack_keys
        sampled = self.identity_readout.entry_keys(keys.detach(), self.sample_local)
        return torch.cat((entry, pack_keys(sampled, self.cfg.hidden).to(entry.dtype)), 1)

    def current_memory_entry(self, deep, hist, hmask, keys=None):
        """This decision's own entry, as later decisions of the same trace will read it."""
        path = torch.stack((hist[:, 1], hist[:, 0], torch.zeros_like(hist[:, 0])), 1)
        valid = torch.stack((hmask[:, 1] > 0, hmask[:, 0] > 0, torch.ones_like(hmask[:, 0], dtype=torch.bool)), 1)
        return self.memory_entry(deep.detach(), keys, path, valid)

    def readout_field(self, keys, memory):
        """Readout at every token: (B, N, FEATURE_DIM+1) features and logit, and the detached probability field."""
        readout = self.identity_readout
        features, logits = readout(keys.flatten(2).transpose(1, 2), readout.project(memory))
        field = torch.sigmoid(logits.detach()).reshape(len(keys), 1, *keys.shape[-3:])
        return torch.cat((features, logits[..., None]), -1), field

    def context(self, x, hist, hmask):
        cfg = self.cfg
        references, mask = self.references(x, hist, hmask)
        readout = cfg.identity_objective == 'readout'
        if readout:
            dense, deep, stem = self.encoder.encode(x['fine'], references, mask, return_stem=True)
        else:
            dense, deep, _ = self.encoder(x['fine'], references, mask)
        if 'history_tokens' in x:
            tokens, padding = x['history_tokens'], x['history_padding']
            memory = {key: x[key] for key in READOUT_MEMORY if key in x}
        else:
            tokens, padding, memory = self.encode_history(x, return_identity=True)
        identity = identity_tokens = keys = None
        if readout:
            keys = self.identity_readout.key_head(stem)
            if cfg.identity_map or cfg.identity_feedback:
                features, identity = self.readout_field(keys, memory)
                if cfg.identity_map:
                    identity_tokens = self.identity_readout_embedding(features)
        elif cfg.identity_map or cfg.identity_feedback:
            from .identity_verifier import identity_field
            identity = identity_field(self, deep, tokens, padding)
        ctx = self.context_from_features(x, hist, references, mask, dense, deep, identity, identity_tokens)
        if readout:
            ctx.update(identity_keys=keys, identity_memory=memory)
        if tokens is None:  # memory 'none': decoder and scorer read only the current observations
            ctx['history_projected'] = ctx['confidence_history_projected'] = None
            return ctx
        ctx['history_tokens'], ctx['history_padding'] = tokens, padding
        history = (ctx['history_tokens'], ctx['history_padding'])
        ctx['history_projected'] = self.history_attention.project_memory(*history)
        if self.cfg.memory == 'decisions':
            ctx['memory_entry'] = self.current_memory_entry(deep, hist, hmask, keys)
        ctx['confidence_history_projected'] = self.confidence_scorer.history_attention.project_memory(*history)
        return ctx

    def select_prediction(self, output, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        return select_refinement(output, self.cfg, confidence_threshold, n_commit)

    def hazard_logits(self, ctx, points):
        # No generator state or proposed suffix enters a segment's evidence.
        points = points.detach()
        samples = self.confidence_scorer.segment_samples(points)
        spatial = self.evidence(ctx, samples.flatten(1, 2))
        spatial = spatial.reshape(*samples.shape[:3], -1)
        identity = (self.identity_samples(ctx, samples.flatten(1, 2)).reshape(samples.shape[:3])
                    if self.cfg.identity_feedback else None)
        return self.confidence_scorer(spatial, points,
                                     ctx['confidence_projected'], ctx['confidence_padding'],
                                     ctx['confidence_history_projected'], identity=identity)

    def decoder_memory(self, ctx):
        memory, padding = ctx['memory'], ctx['padding']
        ctx.update(confidence_projected=self.confidence_scorer.project_memory(memory, padding),
                   confidence_padding=padding)
        return memory, padding

    def proposal_output(self, initial, refinements, scores, valid):
        out = dict(initial_points=initial, refinement_points=torch.stack(refinements, 1),
                   refinement_mask=torch.stack(valid, 1))
        out.update({'refinement_'+name: torch.stack([score[i] for score in scores], 1)
                    for i, name in enumerate(('hazard_logits', 'confidence_logits', 'confidence'))})
        return out


class CoordinateRegressionFollower(ObservationFollower):
    """Coordinate regression with recurrent refinement."""

    def __init__(self, cfg):
        super().__init__(cfg)
        c, h = cfg.hidden, cfg.hidden
        self.query = nn.Sequential(nn.Linear(c+1,h),nn.SiLU(),nn.Linear(h,h))
        layer = PathDecoderLayer(h,cfg.heads,cfg.decoder_ffn,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.decoder = nn.TransformerDecoder(layer,cfg.decoder_layers,norm=nn.LayerNorm(h))
        self.coordinates = nn.Linear(h,2)
        nn.init.normal_(self.coordinates.weight,std=.005)
        nn.init.zeros_(self.coordinates.bias)
        if cfg.path_planes == 'crop':
            values = cfg.path_plane_values
            self.register_buffer('path_planes', torch.tensor(values, dtype=torch.float32), persistent=False)
        # Plain Python values (compiled graphs must not trace the numpy that defines them).
        self.proposal = cfg.proposal_slice
        self.query_scale = ((cfg.query_scale or cfg.n_future*cfg.future_step) if cfg.path_planes == 'future'
                            else float(abs(cfg.path_plane_values).max()))
        if cfg.tube_head:
            from .whole_crop import TubeHead
            self.tube = TubeHead(cfg)
        if cfg.recurrent_refinement_steps:
            # Spatial evidence, previous coordinates, detached failure/survival.
            width = cfg.path_evidence_width+3+2
            self.refinement_fusion = nn.Sequential(nn.Linear(width, cfg.hidden), nn.SiLU(),
                                                  nn.Linear(cfg.hidden, cfg.hidden))
            self.refinement_stage = nn.Embedding(cfg.recurrent_refinement_steps, cfg.hidden)
            nn.init.zeros_(self.refinement_stage.weight)
            if cfg.identity_feedback:
                # Identity-field probability at each proposed point; zero-initialized like the stage embedding.
                self.identity_feedback = nn.Linear(1, cfg.hidden)
                nn.init.zeros_(self.identity_feedback.weight)
                nn.init.zeros_(self.identity_feedback.bias)

    def forward(self, x, hist, hmask, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        ctx = self.context(x, hist, hmask)
        out = self.predict(ctx, hist, confidence_threshold)
        out = self.select_prediction(out, confidence_threshold, n_commit)
        if 'memory_entry' in ctx:
            out['memory_entry'] = ctx['memory_entry']
        return out

    def predict(self, ctx, hist, confidence_threshold=DEFAULT_CONFIDENCE):
        decoded, points, projected, padding = self.prepare_prediction(ctx, hist)
        return self.finish_prediction(ctx, decoded, points, projected, padding, confidence_threshold)

    def prepare_prediction(self, ctx, hist):
        """Initial proposal and attention projections shared by every attempt."""
        cfg = self.cfg
        memory, padding = self.decoder_memory(ctx)

        # These locations initialize feature queries, not output constraints.
        # Forward distance distinguishes the queries; self-attention couples
        # them and cross-attention can retrieve evidence anywhere in the crop.
        planes = self.path_planes if cfg.path_planes == 'crop' else self.planes
        reference = hist.new_zeros(len(hist), len(planes), 3)
        reference[..., 2] = planes
        query = self.query(self.query_features(ctx, reference))
        # Attached features and image projections are reused for this decision.
        projected = [layer.project_memory(memory)
                     for layer in self.decoder.layers]
        decoded = self.decode_cached(query, projected, padding, ctx)
        points = self.decode_coordinates(decoded)
        return decoded, points, projected, padding

    def crop_points(self, decoded):
        """Whole-crop path (B, P, 3): one lateral point per crop plane; no first-connection bound (the commit's
        recovery limit decides whether a far proposal may be committed)."""
        lateral = self.cfg.lateral_limit*torch.tanh(self.coordinates(decoded).float())
        return torch.cat((lateral, self.path_planes[None, :, None].expand(len(decoded), -1, -1)), -1)

    def decode_coordinates(self, decoded):
        """The same absolute-coordinate readout and bounds for every proposal."""
        cfg = self.cfg
        if cfg.path_planes == 'crop':
            return self.crop_points(decoded)[:, self.proposal]
        lateral = cfg.lateral_limit*torch.tanh(self.coordinates(decoded).float())
        first_limit = math.sqrt(max(0., cfg.max_recovery_distance**2-cfg.future_step**2))
        first = lateral[:, :1]
        first = first*(first_limit/first.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)
        lateral = torch.cat((first, lateral[:, 1:]), 1)
        return torch.cat((lateral, self.planes[None, :, None].expand(len(decoded), -1, -1)), -1)

    def finish_prediction(self, ctx, decoded, points, projected, padding,
                          confidence_threshold=DEFAULT_CONFIDENCE):
        from vesuvius.neural_tracing.fiber_follow.models.survival_confidence import survival_predictions
        cfg = self.cfg
        initial = points
        refinements = [points]
        hazards = self.hazard_logits(ctx, points)
        logits, confidence = survival_predictions(hazards)
        scores = [(hazards, logits, confidence)]
        valid = [torch.ones(len(points), device=points.device, dtype=torch.bool)]
        indices = torch.arange(len(points), device=points.device)
        active_ctx = ctx
        for stage in range(cfg.recurrent_refinement_steps):
            # An acceptance through the gate plane ends this row's retries. Dynamic batch
            # compaction skips both generator and scorer work for accepted rows.
            gate = cfg.gate_horizon
            counts, _ = commit_prefix(points, confidence[..., :gate].detach(), confidence_threshold,
                                      gate, cfg.max_recovery_distance)
            keep = torch.nonzero(counts < gate).flatten()
            if not len(keep):
                break
            if len(keep) != len(points):
                indices = indices[keep]
                points, decoded, hazards, confidence = (v[keep] for v in (points, decoded, hazards, confidence))
                projected = PathDecoderLayer.select_memory(projected, keep)
                padding = padding[keep]
                active_ctx = {key: active_ctx[key][keep] for key in
                              ('fine', 'deep', 'fine_fp32', 'deep_fp32', 'confidence_padding',
                               'history_tokens', 'history_padding', 'identity_field') if key in active_ctx} | dict(
                    confidence_projected=PathDecoderLayer.select_memory(active_ctx['confidence_projected'], keep),
                    history_projected=select_history(active_ctx['history_projected'], keep),
                    confidence_history_projected=select_history(active_ctx['confidence_history_projected'], keep))
            decoded, points = self.refine_prediction(active_ctx, points, decoded, hazards, confidence,
                projected, padding, self.refinement_stage.weight[stage])
            refinements.append(refinements[-1].index_copy(0, indices, points))
            hazards = self.hazard_logits(active_ctx, points)
            logits, confidence = survival_predictions(hazards)
            scores.append(tuple(previous.index_copy(0, indices, value)
                                for previous, value in zip(scores[-1], (hazards, logits, confidence))))
            valid.append(torch.zeros_like(valid[0]).index_fill(0, indices, True))
        return self.proposal_output(initial, refinements, scores, valid)

    def refine_prediction(self, ctx, points, decoded, hazards, confidence, projected, padding, stage_embedding):
        # Geometry cannot teach the scorer to manufacture convenient feedback.
        feedback = torch.stack((hazards.sigmoid(), confidence), -1).detach()
        if self.cfg.path_planes == 'crop':
            # Every crop plane is refined; scorer feedback exists for the proposal planes only.
            points = self.crop_points(decoded)
            full = feedback.new_zeros(*points.shape[:2], 2)
            full[:, self.proposal] = feedback
            feedback = full
        evidence = self.evidence(ctx, points)
        refreshed = self.refinement_fusion(torch.cat((evidence, points/16., feedback), -1))
        if self.cfg.identity_feedback:
            # The field carries no gradient, so geometry cannot shape the verifier's answer.
            refreshed = refreshed+self.identity_feedback(self.identity_samples(ctx, points)[..., None].to(refreshed.dtype))
        query = decoded+refreshed+stage_embedding.to(decoded.dtype)
        decoded = self.decode_cached(query, projected, padding, ctx)
        return decoded, self.decode_coordinates(decoded)

    def training_forward(self, x, hist, hmask, threshold, targets=None):
        """Fixed proposal slots; accepted rows retain their last actual attempt.

        The eager inference path still compacts accepted rows. Here acceptance
        is tensor data, so it cannot change the compiled graph or output shapes.
        Losses use refinement_mask to exclude the unused proposals.
        """
        from vesuvius.neural_tracing.fiber_follow.models.survival_confidence import survival_predictions
        ctx = self.context(x, hist, hmask)
        decoded, points, projected, padding = self.prepare_prediction(ctx, hist)
        crop = self.cfg.path_planes == 'crop'
        crop_paths = [self.crop_points(decoded)] if crop else None
        initial = points
        hazards = self.hazard_logits(ctx, points)
        logits, confidence = survival_predictions(hazards)
        refinements, scores = [points], [(hazards, logits, confidence)]
        active = torch.ones(len(hist), device=hist.device, dtype=torch.bool)
        valid = [active]
        for stage in range(self.cfg.recurrent_refinement_steps):
            gate = self.cfg.gate_horizon
            counts, _ = commit_prefix(points, confidence[..., :gate].detach(), threshold,
                                      gate, self.cfg.max_recovery_distance)
            active = active & (counts < gate)
            next_decoded, next_points = self.refine_prediction(ctx, points, decoded, hazards, confidence,
                projected, padding, self.refinement_stage.weight[stage])
            decoded = torch.where(active[:, None, None], next_decoded, decoded)
            points = torch.where(active[:, None, None], next_points, points)
            if crop:
                crop_paths.append(self.crop_points(decoded))
            next_hazards = self.hazard_logits(ctx, points)
            next_logits, next_confidence = survival_predictions(next_hazards)
            hazards, logits, confidence = (torch.where(active[:, None], new, old) for new, old in
                zip((next_hazards, next_logits, next_confidence), (hazards, logits, confidence)))
            refinements.append(points)
            scores.append((hazards, logits, confidence))
            valid.append(active)
        out = self.proposal_output(initial, refinements, scores, valid)
        if crop:
            out['refinement_crop_points'] = torch.stack(crop_paths, 1)
        if self.cfg.tube_head:
            out['tube_logits'] = self.tube(ctx['deep'])
        if 'memory_entry' in ctx:
            out['memory_entry'] = ctx['memory_entry']
        out.update(self.identity_terms(ctx, x))
        out.update(self.verification_terms(ctx, x, points))
        return out

    def decode_cached(self, query, projected, padding, ctx):
        for layer, kv in zip(self.decoder.layers, projected):
            query = layer.forward_cached(query, kv, padding,
                history=ctx['history_projected'], history_attention=getattr(self, 'history_attention', None))
        return self.decoder.norm(query)
