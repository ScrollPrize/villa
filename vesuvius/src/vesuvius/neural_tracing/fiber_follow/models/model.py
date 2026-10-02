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

ARCHITECTURE = 'axial_fiber_slabs_v15'
PATCH_ARCHITECTURE = 'axial_patch4_overlap_fiber_slabs_v15'
TOKEN_ARCHITECTURE = 'axial_patch4_overlap_tokens_fiber_slabs_v15'
STEM_ARCHITECTURE = 'axial_patch4_residual_stem_tokens_fiber_slabs_v15'
TOKEN_STRIDE = (8, 2, 2)
TOKEN_OFFSET = (3, 0, 0)


@dataclass
class DirectConfig:
    model_type: str = "aligned"
    direction_inputs: bool = False
    input_mode: str = 'ct'
    fine: CropSpec = field(default_factory=lambda: CropSpec(depth=120, width=104, behind=48, spacing=.5))
    channels: int = 32
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
    n_history: int = 128
    max_recovery_distance: float = 6.
    patch_radius: float = 1.
    recurrent_refinement_steps: int = 3
    encoder: str = 'patch4'
    token_only: bool = True
    stem_channels: int = 32
    stem_blocks: int = 2
    history_encoder: str = 'fine'
    history_path_tokens: bool = True
    path_geometry_tokens: bool = True

    def __post_init__(self):
        if (self.encoder, self.token_only, self.history_encoder, self.history_path_tokens,
                self.path_geometry_tokens, self.input_mode, self.direction_inputs) != (
                'patch4', True, 'fine', True, True, 'ct', False):
            raise ValueError('Only CT-only patch4 tokens with fine path/geometry history are supported')
        if type(self.history_path_tokens) is not bool or (self.history_path_tokens and self.history_encoder != 'fine'):
            raise ValueError('Path tokens require the fine history encoder')
        if type(self.path_geometry_tokens) is not bool or (self.path_geometry_tokens and not self.history_path_tokens):
            raise ValueError('Path geometry tokens require explicit history path tokens')
        if any(type(value) is not int or value < 1 for value in
               (self.encoder_ffn, self.decoder_ffn)):
            raise ValueError('Feed-forward widths must be positive integers')
        if self.history_encoder not in ('fine', 'legacy'):
            raise ValueError('History encoder must be fine or legacy')
        if self.input_mode not in ('ct', 'ct+presence') or (self.input_mode == 'ct' and self.direction_inputs):
            raise ValueError('CT-only inputs exclude presence and direction fields')
        if self.encoder not in ('conv', 'patch4'):
            raise ValueError('Encoder must be conv or patch4')
        if not isinstance(self.token_only, bool) or (self.token_only and self.encoder != 'patch4'):
            raise ValueError('Token-only features require the patch4 encoder')
        if type(self.stem_channels) is not int or self.stem_channels < 0 or type(self.stem_blocks) is not int or self.stem_blocks < 1:
            raise ValueError('Stem channels must be a nonnegative integer and stem blocks a positive integer')
        if self.stem_channels and not self.token_only:
            raise ValueError('Residual stem requires the token-only patch4 encoder')
        if isinstance(self.fine, dict):
            self.fine = CropSpec(**self.fine)
        c = self.fine
        if self.encoder == 'patch4' and (c.depth % 4 or c.width % 4):
            raise ValueError('Patch4 crop dimensions must be multiples of four')
        if min(c.depth, c.width) < 8 or not 0 <= c.behind < c.depth or not math.isfinite(c.spacing) or c.spacing <= 0:
            raise ValueError('Invalid fine crop')
        if type(self.scorer_layers) is not int or self.scorer_layers < 1:
            raise ValueError('Scorer layers must be a positive integer')
        if min(self.channels, self.hidden, self.heads, self.layers, self.decoder_layers,
               self.n_future, self.n_history) < 1 or self.hidden % self.heads:
            raise ValueError('Positive dimensions required; hidden must divide by heads')
        if not 0 < self.future_step <= self.max_recovery_distance or not math.isfinite(self.max_recovery_distance):
            raise ValueError('Invalid forward spacing or connection limit')
        if not 0 < self.patch_radius < (c.width-1)*c.spacing/2:
            raise ValueError('Local observation patch must fit fine crop')
        if self.n_future*self.future_step > (c.depth-c.behind-1)*c.spacing:
            raise ValueError('Future horizon exceeds fine image')
        if not isinstance(self.recurrent_refinement_steps, int) or self.recurrent_refinement_steps < 0:
            raise ValueError('Recurrent refinement steps must be a nonnegative integer')
        if not isinstance(self.direction_inputs, bool):
            raise ValueError('direction_inputs must be a boolean')

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
    def architecture(self):
        return 'axial_patch4_residual_stem_tokens_fiber_slabs_v17'

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

    def to_dict(self):
        return asdict(self)


def build_model(cfg):
    if cfg.model_type == 'flow_matching':
        from .flow import FlowFollower
        return FlowFollower(cfg)
    if cfg.model_type != 'aligned':
        raise ValueError(f'Unsupported model type: {cfg.model_type}')
    return DirectFollower(cfg)


def select_refinement(output, cfg, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
    """Longest acceptable prefix, then confidence at its end, then earlier pass.

    If every proposal stops, select the best first-point confidence among valid
    connections; the unchanged commit gate still stops. Selection never splices
    paths or transfers one proposal's confidence to another proposal.
    """
    if not 0 <= confidence_threshold <= 1:
        raise ValueError('Confidence threshold must lie in [0, 1]')
    window = cfg.n_future if n_commit is None else n_commit
    curves, confidence = output['refinement_points'], output['refinement_confidence'].detach()
    counts, allowed = commit_prefix(curves, confidence, confidence_threshold, window, cfg.max_recovery_distance)
    accepted = (confidence[..., -1] >= confidence_threshold) & allowed & output['refinement_mask']
    prior_accept = torch.cat((torch.zeros_like(accepted[:, :1]), accepted[:, :-1]), 1).long().cumsum(1) > 0
    valid = output['refinement_mask'] & ~prior_accept
    longest = counts.masked_fill(~valid, -1).max(-1, keepdim=True).values
    score = confidence.gather(-1, (counts-1).clamp_min(0)[..., None]).squeeze(-1)
    score = score.nan_to_num(nan=-torch.inf).masked_fill((counts != longest) | ~allowed | ~valid, -torch.inf)
    selected = score.argmax(-1)
    index = torch.arange(len(curves), device=curves.device)
    result = dict(output, selected_refinement=selected)
    for name in ('points', 'hazard_logits', 'confidence_logits', 'confidence'):
        result[name] = output['refinement_'+name][index, selected]
    return result


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
        attn = self.multihead_attn
        h, heads = attn.embed_dim, attn.num_heads
        q = F.linear(self.norm2(x), attn.in_proj_weight[:h], attn.in_proj_bias[:h])
        q = q.reshape(len(x), -1, heads, h//heads).transpose(1, 2).contiguous()
        value = self.attend_memory(q, kv, padding)
        value = value.transpose(1, 2).contiguous().reshape(len(x), -1, h)
        x = x+self.dropout2(attn.out_proj(value))
        if history is not None:
            x = history_attention.forward_cached(x, *history)
        return x+self._ff_block(self.norm3(x))


class AxialBlock(nn.Module):
    def __init__(self, width, heads, *, ffn=1024, local_convolution=False, rotary=True):
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
        self.architecture = cfg.architecture
        from .patch_encoder import PatchShuffleEncoder
        from .history_slabs import HistoryEncoder, HistoryAttention
        from .survival_confidence import SegmentSurvivalScorer
        from .path_geometry import PathGeometryTokens
        self.encoder = PatchShuffleEncoder(cfg)
        self.reference_token = nn.Sequential(nn.Linear(cfg.hidden+8,cfg.hidden),nn.SiLU(),nn.Linear(cfg.hidden,cfg.hidden))
        self.history_encoder = HistoryEncoder(cfg)
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

    def context_from_features(self, x, hist, references, mask, dense, deep):
        cfg = self.cfg
        image_tokens = deep.flatten(2).transpose(1,2)
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
        if cfg.path_geometry_tokens:
            # Older observed path in this frame, unmasked by crop support. Shared by
            # generator and scorer through the same memory and padding.
            valid = x['path_geometry_valid'].bool()
            geometry = self.path_geometry(x['path_geometry'], valid)
            memory = torch.cat((memory,geometry.to(memory.dtype)),1)
            padding = torch.cat((padding,~valid),1)
        ctx = dict(fine=dense,deep=deep,memory=memory,padding=padding,reference_mask=mask)
        if cfg.recurrent_refinement_steps:
            ctx.update(fine_fp32=sampling_dense, deep_fp32=sampling_dense if cfg.token_only else deep.float())
        return ctx

    def sample_local(self, features, points):
        cfg = self.cfg
        return sample_features(features, points, cfg.fine,
                               cfg.token_stride if cfg.token_only else 1,
                               cfg.token_offset if cfg.token_only else (0,0,0))

    def patches(self, fine, points):
        return self.sample_local(fine, points)[0]

    def query_features(self, ctx, initial):
        patches = self.patches(ctx.get('fine_fp32', ctx['fine']),initial).to(ctx['fine'].dtype)
        return torch.cat((patches,initial[...,2:]/(self.cfg.n_future*self.cfg.future_step)),-1)

    def evidence(self, ctx, points, stage):
        values, support = self.sample_local(ctx.get('deep_fp32', ctx['deep']), points)
        return torch.cat((values.to(ctx['deep'].dtype), support[..., None]), -1)

    def encode_history(self, x):
        extra = {key: x['history_'+key] for key in ('path_points', 'path_tangents', 'path_valid')} if self.cfg.history_path_tokens else {}
        return self.history_encoder(x['history_slabs'], x['history_valid'], x['history_pose'], **extra)

    def context(self, x, hist, hmask):
        references, mask = self.references(x, hist, hmask)
        dense, deep, _ = self.encoder(x['fine'], references, mask)
        ctx = self.context_from_features(x, hist, references, mask, dense, deep)
        if 'history_tokens' in x:
            ctx['history_tokens'], ctx['history_padding'] = x['history_tokens'], x['history_padding']
        else:
            ctx['history_tokens'], ctx['history_padding'] = self.encode_history(x)
        history = (ctx['history_tokens'], ctx['history_padding'])
        ctx['history_projected'] = self.history_attention.project_memory(*history)
        ctx['confidence_history_projected'] = self.confidence_scorer.history_attention.project_memory(*history)
        return ctx

    def select_prediction(self, output, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        return select_refinement(output, self.cfg, confidence_threshold, n_commit)

    def confidence_logits(self, ctx, decoded, points):
        from vesuvius.neural_tracing.fiber_follow.models.survival_confidence import survival_predictions
        return survival_predictions(self.hazard_logits(ctx, points))[0]

    def hazard_logits(self, ctx, points):
        # No generator state or proposed suffix enters a segment's evidence.
        points = points.detach()
        samples = self.confidence_scorer.segment_samples(points)
        spatial = self.evidence(ctx, samples.flatten(1, 2), 'confidence')
        spatial = spatial.reshape(*samples.shape[:3], -1)
        return self.confidence_scorer(spatial, points,
                                     ctx['confidence_projected'], ctx['confidence_padding'],
                                     ctx['confidence_history_projected'])

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


class DirectFollower(ObservationFollower):
    """Aligned direct coordinates with recurrent refinement."""

    def __init__(self, cfg):
        super().__init__(cfg)
        c, h = cfg.hidden, cfg.hidden
        self.query = nn.Sequential(nn.Linear((c if cfg.token_only else 9*c)+1,h),nn.SiLU(),nn.Linear(h,h))
        layer = PathDecoderLayer(h,cfg.heads,cfg.decoder_ffn,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.decoder = nn.TransformerDecoder(layer,cfg.decoder_layers,norm=nn.LayerNorm(h))
        self.coordinates = nn.Linear(h,2)
        nn.init.normal_(self.coordinates.weight,std=.005)
        nn.init.zeros_(self.coordinates.bias)
        if cfg.recurrent_refinement_steps:
            # Spatial evidence, previous coordinates, detached failure/survival.
            width = cfg.path_evidence_width+3+2
            self.refinement_fusion = nn.Sequential(nn.Linear(width, cfg.hidden), nn.SiLU(),
                                                  nn.Linear(cfg.hidden, cfg.hidden))
            self.refinement_stage = nn.Embedding(cfg.recurrent_refinement_steps, cfg.hidden)
            nn.init.zeros_(self.refinement_stage.weight)

    def forward(self, x, hist, hmask, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        ctx = self.context(x, hist, hmask)
        out = self.predict(ctx, hist, confidence_threshold)
        return self.select_prediction(out, confidence_threshold, n_commit)

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
        reference = hist.new_zeros(len(hist), cfg.n_future, 3)
        reference[..., 2] = self.planes
        query = self.query(self.query_features(ctx, reference))
        # Attached features and image projections are reused for this decision.
        projected = [layer.project_memory(memory)
                     for layer in self.decoder.layers]
        decoded = self.decode_cached(query, projected, padding, ctx)
        points = self.decode_coordinates(decoded)
        return decoded, points, projected, padding

    def decode_coordinates(self, decoded):
        """The same absolute-coordinate readout and bounds for every proposal."""
        cfg = self.cfg
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
            # A full-horizon acceptance ends this row's retries. Dynamic batch
            # compaction skips both generator and scorer work for accepted rows.
            counts, _ = commit_prefix(points, confidence.detach(), confidence_threshold,
                                      cfg.n_future, cfg.max_recovery_distance)
            keep = torch.nonzero(counts < cfg.n_future).flatten()
            if not len(keep):
                break
            if len(keep) != len(points):
                indices = indices[keep]
                points, decoded, hazards, confidence = (v[keep] for v in (points, decoded, hazards, confidence))
                projected = PathDecoderLayer.select_memory(projected, keep)
                padding = padding[keep]
                active_ctx = {key: active_ctx[key][keep] for key in
                              ('fine', 'deep', 'fine_fp32', 'deep_fp32', 'confidence_padding',
                               'history_tokens', 'history_padding')} | dict(
                    confidence_projected=PathDecoderLayer.select_memory(active_ctx['confidence_projected'], keep),
                    history_projected=tuple(v[keep] for v in active_ctx['history_projected']),
                    confidence_history_projected=tuple(v[keep] for v in active_ctx['confidence_history_projected']))
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
        evidence = self.evidence(ctx, points, 'refinement')
        # Geometry cannot teach the scorer to manufacture convenient feedback.
        feedback = torch.stack((hazards.sigmoid(), confidence), -1).detach()
        refreshed = self.refinement_fusion(torch.cat((evidence, points/16., feedback), -1))
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
        initial = points
        hazards = self.hazard_logits(ctx, points)
        logits, confidence = survival_predictions(hazards)
        refinements, scores = [points], [(hazards, logits, confidence)]
        active = torch.ones(len(hist), device=hist.device, dtype=torch.bool)
        valid = [active]
        for stage in range(self.cfg.recurrent_refinement_steps):
            counts, _ = commit_prefix(points, confidence.detach(), threshold,
                                      self.cfg.n_future, self.cfg.max_recovery_distance)
            active = active & (counts < self.cfg.n_future)
            next_decoded, next_points = self.refine_prediction(ctx, points, decoded, hazards, confidence,
                projected, padding, self.refinement_stage.weight[stage])
            decoded = torch.where(active[:, None, None], next_decoded, decoded)
            points = torch.where(active[:, None, None], next_points, points)
            next_hazards = self.hazard_logits(ctx, points)
            next_logits, next_confidence = survival_predictions(next_hazards)
            hazards, logits, confidence = (torch.where(active[:, None], new, old) for new, old in
                zip((next_hazards, next_logits, next_confidence), (hazards, logits, confidence)))
            refinements.append(points)
            scores.append((hazards, logits, confidence))
            valid.append(active)
        return self.proposal_output(initial, refinements, scores, valid)

    def decode_cached(self, query, projected, padding, ctx):
        for layer, kv in zip(self.decoder.layers, projected):
            query = layer.forward_cached(query, kv, padding,
                history=ctx['history_projected'], history_attention=self.history_attention)
        return self.decoder.norm(query)

