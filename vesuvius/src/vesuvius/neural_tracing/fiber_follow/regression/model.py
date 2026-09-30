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
from vesuvius.neural_tracing.fiber_follow.shared.policy import DEFAULT_CONFIDENCE, commit_prefix

ARCHITECTURE = 'axial_fiber_slabs_v10'
PATCH_ARCHITECTURE = 'axial_patch4_fiber_slabs_v10'
TOKEN_ARCHITECTURE = 'axial_patch4_tokens_fiber_slabs_v10'
TOKEN_STRIDE = (8, 2, 2)
TOKEN_OFFSET = (3, 0, 0)


@dataclass
class DirectConfig:
    direction_inputs: bool = False
    fine: CropSpec = field(default_factory=lambda: CropSpec(depth=120, width=101, behind=48, spacing=.5))
    channels: int = 32
    hidden: int = 128
    heads: int = 4
    layers: int = 4
    decoder_layers: int = 4
    activation_checkpointing: bool = False
    n_future: int = 16
    future_step: float = 1.
    n_history: int = 128
    max_recovery_distance: float = 6.
    patch_radius: float = 1.
    recurrent_refinement_steps: int = 2
    encoder: str = 'conv'
    token_only: bool = False

    def __post_init__(self):
        if self.encoder not in ('conv', 'patch4'):
            raise ValueError('Encoder must be conv or patch4')
        if not isinstance(self.token_only, bool) or (self.token_only and self.encoder != 'patch4'):
            raise ValueError('Token-only features require the patch4 encoder')
        if isinstance(self.fine, dict):
            self.fine = CropSpec(**self.fine)
        c = self.fine
        if min(c.depth, c.width) < 8 or not 0 <= c.behind < c.depth or not math.isfinite(c.spacing) or c.spacing <= 0:
            raise ValueError('Invalid fine crop')
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
        return 8 if self.direction_inputs else 2

    @property
    def recent_history_points(self):
        return self.n_history

    @property
    def lateral_limit(self):
        return (self.fine.width-1)*self.fine.spacing/2-self.patch_radius

    @property
    def architecture(self):
        if self.token_only:
            return TOKEN_ARCHITECTURE
        return PATCH_ARCHITECTURE if self.encoder == 'patch4' else ARCHITECTURE

    @property
    def path_evidence_width(self):
        return self.hidden+1 if self.token_only else 27*(self.channels+1)+self.hidden+1

    @property
    def token_stride(self):
        return (4, 4, 4) if self.encoder == 'patch4' else TOKEN_STRIDE

    @property
    def token_offset(self):
        return (1.5, 1.5, 1.5) if self.encoder == 'patch4' else TOKEN_OFFSET

    @property
    def token_shape(self):
        return tuple(math.ceil(n/s) for n,s in zip((self.fine.depth,self.fine.width,self.fine.width),self.token_stride))

    def to_dict(self):
        return asdict(self)


def build_model(cfg):
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


class ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(math.gcd(8,channels),channels),nn.SiLU(),
            nn.Conv3d(channels,channels,3,padding=1,bias=False),
            nn.GroupNorm(math.gcd(8,channels),channels),nn.SiLU(),
            nn.Conv3d(channels,channels,3,padding=1,bias=False))

    def forward(self, x):
        return x+self.net(x)


class AxisAttention(nn.Module):
    """Unmasked attention along one whole spatial axis of a BDHWC tensor."""
    def __init__(self, width, heads, axis):
        super().__init__()
        self.heads, self.axis = heads, axis
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width,3*width)
        self.projection = nn.Linear(width,width)

    def forward(self, x):
        moved = x.movedim(self.axis,-2)
        shape = moved.shape
        seq = self.norm(moved).reshape(-1,shape[-2],shape[-1])
        qkv = self.qkv(seq).reshape(seq.shape[0],seq.shape[1],3,self.heads,shape[-1]//self.heads)
        q,k,v = qkv.permute(2,0,3,1,4).unbind(0)
        attended = F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False)
        attended = attended.transpose(1,2).reshape_as(seq)
        return x+self.projection(attended).reshape(shape).movedim(-2,self.axis)


class DepthwiseConv3d(nn.Conv3d):
    """Use NCDHW for depthwise kernels; ordinary convolutions stay channels-last."""
    def forward(self, x):
        # A singleton input-channel dimension makes channels-last weights also
        # report is_contiguous(). Clone explicitly to reset those strides: cuDNN
        # otherwise chooses its slow channels-last path despite contiguous x.
        weight = self.weight.clone(memory_format=torch.contiguous_format)
        return self._conv_forward(x.contiguous(), weight, self.bias)


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
        attn = self.multihead_attn
        h, heads = attn.embed_dim, attn.num_heads
        kv = F.linear(memory, attn.in_proj_weight[h:], attn.in_proj_bias[h:])
        return tuple(v.reshape(len(memory), -1, heads, h//heads).transpose(1, 2).contiguous()
                     for v in kv.chunk(2, dim=-1))

    def compact_memory(self, kv, padding):
        """Gather valid CUDA keys once per decision, before any path attempts."""
        if not kv[0].is_cuda or kv[0].dtype not in (torch.float16, torch.bfloat16):
            return kv
        rows = []
        for row in range(kv[0].shape[0]):
            keep = (~padding[row]).nonzero().flatten()
            torch._check(keep.numel() > 0)
            rows.append(tuple(v[row:row+1].index_select(2, keep) for v in kv))
        return tuple(rows)

    @staticmethod
    def select_memory(projected, keep):
        """Select active inference rows from dense or variable-length cached K/V."""
        if isinstance(projected[0][0], tuple):
            rows = keep.tolist()
            return [tuple(kv[i] for i in rows) for kv in projected]
        return [tuple(v[keep] for v in kv) for kv in projected]

    def attend_memory(self, q, kv, padding):
        """Reuse compacted CUDA K/V with unmasked Flash attention.

        The mask is key padding only: removing keys preserves softmax and
        leaves query order (including causal candidate self-attention) intact.
        Image/plane tokens guarantee at least one valid key in every row.
        Other backends remain available on devices without Flash support.
        """
        if isinstance(kv[0], tuple):
            values = []
            for row, (k, v) in enumerate(kv):
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                                  SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True):
                    values.append(F.scaled_dot_product_attention(q[row:row+1].contiguous(),
                        k, v))
            return torch.cat(values, 0)
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
            x = history_attention(x, *history)
        return x+self._ff_block(self.norm3(x))


class AxialBlock(nn.Module):
    def __init__(self, width, heads, *, local_convolution=True):
        super().__init__()
        self.axes = nn.ModuleList(AxisAttention(width,heads,a) for a in (3,2,1))
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
        self.local = nn.Sequential(DepthwiseConv3d(width,width,3,padding=1,groups=width),
                                   nn.SiLU(),nn.Conv3d(width,width,1)) if local_convolution else None

    def forward(self, x):
        for axis in self.axes:
            x = axis(x)
        x = x+self.mlp(self.norm(x))
        return x+self.local(x.permute(0,4,1,2,3)).permute(0,2,3,4,1) if self.local is not None else x


def token_coordinates(cfg):
    """Token centers in crop-local XYZ, including patch padding at crop edges."""
    d,y,x = torch.meshgrid(*(torch.arange(n).float() for n in cfg.token_shape),indexing='ij')
    xyz = torch.stack((x,y,d),-1)
    xyz = (xyz*xyz.new_tensor(tuple(reversed(cfg.token_stride)))
           +xyz.new_tensor(tuple(reversed(cfg.token_offset))))*cfg.fine.spacing
    return xyz-xyz.new_tensor(((cfg.fine.width-1)*cfg.fine.spacing/2,)*2+(cfg.fine.behind*cfg.fine.spacing,))


class AxialEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c,h = cfg.channels,cfg.hidden
        self.stem = nn.Sequential(nn.Conv3d(cfg.input_channels,c,3,padding=1,bias=False),ResidualConv(c))
        self.down = nn.Sequential(nn.Conv3d(c,2*c,3,stride=2,padding=1,bias=False),ResidualConv(2*c),
                                  nn.Conv3d(2*c,4*c,3,padding=1,bias=False),nn.SiLU(),ResidualConv(4*c))
        self.compress = nn.Conv3d(4*c,h,(4,1,1),stride=(4,1,1))
        self.position = nn.Linear(3,h)
        # Observed path occupancy, mean age and seed occupancy. No annotation masks.
        self.condition = nn.Linear(3,h,bias=False)
        self.blocks = nn.ModuleList(AxialBlock(h,cfg.heads) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(h)
        self.dense_projection = nn.Conv3d(h,c,1)
        self.dense_decoder = nn.Sequential(nn.Conv3d(2*c,c,3,padding=1,bias=False),
            nn.GroupNorm(math.gcd(8,c),c),nn.SiLU(),nn.Conv3d(c,c,3,padding=1,bias=False))
        xyz = token_coordinates(cfg)
        self.register_buffer('token_xyz',xyz.reshape(-1,3),persistent=False)
        d,y,x = torch.meshgrid(torch.arange(cfg.fine.depth).float(),
            torch.arange(cfg.fine.width).float(),torch.arange(cfg.fine.width).float(),indexing='ij')
        points = torch.stack((x,y,d),-1)*cfg.fine.spacing
        points -= points.new_tensor(((cfg.fine.width-1)*cfg.fine.spacing/2,)*2+(cfg.fine.behind*cfg.fine.spacing,))
        self.register_buffer('decode_grid',feature_grid(points,cfg.fine,cfg.token_shape,cfg.token_stride,cfg.token_offset)[None],persistent=False)

    def conditioning(self, references, mask):
        cfg = self.cfg
        points = torch.where(mask[...,None],references,0.).float()
        origin = device_vector(points, (-(cfg.fine.width-1)*cfg.fine.spacing/2,)*2+(-cfg.fine.behind*cfg.fine.spacing,))
        offset = device_vector(points, tuple(reversed(cfg.token_offset)))*cfg.fine.spacing
        index = torch.round((points-origin-offset)/(device_vector(points, tuple(reversed(cfg.token_stride)))*cfg.fine.spacing)).long()
        d,y,x = cfg.token_shape
        index = torch.stack((index[...,0].clamp(0,x-1),index[...,1].clamp(0,y-1),index[...,2].clamp(0,d-1)),-1)
        flat = index[...,0]+x*(index[...,1]+y*index[...,2])
        ages = torch.arange(1,cfg.n_history+2,device=points.device).float()/cfg.n_history
        history = mask.clone()
        history[:,-1] = False
        seed = mask & ~history
        values = torch.stack((history.float(),history*ages[None],seed.float()),-1)
        rendered = points.new_zeros(len(points),d*y*x,3).scatter_add(1,flat[...,None].expand(-1,-1,3),values)
        count = rendered[...,:1]
        rendered = torch.cat((count.clamp_max(1),rendered[...,1:2]/count.clamp_min(1),rendered[...,2:].clamp_max(1)),-1)
        return rendered.reshape(len(points),d,y,x,3)

    def encode(self, image, references, mask):
        """Independent observation features, before any persistent-memory read."""
        fine = self.stem(image)
        down = self.down(fine)
        down = F.pad(down,(0,0,0,0,0,(-down.shape[2])%4))
        tokens = self.compress(down).permute(0,2,3,4,1)
        tokens = tokens+self.position(self.token_xyz/16).reshape(*self.cfg.token_shape,self.cfg.hidden).to(tokens.dtype)
        tokens = tokens+self.condition(self.conditioning(references,mask)).to(tokens.dtype)
        for block in self.blocks:
            if self.cfg.activation_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block,tokens,use_reentrant=False)
            else:
                tokens = block(tokens)
        tokens = self.norm(tokens)
        deep = tokens.permute(0,4,1,2,3)
        return fine, deep

    def decode(self, fine, deep):
        """Build localization features from local appearance and spatial context."""
        low = self.dense_projection(deep)
        up = F.grid_sample(low.float(),self.decode_grid.expand(len(fine),-1,-1,-1,-1),
                           padding_mode='border',align_corners=True).to(fine.dtype)
        return fine+self.dense_decoder(torch.cat((fine,up),1))

    def forward(self, image, references, mask):
        fine, deep = self.encode(image, references, mask)
        return self.decode(fine, deep), deep, deep.flatten(2).transpose(1,2)


class OutputPlaneFeatures(nn.Module):
    """Every lateral pixel on each output plane, with physical XYZ positions."""
    def __init__(self, cfg):
        super().__init__()
        crop = cfg.fine
        planes = torch.arange(1, cfg.n_future+1).float()*cfg.future_step
        lateral = (torch.arange(crop.width).float()-(crop.width-1)/2)*crop.spacing
        f, v, u = torch.meshgrid(planes, lateral, lateral, indexing='ij')
        self.register_buffer('xyz', torch.stack((u, v, f), -1).reshape(-1, 3), persistent=False)
        depth = crop.behind+planes/crop.spacing
        lower = depth.floor().long().clamp(0, crop.depth-1)
        self.register_buffer('lower', lower, persistent=False)
        self.register_buffer('upper', (lower+1).clamp(max=crop.depth-1), persistent=False)
        self.register_buffer('fraction', (depth-depth.floor())[None, None, :, None, None], persistent=False)
        self.on_grid = bool((depth == depth.floor()).all())
        self.projection = nn.Linear(cfg.channels, cfg.hidden)
        self.position = nn.Linear(3, cfg.hidden)

    def sample(self, fine):
        # Gather whole slices without pooling or resampling the lateral axes.
        # Interpolate only depth for configurations whose planes fall between slices.
        values = fine.index_select(2, self.lower)
        if not self.on_grid:
            upper = fine.index_select(2, self.upper)
            values = (values.float()+(upper.float()-values.float())*self.fraction).to(fine.dtype)
        return values.flatten(2).transpose(1, 2)

    def project(self, samples):
        tokens = self.projection(samples)
        return tokens+self.position(self.xyz/16.).to(tokens.dtype)

    def forward(self, fine):
        return self.project(self.sample(fine))


class DirectFollower(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.architecture = cfg.architecture
        c,h = (cfg.hidden if cfg.token_only else cfg.channels),cfg.hidden
        if cfg.encoder == 'patch4':
            from .patch_encoder import PatchShuffleEncoder
            self.encoder = PatchShuffleEncoder(cfg)
        else:
            self.encoder = AxialEncoder(cfg)
        self.reference_token = nn.Sequential(nn.Linear(c+8,h),nn.SiLU(),nn.Linear(h,h))
        self.query = nn.Sequential(nn.Linear((c if cfg.token_only else 9*c)+1,h),nn.SiLU(),nn.Linear(h,h))
        layer = PathDecoderLayer(h,cfg.heads,2*h,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.decoder = nn.TransformerDecoder(layer,cfg.decoder_layers,norm=nn.LayerNorm(h))
        self.coordinates = nn.Linear(h,2)
        nn.init.normal_(self.coordinates.weight,std=.005)
        nn.init.zeros_(self.coordinates.bias)
        self.register_buffer('stencil',torch.tensor([[a,b,0.] for a in (-1.,0.,1.) for b in (-1.,0.,1.)])*cfg.patch_radius,persistent=False)
        self.register_buffer('path_stencil',torch.tensor([[a*cfg.patch_radius,b*cfg.patch_radius,z]
            for z in (-1.,0.,1.) for a in (-1.,0.,1.) for b in (-1.,0.,1.)]),persistent=False)
        self.register_buffer('planes',torch.arange(1,cfg.n_future+1).float()*cfg.future_step,persistent=False)
        from .history_slabs import HistoryEncoder, HistoryAttention
        from .survival_confidence import SegmentSurvivalScorer
        self.history_encoder = HistoryEncoder(cfg)
        self.history_attention = HistoryAttention(h, cfg.heads)
        self.confidence_scorer = SegmentSurvivalScorer(cfg)
        if cfg.recurrent_refinement_steps:
            # Spatial evidence, previous coordinates, detached failure/survival.
            width = cfg.path_evidence_width+3+2
            self.refinement_fusion = nn.Sequential(nn.Linear(width, cfg.hidden), nn.SiLU(),
                                                  nn.Linear(cfg.hidden, cfg.hidden))
            self.refinement_stage = nn.Embedding(cfg.recurrent_refinement_steps, cfg.hidden)
            nn.init.zeros_(self.refinement_stage.weight)
        self.output_plane_features = None if cfg.token_only else OutputPlaneFeatures(cfg)

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
        if self.cfg.token_only:
            return self.sample_local(fine, points)[0]
        b,k,_ = points.shape
        values,_ = sample_features(fine,(points[:,:,None]+self.stencil).reshape(b,k*9,3),self.cfg.fine)
        return values.reshape(b,k,-1)

    def query_features(self, ctx, initial):
        patches = self.patches(ctx.get('fine_fp32', ctx['fine']),initial).to(ctx['fine'].dtype)
        return torch.cat((patches,initial[...,2:]/(self.cfg.n_future*self.cfg.future_step)),-1)

    def evidence(self, ctx, points, stage):
        if self.cfg.token_only:
            values, support = self.sample_local(ctx.get('deep_fp32', ctx['deep']), points)
            return torch.cat((values.to(ctx['deep'].dtype), support[...,None]), -1)
        b,k,_ = points.shape
        local,support = sample_features(ctx.get('fine_fp32', ctx['fine']),(points[:,:,None]+self.path_stencil).reshape(b,k*27,3),self.cfg.fine)
        local = local.to(ctx['fine'].dtype)
        local = torch.cat((local,support[...,None]),-1).reshape(b,k,-1)
        deep,valid = sample_features(ctx.get('deep_fp32', ctx['deep']),points,self.cfg.fine,self.cfg.token_stride,self.cfg.token_offset)
        deep = deep.to(ctx['deep'].dtype)
        return torch.cat((local,deep,valid[...,None]),-1)

    def encode_history(self, x):
        return self.history_encoder(x['history_slabs'], x['history_valid'], x['history_pose'])

    def context(self, x, hist, hmask):
        references, mask = self.references(x, hist, hmask)
        stem, deep = self.encoder.encode(x['fine'], references, mask)
        dense = self.encoder.decode(stem, deep)
        ctx = self.context_from_features(x, hist, references, mask, dense, deep)
        if 'history_tokens' in x:
            ctx['history_tokens'], ctx['history_padding'] = x['history_tokens'], x['history_padding']
        else:
            ctx['history_tokens'], ctx['history_padding'] = self.encode_history(x)
        return ctx

    def forward(self, x, hist, hmask, candidates=None,
                confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        ctx = self.context(x, hist, hmask)
        out = self.predict(ctx, hist, candidates, confidence_threshold)
        return self.select_prediction(out, confidence_threshold, n_commit)

    def select_prediction(self, output, confidence_threshold=DEFAULT_CONFIDENCE, n_commit=None):
        return select_refinement(output, self.cfg, confidence_threshold, n_commit)

    def confidence_logits(self, ctx, decoded, points):
        from .survival_confidence import survival_predictions
        return survival_predictions(self.hazard_logits(ctx, points))[0]

    def hazard_logits(self, ctx, points):
        # No generator state or proposed suffix enters a segment's evidence.
        points = points.detach()
        samples = self.confidence_scorer.segment_samples(points)
        spatial = self.evidence(ctx, samples.flatten(1, 2), 'confidence')
        spatial = spatial.reshape(*samples.shape[:3], -1)
        return self.confidence_scorer(spatial, points,
                                     ctx['confidence_projected'], ctx['confidence_padding'],
                                     (ctx['history_tokens'], ctx['history_padding']))

    def predict(self, ctx, hist, candidates=None, confidence_threshold=DEFAULT_CONFIDENCE):
        decoded, points, projected, padding = self.prepare_prediction(ctx, hist)
        return self.finish_prediction(ctx, decoded, points, projected, padding, candidates, confidence_threshold)

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
        projected = [layer.compact_memory(layer.project_memory(memory), padding)
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

    def decoder_memory(self, ctx):
        memory, padding = ctx['memory'], ctx['padding']
        if self.cfg.token_only:
            ctx.update(confidence_projected=self.confidence_scorer.project_memory(memory, padding),
                       confidence_padding=padding)
            return memory, padding
        # Sample the complete planes once; generator and scorer learn independent
        # channel/position projections of exactly the same spatial evidence.
        samples = self.output_plane_features.sample(ctx['fine'])
        fine = self.output_plane_features.project(samples).to(memory.dtype)
        confidence_fine = self.confidence_scorer.plane_tokens(
            samples, self.output_plane_features.xyz).to(memory.dtype)
        padding = torch.cat((padding, padding.new_zeros(fine.shape[:2])), 1)
        # Every segment reads every plane. K/V stay differentiable and are reused
        # across generated and supplied curves within this decision.
        confidence_memory = torch.cat((memory, confidence_fine), 1)
        ctx.update(confidence_projected=self.confidence_scorer.project_memory(confidence_memory, padding),
                   confidence_padding=padding)
        memory = torch.cat((memory, fine), 1)
        return memory, padding

    def finish_prediction(self, ctx, decoded, points, projected, padding, candidates=None,
                          confidence_threshold=DEFAULT_CONFIDENCE):
        from .survival_confidence import survival_predictions
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
                    confidence_projected=PathDecoderLayer.select_memory(active_ctx['confidence_projected'], keep))
            decoded, points = self.refine_prediction(active_ctx, points, decoded, hazards, confidence,
                projected, padding, self.refinement_stage.weight[stage])
            refinements.append(refinements[-1].index_copy(0, indices, points))
            hazards = self.hazard_logits(active_ctx, points)
            logits, confidence = survival_predictions(hazards)
            scores.append(tuple(previous.index_copy(0, indices, value)
                                for previous, value in zip(scores[-1], (hazards, logits, confidence))))
            valid.append(torch.zeros_like(valid[0]).index_fill(0, indices, True))
        out = self.proposal_output(initial, refinements, scores, valid)
        if candidates is not None:
            out.update(self.score_candidates(ctx, candidates))
        return out

    def proposal_output(self, initial, refinements, scores, valid):
        out = dict(initial_points=initial, refinement_points=torch.stack(refinements, 1),
                   refinement_mask=torch.stack(valid, 1))
        out.update({'refinement_'+name: torch.stack([score[i] for score in scores], 1)
                    for i, name in enumerate(('hazard_logits', 'confidence_logits', 'confidence'))})
        return out

    def score_candidates(self, ctx, candidates):
        from .survival_confidence import survival_predictions
        hazards = torch.stack([self.hazard_logits(ctx, curve) for curve in candidates.unbind(1)], 1)
        logits, confidence = survival_predictions(hazards)
        return dict(candidate_hazard_logits=hazards, candidate_confidence_logits=logits,
                    candidate_confidence=confidence)

    def refine_prediction(self, ctx, points, decoded, hazards, confidence, projected, padding, stage_embedding):
        evidence = self.evidence(ctx, points, 'refinement')
        # Geometry cannot teach the scorer to manufacture convenient feedback.
        feedback = torch.stack((hazards.sigmoid(), confidence), -1).detach()
        refreshed = self.refinement_fusion(torch.cat((evidence, points/16., feedback), -1))
        query = decoded+refreshed+stage_embedding.to(decoded.dtype)
        decoded = self.decode_cached(query, projected, padding, ctx)
        return decoded, self.decode_coordinates(decoded)

    def training_forward(self, x, hist, hmask, threshold):
        """Fixed proposal slots; accepted rows retain their last actual attempt.

        The eager inference path still compacts accepted rows. Here acceptance
        is tensor data, so it cannot change the compiled graph or output shapes.
        Losses use refinement_mask to exclude the unused proposals.
        """
        from .survival_confidence import survival_predictions
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
        out = self.proposal_output(initial, refinements, scores, valid)
        score_context = {key: ctx[key] for key in
                         ('fine', 'deep', 'confidence_projected', 'confidence_padding', 'history_tokens', 'history_padding')}
        if self.cfg.recurrent_refinement_steps:
            score_context.update(fine_fp32=ctx['fine_fp32'], deep_fp32=ctx['deep_fp32'])
        return out, score_context

    def decode_cached(self, query, projected, padding, ctx):
        for layer, kv in zip(self.decoder.layers, projected):
            query = layer.forward_cached(query, kv, padding,
                history=(ctx['history_tokens'], ctx['history_padding']), history_attention=self.history_attention)
        return self.decoder.norm(query)
