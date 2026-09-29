"""Crop-only fiber following: residual 3-D stem, 8x2x2 tokens, full axial attention.

All physical coordinates are trace-grid voxels; CT samples are half a trace voxel.
Bare DirectConfig() retains the legacy crop-only constructor default. Training
launchers explicitly enable memory, and v4 requires it. V2/v3 memory uses a
small observation encoder; v4 retains main-encoder features. Both memory designs
carry bounded state and an immutable seed.
"""
from dataclasses import asdict, dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec

ARCHITECTURE = 'axial_fiber_v3'
MEMORY_ARCHITECTURE = 'axial_fiber_memory_v2'
MEMORY_ARCHITECTURE_V1 = 'axial_fiber_memory_v1'  # no probe; loadable, not trained further
SPATIAL_MEMORY_ARCHITECTURE = 'axial_fiber_memory_v3'
TRAJECTORY_MEMORY_ARCHITECTURE = 'axial_fiber_memory_v4'
TOKEN_STRIDE = (8, 2, 2)  # z, y, x in input samples
TOKEN_OFFSET = (3, 0, 0)  # centre of four stride-two stem positions
IDENTITY_EVIDENCE_WIDTH = 8  # point/mean/min/coverage for seed and history separately


@dataclass
class DirectConfig:
    direction_inputs: bool = False  # six local-frame second moments after CT/presence
    fine: CropSpec = field(default_factory=lambda: CropSpec(depth=120, width=101, behind=48, spacing=.5))
    channels: int = 32
    hidden: int = 128
    heads: int = 4
    layers: int = 4
    decoder_layers: int = 4
    embedding: int = 32
    activation_checkpointing: bool = False
    n_future: int = 16
    future_step: float = 1.
    n_history: int = 128
    max_recovery_distance: float = 6.
    patch_radius: float = 1.
    correction: bool = True
    correction_limit: float = 1.
    correction_steps: int = 2
    memory_slots: int = 0  # zero preserves the crop-only architecture and weights
    memory_steps: int = 32  # preceding observed patches, plus the current head
    memory_stride: int = 4  # reconstructed-history observation spacing in trace voxels
    memory_patch_size: int = 17  # 8-voxel-wide patch at the production .5 spacing
    # The newest memory_grad_steps observations (and the head) backpropagate;
    # older ones are a no-grad burn-in. At least memory_steps: no burn-in.
    memory_grad_steps: int = 32
    memory_version: int = 2  # 1: legacy checkpoints without the probe
    route_grid_step: float = 2.  # v3: maximum lateral lattice spacing, trace voxels
    route_transition_radius: int = 1  # v3: connected lattice neighbors per plane
    route_transition_cost: float = .25
    route_loss_weight: float = 1.
    route_sequence_weight: float = .5  # v3: additional earlier decision from an observed track
    route_refinement_radius: float | None = None  # v3: per-axis displacement from selected cell; None keeps half-cell bound
    trajectory_sequence_weight: float = 0.  # obsolete v4 prefix-reconstruction objective
    feature_memory_grid: tuple = (2, 4, 4)  # v4 spatial tokens per encoded crop
    feature_sequence_length: int = 2  # v4 decisions per gradient chunk
    feature_memory_revision: int = 1

    def __post_init__(self):
        if not isinstance(self.direction_inputs, bool):
            raise ValueError('direction_inputs must be a boolean')
        if isinstance(self.fine, dict):
            self.fine = CropSpec(**self.fine)
        c = self.fine
        if min(c.depth, c.width) < 8 or not 0 <= c.behind < c.depth or not math.isfinite(c.spacing) or c.spacing <= 0:
            raise ValueError('Invalid fine crop')
        if min(self.channels, self.hidden, self.heads, self.layers, self.decoder_layers,
               self.embedding, self.n_future, self.n_history) < 1 or self.hidden % self.heads:
            raise ValueError('Positive dimensions required; hidden must divide by heads')
        if not 0 < self.future_step <= self.max_recovery_distance or not math.isfinite(self.max_recovery_distance):
            raise ValueError('Invalid forward spacing or connection limit')
        if not 0 < self.patch_radius < (c.width-1)*c.spacing/2:
            raise ValueError('Local observation patch must fit fine crop')
        if not math.isfinite(self.correction_limit) or self.correction_limit <= 0:
            raise ValueError('Invalid correction limit')
        if self.route_refinement_radius is not None and (not math.isfinite(self.route_refinement_radius) or self.route_refinement_radius <= 0):
            raise ValueError('Route refinement radius must be finite and positive')
        if not isinstance(self.correction_steps, int) or self.correction_steps < 1:
            raise ValueError('Invalid correction steps')
        if self.n_future*self.future_step > (c.depth-c.behind-1)*c.spacing:
            raise ValueError('Future horizon exceeds fine image')
        if not isinstance(self.memory_slots, int) or self.memory_slots < 0:
            raise ValueError('Memory slots must be a nonnegative integer')
        if self.memory_slots:
            if any(not isinstance(v, int) or v < 1 for v in (self.memory_steps, self.memory_stride, self.memory_grad_steps)):
                raise ValueError('Memory sequence dimensions must be positive integers')
            if self.memory_version not in (1, 2, 3, 4):
                raise ValueError('Unknown memory version')
            if not isinstance(self.memory_patch_size, int) or self.memory_patch_size < 5 or self.memory_patch_size % 2 != 1:
                raise ValueError('Memory patch size must be odd and at least five')
        if self.memory_version == 3:
            if not self.memory_slots:
                raise ValueError('Spatial memory requires positive memory_slots')
            if not math.isfinite(self.route_grid_step) or self.route_grid_step <= 0:
                raise ValueError('Route grid spacing must be finite and positive')
            if not isinstance(self.route_transition_radius, int) or self.route_transition_radius < 1:
                raise ValueError('Route transition radius must be a positive integer')
            if any(not math.isfinite(v) or v < 0 for v in (self.route_transition_cost, self.route_loss_weight, self.route_sequence_weight)):
                raise ValueError('Route cost and loss weight must be finite and nonnegative')
        if self.memory_version == 4:
            if not self.memory_slots:
                raise ValueError('Continuous trajectory memory requires positive memory_slots')
            if self.correction or self.route_refinement_radius is not None:
                raise ValueError('Memory v4 uses one decoder pass; use --no-correction and no route refinement radius')
            if self.trajectory_sequence_weight != 0:
                raise ValueError('V4 supervises every streamed decision; trajectory_sequence_weight must be zero')
            self.feature_memory_grid = tuple(self.feature_memory_grid)
            if len(self.feature_memory_grid) != 3 or any(not isinstance(n, int) or n < 1 for n in self.feature_memory_grid):
                raise ValueError('Feature memory grid requires three positive integers')
            if not isinstance(self.feature_sequence_length, int) or self.feature_sequence_length < 2:
                raise ValueError('Feature sequence length must be at least two')
            if self.feature_memory_revision != 1:
                raise ValueError('Unsupported feature memory revision')

    @property
    def sequence_weight(self):
        return {3: self.route_sequence_weight}.get(self.memory_version, 0.)

    @property
    def sequence_key(self):
        return 'trajectory_sequence' if self.memory_version == 4 else 'route_sequence'

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
    def token_shape(self):
        return tuple(math.ceil(n/s) for n,s in zip((self.fine.depth,self.fine.width,self.fine.width),TOKEN_STRIDE))

    def to_dict(self):
        return asdict(self)


def build_model(cfg):
    """Explicit architecture dispatch; legacy constructors and weights stay intact."""
    if cfg.memory_slots and cfg.memory_version == 4:
        from .trajectory_model import TrajectoryMemoryFollower
        return TrajectoryMemoryFollower(cfg)
    if cfg.memory_slots and cfg.memory_version == 3:
        from .spatial_model import SpatialMemoryFollower
        return SpatialMemoryFollower(cfg)
    return DirectFollower(cfg)


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
    def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal=False):
        if x.is_cuda:
            # Flash does not support this padding mask in the installed PyTorch.
            # Retain the other backends for unsupported devices/dtypes/shapes.
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                              SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True):
                return super()._mha_block(x, mem, attn_mask, key_padding_mask, is_causal)
        return super()._mha_block(x, mem, attn_mask, key_padding_mask, is_causal)


class AxialBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.axes = nn.ModuleList(AxisAttention(width,heads,a) for a in (3,2,1))
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
        self.local = nn.Sequential(DepthwiseConv3d(width,width,3,padding=1,groups=width),
                                   nn.SiLU(),nn.Conv3d(width,width,1))

    def forward(self, x):
        for axis in self.axes:
            x = axis(x)
        x = x+self.mlp(self.norm(x))
        return x+self.local(x.permute(0,4,1,2,3)).permute(0,2,3,4,1)


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
        d,y,x = torch.meshgrid(*(torch.arange(n).float() for n in cfg.token_shape),indexing='ij')
        xyz = torch.stack((2*x,2*y,8*d+3),-1)*cfg.fine.spacing
        xyz -= xyz.new_tensor(((cfg.fine.width-1)*cfg.fine.spacing/2,)*2+(cfg.fine.behind*cfg.fine.spacing,))
        self.register_buffer('token_xyz',xyz.reshape(-1,3),persistent=False)
        d,y,x = torch.meshgrid(torch.arange(cfg.fine.depth).float(),
            torch.arange(cfg.fine.width).float(),torch.arange(cfg.fine.width).float(),indexing='ij')
        points = torch.stack((x,y,d),-1)*cfg.fine.spacing
        points -= points.new_tensor(((cfg.fine.width-1)*cfg.fine.spacing/2,)*2+(cfg.fine.behind*cfg.fine.spacing,))
        self.register_buffer('decode_grid',feature_grid(points,cfg.fine,cfg.token_shape,TOKEN_STRIDE,TOKEN_OFFSET)[None],persistent=False)

    def conditioning(self, references, mask):
        cfg = self.cfg
        points = torch.where(mask[...,None],references,0.).float()
        origin = device_vector(points, (-(cfg.fine.width-1)*cfg.fine.spacing/2,)*2+(-cfg.fine.behind*cfg.fine.spacing,))
        offset = device_vector(points, (0,0,3))*cfg.fine.spacing
        index = torch.round((points-origin-offset)/(device_vector(points, (2,2,8))*cfg.fine.spacing)).long()
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

    def forward(self, image, references, mask):
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
        low = self.dense_projection(deep)
        up = F.grid_sample(low.float(),self.decode_grid.expand(len(image),-1,-1,-1,-1),
                           padding_mode='border',align_corners=True).to(fine.dtype)
        dense = fine+self.dense_decoder(torch.cat((fine,up),1))
        return dense, deep, tokens.flatten(1,3)


class DirectFollower(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, cfg):
        super().__init__()
        if cfg.memory_version in (3, 4):
            raise ValueError('Use build_model(cfg) for memory v3/v4')
        self.cfg = cfg
        c,h = cfg.channels,cfg.hidden
        self.encoder = AxialEncoder(cfg)
        self.embedding = nn.Linear(c,cfg.embedding)
        self.reference_token = nn.Sequential(nn.Linear(c+8,h),nn.SiLU(),nn.Linear(h,h))
        self.query = nn.Sequential(nn.Linear(9*c+1,h),nn.SiLU(),nn.Linear(h,h))
        layer = PathDecoderLayer(h,cfg.heads,2*h,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.decoder = nn.TransformerDecoder(layer,cfg.decoder_layers,norm=nn.LayerNorm(h))
        self.coordinates = nn.Linear(h,2)
        nn.init.normal_(self.coordinates.weight,std=.005)
        nn.init.zeros_(self.coordinates.bias)
        evidence_width = 27*(c+1)+h+1
        def path_layers():
            return [nn.Linear(h+evidence_width+3,h),nn.SiLU(),nn.TransformerEncoderLayer(
                h,cfg.heads,2*h,dropout=0.,activation='gelu',batch_first=True,norm_first=True)]
        if cfg.correction:
            self.correction_head = nn.Sequential(*path_layers(),nn.Linear(h,2))
            nn.init.normal_(self.correction_head[-1].weight,std=.001)
            nn.init.zeros_(self.correction_head[-1].bias)
        self.path_evidence = nn.Sequential(*path_layers())
        self.confidence_head = nn.Sequential(nn.Linear(2*h+IDENTITY_EVIDENCE_WIDTH,h),nn.SiLU(),nn.Linear(h,1))
        # Start neutral; candidate/prefix BCE learns how to use the comparison.
        # Zero new columns also permit a lossless one-time checkpoint expansion.
        with torch.no_grad():
            self.confidence_head[0].weight[:,2*h:].zero_()
        self.register_buffer('stencil',torch.tensor([[a,b,0.] for a in (-1.,0.,1.) for b in (-1.,0.,1.)])*cfg.patch_radius,persistent=False)
        self.register_buffer('path_stencil',torch.tensor([[a*cfg.patch_radius,b*cfg.patch_radius,z]
            for z in (-1.,0.,1.) for a in (-1.,0.,1.) for b in (-1.,0.,1.)]),persistent=False)
        self.register_buffer('planes',torch.arange(1,cfg.n_future+1).float()*cfg.future_step,persistent=False)
        if cfg.memory_slots:
            from .memory import LearnedMemory
            self.recurrent_memory = LearnedMemory(cfg)
            self.architecture = MEMORY_ARCHITECTURE if cfg.memory_version == 2 else MEMORY_ARCHITECTURE_V1

    def context(self, x, hist, hmask):
        cfg = self.cfg
        seed = x.get('seed',hist.new_zeros(len(hist),1,3))
        seed_mask = x.get('seed_mask',hmask.new_zeros(len(hist),1)).bool()
        references = torch.cat((hist,seed),1)
        mask = torch.cat((hmask.bool(),seed_mask),1) & crop_support(references,cfg.fine)
        references = torch.where(mask[...,None],references.float(),0.)
        dense,deep,image_tokens = self.encoder(x['fine'],references,mask)
        local,_ = sample_features(dense,references,cfg.fine)
        embedded = F.normalize(self.embedding(local),dim=-1)
        embedded = torch.where(mask[...,None],embedded,0.)
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
        return dict(fine=dense,deep=deep,memory=memory,padding=padding,
                    reference_embedding=embedded,reference_mask=mask)

    def patches(self, fine, points):
        b,k,_ = points.shape
        values,_ = sample_features(fine,(points[:,:,None]+self.stencil).reshape(b,k*9,3),self.cfg.fine)
        return values.reshape(b,k,-1)

    def query_features(self, ctx, initial):
        return torch.cat((self.patches(ctx['fine'],initial),initial[...,2:]/(self.cfg.n_future*self.cfg.future_step)),-1)

    def evidence(self, ctx, points, stage):
        b,k,_ = points.shape
        local,support = sample_features(ctx['fine'],(points[:,:,None]+self.path_stencil).reshape(b,k*27,3),self.cfg.fine)
        local = torch.cat((local,support[...,None]),-1).reshape(b,k,-1)
        deep,valid = sample_features(ctx['deep'],points,self.cfg.fine,TOKEN_STRIDE,TOKEN_OFFSET)
        return torch.cat((local,deep,valid[...,None]),-1)

    def forward(self, x, hist, hmask, queries=None, candidates=None, memory=None):
        ctx = self.context(x,hist,hmask)
        if self.cfg.memory_slots:
            ctx['recurrent'] = self.recurrent_memory.observe(x, memory)
        elif memory is not None:
            raise ValueError('Persistent memory requires a memory-enabled checkpoint')
        out = self.predict(ctx,hist,candidates)
        if self.cfg.memory_slots:
            out.update({'memory_'+k: v for k,v in ctx['recurrent'].items()})
        out.update(reference_embedding=ctx['reference_embedding'],reference_mask=ctx['reference_mask'])
        if queries is not None:
            values,support = sample_features(ctx['fine'],queries,self.cfg.fine)
            out.update(query_embedding=F.normalize(self.embedding(values),dim=-1),query_support=support)
        return out

    def initial_memory(self, batch, device):
        if not self.cfg.memory_slots:
            raise ValueError('This checkpoint has no persistent memory')
        return self.recurrent_memory.initial_state(batch, device)

    def identity_evidence(self, ctx, values, support):
        """Observed-reference cosine evidence for each prefix, without GT filtering.

        Keep the seed separate from potentially contaminated recent history.
        Missing references/unsupported samples have zero evidence and explicit
        coverage, so missing information is distinguishable from a poor match.
        """
        candidate = F.normalize(self.embedding(values).float(),dim=-1)
        refs,mask = ctx['reference_embedding'].float(),ctx['reference_mask'].bool()
        history = F.normalize((refs[:,:-1]*mask[:,:-1,None]).sum(1),dim=-1)
        seed = F.normalize(refs[:,-1],dim=-1)
        anchors = torch.stack((seed,history),1)
        available = torch.stack((mask[:,-1],mask[:,:-1].any(1)),1)
        valid = support[:,:,None] & available[:,None]
        similarity = (candidate[:,:,None]*anchors[:,None]).sum(-1).clamp(-1.,1.)
        similarity = torch.where(valid,similarity,0.)
        observed = valid.float().cumsum(1)
        mean = similarity.cumsum(1)/observed.clamp_min(1.)
        minimum = similarity.masked_fill(~valid,torch.inf).cummin(1).values
        minimum = torch.where(observed > 0,minimum,0.)
        count = torch.arange(1,values.shape[1]+1,device=values.device)[None,:,None]
        return torch.cat((similarity,mean,minimum,observed/count),-1)

    def confidence_logits(self, ctx, decoded, points):
        """The same prefix classifier scores predictions and supervised candidates."""
        points = points.detach()
        spatial = self.evidence(ctx,points,'confidence')
        evidence = self.path_evidence(torch.cat((decoded,spatial,points/16),-1))
        count = torch.arange(1, points.shape[1]+1, device=points.device)[None, :, None]
        prefix_mean = evidence.cumsum(1)/count
        prefix_max = evidence.cummax(1).values
        # The 3x3x3 evidence stencil already samples the exact candidate centre.
        # Reuse it: another grid_sample would retain a full FP32 crop for backward.
        c = self.cfg.channels
        centre = (len(self.path_stencil)//2)*(c+1)
        comparison = self.identity_evidence(ctx,spatial[...,centre:centre+c],spatial[...,centre+c].bool())
        return self.confidence_head(torch.cat((prefix_mean,prefix_max,comparison),-1)).squeeze(-1).float()

    def predict(self, ctx, hist, candidates=None):
        cfg = self.cfg
        initial = hist.new_zeros(len(hist), cfg.n_future, 3)
        initial[..., 2] = self.planes
        query = self.query(self.query_features(ctx, initial))
        decoded = self.decoder(query, ctx['memory'], memory_key_padding_mask=ctx['padding'])
        if self.cfg.memory_slots:
            decoded = self.recurrent_memory.read(decoded, ctx['recurrent'])
        lateral = cfg.lateral_limit*torch.tanh(self.coordinates(decoded).float())
        points = torch.cat((lateral, initial[..., 2:]), -1)
        initial_points = points
        refinements = [points]
        if cfg.correction:
            # Train the shared refiner through each update; refresh evidence
            # at the new coordinates while encoding the crop only once.
            for _ in range(cfg.correction_steps):
                evidence = self.evidence(ctx, points, 'correction')
                correction = self.correction_head(torch.cat((decoded, evidence, points/16), -1))
                correction = correction.float().tanh()*(cfg.correction_limit/math.sqrt(2))
                lateral = (lateral+correction).clamp(-cfg.lateral_limit, cfg.lateral_limit)
                points = torch.cat((lateral, initial[..., 2:]), -1)
                refinements.append(points)
        # Confidence inspects the actual prediction. Its labels and coordinate
        # sampling are detached; coordinate regression retains its own gradient.
        logits = self.confidence_logits(ctx, decoded, points)
        out = dict(points=points, initial_points=initial_points,
                    refinement_points=torch.stack(refinements, 1),
                    confidence_logits=logits, confidence=logits.sigmoid().cummin(-1).values)
        if candidates is not None:
            out['candidate_confidence_logits'] = torch.stack([
                self.confidence_logits(ctx, decoded, curve) for curve in candidates.unbind(1)], 1)
        return out
