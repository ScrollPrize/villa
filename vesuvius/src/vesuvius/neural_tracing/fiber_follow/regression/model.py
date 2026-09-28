"""Direct curve regression with one axial encoder and spatial observation memory."""
from dataclasses import asdict, dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec

ARCHITECTURE = 'axial_fiber_spatial_memory_v2'
TOKEN_STRIDE = (8, 2, 2)  # z, y, x in input samples
TOKEN_OFFSET = (3, 0, 0)  # centre of four stride-two stem positions


@dataclass
class DirectConfig:
    fine: CropSpec = field(default_factory=lambda: CropSpec(depth=120, width=101, behind=48, spacing=.5))
    channels: int = 32
    hidden: int = 128
    heads: int = 4
    layers: int = 4
    decoder_layers: int = 4
    embedding: int = 32
    identity_temperature: float = .1
    activation_checkpointing: bool = False
    n_future: int = 16
    future_step: float = 1.
    n_history: int = 128
    max_recovery_distance: float = 6.
    correction: bool = True
    correction_limit: float = 1.
    correction_steps: int = 2
    memory_slots: int = 16  # recurrent context retained after spatial eviction
    memory_steps: int = 32  # preceding full-crop observations, plus the current head
    memory_stride: int = 4  # reconstructed-history observation spacing in trace voxels
    # The newest memory_grad_steps observations (and the head) backpropagate;
    # older ones are a no-grad burn-in. At least memory_steps: no burn-in.
    memory_grad_steps: int = 32
    # One random encoding per chronological stratum of valid
    # gradient-window history, up to this budget per state. Scale by stratum
    # size for an unbiased encoder gradient. Writes keep gradients. 0: all.
    memory_encoder_grad_steps: int = 4
    spatial_recent: int = 2  # always-readable observations, excluding the current crop
    spatial_archive: int = 8  # additional full observations, oldest evicted first
    spatial_retrieve: int = 2  # older observations read per candidate curve
    trajectory_window: int = 1  # supervised decisions reused within one optimizer update

    def __post_init__(self):
        if isinstance(self.fine, dict):
            self.fine = CropSpec(**self.fine)
        c = self.fine
        if not math.isfinite(self.identity_temperature) or self.identity_temperature <= 0:
            raise ValueError('Identity temperature must be finite and positive')
        if min(c.depth, c.width) < 8 or not 0 <= c.behind < c.depth or not math.isfinite(c.spacing) or c.spacing <= 0:
            raise ValueError('Invalid fine crop')
        if min(self.channels, self.hidden, self.heads, self.layers, self.decoder_layers,
               self.embedding, self.n_future, self.n_history) < 1 or self.hidden % self.heads:
            raise ValueError('Positive dimensions required; hidden must divide by heads')
        if not 0 < self.future_step <= self.max_recovery_distance or not math.isfinite(self.max_recovery_distance):
            raise ValueError('Invalid forward spacing or connection limit')
        if not math.isfinite(self.correction_limit) or self.correction_limit <= 0:
            raise ValueError('Invalid correction limit')
        if not isinstance(self.correction_steps, int) or self.correction_steps < 1:
            raise ValueError('Invalid correction steps')
        if self.n_future*self.future_step > (c.depth-c.behind-1)*c.spacing:
            raise ValueError('Future horizon exceeds fine image')
        if any(not isinstance(v, int) or v < 1 for v in
               (self.memory_slots, self.memory_steps, self.memory_stride, self.memory_grad_steps,
                self.spatial_recent, self.spatial_archive, self.spatial_retrieve)):
            raise ValueError('Memory dimensions and capacities must be positive integers')
        if not isinstance(self.memory_encoder_grad_steps, int) or self.memory_encoder_grad_steps < 0:
            raise ValueError('Memory encoder gradient steps must be a nonnegative integer')
        if self.spatial_retrieve > self.spatial_archive:
            raise ValueError('Spatial retrieval exceeds archive capacity')
        if self.hidden < self.channels:
            raise ValueError('Memory hidden width must cover fine feature channels')
        if not isinstance(self.trajectory_window,int) or self.trajectory_window < 1:
            raise ValueError('Trajectory window must be a positive integer')

    @property
    def recent_history_points(self):
        return self.n_history

    @property
    def lateral_limit(self):
        return (self.fine.width-1)*self.fine.spacing/2

    @property
    def token_shape(self):
        return tuple(math.ceil(n/s) for n,s in zip((self.fine.depth,self.fine.width,self.fine.width),TOKEN_STRIDE))

    def to_dict(self):
        return asdict(self)


def feature_grid(points, crop, shape, stride=1, offset=(0,0,0)):
    """Exact physical coordinates for a feature lattice (stride/offset in zyx)."""
    if isinstance(stride, (int, float)):
        stride = (stride,)*3
    size = points.new_tensor(tuple(reversed(shape)))
    scale = points.new_tensor(tuple(reversed(stride)))*crop.spacing
    origin = points.new_tensor((-(crop.width-1)*crop.spacing/2,
                                -(crop.width-1)*crop.spacing/2, -crop.behind*crop.spacing))
    origin = origin+points.new_tensor(tuple(reversed(offset)))*crop.spacing
    return 2*(points-origin)/(scale*(size-1).clamp_min(1))-1


def crop_support(points, crop):
    """Input support, independent of the coarser contextual-feature lattice."""
    lo = points.new_tensor((-(crop.width-1)*crop.spacing/2,)*2+(-crop.behind*crop.spacing,))
    hi = points.new_tensor(((crop.width-1)*crop.spacing/2,)*2+((crop.depth-1-crop.behind)*crop.spacing,))
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
        self.stem = nn.Sequential(nn.Conv3d(2,c,3,padding=1,bias=False),ResidualConv(c))
        self.down = nn.Sequential(nn.Conv3d(c,2*c,3,stride=2,padding=1,bias=False),ResidualConv(2*c),
                                  nn.Conv3d(2*c,4*c,3,padding=1,bias=False),nn.SiLU(),ResidualConv(4*c))
        self.compress = nn.Conv3d(4*c,h,(4,1,1),stride=(4,1,1))
        self.position = nn.Linear(3,h)
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

    def forward(self, image, *, checkpoint_blocks=True):
        fine = self.stem(image)
        down = self.down(fine)
        down = F.pad(down,(0,0,0,0,0,(-down.shape[2])%4))
        tokens = self.compress(down).permute(0,2,3,4,1)
        tokens = tokens+self.position(self.token_xyz/16).reshape(*self.cfg.token_shape,self.cfg.hidden).to(tokens.dtype)
        for block in self.blocks:
            if checkpoint_blocks and self.cfg.activation_checkpointing and self.training and torch.is_grad_enabled():
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


def to_device(value, device):
    # Pageable host-to-device copies synchronize the stream; pinned ones queue.
    if torch.device(device).type == 'cuda':
        return value.pin_memory().to(device, non_blocking=True)
    return value.to(device)


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.save_for_backward(scale)
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad*ctx.saved_tensors[0].to(grad.dtype), None


def scale_gradient(value, scale):
    """Identity in the forward pass; multiplies the incoming gradient by ``scale``."""
    return _ScaleGradient.apply(value, scale)


def stratified_history(own, budget):
    """Sample chronological valid pairs; weights are inverse inclusion probabilities."""
    n = len(own)
    groups = min(n, budget)
    selected = {}
    for group in range(groups):
        start, stop = group*n//groups, (group+1)*n//groups
        # Use the checkpointed global CPU RNG. Singleton groups need no draw.
        offset = int(torch.randint(stop-start, ())) if stop-start > 1 else 0
        selected[own[start+offset]] = stop-start
    return selected


class DirectFollower(nn.Module):
    """Point queries read full spatial observations and recurrent history."""
    architecture = ARCHITECTURE
    memory_keys = ('slots','anchor','anchor_valid','anchor_position','anchor_frame',
                   'bank','bank_summary','bank_valid','bank_position','bank_frame',
                   'age','position','frame','seen')
    # Bound full-crop activation memory for encodings without/with gradients.
    observation_batch = (8, 1)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c, h = cfg.channels, cfg.hidden
        self.encoder = AxialEncoder(cfg)
        self.embedding = nn.Linear(c, cfg.embedding)
        self.observation_projection = nn.Linear(c, h)
        self.metadata = nn.Sequential(nn.Linear(13, h), nn.SiLU(), nn.Linear(h, h))
        self.role = nn.Parameter(torch.randn(4, h)*.02)  # seed, recent, slots, visible history
        self.initial_slots = nn.Parameter(torch.randn(cfg.memory_slots, h)*.02)
        self.write_norm = nn.LayerNorm(h)
        self.write_attention = nn.MultiheadAttention(h, cfg.heads, dropout=0., batch_first=True)
        self.write_gate = nn.Linear(2*h, h)
        self.write_proposal = nn.Linear(2*h, h)
        nn.init.constant_(self.write_gate.bias, -2.)
        self.query = nn.Sequential(nn.Linear(c+4, h), nn.SiLU(), nn.Linear(h, h))
        self.identity_value = nn.Linear(c, h, bias=False)
        self.identity_null = nn.Parameter(torch.zeros(h))
        # Shared by geometry, confidence, supplied candidates, and per-write probes.
        layer = PathDecoderLayer(h, cfg.heads, 2*h, dropout=0., activation='gelu',
                                 batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.decoder_layers, norm=nn.LayerNorm(h))
        self.coordinates = nn.Linear(h, 2)
        self.correction = nn.Linear(h, 2)
        self.confidence_head = nn.Sequential(nn.Linear(2*h, h), nn.SiLU(), nn.Linear(h, 1))
        self.probe_head = nn.Linear(h, 4)
        for head in (self.coordinates, self.correction):
            nn.init.normal_(head.weight, std=.001)
            nn.init.zeros_(head.bias)
        self.register_buffer('planes', torch.arange(1, cfg.n_future+1).float()*cfg.future_step, persistent=False)

        self.spatial_count = math.prod(cfg.token_shape)
        # Slot zero is the current head. It is excluded when decoding that head.
        self.capacity = 1+cfg.spatial_recent+cfg.spatial_archive
        self.spatial_position = nn.Linear(3, h, bias=False)
        self.summary_query = nn.Parameter(torch.randn(1, h)*.02)
        self.summary_attention = nn.MultiheadAttention(h, cfg.heads, batch_first=True)
        self.retrieval_query = nn.Linear(h, h, bias=False)
        self.retrieval_key = nn.Linear(h, h, bias=False)
        self.archive_attention = nn.MultiheadAttention(h, cfg.heads, batch_first=True)
        self.retrieval_norm = nn.LayerNorm(h)

    def encode(self, image, *, checkpoint_blocks=True):
        # Appearance is independent of the claimed identity for every observation.
        return self.encoder(image, checkpoint_blocks=checkpoint_blocks)

    def observation(self, image):
        # The caller checkpoints this entire operation. Nested block checkpoints
        # would recompute the axial stack a second time during backward.
        dense, _, tokens = self.encode(image, checkpoint_blocks=False)
        return self.observation_features(dense, tokens)

    def observation_features(self, dense, tokens):
        # One contextual fine descriptor identifies the observed head; the grid
        # supplies its surroundings. Keep the supervised identity feature space.
        fine,_ = sample_features(dense, dense.new_zeros(len(dense),1,3), self.cfg.fine)
        return torch.cat((tokens, F.pad(fine, (0,self.cfg.hidden-self.cfg.channels))),1)

    def head_descriptor(self, observation):
        return observation[...,self.spatial_count:,:self.cfg.channels]

    def seed_descriptor(self, state):
        return self.head_descriptor(state['anchor'])

    def candidate_features(self, dense, points):
        return sample_features(dense, points, self.cfg.fine)

    def initial_memory(self, batch, device):
        h, n, cap = self.cfg.hidden, self.spatial_count+1, self.capacity
        # Use parameter precision for storage, regardless of whether the state
        # is created inside (training) or outside (tracing) an autocast context.
        # Otherwise the same observation is rounded differently in the two paths.
        dtype = self.initial_slots.dtype
        zeros = lambda *s: torch.zeros(batch,*s,device=device)
        features = lambda *s: torch.zeros(batch,*s,device=device,dtype=dtype)
        return dict(slots=self.initial_slots.to(device)[None].expand(batch,-1,-1).clone(),
            anchor=features(n,h),anchor_valid=zeros().bool(),anchor_position=zeros(3),
            anchor_frame=torch.eye(3,device=device)[None].expand(batch,-1,-1).clone(),
            bank=features(cap,n,h),bank_summary=features(cap,h),bank_valid=zeros(cap).bool(),
            bank_position=zeros(cap,3),bank_frame=torch.eye(3,device=device)[None,None].expand(batch,cap,-1,-1).clone(),
            age=zeros(cap),position=zeros(3),frame=torch.eye(3,device=device)[None].expand(batch,-1,-1).clone(),
            seen=zeros().bool())

    def spatial_tokens(self, observation):
        fine = self.observation_projection(self.head_descriptor(observation))
        return torch.cat((observation[...,:self.spatial_count,:],fine),-2)

    def located(self, observation, positions, frames, position, frame, age, role):
        """Place full observation grids in the current query's coordinate frame."""
        # [B,O,N,H], with explicit observation origin, orientation and age.
        features = self.spatial_tokens(observation)
        xyz = torch.cat((self.encoder.token_xyz,self.encoder.token_xyz.new_zeros(1,3)),0)
        world_offset = torch.einsum('ni,boji->bonj',xyz,frames)
        local_offset = torch.einsum('boni,bij->bonj',world_offset,frame)/16
        pose = self.pose(positions,frames,position,frame,age)
        return features+self.spatial_position(local_offset)+pose[:,:,None]+role

    def pose(self, positions, frames, current_pos, current_frame, age):
        delta = torch.einsum('bni,bij->bnj', positions-current_pos[:,None], current_frame)/16
        delta = delta.sign()*torch.log1p(delta.abs())
        rotation = torch.einsum('bji,bnjk->bnik', current_frame, frames).flatten(-2)
        return self.metadata(torch.cat((delta, rotation, torch.log1p(age.clamp_min(0))[...,None]/8),-1))

    def retained(self, state, position, frame):
        # Only explicit head descriptors participate in pointwise InfoNCE space.
        refs = torch.cat((self.seed_descriptor(state),self.head_descriptor(state['bank']).squeeze(-2)),1)
        valid = torch.cat((state['anchor_valid'][:,None],state['bank_valid']),1)
        roles = torch.cat((self.role[0:1],self.role[1:2].expand(self.capacity,-1)),0)
        return dict(state=state,position=position,frame=frame,start=1),refs,valid,roles

    def assemble_context(self, context, image_tokens, visible, mask):
        context = dict(context)
        context['current'] = torch.cat((image_tokens,visible),1)
        context['current_valid'] = torch.cat((torch.ones(image_tokens.shape[:2],device=mask.device,dtype=torch.bool),mask),1)
        return context

    def visible_features(self, local, references):
        meta = references.new_zeros(*references.shape[:2],13)
        meta[...,:3] = references/16
        meta[:,:-1,-1] = torch.arange(1,self.cfg.n_history+1,device=local.device).float().log1p()/8
        return self.observation_projection(local)+self.metadata(meta)+self.role[3]

    def identity_read(self, values, refs, valid, roles):
        query = F.normalize(self.embedding(values).float(),dim=-1)
        keys = F.normalize(self.embedding(refs).float(),dim=-1)
        logits = torch.bmm(query,keys.transpose(1,2))/self.cfg.identity_temperature
        logits = logits.masked_fill(~valid[:,None],-torch.inf)
        logits = torch.cat((logits,logits.new_zeros(*logits.shape[:2],1)),-1)
        v = self.identity_value(refs)+roles[None]
        v = torch.cat((v,self.identity_null[None,None].expand(len(v),1,-1)),1)
        return torch.bmm(logits.softmax(-1).to(v.dtype),v)

    def encode_observation(self, image, device, gradients):
        image = image.to(device, non_blocking=True)
        if gradients and self.training and torch.is_grad_enabled():
            # Retain only the spatial grid and fine head descriptor between observations.
            return checkpoint(self.observation, image, use_reentrant=False)
        return self.observation(image)

    def schedule(self, x, state=None, probe_mask=None):
        """CPU scheduling metadata, copied once per forward before any encoding."""
        seed = x['memory_seed_valid'].detach().cpu().bool()
        if state is not None:
            seed = seed & ~state['anchor_valid'].detach().cpu()
        return dict(active=x['memory_mask'].detach().cpu().bool(), seed=seed,
                    probe=None if probe_mask is None else probe_mask.detach().cpu().bool())

    def encode_observations(self, crops, device, gradients):
        """Encode independent full crops (CPU views) in bounded batches.

        An observation depends only on its own crop, so batching changes values
        only by kernel rounding. Pinned loader views copy asynchronously.
        """
        parts = []
        size = self.observation_batch[bool(gradients)]
        for i in range(0, len(crops), size):
            images = torch.stack([c.to(device, non_blocking=True) for c in crops[i:i+size]])
            parts.append(self.encode_observation(images, device, gradients))
        return torch.cat(parts)

    def observe(self, x, current, state=None, probe_mask=None, schedule=None):
        b, t = x['memory_mask'].shape
        device = current.device
        schedule = self.schedule(x, state, probe_mask) if schedule is None else schedule
        state = self.initial_memory(b,device) if state is None else state
        active_cpu = schedule['active']
        burn = max(0,t-self.cfg.memory_grad_steps-1)
        # Inactive (state, step) pairs never reach memory, so they are not encoded.
        seeds = schedule['seed'].nonzero()[:,0].tolist()
        history = [(i,j) for j in range(t-1) for i in range(b) if active_cpu[i,j]]
        old = [(i,j) for i,j in history if j < burn]
        new = [(i,j) for i,j in history if j >= burn]
        k = self.cfg.memory_encoder_grad_steps
        detached, scale = [], None
        if k and self.training and torch.is_grad_enabled():
            # Sampled from the seeded global generator, so runs stay reproducible.
            keep = {}
            for i in range(b):
                own = [pair for pair in new if pair[0] == i]
                keep.update(stratified_history(own, k))
            detached = [pair for pair in new if pair not in keep]
            new = [pair for pair in new if pair in keep]
            scale = torch.tensor([keep[pair] for pair in new], dtype=torch.float32)
        index = lambda pairs: tuple(to_device(torch.tensor(v),device) for v in zip(*pairs))
        observed = current.new_zeros(b,max(t-1,0),*current.shape[1:])
        if old:
            with torch.no_grad():
                encoded = self.encode_observations([x['history_crops'][i,j] for i,j in old],device,False)
            observed = observed.index_put(index(old),encoded)
        if detached:
            # Gradient-window writes still backpropagate; this encoder input does not.
            with torch.no_grad():
                encoded = self.encode_observations([x['history_crops'][i,j] for i,j in detached],device,False)
            observed = observed.index_put(index(detached),encoded)
        if seeds or new:
            encoded = self.encode_observations([x['seed_crop'][i] for i in seeds]+
                                               [x['history_crops'][i,j] for i,j in new],device,True)
            if new:
                encoded_new = encoded[len(seeds):]
                if scale is not None:
                    encoded_new = scale_gradient(encoded_new,to_device(scale,device)[:,None,None])
                observed = observed.index_put(index(new),encoded_new)
            if seeds:
                valid_seed = torch.zeros(b,dtype=torch.bool)
                valid_seed[seeds] = True
                valid_seed = to_device(valid_seed,device)
                seed = current.new_zeros(b,*current.shape[1:]).index_put(index([(i,) for i in seeds]),encoded[:len(seeds)])
                state = dict(state)
                state['anchor'] = torch.where(valid_seed[:,None,None],seed,state['anchor'])
                state['anchor_position'] = torch.where(valid_seed[:,None],x['memory_seed_position'],state['anchor_position'])
                state['anchor_frame'] = torch.where(valid_seed[:,None,None],x['memory_seed_frame'],state['anchor_frame'])
                state['anchor_valid'] = state['anchor_valid'] | valid_seed
        active_steps = active_cpu.any(0).tolist()
        probe_steps = [self.training]*t if schedule['probe'] is None else schedule['probe'].any(0).tolist()
        # Pack time-major once, instead of launching tiny contiguous copies at
        # every write. Each step now has the same layout as the current head.
        observed = observed.transpose(0,1).contiguous()
        active_rows, position_rows, frame_rows = (
            x[key].transpose(0,1).contiguous()
            for key in ('memory_mask','memory_positions','memory_frames'))
        probes = []
        for j in range(t):
            active = active_rows[j].bool()
            if not active_steps[j]:
                probes.append(current.new_zeros(b,4))
                continue
            with torch.set_grad_enabled(torch.is_grad_enabled() and j >= burn):
                obs = current if j == t-1 else observed[j]
                state,probe = self.write(obs,active,position_rows[j],frame_rows[j],state,
                                        probe=self.training and j >= burn and probe_steps[j])
            probes.append(probe)
        return state,torch.stack(probes,1)

    def write(self, obs, active, position, frame, state, *, probe=True):
        return self._write(obs, active, position, frame, state, probe=probe)

    def write(self, obs, active, position, frame, state, *, probe=True):
        position = torch.where(active[:,None],position,state['position'])
        frame = torch.where(active[:,None,None],frame,state['frame'])
        # Mask before any operation: inactive padding can contain NaN.
        obs = torch.where(active[:,None,None],obs,0.)
        observed = self.spatial_tokens(obs)
        seed = self.observation_projection(self.seed_descriptor(state))
        seed = seed*state['anchor_valid'][:,None,None]
        seed = seed+self.pose(state['anchor_position'][:,None],state['anchor_frame'][:,None],
            position,frame,position.new_zeros(len(obs),1))*state['anchor_valid'][:,None,None]
        previous_pose = self.pose(state['position'][:,None],state['frame'][:,None],position,frame,position.new_zeros(len(obs),1))
        previous_pose = previous_pose*state['seen'][:,None,None]
        evidence = self.write_attention(self.write_norm(state['slots'])+seed+previous_pose,observed,observed,
                                        need_weights=False)[0]
        pair = torch.cat((state['slots'],evidence),-1)
        slots = state['slots']+self.write_gate(pair).sigmoid()*(self.write_proposal(pair).tanh()-state['slots'])
        head = self.observation_projection(self.head_descriptor(obs))
        summary = self.summary_attention(self.summary_query[None]+head,observed,observed,need_weights=False)[0][:,0]

        result = obs.new_zeros(len(obs),4)
        if probe:
            # Evaluate against pre-write evidence: no self-match to a just-added
            # descriptor. This uses exactly the path evaluator used at inference.
            context,refs,valid,roles = self.retained(state,position,frame)
            context.update(start=0,current=observed,current_valid=torch.ones(observed.shape[:2],device=obs.device,dtype=torch.bool))
            decoded = self.evaluate(obs.new_zeros(len(obs),1,3),self.head_descriptor(obs),
                torch.ones(len(obs),1,device=obs.device,dtype=torch.bool),context,refs,valid,roles)
            result = self.probe_head(decoded[:,0]).float()
        updated = dict(state)
        updated['slots'] = torch.where(active[:,None,None],slots,state['slots'])
        for key,value in (('bank',obs),('bank_summary',summary),('bank_position',position),('bank_frame',frame)):
            shifted = torch.cat((value[:,None].to(state[key].dtype),state[key][:,:-1]),1)
            updated[key] = torch.where(active.reshape(len(active),*([1]*(shifted.ndim-1))),shifted,state[key])
        updated['bank_valid'] = torch.where(active[:,None],torch.cat((active[:,None],state['bank_valid'][:,:-1]),1),state['bank_valid'])
        updated['age'] = torch.where(active[:,None],torch.cat((state['age'].new_zeros(len(obs),1),state['age'][:,:-1]+1),1),state['age'])
        updated.update(position=position,frame=frame,seen=state['seen'] | active)
        return updated,result

    def _decode_spatial(self, query, context):
        state,position,frame = (context[k] for k in ('state','position','frame'))
        b = len(query)
        start = context['start']
        recent = slice(start,start+self.cfg.spatial_recent)
        archive = slice(start+self.cfg.spatial_recent,start+self.cfg.spatial_recent+self.cfg.spatial_archive)
        seed = self.located(state['anchor'][:,None],state['anchor_position'][:,None],state['anchor_frame'][:,None],
            position,frame,position.new_zeros(b,1),self.role[0])[:,0]
        nearby = self.located(state['bank'][:,recent],state['bank_position'][:,recent],state['bank_frame'][:,recent],
            position,frame,state['age'][:,recent],self.role[1]).flatten(1,2)
        slots = state['slots']+self.role[2]+self.pose(state['position'][:,None],state['frame'][:,None],
            position,frame,position.new_zeros(b,1))
        tokens = torch.cat((context['current'],seed,nearby,slots),1)
        n = self.spatial_count+1
        valid = torch.cat((context['current_valid'],state['anchor_valid'][:,None].expand(-1,n),
            state['bank_valid'][:,recent,None].expand(-1,-1,n).flatten(1),
            state['seen'][:,None].expand(-1,self.cfg.memory_slots)),1)

        # Cheap, seed-conditioned routing. Each path point has its own weights;
        # the candidate reads the union approximated by its K highest-scoring
        # observation blocks. Stable ordering resolves ties toward newer entries.
        seed_query = self.observation_projection(self.seed_descriptor(state))*state['anchor_valid'][:,None,None]
        query = query+seed_query
        summaries = state['bank_summary'][:,archive]+self.pose(state['bank_position'][:,archive],
            state['bank_frame'][:,archive],position,frame,state['age'][:,archive])+self.role[2]
        scores = torch.bmm(self.retrieval_query(query).float(),self.retrieval_key(summaries).float().transpose(1,2))/math.sqrt(self.cfg.hidden)
        archive_valid = state['bank_valid'][:,archive]
        scores = scores.masked_fill(~archive_valid[:,None],-torch.inf)
        # Null observation ensures an empty archive is finite and may be ignored.
        weights = torch.cat((scores,scores.new_zeros(b,query.shape[1],1)),-1).softmax(-1)[...,:-1]
        query = query+torch.bmm(weights.to(summaries.dtype),summaries)
        selected = scores.amax(1).argsort(dim=-1,descending=True,stable=True)[:,:self.cfg.spatial_retrieve]
        row = torch.arange(b,device=query.device)
        reads = torch.zeros_like(query)
        for k in range(self.cfg.spatial_retrieve):
            index = selected[:,k]+start+self.cfg.spatial_recent
            feature = self.located(state['bank'][row,index][:,None],state['bank_position'][row,index][:,None],
                state['bank_frame'][row,index][:,None],position,frame,state['age'][row,index][:,None],self.role[2])[:,0]
            read = self.archive_attention(self.retrieval_norm(query),feature,feature,need_weights=False)[0]
            weight = weights.gather(2,selected[:,k,None,None].expand(-1,query.shape[1],1))
            reads = reads+read*weight.to(read.dtype)
        return self.decoder(query+reads,tokens,memory_key_padding_mask=~valid)

    def evaluate(self, points, features, support, context, refs, ref_valid, roles):
        query = self.query(torch.cat((features,support[...,None],points/16),-1))
        query = query+self.identity_read(features,refs,ref_valid,roles)
        if self.training and self.cfg.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(self._decode_spatial,query,context,use_reentrant=False)
        return self._decode_spatial(query,context)

    def forward(self, x, hist, hmask, queries=None, candidates=None, memory=None, probe_mask=None,
                candidate_mask=None):
        # Synchronize for scheduling before queueing GPU work, not behind it.
        schedule = self.schedule(x, memory, probe_mask)
        # Training-only work scheduling, never an input to the evaluator. Keep
        # whole curves whenever any point is labeled (including metric points).
        score_candidates = None
        if candidates is not None and candidate_mask is not None:
            if candidate_mask.shape != candidates.shape[:-1]:
                raise ValueError('Candidate scheduling mask must match candidate points')
            score_candidates = candidate_mask.detach().cpu().bool().any(dim=(0,2)).tolist()
        dense, _, image_tokens = self.encode(x['fine'])
        current = self.observation_features(dense, image_tokens)
        # Write the observed head once. Hypothetical candidate scoring never writes.
        state,probes = self.observe(x,current,memory,probe_mask,schedule)
        context,refs,ref_valid,roles = self.retained(state,state['position'],state['frame'])
        references = torch.cat((hist,x['seed']),1)
        mask = torch.cat((hmask.bool(),x['seed_mask'].bool()),1) & crop_support(references,self.cfg.fine)
        references = torch.where(mask[...,None],references,0.)
        local,_ = sample_features(dense,references,self.cfg.fine)
        local = torch.where(mask[...,None],local,0.)
        # The seed identity is always its immutable encoded observation, whether
        # it remains visible or not; visible references retain spatial context.
        reference_embedding = F.normalize(self.embedding(local).float(),dim=-1)
        reference_embedding = torch.cat((reference_embedding[:,:-1],
            F.normalize(self.embedding(self.seed_descriptor(state)).float(),dim=-1)),1)
        reference_mask = torch.cat((mask[:,:-1],state['anchor_valid'][:,None]),1)
        visible = self.visible_features(local,references)
        context = self.assemble_context(context,image_tokens,visible,mask)
        refs = torch.cat((refs,local[:,:-1]),1)
        ref_valid = torch.cat((ref_valid,mask[:,:-1]),1)
        roles = torch.cat((roles,self.role[3:4].expand(self.cfg.n_history,-1)),0)

        def evaluate(points):
            values,support = self.candidate_features(dense,points)
            return self.evaluate(points,values,support,context,refs,ref_valid,roles)

        points = hist.new_zeros(len(hist),self.cfg.n_future,3)
        points[...,2] = self.planes
        decoded = evaluate(points)
        lateral = self.cfg.lateral_limit*self.coordinates(decoded).float().tanh()
        points = torch.cat((lateral,points[...,2:]),-1)
        initial = points
        refinements = [points]
        if self.cfg.correction:
            for _ in range(self.cfg.correction_steps):
                decoded = evaluate(points)
                delta = self.correction(decoded).float().tanh()*(self.cfg.correction_limit/math.sqrt(2))
                lateral = (points[...,:2]+delta).clamp(-self.cfg.lateral_limit,self.cfg.lateral_limit)
                points = torch.cat((lateral,points[...,2:]),-1)
                refinements.append(points)

        def confidence(curve):
            evidence = evaluate(curve.detach())
            count = torch.arange(1,curve.shape[1]+1,device=curve.device)[None,:,None]
            summary = torch.cat((evidence.cumsum(1)/count,evidence.cummax(1).values),-1)
            return self.confidence_head(summary).squeeze(-1).float()

        logits = confidence(points)
        out = dict(points=points,initial_points=initial,refinement_points=torch.stack(refinements,1),
            confidence_logits=logits,confidence=logits.sigmoid().cummin(-1).values,
            reference_embedding=reference_embedding,reference_mask=reference_mask,
            memory_probe=probes,**{'memory_'+k:v for k,v in state.items()})
        if candidates is not None:
            out['candidate_confidence_logits'] = torch.stack([
                confidence(c) if score_candidates is None or score_candidates[j] else logits.new_zeros(logits.shape)
                for j,c in enumerate(candidates.unbind(1))],1)
        if queries is not None:
            features,support = sample_features(dense,queries,self.cfg.fine)
            out.update(query_embedding=F.normalize(self.embedding(features).float(),dim=-1),query_support=support)
        return out
