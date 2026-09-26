"""History-conditioned flow proposals with checkpointed sampling and scoring."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from vesuvius.neural_tracing.fiber_follow.encoder import SpatialEncoder, prepare_model
from vesuvius.neural_tracing.fiber_follow.policy import DEFAULT_MAX_RECOVERY_DISTANCE, recovery_allowed
from vesuvius.neural_tracing.fiber_follow.passage_scorer import PassageScorer, sample_path_features

ARCHITECTURE = 'single_path_flow_v11'

@dataclass
class FollowNetConfig:
    in_channels: int = 3
    depth: int = 176
    width: int = 96
    behind: int = 128
    spacing: float = 1.
    widths: tuple = (24, 64, 128)
    hidden: int = 256
    n_future: int = 16
    future_step: float = 1.
    hist_points: int = 128
    hist_stride: int = 1
    recent_history_points: int = 128
    flow_layers: int = 6
    flow_heads: int = 8
    flow_steps: int = 4
    flow_draws: int = 64
    flow_sigma: tuple = ()
    flow_stencil_radius: float = 2.
    max_recovery_distance: float = DEFAULT_MAX_RECOVERY_DISTANCE
    norm: str = 'group'
    # Missing in old checkpoints: retain their original deterministic behavior.
    # New training runs explicitly select gaussian.
    sampler_mode: str = 'zero'
    # Missing fields retain the original single-path confidence head/checkpoints.
    scorer: str = 'legacy'
    gaussian_candidates: int = 0
    selection_horizon: int = 8

    def __post_init__(self):
        self.widths = tuple(self.widths)
        self.flow_sigma = tuple(tuple(row) for row in self.flow_sigma)
        if self.sampler_mode not in ('zero', 'gaussian'):
            raise ValueError('sampler_mode must be zero or gaussian')
        if self.scorer not in ('legacy', 'passage'):
            raise ValueError('scorer must be legacy or passage')
        if not isinstance(self.gaussian_candidates, int) or self.gaussian_candidates < 0 or self.selection_horizon < 1:
            raise ValueError('Invalid candidate count or selection horizon')
        if self.gaussian_candidates and (self.scorer != 'passage' or self.sampler_mode != 'zero'):
            raise ValueError('Gaussian alternatives require scorer=passage and sampler_mode=zero')
        if min(self.hist_points, self.hist_stride, self.n_future, self.flow_layers,
               self.flow_heads, self.flow_steps, self.flow_draws) < 1 or self.hidden % self.flow_heads:
            raise ValueError('Positive dimensions required; hidden must divide by heads')
        if not 1 <= self.recent_history_points <= self.hist_points*self.hist_stride:
            raise ValueError('Recent history must fit supplied geometric history')
        if not math.isfinite(self.max_recovery_distance) or not 0 < self.future_step <= self.max_recovery_distance:
            raise ValueError('Invalid first-connection limit or forward spacing')
        if self.n_future*self.future_step > (self.depth-self.behind-1)*self.spacing:
            raise ValueError('Future horizon exceeds crop')
        if not math.isfinite(self.flow_stencil_radius) or not 0 < self.flow_stencil_radius < (self.width-1)*self.spacing/2:
            raise ValueError('Feature patch must fit crop')

    def to_dict(self):
        return asdict(self)

def history_tangent(points, mask, count):
    """Weighted line fit in arclength order, newest point first.

    Only the contiguous valid prefix is used. Nearer points get more weight;
    fewer than two distinct points supply no measured tangent.
    """
    count = min(count, points.shape[1])
    p = points[:, :count].float()
    valid = mask[:, :count].float().cumprod(-1)
    s = -torch.arange(count, device=p.device, dtype=p.dtype)
    w = valid / (1 + s.abs())
    p = torch.where(valid[..., None] > 0, p, 0.)
    total = w.sum(-1, keepdim=True).clamp_min(1)
    center = (w * s).sum(-1, keepdim=True) / total
    dp = p - (w[..., None] * p).sum(1, keepdim=True) / total[..., None]
    tangent = (w[..., None] * (s - center)[..., None] * dp).sum(1)
    measured = (valid.sum(-1) >= 2) & (tangent.norm(dim=-1) > 1e-6)
    forward = torch.zeros_like(tangent)
    forward[:, 2] = 1
    return torch.where(measured[:, None], F.normalize(tangent, dim=-1), forward), measured


def time_embedding(t, dim):
    """Sinusoidal features of flow time ``t`` in [0, 1], shape (N, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    angles = t.float()[:, None] * 1e3 * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], -1)


def future_planes(cfg, device=None):
    return cfg.future_step * torch.arange(1, cfg.n_future+1, device=device, dtype=torch.float32)


def initial_residuals(cfg, batch, device, generator=None):
    """Primary start and optional Gaussian alternatives; zero stays in slot zero.

    Call outside compiled training methods so RNG consumption is explicit.
    Per-plane physical scales are applied only by PathFlow.to_voxels.
    """
    shape = (batch, 1, cfg.n_future, 2)
    if cfg.gaussian_candidates:
        return torch.cat((torch.zeros(shape, device=device), torch.randn(
            batch, cfg.gaussian_candidates, cfg.n_future, 2, device=device, generator=generator)), 1)
    if cfg.sampler_mode == 'zero':
        return torch.zeros(shape, device=device, dtype=torch.float32)
    return torch.randn(shape, device=device, dtype=torch.float32, generator=generator)


def prior_mean(hist, hmask, cfg):
    """Straight ahead along the trace heading, at the fixed forward planes.

    The frame axis already follows the trace. On the training data a backward
    tangent extrapolated forward lowered no per-plane residual scale and raised
    the near-plane ones, so the mean carries no history term; history
    conditions the velocity field instead.
    """
    mean = torch.zeros(len(hist), cfg.n_future, 3, device=hist.device)
    mean[..., 2] = future_planes(cfg, hist.device)
    return mean


def observable_half_width(cfg):
    """Lateral extent within which a target's whole observation patch is in the crop."""
    return (cfg.width-1)*cfg.spacing/2 - cfg.flow_stencil_radius


def flow_targets(batch, cfg):
    """Annotated crossings the crop can observe: coordinates, token mask, censored mask.

    A token is supervised when its plane is annotated, the state has not
    departed, and no annotated plane up to it lies outside the observable
    lateral extent. Once the curve leaves the crop, later crossings are
    unobservable even if it re-enters, so censoring is a prefix. Unannotated
    planes never break the prefix, whatever their padding holds. ``censored``
    marks annotated tokens removed by this rule. Dense confidence labels are not
    affected: a curve that leaves the crop toward the fiber is still right.
    """
    ab = batch['plane_ab'].float()
    x1 = torch.cat([ab, future_planes(cfg, ab.device)[None, :, None].expand(len(ab), -1, 1)], -1)
    annotated = batch['plane_mask'].float()*(1-batch['offtrack'].float())[:, None]
    lateral = torch.where(annotated[..., None] > 0, ab, 0.).abs().amax(-1)
    observable = (lateral <= observable_half_width(cfg)).int().cummin(-1).values.float()
    return x1, annotated*observable, annotated*(1-observable)


class DenoisingBlock(nn.Module):
    """Pre-norm bidirectional self attention, image cross attention, and FFN."""
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.image_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2*width)
        self.cross_out = nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def image_cache(self, image):
        B, S, C = image.shape
        kv = self.key_value(self.image_norm(image)).reshape(B, S, 2, self.heads, C//self.heads)
        return kv.permute(2, 0, 3, 1, 4).unbind(0)

    def forward(self, x, padding, image_kv, batch):
        z = self.self_norm(x)
        x = x + self.self_attention(z, z, z, key_padding_mask=padding, need_weights=False)[0]
        N, P, C = x.shape
        # Flatten independent training draws into the query axis. Image K/V
        # remain one copy per state, even with 64 time/noise draws.
        q = self.query(self.cross_norm(x)).reshape(batch, -1, self.heads, C//self.heads).transpose(1, 2)
        z = F.scaled_dot_product_attention(q, *image_kv)
        z = z.transpose(1, 2).reshape(N, P, C)
        x = x + self.cross_out(z)
        return x + self.ff(self.ff_norm(x))


class PathFlow(nn.Module):
    def __init__(self, cfg, feature_channels):
        super().__init__()
        self.cfg = cfg
        sigma = torch.tensor(cfg.flow_sigma, dtype=torch.float32)
        if sigma.shape != (cfg.n_future, 2) or not torch.isfinite(sigma).all() or (sigma < 1).any():
            raise ValueError('flow_sigma requires fitted lateral scales >= 1 on every plane')
        self.register_buffer('sigma', sigma, persistent=False)
        stencil = torch.tensor([[a,b,0.] for a in (-1.,0.,1.) for b in (-1.,0.,1.)])
        self.register_buffer('stencil', stencil*cfg.flow_stencil_radius, persistent=False)
        h = cfg.hidden
        self.history_input = nn.Sequential(nn.Linear(feature_channels+6, h), nn.SiLU(), nn.Linear(h,h))
        self.future_input = nn.Sequential(nn.Linear(feature_channels*9+4, h), nn.SiLU(), nn.Linear(h,h))
        self.image_input = nn.Linear(cfg.widths[-1]+3, h)
        self.time = nn.Sequential(nn.Linear(64,h), nn.SiLU(), nn.Linear(h,h))
        self.context = nn.Linear(h,h)
        self.blocks = nn.ModuleList([DenoisingBlock(h,cfg.flow_heads) for _ in range(cfg.flow_layers)])
        self.final = nn.LayerNorm(h)
        self.velocity = nn.Linear(h,2)
        self.confidence_head = nn.Linear(h,1) if cfg.scorer == 'legacy' else PassageScorer(
            27*(feature_channels+1)+cfg.widths[-1]+1, h, h, cfg.flow_heads)
        if cfg.scorer == 'passage':
            radius = cfg.flow_stencil_radius
            self.register_buffer('path_stencil', torch.tensor([[a*radius, b*radius, z]
                for z in (-1., 0., 1.) for a in (-1., 0., 1.) for b in (-1., 0., 1.)]), persistent=False)
        nn.init.normal_(self.velocity.weight,std=.01)
        nn.init.zeros_(self.velocity.bias)

    def to_voxels(self, y, mu):
        return torch.cat([mu[:,None,:,:2]+self.sigma*y, mu[:,None,:,2:].expand(*y.shape[:-1],1)], -1)

    def sample_features(self, features, coordinates, sampling_grid):
        B = len(coordinates)
        grid = sampling_grid(coordinates.reshape(B,-1,1,3)+self.stencil)
        sampled = F.grid_sample(features.float(), grid[:, :, :, None].float(), align_corners=True, padding_mode='zeros')
        return sampled[...,0].permute(0,2,1,3).reshape(*coordinates.shape[:-1],-1)

    def conditioning(self, features, deep, hist, hmask, sampling_grid):
        cfg = self.cfg
        observed = hist[:,:cfg.recent_history_points].float()
        valid = hmask[:,:cfg.recent_history_points].bool()
        observed = torch.where(valid[...,None],observed,0.)
        grid = sampling_grid(observed)
        supported = valid & torch.isfinite(grid).all(-1) & (grid.abs() <= 1).all(-1)
        local = F.grid_sample(features.float(), grid[:,:,None,None].float(), align_corners=True).flatten(3).squeeze(-1).transpose(1,2)
        local = torch.where(supported[...,None], local, 0.)
        age = torch.arange(1,observed.shape[1]+1,device=hist.device).float()/cfg.recent_history_points
        tokens = self.history_input(torch.cat([local,observed/128,age[None,:,None].expand(len(hist),-1,-1),
                                               valid[...,None].float(),supported[...,None].float()],-1))
        # Average pooling is another factor of two beyond the deepest encoder.
        # Pool its coordinate lattice identically (centres are not crop edges).
        D,H,W = deep.shape[-3:]
        stride = 2**(len(cfg.widths)-1)
        axes = [2*torch.arange(n,device=deep.device).float()*stride/(full-1)-1
                for n,full in zip((D,H,W),(cfg.depth,cfg.width,cfg.width))]
        z,y,x = torch.meshgrid(*axes,indexing='ij')
        coords = torch.stack([x,y,z],0)[None].expand(len(hist),-1,-1,-1,-1)
        coarse = F.avg_pool3d(torch.cat([deep.float(),coords],1),2,ceil_mode=True)
        image = self.image_input(coarse.flatten(2).transpose(1,2))
        return dict(mu=prior_mean(hist,hmask,cfg), history=tokens, valid=valid,
                    supported=supported, coordinates=observed,
                    image_kv=[block.image_cache(image) for block in self.blocks])

    def forward(self, features, context, y, t, fixed, sampling_grid, future_mask=None, return_features=False):
        B,N,K,_ = y.shape
        known = torch.ones(B,K,device=y.device,dtype=torch.bool) if future_mask is None else future_mask.bool()
        y = torch.where(known[:,None,:,None],y,0.)
        future = self.to_voxels(y.float(),fixed['mu'])
        patches = self.sample_features(features,future,sampling_grid)
        plane = fixed['mu'][...,2]/(self.cfg.n_future*self.cfg.future_step)
        supported = (sampling_grid(future).abs() <= 1).all(-1).float()
        tokens = self.future_input(torch.cat([patches,y,plane[:,None,:,None].expand(B,N,K,1),supported[...,None]],-1))
        history = fixed['history'][:,None].expand(-1,N,-1,-1)
        h = torch.cat([history,tokens],2).reshape(B*N,-1,self.cfg.hidden)
        conditioning = self.time(time_embedding(t.reshape(-1),64)) + self.context(context)[:,None].expand(-1,N,-1).reshape(B*N,-1)
        h = h + conditioning[:,None]
        padding = ~torch.cat([fixed['valid'],known],1)[:,None].expand(-1,N,-1).reshape(B*N,-1)
        # Future queries remain available for wholly censored, no-history states;
        # their values are zeroed and carry no localization supervision.
        empty = padding.all(-1)
        padding = padding.clone()
        padding[empty,-1] = False
        for block,kv in zip(self.blocks,fixed['image_kv']):
            h = block(h,padding,kv,B)
        future_features = self.final(h[:,-K:]).reshape(B,N,K,-1)
        velocity = self.velocity(future_features).float()
        return (velocity,future_features) if return_features else velocity


class FollowNet(SpatialEncoder):
    # The full v11 crop is faster without repeated channels-last conversions
    # around group normalization and interpolation. See PERFORMANCE.md.
    cuda_memory_format = torch.contiguous_format

    def __init__(self,cfg):
        super().__init__(cfg)
        self.flow = PathFlow(cfg,cfg.widths[0])

    def future_targets(self, batch):
        """Observable annotated crossings and their token mask; see ``flow_targets``."""
        x1, token_mask, _ = flow_targets(batch, self.cfg)
        return x1, token_mask

    def flow_loss(self, features, context, hist, hmask, batch, generator=None, fixed=None):
        """Stratified flow matching on observable normalized lateral residuals.

        Missing and crop-censored futures are excluded as attention keys and
        from the loss (``flow_targets``). Observation tokens remain available
        on every requested plane.
        """
        cfg = self.cfg
        x1, token_mask, censored = flow_targets(batch, cfg)
        mu = fixed['mu']
        B, P, _ = mu.shape
        D = cfg.flow_draws
        t = (torch.arange(D, device=mu.device)[None] +
             torch.rand(B, D, device=mu.device, generator=generator)) / D
        y0 = torch.randn(B, D, P, 2, device=mu.device, generator=generator)
        residual = torch.where(token_mask[..., None] > 0, x1[..., :2]-mu[..., :2], 0.)
        y1 = residual / self.flow.sigma
        target = torch.where(token_mask[:, None, :, None] > 0, y1[:, None], y0)
        yt = (1-t)[..., None, None]*y0 + t[..., None, None]*target
        velocity = self.flow(features, context, yt, t, fixed, self.sampling_grid, future_mask=token_mask)
        error = (velocity - (target-y0)).square()
        weight = token_mask[:, None, :, None].expand_as(error)
        loss = (error*weight).sum()/weight.sum().clamp_min(1)
        return dict(flow_loss=loss, flow_known_count=token_mask.sum().detach(), flow_known_fraction=token_mask.mean().detach(),
                    flow_censored_fraction=(censored.sum()/(token_mask+censored).sum().clamp_min(1)).detach())

    @torch.no_grad()
    def refine(self, features, context, fixed, *, return_steps=False, initial_noise=None, generator=None):
        B,P,_ = fixed['mu'].shape
        y = initial_residuals(self.cfg, B, features.device, generator) if initial_noise is None else initial_noise
        candidates = 1+self.cfg.gaussian_candidates
        if y.shape != (B, candidates, P, 2):
            raise ValueError('initial_noise must have shape [batch, 1+gaussian_candidates, n_future, 2]')
        y = y.detach().to(device=features.device, dtype=torch.float32)
        curves = [self.flow.to_voxels(y,fixed['mu'])] if return_steps else []
        T = self.cfg.flow_steps
        for i in range(T):
            t = y.new_full((B,candidates),i/T)
            v = self.flow(features,context,y,t,fixed,self.sampling_grid)
            y = y + self.flow(features,context,y+v/(2*T),t+1/(2*T),fixed,self.sampling_grid)/T
            if return_steps:
                curves.append(self.flow.to_voxels(y,fixed['mu']))
        return y, curves

    def encode_conditioning(self,x,hist,hmask):
        """Spatial and static flow features, optionally shared across training passes."""
        features,context,deep = self.encode(x,hist,hmask,return_deep=True)
        fixed = self.flow.conditioning(features,deep,hist,hmask,self.sampling_grid)
        if self.cfg.scorer == 'passage':
            fixed['score_deep'] = deep
        return features,context,fixed

    @torch.no_grad()
    def generate_training_curve(self,x,hist,hmask,*,return_steps=False,encoding=None,initial_noise=None,generator=None):
        """Detached rollout, optionally retaining its initial curve and updates."""
        features,context,fixed = self.encode_conditioning(x,hist,hmask) if encoding is None else encoding
        y,curves = self.refine(features,context,fixed,return_steps=return_steps,initial_noise=initial_noise,generator=generator)
        points = self.flow.to_voxels(y,fixed['mu'])
        if not self.cfg.gaussian_candidates:
            points = points[:,0]
        # Refinement training diagnostics follow candidate zero; selection is
        # evaluated separately after scoring the complete generated pool.
        return (points,torch.stack(curves,1)[:,:,0]) if return_steps else points

    def training_forward(self,x,hist,hmask,targets,points,*,encoding=None):
        return self._forward(x,hist,hmask,targets=targets,training_points=points.detach(),encoding=encoding)

    def score_candidates(self, features, fixed, context, candidates):
        """Adapt this encoder's fine/deep evidence to the general passage scorer."""
        b, k, n, _ = candidates.shape
        points = candidates.detach().reshape(b, k*n, 3)
        local = sample_path_features(features,
            (points[:, :, None]+self.flow.path_stencil).reshape(b, -1, 3), self.crop)
        deep = sample_path_features(fixed['score_deep'], points, self.crop,
                                    stride=2**(len(self.cfg.widths)-1))
        evidence = torch.cat((local.reshape(b, k, n, -1), deep.reshape(b, k, n, -1)), -1)
        return self.flow.confidence_head(evidence, context, candidates, points.new_zeros(b, 3))

    def forward(self,x,hist,hmask,*,targets=None,generator=None,return_steps=False,initial_noise=None):
        return self._forward(x,hist,hmask,targets=targets,generator=generator,return_steps=return_steps,initial_noise=initial_noise)

    def _forward(self,x,hist,hmask,*,targets=None,generator=None,return_steps=False,training_points=None,encoding=None,initial_noise=None):
        features,context,fixed = self.encode_conditioning(x,hist,hmask) if encoding is None else encoding
        out = {}
        if targets is not None:
            out.update(self.flow_loss(features,context,hist,hmask,targets,generator,fixed))
        if training_points is None:
            y,curves = self.refine(features,context,fixed,return_steps=return_steps,initial_noise=initial_noise,generator=generator)
        else:
            pool = training_points[:,None] if training_points.ndim == 3 else training_points
            y = (pool[...,:2]-fixed['mu'][:,None,:,:2])/self.flow.sigma
            curves = []
        # Coordinates/labels are detached; the final evaluation and cached image
        # and history features retain their gradient route into the shared model.
        _,final = self.flow(features,context,y.detach(),y.new_ones(y.shape[:2]),fixed,self.sampling_grid,return_features=True)
        candidates = self.flow.to_voxels(y,fixed['mu']).detach() if training_points is None else pool.detach()
        selected = torch.zeros(len(y), device=y.device, dtype=torch.long)
        if self.cfg.scorer == 'passage':
            candidate_logits = self.score_candidates(features,fixed,final,candidates)
            candidate_confidence = candidate_logits.sigmoid().cummin(-1).values
            eligible = recovery_allowed(candidates,self.cfg.max_recovery_distance)
            horizon = min(self.cfg.selection_horizon,self.cfg.n_future)-1
            # argmax keeps the deterministic candidate on ties. Ranking and
            # acceptance both describe the commit prefix, not a distant exit.
            selected = candidate_confidence[...,horizon].masked_fill(~eligible,-torch.inf).argmax(-1)
            out.update(candidate_points=candidates,candidate_logits=candidate_logits,
                       candidate_confidence=candidate_confidence,selected_candidate=selected)
            logits = candidate_logits[torch.arange(len(y),device=y.device),selected]
        else:
            logits = self.flow.confidence_head(final[:,0]).squeeze(-1).float()
        points = candidates[torch.arange(len(y),device=y.device),selected]
        out.update(points=points,confidence_logits=logits,
                   confidence=logits.sigmoid().cummin(-1).values)
        if return_steps:
            out['denoising_steps'] = torch.stack(curves,1)[torch.arange(len(y),device=y.device),:,selected]
        return out
