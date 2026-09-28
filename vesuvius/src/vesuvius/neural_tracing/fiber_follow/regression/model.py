"""Direct curve regression and bounded local correction from CT and observed history.

All coordinates are trace-grid voxels, regardless of image sampling resolution.
There is no noise process, ODE, candidate ranking, or inherited flow backbone.
"""
from dataclasses import asdict, dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec

ARCHITECTURE = 'direct_curve_v1'


@dataclass
class DirectConfig:
    fine: CropSpec = field(default_factory=lambda: CropSpec(depth=80, width=48, behind=32, spacing=.5))
    coarse: CropSpec = field(default_factory=lambda: CropSpec(depth=88, width=32, behind=64, spacing=2.))
    channels: int = 24
    hidden: int = 128
    heads: int = 4
    layers: int = 4
    n_future: int = 16
    future_step: float = 1.
    n_history: int = 128
    history_stride: int = 4
    max_recovery_distance: float = 6.
    patch_radius: float = 1.
    correction: bool = True
    correction_limit: float = 1.  # maximum adjustment per refinement, trace voxels
    correction_steps: int = 2

    def __post_init__(self):
        for name in ('fine', 'coarse'):
            if isinstance(getattr(self, name), dict):
                setattr(self, name, CropSpec(**getattr(self, name)))
            crop = getattr(self, name)
            if min(crop.depth, crop.width) < 8 or not 0 <= crop.behind < crop.depth:
                raise ValueError('Image crops must be at least eight samples wide/deep with an interior origin')
            if not math.isfinite(crop.spacing) or crop.spacing <= 0:
                raise ValueError('Crop spacing must be positive')
        if min(self.channels, self.hidden, self.heads, self.layers, self.n_future,
               self.n_history, self.history_stride) < 1 or self.hidden % self.heads:
            raise ValueError('Positive dimensions required; hidden must divide by heads')
        if not 0 < self.future_step <= self.max_recovery_distance or not math.isfinite(self.max_recovery_distance):
            raise ValueError('Invalid forward spacing or connection limit')
        if not 0 < self.patch_radius < (self.fine.width-1)*self.fine.spacing/2:
            raise ValueError('Local observation patch must fit fine crop')
        if not math.isfinite(self.correction_limit) or self.correction_limit <= 0:
            raise ValueError('Correction limit must be finite and positive')
        if not isinstance(self.correction_steps, int) or self.correction_steps < 1:
            raise ValueError('Correction steps must be a positive integer')
        if self.n_future*self.future_step > (self.fine.depth-self.fine.behind-1)*self.fine.spacing:
            raise ValueError('Future horizon exceeds fine image')

    @property
    def recent_history_points(self):
        return self.n_history

    @property
    def lateral_limit(self):
        return (self.fine.width-1)*self.fine.spacing/2 - self.patch_radius

    def to_dict(self):
        return asdict(self)


def feature_grid(points, crop, shape, stride=1):
    """Map physical coordinates to a strided-convolution lattice, exactly.

    Padding-one, kernel-three convolutions put element i at input i*stride;
    feature endpoints need not coincide with crop endpoints for even sizes.
    """
    size = points.new_tensor(tuple(reversed(shape)))
    origin = points.new_tensor((-(crop.width-1)*crop.spacing/2,
                                -(crop.width-1)*crop.spacing/2, -crop.behind*crop.spacing))
    return 2*(points-origin)/(crop.spacing*stride*(size-1))-1


def sample_features(features, points, crop, stride=1):
    grid = feature_grid(points.float(), crop, features.shape[-3:], stride)
    supported = (grid.abs() <= 1).all(-1) & torch.isfinite(grid).all(-1)
    grid = torch.where(supported[..., None], grid, 0.)
    values = F.grid_sample(features.float(), grid[:, :, None, None], align_corners=True)
    values = values[:, :, :, 0, 0].transpose(1, 2)
    return torch.where(supported[..., None], values, 0.), supported


class ImageEncoder(nn.Module):
    """Small image pyramid; no full-resolution decoder or history modulation."""
    def __init__(self, channels, inputs=2):
        super().__init__()
        def stage(a, b, stride):
            return nn.Sequential(nn.Conv3d(a, b, 3, stride=stride, padding=1, bias=False),
                                 nn.GroupNorm(math.gcd(8, b), b), nn.SiLU(),
                                 nn.Conv3d(b, b, 3, padding=1, bias=False),
                                 nn.GroupNorm(math.gcd(8, b), b), nn.SiLU())
        self.local = stage(inputs, channels, 1)
        self.down = nn.Sequential(stage(channels, 2*channels, 2), stage(2*channels, 4*channels, 2))

    def forward(self, x):
        local = self.local(x)
        return local, self.down(local)


class ImageContext(nn.Module):
    def __init__(self, cfg: DirectConfig, inputs=2):
        super().__init__()
        self.cfg = cfg
        c, h = cfg.channels, cfg.hidden
        self.fine_encoder = ImageEncoder(c, inputs)
        self.coarse_encoder = ImageEncoder(c, inputs)
        self.image_token = nn.Linear(4*c+4, h)  # features, physical xyz, scale flag
        self.history_token = nn.Sequential(nn.Linear(2*c+6, h), nn.SiLU(), nn.Linear(h, h))

    def image_tokens(self, deep, crop, scale):
        # Pool coordinates with exactly the same cells as features.
        d, y, x = deep.shape[-3:]
        zyx = torch.meshgrid(*(torch.arange(n, device=deep.device).float() for n in (d, y, x)), indexing='ij')
        xyz = torch.stack((zyx[2], zyx[1], zyx[0]), 0)*4*crop.spacing
        xyz -= xyz.new_tensor(((crop.width-1)*crop.spacing/2,
                              (crop.width-1)*crop.spacing/2, crop.behind*crop.spacing))[:, None, None, None]
        xyz = xyz[None].expand(len(deep), -1, -1, -1, -1)/128
        pooled = F.avg_pool3d(torch.cat((deep.float(), xyz), 1), 2)
        tokens = pooled.flatten(2).transpose(1, 2)
        return self.image_token(torch.cat((tokens, torch.full_like(tokens[..., :1], scale)), -1))

    def patches(self, features, points):
        b, k, _ = points.shape
        values, _ = sample_features(features, (points[:, :, None]+self.stencil).reshape(b, k*9, 3), self.cfg.fine)
        return values.reshape(b, k, -1)

    def path_features(self, fine, fine_deep, coarse_deep, points):
        """Inspect the actual path at three longitudinal slices and two scales."""
        b, k, _ = points.shape
        local, support = sample_features(fine,
            (points[:, :, None]+self.path_stencil).reshape(b, k*27, 3), self.cfg.fine)
        local = torch.cat((local, support[..., None].float()), -1).reshape(b, k, -1)
        deep_fine, fine_support = sample_features(fine_deep, points, self.cfg.fine, stride=4)
        deep_coarse, coarse_support = sample_features(coarse_deep, points, self.cfg.coarse, stride=4)
        return torch.cat((local, deep_fine, fine_support[..., None].float(),
                          deep_coarse, coarse_support[..., None].float()), -1)

    def encode_context(self, x, hist, hmask):
        cfg = self.cfg
        fine, fine_deep = self.fine_encoder(x['fine'])
        coarse, coarse_deep = self.coarse_encoder(x['coarse'])
        valid = hmask.bool()
        observed = torch.where(valid[..., None], hist.float(), 0.)
        # Keep eight most recent observations, then one every four voxels.
        indices = torch.arange(cfg.n_history, device=hist.device)
        indices = indices[(indices < 8) | (indices % cfg.history_stride == 0)]
        observed, valid = observed[:, indices], valid[:, indices]
        flocal, fsupport = sample_features(fine, observed, cfg.fine)
        clocal, csupport = sample_features(coarse, observed, cfg.coarse)
        age = (indices.float()+1)/cfg.n_history
        ht = self.history_token(torch.cat((flocal, clocal, observed/128,
            age[None, :, None].expand(len(hist), -1, -1), fsupport[..., None].float(),
            csupport[..., None].float()), -1))
        image = torch.cat((self.image_tokens(fine_deep, cfg.fine, 0.),
                           self.image_tokens(coarse_deep, cfg.coarse, 1.)), 1)
        memory = torch.cat((image, ht), 1)
        padding = torch.cat((torch.zeros(image.shape[:2], dtype=torch.bool, device=hist.device), ~valid), 1)
        return fine, fine_deep, coarse_deep, memory, padding

class DirectFollower(ImageContext):
    architecture = ARCHITECTURE

    def __init__(self, cfg: DirectConfig, extra_query=0, extra_evidence=0):
        super().__init__(cfg)
        c, h = cfg.channels, cfg.hidden
        self.query = nn.Sequential(nn.Linear(9*c+1+extra_query, h), nn.SiLU(), nn.Linear(h, h))
        layer = nn.TransformerDecoderLayer(h, cfg.heads, 2*h, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.layers, norm=nn.LayerNorm(h))
        self.coordinates = nn.Linear(h, 2)
        nn.init.normal_(self.coordinates.weight, std=.005)
        nn.init.zeros_(self.coordinates.bias)
        # Rich evidence: 27 local samples with support flags, plus the deep
        # fine/coarse feature at each proposed point and its support flag.
        evidence_width = 27*(c+1)+2*(4*c+1)+extra_evidence
        def path_layers():
            return [nn.Linear(h+evidence_width+3, h), nn.SiLU(),
                    nn.TransformerEncoderLayer(h, cfg.heads, 2*h, dropout=0.,
                        activation='gelu', batch_first=True, norm_first=True)]
        if cfg.correction:
            self.correction_head = nn.Sequential(*path_layers(), nn.Linear(h, 2))
            nn.init.normal_(self.correction_head[-1].weight, std=.001)
            nn.init.zeros_(self.correction_head[-1].bias)
        self.path_evidence = nn.Sequential(*path_layers())
        self.confidence_head = nn.Sequential(nn.Linear(2*h, h), nn.SiLU(), nn.Linear(h, 1))
        stencil = torch.tensor([[a, b, 0.] for a in (-1., 0., 1.) for b in (-1., 0., 1.)])
        self.register_buffer('stencil', stencil*cfg.patch_radius, persistent=False)
        path_stencil = torch.tensor([[a*cfg.patch_radius, b*cfg.patch_radius, z]
                                    for z in (-1., 0., 1.) for a in (-1., 0., 1.) for b in (-1., 0., 1.)])
        self.register_buffer('path_stencil', path_stencil, persistent=False)
        self.register_buffer('planes', torch.arange(1, cfg.n_future+1).float()*cfg.future_step, persistent=False)

    def context(self, x, hist, hmask):
        fine, fine_deep, coarse_deep, memory, padding = self.encode_context(x, hist, hmask)
        return dict(fine=fine, fine_deep=fine_deep, coarse_deep=coarse_deep, memory=memory, padding=padding)

    def query_features(self, ctx, initial):
        cfg = self.cfg
        return torch.cat((self.patches(ctx['fine'], initial), initial[..., 2:]/(cfg.n_future*cfg.future_step)), -1)

    def evidence(self, ctx, points, stage):
        """Path evidence for the ``'correction'`` or ``'confidence'`` stage."""
        return self.path_features(ctx['fine'], ctx['fine_deep'], ctx['coarse_deep'], points)

    def forward(self, x, hist, hmask):
        return self.predict(self.context(x, hist, hmask), hist)

    def confidence_logits(self, ctx, decoded, points):
        """The same prefix classifier scores predictions and supervised candidates."""
        points = points.detach()
        evidence = self.path_evidence(torch.cat((decoded, self.evidence(ctx, points, 'confidence'),
                                                 points/16), -1))
        count = torch.arange(1, points.shape[1]+1, device=points.device)[None, :, None]
        prefix_mean = evidence.cumsum(1)/count
        prefix_max = evidence.cummax(1).values
        return self.confidence_head(torch.cat((prefix_mean, prefix_max), -1)).squeeze(-1).float()

    def predict(self, ctx, hist, candidates=None):
        cfg = self.cfg
        initial = hist.new_zeros(len(hist), cfg.n_future, 3)
        initial[..., 2] = self.planes
        query = self.query(self.query_features(ctx, initial))
        decoded = self.decoder(query, ctx['memory'], memory_key_padding_mask=ctx['padding'])
        lateral = cfg.lateral_limit*torch.tanh(self.coordinates(decoded).float())
        points = torch.cat((lateral, initial[..., 2:]), -1)
        initial_points = points
        refinements = [points]
        if cfg.correction:
            # Train the shared refiner through each update; refresh evidence
            # at the new coordinates while encoding both images only once.
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


# ---------------------------------------------------------------- visual identity

IDENTITY_ARCHITECTURE = 'direct_identity_v1'
# Appearance convolutions as (kernel along the path, lateral kernel, lateral
# dilation): a stem, then pre-activation residual blocks. Their receptive field
# is exactly one history patch. Version 2 uses 9 x 41 x 41 fine samples
# (±2 along the path, ±10 trace voxels laterally); version 1 retains 7 x 33 x 33.
APPEARANCE_STEM = (3, 3, 1)
APPEARANCE_BLOCKS = {
    1: (((3, 3, 1), (3, 3, 2)), ((1, 3, 2), (1, 3, 2)), ((1, 3, 4), (1, 3, 4))),
    2: (((3, 3, 1), (3, 3, 2)), ((3, 3, 2), (1, 3, 2)), ((1, 3, 6), (1, 3, 6))),
}
# Centre xyz/16, tangent and lateral axis in the head frame, log age, anchor flag.
PATCH_GEOMETRY = 11


def appearance_receptive_field(version=2):
    layers = (APPEARANCE_STEM,)+tuple(layer for block in APPEARANCE_BLOCKS[version] for layer in block)
    return 1+sum(k-1 for k, _, _ in layers), 1+sum((k-1)*d for _, k, d in layers)


@dataclass
class IdentityConfig(DirectConfig):
    # Wider fine crop (±20 trace voxels) so appearance at candidates up to ±10
    # laterally still sees a whole patch-sized receptive field.
    fine: CropSpec = field(default_factory=lambda: CropSpec(depth=80, width=81, behind=32, spacing=.5))
    appearance_version: int = 2
    appearance_channels: int = 32
    embedding: int = 32
    patch_every: int = 4  # history voxels between recent patches
    patch_span: int = 128  # recent history covered by patches
    anchor_patches: int = 4  # seed-segment patches, beyond the recent span
    anchor_every: float = 4.
    max_patch_age: float = 2048.
    appearance_behind: float = 4.  # dense appearance map starts this far behind the head
    persistent_seed: bool = False  # explicit observed seed, present from the first decision

    def __post_init__(self):
        super().__post_init__()
        if self.appearance_version not in APPEARANCE_BLOCKS:
            raise ValueError('Unsupported appearance encoder version')
        if min(self.appearance_channels, self.embedding, self.patch_every, self.patch_span) < 1 or self.anchor_patches < 0:
            raise ValueError('Identity dimensions must be positive')
        if self.patch_span > self.n_history or self.patch_span % self.patch_every:
            raise ValueError('Recent patches must evenly cover observed history')
        if not (0 < self.anchor_every and 0 < self.max_patch_age):
            raise ValueError('Invalid anchor spacing or age limit')
        if not 0 <= self.appearance_start < self.fine.behind:
            raise ValueError('Appearance map must start behind the head, inside the fine crop')
        if self.persistent_seed and self.anchor_patches < 1:
            raise ValueError('Persistent seed requires at least one anchor slot')

    @property
    def patch_crop(self):
        depth, width = appearance_receptive_field(self.appearance_version)
        return CropSpec(depth=depth, width=width, behind=depth//2, spacing=self.fine.spacing)

    @property
    def appearance_start(self):
        return self.fine.behind-int(round(self.appearance_behind/self.fine.spacing))

    @property
    def appearance_crop(self):
        start = self.appearance_start
        return CropSpec(depth=self.fine.depth-start, width=self.fine.width,
                        behind=self.fine.behind-start, spacing=self.fine.spacing)

    @property
    def recent_patches(self):
        return self.patch_span//self.patch_every

    @property
    def n_patches(self):
        return self.recent_patches+self.anchor_patches

    @property
    def anchor_offset(self):
        """Minimum anchor age: every anchor patch lies behind the recent span."""
        return self.patch_span+(self.anchor_patches-1)*self.anchor_every+4


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each voxel. Unlike GroupNorm it never pools
    space, so a patch centre and the same place in a larger crop are identical."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


class AppearanceConv(nn.Module):
    def __init__(self, a, b, along, lateral, dilation):
        super().__init__()
        self.conv = nn.Conv3d(a, b, (along, lateral, lateral), dilation=(1, dilation, dilation), bias=False)
        self.padding = ((along-1)//2, (lateral-1)*dilation//2, (lateral-1)*dilation//2)

    def forward(self, x, same):
        return F.conv3d(x, self.conv.weight, None, 1, self.padding if same else 0, self.conv.dilation)


class AppearanceBlock(nn.Module):
    def __init__(self, channels, layers):
        super().__init__()
        self.norms = nn.ModuleList(ChannelNorm(channels) for _ in layers)
        self.convs = nn.ModuleList(AppearanceConv(channels, channels, *layer) for layer in layers)

    def forward(self, x, same):
        y = x
        for norm, conv in zip(self.norms, self.convs):
            y = conv(F.silu(norm(y)), same)
        if not same:
            trim = [(a-b)//2 for a, b in zip(x.shape[2:], y.shape[2:])]
            x = x[:, :, trim[0]:trim[0]+y.shape[2], trim[1]:trim[1]+y.shape[3], trim[2]:trim[2]+y.shape[4]]
        return x+y


class AppearanceEncoder(nn.Module):
    """CT-only residual encoder, L2-normalized embedding.

    ``same=True`` gives a dense map over a crop; ``same=False`` gives the single
    embedding at the centre of an exact receptive-field patch. Both apply the
    same arithmetic to the same samples, so their embeddings are comparable.
    """
    def __init__(self, channels, embedding, version=2):
        super().__init__()
        self.stem = AppearanceConv(1, channels, *APPEARANCE_STEM)
        self.blocks = nn.ModuleList(AppearanceBlock(channels, layers) for layers in APPEARANCE_BLOCKS[version])
        self.norm = ChannelNorm(channels)
        self.head = nn.Conv3d(channels, embedding, 1)

    def forward(self, x, same=True):
        x = self.stem(x, same)
        for block in self.blocks:
            x = block(x, same)
        return F.normalize(self.head(F.silu(self.norm(x))).float(), dim=1)


class IdentityAttention(nn.Module):
    """Appearance at proposed points queries the traced fiber's appearance tokens.

    No similarity is computed by hand; a learned null token keeps attention
    defined when no history patch is available.
    """
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden
        self.query = nn.Sequential(nn.Linear(cfg.embedding+4, h), nn.SiLU(), nn.Linear(h, h))
        self.null = nn.Parameter(torch.zeros(1, 1, h))
        self.norm = nn.LayerNorm(h)
        self.attention = nn.MultiheadAttention(h, cfg.heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(2*h, h), nn.SiLU(), nn.Linear(h, h))

    def forward(self, embedding, points, support, tokens, mask):
        query = self.query(torch.cat((embedding, points/16, support[..., None].float()), -1))
        keys = torch.cat((self.null.expand(len(tokens), -1, -1).to(tokens.dtype), tokens), 1)
        padding = torch.cat((torch.zeros_like(mask[:, :1]), ~mask), 1)
        attended = self.attention(self.norm(query), keys, keys, key_padding_mask=padding, need_weights=False)[0]
        return self.out(torch.cat((query, attended), -1))


class IdentityFollower(DirectFollower):
    """Direct follower with a CT-only visual memory of the traveled path.

    ``x`` additionally holds fine CT ``patches`` (B, P, D, W, W), sized by
    ``cfg.patch_crop``, along the committed path in each patch's own frame, their ``patch_geometry`` and
    ``patch_mask``, and optionally an independently augmented ``appearance``
    copy of the fine CT channel (training only). The decoder memory, the
    correction head and the confidence head attend to the appearance tokens.
    ``queries`` (B, N, 3) returns appearance embeddings for the identity loss.
    Training may supply ``identity_query_patches`` (B, N, D, W, W) and
    ``identity_query_mask`` for full CT support beyond the main crop. Both
    query classes use the same encoder and patch geometry.
    """
    architecture = IDENTITY_ARCHITECTURE

    def __init__(self, cfg: IdentityConfig):
        e, h = cfg.embedding, cfg.hidden
        super().__init__(cfg, extra_query=9*e, extra_evidence=9*e+h)
        self.appearance = AppearanceEncoder(cfg.appearance_channels, e, cfg.appearance_version)
        self.appearance_token = nn.Sequential(nn.Linear(e+PATCH_GEOMETRY, h), nn.SiLU(), nn.Linear(h, h))
        if cfg.correction:
            self.correction_identity = IdentityAttention(cfg)
        self.confidence_identity = IdentityAttention(cfg)

    def context(self, x, hist, hmask):
        ctx = super().context(x, hist, hmask)
        cfg = self.cfg
        image = x['appearance'] if 'appearance' in x else x['fine'][:, :1]
        ctx['appearance'] = self.appearance(image[:, :, cfg.appearance_start:].to(ctx['fine'].dtype))
        patches = x['patches']
        b, p = patches.shape[:2]
        embedded = self.appearance(patches.reshape(b*p, 1, *patches.shape[2:]).to(ctx['fine'].dtype), same=False)
        embedded = embedded.reshape(b, p, -1)
        mask = x['patch_mask'].bool()
        tokens = self.appearance_token(torch.cat((embedded, x['patch_geometry'].float()), -1))
        ctx.update(history_embedding=embedded, patch_mask=mask, patch_tokens=tokens,
                   memory=torch.cat((ctx['memory'], tokens.to(ctx['memory'].dtype)), 1),
                   padding=torch.cat((ctx['padding'], ~mask), 1))
        return ctx

    def appearance_stencil(self, ctx, points):
        b, k, _ = points.shape
        values, support = sample_features(ctx['appearance'], (points[:, :, None]+self.stencil).reshape(b, k*9, 3),
                                          self.cfg.appearance_crop)
        return values.reshape(b, k, 9, -1), support.reshape(b, k, 9)

    def query_features(self, ctx, initial):
        values, _ = self.appearance_stencil(ctx, initial)
        return torch.cat((super().query_features(ctx, initial), values.flatten(2)), -1)

    def evidence(self, ctx, points, stage):
        values, support = self.appearance_stencil(ctx, points)
        attention = self.correction_identity if stage == 'correction' else self.confidence_identity
        # The stencil centre (index 4) is the appearance at the point itself.
        identity = attention(F.normalize(values[:, :, 4], dim=-1), points.float(), support[:, :, 4],
                             ctx['patch_tokens'], ctx['patch_mask'])
        return torch.cat((super().evidence(ctx, points, stage), values.flatten(2), identity.float()), -1)

    def forward(self, x, hist, hmask, queries=None, candidates=None):
        ctx = self.context(x, hist, hmask)
        out = self.predict(ctx, hist, candidates)
        out.update(history_embedding=ctx['history_embedding'], patch_mask=ctx['patch_mask'].float())
        if queries is not None:
            if 'identity_query_patches' in x:
                patches = x['identity_query_patches']
                b,q = patches.shape[:2]
                values = self.appearance(patches.reshape(b*q,1,*patches.shape[2:]).to(ctx['fine'].dtype),same=False)
                values = values.reshape(b,q,-1)
                support = x['identity_query_mask'] > 0
            else:
                values, support = sample_features(ctx['appearance'], queries.float(), self.cfg.appearance_crop)
            out.update(query_embedding=F.normalize(values, dim=-1), query_support=support.float())
        return out


def config_class(architecture):
    return IdentityConfig if architecture == IDENTITY_ARCHITECTURE else DirectConfig


def follower_class(architecture):
    return IdentityFollower if architecture == IDENTITY_ARCHITECTURE else DirectFollower
