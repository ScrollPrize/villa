"""CT-only conditioned plane likelihoods and bounded, disposable passage search."""
from dataclasses import dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..geometry import CropSpec
from .model import DirectConfig, ImageContext, sample_features

SPATIAL_ARCHITECTURE = 'direct_spatial_passages_v1'


@dataclass
class SpatialConfig(DirectConfig):
    fine: CropSpec = field(default_factory=lambda: CropSpec(depth=160, width=96, behind=48, spacing=.5))
    n_future: int = 48
    correction: bool = False
    output_spacing: float = 1.
    target_sigma: float = .65
    peak_count: int = 12
    routes_per_peak: int = 2
    candidates: int = 8
    lateral_step: float = 2.5
    diversity_radius: float = 1.
    ambiguity_margin: float = 1.
    agreement_radius: float = 1.
    seed_crop: CropSpec = field(default_factory=lambda: CropSpec(depth=16, width=16, behind=8, spacing=.5))

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.seed_crop, dict):
            self.seed_crop = CropSpec(**self.seed_crop)
        if self.correction or min(self.output_spacing, self.target_sigma, self.lateral_step,
                                  self.diversity_radius, self.agreement_radius) <= 0:
            raise ValueError('Spatial prediction requires positive scales and no coordinate refiner')
        if min(self.candidates, self.peak_count, self.routes_per_peak) < 1 or self.ambiguity_margin < 0:
            raise ValueError('Invalid candidate search settings')
        if self.candidates > (self.peak_count+1)*self.routes_per_peak or self.output_width < 3:
            raise ValueError('Candidate count exceeds search pool or output grid too small')

    @property
    def output_width(self):
        return int(2*self.lateral_limit/self.output_spacing)+1


@torch.no_grad()
def extract_passages(logits, lateral_grid, frontier, cfg):
    """Keep several routes per endpoint, then prune by whole-route separation.

    Heatmap scores are only proposal heuristics. A central fallback node keeps
    the graph connected during bootstrap. It is scored and labeled like any
    other proposal. Invalid/unreachable routes retain an explicit mask.
    """
    b, n, w, _ = logits.shape
    logp = logits.float().flatten(2).log_softmax(-1).reshape(b, n, w, w)
    maxima = logp == F.max_pool2d(logp.reshape(b*n, 1, w, w), 3, 1, 1).reshape(b, n, w, w)
    peaks = min(cfg.peak_count, w*w)
    values, indices = logp.masked_fill(~maxima, -torch.inf).flatten(2).topk(peaks, dim=-1)
    xy = lateral_grid[indices]
    center = frontier[:, None, None, :2].expand(-1, n, 1, -1).clamp(-cfg.lateral_limit, cfg.lateral_limit)
    center_grid = (center/cfg.lateral_limit).reshape(b*n, 1, 1, 2)
    central_score = F.grid_sample(logp.reshape(b*n, 1, w, w), center_grid,
                                  align_corners=True).reshape(b, n, 1)
    xy = torch.cat((xy, center), 2)
    values = torch.cat((values, central_score), 2)
    m, r = peaks+1, cfg.routes_per_peak
    batch = torch.arange(b, device=logits.device)[:, None]
    paths = frontier[:, None, None, :2].expand(-1, 1, 1, -1)
    score = logits.new_zeros(b, 1, dtype=torch.float32)
    velocity = torch.zeros_like(paths[:, :, -1])
    for plane in range(n):
        delta = xy[:, plane, None]-paths[:, :, -1, None]
        distance = delta.norm(dim=-1)
        limit = math.sqrt(max(0., cfg.max_recovery_distance**2-cfg.future_step**2)) if plane == 0 else cfg.lateral_step
        cost = score[:, :, None]+values[:, plane, None]-.08*distance.square()
        if plane:
            cost -= .08*(delta-velocity[:, :, None]).square().sum(-1)
        cost = cost.masked_fill(distance > limit, -torch.inf)
        count = min(r, cost.shape[1])
        best, parent = cost.topk(count, dim=1)
        parent = parent.transpose(1, 2).reshape(b, m*count)
        endpoint = torch.arange(m, device=logits.device).repeat_interleave(count)
        previous = paths[batch, parent]
        end = xy[:, plane, endpoint]
        velocity = end-previous[:, :, -1]
        paths = torch.cat((previous, end[:, :, None]), 2)
        score = best.transpose(1, 2).reshape(b, m*count)
    paths = paths[:, :, 1:]
    z = frontier[:, None, None, 2]+torch.arange(1, n+1, device=logits.device)*cfg.future_step
    pool = torch.cat((paths, z.expand(b, len(paths[0]), n)[..., None]), -1)
    pool_valid = torch.isfinite(score)
    # Max route separation retains switch/return patterns sharing the same exit.
    remaining = score.clone()
    selected, supported = [], []
    for _ in range(cfg.candidates):
        index = remaining.argmax(-1)
        candidate = pool[torch.arange(b, device=logits.device), index]
        selected.append(candidate)
        supported.append(torch.isfinite(remaining.gather(1, index[:, None])[:, 0]))
        separation = (pool[..., :2]-candidate[:, None, :, :2]).norm(dim=-1).amax(-1)
        remaining.masked_fill_(separation < cfg.diversity_radius, -torch.inf)
    return torch.stack(selected, 1), torch.stack(supported, 1), pool, pool_valid


def choose_passage(candidates, logits, valid, cfg):
    """Rank full passages; commit only supported, unambiguous short prefixes.

    The scorer supplies every learned acceptance score. Near-equal full-passage
    alternatives veto only prefixes on which they disagree, allowing supported
    progress before a distant unresolved fork. No candidate is averaged.
    """
    prefix_logits = logits.float().cummin(-1).values
    probabilities = prefix_logits.sigmoid()
    score = prefix_logits[..., -1].masked_fill(~valid, -torch.inf)
    selected = score.argmax(-1)
    batch = torch.arange(len(candidates), device=candidates.device)
    curve, confidence = candidates[batch, selected], probabilities[batch, selected]
    separation = (candidates[..., :2]-curve[:, None, :, :2]).norm(dim=-1).cummax(-1).values
    near = valid & (score >= score[batch, selected, None]-cfg.ambiguity_margin)
    rival = near[..., None] & (probabilities >= .5) & (separation > cfg.agreement_radius)
    # A differing rival requires full-passage support at the same externally
    # selected acceptance threshold. Several high-confidence valid variants
    # need not have an arbitrary ordering among themselves.
    confidence = torch.where(rival.any(1), torch.minimum(confidence, confidence[:, -1:]), confidence)
    confidence = confidence.masked_fill(~valid.any(1)[:, None], 0.).cummin(-1).values
    return curve, confidence, selected


class SpatialFollower(ImageContext):
    architecture = SPATIAL_ARCHITECTURE

    def __init__(self, cfg: SpatialConfig):
        super().__init__(cfg, inputs=1)
        c, h = cfg.channels, cfg.hidden
        self.query = nn.Sequential(nn.Linear(9*c+7, h), nn.SiLU(), nn.Linear(h, h))
        layer = nn.TransformerDecoderLayer(h, cfg.heads, 2*h, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.layers, norm=nn.LayerNorm(h))
        self.spatial_key = nn.Sequential(nn.Linear(5*c+3, h), nn.SiLU(), nn.Linear(h, h))
        self.spatial_bias = nn.Linear(h, 1)
        self.plane_query = nn.Linear(h, h)
        self.seed_projection = nn.Linear(c, h)
        self.seed_geometry = nn.Linear(7, h)
        self.seed_type = nn.Parameter(torch.zeros(h))
        self.frontier_token = nn.Linear(6, h)
        self.path_projection = nn.Sequential(nn.Linear(27*(c+1)+2*(4*c+1)+h+6, h), nn.SiLU())
        self.path_decoder = nn.TransformerEncoderLayer(h, cfg.heads, 2*h, dropout=0.,
                          activation='gelu', batch_first=True, norm_first=True)
        self.passage_score = nn.Sequential(nn.Linear(2*h+1, h), nn.SiLU(), nn.Linear(h, 1))
        stencil = torch.tensor([[a, b, 0.] for a in (-1., 0., 1.) for b in (-1., 0., 1.)])
        self.register_buffer('stencil', stencil*cfg.patch_radius, persistent=False)
        self.register_buffer('path_stencil', torch.tensor([[a*cfg.patch_radius, b*cfg.patch_radius, z]
            for z in (-1., 0., 1.) for a in (-1., 0., 1.) for b in (-1., 0., 1.)]), persistent=False)
        self.register_buffer('planes', torch.arange(1, cfg.n_future+1).float()*cfg.future_step, persistent=False)
        axis = torch.linspace(-cfg.lateral_limit, cfg.lateral_limit, cfg.output_width)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('lateral_grid', torch.stack((xx, yy), -1).reshape(-1, 2), persistent=False)

    def score_passages(self, features, decoded, candidates, frontier):
        fine, deep, coarse = features
        b, k, n, _ = candidates.shape
        geometry = candidates.detach()
        evidence = self.path_features(fine, deep, coarse, geometry.reshape(b, k*n, 3)).reshape(b, k, n, -1)
        delta = torch.diff(torch.cat((frontier[:, None, None].expand(-1, k, 1, -1), geometry), 2), dim=2)
        token = self.path_projection(torch.cat((evidence, decoded[:, None].expand(-1, k, -1, -1),
                                                  (geometry-frontier[:, None, None])/64, delta/4), -1))
        token = self.path_decoder(token.reshape(b*k, n, -1)).reshape(b, k, n, -1)
        count = torch.arange(1, n+1, device=token.device)[None, None, :, None]
        pooled = torch.cat((token.cumsum(2)/count, token.cummax(2).values,
                            (count/n).expand(b, k, -1, -1)), -1)
        return self.passage_score(pooled).squeeze(-1).float()

    def forward(self, x, hist, hmask, training_candidates=None):
        cfg = self.cfg
        fine, deep, coarse, memory, padding = self.encode_context(x, hist, hmask)
        b = len(hist)
        frontier = x.get('frontier', hist.new_zeros(b, 3)).float()
        direction = x.get('frontier_direction', hist.new_tensor([0., 0., 1.]).expand(b, -1)).float()
        prompt = torch.cat((frontier/128, direction), -1)
        tokens = [memory, self.frontier_token(prompt)[:, None]]
        masks = [padding, torch.zeros(b, 1, device=hist.device, dtype=torch.bool)]
        if 'seed_ct' in x:
            # Encode raw immutable seed observations with current weights during
            # training. Inference caches raw CT only, so embeddings cannot go stale.
            seed = self.fine_encoder.local(x['seed_ct'])
            seed = F.adaptive_avg_pool3d(seed, 2).flatten(2).transpose(1, 2)
            metadata = x['seed_metadata'].float()
            seed = self.seed_projection(seed)+self.seed_geometry(metadata)[:, None]+self.seed_type
            tokens.append(seed)
            masks.append(~metadata[:, -1:].bool().expand(-1, seed.shape[1]))
        memory, padding = torch.cat(tokens, 1), torch.cat(masks, 1)
        initial = frontier[:, None].expand(-1, cfg.n_future, -1).clone()
        initial[..., 2] += self.planes
        queries = self.query(torch.cat((self.patches(fine, initial),
                                       (self.planes/(cfg.n_future*cfg.future_step))[None, :, None].expand(b, -1, -1),
                                       prompt[:, None].expand(-1, cfg.n_future, -1)), -1))
        decoded = self.decoder(queries, memory, memory_key_padding_mask=padding)
        grid = self.lateral_grid[None, None].expand(b, cfg.n_future, -1, -1)
        z = initial[..., 2, None, None].expand(-1, -1, grid.shape[2], 1)
        locations = torch.cat((grid, z), -1).reshape(b, -1, 3)
        local, support = sample_features(fine, locations, cfg.fine)
        down, _ = sample_features(deep, locations, cfg.fine, stride=4)
        keys = self.spatial_key(torch.cat((local, down, locations/64), -1))
        keys = keys.reshape(b, cfg.n_future, -1, cfg.hidden)
        logits = (keys*self.plane_query(decoded)[:, :, None]).sum(-1)/math.sqrt(cfg.hidden)
        logits = logits+self.spatial_bias(keys).squeeze(-1)
        logits = logits.masked_fill(~support.reshape_as(logits), -1e4)
        logits = logits.reshape(b, cfg.n_future, cfg.output_width, cfg.output_width)
        candidates, valid, pool, pool_valid = extract_passages(logits, self.lateral_grid, frontier, cfg)
        candidate_logits = self.score_passages((fine, deep, coarse), decoded, candidates, frontier)
        points, confidence, selected = choose_passage(candidates, candidate_logits, valid, cfg)
        result = dict(points=points, initial_points=points, refinement_points=points[:, None],
                      confidence=confidence, confidence_logits=candidate_logits[torch.arange(b, device=hist.device), selected],
                      heatmap_logits=logits, candidate_points=candidates, candidate_logits=candidate_logits,
                      candidate_valid=valid, pool_points=pool, pool_valid=pool_valid, selected_candidate=selected)
        if training_candidates is not None:
            result['teacher_points'] = training_candidates
            result['teacher_logits'] = self.score_passages((fine, deep, coarse), decoded, training_candidates, frontier)
        return result
