"""Patch tokenizers over the crop CNN's feature lattice (CropTransformerArchitecture.cell_tokens 'grid' / 'polar').

Every token is a fixed patch of the crop: the CNN features are sampled (trilinearly, models/follower.sample_features)
at fixed points inside the patch, and the points' features and crop support are concatenated and projected to the
transformer width. The layouts:

  * 'grid': cubic patches of ``patch_size`` samples, a regular lattice; the points are the CNN cell centers inside
    the patch, ``patch_points`` per axis (so with patch_size = patch_points * the CNN stride a patch is exactly a
    block of CNN cells);
  * 'polar': the crop axis (the trace heading through the crop center) is cut into the same slabs as the grid's
    depth. Each slab's cross-section is a center disc of radius ``polar_tube_radius`` (one tube token per slab) and
    rings around it, ring k (1, 2, ...) ``polar_ring_width * polar_ring_growth**(k-1)`` wide (growth 1: equal rings;
    above 1: rings widen with radius, so tokens are finest at the trace and coarser outwards); ``polar_ring_widths``
    instead gives the innermost rings' widths explicitly, and each further ring is the previous one times the growth.
    Ring k is split into
    ``polar_sectors * round(inner radius / ring width)`` (at least ``polar_sectors``) equal-angle sectors, so a sector
    is about as wide as its ring is deep (with equal rings as wide as the tube: ``polar_sectors * k``, nearly square at
    6). A sector's points lie on its own radial and tangential axes (``patch_points`` along depth, radius and angle), so
    its projection reads 'outwards' the same way all around the axis; rotating the crop about its axis by
    2*pi/polar_sectors permutes the sectors of every ring. The tube's points are its axis and ``polar_sectors`` points
    at half its radius, at each depth point. Rings extend to the crop corners; sectors with no point inside the crop
    are dropped, and points outside it read zero features with zero support.

    ``polar_pool_spacing`` (None: off): each sector point is the mean of the features over its share of the sector
    (its cell of the patch_points**3 split), sampled at most this many samples apart, so wide outer sectors read their
    whole area rather than patch_points**3 single points (None: the cell's center only).

Sizes are in crop samples; token centers (``centers``) and points are crop-local XYZ, as models/follower.py.
"""
import math

import torch
from torch import nn

from vesuvius.neural_tracing.tracers.models.follower import crop_support, sample_features

CELL_TOKENS = ('cells', 'grid', 'polar')


def _local(cfg, u, v, d):
    """Crop-local XYZ (..., 3) of continuous sample indices along width (u), height (v) and depth (d)."""
    c = cfg.fine
    return torch.stack(((u-(c.width-1)/2)*c.spacing, (v-(c.width-1)/2)*c.spacing, (d-c.behind)*c.spacing), -1)


def slab_depths(cfg):
    """Per slab, the depth sample indices of its points (S, patch_points) and its center (S,)."""
    p, n = cfg.patch_size, cfg.patch_points
    starts = torch.arange(cfg.fine.depth//p, dtype=torch.float64)*p
    points = starts[:, None]+torch.arange(n, dtype=torch.float64)*(p/n)
    return points, points.mean(-1)


def grid_layout(cfg):
    """One group: cubic patches (T, n**3 points) and their centers (T, 3), depth-major."""
    p, n = cfg.patch_size, cfg.patch_points
    depth, depth_center = slab_depths(cfg)
    lateral = torch.arange(cfg.fine.width//p, dtype=torch.float64)[:, None]*p+torch.arange(n, dtype=torch.float64)*(p/n)
    d, v, u = torch.meshgrid(torch.arange(len(depth)), torch.arange(len(lateral)), torch.arange(len(lateral)),
                             indexing='ij')
    pd, pv, pu = torch.meshgrid(torch.arange(n), torch.arange(n), torch.arange(n), indexing='ij')
    flat = lambda t: t.reshape(-1)
    points = _local(cfg, lateral[flat(u)][:, flat(pu)], lateral[flat(v)][:, flat(pv)], depth[flat(d)][:, flat(pd)])
    centers = _local(cfg, lateral[flat(u)].mean(-1), lateral[flat(v)].mean(-1), depth_center[flat(d)])
    return dict(patch=(points, centers))


def polar_rings(cfg):
    """Per ring k = 1.., its inner radius, width and sector count, up to the last ring starting inside the crop's
    corner radius."""
    c = cfg.fine
    corner = (c.width-1)/2*math.sqrt(2)  # samples from the axis to a crop corner
    first = [float(w) for w in cfg.polar_ring_widths or (cfg.polar_ring_width,)]
    rings, inner, width = [], float(cfg.polar_tube_radius), first[0]
    while inner < corner:
        rings.append((inner, width, cfg.polar_sectors*max(1, round(inner/width))))
        inner = inner+width
        width = first[len(rings)] if len(rings) < len(first) else width*cfg.polar_ring_growth
    return rings


def _pooled(extent, spacing):
    """Sub-point offsets (m,) in units of a cell of ``extent`` samples, centered on it: one (its center) without
    pooling, else at most ``spacing`` samples apart."""
    m = 1 if spacing is None else max(1, math.ceil(extent/spacing-1e-9))
    return (torch.arange(m, dtype=torch.float64)+.5)/m-.5


def polar_layout(cfg):
    """Two groups: ring sectors (T, n**3 points, sub-points, 3) and tube discs (S, n*(1+polar_sectors), 1, 3), each
    with centers and sub-point weights (zero: padding to the widest ring's sub-points); slab-major, then ring, then
    sector (angle from +x towards +y). Points are radial-major within a depth point."""
    c, n, p = cfg.fine, cfg.patch_points, cfg.patch_size
    depth, depth_center = slab_depths(cfg)
    middle = (c.width-1)/2
    frac = (torch.arange(n, dtype=torch.float64)+.5)/n
    sub_depth = _pooled(p/n, cfg.polar_pool_spacing)*(p/n)  # depth points are the CNN cell centers of the slab
    rings = []
    for inner, width, count in polar_rings(cfg):
        sub_r = _pooled(width/n, cfg.polar_pool_spacing)
        sub_a = _pooled((inner+width)*2*math.pi/count/n, cfg.polar_pool_spacing)  # sized by the outer arc
        radius = inner+(frac[:, None]+sub_r[None]/n)*width  # (n, m_r)
        sectors = []
        for j in range(count):
            angle = 2*math.pi*(j+frac[:, None]+sub_a[None]/n)/count  # (n, m_a)
            # (n radial, n angular, m_r, m_a): this sector's cross-section sub-points
            r = radius[:, None, :, None].expand(n, n, len(sub_r), len(sub_a))
            a = angle[None, :, None, :].expand(n, n, len(sub_r), len(sub_a))
            mid = 2*math.pi*(j+.5)/count, inner+width/2
            sectors.append((middle+r*a.cos(), middle+r*a.sin(), middle+mid[1]*math.cos(mid[0]),
                            middle+mid[1]*math.sin(mid[0])))
        u = torch.stack([s[0].reshape(n*n, -1) for s in sectors])  # (Q, n*n, m_r*m_a)
        v = torch.stack([s[1].reshape(n*n, -1) for s in sectors])
        q, m = u.shape[0], u.shape[-1]*len(sub_depth)
        # Every cross-section sub-point at every depth sub-point: (slabs, Q, n depth * n*n, m_depth * m_r*m_a)
        lateral = lambda t: t[None, :, None, :, None, :].expand(len(depth), q, n, n*n, len(sub_depth), -1)
        deep = (depth[:, :, None]+sub_depth)[:, None, :, None, :, None].expand(-1, q, n, n*n, -1, u.shape[-1])
        points = _local(cfg, *(t.reshape(len(depth), q, n**3, m) for t in (lateral(u), lateral(v), deep)))
        centers = _local(cfg, torch.tensor([s[2] for s in sectors])[None].expand(len(depth), -1),
                         torch.tensor([s[3] for s in sectors])[None].expand(len(depth), -1),
                         depth_center[:, None].expand(-1, q))
        # The same sectors in every slab (the crop's cross-section does not change along depth).
        keep = crop_support(points.float(), c).flatten(2).any(-1).all(0)
        rings.append((points[:, keep], centers[:, keep]))
    width = max(points.shape[-2] for points, _ in rings)
    pad = lambda t: torch.cat((t, t[..., :1, :].expand(*t.shape[:-2], width-t.shape[-2], 3)), -2)
    weights = torch.cat([torch.cat((torch.ones(*points.shape[:-2], points.shape[-2], dtype=torch.float64),
                                    torch.zeros(*points.shape[:-2], width-points.shape[-2], dtype=torch.float64)), -1)
                         for points, _ in rings], 1)
    sector_points = torch.cat([pad(points) for points, _ in rings], 1).reshape(-1, n**3, width, 3)
    sector_centers = torch.cat([centers for _, centers in rings], 1).reshape(-1, 3)
    # Tube: its axis and polar_sectors points at half its radius, at each depth point.
    turn = torch.arange(cfg.polar_sectors, dtype=torch.float64)*2*math.pi/cfg.polar_sectors
    tu = torch.cat((torch.zeros(1, dtype=torch.float64), cfg.polar_tube_radius/2*turn.cos()))+middle
    tv = torch.cat((torch.zeros(1, dtype=torch.float64), cfg.polar_tube_radius/2*turn.sin()))+middle
    tube_points = _local(cfg, tu[None, None].expand(len(depth), n, -1), tv[None, None].expand(len(depth), n, -1),
                         depth[:, :, None].expand(-1, -1, len(tu))).reshape(len(depth), -1, 1, 3)
    tube_centers = _local(cfg, torch.full_like(depth_center, middle), torch.full_like(depth_center, middle), depth_center)
    return dict(sector=(sector_points, sector_centers, weights.reshape(-1, n**3, width)),
                tube=(tube_points, tube_centers, torch.ones(tube_points.shape[:-1], dtype=torch.float64)))


def patch_layout(cfg):
    """The configured layout's groups: name -> (points (T, P, S, 3), centers (T, 3), sub-point weights (T, P, S)),
    crop-local XYZ; a token point is the weighted mean of its S sub-points."""
    if cfg.cell_tokens == 'grid':
        points, centers = grid_layout(cfg)['patch']
        groups = dict(patch=(points[:, :, None], centers, torch.ones(points.shape[:-1], dtype=torch.float64)[..., None]))
    else:
        groups = polar_layout(cfg)
    return {name: tuple(t.float() for t in group) for name, group in groups.items()}


class PatchTokens(nn.Module):
    """Tokens (B, T, width) of CNN feature lattices: per layout group, a projection of its patches' concatenated point
    features and supports (each point the mean of its sub-points).

    ``levels``: the feature lattices, finest first, as (crop, stride) of each lattice's input (default: one lattice of
    the whole crop at the token stride), and ``channels`` their channel counts (one count: every level's); a token reads the finest level whose crop holds all its points (else the
    coarsest), through its group's projection for that level. A group's tokens are sampled in parts of equal level and
    sub-point count (a polar ring's sectors share both), so wide outer sectors do not pad the narrow inner ones to their
    size; the parts are put back in layout order before projection."""

    def __init__(self, cfg, channels, width, levels=None):
        super().__init__()
        self.levels = tuple(levels or ((cfg.fine, cfg.token_stride),))
        channels = (channels,)*len(self.levels) if isinstance(channels, int) else tuple(channels)
        layout = patch_layout(cfg)
        self.groups = tuple(layout)
        self.shapes = {}  # per group, per part: (level, (tokens, points, sub-points))
        self.projections = {}  # per group, the levels its tokens read
        for name, (points, _, weights) in layout.items():
            counts = (weights > 0).sum(-1)  # (T, P) real sub-points
            if not bool((counts == counts[:, :1]).all()):
                raise ValueError(f'{name}: the points of a token must have equal sub-point counts')
            counts = counts[:, 0]
            real = weights > 0
            level = torch.full((len(points),), len(self.levels)-1)
            for index in reversed(range(len(self.levels)-1)):  # finest level that holds every point of the token
                inside = (crop_support(points, self.levels[index][0]) | ~real).flatten(1).all(-1)
                level = torch.where(inside, index, level)
            order, self.shapes[name] = [], []
            for key in sorted(set(zip(level.tolist(), counts.tolist()))):
                index = ((level == key[0]) & (counts == key[1])).nonzero()[:, 0]
                order.append(index)
                part_points, part_weights = points[index, :, :key[1]], weights[index, :, :key[1]]
                part = len(self.shapes[name])
                self.shapes[name].append((key[0], tuple(part_points.shape[:3])))
                # Flat (N, 3): Module.to(memory_format=channels_last_3d) would reformat a 4- or 5-dimensional buffer.
                self.register_buffer(f'{name}_points_{part}', part_points.reshape(-1, 3), persistent=False)
                self.register_buffer(f'{name}_weights_{part}', part_weights/part_weights.sum(-1, keepdim=True),
                                     persistent=False)
            # Layout position of each token in the concatenated parts.
            self.register_buffer(f'{name}_order', torch.argsort(torch.cat(order)), persistent=False)
            self.projections[name] = sorted(set(level.tolist()))
            for index in self.projections[name]:
                setattr(self, self.projection_name(name, index), nn.Linear(points.shape[1]*(channels[index]+1), width))
        self.register_buffer('centers', torch.cat([group[1] for group in layout.values()]), persistent=False)

    @staticmethod
    def projection_name(group, level):
        return f'{group}_projection' if level == 0 else f'{group}_projection_level{level}'

    def forward(self, features):
        """``features``: one lattice per level (a tensor for a single level)."""
        features = (features,) if torch.is_tensor(features) else tuple(features)
        b = len(features[0])
        tokens = []
        for name in self.groups:
            parts = []
            for part, (level, (t, p, s)) in enumerate(self.shapes[name]):
                points, weights = getattr(self, f'{name}_points_{part}'), getattr(self, f'{name}_weights_{part}')
                crop, stride = self.levels[level]
                values, support = sample_features(features[level], points[None].expand(b, -1, -1), crop, stride)
                values = torch.cat((values, support[..., None].to(values.dtype)), -1).reshape(b, t, p, s, -1)
                values = values[..., 0, :] if s == 1 else (values*weights[..., None].to(values.dtype)).sum(-2)
                parts.append((level, values.reshape(b, t, -1)))
            order = getattr(self, f'{name}_order')
            if len(self.projections[name]) == 1:  # one projection: order the point features, then project
                values = parts[0][1] if len(parts) == 1 else torch.cat([v for _, v in parts], 1)[:, order]
                tokens.append(getattr(self, self.projection_name(name, parts[0][0]))(values))
                continue
            # Parts are sorted by level: project each level's parts together, then restore layout order.
            projected = [getattr(self, self.projection_name(name, level))(torch.cat([v for l, v in parts if l == level], 1))
                         for level in self.projections[name]]
            tokens.append(torch.cat(projected, 1)[:, order])
        return torch.cat(tokens, 1)
