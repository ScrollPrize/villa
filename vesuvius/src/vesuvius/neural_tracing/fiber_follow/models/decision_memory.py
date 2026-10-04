"""Memory made of the main encoder's features from earlier decisions of the same trace.

Each decision's contextual token lattice is sampled once, around its own head, into a
fixed set of entry features: a lateral grid (+/-20 voxels at 4-voxel spacing) on three
planes along the path, plus five-point stencils at the committed path 2, 1 and 0 voxels
behind that head. Later decisions read up to ``SLOTS`` entries chosen by
``select_decisions``: the earliest decision (the seed), up to four older entries on a
power-of-two arclength lattice, and the three most recent entries at least 32 voxels
behind the head, spaced at least 32 voxels apart. Entries are tokenized with their pose
relative to the current crop. Selection on ``prune_decisions`` output equals selection on
the full decision list, so a trace or training chain keeps only O(log length) entries.
"""
import math

import numpy as np
import torch
from torch import nn

SLOTS = 8
MIN_AGE = 32.  # the current crop already covers 24 voxels of committed path
SPACING = 32.
RECENT = 3
OLDER = 4
WINDOW = 64.  # decisions newer than the older-entry boundary minus this stay prunable-safe
DECISION_SPACING = 16.  # simulated decision spacing for states without recorded decisions
LATERAL = np.arange(-20., 20.+1e-9, 4.)
PLANES = (-4., 0., 4.)
PATH_OFFSETS = (-2., -1., 0.)  # arclength relative to the entry's decision head
STENCIL = ((0., 0., 0.), (-1., 0., 0.), (1., 0., 0.), (0., -1., 0.), (0., 1., 0.))
GRID = np.array([(x, y, z) for z in PLANES for y in LATERAL for x in LATERAL], np.float32)
SPATIAL_TOKENS = len(GRID)
ENTRY_FEATURES = SPATIAL_TOKENS+len(PATH_OFFSETS)*len(STENCIL)
TOKENS_PER_ENTRY = SPATIAL_TOKENS+len(PATH_OFFSETS)
MEMORY_CHUNK = 8  # crops per encoder call when encoding entries without a recorded decision


def _older_layout(travelled, length):
    """Recent indices, older-entry boundary and lattice spacing (all deterministic)."""
    t = np.asarray(travelled, np.float64)
    eligible = np.flatnonzero(length-t >= MIN_AGE-1e-6)
    eligible = eligible[eligible > 0]
    recent, limit = [], np.inf
    for i in eligible[::-1]:
        if len(recent) == RECENT:
            break
        if t[i] <= limit+1e-6:
            recent.append(int(i))
            limit = t[i]-SPACING
    recent = recent[::-1]
    boundary = (t[recent[0]] if recent else length-MIN_AGE)-SPACING/2
    lattice = SPACING
    while math.floor((boundary-1e-6)/lattice) > OLDER:
        lattice *= 2
    return eligible, recent, boundary, lattice


def select_decisions(travelled, length):
    """Indices (ascending arclength) of at most ``SLOTS`` earlier decisions to read.

    ``travelled`` holds each earlier decision's observed-path arclength in increasing
    order, and ``length`` the current head's.
    """
    t = np.asarray(travelled, np.float64)
    if not len(t):
        return []
    if np.any(np.diff(t) < -1e-6) or t[-1] > length+1e-6:
        raise ValueError('Decisions must be ordered and precede the current head')
    eligible, recent, boundary, lattice = _older_layout(t, length)
    pool = eligible[t[eligible] < boundary]
    older = []
    for k in range(1, int(math.floor((boundary-1e-6)/lattice))+1):
        if not len(pool):
            break
        j = int(pool[np.argmin(np.abs(t[pool]-k*lattice))])
        if abs(t[j]-k*lattice) <= SPACING/2 and j not in older:
            older.append(j)
    chosen = sorted({0, *older, *recent}, key=lambda i: (t[i], i))
    assert len(chosen) <= SLOTS
    return chosen


def prune_decisions(travelled, length):
    """Indices of decisions that any later selection can still read."""
    t = np.asarray(travelled, np.float64)
    if not len(t):
        return []
    _, _, boundary, _ = _older_layout(t, length)
    keep = set(select_decisions(t, length))
    keep.update(int(i) for i in np.flatnonzero(t >= boundary-WINDOW))
    return sorted(keep)


class DecisionMemory(nn.Module):
    """Entry features from encoder lattices, and memory tokens for the decoder/scorer."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        width = cfg.hidden
        self.spatial_tokens_per_slab = SPATIAL_TOKENS
        self.tokens_per_slab = TOKENS_PER_ENTRY
        self.token_shape = (len(PLANES), len(LATERAL), len(LATERAL))  # GRID order: plane, y, x
        self.register_buffer('grid', torch.from_numpy(GRID), persistent=False)
        self.register_buffer('stencil', torch.tensor(STENCIL), persistent=False)
        self.projection = nn.Linear(width, width)
        self.position = nn.Linear(3, width)
        self.path_projection = nn.Sequential(nn.Linear(len(STENCIL)*width+6, width), nn.SiLU(), nn.Linear(width, width))
        nn.init.zeros_(self.path_projection[-1].weight)
        nn.init.zeros_(self.path_projection[-1].bias)
        # Translation/128 (3), relative rotation (9), log age (1), seed role (1).
        self.pose = nn.Sequential(nn.Linear(14, width), nn.SiLU(), nn.Linear(width, width))
        self.slot = nn.Embedding(SLOTS, width)
        self.norm = nn.LayerNorm(width)

    def entry_features(self, deep, path_points, path_valid):
        """(N, ENTRY_FEATURES, C) BF16 samples of one decision's token lattice.

        ``path_points`` (N, 3, 3) are the committed path at ``PATH_OFFSETS`` in that
        decision's crop frame; invalid points sample the head and are masked later.
        """
        from vesuvius.neural_tracing.fiber_follow.models.model import sample_features
        cfg = self.cfg
        points = torch.where(path_valid[..., None], path_points.float(), 0.)
        stencil = (points[:, :, None]+self.stencil).flatten(1, 2)
        grid = self.grid.to(points)[None].expand(len(points), -1, -1)
        values, _ = sample_features(deep, torch.cat((grid, stencil), 1), cfg.fine, cfg.token_stride, cfg.token_offset)
        return values.to(torch.bfloat16)

    def forward(self, features, valid, pose, path_points, path_tangents, path_valid):
        b, slots = valid.shape
        width = self.projection.out_features
        features = features.float()
        valid_paths = valid[..., None] & path_valid
        points = torch.where(valid_paths[..., None], path_points.float(), 0.)
        tangents = torch.where(valid_paths[..., None], path_tangents.float(), 0.)
        spatial = self.projection(features[:, :, :SPATIAL_TOKENS])+self.position(self.grid/16.)
        stencils = features[:, :, SPATIAL_TOKENS:].reshape(b, slots, len(PATH_OFFSETS), len(STENCIL)*width)
        centers = features[:, :, SPATIAL_TOKENS::len(STENCIL)]
        path = (self.projection(centers)+self.position(points/16.)
                +self.path_projection(torch.cat((stencils, points/16., tangents), -1)))
        tokens = torch.cat((spatial, path.to(spatial.dtype)), 2)
        metadata = self.pose(torch.where(valid[..., None], pose.float(), 0.))
        tokens = self.norm(tokens+metadata[:, :, None]+self.slot.weight[None, :, None])
        token_valid = torch.cat((valid[..., None].expand(-1, -1, SPATIAL_TOKENS), valid_paths), -1)
        padding = (~token_valid).reshape(b, -1)
        return tokens.flatten(1, 2).masked_fill(padding[..., None], 0.), padding
