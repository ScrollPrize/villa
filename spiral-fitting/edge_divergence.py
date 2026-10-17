"""Patch-pair divergence filter for fiber (same-winding) edges.

Two patches that a fiber hops between are within the link tolerance of each
other at the hop, so nothing measured there can tell one sheet from two sheets
pressed together. Away from the hop they differ: patches on the same sheet
stay within a voxel or two everywhere they overlap, while a neighbouring
winding sits a sheet pitch away and a drifting patch crosses from one side to
the other. This module measures that for each fiber edge and drops the edges
whose patches diverge.

For an edge whose fiber leaves patch A at ``from_zyx`` and arrives on patch B at
``to_zyx``, B's valid vertices within ``radius`` of the departure point are
projected onto A's surface near that point, and A's vertices near the arrival
point onto B. Each sample's offset is signed along the inward direction at the
sample (positive: the sampled patch lies in front of the other). A direction
with at least ``min_samples`` samples is measured; the edge is dropped when a
measured direction's median absolute offset exceeds ``max_median_offset`` or,
when ``max_spread`` is set, its p10-p90 spread exceeds it (voxels; the spread
of ordinary same-sheet patch pairs reaches ~4 vx at p95 on this data, so a
spread limit only makes sense well above that, for crossing patches). Edges measured in neither
direction (long hops: the other patch is not near either endpoint) are kept
and counted as unmeasured. Both directed copies of an edge share one decision.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.spatial import cKDTree
import torch

from tifxyz import Patch


@dataclass(frozen=True)
class EdgeDivergenceConfig:
    radius: float = 60.0
    max_median_offset: float = 2.0
    max_spread: float | None = None
    min_samples: int = 6
    kinds: tuple = ('neither',)
    report_kept: bool = False

    @classmethod
    def from_json(cls, value):
        if not isinstance(value, dict):
            raise ValueError('edge_divergence must be a JSON object')
        settings = {}
        for key, setting in value.items():
            name = key.replace('-', '_')
            if name not in cls.__dataclass_fields__:
                raise ValueError(f'unknown edge_divergence option {key!r}')
            if name in settings:
                raise ValueError(f'duplicate edge_divergence option {key!r}')
            settings[name] = setting
        for key in ('radius', 'max_median_offset', 'max_spread'):
            if key in settings:
                val = settings[key]
                if key == 'max_spread' and val is None:
                    continue
                if (type(val) not in (int, float) or not math.isfinite(val) or val < 0
                        or (key == 'radius' and val == 0)):
                    raise ValueError(f'edge_divergence.{key} must be a finite number '
                                     f'{"> 0" if key == "radius" else ">= 0"}')
                settings[key] = float(val)
        if 'min_samples' in settings and (type(settings['min_samples']) is not int
                                          or settings['min_samples'] < 1):
            raise ValueError('edge_divergence.min_samples must be a positive integer')
        if 'kinds' in settings:
            kinds = settings['kinds']
            if (not isinstance(kinds, list) or not kinds
                    or any(k not in ('neither', 'relative') for k in kinds)):
                raise ValueError("edge_divergence.kinds must be a non-empty list of 'neither'/'relative'")
            settings['kinds'] = tuple(kinds)
        if 'report_kept' in settings and type(settings['report_kept']) is not bool:
            raise ValueError('edge_divergence.report_kept must be a JSON boolean')
        return cls(**settings)


class _PatchGeometry:
    """Valid vertices of a patch with a KD-tree, built once per patch."""

    def __init__(self, patch):
        self.patch = patch
        mask = patch.valid_vertex_mask.cpu().numpy()
        self.ij = np.argwhere(mask)
        self.zyx = patch.zyxs.cpu().numpy()[mask].astype(np.float64)
        self.tree = cKDTree(self.zyx) if len(self.zyx) else None

    def near(self, zyx, radius):
        if self.tree is None:
            return np.zeros(0, dtype=np.int64)
        return np.asarray(self.tree.query_ball_point(np.asarray(zyx, dtype=np.float64), radius),
                          dtype=np.int64)

    def local_surface(self, zyx, radius):
        """Sub-patch spanning the valid vertices within ``radius`` (+1 cell)."""
        rows = self.near(zyx, radius)
        if len(rows) == 0:
            return None
        h, w = self.patch.zyxs.shape[:2]
        i0, j0 = np.maximum(self.ij[rows].min(axis=0) - 1, 0)
        i1, j1 = np.minimum(self.ij[rows].max(axis=0) + 1, [h - 1, w - 1])
        if i1 - i0 < 1 or j1 - j0 < 1:
            return None
        sub = self.patch.zyxs[i0:i1 + 1, j0:j1 + 1]
        try:
            return Patch(sub, self.patch.scale, None, None)
        except AssertionError:  # no valid quad in the window
            return None


def signed_offsets(samples, surface, inward):
    """Signed offsets of ``samples`` (N,3 zyx) from ``surface`` along ``inward``.

    Samples whose nearest surface point lies on the sub-patch boundary are
    discarded: the surface may simply end there."""
    if surface is None or len(samples) == 0:
        return np.zeros(0)
    pts = torch.as_tensor(np.asarray(samples, dtype=np.float32))
    ij, dist = surface.project(pts)
    ij, dist = ij.cpu().numpy(), dist.cpu().numpy()
    h, w = surface.zyxs.shape[:2]
    interior = (np.isfinite(dist) & (ij[:, 0] > 0.5) & (ij[:, 0] < h - 1.5)
                & (ij[:, 1] > 0.5) & (ij[:, 1] < w - 1.5))
    if not interior.any():
        return np.zeros(0)
    feet, valid = surface.ij_to_zyx(torch.as_tensor(ij[interior], dtype=torch.float32))
    feet, valid = feet.cpu().numpy().astype(np.float64), valid.cpu().numpy()
    samples = np.asarray(samples, dtype=np.float64)[interior][valid]
    feet = feet[valid]
    if len(samples) == 0:
        return np.zeros(0)
    directions = np.asarray(inward(samples), dtype=np.float64).reshape(-1, 3)
    return ((samples - feet) * directions).sum(axis=1)


def _stats(offsets):
    if len(offsets) == 0:
        return dict(n=0)
    p10, p50, p90 = np.percentile(offsets, [10, 50, 90])
    return dict(n=int(len(offsets)), median=float(p50), p10=float(p10), p90=float(p90),
                median_abs=float(np.median(np.abs(offsets))), spread=float(p90 - p10))


def measure_edge_divergence(geometry_a, geometry_b, from_zyx, to_zyx, inward, config):
    """Divergence of B's surface near the departure point from A, and of A's
    near the arrival point from B. Returns per-direction stats and a verdict."""
    directions = {}
    for name, sampled, other, contact in (('b_from_a', geometry_b, geometry_a, from_zyx),
                                          ('a_from_b', geometry_a, geometry_b, to_zyx)):
        rows = sampled.near(contact, config.radius)
        if len(rows) < config.min_samples:
            directions[name] = dict(n=int(len(rows)))
            continue
        offsets = signed_offsets(sampled.zyx[rows], other.local_surface(contact, config.radius), inward)
        directions[name] = _stats(offsets)
    measured = [d for d in directions.values() if d['n'] >= config.min_samples]
    diverges = any(d['median_abs'] > config.max_median_offset
                   or (config.max_spread is not None and d['spread'] > config.max_spread)
                   for d in measured)
    return dict(measured=bool(measured), diverges=diverges, **directions)


def filter_divergent_edges(rel_adjacency, patches, inward, config, pcl_names=None):
    """Drop fiber edges whose patches diverge; return (adjacency, report).

    ``rel_adjacency`` is build_rel_adjacency's output. Every undirected edge
    (pcl, point pair) is decided once. ``inward`` maps (N,3) zyx to (N,3)
    inward unit vectors."""
    geometries = {}

    def geometry(pid):
        if pid not in geometries:
            geometries[pid] = _PatchGeometry(patches[pid])
        return geometries[pid]

    decisions = {}
    counts = dict(edges=0, measured=0, dropped=0, unmeasured=0, skipped_kind=0)
    dropped = []
    kept = []
    filtered = {}
    for pid, edges in rel_adjacency.items():
        remaining = []
        for edge in edges:
            if edge.get('kind') not in config.kinds:
                remaining.append(edge)
                counts['skipped_kind'] += 1
                continue
            key = (edge['pcl_id'], frozenset((edge['from_point_id'], edge['to_point_id'])))
            if key not in decisions:
                counts['edges'] += 1
                result = measure_edge_divergence(
                    geometry(pid), geometry(edge['neighbor']),
                    np.asarray(edge['from_zyx'], dtype=np.float64),
                    np.asarray(edge['to_zyx'], dtype=np.float64), inward, config)
                decisions[key] = result['diverges']
                record = {
                    'pcl_id': edge['pcl_id'], 'pcl_name': edge.get('pcl_name'),
                    'from_patch': pid, 'to_patch': edge['neighbor'],
                    'from_point_id': edge['from_point_id'], 'to_point_id': edge['to_point_id'],
                    'from_zyx': np.asarray(edge['from_zyx'], dtype=float).tolist(),
                    'to_zyx': np.asarray(edge['to_zyx'], dtype=float).tolist(),
                    'b_from_a': result['b_from_a'], 'a_from_b': result['a_from_b'],
                }
                if result['measured']:
                    counts['measured'] += 1
                    if result['diverges']:
                        counts['dropped'] += 1
                        dropped.append(record)
                    elif config.report_kept:
                        kept.append(record)
                else:
                    counts['unmeasured'] += 1
            if not decisions[key]:
                remaining.append(edge)
        if remaining:
            filtered[pid] = remaining
    report = dict(settings=dict(asdict(config), kinds=list(config.kinds)), counts=counts,
                  dropped_edges=dropped)
    if config.report_kept:
        report['kept_edges'] = kept
    return filtered, report
