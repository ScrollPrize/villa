"""Offline, conservative neighbor tracing and a human-review gallery.

All Python geometry is xyz in the fiber grid (eight base voxels by default).
The native VC3D coordinate adapter is applied only at the binding boundary.
Candidates are *proposals*: this command does not add them to training labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.fiber_follow.shared.data import load_fibers, split_fibers, ZBand
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at, normalize, tangent_at
from vesuvius.neural_tracing.fiber_follow.shared.volume import ChunkedArray


@dataclass(frozen=True)
class MiningConfig:
    seed_presence: float = .8
    path_presence: float = .65
    max_angle: float = 25.
    seed_spacing: float = 20.
    extrapolation: float = 10.  # at each end
    exclusion: float = 2.5
    min_distance: float = 0.  # optional inner boundary of the search band
    max_distance: float = 12.
    block_size: int = 80
    sample_step: float = .25
    grid_scale: float = 8.
    min_path_length: float | None = None
    max_path_length: float | None = None

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values() if v is not None):
            raise ValueError('Mining parameters must be finite')
        if not 0 < self.path_presence <= self.seed_presence <= 1:
            raise ValueError('Require 0 < path_presence <= seed_presence <= 1')
        if not (0 < self.max_angle < 90 and 0 < self.exclusion < self.max_distance):
            raise ValueError('Invalid direction angle or separation limits')
        if not 0 <= self.min_distance < self.max_distance:
            raise ValueError('Require 0 <= min_distance < max_distance')
        if min(self.seed_spacing, self.sample_step, self.grid_scale) <= 0 or self.extrapolation < 0:
            raise ValueError('Invalid seed spacing, sampling step or extrapolation')
        if (self.min_path_length is None) != (self.max_path_length is None):
            raise ValueError('Supply both min_path_length and max_path_length')
        if self.min_path_length is not None and not self.seed_spacing <= self.min_path_length <= self.max_path_length:
            raise ValueError('Require seed_spacing <= min_path_length <= max_path_length')
        if self.block_size < self.path_length_limit + 8:
            raise ValueError('Block must contain the full seed and extrapolation, with a margin')

    @property
    def path_length_limit(self):
        return self.max_path_length if self.max_path_length is not None else self.seed_spacing+2*self.extrapolation


def load_native(build_python=None):
    """Optionally use a freshly built extension without reinstalling the package."""
    import vc
    if build_python:
        vc.__path__.insert(0, str(Path(build_python).resolve() / 'vc'))
    from vc import fiber_trace
    return fiber_trace


def prediction_manifest(fiber_zarrs):
    groups = {}
    for channel in ('presence', 'nx', 'ny'):
        matches = sorted(Path(fiber_zarrs).glob(f'*_{channel}.ome.zarr'))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {channel} store, found {len(matches)}')
        groups[channel] = dict(zarr=str((matches[0]/'3').resolve()), scaledown=3, channels=[channel])
    return dict(version=2, source_to_base=1., groups=groups)


class MiningResources:
    """Worker-local readers and native tracer, shared by preview and bulk runs."""
    def __init__(self, manifest, build_python=None, ct=None, ct_grid_scale=8., grid_scale=8.):
        self.native = load_native(build_python)
        metadata = json.loads(Path(manifest).read_text())
        self.volumes = {c: ChunkedArray(g['zarr'], cache_bytes=64 << 20) for c,g in metadata['groups'].items()}
        for volume in self.volumes.values():
            if volume.codec is not None or volume.dtype != np.dtype('uint8'):
                raise ValueError(f'Decode the uint8 store in place first: {volume.path}')
        if len({v.shape for v in self.volumes.values()}) != 1:
            raise ValueError('Presence and direction shapes differ')
        self.field = self.native.open_prediction_field(str(manifest), cache_bytes=256 << 20, scaledown_power=2)
        # Fiber directions are not sheet normals. Retain isotropic smoothness.
        self.native_config = self.native.TraceConfig(trace_to_base_scale=self.field.trace_to_base_scale,
            smoothness_normal_weight=0., smoothness_tangent_weight=0., cumulative_smoothness_tangent_weight=0.,
            parallel_threads=1)
        self.ct = ChunkedArray(ct, cache_bytes=128 << 20) if ct else None
        self.ct_scale = grid_scale/ct_grid_scale
        if self.ct is not None:
            expected = np.asarray(self.volumes['presence'].shape)*self.ct_scale
            if np.any(np.abs(np.asarray(self.ct.shape)-expected) > 1):
                raise ValueError('CT shape and coordinate scale do not align with predictions')

    def block(self, origin, cfg):
        raw = {c: v.read(origin[::-1], (cfg.block_size,)*3) for c,v in self.volumes.items()}
        directions = self.native.decode_directions(raw['nx'].ravel(), raw['ny'].ravel()).reshape((*raw['nx'].shape, 3))
        return raw['presence'], directions


def dense_line(points, step):
    """Densify every original segment, preserving corners and both endpoints."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Expected a finite polyline with at least two xyz points')
    if not np.isfinite(step) or step <= 0:
        raise ValueError('Sampling step must be positive')
    pieces = []
    for a, b in zip(points[:-1], points[1:]):
        length = np.linalg.norm(b-a)
        if length > 1e-9:
            n = max(1, int(np.ceil(length/step)))
            pieces.append(a + np.arange(n)[:, None]/n*(b-a))
    if not pieces:
        raise ValueError('Polyline has zero length')
    return np.concatenate([*pieces, points[-1:]])


class PolylineIndex:
    """Reusable exact-distance and conservative seed-exclusion indices."""
    def __init__(self, target):
        self.target = np.asarray(target, float)
        self.a = self.target[:-1]
        self.delta = np.diff(self.target, axis=0)
        self.length2 = np.einsum('ij,ij->i', self.delta, self.delta)
        self.tree = cKDTree(self.a+self.delta/2)
        self.half = np.sqrt(self.length2.max())/2

    @property
    def seed_tree(self):
        # Exact-distance queries need only segment midpoints. Densifying a
        # whole annotation is useful only to the mining seed-exclusion query.
        if not hasattr(self, '_seed_tree'):
            self._seed_tree = cKDTree(dense_line(self.target, .5))
        return self._seed_tree


def exact_nearest(points, target, index=None):
    """Exact nearest points on a polyline, including interiors of long segments."""
    p = np.asarray(points, float)
    index = index or PolylineIndex(target)
    # Bound temporary candidate arrays for large offline mining queries.
    if len(p) > 256:
        chunks = [exact_nearest(p[i:i+256], target, index) for i in range(0, len(p), 256)]
        return tuple(np.concatenate(values) for values in zip(*chunks))
    a, delta, length2 = index.a, index.delta, index.length2
    # Midpoint broad phase: the nearest segment must be within current best
    # distance + half the longest segment of the query point.
    tree = index.tree
    _, first = tree.query(p)
    def project(points, ids):
        u = np.clip(((points-a[ids])*delta[ids]).sum(-1)/np.maximum(length2[ids], 1e-20), 0, 1)
        q = a[ids]+u[:, None]*delta[ids]
        return np.linalg.norm(points-q, axis=-1), q, u
    best = project(p, first)[0]
    # Preserve the traversal order of the former single-point queries, including
    # first-candidate tie breaking. SciPy otherwise sorts batched query results.
    groups = tree.query_ball_point(p, best+index.half+1e-8, return_sorted=False)
    counts = np.fromiter(map(len, groups), dtype=np.int64, count=len(p))
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    ids = np.concatenate(groups).astype(np.int64, copy=False)
    distance, nearest, u = project(np.repeat(p, counts, axis=0), ids)
    minima = np.minimum.reduceat(distance, offsets)
    positions = np.arange(len(ids))
    chosen = np.minimum.reduceat(np.where(distance == np.repeat(minima, counts),
                                         positions, len(ids)), offsets)
    return distance[chosen], nearest[chosen], ids[chosen], u[chosen]


def traversed_voxels(points):
    """Every nearest-neighbor voxel traversed by each continuous segment.

    Split at half-integer cell faces, rather than hoping regularly spaced
    samples catch a short excursion through an unsupported voxel.
    """
    cells, tangents = [], []
    for a, b in zip(points[:-1], points[1:]):
        delta = b-a
        if np.linalg.norm(delta) < 1e-9:
            continue
        cuts = [0., 1.]
        for axis in range(3):
            if abs(delta[axis]) < 1e-12:
                continue
            low, high = sorted((a[axis], b[axis]))
            faces = np.arange(np.ceil(low-.5), np.floor(high-.5)+1)+.5
            cuts.extend(((faces-a[axis])/delta[axis]).tolist())
        cuts = np.unique(np.clip(cuts, 0, 1))
        middle = (cuts[:-1]+cuts[1:])/2
        sample = a+np.r_[0., middle, 1.][:, None]*delta
        cells.append(np.round(sample).astype(int))
        tangents.append(np.tile(normalize(delta), (len(sample), 1)))
    return np.concatenate(cells), np.concatenate(tangents)


def seed_pairs(presence, directions, origin_xyz, center, tangent, target, cfg, *, target_index=None):
    """Two distant ridge points in one uncut, high-confidence component.

    Do not erase the target before labeling: that could turn an annotated
    fiber's flank into a seemingly separate negative component.
    """
    alignment = np.abs(directions @ tangent)
    support = (presence >= cfg.path_presence*255) & (alignment >= np.cos(np.deg2rad(cfg.max_angle)))
    high = support & (presence >= cfg.seed_presence*255)
    labels, count = ndimage.label(high)  # six-connected, deliberately conservative
    low_labels, _ = ndimage.label(support)
    objects = ndimage.find_objects(labels)
    # A certified lower bound, accounting for gaps between annotation samples.
    tree = target_index.seed_tree if target_index is not None else cKDTree(dense_line(target, .5))
    candidates = []
    for component in range(1, count+1):
        region = objects[component-1]
        local = np.argwhere(labels[region] == component)
        if len(local) < 12:
            continue
        local += np.array([s.start for s in region])
        world = local[:, ::-1] + origin_xyz
        dist = tree.query(world)[0] - .25
        if dist.min() <= cfg.exclusion:
            continue
        axial = (world-center) @ tangent
        choices = []
        for position in (-cfg.seed_spacing/2, cfg.seed_spacing/2):
            select = np.flatnonzero(np.abs(axial-position) <= 1.)
            if not len(select):
                break
            pts = world[select]
            weights = presence[tuple(local[select].T)].astype(float)**4
            centroid = np.average(pts, axis=0, weights=weights)
            choices.append(select[np.argmin(np.linalg.norm(pts-centroid, axis=1))])
        if len(choices) != 2:
            continue
        pair = world[choices]
        displacement = pair[1]-pair[0]
        if np.linalg.norm(displacement) < cfg.seed_spacing-2 or abs(normalize(displacement) @ tangent) < np.cos(np.deg2rad(cfg.max_angle)):
            continue
        seed_distance = tree.query(pair)[0]
        if seed_distance.max() > cfg.max_distance or seed_distance.min()-.25 < cfg.min_distance:
            continue
        support_id = int(low_labels[tuple(local[choices[0]])])
        candidates.append((float(tree.query(pair)[0].mean()), pair, component, support_id))
    candidates.sort(key=lambda x: (x[0], x[2]))
    return candidates, low_labels


def trace_controls(native, field, controls, native_config, cfg):
    """Trace through two or more controls, then extrapolate from robust end tangents."""
    controls = np.ascontiguousarray(controls, dtype=np.float64)
    if controls.ndim != 2 or controls.shape[1] != 3 or len(controls) < 2 or not np.isfinite(controls).all():
        raise ValueError('Supply at least two finite xyz control points')
    scale = cfg.grid_scale/field.trace_to_base_scale
    reference = controls*scale
    parts, meeting = [], []
    for i in range(len(controls)-1):
        result = native.trace_segment(field, reference, i, i+1, native_config)
        if not result.accepted:
            return None, {'reason': 'native_segment_'+result.reason}
        path = result.fused_line/scale
        if len(path) < 2:
            return None, {'reason': 'native_empty_segment'}
        parts.append(path if not parts else path[1:])
        meeting.append(float(result.meeting_error_trace_voxels/scale))
    trunk = np.concatenate(parts)
    out = trunk
    s = arclength(trunk)
    seed_center = s[-1]/2
    extension = cfg.extrapolation if cfg.max_path_length is None else max(0., (cfg.max_path_length-s[-1])/2)
    if extension:
        if s[-1] < 2:
            return None, {'reason': 'native_short_seed'}
        tangents = (trunk[0]-interp_at(trunk, s, [min(4., s[-1])])[0],
                    trunk[-1]-interp_at(trunk, s, [max(0., s[-1]-4.)])[0])
        tails = []
        for p, direction in zip(trunk[[0, -1]], tangents):
            tail = native.trace_extrapolation(field, p*scale, normalize(direction), extension*scale, native_config)
            if not tail.reached_trace_length and cfg.min_path_length is None:
                return None, {'reason': 'native_tail_'+tail.reason}
            points = np.asarray(tail.points)/scale
            # A stopped extrapolation may still supply a usable prefix. Its
            # retained geometry must pass all ordinary path checks below.
            if not len(points):
                points = p[None]
            if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                    or not np.allclose(points[0], p, atol=1e-6, rtol=0)):
                return None, {'reason': 'native_invalid_tail'}
            tails.append(points)
        out = np.concatenate([tails[0][:0:-1], trunk, tails[1][1:]])
        seed_center += arclength(tails[0])[-1]
    detail = {'reason': 'traced', 'meeting_errors': meeting}
    if cfg.min_path_length is not None:
        detail['seed_center_arc'] = float(seed_center)
    return out, detail


def clip_path(path, start, length):
    """Cut at exact arclengths without shortcutting any original corners."""
    s = arclength(path)
    endpoints = interp_at(path, s, [start, start+length])
    return np.concatenate([endpoints[:1], path[(s > start) & (s < start+length)], endpoints[1:]])


def validated_path(path, target, target_s, origin_xyz, presence, directions, support_labels, support_id, cfg,
                   *, target_index=None, seed_center_arc=None):
    """Prefer a long validated seed-centered window, with an optional minimum.

    Shortening never bypasses validation. Bisection is a conservative search,
    not a guarantee of finding every possible valid subpath or window position.
    """
    def check(candidate):
        return validate_path(candidate, target, target_s, origin_xyz, presence, directions,
                             support_labels, support_id, cfg, target_index=target_index)
    if cfg.min_path_length is None:
        ok, detail = check(path)
        return (path if ok else None), detail
    total = float(arclength(path)[-1])
    if total < cfg.min_path_length:
        return None, dict(reason='path_too_short', length=total)
    center = total/2 if seed_center_arc is None else seed_center_arc
    def window(length):
        return clip_path(path, float(np.clip(center-length/2, 0, total-length)), length)
    high = min(total, cfg.max_path_length)
    candidate = window(high)
    ok, detail = check(candidate)
    attempts, maximum_rejection = 1, None
    if not ok:
        maximum_rejection = detail['reason']
        low = cfg.min_path_length
        if high == low:
            return None, detail
        candidate = window(low)
        ok, detail = check(candidate)
        attempts += 1
        if not ok:
            return None, dict(detail, max_length_rejection=maximum_rejection)
        while high-low > cfg.sample_step:
            mid = (low+high)/2
            trial = window(mid)
            passed, trial_detail = check(trial)
            attempts += 1
            if passed:
                low, candidate, detail = mid, trial, trial_detail
            else:
                high = mid
    return candidate, dict(detail, untrimmed_length=total, validation_attempts=attempts,
                           max_length_rejection=maximum_rejection)


def validate_path(path, target, target_s, origin_xyz, presence, directions, support_labels, support_id, cfg, *, target_index=None):
    """Independent conservative checks; ambiguity is rejected, never relabeled."""
    dense = dense_line(path, cfg.sample_step)
    local = dense-origin_xyz
    upper = np.asarray(presence.shape[::-1])-2
    if np.any(local < 1) or np.any(local > upper):
        return False, {'reason': 'crop_boundary'}
    # Test the actual continuous path through the nearest-voxel component,
    # including visits too short to appear in the regular .25-voxel samples.
    cells, cell_tangents = traversed_voxels(path-origin_xyz)
    idx = tuple(cells[:, ::-1].T)
    if np.any(support_labels[idx] != support_id):
        return False, {'reason': 'left_presence_component'}
    minimum = float(presence[idx].min()/255)
    angles = np.rad2deg(np.arccos(np.clip(np.abs(np.einsum('ij,ij->i', directions[idx], cell_tangents)), 0, 1)))
    angle_max = float(angles.max())
    interpolated = ndimage.map_coordinates(presence.astype(float)/255, local[:, ::-1].T, order=1, prefilter=False)
    minimum = min(minimum, float(interpolated.min()))
    tangent = normalize(np.gradient(dense, axis=0))
    if minimum < cfg.path_presence or angle_max > cfg.max_angle:
        return False, {'reason': 'weak_presence_or_direction', 'min_presence': minimum, 'max_angle': angle_max}
    distances, _, seg, fraction = exact_nearest(dense, target, target_index)
    # Distance to a set is 1-Lipschitz: subtract half the largest dense step
    # to certify the entire continuous candidate, including between samples.
    half_step = np.linalg.norm(np.diff(dense, axis=0), axis=1).max()/2
    lower_bound = float(distances.min()-half_step)
    if lower_bound <= cfg.exclusion:
        return False, {'reason': 'target_exclusion', 'min_distance_lower_bound': lower_bound}
    if lower_bound < cfg.min_distance:
        return False, {'reason': 'inside_search_band', 'min_distance_lower_bound': lower_bound}
    if distances.max()+half_step > cfg.max_distance:
        return False, {'reason': 'not_nearby'}
    arc = target_s[seg]+fraction*np.diff(target_s)[seg]
    if arc.min() < cfg.seed_spacing or arc.max() > target_s[-1]-cfg.seed_spacing:
        return False, {'reason': 'annotation_boundary'}
    gt_tangent = normalize(target[seg+1]-target[seg])
    agreement = np.abs(np.einsum('ij,ij->i', gt_tangent, tangent))
    if agreement.min() < np.cos(np.deg2rad(cfg.max_angle)):
        return False, {'reason': 'different_fiber_type'}
    # A parallel path must progress beside the annotated span, not double back.
    delta = np.diff(arc)
    sign = 1 if arc[-1] >= arc[0] else -1
    if np.any(sign*delta < -.1) or abs(arc[-1]-arc[0]) < .85*arclength(dense)[-1]:
        return False, {'reason': 'ambiguous_projection'}
    return True, dict(reason='passed_automatic_checks', min_presence=minimum, max_direction_angle=angle_max,
                      min_distance_lower_bound=lower_bound, max_distance=float(distances.max()),
                      target_arc_range=[float(arc.min()), float(arc.max())], length=float(arclength(dense)[-1]))


def review_image(destination, fiber, path, controls, metrics, presence, ct, ct_scale):
    """One common flattened view of both paths and their actual 3D separation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from vesuvius.neural_tracing.fiber_follow.shared.diag import _gt_frames

    lo, hi = metrics['target_arc_range']
    arc = np.arange(max(0, lo-6), min(fiber.length, hi+6), .5)
    gt = interp_at(fiber.points, fiber.s, arc)
    frames = _gt_frames(gt)
    # Orient the first view toward the neighbor for readability. The second
    # view retains the orthogonal offset instead of silently projecting it away.
    mid = len(gt)//2
    q = path[np.argmin(np.linalg.norm(path-gt[mid], axis=1))]-gt[mid]
    uv = q @ frames[mid, :, :2]
    theta = np.arctan2(uv[1], uv[0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    frames[:, :, :2] = frames[:, :, :2] @ rot
    def project(line):
        _, nearest, seg, t = exact_nearest(line, gt)
        s = arc[seg]+t*np.diff(arc)[seg]
        off = np.einsum('ni,nij->nj', line-nearest, frames[seg])
        return s, off
    line_s, line_off = project(path)
    seed_s, seed_off = project(controls)
    half = max(8., float(np.ceil(np.abs(line_off[:, :2]).max()+4)))
    lateral = np.arange(-half, half+.25, .25)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                             gridspec_kw={'height_ratios': [1., 1., .38]})
    cyan, orange = '#00dcff', '#ff951a'
    # Use the view oriented toward the neighbor. A second projection can make
    # separate paths look coincident, so show true 3D distance instead.
    depth_lo = min(0., line_off[:, 1].min())-1
    depth_hi = max(0., line_off[:, 1].max())+1
    depths = np.arange(depth_lo, depth_hi+.5, .5)
    base = gt[:, None, :]+lateral[None, :, None]*frames[:, None, :, 0]
    ct_img = np.zeros(base.shape[:2], float)
    pr_img = np.zeros_like(ct_img)
    for depth in depths:
        world = base+depth*frames[:, None, :, 1]
        pr_img = np.maximum(pr_img, presence.sample_nearest(world[..., ::-1])/255)
        if ct is not None:
            ct_img = np.maximum(ct_img, ct.sample_nearest(world[..., ::-1]*ct_scale)/255)
    for row, img in enumerate((ct_img if ct is not None else pr_img, pr_img)):
        ax = axes[row]
        ax.imshow(img.T, origin='lower', aspect='auto', cmap='gray', vmin=0, vmax=1,
                  extent=(arc[0], arc[-1], -half, half))
        ax.axhline(0, color=cyan, lw=2.2, ls='--')
        ax.plot(line_s, line_off[:, 0], color=orange, lw=2.2)
        ax.scatter(seed_s, seed_off[:, 0], marker='D', s=55, facecolors='none', edgecolors=orange, linewidths=2)
        ax.set_title(f'Both paths over flattened {"CT" if row == 0 and ct is not None else "presence"}', fontsize=12)
        ax.set_ylabel('Lateral offset (voxels)')
        x = min(line_s[0], line_s[-1])
        j = int(np.argmin(line_s))
        ax.annotate('PROPOSED NEGATIVE', xy=(line_s[j], line_off[j, 0]),
                    xytext=(x+1, min(half-1.5, line_off[j, 0]+3)), color=orange,
                    weight='bold', fontsize=10,
                    bbox=dict(facecolor='black', alpha=.8, edgecolor='none', pad=4),
                    arrowprops=dict(arrowstyle='-', color=orange, lw=1.5))
        ax.annotate('TARGET ANNOTATION', xy=(x+1, 0), xytext=(x+1, -3),
                    color=cyan, weight='bold', fontsize=10,
                    bbox=dict(facecolor='black', alpha=.8, edgecolor='none', pad=4),
                    arrowprops=dict(arrowstyle='-', color=cyan, lw=1.5))
    distances, _, _, _ = exact_nearest(path, fiber.points)
    axes[2].plot(line_s, distances, color='#563090', lw=2)
    axes[2].fill_between(line_s, 0, distances, color='#563090', alpha=.15)
    axes[2].axhline(0, color='black', lw=.7)
    axes[2].set_ylim(0, max(4, float(distances.max())+1))
    axes[2].set_title('Actual 3D separation from the annotated target (zero would mean touching)', fontsize=11)
    axes[2].set_ylabel('3D distance\n(voxels)')
    axes[2].set_xlabel('Distance along target annotation (fiber-grid voxels)')
    legend = [Line2D([0], [0], color=cyan, lw=2, ls='--', label='Target annotation'),
              Line2D([0], [0], color=orange, lw=2, label='Proposed negative'),
              Line2D([0], [0], color=orange, marker='D', markerfacecolor='none', ls='', label='Two initialization points')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .93), ncol=3, fontsize=11)
    fig.suptitle(f'{destination.stem}  |  Target: {fiber.name} ({fiber.tag})\n'
                 f'Minimum 3D separation ≥ {metrics["min_distance_lower_bound"]:.2f} voxels · '
                 f'min. presence {metrics["min_presence"]:.2f} · length {metrics["length"]:.1f} voxels', fontsize=12)
    fig.text(.5, .01, f'Both large panels show the SAME PAIR · flattened slab {depth_hi-depth_lo:.1f} voxels thick · pending human review',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .89))
    fig.savefig(destination, dpi=140)
    plt.close(fig)



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fibers', required=True)
    ap.add_argument('--fiber-zarrs', required=True)
    ap.add_argument('--ct', help='CT zarr array, including its level')
    ap.add_argument('--ct-grid-scale', type=float, default=8.)
    ap.add_argument('--native-build-python', help='VC build directory ending in /python, for use without installing')
    ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'output'/'neighbor_negatives')
    ap.add_argument('--count', type=int, default=24)
    ap.add_argument('--max-anchors', type=int, default=200)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--val-z', nargs=2, type=float, default=(45000., 48500.), help='Excluded band in base voxels')
    ap.add_argument('--seed-spacing', type=float, default=20.)
    ap.add_argument('--extrapolation', type=float, default=10.)
    ap.add_argument('--min-path-length', type=float, help='Minimum retained path arclength; requires --max-path-length')
    ap.add_argument('--max-path-length', type=float, help='Maximum retained path arclength; overrides --extrapolation')
    ap.add_argument('--block-size', type=int, default=80)
    ap.add_argument('--min-distance', type=float, default=0., help='Inner search radius from the target, in trace-grid voxels')
    ap.add_argument('--max-distance', type=float, default=12., help='Outer search radius from the target, in trace-grid voxels')
    args = ap.parse_args()
    cfg = MiningConfig(seed_spacing=args.seed_spacing, extrapolation=args.extrapolation, block_size=args.block_size,
                       min_path_length=args.min_path_length, max_path_length=args.max_path_length,
                       min_distance=args.min_distance, max_distance=args.max_distance)
    if args.count <= 0 or args.max_anchors <= 0 or args.ct_grid_scale <= 0 or args.val_z[0] >= args.val_z[1]:
        ap.error('Counts, scales and validation band must be valid')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite a nonempty review directory: {args.output}')
    manifest = args.output/'predictions.lasagna.json'
    manifest.write_text(json.dumps(prediction_manifest(args.fiber_zarrs), indent=2))
    resources = MiningResources(manifest, args.native_build_python, args.ct, args.ct_grid_scale, cfg.grid_scale)
    native, field, native_cfg = resources.native, resources.field, resources.native_config
    volumes, ct = resources.volumes, resources.ct
    band = ZBand(*(np.asarray(args.val_z)/cfg.grid_scale))
    fibers, _ = split_fibers(load_fibers(args.fibers, grid_scale=cfg.grid_scale), band)
    fibers = [f for f in fibers if f.length > 2*cfg.block_size]
    if not fibers:
        raise ValueError('No sufficiently long training annotations outside the held-out split')
    rng = np.random.default_rng(args.seed)
    records, attempts, rejected, timings = [], [], Counter(), []
    started = time.perf_counter()
    # Cycle shuffled annotations: keep the first review batch diverse.
    order = rng.permutation(len(fibers))
    for anchor in range(args.max_anchors):
        f = fibers[order[anchor % len(order)]]
        t = float(rng.uniform(cfg.block_size/2, f.length-cfg.block_size/2))
        center = interp_at(f.points, f.s, [t])[0]
        origin = np.floor(center).astype(int)-cfg.block_size//2
        if origin[2] < band.hi and origin[2]+cfg.block_size > band.lo:
            rejected['heldout_crop'] += 1
            continue
        raw = {c: v.read(origin[::-1], (cfg.block_size,)*3) for c, v in volumes.items()}
        directions = native.decode_directions(raw['nx'].ravel(), raw['ny'].ravel()).reshape((*raw['nx'].shape, 3))
        pairs, support = seed_pairs(raw['presence'], directions, origin, center, tangent_at(f.points, f.s, t), f.points, cfg)
        if not pairs:
            rejected['no_confident_seed_pair'] += 1
        accepted_here = 0
        for _, controls, component, support_id in pairs[:4]:
            before = time.perf_counter()
            path, detail = trace_controls(native, field, controls, native_cfg, cfg)
            elapsed = time.perf_counter()-before
            timings.append(elapsed)
            if path is not None:
                path, detail = validated_path(path, f.points, f.s, origin, raw['presence'], directions, support,
                                              support_id, cfg, seed_center_arc=detail.get('seed_center_arc'))
            attempts.append(dict(target=f.name, target_arc=t, seeds_xyz=controls.tolist(), native_seconds=elapsed, **detail))
            if path is None:
                rejected[detail['reason']] += 1
                continue
            # Suppress repeat discoveries of the same nearby traced span.
            if any(r['target'] == f.name and cKDTree(np.asarray(r['negative_xyz'])).query(path)[0].max() < 2 for r in records):
                rejected['duplicate'] += 1
                continue
            name = f'candidate_{len(records)+1:03d}'
            record = dict(id=name, status='pending_review', target=f.name, target_hash=f.source_hash,
                          target_type=f.tag, target_arc=t, grid_scale=cfg.grid_scale,
                          target_xyz=f.points.tolist(), negative_xyz=path.tolist(), seeds_xyz=controls.tolist(),
                          high_component=component, support_component=support_id, metrics=detail, native_seconds=elapsed)
            review_image(args.output/(name+'.png'), f, path, controls, detail, volumes['presence'], ct, cfg.grid_scale/args.ct_grid_scale)
            (args.output/(name+'.json')).write_text(json.dumps(record, indent=2))
            records.append(record)
            accepted_here += 1
            print(f'{name}: {f.tag} distance>={detail["min_distance_lower_bound"]:.2f}, native={elapsed:.3f}s', flush=True)
            if len(records) >= args.count or accepted_here >= 2:
                break
        if len(records) >= args.count:
            break
        if anchor % 10 == 0:
            print(f'anchors={anchor+1}, candidates={len(records)}, rejected={dict(rejected)}', flush=True)
    report = dict(config=asdict(cfg), native_config=native_cfg.to_dict(), arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  candidates=len(records), attempted_traces=len(timings), rejected=dict(rejected), total_seconds=time.perf_counter()-started,
                  native_timing_seconds=({k: float(v) for k,v in zip(('mean','p50','p95'), (np.mean(timings), np.median(timings), np.quantile(timings,.95)))} if timings else {}))
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    (args.output/'attempts.json').write_text(json.dumps(attempts, indent=2))
    entries = '\n'.join(f'<article><h2>{r["id"]} · {html.escape(r["target_type"])}</h2><a href="{r["id"]}.json">Coordinates and checks</a><img loading="lazy" src="{r["id"]}.png"></article>' for r in records)
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Neighbor fiber review</title><style>body{font:16px sans-serif;max-width:1500px;margin:24px auto;background:#eee}img{width:100%}article{background:white;padding:12px;margin:24px 0}</style><h1>Proposed negative fibers — pending review</h1><p>Cyan dashed: annotated target. Orange: proposed negative. Diamonds: initialization points. Both large panels show the SAME PAIR over flattened CT or presence. The bottom plot shows actual 3D separation. These candidates have not been added to training.</p>'+entries)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
