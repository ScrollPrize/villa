"""Recovery-aware rollout diagnostics, independent of legacy first-departure scores.

All distances are trace-grid voxels. Defaults correspond to a 6 CT-voxel
tolerance, 1 CT-voxel sampling interval and 3 CT-voxel event persistence.
These measure agreement with an annotation, not independently verified identity.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

VERSION = 1


class PolylineProjector:
    """Exact nearest-segment projection, with a bounded spatial-tree fallback."""

    def __init__(self, fiber):
        self.a = np.asarray(fiber.points[:-1], dtype=float)
        self.v = np.diff(fiber.points, axis=0)
        self.length = np.linalg.norm(self.v, axis=1)
        if not len(self.length) or np.any(self.length <= 0):
            raise ValueError('Annotation must contain nonzero segments')
        self.s = np.asarray(fiber.s[:-1])
        self.mid = self.a + self.v / 2
        self.tree = cKDTree(self.mid)
        self.half_max = float(self.length.max() / 2)

    def project(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        k = min(8, len(self.a))
        dm, ids = self.tree.query(points, k=k)
        dm, ids = np.asarray(dm).reshape(len(points), k), np.asarray(ids).reshape(len(points), k)
        delta = points[:, None] - self.a[ids]
        u = np.clip(np.sum(delta * self.v[ids], axis=-1) / self.length[ids]**2, 0, 1)
        ds = np.linalg.norm(delta - u[..., None] * self.v[ids], axis=-1)
        best = np.argmin(ds, axis=1)
        ii = np.arange(len(points))
        distance = ds[ii, best]
        arc = self.s[ids[ii, best]] + u[ii, best] * self.length[ids[ii, best]]
        # Any closer segment has a midpoint no farther than best distance +
        # half its length. Inspect every such candidate if k was insufficient.
        if k < len(self.a):
            ambiguous = np.flatnonzero(dm[:, -1] <= distance + self.half_max + 1e-10)
            for i in ambiguous:
                js = np.asarray(self.tree.query_ball_point(points[i], distance[i] + self.half_max + 1e-10))
                delta = points[i] - self.a[js]
                uu = np.clip(np.sum(delta * self.v[js], axis=1) / self.length[js]**2, 0, 1)
                dd = np.linalg.norm(delta - uu[:, None] * self.v[js], axis=1)
                j = int(np.argmin(dd))
                distance[i] = dd[j]
                arc[i] = self.s[js[j]] + uu[j] * self.length[js[j]]
        return distance, arc


def _samples(path, step):
    """Retain every input vertex, subdividing long edges without rounding length."""
    chunks = [path[:1]]
    for a, b in zip(path[:-1], path[1:]):
        length = float(np.linalg.norm(b - a))
        if length > 1e-12:
            n = max(1, int(np.ceil(length / step)))
            chunks.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    p = np.concatenate(chunks)
    return p, np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _union_length(intervals):
    total = 0.
    right = -np.inf
    for lo, hi in sorted(intervals):
        total += max(0., hi - max(lo, right))
        right = max(right, hi)
    return float(total)


def _weighted_quantiles(values, weights, quantiles):
    if not len(values) or weights.sum() <= 0:
        return [None] * len(quantiles)
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    return [float(values[order[min(np.searchsorted(cumulative, q * cumulative[-1]), len(order)-1)]]) for q in quantiles]


def recovery_score(path, fiber, t0, sign, *, tol=3., sample_step=.5,
                   persistence=1.5, max_len=6000., projector=None):
    """Local arclength precision, unique directed coverage, and recoverable events.

    Endpoint distances interpolate linearly within <=sample_step edges when
    partitioning tolerance crossings. Coverage conservatively requires both
    edge endpoints in tolerance and a reachable, forward GT correspondence.
    Nearest-segment jumps cannot credit skipped annotation intervals. Excursion
    confirmation and recovery require persistence units of travelled length;
    event boundaries are backdated to the start of the corresponding run.
    """
    path = np.asarray(path, dtype=float).reshape(-1, 3)
    if not len(path) or not np.isfinite(path).all() or sign not in (-1, 1):
        raise ValueError('A finite nonempty path and direction +/-1 are required')
    if min(tol, sample_step, persistence, max_len) <= 0:
        raise ValueError('Scoring lengths must be positive')
    projector = projector or PolylineProjector(fiber)
    p, travelled = _samples(path, sample_step)
    distance, arc = projector.project(p)
    progress = (arc - t0) * sign
    available_full = float(fiber.length - t0 if sign > 0 else t0)
    available = min(available_full, max_len)
    near = distance <= tol
    coherent = np.zeros(len(p), dtype=bool)
    anchor_progress, anchor_travel = 0., 0.
    for i in range(len(p)):
        reachable = abs(progress[i] - anchor_progress) <= travelled[i] - anchor_travel + 2 * tol + 1e-9
        coherent[i] = near[i] and reachable and progress[i] >= -tol
        if coherent[i]:
            anchor_progress, anchor_travel = progress[i], travelled[i]

    # Independent endpoint censoring can follow a recovered section. A distant
    # crossing of the endpoint plane without supported local progress is ignored.
    endpoint = fiber.points[-1 if sign > 0 else 0]
    tangent = fiber.points[-1] - fiber.points[-2] if sign > 0 else fiber.points[0] - fiber.points[1]
    tangent = tangent / np.linalg.norm(tangent)
    forward = (p - endpoint) @ tangent
    crossing = 0. if available_full <= 1e-9 and near[0] else None
    for i in np.flatnonzero((forward[:-1] < 0) & (forward[1:] >= 0)):
        if crossing is not None:
            break
        edge = travelled[i+1] - travelled[i]
        remaining = available_full - progress[i]
        if not coherent[i] or remaining > edge + 2 * tol:
            continue
        alpha = -forward[i] / (forward[i+1] - forward[i])
        hit = p[i] + alpha * (p[i+1] - p[i])
        if np.linalg.norm(hit - endpoint) <= tol:
            crossing = float(travelled[i] + alpha * edge)
    known = bool(fiber.endpoint_stop[0 if sign < 0 else 1])
    scored_end = float(travelled[-1] if crossing is None or known else crossing)
    unknown = float(travelled[-1] - scored_end)
    physical_end = float(travelled[-1] if crossing is None else crossing)

    # Build within/outside runs using linearly located distance crossings. All
    # portions after a tagged physical endpoint are wrong, regardless of radius.
    runs = []
    intervals = []
    backtrack = 0.
    error_values, error_weights = [], []
    for i, (left, right) in enumerate(zip(travelled[:-1], travelled[1:])):
        right = min(right, scored_end)
        if right <= left:
            break
        fraction = (right - left) / (travelled[i+1] - left)
        d0, d1 = distance[i], distance[i] + fraction * (distance[i+1] - distance[i])
        breaks = [left, right]
        if (d0 <= tol) != (d1 <= tol):
            breaks.append(float(left + (right-left) * (tol-d0)/(d1-d0)))
        if left < physical_end < right:
            breaks.append(physical_end)
        breaks.sort()
        # Far geometric points remain wrong. A near point on a disconnected
        # or unreachable portion cannot become a new correspondence anchor.
        reachable_edge = (coherent[i] or not near[i]) and (coherent[i+1] or not near[i+1])
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            if hi-lo <= 1e-12:
                continue
            dmid = d0 + ((lo+hi)/2-left)/(right-left)*(d1-d0)
            good = dmid <= tol and reachable_edge and (lo+hi)/2 < physical_end
            if runs and runs[-1][2] == good:
                runs[-1][1] = hi
            else:
                runs.append([lo, hi, bool(good)])
            if (lo+hi)/2 < physical_end:
                error_values.append(dmid); error_weights.append(hi-lo)
        if coherent[i] and coherent[i+1] and left < physical_end:
            delta = progress[i+1] - progress[i]
            if delta >= 0 and delta <= travelled[i+1]-left + 2*tol:
                lo = max(0., progress[i])
                hi = min(available, progress[i] + delta * fraction)
                if hi > lo:
                    intervals.append((lo, hi))
            elif delta < 0:
                backtrack += right-left
    correct = sum(hi-lo for lo, hi, good in runs if good)
    wrong = scored_end - correct
    coverage_length = _union_length(intervals)
    events = []
    active = None
    for lo, hi, good in runs:
        if not good and active is None and hi-lo >= persistence:
            active = dict(start=float(lo), end=None, recovered=False)
        elif good and active is not None and hi-lo >= persistence:
            active.update(end=float(lo), recovered=True, length=float(lo-active['start']))
            events.append(active); active = None
    if active is not None:
        active.update(end=scored_end, length=float(scored_end-active['start']))
        events.append(active)
    errors = np.asarray(error_values); weights = np.asarray(error_weights)
    p50, p90, p95, p99 = _weighted_quantiles(errors, weights, [.5, .9, .95, .99])
    return dict(recovery_scoring_version=VERSION, local_tolerance=tol, local_sample_step=sample_step,
        excursion_persistence=persistence, local_correct_length=float(correct), local_wrong_length=float(wrong),
        local_unknown_length=unknown, local_scored_length=scored_end,
        local_precision=float(correct/scored_end) if scored_end else None,
        recovered_coverage_length=coverage_length, recovered_available=available,
        recovered_coverage=float(coverage_length/available) if available else None,
        first_sustained_departure_length=events[0]['start'] if events else None,
        uninterrupted_length=events[0]['start'] if events else min(scored_end, physical_end),
        uninterrupted_censored=not bool(events), excursion_count=len(events),
        recovery_count=sum(e['recovered'] for e in events), excursions=events,
        longest_excursion_length=max((e['length'] for e in events), default=0.),
        recovered_correct_length=float(sum(max(0., hi-max(lo, events[0]['start'])) for lo, hi, good in runs if good)) if events else 0.,
        backtrack_length=float(backtrack), local_endpoint_crossed=crossing is not None,
        local_endpoint_overrun_length=float(travelled[-1]-crossing) if crossing is not None and known else 0.,
        distance_mean=float(np.average(errors, weights=weights)) if len(errors) else None,
        distance_p50=p50, distance_p90=p90, distance_p95=p95, distance_p99=p99,
        distance_max=float(distance[travelled <= physical_end+1e-9].max()),
        local_total_length=float(travelled[-1]),
        stop_distance=float(distance[-1]), stop_gt_progress=float(progress[-1]),
        stop_remaining_annotation=max(0., available_full-float(progress[-1])))
