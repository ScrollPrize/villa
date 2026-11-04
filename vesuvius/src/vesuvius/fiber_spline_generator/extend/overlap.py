"""Conservative matches between overlapping, same-family 3D fibre polylines.

compare(A, B) returns None, an ``overlap`` match, or a ``duplicate`` descriptor.
Overlap spliceA/spliceB are arclengths along the curves AFTER flipA/flipB.
All distances are native voxels; all coordinates stay in the input XYZ frame.
The caller must resolve competing matches; geometrically indistinguishable
parallel fibres cannot be disambiguated by this pairwise function alone.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

STEP = 2.0
MIN_OVERLAP = 24.0
MAX_MEDIAN = 1.5
MAX_P90 = 3.0
MAX_ANGLE = 15.0
MIN_UNIQUE_TAIL = 8.0


def _curve(value):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError('Expected an Nx3 polyline with at least two points')
    if not np.isfinite(points).all():
        raise ValueError('Polyline contains non-finite coordinates')
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]
    if len(points) < 2:
        raise ValueError('Polyline has no nonzero-length segment')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return points, arc


def _at(points, arc, values):
    values = np.asarray(values)
    return np.stack([np.interp(values, arc, points[:, i]) for i in range(3)], axis=-1)


def _sample(points, arc):
    sample_arc = np.r_[np.arange(0., arc[-1], STEP), arc[-1]]
    return _at(points, arc, sample_arc), sample_arc


def _project(query, target, target_arc):
    """Continuous projections onto segments adjacent to the three nearest samples."""
    tree = cKDTree(target)
    _, nearest = tree.query(query, k=min(3, len(target)))
    if nearest.ndim == 1:
        nearest = nearest[:, None]
    indices = np.clip(np.concatenate((nearest - 1, nearest), axis=1), 0, len(target) - 2)
    starts = target[indices]
    deltas = target[indices + 1] - starts
    weights = np.sum((query[:, None, :] - starts) * deltas, axis=2)
    weights /= np.maximum(np.sum(deltas * deltas, axis=2), 1e-15)
    weights = np.clip(weights, 0., 1.)
    candidates = starts + weights[:, :, None] * deltas
    distance_squared = np.sum((query[:, None, :] - candidates) ** 2, axis=2)
    best = np.argmin(distance_squared, axis=1)
    rows = np.arange(len(query))
    index = indices[rows, best]
    along = target_arc[index] + weights[rows, best] * (target_arc[index + 1] - target_arc[index])
    return np.sqrt(distance_squared[rows, best]), along


def _tangent(points, arc, at, radius=6.):
    lo = max(0., float(at) - radius)
    hi = min(float(arc[-1]), float(at) + radius)
    tangent = _at(points, arc, hi) - _at(points, arc, lo)
    length = np.linalg.norm(tangent)
    return tangent / length if length > 1e-8 else None


def _angle(a, b):
    if a is None or b is None:
        return 180.
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1., 1.))))


def _monotonic(correspondence):
    # A quarter voxel tolerates numerical/local projection jitter, not a turn-back.
    return len(correspondence) >= 2 and bool(np.all(np.diff(correspondence) >= -.25))


def _has_loop(points):
    points, arc = _curve(points)
    sampled, sample_arc = _sample(points, arc)
    # Reject nonlocal returns. Adjoining sample pairs and the short splice bridge
    # are explicitly excluded by the >=12-voxel arclength separation.
    pairs = cKDTree(sampled).query_pairs(1.5, output_type='ndarray')
    return bool(len(pairs) and np.any(sample_arc[pairs[:, 1]] - sample_arc[pairs[:, 0]] >= 12.))


def splice(A, B, match):
    """Join only an accepted suffix/head overlap; never extrapolate an endpoint.

    The join retains a <=1.5-voxel bridge between two interior overlap locations.
    The original dense points before/after that join are retained unchanged.
    """
    if match is None or match.get('kind') != 'overlap':
        raise ValueError('Only kind=overlap may be spliced')
    a, _ = _curve(A)
    b, _ = _curve(B)
    a, sa = _curve(a[::-1] if match['flipA'] else a)
    b, sb = _curve(b[::-1] if match['flipB'] else b)
    at_a, at_b = float(match['spliceA']), float(match['spliceB'])
    if not (0. < at_a < sa[-1] and 0. < at_b < sb[-1]):
        raise ValueError('Splice must be interior to both curves')
    point_a = _at(a, sa, at_a)
    point_b = _at(b, sb, at_b)
    if np.linalg.norm(point_a - point_b) > MAX_MEDIAN + 1e-6:
        raise ValueError('Splice gap exceeds 1.5 voxels')
    combined = np.concatenate((a[sa < at_a - 1e-8], point_a[None], point_b[None], b[sb > at_b + 1e-8]))
    return _curve(combined)[0]


def _duplicate(a, sa, b, sb, d_a, on_b, d_b, on_a):
    """A fully supported contained curve is redundancy, not a continuation."""
    possibilities = []
    for contained, short, ss, long, sl, distance, projected in (
        ('A', a, sa, b, sb, d_a, on_b),
        ('B', b, sb, a, sa, d_b, on_a),
    ):
        if ss[-1] < MIN_OVERLAP or ss[-1] > sl[-1] + 2.:
            continue
        median = float(np.median(distance))
        p90 = float(np.percentile(distance, 90))
        if median > MAX_MEDIAN or p90 > MAX_P90 or float(distance.max()) > MAX_P90:
            continue
        flipped = projected[-1] < projected[0]
        correspondence = sl[-1] - projected if flipped else projected
        if not _monotonic(correspondence) or correspondence[-1] - correspondence[0] < .85 * ss[-1]:
            continue
        angles = []
        for short_s, long_s in ((0., projected[0]), (ss[-1], projected[-1])):
            long_tangent = _tangent(long, sl, long_s)
            if long_tangent is not None and flipped:
                long_tangent = -long_tangent
            angles.append(_angle(_tangent(short, ss, short_s), long_tangent))
        if max(angles) > MAX_ANGLE:
            continue
        # flipA=False chooses A's input direction as the descriptor's reference.
        possibilities.append({
            'kind': 'duplicate', 'contained': contained, 'flipA': False, 'flipB': bool(flipped),
            'score': median + .25 * p90 + .02 * max(angles), 'overlapLength': float(ss[-1]),
            'gap': median,
            'metrics': {'medianDistance': median, 'p90Distance': p90, 'endpointTangentAngle': max(angles),
                        'monotonic': True, 'reason': 'Full containment; do not join as a continuation'},
        })
    if len(possibilities) == 2:
        result = min(possibilities, key=lambda item: item['score'])
        result['contained'] = 'both'
        return result
    return possibilities[0] if possibilities else None


def compare(A, B):
    """Return a conservative oriented suffix(A)/prefix(B) overlap, or None.

    Input curves must already be filtered to the same fibre family. Both
    orientation choices are searched independently. Acceptance requires:
    * each overlap extends >=24 voxels and reaches A's end and B's beginning;
    * symmetric median <=1.5, p90 <=3 voxels and directed endpoint angle <=15°;
    * monotonic, well-covered nearest correspondences on both curves;
    * A contributes >8 voxels before overlap; B contributes >8 after overlap;
    * an interior splice <=1.5 voxels apart, with no nonlocal return in the join.

    ``score`` is lower-is-better, in voxel-weighted units, not a probability.
    Returned splice coordinates are arclengths along the oriented ORIGINAL
    curves; use splice(A,B,result) to recover the joined dense polyline.
    """
    original_a, original_sa = _curve(A)
    original_b, original_sb = _curve(B)
    a, sa = _sample(original_a, original_sa)
    b, sb = _sample(original_b, original_sb)
    if min(sa[-1], sb[-1]) < MIN_OVERLAP:
        return None
    d_a, on_b = _project(a, b, sb)
    d_b, on_a = _project(b, a, sa)
    duplicate = _duplicate(a, sa, b, sb, d_a, on_b, d_b, on_a)
    if duplicate is not None:
        return duplicate
    candidates = []
    for flip_a in (False, True):
        aa, saa = (a[::-1], sa[-1] - sa[::-1]) if flip_a else (a, sa)
        da = d_a[::-1] if flip_a else d_a
        original_aa, original_saa = _curve(original_a[::-1] if flip_a else original_a)
        for flip_b in (False, True):
            bb, sbb = (b[::-1], sb[-1] - sb[::-1]) if flip_b else (b, sb)
            db = d_b[::-1] if flip_b else d_b
            a_to_b = on_b[::-1] if flip_a else on_b
            if flip_b:
                a_to_b = sb[-1] - a_to_b
            b_to_a = on_a[::-1] if flip_b else on_a
            if flip_a:
                b_to_a = sa[-1] - b_to_a
            original_bb, original_sbb = _curve(original_b[::-1] if flip_b else original_b)
            if da[-1] > MAX_P90 or db[0] > MAX_P90:
                continue
            overlap_start_a = float(b_to_a[0])
            overlap_end_b = float(a_to_b[-1])
            overlap_a = float(saa[-1] - overlap_start_a)
            overlap_b = overlap_end_b
            if min(overlap_a, overlap_b) < MIN_OVERLAP:
                continue
            if overlap_start_a <= MIN_UNIQUE_TAIL or sbb[-1] - overlap_end_b <= MIN_UNIQUE_TAIL:
                continue
            if max(overlap_a, overlap_b) / min(overlap_a, overlap_b) > 1.25:
                continue
            mask_a = saa >= overlap_start_a - 1e-8
            mask_b = sbb <= overlap_end_b + 1e-8
            corr_a, corr_b = a_to_b[mask_a], b_to_a[mask_b]
            if not _monotonic(corr_a) or not _monotonic(corr_b):
                continue
            if corr_a[-1] - corr_a[0] < .80 * overlap_b or corr_b[-1] - corr_b[0] < .80 * overlap_a:
                continue
            if np.any(corr_a > overlap_end_b + 2.) or np.any(corr_b < overlap_start_a - 2.):
                continue
            median_a, median_b = float(np.median(da[mask_a])), float(np.median(db[mask_b]))
            p90_a, p90_b = float(np.percentile(da[mask_a], 90)), float(np.percentile(db[mask_b], 90))
            # Each direction must pass independently; a densely sampled easy side
            # must not conceal a worse correspondence in the opposite direction.
            median, p90 = max(median_a, median_b), max(p90_a, p90_b)
            if median > MAX_MEDIAN or p90 > MAX_P90:
                continue
            angles = [
                _angle(_tangent(aa, saa, overlap_start_a), _tangent(bb, sbb, 0.)),
                _angle(_tangent(aa, saa, saa[-1]), _tangent(bb, sbb, overlap_end_b)),
            ]
            max_angle = max(angles)
            if max_angle > MAX_ANGLE:
                continue
            margin = min(8., .20 * min(overlap_a, overlap_b))
            options = np.flatnonzero((saa >= overlap_start_a + margin) & (saa <= saa[-1] - margin)
                                     & (a_to_b >= margin) & (a_to_b <= overlap_end_b - margin) & (da <= MAX_MEDIAN))
            if not len(options):
                continue
            # Closest interior correspondence; arclength-centre distance breaks ties.
            centre = (overlap_start_a + saa[-1]) / 2.
            best = min(options, key=lambda i: (round(float(da[i]), 8), abs(float(saa[i]) - centre)))
            splice_a, splice_b = float(saa[best]), float(a_to_b[best])
            point_a = _at(original_aa, original_saa, splice_a)
            point_b = _at(original_bb, original_sbb, splice_b)
            gap = float(np.linalg.norm(point_a - point_b))
            if gap > MAX_MEDIAN:
                continue
            candidate = {
                'kind': 'overlap', 'flipA': flip_a, 'flipB': flip_b,
                'spliceA': splice_a, 'spliceB': splice_b,
                'pointA': point_a.tolist(), 'pointB': point_b.tolist(),
                'score': median + .25 * p90 + .02 * max_angle,
                'overlapLength': min(overlap_a, overlap_b), 'gap': gap,
                'metrics': {'medianDistance': median, 'p90Distance': p90,
                            'medianAtoB': median_a, 'medianBtoA': median_b,
                            'p90AtoB': p90_a, 'p90BtoA': p90_b,
                            'endpointTangentAngle': max_angle, 'monotonic': True,
                            'overlapA': overlap_a, 'overlapB': overlap_b,
                            'uniquePrefixA': overlap_start_a, 'uniqueSuffixB': float(sbb[-1] - overlap_end_b)},
            }
            combined = splice(A, B, candidate)
            if not _has_loop(combined):
                candidates.append(candidate)
    return min(candidates, key=lambda item: item['score']) if candidates else None
