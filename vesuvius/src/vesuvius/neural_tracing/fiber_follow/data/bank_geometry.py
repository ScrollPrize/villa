"""Geometry-only difficulty scores and continuous contact with trusted bank paths."""
import numpy as np

from vesuvius.neural_tracing.fiber_follow.data.neighbor_mining import exact_nearest
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at, normalize


DIFFICULTY_KINDS = ('near_similar', 'curved', 'converging')


def difficulty_scores(line, parent):
    """Compare centerlines at a fixed physical scale, independent of vertex density.

    Scores rank a bounded candidate pool, not annotation trust. Reversal leaves
    scores unchanged. Similar bending distinguishes lookalikes from crossings.
    """
    arc = arclength(line)
    s = np.linspace(0, arc[-1], max(3, int(np.ceil(arc[-1]/4))+1))
    points = interp_at(line, arc, s)
    distance, _, segment, u = exact_nearest(points, parent.points)
    matched = parent.s[segment]+u*np.diff(parent.s)[segment]
    tangent = normalize(np.gradient(points, s, axis=0))
    own = normalize(interp_at(parent.points, parent.s, np.minimum(parent.length, matched+4))
                    -interp_at(parent.points, parent.s, np.maximum(0, matched-4)))
    own *= np.where(np.einsum('ij,ij->i', tangent, own) < 0, -1., 1.)[:, None]
    bend = np.gradient(tangent, s, axis=0)
    own_bend = np.gradient(own, s, axis=0)
    similarity = np.abs(np.einsum('ij,ij->i', tangent, own))*np.exp(-8*np.linalg.norm(bend-own_bend, axis=-1))
    nearby = np.exp(-distance/6.)
    return np.array([np.mean(nearby*similarity),
                     np.quantile(np.linalg.norm(bend, axis=-1), .9),
                     np.max(nearby)*np.ptp(distance)/max(arc[-1], 1.)])


def _quadratic_interval(origin, velocity, radius):
    """Parameter interval in [0,1] inside a sphere (or projected cylinder)."""
    a = float(velocity @ velocity)
    b = float(origin @ velocity)
    c = float(origin @ origin-radius*radius)
    if a < 1e-24:
        return (0., 1.) if c <= 0 else None
    discriminant = b*b-a*c
    if discriminant < 0:
        return None
    root = np.sqrt(max(0., discriminant))
    lo, hi = max(0., (-b-root)/a), min(1., (-b+root)/a)
    return (lo, hi) if lo <= hi else None


def tube_intervals(a, b, line, radius):
    """Exact intersections of a segment with a polyline's closed radius tube.

    Each edge contributes its finite cylinder and endpoint spheres. This catches
    brief contacts between decision heads and between regularly spaced samples.
    """
    delta = b-a
    intervals = []
    # Cheap axis-aligned broad phase; never drops a possible capsule contact.
    low, high = np.minimum(a, b)-radius, np.maximum(a, b)+radius
    candidates = np.flatnonzero(np.all(np.maximum(line[:-1], line[1:]) >= low, axis=1)
                               & np.all(np.minimum(line[:-1], line[1:]) <= high, axis=1))
    for i in candidates:
        p, q = line[i:i+2]
        for end in (p, q):
            span = _quadratic_interval(a-end, delta, radius)
            if span is not None:
                intervals.append(span)
        edge = q-p
        length2 = float(edge @ edge)
        if length2 < 1e-24:
            continue
        start, speed = float((a-p) @ edge/length2), float(delta @ edge/length2)
        axial = (0., 1.)
        if abs(speed) < 1e-15:
            if not 0 <= start <= 1:
                continue
        else:
            bounds = sorted((-start/speed, (1-start)/speed))
            axial = max(0., bounds[0]), min(1., bounds[1])
        radial = _quadratic_interval(a-p-start*edge, delta-speed*edge, radius)
        if radial is not None:
            lo, hi = max(axial[0], radial[0]), min(axial[1], radial[1])
            if lo <= hi:
                intervals.append((lo, hi))
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]+1e-12:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def first_foreign_contact(segment, foreign, target, tolerance=.75, own_tolerance=1.5):
    """First entry into the foreign tube outside the intended annotation tube.

    Returns distance along the committed polyline and xyz, or None. Shared tubes
    are ambiguous and cannot certify a switch. No extrapolation past bank ends.
    """
    segment = np.asarray(segment, float)
    if len(segment) == 1:
        segment = np.repeat(segment, 2, axis=0)
    arc = arclength(segment)
    for j, (a, b) in enumerate(zip(segment[:-1], segment[1:])):
        foreign_spans = tube_intervals(a, b, foreign, tolerance)
        if not foreign_spans:
            continue
        own_spans = tube_intervals(a, b, target, own_tolerance)
        for lo, hi in foreign_spans:
            for x, y in own_spans:
                if y < lo:
                    continue
                if x > lo:
                    break
                lo = np.nextafter(y, np.inf)
            if lo <= hi:
                return float(arc[j]+lo*(arc[j+1]-arc[j])), a+lo*(b-a)
    return None


class BankSwitchDetector:
    """Match only certified relationships for the intended annotation."""
    def __init__(self, banks, tolerance=.75, own_tolerance=1.5):
        if not (np.isfinite(tolerance) and np.isfinite(own_tolerance)
                and 0 < tolerance <= own_tolerance):
            raise ValueError('Require finite 0 < bank tolerance <= annotation tolerance')
        self.banks = banks
        self.tolerance, self.own_tolerance = tolerance, own_tolerance

    def first_contact(self, fi, original_arc, segment):
        best = None
        for bank in self.banks:
            for record in bank.spatial_records(fi, segment, radius=self.tolerance):
                event = first_foreign_contact(segment, record['points'], bank.fibers[fi].points,
                                              self.tolerance, self.own_tolerance)
                if event is not None and (best is None or event[0] < best['distance']):
                    distance, pos = event
                    best = dict(distance=distance, pos=pos,
                        bank_path=f'{bank.root}/{record["shard"]}#{record["index"]}',
                        bank_run=bank.run['digest'])
        return best
