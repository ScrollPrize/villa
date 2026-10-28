"""Integer winding L1 via a capacitated minimum-cost circulation.

No fitted geometry, label bounds, uncertainty, or integer capacity rounding.
Seam transport is evaluated on the actual piecewise-linear paths.
"""
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def integer_targets(distance, sign, family_delta, parallel, offset=.5):
    """Same/next H/V targets; half ties choose smaller outward distance.

    Return both target and validity. Unsigned measurements are usable only
    when BOTH orientations give the same target, usually zero.
    """
    distance, sign, family_delta, parallel = np.broadcast_arrays(
        np.asarray(distance, float), sign, family_delta, np.asarray(parallel, bool))
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError('Invalid measured separation')
    if not np.isin(sign, [-1, 0, 1]).all():
        raise ValueError('Invalid separation sign')
    def directed(s):
        outward = distance - s * np.where(parallel, 0., offset * family_delta)
        minimum = (~parallel) & (s * family_delta < 0)
        return (s * np.maximum(np.ceil(outward - .5), minimum)).astype(np.int64)
    plus, minus = directed(1), directed(-1)
    return np.where(sign < 0, minus, plus), (sign != 0) | (plus == minus)


def fractional_targets(distance, sign, family_delta, parallel, offset=.5):
    """Calibrated signed centers, retaining the baseline outward-order clamp.

    Unsigned evidence retains its baseline unique integer target, if valid;
    selecting one of its fractional orientations would invent information.
    """
    rounded, valid = integer_targets(distance, sign, family_delta, parallel, offset)
    distance, sign, family_delta, parallel = np.broadcast_arrays(
        np.asarray(distance, float), sign, family_delta, np.asarray(parallel, bool))
    outward = distance-sign*np.where(parallel, 0., offset*family_delta)
    minimum = (~parallel) & (sign*family_delta < 0)
    target = sign*np.maximum(outward, minimum)
    return np.where(sign == 0, rounded, target), valid


def split_fractional_edges(a, b, target, weight):
    """Two integer targets with exactly the same L1 cost on integer labels.

    Returns a solve-compatible graph and the source edge index for every arc.
    Zero-weight upper arcs for integral targets are omitted.
    """
    a, b, target, weight = (np.asarray(v) for v in (a, b, target, weight))
    if target.ndim != 1 or any(v.shape != target.shape for v in (a, b, weight)):
        raise ValueError('Edge shapes disagree')
    if not np.isfinite(target).all() or not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError('Invalid fractional targets or weights')
    lower = np.floor(target).astype(np.int64)
    fraction = target-lower
    upper = fraction > 0
    source = np.r_[np.arange(len(a)), np.flatnonzero(upper)]
    return dict(a=a[source], b=b[source], target=np.r_[lower, lower[upper]+1],
                weight=np.r_[weight*(1-fraction), weight[upper]*fraction[upper]],
                one_sided=np.zeros(len(source), bool)), source


def solve(node_count, a, b, target, weight=None, one_sided=None, *, log_progress=False):
    """Minimize sum w|Bk-d|, or w max(0,d-Bk) on one-sided edges.

    Solve min -d.y, B.T y=0, -w<=y<=w (0<=y<=w for hinges).
    HiGHS dual simplex supplies node potentials. Integer arc costs guarantee
    integral optimal potentials; rounding is checked by a primal/dual proof.
    """
    started = time.monotonic()
    a, b, d = np.asarray(a, np.int64), np.asarray(b, np.int64), np.asarray(target)
    w = np.ones(len(a)) if weight is None else np.asarray(weight, float)
    hinge = np.zeros(len(a), bool) if one_sided is None else np.asarray(one_sided, bool)
    if any(x.shape != a.shape for x in (b, d, w, hinge)) or a.ndim != 1:
        raise ValueError('Edge shapes disagree')
    if node_count < 1 or np.any(a < 0) or np.any(b < 0) or np.any(a >= node_count) or np.any(b >= node_count):
        raise ValueError('Invalid graph node')
    if not np.isfinite(d).all() or not np.equal(d, np.rint(d)).all():
        raise ValueError('Targets must be finite integers')
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('Capacities must be finite and positive')
    if not len(a):
        return dict(winding=np.zeros(node_count, np.int64), flow=np.empty(0),
                    component=np.arange(node_count), certificate=dict(primal=0., dual=0., gap=0.,
                    components=node_count, seconds=time.monotonic()-started, optimal=True))
    e = np.arange(len(a))
    incidence = coo_matrix((np.r_[-np.ones(len(a)), np.ones(len(a))],
                           (np.r_[a, b], np.r_[e, e])), shape=(node_count, len(a))).tocsr()
    lo = np.where(hinge, 0., -w)
    result = linprog(-d.astype(float), A_eq=incidence, b_eq=np.zeros(node_count),
                     bounds=np.column_stack((lo, w)), method='highs-ds',options={'disp':log_progress})
    if not result.success:
        raise RuntimeError(result.message)
    return certify_flow(node_count, a, b, d, w, hinge, -result.eqlin.marginals,
                        result.x, seconds=time.monotonic()-started, iterations=int(result.nit), incidence=incidence)


def certify_flow(node_count, a, b, d, w, hinge, raw, y, *, seconds, iterations, incidence=None):
    """Shared unchanged integer primal/dual certificate for circulation backends."""
    if incidence is None:
        e = np.arange(len(a))
        incidence = coo_matrix((np.r_[-np.ones(len(a)), np.ones(len(a))],
                               (np.r_[a, b], np.r_[e, e])), shape=(node_count, len(a))).tocsr()
    lo = np.where(hinge, 0., -w)
    k = np.rint(raw).astype(np.int64)
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(node_count, node_count)).tocsr()
    count, component = connected_components(graph, directed=False)
    first=np.full(count,node_count,dtype=np.int64)
    np.minimum.at(first,component,np.arange(node_count))
    k-=k[first[component]]
    residual = d - (k[b] - k[a])
    primal = float(np.sum(w * np.where(hinge, np.maximum(residual, 0), abs(residual))))
    dual = float(d @ y)
    conservation = float(np.max(abs(incidence @ y), initial=0))
    bounds = float(max(np.max(lo-y, initial=0), np.max(y-w, initial=0)))
    gap = primal-dual
    tolerance = 1e-7 * max(1., abs(primal), abs(dual))
    if abs(gap) > tolerance or conservation > 1e-7 or bounds > 1e-7:
        raise RuntimeError(f'Integer optimality certificate failed: {gap=}, {conservation=}, {bounds=}')
    return dict(winding=k, flow=y, component=component, certificate=dict(
        primal=primal, dual=dual, gap=gap, conservation_max=conservation,
        capacity_violation_max=bounds, potential_rounding_max=float(np.max(abs(raw-np.rint(raw)))),
        components=int(count), seconds=seconds, optimal=True,
        backend='scipy.optimize.linprog/highs-ds capacitated circulation',
        iterations=iterations))


def aggregate_l1_edges(a,b,target,weight):
    """Combine identical unoriented L1 terms; never combine different targets.

    Deterministic float64 capacity sums, with no quantization/target rounding.
    This applies only to ordinary two-sided L1 edges, not one-sided hinges.
    """
    a,b,target,weight=(np.asarray(x) for x in (a,b,target,weight))
    if any(x.shape!=a.shape for x in (b,target,weight)):raise ValueError('Edge shapes disagree')
    if not len(a):return dict(a=a,b=b,target=target,weight=weight)
    swap=a>b
    aa=np.minimum(a,b);bb=np.maximum(a,b);tt=np.where(swap,-target,target)
    order=np.lexsort((tt,bb,aa));aa=aa[order];bb=bb[order];tt=tt[order]
    starts=np.r_[0,np.flatnonzero((np.diff(aa)!=0)|(np.diff(bb)!=0)|(np.diff(tt)!=0))+1]
    return dict(a=aa[starts],b=bb[starts],target=tt[starts],
                weight=np.add.reduceat(np.asarray(weight,dtype=np.float64)[order],starts))


class Seam:
    """Half-surface from an interpolated umbilicus towards image-up (-Y).

    Axis is piecewise linear in Z. Each path segment is split at axis knots,
    making crossing computation exact for these polylines. Positive transport
    is increasing atan2(Y-cY,X-cX). The seam belongs to its positive-angle side.
    """
    def __init__(self, axis, angle=-np.pi/2):
        self.axis = np.asarray(axis, float)
        self.axis = self.axis[np.argsort(self.axis[:, 2])]
        if len(self.axis) < 2 or not np.isfinite(self.axis).all() or np.any(np.diff(self.axis[:, 2]) <= 0):
            raise ValueError('Axis needs distinct increasing Z coordinates')
        self.ray = np.array([np.cos(angle), np.sin(angle)])
        self.ray[abs(self.ray) < 1e-15] = 0
        self.angle = angle

    def relative(self, points):
        p = np.asarray(points, float)
        center = np.column_stack([np.interp(p[:, 2], self.axis[:, 2], self.axis[:, dim]) for dim in (0, 1)])
        return p[:, :2]-center

    def segment(self, start, end):
        a, b = np.asarray(start, float), np.asarray(end, float)
        if not np.isfinite([a, b]).all():
            raise ValueError('Nonfinite path')
        ts = np.array([0., 1.])
        if b[2] != a[2]:
            cuts = (self.axis[:, 2]-a[2])/(b[2]-a[2])
            ts = np.sort(np.r_[ts, cuts[(cuts > 0)&(cuts < 1)]])
        r = self.relative(a[None, :]+ts[:, None]*(b-a))
        # Crossing through the axis is undefined, including a path lying along
        # the cut and passing through its origin.
        dr = np.diff(r, axis=0)
        t = np.clip(-np.sum(r[:-1]*dr, axis=1)/np.maximum(np.sum(dr*dr, axis=1), 1e-300), 0, 1)
        if np.any(np.linalg.norm(r[:-1]+t[:, None]*dr, axis=1) < 1e-8):
            raise ValueError('Path intersects umbilicus; seam transport undefined')
        u = r @ self.ray
        v = self.ray[0]*r[:, 1]-self.ray[1]*r[:, 0]
        up = (v[:-1] < 0)&(v[1:] >= 0)
        down = (v[:-1] >= 0)&(v[1:] < 0)
        chosen = up | down
        if not chosen.any():
            return 0
        t = -v[:-1][chosen]/np.diff(v)[chosen]
        outward = u[:-1][chosen]+t*np.diff(u)[chosen]
        return int(np.sum(np.where(up[chosen], 1, -1)*(outward > 0)))

    def polyline(self, points):
        return np.r_[0, np.cumsum(self.segments(points[:-1], points[1:]))]

    def segments(self, start, end):
        """Batch segments within linear axis intervals; split knot crossings exactly."""
        start, end = np.asarray(start, float), np.asarray(end, float)
        if start.shape != end.shape or start.ndim != 2 or start.shape[1] != 3:
            raise ValueError('Expected matching XYZ segment arrays')
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            raise ValueError('Nonfinite path')
        result = np.zeros(len(start), dtype=np.int64)
        intervals_a = np.searchsorted(self.axis[:, 2], start[:, 2], side='right')
        intervals_b = np.searchsorted(self.axis[:, 2], end[:, 2], side='right')
        simple = intervals_a == intervals_b
        a, b = self.relative(start[simple]), self.relative(end[simple])
        dr = b-a
        t = np.clip(-np.sum(a*dr, axis=1)/np.maximum(np.sum(dr*dr, axis=1), 1e-300), 0, 1)
        if np.any(np.linalg.norm(a+t[:, None]*dr, axis=1) < 1e-8):
            raise ValueError('Path intersects umbilicus; seam transport undefined')
        ua, ub = a @ self.ray, b @ self.ray
        va = self.ray[0]*a[:, 1]-self.ray[1]*a[:, 0]
        vb = self.ray[0]*b[:, 1]-self.ray[1]*b[:, 0]
        up, down = (va < 0)&(vb >= 0), (va >= 0)&(vb < 0)
        chosen = up | down
        crossing = np.zeros(len(a), np.int64)
        t = -va[chosen]/(vb[chosen]-va[chosen])
        crossing[chosen] = np.where(up[chosen], 1, -1)*(ua[chosen]+t*(ub[chosen]-ua[chosen]) > 0)
        result[simple] = crossing
        for index in np.flatnonzero(~simple):
            result[index] = self.segment(start[index], end[index])
        return result


def corrected_targets(measured, fiber_a, fiber_b, connector):
    """Kb(end)-Ka(end)-C=m, with Ki(end)=ki+Ti(end)."""
    return np.asarray(measured)+np.asarray(connector)-np.asarray(fiber_b)+np.asarray(fiber_a)
