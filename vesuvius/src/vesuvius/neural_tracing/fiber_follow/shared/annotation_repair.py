"""Annotation kink repair and fold-back detection, applied once when fibers are loaded.

Annotations occasionally contain short kinks: a tick, V, hook or small step where the polyline
turns sharply within a voxel and then resumes its previous direction. Fibers do not do this;
their own bends turn at most ~15 degrees per voxel even in tight hairpins (validated on CT in
output/trace_noise_realism_20261002/kink_survey). Each kink is replaced by the shortest cubic
Hermite bridge that is as smooth as its surroundings; everything else is left exactly as
annotated. Sharp corners where the direction really changes (hairpins, corners) are kept.

Fold-backs, where the annotation runs out and returns along itself, cannot be repaired this
way; they are reported so the dataset split can keep those fibers out of training.
"""
from __future__ import annotations

import numpy as np

from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at

ANNOTATION_REPAIR = 'kinks_v1'
SHARP_TURN = 30.        # degrees within one voxel that mark a kink
CLUSTER = 8             # fast-turning points closer than this (voxels) are one kink
NET_TURN = 40.          # direction change across a kink above which it is a real corner
NET_REACH = (2, 8)      # voxels before/after the kink used for that direction
SMOOTH_TURN = 15.       # bridge turning target per voxel ...
SURROUNDING_TURN = 1.25  # ... or this multiple of the turning just outside a tight bend
BRIDGE_HALF = (2, 16)   # bridge half-width range beyond the kink (voxels)
REPAIR_PASSES = 4
FOLDBACK_TURN = 150.
FOLDBACK_OVERLAP = 1.5  # out and return legs within this distance (voxels)
FOLDBACK_MIN_HALF = 4   # shorter reversals are kinks


def unit_curve(points, s):
    arcs = np.arange(0., s[-1]+1e-9, 1.)
    return arcs, interp_at(points, s, arcs)


def turning(curve):
    """Angle between successive 1-voxel chords at each point (degrees, 0 at the ends)."""
    out = np.zeros(len(curve))
    if len(curve) >= 3:
        a, b = curve[1:-1]-curve[:-2], curve[2:]-curve[1:-1]
        cos = (a*b).sum(-1)/np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-9)
        out[1:-1] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return out


def sampled_turning(points, s):
    """Per unit-arc point, the larger 1-voxel-chord turning of it and the half-voxel point after it.

    Sampling at both phases makes detection independent of where the unit grid starts, so a
    repaired fiber (slightly shorter, so resampled at a shifted phase) is not repaired again.
    """
    arcs, curve = unit_curve(points, s)
    half = interp_at(points, s, np.clip(arcs+.5, 0., s[-1]))
    shifted = turning(np.r_[curve[:1], half])[1:]  # chords around each half-voxel point
    shifted[-1] = 0.
    return np.maximum(turning(curve), shifted)


def clusters(indices, gap=CLUSTER):
    groups = []
    for i in indices:
        if groups and i-groups[-1][1] <= gap:
            groups[-1][1] = int(i)
        else:
            groups.append([int(i), int(i)])
    return groups


def net_turn(curve, c0, c1):
    near, far = NET_REACH
    if c0-far < 0 or c1+far >= len(curve):
        return 180.
    before, after = curve[c0-near]-curve[c0-far], curve[c1+far]-curve[c1+near]
    cos = before @ after/max(np.linalg.norm(before)*np.linalg.norm(after), 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def direction(curve, i, side, span=5):
    """Unit direction of the curve just outside index ``i`` on ``side`` (-1 before, +1 after)."""
    j = int(np.clip(i+side*span, 0, len(curve)-1))
    d = curve[j]-curve[i] if side > 0 else curve[i]-curve[j]
    return d/max(np.linalg.norm(d), 1e-9)


def bridge(curve, a, b):
    """Cubic Hermite through curve[a] and curve[b] matching the directions just outside them."""
    chord = np.linalg.norm(curve[b]-curve[a])
    return curve[a], curve[b], chord*direction(curve, a, -1), chord*direction(curve, b, +1)


def hermite(ends, u):
    p0, p1, m0, m1 = ends
    u = np.asarray(u, dtype=np.float64)[:, None]
    return (2*u**3-3*u**2+1)*p0+(u**3-2*u**2+u)*m0+(-2*u**3+3*u**2)*p1+(u**3-u**2)*m1


def repair_kinks(points, s):
    """Repaired vertices and the repaired arclength runs [(start, end)] of the final geometry.

    Repeats until a pass changes nothing (at most REPAIR_PASSES), so loading a repaired fiber
    again is a no-op: a repair shortens the fiber slightly, which can move a borderline spot
    across the threshold.
    """
    out, runs = np.array(points, dtype=np.float64, copy=True), []
    for _ in range(REPAIR_PASSES):
        out, found = repair_pass(out, s)
        if not found:
            break
        s = arclength(out)
        runs.extend(found)
    return out, runs


def repair_pass(points, s):
    """One repair pass. Vertices inside a run are placed on its bridge at their arclength fraction;
    all others, including the endpoints, are unchanged, and the vertex count is preserved.
    """
    arcs, curve = unit_curve(points, s)
    turn = sampled_turning(points, s)
    fixed, bridges, n = curve.copy(), [], len(curve)
    # A kink is seeded by a sharp point and extends over neighbouring points that turn faster
    # than SMOOTH_TURN, so shoulders that fall between unit samples are included.
    kinks = [c for c in clusters(np.flatnonzero(turn > SMOOTH_TURN)) if turn[c[0]:c[1]+1].max() > SHARP_TURN]
    for c0, c1 in kinks:
        if net_turn(curve, c0, c1) > NET_TURN:
            continue
        for half in range(BRIDGE_HALF[0], BRIDGE_HALF[1]+1):
            a, b = c0-half, c1+half
            if a < 1 or b > n-2:
                break
            ends = bridge(fixed, a, b)
            trial = fixed.copy()
            trial[a:b+1] = hermite(ends, np.linspace(0., 1., b-a+1))
            around = np.r_[turn[max(1, a-6):a], turn[b+1:min(n-1, b+7)]]
            limit = max(SMOOTH_TURN, SURROUNDING_TURN*around.max()) if len(around) else SMOOTH_TURN
            if turning(trial[a-1:b+2]).max() < limit:
                fixed = trial
                bridges.append((float(arcs[a]), float(arcs[b]), ends))
                break
    out = np.array(points, dtype=np.float64, copy=True)
    for start, end, ends in bridges:  # in order, so a later overlapping bridge wins as in `fixed`
        inside = (s >= start) & (s <= end)
        out[inside] = hermite(ends, (s[inside]-start)/(end-start))
    return out, [(start, end) for start, end, _ in bridges]


def foldbacks(points, s):
    """Arclengths of reversals whose out and return legs overlap for at least FOLDBACK_MIN_HALF voxels."""
    arcs, curve = unit_curve(points, s)
    found = []
    for r in np.flatnonzero(turning(curve) > FOLDBACK_TURN):
        half = 0
        while (r-half-1 >= 0 and r+half+1 < len(curve)
               and np.linalg.norm(curve[r-half-1]-curve[r+half+1]) < FOLDBACK_OVERLAP):
            half += 1
        if half >= FOLDBACK_MIN_HALF:
            found.append(float(arcs[r]))
    return tuple(found)
