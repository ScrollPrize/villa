"""Seeded tracing evaluation against held-out GT fibers."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from dataclasses import asdict

import numpy as np
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.fiber_follow.data import DATA_VERSION, TracedFiber, fiber_manifest
from vesuvius.neural_tracing.fiber_follow.geometry import interp_at, tangent_at
from vesuvius.neural_tracing.fiber_follow.trace import field_axis, point_samples


def make_seeds(fibers: list[TracedFiber], vol, per_fiber: int = 3, min_presence: float = 0.8,
               margin: float = 32.0, seed: int = 0):
    """Seed states at high-presence GT points; one entry per (seed, direction).

    Initial heading is the predicted field axis (fiber mode), or local presence
    PCA (CT-only), signed to agree with the GT direction being evaluated."""
    rng = np.random.default_rng(seed)
    out = []
    for fi, f in enumerate(fibers):
        if f.length < 2 * margin:
            continue
        ts = np.arange(margin, f.length - margin, 4.0)
        pres = point_samples(vol, interp_at(f.points, f.s, ts))
        good = ts[pres >= min_presence]
        if len(good) == 0:
            continue
        for t in rng.choice(good, size=min(per_fiber, len(good)), replace=False):
            p = interp_at(f.points, f.s, np.array([t]))[0]
            tau = tangent_at(f.points, f.s, t)
            ax, _ = field_axis(vol, p)
            for sgn in (1.0, -1.0):
                h = ax if np.dot(ax, sgn * tau) >= 0 else -ax
                out.append(dict(fiber=fi, t=float(t), sign=sgn, pos=p, heading=h))
    return out


def trace_events(path, fiber, t0, sign, tol=3.0, patience=3, tree=None):
    """First sustained departure and first supported endpoint-plane crossing.

    Endpoint crossing is checked on segments, not just nearest-point indices,
    so the crossing segment can be partitioned exactly. Returns cumulative
    path lengths, GT distances/arcs, departure index, and endpoint path length.
    """
    path = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    tree = tree if tree is not None else cKDTree(fiber.points)
    d, j = tree.query(path)
    arc = fiber.s[j]
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))] if len(path) else np.zeros(0)
    bad = d > tol
    run = np.convolve(bad.astype(int), np.ones(patience, int), mode="full")[:len(bad)] if len(bad) else np.zeros(0)
    fail = np.flatnonzero(run >= patience)
    end = max(int(fail[0]) - patience + 1, 0) if len(fail) else len(path)
    endpoint = fiber.points[-1 if sign > 0 else 0]
    tangent = tangent_at(fiber.points, fiber.s, fiber.length if sign > 0 else 0.0) * sign
    forward = (path - endpoint) @ tangent
    crossing = None
    # A seed at the endpoint has no annotated continuation.
    if len(path) and abs((fiber.length - t0) if sign > 0 else t0) < 1e-9 and d[0] <= tol:
        crossing = 0.0
    for i in np.flatnonzero((forward[:-1] < 0) & (forward[1:] >= 0)):
        if i + 1 > end:
            break  # identity was lost before reaching the endpoint
        # A nearby winding can cross the endpoint plane long before the GT
        # traversal reaches its end. Require plausible along-fiber progress.
        remaining = fiber.length - arc[i] if sign > 0 else arc[i]
        if remaining > lengths[i + 1] - lengths[i] + 2 * tol:
            continue
        alpha = -forward[i] / (forward[i + 1] - forward[i])
        hit = path[i] + alpha * (path[i + 1] - path[i])
        if np.linalg.norm(hit - endpoint) <= tol:
            crossing = float(lengths[i] + alpha * (lengths[i + 1] - lengths[i]))
            break
    return lengths, d, arc, end, crossing


def score_trace(path: np.ndarray, fiber: TracedFiber, t0: float, sign: float, tol: float = 3.0,
                patience: int = 3, tree: cKDTree | None = None):
    """Partition every segment into supported, wrong, or unknown length.

    An untagged annotation endpoint censors subsequent continuation. A tagged
    physical endpoint makes subsequent length an endpoint overrun. A departure
    before that boundary remains wrong even if the path later returns to GT.
    """
    lengths, d, arc, end, crossing = trace_events(path, fiber, t0, sign, tol, patience, tree)
    total = float(lengths[-1]) if len(lengths) else 0.0
    avail = float((fiber.length - t0) if sign > 0 else t0)
    prog = (arc[:end] - t0) * sign
    followed = min(float(max(prog.max(), 0.0)) if len(prog) else 0.0, avail)
    endpoint_known = fiber.endpoint_stop[0 if sign < 0 else 1]
    unknown = overrun = 0.0
    diverged = end < len(d) and crossing is None
    if crossing is not None:
        correct = crossing
        followed = avail
        if endpoint_known:
            overrun = total - correct
            offtrack = overrun
        else:
            unknown = total - correct
            offtrack = 0.0
    elif diverged:
        # Include the transition from the last supported point to the first bad point.
        correct = float(lengths[max(0, end - 1)])
        offtrack = total - correct
    else:
        correct, offtrack = total, 0.0
    err_mask = np.arange(len(d)) < end
    if crossing is not None:
        err_mask &= lengths <= crossing
    err = float(d[err_mask].mean()) if err_mask.any() else float("nan")
    avail_nb = avail
    if fiber.brk is not None and fiber.brk.any():
        ahead = (fiber.s[fiber.brk] - t0) * sign
        ahead = ahead[ahead >= 0]
        if len(ahead):
            avail_nb = float(ahead.min())
    return dict(followed=followed, avail=avail, coverage=followed / max(avail, 1e-6),
                avail_nb=avail_nb, coverage_nb=min(followed, avail_nb) / max(avail_nb, 1e-6),
                correct=correct, offtrack=offtrack, unknown=unknown, endpoint_overrun=overrun,
                scored_length=correct + offtrack, length=total,
                precision=correct / max(correct + offtrack, 1e-6),
                verified_fraction=correct / max(total, 1e-6),
                unknown_fraction=unknown / max(total, 1e-6),
                reached_end=bool(followed >= avail - tol), crossed_endpoint=crossing is not None,
                endpoint_known=bool(endpoint_known), diverged=bool(diverged), err=err)


def evaluate(tracer, fibers, seeds, batch: int = 256):
    trees = {}
    rows = []
    for b in range(0, len(seeds), batch):
        chunk = seeds[b:b + batch]
        paths, reasons = tracer.trace(np.stack([s["pos"] for s in chunk]), np.stack([s["heading"] for s in chunk]))
        for s, p, r in zip(chunk, paths, reasons):
            f = fibers[s["fiber"]]
            if s["fiber"] not in trees:
                trees[s["fiber"]] = cKDTree(f.points)
            m = score_trace(p, f, s["t"], s["sign"], tree=trees[s["fiber"]])
            span = next((span for span in f.spans if span.start <= s["t"] <= span.end), None)
            m.update(reason=r, fiber=s["fiber"], fiber_name=f.name,
                     seed_span_mode=span.provenance.interp_mode if span else None,
                     t0=s["t"], sign=s["sign"])
            rows.append(m)
    return rows, summarize(rows)


def summarize(rows):
    if not rows:
        raise ValueError("No traces to evaluate; check the controlled spans and seed selection")
    cov = np.array([r["coverage"] for r in rows])
    fol = np.array([r["followed"] for r in rows])
    avail = np.array([r["avail"] for r in rows])
    correct = sum(r["correct"] for r in rows)
    wrong = sum(r["offtrack"] for r in rows)
    unknown = sum(r["unknown"] for r in rows)
    total = sum(r["length"] for r in rows)
    errors = np.array([r["err"] for r in rows])
    return dict(
        evaluation_version=DATA_VERSION, n=len(rows),
        coverage_mean=float(cov.mean()), coverage_median=float(np.median(cov)),
        length_weighted_coverage=float(fol.sum() / max(avail.sum(), 1e-6)),
        reached_end=float(np.mean([r["reached_end"] for r in rows])),
        diverged=float(np.mean([r["diverged"] for r in rows])),
        followed_median=float(np.median(fol)),
        offtrack_mean=wrong / len(rows), wrong_len_mean=wrong / len(rows),
        err_mean=float(errors[np.isfinite(errors)].mean()) if np.isfinite(errors).any() else float("nan"),
        coverage_nb_mean=float(np.mean([r["coverage_nb"] for r in rows])),
        precision_mean=float(np.mean([r["precision"] for r in rows])),
        length_precision=correct / max(correct + wrong, 1e-6),
        correct_length=correct, wrong_length=wrong, unknown_length=unknown,
        total_length=total, scored_length=correct + wrong,
        verified_length_fraction=correct / max(total, 1e-6),
        unknown_length_fraction=unknown / max(total, 1e-6),
        endpoint_overrun_length=sum(r["endpoint_overrun"] for r in rows),
        known_endpoint_traces=sum(r["endpoint_known"] for r in rows),
    )


def load_or_make_seeds(path, fibers, vol, band, *, rebuild=False, per_fiber=2, seed=0):
    """Fixed seeds bound to the current geometry, volume, and selection policy."""
    metadata = dict(version=DATA_VERSION, fibers=fiber_manifest(fibers),
                    volume=vol.spec.to_dict(), band=asdict(band), per_fiber=per_fiber,
                    seed=seed, min_presence=0.8, margin=32.0)
    if os.path.exists(path) and not rebuild:
        with open(path, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("metadata") != metadata:
            raise ValueError("Incompatible seed cache (labels, volume, or policy changed); "
                             "use a new --seeds path or --rebuild-seeds")
        return cached["seeds"]
    seeds = make_seeds(fibers, vol, per_fiber=per_fiber, seed=seed)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=parent, delete=False) as fh:
        tmp = fh.name
        pickle.dump(dict(metadata=metadata, seeds=seeds), fh)
    os.replace(tmp, path)
    return seeds
