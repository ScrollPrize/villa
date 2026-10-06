"""Evaluation and collection seeds: annotated positions with CT/HV axes, and frozen seed manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

from vesuvius.neural_tracing.tracers.fibers.data.data import TracedFiber
from vesuvius.neural_tracing.tracers.fibers.frames.ct import oriented_seed_heading, SeedHeadingError, SEED_HEADING_POLICY
from vesuvius.neural_tracing.tracers.shared.geometry import interp_at, tangent_at
from vesuvius.neural_tracing.tracers.shared.io import jsonable


def make_seeds(fibers: list[TracedFiber], vol, per_fiber: int = 3,
               margin: float = 32.0, seed: int = 0):
    """Annotated positions, CT/HV axes; one entry per (seed, direction).

    Annotation chooses only the scoring sign of the estimated axis. Missing
    CT orientation is reported and skipped, never replaced by a GT tangent.
    """
    rng = np.random.default_rng(seed)
    out = []
    for fi, f in enumerate(fibers):
        if f.length < 2 * margin:
            continue
        ts = np.arange(margin, f.length - margin, 4.0)
        if len(ts) == 0:
            continue
        for t in rng.choice(ts, size=min(per_fiber, len(ts)), replace=False):
            p = interp_at(f.points, f.s, np.array([t]))[0]
            tau = tangent_at(f.points, f.s, t)
            try:
                ax = oriented_seed_heading(vol, p, f.tag, tau)
            except SeedHeadingError as error:
                warnings.warn(f'Skipping seed for {f.name} at {p.tolist()}: {error}', stacklevel=2)
                continue
            for sgn in (1.0, -1.0):
                h = sgn*ax
                out.append(dict(fiber=fi, t=float(t), sign=sgn, pos=p, heading=h,
                                family=f.tag, seed_heading_policy=SEED_HEADING_POLICY))
    return out


def directed_seed(fiber: TracedFiber, vol, rng, sign: float, margin: float = 32.0):
    """One annotated position and its CT/HV axis in one traversal direction.

    Returns None for a fiber too short for the margin. Missing CT orientation raises
    ``SeedHeadingError``; the caller skips that fiber with an explicit reason.
    """
    if fiber.length < 2*margin:
        return None
    ts = np.arange(margin, fiber.length-margin, 4.0)
    t = float(ts[int(rng.integers(len(ts)))])
    p = interp_at(fiber.points, fiber.s, np.array([t]))[0]
    axis = oriented_seed_heading(vol, p, fiber.tag, tangent_at(fiber.points, fiber.s, t))
    return dict(t=t, sign=float(sign), pos=p, heading=float(sign)*axis, family=fiber.tag,
                seed_heading_policy=SEED_HEADING_POLICY)


def read_manifest(path):
    """Check the frozen payload hash, split membership and seed identities."""
    saved=json.loads(Path(path).read_text())
    payload={k:v for k,v in saved.items() if k!='sha256'}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,default=jsonable).encode()).hexdigest()
    if digest!=saved['sha256']:
        raise ValueError('Frozen seed manifest was modified')
    sets=[set(saved[name+'_fibers']) for name in ('monitor','calibration','final')]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Evaluation splits overlap')
    for name,ids in zip(('monitor','calibration','final'),sets):
        if any(s['fiber'] not in ids for s in saved[name]):
            raise ValueError('Seed appears outside its frozen split')
    return saved
