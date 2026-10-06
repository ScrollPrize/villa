"""Training samples for the autoregressive fiber follower."""

from __future__ import annotations

import glob
import hashlib
import math
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial import cKDTree

import json

from vc3d_fiber_format import parse_vc3d_fiber_format, FiberTraceSegmentMetadata
from vesuvius.neural_tracing.fiber_follow.shared.geometry import (
    CropSpec,
    arclength,
    block_start,
    crop_local_grid,
    frame_from_heading,
    interp_at,
    tangent_at,
    traversal_curve,
)
from vesuvius.neural_tracing.fiber_follow.data.annotation_repair import foldbacks, repair_kinks
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS


DATA_VERSION = 2
MAX_CT_FRAME_REJECTIONS = 64


@dataclass(frozen=True)
class FiberSpan:
    start: float
    end: float
    provenance: FiberTraceSegmentMetadata


@dataclass
class TracedFiber:
    name: str
    points: np.ndarray  # (N, 3) trace-grid xyz; all annotated vertices, gaps <= 1 voxel
    s: np.ndarray
    tag: str
    spans: tuple[FiberSpan, ...] = ()
    endpoint_stop: tuple[bool, bool] = (False, False)
    source_hash: str = ""
    excluded_tail_length: float = 0.0
    kink_repairs: int = 0  # annotation kinks replaced by smooth bridges at load (annotation_repair)
    foldbacks: tuple[float, ...] = ()  # arcs where the annotation doubles back; kept out of training

    @property
    def length(self) -> float:
        return float(self.s[-1])


def load_fibers(fiber_dir: str, grid_scale: float = 8.0, spacing: float = 1.0) -> list[TracedFiber]:
    """Load only geometry between human control points, retaining span provenance.

    Every line point between the outer controls is equally valid supervision,
    regardless of tags or interpolation provenance. An outer control without an
    explicit termination tag is a censored boundary, not a physical endpoint.
    Short annotation kinks are repaired and fold-backs recorded (annotation_repair).
    """
    if grid_scale <= 0 or spacing <= 0:
        raise ValueError("grid_scale and spacing must be positive")
    out = []
    for path in sorted(glob.glob(os.path.join(fiber_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        fib = parse_vc3d_fiber_format(raw, path=path)
        if len(fib.control_points_xyz) < 2:
            continue  # a seed and its extrapolations provide no controlled span
        line = np.asarray(fib.line_points_xyz, dtype=np.float64)
        controls = np.asarray(fib.control_points_xyz, dtype=np.float64)
        if len(line) < 2:
            raise ValueError(f"{path}: controlled fiber has no dense line")
        dist, anchors = cKDTree(line).query(controls)
        if np.any(dist > 1e-4):
            raise ValueError(f"{path}: control points must be anchored in line_points")
        if np.all(np.diff(anchors) < 0):
            line = line[::-1]
            anchors = len(line) - 1 - anchors
        if np.any(np.diff(anchors) <= 0):
            raise ValueError(f"{path}: control points must have distinct ordered line anchors")
        line = line / grid_scale
        pieces, bounds = [], [0]
        for i, (a, b) in enumerate(zip(anchors[:-1], anchors[1:])):
            # Preserve every line vertex, not just human control anchors.
            original = line[a:b + 1]
            pieces_dense = []
            for left, right in zip(original[:-1], original[1:]):
                distance = np.linalg.norm(right-left)
                if distance > 1e-9:
                    count = max(1, int(np.ceil(distance/spacing)))
                    pieces_dense.append(left+(right-left)*np.arange(count)[:, None]/count)
            if not pieces_dense:
                raise ValueError(f"{path}: zero-length controlled span")
            piece = np.concatenate([*pieces_dense, original[-1:]])
            pieces.append(piece if i == 0 else piece[1:])
            bounds.append(bounds[-1]+len(piece)-1)
        pts = np.concatenate(pieces)
        if len(pts) < 8:
            continue
        folds = foldbacks(pts, arclength(pts))
        pts, repairs = repair_kinks(pts, arclength(pts))
        s = arclength(pts)
        spans = [FiberSpan(float(s[a]), float(s[b]), fib.control_point_segments[i])
                 for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
        tag = (fib.metadata.get("hv_classification") or {}).get("automatic_tag", "")
        endpoints = (False, False)
        if fib.version == 3:
            endpoints = tuple("kollesis_termination" in raw["control_points"][i].get("tags", [])
                              for i in (0, -1))
        full_s = arclength(line)
        tails = float(full_s[anchors[0]] + full_s[-1] - full_s[anchors[-1]])
        source_hash = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
        out.append(TracedFiber(os.path.basename(path), pts, s, tag,
                              spans=tuple(spans),
                              endpoint_stop=endpoints, source_hash=source_hash,
                              excluded_tail_length=tails, kink_repairs=len(repairs), foldbacks=folds))
    return out


def fiber_manifest(fibers):
    """Ordered identities for caches containing fiber indices and arc positions."""
    if hasattr(fibers, 'manifest_entries'):
        return fibers.manifest_entries()
    return [dict(name=f.name, source_hash=f.source_hash,
                 geometry_hash=hashlib.sha256(np.asarray(f.points, dtype="<f8").tobytes()).hexdigest(),
                 endpoint_stop=list(f.endpoint_stop)) for f in fibers]


@dataclass(frozen=True)
class ZBand:
    """Held-out band in trace-grid z (inclusive-exclusive)."""

    lo: float
    hi: float

    def contains(self, z):
        return (z >= self.lo) & (z < self.hi)


def split_fibers(fibers: list[TracedFiber], val_band: ZBand):
    """Split by the controlled geometry centroid; filter training states separately.

    Fibers whose annotation doubles back are kept out of training (validation is unchanged).
    """
    train, val = [], []
    for f in fibers:
        z = np.average((f.points[:-1, 2]+f.points[1:, 2])/2, weights=np.diff(f.s))
        if val_band.contains(z):
            val.append(f)
        elif not f.foldbacks:
            train.append(f)
    return train, val


@dataclass
class SampleConfig:
    crop: CropSpec = field(default_factory=lambda: CropSpec(depth=176, width=96, behind=128))
    n_future: int = 16
    future_step: float = 1.0
    n_history: int = 128
    recent_history_points: int = 128  # GT history for observed-state diagnostics only
    history_step: float = 1.0
    # Fresh trace starts, as direct draw allocations: seed-only, 1-8 and 9-32 requested
    # voxels of history, and a prefix drawn uniformly from the available history.
    startup_shares: tuple = (.15, .17, .17, .51)
    # Simulated tracing error (``trace_noise``), fit to on-track 81k rollouts against the repaired
    # annotations (output/trace_noise_realism_20261002/refit_repaired): per-trace scale log-uniform
    # around a median of .412 (spread 3.43), commit-end correlation length, mid-commit bulge and
    # per-trace bias relative to that scale, and the annotation's small-scale wiggle the tracer
    # does not follow (Gaussian sigma, voxels). A zero scale range disables simulated error.
    trace_noise_sigma: tuple = (.2225, .7630)
    trace_noise_length: float = 33.
    trace_noise_bulge: float = .90
    trace_noise_bias: float = .119
    trace_noise_smoothing: float = 1.51
    # Smooth lateral excursions on established fresh traces; the head lies on the rise or return.
    # The rise length is log-uniform (matches real 81k excursions measured the same way).
    excursion_probability: float = .2
    excursion_amplitude: tuple = (3., 6.)
    excursion_rise: tuple = (16., 128.)
    # A seed clicked a little off the centerline (``seed_offsets``): lateral offset magnitude range at the seed,
    # voxels, fading out over ``seed_offset_ramp`` voxels of trace. (0, 0) keeps seeds on the annotation.
    seed_offset: tuple = (0., 0.)
    seed_offset_ramp: float = 16.
    # Share of live-continuation chains that start at a fresh seed (the rest from replayed DAgger prefixes).
    live_seed_start: float = .5
    label_tolerance: float = 1.5  # confidence-label tolerance; separate from the departure threshold
    max_recovery_distance: float = 6.0
    dense_substeps: int = 4

    def __post_init__(self):
        shares = np.asarray(self.startup_shares, np.float64)
        if shares.shape != (len(STARTUP_CATEGORIES),) or not np.isfinite(shares).all() or (shares < 0).any() \
                or abs(shares.sum()-1) > 1e-6:
            raise ValueError('Startup shares need four nonnegative values summing to one')
        self.startup_shares = tuple(float(v) for v in shares)
        for name in ('excursion_amplitude', 'excursion_rise', 'trace_noise_sigma', 'seed_offset'):
            lo, hi = getattr(self, name)
            if not (np.isfinite(lo) and np.isfinite(hi) and 0 <= lo <= hi):
                raise ValueError(f'{name} must be an ordered nonnegative range')
            setattr(self, name, (float(lo), float(hi)))
        if not (self.seed_offset_ramp > 0 and 0 <= self.live_seed_start <= 1):
            raise ValueError('Seed offset ramp must be positive and the live seed-start share in [0, 1]')
        if not 0 <= self.excursion_probability <= 1 or self.excursion_rise[0] <= 0:
            raise ValueError('Excursion probability must lie in [0, 1] with positive rise lengths')
        if not (self.trace_noise_length > 0 and min(self.trace_noise_bulge, self.trace_noise_bias,
                                                     self.trace_noise_smoothing) >= 0):
            raise ValueError('Trace noise length must be positive and its shape terms nonnegative')
        if not (np.isfinite(self.label_tolerance) and self.label_tolerance > 0
                and np.isfinite(self.max_recovery_distance) and self.max_recovery_distance > self.future_step):
            raise ValueError('Label tolerance and recovery limit must be finite and positive')

    @property
    def future_s(self) -> np.ndarray:
        return self.future_step * np.arange(1, self.n_future + 1)


STARTUP_CATEGORIES = ('seed_only', 'short', 'early', 'established')
STARTUP_REQUESTS = ((0, 0), (1, 8), (9, 32), None)
# Realized seed ages, reported separately from draw categories.
SEED_AGE_STRATA = (('seed_only', 0., 0.), ('below_12', 0., 12.), ('12_32', 12., 32.), ('established', 32., np.inf))


def seed_age_stratum(age):
    if age <= 0:
        return 0
    return 1 if age < 12 else 2 if age <= 32 else 3


def training_state_allowed(item, crop: CropSpec, band: ZBand | None):
    """Exclude the perturbed state, entire sampled block, and used history/labels.

    Keep the original 48-voxel position guard and also check the actual rotated
    crop's read footprint (including interpolation support). Applied before I/O
    to fresh and cached states. The circumsphere covers any CT-selected roll.
    """
    if band is None:
        return True
    pos, frame = item["pos"], item["frame"]
    if band.lo - 48 <= pos[2] < band.hi + 48:
        return False
    start = block_start(pos, frame, crop)[0]
    if start < band.hi and start + crop.block_size > band.lo:
        return False
    zs = [np.array([pos[2]])]
    for key, mask in (("hist_local", "hmask"), ("gt_history", "gt_history_mask"),
                      ("fut_local", "fmask")):
        if key in item:
            points = item[key][item[mask] > 0]
            if len(points):
                zs.append((points @ frame.T + pos)[:, 2])
    if "plane_ab" in item:
        # The forward-plane labels can look farther along GT than arclength targets.
        loc = np.c_[item["plane_ab"], item["planes"]]
        loc = loc[item["plane_mask"] > 0]
        if len(loc):
            zs.append((loc @ frame.T + pos)[:, 2])
    if "dense_ab" in item:
        loc = np.c_[item["dense_ab"], item["dense_planes"]][item["dense_mask"] > 0]
        if len(loc):
            zs.append((loc @ frame.T + pos)[:, 2])
    z = np.concatenate(zs)
    return not (z.min() < band.hi and z.max() >= band.lo)


def plane_targets(p, s, t, t_end, pos, frame, planes):
    """Where the GT curve (traversal arc ``s``, from ``t`` up to ``t_end``)
    first crosses each local forward plane c = planes[k]. Returns lateral
    (a, b) per plane (K, 2) and a validity mask (K,)."""
    K = len(planes)
    ab = np.zeros((K, 2), np.float32)
    m = np.zeros(K, np.float32)
    span = min(t_end, t + 2.5 * float(planes[-1])) - t
    if span <= 0:
        return ab, m
    arc = np.r_[t, s[(s > t) & (s < t+span)], t+span]
    loc = (interp_at(p, s, arc) - pos) @ frame
    c = loc[:, 2]
    for k, ck in enumerate(planes):
        hit = np.nonzero((c[:-1] < ck) & (c[1:] >= ck))[0]
        if len(hit):
            i = hit[0]
            w = (ck - c[i]) / max(c[i + 1] - c[i], 1e-9)
            ab[k] = (1 - w) * loc[i, :2] + w * loc[i + 1, :2]
            m[k] = 1.0
    return ab, m


def trace_prefix_length(t, category, rng):
    """Arclength already traced at this decision, i.e. how far back its seed lies.

    Requests are clipped to the available annotation; the realized seed age is
    reported separately from the draw category.
    """
    request = STARTUP_REQUESTS[category]
    if request is None:
        return float(rng.uniform(0., t))
    lo, hi = request
    return min(float(t), float(rng.integers(lo, hi+1))) if hi else 0.


TRACE_COMMIT_LENGTH = 16  # voxels per simulated commit, plus an exponential extra (real commits: p50 16.6)
TRACE_COMMIT_EXTRA = .8


def trace_noise(arcs, p, s, cfg: SampleConfig, rng):
    """Lateral tracing error along a simulated observed path, zero at its seed.

    Mirrors the tracer: the path is a chain of commits starting at the seed. At each commit
    end the lateral offset takes one mean-reverting step (correlation length
    ``cfg.trace_noise_length``) toward a small per-trace bias, so joins carry the kinks seen
    in real traces; within a commit the offset moves smoothly with a quadratic bulge. The
    per-trace scale is log-uniform in ``cfg.trace_noise_sigma``. The tracer also does not
    follow the annotation's small-scale wiggle: the smoothed-minus-raw annotation
    (``cfg.trace_noise_smoothing``) is added, ramped in from the seed. The GT-tangential part
    of the commit error is removed. ``arcs`` are evenly spaced, starting at the seed.
    """
    lo, hi = cfg.trace_noise_sigma
    n = len(arcs)
    noise = np.zeros((n, 3))
    if hi <= 0 or n < 2:
        return noise, 0.
    sigma = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if lo > 0 else float(rng.uniform(lo, hi))
    step = float(arcs[1]-arcs[0])
    bias = rng.normal(size=3)*cfg.trace_noise_bias
    start, current = 0, np.zeros(3)
    while start < n-1:
        size = min(n-1-start, max(1, int(round((TRACE_COMMIT_LENGTH+rng.exponential(TRACE_COMMIT_EXTRA))/step))))
        rho = math.exp(-size*step/cfg.trace_noise_length)
        end = bias+rho*(current-bias)+math.sqrt(1-rho*rho)*rng.normal(size=3)
        u = np.arange(1, size+1)/size
        noise[start+1:start+size+1] = (current+(end-current)*u[:, None]
                                       + 4*cfg.trace_noise_bulge*rng.normal(size=3)*(u*(1-u))[:, None])
        start, current = start+size, end
    noise *= sigma
    tangent = local_tangents(arcs, p, s)
    noise = noise-(noise*tangent).sum(-1, keepdims=True)*tangent
    return noise+annotation_smoothing(arcs, p, s, cfg.trace_noise_smoothing), sigma


def annotation_smoothing(arcs, p, s, sigma):
    """Gaussian-smoothed minus raw annotation at evenly spaced ``arcs``, ramped in over 3 sigma
    from ``arcs[0]`` so the path still starts on its seed."""
    from scipy.ndimage import gaussian_filter1d
    if sigma <= 0 or len(arcs) < 2:
        return np.zeros((len(arcs), 3))
    step = float(arcs[1]-arcs[0])
    pad = int(math.ceil(4*sigma/step))
    grid = arcs[0]+step*np.arange(-pad, len(arcs)+pad, dtype=np.float64)
    raw = interp_at(p, s, np.clip(grid, 0., s[-1]))
    offset = (gaussian_filter1d(raw, sigma/step, axis=0, mode='nearest')-raw)[pad:pad+len(arcs)]
    return offset*smoothstep((arcs-arcs[0])/(3*sigma))[:, None]


def local_tangents(arcs, p, s):
    tangent = interp_at(p, s, np.clip(arcs+3., 0., s[-1]))-interp_at(p, s, np.clip(arcs-3., 0., s[-1]))
    return tangent/np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)


def smoothstep(u):
    u = np.clip(u, 0., 1.)
    return u*u*(3-2*u)


def excursion_offsets(arcs, p, s, cfg: SampleConfig, rng):
    """A smooth lateral departure and return ending at the head, zero before it starts.

    Amplitude is uniform and rise length log-uniform in their configured ranges. The head lies
    uniformly on the rise or the return (within the available prefix), so both
    departing and returning heads occur. The direction is one random lateral vector,
    projected off the local annotation tangent at every point.
    """
    amplitude = float(rng.uniform(*cfg.excursion_amplitude))
    rise = float(np.exp(rng.uniform(*np.log(cfg.excursion_rise))))
    head = float(rng.uniform(0., min(2*rise, arcs[-1]-arcs[0])))
    x = arcs-(arcs[-1]-head)
    profile = amplitude*np.where(x <= rise, smoothstep(x/rise), smoothstep(2-x/rise))*(x >= 0)
    tangent = local_tangents(arcs, p, s)
    direction = rng.normal(size=3)
    direction = direction[None]-(tangent @ direction)[:, None]*tangent
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
    return profile[:, None]*direction, dict(excursion_amplitude=amplitude, excursion_rise=rise,
                                            excursion_phase=head/rise, excursion_head_offset=float(profile[-1]))


def seed_offsets(arcs, p, s, cfg: SampleConfig, rng):
    """Lateral offsets of a trace whose seed was placed slightly off the centerline: a random direction normal to the
    fiber at the seed, magnitude uniform in ``cfg.seed_offset``, fading linearly to zero over ``cfg.seed_offset_ramp``
    voxels of trace (the tracer's first commits return to the fiber). No draws when disabled."""
    lo, hi = cfg.seed_offset
    if hi <= 0:
        return np.zeros((len(arcs), 3))
    tangent = tangent_at(p, s, arcs[0])
    normal = np.cross(tangent, rng.normal(size=3))
    normal /= max(np.linalg.norm(normal), 1e-9)
    weight = np.clip(1-(arcs-arcs[0])/cfg.seed_offset_ramp, 0., 1.)
    return weight[:, None]*normal*float(rng.uniform(lo, hi))


def simulated_trace(p, s, t, cfg: SampleConfig, rng: np.random.Generator, *, startup=None, excursion=None):
    """The observed path of a simulated trace whose head is at traversal arclength ``t`` of curve (p, s).

    Every random draw of ``make_sample`` happens here, in its order; the labels it adds use none.
    Returns (path, arcs, startup category, trace noise sigma, excursion details).
    """
    category = int(rng.choice(len(STARTUP_CATEGORIES), p=cfg.startup_shares)) if startup is None else int(startup)
    count = int(trace_prefix_length(t, category, rng)//cfg.history_step)
    arcs = t-np.arange(count, -1, -1)*cfg.history_step
    noise, sigma = trace_noise(arcs, p, s, cfg, rng)
    path = interp_at(p, s, arcs)+noise
    details = {}
    established = category == STARTUP_CATEGORIES.index('established') and arcs[-1]-arcs[0] >= 8.
    if established and (excursion if excursion is not None else rng.random() < cfg.excursion_probability):
        offsets, details = excursion_offsets(arcs, p, s, cfg, rng)
        path = path+offsets
    return path+seed_offsets(arcs, p, s, cfg, rng), arcs, category, sigma, details


def make_sample(fiber: TracedFiber, t: float, reverse: bool, cfg: SampleConfig, rng: np.random.Generator,
                *, startup=None, excursion=None):
    """One tracer decision on a simulated trace of this fiber, built exactly like inference.

    The trace started ``trace_prefix_length`` back at an annotated seed and followed GT
    with smooth lateral error (``trace_noise``), optionally plus a smooth excursion
    (``excursion_offsets``) on established traces, so the head is offset from GT. Crop
    heading, history and seed reference come from that observed path through the tracer's
    own functions. A path shorter than 12 voxels still holds the seed's CT heading; it and
    the seed tangent are resolved once CT is readable (``resolve_trace_seed``). Labels are
    the GT continuation from the offset head under the shared state contract.
    ``startup`` fixes the draw category; ``excursion`` forces or forbids an excursion.
    """
    p, s = traversal_curve(fiber, reverse)
    path, arcs, category, sigma, details = simulated_trace(p, s, t, cfg, rng, startup=startup, excursion=excursion)
    return decision_on_path(fiber, t, reverse, cfg, path, arcs, category, sigma, details)


def decision_on_path(fiber, t, reverse, cfg, path, arcs, category, sigma, details):
    """The tracer decision whose head ends an observed path (``path``, sampled at traversal arclengths ``arcs``
    ending at ``t``): crop heading, history, seed reference and labels exactly as ``make_sample`` builds them."""
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import linear12_heading, trace_heading
    from vesuvius.neural_tracing.fiber_follow.data.state_labels import constructed_facts, supervise
    from vesuvius.neural_tracing.fiber_follow.tracing.trace import trace_history
    p, s = traversal_curve(fiber, reverse)
    pos = path[-1]
    # Only the CT seed axis's sign comes from the direction of travel.
    seed_direction = tangent_at(p, s, arcs[0])
    heading = trace_heading(path, 0, seed_direction)
    frame = frame_from_heading(heading)  # provisional basis; CT resolves crop roll before sampling
    hist, hmask = trace_history(list(path), cfg.n_history)
    item = dict(_frame_label_context=(fiber, t, reverse, cfg),
                observed_path=path, seed_pos=path[0].copy(), seed_tangent=seed_direction,
                seed_age=float(arclength(path)[-1]), seed_valid=True, seed_heading_family=fiber.tag,
                trace_noise_sigma=sigma, trace_prefix_length=float(arcs[-1]-arcs[0]), startup=category,
                excursion=bool(details), heading_start=0, travelled=float(arclength(path)[-1]),
                pos=pos, frame=frame, hist_local=(hist-pos) @ frame, hmask=hmask.astype(np.float32),
                trace_facts=constructed_facts(fiber, t, reverse, pos, cfg), **details,
                **continuation_targets(fiber, t, reverse, pos, frame, cfg))
    supervise(item)
    if linear12_heading(path, 0) is None:
        item['_pending_seed_heading'] = (fiber, t, reverse, cfg)
    return item


@dataclass(frozen=True)
class EpisodeSpec:
    """Consecutive decisions along one simulated trace, for whole-trace history (``memory='sequence'``)."""
    steps: int = 64  # decisions per episode (fewer on a fiber too short for them, never below min_steps)
    supervised: int = 16  # trailing decisions with a loss; earlier decisions only build history tokens
    commit: int = 12  # voxels between consecutive heads (the tracer's commit)
    min_steps: int = 4

    def __post_init__(self):
        if not (1 <= self.supervised <= self.steps and 2 <= self.min_steps <= self.steps and self.commit >= 1):
            raise ValueError('Episodes need 1 <= supervised <= steps, 2 <= min_steps <= steps and a positive commit')


def episode_decisions(fiber, t0, reverse, steps, commit, cfg: SampleConfig, rng: np.random.Generator):
    """``steps`` decisions on one simulated trace of this fiber, heads at traversal arclengths t0, t0+commit, ...

    The trace is drawn as ``simulated_trace`` draws a decision's observed path (startup prefix before t0, trace noise
    over the whole trace, optionally one excursion ending at the trace's end), then every head ending a prefix of it
    becomes a decision (``decision_on_path``). Returns (decision, committed segment) pairs; a step's segment is the
    observed path from its head to the next head (world xyz), the points that step commits. Requires
    t0 + steps*commit <= the fiber length.
    """
    p, s = traversal_curve(fiber, reverse)
    step = cfg.history_step
    category = int(rng.choice(len(STARTUP_CATEGORIES), p=cfg.startup_shares))
    prefix = int(trace_prefix_length(t0, category, rng)//step)*step
    per_commit = int(round(commit/step))
    count = int(round(prefix/step))+steps*per_commit
    arcs = t0-prefix+step*np.arange(count+1, dtype=np.float64)
    if arcs[-1] > s[-1]+1e-6:
        raise ValueError('Episode runs past the end of its fiber')
    noise, sigma = trace_noise(arcs, p, s, cfg, rng)
    path = interp_at(p, s, arcs)+noise
    details, offsets = {}, np.zeros_like(path)
    if (category == STARTUP_CATEGORIES.index('established') and arcs[-1]-arcs[0] >= 8.
            and rng.random() < cfg.excursion_probability):
        offsets, details = excursion_offsets(arcs, p, s, cfg, rng)
        path = path+offsets
    path = path+seed_offsets(arcs, p, s, cfg, rng)
    out = []
    for j in range(steps):
        i = int(round(prefix/step))+j*per_commit
        departed = details if np.abs(offsets[:i+1]).max() > 0 else {}
        decision = decision_on_path(fiber, float(arcs[i]), reverse, cfg, path[:i+1], arcs[:i+1], category, sigma, departed)
        out.append((decision, path[i:i+per_commit+1].copy()))
    return out


def loader_chunk(cfg, batch):
    """Units per loader item: ``batch`` decisions, or for sequence models one episode (the trainer merges ``batch``
    of them per microbatch, train/sequence.merge_episodes), so loader workers never hold whole episode microbatches
    in shared memory."""
    return 1 if cfg.model_type == 'sequence' else batch


def mark_episode_identity(items):
    """Episode steps in step order: a step whose episode has an earlier step on the original fiber reads that fiber
    through its history tokens, so its identity is observable (observations.identity_evidence)."""
    from vesuvius.neural_tracing.fiber_follow.data.state_labels import DEPARTURE_DISTANCE
    seen = False
    for item in items:
        item['history_identity_evidence'] = seen
        seen = seen or float(item['match_distance']) <= DEPARTURE_DISTANCE
    return items


def episode_tensors(items, commit):
    """Row-aligned episode layout of a batch of episode decisions: episode and step indices, which rows are
    supervised, and each step's committed segment (world xyz, padded to the longest, at least commit+1 points) with
    its mask."""
    length = max([int(round(commit))+1]+[len(item['episode_segment']) for item in items])
    segment = np.zeros((len(items), length, 3), np.float32)
    mask = np.zeros((len(items), length), bool)
    for row, item in enumerate(items):
        points = np.asarray(item['episode_segment'])
        segment[row, :len(points)], mask[row, :len(points)] = points, True
    return dict(episode_index=torch.tensor([it['episode_index'] for it in items], dtype=torch.long),
                episode_step=torch.tensor([it['episode_step'] for it in items], dtype=torch.long),
                episode_supervised=torch.tensor([bool(it['episode_supervised']) for it in items]),
                episode_segment=torch.from_numpy(segment), episode_segment_mask=torch.from_numpy(mask))


def resolve_trace_seed(item, vol):
    """Give a simulated trace the tracer's CT seed heading before its image is built.

    Sets the seed tangent. A path still holding the seed heading (< 12 voxels) also
    takes it as crop heading: local geometry is re-expressed and labels recomputed.
    Missing CT orientation raises ``SeedHeadingError``: training rejects and resamples
    that seed, exactly as collection and evaluation skip it. There is no annotation
    fallback. Synthetic sources mark themselves with ``seed_heading_family``.
    """
    family = item.pop('seed_heading_family', None)
    pending = item.pop('_pending_seed_heading', None)
    if family is None:
        return item
    item['fiber_family'] = family
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import oriented_seed_heading, reframe_item
    from vesuvius.neural_tracing.fiber_follow.data.state_labels import supervise
    heading = oriented_seed_heading(vol, item['seed_pos'], family, item['seed_tangent'])
    item['seed_tangent'] = heading
    if pending is not None:
        fiber, t, reverse, cfg = pending
        frame = frame_from_heading(heading)
        reframe_item(item, frame)
        item.update(continuation_targets(fiber, t, reverse, item['pos'], frame, cfg))
        supervise(item)
    return item


def continuation_targets(fiber, t, reverse, pos, frame, cfg):
    """Dense supervised geometry; prediction-support gaps never remove GT labels.

    t is traversal arclength. Missing forward crossings are unknown, except
    continuation beyond an explicitly tagged physical endpoint. Whether the
    geometry is supervised at all is the state contract's decision.
    """
    p, s = traversal_curve(fiber, reverse)
    tf = t + cfg.future_s
    fut = interp_at(p, s, np.clip(tf, 0, s[-1]))
    history_arc = t - np.arange(cfg.recent_history_points + 1) * cfg.history_step
    gt_history = (interp_at(p, s, np.clip(history_arc, 0, s[-1])) - pos) @ frame
    gt_history_mask = (history_arc >= 0).astype(np.float32)
    dense_planes = np.linspace(cfg.future_step, cfg.future_s[-1],
                               (cfg.n_future-1)*cfg.dense_substeps+1)
    ab, mask = plane_targets(p, s, t, s[-1], pos, frame, cfg.future_s)
    dense_ab, dense_mask = plane_targets(p, s, t, s[-1], pos, frame, dense_planes)
    fmask = (tf <= s[-1]).astype(np.float32)
    end = (p[-1]-pos) @ frame
    # Only expose endpoint labels when it is within the local traversal window.
    known = fiber.endpoint_stop[0 if reverse else 1] and s[-1]-t <= 2.5*cfg.future_s[-1]
    out = dict(gt_history=gt_history.astype(np.float32), gt_history_mask=gt_history_mask,
               fut_local=(fut-pos) @ frame, fmask=fmask,
               plane_ab=ab, plane_mask=mask, planes=cfg.future_s,
               dense_ab=dense_ab, dense_mask=dense_mask, dense_planes=dense_planes,
               end_local=end, endpoint_known=float(known))
    return out


def refresh_frame_targets(item):
    """Rebuild plane crossings and supervision after a learned heading change; labels only."""
    context = item.get('_frame_label_context')
    if context is None:
        return
    fiber, t, reverse, cfg = context
    item.update(continuation_targets(fiber, t, reverse, item['pos'], item['frame'], cfg))
    from vesuvius.neural_tracing.fiber_follow.data.state_labels import supervise
    if not (item['trace_facts']['match_valid'] and not item['trace_facts']['match_ambiguous']):
        item['gt_history_mask'] = np.zeros_like(item['gt_history_mask'])
    supervise(item)


# One task budget, applied within each dataset source. Sampling allocations, not label
# ratios: every delivered action is labeled by the shared state contract.
TASKS = ('fresh', 'live', 'dagger_pre_excursion', 'dagger_recoverable', 'dagger_terminal',
         'dagger_premature_stop', 'dagger_ordinary', 'synthetic_terminal')
TASK = {name: index for index, name in enumerate(TASKS)}
DEFAULT_TASK_SHARES = dict(fresh=.40, live=.25, dagger_pre_excursion=.08, dagger_recoverable=.06,
                           dagger_terminal=.08, dagger_premature_stop=.03, dagger_ordinary=.05,
                           synthetic_terminal=.05)
FALLBACKS = ('none', 'fresh', 'synthetic', 'no_live_state')
SOURCES = ('fresh', 'live', 'replay', 'synthetic')
SOURCE = {name: index for index, name in enumerate(SOURCES)}
TRAVEL_STRATA = (0., 64., 256., np.inf)
EVENT_TABLE_SIZE = 1 << 18


@dataclass(frozen=True)
class TaskBudget:
    """Requested shares per task, plus replay age/reuse limits shared by loader workers.

    ``terminal_fallback_cap`` bounds certified synthetic failures used in place of missing
    real terminal replay, as a fraction of requested terminal slots.
    """
    shares: tuple = tuple(DEFAULT_TASK_SHARES[name] for name in TASKS)
    terminal_fallback_cap: float = .5
    replay_max_age: int = 12000
    replay_event_cap: int = 64

    def __post_init__(self):
        shares = np.asarray(self.shares, np.float64)
        if shares.shape != (len(TASKS),) or not np.isfinite(shares).all() or (shares < 0).any() or abs(shares.sum()-1) > 1e-6:
            raise ValueError(f'Task shares need {len(TASKS)} nonnegative values summing to one ({", ".join(TASKS)})')
        object.__setattr__(self, 'shares', tuple(float(v) for v in shares))
        if not 0 <= self.terminal_fallback_cap <= 1 or self.replay_max_age < 1 or self.replay_event_cap < 1:
            raise ValueError('Invalid terminal fallback cap, replay age ceiling or event cap')

    @classmethod
    def parse(cls, pairs=None, **limits):
        shares = dict(DEFAULT_TASK_SHARES)
        for pair in pairs or ():
            name, _, value = str(pair).partition('=')
            if name not in TASK:
                raise ValueError(f'Unknown task {name!r}; choose from {", ".join(TASKS)}')
            shares[name] = float(value)
        return cls(tuple(shares[name] for name in TASKS), **limits)

    def to_dict(self):
        return dict(shares=dict(zip(TASKS, self.shares)), terminal_fallback_cap=self.terminal_fallback_cap,
                    replay_max_age=self.replay_max_age, replay_event_cap=self.replay_event_cap)


def stable_key(*parts):
    digest = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, 'little') & ((1 << 62)-1)


class ReplayIndex:
    """Eligibility index by class, fiber, episode and event; rows are drawn last.

    Fibers are uniform, then episodes, then events, then rows, so dense windows cannot
    dominate merely by containing more decisions. Ordinary following is additionally
    stratified by travelled length. Each event group carries a stable key for the
    shared reuse counter.
    """
    def __init__(self, caches):
        from vesuvius.neural_tracing.fiber_follow.data.state_labels import REPLAY_CLASS, REPLAY_CLASSES
        self.caches = list(caches)
        self.groups = {name: {} for name in REPLAY_CLASSES}
        self.ordinary = [dict() for _ in range(len(TRAVEL_STRATA)-1)]
        for ci, op in enumerate(self.caches):
            identity = op.provenance["cache_id"]
            classes = np.asarray(op.replay_class)
            for name, index in REPLAY_CLASS.items():
                rows = np.flatnonzero(classes == index)
                if not len(rows):
                    continue
                if name == 'ordinary':
                    strata = np.digitize(np.asarray(op.travelled)[rows], TRAVEL_STRATA[1:-1])
                    keys = np.c_[np.asarray(op.episode)[rows], strata]
                else:
                    strata = None
                    keys = np.c_[np.asarray(op.episode)[rows], np.asarray(op.event_id)[rows]]
                for key in np.unique(keys, axis=0):
                    members = rows[(keys == key).all(1)]
                    episode, event = int(key[0]), int(key[1])
                    group = (ci, episode, event, members, stable_key(identity, episode, name, event))
                    fi = int(op.fiber_idx[members[0]])
                    table = self.ordinary[event] if name == 'ordinary' else self.groups[name]
                    table.setdefault(fi, {}).setdefault((ci, episode), []).append(group)

    def counts(self):
        tables = dict(self.groups, ordinary=None)
        result = {}
        for name, table in tables.items():
            tables_ = self.ordinary if name == 'ordinary' else [table]
            groups = [g for t in tables_ for episodes in t.values() for gs in episodes.values() for g in gs]
            result[name] = dict(fibers=len({f for t in tables_ for f in t}), events=len(groups),
                                rows=int(sum(len(g[3]) for g in groups)))
        return result

    def draw(self, name, rng, eligible):
        """One event group from an eligible cache, or None."""
        tables = [t for t in self.ordinary if t] if name == 'ordinary' else [self.groups[name]]
        if name == 'ordinary' and tables:
            tables = [tables[int(rng.integers(len(tables)))]]
        for table in tables:
            fibers = [f for f, episodes in table.items() if any(e[0] in eligible for e in episodes)]
            if not fibers:
                return None
            episodes = table[fibers[int(rng.integers(len(fibers)))]]
            choices = [key for key in episodes if key[0] in eligible]
            groups = episodes[choices[int(rng.integers(len(choices)))]]
            return groups[int(rng.integers(len(groups)))]
        return None


class FollowDataset(torch.utils.data.IterableDataset):
    """Explicit task budget per batch slot; every draw is relabeled before crop I/O.

    Tasks follow ``TaskBudget``: fresh simulated traces (with startup and excursion
    allocations), live continuation (its own slots, never replay slots), five
    mutually exclusive DAgger replay classes and certified synthetic failures.
    Missing replay falls back to a fresh example of the corresponding kind; missing
    terminal replay may use certified synthetic failures within the configured cap.
    Fallbacks and deficits are recorded per item. Replay age ceilings and per-event
    draw caps are shared across loader workers through shared memory.
    """
    def __init__(self, fibers, vol_spec, cfg, chunk=2, seed=0,
                 cache_bytes=1 << 30, onpolicy=None,
                 window=256., pool_size=12, window_samples=192, replay_index=None, refresh_chunks=8,
                 batch_builder=None, budget=None, length_power=1.):
        import multiprocessing as mp
        self.budget = budget or TaskBudget()
        self.fibers, self.vol_spec, self.cfg = fibers, vol_spec, cfg
        self.chunk, self.seed, self.cache_bytes = chunk, seed, cache_bytes
        self.window, self.pool_size, self.window_samples = window, pool_size, window_samples
        self.replay_index, self.refresh_chunks = replay_index, refresh_chunks
        self.batch_builder = batch_builder
        self.live_continuation = None  # Optional trainer-to-worker prediction feedback.
        self.remote_prefetch = None  # Optional trainer-owned process queue client.
        self.episodes = None  # EpisodeSpec: plans of whole episodes (model 'sequence'); chunk then counts episodes
        self.remote_prefetch_lookahead = 0  # Planned microbatches per source/worker.
        # Shared by every loader worker: current update and per-event replay draw counts.
        self.step = mp.Value('q', 0)
        self.event_draws = mp.Array('i', EVENT_TABLE_SIZE)
        self._replay_paths = None
        self._set_replay(list(onpolicy or []))
        lengths = (fibers.lengths if hasattr(fibers, 'lengths')
                   else np.array([f.length for f in fibers]))
        if not len(lengths):
            raise ValueError('No training fibers')
        if not np.isfinite(length_power) or length_power < 0:
            raise ValueError('Fiber length power must be finite and nonnegative')
        # Power 1 draws uniformly over annotated arclength; larger powers favor long fibers.
        lengths = np.asarray(lengths, dtype=np.float64)
        if length_power != 1:
            lengths = lengths**length_power
        self.weights = lengths / lengths.sum()
        self.terminal_requests = self.synthetic_fallbacks = 0

    def set_step(self, step):
        self.step.value = int(step)

    def _set_replay(self, caches):
        if caches:
            caches = usable_replay(caches, self.fibers, self.cfg.n_history, self.vol_spec.grid_scale)
        self.onpolicy = caches
        self.index = ReplayIndex(caches)
        self._eligible = (None, frozenset())

    def eligible_caches(self):
        step = int(self.step.value)
        if self._eligible[0] != step:
            self._eligible = (step, frozenset(i for i, op in enumerate(self.onpolicy)
                                              if step-int(op.provenance['step']) <= self.budget.replay_max_age))
        return self._eligible[1]

    def claim(self, key):
        """Count one draw of an event against its shared cap; False once exhausted."""
        slot = key % EVENT_TABLE_SIZE
        with self.event_draws.get_lock():
            if self.event_draws[slot] >= self.budget.replay_event_cap:
                return False
            self.event_draws[slot] += 1
            return True

    def max_event_reuse(self):
        return int(max(self.event_draws[:])) if len(self.event_draws) else 0

    def refresh_replay(self):
        if self.replay_index is None or not os.path.exists(self.replay_index):
            return
        with open(self.replay_index) as fh:
            paths = json.load(fh)
        if paths != self._replay_paths:
            self._set_replay(load_replay(paths))
            self._replay_paths = paths

    def prepare(self, item, rng):
        """Let the batch builder attach geometry it reads later (e.g. path patches)."""
        if hasattr(self.batch_builder, 'prepare'):
            return self.batch_builder.prepare(item, self.fibers[item['fiber_ref'][0]], rng)
        return item

    # ----------------------------------------------------------------- task items

    def fresh_location(self, rng, windows):
        location = (self.batch_builder.fresh_location(rng)
                    if hasattr(self.batch_builder, 'fresh_location') else None)
        if location is not None:
            return location['fiber'], location['t'], location['reverse'], location['source']
        while len(windows) < self.pool_size:
            fi = rng.choice(len(self.fibers), p=self.weights)
            windows.append([fi, rng.uniform(0, self.fibers[fi].length), self.window_samples])
        wi = rng.integers(len(windows))
        fi, center, _ = windows[wi]
        f = self.fibers[fi]
        windows[wi][2] -= 1
        if windows[wi][2] <= 0:
            windows.pop(wi)
        rev = bool(rng.integers(2))
        if rng.random() < .1:
            t = max(0, f.length-rng.uniform(0, self.cfg.future_s[-1]*1.5))
        else:
            original_t = np.clip(center+rng.uniform(-self.window/2, self.window/2), 0, f.length)
            t = f.length-original_t if rev else original_t
        return int(fi), float(t), rev, 0

    def fresh_item(self, rng, windows, **options):
        """A simulated trace; ``options`` fix the startup category or force an excursion."""
        item = (self.batch_builder.replace_fresh(self.cfg, rng, **options)
                if hasattr(self.batch_builder, 'replace_fresh') else None)
        if item is None:
            fi, t, rev, location = self.fresh_location(rng, windows)
            item = make_sample(self.fibers[fi], t, rev, self.cfg, rng, **options)
            item.update(fiber_ref=(fi, t, rev), location_source=location)
            item = self.prepare(item, rng)
        item.update(source=SOURCE['fresh'], source_step=-1)
        return item

    def recovery_item(self, rng, windows):
        """Fresh substitute for missing recoverable replay: a displaced excursion head."""
        from vesuvius.neural_tracing.fiber_follow.data.state_labels import RECOVERABLE
        for _ in range(8):
            item = self.fresh_item(rng, windows, startup=STARTUP_CATEGORIES.index('established'), excursion=True)
            if item['supervision'] == RECOVERABLE:
                return item
        return item

    def synthetic_item(self, rng):
        """Certified wrong continuation with visible original-fiber evidence, or None."""
        if not hasattr(self.batch_builder, 'synthetic_terminal'):
            return None
        for _ in range(3):
            item = self.batch_builder.synthetic_terminal(self.cfg, rng)
            if item is None:
                continue
            item = self.prepare(item, rng)
            if item.get('identity_evidence', False):
                return item
        return None

    def replay_draw(self, name, rng):
        """One relabeled row of a replay class, honoring the age ceiling and event cap."""
        eligible = self.eligible_caches()
        for _ in range(8):
            group = self.index.draw(name, rng, eligible)
            if group is None:
                return None
            if not self.claim(group[4]):
                continue
            ci, episode, event, rows, key = group
            item = self.replay_item(self.onpolicy[ci], int(rng.choice(rows)), rng)
            item.update(replay_event=key, replay_episode=stable_key(key, episode))
            return item
        return None

    def replay_item(self, op, j, rng):
        from vesuvius.neural_tracing.fiber_follow.tracing.heading import FRAME_POLICY
        fi, t, reverse = int(op.fiber_idx[j]), float(op.t[j]), bool(op.reverse[j])
        item = label_state(self.fibers[fi], op.pos[j], op.frame[j], op.hist[j], op.hmask[j], self.cfg,
                           t=t, reverse=reverse, trace=replay_facts(op, j, self.cfg))
        item.update(source=SOURCE['replay'], source_step=int(op.provenance['step']),
                    fiber_ref=(fi, self.fibers[fi].length-t if reverse else t, reverse),
                    replay_class=int(op.replay_class[j]), travelled=float(op.travelled[j]),
                    heading_start=int(op.heading_start[j]), frame_policy=op.provenance.get('frame_policy', FRAME_POLICY),
                    labeler_state=replay_labeler_state(op, j))
        item.update({k: getattr(op, k)[j] for k in SEED_FIELDS})
        item['observed_path'] = op.observed_prefix(j)
        return self.prepare(item, rng)

    def episode_items(self, rng, windows, index):
        """One episode (``self.episodes``) on a fresh fiber location: decisions in step order, each prepared like a
        fresh item; the episode shares one photometric draw (one scan) while frames and rolls stay per step."""
        spec = self.episodes
        for _ in range(1000):
            fi, _, reverse, location = self.fresh_location(rng, windows)
            length = self.fibers[fi].length
            steps = min(spec.steps, int(length//spec.commit))
            if steps < spec.min_steps:
                continue
            t0 = float(rng.uniform(0., length-steps*spec.commit))
            items, shared = [], None
            for step, (item, segment) in enumerate(episode_decisions(self.fibers[fi], t0, reverse, steps, spec.commit,
                                                                     self.cfg, rng)):
                item.update(fiber_ref=(fi, float(t0+step*spec.commit), reverse), location_source=location)
                item = self.prepare(item, rng)
                if shared is None:
                    shared = {key: item[key] for key in ('photometric', 'blur_sigma') if key in item}
                item.update(shared, source=SOURCE['fresh'], source_step=-1, task_requested=TASK['fresh'],
                            task_delivered=TASK['fresh'], task_fallback=0, episode_index=index, episode_step=step,
                            episode_supervised=step >= steps-spec.supervised, episode_segment=segment)
                items.append(item)
            return mark_episode_identity(items)
        raise ValueError('Could not draw a training episode: no fiber is long enough')

    def replay_episode_items(self, kind, rng, index):
        """One on-policy episode (``self.episodes``) from the replay caches: consecutive recorded decisions of one
        collected trace whose supervised (trailing) window contains a decision of replay class ``kind``. Each step is
        relabeled and prepared as a replay item; its committed segment is the recorded trace from its head to the next
        head (the last step's history token is never read, so its segment is its head alone). The collector keeps every
        decision for sequence models (``--max-states``, stride 0), so a trace's rows are consecutive decisions.
        Returns None when no eligible event is available."""
        spec = self.episodes
        eligible = self.eligible_caches()
        for _ in range(8):
            group = self.index.draw(kind, rng, eligible)
            if group is None:
                return None
            if not self.claim(group[4]):
                continue
            ci, episode, event, members, key = group
            op = self.onpolicy[ci]
            trace = np.flatnonzero(np.asarray(op.episode) == episode)
            trace = trace[np.argsort(np.asarray(op.source_row)[trace])]
            if np.any(np.diff(np.asarray(op.source_row)[trace]) != 1):
                raise ValueError('Replay episodes need every decision of a trace; collect with --max-states and stride 0')
            anchor = int(np.searchsorted(np.asarray(op.source_row)[trace], int(op.source_row[int(rng.choice(members))])))
            end = min(len(trace)-1, anchor+int(rng.integers(spec.supervised)))
            start = max(0, end-spec.steps+1)
            if end-start+1 < spec.min_steps:
                continue
            rows = trace[start:end+1]
            items, shared = [], None
            for step, j in enumerate(rows):
                item = self.replay_item(op, int(j), rng)
                if shared is None:
                    shared = {name: item[name] for name in ('photometric', 'blur_sigma') if name in item}
                if step+1 < len(rows):
                    segment = op.track_pos[int(op.seq_end[j])-1:int(op.seq_end[rows[step+1]])]
                else:
                    segment = np.asarray(op.pos[j])[None]
                item.update(shared, replay_event=key, replay_episode=stable_key(key, episode), episode_index=index,
                            episode_step=step, episode_supervised=step >= len(rows)-spec.supervised,
                            episode_segment=np.asarray(segment, np.float64))
                items.append(item)
            return mark_episode_identity(items)
        return None

    def episode_plan(self, rng, windows, index):
        """One training episode by the task budget: a simulated trace ('fresh') or an on-policy replay episode
        ('dagger_<class>', falling back to a fresh episode when none is eligible)."""
        name = TASKS[int(rng.choice(len(TASKS), p=self.budget.shares))]
        if name != 'fresh' and not name.startswith('dagger_'):
            raise ValueError(f'Episode training draws fresh or DAgger episodes only, not {name!r}')
        items = self.replay_episode_items(name[len('dagger_'):], rng, index) if name != 'fresh' else None
        delivered, fallback = (name, 'none') if items is not None else ('fresh', 'none' if name == 'fresh' else 'fresh')
        if items is None:
            items = self.episode_items(rng, windows, index)
        for item in items:
            item.update(task_requested=TASK[name], task_delivered=TASK[delivered], task_fallback=FALLBACKS.index(fallback))
        return items

    def live_start(self, rng, windows):
        """Chain starts: seed-only (share ``cfg.live_seed_start``), else a recorded valid pre-excursion/recoverable
        prefix."""
        from vesuvius.neural_tracing.fiber_follow.data.state_labels import FOLLOWING, RECOVERABLE
        if rng.random() >= self.cfg.live_seed_start:
            for name in ('pre_excursion', 'recoverable') if rng.random() < .5 else ('recoverable', 'pre_excursion'):
                item = self.replay_draw(name, rng)
                if item is not None and item['geometry_valid'] and item['supervision'] in (FOLLOWING, RECOVERABLE):
                    item.update(live_start='replay', live_loop_start=0)
                    return item
        item = self.fresh_item(rng, windows, startup=STARTUP_CATEGORIES.index('seed_only'))
        item.update(live_start='seed', live_loop_start=len(item['observed_path'])-1)
        return item

    def task_item(self, task, rng, windows):
        name, fallback, delivered = TASKS[task], 'none', TASKS[task]
        if name == 'fresh':
            item = self.fresh_item(rng, windows)
        elif name == 'live':
            item = (self.live_continuation.placeholder(self, rng, windows) if self.live_continuation is not None
                    else None)
            if item is None:
                item, fallback, delivered = self.fresh_item(rng, windows), 'no_live_state', 'fresh'
        elif name == 'synthetic_terminal':
            item = self.synthetic_item(rng)
            if item is None:
                item, fallback, delivered = self.fresh_item(rng, windows), 'fresh', 'fresh'
        else:
            kind = name[len('dagger_'):]
            item = self.replay_draw(kind, rng)
            if kind == 'terminal':
                self.terminal_requests += 1
                if item is None and self.synthetic_fallbacks < self.budget.terminal_fallback_cap*self.terminal_requests:
                    item = self.synthetic_item(rng)
                    if item is not None:
                        self.synthetic_fallbacks += 1
                        fallback, delivered = 'synthetic', 'synthetic_terminal'
            if item is None:
                fallback, delivered = 'fresh', 'fresh'
                item = (self.recovery_item(rng, windows) if kind == 'recoverable' else
                        self.fresh_item(rng, windows, startup=STARTUP_CATEGORIES.index('established')))
        item.update(task_requested=task, task_delivered=TASK[delivered], task_fallback=FALLBACKS.index(fallback))
        return item

    def prefetch_bounds(self, items, vol):
        if hasattr(self.batch_builder, 'prefetch_bounds'):
            return [bound for item in items for bound in self.batch_builder.prefetch_bounds(item,vol)]
        from vesuvius.neural_tracing.fiber_follow.tracing.heading import frame_prefetch_bounds
        return [bound for item in items for bound in frame_prefetch_bounds(item,self.cfg.crop,vol.input_scale)]

    def prefetch_items(self, items, vol, *, required=False):
        if self.remote_prefetch is None or not items:
            return
        # Whole lookahead windows are retained by the remote process instead of
        # sending disposable per-item hints into a potentially full queue.
        if not required and self.remote_prefetch_lookahead:
            return
        bounds = self.prefetch_bounds(items,vol)
        if required:
            self.remote_prefetch.ensure(vol.ct,bounds)
        else:
            self.remote_prefetch.submit(vol.ct,bounds)

    def resolve_seeds(self, items, vol, rng, windows):
        """CT seed headings; a seed without CT orientation is rejected and its task redrawn."""
        from vesuvius.neural_tracing.fiber_follow.tracing.heading import SeedHeadingError
        rejected = 0
        if self.episodes is not None:
            # A rejected seed redraws its whole episode: every decision of a trace shares the seed.
            groups = {}
            for item in items:
                groups.setdefault(item['episode_index'], []).append(item)
            resolved = []
            for index, group in groups.items():
                for _ in range(MAX_CT_FRAME_REJECTIONS):
                    try:
                        for item in group:
                            resolve_trace_seed(item, vol)
                        break
                    except SeedHeadingError:
                        rejected += 1
                        group = self.episode_items(rng, windows, index)
                        self.prefetch_items(group, vol, required=True)
                else:
                    raise SeedHeadingError('No CT-oriented replacement episode after repeated rejections')
                resolved.extend(group)
            items[:] = resolved
            return rejected
        for index, item in enumerate(items):
            for _ in range(MAX_CT_FRAME_REJECTIONS):
                try:
                    resolve_trace_seed(item, vol)
                    break
                except SeedHeadingError:
                    rejected += 1
                    replacement = self.task_item(item['task_requested'], rng, windows)
                    self.prefetch_items([replacement], vol, required=True)
                    item = items[index] = replacement
            else:
                raise SeedHeadingError('No CT-oriented replacement seed after repeated rejections')
        return rejected

    def __iter__(self):
        from vesuvius.neural_tracing.fiber_follow.tracing.heading import SeedHeadingError
        torch.set_num_threads(1)
        if self.remote_prefetch is not None:
            self.remote_prefetch.ensure_metadata(self.vol_spec)
            vol = FiberVolume(self.vol_spec,cache_bytes=self.cache_bytes,cache_only=True)
        else:
            vol = FiberVolume(self.vol_spec,cache_bytes=self.cache_bytes)
        grid = torch.from_numpy(crop_local_grid(self.cfg.crop)).float()
        lookahead = self.remote_prefetch_lookahead if self.remote_prefetch is not None else 0
        if lookahead < 0:
            raise ValueError('Remote prefetch lookahead must be nonnegative')
        windows = []
        plans, pending = self._iter_plans(vol, windows), deque()
        first = True
        worker = torch.utils.data.get_worker_info()
        live_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 8123, 0 if worker is None else worker.id]))
        rejected = seed_rejections = 0
        while True:
            # Deliver the first batch promptly. Fill the deeper plan queue when
            # the loader asks for its next batch, overlapping the first update.
            while len(pending) < (1 if first else lookahead+1):
                plan = next(plans)
                bounds = self.prefetch_bounds(plan,vol) if lookahead else None
                pending.append((plan,bounds))
            first = False
            if lookahead:
                self.remote_prefetch.lookahead(vol.ct,
                    [bounds for _,bounds in pending],scope=id(self))
            items, bounds = pending.popleft()
            live_outcomes = None
            if self.live_continuation is not None:
                items = [self.live_continuation.resolve(item, self, vol, live_rng)
                         if item.get('live_requested') else item for item in items]
                live_outcomes = self.live_continuation.take_outcomes()
                # Resolved positions differ from the geometry lookahead plan.
                bounds = self.prefetch_bounds(items, vol) if lookahead else None
            if lookahead:
                self.remote_prefetch.ensure(vol.ct,bounds)
            else:
                self.prefetch_items(items,vol,required=True)
            seed_rejections += self.resolve_seeds(items, vol, live_rng, windows)
            try:
                if self.batch_builder is None:
                    raise ValueError('Training requires the common observation builder')
                batch = self.batch_builder(items, vol)
            except SeedHeadingError as exc:
                # Invalid crop or slab orientation context rejects the whole plan.
                # Weak CT orientation falls back in ct_frame; I/O errors propagate.
                rejected += 1
                if rejected >= MAX_CT_FRAME_REJECTIONS:
                    worker = torch.utils.data.get_worker_info()
                    raise SeedHeadingError(
                        f'Could not build a CT-oriented training batch after {rejected} consecutive rejected plans '
                        f'(source={getattr(self.vol_spec, "ct_zarr", "unknown")}, '
                        f'worker={worker.id if worker else 0}): {exc}') from exc
                continue
            if self.episodes is not None:
                batch.update(episode_tensors(items, self.episodes.commit))
            if self.live_continuation is not None:
                batch['_live_states'] = [self.live_continuation.metadata(item) for item in items]
                batch['live_depth'] = torch.tensor([item.get('live_depth', 0) for item in items])
                batch['live_limit'] = torch.tensor([item.get('live_limit', 0) for item in items])
                batch['live_travelled'] = torch.tensor([item.get('live_travelled', 0.) for item in items])
            # Row-aligned counters survive the usual tensor batch movers; totals sit in
            # row 0 so summing never multiplies them by the batch size. Zero counts are omitted.
            counters = dict(ct_frame_rejected_batches=rejected, ct_seed_rejections=seed_rejections,
                            replay_max_event_reuse=self.max_event_reuse() if self.onpolicy else 0)
            if live_outcomes is not None:
                counters.update({'live_'+name: value for name, value in live_outcomes.items()})
            for name, value in counters.items():
                if value:
                    batch[name] = torch.zeros(len(items), dtype=torch.int64)
                    batch[name][0] = value
            rejected = seed_rejections = 0
            yield batch

    def _iter_plans(self, vol, windows):
        """Ordered geometry/augmentation plans, without CT or dense image tensors."""
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed*1000 + (0 if info is None else info.id))
        chunks = 0
        while True:
            if chunks % self.refresh_chunks == 0:
                self.refresh_replay()
            chunks += 1
            items = []
            if self.episodes is not None:
                for index in range(self.chunk):
                    episode = self.episode_plan(rng, windows, index)
                    self.prefetch_items(episode, vol)
                    items.extend(episode)
                yield items
                continue
            for task in rng.choice(len(TASKS), size=self.chunk, p=self.budget.shares):
                item = self.task_item(int(task), rng, windows)
                self.prefetch_items([item], vol)
                items.append(item)
            if (self.remote_prefetch is not None and self.remote_prefetch_lookahead
                    and hasattr(self.batch_builder,'prepare_sampling_feedback')):
                self.batch_builder.prepare_sampling_feedback(items)
            yield items


_GRID_CACHE: dict = {}


def _grid_flat(crop: CropSpec) -> np.ndarray:
    g = _GRID_CACHE.get(crop)
    if g is None:
        g = _GRID_CACHE[crop] = crop_local_grid(crop).reshape(-1, 3).astype(np.float64)
    return g


_CORNER_CACHE: dict = {}


def crop_corners(crop: CropSpec) -> np.ndarray:
    """Eight local (a, b, c) corners; the crop's sample points are their convex hull."""
    corners = _CORNER_CACHE.get(crop)
    if corners is None:
        lc, fc = crop.lateral_coords, crop.forward_coords
        corners = _CORNER_CACHE[crop] = np.array([(a, b, c) for a in (lc[0], lc[-1])
                                                  for b in (lc[0], lc[-1]) for c in (fc[0], fc[-1])], np.float64)
    return corners


def tight_block(pos, frame, crop: CropSpec, scale: float = 1.):
    """zyx start and size of the smallest axis-aligned block around one oriented crop.

    Source-array coordinates (``scale`` source voxels per trace voxel). The block
    holds every sample point plus the trilinear/nearest support voxel above it
    and one guard voxel on each side, so a per-sample sampler that zero-pads
    beyond the block reads exactly the values it reads from the larger
    rotation-invariant ``block_start`` block (arrays fill outside with zero too).
    Unlike ``block_start`` the footprint depends on the frame, so the holdout
    guard keeps using the rotation-invariant block.
    """
    world = np.asarray(pos, np.float64)*scale + crop_corners(crop) @ (np.asarray(frame, np.float64)*scale).T
    zyx = world[:, ::-1]
    lo = np.floor(zyx.min(0)).astype(np.int64) - 1
    hi = np.floor(zyx.max(0)).astype(np.int64) + 3
    return lo, hi - lo


def read_tight_blocks(items, vol: FiberVolume, crop: CropSpec, pool=None):
    """Per-item minimal blocks for per-sample CPU sampling: (list of raw, list of zyx starts).

    Same values as the rotation-invariant ``block_start`` block at every sample point, reading roughly a
    third of the voxels and touching about half the chunks per item.
    """
    scale = getattr(vol, 'input_scale', 1.)
    bounds = [tight_block(it["pos"], it["frame"], crop, scale) for it in items]
    read = lambda b: vol.raw_block(b[0], b[1])
    raw = list(map(read, bounds) if pool is None else pool.map(read, bounds))
    return raw, [start for start, _ in bounds]


def collate_targets(items):
    """Pack annotation-only tensors, independently of a model's image inputs."""
    st = lambda k: torch.from_numpy(np.stack([it[k] for it in items]).astype(np.float32))
    out = {}
    if "fut_local" in items[0]:
        out["gt_history"] = st("gt_history")
        out["gt_history_mask"] = st("gt_history_mask")
        out["plane_ab"] = st("plane_ab")
        out["plane_mask"] = st("plane_mask")
        for key in ("dense_ab", "dense_mask", "end_local", "endpoint_known", "terminal", "match_distance"):
            out[key] = st(key)
        for key in ('geometry_valid', 'confidence_valid'):
            out[key] = torch.tensor([bool(it[key]) for it in items], dtype=torch.bool)
        for key in ('supervision', 'supervision_reason'):
            out[key] = torch.tensor([int(it[key]) for it in items], dtype=torch.long)
    if 'source' in items[0]:
        out['source'] = torch.tensor([it['source'] for it in items], dtype=torch.long)
        out['source_step'] = torch.tensor([it.get('source_step', -1) for it in items], dtype=torch.long)
        out['fiber_id'] = torch.tensor([it['fiber_ref'][0] for it in items], dtype=torch.long)
        out['seed_age'] = torch.tensor([float(it.get('seed_age', 0.)) for it in items], dtype=torch.float32)
        out['travelled'] = torch.tensor([float(it.get('travelled', 0.)) for it in items], dtype=torch.float32)
        for key, default in (('task_requested', -1), ('task_delivered', -1), ('task_fallback', 0),
                             ('startup', -1), ('replay_class', -1), ('replay_episode', -1), ('replay_event', -1)):
            out[key] = torch.tensor([int(it.get(key, default)) for it in items], dtype=torch.long)
        for key in ('excursion', 'live_requested', 'live_continuation', 'live_terminal'):
            out[key] = torch.tensor([bool(it.get(key, False)) for it in items], dtype=torch.bool)
    return out


# ------------------------------------------------------------ on-policy states


def label_state(fiber, pos, frame, hist_world, hmask, cfg, *, t, reverse, trace):
    """Relabel an exact observed frame/history against its original fiber.

    ``t`` is the fiber's own arclength of the matched correspondence and ``trace`` the
    state's trace facts (``state_labels.facts``). This is the one labeling call used by
    the collector, replay, live continuation and synthetic sources.
    """
    from vesuvius.neural_tracing.fiber_follow.data.state_labels import supervise
    if len(hist_world) != cfg.n_history or len(hmask) != cfg.n_history:
        raise ValueError('Replay history must equal n_history')
    traversal_t = fiber.length-t if reverse else t
    item = dict(pos=pos, frame=frame, hist_local=(hist_world-pos) @ frame,
                _frame_label_context=(fiber, traversal_t, reverse, cfg), fiber_family=fiber.tag,
                hmask=np.asarray(hmask, np.float32), trace_facts=dict(trace),
                **continuation_targets(fiber, traversal_t, reverse, pos, frame, cfg))
    if not (trace['match_valid'] and not trace['match_ambiguous']):
        item['gt_history_mask'] = np.zeros_like(item['gt_history_mask'])
    return supervise(item)


def replay_facts(op, j, cfg):
    """Stored trace facts of one replay row, relabeled under this run's label contract."""
    from vesuvius.neural_tracing.fiber_follow.data.state_labels import DEPARTURE_PATIENCE, facts
    return facts(match_distance=float(op.match_distance[j]), window_distance=float(op.window_distance[j]),
                 match_valid=bool(op.match_valid[j]), match_ambiguous=bool(op.match_ambiguous[j]),
                 switched=bool(op.switched[j]), beyond_end=bool(op.beyond_end[j]),
                 tolerance=cfg.label_tolerance, max_recovery_distance=cfg.max_recovery_distance,
                 t=float(op.t[j]), excursion=bool(op.bad_run[j] >= DEPARTURE_PATIENCE), bad_run=int(op.bad_run[j]),
                 bad_run_start=float(op.bad_run_start[j]), departure_distance=float(op.departure_distance[j]),
                 boundary_distance=float(op.boundary_distance[j]), switch_distance=float(op.switch_distance[j]))


def replay_labeler_state(op, j):
    """Exact ``TraceLabeler`` state after row ``j``; restarts resume from it."""
    finite = lambda value: None if not np.isfinite(value) else float(value)
    return dict(t=float(op.t[j]), last_travelled=float(op.travelled[j]), bad_run=int(op.bad_run[j]),
                bad_run_start=finite(op.bad_run_start[j]), started=True,
                departure_distance=finite(op.departure_distance[j]),
                boundary_distance=finite(op.boundary_distance[j]), switch=None)


class OnPolicyStates:
    """Decision states (pos, frame, own-trace history) visited by a policy on GT fibers.

    One schema: every field below is required and validated on load; there are no
    defaults for absent fields. Loaded as memory-mapped .npy files so DataLoader workers
    share one copy (pickling sends only the directory path). Each trace's committed
    polyline is stored once in ``track_pos``; rows reference it by ``seq_start:seq_end``
    (the exclusive end includes the decision head). Nothing here except the observed
    geometry ever becomes a model input.
    """

    # name: (dtype, trailing shape); None trailing dims are checked against arrays.
    FIELDS = dict(
        fiber_idx=('i8', ()), t=('f8', ()), reverse=('?', ()), pos=('f8', (3,)), frame=('f8', (3, 3)),
        hist=('f8', (None, 3)), hmask=('f4', (None,)),
        seed_pos=('f8', (3,)), seed_tangent=('f8', (3,)), seed_age=('f4', ()), seed_valid=('?', ()),
        heading_start=('i8', ()), travelled=('f8', ()), episode=('i8', ()), source_row=('i8', ()),
        seq_start=('i8', ()), seq_end=('i8', ()),
        # Current supervision (state_labels.classify) and its trace facts.
        supervision=('i1', ()),
        match_distance=('f4', ()), window_distance=('f4', ()), match_valid=('?', ()), match_ambiguous=('?', ()),
        switched=('?', ()), beyond_end=('?', ()),
        # Historical events (NaN when absent) and resumable departure patience.
        departure_distance=('f8', ()), boundary_distance=('f8', ()), switch_distance=('f8', ()),
        bad_run=('i4', ()), bad_run_start=('f8', ()),
        # Replay membership.
        replay_class=('i1', ()), event_id=('i8', ()))
    TRACK = dict(track_pos=('f8', (3,)))

    def __init__(self, *, manifest, provenance, **arrays):
        expected = set(self.FIELDS) | set(self.TRACK)
        if set(arrays) != expected:
            raise ValueError(f'Replay fields differ from the schema: missing {sorted(expected-set(arrays))}, '
                             f'unexpected {sorted(set(arrays)-expected)}')
        for key in expected:
            setattr(self, key, np.asarray(arrays[key]))
        self._dir = None
        self.manifest = manifest
        self.provenance = provenance
        self.validate()

    def validate(self):
        n = len(self.pos)
        for key, (dtype, trailing) in {**self.FIELDS, **self.TRACK}.items():
            value = getattr(self, key)
            length = len(self.track_pos) if key in self.TRACK else n
            if len(value) != length or value.ndim != 1+len(trailing) or any(
                    expected is not None and actual != expected for expected, actual in zip(trailing, value.shape[1:])):
                raise ValueError(f'Replay field {key} has shape {value.shape}')
            if np.dtype(dtype).kind != value.dtype.kind:
                raise ValueError(f'Replay field {key} has dtype {value.dtype}')
        if n and np.any((self.seq_start < 0) | (self.seq_start >= self.seq_end) | (self.seq_end > len(self.track_pos))):
            raise ValueError('Replay rows reference tracks outside the saved tracks')
        if 'step' not in self.provenance or 'volume' not in self.provenance:
            raise ValueError('Replay provenance must record its source step and volume')

    def __len__(self):
        return len(self.pos)

    def observed_prefix(self, j):
        """Actual committed polyline through this decision; no future vertices."""
        start, end = int(self.seq_start[j]), int(self.seq_end[j])
        path = self.track_pos[start:end]
        if not np.allclose(path[-1], self.pos[j], atol=1e-5, rtol=0):
            raise ValueError('Replay prefix does not end at decision head')
        if self.seed_valid[j] and not np.allclose(path[0], self.seed_pos[j], atol=1e-5, rtol=0):
            raise ValueError('Replay prefix does not start at its original seed')
        return path

    def save(self, path):
        np.savez(path, __metadata__=json.dumps(dict(fibers=self.manifest, provenance=self.provenance)),
                 **{k: getattr(self, k) for k in (*self.FIELDS, *self.TRACK)})

    def validate_fibers(self, fibers):
        if self.manifest != fiber_manifest(fibers):
            raise ValueError("On-policy cache has incompatible fibers or geometry; "
                             "recollect with collect.py using the current controlled-span labels")
        if len(self) and (np.any(self.fiber_idx < 0) or np.any(self.fiber_idx >= len(fibers))):
            raise ValueError("On-policy cache contains invalid fiber indices")
        if len(self):
            # Decode only referenced fibers in lazy collections, and compare
            # against the same geometry that supplies the collector's arcs.
            indices, inverse = np.unique(self.fiber_idx, return_inverse=True)
            lengths = np.array([fibers[int(i)].length for i in indices])[inverse]
            if np.any(~np.isfinite(self.t)) or np.any((self.t < 0) | (self.t > lengths)):
                raise ValueError("On-policy cache contains arc positions outside controlled spans")

    @classmethod
    def load(cls, path):
        """``path``: .npz from collect.py (mirrored once to a sibling ``_mmap`` directory of
        .npy files) or such a directory."""
        path = os.fspath(path)
        d = path[:-4] + "_mmap" if path.endswith(".npz") else path
        metadata_path = os.path.join(d, "metadata.json")
        names = (*cls.FIELDS, *cls.TRACK)
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as z:
                metadata = json.loads(str(z["__metadata__"].item()))
                missing = [k for k in names if k not in z.files]
                if missing:
                    raise ValueError(f'Replay archive lacks schema fields {missing}; recollect it')
                # Include archive identity so an overwritten NPZ cannot reuse stale mmap arrays.
                stat = os.stat(path)
                metadata["archive"] = [stat.st_size, stat.st_mtime_ns]
                existing = None
                if os.path.exists(metadata_path):
                    with open(metadata_path) as fh:
                        existing = json.load(fh)
                if existing != metadata or any(not os.path.exists(os.path.join(d, k + ".npy")) for k in names):
                    os.makedirs(d, exist_ok=True)
                    for k in names:
                        np.save(os.path.join(d, k + ".npy"), z[k])
                    with open(metadata_path, "w") as fh:
                        json.dump(metadata, fh)
        obj = cls.__new__(cls)
        obj._open(d)
        return obj

    def _open(self, d):
        with open(os.path.join(d, "metadata.json")) as fh:
            metadata = json.load(fh)
        self.manifest = metadata["fibers"]
        self.provenance = metadata["provenance"]
        self._dir = d
        missing = [k for k in (*self.FIELDS, *self.TRACK) if not os.path.exists(os.path.join(d, k + ".npy"))]
        if missing:
            raise ValueError(f'Replay cache lacks schema fields {missing}; recollect it')
        for k in (*self.FIELDS, *self.TRACK):
            setattr(self, k, np.load(os.path.join(d, k + ".npy"), mmap_mode="r"))
        self.validate()

    def __getstate__(self):
        if self._dir is None:
            return self.__dict__
        return {"_dir": self._dir}

    def __setstate__(self, state):
        if state.get("_dir") and len(state) == 1:
            self._open(state["_dir"])
        else:
            self.__dict__.update(state)


def load_replay(paths):
    """Replay caches at ``paths``; one that cannot be loaded is skipped with a warning."""
    caches = []
    for path in paths:
        try:
            caches.append(OnPolicyStates.load(path))
        except (OSError, ValueError, KeyError) as error:
            print(f'Warning: skipping replay cache {path}: {error}', flush=True)
    return caches


def usable_replay(caches, fibers, n_history, grid_scale):
    """The caches whose fibers, history length and coordinate scale match this run; others are skipped with a
    warning (their states would be labeled against different data)."""
    usable = []
    for op in caches:
        try:
            op.validate_fibers(fibers)
            if op.hist.shape[1] != n_history:
                raise ValueError('replay history length differs from n_history')
            if op.provenance['volume']['grid_scale'] != grid_scale:
                raise ValueError('replay world coordinate scale differs from this run')
        except (ValueError, KeyError) as error:
            print(f'Warning: skipping replay cache {getattr(op, "_dir", "")}: {error}', flush=True)
        else:
            usable.append(op)
    return usable
