"""Simulated traces: trace noise, excursions and seed ages, and fresh training samples on an annotated fiber."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vesuvius.neural_tracing.tracers.shared.geometry import (
    CropSpec, arclength, frame_from_heading, interp_at, tangent_at, traversal_curve)
from vesuvius.neural_tracing.tracers.shared.policy import DEFAULT_MAX_RECOVERY_DISTANCE
from vesuvius.neural_tracing.tracers.shared.reference import trace_history
from vesuvius.neural_tracing.tracers.shared.annotations.traced import TracedFiber
from vesuvius.neural_tracing.tracers.fibers.data.targets import continuation_targets
from vesuvius.neural_tracing.tracers.fibers.frames.ct import linear12_heading, trace_heading
from vesuvius.neural_tracing.tracers.fibers.data.state_labels import constructed_facts, supervise


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
    startup_shares: tuple = (.35, .15, .15, .35)
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
    # voxels, fading out over ``seed_offset_ramp`` voxels of trace (default 0-1.5 vox; (0, 0) keeps seeds on the
    # annotation).
    seed_offset: tuple = (0., 1.5)
    seed_offset_ramp: float = 16.
    # Share of live-continuation chains that start at a fresh seed (the rest from replayed DAgger prefixes).
    live_seed_start: float = .7
    # Horizon (vox) of the previous-prediction crop axis for continuation crops: on by default (the gate horizon, 16
    # planes); the trainer sets it from the model (frames.learned.prediction_window: 0 only for a model with
    # crop_axis 'heading_model'). Fresh continuation crops get simulated_prediction; seeds never do.
    prediction_window: float = 16.
    label_tolerance: float = 1.5  # confidence-label tolerance; separate from the departure threshold
    max_recovery_distance: float = DEFAULT_MAX_RECOVERY_DISTANCE
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


# The model's previous prediction, fitted on its own traces (output/prediction_axis_20261006/fit_prediction_error.py:
# mixed_ct_afv_unified_scoring_v1 ckpt_019000, 48940 decisions): per vox past the head, the share of the head's
# lateral offset still present, and the RMS of a smooth lateral bend (calibrated so the simulated prediction axes match
# the real ones' angle distribution).
PREDICTION_OFFSET_DECAY = np.array([1.006, 0.999, 0.985, 0.968, 0.948, 0.927, 0.908, 0.890, 0.878, 0.867, 0.856, 0.847, 0.844, 0.840, 0.837, 0.834])
PREDICTION_BEND_RMS = np.array([0.0305, 0.0555, 0.0765, 0.0962, 0.1135, 0.1318, 0.1496, 0.1673, 0.1821, 0.1989, 0.2163, 0.2346, 0.2502, 0.2682, 0.2876, 0.3081])


def simulated_prediction(fiber, t, reverse, pos, window, rng):
    """A previous decision's prediction past the head ``pos`` (matched at traversal arclength ``t``), as the model
    makes it: the annotation 1..window vox ahead, carrying the head's lateral offset (PREDICTION_OFFSET_DECAY) plus one
    smooth lateral bend of random direction (PREDICTION_BEND_RMS). Returns (window, 3) world points, nearest first."""
    p, s = traversal_curve(fiber, reverse)
    ahead = np.minimum(t+np.arange(1., window+1e-9, 1.), s[-1])
    gt = interp_at(p, s, ahead)
    k = np.minimum(np.arange(len(ahead)), len(PREDICTION_OFFSET_DECAY)-1)
    tangent = lambda at: tangent_at(p, s, float(at))
    t0 = tangent(t)
    offset = np.asarray(pos)-interp_at(p, s, np.array([t]))[0]
    offset = offset-(offset @ t0)*t0
    end = tangent(ahead[-1])
    bend = rng.normal(size=3)
    bend = bend-(bend @ end)*end
    bend = bend/max(np.linalg.norm(bend), 1e-9)
    out = []
    for j, at in enumerate(ahead):
        tg = tangent(at)
        shift = PREDICTION_OFFSET_DECAY[k[j]]*offset+PREDICTION_BEND_RMS[k[j]]*bend
        out.append(gt[j]+shift-(shift @ tg)*tg)
    return np.asarray(out)


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
    path, sigma, _, details = noisy_trace(arcs, p, s, category, cfg, rng, excursion=excursion)
    return path, arcs, category, sigma, details


def noisy_trace(arcs, p, s, category, cfg: SampleConfig, rng: np.random.Generator, *, excursion=None):
    """The observed path at traversal arclengths ``arcs`` of curve (p, s): trace noise, then (established traces of at
    least 8 voxels, with probability ``excursion_probability`` unless ``excursion`` decides) one excursion ending at the
    path's end, then the seed's offset (``seed_offsets``). Returns (path, noise sigma, excursion offsets, excursion
    details)."""
    noise, sigma = trace_noise(arcs, p, s, cfg, rng)
    path = interp_at(p, s, arcs)+noise
    offsets, details = np.zeros_like(path), {}
    established = category == STARTUP_CATEGORIES.index('established') and arcs[-1]-arcs[0] >= 8.
    if established and (excursion if excursion is not None else rng.random() < cfg.excursion_probability):
        offsets, details = excursion_offsets(arcs, p, s, cfg, rng)
        path = path+offsets
    return path+seed_offsets(arcs, p, s, cfg, rng), sigma, offsets, details


def make_sample(fiber: TracedFiber, t: float, reverse: bool, cfg: SampleConfig, rng: np.random.Generator,
                *, startup=None, excursion=None):
    """One tracer decision on a simulated trace of this fiber, built exactly like inference.

    The trace started ``trace_prefix_length`` back at an annotated seed and followed GT
    with smooth lateral error (``trace_noise``), optionally plus a smooth excursion
    (``excursion_offsets``) on established traces, so the head is offset from GT. Crop
    heading, history and seed reference come from that observed path through the tracer's
    own functions. A path shorter than 12 voxels still holds the seed's CT heading; it and
    the seed tangent are resolved once CT is readable (``resolve_trace_seed``). Labels are
    the GT continuation from the offset head under the shared state contract. A continuation (any traced history)
    also gets a simulated previous prediction for its crop axis when ``cfg.prediction_window`` is set; seeds do not.
    ``startup`` fixes the draw category; ``excursion`` forces or forbids an excursion.
    """
    p, s = traversal_curve(fiber, reverse)
    path, arcs, category, sigma, details = simulated_trace(p, s, t, cfg, rng, startup=startup, excursion=excursion)
    return decision_on_path(fiber, t, reverse, cfg, path, arcs, category, sigma, details, rng)


def decision_on_path(fiber, t, reverse, cfg, path, arcs, category, sigma, details, rng=None):
    """The tracer decision whose head ends an observed path (``path``, sampled at traversal arclengths ``arcs``
    ending at ``t``): crop heading, history, seed reference and labels exactly as ``make_sample`` builds them."""
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
    if cfg.prediction_window and rng is not None and len(path) > 1:
        # A continuation (not a seed): the crop axis comes from the previous decision's (simulated) prediction.
        item['prev_prediction'] = simulated_prediction(fiber, t, reverse, pos, cfg.prediction_window, rng)
    return item
