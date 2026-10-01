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
    normalize,
    render_history,
    sample_oriented_fast,
    tangent_at,
)
from vesuvius.neural_tracing.fiber_follow.shared.fast_sample import sample_crop
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS, SEED_DEFAULTS, observed_seed


DATA_VERSION = 2
DATA_POLICY = "controlled_spans_v3"
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

    @property
    def length(self) -> float:
        return float(self.s[-1])


def load_fibers(fiber_dir: str, grid_scale: float = 8.0, spacing: float = 1.0) -> list[TracedFiber]:
    """Load only geometry between human control points, retaining span provenance.

    Every line point between the outer controls is equally valid supervision,
    regardless of tags or interpolation provenance. An outer control without an
    explicit termination tag is a censored boundary, not a physical endpoint.
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
        pieces, spans, start = [], [], 0.0
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
            end = start + float(arclength(piece)[-1])
            spans.append(FiberSpan(start, end, fib.control_point_segments[i]))
            pieces.append(piece if i == 0 else piece[1:])
            start = end
        pts = np.concatenate(pieces)
        if len(pts) < 8:
            continue
        tag = (fib.metadata.get("hv_classification") or {}).get("automatic_tag", "")
        endpoints = (False, False)
        if fib.version == 3:
            endpoints = tuple("kollesis_termination" in raw["control_points"][i].get("tags", [])
                              for i in (0, -1))
        full_s = arclength(line)
        tails = float(full_s[anchors[0]] + full_s[-1] - full_s[anchors[-1]])
        source_hash = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
        out.append(TracedFiber(os.path.basename(path), pts, arclength(pts), tag,
                              spans=tuple(spans),
                              endpoint_stop=endpoints, source_hash=source_hash,
                              excluded_tail_length=tails))
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
    """Split by the controlled geometry centroid; filter training states separately."""
    train, val = [], []
    for f in fibers:
        z = np.average((f.points[:-1, 2]+f.points[1:, 2])/2, weights=np.diff(f.s))
        (val if val_band.contains(z) else train).append(f)
    return train, val


@dataclass
class SampleConfig:
    crop: CropSpec = field(default_factory=lambda: CropSpec(depth=176, width=96, behind=128, history_render="segments", history_sigma=.35))
    n_future: int = 16
    future_step: float = 1.0
    n_history: int = 128
    recent_history_points: int = 128  # GT history for observed-state diagnostics only
    history_step: float = 1.0
    lateral_sigmas: tuple = (0.4, 1.0, 2.0)
    lateral_probs: tuple = (0.5, 0.35, 0.15)
    angle_sigmas_deg: tuple = (2.0, 5.0, 10.0)
    angle_probs: tuple = (0.5, 0.35, 0.15)
    no_history_prob: float = 0.1
    short_history_prob: float = 0.0  # conditional on history being present; opt-in for direct training
    history_jitter: float = 0.0  # independent point noise; disable for smooth history
    history_drift: float = 2.0  # accumulated lateral displacement, smooth over 16--64 voxels
    history_wobble: float = 1.0  # max amplitude (voxels) of slow lateral wobble on the own-trace history
    full_observed_history: bool = False  # direct slab model preserves the complete synthetic prefix
    dense_substeps: int = 4

    @property
    def future_s(self) -> np.ndarray:
        return self.future_step * np.arange(1, self.n_future + 1)


def _mix(rng, sigmas, probs):
    return float(rng.choice(sigmas, p=probs))


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


def make_sample(fiber: TracedFiber, t: float, reverse: bool, cfg: SampleConfig, rng: np.random.Generator,
                *, perturb=True, light_perturbation=None):
    """Sample GT geometry; perturb=False preserves position, tangent and history."""
    p, s = fiber.points, fiber.s
    if reverse:
        p = p[::-1]
        s = s[-1] - s[::-1]
    L = s[-1]
    g = interp_at(p, s, np.array([t]))[0]
    tau = tangent_at(p, s, t)
    # lateral offset perpendicular to tau
    fr = frame_from_heading(tau)
    lat = rng.normal(size=2) * _mix(rng, cfg.lateral_sigmas, cfg.lateral_probs) if perturb else np.zeros(2)
    if light_perturbation is not None:
        if perturb:
            raise ValueError('Light perturbation requires the legacy perturbations to be disabled')
        max_offset, max_angle = light_perturbation
        if t < cfg.history_step:
            max_offset = max_angle = 0.  # A seed-only state has no committed tip to move.
        azimuth = rng.uniform(0, 2*np.pi)
        lat = max_offset*np.sqrt(rng.random())*np.array([np.cos(azimuth), np.sin(azimuth)])
    delta = fr[:, 0] * lat[0] + fr[:, 1] * lat[1]
    pos = g + delta
    ang = math.radians(_mix(rng, cfg.angle_sigmas_deg, cfg.angle_probs)) * rng.normal() if perturb else 0.
    axis_ang = rng.uniform(0, 2 * np.pi) if perturb else 0.
    if light_perturbation is not None:
        ang = math.radians(rng.uniform(-max_angle, max_angle))
        axis_ang = rng.uniform(0, 2*np.pi)
    ax = math.cos(axis_ang) * fr[:, 0] + math.sin(axis_ang) * fr[:, 1]
    heading = normalize(math.cos(ang) * tau + math.sin(ang) * ax)
    frame = frame_from_heading(heading)  # provisional basis; CT resolves crop roll before sampling

    # history: GT points behind with the drift ramping in
    count = cfg.n_history
    if cfg.full_observed_history:
        available = max(1, int(t/cfg.history_step))
        count = max(count, int(rng.integers(1, available+1)) if perturb else available)
    k = np.arange(1, count + 1) * cfg.history_step
    th = t - k
    hmask = (th >= 0).astype(np.float32)
    if perturb and rng.random() < cfg.no_history_prob:
        hmask[:] = 0
    elif perturb:
        # Balance very short and intermediate startup histories. With this
        # option off, retain the existing sampler's random draws exactly.
        if cfg.short_history_prob > 0 and rng.random() < cfg.short_history_prob:
            split = min(8, cfg.n_history)
            lo, hi = (1, split) if cfg.n_history <= 8 or rng.random() < .5 else (9, min(32, cfg.n_history))
            hmask[int(rng.integers(lo, hi+1)):] = 0
        elif rng.random() < 0.2:
            hmask[int(rng.integers(0, cfg.n_history)):] = 0
    hist = interp_at(p, s, np.clip(th, 0, L))
    ramp = np.clip(1.0 - k / (cfg.n_history * cfg.history_step), 0, 1)[:, None]
    if perturb:
        hist = hist + ramp * delta[None] + rng.normal(size=hist.shape) * cfg.history_jitter
    if perturb and cfg.history_wobble > 0:
        # slow lateral wander of our own past path, zero at the current point
        amp = rng.uniform(0, cfg.history_wobble, size=2)
        lam = rng.uniform(15.0, 45.0, size=2)
        ph = rng.uniform(0, 2 * np.pi, size=2)
        w = amp * (np.sin(2 * np.pi * k[:, None] / lam + ph) - np.sin(ph))
        hist = hist + w[:, :1] * fr[:, 0] + w[:, 1:] * fr[:, 1]

    # A smooth displacement accumulates near the present; old observations
    # remain aligned. Include the endpoint displacement in the current point.
    span = rng.uniform(16., 64.) if perturb else 16.
    ramp = np.clip(1-k/span, 0, 1)
    ramp = ramp*ramp*(3-2*ramp)
    displacement = rng.normal(size=2)*cfg.history_drift if perturb else np.zeros(2)
    drift = fr[:,:2] @ displacement
    hist += ramp[:,None]*drift
    pos += drift
    if light_perturbation is not None:
        # Move only the recent tip smoothly. Preserve the earliest observation
        # (the seed), including on startup prefixes shorter than 32 voxels.
        valid = np.flatnonzero(hmask)
        span = min(32., float(k[valid[-1]])) if len(valid) else 0.
        ramp = np.clip(1-k/max(span, 1e-9), 0, 1)
        hist += (ramp*ramp*(3-2*ramp))[:,None]*delta
    observed = np.concatenate((hist[hmask > 0][::-1], pos[None]))
    seed = observed_seed(pos, frame, (hist-pos) @ frame, hmask)
    return dict(_generated_original_history=True, _seed_original_certified=True,
                observed_path=observed, **seed,
                pos=pos, frame=frame, hist_local=((hist-pos) @ frame)[:cfg.n_history], hmask=hmask[:cfg.n_history],
                **continuation_targets(fiber, t, reverse, pos, frame, cfg))


def continuation_targets(fiber, t, reverse, pos, frame, cfg, offtrack=False):
    """Dense supervised geometry; prediction-support gaps never remove GT labels.

    t is traversal arclength. Missing forward crossings are unknown, except
    continuation beyond an explicitly tagged physical endpoint.
    """
    p, s = fiber.points, fiber.s
    if reverse:
        p, s = p[::-1], s[-1]-s[::-1]
    tf = t + cfg.future_s
    fut = interp_at(p, s, np.clip(tf, 0, s[-1]))
    history_arc = t - np.arange(cfg.recent_history_points + 1) * cfg.history_step
    gt_history = (interp_at(p, s, np.clip(history_arc, 0, s[-1])) - pos) @ frame
    gt_history_mask = (history_arc >= 0).astype(np.float32)
    if offtrack:
        gt_history_mask[:] = 0
    dense_planes = np.linspace(cfg.future_step, cfg.future_s[-1],
                               (cfg.n_future-1)*cfg.dense_substeps+1)
    ab, mask = plane_targets(p, s, t, s[-1], pos, frame, cfg.future_s)
    dense_ab, dense_mask = plane_targets(p, s, t, s[-1], pos, frame, dense_planes)
    fmask = (tf <= s[-1]).astype(np.float32)
    if offtrack:
        mask[:] = 0
        dense_mask[:] = 0
        fmask[:] = 0
    end = (p[-1]-pos) @ frame
    # Only expose endpoint labels when it is within the local traversal window.
    known = fiber.endpoint_stop[0 if reverse else 1] and s[-1]-t <= 2.5*cfg.future_s[-1]
    return dict(gt_history=gt_history.astype(np.float32), gt_history_mask=gt_history_mask,
                fut_local=(fut-pos) @ frame, fmask=fmask,
                plane_ab=ab, plane_mask=mask, planes=cfg.future_s,
                dense_ab=dense_ab, dense_mask=dense_mask, dense_planes=dense_planes,
                end_local=end, endpoint_known=float(known), offtrack=float(offtrack))


# Source is independent of the within-source drift/departure stratum.
# Keep replay source IDs stable across saved diagnostics.
REPLAY_SOURCES = dict(recent=2)
DRIFT_BANDS = ((0.,1.), (1.,1.5), (1.5,2.), (2.,3.5))
REPLAY_FAILURES = ('ordinary', 'bank_switch', 'premature_stop', 'endpoint_overshoot', 'pre_switch')


def replay_pools(caches, *, failures=False, correct_only=False, natural_switch_only=False):
    """Drift/departure strata, optionally split by recorded failure, then fiber."""
    pools = [dict() for _ in range(9 if failures else 5)]
    for op in caches:
        off = np.asarray(op.offtrack,bool)
        kinds = np.asarray(op.failure_kind) if failures else np.zeros(len(op), np.int8)
        eligible = np.ones(len(op), bool)
        if natural_switch_only:
            if not failures:
                raise ValueError('Natural switches require failure strata')
            recorded = ((op.seq_start >= 0) & (op.seq_end > op.seq_start+1)
                        & np.isfinite(op.travelled) & (op.travelled > 0))
            # A pre-switch row alone may precede a later forced exploration.
            # Require a witnessed, nonexploratory switched state on that trace.
            natural = recorded & ~op.exploratory & off & (op.failure_kind == 1)
            traces = np.unique(op.seq_start[natural])
            eligible &= (recorded & ~op.exploratory & np.isin(op.seq_start, traces)
                         & np.isin(op.failure_kind, (1, 4)))
        if correct_only:
            # Collector departures are sticky across the entire committed prefix;
            # exploration remains marked after the first forced move. Hard rows
            # also cover stops and retrospective pre-failure windows. Require real
            # model progress: skip zero-progress states and unknown old caches.
            # Eligible prefixes still include their original annotated seed.
            eligible = (~off & ~np.asarray(op.hard, bool) & ~np.asarray(op.exploratory, bool)
                        & (np.asarray(op.failure_kind) == 0) & np.isfinite(op.travelled)
                        & (op.travelled > 0) & (op.seq_start >= 0) & (op.seq_end > op.seq_start+1))
        for band in range(len(pools)):
            if band >= 5:
                member = kinds == band-4
            elif band == 4:
                member = off & (kinds == 0)
            else:
                lo, hi = DRIFT_BANDS[band]
                member = ~off & (kinds == 0) & (op.drift>=lo) & (op.drift<hi)
            member &= eligible
            for fi in np.unique(op.fiber_idx[member]):
                idx = np.flatnonzero(member & (op.fiber_idx==fi))
                pools[band].setdefault(int(fi),[]).append((op,idx))
    return pools


class FollowDataset(torch.utils.data.IterableDataset):
    """Configurable fresh share; remaining draws use rotating recent replay.

    Defaults to 70% fresh and 30% recent replay after dedicated builder budgets.
    Legacy replay reserves 10% for departure. Builders may instead reserve a
    failure budget, uniform over available kinds, then fibers. Remaining draws
    use drift bands and then fibers. Empty legacy strata use fresh
    augmentation. Every draw is relabeled and holdout checked before crop I/O.
    clean_fraction reserves unperturbed GT separately and normalizes the existing
    hard-source weights into the remainder. Generic/legacy callers keep None.
    """
    def __init__(self, fibers, vol_spec, cfg, exclude_band, chunk=2, seed=0,
                 cache_bytes=1 << 30, onpolicy=None,
                 window=256., pool_size=12, window_samples=192, replay_index=None, refresh_chunks=8,
                 batch_builder=None, additional_crops=(), fresh_fraction=.7, clean_fraction=None,
                 correct_replay_only=False, replay_continuation_fraction=None,
                 gt_perturb_probability=0., gt_perturb_max_offset=.5, gt_perturb_max_angle_deg=2.,
                 prefer_real_wrong_turns=False, prefer_replay_for_light_gt=False):
        if not np.isfinite(fresh_fraction) or not 0 <= fresh_fraction <= 1:
            raise ValueError('Fresh fraction must be finite and in [0, 1]')
        self.fresh_fraction = float(fresh_fraction)
        if clean_fraction is not None and (not np.isfinite(clean_fraction) or not 0 <= clean_fraction <= 1):
            raise ValueError('Clean fraction must be finite and in [0, 1]')
        self.clean_fraction = clean_fraction
        self.correct_replay_only = correct_replay_only
        self.prefer_real_wrong_turns = prefer_real_wrong_turns
        self.prefer_replay_for_light_gt = prefer_replay_for_light_gt
        if replay_continuation_fraction is not None:
            if not np.isfinite(replay_continuation_fraction) or not 0 <= replay_continuation_fraction <= 1:
                raise ValueError('Replay continuation fraction must be finite and in [0, 1]')
            if correct_replay_only:
                raise ValueError('Choose correct-only replay or a continuation/failure mix')
        if not np.isfinite(gt_perturb_probability) or not 0 <= gt_perturb_probability <= 1:
            raise ValueError('GT perturb probability must be finite and in [0, 1]')
        if any(not np.isfinite(v) or v < 0 for v in (gt_perturb_max_offset, gt_perturb_max_angle_deg)):
            raise ValueError('GT perturb limits must be finite and nonnegative')
        if gt_perturb_probability and clean_fraction is None:
            raise ValueError('Light GT perturbation requires a reserved GT fraction')
        self.replay_continuation_fraction = replay_continuation_fraction
        self.gt_perturb_probability = gt_perturb_probability
        self.gt_perturbation = (gt_perturb_max_offset, gt_perturb_max_angle_deg)
        self.fibers, self.vol_spec, self.cfg, self.exclude = fibers, vol_spec, cfg, exclude_band
        self.chunk, self.seed, self.cache_bytes = chunk, seed, cache_bytes
        self.window, self.pool_size, self.window_samples = window, pool_size, window_samples
        self.replay_index, self.refresh_chunks = replay_index, refresh_chunks
        self.batch_builder = batch_builder
        self.remote_prefetch = None  # Optional trainer-owned process queue client.
        self.remote_prefetch_lookahead = 0  # Planned microbatches per source/worker.
        self.additional_crops = tuple(additional_crops)
        self._replay_paths = None
        self._set_replay(list(onpolicy or []))
        lengths = (fibers.lengths if hasattr(fibers, 'lengths')
                   else np.array([f.length for f in fibers]))
        if not len(lengths):
            raise ValueError('No training fibers')
        self.weights = lengths / lengths.sum()

    def _validate(self,caches):
        for op in caches:
            op.validate_fibers(self.fibers)
            if op.hist.shape[1] != self.cfg.n_history:
                raise ValueError('Replay history length must equal n_history')
            if op.provenance.get('volume',{}).get('grid_scale',8.) != self.vol_spec.grid_scale:
                raise ValueError('Replay world coordinate scale differs from this run')

    def _set_replay(self,caches):
        self._validate(caches)
        self.onpolicy = caches
        self.replay_failure_fraction = getattr(getattr(self.batch_builder, 'sampling', None),
                                              'replay_failure_fraction', None)
        self.recent_pools = replay_pools(caches, failures=(self.replay_failure_fraction is not None
                                                        or self.replay_continuation_fraction is not None),
                                        correct_only=self.correct_replay_only)
        self.continuation_pools = (replay_pools(caches, correct_only=True)
                                   if self.replay_continuation_fraction is not None or self.prefer_replay_for_light_gt else None)
        self.wrong_turn_pools = (replay_pools(caches, failures=True, natural_switch_only=True)
                                if self.prefer_real_wrong_turns else None)
        if hasattr(self.batch_builder, 'set_replay'):
            self.batch_builder.set_replay(caches)

    def draw_replay(self,rng, *, force=False):
        source = REPLAY_SOURCES['recent'] if force else int(rng.choice((0, REPLAY_SOURCES['recent']),
                                p=(self.fresh_fraction, 1-self.fresh_fraction)))
        pools = self.recent_pools
        if self.replay_continuation_fraction is not None:
            continuation = rng.random() < self.replay_continuation_fraction
            pools = self.continuation_pools if continuation else self.recent_pools
            options = [i for i in (range(4) if continuation else range(4, len(pools))) if pools[i]]
            if not options and not continuation:
                pools = self.continuation_pools
                options = [i for i in range(4) if pools[i]]
            # Never fill a missing correct-prefix slot with a failed trace.
            if not options:
                return None
            band = int(rng.choice(options))
        elif self.replay_failure_fraction is None:
            band = 4 if rng.random()<.1 else int(rng.integers(4))
        else:
            failures = [i for i in range(4, len(self.recent_pools)) if self.recent_pools[i]]
            drift = [i for i in range(4) if self.recent_pools[i]]
            options = failures if rng.random() < self.replay_failure_fraction else drift
            options = options or drift or failures
            band = int(rng.choice(options)) if options else 0
        if source == 0:
            return None
        pool = pools[band]
        if not pool:
            return None
        return self.draw_pool(pool, source, band, rng)

    @staticmethod
    def draw_pool(pool, source, band, rng):
        fi = rng.choice(sorted(pool))
        entries = pool[fi]
        sizes = np.array([len(idx) for _,idx in entries])
        op,idx = entries[rng.choice(len(entries),p=sizes/sizes.sum())]
        return source,band,op,int(rng.choice(idx))

    def wrong_turn_item(self, rng):
        """Use a real natural switch first; synthetic history is the fallback."""
        if self.wrong_turn_pools is not None:
            bands = [i for i in (5, 8) if self.wrong_turn_pools[i]]
            for _ in range(3 if bands else 0):
                band = int(rng.choice(bands))
                draw = self.draw_pool(self.wrong_turn_pools[band], 3, band, rng)
                item = self.replay_item(draw, rng)
                if item is not None:
                    item['real_wrong_turn'] = True
                    item['real_wrong_turn_pre_switch'] = item['failure_kind'] == 4
                    return item
        item = self.batch_builder.memory_switch(self.cfg, rng)
        if item is not None:
            item = self.prepare(item, rng)
            if self.state_allowed(item):
                return item
        return None

    def correct_continuation_item(self, rng):
        bands = [i for i in range(4) if self.continuation_pools[i]]
        for _ in range(3 if bands else 0):
            band = int(rng.choice(bands))
            draw = self.draw_pool(self.continuation_pools[band], REPLAY_SOURCES['recent'], band, rng)
            item = self.replay_item(draw, rng)
            if item is not None:
                item['replay_correct_continuation'] = True
                item['light_gt_replay'] = True
                return item
        return None

    def endpoint_requests(self, rng):
        """Reserve paired decisions and bank following before annotation/replay.

        Failed proposals return to annotation/replay, not another bank budget.
        Pair allocation stays even; expected endpoint shares are unconditional.
        """
        sampling = getattr(self.batch_builder, 'sampling', None)
        decisions = getattr(sampling, 'decision_fraction', 0.)
        following = getattr(sampling, 'bank_following_probability', 0.)
        if not (0 <= decisions <= 1 and 0 <= following <= 1 and decisions+following <= 1):
            raise ValueError('Decision and bank-following endpoint fractions must sum to at most one')
        if decisions and self.chunk % 2:
            raise ValueError('Matched identity decisions require an even batch')
        pairs = int(rng.binomial(self.chunk//2, decisions)) if decisions else 0
        remaining = self.chunk-2*pairs
        bank = int(rng.binomial(remaining, min(1., following/(1-decisions)))) if following and remaining else 0
        return pairs, bank

    def sampling_probabilities(self):
        """Unconditional budgets; unavailable hard examples fall back to clean GT."""
        s = getattr(self.batch_builder, 'sampling', None)
        decision = getattr(s, 'decision_fraction', 0.)
        following = getattr(s, 'bank_following_probability', 0.)
        remaining = 1-decision-following
        weights = dict(decision=decision, bank_following=following,
                       memory_switch=remaining*self.fresh_fraction*getattr(s, 'memory_switch_probability', 0.),
                       recent=remaining*(1-self.fresh_fraction))
        if self.clean_fraction is None:
            return dict(decision=decision, bank_following=following,
                        fresh=remaining*self.fresh_fraction, recent=weights['recent'])
        total = sum(weights.values())
        if total <= 0:
            return dict(clean=1., **dict.fromkeys(weights, 0.))
        return dict(clean=self.clean_fraction,
                    **{k: (1-self.clean_fraction)*v/total for k,v in weights.items()})

    def clean_requests(self, rng):
        # Allocate in pairs so identity pairs cannot consume the clean budget.
        # Counts vary by batch; expected clean share is exactly clean_fraction.
        if self.chunk % 2:
            raise ValueError('Clean/hard allocation requires an even batch')
        probabilities = self.sampling_probabilities()
        counts = rng.multinomial(self.chunk//2, list(probabilities.values()))*2
        return dict(zip(probabilities, counts.tolist()))

    def refresh_replay(self):
        if self.replay_index is None or not os.path.exists(self.replay_index):
            return
        with open(self.replay_index) as fh:
            paths = json.load(fh)
        if paths != self._replay_paths:
            self._set_replay([OnPolicyStates.load(path) for path in paths])
            self._replay_paths = paths

    def prepare(self, item, rng):
        """Let the batch builder attach geometry it reads later (e.g. path patches)."""
        if hasattr(self.batch_builder, 'prepare'):
            fiber = item.get('supervision_fiber')
            if fiber is None:
                fiber = self.fibers[item['fiber_ref'][0]]
            return self.batch_builder.prepare(item, fiber, rng)
        return item

    def state_allowed(self, item):
        allowed = (all(training_state_allowed(item, crop, self.exclude)
                    for crop in (self.cfg.crop, *self.additional_crops))
                and (not hasattr(self.batch_builder, 'footprint_allowed')
                     or self.batch_builder.footprint_allowed(item, self.exclude)))
        if allowed and self.exclude is not None:
            from types import SimpleNamespace
            scale = self.vol_spec.grid_scale/self.vol_spec.ct_grid_scale
            for start,size in self.prefetch_bounds([item],SimpleNamespace(input_scale=scale)):
                if start[0]/scale < self.exclude.hi and (start[0]+size[0])/scale > self.exclude.lo:
                    return False
        return allowed

    def replay_item(self, draw, rng):
        source,band,op,j = draw
        if (source == REPLAY_SOURCES['recent'] and not self.correct_replay_only
                and not self.prefer_replay_for_light_gt and self.replay_continuation_fraction is None
                and hasattr(self.batch_builder,'replace_replay')):
            replacement = self.batch_builder.replace_replay(source,band,self.cfg,rng)
            if replacement is not None:
                replacement = self.prepare(replacement,rng)
                if self.state_allowed(replacement):
                    return replacement
        # A missing/unsafe bank proposal preserves the original replay draw.
        fi,t,reverse = int(op.fiber_idx[j]),float(op.t[j]),bool(op.reverse[j])
        item = label_state(self.fibers[fi],op.pos[j],op.frame[j],op.hist[j],op.hmask[j],self.cfg,
                           t=t,reverse=reverse,offtrack=bool(op.offtrack[j]))
        item.update(source=source,stratum=band,source_step=op.provenance.get('step',-1) or -1,
                    failure_kind=int(op.failure_kind[j]) if hasattr(op, 'failure_kind') else 0,
                    fiber_ref=(fi,self.fibers[fi].length-t if reverse else t,reverse))
        item.update({k: getattr(op, k)[j] for k in SEED_FIELDS if hasattr(op, k)})
        item['observed_path'] = op.observed_prefix(j)
        item['replay_correct_continuation'] = bool(
            (self.correct_replay_only or self.replay_continuation_fraction is not None) and band < 4)
        from .heading import FRAME_POLICY
        item['frame_policy'] = op.provenance.get('frame_policy', FRAME_POLICY)
        item = self.prepare(item,rng)
        return item if self.state_allowed(item) else None

    def prepare_pair(self, pair, rng):
        prepared = [self.prepare(item, rng) for item in pair]
        return prepared if all(self.state_allowed(item) for item in prepared) else []

    def prefetch_bounds(self, items, vol):
        if hasattr(self.batch_builder, 'prefetch_bounds'):
            return [bound for item in items for bound in self.batch_builder.prefetch_bounds(item,vol)]
        from .heading import frame_prefetch_bounds
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

    def __iter__(self):
        from .heading import SeedHeadingError
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
        plans, pending = self._iter_plans(vol), deque()
        first = True
        rejected = 0
        while True:
            # Deliver the first batch promptly. Fill the deeper plan queue when
            # the loader asks for its next batch, overlapping the first update.
            while len(pending) < (1 if first else lookahead+1):
                plan = next(plans)
                bounds = self.prefetch_bounds(plan[0],vol) if lookahead else None
                pending.append((*plan,bounds))
            first = False
            if lookahead:
                self.remote_prefetch.lookahead(vol.ct,
                    [bounds for *_,bounds in pending],scope=id(self))
            items, requested, fraction, bounds = pending.popleft()
            if lookahead:
                self.remote_prefetch.ensure(vol.ct,bounds)
            else:
                self.prefetch_items(items,vol,required=True)
            try:
                batch = (self.batch_builder(items, vol) if self.batch_builder is not None
                         else collate_with_volume(items, vol, self.cfg.crop, grid))
            except SeedHeadingError as exc:
                # Invalid geometry/context still rejects the whole plan to
                # preserve pairs. Weak CT orientation now falls back in ct_frame
                # and never reaches this retry path. I/O errors propagate.
                rejected += 1
                if rejected >= MAX_CT_FRAME_REJECTIONS:
                    worker = torch.utils.data.get_worker_info()
                    raise SeedHeadingError(
                        f'Could not build a CT-oriented training batch after {rejected} consecutive rejected plans '
                        f'(source={getattr(self.vol_spec, "ct_zarr", "unknown")}, '
                        f'worker={worker.id if worker else 0}): {exc}') from exc
                continue
            if fraction:
                batch['decision_requested'] = torch.full((self.chunk,), 2*requested/self.chunk)
            if rejected:
                # A row-aligned counter survives the usual tensor batch movers;
                # put the total in one row so summing never multiplies it by B.
                batch['ct_frame_rejected_batches'] = torch.zeros(self.chunk, dtype=torch.int64)
                batch['ct_frame_rejected_batches'][0] = rejected
                rejected = 0
            yield batch

    def _iter_plans(self, vol):
        """Ordered geometry/augmentation plans, without CT or dense image tensors."""
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed*1000 + (0 if info is None else info.id))
        cfg = self.cfg
        windows = []
        chunks = 0
        while True:
            if chunks % self.refresh_chunks == 0:
                self.refresh_replay()
            chunks += 1
            items = []
            fraction = getattr(getattr(self.batch_builder, 'sampling', None), 'decision_fraction', 0.)
            requests = self.clean_requests(rng) if self.clean_fraction is not None else None
            requested, following = ((requests['decision']//2, requests['bank_following']) if requests is not None
                                    else self.endpoint_requests(rng))
            if requests is not None:
                fraction = self.sampling_probabilities()['decision']
            for _ in range(requested):
                pair = self.batch_builder.decision_pair(cfg, rng)
                if pair is not None:
                    prepared = self.prepare_pair(pair, rng)
                    self.prefetch_items(prepared,vol)
                    items.extend(prepared)
            for _ in range(following):
                item = self.batch_builder.bank_following(cfg, rng)
                if item is not None:
                    item = self.prepare(item, rng)
                    if self.state_allowed(item):
                        self.prefetch_items([item],vol)
                        items.append(item)
            if requests is not None:
                for _ in range(requests['memory_switch']):
                    item = self.wrong_turn_item(rng)
                    if item is not None:
                        self.prefetch_items([item], vol)
                        items.append(item)
                for _ in range(requests['recent']):
                    draw = self.draw_replay(rng, force=True)
                    item = self.replay_item(draw, rng) if draw is not None else None
                    if item is not None:
                        self.prefetch_items([item], vol)
                        items.append(item)
            for attempt in range(max(10000, self.chunk*1000)):
                if len(items) == self.chunk:
                    break
                draw = self.draw_replay(rng) if requests is None and attempt < self.chunk*10 else None
                item = None
                if draw is not None:
                    item = self.replay_item(draw,rng)
                if item is None and requests is None and hasattr(self.batch_builder, 'replace_fresh'):
                    item = self.batch_builder.replace_fresh(cfg,rng)
                    if item is not None:
                        item = self.prepare(item,rng)
                if item is None:
                    # A builder may oversample chosen locations; otherwise draw windows.
                    location = (self.batch_builder.fresh_location(rng)
                                if hasattr(self.batch_builder, 'fresh_location') else None)
                    if location is None:
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
                            t = max(0, f.length-rng.uniform(0, cfg.future_s[-1]*1.5))
                        else:
                            original_t = np.clip(center+rng.uniform(-self.window/2, self.window/2), 0, f.length)
                            t = f.length-original_t if rev else original_t
                    else:
                        fi, t, rev = location['fiber'], location['t'], location['reverse']
                    sample_options = {'perturb': False} if requests is not None else {}
                    gt_perturbed = bool(requests is not None and self.gt_perturb_probability
                                        and rng.random() < self.gt_perturb_probability)
                    if gt_perturbed and self.prefer_replay_for_light_gt:
                        item = self.correct_continuation_item(rng)
                    if item is None:
                        if gt_perturbed:
                            sample_options['light_perturbation'] = self.gt_perturbation
                        item = make_sample(self.fibers[fi], t, rev, cfg, rng, **sample_options)
                        item['gt_perturbed'] = gt_perturbed
                        item['gt_unperturbed'] = requests is not None and not gt_perturbed
                        item['source'], item['source_step'], item['stratum'] = 0, -1, -1
                        item['fiber_ref'] = (int(fi), float(t), bool(rev))
                        item['location_source'] = location['source'] if location else 0
                        item = self.prepare(item, rng)
                if self.state_allowed(item):
                    self.prefetch_items([item],vol)
                    items.append(item)
                if len(items) == self.chunk:
                    break
            if len(items) != self.chunk:
                raise ValueError('Could not fill a training batch outside the held-out band')
            if (self.remote_prefetch is not None and self.remote_prefetch_lookahead
                    and hasattr(self.batch_builder,'prepare_sampling_feedback')):
                self.batch_builder.prepare_sampling_feedback(items)
            yield items, requested, fraction


FUSED_SAMPLER = os.environ.get("FIBER_FOLLOW_FUSED", "1") != "0"
_GRID_CACHE: dict = {}


def _grid_flat(crop: CropSpec) -> np.ndarray:
    g = _GRID_CACHE.get(crop)
    if g is None:
        g = _GRID_CACHE[crop] = crop_local_grid(crop).reshape(-1, 3).astype(np.float64)
    return g


def read_blocks(items, vol: FiberVolume, crop: CropSpec, pool=None, *, presence=False):
    scale = 1. if presence else getattr(vol, 'input_scale', 1.)
    S = int(np.ceil(crop.block_size*scale))
    starts = np.floor(np.stack([block_start(it["pos"], it["frame"], crop) for it in items])*scale).astype(np.int64)
    read = lambda st: vol.presence.read(st, (S, S, S))[None] if presence else vol.raw_block(st, (S, S, S))
    raw = np.stack(list(map(read, starts) if pool is None else pool.map(read, starts)))
    return raw, starts


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


def read_tight_blocks(items, vol: FiberVolume, crop: CropSpec, pool=None, *, presence=False):
    """Per-item minimal blocks for per-sample CPU sampling: (list of raw, list of zyx starts).

    Same values as ``read_blocks`` at every sample point, reading roughly a
    third of the voxels and touching about half the chunks per item.
    """
    scale = 1. if presence else getattr(vol, 'input_scale', 1.)
    bounds = [tight_block(it["pos"], it["frame"], crop, scale) for it in items]
    read = lambda b: vol.presence.read(b[0], b[1])[None] if presence else vol.raw_block(b[0], b[1])
    raw = list(map(read, bounds) if pool is None else pool.map(read, bounds))
    return raw, [start for start, _ in bounds]


def add_presence_input(x, items, vol, crop, grid, pool=None):
    """Insert presence before history, sampling the fiber grid independently of CT.

    Shared by training and rollout. CT retains its native samples; the scalar
    presence prediction is interpolated directly at the same world locations.
    """
    raw, starts = read_blocks(items, vol, crop, pool, presence=True)
    tensor = lambda values: torch.as_tensor(np.asarray(values), device=x.device)
    presence = sample_oriented_fast(tensor(raw), tensor(starts).float(),
                                    tensor([it['pos'] for it in items]).float(),
                                    tensor([it['frame'] for it in items]).float(), grid.to(x.device))
    return torch.cat([x[:, :-1], presence.to(x.dtype), x[:, -1:]], 1)


def render_count(crop: CropSpec) -> int:
    """History points (1 voxel apart) that can land inside the crop."""
    return int(np.ceil(crop.behind * crop.spacing)) + 4


def build_inputs(raw, starts, pos, frames, hist, hmask, grid: torch.Tensor, n_render: int | None = None,
                 input_scale: float = 1., history_sigma: float = 1., history_render: str = 'points') -> torch.Tensor:
    """Tensors (any device) -> model input (B, C, D, H, W). Only the nearest
    ``n_render`` history points are rendered (the rest lie behind the crop)."""
    x = sample_oriented_fast(raw, starts.float(), pos*input_scale, frames*input_scale, grid)
    if n_render is not None:
        hist, hmask = hist[:, :n_render], hmask[:, :n_render]
    return torch.cat([x, render_history(hist, hmask, grid, history_sigma, history_render)], 1)


def collate_with_volume(items, vol: FiberVolume, crop: CropSpec, grid: torch.Tensor | None = None):
    """Worker-side batch. With ``grid`` the model input ``x`` (fp16) is built
    here on CPU; otherwise raw blocks + geometry are returned."""
    from .heading import orient_item
    for item in items:
        orient_item(item, vol)
    if grid is None and vol.spec.mode == 'ct+presence':
        grid = torch.from_numpy(crop_local_grid(crop)).float()
    raw, starts = read_blocks(items, vol, crop)
    scale = getattr(vol, 'input_scale', 1.)
    st = lambda k: torch.from_numpy(np.stack([it[k] for it in items]).astype(np.float32))
    out = dict(raw=torch.from_numpy(raw), starts=torch.from_numpy(starts), pos=st("pos"), frames=st("frame"),
               hist=st("hist_local"), hmask=st("hmask"))
    if grid is not None:
        if FUSED_SAMPLER:
            # one fused numba pass per sample (see fast_sample.py); same values as build_inputs
            nr = render_count(crop)
            gf = _grid_flat(crop)
            x = np.empty((len(items), 2, crop.depth, crop.width, crop.width), np.float16)
            for j, it in enumerate(items):
                x[j] = sample_crop(raw[j], starts[j], it["pos"]*scale, it["frame"]*scale, gf,
                                   it["hist_local"][:nr], it["hmask"][:nr], 2, crop.history_sigma, crop.history_render).reshape(x.shape[1:])
            out = dict(x=torch.from_numpy(x), hist=out["hist"], hmask=out["hmask"])
        else:
            x = build_inputs(out.pop("raw"), out.pop("starts"), out["pos"], out["frames"], out["hist"], out["hmask"],
                             grid, n_render=render_count(crop), input_scale=scale,
                             history_sigma=crop.history_sigma, history_render=crop.history_render)
            out = dict(x=x.half(), hist=out["hist"], hmask=out["hmask"])
        if vol.spec.mode == 'ct+presence':
            out['x'] = add_presence_input(out['x'], items, vol, crop, grid)
    out.update(collate_targets(items))
    return out


def collate_targets(items):
    """Pack annotation-only tensors, independently of a model's image inputs."""
    st = lambda k: torch.from_numpy(np.stack([it[k] for it in items]).astype(np.float32))
    out = {}
    if "fut_local" in items[0]:
        out["fut"] = st("fut_local")
        out["fmask"] = st("fmask")
        out["gt_history"] = st("gt_history")
        out["gt_history_mask"] = st("gt_history_mask")
        out["plane_ab"] = st("plane_ab")
        out["plane_mask"] = st("plane_mask")
        for key in ("dense_ab", "dense_mask", "end_local", "endpoint_known", "offtrack"):
            out[key] = st(key)
    if 'source' in items[0]:
        out['source'] = st('source')
        out['source_step'] = st('source_step')
        out['stratum'] = st('stratum')
        out['failure_kind'] = torch.tensor([it.get('failure_kind', 0) for it in items], dtype=torch.long)
        for key in ('gt_unperturbed', 'gt_perturbed', 'replay_correct_continuation',
                    'real_wrong_turn', 'real_wrong_turn_pre_switch', 'light_gt_replay'):
            out[key] = torch.tensor([it.get(key, False) for it in items], dtype=torch.bool)
    return out


# ------------------------------------------------------------ on-policy states


def label_state(fiber, pos, frame, hist_world, hmask, cfg, *, t, reverse, offtrack=False):
    """Relabel the exact inference frame/history against its original fiber."""
    if len(hist_world) != cfg.n_history or len(hmask) != cfg.n_history:
        raise ValueError('Replay history must equal n_history')
    traversal_t = fiber.length-t if reverse else t
    return dict(pos=pos, frame=frame, hist_local=(hist_world-pos) @ frame,
                hmask=np.asarray(hmask, np.float32),
                **continuation_targets(fiber, traversal_t, reverse, pos, frame, cfg, offtrack))


STATE_VERSION = 8

class OnPolicyStates:
    """States (pos, heading, own-trace history) visited by a tracer on GT fibers.

    Loaded as memory-mapped .npy files so DataLoader workers share one copy
    (pickling sends only the directory path)."""

    FIELDS = ("fiber_idx", "t", "reverse", "pos", "frame", "hist", "hmask", "offtrack",
              "hard", "exploratory")
    # Fields later collectors add; caches without them load with the default.
    # drift: current-position error in trace-grid voxels (NaN when departed or unknown).
    # failure_kind indexes REPLAY_FAILURES. switch_* identifies the first certified
    # foreign contact, including on retained pre-switch rows. None of it is input.
    # Arc positions compared against float64 trace geometry keep full precision.
    FLOAT64_FIELDS = ("t", "travelled", "switch_distance", "switch_pos", "pos", "frame", "hist", "seed_pos", "seed_tangent", "track_pos")
    OPTIONAL = {"drift": lambda n: np.full(n, np.nan, np.float32),
                "source_cache": lambda n: np.full(n, -1, np.int32),
                "source_row": lambda n: np.full(n, -1, np.int64),
                "failure_kind": lambda n: np.zeros(n, np.int8),
                "travelled": lambda n: np.full(n, np.nan, np.float64),
                # First trusted vertex within this trace's observed prefix.
                "heading_start": lambda n: np.zeros(n, np.int64),
                "switch_distance": lambda n: np.full(n, np.nan, np.float64),
                "switch_pos": lambda n: np.full((n,3), np.nan, np.float64),
                "switch_decision": lambda n: np.full(n, -1, np.int64),
                "switch_bank_path": lambda n: np.full(n, '', dtype='U1'),
                "switch_bank_run": lambda n: np.full(n, '', dtype='U64'), **SEED_DEFAULTS}
    # Each trace is stored once. The exclusive end includes the decision head.
    ROW_TRACK = {"seq_start": lambda n: np.full(n, -1, np.int64),
                 "seq_end": lambda n: np.full(n, -1, np.int64)}
    TRACK = {"track_pos": lambda n: np.zeros((n, 3), np.float64)}

    def __init__(self, *, manifest, provenance=None, **arrays):
        for key in self.FIELDS:
            setattr(self, key, np.asarray(arrays[key]))
        for key, default in self.OPTIONAL.items():
            setattr(self, key, np.asarray(arrays[key]) if key in arrays else default(len(self.pos)))
        for key, default in self.ROW_TRACK.items():
            setattr(self, key, np.asarray(arrays[key]) if key in arrays else default(len(self.pos)))
        for key, default in self.TRACK.items():
            setattr(self, key, np.asarray(arrays[key]) if key in arrays else default(0))
        if any(len(getattr(self, k)) != len(self.pos) for k in self.FIELDS + tuple(self.OPTIONAL) + tuple(self.ROW_TRACK)):
            raise ValueError('Replay arrays have different lengths')
        if any(len(getattr(self, k)) != len(self.track_pos) for k in self.TRACK):
            raise ValueError('Replay track arrays have different lengths')
        present = self.seq_end >= 0
        if np.any(present & ((self.seq_start < 0) | (self.seq_start > self.seq_end) | (self.seq_end > len(self.track_pos)))):
            raise ValueError('Replay rows reference tracks outside the saved tracks')
        self._dir = None
        self.manifest = manifest
        self.provenance = provenance or {}

    def __len__(self):
        return len(self.pos)

    def observed_prefix(self, j):
        """Actual committed polyline through this decision; no future vertices."""
        start, end = int(self.seq_start[j]), int(self.seq_end[j])
        if not 0 <= start < end <= len(self.track_pos):
            raise ValueError('Replay needs complete observed prefixes; recollect with replay v8')
        path = self.track_pos[start:end]
        if not np.allclose(path[-1], self.pos[j], atol=1e-5, rtol=0):
            raise ValueError('Replay prefix does not end at decision head')
        if self.seed_valid[j] and not np.allclose(path[0], self.seed_pos[j], atol=1e-5, rtol=0):
            raise ValueError('Replay prefix does not start at its original seed')
        return path

    def save(self, path):
        np.savez(path, __metadata__=json.dumps(dict(version=STATE_VERSION, fibers=self.manifest, provenance=self.provenance)),
                 **{k: getattr(self, k) for k in self.FIELDS + tuple(self.OPTIONAL) + tuple(self.ROW_TRACK) + tuple(self.TRACK)})

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
        """``path``: .npz from collect.py (converted once to a sibling ``_mmap/``
        dir of .npy files) or such a directory."""
        path = os.fspath(path)
        d = path[:-4] + "_mmap_v8" if path.endswith(".npz") else path
        metadata_path = os.path.join(d, "metadata.json")
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as z:
                metadata = json.loads(str(z["__metadata__"].item()))
                if metadata["version"] != STATE_VERSION:
                    raise ValueError(f"Incompatible replay version {metadata['version']}; expected {STATE_VERSION}")
                # Include archive identity so an overwritten NPZ cannot reuse stale mmap arrays.
                stat = os.stat(path)
                metadata["archive"] = [stat.st_size, stat.st_mtime_ns]
                # Recorded so mirrors written with other precisions are rebuilt.
                metadata["float64_fields"] = list(cls.FLOAT64_FIELDS)
                existing = None
                if os.path.exists(metadata_path):
                    with open(metadata_path) as fh:
                        existing = json.load(fh)
                # Track arrays are mirrored only when the archive has them.
                tracks = tuple(k for k in (*cls.ROW_TRACK, *cls.TRACK) if k in z.files)
                if existing != metadata or any(not os.path.exists(os.path.join(d, k + ".npy"))
                                               for k in cls.FIELDS + tuple(cls.OPTIONAL) + tracks):
                    os.makedirs(d, exist_ok=True)
                    for k in cls.FIELDS + tuple(cls.OPTIONAL) + tracks:
                        v = z[k] if k in z.files else cls.OPTIONAL[k](len(z["pos"]))
                        if k not in cls.FLOAT64_FIELDS and v.dtype == np.float64:
                            v = v.astype(np.float32)
                        np.save(os.path.join(d, k + ".npy"), v)
                    with open(metadata_path, "w") as fh:
                        json.dump(metadata, fh)
        obj = cls.__new__(cls)
        obj._open(d)
        return obj

    def _open(self, d):
        metadata_path = os.path.join(d, "metadata.json")
        with open(metadata_path) as fh:
            metadata = json.load(fh)
        if metadata["version"] != STATE_VERSION:
            raise ValueError(f"Incompatible replay version {metadata['version']}; expected {STATE_VERSION}")
        self.manifest = metadata["fibers"]
        self.provenance = metadata["provenance"]
        self._dir = d
        for k in self.FIELDS:
            setattr(self, k, np.load(os.path.join(d, k + ".npy"), mmap_mode="r"))
        for k, default in self.OPTIONAL.items():
            file = os.path.join(d, k + ".npy")
            setattr(self, k, np.load(file, mmap_mode="r") if os.path.exists(file) else default(len(self.pos)))
        for k, default in self.ROW_TRACK.items():
            file = os.path.join(d, k + ".npy")
            setattr(self, k, np.load(file, mmap_mode="r") if os.path.exists(file) else default(len(self.pos)))
        for k, default in self.TRACK.items():
            file = os.path.join(d, k + ".npy")
            setattr(self, k, np.load(file, mmap_mode="r") if os.path.exists(file) else default(0))

    def __getstate__(self):
        if self._dir is None:
            return self.__dict__
        return {"_dir": self._dir}

    def __setstate__(self, state):
        if state.get("_dir") and len(state) == 1:
            self._open(state["_dir"])
        else:
            self.__dict__.update(state)
