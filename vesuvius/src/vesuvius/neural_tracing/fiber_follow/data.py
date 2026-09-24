"""Training samples for the autoregressive fiber follower."""

from __future__ import annotations

import glob
import hashlib
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial import cKDTree

import json

from vc3d_fiber_format import parse_vc3d_fiber_format, FiberTraceSegmentMetadata
from vesuvius.neural_tracing.fiber_follow.geometry import (
    CropSpec,
    arclength,
    block_start,
    crop_local_grid,
    frame_from_heading,
    interp_at,
    normalize,
    random_rotation_about,
    render_history,
    sample_oriented_fast,
    tangent_at,
)
from vesuvius.neural_tracing.fiber_follow.fast_sample import sample_crop
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume, FiberVolumeSpec


DATA_VERSION = 2
DATA_POLICY = "controlled_spans_v2"


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
    brk: np.ndarray | None = None  # (N,) bool: point lies in a low-presence break
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
    return [dict(name=f.name, source_hash=f.source_hash,
                 geometry_hash=hashlib.sha256(np.asarray(f.points, dtype="<f8").tobytes()).hexdigest(),
                 endpoint_stop=list(f.endpoint_stop)) for f in fibers]


def _presence_identity(fibers, vol):
    spec = getattr(vol, "spec", None)
    source = spec.to_dict() if spec is not None else {}
    return json.dumps(dict(version=DATA_VERSION, fibers=fiber_manifest(fibers), volume=source), sort_keys=True)


def gt_presence(fibers: list[TracedFiber], vol: FiberVolume, cache_path: str | None = None) -> dict:
    """Max presence (0..1) in the 3x3x3 neighbourhood of every GT point."""
    cached = {}
    identity = _presence_identity(fibers, vol)
    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=False) as z:
            if "__identity__" in z.files and str(z["__identity__"].item()) == identity:
                cached = {k: z[k] for k in z.files if k != "__identity__"}
    off = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3)
    out, new = {}, False
    for f in fibers:
        p = cached.get(f.name)
        if p is None or len(p) != len(f.points):
            q = f.points[:, None, ::-1] + off[None]
            p = vol.presence.sample_nearest(q).max(1).astype(np.float32) / 255.0
            new = True
        out[f.name] = p
    if cache_path and new:
        np.savez_compressed(cache_path, __identity__=identity, **out)
    return out


def mark_breaks(fibers: list[TracedFiber], presence: dict, thr: float = 0.1, min_len: int = 8) -> None:
    """Flag low-prediction-support stretches, not confirmed physical breaks."""
    for f in fibers:
        low = presence[f.name] < thr
        b = np.concatenate([[0], low.astype(np.int8), [0]])
        d = np.diff(b)
        brk = np.zeros(len(low), bool)
        for a, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            # CP anchors and final samples can be closer than one grid voxel.
            end_s = f.s[e] if e < len(f.s) else f.length
            if end_s - f.s[a] >= min_len:
                brk[a:e] = True
        f.brk = brk


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
    crop: CropSpec = field(default_factory=CropSpec)
    n_candidates: int = 4
    n_future: int = 16
    future_step: float = 2.0
    n_history: int = 128
    history_step: float = 1.0
    lateral_sigmas: tuple = (0.4, 1.0, 2.0)
    lateral_probs: tuple = (0.5, 0.35, 0.15)
    angle_sigmas_deg: tuple = (4.0, 10.0, 20.0)
    angle_probs: tuple = (0.5, 0.35, 0.15)
    no_history_prob: float = 0.1
    history_jitter: float = 0.25  # independent point noise; disable for smooth history
    history_wobble: float = 0.0  # max amplitude (voxels) of slow lateral wobble on the own-trace history
    dense_substeps: int = 4
    heatmap_target: str = 'planes'
    tube_sigma: float = .7

    @property
    def future_s(self) -> np.ndarray:
        return self.future_step * np.arange(1, self.n_future + 1)


def _mix(rng, sigmas, probs):
    return float(rng.choice(sigmas, p=probs))


def training_state_allowed(item, crop: CropSpec, band: ZBand | None):
    """Exclude the perturbed state, entire sampled block, and used history/labels.

    Keep the original 48-voxel position guard and also check the actual rotated
    crop's read footprint (including interpolation support). Applied before I/O
    to fresh and cached states, after their random roll/offset is selected.
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
    for key, mask in (("hist_local", "hmask"), ("fut_local", "fmask")):
        if key in item:
            points = item[key][item[mask] > 0]
            if len(points):
                zs.append((points @ frame.T + pos)[:, 2])
    if "plane_ab" in item:
        # The heatmap labels can look farther along GT than arclength targets.
        loc = np.c_[item["plane_ab"], item["planes"]]
        loc = loc[item["plane_mask"] > 0]
        if len(loc):
            zs.append((loc @ frame.T + pos)[:, 2])
    if "dense_ab" in item:
        loc = np.c_[item["dense_ab"], item["dense_planes"]][item["dense_mask"] > 0]
        if len(loc):
            zs.append((loc @ frame.T + pos)[:, 2])
    if 'tube_segments' in item and len(item['tube_segments']):
        zs.append((item['tube_segments'].reshape(-1, 3) @ frame.T + pos)[:, 2])
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


def make_sample(fiber: TracedFiber, t: float, reverse: bool, cfg: SampleConfig, rng: np.random.Generator):
    """Geometry of one perturbed training state (no volume access)."""
    p, s = fiber.points, fiber.s
    if reverse:
        p = p[::-1]
        s = s[-1] - s[::-1]
    L = s[-1]
    g = interp_at(p, s, np.array([t]))[0]
    tau = tangent_at(p, s, t)
    # lateral offset perpendicular to tau
    fr = frame_from_heading(tau)
    lat = rng.normal(size=2) * _mix(rng, cfg.lateral_sigmas, cfg.lateral_probs)
    delta = fr[:, 0] * lat[0] + fr[:, 1] * lat[1]
    pos = g + delta
    ang = math.radians(_mix(rng, cfg.angle_sigmas_deg, cfg.angle_probs)) * rng.normal()
    axis_ang = rng.uniform(0, 2 * np.pi)
    ax = math.cos(axis_ang) * fr[:, 0] + math.sin(axis_ang) * fr[:, 1]
    heading = normalize(math.cos(ang) * tau + math.sin(ang) * ax)
    frame = random_rotation_about(frame_from_heading(heading), rng.uniform(0, 2 * np.pi))

    # history: GT points behind with the drift ramping in
    k = np.arange(1, cfg.n_history + 1) * cfg.history_step
    th = t - k
    hmask = (th >= 0).astype(np.float32)
    if rng.random() < cfg.no_history_prob:
        hmask[:] = 0
    else:
        # random truncated history (trace just started)
        if rng.random() < 0.2:
            hmask[int(rng.integers(0, cfg.n_history)):] = 0
    hist = interp_at(p, s, np.clip(th, 0, L))
    ramp = np.clip(1.0 - k / (cfg.n_history * cfg.history_step), 0, 1)[:, None]
    hist = hist + ramp * delta[None] + rng.normal(size=hist.shape) * cfg.history_jitter
    if cfg.history_wobble > 0:
        # slow lateral wander of our own past path, zero at the current point
        amp = rng.uniform(0, cfg.history_wobble, size=2)
        lam = rng.uniform(15.0, 45.0, size=2)
        ph = rng.uniform(0, 2 * np.pi, size=2)
        w = amp * (np.sin(2 * np.pi * k[:, None] / lam + ph) - np.sin(ph))
        hist = hist + w[:, :1] * fr[:, 0] + w[:, 1:] * fr[:, 1]

    return dict(pos=pos, frame=frame, hist_local=(hist-pos) @ frame, hmask=hmask,
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
    tube = {}
    if cfg.heatmap_target == 'tube':
        from vesuvius.neural_tracing.fiber_follow.tube import tube_geometry
        endpoints = fiber.endpoint_stop[::-1] if reverse else fiber.endpoint_stop
        tube = tube_geometry(p, pos, frame, cfg.crop, cfg.tube_sigma, endpoints, offtrack)
    return dict(**tube, fut_local=(fut-pos) @ frame, fmask=fmask,
                plane_ab=ab, plane_mask=mask, planes=cfg.future_s,
                dense_ab=dense_ab, dense_mask=dense_mask, dense_planes=dense_planes,
                end_local=end, endpoint_known=float(known), offtrack=float(offtrack),
                replay_candidates=np.zeros((cfg.n_candidates, cfg.n_future, 3), np.float32),
                replay_valid=np.zeros(cfg.n_candidates, np.float32))


class FollowDataset(torch.utils.data.IterableDataset):
    """Fresh perturbations plus recent-weighted exact policy decisions.

    Defaults: 50% fresh, 30% ordinary replay, 20% hard replay when pools exist.
    Missing pools fall back to fresh samples. Windows amortize volume reads.
    """
    def __init__(self, fibers, vol_spec, cfg, exclude_band, chunk=32, seed=0,
                 cache_bytes=1 << 30, onpolicy=None, onpolicy_prob=.3, hard_prob=.2,
                 window=256., pool_size=12, window_samples=192, replay_index=None, refresh_chunks=8):
        self.fibers, self.vol_spec, self.cfg, self.exclude = fibers, vol_spec, cfg, exclude_band
        self.chunk, self.seed, self.cache_bytes = chunk, seed, cache_bytes
        self.window, self.pool_size, self.window_samples = window, pool_size, window_samples
        self.replay_index, self.refresh_chunks = replay_index, refresh_chunks
        self._replay_paths = None
        self.onpolicy = list(onpolicy or [])  # oldest -> newest
        if onpolicy_prob < 0 or hard_prob < 0 or onpolicy_prob+hard_prob > 1:
            raise ValueError('Replay fractions must be nonnegative and sum to at most one')
        self.onpolicy_prob, self.hard_prob = onpolicy_prob, hard_prob
        self._set_replay(self.onpolicy)
        lengths = np.array([f.length for f in fibers])
        if not len(lengths):
            raise ValueError('No training fibers')
        self.weights = lengths / lengths.sum()

    def _set_replay(self, caches):
        self.pools = {False: [], True: []}
        for age, op in enumerate(caches):
            op.validate_fibers(self.fibers)
            if op.hist.shape[1] != self.cfg.n_history:
                raise ValueError('Replay history length must equal n_history')
            for hard in (False, True):
                indices = np.flatnonzero(op.hard == hard)
                if len(indices):
                    self.pools[hard].append((age+1, op, indices))
            provenance = op.provenance
            if provenance and (CropSpec(**provenance['crop']) != self.cfg.crop
                               or FiberVolumeSpec(**provenance['volume']) != self.vol_spec):
                raise ValueError('Replay crop or volume differs from the current run')
            if op.candidates.shape[1:] != (self.cfg.n_candidates, self.cfg.n_future, 3):
                raise ValueError('Replay candidate shape differs from the current run')
            if provenance and provenance['model_cfg']['future_step'] != self.cfg.future_step:
                raise ValueError('Replay forward plane spacing differs from the current run')

    def refresh_replay(self):
        if self.replay_index is None or not os.path.exists(self.replay_index):
            return
        with open(self.replay_index) as fh:
            paths = json.load(fh)
        if paths != self._replay_paths:
            caches = [OnPolicyStates.load(path) for path in paths]
            self._set_replay(caches)
            self._replay_paths = paths

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed*1000 + (0 if info is None else info.id))
        torch.set_num_threads(1)
        vol = FiberVolume(self.vol_spec, cache_bytes=self.cache_bytes)
        cfg = self.cfg
        grid = torch.from_numpy(crop_local_grid(cfg.crop)).float()
        windows = []
        chunks = 0
        while True:
            if chunks % self.refresh_chunks == 0:
                self.refresh_replay()
            chunks += 1
            items = []
            for attempt in range(max(10000, self.chunk*1000)):
                u = rng.random()
                hard = u < self.hard_prob
                replay = u < self.hard_prob+self.onpolicy_prob and attempt < self.chunk*10
                pool = self.pools[hard] if replay else []
                if pool:
                    weights = np.array([entry[0] for entry in pool], dtype=float)
                    _, op, idx = pool[rng.choice(len(pool), p=weights/weights.sum())]
                    j = rng.choice(idx)
                    item = label_state(self.fibers[op.fiber_idx[j]], op.pos[j], op.frame[j],
                                       op.hist[j], op.hmask[j], cfg, t=float(op.t[j]),
                                       reverse=bool(op.reverse[j]), offtrack=bool(op.offtrack[j]))
                    item['source'] = 2 if hard else 1
                    item['source_step'] = op.provenance.get('step', -1)
                    item['replay_candidates'] = op.candidates[j]
                    item['replay_valid'] = np.ones(cfg.n_candidates, np.float32)
                else:
                    while len(windows) < self.pool_size:
                        f = self.fibers[rng.choice(len(self.fibers), p=self.weights)]
                        windows.append([f, rng.uniform(0, f.length), self.window_samples])
                    wi = rng.integers(len(windows))
                    f, center, _ = windows[wi]
                    windows[wi][2] -= 1
                    if windows[wi][2] <= 0:
                        windows.pop(wi)
                    rev = bool(rng.integers(2))
                    if rng.random() < .1:
                        t = max(0, f.length-rng.uniform(0, cfg.future_s[-1]*1.5))
                    else:
                        original_t = np.clip(center+rng.uniform(-self.window/2, self.window/2), 0, f.length)
                        t = f.length-original_t if rev else original_t
                    item = make_sample(f, t, rev, cfg, rng)
                    item['source'], item['source_step'] = 0, -1
                if training_state_allowed(item, cfg.crop, self.exclude):
                    items.append(item)
                if len(items) == self.chunk:
                    break
            if len(items) != self.chunk:
                raise ValueError('Could not fill a training batch outside the held-out band')
            yield collate_with_volume(items, vol, cfg.crop, grid)


FUSED_SAMPLER = os.environ.get("FIBER_FOLLOW_FUSED", "1") != "0"
_GRID_CACHE: dict = {}


def _grid_flat(crop: CropSpec) -> np.ndarray:
    g = _GRID_CACHE.get(crop)
    if g is None:
        g = _GRID_CACHE[crop] = crop_local_grid(crop).reshape(-1, 3).astype(np.float64)
    return g


def read_blocks(items, vol: FiberVolume, crop: CropSpec, pool=None):
    scale = getattr(vol, 'input_scale', 1.)
    S = int(np.ceil(crop.block_size*scale))
    starts = np.floor(np.stack([block_start(it["pos"], it["frame"], crop) for it in items])*scale).astype(np.int64)
    read = lambda st: vol.raw_block(st, (S, S, S))
    raw = np.stack(list(map(read, starts) if pool is None else pool.map(read, starts)))
    return raw, starts


def render_count(crop: CropSpec) -> int:
    """History points (1 voxel apart) that can land inside the crop."""
    return int(np.ceil(crop.behind * crop.spacing)) + 4


def build_inputs(raw, starts, pos, frames, hist, hmask, grid: torch.Tensor, n_render: int | None = None,
                 gate_direction: bool = False, input_scale: float = 1.,
                 history_sigma: float = 1., history_render: str = 'points') -> torch.Tensor:
    """Tensors (any device) -> model input (B, C, D, H, W). Only the nearest
    ``n_render`` history points are rendered (the rest lie behind the crop)."""
    x = sample_oriented_fast(raw, starts.float(), pos*input_scale, frames*input_scale, grid, gate_direction=gate_direction)
    if n_render is not None:
        hist, hmask = hist[:, :n_render], hmask[:, :n_render]
    return torch.cat([x, render_history(hist, hmask, grid, history_sigma, history_render)], 1)


def collate_with_volume(items, vol: FiberVolume, crop: CropSpec, grid: torch.Tensor | None = None):
    """Worker-side batch. With ``grid`` the model input ``x`` (fp16) is built
    here on CPU; otherwise raw blocks + geometry are returned."""
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
            C = {"ct": 1, "fiber": 7, "fiber+ct": 8}[vol.spec.mode] + 1
            x = np.empty((len(items), C, crop.depth, crop.width, crop.width), np.float16)
            for j, it in enumerate(items):
                x[j] = sample_crop(raw[j], starts[j], it["pos"]*scale, it["frame"]*scale, gf, crop.gate_direction,
                                   it["hist_local"][:nr], it["hmask"][:nr], C, crop.history_sigma, crop.history_render).reshape(x.shape[1:])
            out = dict(x=torch.from_numpy(x), hist=out["hist"], hmask=out["hmask"])
        else:
            x = build_inputs(out.pop("raw"), out.pop("starts"), out["pos"], out["frames"], out["hist"], out["hmask"],
                             grid, n_render=render_count(crop), gate_direction=crop.gate_direction, input_scale=scale,
                             history_sigma=crop.history_sigma, history_render=crop.history_render)
            out = dict(x=x.half(), hist=out["hist"], hmask=out["hmask"])
    if "fut_local" in items[0]:
        out["fut"] = st("fut_local")
        out["fmask"] = st("fmask")
        out["plane_ab"] = st("plane_ab")
        out["plane_mask"] = st("plane_mask")
        for key in ("dense_ab", "dense_mask", "end_local", "endpoint_known", "offtrack", "replay_candidates", "replay_valid"):
            out[key] = st(key)
    if 'source' in items[0]:
        out['source'] = st('source')
        out['source_step'] = st('source_step')
    if 'tube_segments' in items[0]:
        from vesuvius.neural_tracing.fiber_follow.tube import render_tube
        tubes = [render_tube(it, crop, it['tube_sigma']) for it in items]
        out['tube_target'] = torch.from_numpy(np.stack([t[0] for t in tubes]))
        out['tube_mask'] = torch.from_numpy(np.stack([t[1] for t in tubes]))
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


STATE_VERSION = 3

class OnPolicyStates:
    """States (pos, heading, own-trace history) visited by a tracer on GT fibers.

    Loaded as memory-mapped .npy files so DataLoader workers share one copy
    (pickling sends only the directory path)."""

    FIELDS = ("fiber_idx", "t", "reverse", "pos", "frame", "hist", "hmask", "offtrack",
              "hard", "exploratory", "candidates", "chosen", "confidence", "rank_scores")

    def __init__(self, *, manifest, provenance=None, **arrays):
        for key in self.FIELDS:
            setattr(self, key, np.asarray(arrays[key]))
        if any(len(getattr(self, k)) != len(self.pos) for k in self.FIELDS):
            raise ValueError('Replay arrays have different lengths')
        self._dir = None
        self.manifest = manifest
        self.provenance = provenance or {}

    def __len__(self):
        return len(self.pos)

    def save(self, path):
        np.savez(path, __metadata__=json.dumps(dict(version=STATE_VERSION, fibers=self.manifest, provenance=self.provenance)),
                 **{k: getattr(self, k) for k in self.FIELDS})

    def validate_fibers(self, fibers):
        if self.manifest != fiber_manifest(fibers):
            raise ValueError("On-policy cache has incompatible fibers or geometry; "
                             "recollect with collect.py using the current controlled-span labels")
        if len(self) and (np.any(self.fiber_idx < 0) or np.any(self.fiber_idx >= len(fibers))):
            raise ValueError("On-policy cache contains invalid fiber indices")
        if len(self):
            lengths = np.array([f.length for f in fibers])[self.fiber_idx]
            if np.any(~np.isfinite(self.t)) or np.any((self.t < 0) | (self.t > lengths)):
                raise ValueError("On-policy cache contains arc positions outside controlled spans")

    @classmethod
    def load(cls, path):
        """``path``: .npz from collect.py (converted once to a sibling ``_mmap/``
        dir of .npy files) or such a directory."""
        path = os.fspath(path)
        d = path[:-4] + "_mmap_v3" if path.endswith(".npz") else path
        metadata_path = os.path.join(d, "metadata.json")
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as z:
                metadata = json.loads(str(z["__metadata__"].item()))
                if metadata["version"] != STATE_VERSION:
                    raise ValueError("Incompatible on-policy cache version; recollect with collect.py")
                # Include archive identity so an overwritten NPZ cannot reuse stale mmap arrays.
                stat = os.stat(path)
                metadata["archive"] = [stat.st_size, stat.st_mtime_ns]
                # Arc positions are checked against float64 controlled-span lengths.
                # Rebuild older derived caches that rounded valid endpoints upward.
                metadata["mmap_t_dtype"] = str(z["t"].dtype)
                existing = None
                if os.path.exists(metadata_path):
                    with open(metadata_path) as fh:
                        existing = json.load(fh)
                if existing != metadata or any(not os.path.exists(os.path.join(d, k + ".npy")) for k in cls.FIELDS):
                    os.makedirs(d, exist_ok=True)
                    for k in cls.FIELDS:
                        v = z[k]
                        if k != "t" and v.dtype == np.float64:
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
            raise ValueError("Incompatible on-policy cache version; recollect with collect.py")
        self.manifest = metadata["fibers"]
        self.provenance = metadata["provenance"]
        self._dir = d
        for k in self.FIELDS:
            setattr(self, k, np.load(os.path.join(d, k + ".npy"), mmap_mode="r"))

    def __getstate__(self):
        if self._dir is None:
            return self.__dict__
        return {"_dir": self._dir}

    def __setstate__(self, state):
        if state.get("_dir") and len(state) == 1:
            self._open(state["_dir"])
        else:
            self.__dict__.update(state)
