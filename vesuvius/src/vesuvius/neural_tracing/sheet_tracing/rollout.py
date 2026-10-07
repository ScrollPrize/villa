"""One tracing step, shared by training, evaluation and inference.

``step``: crops at every point, model, predicted points, and the front curve through them:

- ``trace.curve = 'smoothing'``: a confidence-weighted smoothing B-spline through the predictions, smoothing the
  change from the current front. Low-confidence points get the weight floor, so the spline carries them along parallel
  to their confident neighbours.
- ``trace.curve = 'hermite'``: front points are the curve's control points. Confident points move to their predictions
  and tilt their tangents; low-confidence points take displacement interpolated from confident neighbours.
  The curve is the cubic Hermite through those points.

With ``model.surface_tokens`` the model also sees a grid of CT tokens sampled along the current curve and the
curves of the last ``memory_rows`` rows (columns aligned by arc length; point ``k`` sits at column
``k * spline_samples``), or of the rows at ``memory_lags`` plus, with ``memory_anchor``, the trace's first row.

``advance``: re-space front points (and the curve's columns) by arc length along the curve and push the row into
memory. A masked point is predicted at every step and rejoins the output when its confidence recovers.
``teacher_forced`` instead puts the front on a given (GT) row, to fill the memory before a training rollout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from vesuvius.neural_tracing.sheet_tracing.geometry import (
    FrontSpline, point_frames, carry_stopped, chord_tangents, front_frame, hermite_dense, normalize,
    onto_confident_curve, respace, to_local)
from vesuvius.neural_tracing.sheet_tracing.data.episodes import FIBER_CODE_DIM
from vesuvius.neural_tracing.sheet_tracing.sampling import crop_offsets, sample_crops


@dataclass
class Front:
    pos: torch.Tensor  # (B, N, 3)
    prev: torch.Tensor  # (B, N, 3) previous row (direction of advance)
    conf: torch.Tensor  # (B, N) confidence of the step that produced ``pos``
    memory: list = field(default_factory=list)  # [(tokens (B, N, D), pos (B, N, 3), conf (B, N))], newest last
    curve: torch.Tensor | None = None  # (B, Q, 3) arc-length columns of the curve through ``pos``
    grid: list = field(default_factory=list)  # [(tokens (B, Q, D), curve (B, Q, 3), frames, conf (B, Q))]
    anchor: tuple | None = None  # the first grid row of the trace (kept for model.memory_anchor)
    rows: int = 0  # grid rows produced so far, i.e. the current row's lag behind the anchor


@dataclass
class StepOutput:
    pred: torch.Tensor  # (B, N, 3) each point's own prediction
    conf_logit: torch.Tensor  # (B, N)
    points: torch.Tensor  # (B, N, 3) the curve at the front points (low-confidence points carried along)
    dense: torch.Tensor  # (B, Q, 3) the curve, spline_samples per gap; gap k spans points k, k+1
    tokens: torch.Tensor  # (B, N, D)
    frames: torch.Tensor  # (B, N, 3, 3)
    grid_row: tuple | None = None  # (tokens (B, Q, D), curve, frames, conf) of the current grid row


def select_front(front: Front, index: torch.Tensor) -> Front:
    """The fronts ``index`` (1-D) of a batch of fronts, with their memory."""
    pick = lambda x: None if x is None else x[index]
    rows = lambda r: tuple(pick(x) for x in r)
    return Front(pick(front.pos), pick(front.prev), pick(front.conf), [rows(m) for m in front.memory],
                 pick(front.curve), [rows(g) for g in front.grid], None if front.anchor is None else rows(front.anchor),
                 front.rows)


def concat_fronts(fronts: list[Front]) -> Front:
    """One batch of fronts from several with the same memory layout (as live-chain continuations have)."""
    cat = lambda xs: None if xs[0] is None else torch.cat(xs)
    rows = lambda rs: tuple(cat(list(xs)) for xs in zip(*rs))
    return Front(cat([f.pos for f in fronts]), cat([f.prev for f in fronts]), cat([f.conf for f in fronts]),
                 [rows(m) for m in zip(*(f.memory for f in fronts))], cat([f.curve for f in fronts]),
                 [rows(g) for g in zip(*(f.grid for f in fronts))],
                 None if fronts[0].anchor is None else rows([f.anchor for f in fronts]), max(f.rows for f in fronts))


class Tracer:
    def __init__(self, model, cfg, device):
        self.model, self.cfg = model, cfg
        d, m, t = cfg.data, cfg.model, cfg.trace
        self.mode = 1 if d.fiber_follow else 0  # model input with model.fiber_follow_mode
        self.step_len = d.step
        self.extra_levels = [int(level) for level in (d.extra_levels or [d.volume_level]*len(d.extra_volumes))]
        self.level = d.volume_level
        self.hermite = t.curve == 'hermite'
        if not self.hermite:
            self.spline = FrontSpline(d.front_points, cfg.spline_ctrl(), t.spline_smooth, t.spline_samples, device)
        self.offsets = crop_offsets(m.crop, m.crop_spacing, m.crop_forward, device)
        self.context_offsets = crop_offsets(m.context_crop, m.context_spacing, 0., device) if m.context_crop else None
        s = torch.linspace(0, 1, d.front_points, device=device)
        self.index_features = torch.stack([2*s-1, torch.sin(torch.pi*s), torch.cos(torch.pi*s)], -1)
        S = t.spline_samples
        self.columns = (d.front_points-1)*S+1
        self.point_cols = torch.arange(d.front_points, device=device)*S
        j = torch.arange(self.columns, device=device)
        self.col_gap = (j//S).clamp(max=d.front_points-2)
        self.col_w = ((j-self.col_gap*S)/S)[None, :, None]
        c = torch.linspace(0, 1, self.columns, device=device)
        self.col_features = torch.stack([2*c-1, torch.sin(torch.pi*c), torch.cos(torch.pi*c)], -1)
        if m.surface_tokens:
            self.token_offsets = crop_offsets(m.token_crop, m.token_spacing, 0., device)
        self.profile_offsets = crop_offsets(m.profile_crop, m.profile_spacing, 0., device) if m.profile_crop else None
        self.lags = [int(lag) for lag in m.memory_lags] or list(range(1, m.memory_rows+1))
        self.keep = max(1, m.memory_rows, *self.lags)  # grid rows stored
        self.lag_scale = math.log2(1+2*self.lags[-1]) if m.memory_lags else None

    def _interp_front_points(self, values: torch.Tensor) -> torch.Tensor:
        """Front point values (B, N, ...) linearly interpolated to the curve columns (B, Q, ...)."""
        w = self.col_w.reshape(1, -1, *([1]*(values.ndim-2)))
        return (1-w)*values[:, self.col_gap]+w*values[:, self.col_gap+1]

    def output_columns(self, pos: torch.Tensor, conf: torch.Tensor, dense: torch.Tensor | None = None):
        """The traced row: front points in smoothing mode, or the dense curve in Hermite mode.

        Each Hermite sample gets the lower confidence of its gap's two control points. When omitted, ``dense``
        defaults to the Catmull-Rom curve through ``pos``.
        """
        if not self.hermite:
            return pos, conf
        if dense is None:
            dense = hermite_dense(pos, chord_tangents(pos), self.cfg.trace.spline_samples)
        return dense, torch.minimum(conf[:, self.col_gap], conf[:, self.col_gap+1])

    def column_spacing(self) -> float:
        d = self.cfg.data
        return d.point_spacing/(self.cfg.trace.spline_samples if self.hermite else 1)

    def move(self, front: Front, pos: torch.Tensor) -> None:
        """Move front points to ``pos`` (training perturbations); curve columns follow their gap's endpoints."""
        if front.curve is not None:
            front.curve = front.curve+self._interp_front_points(pos-front.pos)
        front.pos = pos

    def _curve(self, front: Front) -> torch.Tensor:
        if front.curve is not None:
            return front.curve
        S = self.cfg.trace.spline_samples
        return respace(hermite_dense(front.pos, chord_tangents(front.pos), S), self.columns)

    def _grid(self, front: Front, frames, block, origin, extras=()):
        """Model inputs for the surface-token grid, and the current row's (curve, frames, conf)."""
        B = front.pos.shape[0]
        curve = self._curve(front)
        t = chord_tangents(curve)
        a = self._interp_front_points(frames[..., 1])
        a = normalize(a-(a*t).sum(-1, keepdim=True)*t)
        col_frames = torch.stack([t, a, normalize(torch.cross(t, a, dim=-1))], -1)
        col_conf = self._interp_front_points(front.conf)
        crops = sample_crops(block, origin, self.level, curve, col_frames, self.token_offsets, extras)
        center, basis = front_frame(front.pos, front.prev)
        m = self.cfg.model
        n = len(front.grid)
        rows = [(lag, front.grid[n-lag], False) for lag in reversed(self.lags) if lag <= n]  # oldest first
        if m.memory_anchor and front.anchor is not None and front.rows not in [lag for lag, _, _ in rows]:
            rows.insert(0, (front.rows, front.anchor, True))
        R = max(m.memory_rows, 1)
        geom = []
        for lag, (_, pos, fr, conf), anchor in rows+[(0, (None, curve, col_frames, col_conf), False)]:
            features = [to_local(pos, center, basis)/self.step_len, self.col_features.expand(B, -1, 3),
                        torch.full((B, self.columns, 1), min(lag, R)/R, device=pos.device), conf[..., None],
                        (basis.transpose(-1, -2)[:, None]@fr).flatten(-2)]
            if self.lag_scale:
                features.append(pos.new_tensor([min(1., math.log2(1+lag)/self.lag_scale), float(anchor)])
                                .expand(B, self.columns, 2))
            geom.append(torch.cat(features, -1))
        past = torch.stack([row[0] for _, row, _ in rows], 1) if rows else None
        grid = dict(crops=crops, geom=torch.stack(geom, 1), past=past, point_cols=self.point_cols)
        if self.profile_offsets is not None:
            grid['profile'] = sample_crops(block, origin, self.level, curve, col_frames, self.profile_offsets, extras)
        return grid, (curve, col_frames, col_conf)

    def initial(self, seed_rows: torch.Tensor) -> Front:
        """Front from seed rows (B, H+1, N, 3), oldest first; the last row is where tracing starts."""
        B, _, N, _ = seed_rows.shape
        return Front(seed_rows[:, -1], seed_rows[:, -2], seed_rows.new_ones(B, N))

    def _features(self, front: Front, frames: torch.Tensor):
        B, N = front.pos.shape[:2]
        center, basis = front_frame(front.pos, front.prev)
        point = torch.cat([to_local(front.pos, center, basis)/self.step_len,
                           (basis.transpose(-1, -2)[:, None]@frames).flatten(-2),
                           self.index_features.expand(B, N, 3), front.conf[..., None]], -1)
        if not front.memory or not self.cfg.model.memory_rows:
            return point, None, None
        rows = front.memory[-self.cfg.model.memory_rows:]
        R = len(rows)
        tokens = torch.cat([r[0] for r in rows], 1)
        pos = torch.cat([r[1] for r in rows], 1)
        conf = torch.cat([r[2] for r in rows], 1)
        age = torch.cat([torch.full((B, N, 1), (R-i)/self.cfg.model.memory_rows, device=pos.device)
                         for i in range(R)], 1)
        memory = torch.cat([to_local(pos, center, basis)/self.step_len, age, conf[..., None],
                            self.index_features.repeat(R, 1).expand(B, R*N, 3)], -1)
        return point, tokens, memory

    def _fiber_tokens(self, front: Front, fibers: dict) -> dict:
        """Features (B, K, FIBER_FEATURES) and padding (B, K) of the ``model.fiber_tokens`` fiber points nearest the
        front (``fibers``: the batch's ``fiber_*`` tensors): position and direction (sign-canonical) in the front
        frame, family one-hot, seed-fiber flag (episode instance 0) and the instance's code."""
        pts, inst = fibers['fiber_points'], fibers['fiber_instance']
        K = min(self.cfg.model.fiber_tokens, pts.shape[1])
        dist = torch.cdist(pts, front.pos).min(-1).values
        dist = torch.where(inst >= 0, dist, torch.full_like(dist, float('inf')))
        dist, idx = dist.topk(K, dim=1, largest=False)
        take = lambda x: torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
        p, t = take(pts), take(fibers['fiber_tangent'])
        family, instance = torch.gather(fibers['fiber_family'], 1, idx), torch.gather(inst, 1, idx)
        pad = ~torch.isfinite(dist)
        center, basis = front_frame(front.pos, front.prev)
        local = to_local(p, center, basis)/self.step_len
        direction = t@basis
        direction = direction*torch.where(direction[..., :1] < 0, -1., 1.)  # a fiber has no direction of its own
        codes = torch.gather(fibers['fiber_codes'], 1, instance.clamp_min(0)[..., None].expand(-1, -1, FIBER_CODE_DIM))
        feat = torch.cat([local, direction, torch.nn.functional.one_hot(family.clamp(0, 1), 2).to(local.dtype),
                          (instance == 0).to(local.dtype)[..., None], codes], -1)
        return dict(feat=torch.where(pad[..., None], torch.zeros_like(feat), feat), pad=pad)

    def step(self, front: Front, block: torch.Tensor, origin: torch.Tensor, extras=None, fibers=None) -> StepOutput:
        """``extras``: one ``dict(block, origin)`` per extra volume (each at its own pyramid level), or None.
        ``fibers``: the batch's ``fiber_*`` tensors (fiber-started episodes), for ``model.fiber_tokens``."""
        extras = [(e['block'], e['origin'], level) for e, level in zip(extras or [], self.extra_levels)]
        frames = point_frames(front.pos, front.prev)
        crops = sample_crops(block, origin, self.level, front.pos, frames, self.offsets, extras)
        context = None
        if self.context_offsets is not None:
            context = sample_crops(block, origin, self.level, front.pos, frames, self.context_offsets, extras)
        point, mem_tokens, mem_geom = self._features(front, frames)
        grid, row_geometry = None, None
        if self.cfg.model.surface_tokens:
            grid, row_geometry = self._grid(front, frames, block, origin, extras)
            mem_tokens = mem_geom = None
        amp = self.cfg.train.amp == 'bf16' and crops.is_cuda
        with torch.autocast('cuda', torch.bfloat16, enabled=amp):
            mode = (torch.full((crops.shape[0],), self.mode, dtype=torch.long, device=crops.device)
                    if self.cfg.model.fiber_follow_mode else None)
            fiber_tokens = (self._fiber_tokens(front, fibers) if fibers is not None and self.cfg.model.fiber_tokens
                            else None)
            disp, conf_logit, tokens, tilt, row = self.model(crops, point, mem_tokens, mem_geom, grid, context, mode,
                                                             fiber_tokens)
        disp = disp.float()*self.step_len
        disp = torch.stack([disp[..., 0], disp[..., 1]+self.step_len, disp[..., 2]], -1)
        pred = front.pos+(frames@disp[..., None])[..., 0]
        conf_logit = conf_logit.float()
        t = self.cfg.trace
        if self.hermite:
            confident = torch.sigmoid(conf_logit.detach()) > t.confidence_threshold
            points = front.pos+carry_stopped(pred-front.pos, confident)
            if t.stopped == 'curve':
                points = onto_confident_curve(points, confident)
            tangents = chord_tangents(points)
            if tilt is not None:
                tilt = tilt.float()
                tilted = normalize(tangents+tilt[..., :1]*frames[..., 1]+tilt[..., 1:]*frames[..., 2])
                tangents = torch.where(confident[..., None], tilted, tangents)
            dense = hermite_dense(points, tangents, t.spline_samples)
        else:
            weights = torch.sigmoid(conf_logit.detach()).clamp_min(t.min_weight)
            ctrl = self.spline.fit(pred, weights, reference=front.pos)
            dense = self.spline.dense(ctrl)
            at_front_points = self.spline.at_front_points(ctrl)
            points = torch.where((torch.sigmoid(conf_logit.detach()) > t.confidence_threshold)[..., None], pred,
                                 at_front_points)
        grid_row = None if row is None else (row.float(), *row_geometry)
        return StepOutput(pred, conf_logit, points, dense, tokens.float(), frames, grid_row)

    def advance(self, front: Front, out: StepOutput, detach: bool = True) -> Front:
        t = self.cfg.trace
        conf = torch.sigmoid(out.conf_logit)
        pos = respace(out.dense, front.pos.shape[1]) if t.respace else out.points
        tokens = out.tokens
        if detach:
            pos, tokens, conf = pos.detach(), tokens.detach(), conf.detach()
        keep = max(1, self.cfg.model.memory_rows)
        memory = (front.memory+[(tokens, front.pos, conf)])[-keep:]
        grid, anchor, rows = front.grid, front.anchor, front.rows
        if out.grid_row is not None:
            row = out.grid_row
            row = (row[0].detach(), *row[1:]) if detach else row
            grid = (grid+[row])[-self.keep:]
            anchor = row if anchor is None else anchor
            rows += 1
        curve = respace(out.dense, self.columns)
        if detach:
            curve = curve.detach()
        return Front(pos, front.pos, conf, memory, curve, grid, anchor, rows)

    def teacher_forced(self, front: Front, pos: torch.Tensor) -> Front:
        """After ``advance``: the front put on row ``pos`` (B, N, 3) instead, fully confident, keeping the memory;
        its direction is from the row it was traced from."""
        return Front(pos, front.prev, torch.ones_like(front.conf), front.memory, None, front.grid, front.anchor,
                     front.rows)
