"""One tracing step, shared by training, evaluation and inference.

``step``: crops at every point, model, predicted points, and the front curve through them. The front points are the
control points of a cubic Hermite curve: confident points move to their predictions and tilt their tangents;
low-confidence points take displacement interpolated from confident neighbours and are put on the curve through them.
The model also sees a grid of CT tokens sampled along the current curve and the curves of the last ``memory_rows``
rows (columns aligned by arc length; point ``k`` sits at column ``k * spline_samples``).

``advance``: re-space front points (and the curve's columns) by arc length along the curve and push the row into the
grid memory. A masked point is predicted at every step and rejoins the output when its confidence recovers.
``teacher_forced`` instead puts the front on a given (GT) row, to fill the memory before a training rollout.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vesuvius.neural_tracing.tracers.sheets.geometry import (
    point_frames, carry_stopped, chord_tangents, front_frame, hermite_dense, normalize, onto_confident_curve, respace,
    to_local)
from vesuvius.neural_tracing.tracers.sheets.data.episodes import FIBER_CODE_DIM
from vesuvius.neural_tracing.tracers.sheets.sampling import crop_offsets, sample_crops


@dataclass
class Front:
    pos: torch.Tensor  # (B, N, 3)
    prev: torch.Tensor  # (B, N, 3) previous row (direction of advance)
    conf: torch.Tensor  # (B, N) confidence of the step that produced ``pos``
    curve: torch.Tensor | None = None  # (B, Q, 3) arc-length columns of the curve through ``pos``
    grid: list = field(default_factory=list)  # [(tokens (B, Q, D), curve (B, Q, 3), frames, conf (B, Q))], newest last


@dataclass
class StepOutput:
    pred: torch.Tensor  # (B, N, 3) each point's own prediction
    conf_logit: torch.Tensor  # (B, N)
    points: torch.Tensor  # (B, N, 3) the curve at the front points (low-confidence points carried along)
    dense: torch.Tensor  # (B, Q, 3) the curve, spline_samples per gap; gap k spans points k, k+1
    tokens: torch.Tensor  # (B, N, D)
    frames: torch.Tensor  # (B, N, 3, 3)
    grid_row: tuple  # (tokens (B, Q, D), curve, frames, conf) of the current grid row


def select_front(front: Front, index: torch.Tensor) -> Front:
    """The fronts ``index`` (1-D) of a batch of fronts, with their memory."""
    pick = lambda x: None if x is None else x[index]
    rows = lambda r: tuple(pick(x) for x in r)
    return Front(pick(front.pos), pick(front.prev), pick(front.conf), pick(front.curve), [rows(g) for g in front.grid])


def concat_fronts(fronts: list[Front]) -> Front:
    """One batch of fronts from several with the same memory layout (as live-chain continuations have)."""
    cat = lambda xs: None if xs[0] is None else torch.cat(xs)
    rows = lambda rs: tuple(cat(list(xs)) for xs in zip(*rs))
    return Front(cat([f.pos for f in fronts]), cat([f.prev for f in fronts]), cat([f.conf for f in fronts]),
                 cat([f.curve for f in fronts]), [rows(g) for g in zip(*(f.grid for f in fronts))])


class Tracer:
    def __init__(self, model, cfg, device):
        self.model, self.cfg = model, cfg
        d, m, t = cfg.data, cfg.model, cfg.trace
        self.mode = 1 if d.fiber_follow else 0  # model input with model.fiber_follow_mode
        self.step_len = d.step
        self.level = d.volume_level
        self.offsets = crop_offsets(m.crop, m.crop_spacing, m.crop_forward, device)
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
        self.token_offsets = crop_offsets(m.token_crop, m.token_spacing, 0., device)

    def _interp_front_points(self, values: torch.Tensor) -> torch.Tensor:
        """Front point values (B, N, ...) linearly interpolated to the curve columns (B, Q, ...)."""
        w = self.col_w.reshape(1, -1, *([1]*(values.ndim-2)))
        return (1-w)*values[:, self.col_gap]+w*values[:, self.col_gap+1]

    def output_columns(self, pos: torch.Tensor, conf: torch.Tensor, dense: torch.Tensor | None = None):
        """The traced row: the dense Hermite curve, each sample with the lower confidence of its gap's two control
        points. When omitted, ``dense`` defaults to the Catmull-Rom curve through ``pos``.
        """
        if dense is None:
            dense = hermite_dense(pos, chord_tangents(pos), self.cfg.trace.spline_samples)
        return dense, torch.minimum(conf[:, self.col_gap], conf[:, self.col_gap+1])

    def column_spacing(self) -> float:
        d = self.cfg.data
        return d.point_spacing/self.cfg.trace.spline_samples

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

    def _crops(self, block, origin, pos, frames, offsets, neighbor):
        """CT crops at ``pos``, plus the neighbour channel with ``model.neighbor_channel``: the ``neighbor`` volumes
        (``tracing.neighbor.volumes``) sampled raw at the same points, or zeros without them."""
        crops = sample_crops(block, origin, self.level, pos, frames, offsets)
        if not self.cfg.model.neighbor_channel:
            return crops
        if neighbor is None:
            return torch.cat([crops, torch.zeros_like(crops)], 2)
        side = sample_crops(neighbor['volumes'], neighbor['origins'], neighbor['level'], pos, frames, offsets, zscore=False)
        return torch.cat([crops, side.to(crops.dtype)], 2)

    def _grid(self, front: Front, frames, block, origin, neighbor=None):
        """Model inputs for the token grid, and the current row's (curve, frames, conf)."""
        B = front.pos.shape[0]
        curve = self._curve(front)
        t = chord_tangents(curve)
        a = self._interp_front_points(frames[..., 1])
        a = normalize(a-(a*t).sum(-1, keepdim=True)*t)
        col_frames = torch.stack([t, a, normalize(torch.cross(t, a, dim=-1))], -1)
        col_conf = self._interp_front_points(front.conf)
        crops = self._crops(block, origin, curve, col_frames, self.token_offsets, neighbor)
        center, basis = front_frame(front.pos, front.prev)
        R = max(self.cfg.model.memory_rows, 1)
        rows = front.grid[-self.cfg.model.memory_rows:] if self.cfg.model.memory_rows else []  # oldest first
        geom = []
        for lag, (_, pos, fr, conf) in [(len(rows)-i, row) for i, row in enumerate(rows)]+[(0, (None, curve, col_frames,
                                                                                                 col_conf))]:
            geom.append(torch.cat([to_local(pos, center, basis)/self.step_len, self.col_features.expand(B, -1, 3),
                                   torch.full((B, self.columns, 1), min(lag, R)/R, device=pos.device), conf[..., None],
                                   (basis.transpose(-1, -2)[:, None]@fr).flatten(-2)], -1))
        past = torch.stack([row[0] for row in rows], 1) if rows else None
        return dict(crops=crops, geom=torch.stack(geom, 1), past=past), (curve, col_frames, col_conf)

    def initial(self, seed_rows: torch.Tensor) -> Front:
        """Front from seed rows (B, H+1, N, 3), oldest first; the last row is where tracing starts."""
        B, _, N, _ = seed_rows.shape
        return Front(seed_rows[:, -1], seed_rows[:, -2], seed_rows.new_ones(B, N))

    def _features(self, front: Front, frames: torch.Tensor) -> torch.Tensor:
        """Point geometry (B, N, POINT_FEATURES): position and frame in the front frame, place along the front,
        previous confidence."""
        B, N = front.pos.shape[:2]
        center, basis = front_frame(front.pos, front.prev)
        return torch.cat([to_local(front.pos, center, basis)/self.step_len,
                          (basis.transpose(-1, -2)[:, None]@frames).flatten(-2),
                          self.index_features.expand(B, N, 3), front.conf[..., None]], -1)

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

    def step(self, front: Front, block, origin: torch.Tensor, fibers=None, neighbor=None) -> StepOutput:
        """``block``: the episodes' CT blocks (``shared.crop_sampling.gpu.sample_block``) with their level-voxel origins (B, 3).
        ``fibers``: the batch's ``fiber_*`` tensors (fiber-started episodes), for ``model.fiber_tokens``.
        ``neighbor``: the episodes' neighbouring traced surfaces (``tracing.neighbor.volumes``), for
        ``model.neighbor_channel`` (none: an empty channel)."""
        frames = point_frames(front.pos, front.prev)
        crops = self._crops(block, origin, front.pos, frames, self.offsets, neighbor)
        point = self._features(front, frames)
        grid, row_geometry = self._grid(front, frames, block, origin, neighbor)
        amp = self.cfg.train.amp == 'bf16' and crops.is_cuda
        with torch.autocast('cuda', torch.bfloat16, enabled=amp):
            mode = (torch.full((crops.shape[0],), self.mode, dtype=torch.long, device=crops.device)
                    if self.cfg.model.fiber_follow_mode else None)
            fiber_tokens = (self._fiber_tokens(front, fibers) if fibers is not None and self.cfg.model.fiber_tokens
                            else None)
            disp, conf_logit, tokens, tilt, row = self.model(crops, point, grid, mode, fiber_tokens)
        disp = disp.float()*self.step_len
        disp = torch.stack([disp[..., 0], disp[..., 1]+self.step_len, disp[..., 2]], -1)
        pred = front.pos+(frames@disp[..., None])[..., 0]
        conf_logit = conf_logit.float()
        t = self.cfg.trace
        confident = torch.sigmoid(conf_logit.detach()) > t.confidence_threshold
        points = onto_confident_curve(front.pos+carry_stopped(pred-front.pos, confident), confident)
        tangents = chord_tangents(points)
        tilted = normalize(tangents+tilt[..., :1].float()*frames[..., 1]+tilt[..., 1:].float()*frames[..., 2])
        tangents = torch.where(confident[..., None], tilted, tangents)
        dense = hermite_dense(points, tangents, t.spline_samples)
        return StepOutput(pred, conf_logit, points, dense, tokens.float(), frames, (row.float(), *row_geometry))

    def advance(self, front: Front, out: StepOutput) -> Front:
        """The next front, detached from this step's graph (each training step's loss trains that step only)."""
        conf = torch.sigmoid(out.conf_logit).detach()
        pos = respace(out.dense, front.pos.shape[1]).detach()
        row = out.grid_row
        grid = (front.grid+[(row[0].detach(), *row[1:])])[-max(1, self.cfg.model.memory_rows):]
        curve = respace(out.dense, self.columns).detach()
        return Front(pos, front.pos, conf, curve, grid)

    def teacher_forced(self, front: Front, pos: torch.Tensor) -> Front:
        """After ``advance``: the front put on row ``pos`` (B, N, 3) instead, fully confident, keeping the grid
        memory; its direction is from the row it was traced from."""
        return Front(pos, front.prev, torch.ones_like(front.conf), None, front.grid)
