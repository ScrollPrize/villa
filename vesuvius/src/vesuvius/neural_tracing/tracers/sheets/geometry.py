"""Torch geometry for the front: GT surface grids, projection, Hermite front curves and point frames.

Points are level-0 ``x, y, z``. A GT grid is a ``(B, Gu, Gv, 3)`` patch resampled onto a regular ``(u, v)``
lattice (``u`` = advance direction, ``v`` = along the front, both in grid-index units) with a ``(B, Gu, Gv)``
validity mask; invalid nodes hold zeros. Front point frames are ``(..., 3, 3)`` with columns ``(t, a, n)``:
front tangent (increasing point index), advance direction, and normal ``n = t x a``. The GT frame uses the
same convention with ``t = dS/dv`` and ``a = dS/du``, so the normals of the two agree in sign.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.tracers.shared.compile import compile_options


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v/v.norm(dim=-1, keepdim=True).clamp_min(eps)


def _gather(flat: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``flat`` (B, G, C) or (B, G) gathered at ``index`` (B, Q)."""
    if flat.ndim == 2:
        return torch.gather(flat, 1, index)
    return torch.gather(flat, 1, index[..., None].expand(-1, -1, flat.shape[-1]))


def grid_eval(gt: torch.Tensor, valid: torch.Tensor, uv: torch.Tensor):
    """Bilinear surface ``S``, partials ``S_u``, ``S_v`` (per grid unit) and validity at ``uv`` (B, Q, 2)."""
    B, Gu, Gv, _ = gt.shape
    inside = (uv[..., 0] >= 0) & (uv[..., 0] <= Gu-1) & (uv[..., 1] >= 0) & (uv[..., 1] <= Gv-1)
    u = uv[..., 0].clamp(0, Gu-1-1e-4)
    v = uv[..., 1].clamp(0, Gv-1-1e-4)
    u0, v0 = u.floor().long(), v.floor().long()
    fu, fv = (u-u0)[..., None], (v-v0)[..., None]
    flat, vflat = gt.reshape(B, Gu*Gv, 3), valid.reshape(B, Gu*Gv)
    corners, ok = [], inside
    for du in (0, 1):
        for dv in (0, 1):
            index = (u0+du)*Gv+v0+dv
            corners.append(_gather(flat, index))
            ok = ok & _gather(vflat, index)
    p00, p01, p10, p11 = corners
    S = (1-fu)*((1-fv)*p00+fv*p01)+fu*((1-fv)*p10+fv*p11)
    Su = (1-fv)*(p10-p00)+fv*(p11-p01)
    Sv = (1-fu)*(p01-p00)+fu*(p11-p10)
    return S, Su, Sv, ok


def surface_frame(Su: torch.Tensor, Sv: torch.Tensor) -> torch.Tensor:
    t, a = normalize(Sv), normalize(Su)
    return torch.stack([t, a, normalize(torch.cross(t, a, dim=-1))], -1)


_COMPILED_PROJECT = None


def use_compiled_projection(enabled: bool = True) -> None:
    """Route ``project`` through ``torch.compile`` (about 1.4x faster training on an H100: it fuses hundreds of tiny
    kernels per call). Not bit-identical to eager: foot points differ by ~1e-3 vx typically, and a rare query can
    settle on a different Gauss-Newton solution."""
    global _COMPILED_PROJECT
    _COMPILED_PROJECT = torch.compile(_project, **compile_options()) if enabled else None


@torch.no_grad()
def project(gt: torch.Tensor, valid: torch.Tensor, q: torch.Tensor, iters: int = 3):
    """Closest GT surface point to each query; see ``_project``."""
    return (_COMPILED_PROJECT or _project)(gt, valid, q, iters)


def _project(gt: torch.Tensor, valid: torch.Tensor, q: torch.Tensor, iters: int = 3):
    """Closest GT surface point to each query ``q`` (B, Q, 3).

    Nearest valid node, then Gauss-Newton on the bilinear patch (at most one cell per iteration, never
    moving onto an invalid cell). Returns ``uv``, ``S``, frame ``(t, a, n)`` and ``ok``: the cell is valid
    and the projection is not pinned to the grid border (where the true foot point may lie outside).
    """
    B, Gu, Gv, _ = gt.shape
    flat, vflat = gt.reshape(B, Gu*Gv, 3), valid.reshape(B, Gu*Gv)
    dist = torch.cdist(q.float(), flat.float())
    dist.masked_fill_(~vflat[:, None, :], float('inf'))
    index = dist.argmin(-1)
    any_valid = vflat.any(1, keepdim=True)
    uv = torch.stack([index//Gv, index % Gv], -1).to(gt.dtype)
    S, Su, Sv, ok = grid_eval(gt, valid, uv)
    for _ in range(iters):
        # 2x2 normal equations solved explicitly; degenerate cells (|det| tiny) do not move.
        a, b, c = (Su*Su).sum(-1), (Su*Sv).sum(-1), (Sv*Sv).sum(-1)
        r = q-S
        ru, rv = (Su*r).sum(-1), (Sv*r).sum(-1)
        det = a*c-b*b
        good = det > 1e-6*(a*c).clamp_min(1e-12)
        det = torch.where(good, det, torch.ones_like(det))
        du = torch.where(good, (c*ru-b*rv)/det, torch.zeros_like(det))
        dv = torch.where(good, (a*rv-b*ru)/det, torch.zeros_like(det))
        delta = torch.stack([du, dv], -1).clamp(-1, 1)
        trial = torch.stack([(uv[..., 0]+delta[..., 0]).clamp(0, Gu-1), (uv[..., 1]+delta[..., 1]).clamp(0, Gv-1)], -1)
        S2, Su2, Sv2, ok2 = grid_eval(gt, valid, trial)
        keep = ok2 | ~ok
        uv = torch.where(keep[..., None], trial, uv)
        S, Su, Sv = (torch.where(keep[..., None], new, old) for new, old in ((S2, S), (Su2, Su), (Sv2, Sv)))
        ok = torch.where(keep, ok2, ok)
    interior = (uv[..., 0] > .5) & (uv[..., 0] < Gu-1.5) & (uv[..., 1] > .5) & (uv[..., 1] < Gv-1.5)
    return uv, S, surface_frame(Su, Sv), ok & interior & any_valid


def sheet_distance(q: torch.Tensor, S: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Signed point-to-plane distance of ``q`` to the tangent plane at ``S`` (differentiable in ``q``)."""
    return ((q-S)*frame[..., 2]).sum(-1)


def chord_tangents(points: torch.Tensor) -> torch.Tensor:
    """Catmull-Rom unit tangents (B, N >= 3, 3): central chords inside; at each end the neighbour's tangent
    reflected across the end chord (exact for circular arcs and lines, unlike a one-sided chord)."""
    inner = normalize(points[:, 2:]-points[:, :-2])
    c0 = normalize(points[:, 1]-points[:, 0])
    c1 = normalize(points[:, -1]-points[:, -2])
    t1, tn = inner[:, 0], inner[:, -1]
    first = 2*(t1*c0).sum(-1, keepdim=True)*c0-t1
    last = 2*(tn*c1).sum(-1, keepdim=True)*c1-tn
    return normalize(torch.cat([first[:, None], inner, last[:, None]], 1))


def hermite_dense(points: torch.Tensor, tangents: torch.Tensor, samples_per_gap: int) -> torch.Tensor:
    """Cubic Hermite curve through ``points`` (B, N, 3) with unit ``tangents``, sampled ``samples_per_gap`` times
    per gap (plus the last point): (B, (N-1)*S+1, 3). Tangents are scaled by each gap's chord length."""
    s = torch.linspace(0, 1, samples_per_gap+1, device=points.device, dtype=points.dtype)[:-1][None, None, :, None]
    p0, p1 = points[:, :-1, None], points[:, 1:, None]
    chord = (p1-p0).norm(dim=-1, keepdim=True)
    m0, m1 = tangents[:, :-1, None]*chord, tangents[:, 1:, None]*chord
    s2, s3 = s*s, s*s*s
    seg = (2*s3-3*s2+1)*p0+(s3-2*s2+s)*m0+(-2*s3+3*s2)*p1+(s3-s2)*m1
    return torch.cat([seg.flatten(1, 2), points[:, -1:]], 1)


def carry_stopped(disp: torch.Tensor, confident: torch.Tensor) -> torch.Tensor:
    """Displacements (B, N, 3) with each stopped point's replaced by the linear interpolation (by point index)
    of its nearest confident neighbours' (the nearest one beyond the last); unchanged if none is confident."""
    B, N = confident.shape
    k = torch.arange(N, device=disp.device).expand(B, N)
    left = torch.where(confident, k, torch.full_like(k, -1)).cummax(1).values
    right = torch.where(confident, k, torch.full_like(k, N)).flip(1).cummin(1).values.flip(1)
    has_l, has_r = left >= 0, right < N
    li, ri = left.clamp(0, N-1), right.clamp(0, N-1)
    dl, dr = _gather(disp, li), _gather(disp, ri)
    w = ((k-left)/(right-left).clamp_min(1)).clamp(0, 1)[..., None].to(disp.dtype)
    both = (1-w)*dl+w*dr
    carried = torch.where((has_l & has_r)[..., None], both, torch.where(has_l[..., None], dl, dr))
    keep = confident | ~(has_l | has_r)
    return torch.where(keep[..., None], disp, carried)


def onto_confident_curve(points: torch.Tensor, confident: torch.Tensor) -> torch.Tensor:
    """Re-place each stopped point that has confident points on both sides onto the cubic Hermite curve through
    its nearest confident neighbours (tangents from the confident points only), at its place along the front.

    Carrying a stopped point by its neighbours' displacement keeps whatever offset it had (e.g. a sheet away), so
    it could never land back on the sheet and regain confidence; this puts it on the sheet its neighbours are on.
    Points beyond the first or last confident point, and confident points, are returned unchanged.
    """
    B, N = confident.shape
    k = torch.arange(N, device=points.device).expand(B, N)
    left = torch.where(confident, k, torch.full_like(k, -1)).cummax(1).values  # last confident index <= k
    right = torch.where(confident, k, torch.full_like(k, N)).flip(1).cummin(1).values.flip(1)  # first >= k
    prev = torch.cat([torch.full_like(k[:, :1], -1), left[:, :-1]], 1)  # last confident index < k
    nxt = torch.cat([right[:, 1:], torch.full_like(k[:, :1], N)], 1)  # first confident index > k

    def bessel(j):
        """Parabolic (Bessel) tangent at confident point ``j`` from its confident neighbours (unequal index gaps),
        and whether it has one on each side."""
        pj, nj = torch.gather(prev, 1, j), torch.gather(nxt, 1, j)
        both = (pj >= 0) & (nj < N)
        p = _gather(points, j)
        pp, pn = _gather(points, pj.clamp(min=0)), _gather(points, nj.clamp(max=N-1))
        h1 = (j-pj).clamp_min(1).to(points.dtype)[..., None]
        h2 = (nj-j).clamp_min(1).to(points.dtype)[..., None]
        t = (h2*(p-pp)/h1+h1*(pn-p)/h2)/(h1+h2)
        return normalize(t), both

    inside = ~confident & (left >= 0) & (right < N)
    li, ri = left.clamp(0, N-1), right.clamp(0, N-1)
    p0, p1 = _gather(points, li), _gather(points, ri)
    chord = (p1-p0).norm(dim=-1, keepdim=True)
    c = normalize(p1-p0)
    t0, ok0 = bessel(li)
    t1, ok1 = bessel(ri)
    reflect = lambda t: 2*(t*c).sum(-1, keepdim=True)*c-t  # mirror across the chord: exact for circular arcs
    t0, t1 = (torch.where(ok0[..., None], t0, torch.where(ok1[..., None], reflect(t1), c)),
              torch.where(ok1[..., None], t1, torch.where(ok0[..., None], reflect(t0), c)))
    m0, m1 = t0*chord, t1*chord
    s = ((k-left)/(right-left).clamp_min(1)).to(points.dtype)[..., None]
    s2, s3 = s*s, s*s*s
    curve = (2*s3-3*s2+1)*p0+(s3-2*s2+s)*m0+(-2*s3+3*s2)*p1+(s3-s2)*m1
    return torch.where(inside[..., None], curve, points)


def respace(dense: torch.Tensor, count: int) -> torch.Tensor:
    """``count`` points uniformly spaced by arc length along the polyline ``dense`` (B, Q, 3)."""
    seg = (dense[:, 1:]-dense[:, :-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(1)], 1)
    target = torch.linspace(0, 1, count, device=dense.device, dtype=dense.dtype)[None]*cum[:, -1:]
    hi = torch.searchsorted(cum.contiguous(), target.contiguous()).clamp(1, dense.shape[1]-1)
    lo = hi-1
    c0, c1 = torch.gather(cum, 1, lo), torch.gather(cum, 1, hi)
    w = ((target-c0)/(c1-c0).clamp_min(1e-9)).clamp(0, 1)[..., None]
    return (1-w)*_gather(dense, lo)+w*_gather(dense, hi)


def point_frames(pos: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """Per-point ``(t, a, n)`` from the current front and the previous row."""
    t = torch.cat([pos[:, 1:2]-pos[:, :1], pos[:, 2:]-pos[:, :-2], pos[:, -1:]-pos[:, -2:-1]], 1)
    t = normalize(t)
    a = pos-prev
    a = normalize(a-(a*t).sum(-1, keepdim=True)*t)
    return torch.stack([t, a, normalize(torch.cross(t, a, dim=-1))], -1)


def front_frame(pos: torch.Tensor, prev: torch.Tensor):
    """Front centre (B, 3) and one ``(t, a, n)`` frame (B, 3, 3) for rotation-invariant geometry features."""
    center = pos.mean(1)
    t = normalize(pos[:, -1]-pos[:, 0])
    a = (pos-prev).mean(1)
    a = normalize(a-(a*t).sum(-1, keepdim=True)*t)
    return center, torch.stack([t, a, normalize(torch.cross(t, a, dim=-1))], -1)


def to_local(points: torch.Tensor, center: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Coordinates of ``points`` (B, ..., 3) in the front frame."""
    shape = points.shape
    flat = (points.reshape(shape[0], -1, 3)-center[:, None])@frame
    return flat.reshape(shape)


def huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    return F.huber_loss(x, torch.zeros_like(x), delta=delta, reduction='none')
