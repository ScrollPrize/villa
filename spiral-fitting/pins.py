"""Pinned winding radii for the gap expander.

The gap expander maps the *intermediate* radius ``r`` on a ray ``(theta, z)``
(umbilicus-centred, post-flow) to the *canonical* spiral radius ``s(r)``.
The unpinned map ``s_free`` is piecewise linear through the free winding
radii table ``R_free[k]`` (``transforms.GapExpandingTransform``). This module
adds anchors ``(R_j, S_j, w_j)`` -- intermediate radius, canonical target and
kernel mass -- built from *pins*: hard-constraint points pushed through the
flow chain, whose target canonical radius is decided by one fractional
winding coordinate ``T_g`` per constraint component.

Layout of this module:

* ``unpinned_map_forward`` / ``unpinned_map_inverse``: ``s_free`` and its inverse over
  a ``[N, K]`` table (the eager ``GapExpandingTransform`` arithmetic).
* ``build_pinned_ray_map``: transparent radius-space blending, a radial
  slope guard, and a pin-only radius deformation.
* ``pinned_map_forward`` / ``pinned_map_inverse``: the unpinned map composed
  with the radius deformation, evaluated explicitly in either direction.
* ``ray_anchors``: rasterised pin lookup: compressed-sparse-row
  cells, per-pin footprints, singular compact kernel, IDW per winding slot.
* ``PinTable``: the per-step rasterised pin set consumed by
  ``transforms.PinnedGapExpandingTransform``.
* ``PinGraph`` / ``PinRegistry``: the constraint registry: the
  model-independent constraint graph (components, pin geometry) and its
  model-dependent finalisation (integer offsets ``n_i``, footprints, ``T``
  initialisation, consistency report).
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import math

import numpy as np
import torch

TWO_PI = 2.0 * math.pi

# The singular kernel's regulariser: small enough that a query on a pin's own
# ray reproduces that pin to floating-point tolerance (K(0) = 1/KERNEL_TINY
# dominates every other pin in the slot, even hundreds of them at K ~ 1e2
# each), large enough to stay finite in float32 (1e12 * radius ~ 1e15 << 3e38).
KERNEL_TINY = 1.0e-12

PIN_KIND_PATCH = 0
PIN_KIND_CHAIN = 1
PIN_KIND_ISOLATED = 2
PIN_KIND_NAMES = {PIN_KIND_PATCH: 'patch', PIN_KIND_CHAIN: 'chain',
                  PIN_KIND_ISOLATED: 'isolated'}


def wrap_angle(delta):
    """Wrap an angle difference into (-pi, pi]."""
    return delta - TWO_PI * torch.floor((delta + math.pi) / TWO_PI)


# ---------------------------------------------------------------------------
# The free (unpinned) map over a [N, K] winding-radius table
# ---------------------------------------------------------------------------


def unpinned_map_forward(r, pre_pin_winding_radii, dr, theta_norm):
    """``s_free(r)``: intermediate radius -> canonical radius.

    ``pre_pin_winding_radii`` is ``[N, K]`` (winding k's intermediate radius on each ray),
    ``r`` and ``theta_norm`` (theta / 2pi) are ``[N]`` or ``[N, M]`` broadcast
    over the ray. Piecewise linear through ``(pre_pin_winding_radii[k], (k + theta_norm) dr)``
    with the end segments extrapolated, exactly the eager
    ``GapExpandingTransform._inverse`` arithmetic.
    """
    num_windings = pre_pin_winding_radii.shape[-1]
    squeeze = r.dim() == pre_pin_winding_radii.dim() - 1
    if squeeze:
        r = r[..., None]
    inner = torch.searchsorted(pre_pin_winding_radii.detach().contiguous(), r.detach().contiguous()) - 1
    inner = inner.clip(min=0, max=num_windings - 2)
    r_inner = torch.gather(pre_pin_winding_radii, -1, inner)
    r_outer = torch.gather(pre_pin_winding_radii, -1, inner + 1)
    tn = theta_norm[..., None]
    canonical_inner = (inner.to(r.dtype) + tn) * dr
    frac = (r - r_inner) / (r_outer - r_inner)
    out = canonical_inner + frac * dr
    return out.squeeze(-1) if squeeze else out


def unpinned_map_inverse(c, pre_pin_winding_radii, dr, theta_norm):
    """``s_free^{-1}(c)``: canonical radius -> intermediate radius.

    The eager ``GapExpandingTransform._call`` arithmetic: bracket by the
    canonical winding ``floor((c - theta_norm dr) / dr)`` (clipped to the
    table) and lerp the table.
    """
    num_windings = pre_pin_winding_radii.shape[-1]
    squeeze = c.dim() == pre_pin_winding_radii.dim() - 1
    if squeeze:
        c = c[..., None]
    tn = theta_norm[..., None]
    shifted = (c - tn * dr).clamp(min=0.)
    inner = torch.floor(shifted.detach() / dr.detach()).to(torch.int64)
    inner = inner.clip(min=0, max=num_windings - 2)
    r_inner = torch.gather(pre_pin_winding_radii, -1, inner)
    r_outer = torch.gather(pre_pin_winding_radii, -1, inner + 1)
    canonical_inner = (inner.to(c.dtype) + tn) * dr
    frac = (c - canonical_inner) / dr
    out = torch.lerp(r_inner, r_outer, frac)
    return out.squeeze(-1) if squeeze else out


# ---------------------------------------------------------------------------
# The pinned monotone map
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PinnedRayMap:
    """The free winding map and a separate pin-only radius deformation.

    ``unpinned_anchors`` and ``radius_anchors`` include the fixed origin;
    ``anchor_counts`` excludes padding and includes that origin.
    """

    pre_pin_winding_radii: torch.Tensor
    dr: torch.Tensor
    theta_norm: torch.Tensor
    unpinned_anchors: torch.Tensor
    radius_anchors: torch.Tensor
    anchor_counts: torch.Tensor
    anchor_canonical: torch.Tensor
    anchor_radii: torch.Tensor
    anchor_valid: torch.Tensor
    guard_active: torch.Tensor
    guard_order: torch.Tensor
    duplicate_conflicts: torch.Tensor
    base_conflicts: torch.Tensor

    def order_violations(self):
        return ((self.guard_active & self.guard_order).sum(dim=-1)
                + self.duplicate_conflicts + self.base_conflicts)

    def min_rise_violations(self):
        return (self.guard_active & ~self.guard_order).sum(dim=-1)

    def minimum_winding_gap(self, dr):
        """Minimum dR/dc * dr, from overlapping free and deformation intervals."""
        free = self.pre_pin_winding_radii
        gaps = free.diff(dim=-1) * (dr / self.dr)
        # The first and last free segments also cover extrapolation.
        lo = torch.cat([torch.full_like(free[:, :1], -float('inf')), free[:, 1:-1]], dim=-1)
        hi = torch.cat([free[:, 1:-1], torch.full_like(free[:, :1], float('inf'))], dim=-1)

        def minimum_between(left, right):
            overlap = (hi > left[:, None]) & (lo < right[:, None])
            return torch.where(overlap, gaps, float('inf')).min(dim=-1).values

        zero = torch.zeros_like(free[:, 0])
        result = minimum_between(torch.full_like(zero, -float('inf')), zero)
        for j in range(1, self.unpinned_anchors.shape[-1]):
            left, right = self.unpinned_anchors[:, j - 1], self.unpinned_anchors[:, j]
            valid = j < self.anchor_counts
            span = torch.where(valid, right - left, torch.ones_like(right))
            slope = (self.radius_anchors[:, j] - self.radius_anchors[:, j - 1]) / span
            gap = slope * minimum_between(left, right)
            result = torch.minimum(result, torch.where(valid, gap, float('inf')))
        last = torch.gather(self.unpinned_anchors, 1, self.anchor_counts[:, None] - 1)[:, 0]
        return torch.minimum(result, minimum_between(last, torch.full_like(last, float('inf'))))


def _piecewise_triton(query, knot_x, knot_y, knot_counts, ray_id=None):
    from gap_triton import gap_triton_available, pinned_affine_map
    if not gap_triton_available(query, knot_x, knot_y):
        return None
    valid = torch.arange(knot_x.shape[1] - 1, device=knot_x.device)[None] + 1 < knot_counts[:, None]
    span = torch.where(valid, knot_x[:, 1:] - knot_x[:, :-1], 1.)
    scale = torch.where(valid, (knot_y[:, 1:] - knot_y[:, :-1]) / span, 1.)
    scale = torch.cat([scale, torch.ones_like(knot_x[:, :1])], dim=1)
    result = pinned_affine_map(query, knot_x, knot_x, knot_y, scale, knot_counts, ray_ids=ray_id)
    # This radius deformation uses unit slope also below its fixed origin.
    first_x = knot_x[:, 0] if ray_id is None else knot_x[ray_id, 0]
    first_y = knot_y[:, 0] if ray_id is None else knot_y[ray_id, 0]
    if ray_id is None and query.ndim == 2:
        first_x, first_y = first_x[:, None], first_y[:, None]
    return torch.where(query <= first_x, query + first_y - first_x, result)


def piecewise_linear_map(query, knot_x, knot_y, knot_counts):
    """Map pin-only knots with unit-slope extrapolation at both ends.

    Each row includes the origin and may contain no other valid knots.
    Padding is excluded from searches and interpolation arithmetic.
    """
    fused = _piecewise_triton(query, knot_x, knot_y, knot_counts)
    if fused is not None:
        return fused
    squeeze = query.dim() == 1
    if squeeze:
        query = query[:, None]
    valid = torch.arange(knot_x.shape[1], device=knot_x.device)[None] < knot_counts[:, None]
    search_x = torch.where(valid, knot_x.detach(), torch.full_like(knot_x, float('inf')))
    right = torch.searchsorted(search_x.contiguous(), query.detach().contiguous())
    interior = (right > 0) & (right < knot_counts[:, None])
    left = torch.minimum((right - 1).clamp(min=0), knot_counts[:, None] - 1)
    right = torch.minimum(right, knot_counts[:, None] - 1)
    x0, x1 = (torch.gather(knot_x, -1, idx) for idx in (left, right))
    y0, y1 = (torch.gather(knot_y, -1, idx) for idx in (left, right))
    span = torch.where(interior, x1 - x0, torch.ones_like(x0))
    mapped = torch.lerp(y0, y1, (query - x0) / span)
    result = torch.where(interior, mapped, query + (y0 - x0))
    return result.squeeze(-1) if squeeze else result


def piecewise_linear_map_rays(query, knot_x, knot_y, knot_counts, ray_id):
    """Evaluate shared ray knots without expanding a samples-by-knots table."""
    fused = _piecewise_triton(query, knot_x, knot_y, knot_counts, ray_id)
    if fused is not None:
        return fused
    lo = torch.zeros_like(ray_id)
    hi = knot_counts[ray_id]
    for _ in range(knot_x.shape[1].bit_length()):
        mid = (lo + hi) // 2
        value = knot_x.detach()[ray_id, mid.clamp(max=knot_x.shape[1] - 1)]
        below = (mid < knot_counts[ray_id]) & (value < query.detach())
        lo = torch.where(below, mid + 1, lo)
        hi = torch.where(below, hi, mid)
    interior = (lo > 0) & (lo < knot_counts[ray_id])
    left = torch.minimum((lo - 1).clamp(min=0), knot_counts[ray_id] - 1)
    right = torch.minimum(lo, knot_counts[ray_id] - 1)
    x0, x1 = knot_x[ray_id, left], knot_x[ray_id, right]
    y0, y1 = knot_y[ray_id, left], knot_y[ray_id, right]
    span = torch.where(interior, x1 - x0, torch.ones_like(x0))
    return torch.where(interior, torch.lerp(y0, y1, (query - x0) / span), query + y0 - x0)


def pinned_map_inverse_rays(c, ray_map, ray_id):
    """Canonical samples sharing a ray reuse its free table and pin knots."""
    table, dr = ray_map.pre_pin_winding_radii, ray_map.dr
    tn = ray_map.theta_norm[ray_id]
    inner = torch.floor((c.detach() - tn.detach() * dr.detach()).clamp(min=0) / dr.detach())
    inner = inner.long().clamp(0, table.shape[1] - 2)
    u = torch.lerp(table[ray_id, inner], table[ray_id, inner + 1],
                   (c - (inner + tn) * dr) / dr)
    return piecewise_linear_map_rays(u, ray_map.unpinned_anchors,
                                    ray_map.radius_anchors, ray_map.anchor_counts, ray_id)


def _merge_equal_targets(R, S, w, valid):
    """Combine equal canonical targets by support-weighted radius.

    Different radii at exactly the same target cannot both be satisfied by
    an invertible map. Their disagreement is reported, and zero support
    contributes neither to the representative radius nor its strength.
    """
    # Compact to the valid columns first: rays carry far fewer anchors than
    # there are winding slots, and every dense op below scales with width.
    order = torch.argsort((~valid).to(torch.int32), dim=-1, stable=True)
    count = int(valid.sum(dim=-1).max()) if valid.numel() else 0
    R, S, w, valid = (torch.gather(a, -1, order)[:, :max(count, 1)] for a in (R, S, w, valid))
    order = torch.argsort(torch.where(valid, S.detach(), float('inf')), dim=-1, stable=True)
    R, S, w, valid = (torch.gather(a, -1, order) for a in (R, S, w, valid))
    width = S.shape[-1]
    first = torch.cat([torch.ones_like(valid[:, :1]), S[:, 1:] != S[:, :-1]], dim=-1)
    group = (first.to(torch.long).cumsum(dim=-1) - 1).clamp(min=0)
    support = torch.where(valid, w, torch.zeros_like(w))
    mass = torch.zeros_like(w).scatter_add(1, group, support)
    total = torch.zeros_like(R).scatter_add(1, group, support * R)
    members = torch.zeros_like(group).scatter_add(1, group, valid.to(torch.long))
    indices = torch.arange(width, device=S.device)[None].expand_as(group)
    ref = torch.zeros_like(group).scatter_reduce(
        1, group, torch.where(valid, indices, torch.zeros_like(indices)),
        reduce='amax', include_self=True)
    target = torch.gather(S, 1, ref)
    radius = torch.where(members == 1, torch.gather(R, 1, ref), total / mass.clamp(min=torch.finfo(R.dtype).tiny))
    lo = torch.full_like(R, float('inf')).scatter_reduce(
        1, group, torch.where(valid, R, float('inf')), reduce='amin', include_self=True)
    hi = torch.full_like(R, -float('inf')).scatter_reduce(
        1, group, torch.where(valid, R, -float('inf')), reduce='amax', include_self=True)
    conflicts = ((members > 1) & (hi > lo)).sum(dim=-1)
    valid = mass > 0
    order = torch.argsort((~valid).to(torch.int32), dim=-1, stable=True)
    count = int(valid.sum(dim=-1).max()) if valid.numel() else 0
    radius, target, mass, valid = (torch.gather(a, 1, order)[:, :count]
                                  for a in (radius, target, mass, valid))
    return radius, target, mass.clamp(max=1.), valid, conflicts


def _transparent_radius_corrections(u, desired_radius, weight, valid):
    """Solve radius corrections in unpinned-radius coordinates, based at 0.

    delta_j = w_j (R_pin_j - u_j) + (1-w_j) lerp(delta_left, delta_right).
    The outer correction is constant. At zero support a knot is exactly on
    its neighbours' line, so it neither constrains nor changes the map.
    """
    u_prev = torch.cat([torch.zeros_like(u[:, :1]), u[:, :-1]], dim=-1)
    u_next = torch.cat([u[:, 1:], u[:, -1:]], dim=-1)
    next_valid = torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], dim=-1)
    span = u_next - u_prev
    lam = (u_next - u) / torch.where(span > 0, span, torch.ones_like(span))
    lam = torch.where(next_valid, lam, torch.ones_like(lam))
    w = torch.where(valid, weight, torch.ones_like(weight))
    sub = -(1. - w) * lam
    sup = -(1. - w) * (1. - lam)
    rhs = w * torch.where(valid, desired_radius - u, torch.zeros_like(u))
    from gap_triton import tridiagonal_solve
    delta = tridiagonal_solve(sub, torch.ones_like(u), sup, rhs)
    return torch.where(w >= 1., desired_radius, u + delta)


def build_pinned_ray_map(pre_pin_winding_radii, dr, theta_norm, anchor_R, anchor_S,
                         anchor_w, anchor_valid, min_gap):
    """Blend pin radii and build a deformation of the unpinned radius field.

    Pins are ordered by their canonical targets S (which may be fractional
    windings). At each S, u = R_unpinned(S). A tridiagonal solve blends the
    desired radius correction R_pin-u with the neighbouring corrections.
    Evaluation composes the original winding map with this pin-only
    deformation; original winding knots are never merged with pin knots.

    The guard clips each solved interval's radial slope, then accumulates
    the extra rise outwards. Splitting an interval with a zero-weight knot
    therefore does not change the guard or map. The slope floor relative to
    u is min_gap / dr; diagnostics measure the resulting physical gaps.
    Conflicts shift subsequent radii outwards; compatible pins remain exact.
    """
    num_rays = pre_pin_winding_radii.shape[0]
    device = pre_pin_winding_radii.device
    zero = torch.zeros_like(theta_norm)
    base_c = unpinned_map_forward(zero, pre_pin_winding_radii, dr, theta_norm)
    supported = anchor_valid & (anchor_w > 0)
    base_conflicts = (supported & ((anchor_S < base_c[:, None])
                                   | ((anchor_S == base_c[:, None]) & (anchor_R != 0)))).sum(dim=-1)
    valid = supported & (anchor_S > base_c[:, None])
    if anchor_S.shape[-1]:
        R, S, w, valid, duplicate_conflicts = _merge_equal_targets(
            anchor_R, anchor_S, anchor_w, valid)
    else:
        R, S, w = anchor_R, anchor_S, anchor_w
        duplicate_conflicts = torch.zeros_like(base_conflicts)
    if S.shape[-1] == 0:
        empty = torch.zeros([num_rays, 0], dtype=torch.bool, device=device)
        return PinnedRayMap(
            pre_pin_winding_radii, dr, theta_norm, zero[:, None], zero[:, None],
            torch.ones([num_rays], dtype=torch.long, device=device),
            S, R, empty, empty, empty, duplicate_conflicts, base_conflicts)

    S = torch.where(valid, S, base_c[:, None])
    u = unpinned_map_inverse(S, pre_pin_winding_radii, dr, theta_norm)
    u = torch.where(valid, u, torch.zeros_like(u))
    solved = _transparent_radius_corrections(u, R, w, valid)
    prev_u = torch.cat([zero[:, None], u[:, :-1]], dim=-1)
    prev_radius = torch.cat([zero[:, None], solved[:, :-1]], dim=-1)
    rise = solved - prev_radius
    min_ratio = float(min_gap) / dr
    min_rise = min_ratio * (u - prev_u)
    guard = valid & (rise < min_rise)
    extra = torch.where(guard, min_rise - rise, torch.zeros_like(rise)).cumsum(dim=-1)
    resolved = solved + extra
    return PinnedRayMap(
        pre_pin_winding_radii, dr, theta_norm,
        torch.cat([zero[:, None], u], dim=-1),
        torch.cat([zero[:, None], resolved], dim=-1), valid.sum(dim=-1) + 1,
        S, resolved, valid, guard, valid & (rise <= 0), duplicate_conflicts, base_conflicts)


def pinned_map_forward(r, ray_map):
    """Intermediate -> unpinned radius -> canonical coordinate."""
    u = piecewise_linear_map(r, ray_map.radius_anchors, ray_map.unpinned_anchors,
                             ray_map.anchor_counts)
    return unpinned_map_forward(u, ray_map.pre_pin_winding_radii,
                                ray_map.dr, ray_map.theta_norm)


def pinned_map_inverse(c, ray_map):
    """Canonical coordinate -> unpinned radius -> intermediate radius."""
    u = unpinned_map_inverse(c, ray_map.pre_pin_winding_radii,
                             ray_map.dr, ray_map.theta_norm)
    return piecewise_linear_map(u, ray_map.unpinned_anchors, ray_map.radius_anchors,
                                ray_map.anchor_counts)


# ---------------------------------------------------------------------------
# Pin lookup per ray
# ---------------------------------------------------------------------------


def compact_kernel(d):
    """``K(d) = (1 - d) / (d + tiny)`` for ``d < 1``, else 0."""
    inside = d < 1.
    safe_d = torch.where(inside, d, torch.ones_like(d))
    return torch.where(inside, (1. - safe_d) / (safe_d + KERNEL_TINY), torch.zeros_like(d))


class PinCells:
    """Compressed-sparse-row binning of pins over ``(theta_bin, z_bin)`` cells.

    Pins are sorted by cell id; ``offsets[c]..offsets[c+1]`` is cell ``c``'s
    pin range. The theta axis wraps. Cell widths are chosen from the largest
    footprint so a query's neighbourhood is a fixed ``(2 rt + 1) x (2 rz + 1)``
    block of cells. Nothing is ever dropped: every pin is in exactly one
    cell and every query gathers every pin whose footprint can reach it.
    """

    def __init__(self, theta, z, eps_theta, eps_z, min_z, max_z,
                 max_bins=4096):
        device = theta.device
        self.min_z = float(min_z)
        self.max_z = float(max_z)
        self.z_range = max(self.max_z - self.min_z, 1.0e-6)
        max_eps_theta = float(eps_theta.max()) if eps_theta.numel() else TWO_PI
        max_eps_z = float(eps_z.max()) if eps_z.numel() else self.z_range
        # Two bins per max footprint: a footprint reaches at most 2 cells
        # away from the centre, and cells stay small enough that dense
        # patches are spread across several of them.
        self.n_theta = int(min(max_bins, max(1, math.floor(TWO_PI / max(max_eps_theta / 2., 1.0e-9)))))
        self.n_z = int(min(max_bins, max(1, math.floor(self.z_range / max(max_eps_z / 2., 1.0e-9)))))
        self.theta_width = TWO_PI / self.n_theta
        self.z_width = self.z_range / self.n_z
        self.radius_theta = int(math.ceil(max_eps_theta / self.theta_width)) if eps_theta.numel() else 0
        self.radius_z = int(math.ceil(max_eps_z / self.z_width)) if eps_z.numel() else 0
        # A full wrap needs no more than all theta bins.
        self.radius_theta = min(self.radius_theta, self.n_theta // 2)
        self.radius_z = min(self.radius_z, self.n_z)
        tb, zb = self.bin_of(theta, z)
        cell = tb * self.n_z + zb
        self.order = torch.argsort(cell)
        sorted_cell = cell[self.order]
        counts = torch.bincount(sorted_cell, minlength=self.n_theta * self.n_z)
        self.offsets = torch.cat([
            torch.zeros([1], dtype=torch.int64, device=device),
            torch.cumsum(counts, dim=0)])
        self.num_pins = int(theta.shape[0])

    def bin_of(self, theta, z):
        tb = torch.floor(theta / self.theta_width).to(torch.int64) % self.n_theta
        zb = torch.floor((z - self.min_z) / self.z_width).to(torch.int64).clamp(0, self.n_z - 1)
        return tb, zb

    def gather_pairs(self, theta_q, z_q):
        """All (query, sorted-pin) pairs in the queries' neighbourhood cells.

        Returns ``(query_idx, sorted_pin_idx)`` as int64 tensors.
        """
        device = theta_q.device
        num_queries = theta_q.shape[0]
        if self.num_pins == 0 or num_queries == 0:
            empty = torch.empty([0], dtype=torch.int64, device=device)
            return empty, empty
        tb, zb = self.bin_of(theta_q, z_q)
        dts = torch.arange(-self.radius_theta, self.radius_theta + 1, device=device)
        dzs = torch.arange(-self.radius_z, self.radius_z + 1, device=device)
        cell_tb = (tb[:, None, None] + dts[None, :, None]) % self.n_theta
        cell_zb = zb[:, None, None] + dzs[None, None, :]
        in_range = ((cell_zb >= 0) & (cell_zb < self.n_z)).expand(
            num_queries, dts.numel(), dzs.numel())
        cell = (cell_tb * self.n_z + cell_zb.clamp(0, self.n_z - 1)).reshape(-1)
        in_range = in_range.reshape(-1)
        query_of_cell = torch.arange(num_queries, device=device)[:, None, None].expand(
            -1, dts.numel(), dzs.numel()).reshape(-1)
        begin = self.offsets[cell]
        end = self.offsets[cell + 1]
        count = torch.where(in_range, end - begin, torch.zeros_like(begin))
        total = int(count.sum().item())
        if total == 0:
            empty = torch.empty([0], dtype=torch.int64, device=device)
            return empty, empty
        pair_cell = torch.repeat_interleave(torch.arange(count.numel(), device=device), count)
        starts = torch.cumsum(count, dim=0) - count
        within = torch.arange(total, device=device) - starts[pair_cell]
        pin_idx = begin[pair_cell] + within
        return query_of_cell[pair_cell], pin_idx


def _accumulate_anchor_sums(theta_q, z_q, cells, pin_theta, pin_z, pin_eps_theta,
                            pin_eps_z, pin_slot, pin_r, pin_target_shifted, dr,
                            num_slots, mass, R_sum, S_sum):
    """Add one cell structure's kernel sums into the per-(query, slot) totals.

    A pin's winding slot and target are expressed **relative to the query
    ray**: a pin on the far side of the theta = 0 seam from the query is the
    same sheet one raw winding over, so its slot shifts by the seam step
    ``round((theta_i - theta_q) / 2pi)`` and its shifted target by ``dr``
    times that step. The anchor target is then the IDW of the pins' shifted
    targets plus the query's own ``dr theta_q / 2pi`` -- continuous across the
    seam, where a per-pin target at the pin's own angle would not be.
    """
    num_queries = theta_q.shape[0]
    dtype = mass.dtype
    from gap_triton import anchor_sums
    fused = anchor_sums(
        theta_q, z_q, pin_theta, pin_z, pin_r, pin_target_shifted, dr,
        offsets=cells.offsets, pin_eps_theta=pin_eps_theta, pin_eps_z=pin_eps_z,
        pin_slot=pin_slot, n_theta=cells.n_theta, n_z=cells.n_z,
        theta_width=cells.theta_width, z_width=cells.z_width, min_z=cells.min_z,
        radius_theta=cells.radius_theta, radius_z=cells.radius_z, num_slots=num_slots) \
        if torch.is_tensor(dr) else None
    if fused is not None:
        return mass + fused[0], R_sum + fused[1], S_sum + fused[2]
    q_idx, p_idx = cells.gather_pairs(theta_q.detach(), z_q.detach())
    if q_idx.numel() == 0:
        return mass, R_sum, S_sum
    delta = theta_q[q_idx] - pin_theta[p_idx]
    wrap = torch.round(delta.detach() / TWO_PI)          # -1, 0, +1
    u = (delta - wrap * TWO_PI) / pin_eps_theta[p_idx]
    v = (z_q[q_idx] - pin_z[p_idx]) / pin_eps_z[p_idx]
    # The tiny offset keeps sqrt's backward finite on a pin's own ray
    # (0/0 otherwise); it moves d by far less than KERNEL_TINY matters.
    d = torch.sqrt(u * u + v * v + 1.0e-30)
    k = compact_kernel(d).to(dtype)
    # Seen from the query, the pin's raw winding is n_i - wrap.
    slot = pin_slot[p_idx] - wrap.to(torch.int64)
    keep = (k > 0) & (slot >= 0) & (slot < num_slots)
    q_idx, p_idx, k, slot, wrap = q_idx[keep], p_idx[keep], k[keep], slot[keep], wrap[keep]
    flat = q_idx * num_slots + slot
    target = pin_target_shifted[p_idx] - dr * wrap.to(dtype)
    mass = mass.reshape(-1).index_add(0, flat, k).reshape(num_queries, num_slots)
    R_sum = R_sum.reshape(-1).index_add(0, flat, k * pin_r[p_idx]).reshape(num_queries, num_slots)
    S_sum = S_sum.reshape(-1).index_add(0, flat, k * target).reshape(num_queries, num_slots)
    return mass, R_sum, S_sum


def _anchors_from_sums(mass, R_sum, S_sum, theta_q, dr):
    valid = mass > 0
    safe_mass = torch.where(valid, mass, torch.ones_like(mass))
    R = torch.where(valid, R_sum / safe_mass, torch.zeros_like(R_sum))
    S = torch.where(valid, S_sum / safe_mass + (dr * theta_q / TWO_PI)[:, None], torch.zeros_like(S_sum))
    w = torch.clamp(mass, max=1.)
    return R, S, w, valid


def ray_anchors(theta_q, z_q, cells, pin_theta, pin_z, pin_eps_theta, pin_eps_z,
                pin_slot, pin_r, pin_target_shifted, dr, num_slots):
    """Per-ray anchors ``(R_k, S_k, w_k, valid_k)`` over winding slots.

    All ``pin_*`` arrays are in ``cells``' sorted order; ``pin_target_shifted``
    is ``dr (T + n_i)`` (no angular term). Returns dense ``[N, num_slots]``
    tensors; slots with zero kernel mass are invalid and are never divided
    (0/0 is masked before the division).
    """
    num_queries = theta_q.shape[0]
    mass = torch.zeros([num_queries, num_slots], dtype=pin_r.dtype, device=theta_q.device)
    R_sum = torch.zeros_like(mass)
    S_sum = torch.zeros_like(mass)
    mass, R_sum, S_sum = _accumulate_anchor_sums(
        theta_q, z_q, cells, pin_theta, pin_z, pin_eps_theta, pin_eps_z,
        pin_slot, pin_r, pin_target_shifted, dr, num_slots, mass, R_sum, S_sum)
    return _anchors_from_sums(mass, R_sum, S_sum, theta_q, dr)


# ---------------------------------------------------------------------------
# The per-step pin table
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PinFootprints:
    eps_theta: torch.Tensor
    eps_z: torch.Tensor


@dataclasses.dataclass
class CoincidenceGroups:
    """Result of the coincident-pin compatibility pass.

    ``group`` maps each registry pin to its rasterised pin; ``num_groups`` is
    the rasterised count. ``conflict_groups`` lists conflicting groups with a
    tag ('fold' when every member shares component and ``n``, else
    'cross_component').
    """

    group: torch.Tensor
    num_groups: int
    conflicts: list

    @property
    def num_conflicts(self):
        return len(self.conflicts)


def compute_coincidence_groups(theta, z, slot, component, n, r, eps_theta,
                               eps_z, min_z, max_z, local_gap, *,
                               coincidence_frac, conflict_tolerance, kinds=None,
                               chunk_size=1 << 20):
    """Merge pins in the same slot on (nearly) the same ray.

    Two pins are coincident when their ``(theta, z)`` distance, in each pin's
    own footprint units, is below ``coincidence_frac``. Coincident pins whose
    radii differ by less than ``conflict_tolerance * local_gap`` are
    compatible; otherwise the group is a conflict (still averaged here).
    """
    device = theta.device
    num = theta.shape[0]
    group = torch.arange(num, device=device)
    if num == 0:
        return CoincidenceGroups(group, 0, [])
    # Candidate pairs per footprint class (cells sized by that class's
    # coincidence radius) and per query chunk, so dense registries never
    # materialise one huge pair list. A pair must be within the coincidence
    # fraction of BOTH footprints, so gathering against the smaller class's
    # radius loses nothing.
    class_theta = torch.ceil(torch.log2(eps_theta / eps_theta.min().clamp(min=1e-9))).to(torch.int64)
    class_z = torch.ceil(torch.log2(eps_z / eps_z.min().clamp(min=1e-9))).to(torch.int64)
    key = class_theta * 64 + class_z
    pair_q, pair_p = [], []
    for value in torch.unique(key).tolist():
        members = torch.nonzero(key == value, as_tuple=True)[0]
        cells = PinCells(theta[members], z[members], eps_theta[members] * coincidence_frac,
                         eps_z[members] * coincidence_frac, min_z, max_z)
        member_of_sorted = members[cells.order]
        for start in range(0, num, chunk_size):
            q_local = torch.arange(start, min(num, start + chunk_size), device=device)
            q_idx, p_sorted = cells.gather_pairs(theta[q_local], z[q_local])
            if q_idx.numel() == 0:
                continue
            q_idx = q_local[q_idx]
            p_idx = member_of_sorted[p_sorted]
            wrap = torch.round((theta[q_idx] - theta[p_idx]) / TWO_PI).to(torch.int64)
            pair = (q_idx < p_idx) & (slot[q_idx] == slot[p_idx] - wrap)
            q_idx, p_idx = q_idx[pair], p_idx[pair]
            if q_idx.numel() == 0:
                continue
            dtheta = wrap_angle(theta[q_idx] - theta[p_idx])
            dz = z[q_idx] - z[p_idx]
            d_a = torch.sqrt((dtheta / eps_theta[q_idx]) ** 2 + (dz / eps_z[q_idx]) ** 2)
            d_b = torch.sqrt((dtheta / eps_theta[p_idx]) ** 2 + (dz / eps_z[p_idx]) ** 2)
            close = torch.maximum(d_a, d_b) < coincidence_frac
            pair_q.append(q_idx[close])
            pair_p.append(p_idx[close])
    if pair_q:
        q_idx = torch.cat(pair_q)
        p_idx = torch.cat(pair_p)
    else:
        q_idx = p_idx = torch.empty([0], dtype=torch.int64, device=device)
    if q_idx.numel() == 0:
        return CoincidenceGroups(group, num, [])
    import scipy.sparse
    import scipy.sparse.csgraph
    rows = q_idx.cpu().numpy()
    cols = p_idx.cpu().numpy()
    adjacency = scipy.sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num, num))
    num_groups, labels = scipy.sparse.csgraph.connected_components(
        adjacency, directed=False)
    labels = labels.astype(np.int64)
    labels_t = torch.from_numpy(labels).to(device)
    counts = np.bincount(labels, minlength=num_groups)
    conflicts = []
    if (counts > 1).any():
        # Host-side grouping of the multi-member groups only (no per-pin
        # device round trips), then per-group conflict checks in numpy.
        multi_pins = np.nonzero(counts[labels] > 1)[0]
        multi_pins = multi_pins[np.argsort(labels[multi_pins], kind='stable')]
        multi_labels = labels[multi_pins]
        boundaries = np.nonzero(np.diff(multi_labels))[0] + 1
        member_lists = np.split(multi_pins, boundaries)
        r_cpu = r.detach().cpu().numpy()
        gap_cpu = local_gap.detach().cpu().numpy()
        comp_cpu = component.cpu().numpy()
        n_cpu = n.cpu().numpy()
        kinds_cpu = kinds.cpu().numpy() if kinds is not None else None
        # Vectorised spread/tolerance per group; only conflicts are listed.
        group_max = np.full(num_groups, -np.inf)
        group_min = np.full(num_groups, np.inf)
        group_gap = np.full(num_groups, np.inf)
        np.maximum.at(group_max, multi_labels, r_cpu[multi_pins])
        np.minimum.at(group_min, multi_labels, r_cpu[multi_pins])
        np.minimum.at(group_gap, multi_labels, gap_cpu[multi_pins])
        for members in member_lists:
            label = int(labels[members[0]])
            spread = float(group_max[label] - group_min[label])
            tolerance = conflict_tolerance * float(group_gap[label])
            if spread < tolerance:
                continue
            members = [int(m) for m in members]
            same = (len({int(comp_cpu[m]) for m in members}) == 1
                    and len({int(n_cpu[m]) for m in members}) == 1)
            conflicts.append({
                'group': label,
                'members': list(members),
                'components': sorted({int(comp_cpu[m]) for m in members}),
                'n': sorted({int(n_cpu[m]) for m in members}),
                'kinds': sorted({PIN_KIND_NAMES[int(kinds_cpu[m])] for m in members}) if kinds_cpu is not None else [],
                'radius_spread': spread,
                'tag': 'fold' if same else 'cross_component',
            })
    return CoincidenceGroups(labels_t, int(num_groups), conflicts)


def _segment_mean(values, group, num_groups):
    counts = torch.bincount(group, minlength=num_groups).to(values.dtype)
    sums = torch.zeros([num_groups], dtype=values.dtype, device=values.device).index_add(0, group, values)
    return sums / counts.clamp(min=1.)


def _segment_max(values, group, num_groups):
    out = torch.full([num_groups], -float('inf'), dtype=values.dtype, device=values.device)
    return out.scatter_reduce(0, group, values, reduce='amax', include_self=True)


class PinTable:
    """The rasterised pin set for one transform instance.

    Built from the per-step differentiable pins ``(z, theta, r,
    r_target_shifted)`` (the last being ``dr (T + n_i)``, without the angular
    term) plus the registry's static per-pin data. Coincidence groups are merged
    here (segment means of the differentiable values). The merged pins are
    then binned into :class:`PinCells` **per footprint class** (octaves of
    ``eps_theta`` and ``eps_z``): one cell structure sized from the largest
    footprint would make every query gather the dense small-footprint pins
    of whole cells that cannot reach it, so each class has cells matched to
    its own footprints and a query gathers only pins that can plausibly
    touch it. The classes accumulate into the same per-(ray, slot) kernel
    sums, so the result is identical to a single table.
    ``anchors(theta, z)`` serves the transform's per-ray lookup.
    """

    def __init__(self, z, theta, r, r_target_shifted, slot, eps_theta, eps_z,
                 num_slots, min_z, max_z, dr, groups=None, num_registry_pins=None, layout=None):
        device = theta.device
        self.dr = dr
        num_pins = theta.shape[0]
        if groups is None:
            group = torch.arange(num_pins, device=device)
            num_groups = num_pins
        else:
            group, num_groups = groups.group, groups.num_groups
        # The differentiable inputs are kept as given; every graph operation
        # on them (coincidence merge, per-class reordering, the kernel) runs
        # inside anchors(), so each transform evaluation owns its own graph
        # and one table can serve many independent loss-family backwards.
        # Only the cell binning uses detached coordinates.
        self._group = group
        self._num_groups = int(num_groups)
        self._merge = num_groups != num_pins
        with torch.no_grad():
            if self._merge:
                # Merge coincident pins. Members may sit on opposite sides of
                # the theta = 0 seam (the same sheet, raw winding differing by
                # one), so express every member's slot and shifted target in
                # the group's reference-angle frame before averaging.
                theta_c = _segment_mean(torch.cos(theta), group, num_groups)
                theta_s = _segment_mean(torch.sin(theta), group, num_groups)
                theta_ref = torch.atan2(theta_s, theta_c) % TWO_PI
                wrap = torch.round((theta_ref[group] - theta.detach()) / TWO_PI)
                slot_norm = slot - wrap.to(torch.int64)
                slot_out = torch.zeros([num_groups], dtype=slot.dtype, device=device)
                slot_out.scatter_(0, group, slot_norm)
                eps_theta = _segment_max(eps_theta, group, num_groups)
                eps_z = _segment_max(eps_z, group, num_groups)
                self._merge_wrap = wrap
                self._merge_theta_ref = theta_ref
                slot = slot_out
            else:
                self._merge_wrap = None
        self._z_source = z
        self._theta_source = theta
        self._r_source = r
        self._target_source = r_target_shifted
        self.num_slots = int(num_slots)
        self.num_registry_pins = int(num_pins if num_registry_pins is None else num_registry_pins)
        self.num_pins = int(num_groups)
        self.min_z, self.max_z = float(min_z), float(max_z)
        with torch.no_grad():
            theta_d, z_d, _, _ = self._values()
        slot = slot.clamp(0, self.num_slots - 1)
        if layout is not None:
            self.classes = layout
            return
        # Footprint classes: octaves of each width, keyed jointly.
        if self.num_pins:
            class_theta = torch.ceil(torch.log2(eps_theta / eps_theta.min().clamp(min=1e-9))).to(torch.int64)
            class_z = torch.ceil(torch.log2(eps_z / eps_z.min().clamp(min=1e-9))).to(torch.int64)
            key = class_theta * 64 + class_z
            keys = torch.unique(key)
        else:
            key = torch.zeros([0], dtype=torch.int64, device=device)
            keys = key
        self.classes = []
        for value in keys.tolist():
            members = torch.nonzero(key == value, as_tuple=True)[0]
            cells = PinCells(theta_d[members], z_d[members], eps_theta[members], eps_z[members],
                             self.min_z, self.max_z)
            order = members[cells.order]
            self.classes.append({
                'cells': cells, 'order': order,
                'eps_theta': eps_theta[order], 'eps_z': eps_z[order],
                'slot': slot[order],
                'bin_theta': cells.bin_of(theta_d[order], z_d[order])[0],
                'bin_z': cells.bin_of(theta_d[order], z_d[order])[1],
            })

    def _values(self):
        """Merged (theta, z, r, r_target_shifted), a fresh graph per call."""
        if not self._merge:
            return self._theta_source, self._z_source, self._r_source, self._target_source
        group, num_groups = self._group, self._num_groups
        wrap = self._merge_wrap
        # Unwrap members to the reference angle so the circular mean is a
        # plain mean (differentiable). Left in the reference frame (it may
        # sit just outside [0, 2pi)): the merged slot was normalised to that
        # frame, and the lookup and the cells wrap angles themselves.
        theta = _segment_mean(self._theta_source + wrap * TWO_PI, group, num_groups)
        z = _segment_mean(self._z_source, group, num_groups)
        r = _segment_mean(self._r_source, group, num_groups)
        target = _segment_mean(self._target_source - self.dr * wrap.to(self._target_source.dtype),
                               group, num_groups)
        return theta, z, r, target

    def anchors(self, theta_q, z_q):
        num_queries = theta_q.shape[0]
        theta, z, r, r_target = self._values()
        mass = torch.zeros([num_queries, self.num_slots], dtype=r.dtype, device=theta_q.device)
        R_sum = torch.zeros_like(mass)
        S_sum = torch.zeros_like(mass)
        for c in self.classes:
            order = c['order']
            mass, R_sum, S_sum = _accumulate_anchor_sums(
                theta_q, z_q, c['cells'], theta[order], z[order], c['eps_theta'], c['eps_z'],
                c['slot'], r[order], r_target[order], self.dr, self.num_slots, mass, R_sum, S_sum)
        return _anchors_from_sums(mass, R_sum, S_sum, theta_q, self.dr)


# ---------------------------------------------------------------------------
# Footprints (per-pin kernel widths)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FootprintRule:
    spacing_factor: float = 1.5
    min_arc_voxels: float = 3.0
    max_theta_radians: float = 0.25
    min_z_voxels: float = 3.0
    max_z_voxels: float = 200.0

    def apply(self, spacing_theta, spacing_z, radius):
        """Footprint widths from own-object spacings (nan = no neighbour)."""
        min_theta = self.min_arc_voxels / radius.clamp(min=1.0)
        eps_theta = torch.nan_to_num(spacing_theta * self.spacing_factor, nan=0.0)
        eps_z = torch.nan_to_num(spacing_z * self.spacing_factor, nan=0.0)
        eps_theta = torch.maximum(eps_theta, min_theta).clamp(max=self.max_theta_radians)
        eps_z = eps_z.clamp(min=self.min_z_voxels, max=self.max_z_voxels)
        return eps_theta, eps_z


def grid_neighbour_spacing(theta_grid, z_grid, valid):
    """Own-object spacing of quad-centre pins on a 2-D grid.

    For each valid cell, the largest wrapped ``|dtheta|`` and ``|dz|`` to
    its valid 4-neighbours (nan where a cell has no valid neighbour). Torch
    tensors ``[H, W]``.
    """
    nan = torch.full_like(theta_grid, float('nan'))
    theta_v = torch.where(valid, theta_grid, nan)
    z_v = torch.where(valid, z_grid, nan)
    best_t = torch.zeros_like(theta_grid)
    best_z = torch.zeros_like(theta_grid)
    any_neighbour = torch.zeros_like(valid)
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        nt = torch.roll(theta_v, shifts=shift, dims=axis)
        nz = torch.roll(z_v, shifts=shift, dims=axis)
        nvalid = torch.roll(valid, shifts=shift, dims=axis)
        # roll wraps around; kill the wrapped row/column.
        index = torch.arange(theta_grid.shape[axis], device=theta_grid.device)
        edge = (index < shift) if shift > 0 else (index >= theta_grid.shape[axis] + shift)
        edge = edge.view(-1, 1) if axis == 0 else edge.view(1, -1)
        nvalid = nvalid & ~edge
        dt = wrap_angle(nt - theta_v).abs()
        dz = (nz - z_v).abs()
        dt = torch.where(nvalid, dt, torch.zeros_like(dt))
        dz = torch.where(nvalid, dz, torch.zeros_like(dz))
        best_t = torch.maximum(best_t, torch.nan_to_num(dt, nan=0.0))
        best_z = torch.maximum(best_z, torch.nan_to_num(dz, nan=0.0))
        any_neighbour |= nvalid & valid
    best_t = torch.where(any_neighbour, best_t, nan)
    best_z = torch.where(any_neighbour, best_z, nan)
    return best_t, best_z


def chain_neighbour_spacing(theta, z):
    """Own-object spacing along a 1-D chain: max over both chain neighbours."""
    n = theta.shape[0]
    nan = torch.full_like(theta, float('nan'))
    if n < 2:
        return nan, nan
    dt = wrap_angle(theta[1:] - theta[:-1]).abs()
    dz = (z[1:] - z[:-1]).abs()
    pad = torch.zeros([1], dtype=theta.dtype, device=theta.device)
    dt_prev = torch.cat([pad, dt])
    dt_next = torch.cat([dt, pad])
    dz_prev = torch.cat([pad, dz])
    dz_next = torch.cat([dz, pad])
    return torch.maximum(dt_prev, dt_next), torch.maximum(dz_prev, dz_next)


# ---------------------------------------------------------------------------
# The constraint registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _GraphNode:
    """One node of the constraint graph.

    ``kind``: 'patch' (a whole verified patch, one node), 'point' (a PCL,
    strip or attached point). ``zyx`` for points; ``patch_index`` for
    patches. ``winding`` is the point's relative winding annotation.
    """

    kind: str
    label: str
    zyx: np.ndarray | None = None
    patch_index: int | None = None
    winding: float = 0.0
    is_pin: bool = True
    pin_kind: int = PIN_KIND_CHAIN
    chain_prev: int | None = None
    chain_next: int | None = None
    attach_ij: tuple | None = None    # (i, j) of the attached quad on patch_index


@dataclasses.dataclass
class _GraphEdge:
    """``n_v = n_u + delta(u, v)`` where ``delta`` is resolved at finalisation.

    ``kind`` 'chain': consecutive chain points (or a link junction); the
    delta is ``(w_v - w_u) - step(theta_u -> theta_v)``. ``kind`` 'attach':
    ``u`` is a patch, ``v`` an attached point with ``attach_ij``; the delta is
    ``-pot_v`` (the quad potential transported to the point). ``kind``
    'overlap': ``u`` and ``v`` are two patches whose quad centres ``ij_u`` /
    ``ij_v`` coincide physically (the same sheet observed twice), so those
    two pins share one shifted radius: ``O_v - O_u = pot_v - pot_u -
    step(theta_u -> theta_v)``, integers from the theta topology only.
    """

    u: int
    v: int
    kind: str
    label: str = ''
    ij_u: tuple | None = None
    ij_v: tuple | None = None


@dataclasses.dataclass
class _AbsoluteAnchor:
    node: int
    winding: float
    label: str


class PinGraph:
    """The model-independent constraint graph.

    Built once from the fit inputs (verified patches, cross-patch PCLs,
    unattached strips and their link components). Nodes are whole patches
    and individual points; edges carry the integer relations described in
    :class:`_GraphEdge`. Components are connected components of this graph
    (deterministically numbered by their smallest node label), so the
    component count -- the size of ``T`` -- is known before any model exists.
    """

    def __init__(self):
        self.nodes: list[_GraphNode] = []
        self.edges: list[_GraphEdge] = []
        self.absolute: list[_AbsoluteAnchor] = []
        self.patch_nodes: dict[int, int] = {}
        self.patch_grid_stride = 1
        self._component_of = None
        self._num_components = None
        self._component_labels = None

    # -- construction -----------------------------------------------------

    def add_patch(self, patch_index, label):
        node = len(self.nodes)
        self.nodes.append(_GraphNode(kind='patch', label=label, patch_index=patch_index,
                                     is_pin=True, pin_kind=PIN_KIND_PATCH))
        self.patch_nodes[patch_index] = node
        return node

    def add_point(self, zyx, label, winding=0.0, is_pin=True, attach_ij=None,
                  patch_index=None):
        node = len(self.nodes)
        attached = (patch_index is not None and patch_index in self.patch_nodes
                    and attach_ij is not None)
        self.nodes.append(_GraphNode(
            kind='point', label=label, zyx=np.asarray(zyx, dtype=np.float32),
            winding=float(winding), is_pin=is_pin,
            attach_ij=tuple(float(v) for v in attach_ij) if attached else None,
            patch_index=patch_index if attached else None))
        if attached:
            self.edges.append(_GraphEdge(self.patch_nodes[patch_index], node, 'attach', label))
        return node

    def add_overlap_edge(self, patch_index_u, patch_index_v, ij_u, ij_v, label=''):
        u = self.patch_nodes[patch_index_u]
        v = self.patch_nodes[patch_index_v]
        if u != v:
            self.edges.append(_GraphEdge(
                u, v, 'overlap', label, ij_u=(int(ij_u[0]), int(ij_u[1])),
                ij_v=(int(ij_v[0]), int(ij_v[1]))))

    def add_chain_edge(self, u, v, label=''):
        if u != v:
            self.edges.append(_GraphEdge(u, v, 'chain', label))
            if self.nodes[u].chain_next is None and self.nodes[v].chain_prev is None:
                self.nodes[u].chain_next = v
                self.nodes[v].chain_prev = u

    def add_absolute(self, node, winding, label=''):
        self.absolute.append(_AbsoluteAnchor(node, float(winding), label))

    # -- components -------------------------------------------------------

    def _components(self):
        if self._component_of is not None:
            return
        import scipy.sparse
        import scipy.sparse.csgraph
        num = len(self.nodes)
        if num == 0:
            self._component_of = np.zeros([0], dtype=np.int64)
            self._num_components = 0
            self._component_labels = []
            return
        rows = np.asarray([e.u for e in self.edges], dtype=np.int64)
        cols = np.asarray([e.v for e in self.edges], dtype=np.int64)
        adjacency = scipy.sparse.coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num, num))
        count, labels = scipy.sparse.csgraph.connected_components(adjacency, directed=False)
        # Deterministic numbering: by the smallest node label in each component.
        first_label = {}
        for node, comp in enumerate(labels):
            label = self.nodes[node].label
            if comp not in first_label or label < first_label[comp]:
                first_label[comp] = label
        order = sorted(range(count), key=lambda comp: first_label[comp])
        remap = np.empty(count, dtype=np.int64)
        remap[np.asarray(order)] = np.arange(count)
        self._component_of = remap[labels]
        self._num_components = int(count)
        self._component_labels = [first_label[comp] for comp in order]

    @property
    def num_components(self):
        self._components()
        return self._num_components

    @property
    def component_of(self):
        self._components()
        return self._component_of

    def fingerprint(self):
        """Stable identity of the component structure (for checkpoints)."""
        self._components()
        digest = hashlib.sha256()
        for comp in range(self._num_components):
            members = sorted(self.nodes[n].label for n in np.nonzero(self._component_of == comp)[0])
            digest.update(('|'.join(members) + '\n').encode('utf-8'))
        return digest.hexdigest()

    def summary(self):
        counts = collections.Counter(node.kind for node in self.nodes)
        return (f'pin graph: {len(self.nodes)} nodes ({counts.get("patch", 0)} patches, '
                f'{counts.get("point", 0)} points), {len(self.edges)} edges, '
                f'{self.num_components} components, {len(self.absolute)} absolute anchors')


@dataclasses.dataclass
class PinRegistry:
    """Flat per-pin tensors, finalised against one model state.

    ``zyx`` scroll-space positions; ``component`` ``g(i)``; ``n0`` the
    integer offset at construction; ``theta0`` the intermediate-space angle
    at construction; ``eps_theta`` / ``eps_z`` footprints; ``kind`` object
    tag. ``fixed_T`` / ``fixed_T_value`` mark absolute-winding components.
    """

    zyx: torch.Tensor
    component: torch.Tensor
    n0: torch.Tensor
    theta0: torch.Tensor
    eps_theta: torch.Tensor
    eps_z: torch.Tensor
    kind: torch.Tensor
    num_components: int
    fixed_T: torch.Tensor
    fixed_T_value: torch.Tensor
    initial_T: torch.Tensor
    fingerprint: str
    consistency_report: dict
    local_gap: torch.Tensor  # free winding gap near each pin at construction
    # Per verified patch (atlas order): its component and offset O_P such
    # that the patch's root-frame unwrapped winding is T_g + O_P; -1 / 0 when
    # the patch contributed no pins.
    patch_component: torch.Tensor
    patch_offset: torch.Tensor
    neighbours: torch.Tensor | None = None  # own-object grid/chain adjacency, -1 = absent

    @property
    def num_pins(self):
        return int(self.zyx.shape[0])

    def to(self, device):
        return PinRegistry(**{
            f.name: (getattr(self, f.name).to(device) if torch.is_tensor(getattr(self, f.name))
                     else getattr(self, f.name))
            for f in dataclasses.fields(self)})

    def state_dict(self):
        return {
            'zyx': self.zyx.cpu(), 'component': self.component.cpu(), 'n0': self.n0.cpu(),
            'theta0': self.theta0.cpu(), 'eps_theta': self.eps_theta.cpu(),
            'eps_z': self.eps_z.cpu(), 'kind': self.kind.cpu(),
            'num_components': self.num_components, 'fixed_T': self.fixed_T.cpu(),
            'fixed_T_value': self.fixed_T_value.cpu(), 'initial_T': self.initial_T.cpu(),
            'fingerprint': self.fingerprint, 'local_gap': self.local_gap.cpu(),
            'patch_component': self.patch_component.cpu(),
            'patch_offset': self.patch_offset.cpu(),
            'neighbours': self.neighbours.cpu() if self.neighbours is not None else None,
        }

    @classmethod
    def from_state_dict(cls, state, device='cpu'):
        return cls(
            zyx=state['zyx'].to(device), component=state['component'].to(device),
            n0=state['n0'].to(device), theta0=state['theta0'].to(device),
            eps_theta=state['eps_theta'].to(device), eps_z=state['eps_z'].to(device),
            kind=state['kind'].to(device), num_components=int(state['num_components']),
            fixed_T=state['fixed_T'].to(device), fixed_T_value=state['fixed_T_value'].to(device),
            initial_T=state['initial_T'].to(device), fingerprint=str(state['fingerprint']),
            consistency_report={}, local_gap=state['local_gap'].to(device),
            patch_component=state['patch_component'].to(device),
            patch_offset=state['patch_offset'].to(device),
            neighbours=state['neighbours'].to(device) if state.get('neighbours') is not None else None)

    def adjusted_n(self, theta):
        """``n_i(t) = n_i(0) + round((theta0_i - theta_i(t)) / 2pi)``."""
        return self.n0 + torch.round((self.theta0 - theta) / TWO_PI).to(torch.int32)

    def summary(self):
        counts = torch.bincount(self.component, minlength=self.num_components)
        kinds = torch.bincount(self.kind, minlength=3).tolist()
        counts_np = counts.cpu().numpy()
        lines = [
            f'pin registry: {self.num_pins} pins in {self.num_components} components '
            f'({kinds[0]} patch, {kinds[1]} chain, {kinds[2]} isolated); '
            f'{int(self.fixed_T.sum())} absolute components; '
            f'points per component min/median/max = '
            f'{int(counts_np.min()) if counts_np.size else 0}/'
            f'{int(np.median(counts_np)) if counts_np.size else 0}/'
            f'{int(counts_np.max()) if counts_np.size else 0}',
        ]
        for kind, name in PIN_KIND_NAMES.items():
            mask = self.kind == kind
            if mask.any():
                et = self.eps_theta[mask]
                ez = self.eps_z[mask]
                lines.append(
                    f'  footprint[{name}]: eps_theta median {float(et.median()):.4f} rad '
                    f'(min {float(et.min()):.4f}, max {float(et.max()):.4f}); '
                    f'eps_z median {float(ez.median()):.1f} vox '
                    f'(min {float(ez.min()):.1f}, max {float(ez.max()):.1f})')
        report = self.consistency_report
        if report:
            lines.append(
                f'  consistency: {report.get("inconsistent_edges", 0)} inconsistent edges '
                f'in {len(report.get("components", []))} components')
        return '\n'.join(lines)


def _robust_integer_offsets(num_nodes, edges, deltas, component_of, num_components,
                            root_labels, max_rounds=4):
    """Integer node offsets ``n`` with ``n_v - n_u = delta`` on as many edges
    as possible.

    Per round: least-squares potentials over the kept edges (one node per
    component pinned at 0), rounded to integers; edges the rounding violates
    are dropped for the next round. Returns ``(n, inconsistent)`` where
    ``inconsistent`` lists ``(edge_index, mismatch)`` for the dropped edges.
    """
    import scipy.sparse
    import scipy.sparse.linalg
    n = np.zeros(num_nodes, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    deltas = np.asarray(deltas, dtype=np.float64)
    if num_nodes == 0:
        return n, []
    # One root per component: the smallest label.
    roots = {}
    for node in range(num_nodes):
        comp = int(component_of[node])
        if comp not in roots or root_labels[node] < root_labels[roots[comp]]:
            roots[comp] = node
    root_nodes = np.asarray(sorted(roots.values()), dtype=np.int64)
    keep = np.ones(len(edges), dtype=bool)
    for _ in range(max_rounds):
        e = edges[keep]
        d = deltas[keep]
        rows = np.concatenate([np.arange(len(e)), np.arange(len(e)),
                               len(e) + np.arange(len(root_nodes))])
        cols = np.concatenate([e[:, 1], e[:, 0], root_nodes])
        vals = np.concatenate([np.ones(len(e)), -np.ones(len(e)), np.ones(len(root_nodes))])
        A = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(len(e) + len(root_nodes), num_nodes))
        b = np.concatenate([d, np.zeros(len(root_nodes))])
        solution = scipy.sparse.linalg.lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
        n = np.rint(solution).astype(np.int64)
        residual = n[edges[:, 1]] - n[edges[:, 0]] - deltas.astype(np.int64)
        bad = keep & (residual != 0)
        if not bad.any():
            break
        keep &= ~bad
    residual = n[edges[:, 1]] - n[edges[:, 0]] - deltas.astype(np.int64)
    inconsistent = [(int(k), int(residual[k])) for k in np.nonzero(residual != 0)[0]]
    return n, inconsistent


def _crossing_step(theta_from, theta_to):
    """+1 / -1 seam step for a |dtheta| < pi move (matches ThetaCrossingMap)."""
    delta = theta_to - theta_from
    return (delta > math.pi).astype(np.int64) - (delta < -math.pi).astype(np.int64)


def finalize_registry(graph, *, intermediate_transform, dr, crossing_map,
                      patch_atlas, footprint_rule, free_gap_fn, min_z, max_z,
                      device, canonical_transform=None, patch_pin_kind=PIN_KIND_PATCH,
                      chunk_size=262144):
    """Resolve the graph against the current model into a :class:`PinRegistry`.

    ``intermediate_transform`` maps scroll -> intermediate space (the flow
    chain without the gap expander); ``canonical_transform`` is the full
    *unpinned* scroll -> spiral transform, used to estimate ``T`` in canonical
    winding units (the intermediate radius over ``dr`` ignores the learned
    gap table and is windings off at large radius); ``crossing_map`` is a refreshed
    ``ThetaCrossingMap`` whose potentials were computed under the *same*
    transform; ``free_gap_fn(theta, z, slot)`` returns the free winding gap
    (for the coincidence tolerance). ``patch_atlas`` supplies quad centres.
    """
    graph._components()
    num_nodes = len(graph.nodes)
    component_of = graph.component_of
    dr_f = float(dr.detach()) if torch.is_tensor(dr) else float(dr)

    # -- intermediate-space angles of every point node -------------------
    point_nodes = [i for i, node in enumerate(graph.nodes) if node.kind == 'point']
    theta_node = np.full(num_nodes, np.nan, dtype=np.float64)
    if point_nodes:
        zyx = torch.from_numpy(np.stack([graph.nodes[i].zyx for i in point_nodes])).to(device)
        with torch.no_grad():
            pieces = []
            for start in range(0, zyx.shape[0], chunk_size):
                pieces.append(intermediate_transform(zyx[start:start + chunk_size]))
            inter = torch.cat(pieces, dim=0)
        theta = (torch.atan2(inter[:, 1], inter[:, 2]) % TWO_PI).double().cpu().numpy()
        theta_node[np.asarray(point_nodes)] = theta

    # -- potentials of attached points --------------------------------
    pot_node = np.zeros(num_nodes, dtype=np.int64)
    attached = [i for i in point_nodes if graph.nodes[i].attach_ij is not None]
    if attached:
        quad_ids = torch.from_numpy(patch_atlas.theta_node_ids(
            np.asarray([graph.nodes[i].patch_index for i in attached], dtype=np.int64),
            np.asarray([graph.nodes[i].attach_ij for i in attached], dtype=np.float32)))
        sampled_theta = torch.as_tensor(theta_node[np.asarray(attached)], dtype=torch.float32)
        pots = crossing_map.winding_potentials(
            quad_ids.to(crossing_map.device), sampled_theta.to(crossing_map.device))
        crossing_map.assert_no_pending_potential_errors()
        pot_node[np.asarray(attached)] = pots.cpu().numpy().astype(np.int64)

    # -- overlap edges: potentials and angles of both quad centres ---------
    overlap_edges = [k for k, edge in enumerate(graph.edges) if edge.kind == 'overlap']
    overlap_delta = {}
    if overlap_edges:
        patch_idx = np.asarray([
            [graph.nodes[graph.edges[k].u].patch_index, graph.nodes[graph.edges[k].v].patch_index]
            for k in overlap_edges], dtype=np.int64)
        ijs = np.asarray([[graph.edges[k].ij_u, graph.edges[k].ij_v] for k in overlap_edges], dtype=np.float32)
        node_ids = patch_atlas.theta_node_ids(patch_idx.reshape(-1), ijs.reshape(-1, 2))
        pots = crossing_map.winding_potentials(
            torch.from_numpy(node_ids).to(crossing_map.device)).cpu().numpy().astype(np.int64).reshape(-1, 2)
        crossing_map.assert_no_pending_potential_errors()
        centres = patch_atlas.lookup(
            torch.from_numpy(patch_idx.reshape(-1)), torch.from_numpy(ijs.reshape(-1, 2)) + 0.5).to(device)
        with torch.no_grad():
            inter = torch.cat([intermediate_transform(centres[s:s + chunk_size])
                               for s in range(0, centres.shape[0], chunk_size)], dim=0)
        thetas = (torch.atan2(inter[:, 1], inter[:, 2]) % TWO_PI).double().cpu().numpy().reshape(-1, 2)
        steps = _crossing_step(thetas[:, 0], thetas[:, 1])
        for row, k in enumerate(overlap_edges):
            overlap_delta[k] = int(pots[row, 1] - pots[row, 0] - steps[row])

    # -- edge deltas ------------------------------------------------------
    deltas = np.zeros(len(graph.edges), dtype=np.int64)
    for k, edge in enumerate(graph.edges):
        u, v = graph.nodes[edge.u], graph.nodes[edge.v]
        if edge.kind == 'chain':
            deltas[k] = (round(v.winding - u.winding)
                         - int(_crossing_step(np.asarray(theta_node[edge.u]), np.asarray(theta_node[edge.v]))))
        elif edge.kind == 'attach':
            deltas[k] = -pot_node[edge.v]
        elif edge.kind == 'overlap':
            deltas[k] = overlap_delta[k]
        else:
            raise ValueError(edge.kind)

    # -- offsets: robust integer potentials per component -----------------
    # A BFS tree would let one wrong edge (a mis-annotated PCL, an overlap
    # between touching sheets) misplace a whole subtree by a winding. Solve
    # the least-squares potentials over all edges instead, round, drop the
    # edges the rounded solution violates and re-solve; the survivors are
    # consistent and the dropped edges are reported.
    n_node, inconsistent = _robust_integer_offsets(
        num_nodes, [(e.u, e.v) for e in graph.edges], deltas, component_of,
        graph.num_components, root_labels=[node.label for node in graph.nodes])
    by_component = collections.defaultdict(list)
    for k, mismatch in inconsistent:
        edge = graph.edges[k]
        by_component[int(component_of[edge.u])].append({
            'edge': edge.label or f'{graph.nodes[edge.u].label}->{graph.nodes[edge.v].label}',
            'kind': edge.kind, 'mismatch': int(mismatch)})
    by_kind = collections.Counter(graph.edges[k].kind for k, _ in inconsistent)
    consistency_report = {
        'inconsistent_edges': len(inconsistent),
        'inconsistent_edges_by_kind': dict(by_kind),
        'components': dict(by_component),
    }
    # -- absolute components ---------------------------------------------
    fixed_T = np.zeros(graph.num_components, dtype=bool)
    fixed_T_value = np.zeros(graph.num_components, dtype=np.float32)
    for anchor in graph.absolute:
        comp = int(component_of[anchor.node])
        # T_P = w + pot_p, T_g = T_P - O_P where the patch node's n is O_P; the
        # anchor is the attached point itself with n_p = O_P - pot_p, so
        # T_g = w - n_p.
        value = anchor.winding - float(n_node[anchor.node])
        if fixed_T[comp] and abs(fixed_T_value[comp] - value) > 1e-6:
            consistency_report.setdefault('absolute_conflicts', []).append({
                'component': comp, 'anchor': anchor.label,
                'values': [float(fixed_T_value[comp]), float(value)]})
        fixed_T[comp] = True
        fixed_T_value[comp] = value

    # -- emit pins --------------------------------------------------------
    pin_zyx, pin_comp, pin_n, pin_theta, pin_kind = [], [], [], [], []
    spacing_theta, spacing_z, pin_z = [], [], []
    neighbours = []
    emitted = 0
    for i, node in enumerate(graph.nodes):
        if node.kind != 'patch':
            continue
        # Quad centres of the patch's sampling-valid quads (its theta nodes).
        grid = _patch_quad_grid(patch_atlas, node.patch_index, graph.patch_grid_stride)
        if grid is None:
            continue
        zyx_g, node_ids_g, valid_g, _ = grid
        with torch.no_grad():
            flat = zyx_g.reshape(-1, 3).to(device)
            pieces = []
            for start in range(0, flat.shape[0], chunk_size):
                pieces.append(intermediate_transform(flat[start:start + chunk_size]))
            inter = torch.cat(pieces, dim=0).reshape(*zyx_g.shape[:2], 3)
        theta_g = torch.atan2(inter[..., 1], inter[..., 2]) % TWO_PI
        z_g = inter[..., 0]
        valid_t = torch.from_numpy(valid_g).to(device)
        pots = crossing_map.winding_potentials(
            torch.from_numpy(node_ids_g.reshape(-1)).to(crossing_map.device)).reshape(*valid_g.shape)
        crossing_map.assert_no_pending_potential_errors()
        sp_t, sp_z = grid_neighbour_spacing(theta_g, z_g, valid_t)
        sel = valid_t.reshape(-1)
        ids = torch.full(valid_t.shape, -1, dtype=torch.long, device=device)
        ids[valid_t] = torch.arange(int(valid_t.sum()), device=device) + emitted
        adjacent = []
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            other = ids.roll(shift, axis)
            if axis == 0:
                other[0 if shift > 0 else -1, :] = -1
            else:
                other[:, 0 if shift > 0 else -1] = -1
            adjacent.append(other[valid_t])
        neighbours.append(torch.stack(adjacent, dim=-1))
        emitted += int(valid_t.sum())
        pin_zyx.append(zyx_g.reshape(-1, 3).to(device)[sel])
        pin_theta.append(theta_g.reshape(-1)[sel])
        pin_z.append(z_g.reshape(-1)[sel])
        pin_n.append((int(n_node[i]) - pots.to(device).reshape(-1)[sel].to(torch.int64)))
        pin_comp.append(torch.full([int(sel.sum())], int(component_of[i]), dtype=torch.int64, device=device))
        pin_kind.append(torch.full([int(sel.sum())], patch_pin_kind, dtype=torch.int64, device=device))
        spacing_theta.append(sp_t.reshape(-1)[sel])
        spacing_z.append(sp_z.reshape(-1)[sel])
    # Point pins.
    point_pins = [i for i in point_nodes if graph.nodes[i].is_pin]
    if point_pins:
        idx = np.asarray(point_pins)
        zyx = torch.from_numpy(np.stack([graph.nodes[i].zyx for i in idx])).to(device)
        with torch.no_grad():
            inter = torch.cat([intermediate_transform(zyx[s:s + chunk_size])
                               for s in range(0, zyx.shape[0], chunk_size)], dim=0)
        theta_p = torch.atan2(inter[:, 1], inter[:, 2]) % TWO_PI
        z_p = inter[:, 0]
        # Chain spacing: neighbours along the chain (nan for isolated points).
        sp_t = torch.full_like(theta_p, float('nan'))
        sp_z = torch.full_like(theta_p, float('nan'))
        theta_all = torch.from_numpy(theta_node).to(device)
        z_all = torch.full([num_nodes], float('nan'), dtype=torch.float64, device=device)
        z_all[torch.from_numpy(idx).to(device)] = z_p.double()
        # z of non-pin connector nodes: transform them too for spacing.
        connectors = [i for i in point_nodes if not graph.nodes[i].is_pin]
        if connectors:
            czyx = torch.from_numpy(np.stack([graph.nodes[i].zyx for i in connectors])).to(device)
            with torch.no_grad():
                cinter = intermediate_transform(czyx)
            z_all[torch.from_numpy(np.asarray(connectors)).to(device)] = cinter[:, 0].double()
        kinds = torch.full([len(idx)], PIN_KIND_ISOLATED, dtype=torch.int64, device=device)
        prev_idx = np.asarray([graph.nodes[i].chain_prev if graph.nodes[i].chain_prev is not None else -1 for i in idx])
        next_idx = np.asarray([graph.nodes[i].chain_next if graph.nodes[i].chain_next is not None else -1 for i in idx])
        for neighbour_idx in (prev_idx, next_idx):
            has = neighbour_idx >= 0
            if not has.any():
                continue
            nb = torch.from_numpy(neighbour_idx[has]).to(device)
            here = torch.from_numpy(np.nonzero(has)[0]).to(device)
            dt = wrap_angle(theta_all[nb] - theta_all[torch.from_numpy(idx[has]).to(device)]).abs().to(theta_p.dtype)
            dz = (z_all[nb] - z_p[here].double()).abs().to(theta_p.dtype)
            sp_t[here] = torch.maximum(torch.nan_to_num(sp_t[here], nan=0.0), dt)
            sp_z[here] = torch.maximum(torch.nan_to_num(sp_z[here], nan=0.0), dz)
            kinds[here] = PIN_KIND_CHAIN
        node_to_pin = {int(node): emitted + j for j, node in enumerate(idx)}
        adjacent = torch.full((len(idx), 4), -1, dtype=torch.long, device=device)
        for column, neighbour_idx in enumerate((prev_idx, next_idx)):
            adjacent[:, column] = torch.tensor([node_to_pin.get(int(i), -2 if i >= 0 else -1) for i in neighbour_idx], device=device)
        neighbours.append(adjacent)
        pin_zyx.append(zyx)
        pin_theta.append(theta_p)
        pin_z.append(z_p)
        pin_n.append(torch.from_numpy(n_node[idx]).to(device))
        pin_comp.append(torch.from_numpy(component_of[idx]).to(device))
        pin_kind.append(kinds)
        spacing_theta.append(sp_t)
        spacing_z.append(sp_z)

    def cat(parts, dtype=None, empty_shape=(0,)):
        if parts:
            out = torch.cat(parts, dim=0)
            return out.to(dtype) if dtype is not None else out
        return torch.empty(empty_shape, dtype=dtype or torch.float32, device=device)

    zyx_all = cat(pin_zyx, torch.float32, (0, 3))
    theta_all = cat(pin_theta, torch.float32)
    z_all = cat(pin_z, torch.float32)
    n_all = cat(pin_n, torch.int32)
    comp_all = cat(pin_comp, torch.int64)
    kind_all = cat(pin_kind, torch.int64)
    sp_t_all = cat(spacing_theta, torch.float32)
    sp_z_all = cat(spacing_z, torch.float32)

    # Radii for the footprint floor and T initialisation.
    with torch.no_grad():
        inter = torch.cat([intermediate_transform(zyx_all[s:s + chunk_size])
                           for s in range(0, zyx_all.shape[0], chunk_size)], dim=0) \
            if zyx_all.shape[0] else torch.empty([0, 3], device=device)
        radius = torch.linalg.norm(inter[:, 1:], dim=-1) if inter.shape[0] else torch.empty([0], device=device)
    eps_theta, eps_z = footprint_rule.apply(sp_t_all, sp_z_all, radius)

    # T initialisation: median over the component of the canonical shifted
    # winding minus n, under the unpinned model.
    if canonical_transform is not None and zyx_all.shape[0]:
        with torch.no_grad():
            spiral = torch.cat([canonical_transform(zyx_all[s:s + chunk_size])
                                for s in range(0, zyx_all.shape[0], chunk_size)], dim=0)
        canonical_radius = torch.linalg.norm(spiral[:, 1:], dim=-1)
    else:
        canonical_radius = radius
    shifted = canonical_radius - theta_all / TWO_PI * dr_f
    estimate = shifted / dr_f - n_all.to(torch.float32)
    initial_T = torch.zeros([graph.num_components], dtype=torch.float32, device=device)
    for comp in range(graph.num_components):
        mask = comp_all == comp
        if mask.any():
            initial_T[comp] = estimate[mask].median()
    fixed_T_t = torch.from_numpy(fixed_T).to(device)
    fixed_T_value_t = torch.from_numpy(fixed_T_value).to(device)
    initial_T = torch.where(fixed_T_t, fixed_T_value_t, initial_T)

    slot = torch.round(initial_T[comp_all] + n_all.to(torch.float32)).to(torch.int64)
    with torch.no_grad():
        local_gap = free_gap_fn(theta_all, z_all, slot) if zyx_all.shape[0] else torch.empty([0], device=device)

    num_patches = len(patch_atlas._patches) if patch_atlas is not None else 0
    patch_component = torch.full([num_patches], -1, dtype=torch.int64)
    patch_offset = torch.zeros([num_patches], dtype=torch.int64)
    for patch_index, node in graph.patch_nodes.items():
        if patch_index < num_patches:
            patch_component[patch_index] = int(component_of[node])
            patch_offset[patch_index] = int(n_node[node])

    return PinRegistry(
        zyx=zyx_all, component=comp_all, n0=n_all, theta0=theta_all,
        eps_theta=eps_theta.to(torch.float32), eps_z=eps_z.to(torch.float32),
        kind=kind_all, num_components=graph.num_components,
        fixed_T=fixed_T_t, fixed_T_value=fixed_T_value_t, initial_T=initial_T,
        fingerprint=graph.fingerprint(), consistency_report=consistency_report,
        local_gap=local_gap.to(torch.float32),
        patch_component=patch_component.to(device), patch_offset=patch_offset.to(device),
        neighbours=cat(neighbours, torch.int64, (0, 4)))


def _patch_quad_grid(patch_atlas, patch_index, stride, with_node_ids=True):
    """Quad-centre zyx grid, theta node ids and validity for one patch.

    Returns ``(zyx [h, w, 3] (CPU), node_ids [h, w], valid [h, w], ijs [h, w, 2])``
    over the bounding box of the kept quads with every ``stride``-th quad
    kept, or None when the patch has no sampling-valid quad.
    """
    patch = patch_atlas._patches[patch_index]
    mask = np.asarray(patch._sampling_valid_quad_mask_np, dtype=bool)
    if not mask.any():
        return None
    H, W = mask.shape
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    keep = mask & (ii % stride == 0) & (jj % stride == 0)
    if not keep.any():
        return None
    # Restrict to the bounding box of kept quads to bound the work.
    rows = np.nonzero(keep.any(axis=1))[0]
    cols = np.nonzero(keep.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    ii_box = ii[r0:r1:stride, c0:c1:stride]
    jj_box = jj[r0:r1:stride, c0:c1:stride]
    keep_box = keep[r0:r1:stride, c0:c1:stride]
    ijs = np.stack([ii_box, jj_box], axis=-1).astype(np.float32)
    flat_ijs = ijs.reshape(-1, 2)
    flat_keep = keep_box.reshape(-1)
    zyx = torch.full([flat_ijs.shape[0], 3], -1.0, dtype=torch.float32)
    node_ids = np.zeros(flat_ijs.shape[0], dtype=np.int64)
    if flat_keep.any():
        idx_t = torch.full([int(flat_keep.sum())], int(patch_index), dtype=torch.int64)
        centres = torch.from_numpy(flat_ijs[flat_keep]) + 0.5
        zyx[torch.from_numpy(flat_keep)] = patch_atlas.lookup(idx_t, centres).to(torch.float32).cpu()
        if with_node_ids:
            node_ids[flat_keep] = patch_atlas.theta_node_ids(
                np.full(int(flat_keep.sum()), patch_index, dtype=np.int64), flat_ijs[flat_keep])
    return (zyx.reshape(*keep_box.shape, 3), node_ids.reshape(keep_box.shape), keep_box,
            np.stack([ii_box, jj_box], axis=-1))


def patch_overlap_pairs(patch_atlas, tolerance, *, max_pairs_per_patch_pair=3,
                        stride=1):
    """Representative coincident quad-centre pairs between different patches.

    Quad centres of all sampling-valid quads (every ``stride``-th) go into
    one KD-tree; centre pairs from different patches within ``tolerance``
    scroll voxels are coincident observations of one sheet. For each patch
    pair the closest ``max_pairs_per_patch_pair`` pairs are kept (one links
    the patches, the rest check the link's consistency). Returns a list of
    ``(patch_index_a, ij_a, patch_index_b, ij_b, distance)``.
    """
    from scipy.spatial import cKDTree
    centres, owners, ijs = [], [], []
    for patch_index in range(len(patch_atlas._patches)):
        grid = _patch_quad_grid(patch_atlas, patch_index, stride, with_node_ids=False)
        if grid is None:
            continue
        zyx_g, _, valid_g, ij_g = grid
        keep = valid_g.reshape(-1)
        centres.append(zyx_g.reshape(-1, 3).numpy()[keep])
        ijs.append(ij_g.reshape(-1, 2)[keep])
        owners.append(np.full(int(keep.sum()), patch_index, dtype=np.int64))
    if not centres:
        return []
    centres = np.concatenate(centres)
    owners = np.concatenate(owners)
    ijs = np.concatenate(ijs)
    tree = cKDTree(centres)
    pairs = tree.query_pairs(r=float(tolerance), output_type='ndarray')
    if len(pairs) == 0:
        return []
    cross = pairs[owners[pairs[:, 0]] != owners[pairs[:, 1]]]
    if len(cross) == 0:
        return []
    a, b = owners[cross[:, 0]], owners[cross[:, 1]]
    swap = a > b
    cross[swap] = cross[swap][:, ::-1]
    a, b = owners[cross[:, 0]], owners[cross[:, 1]]
    distance = np.linalg.norm(centres[cross[:, 0]] - centres[cross[:, 1]], axis=-1)
    order = np.lexsort((distance, b, a))
    cross, a, b, distance = cross[order], a[order], b[order], distance[order]
    key = a * len(patch_atlas._patches) + b
    _, first = np.unique(key, return_index=True)
    rank = np.arange(len(key)) - np.repeat(first, np.diff(np.append(first, len(key))))
    keep = rank < max_pairs_per_patch_pair
    return [(int(a[k]), tuple(int(v) for v in ijs[cross[k, 0]]), int(b[k]),
             tuple(int(v) for v in ijs[cross[k, 1]]), float(distance[k]))
            for k in np.nonzero(keep)[0]]


def build_pin_graph(*, verified_patches, patch_atlas, cross_patch_pcls,
                    unattached_pcl_strips, unattached_components,
                    unattached_component_edges, patch_grid_stride=1,
                    strip_radial_offsets=None, overlap_pairs=None):
    """Assemble the :class:`PinGraph` from the fit's prepared inputs.

    ``verified_patches`` is the id -> Patch mapping whose order matches
    ``patch_atlas``; ``cross_patch_pcls`` the list of cross-patch PCLs (with
    ``points_by_patch``, chains and ``on_patch`` links); the strip arguments
    are ``FitContext``'s unattached strip list and its link-component view.
    Strip points that duplicate a cross-patch PCL point (same source point)
    reuse that node. Strip points carrying a non-zero radial offset are
    connectors, not pins (their target is not a bare winding coordinate).
    """
    graph = PinGraph()
    graph.patch_grid_stride = int(max(1, patch_grid_stride))
    for pid in verified_patches:
        graph.add_patch(patch_atlas.id_to_idx[pid], f'patch:{pid}')
    for a, ij_a, b, ij_b, distance in overlap_pairs or []:
        if a in graph.patch_nodes and b in graph.patch_nodes:
            graph.add_overlap_edge(a, b, ij_a, ij_b, f'overlap:{a}:{b}')

    point_node = {}  # id(point dict) -> node

    def node_for_point(pcl_id, key, point):
        node = point_node.get(id(point))
        if node is not None:
            return node
        on_patch = point.get('on_patch')
        patch_index = None
        attach_ij = None
        if on_patch is not None and on_patch['id'] in verified_patches:
            # Same rule as losses._valid_patch_annotation: the annotation's
            # quad must be a retained sampling-valid quad, else the point is
            # not attached (it stays a chain point).
            mask = verified_patches[on_patch['id']]._sampling_valid_quad_mask_np
            i_q = min(max(int(on_patch['ij'][0]), 0), mask.shape[0] - 1)
            j_q = min(max(int(on_patch['ij'][1]), 0), mask.shape[1] - 1)
            if mask[i_q, j_q]:
                patch_index = patch_atlas.id_to_idx[on_patch['id']]
                attach_ij = (float(i_q), float(j_q))
        node = graph.add_point(
            point['zyx'], f'pcl:{pcl_id}:{key}', winding=point.get('winding_annotation', 0.0),
            attach_ij=attach_ij, patch_index=patch_index)
        point_node[id(point)] = node
        return node

    zyx_node = {}
    for pcl in cross_patch_pcls:
        pcl_id = pcl.get('id', pcl.get('name', '?'))
        chain = list(pcl['chain'].iter_chain())
        keys = {id(point): key for key, point in pcl['points'].items()}
        nodes = [node_for_point(pcl_id, keys.get(id(point), i), point) for i, point in enumerate(chain)]
        for point, node in zip(chain, nodes):
            zyx_node[np.asarray(point['zyx'], dtype=np.float32).tobytes()] = node
        for u, v in zip(nodes[:-1], nodes[1:]):
            graph.add_chain_edge(u, v, f'pcl:{pcl_id}')
        if pcl.get('metadata', {}).get('winding_is_absolute', False):
            for point in pcl['points'].values():
                node = point_node.get(id(point))
                if node is not None and graph.nodes[node].attach_ij is not None:
                    graph.add_absolute(node, point['winding_annotation'], f'abs:{pcl_id}')

    strip_nodes = []
    for strip_idx, strip in enumerate(unattached_pcl_strips):
        zyxs = np.asarray(strip['zyxs'], dtype=np.float32)
        windings = np.asarray(strip['windings'], dtype=np.float32)
        offsets = None
        if strip_radial_offsets is not None:
            offsets = strip_radial_offsets[strip_idx]
        elif strip.get('radial_offsets') is not None:
            offsets = np.asarray(strip['radial_offsets'], dtype=np.float32)
        nodes = []
        for j in range(zyxs.shape[0]):
            key = zyxs[j].tobytes()
            node = zyx_node.get(key)
            if node is None:
                is_pin = offsets is None or float(offsets[j]) == 0.0
                node = graph.add_point(zyxs[j], f'strip:{strip["id"]}:{j}',
                                       winding=float(windings[j]), is_pin=is_pin)
                zyx_node[key] = node
            nodes.append(node)
        for u, v in zip(nodes[:-1], nodes[1:]):
            graph.add_chain_edge(u, v, f'strip:{strip["id"]}')
        strip_nodes.append(nodes)
    for edges in unattached_component_edges:
        for strip_a, pos_a, strip_b, pos_b in edges:
            graph.add_chain_edge(strip_nodes[strip_a][pos_a], strip_nodes[strip_b][pos_b],
                                 f'link:{strip_a}:{strip_b}')
    return graph
