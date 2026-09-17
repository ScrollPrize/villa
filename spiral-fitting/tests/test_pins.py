"""Pinned winding radii (pinned_spiral_plan.md stage 2a): unit tests over the
pure one-ray / per-ray functions in pins.py, plus the transform-level
integration (PinnedGapExpandingTransform, compute_pins, the registry graph).

Numbered tests follow the plan's list (1, 2, 2a, 2b, 2c, 3, 4, 5, 5b, 5c, 6,
7, 7b) and the synthetic end-to-end check (8, reduced to the transform level).
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pins
from pins import (
    FootprintRule, PinCells, TWO_PI, build_pinned_ray_map,
    chain_neighbour_spacing, compact_kernel, compute_coincidence_groups,
    free_map_forward, free_map_inverse, grid_neighbour_spacing,
    pinned_map_forward, pinned_map_inverse, ray_anchors,
)



@pytest.fixture(autouse=True)
def _double_precision():
    # The pure functions are checked in float64 (tolerances of 1e-9 on radii
    # of order 100); restore the default so other test modules see float32.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


DR = 16.0
MIN_GAP = 1.0
NUM_WINDINGS = 12


def PinTable(z, theta, r, target, slot, eps_theta, eps_z, num_slots, min_z, max_z, groups=None):
    """pins.PinTable taking each pin's full target at its own angle (the
    tests' natural quantity); the table stores the shifted target dr (T + n)."""
    return pins.PinTable(z, theta, r, target - DR * theta / TWO_PI, slot, eps_theta, eps_z,
                         num_slots, min_z, max_z, torch.tensor(DR), groups=groups)


def _free_table(num_rays, theta, gen, num_windings=NUM_WINDINGS, dr=DR, jitter=0.6):
    """Random monotone winding-radius tables: dr * (k + theta/2pi) with
    per-winding gap jitter (gaps stay above MIN_GAP)."""
    gaps = dr * (1.0 + jitter * (torch.rand(num_rays, num_windings - 1, generator=gen) - 0.5))
    gaps = gaps.clamp(min=MIN_GAP * 1.5)
    zero = dr * theta[:, None] / TWO_PI
    return torch.cat([zero, zero + torch.cumsum(gaps, dim=-1)], dim=-1)


def _identity_table(num_rays, theta, num_windings=NUM_WINDINGS, dr=DR):
    k = torch.arange(num_windings)[None, :].to(torch.get_default_dtype())
    return dr * (k + theta[:, None] / TWO_PI)


def _pad(rows, width=None):
    """Pad a list of 1-D tensors to [N, J]; returns (padded, valid)."""
    width = width or max(len(r) for r in rows)
    out = torch.zeros(len(rows), width)
    valid = torch.zeros(len(rows), width, dtype=torch.bool)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
        valid[i, :len(r)] = True
    return out, valid


def _ray_map(table, theta, R, S, w, valid=None, dr=DR, min_gap=MIN_GAP):
    if valid is None:
        valid = torch.ones_like(R, dtype=torch.bool)
    dr_t = torch.as_tensor(dr)
    return build_pinned_ray_map(table, dr_t, theta / TWO_PI, R, S, w, valid, min_gap)


def _forward(r, table, theta, ray_map, dr=DR):
    return pinned_map_forward(r, table, torch.as_tensor(dr), theta / TWO_PI, ray_map)


def _inverse(c, table, theta, ray_map, dr=DR):
    return pinned_map_inverse(c, table, torch.as_tensor(dr), theta / TWO_PI, ray_map)


def _free(r, table, theta, dr=DR):
    return free_map_forward(r, table, torch.as_tensor(dr), theta / TWO_PI)


# ---------------------------------------------------------------------------
# 1. exactness
# ---------------------------------------------------------------------------


def test_pinned_map_exact_on_pins():
    gen = torch.Generator().manual_seed(1)
    num_rays = 64
    theta = torch.rand(num_rays, generator=gen) * TWO_PI
    table = _free_table(num_rays, theta, gen)
    # Consistent pins: increasing R and increasing S, w = 1.
    J = 5
    R = torch.sort(20 + torch.rand(num_rays, J, generator=gen) * 140, dim=-1).values
    S = torch.sort(10 + torch.rand(num_rays, J, generator=gen) * 160, dim=-1).values
    # Keep targets spaced by more than the min rise can demand.
    S = S + torch.arange(J)[None, :] * 4.0
    w = torch.ones(num_rays, J)
    ray_map = _ray_map(table, theta, R, S, w)
    assert int(ray_map.guard_active.sum()) == 0
    s_at_pins = _forward(R, table, theta, ray_map)
    assert (s_at_pins - S).abs().max() < 1e-9


# ---------------------------------------------------------------------------
# 2. monotone under adversarial anchors, guard counting
# ---------------------------------------------------------------------------


def _hand_count_guard(table_row, theta_row, R, S, w, dr=DR, min_gap=MIN_GAP):
    """Reference one-ray solution in plain numpy: the tridiagonal blend of
    plan 2a.4 (as implemented; see pins.build_pinned_ray_map) followed by
    the hard-max ordering guard."""
    order = np.argsort(R, kind='stable')
    R, S, w = R[order], S[order], w[order]
    sfree = lambda r: float(_free(torch.tensor([r]), table_row[None], theta_row[None])[0])
    J = len(R)
    u0 = sfree(0.0)
    u = np.asarray([sfree(r) for r in R])
    u_prev = np.concatenate([[u0], u[:-1]])
    u_next = np.concatenate([u[1:], u[-1:]])
    lam = np.where(u_next > u_prev, (u_next - u) / np.where(u_next > u_prev, u_next - u_prev, 1.0), 0.5)
    lam[-1] = 1.0
    A = np.eye(J)
    b = w * (S - u)
    for j in range(J):
        if j > 0:
            A[j, j - 1] = -(1 - w[j]) * lam[j]
        if j + 1 < J:
            A[j, j + 1] = -(1 - w[j]) * (1 - lam[j])
    delta = np.linalg.solve(A, b)
    s_prev, u_prev_v = u0, u0
    s_at, violations, causes = [], 0, []
    for j in range(J):
        free_rise = u[j] - u_prev_v
        min_rise = min_gap * free_rise / dr
        candidate = u[j] + delta[j]
        if candidate - s_prev < min_rise:
            violations += 1
            causes.append('order' if S[j] < s_prev else 'min_rise')
            s_new = s_prev + min_rise
        else:
            s_new = candidate
        s_at.append(s_new)
        s_prev, u_prev_v = s_new, u[j]
    return np.asarray(s_at), violations, causes, order


def test_pinned_map_monotone():
    gen = torch.Generator().manual_seed(2)
    num_rays = 48
    theta = torch.rand(num_rays, generator=gen) * TWO_PI
    table = _free_table(num_rays, theta, gen)
    J = 6
    # Adversarial: R in arbitrary order, S in arbitrary order (many swaps), w in (0, 1].
    R = 20 + torch.rand(num_rays, J, generator=gen) * 140
    S = 10 + torch.rand(num_rays, J, generator=gen) * 160
    w = 0.2 + 0.8 * torch.rand(num_rays, J, generator=gen)
    w[:, ::2] = 1.0
    ray_map = _ray_map(table, theta, R, S, w)

    grid = torch.linspace(0.5, 220.0, 700)[None, :].expand(num_rays, -1)
    s_grid = _forward(grid, table, theta, ray_map)
    assert torch.all(s_grid.diff(dim=-1) > 0), 'pinned map must be strictly increasing'

    for i in range(num_rays):
        s_at, violations, causes, order = _hand_count_guard(
            table[i], theta[i], R[i].numpy(), S[i].numpy(), w[i].numpy())
        assert int(ray_map.guard_active[i].sum()) == violations
        assert int(ray_map.order_violations()[i]) == causes.count('order')
        assert int(ray_map.min_rise_violations()[i]) == causes.count('min_rise')
        # Anchors not touched by the guard with w == 1 are exact; every anchor
        # value matches the hand recursion bit-for-bit up to fp association.
        got = ray_map.s_at[i, 1:].numpy()
        assert np.allclose(got, s_at, atol=1e-9)
        for j in range(J):
            if not ray_map.guard_active[i, j + 1] and w[i, order[j]] == 1.0:
                assert abs(got[j] - S[i, order[j]].item()) < 1e-9


def test_hard_max_leaves_non_violating_anchors_bit_identical():
    gen = torch.Generator().manual_seed(3)
    theta = torch.rand(8, generator=gen) * TWO_PI
    table = _free_table(8, theta, gen)
    R = torch.sort(20 + torch.rand(8, 4, generator=gen) * 140, dim=-1).values
    S = torch.sort(10 + torch.rand(8, 4, generator=gen) * 160, dim=-1).values + torch.arange(4)[None] * 5.
    w = torch.ones(8, 4)
    # Force one violation on ray 0 by swapping its last two targets far apart.
    S[0, 3] = S[0, 2] - 30.0
    ray_map = _ray_map(table, theta, R, S, w)
    assert int(ray_map.guard_active[0].sum()) == 1
    assert int(ray_map.guard_active[1:].sum()) == 0
    # Non-violating anchors are exactly their targets (no smoothing leak).
    assert torch.equal(ray_map.s_at[1:, 1:], S[1:])


# ---------------------------------------------------------------------------
# 2a. anchor crossing matrix
# ---------------------------------------------------------------------------


def _two_anchor_sweep(S_a, S_b, w_a, w_b, sweep):
    """Anchor a fixed at R=80, anchor b swept through it. Returns per-sweep
    (s at a probe grid, order violations, min-rise violations, ray maps)."""
    theta = torch.zeros(len(sweep))
    table = _identity_table(len(sweep), theta)
    R = torch.stack([torch.full_like(sweep, 80.0), sweep], dim=-1)
    S = torch.tensor([[S_a, S_b]]).expand(len(sweep), -1).clone()
    w = torch.tensor([[w_a, w_b]]).expand(len(sweep), -1).clone()
    ray_map = _ray_map(table, theta, R, S, w)
    grid = torch.linspace(1.0, 180.0, 600)[None].expand(len(sweep), -1)
    s = _forward(grid, table, theta, ray_map)
    return s, ray_map


def test_anchor_crossing_matrix():
    sweep = torch.linspace(60.0, 100.0, 81)
    # (i) targets in the new order on both sides: continuous through the crossing.
    # Anchor a at R=80 with S=80; anchor b swept from 60 to 100 with S = R_b
    # (identity), so its target agrees with its radial order on both sides.
    theta = torch.zeros(len(sweep))
    table = _identity_table(len(sweep), theta)
    R = torch.stack([torch.full_like(sweep, 80.0), sweep], dim=-1)
    S = torch.stack([torch.full_like(sweep, 80.0), sweep], dim=-1)
    w = torch.ones_like(R)
    ray_map = _ray_map(table, theta, R, S, w)
    grid = torch.linspace(1.0, 180.0, 600)[None].expand(len(sweep), -1)
    s = _forward(grid, table, theta, ray_map)
    assert int(ray_map.guard_active.sum()) == 0
    # Consecutive sweep positions differ by a small amount everywhere.
    assert (s[1:] - s[:-1]).abs().max() < 0.6 * float(sweep[1] - sweep[0]) + 1e-9
    assert (_forward(R, table, theta, ray_map) - S).abs().max() < 1e-9

    # (ii) targets out of order: S_a = 90 fixed, S_b = 70 fixed; when b passes
    # beyond a in R the sorted targets decrease -> guard fires ('order'),
    # and s jumps by about S_a - S_b at the crossing.
    s, ray_map = _two_anchor_sweep(90.0, 70.0, 1.0, 1.0, sweep)
    before = sweep < 80.0
    after = sweep > 80.0
    assert int(ray_map.order_violations()[before].sum()) == 0
    assert torch.all(ray_map.order_violations()[after] == 1)
    assert int(ray_map.min_rise_violations().sum()) == 0
    # Just after the crossing, s(R_b) is forced up from S_b to ~S_a (plus the
    # min rise): the map jumps by about S_a - S_b at the crossing.
    idx_after = int(torch.nonzero(after)[0])
    idx_before = idx_after - 2   # sweep hits R_a exactly in between; skip the coincident position
    theta = torch.zeros(len(sweep))
    table = _identity_table(len(sweep), theta)
    s_at_b = _forward(sweep[:, None], table, theta, ray_map)[:, 0]
    assert abs(float(s_at_b[idx_before]) - 70.0) < 1e-9
    jump = float(s_at_b[idx_after] - s_at_b[idx_before])
    assert abs(jump - (90.0 - 70.0)) < 3.0

    # (iii) equal targets, unequal weights: the jump across the crossing is
    # bounded by |w_a - w_b| * |S - a_free|; a guard firing is counted.
    w_a, w_b = 1.0, 0.95
    s, ray_map = _two_anchor_sweep(80.0, 80.0, w_a, w_b, sweep)
    idx_after = int(torch.nonzero(sweep > 80.0)[0])
    s_before, s_after = s[idx_after - 2], s[idx_after]
    a_free_gap = float(sweep[idx_after] - 80.0)  # identity free map
    bound = abs(w_a - w_b) * (abs(80.0 - 80.0) + a_free_gap) + 2 * float(sweep[1] - sweep[0]) + MIN_GAP
    assert float((s_after - s_before).abs().max()) <= bound
    for i in range(len(sweep)):
        S_sorted = ray_map.s_at[i]
        assert torch.isfinite(S_sorted).all()

    # (iv) exactly coincident R (free_rise = 0): finite, monotone, no NaN.
    theta = torch.zeros(1)
    table = _identity_table(1, theta)
    R = torch.tensor([[80.0, 80.0]])
    S = torch.tensor([[80.0, 84.0]])
    w = torch.ones(1, 2)
    ray_map = _ray_map(table, theta, R, S, w)
    grid = torch.linspace(1.0, 180.0, 400)[None]
    s = _forward(grid, table, theta, ray_map)
    assert torch.isfinite(s).all()
    assert torch.all(s.diff(dim=-1) >= 0)
    inv = _inverse(s, table, theta, ray_map)
    assert torch.isfinite(inv).all()


# ---------------------------------------------------------------------------
# 2b. multi-turn patch and dense cells in the CSR table
# ---------------------------------------------------------------------------


def _anchors_from_pins(theta_q, z_q, pin_theta, pin_z, pin_r, pin_target, pin_slot,
                       eps_theta, eps_z, num_slots, min_z=0.0, max_z=100.0):
    table = PinTable(pin_z, pin_theta, pin_r, pin_target, pin_slot, eps_theta, eps_z,
                     num_slots, min_z, max_z)
    return table.anchors(theta_q, z_q)


def test_pin_table_multi_turn_patch():
    # One component, two pins with different slots on the same ray.
    theta_q = torch.tensor([1.0])
    z_q = torch.tensor([50.0])
    pin_theta = torch.tensor([1.0, 1.0])
    pin_z = torch.tensor([50.0, 50.0])
    pin_r = torch.tensor([70.0, 90.0])
    pin_target = torch.tensor([66.0, 82.0])
    pin_slot = torch.tensor([4, 5])
    eps_theta = torch.full([2], 0.1)
    eps_z = torch.full([2], 5.0)
    R, S, w, valid = _anchors_from_pins(theta_q, z_q, pin_theta, pin_z, pin_r, pin_target,
                                        pin_slot, eps_theta, eps_z, NUM_WINDINGS)
    assert valid[0, 4] and valid[0, 5] and int(valid.sum()) == 2
    table = _identity_table(1, theta_q)
    ray_map = _ray_map(table, theta_q, R, S, w, valid)
    s = _forward(pin_r[None], table, theta_q, ray_map)
    assert (s[0] - pin_target).abs().max() < 1e-9

    # A cell with several hundred pins (dense grid, wide footprint): all present, all exact.
    n_side = 20
    tt, zz = torch.meshgrid(torch.linspace(1.0, 1.02, n_side), torch.linspace(50.0, 52.0, n_side), indexing='ij')
    pin_theta = tt.reshape(-1)
    pin_z = zz.reshape(-1)
    num = pin_theta.numel()
    pin_r = 70.0 + 0.5 * torch.rand(num, generator=torch.Generator().manual_seed(4))
    pin_target = 66.0 + 0.3 * (pin_r - 70.0)
    pin_slot = torch.full([num], 4)
    eps_theta = torch.full([num], 0.2)   # far wider than the whole grid
    eps_z = torch.full([num], 20.0)
    table_obj = PinTable(pin_z, pin_theta, pin_r, pin_target, pin_slot, eps_theta, eps_z,
                         NUM_WINDINGS, 0.0, 100.0)
    assert table_obj.num_pins == num
    q_idx, p_idx = table_obj.classes[0]['cells'].gather_pairs(pin_theta, pin_z)
    # Every pin sees every other pin (uncapped table).
    assert q_idx.numel() == num * num
    R, S, w, valid = table_obj.anchors(pin_theta, pin_z)
    table = _identity_table(num, pin_theta)
    ray_map = _ray_map(table, pin_theta, R, S, w, valid)
    s = _forward(pin_r, table, pin_theta, ray_map)
    assert (s - pin_target).abs().max() < 1e-6


# ---------------------------------------------------------------------------
# 2c. coincidence pass
# ---------------------------------------------------------------------------


def test_pin_coincidence_pass():
    theta = torch.tensor([1.0, 1.0 + 1e-4, 2.0, 2.0])
    z = torch.tensor([50.0, 50.0, 60.0, 60.0])
    slot = torch.tensor([4, 4, 5, 5])
    component = torch.tensor([0, 0, 1, 2])
    n = torch.tensor([4, 4, 5, 5])
    r = torch.tensor([70.0, 70.5, 90.0, 96.0])
    eps_theta = torch.full([4], 0.1)
    eps_z = torch.full([4], 5.0)
    local_gap = torch.full([4], DR)
    groups = compute_coincidence_groups(
        theta, z, slot, component, n, r, eps_theta, eps_z, 0.0, 100.0, local_gap,
        coincidence_frac=0.05, conflict_tolerance=0.1)
    assert groups.num_groups == 2
    assert groups.group[0] == groups.group[1] and groups.group[2] == groups.group[3]
    assert groups.num_conflicts == 1
    conflict = groups.conflicts[0]
    assert conflict['tag'] == 'cross_component'
    assert set(conflict['members']) == {2, 3}
    # A fold: same component and n, incompatible radii.
    component = torch.tensor([0, 0, 1, 1])
    groups = compute_coincidence_groups(
        theta, z, slot, component, n, r, eps_theta, eps_z, 0.0, 100.0, local_gap,
        coincidence_frac=0.05, conflict_tolerance=0.1)
    assert groups.conflicts[0]['tag'] == 'fold'

    # The compatible pair merges into one pin and that pin is exact.
    target = torch.tensor([66.0, 66.0, 82.0, 82.0])
    table_obj = PinTable(z, theta, r, target, slot, eps_theta, eps_z, NUM_WINDINGS, 0.0, 100.0,
                         groups=groups)
    assert table_obj.num_pins == 2
    theta_q = torch.tensor([1.0])
    z_q = torch.tensor([50.0])
    R, S, w, valid = table_obj.anchors(theta_q, z_q)
    assert valid[0, 4] and not valid[0, 5]
    assert abs(float(R[0, 4]) - 70.25) < 1e-9 and abs(float(S[0, 4]) - 66.0) < 1e-3  # merged pin sits at the mean angle
    table = _identity_table(1, theta_q)
    ray_map = _ray_map(table, theta_q, R, S, w, valid)
    assert abs(float(_forward(torch.tensor([[70.25]]), table, theta_q, ray_map)) - 66.0) < 1e-3


# ---------------------------------------------------------------------------
# 3. no pins is the free map
# ---------------------------------------------------------------------------


def test_pinned_map_no_pins_is_free():
    gen = torch.Generator().manual_seed(5)
    num_rays = 32
    theta = torch.rand(num_rays, generator=gen) * TWO_PI
    table = _free_table(num_rays, theta, gen)
    grid = torch.linspace(0.5, 220.0, 500)[None].expand(num_rays, -1)
    free = _free(grid, table, theta)
    # w = 0 for every anchor.
    R = 20 + torch.rand(num_rays, 5, generator=gen) * 140
    S = 10 + torch.rand(num_rays, 5, generator=gen) * 160
    w = torch.zeros(num_rays, 5)
    ray_map = _ray_map(table, theta, R, S, w)
    assert int(ray_map.guard_active.sum()) == 0
    assert (_forward(grid, table, theta, ray_map) - free).abs().max() < 1e-9
    # No anchors at all.
    empty = torch.zeros(num_rays, 0)
    ray_map = _ray_map(table, theta, empty, empty, empty, torch.zeros(num_rays, 0, dtype=torch.bool))
    assert (_forward(grid, table, theta, ray_map) - free).abs().max() < 1e-9


def test_zero_mass_anchor_is_transparent():
    gen = torch.Generator().manual_seed(6)
    theta = torch.rand(16, generator=gen) * TWO_PI
    table = _free_table(16, theta, gen)
    R_a = torch.full([16, 1], 60.0)
    S_a = torch.full([16, 1], 58.0)
    grid = torch.linspace(0.5, 220.0, 500)[None].expand(16, -1)
    base = _forward(grid, table, theta, _ray_map(table, theta, R_a, S_a, torch.ones(16, 1)))
    for R_b in (30.0, 61.0, 150.0):
        R = torch.cat([R_a, torch.full([16, 1], R_b)], dim=-1)
        S = torch.cat([S_a, torch.full([16, 1], 999.0)], dim=-1)
        w = torch.cat([torch.ones(16, 1), torch.zeros(16, 1)], dim=-1)
        ray_map = _ray_map(table, theta, R, S, w)
        assert int(ray_map.guard_active.sum()) == 0
        assert (_forward(grid, table, theta, ray_map) - base).abs().max() < 1e-9


# ---------------------------------------------------------------------------
# 4. inverse round trip
# ---------------------------------------------------------------------------


def test_pinned_map_inverse_roundtrip():
    gen = torch.Generator().manual_seed(7)
    num_rays = 40
    theta = torch.rand(num_rays, generator=gen) * TWO_PI
    table = _free_table(num_rays, theta, gen)
    R = 20 + torch.rand(num_rays, 6, generator=gen) * 140
    S = 10 + torch.rand(num_rays, 6, generator=gen) * 160
    w = torch.rand(num_rays, 6, generator=gen)
    ray_map = _ray_map(table, theta, R, S, w)
    r = 0.5 + torch.rand(num_rays, 300, generator=gen) * 220
    c = _forward(r, table, theta, ray_map)
    back = _inverse(c, table, theta, ray_map)
    assert (back - r).abs().max() < 1e-4
    # And the other direction, avoiding the jump regime: only rays with no guard.
    clean = ray_map.guard_active.sum(dim=-1) == 0
    if clean.any():
        c = 1.0 + torch.rand(num_rays, 300, generator=gen) * 200
        r = _inverse(c, table, theta, ray_map)
        again = _forward(r, table, theta, ray_map)
        assert (again - c)[clean].abs().max() < 1e-4


# ---------------------------------------------------------------------------
# 5. continuity of the lookup
# ---------------------------------------------------------------------------


def _single_pin_table(theta_p, z_p, r_p, target_p, eps_theta, eps_z, slot=4):
    return PinTable(torch.tensor([z_p]), torch.tensor([theta_p]), torch.tensor([r_p]),
                    torch.tensor([target_p]), torch.tensor([slot]), torch.tensor([eps_theta]),
                    torch.tensor([eps_z]), NUM_WINDINGS, 0.0, 100.0)


def _map_at(table_obj, theta_q, z_q, r, free_table=None):
    theta_q = torch.as_tensor(theta_q).reshape(-1)
    z_q = torch.as_tensor(z_q).reshape(-1).expand(theta_q.shape)
    free_table = _identity_table(theta_q.numel(), theta_q) if free_table is None else free_table
    R, S, w, valid = table_obj.anchors(theta_q, z_q)
    ray_map = _ray_map(free_table, theta_q, R, S, w, valid)
    r = torch.as_tensor(r).reshape(1, -1).expand(theta_q.numel(), -1)
    return _forward(r, free_table, theta_q, ray_map), ray_map, free_table


def test_pin_lookup_continuity():
    table_obj = _single_pin_table(1.0, 50.0, 70.0, 66.0, 0.1, 5.0)
    grid = torch.linspace(1.0, 180.0, 300)
    # Across a cell boundary: consecutive rays differ by O(delta_theta).
    width = table_obj.classes[0]['cells'].theta_width
    boundary = math.floor(1.03 / width) * width
    thetas = torch.tensor([boundary - 1e-4, boundary + 1e-4])
    s, ray_map, _ = _map_at(table_obj, thetas, 50.0, grid)
    assert (s[0] - s[1]).abs().max() < 1e-2

    # Footprint edge from both sides: the complete map equals the free map.
    for sign in (-1.0, 1.0):
        thetas = torch.tensor([1.0 + sign * (0.1 - 1e-6), 1.0 + sign * (0.1 + 1e-6)])
        s, ray_map, free_table = _map_at(table_obj, thetas, 50.0, grid)
        free = _free(grid[None].expand(2, -1), free_table, thetas)
        assert (s - free).abs().max() < 1e-4
        assert not torch.isnan(s).any()

    # Appearing anchor between two existing anchors: continuous at its footprint edge.
    pin_theta = torch.tensor([1.0, 1.0, 1.0])
    pin_z = torch.tensor([50.0, 50.0, 50.0])
    pin_r = torch.tensor([50.0, 90.0, 70.0])
    pin_target = torch.tensor([48.0, 86.0, 68.0])
    pin_slot = torch.tensor([3, 5, 4])
    eps_theta = torch.tensor([1.0, 1.0, 0.1])   # the middle anchor has the narrow footprint
    eps_z = torch.full([3], 50.0)
    three = PinTable(pin_z, pin_theta, pin_r, pin_target, pin_slot, eps_theta, eps_z, NUM_WINDINGS, 0.0, 100.0)
    thetas = torch.tensor([1.1 - 1e-6, 1.1 + 1e-6])
    s, ray_map, _ = _map_at(three, thetas, 50.0, grid)
    assert (s[0] - s[1]).abs().max() < 1e-4

    # Across the theta = 0 seam.
    seam = _single_pin_table(0.02, 50.0, 70.0, 66.0, 0.1, 5.0)
    thetas = torch.tensor([TWO_PI - 1e-5, 1e-5])
    s, ray_map, _ = _map_at(seam, thetas, 50.0, grid)
    assert (s[0] - s[1]).abs().max() < 1e-2
    # The pin is reached from the wrapped side too.
    assert bool(ray_map.valid[0, 1])

    # Zero-mass slots are never divided (no NaN/inf under anomaly detection).
    with torch.autograd.detect_anomaly():
        theta_q = torch.tensor([1.05], requires_grad=True)   # inside the pin's footprint; other slots empty
        z_q = torch.tensor([50.0])
        R, S, w, valid = table_obj.anchors(theta_q, z_q)
        free_table = _identity_table(1, theta_q.detach())
        ray_map = _ray_map(free_table, theta_q.detach(), R, S, w, valid)
        s = _forward(grid[None], free_table, theta_q.detach(), ray_map)
        assert torch.isfinite(s).all()
        (S.sum() + R.sum() + w.sum()).backward()
        assert torch.isfinite(theta_q.grad).all()

    # Outside every footprint: equals the free map.
    s, ray_map, free_table = _map_at(table_obj, torch.tensor([3.0]), 50.0, grid)
    assert torch.equal(s, _free(grid[None], free_table, torch.tensor([3.0])))


def test_pin_grid_between_pins_stays_near_targets():
    # Pins on a regular grid with the default spacing factor: between pins the
    # map stays within the off-ray bound (sheet radial gradient times half a
    # grid step) of the pin targets; no reversion to the free map.
    n_t, n_z = 9, 7
    dtheta, dz = 0.05, 8.0
    tt, zz = torch.meshgrid(1.0 + dtheta * torch.arange(n_t), 40.0 + dz * torch.arange(n_z), indexing='ij')
    pin_theta = tt.reshape(-1)
    pin_z = zz.reshape(-1)
    slope = DR / TWO_PI      # canonical sheet radial gradient per radian
    # Sheet in intermediate space: r = 70 + 2 * slope * (theta - 1) (twice the canonical slope).
    pin_r = 70.0 + 2.0 * slope * (pin_theta - 1.0)
    pin_target = DR * (4 + pin_theta / TWO_PI)   # winding 4
    pin_slot = torch.full([pin_theta.numel()], 4)
    rule = FootprintRule()
    sp_t, sp_z = grid_neighbour_spacing(tt, zz, torch.ones_like(tt, dtype=torch.bool))
    eps_theta, eps_z = rule.apply(sp_t.reshape(-1), sp_z.reshape(-1), pin_r)
    assert (eps_theta - 1.5 * dtheta).abs().max() < 1e-9
    assert (eps_z - 1.5 * dz).abs().max() < 1e-9
    table_obj = PinTable(pin_z, pin_theta, pin_r, pin_target, pin_slot, eps_theta, eps_z, NUM_WINDINGS, 0.0, 100.0)
    # Query rays between the pins (interior of the grid).
    theta_q = 1.0 + dtheta * (torch.arange(2, n_t - 2) + 0.5)
    z_q = torch.full_like(theta_q, 40.0 + dz * 3.5)
    R, S, w, valid = table_obj.anchors(theta_q, z_q)
    free_table = _identity_table(theta_q.numel(), theta_q)
    ray_map = _ray_map(free_table, theta_q, R, S, w, valid)
    r_sheet = 70.0 + 2.0 * slope * (theta_q - 1.0)
    s = _forward(r_sheet[:, None], free_table, theta_q, ray_map)[:, 0]
    target = DR * (4 + theta_q / TWO_PI)
    bound = 2.0 * slope * (dtheta / 2) + 1e-6
    assert (s - target).abs().max() < bound
    # ...whereas the free (identity) map is far off there.
    assert (_free(r_sheet[:, None], free_table, theta_q)[:, 0] - target).abs().min() > 5 * bound


# ---------------------------------------------------------------------------
# 5b. footprints
# ---------------------------------------------------------------------------


def test_pin_footprints():
    rule = FootprintRule()
    # Dense grid.
    tt, zz = torch.meshgrid(0.02 * torch.arange(6), 10.0 * torch.arange(5), indexing='ij')
    sp_t, sp_z = grid_neighbour_spacing(tt, zz, torch.ones_like(tt, dtype=torch.bool))
    eps_t, eps_z = rule.apply(sp_t.reshape(-1), sp_z.reshape(-1), torch.full([30], 500.0))
    assert (eps_t - 0.03).abs().max() < 1e-9 and (eps_z - 15.0).abs().max() < 1e-9
    # Sparse chain.
    theta_c = torch.tensor([1.0, 1.1, 1.2, 1.3])
    z_c = torch.tensor([50.0, 53.0, 56.0, 59.0])
    sp_t, sp_z = chain_neighbour_spacing(theta_c, z_c)
    eps_t, eps_z = rule.apply(sp_t, sp_z, torch.full([4], 500.0))
    assert (eps_t - 0.15).abs().max() < 1e-9 and (eps_z - 4.5).abs().max() < 1e-9
    # Isolated point: floors.
    eps_t, eps_z = rule.apply(torch.tensor([float('nan')]), torch.tensor([float('nan')]), torch.tensor([500.0]))
    assert abs(float(eps_t) - 3.0 / 500.0) < 1e-12 and float(eps_z) == 3.0

    # Grid pins do not reach rays more than one grid step outside the patch;
    # chain pins bridge the chain's own gaps.
    grid_theta = tt.reshape(-1)
    grid_z = zz.reshape(-1)
    theta_all = torch.cat([grid_theta, theta_c])
    z_all = torch.cat([grid_z, z_c])
    kinds = torch.cat([torch.zeros(30, dtype=torch.int64), torch.ones(4, dtype=torch.int64)])
    sp_t_g, sp_z_g = grid_neighbour_spacing(tt, zz, torch.ones_like(tt, dtype=torch.bool))
    sp_t_c, sp_z_c = chain_neighbour_spacing(theta_c, z_c)
    eps_t, eps_z = rule.apply(torch.cat([sp_t_g.reshape(-1), sp_t_c]), torch.cat([sp_z_g.reshape(-1), sp_z_c]),
                              torch.full([34], 500.0))
    r = torch.full([34], 70.0)
    target = torch.full([34], 66.0)
    slot = torch.full([34], 4)
    table_obj = PinTable(z_all, theta_all, r, target, slot, eps_t, eps_z, NUM_WINDINGS, 0.0, 100.0)
    # A ray 1.5 grid steps in theta outside the grid (and far from the chain): no mass.
    _, _, w, valid = table_obj.anchors(torch.tensor([0.1 + 0.03 + 1e-6]), torch.tensor([20.0]))
    assert not bool(valid.any())
    # Midway between two chain points: mass from the chain.
    _, _, w, valid = table_obj.anchors(torch.tensor([1.05]), torch.tensor([51.5]))
    assert bool(valid[0, 4]) and float(w[0, 4]) > 0


# ---------------------------------------------------------------------------
# 5c. seam crossing keeps the target
# ---------------------------------------------------------------------------


def test_pin_seam_crossing():
    registry = pins.PinRegistry(
        zyx=torch.zeros(1, 3), component=torch.zeros(1, dtype=torch.int64),
        n0=torch.tensor([3], dtype=torch.int32), theta0=torch.tensor([0.01]),
        eps_theta=torch.tensor([0.1]), eps_z=torch.tensor([5.0]), kind=torch.zeros(1, dtype=torch.int64),
        num_components=1, fixed_T=torch.zeros(1, dtype=torch.bool), fixed_T_value=torch.zeros(1),
        initial_T=torch.tensor([2.4]), fingerprint='', consistency_report={}, local_gap=torch.tensor([DR]),
        patch_component=torch.zeros(0, dtype=torch.int64), patch_offset=torch.zeros(0, dtype=torch.int64))
    T = 2.4
    theta_before = torch.tensor([0.01])
    theta_after = torch.tensor([TWO_PI - 0.01])
    n_before = registry.adjusted_n(theta_before)
    n_after = registry.adjusted_n(theta_after)
    assert int(n_before) == 3 and int(n_after) == 2
    target_before = DR * (T + n_before.double() + theta_before / TWO_PI)
    target_after = DR * (T + n_after.double() + theta_after / TWO_PI)
    # Same physical sheet: the target radius moves only by the tiny angular change.
    assert abs(float(target_after - target_before) - DR * (-0.02 / TWO_PI)) < 1e-9
    slot_before = torch.round(T + n_before.double())
    slot_after = torch.round(T + n_after.double())
    assert int(slot_before) != int(slot_after)   # a seam crossing is a slot change


# ---------------------------------------------------------------------------
# 7. gradient check
# ---------------------------------------------------------------------------


def test_pinned_map_gradcheck():
    theta = torch.tensor([0.7, 2.1])
    gen = torch.Generator().manual_seed(8)
    gaps = (DR * (1.0 + 0.3 * (torch.rand(2, NUM_WINDINGS - 1, generator=gen) - 0.5))).requires_grad_(True)
    T = torch.tensor([3.3], requires_grad=True)
    pin_r = torch.tensor([60.0, 75.0], requires_grad=True)
    n = torch.tensor([0.0, 1.0])
    query = torch.tensor([[40.0, 65.0, 120.0], [50.0, 80.0, 130.0]])

    def f(gaps, T, pin_r):
        zero = DR * theta[:, None] / TWO_PI
        table = torch.cat([zero, zero + torch.cumsum(gaps, dim=-1)], dim=-1)
        target = DR * (T + n + theta / TWO_PI)
        R = pin_r[:, None]
        S = target[:, None]
        w = torch.full_like(R, 0.8)
        ray_map = _ray_map(table, theta, R, S, w)
        return _forward(query, table, theta, ray_map), _inverse(query, table, theta, ray_map)

    assert torch.autograd.gradcheck(f, (gaps, T, pin_r), eps=1e-6, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# 7b. DT gradient on T through one pin
# ---------------------------------------------------------------------------


def test_dt_gradient_on_T_one_pin():
    theta = torch.tensor([1.0])
    table = _identity_table(1, theta)
    k = 4
    r_pin = torch.tensor([[DR * (k + 0.3 + theta[0] / TWO_PI)]])  # the sheet sits at winding 4.3 in the free map
    for T_value in np.linspace(k - 0.4, k + 0.4, 17):
        T = torch.tensor([float(T_value)], requires_grad=True)
        target = DR * (T + theta / TWO_PI)
        ray_map = _ray_map(table, theta, r_pin, target[:, None], torch.ones(1, 1))
        # Snapped, detached DT target at the same angle; its pinned inverse is
        # where the loss pulls the sample.
        snapped = (DR * (torch.round(T) + theta / TWO_PI)).detach()
        inv = _inverse(snapped[:, None], table, theta, ray_map)
        residual = (inv - r_pin).abs().sum()
        # s'(r_i) on the side of the anchor where the snapped target lies (the
        # map is piecewise linear, with a kink at the anchor); one-sided
        # numerical derivative.
        side = 1.0 if round(T_value) > T_value else -1.0
        with torch.no_grad():
            s_prime = float(side * (_forward(r_pin + side * 1e-3, table, theta, ray_map)
                                    - _forward(r_pin, table, theta, ray_map)) / 1e-3)
        expected = DR * abs(round(T_value) - T_value) / s_prime
        assert abs(float(residual) - expected) < 1e-6
        if abs(round(T_value) - T_value) < 1e-9:
            assert float(residual) < 1e-9
            continue
        residual.backward()
        assert math.copysign(1.0, float(T.grad)) == -math.copysign(1.0, round(T_value) - T_value)
    # Unsnapped target: identically zero residual, no gradient.
    T = torch.tensor([k + 0.3], requires_grad=True)
    target = DR * (T + theta / TWO_PI)
    ray_map = _ray_map(table, theta, r_pin, target[:, None], torch.ones(1, 1))
    inv = _inverse(target.detach()[:, None], table, theta, ray_map)
    residual = (inv - r_pin).abs().sum()
    assert float(residual) < 1e-9
    residual.backward()
    assert abs(float(T.grad)) < 1e-9


def test_robust_integer_offsets_isolates_the_wrong_edge():
    # Nodes 0..5 in one component; a well-connected consistent graph plus
    # one wrong edge (3 -> 4 claims +2 where every cycle says +1). The tree
    # walk would misplace node 4's subtree; the robust solve keeps the
    # consistent edges and names the wrong one.
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 3), (2, 4), (1, 5), (0, 5)]
    deltas = [1, 1, 1, 2, 1, 3, 2, 4, 5]   # true offsets n_k = k
    n, inconsistent = pins._robust_integer_offsets(
        6, edges, deltas, np.zeros(6, dtype=np.int64), 1, root_labels=[str(k) for k in range(6)])
    assert n.tolist() == [0, 1, 2, 3, 4, 5]
    assert [k for k, _ in inconsistent] == [3]
    assert inconsistent[0][1] == -1
