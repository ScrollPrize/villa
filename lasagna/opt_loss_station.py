"""Station-keeping loss: anchor the mesh to the seed point.

Tracks the ray-surface intersection position incrementally (one quad per
winding, analytic solve) instead of brute-forcing all quads every step.

Two loss components:
  - Normal-offset: central winding's offset from seed along the GT normal
    → pushes all windings jointly along their per-vertex GT normals.
  - XY-centering: per-winding, how far the intersection is from the model
    center in grid-index space → pushes each winding tangentially (GT tangent).
"""
from __future__ import annotations

import torch

import model as fit_model


# Module state — set once via set_seed(), persists across stages.
_seed: torch.Tensor | None = None   # (3,) base coords
_n_gt: torch.Tensor | None = None   # (3,) GT unit normal at seed
_h_frac: list[float] = []           # tracked h position per winding
_w_frac: list[float] = []           # tracked w position per winding


def set_seed(seed_xyz: torch.Tensor, data: "fit_data.FitData3D",
             *, Hm: int, Wm: int, D: int = 1) -> None:
    """Set the seed point and sample the GT normal from the data volume.

    seed_xyz: (3,) tensor in base (VC3D) coords.
    data: loaded FitData3D for grid_sample.
    Hm, Wm: model grid dimensions (for initializing tracked position).
    D: number of windings.
    """
    global _seed, _n_gt, _h_frac, _w_frac
    _seed = seed_xyz.detach().clone()

    # Sample GT normal at seed point
    query = seed_xyz.view(1, 1, 1, 3)  # (D=1, H=1, W=1, 3)
    sampled = data.grid_sample_fullres(query)
    nx = sampled.nx.squeeze().item()
    ny = sampled.ny.squeeze().item()
    nz_sq = max(0.0, 1.0 - nx * nx - ny * ny)
    nz = nz_sq ** 0.5
    _n_gt = torch.tensor([nx, ny, nz], device=seed_xyz.device, dtype=torch.float32)
    norm = _n_gt.norm().clamp(min=1e-8)
    _n_gt = _n_gt / norm

    # Initialize tracked position at grid center for each winding
    _h_frac = [(Hm - 1) / 2.0] * D
    _w_frac = [(Wm - 1) / 2.0] * D

    print(f"[station] seed=({seed_xyz[0]:.0f},{seed_xyz[1]:.0f},{seed_xyz[2]:.0f}) "
          f"n_gt=({_n_gt[0]:.3f},{_n_gt[1]:.3f},{_n_gt[2]:.3f}) "
          f"grid={Hm}x{Wm} D={D}", flush=True)


def reset() -> None:
    global _seed, _n_gt, _h_frac, _w_frac
    _seed = None
    _n_gt = None
    _h_frac = []
    _w_frac = []


# ---------------------------------------------------------------------------
# Analytic ray-quad intersection (single quad, scalar)
# ---------------------------------------------------------------------------

def _intersect_single_quad(
    O: torch.Tensor,    # (3,) ray origin
    n: torch.Tensor,    # (3,) ray direction
    P00: torch.Tensor,  # (3,)
    P10: torch.Tensor,  # (3,)
    P01: torch.Tensor,  # (3,)
    P11: torch.Tensor,  # (3,)
    frac_h: float,      # expected u (for root selection)
    frac_w: float,      # expected v (for root selection)
    eps: float = 1e-12,
) -> tuple[float, float, torch.Tensor]:
    """Analytic ray vs bilinear quad intersection.

    Same solver as _intersect_ext_surfaces in model.py.
    Returns (u, v, conn_pt).
    """
    a = P10 - P00
    b = P01 - P00
    c = P11 - P10 - P01 + P00
    g = P00 - O

    def cross2(vec: torch.Tensor, i: int, j: int) -> torch.Tensor:
        return vec[i] * n[j] - vec[j] * n[i]

    Ap = [cross2(a, 0, 1), cross2(a, 0, 2), cross2(a, 1, 2)]
    Bp = [cross2(b, 0, 1), cross2(b, 0, 2), cross2(b, 1, 2)]
    Cp = [cross2(c, 0, 1), cross2(c, 0, 2), cross2(c, 1, 2)]
    Gp = [cross2(g, 0, 1), cross2(g, 0, 2), cross2(g, 1, 2)]

    qpairs = [(0, 1), (0, 2), (1, 2)]
    alphas, betas_q, gammas = [], [], []
    for p, q in qpairs:
        alphas.append(Ap[p] * Cp[q] - Ap[q] * Cp[p])
        betas_q.append(Ap[p] * Bp[q] - Ap[q] * Bp[p] + Gp[p] * Cp[q] - Gp[q] * Cp[p])
        gammas.append(Gp[p] * Bp[q] - Gp[q] * Bp[p])

    # Pick best-conditioned plane pair
    abs_a = [aa.abs().item() for aa in alphas]
    best_idx = max(range(3), key=lambda i: abs_a[i])
    alpha = alphas[best_idx]
    beta = betas_q[best_idx]
    gamma = gammas[best_idx]

    # Quadratic for u
    alpha_f = alpha.item()
    beta_f = beta.item()
    gamma_f = gamma.item()

    if abs(alpha_f) < eps:
        # Linear
        if abs(beta_f) < eps:
            u_val = frac_h
        else:
            u_val = -gamma_f / beta_f
    else:
        disc = beta_f * beta_f - 4.0 * alpha_f * gamma_f
        if disc < 0:
            disc = 0.0
        sqrt_disc = disc ** 0.5
        u1 = (-beta_f + sqrt_disc) / (2.0 * alpha_f)
        u2 = (-beta_f - sqrt_disc) / (2.0 * alpha_f)
        u_val = u1 if abs(u1 - frac_h) <= abs(u2 - frac_h) else u2

    # Back-substitute for v: pick best-conditioned denominator
    denom_v = [Bp[k].item() + u_val * Cp[k].item() for k in range(3)]
    numer_v = [-(Gp[k].item() + u_val * Ap[k].item()) for k in range(3)]
    abs_dv = [abs(d) for d in denom_v]
    best_v = max(range(3), key=lambda i: abs_dv[i])
    if abs_dv[best_v] < eps:
        v_val = frac_w
    else:
        v_val = numer_v[best_v] / denom_v[best_v]

    # Intersection point
    u_t = torch.tensor(u_val, device=O.device, dtype=O.dtype)
    v_t = torch.tensor(v_val, device=O.device, dtype=O.dtype)
    conn_pt = P00 + u_t * a + v_t * b + (u_t * v_t) * c

    return u_val, v_val, conn_pt


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def station_loss(
    *, res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Station-keeping loss anchored to seed point."""
    global _h_frac, _w_frac

    dev = res.xyz_lr.device
    D, Hm, Wm, _ = res.xyz_lr.shape
    zero = torch.zeros((), device=dev)
    ones = torch.ones(1, 1, 1, 1, device=dev)

    dummy = (zero.view(1, 1, 1, 1),)
    if _seed is None or _n_gt is None:
        return {
            "station_n": (zero, dummy, (ones,)),
            "station_t": (zero, dummy, (ones,)),
        }

    seed = _seed.to(dev)
    n_gt = _n_gt.to(dev)
    d_center = (D - 1) // 2
    xyz_det = res.xyz_lr.detach()

    # Ensure tracked state matches current D
    while len(_h_frac) < D:
        _h_frac.append((Hm - 1) / 2.0)
        _w_frac.append((Wm - 1) / 2.0)

    h_mid = (Hm - 1) / 2.0
    w_mid = (Wm - 1) / 2.0

    # --- Tracked intersection per winding (non-differentiable) ---
    intersections: list[tuple[int, torch.Tensor, float, float]] = []
    with torch.no_grad():
        for d in range(D):
            hf = _h_frac[d]
            wf = _w_frac[d]

            # Clamp to valid quad range
            row = max(0, min(int(hf), Hm - 2))
            col = max(0, min(int(wf), Wm - 2))
            frac_h = hf - row
            frac_w = wf - col

            surf = xyz_det[d]  # (Hm, Wm, 3)
            P00 = surf[row, col]
            P10 = surf[row + 1, col]
            P01 = surf[row, col + 1]
            P11 = surf[row + 1, col + 1]

            u_val, v_val, conn_pt = _intersect_single_quad(
                seed, n_gt, P00, P10, P01, P11, frac_h, frac_w)

            # Update tracked position and clamp to surface
            new_hf = row + u_val
            new_wf = col + v_val
            _h_frac[d] = max(0.0, min(float(Hm - 1), new_hf))
            _w_frac[d] = max(0.0, min(float(Wm - 1), new_wf))

            intersections.append((d, conn_pt, _h_frac[d], _w_frac[d]))

    # --- Normal-offset loss (from central winding, applied jointly) ---
    loss_normal = zero
    with torch.no_grad():
        center_hits = [x for x in intersections if x[0] == d_center]
        if center_hits:
            _, p_int_c, _, _ = center_hits[0]
            offset = ((p_int_c - seed) * n_gt).sum()  # signed scalar
            target_n = xyz_det - offset * n_gt  # broadcast (D, Hm, Wm, 3)
        else:
            target_n = None

    if target_n is not None:
        loss_normal = (res.xyz_lr - target_n).square().mean()

    # --- XY-centering loss (per-winding, independent) ---
    loss_xy = zero
    n_xy_hits = 0

    for d, _, h_frac_val, w_frac_val in intersections:
        with torch.no_grad():
            dh = h_frac_val - h_mid
            dw = w_frac_val - w_mid
            # Per-vertex tangent directions from finite differences
            surf = xyz_det[d]  # (Hm, Wm, 3)
            th = torch.zeros_like(surf)
            th[:-1] = surf[1:] - surf[:-1]
            th[-1] = th[-2]
            tw = torch.zeros_like(surf)
            tw[:, :-1] = surf[:, 1:] - surf[:, :-1]
            tw[:, -1] = tw[:, -2]
            target_xy_d = surf + dh * th + dw * tw  # (Hm, Wm, 3)

        loss_xy = loss_xy + (res.xyz_lr[d] - target_xy_d).square().mean()
        n_xy_hits += 1

    if n_xy_hits > 0:
        loss_xy = loss_xy / n_xy_hits

    mask = res.mask_lr
    lm_n = loss_normal.detach().expand(D, 1, Hm, Wm)
    lm_t = loss_xy.detach().expand(D, 1, Hm, Wm)
    return {
        "station_n": (loss_normal, (lm_n,), (mask,)),
        "station_t": (loss_xy, (lm_t,), (mask,)),
    }
