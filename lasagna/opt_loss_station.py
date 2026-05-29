"""Station-keeping loss: anchor the mesh to the seed point.

For each winding, initializes a seed ray anchor by minimizing station_t over
valid ray/surface intersections, then updates the tracked fractional quad
position locally on later calls. If the local quad loses the ray intersection,
the update walks direct neighbor quads before falling back to the same
brute-force ray search used for initialization.

Two loss components:
  - Normal-offset: central winding's signed offset from the seed along the
    closest point's detached model normal, applied jointly to all windings
    along that same base point normal.
  - XY-centering: per-winding, how far the closest point is from the model
    center in grid-index space, pushing each winding along model tangents.
"""
from __future__ import annotations

import torch

import model as fit_model


# Module state - set once via set_seed(), persists across stages.
_seed: torch.Tensor | None = None   # (3,) base coords
_h_frac: list[float] = []           # tracked h position per winding
_w_frac: list[float] = []           # tracked w position per winding
_anchor_initialized: list[bool] = []
_ray_dir: list[torch.Tensor | None] = []
_printed_initial: bool = False
_printed_recovery_events: int = 0

_STATION_N_LOCAL_SIGMA_IDX = 20.0
_STATION_N_OUTSIDE_WEIGHT = 0.1
_STATION_LOCAL_UPDATE_MAX_ITERS = 20
_STATION_RECOVERY_PRINT_LIMIT = 20


def set_seed(seed_xyz: torch.Tensor, data: "fit_data.FitData3D",
             *, Hm: int, Wm: int, D: int = 1) -> None:
    """Set the seed point.

    seed_xyz: (3,) tensor in base (VC3D) coords.
    data: retained for call-site compatibility.
    Hm, Wm: model grid dimensions (for initializing tracked position).
    D: number of windings.
    """
    del data

    global _seed, _h_frac, _w_frac, _anchor_initialized, _ray_dir, _printed_initial, _printed_recovery_events
    _seed = seed_xyz.detach().clone()

    # Initialize tracked position at grid center for each winding
    _h_frac = [(Hm - 1) / 2.0] * D
    _w_frac = [(Wm - 1) / 2.0] * D
    _anchor_initialized = [False] * D
    _ray_dir = [None] * D
    _printed_initial = False
    _printed_recovery_events = 0

    print(f"[station] seed=({seed_xyz[0]:.0f},{seed_xyz[1]:.0f},{seed_xyz[2]:.0f}) "
          f"anchor_source=station_ray "
          f"grid={Hm}x{Wm} D={D}", flush=True)


def reset() -> None:
    global _seed, _h_frac, _w_frac, _anchor_initialized, _ray_dir, _printed_initial, _printed_recovery_events
    _seed = None
    _h_frac = []
    _w_frac = []
    _anchor_initialized = []
    _ray_dir = []
    _printed_initial = False
    _printed_recovery_events = 0


def _station_mesh_scale(res: fit_model.FitResult3D) -> float:
    return max(1.0e-6, float(res.params.mesh_step))


def _huberized_loss_map(residual: torch.Tensor, *, delta: float) -> torch.Tensor:
    d = residual.new_tensor(max(1.0e-6, float(delta)))
    abs_r = residual.abs()
    quad = torch.minimum(abs_r, d)
    linear = abs_r - quad
    return quad.square() + 2.0 * d * linear


def _huberized_mse(residual: torch.Tensor, *, delta: float) -> torch.Tensor:
    """MSE near zero, linear beyond delta."""
    return _huberized_loss_map(residual, delta=delta).mean()


def _station_normal_weights(
    *,
    D: int,
    Hm: int,
    Wm: int,
    h_center: float,
    w_center: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    h = torch.arange(Hm, device=device, dtype=dtype).view(Hm, 1)
    w = torch.arange(Wm, device=device, dtype=dtype).view(1, Wm)
    dist_idx = torch.sqrt((h - float(h_center)).square() + (w - float(w_center)).square())
    sigma = max(1.0e-6, float(_STATION_N_LOCAL_SIGMA_IDX))
    falloff = torch.exp(-0.5 * (dist_idx / sigma).square())
    weights = float(_STATION_N_OUTSIDE_WEIGHT) + (1.0 - float(_STATION_N_OUTSIDE_WEIGHT)) * falloff
    return weights.view(1, Hm, Wm).expand(D, Hm, Wm)


def _print_initial_station_diagnostic(
    *,
    seed: torch.Tensor,
    p_int: torch.Tensor | None,
    h_frac: float | None,
    w_frac: float | None,
    normal: torch.Tensor | None,
    normal_offset: torch.Tensor | None,
    mesh_step: float,
) -> None:
    if p_int is None:
        print("[station] initial: no center-winding closest point", flush=True)
        return
    delta = p_int - seed
    dist = float(delta.norm().detach().cpu())
    step = max(1.0e-6, float(mesh_step))
    norm = dist / step
    n_off = float(normal_offset.detach().cpu()) if normal_offset is not None else float("nan")
    if abs(n_off) < 5.0e-4:
        n_off = 0.0
    p = p_int.detach().cpu().tolist()
    n = normal.detach().cpu().tolist() if normal is not None else [float("nan")] * 3
    print(
        f"[station] initial: anchor=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
        f"seed_dist={dist:.3f}vx seed_dist_norm={norm:.3f} "
        f"mesh_step={step:.3f} normal_offset={n_off:.3f}vx "
        f"anchor_normal=({n[0]:.3f},{n[1]:.3f},{n[2]:.3f}) "
        f"h={float(h_frac) if h_frac is not None else float('nan'):.3f} "
        f"w={float(w_frac) if w_frac is not None else float('nan'):.3f}",
        flush=True,
    )


def _print_station_recovery_event(message: str) -> None:
    global _printed_recovery_events
    if _printed_recovery_events >= _STATION_RECOVERY_PRINT_LIMIT:
        return
    print(message, flush=True)
    _printed_recovery_events += 1


# ---------------------------------------------------------------------------
# Analytic ray-quad intersection compatibility helper
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

    Kept here for callers that still need station-style cached ray updates.
    Station loss itself now uses the closest-point helpers below.
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

    abs_a = [aa.abs().item() for aa in alphas]
    best_idx = max(range(3), key=lambda i: abs_a[i])
    alpha = alphas[best_idx]
    beta = betas_q[best_idx]
    gamma = gammas[best_idx]

    alpha_f = alpha.item()
    beta_f = beta.item()
    gamma_f = gamma.item()

    if abs(alpha_f) < eps:
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

    denom_v = [Bp[k].item() + u_val * Cp[k].item() for k in range(3)]
    numer_v = [-(Gp[k].item() + u_val * Ap[k].item()) for k in range(3)]
    abs_dv = [abs(d) for d in denom_v]
    best_v = max(range(3), key=lambda i: abs_dv[i])
    if abs_dv[best_v] < eps:
        v_val = frac_w
    else:
        v_val = numer_v[best_v] / denom_v[best_v]

    u_t = torch.tensor(u_val, device=O.device, dtype=O.dtype)
    v_t = torch.tensor(v_val, device=O.device, dtype=O.dtype)
    conn_pt = P00 + u_t * a + v_t * b + (u_t * v_t) * c

    return u_val, v_val, conn_pt


# ---------------------------------------------------------------------------
# Triangle closest-point helper retained for legacy callers
# ---------------------------------------------------------------------------

def _closest_points_on_triangles(
    p: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized closest point from p to triangles (a, b, c).

    Returns closest points and barycentric coordinates in the same vertex order.
    """
    ab = b - a
    ac = c - a
    ap = p.view(1, 3) - a

    def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x * y).sum(dim=-1)

    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    bp = p.view(1, 3) - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    cp = p.view(1, 3) - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    denom = (va + vb + vc).clamp_min(eps)
    face_v = vb / denom
    face_w = vc / denom
    closest = a + face_v.view(-1, 1) * ab + face_w.view(-1, 1) * ac
    bary = torch.stack((1.0 - face_v - face_w, face_v, face_w), dim=-1)

    mask_a = (d1 <= 0.0) & (d2 <= 0.0)
    closest = torch.where(mask_a.view(-1, 1), a, closest)
    bary = torch.where(
        mask_a.view(-1, 1),
        torch.tensor([1.0, 0.0, 0.0], device=a.device, dtype=a.dtype).view(1, 3),
        bary,
    )

    mask_b = (d3 >= 0.0) & (d4 <= d3)
    closest = torch.where(mask_b.view(-1, 1), b, closest)
    bary = torch.where(
        mask_b.view(-1, 1),
        torch.tensor([0.0, 1.0, 0.0], device=a.device, dtype=a.dtype).view(1, 3),
        bary,
    )

    mask_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    ab_v = d1 / (d1 - d3).clamp_min(eps)
    closest_ab = a + ab_v.view(-1, 1) * ab
    bary_ab = torch.stack((1.0 - ab_v, ab_v, torch.zeros_like(ab_v)), dim=-1)
    closest = torch.where(mask_ab.view(-1, 1), closest_ab, closest)
    bary = torch.where(mask_ab.view(-1, 1), bary_ab, bary)

    mask_c = (d6 >= 0.0) & (d5 <= d6)
    closest = torch.where(mask_c.view(-1, 1), c, closest)
    bary = torch.where(
        mask_c.view(-1, 1),
        torch.tensor([0.0, 0.0, 1.0], device=a.device, dtype=a.dtype).view(1, 3),
        bary,
    )

    mask_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    ac_w = d2 / (d2 - d6).clamp_min(eps)
    closest_ac = a + ac_w.view(-1, 1) * ac
    bary_ac = torch.stack((1.0 - ac_w, torch.zeros_like(ac_w), ac_w), dim=-1)
    closest = torch.where(mask_ac.view(-1, 1), closest_ac, closest)
    bary = torch.where(mask_ac.view(-1, 1), bary_ac, bary)

    mask_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    bc_w = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(eps)
    closest_bc = b + bc_w.view(-1, 1) * (c - b)
    bary_bc = torch.stack((torch.zeros_like(bc_w), 1.0 - bc_w, bc_w), dim=-1)
    closest = torch.where(mask_bc.view(-1, 1), closest_bc, closest)
    bary = torch.where(mask_bc.view(-1, 1), bary_bc, bary)

    return closest, bary


def _quad_model_normal(
    P00: torch.Tensor,
    P10: torch.Tensor,
    P01: torch.Tensor,
    P11: torch.Tensor,
    frac_h: float,
    frac_w: float,
) -> torch.Tensor:
    u = torch.tensor(frac_h, device=P00.device, dtype=P00.dtype)
    v = torch.tensor(frac_w, device=P00.device, dtype=P00.dtype)
    dh = (1.0 - v) * (P10 - P00) + v * (P11 - P01)
    dw = (1.0 - u) * (P01 - P00) + u * (P11 - P10)
    n = torch.cross(dh, dw, dim=0)
    n_norm = n.norm()
    if float(n_norm.detach().cpu()) < 1.0e-8:
        n = torch.tensor([0.0, 0.0, 1.0], device=P00.device, dtype=P00.dtype)
        n_norm = n.norm()
    return n / n_norm.clamp(min=1.0e-8)


def _bilinear_point(
    P00: torch.Tensor,
    P10: torch.Tensor,
    P01: torch.Tensor,
    P11: torch.Tensor,
    frac_h: float,
    frac_w: float,
) -> torch.Tensor:
    u = torch.tensor(frac_h, device=P00.device, dtype=P00.dtype)
    v = torch.tensor(frac_w, device=P00.device, dtype=P00.dtype)
    return P00 + u * (P10 - P00) + v * (P01 - P00) + (u * v) * (P11 - P10 - P01 + P00)


def _station_seed_ray_direction(
    *,
    seed: torch.Tensor,
    ref_point: torch.Tensor,
    fallback_normal: torch.Tensor,
) -> torch.Tensor:
    ray = ref_point - seed
    ray_norm = ray.norm()
    fallback = fallback_normal / fallback_normal.norm().clamp(min=1.0e-8)
    if float(ray_norm.detach().cpu()) <= 1.0e-6:
        return fallback
    ray_unit = ray / ray_norm.clamp(min=1.0e-8)
    if float((ray_unit * fallback).sum().abs().detach().cpu()) <= 1.0e-3:
        return fallback
    return ray_unit


def _station_seed_ray_direction_at_grid_position(
    *,
    seed: torch.Tensor,
    surf: torch.Tensor,
    h_ref: float,
    w_ref: float,
) -> torch.Tensor | None:
    Hm, Wm, _ = surf.shape
    if Hm < 2 or Wm < 2:
        return None
    row = max(0, min(int(h_ref), Hm - 2))
    col = max(0, min(int(w_ref), Wm - 2))
    frac_h = max(0.0, min(1.0, float(h_ref) - float(row)))
    frac_w = max(0.0, min(1.0, float(w_ref) - float(col)))
    n_fallback = _quad_model_normal(
        surf[row, col],
        surf[row + 1, col],
        surf[row, col + 1],
        surf[row + 1, col + 1],
        frac_h,
        frac_w,
    )
    ref_point = _bilinear_point(
        surf[row, col],
        surf[row + 1, col],
        surf[row, col + 1],
        surf[row + 1, col + 1],
        frac_h,
        frac_w,
    )
    return _station_seed_ray_direction(seed=seed, ref_point=ref_point, fallback_normal=n_fallback)


def _station_anchor_from_local_ray_update(
    seed: torch.Tensor,
    surf: torch.Tensor,
    h_ref: float,
    w_ref: float,
    n_ray: torch.Tensor,
    *,
    max_iters: int = _STATION_LOCAL_UPDATE_MAX_ITERS,
) -> tuple[tuple[torch.Tensor, float, float, torch.Tensor] | None, int]:
    Hm, Wm, _ = surf.shape
    if Hm < 2 or Wm < 2:
        return None, 0
    if not (torch.isfinite(seed).all().item()):
        return None, 0
    if not (torch.isfinite(n_ray).all().item()):
        return None, 0
    n_ray = n_ray.to(device=surf.device, dtype=surf.dtype)
    n_ray = n_ray / n_ray.norm().clamp(min=1.0e-8)

    row = max(0, min(int(h_ref), Hm - 2))
    col = max(0, min(int(w_ref), Wm - 2))
    frac_h = max(0.0, min(1.0, float(h_ref) - float(row)))
    frac_w = max(0.0, min(1.0, float(w_ref) - float(col)))

    max_iters = max(1, int(max_iters))
    for it in range(1, max_iters + 1):
        try:
            u, v, point = _intersect_single_quad(
                seed,
                n_ray,
                surf[row, col],
                surf[row + 1, col],
                surf[row, col + 1],
                surf[row + 1, col + 1],
                frac_h,
                frac_w,
            )
        except Exception:
            return None, it
        if not (torch.isfinite(torch.tensor([u, v], device=surf.device, dtype=surf.dtype)).all().item()):
            return None, it
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            h_frac = max(0.0, min(float(Hm - 1), float(row) + float(u)))
            w_frac = max(0.0, min(float(Wm - 1), float(col) + float(v)))
            normal = _quad_model_normal(
                surf[row, col],
                surf[row + 1, col],
                surf[row, col + 1],
                surf[row + 1, col + 1],
                float(u),
                float(v),
            )
            return (point, h_frac, w_frac, normal), it

        step_h = -1 if u < 0.0 else (1 if u > 1.0 else 0)
        step_w = -1 if v < 0.0 else (1 if v > 1.0 else 0)
        new_row = max(0, min(Hm - 2, row + step_h))
        new_col = max(0, min(Wm - 2, col + step_w))
        if new_row == row and new_col == col:
            return None, it
        target_h = row + float(u)
        target_w = col + float(v)
        row = new_row
        col = new_col
        frac_h = max(0.0, min(1.0, target_h - float(row)))
        frac_w = max(0.0, min(1.0, target_w - float(col)))

    return None, max_iters


def _station_anchor_from_ray_min_station_t(
    seed: torch.Tensor,
    surf: torch.Tensor,
    h_ref: float,
    w_ref: float,
) -> tuple[tuple[torch.Tensor, float, float, torch.Tensor], torch.Tensor] | None:
    Hm, Wm, _ = surf.shape
    if Hm < 2 or Wm < 2:
        return None
    n_ray = _station_seed_ray_direction_at_grid_position(
        seed=seed,
        surf=surf,
        h_ref=h_ref,
        w_ref=w_ref,
    )
    if n_ray is None:
        return None

    p00 = surf[:-1, :-1].reshape(-1, 3)
    p10 = surf[1:, :-1].reshape(-1, 3)
    p01 = surf[:-1, 1:].reshape(-1, 3)
    p11 = surf[1:, 1:].reshape(-1, 3)
    row_ids = (
        torch.arange(Hm - 1, device=surf.device, dtype=surf.dtype)
        .view(Hm - 1, 1)
        .expand(Hm - 1, Wm - 1)
        .reshape(-1)
    )
    col_ids = (
        torch.arange(Wm - 1, device=surf.device, dtype=surf.dtype)
        .view(1, Wm - 1)
        .expand(Hm - 1, Wm - 1)
        .reshape(-1)
    )
    frac_h0 = (torch.full_like(row_ids, float(h_ref)) - row_ids).clamp(0.0, 1.0)
    frac_w0 = (torch.full_like(col_ids, float(w_ref)) - col_ids).clamp(0.0, 1.0)
    n = n_ray.view(1, 3).expand_as(p00)
    u, v = fit_model.Model3D._ray_bilinear_intersect(
        seed.view(1, 3).expand_as(p00),
        n,
        p00,
        p10,
        p01,
        p11,
        frac_h0,
        frac_w0,
    )
    valid = torch.isfinite(u) & torch.isfinite(v) & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    if not bool(valid.any().detach().cpu()):
        return None
    h_all = row_ids + u
    w_all = col_ids + v
    station_t = (h_all - float(h_ref)).square() + (w_all - float(w_ref)).square()
    score = torch.where(valid, station_t, torch.full_like(station_t, float("inf")))
    idx = int(torch.argmin(score).detach().cpu())
    row = int(row_ids[idx].detach().cpu())
    col = int(col_ids[idx].detach().cpu())
    h_frac = float(h_all[idx].detach().cpu())
    w_frac = float(w_all[idx].detach().cpu())
    frac_h = float(u[idx].detach().cpu())
    frac_w = float(v[idx].detach().cpu())
    point = _bilinear_point(
        surf[row, col],
        surf[row + 1, col],
        surf[row, col + 1],
        surf[row + 1, col + 1],
        frac_h,
        frac_w,
    )
    normal = _quad_model_normal(
        surf[row, col],
        surf[row + 1, col],
        surf[row, col + 1],
        surf[row + 1, col + 1],
        frac_h,
        frac_w,
    )
    return (point, h_frac, w_frac, normal), n_ray.detach()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def station_loss(
    *, res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Station-keeping loss anchored to seed point."""
    global _h_frac, _w_frac, _anchor_initialized, _ray_dir, _printed_initial

    dev = res.xyz_lr.device
    D, Hm, Wm, _ = res.xyz_lr.shape
    zero = res.xyz_lr.new_zeros(())
    ones = res.xyz_lr.new_ones(1, 1, 1, 1)

    dummy = (zero.view(1, 1, 1, 1),)
    if _seed is None:
        return {
            "station_n": (zero, dummy, (ones,)),
            "station_t": (zero, dummy, (ones,)),
        }

    seed = _seed.to(device=dev, dtype=res.xyz_lr.dtype)
    d_center = (D - 1) // 2
    xyz_det = res.xyz_lr.detach()

    # Ensure tracked state matches current D
    while len(_h_frac) < D:
        _h_frac.append((Hm - 1) / 2.0)
        _w_frac.append((Wm - 1) / 2.0)
    while len(_anchor_initialized) < D:
        _anchor_initialized.append(False)
    while len(_ray_dir) < D:
        _ray_dir.append(None)

    h_mid = (Hm - 1) / 2.0
    w_mid = (Wm - 1) / 2.0
    mesh_scale = _station_mesh_scale(res)
    huber_delta = 1.0

    # --- Station ray anchor per winding (non-differentiable anchor update) ---
    anchors: list[tuple[int, torch.Tensor, float, float, torch.Tensor]] = []
    with torch.no_grad():
        for d in range(D):
            surf = xyz_det[d]  # (Hm, Wm, 3)
            if _anchor_initialized[d]:
                ray = _ray_dir[d]
                if ray is None:
                    ray = _station_seed_ray_direction_at_grid_position(
                        seed=seed,
                        surf=surf,
                        h_ref=_h_frac[d],
                        w_ref=_w_frac[d],
                    )
                    _ray_dir[d] = None if ray is None else ray.detach()
                if ray is None:
                    anchor = None
                    update_iters = 0
                else:
                    anchor, update_iters = _station_anchor_from_local_ray_update(
                        seed,
                        surf,
                        h_ref=_h_frac[d],
                        w_ref=_w_frac[d],
                        n_ray=ray,
                    )
                if anchor is not None and update_iters > 1:
                    _print_station_recovery_event(
                        f"[station] local ray recovery d={d} iters={update_iters} "
                        f"from=({_h_frac[d]:.3f},{_w_frac[d]:.3f}) "
                        f"to=({anchor[1]:.3f},{anchor[2]:.3f})"
                    )
                if anchor is None:
                    _print_station_recovery_event(
                        f"[station] local ray recovery failed d={d} "
                        f"iters={update_iters}; running brute-force ray search"
                    )
                    anchor = _station_anchor_from_ray_min_station_t(
                        seed,
                        surf,
                        h_ref=_h_frac[d],
                        w_ref=_w_frac[d],
                    )
                    if anchor is not None:
                        anchor, ray = anchor
                        _ray_dir[d] = ray.detach()
                        _print_station_recovery_event(
                            f"[station] brute-force ray recovery d={d} "
                            f"from=({_h_frac[d]:.3f},{_w_frac[d]:.3f}) "
                            f"to=({anchor[1]:.3f},{anchor[2]:.3f})"
                        )
            else:
                anchor = _station_anchor_from_ray_min_station_t(
                    seed,
                    surf,
                    h_ref=_h_frac[d],
                    w_ref=_w_frac[d],
                )
                if anchor is not None:
                    anchor, ray = anchor
                    _ray_dir[d] = ray.detach()
                    _anchor_initialized[d] = True
            if anchor is None:
                continue

            conn_pt, h_frac, w_frac, normal = anchor
            _h_frac[d] = h_frac
            _w_frac[d] = w_frac
            anchors.append((d, conn_pt, h_frac, w_frac, normal))

    # --- Normal-offset loss (from central winding, applied jointly) ---
    loss_normal = zero
    with torch.no_grad():
        center_hits = [x for x in anchors if x[0] == d_center]
        if center_hits:
            _, p_int_c, h_int_c, w_int_c, n_model_c = center_hits[0]
            offset = ((p_int_c - seed) * n_model_c).sum()  # signed scalar
            target_n = xyz_det - offset * n_model_c
            normal_weights = _station_normal_weights(
                D=D,
                Hm=Hm,
                Wm=Wm,
                h_center=h_int_c,
                w_center=w_int_c,
                device=dev,
                dtype=res.xyz_lr.dtype,
            )
        else:
            p_int_c = None
            h_int_c = None
            w_int_c = None
            n_model_c = None
            offset = None
            target_n = None
            normal_weights = None
        if not _printed_initial:
            _print_initial_station_diagnostic(
                seed=seed,
                p_int=p_int_c,
                h_frac=h_int_c,
                w_frac=w_int_c,
                normal=n_model_c,
                normal_offset=offset,
                mesh_step=float(res.params.mesh_step),
            )
            _printed_initial = True

    if target_n is not None:
        normal_distance_vx = ((res.xyz_lr - target_n) * n_model_c.view(1, 1, 1, 3)).sum(dim=-1)
        normal_lm = _huberized_loss_map(normal_distance_vx / mesh_scale, delta=huber_delta)
        loss_normal = (normal_lm * normal_weights).mean()

    # --- XY-centering loss (per-winding, independent) ---
    loss_xy = zero
    n_xy_hits = 0

    for d, _, h_frac_val, w_frac_val, _ in anchors:
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

        tangent_distance_vx = (res.xyz_lr[d] - target_xy_d).norm(dim=-1)
        loss_xy = loss_xy + _huberized_mse(tangent_distance_vx / mesh_scale, delta=huber_delta)
        n_xy_hits += 1

    if n_xy_hits > 0:
        loss_xy = loss_xy / n_xy_hits

    mask = res.mask_lr
    if target_n is not None:
        lm_n = (normal_lm * normal_weights).detach().unsqueeze(1)
    else:
        lm_n = loss_normal.detach().expand(D, 1, Hm, Wm)
    lm_t = loss_xy.detach().expand(D, 1, Hm, Wm)
    return {
        "station_n": (loss_normal, (lm_n,), (mask,)),
        "station_t": (loss_xy, (lm_t,), (mask,)),
    }
