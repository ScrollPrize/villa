"""Station-keeping loss: anchor the mesh to the seed point.

Intersects a ray (seed + t * GT_normal) with all winding surfaces using
ray-quad intersection. Everything except the final L2 is non-differentiable
— gradients come purely from the proxy-point construction.

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


def set_seed(seed_xyz: torch.Tensor, data: "fit_data.FitData3D") -> None:
    """Set the seed point and sample the GT normal from the data volume.

    seed_xyz: (3,) tensor in base (VC3D) coords.
    data: loaded FitData3D for grid_sample.
    """
    global _seed, _n_gt
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
    print(f"[station] seed=({seed_xyz[0]:.0f},{seed_xyz[1]:.0f},{seed_xyz[2]:.0f}) "
          f"n_gt=({_n_gt[0]:.3f},{_n_gt[1]:.3f},{_n_gt[2]:.3f})", flush=True)


def reset() -> None:
    global _seed, _n_gt
    _seed = None
    _n_gt = None


# ---------------------------------------------------------------------------
# Ray-quad intersection (bilinear patch, iterative Newton)
# ---------------------------------------------------------------------------

def _ray_quad_intersect(
    origin: torch.Tensor,     # (3,)
    direction: torch.Tensor,  # (3,)
    p00: torch.Tensor,        # (N, 3)
    p10: torch.Tensor,        # (N, 3)
    p01: torch.Tensor,        # (N, 3)
    p11: torch.Tensor,        # (N, 3)
    n_iters: int = 8,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized ray vs bilinear quad intersection.

    Bilinear patch: P(u,v) = (1-u)(1-v)*p00 + u*(1-v)*p10 + (1-u)*v*p01 + u*v*p11
    Ray: R(t) = origin + t*direction
    Solve P(u,v) = R(t) via Newton iteration.

    Returns (t, u, v, hit) all shape (N,). hit is bool mask.
    """
    N = p00.shape[0]
    dev = p00.device

    # Initial guess: u=0.5, v=0.5, t from centroid
    u = torch.full((N,), 0.5, device=dev)
    v = torch.full((N,), 0.5, device=dev)
    centroid = 0.25 * (p00 + p10 + p01 + p11)
    t = ((centroid - origin.unsqueeze(0)) * direction.unsqueeze(0)).sum(-1) / \
        (direction * direction).sum().clamp(min=eps)

    for _ in range(n_iters):
        # P(u,v)
        P = ((1 - u) * (1 - v)).unsqueeze(-1) * p00 + \
            (u * (1 - v)).unsqueeze(-1) * p10 + \
            ((1 - u) * v).unsqueeze(-1) * p01 + \
            (u * v).unsqueeze(-1) * p11  # (N, 3)

        # Residual: P(u,v) - R(t)
        R = origin.unsqueeze(0) + t.unsqueeze(-1) * direction.unsqueeze(0)
        residual = P - R  # (N, 3)

        # Jacobian columns: dP/du, dP/dv, -direction
        dPdu = (-(1 - v)).unsqueeze(-1) * p00 + ((1 - v)).unsqueeze(-1) * p10 + \
               (-v).unsqueeze(-1) * p01 + v.unsqueeze(-1) * p11  # (N, 3)
        dPdv = (-(1 - u)).unsqueeze(-1) * p00 + (-u).unsqueeze(-1) * p10 + \
               ((1 - u)).unsqueeze(-1) * p01 + u.unsqueeze(-1) * p11  # (N, 3)
        neg_d = -direction.unsqueeze(0).expand(N, 3)  # (N, 3)

        # Solve 3x3 system [dPdu | dPdv | neg_d] * [du, dv, dt]^T = -residual
        # Using Cramer's rule vectorized
        J0, J1, J2 = dPdu, dPdv, neg_d
        rhs = -residual

        # Cross products for Cramer
        c01 = torch.linalg.cross(J0, J1)  # (N, 3)
        det = (c01 * J2).sum(-1)  # (N,)
        safe_det = torch.where(det.abs() > eps, det, torch.ones_like(det))

        c_rhs_1 = torch.linalg.cross(rhs, J1)
        du = (c_rhs_1 * J2).sum(-1) / safe_det
        c0_rhs = torch.linalg.cross(J0, rhs)
        dv = (c0_rhs * J2).sum(-1) / safe_det
        dt = (c01 * rhs).sum(-1) / safe_det

        # Mask degenerate
        good = det.abs() > eps
        u = u + torch.where(good, du, torch.zeros_like(du))
        v = v + torch.where(good, dv, torch.zeros_like(dv))
        t = t + torch.where(good, dt, torch.zeros_like(dt))

    # Check convergence: u,v in [0,1] and residual small
    P_final = ((1 - u) * (1 - v)).unsqueeze(-1) * p00 + \
              (u * (1 - v)).unsqueeze(-1) * p10 + \
              ((1 - u) * v).unsqueeze(-1) * p01 + \
              (u * v).unsqueeze(-1) * p11
    R_final = origin.unsqueeze(0) + t.unsqueeze(-1) * direction.unsqueeze(0)
    res_norm = (P_final - R_final).norm(dim=-1)

    hit = (u >= -eps) & (u <= 1.0 + eps) & (v >= -eps) & (v <= 1.0 + eps) & (res_norm < 1.0)

    return t, u, v, hit


def _intersect_winding(
    seed: torch.Tensor,     # (3,)
    n_gt: torch.Tensor,     # (3,)
    surf: torch.Tensor,     # (H, W, 3) — detached
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Intersect ray with one winding surface using quad patches.

    All computation is non-differentiable.
    Returns (p_int, h_frac, w_frac, quad_normal, did_hit).
    quad_normal is the geometric normal of the hit quad (for orienting per-vertex normals).
    """
    H, W, _ = surf.shape
    # Build quad corners — all (H-1)*(W-1) quads in parallel
    p00 = surf[:-1, :-1].reshape(-1, 3)
    p10 = surf[1:, :-1].reshape(-1, 3)
    p01 = surf[:-1, 1:].reshape(-1, 3)
    p11 = surf[1:, 1:].reshape(-1, 3)

    t, u, v, hit = _ray_quad_intersect(seed, n_gt, p00, p10, p01, p11)

    if not hit.any():
        dev = surf.device
        z3 = torch.zeros(3, device=dev)
        return (z3, torch.tensor(0.0, device=dev),
                torch.tensor(0.0, device=dev), z3, False)

    # Grid indices for each quad
    rows = torch.arange(H - 1, device=surf.device).unsqueeze(1).expand(H - 1, W - 1).reshape(-1).float()
    cols = torch.arange(W - 1, device=surf.device).unsqueeze(0).expand(H - 1, W - 1).reshape(-1).float()

    # Fractional grid position: quad (r,c) with parameters (u,v)
    h_frac = rows + u
    w_frac = cols + v

    # Pick the hit with smallest |t|
    t_masked = torch.where(hit, t.abs(), torch.full_like(t, float("inf")))
    best = t_masked.argmin()

    # Intersection point on the bilinear patch
    ub, vb = u[best], v[best]
    p_int = ((1 - ub) * (1 - vb)) * p00[best] + (ub * (1 - vb)) * p10[best] + \
            ((1 - ub) * vb) * p01[best] + (ub * vb) * p11[best]

    # Quad geometric normal from diagonals cross product
    diag1 = p11[best] - p00[best]
    diag2 = p01[best] - p10[best]
    quad_normal = torch.linalg.cross(diag1, diag2)
    quad_normal = quad_normal / quad_normal.norm().clamp(min=1e-8)

    return p_int, h_frac[best], w_frac[best], quad_normal, True


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def station_loss(
    *, res: fit_model.FitResult3D,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Station-keeping loss anchored to seed point.

    Everything except the final L2(xyz_lr, proxy) is computed without gradients.
    """
    dev = res.xyz_lr.device
    D, Hm, Wm, _ = res.xyz_lr.shape
    zero = torch.zeros((), device=dev)
    ones = torch.ones(1, 1, 1, 1, device=dev)

    if _seed is None or _n_gt is None:
        return zero, (zero.view(1, 1, 1, 1),), (ones,)

    seed = _seed.to(dev)
    n_gt = _n_gt.to(dev)
    d_center = (D - 1) // 2
    xyz_det = res.xyz_lr.detach()

    # --- Sample GT normals at all mesh vertices (all non-differentiable) ---
    with torch.no_grad():
        gt_sampled = res.data.grid_sample_fullres(xyz_det)
        gt_nx = gt_sampled.nx.squeeze(0).squeeze(0)  # (D, Hm, Wm)
        gt_ny = gt_sampled.ny.squeeze(0).squeeze(0)
        gt_nz = torch.sqrt((1.0 - gt_nx * gt_nx - gt_ny * gt_ny).clamp(min=0.0))
        gt_normals = torch.stack([gt_nx, gt_ny, gt_nz], dim=-1)  # (D, Hm, Wm, 3)
        gt_norm = gt_normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        gt_normals = gt_normals / gt_norm

    # --- Intersect all windings (non-differentiable) ---
    with torch.no_grad():
        intersections: list[tuple[int, torch.Tensor, float, float, torch.Tensor]] = []
        for d in range(D):
            p_int, h_frac, w_frac, quad_n, did_hit = _intersect_winding(seed, n_gt, xyz_det[d])
            if did_hit:
                intersections.append((d, p_int, h_frac.item(), w_frac.item(), quad_n))

    if not intersections:
        station_loss._miss_count = getattr(station_loss, "_miss_count", 0) + 1
        if station_loss._miss_count <= 1 or station_loss._miss_count % 100 == 0:
            print(f"[station] WARNING: ray missed all windings (x{station_loss._miss_count})", flush=True)
        return zero, (zero.view(1, 1, 1, 1),), (ones,)

    # --- Normal-offset loss (from central winding, applied jointly) ---
    # Proxy: shift every vertex by -offset along its per-vertex GT normal,
    # oriented consistently with the intersection quad normal.
    loss_normal = zero
    with torch.no_grad():
        center_hit = [x for x in intersections if x[0] == d_center]
        if center_hit:
            _, p_int_c, _, _, quad_n_c = center_hit[0]
            offset = ((p_int_c - seed) * n_gt).sum()  # signed scalar

            # Orient GT normals: use the hit quad's geometric normal compared
            # to n_gt to get a single consistent sign for the whole mesh
            flip = (quad_n_c * n_gt).sum().sign()  # scalar: +1 or -1
            oriented_normals = gt_normals * flip  # (D, Hm, Wm, 3)

            target_n = xyz_det - offset * oriented_normals
        else:
            target_n = None

    if target_n is not None:
        loss_normal = ((res.xyz_lr - target_n) ** 2).mean()

    # --- XY-centering loss (per-winding, independent) ---
    # Each winding's intersection gives a grid position. Shift all vertices
    # so the intersection moves toward the grid center.
    h_mid = (Hm - 1) / 2.0
    w_mid = (Wm - 1) / 2.0
    loss_xy = zero
    n_xy_hits = 0

    for d, _, h_frac_val, w_frac_val, _ in intersections:
        with torch.no_grad():
            dh = h_frac_val - h_mid
            dw = w_frac_val - w_mid
            # Tangent directions from mesh finite differences
            raw_th = (xyz_det[d, 1:, :, :] - xyz_det[d, :-1, :, :]).mean(dim=(0, 1))  # (3,)
            raw_tw = (xyz_det[d, :, 1:, :] - xyz_det[d, :, :-1, :]).mean(dim=(0, 1))  # (3,)
            shift_vec = dh * raw_th + dw * raw_tw  # (3,)
            target_xy_d = xyz_det[d] + shift_vec  # (Hm, Wm, 3)

        loss_xy = loss_xy + ((res.xyz_lr[d] - target_xy_d) ** 2).mean()
        n_xy_hits += 1

    if n_xy_hits > 0:
        loss_xy = loss_xy / n_xy_hits

    loss = loss_normal + loss_xy

    # Dummy loss map/mask for visualization pipeline
    lm = loss.detach().expand(D, 1, Hm, Wm)
    mask = res.mask_lr
    return loss, (lm,), (mask,)
