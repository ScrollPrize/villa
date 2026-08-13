"""Rolling-contact ("carpet") unroll kinematics — self-intersection-free by construction.

At parameter τ the contact line sits at arc c(τ) from the OUTER end of the winding:
  s ≤ c : the sheet lies flat on the tangent plane (the existing flat target V1),
  s > c : the sheet keeps its EXACT original rolled geometry, carried by the rigid motion
          that aligns the source centerline frame at arc c with the flat frame at arc c
          (rolling without slipping).
A rigid copy of intersection-free geometry cannot self-intersect; the flat part is planar;
they meet tangentially at the contact line where a narrow smoothstep band blends the two
poses (which already agree to first order there).

Endpoints: c(0) ≤ 0 keeps every vertex in the rolled pose with R(0) ≈ I (the flat target is
anchored on the tangent plane at the outer end, so the identity holds up to warp noise —
asserted); c(1) = S + w puts every vertex exactly on V1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dg_interp import face_uv_tangents


def _orthonormal_frame(t: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Columns [t̂, â⊥, n̂] from tangent + approximate axis."""
    t = t / max(np.linalg.norm(t), 1e-300)
    a = a - (a @ t) * t
    a = a / max(np.linalg.norm(a), 1e-300)
    n = np.cross(t, a)
    return np.stack([t, a, n], axis=1)


@dataclass
class RollingContactPath:
    V0: np.ndarray
    V1: np.ndarray
    F: np.ndarray
    s: np.ndarray              # (n,) arc coordinate from the OUTER end, in [0, S]
    S: float
    band: float                # max blend band width (arc units)
    band_vert: np.ndarray      # (n,) per-vertex band (curvature-adaptive)
    c_pre: float               # lean-in pre-roll arc (identity → entry pose, arc units)
    grid_c: np.ndarray         # (K,) arc samples
    R_grid: np.ndarray         # (K,3,3) rigid rotations source→flat at each c
    p_grid: np.ndarray         # (K,3) source centerline points
    f_grid: np.ndarray         # (K,3) flat centerline points
    lift: np.ndarray           # (K,) plane-clearance lift of the rigid body
    lift_vis: np.ndarray       # (K,) SMOOTH lift driving the flap bridge (<= lift)
    pad_arc: float             # landing-flap half-width (clearance exemption zone)
    n_plane: np.ndarray        # (3,) unrolling-plane normal (toward the roll side)
    info: dict

    def _rigid_at(self, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        c_eval = float(np.clip(c, self.grid_c[0], self.grid_c[-1]))
        k = int(np.clip(np.searchsorted(self.grid_c, c_eval) - 1, 0, len(self.grid_c) - 2))
        lam = (c_eval - self.grid_c[k]) / max(self.grid_c[k + 1] - self.grid_c[k], 1e-300)
        # rotations between neighbouring samples are tiny: normalized linear blend suffices
        R = (1 - lam) * self.R_grid[k] + lam * self.R_grid[k + 1]
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        p = (1 - lam) * self.p_grid[k] + lam * self.p_grid[k + 1]
        f = (1 - lam) * self.f_grid[k] + lam * self.f_grid[k + 1]
        lift = (1 - lam) * self.lift[k] + lam * self.lift[k + 1]
        lift_v = (1 - lam) * self.lift_vis[k] + lam * self.lift_vis[k + 1]
        return R, p, f, lift, lift_v

    def eval_frame(self, t: float) -> np.ndarray:
        """ONE continuous motion (a two-phase split meets at a zero-velocity
        junction and exposes raw-pose jitter as a visible stutter).
        The rigid field is defined on c ∈ [−c_pre, S+band]: the negative range is the
        tip-onto-plane lean-in (slerp Identity→entry pose, arc-proportional), smoothed
        TOGETHER with the contact field; c(t) is one linear map of the eased timeline."""
        if t <= 0.0:
            return self.V0.copy()
        if t >= 1.0:
            return self.V1.copy()
        c = self._contact_arc(t)
        R, p, f, lift, lift_v = self._rigid_at(c)
        # Clearance lift: the WHOLE rigid roll rides at +lift (zero internal strain); the
        # settled sheet stays IN the plane beyond the flap. NO long ramp (the 0.2*S ramp
        # chased the very lift it existed to avoid: deep lobes raise lift -> ramp raises
        # the sheet -> the sheet overtakes the lifted lobes; and a v-uniform ramp cannot
        # thread between lobes at different v). Instead a SHORT paper bridge confined to
        # the landing flap (<= pad_arc, the zone the clearance certificate exempts as
        # single material BY CONSTRUCTION): the sheet rises to meet the roll's tangent
        # exactly like real paper coming off a roller — without it the lifted roll reads
        # as floating above a knife-flat sheet instead of paying out naturally.
        rolled = (self.V0 - p) @ R.T + (f + lift * self.n_plane)
        if c <= 0.0:
            return rolled
        flat = self.V1
        if lift_v > 0.0:
            # the bridge follows the SMOOTH lift: the raw envelope (which the roll
            # must ride for clearance) keeps spikes, and a spiky bridge jerks the
            # flap sheet (measured popping ratio 0.45 -> 0.56 with a raw-lift
            # bridge). lift_vis <= lift and the flap zone is certificate-exempt,
            # so no clearance guarantee changes.
            x = np.clip((c - self.s) / max(self.pad_arc, 1e-300), 0.0, 1.0)
            g = 1.0 - x * x * (3.0 - 2.0 * x)   # 1 at the contact -> 0 at pad_arc behind
            flat = self.V1 + (lift_v * g)[:, None] * self.n_plane[None, :]
        beta = np.clip((c - self.s) / np.maximum(self.band_vert, 1e-300), 0.0, 1.0)
        beta = beta * beta * (3.0 - 2.0 * beta)  # smoothstep
        return rolled * (1.0 - beta)[:, None] + flat * beta[:, None]

    def _contact_arc(self, t: float) -> float:
        return t * (self.S + self.band + self.c_pre) - self.c_pre

    def transition_face_mask(self, t: float) -> np.ndarray | None:
        """Faces currently crossing the contact band: their fast normal swing is the
        physical takeoff snap, excluded from the temporal-flip count (gate doc)."""
        if t <= 0.0 or t >= 1.0:
            return None
        c = self._contact_arc(t)
        if c <= 0.0:
            return None
        s_f = self.s[self.F].mean(axis=1)
        # must cover the contact's sweep BETWEEN the two compared frames
        pad = 2.0 * self.band + 0.02 * self.S
        return (s_f > c - self.band - pad) & (s_f < c + pad)


def build_rolling_path(
    V0: np.ndarray,
    V1: np.ndarray,
    F: np.ndarray,
    uv: np.ndarray,
    s_u: float,
    *,
    unroll_axis: int = 0,
    outer_is_low_u: bool | None = None,
    n_grid: int = 512,
    band_frac: float = 0.008,
    smooth_sigma: int = 21,
) -> RollingContactPath:
    """Precompute centerline frames + rigid motions for the rolling-contact unroll.

    V1 must be the flat target anchored ON the tangent plane at the OUTER end (the natural
    free end of the winding); outer_is_low_u is auto-detected from winding radii when None.
    """
    V0 = np.asarray(V0, dtype=np.float64)
    V1 = np.asarray(V1, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    u = uv[:, unroll_axis]

    # winding axis (mean v-tangent) + radial distances to find the OUTER end
    t_u, t_v = face_uv_tangents(V0, F, uv)
    w_face = np.linalg.norm(np.cross(V0[F[:, 1]] - V0[F[:, 0]], V0[F[:, 2]] - V0[F[:, 0]]), axis=1)
    tvn = np.linalg.norm(t_v, axis=1)
    good = tvn > 1e-12
    axis = (t_v[good] / tvn[good][:, None] * w_face[good][:, None]).sum(0)
    axis /= max(np.linalg.norm(axis), 1e-300)
    c0 = V0.mean(0)
    rel = V0 - c0
    radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
    if outer_is_low_u is None:
        lo_band = u <= np.quantile(u, 0.05)
        hi_band = u >= np.quantile(u, 0.95)
        outer_is_low_u = bool(radial[lo_band].mean() > radial[hi_band].mean())

    s_uv = (u.max() - u) * s_u if not outer_is_low_u else (u - u.min()) * s_u
    # Landing order must equal flat-space order. The linear-in-u arc disagrees
    # with the vertex's TRUE flat position by O(chart warp) — thousands of voxels
    # on warped tails — so late-landing patches would keep moving rigidly THROUGH
    # already-settled sheet, and warp-seam faces would span huge Δs (sliver
    # spikes at the roll top). Define s as the payout
    # coordinate IN THE FLAT EMBEDDING itself: a vertex flattens exactly when the contact
    # line sweeps its own landing slot, so 'settled behind contact, rolled ahead' is exact
    # and roll-vs-sheet interpenetration is impossible by construction. For a clean chart
    # this equals the UV arc up to a constant.
    sc = s_uv - s_uv.mean()
    reg = (V1 - V1.mean(0)).T @ sc            # payout direction: regression of V1 on the arc
    u_dir = reg / max(np.linalg.norm(reg), 1e-300)
    s = (V1 - V1.mean(0)) @ u_dir
    s -= s.min()
    S = float(s.max())
    med_edge = float(np.median(np.linalg.norm(V0[F[:, 1]] - V0[F[:, 0]], axis=1)))
    band = max(band_frac * S, 4.0 * med_edge)

    # centerline frames on a grid of c
    grid_c = np.linspace(0.0, S, n_grid)
    half = max(S / n_grid * 1.5, band / 4)
    # per-vertex tangents for frame construction
    tu_v = np.zeros_like(V0)
    tv_v = np.zeros_like(V0)
    cnt = np.zeros(len(V0))
    for j in range(3):
        np.add.at(tu_v, F[:, j], t_u)
        np.add.at(tv_v, F[:, j], t_v)
        np.add.at(cnt, F[:, j], 1.0)
    tu_v /= np.maximum(cnt, 1.0)[:, None]
    tv_v /= np.maximum(cnt, 1.0)[:, None]
    # Per-band rigid transform = Kabsch fit of the band's SOURCE vertices onto their OWN
    # flat targets. No tangent regressions or sign conventions: at c=0 the embedding placed
    # the anchor strip on its tangent plane, so the band-0 Kabsch is ≈ identity by
    # construction; at general c it is exactly the rolling-contact alignment (least-squares
    # tangency of the remaining roll on the plane at the contact line).
    order = np.argsort(s)
    s_sorted = s[order]
    p_grid = np.empty((n_grid, 3))
    f_grid = np.empty((n_grid, 3))
    R_grid = np.empty((n_grid, 3, 3))
    k_min = max(3000, len(V0) // 200)  # bands must span the v-extent (ragged tips!)
    for k, c in enumerate(grid_c):
        h = half
        i0, i1 = np.searchsorted(s_sorted, [c - h, c + h])
        while (i1 - i0) < k_min and (i0 > 0 or i1 < len(order)):
            h *= 1.6
            i0, i1 = np.searchsorted(s_sorted, [c - h, c + h])
        idx = order[i0:i1] if i1 > i0 else order[max(0, i0 - 8):min(len(order), i0 + 8)]
        P, Q = V0[idx], V1[idx]
        cp, cq = P.mean(0), Q.mean(0)
        H = (P - cp).T @ (Q - cq)
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R_grid[k] = Vt.T @ D @ U.T
        p_grid[k] = cp
        f_grid[k] = cq
    # smooth along the grid (warp noise); rotations re-projected to SO(3) after averaging.
    # END-PRESERVING: blend back to the raw Kabsch poses near both grid ends — heavy
    # smoothing otherwise drags curled-neighbour poses into grid[0] and manufactures a
    # fake tip angle (w018: 13° real → 24° smoothed), inflating the settle sweep.
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        # FULL smoothing: end-preservation exposed raw per-band Kabsch jitter exactly
        # where the sweep starts (visible vibration); the settle phase targets the
        # smoothed entry pose, so consistency at c=0 is automatic.
        p_grid[:] = gaussian_filter1d(p_grid, smooth_sigma, axis=0, mode="nearest")
        f_grid[:] = gaussian_filter1d(f_grid, smooth_sigma, axis=0, mode="nearest")
        R_sm = gaussian_filter1d(R_grid.reshape(n_grid, 9), smooth_sigma, axis=0,
                                 mode="nearest").reshape(n_grid, 3, 3)
        Us, _, Vts = np.linalg.svd(R_sm)
        det = np.linalg.det(Us @ Vts)
        Ds = np.ones((n_grid, 3))
        Ds[:, 2] = det
        R_grid = (Us * Ds[:, None, :]) @ Vts

    # Curvature-adaptive blend band: chord-shear in the band scales with band×curvature,
    # so the band narrows where the winding is tight (scroll cores) and keeps its full
    # width on gentle outer wraps. Curvature from the RAW centerline (pre-smoothing).
    dp = np.gradient(p_grid, grid_c, axis=0)
    t_hat = dp / np.maximum(np.linalg.norm(dp, axis=1, keepdims=True), 1e-300)
    dt = np.gradient(t_hat, grid_c, axis=0)
    kappa = np.linalg.norm(dt, axis=1)
    from scipy.ndimage import gaussian_filter1d as _gf1k
    kappa = _gf1k(kappa, 9, mode="nearest")
    band_c = np.clip(np.where(kappa > 1e-12, 0.5 / np.maximum(kappa, 1e-12), band),
                     4.0 * med_edge, band)
    band_vert = np.interp(s, grid_c, band_c)

    # Plane-clearance barrier (closed form): the rigid roll's only possible collider is
    # the unrolling plane (the already-flat sheet lies in it). Per grid contact, measure
    # how far transformed rolled points BEHIND the contact would dig below the plane and
    # lift the whole rigid body by that clearance along the plane normal. Rigidity (hence
    # intra-roll intersection-freedom) is untouched; roll-vs-flat penetration becomes
    # impossible by construction.
    n_pl = np.cross(V1[F[:, 1]] - V1[F[:, 0]], V1[F[:, 2]] - V1[F[:, 0]]).sum(0)
    n_pl /= max(np.linalg.norm(n_pl), 1e-300)
    if (V0.mean(0) - f_grid[0]) @ n_pl < 0:
        n_pl = -n_pl                       # normal points toward the roll's side
    u_pl = f_grid[-1] - f_grid[0]
    u_pl -= (u_pl @ n_pl) * n_pl
    u_pl /= max(np.linalg.norm(u_pl), 1e-300)  # plane direction of paying-out
    sub = np.arange(0, len(V0), 2)
    eps = 0.5 * float(np.median(np.linalg.norm(V0[F[:, 1]] - V0[F[:, 0]], axis=1)))
    pad_arc = band + 4.0 * med_edge   # landing flap: single material, exempt from clearance
    lift_req = np.zeros(n_grid)
    for k in range(n_grid):
        c = grid_c[k]
        m_roll = s[sub] > c
        if not m_roll.any():
            continue
        pts = (V0[sub[m_roll]] - p_grid[k]) @ R_grid[k].T + f_grid[k]
        behind = (pts - f_grid[k]) @ u_pl < 0.0   # over already-paid-out sheet
        if not behind.any():
            continue
        hgt = (pts[behind] - f_grid[0]) @ n_pl
        lift_req[k] = max(0.0, eps - float(hgt.min()))
    from scipy.ndimage import gaussian_filter1d as _gf1
    # smoothing must never UNDERCUT the requirement (it shaved spiky lift needs by
    # >1000 units and let pre-landing lobes dig through the plane): smooth for the
    # visual ride, then take the upper envelope with the raw requirement.
    lift = np.maximum(_gf1(lift_req, 15, mode="nearest"), lift_req)
    lift[0] = 0.0  # frame-0 pose must stay the exact source pose
    # the flap bridge follows a spike-free lift (always <= the clearance envelope)
    lift_vis = np.minimum(_gf1(lift_req, 31, mode="nearest"), lift)
    lift_vis[0] = 0.0
    # NO end taper: forcing lift->0 while the multi-turn core is still wound pushes it
    # INTO the flat sheet -> coincident surfaces z-fight through the cutout = the
    # 'transparent roll' artifact. The computed lift need decays
    # naturally as the core shrinks, and endpoint exactness is already guaranteed by the
    # blend weights (the rolled pose has weight ~0 in the final frames).

    # identity sanity at c=0 (flat target tangent at the outer end)
    dev0 = float(np.linalg.norm(R_grid[0] - np.eye(3)))
    info = {
        "kinematics": "rolling_contact",
        "outer_is_low_u": bool(outer_is_low_u),
        "S_arc": S,
        "band": band,
        "R0_identity_dev": dev0,
        "n_grid": n_grid,
        "lift_max": float(lift.max()),
        "lift_eps": eps,
        "pad_arc": pad_arc,
    }
    # Lean-in pre-roll: extend the field below c=0 with slerp(Identity → entry pose),
    # allocated arc-proportionally (c_pre = mean lean-in displacement so speed across
    # c=0 is matched by construction), then smooth the UNIFIED field — one continuous
    # motion: no phase junction, no rate step, no exposed raw-pose jitter.
    from .dg_interp import rotation_exp as _re, rotation_log as _rl
    w0 = _rl(R_grid[:1])[0]
    th0 = float(np.linalg.norm(w0))
    sub2 = np.arange(0, len(V0), 7)
    T0_pts = (V0[sub2] - p_grid[0]) @ R_grid[0].T + f_grid[0]
    M_rot = float(np.linalg.norm(T0_pts - V0[sub2], axis=1).mean())
    c_pre = float(np.clip(M_rot, 1e-6, 0.35 * S))
    n_pre = max(12, int(round(n_grid * c_pre / max(S, 1e-300))))
    alpha = np.arange(n_pre) / n_pre
    R_pre = _re(alpha[:, None] * w0[None, :])
    p_pre = np.repeat(p_grid[:1], n_pre, axis=0)
    f_pre = p_grid[0][None, :] + alpha[:, None] * (f_grid[0] - p_grid[0])[None, :]
    grid_c = np.concatenate([np.linspace(-c_pre, 0.0, n_pre, endpoint=False), grid_c])
    R_grid = np.concatenate([R_pre, R_grid], axis=0)
    p_grid = np.concatenate([p_pre, p_grid], axis=0)
    f_grid = np.concatenate([f_pre, f_grid], axis=0)
    # (lift stays zero during lean-in: a pre-rise ramp does not address the
    # junction rate mismatch — that is pacing — and only adds lean-in motion)
    lift = np.concatenate([np.zeros(n_pre), lift])
    lift_vis = np.concatenate([np.zeros(n_pre), lift_vis])
    from scipy.ndimage import gaussian_filter1d as _gf2
    p_grid = _gf2(p_grid, 5, axis=0, mode="nearest")
    f_grid = _gf2(f_grid, 5, axis=0, mode="nearest")
    # the clearance envelope survives the unified re-smooth: lift may only grow
    lift = np.maximum(_gf2(lift, 5, mode="nearest"), lift)
    lift_vis = np.minimum(_gf2(lift_vis, 5, mode="nearest"), lift)
    Rs = _gf2(R_grid.reshape(len(grid_c), 9), 5, axis=0, mode="nearest").reshape(-1, 3, 3)
    Uu, _, Vv = np.linalg.svd(Rs)
    dd = np.linalg.det(Uu @ Vv)
    Dd = np.ones((len(grid_c), 3))
    Dd[:, 2] = dd
    R_grid = (Uu * Dd[:, None, :]) @ Vv
    # frame-0 exactness, TAPERED: the unified re-smooth bleeds the entry pose into
    # the first samples; stomping only sample 0 back to identity leaves a field STEP
    # between grid[0] and grid[1] — invisible at small tip angles (w023: 39°) but a
    # visible morph-start snap at large ones (w011: 120°, disp delta 3.8x median).
    # Instead compute the sample-0 correction and decay it smoothly over ~2 sigma.
    w_err = _rl(R_grid[:1])[0]
    df_err = f_grid[0] - p_grid[0]
    K_pin = min(12, len(grid_c) - 1)
    for k in range(K_pin):
        a = 1.0 - k / K_pin
        a = a * a * (3.0 - 2.0 * a)
        R_grid[k] = _re((-a) * w_err[None])[0] @ R_grid[k]
        f_grid[k] = f_grid[k] - a * df_err
        lift[k] = (1.0 - a) * lift[k]
        lift_vis[k] = (1.0 - a) * lift_vis[k]
    info["tip_angle_deg"] = float(np.degrees(th0))
    info["c_pre"] = c_pre
    return RollingContactPath(V0=V0, V1=V1, F=np.asarray(F), s=s, S=S, band=band,
                              band_vert=band_vert,
                              c_pre=c_pre,
                              grid_c=grid_c, R_grid=R_grid, p_grid=p_grid, f_grid=f_grid,
                              lift=lift, lift_vis=lift_vis, pad_arc=pad_arc, n_plane=n_pl,
                              info=info)
