"""Deformation-gradient interpolation (Alexa-2000 / Sumner) for the scroll unwrap.

Pipeline per mesh:
  1. F_i = T_tgt T_src⁻¹ per face (T = [e1 e2 n̂] frame, Sumner's construction).
  2. Polar F_i = R_i S_i via batch SVD with the det<0 fix (S then has a negative eigenvalue
     for inverted faces — counted and reported, not hidden).
  3. w_i = log(R_i) made branch-consistent over the face dual graph (BFS from anchor seed,
     candidates (θ+2πk)·â, parent-axis substitution at θ≈0) — multi-turn unrolls need
     angles continuous up to several π.
  4. Frame t: F_i(t) = exp(t·w_i)((1−t)I + t·S_i); reconstruct vertices from the Poisson
     system Gᵀ A G x = Gᵀ A f(t) with the anchor vertex Dirichlet-pinned; factor once.
  5. Per-frame residual rigid drift removed by Kabsch on the anchor strip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph  # noqa: F401  (enables sp.csgraph)

from .operators import face_areas, grad_operator
from .solver import FactorizedSPD


# ---------- per-face frames and gradients ----------

def face_frames(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """(m,3,3) matrices with columns [e1, e2, n̂] (unit normal as third column)."""
    V = np.asarray(V, dtype=np.float64)
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    n = np.cross(e1, e2)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm = np.where(norm < 1e-300, 1e-300, norm)
    n_hat = n / norm
    return np.stack([e1, e2, n_hat], axis=2)


def deformation_gradients(V_src: np.ndarray, V_tgt: np.ndarray, F: np.ndarray) -> np.ndarray:
    """(m,3,3) per-face deformation gradients source→target."""
    T_src = face_frames(V_src, F)
    T_tgt = face_frames(V_tgt, F)
    return T_tgt @ np.linalg.inv(T_src)


def polar_decompose(Fm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch polar decomposition F = R S, R ∈ SO(3).

    Returns (R, S, inverted_mask). For det(F)<0 faces S gets a negative eigenvalue
    (proper rotation preserved); those faces are flagged.
    """
    U, sig, Vt = np.linalg.svd(Fm)
    det = np.linalg.det(U @ Vt)
    D = np.ones((len(Fm), 3))
    D[:, 2] = det  # ±1
    R = (U * D[:, None, :]) @ Vt
    Vm = Vt.transpose(0, 2, 1)
    S = (Vm * (D * sig)[:, None, :]) @ Vt
    return R, S, det < 0


# ---------- rotation logs / exp ----------

def rotation_log(R: np.ndarray) -> np.ndarray:
    """Batch principal log of rotation matrices → axis-angle vectors (m,3), θ ∈ [0, π].

    Branching is on θ (not sin θ): the generic skew/(2 sin θ) formula amplifies trace
    rounding noise without bound as θ→π, so the last milliradian uses the symmetric-part
    axis extraction with the global sign recovered from the (tiny but sign-reliable)
    antisymmetric part.
    """
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(tr)
    skew = np.stack(
        [R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=1
    )
    out = np.zeros_like(skew)

    near0 = theta < 1e-6
    nearpi = theta > np.pi - 1e-3
    gen = ~(near0 | nearpi)

    out[near0] = 0.5 * skew[near0]
    if gen.any():
        out[gen] = skew[gen] * (theta[gen] / (2.0 * np.sin(theta[gen])))[:, None]
    if nearpi.any():
        Rp = R[nearpi]
        B = (Rp + np.eye(3)) * 0.5  # ≈ â âᵀ + O(π−θ)
        diag = np.maximum(np.einsum("mii->mi", B), 0.0)
        axis = np.sqrt(diag)
        k = np.argmax(axis, axis=1)
        m_idx = np.arange(len(Rp))
        sign = np.ones_like(axis)
        for j in range(3):
            off = B[m_idx, k, j]
            sign[:, j] = np.where(off < 0, -1.0, 1.0)
        sign[m_idx, k] = 1.0
        axis = axis * sign
        axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-300)
        # global sign: skew = 2 sin θ · â, sign reliable whenever sin θ isn't exactly 0
        flip = np.einsum("mi,mi->m", axis, skew[nearpi]) < 0
        axis[flip] = -axis[flip]
        out[nearpi] = axis * theta[nearpi][:, None]
    return out


def rotation_exp(w: np.ndarray) -> np.ndarray:
    """Batch Rodrigues: axis-angle vectors (m,3) → rotation matrices (m,3,3)."""
    theta = np.linalg.norm(w, axis=1)
    m = len(w)
    R = np.tile(np.eye(3), (m, 1, 1))
    nz = theta > 1e-12
    if not nz.any():
        return R
    k = w[nz] / theta[nz][:, None]
    K = np.zeros((nz.sum(), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    st = np.sin(theta[nz])[:, None, None]
    ct = (1.0 - np.cos(theta[nz]))[:, None, None]
    R[nz] = np.eye(3) + st * K + ct * (K @ K)
    return R


# ---------- dual graph + branch-consistent unwrapping ----------

def face_adjacency(F: np.ndarray) -> sp.csr_matrix:
    """Face-face adjacency over shared edges (m×m, symmetric, boolean)."""
    m = len(F)
    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    edges.sort(axis=1)
    face_id = np.tile(np.arange(m), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    es, fs = edges[order], face_id[order]
    same = (es[1:] == es[:-1]).all(axis=1)
    a, b = fs[:-1][same], fs[1:][same]
    data = np.ones(len(a), dtype=bool)
    A = sp.csr_matrix((data, (a, b)), shape=(m, m))
    return A + A.T


def branch_consistent_logs(
    R: np.ndarray, F: np.ndarray, seed_face: int, k_range: int = 4
) -> tuple[np.ndarray, dict]:
    """Unwrap per-face rotation logs continuously over the mesh (BFS by frontier, vectorized).

    Candidates per face: (θ + 2πk)·â for k ∈ [−k_range, k_range]; faces with θ≈0 borrow the
    parent's axis. Choice minimizes distance to the parent's unwrapped log.
    """
    w_raw = rotation_log(R)
    theta = np.linalg.norm(w_raw, axis=1)
    safe = theta > 1e-9
    axis = np.zeros_like(w_raw)
    axis[safe] = w_raw[safe] / theta[safe][:, None]

    A = face_adjacency(F)
    m = len(F)
    w = w_raw.copy()
    visited = np.zeros(m, dtype=bool)
    visited[seed_face] = True
    frontier = np.array([seed_face])
    ks = np.arange(-k_range, k_range + 1)
    max_abs_angle = theta[seed_face]
    n_adjusted = 0

    indptr, indices = A.indptr, A.indices
    while frontier.size:
        # neighbors of the frontier
        nbr_lists = [indices[indptr[f]:indptr[f + 1]] for f in frontier]
        parents = np.repeat(frontier, [len(x) for x in nbr_lists])
        nbrs = np.concatenate(nbr_lists) if nbr_lists else np.array([], dtype=int)
        new_mask = ~visited[nbrs]
        nbrs, parents = nbrs[new_mask], parents[new_mask]
        if nbrs.size == 0:
            break
        # first occurrence wins (a face can be reached from two frontier parents)
        uniq, first = np.unique(nbrs, return_index=True)
        nbrs, parents = uniq, parents[first]

        ax = axis[nbrs].copy()
        th = theta[nbrs].copy()
        # θ≈0 → borrow parent's axis direction (preimage of identity is 2πk·any-axis)
        degenerate = th <= 1e-9
        if degenerate.any():
            wp = w[parents[degenerate]]
            wp_n = np.linalg.norm(wp, axis=1, keepdims=True)
            ax[degenerate] = np.where(wp_n > 1e-12, wp / np.maximum(wp_n, 1e-300), 0.0)
        cand = ax[:, None, :] * (th[:, None] + 2.0 * np.pi * ks[None, :])[:, :, None]  # (b,K,3)
        d = np.linalg.norm(cand - w[parents][:, None, :], axis=2)
        best = np.argmin(d, axis=1)
        w[nbrs] = cand[np.arange(len(nbrs)), best]
        n_adjusted += int((best != k_range).sum())
        visited[nbrs] = True
        max_abs_angle = max(max_abs_angle, float(np.linalg.norm(w[nbrs], axis=1).max()))
        frontier = nbrs

    info = {
        "n_faces": m,
        "n_unreached": int((~visited).sum()),
        "n_branch_adjusted": n_adjusted,
        "max_unwrapped_angle_rad": float(max_abs_angle),
        "max_unwrapped_turns": float(max_abs_angle / (2 * np.pi)),
    }
    return w, info


# ---------- winding decomposition (production rotation field) ----------

def face_uv_tangents(V: np.ndarray, F: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-face world tangents (∂p/∂u, ∂p/∂v) from the UV Jacobian, each (m,3)."""
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    d1 = uv[F[:, 1]] - uv[F[:, 0]]
    d2 = uv[F[:, 2]] - uv[F[:, 0]]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    bad = np.abs(det) < 1e-18
    safe = np.where(bad, 1.0, det)
    t_u = (d2[:, 1, None] * e1 - d1[:, 1, None] * e2) / safe[:, None]
    t_v = (-d2[:, 0, None] * e1 + d1[:, 0, None] * e2) / safe[:, None]
    t_u[bad] = 0.0
    t_v[bad] = 0.0
    return t_u, t_v


def _scalar_graph_unwrap(
    phi: np.ndarray, F: np.ndarray, seed_face: int, conf: np.ndarray | None = None
) -> tuple[np.ndarray, int]:
    """Unwrap a per-face angle field (defined mod 2π) to be continuous across the dual graph.

    With a confidence mask, the spanning tree is a Dijkstra shortest-path tree whose edges
    prefer confident-confident hops (cost 1) and cross junk only when no clean route exists
    (cost 1000): a single garbage face then corrupts nothing downstream — paths route
    around damage instead of through it.

    Branch reference across junk: a junk face's raw φ is untrusted (its t_u may be
    garbage), so letting it steer the 2π-rounding lets a junk moat shift everything
    beyond it onto a wrong 2π branch — whole junk regions (e.g. an invisible anchor
    strip) then secretly orbit the axis ±k full turns mid-transit (the t=1 state is
    unaffected: a 2πk rotation about the axis is the identity matrix — the branch is
    purely a transit choice, and the coherent choice is the visible sheet's). Junk faces
    therefore INHERIT the branch from their confident ancestry (keeping their own value
    mod 2π); confident faces accumulate their own wrapped deltas as before.
    """
    A = face_adjacency(F)
    out = phi.astype(np.float64).copy()
    two_pi = 2.0 * np.pi

    if conf is None:
        order, pred, n_unreached = _bfs_tree(A, seed_face)
    else:
        Aw = A.tocoo()
        cheap = conf[Aw.row] & conf[Aw.col]
        w = np.where(cheap, 1.0, 1000.0)
        G = sp.csr_matrix((w, (Aw.row, Aw.col)), shape=A.shape)
        dist, pred = sp.csgraph.dijkstra(G, directed=False, indices=seed_face,
                                         return_predecessors=True)
        reach = np.isfinite(dist)
        n_unreached = int((~reach).sum())
        order = np.argsort(dist[reach], kind="stable")
        order = np.flatnonzero(reach)[order]
    # apply 2π-rounding in tree order (parents always precede children)
    pred = np.asarray(pred)
    if conf is None:
        for f in order:
            p = pred[f]
            if p < 0:
                continue
            out[f] += two_pi * np.round((out[p] - out[f]) / two_pi)
    else:
        ref = out.copy()  # branch carrier: junk inherits, confidence accumulates
        for f in order:
            p = pred[f]
            if p < 0:
                continue
            if conf[f]:
                d = out[f] - ref[p]
                d -= two_pi * np.round(d / two_pi)
                ref[f] = ref[p] + d
                out[f] = ref[f]
            else:
                ref[f] = ref[p]
                out[f] += two_pi * np.round((ref[p] - out[f]) / two_pi)
    return out, n_unreached


def _bfs_tree(A: sp.csr_matrix, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    order, pred = sp.csgraph.breadth_first_order(A, seed, directed=False,
                                                 return_predecessors=True)
    n_unreached = A.shape[0] - len(order)
    return np.asarray(order), np.asarray(pred), n_unreached


def _axis_rotations(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Batch rotations about ONE fixed axis by per-face angles (m,) → (m,3,3)."""
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    K2 = K @ K
    s = np.sin(angles)[:, None, None]
    c = (1.0 - np.cos(angles))[:, None, None]
    return np.eye(3) + s * K + c * K2


def winding_decomposition(
    R: np.ndarray,
    V0: np.ndarray,
    F: np.ndarray,
    uv: np.ndarray,
    V1: np.ndarray,
    seed_face: int,
    axis_mode: str = "global",
    face_conf: np.ndarray | None = None,
    axis_from_conf: bool = False,
) -> dict:
    """Decompose per-face rotations as R_i = exp(w_res,i) · R_axis(â, φ_i).

    â_i = the LOCAL winding axis per face (source v-tangent, sign-aligned to the global
    axis; warped wraps bend their axis across the sheet — a global axis misorients whole
    regions mid-transit with error amplified by φ);
    φ_i = angle from the (per-face axis-orthogonalized) flat u-direction to the source
    u-tangent about â_i, unwrapped as a SCALAR over the dual graph (no matrix-log branches);
    w_res,i = principal log of R_i·R_wind,iᵀ — bump-scale, branch-free.
    Exact at t=1 for any number of turns.
    """
    # frame from the flat target: û_f, v̂_f (regression of V1 on uv axes)
    uvc = uv - uv.mean(0)
    V1c = V1 - V1.mean(0)
    u_dir = V1c.T @ uvc[:, 0]
    v_dir = V1c.T @ uvc[:, 1]
    v_hat = v_dir / np.linalg.norm(v_dir)
    u_perp = u_dir - (u_dir @ v_hat) * v_hat
    b1g = u_perp / np.linalg.norm(u_perp)          # flat u-direction ⊥ global axis

    t_u, t_v = face_uv_tangents(V0, F, uv)
    if axis_mode in ("local", "smooth"):
        # local winding axes: normalized source v-tangents, sign-aligned to the global axis
        nv = np.linalg.norm(t_v, axis=1)
        ax = np.where((nv > 1e-12)[:, None], t_v / np.maximum(nv, 1e-300)[:, None], v_hat[None, :])
        flip = (ax @ v_hat) < 0
        ax[flip] = -ax[flip]
        if axis_mode == "smooth":
            # diffuse the axis field over the dual graph: keeps real sub-sheet axis bends
            # (merged segments!) while killing per-face bump noise. With axis_from_conf,
            # junk faces are not axis SOURCES (their t_v is untrusted): confident axes are
            # re-imposed each sweep, harmonically extending the trusted field into junk.
            Adj = face_adjacency(F).astype(np.float64)
            deg = np.maximum(np.asarray(Adj.sum(axis=1)).ravel(), 1.0)
            fixed = None
            if axis_from_conf and face_conf is not None and face_conf.any():
                fixed = face_conf
                ax_fixed = ax[fixed].copy()
            for _ in range(60):
                ax = 0.5 * ax + 0.5 * (Adj @ ax) / deg[:, None]
                if fixed is not None:
                    ax[fixed] = ax_fixed
                ax /= np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-300)
    else:  # 'global': one axis everywhere — best when the wrap's axis barely bends
        ax = np.tile(v_hat, (len(F), 1))

    # per-face reference frame ⊥ local axis
    b1 = b1g[None, :] - (ax @ b1g)[:, None] * ax
    b1n = np.linalg.norm(b1, axis=1)
    deg = b1n < 1e-9
    if deg.any():
        alt = np.cross(ax[deg], v_hat)
        b1[deg] = alt
        b1n = np.linalg.norm(b1, axis=1)
    b1 /= np.maximum(b1n, 1e-300)[:, None]
    b2 = np.cross(ax, b1)

    t_perp = t_u - np.einsum("ij,ij->i", t_u, ax)[:, None] * ax
    nrm = np.linalg.norm(t_perp, axis=1)
    ok = nrm > 1e-12
    x = np.where(ok, np.einsum("ij,ij->i", t_perp, b1), 1.0)
    y = np.where(ok, np.einsum("ij,ij->i", t_perp, b2), 0.0)
    phi_raw = np.arctan2(y, x)                     # angle FROM flat-u TO source-u, about â_i
    conf = ok if face_conf is None else (ok & face_conf)
    if not conf[seed_face]:
        # seed must be confident; move to the nearest confident face
        if conf.any():
            cent = V0[F].mean(1)
            cand = np.flatnonzero(conf)
            seed_face = int(cand[np.argmin(((cent[cand] - cent[seed_face]) ** 2).sum(1))])
        else:
            conf = None
    phi, n_unreached = _scalar_graph_unwrap(phi_raw, F, seed_face, conf=conf)

    # Per-segment re-branching (merged traces): sub-sheets glued in UV at different
    # winding depths leave |Δφ| > π seams in the unwrapped field — mid-transit the deeper
    # segment orbits k extra turns and the seam shears violently. A 2πk offset about the
    # axis is the identity matrix, so pulling every segment onto its neighbour's branch
    # is exact at t=1 and only changes the transit, making glued segments travel
    # together. Seam edges (|Δφ| > π) partition faces into segments; BFS from the seed's
    # segment assigns each newly-met segment the 2πk that cancels the median seam jump.
    two_pi = 2.0 * np.pi
    A_f = face_adjacency(F)
    Au = sp.triu(A_f.tocoo())
    ei, ej = Au.row, Au.col
    dphi = phi[ei] - phi[ej]
    seam = np.abs(dphi) > np.pi
    n_segments = 1
    if seam.any():
        keep = ~seam
        Ain = sp.csr_matrix((np.ones(int(keep.sum()), dtype=bool), (ei[keep], ej[keep])),
                            shape=A_f.shape)
        n_segments, seg = sp.csgraph.connected_components(Ain + Ain.T, directed=False)
        if n_segments > 1:
            si, sj = seg[ei[seam]], seg[ej[seam]]
            dseam = dphi[seam]  # phi[si side] − phi[sj side]
            offset = np.zeros(n_segments)
            done = np.zeros(n_segments, dtype=bool)
            frontier = [int(seg[seed_face])]
            done[frontier[0]] = True
            while frontier:
                nxt = []
                for s in frontier:
                    for a, b, dd in ((si, sj, dseam), (sj, si, -dseam)):
                        m = (a == s) & ~done[b]
                        for nb in np.unique(b[m]):
                            # want phi_nb + off_nb ≈ phi_s + off_s across the seam:
                            # off_nb = off_s + 2πk, k from the median seam jump
                            k = np.round(np.median(dd[m & (b == nb)]) / two_pi)
                            offset[nb] = offset[s] + two_pi * k
                            done[nb] = True
                            nxt.append(int(nb))
                frontier = nxt
            phi = phi + offset[seg]

    # R maps source → target tangents: winding part must REMOVE φ
    R_wind = rotation_exp(ax * (-phi)[:, None])
    R_res = R @ R_wind.transpose(0, 2, 1)
    w_res = rotation_log(R_res)
    res_norm = np.linalg.norm(w_res, axis=1)
    n_big_res = int((res_norm > np.pi * 0.9).sum())
    return {
        "phi": -phi,                                # signed winding rotation source→target
        "axis": v_hat,
        "axis_per_face": ax,
        "w_res": w_res,
        "n_phi_segments": int(n_segments),
        "residual_norm_p99": float(np.quantile(res_norm, 0.99)),
        "residual_norm_max": float(res_norm.max()),
        "n_big_residual": n_big_res,
        "n_unreached": n_unreached,
        "phi_span_rad": float(phi.max() - phi.min()),
        "phi_span_turns": float((phi.max() - phi.min()) / (2 * np.pi)),
    }


def winding_rotations(decomp: dict, t: float) -> np.ndarray:
    """R_i(t) = exp(t·w_res,i) · exp(â_i · t·φ_i)."""
    R_res_t = rotation_exp(t * decomp["w_res"])
    R_wind_t = rotation_exp(decomp["axis_per_face"] * (t * decomp["phi"])[:, None])
    return R_res_t @ R_wind_t


# ---------- the interpolator ----------

@dataclass
class UnwrapPath:
    """Precomputed interpolation state for one mesh (factor once, evaluate many)."""

    V0: np.ndarray            # (n,3) float64 source
    V1: np.ndarray            # (n,3) float64 target (embedded flat sheet)
    F: np.ndarray             # (m,3) int32
    uv: np.ndarray            # (n,2) normalized UV (winding decomposition needs it)
    decomp: dict              # winding decomposition (phi, axis, w_res, diagnostics)
    S: np.ndarray             # (m,3,3) symmetric stretch
    G: sp.csr_matrix          # (3m,n) gradient operator
    A: np.ndarray             # (m,) face areas (source)
    J0: np.ndarray            # (m,3,3) source per-face world gradients of coords
    solver: FactorizedSPD
    L: sp.csr_matrix
    keep: np.ndarray          # index map after pinning
    pins: np.ndarray          # pinned vertices: anchor + one per extra connected component
    anchor_idx: np.ndarray    # vertex indices of the anchor strip
    info: dict
    face_conf: np.ndarray | None = None
    align_idx: np.ndarray | None = None  # strip subset used for rigid drift removal
    align_rotation: bool = True          # False: translation-only (orbiting strip)

    def _poisson_solve(self, target_grads: np.ndarray, t: float) -> np.ndarray:
        """Solve Gᵀ A G x = Gᵀ A f for per-face target gradient matrices (m,3,3)."""
        m = len(self.F)
        targ = target_grads.transpose(1, 0, 2).reshape(3 * m, 3)  # rows [dx; dy; dz]
        W = np.tile(self.A, 3)[:, None]
        rhs = self.G.T @ (W * targ)
        x_pins = (1.0 - t) * self.V0[self.pins] + t * self.V1[self.pins]
        rhs_red = rhs[self.keep] - self.L[self.keep][:, self.pins] @ x_pins
        X = self.solver.solve(rhs_red)
        out = np.empty_like(self.V0)
        out[self.keep] = X
        out[self.pins] = x_pins
        idx = self.anchor_idx if self.align_idx is None else self.align_idx
        strip_target = (1.0 - t) * self.V0[idx] + t * self.V1[idx]
        if not self.align_rotation:
            return out + (strip_target.mean(0) - out[idx].mean(0))
        return _rigid_align(out, idx, strip_target)

    def eval_frame(self, t: float) -> np.ndarray:
        """Vertex positions at interpolation parameter t ∈ [0,1]."""
        if t <= 0.0:
            return self.V0.copy()
        if t >= 1.0:
            return self.V1.copy()
        m = len(self.F)
        Rt = winding_rotations(self.decomp, t)
        St = (1.0 - t) * np.tile(np.eye(3), (m, 1, 1)) + t * self.S
        Ft = Rt @ St
        # target per-face world gradients: M[d,c] = ∂(φ_c)/∂x_d = (J0 Fᵀ)[d,c]
        M = self.J0 @ Ft.transpose(0, 2, 1)
        return self._poisson_solve(M, t)


def _rigid_align(X: np.ndarray, idx: np.ndarray, target_pts: np.ndarray) -> np.ndarray:
    """Rigidly move X so X[idx] best matches target_pts (Kabsch, det=+1)."""
    P = X[idx]
    cp, ct = P.mean(0), target_pts.mean(0)
    H = (P - cp).T @ (target_pts - ct)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return (X - cp) @ R.T + ct


class SubsteppedPath:
    """Incremental (geodesic-like) integration of the unwrap.

    The single-shot field F_i(t) relative to the SOURCE becomes non-integrable mid-transit
    on warped wraps (spatially varying rotation axes) — the Poisson projection then shears.
    Fix (design-review prescription): split [0,1] into K intervals; at each key re-derive
    the deformation gradients from the LAST SOLVED STATE to the final target (re-polar +
    re-branch), so each interval only integrates a small, near-integrable rotation. The
    Poisson operator lives on the source mesh and its factorization is reused throughout.

    Frames must be evaluated in nondecreasing t order (interval fields are built lazily
    and discarded).
    """

    def __init__(self, base: UnwrapPath, n_substeps: int) -> None:
        self.base = base
        self.K = int(n_substeps)
        self.keys_t = np.linspace(0.0, 1.0, self.K + 1)
        self._key_V: list[np.ndarray | None] = [base.V0] + [None] * self.K
        self._interval: int = -1
        self._field: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None  # (w, S, Jk)
        self.info = dict(base.info, n_substeps=self.K)

    def _build_field(self, k: int) -> None:
        """Field for interval k: deformation gradients from key-k state to the FINAL target."""
        b = self.base
        Vk = self._key_V[k]
        assert Vk is not None
        Fg = deformation_gradients(Vk, b.V1, b.F)
        R, S, _ = polar_decompose(Fg)
        decomp = winding_decomposition(R, Vk, b.F, b.uv, b.V1, seed_face=b.info["seed_face"],
                                       face_conf=b.face_conf)
        GVk = b.G @ Vk
        m = len(b.F)
        Jk = GVk.reshape(3, m, 3).transpose(1, 0, 2)
        self._field = (decomp, S, Jk)
        self._interval = k

    def _solve(self, k: int, s: float, t_global: float) -> np.ndarray:
        b = self.base
        if self._interval != k:
            self._build_field(k)
        decomp, S, Jk = self._field
        m = len(b.F)
        Rt = winding_rotations(decomp, s)
        St = (1.0 - s) * np.tile(np.eye(3), (m, 1, 1)) + s * S
        Ft = Rt @ St
        M = Jk @ Ft.transpose(0, 2, 1)
        return b._poisson_solve(M, t_global)

    def _ensure_key(self, k: int) -> None:
        if self._key_V[k] is not None:
            return
        self._ensure_key(k - 1)
        t_k = self.keys_t[k]
        t_prev = self.keys_t[k - 1]
        s = (t_k - t_prev) / max(1.0 - t_prev, 1e-15)
        self._key_V[k] = self._solve(k - 1, s, t_k)
        # free old key states beyond the previous one (sequential evaluation)
        if k >= 2:
            self._key_V[k - 2] = None if k - 2 != 0 else self._key_V[0]

    def eval_frame(self, t: float) -> np.ndarray:
        b = self.base
        if t <= 0.0:
            return b.V0.copy()
        if t >= 1.0:
            return b.V1.copy()
        k = min(int(np.searchsorted(self.keys_t, t, side="right")) - 1, self.K - 1)
        self._ensure_key(k)
        t_k = self.keys_t[k]
        s = (t - t_k) / max(1.0 - t_k, 1e-15)
        return self._solve(k, s, t)


def build_unwrap_path(
    V0: np.ndarray,
    V1: np.ndarray,
    F: np.ndarray,
    uv: np.ndarray,
    anchor_idx: np.ndarray,
    *,
    solver_backend: str = "auto",
    axis_mode: str = "smooth",
    face_conf: np.ndarray | None = None,
    junk_weight: float = 0.05,
    axis_from_conf: bool = False,
    repo_root=None,
) -> UnwrapPath:
    """Precompute everything needed to evaluate frames (factorization reused across frames).

    face_conf (e.g. texture-alpha visibility): confident faces steer the φ unwrap routing
    AND carry full weight in the Poisson energy; junk faces get `junk_weight`× area so they
    follow the visible content instead of bending it (the solve is global).
    axis_from_conf: in 'smooth' axis mode, junk faces are not axis sources (their t_v is
    untrusted); the confident axis field is harmonically extended into junk instead.
    """
    V0 = np.asarray(V0, dtype=np.float64)
    V1 = np.asarray(V1, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    uv = np.asarray(uv, dtype=np.float64)

    Fg = deformation_gradients(V0, V1, F)
    R, S, inverted = polar_decompose(Fg)
    # seed: face whose centroid is nearest the anchor strip centroid
    anchor_c = V0[anchor_idx].mean(0)
    cent = V0[F].mean(1)
    seed = int(np.argmin(((cent - anchor_c) ** 2).sum(1)))
    decomp = winding_decomposition(R, V0, F, uv, V1, seed_face=seed, axis_mode=axis_mode,
                                   face_conf=face_conf, axis_from_conf=axis_from_conf)
    # Model-incoherent faces (residual ≈ π): their source→target rotation is a half-turn
    # away from anything the axis-winding model can express — locally misregistered UV
    # patches / fold-backs in merged traces (their TRUE transit is a local π spin; on
    # healthy meshes only isolated slivers). They cannot be animated coherently with the
    # sheet, so they become PASSENGERS: (1) the φ unwrap re-routes around them so they
    # corrupt nothing downstream, (2) their Poisson rows get junk weight so neighbors drag
    # them instead of being crumpled by them. Exactness at t=1 is untouched (the t=1
    # field equals the exact target gradients for every face).
    res_norm = np.linalg.norm(decomp["w_res"], axis=1)
    incoherent = res_norm > 0.9 * np.pi
    if incoherent.any():
        conf2 = (~incoherent) if face_conf is None else (face_conf & ~incoherent)
        if conf2.any():
            decomp = winding_decomposition(R, V0, F, uv, V1, seed_face=seed,
                                           axis_mode=axis_mode, face_conf=conf2,
                                           axis_from_conf=axis_from_conf)
            res_norm = np.linalg.norm(decomp["w_res"], axis=1)
            incoherent = res_norm > 0.9 * np.pi
        # NOTE: 1-ring dilation of this mask over-suppresses: the
        # full-weight boundary ring anchors patch borders to the sheet; down-weighting
        # it gives patches a longer free leash (measured SD peak 1.65 -> 2.06).
    unwrap_info = {k: v for k, v in decomp.items()
                   if k not in ("phi", "axis", "axis_per_face", "w_res")}
    unwrap_info["n_incoherent_passengers"] = int(incoherent.sum())

    # Rigid drift removal anchors on the strip's MODEL-STATIONARY subset: faces within
    # half a turn of φ=0 and model-coherent. Merged traces can have most of their strip
    # on sub-sheets wound 1-3 full turns deep (their strip faces genuinely ORBIT the
    # axis mid-transit); a Kabsch over orbiting points swings the whole sheet by up
    # to ~100° in a frame. On healthy meshes the whole strip
    # qualifies and the alignment is unchanged.
    n = len(V0)
    strip_mask = np.zeros(n, dtype=bool)
    strip_mask[np.asarray(anchor_idx)] = True
    fs = strip_mask[F].any(axis=1)
    stationary = fs & (np.abs(decomp["phi"]) <= np.pi) & (res_norm <= 0.9 * np.pi)
    align_verts = np.intersect1d(np.unique(F[stationary]), np.asarray(anchor_idx))
    align_rotation = True
    if len(align_verts) < max(8, 0.01 * len(anchor_idx)):
        # No model-stationary strip subset exists (the whole strip is wound k turns deep
        # — merged multi-depth traces): a rotation Kabsch onto an orbiting strip swings
        # the entire sheet. Fall back to translation-only drift removal; the solution's
        # orientation is already fully determined by the gradient field.
        align_verts = np.asarray(anchor_idx)
        align_rotation = False
    unwrap_info["n_align_verts"] = int(len(align_verts))
    unwrap_info["align_rotation"] = align_rotation

    G = grad_operator(V0, F)
    A = face_areas(V0, F)
    w_face = np.ones(len(F))
    if face_conf is not None:
        w_face[~face_conf] = junk_weight
    w_face[incoherent] = junk_weight
    if (w_face != 1.0).any():
        A = A * w_face
    W = sp.diags(np.tile(A, 3))
    L = (G.T @ W @ G).tocsr()

    GV0 = G @ V0                                   # (3m,3)
    m = len(F)
    J0 = GV0.reshape(3, m, 3).transpose(1, 0, 2)   # (m,3,3) rows=world deriv, cols=coords

    # one pin per connected component (debris islets would otherwise leave L singular);
    # the main component is pinned at the anchor vertex.
    n = len(V0)
    adj = sp.csr_matrix(
        (np.ones(3 * m, dtype=bool),
         (np.concatenate([F[:, 0], F[:, 1], F[:, 2]]),
          np.concatenate([F[:, 1], F[:, 2], F[:, 0]]))),
        shape=(n, n),
    )
    n_comp, labels = sp.csgraph.connected_components(adj, directed=False)
    pins = [int(anchor_idx[0])]
    main_label = labels[pins[0]]
    for c in range(n_comp):
        if c != main_label:
            pins.append(int(np.argmax(labels == c)))
    pins = np.array(sorted(pins))
    keep = np.setdiff1d(np.arange(n), pins)
    L_red = L[keep][:, keep].tocsr()
    solver = FactorizedSPD(L_red, backend=solver_backend, repo_root=repo_root)

    info = {
        "n_inverted_source_faces": int(inverted.sum()),
        **unwrap_info,
        "seed_face": seed,
        "n_components": int(n_comp),
        "pins": pins.tolist() if len(pins) <= 16 else f"{len(pins)} pins",
    }
    return UnwrapPath(
        V0=V0, V1=V1, F=F, uv=uv, decomp=decomp, S=S, G=G, A=A, J0=J0,
        solver=solver, L=L, keep=keep, pins=pins,
        anchor_idx=np.asarray(anchor_idx), info=info, face_conf=face_conf,
        align_idx=align_verts, align_rotation=align_rotation,
    )
