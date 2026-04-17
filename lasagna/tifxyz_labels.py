"""Derive lasagna training channels from tifxyz surface masks on GPU.

All operations use PyTorch CUDA tensors. CuPy EDT is used for distance
transforms via zero-copy DLPack interop.

The 8 output channels are:
  cos (1ch), grad_mag (1ch), dir_z (2ch), dir_y (2ch), dir_x (2ch)
plus a validity mask indicating which voxels have reliable cos/grad_mag.

Reference algorithm: lasagna/labels_to_winding_volume.py lines 236-411.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


TAG = "[tifxyz_labels]"


# ---------------------------------------------------------------------------
# CuPy EDT interop
# ---------------------------------------------------------------------------

def edt_torch(mask: torch.Tensor) -> torch.Tensor:
    """Euclidean distance transform on a CUDA bool/uint8 tensor.

    Uses CuPy EDT via zero-copy DLPack round-trip. Input should be a 3D
    tensor on CUDA. Returns a float32 CUDA tensor of the same shape.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cupy_edt

    cp_arr = cp.from_dlpack(mask.contiguous())
    dt = cupy_edt(cp_arr, float64_distances=False)
    return torch.from_dlpack(dt).float()


def edt_torch_with_indices(
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """EDT + feature transform on a CUDA bool/uint8 tensor.

    Input voxels ≠ 0 are foreground; distance from each foreground
    voxel to the nearest background (zero) voxel, plus the ZYX
    coordinates of that nearest background voxel.

    Returns:
        dist: (Z, Y, X) float32 — CUDA.
        idx:  (3, Z, Y, X) int64 — CUDA.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cupy_edt

    cp_arr = cp.from_dlpack(mask.contiguous())
    dt, idx = cupy_edt(
        cp_arr,
        return_distances=True,
        return_indices=True,
        float64_distances=False,
    )
    return torch.from_dlpack(dt).float(), torch.from_dlpack(idx).long()


# ---------------------------------------------------------------------------
# Direction encoding
# ---------------------------------------------------------------------------

def encode_direction_channels(
    nx: torch.Tensor,
    ny: torch.Tensor,
    nz: torch.Tensor,
) -> torch.Tensor:
    """Double-angle direction encoding from 3D normal components.

    Returns (6, *spatial) tensor: [dir0_z, dir1_z, dir0_y, dir1_y, dir0_x, dir1_x].
    Each value in [0, 1].
    """
    eps = 1e-8
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def _encode_dir(gx: torch.Tensor, gy: torch.Tensor):
        r2 = gx * gx + gy * gy + eps
        cos2t = (gx * gx - gy * gy) / r2
        sin2t = 2.0 * gx * gy / r2
        d0 = 0.5 + 0.5 * cos2t
        d1 = 0.5 + 0.5 * (cos2t - sin2t) * inv_sqrt2
        return d0, d1

    dir0_z, dir1_z = _encode_dir(nx, ny)   # Z-slices (XY plane)
    dir0_y, dir1_y = _encode_dir(nx, nz)   # Y-slices (XZ plane)
    dir0_x, dir1_x = _encode_dir(ny, nz)   # X-slices (YZ plane)
    return torch.stack([dir0_z, dir1_z, dir0_y, dir1_y, dir0_x, dir1_x], dim=0)


def encode_from_tensor(
    tensor6: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Double-angle direction encoding computed directly from the
    6 unique components of the second-moment tensor ``N = n·nᵀ``.

    Args:
        tensor6: ``(6, ...)`` float32 tensor, channels in the order
            ``(nx², ny², nz², nx·ny, nx·nz, ny·nz)``. Any trailing
            spatial dims are preserved.

    Returns:
        ``(6, ...)`` float32 — the same 6-channel encoding produced
        by :func:`encode_direction_channels` when the input is a
        single unit vector, but computed from tensor components so
        that linear averaging of the tensor (which is the correct
        sign-invariant operation on direction distributions) flows
        straight into the encoding without ever constructing an
        intermediate "mean normal".

    This fixes the slerp-encoding spike: slerp between two sign-
    ambiguous normals can cross the encoding's singular axis and
    push an encoded channel to its extreme even when both endpoints
    are far from it. Blending the tensor is the correct operation
    for sign-invariant directions and stays smooth through those
    configurations.
    """
    nx2 = tensor6[0]
    ny2 = tensor6[1]
    nz2 = tensor6[2]
    nxny = tensor6[3]
    nxnz = tensor6[4]
    nynz = tensor6[5]
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def _pair(a2: torch.Tensor, b2: torch.Tensor, ab: torch.Tensor):
        r2 = a2 + b2 + eps
        cos2t = (a2 - b2) / r2
        sin2t = 2.0 * ab / r2
        d0 = 0.5 + 0.5 * cos2t
        d1 = 0.5 + 0.5 * (cos2t - sin2t) * inv_sqrt2
        return d0, d1

    dir0_z, dir1_z = _pair(nx2, ny2, nxny)   # Z-slices (XY plane)
    dir0_y, dir1_y = _pair(nx2, nz2, nxnz)   # Y-slices (XZ plane)
    dir0_x, dir1_x = _pair(ny2, nz2, nynz)   # X-slices (YZ plane)
    return torch.stack(
        [dir0_z, dir1_z, dir0_y, dir1_y, dir0_x, dir1_x], dim=0,
    )


def decode_to_tensor(
    encoded: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Invert :func:`encode_from_tensor` for a unit-vector direction.

    Given a 6-channel ``(d0_z, d1_z, d0_y, d1_y, d0_x, d1_x)`` input
    that represents an encoded unit direction, solve the coupled
    plane equations for the 6 unique second-moment tensor components
    ``(nx², ny², nz², nx·ny, nx·nz, ny·nz)``.

    Args:
        encoded: ``(6, ...)`` float32 double-angle encoding.

    Returns:
        ``(6, ...)`` float32 tensor moments. For inputs that are
        exactly the encoding of a unit vector, the output is the
        corresponding rank-1 ``n·nᵀ``. For predicted (non-unit)
        encodings, it's a regularized best-fit tensor — suitable
        for computing angular distances against a GT tensor.

    Robust to encoding-singularity configurations via clamped
    denominators: at ``c_y → -1`` (pred → "nx is degenerate in the
    XZ plane"), the solution gracefully drives ``nx²`` toward 0.
    """
    sqrt2 = math.sqrt(2.0)
    d0_z = encoded[0]
    d1_z = encoded[1]
    d0_y = encoded[2]
    d1_y = encoded[3]
    d0_x = encoded[4]
    d1_x = encoded[5]

    # Recover (cos 2θ, sin 2θ) per plane — direct inverse of the
    # _pair() branch in encode_from_tensor.
    c_z = 2.0 * d0_z - 1.0
    s_z = c_z - (2.0 * d1_z - 1.0) * sqrt2
    c_y = 2.0 * d0_y - 1.0
    s_y = c_y - (2.0 * d1_y - 1.0) * sqrt2
    c_x = 2.0 * d0_x - 1.0
    s_x = c_x - (2.0 * d1_x - 1.0) * sqrt2

    # Solve the coupled system for the three plane sums.
    #
    #   nx² = A(1+c_z)/2 = B(1+c_y)/2   →   B = A·(1+c_z)/(1+c_y)
    #   ny² = A(1-c_z)/2 = C(1+c_x)/2   →   C = A·(1-c_z)/(1+c_x)
    #   A + B + C = 2                   (∵ nx²+ny²+nz² = 1)
    one_plus_cy = (1.0 + c_y).clamp(min=eps)
    one_plus_cx = (1.0 + c_x).clamp(min=eps)
    ratio_b = (1.0 + c_z) / one_plus_cy
    ratio_c = (1.0 - c_z) / one_plus_cx
    denom = (1.0 + ratio_b + ratio_c).clamp(min=eps)
    A = 2.0 / denom
    B = A * ratio_b
    C = A * ratio_c

    nx2 = (A * (1.0 + c_z) * 0.5).clamp(min=0.0, max=1.0)
    ny2 = (A * (1.0 - c_z) * 0.5).clamp(min=0.0, max=1.0)
    nz2 = (B * (1.0 - c_y) * 0.5).clamp(min=0.0, max=1.0)

    # Off-diagonals recovered directly from sin 2θ: sin2t = 2·ab/(a²+b²).
    nxny = s_z * A * 0.5
    nxnz = s_y * B * 0.5
    nynz = s_x * C * 0.5

    return torch.stack([nx2, ny2, nz2, nxny, nxnz, nynz], dim=0)


def tensor_unsigned_angle_deg(
    t_pred: torch.Tensor,
    t_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unsigned mean angular error (in degrees) between two
    second-moment tensor fields — inherently sign-invariant.

    For two rank-1 tensors ``N_p = n_p · n_pᵀ`` and ``N_g = n_g · n_gᵀ``
    built from unit vectors, the Frobenius inner product satisfies
    ``⟨N_p, N_g⟩_F = (n_p · n_g)²`` — a **sign-invariant** squared
    cosine of the angle between the two (unsigned) directions. We
    then take ``arccos(|cos|) · 180/π`` to get the unsigned angle
    in [0°, 90°].

    Args:
        t_pred: ``(6, ...)`` float32 — pred tensor components.
        t_gt:   ``(6, ...)`` float32 — GT tensor components.
        mask:   optional ``(...)`` float/bool — weight or selection
            mask. When provided, the reported scalar is the
            mask-weighted mean angle.

    Returns:
        scalar ``torch.Tensor`` — mean unsigned angular error in
        degrees across the masked region (or the whole volume
        when ``mask`` is None).

    The off-diagonal tensor components are carried with a factor
    of 2 in the sum because the symmetric matrix has each off-
    diagonal pair stored once in the 6-vector but contributes
    twice in the full ``tr(A · B)``.
    """
    cos_sq = (
        t_pred[0] * t_gt[0]
        + t_pred[1] * t_gt[1]
        + t_pred[2] * t_gt[2]
        + 2.0 * (
            t_pred[3] * t_gt[3]
            + t_pred[4] * t_gt[4]
            + t_pred[5] * t_gt[5]
        )
    )
    cos_sq = cos_sq.clamp(min=0.0, max=1.0)
    # Small subtraction under the sqrt so ``acos`` gets a value
    # strictly inside [-1, 1]; 1e-12 floors identical tensors at
    # ~5.7e-5 degrees rather than 0.028°.
    cos_abs = torch.sqrt((cos_sq - 1e-12).clamp(min=0.0))
    angle_rad = torch.acos(cos_abs.clamp(max=1.0))
    angle_deg = angle_rad * (180.0 / math.pi)
    if mask is not None:
        m = mask.float()
        num = (angle_deg * m).sum()
        denom = m.sum().clamp(min=1.0)
        return num / denom
    return angle_deg.mean()


def slerp_unit(
    n1: torch.Tensor,
    n2: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Sign-invariant spherical linear interpolation between two unit
    vector fields.

    Shapes:
        n1, n2 : ``(3, ...)`` — any trailing spatial dims.
        t      : broadcasts against the spatial dims of ``n1``.
                 Either trailing-spatial (e.g. ``(Z, Y, X)``) or
                 channel-prefixed ``(1, Z, Y, X)``.

    Returns ``(3, ...)`` unit vectors interpolated at fraction ``t``.

    Numerically safe:
      - **Hemisphere collapse**: if ``n1 · n2 < 0`` flip ``n2`` so
        slerp always takes the shorter arc. The double-angle
        direction encoding is sign-invariant, so mirroring into
        ``n1``'s hemisphere is the correct representative.
      - **Parallel fallback**: when ``sin(θ) < 1e-4`` fall back to
        plain lerp — slerp degenerates to lerp in the limit, and
        dividing by a tiny sine blows up.
      - **Renormalize**: kill fp drift with a final unit-length
        projection.
    """
    if t.ndim == n1.ndim - 1:
        t = t.unsqueeze(0)
    dot = (n1 * n2).sum(dim=0, keepdim=True)
    sign = torch.where(
        dot < 0, torch.full_like(dot, -1.0), torch.full_like(dot, 1.0),
    )
    n2 = n2 * sign
    dot = (dot * sign).clamp(max=1.0 - 1e-6)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    parallel = sin_theta < 1e-4
    w1 = torch.where(
        parallel, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta,
    )
    w2 = torch.where(
        parallel,       t, torch.sin(       t * theta) / sin_theta,
    )
    out = w1 * n1 + w2 * n2
    return out / out.norm(dim=0, keepdim=True).clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Scale-space validity pooling (shared with ScaleSpaceLoss3D and vis)
# ---------------------------------------------------------------------------

def scale_space_pool_validity(m: torch.Tensor) -> torch.Tensor:
    """Single step of validity scale-space pooling — **any-valid** rule.

    A coarse voxel is valid iff *at least one* of its eight fine children
    was valid. Implemented as a plain ``max_pool3d`` on the validity
    mask. Both ``ScaleSpaceLoss3D`` and ``lasagna3d dataset vis`` call
    this so the validity mask the loss applies at each scale is exactly
    what the visualization shows.

    This pairs with the masked-average pooling of the prediction and
    target tensors in ``ScaleSpaceLoss3D``: at coarser scales we average
    the signal only over valid voxels, so as long as the block contains
    *any* valid voxel we have a meaningful coarse target and should keep
    supervising it.
    """
    return F.max_pool3d(m, kernel_size=2, stride=2)


def scale_space_validity_pyramid(
    m: torch.Tensor, num_scales: int,
) -> list[torch.Tensor]:
    """Return ``[level_0, level_1, ..., level_{n-1}]`` of validity masks.

    ``level_0 = m`` (full res). Each subsequent level is one
    ``scale_space_pool_validity`` step. Stops early if spatial dims drop
    below 2. Used by vis to display the full pyramid; loss code keeps
    calling ``scale_space_pool_validity`` incrementally to avoid holding
    every level in memory at once.
    """
    levels = [m]
    for _ in range(num_scales - 1):
        cur = levels[-1]
        if cur.shape[-1] < 2 or cur.shape[-2] < 2 or cur.shape[-3] < 2:
            break
        levels.append(scale_space_pool_validity(cur))
    return levels


# ---------------------------------------------------------------------------
# Chain reconstruction from per-mask metadata
# ---------------------------------------------------------------------------

def chains_from_surface_info(surface_chain_info: list[dict]) -> list[list[int]]:
    """Group per-surface chain metadata into ordered chains.

    Input: one dict per surface_mask with keys chain/pos/has_prev/has_next
    (produced by ``tifxyz_lasagna_dataset.build_patch_chains`` and wired
    through the dataset). Returns a list of chains, each a list of
    surface_mask indices sorted by pos in that chain. Surfaces with
    ``chain == -1`` are treated as standalone singletons.
    """
    groups: dict[int, list[tuple[int, int]]] = {}
    standalone: list[int] = []
    for surf_idx, info in enumerate(surface_chain_info):
        cid = int(info.get("chain", -1))
        if cid < 0:
            standalone.append(surf_idx)
            continue
        groups.setdefault(cid, []).append((int(info.get("pos", 0)), surf_idx))
    chains: list[list[int]] = []
    for cid in sorted(groups.keys()):
        ordered = [surf_idx for _, surf_idx in sorted(groups[cid])]
        chains.append(ordered)
    for surf_idx in standalone:
        chains.append([surf_idx])
    return chains


# ---------------------------------------------------------------------------
# DT gradient helpers
# ---------------------------------------------------------------------------

def _gradient_3d(vol: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute spatial gradient of a 3D volume via central differences.

    Returns (gz, gy, gx) each with the same shape as vol.
    Uses torch.gradient which handles boundary conditions with forward/backward diffs.
    """
    gz, gy, gx = torch.gradient(vol, dim=(0, 1, 2))
    return gz, gy, gx


def _dot_product_gradients(
    dt_a: torch.Tensor,
    dt_b: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute dot product of gradients of two distance transforms.

    If mask is provided, only computes within masked region (for memory
    efficiency on large volumes). Returns full-shape tensor.
    """
    dot = torch.zeros_like(dt_a)
    for ax in range(3):
        ga = torch.gradient(dt_a, dim=ax)[0]
        gb = torch.gradient(dt_b, dim=ax)[0]
        if mask is not None:
            dot[mask] += (ga[mask] * gb[mask])
        else:
            dot += ga * gb
        del ga, gb
    return dot


# ---------------------------------------------------------------------------
# Core: cos, grad_mag, validity derivation
# ---------------------------------------------------------------------------

def derive_cos_gradmag_validity(
    dts: list[torch.Tensor],
    surface_masks: list[torch.Tensor],
    chains: list[list[int]],
    fts: Optional[list[torch.Tensor]] = None,
    tensor_moments: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    """Derive cos / grad_mag / validity for all voxels bracketed between
    chain-adjacent surfaces; optionally also a densified second-moment
    tensor volume linearly blended at the same bracket.

    The ordering is externally supplied (see
    ``tifxyz_lasagna_dataset.build_patch_chains``). Unlike the old
    EDT-scanning path, we support multiple independent chains in the same
    patch and ONLY supervise voxels that lie strictly between two
    chain-adjacent surfaces — i.e., the "between neighboring surfaces"
    region. Voxels outside the envelope of a chain, or inside a chain-of-one,
    are left invalid.

    Args:
        dts: list of N float32 CUDA tensors — EDT of complement(~mask_k).
            Indexed by surface-mask index (same order as ``surface_masks``).
        surface_masks: list of N bool CUDA tensors.
        chains: list of chains, each a list of surface-mask indices in order.
        fts: optional list of per-wrap feature-transform tensors aligned
            with ``dts``. Each entry is ``(3, Z, Y, X)`` int64 giving the
            ZYX coordinates of the nearest on-wrap voxel per voxel.
            When provided together with ``tensor_moments`` the function
            also builds a dense tensor-moment volume by gathering at
            those indices and **linearly** blending the chain-adjacent
            bracket pair with the same routing as cos / grad_mag.
        tensor_moments: optional ``(6, Z, Y, X)`` float32 splatted
            second-moment tensor ``(nx², ny², nz², nx·ny, nx·nz, ny·nz)``
            — the sign-invariant representation of raw normals. Linear
            blending of these components is the correct sign-invariant
            operation on direction distributions and avoids the slerp
            encoding-singularity spike that happens when a geodesic
            midpoint lands on an encoding pole.

    Returns:
        cos, grad_mag, valid — all (Z, Y, X); cos/grad_mag are 0 outside
        ``valid``.
        t_dense — ``(6, Z, Y, X)`` float32 densified tensor-moment volume
            when ``fts`` + ``tensor_moments`` were provided, else
            ``None``. Zero outside ``valid``.
    """
    if not dts:
        raise ValueError("derive_cos_gradmag_validity requires at least one surface")
    shape = dts[0].shape
    device = dts[0].device
    N_total = len(dts)

    cos_out = torch.zeros(shape, device=device)
    grad_mag_out = torch.zeros(shape, device=device)
    valid_out = torch.zeros(shape, dtype=torch.bool, device=device)

    want_dir = fts is not None and tensor_moments is not None
    if want_dir:
        t_dense = torch.zeros_like(tensor_moments)
    else:
        t_dense = None

    if N_total == 0:
        return cos_out, grad_mag_out, valid_out, t_dense

    # Global "which surface is nearest" across ALL surfaces (any chain).
    # This decides, for each voxel, which surface's bracketing logic applies.
    dt_stack = torch.stack(dts, dim=0)           # (N_total, Z, Y, X)
    _, nearest_surf = dt_stack.min(dim=0)        # (Z, Y, X) int
    del dt_stack

    # Precompute DT gradients once per surface (shared across pos iterations).
    grads: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for dt in dts:
        gz, gy, gx = torch.gradient(dt, dim=(0, 1, 2))
        grads.append((gz, gy, gx))

    def _dot(i: int, j: int) -> torch.Tensor:
        gi = grads[i]
        gj = grads[j]
        return gi[0] * gj[0] + gi[1] * gj[1] + gi[2] * gj[2]

    for chain in chains:
        L = len(chain)
        if L < 2:
            # Chain-of-one has no neighbor pair — no bracketed voxels.
            continue
        for pos, surf_idx in enumerate(chain):
            if surf_idx < 0 or surf_idx >= N_total:
                continue
            is_nearest = nearest_surf == surf_idx
            if not is_nearest.any():
                continue

            has_prev = pos > 0
            has_next = pos < L - 1
            if not has_prev and not has_next:
                continue

            dt_near = dts[surf_idx]

            # "Between near and neighbor X" ↔ dot(grad(near), grad(X)) < 0:
            # the two gradients point toward each other.
            if has_prev:
                prev_idx = chain[pos - 1]
                dot_prev = _dot(surf_idx, prev_idx)
                between_prev = (dot_prev < 0) & is_nearest
                dt_prev = dts[prev_idx]
            else:
                between_prev = torch.zeros(shape, dtype=torch.bool, device=device)
                dt_prev = dt_near
            if has_next:
                next_idx = chain[pos + 1]
                dot_next = _dot(surf_idx, next_idx)
                between_next = (dot_next < 0) & is_nearest
                dt_next = dts[next_idx]
            else:
                between_next = torch.zeros(shape, dtype=torch.bool, device=device)
                dt_next = dt_near

            # Near-surface carve-out (1-voxel halo, dt_near < 1.5).
            #
            # The dt-gradient between-ness test is unreliable in three
            # cases right at/next to the surface:
            #
            #   1. On-surface (dt_near=0): grad(dt_near)≈0 via central
            #      differences — dot test degenerates to ~0.
            #   2. Adjacent (dt_near=1): along axes parallel to the
            #      surface, central differences give (1−1)/2=0, so
            #      grad(dt_near) collapses to a single component. If
            #      grad(dt_prev) has its nonzero component on a
            #      different axis (curved/non-coplanar prev), the dot
            #      can be exactly 0 — strict `< 0` rejects it.
            #   3. Argmin tie-break at dt=1 from two surfaces (sheets
            #      two voxels apart, intersections): only the lower
            #      index "wins" is_nearest, the loser never processes
            #      that voxel, and the winner's gradient test may also
            #      fail.
            #
            # All three live within a 1-voxel halo. We bypass the dot
            # test there and let the bracketing formula produce the
            # natural cos value (1 at dt_near=0, ~0.976 at dt_near=1
            # — matching what the dot test already supervises at the
            # next voxel out).
            near_surface = (dt_near < 1.5) & is_nearest

            # The dt-gradient test, when it works, tells us *which side*
            # of `near` the voxel is on (between prev or between next).
            # The carve-out only kicks in for voxels where the test is
            # unreliable — so we should only let the carve-out *route*
            # voxels that the dt test couldn't classify.
            unrouted_carveout = near_surface & ~between_prev & ~between_next

            # Routing precedence: trust the dt test first; for unrouted
            # carve-out voxels prefer `next` (so on-surface voxels hit
            # d_lo = dt_near = 0 → cos(0) = 1 directly), falling back to
            # `prev` only when there is no next neighbour.
            use_next = between_next.clone()
            use_prev = between_prev.clone()
            if has_next:
                use_next = use_next | unrouted_carveout
            elif has_prev:
                # No next neighbour — carve-out has to use prev side.
                use_prev = use_prev | unrouted_carveout
            # Mutual exclusivity: a voxel routed via use_next cannot
            # also be routed via use_prev (off-surface bracketed voxels
            # already disjoint by construction; this matters for the
            # rare double-bracket case at degenerate geometries).
            use_prev = use_prev & ~use_next
            local_valid = use_prev | use_next
            if not local_valid.any():
                continue

            d_lo = torch.where(use_prev, dt_prev, dt_near)
            d_hi = torch.where(use_prev, dt_near, dt_next)
            spacing = d_lo + d_hi
            # Full-period winding cos over the *entire* inter-sheet gap:
            # frac = d_lo / spacing runs 0 → 1 from the "lo" surface to
            # the "hi" surface, and 0.5 + 0.5*cos(2π·frac) goes 1 → 0 → 1
            # smoothly across it. The old half-period + clamp formula
            # clipped the second half of every gap to 0 and produced a
            # discontinuous jump at the midway transition between the
            # use_prev and use_next branches.
            frac = d_lo / (spacing + 1e-6)
            cos_full = 0.5 + 0.5 * torch.cos(2.0 * math.pi * frac)
            grad_mag_full = (1.0 / (spacing + 1e-6)).clamp(max=0.5)

            cos_out = torch.where(local_valid, cos_full, cos_out)
            grad_mag_out = torch.where(local_valid, grad_mag_full, grad_mag_out)
            valid_out |= local_valid

            # Second-moment tensor linear blend — uses the SAME
            # routed bracket pair and the SAME ``frac`` as cos /
            # grad_mag. The tensor ``N = n·nᵀ`` is the sign-
            # invariant representation of an unsigned direction
            # (``nnᵀ = (−n)(−n)ᵀ``), and linear interpolation of
            # its components is the correct averaging operation
            # for direction distributions. This avoids both the
            # sign ambiguity of raw-normal averaging AND the
            # encoding-singularity spike that slerp through a
            # geodesic midpoint hits when the midpoint lands on an
            # encoding pole. The final 6-channel encoding is
            # derived from the blended tensor in compute_patch_labels
            # via ``encode_from_tensor``.
            #
            # On each side: lo = the surface whose dt runs from 0
            # at the surface to (spacing) at the opposite one.
            #   use_prev voxels: lo = prev, hi = near.
            #   use_next voxels: lo = near, hi = next.
            if want_dir:
                def _gather(ft: torch.Tensor) -> torch.Tensor:
                    return tensor_moments[:, ft[0], ft[1], ft[2]]

                t_near = _gather(fts[surf_idx])
                t_lo = torch.zeros_like(t_near)
                t_hi = torch.zeros_like(t_near)
                if has_prev:
                    t_prev = _gather(fts[prev_idx])
                    use_prev_b = use_prev.unsqueeze(0)
                    t_lo = torch.where(use_prev_b, t_prev, t_lo)
                    t_hi = torch.where(use_prev_b, t_near, t_hi)
                    del t_prev
                if has_next:
                    t_next = _gather(fts[next_idx])
                    use_next_b = use_next.unsqueeze(0)
                    t_lo = torch.where(use_next_b, t_near, t_lo)
                    t_hi = torch.where(use_next_b, t_next, t_hi)
                    del t_next
                del t_near

                frac_b = frac.unsqueeze(0)
                t_blend = (1.0 - frac_b) * t_lo + frac_b * t_hi

                local_valid_b = local_valid.unsqueeze(0)
                t_dense = torch.where(local_valid_b, t_blend, t_dense)
                del t_lo, t_hi, t_blend

    return cos_out, grad_mag_out, valid_out, t_dense


# ---------------------------------------------------------------------------
# Same-surface merge (shared by training + vis)
# ---------------------------------------------------------------------------

def detect_same_surface_groups(
    dts: list[torch.Tensor],
    surface_masks: list[torch.Tensor],
    surface_chain_info: list[dict],
    threshold: float,
) -> list[list[int]]:
    """Find groups of surfaces that should be treated as the same shell.

    A pair ``(i, j)`` is merged iff all of these hold:

    1. They share a ``chain`` and their ``pos`` differs by exactly 1
       (consecutive in chain ordering).
    2. They come from different source segments
       (``segment_idx_i != segment_idx_j``).
    3. The unsigned 25th percentile of ``dts[j]`` sampled over voxels
       where ``mask_i`` is true is ``<= threshold`` (or symmetrically
       the reverse direction).

    ``dts[k]`` is ``edt_torch(~mask_k)`` — the same list the caller
    already built for loss derivation, so detection reuses those
    tensors instead of running fresh distance transforms.

    Returns one list of original slot indices per group (singletons
    included), in order of first appearance.
    """
    N = len(surface_masks)
    parent = list(range(N))

    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    thr = float(threshold)
    for i in range(N):
        info_i = surface_chain_info[i]
        chain_i = info_i.get("chain", -1)
        pos_i = int(info_i.get("pos", 0))
        seg_i = info_i.get("segment_idx", None)
        mask_i = surface_masks[i]
        if mask_i.dtype != torch.bool:
            mask_i = mask_i > 0.5
        if not bool(mask_i.any()):
            continue
        for j in range(i + 1, N):
            info_j = surface_chain_info[j]
            if info_j.get("chain", -2) != chain_i:
                continue
            if abs(int(info_j.get("pos", 0)) - pos_i) != 1:
                continue
            seg_j = info_j.get("segment_idx", None)
            if seg_i is None or seg_j is None or seg_i == seg_j:
                continue
            mask_j = surface_masks[j]
            if mask_j.dtype != torch.bool:
                mask_j = mask_j > 0.5
            if not bool(mask_j.any()):
                continue
            # Symmetric: p25 distance from A's support to B, and
            # from B's support to A; take the smaller so we don't miss
            # a pair where one side happens to be tiny.
            q25_ij = float(torch.quantile(
                dts[j][mask_i].float(), 0.25).item())
            q25_ji = float(torch.quantile(
                dts[i][mask_j].float(), 0.25).item())
            if min(q25_ij, q25_ji) <= thr:
                _union(i, j)

    groups_by_root: dict[int, list[int]] = {}
    for i in range(N):
        r = _find(i)
        groups_by_root.setdefault(r, []).append(i)
    # Order groups by their first member's original index.
    ordered_roots = sorted(groups_by_root.keys())
    return [sorted(groups_by_root[r]) for r in ordered_roots]


def apply_same_surface_merge(
    dts: list[torch.Tensor],
    surface_masks: list[torch.Tensor],
    surface_chain_info: list[dict],
    groups: list[list[int]],
    fts: Optional[list[torch.Tensor]] = None,
    normals_valid: Optional[torch.Tensor] = None,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[dict],
    list[int],
    Optional[list[torch.Tensor]],
]:
    """Collapse each group of surfaces into one merged representative.

    For every group of size ≥ 2:
      - merged mask = element-wise OR of the member masks
      - merged EDT  = element-wise ``torch.minimum`` of the member dts.
        This is exact: ``dt(~(A∪B)) = min(dt(~A), dt(~B))`` because
        the distance to the union is the min of per-set distances.
      - merged chain_info dict is cloned from the first member; a
        ``merged_from`` tuple records the original slot indices, so
        downstream code (vis) can still iterate all originals.

    Returns ``(merged_dts, merged_masks, merged_chain_info,
    merge_groups, merged_fts)`` where ``merge_groups[k]`` is the
    merged slot that original slot ``k`` belongs to and
    ``merged_fts`` is ``None`` iff ``fts`` was ``None``.

    When ``fts`` is provided, each merged group's feature transform
    is recomputed from the merged mask via ``edt_torch_with_indices``
    so the nearest-on-wrap lookup is exact for the unioned surface.
    If ``normals_valid`` is also supplied, the feature transform is
    computed from ``~(merged_mask & normals_valid)`` instead — this
    ensures the slerp gather in ``derive_cos_gradmag_validity``
    always lands on a voxel that actually carries splatted raw
    normals (surface-mask voxels outside the splat footprint would
    otherwise produce (0, 0, 0) gathers and slerp garbage).
    """
    N = sum(len(g) for g in groups)
    merge_groups = [0] * N
    merged_dts: list[torch.Tensor] = []
    merged_masks: list[torch.Tensor] = []
    merged_info: list[dict] = []
    merged_fts: Optional[list[torch.Tensor]] = [] if fts is not None else None
    for new_slot, group in enumerate(groups):
        for original in group:
            merge_groups[original] = new_slot
        if len(group) == 1:
            k = group[0]
            merged_masks.append(surface_masks[k])
            merged_dts.append(dts[k])
            merged_info.append(dict(surface_chain_info[k]))
            if merged_fts is not None:
                merged_fts.append(fts[k])
            continue
        m = surface_masks[group[0]]
        if m.dtype != torch.bool:
            m = m > 0.5
        d = dts[group[0]]
        for k in group[1:]:
            mk = surface_masks[k]
            if mk.dtype != torch.bool:
                mk = mk > 0.5
            m = m | mk
            d = torch.minimum(d, dts[k])
        info = dict(surface_chain_info[group[0]])
        info["merged_from"] = tuple(group)
        merged_masks.append(m)
        merged_dts.append(d)
        merged_info.append(info)
        if merged_fts is not None:
            # Recompute the feature transform from the merged mask,
            # intersected with normals_valid when available so the
            # gather always lands on splatted voxels.
            if normals_valid is not None:
                ft_src = m & normals_valid
            else:
                ft_src = m
            _, merged_ft = edt_torch_with_indices((~ft_src).to(torch.uint8))
            merged_fts.append(merged_ft)
    return merged_dts, merged_masks, merged_info, merge_groups, merged_fts


# ---------------------------------------------------------------------------
# Top-level: compute all patch labels
# ---------------------------------------------------------------------------

def compute_patch_labels(
    surface_masks: list[torch.Tensor],
    tensor_moments: torch.Tensor,
    normals_valid: torch.Tensor,
    surface_chain_info: list[dict],
    device: torch.device = None,
    same_surface_groups: list[list[int]] | None = None,
    precomputed_dts: list[torch.Tensor] | None = None,
    precomputed_fts: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute 8 training channels + validity using externally supplied chains.

    The chain ordering comes from ``tifxyz_lasagna_dataset.build_patch_chains``
    (geometry + filename-winding compatibility) and is handed down via
    ``surface_chain_info`` — one dict per surface_mask with
    chain/pos/has_prev/has_next. We no longer scan EDTs to build a chain.

    Validity (cos + grad_mag) is restricted to voxels that are strictly
    between two chain-adjacent surfaces. Direction supervision is densely
    filled inside the same bracket by **linearly blending** the splatted
    ``tensor_moments`` at the chain-adjacent pair (see
    :func:`derive_cos_gradmag_validity`). The final densified tensor is
    encoded to the 6-channel double-angle representation via
    :func:`encode_from_tensor` ONCE at the end, just before writing
    ``targets[2:8]``. The tensor representation is sign-invariant and
    smooth under linear interpolation, which avoids both the sign
    ambiguity of raw-normal averaging and the encoding-singularity spike
    that slerp through a geodesic midpoint hits.

    Merging is controlled by ``same_surface_groups`` — a
    ``list[list[int]]`` of original slot indices to collapse
    together, computed upstream by whoever cares about the merge
    (training: :func:`compute_batch_targets`; analysis:
    :func:`run_dataset_overlap`). ``compute_patch_labels`` itself
    does not detect. It applies
    :func:`apply_same_surface_merge` when groups are given and
    otherwise leaves the surfaces alone. Keeping the detection out
    of this helper means every caller runs detection in exactly
    one place.

    Args:
        surface_masks: list of N bool tensors (Z, Y, X).
        tensor_moments: (6, Z, Y, X) float32 — sparsely splatted
            second-moment tensor components
            ``(nx², ny², nz², nx·ny, nx·nz, ny·nz)``. Sign-invariant
            representation of unsigned direction. Linearly blended at
            the chain-adjacent bracket, encoded via
            ``encode_from_tensor`` to produce ``targets[2:8]``.
        normals_valid: (Z, Y, X) bool — where splatting produced
            valid tensor moments. Voxels here keep their original
            splatted tensor at ``dir_sparse_mask``-weight 1.0.
        surface_chain_info: list of N per-mask chain dicts (aligned order).
        device: target device (defaults to surface_masks[0].device).
        same_surface_groups: optional ``list[list[int]]`` of original
            slot indices to collapse. ``None`` leaves surfaces alone.
        precomputed_dts: optional list of per-surface EDTs to reuse.
            When the caller already ran the distance transforms (e.g.
            during detection), passing them here avoids redoing the
            EDT loop. Must be aligned with ``surface_masks``.
        precomputed_fts: optional list of per-surface feature transforms
            (``(3, Z, Y, X)`` int64, one per surface) aligned with
            ``precomputed_dts``. If omitted but ``tensor_moments`` is
            provided, they're computed here via
            ``edt_torch_with_indices``.

    Returns:
        dict with keys:
          - 'targets'              (8, Z, Y, X) — channel 0 cos, 1 grad_mag,
                                                 2..7 encoded directions.
          - 'validity'             (Z, Y, X) bool — cos/grad_mag supervision.
          - 'dir_sparse_mask'      (Z, Y, X) bool — voxels with original
                                                    splatted raw normals
                                                    (hard supervision).
          - 'dir_dense_mask'       (Z, Y, X) bool — voxels filled by the
                                                    slerp blend inside the
                                                    validity bracket.
          - 'dir_axis_weight'      (6, Z, Y, X) float32 — per-plane
                                                          relevance weight
                                                          (sqrt of in-plane
                                                          normal magnitude).
          - 'merge_groups'         list[int] of length N_original,
            mapping each original surface slot to its merged slot index.
          - 'merged_surface_masks' list[bool Tensor], len N_merged —
            the post-merge masks actually consumed by
            derive_cos_gradmag_validity. Identity with the input
            when the merge is off (or every group is a singleton).
          - 'merged_chain_info'    list[dict], len N_merged — the
            post-merge chain_info entries (representative's label /
            chain / pos / has_prev / has_next, plus ``merged_from``
            for multi-member groups). Identity otherwise.

        Exposing the merged lists lets callers (e.g. the vis) reuse
        the exact state the loss saw instead of re-deriving it.
    """
    N = len(surface_masks)
    if device is None and N > 0:
        device = surface_masks[0].device

    shape = surface_masks[0].shape if N > 0 else (1, 1, 1)

    empty_normals_valid = (
        normals_valid if normals_valid is not None
        else torch.zeros(shape, dtype=torch.bool, device=device)
    )

    if N == 0:
        return {
            "targets": torch.zeros(8, *shape, device=device),
            "validity": torch.zeros(shape, dtype=torch.bool, device=device),
            "dir_sparse_mask": torch.zeros(shape, dtype=torch.bool, device=device),
            "dir_dense_mask": torch.zeros(shape, dtype=torch.bool, device=device),
            "dir_axis_weight": torch.zeros(6, *shape, device=device),
            "merge_groups": [],
            "merged_surface_masks": [],
            "merged_chain_info": [],
        }

    # EDT of complement for each surface — reuse precomputed if caller
    # already ran them (e.g. compute_batch_targets does detection).
    # ``dts`` are always built from ``~surface_mask`` so cos routing +
    # same-surface detection see the same distance fields they always
    # did. But the feature transforms ``fts`` used for the slerp
    # gather must point to voxels that actually carry splatted raw
    # normals — ``surface_masks`` and ``normals_valid`` are built by
    # different rasterizers (line-drawing vs trilinear splat), so
    # there can be surface-mask voxels where the splat never landed.
    # Computing fts from ``~(surface_mask & normals_valid)`` ensures
    # the gather always lands on a voxel with real data and avoids
    # the "slerp from a zero vector" artifact.
    want_dir = (
        tensor_moments is not None and tensor_moments.numel() > 0
    )
    nv_bool: Optional[torch.Tensor] = None
    if want_dir:
        nv_bool = empty_normals_valid.bool()

    def _ft_src(mask: torch.Tensor) -> torch.Tensor:
        # Source set for the feature transform: intersect the
        # voxelized surface mask with the splat validity so the
        # nearest-on-wrap lookup always hits a splatted voxel.
        if nv_bool is not None:
            return (mask & nv_bool)
        return mask

    if precomputed_dts is not None:
        assert len(precomputed_dts) == N, (
            "precomputed_dts must be aligned with surface_masks"
        )
        dts = list(precomputed_dts)
        if precomputed_fts is not None:
            assert len(precomputed_fts) == N, (
                "precomputed_fts must be aligned with surface_masks"
            )
            fts: Optional[list[torch.Tensor]] = list(precomputed_fts)
        elif want_dir:
            # Caller gave us dts but not fts — backfill the feature
            # transforms from the splat-intersected source set.
            fts = [
                edt_torch_with_indices(
                    (~_ft_src(mask)).to(torch.uint8),
                )[1]
                for mask in surface_masks
            ]
        else:
            fts = None
    else:
        dts = []
        fts = [] if want_dir else None
        for mask in surface_masks:
            dts.append(edt_torch((~mask).to(torch.uint8)))
            if want_dir:
                _, ft = edt_torch_with_indices(
                    (~_ft_src(mask)).to(torch.uint8),
                )
                fts.append(ft)

    if same_surface_groups is not None and N >= 2:
        dts, surface_masks, surface_chain_info, merge_groups, fts = \
            apply_same_surface_merge(
                dts, surface_masks, surface_chain_info, same_surface_groups,
                fts=fts,
                normals_valid=nv_bool,
            )
    else:
        merge_groups = list(range(N))

    chains = chains_from_surface_info(surface_chain_info)
    cos, grad_mag, valid, t_dense = derive_cos_gradmag_validity(
        dts, surface_masks, chains,
        fts=fts,
        tensor_moments=tensor_moments if want_dir else None,
    )

    targets = torch.zeros(8, *shape, device=device)
    targets[0] = cos
    targets[1] = grad_mag

    if want_dir and t_dense is not None:
        # Sparse override: voxels where the dataset actually splatted
        # a tensor keep that hard value unchanged — they are the
        # "ground truth" points. Everywhere else inside the validity
        # bracket we use the chain-adjacent linear tensor blend from
        # ``t_dense``.
        sparse_mask = empty_normals_valid.bool()
        sparse_mask_b = sparse_mask.unsqueeze(0)
        t_final = torch.where(sparse_mask_b, tensor_moments, t_dense)

        # Encode ONCE, at the very end, directly from the tensor.
        # encode_from_tensor computes the double-angle channels
        # from the six second-moment components without ever going
        # through a mean-direction vector — so linear interpolation
        # of the tensor flows straight into a smooth encoded target
        # with no slerp singularity spikes.
        targets[2:8] = encode_from_tensor(t_final)

        # Per-plane relevance weight — a degenerate direction in a
        # given slice plane has near-zero in-plane "magnitude", and
        # the tensor gives us that magnitude directly as the sum of
        # the two diagonal components for the plane's axis pair.
        # ``sqrt(nx² + ny²)`` on a single unit vector equals the
        # 2D projection magnitude; on a blended tensor it's the
        # "mean in-plane second moment", which is the correct
        # generalization.
        eps = 1e-6
        nx2 = t_final[0]
        ny2 = t_final[1]
        nz2 = t_final[2]
        w_z = torch.sqrt(nx2 + ny2 + eps)  # XY plane (Z-slices)
        w_y = torch.sqrt(nx2 + nz2 + eps)  # XZ plane (Y-slices)
        w_x = torch.sqrt(ny2 + nz2 + eps)  # YZ plane (X-slices)
        dir_axis_weight = torch.stack(
            [w_z, w_z, w_y, w_y, w_x, w_x], dim=0,
        )

        # Two masks: hard (original splat) vs densified (chain blend).
        # Together they cover the full direction supervision region.
        dir_sparse_mask = sparse_mask
        dir_dense_mask = valid & ~sparse_mask
    else:
        dir_sparse_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        dir_dense_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        dir_axis_weight = torch.zeros(6, *shape, device=device)

    return {
        "targets": targets,
        "validity": valid,
        "dir_sparse_mask": dir_sparse_mask,
        "dir_dense_mask": dir_dense_mask,
        "dir_axis_weight": dir_axis_weight,
        "merge_groups": merge_groups,
        "merged_surface_masks": list(surface_masks),
        "merged_chain_info": list(surface_chain_info),
    }
