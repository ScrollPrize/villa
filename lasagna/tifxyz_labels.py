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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive cos / grad_mag / validity for all voxels bracketed between
    chain-adjacent surfaces.

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

    Returns:
        cos, grad_mag, valid — all (Z, Y, X); cos/grad_mag are 0 outside
        ``valid``.
    """
    if not dts:
        raise ValueError("derive_cos_gradmag_validity requires at least one surface")
    shape = dts[0].shape
    device = dts[0].device
    N_total = len(dts)

    cos_out = torch.zeros(shape, device=device)
    grad_mag_out = torch.zeros(shape, device=device)
    valid_out = torch.zeros(shape, dtype=torch.bool, device=device)

    if N_total == 0:
        return cos_out, grad_mag_out, valid_out

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
            grad_mag_full = 1.0 / (spacing + 1e-6)

            cos_out = torch.where(local_valid, cos_full, cos_out)
            grad_mag_out = torch.where(local_valid, grad_mag_full, grad_mag_out)
            valid_out |= local_valid

    return cos_out, grad_mag_out, valid_out


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
    3. The unsigned median of ``dts[j]`` sampled over voxels where
       ``mask_i`` is true is ``<= threshold`` (or symmetrically the
       reverse direction).

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
            # Symmetric: median distance from A's support to B, and
            # from B's support to A; take the smaller so we don't miss
            # a pair where one side happens to be tiny.
            med_ij = float(torch.median(dts[j][mask_i]).item())
            med_ji = float(torch.median(dts[i][mask_j]).item())
            if min(med_ij, med_ji) <= thr:
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
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict], list[int]]:
    """Collapse each group of surfaces into one merged representative.

    For every group of size ≥ 2:
      - merged mask = element-wise OR of the member masks
      - merged EDT  = element-wise ``torch.minimum`` of the member dts.
        This is exact: ``dt(~(A∪B)) = min(dt(~A), dt(~B))`` because
        the distance to the union is the min of per-set distances.
      - merged chain_info dict is cloned from the first member; a
        ``merged_from`` tuple records the original slot indices, so
        downstream code (vis) can still iterate all originals.

    Returns ``(merged_dts, merged_masks, merged_chain_info, merge_groups)``
    where ``merge_groups[k]`` is the merged slot that original slot
    ``k`` belongs to.
    """
    N = sum(len(g) for g in groups)
    merge_groups = [0] * N
    merged_dts: list[torch.Tensor] = []
    merged_masks: list[torch.Tensor] = []
    merged_info: list[dict] = []
    for new_slot, group in enumerate(groups):
        for original in group:
            merge_groups[original] = new_slot
        if len(group) == 1:
            k = group[0]
            merged_masks.append(surface_masks[k])
            merged_dts.append(dts[k])
            merged_info.append(dict(surface_chain_info[k]))
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
    return merged_dts, merged_masks, merged_info, merge_groups


# ---------------------------------------------------------------------------
# Top-level: compute all patch labels
# ---------------------------------------------------------------------------

def compute_patch_labels(
    surface_masks: list[torch.Tensor],
    direction_channels: torch.Tensor,
    normals_valid: torch.Tensor,
    surface_chain_info: list[dict],
    device: torch.device = None,
    same_surface_threshold: float | None = None,
) -> dict[str, torch.Tensor]:
    """Compute 8 training channels + validity using externally supplied chains.

    The chain ordering comes from ``tifxyz_lasagna_dataset.build_patch_chains``
    (geometry + filename-winding compatibility) and is handed down via
    ``surface_chain_info`` — one dict per surface_mask with
    chain/pos/has_prev/has_next. We no longer scan EDTs to build a chain.

    Validity (cos + grad_mag) is restricted to voxels that are strictly
    between two chain-adjacent surfaces. Normals_valid (direction channels)
    is passed through unchanged — it already only covers voxels on a surface.

    When ``same_surface_threshold`` is set, duplicate wraps (consecutive
    in chain, from different segments, with unsigned median distance
    <= threshold) are merged into a single surface before the chain
    bracketing runs. The merge reuses the already-computed per-surface
    EDTs; see :func:`detect_same_surface_groups` and
    :func:`apply_same_surface_merge`.

    Args:
        surface_masks: list of N bool tensors (Z, Y, X).
        direction_channels: (6, Z, Y, X) float32 — pre-splatted direction values.
        normals_valid: (Z, Y, X) bool — where splatting produced valid normals.
        surface_chain_info: list of N per-mask chain dicts (aligned order).
        device: target device (defaults to surface_masks[0].device).
        same_surface_threshold: optional voxel-median distance threshold
            for the same-surface merge. ``None`` disables merging and
            keeps ``merge_groups`` as the identity mapping.

    Returns:
        dict with keys:
          - 'targets'        (8, Z, Y, X)
          - 'validity'       (Z, Y, X)
          - 'normals_valid'  (Z, Y, X)
          - 'merge_groups'   list[int] of length N, mapping each
            original surface slot to its merged slot index.
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
            "normals_valid": empty_normals_valid,
            "merge_groups": [],
        }

    # EDT of complement for each surface
    dts = []
    for mask in surface_masks:
        complement = ~mask
        dts.append(edt_torch(complement.to(torch.uint8)))

    if same_surface_threshold is not None and N >= 2:
        groups = detect_same_surface_groups(
            dts, surface_masks, surface_chain_info,
            threshold=float(same_surface_threshold),
        )
        dts, surface_masks, surface_chain_info, merge_groups = \
            apply_same_surface_merge(
                dts, surface_masks, surface_chain_info, groups,
            )
    else:
        merge_groups = list(range(N))

    chains = chains_from_surface_info(surface_chain_info)
    cos, grad_mag, valid = derive_cos_gradmag_validity(dts, surface_masks, chains)

    targets = torch.zeros(8, *shape, device=device)
    targets[0] = cos
    targets[1] = grad_mag
    if direction_channels is not None and direction_channels.numel() > 0:
        targets[2:8] = direction_channels

    return {
        "targets": targets,
        "validity": valid,
        "normals_valid": empty_normals_valid,
        "merge_groups": merge_groups,
    }
