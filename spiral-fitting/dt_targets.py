"""Whole-object DT target determination.

The DT losses (patch, unattached-pcl strip, track) pull every sampled point towards a
single integer-winding target per object. Historically that target was
round(median(this step's sampled strip) / dr): recomputed from a small fresh sample
every step, so an object sitting roughly halfway between two windings flip-flops its
target across the rounding boundary from step to step -- a symmetric tug-of-war that
can freeze the fit with the object stuck between windings (unless neighbouring DT
constraints happen to break the tie).

This module instead determines each object's target winding from a sparse, no-grad
sample of the WHOLE object.  Normally its median is snapped to the nearest winding. If
a majority of its points are genuinely away from every winding, however, it is treated
as floating in a gap and targets the outward candidate: the DT loss then pulls a
more-outward spiral winding inward onto the object. Targets are cached and refreshed
on the shared theta-crossing-map cadence (the target-determination pass transforms
every object's sample points, so recomputing it each step would roughly duplicate the
loss forward cost).

Frames: unwrapped shifted radii are only defined up to an integer number of windings
(the choice of unwrap reference point). Patches reuse the shared theta map's root
frame directly; their target cache is invalidated whenever that map refreshes.
Strips/tracks retain a sparse field of integer unwrap adjustments and theta at known
within-strip point indices, because main tracks are not part of the shared theta map.
A strip loss anchors to its nearest cached position and transfers the target through
that anchor alone (see snap_strip_dt_target).
"""

import numpy as np
import torch

from spiral_sampling import load_spiral_sampling
from sample_spiral import get_theta_and_radii


_spiral_sampling = load_spiral_sampling()


class DtTargetCacheManager:
    """Refresh independently keyed DT-target caches at a fixed step interval."""

    def __init__(self, update_interval, on_first_update=None):
        self.update_interval = max(1, int(update_interval))
        self.on_first_update = on_first_update
        self._caches = {}
        self._last_updates = {}

    def reset(self):
        # Invalidate everything (e.g. after interactive input appends change the
        # object pools the caches index into); next get() recomputes.
        self._caches.clear()
        self._last_updates.clear()

    def get(self, kind, iteration, compute_fn):
        last_update = self._last_updates.get(kind)
        if last_update is None or iteration - last_update >= self.update_interval:
            cache = compute_fn()
            self._caches[kind] = cache
            self._last_updates[kind] = iteration
            if last_update is None and self.on_first_update is not None:
                self.on_first_update(kind, cache)
        return self._caches[kind]


def _transform_in_chunks(transform, zyxs, chunk_size):
    if zyxs.shape[0] <= chunk_size:
        return transform(zyxs)
    return torch.cat(
        [transform(zyxs[start:start + chunk_size]) for start in range(0, zyxs.shape[0], chunk_size)],
        dim=0,
    )


def _smallest_signed_dtype(minimum, maximum, *, prefer_int16=True):
    """Return the smallest practical signed dtype covering an integer range."""
    candidates = (torch.int16, torch.int32, torch.int64) if prefer_int16 \
        else (torch.int32, torch.int64)
    for dtype in candidates:
        bounds = torch.iinfo(dtype)
        if minimum >= bounds.min and maximum <= bounds.max:
            return dtype
    raise OverflowError(f'integer range [{minimum}, {maximum}] exceeds int64')


def _compact_integer_tensor(values, name, *, prefer_int16=True):
    """Losslessly convert an integer-valued tensor to a compact host tensor."""
    values = values.detach()
    if values.numel() == 0:
        dtype = torch.int16 if prefer_int16 else torch.int32
        return values.to(device='cpu', dtype=dtype)
    if values.is_floating_point():
        rounded = torch.round(values)
        error = (values - rounded).abs().max()
        if float(error) > 1.e-4:
            raise ValueError(
                f'{name} is not integer-valued (maximum rounding error '
                f'{float(error):.6g})')
        values = rounded
    minimum = int(values.min())
    maximum = int(values.max())
    dtype = _smallest_signed_dtype(
        minimum, maximum, prefer_int16=prefer_int16)
    return values.to(device='cpu', dtype=dtype)


def _stage_host_tensor(values, indices, device, *, dtype=None):
    """Gather a small host subset and upload it to the loss device."""
    selected = values[indices]
    if dtype is not None:
        selected = selected.to(dtype=dtype)
    if device.type == 'cuda' and not selected.is_pinned():
        selected = selected.pin_memory()
    return selected.to(device=device, non_blocking=selected.is_pinned())


def snap_dt_target(sample_median, dr_per_winding):
    # Legacy per-sample DT target: round the sampled strip's median shifted-radius to
    # the nearest integer winding, in the sample's own unwrap frame.
    return torch.round(sample_median / dr_per_winding) * dr_per_winding


def _select_target_from_medians(median, median_distance_to_sheet, floating_threshold):
    """Apply the shared attached/floating whole-object target policy."""
    floating = median_distance_to_sheet > float(floating_threshold)
    nearest = torch.floor(median + 0.5)
    return torch.where(floating, torch.ceil(median), nearest)


def select_whole_object_target(values, dr_per_winding, floating_threshold):
    """Select one winding from a whole object's unwrapped shifted radii.

    ``floating_threshold`` is in winding units.  The median distance of the points to
    their nearest integer winding determines whether the object is floating, so a
    minority tail touching the outer sheet cannot pull an otherwise attached object
    outward.  Floating objects choose ceil(median), explicitly grabbing the outer
    sheet; attached objects use nearest-half-up rather than ties-to-even.
    """
    normalised = values / dr_per_winding
    median = normalised.median()
    distance_to_sheet = (normalised - torch.round(normalised)).abs()
    return _select_target_from_medians(
        median, distance_to_sheet.median(), floating_threshold,
    )


def strip_dt_target_in_sample_frame(
    sample_radii, sample_local_idx, sample_theta, sample_adjustments,
    dr_per_winding, cache, cache_idx,
):
    """Per-strip DT target winding, for the strip-shaped losses (unattached pcls, tracks).

    The single entry point covering both target modes: with no cache (legacy
    cfg['dt_target_mode'] == 'strip_median', under which callers never build one),
    every strip gets its own sampled median snapped to the nearest winding
    (snap_dt_target). Otherwise the cached whole-strip target is transferred into
    each sample's unwrap frame (snap_strip_dt_target), with the snapped median as
    per-strip fallback where the cache holds no usable entry. sample_local_idx and
    cache_idx may be numpy arrays or tensors."""
    median_target = snap_dt_target(sample_radii.median(dim=-1, keepdim=True).values, dr_per_winding)
    if cache is None:
        return median_target
    device = sample_theta.device
    sample_local_idx = torch.as_tensor(sample_local_idx, dtype=torch.int64, device=device)
    cache_idx = torch.as_tensor(cache_idx, dtype=torch.int64, device=device)
    target, valid = snap_strip_dt_target(
        sample_local_idx, sample_theta, sample_adjustments, dr_per_winding, cache, cache_idx,
    )
    return torch.where(valid[:, None], target, median_target)


def patch_dt_target_in_sample_frame(
    sample_radii, sample_ijs, sample_theta, sample_adjustments,
    dr_per_winding, cache, patch_indices, sample_mask=None,
):
    """Per-patch DT target winding for padded uniform 2D samples.

    ``sample_mask`` excludes padding from both the median fallback and cache
    anchor selection. ``patch_indices`` has one cache row per sampled patch and
    is broadcast over any leading sample dimensions."""
    if sample_mask is None:
        sample_mask = torch.ones_like(sample_radii, dtype=torch.bool)
    else:
        sample_mask = torch.as_tensor(
            sample_mask, dtype=torch.bool, device=sample_radii.device)
    counts = sample_mask.sum(dim=-1).clamp(min=1)
    sortable = torch.where(
        sample_mask, sample_radii, torch.full_like(sample_radii, torch.inf))
    sorted_radii = sortable.sort(dim=-1).values
    median_idx = torch.div(counts - 1, 2, rounding_mode='floor')
    sample_median = torch.gather(
        sorted_radii, -1, median_idx[..., None])
    median_target = snap_dt_target(sample_median, dr_per_winding)
    if cache is None:
        return median_target
    if cache.get('frame') == 'theta_potential':
        # Patch loss samples have already been lifted into the shared
        # ThetaCrossingMap's per-patch root frame. Whole-patch targets are
        # cached in that same frame, so no sparse anchor search is necessary.
        device = sample_theta.device
        cache_idx = torch.as_tensor(
            patch_indices, dtype=torch.int64, device='cpu')
        target = _stage_host_tensor(
            cache['target_relative'], cache_idx, device,
            dtype=sample_radii.dtype)
        valid = _stage_host_tensor(cache['valid'], cache_idx, device)
        target = torch.broadcast_to(target, sample_theta.shape[:-1])
        valid = torch.broadcast_to(valid, sample_theta.shape[:-1])
        return torch.where(
            valid[..., None], target[..., None] * dr_per_winding,
            median_target)
    cache_idx = torch.as_tensor(patch_indices, dtype=torch.int64, device=sample_theta.device)
    cache_idx = torch.broadcast_to(cache_idx, sample_theta.shape[:-1])
    target, valid = snap_patch_dt_target(
        sample_ijs, sample_theta, sample_adjustments, dr_per_winding, cache,
        cache_idx, sample_mask=sample_mask,
    )
    return torch.where(valid[..., None], target, median_target)


def _transfer_target_through_anchor(
    target_relative, sample_anchor_theta, sample_anchor_adjustment,
    cache_anchor_theta, cache_anchor_adjustment,
):
    # Move a cached integer target winding into the sample's unwrap frame: apply the
    # anchor's integer adjustments in the two frames, plus a +/-1 correction when its
    # wrapped theta has crossed the theta=0 seam between cache time and sample time.
    theta_delta = sample_anchor_theta - cache_anchor_theta
    local_crossing = (
        (theta_delta > np.pi).to(theta_delta.dtype)
        - (theta_delta < -np.pi).to(theta_delta.dtype)
    )
    return target_relative + sample_anchor_adjustment - cache_anchor_adjustment - local_crossing


def _stage_strip_dt_cache(cache, cache_idx, device, value_dtype):
    """Stage only the requested strips from a compact host cache."""
    if 'offsets' not in cache:
        return cache, cache_idx

    cache_idx_cpu = torch.as_tensor(
        cache_idx, dtype=torch.int64, device='cpu')
    unique, inverse = torch.unique(
        cache_idx_cpu.reshape(-1), sorted=True, return_inverse=True)
    offsets = cache['offsets'].to(torch.int64)
    begins = offsets[unique]
    counts = offsets[unique + 1] - begins
    staged_offsets = torch.zeros(unique.numel() + 1, dtype=torch.int64)
    torch.cumsum(counts, dim=0, out=staged_offsets[1:])
    total = int(staged_offsets[-1])
    if total:
        staged_strip = torch.repeat_interleave(
            torch.arange(unique.numel(), dtype=torch.int64), counts)
        within = (
            torch.arange(total, dtype=torch.int64)
            - staged_offsets[:-1][staged_strip])
        source = begins[staged_strip] + within
        local_idx = cache['local_idx'][source].to(torch.int64)
        keys_cpu = staged_strip * cache['key_scale'] + local_idx
    else:
        source = torch.empty(0, dtype=torch.int64)
        keys_cpu = torch.empty(0, dtype=torch.int64)

    staged = {
        'keys': _stage_host_tensor(keys_cpu, slice(None), device),
        'key_scale': cache['key_scale'],
        'theta': _stage_host_tensor(
            cache['theta'], source, device, dtype=value_dtype),
        'adjustment': _stage_host_tensor(
            cache['adjustment'], source, device, dtype=value_dtype),
        'target_relative': _stage_host_tensor(
            cache['target_relative'], unique, device, dtype=value_dtype),
        'valid': _stage_host_tensor(cache['valid'], unique, device),
    }
    remapped = inverse.reshape(cache_idx_cpu.shape).to(device=device)
    return staged, remapped


def snap_strip_dt_target(
    sample_local_idx, sample_theta, sample_adjustments,
    dr_per_winding, cache, cache_idx,
):
    """Express cached strip targets in sampled-subset unwrap frames.

    Each sampled strip is anchored to the cached (decimated) point nearest one of its
    sampled points by within-strip point index.  Only integer unwrap adjustments and
    the theta wrap difference at the anchor establish the frame correspondence;
    radii deliberately appear nowhere among the inputs, so a bimodal strip or drift
    since the last cache update cannot shift the target by a winding (mirrors
    snap_patch_dt_target's UV anchoring).

    sample_local_idx (K, P) are the sampled points' within-strip indices,
    sample_theta / sample_adjustments (K, P) the loss sample's wrapped theta and
    cumulative crossing adjustments (radius units), cache_idx the strip rows --
    (K,) when a row samples a single strip, or (K, P) per-point when a row mixes
    strips (a chain walk through a fiber-link component); the anchor is then the
    row's best valid (point, cache) pair and the target comes from the anchor
    point's strip.
    Returns (target (K, 1), valid (K,)); valid is False where the cache holds no
    usable entry for any of the row's strips (strip_dt_target_in_sample_frame then
    falls back to the snapped sample median).
    """
    cache, cache_idx = _stage_strip_dt_cache(
        cache, cache_idx, sample_theta.device, sample_theta.dtype)
    cache_idx = torch.as_tensor(
        cache_idx, dtype=torch.int64, device=sample_theta.device)
    if cache_idx.dim() == 1:
        cache_idx = cache_idx[:, None].expand_as(sample_local_idx)
    point_valid = cache['valid'][cache_idx]  # (K, P)
    valid = point_valid.any(dim=-1)
    keys = cache['keys']
    if keys.numel() == 0:
        target = torch.zeros(cache_idx.shape[0], 1, device=cache_idx.device, dtype=sample_theta.dtype)
        return target, torch.zeros_like(valid)
    # Composite keys (strip * key_scale + local index) are globally sorted, so the
    # nearest cached point of the right strip brackets each sample key's insertion
    # position; at least one bracket lies in the strip's own segment whenever the
    # strip has any cached points.
    sample_keys = cache_idx * cache['key_scale'] + sample_local_idx
    positions = torch.searchsorted(keys, sample_keys)
    candidates = torch.stack(
        [(positions - 1).clamp(min=0), positions.clamp(max=keys.numel() - 1)], dim=-1,
    )  # (K, P, 2)
    candidate_keys = keys[candidates]
    same_strip = torch.div(candidate_keys, cache['key_scale'], rounding_mode='floor') == cache_idx[..., None]
    gaps = (candidate_keys - sample_keys[..., None]).abs()
    gaps = gaps.masked_fill(~same_strip | ~point_valid[..., None], torch.iinfo(torch.int64).max)
    best = gaps.flatten(start_dim=1).argmin(dim=-1)  # closest sample/cache pair per row
    rows = torch.arange(best.shape[0], device=best.device)
    sample_anchor_idx = torch.div(best, 2, rounding_mode='floor')
    cache_anchor_idx = candidates.flatten(start_dim=1)[rows, best]

    target_winding = _transfer_target_through_anchor(
        cache['target_relative'][cache_idx[rows, sample_anchor_idx]],
        sample_theta.detach()[rows, sample_anchor_idx],
        sample_adjustments[rows, sample_anchor_idx] / dr_per_winding.detach(),
        cache['theta'][cache_anchor_idx],
        cache['adjustment'][cache_anchor_idx],
    )
    return target_winding[:, None] * dr_per_winding, valid


def snap_patch_dt_target(
    sample_ijs, sample_theta, sample_adjustments,
    dr_per_winding, cache, cache_idx, sample_mask=None,
):
    """Express cached patch targets in sampled-set unwrap frames.

    Each sampled set is anchored to its closest valid sparse cache point in UV
    space.  Only integer unwrap adjustments establish the frame correspondence;
    radii deliberately appear nowhere among the inputs, so real radial variation
    cannot shift the target by a winding.  Returns (target (..., 1), valid (...,));
    valid is False where the cache holds no usable entry for the patch, or where the
    nearest usable anchor is farther than the patch's anchor_dist_sq_limit in UV --
    e.g. a sample from a fragment disconnected from the main component --
    since the transfer's |dtheta| < pi assumption may fail across such a gap
    (patch_dt_target_in_sample_frame then falls back to the snapped sample median).
    """
    cache_ijs = cache['ijs'][cache_idx]
    cache_valid_points = cache['point_valid'][cache_idx]
    # (..., P, K): find the closest sample/cache pair, rather than committing to a
    # particular point in the randomly sampled set.
    distances_sq = ((sample_ijs[..., :, None, :] - cache_ijs[..., None, :, :]) ** 2).sum(dim=-1)
    distances_sq = distances_sq.masked_fill(~cache_valid_points[..., None, :], float('inf'))
    if sample_mask is not None:
        sample_mask = torch.as_tensor(
            sample_mask, dtype=torch.bool, device=distances_sq.device)
        distances_sq = distances_sq.masked_fill(
            ~sample_mask[..., :, None], float('inf'))
    num_cache_points = cache_ijs.shape[-2]
    anchor_dist_sq, nearest_flat = distances_sq.flatten(start_dim=-2).min(dim=-1)
    sample_anchor_idx = torch.div(nearest_flat, num_cache_points, rounding_mode='floor')
    cache_anchor_idx = nearest_flat % num_cache_points

    def gather_at(values, anchor_idx):
        return torch.gather(values, -1, anchor_idx[..., None]).squeeze(-1)

    target_winding = _transfer_target_through_anchor(
        cache['target_relative'][cache_idx],
        gather_at(sample_theta.detach(), sample_anchor_idx),
        gather_at(sample_adjustments, sample_anchor_idx) / dr_per_winding.detach(),
        gather_at(cache['theta'][cache_idx], cache_anchor_idx),
        gather_at(cache['relative_adjustment'][cache_idx], cache_anchor_idx),
    )
    valid = cache['valid'][cache_idx] & (anchor_dist_sq <= cache['anchor_dist_sq_limit'][cache_idx])
    return target_winding[..., None] * dr_per_winding, valid


def prepare_patch_dt_target_samples(patches, num_points, max_stride_voxels):
    # For every patch, precompute a sparse whole-grid sample for DT target
    # determination: split the in-ROI valid-quad bounding box into ~num_points blocks
    # (denser when needed so no block exceeds max_stride_voxels in physical voxel
    # coordinates, keeping |dtheta| between neighbouring samples small enough for the
    # unwrap flood fill), and in
    # each block keep the valid quad nearest the block centre, sampled at its quad
    # centre so the atlas bilinear lookup stays on a valid quad. Choosing one
    # representative per BLOCK (rather than a strict subgrid of quads) means small
    # holes in the valid mask don't disconnect the sample. Stores per patch:
    #   _dt_target_ijs         (K, 2) float32 fractional grid coords
    #   _dt_target_block_rc    (K, 2) int32 block-grid coords (for the unwrap flood fill)
    #   _dt_target_block_shape (nr, nc)
    # Requires patch._sampling_valid_quad_mask_np (set by prepare_patch_sampling_cache).
    # Sampling is deterministic so caches stay identical across DDP ranks.
    # Also stores _dt_target_anchor_max_dist_sq: the squared UV distance (~2 block
    # diagonals) beyond which a loss sample must not anchor to a cache point. A healthy
    # anchor lies in the sample's own or an adjacent block; anything farther sits across
    # a hole or disconnected fragment where the |dtheta| < pi transfer assumption may
    # fail, so snap_patch_dt_target reports the strip invalid (median fallback) instead.
    for patch in patches:
        mask = patch._sampling_valid_quad_mask_np
        scale = np.asarray(
            patch.scale.detach().cpu() if hasattr(patch.scale, 'detach') else patch.scale,
            dtype=np.float64,
        ).reshape(-1)
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        box_h, box_w = r1 - r0, c1 - c0
        target = max(1, int(num_points))
        nr = int(round(np.sqrt(target * box_h / box_w)))
        nr = min(max(nr, 1), box_h)
        nc = min(max(int(round(target / nr)), 1), box_w)
        if max_stride_voxels and max_stride_voxels > 0:
            # patch.scale is grid cells per voxel, so convert the shared physical
            # stride to independent row/column grid-cell bounds.
            row_stride = max(1, int(np.floor(float(max_stride_voxels) * scale[0])))
            col_stride = max(1, int(np.floor(float(max_stride_voxels) * scale[1])))
            nr = min(max(nr, -(-box_h // row_stride)), box_h)
            nc = min(max(nc, -(-box_w // col_stride)), box_w)
        row_edges = np.linspace(r0, r1, nr + 1)
        col_edges = np.linspace(c0, c1, nc + 1)
        patch._dt_target_anchor_max_dist_sq = 4.0 * ((box_h / nr) ** 2 + (box_w / nc) ** 2)
        if _spiral_sampling is not None:
            prepared = _spiral_sampling.prepare_dt_samples(
                np.ascontiguousarray(mask, dtype=bool),
                np.ascontiguousarray(row_edges, dtype=np.int64),
                np.ascontiguousarray(col_edges, dtype=np.int64),
            )
            patch._dt_target_ijs = np.asarray(
                prepared['ijs'], dtype=np.float32)
            patch._dt_target_block_rc = np.asarray(
                prepared['block_rc'], dtype=np.int32)
            patch._dt_target_block_shape = (nr, nc)
            continue
        ijs = []
        block_rc = []
        for bi in range(nr):
            row_lo = int(row_edges[bi])
            row_hi = max(int(row_edges[bi + 1]), row_lo + 1)
            for bj in range(nc):
                col_lo = int(col_edges[bj])
                col_hi = max(int(col_edges[bj + 1]), col_lo + 1)
                sub = mask[row_lo:row_hi, col_lo:col_hi]
                if not sub.any():
                    continue
                ii, jj = np.nonzero(sub)
                centre_i = (row_hi - row_lo - 1) / 2
                centre_j = (col_hi - col_lo - 1) / 2
                k = int(np.argmin((ii - centre_i) ** 2 + (jj - centre_j) ** 2))
                ijs.append((row_lo + ii[k] + 0.5, col_lo + jj[k] + 0.5))
                block_rc.append((bi, bj))
        patch._dt_target_ijs = np.asarray(ijs, dtype=np.float32).reshape(-1, 2)
        patch._dt_target_block_rc = np.asarray(block_rc, dtype=np.int32).reshape(-1, 2)
        patch._dt_target_block_shape = (nr, nc)


def _unwrap_block_samples(theta, block_rc, block_shape):
    # 2D theta-unwrap over the sparse block grid: flood fill across 4-neighbouring
    # blocks, accumulating an integer winding adjustment per sample from theta=0
    # crossings (same sign convention as get_theta_crossing_step_adjustments; assumes
    # |dtheta| < pi between neighbouring blocks, so any spanning tree gives the same
    # adjustments). Returns (adjustments, main_component_mask):
    # samples outside the largest connected component have an unknown integer frame
    # offset relative to it, so callers must exclude them from pooling.
    num_samples = len(theta)
    nr, nc = block_shape
    if _spiral_sampling is not None:
        result = _spiral_sampling.unwrap_block_samples(
            np.ascontiguousarray(theta, dtype=np.float32),
            np.ascontiguousarray(block_rc, dtype=np.int32),
            int(nr),
            int(nc),
        )
        return (
            np.asarray(result['adjustments'], dtype=np.int64),
            np.asarray(result['main'], dtype=bool),
        )
    idx_grid = np.full((nr, nc), -1, dtype=np.int64)
    idx_grid[block_rc[:, 0], block_rc[:, 1]] = np.arange(num_samples)
    adjustments = np.zeros(num_samples, dtype=np.int64)
    component = np.full(num_samples, -1, dtype=np.int64)
    num_components = 0
    for seed in range(num_samples):
        if component[seed] >= 0:
            continue
        component[seed] = num_components
        queue = [seed]
        while queue:
            cur = queue.pop()
            r, c = block_rc[cur]
            for nb_r, nb_c in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= nb_r < nr and 0 <= nb_c < nc):
                    continue
                nb = idx_grid[nb_r, nb_c]
                if nb < 0 or component[nb] >= 0:
                    continue
                dtheta = theta[nb] - theta[cur]
                step = int(dtheta > np.pi) - int(dtheta < -np.pi)
                adjustments[nb] = adjustments[cur] + step
                component[nb] = num_components
                queue.append(nb)
        num_components += 1
    sizes = np.bincount(component, minlength=max(num_components, 1))
    main = int(np.argmax(sizes)) if num_components > 0 else 0
    return adjustments, component == main


@torch.no_grad()
def compute_patch_dt_target_cache(
    slice_to_spiral_transform, dr_per_winding, patches, patch_atlas,
    crossing_map, floating_threshold, chunk_size=65536,
):
    """Choose one target per patch in the shared theta-potential frame.

    The ordinary patch loss path already lifts every sampled point with
    ``ThetaCrossingMap.adjustments_from_potentials``. Reusing that frame here
    avoids retaining a second UV/theta/adjustment atlas. The caller invalidates
    this cache whenever the theta map refreshes.
    """
    device = dr_per_winding.device
    num_patches = len(patches)
    counts = np.array([len(p._dt_target_ijs) for p in patches], dtype=np.int64)
    total = int(counts.sum())
    target_relative = torch.zeros(
        num_patches, dtype=torch.int64, device=device)
    valid = torch.from_numpy(counts > 0).to(device=device)
    if total > 0:
        ijs_np = np.concatenate([p._dt_target_ijs for p in patches], axis=0)
        patch_idx_np = np.repeat(np.arange(num_patches, dtype=np.int64), counts)
        node_ids_np = patch_atlas.theta_node_ids(patch_idx_np, ijs_np)
        # The atlas is host-resident: the gather runs on CPU and only the
        # interpolated points land on the device.
        zyxs = patch_atlas.lookup(
            torch.from_numpy(patch_idx_np), torch.from_numpy(ijs_np))
        spiral_zyxs = _transform_in_chunks(slice_to_spiral_transform, zyxs, chunk_size)
        theta, _, shifted = get_theta_and_radii(
            spiral_zyxs[..., 1:], dr_per_winding)
        node_ids = torch.as_tensor(
            node_ids_np, dtype=torch.int64, device=theta.device)
        adjustments = crossing_map.winding_potentials(
            node_ids, theta).to(shifted.dtype)
        values = shifted + adjustments * dr_per_winding.detach()
        values_cpu = values.float().cpu()
        dr = float(dr_per_winding.detach().cpu())
        offsets = np.concatenate([[0], np.cumsum(counts)])
        for n in range(num_patches):
            lo, hi = int(offsets[n]), int(offsets[n + 1])
            if hi == lo:
                continue
            # select_whole_object_target already returns winding units. Store
            # that integer directly; dividing by dr here would scale twice.
            selected = select_whole_object_target(
                values_cpu[lo:hi], dr, floating_threshold)
            target_relative[n] = torch.round(selected).to(torch.int64)
    return {
        'frame': 'theta_potential',
        'target_relative': _compact_integer_tensor(
            target_relative, 'patch target winding'),
        'valid': valid.cpu(),
        'num_points': total,
    }


def _compact_strip_cache_to_host(cache):
    """Convert a construction cache into ragged, compact host storage."""
    num_strips = int(cache['valid'].numel())
    keys = cache['keys']
    key_scale = int(cache['key_scale'])
    if keys.numel():
        strip_id = torch.div(keys, key_scale, rounding_mode='floor')
        counts = torch.bincount(strip_id, minlength=num_strips)
        local_idx = keys - strip_id * key_scale
    else:
        counts = torch.zeros(
            num_strips, dtype=torch.int64, device=keys.device)
        local_idx = torch.empty(0, dtype=torch.int64, device=keys.device)
    offsets = torch.zeros(
        num_strips + 1, dtype=torch.int64, device=counts.device)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return {
        'storage': 'host_compact',
        'offsets': _compact_integer_tensor(
            offsets, 'strip cache offsets', prefer_int16=False),
        'local_idx': _compact_integer_tensor(
            local_idx, 'strip local point index'),
        # Theta remains float32. Half precision near the +/-pi branch cut
        # cannot preserve exact crossing classification.
        'theta': cache['theta'].detach().to(device='cpu', dtype=torch.float32),
        'adjustment': _compact_integer_tensor(
            cache['adjustment'], 'strip unwrap adjustment'),
        'target_relative': _compact_integer_tensor(
            cache['target_relative'], 'strip target winding'),
        'valid': cache['valid'].detach().to(device='cpu', dtype=torch.bool),
        'key_scale': key_scale,
        'num_points': int(cache['num_points']),
    }


def _widest_integer_dtype(caches, field):
    rank = {torch.int16: 0, torch.int32: 1, torch.int64: 2}
    return max((cache[field].dtype for cache in caches), key=rank.__getitem__)


def _merge_compact_strip_caches(caches, key_scale):
    """Join already-host-resident strip chunks without returning to CUDA."""
    if not caches:
        raise ValueError('expected at least one strip cache chunk')
    counts = torch.cat([
        cache['offsets'][1:].to(torch.int64)
        - cache['offsets'][:-1].to(torch.int64)
        for cache in caches
    ])
    offsets = torch.zeros(counts.numel() + 1, dtype=torch.int64)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    def join_integer(field):
        dtype = _widest_integer_dtype(caches, field)
        return torch.cat([
            cache[field].to(dtype=dtype) for cache in caches
        ])

    return {
        'storage': 'host_compact',
        'offsets': _compact_integer_tensor(
            offsets, 'strip cache offsets', prefer_int16=False),
        'local_idx': join_integer('local_idx'),
        'theta': torch.cat([cache['theta'] for cache in caches]),
        'adjustment': join_integer('adjustment'),
        'target_relative': join_integer('target_relative'),
        'valid': torch.cat([cache['valid'] for cache in caches]),
        'key_scale': int(key_scale),
        'num_points': sum(cache['num_points'] for cache in caches),
    }


@torch.no_grad()
def compute_strip_dt_target_cache(
    slice_to_spiral_transform, dr_per_winding, zyxs, starts,
    windings=None, floating_threshold=0.25, num_points_per_strip=None, max_stride=None,
    chunk_size=65536, max_total_points=None, _compact_host=True,
):
    # Whole-strip DT target determination for ordered point strips (unattached-pcl
    # strips and tracks), given their flat concatenated bundle: zyxs (N, 3) and
    # starts (T+1,) both on device, plus per-point winding-annotation offsets
    # `windings` (N,; None => zeros, i.e. tracks). Long strips are decimated to at
    # approximately num_points_per_strip evenly-spaced points. max_stride is a hard
    # upper bound, in voxels, on the gap between retained points; strip points are
    # nominally at ~voxel spacing, so it is applied directly as an index stride, and
    # long strips get more than the target count when necessary. Both endpoints are
    # retained. This bounds the sampling distance underlying the theta-unwrap
    # adjacency assumption in the same way patch sampling bounds its grid stride
    # (there converted to grid cells via patch.scale). Values are unwrapped per
    # strip (segmented cumsum) and
    # annotation-normalised; the returned per-strip 'target_relative' lives in that
    # space, in the unwrap frame of each strip's first retained point. Alongside the
    # target, the cache keeps every retained point's within-strip index (as globally
    # sorted composite 'keys'), wrapped theta, and integer unwrap adjustment, which
    # snap_strip_dt_target uses to transfer the target into a loss sample's frame.
    device = dr_per_winding.device
    lengths = starts[1:] - starts[:-1]
    num_strips = int(lengths.numel())
    key_scale = int(lengths.max()) + 1 if num_strips > 0 else 1
    empty_cache = {
        'keys': torch.zeros(0, dtype=torch.int64, device=device),
        'key_scale': key_scale,
        'theta': torch.zeros(0, dtype=torch.float32, device=device),
        'adjustment': torch.zeros(0, dtype=torch.float32, device=device),
        'target_relative': torch.zeros(num_strips, dtype=torch.float32, device=device),
        'valid': torch.zeros(num_strips, dtype=torch.bool, device=device),
        'num_points': 0,
    }
    total = int(starts[-1]) if num_strips > 0 else 0
    if total == 0:
        return (_compact_strip_cache_to_host(empty_cache)
                if _compact_host else empty_cache)

    # Production-scale track sets (tens of millions of points) make the
    # single-shot segmented sorts below spike tens of GB (observed OOM at
    # z8500-16500, 2026-07-19). Split the strips into contiguous groups of at
    # most max_total_points, build each group's cache independently, and
    # compact each group to host memory immediately, then merge its ragged
    # arrays there. Group boundaries respect strip boundaries and ascend by
    # strip id, so offsets and per-strip targets concatenate directly.
    if max_total_points and total > int(max_total_points) and num_strips > 1:
        starts_cpu = starts.detach().cpu()
        subcaches = []
        s0 = 0
        while s0 < num_strips:
            limit = int(starts_cpu[s0]) + int(max_total_points)
            s1 = int(torch.searchsorted(starts_cpu, torch.tensor(limit),
                                        right=False))
            s1 = max(s0 + 1, min(s1, num_strips))
            p0, p1 = int(starts_cpu[s0]), int(starts_cpu[s1])
            sub = compute_strip_dt_target_cache(
                slice_to_spiral_transform, dr_per_winding,
                zyxs[p0:p1], starts[s0:s1 + 1] - starts[s0],
                windings=windings[p0:p1] if windings is not None else None,
                floating_threshold=floating_threshold,
                num_points_per_strip=num_points_per_strip,
                max_stride=max_stride, chunk_size=chunk_size,
                max_total_points=None,
            )
            subcaches.append(sub)
            s0 = s1
        if not _compact_host:
            raise ValueError(
                'chunked strip target caches require compact host output')
        return _merge_compact_strip_caches(subcaches, key_scale)

    target_counts = lengths.clone()
    if num_points_per_strip and int(num_points_per_strip) > 0:
        target_counts = torch.clamp(target_counts, max=int(num_points_per_strip))
    if max_stride and int(max_stride) > 0:
        # max_stride is in voxels; strip points are nominally at ~voxel spacing, so it
        # is applied directly as an index stride (patches instead convert theirs to
        # grid cells via patch.scale).
        # ceil((length - 1) / stride) intervals require one more endpoint.
        min_counts_for_stride = torch.div(
            (lengths - 1).clamp(min=0) + int(max_stride) - 1,
            int(max_stride), rounding_mode='floor',
        ) + 1
        target_counts = torch.maximum(target_counts, min_counts_for_stride)
    counts = torch.minimum(target_counts, lengths)

    if not torch.equal(counts, lengths):
        new_starts = torch.zeros(num_strips + 1, dtype=torch.int64, device=device)
        torch.cumsum(counts, dim=0, out=new_starts[1:])
        strip_id = torch.repeat_interleave(torch.arange(num_strips, device=device), counts)
        local = torch.arange(int(new_starts[-1]), device=device) - new_starts[:-1][strip_id]
        denominators = (counts[strip_id] - 1).clamp(min=1)
        local_idx = torch.div(
            local * (lengths[strip_id] - 1), denominators, rounding_mode='floor',
        )
        src = starts[:-1][strip_id] + local_idx
        # zyxs may live in host RAM (tracks keep the full point cloud on CPU);
        # gather there, then move only the decimated sample to the device.
        zyxs = zyxs[src.to(zyxs.device)].to(device)
        if windings is not None:
            windings = windings[src.to(windings.device)].to(device)
        starts = new_starts
        lengths = counts
    else:
        strip_id = torch.repeat_interleave(torch.arange(num_strips, device=device), lengths)
        local_idx = torch.arange(total, device=device) - starts[:-1][strip_id]
        zyxs = zyxs.to(device)
        if windings is not None:
            windings = windings.to(device)

    spiral_zyxs = _transform_in_chunks(slice_to_spiral_transform, zyxs, chunk_size)
    theta, _, shifted = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
    dr = dr_per_winding.detach()

    if shifted.numel() > 1:
        same_strip = strip_id[1:] == strip_id[:-1]
        theta_diffs = torch.diff(theta.detach())
        step_adjustments = (
            (theta_diffs > np.pi).to(torch.int32)
            - (theta_diffs < -np.pi).to(torch.int32))
        step_adjustments = torch.where(
            same_strip, step_adjustments,
            torch.zeros_like(step_adjustments))
        cumsum_flat = torch.cat([
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(step_adjustments, dim=0),
        ], dim=0)
        adjustments = cumsum_flat - cumsum_flat[starts[:-1][strip_id]]
    else:
        adjustments = torch.zeros_like(shifted, dtype=torch.int32)
    values = shifted + adjustments.to(shifted.dtype) * dr
    if windings is not None:
        values = values - windings * dr

    # Determine the same ambiguity-aware whole-object target as patches, using
    # segmented sorts to obtain per-strip medians without a GPU-synchronising loop.
    valid = lengths > 0
    normalised = values / dr
    distance_to_sheet = (normalised - torch.round(normalised)).abs()

    def segmented_median(v):
        order = torch.argsort(v)
        order = order[torch.argsort(strip_id[order], stable=True)]
        median_idx = starts[:-1] + torch.div((lengths - 1).clamp(min=0), 2, rounding_mode='floor')
        median_idx = median_idx.clamp(max=v.numel() - 1)
        return v[order][median_idx]

    selected = _select_target_from_medians(
        segmented_median(normalised),
        segmented_median(distance_to_sheet),
        floating_threshold,
    )
    target_relative = torch.where(valid, selected, torch.zeros_like(selected))
    result = {
        'keys': strip_id * key_scale + local_idx,
        'key_scale': key_scale,
        'theta': theta,
        'adjustment': adjustments,
        'target_relative': target_relative,
        'valid': valid,
        'num_points': int(values.numel()),
    }
    return (_compact_strip_cache_to_host(result)
            if _compact_host else result)
