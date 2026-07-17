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
more-outward spiral winding inward onto the object. Targets are cached and
refreshed every cfg['dt_target_update_interval'] steps (the target-determination pass
transforms every object's sample points, so recomputing it each step would roughly
duplicate the loss forward cost).

Frames: unwrapped shifted radii are only defined up to an integer number of windings
(the choice of unwrap reference point). Caches therefore retain, per object, a sparse
field of integer unwrap adjustments at known positions -- UV coordinates for patches,
within-strip point indices for strips/tracks -- plus theta at cache time. A loss
sample anchors to its nearest cached position and transfers the target through the
integer adjustments and the theta wrap difference at the anchor alone, so real radial
variation (or drift since the last cache update) is never mistaken for an
unwrap-frame change (see snap_patch_dt_target / snap_strip_dt_target).
"""

import numpy as np
import torch

from sample_spiral import (
    get_theta_and_radii,
    get_theta_crossing_step_adjustments,
)


class DtTargetCacheManager:
    """Refresh independently keyed DT-target caches at a fixed step interval."""

    def __init__(self, update_interval, on_first_update=None):
        self.update_interval = max(1, int(update_interval))
        self.on_first_update = on_first_update
        self._caches = {}
        self._last_updates = {}

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
    dr_per_winding, cache, patch_indices,
):
    """Per-strip DT target winding for patch row/column strips; see strip_dt_target_in_sample_frame.

    patch_indices (numpy or tensor; one cache row per sampled patch) is broadcast
    over any leading sample dims (e.g. the row/column direction axis)."""
    median_target = snap_dt_target(sample_radii.median(dim=-1, keepdim=True).values, dr_per_winding)
    if cache is None:
        return median_target
    cache_idx = torch.as_tensor(patch_indices, dtype=torch.int64, device=sample_theta.device)
    cache_idx = torch.broadcast_to(cache_idx, sample_theta.shape[:-1])
    target, valid = snap_patch_dt_target(
        sample_ijs, sample_theta, sample_adjustments, dr_per_winding, cache, cache_idx,
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
    cumulative crossing adjustments (radius units), cache_idx (K,) the strip rows.
    Returns (target (K, 1), valid (K,)); valid is False where the cache holds no
    usable entry for the strip (strip_dt_target_in_sample_frame then falls back to the
    snapped sample median).
    """
    valid = cache['valid'][cache_idx]
    keys = cache['keys']
    if keys.numel() == 0:
        target = torch.zeros(*cache_idx.shape, 1, device=cache_idx.device, dtype=sample_theta.dtype)
        return target, torch.zeros_like(valid)
    # Composite keys (strip * key_scale + local index) are globally sorted, so the
    # nearest cached point of the right strip brackets each sample key's insertion
    # position; at least one bracket lies in the strip's own segment whenever the
    # strip has any cached points.
    sample_keys = cache_idx[:, None] * cache['key_scale'] + sample_local_idx
    positions = torch.searchsorted(keys, sample_keys)
    candidates = torch.stack(
        [(positions - 1).clamp(min=0), positions.clamp(max=keys.numel() - 1)], dim=-1,
    )  # (K, P, 2)
    candidate_keys = keys[candidates]
    same_strip = torch.div(candidate_keys, cache['key_scale'], rounding_mode='floor') == cache_idx[:, None, None]
    gaps = (candidate_keys - sample_keys[..., None]).abs()
    gaps = gaps.masked_fill(~same_strip, torch.iinfo(torch.int64).max)
    best = gaps.flatten(start_dim=1).argmin(dim=-1)  # closest sample/cache pair per strip
    rows = torch.arange(best.shape[0], device=best.device)
    sample_anchor_idx = torch.div(best, 2, rounding_mode='floor')
    cache_anchor_idx = candidates.flatten(start_dim=1)[rows, best]

    target_winding = _transfer_target_through_anchor(
        cache['target_relative'][cache_idx],
        sample_theta.detach()[rows, sample_anchor_idx],
        sample_adjustments[rows, sample_anchor_idx] / dr_per_winding.detach(),
        cache['theta'][cache_anchor_idx],
        cache['adjustment'][cache_anchor_idx],
    )
    return target_winding[:, None] * dr_per_winding, valid


def snap_patch_dt_target(
    sample_ijs, sample_theta, sample_adjustments,
    dr_per_winding, cache, cache_idx,
):
    """Express cached patch targets in sampled-strip unwrap frames.

    Each sampled strip is anchored to its closest valid sparse cache point in UV
    space.  Only integer unwrap adjustments establish the frame correspondence;
    radii deliberately appear nowhere among the inputs, so real radial variation
    cannot shift the target by a winding.  Returns (target (..., 1), valid (...,));
    valid is False where the cache holds no usable entry for the patch, or where the
    nearest usable anchor is farther than the patch's anchor_dist_sq_limit in UV --
    e.g. a strip sampled from a fragment disconnected from the main component --
    since the transfer's |dtheta| < pi assumption may fail across such a gap
    (patch_dt_target_in_sample_frame then falls back to the snapped sample median).
    """
    cache_ijs = cache['ijs'][cache_idx]
    cache_valid_points = cache['point_valid'][cache_idx]
    # (..., P, K): find the closest sample/cache pair, rather than committing to a
    # particular point on the randomly sampled strip.
    distances_sq = ((sample_ijs[..., :, None, :] - cache_ijs[..., None, :, :]) ** 2).sum(dim=-1)
    distances_sq = distances_sq.masked_fill(~cache_valid_points[..., None, :], float('inf'))
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
            scale = np.asarray(
                patch.scale.detach().cpu() if hasattr(patch.scale, 'detach') else patch.scale,
                dtype=np.float64,
            ).reshape(-1)
            # patch.scale is grid cells per voxel, so convert the shared physical
            # stride to independent row/column grid-cell bounds.
            row_stride = max(1, int(np.floor(float(max_stride_voxels) * scale[0])))
            col_stride = max(1, int(np.floor(float(max_stride_voxels) * scale[1])))
            nr = min(max(nr, -(-box_h // row_stride)), box_h)
            nc = min(max(nc, -(-box_w // col_stride)), box_w)
        row_edges = np.linspace(r0, r1, nr + 1)
        col_edges = np.linspace(c0, c1, nc + 1)
        patch._dt_target_anchor_max_dist_sq = 4.0 * ((box_h / nr) ** 2 + (box_w / nc) ** 2)
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
def compute_patch_dt_target_cache(slice_to_spiral_transform, dr_per_winding, patches, patch_atlas, floating_threshold, chunk_size=65536):
    # Whole-patch DT target determination: transform every patch's precomputed sparse
    # sample (see prepare_patch_dt_target_samples) through the current scroll->spiral
    # map in one batched pass, 2D-unwrap each patch's shifted radii over its block
    # grid, and pool the largest connected component. Returns padded per-patch GPU
    # tensors containing sparse UV coordinates, theta, relative integer adjustments,
    # and the selected target relative to the same reference adjustment. These let a
    # loss strip transfer the target through a nearby UV anchor without comparing
    # shifted radii. 'valid' is False where no usable sample exists and
    # 'anchor_dist_sq_limit' bounds how far that anchor may be (in both cases
    # snap_patch_dt_target falls back to the strip median); scalar stats
    # ('num_points', 'main_component_fraction') are included for logging.
    device = dr_per_winding.device
    num_patches = len(patches)
    counts = np.array([len(p._dt_target_ijs) for p in patches], dtype=np.int64)
    total = int(counts.sum())
    max_count = int(counts.max()) if num_patches else 0
    padded_ijs = np.zeros((num_patches, max_count, 2), dtype=np.float32)
    padded_theta = np.zeros((num_patches, max_count), dtype=np.float32)
    relative_adjustments = np.zeros((num_patches, max_count), dtype=np.float32)
    point_valid = np.zeros((num_patches, max_count), dtype=bool)
    target_relative = np.zeros(num_patches, dtype=np.float32)
    valid = counts > 0
    main_component_points = 0
    if total > 0:
        ijs_np = np.concatenate([p._dt_target_ijs for p in patches], axis=0)
        patch_idx_np = np.repeat(np.arange(num_patches, dtype=np.int64), counts)
        ijs_gpu = torch.from_numpy(ijs_np).to(device=device)
        patch_idx_gpu = torch.from_numpy(patch_idx_np).to(device=device)
        zyxs = patch_atlas.lookup(patch_idx_gpu, ijs_gpu)
        spiral_zyxs = _transform_in_chunks(slice_to_spiral_transform, zyxs, chunk_size)
        theta_t, _, shifted_t = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
        theta_np = theta_t.cpu().numpy()
        shifted_np = shifted_t.float().cpu().numpy()
        dr = float(dr_per_winding.detach())
        offsets = np.concatenate([[0], np.cumsum(counts)])
        for n, patch in enumerate(patches):
            lo, hi = offsets[n], offsets[n + 1]
            if hi == lo:
                continue
            adjustments, main = _unwrap_block_samples(
                theta_np[lo:hi], patch._dt_target_block_rc, patch._dt_target_block_shape,
            )
            values = shifted_np[lo:hi][main] + adjustments[main] * dr
            # Normalize both the sparse adjustment field and target to one arbitrary
            # point in the main component. The arbitrary BFS frame then cancels.
            reference_adjustment = adjustments[np.flatnonzero(main)[0]]
            count = hi - lo
            padded_ijs[n, :count] = patch._dt_target_ijs
            padded_theta[n, :count] = theta_np[lo:hi]
            relative_adjustments[n, :count] = adjustments - reference_adjustment
            point_valid[n, :count] = main
            target_relative[n] = float(select_whole_object_target(
                torch.from_numpy(values), dr, floating_threshold,
            )) - reference_adjustment
            main_component_points += int(main.sum())
    anchor_dist_sq_limits = np.array(
        [p._dt_target_anchor_max_dist_sq for p in patches], dtype=np.float32,
    ).reshape(num_patches)
    return {
        'ijs': torch.from_numpy(padded_ijs).to(device=device),
        'theta': torch.from_numpy(padded_theta).to(device=device),
        'relative_adjustment': torch.from_numpy(relative_adjustments).to(device=device),
        'point_valid': torch.from_numpy(point_valid).to(device=device),
        'target_relative': torch.from_numpy(target_relative).to(device=device),
        'valid': torch.from_numpy(valid).to(device=device),
        'anchor_dist_sq_limit': torch.from_numpy(anchor_dist_sq_limits).to(device=device),
        'num_points': total,
        'main_component_fraction': main_component_points / max(total, 1),
    }


@torch.no_grad()
def compute_strip_dt_target_cache(
    slice_to_spiral_transform, dr_per_winding, zyxs, starts,
    windings=None, floating_threshold=0.25, num_points_per_strip=None, max_stride=None,
    chunk_size=65536,
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
        return empty_cache

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
        zyxs = zyxs[src]
        if windings is not None:
            windings = windings[src]
        starts = new_starts
        lengths = counts
    else:
        strip_id = torch.repeat_interleave(torch.arange(num_strips, device=device), lengths)
        local_idx = torch.arange(total, device=device) - starts[:-1][strip_id]

    spiral_zyxs = _transform_in_chunks(slice_to_spiral_transform, zyxs, chunk_size)
    theta, _, shifted = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
    dr = dr_per_winding.detach()

    if shifted.numel() > 1:
        same_strip = strip_id[1:] == strip_id[:-1]
        step_adjustments = get_theta_crossing_step_adjustments(theta, dr)
        step_adjustments = torch.where(same_strip, step_adjustments, torch.zeros_like(step_adjustments))
        cumsum_flat = torch.cat([
            torch.zeros(1, device=device, dtype=step_adjustments.dtype),
            torch.cumsum(step_adjustments, dim=0),
        ], dim=0)
        adjustments = cumsum_flat - cumsum_flat[starts[:-1][strip_id]]
    else:
        adjustments = torch.zeros_like(shifted)
    values = shifted + adjustments
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
    return {
        'keys': strip_id * key_scale + local_idx,
        'key_scale': key_scale,
        'theta': theta,
        'adjustment': adjustments / dr,
        'target_relative': target_relative,
        'valid': valid,
        'num_points': int(values.numel()),
    }
