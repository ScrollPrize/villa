import itertools

import numpy as np
import torch
import torch.nn.functional as F

import strip_path_pools
from sample_spiral import (
    canonical_winding_samples,
    get_theta_and_radii,
    radius_from_unwrapped_shifted,
    unwrap_shifted_radii,
)
from spiral_helpers import _huber_abs


cfg = None
z_begin = None
z_end = None


def configure_losses(config, z_begin_value, z_end_value):
    global cfg, z_begin, z_end
    cfg = config
    z_begin = z_begin_value
    z_end = z_end_value
    if cfg['patch_strip_sampling'] == 'dijkstra':
        strip_path_pools.warm_workers()


def _masked_mean(values, mask):
    mask_f = mask.to(values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp(min=1.)



def get_shell_outer_loss(shell_map, slice_to_spiral_transform, dr_per_winding, outer_winding_idx):
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if shell_map is None or outer_winding_idx is None:
        return zero, {}

    num_samples = max(1, int(cfg['shell_num_samples']))
    huber_delta = torch.as_tensor(cfg['shell_huber_delta'], device=device, dtype=torch.float32)

    outer_spiral = canonical_winding_samples([outer_winding_idx], num_samples, dr_per_winding, device, z_begin, z_end)[0]
    outer_scan = slice_to_spiral_transform.inv(outer_spiral)

    target_r, scan_r, confidence, valid = shell_map.lookup(outer_scan)
    residual = scan_r - target_r
    shell_outer_loss = _masked_mean(_huber_abs(residual, huber_delta), valid)

    metrics = {}
    with torch.no_grad():
        if valid.any():
            abs_residual = residual[valid].abs()
            metrics = {
                'shell_outer_error_mean': abs_residual.mean(),
                'shell_outer_error_p95': torch.quantile(abs_residual, 0.95),
                'shell_confidence_mean': confidence[valid].mean(),
            }

    return shell_outer_loss, metrics



def run_containing_index(mask_1d: np.ndarray, idx: int) -> tuple[int, int] | None:
    """Return (start, end) of the contiguous True run containing idx."""
    padded = np.concatenate([[False], mask_1d, [False]])
    diff = np.diff(padded.astype(int))  # diff will be +1 at start of runs, -1 at end of runs
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0] - 1
    run_idx = np.searchsorted(run_starts, idx, side='right') - 1
    return run_starts[run_idx], run_ends[run_idx] + 1



# ============================================================================================
# 'dijkstra' strip sampling (cfg['patch_strip_sampling'] == 'dijkstra'): instead of straight
# rows/columns (and cardinal L-shapes), strips are geodesic shortest paths on the 8-connected
# valid-quad graph, from a start cell to a 'distant' reachable endpoint, skirting holes and
# ragged edges. Consecutive path cells are grid-adjacent, so a path is a contiguous walk and
# unwrap_shifted_radii can stitch theta=0 crossings along it exactly as for straight
# strips. The paths come from per-patch / per-anchor pools built and continuously refreshed by
# background worker processes (see strip_path_pools.py; strip_paths.py has the actual path
# computation); here we only subsample positions along a pooled path + subpixel jitter.
# ============================================================================================

def _sample_points_along_path(path_ij, num_points):
    # Subsample `num_points` positions along a path (sorted in traversal order, so the unwrap
    # sees a contiguous walk) with per-point subpixel jitter in both axes; path cells are valid
    # quads, so jittered points stay on valid quads.
    path_len = path_ij.shape[0]
    positions = np.sort(np.random.choice(path_len, num_points, replace=num_points > path_len))
    return path_ij[positions].astype(np.float32) + np.random.uniform(0., 1., size=[num_points, 2]).astype(np.float32)


def _sample_dijkstra_strips_at_ij(patch, i_q, j_q, num_points):
    # 'dijkstra'-mode replacement for _sample_l_shapes_at_ij: 4 geodesic strips from the
    # annotated cell, one per cardinal cone; None while the anchor's pools are still being
    # built in the background. Caller guarantees valid_quad[i_q, j_q].
    pools = strip_path_pools.get_anchor_path_pools(patch, i_q, j_q)
    if pools is None:
        return None
    return [
        _sample_points_along_path(pool[np.random.randint(len(pool))], num_points)
        for pool in pools
    ]


def _sample_strip_ijs(line_valid, seed, fixed_coord, axis, num_points):
    # Sample num_points fractional ijs along the contiguous True run of `line_valid`
    # containing `seed`, fixed at `fixed_coord` along `axis` (axis=0 -> fixed i, varying j;
    # axis=1 -> fixed j, varying i), with sub-pixel jitter. Caller guarantees line_valid[seed].
    # The contiguous range lets unwrap_shifted_radii reliably handle theta=0 crossings.
    lo, hi = run_containing_index(line_valid, seed)
    run_len = hi - lo
    coords = np.sort(np.random.choice(run_len, num_points, replace=num_points > run_len))
    ijs = np.empty([num_points, 2], dtype=np.float32)
    var_axis = 1 - axis
    ijs[:, axis] = fixed_coord + float(np.random.uniform(0., 1.))
    ijs[:, var_axis] = lo + coords + np.random.uniform(0., 1., size=num_points)
    return ijs




def _aggregate_dt_track_losses(track_losses, across_p, active_mask=None):
    # Power-mean across tracks/patches: ((sum x^p) / n)^(1/p). When `active_mask` is given
    # (progressive DT gating), only the masked-in tracks contribute and n is the number active;
    # returns a zero scalar when none are active.
    if active_mask is not None:
        track_losses = track_losses[active_mask]
    if track_losses.numel() == 0:
        return torch.zeros([], device=track_losses.device)
    return ((track_losses ** across_p).sum() / track_losses.numel()) ** (1 / across_p)



def _progressive_dt_active_mask(snapped_winding, dr_per_winding, dt_max_winding):
    # Boolean mask over tracks/patches whose snapped spiral-space winding index is within the
    # progressive cutoff (see get_progressive_dt_max_winding); None when gating is disabled.
    # `snapped_winding` is the per-track round(median(shifted_radius)/dr)*dr target (sampled in
    # scroll space, transformed to spiral space upstream); we divide dr_per_winding back out to
    # recover the integer winding index.
    if dt_max_winding is None:
        return None
    winding_idx = (snapped_winding / dr_per_winding).detach()
    return winding_idx <= dt_max_winding



def _sample_patch_tracks(slice_to_spiral_transform, dr_per_winding, patches, patch_atlas, patch_indices, extra_zyxs=None, num_points_per_patch=None):
    if len(patch_indices) == 0:
        raise ValueError('Expected at least one patch index')

    # For each patch, we take one row and one column ('straight' mode; _sample_strip_ijs picks
    # a contiguous subrange of each) or two wiggly geodesic strips between distant points,
    # skirting around gaps/holes ('dijkstra' mode; important for long, ragged traces). Either
    # way each strip is a contiguous walk, so unwrap_shifted_radii can reliably handle
    # theta=0 crossings between consecutive sorted samples.

    if num_points_per_patch is None:
        num_points_per_patch = cfg['num_points_per_patch']
    num_points_per_direction = num_points_per_patch // 2
    N = len(patch_indices)

    use_dijkstra_strips = cfg['patch_strip_sampling'] == 'dijkstra'
    if use_dijkstra_strips:
        touched_patches = [patches[patch_idx] for patch_idx in dict.fromkeys(patch_indices)]
        strip_path_pools.ensure_patch_path_pools(touched_patches)
        # Submitted before the sampling below so the workers refresh while this step proceeds.
        for patch in touched_patches:
            strip_path_pools.submit_patch_pool_refresh(patch)

    P = num_points_per_direction
    horizontal_ijs_by_patch = np.empty([N, P, 2], dtype=np.float32)
    vertical_ijs_by_patch = np.empty([N, P, 2], dtype=np.float32)
    rand = np.random.random
    randint = np.random.randint
    fixed_jitters_h = rand(N).astype(np.float32)
    fixed_jitters_v = rand(N).astype(np.float32)
    var_jitters_h = rand((N, P)).astype(np.float32)
    var_jitters_v = rand((N, P)).astype(np.float32)
    for n, patch_idx in enumerate(patch_indices):
        patch = patches[patch_idx]

        if use_dijkstra_strips:
            # Two independent geodesic strips per patch (no horizontal/vertical distinction;
            # the 'horizontal'/'vertical' arrays are just the two strip slots). Snapshot the
            # pool once: a background refresh may swap the list, but never mutates it.
            pool = patch._strip_path_pool
            path_a, path_b = (pool[k] for k in np.random.choice(len(pool), 2, replace=len(pool) < 2))
            horizontal_ijs_by_patch[n] = _sample_points_along_path(path_a, P)
            vertical_ijs_by_patch[n] = _sample_points_along_path(path_b, P)
            continue

        # Horizontal: pick a row uniformly from rows-with-valid-quads, then pick a run
        # within that row weighted by length (matches original `np.random.choice(flatnonzero)`).
        rows_h = patch._sampling_valid_quad_rows
        k = randint(rows_h.shape[0])
        row_idx = rows_h[k]
        cum_h = patch._h_runs_cum[k]
        total_h = cum_h[-1]
        if cum_h.shape[0] == 1:
            r = 0
        else:
            r = np.searchsorted(cum_h, randint(total_h), side='right')
        lo_h = patch._h_runs_los[k][r]
        hi_h = patch._h_runs_his[k][r]
        run_len_h = hi_h - lo_h
        coords_h = np.sort(np.random.choice(run_len_h, P, replace=P > run_len_h))
        horizontal_ijs_by_patch[n, :, 0] = row_idx + fixed_jitters_h[n]
        horizontal_ijs_by_patch[n, :, 1] = lo_h + coords_h + var_jitters_h[n]

        # Vertical: same but with rows/cols swapped (fixed-coord is the column).
        cols_v = patch._sampling_valid_quad_cols
        k = randint(cols_v.shape[0])
        col_idx = cols_v[k]
        cum_v = patch._v_runs_cum[k]
        total_v = cum_v[-1]
        if cum_v.shape[0] == 1:
            r = 0
        else:
            r = np.searchsorted(cum_v, randint(total_v), side='right')
        lo_v = patch._v_runs_los[k][r]
        hi_v = patch._v_runs_his[k][r]
        run_len_v = hi_v - lo_v
        coords_v = np.sort(np.random.choice(run_len_v, P, replace=P > run_len_v))
        vertical_ijs_by_patch[n, :, 1] = col_idx + fixed_jitters_v[n]
        vertical_ijs_by_patch[n, :, 0] = lo_v + coords_v + var_jitters_v[n]

    # Batched bilinear interp on GPU: ijs are guaranteed to fall on valid quads by the
    # _sample_strip_ijs sampler (it draws i0/j0 from `_sampling_valid_quad_*`), so we
    # skip the per-call validity check used by patch.ij_to_zyx.
    combined_ijs_np = np.stack([horizontal_ijs_by_patch, vertical_ijs_by_patch], axis=0)  # (2, N, P, 2)
    combined_ijs_gpu = torch.from_numpy(combined_ijs_np).cuda(non_blocking=True)
    patch_indices_gpu = torch.from_numpy(np.ascontiguousarray(patch_indices, dtype=np.int64)).cuda(non_blocking=True)
    patch_idx_per_sample = patch_indices_gpu[None, :, None].expand(2, N, num_points_per_direction)
    all_slice_zyxs = patch_atlas.lookup(patch_idx_per_sample, combined_ijs_gpu)

    # When the caller has extra points (umbilicus, shell, ...), pack them into the same
    # forward ODE call to amortise the per-call overhead.
    patches_flat = all_slice_zyxs.reshape(-1, 3)
    if extra_zyxs is not None:
        combined_spiral = slice_to_spiral_transform(torch.cat([patches_flat, extra_zyxs], dim=0))
        n_patch_pts = patches_flat.shape[0]
        all_spiral_zyxs = combined_spiral[:n_patch_pts].reshape(*all_slice_zyxs.shape)
        extra_spiral = combined_spiral[n_patch_pts:]
    else:
        all_spiral_zyxs = slice_to_spiral_transform(patches_flat).reshape(*all_slice_zyxs.shape)
        extra_spiral = None

    all_theta, _, all_shifted_radii = get_theta_and_radii(all_spiral_zyxs[..., 1:], dr_per_winding)
    all_shifted_radii, all_crossing_adjustments = unwrap_shifted_radii(
        all_theta, all_shifted_radii, dr_per_winding,
    )

    return (
        all_slice_zyxs,
        all_spiral_zyxs,
        all_theta,
        all_shifted_radii,
        all_crossing_adjustments,
        extra_spiral,
    )



def _patch_radius_and_dt_losses(
    slice_to_spiral_transform, dr_per_winding,
    all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii, all_crossing_adjustments,
    num_patches_for_radius, num_patches_for_dt, compute_dt, dt_max_winding,
    radius_loss_margin, radius_loss_inv, radius_within_norm_p,
    dt_loss_margin, dt_norm_p, dt_within_patch_norm_p,
):
    # Shared radius + DT patch losses, operating on pre-sampled row/column tracks
    # (all_*; see _sample_patch_tracks). Pulled out of get_patch_and_umbilicus_losses so the
    # same loss can serve both the verified and the untrusted ('unverified') patch sets with
    # independent hyperparameters. Returns (mean_radius_deviation, patch_dt_loss).
    radius_hinge_margin = dr_per_winding.detach() * radius_loss_margin
    dt_hinge_margin = dr_per_winding.detach() * dt_loss_margin

    # Each patch row/col should lie at constant shifted-radius.
    radius_shifted_radii = all_shifted_radii[:, :num_patches_for_radius]
    if radius_loss_inv:
        # Express the loss in scroll space like the DT loss below: construct target
        # spiral-space points at the track's mean shifted-radius (continuous, not snapped
        # to an integer winding) but with each point's own z and theta, transform back to
        # scroll space, and penalise the distance from the original sampled points.
        radius_slice_zyxs = all_slice_zyxs[:, :num_patches_for_radius]
        radius_spiral_zyxs = all_spiral_zyxs[:, :num_patches_for_radius]
        radius_theta = all_theta[:, :num_patches_for_radius]
        radius_crossing_adjustments = all_crossing_adjustments[:, :num_patches_for_radius]

        mean_shifted_radii = radius_shifted_radii.mean(dim=-1, keepdim=True)
        radius_target_radii = radius_from_unwrapped_shifted(
            radius_theta, mean_shifted_radii, radius_crossing_adjustments, dr_per_winding,
        )
        radius_target_spiral_zyxs = torch.stack([
            radius_spiral_zyxs[..., 0],
            torch.sin(radius_theta) * radius_target_radii,
            torch.cos(radius_theta) * radius_target_radii,
        ], dim=-1).detach()

        radius_target_scroll_zyxs = slice_to_spiral_transform.inv(radius_target_spiral_zyxs.reshape(-1, 3)).reshape(*radius_target_spiral_zyxs.shape)

        radius_point_distances = torch.linalg.norm(radius_slice_zyxs - radius_target_scroll_zyxs, dim=-1)
        mean_radius_deviation = F.relu(radius_point_distances - radius_hinge_margin).mean()
    else:
        # Penalise deviation from the track's mean shifted-radius directly in spiral space.
        mean_radii = radius_shifted_radii.mean(dim=-1, keepdim=True)
        radius_deviations = (radius_shifted_radii - mean_radii).abs()
        radius_deviations_hinge = F.relu(radius_deviations - radius_hinge_margin)
        if radius_within_norm_p == 1.0:
            mean_radius_deviation = radius_deviations_hinge.mean()
        else:
            d = radius_deviations_hinge + 1.e-5
            per_track = (d ** radius_within_norm_p).mean(dim=-1) ** (1.0 / radius_within_norm_p)
            mean_radius_deviation = per_track.mean()

    if compute_dt:
        dt_slice_zyxs = all_slice_zyxs[:, :num_patches_for_dt]
        dt_spiral_zyxs = all_spiral_zyxs[:, :num_patches_for_dt]
        dt_theta = all_theta[:, :num_patches_for_dt]
        dt_shifted_radii = all_shifted_radii[:, :num_patches_for_dt]
        dt_crossing_adjustments = all_crossing_adjustments[:, :num_patches_for_dt]

        # Define the DT target from the same sampled row/column tracks as the radius loss:
        # each track is snapped to the nearest integer-winding shifted-radius, then every
        # sampled point on the track is pulled towards the corresponding point on that
        # target winding.
        target_shifted_radii = torch.round(dt_shifted_radii.median(dim=-1, keepdim=True).values / dr_per_winding) * dr_per_winding
        target_radii = radius_from_unwrapped_shifted(
            dt_theta, target_shifted_radii, dt_crossing_adjustments, dr_per_winding,
        )
        target_spiral_zyxs = torch.stack([
            dt_spiral_zyxs[..., 0],
            torch.sin(dt_theta) * target_radii,
            torch.cos(dt_theta) * target_radii,
        ], dim=-1).detach()

        target_scroll_zyxs = slice_to_spiral_transform.inv(target_spiral_zyxs.reshape(-1, 3)).reshape(*target_spiral_zyxs.shape)

        point_distances = torch.linalg.norm(dt_slice_zyxs - target_scroll_zyxs, dim=-1)
        point_distances = F.relu(point_distances - dt_hinge_margin) + 1.e-5  # epsilon to avoid NaN in p-norm backward
        track_losses = (point_distances ** dt_within_patch_norm_p).mean(dim=-1) ** (1 / dt_within_patch_norm_p)
        # Progressive DT: only patches whose snapped winding is within the current cutoff contribute.
        active_mask = _progressive_dt_active_mask(target_shifted_radii.squeeze(-1), dr_per_winding, dt_max_winding)
        patch_dt_loss = _aggregate_dt_track_losses(track_losses, dt_norm_p, active_mask)
    else:
        patch_dt_loss = torch.zeros([], device=dr_per_winding.device)

    return mean_radius_deviation, patch_dt_loss



def get_patch_and_umbilicus_losses(slice_to_spiral_transform, dr_per_winding, num_patches_for_radius, num_patches_for_dt, patches, patch_atlas, patch_sampling_probabilities, umbilicus_zyx, compute_dt=True, shell_valid_zyxs=None, shell_outer_winding_idx=None, dt_max_winding=None):

    # Sample once and share the tracks between the radius and DT losses; the loss using
    # fewer patches takes a prefix of the larger sample.
    num_patches_to_sample = max(num_patches_for_radius, num_patches_for_dt) if compute_dt else num_patches_for_radius
    patch_indices = np.random.choice(len(patches), num_patches_to_sample, p=patch_sampling_probabilities, replace=True)

    n_umb = umbilicus_zyx.shape[0]
    if shell_valid_zyxs is not None:
        num_shell_samples = min(int(cfg['shell_num_samples']), shell_valid_zyxs.shape[0])
        sample_idx = torch.randint(shell_valid_zyxs.shape[0], (num_shell_samples,), device=shell_valid_zyxs.device)
        extra_zyxs = torch.cat([umbilicus_zyx, shell_valid_zyxs[sample_idx]], dim=0)
    else:
        extra_zyxs = umbilicus_zyx

    (
        all_slice_zyxs,
        all_spiral_zyxs,
        all_theta,
        all_shifted_radii,
        all_crossing_adjustments,
        extra_spiral,
    ) = _sample_patch_tracks(
        slice_to_spiral_transform,
        dr_per_winding,
        patches,
        patch_atlas,
        patch_indices,
        extra_zyxs,
    )
    umbilicus_spiral = extra_spiral[:n_umb]
    shell_spiral_zyxs = extra_spiral[n_umb:] if shell_valid_zyxs is not None else None

    mean_radius_deviation, patch_dt_loss = _patch_radius_and_dt_losses(
        slice_to_spiral_transform, dr_per_winding,
        all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii, all_crossing_adjustments,
        num_patches_for_radius, num_patches_for_dt, compute_dt, dt_max_winding,
        cfg['patch_radius_loss_margin'], cfg['patch_radius_loss_inv'], cfg['patch_radius_within_norm_p'],
        cfg['patch_dt_loss_margin'], cfg['patch_dt_norm_p'], cfg['patch_dt_within_patch_norm_p'],
    )

    # Umbilicus should map to the spiral origin (yx ≈ 0)
    umbilicus_loss = umbilicus_spiral[..., 1:].abs().mean()

    if shell_spiral_zyxs is not None:
        radius_hinge_margin = dr_per_winding.detach() * cfg['patch_radius_loss_margin']
        _, _, shell_shifted_radii = get_theta_and_radii(shell_spiral_zyxs[..., 1:], dr_per_winding)
        shell_target = dr_per_winding * float(shell_outer_winding_idx)
        shell_patch_radius_loss = F.relu((shell_shifted_radii - shell_target).abs() - radius_hinge_margin).mean()
    else:
        shell_patch_radius_loss = torch.zeros([], device=dr_per_winding.device)

    return mean_radius_deviation, umbilicus_loss, patch_dt_loss, shell_patch_radius_loss



def get_unverified_patch_losses(slice_to_spiral_transform, dr_per_winding, num_patches_for_radius, num_patches_for_dt, patches, patch_atlas, patch_sampling_probabilities, compute_dt=True, dt_max_winding=None):
    # Radius + DT losses for the untrusted 'unverified' patch set. Same machinery as the
    # verified patches (shared _sample_patch_tracks + _patch_radius_and_dt_losses) but with the
    # independent unverified_* hyperparameters and no umbilicus/shell extras. These patches are
    # masked away near trusted geometry upstream (see _mask_patches_near_trusted_geometry), so
    # they only constrain regions the verified inputs don't cover.
    num_patches_to_sample = max(num_patches_for_radius, num_patches_for_dt) if compute_dt else num_patches_for_radius
    patch_indices = np.random.choice(len(patches), num_patches_to_sample, p=patch_sampling_probabilities, replace=True)

    (
        all_slice_zyxs,
        all_spiral_zyxs,
        all_theta,
        all_shifted_radii,
        all_crossing_adjustments,
        _,
    ) = _sample_patch_tracks(
        slice_to_spiral_transform,
        dr_per_winding,
        patches,
        patch_atlas,
        patch_indices,
        num_points_per_patch=cfg['unverified_num_points_per_patch'],
    )

    return _patch_radius_and_dt_losses(
        slice_to_spiral_transform, dr_per_winding,
        all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii, all_crossing_adjustments,
        num_patches_for_radius, num_patches_for_dt, compute_dt, dt_max_winding,
        cfg['unverified_patch_radius_loss_margin'], cfg['unverified_patch_radius_loss_inv'], cfg['unverified_patch_radius_within_norm_p'],
        cfg['unverified_patch_dt_loss_margin'], cfg['unverified_patch_dt_norm_p'], cfg['unverified_patch_dt_within_patch_norm_p'],
    )



def _sample_single_l_shape(valid_quad, i_q, j_q, leg1_axis, leg1_dir, leg2_dir, num_points):
    # Sample a single L-shape on `valid_quad` starting at (i_q, j_q). Leg 1 walks along
    # `leg1_axis` (0 -> varying j, 1 -> varying i) in direction `leg1_dir` (+1 or -1) to a
    # uniformly random turn point inside the contiguous valid run. Leg 2 walks from the
    # turn point along the perpendicular axis in direction `leg2_dir` (+1 or -1) to the end
    # of its valid run. Returns a float32 [num_points, 2] sampled in traversal order, with
    # subpixel jitter; the fixed-axis jitter is shared within each leg (matching the
    # _sample_strip_ijs convention), so the unwrap can stitch theta=0 crossings along the
    # full L (the only ~sqrt(2)-quad jump is across the corner, still well within the
    # |dtheta| < pi requirement). Caller guarantees valid_quad[i_q, j_q].

    if leg1_axis == 0:
        line1_valid = valid_quad[i_q, :]
        var_start1 = j_q
    else:
        line1_valid = valid_quad[:, j_q]
        var_start1 = i_q
    lo1, hi1 = run_containing_index(line1_valid, var_start1)
    var_far1 = (hi1 - 1) if leg1_dir > 0 else lo1
    leg1_max_steps = abs(var_far1 - var_start1)
    turn_step = int(np.random.randint(0, leg1_max_steps + 1))
    var_turn = var_start1 + leg1_dir * turn_step

    if leg1_axis == 0:
        i_turn, j_turn = i_q, var_turn
    else:
        i_turn, j_turn = var_turn, j_q

    leg2_axis = 1 - leg1_axis
    if leg2_axis == 0:
        line2_valid = valid_quad[i_turn, :]
        var_start2 = j_turn
    else:
        line2_valid = valid_quad[:, j_turn]
        var_start2 = i_turn
    lo2, hi2 = run_containing_index(line2_valid, var_start2)
    var_far2 = (hi2 - 1) if leg2_dir > 0 else lo2
    leg2_max_steps = abs(var_far2 - var_start2)

    total_steps = turn_step + leg2_max_steps  # leg 1 spans [0, turn_step]; leg 2 spans (turn_step, total_steps]
    num_positions = total_steps + 1
    steps = np.sort(np.random.choice(num_positions, num_points, replace=num_points > num_positions))

    ijs = np.empty([num_points, 2], dtype=np.float32)
    leg1_fixed_jitter = float(np.random.uniform(0, 1))
    leg2_fixed_jitter = float(np.random.uniform(0, 1))

    on_leg1 = steps <= turn_step
    leg1_steps = steps[on_leg1]
    leg2_steps = steps[~on_leg1] - turn_step

    leg1_var = (var_start1 + leg1_dir * leg1_steps).astype(np.float32) + np.random.uniform(0., 1., size=leg1_steps.shape).astype(np.float32)
    leg1_fixed = float(i_q if leg1_axis == 0 else j_q) + leg1_fixed_jitter
    if leg1_axis == 0:
        ijs[on_leg1, 0] = leg1_fixed
        ijs[on_leg1, 1] = leg1_var
    else:
        ijs[on_leg1, 0] = leg1_var
        ijs[on_leg1, 1] = leg1_fixed

    leg2_var = (var_start2 + leg2_dir * leg2_steps).astype(np.float32) + np.random.uniform(0., 1., size=leg2_steps.shape).astype(np.float32)
    leg2_fixed = float(i_turn if leg2_axis == 0 else j_turn) + leg2_fixed_jitter
    if leg2_axis == 0:
        ijs[~on_leg1, 0] = leg2_fixed
        ijs[~on_leg1, 1] = leg2_var
    else:
        ijs[~on_leg1, 0] = leg2_var
        ijs[~on_leg1, 1] = leg2_fixed

    return ijs



def _sample_l_shapes_at_ij(patch, i, j, num_points):
    # Sample 4 strips anchored on the annotated point (i, j) of `patch`, one per cardinal
    # primary direction. In 'dijkstra' mode these are geodesic strips to distant endpoints
    # (one per cardinal cone; see _sample_dijkstra_strips_at_ij); otherwise L-shapes, one per
    # primary direction: right (+j), left (-j), down (+i), up (-i). For each L, leg 2's
    # perpendicular direction is chosen uniformly at random. Returns a list of 4 float32
    # [num_points, 2] arrays sampled in traversal order, or None if (i, j) doesn't lie on
    # a valid quad (or, in dijkstra mode, while this anchor's path pools are still being
    # built in the background). Each L is a single contiguous walk in patch space, so the unwrap can
    # handle theta=0 seam crossings along the bent strip just as it does along a straight
    # row/column.
    valid_quad = patch._sampling_valid_quad_mask_np
    H_q, W_q = valid_quad.shape
    i_q = min(max(int(i), 0), H_q - 1)
    j_q = min(max(int(j), 0), W_q - 1)
    if not valid_quad[i_q, j_q]:
        return None

    if cfg['patch_strip_sampling'] == 'dijkstra':
        return _sample_dijkstra_strips_at_ij(patch, i_q, j_q, num_points)

    primary_specs = [(0, +1), (0, -1), (1, +1), (1, -1)]  # (leg1_axis, leg1_dir)
    return [
        _sample_single_l_shape(
            valid_quad, i_q, j_q, leg1_axis, leg1_dir,
            leg2_dir=int(np.random.choice([-1, +1])),
            num_points=num_points,
        )
        for leg1_axis, leg1_dir in primary_specs
    ]



def get_patch_rel_winding_loss(slice_to_spiral_transform, dr_per_winding, patches_dict, patch_atlas, point_collections):
    # For pairs of annotated PCL points on different patches, constrain the spiral
    # shifted-radius gap to match the annotated winding-number difference. Each
    # cross-patch pcl exposes its attached points grouped by patch
    # (pcl['points_by_patch']); we form the set of all pairs (p1, p2) whose patches
    # differ and sample uniformly from it. For each annotated point we build 4
    # L-shaped strips: from (i, j), walk along one of the cardinal patch directions
    # (right, left, down, up) to a uniformly-random turn point inside the contiguous
    # valid run, then 90-degree-turn into a uniformly-random perpendicular direction
    # and walk to the end of that valid run. Each L is sampled in traversal order
    # along its bent path, so unwrap_shifted_radii can stitch theta=0 seam
    # crossings along the whole strip (the corner only introduces a ~sqrt(2)-quad ij
    # jump). We then pool all 4 L-strips per annotated point into one set of sample
    # points and take a single all-pairs diff between p1's and p2's pooled sets,
    # regressing it onto winding_diff * dr_per_winding. If the selected PCL
    # points (adjacent mode) or the PCL chain between them (non-adjacent mode)
    # crosses theta=0, adjust the expected delta by that branch-cut jump.

    num_points_per_strip = cfg['num_points_per_patch'] // 2
    num_strips_per_pcl = 4
    num_strips_per_pair = 2 * num_strips_per_pcl  # 8

    # Each entry: (ls1, ls2, pid1, pid2, winding_diff, pcl_chain_zyxs), where
    # ls* is a list of 4 L-shape ij strips and pcl_chain_zyxs is ordered p1 -> p2.
    strip_pairs = []

    # Single-point pcls (possible only for winding_is_absolute pcls) can't form a
    # cross-patch pair, so exclude them from the candidate pool before sampling.
    candidate_pcls = [pcl for pcl in point_collections if len(pcl['points']) > 1]
    num_pcls_per_step = min(cfg['rel_winding_num_pcls'], len(candidate_pcls))
    if num_pcls_per_step <= 0:
        return torch.zeros([], device='cuda')
    selected_idxs = np.random.choice(len(candidate_pcls), num_pcls_per_step, replace=False)
    selected_pcls = [candidate_pcls[i] for i in selected_idxs]

    for pcl in selected_pcls:
        sorted_pcl_points = None
        sorted_pcl_point_idx = None
        if not cfg['rel_winding_adjacent_patches_only']:
            sorted_pcl_points = [
                point for _, point in sorted(pcl['points'].items(), key=lambda kv: int(kv[0]))
            ]
            sorted_pcl_point_idx = {id(point): idx for idx, point in enumerate(sorted_pcl_points)}

        # Pair patches either only with their immediate neighbour in the pcl's
        # patch ordering (first-seen order; built in main()),
        # or with every other patch.
        if cfg['rel_winding_adjacent_patches_only']:
            cross_pairs = [(p1, p2) for p1, p2 in zip(pcl['points_by_patch'], list(pcl['points_by_patch'])[1:])]
        else:
            cross_pairs = list(itertools.combinations(pcl['points_by_patch'], r=2))
        if not cross_pairs:
            continue

        num_pairs_for_pcl = min(len(cross_pairs), cfg['rel_winding_num_patch_pairs_per_pcl'])
        if num_pairs_for_pcl <= 0:
            continue
        chosen = np.random.choice(len(cross_pairs), num_pairs_for_pcl, replace=False)
        pid_pairs = [cross_pairs[i] for i in chosen]

        for pid1, pid2 in pid_pairs:
            points1 = pcl['points_by_patch'][pid1]
            points2 = pcl['points_by_patch'][pid2]
            p1 = points1[np.random.randint(len(points1))]
            p2 = points2[np.random.randint(len(points2))]
            winding_diff = p2['winding_annotation'] - p1['winding_annotation']
            i1, j1 = int(p1['on_patch']['ij'][0]), int(p1['on_patch']['ij'][1])
            i2, j2 = int(p2['on_patch']['ij'][0]), int(p2['on_patch']['ij'][1])

            ls1 = _sample_l_shapes_at_ij(patches_dict[pid1], i1, j1, num_points_per_strip)
            ls2 = _sample_l_shapes_at_ij(patches_dict[pid2], i2, j2, num_points_per_strip)
            if ls1 is None or ls2 is None:
                continue

            if cfg['rel_winding_adjacent_patches_only']:
                pcl_chain = [p1, p2]
            else:
                idx1, idx2 = sorted_pcl_point_idx[id(p1)], sorted_pcl_point_idx[id(p2)]
                if idx1 <= idx2:
                    pcl_chain = sorted_pcl_points[idx1:idx2 + 1]
                else:
                    pcl_chain = list(reversed(sorted_pcl_points[idx2:idx1 + 1]))
            pcl_chain_zyxs = np.stack([point['zyx'] for point in pcl_chain], axis=0).astype(np.float32)
            strip_pairs.append((ls1, ls2, pid1, pid2, winding_diff, pcl_chain_zyxs))

    if not strip_pairs:
        return torch.zeros([], device='cuda')

    # Flatten: 8 strips per pair, ordered as p1's 4 strips followed by p2's 4 strips.
    total_strips = len(strip_pairs) * num_strips_per_pair
    flat_ijs = np.empty([total_strips, num_points_per_strip, 2], dtype=np.float32)
    flat_pids = []
    for k, (ls1, ls2, pid1, pid2, _, _) in enumerate(strip_pairs):
        base = k * num_strips_per_pair
        for s, strip in enumerate(ls1):
            flat_ijs[base + s] = strip
        for s, strip in enumerate(ls2):
            flat_ijs[base + num_strips_per_pcl + s] = strip
        flat_pids.extend([pid1] * num_strips_per_pcl + [pid2] * num_strips_per_pcl)

    # Batched GPU bilinear interp across all strips.
    patch_idx_per_strip_np = np.fromiter(
        (patch_atlas.id_to_idx[pid] for pid in flat_pids),
        dtype=np.int64,
        count=total_strips,
    )
    patch_idx_per_strip_gpu = torch.from_numpy(patch_idx_per_strip_np).cuda(non_blocking=True)
    ijs_gpu = torch.from_numpy(flat_ijs).cuda(non_blocking=True)
    patch_idx_per_sample = patch_idx_per_strip_gpu[:, None].expand(total_strips, num_points_per_strip)
    flat_zyxs = patch_atlas.lookup(patch_idx_per_sample, ijs_gpu)

    # Mask out strip samples whose z falls outside [z_begin - margin, z_end + margin).
    # Computed before unwrapping but applied after, since unwrap_shifted_radii
    # needs the full sequential strip to stitch theta=0 crossings.
    z_margin = cfg['patch_loss_z_margin']
    z_mask = (flat_zyxs[..., 0] >= z_begin - z_margin) & (flat_zyxs[..., 0] < z_end + z_margin)

    flat_spiral = slice_to_spiral_transform(flat_zyxs.reshape(-1, 3)).reshape(*flat_zyxs.shape)
    theta, _, shifted_radii = get_theta_and_radii(flat_spiral[..., 1:], dr_per_winding)
    shifted_radii, _ = unwrap_shifted_radii(theta, shifted_radii, dr_per_winding)

    # [num_pairs, 8, num_points_per_strip] -> pool each side's 4 strips into a single set.
    shifted_radii = shifted_radii.reshape(len(strip_pairs), num_strips_per_pair, num_points_per_strip)
    z_mask = z_mask.reshape(len(strip_pairs), num_strips_per_pair, num_points_per_strip)
    num_points_per_side = num_strips_per_pcl * num_points_per_strip
    p1_r = shifted_radii[:, :num_strips_per_pcl].reshape(len(strip_pairs), num_points_per_side)
    p2_r = shifted_radii[:, num_strips_per_pcl:].reshape(len(strip_pairs), num_points_per_side)
    m1 = z_mask[:, :num_strips_per_pcl].reshape(len(strip_pairs), num_points_per_side)
    m2 = z_mask[:, num_strips_per_pcl:].reshape(len(strip_pairs), num_points_per_side)

    winding_diffs = torch.tensor(
        [sp[4] for sp in strip_pairs],
        device='cuda',
        dtype=torch.float32,
    )
    pcl_seam_adjustments = []
    for _, _, _, _, _, pcl_chain_zyxs in strip_pairs:
        chain_zyxs = torch.from_numpy(pcl_chain_zyxs).cuda(non_blocking=True)
        chain_spiral = slice_to_spiral_transform(chain_zyxs)
        chain_theta, _, _ = get_theta_and_radii(chain_spiral[..., 1:], dr_per_winding)
        zero_shifted = torch.zeros_like(chain_theta)
        _, chain_adjustments = unwrap_shifted_radii(chain_theta, zero_shifted, dr_per_winding)
        pcl_seam_adjustments.append(chain_adjustments[-1])
    pcl_seam_adjustments = torch.stack(pcl_seam_adjustments)
    expected_diff = ((winding_diffs * dr_per_winding) - pcl_seam_adjustments)[:, None, None]

    diff = p2_r[:, :, None] - p1_r[:, None, :]
    pair_mask = m2[:, :, None] & m1[:, None, :]
    err = (diff - expected_diff).abs()
    return (err * pair_mask).sum() / pair_mask.sum().clamp(min=1)



def get_patch_abs_winding_loss(slice_to_spiral_transform, dr_per_winding, patches_dict, patch_atlas, point_collections):
    # For PCL points carrying an absolute winding annotation (only pcls flagged
    # metadata.winding_is_absolute), pin the spiral shifted-radius at each annotated
    # point to its absolute target, winding_annotation * dr_per_winding (the spiral has
    # radius 0 at winding 0 and grows at dr_per_winding, so shifted_radius == winding *
    # dr_per_winding). This mirrors get_patch_rel_winding_loss, but anchors each point's
    # absolute winding instead of regressing a pair's winding difference: we sample some
    # absolute-winding pcls, some attached points within each, build 4 L-shaped strips
    # per point (sampled in traversal order so unwrap_shifted_radii can stitch
    # theta=0 seam crossings), then drive every in-roi strip sample's shifted radius to
    # the point's target. Each L starts at the annotated point, so its unwrapped
    # shifted-radius keeps the true absolute scale at the anchor.

    num_points_per_strip = cfg['num_points_per_patch'] // 2
    num_strips_per_point = 4

    # Each entry: (ls, pid, winding_annotation) where ls is a list of 4 L-shape ij strips.
    strips = []

    abs_pcls = [pcl for pcl in point_collections if pcl.get('metadata', {}).get('winding_is_absolute', False)]
    num_pcls_per_step = min(cfg['abs_winding_num_pcls'], len(abs_pcls))
    if num_pcls_per_step <= 0:
        return torch.zeros([], device='cuda')
    selected_idxs = np.random.choice(len(abs_pcls), num_pcls_per_step, replace=False)
    selected_pcls = [abs_pcls[i] for i in selected_idxs]

    for pcl in selected_pcls:
        # An absolute-winding pcl's attached points, flattened across its patches.
        attached = [p for pts in pcl['points_by_patch'].values() for p in pts]
        if not attached:
            continue
        num_points_for_pcl = min(len(attached), cfg['abs_winding_num_points_per_pcl'])
        chosen = np.random.choice(len(attached), num_points_for_pcl, replace=False)
        for idx in chosen:
            p = attached[idx]
            pid = p['on_patch']['id']
            i, j = int(p['on_patch']['ij'][0]), int(p['on_patch']['ij'][1])
            ls = _sample_l_shapes_at_ij(patches_dict[pid], i, j, num_points_per_strip)
            if ls is None:
                continue
            strips.append((ls, pid, p['winding_annotation']))

    if not strips:
        return torch.zeros([], device='cuda')

    # Flatten: 4 strips per annotated point.
    total_strips = len(strips) * num_strips_per_point
    flat_ijs = np.empty([total_strips, num_points_per_strip, 2], dtype=np.float32)
    flat_pids = []
    for k, (ls, pid, _) in enumerate(strips):
        base = k * num_strips_per_point
        for s, strip in enumerate(ls):
            flat_ijs[base + s] = strip
        flat_pids.extend([pid] * num_strips_per_point)

    # Batched GPU bilinear interp across all strips.
    patch_idx_per_strip_np = np.fromiter(
        (patch_atlas.id_to_idx[pid] for pid in flat_pids),
        dtype=np.int64,
        count=total_strips,
    )
    patch_idx_per_strip_gpu = torch.from_numpy(patch_idx_per_strip_np).cuda(non_blocking=True)
    ijs_gpu = torch.from_numpy(flat_ijs).cuda(non_blocking=True)
    patch_idx_per_sample = patch_idx_per_strip_gpu[:, None].expand(total_strips, num_points_per_strip)
    flat_zyxs = patch_atlas.lookup(patch_idx_per_sample, ijs_gpu)

    # Mask out strip samples whose z falls outside [z_begin - margin, z_end + margin).
    # Computed before unwrapping but applied after, since unwrap_shifted_radii
    # needs the full sequential strip to stitch theta=0 crossings.
    z_margin = cfg['patch_loss_z_margin']
    z_mask = (flat_zyxs[..., 0] >= z_begin - z_margin) & (flat_zyxs[..., 0] < z_end + z_margin)

    flat_spiral = slice_to_spiral_transform(flat_zyxs.reshape(-1, 3)).reshape(*flat_zyxs.shape)
    theta, _, shifted_radii = get_theta_and_radii(flat_spiral[..., 1:], dr_per_winding)
    shifted_radii, _ = unwrap_shifted_radii(theta, shifted_radii, dr_per_winding)

    # [num_points, 4, num_points_per_strip] -> pool each point's 4 strips into one set.
    num_samples_per_point = num_strips_per_point * num_points_per_strip
    shifted_radii = shifted_radii.reshape(len(strips), num_samples_per_point)
    mask = z_mask.reshape(len(strips), num_samples_per_point)

    winding_annotations = torch.tensor(
        [s[2] for s in strips],
        device='cuda',
        dtype=torch.float32,
    )
    target_shifted = (winding_annotations * dr_per_winding)[:, None]

    err = (shifted_radii - target_shifted).abs()
    return (err * mask).sum() / mask.sum().clamp(min=1)



def _decode_uint8_normal_component(value):
    return (value - 128.0) / 127.0



def get_radial_normal_in_scroll_space(slice_to_spiral_transform, scroll_zyx, spiral_zyx=None, epsilon=6.0):
    # At each scroll-space point, pull the spiral-space cylinder normal (the outward radial
    # direction normalize(spiral_yx)) back to scroll space as a covector, J^T n_spiral, where
    # J = d(spiral) / d(scroll) is estimated by central differences. This is the geometrically
    # correct transport of a surface normal (covector) -- unlike a tangent-vector pushforward J n.
    # Returns the normalised scroll-space normal direction (num_points, 3) in zyx.
    #
    # Gradient flows through the transform parameters via the Jacobian only; the sample positions
    # (scroll_zyx) and the radial direction are held fixed, matching the dense-normals loss. If the
    # forward image spiral_zyx is supplied it is reused for the radial direction (and treated as a
    # constant); otherwise it is computed here from scroll_zyx.
    device = scroll_zyx.device
    num_points = scroll_zyx.shape[0]
    scroll_zyx = scroll_zyx.detach()

    basis_zyx = torch.eye(3, device=device, dtype=scroll_zyx.dtype) * epsilon
    scroll_plus = (scroll_zyx[None, :, :] + basis_zyx[:, None, :]).reshape(-1, 3)
    scroll_minus = (scroll_zyx[None, :, :] - basis_zyx[:, None, :]).reshape(-1, 3)
    if spiral_zyx is None:
        combined_spiral = slice_to_spiral_transform(torch.cat([scroll_zyx, scroll_plus, scroll_minus], dim=0))
        spiral_zyx = combined_spiral[:num_points]
        spiral_plus, spiral_minus = combined_spiral[num_points:].chunk(2, dim=0)
    else:
        spiral_plus, spiral_minus = slice_to_spiral_transform(torch.cat([scroll_plus, scroll_minus], dim=0)).chunk(2, dim=0)

    spiral_outward_yx = F.normalize(spiral_zyx[:, 1:].detach(), dim=-1)
    spiral_outward_zyx = torch.cat([torch.zeros_like(spiral_outward_yx[:, :1]), spiral_outward_yx], dim=-1)

    spiral_plus = spiral_plus.view(3, num_points, 3)
    spiral_minus = spiral_minus.view(3, num_points, 3)
    jacobian_columns = (spiral_plus - spiral_minus) / (2.0 * epsilon)  # scroll basis axis, point, spiral zyx
    return F.normalize((jacobian_columns * spiral_outward_zyx[None, :, :]).sum(dim=-1).transpose(0, 1), dim=-1)



def sample_spiral_surface_frame(dr_per_winding, outer_winding_idx, num_points):
    # Sample points from discrete spiral windings embedded in spiral yx (over the z-ROI) and return
    # each point's orthonormal in-surface frame in spiral space: e1 = z-axis, e2 = the winding tangent.
    # Winding indices are sampled with probability proportional to their approximate circumference,
    # which is the simple large-radius approximation to uniform area over the wound surface. The inner
    # core is excluded because there is no scroll surface there.
    # Returns (spiral_zyx, e1, e2), each (num_points, 3) in zyx.
    device = dr_per_winding.device
    winding_weights = torch.arange(1, int(outer_winding_idx), device=device, dtype=dr_per_winding.dtype) + 0.5
    winding_idx = torch.multinomial(winding_weights, num_points, replacement=True).to(dr_per_winding.dtype) + 1.0
    theta = torch.rand([num_points], device=device) * (2 * torch.pi)
    radius = (winding_idx + theta / (2 * torch.pi)) * dr_per_winding.detach()
    z = torch.empty([num_points], device=device).uniform_(float(z_begin), float(z_end - 1))
    spiral_zyx = torch.stack([z, torch.sin(theta) * radius, torch.cos(theta) * radius], dim=-1)

    dr_dtheta = dr_per_winding.detach() / (2 * torch.pi)
    tangent_y = torch.cos(theta) * radius + torch.sin(theta) * dr_dtheta
    tangent_x = -torch.sin(theta) * radius + torch.cos(theta) * dr_dtheta
    tangential_yx = F.normalize(torch.stack([tangent_y, tangent_x], dim=-1), dim=-1)
    e1 = F.pad(torch.zeros_like(tangential_yx), (1, 0), value=1.)  # (1, 0, 0) -> z-axis
    e2 = F.pad(tangential_yx, (1, 0), value=0.)  # (0, ty, tx)
    return spiral_zyx, e1, e2



def get_lasagna_losses(slice_to_spiral_transform, dr_per_winding, lasagna_volume, outer_winding_idx, num_points, epsilon=None):
    # Sample points uniformly over the spiral cylinder (a disk of radius
    # dr_per_winding * outer_winding_idx in spiral yx, over the z-ROI). Two losses are computed:
    #   (normals) the spiral radial covector at each sample is pulled back to scroll space via
    #             central-difference J^T (a normal is a covector, not a finite-length displacement)
    #             and matched in direction to the precomputed nx/ny scroll-space normal.
    #   (spacing) at each sample, shift inward and outward by dr_per_winding/2 along the spiral
    #             radial direction (so the two endpoints span exactly one winding in spiral
    #             space), map both endpoints to scroll space, and integrate the winding-density
    #             field (grad_mag, windings per voxel) along the scroll-space segment between
    #             them. grad_mag is a density, not a distance, so the number of windings the
    #             segment actually crosses is the line integral of that density along it; for a
    #             correct fit the integral equals 1 (one winding). The density is decoded from
    #             grad_mag in windings per full-resolution voxel.
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if lasagna_volume is None or outer_winding_idx is None:
        return zero, zero

    volume = lasagna_volume['volume']  # 3 (nx, ny, grad_mag), z, y, x  uint8
    z_size, y_size, x_size = lasagna_volume['shape']
    z_origin = lasagna_volume['z_origin']
    lasagna_scale = lasagna_volume['lasagna_scale']
    if epsilon is None:
        epsilon = cfg['dense_normals_finite_difference_epsilon']

    dr = dr_per_winding.detach()
    r_max = dr * float(outer_winding_idx)
    r_min = dr  # inner endpoint sits at radius - dr/2 >= dr/2 > 0
    theta = torch.rand([num_points], device=device) * (2 * torch.pi)
    radius = torch.sqrt(torch.rand([num_points], device=device) * (r_max ** 2 - r_min ** 2) + r_min ** 2)
    z = torch.empty([num_points], device=device).uniform_(float(z_begin), float(z_end - 1))
    sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
    spiral_zyx = torch.stack([z, sin_theta * radius, cos_theta * radius], dim=-1)
    radius_inner = radius - dr / 2
    radius_outer = radius + dr / 2
    spiral_inner = torch.stack([z, sin_theta * radius_inner, cos_theta * radius_inner], dim=-1)
    spiral_outer = torch.stack([z, sin_theta * radius_outer, cos_theta * radius_outer], dim=-1)

    scroll_samples = slice_to_spiral_transform.inv(torch.cat([spiral_inner, spiral_outer, spiral_zyx], dim=0))
    scroll_inner, scroll_outer, scroll_center = scroll_samples.chunk(3, dim=0)
    scroll_displacement = scroll_outer - scroll_inner  # spans exactly one winding in spiral space
    scroll_segment_length = torch.linalg.norm(scroll_displacement, dim=-1).clamp(min=1.e-8)

    # Look up the precomputed scroll-space targets at the midpoint of the displacement (the
    # geometric centre of the one-winding step in scroll space).
    scroll_mid = ((scroll_inner + scroll_outer) / 2).detach()
    sample_zyx = (scroll_mid / lasagna_scale).round().long()
    zi = sample_zyx[:, 0] - z_origin
    yi = sample_zyx[:, 1]
    xi = sample_zyx[:, 2]
    in_bounds = (zi >= 0) & (zi < z_size) & (yi >= 0) & (yi < y_size) & (xi >= 0) & (xi < x_size)
    zi = zi.clamp(0, z_size - 1)
    yi = yi.clamp(0, y_size - 1)
    xi = xi.clamp(0, x_size - 1)
    nx_u8 = volume[0, zi, yi, xi]
    ny_u8 = volume[1, zi, yi, xi]
    normal_weight = (((nx_u8 != 0) | (ny_u8 != 0)) & in_bounds).float()
    nx = _decode_uint8_normal_component(nx_u8.float())
    ny = _decode_uint8_normal_component(ny_u8.float())
    nz = torch.sqrt((1. - nx * nx - ny * ny).clamp(min=0.))
    target_normal = F.normalize(torch.stack([nz, ny, nx], dim=-1), dim=-1)  # zyx

    scroll_normal = get_radial_normal_in_scroll_space(slice_to_spiral_transform, scroll_center, spiral_zyx=spiral_zyx, epsilon=epsilon)
    normals_residual = 1. - (scroll_normal * target_normal).sum(dim=-1).abs()
    normals_loss = (normals_residual * normal_weight).sum() / normal_weight.sum().clamp(min=1)

    # grad_mag encodes a winding density (windings per base-volume voxel); the decode factor below
    # also rescales it to current-grid windings/voxel. The number of windings actually crossed by
    # the one-winding scroll-space segment (scroll_inner -> scroll_outer) is the line integral of
    # this density along it, so we sample the density at evenly spaced midpoints along the segment
    # and accumulate density * dl (a midpoint Riemann sum). For a correct fit the integral equals 1.
    density_decode = cfg['grad_mag_factor'] / cfg['grad_mag_encode_scale'] * lasagna_scale
    num_steps = int(cfg['spacing_integration_steps'])
    step_frac = (torch.arange(num_steps, device=device).float() + 0.5) / num_steps  # midpoints in [0, 1]
    # [num_points, num_steps, 3] scroll-space samples along scroll_inner -> scroll_outer
    integration_zyx = scroll_inner[:, None, :] + step_frac[None, :, None] * scroll_displacement[:, None, :]
    int_idx = (integration_zyx.detach() / lasagna_scale).round().long()
    izi = int_idx[..., 0] - z_origin
    iyi = int_idx[..., 1]
    ixi = int_idx[..., 2]
    int_in_bounds = (izi >= 0) & (izi < z_size) & (iyi >= 0) & (iyi < y_size) & (ixi >= 0) & (ixi < x_size)
    izi = izi.clamp(0, z_size - 1)
    iyi = iyi.clamp(0, y_size - 1)
    ixi = ixi.clamp(0, x_size - 1)
    grad_mag_u8 = volume[2, izi, iyi, ixi]  # [num_points, num_steps]
    sample_valid = (grad_mag_u8 != 0) & int_in_bounds
    density = grad_mag_u8.float() * density_decode  # current-grid windings/voxel
    # dl is the per-step scroll-space length (current-grid voxels); gradient flows through it so the
    # loss can stretch/compress the mapping until the integrated winding count matches.
    dl = scroll_segment_length / num_steps
    integrated_windings = (density * sample_valid.float()).sum(dim=-1) * dl
    # Only score samples whose whole segment lies inside the valid field; a partially covered path
    # would under-integrate and unfairly compare against 1.
    spacing_weight = sample_valid.all(dim=-1).float()
    spacing_residual = (integrated_windings - 1.).abs()
    spacing_loss = (spacing_residual * spacing_weight).sum() / spacing_weight.sum().clamp(min=1)

    return normals_loss, spacing_loss



def get_unattached_pcl_strip_losses(
    slice_to_spiral_transform,
    dr_per_winding,
    pcl_strips,
    get_or_build_unattached_pcl_flat,
    num_pcls_per_step,
    num_points_per_pcl,
    compute_dt,
    dt_max_winding=None,
):
    # Unattached pcls are treated as ordered strips, indexed by int(point_id), and
    # assumed to be locally dense enough that adjacent samples have |dtheta| < pi
    # (so unwrap_shifted_radii can stitch theta=0 crossings, exactly like a patch
    # row/column). Two losses are computed, analogous to the patch radius
    # and DT losses: (1) shifted-radius should be constant along the strip after
    # subtracting per-point winding-annotation offsets; (2) each point should snap to
    # its target winding, with the target taken from the snapped strip median.
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if not pcl_strips:
        return zero, zero

    num_to_sample = min(num_pcls_per_step, len(pcl_strips))
    chosen = np.random.choice(len(pcl_strips), num_to_sample, replace=False)

    flat = get_or_build_unattached_pcl_flat(pcl_strips, device)
    if flat is None or flat['total'] == 0:
        return zero, zero

    starts_cpu = flat['starts_cpu'].numpy()
    sampled_flat_indices = np.empty([num_to_sample, num_points_per_pcl], dtype=np.int64)
    for k, pcl_idx in enumerate(chosen):
        strip = pcl_strips[pcl_idx]
        N = len(strip['zyxs'])
        coords = np.sort(np.random.choice(N, num_points_per_pcl, replace=num_points_per_pcl > N))
        sampled_flat_indices[k] = starts_cpu[pcl_idx] + coords

    sampled_flat_indices_t = torch.from_numpy(sampled_flat_indices).to(device=device)
    zyxs_t = flat['zyxs'][sampled_flat_indices_t]
    winding_t = flat['windings'][sampled_flat_indices_t]

    spiral_zyxs = slice_to_spiral_transform(zyxs_t.reshape(-1, 3)).reshape(*zyxs_t.shape)
    theta, _, shifted_radii = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
    shifted_radii, crossing_adjustments = unwrap_shifted_radii(theta, shifted_radii, dr_per_winding)

    # Normalise so a pcl with mixed annotations still reads as a single 'strip'.
    normalised_radii = shifted_radii - winding_t * dr_per_winding

    radius_hinge_margin = dr_per_winding.detach() * cfg['patch_radius_loss_margin']
    dt_hinge_margin = dr_per_winding.detach() * cfg['patch_dt_loss_margin']

    mean_radii = normalised_radii.mean(dim=-1, keepdim=True)
    radius_deviations = (normalised_radii - mean_radii).abs()
    radius_loss = F.relu(radius_deviations - radius_hinge_margin).mean()

    if not compute_dt:
        return radius_loss, zero

    target_normalised = torch.round(normalised_radii.median(dim=-1, keepdim=True).values / dr_per_winding) * dr_per_winding
    target_shifted = target_normalised + winding_t * dr_per_winding
    target_radii = radius_from_unwrapped_shifted(
        theta, target_shifted, crossing_adjustments, dr_per_winding,
    )
    target_spiral_zyxs = torch.stack([
        spiral_zyxs[..., 0],
        torch.sin(theta) * target_radii,
        torch.cos(theta) * target_radii,
    ], dim=-1).detach()
    target_scroll_zyxs = slice_to_spiral_transform.inv(target_spiral_zyxs.reshape(-1, 3)).reshape(*target_spiral_zyxs.shape)

    within_p = cfg['patch_dt_within_patch_norm_p']
    across_p = cfg['patch_dt_norm_p']
    point_distances = torch.linalg.norm(zyxs_t - target_scroll_zyxs, dim=-1)
    point_distances = F.relu(point_distances - dt_hinge_margin) + 1.e-5
    track_losses = (point_distances ** within_p).mean(dim=-1) ** (1 / within_p)
    # Progressive DT: only strips whose snapped (raw, spiral-space) winding is within the current
    # cutoff contribute. Use shifted_radii (the strip's actual spiral position), not normalised_radii.
    strip_snapped_winding = torch.round(shifted_radii.median(dim=-1).values / dr_per_winding) * dr_per_winding
    active_mask = _progressive_dt_active_mask(strip_snapped_winding, dr_per_winding, dt_max_winding)
    dt_loss = _aggregate_dt_track_losses(track_losses, across_p, active_mask)

    return radius_loss, dt_loss



def get_symmetric_dirichlet_loss(slice_to_spiral_transform, dr_per_winding, outer_winding_idx, num_points, epsilon=None):
    # In-surface symmetric Dirichlet energy of the spiral<->scroll map, evaluated at points sampled
    # uniformly over the spiral cylinder (see sample_spiral_surface_frame).
    # At each point we take the orthonormal in-surface frame (e1, e2) in spiral space, map it to scroll
    # space through the inverse transform by finite differences to get its scroll-space image (a, b), and
    # form the 2x2 induced metric G = [[a.a, a.b], [a.b, b.b]]. The energy ||J||_F^2 + ||J^{-1}||_F^2 =
    # tr(G) + tr(G^{-1}) = (s1^2 + s2^2) + (1/s1^2 + 1/s2^2) is minimised (value 4) at an in-surface
    # isometry and diverges as the map degenerates (singular value -> 0 or inf), acting as a barrier
    # against in-surface collapse / element flips. We subtract 4 so the reported value is 0 at rest.
    device = dr_per_winding.device
    if outer_winding_idx is None:
        return torch.zeros([], device=device)
    if epsilon is None:
        epsilon = cfg['sym_dirichlet_finite_difference_epsilon']

    spiral_zyx, e1, e2 = sample_spiral_surface_frame(dr_per_winding, outer_winding_idx, num_points)

    spiral_shift_1 = spiral_zyx + e1 * epsilon
    spiral_shift_2 = spiral_zyx + e2 * epsilon
    combined_spiral = torch.cat([spiral_zyx, spiral_shift_1, spiral_shift_2], dim=0)
    combined_scroll = slice_to_spiral_transform.inv(combined_spiral)
    scroll_zyx, scroll_shift_1, scroll_shift_2 = combined_scroll.chunk(3, dim=0)

    a = (scroll_shift_1 - scroll_zyx) / epsilon
    b = (scroll_shift_2 - scroll_zyx) / epsilon
    g11 = (a * a).sum(dim=-1)
    g22 = (b * b).sum(dim=-1)
    g12 = (a * b).sum(dim=-1)
    trace_g = g11 + g22
    det_g = g11 * g22 - g12 * g12
    # Energy is tr(G) + tr(G^{-1}) = (s1^2 + s2^2) + (1/s1^2 + 1/s2^2), regularised per-eigenvalue so a
    # vanishing singular value contributes a finite-but-large 1/(lambda+eps) barrier. We compute the
    # regularised inverse-eigenvalue sum directly from trace_g, det_g via the algebraic identity
    #   1/(l1+eps) + 1/(l2+eps) = ((l1+eps) + (l2+eps)) / ((l1+eps)(l2+eps))
    #                           = (trace_g + 2*eps) / (det_g + eps*trace_g + eps**2)
    inverse_eps = 1e-3
    inverse_term = (trace_g + 2.0 * inverse_eps) / (det_g + inverse_eps * trace_g + inverse_eps ** 2)
    energy = (trace_g + inverse_term - 4.0).clamp(min=0.0)
    # Per-sample cap so a single near-degenerate sample doesn't dominate the batch mean / gradient.
    energy = energy.clamp(max=1.e2)
    return energy.mean()
