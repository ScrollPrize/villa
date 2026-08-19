import itertools
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import geom_utils
import prefetch
from dt_targets import patch_dt_target_in_sample_frame, strip_dt_target_in_sample_frame
from loss_maps import diagnostics_enabled, record_loss_samples
import strip_path_pools
from sample_spiral import (
    canonical_winding_samples,
    get_theta_and_radii,
    radius_from_unwrapped_shifted,
)
from spiral_helpers import _huber_abs


def _masked_mean(values, mask):
    mask_f = mask.to(values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp(min=1.)


def _pcl_sampling_group_weight(group, cfg):
    # Look up the per-step sampling weight of a sampling group in
    # cfg['pcl_sampling_weights']. Keys are matched on the group's basename with the
    # .json suffix stripped, so the source json stem (e.g. 'relative_windings') or the
    # single 'fibers' group. When the dict is in use every group must
    # have an explicit key, so a missing one is an error rather than a silent default.
    key = os.path.splitext(os.path.basename(str(group)))[0]
    try:
        return float(cfg['pcl_sampling_weights'][key])
    except KeyError:
        raise KeyError(
            f'pcl_sampling_weights has no entry for sampling group {key!r}; '
            f'when set, it must list a weight for every group'
        )


def build_pcl_sampling_strata(sampling_groups, cfg, member_weights=None):
    # Precompute the per-step sampling pool for _choose_pcl_indices from each pool
    # member's sampling group (source json file; all fibers share one 'fibers'
    # group). Members whose group is None are ineligible and excluded. When
    # cfg['pcl_sampling_weights'] is a dict, every group must have an explicit weight
    # and groups with weight <= 0 are switched off (dropped from the pool entirely).
    # Otherwise all groups stay eligible; the legacy stratified_pcl_sampling flag
    # controls whether selection uses equal strata or the combined pool.
    # member_weights (parallel to sampling_groups) sets each member's relative draw
    # probability within its stratum (and within the combined pool); used to give a
    # fiber-link component the sampling pressure of its member count rather than a
    # single strip's. None means uniform.
    # Returns {'strata': [int64 pool-index array per group], 'groups': [group name
    # per stratum], 'weights': float weight per stratum, 'member_probs': [per-stratum
    # draw probabilities or None], 'all': all eligible indices, 'all_probs': draw
    # probabilities over 'all' or None, 'effective_size': eligible member count
    # after expanding component multiplicities}.
    sampling_groups = list(sampling_groups)
    if member_weights is not None:
        member_weights = np.asarray(list(member_weights), dtype=np.float64)
        assert len(member_weights) == len(sampling_groups)
    group_to_indices = {}
    for idx, group in enumerate(sampling_groups):
        if group is None:
            continue
        group_to_indices.setdefault(group, []).append(idx)
    weighted = cfg['pcl_sampling_weights'] is not None
    strata, groups, weights, member_probs = [], [], [], []
    for group, indices in group_to_indices.items():
        weight = _pcl_sampling_group_weight(group, cfg) if weighted else 1.0
        if weighted and weight <= 0:
            continue  # switched off
        indices = np.asarray(indices, dtype=np.int64)
        strata.append(indices)
        groups.append(group)
        weights.append(weight)
        if member_weights is None:
            member_probs.append(None)
        else:
            w = member_weights[indices]
            member_probs.append(w / w.sum())
    all_indices = np.concatenate(strata) if strata else np.empty(0, dtype=np.int64)
    all_probs = None
    if member_weights is not None and len(all_indices):
        w = member_weights[all_indices]
        all_probs = w / w.sum()
        effective_size = int(round(w.sum()))
    else:
        effective_size = len(all_indices)
    return {
        'strata': strata,
        'groups': groups,
        'weights': np.asarray(weights, dtype=np.float64),
        'member_probs': member_probs,
        'all': all_indices,
        'all_probs': all_probs,
        'effective_size': effective_size,
    }


def _choose_pcl_indices(sampling_strata, num_to_sample, cfg):
    # Choose num_to_sample pool indices from a build_pcl_sampling_strata() bundle.
    # Explicit weights allocate draws proportionally. Without them, the legacy
    # stratified_pcl_sampling switch selects equal group shares or uniform sampling
    # over the combined pool. Per-member weights (member_probs / all_probs), when
    # the bundle carries them, skew the within-pool draws.
    weighted = cfg['pcl_sampling_weights'] is not None
    if not weighted and not cfg['pcl_stratified_pcl_sampling']:
        return np.random.choice(sampling_strata['all'], num_to_sample,
                                replace=num_to_sample > len(sampling_strata['all']),
                                p=sampling_strata['all_probs'])
    strata = sampling_strata['strata']
    weights = sampling_strata['weights'] if weighted else np.ones(
        len(strata), dtype=np.float64)
    shares = num_to_sample * weights / weights.sum()
    quotas = np.floor(shares).astype(np.int64)
    remainder = num_to_sample - int(quotas.sum())
    if remainder > 0:
        frac = shares - quotas
        probs = frac / frac.sum() if frac.sum() > 0 else weights / weights.sum()
        quotas[np.random.choice(len(strata), remainder, replace=False, p=probs)] += 1
    chosen = [
        np.random.choice(stratum, quota, replace=quota > len(stratum), p=probs)
        for stratum, quota, probs in zip(strata, quotas, sampling_strata['member_probs'])
        if quota > 0
    ]
    return np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)



def get_shell_outer_loss(shell_map, slice_to_spiral_transform, dr_per_winding, outer_winding_idx, *, cfg, z_begin, z_end):
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if shell_map is None or outer_winding_idx is None:
        return zero, {}

    num_samples = max(1, int(cfg['sample_count_shell_samples']))
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
# ragged edges. Consecutive path cells are grid-adjacent, so their crossing-map
# edges stitch theta=0 crossings exactly as for straight
# strips. The paths come from per-patch / per-anchor pools built and continuously refreshed by
# background worker processes (see strip_path_pools.py; strip_paths.py has the actual path
# computation); here we only subsample positions along a pooled path + subpixel jitter.
# ============================================================================================

def _sample_points_along_path(path_ij, num_points, rng=None, return_positions=False):
    # Subsample positions in traversal order with per-point subpixel jitter.
    # With return_positions, also return each sample's node-path position.
    rng = np.random if rng is None else rng
    path_len = path_ij.shape[0]
    positions = np.sort(rng.choice(path_len, num_points, replace=num_points > path_len))
    ijs = path_ij[positions].astype(np.float32) + rng.uniform(
        0., 1., size=[num_points, 2]).astype(np.float32)
    if return_positions:
        return ijs, positions.astype(np.int64)
    return ijs


def build_serpentine_quad_path(valid_quad_mask):
    # Order every valid quad along a boustrophedon walk (row by row, alternating
    # direction) and explicitly register every consecutive edge. Only used for small patches (area below
    # cfg['patch_2d_sampling_max_area']): holes and row turns make jumps whose
    # theta change must stay below pi, which a small physical extent guarantees.
    rows = []
    for i in np.flatnonzero(valid_quad_mask.any(axis=1)):
        js = np.flatnonzero(valid_quad_mask[i])
        if len(rows) % 2:
            js = js[::-1]
        rows.append(np.stack(
            [np.full(js.shape[0], i, dtype=np.int64), js.astype(np.int64)],
            axis=1))
    return np.ascontiguousarray(np.concatenate(rows, axis=0))


def _sample_dijkstra_strips_at_ij(
    patch, patch_idx, patch_atlas, i_q, j_q, num_points,
):
    # 'dijkstra'-mode replacement for _sample_l_shapes_at_ij: 4 geodesic strips from the
    # annotated cell, one per cardinal cone; None while the anchor's pools are still being
    # built in the background. Caller guarantees valid_quad[i_q, j_q].
    pools = strip_path_pools.get_anchor_path_pools(patch, i_q, j_q)
    if pools is None:
        return None
    result = []
    for pool in pools:
        path = pool[np.random.randint(len(pool))]
        ijs, positions = _sample_points_along_path(
            path, num_points, return_positions=True)
        result.append(PatchWalk(
            ijs=ijs,
            walk=_patch_sampled_walk(
                patch_atlas, patch_idx, path, positions),
        ))
    return result

def _sample_strip_ijs(line_valid, seed, fixed_coord, axis, num_points):
    # Sample num_points fractional ijs along the contiguous True run of `line_valid`
    # containing `seed`, fixed at `fixed_coord` along `axis` (axis=0 -> fixed i, varying j;
    # axis=1 -> fixed j, varying i), with sub-pixel jitter. Caller guarantees line_valid[seed].
    # The contiguous range is represented as registered crossing-map edges.
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


@geom_utils.maybe_compile
def _masked_all_pairs_l1(p1, p2, mask1, mask2, expected_diff):
    """Mean ``abs(p2 - p1 - expected_diff)`` over every valid point pair.

    Sorting one side and using prefix sums computes the same all-pairs L1
    objective in O(P log P) work and O(P) memory per batch item, instead of
    materialising the O(P**2) broadcast tensors.
    """
    num_points = p1.shape[-1]
    valid_counts1 = mask1.sum(dim=-1)
    valid_counts2 = mask2.sum(dim=-1)

    # abs(p2 - p1 - expected) == abs(p2 - (p1 + expected)). Invalid
    # entries sort to the end and are excluded from the prefix sums.
    shifted_p1 = p1 + expected_diff[:, None]
    sortable_p1 = torch.where(mask1, shifted_p1, torch.full_like(shifted_p1, torch.inf))
    sorted_p1 = sortable_p1.sort(dim=-1).values
    sorted_positions = torch.arange(num_points, device=p1.device)
    sorted_valid = sorted_positions[None, :] < valid_counts1[:, None]
    prefix = F.pad(
        torch.where(sorted_valid, sorted_p1, torch.zeros_like(sorted_p1)).cumsum(dim=-1),
        (1, 0),
    )

    # Values strictly below and above p2 contribute their signed distance.
    # The left/right split deliberately excludes exact ties, matching abs()'s
    # zero subgradient there.
    left_count = torch.searchsorted(sorted_p1, p2, right=False)
    right_begin = torch.searchsorted(sorted_p1, p2, right=True)
    left_sum = prefix.gather(dim=-1, index=left_count)
    right_prefix = prefix.gather(dim=-1, index=right_begin)
    total = prefix.gather(dim=-1, index=valid_counts1[:, None]).squeeze(-1)
    per_p2_sum = (
        p2 * left_count - left_sum
        + (total[:, None] - right_prefix)
        - p2 * (valid_counts1[:, None] - right_begin)
    )

    total_error = (per_p2_sum * mask2).sum()
    num_valid_pairs = (valid_counts1 * valid_counts2).sum()
    return total_error / num_valid_pairs.clamp(min=1)



@dataclass(slots=True)
class SampledWalk:
    """One topology-node walk and the sparse picks drawn from it."""

    node_ids: np.ndarray
    pick_positions: np.ndarray
    connect_fractional_picks: bool
    # Most strip losses define their theta frame at the first sparse pick.
    # Anchor-supervised walks (relative/absolute winding PCLs) instead carry
    # the exact annotated PCL node whose raw shifted-radius frame must be
    # transported through the walk, even when position zero was not sampled.
    reference_node_id: int | None = None


@dataclass(slots=True)
class PatchWalk:
    """Fractional patch picks accompanying a normalized topology walk."""

    ijs: np.ndarray
    walk: SampledWalk


@dataclass(slots=True)
class PackedWalks:
    """Walks resolved to cached canonical edges, with no XYZ payload."""

    edge_ids: torch.Tensor
    directions: torch.Tensor
    edge_valid: torch.Tensor
    pick_positions: torch.Tensor
    correction_node_ids: torch.Tensor
    walk_start_node_ids: torch.Tensor
    reference_node_ids: torch.Tensor


def _patch_sampled_walk(patch_atlas, patch_idx, path_ij, positions):
    """Resolve one patch-local dense path to the common global-node form."""
    path_ij = np.asarray(path_ij)
    patch_indices = np.full(path_ij.shape[0], patch_idx, dtype=np.int64)
    node_ids = patch_atlas.theta_node_ids(patch_indices, path_ij)
    return SampledWalk(
        node_ids=np.ascontiguousarray(node_ids, dtype=np.int64),
        pick_positions=np.ascontiguousarray(positions, dtype=np.int64),
        connect_fractional_picks=True,
    )


def _reconstruct_straight_walks(ijs_np, patch_indices, patch_atlas, walks):
    # Straight-mode strips (python or native sampler) confine their picks to one
    # contiguous valid run of a single grid line, so the node walk can be
    # rebuilt from the picks alone: the fixed coordinate is shared by every
    # pick, and every integer cell between the extreme picks' floor cells lies
    # in the same run. Rows already populated by an explicitly threaded path
    # (such as a serpentine 2D sample) are left untouched.
    _, N, P, _ = ijs_np.shape
    for slot in range(2):
        fixed_axis, var_axis = slot, 1 - slot
        for n in range(N):
            row = slot * N + n
            if walks[row] is not None:
                continue
            var = ijs_np[slot, n, :, var_axis]
            var_floor = np.floor(var).astype(np.int64)
            lo = var_floor.min()
            walk_len = int(var_floor.max() - lo + 1)
            path_ij = np.empty([walk_len, 2], dtype=np.int64)
            path_ij[:, var_axis] = lo + np.arange(walk_len, dtype=np.int64)
            path_ij[:, fixed_axis] = int(np.floor(
                ijs_np[slot, n, 0, fixed_axis]))
            walks[row] = _patch_sampled_walk(
                patch_atlas, patch_indices[n], path_ij, var_floor - lo)
    return walks


def _build_patch_ijs(
    patches, patch_indices, num_points_per_direction, rng, cfg, patch_atlas,
):
    # CPU part of the patch-strip sampler: for each patch, one row strip and
    # one column strip of fractional ijs. Pure numpy + `rng` so it can run on
    # the prefetch worker for the next step while the GPU works on this one.
    # Also returns the complete topology path and pick positions for every
    # path-sampled strip; straight paths are reconstructed by the caller.
    N = len(patch_indices)
    walks = [None] * (2 * N)

    use_dijkstra_strips = cfg['patch_strip_sampling'] == 'dijkstra'
    if use_dijkstra_strips:
        # Small 2D-sampled patches never draw from the geodesic pools, so don't
        # build or refresh pools for them.
        touched_patches = [patches[patch_idx] for patch_idx in dict.fromkeys(patch_indices)
                           if getattr(patches[patch_idx], '_sampling_2d_path', None) is None]
        strip_path_pools.ensure_patch_path_pools(touched_patches)
        # Submitted before the sampling below so the workers refresh while this step proceeds.
        for patch in touched_patches:
            strip_path_pools.submit_patch_pool_refresh(patch)

    P = num_points_per_direction
    horizontal_ijs_by_patch = np.empty([N, P, 2], dtype=np.float32)
    vertical_ijs_by_patch = np.empty([N, P, 2], dtype=np.float32)
    rand = rng.random
    randint = rng.randint
    fixed_jitters_h = rand(N).astype(np.float32)
    fixed_jitters_v = rand(N).astype(np.float32)
    var_jitters_h = rand((N, P)).astype(np.float32)
    var_jitters_v = rand((N, P)).astype(np.float32)
    for n, patch_idx in enumerate(patch_indices):
        patch = patches[patch_idx]

        path_2d = getattr(patch, '_sampling_2d_path', None)
        if path_2d is not None:
            # Small patch (area below cfg['patch_2d_sampling_max_area']): both
            # strip slots become independent sparse 2D samples over the whole
            # patch, drawn along the precomputed serpentine quad walk so the
            # downstream crossing gather sees a contiguous source path.
            horizontal_ijs_by_patch[n], positions_h = _sample_points_along_path(
                path_2d, P, rng, return_positions=True)
            vertical_ijs_by_patch[n], positions_v = _sample_points_along_path(
                path_2d, P, rng, return_positions=True)
            walks[n] = _patch_sampled_walk(
                patch_atlas, patch_idx, path_2d, positions_h)
            walks[N + n] = _patch_sampled_walk(
                patch_atlas, patch_idx, path_2d, positions_v)
            continue

        if use_dijkstra_strips:
            # Two independent geodesic strips per patch (no horizontal/vertical distinction;
            # the 'horizontal'/'vertical' arrays are just the two strip slots). Snapshot the
            # pool once: a background refresh may swap the list, but never mutates it.
            pool = patch._strip_path_pool
            path_a, path_b = (
                pool[k] for k in rng.choice(len(pool), 2, replace=len(pool) < 2)
            )
            horizontal_ijs_by_patch[n], positions_a = _sample_points_along_path(
                path_a, P, rng, return_positions=True)
            vertical_ijs_by_patch[n], positions_b = _sample_points_along_path(
                path_b, P, rng, return_positions=True)
            walks[n] = _patch_sampled_walk(
                patch_atlas, patch_idx, path_a, positions_a)
            walks[N + n] = _patch_sampled_walk(
                patch_atlas, patch_idx, path_b, positions_b)
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
        coords_h = np.sort(rng.choice(run_len_h, P, replace=P > run_len_h))
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
        coords_v = np.sort(rng.choice(run_len_v, P, replace=P > run_len_v))
        vertical_ijs_by_patch[n, :, 1] = col_idx + fixed_jitters_v[n]
        vertical_ijs_by_patch[n, :, 0] = lo_v + coords_v + var_jitters_v[n]

    return np.stack([horizontal_ijs_by_patch, vertical_ijs_by_patch], axis=0), walks  # (2, N, P, 2)


def _pack_walks(walks, crossing_map):
    """Pack ordered normalized walks on the theta-map device."""
    if not walks:
        return None
    num_walks = len(walks)
    device = crossing_map.device
    num_points = np.asarray(walks[0].pick_positions).size
    max_walk_len = 0
    normalized = []
    for k, walk in enumerate(walks):
        node_ids = np.asarray(walk.node_ids, dtype=np.int64)
        positions = np.asarray(walk.pick_positions, dtype=np.int64)
        if node_ids.ndim != 1 or node_ids.size == 0:
            raise ValueError(f'sampled walk {k} must contain a nonempty 1-D node path')
        if positions.ndim != 1 or positions.size != num_points:
            raise ValueError('sampled walks must have equal-length 1-D pick positions')
        if positions.size and (
            (positions < 0).any() or (positions >= node_ids.size).any()
        ):
            raise ValueError(f'sampled walk {k} contains an out-of-range pick position')
        if (node_ids < 0).any() or (node_ids >= crossing_map.num_nodes).any():
            raise ValueError(f'sampled walk {k} contains an unregistered node id')
        normalized.append((node_ids, positions))
        max_walk_len = max(max_walk_len, node_ids.size)

    node_ids_np = np.empty((num_walks, max_walk_len), dtype=np.int64)
    pick_positions_np = np.empty((num_walks, num_points), dtype=np.int64)
    edge_valid_np = np.zeros((num_walks, max_walk_len - 1), dtype=bool)
    correction_node_ids_np = np.full(
        (num_walks, num_points), -1, dtype=np.int64)
    walk_start_node_ids_np = np.empty(num_walks, dtype=np.int64)
    reference_node_ids_np = np.full(num_walks, -1, dtype=np.int64)
    for k, (walk, (walk_nodes, positions)) in enumerate(zip(walks, normalized)):
        walk_len = walk_nodes.size
        node_ids_np[k, :walk_len] = walk_nodes
        node_ids_np[k, walk_len:] = walk_nodes[-1]
        pick_positions_np[k] = positions
        edge_valid_np[k, :walk_len - 1] = True
        walk_start_node_ids_np[k] = walk_nodes[0]
        if walk.reference_node_id is not None:
            reference_node_id = int(walk.reference_node_id)
            if not 0 <= reference_node_id < crossing_map.num_nodes:
                raise ValueError(
                    f'sampled walk {k} contains an unregistered reference node id')
            reference_node_ids_np[k] = reference_node_id
        if walk.connect_fractional_picks:
            correction_node_ids_np[k] = walk_nodes[positions]

    node_ids = torch.as_tensor(node_ids_np, dtype=torch.int64, device=device)
    pick_positions = torch.as_tensor(
        pick_positions_np, dtype=torch.int64, device=device)
    edge_valid = torch.as_tensor(
        edge_valid_np, dtype=torch.bool, device=device)
    correction_node_ids = torch.as_tensor(
        correction_node_ids_np, dtype=torch.int64, device=device)
    walk_start_node_ids = torch.as_tensor(
        walk_start_node_ids_np, dtype=torch.int64, device=device)
    reference_node_ids = torch.as_tensor(
        reference_node_ids_np, dtype=torch.int64, device=device)

    edge_ids = torch.zeros_like(edge_valid, dtype=torch.int64)
    directions = torch.ones_like(edge_valid, dtype=torch.int8)
    if max_walk_len > 1:
        pairs = torch.stack([node_ids[:, :-1], node_ids[:, 1:]], dim=-1)
        resolved, resolved_dir = crossing_map.resolve_edges(pairs[edge_valid])
        edge_ids[edge_valid] = resolved
        directions[edge_valid] = resolved_dir
    return PackedWalks(
        edge_ids=edge_ids,
        directions=directions,
        edge_valid=edge_valid,
        pick_positions=pick_positions,
        correction_node_ids=correction_node_ids,
        walk_start_node_ids=walk_start_node_ids,
        reference_node_ids=reference_node_ids,
    )


def _sample_patch_batch(key, patches, sampling_probabilities, num_to_sample,
                        num_points_per_direction, cfg, patch_atlas=None,
                        crossing_map=None):
    # Returns sampled ijs, patch indices, sparse interpolated XYZ picks, and
    # packed edge/pick metadata. The full node paths never carry XYZ and never
    # enter the transform on an ordinary loss step.
    if num_to_sample <= 0:
        raise ValueError('Expected at least one patch index')

    def build(rng):
        patch_indices = rng.choice(len(patches), num_to_sample,
                                   p=sampling_probabilities, replace=True)
        native_atlas = getattr(patch_atlas, 'sampling_atlas', None)
        if native_atlas is not None and cfg['patch_strip_sampling'] != 'dijkstra':
            seed = int(rng.randint(0, np.iinfo(np.int64).max))
            native_walks = native_atlas.sample_patch_walks(
                np.ascontiguousarray(patch_indices, dtype=np.int64),
                num_points_per_direction,
                seed,
            )
            ijs_np = np.asarray(native_walks['ijs'])
            path_ijs = np.asarray(native_walks['path_ijs'])
            path_offsets = np.asarray(native_walks['path_offsets'])
            native_positions = np.asarray(native_walks['pick_positions'])
            # The native sampler only knows strips; rows for small 2D-sampled
            # patches (see cfg['patch_2d_sampling_max_area']) are overwritten
            # with serpentine whole-patch samples.
            walks = [
                _patch_sampled_walk(
                    patch_atlas,
                    patch_indices[row % num_to_sample],
                    path_ijs[path_offsets[row]:path_offsets[row + 1]],
                    native_positions.reshape(
                        2 * num_to_sample, num_points_per_direction)[row],
                )
                for row in range(2 * num_to_sample)
            ]
            small_ns = [
                n for n, patch_idx in enumerate(patch_indices)
                if getattr(patches[patch_idx], '_sampling_2d_path', None) is not None
            ]
            if small_ns:
                ijs_np = np.array(ijs_np)  # the native buffer may be read-only
                for n in small_ns:
                    walks[n] = None
                    walks[num_to_sample + n] = None
                    path_2d = patches[patch_indices[n]]._sampling_2d_path
                    ijs_np[0, n], positions_h = _sample_points_along_path(
                        path_2d, num_points_per_direction, rng,
                        return_positions=True)
                    ijs_np[1, n], positions_v = _sample_points_along_path(
                        path_2d, num_points_per_direction, rng,
                        return_positions=True)
                    walks[n] = _patch_sampled_walk(
                        patch_atlas, patch_indices[n], path_2d, positions_h)
                    walks[num_to_sample + n] = _patch_sampled_walk(
                        patch_atlas, patch_indices[n], path_2d, positions_v)
        else:
            ijs_np, walks = _build_patch_ijs(
                patches, patch_indices, num_points_per_direction, rng, cfg,
                patch_atlas)
        if cfg['patch_strip_sampling'] != 'dijkstra':
            # Straight strips (python or native sampler alike) are rebuilt
            # from the picks; path-sampled strips are already threaded above.
            walks = _reconstruct_straight_walks(
                ijs_np, patch_indices, patch_atlas, walks)
        if any(walk is None for walk in walks):
            raise RuntimeError('patch sampler did not produce every walk row')
        packed_walks = _pack_walks(walks, crossing_map)
        ijs_cpu = torch.from_numpy(ijs_np)
        idx_cpu = torch.from_numpy(
            np.ascontiguousarray(patch_indices, dtype=np.int64))
        _, N, P, _ = ijs_cpu.shape
        slice_zyxs_gpu = patch_atlas.lookup(
            idx_cpu[None, :, None].expand(2, N, P), ijs_cpu)
        target_device = patch_atlas.device
        ijs_gpu = ijs_cpu.to(device=target_device, non_blocking=True)
        idx_gpu = idx_cpu.to(device=target_device, non_blocking=True)
        return ijs_gpu, idx_gpu, slice_zyxs_gpu, packed_walks

    if prefetch.prefetch_enabled() and torch.cuda.is_available():
        pf = prefetch.get_prefetcher()
        rng = pf.np_rng(key)
        return pf.pop_or_run((key, id(crossing_map), num_to_sample,
                              num_points_per_direction),
                             lambda: build(rng))
    return build(prefetch.LegacyNumpyRandom)


def _unwrap_sampled_tracks(
    crossing_map, dr_per_winding, theta, shifted_radii, packed_walks,
):
    crossing_adjustments = crossing_map.adjustments(
        packed_walks,
        theta.reshape(-1, theta.shape[-1]),
        dr_per_winding,
    )
    crossing_adjustments = crossing_adjustments.reshape(theta.shape)
    return shifted_radii + crossing_adjustments, crossing_adjustments


def _sample_patch_tracks(slice_to_spiral_transform, dr_per_winding, patches,
                         patch_atlas, batch, crossing_map, extra_zyxs=None):
    # For each patch, take one row and one column in straight mode, or two
    # geodesic paths in dijkstra mode. Either representation is a contiguous
    # walk. Cached edge crossings are gathered along the complete node path,
    # irrespective of how sparse the fractional picks are.

    # The bilinear atlas gather already ran on the CPU at batch-build time
    # (see _sample_patch_batch); the batch carries the interpolated points.
    combined_ijs_gpu, patch_indices_gpu, all_slice_zyxs, packed_walks = batch

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
    all_shifted_radii, all_crossing_adjustments = _unwrap_sampled_tracks(
        crossing_map, dr_per_winding, all_theta,
        all_shifted_radii, packed_walks,
    )

    return (
        combined_ijs_gpu,
        all_slice_zyxs,
        all_spiral_zyxs,
        all_theta,
        all_shifted_radii,
        all_crossing_adjustments,
        extra_spiral,
    )




def _patch_radius_and_dt_losses(
    slice_to_spiral_transform, dr_per_winding,
    all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii,
    all_crossing_adjustments,
    num_patches_for_radius, num_patches_for_dt, compute_dt, dt_max_winding,
    radius_loss_margin, radius_loss_inv, radius_within_norm_p,
    dt_loss_margin, dt_norm_p, dt_within_patch_norm_p,
    patch_indices=None, sample_ijs=None, dt_target_cache=None,
    diagnostic_prefix='patch',
):
    # Shared radius + DT patch losses, operating on pre-sampled row/column tracks
    # (all_*; see _sample_patch_tracks). Pulled out of get_patch_and_umbilicus_losses so the
    # same loss can serve both the verified and the untrusted ('unverified') patch sets with
    # independent hyperparameters. Returns (mean_radius_deviation, patch_dt_loss).
    # `dt_target_cache` is the whole-object DT target cache (see dt_targets.py) or None
    # in legacy strip-median mode; `patch_indices` maps the sampled tracks to cache rows.
    radius_hinge_margin = dr_per_winding.detach() * radius_loss_margin
    dt_hinge_margin = dr_per_winding.detach() * dt_loss_margin

    # Each patch row/col should lie at constant shifted-radius.
    radius_shifted_radii = all_shifted_radii[:, :num_patches_for_radius]
    radius_slice_zyxs = all_slice_zyxs[:, :num_patches_for_radius]
    radius_spiral_zyxs = all_spiral_zyxs[:, :num_patches_for_radius]
    radius_theta = all_theta[:, :num_patches_for_radius]
    radius_crossing_adjustments = all_crossing_adjustments[:, :num_patches_for_radius]
    mean_shifted_radii = radius_shifted_radii.mean(dim=-1, keepdim=True)
    radius_target_spiral_zyxs = None
    if radius_loss_inv or diagnostics_enabled():
        radius_target_radii = radius_from_unwrapped_shifted(
            radius_theta,
            mean_shifted_radii,
            radius_crossing_adjustments,
            dr_per_winding,
        )
        radius_target_spiral_zyxs = torch.stack([
            radius_spiral_zyxs[..., 0],
            torch.sin(radius_theta) * radius_target_radii,
            torch.cos(radius_theta) * radius_target_radii,
        ], dim=-1).detach()

    if radius_loss_inv:
        # Express the loss in scroll space like the DT loss below: construct target
        # spiral-space points at the track's mean shifted-radius (continuous, not snapped
        # to an integer winding) but with each point's own z and theta, transform back to
        # scroll space, and penalise the distance from the original sampled points.
        radius_target_scroll_zyxs = slice_to_spiral_transform.inv(radius_target_spiral_zyxs.reshape(-1, 3)).reshape(*radius_target_spiral_zyxs.shape)

        radius_point_distances = torch.linalg.norm(radius_slice_zyxs - radius_target_scroll_zyxs, dim=-1)
        radius_point_residuals = F.relu(radius_point_distances - radius_hinge_margin)
        mean_radius_deviation = radius_point_residuals.mean()
        record_loss_samples(
            f'{diagnostic_prefix}_radius', radius_spiral_zyxs,
            radius_point_residuals,
            display_spiral_zyx=radius_target_spiral_zyxs,
        )
    else:
        # Penalise deviation from the track's mean shifted-radius directly in spiral space.
        radius_deviations = (radius_shifted_radii - mean_shifted_radii).abs()
        radius_deviations_hinge = F.relu(radius_deviations - radius_hinge_margin)
        if radius_within_norm_p == 1.0:
            mean_radius_deviation = radius_deviations_hinge.mean()
        else:
            d = radius_deviations_hinge + 1.e-5
            per_track = (d ** radius_within_norm_p).mean(dim=-1) ** (1.0 / radius_within_norm_p)
            mean_radius_deviation = per_track.mean()
        record_loss_samples(
            f'{diagnostic_prefix}_radius', radius_spiral_zyxs,
            radius_deviations_hinge,
            display_spiral_zyx=radius_target_spiral_zyxs,
        )

    if compute_dt:
        dt_slice_zyxs = all_slice_zyxs[:, :num_patches_for_dt]
        dt_spiral_zyxs = all_spiral_zyxs[:, :num_patches_for_dt]
        dt_theta = all_theta[:, :num_patches_for_dt]
        dt_shifted_radii = all_shifted_radii[:, :num_patches_for_dt]
        dt_crossing_adjustments = all_crossing_adjustments[:, :num_patches_for_dt]

        # Define the DT target winding (see patch_dt_target_in_sample_frame: whole-patch cached
        # target when available, else the track's own snapped median). Every sampled
        # point on the track is then pulled towards that target winding.
        target_shifted_radii = patch_dt_target_in_sample_frame(
            dt_shifted_radii,
            sample_ijs[:, :num_patches_for_dt] if sample_ijs is not None else None,
            dt_theta,
            dt_crossing_adjustments,
            dr_per_winding,
            dt_target_cache,
            patch_indices[:num_patches_for_dt] if patch_indices is not None else None,
        )
        target_radii = radius_from_unwrapped_shifted(
            dt_theta,
            target_shifted_radii,
            dt_crossing_adjustments,
            dr_per_winding,
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
        diagnostic_mask = (active_mask[..., None] if active_mask is not None else None)
        record_loss_samples(
            f'{diagnostic_prefix}_dt', dt_spiral_zyxs,
            point_distances, diagnostic_mask,
            display_spiral_zyx=target_spiral_zyxs,
        )
    else:
        patch_dt_loss = torch.zeros([], device=dr_per_winding.device)

    return mean_radius_deviation, patch_dt_loss



def get_patch_and_umbilicus_losses(slice_to_spiral_transform, dr_per_winding, num_patches_for_radius, num_patches_for_dt, patches, patch_atlas, patch_sampling_probabilities, umbilicus_zyx, compute_dt=True, shell_valid_zyxs=None, shell_outer_winding_idx=None, dt_max_winding=None, dt_target_cache=None, *, crossing_map, cfg):

    n_umb = umbilicus_zyx.shape[0]
    if shell_valid_zyxs is not None:
        num_shell_samples = min(int(cfg['sample_count_shell_samples']), shell_valid_zyxs.shape[0])
        sample_idx = torch.randint(shell_valid_zyxs.shape[0], (num_shell_samples,), device=shell_valid_zyxs.device)
        extra_zyxs = torch.cat([umbilicus_zyx, shell_valid_zyxs[sample_idx]], dim=0)
    else:
        extra_zyxs = umbilicus_zyx

    if len(patches) == 0:
        # supervision-free (disable_patches) fits: the umbilicus and shell
        # anchors still apply; the patch radius/DT terms are inert zeros
        extra_spiral = slice_to_spiral_transform(extra_zyxs)
        mean_radius_deviation = torch.zeros([], device=dr_per_winding.device)
        patch_dt_loss = torch.zeros([], device=dr_per_winding.device)
    else:
        # Sample once and share the tracks between the radius and DT losses; the loss using
        # fewer patches takes a prefix of the larger sample.
        num_patches_to_sample = max(num_patches_for_radius, num_patches_for_dt) if compute_dt else num_patches_for_radius
        batch = _sample_patch_batch(
            'verified_patches', patches, patch_sampling_probabilities,
            num_patches_to_sample, cfg['sample_count_points_per_patch'] // 2,
            cfg, patch_atlas, crossing_map)

        (
            sample_ijs,
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
            batch,
            crossing_map,
            extra_zyxs,
        )

        mean_radius_deviation, patch_dt_loss = _patch_radius_and_dt_losses(
            slice_to_spiral_transform, dr_per_winding,
            all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii,
            all_crossing_adjustments,
            num_patches_for_radius, num_patches_for_dt, compute_dt, dt_max_winding,
            cfg['patch_radius_loss_margin'], cfg['patch_radius_loss_inv'], cfg['patch_radius_within_norm_p'],
            cfg['patch_dt_loss_margin'], cfg['patch_dt_norm_p'], cfg['patch_dt_within_patch_norm_p'],
            patch_indices=batch[1], sample_ijs=sample_ijs, dt_target_cache=dt_target_cache,
            diagnostic_prefix='patch',
        )

    umbilicus_spiral = extra_spiral[:n_umb]
    shell_spiral_zyxs = extra_spiral[n_umb:] if shell_valid_zyxs is not None else None

    # Umbilicus should map to the spiral origin (yx ≈ 0)
    umbilicus_loss = umbilicus_spiral[..., 1:].abs().mean()

    if shell_spiral_zyxs is not None:
        radius_hinge_margin = dr_per_winding.detach() * cfg['patch_radius_loss_margin']
        shell_theta, _, shell_shifted_radii = get_theta_and_radii(
            shell_spiral_zyxs[..., 1:], dr_per_winding)
        shell_target = dr_per_winding * float(shell_outer_winding_idx)
        shell_patch_radius_residual = F.relu(
            (shell_shifted_radii - shell_target).abs() - radius_hinge_margin)
        shell_patch_radius_loss = shell_patch_radius_residual.mean()
        shell_target_radii = (
            shell_target
            + shell_theta / (2 * np.pi) * dr_per_winding.detach()
        )
        shell_target_spiral_zyxs = torch.stack([
            shell_spiral_zyxs[..., 0],
            torch.sin(shell_theta) * shell_target_radii,
            torch.cos(shell_theta) * shell_target_radii,
        ], dim=-1).detach()
        record_loss_samples(
            'shell_patch_radius', shell_spiral_zyxs,
            shell_patch_radius_residual,
            display_spiral_zyx=shell_target_spiral_zyxs,
        )
    else:
        shell_patch_radius_loss = torch.zeros([], device=dr_per_winding.device)

    return mean_radius_deviation, umbilicus_loss, patch_dt_loss, shell_patch_radius_loss



def get_unverified_patch_losses(slice_to_spiral_transform, dr_per_winding, num_patches_for_radius, num_patches_for_dt, patches, patch_atlas, patch_sampling_probabilities, compute_dt=True, dt_max_winding=None, dt_target_cache=None, *, crossing_map, cfg):
    # Radius + DT losses for the untrusted 'unverified' patch set. Same machinery as the
    # verified patches (shared _sample_patch_tracks + _patch_radius_and_dt_losses) but with the
    # independent unverified_* hyperparameters and no umbilicus/shell extras. These patches are
    # masked away near trusted geometry upstream (see _mask_patches_near_trusted_geometry), so
    # they only constrain regions the verified inputs don't cover.
    num_patches_to_sample = max(num_patches_for_radius, num_patches_for_dt) if compute_dt else num_patches_for_radius
    batch = _sample_patch_batch(
        'unverified_patches', patches, patch_sampling_probabilities,
        num_patches_to_sample, cfg['sample_count_unverified_points_per_patch'] // 2,
        cfg, patch_atlas, crossing_map)

    (
        sample_ijs,
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
        batch,
        crossing_map,
    )

    return _patch_radius_and_dt_losses(
        slice_to_spiral_transform, dr_per_winding,
        all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii,
        all_crossing_adjustments,
        num_patches_for_radius, num_patches_for_dt, compute_dt, dt_max_winding,
        cfg['patch_unverified_patch_radius_loss_margin'], cfg['patch_unverified_patch_radius_loss_inv'], cfg['patch_unverified_patch_radius_within_norm_p'],
        cfg['patch_unverified_patch_dt_loss_margin'], cfg['patch_unverified_patch_dt_norm_p'], cfg['patch_unverified_patch_dt_within_patch_norm_p'],
        patch_indices=batch[1], sample_ijs=sample_ijs, dt_target_cache=dt_target_cache,
        diagnostic_prefix='unverified_patch',
    )



def _sample_single_l_shape(
    valid_quad, patch_idx, patch_atlas, i_q, j_q,
    leg1_axis, leg1_dir, leg2_dir, num_points,
):
    # Sample a single L-shape on `valid_quad` starting at (i_q, j_q). Leg 1 walks along
    # `leg1_axis` (0 -> varying j, 1 -> varying i) in direction `leg1_dir` (+1 or -1) to a
    # uniformly random turn point inside the contiguous valid run. Leg 2 walks from the
    # turn point along the perpendicular axis in direction `leg2_dir` (+1 or -1) to the end
    # of its valid run. Returns a float32 [num_points, 2] sampled in traversal order, with
    # subpixel jitter; the fixed-axis jitter is shared within each leg (matching the
    # _sample_strip_ijs convention), preserving the registered node traversal along the
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

    path = np.empty([num_positions, 2], dtype=np.int64)
    dense_steps = np.arange(num_positions, dtype=np.int64)
    dense_leg1 = dense_steps <= turn_step
    if leg1_axis == 0:
        path[dense_leg1, 0] = i_q
        path[dense_leg1, 1] = var_start1 + leg1_dir * dense_steps[dense_leg1]
        path[~dense_leg1, 0] = i_turn + leg2_dir * (dense_steps[~dense_leg1] - turn_step)
        path[~dense_leg1, 1] = j_turn
    else:
        path[dense_leg1, 0] = var_start1 + leg1_dir * dense_steps[dense_leg1]
        path[dense_leg1, 1] = j_q
        path[~dense_leg1, 0] = i_turn
        path[~dense_leg1, 1] = j_turn + leg2_dir * (dense_steps[~dense_leg1] - turn_step)
    return PatchWalk(
        ijs=ijs,
        walk=_patch_sampled_walk(
            patch_atlas, patch_idx, path, steps.astype(np.int64)),
    )



def _sample_l_shapes_at_ij(
    patch, patch_idx, patch_atlas, i, j, num_points, cfg,
):
    # Sample 4 strips anchored on the annotated point (i, j) of `patch`, one per cardinal
    # primary direction. In 'dijkstra' mode these are geodesic strips to distant endpoints
    # (one per cardinal cone; see _sample_dijkstra_strips_at_ij); otherwise L-shapes, one per
    # primary direction: right (+j), left (-j), down (+i), up (-i). For each L, leg 2's
    # perpendicular direction is chosen uniformly at random. Returns a list of 4 float32
    # [num_points, 2] arrays sampled in traversal order, or None if (i, j) doesn't lie on
    # a valid quad (or, in dijkstra mode, while this anchor's path pools are still being
    # built in the background). Each L is a single contiguous walk in patch space, so cached
    # crossings handle theta=0 seams along the bent strip just as along a straight
    # row/column.
    valid_quad = patch._sampling_valid_quad_mask_np
    H_q, W_q = valid_quad.shape
    i_q = min(max(int(i), 0), H_q - 1)
    j_q = min(max(int(j), 0), W_q - 1)
    if not valid_quad[i_q, j_q]:
        return None

    if cfg['patch_strip_sampling'] == 'dijkstra':
        return _sample_dijkstra_strips_at_ij(
            patch, patch_idx, patch_atlas, i_q, j_q, num_points)

    primary_specs = [(0, +1), (0, -1), (1, +1), (1, -1)]  # (leg1_axis, leg1_dir)
    return [
        _sample_single_l_shape(
            valid_quad, patch_idx, patch_atlas, i_q, j_q,
            leg1_axis, leg1_dir,
            leg2_dir=int(np.random.choice([-1, +1])),
            num_points=num_points,
        )
        for leg1_axis, leg1_dir in primary_specs
    ]


def _sample_l_shapes_batch(patches_dict, patch_atlas, requests, num_points, cfg):
    """Sample four L-shapes for each ``(patch_id, i, j)`` request."""
    if not requests:
        return []
    native_atlas = getattr(patch_atlas, 'sampling_atlas', None)
    if cfg['patch_strip_sampling'] == 'dijkstra':
        native_atlas = None
    if native_atlas is None:
        return [
            _sample_l_shapes_at_ij(
                patches_dict[pid], patch_atlas.id_to_idx[pid], patch_atlas,
                i, j, num_points, cfg)
            for pid, i, j in requests
        ]
    patch_indices = np.fromiter(
        (patch_atlas.id_to_idx[pid] for pid, _, _ in requests),
        dtype=np.int64,
        count=len(requests),
    )
    anchors = np.asarray([(i, j) for _, i, j in requests], dtype=np.int64)
    result = native_atlas.sample_l_shapes(
        patch_indices,
        np.ascontiguousarray(anchors),
        num_points,
        int(np.random.randint(0, np.iinfo(np.int64).max)),
    )
    ijs = np.asarray(result['ijs'])
    pick_positions = np.asarray(result['pick_positions'])
    waypoints = np.asarray(result['waypoints'])
    valid = np.asarray(result['valid'], dtype=bool)
    sampled = []
    for k in range(len(requests)):
        if not valid[k]:
            sampled.append(None)
            continue
        shapes = []
        for s in range(4):
            anchor, turn, end = waypoints[k, s]
            leg1_delta = np.sign(turn - anchor)
            leg2_delta = np.sign(end - turn)
            leg1_len = int(np.abs(turn - anchor).sum())
            leg2_len = int(np.abs(end - turn).sum())
            leg1 = anchor + np.arange(leg1_len + 1)[:, None] * leg1_delta
            leg2 = turn + np.arange(1, leg2_len + 1)[:, None] * leg2_delta
            path = np.concatenate([leg1, leg2], axis=0).astype(np.int64)
            shapes.append(PatchWalk(
                ijs=ijs[k, s],
                walk=_patch_sampled_walk(
                    patch_atlas, patch_indices[k], path,
                    pick_positions[k, s]),
            ))
        sampled.append(shapes)
    return sampled


def _set_walk_reference_node(patch_walks, reference_node_id):
    """Anchor sampled patch walks in an exact PCL node's theta frame."""
    for patch_walk in patch_walks:
        patch_walk.walk.reference_node_id = int(reference_node_id)


def _pcl_chain_seam_adjustments(crossing_map, dr_per_winding, chain_node_ids):
    values = []
    for node_ids in chain_node_ids:
        edge_ids, directions = crossing_map.resolve_walks(node_ids)
        winding_steps = (
            crossing_map.crossings[edge_ids]
            * directions.to(crossing_map.crossings.dtype))
        values.append(winding_steps.to(torch.int32).sum())
    return torch.stack(values).to(dr_per_winding.dtype) * dr_per_winding.detach()


def get_patch_rel_winding_loss(slice_to_spiral_transform, dr_per_winding, patches_dict, patch_atlas, point_collections, sampling_strata, *, crossing_map, cfg, z_begin, z_end):
    # For pairs of annotated PCL points on different patches, constrain the spiral
    # shifted-radius gap to match the annotated winding-number difference. Each
    # cross-patch pcl exposes its attached points grouped by patch
    # (pcl['points_by_patch']); we form the set of all pairs (p1, p2) whose patches
    # differ and sample uniformly from it. For each annotated point we build 4
    # L-shaped strips: from (i, j), walk along one of the cardinal patch directions
    # (right, left, down, up) to a uniformly-random turn point inside the contiguous
    # valid run, then 90-degree-turn into a uniformly-random perpendicular direction
    # and walk to the end of that valid run. Each L is sampled in traversal order,
    # and cached crossings are gathered along its complete node path. We then pool
    # all 4 L-strips per annotated point into one set of sample
    # points and take a single all-pairs diff between p1's and p2's pooled sets,
    # regressing it onto winding_diff * dr_per_winding. If the PCL chain between
    # the selected points crosses theta=0, adjust the expected delta by that
    # branch-cut jump.

    num_points_per_strip = cfg['sample_count_points_per_patch'] // 2
    num_strips_per_pcl = 4
    num_strips_per_pair = 2 * num_strips_per_pcl  # 8

    # Each entry holds two groups of four L walks, patch ids, the annotated
    # winding difference, and ordered PCL-chain node ids from p1 to p2.
    strip_pairs = []
    pair_requests = []

    # sampling_strata indexes into point_collections and already excludes single-point
    # pcls (possible only for winding_is_absolute pcls), which can't form a cross-patch
    # pair; see the build_pcl_sampling_strata call in fit_spiral.main.
    num_pcls_per_step = min(
        cfg['sample_count_relative_winding_pcls'],
        sampling_strata['effective_size'],
    )
    if num_pcls_per_step <= 0:
        return torch.zeros([], device=dr_per_winding.device)
    selected_idxs = _choose_pcl_indices(sampling_strata, num_pcls_per_step, cfg)
    selected_pcls = [point_collections[i] for i in selected_idxs]

    for pcl in selected_pcls:
        # Uniform chain interface (spiral_helpers.Chain): id-sorted order for
        # ordinary pcls, the fiber-graph route (hopping fibers at junctions)
        # for merged fiber-link components -- whose id-sorted order is NOT
        # chain-valid across members. The full chain is used even in
        # adjacent-patches mode, since adjacent patches may sit far apart along
        # the pcl (or on different fibers).
        chain = pcl['chain']

        # Pair patches either only with their immediate neighbour in the pcl's
        # patch ordering (first-seen order; built in main()),
        # or with every other patch.
        if cfg['pcl_rel_winding_adjacent_patches_only']:
            cross_pairs = [(p1, p2) for p1, p2 in zip(pcl['points_by_patch'], list(pcl['points_by_patch'])[1:])]
        else:
            cross_pairs = list(itertools.combinations(pcl['points_by_patch'], r=2))
        if not cross_pairs:
            continue

        num_pairs_for_pcl = min(len(cross_pairs), cfg['sample_count_relative_winding_patch_pairs_per_pcl'])
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

            pcl_chain_node_ids = np.fromiter(
                (point['_theta_node_id'] for point in chain.points_between(p1, p2)),
                dtype=np.int64)
            pair_requests.append((
                (pid1, i1, j1), (pid2, i2, j2),
                pid1, pid2, winding_diff, pcl_chain_node_ids,
                p1['_theta_node_id'], p2['_theta_node_id'],
            ))

    sampled_l_shapes = _sample_l_shapes_batch(
        patches_dict,
        patch_atlas,
        [request for pair in pair_requests for request in pair[:2]],
        num_points_per_strip,
        cfg,
    )
    for pair_index, pair in enumerate(pair_requests):
        ls1 = sampled_l_shapes[2 * pair_index]
        ls2 = sampled_l_shapes[2 * pair_index + 1]
        if ls1 is None or ls2 is None:
            continue
        _set_walk_reference_node(ls1, pair[6])
        _set_walk_reference_node(ls2, pair[7])
        # Keep the downstream tuple limited to loss payload; the exact anchor
        # node IDs now live on the sampled walks themselves.
        strip_pairs.append((ls1, ls2, *pair[2:6]))

    if not strip_pairs:
        return torch.zeros([], device=dr_per_winding.device)

    # Flatten: 8 strips per pair, ordered as p1's 4 strips followed by p2's 4 strips.
    total_strips = len(strip_pairs) * num_strips_per_pair
    flat_ijs = np.empty([total_strips, num_points_per_strip, 2], dtype=np.float32)
    flat_pids = []
    flat_walks = []
    for k, (ls1, ls2, pid1, pid2, _, _) in enumerate(strip_pairs):
        base = k * num_strips_per_pair
        for s, strip in enumerate(ls1):
            flat_ijs[base + s] = strip.ijs
            flat_walks.append(strip)
        for s, strip in enumerate(ls2):
            flat_ijs[base + num_strips_per_pcl + s] = strip.ijs
            flat_walks.append(strip)
        flat_pids.extend([pid1] * num_strips_per_pcl + [pid2] * num_strips_per_pcl)

    # Batched bilinear gather on the device-resident atlas.
    patch_idx_per_strip_np = np.fromiter(
        (patch_atlas.id_to_idx[pid] for pid in flat_pids),
        dtype=np.int64,
        count=total_strips,
    )
    patch_idx_per_strip = torch.from_numpy(patch_idx_per_strip_np)
    patch_idx_per_sample = patch_idx_per_strip[:, None].expand(total_strips, num_points_per_strip)
    flat_zyxs = patch_atlas.lookup(patch_idx_per_sample, torch.from_numpy(flat_ijs))

    # Mask out strip samples whose z falls outside [z_begin - margin, z_end + margin).
    # Computed before applying cached crossing adjustments but masked afterward.
    z_margin = cfg['patch_loss_z_margin']
    z_mask = (flat_zyxs[..., 0] >= z_begin - z_margin) & (flat_zyxs[..., 0] < z_end + z_margin)

    flat_spiral = slice_to_spiral_transform(flat_zyxs.reshape(-1, 3)).reshape(*flat_zyxs.shape)
    theta, _, shifted_radii = get_theta_and_radii(flat_spiral[..., 1:], dr_per_winding)
    packed_walks = _pack_walks(
        [patch_walk.walk for patch_walk in flat_walks], crossing_map)
    shifted_radii, _ = _unwrap_sampled_tracks(
        crossing_map, dr_per_winding, theta, shifted_radii, packed_walks)

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
        device=dr_per_winding.device,
        dtype=torch.float32,
    )
    pcl_seam_adjustments = _pcl_chain_seam_adjustments(
        crossing_map,
        dr_per_winding,
        [strip_pair[5] for strip_pair in strip_pairs],
    )
    expected_diff = (winding_diffs * dr_per_winding) - pcl_seam_adjustments
    if diagnostics_enabled():
        # Attribute each pair's exact masked all-pairs residual uniformly to
        # the samples on both supporting patch strips.
        pair_residuals = []
        for pair_index in range(len(strip_pairs)):
            pair_residuals.append(_masked_all_pairs_l1(
                p1_r[pair_index:pair_index + 1],
                p2_r[pair_index:pair_index + 1],
                m1[pair_index:pair_index + 1],
                m2[pair_index:pair_index + 1],
                expected_diff[pair_index:pair_index + 1],
            ))
        pair_residuals = torch.stack(pair_residuals)
        diagnostic_spiral = flat_spiral.reshape(
            len(strip_pairs), num_strips_per_pair, num_points_per_strip, 3)
        record_loss_samples(
            'rel_winding', diagnostic_spiral,
            pair_residuals[:, None, None], z_mask,
        )
    return _masked_all_pairs_l1(p1_r, p2_r, m1, m2, expected_diff)



def get_patch_abs_winding_loss(slice_to_spiral_transform, dr_per_winding, patches_dict, patch_atlas, point_collections, *, crossing_map, cfg, z_begin, z_end):
    # For PCL points carrying an absolute winding annotation (only pcls flagged
    # metadata.winding_is_absolute), pin the spiral shifted-radius at each annotated
    # point to its absolute target, winding_annotation * dr_per_winding (the spiral has
    # radius 0 at winding 0 and grows at dr_per_winding, so shifted_radius == winding *
    # dr_per_winding). This mirrors get_patch_rel_winding_loss, but anchors each point's
    # absolute winding instead of regressing a pair's winding difference: we sample some
    # absolute-winding pcls, some attached points within each, build 4 L-shaped strips
    # per point (sampled in traversal order so cached crossings can be gathered),
    # then drive every in-roi strip sample's shifted radius to
    # the point's target. Each L starts at the annotated point, so its unwrapped
    # shifted-radius keeps the true absolute scale at the anchor.

    num_points_per_strip = cfg['sample_count_points_per_patch'] // 2
    num_strips_per_point = 4

    # Each entry: (ls, pid, winding_annotation) where ls is a list of 4 L-shape ij strips.
    strips = []
    strip_requests = []

    abs_pcls = [pcl for pcl in point_collections if pcl.get('metadata', {}).get('winding_is_absolute', False)]
    num_pcls_per_step = min(cfg['sample_count_absolute_winding_pcls'], len(abs_pcls))
    if num_pcls_per_step <= 0:
        return torch.zeros([], device=dr_per_winding.device)
    selected_idxs = np.random.choice(len(abs_pcls), num_pcls_per_step, replace=False)
    selected_pcls = [abs_pcls[i] for i in selected_idxs]

    for pcl in selected_pcls:
        # An absolute-winding pcl's attached points, flattened across its patches.
        attached = [p for pts in pcl['points_by_patch'].values() for p in pts]
        if not attached:
            continue
        num_points_for_pcl = min(len(attached), cfg['sample_count_absolute_winding_points_per_pcl'])
        chosen = np.random.choice(len(attached), num_points_for_pcl, replace=False)
        for idx in chosen:
            p = attached[idx]
            pid = p['on_patch']['id']
            i, j = int(p['on_patch']['ij'][0]), int(p['on_patch']['ij'][1])
            strip_requests.append((
                (pid, i, j), pid, p['winding_annotation'],
                p['_theta_node_id'],
            ))

    sampled_l_shapes = _sample_l_shapes_batch(
        patches_dict,
        patch_atlas,
        [entry[0] for entry in strip_requests],
        num_points_per_strip,
        cfg,
    )
    for entry, ls in zip(strip_requests, sampled_l_shapes):
        if ls is not None:
            _set_walk_reference_node(ls, entry[3])
            strips.append((ls, entry[1], entry[2]))

    if not strips:
        return torch.zeros([], device=dr_per_winding.device)

    # Flatten: 4 strips per annotated point.
    total_strips = len(strips) * num_strips_per_point
    flat_ijs = np.empty([total_strips, num_points_per_strip, 2], dtype=np.float32)
    flat_pids = []
    flat_walks = []
    for k, (ls, pid, _) in enumerate(strips):
        base = k * num_strips_per_point
        for s, strip in enumerate(ls):
            flat_ijs[base + s] = strip.ijs
            flat_walks.append(strip)
        flat_pids.extend([pid] * num_strips_per_point)

    # Batched bilinear gather on the device-resident atlas.
    patch_idx_per_strip_np = np.fromiter(
        (patch_atlas.id_to_idx[pid] for pid in flat_pids),
        dtype=np.int64,
        count=total_strips,
    )
    patch_idx_per_strip = torch.from_numpy(patch_idx_per_strip_np)
    patch_idx_per_sample = patch_idx_per_strip[:, None].expand(total_strips, num_points_per_strip)
    flat_zyxs = patch_atlas.lookup(patch_idx_per_sample, torch.from_numpy(flat_ijs))

    # Mask out strip samples whose z falls outside [z_begin - margin, z_end + margin).
    # Computed before applying cached crossing adjustments but masked afterward.
    z_margin = cfg['patch_loss_z_margin']
    z_mask = (flat_zyxs[..., 0] >= z_begin - z_margin) & (flat_zyxs[..., 0] < z_end + z_margin)

    flat_spiral = slice_to_spiral_transform(flat_zyxs.reshape(-1, 3)).reshape(*flat_zyxs.shape)
    theta, _, shifted_radii = get_theta_and_radii(flat_spiral[..., 1:], dr_per_winding)
    packed_walks = _pack_walks(
        [patch_walk.walk for patch_walk in flat_walks], crossing_map)
    shifted_radii, crossing_adjustments = _unwrap_sampled_tracks(
        crossing_map, dr_per_winding, theta, shifted_radii, packed_walks)

    # [num_points, 4, num_points_per_strip] -> pool each point's 4 strips into one set.
    num_samples_per_point = num_strips_per_point * num_points_per_strip
    shifted_radii = shifted_radii.reshape(len(strips), num_samples_per_point)
    mask = z_mask.reshape(len(strips), num_samples_per_point)

    winding_annotations = torch.tensor(
        [s[2] for s in strips],
        device=dr_per_winding.device,
        dtype=torch.float32,
    )
    target_shifted = (winding_annotations * dr_per_winding)[:, None]

    err = (shifted_radii - target_shifted).abs()
    target_shifted_per_strip = target_shifted[:, None, :].expand(
        -1, num_strips_per_point, num_points_per_strip,
    ).reshape(total_strips, num_points_per_strip)
    target_radii = radius_from_unwrapped_shifted(
        theta, target_shifted_per_strip, crossing_adjustments,
        dr_per_winding,
    )
    target_spiral = torch.stack([
        flat_spiral[..., 0],
        torch.sin(theta) * target_radii,
        torch.cos(theta) * target_radii,
    ], dim=-1).detach()
    record_loss_samples(
        'abs_winding',
        flat_spiral.reshape(len(strips), num_strips_per_point,
                            num_points_per_strip, 3),
        err.reshape(len(strips), num_strips_per_point, num_points_per_strip),
        mask.reshape(len(strips), num_strips_per_point, num_points_per_strip),
        display_spiral_zyx=target_spiral.reshape(
            len(strips), num_strips_per_point, num_points_per_strip, 3),
    )
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



def sample_spiral_surface_frame(dr_per_winding, outer_winding_idx, num_points, z_begin, z_end):
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



def iter_lasagna_losses(slice_to_spiral_transform, dr_per_winding, lasagna_volume, outer_winding_idx, num_points, epsilon=None, compute_spacing=True, *, cfg, z_begin, z_end):
    # Sample points uniformly over the spiral cylinder (a disk of radius
    # dr_per_winding * outer_winding_idx in spiral yx, over the z-ROI). Two losses are computed:
    #   (normals) the spiral radial covector at each sample is pulled back to scroll space via
    #             central-difference J^T (a normal is a covector, not a finite-length displacement)
    #             and matched in direction to the precomputed nx/ny scroll-space normal.
    #   (spacing) [the legacy dense_spacing_mode='grad_mag' objective, retained
    #             unchanged for comparison/rollback; the production mode is the
    #             'phase' bundle in sdt_losses.py, and compute_spacing=False skips
    #             this entirely] at each sample, shift inward and outward by dr_per_winding/2
    #             along the spiral radial direction (so the two endpoints span exactly one
    #             winding in spiral space), map both endpoints to scroll space, and
    #             integrate the winding-density field (grad_mag, windings per voxel) along
    #             the scroll-space segment between them. grad_mag is a density, not a
    #             distance, so the number of windings the segment actually crosses is the
    #             line integral of that density along it; for a correct fit the integral
    #             equals 1 (one winding). The density is decoded from grad_mag in windings
    #             per full-resolution voxel.
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if lasagna_volume is None or outer_winding_idx is None:
        if compute_spacing:
            yield 'dense_spacing', zero
        yield 'dense_normals', zero
        return

    backend = lasagna_volume.get('backend', 'dense_test')
    volume = lasagna_volume.get('volume')  # dense: 3 (nx, ny, grad_mag), z, y, x uint8
    z_size, y_size, x_size = lasagna_volume['shape']
    z_origin = lasagna_volume['z_origin']
    y_origin = lasagna_volume.get('y_origin', 0)
    x_origin = lasagna_volume.get('x_origin', 0)
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
    yi = sample_zyx[:, 1] - y_origin
    xi = sample_zyx[:, 2] - x_origin
    in_bounds = (zi >= 0) & (zi < z_size) & (yi >= 0) & (yi < y_size) & (xi >= 0) & (xi < x_size)
    zi = zi.clamp(0, z_size - 1)
    yi = yi.clamp(0, y_size - 1)
    xi = xi.clamp(0, x_size - 1)

    # Build both sparse requests before touching the shared CUDA cache.
    if compute_spacing:
        density_decode = cfg['dense_grad_mag_factor'] / cfg['dense_grad_mag_encode_scale'] * lasagna_scale
        num_steps = int(cfg['dense_spacing_integration_steps'])
        step_frac = (torch.arange(num_steps, device=device).float() + 0.5) / num_steps
        integration_zyx = scroll_inner[:, None, :] + step_frac[None, :, None] * scroll_displacement[:, None, :]
        int_idx = (integration_zyx.detach() / lasagna_scale).round().long()
        izi = int_idx[..., 0] - z_origin
        iyi = int_idx[..., 1] - y_origin
        ixi = int_idx[..., 2] - x_origin
        int_in_bounds = (izi >= 0) & (izi < z_size) & (iyi >= 0) & (iyi < y_size) & (ixi >= 0) & (ixi < x_size)
        izi = izi.clamp(0, z_size - 1)
        iyi = iyi.clamp(0, y_size - 1)
        ixi = ixi.clamp(0, x_size - 1)
    else:
        integration_zyx = None

    if backend == 'sparse_cuda':
        normal_indices = torch.stack([zi, yi, xi], dim=-1)
        if compute_spacing:
            grad_indices = torch.stack([izi, iyi, ixi], dim=-1)
        else:
            grad_indices = torch.zeros([0, 3], dtype=torch.int64, device=device)
        normal_u8, grad_mag_u8 = lasagna_volume['store'].gather_pair(
            normal_indices, grad_indices, device)
        nx_u8, ny_u8 = normal_u8.unbind(dim=-1)
        if compute_spacing:
            grad_mag_u8 = grad_mag_u8.reshape(izi.shape)
    elif backend in ('dense', 'dense_test'):
        nx_u8 = volume[0, zi, yi, xi]
        ny_u8 = volume[1, zi, yi, xi]
        grad_mag_u8 = volume[2, izi, iyi, ixi] if compute_spacing else None
    else:
        raise ValueError(f'unsupported lasagna backend {backend!r}')
    normal_weight = (((nx_u8 != 0) | (ny_u8 != 0)) & in_bounds).float()
    nx = _decode_uint8_normal_component(nx_u8.float())
    ny = _decode_uint8_normal_component(ny_u8.float())
    nz = torch.sqrt((1. - nx * nx - ny * ny).clamp(min=0.))
    target_normal = F.normalize(torch.stack([nz, ny, nx], dim=-1), dim=-1)  # zyx

    if compute_spacing:
        # grad_mag encodes a winding density (windings per base-volume voxel); the decode factor below
        # also rescales it to current-grid windings/voxel. The number of windings actually crossed by
        # the one-winding scroll-space segment (scroll_inner -> scroll_outer) is the line integral of
        # this density along it, so we sample the density at evenly spaced midpoints along the segment
        # and accumulate density * dl (a midpoint Riemann sum). For a correct fit the integral equals 1.
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
        record_loss_samples('dense_spacing', spiral_zyx, spacing_residual,
                            spacing_weight.bool())

    scroll_center_detached = scroll_center.detach()
    spiral_zyx_detached = spiral_zyx.detach()
    if compute_spacing:
        yield 'dense_spacing', spacing_loss
        del spacing_loss

    # The caller has released the endpoint/integration graph.  Dense normals
    # use detached sample positions and build their own finite-difference graph,
    # so the two large transform graphs never need to coexist.
    del scroll_samples, scroll_inner, scroll_outer, scroll_center
    del scroll_displacement, scroll_segment_length, integration_zyx
    scroll_normal = get_radial_normal_in_scroll_space(
        slice_to_spiral_transform,
        scroll_center_detached,
        spiral_zyx=spiral_zyx_detached,
        epsilon=epsilon,
    )
    normals_residual = 1. - (scroll_normal * target_normal).sum(dim=-1).abs()
    normals_loss = (normals_residual * normal_weight).sum() / normal_weight.sum().clamp(min=1)
    record_loss_samples('dense_normals', spiral_zyx_detached, normals_residual,
                        normal_weight.bool())
    yield 'dense_normals', normals_loss



def _sample_component_walk(members, edges, strip_lengths, branch_probability):
    """Sample a chain walk through a link component.

    members are the component's strip indices; edges its junctions as
    (strip_a, pos_a, strip_b, pos_b); strip_lengths maps strip index -> point
    count. Starting from a random member end, walk along the strip; at each
    junction passed, hop onto the linked strip (at its junction position, in a
    random direction) with branch_probability, never revisiting a strip.
    Returns ordered segments [(strip, pos_from, pos_to)] (inclusive, pos_from >
    pos_to when walking backwards); consecutive segments meet at a junction,
    whose two nearly-coincident endpoints appear as consecutive walk points, so
    the crossing map treats the hop like any other registered edge."""
    junctions = {s: [] for s in members}
    for strip_a, pos_a, strip_b, pos_b in edges:
        junctions[strip_a].append((pos_a, strip_b, pos_b))
        junctions[strip_b].append((pos_b, strip_a, pos_a))
    strip = members[np.random.randint(len(members))]
    direction = 1 if np.random.rand() < 0.5 else -1
    pos = 0 if direction == 1 else strip_lengths[strip] - 1
    visited = {strip}
    segments = []
    while True:
        ahead = [(p, other, other_pos) for p, other, other_pos in junctions[strip]
                 if other not in visited
                 and (p >= pos if direction == 1 else p <= pos)]
        ahead.sort(key=lambda t: t[0], reverse=direction == -1)
        hopped = False
        for p, other, other_pos in ahead:
            if np.random.rand() < branch_probability:
                segments.append((strip, pos, p))
                visited.add(other)
                strip, pos = other, other_pos
                direction = 1 if np.random.rand() < 0.5 else -1
                hopped = True
                break
        if not hopped:
            end = strip_lengths[strip] - 1 if direction == 1 else 0
            segments.append((strip, pos, end))
            return segments


def get_unattached_pcl_strip_losses(
    slice_to_spiral_transform,
    dr_per_winding,
    pcl_strips,
    component_strip_lists,
    component_edges,
    sampling_strata,
    get_or_build_unattached_pcl_flat,
    num_pcls_per_step,
    num_points_per_pcl,
    compute_dt,
    dt_max_winding=None,
    dt_target_cache=None,
    *,
    crossing_map,
    cfg,
):
    # Unattached pcls are treated as ordered strips, indexed by int(point_id), and
    # assumed to be locally dense enough that adjacent STRIP points have
    # |dtheta| < pi. The per-row samples themselves may be far sparser than that
    # (a fiber spanning several windings sampled at num_points_per_pcl points),
    # so theta=0 crossings are gathered along the complete cached node walk and
    # applied to the sampled points. Two losses are computed, analogous to the patch radius
    # and DT losses: (1) shifted-radius should be constant along the strip after
    # subtracting per-point winding-annotation offsets; (2) each point should snap to
    # its target winding, with the target taken from the snapped strip median (or,
    # when dt_target_cache is given, the cached whole-strip quantile target from
    # dt_targets.py, transferred into this sample's unwrap frame through the cached
    # point nearest a sampled point by within-strip index).
    #
    # Graph awareness (cross-fiber links): sampling_strata indexes *components* --
    # groups of strips joined by same-winding links (component_strip_lists gives
    # each component's member strip indices, component_edges its junctions as
    # (strip_a, pos_a, strip_b, pos_b)). Each chosen component contributes one row
    # sampled along a chain *walk* through its strips (_sample_component_walk):
    # along a strip, hopping to the linked strip at a junction with
    # cfg['loss_fiber_link_branch_probability'], so a junction hop is an ordinary
    # registered step and cached crossings continue through it. The constant-shifted-radius target (1) along the
    # walk then pulls points on either side of every traversed junction onto one
    # shared winding; over steps, random walks cover all of a component's
    # junctions. Rows mix strips, so the DT snap (2) passes per-point strip
    # indices to the cache lookup. A singleton component reduces exactly to the
    # legacy per-strip row.
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if not pcl_strips:
        return zero, zero

    num_to_sample = min(num_pcls_per_step, sampling_strata['effective_size'])
    if num_to_sample <= 0:
        return zero, zero
    chosen_comps = _choose_pcl_indices(sampling_strata, num_to_sample, cfg)

    flat = get_or_build_unattached_pcl_flat(pcl_strips, device)
    if flat is None or flat['total'] == 0:
        return zero, zero

    branch_probability = cfg['loss_fiber_link_branch_probability']
    num_rows = len(chosen_comps)
    starts_cpu = flat['starts_cpu'].numpy()
    sampled_strip_indices = np.empty([num_rows, num_points_per_pcl], dtype=np.int64)
    sampled_local_indices = np.empty([num_rows, num_points_per_pcl], dtype=np.int64)
    sampled_flat_indices = np.empty([num_rows, num_points_per_pcl], dtype=np.int64)
    walks = []
    for k, comp_idx in enumerate(chosen_comps):
        members = component_strip_lists[comp_idx]
        edges = component_edges[comp_idx]
        if len(members) == 1 or not edges:
            strip_idx = members[np.random.randint(len(members))]
            segments = [(strip_idx, 0, len(pcl_strips[strip_idx]['zyxs']) - 1)]
        else:
            segments = _sample_component_walk(
                members, edges,
                {s: len(pcl_strips[s]['zyxs']) for s in members},
                branch_probability,
            )
        walk_strips = np.concatenate([
            np.full(abs(pos_to - pos_from) + 1, strip_idx, dtype=np.int64)
            for strip_idx, pos_from, pos_to in segments])
        walk_locals = np.concatenate([
            np.arange(pos_from, pos_to + 1, dtype=np.int64) if pos_from <= pos_to
            else np.arange(pos_from, pos_to - 1, -1, dtype=np.int64)
            for strip_idx, pos_from, pos_to in segments])
        walk_len = len(walk_locals)
        picks = np.sort(np.random.choice(
            walk_len, num_points_per_pcl, replace=num_points_per_pcl > walk_len))
        sampled_strip_indices[k] = walk_strips[picks]
        sampled_local_indices[k] = walk_locals[picks]
        sampled_flat_indices[k] = starts_cpu[sampled_strip_indices[k]] + sampled_local_indices[k]
        node_path = np.concatenate([
            pcl_strips[strip_idx]['_theta_node_ids'][np.arange(
                pos_from, pos_to + (1 if pos_from <= pos_to else -1),
                1 if pos_from <= pos_to else -1)]
            for strip_idx, pos_from, pos_to in segments
        ])
        walks.append(SampledWalk(
            node_ids=node_path,
            pick_positions=picks,
            connect_fractional_picks=False,
        ))

    sampled_flat_indices_t = torch.from_numpy(sampled_flat_indices).to(device=device)
    zyxs_t = flat['zyxs'][sampled_flat_indices_t]
    winding_t = flat['windings'][sampled_flat_indices_t]

    packed_walks = _pack_walks(walks, crossing_map)

    spiral_zyxs = slice_to_spiral_transform(zyxs_t.reshape(-1, 3)).reshape(*zyxs_t.shape)
    theta, _, shifted_radii = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
    shifted_radii, crossing_adjustments = _unwrap_sampled_tracks(
        crossing_map, dr_per_winding, theta, shifted_radii, packed_walks,
    )

    # Normalise so a pcl with mixed annotations still reads as a single 'strip'.
    normalised_radii = shifted_radii - winding_t * dr_per_winding

    radius_hinge_margin = dr_per_winding.detach() * cfg['patch_radius_loss_margin']
    dt_hinge_margin = dr_per_winding.detach() * cfg['patch_dt_loss_margin']

    mean_radii = normalised_radii.mean(dim=-1, keepdim=True)
    radius_deviations = (normalised_radii - mean_radii).abs()
    radius_point_residuals = F.relu(radius_deviations - radius_hinge_margin)
    radius_loss = radius_point_residuals.mean()
    if diagnostics_enabled():
        radius_target_shifted = mean_radii + winding_t * dr_per_winding
        radius_target_radii = radius_from_unwrapped_shifted(
            theta, radius_target_shifted, crossing_adjustments,
            dr_per_winding,
        )
        radius_target_spiral_zyxs = torch.stack([
            spiral_zyxs[..., 0],
            torch.sin(theta) * radius_target_radii,
            torch.cos(theta) * radius_target_radii,
        ], dim=-1).detach()
        record_loss_samples(
            'unattached_pcl_radius', spiral_zyxs,
            radius_point_residuals,
            display_spiral_zyx=radius_target_spiral_zyxs,
        )

    if not compute_dt:
        return radius_loss, zero

    # Per-point strip indices: a walk row can span several strips, each with its
    # own cache entry; the snap anchors the row on its best valid (point, cache)
    # pair and takes that strip's cached target (the component is same-winding, so
    # any member's target names the same winding).
    target_normalised = strip_dt_target_in_sample_frame(
        normalised_radii, sampled_local_indices, theta, crossing_adjustments,
        dr_per_winding, dt_target_cache, sampled_strip_indices,
    )
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
    record_loss_samples(
        'unattached_pcl_dt', spiral_zyxs, point_distances,
        active_mask[..., None] if active_mask is not None else None,
        display_spiral_zyx=target_spiral_zyxs,
    )

    return radius_loss, dt_loss



def get_symmetric_dirichlet_loss(slice_to_spiral_transform, dr_per_winding, outer_winding_idx, num_points, epsilon=None, *, cfg, z_begin, z_end):
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
        epsilon = cfg['model_sym_dirichlet_finite_difference_epsilon']

    spiral_zyx, e1, e2 = sample_spiral_surface_frame(dr_per_winding, outer_winding_idx, num_points, z_begin, z_end)

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
    record_loss_samples('sym_dirichlet', spiral_zyx, energy)
    return energy.mean()
