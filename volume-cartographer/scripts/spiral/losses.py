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



def _sample_patch_points(patch_indices, cap, rng, patch_atlas):
    """Sample patch quads through the required vc sampling binding."""
    sampling_atlas = getattr(patch_atlas, 'sampling_atlas', None)
    if sampling_atlas is None or not hasattr(sampling_atlas, 'sample_patch_points'):
        raise RuntimeError(
            'Patch sampling requires '
            'vc.spiral_sampling.PatchSamplingAtlas.sample_patch_points; '
            'rebuild and install the vc_spiral_sampling extension')
    seed = int(rng.randint(0, np.iinfo(np.int64).max))
    sampled = sampling_atlas.sample_patch_points(
        np.ascontiguousarray(patch_indices, dtype=np.int64), cap, seed)
    return (
        np.asarray(sampled['ijs'], dtype=np.float32),
        np.asarray(sampled['counts'], dtype=np.int64),
        np.asarray(sampled['node_ordinals'], dtype=np.int64)
        if 'node_ordinals' in sampled else None,
    )




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
class PackedWalks:
    """Walks resolved to cached canonical edges, with no XYZ payload."""

    edge_ids: torch.Tensor
    directions: torch.Tensor
    edge_valid: torch.Tensor
    pick_positions: torch.Tensor
    correction_node_ids: torch.Tensor
    walk_start_node_ids: torch.Tensor
    reference_node_ids: torch.Tensor

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

    edge_ids_np = np.zeros(edge_valid_np.shape, dtype=np.int64)
    directions_np = np.ones(edge_valid_np.shape, dtype=np.int8)
    if max_walk_len > 1:
        pairs_np = np.stack(
            [node_ids_np[:, :-1], node_ids_np[:, 1:]], axis=-1)
        resolved, resolved_dir = crossing_map.resolve_edges(
            pairs_np[edge_valid_np])
        edge_ids_np[edge_valid_np] = resolved.numpy()
        directions_np[edge_valid_np] = resolved_dir.numpy()

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

    edge_ids = torch.as_tensor(
        edge_ids_np, dtype=torch.int64, device=device)
    directions = torch.as_tensor(
        directions_np, dtype=torch.int8, device=device)
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
                        point_cap, cfg, patch_atlas=None, crossing_map=None):
    """Return padded uniform 2D samples and their validity/topology metadata."""
    if num_to_sample <= 0:
        raise ValueError('Expected at least one patch index')

    def build(rng):
        patch_indices = rng.choice(len(patches), num_to_sample,
                                   p=sampling_probabilities, replace=True)
        ijs_np, counts_np, node_ordinals_np = _sample_patch_points(
            patch_indices, point_cap, rng, patch_atlas)
        row_indices = np.broadcast_to(
            np.asarray(patch_indices, dtype=np.int64)[:, None],
            (num_to_sample, point_cap),
        )
        node_ids_np = (
            patch_atlas.theta_node_ids_from_ordinals(node_ordinals_np)
            if (node_ordinals_np is not None
                and hasattr(patch_atlas, 'theta_node_ids_from_ordinals'))
            else patch_atlas.theta_node_ids(row_indices, ijs_np)
        )
        ijs_cpu = torch.from_numpy(ijs_np)
        idx_cpu = torch.from_numpy(
            np.ascontiguousarray(patch_indices, dtype=np.int64))
        node_ids_cpu = torch.from_numpy(
            np.ascontiguousarray(node_ids_np, dtype=np.int64))
        counts_cpu = torch.from_numpy(
            np.ascontiguousarray(counts_np, dtype=np.int64))
        sample_mask_cpu = (
            torch.arange(point_cap)[None, :] < counts_cpu[:, None])
        slice_zyxs_gpu = patch_atlas.lookup(
            idx_cpu[:, None].expand(num_to_sample, point_cap), ijs_cpu)
        target_device = patch_atlas.device
        ijs_gpu = ijs_cpu.to(device=target_device, non_blocking=True)
        idx_gpu = idx_cpu.to(device=target_device, non_blocking=True)
        node_ids_gpu = node_ids_cpu.to(device=target_device, non_blocking=True)
        sample_mask_gpu = sample_mask_cpu.to(
            device=target_device, non_blocking=True)
        return (ijs_gpu, idx_gpu, slice_zyxs_gpu, node_ids_gpu,
                sample_mask_gpu)

    if prefetch.prefetch_enabled() and torch.cuda.is_available():
        pf = prefetch.get_prefetcher()
        rng = pf.np_rng(key)
        return pf.pop_or_run((key, id(crossing_map), num_to_sample,
                              point_cap),
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
    # The bilinear atlas gather already ran on the CPU at batch-build time
    # (see _sample_patch_batch); the batch carries the interpolated points.
    (combined_ijs_gpu, patch_indices_gpu, all_slice_zyxs,
     sample_node_ids, sample_mask) = batch

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
    all_crossing_adjustments = crossing_map.adjustments_from_potentials(
        sample_node_ids, all_theta, dr_per_winding)
    all_shifted_radii = all_shifted_radii + all_crossing_adjustments

    return (
        combined_ijs_gpu,
        all_slice_zyxs,
        all_spiral_zyxs,
        all_theta,
        all_shifted_radii,
        all_crossing_adjustments,
        sample_mask,
        extra_spiral,
    )




def _patch_radius_and_dt_losses(
    slice_to_spiral_transform, dr_per_winding,
    all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii,
    all_crossing_adjustments,
    num_patches_for_radius, num_patches_for_dt, compute_dt, dt_max_winding,
    radius_loss_margin, radius_loss_inv, radius_within_norm_p,
    dt_loss_margin, dt_norm_p, dt_within_patch_norm_p,
    patch_indices=None, sample_ijs=None, dt_target_cache=None, sample_mask=None,
    diagnostic_prefix='patch',
):
    # Shared radius + DT patch losses, operating on padded uniform 2D samples
    # (all_*; see _sample_patch_tracks). Pulled out of get_patch_and_umbilicus_losses so the
    # same loss can serve both the verified and the untrusted ('unverified') patch sets with
    # independent hyperparameters. Returns (mean_radius_deviation, patch_dt_loss).
    # `dt_target_cache` is the whole-object DT target cache (see dt_targets.py) or None
    # in legacy strip-median mode; `patch_indices` maps the sampled tracks to cache rows.
    radius_hinge_margin = dr_per_winding.detach() * radius_loss_margin
    dt_hinge_margin = dr_per_winding.detach() * dt_loss_margin
    if sample_mask is None:
        sample_mask = torch.ones_like(all_shifted_radii, dtype=torch.bool)

    radius_shifted_radii = all_shifted_radii[:num_patches_for_radius]
    radius_slice_zyxs = all_slice_zyxs[:num_patches_for_radius]
    radius_spiral_zyxs = all_spiral_zyxs[:num_patches_for_radius]
    radius_theta = all_theta[:num_patches_for_radius]
    radius_crossing_adjustments = all_crossing_adjustments[:num_patches_for_radius]
    radius_mask = sample_mask[:num_patches_for_radius]
    radius_counts = radius_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    mean_shifted_radii = (
        (radius_shifted_radii * radius_mask).sum(dim=-1, keepdim=True)
        / radius_counts)
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
        mean_radius_deviation = _masked_mean(
            radius_point_residuals, radius_mask)
        record_loss_samples(
            f'{diagnostic_prefix}_radius', radius_spiral_zyxs,
            radius_point_residuals, radius_mask,
            display_spiral_zyx=radius_target_spiral_zyxs,
        )
    else:
        # Penalise deviation from the track's mean shifted-radius directly in spiral space.
        radius_deviations = (radius_shifted_radii - mean_shifted_radii).abs()
        radius_deviations_hinge = F.relu(radius_deviations - radius_hinge_margin)
        if radius_within_norm_p == 1.0:
            mean_radius_deviation = _masked_mean(
                radius_deviations_hinge, radius_mask)
        else:
            d = radius_deviations_hinge + 1.e-5
            per_track = (
                ((d ** radius_within_norm_p) * radius_mask).sum(dim=-1)
                / radius_counts.squeeze(-1)
            ) ** (1.0 / radius_within_norm_p)
            mean_radius_deviation = per_track.mean()
        record_loss_samples(
            f'{diagnostic_prefix}_radius', radius_spiral_zyxs,
            radius_deviations_hinge, radius_mask,
            display_spiral_zyx=radius_target_spiral_zyxs,
        )

    if compute_dt:
        dt_slice_zyxs = all_slice_zyxs[:num_patches_for_dt]
        dt_spiral_zyxs = all_spiral_zyxs[:num_patches_for_dt]
        dt_theta = all_theta[:num_patches_for_dt]
        dt_shifted_radii = all_shifted_radii[:num_patches_for_dt]
        dt_crossing_adjustments = all_crossing_adjustments[:num_patches_for_dt]
        dt_mask = sample_mask[:num_patches_for_dt]

        # Define the DT target winding (see patch_dt_target_in_sample_frame: whole-patch cached
        # target when available, else the track's own snapped median). Every sampled
        # point on the track is then pulled towards that target winding.
        target_shifted_radii = patch_dt_target_in_sample_frame(
            dt_shifted_radii,
            sample_ijs[:num_patches_for_dt] if sample_ijs is not None else None,
            dt_theta,
            dt_crossing_adjustments,
            dr_per_winding,
            dt_target_cache,
            patch_indices[:num_patches_for_dt] if patch_indices is not None else None,
            sample_mask=dt_mask,
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
        dt_counts = dt_mask.sum(dim=-1).clamp(min=1)
        track_losses = (
            ((point_distances ** dt_within_patch_norm_p) * dt_mask).sum(dim=-1)
            / dt_counts
        ) ** (1 / dt_within_patch_norm_p)
        # Progressive DT: only patches whose snapped winding is within the current cutoff contribute.
        active_mask = _progressive_dt_active_mask(target_shifted_radii.squeeze(-1), dr_per_winding, dt_max_winding)
        patch_dt_loss = _aggregate_dt_track_losses(track_losses, dt_norm_p, active_mask)
        diagnostic_mask = dt_mask
        if active_mask is not None:
            diagnostic_mask = diagnostic_mask & active_mask[..., None]
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
            num_patches_to_sample, cfg['sample_count_points_per_patch'],
            cfg, patch_atlas, crossing_map)

        (
            sample_ijs,
            all_slice_zyxs,
            all_spiral_zyxs,
            all_theta,
            all_shifted_radii,
            all_crossing_adjustments,
            sample_mask,
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
            sample_mask=sample_mask,
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
        num_patches_to_sample, cfg['sample_count_unverified_points_per_patch'],
        cfg, patch_atlas, crossing_map)

    (
        sample_ijs,
        all_slice_zyxs,
        all_spiral_zyxs,
        all_theta,
        all_shifted_radii,
        all_crossing_adjustments,
        sample_mask,
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
        sample_mask=sample_mask,
        diagnostic_prefix='unverified_patch',
    )



def _pcl_chain_seam_adjustments(crossing_map, dr_per_winding, chain_node_ids):
    values = []
    for node_ids in chain_node_ids:
        edge_ids, directions = crossing_map.resolve_walks(node_ids)
        edge_ids = edge_ids.to(crossing_map.device)
        directions = directions.to(crossing_map.device)
        winding_steps = (
            crossing_map.crossings[edge_ids]
            * directions.to(crossing_map.crossings.dtype))
        values.append(winding_steps.to(torch.int32).sum())
    return torch.stack(values).to(dr_per_winding.dtype) * dr_per_winding.detach()


def _valid_patch_annotation(patches_dict, patch_atlas, pid, i, j):
    """Resolve an annotation to its retained valid quad and theta node."""
    patch = patches_dict[pid]
    mask = patch._sampling_valid_quad_mask_np
    i_q = min(max(int(i), 0), mask.shape[0] - 1)
    j_q = min(max(int(j), 0), mask.shape[1] - 1)
    if not mask[i_q, j_q]:
        return None
    patch_idx = patch_atlas.id_to_idx[pid]
    node_id = patch_atlas.theta_node_ids(
        np.asarray([patch_idx], dtype=np.int64),
        np.asarray([[i_q, j_q]], dtype=np.float32),
    )[0]
    return patch_idx, int(node_id)


def _sample_requested_patch_rows(patch_indices, point_cap, patch_atlas):
    """Sample already-validated annotation patch rows without replacement."""
    patch_indices = np.asarray(patch_indices, dtype=np.int64)
    ijs_np, counts_np, node_ordinals_np = _sample_patch_points(
        patch_indices, point_cap, prefetch.LegacyNumpyRandom, patch_atlas)
    row_patch_indices = np.broadcast_to(
        patch_indices[:, None], (len(patch_indices), point_cap))
    node_ids_np = (
        patch_atlas.theta_node_ids_from_ordinals(node_ordinals_np)
        if (node_ordinals_np is not None
            and hasattr(patch_atlas, 'theta_node_ids_from_ordinals'))
        else patch_atlas.theta_node_ids(row_patch_indices, ijs_np)
    )
    ijs = torch.from_numpy(ijs_np).to(patch_atlas.device)
    patch_indices_t = torch.from_numpy(patch_indices).to(patch_atlas.device)
    zyxs = patch_atlas.lookup(
        patch_indices_t[:, None].expand(-1, point_cap), ijs)
    node_ids = torch.from_numpy(
        np.ascontiguousarray(node_ids_np, dtype=np.int64)).to(
            patch_atlas.device)
    counts = torch.from_numpy(counts_np).to(patch_atlas.device)
    mask = torch.arange(point_cap, device=patch_atlas.device)[None, :] < counts[:, None]
    return ijs, zyxs, node_ids, mask


def get_patch_rel_winding_loss(slice_to_spiral_transform, dr_per_winding,
                               patches_dict, patch_atlas, point_collections,
                               sampling_strata, *, crossing_map, cfg,
                               z_begin, z_end):
    """Relative winding supervision over unordered uniform patch samples."""
    point_cap = cfg['sample_count_points_per_patch']
    num_pcls = min(
        cfg['sample_count_relative_winding_pcls'],
        sampling_strata['effective_size'])
    if num_pcls <= 0:
        return torch.zeros([], device=dr_per_winding.device)

    rows = []
    for pcl_idx in _choose_pcl_indices(sampling_strata, num_pcls, cfg):
        pcl = point_collections[pcl_idx]
        patch_ids = list(pcl['points_by_patch'])
        if cfg['pcl_rel_winding_adjacent_patches_only']:
            candidates = list(zip(patch_ids, patch_ids[1:]))
        else:
            candidates = list(itertools.combinations(patch_ids, 2))
        pair_count = min(
            len(candidates),
            cfg['sample_count_relative_winding_patch_pairs_per_pcl'])
        if pair_count <= 0:
            continue
        for pair_idx in np.random.choice(
                len(candidates), pair_count, replace=False):
            pid1, pid2 = candidates[pair_idx]
            p1s = pcl['points_by_patch'][pid1]
            p2s = pcl['points_by_patch'][pid2]
            p1 = p1s[np.random.randint(len(p1s))]
            p2 = p2s[np.random.randint(len(p2s))]
            ij1 = p1['on_patch']['ij']
            ij2 = p2['on_patch']['ij']
            attached1 = _valid_patch_annotation(
                patches_dict, patch_atlas, pid1, ij1[0], ij1[1])
            attached2 = _valid_patch_annotation(
                patches_dict, patch_atlas, pid2, ij2[0], ij2[1])
            if attached1 is None or attached2 is None:
                continue
            chain_nodes = np.fromiter(
                (point['_theta_node_id'] for point in
                 pcl['chain'].points_between(p1, p2)), dtype=np.int64)
            rows.append({
                'patch_indices': (attached1[0], attached2[0]),
                'patch_nodes': (attached1[1], attached2[1]),
                'reference_nodes': (
                    int(p1['_theta_node_id']), int(p2['_theta_node_id'])),
                'winding_diff': p2['winding_annotation'] - p1['winding_annotation'],
                'chain_nodes': chain_nodes,
            })
    if not rows:
        return torch.zeros([], device=dr_per_winding.device)

    flat_patch_indices = np.asarray(
        [idx for row in rows for idx in row['patch_indices']], dtype=np.int64)
    _, flat_zyxs, flat_node_ids, flat_mask = _sample_requested_patch_rows(
        flat_patch_indices, point_cap, patch_atlas)
    flat_spiral = slice_to_spiral_transform(
        flat_zyxs.reshape(-1, 3)).reshape(*flat_zyxs.shape)
    theta, _, shifted = get_theta_and_radii(
        flat_spiral[..., 1:], dr_per_winding)
    reference_nodes = torch.as_tensor(
        [node for row in rows for node in row['reference_nodes']],
        dtype=torch.int64, device=dr_per_winding.device)
    patch_nodes = torch.as_tensor(
        [node for row in rows for node in row['patch_nodes']],
        dtype=torch.int64, device=dr_per_winding.device)
    adjustments = crossing_map.adjustments_from_potentials(
        flat_node_ids, theta, dr_per_winding,
        reference_node_ids=reference_nodes,
        reference_patch_node_ids=patch_nodes)
    shifted = (shifted + adjustments).reshape(len(rows), 2, point_cap)
    mask = flat_mask.reshape(len(rows), 2, point_cap)
    z_margin = cfg['patch_loss_z_margin']
    z_mask = (
        (flat_zyxs[..., 0] >= z_begin - z_margin)
        & (flat_zyxs[..., 0] < z_end + z_margin)
    ).reshape(len(rows), 2, point_cap)
    mask = mask & z_mask
    expected = torch.as_tensor(
        [row['winding_diff'] for row in rows],
        dtype=dr_per_winding.dtype, device=dr_per_winding.device,
    ) * dr_per_winding
    expected -= _pcl_chain_seam_adjustments(
        crossing_map, dr_per_winding,
        [row['chain_nodes'] for row in rows])
    loss = _masked_all_pairs_l1(
        shifted[:, 0], shifted[:, 1], mask[:, 0], mask[:, 1], expected)
    if diagnostics_enabled():
        residual = torch.stack([
            _masked_all_pairs_l1(
                shifted[row:row + 1, 0], shifted[row:row + 1, 1],
                mask[row:row + 1, 0], mask[row:row + 1, 1],
                expected[row:row + 1])
            for row in range(len(rows))
        ])
        record_loss_samples(
            'rel_winding', flat_spiral.reshape(len(rows), 2, point_cap, 3),
            residual[:, None, None], mask)
    return loss


def get_patch_abs_winding_loss(slice_to_spiral_transform, dr_per_winding,
                               patches_dict, patch_atlas, point_collections,
                               *, crossing_map, cfg, z_begin, z_end):
    """Absolute winding supervision over unordered uniform patch samples."""
    abs_pcls = [
        pcl for pcl in point_collections
        if pcl.get('metadata', {}).get('winding_is_absolute', False)]
    num_pcls = min(cfg['sample_count_absolute_winding_pcls'], len(abs_pcls))
    if num_pcls <= 0:
        return torch.zeros([], device=dr_per_winding.device)

    rows = []
    for pcl_idx in np.random.choice(len(abs_pcls), num_pcls, replace=False):
        pcl = abs_pcls[pcl_idx]
        attached = [point for points in pcl['points_by_patch'].values()
                    for point in points]
        point_count = min(
            len(attached), cfg['sample_count_absolute_winding_points_per_pcl'])
        if point_count <= 0:
            continue
        for point_idx in np.random.choice(
                len(attached), point_count, replace=False):
            point = attached[point_idx]
            pid = point['on_patch']['id']
            ij = point['on_patch']['ij']
            resolved = _valid_patch_annotation(
                patches_dict, patch_atlas, pid, ij[0], ij[1])
            if resolved is None:
                continue
            rows.append((
                resolved[0], resolved[1], int(point['_theta_node_id']),
                point['winding_annotation']))
    if not rows:
        return torch.zeros([], device=dr_per_winding.device)

    point_cap = cfg['sample_count_points_per_patch']
    _, zyxs, node_ids, mask = _sample_requested_patch_rows(
        [row[0] for row in rows], point_cap, patch_atlas)
    spiral = slice_to_spiral_transform(
        zyxs.reshape(-1, 3)).reshape(*zyxs.shape)
    theta, _, shifted = get_theta_and_radii(
        spiral[..., 1:], dr_per_winding)
    reference_nodes = torch.as_tensor(
        [row[2] for row in rows], dtype=torch.int64,
        device=dr_per_winding.device)
    patch_nodes = torch.as_tensor(
        [row[1] for row in rows], dtype=torch.int64,
        device=dr_per_winding.device)
    adjustments = crossing_map.adjustments_from_potentials(
        node_ids, theta, dr_per_winding,
        reference_node_ids=reference_nodes,
        reference_patch_node_ids=patch_nodes)
    shifted = shifted + adjustments
    z_margin = cfg['patch_loss_z_margin']
    mask = mask & (zyxs[..., 0] >= z_begin - z_margin) & (
        zyxs[..., 0] < z_end + z_margin)
    target = torch.as_tensor(
        [row[3] for row in rows], dtype=dr_per_winding.dtype,
        device=dr_per_winding.device)[:, None] * dr_per_winding
    error = (shifted - target).abs()

    target_radii = radius_from_unwrapped_shifted(
        theta, target, adjustments, dr_per_winding)
    target_spiral = torch.stack([
        spiral[..., 0], torch.sin(theta) * target_radii,
        torch.cos(theta) * target_radii], dim=-1).detach()
    record_loss_samples(
        'abs_winding', spiral, error, mask,
        display_spiral_zyx=target_spiral)
    return _masked_mean(error, mask)


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
