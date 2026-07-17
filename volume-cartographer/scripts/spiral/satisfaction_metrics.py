import json
import os

import numpy as np
import torch

from sample_spiral import (
    get_theta_and_radii,
    get_theta_crossing_step_adjustments,
    radius_from_unwrapped_shifted,
)
from spiral_helpers import (
    compute_winding_range_and_input_extents,
    save_mesh,
    _segmented_median_per_strip,
    _warn_if_inputs_exceed_flow_bounds,
)
from tracks import get_track_satisfied_counts_in_chunks
from visualization import save_overlay


# Thresholds defining the patch-satisfaction metrics.
metrics_config = {
    'satisfaction_radius_tolerance': 0.45,  # spiral-space, in units of dr_per_winding
    'satisfaction_distance_tolerance': 6.0,  # absolute scan-space distance, in voxels
    'satisfied_patch_quad_fraction': 0.95,  # min fraction of valid quads satisfied for a patch to count as satisfied
    'boundary_satisfied_patch_quad_fraction': 0.90,  # min fraction of boundary quads satisfied for the boundary metric
}


def get_patch_satisfied_areas(slice_to_spiral_transform, dr_per_winding, patches, z_begin, z_end, verbose=False, metrics_overrides=None):
    """Per-patch satisfaction metrics.

    Returns ``(satisfied_patches, satisfied_areas, total_areas, satisfied_quad_masks,
    boundary_satisfied_count, target_winding_idx_per_patch)``: a bool flag per patch
    indicating whether at least ``satisfied_patch_quad_fraction`` of
    its valid quads are satisfied, the satisfied/total area tensors, the per-patch
    (H-1, W-1) bool quad masks, a bool flag per patch indicating whether at least
    ``boundary_satisfied_patch_quad_fraction`` of its boundary quads
    (in-ROI valid quads with at least one 4-neighbor that is out-of-bounds or not
    in-ROI-valid) are satisfied, and the per-patch (H-1, W-1) int64 winding-index
    tensors (the integer output-mesh winding each quad's snap-target sits on; -1 where
    the quad has no target set, e.g. invalid quads or quads in disconnected unwrap
    components).

    For each patch we first find valid quads whose footprint touches the z-ROI. Each
    such quad is then evaluated only at its center point, defined as the mean of its
    four scan-space corners. We (1) take a vertical column at the patch's central valid
    quad-column, (2) snap its median shifted-radius to the nearest integer-winding
    shifted-radius (the "target"), then (3) walk each quad-row outward from that center
    column, unwrapping shifted-radius across theta=0 crossings (signed, so left and
    right work alike). The satisfied area for the patch is patch.area scaled by
    satisfied-quads / valid-quads.

    A quad is satisfied when its center point passes both (a) the spiral-space
    shifted-radius tolerance of `satisfaction_radius_tolerance * dr_per_winding`, and
    (b) the absolute scan-space distance tolerance of
    `satisfaction_distance_tolerance` voxels to the corresponding point on the target
    winding.

    ``metrics_overrides`` optionally overrides individual ``metrics_config`` entries
    for this call only (e.g. looser thresholds for splicing decisions).
    """
    thresholds = dict(metrics_config)
    if metrics_overrides:
        thresholds.update(metrics_overrides)
    spiral_tolerance = dr_per_winding.detach() * thresholds['satisfaction_radius_tolerance']
    scan_tolerance = thresholds['satisfaction_distance_tolerance']
    dr = dr_per_winding.detach()
    device = dr_per_winding.device

    satisfied_patches = torch.ones(len(patches), dtype=torch.bool)
    boundary_satisfied_patches = torch.ones(len(patches), dtype=torch.bool)
    satisfied_areas = torch.zeros(len(patches), dtype=torch.float64)
    total_areas = torch.zeros(len(patches), dtype=torch.float64)
    satisfied_quad_masks = [torch.zeros([max(p.zyxs.shape[0] - 1, 0), max(p.zyxs.shape[1] - 1, 0)], dtype=torch.bool) for p in patches]
    target_winding_idx_per_patch = [torch.full([max(p.zyxs.shape[0] - 1, 0), max(p.zyxs.shape[1] - 1, 0)], -1, dtype=torch.int64) for p in patches]

    with torch.no_grad():
        for patch_index, patch in enumerate(patches):
            patch_zyxs = patch.zyxs.to(device=device, dtype=torch.float32)
            patch_valid_quad_mask_full = patch.valid_quad_mask.to(device=device)
            quad_center_zyxs = (
                patch_zyxs[:-1, :-1]
                + patch_zyxs[1:, :-1]
                + patch_zyxs[:-1, 1:]
                + patch_zyxs[1:, 1:]
            ) / 4
            quad_zs = torch.stack([
                patch_zyxs[:-1, :-1, 0],
                patch_zyxs[1:, :-1, 0],
                patch_zyxs[:-1, 1:, 0],
                patch_zyxs[1:, 1:, 0],
            ], dim=0)
            quad_touches_roi_mask = (quad_zs.amax(dim=0) >= z_begin) & (quad_zs.amin(dim=0) < z_end)
            in_roi_valid_quad_mask = patch_valid_quad_mask_full & quad_touches_roi_mask

            total_full_valid_quads = int(patch_valid_quad_mask_full.sum().item())
            total_areas[patch_index] = float(patch.area) * int(in_roi_valid_quad_mask.sum().item()) / max(total_full_valid_quads, 1)
            if not in_roi_valid_quad_mask.any():
                continue

            Hq, Wq = quad_center_zyxs.shape[:2]
            valid_idx_i, valid_idx_j = torch.where(in_roi_valid_quad_mask)
            valid_zyxs = quad_center_zyxs[valid_idx_i, valid_idx_j]

            chunk = 65536
            spiral_pieces = []
            for start in range(0, valid_zyxs.shape[0], chunk):
                spiral_pieces.append(slice_to_spiral_transform(valid_zyxs[start : start + chunk]))
            spiral_zyxs_valid = torch.cat(spiral_pieces, dim=0) if len(spiral_pieces) > 1 else spiral_pieces[0]
            theta_v, _, shifted_radius_v = get_theta_and_radii(spiral_zyxs_valid[..., 1:], dr_per_winding)

            theta_all = torch.full([Hq, Wq], float('nan'), device=device)
            shifted_radius_all = torch.full([Hq, Wq], float('nan'), device=device)
            spiral_z_all = torch.full([Hq, Wq], float('nan'), device=device)
            theta_all[valid_idx_i, valid_idx_j] = theta_v
            shifted_radius_all[valid_idx_i, valid_idx_j] = shifted_radius_v
            spiral_z_all[valid_idx_i, valid_idx_j] = spiral_zyxs_valid[..., 0]

            cols_with_valid = torch.where(in_roi_valid_quad_mask.any(dim=0))[0]
            if len(cols_with_valid) == 0:
                continue
            center_col = int(cols_with_valid[len(cols_with_valid) // 2].item())

            satisfied_quad_mask = torch.zeros([Hq, Wq], dtype=torch.bool, device=device)
            target_raw_shifted_all = torch.full([Hq, Wq], float('nan'), device=device)
            valid_quad_mask_np = in_roi_valid_quad_mask.cpu().numpy()
            row_infos = [None] * Hq

            def seed_branch_offset(subrow, anchor_col):
                anchor_pos = min(max(anchor_col - subrow['j_min'], 0), subrow['unwrapped_shifted'].numel() - 1)
                subrow['branch_offset'] = subrow['cum_adj'][anchor_pos]

            def propagate_branch_offset(source, source_pos, target, target_pos):
                if source['branch_offset'] is None or target['branch_offset'] is not None:
                    return False
                shifted_diff = target['unwrapped_shifted'][target_pos] - source['unwrapped_shifted'][source_pos]
                winding_delta = torch.round(shifted_diff / dr) * dr
                target['branch_offset'] = source['branch_offset'] + winding_delta
                return True

            all_subrows = []

            for i in range(Hq):
                row_valid = valid_quad_mask_np[i]
                if not np.any(row_valid):
                    continue
                steps = np.nonzero(np.diff(np.concatenate([[0], row_valid, [0]])))[0]
                subrows = np.stack([steps[::2], steps[1::2]], axis=1)
                subrow_infos = []
                for j_min, j_max in subrows:
                    row_thetas = theta_all[i, j_min:j_max]
                    row_shifted = shifted_radius_all[i, j_min:j_max]
                    if row_thetas.numel() <= 1:
                        cum_adj = torch.zeros_like(row_thetas)
                    else:
                        step_adj = get_theta_crossing_step_adjustments(row_thetas, dr)
                        cum_adj = torch.cat([torch.zeros([1], device=device, dtype=row_thetas.dtype), torch.cumsum(step_adj, dim=0)], dim=0)
                    subrow_infos.append({
                        'row_idx': i,
                        'j_min': int(j_min),
                        'j_max': int(j_max),
                        'cum_adj': cum_adj,
                        'unwrapped_shifted': row_shifted + cum_adj,
                        'branch_offset': None,
                        'neighbors': [],
                    })
                row_infos[i] = subrow_infos
                all_subrows.extend(subrow_infos)

            for i in range(Hq - 1):
                upper_subrows = row_infos[i]
                lower_subrows = row_infos[i + 1]
                if upper_subrows is None or lower_subrows is None:
                    continue
                upper_idx = 0
                lower_idx = 0
                while upper_idx < len(upper_subrows) and lower_idx < len(lower_subrows):
                    upper = upper_subrows[upper_idx]
                    lower = lower_subrows[lower_idx]
                    overlap_min = max(upper['j_min'], lower['j_min'])
                    overlap_max = min(upper['j_max'], lower['j_max'])
                    if overlap_max > overlap_min:
                        j_anchor = (overlap_min + overlap_max - 1) // 2
                        upper_pos = j_anchor - upper['j_min']
                        lower_pos = j_anchor - lower['j_min']
                        upper['neighbors'].append((lower, upper_pos, lower_pos))
                        lower['neighbors'].append((upper, lower_pos, upper_pos))
                    if upper['j_max'] <= lower['j_max']:
                        upper_idx += 1
                    else:
                        lower_idx += 1

            rows_with_center = torch.where(in_roi_valid_quad_mask[:, center_col])[0]
            if len(rows_with_center) == 0:
                continue
            center_row = int(rows_with_center[len(rows_with_center) // 2].item())
            center_subrows = row_infos[center_row]
            center_subrow = None
            for subrow in center_subrows:
                if subrow['j_min'] <= center_col < subrow['j_max']:
                    seed_branch_offset(subrow, center_col)
                    center_subrow = subrow
                    break
            if center_subrow is None:
                continue

            queue = [center_subrow]
            queue_pos = 0
            while queue_pos < len(queue):
                source = queue[queue_pos]
                queue_pos += 1
                for target, source_pos, target_pos in source['neighbors']:
                    if propagate_branch_offset(source, source_pos, target, target_pos):
                        queue.append(target)

            # Snap-target winding is chosen from the center column across the connected
            # component. Gather the BRANCH-CONSISTENT (unwrapped, branch-offset-removed)
            # shifted radius at that column - the same `adjusted_shifted` frame the
            # satisfaction check below compares against - NOT the raw shifted_radius_all.
            # Raw values jump by ~dr across a theta=0 crossing even on the same physical
            # winding, so a raw median lands between windings and mismatches every quad
            # (spuriously reporting a well-fit seam-crossing patch as 0% satisfied).
            component_col_shifted = []
            for subrow in all_subrows:
                if subrow['branch_offset'] is None or not (subrow['j_min'] <= center_col < subrow['j_max']):
                    continue
                pos = center_col - subrow['j_min']
                component_col_shifted.append(subrow['unwrapped_shifted'][pos] - subrow['branch_offset'])
            if len(component_col_shifted) == 0:
                continue
            component_col_shifted = torch.stack(component_col_shifted)
            median_shifted_radius = torch.median(component_col_shifted)
            modulus = median_shifted_radius % dr
            target_shifted_radius = torch.where(
                modulus < dr / 2,
                median_shifted_radius - modulus,
                median_shifted_radius + dr - modulus,
            )

            if verbose and any(subrow['branch_offset'] is None for subrow in all_subrows):
                print(f'Warning: patch {patch_index} has multiple disconnected subrow components; using only the component containing the center column')

            for subrow in all_subrows:
                branch_offset = subrow['branch_offset']
                if branch_offset is None:
                    continue
                i = subrow['row_idx']
                j_min = subrow['j_min']
                j_max = subrow['j_max']
                cum_adj = subrow['cum_adj']
                adjusted_shifted = subrow['unwrapped_shifted'] - branch_offset

                in_band = (adjusted_shifted - target_shifted_radius).abs() <= spiral_tolerance
                satisfied_quad_mask[i, j_min:j_max] = in_band

                # Per-quad raw target shifted-radius (consistent with the unwrap, so the
                # target sits on the same physical winding across theta=0 crossings).
                target_raw_shifted_all[i, j_min:j_max] = target_shifted_radius - cum_adj + branch_offset

            # Scan-space distance check: for every quad-center with a per-row target set,
            # build the corresponding spiral-space point on the target winding (same
            # theta, same z, target shifted-radius), invert to scan space, and require
            # the scan-voxel distance to the original quad-center be within tolerance.
            target_set_mask = (~torch.isnan(target_raw_shifted_all)) & in_roi_valid_quad_mask
            scan_in_band = torch.zeros([Hq, Wq], dtype=torch.bool, device=device)
            if target_set_mask.any():
                sel_i, sel_j = torch.where(target_set_mask)
                theta_sel = theta_all[sel_i, sel_j]
                target_raw_sel = target_raw_shifted_all[sel_i, sel_j]
                target_radius_sel = target_raw_sel + theta_sel / (2 * np.pi) * dr
                target_spiral_zyx_sel = torch.stack([
                    spiral_z_all[sel_i, sel_j],
                    torch.sin(theta_sel) * target_radius_sel,
                    torch.cos(theta_sel) * target_radius_sel,
                ], dim=-1)
                orig_scan_sel = quad_center_zyxs[sel_i, sel_j]
                target_scan_pieces = []
                for start in range(0, target_spiral_zyx_sel.shape[0], chunk):
                    target_scan_pieces.append(slice_to_spiral_transform.inv(target_spiral_zyx_sel[start : start + chunk]))
                target_scan_sel = torch.cat(target_scan_pieces, dim=0) if len(target_scan_pieces) > 1 else target_scan_pieces[0]
                scan_distances_sel = torch.linalg.norm(target_scan_sel - orig_scan_sel, dim=-1)
                scan_in_band[sel_i, sel_j] = scan_distances_sel <= scan_tolerance

            satisfied_quad_mask = satisfied_quad_mask & scan_in_band & in_roi_valid_quad_mask

            # Per-quad output-mesh winding index, derived from the raw (per-row) target
            # shifted-radius. NaN entries (quads without a target set) become -1.
            target_winding_idx_full = torch.where(
                torch.isnan(target_raw_shifted_all),
                torch.full_like(target_raw_shifted_all, -1.),
                torch.round(target_raw_shifted_all / dr),
            ).to(torch.int64)
            target_winding_idx_per_patch[patch_index] = target_winding_idx_full.cpu()

            # Boundary = in-ROI-valid quads with at least one 4-neighbor that is
            # out-of-bounds or not in in_roi_valid_quad_mask.
            padded = torch.nn.functional.pad(in_roi_valid_quad_mask, (1, 1, 1, 1), value=False)
            all_neighbors_in = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
            boundary_quad_mask = in_roi_valid_quad_mask & ~all_neighbors_in

            total_valid_quads = int(in_roi_valid_quad_mask.sum().item())
            satisfied_quad_masks[patch_index] = satisfied_quad_mask.cpu()
            if total_valid_quads == 0:
                continue
            num_satisfied_quads = int(satisfied_quad_mask.sum().item())
            satisfied_areas[patch_index] = float(patch.area) * num_satisfied_quads / max(total_full_valid_quads, 1)
            satisfied_patches[patch_index] = num_satisfied_quads >= thresholds['satisfied_patch_quad_fraction'] * total_valid_quads
            num_boundary_quads = int(boundary_quad_mask.sum().item())
            if num_boundary_quads > 0:
                num_satisfied_boundary_quads = int((boundary_quad_mask & satisfied_quad_mask).sum().item())
                boundary_satisfied_patches[patch_index] = num_satisfied_boundary_quads >= thresholds['boundary_satisfied_patch_quad_fraction'] * num_boundary_quads

    return satisfied_patches, satisfied_areas, total_areas, satisfied_quad_masks, boundary_satisfied_patches, target_winding_idx_per_patch


def _build_strip_spiral_context(slice_to_spiral_transform, dr_per_winding, flat, num_strips):
    # Shared front-half of the per-strip satisfaction pass: given a flat bundle
    # from the caller, transform points into spiral space, unwrap theta across
    # strip boundaries, and produce the per-point normalised
    # shifted-radius (`unwrapped_shifted - windings * dr`). Returns
    # `(ctx, lengths_cpu, num_strips)` where `ctx` is None when there are no
    # points; downstream target-winding selectors (median / mode) operate on
    # `ctx['normalised_radii']` and feed the picked per-strip target through
    # `_strip_satisfaction_from_target`.
    spiral_tolerance = dr_per_winding.detach() * metrics_config['satisfaction_radius_tolerance']
    scan_tolerance = metrics_config['satisfaction_distance_tolerance']
    dr = dr_per_winding.detach()
    device = dr_per_winding.device
    S = num_strips

    if flat is None or flat['total'] == 0:
        lengths_cpu = flat['lengths_cpu'] if flat is not None else torch.zeros(S, dtype=torch.int64)
        return None, lengths_cpu, S

    chunk = 65536

    def transform_in_chunks(zyxs, fn):
        if zyxs.shape[0] <= chunk:
            return fn(zyxs)
        pieces = []
        for st in range(0, zyxs.shape[0], chunk):
            pieces.append(fn(zyxs[st:st + chunk]))
        return torch.cat(pieces, dim=0)

    zyxs = flat['zyxs']
    windings = flat['windings']
    strip_id = flat['strip_id']
    starts = flat['starts']
    lengths = flat['lengths']
    lengths_cpu = flat['lengths_cpu']
    T = flat['total']

    with torch.no_grad():
        spiral_zyxs = transform_in_chunks(zyxs, slice_to_spiral_transform)
        theta, _, shifted_radii = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)

        # Segmented version of unwrap_shifted_radii: build
        # adjustments via a global cumsum where step_adj is zeroed across strip
        # boundaries, then subtract each strip's start value so each strip
        # starts at 0 in its own frame.
        if T > 1:
            same_strip = strip_id[1:] == strip_id[:-1]
            step_adj = get_theta_crossing_step_adjustments(theta, dr)
            step_adj = torch.where(same_strip, step_adj, torch.zeros_like(step_adj))
            cumsum_inner = torch.cumsum(step_adj, dim=0)
            cumsum_flat = torch.cat([
                torch.zeros(1, device=device, dtype=cumsum_inner.dtype),
                cumsum_inner,
            ], dim=0)
            adjustments = cumsum_flat - cumsum_flat[starts[:-1][strip_id]]
        else:
            adjustments = torch.zeros_like(shifted_radii)
        unwrapped_shifted = shifted_radii + adjustments

        normalised_radii = unwrapped_shifted - windings * dr

    ctx = {
        'spiral_tolerance': spiral_tolerance,
        'scan_tolerance': scan_tolerance,
        'dr': dr,
        'device': device,
        'S': S,
        'T': T,
        'transform_in_chunks': transform_in_chunks,
        'slice_to_spiral_transform': slice_to_spiral_transform,
        'zyxs': zyxs,
        'windings': windings,
        'strip_id': strip_id,
        'starts': starts,
        'lengths': lengths,
        'lengths_cpu': lengths_cpu,
        'spiral_zyxs': spiral_zyxs,
        'theta': theta,
        'adjustments': adjustments,
        'unwrapped_shifted': unwrapped_shifted,
        'normalised_radii': normalised_radii,
    }
    return ctx, lengths_cpu, S


def _strip_satisfaction_from_target(ctx, target_normalised_per_strip):
    # Given a per-strip target normalised shifted-radius, count points whose
    # spiral-space radius and scan-space distance both fall within the
    # satisfaction tolerances. Returns
    # `(satisfied_counts_cpu, per_point_satisfaction_cpu_list)`.
    dr = ctx['dr']
    device = ctx['device']
    S = ctx['S']
    strip_id = ctx['strip_id']
    windings = ctx['windings']
    theta = ctx['theta']
    adjustments = ctx['adjustments']
    unwrapped_shifted = ctx['unwrapped_shifted']
    spiral_zyxs = ctx['spiral_zyxs']
    zyxs = ctx['zyxs']
    lengths_cpu = ctx['lengths_cpu']
    spiral_tolerance = ctx['spiral_tolerance']
    scan_tolerance = ctx['scan_tolerance']
    transform_in_chunks = ctx['transform_in_chunks']
    slice_to_spiral_transform = ctx['slice_to_spiral_transform']

    with torch.no_grad():
        target_normalised = target_normalised_per_strip[strip_id]
        target_shifted = target_normalised + windings * dr
        spiral_in_band = (unwrapped_shifted - target_shifted).abs() <= spiral_tolerance

        target_radii = radius_from_unwrapped_shifted(
            theta, target_shifted, adjustments, dr,
        )
        target_spiral_zyxs = torch.stack([
            spiral_zyxs[..., 0],
            torch.sin(theta) * target_radii,
            torch.cos(theta) * target_radii,
        ], dim=-1)
        target_scroll_zyxs = transform_in_chunks(target_spiral_zyxs, slice_to_spiral_transform.inv)
        scan_distances = torch.linalg.norm(target_scroll_zyxs - zyxs, dim=-1)
        scan_in_band = scan_distances <= scan_tolerance

        satisfied = spiral_in_band & scan_in_band

        satisfied_counts_dev = torch.zeros(S, dtype=torch.int64, device=device)
        satisfied_counts_dev.scatter_add_(0, strip_id, satisfied.to(torch.int64))
        satisfied_counts = satisfied_counts_dev.cpu()

        per_point_satisfaction = list(torch.split(satisfied.cpu(), lengths_cpu.tolist()))

    return satisfied_counts, per_point_satisfaction


def get_unattached_pcl_satisfied_counts(slice_to_spiral_transform, dr_per_winding, pcl_strips, get_flat_bundle):
    # For each unattached pcl, treat its id-sorted points as a strip (so theta=0
    # crossings can be unwrapped, mirroring the patch row-walk in
    # get_patch_satisfied_areas), pick the snapped median normalised shifted-radius
    # as the target winding, then count points that satisfy both the same spiral-
    # space radius tolerance and the same scan-space distance tolerance used for
    # quad satisfaction. Returns three values: (satisfied_count_per_pcl,
    # total_count_per_pcl, per_point_satisfaction) — the first two are 1-D int64
    # tensors, and per_point_satisfaction is a list of CPU bool tensors (one per
    # pcl, of length N for that pcl; empty pcls get an empty tensor).
    #
    # All strips are processed in a single batched pass: points are concatenated
    # into one flat (T, 3) tensor, the scan->spiral transform runs once over
    # everything, then unwrap / median / satisfaction are done with segmented
    # cumsum and a single composite-key sort (no Python-level per-strip loop).
    flat = get_flat_bundle(pcl_strips, dr_per_winding.device)
    ctx, lengths_cpu, S = _build_strip_spiral_context(
        slice_to_spiral_transform, dr_per_winding, flat, len(pcl_strips),
    )
    if ctx is None:
        per_point = [torch.zeros([int(n.item())], dtype=torch.bool) for n in lengths_cpu]
        return torch.zeros(S, dtype=torch.int64), lengths_cpu.clone(), per_point

    dr = ctx['dr']

    with torch.no_grad():
        medians = _segmented_median_per_strip(ctx)
        target_normalised_per_strip = torch.round(medians / dr) * dr

    satisfied_counts, per_point_satisfaction = _strip_satisfaction_from_target(ctx, target_normalised_per_strip)
    return satisfied_counts, lengths_cpu.clone(), per_point_satisfaction


def save_overlay_and_print_satisfaction(
    suffix,
    *,
    spiral_and_transform,
    slice_to_spiral_transform,
    dr_per_winding,
    patches_list,
    patches_dict,
    unattached_pcl_strips,
    tracks,
    unverified_patches_list,
    unverified_patches_dict,
    out_path,
    cfg,
    z_begin,
    z_end,
    flow_field_radius,
    flow_min_corner_spiral_zyx,
    flow_max_corner_spiral_zyx,
    zs_for_visualisation,
    slice_yx,
    scroll_slices_for_visualisation,
    prediction_slices_for_visualisation,
    quad_label_map,
    z_to_umbilicus_yx,
    render_volume_scale,
    voxel_size_um,
    get_or_build_unattached_pcl_flat,
    run_tag=None,
):
    satisfied_patches, satisfied_areas, total_areas, satisfied_quad_masks, boundary_satisfied_patches, target_winding_idx_per_patch = get_patch_satisfied_areas(
        slice_to_spiral_transform, dr_per_winding, patches_list, z_begin, z_end, verbose=True,
    )
    satisfied_count = satisfied_patches.sum().item()
    boundary_satisfied_count = boundary_satisfied_patches.sum().item()
    total_count = satisfied_patches.numel()
    satisfied_ratio = satisfied_count / max(total_count, 1)
    print(f'satisfied_patches = {satisfied_count}/{total_count} ({satisfied_ratio * 100:.1f}%)')
    # Same binary satisfied/not per patch, but weight the overall fraction by patch area
    # so a few large satisfied patches count more than many tiny ones.
    all_patches_area = float(total_areas.sum().item())
    satisfied_patches_area = float(total_areas[satisfied_patches].sum().item())
    area_weighted_satisfied_ratio = satisfied_patches_area / max(all_patches_area, 1e-9)
    print(f'satisfied_patches_area_weighted = {satisfied_patches_area:.1f}/{all_patches_area:.1f} ({area_weighted_satisfied_ratio * 100:.1f}%)')
    boundary_satisfied_ratio = boundary_satisfied_count / max(total_count, 1)
    print(f'boundary_satisfied_patches = {boundary_satisfied_count}/{total_count} ({boundary_satisfied_ratio * 100:.1f}%)')
    satisfied_area = float(satisfied_areas.sum().item())
    total_area = float(total_areas.sum().item())
    satisfied_area_ratio = satisfied_area / max(total_area, 1e-9)
    print(f'satisfied_area = {satisfied_area:.1f}/{total_area:.1f} ({satisfied_area_ratio * 100:.1f}%)')
    unattached_pcl_per_point_satisfied = []
    unattached_pcl_fully_satisfied = torch.zeros(len(unattached_pcl_strips), dtype=torch.bool)
    if unattached_pcl_strips:
        unattached_pcl_satisfied_counts, unattached_pcl_total_counts, unattached_pcl_per_point_satisfied = get_unattached_pcl_satisfied_counts(
            slice_to_spiral_transform, dr_per_winding, unattached_pcl_strips, get_or_build_unattached_pcl_flat,
        )
        unattached_pcl_fully_satisfied = (unattached_pcl_satisfied_counts == unattached_pcl_total_counts)
        fully_satisfied_pcls = int(unattached_pcl_fully_satisfied.sum().item())
        num_pcls = len(unattached_pcl_strips)
        fully_satisfied_ratio = fully_satisfied_pcls / max(num_pcls, 1)
        print(f'satisfied_unattached_pcls = {fully_satisfied_pcls}/{num_pcls} ({fully_satisfied_ratio * 100:.1f}%)')
        satisfied_points = int(unattached_pcl_satisfied_counts.sum().item())
        total_points = int(unattached_pcl_total_counts.sum().item())
        satisfied_point_ratio = satisfied_points / max(total_points, 1)
        print(f'satisfied_unattached_pcl_points = {satisfied_points}/{total_points} ({satisfied_point_ratio * 100:.1f}%)')
    if tracks:
        # Free the patch/pcl eval tensors before the (much larger) track eval,
        # and chunk the track eval so the full track set does not have to be
        # materialised on the GPU at once. Guard against OOM so that a failure
        # to compute the secondary track metric never prevents the mesh/overlay
        # from being saved.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            track_satisfied_counts, track_total_counts = get_track_satisfied_counts_in_chunks(
                slice_to_spiral_transform, dr_per_winding, tracks, metrics_config,
            )
            track_fully_satisfied = (track_satisfied_counts == track_total_counts)
            fully_satisfied_tracks = int(track_fully_satisfied.sum().item())
            num_valid_tracks = int(track_total_counts.numel())
            fully_satisfied_track_ratio = fully_satisfied_tracks / max(num_valid_tracks, 1)
            print(f'satisfied_tracks = {fully_satisfied_tracks}/{num_valid_tracks} ({fully_satisfied_track_ratio * 100:.1f}%)')
            track_satisfied_points = int(track_satisfied_counts.sum().item())
            track_total_points = int(track_total_counts.sum().item())
            track_satisfied_point_ratio = track_satisfied_points / max(track_total_points, 1)
            print(f'satisfied_track_points = {track_satisfied_points}/{track_total_points} ({track_satisfied_point_ratio * 100:.1f}%)')
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print('WARNING: skipped satisfied_tracks metric (CUDA OOM during track evaluation)')
    # Unverified patches are reported entirely separately so they never inflate the verified
    # satisfaction numbers.
    unverified_patch_satisfaction_entries = []
    if unverified_patches_list:
        u_satisfied, u_sat_areas, u_tot_areas, _, _, _ = get_patch_satisfied_areas(
            slice_to_spiral_transform, dr_per_winding, unverified_patches_list, z_begin, z_end, verbose=False,
        )
        u_count = int(u_satisfied.sum().item())
        u_total = u_satisfied.numel()
        u_ratio = u_count / max(u_total, 1)
        print(f'unverified_satisfied_patches = {u_count}/{u_total} ({u_ratio * 100:.1f}%)')
        u_sat_area = float(u_sat_areas.sum().item())
        u_tot_area = float(u_tot_areas.sum().item())
        u_area_ratio = u_sat_area / max(u_tot_area, 1e-9)
        print(f'unverified_satisfied_area = {u_sat_area:.1f}/{u_tot_area:.1f} ({u_area_ratio * 100:.1f}%)')
        for pid, sat_area_t, tot_area_t in zip(unverified_patches_dict.keys(), u_sat_areas.tolist(), u_tot_areas.tolist()):
            fraction = sat_area_t / tot_area_t if tot_area_t > 0 else 0.0
            unverified_patch_satisfaction_entries.append({
                'id': pid,
                'satisfied_area': sat_area_t,
                'total_area': tot_area_t,
                'fraction': fraction,
            })
        unverified_patch_satisfaction_entries.sort(key=lambda e: e['fraction'])

    patch_ids = list(patches_dict.keys())
    patch_satisfaction_entries = []
    for pid, sat_area_t, tot_area_t in zip(patch_ids, satisfied_areas.tolist(), total_areas.tolist()):
        fraction = sat_area_t / tot_area_t if tot_area_t > 0 else 0.0
        patch_satisfaction_entries.append({
            'id': pid,
            'satisfied_area': sat_area_t,
            'total_area': tot_area_t,
            'fraction': fraction,
        })
    patch_satisfaction_entries.sort(key=lambda e: e['fraction'])
    pcl_satisfaction_entries = []
    if unattached_pcl_strips:
        sat_counts = unattached_pcl_satisfied_counts.tolist()
        tot_counts = unattached_pcl_total_counts.tolist()
        for strip, sc, tc in zip(unattached_pcl_strips, sat_counts, tot_counts):
            fraction = sc / tc if tc > 0 else 0.0
            pcl_satisfaction_entries.append({
                'id': strip.get('id'),
                'name': strip.get('name'),
                'source_file': strip.get('source_file'),
                'satisfied_points': int(sc),
                'total_points': int(tc),
                'fraction': fraction,
            })
        pcl_satisfaction_entries.sort(key=lambda e: e['fraction'])
    with open(f'{out_path}/satisfied_{suffix}.json', 'w') as f:
        json.dump({
            'patches': patch_satisfaction_entries,
            'pcls': pcl_satisfaction_entries,
            'unverified_patches': unverified_patch_satisfaction_entries,
        }, f, indent=2)
    # Flatten per-patch (H-1, W-1) masks in patch order to match the rasteriser's quad-id offsets,
    # then combine with patch-level overall satisfaction into a 0/1/2 status per quad.
    if satisfied_quad_masks:
        satisfied_quads_flat = torch.cat([m.flatten() for m in satisfied_quad_masks])
        quads_per_patch = torch.tensor([m.numel() for m in satisfied_quad_masks], dtype=torch.int64)
        overall_satisfied_per_quad = satisfied_patches.to(torch.bool).repeat_interleave(quads_per_patch)
    else:
        satisfied_quads_flat = torch.zeros([0], dtype=torch.bool)
        overall_satisfied_per_quad = torch.zeros([0], dtype=torch.bool)
    quad_status_flat = torch.where(
        overall_satisfied_per_quad,
        torch.full_like(satisfied_quads_flat, 2, dtype=torch.int64),
        satisfied_quads_flat.to(torch.int64),
    )
    if os.environ.get('FIT_SPIRAL_SKIP_SAVE_OVERLAY') != '1':
        winding_range, patch_extents, pcl_extents = compute_winding_range_and_input_extents(
            slice_to_spiral_transform, dr_per_winding, patches_list, unattached_pcl_strips,
            cfg, z_begin, z_end, get_or_build_unattached_pcl_flat,
        )
        _warn_if_inputs_exceed_flow_bounds(
            list(patches_dict.keys()), patch_extents,
            unattached_pcl_strips, pcl_extents,
            flow_field_radius,
            cfg,
        )
        save_overlay(
            spiral_and_transform,
            flow_min_corner_spiral_zyx, flow_max_corner_spiral_zyx,
            zs_for_visualisation, slice_yx,
            scroll_slices_for_visualisation, prediction_slices_for_visualisation,
            quad_label_map, quad_status_flat,
            unattached_pcl_strips, unattached_pcl_per_point_satisfied, unattached_pcl_fully_satisfied,
            z_to_umbilicus_yx,
            winding_range,
            tracks,
            out_path, suffix,
            render_volume_scale=render_volume_scale,
        )
    if os.environ.get('FIT_SPIRAL_SKIP_SAVE_MESH') != '1':
        def get_patch_satisfied_areas_for_mesh(transform, winding_delta, patches, verbose=False, metrics_overrides=None):
            return get_patch_satisfied_areas(
                transform, winding_delta, patches, z_begin, z_end, verbose=verbose,
                metrics_overrides=metrics_overrides,
            )

        save_mesh(
            slice_to_spiral_transform, dr_per_winding, patches_list, unattached_pcl_strips,
            out_path, cfg, z_begin, z_end, voxel_size_um,
            get_or_build_unattached_pcl_flat, get_patch_satisfied_areas_for_mesh,
            run_tag=run_tag, name=suffix,
        )
