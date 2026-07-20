import json
import glob
import os

import numpy as np
import scipy.ndimage
import torch
from tqdm import tqdm

from sample_spiral import get_spiral_yxs, get_theta_and_radii
from tifxyz import load_tifxyz, save_tifxyz, save_combined_tifxyz


def scale_patch(patch, downsample_factor):
    patch.scale *= downsample_factor
    patch.zyxs /= downsample_factor
    patch.area /= downsample_factor ** 2
    patch.release_derived_caches()


def patch_intersects_z_roi(patch, z_begin, z_end):
    zs = patch.valid_zyxs[..., 0]
    if zs.numel() == 0:
        return False
    return bool(((zs >= z_begin) & (zs < z_end)).any().item())


def scale_counts_for_z_range(
    config,
    z_begin,
    z_end,
    reference_z_range_num_slices,
    z_range_scaled_count_keys,
    floors=None,
):
    """Scale per-step sample counts with the z-range, respecting floors.

    Floors exist for losses whose per-sample information is sparse: the
    phase bundle sees ~6 winding gradient sites per pair, so
    volume-proportional scaling starves it on narrow windows (a 300-slice
    session got ~380 pairs from the 12k default and corrected at half
    grad_mag's rate - 2026-07-17 sampling-scale probes).
    """
    num_slices = z_end - z_begin
    scale = num_slices / reference_z_range_num_slices
    for key in z_range_scaled_count_keys:
        floor = 1 if floors is None else int(floors.get(key, 1))
        config[key] = max(floor, round(config[key] * scale))
    return scale, num_slices


SAMPLING_COUNT_FLOORS = {
    'dense_spacing_num_pairs': 8_000,
    'dense_spacing_density_extra_pairs': 16_000,
}


def _decimate_ordered_points_min_spacing(points, min_spacing):
    if min_spacing <= 0 or len(points) <= 1:
        return points

    keep = [0]
    last_kept = points[0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - last_kept) >= min_spacing:
            keep.append(i)
            last_kept = points[i]
    return points[keep]


def load_fiber_point_collection(path, collection_id, coordinate_scale=0.25, min_point_spacing=20.0):
    # Fiber JSONs are stored as one vc3d_fiber per file. Their control_points are
    # x/y/z coordinates at 4x the scale used by the regular PCL JSONs. line_points
    # are derived rendering geometry and must not be used as fitting constraints.
    with open(path, 'r') as f:
        data = json.load(f)

    points_xyz = data.get('control_points') or []
    if not points_xyz:
        print(f'WARNING: fiber {path} has no control_points; skipping')
        return None

    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        print(f'WARNING: fiber {path} points have shape {points_xyz.shape}; expected (N, 3); skipping')
        return None

    points_xyz = points_xyz * coordinate_scale
    original_num_points = len(points_xyz)
    points_xyz = _decimate_ordered_points_min_spacing(points_xyz, min_point_spacing)
    name = data.get('name') or os.path.splitext(os.path.basename(path))[0]
    thing = data.get('webknossos', {}).get('thing', {})
    wk_color = thing.get('color', {})
    color = [
        float(wk_color.get('r', 0.0)),
        float(wk_color.get('g', 0.0)),
        float(wk_color.get('b', 0.0)),
    ]

    collection = {
        'id': collection_id,
        'name': name,
        'points': {},
        'metadata': {
            'source_format': data.get('type', 'vc3d_fiber'),
            'fiber_version': data.get('version'),
            'fiber_generation': data.get('generation'),
            'fiber_sequence': data.get('sequence'),
            'fiber_started_at': data.get('started_at'),
            'fiber_tags': data.get('tags', []),
            'hv_classification': data.get('hv_classification', {}),
            'input_coordinate_scale': coordinate_scale,
            'fiber_min_point_spacing': min_point_spacing,
            'fiber_original_num_points': original_num_points,
        },
        'color': color,
    }
    for point_id, p in enumerate(points_xyz):
        collection['points'][point_id] = {
            'id': point_id,
            'collectionId': collection_id,
            'p': p.tolist(),
            'winding_annotation': float('nan'),
            'creation_time': 0,
        }
    return collection


def load_fiber_point_collections(path, next_id, min_point_spacing=20.0):
    if not path:
        return {}, next_id
    fiber_paths = sorted(glob.glob(os.path.join(path, '*.json')))
    if not fiber_paths:
        print(f'no fiber point collections found in {path}')
        return {}, next_id

    point_collections = {}
    total_points = 0
    skipped = 0
    for fiber_path in fiber_paths:
        try:
            pcl = load_fiber_point_collection(fiber_path, next_id, min_point_spacing=min_point_spacing)
        except Exception as e:
            print(f'WARNING: failed to load fiber {fiber_path}: {e}')
            skipped += 1
            continue
        if pcl is None:
            skipped += 1
            continue
        pcl['source_file'] = fiber_path
        point_collections[next_id] = pcl
        total_points += len(pcl['points'])
        next_id += 1

    print(
        f'Loaded {len(point_collections)} fiber point collections '
        f'({total_points} points, min spacing {min_point_spacing:g} vx) from {path}'
        + (f'; skipped {skipped}' if skipped else '')
    )
    return point_collections, next_id


def _huber_abs(residual, delta):
    abs_residual = residual.abs()
    return torch.where(
        abs_residual <= delta,
        0.5 * residual ** 2 / delta,
        abs_residual - 0.5 * delta,
    )


def _get_patch_valid_points(patch, device, z_begin, z_end, max_points=None, fixed_num_points=None):
    valid_mask = patch.valid_vertex_mask
    z_in_roi = (patch.zyxs[..., 0] >= z_begin) & (patch.zyxs[..., 0] < z_end)
    valid_indices = torch.where(valid_mask & z_in_roi)
    if len(valid_indices[0]) == 0:
        valid_indices = torch.where(valid_mask)
    n = len(valid_indices[0])
    if fixed_num_points is not None:
        sel = np.random.choice(n, fixed_num_points, replace=(n < fixed_num_points))
        valid_indices = (valid_indices[0][sel], valid_indices[1][sel])
    elif max_points is not None and n > max_points:
        sel = np.random.choice(n, max_points, replace=False)
        valid_indices = (valid_indices[0][sel], valid_indices[1][sel])
    return patch.zyxs[valid_indices[0], valid_indices[1], :].to(device=device, dtype=torch.float32)


def get_face_indices(h, w):
    indices = torch.arange(h * w).view(h, w)
    top_left = indices[:-1, :-1].flatten()
    top_right = indices[:-1, 1:].flatten()
    bottom_left = indices[1:, :-1].flatten()
    bottom_right = indices[1:, 1:].flatten()
    return torch.cat([
        torch.stack([bottom_left, top_left, top_right], dim=1),
        torch.stack([bottom_left, top_right, bottom_right], dim=1)
    ], dim=0)


@torch.inference_mode()
def compute_winding_range_and_input_extents(
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    unattached_pcl_strips,
    cfg,
    z_begin,
    z_end,
    get_or_build_unattached_pcl_flat,
    authoritative_zyx_lines=(),
):
    """Compute output winding range plus max observed radius/winding per patch and PCL."""
    device = dr_per_winding.device
    dr = dr_per_winding.detach()
    min_w = None
    max_w = None

    def update_from_winding_indices(winding_indices):
        nonlocal min_w, max_w
        local_min = int(winding_indices.min().item())
        local_max = int(winding_indices.max().item())
        min_w = local_min if min_w is None else min(min_w, local_min)
        max_w = local_max if max_w is None else max(max_w, local_max)

    chunk = 65536

    def transform_in_chunks(zyxs):
        spiral_pieces = []
        for start in range(0, zyxs.shape[0], chunk):
            spiral_pieces.append(slice_to_spiral_transform(zyxs[start:start + chunk]))
        return torch.cat(spiral_pieces, dim=0) if len(spiral_pieces) > 1 else spiral_pieces[0]

    patch_extents = [(None, None)] * len(patches)
    for patch_index, patch in enumerate(patches):
        patch_zyxs = patch.zyxs.to(device=device, dtype=torch.float32)
        if patch_zyxs.shape[0] < 2 or patch_zyxs.shape[1] < 2:
            continue
        valid_quad_mask = patch.valid_quad_mask.to(device=device)
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
        quad_touches_roi = (quad_zs.amax(dim=0) >= z_begin) & (quad_zs.amin(dim=0) < z_end)
        mask = valid_quad_mask & quad_touches_roi
        if not mask.any():
            continue
        spiral_zyxs = transform_in_chunks(quad_center_zyxs[mask])
        _, radius, shifted_radius = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
        winding_indices = (shifted_radius / dr).round().to(torch.int64).clamp_min(0)
        update_from_winding_indices(winding_indices)
        patch_extents[patch_index] = (
            float(radius.max().item()),
            int(winding_indices.max().item()),
        )

    pcl_extents = [(None, None)] * len(unattached_pcl_strips)
    if unattached_pcl_strips:
        flat = get_or_build_unattached_pcl_flat(unattached_pcl_strips, device)
        if flat is not None and flat['total'] > 0:
            zyxs = flat['zyxs']
            strip_id = flat['strip_id']
            num_strips = flat['num_strips']
            in_roi = (zyxs[:, 0] >= z_begin) & (zyxs[:, 0] < z_end)
            if in_roi.any():
                zyxs_roi = zyxs[in_roi]
                strip_id_roi = strip_id[in_roi]
                spiral_zyxs = transform_in_chunks(zyxs_roi)
                _, radius, shifted_radius = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
                winding_indices = (shifted_radius / dr).round().to(torch.int64).clamp_min(0)
                update_from_winding_indices(winding_indices)
                per_strip_max_r = torch.zeros(num_strips, dtype=torch.float32, device=device)
                per_strip_max_w = torch.full((num_strips,), -1, dtype=torch.int64, device=device)
                strip_has_roi = torch.zeros(num_strips, dtype=torch.bool, device=device)
                per_strip_max_r.scatter_reduce_(0, strip_id_roi, radius.to(torch.float32), reduce='amax')
                per_strip_max_w.scatter_reduce_(0, strip_id_roi, winding_indices, reduce='amax')
                strip_has_roi.scatter_(0, strip_id_roi, torch.ones_like(strip_id_roi, dtype=torch.bool))
                per_strip_max_r_cpu = per_strip_max_r.cpu().tolist()
                per_strip_max_w_cpu = per_strip_max_w.cpu().tolist()
                strip_has_roi_cpu = strip_has_roi.cpu().tolist()
                for k in range(num_strips):
                    if strip_has_roi_cpu[k]:
                        pcl_extents[k] = (per_strip_max_r_cpu[k], per_strip_max_w_cpu[k])

    # Tracks are authoritative fit geometry too.  Including them makes
    # tracks-only/disable-patches sessions derive the same output upper bound
    # instead of silently producing no preview.  Track DBMs can contain millions
    # of short lines, so batch their points before moving them to the GPU.  A
    # transform call per line makes preview generation dominated by CUDA launch
    # overhead (and took hours for multi-million-track datasets).
    pending_track_points = []
    pending_track_point_count = 0

    def update_from_track_points(zyxs):
        # Reduce each transformed chunk immediately.  In addition to bounding
        # memory, this lets callers reuse an already-flat GPU track tensor
        # without concatenating a second full-dataset transform result.
        for start in range(0, zyxs.shape[0], chunk):
            spiral_zyxs = slice_to_spiral_transform(zyxs[start:start + chunk])
            _, _, shifted_radius = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
            update_from_winding_indices(
                (shifted_radius / dr).round().to(torch.int64).clamp_min(0)
            )

    def flush_pending_track_points():
        nonlocal pending_track_point_count
        if not pending_track_points:
            return
        points = (pending_track_points[0] if len(pending_track_points) == 1
                  else np.concatenate(pending_track_points, axis=0))
        update_from_track_points(torch.as_tensor(points, device=device, dtype=torch.float32))
        pending_track_points.clear()
        pending_track_point_count = 0

    for line in authoritative_zyx_lines or ():
        if torch.is_tensor(line):
            # Keep tensor callers supported without trying to convert CUDA data
            # through NumPy.  Flush host-side points first to preserve the bound.
            flush_pending_track_points()
            zyxs = line.to(device=device, dtype=torch.float32).reshape(-1, 3)
            in_roi = (zyxs[:, 0] >= z_begin) & (zyxs[:, 0] < z_end)
            if in_roi.any():
                update_from_track_points(zyxs[in_roi])
            continue

        points = np.asarray(line, dtype=np.float32).reshape(-1, 3)
        if points.shape[0] == 0:
            continue
        in_roi = (points[:, 0] >= z_begin) & (points[:, 0] < z_end)
        if not in_roi.any():
            continue
        points = points[in_roi]
        pending_track_points.append(points)
        pending_track_point_count += points.shape[0]
        if pending_track_point_count >= chunk:
            flush_pending_track_points()
    flush_pending_track_points()

    first_winding = cfg['output_first_winding']
    if min_w is None:
        output_winding_range = (first_winding, first_winding)
    else:
        margin = cfg['output_winding_margin']
        output_winding_range = (max(min_w - margin, first_winding), max_w + 1 + margin)
    return output_winding_range, patch_extents, pcl_extents


def _infer_shell_outer_winding_idx(
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    unattached_pcl_strips,
    cfg,
    z_begin,
    z_end,
    get_or_build_unattached_pcl_flat,
):
    _, patch_extents, pcl_extents = compute_winding_range_and_input_extents(
        slice_to_spiral_transform,
        dr_per_winding,
        patches,
        unattached_pcl_strips,
        cfg,
        z_begin,
        z_end,
        get_or_build_unattached_pcl_flat,
    )
    observed_max = None
    for _, max_w in patch_extents + pcl_extents:
        if max_w is None:
            continue
        observed_max = max_w if observed_max is None else max(observed_max, max_w)
    if observed_max is None:
        observed_max = cfg['output_first_winding']
    return int(observed_max + cfg['shell_outer_winding_margin'])


def _warn_if_inputs_exceed_flow_bounds(
    patch_ids,
    patch_extents,
    unattached_pcl_strips,
    pcl_extents,
    flow_field_radius,
    cfg,
):
    gap_expander_num_windings = cfg['gap_expander_num_windings']

    over_radius_patches = []
    over_winding_patches = []
    for pid, (max_r, max_w) in zip(patch_ids, patch_extents):
        if max_r is None:
            continue
        if max_r > flow_field_radius:
            over_radius_patches.append((pid, max_r))
        if max_w >= gap_expander_num_windings:
            over_winding_patches.append((pid, max_w))

    over_radius_pcls = []
    over_winding_pcls = []
    for k, (strip, (max_r, max_w)) in enumerate(zip(unattached_pcl_strips, pcl_extents)):
        if max_r is None:
            continue
        name = strip.get('name') or strip.get('id') or strip.get('source_file') or f'#{k}'
        if max_r > flow_field_radius:
            over_radius_pcls.append((name, max_r))
        if max_w >= gap_expander_num_windings:
            over_winding_pcls.append((name, max_w))

    def _print_offenders(kind, value_label, threshold_label, patches, pcls, fmt):
        if not (patches or pcls):
            return
        print(f'WARNING: {len(patches)} patch(es) and {len(pcls)} unattached pcl(s) have {value_label} exceeding {threshold_label}:')
        for pid, v in sorted(patches, key=lambda e: -e[1])[:10]:
            print(f'  patch {pid}: max {kind} {fmt(v)}')
        if len(patches) > 10:
            print(f'  ... and {len(patches) - 10} more patches')
        for name, v in sorted(pcls, key=lambda e: -e[1])[:10]:
            print(f'  pcl {name}: max {kind} {fmt(v)}')
        if len(pcls) > 10:
            print(f'  ... and {len(pcls) - 10} more pcls')

    _print_offenders(
        'spiral radius', 'spiral-space radius', f'flow_bounds_radius ({flow_field_radius})',
        over_radius_patches, over_radius_pcls, lambda v: f'{v:.1f}',
    )
    _print_offenders(
        'winding idx', 'winding index', f'gap_expander_num_windings ({gap_expander_num_windings})',
        over_winding_patches, over_winding_pcls, lambda v: f'{v}',
    )


def _rasterize_triangles_into_mesh(
    tri_uvs,
    tri_scrolls,
    tri_target_w,
    scroll_zyxs,
    winding_offsets_t,
    num_thetas_t,
):
    device = scroll_zyxs.device
    T = tri_uvs.shape[0]
    if T == 0:
        return

    num_zs = scroll_zyxs.shape[0]
    v_lim_per_tri = num_thetas_t[tri_target_w]

    u_min = torch.floor(tri_uvs[..., 0].min(dim=-1).values).to(torch.long)
    u_max = torch.ceil(tri_uvs[..., 0].max(dim=-1).values).to(torch.long)
    v_min = torch.floor(tri_uvs[..., 1].min(dim=-1).values).to(torch.long)
    v_max = torch.ceil(tri_uvs[..., 1].max(dim=-1).values).to(torch.long)
    u_min = u_min.clamp(min=0, max=num_zs - 1)
    u_max = u_max.clamp(min=0, max=num_zs - 1)
    v_min = v_min.clamp(min=0)
    v_max = torch.minimum(v_max, v_lim_per_tri - 1)
    valid_bbox = (u_min <= u_max) & (v_min <= v_max)

    bbox_h = (u_max - u_min + 1).clamp(min=1)
    bbox_w = (v_max - v_min + 1).clamp(min=1)

    chunk_size = 16384
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        valid_c = valid_bbox[s:e]
        if not valid_c.any():
            continue
        u_min_c = u_min[s:e]
        v_min_c = v_min[s:e]
        bbox_h_c = bbox_h[s:e]
        bbox_w_c = bbox_w[s:e]
        max_h = int(bbox_h_c[valid_c].max().item())
        max_w = int(bbox_w_c[valid_c].max().item())

        du_grid, dv_grid = torch.meshgrid(
            torch.arange(max_h, device=device),
            torch.arange(max_w, device=device),
            indexing='ij',
        )
        us = u_min_c[:, None, None] + du_grid[None]
        vs = v_min_c[:, None, None] + dv_grid[None]
        in_bbox = (
            (du_grid[None] < bbox_h_c[:, None, None])
            & (dv_grid[None] < bbox_w_c[:, None, None])
            & valid_c[:, None, None]
        )

        tri_uvs_c = tri_uvs[s:e]
        pts = torch.stack([us.float(), vs.float()], dim=-1)
        a = tri_uvs_c[:, 0]
        b = tri_uvs_c[:, 1]
        c = tri_uvs_c[:, 2]
        v0 = b - a
        v1 = c - a
        v2 = pts - a[:, None, None]
        d00 = (v0 * v0).sum(-1)
        d01 = (v0 * v1).sum(-1)
        d11 = (v1 * v1).sum(-1)
        d20 = (v2 * v0[:, None, None]).sum(-1)
        d21 = (v2 * v1[:, None, None]).sum(-1)
        denom = d00 * d11 - d01 * d01
        nonzero = denom.abs() >= 1e-9
        denom_safe = torch.where(nonzero, denom, torch.ones_like(denom))
        beta = (d11[:, None, None] * d20 - d01[:, None, None] * d21) / denom_safe[:, None, None]
        gamma = (d00[:, None, None] * d21 - d01[:, None, None] * d20) / denom_safe[:, None, None]
        alpha = 1 - beta - gamma
        inside = (alpha >= -1e-6) & (beta >= -1e-6) & (gamma >= -1e-6) & nonzero[:, None, None]
        mask = in_bbox & inside
        if not mask.any():
            continue

        tri_scrolls_c = tri_scrolls[s:e]
        sa = tri_scrolls_c[:, 0][:, None, None, :]
        sb = tri_scrolls_c[:, 1][:, None, None, :]
        sc = tri_scrolls_c[:, 2][:, None, None, :]
        interp = alpha[..., None] * sa + beta[..., None] * sb + gamma[..., None] * sc

        sel = mask.reshape(-1)
        target_u = us.reshape(-1)[sel]
        target_v_local = vs.reshape(-1)[sel]
        target_w_flat = tri_target_w[s:e][:, None, None].expand(-1, max_h, max_w).reshape(-1)[sel]
        target_v_global = winding_offsets_t[target_w_flat] + target_v_local
        scroll_zyxs[target_u, target_v_global] = interp.reshape(-1, 3)[sel]


@torch.inference_mode()
def _build_spliced_overlay(
    scroll_zyxs,
    num_thetas_by_winding,
    z0,
    grid_spacing,
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    satisfied_patches,
    boundary_satisfied_patches,
    target_winding_idx_per_patch,
):
    device = scroll_zyxs.device
    dr = dr_per_winding.detach()
    num_windings = len(num_thetas_by_winding)
    winding_offsets_t = torch.cat([
        torch.zeros([1], dtype=torch.long, device=device),
        torch.cumsum(torch.tensor(num_thetas_by_winding, dtype=torch.long, device=device), dim=0),
    ])
    num_thetas_t = torch.tensor(num_thetas_by_winding, dtype=torch.long, device=device)

    all_quad_uvs = []
    all_quad_scrolls = []
    all_target_w = []
    chunk = 65536

    for patch_idx, patch in enumerate(patches):
        if not (bool(satisfied_patches[patch_idx]) or bool(boundary_satisfied_patches[patch_idx])):
            continue
        target_winding_idx = target_winding_idx_per_patch[patch_idx].to(device)
        quad_mask = (target_winding_idx >= 0) & (target_winding_idx < num_windings)
        if not quad_mask.any():
            continue

        patch_zyxs = patch.zyxs.to(device=device, dtype=torch.float32)
        Hv, Wv = patch_zyxs.shape[:2]

        def chunked_transform(flat):
            pieces = []
            for s in range(0, flat.shape[0], chunk):
                pieces.append(slice_to_spiral_transform(flat[s : s + chunk]))
            return torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]

        vertex_spiral = chunked_transform(patch_zyxs.reshape(-1, 3)).reshape(Hv, Wv, 3)
        v_theta, _, v_shifted = get_theta_and_radii(vertex_spiral[..., 1:], dr_per_winding)
        v_winding_raw = (v_shifted / dr).round().to(torch.int64)

        quad_center_scroll = (
            patch_zyxs[:-1, :-1]
            + patch_zyxs[1:, :-1]
            + patch_zyxs[:-1, 1:]
            + patch_zyxs[1:, 1:]
        ) / 4
        center_spiral = chunked_transform(quad_center_scroll.reshape(-1, 3)).reshape(*quad_center_scroll.shape)
        c_theta, _, _ = get_theta_and_radii(center_spiral[..., 1:], dr_per_winding)

        qi, qj = torch.where(quad_mask)
        w_target = target_winding_idx[qi, qj].to(torch.float32)
        ref_full = c_theta[qi, qj] + w_target * (2 * np.pi)

        vi = torch.stack([qi, qi, qi + 1, qi + 1], dim=-1)
        vj = torch.stack([qj, qj + 1, qj, qj + 1], dim=-1)
        vert_spiral = vertex_spiral[vi, vj]
        vert_scroll = patch_zyxs[vi, vj]
        vert_theta = v_theta[vi, vj]
        vert_w_raw = v_winding_raw[vi, vj].to(torch.float32)
        vert_full = vert_theta + vert_w_raw * (2 * np.pi)
        diff = vert_full - ref_full[:, None]
        vert_full_snapped = vert_full - torch.round(diff / (2 * np.pi)) * (2 * np.pi)

        u_coords = (vert_spiral[..., 0] - z0) / grid_spacing
        theta_step_per_quad = grid_spacing / ((w_target + 0.5) * dr)
        v_coords = (vert_full_snapped - w_target[:, None] * (2 * np.pi)) / theta_step_per_quad[:, None]

        all_quad_uvs.append(torch.stack([u_coords, v_coords], dim=-1))
        all_quad_scrolls.append(vert_scroll)
        all_target_w.append(target_winding_idx[qi, qj])

    if not all_quad_uvs:
        return

    quad_uvs = torch.cat(all_quad_uvs, dim=0)
    quad_scrolls = torch.cat(all_quad_scrolls, dim=0)
    quad_target_w = torch.cat(all_target_w, dim=0)
    Nq = quad_uvs.shape[0]

    tri_local = torch.tensor([[0, 1, 3], [0, 3, 2]], device=device, dtype=torch.long)
    quad_repeat = torch.arange(Nq, device=device).repeat_interleave(2)
    tri_local_flat = tri_local.unsqueeze(0).expand(Nq, -1, -1).reshape(-1, 3)
    tri_uvs = quad_uvs[quad_repeat[:, None].expand(-1, 3), tri_local_flat]
    tri_scrolls = quad_scrolls[quad_repeat[:, None].expand(-1, 3), tri_local_flat]
    tri_target_w = quad_target_w[quad_repeat]

    _rasterize_triangles_into_mesh(
        tri_uvs, tri_scrolls, tri_target_w,
        scroll_zyxs, winding_offsets_t, num_thetas_t,
    )


@torch.inference_mode()
def save_mesh(
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    unattached_pcl_strips,
    out_path,
    cfg,
    z_begin,
    z_end,
    voxel_size_um,
    get_or_build_unattached_pcl_flat,
    get_patch_satisfied_areas,
    tracks=(),
    run_tag=None,
    name='mesh',
):
    (min_winding_idx, max_winding_idx), _, _ = compute_winding_range_and_input_extents(
        slice_to_spiral_transform,
        dr_per_winding,
        patches,
        unattached_pcl_strips,
        cfg,
        z_begin,
        z_end,
        get_or_build_unattached_pcl_flat,
        authoritative_zyx_lines=tracks,
    )
    if cfg['shell_outer_winding_idx'] is not None:
        max_winding_idx = min(max_winding_idx, cfg['shell_outer_winding_idx'])
    print(f'save_mesh {name}: winding range [{min_winding_idx}, {max_winding_idx})')
    grid_spacing = cfg['output_step_size']
    z_margin = cfg['flow_bounds_z_margin']
    spiral_yxs_by_winding = get_spiral_yxs(max_winding_idx, dr_per_winding, grid_spacing, group_by_winding=True)
    num_thetas_by_winding = [len(yxs_for_winding) for yxs_for_winding in spiral_yxs_by_winding]
    spiral_yxs = torch.cat(spiral_yxs_by_winding, dim=0)
    z0 = z_begin - z_margin
    spiral_zs = torch.arange(z0, z_end + z_margin, grid_spacing, dtype=torch.float32, device=spiral_yxs.device)
    spiral_zyxs = torch.cat([spiral_zs[:, None, None].expand(-1, spiral_yxs.shape[0], 1), spiral_yxs[None, :, :].expand(spiral_zs.shape[0], -1, 2)], dim=-1)
    chunk = 65536
    flat_spiral_zyxs = spiral_zyxs.reshape(-1, 3)
    scroll_pieces = []
    for start in range(0, flat_spiral_zyxs.shape[0], chunk):
        scroll_pieces.append(slice_to_spiral_transform.inv(flat_spiral_zyxs[start : start + chunk]))
    scroll_zyxs = torch.cat(scroll_pieces, dim=0).reshape(*spiral_zyxs.shape)

    out_of_roi = (scroll_zyxs[..., 0] < z_begin) | (scroll_zyxs[..., 0] >= z_end)
    scroll_zyxs[out_of_roi] = -1.0

    spliced_scroll_zyxs = scroll_zyxs.clone()
    # Splicing is deliberately more permissive than the reported satisfaction
    # metrics: it should accept a mostly aligned patch without relabelling that
    # patch as fully satisfied in the user-facing metrics.
    splicing_metrics_overrides = {
        'satisfaction_radius_tolerance': 0.495,
        'satisfaction_distance_tolerance': 12.0,
        'satisfied_patch_quad_fraction': 0.90,
    }
    satisfied_patches, _, _, _, boundary_satisfied_patches, target_winding_idx_per_patch = get_patch_satisfied_areas(
        slice_to_spiral_transform, dr_per_winding, patches,
        metrics_overrides=splicing_metrics_overrides,
    )
    _build_spliced_overlay(
        spliced_scroll_zyxs, num_thetas_by_winding, z0, grid_spacing,
        slice_to_spiral_transform, dr_per_winding,
        patches,
        satisfied_patches, boundary_satisfied_patches, target_winding_idx_per_patch,
    )

    step_size = grid_spacing
    tag_suffix = f'_{run_tag}' if run_tag else ''
    out_dir = f'{out_path}/meshes/{name}{tag_suffix}'
    os.makedirs(out_dir, exist_ok=True)
    for uuid_suffix, variant_zyxs in [('', scroll_zyxs), ('_spliced', spliced_scroll_zyxs)]:
        offset = 0
        for winding_idx, num_thetas in enumerate(tqdm(num_thetas_by_winding, desc=f'saving winding patches ({name}{uuid_suffix})')):
            if num_thetas >= 2 and winding_idx >= min_winding_idx:
                winding_slice = variant_zyxs[:, offset:offset + num_thetas]
                invalid_mask = (winding_slice == -1.0).all(dim=-1).cpu().numpy()
                winding_zyxs = winding_slice.cpu().numpy().astype(np.float32)
                winding_zyxs[invalid_mask] = -1.0
                save_tifxyz(
                    winding_zyxs,
                    out_dir,
                    uuid=f'w{winding_idx:03d}{uuid_suffix}{tag_suffix}',
                    step_size=step_size,
                    voxel_size_um=voxel_size_um,
                    source=f'fit_spiral {name}{uuid_suffix}',
                )
            offset += num_thetas


@torch.inference_mode()
def save_combined_preview(
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    unattached_pcl_strips,
    generation_path,
    cfg,
    z_begin,
    z_end,
    voxel_size_um,
    get_or_build_unattached_pcl_flat,
    tracks=(),
    *,
    surface_id,
):
    """Write the fast interactive raw preview as one disconnected surface."""
    (_, derived_upper), _, _ = compute_winding_range_and_input_extents(
        slice_to_spiral_transform,
        dr_per_winding,
        patches,
        unattached_pcl_strips,
        cfg,
        z_begin,
        z_end,
        get_or_build_unattached_pcl_flat,
        authoritative_zyx_lines=tracks,
    )
    configured_outer = cfg.get('shell_outer_winding_idx')
    exclusive_upper = derived_upper
    if configured_outer is not None:
        configured_upper = int(configured_outer) + 1
        exclusive_upper = (configured_upper if derived_upper <= int(cfg['output_first_winding'])
                           else min(exclusive_upper, configured_upper))
    first_winding = 10
    last_winding = int(exclusive_upper) - 1
    if last_winding < first_winding:
        raise RuntimeError(
            f'No preview winding is at or above {first_winding}; derived last winding is {last_winding}'
        )

    grid_spacing = int(cfg['output_step_size'])
    z_margin = int(cfg['flow_bounds_z_margin'])
    spiral_yxs_by_winding = get_spiral_yxs(
        last_winding + 1,
        dr_per_winding,
        grid_spacing,
        group_by_winding=True,
    )
    z0 = z_begin - z_margin
    spiral_zs = torch.arange(
        z0,
        z_end + z_margin,
        grid_spacing,
        dtype=torch.float32,
        device=dr_per_winding.device,
    )
    winding_grids = {}
    for winding in range(first_winding, last_winding + 1):
        yxs = spiral_yxs_by_winding[winding]
        if yxs.shape[0] < 2:
            raise RuntimeError(f'Preview winding {winding} has fewer than two theta samples')
        spiral = torch.cat([
            spiral_zs[:, None, None].expand(-1, yxs.shape[0], 1),
            yxs[None, :, :].expand(spiral_zs.shape[0], -1, 2),
        ], dim=-1)
        flat = spiral.reshape(-1, 3)
        pieces = []
        for start in range(0, flat.shape[0], 65536):
            pieces.append(slice_to_spiral_transform.inv(flat[start:start + 65536]))
        scroll = torch.cat(pieces, dim=0).reshape_as(spiral)
        outside = (scroll[..., 0] < z_begin) | (scroll[..., 0] >= z_end)
        scroll[outside] = -1.0
        winding_grids[winding] = scroll.cpu().numpy().astype(np.float32)

    manifest = save_combined_tifxyz(
        winding_grids,
        generation_path,
        surface_id,
        grid_spacing,
        voxel_size_um,
        source='fit_spiral interactive preview',
        first_winding=first_winding,
    )
    return manifest


def load_patches(patches_path, segment_id_filter=lambda s: True):
    patches = {}
    failed_count = 0
    for entry in sorted(os.listdir(patches_path)):
        segment_path = os.path.join(patches_path, entry)
        meta_path = os.path.join(segment_path, 'meta.json')
        if not os.path.isdir(segment_path) or not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            print(f'Warning: failed to read {meta_path}: {e}')
            continue
        if meta.get('format') != 'tifxyz' or not segment_id_filter(entry):
            continue
        try:
            patches[entry] = load_tifxyz(segment_path)
        except Exception as e:
            print(f'Failed to load segment {entry}: {e}')
            failed_count += 1
    if not patches:
        raise RuntimeError('No patches could be loaded')
    print(f'Loaded {len(patches)} patches, {failed_count} failed')
    return patches


def erode_patch_valid_region(patch, num_cells):
    """Erode the patch's valid-vertex region inward by `num_cells` grid cells."""
    valid = patch.valid_vertex_mask.cpu().numpy()
    eroded = scipy.ndimage.binary_erosion(valid, iterations=num_cells, border_value=0)
    remove = valid & ~eroded
    if not remove.any():
        return True
    patch.zyxs[torch.from_numpy(remove)] = -1.0
    new_valid_vertex = torch.any(patch.zyxs != -1, dim=-1)
    new_valid_quad = (
        new_valid_vertex[:-1, :-1] & new_valid_vertex[1:, :-1]
        & new_valid_vertex[:-1, 1:] & new_valid_vertex[1:, 1:]
    )
    if not bool(new_valid_quad.any()):
        return False
    patch.__post_init__()
    return True


def _segmented_median_per_strip(ctx):
    # Segmented median: sort the flat values with a composite key
    # (strip_id-major, normalised_radii-minor) so values for each strip end
    # up contiguous and sorted within their range.
    normalised_radii = ctx['normalised_radii']
    strip_id = ctx['strip_id']
    starts = ctx['starts']
    lengths = ctx['lengths']
    S = ctx['S']
    device = ctx['device']
    if normalised_radii.numel() == 0:
        return torch.zeros(S, dtype=normalised_radii.dtype, device=device)

    val_min = normalised_radii.min().to(torch.float64)
    val_max = normalised_radii.max().to(torch.float64)
    val_range = (val_max - val_min) + 1.0
    composite = (
        strip_id.to(torch.float64) * val_range
        + (normalised_radii.to(torch.float64) - val_min)
    )
    order = torch.argsort(composite)
    sorted_norm = normalised_radii[order]
    median_indices = starts[:-1] + (lengths - 1) // 2
    return sorted_norm[median_indices]
