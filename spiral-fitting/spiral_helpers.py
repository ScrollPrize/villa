import os
import time

import numpy as np
import scipy.ndimage
import torch
from tqdm import tqdm

from fit_session import fit_level_scale
from sample_spiral import get_spiral_yxs, get_theta_and_radii
from surface_orientation import GridLayout, export_metadata
from tifxyz import save_tifxyz, save_combined_tifxyz


def patch_intersects_z_roi(patch, z_begin, z_end):
    zs = patch.valid_zyxs[..., 0]
    if zs.numel() == 0:
        return False
    return bool(((zs >= z_begin) & (zs < z_end)).any().item())


def largest_patch_quad_component(mask):
    """Return only the largest 8-connected True component of a quad mask."""
    mask = np.asarray(mask, dtype=bool)
    component_labels, num_components = scipy.ndimage.label(
        mask, structure=np.ones((3, 3), dtype=bool))
    if num_components <= 1:
        return mask.copy()
    component_sizes = np.bincount(component_labels.reshape(-1))
    component_sizes[0] = 0
    return component_labels == int(component_sizes.argmax())


def patch_sampling_quad_mask(patch, z_begin, z_end, z_margin):
    """The quads every patch loss samples and resolves annotations on.

    Valid quads (so bilinear interpolation at (i + di, j + dj) is defined for
    di, dj in [0, 1)) whose mean z lies in [z_begin - z_margin, z_end +
    z_margin), falling back to all valid quads when none does, reduced to
    their largest 8-connected component.
    """
    valid_quad_mask_np = patch.valid_quad_mask.cpu().numpy()
    # Restrict sampling to quads whose representative z is in [z_begin, z_end),
    # so patch-loss tracks don't waste samples outside the optimisation ROI.
    zyxs_z_np = patch.zyxs[..., 0].cpu().numpy()
    quad_zs_np = (zyxs_z_np[:-1, :-1] + zyxs_z_np[1:, :-1] + zyxs_z_np[:-1, 1:] + zyxs_z_np[1:, 1:]) / 4
    z_in_roi_np = (
            (quad_zs_np >= z_begin - z_margin)
            & (quad_zs_np < z_end + z_margin)
    )
    in_roi_quad_mask_np = valid_quad_mask_np & z_in_roi_np
    if not in_roi_quad_mask_np.any():
        # Fallback if no quad falls in the ROI; should be rare since patches
        # entirely outside the z-ROI are dropped earlier.
        in_roi_quad_mask_np = valid_quad_mask_np
    # Every patch loss and its cached theta lift use one connected
    # surface. Ignore detached islands rather than inventing an
    # integer winding offset between components. Eight-connectivity
    # matches both patch edge topology and the former Dijkstra graph.
    return largest_patch_quad_component(in_roi_quad_mask_np)


def prepare_patch_for_fit(patch, z_begin, z_end, erode_cells_default):
    """Erode a loaded patch for fitting; return why it was dropped, or None.

    Applies the patch's own erosion request (else ``erode_cells_default``)
    in place, then requires a valid vertex inside [z_begin, z_end).
    """
    cells_to_erode = patch.erosion_cells(erode_cells_default)
    if (cells_to_erode > 0
            and not erode_patch_valid_region(patch, cells_to_erode)):
        return 'erosion'
    if not patch_intersects_z_roi(patch, z_begin, z_end):
        return 'z ROI after erosion'
    patch.release_derived_caches()
    return None


def scale_counts_for_z_range(
    config,
    z_begin,
    z_end,
    reference_z_range_num_slices,
    z_range_scaled_count_keys,
):
    """Scale per-step sample counts with the z-range (never below 1)."""
    num_slices = z_end - z_begin
    scale = num_slices / reference_z_range_num_slices
    for key in z_range_scaled_count_keys:
        config[key] = max(1, round(config[key] * scale))
    return scale, num_slices


# All per-step sample-count defaults are tuned for a 9500-slice z-range;
# scale_counts_for_z_range() scales them relative to this reference.
REFERENCE_Z_RANGE_NUM_SLICES = 9500


def scale_and_split_counts(config, z_begin, z_end, count_keys, world_size=None):
    """Scale per-step counts for the z-range, then split them across ranks.

    The shared scale-then-split sequence used by both the headless CLI
    (fit_spiral.__main__) and the interactive runtime (spiral_runtime).
    The callers pass their own key sets: the CLI keeps its historical
    tuple while the runtime derives keys from the Config catalog's
    scale_with_z fields. ``world_size`` is passed explicitly by callers that
    hold a DistributedContext; None falls back to the installed process
    context. Returns (scale, num_slices, split_divisor) for caller-side
    reporting.
    """
    from ddp_helpers import split_counts_across_ranks

    scale, num_slices = scale_counts_for_z_range(
        config, z_begin, z_end,
        REFERENCE_Z_RANGE_NUM_SLICES, count_keys,
    )
    split_divisor = split_counts_across_ranks(
        config, count_keys, world_size=world_size)
    return scale, num_slices, split_divisor


def classify_fiber_hv(hv_classification, zyxs, *, min_z_fraction,
                      min_auto_certainty):
    """Return 'V', 'H', or None for one fiber polyline.

    Precedence: VC3D's manual H/V tag, then its automatic tag when the
    recorded certainty (0..1) reaches ``min_auto_certainty``, then a geometric
    fallback on the polyline itself: the strip is vertical when its z extent
    covers at least ``min_z_fraction`` of its path length, horizontal when the
    z extent covers at most ``1 - min_z_fraction`` of it, and untagged
    otherwise. These tags drive radial offsets.
    """
    hv = hv_classification if isinstance(hv_classification, dict) else {}
    manual = str(hv.get('manual_tag') or '').strip().upper()
    if manual in ('H', 'V'):
        return manual
    automatic = str(hv.get('automatic_tag') or '').strip().upper()
    if automatic in ('H', 'V'):
        try:
            certainty = float(hv.get('automatic_certainty', 0.0))
        except (TypeError, ValueError):
            certainty = 0.0
        if certainty >= min_auto_certainty:
            return automatic
    zyxs = np.asarray(zyxs, dtype=np.float64)
    if zyxs.ndim != 2 or len(zyxs) < 2:
        return None
    path_length = float(np.linalg.norm(np.diff(zyxs, axis=0), axis=-1).sum())
    if path_length <= 0.0:
        return None
    z_fraction = float(zyxs[:, 0].max() - zyxs[:, 0].min()) / path_length
    if z_fraction >= min_z_fraction:
        return 'V'
    if z_fraction <= 1.0 - min_z_fraction:
        return 'H'
    return None


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


#: Points transformed per input kind when a caller asks for an estimated
#: winding range rather than an exact one. Each transform call is an RK4 flow
#: integration, so an exact pass over a multi-million-point track database
#: costs minutes; see ``point_budget`` below.
ESTIMATED_WINDING_RANGE_POINT_BUDGET = 1_000_000

#: Points a patch keeps when the budget is spread over many patches. Patches
#: are small local grids, so a floor per patch matters more than an exactly
#: honoured total.
MIN_ESTIMATED_POINTS_PER_PATCH = 4096


def _stride_to_budget(points, budget):
    """Subsample ``points`` along its first dimension to at most ``budget``.

    Striding is sound here because the winding index is
    ``round(shifted_radius / dr_per_winding)`` with ``dr_per_winding`` in the
    hundreds of voxels, while consecutive samples within a patch grid or along
    a track lie a single output step apart and the transform is a
    diffeomorphism with an O(1) local Jacobian. Dropping every ``k``-th sample
    therefore moves the observed extreme by roughly ``k * spacing / dr``
    windings, well under the ``output_winding_margin`` the caller already adds,
    and it can only ever *under*-report the range, never overstate it.
    """
    if budget is None or points.shape[0] <= budget:
        return points
    return points[::-(-points.shape[0] // int(budget))]


@torch.inference_mode()
def compute_winding_range_and_input_extents(
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    cfg,
    z_begin,
    z_end,
    authoritative_zyx_lines=(),
    point_budget=None,
):
    """Compute output winding range plus max observed radius/winding per patch.

    ``point_budget`` caps how many points of each input kind are actually
    transformed (per patch for patches, in total for tracks). ``None``
    transforms every point, which is what a run's authoritative output needs;
    a budget trades a fraction of a winding of accuracy for a bounded,
    dataset-size-independent cost, which is what an interactive preview needs.
    Per-patch extents are estimated from the same subsample, so a caller that
    needs them exactly must leave the budget unset.
    """
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

    patch_budget = (
        None if point_budget is None
        else max(MIN_ESTIMATED_POINTS_PER_PATCH,
                 int(point_budget) // max(1, len(patches))))
    # The flow integration is elementwise, so the chunk size only trades
    # transient memory against kernel-launch overhead.
    chunk = 1 << 18

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
        spiral_zyxs = transform_in_chunks(
            _stride_to_budget(quad_center_zyxs[mask], patch_budget))
        _, radius, shifted_radius = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
        winding_indices = (shifted_radius / dr).round().to(torch.int64).clamp_min(0)
        update_from_winding_indices(winding_indices)
        patch_extents[patch_index] = (
            float(radius.max().item()),
            int(winding_indices.max().item()),
        )

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

    lines = authoritative_zyx_lines or ()
    if point_budget is not None and not isinstance(lines, (list, tuple)):
        # A track collection is walked line by line below. Thin the walk itself
        # rather than only the transform: with a budget the estimate is drawn
        # from a uniform sample of the whole collection, never from its first
        # tracks (which are not representative of the outermost winding).
        lines = list(lines)
    if point_budget is not None and len(lines) > 1:
        expected_points_per_line = max(1.0, float(np.mean(
            [len(line) for line in lines[:min(len(lines), 1024)]]) or 1.0))
        wanted_lines = max(1, int(int(point_budget) / expected_points_per_line))
        if wanted_lines < len(lines):
            lines = lines[::-(-len(lines) // wanted_lines)]

    for line in lines:
        if torch.is_tensor(line):
            # Keep tensor callers supported without trying to convert CUDA data
            # through NumPy.  Flush host-side points first to preserve the bound.
            flush_pending_track_points()
            # Thin before the host-to-device copy, coarsely, so the ROI filter
            # still sees enough points to leave the budget's worth behind.
            zyxs = _stride_to_budget(
                line.reshape(-1, 3),
                None if point_budget is None else 8 * int(point_budget))
            zyxs = zyxs.to(device=device, dtype=torch.float32)
            in_roi = (zyxs[:, 0] >= z_begin) & (zyxs[:, 0] < z_end)
            if in_roi.any():
                update_from_track_points(
                    _stride_to_budget(zyxs[in_roi], point_budget))
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
    return output_winding_range, patch_extents


def _infer_shell_outer_winding_idx(
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    cfg,
    z_begin,
    z_end,
):
    _, patch_extents = compute_winding_range_and_input_extents(
        slice_to_spiral_transform,
        dr_per_winding,
        patches,
        cfg,
        z_begin,
        z_end,
    )
    observed_max = None
    for _, max_w in patch_extents:
        if max_w is None:
            continue
        observed_max = max_w if observed_max is None else max(observed_max, max_w)
    if observed_max is None:
        observed_max = cfg['output_first_winding']
    return int(observed_max + cfg['shell_outer_winding_margin'])


def _resolve_shell_outer_winding_idx(cfg):
    # Winding bound shared by every sampler that integrates over the spiral
    # cylinder: the dense lasagna losses, the symmetric Dirichlet
    # regulariser and the native min_spacing barrier. Resolved once per
    # run from the config; the shell branch may still override it with the
    # inferred value. None disables those samplers (they early-return zero),
    # whatever their weights: _structurally_disabled_dense_weight_keys
    # reports that combination so the run can warn instead of staying silent.
    configured = cfg['shell_outer_winding_idx']
    if configured is None:
        return None
    try:
        resolved = int(configured)
    except (TypeError, ValueError):
        raise ValueError(
            f'shell_outer_winding_idx must be an integer >= 2 or None, '
            f'got {configured!r}')
    if resolved < 2:
        # sample_spiral_surface_frame draws windings from [1, idx); 0 and 1
        # crash multinomial/arange at the first step with an opaque error.
        raise ValueError(
            f'shell_outer_winding_idx must be an integer >= 2 or None, '
            f'got {configured!r}')
    return resolved


def resolve_outer_winding_idx_and_notes(cfg, shell_active, infer_outer_winding_idx):
    """Resolve the run's outer winding index and the lines to print for it.

    ``infer_outer_winding_idx`` is a zero-argument callable, invoked only
    when shell losses are active and the config key is None (the historical
    inference path). Returns ``(index_or_none, notes)``.
    """
    idx = _resolve_shell_outer_winding_idx(cfg)
    notes = []
    if shell_active:
        if idx is None:
            idx = int(infer_outer_winding_idx())
            notes.append(f'inferred shell_outer_winding_idx = {idx}')
        else:
            notes.append(f'using configured shell_outer_winding_idx = {idx}')
    elif idx is not None:
        notes.append(
            'no outer-shell losses; using configured shell_outer_winding_idx '
            f'= {idx} for the dense and regularisation losses')
    if idx is not None:
        min_gap = idx + 3
        capacity = cfg.get(
            'model_gap_expander_capacity_windings',
            cfg['model_gap_expander_num_windings'])
        if capacity < min_gap:
            raise ValueError(
                f'shell_outer_winding_idx {idx} requires '
                f'model_gap_expander_capacity_windings >= {min_gap}, got '
                f'{capacity}; increase the allocated capacity or lower the '
                'active shell_outer_winding_idx')
    return idx, notes


# Every loss weight whose term samples out to shell_outer_winding_idx and is
# therefore silently zero while that index is unresolved.
_DENSE_WEIGHT_KEYS_NEEDING_OUTER_WINDING_IDX = (
    'loss_weight_dense_normals',
    'loss_weight_dense_spacing',
    'loss_weight_dense_spacing_density',
    'loss_weight_min_spacing',
    'loss_weight_sym_dirichlet',
)


def _structurally_disabled_dense_weight_keys(cfg, shell_outer_winding_idx):
    """Nonzero weights that cannot produce a loss because no outer winding
    index could be resolved (config key is None and no outer shell inferred
    one)."""
    if shell_outer_winding_idx is not None:
        return ()
    return tuple(
        key for key in _DENSE_WEIGHT_KEYS_NEEDING_OUTER_WINDING_IDX
        if cfg.get(key, 0.0) > 0
    )


def _warn_if_inputs_exceed_flow_bounds(
    patch_ids,
    patch_extents,
    flow_field_radius,
    cfg,
):
    gap_expander_num_windings = cfg.get(
        'model_gap_expander_capacity_windings',
        cfg['model_gap_expander_num_windings'])

    over_radius_patches = []
    over_winding_patches = []
    for pid, (max_r, max_w) in zip(patch_ids, patch_extents):
        if max_r is None:
            continue
        if max_r > flow_field_radius:
            over_radius_patches.append((pid, max_r))
        if max_w >= gap_expander_num_windings:
            over_winding_patches.append((pid, max_w))

    def _print_offenders(kind, value_label, threshold_label, patches, fmt):
        if not patches:
            return
        print(f'WARNING: {len(patches)} patch(es) have {value_label} exceeding {threshold_label}:')
        for pid, v in sorted(patches, key=lambda e: -e[1])[:10]:
            print(f'  patch {pid}: max {kind} {fmt(v)}')
        if len(patches) > 10:
            print(f'  ... and {len(patches) - 10} more patches')

    _print_offenders(
        'spiral radius', 'spiral-space radius', f'flow_bounds_radius ({flow_field_radius})',
        over_radius_patches, lambda v: f'{v:.1f}',
    )
    _print_offenders(
        'winding idx', 'winding index',
        f'gap_expander_capacity_windings ({gap_expander_num_windings})',
        over_winding_patches, lambda v: f'{v}',
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
    patch_atlas,
    patch_evaluation,
    *,
    first_winding=0,
):
    """Overwrite ``scroll_zyxs`` with patch geometry wherever a patch that
    passes the splicing profile covers the grid.

    ``scroll_zyxs`` is ``[z, theta, 3]`` over the windings
    ``first_winding .. first_winding + len(num_thetas_by_winding) - 1``
    laid side by side along the theta axis; ``num_thetas_by_winding[k]``
    is the theta count of winding ``first_winding + k``.
    """
    started = time.perf_counter()
    device = scroll_zyxs.device
    dr = dr_per_winding.detach()
    first_winding = int(first_winding)
    num_windings = first_winding + len(num_thetas_by_winding)
    # Index the offset/count tables by absolute winding: windings below the
    # grid's first one have no columns and are excluded from eligibility.
    num_thetas_full = [0] * first_winding + [int(n) for n in num_thetas_by_winding]
    winding_offsets_t = torch.cat([
        torch.zeros([1], dtype=torch.long, device=device),
        torch.cumsum(torch.tensor(num_thetas_full, dtype=torch.long, device=device), dim=0),
    ])
    num_thetas_t = torch.tensor(num_thetas_full, dtype=torch.long, device=device)

    profile = patch_evaluation.profiles['splicing']
    eligible_patches = (profile.satisfied_patches
                        | profile.boundary_satisfied_patches)
    packed_patch_indices = patch_evaluation.patch_indices
    target_winding_cpu = patch_evaluation.target_winding_indices.cpu()
    eligible = (eligible_patches[packed_patch_indices]
                & (target_winding_cpu >= first_winding)
                & (target_winding_cpu < num_windings))
    eligible_indices = torch.where(eligible)[0]
    if eligible_indices.numel() == 0:
        print('splice rasterization: 0 eligible quads, 0 vertex batches, 0.00s')
        return
    tri_local = torch.tensor(
        [[0, 1, 3], [0, 3, 2]], device=device, dtype=torch.long)
    quad_batch_size = 16384
    transform_chunk_size = 65536
    batch_count = 0
    unique_vertex_total = 0
    for batch_begin in range(0, eligible_indices.numel(), quad_batch_size):
        packed_ids_cpu = eligible_indices[
            batch_begin:batch_begin + quad_batch_size]
        corner_ids_cpu = patch_evaluation.corner_vertex_ids[packed_ids_cpu]
        unique_ids_cpu, inverse_cpu = torch.unique(
            corner_ids_cpu.reshape(-1), sorted=True, return_inverse=True)
        unique_scroll = patch_atlas.vertex_zyxs(unique_ids_cpu.to(device=device))
        spiral_pieces = []
        for begin in range(0, unique_scroll.shape[0], transform_chunk_size):
            spiral_pieces.append(slice_to_spiral_transform(
                unique_scroll[begin:begin + transform_chunk_size]))
        unique_spiral = (torch.cat(spiral_pieces) if len(spiral_pieces) > 1
                         else spiral_pieces[0])
        inverse = inverse_cpu.to(device=device)
        quad_scrolls = unique_scroll[inverse].reshape(-1, 4, 3)
        quad_spirals = unique_spiral[inverse].reshape(-1, 4, 3)
        vertex_theta, _, vertex_shifted = get_theta_and_radii(
            quad_spirals[..., 1:], dr_per_winding)
        vertex_winding = (vertex_shifted / dr).round().to(torch.float32)

        # The evaluation's per-quad tensors may live on the host (after an
        # out-of-memory fallback): gather there, move only the batch.
        quad_target_w = patch_evaluation.target_winding_indices[
            packed_ids_cpu.to(device=patch_evaluation.target_winding_indices.device)
        ].to(device=device)
        target_w_float = quad_target_w.to(torch.float32)
        center_theta = patch_evaluation.center_theta[
            packed_ids_cpu.to(device=patch_evaluation.center_theta.device)
        ].to(device=device)
        reference_full = center_theta + target_w_float * (2 * np.pi)
        vertex_full = vertex_theta + vertex_winding * (2 * np.pi)
        vertex_full -= torch.round(
            (vertex_full - reference_full[:, None]) / (2 * np.pi)
        ) * (2 * np.pi)
        u_coords = (quad_spirals[..., 0] - z0) / grid_spacing
        theta_step = grid_spacing / ((target_w_float + 0.5) * dr)
        v_coords = ((vertex_full - target_w_float[:, None] * (2 * np.pi))
                    / theta_step[:, None])
        quad_uvs = torch.stack([u_coords, v_coords], dim=-1)

        num_quads = quad_uvs.shape[0]
        quad_repeat = torch.arange(
            num_quads, device=device).repeat_interleave(2)
        tri_indices = tri_local.unsqueeze(0).expand(
            num_quads, -1, -1).reshape(-1, 3)
        tri_uvs = quad_uvs[
            quad_repeat[:, None].expand(-1, 3), tri_indices]
        tri_scrolls = quad_scrolls[
            quad_repeat[:, None].expand(-1, 3), tri_indices]
        _rasterize_triangles_into_mesh(
            tri_uvs, tri_scrolls, quad_target_w[quad_repeat],
            scroll_zyxs, winding_offsets_t, num_thetas_t)
        unique_vertex_total += unique_ids_cpu.numel()
        batch_count += 1
    elapsed = time.perf_counter() - started
    print('splice rasterization: '
          f'{eligible_indices.numel():,} eligible quads, '
          f'{unique_vertex_total:,} transformed unique vertices in '
          f'{batch_count} batches, {elapsed:.2f}s')


@torch.inference_mode()
def save_mesh(
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    out_path,
    cfg,
    z_begin,
    z_end,
    voxel_size_um,
    *,
    winding_range,
    patch_satisfaction_evaluation,
    patch_atlas,
    z_direction_is_top_to_bottom,
    tracks=(),
    run_tag=None,
    name='mesh',
    progress=None,
):
    """Write one tifxyz per winding in the export layout (surface_orientation)
    for ``z_direction_is_top_to_bottom``."""
    min_winding_idx, max_winding_idx = winding_range
    if cfg['shell_outer_winding_idx'] is not None:
        max_winding_idx = min(max_winding_idx, cfg['shell_outer_winding_idx'])
    print(f'save_mesh {name}: winding range [{min_winding_idx}, {max_winding_idx})')
    grid_spacing = cfg['output_step_size']
    z_margin = cfg['model_flow_bounds_z_margin']
    spiral_yxs_by_winding = get_spiral_yxs(max_winding_idx, dr_per_winding, grid_spacing, group_by_winding=True,
                                           device=dr_per_winding.device)
    num_thetas_by_winding = [len(yxs_for_winding) for yxs_for_winding in spiral_yxs_by_winding]
    spiral_yxs = torch.cat(spiral_yxs_by_winding, dim=0)
    z0 = z_begin - z_margin
    spiral_zs = torch.arange(z0, z_end + z_margin, grid_spacing, dtype=torch.float32, device=spiral_yxs.device)
    spiral_zyxs = torch.cat([spiral_zs[:, None, None].expand(-1, spiral_yxs.shape[0], 1), spiral_yxs[None, :, :].expand(spiral_zs.shape[0], -1, 2)], dim=-1)
    chunk = 65536
    flat_spiral_zyxs = spiral_zyxs.reshape(-1, 3)
    scroll_pieces = []
    transform_chunk_total = (
        flat_spiral_zyxs.shape[0] + chunk - 1) // chunk
    if progress is not None:
        progress.begin(
            'finalizing', 'Transforming final mesh',
            step=0, total_steps=transform_chunk_total, unit='chunks')
    for chunk_number, start in enumerate(
            range(0, flat_spiral_zyxs.shape[0], chunk), start=1):
        scroll_pieces.append(slice_to_spiral_transform.inv(flat_spiral_zyxs[start : start + chunk]))
        if progress is not None:
            progress.update(chunk_number)
    scroll_zyxs = torch.cat(scroll_pieces, dim=0).reshape(*spiral_zyxs.shape)

    out_of_roi = (scroll_zyxs[..., 0] < z_begin) | (scroll_zyxs[..., 0] >= z_end)
    scroll_zyxs[out_of_roi] = -1.0

    spliced_scroll_zyxs = scroll_zyxs.clone()
    _build_spliced_overlay(
        spliced_scroll_zyxs, num_thetas_by_winding, z0, grid_spacing,
        slice_to_spiral_transform, dr_per_winding,
        patch_atlas, patch_satisfaction_evaluation,
    )

    step_size = grid_spacing
    tag_suffix = f'_{run_tag}' if run_tag else ''
    out_dir = f'{out_path}/meshes/{name}{tag_suffix}'
    os.makedirs(out_dir, exist_ok=True)
    output_total = 2 * len(num_thetas_by_winding)
    output_done = 0
    layout = GridLayout.export(z_direction_is_top_to_bottom)
    layout_metadata = export_metadata(z_direction_is_top_to_bottom)
    if progress is not None:
        progress.begin(
            'finalizing', 'Writing final mesh windings',
            step=0, total_steps=output_total, unit='windings')
    for uuid_suffix, variant_zyxs in [('', scroll_zyxs), ('_spliced', spliced_scroll_zyxs)]:
        offset = 0
        for winding_idx, num_thetas in enumerate(tqdm(
                num_thetas_by_winding,
                desc=f'saving winding patches ({name}{uuid_suffix})',
                disable=progress is not None)):
            if num_thetas >= 2 and winding_idx >= min_winding_idx:
                winding_slice = variant_zyxs[:, offset:offset + num_thetas]
                invalid_mask = (winding_slice == -1.0).all(dim=-1).cpu().numpy()
                winding_zyxs = winding_slice.cpu().numpy().astype(np.float32)
                winding_zyxs[invalid_mask] = -1.0
                save_tifxyz(
                    layout.apply(winding_zyxs),
                    out_dir,
                    uuid=f'w{winding_idx:03d}{uuid_suffix}{tag_suffix}',
                    step_size=step_size,
                    voxel_size_um=voxel_size_um,
                    source=f'fit_spiral {name}{uuid_suffix}',
                    layout_metadata=layout_metadata,
                    coordinate_scale=fit_level_scale(cfg),
                )
            offset += num_thetas
            output_done += 1
            if progress is not None:
                progress.update(
                    output_done, detail=f'winding {winding_idx}{uuid_suffix}')


@torch.inference_mode()
def save_combined_preview(
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    generation_path,
    cfg,
    z_begin,
    z_end,
    voxel_size_um,
    tracks=(),
    *,
    surface_id,
    z_direction_is_top_to_bottom,
    base_shape_zyx=None,
    progress=None,
    input_extent_transform=None,
    patch_atlas=None,
    patch_satisfaction_evaluation=None,
):
    """Write the authoritative connected preview used by VC3D and Lasagna.

    The preview's outer bound is the configured ``shell_outer_winding_idx``
    whenever there is one: it is the winding every dense sampler already
    integrates out to, so the model is defined there, and taking it directly
    skips a pass that transformed every patch and track point through the
    flow ODE to derive a bound the configuration already states. Only a run
    that leaves the index unset derives the bound, and then from a bounded
    sample rather than from every point.

    ``slice_to_spiral_transform`` maps true scroll space to spiral space and
    its inverse places the surface. ``input_extent_transform`` (default: the
    same transform) is the one the inputs are read through when the bound is
    derived: after a constraint-bake reset the resident inputs live in baked
    space and are read through the live chain, while the surface is pulled
    back through the composed frozen+live chain.

    When ``patch_satisfaction_evaluation`` (from
    ``evaluate_patch_satisfaction_packed`` with the splicing profile) and its
    ``patch_atlas`` are given, the transformed surface is spliced onto the
    satisfied patches exactly as the final ``_spliced`` meshes are: wherever
    such a patch covers the grid, its own geometry replaces the model's. The
    caller passes ``None`` only when the resident patches are not in true
    scroll space (after a constraint bake), where splicing would write
    baked-space coordinates into a scroll-space surface.

    The surface is written in the export layout (surface_orientation) for
    ``z_direction_is_top_to_bottom``; the splice here and the loss maps work in
    spiral order, the loss maps mapping through the manifest's layout.
    """
    if input_extent_transform is None:
        input_extent_transform = slice_to_spiral_transform
    configured_outer = cfg.get('shell_outer_winding_idx')
    if configured_outer is not None:
        exclusive_upper = int(configured_outer) + 1
    else:
        (_, exclusive_upper), _ = compute_winding_range_and_input_extents(
            input_extent_transform,
            dr_per_winding,
            patches,
            cfg,
            z_begin,
            z_end,
            authoritative_zyx_lines=tracks,
            point_budget=ESTIMATED_WINDING_RANGE_POINT_BUDGET,
        )
    first_winding = 10
    last_winding = int(exclusive_upper) - 1
    if last_winding < first_winding:
        raise RuntimeError(
            f'No preview winding is at or above {first_winding}; last winding '
            f'is {last_winding} (from '
            f'{"configured shell_outer_winding_idx" if configured_outer is not None else "the derived input extent"})'
        )

    grid_spacing = int(cfg['output_step_size'])
    z_margin = int(cfg['model_flow_bounds_z_margin'])
    spiral_yxs_by_winding = get_spiral_yxs(
        last_winding + 1,
        dr_per_winding,
        grid_spacing,
        group_by_winding=True,
        device=dr_per_winding.device,
    )
    z0 = z_begin - z_margin
    spiral_zs = torch.arange(
        z0,
        z_end + z_margin,
        grid_spacing,
        dtype=torch.float32,
        device=dr_per_winding.device,
    )
    preview_windings = list(range(first_winding, last_winding + 1))
    winding_scrolls = []
    total_windings = len(preview_windings)
    if progress is not None:
        progress.begin(
            'exporting_preview', 'Transforming preview windings',
            step=0, total_steps=total_windings, unit='windings')
    for winding_number, winding in enumerate(preview_windings, start=1):
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
        winding_scrolls.append(scroll)
        if progress is not None:
            progress.update(winding_number, detail=f'winding {winding}')

    if patch_satisfaction_evaluation is not None:
        # Same splice as save_mesh's `_spliced` variant: the windings are
        # laid side by side along theta, the satisfied patches are
        # rasterized over them, then the grid is split back per winding.
        if progress is not None:
            progress.begin(
                'exporting_preview', 'Splicing patches into preview surface')
        num_thetas_by_winding = [
            int(scroll.shape[1]) for scroll in winding_scrolls]
        combined_scroll = torch.cat(winding_scrolls, dim=1)
        _build_spliced_overlay(
            combined_scroll, num_thetas_by_winding, z0, grid_spacing,
            slice_to_spiral_transform, dr_per_winding,
            patch_atlas, patch_satisfaction_evaluation,
            first_winding=first_winding,
        )
        winding_scrolls = list(
            torch.split(combined_scroll, num_thetas_by_winding, dim=1))
    winding_grids = {
        winding: scroll.cpu().numpy().astype(np.float32)
        for winding, scroll in zip(preview_windings, winding_scrolls)
    }

    if progress is not None:
        progress.begin(
            'exporting_preview', 'Writing preview surface',
            detail=f'{total_windings} windings')
    manifest = save_combined_tifxyz(
        winding_grids,
        generation_path,
        surface_id,
        grid_spacing,
        voxel_size_um,
        source='fit_spiral interactive preview',
        z_direction_is_top_to_bottom=z_direction_is_top_to_bottom,
        first_winding=first_winding,
        cleanup_erosion_cells=3,
        base_shape_zyx=base_shape_zyx,
        coordinate_scale=fit_level_scale(cfg),
    )
    return manifest


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
