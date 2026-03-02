"""Topology-aware post-processing for per-chunk finalization.

Adapted from the interference-with-global-interpolation notebook.
Runs the full topo pipeline (hysteresis thresholding → component analysis →
sheet fitting → cleanup) as an optional replacement for simple sigmoid/softmax
finalization in the blending pipeline.
"""

import warnings
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import griddata
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
    median_filter,
)
from scipy.spatial import cKDTree
from skimage.measure import euler_number, label
from skimage.morphology import ball, remove_small_objects

from numba import jit, prange


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class TopoPostprocessConfig:
    topo_t_low: float = 0.2
    topo_t_high: float = 0.83
    topo_z_radius: int = 1
    topo_xy_radius: int = 0
    topo_dust_min_size: int = 100
    topo_min_object_size: int = 1000
    topo_final_min_object_size: int = 2000
    topo_grid_resolution: int = 100
    topo_thickness: int = 3
    topo_smoothing: float = 1.0
    topo_overlap_buffer: int = 0
    topo_min_coverage: float = 0.65
    topo_min_dice: float = 0.7
    topo_max_distance: int = 10
    topo_samples_per_edge: int = 8
    topo_alt_t_lows: tuple = (0.5, 0.7)
    topo_border_crop: int = 3


# ============================================================================
# NUMBA-OPTIMIZED RASTERIZATION
# ============================================================================

@jit(nopython=True, fastmath=True)
def rasterize_triangle_numba(p1, p2, p3, volume):
    """Numba-optimized triangle rasterization."""
    min_z = max(0, int(np.floor(min(p1[0], p2[0], p3[0]))))
    max_z = min(volume.shape[0] - 1, int(np.ceil(max(p1[0], p2[0], p3[0]))))
    min_y = max(0, int(np.floor(min(p1[1], p2[1], p3[1]))))
    max_y = min(volume.shape[1] - 1, int(np.ceil(max(p1[1], p2[1], p3[1]))))
    min_x = max(0, int(np.floor(min(p1[2], p2[2], p3[2]))))
    max_x = min(volume.shape[2] - 1, int(np.ceil(max(p1[2], p2[2], p3[2]))))

    v0 = p2 - p1
    v1 = p3 - p1
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    denom = d00 * d11 - d01 * d01

    if abs(denom) < 1e-10:
        return

    inv_denom = 1.0 / denom

    for z in range(min_z, max_z + 1):
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                v2_0 = z - p1[0]
                v2_1 = y - p1[1]
                v2_2 = x - p1[2]
                d20 = v2_0 * v0[0] + v2_1 * v0[1] + v2_2 * v0[2]
                d21 = v2_0 * v1[0] + v2_1 * v1[1] + v2_2 * v1[2]
                v = (d11 * d20 - d01 * d21) * inv_denom
                w = (d00 * d21 - d01 * d20) * inv_denom
                u = 1.0 - v - w
                if u >= -0.01 and v >= -0.01 and w >= -0.01:
                    volume[z, y, x] = True


@jit(nopython=True, fastmath=True, parallel=True)
def rasterize_surface_numba(grid_points, volume, samples_per_edge=5):
    """Numba-optimized surface rasterization with parallel processing."""
    grid_resolution = grid_points.shape[0]

    for i in prange(grid_resolution - 1):
        for j in range(grid_resolution - 1):
            p1 = grid_points[i, j]
            p2 = grid_points[i+1, j]
            p3 = grid_points[i, j+1]
            p4 = grid_points[i+1, j+1]

            if (np.isnan(p1[0]) or np.isnan(p2[0]) or
                np.isnan(p3[0]) or np.isnan(p4[0])):
                continue

            for u_idx in range(samples_per_edge):
                u = u_idx / (samples_per_edge - 1) if samples_per_edge > 1 else 0.5
                for v_idx in range(samples_per_edge):
                    v = v_idx / (samples_per_edge - 1) if samples_per_edge > 1 else 0.5

                    point_0 = ((1-u)*(1-v)*p1[0] + u*(1-v)*p2[0] +
                              (1-u)*v*p3[0] + u*v*p4[0])
                    point_1 = ((1-u)*(1-v)*p1[1] + u*(1-v)*p2[1] +
                              (1-u)*v*p3[1] + u*v*p4[1])
                    point_2 = ((1-u)*(1-v)*p1[2] + u*(1-v)*p2[2] +
                              (1-u)*v*p3[2] + u*v*p4[2])

                    iz = int(np.round(point_0))
                    iy = int(np.round(point_1))
                    ix = int(np.round(point_2))

                    if (0 <= iz < volume.shape[0] and
                        0 <= iy < volume.shape[1] and
                        0 <= ix < volume.shape[2]):
                        volume[iz, iy, ix] = True


@jit(nopython=True, fastmath=True)
def check_triangle_in_bounds(p1, p2, p3, shape):
    """Check if triangle intersects volume bounds."""
    min_z = min(p1[0], p2[0], p3[0])
    max_z = max(p1[0], p2[0], p3[0])
    min_y = min(p1[1], p2[1], p3[1])
    max_y = max(p1[1], p2[1], p3[1])
    min_x = min(p1[2], p2[2], p3[2])
    max_x = max(p1[2], p2[2], p3[2])

    if max_z < 0 or min_z >= shape[0]: return False
    if max_y < 0 or min_y >= shape[1]: return False
    if max_x < 0 or min_x >= shape[2]: return False
    return True


# ============================================================================
# VECTORIZED OVERLAP DETECTION
# ============================================================================

def detect_overlaps_vectorized(fitted_sheets, num_components):
    """Vectorized overlap detection using scipy operations."""
    shape = list(fitted_sheets.values())[0].shape
    count_map = np.zeros(shape, dtype=np.int32)
    for i in range(1, num_components + 1):
        count_map += fitted_sheets[i].astype(np.int32)

    potential_overlap = count_map > 1
    if not np.any(potential_overlap):
        return np.zeros(shape, dtype=bool)

    labeled_result = np.zeros(shape, dtype=np.int32)
    for i in range(1, num_components + 1):
        labeled_result[fitted_sheets[i]] = i

    from scipy.ndimage import generic_filter

    def has_different_neighbor(values):
        center = values[13]
        if center == 0:
            return 0
        for val in values:
            if val > 0 and val != center:
                return 1
        return 0

    overlap_mask = np.zeros(shape, dtype=bool)
    coords = np.column_stack(np.nonzero(potential_overlap))
    if len(coords) == 0:
        return overlap_mask

    min_coords = np.maximum(coords.min(axis=0) - 1, 0)
    max_coords = np.minimum(coords.max(axis=0) + 2, shape)
    slices = tuple(slice(min_coords[i], max_coords[i]) for i in range(3))
    roi_labeled = labeled_result[slices]
    roi_potential = potential_overlap[slices]

    roi_overlap = generic_filter(
        roi_labeled,
        has_different_neighbor,
        size=3,
        mode='constant',
        cval=0
    ).astype(bool)

    roi_overlap = roi_overlap & roi_potential
    overlap_mask[slices] = roi_overlap
    return overlap_mask


# ============================================================================
# ALGORITHMIC OPTIMIZATIONS
# ============================================================================

def adaptive_grid_resolution(component, base_resolution=100, max_resolution=150):
    """Dynamically adjust grid resolution based on component size."""
    num_voxels = np.sum(component)
    if num_voxels < 500:
        return min(30, base_resolution)
    elif num_voxels < 2000:
        return min(50, base_resolution)
    elif num_voxels < 5000:
        return min(70, base_resolution)
    elif num_voxels < 15000:
        return base_resolution
    else:
        return min(max_resolution, base_resolution + 20)


def should_skip_smoothing(component, coverage_threshold=0.8):
    """Determine if a component needs smoothing based on planarity."""
    coords = np.column_stack(np.nonzero(component))
    coords_mean = coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(coords - coords_mean, full_matrices=False)
    if S[2] / S[0] < 0.05:
        return True
    return False


def zero_volume_faces(volume, thickness=5):
    """Optimized face zeroing using slicing."""
    result = volume.copy()
    result[:thickness, :, :] = False
    result[-thickness:, :, :] = False
    result[:, :thickness, :] = False
    result[:, -thickness:, :] = False
    result[:, :, :thickness] = False
    result[:, :, -thickness:] = False
    return result


# ============================================================================
# OPTIMIZED MAIN FITTING FUNCTION
# ============================================================================

def fit_curved_sheet_to_component_optimized(
    component,
    grid_resolution=100,
    thickness=3,
    smoothing=1.0,
    use_median_filter=True,
    max_distance=10,
    use_numba=True,
    adaptive_resolution=True,
    samples_per_edge=8
):
    """Fit a curved sheet to a binary component via SVD + griddata interpolation."""
    coords = np.column_stack(np.nonzero(component))
    if len(coords) < 10:
        return component.copy()

    if adaptive_resolution:
        grid_resolution = adaptive_grid_resolution(component, grid_resolution)

    coords_mean = coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(coords - coords_mean, full_matrices=False)
    tangent1, tangent2 = Vt[0], Vt[1]
    normal_guess = Vt[2]

    uv_coords = (coords - coords_mean) @ np.column_stack([tangent1, tangent2])
    w_coords = (coords - coords_mean) @ normal_guess

    if len(coords) > 5000:
        indices = np.random.choice(len(coords), 5000, replace=False)
        uv_coords_sample = uv_coords[indices]
        w_coords_sample = w_coords[indices]
    else:
        uv_coords_sample = uv_coords
        w_coords_sample = w_coords

    u_min, u_max = uv_coords[:,0].min(), uv_coords[:,0].max()
    v_min, v_max = uv_coords[:,1].min(), uv_coords[:,1].max()
    u_padding = (u_max - u_min) * 0.05
    v_padding = (v_max - v_min) * 0.05

    grid_u, grid_v = np.meshgrid(
        np.linspace(u_min - u_padding, u_max + u_padding, num=grid_resolution),
        np.linspace(v_min - v_padding, v_max + v_padding, num=grid_resolution),
        indexing='ij'
    )

    try:
        w_grid = griddata(uv_coords_sample, w_coords_sample, (grid_u, grid_v), method='linear')
    except Exception:
        w_grid = griddata(uv_coords_sample, w_coords_sample, (grid_u, grid_v), method='nearest')

    if np.any(np.isnan(w_grid)):
        mask = np.isnan(w_grid)
        w_grid_nearest = griddata(uv_coords_sample, w_coords_sample, (grid_u, grid_v), method='nearest')
        w_grid[mask] = w_grid_nearest[mask]

    if use_median_filter:
        w_grid = median_filter(w_grid, size=3)

    skip_smooth = should_skip_smoothing(component)

    if smoothing > 0 and not skip_smooth:
        w_grid = gaussian_filter(w_grid, sigma=smoothing)

    # Grid trimming with KDTree
    tree = cKDTree(uv_coords)
    threshold = (u_max - u_min + v_max - v_min) / (2 * grid_resolution) * 2

    grid_uv_flat = np.column_stack([grid_u.ravel(), grid_v.ravel()])
    distances, _ = tree.query(grid_uv_flat, k=1)
    distances = distances.reshape(grid_resolution, grid_resolution)

    original_data_mask = distances <= threshold

    # Flood-fill from edges
    grid_mask = np.ones_like(w_grid, dtype=bool)
    visited = np.zeros_like(w_grid, dtype=bool)
    queue = deque()

    for i in range(grid_resolution):
        queue.append((i, 0))
        queue.append((i, grid_resolution - 1))
        visited[i, 0] = True
        visited[i, grid_resolution - 1] = True

    for j in range(1, grid_resolution - 1):
        queue.append((0, j))
        queue.append((grid_resolution - 1, j))
        visited[0, j] = True
        visited[grid_resolution - 1, j] = True

    while queue:
        i, j = queue.popleft()

        if not original_data_mask[i, j]:
            grid_mask[i, j] = False

            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if (0 <= ni < grid_resolution and
                    0 <= nj < grid_resolution and
                    not visited[ni, nj]):
                    visited[ni, nj] = True
                    queue.append((ni, nj))

    w_grid[~grid_mask] = np.nan

    grid_points = (coords_mean +
                   grid_u[...,None] * tangent1 +
                   grid_v[...,None] * tangent2 +
                   w_grid[...,None] * normal_guess)

    Z, Y, X = component.shape
    sheet_volume = np.zeros_like(component, dtype=bool)

    if use_numba:
        rasterize_surface_numba(grid_points, sheet_volume, samples_per_edge=samples_per_edge)
    else:
        _rasterize_surface_dense_sampling_original(grid_points, sheet_volume, samples_per_quad=samples_per_edge)

    sheet_volume = zero_volume_faces(sheet_volume, thickness=5)

    if thickness > 0:
        iterations = max(1, thickness // 2)
        struct_elem = np.array([
            [[0,1,0], [1,1,1], [0,1,0]],
            [[1,1,1], [1,1,1], [1,1,1]],
            [[0,1,0], [1,1,1], [0,1,0]]
        ], dtype=bool)
        sheet_volume = binary_dilation(sheet_volume, structure=struct_elem, iterations=iterations)

    for z in range(Z):
        if np.any(sheet_volume[z]):
            sheet_volume[z] = binary_fill_holes(sheet_volume[z])

    return sheet_volume


def _rasterize_surface_dense_sampling_original(grid_points, volume, samples_per_quad=5):
    """Original Python implementation for fallback."""
    grid_resolution = grid_points.shape[0]
    for i in range(grid_resolution - 1):
        for j in range(grid_resolution - 1):
            p1 = grid_points[i, j]
            p2 = grid_points[i+1, j]
            p3 = grid_points[i, j+1]
            p4 = grid_points[i+1, j+1]
            if (np.isnan(p1).any() or np.isnan(p2).any() or
                np.isnan(p3).any() or np.isnan(p4).any()):
                continue
            for u in np.linspace(0, 1, samples_per_quad):
                for v in np.linspace(0, 1, samples_per_quad):
                    point = ((1-u)*(1-v)*p1 + u*(1-v)*p2 + (1-u)*v*p3 + u*v*p4)
                    point = point.round().astype(int)
                    if (0 <= point[0] < volume.shape[0] and
                        0 <= point[1] < volume.shape[1] and
                        0 <= point[2] < volume.shape[2]):
                        volume[point[0], point[1], point[2]] = True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_dice_score(mask1, mask2):
    """Calculate Dice coefficient between two binary masks."""
    intersection = np.sum(mask1 & mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)
    if sum_masks == 0:
        return 0.0
    return 2.0 * intersection / sum_masks


def calculate_coverage_score(original, fitted):
    """Calculate how well the fitted sheet covers the original positive pixels."""
    original_pixels = np.sum(original)
    if original_pixels == 0:
        return 0.0
    return np.sum(original & fitted) / original_pixels


# ============================================================================
# PARALLEL COMPONENT PROCESSING
# ============================================================================

def process_component_wrapper(args):
    """Wrapper for parallel processing of components."""
    component_id, component_mask, grid_resolution, thickness, smoothing, max_distance, samples_per_edge = args
    try:
        fitted = fit_curved_sheet_to_component_optimized(
            component_mask,
            grid_resolution=grid_resolution,
            thickness=thickness,
            smoothing=smoothing,
            max_distance=max_distance,
            use_numba=True,
            adaptive_resolution=True,
            samples_per_edge=samples_per_edge
        )
        return component_id, fitted
    except Exception as e:
        print(f"Error processing component {component_id}: {e}")
        return component_id, component_mask


def _evaluate_component_worker(args):
    """
    Worker for parallel per-component evaluation (erode + quality check + alternatives).
    Returns a dict describing what to write into result_labeled.
    """
    (i, is_correct, component_mask, fitted_after_overlap,
     grid_resolution, thickness, smoothing, max_distance, samples_per_edge,
     alt_min_dice, alt_min_coverage, min_dice, min_coverage,
     alternative_volumes, erosion_iterations, struct_elem) = args

    original_component = component_mask

    if is_correct:
        return {
            'id': i, 'status': 'correct',
            'main_mask': fitted_after_overlap,
            'extra_components': [], 'dice': 1.0, 'coverage': 1.0,
        }

    if not np.any(fitted_after_overlap):
        dice, coverage = 0.0, 0.0
        eroded = None
    else:
        if erosion_iterations > 0:
            eroded = binary_erosion(fitted_after_overlap, structure=struct_elem,
                                    iterations=erosion_iterations)
        else:
            eroded = fitted_after_overlap
        eroded = binary_fill_holes(eroded)

        dice = calculate_dice_score(original_component, eroded)
        coverage = calculate_coverage_score(original_component, eroded)

    if eroded is not None and dice >= min_dice and coverage >= min_coverage:
        return {
            'id': i, 'status': 'fitted',
            'main_mask': eroded,
            'extra_components': [], 'dice': dice, 'coverage': coverage,
        }

    if alternative_volumes is not None and len(alternative_volumes) > 0:
        all_good_results = []
        remaining_region = original_component.copy()

        for alt_idx, alt_volume in enumerate(alternative_volumes):
            if not np.any(remaining_region):
                break

            alt_mask = alt_volume & remaining_region
            if not np.any(alt_mask):
                continue

            alt_labeled = label(alt_mask)
            num_alt_comps = alt_labeled.max()

            solved_in_this_alt = np.zeros_like(alt_volume, dtype=bool)
            unsolved_in_this_alt = np.zeros_like(alt_volume, dtype=bool)

            for comp_idx in range(1, num_alt_comps + 1):
                alt_comp = (alt_labeled == comp_idx)

                if np.sum(alt_comp) < 100:
                    continue

                try:
                    alt_fitted = fit_curved_sheet_to_component_optimized(
                        alt_comp,
                        grid_resolution=grid_resolution,
                        thickness=thickness,
                        smoothing=smoothing,
                        max_distance=max_distance,
                        use_numba=True,
                        adaptive_resolution=True,
                        samples_per_edge=samples_per_edge
                    )

                    alt_dice = calculate_dice_score(alt_comp, alt_fitted)
                    alt_coverage = calculate_coverage_score(alt_comp, alt_fitted)

                    if alt_dice >= alt_min_dice and alt_coverage >= alt_min_coverage:
                        all_good_results.append({
                            'fitted': alt_fitted,
                            'dice': alt_dice, 'coverage': alt_coverage,
                            'alt_idx': alt_idx, 'comp_idx': comp_idx,
                            'source_comp': alt_comp,
                        })
                        solved_in_this_alt |= alt_comp
                    else:
                        unsolved_in_this_alt |= alt_comp

                except Exception:
                    pass

            if np.any(solved_in_this_alt):
                remaining_region = unsolved_in_this_alt

        if len(all_good_results) > 0:
            if len(all_good_results) > 1:
                alt_fitted_sheets = {idx+1: r['fitted'] for idx, r in enumerate(all_good_results)}
                alt_overlap = detect_overlaps_vectorized(alt_fitted_sheets, len(all_good_results))
                alt_labeled_result = np.zeros_like(alt_volume, dtype=np.int32)
                for idx in range(1, len(all_good_results) + 1):
                    mask = alt_fitted_sheets[idx] & ~alt_overlap
                    alt_labeled_result[mask] = idx

                combined_alternatives = np.zeros_like(alt_volume, dtype=bool)
                for idx in range(1, len(all_good_results) + 1):
                    alt_comp_mask = (alt_labeled_result == idx)
                    if not np.any(alt_comp_mask):
                        continue
                    if erosion_iterations > 0:
                        ea = binary_erosion(alt_comp_mask, structure=struct_elem,
                                            iterations=erosion_iterations)
                    else:
                        ea = alt_comp_mask
                    ea = binary_fill_holes(ea)
                    combined_alternatives |= ea
            else:
                combined_alternatives = all_good_results[0]['fitted']
                if erosion_iterations > 0:
                    combined_alternatives = binary_erosion(
                        combined_alternatives, structure=struct_elem, iterations=erosion_iterations)
                    combined_alternatives = binary_fill_holes(combined_alternatives)

            return {
                'id': i, 'status': 'alternative',
                'main_mask': None,
                'extra_components': [combined_alternatives],
                'dice': dice, 'coverage': coverage,
            }

    return {
        'id': i, 'status': 'removed',
        'main_mask': None,
        'extra_components': [], 'dice': dice, 'coverage': coverage,
    }


# ============================================================================
# ITERATIVE BETA1 RE-INTERPOLATION
# ============================================================================

def _reinterpolate_bad_components(
    result_labeled,
    grid_resolution, thickness, smoothing, max_distance, samples_per_edge,
    overlap_buffer, min_dice, min_coverage, alt_min_dice, alt_min_coverage,
    alternative_volumes,
    max_iterations=3,
):
    """
    Check beta1 (= 1 - Euler number) for every component in result_labeled.
    Components with beta1 > 0 are re-processed through the full pipeline.
    Modifies result_labeled in-place and returns it.
    """
    erosion_iterations = overlap_buffer // 2
    struct_elem = ball(1) if erosion_iterations > 0 else None

    for iteration in range(max_iterations):
        current_binary = result_labeled > 0
        check_labeled = label(current_binary)
        num_check = check_labeled.max()

        bad_ids = []
        for cid in range(1, num_check + 1):
            comp = (check_labeled == cid)
            chi = euler_number(comp.astype(int), connectivity=1)
            beta1 = 1 - chi
            if beta1 > 0:
                bad_ids.append(cid)

        if not bad_ids:
            break

        # Step A: Fit sheets
        fit_args = [
            (cid,
             (check_labeled == cid),
             grid_resolution,
             thickness + overlap_buffer,
             smoothing,
             max_distance,
             samples_per_edge)
            for cid in bad_ids
        ]
        fit_results = [process_component_wrapper(a) for a in fit_args]

        fitted_sheets = {cid: fitted for cid, fitted in fit_results}

        # Step B: Overlap detection among re-fitted sheets
        if len(fitted_sheets) > 1:
            id_to_idx = {cid: idx + 1 for idx, cid in enumerate(bad_ids)}
            idx_sheets = {id_to_idx[cid]: fitted_sheets[cid] for cid in bad_ids}
            overlap_mask = detect_overlaps_vectorized(idx_sheets, len(bad_ids))
        else:
            shape = list(fitted_sheets.values())[0].shape
            overlap_mask = np.zeros(shape, dtype=bool)

        fitted_after_overlap = {
            cid: fitted_sheets[cid] & ~overlap_mask for cid in bad_ids
        }

        # Step C: Evaluate each bad component
        eval_args = [
            (cid,
             False,
             (check_labeled == cid),
             fitted_after_overlap[cid],
             grid_resolution, thickness + overlap_buffer, smoothing, max_distance, samples_per_edge,
             alt_min_dice, alt_min_coverage, min_dice, min_coverage,
             alternative_volumes, erosion_iterations, struct_elem)
            for cid in bad_ids
        ]
        eval_results = [_evaluate_component_worker(a) for a in eval_args]

        # Step D: Update result_labeled
        next_label = result_labeled.max() + 1

        for res in eval_results:
            cid = res['id']
            old_mask = (check_labeled == cid)

            orig_labels = result_labeled[old_mask]
            dominant_label = (
                int(np.bincount(orig_labels[orig_labels > 0]).argmax())
                if np.any(orig_labels > 0) else 0
            )
            if dominant_label == 0:
                dominant_label = next_label
                next_label += 1

            result_labeled[old_mask] = 0

            if res['status'] in ('fitted', 'correct') and res['main_mask'] is not None:
                result_labeled[res['main_mask'] & (result_labeled == 0)] = dominant_label

            elif res['status'] == 'alternative':
                for extra_mask in res['extra_components']:
                    result_labeled[extra_mask & (result_labeled == 0)] = next_label
                    next_label += 1

    return result_labeled


# ============================================================================
# MAIN PARALLEL PROCESSING FUNCTION
# ============================================================================

def process_multiple_components_parallel(
    volume,
    alternative_volumes=None,
    grid_resolution=80,
    thickness=3,
    smoothing=2.0,
    overlap_buffer=2,
    min_coverage=0.60,
    min_dice=0.6,
    alt_min_coverage=None,
    alt_min_dice=None,
    max_distance=10,
    use_parallel=False,
    n_jobs=1,
    samples_per_edge=8,
    max_reinterp_iterations=1,
):
    """
    Component processing with Euler-based pre-filtering, Numba JIT rasterization,
    fitting/evaluation, and iterative beta1 re-interpolation.
    """
    labeled_volume = label(volume)
    num_components = labeled_volume.max()

    if alt_min_coverage is None:
        alt_min_coverage = min_coverage
    if alt_min_dice is None:
        alt_min_dice = min_dice

    component_masks = {i: (labeled_volume == i) for i in range(1, num_components + 1)}

    # STEP 1: Euler-based topology analysis
    correct_components = []
    needs_interpolation = []

    for i in range(1, num_components + 1):
        chi = euler_number(component_masks[i].astype(int), connectivity=1)
        beta1 = 1 - chi
        if beta1 <= 0:
            correct_components.append(i)
        else:
            needs_interpolation.append(i)

    # STEP 2: Fit sheets to components that need it
    fitted_sheets = {}

    for i in correct_components:
        fitted_sheets[i] = component_masks[i]

    if len(needs_interpolation) > 0:
        fit_args = [
            (i, component_masks[i], grid_resolution,
             thickness + overlap_buffer, smoothing, max_distance, samples_per_edge)
            for i in needs_interpolation
        ]
        fit_results = [process_component_wrapper(a) for a in fit_args]

        for cid, fitted in fit_results:
            fitted_sheets[cid] = fitted

    # STEP 3: Overlap detection
    overlap_mask = detect_overlaps_vectorized(fitted_sheets, num_components)

    fitted_after_overlap = {}
    for i in range(1, num_components + 1):
        fitted_after_overlap[i] = fitted_sheets[i] & ~overlap_mask

    # STEP 4: Evaluate components
    erosion_iterations = overlap_buffer // 2
    struct_elem = ball(1) if erosion_iterations > 0 else None

    eval_args = [
        (i,
         i in correct_components,
         component_masks[i],
         fitted_after_overlap[i],
         grid_resolution, thickness + overlap_buffer, smoothing, max_distance, samples_per_edge,
         alt_min_dice, alt_min_coverage, min_dice, min_coverage,
         alternative_volumes, erosion_iterations, struct_elem)
        for i in range(1, num_components + 1)
    ]

    eval_results = [_evaluate_component_worker(a) for a in eval_args]

    # STEP 5: Assemble result_labeled
    result_labeled = np.zeros_like(volume, dtype=np.int32)

    next_label = num_components + 1

    for res in eval_results:
        i = res['id']

        if res['status'] == 'correct':
            if res['main_mask'] is not None:
                result_labeled[res['main_mask']] = i

        elif res['status'] == 'fitted':
            if res['main_mask'] is not None:
                result_labeled[res['main_mask'] & (result_labeled == 0)] = i

        elif res['status'] == 'alternative':
            for extra_mask in res['extra_components']:
                result_labeled[extra_mask & (result_labeled == 0)] = next_label
                next_label += 1

    # STEP 6: Iterative beta1 re-interpolation
    result_labeled = _reinterpolate_bad_components(
        result_labeled,
        grid_resolution=grid_resolution,
        thickness=thickness,
        smoothing=smoothing,
        max_distance=max_distance,
        samples_per_edge=samples_per_edge,
        overlap_buffer=overlap_buffer,
        min_dice=min_dice,
        min_coverage=min_coverage,
        alt_min_dice=alt_min_dice,
        alt_min_coverage=alt_min_coverage,
        alternative_volumes=alternative_volumes,
        max_iterations=max_reinterp_iterations,
    )

    result_binary = result_labeled > 0
    return result_binary, result_labeled


# ============================================================================
# TOPO THRESHOLD FUNCTIONS (from Cell 1)
# ============================================================================

def build_anisotropic_struct(z_radius: int, xy_radius: int):
    """Build a 3D ellipsoidal/cylindrical kernel for morphological closing."""
    z, r = z_radius, xy_radius

    if z == 0 and r == 0:
        return None

    if z == 0 and r > 0:
        size = 2 * r + 1
        struct = np.zeros((1, size, size), dtype=bool)
        cy, cx = r, r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[0, cy + dy, cx + dx] = True
        return struct

    if z > 0 and r == 0:
        struct = np.zeros((2 * z + 1, 1, 1), dtype=bool)
        struct[:, 0, 0] = True
        return struct

    depth = 2 * z + 1
    size = 2 * r + 1
    struct = np.zeros((depth, size, size), dtype=bool)
    cz, cy, cx = z, r, r
    for dz in range(-z, z + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[cz + dz, cy + dy, cx + dx] = True
    return struct


def topo_postprocess(probs, T_low=0.6, T_high=0.9, z_radius=1, xy_radius=1,
                     dust_min_size=500):
    """3D hysteresis thresholding + anisotropic closing + dust removal."""
    # Step 1: 3D Hysteresis
    strong = probs >= T_high
    weak = probs >= T_low

    if not strong.any():
        return np.zeros_like(probs, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros_like(probs, dtype=np.uint8)

    # Step 2: 3D Anisotropic Closing
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

    # Step 3: Dust Removal
    if dust_min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)

    return mask.astype(np.uint8)


# ============================================================================
# PER-CHUNK ENTRY POINT
# ============================================================================

def apply_topo_finalization(logits_np, num_classes, config: TopoPostprocessConfig):
    """Per-chunk topo finalization — replaces apply_finalization.

    Args:
        logits_np: Blended logits array of shape (C, Z, Y, X).
        num_classes: Number of output classes.
        config: TopoPostprocessConfig with all parameters.

    Returns:
        (result, is_empty) where result is uint8 array of shape (1, Z, Y, X)
        and is_empty is True if the result is all zeros.
    """
    # 1. Logits → probabilities
    if num_classes == 1:
        # Binary: sigmoid
        prob_map = 1.0 / (1.0 + np.exp(-logits_np[0]))  # (Z, Y, X)
    else:
        # Multiclass: softmax, take class 1 probability
        exp_logits = np.exp(logits_np - logits_np.max(axis=0, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)
        prob_map = probs[1]  # (Z, Y, X)

    border = config.topo_border_crop

    # Check if volume is too small for border cropping
    if any(s <= 2 * border for s in prob_map.shape):
        # Volume too small for border crop, skip topo processing
        result = np.zeros((1,) + prob_map.shape, dtype=np.uint8)
        return result, True

    # 2. Run topo_postprocess at multiple threshold levels
    pred = topo_postprocess(
        prob_map,
        T_low=config.topo_t_low,
        T_high=config.topo_t_high,
        z_radius=config.topo_z_radius,
        xy_radius=config.topo_xy_radius,
        dust_min_size=config.topo_dust_min_size,
    )

    alt_preds = []
    for alt_t_low in config.topo_alt_t_lows:
        alt_pred = topo_postprocess(
            prob_map,
            T_low=alt_t_low,
            T_high=config.topo_t_high,
            z_radius=config.topo_z_radius,
            xy_radius=config.topo_xy_radius,
            dust_min_size=config.topo_dust_min_size,
        )
        alt_preds.append(alt_pred)

    # 3. zero_volume_faces + remove_small_objects on each
    pred = zero_volume_faces(pred.astype(bool), thickness=config.topo_thickness)
    pred = remove_small_objects(pred.astype(bool), min_size=config.topo_min_object_size, connectivity=3)

    for idx in range(len(alt_preds)):
        alt_preds[idx] = zero_volume_faces(alt_preds[idx].astype(bool), thickness=config.topo_thickness)
        alt_preds[idx] = remove_small_objects(alt_preds[idx].astype(bool), min_size=config.topo_min_object_size, connectivity=3)

    # 4. Fill holes on primary pred
    pred = binary_fill_holes(pred.astype(bool))

    # 5. Crop border from all
    s = slice(border, -border)
    pred = pred[s, s, s]
    alt_preds = [ap[s, s, s] for ap in alt_preds]

    # Early exit if nothing survived
    if not np.any(pred) and not any(np.any(ap) for ap in alt_preds):
        result = np.zeros((1,) + prob_map.shape, dtype=np.uint8)
        return result, True

    # 6. Advanced component processing
    result_binary, _result_labeled = process_multiple_components_parallel(
        pred,
        grid_resolution=config.topo_grid_resolution,
        thickness=config.topo_thickness,
        smoothing=config.topo_smoothing,
        overlap_buffer=config.topo_overlap_buffer,
        min_coverage=config.topo_min_coverage,
        min_dice=config.topo_min_dice,
        max_distance=config.topo_max_distance,
        alternative_volumes=alt_preds,
        alt_min_coverage=0.75,
        alt_min_dice=0.45,
        samples_per_edge=config.topo_samples_per_edge,
    )

    # 7. Final cleanup — result_binary is (Z', Y', X'); take first if tuple
    if isinstance(result_binary, tuple):
        result_binary = result_binary[0]

    pred_final = remove_small_objects(
        result_binary.astype(bool),
        min_size=config.topo_final_min_object_size,
        connectivity=3
    )

    # 8. Pad back the border
    pred_final = np.pad(
        pred_final,
        pad_width=((border, border), (border, border), (border, border)),
        mode="constant",
        constant_values=0,
    )

    # Shape back to (1, Z, Y, X) for consistency with finalize output
    result = pred_final.astype(np.uint8)[np.newaxis, ...]
    is_empty = not np.any(result)
    return result, is_empty
