import numpy as np
from scipy import ndimage

from vesuvius.neural_tracing.datasets.direction_helpers import (
    build_split_surface_masks_and_trace_targets,
    _compute_surface_tangent_axis,
    _scatter_trace_line_numba,
    _scatter_trace_point_numba,
    _velocity_axis_and_sign,
)
from vesuvius.neural_tracing.datasets.common import voxelize_surface_grid


def _reference_surface_velocity_vectors(surface_grid: np.ndarray, cond_direction: str):
    grid = np.asarray(surface_grid, dtype=np.float32)
    finite = np.isfinite(grid).all(axis=2)
    axis, sign = _velocity_axis_and_sign(cond_direction)
    tangent, tangent_valid = _compute_surface_tangent_axis(grid, finite, axis=axis)
    vectors = tangent * np.float32(sign)
    norms = np.linalg.norm(vectors, axis=2)
    valid = finite & tangent_valid & np.isfinite(norms) & (norms > 1e-6)
    vectors_out = np.zeros_like(vectors, dtype=np.float32)
    vectors_out[valid] = vectors[valid] / norms[valid, None]
    return vectors_out, valid


def _reference_scatter_trace_surface(
    velocity_accum,
    weights,
    surface_grid,
    vectors,
    valid,
    surface_attract,
    surface_attract_weight,
    surface_attract_best_dist_sq,
    enable_surface_attract,
    surface_attract_radius,
):
    rows, cols = valid.shape
    for r in range(rows):
        for c in range(cols):
            if valid[r, c]:
                _scatter_trace_point_numba(
                    velocity_accum,
                    weights,
                    surface_attract,
                    surface_attract_weight,
                    surface_attract_best_dist_sq,
                    enable_surface_attract,
                    surface_attract_radius,
                    surface_grid[r, c],
                    vectors[r, c],
                )
    for r in range(rows):
        for c in range(cols - 1):
            if valid[r, c] and valid[r, c + 1]:
                _scatter_trace_line_numba(
                    velocity_accum,
                    weights,
                    surface_attract,
                    surface_attract_weight,
                    surface_attract_best_dist_sq,
                    enable_surface_attract,
                    surface_attract_radius,
                    surface_grid[r, c],
                    surface_grid[r, c + 1],
                    vectors[r, c],
                    vectors[r, c + 1],
                )
    for r in range(rows - 1):
        for c in range(cols):
            if valid[r, c] and valid[r + 1, c]:
                _scatter_trace_line_numba(
                    velocity_accum,
                    weights,
                    surface_attract,
                    surface_attract_weight,
                    surface_attract_best_dist_sq,
                    enable_surface_attract,
                    surface_attract_radius,
                    surface_grid[r, c],
                    surface_grid[r + 1, c],
                    vectors[r, c],
                    vectors[r + 1, c],
                )


def _reference_build_trace_targets(
    crop_size,
    cond_direction,
    surface,
    *,
    dilation_radius=0.0,
    surface_attract_radius=0.0,
):
    crop_size = tuple(int(v) for v in crop_size)
    velocity_accum = np.zeros((3, *crop_size), dtype=np.float32)
    weights = np.zeros(crop_size, dtype=np.float32)
    attract_radius = max(float(surface_attract_radius), 0.0)
    surface_attract = None
    surface_attract_weight = None
    surface_attract_best_dist_sq = None
    if attract_radius > 0.0:
        surface_attract = np.zeros((3, *crop_size), dtype=np.float32)
        surface_attract_weight = np.zeros(crop_size, dtype=np.float32)
        surface_attract_best_dist_sq = np.full(crop_size, np.inf, dtype=np.float32)

    vectors, valid = _reference_surface_velocity_vectors(surface, cond_direction)
    if valid.any():
        if surface_attract is None:
            surface_attract_arg = np.zeros((3, 1, 1, 1), dtype=np.float32)
            surface_attract_weight_arg = np.zeros((1, 1, 1), dtype=np.float32)
            surface_attract_best_dist_sq_arg = np.zeros((1, 1, 1), dtype=np.float32)
        else:
            surface_attract_arg = surface_attract
            surface_attract_weight_arg = surface_attract_weight
            surface_attract_best_dist_sq_arg = surface_attract_best_dist_sq
        _reference_scatter_trace_surface(
            velocity_accum,
            weights,
            surface,
            vectors,
            valid,
            surface_attract_arg,
            surface_attract_weight_arg,
            surface_attract_best_dist_sq_arg,
            surface_attract is not None,
            attract_radius,
        )

    valid_vox = weights > 0.0
    if not valid_vox.any():
        return None

    active_coords = np.nonzero(valid_vox)
    active_velocity = velocity_accum[:, valid_vox] / weights[valid_vox][None]
    norms = np.linalg.norm(active_velocity, axis=0)
    active_finite = (
        np.isfinite(active_velocity).all(axis=0)
        & np.isfinite(norms)
        & (norms > 1e-6)
    )

    velocity = np.zeros_like(velocity_accum, dtype=np.float32)
    valid_vox = np.zeros(crop_size, dtype=bool)
    if active_finite.any():
        finite_coords = tuple(coord[active_finite] for coord in active_coords)
        velocity[(slice(None), *finite_coords)] = (
            active_velocity[:, active_finite] / norms[active_finite][None]
        )
        valid_vox[finite_coords] = True

    radius = float(dilation_radius)
    if radius > 0.0:
        nearest_dist, nearest_idx = ndimage.distance_transform_edt(
            ~valid_vox,
            return_distances=True,
            return_indices=True,
        )
        band = np.isfinite(nearest_dist) & (nearest_dist <= radius)
        if band.any():
            velocity[:, band] = velocity[
                :,
                nearest_idx[0][band],
                nearest_idx[1][band],
                nearest_idx[2][band],
            ]
            valid_vox = band

    result = {
        "velocity_dir": velocity.astype(np.float32, copy=False),
        "trace_loss_weight": valid_vox[None].astype(np.float32, copy=False),
    }
    if surface_attract is not None and surface_attract_weight is not None:
        result["surface_attract"] = surface_attract.astype(np.float32, copy=False)
        result["surface_attract_weight"] = surface_attract_weight[None].astype(np.float32, copy=False)
    return result


def test_scatter_trace_point_matches_zero_length_line_with_surface_attract():
    crop_size = (9, 10, 11)
    p = np.array([4.25, 5.5, 6.75], dtype=np.float32)
    v = np.array([0.25, -0.5, 1.25], dtype=np.float32)
    radius = 2.0

    line_velocity = np.zeros((3, *crop_size), dtype=np.float32)
    line_weights = np.zeros(crop_size, dtype=np.float32)
    line_attract = np.zeros((3, *crop_size), dtype=np.float32)
    line_attract_weight = np.zeros(crop_size, dtype=np.float32)
    line_best_dist = np.full(crop_size, np.inf, dtype=np.float32)

    point_velocity = np.zeros_like(line_velocity)
    point_weights = np.zeros_like(line_weights)
    point_attract = np.zeros_like(line_attract)
    point_attract_weight = np.zeros_like(line_attract_weight)
    point_best_dist = np.full(crop_size, np.inf, dtype=np.float32)

    _scatter_trace_line_numba(
        line_velocity,
        line_weights,
        line_attract,
        line_attract_weight,
        line_best_dist,
        True,
        radius,
        p,
        p,
        v,
        v,
    )
    _scatter_trace_point_numba(
        point_velocity,
        point_weights,
        point_attract,
        point_attract_weight,
        point_best_dist,
        True,
        radius,
        p,
        v,
    )

    np.testing.assert_array_equal(point_velocity, line_velocity)
    np.testing.assert_array_equal(point_weights, line_weights)
    np.testing.assert_array_equal(point_attract, line_attract)
    np.testing.assert_array_equal(point_attract_weight, line_attract_weight)
    np.testing.assert_array_equal(point_best_dist, line_best_dist)


def test_split_surface_masks_and_trace_targets_match_reference():
    rng = np.random.default_rng(2468)
    crop_size = (23, 24, 25)
    rows_a, rows_b, cols_a, cols_b = 4, 5, 5, 6
    rows = rows_a + rows_b
    cols = cols_a + cols_b
    rr, cc = np.meshgrid(
        np.arange(rows, dtype=np.float32),
        np.arange(cols, dtype=np.float32),
        indexing="ij",
    )
    surface = np.stack(
        [
            6.0 + 0.25 * rr + 0.08 * cc,
            7.0 + 0.35 * rr,
            8.0 + 0.45 * cc,
        ],
        axis=-1,
    ).astype(np.float32)
    surface += rng.normal(scale=0.025, size=surface.shape).astype(np.float32)

    col_cond_surface = surface[:, :cols_a]
    col_masked_surface = surface[:, cols_a:]
    row_cond_surface = surface[:rows_a]
    row_masked_surface = surface[rows_a:]

    cases = (
        ("left", col_cond_surface, col_masked_surface, np.concatenate([col_cond_surface, col_masked_surface], axis=1)),
        ("right", col_cond_surface, col_masked_surface, np.concatenate([col_masked_surface, col_cond_surface], axis=1)),
        ("up", row_cond_surface, row_masked_surface, np.concatenate([row_cond_surface, row_masked_surface], axis=0)),
        ("down", row_cond_surface, row_masked_surface, np.concatenate([row_masked_surface, row_cond_surface], axis=0)),
    )

    for cond_direction, cond_surface, masked_surface, reference_surface in cases:
        reference = _reference_build_trace_targets(
            crop_size,
            cond_direction,
            reference_surface,
            dilation_radius=0.0,
            surface_attract_radius=0.0,
        )
        fused = build_split_surface_masks_and_trace_targets(
            crop_size,
            cond_direction,
            cond_surface_local=cond_surface,
            masked_surface_local=masked_surface,
        )

        assert reference is not None
        assert fused is not None
        np.testing.assert_array_equal(fused["cond_gt"], voxelize_surface_grid(cond_surface, crop_size))
        np.testing.assert_array_equal(fused["masked_seg"], voxelize_surface_grid(masked_surface, crop_size))
        np.testing.assert_array_equal(fused["velocity_dir"], reference["velocity_dir"])
        np.testing.assert_array_equal(fused["trace_loss_weight"], reference["trace_loss_weight"])
