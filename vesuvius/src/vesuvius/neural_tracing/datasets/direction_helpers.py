import random

import numpy as np
from numba import njit
from scipy import ndimage


def _estimate_mean_unit_direction_from_field(disp_np: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Estimate one robust mean unit direction from a 3-channel dense field."""
    disp = np.asarray(disp_np, dtype=np.float32)
    if disp.ndim != 4 or disp.shape[0] != 3:
        raise ValueError(f"disp_np must have shape (3, D, H, W), got {tuple(disp.shape)}")
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != tuple(disp.shape[1:]):
        raise ValueError(
            "mask shape must match displacement spatial dims "
            f"{tuple(disp.shape[1:])}, got {tuple(mask_bool.shape)}"
        )
    if not bool(mask_bool.any()):
        return None

    vecs = disp[:, mask_bool].T  # [N, 3]
    if vecs.size == 0:
        return None
    finite = np.isfinite(vecs).all(axis=1)
    vecs = vecs[finite]
    if vecs.shape[0] == 0:
        return None

    mags = np.linalg.norm(vecs, axis=1)
    vecs = vecs[mags > 1e-6]
    if vecs.shape[0] == 0:
        return None

    unit_vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-6)
    mean_vec = np.mean(unit_vecs, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(mean_vec))
    if not np.isfinite(norm) or norm <= 1e-6:
        return None
    return (mean_vec / norm).astype(np.float32, copy=False)


def _compute_surface_tangent_axis(surface_grid: np.ndarray, surface_valid: np.ndarray, axis: int):
    """Estimate local tangent vectors along one grid axis."""
    grid = np.asarray(surface_grid, dtype=np.float32)
    valid = np.asarray(surface_valid, dtype=bool)
    if grid.ndim != 3 or grid.shape[2] != 3:
        raise ValueError(f"surface_grid must have shape (H, W, 3), got {tuple(grid.shape)}")
    if valid.shape != grid.shape[:2]:
        raise ValueError(
            f"surface_valid shape {tuple(valid.shape)} must match grid shape {tuple(grid.shape[:2])}"
        )
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis!r}")

    tangent = np.zeros_like(grid, dtype=np.float32)
    tangent_valid = np.zeros(valid.shape, dtype=bool)
    h, w = valid.shape

    if axis == 0:
        if h >= 3:
            central_ok = valid[1:-1, :] & valid[:-2, :] & valid[2:, :]
            central_delta = 0.5 * (grid[2:, :, :] - grid[:-2, :, :])
            tangent[1:-1, :, :][central_ok] = central_delta[central_ok]
            tangent_valid[1:-1, :][central_ok] = True

        if h >= 2:
            diff = grid[1:, :, :] - grid[:-1, :, :]
            diff_ok = valid[1:, :] & valid[:-1, :]

            use_forward = (~tangent_valid[:-1, :]) & diff_ok
            tangent[:-1, :, :][use_forward] = diff[use_forward]
            tangent_valid[:-1, :][use_forward] = True

            use_backward = (~tangent_valid[1:, :]) & diff_ok
            tangent[1:, :, :][use_backward] = diff[use_backward]
            tangent_valid[1:, :][use_backward] = True
    else:
        if w >= 3:
            central_ok = valid[:, 1:-1] & valid[:, :-2] & valid[:, 2:]
            central_delta = 0.5 * (grid[:, 2:, :] - grid[:, :-2, :])
            tangent[:, 1:-1, :][central_ok] = central_delta[central_ok]
            tangent_valid[:, 1:-1][central_ok] = True

        if w >= 2:
            diff = grid[:, 1:, :] - grid[:, :-1, :]
            diff_ok = valid[:, 1:] & valid[:, :-1]

            use_forward = (~tangent_valid[:, :-1]) & diff_ok
            tangent[:, :-1, :][use_forward] = diff[use_forward]
            tangent_valid[:, :-1][use_forward] = True

            use_backward = (~tangent_valid[:, 1:]) & diff_ok
            tangent[:, 1:, :][use_backward] = diff[use_backward]
            tangent_valid[:, 1:][use_backward] = True

    return tangent, tangent_valid


def _scatter_vector_line(accum: np.ndarray, weights: np.ndarray, p0: np.ndarray, p1: np.ndarray, v0: np.ndarray, v1: np.ndarray) -> None:
    delta = p1 - p0
    steps = int(np.ceil(float(np.max(np.abs(delta)))))
    steps = max(steps, 1)
    d, h, w = weights.shape
    for i in range(steps + 1):
        t = i / steps
        p = p0 * (1.0 - t) + p1 * t
        z, y, x = np.rint(p).astype(np.int64)
        if z < 0 or z >= d or y < 0 or y >= h or x < 0 or x >= w:
            continue
        v = v0 * (1.0 - t) + v1 * t
        norm = float(np.linalg.norm(v))
        if not np.isfinite(norm) or norm <= 1e-6:
            continue
        accum[:, z, y, x] += (v / norm).astype(np.float32, copy=False)
        weights[z, y, x] += 1.0


def _scatter_velocity_surface(accum: np.ndarray, weights: np.ndarray, surface_grid: np.ndarray, vectors: np.ndarray, valid: np.ndarray) -> None:
    rows, cols = valid.shape
    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            _scatter_vector_line(
                accum,
                weights,
                surface_grid[r, c],
                surface_grid[r, c],
                vectors[r, c],
                vectors[r, c],
            )

    for r in range(rows):
        for c in range(cols - 1):
            if not (valid[r, c] and valid[r, c + 1]):
                continue
            _scatter_vector_line(
                accum,
                weights,
                surface_grid[r, c],
                surface_grid[r, c + 1],
                vectors[r, c],
                vectors[r, c + 1],
            )

    for r in range(rows - 1):
        for c in range(cols):
            if not (valid[r, c] and valid[r + 1, c]):
                continue
            _scatter_vector_line(
                accum,
                weights,
                surface_grid[r, c],
                surface_grid[r + 1, c],
                vectors[r, c],
                vectors[r + 1, c],
            )


def _velocity_axis_and_sign(cond_direction: str) -> tuple[int, float]:
    direction = str(cond_direction).lower()
    if direction == "up":
        return 0, 1.0
    if direction == "down":
        return 0, -1.0
    if direction == "left":
        return 1, 1.0
    if direction == "right":
        return 1, -1.0
    raise ValueError(
        "cond_direction must be one of {'up', 'down', 'left', 'right'}, "
        f"got {cond_direction!r}"
    )


def _surface_velocity_vectors(surface_grid: np.ndarray, cond_direction: str):
    grid = np.asarray(surface_grid, dtype=np.float32)
    if grid.ndim != 3 or grid.shape[2] != 3:
        raise ValueError(f"surface_grid must have shape (H, W, 3), got {tuple(grid.shape)}")

    finite = np.isfinite(grid).all(axis=2)
    axis, sign = _velocity_axis_and_sign(cond_direction)
    tangent, tangent_valid = _compute_surface_tangent_axis(grid, finite, axis=axis)
    vectors = tangent * np.float32(sign)
    norms = np.linalg.norm(vectors, axis=2)
    valid = finite & tangent_valid & np.isfinite(norms) & (norms > 1e-6)
    vectors_out = np.zeros_like(vectors, dtype=np.float32)
    vectors_out[valid] = vectors[valid] / norms[valid, None]
    return vectors_out, valid


@njit
def _remaining_trace_distance_axis_numba(grid: np.ndarray, valid: np.ndarray, axis: int, sign: float) -> np.ndarray:
    dist = np.zeros(valid.shape, dtype=np.float32)
    h, w = valid.shape

    if axis == 0:
        outer_count = w
        inner_count = h
    else:
        outer_count = h
        inner_count = w

    forward = sign > 0.0
    for outer in range(outer_count):
        inner = 0
        while inner < inner_count:
            is_valid = bool(valid[inner, outer]) if axis == 0 else bool(valid[outer, inner])
            if not is_valid:
                inner += 1
                continue

            start = inner
            while inner + 1 < inner_count:
                next_valid = bool(valid[inner + 1, outer]) if axis == 0 else bool(valid[outer, inner + 1])
                if not next_valid:
                    break
                inner += 1
            end = inner

            prev_z = 0.0
            prev_y = 0.0
            prev_x = 0.0
            has_prev = False
            accum = 0.0
            count = end - start + 1
            for offset in range(count):
                i = end - offset if forward else start + offset
                if axis == 0:
                    pz = float(grid[i, outer, 0])
                    py = float(grid[i, outer, 1])
                    px = float(grid[i, outer, 2])
                else:
                    pz = float(grid[outer, i, 0])
                    py = float(grid[outer, i, 1])
                    px = float(grid[outer, i, 2])
                if has_prev:
                    dz = prev_z - pz
                    dy = prev_y - py
                    dx = prev_x - px
                    step = np.sqrt(dz * dz + dy * dy + dx * dx)
                    if np.isfinite(step):
                        accum += step
                if axis == 0:
                    dist[i, outer] = np.float32(accum)
                else:
                    dist[outer, i] = np.float32(accum)
                prev_z = pz
                prev_y = py
                prev_x = px
                has_prev = True

            inner += 1

    return dist


def _remaining_trace_distance_axis(surface_grid: np.ndarray, surface_valid: np.ndarray, axis: int, sign: float) -> np.ndarray:
    """Estimate remaining arc length to the away-from-conditioning end of each row/col segment."""
    grid = np.asarray(surface_grid, dtype=np.float32)
    valid = np.asarray(surface_valid, dtype=bool)
    if grid.ndim != 3 or grid.shape[2] != 3:
        raise ValueError(f"surface_grid must have shape (H, W, 3), got {tuple(grid.shape)}")
    if valid.shape != grid.shape[:2]:
        raise ValueError(
            f"surface_valid shape {tuple(valid.shape)} must match grid shape {tuple(grid.shape[:2])}"
        )
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis!r}")
    return _remaining_trace_distance_axis_numba(grid, valid, axis, float(sign))


@njit
def _stamp_trace_surface_attract_numba(
    surface_attract: np.ndarray,
    surface_attract_weight: np.ndarray,
    surface_attract_best_dist_sq: np.ndarray,
    pz: float,
    py: float,
    px: float,
    radius: float,
) -> None:
    if radius <= 0.0:
        return
    if not (np.isfinite(pz) and np.isfinite(py) and np.isfinite(px)):
        return

    depth, height, width = surface_attract_weight.shape
    radius_sq = float(radius * radius)
    z0 = max(0, int(np.floor(pz - radius)))
    z1 = min(depth - 1, int(np.ceil(pz + radius)))
    y0 = max(0, int(np.floor(py - radius)))
    y1 = min(height - 1, int(np.ceil(py + radius)))
    x0 = max(0, int(np.floor(px - radius)))
    x1 = min(width - 1, int(np.ceil(px + radius)))

    for z in range(z0, z1 + 1):
        dz = float(pz - z)
        for y in range(y0, y1 + 1):
            dy = float(py - y)
            for x in range(x0, x1 + 1):
                dx = float(px - x)
                dist_sq = dz * dz + dy * dy + dx * dx
                if dist_sq > radius_sq or dist_sq >= float(surface_attract_best_dist_sq[z, y, x]):
                    continue
                surface_attract_best_dist_sq[z, y, x] = np.float32(dist_sq)
                surface_attract[0, z, y, x] = np.float32(dz)
                surface_attract[1, z, y, x] = np.float32(dy)
                surface_attract[2, z, y, x] = np.float32(dx)
                surface_attract_weight[z, y, x] = 1.0


@njit
def _scatter_trace_line_numba(
    velocity_accum: np.ndarray,
    trace_dist_accum: np.ndarray,
    trace_stop_accum: np.ndarray,
    weights: np.ndarray,
    surface_attract: np.ndarray,
    surface_attract_weight: np.ndarray,
    surface_attract_best_dist_sq: np.ndarray,
    enable_surface_attract: bool,
    surface_attract_radius: float,
    p0: np.ndarray,
    p1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    d0: float,
    d1: float,
    s0: float,
    s1: float,
) -> None:
    dz_line = float(p1[0] - p0[0])
    dy_line = float(p1[1] - p0[1])
    dx_line = float(p1[2] - p0[2])
    max_delta = max(abs(dz_line), abs(dy_line), abs(dx_line))
    steps = int(np.ceil(max_delta))
    steps = max(steps, 1)
    depth, height, width = weights.shape
    for i in range(steps + 1):
        t = i / steps
        pz = float(p0[0] * (1.0 - t) + p1[0] * t)
        py = float(p0[1] * (1.0 - t) + p1[1] * t)
        px = float(p0[2] * (1.0 - t) + p1[2] * t)
        z = int(np.rint(pz))
        y = int(np.rint(py))
        x = int(np.rint(px))
        if z < 0 or z >= depth or y < 0 or y >= height or x < 0 or x >= width:
            continue

        vz = float(v0[0] * (1.0 - t) + v1[0] * t)
        vy = float(v0[1] * (1.0 - t) + v1[1] * t)
        vx = float(v0[2] * (1.0 - t) + v1[2] * t)
        norm = np.sqrt(vz * vz + vy * vy + vx * vx)
        if not np.isfinite(norm) or norm <= 1e-6:
            continue

        dist = float(d0 * (1.0 - t) + d1 * t)
        stop = float(s0 * (1.0 - t) + s1 * t)
        if not np.isfinite(dist) or not np.isfinite(stop):
            continue

        velocity_accum[0, z, y, x] += np.float32(vz / norm)
        velocity_accum[1, z, y, x] += np.float32(vy / norm)
        velocity_accum[2, z, y, x] += np.float32(vx / norm)
        trace_dist_accum[z, y, x] += np.float32(max(dist, 0.0))
        trace_stop_accum[z, y, x] += np.float32(min(max(stop, 0.0), 1.0))
        weights[z, y, x] += 1.0

        if enable_surface_attract:
            _stamp_trace_surface_attract_numba(
                surface_attract,
                surface_attract_weight,
                surface_attract_best_dist_sq,
                pz,
                py,
                px,
                surface_attract_radius,
            )


@njit
def _scatter_trace_surface_numba(
    velocity_accum: np.ndarray,
    trace_dist_accum: np.ndarray,
    trace_stop_accum: np.ndarray,
    weights: np.ndarray,
    surface_grid: np.ndarray,
    vectors: np.ndarray,
    trace_dist: np.ndarray,
    trace_stop: np.ndarray,
    valid: np.ndarray,
    surface_attract: np.ndarray,
    surface_attract_weight: np.ndarray,
    surface_attract_best_dist_sq: np.ndarray,
    enable_surface_attract: bool,
    surface_attract_radius: float,
) -> None:
    rows, cols = valid.shape

    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            _scatter_trace_line_numba(
                velocity_accum,
                trace_dist_accum,
                trace_stop_accum,
                weights,
                surface_attract,
                surface_attract_weight,
                surface_attract_best_dist_sq,
                enable_surface_attract,
                surface_attract_radius,
                surface_grid[r, c],
                surface_grid[r, c],
                vectors[r, c],
                vectors[r, c],
                float(trace_dist[r, c]),
                float(trace_dist[r, c]),
                float(trace_stop[r, c]),
                float(trace_stop[r, c]),
            )

    for r in range(rows):
        for c in range(cols - 1):
            if not (valid[r, c] and valid[r, c + 1]):
                continue
            _scatter_trace_line_numba(
                velocity_accum,
                trace_dist_accum,
                trace_stop_accum,
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
                float(trace_dist[r, c]),
                float(trace_dist[r, c + 1]),
                float(trace_stop[r, c]),
                float(trace_stop[r, c + 1]),
            )

    for r in range(rows - 1):
        for c in range(cols):
            if not (valid[r, c] and valid[r + 1, c]):
                continue
            _scatter_trace_line_numba(
                velocity_accum,
                trace_dist_accum,
                trace_stop_accum,
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
                float(trace_dist[r, c]),
                float(trace_dist[r + 1, c]),
                float(trace_stop[r, c]),
                float(trace_stop[r + 1, c]),
            )


def _scatter_trace_surface(
    velocity_accum: np.ndarray,
    trace_dist_accum: np.ndarray,
    trace_stop_accum: np.ndarray,
    weights: np.ndarray,
    surface_grid: np.ndarray,
    vectors: np.ndarray,
    trace_dist: np.ndarray,
    trace_stop: np.ndarray,
    valid: np.ndarray,
    surface_attract: np.ndarray | None = None,
    surface_attract_weight: np.ndarray | None = None,
    surface_attract_best_dist_sq: np.ndarray | None = None,
    surface_attract_radius: float = 0.0,
) -> None:
    enable_surface_attract = (
        surface_attract is not None
        and surface_attract_weight is not None
        and surface_attract_best_dist_sq is not None
        and surface_attract_radius > 0.0
    )
    if not enable_surface_attract:
        surface_attract = np.zeros((3, 1, 1, 1), dtype=np.float32)
        surface_attract_weight = np.zeros((1, 1, 1), dtype=np.float32)
        surface_attract_best_dist_sq = np.zeros((1, 1, 1), dtype=np.float32)
    _scatter_trace_surface_numba(
        velocity_accum,
        trace_dist_accum,
        trace_stop_accum,
        weights,
        surface_grid,
        vectors,
        trace_dist,
        trace_stop,
        valid,
        surface_attract,
        surface_attract_weight,
        surface_attract_best_dist_sq,
        enable_surface_attract,
        surface_attract_radius,
    )


def build_away_from_conditioning_velocity_target(
    crop_size,
    cond_direction: str,
    *,
    cond_surface_local: np.ndarray | None = None,
    masked_surface_local: np.ndarray | None = None,
    include_conditioning: bool = True,
    include_masked: bool = True,
    dilation_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build a crop-local unit velocity target pointing away from conditioning.

    The target is derived from ordered surface-grid tangents, not from nearest-
    surface EDT displacement. Vectors are accumulated onto rasterized surface
    voxels and optionally copied into a narrow Euclidean band around them.
    """
    crop_size = tuple(int(v) for v in crop_size)
    if len(crop_size) != 3:
        raise ValueError(f"crop_size must have length 3, got {crop_size!r}")

    accum = np.zeros((3, *crop_size), dtype=np.float32)
    weights = np.zeros(crop_size, dtype=np.float32)

    surfaces = []
    if include_conditioning and cond_surface_local is not None:
        surfaces.append(np.asarray(cond_surface_local, dtype=np.float32))
    if include_masked and masked_surface_local is not None:
        surfaces.append(np.asarray(masked_surface_local, dtype=np.float32))

    for surface in surfaces:
        vectors, valid = _surface_velocity_vectors(surface, cond_direction)
        if bool(valid.any()):
            _scatter_velocity_surface(accum, weights, surface, vectors, valid)

    valid_vox = weights > 0.0
    if not bool(valid_vox.any()):
        return None

    velocity = np.zeros_like(accum, dtype=np.float32)
    velocity[:, valid_vox] = accum[:, valid_vox] / weights[valid_vox][None]
    norms = np.linalg.norm(velocity, axis=0)
    finite = np.isfinite(velocity).all(axis=0) & np.isfinite(norms) & (norms > 1e-6)
    velocity[:, finite] = velocity[:, finite] / norms[finite][None]
    velocity[:, ~finite] = 0.0
    valid_vox = finite

    radius = float(dilation_radius)
    if radius > 0.0:
        nearest_dist, nearest_idx = ndimage.distance_transform_edt(
            ~valid_vox,
            return_distances=True,
            return_indices=True,
        )
        band = np.isfinite(nearest_dist) & (nearest_dist <= radius)
        if bool(band.any()):
            velocity[:, band] = velocity[
                :,
                nearest_idx[0][band],
                nearest_idx[1][band],
                nearest_idx[2][band],
            ]
            valid_vox = band

    loss_weight = valid_vox[None].astype(np.float32, copy=False)
    return velocity.astype(np.float32, copy=False), loss_weight


def build_away_from_conditioning_trace_targets(
    crop_size,
    cond_direction: str,
    *,
    cond_surface_local: np.ndarray | None = None,
    masked_surface_local: np.ndarray | None = None,
    include_conditioning: bool = True,
    include_masked: bool = True,
    dilation_radius: float = 0.0,
    stop_radius: float = 1.0,
    surface_attract_radius: float = 0.0,
    include_trace_dist: bool = True,
    include_trace_stop: bool = True,
) -> dict[str, np.ndarray] | None:
    """Build ODE-style trace supervision from ordered surface-grid coordinates.

    The velocity is the unit away-from-conditioning tangent. ``trace_dist`` is
    remaining arc length along the selected row/col axis to the terminal end of
    each valid contiguous surface segment. ``trace_stop`` is 1 near that
    terminal end and 0 elsewhere on supervised voxels.
    """
    crop_size = tuple(int(v) for v in crop_size)
    if len(crop_size) != 3:
        raise ValueError(f"crop_size must have length 3, got {crop_size!r}")

    compute_trace_progress = bool(include_trace_dist) or bool(include_trace_stop)
    velocity_accum = np.zeros((3, *crop_size), dtype=np.float32)
    trace_dist_accum = np.zeros(crop_size, dtype=np.float32)
    trace_stop_accum = np.zeros(crop_size, dtype=np.float32)
    weights = np.zeros(crop_size, dtype=np.float32)
    attract_radius = max(float(surface_attract_radius), 0.0)
    surface_attract = None
    surface_attract_weight = None
    surface_attract_best_dist_sq = None
    if attract_radius > 0.0:
        surface_attract = np.zeros((3, *crop_size), dtype=np.float32)
        surface_attract_weight = np.zeros(crop_size, dtype=np.float32)
        surface_attract_best_dist_sq = np.full(crop_size, np.inf, dtype=np.float32)

    surfaces = []
    if include_conditioning and cond_surface_local is not None:
        surfaces.append(np.asarray(cond_surface_local, dtype=np.float32))
    if include_masked and masked_surface_local is not None:
        surfaces.append(np.asarray(masked_surface_local, dtype=np.float32))

    axis, sign = _velocity_axis_and_sign(cond_direction)
    stop_radius = max(float(stop_radius), 0.0)
    for surface in surfaces:
        grid = np.asarray(surface, dtype=np.float32)
        finite = np.isfinite(grid).all(axis=2)
        vectors, tangent_valid = _surface_velocity_vectors(grid, cond_direction)
        valid = finite & tangent_valid
        if not bool(valid.any()):
            continue
        if compute_trace_progress:
            trace_dist = _remaining_trace_distance_axis(grid, valid, axis=axis, sign=sign)
            trace_stop = (trace_dist <= stop_radius).astype(np.float32, copy=False)
        else:
            trace_dist = np.zeros(valid.shape, dtype=np.float32)
            trace_stop = np.zeros(valid.shape, dtype=np.float32)
        if not compute_trace_progress and attract_radius <= 0.0:
            _scatter_velocity_surface(velocity_accum, weights, grid, vectors, valid)
            continue
        _scatter_trace_surface(
            velocity_accum,
            trace_dist_accum,
            trace_stop_accum,
            weights,
            grid,
            vectors,
            trace_dist,
            trace_stop,
            valid,
            surface_attract=surface_attract,
            surface_attract_weight=surface_attract_weight,
            surface_attract_best_dist_sq=surface_attract_best_dist_sq,
            surface_attract_radius=attract_radius,
        )

    valid_vox = weights > 0.0
    if not bool(valid_vox.any()):
        return None

    velocity = np.zeros_like(velocity_accum, dtype=np.float32)
    trace_dist_out = np.zeros(crop_size, dtype=np.float32)
    trace_stop_out = np.zeros(crop_size, dtype=np.float32)
    velocity[:, valid_vox] = velocity_accum[:, valid_vox] / weights[valid_vox][None]
    trace_dist_out[valid_vox] = trace_dist_accum[valid_vox] / weights[valid_vox]
    trace_stop_out[valid_vox] = trace_stop_accum[valid_vox] / weights[valid_vox]

    norms = np.linalg.norm(velocity, axis=0)
    finite = np.isfinite(velocity).all(axis=0) & np.isfinite(norms) & (norms > 1e-6)
    finite &= np.isfinite(trace_dist_out) & np.isfinite(trace_stop_out)
    velocity[:, finite] = velocity[:, finite] / norms[finite][None]
    velocity[:, ~finite] = 0.0
    trace_dist_out[~finite] = 0.0
    trace_stop_out[~finite] = 0.0
    valid_vox = finite

    radius = float(dilation_radius)
    if radius > 0.0:
        nearest_dist, nearest_idx = ndimage.distance_transform_edt(
            ~valid_vox,
            return_distances=True,
            return_indices=True,
        )
        band = np.isfinite(nearest_dist) & (nearest_dist <= radius)
        if bool(band.any()):
            velocity[:, band] = velocity[
                :,
                nearest_idx[0][band],
                nearest_idx[1][band],
                nearest_idx[2][band],
            ]
            if include_trace_dist:
                trace_dist_out[band] = trace_dist_out[
                    nearest_idx[0][band],
                    nearest_idx[1][band],
                    nearest_idx[2][band],
                ]
            if include_trace_stop:
                trace_stop_out[band] = trace_stop_out[
                    nearest_idx[0][band],
                    nearest_idx[1][band],
                    nearest_idx[2][band],
                ]
            valid_vox = band

    result = {
        "velocity_dir": velocity.astype(np.float32, copy=False),
        "trace_loss_weight": valid_vox[None].astype(np.float32, copy=False),
    }
    if include_trace_dist:
        result["trace_dist"] = trace_dist_out[None].astype(np.float32, copy=False)
    if include_trace_stop:
        result["trace_stop"] = trace_stop_out[None].astype(np.float32, copy=False)
    if surface_attract is not None and surface_attract_weight is not None:
        result["surface_attract"] = surface_attract.astype(np.float32, copy=False)
        result["surface_attract_weight"] = surface_attract_weight[None].astype(np.float32, copy=False)
    return result


def estimate_global_unit_normal_from_surface_grid(surface_grid: np.ndarray) -> np.ndarray:
    """Estimate a global unit normal from one ordered surface grid.

    Returns a zero vector when no stable estimate can be formed.
    """
    grid = np.asarray(surface_grid, dtype=np.float32)
    if grid.ndim != 3 or grid.shape[2] != 3:
        return np.zeros((3,), dtype=np.float32)

    valid = np.isfinite(grid).all(axis=2)
    if not bool(valid.any()):
        return np.zeros((3,), dtype=np.float32)

    row_tangent, row_tangent_valid = _compute_surface_tangent_axis(grid, valid, axis=0)
    col_tangent, col_tangent_valid = _compute_surface_tangent_axis(grid, valid, axis=1)
    normals = np.cross(col_tangent, row_tangent)
    norms = np.linalg.norm(normals, axis=2)
    finite = np.isfinite(normals).all(axis=2) & np.isfinite(norms)
    normals_valid = valid & row_tangent_valid & col_tangent_valid & finite & (norms > 1e-6)
    if not bool(normals_valid.any()):
        return np.zeros((3,), dtype=np.float32)

    unit_normals = normals[normals_valid] / norms[normals_valid, None]
    mean_vec = np.mean(unit_normals, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    mean_norm = float(np.linalg.norm(mean_vec))
    if not np.isfinite(mean_norm) or mean_norm <= 1e-6:
        return np.zeros((3,), dtype=np.float32)
    return (mean_vec / mean_norm).astype(np.float32, copy=False)


def estimate_triplet_unit_direction(disp_np: np.ndarray, cond_mask: np.ndarray) -> np.ndarray:
    """Estimate one robust unit direction from dense displacement on conditioning voxels."""
    disp = np.asarray(disp_np, dtype=np.float32)
    mask = np.asarray(cond_mask, dtype=bool)
    if disp.ndim != 4 or disp.shape[0] != 3:
        raise ValueError(f"disp_np must have shape (3, D, H, W), got {tuple(disp.shape)}")
    if mask.shape != tuple(disp.shape[1:]):
        raise ValueError(
            f"cond_mask shape must match displacement spatial dims {tuple(disp.shape[1:])}, got {tuple(mask.shape)}"
        )
    if not bool(mask.any()):
        return np.zeros((3,), dtype=np.float32)

    vecs = disp[:, mask].T  # [N, 3]
    if vecs.size == 0:
        return np.zeros((3,), dtype=np.float32)
    finite = np.isfinite(vecs).all(axis=1)
    vecs = vecs[finite]
    if vecs.shape[0] == 0:
        return np.zeros((3,), dtype=np.float32)

    mags = np.linalg.norm(vecs, axis=1)
    vecs = vecs[mags > 1e-6]
    if vecs.shape[0] == 0:
        return np.zeros((3,), dtype=np.float32)

    unit_vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-6)
    mean_vec = np.mean(unit_vecs, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(mean_vec))
    if not np.isfinite(norm) or norm <= 1e-6:
        return np.zeros((3,), dtype=np.float32)
    return (mean_vec / norm).astype(np.float32, copy=False)


def build_triplet_direction_priors(
    crop_size,
    cond_mask: np.ndarray,
    ch0_dir: np.ndarray,
    ch1_dir: np.ndarray,
    mask_mode: str = "cond",
) -> np.ndarray:
    """Build broadcast direction priors for 2 triplet displacement branches."""
    crop_size = tuple(int(v) for v in crop_size)
    if len(crop_size) != 3:
        raise ValueError(f"crop_size must be length 3, got {crop_size}")
    priors = np.zeros((6, *crop_size), dtype=np.float32)
    v0 = np.asarray(ch0_dir, dtype=np.float32).reshape(3)
    v1 = np.asarray(ch1_dir, dtype=np.float32).reshape(3)
    for axis in range(3):
        priors[axis, ...] = v0[axis]
        priors[axis + 3, ...] = v1[axis]

    mode = str(mask_mode).lower()
    if mode == "cond":
        mask = np.asarray(cond_mask, dtype=np.float32)
        if mask.shape != crop_size:
            raise ValueError(f"cond_mask shape must match crop_size {crop_size}, got {tuple(mask.shape)}")
        priors *= mask[None]
    elif mode != "full":
        raise ValueError(f"mask_mode must be 'cond' or 'full', got {mask_mode!r}")
    return priors


def build_triplet_direction_priors_from_displacements(
    crop_size,
    cond_mask: np.ndarray,
    behind_disp_np: np.ndarray,
    front_disp_np: np.ndarray,
    mask_mode: str = "cond",
) -> np.ndarray:
    ch0_dir = estimate_triplet_unit_direction(behind_disp_np, cond_mask)
    ch1_dir = estimate_triplet_unit_direction(front_disp_np, cond_mask)
    return build_triplet_direction_priors(
        crop_size,
        cond_mask,
        ch0_dir,
        ch1_dir,
        mask_mode=mask_mode,
    )


def build_triplet_direction_priors_from_conditioning_surface(
    crop_size,
    cond_mask: np.ndarray,
    cond_surface_local: np.ndarray,
    mask_mode: str = "cond",
) -> np.ndarray | None:
    """Build triplet priors from conditioning-surface geometry only (no neighbor GT)."""
    global_unit_normal = estimate_global_unit_normal_from_surface_grid(cond_surface_local)
    if float(np.linalg.norm(global_unit_normal)) <= 1e-6:
        return None
    return build_triplet_direction_priors(
        crop_size,
        cond_mask,
        global_unit_normal,
        -global_unit_normal,
        mask_mode=mask_mode,
    )


def swap_triplet_branch_channels(
    dense_gt_np: np.ndarray,
    dir_priors_np: np.ndarray = None,
):
    """Swap branch channel groups [0:3] and [3:6] for GT and optional priors."""
    dense = np.asarray(dense_gt_np, dtype=np.float32)
    if dense.ndim != 4 or dense.shape[0] < 6:
        raise ValueError(f"dense_gt_np must have at least 6 channels, got shape {tuple(dense.shape)}")
    swapped_dense = np.concatenate([dense[3:6], dense[0:3]], axis=0).astype(np.float32, copy=False)
    if dir_priors_np is None:
        return swapped_dense, None
    priors = np.asarray(dir_priors_np, dtype=np.float32)
    if priors.ndim != 4 or priors.shape[0] != 6:
        raise ValueError(f"dir_priors_np must have shape (6, D, H, W), got {tuple(priors.shape)}")
    swapped_priors = np.concatenate([priors[3:6], priors[0:3]], axis=0).astype(np.float32, copy=False)
    return swapped_dense, swapped_priors


def maybe_swap_triplet_branch_channels(
    dense_gt_np: np.ndarray,
    dir_priors_np: np.ndarray = None,
    swap_prob: float = 0.0,
    rng=None,
):
    """Randomly swap triplet branch channels, returning updated tensors and channel order."""
    p = float(swap_prob)
    if not np.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"swap_prob must satisfy 0 <= p <= 1, got {swap_prob!r}")
    if rng is None:
        rng = random

    triplet_channel_order_np = np.array([0, 1], dtype=np.int64)
    if p > 0.0 and float(rng.random()) < p:
        dense_gt_np, dir_priors_np = swap_triplet_branch_channels(dense_gt_np, dir_priors_np)
        triplet_channel_order_np = np.array([1, 0], dtype=np.int64)
    return dense_gt_np, dir_priors_np, triplet_channel_order_np


def align_triplet_branch_channels_to_priors(
    dense_gt_np: np.ndarray,
    dir_priors_np: np.ndarray,
    cond_mask: np.ndarray | None = None,
):
    """Deterministically align triplet GT branch order to prior slots.

    Returns:
        dense_gt_np: possibly swapped dense GT channels
        dir_priors_np: unchanged
        channel_order_np: mapping from current channels to original branch ids
            ([0, 1] for no swap, [1, 0] when swapped)
    """
    dense = np.asarray(dense_gt_np, dtype=np.float32)
    priors = np.asarray(dir_priors_np, dtype=np.float32)
    if dense.ndim != 4 or dense.shape[0] < 6:
        raise ValueError(f"dense_gt_np must have at least 6 channels, got shape {tuple(dense.shape)}")
    if priors.ndim != 4 or priors.shape[0] != 6:
        raise ValueError(f"dir_priors_np must have shape (6, D, H, W), got {tuple(priors.shape)}")
    if dense.shape[1:] != priors.shape[1:]:
        raise ValueError(
            "dense_gt_np and dir_priors_np must share spatial shape, got "
            f"{tuple(dense.shape[1:])} vs {tuple(priors.shape[1:])}"
        )

    if cond_mask is None:
        mask = np.any(np.abs(priors) > 0, axis=0)
    else:
        mask = np.asarray(cond_mask, dtype=bool)
        if mask.shape != tuple(dense.shape[1:]):
            raise ValueError(
                "cond_mask shape must match spatial shape "
                f"{tuple(dense.shape[1:])}, got {tuple(mask.shape)}"
            )
    if not bool(mask.any()):
        return dense_gt_np, dir_priors_np, np.array([0, 1], dtype=np.int64)

    # Side-based canonicalization:
    # Use slot-0 prior as oriented +n direction and assign branch channels by
    # signed projection on conditioning voxels.
    n = _estimate_mean_unit_direction_from_field(priors[0:3], mask)
    if n is None:
        return dense_gt_np, dir_priors_np, np.array([0, 1], dtype=np.int64)

    def _median_signed_projection(disp_field: np.ndarray) -> float | None:
        vecs = np.asarray(disp_field, dtype=np.float32)[:, mask].T
        if vecs.size == 0:
            return None
        finite = np.isfinite(vecs).all(axis=1)
        vecs = vecs[finite]
        if vecs.shape[0] == 0:
            return None
        mags = np.linalg.norm(vecs, axis=1)
        vecs = vecs[mags > 1e-6]
        if vecs.shape[0] == 0:
            return None
        return float(np.median(vecs @ n))

    s0 = _median_signed_projection(dense[0:3])
    s1 = _median_signed_projection(dense[3:6])
    if s0 is None or s1 is None:
        return dense_gt_np, dir_priors_np, np.array([0, 1], dtype=np.int64)

    # Swap when branch-1 is more +n than branch-0 so slot-0 consistently maps
    # to the +n side defined by dir_priors[0:3].
    if s1 > s0:
        dense_swapped = np.concatenate([dense[3:6], dense[0:3]], axis=0).astype(np.float32, copy=False)
        return dense_swapped, dir_priors_np, np.array([1, 0], dtype=np.int64)
    return dense_gt_np, dir_priors_np, np.array([0, 1], dtype=np.int64)
