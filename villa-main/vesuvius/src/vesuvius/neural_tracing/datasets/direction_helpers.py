import numpy as np
from numba import njit

from vesuvius.neural_tracing.datasets.common import voxelize_surface_grid_into


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
def _scatter_trace_point_values_numba(
    velocity_accum: np.ndarray,
    weights: np.ndarray,
    surface_attract: np.ndarray,
    surface_attract_weight: np.ndarray,
    surface_attract_best_dist_sq: np.ndarray,
    enable_surface_attract: bool,
    surface_attract_radius: float,
    pz_in: float,
    py_in: float,
    px_in: float,
    vz_in: float,
    vy_in: float,
    vx_in: float,
) -> None:
    depth, height, width = weights.shape
    pz = float(pz_in)
    py = float(py_in)
    px = float(px_in)
    z = int(np.rint(pz))
    y = int(np.rint(py))
    x = int(np.rint(px))
    if z < 0 or z >= depth or y < 0 or y >= height or x < 0 or x >= width:
        return

    # The previous zero-length line path used steps=1 and visited the same
    # point twice. Keep the t=0 and t=1 arithmetic so accumulated float32
    # values and weights remain bit-for-bit compatible with the old path.
    for i in range(2):
        t = i / 1
        vz = float(vz_in * (1.0 - t) + vz_in * t)
        vy = float(vy_in * (1.0 - t) + vy_in * t)
        vx = float(vx_in * (1.0 - t) + vx_in * t)
        norm = np.sqrt(vz * vz + vy * vy + vx * vx)
        if not np.isfinite(norm) or norm <= 1e-6:
            continue

        velocity_accum[0, z, y, x] += np.float32(vz / norm)
        velocity_accum[1, z, y, x] += np.float32(vy / norm)
        velocity_accum[2, z, y, x] += np.float32(vx / norm)
        weights[z, y, x] += 1.0

    if enable_surface_attract:
        pz = float(pz_in * (1.0 - 0.0) + pz_in * 0.0)
        py = float(py_in * (1.0 - 0.0) + py_in * 0.0)
        px = float(px_in * (1.0 - 0.0) + px_in * 0.0)
        _stamp_trace_surface_attract_numba(
            surface_attract,
            surface_attract_weight,
            surface_attract_best_dist_sq,
            pz,
            py,
            px,
            surface_attract_radius,
        )
        pz = float(pz_in * (1.0 - 1.0) + pz_in * 1.0)
        py = float(py_in * (1.0 - 1.0) + py_in * 1.0)
        px = float(px_in * (1.0 - 1.0) + px_in * 1.0)
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
def _scatter_trace_point_numba(
    velocity_accum: np.ndarray,
    weights: np.ndarray,
    surface_attract: np.ndarray,
    surface_attract_weight: np.ndarray,
    surface_attract_best_dist_sq: np.ndarray,
    enable_surface_attract: bool,
    surface_attract_radius: float,
    p: np.ndarray,
    v: np.ndarray,
) -> None:
    _scatter_trace_point_values_numba(
        velocity_accum,
        weights,
        surface_attract,
        surface_attract_weight,
        surface_attract_best_dist_sq,
        enable_surface_attract,
        surface_attract_radius,
        float(p[0]),
        float(p[1]),
        float(p[2]),
        float(v[0]),
        float(v[1]),
        float(v[2]),
    )


@njit
def _scatter_trace_line_values_numba(
    velocity_accum: np.ndarray,
    weights: np.ndarray,
    surface_attract: np.ndarray,
    surface_attract_weight: np.ndarray,
    surface_attract_best_dist_sq: np.ndarray,
    enable_surface_attract: bool,
    surface_attract_radius: float,
    p0z: float,
    p0y: float,
    p0x: float,
    p1z: float,
    p1y: float,
    p1x: float,
    v0z: float,
    v0y: float,
    v0x: float,
    v1z: float,
    v1y: float,
    v1x: float,
) -> None:
    dz_line = float(p1z - p0z)
    dy_line = float(p1y - p0y)
    dx_line = float(p1x - p0x)
    max_delta = max(abs(dz_line), abs(dy_line), abs(dx_line))
    steps = int(np.ceil(max_delta))
    steps = max(steps, 1)
    depth, height, width = weights.shape
    for i in range(steps + 1):
        t = i / steps
        pz = float(p0z * (1.0 - t) + p1z * t)
        py = float(p0y * (1.0 - t) + p1y * t)
        px = float(p0x * (1.0 - t) + p1x * t)
        z = int(np.rint(pz))
        y = int(np.rint(py))
        x = int(np.rint(px))
        if z < 0 or z >= depth or y < 0 or y >= height or x < 0 or x >= width:
            continue

        vz = float(v0z * (1.0 - t) + v1z * t)
        vy = float(v0y * (1.0 - t) + v1y * t)
        vx = float(v0x * (1.0 - t) + v1x * t)
        norm = np.sqrt(vz * vz + vy * vy + vx * vx)
        if not np.isfinite(norm) or norm <= 1e-6:
            continue

        velocity_accum[0, z, y, x] += np.float32(vz / norm)
        velocity_accum[1, z, y, x] += np.float32(vy / norm)
        velocity_accum[2, z, y, x] += np.float32(vx / norm)
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
def _scatter_trace_line_numba(
    velocity_accum: np.ndarray,
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
) -> None:
    _scatter_trace_line_values_numba(
        velocity_accum,
        weights,
        surface_attract,
        surface_attract_weight,
        surface_attract_best_dist_sq,
        enable_surface_attract,
        surface_attract_radius,
        float(p0[0]),
        float(p0[1]),
        float(p0[2]),
        float(p1[0]),
        float(p1[1]),
        float(p1[2]),
        float(v0[0]),
        float(v0[1]),
        float(v0[2]),
        float(v1[0]),
        float(v1[1]),
        float(v1[2]),
    )


@njit
def _logical_surface_shape(surface_a: np.ndarray, surface_b: np.ndarray, has_b: bool, concat_axis: int):
    if not has_b:
        return surface_a.shape[0], surface_a.shape[1]
    if concat_axis == 0:
        return surface_a.shape[0] + surface_b.shape[0], surface_a.shape[1]
    return surface_a.shape[0], surface_a.shape[1] + surface_b.shape[1]


@njit
def _logical_surface_value(
    surface_a: np.ndarray,
    surface_b: np.ndarray,
    has_b: bool,
    concat_axis: int,
    r: int,
    c: int,
    k: int,
) -> float:
    if not has_b:
        return float(surface_a[r, c, k])
    if concat_axis == 0:
        rows_a = surface_a.shape[0]
        if r < rows_a:
            return float(surface_a[r, c, k])
        return float(surface_b[r - rows_a, c, k])
    cols_a = surface_a.shape[1]
    if c < cols_a:
        return float(surface_a[r, c, k])
    return float(surface_b[r, c - cols_a, k])


@njit
def _logical_surface_point_finite(
    surface_a: np.ndarray,
    surface_b: np.ndarray,
    has_b: bool,
    concat_axis: int,
    r: int,
    c: int,
) -> bool:
    z = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 0)
    y = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 1)
    x = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 2)
    return np.isfinite(z) and np.isfinite(y) and np.isfinite(x)


@njit
def _logical_surface_vector(
    surface_a: np.ndarray,
    surface_b: np.ndarray,
    has_b: bool,
    concat_axis: int,
    axis: int,
    sign: float,
    r: int,
    c: int,
):
    rows, cols = _logical_surface_shape(surface_a, surface_b, has_b, concat_axis)
    if not _logical_surface_point_finite(surface_a, surface_b, has_b, concat_axis, r, c):
        return False, 0.0, 0.0, 0.0

    tangent_valid = False
    tz = 0.0
    ty = 0.0
    tx = 0.0
    if axis == 0:
        if (
            rows >= 3
            and r > 0
            and r < rows - 1
            and _logical_surface_point_finite(surface_a, surface_b, has_b, concat_axis, r - 1, c)
            and _logical_surface_point_finite(surface_a, surface_b, has_b, concat_axis, r + 1, c)
        ):
            tz = np.float32(0.5) * (
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r + 1, c, 0)
                - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r - 1, c, 0)
            )
            ty = np.float32(0.5) * (
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r + 1, c, 1)
                - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r - 1, c, 1)
            )
            tx = np.float32(0.5) * (
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r + 1, c, 2)
                - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r - 1, c, 2)
            )
            tangent_valid = True
        elif (
            rows >= 2
            and r < rows - 1
            and _logical_surface_point_finite(surface_a, surface_b, has_b, concat_axis, r + 1, c)
        ):
            tz = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r + 1, c, 0) - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 0)
            ty = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r + 1, c, 1) - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 1)
            tx = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r + 1, c, 2) - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 2)
            tangent_valid = True
        elif (
            rows >= 2
            and r > 0
            and _logical_surface_point_finite(surface_a, surface_b, has_b, concat_axis, r - 1, c)
        ):
            tz = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 0) - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r - 1, c, 0)
            ty = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 1) - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r - 1, c, 1)
            tx = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 2) - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r - 1, c, 2)
            tangent_valid = True
    else:
        if (
            cols >= 3
            and c > 0
            and c < cols - 1
            and _logical_surface_point_finite(surface_a, surface_b, has_b, concat_axis, r, c - 1)
            and _logical_surface_point_finite(surface_a, surface_b, has_b, concat_axis, r, c + 1)
        ):
            tz = np.float32(0.5) * (
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c + 1, 0)
                - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c - 1, 0)
            )
            ty = np.float32(0.5) * (
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c + 1, 1)
                - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c - 1, 1)
            )
            tx = np.float32(0.5) * (
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c + 1, 2)
                - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c - 1, 2)
            )
            tangent_valid = True
        elif (
            cols >= 2
            and c < cols - 1
            and _logical_surface_point_finite(surface_a, surface_b, has_b, concat_axis, r, c + 1)
        ):
            tz = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c + 1, 0) - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 0)
            ty = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c + 1, 1) - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 1)
            tx = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c + 1, 2) - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 2)
            tangent_valid = True
        elif (
            cols >= 2
            and c > 0
            and _logical_surface_point_finite(surface_a, surface_b, has_b, concat_axis, r, c - 1)
        ):
            tz = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 0) - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c - 1, 0)
            ty = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 1) - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c - 1, 1)
            tx = _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 2) - _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c - 1, 2)
            tangent_valid = True

    if not tangent_valid:
        return False, 0.0, 0.0, 0.0

    vz = np.float32(tz) * np.float32(sign)
    vy = np.float32(ty) * np.float32(sign)
    vx = np.float32(tx) * np.float32(sign)
    norm = np.float32(np.sqrt(np.float32(vz * vz + vy * vy + vx * vx)))
    if not np.isfinite(norm) or norm <= 1e-6:
        return False, 0.0, 0.0, 0.0
    return True, np.float32(vz / norm), np.float32(vy / norm), np.float32(vx / norm)


@njit
def _fill_logical_surface_vectors_numba(
    vectors: np.ndarray,
    valid: np.ndarray,
    surface_a: np.ndarray,
    surface_b: np.ndarray,
    has_b: bool,
    concat_axis: int,
    tangent_axis: int,
    tangent_sign: float,
) -> None:
    rows, cols = _logical_surface_shape(surface_a, surface_b, has_b, concat_axis)
    for r in range(rows):
        for c in range(cols):
            point_valid, vz, vy, vx = _logical_surface_vector(
                surface_a, surface_b, has_b, concat_axis, tangent_axis, tangent_sign, r, c
            )
            if not point_valid:
                continue
            vectors[r, c, 0] = np.float32(vz)
            vectors[r, c, 1] = np.float32(vy)
            vectors[r, c, 2] = np.float32(vx)
            valid[r, c] = True


@njit
def _scatter_logical_trace_surface_numba(
    velocity_accum: np.ndarray,
    weights: np.ndarray,
    surface_a: np.ndarray,
    surface_b: np.ndarray,
    has_b: bool,
    concat_axis: int,
    vectors: np.ndarray,
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
            _scatter_trace_point_values_numba(
                velocity_accum,
                weights,
                surface_attract,
                surface_attract_weight,
                surface_attract_best_dist_sq,
                enable_surface_attract,
                surface_attract_radius,
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 0),
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 1),
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 2),
                float(vectors[r, c, 0]),
                float(vectors[r, c, 1]),
                float(vectors[r, c, 2]),
            )

    for r in range(rows):
        for c in range(cols - 1):
            if not (valid[r, c] and valid[r, c + 1]):
                continue
            _scatter_trace_line_values_numba(
                velocity_accum,
                weights,
                surface_attract,
                surface_attract_weight,
                surface_attract_best_dist_sq,
                enable_surface_attract,
                surface_attract_radius,
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 0),
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 1),
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 2),
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c + 1, 0),
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c + 1, 1),
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c + 1, 2),
                float(vectors[r, c, 0]),
                float(vectors[r, c, 1]),
                float(vectors[r, c, 2]),
                float(vectors[r, c + 1, 0]),
                float(vectors[r, c + 1, 1]),
                float(vectors[r, c + 1, 2]),
            )

    for r in range(rows - 1):
        for c in range(cols):
            if not (valid[r, c] and valid[r + 1, c]):
                continue
            _scatter_trace_line_values_numba(
                velocity_accum,
                weights,
                surface_attract,
                surface_attract_weight,
                surface_attract_best_dist_sq,
                enable_surface_attract,
                surface_attract_radius,
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 0),
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 1),
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r, c, 2),
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r + 1, c, 0),
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r + 1, c, 1),
                _logical_surface_value(surface_a, surface_b, has_b, concat_axis, r + 1, c, 2),
                float(vectors[r, c, 0]),
                float(vectors[r, c, 1]),
                float(vectors[r, c, 2]),
                float(vectors[r + 1, c, 0]),
                float(vectors[r + 1, c, 1]),
                float(vectors[r + 1, c, 2]),
            )


@njit
def _voxelize_split_surfaces_and_scatter_trace_numba(
    cond_mask: np.ndarray,
    masked_mask: np.ndarray,
    velocity_accum: np.ndarray,
    weights: np.ndarray,
    cond_surface: np.ndarray,
    masked_surface: np.ndarray,
    surface_a: np.ndarray,
    surface_b: np.ndarray,
    concat_axis: int,
    tangent_axis: int,
    tangent_sign: float,
):
    voxelize_surface_grid_into(cond_mask, cond_surface)
    voxelize_surface_grid_into(masked_mask, masked_surface)

    if int(concat_axis) == 0:
        rows = int(surface_a.shape[0] + surface_b.shape[0])
        cols = int(surface_a.shape[1])
    else:
        rows = int(surface_a.shape[0])
        cols = int(surface_a.shape[1] + surface_b.shape[1])

    vectors = np.zeros((rows, cols, 3), dtype=np.float32)
    valid = np.zeros((rows, cols), dtype=np.bool_)
    _fill_logical_surface_vectors_numba(
        vectors,
        valid,
        surface_a,
        surface_b,
        True,
        int(concat_axis),
        int(tangent_axis),
        float(tangent_sign),
    )
    if not bool(valid.any()):
        return False

    surface_attract = np.zeros((3, 1, 1, 1), dtype=np.float32)
    surface_attract_weight = np.zeros((1, 1, 1), dtype=np.float32)
    surface_attract_best_dist_sq = np.zeros((1, 1, 1), dtype=np.float32)
    _scatter_logical_trace_surface_numba(
        velocity_accum,
        weights,
        surface_a,
        surface_b,
        True,
        int(concat_axis),
        vectors,
        valid,
        surface_attract,
        surface_attract_weight,
        surface_attract_best_dist_sq,
        False,
        0.0,
    )
    return True


def build_split_surface_masks_and_trace_targets(
    crop_size,
    cond_direction: str,
    *,
    cond_surface_local: np.ndarray,
    masked_surface_local: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Build split masks and sparse trace targets from split ordered surfaces.

    This matches separately voxelizing the two split masks and then scattering
    sparse trace targets over their logical concatenation, while avoiding
    separate full-volume temporaries for the split masks.
    """
    crop_size = tuple(int(v) for v in crop_size)
    if len(crop_size) != 3:
        raise ValueError(f"crop_size must have length 3, got {crop_size!r}")

    cond_surface = np.asarray(cond_surface_local, dtype=np.float32)
    masked_surface = np.asarray(masked_surface_local, dtype=np.float32)
    if cond_surface.ndim != 3 or cond_surface.shape[2] != 3:
        raise ValueError(f"cond_surface_local must have shape (H, W, 3), got {tuple(cond_surface.shape)}")
    if masked_surface.ndim != 3 or masked_surface.shape[2] != 3:
        raise ValueError(f"masked_surface_local must have shape (H, W, 3), got {tuple(masked_surface.shape)}")

    direction = str(cond_direction).lower()
    if direction == "left":
        surface_a = cond_surface
        surface_b = masked_surface
        concat_axis = 1
    elif direction == "right":
        surface_a = masked_surface
        surface_b = cond_surface
        concat_axis = 1
    elif direction == "up":
        surface_a = cond_surface
        surface_b = masked_surface
        concat_axis = 0
    elif direction == "down":
        surface_a = masked_surface
        surface_b = cond_surface
        concat_axis = 0
    else:
        raise ValueError(
            "cond_direction must be one of {'up', 'down', 'left', 'right'}, "
            f"got {cond_direction!r}"
        )

    cond_mask = np.zeros(crop_size, dtype=np.uint8)
    masked_mask = np.zeros(crop_size, dtype=np.uint8)
    velocity_accum = np.zeros((3, *crop_size), dtype=np.float32)
    weights = np.zeros(crop_size, dtype=np.float32)
    tangent_axis, tangent_sign = _velocity_axis_and_sign(direction)
    ok = _voxelize_split_surfaces_and_scatter_trace_numba(
        cond_mask,
        masked_mask,
        velocity_accum,
        weights,
        cond_surface,
        masked_surface,
        surface_a,
        surface_b,
        int(concat_axis),
        int(tangent_axis),
        float(tangent_sign),
    )
    if not ok:
        return None

    valid_vox = weights > 0.0
    if not bool(valid_vox.any()):
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
    if bool(active_finite.any()):
        finite_coords = tuple(coord[active_finite] for coord in active_coords)
        velocity[(slice(None), *finite_coords)] = (
            active_velocity[:, active_finite] / norms[active_finite][None]
        )
        valid_vox[finite_coords] = True

    return {
        "cond_gt": cond_mask,
        "masked_seg": masked_mask,
        "velocity_dir": velocity.astype(np.float32, copy=False),
        "trace_loss_weight": valid_vox[None].astype(np.float32, copy=False),
    }

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
