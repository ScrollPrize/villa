"""Shared float64 triangle-sweep math; extraction preserves arithmetic and ordering."""
import numpy as np

DIAGONALS = {
    "AC": ((0, 2, 1), (0, 3, 2)),
    "BD": ((0, 3, 1), (1, 3, 2)),
}


def triangulate(grid, diagonal):
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 3 or grid.shape[-1] != 3 or min(grid.shape[:2]) < 2 or not np.isfinite(grid).all():
        raise ValueError("Require finite grid of three-component vectors")
    corners = np.stack([grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]], axis=-2)
    return np.stack([corners[..., indices, :] for indices in DIAGONALS[diagonal]]).reshape(-1, 3, 3)


def area_coefficients(triangles, vectors):
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    dn1 = vectors[:, 1] - vectors[:, 0]
    dn2 = vectors[:, 2] - vectors[:, 0]
    return np.cross(dn1, dn2), np.cross(dn1, e2) + np.cross(e1, dn2), np.cross(e1, e2)


def quadratic_minimum(a, b, c, lo, hi):
    left = a * lo**2 + b * lo + c
    right = a * hi**2 + b * hi + c
    depths = np.where(left <= right, lo, hi)
    best = np.minimum(left, right)
    stationary = np.divide(-b, 2 * a, out=np.zeros_like(b), where=a != 0)
    interior = (a > 0) & (stationary > lo) & (stationary < hi)
    middle = a * stationary**2 + b * stationary + c
    improve = interior & (middle < best)
    return np.where(improve, middle, best), np.where(improve, stationary, depths)


def evaluate(triangles, vectors, direction, lo, hi):
    a, b, c = area_coefficients(triangles, vectors)
    projected_coefficients = (a @ direction, b @ direction, c @ direction)
    volume_coefficients = tuple(np.einsum("ni,nvi->nv", coefficient, vectors) for coefficient in (a, b, c))
    if np.any(projected_coefficients[2] <= 0) or np.any(volume_coefficients[2] <= 0):
        raise ValueError("Fixed base contains a nonpositive oriented triangle")
    projected, projected_depth = quadratic_minimum(*projected_coefficients, lo, hi)
    volume, volume_depth = quadratic_minimum(*volume_coefficients, lo, hi)
    # Independently evaluate actual displaced vertices at every reported minimum.
    pp = triangles + projected_depth[:, None, None] * vectors
    projected_direct = np.cross(pp[:, 1] - pp[:, 0], pp[:, 2] - pp[:, 0]) @ direction
    vp = triangles[:, None] + volume_depth[..., None, None] * vectors[:, None]
    volume_direct = np.sum(np.cross(vp[:, :, 1] - vp[:, :, 0], vp[:, :, 2] - vp[:, :, 0]) * vectors, axis=-1)
    projection_residual = float(np.abs(projected - projected_direct).max())
    volume_residual = float(np.abs(volume - volume_direct).max())
    if max(projection_residual, volume_residual) > 1e-8:
        raise ValueError("Direct displaced-triangle oracle differs from analytic minima")
    projected_bad = projected <= 0
    volume_bad = (volume <= 0).any(axis=1)
    return {
        "interval_native": [lo, hi], "triangles": len(triangles),
        "projected_nonpositive_triangles": int(projected_bad.sum()),
        "volume_nonpositive_triangles": int(volume_bad.sum()),
        "volume_nonpositive_vertex_quadratics": int((volume <= 0).sum()),
        "minimum_projected_area_ratio": float((projected / projected_coefficients[2]).min()),
        "minimum_volume_jacobian_ratio": float((volume / volume_coefficients[2]).min()),
        "direct_projected_minimum_max_absolute_residual": projection_residual,
        "direct_volume_minimum_max_absolute_residual": volume_residual,
    }, projected_bad, volume_bad
