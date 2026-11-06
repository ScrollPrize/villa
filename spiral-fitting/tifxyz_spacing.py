"""Vertex spacing and area of a tifxyz grid, measured in 3D (numpy only, shared by the tifxyz writers)."""

import numpy as np


def grid_spacing_and_area(zyxs, valid_vertex, step_size):
    """Vertex spacing along columns and rows, and surface area, measured in 3D.

    A fitted winding is sampled every ``step_size`` in spiral space and then
    mapped into the scroll, so its real vertex spacing differs from
    ``step_size`` (23-30 voxels for a 20-voxel step on PHerc0813 and
    PHerc0191). ``scale`` must be the reciprocal of the real spacing
    (tifxyz_format.md 5.1), or renders and areas are off by that ratio.
    Returns ``(spacing_cols, spacing_rows, area_vx2)``: the median distance
    between valid neighbours along each axis (``step_size`` if an axis has
    none, or they coincide) and the summed area of the valid quads, rounded
    to whole voxels². Works for ZYX and XYZ grids alike.
    """
    points = np.asarray(zyxs, dtype=np.float64)

    def median_step(a, b, ok):
        distances = np.linalg.norm(a - b, axis=-1)[ok]
        spacing = float(np.median(distances)) if distances.size else 0.0
        return spacing if spacing > 0 and np.isfinite(spacing) else float(step_size)

    spacing_cols = median_step(points[:, 1:], points[:, :-1], valid_vertex[:, 1:] & valid_vertex[:, :-1])
    spacing_rows = median_step(points[1:], points[:-1], valid_vertex[1:] & valid_vertex[:-1])
    valid_quad = valid_vertex[:-1, :-1] & valid_vertex[1:, :-1] & valid_vertex[:-1, 1:] & valid_vertex[1:, 1:]
    p00, p01, p10, p11 = points[:-1, :-1], points[:-1, 1:], points[1:, :-1], points[1:, 1:]
    quad_area = 0.5 * (np.linalg.norm(np.cross(p01 - p00, p10 - p00), axis=-1)
                       + np.linalg.norm(np.cross(p01 - p11, p10 - p11), axis=-1))
    return spacing_cols, spacing_rows, int(round(float(quad_area[valid_quad].sum())))
