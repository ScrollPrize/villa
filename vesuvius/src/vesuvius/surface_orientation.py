"""Label-free hint for the depth orientation of a TIFXYZ surface.

``vc_render_tifxyz`` stacks layers along the grid normal ``N = dP/dU x dP/dV``
(``grid_normal`` in ``Geometry.cpp``). ``--flip-normals`` makes the layer
index grow toward the scroll centre, which is the order of the published
surface volumes and the training labels (see the comment next to the
option in ``vc_render_tifxyz.cpp``). Which flag gives that order depends on
the handedness of this particular grid. With the wrong order the ink model
can return a plausible map at chance level (villa#1648).

Over a few millimetres a wrapped scroll sheet usually curves around the
scroll axis, so its curvature vector points inward. This module measures the
share of ``|kappa . N|`` that points inward over a fixed physical stencil.
This is a heuristic: where the sheet bends the other way the hint is wrong,
as it is on parts of flat or warped fragments.

Note that :meth:`vesuvius.tifxyz.Tifxyz.compute_normals` returns ``ty x tx``,
which is ``-N`` in this convention. The module depends only on numpy, so
that ``surface_preflight`` keeps working without the tifxyz extras.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

DEFAULT_STENCIL_MM = 4.0
DEFAULT_MIN_DECISIVENESS = 0.5
DEFAULT_MAX_VERTICES = 4_000_000
DEFAULT_MIN_VERTICES = 100
_VOXEL_SIZE_IN_NAME = re.compile(r"-(\d+(?:\.\d+)?)um\.tifxyz$")


def voxel_size_from_name(path: Path | str) -> Optional[float]:
    """Voxel size in micrometres from a published mesh name such as
    ``20260317000000-on-20250728140407-9.362um.tifxyz``; ``None`` if absent."""
    match = _VOXEL_SIZE_IN_NAME.search(Path(path).name)
    return float(match.group(1)) if match else None


def grid_normals(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``(P[c+1] - P[c-1]) x (P[r+1] - P[r-1])`` as in ``grid_normal_int``.

    ``points`` is ``(rows, cols, 3)`` in x, y, z order.  Vertices without four
    valid neighbours get NaN.
    """
    normals = np.full(points.shape, np.nan, dtype=points.dtype)
    if points.shape[0] < 3 or points.shape[1] < 3:
        return normals
    along_u = points[1:-1, 2:] - points[1:-1, :-2]
    along_v = points[2:, 1:-1] - points[:-2, 1:-1]
    usable = (
        valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
    )
    cross = np.cross(along_u, along_v)
    length = np.linalg.norm(cross, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cross = cross / length
    usable &= length[..., 0] > 0
    cross[~usable] = np.nan
    normals[1:-1, 1:-1] = cross
    return normals


def depth_orientation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    scale: float | tuple[float, float],
    voxel_size_um: float,
    valid: Optional[np.ndarray] = None,
    stencil_mm: float = DEFAULT_STENCIL_MM,
    min_decisiveness: float = DEFAULT_MIN_DECISIVENESS,
    max_vertices: int = DEFAULT_MAX_VERTICES,
    min_vertices: int = DEFAULT_MIN_VERTICES,
) -> dict[str, Any]:
    """Which way the renderer's grid normal points relative to the scroll centre.

    Parameters
    ----------
    x, y, z:
        Stored-resolution coordinate rasters in voxels of the mesh's volume.
    scale:
        The tifxyz ``meta.json`` scale (grid cells per voxel along columns and
        rows), e.g. ``(0.05, 0.05)``; a single number applies to both.
    voxel_size_um:
        Voxel size of the volume the coordinates refer to.
    valid:
        Optional vertex mask; by default finite vertices with ``x != -1`` and
        ``z > 0``.
    stencil_mm:
        Physical length of the curvature stencil on each side of a vertex.
    min_decisiveness:
        ``|2 * inward_fraction - 1|`` below which the verdict is undetermined.
    max_vertices:
        Larger grids are decimated by a whole factor first, which keeps memory
        bounded; the stencil stays the same physical length, and holes in the
        full-resolution grid still break the decimated edges across them.
    min_vertices:
        Fewer vertices with a usable stencil give ``too_small``.

    Returns
    -------
    dict
        ``inward_fraction`` (share of ``|kappa . N|`` with ``N`` pointing
        inward), ``decisiveness``, ``grid_normal`` (``"inward"``,
        ``"outward"`` or ``"undetermined"``), ``status`` (``"determined"``,
        ``"undetermined"`` or ``"too_small"``) and ``vc_render_flip_normals``:
        ``True`` when ``vc_render_tifxyz`` needs ``--flip-normals`` for the
        layer index to grow toward the scroll centre, ``False`` when it must
        not be passed, ``None`` otherwise.  The verdict assumes a wrapped
        scroll sheet; it means nothing for a flat fragment.
    """
    scale_u, scale_v = (scale, scale) if np.isscalar(scale) else (scale[0], scale[1])
    scale_u, scale_v, voxel_size_um = float(scale_u), float(scale_v), float(voxel_size_um)
    if not all(np.isfinite(v) and v > 0 for v in (scale_u, scale_v, voxel_size_um, stencil_mm)):
        raise ValueError("scale, voxel_size_um and stencil_mm must be positive and finite")
    x, y, z = (np.asarray(a) for a in (x, y, z))
    rows, cols = x.shape
    step = max(1, int(np.ceil(np.sqrt(rows * cols / max_vertices))))
    if valid is None:
        valid = np.ones(x.shape, bool)
    valid = np.asarray(valid, bool)
    # An edge of the decimated grid is broken when any full-resolution vertex
    # on it is invalid, so that decimation cannot step over holes.
    gaps = [None, None]
    if step > 1:
        full_valid = valid.copy()
        for a in (x, y, z):
            full_valid &= np.isfinite(a)
        full_valid &= (x != -1) & (z > 0)
        invalid = ~full_valid
        for axis in (0, 1):
            inv = invalid[:, ::step] if axis == 0 else invalid[::step, :]
            before = np.cumsum(inv, axis=axis, dtype=np.int32)
            pad = [(0, 0), (0, 0)]
            pad[axis] = (1, 0)
            before = np.pad(before, pad)
            starts = np.arange(0, (inv.shape[axis] - 1) // step * step, step)
            ends = starts + step + 1
            gaps[axis] = (np.take(before, ends, axis=axis) - np.take(before, starts, axis=axis)) > 0
        del full_valid, invalid
    valid = valid[::step, ::step]
    points = np.stack(
        [np.asarray(a[::step, ::step], np.float32) for a in (x, y, z)], axis=-1
    )
    valid = valid & np.isfinite(points).all(-1) & (points[..., 0] != -1) & (points[..., 2] > 0)
    # Grid spacing in voxels along columns (U) and rows (V) after decimation.
    spacing_u, spacing_v = step / scale_u, step / scale_v
    k_u = max(1, int(round(stencil_mm * 1000.0 / voxel_size_um / spacing_u)))
    k_v = max(1, int(round(stencil_mm * 1000.0 / voxel_size_um / spacing_v)))
    report: dict[str, Any] = {
        "stencil_mm": stencil_mm,
        "stencil_cells": [k_u, k_v],
        "decimation": step,
        "voxel_size_um": voxel_size_um,
        "min_decisiveness": min_decisiveness,
        "vertices_used": 0,
        "inward_fraction": None,
        "decisiveness": 0.0,
        "grid_normal": "undetermined",
        "status": "too_small",
        "vc_render_flip_normals": None,
    }

    normals = grid_normals(points, valid)
    curvature = np.zeros(points.shape, np.float32)
    has_curvature = np.zeros(valid.shape, bool)
    for axis, k, spacing in ((1, k_u, spacing_u), (0, k_v, spacing_v)):
        if points.shape[axis] <= 2 * k:
            continue
        lo = [slice(None)] * 2
        mid = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[axis], mid[axis], hi[axis] = slice(None, -2 * k), slice(k, -k), slice(2 * k, None)
        lo, mid, hi = tuple(lo), tuple(mid), tuple(hi)
        ok = valid[lo] & valid[mid] & valid[hi]
        # A stencil that spans a seam, hole or spike measures the jump, not the
        # curvature. Every vertex along it must be valid and every grid edge
        # at most 3x the nominal spacing; both half-chords must also be no
        # longer than 1.5x the grid run.
        edge = [slice(None)] * 2
        edge_next = [slice(None)] * 2
        edge[axis], edge_next[axis] = slice(None, -1), slice(1, None)
        edge, edge_next = tuple(edge), tuple(edge_next)
        bad = ~(valid[edge] & valid[edge_next])
        bad |= np.linalg.norm(points[edge_next] - points[edge], axis=-1) > 3 * spacing
        if gaps[axis] is not None:
            gap = gaps[axis]
            n = bad.shape[axis]
            bad |= gap[:n] if axis == 0 else gap[:, :n]
        pad = [(0, 0), (0, 0)]
        pad[axis] = (1, 0)
        bad_before = np.pad(np.cumsum(bad, axis=axis, dtype=np.int32), pad)
        # bad edges between lo and hi = bad_before[hi] - bad_before[lo]
        ok &= (bad_before[hi] - bad_before[lo]) == 0
        limit = 1.5 * k * spacing
        ok &= np.linalg.norm(points[hi] - points[mid], axis=-1) <= limit
        ok &= np.linalg.norm(points[mid] - points[lo], axis=-1) <= limit
        second = points[hi] - 2 * points[mid] + points[lo]
        curvature[mid] += np.where(ok[..., None], second, 0.0)
        has_curvature[mid] |= ok

    use = has_curvature & np.isfinite(normals).all(-1)
    dot = np.einsum("ijc,ijc->ij", curvature, np.nan_to_num(normals))[use]
    del curvature, normals
    weight = np.abs(dot)
    # Cap each vertex at 3x the 90th percentile so that a few large defects
    # that pass the continuity checks cannot outvote the sheet's ordinary
    # curvature. (Not the median: along the scroll axis the sheet is nearly
    # straight, and many vertices carry almost no weight.)
    if weight.size:
        weight = np.minimum(weight, 3.0 * np.percentile(weight, 90))
    report["vertices_used"] = int(use.sum())
    if report["vertices_used"] < min_vertices or weight.sum() == 0:
        return report

    inward = float(weight[dot > 0].sum() / weight.sum())
    decisiveness = abs(2.0 * inward - 1.0)
    report["inward_fraction"] = inward
    report["decisiveness"] = decisiveness
    report["status"] = "undetermined"
    if decisiveness >= min_decisiveness:
        report["status"] = "determined"
        report["grid_normal"] = "inward" if inward > 0.5 else "outward"
        # Layers are stacked along +N unless flipped; they must grow inward.
        report["vc_render_flip_normals"] = inward < 0.5
    return report
