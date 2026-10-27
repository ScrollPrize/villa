"""Grid isometry metrics for tifxyz surfaces.

A tifxyz surface is a 2D grid of 3D points whose ``meta.json`` promises, via
``scale``, that neighbouring nodes sit ``1 / scale`` voxels apart. Nothing in the
format records whether the grid keeps that promise, so segments cannot be
compared for stretch by a standard number. This module measures that: how close
each grid edge is to the promised step, the grid angle, and flatboi's
area-weighted stretch metrics (see ``volume-cartographer/libs/flatboi/flatboi.cpp``
``stretch_metrics`` / ``triangle_stretch``).

The result is a plain dict, written under the ``isometry`` key of ``meta.json`` by
``vesuvius.tifxyz.TifxyzWriter``. Invalid vertices are skipped exactly as
``vc::surface`` skips cells for the area metric: a corner that is non-finite or the
``(-1, -1, -1)`` sentinel is not part of any edge, cell or triangle.

This is the Python half of ScrollPrize/villa issue #2279; the C++ writers
(``GrowPatch`` / ``GrowSurface``) are the schema authority, so the metric
definitions here are stated explicitly and mirror the issue's specification —
and the reference implementation the issue links to (``kit tifxyz isometry``
from ChaseHendrick/Scrolls), which this module matches field for field on real
segments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import tifffile
from numpy.typing import NDArray

# Bumped when the metric thresholds/definitions change, so a stored block can be
# read against the rules that produced it.
ISOMETRY_VERSION = 1

# A vertex is the format's invalid sentinel when all three coordinates are -1.
_SENTINEL = -1.0

# An edge is "stretched" (the sheet is pulled apart or collapsed there) when it
# is more than 1.5x or less than 1/1.5x the promised step.
_STRETCH_FACTOR = 1.5

# Shares reported by within_10pct / within_25pct: the multiplicative band the issue
# asks for ("within a factor 1.1 and 1.25 of step"), i.e. ratio in [1/factor, factor].
_WITHIN_FACTOR_10 = 1.1
_WITHIN_FACTOR_25 = 1.25


def _valid_vertices(
    x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], valid: Optional[NDArray[Any]]
) -> NDArray[np.bool_]:
    """True where a grid node is a real point (finite and not the -1 sentinel)."""
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    sentinel = (x == _SENTINEL) & (y == _SENTINEL) & (z == _SENTINEL)
    ok = finite & ~sentinel
    if valid is not None:
        ok = ok & np.asarray(valid, dtype=bool)
    return ok


def _edges(
    points: NDArray[np.float64], ok: NDArray[np.bool_], axis: int, step: float
) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Ratios (edge length / promised step) and pair mask for every valid edge.

    ``axis=1`` walks the u (column) direction, ``axis=0`` the v (row) direction.
    """
    if axis == 1:
        pair = ok[:, :-1] & ok[:, 1:]
        delta = points[:, 1:, :] - points[:, :-1, :]
    else:
        pair = ok[:-1, :] & ok[1:, :]
        delta = points[1:, :, :] - points[:-1, :, :]
    lengths = np.linalg.norm(delta, axis=-1)[pair]
    ratios = lengths / step if step else lengths
    return ratios, pair


def _triangle_stretch_terms(
    v3: NDArray[np.float64], uv2: NDArray[np.float64], faces: NDArray[np.int64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Vectorized port of flatboi's ``triangle_stretch``.

    Returns (L2-squared, Linf = largest singular value, area3d, area2d).
    """
    q1, q2, q3 = (v3[faces[:, k]] for k in range(3))
    p1, p2, p3 = (uv2[faces[:, k]] for k in range(3))
    s1, t1 = p1[:, 0], p1[:, 1]
    s2, t2 = p2[:, 0], p2[:, 1]
    s3, t3 = p3[:, 0], p3[:, 1]
    area = ((s2 - s1) * (t3 - t1) - (s3 - s1) * (t2 - t1)) / 2.0
    area_abs = np.abs(area)
    good = area_abs >= 1e-30
    area_safe = np.where(good, area, 1.0)
    ss = (q1 * (t2 - t3)[:, None] + q2 * (t3 - t1)[:, None] + q3 * (t1 - t2)[:, None]) / (
        2.0 * area_safe[:, None]
    )
    st = (q1 * (s3 - s2)[:, None] + q2 * (s1 - s3)[:, None] + q3 * (s2 - s1)[:, None]) / (
        2.0 * area_safe[:, None]
    )
    a = np.einsum("ij,ij->i", ss, ss)
    b = np.einsum("ij,ij->i", ss, st)
    c = np.einsum("ij,ij->i", st, st)
    g = np.sqrt(np.maximum(((a + c) + np.sqrt((a - c) ** 2 + 4.0 * b * b)) / 2.0, 0.0))
    l2sq = (a + c) / 2.0
    ab = np.linalg.norm(q2 - q1, axis=1)
    bc = np.linalg.norm(q3 - q2, axis=1)
    ca = np.linalg.norm(q1 - q3, axis=1)
    s = (ab + bc + ca) / 2.0
    area3d = np.sqrt(np.maximum(s * (s - ab) * (s - bc) * (s - ca), 0.0))
    return (
        np.where(good, l2sq, 0.0),
        np.where(good, g, 0.0),
        np.where(good, area3d, 0.0),
        np.where(good, area_abs, 0.0),
    )


def _weighted_median(data: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    if data.size == 0 or weights.sum() <= 0:
        return 0.0
    order = np.argsort(data, kind="stable")
    acc = np.cumsum(weights[order])
    k = int(np.searchsorted(acc, 0.5 * weights.sum()))
    return float(data[order[min(k, len(order) - 1)]])


def _flatboi_stretch_metrics(
    v3: NDArray[np.float64], uv2: NDArray[np.float64], faces: NDArray[np.int64]
) -> dict:
    """flatboi ``stretch_metrics`` verbatim: L2 mean/median, Linf, area error."""
    if faces.size == 0:
        return {"l2_mean": 0.0, "l2_median": 0.0, "linf": 0.0, "area_error": 0.0}
    l2sq, g, area3d, area2d = _triangle_stretch_terms(v3, uv2, faces)
    sum_a3 = area3d.sum()
    l2_mean = float(np.sqrt((l2sq * area3d).sum() / sum_a3)) if sum_a3 > 0 else 0.0
    sum_a2 = area2d.sum()
    alpha = area3d / sum_a3 if sum_a3 > 0 else np.zeros_like(area3d)
    beta = area2d / sum_a2 if sum_a2 > 0 else np.zeros_like(area2d)
    per_tri = np.where(alpha > beta, 1.0 - beta / (alpha + 1e-30), 1.0 - alpha / (beta + 1e-30))
    return {
        "l2_mean": l2_mean,
        "l2_median": _weighted_median(l2sq, area3d),
        "linf": float(g.max()) if g.size else 0.0,
        "area_error": float(per_tri.sum() / max(1, len(faces))),
    }


def compute_grid_isometry(
    x: NDArray[Any],
    y: NDArray[Any],
    z: NDArray[Any],
    *,
    valid: Optional[NDArray[Any]] = None,
    scale: Tuple[float, float] = (1.0, 1.0),
    version: int = ISOMETRY_VERSION,
) -> dict:
    """Measure how isometric a tifxyz grid is against its promised step.

    Parameters
    ----------
    x, y, z : NDArray
        Grid coordinate arrays of shape (H, W) at stored resolution.
    valid : NDArray, optional
        Boolean validity mask of shape (H, W); ``None`` uses only the finite /
        non-sentinel test.
    scale : tuple
        The surface's ``(scale_y, scale_x)``, so the promised step is
        ``(1 / scale_x, 1 / scale_y)`` voxels along the u (column) and v (row) axes.
    version : int
        Written as ``version`` so a stored block can be read against the rules
        that produced it.

    Returns
    -------
    dict
        ``version``, ``step_vx``, ``edges``, ``within_10pct``, ``within_25pct``,
        ``median_abs_log_ratio``, ``area_ratio``, ``angle_deg`` (a ``[p5, p50, p95]``
        list), ``stretched_nodes``, ``stretched_area_vx2`` and flatboi's ``l2_mean`` /
        ``l2_median`` / ``linf`` / ``area_error``.

        ``within_10pct`` / ``within_25pct`` are the share of edges whose length ratio
        to the promised step is in the multiplicative band ``[1/1.1, 1.1]`` /
        ``[1/1.25, 1.25]`` — the issue's "within a factor 1.1 / 1.25 of step".

        ``stretched_area_vx2`` is ``stretched_nodes`` times the promised cell area
        (``step_u * step_v``), the issue's "their area at one grid cell per node".
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(f"x/y/z shapes differ: {x.shape}, {y.shape}, {z.shape}")

    scale_y, scale_x = float(scale[0]), float(scale[1])
    step_u = 1.0 / scale_x if scale_x else 1.0
    step_v = 1.0 / scale_y if scale_y else 1.0

    block: dict = {
        "version": int(version),
        "step_vx": [step_u, step_v],
        "edges": 0,
        "within_10pct": 0.0,
        "within_25pct": 0.0,
        "median_abs_log_ratio": 0.0,
        "area_ratio": 0.0,
        "angle_deg": [0.0, 0.0, 0.0],
        "stretched_nodes": 0,
        "stretched_area_vx2": 0.0,
        "l2_mean": 0.0,
        "l2_median": 0.0,
        "linf": 0.0,
        "area_error": 0.0,
    }

    ok = _valid_vertices(x, y, z, valid)
    if ok.shape[0] < 2 or ok.shape[1] < 2 or not ok.any():
        return block

    points = np.stack([x, y, z], axis=-1)
    ratio_u, pair_u = _edges(points, ok, axis=1, step=step_u)
    ratio_v, pair_v = _edges(points, ok, axis=0, step=step_v)
    ratios = np.concatenate([ratio_u, ratio_v])
    block["edges"] = int(ratios.size)
    if ratios.size:
        block["within_10pct"] = float(
            np.mean((ratios >= 1.0 / _WITHIN_FACTOR_10) & (ratios <= _WITHIN_FACTOR_10))
        )
        block["within_25pct"] = float(
            np.mean((ratios >= 1.0 / _WITHIN_FACTOR_25) & (ratios <= _WITHIN_FACTOR_25))
        )
        block["median_abs_log_ratio"] = float(np.median(np.abs(np.log(ratios))))

    # Vertices carried by a stretched edge.
    low = 1.0 / _STRETCH_FACTOR
    stretched = np.zeros_like(ok)
    for ratio, pair, axis in ((ratio_u, pair_u, 1), (ratio_v, pair_v, 0)):
        if not ratio.size:
            continue
        bad = np.zeros_like(pair)
        bad[pair] = (ratio > _STRETCH_FACTOR) | (ratio < low)
        touched = np.zeros_like(ok)
        if axis == 1:
            touched[:, :-1] = bad
            touched[:, 1:] |= bad
        else:
            touched[:-1, :] = bad
            touched[1:, :] |= bad
        stretched |= touched
    block["stretched_nodes"] = int((stretched & ok).sum())
    # Issue #2279 defines this as "their area at one grid cell per node": the stretched
    # node count times the promised cell area. Summing the 3D area of every cell a
    # stretched node touches instead would over-count, since one node can touch four
    # cells (the reference implementation, kit tifxyz isometry, uses the count form).
    block["stretched_area_vx2"] = float(block["stretched_nodes"] * step_u * step_v)

    valid_cell = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    n_cells = int(valid_cell.sum())
    if n_cells:
        rows, cols = np.nonzero(valid_cell)
        p00 = points[rows, cols]
        p10 = points[rows, cols + 1]
        p01 = points[rows + 1, cols]
        p11 = points[rows + 1, cols + 1]
        tri_a = 0.5 * np.linalg.norm(np.cross(p10 - p00, p11 - p00), axis=-1)
        tri_b = 0.5 * np.linalg.norm(np.cross(p11 - p00, p01 - p00), axis=-1)
        cell_area = tri_a + tri_b
        if step_u and step_v:
            block["area_ratio"] = float(cell_area.sum() / (step_u * step_v * n_cells))

        # Grid angle at each cell's (r, c) corner, between its u and v edges.
        e_u = p10 - p00
        e_v = p01 - p00
        denom = np.linalg.norm(e_u, axis=-1) * np.linalg.norm(e_v, axis=-1)
        good = denom > 0
        if good.any():
            cos = np.clip(np.einsum("ij,ij->i", e_u[good], e_v[good]) / denom[good], -1.0, 1.0)
            angle = np.degrees(np.arccos(cos))
            # A [p5, p50, p95] list, the shape the reference (kit tifxyz isometry) and
            # issue #2279's block use, so a stored block reads the same either way.
            block["angle_deg"] = [float(v) for v in np.percentile(angle, (5.0, 50.0, 95.0))]

        # flatboi's stretch metrics over the two triangles of every valid cell.
        height, width = points.shape[:2]
        col_idx, row_idx = np.meshgrid(np.arange(width), np.arange(height))
        uv2 = np.stack([col_idx * step_u, row_idx * step_v], axis=-1).astype(np.float64)
        corner = rows * width + cols
        right = corner + 1
        down = corner + width
        down_right = corner + width + 1
        faces = np.concatenate(
            [
                np.stack([corner, right, down_right], axis=-1),
                np.stack([corner, down_right, down], axis=-1),
            ],
            axis=0,
        ).astype(np.int64)
        block.update(
            _flatboi_stretch_metrics(
                points.reshape(-1, 3), uv2.reshape(-1, 2), faces
            )
        )
    return block


def update_meta_isometry(
    segment: Union[str, Path], *, version: int = ISOMETRY_VERSION
) -> dict:
    """Write the ``isometry`` block into an existing tifxyz ``meta.json``.

    The coordinate TIFFs are never touched — only ``meta.json`` is rewritten — so
    this can stamp the block onto an already saved (or published) segment. It is
    the Python counterpart of the ``vc_tifxyz --isometry`` backfill proposed in
    ScrollPrize/villa issue #2279.

    Parameters
    ----------
    segment : str | Path
        A tifxyz directory containing ``x.tif``, ``y.tif``, ``z.tif`` and
        ``meta.json`` (and optionally ``mask.tif``).
    version : int
        Written as ``version`` in the block.

    Returns
    -------
    dict
        The block that was written.
    """
    segment = Path(segment)
    meta_path = segment / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no meta.json in {segment}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    try:
        x = tifffile.imread(str(segment / "x.tif"))
        y = tifffile.imread(str(segment / "y.tif"))
        z = tifffile.imread(str(segment / "z.tif"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{segment} is not a tifxyz segment: {exc}") from exc

    # meta.json stores scale as [scale_x, scale_y]; the metric takes (scale_y, scale_x).
    scale = meta.get("scale") or [1.0, 1.0]
    valid = None
    mask_path = segment / "mask.tif"
    if mask_path.exists():
        valid = tifffile.imread(str(mask_path)) > 0

    block = compute_grid_isometry(
        x,
        y,
        z,
        valid=valid,
        scale=(float(scale[1]), float(scale[0])),
        version=version,
    )
    meta["isometry"] = block
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return block
