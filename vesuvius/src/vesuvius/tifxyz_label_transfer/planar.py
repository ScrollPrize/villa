"""Optional global-planar approximation for TIFXYZ label transfer.

This method deliberately trades local geometric correctness for one fitted 2D
affine. The production nearest-triangle method remains in :mod:`core`.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .core import (
    Surface,
    SurfaceMapper,
    estimate_surface_spacing,
    infer_output_shape,
)


def fit_planar_label_transform(
    source: Surface,
    target: Surface,
    label_shape: Tuple[int, int],
    output_shape: Tuple[int, int],
    affine: Optional[NDArray[np.float64]] = None,
    max_distance: Optional[float] = None,
    nearest_vertices: int = 8,
    label_offset_yx: Tuple[float, float] = (0.0, 0.0),
    vertex_index: str = "kdtree",
    sample_vertices: int = 200_000,
) -> Tuple[NDArray[np.float64], dict]:
    """Fit one 2D affine mapping output pixels to source-label pixels.

    Samples valid target vertices, maps them to the source surface with the
    same geometry machinery ``transfer_array`` uses, and least-squares fits
    ``label_yx = matrix[:, :2] @ output_yx + matrix[:, 2]`` in continuous
    pixel coordinates. Returns the 2x3 matrix and a fit report. Unlike the
    per-pixel transfer this cannot represent fold seams or non-affine drift
    between the two parameterizations; the residuals in the report say how
    much of that the fit is ignoring.
    """

    effective_affine = (
        np.eye(4, dtype=np.float64) if affine is None else affine
    )
    if max_distance is None:
        source_spacing = estimate_surface_spacing(source, effective_affine)
        target_spacing = estimate_surface_spacing(target)
        max_distance = max(1e-3, 0.75 * min(source_spacing, target_spacing))
    if not math.isfinite(max_distance) or max_distance <= 0:
        raise ValueError(f"max_distance must be positive; got {max_distance}")
    if sample_vertices < 3:
        raise ValueError("sample_vertices must be at least 3")

    mapper = SurfaceMapper(
        source,
        affine=effective_affine,
        nearest_vertices=nearest_vertices,
        vertex_index=vertex_index,
        index_max_distance=max_distance,
    )
    assert target.valid is not None
    vertex_rows, vertex_cols = np.nonzero(target.valid)
    if vertex_rows.size == 0:
        raise ValueError("target surface has no valid vertices")
    stride = max(1, vertex_rows.size // sample_vertices)
    vertex_rows = vertex_rows[::stride]
    vertex_cols = vertex_cols[::stride]
    points = np.column_stack(
        (
            target.x[vertex_rows, vertex_cols],
            target.y[vertex_rows, vertex_cols],
            target.z[vertex_rows, vertex_cols],
        )
    )
    rows, cols, accepted, _ = mapper.locate(points, max_distance=max_distance)
    if int(accepted.sum()) < 3:
        raise ValueError(
            "planar fit needs at least 3 mapped correspondences; got "
            f"{int(accepted.sum())} of {points.shape[0]} sampled vertices"
        )

    label_height, label_width = label_shape
    source_height, source_width = source.shape
    output_height, output_width = output_shape
    target_height, target_width = target.shape
    label_yx = np.column_stack(
        (
            rows[accepted] * label_height / source_height
            - label_offset_yx[0],
            cols[accepted] * label_width / source_width - label_offset_yx[1],
        )
    )
    # Pixel-index coordinates of the sampled vertices on the output canvas,
    # matching bilinear_field_tile's centre convention.
    output_yx = np.column_stack(
        (
            vertex_rows[accepted] * output_height / target_height - 0.5,
            vertex_cols[accepted] * output_width / target_width - 0.5,
        )
    )
    design = np.column_stack(
        (output_yx, np.ones(output_yx.shape[0], dtype=np.float64))
    )

    def solve(rows_mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        coefficients, _, rank, _ = np.linalg.lstsq(
            design[rows_mask], label_yx[rows_mask], rcond=None
        )
        if rank < 3:
            raise ValueError(
                "planar fit is degenerate; sampled correspondences are "
                "collinear"
            )
        return coefficients

    coefficients = solve(np.ones(design.shape[0], dtype=bool))
    residual = np.linalg.norm(
        label_yx - design @ coefficients, axis=1
    )
    inliers = residual <= max(1.0, 5.0 * float(np.median(residual)))
    if 3 <= int(inliers.sum()) < residual.size:
        coefficients = solve(inliers)
        residual = np.linalg.norm(label_yx - design @ coefficients, axis=1)

    matrix = coefficients.T  # (2, 3)
    singular_values = np.linalg.svd(matrix[:, :2], compute_uv=False)
    report = {
        "mode": "planar-affine",
        "matrix_output_to_label_px": matrix.tolist(),
        "sampled_target_vertices": int(points.shape[0]),
        "mapped_correspondences": int(accepted.sum()),
        "fit_inliers": int(inliers.sum()),
        "residual_label_px": {
            "p50": float(np.percentile(residual, 50)),
            "p95": float(np.percentile(residual, 95)),
            "max": float(residual.max()),
        },
        "linear_part_singular_values": singular_values.tolist(),
    }
    return matrix, report


def transfer_array_planar(
    source: Surface,
    target: Surface,
    source_label: NDArray,
    output_shape: Optional[Sequence[int]] = None,
    affine: Optional[NDArray[np.float64]] = None,
    max_distance: Optional[float] = None,
    nearest_vertices: int = 8,
    label_offset_yx: Tuple[float, float] = (0.0, 0.0),
    vertex_index: str = "kdtree",
    sample_vertices: int = 200_000,
    fill_value: int | float = 0,
    output: Optional[NDArray] = None,
    valid_output: Optional[NDArray[np.uint8]] = None,
    source_validity: Optional[NDArray] = None,
) -> Tuple[NDArray, NDArray[np.uint8], NDArray[np.float64], dict]:
    """Transfer a label with one global 2D affine instead of per-pixel maps.

    The correspondence between the two parameterizations is discovered from
    a sample of target vertices, then the whole label raster is warped at
    once. Every output pixel whose affine-mapped position lands inside the
    label raster is filled — fold seams and locally rejected geometry do not
    punch holes, at the cost of ignoring any non-affine component of the
    true mapping (quantified by ``residual_label_px`` in the report).
    """

    label = np.asarray(source_label)
    if label.ndim != 2:
        raise ValueError(f"source label must be 2D; got shape {label.shape}")
    propagated_validity = (
        None
        if source_validity is None
        else np.asarray(source_validity, dtype=bool)
    )
    if (
        propagated_validity is not None
        and propagated_validity.shape != label.shape
    ):
        raise ValueError(
            "source validity shape must match source label shape; "
            f"got {propagated_validity.shape} and {label.shape}"
        )
    resolved_shape = infer_output_shape(
        source, label.shape, target, explicit_shape=output_shape
    )
    if output is None:
        output = np.full(resolved_shape, fill_value, dtype=label.dtype)
    elif output.shape != resolved_shape or output.dtype != label.dtype:
        raise ValueError(
            f"output must have shape {resolved_shape} and dtype {label.dtype}; "
            f"got shape={output.shape}, dtype={output.dtype}"
        )
    if valid_output is None:
        valid_output = np.zeros(resolved_shape, dtype=np.uint8)
    elif valid_output.shape != resolved_shape:
        raise ValueError("valid output shape does not match resolved output shape")

    matrix, report = fit_planar_label_transform(
        source,
        target,
        label.shape,
        resolved_shape,
        affine=affine,
        max_distance=max_distance,
        nearest_vertices=nearest_vertices,
        label_offset_yx=label_offset_yx,
        vertex_index=vertex_index,
        sample_vertices=sample_vertices,
    )
    # Block-wise floor warp, matching transfer_array's floor-of-continuous
    # label sampling. Flooring the corner-frame coordinate keeps boundary
    # pixels stable where a rounded centre-frame coordinate would sit on a
    # knife edge at exactly 0 or size-1.
    label_height, label_width = label.shape
    column_indices = np.arange(resolved_shape[1], dtype=np.float64)[None, :]
    block_rows = max(1, min(resolved_shape[0], 1024))
    valid_count = 0
    for row_start in range(0, resolved_shape[0], block_rows):
        row_end = min(resolved_shape[0], row_start + block_rows)
        row_indices = np.arange(row_start, row_end, dtype=np.float64)[:, None]
        label_rows = np.floor(
            matrix[0, 0] * row_indices
            + matrix[0, 1] * column_indices
            + matrix[0, 2]
        ).astype(np.int64)
        label_cols = np.floor(
            matrix[1, 0] * row_indices
            + matrix[1, 1] * column_indices
            + matrix[1, 2]
        ).astype(np.int64)
        in_bounds = (
            (label_rows >= 0)
            & (label_rows < label_height)
            & (label_cols >= 0)
            & (label_cols < label_width)
        )
        label_rows = np.where(in_bounds, label_rows, 0)
        label_cols = np.where(in_bounds, label_cols, 0)
        block_valid = in_bounds
        if propagated_validity is not None:
            block_valid = block_valid & propagated_validity[
                label_rows, label_cols
            ]
        output[row_start:row_end] = np.where(
            block_valid, label[label_rows, label_cols], fill_value
        )
        valid_output[row_start:row_end] = np.where(block_valid, 255, 0)
        valid_count += int(block_valid.sum())
    report["output_valid_fraction"] = float(
        valid_count / (resolved_shape[0] * resolved_shape[1])
    )
    return output, valid_output, matrix, report
