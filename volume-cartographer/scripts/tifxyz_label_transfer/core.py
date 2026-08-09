"""Core geometry operations for TIFXYZ-to-TIFXYZ label transfer.

The mapper works in the common 3D volume coordinate system:

1. Sample each target output pixel on the target TIFXYZ surface.
2. Find the nearest triangle on the source TIFXYZ surface.
3. Recover source grid coordinates with triangle barycentrics.
4. Sample the categorical source label with nearest-neighbour interpolation.

TIFXYZ arrays contain vertices, while ``vc_render_tifxyz`` samples pixel
centres between them.  The conversion used here mirrors that convention:

    grid_coordinate = (pixel_index + 0.5) * stored_size / rendered_size

No render command is required as long as the label covers the complete,
unrotated and uncropped source canvas.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]


def _positive_lround(value: float) -> int:
    """Match C++ std::lround for a known-positive value."""

    return int(math.floor(value + 0.5))


def _surface_coordinate_fingerprint(
    x: FloatArray, y: FloatArray, z: FloatArray
) -> str:
    """Digest of a strided coordinate sample, identifying a surface.

    Shapes alone cannot distinguish two segments, so UV-cache keys include
    this. A 64x64 stride sample keeps the cost negligible for any raster.
    """

    digest = hashlib.sha256()
    for array in (x, y, z):
        row_stride = max(1, array.shape[0] // 64)
        col_stride = max(1, array.shape[1] // 64)
        sample = np.ascontiguousarray(
            array[::row_stride, ::col_stride], dtype=np.float32
        )
        digest.update(sample.shape[0].to_bytes(4, "little"))
        digest.update(sample.shape[1].to_bytes(4, "little"))
        digest.update(sample.tobytes())
    return digest.hexdigest()[:16]


def _resolve_worker_count(workers: Optional[int], item_count: int) -> int:
    if workers is None:
        workers = os.cpu_count() or 1
    if workers < 1:
        raise ValueError(f"workers must be positive; got {workers}")
    return max(1, min(workers, item_count))


def _run_ordered(process, items, worker_count, on_result=None):
    """Run ``process`` over ``items``, delivering results in item order.

    Workers may only write to disjoint output regions; ``on_result`` runs on
    the calling thread in submission order, so order-sensitive accumulation
    (statistics, progress) stays deterministic regardless of thread timing.
    The submission window is bounded to keep pending results from piling up.
    """

    if worker_count <= 1:
        for item in items:
            result = process(item)
            if on_result is not None:
                on_result(result)
        return
    window = worker_count * 2
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        pending: deque = deque()
        for item in items:
            pending.append(pool.submit(process, item))
            if len(pending) >= window:
                result = pending.popleft().result()
                if on_result is not None:
                    on_result(result)
        while pending:
            result = pending.popleft().result()
            if on_result is not None:
                on_result(result)


@dataclass
class Surface:
    """A stored-resolution TIFXYZ coordinate grid."""

    x: FloatArray
    y: FloatArray
    z: FloatArray
    scale_yx: Tuple[float, float] = (1.0, 1.0)
    valid: Optional[BoolArray] = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.x.ndim != 2 or self.y.ndim != 2 or self.z.ndim != 2:
            raise ValueError("TIFXYZ coordinate arrays must be two-dimensional")
        if self.x.shape != self.y.shape or self.x.shape != self.z.shape:
            raise ValueError(
                "TIFXYZ coordinate shapes differ: "
                f"x={self.x.shape}, y={self.y.shape}, z={self.z.shape}"
            )
        if self.x.shape[0] < 2 or self.x.shape[1] < 2:
            raise ValueError(
                f"TIFXYZ coordinate grid must be at least 2x2; got {self.x.shape}"
            )
        if len(self.scale_yx) != 2 or any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in self.scale_yx
        ):
            raise ValueError(f"invalid TIFXYZ scale: {self.scale_yx}")

        coordinate_valid = (
            np.isfinite(self.x)
            & np.isfinite(self.y)
            & np.isfinite(self.z)
            & (self.x != -1)
            & (self.y != -1)
            & (self.z > 0)
        )
        if self.valid is None:
            self.valid = coordinate_valid
        else:
            if self.valid.shape != self.x.shape:
                raise ValueError(
                    f"valid mask shape {self.valid.shape} does not match "
                    f"coordinate shape {self.x.shape}"
                )
            self.valid = np.asarray(self.valid, dtype=bool) & coordinate_valid

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.x.shape[0]), int(self.x.shape[1])

    @property
    def full_resolution_shape(self) -> Tuple[int, int]:
        height, width = self.shape
        scale_y, scale_x = self.scale_yx
        return (
            max(1, _positive_lround(height / scale_y)),
            max(1, _positive_lround(width / scale_x)),
        )


@dataclass(frozen=True)
class AffineChoice:
    """Selected affine and the evidence used to select its direction."""

    matrix: NDArray[np.float64]
    direction: str
    forward_median: Optional[float] = None
    forward_p95: Optional[float] = None
    inverse_median: Optional[float] = None
    inverse_p95: Optional[float] = None


@dataclass
class MappingStats:
    """Streaming mapping statistics."""

    target_pixels: int = 0
    target_surface_valid: int = 0
    mapped_pixels: int = 0
    seam_filled_pixels: int = 0
    inherited_filled_pixels: int = 0
    distance_sum: float = 0.0
    distance_min: float = math.inf
    distance_max: float = 0.0
    _distance_samples: Optional[list[NDArray[np.float32]]] = None
    _distance_sample_count: int = 0

    def __post_init__(self) -> None:
        if self._distance_samples is None:
            self._distance_samples = []

    def add(
        self,
        target_pixels: int,
        target_surface_valid: int,
        distances: FloatArray,
        sample_limit: int = 1_000_000,
    ) -> None:
        self.target_pixels += int(target_pixels)
        self.target_surface_valid += int(target_surface_valid)
        if distances.size == 0:
            return
        finite = np.asarray(distances[np.isfinite(distances)], dtype=np.float32)
        if finite.size == 0:
            return
        self.mapped_pixels += int(finite.size)
        self.distance_sum += float(finite.sum(dtype=np.float64))
        self.distance_min = min(self.distance_min, float(finite.min()))
        self.distance_max = max(self.distance_max, float(finite.max()))

        assert self._distance_samples is not None
        remaining = max(0, sample_limit - self._distance_sample_count)
        if remaining:
            if finite.size <= remaining:
                self._distance_samples.append(finite.copy())
                self._distance_sample_count += int(finite.size)
            else:
                indices = np.linspace(
                    0, finite.size - 1, num=remaining, dtype=np.int64
                )
                self._distance_samples.append(finite[indices].copy())
                self._distance_sample_count += int(remaining)

    def as_dict(self) -> dict:
        coverage = (
            self.mapped_pixels / self.target_surface_valid
            if self.target_surface_valid
            else 0.0
        )
        result = {
            "target_pixels": self.target_pixels,
            "target_surface_valid_pixels": self.target_surface_valid,
            "mapped_pixels": self.mapped_pixels,
            "seam_filled_pixels": self.seam_filled_pixels,
            "inherited_filled_pixels": self.inherited_filled_pixels,
            "mapping_coverage": coverage,
            "distance_mean": (
                self.distance_sum / self.mapped_pixels
                if self.mapped_pixels
                else None
            ),
            "distance_min": (
                self.distance_min if self.mapped_pixels else None
            ),
            "distance_max": (
                self.distance_max if self.mapped_pixels else None
            ),
        }
        assert self._distance_samples is not None
        if self._distance_samples:
            samples = np.concatenate(self._distance_samples)
            result["distance_p50"] = float(np.percentile(samples, 50))
            result["distance_p95"] = float(np.percentile(samples, 95))
            result["distance_p99"] = float(np.percentile(samples, 99))
        else:
            result["distance_p50"] = None
            result["distance_p95"] = None
            result["distance_p99"] = None
        return result


def load_affine(path: Path | str) -> NDArray[np.float64]:
    """Load a registration JSON affine as a 4x4 matrix."""

    affine_path = Path(path)
    with affine_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "transformation_matrix" not in data:
        raise ValueError(
            f"{affine_path} does not contain 'transformation_matrix'"
        )
    matrix = np.asarray(data["transformation_matrix"], dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.vstack((matrix, np.array([0.0, 0.0, 0.0, 1.0])))
    if matrix.shape != (4, 4):
        raise ValueError(
            f"affine must be 3x4 or 4x4; got shape {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("affine contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
        raise ValueError("affine bottom row must be [0, 0, 0, 1]")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-15:
        raise ValueError("affine is singular")
    return matrix


def apply_affine(
    points: FloatArray, matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply an XYZ homogeneous affine to an ``(N, 3)`` point array."""

    points_array = np.asarray(points)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3); got {points_array.shape}")
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    return points_array.astype(np.float64, copy=False) @ linear.T + translation


def _sample_valid_points(
    surface: Surface, limit: int
) -> NDArray[np.float64]:
    assert surface.valid is not None
    flat_indices = np.flatnonzero(surface.valid)
    if flat_indices.size == 0:
        raise ValueError(f"surface {surface.name!r} has no valid points")
    if flat_indices.size > limit:
        selection = np.linspace(
            0, flat_indices.size - 1, num=limit, dtype=np.int64
        )
        flat_indices = flat_indices[selection]
    return np.column_stack(
        (
            surface.x.ravel()[flat_indices],
            surface.y.ravel()[flat_indices],
            surface.z.ravel()[flat_indices],
        )
    ).astype(np.float64, copy=False)


def _alignment_score(
    source_points: FloatArray,
    target_tree: cKDTree,
    matrix: NDArray[np.float64],
) -> Tuple[float, float]:
    transformed = apply_affine(source_points, matrix)
    distances, _ = target_tree.query(transformed, k=1, workers=-1)
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return math.inf, math.inf
    return float(np.median(finite)), float(np.percentile(finite, 95))


def choose_affine_direction(
    source: Surface,
    target: Surface,
    matrix: NDArray[np.float64],
    direction: str = "auto",
    sample_limit: int = 100_000,
    ambiguity_ratio: float = 0.9,
) -> AffineChoice:
    """Choose whether a registration affine should be used forward or inverted.

    ``forward`` means ``p_target = M @ p_source``.  In ``auto`` mode both
    directions are scored using sampled nearest target-vertex distances.
    """

    if direction not in {"auto", "forward", "inverse"}:
        raise ValueError(
            "affine direction must be 'auto', 'forward', or 'inverse'"
        )
    inverse = np.linalg.inv(matrix)
    if direction == "forward":
        return AffineChoice(matrix=matrix, direction="forward")
    if direction == "inverse":
        return AffineChoice(matrix=inverse, direction="inverse")

    source_points = _sample_valid_points(source, sample_limit)
    target_points = _sample_valid_points(target, sample_limit)
    target_tree = cKDTree(target_points, compact_nodes=True, balanced_tree=True)
    forward_median, forward_p95 = _alignment_score(
        source_points, target_tree, matrix
    )
    inverse_median, inverse_p95 = _alignment_score(
        source_points, target_tree, inverse
    )

    if not math.isfinite(forward_median) and not math.isfinite(inverse_median):
        raise ValueError("could not score either affine direction")
    best = min(forward_median, inverse_median)
    worst = max(forward_median, inverse_median)
    if worst == 0.0 or best / worst > ambiguity_ratio:
        raise ValueError(
            "affine direction is ambiguous from surface geometry: "
            f"forward median={forward_median:.4g}, "
            f"inverse median={inverse_median:.4g}; "
            "pass --affine-direction forward or inverse"
        )

    if forward_median < inverse_median:
        selected, selected_direction = matrix, "forward"
    else:
        selected, selected_direction = inverse, "inverse"
    return AffineChoice(
        matrix=selected,
        direction=selected_direction,
        forward_median=forward_median,
        forward_p95=forward_p95,
        inverse_median=inverse_median,
        inverse_p95=inverse_p95,
    )


def infer_output_shape(
    source: Surface,
    source_label_shape: Sequence[int],
    target: Surface,
    explicit_shape: Optional[Sequence[int]] = None,
) -> Tuple[int, int]:
    """Infer the target label shape from TIFXYZ and source-label dimensions.

    The source label reveals the effective render scale:

    ``render_scale_y = label_height * source_scale_y / source_stored_height``

    Applying that render scale to the target TIFXYZ gives the corresponding
    target canvas.  Stored-resolution and full-resolution labels are therefore
    handled without special cases.
    """

    if explicit_shape is not None:
        if len(explicit_shape) != 2:
            raise ValueError("explicit output shape must contain HEIGHT WIDTH")
        output = int(explicit_shape[0]), int(explicit_shape[1])
        if output[0] <= 0 or output[1] <= 0:
            raise ValueError(f"invalid explicit output shape: {output}")
        return output

    if len(source_label_shape) < 2:
        raise ValueError(f"invalid source label shape: {source_label_shape}")
    label_height = int(source_label_shape[-2])
    label_width = int(source_label_shape[-1])
    if label_height <= 0 or label_width <= 0:
        raise ValueError(f"invalid source label shape: {source_label_shape}")

    source_height, source_width = source.shape
    source_scale_y, source_scale_x = source.scale_yx
    target_height, target_width = target.shape
    target_scale_y, target_scale_x = target.scale_yx

    render_scale_y = label_height * source_scale_y / source_height
    render_scale_x = label_width * source_scale_x / source_width
    inferred = (
        max(
            1,
            _positive_lround(
                target_height * render_scale_y / target_scale_y
            ),
        ),
        max(
            1,
            _positive_lround(
                target_width * render_scale_x / target_scale_x
            ),
        ),
    )

    # Renders are cropped independently of the TIFXYZ canvas, so a label
    # raster within a fraction of a percent of the source canvas is a
    # crop/offset of it, not evidence of a different render scale. Snap the
    # output to the target's native canvas in that case; otherwise a
    # spurious ~0.1% scale forces every downstream consumer to resample.
    native_candidates = (
        (
            max(1, _positive_lround(target_height / target_scale_y)),
            max(1, _positive_lround(target_width / target_scale_x)),
        ),
        (target_height, target_width),
    )
    for candidate in native_candidates:
        if all(
            abs(actual - expected) <= max(2.0, 0.005 * expected)
            for actual, expected in zip(inferred, candidate)
        ):
            return candidate
    return inferred


def estimate_surface_spacing(
    surface: Surface,
    matrix: Optional[NDArray[np.float64]] = None,
    sample_limit: int = 200_000,
) -> float:
    """Estimate median stored-grid edge length in the mapped coordinate space."""

    assert surface.valid is not None
    distances: list[FloatArray] = []
    for axis in (0, 1):
        if axis == 0:
            edge_valid = surface.valid[:-1, :] & surface.valid[1:, :]
            a = (
                surface.x[:-1, :],
                surface.y[:-1, :],
                surface.z[:-1, :],
            )
            b = (
                surface.x[1:, :],
                surface.y[1:, :],
                surface.z[1:, :],
            )
        else:
            edge_valid = surface.valid[:, :-1] & surface.valid[:, 1:]
            a = (
                surface.x[:, :-1],
                surface.y[:, :-1],
                surface.z[:, :-1],
            )
            b = (
                surface.x[:, 1:],
                surface.y[:, 1:],
                surface.z[:, 1:],
            )

        indices = np.flatnonzero(edge_valid)
        if indices.size == 0:
            continue
        per_axis_limit = max(1, sample_limit // 2)
        if indices.size > per_axis_limit:
            take = np.linspace(
                0, indices.size - 1, num=per_axis_limit, dtype=np.int64
            )
            indices = indices[take]
        pa = np.column_stack(tuple(component.ravel()[indices] for component in a))
        pb = np.column_stack(tuple(component.ravel()[indices] for component in b))
        if matrix is not None:
            pa = apply_affine(pa, matrix)
            pb = apply_affine(pb, matrix)
        edge_distances = np.linalg.norm(pb - pa, axis=1)
        edge_distances = edge_distances[
            np.isfinite(edge_distances) & (edge_distances > 0)
        ]
        if edge_distances.size:
            distances.append(edge_distances)
    if not distances:
        raise ValueError(f"cannot estimate spacing for surface {surface.name!r}")
    return float(np.median(np.concatenate(distances)))


def _maximum_edge_length(
    surface: Surface,
    matrix: Optional[NDArray[np.float64]] = None,
) -> float:
    """Upper-bound the mapped stored-grid edge length over the whole surface.

    Unlike :func:`estimate_surface_spacing` this examines every valid edge
    (no sampling) because it feeds a completeness guarantee: stretched or
    folded TIFXYZ grids have edges far above the median. Affine mapping is
    bounded by the linear part's spectral norm, since
    ``|A(p) - A(q)| = |M (p - q)|``.
    """

    assert surface.valid is not None
    best = 0.0
    for axis in (0, 1):
        if axis == 0:
            edge_valid = surface.valid[:-1, :] & surface.valid[1:, :]
            dx = surface.x[1:, :] - surface.x[:-1, :]
            dy = surface.y[1:, :] - surface.y[:-1, :]
            dz = surface.z[1:, :] - surface.z[:-1, :]
        else:
            edge_valid = surface.valid[:, :-1] & surface.valid[:, 1:]
            dx = surface.x[:, 1:] - surface.x[:, :-1]
            dy = surface.y[:, 1:] - surface.y[:, :-1]
            dz = surface.z[:, 1:] - surface.z[:, :-1]
        lengths = np.sqrt(
            np.square(dx, dtype=np.float64)
            + np.square(dy, dtype=np.float64)
            + np.square(dz, dtype=np.float64)
        )[edge_valid]
        lengths = lengths[np.isfinite(lengths)]
        if lengths.size:
            best = max(best, float(lengths.max()))
    if best <= 0.0:
        raise ValueError(
            f"cannot bound edge length for surface {surface.name!r}"
        )
    if matrix is not None:
        linear = np.asarray(matrix, dtype=np.float64)[:3, :3]
        best *= float(
            np.linalg.svd(linear, compute_uv=False)[0]
        )
    return best


def _closest_points_on_triangles(
    points: FloatArray,
    a: FloatArray,
    b: FloatArray,
    c: FloatArray,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return squared distances and barycentric weights for N point/triangle pairs."""

    p = np.asarray(points, dtype=np.float64)
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    c64 = np.asarray(c, dtype=np.float64)
    count = p.shape[0]
    best_distance = np.full(count, np.inf, dtype=np.float64)
    best_bary = np.zeros((count, 3), dtype=np.float64)

    def consider_segment(
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        start_weights: Tuple[float, float, float],
        end_weights: Tuple[float, float, float],
    ) -> None:
        edge = end - start
        length_sq = np.einsum("ij,ij->i", edge, edge)
        safe = length_sq > 1e-20
        t = np.zeros(count, dtype=np.float64)
        t[safe] = np.einsum(
            "ij,ij->i", p[safe] - start[safe], edge[safe]
        ) / length_sq[safe]
        np.clip(t, 0.0, 1.0, out=t)
        closest = start + t[:, None] * edge
        distance = np.einsum("ij,ij->i", p - closest, p - closest)
        update = distance < best_distance
        if not np.any(update):
            return
        weights_start = np.asarray(start_weights, dtype=np.float64)
        weights_end = np.asarray(end_weights, dtype=np.float64)
        weights = (
            (1.0 - t[:, None]) * weights_start[None, :]
            + t[:, None] * weights_end[None, :]
        )
        best_distance[update] = distance[update]
        best_bary[update] = weights[update]

    consider_segment(a64, b64, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    consider_segment(b64, c64, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    consider_segment(c64, a64, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))

    ab = b64 - a64
    ac = c64 - a64
    ap = p - a64
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab)
    d21 = np.einsum("ij,ij->i", ap, ac)
    denominator = d00 * d11 - d01 * d01
    nondegenerate = np.abs(denominator) > 1e-20
    v = np.zeros(count, dtype=np.float64)
    w = np.zeros(count, dtype=np.float64)
    v[nondegenerate] = (
        d11[nondegenerate] * d20[nondegenerate]
        - d01[nondegenerate] * d21[nondegenerate]
    ) / denominator[nondegenerate]
    w[nondegenerate] = (
        d00[nondegenerate] * d21[nondegenerate]
        - d01[nondegenerate] * d20[nondegenerate]
    ) / denominator[nondegenerate]
    u = 1.0 - v - w
    inside = nondegenerate & (u >= 0.0) & (v >= 0.0) & (w >= 0.0)
    if np.any(inside):
        projection = a64 + v[:, None] * ab + w[:, None] * ac
        distance = np.einsum("ij,ij->i", p - projection, p - projection)
        update = inside & (distance < best_distance)
        best_distance[update] = distance[update]
        best_bary[update, 0] = u[update]
        best_bary[update, 1] = v[update]
        best_bary[update, 2] = w[update]
    return best_distance, best_bary


_CELL_NEIGHBOR_OFFSETS = np.array(
    [
        (di, dj, dk)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        for dk in (-1, 0, 1)
    ],
    dtype=np.int64,
)


class GridVertexIndex:
    """Uniform 3D bucket index over near-uniformly spaced points.

    TIFXYZ stored grids have near-constant vertex spacing, so nearest-vertex
    search does not need a general KD-tree: bucketing the points into a
    uniform grid and scanning each query's 3x3x3 cell neighbourhood finds
    every point within ``cell_size`` of the query (the neighbourhood covers
    at least ``cell_size`` beyond the query's own cell along every axis).
    Queries in empty regions return no candidates instead of walking to a
    distant winding, which the distance threshold would reject anyway.
    """

    def __init__(self, points: FloatArray, cell_size: float) -> None:
        candidate_points = np.asarray(points, dtype=np.float64)
        if candidate_points.ndim != 2 or candidate_points.shape[1] != 3:
            raise ValueError(
                f"points must have shape (N, 3); got {candidate_points.shape}"
            )
        if candidate_points.shape[0] == 0:
            raise ValueError("cannot index zero points")
        if not math.isfinite(cell_size) or cell_size <= 0:
            raise ValueError(f"cell_size must be positive; got {cell_size}")
        self.count = int(candidate_points.shape[0])
        self.cell_size = float(cell_size)
        self.origin = candidate_points.min(axis=0)
        cells = np.floor(
            (candidate_points - self.origin) / self.cell_size
        ).astype(np.int64)
        self.grid_shape = tuple(int(v) for v in cells.max(axis=0) + 1)
        cell_ids = np.ravel_multi_index(
            (cells[:, 0], cells[:, 1], cells[:, 2]), self.grid_shape
        )
        self.order = np.argsort(cell_ids, kind="stable")
        self.sorted_cell_ids = cell_ids[self.order]
        self.sorted_points = candidate_points[self.order]

    def query(
        self,
        points: FloatArray,
        k: int,
        chunk_size: int = 16_384,
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Return ``(distances, indices)`` of up to ``k`` nearest candidates.

        Matches ``cKDTree.query`` conventions for missing neighbours: their
        index is ``self.count`` and their distance is ``inf``. Only points in
        the query's 3x3x3 cell neighbourhood are candidates.
        """

        if k < 1:
            raise ValueError("k must be positive")
        query = np.asarray(points, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 3:
            raise ValueError(f"query points must be (N, 3); got {query.shape}")
        total = query.shape[0]
        distances = np.full((total, k), np.inf, dtype=np.float64)
        indices = np.full((total, k), self.count, dtype=np.int64)
        spans = [
            (start, min(total, start + max(1, chunk_size)))
            for start in range(0, total, max(1, chunk_size))
        ]
        if len(spans) <= 1:
            for start, end in spans:
                self._query_chunk(
                    query[start:end],
                    k,
                    distances[start:end],
                    indices[start:end],
                )
            return distances, indices
        # Chunks write to disjoint output slices; the large NumPy kernels
        # release the GIL, so threads parallelise like cKDTree's workers=-1.
        with ThreadPoolExecutor(max_workers=min(len(spans), os.cpu_count() or 1)) as pool:
            futures = [
                pool.submit(
                    self._query_chunk,
                    query[start:end],
                    k,
                    distances[start:end],
                    indices[start:end],
                )
                for start, end in spans
            ]
            for future in futures:
                future.result()
        return distances, indices

    def _query_chunk(
        self,
        chunk: NDArray[np.float64],
        k: int,
        distances_out: NDArray[np.float64],
        indices_out: NDArray[np.int64],
    ) -> None:
        count = chunk.shape[0]
        if count == 0:
            return
        grid_shape = np.asarray(self.grid_shape, dtype=np.int64)
        cell_float = np.floor((chunk - self.origin) / self.cell_size)
        # Non-finite or absurdly distant queries must not wrap during the
        # int64 cast; anything outside [-2, shape + 1] has no candidates.
        cell_float = np.where(np.isfinite(cell_float), cell_float, -10.0)
        cells = np.clip(cell_float, -2.0, grid_shape + 1.0).astype(np.int64)

        neighbors = cells[:, None, :] + _CELL_NEIGHBOR_OFFSETS[None, :, :]
        in_grid = np.all(
            (neighbors >= 0) & (neighbors < grid_shape[None, None, :]), axis=2
        )
        safe = np.clip(neighbors, 0, grid_shape[None, None, :] - 1)
        ids = np.ravel_multi_index(
            (safe[..., 0], safe[..., 1], safe[..., 2]), self.grid_shape
        ).ravel()
        starts = np.searchsorted(self.sorted_cell_ids, ids, side="left")
        ends = np.searchsorted(self.sorted_cell_ids, ids, side="right")
        counts = np.where(in_grid.ravel(), ends - starts, 0)
        per_query = counts.reshape(count, -1).sum(axis=1)
        width = int(per_query.max(initial=0))
        if width == 0:
            return

        # Flatten every (query, neighbour-cell) candidate range. Candidates
        # stay grouped by query because counts iterates query-major.
        range_ends = np.cumsum(counts)
        candidate_total = int(range_ends[-1])
        flat = np.arange(candidate_total, dtype=np.int64)
        within_range = flat - np.repeat(range_ends - counts, counts)
        candidate_sorted = np.repeat(starts, counts) + within_range
        query_ends = np.cumsum(per_query)
        query_of_candidate = np.repeat(
            np.arange(count, dtype=np.int64), per_query
        )
        column = flat - np.repeat(query_ends - per_query, per_query)

        difference = (
            self.sorted_points[candidate_sorted] - chunk[query_of_candidate]
        )
        distance_sq = np.einsum("ij,ij->i", difference, difference)
        padded_distance = np.full((count, width), np.inf, dtype=np.float64)
        padded_index = np.full((count, width), self.count, dtype=np.int64)
        padded_distance[query_of_candidate, column] = distance_sq
        padded_index[query_of_candidate, column] = self.order[candidate_sorted]

        if width > k:
            keep = np.argpartition(padded_distance, k - 1, axis=1)[:, :k]
            selected_distance = np.take_along_axis(padded_distance, keep, 1)
            selected_index = np.take_along_axis(padded_index, keep, 1)
        else:
            selected_distance = padded_distance
            selected_index = padded_index
        found = np.isfinite(selected_distance)
        columns = selected_distance.shape[1]
        distances_out[:, :columns][found] = np.sqrt(selected_distance[found])
        indices_out[:, :columns][found] = selected_index[found]


class SurfaceMapper:
    """Nearest-triangle lookup from target XYZ points to source grid UV."""

    def __init__(
        self,
        source: Surface,
        affine: Optional[NDArray[np.float64]] = None,
        nearest_vertices: int = 8,
        vertex_index: str = "kdtree",
        index_max_distance: Optional[float] = None,
    ) -> None:
        if nearest_vertices < 1:
            raise ValueError("nearest_vertices must be positive")
        if vertex_index not in {"grid", "kdtree"}:
            raise ValueError(
                f"vertex_index must be 'grid' or 'kdtree'; got {vertex_index!r}"
            )
        self.source = source
        self.affine = (
            np.eye(4, dtype=np.float64)
            if affine is None
            else np.asarray(affine, dtype=np.float64)
        )
        self.nearest_vertices = int(nearest_vertices)
        self.vertex_index = vertex_index

        assert source.valid is not None
        self.valid_flat = np.flatnonzero(source.valid)
        if self.valid_flat.size == 0:
            raise ValueError(f"source surface {source.name!r} has no valid points")
        source_points = np.column_stack(
            (
                source.x.ravel()[self.valid_flat],
                source.y.ravel()[self.valid_flat],
                source.z.ravel()[self.valid_flat],
            )
        )
        transformed = apply_affine(source_points, self.affine)
        self.tree: Optional[cKDTree] = None
        self.grid_index: Optional[GridVertexIndex] = None
        self.index_guaranteed_distance = math.inf
        if vertex_index == "kdtree":
            self.tree = cKDTree(
                transformed, compact_nodes=True, balanced_tree=True, leafsize=32
            )
        else:
            try:
                spacing = estimate_surface_spacing(source, self.affine)
                max_edge = _maximum_edge_length(source, self.affine)
            except ValueError:
                # Without a single valid stored-grid edge no quad can map
                # anywhere, so any bucket size is correct; keep construction
                # usable with a bounding-box heuristic.
                extent = transformed.max(axis=0) - transformed.min(axis=0)
                spacing = max(
                    1e-6,
                    float(np.max(extent))
                    / max(1.0, transformed.shape[0] ** (1.0 / 3.0)),
                )
                max_edge = spacing
            if index_max_distance is None:
                index_max_distance = 0.75 * spacing
            self.index_guaranteed_distance = float(index_max_distance)
            # A triangle whose closest point is within the acceptance
            # distance has all vertices within that distance plus one quad
            # diagonal. The diagonal is bounded by two edges, so use the
            # true maximum edge length — stretched or folded grids exceed
            # the median spacing by large factors.
            cell_size = self.index_guaranteed_distance + 2.0 * max_edge
            self.grid_index = GridVertexIndex(transformed, cell_size)

        height, width = source.shape
        self.tx = np.full((height, width), np.nan, dtype=np.float64)
        self.ty = np.full((height, width), np.nan, dtype=np.float64)
        self.tz = np.full((height, width), np.nan, dtype=np.float64)
        self.tx.ravel()[self.valid_flat] = transformed[:, 0]
        self.ty.ravel()[self.valid_flat] = transformed[:, 1]
        self.tz.ravel()[self.valid_flat] = transformed[:, 2]

    def locate(
        self,
        points: FloatArray,
        max_distance: float,
        query_workers: int = -1,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        BoolArray,
        NDArray[np.float64],
    ]:
        """Map target XYZ points to source ``(row, col)`` grid coordinates.

        ``query_workers`` bounds the KDTree query's internal threads; callers
        that already parallelise across batches should pass 1 to avoid
        oversubscription.
        """

        query = np.asarray(points, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 3:
            raise ValueError(f"query points must be (N, 3); got {query.shape}")
        count = query.shape[0]
        best_distance_sq = np.full(count, np.inf, dtype=np.float64)
        best_row = np.full(count, np.nan, dtype=np.float64)
        best_col = np.full(count, np.nan, dtype=np.float64)
        if count == 0:
            return (
                best_row,
                best_col,
                np.zeros(0, dtype=bool),
                np.full(0, np.inf, dtype=np.float64),
            )

        k = min(self.nearest_vertices, self.valid_flat.size)
        if self.grid_index is not None:
            if (
                math.isfinite(max_distance)
                and max_distance > self.index_guaranteed_distance * (1 + 1e-9)
            ):
                raise ValueError(
                    f"max_distance {max_distance} exceeds the grid vertex "
                    "index guarantee "
                    f"{self.index_guaranteed_distance}; construct the "
                    "SurfaceMapper with index_max_distance=max_distance or "
                    "vertex_index='kdtree'"
                )
            _, neighbor_indices = self.grid_index.query(query, k=k)
        else:
            assert self.tree is not None
            _, neighbor_indices = self.tree.query(
                query,
                k=k,
                workers=query_workers,
            )
        if neighbor_indices.ndim == 1:
            neighbor_indices = neighbor_indices[:, None]

        source_height, source_width = self.source.shape
        assert self.source.valid is not None

        # Each neighbour vertex touches up to four stored-grid quads, and
        # adjacent neighbours share most of them. Deduplicate the candidate
        # quads per query and evaluate only distinct (query, quad) pairs;
        # this replaces k * 4 * 2 full-width triangle passes with roughly
        # one pass over the distinct quads.
        neighbor_exists = neighbor_indices < self.valid_flat.size
        safe_neighbors = np.where(neighbor_exists, neighbor_indices, 0)
        flat = self.valid_flat[safe_neighbors]
        vertex_rows = (flat // source_width).astype(np.int64)
        vertex_cols = (flat % source_width).astype(np.int64)
        offset_rows = np.array([-1, -1, 0, 0], dtype=np.int64)
        offset_cols = np.array([-1, 0, -1, 0], dtype=np.int64)
        cell_rows = vertex_rows[:, :, None] + offset_rows[None, None, :]
        cell_cols = vertex_cols[:, :, None] + offset_cols[None, None, :]
        cell_exists = (
            neighbor_exists[:, :, None]
            & (cell_rows >= 0)
            & (cell_cols >= 0)
            & (cell_rows < source_height - 1)
            & (cell_cols < source_width - 1)
        )
        cell_ids = np.where(
            cell_exists, cell_rows * (source_width - 1) + cell_cols, -1
        ).reshape(count, -1)
        cell_ids.sort(axis=1)
        keep = cell_ids >= 0
        keep[:, 1:] &= cell_ids[:, 1:] != cell_ids[:, :-1]

        pair_query, pair_slot = np.nonzero(keep)
        if pair_query.size:
            pair_cell = cell_ids[pair_query, pair_slot]
            cell_row = pair_cell // (source_width - 1)
            cell_col = pair_cell % (source_width - 1)
            quad_valid = (
                self.source.valid[cell_row, cell_col]
                & self.source.valid[cell_row, cell_col + 1]
                & self.source.valid[cell_row + 1, cell_col]
                & self.source.valid[cell_row + 1, cell_col + 1]
            )
            pair_query = pair_query[quad_valid]
            cell_row = cell_row[quad_valid]
            cell_col = cell_col[quad_valid]
        if pair_query.size:
            pair_points = query[pair_query]
            p00 = self._points_at(cell_row, cell_col)
            p10 = self._points_at(cell_row, cell_col + 1)
            p01 = self._points_at(cell_row + 1, cell_col)
            p11 = self._points_at(cell_row + 1, cell_col + 1)

            first_distance, first_bary = _closest_points_on_triangles(
                pair_points, p00, p10, p01
            )
            second_distance, second_bary = _closest_points_on_triangles(
                pair_points, p10, p11, p01
            )
            use_second = second_distance < first_distance
            pair_distance = np.where(use_second, second_distance, first_distance)
            pair_row = cell_row + np.where(
                use_second,
                second_bary[:, 1] + second_bary[:, 2],
                first_bary[:, 2],
            )
            pair_col = cell_col + np.where(
                use_second,
                second_bary[:, 0] + second_bary[:, 1],
                first_bary[:, 1],
            )

            order = np.lexsort((pair_distance, pair_query))
            sorted_queries = pair_query[order]
            first_of_query = np.flatnonzero(
                np.r_[True, sorted_queries[1:] != sorted_queries[:-1]]
            )
            selected = order[first_of_query]
            chosen = pair_query[selected]
            best_distance_sq[chosen] = pair_distance[selected]
            best_row[chosen] = pair_row[selected]
            best_col[chosen] = pair_col[selected]

        accepted = best_distance_sq <= max_distance * max_distance
        distances = np.sqrt(best_distance_sq)
        distances[~accepted] = np.inf
        return best_row, best_col, accepted, distances

    def build_target_uv_map(
        self,
        target: Surface,
        max_distance: float,
        query_batch_size: int = 65_536,
        progress: Optional[Callable[[int, int], None]] = None,
        workers: Optional[int] = None,
    ) -> Tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        BoolArray,
    ]:
        """Map target stored-grid vertices to source stored-grid coordinates.

        Batches run on ``workers`` threads (default: all cores); each batch
        writes a disjoint slice of the output fields, so results are
        identical to the serial order regardless of thread count.
        """

        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be positive")
        assert target.valid is not None
        target_valid_flat = np.flatnonzero(target.valid)
        target_height, target_width = target.shape
        mapped_rows = np.full(target.shape, np.nan, dtype=np.float32)
        mapped_cols = np.full(target.shape, np.nan, dtype=np.float32)
        mapped_distances = np.full(target.shape, np.inf, dtype=np.float32)
        mapped_valid = np.zeros(target.shape, dtype=bool)
        batch_count = max(
            1, math.ceil(target_valid_flat.size / query_batch_size)
        )
        worker_count = _resolve_worker_count(workers, batch_count)

        target_x_flat = target.x.ravel()
        target_y_flat = target.y.ravel()
        target_z_flat = target.z.ravel()
        mapped_rows_flat = mapped_rows.ravel()
        mapped_cols_flat = mapped_cols.ravel()
        mapped_distances_flat = mapped_distances.ravel()
        mapped_valid_flat = mapped_valid.ravel()

        def process_batch(start: int) -> None:
            indices = target_valid_flat[start : start + query_batch_size]
            points = np.column_stack(
                (
                    target_x_flat[indices],
                    target_y_flat[indices],
                    target_z_flat[indices],
                )
            )
            rows, cols, accepted, distances = self.locate(
                points,
                max_distance=max_distance,
                query_workers=1 if worker_count > 1 else -1,
            )
            accepted_indices = indices[accepted]
            mapped_rows_flat[accepted_indices] = rows[accepted].astype(
                np.float32
            )
            mapped_cols_flat[accepted_indices] = cols[accepted].astype(
                np.float32
            )
            mapped_distances_flat[accepted_indices] = distances[
                accepted
            ].astype(np.float32)
            mapped_valid_flat[accepted_indices] = True

        completed = 0

        def on_batch_done(_: None) -> None:
            nonlocal completed
            completed += 1
            if progress is not None:
                progress(completed, batch_count)

        _run_ordered(
            process_batch,
            range(0, target_valid_flat.size, query_batch_size),
            worker_count,
            on_result=on_batch_done,
        )

        # Keep shape references explicit for static checkers and catch accidental
        # ravel/indexing mistakes during future refactors.
        assert mapped_rows.shape == (target_height, target_width)
        return mapped_rows, mapped_cols, mapped_distances, mapped_valid

    def _points_at(
        self, rows: NDArray[np.int64], cols: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        return np.column_stack(
            (self.tx[rows, cols], self.ty[rows, cols], self.tz[rows, cols])
        )


def iter_tiles(
    shape: Tuple[int, int], tile_size: int
) -> Iterator[Tuple[int, int, int, int]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    height, width = shape
    for row_start in range(0, height, tile_size):
        row_end = min(height, row_start + tile_size)
        for col_start in range(0, width, tile_size):
            col_end = min(width, col_start + tile_size)
            yield row_start, row_end, col_start, col_end


def bilinear_field_tile(
    field: FloatArray,
    valid: BoolArray,
    output_shape: Tuple[int, int],
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> Tuple[NDArray[np.float64], BoolArray]:
    """Interpolate one stored-grid field at output pixel centres."""

    source_height, source_width = field.shape
    output_height, output_width = output_shape
    grid_y = (
        (np.arange(row_start, row_end, dtype=np.float64) + 0.5)
        * source_height
        / output_height
    )
    grid_x = (
        (np.arange(col_start, col_end, dtype=np.float64) + 0.5)
        * source_width
        / output_width
    )
    y0 = np.clip(np.floor(grid_y).astype(np.int64), 0, source_height - 1)
    x0 = np.clip(np.floor(grid_x).astype(np.int64), 0, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    wy = (grid_y - y0)[:, None]
    wx = (grid_x - x0)[None, :]

    v00 = valid[y0[:, None], x0[None, :]]
    v01 = valid[y0[:, None], x1[None, :]]
    v10 = valid[y1[:, None], x0[None, :]]
    v11 = valid[y1[:, None], x1[None, :]]
    interpolated_valid = v00 & v01 & v10 & v11
    f00 = np.where(v00, field[y0[:, None], x0[None, :]], 0.0)
    f01 = np.where(v01, field[y0[:, None], x1[None, :]], 0.0)
    f10 = np.where(v10, field[y1[:, None], x0[None, :]], 0.0)
    f11 = np.where(v11, field[y1[:, None], x1[None, :]], 0.0)
    top = f00 * (1.0 - wx) + f01 * wx
    bottom = f10 * (1.0 - wx) + f11 * wx
    result = top * (1.0 - wy) + bottom * wy
    interpolated_valid &= np.isfinite(result)
    return result, interpolated_valid


def _sample_mapper_surface_xyz(
    mapper: SurfaceMapper,
    rows: NDArray[np.float64],
    cols: NDArray[np.float64],
    valid: BoolArray,
) -> Tuple[NDArray[np.float64], BoolArray]:
    """Sample transformed source XYZ using the mapper's triangle split."""

    source_height, source_width = mapper.source.shape
    finite = valid & np.isfinite(rows) & np.isfinite(cols)
    in_bounds = (
        finite
        & (rows >= 0.0)
        & (rows <= source_height - 1)
        & (cols >= 0.0)
        & (cols <= source_width - 1)
    )
    safe_rows = np.clip(np.where(in_bounds, rows, 0.0), 0, source_height - 1)
    safe_cols = np.clip(np.where(in_bounds, cols, 0.0), 0, source_width - 1)
    row0 = np.minimum(
        np.floor(safe_rows).astype(np.int64), source_height - 2
    )
    col0 = np.minimum(
        np.floor(safe_cols).astype(np.int64), source_width - 2
    )
    row_fraction = safe_rows - row0
    col_fraction = safe_cols - col0

    assert mapper.source.valid is not None
    quad_valid = (
        in_bounds
        & mapper.source.valid[row0, col0]
        & mapper.source.valid[row0, col0 + 1]
        & mapper.source.valid[row0 + 1, col0]
        & mapper.source.valid[row0 + 1, col0 + 1]
    )
    xyz = np.full((*rows.shape, 3), np.nan, dtype=np.float64)
    for coordinate, field in enumerate((mapper.tx, mapper.ty, mapper.tz)):
        p00 = field[row0, col0]
        p10 = field[row0, col0 + 1]
        p01 = field[row0 + 1, col0]
        p11 = field[row0 + 1, col0 + 1]
        first_triangle = row_fraction + col_fraction <= 1.0
        first_value = (
            p00
            + col_fraction * (p10 - p00)
            + row_fraction * (p01 - p00)
        )
        second_value = (
            (1.0 - row_fraction) * p10
            + (row_fraction + col_fraction - 1.0) * p11
            + (1.0 - col_fraction) * p01
        )
        xyz[..., coordinate] = np.where(
            first_triangle, first_value, second_value
        )
    sampled_valid = quad_valid & np.all(np.isfinite(xyz), axis=-1)
    return xyz, sampled_valid


def _fill_uv_field(
    rows: NDArray[np.float32],
    cols: NDArray[np.float32],
    valid: BoolArray,
    smoothing_iterations: int = 64,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], BoolArray]:
    """Complete an incomplete target→source UV field across its gaps.

    Invalid vertices first take the value of their nearest valid vertex,
    then Jacobi relaxation on the invalid set only turns that
    piecewise-constant fill into a smooth continuation of the measured
    field. Valid vertices are never modified. Returns float64 fields and
    the mask of vertices that were filled.
    """

    import scipy.ndimage

    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        raise ValueError("cannot fill a UV field with no valid vertices")
    fill_mask = ~valid
    filled_rows = np.asarray(rows, dtype=np.float64).copy()
    filled_cols = np.asarray(cols, dtype=np.float64).copy()
    if not fill_mask.any():
        return filled_rows, filled_cols, fill_mask
    nearest = scipy.ndimage.distance_transform_edt(
        fill_mask, return_distances=False, return_indices=True
    )
    filled_rows = filled_rows[tuple(nearest)]
    filled_cols = filled_cols[tuple(nearest)]
    for _ in range(max(0, smoothing_iterations)):
        for field in (filled_rows, filled_cols):
            padded = np.pad(field, 1, mode="edge")
            neighbours = 0.25 * (
                padded[:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
            )
            field[fill_mask] = neighbours[fill_mask]
    return filled_rows, filled_cols, fill_mask


def transfer_array(
    source: Surface,
    target: Surface,
    source_label: NDArray,
    output_shape: Optional[Sequence[int]] = None,
    affine: Optional[NDArray[np.float64]] = None,
    max_distance: Optional[float] = None,
    nearest_vertices: int = 8,
    tile_size: int = 256,
    fill_value: int | float = 0,
    output: Optional[NDArray] = None,
    valid_output: Optional[NDArray[np.uint8]] = None,
    distance_output: Optional[NDArray[np.float32]] = None,
    query_batch_size: int = 65_536,
    progress: Optional[Callable[[int, int], None]] = None,
    source_validity: Optional[NDArray] = None,
    label_offset_yx: Tuple[float, float] = (0.0, 0.0),
    vertex_index: str = "kdtree",
    fill_seams: bool = False,
    workers: Optional[int] = None,
    uv_cache: Optional[str | Path] = None,
    additional_source_labels: Optional[Sequence[NDArray]] = None,
    additional_outputs: Optional[Sequence[NDArray]] = None,
    materialize_output: bool = True,
    rasterizer: str = "auto",
    tile_callback: Optional[
        Callable[
            [
                Tuple[int, int, int, int],
                Sequence[NDArray],
                NDArray[np.uint8],
            ],
            None,
        ]
    ] = None,
) -> Tuple[
    Optional[NDArray],
    Optional[NDArray[np.uint8]],
    Optional[NDArray[np.float32]],
    MappingStats,
]:
    """Transfer a complete 2D categorical label image between surfaces.

    Optional output arrays allow callers to provide disk-backed memmaps.

    ``workers`` threads the mapping batches and output tiles (default: all
    cores). Workers write disjoint regions and statistics are accumulated
    in submission order, so outputs and reports do not depend on it.

    ``uv_cache`` names an ``.npz`` file holding the mapped UV field, which
    depends only on the surface pair, affine, and matching parameters — not
    on the label. A matching cache skips the mapping phase entirely (labels
    of the same segment share it); a stale one is recomputed and rewritten.

    ``label_offset_yx`` declares, in label pixels, that label pixel ``(i, j)``
    depicts source-canvas position ``(i + dy, j + dx)`` instead of ``(i, j)``.
    This corrects a constant canvas offset between the raster the labels were
    drawn on and the source TIFXYZ canvas. Mapped pixels whose corrected
    label position falls outside the raster are marked invalid, not clamped.

    ``fill_seams`` additionally fills target pixels whose geometry mapping
    was rejected (fold seams, distance failures) by smoothly continuing the
    measured UV field across the gaps. Filled pixels are written with
    validity value 128 instead of 255 so downstream consumers can always
    tell measured from interpolated.

    If ``materialize_output`` is false, no full-resolution result arrays are
    allocated. Instead, ``tile_callback`` receives each completed tile in
    deterministic row-major order as ``(bounds, labels, validity)``. This is
    used by the CLI's compressed-TIFF streaming path.

    ``rasterizer`` selects the optional compiled per-pixel kernel. ``auto``
    uses it when built and otherwise preserves the NumPy implementation.
    """

    label = np.asarray(source_label)
    if label.ndim != 2:
        raise ValueError(f"source label must be 2D; got shape {label.shape}")
    extra_labels = [
        np.asarray(item) for item in (additional_source_labels or ())
    ]
    if any(item.shape != label.shape for item in extra_labels):
        raise ValueError("all source labels must have the same 2D shape")
    if additional_outputs is None:
        extra_outputs: list[NDArray] = []
    else:
        extra_outputs = list(additional_outputs)
    if materialize_output and len(extra_outputs) != len(extra_labels):
        raise ValueError(
            "additional_outputs must contain one array per additional label"
        )
    if not materialize_output:
        if tile_callback is None:
            raise ValueError(
                "tile_callback is required when materialize_output is false"
            )
        if output is not None or valid_output is not None or extra_outputs:
            raise ValueError(
                "full output arrays cannot be supplied when streaming tiles"
            )
        if distance_output is not None:
            raise ValueError(
                "distance output is not supported by the tile streaming path"
            )
    all_labels = [label] + extra_labels
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
    # Pixels the previous stage seam-filled (value 128) must stay 128 in
    # this stage's output even when the mapping here is measured:
    # interpolated provenance survives the whole chain.
    source_validity_values = (
        None
        if source_validity is None
        else np.asarray(source_validity, dtype=np.uint8)
    )
    resolved_shape = infer_output_shape(
        source, label.shape, target, explicit_shape=output_shape
    )
    if not materialize_output:
        output = None
    elif output is None:
        output = np.full(resolved_shape, fill_value, dtype=label.dtype)
    elif output.shape != resolved_shape or output.dtype != label.dtype:
        raise ValueError(
            f"output must have shape {resolved_shape} and dtype {label.dtype}; "
            f"got shape={output.shape}, dtype={output.dtype}"
        )
    for index, (extra_label, extra_output) in enumerate(
        zip(extra_labels, extra_outputs)
    ):
        if (
            extra_output.shape != resolved_shape
            or extra_output.dtype != extra_label.dtype
        ):
            raise ValueError(
                f"additional output {index} must have shape {resolved_shape} "
                f"and dtype {extra_label.dtype}; got "
                f"shape={extra_output.shape}, dtype={extra_output.dtype}"
            )
    if not materialize_output:
        valid_output = None
    elif valid_output is None:
        valid_output = np.zeros(resolved_shape, dtype=np.uint8)
    elif valid_output.shape != resolved_shape:
        raise ValueError("valid output shape does not match resolved output shape")
    if distance_output is not None and distance_output.shape != resolved_shape:
        raise ValueError("distance output shape does not match resolved output shape")

    effective_affine = (
        np.eye(4, dtype=np.float64) if affine is None else affine
    )
    if max_distance is None:
        source_spacing = estimate_surface_spacing(source, effective_affine)
        target_spacing = estimate_surface_spacing(target)
        max_distance = max(1e-3, 0.75 * min(source_spacing, target_spacing))
    if not math.isfinite(max_distance) or max_distance <= 0:
        raise ValueError(f"max_distance must be positive; got {max_distance}")

    mapper = SurfaceMapper(
        source,
        affine=effective_affine,
        nearest_vertices=nearest_vertices,
        vertex_index=vertex_index,
        index_max_distance=max_distance,
    )
    stats = MappingStats()
    source_height, source_width = source.shape
    label_height, label_width = label.shape

    tiles = list(iter_tiles(resolved_shape, tile_size))
    assert target.valid is not None
    target_vertex_count = int(target.valid.sum())
    mapping_batches = max(1, math.ceil(target_vertex_count / query_batch_size))
    progress_offset = 0

    def mapping_progress(complete: int, total: int) -> None:
        if progress is not None:
            progress(complete, mapping_batches + len(tiles))

    cache_meta = json.dumps(
        {
            "source_stored_shape": list(source.shape),
            "target_stored_shape": list(target.shape),
            "source_fingerprint": _surface_coordinate_fingerprint(
                source.x, source.y, source.z
            ),
            "target_fingerprint": _surface_coordinate_fingerprint(
                target.x, target.y, target.z
            ),
            "affine": np.asarray(effective_affine, dtype=np.float64)
            .ravel()
            .tolist(),
            "max_distance": float(max_distance),
            "nearest_vertices": int(nearest_vertices),
        },
        sort_keys=True,
    )
    uv_cache_path = None if uv_cache is None else Path(uv_cache)
    cached_uv = None
    if uv_cache_path is not None and uv_cache_path.exists():
        with np.load(uv_cache_path, allow_pickle=False) as data:
            if str(data["meta"]) == cache_meta:
                cached_uv = (
                    np.asarray(data["rows"], dtype=np.float32),
                    np.asarray(data["cols"], dtype=np.float32),
                    np.asarray(data["valid"], dtype=bool),
                )
            else:
                print(
                    f"UV cache {uv_cache_path} does not match this "
                    "surface pair / matching configuration; recomputing"
                )
    if cached_uv is not None:
        uv_rows, uv_cols, uv_valid = cached_uv
        mapping_progress(mapping_batches, mapping_batches)
    else:
        uv_rows, uv_cols, _, uv_valid = mapper.build_target_uv_map(
            target,
            max_distance=max_distance,
            query_batch_size=query_batch_size,
            progress=mapping_progress,
            workers=workers,
        )
        if uv_cache_path is not None:
            # Unique per writer: concurrent jobs sharing a cache path must
            # not clobber each other's partial file (os.replace stays the
            # atomic publish step; last writer wins with identical content).
            temporary = uv_cache_path.with_name(
                f"{uv_cache_path.name}.partial.{os.getpid()}"
            )
            with open(temporary, "wb") as handle:
                np.savez(
                    handle,
                    rows=uv_rows,
                    cols=uv_cols,
                    valid=uv_valid,
                    meta=cache_meta,
                )
            os.replace(temporary, uv_cache_path)
    progress_offset = mapping_batches

    filled_uv_rows: Optional[NDArray[np.float64]] = None
    filled_uv_cols: Optional[NDArray[np.float64]] = None
    filled_uv_valid: Optional[BoolArray] = None
    if fill_seams:
        filled_uv_rows, filled_uv_cols, _ = _fill_uv_field(
            uv_rows, uv_cols, uv_valid
        )
        filled_uv_valid = np.ones_like(uv_valid)

    from .native import NativeRasterizer, resolve_rasterizer

    selected_rasterizer = resolve_rasterizer(rasterizer)
    native_rasterizer = (
        None
        if selected_rasterizer == "python"
        else NativeRasterizer(
            target_fields=(target.x, target.y, target.z),
            target_valid=target.valid,
            uv_rows=uv_rows,
            uv_cols=uv_cols,
            uv_valid=uv_valid,
            source_fields=(mapper.tx, mapper.ty, mapper.tz),
            source_valid=mapper.source.valid,
            label_shape=label.shape,
            output_shape=resolved_shape,
            label_offset_yx=label_offset_yx,
            max_distance=max_distance,
            filled_uv_rows=filled_uv_rows,
            filled_uv_cols=filled_uv_cols,
            source_validity=source_validity_values,
        )
    )
    native_flat_labels = (
        None
        if native_rasterizer is None
        else [current_label.ravel() for current_label in all_labels]
    )

    def process_tile(
        tile: Tuple[int, int, int, int],
    ) -> Tuple[
        int,
        int,
        NDArray[np.float64],
        int,
        int,
        Tuple[int, int, int, int],
        Optional[list[NDArray]],
        Optional[NDArray[np.uint8]],
    ]:
        row_start, row_end, col_start, col_end = tile
        if native_rasterizer is not None:
            native = native_rasterizer.rasterize(tile)
            tile_shape = row_end - row_start, col_end - col_start
            if materialize_output:
                assert output is not None
                assert valid_output is not None
                tile_label = output[row_start:row_end, col_start:col_end]
                tile_labels = [tile_label] + [
                    item[row_start:row_end, col_start:col_end]
                    for item in extra_outputs
                ]
                tile_valid = valid_output[
                    row_start:row_end, col_start:col_end
                ]
            else:
                tile_labels = [
                    np.full(
                        tile_shape,
                        np.asarray(
                            fill_value, dtype=current_label.dtype
                        ).item(),
                        dtype=current_label.dtype,
                    )
                    for current_label in all_labels
                ]
                tile_valid = np.zeros(tile_shape, dtype=np.uint8)
            tile_valid[...] = native.validity
            sampled_rows, sampled_cols = np.nonzero(
                native.source_indices >= 0
            )
            sampled_indices = native.source_indices[
                sampled_rows, sampled_cols
            ]
            assert native_flat_labels is not None
            for flat_label, current_output in zip(
                native_flat_labels, tile_labels
            ):
                current_output[sampled_rows, sampled_cols] = (
                    flat_label[sampled_indices]
                )
            measured = np.isfinite(native.distances)
            if distance_output is not None:
                tile_distance = distance_output[
                    row_start:row_end, col_start:col_end
                ]
                tile_distance[measured] = native.distances[measured]
            return (
                native.validity.size,
                native.target_surface_valid,
                native.distances[measured],
                native.seam_filled_pixels,
                native.inherited_filled_pixels,
                tile,
                tile_labels if not materialize_output else None,
                tile_valid if not materialize_output else None,
            )

        target_x, target_pixel_valid = bilinear_field_tile(
            target.x,
            target.valid,
            resolved_shape,
            row_start,
            row_end,
            col_start,
            col_end,
        )
        target_y, target_y_valid = bilinear_field_tile(
            target.y,
            target.valid,
            resolved_shape,
            row_start,
            row_end,
            col_start,
            col_end,
        )
        target_z, target_z_valid = bilinear_field_tile(
            target.z,
            target.valid,
            resolved_shape,
            row_start,
            row_end,
            col_start,
            col_end,
        )
        target_pixel_valid &= target_y_valid & target_z_valid
        mapped_rows, rows_valid = bilinear_field_tile(
            uv_rows,
            uv_valid,
            resolved_shape,
            row_start,
            row_end,
            col_start,
            col_end,
        )
        mapped_cols, cols_valid = bilinear_field_tile(
            uv_cols,
            uv_valid,
            resolved_shape,
            row_start,
            row_end,
            col_start,
            col_end,
        )
        source_xyz, source_xyz_valid = _sample_mapper_surface_xyz(
            mapper,
            mapped_rows,
            mapped_cols,
            rows_valid & cols_valid,
        )
        target_xyz = np.stack((target_x, target_y, target_z), axis=-1)
        distances = np.linalg.norm(source_xyz - target_xyz, axis=-1)
        distance_valid = (
            source_xyz_valid
            & np.isfinite(distances)
            & (distances <= max_distance)
        )
        accepted = target_pixel_valid & distance_valid

        tile_shape = row_end - row_start, col_end - col_start
        if materialize_output:
            assert output is not None
            assert valid_output is not None
            tile_label = output[row_start:row_end, col_start:col_end]
            tile_labels = [tile_label] + [
                item[row_start:row_end, col_start:col_end]
                for item in extra_outputs
            ]
            tile_valid = valid_output[row_start:row_end, col_start:col_end]
        else:
            tile_labels = [
                np.full(
                    tile_shape,
                    np.asarray(fill_value, dtype=current_label.dtype).item(),
                    dtype=current_label.dtype,
                )
                for current_label in all_labels
            ]
            tile_valid = np.zeros(tile_shape, dtype=np.uint8)
        tile_distance = (
            None
            if distance_output is None
            else distance_output[row_start:row_end, col_start:col_end]
        )
        accepted_rows, accepted_cols = np.nonzero(accepted)
        inherited_count = 0
        if accepted_rows.size:
            label_rows = np.floor(
                mapped_rows[accepted_rows, accepted_cols]
                * label_height
                / source_height
                - label_offset_yx[0]
            ).astype(np.int64)
            label_cols = np.floor(
                mapped_cols[accepted_rows, accepted_cols]
                * label_width
                / source_width
                - label_offset_yx[1]
            ).astype(np.int64)
            in_range = (
                (label_rows >= 0)
                & (label_rows < label_height)
                & (label_cols >= 0)
                & (label_cols < label_width)
            )
            accepted_rows = accepted_rows[in_range]
            accepted_cols = accepted_cols[in_range]
            label_rows = label_rows[in_range]
            label_cols = label_cols[in_range]
            if propagated_validity is not None:
                source_is_valid = propagated_validity[
                    label_rows, label_cols
                ]
                accepted_rows = accepted_rows[source_is_valid]
                accepted_cols = accepted_cols[source_is_valid]
                label_rows = label_rows[source_is_valid]
                label_cols = label_cols[source_is_valid]
            for current_label, current_output in zip(
                all_labels, tile_labels
            ):
                current_output[accepted_rows, accepted_cols] = current_label[
                    label_rows, label_cols
                ]
            if source_validity_values is not None:
                sampled_validity = source_validity_values[
                    label_rows, label_cols
                ]
                inherited = sampled_validity == 128
                tile_valid[accepted_rows, accepted_cols] = np.where(
                    inherited, np.uint8(128), np.uint8(255)
                )
                inherited_count = int(inherited.sum())
            else:
                tile_valid[accepted_rows, accepted_cols] = 255
            if tile_distance is not None:
                tile_distance[accepted_rows, accepted_cols] = distances[
                    accepted_rows, accepted_cols
                ]

        seam_filled_count = 0
        if fill_seams:
            assert filled_uv_rows is not None
            assert filled_uv_cols is not None
            assert filled_uv_valid is not None
            seam_rows_grid, _ = bilinear_field_tile(
                filled_uv_rows,
                filled_uv_valid,
                resolved_shape,
                row_start,
                row_end,
                col_start,
                col_end,
            )
            seam_cols_grid, _ = bilinear_field_tile(
                filled_uv_cols,
                filled_uv_valid,
                resolved_shape,
                row_start,
                row_end,
                col_start,
                col_end,
            )
            seam_mask = target_pixel_valid & (tile_valid == 0)
            seam_pixel_rows, seam_pixel_cols = np.nonzero(seam_mask)
            if seam_pixel_rows.size:
                seam_label_rows = np.floor(
                    seam_rows_grid[seam_pixel_rows, seam_pixel_cols]
                    * label_height
                    / source_height
                    - label_offset_yx[0]
                ).astype(np.int64)
                seam_label_cols = np.floor(
                    seam_cols_grid[seam_pixel_rows, seam_pixel_cols]
                    * label_width
                    / source_width
                    - label_offset_yx[1]
                ).astype(np.int64)
                seam_in_range = (
                    (seam_label_rows >= 0)
                    & (seam_label_rows < label_height)
                    & (seam_label_cols >= 0)
                    & (seam_label_cols < label_width)
                )
                seam_pixel_rows = seam_pixel_rows[seam_in_range]
                seam_pixel_cols = seam_pixel_cols[seam_in_range]
                seam_label_rows = seam_label_rows[seam_in_range]
                seam_label_cols = seam_label_cols[seam_in_range]
                if propagated_validity is not None:
                    seam_source_valid = propagated_validity[
                        seam_label_rows, seam_label_cols
                    ]
                    seam_pixel_rows = seam_pixel_rows[seam_source_valid]
                    seam_pixel_cols = seam_pixel_cols[seam_source_valid]
                    seam_label_rows = seam_label_rows[seam_source_valid]
                    seam_label_cols = seam_label_cols[seam_source_valid]
                for current_label, current_output in zip(
                    all_labels, tile_labels
                ):
                    current_output[
                        seam_pixel_rows, seam_pixel_cols
                    ] = current_label[seam_label_rows, seam_label_cols]
                tile_valid[seam_pixel_rows, seam_pixel_cols] = 128
                seam_filled_count = int(seam_pixel_rows.size)

        return (
            target_pixel_valid.size,
            int(target_pixel_valid.sum()),
            distances[accepted_rows, accepted_cols],
            seam_filled_count,
            inherited_count,
            tile,
            tile_labels if not materialize_output else None,
            tile_valid if not materialize_output else None,
        )

    tiles_done = 0

    def on_tile_done(
        result: Tuple[
            int,
            int,
            NDArray[np.float64],
            int,
            int,
            Tuple[int, int, int, int],
            Optional[list[NDArray]],
            Optional[NDArray[np.uint8]],
        ],
    ) -> None:
        nonlocal tiles_done
        (
            target_pixels,
            target_surface_valid,
            tile_distances,
            seams,
            inherited,
            tile_bounds,
            streamed_labels,
            streamed_validity,
        ) = result
        stats.add(
            target_pixels=target_pixels,
            target_surface_valid=target_surface_valid,
            distances=tile_distances,
        )
        stats.seam_filled_pixels += seams
        stats.inherited_filled_pixels += inherited
        if tile_callback is not None:
            if streamed_labels is None or streamed_validity is None:
                raise RuntimeError("streaming tile payload was not produced")
            tile_callback(tile_bounds, streamed_labels, streamed_validity)
        tiles_done += 1
        if progress is not None:
            progress(
                progress_offset + tiles_done, mapping_batches + len(tiles)
            )

    _run_ordered(
        process_tile,
        tiles,
        _resolve_worker_count(workers, len(tiles)),
        on_result=on_tile_done,
    )
    return output, valid_output, distance_output, stats
