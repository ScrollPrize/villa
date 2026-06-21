from __future__ import annotations

import math

import numpy as np

from vesuvius.neural_tracing.fiber_trace.labels import (
    IGNORE_INDEX,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
)


def normalize_vector(vec: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError(f"cannot normalize vector with norm {norm!r}")
    return (arr / norm).astype(np.float32, copy=False)


def normalize_vectors(vectors: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norms, eps, None)


def tangent_at_line_index(line_points_zyx: np.ndarray, index: int) -> np.ndarray:
    points = np.asarray(line_points_zyx, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError(
            f"line_points_zyx must have shape [N>=2, 3], got {points.shape!r}"
        )
    idx = int(np.clip(index, 0, points.shape[0] - 1))
    if idx == 0:
        tangent = points[1] - points[0]
    elif idx == points.shape[0] - 1:
        tangent = points[-1] - points[-2]
    else:
        tangent = points[idx + 1] - points[idx - 1]
    return normalize_vector(tangent)


def tangent_at_point(line_points_zyx: np.ndarray, point_zyx: np.ndarray) -> np.ndarray:
    points = np.asarray(line_points_zyx, dtype=np.float32)
    point = np.asarray(point_zyx, dtype=np.float32)
    idx = int(np.argmin(np.sum((points - point[None, :]) ** 2, axis=1)))
    return tangent_at_line_index(points, idx)


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    for _ in range(16):
        vec = rng.normal(size=3).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if np.isfinite(norm) and norm > 1e-6:
            return (vec / norm).astype(np.float32, copy=False)
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _orthonormal_basis_from_forward(fw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fw_norm = normalize_vector(fw)
    axis = np.zeros(3, dtype=np.float32)
    axis[int(np.argmin(np.abs(fw_norm)))] = 1.0
    u = axis - np.dot(axis, fw_norm) * fw_norm
    u = normalize_vector(u)
    v = np.cross(fw_norm, u).astype(np.float32, copy=False)
    v = normalize_vector(v)
    return u, v


def perturb_direction(
    direction: np.ndarray,
    *,
    max_angle_degrees: float,
    rng: np.random.Generator,
) -> np.ndarray:
    base = normalize_vector(direction)
    max_angle = max(0.0, float(max_angle_degrees)) * math.pi / 180.0
    if max_angle <= 1e-8:
        return base
    u, v = _orthonormal_basis_from_forward(base)
    angle = float(rng.uniform(0.0, max_angle))
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    offset = math.cos(phase) * u + math.sin(phase) * v
    return normalize_vector(math.cos(angle) * base + math.sin(angle) * offset)


def construct_up_vector(
    fw_zyx: np.ndarray, normal_zyx: np.ndarray | None = None
) -> np.ndarray:
    fw = normalize_vector(fw_zyx)
    if normal_zyx is not None:
        normal = np.asarray(normal_zyx, dtype=np.float32)
        up = normal - float(np.dot(normal, fw)) * fw
        norm = float(np.linalg.norm(up))
        if np.isfinite(norm) and norm > 1e-6:
            return (up / norm).astype(np.float32, copy=False)
    up, _ = _orthonormal_basis_from_forward(fw)
    return up


def construct_up_vectors(
    fw_zyx: np.ndarray, normal_zyx: np.ndarray | None = None
) -> np.ndarray:
    fw = normalize_vectors(np.asarray(fw_zyx, dtype=np.float32))
    flat_fw = fw.reshape(-1, 3)
    flat_normals = (
        None
        if normal_zyx is None
        else np.asarray(normal_zyx, dtype=np.float32).reshape(-1, 3)
    )
    out = np.empty_like(flat_fw, dtype=np.float32)
    for idx, vec in enumerate(flat_fw):
        normal = None if flat_normals is None else flat_normals[idx]
        out[idx] = construct_up_vector(vec, normal)
    return out.reshape(fw.shape)


def nearest_polyline_distance_and_tangent(
    coords_zyx: np.ndarray,
    line_points_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest segment distance and tangent for zyx coordinates.

    Args:
        coords_zyx: `[..., 3]` coordinate grid.
        line_points_zyx: `[N, 3]` fiber polyline.
    """
    coords = np.asarray(coords_zyx, dtype=np.float32)
    points = np.asarray(line_points_zyx, dtype=np.float32)
    if coords.shape[-1] != 3:
        raise ValueError(f"coords_zyx last dimension must be 3, got {coords.shape!r}")
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError(
            f"line_points_zyx must have shape [N>=2, 3], got {points.shape!r}"
        )

    flat = coords.reshape(-1, 3)
    best_dist_sq = np.full(flat.shape[0], np.inf, dtype=np.float32)
    best_tangent = np.zeros((flat.shape[0], 3), dtype=np.float32)

    for start, end in zip(points[:-1], points[1:], strict=True):
        seg = end - start
        seg_len_sq = float(np.dot(seg, seg))
        if not np.isfinite(seg_len_sq) or seg_len_sq <= 1e-12:
            continue
        tangent = (seg / math.sqrt(seg_len_sq)).astype(np.float32, copy=False)
        rel = flat - start[None, :]
        t = np.clip((rel @ seg) / seg_len_sq, 0.0, 1.0).astype(np.float32, copy=False)
        closest = start[None, :] + t[:, None] * seg[None, :]
        dist_sq = np.sum((flat - closest) ** 2, axis=1)
        update = dist_sq < best_dist_sq
        best_dist_sq[update] = dist_sq[update]
        best_tangent[update] = tangent

    if not bool(np.isfinite(best_dist_sq).any()):
        raise ValueError("fiber line has no non-degenerate segments")

    shape = coords.shape[:-1]
    dist = np.sqrt(best_dist_sq).reshape(shape).astype(np.float32, copy=False)
    tangent = best_tangent.reshape(*shape, 3).astype(np.float32, copy=False)
    return dist, tangent


def classify_voxels(
    *,
    crop_origin_zyx: np.ndarray,
    crop_shape: tuple[int, int, int],
    line_points_zyx: np.ndarray,
    cond_fw_zyx: np.ndarray,
    valid_mask: np.ndarray,
    positive_radius: float = 1.5,
    ignore_radius: float = 3.0,
    positive_cosine: float = 0.8660254037844386,
    negative_cosine: float = 0.25881904510252074,
) -> dict[str, np.ndarray]:
    crop_shape = tuple(int(v) for v in crop_shape)
    if len(crop_shape) != 3:
        raise ValueError(f"crop_shape must have 3 entries, got {crop_shape!r}")
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != crop_shape:
        raise ValueError(
            f"valid_mask shape {valid.shape!r} must match crop_shape {crop_shape!r}"
        )

    zz, yy, xx = np.meshgrid(
        np.arange(crop_shape[0], dtype=np.float32),
        np.arange(crop_shape[1], dtype=np.float32),
        np.arange(crop_shape[2], dtype=np.float32),
        indexing="ij",
    )
    origin = np.asarray(crop_origin_zyx, dtype=np.float32)
    coords = np.stack([zz + origin[0], yy + origin[1], xx + origin[2]], axis=-1)
    distance, tangent = nearest_polyline_distance_and_tangent(coords, line_points_zyx)
    tangent = normalize_vectors(tangent)
    cond_fw = normalize_vector(cond_fw_zyx)
    agreement = np.sum(tangent * cond_fw.reshape(1, 1, 1, 3), axis=-1)

    labels = np.full(crop_shape, IGNORE_INDEX, dtype=np.int64)
    near = distance <= float(positive_radius)
    far = distance > float(ignore_radius)
    aligned = agreement >= float(positive_cosine)
    disagreed = agreement <= float(negative_cosine)

    positive = valid & near & aligned
    negative = valid & (far | (near & disagreed))
    labels[negative] = NEGATIVE_LABEL
    labels[positive] = POSITIVE_LABEL

    target_fw = np.moveaxis(tangent, -1, 0).astype(np.float32, copy=False)
    target_up = construct_up_vectors(tangent).astype(np.float32, copy=False)
    target_up = np.moveaxis(target_up, -1, 0)

    return {
        "labels": labels,
        "target_fw": target_fw,
        "target_up": target_up,
        "distance": distance,
        "agreement": agreement.astype(np.float32, copy=False),
    }
