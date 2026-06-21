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


def tangent_at_line_index(line_points_xyz: np.ndarray, index: int) -> np.ndarray:
    points = np.asarray(line_points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError(
            f"line_points_xyz must have shape [N>=2, 3], got {points.shape!r}"
        )
    idx = int(np.clip(index, 0, points.shape[0] - 1))
    if idx == 0:
        tangent = points[1] - points[0]
    elif idx == points.shape[0] - 1:
        tangent = points[-1] - points[-2]
    else:
        tangent = points[idx + 1] - points[idx - 1]
    return normalize_vector(tangent)


def tangent_at_point(line_points_xyz: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(line_points_xyz, dtype=np.float32)
    point = np.asarray(point_xyz, dtype=np.float32)
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


def zyx_to_xyz(coords_zyx: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords_zyx)
    if arr.shape[-1] != 3:
        raise ValueError(f"coords_zyx last dimension must be 3, got {arr.shape!r}")
    return arr[..., (2, 1, 0)]


def xyz_to_zyx(coords_xyz: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords_xyz)
    if arr.shape[-1] != 3:
        raise ValueError(f"coords_xyz last dimension must be 3, got {arr.shape!r}")
    return arr[..., (2, 1, 0)]


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
    fw_xyz: np.ndarray,
    normal_xyz: np.ndarray | None = None,
    *,
    allow_arbitrary_up_fallback: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    fw = normalize_vector(fw_xyz, eps=eps)
    if normal_xyz is None:
        if allow_arbitrary_up_fallback:
            up, _ = _orthonormal_basis_from_forward(fw)
            return up
        raise ValueError(
            "normal_xyz is required for up-vector construction; set "
            "allow_arbitrary_up_fallback=True only for explicitly configured fallback"
        )

    normal = normalize_vector(np.asarray(normal_xyz, dtype=np.float32), eps=eps)
    up = normal - float(np.dot(normal, fw)) * fw
    norm = float(np.linalg.norm(up))
    if np.isfinite(norm) and norm > eps:
        return (up / norm).astype(np.float32, copy=False)
    if allow_arbitrary_up_fallback:
        up, _ = _orthonormal_basis_from_forward(fw)
        return up
    raise ValueError(
        "Lasagna normal projects to a degenerate up vector for the supplied fw_xyz"
    )


def construct_up_vectors(
    fw_xyz: np.ndarray,
    normal_xyz: np.ndarray | None = None,
    *,
    allow_arbitrary_up_fallback: bool = False,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    fw = normalize_vectors(np.asarray(fw_xyz, dtype=np.float32), eps=eps)
    flat_fw = fw.reshape(-1, 3)
    if normal_xyz is None:
        if not allow_arbitrary_up_fallback:
            raise ValueError(
                "normal_xyz is required for up-vector construction; set "
                "allow_arbitrary_up_fallback=True only for explicitly configured fallback"
            )
        flat_normals = None
    else:
        normals = normalize_vectors(np.asarray(normal_xyz, dtype=np.float32), eps=eps)
        if normals.shape != fw.shape:
            raise ValueError(
                f"normal_xyz shape {normals.shape!r} must match fw_xyz shape {fw.shape!r}"
            )
        flat_normals = normals.reshape(-1, 3)

    out = np.zeros_like(flat_fw, dtype=np.float32)
    valid = np.zeros((flat_fw.shape[0],), dtype=bool)
    for idx, vec in enumerate(flat_fw):
        normal = None if flat_normals is None else flat_normals[idx]
        try:
            out[idx] = construct_up_vector(
                vec,
                normal,
                allow_arbitrary_up_fallback=allow_arbitrary_up_fallback,
                eps=eps,
            )
            valid[idx] = True
        except ValueError:
            valid[idx] = False
    return out.reshape(fw.shape), valid.reshape(fw.shape[:-1])


def decode_lasagna_normals_xyz(
    nx_encoded: np.ndarray,
    ny_encoded: np.ndarray,
    *,
    invalid_tolerance: float = 1e-4,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode Lasagna uint8 hemisphere normals into xyz vectors.

    Lasagna stores only `nx` and `ny` as `(value - 128) / 127`; `nz` is the
    non-negative hemisphere component.
    """
    nx_raw = np.asarray(nx_encoded)
    ny_raw = np.asarray(ny_encoded)
    if nx_raw.shape != ny_raw.shape:
        raise ValueError(
            f"nx/ny normal shapes must match, got {nx_raw.shape!r} vs {ny_raw.shape!r}"
        )
    nx = (nx_raw.astype(np.float32, copy=False) - 128.0) / 127.0
    ny = (ny_raw.astype(np.float32, copy=False) - 128.0) / 127.0
    nz_sq = 1.0 - nx * nx - ny * ny
    valid = np.isfinite(nx) & np.isfinite(ny) & (nz_sq >= -float(invalid_tolerance))
    nz = np.sqrt(np.maximum(nz_sq, 0.0)).astype(np.float32, copy=False)
    normals = np.stack([nx, ny, nz], axis=-1).astype(np.float32, copy=False)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid &= np.isfinite(norms[..., 0]) & (norms[..., 0] > eps)
    normals = np.divide(
        normals,
        np.maximum(norms, eps),
        out=np.zeros_like(normals, dtype=np.float32),
    )
    normals[~valid] = 0.0
    return normals.astype(np.float32, copy=False), valid.astype(bool, copy=False)


def nearest_polyline_distance_and_tangent(
    coords_xyz: np.ndarray,
    line_points_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest segment distance and tangent for xyz coordinates.

    Args:
        coords_xyz: `[..., 3]` coordinate grid.
        line_points_xyz: `[N, 3]` fiber polyline.
    """
    coords = np.asarray(coords_xyz, dtype=np.float32)
    points = np.asarray(line_points_xyz, dtype=np.float32)
    if coords.shape[-1] != 3:
        raise ValueError(f"coords_xyz last dimension must be 3, got {coords.shape!r}")
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError(
            f"line_points_xyz must have shape [N>=2, 3], got {points.shape!r}"
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
    line_points_xyz: np.ndarray,
    cond_fw_xyz: np.ndarray,
    valid_mask: np.ndarray,
    normal_xyz: np.ndarray | None = None,
    normal_valid_mask: np.ndarray | None = None,
    allow_arbitrary_up_fallback: bool = False,
    degenerate_up_policy: str = "invalid",
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
    if normal_valid_mask is not None:
        normal_valid = np.asarray(normal_valid_mask, dtype=bool)
        if normal_valid.shape != crop_shape:
            raise ValueError(
                f"normal_valid_mask shape {normal_valid.shape!r} must match crop_shape {crop_shape!r}"
            )
        valid = valid & normal_valid
    if not bool(valid.any()):
        raise ValueError("valid_mask contains no valid voxels after normal validation")

    zz, yy, xx = np.meshgrid(
        np.arange(crop_shape[0], dtype=np.float32),
        np.arange(crop_shape[1], dtype=np.float32),
        np.arange(crop_shape[2], dtype=np.float32),
        indexing="ij",
    )
    origin = np.asarray(crop_origin_zyx, dtype=np.float32)
    coords_xyz = np.stack(
        [xx + origin[2], yy + origin[1], zz + origin[0]],
        axis=-1,
    )
    distance, tangent_xyz = nearest_polyline_distance_and_tangent(
        coords_xyz, line_points_xyz
    )
    tangent_xyz = normalize_vectors(tangent_xyz)
    cond_fw = normalize_vector(cond_fw_xyz)
    agreement = np.sum(tangent_xyz * cond_fw.reshape(1, 1, 1, 3), axis=-1)

    labels = np.full(crop_shape, IGNORE_INDEX, dtype=np.int64)
    near = distance <= float(positive_radius)
    far = distance > float(ignore_radius)
    aligned = agreement >= float(positive_cosine)
    disagreed = agreement <= float(negative_cosine)

    positive = valid & near & aligned
    negative = valid & (far | (near & disagreed))
    labels[negative] = NEGATIVE_LABEL
    labels[positive] = POSITIVE_LABEL

    up_policy = str(degenerate_up_policy)
    if up_policy not in {"invalid", "raise"}:
        raise ValueError(
            "degenerate_up_policy must be 'invalid' or 'raise', "
            f"got {degenerate_up_policy!r}"
        )
    target_up_xyz, target_up_valid = construct_up_vectors(
        tangent_xyz,
        normal_xyz,
        allow_arbitrary_up_fallback=allow_arbitrary_up_fallback,
    )
    target_up_valid = target_up_valid & valid
    if up_policy == "raise" and bool((positive & ~target_up_valid).any()):
        raise ValueError("positive voxels contain degenerate Lasagna normal up vectors")

    target_fw_xyz = np.moveaxis(tangent_xyz, -1, 0).astype(np.float32, copy=False)
    target_up_xyz = np.moveaxis(target_up_xyz, -1, 0).astype(np.float32, copy=False)

    return {
        "labels": labels,
        "target_fw_xyz": target_fw_xyz,
        "target_up_xyz": target_up_xyz,
        "target_up_valid": target_up_valid.astype(bool, copy=False),
        "distance": distance,
        "agreement": agreement.astype(np.float32, copy=False),
    }
