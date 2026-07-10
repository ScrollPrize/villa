"""Affine transforms and cross-frame resampling for zarr volumes.

The ``transform.json`` schema used by the Vesuvius registration tooling
(schema version 1.0.0) stores a homogeneous 3x4 matrix in XYZ order such
that ``p_fixed = M @ p_moving``. Zarr arrays in this codebase are indexed
in ZYX voxel order, so helpers here expose a ZYX form of the matrix and
apply it to patches fetched from zarr arrays.

See ``foundation/volume-registration/transform_schema.json`` for the
schema; this module intentionally avoids ``jsonschema`` so we don't pull
a new dependency for a validated read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

from vesuvius.data.utils import open_zarr


SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class TransformDocument:
    """Parsed contents of a ``transform.json`` file."""

    matrix_xyz: np.ndarray  # 4x4 homogeneous, XYZ order
    fixed_landmarks: List[List[float]]
    moving_landmarks: List[List[float]]
    fixed_volume: str


def _read_bytes(path_or_url: str) -> bytes:
    """Read bytes from a local path, ``http(s)://`` URL, or ``s3://`` URI."""
    text = str(path_or_url)
    if text.startswith(("http://", "https://", "s3://")):
        import fsspec

        storage_options: Dict[str, Any] = {}
        if text.startswith("s3://"):
            storage_options["anon"] = True
        with fsspec.open(text, mode="rb", **storage_options) as f:
            return f.read()
    return Path(text).read_bytes()


def read_transform_json(path_or_url: str) -> TransformDocument:
    """Load and lightly validate a ``transform.json`` at ``path_or_url``.

    Accepts local paths, ``http(s)://`` URLs, and ``s3://`` URIs (anonymous).
    Returns the matrix as a 4x4 homogeneous numpy array in XYZ order.
    """
    raw = _read_bytes(path_or_url)
    data = json.loads(raw.decode("utf-8"))

    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported transform.json schema_version {version!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )

    for key in ("fixed_volume", "transformation_matrix", "fixed_landmarks", "moving_landmarks"):
        if key not in data:
            raise ValueError(f"transform.json missing required key '{key}'")

    matrix_3x4 = np.asarray(data["transformation_matrix"], dtype=np.float64)
    if matrix_3x4.shape != (3, 4):
        raise ValueError(
            f"transform.json transformation_matrix must be 3x4, got shape {matrix_3x4.shape}"
        )

    matrix_4x4 = np.vstack([matrix_3x4, [0.0, 0.0, 0.0, 1.0]])

    return TransformDocument(
        matrix_xyz=matrix_4x4,
        fixed_landmarks=[list(map(float, p)) for p in data["fixed_landmarks"]],
        moving_landmarks=[list(map(float, p)) for p in data["moving_landmarks"]],
        fixed_volume=str(data["fixed_volume"]),
    )


def invert_affine_matrix(matrix: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous affine transform."""
    if matrix.shape != (4, 4):
        raise ValueError(f"matrix must be 4x4, got {matrix.shape}")
    return np.linalg.inv(matrix)


_SWAP = np.array(
    [
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)


def get_swap_matrix() -> np.ndarray:
    """Homogeneous matrix that swaps XYZ and ZYX coordinate orders."""
    return _SWAP.copy()


def matrix_swap_xyz_zyx(matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 affine between XYZ and ZYX coordinate orders.

    The swap is self-inverse: ``matrix_swap_xyz_zyx(matrix_swap_xyz_zyx(M)) == M``.
    """
    if matrix.shape != (4, 4):
        raise ValueError(f"matrix must be 4x4, got {matrix.shape}")
    return _SWAP @ matrix @ _SWAP.T


def label_to_image_zyx_matrix(matrix_xyz: np.ndarray, invert: bool = True) -> np.ndarray:
    """Compute the ZYX-ordered affine from label coords to image coords.

    The ``transform.json`` matrix is XYZ and maps ``p_fixed = M @ p_moving``.
    In our setting the labels volume lives in the fixed frame and the image
    volume is the moving frame, so ``image_xyz = inv(M) @ label_xyz``.
    Setting ``invert=False`` skips the inversion (useful for tests).
    """
    m = matrix_xyz if not invert else invert_affine_matrix(matrix_xyz)
    return matrix_swap_xyz_zyx(m)


def image_to_label_zyx_matrix(matrix_xyz: np.ndarray) -> np.ndarray:
    """ZYX-ordered affine from image coords to label coords.

    With ``p_fixed = M @ p_moving`` and labels=fixed, image=moving in our
    pipeline, the forward direction ``label_xyz = M @ image_xyz`` maps each
    image voxel to its corresponding label voxel.
    """
    return matrix_swap_xyz_zyx(matrix_xyz)


def image_patch_label_aabb(
    matrix_image_to_label_zyx: np.ndarray,
    position_image_zyx: Tuple[int, int, int],
    patch_shape_zyx: Tuple[int, int, int],
    label_shape_zyx: Optional[Tuple[int, int, int]] = None,
    margin: int = 1,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Compute the label-volume AABB enclosing the forward-mapped image patch.

    Mirror of ``label_patch_image_aabb`` but in the image→label direction.
    """
    return label_patch_image_aabb(
        matrix_image_to_label_zyx,
        position_image_zyx,
        patch_shape_zyx,
        image_shape_zyx=label_shape_zyx,
        margin=margin,
    )


def resample_label_to_image_grid(
    labels_array,
    matrix_image_to_label_zyx: np.ndarray,
    position_image_zyx: Tuple[int, int, int],
    patch_shape_zyx: Tuple[int, int, int],
    *,
    order: int = 0,
    cval: float = 0.0,
) -> np.ndarray:
    """Resample a patch of ``labels_array`` onto the image patch grid.

    Default ``order=0`` (nearest neighbor) preserves binary label values.
    The label slab fetched from disk is bounded by the forward-mapped AABB
    of the image patch (much smaller than the image patch itself when the
    image voxel size is finer than the label voxel size, which is the
    typical fibers setting).
    """
    label_shape = tuple(labels_array.shape[-3:])
    start, stop = image_patch_label_aabb(
        matrix_image_to_label_zyx,
        position_image_zyx,
        patch_shape_zyx,
        label_shape_zyx=label_shape,
    )

    if any(st >= sp for st, sp in zip(start, stop)):
        return np.full(patch_shape_zyx, cval, dtype=np.float32)

    slab = np.asarray(
        labels_array[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]],
        dtype=np.float32,
    )

    dz, dy, dx = patch_shape_zyx
    zs = np.arange(dz, dtype=np.float64)
    ys = np.arange(dy, dtype=np.float64)
    xs = np.arange(dx, dtype=np.float64)
    gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")
    grid = np.stack(
        [
            gz + position_image_zyx[0],
            gy + position_image_zyx[1],
            gx + position_image_zyx[2],
        ],
        axis=0,
    ).reshape(3, -1)

    mapped = apply_affine_zyx(matrix_image_to_label_zyx, grid.T).T
    mapped -= np.array(start, dtype=np.float64).reshape(3, 1)

    resampled = ndimage.map_coordinates(
        slab,
        mapped,
        order=order,
        mode="constant",
        cval=cval,
        prefilter=False,
    )
    return resampled.reshape(patch_shape_zyx).astype(np.float32)


def apply_affine_zyx(matrix_zyx: np.ndarray, points_zyx: np.ndarray) -> np.ndarray:
    """Apply a 4x4 ZYX affine to an ``(N, 3)`` array of ZYX points."""
    if matrix_zyx.shape != (4, 4):
        raise ValueError(f"matrix must be 4x4, got {matrix_zyx.shape}")
    pts = np.asarray(points_zyx, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (matrix_zyx @ homogeneous.T).T[:, :3]


def label_patch_image_aabb(
    matrix_label_to_image_zyx: np.ndarray,
    position_label_zyx: Tuple[int, int, int],
    patch_shape_zyx: Tuple[int, int, int],
    image_shape_zyx: Optional[Tuple[int, int, int]] = None,
    margin: int = 1,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Compute the image-volume AABB enclosing the mapped label patch.

    Returns ``(start_zyx, stop_zyx)`` integer voxel bounds, padded by
    ``margin`` voxels on each side for trilinear interpolation, clipped to
    ``[0, image_shape_zyx)`` if provided. ``stop_zyx`` is exclusive.
    """
    pz, py, px = position_label_zyx
    dz, dy, dx = patch_shape_zyx
    corners = np.array(
        [
            [pz, py, px],
            [pz + dz, py, px],
            [pz, py + dy, px],
            [pz, py, px + dx],
            [pz + dz, py + dy, px],
            [pz + dz, py, px + dx],
            [pz, py + dy, px + dx],
            [pz + dz, py + dy, px + dx],
        ],
        dtype=np.float64,
    )

    mapped = apply_affine_zyx(matrix_label_to_image_zyx, corners)
    lo = np.floor(mapped.min(axis=0)).astype(np.int64) - margin
    hi = np.ceil(mapped.max(axis=0)).astype(np.int64) + margin

    if image_shape_zyx is not None:
        shape_arr = np.asarray(image_shape_zyx, dtype=np.int64)
        lo = np.clip(lo, 0, shape_arr)
        hi = np.clip(hi, 0, shape_arr)

    return tuple(int(v) for v in lo), tuple(int(v) for v in hi)  # type: ignore[return-value]


def resample_image_to_label_grid(
    image_array,
    matrix_label_to_image_zyx: np.ndarray,
    position_label_zyx: Tuple[int, int, int],
    patch_shape_zyx: Tuple[int, int, int],
    *,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """Resample a patch of ``image_array`` onto the label patch grid.

    The image patch returned has shape ``patch_shape_zyx`` and matches the
    label patch at label-voxel position ``position_label_zyx`` after the
    affine transform. ``image_array`` can be a zarr array or anything that
    supports numpy-style slicing (``arr[start:stop, ...]``).
    """
    image_shape = tuple(image_array.shape[-3:])
    start, stop = label_patch_image_aabb(
        matrix_label_to_image_zyx,
        position_label_zyx,
        patch_shape_zyx,
        image_shape_zyx=image_shape,
    )

    if any(st >= sp for st, sp in zip(start, stop)):
        return np.full(patch_shape_zyx, cval, dtype=np.float32)

    slab = np.asarray(
        image_array[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]],
        dtype=np.float32,
    )

    dz, dy, dx = patch_shape_zyx
    zs = np.arange(dz, dtype=np.float64)
    ys = np.arange(dy, dtype=np.float64)
    xs = np.arange(dx, dtype=np.float64)
    grid_z, grid_y, grid_x = np.meshgrid(zs, ys, xs, indexing="ij")
    grid = np.stack(
        [
            grid_z + position_label_zyx[0],
            grid_y + position_label_zyx[1],
            grid_x + position_label_zyx[2],
        ],
        axis=0,
    ).reshape(3, -1)

    mapped = apply_affine_zyx(matrix_label_to_image_zyx, grid.T).T
    mapped -= np.array(start, dtype=np.float64).reshape(3, 1)

    resampled = ndimage.map_coordinates(
        slab,
        mapped,
        order=order,
        mode="constant",
        cval=cval,
        prefilter=False,
    )
    return resampled.reshape(patch_shape_zyx).astype(np.float32)


def matrix_checksum(matrix: np.ndarray) -> str:
    """Short hex digest of a transform matrix, useful as a cache key."""
    import hashlib

    return hashlib.sha1(np.ascontiguousarray(matrix, dtype=np.float64).tobytes()).hexdigest()[:16]


__all__ = [
    "SCHEMA_VERSION",
    "TransformDocument",
    "read_transform_json",
    "invert_affine_matrix",
    "get_swap_matrix",
    "matrix_swap_xyz_zyx",
    "label_to_image_zyx_matrix",
    "image_to_label_zyx_matrix",
    "apply_affine_zyx",
    "label_patch_image_aabb",
    "image_patch_label_aabb",
    "resample_image_to_label_grid",
    "resample_label_to_image_grid",
    "matrix_checksum",
]


# Keep ``open_zarr`` reachable for callers that want a consistent I/O path.
_open_zarr = open_zarr
