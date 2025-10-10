import json
from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np


def axis_perm(order: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Return the permutation for reordering axes and its inverse."""
    order = order.lower()
    mapping = {"x": 0, "y": 1, "z": 2}
    try:
        perm = tuple(mapping[c] for c in order)
    except KeyError as exc:
        raise ValueError(f"Invalid axis label '{exc.args[0]}' in axis order '{order}'.") from exc
    inv = tuple(np.argsort(perm))
    return perm, inv


def load_transform_from_json(json_path: Union[str, Path]) -> np.ndarray:
    """Load a 3x4/4x4 affine transform matrix from JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    if "transformation_matrix" not in data:
        raise ValueError("JSON file must contain 'transformation_matrix'.")

    matrix = np.asarray(data["transformation_matrix"], dtype=np.float64)

    if matrix.shape == (3, 4):
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        matrix4 = np.vstack([matrix, bottom])
    elif matrix.shape == (4, 4):
        matrix4 = matrix
        if not (np.allclose(matrix4[3, :3], 0.0) and np.isclose(matrix4[3, 3], 1.0)):
            raise ValueError("Bottom row of affine matrix must be [0, 0, 0, 1].")
    else:
        raise ValueError("Transformation matrix must have shape 3x4 or 4x4 (row-major).")

    return matrix4


def compute_inv_transpose(linear: np.ndarray) -> Union[np.ndarray, None]:
    """Compute (A^{-1})^T for the linear 3x3 part; returns None if singular."""
    try:
        return np.linalg.inv(linear).T
    except np.linalg.LinAlgError:
        return None


def _reshape_vec3(array: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Ensure array is 2D with shape (N, 3) and record if we need to squeeze."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError("Expected length-3 vector.")
        return arr.reshape(1, 3), True
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr, False
    raise ValueError("Expected array of shape (3,) or (N, 3).")


def _restore_shape(array: np.ndarray, squeeze: bool) -> np.ndarray:
    return array[0] if squeeze else array


def apply_affine_to_points(points: np.ndarray,
                           matrix4: np.ndarray,
                           perm: Iterable[int] = None) -> np.ndarray:
    """Apply a 4x4 affine transform to points, optionally in a permuted axis order."""
    pts, squeeze = _reshape_vec3(points)
    pts_ord = pts[:, perm] if perm is not None else pts
    ones = np.ones((pts_ord.shape[0], 1), dtype=np.float64)
    homog = np.concatenate([pts_ord, ones], axis=1)
    transformed_ord = (matrix4 @ homog.T).T[:, :3]
    if perm is not None:
        inv_perm = np.argsort(perm)
        transformed = transformed_ord[:, inv_perm]
    else:
        transformed = transformed_ord
    return _restore_shape(transformed, squeeze)


def transform_normals(normals: np.ndarray,
                      linear: np.ndarray,
                      perm: Iterable[int] = None,
                      inv_transpose: np.ndarray = None) -> np.ndarray:
    """Transform and renormalize normals using the affine linear part."""
    nrm, squeeze = _reshape_vec3(normals)
    nrm_ord = nrm[:, perm] if perm is not None else nrm

    inv_t = inv_transpose
    if inv_t is None:
        inv_t = compute_inv_transpose(linear)

    if inv_t is not None:
        transformed_ord = nrm_ord @ inv_t.T
    else:
        transformed_ord = nrm_ord @ linear.T

    lengths = np.linalg.norm(transformed_ord, axis=1)
    nonzero = lengths > 0
    if np.any(nonzero):
        transformed_ord[nonzero] /= lengths[nonzero][:, None]
    if np.any(~nonzero):
        transformed_ord[~nonzero] = 0.0

    if perm is not None:
        inv_perm = np.argsort(perm)
        transformed = transformed_ord[:, inv_perm]
    else:
        transformed = transformed_ord
    return _restore_shape(transformed, squeeze)
