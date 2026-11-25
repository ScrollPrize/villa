"""Transform files in the project's schema (`transform_schema.json`), and the XYZ/ZYX swaps,
from `transform_utils.py` without its volume and SimpleITK imports."""

import json
from pathlib import Path
from typing import Optional

import jsonschema
import numpy as np


def invert_affine_matrix(matrix: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transformation matrix.

    Args:
        matrix: 4x4 homogeneous transformation matrix (numpy array)

    Returns:
        Inverted 4x4 homogeneous transformation matrix (numpy array)
    """
    return np.linalg.inv(matrix)


def get_swap_matrix() -> np.ndarray:
    """Get the homogeneous matrix that swaps XYZ and ZYX coordinate orders."""
    return np.array(
        [
            [0, 0, 1, 0],  # z -> x
            [0, 1, 0, 0],  # y -> y
            [1, 0, 0, 0],  # x -> z
            [0, 0, 0, 1],  # homogeneous coordinate
        ]
    )


def matrix_swap_xyz_zyx(matrix: np.ndarray) -> np.ndarray:
    """
    Swap a 4x4 affine transform matrix between neuroglancer (ZYX order) and SITK (XYZ order).
    """
    reorder_matrix = get_swap_matrix()
    return reorder_matrix @ matrix @ reorder_matrix.T


def matrix_swap_output_xyz_zyx(matrix: np.ndarray) -> np.ndarray:
    """
    Swap only the output coordinate order of a 4x4 affine transform matrix.

    This converts a matrix between:
    - XYZ output space and ZYX output space
    while leaving the input coordinate order unchanged.
    """
    return get_swap_matrix() @ matrix


def points_swap_xyz_zyx(points: list) -> list:
    """Swap points between neuroglancer (ZYX order) and SITK (XYZ order)."""
    return [[point[2], point[1], point[0]] for point in points]




def read_transform_json(
    input_path: str, invert: bool = False
) -> tuple[np.ndarray, list, list]:
    """Read transform and landmarks from JSON file."""
    # Load the schema
    schema_path = Path(__file__).parent / "transform_schema.json"
    with open(schema_path, "r") as f:
        schema = json.load(f)

    # Load and validate the data
    with open(input_path, "r") as f:
        data = json.load(f)

    # Validate against schema
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"JSON file does not match schema: {e.message}")

    matrix = np.array(data["transformation_matrix"])
    matrix = np.vstack([matrix, [0, 0, 0, 1]])  # Add homogeneous row

    fixed_landmarks = data["fixed_landmarks"]
    moving_landmarks = data["moving_landmarks"]

    if invert:
        matrix = invert_affine_matrix(matrix)
        fixed_landmarks, moving_landmarks = moving_landmarks, fixed_landmarks

    return matrix, fixed_landmarks, moving_landmarks


def write_transform_json(
    output_path: str,
    fixed_volume: str,
    matrix: np.ndarray,
    fixed_landmarks: list,
    moving_landmarks: list,
    fit: Optional[dict] = None,
    deformation: Optional[dict] = None,
) -> None:
    """Write transform and landmarks to JSON file in the schema format."""
    # Convert numpy types to native Python types for JSON serialization
    matrix_list = matrix[:-1, :].tolist()  # Remove homogeneous row and convert to list
    fixed_landmarks_list = [
        [float(coord) for coord in point] for point in fixed_landmarks
    ]
    moving_landmarks_list = [
        [float(coord) for coord in point] for point in moving_landmarks
    ]

    data = {
        # Each version only adds an optional object -- 1.1.0 the "fit" record, 1.2.0 the
        # "deformation", 1.3.0 that deformation's "inverse" -- so a file stays readable by
        # anything that understood an earlier one, and a reader that ignores the
        # deformation gets precisely the transform it would have been given before.
        "schema_version": (
            "1.3.0" if deformation and deformation.get("inverse")
            else "1.2.0" if deformation
            else "1.1.0" if fit
            else "1.0.0"
        ),
        "fixed_volume": fixed_volume,
        "transformation_matrix": matrix_list,
        "fixed_landmarks": fixed_landmarks_list,
        "moving_landmarks": moving_landmarks_list,
    }
    if fit:
        data["fit"] = fit
    if deformation:
        data["deformation"] = deformation

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
