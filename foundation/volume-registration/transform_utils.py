import json
from pathlib import Path
from typing import Optional, NamedTuple
from urllib.parse import urljoin, urlparse

import jsonschema
import numpy as np
import requests
import SimpleITK as sitk
import zarr

############ TODO REMOVE AND FIX ON SERVER SIDE ############
import fsspec


# Monkey patch the HTTP filesystem to convert 500 errors to 404 for missing chunks
try:
    from fsspec.implementations.http import HTTPFileSystem

    original_raise_not_found = HTTPFileSystem._raise_not_found_for_status
except (ImportError, AttributeError):
    # Fallback for older fsspec versions
    original_raise_not_found = (
        fsspec.implementations.http.HTTPFileSystem._raise_not_found_for_status
    )


def patched_raise_not_found_for_status(self, response, url):
    """Convert 500 errors to 404 for missing Zarr chunks."""
    if response.status == 500 and "zarr" in url.lower():
        # Treat 500 as missing chunk (404) for Zarr URLs
        from aiohttp.client_exceptions import ClientResponseError

        raise ClientResponseError(
            request_info=response.request_info,
            history=response.history,
            status=404,  # Convert 500 to 404
            message="Not Found (converted from 500)",
            headers=response.headers,
        )
    else:
        return original_raise_not_found(self, response, url)


# Apply the patch
try:
    from fsspec.implementations.http import HTTPFileSystem

    HTTPFileSystem._raise_not_found_for_status = patched_raise_not_found_for_status
except (ImportError, AttributeError):
    # Fallback for older fsspec versions
    fsspec.implementations.http.HTTPFileSystem._raise_not_found_for_status = (
        patched_raise_not_found_for_status
    )
############ TODO REMOVE AND FIX ON SERVER SIDE ############


class Dimensions(NamedTuple):
    """Structure for volume dimensions with x, y, z coordinates and voxel size."""

    voxels_x: int
    voxels_y: int
    voxels_z: int
    voxel_size_um: float


def sanity_check_zarr_store(store):
    """Check if the store is a valid OME-ZARR file."""
    try:
        # Basic functionality checks
        assert store is not None
        assert hasattr(store, "keys")
        assert hasattr(store, "attrs")
    except AssertionError as e:
        raise ValueError(f"Invalid OME-ZARR file: {store.path}") from e


def get_volume_dimensions(
    path: str, provided_voxel_size: Optional[float] = None
) -> Dimensions:
    """Get volume dimensions (from Zarr array shape).

    Also get voxel size in microns (from metadata.json if it exists, otherwise use provided value).
    If both are provided, make sure they are the same.
    """

    # Get volume dimensions in voxels
    with zarr.open(path, mode="r") as store:
        sanity_check_zarr_store(store)
        voxels_z, voxels_y, voxels_x = store["0"].shape

    # Get voxel size from metadata.json if it exists
    # Check if path is a URL or local path
    parsed = urlparse(path)
    is_remote = parsed.scheme in ("http", "https")

    metadata = None
    metadata_path = None

    if is_remote:
        # Remote path
        metadata_url = urljoin(path + "/", "metadata.json")
        try:
            response = requests.get(metadata_url, timeout=10)
            if response.status_code == 200:
                metadata = response.json()
        except (
            requests.RequestException,
            json.JSONDecodeError,
        ) as e:
            raise RuntimeError(
                f"Could not fetch/parse metadata from {metadata_url}: {e}"
            )
        metadata_path = metadata_url
    else:
        # Local path
        metadata_path = Path(path) / "metadata.json"
        if metadata_path.exists():
            try:
                with metadata_path.open("r") as f:
                    metadata = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Could not parse metadata from {metadata_path}: {e}"
                )

    if metadata is not None:
        metadata_voxel_size_mm = (
            metadata.get("scan")
            .get("tomo")
            .get("acquisition")
            .get("detector")
            .get("samplePixelSize")
        )
        metadata_voxel_size_um = metadata_voxel_size_mm * 1000
        if provided_voxel_size is not None:
            assert (
                metadata_voxel_size_um == provided_voxel_size
            ), "Voxel size from metadata.json and provided voxel size do not match"
        return Dimensions(
            voxels_x=voxels_x,
            voxels_y=voxels_y,
            voxels_z=voxels_z,
            voxel_size_um=metadata_voxel_size_um,
        )
    else:
        assert (
            provided_voxel_size is not None
        ), f"No metadata.json found at {metadata_path} and no voxel size provided directly"
        return Dimensions(
            voxels_x=voxels_x,
            voxels_y=voxels_y,
            voxels_z=voxels_z,
            voxel_size_um=provided_voxel_size,
        )


def affine_matrix_to_sitk_transform(matrix: np.ndarray) -> sitk.AffineTransform:
    """Convert a 4x4 homogeneous transformation matrix to a SimpleITK AffineTransform.

    Args:
        matrix: 4x4 homogeneous transformation matrix (numpy array)

    Returns:
        SimpleITK AffineTransform object
    """
    # Ensure matrix is 4x4
    if matrix.shape != (4, 4):
        raise ValueError(f"Matrix must be 4x4, got shape {matrix.shape}")

    # Extract the 3x3 affine matrix and translation components
    affine_matrix = matrix[:3, :3].flatten().tolist()
    translation = matrix[:3, 3].tolist()

    # Create and configure the AffineTransform
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(affine_matrix)
    transform.SetTranslation(translation)

    return transform


def sitk_affine_transform_to_matrix(
    transform: sitk.AffineTransform,
) -> np.ndarray:
    """Convert a SimpleITK AffineTransform to a 4x4 homogeneous transformation matrix.

    Args:
        transform: SimpleITK AffineTransform object

    Returns:
        Inverted 4x4 homogeneous transformation matrix (numpy array)
    """
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(transform.GetMatrix()).reshape(3, 3)
    matrix[:3, 3] = transform.GetTranslation()
    return matrix


def invert_affine_matrix(matrix: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transformation matrix.

    Args:
        matrix: 4x4 homogeneous transformation matrix (numpy array)

    Returns:
        Inverted 4x4 homogeneous transformation matrix (numpy array)
    """
    return np.linalg.inv(matrix)


def matrix_swap_xyz_zyx(matrix: np.ndarray) -> np.ndarray:
    """
    Swap a 4x4 affine transform matrix between neuroglancer (ZYX order) and SITK (XYZ order).
    """
    # Swap the first and third rows/columns
    reorder_matrix = np.array(
        [
            [0, 0, 1, 0],  # z -> x
            [0, 1, 0, 0],  # y -> y
            [1, 0, 0, 0],  # x -> z
            [0, 0, 0, 1],  # homogeneous coordinate
        ]
    )

    # Apply the reordering
    output_matrix = reorder_matrix @ matrix @ reorder_matrix.T

    return output_matrix


def points_swap_xyz_zyx(points: list) -> list:
    """Swap points between neuroglancer (ZYX order) and SITK (XYZ order)."""
    return [[point[2], point[1], point[0]] for point in points]


def fit_affine_transform_from_points(fixed_points, moving_points):
    """Fit an affine transform from corresponding point pairs.

    Args:
        fixed_points: List of points in fixed space (Nx3)
        moving_points: List of points in moving space (Nx3)

    Returns:
        4x4 affine transformation matrix or None if insufficient points
    """
    if len(fixed_points) != len(moving_points) or len(fixed_points) < 4:
        return None

    # Convert to numpy arrays
    fixed_array = np.array(fixed_points)  # Points in fixed space
    moving_array = np.array(moving_points)  # Points in moving space

    # Add homogeneous coordinate to moving points
    moving_homogeneous = np.column_stack([moving_array, np.ones(len(moving_array))])

    # Solve for transform: fixed = transform @ moving_homogeneous.T
    # We want: transform @ moving_homogeneous.T = fixed_array.T
    # So: transform = fixed_array.T @ pinv(moving_homogeneous.T)
    transform_3x4 = fixed_array.T @ np.linalg.pinv(moving_homogeneous.T)

    # Add bottom row to make it 4x4
    transform_4x4 = np.vstack([transform_3x4, [0, 0, 0, 1]])

    return transform_4x4


def check_images_with_transform(
    fixed_image: sitk.Image, moving_image: sitk.Image, transform: sitk.Transform
) -> None:
    """
    Check overlap between two images with a transform between them.
    """
    # Resample moving image to fixed image space to check overlap
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_moving = resampler.Execute(moving_image)

    composite_image = sitk.Compose(
        fixed_image, resampled_moving, fixed_image // 2.0 + resampled_moving // 2.0
    )
    # Cast to uint8 for display
    composite_image = sitk.Cast(composite_image, sitk.sitkVectorUInt8)
    sitk.Show(composite_image, "Composite: Fixed (R), Resampled Moving (G)")

    # Check overlap by looking at non-zero regions
    fixed_array = sitk.GetArrayFromImage(fixed_image)
    moving_array = sitk.GetArrayFromImage(resampled_moving)

    fixed_nonzero = np.count_nonzero(fixed_array)
    moving_nonzero = np.count_nonzero(moving_array)

    # Simple overlap check: count voxels where both images have non-zero values
    overlap_mask = (fixed_array > 0) & (moving_array > 0)
    overlap_count = np.count_nonzero(overlap_mask)

    print(f"Fixed image non-zero voxels: {fixed_nonzero}")
    print(f"Moving image non-zero voxels: {moving_nonzero}")
    print(f"Overlap voxels: {overlap_count}")
    print(
        f"Overlap percentage: {overlap_count / max(fixed_nonzero, moving_nonzero) * 100:.2f}%"
    )

    if overlap_count == 0:
        raise ValueError(
            "No overlap detected! Images are too far apart for registration."
        )


def read_transform_json(input_path: str) -> tuple[np.ndarray, list, list]:
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

    return matrix, data["fixed_landmarks"], data["moving_landmarks"]


def write_transform_json(
    output_path: str,
    fixed_volume: str,
    matrix: np.ndarray,
    fixed_landmarks: list,
    moving_landmarks: list,
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
        "schema_version": "1.0.0",
        "fixed_volume": fixed_volume,
        "transformation_matrix": matrix_list,
        "fixed_landmarks": fixed_landmarks_list,
        "moving_landmarks": moving_landmarks_list,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
