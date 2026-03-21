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


class MeshInfo(NamedTuple):
    """Basic mesh metadata in native XYZ source coordinates."""

    vertex_count: int
    polygon_count: int
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]

    @property
    def center_xyz(self) -> tuple[float, float, float]:
        return tuple(
            (self.bounds_min_xyz[i] + self.bounds_max_xyz[i]) / 2.0 for i in range(3)
        )


class MeshGeometry(NamedTuple):
    """Mesh geometry in native XYZ source coordinates."""

    vertices_xyz: np.ndarray
    faces: np.ndarray


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
    metadata_error: Optional[Exception] = None

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
            metadata_error = e
        metadata_path = metadata_url
    else:
        # Local path
        metadata_path = Path(path) / "metadata.json"
        if metadata_path.exists():
            try:
                with metadata_path.open("r") as f:
                    metadata = json.load(f)
            except json.JSONDecodeError as e:
                metadata_error = e

    if metadata_error is not None and provided_voxel_size is None:
        raise RuntimeError(
            f"Could not fetch/parse metadata from {metadata_path}: {metadata_error}"
        )

    metadata_voxel_size_um = None
    if isinstance(metadata, dict):
        scan = metadata.get("scan") or {}
        tomo = scan.get("tomo") or {}
        acquisition = tomo.get("acquisition") or {}
        detector = acquisition.get("detector") or {}
        metadata_voxel_size_mm = detector.get("samplePixelSize")
        if metadata_voxel_size_mm is not None:
            try:
                metadata_voxel_size_um = float(metadata_voxel_size_mm) * 1000
            except (TypeError, ValueError):
                metadata_voxel_size_um = None

    if metadata_voxel_size_um is not None:
        if provided_voxel_size is not None and not np.isclose(
            metadata_voxel_size_um, provided_voxel_size
        ):
            raise ValueError(
                "Voxel size from metadata.json and provided voxel size do not match: "
                f"{metadata_voxel_size_um} != {provided_voxel_size} (microns)"
            )
        voxel_size_um = metadata_voxel_size_um
    else:
        if provided_voxel_size is None:
            if metadata is not None:
                raise ValueError(
                    f"metadata.json found at {metadata_path} but voxel size was not found at "
                    "'scan.tomo.acquisition.detector.samplePixelSize'. "
                    "Provide voxel size directly with --fixed-voxel-size/--moving-voxel-size."
                )
            raise ValueError(
                f"No metadata.json found at {metadata_path} and no voxel size provided directly"
            )
        voxel_size_um = provided_voxel_size

    return Dimensions(
        voxels_x=voxels_x,
        voxels_y=voxels_y,
        voxels_z=voxels_z,
        voxel_size_um=voxel_size_um,
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


def strip_vtk_url_prefix(path: str) -> str:
    """Strip the vtk:// prefix from a Neuroglancer VTK mesh URL if present."""
    if path.startswith("vtk://"):
        return path[len("vtk://") :]
    return path


def _iter_text_lines(path: str):
    """Yield lines from a local path or remote URL as Unicode strings."""
    parsed = urlparse(path)
    if parsed.scheme in ("http", "https"):
        with requests.get(path, stream=True, timeout=30) as response:
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            for line in response.iter_lines(decode_unicode=True):
                if line is not None:
                    yield line
        return

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")


def _read_json_if_exists(path: str) -> Optional[dict]:
    """Read a local or remote JSON document if it exists, otherwise return None."""
    parsed = urlparse(path)
    if parsed.scheme in ("http", "https"):
        try:
            response = requests.get(path, timeout=10)
            if response.status_code != 200:
                return None
            return response.json()
        except (requests.RequestException, json.JSONDecodeError):
            return None

    json_path = Path(path)
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mesh_info_from_sidecar(path: str) -> Optional[MeshInfo]:
    data = _read_json_if_exists(path)
    if data is None:
        return None

    try:
        return MeshInfo(
            vertex_count=int(data["vertex_count"]),
            polygon_count=int(data["polygon_count"]),
            bounds_min_xyz=tuple(float(v) for v in data["bounds_min_xyz"]),
            bounds_max_xyz=tuple(float(v) for v in data["bounds_max_xyz"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid mesh sidecar metadata at {path}: {e}") from e


def get_vtk_mesh_info(path: str) -> MeshInfo:
    """Get mesh bounds and counts from an ASCII legacy VTK POLYDATA file."""
    mesh_path = strip_vtk_url_prefix(path)
    sidecar = _mesh_info_from_sidecar(mesh_path + ".json")
    if sidecar is not None:
        return sidecar

    point_count = None
    polygon_count = 0
    mins = [np.inf, np.inf, np.inf]
    maxs = [-np.inf, -np.inf, -np.inf]
    remaining_coords = 0
    buffered_numbers: list[float] = []

    for raw_line in _iter_text_lines(mesh_path):
        line = raw_line.strip()
        if not line:
            continue

        if remaining_coords > 0:
            buffered_numbers.extend(float(value) for value in line.split())
            while remaining_coords > 0 and len(buffered_numbers) >= 3:
                x, y, z = buffered_numbers[:3]
                del buffered_numbers[:3]
                mins[0] = min(mins[0], x)
                mins[1] = min(mins[1], y)
                mins[2] = min(mins[2], z)
                maxs[0] = max(maxs[0], x)
                maxs[1] = max(maxs[1], y)
                maxs[2] = max(maxs[2], z)
                remaining_coords -= 3
            continue

        upper = line.upper()
        if upper.startswith("POINTS "):
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed POINTS line in {mesh_path}: {line}")
            point_count = int(parts[1])
            remaining_coords = point_count * 3
            continue

        if upper.startswith("POLYGONS "):
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed POLYGONS line in {mesh_path}: {line}")
            polygon_count = int(parts[1])

    if point_count is None:
        raise ValueError(f"No POINTS section found in {mesh_path}")
    if remaining_coords != 0:
        raise ValueError(f"Incomplete POINTS section in {mesh_path}")

    return MeshInfo(
        vertex_count=point_count,
        polygon_count=polygon_count,
        bounds_min_xyz=tuple(float(v) for v in mins),
        bounds_max_xyz=tuple(float(v) for v in maxs),
    )


def read_vtk_mesh_geometry(path: str) -> MeshGeometry:
    """Read geometry from an ASCII legacy VTK POLYDATA mesh."""
    mesh_path = strip_vtk_url_prefix(path)
    point_count = None
    polygon_count = None
    remaining_coords = 0
    remaining_polygon_ints = 0
    buffered_numbers: list[float] = []
    buffered_ints: list[int] = []
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    for raw_line in _iter_text_lines(mesh_path):
        line = raw_line.strip()
        if not line:
            continue

        if remaining_coords > 0:
            buffered_numbers.extend(float(value) for value in line.split())
            while remaining_coords > 0 and len(buffered_numbers) >= 3:
                vertices.append(buffered_numbers[:3])
                del buffered_numbers[:3]
                remaining_coords -= 3
            continue

        if remaining_polygon_ints > 0:
            buffered_ints.extend(int(value) for value in line.split())
            while remaining_polygon_ints > 0 and buffered_ints:
                if len(buffered_ints) < 4:
                    break
                size = buffered_ints[0]
                if size != 3:
                    raise ValueError(
                        f"Only triangular POLYGONS are supported in {mesh_path}, found polygon size {size}"
                    )
                faces.append(buffered_ints[1:4])
                del buffered_ints[:4]
                remaining_polygon_ints -= 4
            continue

        upper = line.upper()
        if upper.startswith("POINTS "):
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed POINTS line in {mesh_path}: {line}")
            point_count = int(parts[1])
            remaining_coords = point_count * 3
            continue

        if upper.startswith("POLYGONS "):
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed POLYGONS line in {mesh_path}: {line}")
            polygon_count = int(parts[1])
            remaining_polygon_ints = int(parts[2])
            continue

    if point_count is None or polygon_count is None:
        raise ValueError(f"Missing POINTS or POLYGONS section in {mesh_path}")
    if remaining_coords != 0 or remaining_polygon_ints != 0:
        raise ValueError(f"Incomplete geometry section in {mesh_path}")
    if len(vertices) != point_count:
        raise ValueError(
            f"Expected {point_count} vertices in {mesh_path}, found {len(vertices)}"
        )
    if len(faces) != polygon_count:
        raise ValueError(
            f"Expected {polygon_count} polygons in {mesh_path}, found {len(faces)}"
        )

    return MeshGeometry(
        vertices_xyz=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
    )


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
