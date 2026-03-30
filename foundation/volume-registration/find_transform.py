"""Find the transform between a fixed Zarr volume and a moving source.

Note that neuroglancer uses ZYX coordinates, so that convention is followed for neuroglancer-related bits (most of this file).
Elsewhere, for example writing to a JSON file, we use XYZ coordinates.
"""

from typing import NamedTuple, Optional
from pathlib import Path
import argparse
import http.server
import sys
import threading
import time
from urllib.parse import quote, urlparse
import webbrowser

import neuroglancer
import numpy as np
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def prange(*args):
        return range(*args)

from registration import align_zarrs
from transform_utils import (
    get_volume_dimensions,
    read_vtk_mesh_geometry,
    get_vtk_mesh_info,
    invert_affine_matrix,
    fit_affine_transform_from_points,
    fit_constrained_transform_from_points,
    MeshGeometry,
    matrix_swap_output_xyz_zyx,
    matrix_swap_xyz_zyx,
    points_swap_xyz_zyx,
    read_transform_json,
    strip_vtk_url_prefix,
    write_transform_json,
)


GREEN_SHADER = """
void main() {
    emitRGB(vec3(0, toNormalized(getDataValue()), 0));
}
"""

MAGENTA_SHADER = """
void main() {
    emitRGB(vec3(toNormalized(getDataValue()), 0, toNormalized(getDataValue())));
}
"""

MAGENTA_MESH_SHADER = """
void main() {
    emitRGB(vec3(1.0, 0.0, 1.0));
}
"""

GREEN_POINTS_SHADER = """
void main() {
    setColor(vec3(0, 1, 0));
    setPointMarkerSize(5.0);
}
"""

MAGENTA_POINTS_SHADER = """
void main() {
    setColor(vec3(1, 0, 1));
    setPointMarkerSize(5.0);
}
"""

ERROR_POINTS_SHADER = """
void main() {
  float e = clamp(prop_error() / prop_max_error(), 0.0, 1.0);
  setColor(vec4(e, 1.0 - e, 0.0, 1.0));
  setPointMarkerSize(mix(5.0, 15.0, e));
}
"""

FIXED_POINTS_LAYER_STR = "fixed_points"
MOVING_POINTS_LAYER_STR = "moving_points"
ERROR_POINTS_LAYER_STR = "landmark_errors"
MESH_SLICE_LAYER_STR = "moving_mesh_slices"

UNITLESS_DIMENSIONS = neuroglancer.CoordinateSpace(
    names=["z", "y", "x"],
    units="",
    scales=[1, 1, 1],
)

XYZ_DIMENSIONS = neuroglancer.CoordinateSpace(
    names=["x", "y", "z"],
    units="",
    scales=[1, 1, 1],
)


class MovingSourceDescriptor(NamedTuple):
    source_type: str
    layer_url: str
    center_source: np.ndarray
    unit_size_um: float
    input_dimensions: neuroglancer.CoordinateSpace


class LocalMeshHttpServer(NamedTuple):
    server: http.server.ThreadingHTTPServer
    thread: threading.Thread
    base_url: str
    directory: Path


class MeshSliceTransformCache(NamedTuple):
    transform_key: tuple
    transformed_vertices_zyx: np.ndarray
    face_axis_mins: np.ndarray
    face_axis_maxs: np.ndarray


local_mesh_http_server: Optional[LocalMeshHttpServer] = None
moving_mesh_geometry: Optional[MeshGeometry] = None
moving_mesh_vertices_homogeneous: Optional[np.ndarray] = None
mesh_slice_transform_cache: Optional[MeshSliceTransformCache] = None
mesh_slice_max_segments = 256
numba_mesh_slice_kernels_warmed = False
slice_overlay_update_pending = False
last_mesh_slice_overlay_key = None
constrained_fit_mode = False


def infer_moving_source_type(path: str, explicit_type: str) -> str:
    """Infer the moving source type from the CLI args."""
    if explicit_type != "auto":
        return explicit_type

    stripped = strip_vtk_url_prefix(path)
    if path.startswith("vtk://") or stripped.lower().endswith(".vtk"):
        return "mesh"
    return "zarr"


def is_remote_mesh_path(path: str) -> bool:
    """Check whether a VTK mesh path ultimately points to HTTP(S)."""
    stripped = strip_vtk_url_prefix(path)
    parsed = urlparse(stripped)
    return parsed.scheme in ("http", "https")


def ensure_local_mesh_http_server(directory: Path) -> LocalMeshHttpServer:
    """Serve a local mesh directory over HTTP with permissive CORS for Neuroglancer."""
    global local_mesh_http_server

    directory = directory.resolve()
    if local_mesh_http_server is not None:
        if local_mesh_http_server.directory == directory:
            return local_mesh_http_server

    class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            super().end_headers()

        def log_message(self, format, *args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), CORSRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    local_mesh_http_server = LocalMeshHttpServer(
        server=server,
        thread=thread,
        base_url=base_url,
        directory=directory,
    )
    print(f"Serving local mesh files from {directory} at {base_url}")
    return local_mesh_http_server


def resolve_mesh_layer_url(path: str) -> str:
    """Resolve a mesh path to a Neuroglancer VTK layer URL."""
    stripped = strip_vtk_url_prefix(path)
    if is_remote_mesh_path(path):
        return f"vtk://{stripped}"

    local_path = Path(stripped).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Moving mesh file does not exist: {local_path}")

    server = ensure_local_mesh_http_server(local_path.parent)
    return f"vtk://{server.base_url}/{quote(local_path.name)}"


def build_moving_source_descriptor(
    path: str,
    source_type: str,
    moving_voxel_size: Optional[float],
    moving_source_unit_size: Optional[float],
) -> MovingSourceDescriptor:
    """Build the moving source descriptor used by the interactive tool."""
    if source_type == "zarr":
        moving_dimensions = get_volume_dimensions(path, moving_voxel_size)
        return MovingSourceDescriptor(
            source_type="zarr",
            layer_url=f"zarr://{path}",
            center_source=np.array(
                [
                    moving_dimensions.voxels_z / 2.0,
                    moving_dimensions.voxels_y / 2.0,
                    moving_dimensions.voxels_x / 2.0,
                ],
                dtype=np.float64,
            ),
            unit_size_um=moving_dimensions.voxel_size_um,
            input_dimensions=UNITLESS_DIMENSIONS,
        )

    unit_size_um = (
        moving_source_unit_size
        if moving_source_unit_size is not None
        else moving_voxel_size
    )
    if unit_size_um is None:
        raise ValueError(
            "Mesh moving sources require --moving-source-unit-size (or --moving-voxel-size as a compatibility alias)"
        )

    mesh_info = get_vtk_mesh_info(path)
    mesh_url = resolve_mesh_layer_url(path)
    return MovingSourceDescriptor(
        source_type="mesh",
        layer_url=mesh_url,
        center_source=np.array(mesh_info.center_xyz, dtype=np.float64),
        unit_size_um=unit_size_um,
        input_dimensions=XYZ_DIMENSIONS,
    )


def make_initial_transform(
    moving_source: MovingSourceDescriptor,
    scale_factor: float,
    fixed_center_zyx: np.ndarray,
) -> np.ndarray:
    """Make the initial moving-source transform in viewer coordinates."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale_factor
    if moving_source.source_type == "mesh":
        transform = matrix_swap_output_xyz_zyx(transform)
        transformed_center = transform @ np.append(moving_source.center_source, 1.0)
        transform[:3, 3] = fixed_center_zyx - transformed_center[:3]
        return transform
    return transform


def init_layers(
    viewer: neuroglancer.Viewer,
    fixed_path: str,
    fixed_dimensions,
    moving_source: MovingSourceDescriptor,
    scale_factor: float,
) -> None:
    """Initialize the fixed Zarr volume and the moving source layers."""
    with viewer.txn() as state:
        fixed_source = neuroglancer.LayerDataSource(
            url=f"zarr://{fixed_path}",
            transform=neuroglancer.CoordinateSpaceTransform(
                output_dimensions=UNITLESS_DIMENSIONS,
            ),
        )
        state.layers.append(
            name="fixed",
            layer=neuroglancer.ImageLayer(
                source=fixed_source,
            ),
            shader=GREEN_SHADER,
            blend="additive",
            opacity=1.0,
        )

        fixed_center_zyx = np.array(
            [
                fixed_dimensions.voxels_z / 2.0,
                fixed_dimensions.voxels_y / 2.0,
                fixed_dimensions.voxels_x / 2.0,
            ],
            dtype=np.float64,
        )
        initial_transform = make_initial_transform(
            moving_source, scale_factor, fixed_center_zyx
        )[:-1, :]
        transform_kwargs = dict(
            output_dimensions=UNITLESS_DIMENSIONS,
            matrix=initial_transform,
        )
        if moving_source.source_type == "mesh":
            transform_kwargs.update(
                input_dimensions=moving_source.input_dimensions,
                source_rank=3,
            )

        moving_layer_source = neuroglancer.LayerDataSource(
            url=moving_source.layer_url,
            transform=neuroglancer.CoordinateSpaceTransform(**transform_kwargs),
        )

        if moving_source.source_type == "mesh":
            state.layers.append(
                name="moving",
                layer=neuroglancer.SingleMeshLayer(
                    source=moving_layer_source,
                    shader=MAGENTA_MESH_SHADER,
                ),
            )
        else:
            state.layers.append(
                name="moving",
                layer=neuroglancer.ImageLayer(
                    source=moving_layer_source,
                ),
                shader=MAGENTA_SHADER,
                blend="additive",
                opacity=1.0,
            )

def toggle_color(_):
    with viewer.txn() as state:
        if "vec3" in state.layers["fixed"].shader:
            state.layers["fixed"].shader = ""
            state.layers["moving"].shader = ""
        else:
            state.layers["fixed"].shader = GREEN_SHADER
            state.layers["moving"].shader = (
                MAGENTA_MESH_SHADER
                if moving_source.source_type == "mesh"
                else MAGENTA_SHADER
            )


def make_rotation_matrix(axis: str, angle_deg: float):
    """Make a rotation matrix for the given axis and angle."""
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    if axis == "z":
        return np.array(
            [
                [1, 0, 0, 0],
                [0, cos_theta, -sin_theta, 0],
                [0, sin_theta, cos_theta, 0],
                [0, 0, 0, 1],
            ]
        )
    elif axis == "y":
        return np.array(
            [
                [cos_theta, 0, sin_theta, 0],
                [0, 1, 0, 0],
                [-sin_theta, 0, cos_theta, 0],
                [0, 0, 0, 1],
            ]
        )
    elif axis == "x":
        return np.array(
            [
                [cos_theta, -sin_theta, 0, 0],
                [sin_theta, cos_theta, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
    else:
        raise ValueError(f"Invalid axis: {axis}")


def make_flip_matrix(axis: str):
    """Make a flip matrix for the given axis."""
    if axis == "z":
        return np.array(
            [
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
    elif axis == "y":
        return np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
    elif axis == "x":
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )
    else:
        raise ValueError(f"Invalid axis: {axis}")


def make_translate_matrix(axis: str, amount: float):
    """Make a translation matrix for the given axis and amount."""
    if axis == "z":
        return np.array([[1, 0, 0, amount], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    elif axis == "y":
        return np.array([[1, 0, 0, 0], [0, 1, 0, amount], [0, 0, 1, 0], [0, 0, 0, 1]])
    elif axis == "x":
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, amount], [0, 0, 0, 1]])
    else:
        raise ValueError(f"Invalid axis: {axis}")


def apply_matrix_to_centered(
    state: neuroglancer.ViewerState,
    transform_matrix: np.ndarray,
    moving_source: MovingSourceDescriptor,
):
    """Apply a transformation matrix to a layer, centered on the moving source.

    Args:
        state: The neuroglancer viewer state
        transform_matrix: 4x4 homogeneous transformation matrix to apply
        moving_source: Moving source descriptor in source coordinates
    """
    original_matrix = get_current_transform(state)

    # Current position of the moving source center in fixed space.
    cz, cy, cx, _ = original_matrix @ np.append(moving_source.center_source, 1.0)

    # Center volume about origin
    translate_to_origin_mat = np.array(
        [
            [1, 0, 0, -cz],
            [0, 1, 0, -cy],
            [0, 0, 1, -cx],
            [0, 0, 0, 1],
        ]
    )
    # Translate back to original position
    translate_back_mat = np.array(
        [
            [1, 0, 0, cz],
            [0, 1, 0, cy],
            [0, 0, 1, cx],
            [0, 0, 0, 1],
        ]
    )

    matrix = (
        translate_back_mat
        @ transform_matrix
        @ translate_to_origin_mat
        @ original_matrix
    )
    set_current_transform(state, matrix)


def _make_rotator(axis: str, angle_deg: float):
    def handler(_):
        with viewer.txn() as state:
            rotate_mat = make_rotation_matrix(axis, angle_deg)
            apply_matrix_to_centered(state, rotate_mat, moving_source)

    return handler


def _make_flipper(axis: str):
    def handler(_):
        with viewer.txn() as state:
            flip_mat = make_flip_matrix(axis)
            apply_matrix_to_centered(state, flip_mat, moving_source)

    return handler


def _make_translator(axis: str, amount: float):
    def handler(_):
        with viewer.txn() as state:
            translate_mat = make_translate_matrix(axis, amount)
            apply_matrix_to_centered(state, translate_mat, moving_source)

    return handler


def schema_to_current_transform(matrix_xyz: np.ndarray) -> np.ndarray:
    """Convert a schema-space transform into the viewer's current transform space."""
    if moving_source.source_type == "mesh":
        return matrix_swap_output_xyz_zyx(matrix_xyz)
    return matrix_swap_xyz_zyx(matrix_xyz)


def current_to_schema_transform(matrix: np.ndarray) -> np.ndarray:
    """Convert the viewer's current transform into schema-space XYZ coordinates."""
    if moving_source.source_type == "mesh":
        return matrix_swap_output_xyz_zyx(matrix)
    return matrix_swap_xyz_zyx(matrix)


def schema_fixed_points_to_current(points_xyz: list) -> list:
    """Convert schema fixed landmarks to viewer fixed-space landmarks."""
    return points_swap_xyz_zyx(points_xyz)


def schema_moving_points_to_current(points_xyz: list) -> list:
    """Convert schema moving landmarks to the current moving-source coordinate order."""
    if moving_source.source_type == "mesh":
        return points_xyz
    return points_swap_xyz_zyx(points_xyz)


def current_fixed_points_to_schema(points: list) -> list:
    """Convert viewer fixed-space landmarks to schema XYZ coordinates."""
    return points_swap_xyz_zyx(points)


def current_moving_points_to_schema(points: list) -> list:
    """Convert current moving-source landmarks to schema XYZ coordinates."""
    if moving_source.source_type == "mesh":
        return points
    return points_swap_xyz_zyx(points)


def make_mesh_slice_overlay_key(
    position_zyx: np.ndarray, transform: np.ndarray
) -> tuple:
    """Make a compact key for mesh slice overlay invalidation."""
    return (
        tuple(np.round(position_zyx, 4)),
        tuple(np.round(transform.reshape(-1), 6)),
    )


def make_transform_only_key(transform: np.ndarray) -> tuple:
    """Make a compact key for the moving mesh affine transform alone."""
    return tuple(np.round(transform.reshape(-1), 6))


def ensure_mesh_slice_transform_cache(transform: np.ndarray) -> MeshSliceTransformCache:
    """Cache transformed mesh geometry and per-axis face bounds for slice updates."""
    global mesh_slice_transform_cache

    if moving_mesh_geometry is None or moving_mesh_vertices_homogeneous is None:
        raise RuntimeError("Mesh slice overlay requested without loaded mesh geometry")

    transform_key = make_transform_only_key(transform)
    if (
        mesh_slice_transform_cache is not None
        and mesh_slice_transform_cache.transform_key == transform_key
    ):
        return mesh_slice_transform_cache

    transform32 = transform.astype(np.float32, copy=False)
    transformed_vertices_zyx = (
        transform32 @ moving_mesh_vertices_homogeneous.T
    ).T[:, :3].astype(np.float32, copy=False)

    i0, i1, i2 = moving_mesh_geometry.faces.T
    face_axis_mins = np.empty((3, moving_mesh_geometry.faces.shape[0]), dtype=np.float32)
    face_axis_maxs = np.empty((3, moving_mesh_geometry.faces.shape[0]), dtype=np.float32)
    for axis in range(3):
        axis_values = transformed_vertices_zyx[:, axis]
        v0 = axis_values[i0]
        v1 = axis_values[i1]
        v2 = axis_values[i2]
        face_axis_mins[axis, :] = np.minimum(np.minimum(v0, v1), v2)
        face_axis_maxs[axis, :] = np.maximum(np.maximum(v0, v1), v2)

    mesh_slice_transform_cache = MeshSliceTransformCache(
        transform_key=transform_key,
        transformed_vertices_zyx=transformed_vertices_zyx,
        face_axis_mins=face_axis_mins,
        face_axis_maxs=face_axis_maxs,
    )
    return mesh_slice_transform_cache


@njit(cache=True)
def _add_unique_intersection_point(
    points: np.ndarray, count: int, x: np.float32, y: np.float32, z: np.float32, eps: float
) -> int:
    for i in range(count):
        if (
            abs(points[i, 0] - x) <= eps
            and abs(points[i, 1] - y) <= eps
            and abs(points[i, 2] - z) <= eps
        ):
            return count
    if count < 2:
        points[count, 0] = x
        points[count, 1] = y
        points[count, 2] = z
        return count + 1
    return count


@njit(cache=True)
def _triangle_plane_segment_into(
    transformed_vertices_zyx: np.ndarray,
    i0: int,
    i1: int,
    i2: int,
    axis: int,
    plane_coord: float,
    eps: float,
    segment_points: np.ndarray,
) -> int:
    d0 = transformed_vertices_zyx[i0, axis] - plane_coord
    d1 = transformed_vertices_zyx[i1, axis] - plane_coord
    d2 = transformed_vertices_zyx[i2, axis] - plane_coord

    if abs(d0) <= eps and abs(d1) <= eps and abs(d2) <= eps:
        return 0

    count = 0
    edges = ((i0, i1, d0, d1), (i1, i2, d1, d2), (i2, i0, d2, d0))
    for edge in edges:
        a = edge[0]
        b = edge[1]
        da = edge[2]
        db = edge[3]

        if abs(da) <= eps and abs(db) <= eps:
            continue

        if abs(da) <= eps:
            count = _add_unique_intersection_point(
                segment_points,
                count,
                transformed_vertices_zyx[a, 0],
                transformed_vertices_zyx[a, 1],
                transformed_vertices_zyx[a, 2],
                eps,
            )
            continue

        if abs(db) <= eps:
            count = _add_unique_intersection_point(
                segment_points,
                count,
                transformed_vertices_zyx[b, 0],
                transformed_vertices_zyx[b, 1],
                transformed_vertices_zyx[b, 2],
                eps,
            )
            continue

        if da * db < 0.0:
            t = da / (da - db)
            x = transformed_vertices_zyx[a, 0] + t * (
                transformed_vertices_zyx[b, 0] - transformed_vertices_zyx[a, 0]
            )
            y = transformed_vertices_zyx[a, 1] + t * (
                transformed_vertices_zyx[b, 1] - transformed_vertices_zyx[a, 1]
            )
            z = transformed_vertices_zyx[a, 2] + t * (
                transformed_vertices_zyx[b, 2] - transformed_vertices_zyx[a, 2]
            )
            count = _add_unique_intersection_point(
                segment_points,
                count,
                np.float32(x),
                np.float32(y),
                np.float32(z),
                eps,
            )

    return count


@njit(cache=True, parallel=True)
def _count_mesh_slice_hits_numba_parallel(
    transformed_vertices_zyx: np.ndarray,
    faces: np.ndarray,
    position_zyx: np.ndarray,
    face_axis_mins: np.ndarray,
    face_axis_maxs: np.ndarray,
    eps: float,
) -> np.ndarray:
    num_faces = faces.shape[0]
    num_tasks = 3 * num_faces
    counts = np.zeros(num_tasks, dtype=np.uint8)

    for task_index in prange(num_tasks):
        axis = task_index // num_faces
        face_index = task_index - axis * num_faces
        plane_coord = position_zyx[axis]
        plane_lower = plane_coord - eps
        plane_upper = plane_coord + eps

        if (
            face_axis_mins[axis, face_index] <= plane_upper
            and face_axis_maxs[axis, face_index] >= plane_lower
        ):
            segment_points = np.empty((2, 3), dtype=np.float32)
            num_points = _triangle_plane_segment_into(
                transformed_vertices_zyx,
                faces[face_index, 0],
                faces[face_index, 1],
                faces[face_index, 2],
                axis,
                plane_coord,
                eps,
                segment_points,
            )
            if num_points == 2:
                counts[task_index] = 1

    return counts


@njit(cache=True, parallel=True)
def _write_mesh_slice_segments_numba_parallel(
    transformed_vertices_zyx: np.ndarray,
    faces: np.ndarray,
    position_zyx: np.ndarray,
    face_axis_mins: np.ndarray,
    face_axis_maxs: np.ndarray,
    task_offsets: np.ndarray,
    task_counts: np.ndarray,
    total_segments: int,
    eps: float,
) -> np.ndarray:
    num_faces = faces.shape[0]
    num_tasks = task_counts.shape[0]
    segments = np.empty((total_segments, 6), dtype=np.float32)

    for task_index in prange(num_tasks):
        if task_counts[task_index] == 0:
            continue

        axis = task_index // num_faces
        face_index = task_index - axis * num_faces
        plane_coord = position_zyx[axis]
        segment_points = np.empty((2, 3), dtype=np.float32)
        num_points = _triangle_plane_segment_into(
            transformed_vertices_zyx,
            faces[face_index, 0],
            faces[face_index, 1],
            faces[face_index, 2],
            axis,
            plane_coord,
            eps,
            segment_points,
        )
        if num_points == 2:
            segment_index = task_offsets[task_index]
            segments[segment_index, 0] = segment_points[0, 0]
            segments[segment_index, 1] = segment_points[0, 1]
            segments[segment_index, 2] = segment_points[0, 2]
            segments[segment_index, 3] = segment_points[1, 0]
            segments[segment_index, 4] = segment_points[1, 1]
            segments[segment_index, 5] = segment_points[1, 2]

    return segments


def _compute_mesh_slice_segments_numba_parallel(
    transformed_vertices_zyx: np.ndarray,
    faces: np.ndarray,
    position_zyx: np.ndarray,
    face_axis_mins: np.ndarray,
    face_axis_maxs: np.ndarray,
    eps: float,
) -> np.ndarray:
    task_counts = _count_mesh_slice_hits_numba_parallel(
        transformed_vertices_zyx,
        faces,
        position_zyx,
        face_axis_mins,
        face_axis_maxs,
        eps,
    )

    if task_counts.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    task_offsets = np.cumsum(task_counts, dtype=np.int32) - task_counts
    total_segments = int(task_offsets[-1] + task_counts[-1])
    if total_segments == 0:
        return np.empty((0, 6), dtype=np.float32)

    return _write_mesh_slice_segments_numba_parallel(
        transformed_vertices_zyx,
        faces,
        position_zyx,
        face_axis_mins,
        face_axis_maxs,
        task_offsets,
        task_counts,
        total_segments,
        eps,
    )


def warm_numba_mesh_slice_kernels() -> None:
    """Compile numba kernels once to avoid the first interactive JIT pause."""
    global numba_mesh_slice_kernels_warmed

    if not NUMBA_AVAILABLE or numba_mesh_slice_kernels_warmed:
        return

    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    face_axis_mins = np.zeros((3, 1), dtype=np.float32)
    face_axis_maxs = np.ones((3, 1), dtype=np.float32)
    _compute_mesh_slice_segments_numba_parallel(
        vertices,
        faces,
        position,
        face_axis_mins,
        face_axis_maxs,
        np.float32(1e-5),
    )
    numba_mesh_slice_kernels_warmed = True


def intersect_triangle_with_plane(
    triangle: np.ndarray, axis: int, plane_coord: float, eps: float = 1e-5
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Intersect one triangle with an axis-aligned plane in fixed ZYX space."""
    distances = triangle[:, axis] - plane_coord
    if np.all(np.abs(distances) <= eps):
        return None

    points = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        d0 = distances[i]
        d1 = distances[j]
        p0 = triangle[i]
        p1 = triangle[j]

        if abs(d0) <= eps and abs(d1) <= eps:
            continue
        if abs(d0) <= eps:
            points.append(p0)
            continue
        if abs(d1) <= eps:
            points.append(p1)
            continue
        if d0 * d1 < 0:
            t = d0 / (d0 - d1)
            points.append(p0 + t * (p1 - p0))

    unique_points = []
    for point in points:
        if not any(np.allclose(point, other, atol=eps) for other in unique_points):
            unique_points.append(point)

    if len(unique_points) < 2:
        return None
    if len(unique_points) == 2:
        return unique_points[0], unique_points[1]

    farthest_pair = None
    farthest_distance = -1.0
    for i in range(len(unique_points)):
        for j in range(i + 1, len(unique_points)):
            distance = float(np.linalg.norm(unique_points[i] - unique_points[j]))
            if distance > farthest_distance:
                farthest_distance = distance
                farthest_pair = (unique_points[i], unique_points[j])
    return farthest_pair


def compute_mesh_slice_segments(
    transformed_vertices_zyx: np.ndarray,
    faces: np.ndarray,
    position_zyx: np.ndarray,
    *,
    face_axis_mins: Optional[np.ndarray] = None,
    face_axis_maxs: Optional[np.ndarray] = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute contour segments for the orthogonal XY/XZ/YZ slice planes."""
    if (
        NUMBA_AVAILABLE
        and face_axis_mins is not None
        and face_axis_maxs is not None
    ):
        return _compute_mesh_slice_segments_numba_parallel(
            transformed_vertices_zyx,
            faces,
            position_zyx.astype(np.float32, copy=False),
            face_axis_mins,
            face_axis_maxs,
            np.float32(eps),
        )

    segments = []
    for axis in range(3):
        plane_coord = float(position_zyx[axis])
        if face_axis_mins is not None and face_axis_maxs is not None:
            intersects = (face_axis_mins[axis, :] <= plane_coord + eps) & (
                face_axis_maxs[axis, :] >= plane_coord - eps
            )
        else:
            axis_values = transformed_vertices_zyx[:, axis]
            triangle_axis_values = axis_values[faces]
            intersects = (triangle_axis_values.min(axis=1) <= plane_coord + eps) & (
                triangle_axis_values.max(axis=1) >= plane_coord - eps
            )
        triangles = transformed_vertices_zyx[faces[intersects]]

        for triangle in triangles:
            segment = intersect_triangle_with_plane(triangle, axis, plane_coord, eps)
            if segment is not None:
                segments.append(
                    (
                        segment[0][0],
                        segment[0][1],
                        segment[0][2],
                        segment[1][0],
                        segment[1][1],
                        segment[1][2],
                    )
                )
    if not segments:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(segments, dtype=np.float32)


def request_mesh_slice_overlay_update() -> None:
    """Coalesce mesh slice overlay updates onto the Neuroglancer callback queue."""
    global slice_overlay_update_pending

    moving_source_descriptor = globals().get("moving_source")
    if (
        "viewer" not in globals()
        or globals().get("viewer") is None
        or moving_source_descriptor is None
        or moving_source_descriptor.source_type != "mesh"
        or moving_mesh_geometry is None
        or slice_overlay_update_pending
    ):
        return

    slice_overlay_update_pending = True
    viewer.defer_callback(update_mesh_slice_overlay)


def invalidate_mesh_slice_overlay_cache() -> None:
    """Clear the last rendered slice key so the next request always refreshes."""
    global last_mesh_slice_overlay_key

    last_mesh_slice_overlay_key = None


def update_mesh_slice_overlay() -> None:
    """Update the moving mesh slice contour overlay."""
    global slice_overlay_update_pending
    global last_mesh_slice_overlay_key

    slice_overlay_update_pending = False
    if moving_source.source_type != "mesh" or moving_mesh_geometry is None:
        return

    state = viewer.state
    position_zyx = np.asarray(state.position, dtype=np.float64)
    transform = get_current_transform(state)
    overlay_key = make_mesh_slice_overlay_key(position_zyx, transform)
    if overlay_key == last_mesh_slice_overlay_key:
        return

    transform_cache = ensure_mesh_slice_transform_cache(transform)
    segments = compute_mesh_slice_segments(
        transform_cache.transformed_vertices_zyx,
        moving_mesh_geometry.faces,
        position_zyx,
        face_axis_mins=transform_cache.face_axis_mins,
        face_axis_maxs=transform_cache.face_axis_maxs,
    )

    if mesh_slice_max_segments > 0 and len(segments) > mesh_slice_max_segments:
        step = len(segments) / mesh_slice_max_segments
        keep_indices = np.array(
            [
                min(int(i * step), len(segments) - 1)
                for i in range(mesh_slice_max_segments)
            ],
            dtype=np.int64,
        )
        segments = segments[keep_indices]

    annotations = [
        neuroglancer.LineAnnotation(
            id=f"mesh_slice_{i}",
            point_a=segment[:3],
            point_b=segment[3:6],
        )
        for i, segment in enumerate(segments, start=1)
    ]

    last_mesh_slice_overlay_key = overlay_key
    with viewer.txn() as s:
        if not annotations:
            if MESH_SLICE_LAYER_STR in s.layers:
                del s.layers[MESH_SLICE_LAYER_STR]
            return

        if MESH_SLICE_LAYER_STR not in s.layers:
            s.layers.append(
                name=MESH_SLICE_LAYER_STR,
                layer=neuroglancer.LocalAnnotationLayer(
                    dimensions=UNITLESS_DIMENSIONS,
                    annotation_color="#ff66ff",
                    annotations=annotations,
                ),
            )
        else:
            layer = s.layers[MESH_SLICE_LAYER_STR].layer
            layer.annotations = annotations
            layer.annotation_color = "#ff66ff"


def load_transform(
    viewer_state: neuroglancer.ViewerState,
    input_path: str,
    invert_initial_transform: bool,
) -> None:
    """Read a transform from a file. If provided, also loads landmarks into viewer state."""
    matrix_xyz, fixed_landmarks_xyz, moving_landmarks_xyz = read_transform_json(
        input_path, invert_initial_transform
    )

    matrix = schema_to_current_transform(matrix_xyz)
    fixed_landmarks = schema_fixed_points_to_current(fixed_landmarks_xyz)
    moving_landmarks = schema_moving_points_to_current(moving_landmarks_xyz)

    set_current_transform(viewer_state, matrix)

    # Load landmarks into layers using the same logic as interactive point adding
    for point in fixed_landmarks:
        add_point_from_coords(viewer_state, point, "fixed")

    for point in moving_landmarks:
        add_point_from_coords(viewer_state, point, "moving")

    # Update error layer now that landmarks and transform are loaded
    update_error_layer(viewer_state)


def save_current_transform(
    state: neuroglancer.ViewerState, output_path: Optional[str], fixed_volume_path: str
) -> None:
    """Save the current transform and landmarks to a JSON file, and print the shareable URL."""
    # Get current transform (in ZYX coordinates)
    matrix = get_current_transform(state)

    # Get points from layers if they exist
    fixed_landmarks = []
    moving_landmarks = []

    if FIXED_POINTS_LAYER_STR in state.layers:
        fixed_annotations = state.layers[FIXED_POINTS_LAYER_STR].layer.annotations
        fixed_landmarks = [list(ann.point) for ann in fixed_annotations]

    if MOVING_POINTS_LAYER_STR in state.layers:
        moving_annotations = state.layers[MOVING_POINTS_LAYER_STR].layer.annotations
        moving_landmarks = [list(ann.point) for ann in moving_annotations]

    # Convert to schema XYZ coordinates.
    xyz_matrix = current_to_schema_transform(matrix)
    xyz_moving_landmarks = current_moving_points_to_schema(moving_landmarks)
    xyz_fixed_landmarks = current_fixed_points_to_schema(fixed_landmarks)

    # Write to file or print to stdout
    if output_path is None:
        print("--output-transform not provided, printing transform to stdout:")
        print(matrix)
        print("To save to file, use --output-transform <path>")
    else:
        print(f"Writing transform to {output_path}")
        write_transform_json(
            output_path,
            Path(fixed_volume_path).stem,
            xyz_matrix,
            xyz_fixed_landmarks,
            xyz_moving_landmarks,
        )

    # Print the state URL
    print(
        f"Shareable URL: https://neuroglancer-demo.appspot.com/#!{neuroglancer.url_state.to_url_fragment(state)}"
    )


def get_current_transform(state: neuroglancer.ViewerState) -> np.ndarray:
    """Get the current transform from the viewer state."""
    transform = state.layers["moving"].layer.source[0].transform.matrix
    # Add homogeneous coordinate
    transform = np.concatenate([transform, [[0, 0, 0, 1]]], axis=0)
    return transform


def set_current_transform(
    state: neuroglancer.ViewerState, transform: np.ndarray
) -> None:
    """Set the current transform in the viewer state."""
    global mesh_slice_transform_cache

    # Remove homogeneous coordinate
    transform = transform[:-1, :]
    state.layers["moving"].layer.source[0].transform.matrix = transform

    # Also update moving_points layer if it exists
    if MOVING_POINTS_LAYER_STR in state.layers:
        state.layers[MOVING_POINTS_LAYER_STR].layer.source[
            0
        ].transform.matrix = transform

    if moving_source.source_type == "mesh":
        mesh_slice_transform_cache = None
        invalidate_mesh_slice_overlay_cache()
        request_mesh_slice_overlay_update()


def add_point_from_coords(state: neuroglancer.ViewerState, point_coords, point_type):
    """Add a point to the specified points layer from given coordinates."""
    # For loading from JSON, we already have the coordinates in the right space
    if point_type == "fixed":
        layer_name = FIXED_POINTS_LAYER_STR
        shader = GREEN_POINTS_SHADER
    elif point_type == "moving":
        layer_name = MOVING_POINTS_LAYER_STR
        shader = MAGENTA_POINTS_SHADER
    else:
        raise ValueError(f"Unknown point type: {point_type}")

    # Get current number of points in layer for ID
    if layer_name in state.layers:
        num_points = len(state.layers[layer_name].layer.annotations)
    else:
        num_points = 0
    point_id = f"{point_type[0]}p_{num_points + 1}"

    # Make the layer if it does not exist
    if layer_name not in state.layers:
        state.layers.append(
            name=layer_name,
            layer=neuroglancer.LocalAnnotationLayer(
                dimensions=UNITLESS_DIMENSIONS,
                shader=shader,
            ),
        )

        if point_type == "moving":
            moving_points_transform = state.layers[layer_name].layer.source[0].transform
            moving_points_transform.input_dimensions = moving_source.input_dimensions
            moving_points_transform.source_rank = 3

        # For moving points layer, apply the current transform when adding first point
        # (to have the moving points render in the correct position)
        if (
            point_type == "moving"
            and len(state.layers[layer_name].layer.annotations) == 0
        ):
            current_transform = get_current_transform(state)
            set_current_transform(state, current_transform)

    # Add annotation to the layer
    state.layers[layer_name].layer.annotations.append(
        neuroglancer.PointAnnotation(point=point_coords, id=point_id)
    )

    # Check if we can fit a transform automatically
    update_transform_if_sufficient_points(state)


def add_point(action_state, point_type):
    """Add a point to the specified points layer at the current mouse position."""
    with viewer.txn() as state:
        # Get mouse position from the action state
        mouse_position = action_state.mouse_voxel_coordinates
        if mouse_position is None:
            print("No mouse position available")
            return

        # Convert to fixed space point
        fixed_space_point = [
            float(mouse_position[0]),
            float(mouse_position[1]),
            float(mouse_position[2]),
        ]

        print(f"Adding {point_type} point at {fixed_space_point}")

        if point_type == "fixed":
            # Store in fixed space
            add_point_from_coords(state, fixed_space_point, "fixed")
        elif point_type == "moving":
            # Transform to moving space for storage
            fixed_point_homogeneous = np.array([*fixed_space_point, 1.0])
            current_transform = get_current_transform(state)
            inverse_transform = invert_affine_matrix(current_transform)
            moving_point_homogeneous = inverse_transform @ fixed_point_homogeneous

            moving_space_point = [
                float(moving_point_homogeneous[0]),
                float(moving_point_homogeneous[1]),
                float(moving_point_homogeneous[2]),
            ]
            add_point_from_coords(state, moving_space_point, "moving")
        else:
            raise ValueError(f"Unknown point type: {point_type}")


def add_fixed_point(action_state):
    """Add a point to the fixed points layer at the current mouse position."""
    add_point(action_state, "fixed")


def add_moving_point(action_state):
    """Add a point to the moving points layer at the current mouse position."""
    add_point(action_state, "moving")


def fine_align(_):
    """Run zarr alignment and update the moving layer with the result."""
    if moving_source.source_type != "zarr":
        print("Fine alignment is only available when the moving source is a Zarr volume")
        return

    with viewer.txn() as state:
        current_transform = get_current_transform(state)

        # Run alignment
        print("Running zarr alignment...")
        refined_transform = align_zarrs(args.fixed, args.moving, current_transform)
        print(f"Refined transform: {refined_transform}")

        # Apply the refined transform
        set_current_transform(state, refined_transform)
        update_error_layer(state)
        print("Alignment complete - transform updated")


def find_nearest_point(current_position, state, point_type_filter=None):
    """Find the nearest point to the current cursor position.

    Args:
        current_position: [x, y, z] coordinates to search from
        state: neuroglancer viewer state
        point_type_filter: Optional filter - "fixed", "moving", or None for any type

    Returns:
        For point_type_filter=None: (point_type, point_index, distance, point_coords)
        For specific filter: (point_index, distance, point_coords) or (None, None, None)
    """
    min_distance = float("inf")
    nearest_point_type = None
    nearest_index = None
    nearest_coords = None

    # Check fixed points
    if (
        point_type_filter is None or point_type_filter == "fixed"
    ) and FIXED_POINTS_LAYER_STR in state.layers:
        fixed_annotations = state.layers[FIXED_POINTS_LAYER_STR].layer.annotations
        for i, ann in enumerate(fixed_annotations):
            point = list(ann.point)
            distance = np.sqrt(
                (current_position[0] - point[0]) ** 2
                + (current_position[1] - point[1]) ** 2
                + (current_position[2] - point[2]) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                nearest_point_type = "fixed"
                nearest_index = i
                nearest_coords = point

    # Check moving points (transform to fixed coordinates first)
    if (
        point_type_filter is None or point_type_filter == "moving"
    ) and MOVING_POINTS_LAYER_STR in state.layers:
        moving_annotations = state.layers[MOVING_POINTS_LAYER_STR].layer.annotations
        current_transform = get_current_transform(state)

        for i, ann in enumerate(moving_annotations):
            # Transform moving point to fixed coordinates
            moving_point_homogeneous = np.array([*ann.point, 1.0])
            fixed_point_homogeneous = current_transform @ moving_point_homogeneous
            fixed_point = fixed_point_homogeneous[:3]

            distance = np.sqrt(
                (current_position[0] - fixed_point[0]) ** 2
                + (current_position[1] - fixed_point[1]) ** 2
                + (current_position[2] - fixed_point[2]) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                nearest_point_type = "moving"
                nearest_index = i
                nearest_coords = list(ann.point)  # Return original moving coordinates

    if nearest_point_type is None:
        if point_type_filter is None:
            return None, None, None, None
        else:
            return None, None, None

    if point_type_filter is None:
        return nearest_point_type, nearest_index, min_distance, nearest_coords
    else:
        return nearest_index, min_distance, nearest_coords


def navigate_to_fixed_point(action_state, direction):
    """Navigate to the previous/next fixed point relative to current cursor position."""
    with viewer.txn() as state:
        # Get current cursor position
        current_position = action_state.mouse_voxel_coordinates
        if current_position is None:
            # Fallback to viewer position if mouse position is not available
            current_position = state.position

        # Get all fixed points
        if FIXED_POINTS_LAYER_STR not in state.layers:
            print("No fixed points available")
            return

        fixed_annotations = state.layers[FIXED_POINTS_LAYER_STR].layer.annotations
        if len(fixed_annotations) == 0:
            print("No fixed points available")
            return

        # Find nearest fixed point to current position
        nearest_index, _, _ = find_nearest_point(current_position, state, "fixed")
        if nearest_index is None:
            return

        # Calculate target index based on direction
        if direction == "previous":
            target_index = (nearest_index - 1) % len(fixed_annotations)
        else:  # next
            target_index = (nearest_index + 1) % len(fixed_annotations)

        target_point = list(fixed_annotations[target_index].point)

        # Navigate viewer to target point
        state.position = target_point


def navigate_to_previous_fixed_point(action_state):
    """Navigate to the previous fixed point."""
    navigate_to_fixed_point(action_state, "previous")


def navigate_to_next_fixed_point(action_state):
    """Navigate to the next fixed point."""
    navigate_to_fixed_point(action_state, "next")


def update_transform_if_sufficient_points(state):
    """Update transform if there are sufficient matching point pairs.

    Uses constrained fitting (5 DOF + z-flip, min 3 pairs) or unconstrained
    affine fitting (12 DOF, min 4 pairs) depending on constrained_fit_mode.
    """
    global constrained_fit_mode
    if (
        FIXED_POINTS_LAYER_STR in state.layers
        and MOVING_POINTS_LAYER_STR in state.layers
    ):
        fixed_annotations = state.layers[FIXED_POINTS_LAYER_STR].layer.annotations
        moving_annotations = state.layers[MOVING_POINTS_LAYER_STR].layer.annotations

        min_points = 3 if constrained_fit_mode else 4
        if (
            len(fixed_annotations) == len(moving_annotations)
            and len(fixed_annotations) >= min_points
        ):
            fixed_points_list = [list(ann.point) for ann in fixed_annotations]
            moving_points_list = [list(ann.point) for ann in moving_annotations]

            if constrained_fit_mode:
                mode_label = "constrained"
                transform = fit_constrained_transform_from_points(
                    fixed_points_list, moving_points_list
                )
            else:
                mode_label = "unconstrained"
                transform = fit_affine_transform_from_points(
                    fixed_points_list, moving_points_list
                )
            if transform is not None:
                set_current_transform(state, transform)
                update_error_layer(state)
                print(
                    f"Updated transform ({mode_label}) from {len(fixed_annotations)} point pairs"
                )
                return True
    return False


def renumber_point_ids(state, point_type):
    """Renumber point IDs to keep them sequential after deletion."""
    if point_type == "fixed":
        layer_name = FIXED_POINTS_LAYER_STR
        prefix = "fp"
    else:
        layer_name = MOVING_POINTS_LAYER_STR
        prefix = "mp"

    if layer_name not in state.layers:
        return

    annotations = state.layers[layer_name].layer.annotations
    for i, ann in enumerate(annotations):
        ann.id = f"{prefix}_{i + 1}"


def compute_landmark_errors(state: neuroglancer.ViewerState):
    """Compute per-landmark errors: distance between fixed point and transformed moving point.

    Returns list of (index, fixed_point_zyx, error_distance) tuples, or empty list if
    there aren't matched pairs.
    """
    if (
        FIXED_POINTS_LAYER_STR not in state.layers
        or MOVING_POINTS_LAYER_STR not in state.layers
    ):
        return []

    fixed_annotations = state.layers[FIXED_POINTS_LAYER_STR].layer.annotations
    moving_annotations = state.layers[MOVING_POINTS_LAYER_STR].layer.annotations

    n_pairs = min(len(fixed_annotations), len(moving_annotations))
    if n_pairs == 0:
        return []

    current_transform = get_current_transform(state)
    errors = []
    for i in range(n_pairs):
        fixed_pt = np.array(fixed_annotations[i].point)
        moving_pt = np.array(moving_annotations[i].point)
        # Transform moving point to fixed space
        transformed = (current_transform @ np.append(moving_pt, 1.0))[:3]
        error = float(np.linalg.norm(fixed_pt - transformed))
        errors.append((i, list(fixed_pt), error))
    return errors


def update_error_layer(state: neuroglancer.ViewerState):
    """Rebuild the landmark error annotation layer with current errors."""
    errors = compute_landmark_errors(state)

    # Remove existing error layer
    if ERROR_POINTS_LAYER_STR in state.layers:
        del state.layers[ERROR_POINTS_LAYER_STR]

    if not errors:
        return

    max_error = max(e for _, _, e in errors)
    if max_error == 0:
        max_error = 1.0  # avoid division by zero in shader

    annotations = []
    for i, fixed_pt, error in errors:
        annotations.append(
            neuroglancer.PointAnnotation(
                point=fixed_pt,
                id=f"err_{i + 1}",
                description=f"#{i + 1}  error: {error:.1f}",
                props=[error, max_error],
            )
        )

    state.layers.append(
        name=ERROR_POINTS_LAYER_STR,
        layer=neuroglancer.LocalAnnotationLayer(
            dimensions=UNITLESS_DIMENSIONS,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(
                    id="error",
                    type="float32",
                    description="Transform error (voxels)",
                ),
                neuroglancer.AnnotationPropertySpec(
                    id="max_error",
                    type="float32",
                    description="Max error across all landmarks",
                ),
            ],
            annotations=annotations,
            shader=ERROR_POINTS_SHADER,
        ),
    )

    # Print error summary to terminal
    sorted_errors = sorted(errors, key=lambda x: x[2], reverse=True)
    print(f"\n{'─' * 40}")
    print(f"  Landmark errors ({len(errors)} pairs)")
    print(f"{'─' * 40}")
    print(f"  {'#':>3}  {'Error':>10}")
    print(f"  {'─' * 3}  {'─' * 10}")
    for i, _, error in sorted_errors:
        print(f"  {i + 1:>3}  {error:>10.2f}")
    rms = np.sqrt(np.mean([e ** 2 for _, _, e in errors]))
    print(f"{'─' * 40}")
    print(f"  RMS error: {rms:.2f}")
    print(f"  Max error: {max_error:.2f}")
    print(f"{'─' * 40}\n")


def navigate_to_worst_landmark(_):
    """Navigate the viewer to the landmark with the largest error."""
    with viewer.txn() as state:
        errors = compute_landmark_errors(state)
        if not errors:
            print("No landmark pairs to evaluate")
            return
        worst = max(errors, key=lambda x: x[2])
        state.position = worst[1]
        print(f"Navigated to landmark #{worst[0] + 1} (error: {worst[2]:.2f})")


def perturb_nearest_fixed_point(action_state, axis, amount):
    """Perturb the nearest fixed point by the given amount along the specified axis."""
    with viewer.txn() as state:
        # Get current cursor position
        current_position = action_state.mouse_voxel_coordinates
        if current_position is None:
            # Fallback to viewer position if mouse position is not available
            current_position = state.position

        # Find nearest fixed point
        nearest_index, distance, _ = find_nearest_point(
            current_position, state, "fixed"
        )

        if nearest_index is None:
            print("No fixed points available to perturb")
            return

        # Get the fixed point annotation
        fixed_annotations = state.layers[FIXED_POINTS_LAYER_STR].layer.annotations
        point_annotation = fixed_annotations[nearest_index]

        # Update the point coordinates
        current_point = list(point_annotation.point)
        if axis == "x":
            current_point[2] += amount  # X is index 2 in ZYX
        elif axis == "y":
            current_point[1] += amount  # Y is index 1 in ZYX
        elif axis == "z":
            current_point[0] += amount  # Z is index 0 in ZYX

        point_annotation.point = current_point

        print(
            f"Perturbed fixed point {point_annotation.id} by {amount} along {axis}-axis"
        )
        print(f"New position: {current_point}")

        # Update transform if sufficient points exist
        update_transform_if_sufficient_points(state)


def _make_point_perturber(axis: str, amount: float):
    """Create a function that perturbs the nearest fixed point along the given axis."""

    def handler(action_state):
        perturb_nearest_fixed_point(action_state, axis, amount)

    return handler


def delete_nearest_point(action_state):
    """Delete the nearest point (fixed or moving) to the current cursor position."""
    with viewer.txn() as state:
        # Get current cursor position
        current_position = action_state.mouse_voxel_coordinates
        if current_position is None:
            # Fallback to viewer position if mouse position is not available
            current_position = state.position

        # Find nearest point of any type
        point_type, point_index, distance, _ = find_nearest_point(
            current_position, state
        )

        if point_type is None:
            print("No points available to delete")
            return

        # Delete the point
        if point_type == "fixed":
            layer_name = FIXED_POINTS_LAYER_STR
        else:
            layer_name = MOVING_POINTS_LAYER_STR

        annotations = state.layers[layer_name].layer.annotations
        deleted_point = annotations[point_index]
        del annotations[point_index]

        # Renumber remaining points to keep IDs sequential
        renumber_point_ids(state, point_type)

        print(
            f"Deleted {point_type} point {deleted_point.id} at {list(deleted_point.point)}"
        )
        print(f"Distance from cursor: {distance:.2f}")

        # Update transform if sufficient points remain
        update_transform_if_sufficient_points(state)


def toggle_constrained_fit(_):
    """Toggle between constrained and unconstrained affine fitting."""
    global constrained_fit_mode
    if not constrained_fit_mode and moving_source.source_type == "mesh":
        print("Constrained fit mode is not supported with mesh moving sources.")
        return
    constrained_fit_mode = not constrained_fit_mode
    mode = "CONSTRAINED (5 DOF + z-flip)" if constrained_fit_mode else "UNCONSTRAINED (12 DOF)"
    print(f"Fit mode: {mode}")
    with viewer.txn() as state:
        update_transform_if_sufficient_points(state)


def write_current_transform(_):
    """Write the current transform and print the shareable URL."""
    with viewer.txn() as state:
        save_current_transform(state, args.output_transform, args.fixed)


def add_actions_and_keybinds(
    viewer: neuroglancer.Viewer,
    *,
    enable_fine_align: bool = True,
    small_rotate_deg: float = 1.0,
    large_rotate_deg: float = 90.0,
    small_translate_voxels: float = 10.0,
    large_translate_voxels: float = 1000.0,
    point_perturb_voxels: float = 1.0,
) -> None:

    viewer.actions.add("toggle-color", toggle_color)
    viewer.actions.add("toggle-constrained-fit", toggle_constrained_fit)
    viewer.actions.add("write-transform", write_current_transform)
    if enable_fine_align:
        viewer.actions.add("fine-align", fine_align)
    viewer.actions.add("add-fixed-point", add_fixed_point)
    viewer.actions.add("add-moving-point", add_moving_point)
    viewer.actions.add("delete-nearest-point", delete_nearest_point)
    viewer.actions.add("previous-fixed-point", navigate_to_previous_fixed_point)
    viewer.actions.add("next-fixed-point", navigate_to_next_fixed_point)
    viewer.actions.add("goto-worst-landmark", navigate_to_worst_landmark)
    viewer.actions.add("rot-x-plus-small", _make_rotator("x", small_rotate_deg))
    viewer.actions.add("rot-x-minus-small", _make_rotator("x", -small_rotate_deg))
    viewer.actions.add("rot-y-plus-small", _make_rotator("y", small_rotate_deg))
    viewer.actions.add("rot-y-minus-small", _make_rotator("y", -small_rotate_deg))
    viewer.actions.add("rot-z-plus-small", _make_rotator("z", small_rotate_deg))
    viewer.actions.add("rot-z-minus-small", _make_rotator("z", -small_rotate_deg))
    viewer.actions.add("rot-x-plus-large", _make_rotator("x", large_rotate_deg))
    viewer.actions.add("rot-x-minus-large", _make_rotator("x", -large_rotate_deg))
    viewer.actions.add("rot-y-plus-large", _make_rotator("y", large_rotate_deg))
    viewer.actions.add("rot-y-minus-large", _make_rotator("y", -large_rotate_deg))
    viewer.actions.add("rot-z-plus-large", _make_rotator("z", large_rotate_deg))
    viewer.actions.add("rot-z-minus-large", _make_rotator("z", -large_rotate_deg))
    viewer.actions.add("flip-x", _make_flipper("x"))
    viewer.actions.add("flip-y", _make_flipper("y"))
    viewer.actions.add("flip-z", _make_flipper("z"))
    viewer.actions.add("trans-x-plus-small", _make_translator("x", small_translate_voxels))
    viewer.actions.add(
        "trans-x-minus-small", _make_translator("x", -small_translate_voxels)
    )
    viewer.actions.add("trans-y-plus-small", _make_translator("y", small_translate_voxels))
    viewer.actions.add(
        "trans-y-minus-small", _make_translator("y", -small_translate_voxels)
    )
    viewer.actions.add("trans-z-plus-small", _make_translator("z", small_translate_voxels))
    viewer.actions.add(
        "trans-z-minus-small", _make_translator("z", -small_translate_voxels)
    )
    viewer.actions.add("trans-x-plus-large", _make_translator("x", large_translate_voxels))
    viewer.actions.add(
        "trans-x-minus-large", _make_translator("x", -large_translate_voxels)
    )
    viewer.actions.add("trans-y-plus-large", _make_translator("y", large_translate_voxels))
    viewer.actions.add(
        "trans-y-minus-large", _make_translator("y", -large_translate_voxels)
    )
    viewer.actions.add("trans-z-plus-large", _make_translator("z", large_translate_voxels))
    viewer.actions.add(
        "trans-z-minus-large", _make_translator("z", -large_translate_voxels)
    )
    viewer.actions.add(
        "perturb-fixed-point-x-plus", _make_point_perturber("x", point_perturb_voxels)
    )
    viewer.actions.add(
        "perturb-fixed-point-x-minus", _make_point_perturber("x", -point_perturb_voxels)
    )
    viewer.actions.add(
        "perturb-fixed-point-y-plus", _make_point_perturber("y", point_perturb_voxels)
    )
    viewer.actions.add(
        "perturb-fixed-point-y-minus", _make_point_perturber("y", -point_perturb_voxels)
    )
    viewer.actions.add(
        "perturb-fixed-point-z-plus", _make_point_perturber("z", point_perturb_voxels)
    )
    viewer.actions.add(
        "perturb-fixed-point-z-minus", _make_point_perturber("z", -point_perturb_voxels)
    )

    with viewer.config_state.txn() as s:
        s.input_event_bindings.viewer["keyc"] = "toggle-color"
        s.input_event_bindings.viewer["keym"] = "toggle-constrained-fit"
        s.input_event_bindings.viewer["keyw"] = "write-transform"
        if enable_fine_align:
            s.input_event_bindings.viewer["keyf"] = "fine-align"
        s.input_event_bindings.viewer["alt+keyx"] = "delete-nearest-point"
        s.input_event_bindings.viewer["alt+digit1"] = "add-fixed-point"
        s.input_event_bindings.viewer["alt+digit2"] = "add-moving-point"
        s.input_event_bindings.viewer["alt+keya"] = "rot-x-plus-small"
        s.input_event_bindings.viewer["alt+keyq"] = "rot-x-minus-small"
        s.input_event_bindings.viewer["alt+keys"] = "rot-y-plus-small"
        s.input_event_bindings.viewer["alt+keyw"] = "rot-y-minus-small"
        s.input_event_bindings.viewer["alt+keyd"] = "rot-z-plus-small"
        s.input_event_bindings.viewer["alt+keye"] = "rot-z-minus-small"
        s.input_event_bindings.viewer["alt+shift+keya"] = "rot-x-plus-large"
        s.input_event_bindings.viewer["alt+shift+keyq"] = "rot-x-minus-large"
        s.input_event_bindings.viewer["alt+shift+keys"] = "rot-y-plus-large"
        s.input_event_bindings.viewer["alt+shift+keyw"] = "rot-y-minus-large"
        s.input_event_bindings.viewer["alt+shift+keyd"] = "rot-z-plus-large"
        s.input_event_bindings.viewer["alt+shift+keye"] = "rot-z-minus-large"
        s.input_event_bindings.viewer["alt+keyf"] = "flip-x"
        s.input_event_bindings.viewer["alt+keyg"] = "flip-y"
        s.input_event_bindings.viewer["alt+keyh"] = "flip-z"
        s.input_event_bindings.viewer["alt+keyj"] = "trans-x-plus-small"
        s.input_event_bindings.viewer["alt+keyu"] = "trans-x-minus-small"
        s.input_event_bindings.viewer["alt+keyk"] = "trans-y-plus-small"
        s.input_event_bindings.viewer["alt+keyi"] = "trans-y-minus-small"
        s.input_event_bindings.viewer["alt+keyl"] = "trans-z-plus-small"
        s.input_event_bindings.viewer["alt+keyo"] = "trans-z-minus-small"
        s.input_event_bindings.viewer["alt+shift+keyj"] = "trans-x-plus-large"
        s.input_event_bindings.viewer["alt+shift+keyu"] = "trans-x-minus-large"
        s.input_event_bindings.viewer["alt+shift+keyk"] = "trans-y-plus-large"
        s.input_event_bindings.viewer["alt+shift+keyi"] = "trans-y-minus-large"
        s.input_event_bindings.viewer["alt+shift+keyl"] = "trans-z-plus-large"
        s.input_event_bindings.viewer["alt+shift+keyo"] = "trans-z-minus-large"
        s.input_event_bindings.viewer["alt+bracketleft"] = "previous-fixed-point"
        s.input_event_bindings.viewer["alt+bracketright"] = "next-fixed-point"
        s.input_event_bindings.viewer["alt+shift+bracketright"] = "goto-worst-landmark"
        s.input_event_bindings.viewer["shift+keyj"] = "perturb-fixed-point-x-plus"
        s.input_event_bindings.viewer["shift+keyu"] = "perturb-fixed-point-x-minus"
        s.input_event_bindings.viewer["shift+keyk"] = "perturb-fixed-point-y-plus"
        s.input_event_bindings.viewer["shift+keyi"] = "perturb-fixed-point-y-minus"
        s.input_event_bindings.viewer["shift+keyl"] = "perturb-fixed-point-z-plus"
        s.input_event_bindings.viewer["shift+keyo"] = "perturb-fixed-point-z-minus"


def set_initial_transform(
    viewer: neuroglancer.Viewer,
    initial_transform: str,
    invert_initial_transform: bool,
) -> None:
    # Wait a second to make sure the viewer is ready
    time.sleep(1)

    if initial_transform is not None:
        with viewer.txn() as state:
            load_transform(state, initial_transform, invert_initial_transform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed",
        type=str,
        required=True,
        help="Path to fixed volume (local or remote Zarr)",
    )
    parser.add_argument(
        "--moving",
        type=str,
        required=True,
        help="Path or URL to the moving source (local/remote Zarr or VTK mesh)",
    )
    parser.add_argument(
        "--moving-type",
        choices=("auto", "zarr", "mesh"),
        default="auto",
        help="Moving source type. Defaults to auto-detection from the path.",
    )
    parser.add_argument(
        "--fixed-voxel-size",
        type=float,
        help="Voxel size of fixed volume in microns (if not provided, will try to read from metadata.json)",
    )
    parser.add_argument(
        "--moving-voxel-size",
        type=float,
        help="Voxel size of the moving Zarr volume in microns. For mesh sources this is accepted as a compatibility alias for --moving-source-unit-size.",
    )
    parser.add_argument(
        "--moving-source-unit-size",
        type=float,
        help="Native moving-source unit size in microns. Required for mesh moving sources unless --moving-voxel-size is used as an alias.",
    )
    parser.add_argument(
        "--small-rotate-deg",
        type=float,
        default=1.0,
        help="Rotation step in degrees for Alt+<rotation key> (default: 1.0)",
    )
    parser.add_argument(
        "--large-rotate-deg",
        type=float,
        default=90.0,
        help="Rotation step in degrees for Alt+Shift+<rotation key> (default: 90.0)",
    )
    parser.add_argument(
        "--small-translate-voxels",
        type=float,
        default=10.0,
        help="Translation step in voxels for Alt+<translation key> (default: 10.0)",
    )
    parser.add_argument(
        "--large-translate-voxels",
        type=float,
        default=1000.0,
        help="Translation step in voxels for Alt+Shift+<translation key> (default: 1000.0)",
    )
    parser.add_argument(
        "--point-perturb-voxels",
        type=float,
        default=1.0,
        help="Fixed-point perturb step in voxels for Shift+<j/u/k/i/l/o> (default: 1.0)",
    )
    parser.add_argument(
        "--mesh-slice-max-segments",
        type=int,
        default=256,
        help="Maximum number of contour segments to keep in the moving_mesh_slices overlay. Lower values trade detail for responsiveness. Defaults to 256. Use 0 or a negative value to disable downsampling.",
    )
    parser.add_argument(
        "--output-transform",
        type=str,
        help="Path to write the transform to (if not provided, print to stdout)",
    )
    parser.add_argument(
        "--initial-transform",
        type=str,
        help="Path to read an initial transform from (if not provided, no initial transform will be applied)",
    )
    parser.add_argument(
        "--invert-initial-transform",
        action="store_true",
        help="Invert the initial transform before applying it",
    )
    parser.add_argument(
        "--constrained-fit",
        action="store_true",
        help="Start in constrained fit mode (5 DOF + z-flip instead of 12 DOF affine). Toggle at runtime with 'm'.",
    )
    args = parser.parse_args()

    if not sys.flags.interactive:
        print(
            f"Running in non-interactive mode. Use `python -i {Path(__file__).name}` to run in interactive mode (required for neuroglancer)."
        )
        sys.exit(1)

    if args.constrained_fit:
        constrained_fit_mode = True
        print("Starting in CONSTRAINED fit mode (5 DOF + z-flip). Press 'm' to toggle.")

    if args.output_transform is not None:
        if not args.output_transform.endswith(".json"):
            raise ValueError("Output transform path must end with .json")
        if not Path(args.output_transform).parent.exists():
            raise ValueError(
                f"Output transform path {args.output_transform} does not exist"
            )

    fixed_dimensions = get_volume_dimensions(args.fixed, args.fixed_voxel_size)
    mesh_slice_max_segments = args.mesh_slice_max_segments
    moving_source_type = infer_moving_source_type(args.moving, args.moving_type)
    moving_source = build_moving_source_descriptor(
        args.moving,
        moving_source_type,
        args.moving_voxel_size,
        args.moving_source_unit_size,
    )
    scale_factor = moving_source.unit_size_um / fixed_dimensions.voxel_size_um

    if moving_source.source_type == "mesh" and constrained_fit_mode:
        raise ValueError(
            "Constrained fit mode is not supported with mesh moving sources "
            "(mesh landmarks use XYZ order, but the constrained solver assumes ZYX)."
        )

    if moving_source.source_type == "mesh":
        moving_mesh_geometry = read_vtk_mesh_geometry(args.moving)
        moving_mesh_vertices_homogeneous = np.column_stack(
            [
                moving_mesh_geometry.vertices_xyz.astype(np.float32, copy=False),
                np.ones(len(moving_mesh_geometry.vertices_xyz), dtype=np.float32),
            ]
        )
        warm_numba_mesh_slice_kernels()
        mesh_info = get_vtk_mesh_info(args.moving)
        print(
            "Loaded moving mesh: "
            f"{strip_vtk_url_prefix(args.moving)} "
            f"({mesh_info.vertex_count} vertices, {mesh_info.polygon_count} polygons)"
        )
        print(f"Resolved moving mesh URL: {moving_source.layer_url}")
        print(
            "Mesh center in source XYZ coordinates: "
            f"{tuple(round(v, 6) for v in mesh_info.center_xyz)}"
        )
        print(
            "Mesh slice overlay backend: "
            f"{'numba-parallel' if NUMBA_AVAILABLE else 'numpy/python'}"
        )

    viewer = neuroglancer.Viewer()

    add_actions_and_keybinds(
        viewer,
        enable_fine_align=moving_source.source_type == "zarr",
        small_rotate_deg=args.small_rotate_deg,
        large_rotate_deg=args.large_rotate_deg,
        small_translate_voxels=args.small_translate_voxels,
        large_translate_voxels=args.large_translate_voxels,
        point_perturb_voxels=args.point_perturb_voxels,
    )

    init_layers(viewer, args.fixed, fixed_dimensions, moving_source, scale_factor)

    print(f"Opening neuroglancer viewer... at URL: {viewer.get_viewer_url()}")
    webbrowser.open_new(viewer.get_viewer_url())

    set_initial_transform(viewer, args.initial_transform, args.invert_initial_transform)

    if moving_source.source_type == "mesh":
        request_mesh_slice_overlay_update()
        if NUMBA_AVAILABLE:
            viewer.shared_state.add_changed_callback(request_mesh_slice_overlay_update)
        else:
            print(
                "Mesh slice overlay live pan/slice updates are disabled because numba is unavailable in this environment"
            )
