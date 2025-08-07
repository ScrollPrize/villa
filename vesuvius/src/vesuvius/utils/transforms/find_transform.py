"""Find the transform between two provided volumes."""

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Optional, NamedTuple
from urllib.parse import urljoin, urlparse
import webbrowser

import neuroglancer
import numpy as np
import requests
import zarr


class Dimensions(NamedTuple):
    """Structure for volume dimensions with x, y, z coordinates and voxel size."""

    voxels_x: int
    voxels_y: int
    voxels_z: int
    voxel_size_um: float


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
    """Get volume dimensions (from Zarr) and voxel size (from metadata.json if it exists, otherwise use provided value).

    If both are provided, make sure they are the same.
    """

    # Get volume dimensions in voxels
    with zarr.open(path, mode="r") as store:
        sanity_check_zarr_store(store)
        voxels_x, voxels_y, voxels_z = store["0"].shape

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


def add_moving_and_fixed_layers(
    state: neuroglancer.ViewerState,
    fixed_path: str,
    moving_path: str,
    scale_factor: float,
):
    # Some unitless dimensions for now
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"],
        units="",
        scales=[1, 1, 1],
    )

    # Open fixed volume
    fixed_source = neuroglancer.LayerDataSource(
        url=f"zarr://{fixed_path}",
        transform=neuroglancer.CoordinateSpaceTransform(
            output_dimensions=dimensions,
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

    moving_source = neuroglancer.LayerDataSource(
        url=f"zarr://{moving_path}",
        transform=neuroglancer.CoordinateSpaceTransform(
            output_dimensions=dimensions,
            matrix=[
                [scale_factor, 0, 0, 0],
                [0, scale_factor, 0, 0],
                [0, 0, scale_factor, 0],
            ],
        ),
    )
    state.layers.append(
        name="moving",
        layer=neuroglancer.ImageLayer(
            source=moving_source,
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
            state.layers["moving"].shader = MAGENTA_SHADER


def make_rotation_matrix(axis: str, angle_deg: float):
    """Make a rotation matrix for the given axis and angle."""
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    if axis == "x":
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
    elif axis == "z":
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
    if axis == "x":
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
    elif axis == "z":
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


def apply_matrix_to_centered(
    state: neuroglancer.ViewerState,
    transform_matrix: np.ndarray,
    volume_dimensions: Dimensions,
):
    """Apply a transformation matrix to a layer, centered on the volume's center.

    Args:
        state: The neuroglancer viewer state
        layer_name: Name of the layer to transform
        transform_matrix: 4x4 homogeneous transformation matrix to apply
        volume_dimensions: Dimensions of the volume being transformed
    """
    original_matrix = state.layers["moving"].layer.source[0].transform.matrix
    # Add homogeneous coordinate
    original_matrix = np.concatenate([original_matrix, [[0, 0, 0, 1]]], axis=0)

    # Current position of volume center in fixed space
    cx, cy, cz, _ = original_matrix @ np.array(
        [
            volume_dimensions.voxels_x / 2,
            volume_dimensions.voxels_y / 2,
            volume_dimensions.voxels_z / 2,
            1,
        ]
    )

    # Center volume about origin
    translate_to_origin_mat = np.array(
        [
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 1, -cz],
            [0, 0, 0, 1],
        ]
    )
    # Translate back to original position
    translate_back_mat = np.array(
        [
            [1, 0, 0, cx],
            [0, 1, 0, cy],
            [0, 0, 1, cz],
            [0, 0, 0, 1],
        ]
    )

    matrix = (
        translate_back_mat
        @ transform_matrix
        @ translate_to_origin_mat
        @ original_matrix
    )
    # Remove homogeneous coordinate
    matrix = matrix[:-1, :]
    # Set new transform
    state.layers["moving"].layer.source[0].transform.matrix = matrix


def _make_rotate_command(axis: str, angle_deg: float):
    def handler(_):
        with viewer.txn() as state:
            rotate_mat = make_rotation_matrix(axis, angle_deg)
            apply_matrix_to_centered(state, rotate_mat, moving_dimensions)

    return handler


def _make_flip_command(axis: str):
    def handler(_):
        with viewer.txn() as state:
            flip_mat = make_flip_matrix(axis)
            apply_matrix_to_centered(state, flip_mat, moving_dimensions)

    return handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=str, required=True)
    parser.add_argument("--moving", type=str, required=True)
    parser.add_argument("--fixed-voxel-size", type=float)
    parser.add_argument("--moving-voxel-size", type=float)
    args = parser.parse_args()

    if not sys.flags.interactive:
        print(
            f"Running in non-interactive mode. Use `python -i {Path(__file__).name}` to run in interactive mode (required for neuroglancer)."
        )
        sys.exit(1)

    fixed_dimensions = get_volume_dimensions(args.fixed, args.fixed_voxel_size)
    moving_dimensions = get_volume_dimensions(args.moving, args.moving_voxel_size)
    scale_factor = moving_dimensions.voxel_size_um / fixed_dimensions.voxel_size_um

    viewer = neuroglancer.Viewer()

    viewer.actions.add("toggle-color", toggle_color)
    viewer.actions.add("rotate-x-plus-5", _make_rotate_command("x", 5))
    viewer.actions.add("rotate-x-minus-5", _make_rotate_command("x", -5))
    viewer.actions.add("rotate-y-plus-5", _make_rotate_command("y", 5))
    viewer.actions.add("rotate-y-minus-5", _make_rotate_command("y", -5))
    viewer.actions.add("rotate-z-plus-5", _make_rotate_command("z", 5))
    viewer.actions.add("rotate-z-minus-5", _make_rotate_command("z", -5))
    viewer.actions.add("rotate-x-plus-90", _make_rotate_command("x", 90))
    viewer.actions.add("rotate-x-minus-90", _make_rotate_command("x", -90))
    viewer.actions.add("rotate-y-plus-90", _make_rotate_command("y", 90))
    viewer.actions.add("rotate-y-minus-90", _make_rotate_command("y", -90))
    viewer.actions.add("rotate-z-plus-90", _make_rotate_command("z", 90))
    viewer.actions.add("rotate-z-minus-90", _make_rotate_command("z", -90))
    viewer.actions.add("flip-x", _make_flip_command("x"))
    viewer.actions.add("flip-y", _make_flip_command("y"))
    viewer.actions.add("flip-z", _make_flip_command("z"))

    with viewer.config_state.txn() as s:
        s.input_event_bindings.viewer["keyc"] = "toggle-color"
        s.input_event_bindings.viewer["alt+keya"] = "rotate-x-plus-5"
        s.input_event_bindings.viewer["alt+keyq"] = "rotate-x-minus-5"
        s.input_event_bindings.viewer["alt+keys"] = "rotate-y-plus-5"
        s.input_event_bindings.viewer["alt+keyw"] = "rotate-y-minus-5"
        s.input_event_bindings.viewer["alt+keyd"] = "rotate-z-plus-5"
        s.input_event_bindings.viewer["alt+keye"] = "rotate-z-minus-5"
        s.input_event_bindings.viewer["alt+shift+keya"] = "rotate-x-plus-90"
        s.input_event_bindings.viewer["alt+shift+keyq"] = "rotate-x-minus-90"
        s.input_event_bindings.viewer["alt+shift+keys"] = "rotate-y-plus-90"
        s.input_event_bindings.viewer["alt+shift+keyw"] = "rotate-y-minus-90"
        s.input_event_bindings.viewer["alt+shift+keyd"] = "rotate-z-plus-90"
        s.input_event_bindings.viewer["alt+shift+keye"] = "rotate-z-minus-90"
        s.input_event_bindings.viewer["alt+keyf"] = "flip-x"
        s.input_event_bindings.viewer["alt+keyg"] = "flip-y"
        s.input_event_bindings.viewer["alt+keyh"] = "flip-z"

    with viewer.txn() as state:
        add_moving_and_fixed_layers(state, args.fixed, args.moving, scale_factor)

    webbrowser.open_new(viewer.get_viewer_url())

    # Buttons or command line args to do basic rotations, flips, moves
    # Allow clicking to set points
    # Have some mechanism to have current active layer
    # Once enough points: find affine transform
    # Apply affine transform in real time to the moving layer
    # Save the affine transform
    # Three column layout? Maybe not.
