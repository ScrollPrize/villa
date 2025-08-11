"""Find the transform between two provided volumes."""

import argparse
from pathlib import Path
import sys
import webbrowser

import neuroglancer
import numpy as np

from registration import align_zarrs
from transform_utils import Dimensions, get_volume_dimensions


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


def add_moving_and_fixed_layers(
    viewer: neuroglancer.Viewer,
    fixed_path: str,
    moving_path: str,
    scale_factor: float,
) -> None:
    with viewer.txn() as state:
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
                # Scale the moving volume to match the fixed volume
                # If an initial transform is provided, it will be applied after this and override it
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


def make_translate_matrix(axis: str, amount: float):
    """Make a translation matrix for the given axis and amount."""
    if axis == "x":
        return np.array([[1, 0, 0, amount], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    elif axis == "y":
        return np.array([[1, 0, 0, 0], [0, 1, 0, amount], [0, 0, 1, 0], [0, 0, 0, 1]])
    elif axis == "z":
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, amount], [0, 0, 0, 1]])
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


def _make_rotator(axis: str, angle_deg: float):
    def handler(_):
        with viewer.txn() as state:
            rotate_mat = make_rotation_matrix(axis, angle_deg)
            apply_matrix_to_centered(state, rotate_mat, moving_dimensions)

    return handler


def _make_flipper(axis: str):
    def handler(_):
        with viewer.txn() as state:
            flip_mat = make_flip_matrix(axis)
            apply_matrix_to_centered(state, flip_mat, moving_dimensions)

    return handler


def _make_translator(axis: str, amount: float):
    def handler(_):
        with viewer.txn() as state:
            translate_mat = make_translate_matrix(axis, amount)
            apply_matrix_to_centered(state, translate_mat, moving_dimensions)

    return handler


def write_transform_to_file(transform: np.ndarray, output_path: str) -> None:
    """Write a transform to a file."""
    np.savetxt(output_path, transform)


def read_transform_from_file(input_path: str) -> np.ndarray:
    """Read a transform from a file."""
    with open(input_path, "r") as f:
        return np.loadtxt(f, dtype=np.float64)


def fine_align(_):
    """Run zarr alignment and update the moving layer with the result."""
    with viewer.txn() as state:
        current_transform = state.layers["moving"].layer.source[0].transform.matrix
        # Add homogeneous coordinate
        current_transform = np.concatenate([current_transform, [[0, 0, 0, 1]]], axis=0)

        # Run alignment
        print("Running zarr alignment...")
        refined_transform = align_zarrs(args.fixed, args.moving, current_transform)

        # Apply the refined transform
        # Remove homogeneous coordinate
        refined_transform = refined_transform[:-1, :]
        state.layers["moving"].layer.source[0].transform.matrix = refined_transform
        print("Alignment complete - transform updated")


def write_current_transform(_):
    """Write the current transform and print the shareable URL."""
    with viewer.txn() as state:
        transform = state.layers["moving"].layer.source[0].transform.matrix
        # Add homogeneous coordinate
        transform = np.concatenate([transform, [[0, 0, 0, 1]]], axis=0)
        # Write to file
        if args.output_transform is None:
            print("--output-transform not provided, printing transform to stdout:")
            print(transform)
            print("To save to file, use --output-transform <path>")
        else:
            print(f"Writing transform to {args.output_transform}")
            write_transform_to_file(transform, args.output_transform)

        # In either case print the state URL
        print(
            f"Shareable URL: https://neuroglancer-demo.appspot.com/#!{neuroglancer.url_state.to_url_fragment(state)}"
        )


def add_actions_and_keybinds(viewer: neuroglancer.Viewer) -> None:
    SMALL_ROTATE_STEP = 1
    LARGE_ROTATE_STEP = 90
    TRANSLATE_STEP = 10

    viewer.actions.add("toggle-color", toggle_color)
    viewer.actions.add("write-transform", write_current_transform)
    viewer.actions.add("fine-align", fine_align)
    viewer.actions.add("rot-x-plus-small", _make_rotator("x", SMALL_ROTATE_STEP))
    viewer.actions.add("rot-x-minus-small", _make_rotator("x", -SMALL_ROTATE_STEP))
    viewer.actions.add("rot-y-plus-small", _make_rotator("y", SMALL_ROTATE_STEP))
    viewer.actions.add("rot-y-minus-small", _make_rotator("y", -SMALL_ROTATE_STEP))
    viewer.actions.add("rot-z-plus-small", _make_rotator("z", SMALL_ROTATE_STEP))
    viewer.actions.add("rot-z-minus-small", _make_rotator("z", -SMALL_ROTATE_STEP))
    viewer.actions.add("rot-x-plus-large", _make_rotator("x", LARGE_ROTATE_STEP))
    viewer.actions.add("rot-x-minus-large", _make_rotator("x", -LARGE_ROTATE_STEP))
    viewer.actions.add("rot-y-plus-large", _make_rotator("y", LARGE_ROTATE_STEP))
    viewer.actions.add("rot-y-minus-large", _make_rotator("y", -LARGE_ROTATE_STEP))
    viewer.actions.add("rot-z-plus-large", _make_rotator("z", LARGE_ROTATE_STEP))
    viewer.actions.add("rot-z-minus-large", _make_rotator("z", -LARGE_ROTATE_STEP))
    viewer.actions.add("flip-x", _make_flipper("x"))
    viewer.actions.add("flip-y", _make_flipper("y"))
    viewer.actions.add("flip-z", _make_flipper("z"))
    viewer.actions.add("trans-x-plus-small", _make_translator("x", TRANSLATE_STEP))
    viewer.actions.add("trans-x-minus-small", _make_translator("x", -TRANSLATE_STEP))
    viewer.actions.add("trans-y-plus-small", _make_translator("y", TRANSLATE_STEP))
    viewer.actions.add("trans-y-minus-small", _make_translator("y", -TRANSLATE_STEP))
    viewer.actions.add("trans-z-plus-small", _make_translator("z", TRANSLATE_STEP))
    viewer.actions.add("trans-z-minus-small", _make_translator("z", -TRANSLATE_STEP))

    with viewer.config_state.txn() as s:
        s.input_event_bindings.viewer["keyc"] = "toggle-color"
        s.input_event_bindings.viewer["keyw"] = "write-transform"
        s.input_event_bindings.viewer["keyf"] = "fine-align"
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


def set_initial_transform(viewer: neuroglancer.Viewer, initial_transform: str) -> None:
    if initial_transform is not None:
        with viewer.txn() as state:
            matrix = read_transform_from_file(initial_transform)
            # Remove homogeneous coordinate
            matrix = matrix[:-1, :]
            state.layers["moving"].layer.source[0].transform.matrix = matrix


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
        help="Path to moving volume (local or remote Zarr)",
    )
    parser.add_argument(
        "--fixed-voxel-size",
        type=float,
        help="Voxel size of fixed volume in microns (if not provided, will try to read from metadata.json)",
    )
    parser.add_argument(
        "--moving-voxel-size",
        type=float,
        help="Voxel size of moving volume in microns (if not provided, will try to read from metadata.json)",
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

    add_actions_and_keybinds(viewer)

    add_moving_and_fixed_layers(viewer, args.fixed, args.moving, scale_factor)

    set_initial_transform(viewer, args.initial_transform)

    webbrowser.open_new(viewer.get_viewer_url())

    # TODO
    # Create registration file with stub registration method and map it in this file to a keybind
    # Figure out what resolutions are available
    # If low enough, try SimpleITK registration using lowest resolution
    # Then try with higher and/or chunking?
