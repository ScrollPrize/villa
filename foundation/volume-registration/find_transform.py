"""Find the transform between two provided volumes.

Note that neuroglancer uses ZYX coordinates, so that convention is followed for neuroglancer-related bits (most of this file).
Elsewhere, for example writing to a JSON file, we use XYZ coordinates.
"""

from typing import Optional
from pathlib import Path
import argparse
import sys
import webbrowser

import neuroglancer
import numpy as np
import time

from registration import align_zarrs
from transform_utils import (
    Dimensions,
    get_volume_dimensions,
    invert_affine_matrix,
    fit_affine_transform_from_points,
    matrix_swap_xyz_zyx,
    points_swap_xyz_zyx,
    read_transform_json,
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

FIXED_POINTS_LAYER_STR = "fixed_points"
MOVING_POINTS_LAYER_STR = "moving_points"

UNITLESS_DIMENSIONS = neuroglancer.CoordinateSpace(
    names=["z", "y", "x"],
    units="",
    scales=[1, 1, 1],
)


def init_volume_layers(
    viewer: neuroglancer.Viewer,
    fixed_path: str,
    moving_path: str,
    scale_factor: float,
) -> None:
    with viewer.txn() as state:
        # Open fixed volume
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

        moving_source = neuroglancer.LayerDataSource(
            url=f"zarr://{moving_path}",
            transform=neuroglancer.CoordinateSpaceTransform(
                output_dimensions=UNITLESS_DIMENSIONS,
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
    volume_dimensions: Dimensions,
):
    """Apply a transformation matrix to a layer, centered on the volume's center.

    Args:
        state: The neuroglancer viewer state
        layer_name: Name of the layer to transform
        transform_matrix: 4x4 homogeneous transformation matrix to apply
        volume_dimensions: Dimensions of the volume being transformed
    """
    original_matrix = get_current_transform(state)

    # Current position of volume center in fixed space
    cz, cy, cx, _ = original_matrix @ np.array(
        [
            volume_dimensions.voxels_z / 2,
            volume_dimensions.voxels_y / 2,
            volume_dimensions.voxels_x / 2,
            1,
        ]
    )

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


def load_transform(
    viewer_state: neuroglancer.ViewerState,
    input_path: str,
    invert_initial_transform: bool,
) -> None:
    """Read a transform from a file. If provided, also loads landmarks into viewer state."""
    matrix_xyz, fixed_landmarks_xyz, moving_landmarks_xyz = read_transform_json(
        input_path, invert_initial_transform
    )

    # Convert to ZYX for neuroglancer
    matrix = matrix_swap_xyz_zyx(matrix_xyz)
    fixed_landmarks = points_swap_xyz_zyx(fixed_landmarks_xyz)
    moving_landmarks = points_swap_xyz_zyx(moving_landmarks_xyz)

    set_current_transform(viewer_state, matrix)

    # Load landmarks into layers using the same logic as interactive point adding
    for point in fixed_landmarks:
        add_point_from_coords(viewer_state, point, "fixed")

    for point in moving_landmarks:
        add_point_from_coords(viewer_state, point, "moving")


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

    # Convert to XYZ coordinates for the schema
    xyz_matrix = matrix_swap_xyz_zyx(matrix)
    xyz_moving_landmarks = points_swap_xyz_zyx(moving_landmarks)
    xyz_fixed_landmarks = points_swap_xyz_zyx(fixed_landmarks)

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
    # Remove homogeneous coordinate
    transform = transform[:-1, :]
    state.layers["moving"].layer.source[0].transform.matrix = transform

    # Also update moving_points layer if it exists
    if MOVING_POINTS_LAYER_STR in state.layers:
        state.layers[MOVING_POINTS_LAYER_STR].layer.source[
            0
        ].transform.matrix = transform


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
    with viewer.txn() as state:
        current_transform = get_current_transform(state)

        # Run alignment
        print("Running zarr alignment...")
        refined_transform = align_zarrs(args.fixed, args.moving, current_transform)
        print(f"Refined transform: {refined_transform}")

        # Apply the refined transform
        set_current_transform(state, refined_transform)
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
    """Update transform if there are 4+ matching point pairs."""
    if (
        FIXED_POINTS_LAYER_STR in state.layers
        and MOVING_POINTS_LAYER_STR in state.layers
    ):
        fixed_annotations = state.layers[FIXED_POINTS_LAYER_STR].layer.annotations
        moving_annotations = state.layers[MOVING_POINTS_LAYER_STR].layer.annotations

        if (
            len(fixed_annotations) == len(moving_annotations)
            and len(fixed_annotations) >= 4
        ):
            # Extract point coordinates from annotations
            fixed_points_list = [list(ann.point) for ann in fixed_annotations]
            moving_points_list = [list(ann.point) for ann in moving_annotations]

            transform = fit_affine_transform_from_points(
                fixed_points_list, moving_points_list
            )
            if transform is not None:
                set_current_transform(state, transform)
                print(
                    f"Automatically updated transform from {len(fixed_annotations)} point pairs"
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


def write_current_transform(_):
    """Write the current transform and print the shareable URL."""
    with viewer.txn() as state:
        save_current_transform(state, args.output_transform, args.fixed)


def add_actions_and_keybinds(viewer: neuroglancer.Viewer) -> None:
    SMALL_ROTATE_STEP = 1
    LARGE_ROTATE_STEP = 90
    TRANSLATE_STEP = 10

    viewer.actions.add("toggle-color", toggle_color)
    viewer.actions.add("write-transform", write_current_transform)
    viewer.actions.add("fine-align", fine_align)
    viewer.actions.add("add-fixed-point", add_fixed_point)
    viewer.actions.add("add-moving-point", add_moving_point)
    viewer.actions.add("delete-nearest-point", delete_nearest_point)
    viewer.actions.add("previous-fixed-point", navigate_to_previous_fixed_point)
    viewer.actions.add("next-fixed-point", navigate_to_next_fixed_point)
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
    viewer.actions.add("perturb-fixed-point-x-plus", _make_point_perturber("x", 1))
    viewer.actions.add("perturb-fixed-point-x-minus", _make_point_perturber("x", -1))
    viewer.actions.add("perturb-fixed-point-y-plus", _make_point_perturber("y", 1))
    viewer.actions.add("perturb-fixed-point-y-minus", _make_point_perturber("y", -1))
    viewer.actions.add("perturb-fixed-point-z-plus", _make_point_perturber("z", 1))
    viewer.actions.add("perturb-fixed-point-z-minus", _make_point_perturber("z", -1))

    with viewer.config_state.txn() as s:
        s.input_event_bindings.viewer["keyc"] = "toggle-color"
        s.input_event_bindings.viewer["keyw"] = "write-transform"
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
        s.input_event_bindings.viewer["alt+shift+keyj"] = "perturb-fixed-point-x-plus"
        s.input_event_bindings.viewer["alt+shift+keyu"] = "perturb-fixed-point-x-minus"
        s.input_event_bindings.viewer["alt+shift+keyk"] = "perturb-fixed-point-y-plus"
        s.input_event_bindings.viewer["alt+shift+keyi"] = "perturb-fixed-point-y-minus"
        s.input_event_bindings.viewer["alt+shift+keyl"] = "perturb-fixed-point-z-plus"
        s.input_event_bindings.viewer["alt+shift+keyo"] = "perturb-fixed-point-z-minus"
        s.input_event_bindings.viewer["alt+bracketleft"] = "previous-fixed-point"
        s.input_event_bindings.viewer["alt+bracketright"] = "next-fixed-point"


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
    parser.add_argument(
        "--invert-initial-transform",
        action="store_true",
        help="Invert the initial transform before applying it",
    )
    args = parser.parse_args()

    if not sys.flags.interactive:
        print(
            f"Running in non-interactive mode. Use `python -i {Path(__file__).name}` to run in interactive mode (required for neuroglancer)."
        )
        sys.exit(1)

    if args.output_transform is not None:
        if not args.output_transform.endswith(".json"):
            raise ValueError("Output transform path must end with .json")
        if not Path(args.output_transform).parent.exists():
            raise ValueError(
                f"Output transform path {args.output_transform} does not exist"
            )

    fixed_dimensions = get_volume_dimensions(args.fixed, args.fixed_voxel_size)
    moving_dimensions = get_volume_dimensions(args.moving, args.moving_voxel_size)
    scale_factor = moving_dimensions.voxel_size_um / fixed_dimensions.voxel_size_um

    viewer = neuroglancer.Viewer()

    add_actions_and_keybinds(viewer)

    init_volume_layers(viewer, args.fixed, args.moving, scale_factor)

    webbrowser.open_new(viewer.get_viewer_url())

    set_initial_transform(viewer, args.initial_transform, args.invert_initial_transform)
