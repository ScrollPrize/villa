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


def check_ome_zarr(path: str) -> bool:
    """Check if the path is a valid OME-ZARR file."""
    try:
        with zarr.open(path, mode="r") as store:
            # Basic functionality checks
            assert store is not None
            assert hasattr(store, "keys")
            assert hasattr(store, "attrs")
            return True
    except Exception as e:
        print(f"Error opening {path}: {e}")
        return False


def get_dimensions(
    path: str, provided_voxel_size: Optional[float] = None
) -> Dimensions:
    """Get voxel size from metadata.json if it exists, otherwise use provided value.

    If both are provided, make sure they are the same."""

    # Get volume dimensions in voxels
    with zarr.open(path, mode="r") as store:
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

    # Make sure Zarrs are openable
    if not check_ome_zarr(args.fixed):
        raise ValueError(f"Invalid OME-ZARR file: {args.fixed}")
    if not check_ome_zarr(args.moving):
        raise ValueError(f"Invalid OME-ZARR file: {args.moving}")

    # Establish voxel size for each volume
    fixed_dimensions = get_dimensions(args.fixed, args.fixed_voxel_size)
    moving_dimensions = get_dimensions(args.moving, args.moving_voxel_size)
    scale_factor = moving_dimensions.voxel_size_um / fixed_dimensions.voxel_size_um

    print(f"Fixed voxel size (um): {fixed_dimensions.voxel_size_um}")
    print(f"Moving voxel size (um): {moving_dimensions.voxel_size_um}")

    viewer = neuroglancer.Viewer()

    def toggle_color(_):
        with viewer.txn() as state:
            if "vec3" in state.layers["fixed"].shader:
                state.layers["fixed"].shader = ""
                state.layers["moving"].shader = ""
            else:
                state.layers["fixed"].shader = GREEN_SHADER
                state.layers["moving"].shader = MAGENTA_SHADER

    def rotate_90(_):
        # TODO: make this work for any axis based on which viewport is active
        with viewer.txn() as state:
            matrix = state.layers["moving"].layer.source[0].transform.matrix
            # add homogeneous coordinate
            matrix = np.concatenate([matrix, [[0, 0, 0, 1]]], axis=0)
            # TODO LEFT OFFincorporate translation to rotate about center
            # rotate 90 degrees around z axis
            matrix = np.matmul(
                matrix,
                np.array(
                    [
                        [0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                ),
            )
            # remove homogeneous coordinate
            matrix = matrix[:-1, :]
            # set new transform
            state.layers["moving"].layer.source[0].transform.matrix = matrix

    viewer.actions.add("toggle-color", toggle_color)
    viewer.actions.add("rotate-90", rotate_90)
    with viewer.config_state.txn() as s:
        s.input_event_bindings.viewer["keyc"] = "toggle-color"
        s.input_event_bindings.viewer["alt+keyr"] = "rotate-90"

    with viewer.txn() as state:
        # Some unitless dimensions for now
        dimensions = neuroglancer.CoordinateSpace(
            names=["x", "y", "z"],
            units="",
            scales=[1, 1, 1],
        )

        # Open fixed volume
        fixed_source = neuroglancer.LayerDataSource(
            url=f"zarr://{args.fixed}",
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

        # Open moving volume
        moving_shape = zarr.open(args.moving, mode="r")["0"].shape
        z, y, x = moving_shape

        moving_source = neuroglancer.LayerDataSource(
            url=f"zarr://{args.moving}",
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

    # Open viewer in browser
    webbrowser.open_new(viewer.get_viewer_url())

    # Buttons or command line args to do basic rotations, flips, moves
    # Allow clicking to set points
    # Have some mechanism to have current active layer
    # Once enough points: find affine transform
    # Apply affine transform in real time to the moving layer
    # Save the affine transform
    # Three column layout? Maybe not.
