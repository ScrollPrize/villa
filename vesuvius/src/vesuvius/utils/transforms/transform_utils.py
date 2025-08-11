import json
from pathlib import Path
from typing import Optional, NamedTuple
from urllib.parse import urljoin, urlparse

import requests
import zarr


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