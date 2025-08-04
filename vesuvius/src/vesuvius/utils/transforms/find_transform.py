"""Find the transform between two provided volumes."""

import argparse
import json
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
import zarr


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


def get_voxel_size(path: str, provided_voxel_size: Optional[float] = None) -> float:
    """Get voxel size from metadata.json if it exists, otherwise use provided value.

    If both are provided, make sure they are the same."""
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
        return metadata_voxel_size_um
    else:
        assert (
            provided_voxel_size is not None
        ), f"No metadata.json found at {metadata_path} and no voxel size provided directly"
        return provided_voxel_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=str, required=True)
    parser.add_argument("--moving", type=str, required=True)
    parser.add_argument("--fixed-voxel-size", type=float)
    parser.add_argument("--moving-voxel-size", type=float)
    args = parser.parse_args()

    # Make sure Zarrs are openable
    if not check_ome_zarr(args.fixed):
        raise ValueError(f"Invalid OME-ZARR file: {args.fixed}")
    if not check_ome_zarr(args.moving):
        raise ValueError(f"Invalid OME-ZARR file: {args.moving}")

    # Establish voxel size for each volume
    fixed_voxel_size = get_voxel_size(args.fixed, args.fixed_voxel_size)
    moving_voxel_size = get_voxel_size(args.moving, args.moving_voxel_size)

    print(f"Fixed voxel size: {fixed_voxel_size}")
    print(f"Moving voxel size: {moving_voxel_size}")

    # fixed_vol = zarr.open(args.fixed, mode="r")
    # moving_vol = zarr.open(args.moving, mode="r")
