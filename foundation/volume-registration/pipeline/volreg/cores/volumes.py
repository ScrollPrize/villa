"""Reading OME-Zarr volumes: every pyramid level with its scale, and voxel boxes with retries.

From `transform_utils.py` and `blob_snap.py`, unchanged.
"""

import time

import numpy as np
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


# The public stores intermittently answer a chunk request with a server error even though
# the chunk is there and reads fine a moment later, so a read is retried before it counts
# as a failure. Treating those as missing data instead would quietly replace real voxels
# with the fill value, which is far worse than waiting.
PATCH_READ_ATTEMPTS = 3
PATCH_READ_RETRY_DELAY_S = 0.25


def sanity_check_zarr_store(store):
    """Check if the store is a valid OME-ZARR file."""
    try:
        # Basic functionality checks
        assert store is not None
        assert hasattr(store, "keys")
        assert hasattr(store, "attrs")
    except AssertionError as e:
        raise ValueError(f"Invalid OME-ZARR file: {store.path}") from e




def open_volume_levels(path: str) -> list:
    """Open every resolution level of an OME-Zarr volume, with its downsampling factor.

    Returns [(array, scale), ...] finest first, where `scale` is how many level-0 voxels
    one voxel of that level spans. Anisotropic levels are not useful to the callers here
    and are dropped; a store with no readable multiscale metadata comes back as level 0
    alone, which is what the callers did before this existed.
    """
    store = zarr.open(path, mode="r")
    sanity_check_zarr_store(store)
    try:
        datasets = store.attrs["multiscales"][0]["datasets"]
    except (KeyError, IndexError, TypeError):
        return [(store["0"], 1.0)]

    levels = []
    for dataset in datasets:
        scales = [
            transformation["scale"]
            for transformation in dataset.get("coordinateTransformations", [])
            if transformation.get("type") == "scale"
        ]
        if not scales or len(set(scales[0])) != 1:
            continue
        try:
            levels.append((store[dataset["path"]], float(scales[0][0])))
        except KeyError:
            continue
    if not levels:
        return [(store["0"], 1.0)]
    return sorted(levels, key=lambda level: level[1])




def read_patch(array, lo, hi, attempts: int = PATCH_READ_ATTEMPTS) -> np.ndarray:
    """Read a voxel box, retrying the transient failures the remote stores produce."""
    last_error = None
    for attempt in range(attempts):
        try:
            return np.asarray(array[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        except Exception as e:  # noqa: BLE001 - any read failure is worth one more try
            last_error = e
            if attempt + 1 < attempts:
                time.sleep(PATCH_READ_RETRY_DELAY_S * (attempt + 1))
    raise last_error

