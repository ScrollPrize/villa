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




def by_z_range(array):
    """Whether a box of `array` can be read as byte ranges: a zarr v2 array behind fsspec
    whose chunks are stored raw (no compressor, no filter) in C order. Then the planes
    z0..z1 of a chunk are one contiguous run of bytes."""
    store = getattr(array, "store", None)
    return (isinstance(store, zarr.storage.FSStore) and array._compressor is None
            and not array._filters and array.order == "C")


def read_z_ranges(array, lo, hi):
    """A voxel box read with one ranged request per chunk it touches, for only the planes
    the box needs from that chunk. A patch's fine box spans about ten 128^3 chunks, so a
    whole-chunk read fetches four to seven times the box; this fetches about half of
    that. A missing chunk reads as the fill value, as zarr reads it."""
    lo, hi = np.asarray(lo, int), np.asarray(hi, int)
    chunk = np.asarray(array.chunks)
    plane = int(np.prod(chunk[1:])) * array.dtype.itemsize
    store = array.store
    first, last = lo // chunk, -(-hi // chunk)
    out = np.full(tuple(hi - lo), array.fill_value, array.dtype)
    places, urls, starts, ends = [], [], [], []
    for index in np.ndindex(*(last - first)):
        index = first + np.asarray(index)
        corner = index * chunk
        a, b = np.maximum(lo, corner), np.minimum(hi, corner + chunk)
        key = store._normalize_key(array._chunk_key(tuple(int(i) for i in index)))
        urls.append(f"{store.path.rstrip('/')}/{key}")
        starts.append(int(a[0] - corner[0]) * plane)
        ends.append(int(b[0] - corner[0]) * plane)
        places.append((a, b, corner))
    found = store.fs.cat_ranges(urls, starts, ends, on_error="return")
    for (a, b, corner), data, start, end in zip(places, found, starts, ends):
        if isinstance(data, FileNotFoundError):
            continue
        if isinstance(data, BaseException):
            raise data
        if len(data) != end - start:
            raise IOError(f"asked for {end - start} bytes, got {len(data)}")
        planes = np.frombuffer(data, array.dtype).reshape(-1, *chunk[1:])
        out[tuple(slice(p - q, r - q) for p, q, r in zip(a, lo, b))] = \
            planes[:, a[1] - corner[1]:b[1] - corner[1], a[2] - corner[2]:b[2] - corner[2]]
    return out


def read_patch(array, lo, hi, attempts: int = PATCH_READ_ATTEMPTS) -> np.ndarray:
    """Read a voxel box, retrying the transient failures the remote stores produce. A box
    inside an array stored raw is read by z-range (`read_z_ranges`); anything else, and a
    box reaching past the array, through zarr."""
    ranged = (by_z_range(array) and (np.asarray(lo) >= 0).all()
              and (np.asarray(hi) <= np.asarray(array.shape)).all())
    last_error = None
    for attempt in range(attempts):
        try:
            if ranged:
                return read_z_ranges(array, lo, hi)
            return np.asarray(array[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        except Exception as e:  # noqa: BLE001 - any read failure is worth one more try
            last_error = e
            if attempt + 1 < attempts:
                time.sleep(PATCH_READ_RETRY_DELAY_S * (attempt + 1))
    raise last_error

