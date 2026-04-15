"""
Fast occupancy bitmap for zarr v2 arrays.

Given a zarr array whose chunks are laid out as `<dim0>.<dim1>.<...>` files,
build a boolean numpy array over the chunk grid marking which chunks actually
exist on disk/S3. This lets callers pre-filter patches that fall entirely in
empty regions of a sparse input volume without ever reading the underlying data.

Also provides a disk cache keyed by the array's `.zarray` ETag (S3) or mtime
(local), so repeat runs pay no listing cost after the first.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import fsspec
import numpy as np
from tqdm.auto import tqdm

from vesuvius.utils.k8s import get_tqdm_kwargs


SIDECAR_NAME = ".chunk_occupancy.npz"


def _local_cache_dir() -> Path:
    d = Path(os.environ.get("VESUVIUS_CACHE_DIR", Path.home() / ".cache" / "vesuvius")) / "chunk_occupancy"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _local_cache_path(array_url: str) -> Path:
    key = hashlib.sha1(array_url.encode()).hexdigest()
    return _local_cache_dir() / f"{key}.npz"


def _sidecar_url(array_url: str) -> str:
    return array_url.rstrip("/") + "/" + SIDECAR_NAME


def _zarray_signature(array_url: str) -> Optional[str]:
    """Return a short stable signature for the array's .zarray (ETag on S3, mtime+size locally)."""
    zarray_url = array_url.rstrip("/") + "/.zarray"
    try:
        if array_url.startswith("s3://"):
            fs = fsspec.filesystem("s3", anon=False)
            info = fs.info(zarray_url.replace("s3://", ""))
            etag = info.get("ETag") or info.get("etag")
            if etag:
                return str(etag).strip('"')
            return f"size={info.get('size')}"
        else:
            st = os.stat(zarray_url)
            return f"{int(st.st_mtime_ns)}-{st.st_size}"
    except Exception:
        return None


def _serialize_bitmap(bitmap: np.ndarray, sig: str) -> bytes:
    """Encode the occupancy bitmap + signature into a single .npz byte blob."""
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, occupancy=bitmap, signature=np.array(sig))
    return buf.getvalue()


def _deserialize_bitmap(blob: bytes, expected_sig: str, expected_grid: Tuple[int, ...]) -> Optional[np.ndarray]:
    import io

    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            bitmap = data["occupancy"]
            sig = str(data["signature"].item()) if "signature" in data.files else ""
    except Exception as e:
        warnings.warn(f"Failed to decode chunk occupancy cache: {e}")
        return None
    if sig != expected_sig:
        return None
    if tuple(bitmap.shape) != tuple(expected_grid):
        return None
    return bitmap.astype(bool, copy=False)


def _load_cached(
    array_url: str, sig: str, expected_grid: Tuple[int, ...]
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Try sidecar next to the array first (shared across partitions), then local cache.

    Returns (bitmap, source) where `source` is one of:
      - "sidecar:<url>"  sidecar file next to the zarr array
      - "local:<path>"   per-host local fallback cache
      - None             no cache hit (bitmap is also None)
    """
    sidecar = _sidecar_url(array_url)
    try:
        if array_url.startswith("s3://"):
            fs = fsspec.filesystem("s3", anon=False)
            key = sidecar.replace("s3://", "")
            if fs.exists(key):
                with fs.open(key, "rb") as f:
                    blob = f.read()
                bitmap = _deserialize_bitmap(blob, sig, expected_grid)
                if bitmap is not None:
                    return bitmap, f"sidecar:{sidecar}"
        else:
            if os.path.exists(sidecar):
                with open(sidecar, "rb") as f:
                    blob = f.read()
                bitmap = _deserialize_bitmap(blob, sig, expected_grid)
                if bitmap is not None:
                    return bitmap, f"sidecar:{sidecar}"
    except Exception as e:
        warnings.warn(f"Failed to load sidecar chunk occupancy from {sidecar}: {e}")

    local = _local_cache_path(array_url)
    if local.exists():
        try:
            with open(local, "rb") as f:
                blob = f.read()
            bitmap = _deserialize_bitmap(blob, sig, expected_grid)
            if bitmap is not None:
                return bitmap, f"local:{local}"
        except Exception as e:
            warnings.warn(f"Failed to load local chunk occupancy cache at {local}: {e}")
    return None, None


def _save_cached(array_url: str, sig: str, bitmap: np.ndarray) -> None:
    """Write sidecar next to the array (so all partitions share it); on failure, fall back to local cache."""
    blob = _serialize_bitmap(bitmap, sig)
    sidecar = _sidecar_url(array_url)
    try:
        if array_url.startswith("s3://"):
            fs = fsspec.filesystem("s3", anon=False)
            key = sidecar.replace("s3://", "")
            with fs.open(key, "wb") as f:
                f.write(blob)
            print(f"  Chunk occupancy cache SAVED to sidecar: {sidecar}")
            return
        else:
            tmp = sidecar + ".tmp"
            with open(tmp, "wb") as f:
                f.write(blob)
            os.replace(tmp, sidecar)
            print(f"  Chunk occupancy cache SAVED to sidecar: {sidecar}")
            return
    except Exception as e:
        warnings.warn(
            f"Could not write chunk occupancy sidecar to {sidecar}: {e}. "
            f"Falling back to local per-host cache (partitions will rebuild independently)."
        )

    local = _local_cache_path(array_url)
    try:
        tmp = local.with_name(local.name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, local)
        print(f"  Chunk occupancy cache SAVED to local fallback: {local}")
    except Exception as e:
        warnings.warn(f"Failed to save local chunk occupancy cache to {local}: {e}")


def _iter_chunk_filenames_s3(array_url: str):
    """Stream chunk filenames under an s3:// zarr v2 array prefix.

    Drives `list_objects_v2` page by page via s3fs's sync `call_s3`, so we can
    yield results without materialising the full key list (which at ~9M files
    would be ~GB of Python strings and minutes of silence).
    """
    fs = fsspec.filesystem("s3", anon=False)
    bucket_and_prefix = array_url.replace("s3://", "").rstrip("/")
    bucket, _, prefix = bucket_and_prefix.partition("/")
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    continuation = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if continuation is not None:
            kwargs["ContinuationToken"] = continuation
        resp = fs.call_s3("list_objects_v2", **kwargs)
        for obj in resp.get("Contents", []) or []:
            key = obj["Key"]
            # Skip anything in a sub-prefix (shouldn't happen for a zarr v2
            # array's flat chunk layout, but guard anyway) and the sidecar.
            rest = key[len(prefix):] if prefix else key
            if "/" in rest:
                continue
            yield rest
        if not resp.get("IsTruncated"):
            return
        continuation = resp.get("NextContinuationToken")
        if continuation is None:
            return


def _iter_chunk_filenames(array_url: str):
    """Yield chunk filenames (basenames) under a zarr v2 array directory."""
    if array_url.startswith("s3://"):
        yield from _iter_chunk_filenames_s3(array_url)
    else:
        try:
            with os.scandir(array_url) as it:
                for entry in it:
                    if entry.is_file():
                        yield entry.name
        except FileNotFoundError:
            return


def build_chunk_occupancy(
    array_url: str,
    chunks: Tuple[int, ...],
    shape: Tuple[int, ...],
    *,
    verbose: bool = False,
    use_cache: bool = True,
) -> Optional[np.ndarray]:
    """
    Build a boolean bitmap over the spatial chunk grid of a zarr v2 array.

    Args:
        array_url: URL pointing at the zarr array directory (contains `.zarray`
            and `N.C.Z.Y.X`-style chunk files). May be local or `s3://...`.
        chunks: Per-axis chunk sizes of the array.
        shape:  Per-axis array shape.
        verbose: Log progress/counts.
        use_cache: Honour (and populate) the on-disk cache.

    Returns:
        A boolean numpy array over the *spatial* chunk grid (leading channel
        axis dropped for 4D `(C,Z,Y,X)` inputs), or `None` if the layout
        cannot be parsed. Callers should fall back to lazy detection on None.
    """
    if len(chunks) != len(shape):
        warnings.warn(f"chunks/shape rank mismatch for {array_url}: {chunks} vs {shape}")
        return None

    # Identify the spatial axes: drop a leading channel axis for 4D inputs.
    rank = len(shape)
    if rank == 3:
        spatial_axes = (0, 1, 2)
    elif rank == 4:
        spatial_axes = (1, 2, 3)
    else:
        warnings.warn(f"Unsupported array rank {rank} for {array_url}; skipping chunk index")
        return None

    grid_dims = tuple(
        int(np.ceil(shape[ax] / chunks[ax])) if chunks[ax] > 0 else 0
        for ax in spatial_axes
    )
    if any(d == 0 for d in grid_dims):
        warnings.warn(f"Degenerate chunk grid for {array_url}: {grid_dims}")
        return None

    sig = _zarray_signature(array_url) if use_cache else None
    if sig is not None:
        cached, source = _load_cached(array_url, sig, grid_dims)
        if cached is not None:
            occ = int(cached.sum())
            total = int(cached.size)
            print(
                f"  Chunk occupancy cache HIT ({source}): "
                f"{occ}/{total} ({100.0 * occ / total:.1f}%) chunks occupied for {array_url}"
            )
            return cached
        print(f"  Chunk occupancy cache MISS for {array_url} (will build and cache)")
    elif use_cache:
        print(
            f"  Chunk occupancy cache skipped for {array_url} "
            f"(could not read .zarray signature; no caching this run)"
        )
    else:
        print(f"  Chunk occupancy cache disabled for {array_url}")

    print(f"  Building chunk occupancy index for {array_url} (grid {grid_dims})...")

    occupancy = np.zeros(grid_dims, dtype=bool)

    parsed = 0
    skipped = 0
    # Stream + count. tqdm runs with total=None so it just shows the running
    # count and rate — S3 has no way to know the total up front, and the
    # theoretical max (grid.prod) is a poor estimate for sparse volumes.
    pbar = tqdm(
        desc="  Listing chunks",
        unit=" files",
        unit_scale=True,
        leave=False,
        disable=not verbose,
        **get_tqdm_kwargs(),
    )
    try:
        for filename in _iter_chunk_filenames(array_url):
            pbar.update(1)
            if not filename or not filename[0].isdigit():
                continue
            parts = filename.split(".")
            if len(parts) != rank:
                continue
            try:
                idxs = tuple(int(p) for p in parts)
            except ValueError:
                skipped += 1
                continue
            spatial_idx = tuple(idxs[ax] for ax in spatial_axes)
            if any(spatial_idx[i] >= grid_dims[i] for i in range(len(grid_dims))):
                skipped += 1
                continue
            occupancy[spatial_idx] = True
            parsed += 1
    finally:
        pbar.close()

    if parsed == 0:
        warnings.warn(
            f"No parseable chunk files found under {array_url}; falling back to lazy empty-patch detection"
        )
        return None

    if verbose:
        occ = int(occupancy.sum())
        total = int(occupancy.size)
        print(
            f"  Chunk occupancy: {occ}/{total} ({100.0 * occ / total:.1f}%) "
            f"chunks occupied (parsed {parsed} files, skipped {skipped})"
        )

    if use_cache and sig is not None:
        _save_cached(array_url, sig, occupancy)

    return occupancy


def compute_patch_non_empty_mask(
    occupancy: np.ndarray,
    positions,
    patch_size: Tuple[int, int, int],
    chunks_spatial: Tuple[int, int, int],
) -> np.ndarray:
    """
    For each `(z, y, x)` patch origin in `positions`, test whether the patch
    footprint intersects any occupied chunk in `occupancy`.

    Args:
        occupancy: 3D bool bitmap over the spatial chunk grid.
        positions: iterable of (z, y, x) patch start coordinates.
        patch_size: (pZ, pY, pX).
        chunks_spatial: per-spatial-axis chunk sizes of the underlying array.

    Returns:
        Boolean numpy array of length len(positions). True = non-empty.
    """
    pZ, pY, pX = patch_size
    cZ, cY, cX = chunks_spatial
    gZ, gY, gX = occupancy.shape

    n = len(positions)
    mask = np.zeros(n, dtype=bool)

    for i, (z, y, x) in enumerate(positions):
        z0 = z // cZ
        y0 = y // cY
        x0 = x // cX
        z1 = min(gZ, (z + pZ - 1) // cZ + 1)
        y1 = min(gY, (y + pY - 1) // cY + 1)
        x1 = min(gX, (x + pX - 1) // cX + 1)
        if z0 >= gZ or y0 >= gY or x0 >= gX:
            continue
        if occupancy[z0:z1, y0:y1, x0:x1].any():
            mask[i] = True

    return mask
