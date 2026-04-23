"""
Fast occupancy bitmap for zarr v2 arrays.

Given a zarr array whose chunks are laid out as `<dim0>.<dim1>.<...>` files,
build a boolean numpy array over the chunk grid marking which chunks actually
exist on disk/S3. This lets callers pre-filter patches that fall entirely in
empty regions of a sparse input volume without ever reading the underlying data.

Also provides a disk cache keyed by the array's `.zarray` ETag (S3) or mtime
(local), so repeat runs pay no listing cost after the first.

Cache lookup order (read): explicit override URL (env var
``VESUVIUS_CHUNK_OCCUPANCY_URL``) → sidecar next to the input array →
per-host local fallback. Cache save order (write): override URL if set,
otherwise sidecar, otherwise local. The override URL is always accessed
with the caller's default (authenticated) credentials — it is intended to
live on a writable working prefix, even when the input array is anonymous.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import fsspec
import numpy as np
from tqdm.auto import tqdm

from vesuvius.utils.k8s import get_tqdm_kwargs


SIDECAR_NAME = ".chunk_occupancy.npz"
ENV_OVERRIDE_URL = "VESUVIUS_CHUNK_OCCUPANCY_URL"

_LIST_MAX_WORKERS = int(os.environ.get("VESUVIUS_CHUNK_INDEX_WORKERS", "32"))


def _override_cache_url() -> Optional[str]:
    url = os.environ.get(ENV_OVERRIDE_URL, "").strip()
    return url or None


def _url_to_fs(url: str, *, anon: bool = False):
    """Resolve (fs, fs_path) for ``url``. Apply ``anon`` only to S3 URLs."""
    storage_options = {"anon": True} if anon and url.startswith("s3://") else {}
    return fsspec.core.url_to_fs(url, **storage_options)


def _local_cache_dir() -> Path:
    d = Path(os.environ.get("VESUVIUS_CACHE_DIR", Path.home() / ".cache" / "vesuvius")) / "chunk_occupancy"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _local_cache_path(array_url: str) -> Path:
    key = hashlib.sha1(array_url.encode()).hexdigest()
    return _local_cache_dir() / f"{key}.npz"


def _sidecar_url(array_url: str) -> str:
    return array_url.rstrip("/") + "/" + SIDECAR_NAME


def _zarray_signature(array_url: str, *, anon: bool = False) -> Optional[str]:
    """Return a short stable signature for the array's .zarray (ETag on S3, mtime+size locally)."""
    zarray_url = array_url.rstrip("/") + "/.zarray"
    try:
        fs, fs_path = _url_to_fs(zarray_url, anon=anon)
        info = fs.info(fs_path)
        etag = info.get("ETag") or info.get("etag")
        if etag:
            return str(etag).strip('"')
        mtime = info.get("mtime")
        size = info.get("size")
        if mtime is not None and size is not None:
            # Local-style signature when ETag is unavailable.
            try:
                mtime_ns = int(mtime.timestamp() * 1_000_000_000) if hasattr(mtime, "timestamp") else int(mtime)
            except Exception:
                mtime_ns = 0
            return f"{mtime_ns}-{size}"
        if size is not None:
            return f"size={size}"
        return None
    except Exception:
        return None


def serialize_bitmap(bitmap: np.ndarray, sig: str) -> bytes:
    """Encode the occupancy bitmap + signature into a single .npz byte blob."""
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, occupancy=bitmap, signature=np.array(sig))
    return buf.getvalue()


def deserialize_bitmap(blob: bytes, expected_sig: str, expected_grid: Tuple[int, ...]) -> Optional[np.ndarray]:
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


# Backwards-compatible aliases for internal callers.
_serialize_bitmap = serialize_bitmap
_deserialize_bitmap = deserialize_bitmap


def _read_cache_blob(url: str) -> Optional[bytes]:
    """Read raw bytes from a cache URL, or None on miss. Never uses anon."""
    try:
        fs, fs_path = _url_to_fs(url)
        if not fs.exists(fs_path):
            return None
        with fs.open(fs_path, "rb") as f:
            return f.read()
    except Exception as e:
        warnings.warn(f"Failed to read chunk occupancy cache at {url}: {e}")
        return None


def _write_cache_blob(url: str, blob: bytes) -> bool:
    """Write bytes to a cache URL. Returns True on success."""
    try:
        fs, fs_path = _url_to_fs(url)
        with fs.open(fs_path, "wb") as f:
            f.write(blob)
        return True
    except Exception as e:
        warnings.warn(f"Failed to write chunk occupancy cache at {url}: {e}")
        return False


def _load_cached(
    array_url: str, sig: str, expected_grid: Tuple[int, ...]
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Try override URL, then sidecar next to the array, then per-host local cache.

    Returns (bitmap, source) where `source` is one of:
      - "override:<url>"  explicit ``VESUVIUS_CHUNK_OCCUPANCY_URL`` cache
      - "sidecar:<url>"   sidecar file next to the zarr array
      - "local:<path>"    per-host local fallback cache
      - None              no cache hit (bitmap is also None)
    """
    override = _override_cache_url()
    if override:
        blob = _read_cache_blob(override)
        if blob is not None:
            bitmap = _deserialize_bitmap(blob, sig, expected_grid)
            if bitmap is not None:
                return bitmap, f"override:{override}"

    sidecar = _sidecar_url(array_url)
    blob = _read_cache_blob(sidecar)
    if blob is not None:
        bitmap = _deserialize_bitmap(blob, sig, expected_grid)
        if bitmap is not None:
            return bitmap, f"sidecar:{sidecar}"

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
    """Write override URL first, then sidecar, then per-host local fallback."""
    blob = _serialize_bitmap(bitmap, sig)

    override = _override_cache_url()
    if override:
        if _write_cache_blob(override, blob):
            print(f"  Chunk occupancy cache SAVED to override: {override}")
            return

    sidecar = _sidecar_url(array_url)
    try:
        fs, fs_path = _url_to_fs(sidecar)
        if array_url.startswith("s3://"):
            with fs.open(fs_path, "wb") as f:
                f.write(blob)
        else:
            tmp = fs_path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(blob)
            os.replace(tmp, fs_path)
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


def _read_zarray(array_url: str, *, anon: bool = False) -> Optional[dict]:
    """Fetch and parse `.zarray` for the given array URL.

    Returns the parsed JSON dict, or None on any failure (caller falls back
    to the lazy empty-detection path).
    """
    zarray_url = array_url.rstrip("/") + "/.zarray"
    try:
        fs, fs_path = _url_to_fs(zarray_url, anon=anon)
        with fs.open(fs_path, "rb") as f:
            return json.loads(f.read())
    except Exception as e:
        warnings.warn(f"Failed to read {zarray_url}: {e}")
        return None


def _list_chunks_s3(array_url: str, sub_prefix: str, *, anon: bool = False) -> List[str]:
    """List chunk keys under an S3 zarr v2 array at `sub_prefix`.

    Returns keys relative to the array prefix (e.g. `"0/12/3/7"` for a
    `/`-separated 4D array, or `"0.12.3.7"` for a `.`-separated one).
    """
    fs = fsspec.filesystem("s3", anon=anon)
    bucket_and_prefix = array_url.replace("s3://", "").rstrip("/")
    bucket, _, prefix = bucket_and_prefix.partition("/")
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    full_prefix = prefix + sub_prefix

    results: List[str] = []
    continuation = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": full_prefix, "MaxKeys": 1000}
        if continuation is not None:
            kwargs["ContinuationToken"] = continuation
        resp = fs.call_s3("list_objects_v2", **kwargs)
        for obj in resp.get("Contents", []) or []:
            key = obj["Key"]
            rest = key[len(prefix):] if prefix else key
            results.append(rest)
        if not resp.get("IsTruncated"):
            break
        continuation = resp.get("NextContinuationToken")
        if continuation is None:
            break
    return results


def _list_chunks_local(array_url: str, sub_prefix: str, sep: str) -> List[str]:
    """List chunk keys under a local zarr v2 array at `sub_prefix`.

    For `sep == "."` the chunks are flat files directly under `array_url`,
    so `sub_prefix` is a filename-prefix filter. For `sep == "/"` the
    chunks are nested in directories — walk the deepest level and
    reassemble paths relative to `array_url`.
    """
    base = array_url.rstrip("/")
    results: List[str] = []

    if sep == ".":
        try:
            with os.scandir(base) as it:
                for entry in it:
                    if entry.is_file() and entry.name.startswith(sub_prefix):
                        results.append(entry.name)
        except FileNotFoundError:
            pass
        return results

    # Nested-directory layout: sub_prefix is e.g. "0/12/" — recurse under it.
    start = os.path.join(base, sub_prefix.rstrip("/")) if sub_prefix else base
    if not os.path.isdir(start):
        return results

    prefix_depth = len(base.rstrip("/")) + 1  # strip "base/" from absolute paths
    stack = [start]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file():
                        rel = entry.path[prefix_depth:]
                        results.append(rel)
        except FileNotFoundError:
            continue
    return results


def build_chunk_occupancy(
    array_url: str,
    chunks: Tuple[int, ...],
    shape: Tuple[int, ...],
    *,
    verbose: bool = False,
    use_cache: bool = True,
    anon: bool = False,
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
        anon: Use unsigned S3 requests when reading the input array. The
            override cache URL (``VESUVIUS_CHUNK_OCCUPANCY_URL``) is always
            accessed with default credentials and never inherits this flag.

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

    sig = _zarray_signature(array_url, anon=anon) if use_cache else None
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

    zarray = _read_zarray(array_url, anon=anon)
    if zarray is None:
        return None
    sep = zarray.get("dimension_separator", ".")
    if sep not in (".", "/"):
        warnings.warn(f"Unknown dimension_separator {sep!r} for {array_url}; skipping chunk index")
        return None

    # Partition the chunk keyspace along the leading axis (z, or c×z for 4D).
    # Each sub-prefix corresponds to one "row" of chunks and is listed
    # independently — in parallel on S3, sequentially on local FS.
    z_grid = grid_dims[0]
    if rank == 4:
        c_grid = int(np.ceil(shape[0] / chunks[0])) if chunks[0] > 0 else 0
        sub_prefixes = [f"{c}{sep}{z}{sep}" for c in range(c_grid) for z in range(z_grid)]
    else:
        sub_prefixes = [f"{z}{sep}" for z in range(z_grid)]

    print(
        f"  Building chunk occupancy index for {array_url} "
        f"(grid {grid_dims}, separator {sep!r}, {len(sub_prefixes)} rows)..."
    )

    occupancy = np.zeros(grid_dims, dtype=bool)
    parsed = 0
    skipped = 0

    def _consume(relpaths: List[str]) -> None:
        nonlocal parsed, skipped
        for rel in relpaths:
            if not rel or not rel[0].isdigit():
                continue
            parts = rel.split(sep)
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

    pbar_kwargs = {
        "total": len(sub_prefixes),
        "desc": "  Listing chunks",
        "unit": "row",
        "leave": False,
        "disable": not verbose,
        **get_tqdm_kwargs(),
    }
    pbar = tqdm(**pbar_kwargs)
    try:
        if array_url.startswith("s3://"):
            with ThreadPoolExecutor(max_workers=min(_LIST_MAX_WORKERS, max(1, len(sub_prefixes)))) as ex:
                futures = [ex.submit(_list_chunks_s3, array_url, sp, anon=anon) for sp in sub_prefixes]
                for fut in as_completed(futures):
                    _consume(fut.result())
                    pbar.update(1)
                    pbar.set_postfix_str(f"found={parsed}")
        else:
            for sp in sub_prefixes:
                _consume(_list_chunks_local(array_url, sp, sep))
                pbar.update(1)
                pbar.set_postfix_str(f"found={parsed}")
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
