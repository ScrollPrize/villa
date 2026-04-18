"""Pre-fetch all zarr chunks needed for a training config.

Phase 1: Iterate the dataset with a logging cache stub to discover all
         (S3 URI, zarr key) pairs that would be requested during training.
Phase 2: Download missing chunks in parallel with boto3 (128 workers)
         into the same cache directory layout as _DiskCacheStore.

Usage:
  python lasagna/scripts/prefetch_training_cache.py \
      --config lasagna/configs/tifxyz_train_s3.json \
      --patch-size 192 --workers 128
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numba as _numba
import numpy as _np

@_numba.njit
def _numba_warmup():
    return _np.zeros(1, dtype=_np.int32)
_numba_warmup()
del _numba, _np, _numba_warmup

import argparse
import collections
import json
import math
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# ---------------------------------------------------------------------------
# Logging cache stub — records all requested keys
# ---------------------------------------------------------------------------

import zarr.abc.store

class _LoggingCacheStore(zarr.abc.store.Store):
    """Drop-in for _DiskCacheStore that logs requested keys.

    Delegates list/exists to the real remote store so zarr can discover
    metadata.  Only ``get`` is intercepted to log + delegate.
    """

    _requests: list[tuple[str, str]] = []
    _lock = threading.Lock()

    def __init__(self, remote, cache_dir, url, **kwargs):
        super().__init__(read_only=True)
        self._remote = remote
        self._url = url.rstrip("/")

    @classmethod
    def get_requests(cls) -> list[tuple[str, str]]:
        return list(cls._requests)

    @classmethod
    def clear(cls):
        cls._requests.clear()

    async def _open(self):
        self._is_open = True

    async def get(self, key, prototype, byte_range=None):
        with self._lock:
            self._requests.append((self._url, key))
        # Metadata keys: delegate to remote so zarr can open arrays.
        # Chunk data keys: return None (we only need the key names).
        _meta_suffixes = (".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json")
        if any(key.endswith(s) for s in _meta_suffixes):
            return await self._remote.get(key, prototype, byte_range=byte_range)
        return None

    async def get_partial_values(self, prototype, key_ranges):
        return await self._remote.get_partial_values(prototype, key_ranges)

    async def set(self, key, value):
        raise PermissionError("read-only")

    async def set_if_not_exists(self, key, value):
        raise PermissionError("read-only")

    async def delete(self, key):
        raise PermissionError("read-only")

    async def exists(self, key):
        return await self._remote.exists(key)

    async def is_empty(self, prefix=""):
        return await self._remote.is_empty(prefix)

    @property
    def supports_writes(self):
        return False

    @property
    def supports_deletes(self):
        return False

    @property
    def supports_partial_writes(self):
        return False

    @property
    def supports_listing(self):
        return self._remote.supports_listing

    def list(self):
        return self._remote.list()

    def list_prefix(self, prefix):
        return self._remote.list_prefix(prefix)

    def list_dir(self, prefix):
        return self._remote.list_dir(prefix)

    def __eq__(self, other):
        return isinstance(other, _LoggingCacheStore) and self._url == other._url


# ---------------------------------------------------------------------------
# Parallel S3 download (adapted from download_omezarr.py)
# ---------------------------------------------------------------------------

import boto3
import botocore
import botocore.config
from botocore import UNSIGNED

_thread_local = threading.local()


def _get_s3_client(anon: bool = False):
    client = getattr(_thread_local, "s3_client", None)
    if client is not None:
        return client
    cfg = botocore.config.Config(
        max_pool_connections=4,
        retries={"max_attempts": 0},
        connect_timeout=10,
        read_timeout=30,
    )
    if anon:
        cfg = cfg.merge(botocore.config.Config(signature_version=UNSIGNED))
    session = boto3.Session()
    client = session.client("s3", config=cfg)
    _thread_local.s3_client = client
    return client


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    rest = uri[5:]
    slash = rest.index("/")
    return rest[:slash], rest[slash + 1:]


def _cache_path_for_key(cache_dir: str, url: str, key: str) -> str:
    """Compute the local cache path matching _DiskCacheStore layout."""
    normalized = url.rstrip("/")
    scheme, sep, rest = normalized.partition("://")
    subdir = os.path.join(scheme, rest) if sep else normalized
    return os.path.join(cache_dir, subdir, key)


def _s3_key_for_zarr_key(url: str, zarr_key: str) -> str:
    """Compute the S3 object key for a zarr store key."""
    _, prefix = _parse_s3_uri(url)
    prefix = prefix.rstrip("/")
    return f"{prefix}/{zarr_key}"


def _download_one(
    bucket: str, s3_key: str, local_path: str,
    anon: bool, stats: dict, max_retries: int = 3,
) -> bool:
    for attempt in range(max_retries):
        try:
            client = _get_s3_client(anon)
            resp = client.get_object(Bucket=bucket, Key=s3_key)
            data = resp["Body"].read()
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            tmp = f"{local_path}.tmp.{os.getpid()}.{threading.get_ident()}"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, local_path)
            with stats["lock"]:
                stats["downloaded"] += 1
                stats["bytes"] += len(data)
            return True
        except botocore.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                # Write negative marker
                marker = local_path + ".__notfound__"
                os.makedirs(os.path.dirname(marker) or ".", exist_ok=True)
                try:
                    with open(marker, "wb") as f:
                        pass
                except OSError:
                    pass
                with stats["lock"]:
                    stats["downloaded"] += 1
                return True
            if attempt == max_retries - 1:
                print(f"\n  WARN: failed {s3_key}: {e}", file=sys.stderr)
                with stats["lock"]:
                    stats["failed"] += 1
                return False
            time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n  WARN: failed {s3_key}: {e}", file=sys.stderr)
                with stats["lock"]:
                    stats["failed"] += 1
                return False
            time.sleep(0.5 * (attempt + 1))
    return False


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TAG = "[prefetch]"


def main():
    parser = argparse.ArgumentParser(
        description="Pre-fetch zarr cache for training config.",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--label-patch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only discover and count chunks, don't download.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    config["patch_size"] = args.patch_size
    config["label_patch_size"] = args.label_patch_size or args.patch_size

    cache_dir = config.get("volume_cache_dir")
    if not cache_dir:
        print(f"{TAG} ERROR: config has no volume_cache_dir", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------------
    # Phase 1: Discover all zarr keys by iterating the dataset
    # ---------------------------------------------------------------
    print(f"{TAG} Phase 1: Discovering required chunks...", flush=True)

    _LASAGNA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _LASAGNA_DIR not in sys.path:
        sys.path.insert(0, _LASAGNA_DIR)

    from tifxyz_lasagna_dataset import TifxyzLasagnaDataset, _apply_affine_zyx

    patch_size = args.patch_size
    label_patch_size = args.label_patch_size or args.patch_size
    P = patch_size
    L = label_patch_size
    # paste_off ranges [-L/2, P-L/2]. Max deviation from centered = P/2.
    pad = P // 2 if P > L else 0

    # Build dataset (normal store, no monkey-patching).
    dataset = TifxyzLasagnaDataset(
        config, apply_augmentation=False,
        include_geometry=False,
        include_patch_ref=False,
    )
    n_patches = len(dataset)
    print(f"{TAG} {n_patches} patches", flush=True)

    scale_aug_enabled = float(config.get("scale_aug_prob", 0)) > 0
    scale_aug_factor = int(config.get("scale_aug_factor", 2))

    # --- Detect chunk key format by reading one chunk via logging store ---
    # Monkey-patch briefly: read a single voxel from each volume to see
    # what key zarr requests, then restore.
    from vesuvius.neural_tracing.datasets import common as _common
    _OrigStore = _common._DiskCacheStore
    _common._DiskCacheStore = _LoggingCacheStore
    _LoggingCacheStore.clear()

    ds_uri_map: dict[int, str] = {}
    for di, ds in enumerate(config["datasets"]):
        vp = ds.get("volume_path")
        if vp:
            ds_uri_map[di] = vp.rstrip("/")

    # For each unique (uri, scale), open the zarr and read one voxel
    # to discover the chunk key format.
    chunk_key_formats: dict[tuple[str, int], str] = {}  # (uri, scale) -> e.g. "{scale}/c/{iz}/{iy}/{ix}"
    seen_vol_keys: set[tuple[str, int]] = set()
    for patch in dataset.patches:
        uri = ds_uri_map.get(patch.dataset_idx)
        if not uri or not uri.startswith("s3://"):
            continue
        for sc in [patch.scale] + ([patch.scale + 1] if scale_aug_enabled else []):
            vk = (uri, sc)
            if vk in seen_vol_keys:
                continue
            seen_vol_keys.add(vk)
            try:
                from vesuvius.neural_tracing.datasets.common import open_zarr, open_zarr_group
                if sc == patch.scale:
                    arr = open_zarr(uri, scale=sc, config=config)
                else:
                    grp = open_zarr_group(uri, config=config)
                    arr = grp[str(sc)]
                # Read a single voxel to trigger one chunk key request
                _ = arr[0, 0, 0]
            except Exception:
                pass

    _common._DiskCacheStore = _OrigStore

    # Parse logged keys to find chunk key pattern
    for url, key in _LoggingCacheStore.get_requests():
        if any(key.endswith(s) for s in (".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json")):
            continue
        # This is a chunk key like "2/c/0/0/0" or "2/0.0.0"
        # Extract the scale from the prefix and determine format
        parts = key.split("/")
        try:
            sc = int(parts[0])
        except (ValueError, IndexError):
            continue
        vk = (url, sc)
        if vk not in chunk_key_formats:
            # Determine format from the key structure
            rest = key[len(parts[0]) + 1:]  # after "{scale}/"
            if rest.startswith("c/") or rest.startswith("c\\"):
                chunk_key_formats[vk] = "v3"  # {scale}/c/{iz}/{iy}/{ix}
            elif "." in rest:
                chunk_key_formats[vk] = "v2dot"  # {scale}/{iz}.{iy}.{ix}
            else:
                chunk_key_formats[vk] = "v2slash"  # {scale}/{iz}/{iy}/{ix}

    print(f"{TAG} Detected chunk key formats: "
          + ", ".join(f"{u} s{s}: {f}" for (u, s), f in sorted(chunk_key_formats.items())),
          flush=True)

    def _make_chunk_key(scale, iz, iy, ix, fmt):
        if fmt == "v3":
            return f"{scale}/c/{iz}/{iy}/{ix}"
        elif fmt == "v2dot":
            return f"{scale}/{iz}.{iy}.{ix}"
        else:
            return f"{scale}/{iz}/{iy}/{ix}"

    # --- Compute chunk indices from patch bboxes (fast, no IO) ---
    def _enumerate_chunk_indices(shape, chunks, bbox_min, bbox_max):
        result = set()
        nz = math.ceil(shape[0] / chunks[0])
        ny = math.ceil(shape[1] / chunks[1])
        nx = math.ceil(shape[2] / chunks[2])
        iz0 = max(0, int(bbox_min[0]) // chunks[0])
        iy0 = max(0, int(bbox_min[1]) // chunks[1])
        ix0 = max(0, int(bbox_min[2]) // chunks[2])
        iz1 = min(nz, math.ceil(max(int(bbox_max[0]), 1) / chunks[0]))
        iy1 = min(ny, math.ceil(max(int(bbox_max[1]), 1) / chunks[1]))
        ix1 = min(nx, math.ceil(max(int(bbox_max[2]), 1) / chunks[2]))
        for iz in range(iz0, iz1):
            for iy in range(iy0, iy1):
                for ix in range(ix0, ix1):
                    result.add((iz, iy, ix))
        return result

    # Collect volume metadata (shape, chunks) from the already-opened arrays
    vol_meta: dict[tuple[str, int], tuple] = {}  # (uri, scale) -> (shape, chunks)
    for patch in dataset.patches:
        uri = ds_uri_map.get(patch.dataset_idx)
        if not uri:
            continue
        vk = (uri, patch.scale)
        if vk not in vol_meta:
            arr = patch.volume
            chunks = arr.chunks if hasattr(arr, 'chunks') else arr.metadata.chunk_grid.chunk_shape
            vol_meta[vk] = (arr.shape, chunks)
        if scale_aug_enabled and patch.volume_group is not None:
            vk_aug = (uri, patch.scale + 1)
            if vk_aug not in vol_meta:
                try:
                    aug_arr = patch.volume_group[str(patch.scale + 1)]
                    chunks = aug_arr.chunks if hasattr(aug_arr, 'chunks') else aug_arr.metadata.chunk_grid.chunk_shape
                    vol_meta[vk_aug] = (aug_arr.shape, chunks)
                except (KeyError, IndexError):
                    pass

    # Enumerate chunks per volume/scale
    chunks_needed: dict[tuple[str, int], set[tuple[int, int, int]]] = collections.defaultdict(set)
    t_scan = time.monotonic()

    for i, patch in enumerate(dataset.patches):
        uri = ds_uri_map.get(patch.dataset_idx)
        if not uri:
            continue

        z0, z1, y0, y1, x0, x1 = patch.world_bbox

        if patch.cache_to_volume is not None:
            corners_zyx = np.array([
                [z, y, x]
                for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)
            ], dtype=np.float64)
            corners_vol = _apply_affine_zyx(patch.cache_to_volume, corners_zyx)
            aabb_center = (corners_vol.min(axis=0) + corners_vol.max(axis=0)) / 2.0
            label_min = np.round(
                aabb_center - np.array((L, L, L), dtype=np.float64) / 2.0
            ).astype(np.int64)
        else:
            label_min = np.array([z0, y0, x0], dtype=np.int64)

        centered_off = np.array([(P - L) // 2] * 3, dtype=np.int64)
        base_min = label_min - centered_off
        base_max = base_min + P
        worst_min = base_min - pad
        worst_max = base_max + pad

        # Base scale
        vk = (uri, patch.scale)
        meta = vol_meta.get(vk)
        if meta:
            shape, chk = meta
            cmin = np.clip(worst_min, 0, np.array(shape) - 1)
            cmax = np.clip(worst_max, 1, np.array(shape))
            chunks_needed[vk].update(
                _enumerate_chunk_indices(shape, chk, cmin, cmax)
            )

        # Scale aug
        if scale_aug_enabled:
            f = scale_aug_factor
            vk_aug = (uri, patch.scale + 1)
            meta_aug = vol_meta.get(vk_aug)
            if meta_aug:
                shape_aug, chk_aug = meta_aug
                max_offset_world = (P - P // f) * f
                aug_world_min = worst_min - max_offset_world
                aug_min = aug_world_min // f
                aug_max = aug_min + P + max_offset_world // f
                cmin = np.clip(aug_min, 0, np.array(shape_aug) - 1)
                cmax = np.clip(aug_max, 1, np.array(shape_aug))
                chunks_needed[vk_aug].update(
                    _enumerate_chunk_indices(shape_aug, chk_aug, cmin, cmax)
                )

        if (i + 1) % 2000 == 0 or i == n_patches - 1:
            elapsed = time.monotonic() - t_scan
            rate = (i + 1) / max(elapsed, 0.01)
            eta = (n_patches - i - 1) / max(rate, 0.01)
            total = sum(len(v) for v in chunks_needed.values())
            sys.stderr.write(
                f"\r{TAG} scan: {i + 1}/{n_patches} "
                f"({rate:.0f}/s, ETA {eta:.0f}s, {total} chunks)  "
            )
            sys.stderr.flush()
    sys.stderr.write("\n")

    # Build (url, zarr_key) set
    chunk_requests: set[tuple[str, str]] = set()
    for (uri, scale), idxs in chunks_needed.items():
        fmt = chunk_key_formats.get((uri, scale), "v2slash")
        for iz, iy, ix in idxs:
            chunk_requests.add((uri, _make_chunk_key(scale, iz, iy, ix, fmt)))

    # Add metadata keys
    for (uri, scale) in chunks_needed.keys():
        for meta in [".zarray", ".zattrs", "zarr.json", ".zgroup"]:
            chunk_requests.add((uri, f"{scale}/{meta}"))
        for meta in [".zattrs", ".zgroup", "zarr.json", ".zmetadata"]:
            chunk_requests.add((uri, meta))

    total = sum(len(v) for v in chunks_needed.values())
    print(f"{TAG} {total} unique chunks, {len(chunk_requests)} total keys", flush=True)
    for vk, idxs in sorted(chunks_needed.items()):
        print(f"  {vk[0]} scale={vk[1]}: {len(idxs)} chunks ({chunk_key_formats.get(vk, '?')})")

    # ---------------------------------------------------------------
    # Phase 2: Check cache and download missing
    # ---------------------------------------------------------------

    # Check which are already cached
    to_download: list[tuple[str, str, str, str]] = []  # (bucket, s3_key, local_path, label)
    n_cached = 0
    n_skipped_non_s3 = 0
    for url, zarr_key in chunk_requests:
        local_path = _cache_path_for_key(cache_dir, url, zarr_key)
        marker = local_path + ".__notfound__"
        if os.path.isfile(local_path) or os.path.isfile(marker):
            n_cached += 1
            continue
        if not url.startswith("s3://"):
            n_skipped_non_s3 += 1
            continue
        bucket, _ = _parse_s3_uri(url)
        s3_key = _s3_key_for_zarr_key(url, zarr_key)
        to_download.append((bucket, s3_key, local_path, f"{url}/{zarr_key}"))
    if n_skipped_non_s3:
        print(f"{TAG} skipped {n_skipped_non_s3} non-S3 keys (HTTPS etc.)",
              flush=True)

    print(f"{TAG} {n_cached} already cached, {len(to_download)} to download",
          flush=True)

    if args.dry_run:
        print(f"{TAG} Dry run — not downloading.", flush=True)
        # Show per-volume breakdown
        by_url: dict[str, int] = collections.Counter()
        for url, _ in chunk_requests:
            by_url[url] += 1
        for url, count in sorted(by_url.items()):
            print(f"  {url}: {count} keys")
        return

    if not to_download:
        print(f"{TAG} Everything is cached. Done.", flush=True)
        return

    # Download
    print(f"{TAG} Phase 2: Downloading {len(to_download)} chunks "
          f"with {args.workers} workers...", flush=True)

    stats = {
        "downloaded": 0,
        "failed": 0,
        "bytes": 0,
        "lock": threading.Lock(),
    }
    total = len(to_download)
    t0 = time.monotonic()

    # Progress thread
    stop_evt = threading.Event()

    def _progress():
        while not stop_evt.is_set():
            with stats["lock"]:
                done = stats["downloaded"] + stats["failed"]
                byt = stats["bytes"]
                failed = stats["failed"]
            elapsed = time.monotonic() - t0
            rate = byt / max(elapsed, 0.01)
            remain = total - done
            eta = (remain / max(done, 1)) * elapsed if done > 0 else 0
            sys.stderr.write(
                f"\r{TAG} {done}/{total} "
                f"({_fmt_bytes(int(rate))}/s, fail={failed}, "
                f"ETA={eta:.0f}s)  "
            )
            sys.stderr.flush()
            stop_evt.wait(1.0)

    prog = threading.Thread(target=_progress, daemon=True)
    prog.start()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for bucket, s3_key, local_path, label in to_download:
            fut = pool.submit(
                _download_one, bucket, s3_key, local_path, False, stats,
            )
            futures.append(fut)
        for f in as_completed(futures):
            f.result()

    stop_evt.set()
    prog.join(timeout=2)

    elapsed = time.monotonic() - t0
    print(f"\n{TAG} Done: {stats['downloaded']} downloaded, "
          f"{stats['failed']} failed, "
          f"{_fmt_bytes(stats['bytes'])} in {elapsed:.1f}s "
          f"({_fmt_bytes(int(stats['bytes'] / max(elapsed, 0.01)))}/s)",
          flush=True)


if __name__ == "__main__":
    main()
