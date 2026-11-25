"""Pre-fetch all zarr chunks needed for a training config.

Phase 1: Enumerate, from the dataset's patches, the chunks training can read.
Phase 2: Read them through the dataset's own readers, which keep what they fetch
         in the config's ``volume_cache_dir``.

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
import sys
import time

import numpy as np
from vc_reader.reader.auth import is_remote

# ---------------------------------------------------------------------------
# Chunk reads
# ---------------------------------------------------------------------------


def _chunk_region(reader, chunk: tuple[int, int, int]) -> tuple[list[int], list[int]]:
    start = [i * c for i, c in zip(chunk, reader.chunks)]
    return start, [min(c, n - a) for c, n, a in zip(reader.chunks, reader.shape, start)]


def _fetch_chunks(reader, chunks: list[tuple[int, int, int]], *, max_retries: int = 3) -> bool:
    """Read these chunks, so that the reader stores them in its disk cache."""
    regions = [_chunk_region(reader, chunk) for chunk in chunks]
    for attempt in range(max_retries):
        try:
            reader.read_regions(regions)
            return True
        except Exception as error:
            if attempt == max_retries - 1:
                print(f"\n  WARN: failed {reader.location} chunks {chunks[0]}..: {error}", file=sys.stderr)
                return False
            time.sleep(0.5 * (attempt + 1))
    return False


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
    parser.add_argument("--refine", action="store_true",
                        help="Also fetch coarse (scale+1) and fine (scale-1) "
                             "chunks for multi-scale refinement training.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only discover and count chunks, don't download.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    config["patch_size"] = args.patch_size
    config["label_patch_size"] = args.label_patch_size or args.patch_size
    if args.refine:
        config["refine_mode"] = True

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

    from dtypes import numpy_float_hi
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
    refine_mode = bool(config.get("refine_mode", False))

    ds_uri_map: dict[int, str] = {}
    for di, ds in enumerate(config["datasets"]):
        vp = ds.get("volume_path")
        if vp:
            ds_uri_map[di] = vp.rstrip("/")

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

    # The dataset's readers, which carry its cache settings
    vol_meta: dict[tuple[str, int], tuple] = {}  # (uri, scale) -> (shape, chunks)
    readers: dict[tuple[str, int], object] = {}
    def _add_vol_meta(uri, scale, arr):
        vk = (uri, scale)
        if vk not in vol_meta:
            vol_meta[vk] = (arr.shape, arr.chunks)
            readers[vk] = arr
    for patch in dataset.patches:
        uri = ds_uri_map.get(patch.dataset_idx)
        if not uri:
            continue
        _add_vol_meta(uri, patch.scale, patch.volume)
        if patch.volume_group is not None:
            # Scale aug: coarser level
            if scale_aug_enabled:
                try:
                    _add_vol_meta(uri, patch.scale + 1,
                                  patch.volume_group[str(patch.scale + 1)])
                except (KeyError, IndexError):
                    pass
            # Refine: coarser + finer levels
            if refine_mode:
                try:
                    _add_vol_meta(uri, patch.scale + 1,
                                  patch.volume_group[str(patch.scale + 1)])
                except (KeyError, IndexError):
                    pass
                if patch.scale - 1 >= 0:
                    try:
                        _add_vol_meta(uri, patch.scale - 1,
                                      patch.volume_group[str(patch.scale - 1)])
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
            ], dtype=numpy_float_hi)
            corners_vol = _apply_affine_zyx(patch.cache_to_volume, corners_zyx)
            aabb_center = (corners_vol.min(axis=0) + corners_vol.max(axis=0)) / 2.0
            label_min = np.round(
                aabb_center - np.array((L, L, L), dtype=numpy_float_hi) / 2.0
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

        # Refine multi-scale: coarse (scale+1) and fine (scale-1)
        if refine_mode:
            base_center = (worst_min + worst_max) // 2

            # Coarse (zarr level+1): P coarse voxels, each = 2 base voxels.
            # coarse_min = (base_center - P) // 2 + coarse_offset // 2
            # coarse_offset in [0, P//2), so worst-case envelope:
            vk_m1 = (uri, patch.scale + 1)
            meta_m1 = vol_meta.get(vk_m1)
            if meta_m1:
                shape_m1, chk_m1 = meta_m1
                coarse_min_worst = (base_center - P) // 2
                coarse_max_worst = coarse_min_worst + P // 4 + P
                cmin = np.clip(coarse_min_worst, 0, np.array(shape_m1) - 1)
                cmax = np.clip(coarse_max_worst, 1, np.array(shape_m1))
                chunks_needed[vk_m1].update(
                    _enumerate_chunk_indices(shape_m1, chk_m1, cmin, cmax)
                )

            # Fine (zarr level-1): P fine voxels, each = 0.5 base voxels.
            # fine_ct_min = (base_min + fine_offset) * 2
            # fine_offset in [0, P//2], so worst-case envelope:
            if patch.scale - 1 >= 0:
                vk_p1 = (uri, patch.scale - 1)
                meta_p1 = vol_meta.get(vk_p1)
                if meta_p1:
                    shape_p1, chk_p1 = meta_p1
                    fine_min_worst = worst_min * 2
                    fine_max_worst = (worst_max + P // 2) * 2 + P
                    cmin = np.clip(fine_min_worst, 0, np.array(shape_p1) - 1)
                    cmax = np.clip(fine_max_worst, 1, np.array(shape_p1))
                    chunks_needed[vk_p1].update(
                        _enumerate_chunk_indices(shape_p1, chk_p1, cmin, cmax)
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

    total = sum(len(v) for v in chunks_needed.values())
    print(f"{TAG} {total} unique chunks", flush=True)
    for vk, idxs in sorted(chunks_needed.items()):
        print(f"  {vk[0]} scale={vk[1]}: {len(idxs)} chunks")

    # ---------------------------------------------------------------
    # Phase 2: Read everything; what is in the disk cache is not fetched
    # ---------------------------------------------------------------

    if args.dry_run:
        print(f"{TAG} Dry run — not downloading.", flush=True)
        return

    print(f"{TAG} Phase 2: Reading {total} chunks, {args.workers} at a time...", flush=True)
    done = failed = 0
    t0 = time.monotonic()
    for vk, idxs in sorted(chunks_needed.items()):
        reader = readers[vk]
        if not is_remote(reader.location):
            print(f"{TAG} {reader.location} is local, nothing to fetch", flush=True)
            continue
        ordered = sorted(idxs)
        for begin in range(0, len(ordered), args.workers):
            batch = ordered[begin:begin + args.workers]
            if not _fetch_chunks(reader, batch):
                failed += len(batch)
            done += len(batch)
            elapsed = time.monotonic() - t0
            eta = (total - done) / max(done, 1) * elapsed
            sys.stderr.write(f"\r{TAG} {done}/{total} (fail={failed}, ETA={eta:.0f}s)  ")
            sys.stderr.flush()
        reader.flush()

    print(f"\n{TAG} Done: {done - failed} read, {failed} failed in {time.monotonic() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
