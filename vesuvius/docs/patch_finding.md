# Patch Discovery Overview

The training pipeline pre-screens label volumes to build a list of “valid” patch start positions. This avoids training on empty space and keeps the sampler focused on voxels that contain supervision. The logic lives in `vesuvius.models.datasets.find_valid_patches`, and is orchestrated through the chunk slicer (`vesuvius.models.datasets.slicers.chunk.ChunkSlicer`).

## How It Works

- **Resolution selection** – For each label volume, the finder opens the requested OME-Zarr level (`downsample_level`, default `1`). If the pyramid level is missing it falls back to full resolution and logs the fallback (`find_valid_patches.py:256-299`).
- **Patch enumeration** – Candidate start coordinates are generated in a grid that matches the configured patch size. The step is equal to the patch size for each spatial axis, so every candidate window is non-overlapping at the discovery stage (`find_valid_patches.py:204-243`).
- **Channel collapse** – When a label array has extra axes (e.g. 3×3 surface frames), the code reduces it to a scalar mask per voxel:
  - If you set `valid_patch_channel`, that flattened index or per-axis tuple is used to slice the chosen channel (`find_valid_patches.py:52-111`).
  - Otherwise it computes the L2 norm across the extra dimensions, which works for vector-valued labels (`find_valid_patches.py:107-111`).
- **Mask thresholding** – The reduced patch is binarized with `abs(value) > 0`, producing a boolean mask for subsequent checks (`find_valid_patches.py:128-173`).
- **Quality filters** – Two metrics are calculated per candidate:
  - Bounding-box coverage: the ratio of the labeled bounding box volume to the full patch (`find_valid_patches.py:142-145` and `155-168`).
  - Labeled voxel ratio: the fraction of voxels inside the patch that are non-zero (`find_valid_patches.py:147-149` and `171-173`).
  A patch is kept only if both exceed their configured thresholds.
- **Parallelization and caching** – If `num_workers > 0`, the finder uses a process pool and can cache results via `save_valid_patches`. Setting `num_workers <= 0` forces the sequential path, which shares the same channel-collapse rules (`slicers/chunk.py:160-320`).
- **Fallback enumeration** – After validation, unlabeled volumes (or runs with validation disabled) fall back to a full enumeration so that every stride position is still available if needed (`slicers/chunk.py:216-234`).

## YAML Configuration Knobs

All knobs live under `dataset_config` unless noted otherwise.

- `min_labeled_ratio` – Minimum fraction of voxels that must be labeled within a patch (default `0.10`). Lower it to admit sparser annotations.
- `min_bbox_percent` – Minimum bounding-box coverage relative to the patch size (default `0.95`). Helps reject tiny slivers of mask at the edge of the patch.
- `downsample_level` – Zarr pyramid level to inspect when searching for patches (default `1`). Use `0` to operate at full resolution.
- `num_workers` – Process count for the patch finder. Set to `0` (or omit) to run sequentially, which is useful when multiprocessing is restricted.
- `skip_patch_validation` – If `true`, bypasses the labeled-voxel checks entirely and enumerates every stride position. Handy for debugging, but unlabeled patches will be included.
- `cache_valid_patches` – Enables on-disk caching of discovery results so repeated runs don’t rescan large volumes.
- `allow_unlabeled_data` – Allows volumes without labels to participate in the dataset. Those volumes skip validation but still contribute patches via the fallback enumeration.
- `chunk_stride` (top-level attribute read by the manager) – Overrides the stride used when enumerating post-validation patches. The default stride equals the patch size.
- `targets.<name>.valid_patch_channel` – Optional channel selector for multi-channel labels. Accepts:
  - An integer `flatten_index` referencing the flattened extra axes.
  - A tuple/list of indices matching the extra axes (e.g. `[2, 2]` to pick the normal’s Z component in a 3×3 surface frame).
  - A dict wrapper such as `{indices: [2, 2]}` or `{flatten_index: 8}`.
  If omitted, the finder uses the L2 norm across all channels.

## Practical Tips
- Keep `cache_valid_patches` enabled during experimentation. Delete the cache or bump a config parameter to force a refresh after significant changes to the labels.
