the tifxyz dataset is intended for 3d ink training using shared 3d patch bboxes across all tifxyz in a dataset.

___

**patch finding**

patches are dataset based.

for each dataset:
- compute the minimum world-space bbox that contains all labeled tifxyz points in that dataset
- tile that bbox with `patch_size` and `overlap_fraction`
- keep only bboxes which:
  - contain tifxyz points from at least one segment
  - contain positive ink supervision from at least one segment

the final patch list is a list of per-patch dictionaries, where each dict contains:
- `world_bbox`
- the shared volume/scale reference
- a list of tifxyz segments that intersect that bbox
- for each tifxyz in the bbox, a padded stored-resolution `(row_start, row_stop, col_start, col_stop)` slice for loading its points
- a `supervised_segment_indices` list used at load time

at `__getitem__` time, one segment is chosen randomly from the supervised subset for that patch, and sampling/projection proceeds from that single tifxyz.

`wrap_mode` controls this selection behavior:
- `"single"` (default): choose one random supervised tifxyz for the patch
- `"all"`: return stacked per-wrap tensors for every supervised tifxyz in the patch

patch computation can be cached to disk , and reused with the `patch_cache_filename` key. for the flat dataset training path, this key is used as an explicit cache path override; otherwise it defaults to `<out_dir>/flat_ink_patches_ps-<patch_size>.json`. patches can be forced to recompute by setting `"patch_cache_force_recompute": true`
___ 

**sampling**

for a given dataset idx , we generate the training data like so:

- we create a initial 3d crop volume with values all set to (2) 
- on the chosen tifxyz segment points which fall within the `world_bbox` , we upsample them to obtain a dense grid
- continuing to operate on 2d points within the `world_bbox` , we compute an EDT on the labeled voxels, and any voxels within `bg_dilate_distance` that do not contain the label are set to value 100 
- we project the bg label of 100 along the segmentations surface normals, up to `bg_distance`, setting any voxels it intersects to 0
- we project the ink label of value 1 along the segmentations surface normals, up to `fg_distance`, setting any intersecting voxels to value 1
- `fg_distance` defaults to `10`; if `bg_distance` is omitted, it reuses `fg_distance`
- any untouched voxels remain the ignore value (2)

___

**dataset format**

a "dataset" is a collection of tifxyz segmentations which belong to a common volume. they are specified within the config json like this. the tifxyz folder shuold be a parent folder containing tifxyz folders.

optional: set `"label_version": "v2"` to force `_inklabels_v2` / `_supervision_mask_v2`. when omitted, the loader uses the highest available version for labels and supervision masks. the unversioned base files are treated as the original version.

```json
"datasets": [
        {
            "volume_path": "/path/to/dataset1/volume1.zarr",
            "volume_scale": 0,
            "segments_path": "/path/to/dataset1/tifxyz"
        },
        {
            "volume_path": "/path/to/dataset2/volume2.zarr",
            "volume_scale": 0,
            "segments_path": "/path/to/dataset2/tifxyz"
        }
```
___

**augmentation**

augmentations are handled by the vesuvius augmentation module, and support both isotropic and anisotropic 3d patch sizes automatically

___

**inspect cli**

you can run the dataset inspect entrypoint with a specific config file:

```bash
python -m tifxyz_dataset.tifxyz_dataset --config /path/to/config.json
```

___

**chunk downloader**

you can download only the shared-volume chunks touched by tifxyz patchfinding with:

```bash
python -m preprocessing.download_required_zarr_chunks \
  --datasets-root /home/sean/Desktop/new_labels_fixed_matched \
  --output-root /home/sean/Desktop/tifxyz_required_zarrs \
  --volumes-json /path/to/volumes.json \
  --dataset 1667 \
  --recompress balanced \
  --overwrite
```

the volumes json can be either a plain dataset-to-path mapping:

```json
{
  "1667": "https://example.org/SCROLLS_HEL_2.399um_78keV_0.22m_PHerc_1667_TA_0001_masked.zarr"
}
```

or objects with explicit scale/auth:

```json
{
  "1667": {
    "volume_path": "https://example.org/SCROLLS_HEL_2.399um_78keV_0.22m_PHerc_1667_TA_0001_masked.zarr",
    "volume_scale": 0,
    "volume_auth_json": "/path/to/auth.json"
  }
}
```

for dataset `1667`, the output is written to `/home/sean/Desktop/tifxyz_required_zarrs/1667.zarr`.
the downloader writes a multiscale mirror: it computes chunk coverage at the requested scale, then scales that coverage to every other numeric scale in the source group and copies those chunks too.
the output is recompressed by default. `--recompress fast` uses blosc lz4 `clevel=5` with bitshuffle; `--recompress balanced` uses blosc zstd `clevel=3` with shuffle.

___

**configurable losses**

the local training script now reads a `loss.terms` list from the config and sums the weighted terms.
only terms listed in `loss.terms` are added. if `loss.terms` is omitted, training falls back to the base BCE + Dice term only.

```json
"loss": {
  "terms": [
    {
      "name": "LabelSmoothedDCAndBCELoss",
      "metric_name": "base",
      "weight": 1.0,
      "weight_dice": 0.25,
      "weight_ce": 1.0
    },
    {
      "name": "MaskedBettiMatchingLoss",
      "metric_name": "betti",
      "weight": 0.05,
      "filtration": "superlevel",
      "include_unmatched_target": false,
      "push_unmatched_to": "diagonal"
    },
    {
      "name": "BoundaryLoss",
      "metric_name": "boundary",
      "weight": 0.1,
      "kernel_size": 3,
      "weight_bce": 1.0,
      "weight_dice": 1.0
    }
  ]
}
```

the current local registry supports:

- `LabelSmoothedDCAndBCELoss`
- `MaskedBettiMatchingLoss`
- `BoundaryLoss`

to build the required `betti_matching` extension locally, run:

```bash
python tifxyz_dataset/build_betti.py
```
