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

patch computation can be cached to disk , and reused with the `patch_cache_filename` key. patches can be forced to recompute by setting `"patch_cache_force_recompute": true`
___ 

**sampling**

for a given dataset idx , we generate the training data like so:

- we create a initial 3d crop volume with values all set to (2) 
- on the chosen tifxyz segment points which fall within the `world_bbox` , we upsample them to obtain a dense grid
- continuing to operate on 2d points within the `world_bbox` , we compute an EDT on the labeled voxels, and any voxels within `bg_dilate_distance` that do not contain the label are set to value 100 
- we project the bg label of 100 along the segmentations surface normals, up to `bg_distance`, setting any voxels it intersects to 0
- we project the ink label of value 1 along the segmentations surface normals, up to `label_distance`, setting any intersecting voxels to value 1
- any untouched voxels remain the ignore value (2)

___

**dataset format**

a "dataset" is a collection of tifxyz segmentations which belong to a common volume. they are specified within the config json like this. the tifxyz folder shuold be a parent folder containing tifxyz folders.

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
