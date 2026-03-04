the tifxyz dataset is intended for 3d ink training on one of two different "modes", both utilizing the same shared patch finding methods. 

___

**patch finding** 

patches are segment based. in each segment, we create 3d bboxes along the points by greedily adding points, starting from 0,0 in the 2d grid, and adding the bbox to a list when it meets requirements for :
- patch size,
- label occupancy,
- segmentation "span" along the principle axis the segment is "moving along" 

this is repeated for each tifxyz segment in the datasets "segments" folder

the final list of patches is a list of per-patch dictionaries, where each dict contains the dataset/segment identity, the accepted 3D bbox (world_bbox), grid metadata
  (z_band, grid_index), coverage stats (valid_point_count, positive_point_count, positive_fraction, span_zyx), and load-time references (segment, volume, scale,
  ink_label_path)

patch computation can be cached to disk , and reused with the `patch_cache_filename` key. patches can be forced to recompute by setting `"patch_cache_force_recompute": true`
___ 

**modes**

in the configuration json , you can specify one of two different "modes". 

*in single_wrap mode* , for a given dataset idx , we generate the training data like so: 

- we create a initial 3d crop volume with values all set to (2) 
- on the segmentation points which fall within the `world_bbox` , we upsample them to obtain a dense grid
- continuing to operate on 2d points within the `world_bbox` , we compute an EDT on the labeled voxels, and any voxels within `bg_dilate_distance` that do not contain the label are set to value 100 
- we project the bg label of 100 along the segmentations surface normals, up to `bg_distance`, setting any voxels it intersects to 0
- we project the ink label of value 1 along the segmentations surface normals, up to `label_distance`, setting any intersecting voxels to value 1
- any untouched voxels remain the ignore value (2) 

*in multi_wrap mode* , the dataset is sampled identically , with one change: 
- when the idx is loaded, we check if any other segmentation containts points which fall within the `world_bbox` of this idx, and if any do, we do the same exact steps as single wrap mode, but for each additional segmentation that lies within this crop, and attempt to limit projection distances to prevent intersecting labels

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