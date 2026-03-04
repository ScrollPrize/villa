"""Data loading and preprocessing modules for the train_resnet3d pipeline.

Layout:
- `ops`: stable public import surface for data helpers
- `readers`: stable reader import surface
- `image_readers`: image/layer/label/mask readers
- `zarr_volume`: zarr volume/path utilities
- `patching`: patch extraction + mask store helpers
- `transforms_runtime`: runtime transforms + augmentation hooks
- `datasets_runtime`: dataset classes and flat-index utilities
- `augmentations`: train/valid augmentation + normalization transforms
- `segments`: stable public segment API used by dataset assembly
- `segment_groups`: group metadata/context helpers
- `segment_trainval`: train/val segment loading implementations
- `segment_stitching`: stitch/train-viz/log-only segment loaders
- `dataloaders`: dataloader factory helpers
- `build_state`: normalized data_state schema
- `patch_index_cache`: patch-index caching and reuse
- `normalization_stats`: fold-level normalization statistics
- `segment_metadata`: compact metadata readers
"""

__all__ = [
    "ops",
    "readers",
    "image_readers",
    "zarr_volume",
    "patching",
    "transforms_runtime",
    "datasets_runtime",
    "augmentations",
    "segments",
    "segment_groups",
    "segment_trainval",
    "segment_stitching",
    "dataloaders",
    "build_state",
    "patch_index_cache",
    "normalization_stats",
    "segment_metadata",
]
