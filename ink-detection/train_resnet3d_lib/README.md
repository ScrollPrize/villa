# train_resnet3d_lib Layout

This folder is organized around the main research edit loops.

## Entry points
- `../train_resnet3d.py`: train/validate orchestration entry

## Core files
- **Run setup + trainer wiring**: `training.py`
- **Lightning module (init + train/val runtime)**: `model.py`
- **Dataset pipeline (Zarr-primary backend adapter dispatch)**: `datasets_builder.py`
- **Stitch manager runtime**: `stitch_manager.py`
- **Runtime orchestration + W&B + checkpointing**: `runtime/`
- **Metadata/config validation (authoritative static checks)**: `config_metadata_apply.py`

## Focused algorithm/util files
- **Loss functions**: `modeling/losses.py`
- **GroupDRO objective logic**: `modeling/group_dro.py`
- **Backbone/decoder architecture helpers**: `modeling/architecture.py`
- **Optimizer/scheduler wiring**: `modeling/optimizers_runtime.py`
- **Augmentation + runtime sample transforms**: `data/augmentations.py`

## Data backends and patch plumbing
- **Train/val backend adapters (shared API for Zarr/TIFF)**: `data/segment_trainval.py`
- **Stitch/log-only backend adapters**: `data/segment_stitching.py`
- **Image/zarr readers + metadata helpers**: `data/image_readers.py`, `data/zarr_volume.py`
- **Patch extraction + caches + dataset classes**: `data/patching.py`, `data/patch_index_cache.py`, `data/datasets_runtime.py`
