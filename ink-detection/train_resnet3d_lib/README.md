# train_resnet3d_lib Layout

This folder is organized by workflow concern so common iteration loops are easy to find.

## Entry points
- `../train_resnet3d.py`: train/validate orchestration entry

## Where to modify common things
- **Training loop/trainer setup**: `training.py`
- **Model architecture + Lightning module**: `model.py`
- **Model init wiring (backbone/decoder/stitcher)**: `modeling/runtime_init.py`
- **Train/val step runtime logic**: `modeling/train_val_runtime.py`
- **Optimizer/scheduler wiring**: `modeling/optimizers_runtime.py`
- **Model architecture blocks (decoder/norm conversion)**: `modeling/architecture.py`
- **Per-sample losses**: `modeling/losses.py`
- **Model/objective config wiring**: `modeling/model_config.py`
- **Loss recipe + soft-label controls from metadata**: `config_metadata_apply.py`, `modeling/runtime_init.py`
- **GroupDRO logic**: `modeling/group_dro.py`
- **Metadata -> CFG application**: `config_metadata_apply.py`
- **Augmentations + normalization transforms**: `data/augmentations.py`
- **Dataset build pipeline**: `datasets_builder.py`
- **Segment API (stable import surface)**: `data/segments.py`
- **Group metadata/context logic**: `data/segment_groups.py`
- **Train/val segment loading logic**: `data/segment_trainval.py`
- **Stitch/log-only segment loading logic**: `data/segment_stitching.py`
- **Image/zarr readers + volume utils**: `data/readers.py`
- **Image/layer/label/mask readers**: `data/image_readers.py`
- **Zarr volume/path utilities**: `data/zarr_volume.py`
- **Patch extraction + mask-store internals**: `data/patching.py`
- **Dataset classes/index flattening**: `data/datasets_runtime.py`
- **Runtime transform hooks**: `data/transforms_runtime.py`
- **Stable data API surface**: `data/ops.py`
- **Patch-index caching**: `data/patch_index_cache.py`
- **Fold normalization stats**: `data/normalization_stats.py`
- **Stitch manager orchestration**: `stitch_manager.py`
- **Stitch internals (roi/pass/buffer/media/metrics)**: `stitching/`
- **Stitch manager init/runtime helpers**: `stitching/manager_setup.py`, `stitching/manager_runtime.py`
- **Stitch metrics config + aggregation helpers**: `stitching/metrics_runtime_config.py`, `stitching/metrics_runtime_aggregation.py`
- **Run/orchestration/W&B/checkpoint runtime**: `runtime/`
