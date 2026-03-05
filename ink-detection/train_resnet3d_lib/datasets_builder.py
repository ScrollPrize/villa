import math

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from samplers import GroupStratifiedBatchSampler

from train_resnet3d_lib.config import CFG, log
from train_resnet3d_lib.data.datasets_runtime import (
    CustomDataset,
    LazyZarrTrainDataset,
)
from train_resnet3d_lib.data.image_readers import build_group_mappings
from train_resnet3d_lib.data.segment_metadata import get_segment_meta as _segment_meta
from train_resnet3d_lib.data.transforms_runtime import get_transforms
from train_resnet3d_lib.data.normalization_stats import (
    prepare_fold_label_foreground_percentile_clip_zscore_stats,
)
from train_resnet3d_lib.data.segment_trainval import (
    load_train_segment_for_backend,
    load_val_segment_for_backend,
)
from train_resnet3d_lib.data.segment_stitching import (
    build_train_stitch_outputs,
    build_log_only_outputs,
)

_SUPPORTED_DATA_BACKENDS = ("zarr", "tiff")


def _normalize_data_backend(data_backend):
    backend = str(data_backend).strip().lower()
    if backend not in _SUPPORTED_DATA_BACKENDS:
        raise ValueError(f"Unknown training.data_backend: {data_backend!r}. Expected 'zarr' or 'tiff'.")
    return backend


def init_dataset_tracking(*, include_train_xyxys):
    return {
        "include_train_xyxys": bool(include_train_xyxys),
        "val_loaders": [],
        "val_stitch_shapes": [],
        "val_stitch_segment_ids": [],
        "val_mask_borders": {},
        "val_mask_bboxes": {},
        "stitch_val_dataloader_idx": None,
        "stitch_pred_shape": None,
        "stitch_segment_id": None,
    }


def append_val_entry(
    tracking,
    *,
    fragment_id,
    val_loader,
    mask_shape,
    mask_border=None,
    mask_bbox=None,
    valid_id=None,
):
    tracking["val_loaders"].append(val_loader)
    tracking["val_stitch_shapes"].append(mask_shape)
    tracking["val_stitch_segment_ids"].append(fragment_id)

    if mask_border is not None:
        tracking["val_mask_borders"][str(fragment_id)] = mask_border
    if mask_bbox is not None:
        tracking["val_mask_bboxes"][str(fragment_id)] = mask_bbox

    if fragment_id == valid_id:
        tracking["stitch_val_dataloader_idx"] = len(tracking["val_loaders"]) - 1
        tracking["stitch_pred_shape"] = mask_shape
        tracking["stitch_segment_id"] = fragment_id


def build_data_state(
    *,
    train_loader,
    group_names,
    group_idx_by_segment,
    train_group_counts,
    steps_per_epoch,
    train_stitch_loaders,
    train_stitch_shapes,
    train_stitch_segment_ids,
    train_mask_borders,
    train_mask_bboxes,
    log_only_loaders,
    log_only_shapes,
    log_only_segment_ids,
    log_only_bboxes,
    tracking,
):
    return {
        "train_loader": train_loader,
        "val_loaders": tracking["val_loaders"],
        "group_names": group_names,
        "group_idx_by_segment": group_idx_by_segment,
        "train_group_counts": train_group_counts,
        "steps_per_epoch": steps_per_epoch,
        "train_stitch_loaders": train_stitch_loaders,
        "train_stitch_shapes": train_stitch_shapes,
        "train_stitch_segment_ids": train_stitch_segment_ids,
        "train_mask_borders": train_mask_borders,
        "train_mask_bboxes": train_mask_bboxes,
        "val_mask_borders": tracking["val_mask_borders"],
        "val_mask_bboxes": tracking["val_mask_bboxes"],
        "log_only_stitch_loaders": log_only_loaders,
        "log_only_stitch_shapes": log_only_shapes,
        "log_only_stitch_segment_ids": log_only_segment_ids,
        "log_only_mask_bboxes": log_only_bboxes,
        "include_train_xyxys": tracking["include_train_xyxys"],
        "stitch_val_dataloader_idx": tracking["stitch_val_dataloader_idx"],
        "stitch_pred_shape": tracking["stitch_pred_shape"],
        "stitch_segment_id": tracking["stitch_segment_id"],
        "val_stitch_shapes": tracking["val_stitch_shapes"],
        "val_stitch_segment_ids": tracking["val_stitch_segment_ids"],
    }


def _build_group_metadata(fragment_ids, segments_metadata, group_key):
    group_names, _group_name_to_idx, fragment_to_group_idx = build_group_mappings(
        fragment_ids,
        segments_metadata,
        group_key=group_key,
    )
    return group_names, fragment_to_group_idx


def _segment_group_context(fragment_id, segments_metadata, fragment_to_group_idx, group_names):
    seg_meta = _segment_meta(segments_metadata, fragment_id)
    group_idx = int(fragment_to_group_idx[fragment_id])
    group_name = group_names[group_idx] if group_idx < len(group_names) else str(group_idx)
    return seg_meta, group_idx, group_name


def summarize_patch_counts(split_name, fragment_ids_list, counts_by_segment, *, group_names, fragment_to_group_idx):
    total = int(sum(int(counts_by_segment.get(fid, 0)) for fid in fragment_ids_list))
    counts_by_group = {name: 0 for name in group_names}
    for fid in fragment_ids_list:
        n = int(counts_by_segment.get(fid, 0))
        gidx = fragment_to_group_idx.get(fid, 0)
        gname = group_names[gidx] if gidx < len(group_names) else str(gidx)
        counts_by_group[gname] = int(counts_by_group.get(gname, 0)) + n

    log(f"{split_name} patch counts total={total}")
    for fid in fragment_ids_list:
        n = int(counts_by_segment.get(fid, 0))
        gidx = fragment_to_group_idx.get(fid, 0)
        gname = group_names[gidx] if gidx < len(group_names) else str(gidx)
        log(f"  {split_name} segment={fid} group={gname} patches={n}")
    log(f"{split_name} patch counts by group {counts_by_group}")


def _validate_group_universe_for_objective(group_names, train_group_counts):
    zero_groups = [
        str(group_name)
        for group_name, group_count in zip(group_names, train_group_counts)
        if int(group_count) <= 0
    ]
    if not zero_groups:
        return
    log(f"train groups with zero patches: {zero_groups}")
    objective = str(getattr(CFG, "objective", "erm")).strip().lower()
    if objective == "group_dro":
        raise ValueError(
            "training.objective=group_dro requires every configured group to have at least one training patch; "
            f"found zero-patch groups: {zero_groups}. "
            "Check effective training.train_segments/sweep overrides and segment masks."
        )


def _collect_train_segments(train_fragment_ids, *, load_train_fn, consume_train_fn):
    train_patch_counts_by_segment = {}
    train_mask_borders = {}
    train_mask_bboxes = {}
    for fragment_id in train_fragment_ids:
        result = load_train_fn(fragment_id)
        patch_count = int(result["patch_count"])
        train_patch_counts_by_segment[fragment_id] = patch_count

        mask_border = result.get("mask_border")
        if mask_border is not None:
            train_mask_borders[str(fragment_id)] = mask_border
        mask_bbox = result.get("mask_bbox")
        if mask_bbox is not None:
            train_mask_bboxes[str(fragment_id)] = mask_bbox

        consume_train_fn(fragment_id, result)
    return train_patch_counts_by_segment, train_mask_borders, train_mask_bboxes


def _collect_val_segments(val_fragment_ids, *, load_val_fn, tracking):
    val_patch_counts_by_segment = {}
    for fragment_id in val_fragment_ids:
        result = load_val_fn(fragment_id)
        val_patch_counts_by_segment[fragment_id] = int(result["patch_count"])
        if result["val_loader"] is None:
            continue
        append_val_entry(
            tracking,
            fragment_id=fragment_id,
            val_loader=result["val_loader"],
            mask_shape=result["mask_shape"],
            mask_border=result["mask_border"],
            mask_bbox=result.get("mask_bbox"),
            valid_id=CFG.valid_id,
        )
    return val_patch_counts_by_segment


def build_train_loader(train_images, train_masks, train_groups, group_names, *, train_transform):
    train_dataset = CustomDataset(
        train_images,
        CFG,
        labels=train_masks,
        groups=train_groups,
        transform=train_transform,
    )
    return _build_train_loader_from_dataset(train_dataset, train_groups, group_names)


def _build_train_loader_from_dataset(train_dataset, train_groups, group_names):
    group_array = torch.as_tensor(train_groups, dtype=torch.long)
    group_counts = torch.bincount(group_array, minlength=len(group_names)).float()
    train_group_counts = [int(x) for x in group_counts.tolist()]
    log(f"train group counts {dict(zip(group_names, train_group_counts))}")
    _validate_group_universe_for_objective(group_names, train_group_counts)

    if CFG.sampler == "shuffle":
        train_sampler = None
        train_shuffle = True
        train_batch_sampler = None
    elif CFG.sampler == "group_balanced":
        group_weights = len(train_dataset) / group_counts.clamp_min(1)
        weights = group_weights[group_array]
        train_sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
        train_shuffle = False
        train_batch_sampler = None
    elif CFG.sampler == "group_stratified":
        train_sampler = None
        train_shuffle = False
        epoch_size_mode = str(getattr(CFG, "group_stratified_epoch_size_mode", "dataset")).strip().lower()
        log(f"group_stratified sampler epoch_size_mode={epoch_size_mode!r}")
        train_batch_sampler = GroupStratifiedBatchSampler(
            train_groups,
            batch_size=CFG.train_batch_size,
            seed=getattr(CFG, "seed", 0),
            drop_last=True,
            epoch_size_mode=epoch_size_mode,
        )
    else:
        raise ValueError(f"Unknown training.sampler: {CFG.sampler!r}")

    if train_batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.train_batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    return train_loader, train_group_counts


def build_train_loader_lazy(
    train_volumes_by_segment,
    train_masks_by_segment,
    train_xyxys_by_segment,
    train_sample_bbox_indices_by_segment,
    train_groups_by_segment,
    group_names,
    *,
    train_transform,
):
    train_dataset = LazyZarrTrainDataset(
        train_volumes_by_segment,
        train_masks_by_segment,
        train_xyxys_by_segment,
        train_groups_by_segment,
        CFG,
        transform=train_transform,
        sample_bbox_indices_by_segment=train_sample_bbox_indices_by_segment,
    )
    train_groups = [int(x) for x in train_dataset.sample_groups.tolist()]
    return _build_train_loader_from_dataset(train_dataset, train_groups, group_names)


def log_training_budget(train_loader):
    steps_per_epoch = len(train_loader)
    accum = int(getattr(CFG, "accumulate_grad_batches", 1) or 1)
    if accum > 1:
        steps_per_epoch = int(math.ceil(steps_per_epoch / accum))

    micro_steps_per_epoch = int(len(train_loader))
    optimizer_steps_per_epoch = int(steps_per_epoch)
    total_optimizer_steps = int(optimizer_steps_per_epoch * int(CFG.epochs))
    effective_batch_size = int(int(CFG.train_batch_size) * int(accum))

    log(
        "train budget "
        f"len(train_loader)={micro_steps_per_epoch} accumulate_grad_batches={accum} "
        f"optimizer_steps_per_epoch={optimizer_steps_per_epoch} epochs={int(CFG.epochs)} "
        f"total_optimizer_steps={total_optimizer_steps} effective_batch_size={effective_batch_size}"
    )
    log(
        "scheduler budget "
        f"scheduler={getattr(CFG, 'scheduler', None)!r} "
        f"onecycle steps_per_epoch={optimizer_steps_per_epoch} epochs={int(CFG.epochs)} "
        f"max_lr={float(CFG.lr)} div_factor={float(getattr(CFG, 'onecycle_div_factor', 25.0))} "
        f"pct_start={float(getattr(CFG, 'onecycle_pct_start', 0.15))}"
    )
    return steps_per_epoch


def _load_with_group_context(fragment_id, *, segments_metadata, fragment_to_group_idx, group_names, loader_fn, **kwargs):
    seg_meta, group_idx, group_name = _segment_group_context(
        fragment_id,
        segments_metadata,
        fragment_to_group_idx,
        group_names,
    )
    return loader_fn(fragment_id, seg_meta, group_idx, group_name, **kwargs)


def _segment_group_name(*, segments_metadata, fragment_id, group_key):
    if fragment_id not in segments_metadata:
        raise KeyError(f"segments metadata missing segment id: {fragment_id!r}")
    seg_meta = segments_metadata[fragment_id]
    if not isinstance(seg_meta, dict):
        raise TypeError(
            f"segments[{fragment_id!r}] must be an object, got {type(seg_meta).__name__}"
        )
    if group_key not in seg_meta:
        raise KeyError(f"segments[{fragment_id!r}] missing required group key {group_key!r}")
    return str(seg_meta[group_key])


def _build_group_metadata_for_active_splits(
    *,
    segments_metadata,
    train_fragment_ids,
    val_fragment_ids,
    log_only_segments,
    group_key,
):
    group_names, fragment_to_group_idx = _build_group_metadata(
        train_fragment_ids,
        segments_metadata,
        group_key,
    )
    group_name_to_idx = {str(name): int(idx) for idx, name in enumerate(group_names)}

    def _assign_if_needed(fragment_id, *, split_name):
        if fragment_id in fragment_to_group_idx:
            return
        group_name = _segment_group_name(
            segments_metadata=segments_metadata,
            fragment_id=fragment_id,
            group_key=group_key,
        )
        if group_name not in group_name_to_idx:
            raise ValueError(
                f"{split_name} segment {fragment_id!r} belongs to group {group_name!r}, "
                "which is not represented in training.train_segments. "
                "Group-based logging/objectives require every active split group to appear in train."
            )
        fragment_to_group_idx[fragment_id] = int(group_name_to_idx[group_name])

    for fragment_id in val_fragment_ids:
        _assign_if_needed(fragment_id, split_name="validation")
    for fragment_id in log_only_segments:
        _assign_if_needed(fragment_id, split_name="stitch_log_only")

    return group_names, fragment_to_group_idx


def _build_datasets_for_backend(
    *,
    data_backend,
    segments_metadata,
    train_fragment_ids,
    val_fragment_ids,
    group_names,
    fragment_to_group_idx,
    group_idx_by_segment,
    train_label_suffix,
    train_mask_suffix,
    val_label_suffix,
    val_mask_suffix,
    train_transform,
    valid_transform,
    log_only_segments,
    log_only_downsample,
    shared_volume_cache=None,
):
    backend = _normalize_data_backend(data_backend)
    is_zarr = backend == "zarr"
    include_train_xyxys = bool(getattr(CFG, "stitch_train", False))
    tracking = init_dataset_tracking(include_train_xyxys=include_train_xyxys)

    train_groups_by_segment = {str(fid): int(fragment_to_group_idx[fid]) for fid in train_fragment_ids}
    train_volumes_by_segment = {}
    train_masks_by_segment = {}
    train_xyxys_by_segment = {}
    train_sample_bbox_indices_by_segment = {}

    train_images = []
    train_masks = []
    train_groups = []
    train_stitch_candidates = {} if not is_zarr else None
    layers_cache = {} if not is_zarr else None
    overlap_segments = set(train_fragment_ids) & set(val_fragment_ids)

    if is_zarr:
        if shared_volume_cache is None:
            raise ValueError("shared_volume_cache is required when training.data_backend is 'zarr'")
    log(f"building datasets ({backend}{' lazy' if is_zarr else ''})")

    def load_train_for_backend(fragment_id):
        return _load_with_group_context(
            fragment_id,
            segments_metadata=segments_metadata,
            fragment_to_group_idx=fragment_to_group_idx,
            group_names=group_names,
            loader_fn=load_train_segment_for_backend,
            data_backend=backend,
            overlap_segments=overlap_segments,
            layers_cache=layers_cache,
            volume_cache=shared_volume_cache,
            include_train_xyxys=include_train_xyxys,
            label_suffix=train_label_suffix,
            mask_suffix=train_mask_suffix,
        )

    def load_val_for_backend(fragment_id):
        return _load_with_group_context(
            fragment_id,
            segments_metadata=segments_metadata,
            fragment_to_group_idx=fragment_to_group_idx,
            group_names=group_names,
            loader_fn=load_val_segment_for_backend,
            data_backend=backend,
            layers_cache=layers_cache,
            volume_cache=shared_volume_cache,
            include_train_xyxys=include_train_xyxys,
            valid_transform=valid_transform,
            label_suffix=val_label_suffix,
            mask_suffix=val_mask_suffix,
        )

    def consume_train_for_backend(fragment_id, result):
        if is_zarr:
            if int(result["patch_count"]) <= 0:
                return
            sid = result["sid"]
            train_volumes_by_segment[sid] = result["volume"]
            train_masks_by_segment[sid] = result["mask_store"]
            train_xyxys_by_segment[sid] = result["xyxys"]
            train_sample_bbox_indices_by_segment[sid] = result["sample_bbox_indices"]
            return

        if train_stitch_candidates is not None and result["stitch_candidate"] is not None:
            train_stitch_candidates[str(fragment_id)] = result["stitch_candidate"]
        train_images.extend(result["images"])
        train_masks.extend(result["masks"])
        train_groups.extend([int(result["group_idx"])] * len(result["images"]))

    train_patch_counts_by_segment, train_mask_borders, train_mask_bboxes = _collect_train_segments(
        train_fragment_ids,
        load_train_fn=load_train_for_backend,
        consume_train_fn=consume_train_for_backend,
    )

    val_patch_counts_by_segment = _collect_val_segments(
        val_fragment_ids,
        load_val_fn=load_val_for_backend,
        tracking=tracking,
    )

    summarize_patch_counts(
        "train",
        train_fragment_ids,
        train_patch_counts_by_segment,
        group_names=group_names,
        fragment_to_group_idx=fragment_to_group_idx,
    )
    summarize_patch_counts(
        "val",
        val_fragment_ids,
        val_patch_counts_by_segment,
        group_names=group_names,
        fragment_to_group_idx=fragment_to_group_idx,
    )

    if is_zarr:
        train_patches_total = int(sum(int(v) for v in train_patch_counts_by_segment.values()))
    else:
        train_patches_total = int(len(train_images))
    log(
        f"dataset built ({backend}) "
        f"train_patches={train_patches_total} val_loaders={len(tracking['val_loaders'])}"
    )
    if train_patches_total == 0:
        raise ValueError("No training data was built (all segments produced 0 training patches).")
    if len(tracking["val_loaders"]) == 0:
        raise ValueError("No validation data was built (all segments produced 0 validation patches).")

    if is_zarr:
        train_loader, train_group_counts = build_train_loader_lazy(
            train_volumes_by_segment,
            train_masks_by_segment,
            train_xyxys_by_segment,
            train_sample_bbox_indices_by_segment,
            train_groups_by_segment,
            group_names,
            train_transform=train_transform,
        )
    else:
        train_loader, train_group_counts = build_train_loader(
            train_images,
            train_masks,
            train_groups,
            group_names,
            train_transform=train_transform,
        )
    steps_per_epoch = log_training_budget(train_loader)

    train_stitch_kwargs = {
        "data_backend": backend,
        "train_fragment_ids": train_fragment_ids,
        "stitch_segment_id": tracking["stitch_segment_id"],
        "valid_transform": valid_transform,
    }
    if is_zarr:
        train_stitch_kwargs.update(
            {
                "train_volumes_by_segment": train_volumes_by_segment,
                "train_masks_by_segment": train_masks_by_segment,
                "train_xyxys_by_segment": train_xyxys_by_segment,
                "train_sample_bbox_indices_by_segment": train_sample_bbox_indices_by_segment,
                "train_groups_by_segment": train_groups_by_segment,
            }
        )
    else:
        train_stitch_kwargs["train_stitch_candidates"] = train_stitch_candidates
    train_stitch_loaders, train_stitch_shapes, train_stitch_segment_ids = build_train_stitch_outputs(
        **train_stitch_kwargs
    )

    log_only_kwargs = {
        "data_backend": backend,
        "log_only_segments": log_only_segments,
        "segments_metadata": segments_metadata,
        "valid_transform": valid_transform,
        "mask_suffix": val_mask_suffix,
        "log_only_downsample": log_only_downsample,
    }
    if is_zarr:
        log_only_kwargs["volume_cache"] = shared_volume_cache
    else:
        log_only_kwargs["layers_cache"] = layers_cache
    log_only_loaders, log_only_shapes, log_only_segment_ids, log_only_bboxes = build_log_only_outputs(
        **log_only_kwargs
    )

    return build_data_state(
        train_loader=train_loader,
        group_names=group_names,
        group_idx_by_segment=group_idx_by_segment,
        train_group_counts=train_group_counts,
        steps_per_epoch=steps_per_epoch,
        train_stitch_loaders=train_stitch_loaders,
        train_stitch_shapes=train_stitch_shapes,
        train_stitch_segment_ids=train_stitch_segment_ids,
        train_mask_borders=train_mask_borders,
        train_mask_bboxes=train_mask_bboxes,
        log_only_loaders=log_only_loaders,
        log_only_shapes=log_only_shapes,
        log_only_segment_ids=log_only_segment_ids,
        log_only_bboxes=log_only_bboxes,
        tracking=tracking,
    )


def build_datasets(run_state):
    segments_metadata = run_state["segments_metadata"]
    train_fragment_ids = run_state["train_fragment_ids"]
    val_fragment_ids = run_state["val_fragment_ids"]
    group_key = run_state["group_key"]
    log_only_segments = [str(x) for x in (getattr(CFG, "stitch_log_only_segments", []) or [])]

    group_names, fragment_to_group_idx = _build_group_metadata_for_active_splits(
        segments_metadata=segments_metadata,
        train_fragment_ids=train_fragment_ids,
        val_fragment_ids=val_fragment_ids,
        log_only_segments=log_only_segments,
        group_key=group_key,
    )
    log(f"group universe n_groups={len(group_names)} groups={group_names}")
    group_idx_by_segment = {str(fragment_id): int(group_idx) for fragment_id, group_idx in fragment_to_group_idx.items()}

    train_label_suffix = getattr(CFG, "train_label_suffix", "")
    train_mask_suffix = getattr(CFG, "train_mask_suffix", "")
    val_label_suffix = getattr(CFG, "val_label_suffix", "_val")
    val_mask_suffix = getattr(CFG, "val_mask_suffix", "_val")
    cv_fold = getattr(CFG, "cv_fold", None)
    log(
        "label/mask suffixes "
        f"cv_fold={cv_fold!r} "
        f"train=(label={train_label_suffix!r}, mask={train_mask_suffix!r}) "
        f"val=(label={val_label_suffix!r}, mask={val_mask_suffix!r})"
    )

    data_backend = _normalize_data_backend(getattr(CFG, "data_backend", "zarr"))
    log(f"data backend={data_backend}")
    if bool(getattr(CFG, "dataset_cache_enabled", True)) and not bool(getattr(CFG, "dataset_cache_check_hash", True)):
        log("WARNING: dataset cache hash validation is disabled (metadata.training.dataset_cache_check_hash=false)")

    shared_volume_cache = {}
    prepare_fold_label_foreground_percentile_clip_zscore_stats(
        segments_metadata=segments_metadata,
        train_fragment_ids=train_fragment_ids,
        data_backend=data_backend,
        train_label_suffix=train_label_suffix,
        train_mask_suffix=train_mask_suffix,
        volume_cache=shared_volume_cache,
    )

    train_transform = get_transforms(data="train", cfg=CFG)
    valid_transform = get_transforms(data="valid", cfg=CFG)
    log_only_downsample = int(getattr(CFG, "stitch_log_only_downsample", getattr(CFG, "stitch_downsample", 1)))

    return _build_datasets_for_backend(
        data_backend=data_backend,
        segments_metadata=segments_metadata,
        train_fragment_ids=train_fragment_ids,
        val_fragment_ids=val_fragment_ids,
        group_names=group_names,
        fragment_to_group_idx=fragment_to_group_idx,
        group_idx_by_segment=group_idx_by_segment,
        train_label_suffix=train_label_suffix,
        train_mask_suffix=train_mask_suffix,
        val_label_suffix=val_label_suffix,
        val_mask_suffix=val_mask_suffix,
        train_transform=train_transform,
        valid_transform=valid_transform,
        log_only_segments=log_only_segments,
        log_only_downsample=log_only_downsample,
        shared_volume_cache=shared_volume_cache,
    )
