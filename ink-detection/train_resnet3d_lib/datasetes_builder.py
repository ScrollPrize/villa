import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from samplers import GroupStratifiedBatchSampler

from train_resnet3d_lib.config import CFG, log
from train_resnet3d_lib.data_ops import (
    build_group_mappings,
    read_image_layers,
    read_image_mask,
    read_image_fragment_mask,
    extract_patches,
    extract_patches_infer,
    get_transforms,
    CustomDataset,
    CustomDatasetTest,
    _mask_border,
    _downsample_bool_mask_any,
    _mask_bbox_downsample,
)


def build_group_metadata(fragment_ids, segments_metadata, group_key):
    group_names, _group_name_to_idx, fragment_to_group_idx = build_group_mappings(
        fragment_ids,
        segments_metadata,
        group_key=group_key,
    )
    return group_names, fragment_to_group_idx


def load_train_segment(
    fragment_id,
    seg_meta,
    group_idx,
    group_name,
    *,
    overlap_segments,
    layers_cache,
    include_train_xyxys,
    label_suffix,
    mask_suffix,
):
    t0 = time.time()
    log(f"load train segment={fragment_id} group={group_name}")
    layers = read_image_layers(
        fragment_id,
        layer_range=seg_meta.get("layer_range"),
    )
    if fragment_id in overlap_segments:
        layers_cache[fragment_id] = layers

    image, mask, fragment_mask = read_image_mask(
        fragment_id,
        reverse_layers=bool(seg_meta.get("reverse_layers", False)),
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
        images=layers,
    )
    log(
        f"loaded train segment={fragment_id} "
        f"image={tuple(image.shape)} label={tuple(mask.shape)} mask={tuple(fragment_mask.shape)} "
        f"in {time.time() - t0:.1f}s"
    )
    log(f"extract train patches segment={fragment_id}")
    t1 = time.time()
    frag_train_images, frag_train_masks, frag_train_xyxys = extract_patches(
        image,
        mask,
        fragment_mask,
        include_xyxys=include_train_xyxys,
        filter_empty_tile=True,
    )
    log(f"patches train segment={fragment_id} n={len(frag_train_images)} in {time.time() - t1:.1f}s")
    patch_count = int(len(frag_train_images))

    stitch_candidate = None
    if include_train_xyxys and patch_count > 0:
        frag_train_xyxys = (
            np.stack(frag_train_xyxys) if len(frag_train_xyxys) > 0 else np.zeros((0, 4), dtype=np.int64)
        )
        stitch_candidate = (
            frag_train_images,
            frag_train_masks,
            frag_train_xyxys,
            group_idx,
            tuple(mask.shape),
        )

    mask_border = None
    if include_train_xyxys:
        mask_border = _mask_border(
            _downsample_bool_mask_any(fragment_mask, int(getattr(CFG, "stitch_downsample", 1)))
        )
    mask_bbox = None
    if bool(getattr(CFG, "stitch_use_roi", False)):
        mask_bbox = _mask_bbox_downsample(fragment_mask, int(getattr(CFG, "stitch_downsample", 1)))

    return {
        "patch_count": patch_count,
        "images": frag_train_images,
        "masks": frag_train_masks,
        "group_idx": group_idx,
        "stitch_candidate": stitch_candidate,
        "mask_border": mask_border,
        "mask_bbox": mask_bbox,
    }


def load_val_segment(
    fragment_id,
    seg_meta,
    group_idx,
    group_name,
    *,
    layers_cache,
    include_train_xyxys,
    valid_transform,
    label_suffix,
    mask_suffix,
):
    t0 = time.time()
    log(f"load val segment={fragment_id} group={group_name}")
    layers = layers_cache.get(fragment_id)
    if layers is None:
        layers = read_image_layers(
            fragment_id,
            layer_range=seg_meta.get("layer_range"),
        )
    else:
        log(f"reuse layers cache for val segment={fragment_id}")

    image_val, mask_val, fragment_mask_val = read_image_mask(
        fragment_id,
        reverse_layers=bool(seg_meta.get("reverse_layers", False)),
        label_suffix=label_suffix,
        mask_suffix=mask_suffix,
        images=layers,
    )
    log(
        f"loaded val segment={fragment_id} "
        f"image={tuple(image_val.shape)} label={tuple(mask_val.shape)} mask={tuple(fragment_mask_val.shape)} "
        f"in {time.time() - t0:.1f}s"
    )
    log(f"extract val patches segment={fragment_id}")
    t1 = time.time()
    frag_val_images, frag_val_masks, frag_val_xyxys = extract_patches(
        image_val,
        mask_val,
        fragment_mask_val,
        include_xyxys=True,
        filter_empty_tile=False,
    )
    log(f"patches val segment={fragment_id} n={len(frag_val_images)} in {time.time() - t1:.1f}s")

    patch_count = int(len(frag_val_images))
    if patch_count == 0:
        return {
            "patch_count": patch_count,
            "val_loader": None,
            "mask_shape": None,
            "mask_border": None,
        }

    frag_val_xyxys = np.stack(frag_val_xyxys) if len(frag_val_xyxys) > 0 else np.zeros((0, 4), dtype=np.int64)
    frag_val_groups = [group_idx] * len(frag_val_images)
    val_dataset = CustomDataset(
        frag_val_images,
        CFG,
        xyxys=frag_val_xyxys,
        labels=frag_val_masks,
        groups=frag_val_groups,
        transform=valid_transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    mask_border = None
    if include_train_xyxys:
        mask_border = _mask_border(
            _downsample_bool_mask_any(fragment_mask_val, int(getattr(CFG, "stitch_downsample", 1)))
        )
    mask_bbox = None
    if bool(getattr(CFG, "stitch_use_roi", False)):
        mask_bbox = _mask_bbox_downsample(fragment_mask_val, int(getattr(CFG, "stitch_downsample", 1)))

    return {
        "patch_count": patch_count,
        "val_loader": val_loader,
        "mask_shape": tuple(mask_val.shape),
        "mask_border": mask_border,
        "mask_bbox": mask_bbox,
    }


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


def build_train_loader(train_images, train_masks, train_groups, group_names, *, train_transform):
    train_dataset = CustomDataset(
        train_images,
        CFG,
        labels=train_masks,
        groups=train_groups,
        transform=train_transform,
    )

    group_array = torch.as_tensor(train_groups, dtype=torch.long)
    group_counts = torch.bincount(group_array, minlength=len(group_names)).float()
    train_group_counts = [int(x) for x in group_counts.tolist()]
    log(f"train group counts {dict(zip(group_names, train_group_counts))}")

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
        train_batch_sampler = GroupStratifiedBatchSampler(
            train_groups,
            batch_size=CFG.train_batch_size,
            seed=getattr(CFG, "seed", 0),
            drop_last=True,
        )

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


def build_train_stitch_loaders(train_fragment_ids, train_stitch_candidates, stitch_segment_id, *, valid_transform):
    train_stitch_loaders = []
    train_stitch_shapes = []
    train_stitch_segment_ids = []
    if not bool(getattr(CFG, "stitch_train", False)):
        return train_stitch_loaders, train_stitch_shapes, train_stitch_segment_ids

    if bool(getattr(CFG, "stitch_all_val", False)):
        requested_ids = [str(fid) for fid in train_fragment_ids if str(fid) in train_stitch_candidates]
    else:
        requested_ids = []
        if stitch_segment_id is not None and str(stitch_segment_id) in train_stitch_candidates:
            requested_ids = [str(stitch_segment_id)]
        else:
            for fid in train_fragment_ids:
                if str(fid) in train_stitch_candidates:
                    requested_ids = [str(fid)]
                    break
        if not requested_ids:
            log(
                "WARNING: stitch_train is enabled but no train segments had stitch candidates. "
                "No train visualization stitch will be produced."
            )

    for segment_id in requested_ids:
        entry = train_stitch_candidates.get(str(segment_id))
        if entry is None:
            continue
        seg_images, seg_masks, seg_xyxys, group_idx, seg_shape = entry
        seg_groups = [int(group_idx)] * len(seg_images)
        train_dataset_viz = CustomDataset(
            seg_images,
            CFG,
            xyxys=seg_xyxys,
            labels=seg_masks,
            groups=seg_groups,
            transform=valid_transform,
        )
        train_loader_viz = DataLoader(
            train_dataset_viz,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        train_stitch_loaders.append(train_loader_viz)
        train_stitch_shapes.append(tuple(seg_shape))
        train_stitch_segment_ids.append(str(segment_id))

    return train_stitch_loaders, train_stitch_shapes, train_stitch_segment_ids


def build_log_only_stitch_loaders(
    log_only_segments,
    *,
    segments_metadata,
    layers_cache,
    valid_transform,
    mask_suffix,
    log_only_downsample,
):
    log_only_loaders = []
    log_only_shapes = []
    log_only_segment_ids = []
    log_only_bboxes = {}

    for fragment_id in log_only_segments:
        seg_meta = segments_metadata.get(fragment_id, {}) or {}
        layers = layers_cache.get(fragment_id)
        if layers is None:
            layers = read_image_layers(
                fragment_id,
                layer_range=seg_meta.get("layer_range"),
            )
        else:
            log(f"reuse layers cache for log-only segment={fragment_id}")

        image, fragment_mask = read_image_fragment_mask(
            fragment_id,
            reverse_layers=bool(seg_meta.get("reverse_layers", False)),
            mask_suffix=mask_suffix,
            images=layers,
        )

        log(f"extract log-only patches segment={fragment_id}")
        t0 = time.time()
        images, xyxys = extract_patches_infer(image, fragment_mask, include_xyxys=True)
        log(f"patches log-only segment={fragment_id} n={len(images)} in {time.time() - t0:.1f}s")
        if len(images) == 0:
            continue

        xyxys = np.stack(xyxys) if len(xyxys) > 0 else np.zeros((0, 4), dtype=np.int64)
        dataset = CustomDatasetTest(images, xyxys, CFG, transform=valid_transform)
        loader = DataLoader(
            dataset,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        log_only_loaders.append(loader)
        log_only_shapes.append(tuple(fragment_mask.shape))
        log_only_segment_ids.append(str(fragment_id))

        bbox = _mask_bbox_downsample(fragment_mask, int(log_only_downsample))
        if bbox is not None:
            log_only_bboxes[str(fragment_id)] = bbox

    return log_only_loaders, log_only_shapes, log_only_segment_ids, log_only_bboxes


def build_datasets(run_state):
    segments_metadata = run_state["segments_metadata"]
    fragment_ids = run_state["fragment_ids"]
    train_fragment_ids = run_state["train_fragment_ids"]
    val_fragment_ids = run_state["val_fragment_ids"]
    group_key = run_state["group_key"]

    group_names, fragment_to_group_idx = build_group_metadata(
        fragment_ids,
        segments_metadata,
        group_key,
    )

    train_transform = get_transforms(data="train", cfg=CFG)
    valid_transform = get_transforms(data="valid", cfg=CFG)

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

    train_images = []
    train_masks = []
    train_groups = []
    train_patch_counts_by_segment = {}
    train_stitch_candidates = {}
    train_mask_borders = {}
    train_mask_bboxes = {}

    val_loaders = []
    val_stitch_shapes = []
    val_stitch_segment_ids = []
    val_patch_counts_by_segment = {}
    val_mask_borders = {}
    val_mask_bboxes = {}
    stitch_val_dataloader_idx = None
    stitch_pred_shape = None
    stitch_segment_id = None

    log("building datasets")
    train_set = set(train_fragment_ids)
    val_set = set(val_fragment_ids)
    overlap_segments = train_set & val_set
    layers_cache = {}
    include_train_xyxys = bool(getattr(CFG, "stitch_train", False))

    for fragment_id in train_fragment_ids:
        seg_meta = segments_metadata.get(fragment_id, {}) or {}
        group_idx = fragment_to_group_idx[fragment_id]
        group_name = group_names[group_idx] if group_idx < len(group_names) else str(group_idx)

        result = load_train_segment(
            fragment_id,
            seg_meta,
            group_idx,
            group_name,
            overlap_segments=overlap_segments,
            layers_cache=layers_cache,
            include_train_xyxys=include_train_xyxys,
            label_suffix=train_label_suffix,
            mask_suffix=train_mask_suffix,
        )

        train_patch_counts_by_segment[fragment_id] = result["patch_count"]
        if result["stitch_candidate"] is not None:
            train_stitch_candidates[str(fragment_id)] = result["stitch_candidate"]
        if result["mask_border"] is not None:
            train_mask_borders[str(fragment_id)] = result["mask_border"]
        if result.get("mask_bbox") is not None:
            train_mask_bboxes[str(fragment_id)] = result["mask_bbox"]
        train_images.extend(result["images"])
        train_masks.extend(result["masks"])
        train_groups.extend([group_idx] * len(result["images"]))

    for fragment_id in val_fragment_ids:
        seg_meta = segments_metadata.get(fragment_id, {}) or {}
        group_idx = fragment_to_group_idx[fragment_id]
        group_name = group_names[group_idx] if group_idx < len(group_names) else str(group_idx)

        result = load_val_segment(
            fragment_id,
            seg_meta,
            group_idx,
            group_name,
            layers_cache=layers_cache,
            include_train_xyxys=include_train_xyxys,
            valid_transform=valid_transform,
            label_suffix=val_label_suffix,
            mask_suffix=val_mask_suffix,
        )

        val_patch_counts_by_segment[fragment_id] = result["patch_count"]
        if result["val_loader"] is None:
            continue

        val_loaders.append(result["val_loader"])
        val_stitch_shapes.append(result["mask_shape"])
        val_stitch_segment_ids.append(fragment_id)
        if result["mask_border"] is not None:
            val_mask_borders[str(fragment_id)] = result["mask_border"]
        if result.get("mask_bbox") is not None:
            val_mask_bboxes[str(fragment_id)] = result["mask_bbox"]
        if fragment_id == CFG.valid_id:
            stitch_val_dataloader_idx = len(val_loaders) - 1
            stitch_pred_shape = result["mask_shape"]
            stitch_segment_id = fragment_id

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

    log(f"dataset built train_patches={len(train_images)} val_loaders={len(val_loaders)}")
    if len(val_loaders) == 0:
        raise ValueError("No validation data was built (all segments produced 0 validation patches).")

    train_loader, train_group_counts = build_train_loader(
        train_images,
        train_masks,
        train_groups,
        group_names,
        train_transform=train_transform,
    )
    steps_per_epoch = log_training_budget(train_loader)

    train_stitch_loaders, train_stitch_shapes, train_stitch_segment_ids = build_train_stitch_loaders(
        train_fragment_ids,
        train_stitch_candidates,
        stitch_segment_id,
        valid_transform=valid_transform,
    )

    log_only_segments = list(getattr(CFG, "stitch_log_only_segments", []) or [])
    log_only_loaders = []
    log_only_shapes = []
    log_only_segment_ids = []
    log_only_bboxes = {}
    if log_only_segments:
        log_only_loaders, log_only_shapes, log_only_segment_ids, log_only_bboxes = build_log_only_stitch_loaders(
            log_only_segments,
            segments_metadata=segments_metadata,
            layers_cache=layers_cache,
            valid_transform=valid_transform,
            mask_suffix=val_mask_suffix,
            log_only_downsample=int(getattr(CFG, "stitch_log_only_downsample", getattr(CFG, "stitch_downsample", 1))),
        )

    return {
        "train_loader": train_loader,
        "val_loaders": val_loaders,
        "group_names": group_names,
        "train_group_counts": train_group_counts,
        "steps_per_epoch": steps_per_epoch,
        "train_stitch_loaders": train_stitch_loaders,
        "train_stitch_shapes": train_stitch_shapes,
        "train_stitch_segment_ids": train_stitch_segment_ids,
        "train_mask_borders": train_mask_borders,
        "train_mask_bboxes": train_mask_bboxes,
        "val_mask_borders": val_mask_borders,
        "val_mask_bboxes": val_mask_bboxes,
        "log_only_stitch_loaders": log_only_loaders,
        "log_only_stitch_shapes": log_only_shapes,
        "log_only_stitch_segment_ids": log_only_segment_ids,
        "log_only_mask_bboxes": log_only_bboxes,
        "include_train_xyxys": include_train_xyxys,
        "stitch_val_dataloader_idx": stitch_val_dataloader_idx,
        "stitch_pred_shape": stitch_pred_shape,
        "stitch_segment_id": stitch_segment_id,
        "val_stitch_shapes": val_stitch_shapes,
        "val_stitch_segment_ids": val_stitch_segment_ids,
    }
