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
