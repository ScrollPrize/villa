def initialize_manager_state(
    manager,
    *,
    stitch_downsample,
    stitch_log_only_downsample,
    stitch_train,
    stitch_train_every_n_epochs,
    stitch_log_only_every_n_epochs,
    stitch_use_roi,
    stitch_val_bboxes,
    stitch_train_bboxes,
    stitch_log_only_bboxes,
):
    manager.downsample = max(1, int(stitch_downsample or 1))
    manager.log_only_downsample = int(stitch_log_only_downsample or manager.downsample)

    manager._roi_meta_by_split = {"val": {}, "train": {}, "log_only": {}}
    manager._roi_buffers_by_split = {"val": {}, "log_only": {}}
    manager._val_loader_to_segment = {}

    manager.train_segment_ids = []
    manager.train_loaders = []
    manager.train_enabled = bool(stitch_train)
    manager.train_every_n_epochs = max(1, int(stitch_train_every_n_epochs or 1))

    manager.log_only_segment_ids = []
    manager.log_only_loaders = []
    manager.log_only_every_n_epochs = max(1, int(stitch_log_only_every_n_epochs or 10))

    manager.borders_by_split = {"train": {}, "val": {}}
    manager.use_roi = bool(stitch_use_roi)
    manager.val_bboxes = dict(stitch_val_bboxes or {})
    manager.train_bboxes = dict(stitch_train_bboxes or {})
    manager.log_only_bboxes = dict(stitch_log_only_bboxes or {})

    manager._gaussian_cache = {}
    manager._gaussian_sigma_scale = 1.0 / 8.0
    manager._gaussian_min_weight = 1e-6


def _register_val_segments(
    manager,
    *,
    stitch_all_val,
    stitch_all_val_shapes,
    stitch_all_val_segment_ids,
    stitch_val_dataloader_idx,
    stitch_pred_shape,
    stitch_segment_id,
):
    if bool(stitch_all_val):
        if stitch_all_val_shapes is None or stitch_all_val_segment_ids is None:
            raise ValueError("stitch_all_val requires stitch_all_val_shapes and stitch_all_val_segment_ids")
        if len(stitch_all_val_shapes) != len(stitch_all_val_segment_ids):
            raise ValueError(
                "stitch_all_val_shapes and stitch_all_val_segment_ids must have the same length "
                f"(got {len(stitch_all_val_shapes)} vs {len(stitch_all_val_segment_ids)})"
            )
        for loader_idx, (segment_id, shape) in enumerate(
            zip(stitch_all_val_segment_ids or [], stitch_all_val_shapes or [])
        ):
            sid = str(segment_id)
            if sid in manager._roi_meta_by_split["val"]:
                raise ValueError(f"duplicate val stitch segment id: {sid!r}")
            bbox = manager.val_bboxes.get(sid)
            manager._register_segment("val", sid, shape, bbox, manager.downsample)
            manager._val_loader_to_segment[int(loader_idx)] = sid
        return

    if stitch_val_dataloader_idx is None or stitch_pred_shape is None:
        return

    sid = str(stitch_segment_id if stitch_segment_id is not None else stitch_val_dataloader_idx)
    bbox = manager.val_bboxes.get(sid)
    manager._register_segment("val", sid, stitch_pred_shape, bbox, manager.downsample)
    manager._val_loader_to_segment[int(stitch_val_dataloader_idx)] = sid


def _register_train_segments(manager, *, stitch_train_segment_ids, stitch_train_shapes):
    if stitch_train_shapes is None and stitch_train_segment_ids is None:
        return
    if stitch_train_shapes is None or stitch_train_segment_ids is None:
        raise ValueError("stitch_train_shapes and stitch_train_segment_ids must both be set or both be None")
    if len(stitch_train_shapes) != len(stitch_train_segment_ids):
        raise ValueError(
            "stitch_train_shapes and stitch_train_segment_ids must have the same length "
            f"(got {len(stitch_train_shapes)} vs {len(stitch_train_segment_ids)})"
        )
    for segment_id, shape in zip(stitch_train_segment_ids or [], stitch_train_shapes or []):
        sid = str(segment_id)
        if sid in manager._roi_meta_by_split["train"]:
            raise ValueError(f"duplicate train stitch segment id: {sid!r}")
        bbox = manager.train_bboxes.get(sid)
        manager._register_segment("train", sid, shape, bbox, manager.downsample)
        manager.train_segment_ids.append(sid)


def _register_log_only_segments(manager, *, stitch_log_only_segment_ids, stitch_log_only_shapes):
    if stitch_log_only_shapes is None and stitch_log_only_segment_ids is None:
        return
    if stitch_log_only_shapes is None or stitch_log_only_segment_ids is None:
        raise ValueError(
            "stitch_log_only_shapes and stitch_log_only_segment_ids must both be set or both be None"
        )
    if len(stitch_log_only_shapes) != len(stitch_log_only_segment_ids):
        raise ValueError(
            "stitch_log_only_shapes and stitch_log_only_segment_ids must have the same length "
            f"(got {len(stitch_log_only_shapes)} vs {len(stitch_log_only_segment_ids)})"
        )
    for segment_id, shape in zip(stitch_log_only_segment_ids or [], stitch_log_only_shapes or []):
        sid = str(segment_id)
        if sid in manager._roi_meta_by_split["log_only"]:
            raise ValueError(f"duplicate log-only stitch segment id: {sid!r}")
        bbox = manager.log_only_bboxes.get(sid)
        manager._register_segment("log_only", sid, shape, bbox, manager.log_only_downsample)
        manager.log_only_segment_ids.append(sid)


def register_initial_segments(
    manager,
    *,
    stitch_all_val,
    stitch_all_val_shapes,
    stitch_all_val_segment_ids,
    stitch_val_dataloader_idx,
    stitch_pred_shape,
    stitch_segment_id,
    stitch_train_shapes,
    stitch_train_segment_ids,
    stitch_log_only_shapes,
    stitch_log_only_segment_ids,
):
    _register_val_segments(
        manager,
        stitch_all_val=stitch_all_val,
        stitch_all_val_shapes=stitch_all_val_shapes,
        stitch_all_val_segment_ids=stitch_all_val_segment_ids,
        stitch_val_dataloader_idx=stitch_val_dataloader_idx,
        stitch_pred_shape=stitch_pred_shape,
        stitch_segment_id=stitch_segment_id,
    )
    _register_train_segments(
        manager,
        stitch_train_segment_ids=stitch_train_segment_ids,
        stitch_train_shapes=stitch_train_shapes,
    )
    _register_log_only_segments(
        manager,
        stitch_log_only_segment_ids=stitch_log_only_segment_ids,
        stitch_log_only_shapes=stitch_log_only_shapes,
    )

    manager.enabled = len(manager._roi_buffers_by_split["val"]) > 0


