from __future__ import annotations

from train_resnet3d_lib.config import log, parse_bool_strict


def resolve_stitch_metadata(merged_config):
    stitch_cfg = dict(merged_config["stitch"])
    training_cfg = dict(merged_config.get("training") or {})
    segment_ids = [str(x).strip() for x in stitch_cfg["segment_ids"]]
    segment_ids = [x for x in segment_ids if x]
    downsample = stitch_cfg.get("downsample")
    if downsample is not None:
        downsample = int(downsample)
        if downsample <= 0:
            raise ValueError(f"metadata_json.stitch.downsample must be > 0, got {downsample}")

    return {
        "segment_ids": segment_ids,
        "mask_suffix": str(stitch_cfg.get("mask_suffix", training_cfg.get("val_mask_suffix", "_val"))),
        "downsample": downsample,
        "schedule": {
            "train_every_n_epochs": stitch_cfg.get("train_every_n_epochs"),
            "eval_every_n_epochs": stitch_cfg.get("eval_every_n_epochs"),
            "eval_plus_one": stitch_cfg.get("eval_plus_one"),
        },
    }


def apply_top_level_stitch_to_cfg(cfg, merged_config):
    stitch_raw = merged_config.get("stitch")
    if stitch_raw is None:
        return

    stitch_cfg = dict(stitch_raw)

    train_every = stitch_cfg.get("train_every_n_epochs")
    eval_every = stitch_cfg.get("eval_every_n_epochs")
    eval_plus_one = stitch_cfg.get("eval_plus_one")
    downsample = stitch_cfg.get("downsample")

    if train_every is not None:
        cfg.stitch_train_every_n_epochs = int(train_every)
    if eval_every is not None:
        cfg.eval_stitch_every_n_epochs = int(eval_every)
    if eval_plus_one is not None:
        cfg.eval_stitch_every_n_epochs_plus_one = parse_bool_strict(
            eval_plus_one,
            key="metadata_json.stitch.eval_plus_one",
        )
    if downsample is not None:
        downsample = int(downsample)
        if downsample <= 0:
            raise ValueError(f"metadata_json.stitch.downsample must be > 0, got {downsample}")
        cfg.stitch_downsample = downsample

    log(
        "stitch schedule "
        f"downsample={int(getattr(cfg, 'stitch_downsample', 1))} "
        f"train_every_n_epochs={int(getattr(cfg, 'stitch_train_every_n_epochs', 1))} "
        f"eval_every_n_epochs={int(getattr(cfg, 'eval_stitch_every_n_epochs', 1))} "
        f"eval_plus_one={bool(getattr(cfg, 'eval_stitch_every_n_epochs_plus_one', False))}"
    )


def validate_stitch_segment_ids(merged_config, segment_ids):
    segments = dict(merged_config.get("segments") or {})
    known_segment_ids = {str(x) for x in segments.keys()}
    missing_segment_ids = [sid for sid in segment_ids if sid not in known_segment_ids]
    if missing_segment_ids:
        raise ValueError(f"stitch segment ids are not defined in metadata_json.segments: {missing_segment_ids!r}")
