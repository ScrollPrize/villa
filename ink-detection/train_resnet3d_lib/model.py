from dataclasses import asdict, is_dataclass
from types import SimpleNamespace

import pytorch_lightning as pl
import torch

from train_resnet3d_lib.modeling.losses import build_bce_targets, compute_per_sample_loss_and_dice
from train_resnet3d_lib.modeling.optimizers_runtime import configure_optimizers as configure_optimizers_runtime
from train_resnet3d_lib.modeling.runtime_init import (
    initialize_regression_state,
    save_regression_hyperparameters,
)
from train_resnet3d_lib.modeling.train_val_runtime import (
    accumulate_train_stats,
    accumulate_validation_stats,
    compute_objective_loss,
    finalize_training_batch,
    initialize_validation_metrics,
    log_train_epoch_metrics,
    log_validation_epoch_metrics,
    reset_train_epoch_accumulators,
    reset_validation_epoch_accumulators,
    sync_validation_accumulators,
    update_validation_stream_metrics,
)


def _cfg_to_dict(value, *, key):
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return dict(asdict(value))
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"{key} must be a dict/dataclass/object with attributes, got {type(value).__name__}")


def _coerce_shape_tuple(value, *, key):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be a 2-element sequence, got {value!r}")
    return int(value[0]), int(value[1])


def _coerce_shape_list(value, *, key):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a list of 2-element sequences")
    out = []
    for idx, item in enumerate(value):
        out.append(_coerce_shape_tuple(item, key=f"{key}[{idx}]"))
    return out


def _coerce_model_cfg(model_cfg):
    data = _cfg_to_dict(model_cfg, key="model_cfg")
    n_groups = int(data.get("n_groups", 1) or 1)
    group_names = data.get("group_names")
    if group_names is None:
        group_names = [str(i) for i in range(n_groups)]
    else:
        group_names = [str(x) for x in group_names]
    stitch_group_idx_by_segment = {
        str(segment_id): int(group_idx)
        for segment_id, group_idx in dict(data.get("stitch_group_idx_by_segment") or {}).items()
    }
    return {
        "size": int(data.get("size", 256)),
        "enc": str(data.get("enc", "i3d")),
        "with_norm": bool(data.get("with_norm", False)),
        "total_steps": int(data.get("total_steps", 1)),
        "n_groups": n_groups,
        "group_names": group_names,
        "stitch_group_idx_by_segment": stitch_group_idx_by_segment,
        "norm": str(data.get("norm", "batch")),
        "group_norm_groups": int(data.get("group_norm_groups", 32)),
    }


def _coerce_objective_cfg(objective_cfg):
    data = _cfg_to_dict(objective_cfg, key="objective_cfg")
    return {
        "objective": str(data.get("objective", "erm")),
        "loss_mode": str(data.get("loss_mode", "batch")),
        "loss_recipe": str(data.get("loss_recipe", "dice_bce")).lower(),
        "bce_smooth_factor": float(data.get("bce_smooth_factor", 0.25)),
        "soft_label_positive": float(data.get("soft_label_positive", 1.0)),
        "soft_label_negative": float(data.get("soft_label_negative", 0.0)),
        "robust_step_size": data.get("robust_step_size"),
        "group_counts": [int(x) for x in list(data.get("group_counts") or [])],
        "group_dro_gamma": float(data.get("group_dro_gamma", 0.1)),
        "group_dro_btl": bool(data.get("group_dro_btl", False)),
        "group_dro_alpha": data.get("group_dro_alpha"),
        "group_dro_normalize_loss": bool(data.get("group_dro_normalize_loss", False)),
        "group_dro_min_var_weight": float(data.get("group_dro_min_var_weight", 0.0)),
        "group_dro_adj": data.get("group_dro_adj"),
        "erm_group_topk": int(data.get("erm_group_topk", 0)),
    }


def _coerce_stitch_cfg(stitch_cfg):
    data = _cfg_to_dict(stitch_cfg, key="stitch_cfg")
    downsample = max(1, int(data.get("stitch_downsample", 1) or 1))
    return {
        "stitch_val_dataloader_idx": (
            None
            if data.get("stitch_val_dataloader_idx") is None
            else int(data["stitch_val_dataloader_idx"])
        ),
        "stitch_pred_shape": _coerce_shape_tuple(
            data.get("stitch_pred_shape"),
            key="stitch_cfg.stitch_pred_shape",
        ),
        "stitch_segment_id": (
            None if data.get("stitch_segment_id") is None else str(data["stitch_segment_id"])
        ),
        "stitch_all_val": bool(data.get("stitch_all_val", False)),
        "stitch_downsample": downsample,
        "stitch_all_val_shapes": _coerce_shape_list(
            data.get("stitch_all_val_shapes"),
            key="stitch_cfg.stitch_all_val_shapes",
        ),
        "stitch_all_val_segment_ids": [str(x) for x in (data.get("stitch_all_val_segment_ids") or [])],
        "stitch_train_shapes": _coerce_shape_list(
            data.get("stitch_train_shapes"),
            key="stitch_cfg.stitch_train_shapes",
        ),
        "stitch_train_segment_ids": [str(x) for x in (data.get("stitch_train_segment_ids") or [])],
        "stitch_use_roi": bool(data.get("stitch_use_roi", False)),
        "stitch_val_bboxes": dict(data.get("stitch_val_bboxes") or {}),
        "stitch_train_bboxes": dict(data.get("stitch_train_bboxes") or {}),
        "stitch_log_only_shapes": _coerce_shape_list(
            data.get("stitch_log_only_shapes"),
            key="stitch_cfg.stitch_log_only_shapes",
        ),
        "stitch_log_only_segment_ids": [str(x) for x in (data.get("stitch_log_only_segment_ids") or [])],
        "stitch_log_only_bboxes": dict(data.get("stitch_log_only_bboxes") or {}),
        "stitch_log_only_downsample": max(
            1,
            int(data.get("stitch_log_only_downsample", downsample) or downsample),
        ),
        "stitch_log_only_every_n_epochs": max(
            1,
            int(data.get("stitch_log_only_every_n_epochs", 10) or 10),
        ),
        "stitch_train": bool(data.get("stitch_train", False)),
        "stitch_train_every_n_epochs": max(
            1,
            int(data.get("stitch_train_every_n_epochs", 1) or 1),
        ),
    }


def _coerce_regression_model_state(
    *,
    model_state=None,
    model_cfg=None,
    objective_cfg=None,
    stitch_cfg=None,
):
    if model_state is not None:
        state = _cfg_to_dict(model_state, key="model_state")
        model_cfg = state.get("model_cfg", model_cfg)
        objective_cfg = state.get("objective_cfg", objective_cfg)
        stitch_cfg = state.get("stitch_cfg", stitch_cfg)

    return {
        "model_cfg": _coerce_model_cfg(model_cfg),
        "objective_cfg": _coerce_objective_cfg(objective_cfg),
        "stitch_cfg": _coerce_stitch_cfg(stitch_cfg),
    }


class RegressionPLModel(pl.LightningModule):
    def __init__(
        self,
        *,
        model_state: dict | None = None,
        model_cfg: dict | None = None,
        objective_cfg: dict | None = None,
        stitch_cfg: dict | None = None,
    ):
        super(RegressionPLModel, self).__init__()
        state = _coerce_regression_model_state(
            model_state=model_state,
            model_cfg=model_cfg,
            objective_cfg=objective_cfg,
            stitch_cfg=stitch_cfg,
        )
        model_cfg_obj = SimpleNamespace(**state["model_cfg"])
        objective_cfg_obj = SimpleNamespace(**state["objective_cfg"])
        stitch_cfg_obj = SimpleNamespace(**state["stitch_cfg"])
        save_regression_hyperparameters(
            self,
            model_cfg=model_cfg_obj,
            objective_cfg=objective_cfg_obj,
            stitch_cfg=stitch_cfg_obj,
        )
        initialize_regression_state(
            self,
            model_cfg=model_cfg_obj,
            objective_cfg=objective_cfg_obj,
            stitch_cfg=stitch_cfg_obj,
        )

    def set_stitch_borders(self, *, train_borders=None, val_borders=None):
        self._stitcher.set_borders(train_borders=train_borders, val_borders=val_borders)

    def set_train_stitch_loaders(self, loaders, segment_ids):
        self._stitcher.set_train_loaders(loaders, segment_ids)

    def set_log_only_stitch_loaders(self, loaders, segment_ids):
        self._stitcher.set_log_only_loaders(loaders, segment_ids)

    def _accumulate_stitch_predictions(self, *, outputs, xyxys, pred_buf, count_buf, offset=(0, 0)):
        self._stitcher.accumulate_to_buffers(
            outputs=outputs,
            xyxys=xyxys,
            pred_buf=pred_buf,
            count_buf=count_buf,
            offset=offset,
        )

    def on_train_epoch_start(self):
        reset_train_epoch_accumulators(self)

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        if self.with_norm:
            x = self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask

    def compute_per_sample_loss_and_dice(self, logits, targets):
        return compute_per_sample_loss_and_dice(
            logits,
            targets,
            loss_recipe=self.loss_recipe,
            smooth_factor=self.bce_smooth_factor,
            soft_label_positive=self.soft_label_positive,
            soft_label_negative=self.soft_label_negative,
        )

    def build_bce_targets(self, targets):
        return build_bce_targets(
            targets,
            smooth_factor=self.bce_smooth_factor,
            soft_label_positive=self.soft_label_positive,
            soft_label_negative=self.soft_label_negative,
        )

    def training_step(self, batch, batch_idx):
        x, y, group_idx = batch
        outputs = self(x)
        per_sample_loss, per_sample_dice, per_sample_bce, per_sample_dice_loss = self.compute_per_sample_loss_and_dice(
            outputs,
            y,
        )

        loss = compute_objective_loss(
            self,
            outputs=outputs,
            targets=y,
            per_sample_loss=per_sample_loss,
            per_sample_dice=per_sample_dice,
            per_sample_bce=per_sample_bce,
            per_sample_dice_loss=per_sample_dice_loss,
            group_idx=group_idx,
        )

        finalize_training_batch(
            self,
            loss=loss,
            per_sample_loss=per_sample_loss,
            per_sample_dice=per_sample_dice,
            group_idx=group_idx,
        )
        return {"loss": loss}

    def on_train_epoch_end(self):
        log_train_epoch_metrics(self)

    def on_validation_epoch_start(self):
        reset_validation_epoch_accumulators(self)
        initialize_validation_metrics(self)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, xyxys, group_idx = batch
        outputs = self(x)
        per_sample_loss, per_sample_dice, per_sample_bce, per_sample_dice_loss = self.compute_per_sample_loss_and_dice(
            outputs,
            y,
        )

        accumulate_validation_stats(
            self,
            per_sample_loss=per_sample_loss,
            per_sample_dice=per_sample_dice,
            per_sample_bce=per_sample_bce,
            per_sample_dice_loss=per_sample_dice_loss,
            group_idx=group_idx,
        )
        update_validation_stream_metrics(self, outputs=outputs, targets=y)

        self._stitcher.accumulate_val(outputs=outputs, xyxys=xyxys, dataloader_idx=dataloader_idx)
        return {"loss": per_sample_loss.mean()}

    def on_validation_epoch_end(self):
        sync_validation_accumulators(self)
        log_validation_epoch_metrics(self)
        self._stitcher.on_validation_epoch_end(self)

    def configure_optimizers(self):
        return configure_optimizers_runtime(self)


__all__ = ["RegressionPLModel"]
