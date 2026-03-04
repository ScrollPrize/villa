from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from train_resnet3d_lib.config import CFG


@dataclass(frozen=True)
class ModelConfig:
    size: int
    enc: str
    with_norm: bool
    total_steps: int
    n_groups: int
    group_names: list[str]
    stitch_group_idx_by_segment: dict[str, int]
    norm: str
    group_norm_groups: int


@dataclass(frozen=True)
class ObjectiveConfig:
    objective: str
    loss_mode: str
    robust_step_size: float | None
    group_counts: list[int]
    group_dro_gamma: float
    group_dro_btl: bool
    group_dro_alpha: float | None
    group_dro_normalize_loss: bool
    group_dro_min_var_weight: float
    group_dro_adj: Any
    erm_group_topk: int


@dataclass(frozen=True)
class StitchConfig:
    stitch_val_dataloader_idx: int | None
    stitch_pred_shape: tuple[int, int] | None
    stitch_segment_id: str | None
    stitch_all_val: bool
    stitch_downsample: int
    stitch_all_val_shapes: list[tuple[int, int]]
    stitch_all_val_segment_ids: list[str]
    stitch_train_shapes: list[tuple[int, int]]
    stitch_train_segment_ids: list[str]
    stitch_use_roi: bool
    stitch_val_bboxes: dict[str, Any]
    stitch_train_bboxes: dict[str, Any]
    stitch_log_only_shapes: list[tuple[int, int]]
    stitch_log_only_segment_ids: list[str]
    stitch_log_only_bboxes: dict[str, Any]
    stitch_log_only_downsample: int
    stitch_log_only_every_n_epochs: int
    stitch_train: bool
    stitch_train_every_n_epochs: int


def _as_dict(value, *, key):
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"{key} must be a dict or dataclass instance, got {type(value).__name__}")


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


def coerce_model_config(model_cfg):
    if isinstance(model_cfg, ModelConfig):
        return model_cfg
    data = _as_dict(model_cfg, key="model_cfg")
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
    return ModelConfig(
        size=int(data.get("size", 256)),
        enc=str(data.get("enc", "i3d")),
        with_norm=bool(data.get("with_norm", False)),
        total_steps=int(data.get("total_steps", 1)),
        n_groups=n_groups,
        group_names=group_names,
        stitch_group_idx_by_segment=stitch_group_idx_by_segment,
        norm=str(data.get("norm", "batch")),
        group_norm_groups=int(data.get("group_norm_groups", 32)),
    )


def coerce_objective_config(objective_cfg):
    if isinstance(objective_cfg, ObjectiveConfig):
        return objective_cfg
    data = _as_dict(objective_cfg, key="objective_cfg")
    return ObjectiveConfig(
        objective=str(data.get("objective", "erm")),
        loss_mode=str(data.get("loss_mode", "batch")),
        robust_step_size=data.get("robust_step_size"),
        group_counts=[int(x) for x in list(data.get("group_counts") or [])],
        group_dro_gamma=float(data.get("group_dro_gamma", 0.1)),
        group_dro_btl=bool(data.get("group_dro_btl", False)),
        group_dro_alpha=data.get("group_dro_alpha"),
        group_dro_normalize_loss=bool(data.get("group_dro_normalize_loss", False)),
        group_dro_min_var_weight=float(data.get("group_dro_min_var_weight", 0.0)),
        group_dro_adj=data.get("group_dro_adj"),
        erm_group_topk=int(data.get("erm_group_topk", 0)),
    )


def coerce_stitch_config(stitch_cfg):
    if isinstance(stitch_cfg, StitchConfig):
        return stitch_cfg
    data = _as_dict(stitch_cfg, key="stitch_cfg")
    downsample = max(1, int(data.get("stitch_downsample", 1) or 1))
    return StitchConfig(
        stitch_val_dataloader_idx=(
            None
            if data.get("stitch_val_dataloader_idx") is None
            else int(data["stitch_val_dataloader_idx"])
        ),
        stitch_pred_shape=_coerce_shape_tuple(
            data.get("stitch_pred_shape"),
            key="stitch_cfg.stitch_pred_shape",
        ),
        stitch_segment_id=(
            None
            if data.get("stitch_segment_id") is None
            else str(data["stitch_segment_id"])
        ),
        stitch_all_val=bool(data.get("stitch_all_val", False)),
        stitch_downsample=downsample,
        stitch_all_val_shapes=_coerce_shape_list(
            data.get("stitch_all_val_shapes"),
            key="stitch_cfg.stitch_all_val_shapes",
        ),
        stitch_all_val_segment_ids=[str(x) for x in (data.get("stitch_all_val_segment_ids") or [])],
        stitch_train_shapes=_coerce_shape_list(
            data.get("stitch_train_shapes"),
            key="stitch_cfg.stitch_train_shapes",
        ),
        stitch_train_segment_ids=[str(x) for x in (data.get("stitch_train_segment_ids") or [])],
        stitch_use_roi=bool(data.get("stitch_use_roi", False)),
        stitch_val_bboxes=dict(data.get("stitch_val_bboxes") or {}),
        stitch_train_bboxes=dict(data.get("stitch_train_bboxes") or {}),
        stitch_log_only_shapes=_coerce_shape_list(
            data.get("stitch_log_only_shapes"),
            key="stitch_cfg.stitch_log_only_shapes",
        ),
        stitch_log_only_segment_ids=[str(x) for x in (data.get("stitch_log_only_segment_ids") or [])],
        stitch_log_only_bboxes=dict(data.get("stitch_log_only_bboxes") or {}),
        stitch_log_only_downsample=max(
            1,
            int(data.get("stitch_log_only_downsample", downsample) or downsample),
        ),
        stitch_log_only_every_n_epochs=max(
            1,
            int(data.get("stitch_log_only_every_n_epochs", 10) or 10),
        ),
        stitch_train=bool(data.get("stitch_train", False)),
        stitch_train_every_n_epochs=max(
            1,
            int(data.get("stitch_train_every_n_epochs", 1) or 1),
        ),
    )


def build_regression_model_configs(run_state: dict[str, Any], data_state: dict[str, Any]):
    model_cfg = ModelConfig(
        enc="i3d",
        size=int(getattr(CFG, "size", 256)),
        with_norm=False,
        total_steps=int(data_state["steps_per_epoch"]),
        n_groups=int(len(data_state["group_names"])),
        group_names=list(data_state["group_names"]),
        stitch_group_idx_by_segment=dict(data_state["group_idx_by_segment"]),
        norm=str(getattr(CFG, "norm", "batch")),
        group_norm_groups=int(getattr(CFG, "group_norm_groups", 32)),
    )

    objective_cfg = ObjectiveConfig(
        objective=str(getattr(CFG, "objective", "erm")),
        loss_mode=str(getattr(CFG, "loss_mode", "batch")),
        robust_step_size=run_state["robust_step_size"],
        group_counts=list(data_state["train_group_counts"]),
        group_dro_gamma=float(run_state["group_dro_gamma"]),
        group_dro_btl=bool(run_state["group_dro_btl"]),
        group_dro_alpha=run_state["group_dro_alpha"],
        group_dro_normalize_loss=bool(run_state["group_dro_normalize_loss"]),
        group_dro_min_var_weight=float(run_state["group_dro_min_var_weight"]),
        group_dro_adj=run_state["group_dro_adj"],
        erm_group_topk=int(getattr(CFG, "erm_group_topk", 0)),
    )

    stitch_cfg = StitchConfig(
        stitch_val_dataloader_idx=data_state["stitch_val_dataloader_idx"],
        stitch_pred_shape=data_state["stitch_pred_shape"],
        stitch_segment_id=(
            None if data_state["stitch_segment_id"] is None else str(data_state["stitch_segment_id"])
        ),
        stitch_all_val=bool(getattr(CFG, "stitch_all_val", False)),
        stitch_downsample=int(getattr(CFG, "stitch_downsample", 1)),
        stitch_all_val_shapes=list(data_state["val_stitch_shapes"]),
        stitch_all_val_segment_ids=[str(x) for x in data_state["val_stitch_segment_ids"]],
        stitch_train_shapes=list(data_state["train_stitch_shapes"]),
        stitch_train_segment_ids=[str(x) for x in data_state["train_stitch_segment_ids"]],
        stitch_use_roi=bool(getattr(CFG, "stitch_use_roi", False)),
        stitch_val_bboxes=dict(data_state.get("val_mask_bboxes") or {}),
        stitch_train_bboxes=dict(data_state.get("train_mask_bboxes") or {}),
        stitch_log_only_shapes=list(data_state.get("log_only_stitch_shapes") or []),
        stitch_log_only_segment_ids=[str(x) for x in (data_state.get("log_only_stitch_segment_ids") or [])],
        stitch_log_only_bboxes=dict(data_state.get("log_only_mask_bboxes") or {}),
        stitch_log_only_downsample=int(
            getattr(CFG, "stitch_log_only_downsample", getattr(CFG, "stitch_downsample", 1))
        ),
        stitch_log_only_every_n_epochs=int(getattr(CFG, "stitch_log_only_every_n_epochs", 10)),
        stitch_train=bool(getattr(CFG, "stitch_train", False)),
        stitch_train_every_n_epochs=int(getattr(CFG, "stitch_train_every_n_epochs", 1)),
    )

    return model_cfg, objective_cfg, stitch_cfg
