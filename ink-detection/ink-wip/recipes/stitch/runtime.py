from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any

from ink.core.types import DataBundle
from ink.recipes.stitch.data import StitchData, stitch_data_to_config
from ink.recipes.stitch.eval_runtime import (
    accumulate_val as _accumulate_val_impl,
    finalize_validation_epoch as _finalize_validation_epoch_impl,
    sync_eval_buffers_distributed as _sync_eval_buffers_distributed_impl,
)
from ink.recipes.stitch.state import (
    StitchExecutionContext,
    _StitchState,
    _UNSET,
    _deep_merge_dicts,
    _log,
    _noop_log,
)
from ink.recipes.stitch.terms import compute_stitched_component_loss
from ink.recipes.stitch.train_component_runtime import (
    stitch_saved_tensors_context as _stitch_saved_tensors_context_impl,
)
from ink.recipes.stitch.train_runtime import (
    TrainStitchRuntime,
    _accumulate_to_buffers as _accumulate_to_buffers_impl,
    _normalize_xyxys,
    compute_train_stitch_loss,
    run_train_stitch_pass,
)


@dataclass
class EvalStitchRuntime:
    data: StitchData
    state: _StitchState
    execution: StitchExecutionContext
    train: TrainStitchRuntime | None = None
    log: object = _noop_log

    def accumulate_val(self, *, outputs, xyxys, dataloader_idx):
        return accumulate_val(self, outputs=outputs, xyxys=xyxys, dataloader_idx=dataloader_idx)

    def finalize_epoch(self, model):
        return finalize_validation_epoch(self, model)


@dataclass
class StitchRuntime:
    data: StitchData
    state: _StitchState
    execution: StitchExecutionContext
    train: TrainStitchRuntime
    eval: EvalStitchRuntime
    log: object = _noop_log

    @property
    def enabled(self) -> bool:
        return self.state.enabled

    def set_execution_context(
        self,
        *,
        precision_context_factory=_UNSET,
        sanity_checking=_UNSET,
        is_global_zero=_UNSET,
        distributed_world_size=_UNSET,
        distributed_reduce_sum=_UNSET,
    ) -> None:
        if precision_context_factory is not _UNSET:
            self.execution.precision_context_factory = precision_context_factory
        if sanity_checking is not _UNSET:
            self.execution.sanity_checking = bool(sanity_checking)
        if is_global_zero is not _UNSET:
            self.execution.is_global_zero = bool(is_global_zero)
        if distributed_world_size is not _UNSET:
            world_size = int(distributed_world_size)
            if world_size < 1:
                raise ValueError(f"distributed_world_size must be >= 1, got {world_size!r}")
            self.execution.distributed_world_size = world_size
        if distributed_reduce_sum is not _UNSET:
            self.execution.distributed_reduce_sum = distributed_reduce_sum

    def set_borders(self, *, train_borders=None, eval_borders=None, val_borders=None) -> None:
        resolved_eval_borders = eval_borders if eval_borders is not None else val_borders
        if train_borders is not None:
            self.data.layout.borders_by_split["train"] = dict(train_borders)
        if resolved_eval_borders is not None:
            self.data.layout.borders_by_split["eval"] = dict(resolved_eval_borders)

    def set_train_loaders(self, loaders, segment_ids=None) -> None:
        self.train.set_loaders(loaders, segment_ids=segment_ids)

    def set_train_component_datasets(self, datasets, component_keys=None) -> None:
        self.train.set_component_datasets(datasets, component_keys=component_keys)

    def train_loader_for_segment(self, segment_id):
        return self.train.loader_for_segment(segment_id)

    def train_dataset_for_component(self, component_key):
        return self.train.dataset_for_component(component_key)

    @classmethod
    def from_config(cls, stitch_data: StitchData | dict | None = None, *, logger=None, patch_loss=None) -> StitchRuntime:
        data = StitchData.from_config(stitch_data or {})
        state = _StitchState(data)
        execution = StitchExecutionContext()
        log = logger or _noop_log
        train = TrainStitchRuntime(
            data=data,
            state=state,
            execution=execution,
            patch_loss=patch_loss,
            patch_loss_weight=float(data.train.loss.patch_loss_weight),
            gradient_checkpointing=bool(data.train.loss.gradient_checkpointing),
            save_on_cpu=bool(data.train.loss.save_on_cpu),
            log=log,
        )
        eval_runtime = EvalStitchRuntime(data=data, state=state, execution=execution, train=train, log=log)
        return cls(
            data=data,
            state=state,
            execution=execution,
            train=train,
            eval=eval_runtime,
            log=log,
        )

    @classmethod
    def from_bundle(cls, bundle: DataBundle, *, logger=None, patch_loss=None) -> StitchRuntime:
        return cls.from_config(StitchData.from_bundle(bundle), logger=logger, patch_loss=patch_loss)


@dataclass(frozen=True)
class StitchRuntimeRecipe:
    config: StitchData | None = None

    def build(self, bundle: DataBundle, *, logger=None, patch_loss=None) -> StitchRuntime:
        bundle_cfg = stitch_data_to_config(StitchData.from_bundle(bundle))
        if self.config is None:
            merged_cfg = bundle_cfg
        else:
            merged_cfg = _deep_merge_dicts(bundle_cfg, stitch_data_to_config(self.config))
        return StitchRuntime.from_config(merged_cfg, logger=logger, patch_loss=patch_loss)


def _accumulate_to_buffers(
    owner,
    *,
    outputs,
    xyxys,
    pred_buf,
    count_buf,
    offset=(0, 0),
):
    return _accumulate_to_buffers_impl(
        owner,
        outputs=outputs,
        xyxys=xyxys,
        pred_buf=pred_buf,
        count_buf=count_buf,
        offset=offset,
    )


def _stitch_saved_tensors_context(owner):
    return _stitch_saved_tensors_context_impl(owner, log=_log)


def accumulate_val(owner, *, outputs, xyxys, dataloader_idx):
    return _accumulate_val_impl(
        owner,
        outputs=outputs,
        xyxys=xyxys,
        dataloader_idx=dataloader_idx,
        normalize_xyxys=_normalize_xyxys,
        accumulate_to_buffers=_accumulate_to_buffers,
    )


def _sync_eval_buffers_distributed(owner, state, model) -> bool:
    return _sync_eval_buffers_distributed_impl(
        owner,
        state=state,
        model=model,
    )


def finalize_validation_epoch(owner, model):
    return _finalize_validation_epoch_impl(owner, model)
