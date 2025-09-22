"""Base trainer substrate for auxiliary-target workflows."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import torch
from ...train import BaseTrainer


class BaseAuxTrainer(BaseTrainer):
    """Base trainer that injects auxiliary targets derived from source labels."""

    SUPPORTED_TASK_TYPE: str | None = None

    def __init__(self, mgr=None, verbose: bool = True) -> None:
        super().__init__(mgr=mgr, verbose=verbose)
        all_aux = {
            name: cfg
            for name, cfg in getattr(self.mgr, "targets", {}).items()
            if cfg.get("auxiliary_task", False)
        }
        if self.SUPPORTED_TASK_TYPE is None:
            self._aux_target_configs = all_aux
        else:
            self._aux_target_configs = {
                name: cfg
                for name, cfg in all_aux.items()
                if str(cfg.get("task_type", "")).lower() == self.SUPPORTED_TASK_TYPE
            }
            unsupported = [
                name
                for name, cfg in all_aux.items()
                if str(cfg.get("task_type", "")).lower() != self.SUPPORTED_TASK_TYPE
            ]
            if unsupported:
                allowed = self.SUPPORTED_TASK_TYPE
                raise ValueError(
                    f"{self.__class__.__name__} supports auxiliary task type '{allowed}'. "
                    f"Unsupported tasks present: {unsupported}"
                )

    # ------------------------------------------------------------------ hooks

    def _prepare_sample(self, sample: dict, *, is_training: bool) -> dict:
        if not self._aux_target_configs:
            return sample
        return self._inject_aux_targets(sample, is_training=is_training)

    def _prepare_batch(self, batch: dict, *, is_training: bool) -> dict:
        if not self._aux_target_configs:
            return batch
        image = batch.get("image")
        if not isinstance(image, torch.Tensor) or image.ndim == 0:
            return batch

        batch_size = image.shape[0]
        per_sample: List[dict] = []
        for idx in range(batch_size):
            sample = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                    sample[key] = value[idx]
                elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                    sample[key] = value[idx]
                else:
                    sample[key] = value
            per_sample.append(self._inject_aux_targets(sample, is_training=is_training))

        merged: Dict[str, List] = {}
        for sample in per_sample:
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    merged.setdefault(key, []).append(value)
                elif isinstance(value, (list, tuple)):
                    merged.setdefault(key, []).append(value)
                else:
                    merged.setdefault(key, [value])

        rebuilt: Dict[str, object] = {}
        for key, values in merged.items():
            if isinstance(values[0], torch.Tensor) and len(values) == batch_size:
                rebuilt[key] = torch.stack(values, dim=0)
            elif len(values) == batch_size:
                rebuilt[key] = list(values)
            else:
                rebuilt[key] = values[0]
        return rebuilt

    # ------------------------------------------------------------------ helpers

    def _inject_aux_targets(self, sample: dict, *, is_training: bool) -> dict:
        augmented = deepcopy(sample)
        regression_keys: List[str] = list(augmented.get("regression_keys", []) or [])

        for aux_name, cfg in self._aux_target_configs.items():
            aux_tensor = self._compute_aux_tensor(
                aux_name,
                cfg,
                augmented,
                is_training=is_training,
            )
            if aux_tensor is None:
                continue

            augmented[aux_name] = aux_tensor
            if aux_name not in regression_keys:
                regression_keys.append(aux_name)

        if regression_keys:
            augmented["regression_keys"] = regression_keys
        return augmented

    # ------------------------------------------------------------------ hooks for subclasses

    def _compute_aux_tensor(
        self,
        aux_name: str,
        target_cfg: Dict,
        sample: dict,
        *,
        is_training: bool,
    ) -> torch.Tensor | None:
        raise NotImplementedError

    def _get_source_tensor(self, sample: dict, target_cfg: Dict) -> torch.Tensor | None:
        source_name = target_cfg.get("source_target")
        tensor = sample.get(source_name)
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
            return tensor
        return None

    def _compute_loss_value(
        self,
        loss_fn,
        prediction,
        ground_truth,
        *,
        target_name: str,
        targets_dict: dict,
        outputs: dict,
    ):
        target_cfg = self.mgr.targets.get(target_name, {})
        source_name = target_cfg.get('source_target')
        if source_name and source_name in outputs:
            try:
                return loss_fn(prediction, ground_truth, source_pred=outputs[source_name])
            except TypeError:
                pass
        return super()._compute_loss_value(
            loss_fn,
            prediction,
            ground_truth,
            target_name=target_name,
            targets_dict=targets_dict,
            outputs=outputs,
        )
