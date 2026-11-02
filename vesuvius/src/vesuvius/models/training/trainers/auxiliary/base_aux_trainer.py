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
        if batch_size == 0:
            return batch
        generated: Dict[str, List[torch.Tensor | None]] = {
            name: [] for name in self._aux_target_configs
        }
        for idx in range(batch_size):
            sample = {}
            for key, value in batch.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == batch_size
                ):
                    sample[key] = value[idx]
                elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                    sample[key] = value[idx]
                else:
                    sample[key] = value
            for aux_name, cfg in self._aux_target_configs.items():
                generated[aux_name].append(
                    self._compute_aux_tensor(
                        aux_name,
                        cfg,
                        sample,
                        is_training=is_training,
                    )
                )

        augmented = dict(batch)
        generated_names: List[str] = []
        for aux_name, tensors in generated.items():
            available = [tensor is not None for tensor in tensors]
            if not all(available):
                missing = [idx for idx, present in enumerate(available) if not present]
                raise ValueError(
                    f"Auxiliary target '{aux_name}' could not be generated for "
                    f"batch samples {missing}; every sample must provide its "
                    "source target."
                )
            augmented[aux_name] = torch.stack(tensors, dim=0)
            generated_names.append(aux_name)

        regression_keys = _flatten_metadata_names(batch.get("regression_keys"))
        for aux_name in generated_names:
            if aux_name not in regression_keys:
                regression_keys.append(aux_name)
        if regression_keys:
            augmented["regression_keys"] = regression_keys
        return augmented

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


def _flatten_metadata_names(value) -> List[str]:
    """Normalize collated metadata into one ordered list of string keys."""
    names: List[str] = []
    if isinstance(value, str):
        names.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            for name in _flatten_metadata_names(item):
                if name not in names:
                    names.append(name)
    return names
