"""Trainer for nearest-component auxiliary targets."""

from __future__ import annotations

import numpy as np
import torch

from ...auxiliary_tasks.aux_nearest_component import compute_nearest_component
from .base_aux_trainer import BaseAuxTrainer


class NearestComponentTrainer(BaseAuxTrainer):
    """Trainer that injects nearest-component direction and distance targets."""

    SUPPORTED_TASK_TYPE = "nearest_component"

    def _compute_aux_tensor(self, aux_name, target_cfg, sample, *, is_training):
        source_tensor = self._get_source_tensor(sample, target_cfg)
        if source_tensor is None:
            return None

        source_np = source_tensor.detach().cpu().numpy()
        if source_np.ndim < 3:
            return None

        binary_mask = (source_np[0] > 0).astype(np.uint8)
        is_2d = source_np.ndim == 3
        aux_np = compute_nearest_component(
            binary_mask,
            is_2d,
            sdf_sigma=float(target_cfg.get("sdf_sigma", 0.0)),
        )
        aux_np = np.ascontiguousarray(aux_np, dtype=np.float32)
        return torch.from_numpy(aux_np)
