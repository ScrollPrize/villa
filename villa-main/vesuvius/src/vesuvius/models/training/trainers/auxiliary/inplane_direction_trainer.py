"""Trainer for in-plane direction auxiliary targets."""

from __future__ import annotations

import numpy as np
import torch

from ...auxiliary_tasks.aux_inplane_direction import compute_inplane_direction
from .base_aux_trainer import BaseAuxTrainer


class InplaneDirectionTrainer(BaseAuxTrainer):
    """Trainer that derives in-plane direction fields from segmentation masks."""

    SUPPORTED_TASK_TYPE = "inplane_direction"

    def _compute_aux_tensor(self, aux_name, target_cfg, sample, *, is_training):
        source_tensor = self._get_source_tensor(sample, target_cfg)
        if source_tensor is None:
            return None

        source_np = source_tensor.detach().cpu().numpy()
        if source_np.ndim < 3:
            return None

        binary_mask = (source_np[0] > 0).astype(np.uint8)
        is_2d = source_np.ndim == 3
        aux_np = compute_inplane_direction(
            binary_mask,
            is_2d,
            compute_from=str(target_cfg.get("compute_from", "sdt")).lower(),
            grad_sigma=float(target_cfg.get("grad_sigma", 1.0)),
            tensor_sigma=float(target_cfg.get("tensor_sigma", 1.5)),
            ignore_index=target_cfg.get("ignore_index", -100),
        )
        aux_np = np.ascontiguousarray(aux_np, dtype=np.float32)
        return torch.from_numpy(aux_np)
