"""Trainer for surface-normal auxiliary targets."""

from __future__ import annotations

import numpy as np
import torch

from ...auxiliary_tasks.aux_surface_normals import compute_surface_normals_from_sdt
from .base_aux_trainer import BaseAuxTrainer


class SurfaceNormalsTrainer(BaseAuxTrainer):
    """Trainer that derives surface-normal targets from primary masks."""

    SUPPORTED_TASK_TYPE = "surface_normals"

    def _compute_aux_tensor(self, aux_name, target_cfg, sample, *, is_training):
        source_tensor = self._get_source_tensor(sample, target_cfg)
        if source_tensor is None:
            return None

        source_np = source_tensor.detach().cpu().numpy()
        if source_np.ndim < 3:
            return None

        is_2d = source_np.ndim == 3
        binary_mask = (source_np[0] > 0).astype(np.uint8)
        normals, _ = compute_surface_normals_from_sdt(binary_mask, is_2d=is_2d, return_sdt=False)
        aux_np = np.ascontiguousarray(normals, dtype=np.float32)
        return torch.from_numpy(aux_np)
