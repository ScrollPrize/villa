"""Trainer for distance transform auxiliary targets."""

from __future__ import annotations

import numpy as np
import torch
import vc_reader.transforms as A

from .base_aux_trainer import BaseAuxTrainer


class DistanceTransformTrainer(BaseAuxTrainer):
    """Trainer that derives distance-transform targets alongside primary supervision."""

    SUPPORTED_TASK_TYPE = "distance_transform"

    def _compute_aux_tensor(self, aux_name, target_cfg, sample, *, is_training):
        source_tensor = self._get_source_tensor(sample, target_cfg)
        if source_tensor is None:
            return None

        source_np = source_tensor.detach().cpu().numpy()
        if source_np.ndim < 3:
            return None

        binary_mask = (source_np[0] > 0).astype(np.uint8)

        mode = str(target_cfg.get("distance_type", "signed")).lower()
        if mode not in ("inside", "outside"):
            mode = "signed"
        aux_np = A.DistanceTransform(mode=mode, implicit_background=True)(binary_mask[np.newaxis])
        return torch.from_numpy(np.ascontiguousarray(aux_np, dtype=np.float32))
