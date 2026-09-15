from __future__ import annotations

from .base_aux_trainer import BaseAuxTrainer
from .distance_transform_trainer import DistanceTransformTrainer
from .inplane_direction_trainer import InplaneDirectionTrainer
from .nearest_component_trainer import NearestComponentTrainer
from .structure_tensor_trainer import StructureTensorTrainer
from .surface_normals_trainer import SurfaceNormalsTrainer


class AuxiliaryTrainer(BaseAuxTrainer):
    TASK_TRAINERS = {
        trainer.SUPPORTED_TASK_TYPE: trainer
        for trainer in (
            DistanceTransformTrainer,
            SurfaceNormalsTrainer,
            StructureTensorTrainer,
            InplaneDirectionTrainer,
            NearestComponentTrainer,
        )
    }

    def __init__(self, mgr=None, verbose: bool = True) -> None:
        super().__init__(mgr=mgr, verbose=verbose)
        unsupported = [
            name
            for name, cfg in self._aux_target_configs.items()
            if str(cfg.get("task_type", "")).lower() not in self.TASK_TRAINERS
        ]
        if unsupported:
            raise ValueError(
                f"{self.__class__.__name__} supports auxiliary task types {sorted(self.TASK_TRAINERS)}. "
                f"Unsupported tasks present: {unsupported}"
            )

    def _compute_aux_tensor(self, aux_name, target_cfg, sample, *, is_training):
        trainer = self.TASK_TRAINERS[str(target_cfg.get("task_type", "")).lower()]
        return trainer._compute_aux_tensor(self, aux_name, target_cfg, sample, is_training=is_training)
