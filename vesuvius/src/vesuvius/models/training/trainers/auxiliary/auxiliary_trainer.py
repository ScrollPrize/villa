"""Unified trainer for configurations containing multiple auxiliary tasks."""

from __future__ import annotations

from .base_aux_trainer import BaseAuxTrainer
from .distance_transform_trainer import DistanceTransformTrainer
from .inplane_direction_trainer import InplaneDirectionTrainer
from .nearest_component_trainer import NearestComponentTrainer
from .structure_tensor_trainer import StructureTensorTrainer
from .surface_normals_trainer import SurfaceNormalsTrainer


_TASK_TRAINERS = {
    "distance_transform": DistanceTransformTrainer,
    "inplane_direction": InplaneDirectionTrainer,
    "nearest_component": NearestComponentTrainer,
    "structure_tensor": StructureTensorTrainer,
    "surface_normals": SurfaceNormalsTrainer,
}


class AuxiliaryTrainer(BaseAuxTrainer):
    """Inject every supported auxiliary target declared by the configuration."""

    def __init__(self, mgr=None, verbose: bool = True) -> None:
        super().__init__(mgr=mgr, verbose=verbose)
        unsupported = {
            str(cfg.get("task_type", "")).lower() or "<missing>"
            for cfg in self._aux_target_configs.values()
        } - set(_TASK_TRAINERS)
        if unsupported:
            raise ValueError(
                "Unsupported auxiliary task types: " + ", ".join(sorted(unsupported))
            )

    def _compute_aux_tensor(self, aux_name, target_cfg, sample, *, is_training):
        task_type = str(target_cfg.get("task_type", "")).lower()
        trainer_type = _TASK_TRAINERS.get(task_type)
        if trainer_type is None:
            raise ValueError(f"Unsupported auxiliary task type: {task_type or '<missing>'}")
        # The task-specific methods are stateless generators and depend only on
        # BaseAuxTrainer's source-target helper. Reuse them so their established
        # numerical behavior remains the single implementation.
        return trainer_type._compute_aux_tensor(
            self,
            aux_name,
            target_cfg,
            sample,
            is_training=is_training,
        )
