"""Auxiliary trainers for task-specific supervision."""

from .base_aux_trainer import BaseAuxTrainer
from .distance_transform_trainer import DistanceTransformTrainer
from .surface_normals_trainer import SurfaceNormalsTrainer
from .structure_tensor_trainer import StructureTensorTrainer
from .inplane_direction_trainer import InplaneDirectionTrainer
from .nearest_component_trainer import NearestComponentTrainer

__all__ = [
    "BaseAuxTrainer",
    "DistanceTransformTrainer",
    "SurfaceNormalsTrainer",
    "StructureTensorTrainer",
    "InplaneDirectionTrainer",
    "NearestComponentTrainer",
]
