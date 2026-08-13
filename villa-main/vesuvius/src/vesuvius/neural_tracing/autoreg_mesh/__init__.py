"""Autoregressive tifxyz mesh completion MVP."""

from .config import (
    load_autoreg_mesh_config,
    setdefault_autoreg_mesh_config,
    validate_autoreg_mesh_config,
)
from .dataset import AutoregMeshDataset, autoreg_mesh_collate
from .infer import infer_autoreg_mesh
from .losses import compute_autoreg_mesh_losses
from .model import AutoregMeshModel
from .serialization import (
    DIRECTION_TO_ID,
    ID_TO_DIRECTION,
    deserialize_continuation_grid,
    deserialize_full_grid,
    serialize_split_conditioning_example,
)

__all__ = [
    "AutoregMeshDataset",
    "AutoregMeshModel",
    "DIRECTION_TO_ID",
    "ID_TO_DIRECTION",
    "autoreg_mesh_collate",
    "compute_autoreg_mesh_losses",
    "deserialize_continuation_grid",
    "deserialize_full_grid",
    "infer_autoreg_mesh",
    "load_autoreg_mesh_config",
    "serialize_split_conditioning_example",
    "setdefault_autoreg_mesh_config",
    "validate_autoreg_mesh_config",
]
