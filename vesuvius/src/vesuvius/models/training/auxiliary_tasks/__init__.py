"""
Auxiliary tasks module for Vesuvius models.

This module provides factory functions and implementations for various auxiliary tasks
such as distance transforms and surface normals that can be used during training.
"""

from .apply_aux_targets import (
    create_auxiliary_task,
    compute_auxiliary_loss,
    preserve_auxiliary_targets,
    restore_auxiliary_targets,
    apply_auxiliary_tasks_from_config
)

from .aux_distance_transform import (
    create_distance_transform_config,
    compute_distance_transform
)

from .aux_surface_normals import (
    create_surface_normals_config,
    compute_surface_normals_from_sdt
)

from .aux_structure_tensor import (
    create_structure_tensor_config,
    compute_structure_tensor
)

from .aux_inplane_direction import (
    create_inplane_direction_config,
    compute_inplane_direction
)

from .aux_nearest_component import (
    create_nearest_component_config,
    compute_nearest_component
)
