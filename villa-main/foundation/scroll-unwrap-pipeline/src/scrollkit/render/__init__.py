"""Headless EGL rendering: scene construction, texture-orientation handling, parity QA.

Importing this package pins the headless EGL environment (scene module side effect)
BEFORE pyvista/vtk initialize — import scrollkit.render before anything GL-touching.
"""

from .scene import (
    ORIENTATIONS,
    SceneRenderer,
    auto_camera,
    load_texture,
    orient_texture_array,
    render_mesh_arrays,
    split_wedge_to_vertex,
)
from .parity import render_parity_ssim

__all__ = [
    "ORIENTATIONS",
    "SceneRenderer",
    "auto_camera",
    "load_texture",
    "orient_texture_array",
    "render_mesh_arrays",
    "render_parity_ssim",
    "split_wedge_to_vertex",
]
