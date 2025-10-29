"""
Utilities for working with quad meshes in Python.
"""

from __future__ import annotations

from os import PathLike
from typing import Optional, Union

import numpy as np

from . import vc_tracing as _tracing

QuadMesh = _tracing.QuadSurface

__all__ = [
    "QuadMesh",
    "load_quadmesh",
    "save_quadmesh",
    "quadmesh_to_numpy",
    "quadmesh_to_open3d",
]

PathInput = Union[str, bytes, PathLike[str], PathLike[bytes]]


def load_quadmesh(path: PathInput, *, ignore_mask: bool = False) -> QuadMesh:
    """
    Load a quad mesh (tifxyz) from disk.
    """
    return _tracing.load_quadmesh(path, ignore_mask=ignore_mask)


def save_quadmesh(
    mesh: QuadMesh,
    path: PathInput,
    *,
    uuid: Optional[str] = None,
    force_overwrite: bool = False,
) -> None:
    """
    Persist a quad mesh to disk using the same layout as the CLI tools.
    """
    mesh.save(path, uuid=uuid, force_overwrite=force_overwrite)


def quadmesh_to_numpy(mesh: QuadMesh) -> np.ndarray:
    """
    Return the XYZ positions as a ``(rows, cols, 3)`` NumPy array of ``float32``.
    """
    return mesh.points()


def quadmesh_to_open3d(
    mesh: QuadMesh,
    *,
    triangulate: bool = True,
    compute_vertex_normals: bool = True,
):
    """
    Convert the quad mesh into an Open3D geometry.
    """
    try:
        import open3d as o3d  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("open3d is required for quadmesh_to_open3d") from exc

    points = quadmesh_to_numpy(mesh)
    if points.size == 0:
        raise ValueError("quad mesh has no points")

    rows, cols, _ = points.shape
    vertices = points.reshape((-1, 3)).astype(np.float64, copy=False)

    if rows < 2 or cols < 2 or not triangulate:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(vertices)
        if compute_vertex_normals:
            cloud.estimate_normals()
        return cloud

    grid_rows = rows - 1
    grid_cols = cols - 1
    base = (
        np.arange(grid_rows, dtype=np.int64)[:, None] * cols
        + np.arange(grid_cols, dtype=np.int64)[None, :]
    )

    tri1 = np.stack((base, base + 1, base + cols), axis=-1)
    tri2 = np.stack((base + 1, base + cols + 1, base + cols), axis=-1)
    triangles = np.concatenate((tri1.reshape(-1, 3), tri2.reshape(-1, 3)), axis=0)

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32, copy=False))
    if compute_vertex_normals:
        mesh_o3d.compute_vertex_normals()
    return mesh_o3d
