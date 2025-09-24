"""Mesh voxelization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Union

import numpy as np

from .handles import MeshHandle
from .types import MeshPayload


@dataclass(frozen=True)
class MeshVoxelizationResult:
    """Binary voxel grid plus spatial metadata."""

    voxels: np.ndarray  # bool array indexed as (z, y, x)
    origin: np.ndarray  # world-space origin of voxel (0, 0, 0)
    voxel_size: float   # isotropic voxel edge length


def _payload_to_trimesh(payload: MeshPayload):
    try:
        import trimesh  # type: ignore
    except ImportError as exc:  # pragma: no cover - explicit dependency check
        raise RuntimeError(
            "mesh_to_binary_voxels requires 'trimesh'. Install it or provide a custom voxelizer."
        ) from exc

    return trimesh.Trimesh(
        vertices=np.asarray(payload.vertices, dtype=np.float32),
        faces=np.asarray(payload.faces, dtype=np.int64),
        process=False,
    )


def mesh_to_binary_voxels(
    mesh: Union[MeshHandle, MeshPayload],
    voxel_size: float,
    *,
    fill_solid: bool = True,
) -> MeshVoxelizationResult:
    """Convert a mesh into a binary voxel occupancy grid.

    Parameters
    ----------
    mesh:
        Either a :class:`MeshHandle` or a preloaded :class:`MeshPayload`.
    voxel_size:
        Edge length of each cubic voxel in the same units as the mesh coordinates.
    fill_solid:
        When ``True`` (default), the interior of watertight meshes is flood-filled so the
        output represents a solid volume. Set to ``False`` to keep only the shell voxels.

    Returns
    -------
    MeshVoxelizationResult
        Boolean array with axes ordered as ``(z, y, x)`` plus origin and voxel spacing.
    """

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    payload = mesh.read() if isinstance(mesh, MeshHandle) else mesh

    tri_mesh = _payload_to_trimesh(payload)
    grid = tri_mesh.voxelized(pitch=float(voxel_size))
    if fill_solid:
        grid = grid.fill()

    matrix = grid.matrix.astype(bool, copy=False)
    # trimesh uses (x, y, z); datasets typically expect (z, y, x)
    matrix = np.transpose(matrix, (2, 1, 0))
    transform = getattr(grid, "transform", None)
    if transform is None:
        translation = getattr(grid, "translation", None)
        if translation is None:
            origin_xyz = np.zeros(3, dtype=np.float32)
        else:
            origin_xyz = np.asarray(translation, dtype=np.float32)
    else:
        origin_xyz = np.asarray(transform[:3, 3], dtype=np.float32)

    origin = origin_xyz[[2, 1, 0]]  # reorder to (z, y, x)

    return MeshVoxelizationResult(voxels=matrix, origin=origin, voxel_size=float(voxel_size))
