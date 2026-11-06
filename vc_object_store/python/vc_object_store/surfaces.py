"""Surfaces as numpy grids: read one in a volume's voxels, create one from a grid.

A stored surface's geometry is a ``points`` array (rows x columns x 3, xyz, -1 for a missing vertex; the store's validity rule decides which
vertices are missing), an optional
``valid`` array and per-vertex channels, with a grid ``scale``; ``meta.coordinates`` binds it to a volume.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._native import Surface
from .locations import attached

INVALID = -1.0


class SurfaceGrid(NamedTuple):
    points: np.ndarray  # (rows, columns, 3) float32 xyz, -1 at missing vertices
    valid: np.ndarray  # (rows, columns) bool
    scale: tuple  # grid scale in the read volume's voxels
    record: dict
    version: int  # the numbered object version read, in ``generation``
    generation: str
    geometry_revision: str  # payload identity of its geometry (reads its channels)
    channels: frozenset  # names of the per-vertex channels (not points or valid)


def read_surface(store, surface_uuid, *, volume, version=None, generation=None, target_scale=1.0) -> SurfaceGrid:
    """The surface (its current version, or numbered ``version`` in ``generation``, default the store's active
    one) in voxels of ``volume`` at ``target_scale``, through the store's mapping to it (its binding, or a
    registration). ``store`` is a store file or a vc_object_server endpoint (``locations``). Raises ValueError when
    the surface (version) does not exist or does not map to ``volume``."""
    with attached(store, read_only=True) as (session, attachment):
        grid = session.read_surface_grid(attachment, str(surface_uuid), str(volume), float(target_scale),
                                         None if version is None else int(version), generation)
    return surface_grid(grid)


def surface_grid(grid: dict) -> SurfaceGrid:
    """A ``SurfaceGrid`` from ``ObjectStoreSession.read_surface_grid``'s answer."""
    return SurfaceGrid(np.asarray(grid["points"], dtype=np.float32), np.asarray(grid["valid"]).astype(bool),
                       tuple(float(s) for s in grid["scale"]), grid["record"], grid["version"], grid["generation"],
                       grid["geometry_revision"], frozenset(grid["channels"]))


def create_surface(store, xyz, *, name, scale, volume, tool, parameters=None, sources=(), meta=None,
                   channels=None, tags=(), author=None, voxel_scale=1.0):
    """A new surface holding ``xyz`` ((rows, columns, 3) in voxels of ``volume`` at ``voxel_scale`` of its own;
    vertices are missing by the store's validity rule) with grid ``scale`` (one number or a pair), bound to those
    voxels, with per-vertex ``channels`` and
    meta.provenance {tool, parameters, sources}. ``meta`` is kept except tifxyz file keys (the store's
    createSurface). ``store`` is a store file or a vc_object_server endpoint. Returns its UUID."""
    points = np.ascontiguousarray(np.asarray(xyz, dtype="<f4"))
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"expected a (rows, columns, 3) grid, got {points.shape}")
    pair = (scale, scale) if np.isscalar(scale) else tuple(scale)
    geometry = Surface.from_arrays(points, [float(pair[0]), float(pair[1])])
    arrays = []
    for channel_name, values in sorted((channels or {}).items()):
        data = np.ascontiguousarray(values)
        if tuple(data.shape[:2]) != tuple(points.shape[:2]):
            raise ValueError(f"channel {channel_name!r} shape {data.shape} does not match the grid {points.shape}")
        arrays.append(dict(name=channel_name, dtype=data.dtype.str, shape=list(data.shape), data=data.tobytes()))
    geometry.channels = arrays
    coordinates = dict(volume_uuid=str(volume), scale_factor=float(voxel_scale))
    with attached(store, author=author) as (session, attachment):
        return session.create_surface(attachment, name, list(tags), coordinates, tool, parameters or {}, list(sources),
                                      meta or {}, geometry)
