"""Surfaces as numpy grids: read one in a volume's voxels, create one from a grid.

A stored surface's geometry is a ``points`` array (rows x columns x 3, xyz, -1 for a missing vertex; the store's validity rule decides which
vertices are missing), an optional
``valid`` array and per-vertex channels, with a grid ``scale``; ``meta.coordinates`` binds it to a volume.
"""
from __future__ import annotations

import getpass
import uuid as uuidlib
from typing import NamedTuple

import numpy as np

from ._native import ObjectStore, Surface

INVALID = -1.0


class SurfaceGrid(NamedTuple):
    points: np.ndarray  # (rows, columns, 3) float32 xyz, -1 at missing vertices
    valid: np.ndarray  # (rows, columns) bool
    scale: tuple  # grid scale in the read volume's voxels
    record: dict
    version: int  # the numbered object version read (in ``generation``, the store's active one)
    generation: str
    geometry_revision: str  # payload identity of its geometry (reads its channels)
    channels: frozenset  # names of the per-vertex channels (not points or valid)


def read_surface(store, surface_uuid, *, volume, version=None, target_scale=1.0) -> SurfaceGrid:
    """The surface (its current version, or numbered ``version`` in the store's active generation) in voxels of
    ``volume`` at ``target_scale``, through the store's mapping to it (its binding, or a registration). Raises
    ValueError when the surface (version) does not exist or does not map to ``volume``."""
    with ObjectStore.open(str(store), read_only=True, reader_pool_size=1) as db:
        # Every read through db comes before the snapshot, which holds the one pooled reader.
        status = db.status()
        generation = status["generation"]
        if version is None:
            record = db.object(surface_uuid)
            if record is None or record["kind"] != "surface":
                raise ValueError(f"no surface {surface_uuid} in {store}")
            geometry = None
        else:
            revision = db.object_at_version(status["store"], generation, surface_uuid, int(version))
            if revision is None or revision["deleted"] or revision["record"]["kind"] != "surface":
                raise ValueError(f"no surface {surface_uuid} version {version} in {store}")
            if revision["geometry_revision"] is None:
                raise ValueError(f"surface {surface_uuid} version {version} has no geometry")
            record, geometry = revision["record"], revision["geometry_revision"]
        with db.geometry_snapshot() as snapshot:
            canonical = snapshot.surface(surface_uuid, geometry)
            read_version = canonical.version if version is None else int(version)
            try:
                view = canonical.view_transformed_at(volume, target_scale)
            except RuntimeError as error:
                raise ValueError(f"surface {surface_uuid} does not map to volume {volume}: {error}") from error
            tile = view.read_tile(0, 0, canonical.rows, canonical.columns)
            revision = canonical.revision
        if version is None:  # the record of the version the snapshot read
            record = db.object_at_version(status["store"], generation, surface_uuid, read_version)["record"]
    points = np.array(tile["points"], dtype=np.float32)
    valid = np.asarray(tile["validity"]).astype(bool)
    points[~valid] = INVALID
    channels = frozenset(array["name"] for array in revision["arrays"]) - {"points", "valid"}
    return SurfaceGrid(points, valid, tuple(float(s) for s in tile["scale"]), record, read_version, generation,
                       revision["uuid"], channels)


def create_surface(store, xyz, *, name, scale, volume, tool, parameters=None, sources=(), meta=None,
                   channels=None, tags=(), author=None, voxel_scale=1.0):
    """A new surface holding ``xyz`` ((rows, columns, 3) in voxels of ``volume`` at ``voxel_scale`` of its own;
    vertices are missing by the store's validity rule) with grid ``scale`` (one number or a pair), bound to those
    voxels, with per-vertex ``channels`` and
    meta.provenance {tool, parameters, sources}. ``meta`` is kept except tifxyz file keys (the store's
    createSurface). Returns its UUID."""
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
    surface_uuid = str(uuidlib.uuid4())
    with ObjectStore.open(str(store), author=author or getpass.getuser()) as db:
        with db.write(expected_versions={surface_uuid: None}) as tx:
            tx.create_surface(surface_uuid, name, list(tags), dict(volume_uuid=str(volume), scale_factor=float(voxel_scale)), tool,
                              parameters or {}, list(sources), meta or {}, geometry)
    return surface_uuid
