"""Surfaces as numpy grids: read one in a volume's voxels, create one from a grid.

A stored surface's geometry is a ``points`` array (rows x columns x 3, xyz, -1 for a missing vertex), an optional
``valid`` array and per-vertex channels, with a grid ``scale``; ``meta.coordinates`` binds it to a volume.
"""
from __future__ import annotations

import getpass
import uuid as uuidlib

import numpy as np

from ._native import ObjectStore, Surface

INVALID = -1.0
# tifxyz meta.json keys that describe the file, not the surface.
_TIFXYZ_FILE_KEYS = ("uuid", "name", "format", "type", "scale", "bbox")


def read_surface(store, surface_uuid, *, volume, geometry_revision=None, target_scale=1.0):
    """(points, valid, scale, record): the surface (its current geometry, or ``geometry_revision``) in voxels of
    ``volume`` at ``target_scale``, through the store's mapping to it (its binding, or a registration). ``points``
    is (rows, columns, 3) float32 xyz with -1 at missing vertices, ``valid`` a bool mask, ``scale`` the grid
    scale in those voxels. Raises when the surface does not map to ``volume``."""
    with ObjectStore.open(str(store), read_only=True, reader_pool_size=1) as db:
        record = db.object(surface_uuid)
        if record is None or record["kind"] != "surface":
            raise ValueError(f"no surface {surface_uuid} in {store}")
        with db.geometry_snapshot() as snapshot:
            canonical = snapshot.surface(surface_uuid, geometry_revision)
            tile = canonical.view_transformed_at(volume, target_scale).read_tile(
                0, 0, canonical.rows, canonical.columns)
    points = np.array(tile["points"], dtype=np.float32)
    valid = np.asarray(tile["validity"]).astype(bool)
    points[~valid] = INVALID
    return points, valid, tuple(float(s) for s in tile["scale"]), record


def create_surface(store, xyz, *, name, scale, volume, tool, parameters=None, sources=(), meta=None,
                   channels=None, tags=(), author=None):
    """A new surface holding ``xyz`` ((rows, columns, 3) in voxels of ``volume``; non-finite or -1 vertices are
    missing) with grid ``scale`` (one number or a pair), bound to ``volume``, with per-vertex ``channels`` and
    meta.provenance {tool, parameters, sources}. ``meta`` is kept except tifxyz file keys. Returns its UUID."""
    points = np.ascontiguousarray(np.asarray(xyz, dtype="<f4"))
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"expected a (rows, columns, 3) grid, got {points.shape}")
    points[~np.isfinite(points).all(axis=2) | (points == INVALID).any(axis=2)] = INVALID
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
    record_meta = {key: value for key, value in (meta or {}).items() if key not in _TIFXYZ_FILE_KEYS}
    record_meta["coordinates"] = dict(volume_uuid=str(volume), scale_factor=1)
    record_meta["provenance"] = dict(tool=tool, parameters=parameters or {}, sources=list(sources))
    record = dict(uuid=surface_uuid, kind="surface", name=name, tags=list(tags), meta=record_meta)
    with ObjectStore.open(str(store), author=author or getpass.getuser()) as db:
        with db.write(expected_revisions={surface_uuid: None}) as tx:
            tx.upsert_object(record)
            tx.put_geometry(surface_uuid, geometry)
    return surface_uuid
