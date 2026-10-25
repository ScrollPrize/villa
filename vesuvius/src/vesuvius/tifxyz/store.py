"""Surfaces in a VC object store: read one as a ``Tifxyz`` and create new ones from a coordinate grid.

A surface's geometry is a ``points`` array (rows x columns x 3, xyz, -1 for a missing vertex) with an optional
``valid`` array (nonzero = valid) and a grid ``scale`` attribute, as VC3D stores them.
"""
import getpass
import uuid as uuidlib

import numpy as np
from vc_object_store import ObjectStore, Surface, geometry_scale_to_volume

from vesuvius.tifxyz.types import Tifxyz

INVALID = -1.0


def read_surface(path, surface_uuid, *, volume, geometry_revision=None):
    """The surface (its current geometry, or ``geometry_revision``) in voxels of ``volume`` as a Tifxyz.

    Missing vertices, by the store's rule (coordinates and the ``valid`` array), are -1. Raises when the surface
    is not bound to ``volume``."""
    with ObjectStore.open(str(path), read_only=True, reader_pool_size=1) as store:
        record = store.object(surface_uuid)
        if record is None or record['kind'] != 'surface':
            raise ValueError(f'No surface {surface_uuid} in {path}')
        to_volume = geometry_scale_to_volume(record['meta'], volume)
        if to_volume is None:
            raise ValueError(f'Surface {surface_uuid} is not bound to volume {volume}')
        with store.geometry_snapshot() as snapshot:
            view = snapshot.surface(surface_uuid, geometry_revision)
            points, valid = view.read_arrays()
            scale = list(view.scale)
    points[~valid] = INVALID
    if to_volume != 1.0:
        points[valid] *= np.float32(to_volume)
        scale = [s/to_volume for s in scale]
    return Tifxyz(_x=points[..., 0], _y=points[..., 1], _z=points[..., 2], uuid=surface_uuid,
                  _scale=tuple(scale), extra=dict(record['meta']), _mask=valid)


def create_surface(path, zyxs, *, name, scale, volume, tool, parameters=None, sources=(), meta=None, tags=(),
                   author=None):
    """A new surface holding ``zyxs`` (rows x columns x 3, zyx in voxels of ``volume``; non-finite or -1 vertices
    are missing), bound to ``volume``. Its meta records ``meta``, the coordinate binding and the provenance
    {tool, parameters, sources}. Returns the new UUID."""
    zyxs = np.asarray(zyxs, dtype=np.float32)
    if zyxs.ndim != 3 or zyxs.shape[2] != 3:
        raise ValueError(f'Expected a rows x columns x 3 grid, got {zyxs.shape}')
    points = np.ascontiguousarray(zyxs[..., ::-1], dtype='<f4')
    points[~np.isfinite(points).all(axis=2) | (points == INVALID).any(axis=2)] = INVALID
    surface_uuid = str(uuidlib.uuid4())
    record_meta = dict(meta or {})
    record_meta['coordinates'] = dict(volume_uuid=str(volume), scale_factor=1)
    record_meta['provenance'] = dict(tool=tool, parameters=parameters or {}, sources=list(sources))
    record = dict(uuid=surface_uuid, kind='surface', name=name, tags=list(tags), meta=record_meta)
    with ObjectStore.open(str(path), author=author or getpass.getuser()) as store:
        with store.write(expected_revisions={surface_uuid: None}) as tx:
            tx.upsert_object(record)
            tx.put_geometry(surface_uuid, Surface.from_arrays(points, [float(scale[0]), float(scale[1])]))
    return surface_uuid
