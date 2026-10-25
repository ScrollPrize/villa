"""Surfaces in a VC object store: read one as a ``Tifxyz`` and create new ones from a coordinate grid.

A surface's geometry is a ``points`` array (rows x columns x 3, xyz, -1 for a missing vertex) with an optional
``valid`` array (nonzero = valid) and a grid ``scale`` attribute, as VC3D stores them.
"""
import getpass
import uuid as uuidlib

import numpy as np

from vesuvius.stored_objects import open_store, volume_scale
from vesuvius.tifxyz.types import Tifxyz

INVALID = -1.0


def read_surface(path, surface_uuid, *, volume, geometry_revision=None):
    """The surface (its current geometry, or ``geometry_revision``) in voxels of ``volume`` as a Tifxyz.

    Vertices the ``valid`` array marks invalid are missing (-1). Raises when the surface is not bound to
    ``volume``."""
    with open_store(path) as store:
        record = store.object(surface_uuid)
        if record is None or record['kind'] != 'surface':
            raise ValueError(f'No surface {surface_uuid} in {path}')
        to_volume = volume_scale(record['meta'], volume)
        if to_volume is None:
            raise ValueError(f'Surface {surface_uuid} is not bound to volume {volume}')
        revision = geometry_revision or store.current_geometry(surface_uuid)['uuid']
        geometry = store.geometry_revision(revision)
        shapes = {array['name']: array['shape'] for array in geometry['arrays']}
        points = np.frombuffer(store.read_geometry_array(revision, 'points'), dtype='<f4').reshape(shapes['points'])
        points = points.copy()
        missing = points[..., 0] == INVALID
        if 'valid' in shapes:
            valid = np.frombuffer(store.read_geometry_array(revision, 'valid'), dtype='u1').reshape(shapes['valid'])
            missing |= valid == 0
        points[missing] = INVALID
        scale = [float(s) for s in geometry['attributes']['scale']]
    if to_volume != 1.0:
        points[~missing] *= np.float32(to_volume)
        scale = [s/to_volume for s in scale]
    return Tifxyz(_x=points[..., 0], _y=points[..., 1], _z=points[..., 2], uuid=surface_uuid,
                  _scale=tuple(scale), extra=dict(record['meta']), _mask=~missing)


def create_surface(path, zyxs, *, name, scale, volume, tool, parameters=None, sources=(), meta=None, tags=(),
                   author=None):
    """A new surface holding ``zyxs`` (rows x columns x 3, zyx in voxels of ``volume``; non-finite or -1 vertices
    are missing), bound to ``volume``. Its meta records ``meta``, the coordinate binding and the provenance
    {tool, parameters, sources}. Returns the new UUID."""
    zyxs = np.asarray(zyxs, dtype=np.float32)
    if zyxs.ndim != 3 or zyxs.shape[2] != 3:
        raise ValueError(f'Expected a rows x columns x 3 grid, got {zyxs.shape}')
    points = np.ascontiguousarray(zyxs[..., ::-1], dtype='<f4')
    missing = ~np.isfinite(points).all(axis=2) | (points == INVALID).any(axis=2)
    points[missing] = INVALID
    surface_uuid = str(uuidlib.uuid4())
    record_meta = dict(meta or {})
    record_meta['coordinates'] = dict(volume_uuid=str(volume), scale_factor=1)
    record_meta['provenance'] = dict(tool=tool, parameters=parameters or {}, sources=list(sources))
    record = dict(uuid=surface_uuid, kind='surface', name=name, tags=list(tags), meta=record_meta)
    geometry = dict(kind='surface', attributes=dict(scale=[float(scale[0]), float(scale[1])]),
                    arrays=[dict(name='points', dtype='<f4', shape=list(points.shape), data=points.tobytes())])
    with open_store(path, read_only=False, author=author or getpass.getuser()) as store:
        with store.write(expected_revisions={surface_uuid: None}) as tx:
            tx.upsert_object(record)
            tx.put_geometry(surface_uuid, geometry)
    return surface_uuid
