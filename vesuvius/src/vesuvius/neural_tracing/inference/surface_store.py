"""Surfaces in a VC object store: read one as a ``tifxyz.Surface`` and create new ones from a coordinate grid.

The store's ``vc_object_store.surfaces`` does the reading (in a volume's voxels, through the store's mapping) and
writing; this adapts its xyz grids to ``tifxyz.Surface`` and zyx arrays.
"""
import numpy as np
from vc_object_store import surfaces
from vc_reader import tifxyz


def read_surface(path, surface_uuid, *, volume, geometry_revision=None):
    """The surface (its current geometry, or ``geometry_revision``) in voxels of ``volume``.

    Missing vertices, by the store's rule (coordinates and the ``valid`` array), are -1. Raises when the surface
    does not map to ``volume``."""
    grid = surfaces.read_surface(path, surface_uuid, volume=volume, geometry_revision=geometry_revision)
    return tifxyz.Surface(grid.points, mask=grid.valid, scale=tuple(grid.scale), uuid=surface_uuid,
                          extra=dict(grid.record['meta']))


def create_surface(path, zyxs, *, name, scale, volume, tool, parameters=None, sources=(), meta=None, tags=(),
                   author=None):
    """A new surface holding ``zyxs`` (rows x columns x 3, zyx in voxels of ``volume``; non-finite or -1 vertices
    are missing), bound to ``volume``. Its meta records ``meta``, the coordinate binding and the provenance
    {tool, parameters, sources}. Returns the new UUID."""
    zyxs = np.asarray(zyxs, dtype=np.float32)
    if zyxs.ndim != 3 or zyxs.shape[2] != 3:
        raise ValueError(f'Expected a rows x columns x 3 grid, got {zyxs.shape}')
    return surfaces.create_surface(path, zyxs[..., ::-1], name=name, scale=scale, volume=volume, tool=tool,
                                   parameters=parameters, sources=sources, meta=meta, tags=tags, author=author)
