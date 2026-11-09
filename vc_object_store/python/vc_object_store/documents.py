"""Documents attached to objects: a tool's results for a surface, stored as a point collection hanging off it by
an ``attachment`` relationship (objects/AttachedDocument.hpp). Each record of the document's ``list_key`` list is
one point at its ``position_key`` xyz, annotated with the record; the rest of the document travels with it.

``store`` is a store file or a vc_object_server endpoint (``locations``).
"""
from __future__ import annotations

from .locations import attached


def attach_document(store, object_uuid, document, *, role, list_key, position_key, name, volume, author=None,
                    voxel_scale=1.0) -> str:
    """Attach ``document`` to ``object_uuid`` (its current version) in ``role`` as a new point collection named
    ``name`` whose positions are in voxels of ``volume`` at ``voxel_scale`` of its own. Returns the collection's
    UUID."""
    coordinates = dict(volume_uuid=str(volume), scale_factor=float(voxel_scale))
    with attached(store, author=author) as (session, attachment):
        return session.attach_document(attachment, str(object_uuid), document, role, list_key, position_key, name,
                                       coordinates)


def attached_document(store, object_uuid, role):
    """The document attached to ``object_uuid`` in ``role``, or None."""
    with attached(store, read_only=True) as (session, attachment):
        return session.attached_document(attachment, str(object_uuid), role)


def point_collections(store, uuids, volume, voxel_scale=1.0) -> list[dict]:
    """The point collections ``uuids`` (any role; objects/AttachedDocument.hpp ``pointCollections``) in ``volume``'s
    voxels at ``voxel_scale`` of its own, in that order: each ``uuid``, ``name``, point ``ids`` and ``points``, one
    annotation per point and the rest of the collection as ``source``."""
    with attached(store, read_only=True) as (session, attachment):
        return session.point_collections(attachment, [str(u) for u in uuids], str(volume), float(voxel_scale))
