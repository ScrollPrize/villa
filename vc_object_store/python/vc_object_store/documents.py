"""Documents attached to objects: a tool's results for a surface, stored as a point collection hanging off it by
an ``attachment`` relationship (objects/AttachedDocument.hpp). Each record of the document's ``list_key`` list is
one point at its ``position_key`` xyz, annotated with the record; the rest of the document travels with it.
"""
from __future__ import annotations

import getpass
import uuid as uuidlib

from ._native import ObjectStore


def attach_document(store, object_uuid, document, *, role, list_key, position_key, name, volume, author=None) -> str:
    """Attach ``document`` to ``object_uuid`` in ``role`` as a new point collection named ``name`` whose positions
    are in voxels of ``volume``. Returns the collection's UUID."""
    collection_uuid = str(uuidlib.uuid4())
    coordinates = dict(volume_uuid=str(volume), scale_factor=1)
    with ObjectStore.open(str(store), author=author or getpass.getuser()) as db:
        head = db.current_object_revision(object_uuid)
        expected = {collection_uuid: None, object_uuid: head["uuid"] if head else None}
        with db.write(expected_revisions=expected) as tx:
            tx.attach_document(object_uuid, collection_uuid, document, role, list_key, position_key, name, coordinates)
    return collection_uuid


def attached_document(store, object_uuid, role):
    """The document attached to ``object_uuid`` in ``role``, or None."""
    with ObjectStore.open(str(store), read_only=True, reader_pool_size=1) as db:
        return db.attached_document(object_uuid, role)



def point_collections(store, uuids, volume) -> list[dict]:
    """The point collections ``uuids`` (any role; objects/AttachedDocument.hpp ``pointCollections``) in ``volume``'s
    voxels, in that order: each ``uuid``, ``name``, point ``ids`` and ``points``, one annotation per point and the
    rest of the collection as ``source``."""
    with ObjectStore.open(str(store), read_only=True, reader_pool_size=1) as db:
        return db.point_collections([str(u) for u in uuids], str(volume))
