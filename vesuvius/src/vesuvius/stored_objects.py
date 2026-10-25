"""Objects in a VC object store (``vc_object_store``), any kind: store handles and volume binding.

An object's geometry is stored in the frame its ``meta.coordinates`` declares ({volume_uuid, scale_factor}:
stored = volume voxel * scale_factor).
"""
from vc_object_store import ObjectStore


def open_store(path, *, read_only=True, author=None):
    """A store handle; a context manager. Not shareable across processes: open one per process.

    Writers open the same file a caller's session uses (a local store, or a hosted hub's local replica); that
    session pushes the writes to the hub."""
    if read_only:
        return ObjectStore.open(str(path), read_only=True, reader_pool_size=1)
    return ObjectStore.open(str(path), author=author)


def volume_scale(meta, volume):
    """Stored-to-volume-voxel factor of an object bound to ``volume``, else None."""
    coordinates = (meta or {}).get('coordinates') or {}
    if coordinates.get('volume_uuid') != volume or not coordinates.get('scale_factor'):
        return None
    return 1.0/float(coordinates['scale_factor'])
