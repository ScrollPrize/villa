"""Write small object stores for the object-store input tests.

Requires the vc_object_store Python module (PYTHONPATH=<store build>/python).
"""
import numpy as np

VOLUME = '20260411134726'


def frame(scale=1, volume=VOLUME):
    return {'volume_uuid': volume, 'scale_factor': scale}


def array(name, values, dtype):
    data = np.ascontiguousarray(values, dtype=dtype)
    return {'name': name, 'dtype': dtype, 'shape': list(data.shape),
            'data': data.tobytes()}


def put(db, kind, arrays, attributes=None, *, coordinates=None, meta=None,
        **record):
    """Create an object with geometry; ``coordinates`` None leaves its frame
    undeclared. Returns its UUID."""
    meta = dict(meta or {})
    if coordinates is not None:
        meta['coordinates'] = coordinates
    uid = db.upsert_object({'kind': kind, 'meta': meta, **record})
    db.put_geometry(uid, {'kind': kind, 'attributes': attributes or {},
                          'arrays': arrays})
    return uid


def put_patch(db, zyxs, scale=(1, 1), *, coordinates=frame(), valid=None, **record):
    """A patch from a (rows, cols, 3) fit-frame ZYX grid; -1 rows stay invalid.
    ``valid`` (rows, cols) is stored as the surface validity channel."""
    zyxs = np.asarray(zyxs, dtype=np.float64)
    factor = (coordinates or {}).get('scale_factor') or 1
    invalid = np.all(zyxs == -1, axis=-1)
    xyz = zyxs[..., ::-1] * factor
    xyz[invalid] = -1
    arrays = [array('points', xyz, '<f4')]
    if valid is not None:
        arrays.append(array('valid', np.where(valid, 255, 0), '|u1'))
    return put(db, 'surface', arrays,
               {'scale': [s / factor for s in scale]},
               coordinates=coordinates, **record)


def put_track(db, stored_zyxs, key, family, *, dataset='tracks.vctracks',
              coordinates=frame(), **record):
    """A track from stored integer ZYX voxels, as a packed track store row
    ``key`` of ``dataset`` ingests it."""
    xyz = np.asarray(stored_zyxs, dtype=np.int64)[:, ::-1]
    return put(db, 'track', [array('points', xyz, '<i4')],
               coordinates=coordinates,
               track={'dataset_ref': dataset, 'external_key': str(key),
                      'family': family}, **record)


def crossing_edge(db, a, a_index, a_arclength, b, b_index, b_arclength,
                  clearance, **record):
    """An explicit track crossing at stored point indices, its arclength
    positions and clearance in stored voxels."""
    anchor = lambda index: {'points': [{'array': 'points', 'index': [index]}]}
    return db.add_edge({
        'kind': 'crossing', 'a': a, 'b': b,
        'a_anchor': anchor(a_index), 'b_anchor': anchor(b_index),
        'meta': {'track_crossing': {
            'clearance_voxels': clearance,
            'endpoints': {a: {'arclength_voxels': a_arclength},
                          b: {'arclength_voxels': b_arclength}}}},
        **record})


def write_reference_store(db):
    """Ingest object_store_reference.PATCHES. Returns {patch name: UUID}."""
    import object_store_reference as fixture

    return {patch['name']: put_patch(db, patch['zyxs'], patch['scale'],
                                     uuid=patch['uuid'], name=patch['name'])
            for patch in fixture.PATCHES}


# ---------------------------------------------------------------------------
# The small store the startup, position and service tests share
# ---------------------------------------------------------------------------

def plane(y=100.0, size=8, dx=0.0):
    """A patch grid in the plane y: row r at z 4+4r, column c at x 40+dx+4c."""
    rows, cols = np.mgrid[:size, :size]
    return np.stack([4.0 + rows * 4, np.full(rows.shape, y),
                     40.0 + dx + cols * 4], axis=-1)


def write_walk_store(path):
    """Three patches: 'near' at y=100, 'walk' overlapping it two columns
    along, and 'far' at y=140 overlapping neither. Returns ``path``."""
    from vc_object_store import ObjectStore
    path = str(path)
    with ObjectStore.create(path, author='ingest', hostname='host') as db:
        put_patch(db, plane(), (0.25, 0.25), name='near')
        put_patch(db, plane(140.0), (0.25, 0.25), name='far')
        put_patch(db, plane(dx=8.0), (0.25, 0.25), name='walk')
    return path


def names(path):
    """{object name: UUID} of a store."""
    import object_store_inputs as osi
    from vc_object_store import ObjectStore
    with ObjectStore.open(str(path), read_only=True) as db:
        return {record['name']: record['uuid'] for kind in osi.KINDS
                for record in db.objects_by_kind(kind)}


def edges(path, kind='same-surface'):
    """A store's edges of ``kind``, sorted, with their review state."""
    from vc_object_store import ObjectStore
    with ObjectStore.open(str(path), read_only=True) as db:
        return sorted((edge['uuid'], edge['a'], edge['b'], edge['approval_state'],
                       edge['version'], edge['changed_at'])
                      for edge in db.edges_by_kind(kind))


def fit_config(path, author='fitter', **overrides):
    """The reference fit configuration reading the store at ``path``."""
    import object_store_reference as reference
    config = reference.fit_config()
    config.update(object_store_location=str(path), object_store_author=author)
    config.update(overrides)
    return config


def shared_edit(path, change, author='editor'):
    """An editor's write to the store file (shared with the service on its
    host): returns the commit sequence it reached."""
    from vc_object_store import ObjectStore
    with ObjectStore.open(str(path), author=author) as db:
        change(db)
        return int(db.status()['commit_sequence'])


def head(path):
    """A store's head commit sequence."""
    from vc_object_store import ObjectStore
    with ObjectStore.open(str(path), read_only=True) as db:
        return int(db.status()['commit_sequence'])
