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


def put_patch(db, zyxs, scale=(1, 1), *, coordinates=frame(), **record):
    """A patch from a (rows, cols, 3) fit-frame ZYX grid; -1 rows stay invalid."""
    zyxs = np.asarray(zyxs, dtype=np.float64)
    factor = (coordinates or {}).get('scale_factor') or 1
    invalid = np.all(zyxs == -1, axis=-1)
    xyz = zyxs[..., ::-1] * factor
    xyz[invalid] = -1
    return put(db, 'patch', [array('points', xyz, '<f4')],
               {'scale': [s / factor for s in scale]},
               coordinates=coordinates, **record)


def put_pcl(db, points, *, coordinates=frame(), absolute=None, source=None,
            **record):
    """A PCL from {point id: (fit ZYX, wind_a or None)}; ``source`` adds
    ``meta.source`` fields."""
    factor = (coordinates or {}).get('scale_factor') or 1
    ids = list(points)
    xyz = [np.asarray(points[i][0], dtype=np.float64)[::-1] * factor for i in ids]
    source = {**(source or {}),
              'point_attributes': [{'wind_a': points[i][1]} for i in ids]}
    if absolute is not None:
        source['winding_is_absolute'] = absolute
    return put(db, 'point_collection',
               [array('points', np.reshape(xyz, (-1, 3)), '<f8'),
                array('point_ids', ids, '<i8')],
               coordinates=coordinates, meta={'source': source}, **record)


def put_fiber(db, line_zyxs, control_positions, control_ids=None, *,
              coordinates=frame(), **record):
    """A fiber whose controls lie on the listed line positions."""
    factor = (coordinates or {}).get('scale_factor') or 1
    line = np.asarray(line_zyxs, dtype=np.float64)[:, ::-1] * factor
    controls = line[list(control_positions)]
    if control_ids is None:
        control_ids = list(range(len(control_positions)))
    return put(db, 'fiber',
               [array('control_points', controls, '<f8'),
                array('line_points', line, '<f8'),
                array('control_point_ids', control_ids, '<i8')],
               coordinates=coordinates, **record)


def branch(db, a, a_control_id, b, b_control_id, *, adjacent=False, **record):
    anchor = lambda ident: {'points': [{'array': 'control_point_ids',
                                         'index': [ident]}],
                            'direction': [1, 0, 0]}
    return db.add_edge({'kind': 'fiber_branch', 'a': a, 'b': b,
                        'a_anchor': anchor(a_control_id),
                        'b_anchor': anchor(b_control_id),
                        'meta': {'adjacent': adjacent}, **record})


def pcl_patch_edge(db, pcl, point_id, patch, ij, winding=None, **record):
    """An explicit point-patch edge at patch grid (row, col) ``ij``."""
    row, col = (int(np.floor(v)) for v in ij)
    return db.add_edge({
        'kind': 'point-patch', 'a': pcl, 'b': patch,
        'a_anchor': {'points': [{'array': 'point_ids', 'index': [point_id]}]},
        'b_anchor': {'points': [{'array': 'points', 'index': [row, col]}]},
        'meta': {'point_collection_patch': {
            'ij': [float(v) for v in ij], 'winding_annotation': winding}},
        **record})


def fiber_patch_edge(db, fiber, line_index, patch, ij, **record):
    """An explicit fiber-patch contact at patch grid (row, col) ``ij``."""
    row, col = (int(np.floor(v)) for v in ij)
    return db.add_edge({
        'kind': 'same-surface', 'a': fiber, 'b': patch,
        'a_anchor': {'points': [{'array': 'line_points', 'index': [line_index]}]},
        'b_anchor': {'points': [{'array': 'points', 'index': [row, col]}]},
        'meta': {'fiber_patch': {'median_line_index': line_index,
                                 'patch_grid': [float(ij[1]), float(ij[0])]}},
        **record})


def reference_attachments(reference):
    """{rounded ZYX: [(patch name, ij), ...]} from a reference summary."""
    attachments = {}
    for pcl in reference['constraints']['cross_patch']:
        for name, points in pcl['points_by_patch']:
            for key, ij in points:
                entry = (name, tuple(ij))
                found = attachments.setdefault(tuple(key), [])
                if entry not in found:
                    found.append(entry)
    return attachments


def write_reference_store(db, reference, *, fibers=True):
    """Ingest object_store_reference.FIXTURE with its reference attachments
    as explicit edges. Returns {object name: UUID}."""
    import object_store_reference as fixture
    from object_store_reference import _point_key

    attachments = reference_attachments(reference)

    def attached(zyx):
        return attachments.get(tuple(_point_key({'zyx': np.float32(zyx)})), [])

    uuids = {}
    for patch in fixture.PATCHES:
        uuids[patch['name']] = put_patch(
            db, patch['zyxs'], patch['scale'], uuid=patch['uuid'],
            name=patch['name'])
    for document in fixture.PCL_DOCUMENTS:
        for collection in document['collections']:
            points = {point_id: (point['zyx'], point['wind'])
                      for point_id, point in collection['points'].items()}
            uid = put_pcl(db, points, uuid=collection['uuid'],
                          name=collection['name'],
                          absolute=document['role'] == 'absolute')
            uuids[collection['name']] = uid
            for point_id, point in collection['points'].items():
                for name, ij in attached(point['zyx']):
                    pcl_patch_edge(db, uid, point_id, uuids[name], ij)
    if not fibers:
        return uuids
    for fiber in fixture.FIBERS:
        uid = put_fiber(db, fiber['line'], fiber['controls'],
                        [100 + 7 * k for k in range(len(fiber['controls']))],
                        uuid=fiber['uuid'], name=fiber['name'])
        uuids[fiber['name']] = uid
        for line_index, zyx in enumerate(fiber['line']):
            for name, ij in attached(zyx):
                fiber_patch_edge(db, uid, line_index, uuids[name], ij)
    controls = {fiber['name']: fiber['controls'] for fiber in fixture.FIBERS}
    for a, a_control, b, b_control in fixture.FIBER_BRANCHES:
        branch(db, uuids[a], 100 + 7 * a_control, uuids[b], 100 + 7 * b_control)
        assert a_control < len(controls[a]) and b_control < len(controls[b])
    return uuids
