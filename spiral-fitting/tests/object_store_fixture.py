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


def put_pcl(db, points, *, coordinates=frame(), absolute=None, **record):
    """A PCL from {point id: (fit ZYX, wind_a or None)}."""
    factor = (coordinates or {}).get('scale_factor') or 1
    ids = list(points)
    xyz = [np.asarray(points[i][0], dtype=np.float64)[::-1] * factor for i in ids]
    source = {'point_attributes': [{'wind_a': points[i][1]} for i in ids]}
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
