"""Presence snapping of fiber labels (data/fiber_snapping.py, scripts/snap_fibers.py) and the AFV writer (data/afv.py)."""
import json
import sqlite3

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.data.fiber_snapping import SnapConfig, resample, snap_polylines

AFV_SCHEMA = (
    'CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)',
    '''CREATE TABLE fibers(id INTEGER PRIMARY KEY, name TEXT NOT NULL, family TEXT NOT NULL,
       point_count INTEGER NOT NULL CHECK(point_count >= 2), length REAL NOT NULL,
       min_x REAL NOT NULL, max_x REAL NOT NULL, min_y REAL NOT NULL, max_y REAL NOT NULL,
       min_z REAL NOT NULL, max_z REAL NOT NULL, annotation TEXT NOT NULL)''',
    '''CREATE TABLE blocks(id INTEGER PRIMARY KEY, fiber_id INTEGER NOT NULL REFERENCES fibers(id),
       first_segment INTEGER NOT NULL, points BLOB NOT NULL, UNIQUE(fiber_id, first_segment))''',
    'CREATE VIRTUAL TABLE block_bounds USING rtree(id, min_x, max_x, min_y, max_y, min_z, max_z)')


class ArrayVolume:
    """A zyx numpy volume with the ChunkedArray read interface."""
    def __init__(self, values):
        self.values, self.shape, self.dtype = values, values.shape, values.dtype

    def read(self, start, size):
        return self.values[tuple(slice(a, a+n) for a, n in zip(start, size))]


def tube_volume(centre, shape=(48, 48, 96), sigma=.8):
    """uint8 presence: a Gaussian tube (sigma voxels) around ``centre`` (xyz samples)."""
    from scipy.spatial import cKDTree
    z, y, x = np.indices(shape)
    distance = cKDTree(resample(centre, .1)).query(np.c_[x.ravel(), y.ravel(), z.ravel()])[0].reshape(shape)
    return (255*np.exp(-.5*(distance/sigma)**2)).astype(np.uint8)


def wavy(offset=(0., 0., 0.)):
    x = np.arange(4., 92., 1.)
    return np.c_[x, 24+3*np.sin(x/9.), 24+2*np.cos(x/13.)]+np.asarray(offset)


def test_snapping_moves_an_offset_label_onto_the_presence_centre():
    centre = wavy()
    presence = ArrayVolume(tube_volume(centre))
    label = wavy((0., 1.4, -.9))  # off the fiber by ~1.7 voxels
    snapped = snap_polylines([label], presence, SnapConfig(tile=32, threads=2), device='cpu')[0]
    from scipy.spatial import cKDTree
    error = cKDTree(resample(centre, .1)).query(snapped[8:-8])[0]
    before = cKDTree(resample(centre, .1)).query(label[8:-8])[0]
    assert before.mean() > 1.5 and error.mean() < .25 and error.max() < .6
    # A label already on the fiber stays there.
    kept = snap_polylines([centre], presence, SnapConfig(tile=32, threads=2), device='cpu')[0]
    assert cKDTree(resample(centre, .1)).query(kept[8:-8])[0].max() < .25


def make_afv(path, fibers):
    c = sqlite3.connect(path)
    c.execute('PRAGMA application_id = %d' % 0x56434643)
    c.execute('PRAGMA user_version = 1')
    for sql in AFV_SCHEMA:
        c.execute(sql)
    for fid, (name, family, xyz) in enumerate(fibers, 1):
        c.execute('INSERT INTO fibers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                  (fid, name, family, len(xyz), 0., *[0.]*6, json.dumps({'source': 'test'})))
        c.execute('INSERT INTO blocks VALUES (?,?,?,?)', (fid, fid, 0, np.ascontiguousarray(xyz, '<f8').tobytes()))
        c.execute('INSERT INTO block_bounds VALUES (?,?,?,?,?,?,?)', (fid, *[v for k in range(3) for v in (xyz[:, k].min(), xyz[:, k].max())]))
    meta = dict(frame=dict(vc_open_data_coordinate_space='test/1'), root={}, fiber_count=len(fibers),
                point_count=sum(len(f[2]) for f in fibers), uuid='u')
    c.executemany('INSERT INTO metadata VALUES (?, ?)', [(k, json.dumps(v)) for k, v in meta.items()])
    c.execute("INSERT INTO metadata VALUES ('complete', 'true')")
    c.commit()
    c.close()


def test_afv_snapping_keeps_ids_names_and_families_and_reads_back(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.data.afv import AFVFibers, read_afv_polylines
    from vesuvius.neural_tracing.fiber_follow.data import fiber_snapping as snap_fibers
    scale = 8.
    centre = wavy()
    make_afv(tmp_path/'in.afv', [('a', 'H', wavy((0., 1.4, -.9))*scale), ('b', 'V', centre*scale)])
    presence = ArrayVolume(tube_volume(centre))
    result = snap_fibers.snap_afv(tmp_path/'in.afv', tmp_path/'out.afv', presence, scale, 0., SnapConfig(tile=32, threads=2),
                                  'cpu', 1., 0, dict(method='test'))
    assert result['fibers'] == 2
    rows = read_afv_polylines(tmp_path/'out.afv')
    assert [(r['id'], r['name'], r['family']) for r, _ in rows] == [(1, 'a', 'H'), (2, 'b', 'V')]
    assert json.loads(rows[0][0]['annotation']) == {'source': 'test'}
    fibers = AFVFibers(tmp_path/'out.afv', scale, split='all')
    assert fibers.metadata['snapping'] == {'method': 'test'} and len(fibers) == 2
    from scipy.spatial import cKDTree
    assert cKDTree(resample(centre, .1)).query(fibers[0].points[10:-10])[0].mean() < .3
    with pytest.raises(FileExistsError):
        snap_fibers.snap_afv(tmp_path/'in.afv', tmp_path/'out.afv', presence, scale, 0., SnapConfig(tile=32), 'cpu', 1., 0, {})


def test_json_snapping_rewrites_the_line_and_keeps_controls_on_it(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.data import fiber_snapping as snap_fibers
    scale = 8.
    centre = wavy()
    line = wavy((0., 1.4, -.9))*scale
    document = dict(type='vc3d_fiber', version=3, line_points=line.tolist(),
                    control_points=[dict(position=line[i].tolist()) for i in (0, 40, len(line)-1)])
    (tmp_path/'in').mkdir()
    (tmp_path/'in'/'f.json').write_text(json.dumps(document))
    presence = ArrayVolume(tube_volume(centre))
    snap_fibers.snap_json([tmp_path/'in'/'f.json'], tmp_path/'out', presence, scale, 0., SnapConfig(tile=32, threads=2),
                          'cpu', 1., dict(method='test'))
    out = json.loads((tmp_path/'out'/'f.json').read_text())
    new = np.asarray(out['line_points'])
    assert all(np.min(np.linalg.norm(new-np.asarray(c['position']), axis=1)) < 1e-9 for c in out['control_points'])
    from scipy.spatial import cKDTree
    assert cKDTree(resample(centre, .1)).query(new[10:-10]/scale)[0].mean() < .3
    assert json.loads((tmp_path/'out'/'snapping.json').read_text())['files'] == ['f.json']
