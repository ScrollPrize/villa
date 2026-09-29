"""Small Automated Fiber Volumes written in the format documented in
volume-cartographer/docs/fiber-collections.md, for tests."""
import json
import math
import sqlite3
import struct
import uuid

BASE_SHAPE_ZYX = [400, 300, 300]
FRAME = {'vc_open_data_coordinate_space': 'test', 'coordinate_base_shape_zyx': BASE_SHAPE_ZYX}
SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE fibers(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, family TEXT NOT NULL,
  point_count INTEGER NOT NULL CHECK(point_count >= 2), length REAL NOT NULL,
  min_x REAL NOT NULL, max_x REAL NOT NULL, min_y REAL NOT NULL, max_y REAL NOT NULL,
  min_z REAL NOT NULL, max_z REAL NOT NULL, annotation TEXT NOT NULL);
CREATE INDEX fibers_length ON fibers(length DESC, id);
CREATE TABLE blocks(
  id INTEGER PRIMARY KEY, fiber_id INTEGER NOT NULL REFERENCES fibers(id),
  first_segment INTEGER NOT NULL, points BLOB NOT NULL,
  UNIQUE(fiber_id, first_segment));
CREATE VIRTUAL TABLE block_bounds USING rtree(id, min_x, max_x, min_y, max_y, min_z, max_z);
"""


def fiber(z, *, count=40, step=2.0, tag='H'):
    """A straight native VC3D fiber along X at height z."""
    line = [[20.0 + step * i, 150.0, float(z)] for i in range(count)]
    return {
        'type': 'vc3d_fiber', 'version': 1,
        'line_points': line, 'control_points': [line[0], line[count // 2], line[-1]],
        'branches': [],
        'hv_classification': {'automatic_tag': tag, 'automatic_certainty': 1.0, 'manual_tag': tag},
        'sequence': 1, 'started_at': '20260101T000000000', 'tags': [], 'username': 'test',
    }


def _bounds(points):
    return [f(p[axis] for p in points) for axis in range(3) for f in (min, max)]


def write_afv(path, fibers, *, frame=FRAME):
    db = sqlite3.connect(path)
    try:
        db.executescript(SCHEMA)
        db.execute('PRAGMA application_id=%d' % 0x56434643)
        db.execute('PRAGMA user_version=1')
        block_id = point_total = 0
        for fiber_id, data in enumerate(fibers, 1):
            points = data['line_points']
            family = data['hv_classification']['manual_tag']
            db.execute('INSERT INTO fibers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)', (
                fiber_id, f'fiber_{fiber_id}', family if family in ('H', 'V') else '', len(points),
                sum(math.dist(a, b) for a, b in zip(points, points[1:])), *_bounds(points),
                json.dumps({k: v for k, v in data.items() if k != 'line_points'})))
            for start in range(0, len(points) - 1, 256):
                part = points[start:start + 257]
                block_id += 1
                db.execute('INSERT INTO blocks VALUES(?,?,?,?)', (
                    block_id, fiber_id, start, b''.join(struct.pack('<ddd', *p) for p in part)))
                db.execute('INSERT INTO block_bounds VALUES(?,?,?,?,?,?,?)', (block_id, *_bounds(part)))
            point_total += len(points)
        metadata = {'complete': True, 'uuid': str(uuid.uuid4()), 'frame': frame, 'root': {},
                    'fiber_count': len(fibers), 'point_count': point_total}
        db.executemany('INSERT INTO metadata VALUES(?,?)', [(k, json.dumps(v)) for k, v in metadata.items()])
        db.commit()
    finally:
        db.close()
    return path
