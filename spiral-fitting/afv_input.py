"""Read-only AFV input for the standard Spiral fiber loader.

Spatial selection only avoids decoding irrelevant fibers. Complete stored
polylines enter Spiral's existing decimation and z-window logic unchanged.
No individual fiber files are written.
"""
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


def _metadata(db):
    db.execute('PRAGMA query_only=ON')
    if (db.execute('PRAGMA application_id').fetchone()[0] != 0x56434643
            or db.execute('PRAGMA user_version').fetchone()[0] != 1):
        raise ValueError('Unsupported AFV database')
    meta = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
    if meta.get('complete') is not True:
        raise ValueError('AFV is incomplete')
    root, frame = meta['root'], meta['frame']
    # The format allows the coordinate fields in frame or root.
    coordinates = dict(frame)
    for key, value in root.items():
        if key == 'coordinate_base_shape_zyx' or key.startswith('vc_open_data_'):
            if coordinates.setdefault(key, value) != value:
                raise ValueError(f'AFV frame and root disagree on {key}')
    if (root.get('scale', 1) != 1
            or coordinates.get('vc_open_data_source_coordinate_level', 0) != 0
            or coordinates.get('vc_open_data_source_coordinate_scale_factor', 1) != 1):
        raise ValueError('AFV inputs must use native L0 coordinates')
    shape = coordinates.get('coordinate_base_shape_zyx')
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError('AFV must declare its coordinate domain')
    return meta, coordinates


def validate_afv_container(path):
    """Check the shared container contract without decoding every fiber.

    Returns the coordinate fields the fibers inherit.
    """
    db = sqlite3.connect(Path(path).resolve(strict=True).as_uri()+'?mode=ro', uri=True)
    try:
        return _metadata(db)[1]
    finally:
        db.close()


def iter_afv_fibers(path, *, z_range=None):
    """Yield the fibers; z_range filters native AFV coordinates."""
    path = Path(path).resolve(strict=True)
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        meta, coordinates = _metadata(db)
        query = 'SELECT id,name,point_count,annotation FROM fibers'
        params = ()
        if z_range is not None:
            lo, hi = map(float, z_range)
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError('Invalid AFV z range')
            query += ' WHERE max_z>=? AND min_z<?'
            params = (lo, hi)
        query += ' ORDER BY id'
        for identity, name, count, annotation in db.execute(query, params):
            parts, size = [], 0
            revision = hashlib.sha256(annotation.encode())
            for first, blob in db.execute(
                    'SELECT first_segment,points FROM blocks WHERE fiber_id=? ORDER BY first_segment',
                    (identity,)):
                if len(blob) % 24 or len(blob) < 48:
                    raise ValueError(f'Invalid AFV block in fiber {identity}')
                points = np.frombuffer(blob, dtype='<f8').reshape(-1, 3)
                if not np.isfinite(points).all():
                    raise ValueError(f'Nonfinite AFV geometry in fiber {identity}')
                if first != (size - 1 if parts else 0):
                    raise ValueError(f'Noncontiguous AFV blocks in fiber {identity}')
                if parts and not np.array_equal(parts[-1][-1], points[0]):
                    raise ValueError(f'AFV block boundary mismatch in fiber {identity}')
                revision.update(blob)
                part = points[1:] if parts else points
                parts.append(part)
                size += len(part)
            if size != count:
                raise ValueError(f'AFV point count mismatch in fiber {identity}')
            data = json.loads(annotation)
            if data.get('branches') or data.get('adjacent_branches'):
                raise ValueError('AFV cross-fiber branch references are not supported by this reader')
            for key, value in coordinates.items():
                if key in data and data[key] != value:
                    raise ValueError(f'AFV fiber {identity} has a conflicting coordinate frame')
                data[key] = value
            data['line_points'] = np.concatenate(parts).tolist()
            data.setdefault('name', name)
            logical = f"afv-{meta['uuid']}-{identity}"
            yield logical, data, {
                'afv_path': str(path), 'afv_uuid': meta['uuid'], 'afv_id': identity,
                'afv_revision': revision.hexdigest(), 'read_only': True,
            }
    finally:
        db.close()
