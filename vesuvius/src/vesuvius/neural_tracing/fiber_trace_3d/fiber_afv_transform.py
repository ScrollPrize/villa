"""Register whole AFV polylines into another scan, optionally selecting a Z slab."""
import json
from pathlib import Path
import sqlite3
import time
import uuid

import numpy as np

from vesuvius.neural_tracing.tracers.shared.annotations.afv import read_afv_polylines, write_afv
from .fiber_min_flow_experiment import write, digest


def mapped_bounds(lo, hi, matrix):
    """Exact affine bounds of axis-aligned boxes (not sampled corners)."""
    center = (lo + hi) / 2
    radius = (hi - lo) / 2
    mid = center @ matrix[:3, :3].T + matrix[:3, 3]
    extent = radius @ np.abs(matrix[:3, :3]).T
    return mid - extent, mid + extent


def run(config):
    started = time.monotonic()
    source = Path(config['afv']); root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    transform = json.loads(Path(config['transform']).read_text())
    matrix = np.vstack((np.asarray(transform['transformation_matrix'], dtype=float), [0, 0, 0, 1]))
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Expected finite XYZ 3x4 affine transform')
    with sqlite3.connect(f'file:{source.resolve()}?mode=ro', uri=True) as c:
        metadata = {k: json.loads(v) for k, v in c.execute('SELECT key,value FROM metadata')}
        rows = c.execute('SELECT id,min_x,min_y,min_z,max_x,max_y,max_z FROM fibers ORDER BY id').fetchall()
    frame = metadata['frame']
    if (frame.get('vc_open_data_source_coordinate_level', 0) != 0 or
            frame.get('vc_open_data_source_coordinate_scale_factor', 1) != 1):
        raise ValueError('Source must contain native L0 coordinates')
    if frame['vc_open_data_coordinate_space'] != config['source_coordinate_space']:
        raise ValueError('Source coordinate space mismatch')
    if transform['fixed_volume'] != config['fixed_volume']:
        raise ValueError('Registration fixed volume mismatch')
    if config.get('invert', False):
        matrix = np.linalg.inv(matrix)
    with sqlite3.connect(f"file:{Path(config['frame_template']).resolve()}?mode=ro", uri=True) as c:
        target = {k: json.loads(v) for k, v in c.execute('SELECT key,value FROM metadata') if k in ('root','frame')}
    if target['frame']['vc_open_data_coordinate_space'] != config['target_coordinate_space']:
        raise ValueError('Target coordinate space mismatch')
    limits = config['z_range']
    source_ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    bounds = np.asarray([r[1:] for r in rows], dtype=float).reshape(-1,6)
    lo, hi = mapped_bounds(bounds[:, :3], bounds[:, 3:], matrix)
    ids = source_ids[(hi[:, 2] >= limits[0]) & (lo[:, 2] <= limits[1])]
    print(json.dumps(dict(stage='affine_candidates', candidates=len(ids), total=len(rows))), flush=True)
    del rows, bounds, lo, hi
    selected = []
    batch = int(config.get('batch_fibers', 8192))
    for begin in range(0, len(ids), batch):
        for row, xyz in read_afv_polylines(source, ids[begin:begin+batch]):
            points = xyz @ matrix[:3, :3].T + matrix[:3, 3]
            if points[:, 2].max() >= limits[0] and points[:, 2].min() <= limits[1]:
                selected.append((row, points))
        print(json.dumps(dict(stage='affine_polylines', done=min(begin+batch,len(ids)), total=len(ids),
                              selected=len(selected), seconds=time.monotonic()-started)), flush=True)
    provenance = dict(operation='affine_registration_whole_fiber_selection', source=str(source),
        source_uuid=metadata.get('uuid'), source_metadata=metadata, transform=transform,
        applied_matrix_xyz=matrix.tolist(), z_range=limits, clipped=False)
    signature = digest(config['transform']) + str(source.resolve()) + str(limits)
    result = write_afv(None, root/'fibers.afv', selected,
        dict(**target, uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, signature)), provenance=provenance))
    result.update(seconds=time.monotonic()-started, candidates=len(ids), source_fibers=metadata['fiber_count'])
    write(root/'import.json', result)
    print(json.dumps(result), flush=True)
    return result
