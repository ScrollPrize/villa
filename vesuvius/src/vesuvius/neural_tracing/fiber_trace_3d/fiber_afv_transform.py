"""Register whole AFV polylines into another scan, optionally selecting a Z slab."""
import json
from pathlib import Path
import time
import uuid

import numpy as np

from vc_object_store import AfvSource
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_initial_fibers, fiber_volume
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
    cursor=AfvSource(source)
    metadata=cursor.metadata
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
    volume=config.get('volume') or fiber_volume(config['frame_template'])
    if volume != config['target_coordinate_space'].split('/')[-1]:
        raise ValueError('Target coordinate space mismatch')
    limits = config['z_range']
    namespace=uuid.UUID(cursor.uuid)
    counts=dict(candidates=0,selected=0)
    def rows():
        for fid,name,family,count,length,bbox,annotation,raw in cursor:
            lo,hi=mapped_bounds(np.asarray(bbox[:3]),np.asarray(bbox[3:]),matrix)
            if hi[2]<limits[0] or lo[2]>limits[1]:continue
            counts['candidates']+=1
            xyz=np.frombuffer(raw,dtype='<f8').reshape(-1,3)
            points=xyz @ matrix[:3,:3].T + matrix[:3,3]
            if points[:,2].max()<limits[0] or points[:,2].min()>limits[1]:continue
            counts['selected']+=1
            yield dict(id=str(uuid.uuid5(namespace,'fiber:'+str(fid))),name=name,family=family,
                       meta=annotation),points
    result=write_initial_fibers(root/'fibers.vcobj',rows(),volume=volume,
                                workers=int(config.get('workers',32)),packed=True)
    result.update(seconds=time.monotonic()-started,candidates=counts['candidates'],source_fibers=cursor.fiber_count,
                  source_uuid=cursor.uuid,transform_sha256=digest(config['transform']),z_range=limits)
    write(root/'import.json', result)
    print(json.dumps(result), flush=True)
    return result
