"""Import completed stream-tracer tiles as whole, independently classified fibers."""
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import json
from pathlib import Path
import time
import uuid
import numpy as np
from .fiber_route_import import read_packed
from .fiber_min_flow_experiment import write
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_initial_fibers, fiber_volume
from vesuvius.neural_tracing.tracers.shared.annotations.open_data import open_data_volume


def tile_rows(request):
    path,z_range=request;data=read_packed(path)
    if data.get('version')!=2 or data.get('attribute_schema_version')!=1 or data.get('coordinate_units')!='base_voxels':
        raise ValueError('Unsupported stream tile: '+str(path))
    rows=[];ids=[]
    for piece in data['pieces']:
        fid=int(piece['id']);ids.append(fid)
        xyz=np.asarray(piece['points_base'],dtype=np.float64)
        if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<2 or not np.isfinite(xyz).all():
            raise ValueError(f'Invalid stream geometry: {path}:{fid}')
        if z_range is not None and (xyz[:,2].max()<z_range[0] or xyz[:,2].min()>z_range[1]):continue
        provenance={k:piece[k] for k in ('id','links','reasons','seed_base','seed_direction','fiber_attributes')}
        rows.append((dict(id=fid,name=f'stream_{fid}',family='unknown',meta=provenance),xyz))
    return ids,rows


def run(config):
    started=time.monotonic();source=Path(config['tiles']);root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source.parent/'run.json').read_text());summary=json.loads((source.parent/'summary.json').read_text())
    if not summary.get('complete') or manifest.get('output_schema')!='fiber-stream-attributes-v2' or manifest.get('coordinates')!='base voxels':
        raise ValueError('Require completed native-base stream output')
    volume=open_data_volume(manifest['config']['stream']['ct_locator'])
    expected=config.get('volume') or fiber_volume(config['frame_template'])
    if not volume or expected!=volume[1]:raise ValueError('Coordinate frame mismatch')
    namespace=uuid.uuid5(uuid.NAMESPACE_URL,manifest['identity'])
    paths=sorted(source.glob('*.msgpack.zst'),key=lambda p:tuple(map(int,p.name.split('.')[0].split('_'))))
    if len(paths)!=summary['finished_tiles'] or summary['finished_tiles']!=summary['total_tiles']:raise ValueError('Incomplete tile set')
    seen=set();selected=0;workers=int(config.get('workers',16))
    def rows():
        nonlocal selected
        with ProcessPoolExecutor(max_workers=workers) as pool:
            stream=iter(paths);pending=deque()
            for _ in range(workers*2):
                path=next(stream,None)
                if path is not None:pending.append(pool.submit(tile_rows,(str(path),config.get('z_range'))))
            done=0
            while pending:
                ids,items=pending.popleft().result()
                if len(set(ids))!=len(ids) or seen.intersection(ids):raise ValueError('Duplicate stream piece ID')
                seen.update(ids);selected+=len(items);done+=1
                for row,xyz in items:
                    row['id']=str(uuid.uuid5(namespace,'fiber:'+str(row['id'])))
                    yield row,xyz
                path=next(stream,None)
                if path is not None:pending.append(pool.submit(tile_rows,(str(path),config.get('z_range'))))
                if done%1000==0:print(json.dumps(dict(stage='stream_import',tiles=done,total_tiles=len(paths),source_fibers=len(seen),selected=selected,seconds=time.monotonic()-started)),flush=True)
        if len(seen)!=summary['pieces']:raise ValueError('Source piece count mismatch')
    result=write_initial_fibers(root/'fibers.vcobj',rows(),volume=volume[1],workers=workers,packed=True)
    result.update(source_fibers=len(seen),z_range=config.get('z_range'),whole_fibers=True,seconds=time.monotonic()-started)
    write(root/'import.json',result);print(json.dumps(result),flush=True)
    return result
