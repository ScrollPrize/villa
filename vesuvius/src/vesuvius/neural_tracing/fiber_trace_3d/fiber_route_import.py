"""Import native route pages without duplicating component index records."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import time
import uuid

import msgpack
import numpy as np
from numcodecs import Zstd

from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_initial_fibers, fiber_volume
from .fiber_min_flow_experiment import write


def read_packed(path):
    return msgpack.unpackb(Zstd().decode(Path(path).read_bytes()), raw=False)


def route_rows(path, generation, z_range):
    data = read_packed(path)
    if data['generation'] != generation:
        raise ValueError(f'Route generation mismatch: {path}')
    rows = []
    bounds = []
    for index, fiber in enumerate(data['fibers']):
        xyz = np.asarray(fiber['points_base'], dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
            raise ValueError(f'Invalid route geometry: {path}:{index}')
        lo, hi = xyz[:, 2].min(), xyz[:, 2].max()
        bounds.append((lo, hi))
        if hi < z_range[0] or lo > z_range[1]:
            continue
        length = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
        if not np.isclose(length, fiber['length_base'], rtol=1e-8, atol=1e-6):
            raise ValueError(f'Route length mismatch: {path}:{index}')
        name = f"component_{data['component']}_page_{data['page']}_route_{index}"
        provenance = {k: v for k, v in fiber.items() if k != 'points_base'}
        rows.append((dict(name=name, family='unknown', meta=provenance), xyz))
    return rows, bounds


def run(config):
    started = time.monotonic()
    source = Path(config['routes'])
    root = Path(config['output_root']); root.mkdir(parents=True, exist_ok=True)
    manifest = read_packed(source/'routes.manifest.msgpack.zst')
    if manifest['format'] != 'vc-fiber-routes-v4-reduced':
        raise ValueError('Unsupported route format')
    paths = sorted((p for p in source.iterdir() if re.fullmatch(r'component_\d+\.routes\.\d+\.msgpack\.zst', p.name)),
                   key=lambda p: tuple(map(int, re.findall(r'\d+', p.name))))
    rows = []; bounds = []
    # Fixed-size groups bound outstanding file reads and preserve source order.
    with ThreadPoolExecutor(max_workers=int(config.get('workers', 32))) as pool:
        for offset in range(0, len(paths), 2048):
            batch = paths[offset:offset+2048]
            for selected, extent in pool.map(lambda p: route_rows(p, manifest['generation'], config['z_range']), batch):
                rows.extend(selected); bounds.extend(extent)
            print(json.dumps(dict(stage='route_import', pages=min(offset+2048,len(paths)), total_pages=len(paths),
                                  selected=len(rows), seconds=time.monotonic()-started)), flush=True)
    if not rows:
        raise ValueError(f'No routes intersect requested z range; source bounds={np.asarray(bounds).min(0), np.asarray(bounds).max(0)}')
    # Stable identities are independent of crop/output path.
    namespace=uuid.uuid5(uuid.NAMESPACE_URL,str(manifest['generation']))
    for row,_ in rows: row['id']=str(uuid.uuid5(namespace,row['name']))
    volume=config.get('volume') or fiber_volume(config['frame_template'])
    result=write_initial_fibers(root/'fibers.vcobj',rows,volume=volume,
                                workers=int(config.get('workers',32)),packed=True)
    result.update(source_routes=len(bounds), z_range=config['z_range'],
                  source_z_bounds=[float(np.min(bounds)), float(np.max(bounds))], seconds=time.monotonic()-started)
    write(root/'import.json', result)
    print(json.dumps(result), flush=True)
    return result
