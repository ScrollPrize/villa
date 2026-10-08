"""Build the fresh-location weight caches (data/location_weights.py) for every source of a dataset configuration.
Usage: python scripts/build_location_weights.py DATASET_CONFIG [--sources NAME ...] [--out DIR] [--workers 16] [--force]
The trainer reads them from output/location_weights (or the run option location_weights_dir)."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from vesuvius.neural_tracing.tracers.fibers.data.location_weights import CELL, STRIDE, VERSION
from vesuvius.neural_tracing.tracers.fibers.evaluation.audit import failure_scoring as fs
from vesuvius.neural_tracing.tracers.fibers.evaluation.audit.failure_scoring import smoothed


def bend_profile(points, s, stride=STRIDE):
    """(arclength positions every ``stride`` vox, bend score in degrees there) for one fiber."""
    if s[-1] < 2:
        return np.zeros(1), np.zeros(1)
    p, arcs = smoothed(points, s)
    d = np.gradient(p, axis=0)
    tangent = d/np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    at = np.arange(0., s[-1]+1e-9, stride)
    idx = lambda x: np.clip(np.searchsorted(arcs, x), 0, len(arcs)-1)
    angle = lambda a, b: np.degrees(np.arccos(np.clip((tangent[idx(a)]*tangent[idx(b)]).sum(1), -1, 1)))
    return at, np.maximum.reduce([angle(at-12, at+12), angle(at, at+32), angle(at, at-32)])


def _compression_task(task):
    name, point = task
    m = fs.compression(fs.volume(name), np.asarray(point))
    return m['sheetness'], m['sheet_modulation']


def build(source, document_path, fibers, out, workers=16):
    """Score every fiber of ``source`` (a dataset-config source entry; ``fibers`` its annotated fibers) and write the
    cache to ``out``."""
    names, starts, arcs, bends, positions = [], [0], [], [], []
    for f in fibers:
        at, bend = bend_profile(np.asarray(f.points), np.asarray(f.s))
        names.append(f.name)
        arcs.append(at)
        bends.append(bend)
        positions.append(np.stack([np.interp(at, f.s, f.points[:, k]) for k in range(3)], -1))
        starts.append(starts[-1]+len(at))
    arcs, bends, positions = np.concatenate(arcs), np.concatenate(bends), np.concatenate(positions)
    cells = np.floor(positions/CELL).astype(np.int64)
    _, first, cell_of = np.unique(cells, axis=0, return_index=True, return_inverse=True)
    print(f"{source['name']}: {len(names)} fibers, {len(arcs)} locations, {len(first)} occupied {CELL:g}-vox cells", flush=True)
    with ProcessPoolExecutor(workers, initializer=fs._init, initargs=(str(document_path),)) as pool:
        measured = np.asarray(list(pool.map(_compression_task, [(source['name'], positions[i]) for i in first], chunksize=16)))
    sheetness, modulation = measured[cell_of.ravel(), 0], measured[cell_of.ravel(), 1]
    pct = lambda v: np.searchsorted(np.sort(v), v)/len(v)
    compression = np.maximum(1-pct(sheetness), 1-pct(modulation))
    bent, compressed = bends >= np.percentile(bends, 90), compression >= np.percentile(compression, 90)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, names=np.asarray(names), starts=np.asarray(starts), arcs=arcs.astype(np.float32),
                        bend=bends.astype(np.float32), sheetness=sheetness.astype(np.float32),
                        modulation=modulation.astype(np.float32), bent=bent, compressed=compressed,
                        meta=json.dumps(dict(version=VERSION, stride=STRIDE, cell=CELL, source=source['name'],
                                             bend_p90=float(np.percentile(bends, 90)), cells=int(len(first)))))
    print(f"{source['name']}: bend p90 {np.percentile(bends, 90):.1f} deg; wrote {out}", flush=True)



def main(argv=None):
    from vesuvius.neural_tracing.tracers.fibers.data import location_weights
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import AFVFibers
    from vesuvius.neural_tracing.tracers.shared.annotations.vc3d import load_fibers
    from vesuvius.neural_tracing.tracers.fibers.data.datasets import read_dataset_config
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dataset_config')
    ap.add_argument('--sources', nargs='+')
    ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    document, _ = read_dataset_config(args.dataset_config)
    for source in document['sources']:
        if args.sources and source['name'] not in args.sources:
            continue
        out = location_weights.cache_path(source, args.out)
        if out.exists() and not args.force:
            print(f"{source['name']}: {out} exists", flush=True)
            continue
        if source['kind'] == 'paris4':
            fibers = load_fibers(source['fibers'], grid_scale=source['grid_scale'])
        else:
            collection = AFVFibers(source['path'], source['grid_scale'], split='all', sha256=source.get('sha256'))
            fibers = [collection.geometry(i) for i in range(len(collection))]
        build(source, Path(args.dataset_config).resolve(), fibers, out, args.workers)


if __name__ == '__main__':
    main()
