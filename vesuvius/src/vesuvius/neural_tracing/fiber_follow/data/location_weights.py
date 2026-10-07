"""Fresh-location weights that sample heavily bent fibers and compressed sheets more often (but not predominantly).

Per source, a cache (``build`` / scripts/build_location_weights.py) holds for every annotated fiber, every ``STRIDE``
vox of arclength:
  bend         the largest of: the fiber's bend over +-12 vox and its bend over the next 32 vox in either direction
               (the annotation smoothed at sigma 4 vox, evaluation.failure_scoring.smoothed), in degrees
  compression  sheetness and sheet modulation (evaluation.failure_scoring.compression) at one fiber point per occupied
               ``CELL``-vox grid cell (compression varies slowly; this keeps CT reads to a few thousand per source)
Flags (per source): bend at or above its p90; compression score max(1 - pct(sheetness), 1 - pct(modulation)) at or
above its p90, so each condition marks 10% of locations, as in evaluation.failure_scoring. A location's weight is
1 + bend_boost * bent + compression_boost * compressed; fresh windows and positions within them are drawn in proportion
(FollowDataset.fresh_location). With boosts of 1 each condition rises from ~10% to ~17-18% of fresh crops.
"""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STRIDE = 8.
CELL = 32.
VERSION = 1
DEFAULT_DIR = Path(__file__).resolve().parents[1]/'output'/'location_weights'


def cache_path(source, directory=None):
    """The cache file of a dataset source: its name and the identity of its annotations."""
    if source['kind'] == 'paris4':
        identity = hashlib.sha256(str(Path(source['fibers']).resolve()).encode()).hexdigest()[:12]
    else:
        identity = str(source['sha256'])[:12]
    return Path(directory or DEFAULT_DIR)/f"{source['name']}_{identity}_v{VERSION}.npz"


def bend_profile(points, s, stride=STRIDE):
    """(arclength positions every ``stride`` vox, bend score in degrees there) for one fiber."""
    from vesuvius.neural_tracing.fiber_follow.evaluation.failure_scoring import smoothed
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
    from vesuvius.neural_tracing.fiber_follow.evaluation import failure_scoring as fs
    name, point = task
    m = fs.compression(fs.volume(name), np.asarray(point))
    return m['sheetness'], m['sheet_modulation']


def build(source, document_path, fibers, out, workers=16):
    """Score every fiber of ``source`` (a dataset-config source entry; ``fibers`` its annotated fibers) and write the
    cache to ``out``."""
    from vesuvius.neural_tracing.fiber_follow.evaluation import failure_scoring as fs
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


def load(path, fibers, bend_boost=1., compression_boost=1.):
    """Per fiber (aligned with ``fibers`` by name): (arclength positions, location weights), or None for a fiber the
    cache does not hold (uniform)."""
    with np.load(path) as data:  # read each array once: an npz decompresses on every access
        names, starts, arcs = data['names'], data['starts'], data['arcs'].astype(np.float64)
        weight = 1.+bend_boost*data['bent']+compression_boost*data['compressed']
    index = {str(n): i for i, n in enumerate(names)}
    wanted = [row[1] for row in fibers.catalog] if hasattr(fibers, 'catalog') else [f.name for f in fibers]
    out = []
    for name in wanted:
        i = index.get(str(name))
        out.append(None if i is None else (arcs[starts[i]:starts[i+1]], weight[starts[i]:starts[i+1]]))
    return out
