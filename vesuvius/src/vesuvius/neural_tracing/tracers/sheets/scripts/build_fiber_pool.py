"""Every fiber of a scroll as point samples for the model's fiber inputs (``data.fiber_pool``): a directory of .npy
arrays, memory-mapped by every loader worker (``data/fibers.py``). Per point, every ``--sample`` voxels along each
fiber: ``points`` (M, 3) float32 in patch level-0 voxels, ``tangent`` (M, 3) unit direction along the fiber, ``arc``
(M,) float32 arc length from the fiber's start, ``instance`` (M,) int32 index into ``keys`` (one per fiber:
``afv:<id>`` or ``json:<file stem>``, the ids ``match_fibers_to_patches.py`` writes) and ``family`` (M,) int8 (0
horizontal, 1 vertical); points are sorted by ``CELL``-voxel cell, ``cell_keys`` / ``cell_starts`` index them.

AFV fibers of any family (``AFVFibers``, coordinates divided by ``--grid-scale``), plus VC3D fiber JSONs with
``--fiber-jsons``. An AFV fiber running along a JSON fiber (at least ``--dedup-fraction`` of its samples within
``--dedup-distance`` voxels of it) is the same physical fiber and is left out, so no fiber appears as two instances;
``aliases.json`` maps it to that JSON fiber.

  python scripts/build_fiber_pool.py --afv PARIS4.afv --grid-scale 4 --fiber-jsons FIBERS_DIR --out POOL_DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.tracers.fibers.data.afv import AFVFibers
from vesuvius.neural_tracing.tracers.sheets.data.fibers import CELL, cell_key

FAMILY = {'H': 0, 'V': 1}


def resample(points: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points every ``step`` voxels of arc length along a polyline, unit tangents and arc lengths there."""
    arc = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    at = np.arange(0., arc[-1]+1e-9, step)
    pts = np.stack([np.interp(at, arc, points[:, k]) for k in range(3)], 1)
    tangent = np.gradient(pts, axis=0) if len(pts) > 1 else np.zeros_like(pts)
    return pts, tangent/np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9), at


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--afv', required=True)
    ap.add_argument('--grid-scale', type=float, default=4.)
    ap.add_argument('--fiber-jsons')
    ap.add_argument('--sample', type=float, default=5.)
    ap.add_argument('--dedup-distance', type=float, default=4.)
    ap.add_argument('--dedup-fraction', type=float, default=.5)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    fibers = []  # (key, family, points, tangents)
    for path in sorted(Path(args.fiber_jsons).glob('*.json')) if args.fiber_jsons else []:
        fiber = json.loads(path.read_text())
        hv = fiber.get('hv_classification') or {}
        tag = hv.get('manual_tag') or hv.get('automatic_tag') or ''
        if tag in FAMILY and len(fiber.get('line_points') or []) > 1:
            fibers.append((f'json:{path.stem}', FAMILY[tag],
                           *resample(np.asarray(fiber['line_points'], np.float64)/args.grid_scale, args.sample)))
    print(f'{len(fibers)} JSON fibers', flush=True)
    json_tree = cKDTree(np.concatenate([f[2] for f in fibers])) if fibers else None
    owner = np.concatenate([np.full(len(f[2]), k) for k, f in enumerate(fibers)]) if fibers else None
    afv = AFVFibers(args.afv, grid_scale=args.grid_scale, split='all')
    aliases = {}  # dropped AFV fiber -> the JSON fiber it duplicates (fiber runs may name either)
    for i, row in enumerate(afv.catalog):
        pts, tan, arc = resample(afv.geometry(i).points, args.sample)
        if json_tree is not None:
            near, j = json_tree.query(pts, distance_upper_bound=args.dedup_distance)
            hit = np.isfinite(near)
            if hit.mean() >= args.dedup_fraction:
                aliases[f'afv:{row[0]}'] = fibers[int(np.bincount(owner[j[hit]]).argmax())][0]
                continue
        fibers.append((f'afv:{row[0]}', FAMILY.get(row[2], 0), pts, tan, arc))
    print(f'{len(afv.catalog)} AFV fibers, {len(aliases)} left out as duplicates of JSON fibers', flush=True)
    points = np.concatenate([f[2] for f in fibers]).astype(np.float32)
    keys = cell_key(points)
    order = np.argsort(keys, kind='stable')
    cells, starts = np.unique(keys[order], return_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    arrays = dict(points=points[order], tangent=np.concatenate([f[3] for f in fibers]).astype(np.float32)[order],
                  arc=np.concatenate([f[4] for f in fibers]).astype(np.float32)[order],
                  instance=np.concatenate([np.full(len(f[2]), k, np.int32) for k, f in enumerate(fibers)])[order],
                  family=np.concatenate([np.full(len(f[2]), f[1], np.int8) for f in fibers])[order],
                  cell_keys=cells, cell_starts=np.append(starts, len(points)).astype(np.int64))
    for name, a in arrays.items():
        np.save(args.out/f'{name}.npy', a)
    (args.out/'keys.json').write_text(json.dumps([f[0] for f in fibers]))
    (args.out/'aliases.json').write_text(json.dumps(aliases))
    (args.out/'meta.json').write_text(json.dumps(dict(cell=CELL, sample=args.sample, grid_scale=args.grid_scale,
                                                      afv=str(args.afv), fiber_jsons=args.fiber_jsons)))
    print(f'{len(fibers)} fibers, {len(points)} points -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
