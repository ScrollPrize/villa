"""Pack fiber-on-patch runs (sheets/scripts/match_fibers_to_patches.py JSON lines) into the sheet-strip index.

Each run is a span of an AFV fiber that lies on one tifxyz patch: its fiber (AFV catalog index), patch, arc span on
the fiber and per-sample patch grid position (row, col) and position (patch voxels). Runs are split by height: a run
is validation when all its samples lie in the z band ``--val-z`` (patch voxels), training when all lie at least
``--val-margin`` outside it, else unused.

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.build_index --afv FIBERS.afv \\
        --runs H_runs.jsonl V_runs.jsonl --patches-dir PATCHES --out INDEX_DIR

Output (INDEX_DIR): ``runs_*.npy`` (one row per run, sorted by fiber then start), ``sample_*.npy`` (``run_samples``
gives each run's slice), ``patches.json`` (sheets/data/patches.PatchInfo rows) and ``meta.json``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from vesuvius.neural_tracing.tracers.shared.annotations.afv import AFVFibers
from vesuvius.neural_tracing.tracers.sheets import config as sheet_config
from vesuvius.neural_tracing.tracers.sheets.data.patches import build_index

SPLITS = dict(train=0, validation=1, unused=-1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--afv', required=True)
    ap.add_argument('--runs', nargs='+', required=True, help='match_fibers_to_patches.py outputs (any families)')
    ap.add_argument('--patches-dir', nargs='+', required=True, help='as given to match_fibers_to_patches.py')
    ap.add_argument('--grid-scale', type=float, default=4., help='AFV voxels per patch voxel (as for matching)')
    ap.add_argument('--val-z', type=float, nargs=2, default=(11250., 12125.),
                    help='validation z band, patch voxels (default: the fiber runs\' val_z 45000-48500 base voxels)')
    ap.add_argument('--val-margin', type=float, default=80.)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    index = build_index(args.patches_dir, sheet_config.OUTPUT/'cache')
    patch_id = {(info.root, info.name): i for i, info in enumerate(index)}
    fibers = AFVFibers(args.afv, split='all')
    catalog = {row[0]: i for i, row in enumerate(fibers.catalog)}
    rows, rc, xyz = [], [], []
    for path in args.runs:
        with open(path) as stream:
            for line in stream:
                run = json.loads(line)
                points = np.asarray(run['xyz'], np.float32)
                z = points[:, 2]
                lo, hi = args.val_z
                split = ('validation' if ((z >= lo) & (z <= hi)).all() else
                         'train' if ((z < lo-args.val_margin) | (z > hi+args.val_margin)).all() else 'unused')
                rows.append((catalog[run['fiber']], patch_id[(run['root'], run['patch'])], run['start'], run['end'],
                             run['offset'], SPLITS[split]))
                rc.append(np.asarray(run['rc'], np.int32))
                xyz.append(points)
    order = sorted(range(len(rows)), key=lambda i: (rows[i][0], rows[i][2]))
    args.out.mkdir(parents=True, exist_ok=True)
    columns = dict(fiber=np.int64, patch=np.int32, start=np.float32, end=np.float32, offset=np.float32, split=np.int8)
    for k, (name, dtype) in enumerate(columns.items()):
        np.save(args.out/f'runs_{name}.npy', np.array([rows[i][k] for i in order], dtype))
    counts = np.array([len(rc[i]) for i in order], np.int64)
    np.save(args.out/'run_samples.npy', np.concatenate(([0], np.cumsum(counts))))
    np.save(args.out/'sample_rc.npy', np.concatenate([rc[i] for i in order]))
    np.save(args.out/'sample_xyz.npy', np.concatenate([xyz[i] for i in order]))
    (args.out/'patches.json').write_text(json.dumps([asdict(info) for info in index]))
    split = np.array([rows[i][5] for i in order])
    length = np.array([rows[i][3]-rows[i][2] for i in order])
    summary = {name: dict(runs=int((split == value).sum()), length=float(length[split == value].sum()),
                          fibers=len({rows[i][0] for i in order if rows[i][5] == value}))
               for name, value in SPLITS.items()}
    meta = dict(afv=str(Path(args.afv).resolve()), afv_uuid=fibers.metadata.get('uuid'), runs=[str(Path(p).resolve()) for p in args.runs],
                patches_dir=[str(Path(p).resolve()) for p in args.patches_dir], afv_per_patch_voxel=args.grid_scale,
                val_z=list(args.val_z), val_margin=args.val_margin, splits=summary)
    (args.out/'meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
