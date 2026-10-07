"""Match AFV fibers of one family (e.g. horizontal) to the tifxyz patches whose surface they lie on.

Every valid patch node gets a surface normal from its grid neighbours; all nodes go into one KD-tree. Each fiber
(``AFVFibers``, plus VC3D fiber JSONs with ``--fiber-jsons``; coordinates divided by ``--grid-scale`` into the
patches' level-0 voxels) is sampled every ``--sample`` voxels; a sample lies on a patch when one of its
``--candidates`` nearest nodes of that patch is within ``--max-tangential`` grid cells of it sideways and, along the
node's normal turned outward (away from the scroll's umbilicus at that height), between ``--max-inward`` voxels
inward and ``--max-outward`` voxels outward of it. A fiber is a thin tube on its sheet's surface, so it touches the
patch; the next sheet is about 12 voxels away. (On the Paris4 AFV, matched horizontal fibers sit at +0.1 voxels on
average with a +-2 window.) Runs of consecutive on-patch samples at least ``--min-run`` voxels long are written, one
JSON line each: fiber id, patch (root, name), the run's arc-length start and end on the fiber (voxels), mean outward
offset, and each sample's patch grid position (row, col of its node) and position (xyz). A summary of how many
patches have runs long enough for each front size follows.

  python scripts/match_fibers_to_patches.py --afv PARIS4.afv --grid-scale 4 --patches-dir P1 P2 \\
      --umbilicus umbilicus.json --out matches.jsonl
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.tracers.shared.annotations.afv import AFVFibers
from vesuvius.neural_tracing.tracers.sheets import config as C
from vesuvius.neural_tracing.tracers.sheets.data.patches import build_index

sys.path.insert(0, str(Path(__file__).parent))
from trace_from_fibers import fiber_tag  # noqa: E402  (a sibling script)

_STATE = {}  # node arrays and tree, shared with forked workers


def patch_nodes(info) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valid nodes (N, 3), their unit normals (N, 3) and grid (row, col) (N, 2) of one patch."""
    xyz = np.stack([np.asarray(tifffile.memmap(info.path/f'{c}.tif'), np.float32) for c in 'xyz'], -1)
    if min(xyz.shape[:2]) < 2:  # a single row or column has no surface normal
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int32)
    ok = np.all(xyz != -1, -1) & np.all(np.isfinite(xyz), -1)
    dr, dc = np.gradient(np.where(ok[..., None], xyz, np.nan), axis=(0, 1))
    normal = np.cross(dc, dr)
    norm = np.linalg.norm(normal, axis=-1)
    good = ok & np.isfinite(norm) & (norm > 1e-6)
    rows, cols = np.nonzero(good)
    return xyz[good], (normal[good]/norm[good][:, None]).astype(np.float32), np.stack([rows, cols], 1).astype(np.int32)


def match_fiber(job) -> list[dict]:
    """``job``: an AFV catalog index (geometry read in this process, on its own connection), or (fiber id, points)."""
    s = _STATE
    if isinstance(job, tuple):
        fiber_id, points = job
    else:
        fiber_id, points = s['fibers'].catalog[job][0], s['fibers'].geometry(job).points
    arc = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    at = np.arange(0., arc[-1], s['sample'])
    samples = np.stack([np.interp(at, arc, points[:, k]) for k in range(3)], 1)
    dist, idx = s['tree'].query(samples, k=s['candidates'], distance_upper_bound=s['reach'])
    found = np.isfinite(dist)
    idx = np.where(found, idx, 0)
    delta = samples[:, None]-s['pos'][idx]
    offset = np.einsum('mkd,mkd->mk', delta, s['nrm'][idx])  # > 0: the fiber is outward of the patch
    side = np.linalg.norm(delta-offset[..., None]*s['nrm'][idx], axis=-1)
    patch = s['patch'][idx]
    ok = (found & (offset >= -s['max_inward']) & (offset <= s['max_outward'])
          & (side <= s['max_tangential']*s['cell'][patch]))
    out = []
    for p in np.unique(patch[ok]):
        hit = ok & (patch == p)
        on = hit.any(1)
        best = np.where(hit, side, np.inf).argmin(1)  # this patch's nearest qualifying node per sample
        edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(np.int8), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            if (b-a-1)*s['sample'] < s['min_run']:
                continue
            node = idx[np.arange(a, b), best[a:b]]
            out.append(dict(fiber=fiber_id if isinstance(fiber_id, str) else int(fiber_id), root=s['roots'][p], patch=s['names'][p], start=float(at[a]),
                            end=float(at[b-1]), offset=float(offset[np.arange(a, b), best[a:b]].mean()),
                            rc=s['rc'][node].tolist(), xyz=np.round(samples[a:b], 2).tolist()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--afv', required=True)
    ap.add_argument('--grid-scale', type=float, default=4., help='AFV voxels per patch level-0 voxel')
    ap.add_argument('--patches-dir', nargs='+', required=True)
    ap.add_argument('--fiber-jsons', help='also a directory of VC3D fiber JSONs (line_points in AFV voxels)')
    ap.add_argument('--family', default='H')
    ap.add_argument('--sample', type=float, default=5.)
    ap.add_argument('--candidates', type=int, default=32)
    ap.add_argument('--umbilicus', required=True, help='umbilicus.json (control points x, y, z in patch voxels)')
    ap.add_argument('--max-inward', type=float, default=2., help='voxels the fiber may lie inward of the patch surface')
    ap.add_argument('--max-outward', type=float, default=2., help='voxels it may lie outward of it')
    ap.add_argument('--max-tangential', type=float, default=.75, help='grid cells')
    ap.add_argument('--min-run', type=float, default=240.)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--workers', type=int, default=48)
    args = ap.parse_args(argv)
    t0 = time.time()
    index = build_index(args.patches_dir, C.OUTPUT/'cache')
    pos, nrm, rc, patch = [], [], [], []
    for i, info in enumerate(index):
        p, n, r = patch_nodes(info)
        pos.append(p), nrm.append(n), rc.append(r), patch.append(np.full(len(p), i, np.int32))
    pos, nrm, rc, patch = (np.concatenate(a) for a in (pos, nrm, rc, patch))
    cp = sorted(json.loads(Path(args.umbilicus).read_text())['control_points'], key=lambda c: c['z'])
    zs = np.array([c['z'] for c in cp], np.float64)
    axis_xy = np.stack([np.interp(pos[:, 2], zs, [c[k] for c in cp]) for k in 'xy'], 1)
    radial = pos[:, :2]-axis_xy
    nrm *= np.where((nrm[:, :2]*radial).sum(1) < 0, -1., 1.)[:, None].astype(np.float32)  # outward normals
    cells = np.array([info.cell for info in index], np.float32)
    print(f'{len(index)} patches, {len(pos)} nodes ({time.time()-t0:.0f} s)', flush=True)
    tree = cKDTree(pos)
    print(f'tree built ({time.time()-t0:.0f} s)', flush=True)
    _STATE.update(tree=tree, pos=pos, nrm=nrm, rc=rc, patch=patch, cell=cells, sample=args.sample,
                  candidates=args.candidates, max_inward=args.max_inward, max_outward=args.max_outward,
                  max_tangential=args.max_tangential, min_run=args.min_run,
                  reach=float(cells.max()*args.max_tangential+max(args.max_inward, args.max_outward)),
                  roots=[info.root for info in index], names=[info.name for info in index])
    fibers = AFVFibers(args.afv, grid_scale=args.grid_scale, split='all')
    chosen = [i for i, row in enumerate(fibers.catalog) if row[2] == args.family]
    print(f'{len(chosen)} {args.family} fibers of {len(fibers)}', flush=True)
    _STATE['fibers'] = fibers
    jobs = list(chosen)
    for path in sorted(Path(args.fiber_jsons).glob('*.json')) if args.fiber_jsons else []:
        fiber = json.loads(path.read_text())
        if fiber_tag(fiber) == args.family:  # VC3D fiber annotations: id 'json:<file stem>'
            jobs.append((f'json:{path.stem}', np.asarray(fiber['line_points'], np.float64)/args.grid_scale))
    print(f'{len(jobs)-len(chosen)} {args.family} fibers from {args.fiber_jsons}', flush=True)
    chosen = jobs
    runs, done = 0, 0
    per_patch = {}
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(args.workers) as pool, args.out.open('w') as fh:
        for result in pool.imap_unordered(match_fiber, chosen, chunksize=64):
            done += 1
            for r in result:
                fh.write(json.dumps(r)+'\n')
                runs += 1
                key = (r['root'], r['patch'])
                per_patch[key] = max(per_patch.get(key, 0.), r['end']-r['start'])
            if done % 5000 == 0:
                print(f'{done}/{len(chosen)} fibers, {runs} runs on {len(per_patch)} patches ({time.time()-t0:.0f} s)',
                      flush=True)
    longest = np.array(list(per_patch.values()))
    summary = dict(patches=len(index), fibers=len(chosen), runs=runs, patches_matched=len(per_patch),
                   patches_with_run_for_points={n: int((longest >= (n-1)*60.).sum()) for n in range(5, 17)},
                   seconds=round(time.time()-t0))
    args.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
