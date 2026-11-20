"""Strips (an infer_all or split.py run) as surface patches for the spiral fit.

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.fit_patches RUN PATCHES \\
        [--prefix strip-net-] [--weight 1.0] [--min-nodes 500] [--stride 2] [--min-strided-nodes 16] \\
        [--umbilicus UMBILICUS.json --core-radius 230 --core-weight-scale 0.2]

Every strip RUN/strips/BUCKET/fiber_ID with at least --min-nodes kept nodes becomes the tifxyz PATCHES/<prefix>ID:
every --stride-th row and column of x/y/z/expected_error.tif (strips come at about 4x5 patch-voxel quads, far finer
than the fit needs; meta ``scale`` is divided by the stride so ``area_vx2`` stays consistent), trace.json, and
meta.json with ``loss_weight_scale`` = --weight. With --umbilicus (control points in patch voxels), a strip whose
median radius from the umbilicus is under --core-radius patch voxels gets ``loss_weight_scale`` = --core-weight-scale
(and ``inner_core_radius``): strips are least reliable in the tightly wound core. A strided strip with fewer than
--min-strided-nodes nodes or fewer than 2 rows or columns is not written. The radius uses the full-resolution strip.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile


def umbilicus_xy(path):
    """z -> (x, y) of an umbilicus.json (``control_points`` with x, y, z), linear between its points."""
    points = json.load(open(path))['control_points']
    z, x, y = (np.array([p[k] for p in points]) for k in 'zxy')
    order = np.argsort(z)
    return lambda at: (np.interp(at, z[order], x[order]), np.interp(at, z[order], y[order]))


def median_radius(strip, centre):
    """Median distance of a strip's kept nodes from the umbilicus at their z."""
    z = tifffile.imread(strip / 'z.tif')
    keep = z != -1
    cx, cy = centre(z[keep])
    return float(np.median(np.hypot(tifffile.imread(strip / 'x.tif')[keep] - cx,
                                    tifffile.imread(strip / 'y.tif')[keep] - cy)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=Path)
    ap.add_argument('patches', type=Path)
    ap.add_argument('--prefix', default='strip-net-')
    ap.add_argument('--weight', type=float, default=1.0)
    ap.add_argument('--min-nodes', type=int, default=500, help='skip strips with fewer kept nodes')
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--min-strided-nodes', type=int, default=16)
    ap.add_argument('--umbilicus', help='umbilicus.json in patch voxels; enables the core weight')
    ap.add_argument('--core-radius', type=float, default=230.0)
    ap.add_argument('--core-weight-scale', type=float, default=0.2)
    a = ap.parse_args(argv)
    if a.patches.exists():
        raise SystemExit(f'{a.patches} exists')
    centre = umbilicus_xy(a.umbilicus) if a.umbilicus else None
    a.patches.mkdir(parents=True)
    s = a.stride
    written = short = core = dropped = 0
    for strip in sorted(Path(a.run, 'strips').glob('*/fiber_*')):
        if not (strip / 'meta.json').is_file() or not (strip / 'z.tif').is_file():
            continue
        z = tifffile.imread(strip / 'z.tif')
        if int(np.sum(z != -1)) < a.min_nodes:
            short += 1
            continue
        meta = json.loads((strip / 'meta.json').read_text())
        meta['loss_weight_scale'] = a.weight
        meta['source_strip'] = str(strip)
        if centre is not None:
            radius = median_radius(strip, centre)
            if radius < a.core_radius:
                meta['loss_weight_scale'] = a.core_weight_scale
                meta['inner_core_radius'] = radius
                core += 1
        strided = z[::s, ::s]
        if int((strided != -1).sum()) < a.min_strided_nodes or min(strided.shape) < 2:
            dropped += 1
            continue
        dest = a.patches / f'{a.prefix}{strip.name[len("fiber_"):]}'
        dest.mkdir()
        for name in ('x.tif', 'y.tif', 'z.tif', 'expected_error.tif'):
            if (strip / name).is_file():
                tifffile.imwrite(dest / name, np.ascontiguousarray(tifffile.imread(strip / name)[::s, ::s]))
        if (strip / 'trace.json').is_file():
            shutil.copy2(strip / 'trace.json', dest / 'trace.json')
        meta['scale'] = [v / s for v in meta['scale']]
        meta['grid_stride'] = s * int(meta.get('grid_stride', 1))
        (dest / 'meta.json').write_text(json.dumps(meta, indent=1))
        written += 1
    print(f'{a.patches}: {written} patches as {a.prefix}* (weight {a.weight}, {core} core-weighted, stride {s}); '
          f'skipped {short} with < {a.min_nodes} nodes, {dropped} with < {a.min_strided_nodes} strided nodes')


if __name__ == '__main__':
    main()
