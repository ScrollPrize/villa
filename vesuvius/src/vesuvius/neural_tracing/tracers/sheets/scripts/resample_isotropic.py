"""Copy tifxyz patches whose grid spacing differs between rows and columns onto a square grid.

The patch index (``data/patches.py``) gives each patch one cell size, so a grid stepping e.g. 19.7 vx between rows but
14.8 vx between columns would be sampled stretched along the columns. Each patch here is linearly resampled along
its columns to its median row spacing (a new node is valid when both nodes it interpolates are), or along its rows
to its column spacing when that is the larger one; ``meta.json`` ``scale`` becomes 1 / that spacing. Patches already
square within ``--tolerance`` are copied unchanged.

  python scripts/resample_isotropic.py SRC_DIR OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile


def steps(xyz: np.ndarray, ok: np.ndarray) -> tuple[float, float]:
    """Median distance (voxels) between valid row neighbours and between valid column neighbours."""
    dr = np.linalg.norm(xyz[1:]-xyz[:-1], axis=-1)[ok[1:] & ok[:-1]]
    dc = np.linalg.norm(xyz[:, 1:]-xyz[:, :-1], axis=-1)[ok[:, 1:] & ok[:, :-1]]
    return float(np.median(dr)), float(np.median(dc))


def resample_columns(xyz: np.ndarray, ok: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Columns resampled so that ``factor`` old columns become one new column step."""
    cols = xyz.shape[1]
    pos = np.arange(int(np.floor((cols-1)/factor+1e-6))+1)*factor  # never past the last column
    j0 = np.clip(np.floor(pos).astype(int), 0, cols-2)
    f = (pos-j0)[None, :, None]
    out = (1-f)*xyz[:, j0]+f*xyz[:, j0+1]
    valid = ok[:, j0] & ok[:, j0+1]
    return np.where(valid[..., None], out, -1).astype(np.float32), valid


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--tolerance', type=float, default=.05, help='relative row/column spacing difference left as is')
    args = ap.parse_args(argv)
    src, out = Path(args.src), Path(args.out)
    for patch in sorted(p for p in src.iterdir() if (p/'x.tif').is_file()):
        meta = json.loads((patch/'meta.json').read_text())
        xyz = np.stack([np.asarray(tifffile.imread(patch/f'{c}.tif'), np.float32) for c in 'xyz'], -1)
        ok = np.all(xyz != -1, -1) & np.all(np.isfinite(xyz), -1)
        row, col = steps(xyz, ok)
        dst = out/patch.name
        if abs(col/row-1) <= args.tolerance:
            shutil.copytree(patch, dst, dirs_exist_ok=True)
            print(f'{patch.name}: square ({row:.1f} x {col:.1f} vx), copied', flush=True)
            continue
        if row > col:  # finer columns: resample them to the row spacing
            xyz, ok = resample_columns(xyz, ok, row/col)
        else:  # finer rows: the same along the other axis
            xyz, ok = resample_columns(xyz.transpose(1, 0, 2), ok.T, col/row)
            xyz, ok = xyz.transpose(1, 0, 2), ok.T
        cell = max(row, col)
        dst.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate('xyz'):
            tifffile.imwrite(dst/f'{c}.tif', np.ascontiguousarray(xyz[..., i]))
        meta['scale'] = [1./cell, 1./cell]
        (dst/'meta.json').write_text(json.dumps(meta))
        r2, c2 = steps(xyz, ok)
        print(f'{patch.name}: {row:.1f} x {col:.1f} vx -> {r2:.1f} x {c2:.1f} vx, {xyz.shape[0]} x {xyz.shape[1]} '
              f'nodes, {int(ok.sum())} valid', flush=True)


if __name__ == '__main__':
    main()
