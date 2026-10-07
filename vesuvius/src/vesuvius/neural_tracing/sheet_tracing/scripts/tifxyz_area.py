"""Add ``area_vx2`` and ``area_cm2`` (the meta.json keys VC3D reads) to existing tifxyz folders, e.g. traces written
before the tracer recorded them. Area counts the quads with four valid corners (``infer.surface_area_vx2``).

  python scripts/tifxyz_area.py --voxel-size 9.6 TIFXYZ_DIR [...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

from vesuvius.neural_tracing.sheet_tracing.infer import add_area


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+', type=Path)
    ap.add_argument('--voxel-size', type=float, required=True, help='level-0 voxel size in micrometres (s1_ds2: 9.6)')
    args = ap.parse_args(argv)
    for d in args.dirs:
        xyz = np.stack([tifffile.imread(d/f'{c}.tif') for c in 'xyz'], -1)
        meta = add_area(d/'meta.json', xyz, args.voxel_size)
        print(f"{d.name}: {meta['area_vx2']:.0f} vx^2 = {meta['area_cm2']:.4f} cm^2", flush=True)


if __name__ == '__main__':
    main()
