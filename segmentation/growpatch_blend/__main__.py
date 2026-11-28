"""Refine a GrowPatch tifxyz segment toward a second trace of the same surface prediction.

    python -m segmentation.growpatch_blend SEGMENT_DIR PREDICTION_ZARR [--out NEW_SEGMENT_DIR] [--workers N]

SEGMENT_DIR is a tifxyz segment (for example a volpkg `paths/<id>` folder written by vc_grow_seg_from_seed).
PREDICTION_ZARR is the level-0 array of the surface prediction the segment was grown on (local path, http or s3),
in the same voxel grid as the segment. Only the box around the segment (plus a margin) is read.

The result is a new tifxyz segment next to the input (or at --out) on a grid 4x denser than an 8-voxel GrowPatch
grid, with the per-vertex move in `move.npy` and a summary in `blend_report.json`.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from vc_reader import tifxyz
from vc_reader.reader.zarr_reader import ZarrReader

from .blend import SPACING, tiled_blend
from .geometry import RADIUS

MARGIN = int(RADIUS * 1.75) + 24   # tile charts reach this far past the mesh


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m segmentation.growpatch_blend", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("segment", type=Path, help="input tifxyz segment folder")
    p.add_argument("prediction", help="surface-prediction Zarr array (level 0), local path or URL")
    p.add_argument("--out", type=Path, default=None, help="output segment folder (default: <segment>_blend next to it)")
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args(argv)

    surface = tifxyz.read(a.segment)
    xyz = np.array(surface.xyz, dtype=np.float64)
    valid = np.array(surface.valid, dtype=bool)
    pts = xyz[valid]
    lo = np.maximum(np.floor(pts.min(0) - MARGIN).astype(int), 0)
    hi = np.ceil(pts.max(0) + MARGIN).astype(int) + 1
    started = time.time()
    with ZarrReader.open(a.prediction) as arr:
        shape_xyz = np.asarray(arr.shape[::-1][:3])
        hi = np.minimum(hi, shape_xyz)
        cube = np.ascontiguousarray(arr[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]])
    print(f"read prediction box {lo.tolist()}..{hi.tolist()} ({cube.nbytes / 2**20:.0f} MiB) in {time.time() - started:.0f}s", flush=True)

    out, ok, move, report = tiled_blend(xyz, valid, cube, lo.astype(float), SPACING, workers=a.workers)
    dest = a.out or a.segment.with_name(f"{a.segment.name}_blend")
    scale = float(surface.scale[0]) * report["upsample"]
    grid = np.where(ok[..., None], out, -1.0).astype(np.float32)
    result = tifxyz.Surface(grid, mask=ok, scale=(scale, scale),
                            extra={"name": dest.name, "source": "growpatch_blend", "parent": surface.uuid})
    tifxyz.write(dest, result)
    np.save(dest / "move.npy", move.astype(np.float32))
    report.update(input=str(a.segment), prediction=str(a.prediction), seconds=round(time.time() - started, 1))
    (dest / "blend_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
