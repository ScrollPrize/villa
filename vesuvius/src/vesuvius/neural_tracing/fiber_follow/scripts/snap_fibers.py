"""Snap fiber labels to a fiber presence prediction (data/fiber_snapping.py): AFV files or VC3D fiber JSONs.

    python scripts/snap_fibers.py INPUT OUTPUT --presence PRESENCE_ZARR_LEVEL [--native-per-voxel 8] [options]

INPUT is an .afv file (OUTPUT: a new .afv with the same schema, fiber ids, names, families and annotations, the
snapped geometry and a ``snapping`` metadata record) or a VC3D fiber JSON / directory of them (OUTPUT: a directory of
the same files with snapped ``line_points``, each control point moved onto the snapped line at its original arclength
fraction, and ``snapping.json`` beside them). Geometry is native base-voxel xyz; it is mapped to presence voxels as
native/--native-per-voxel - --offset (e.g. the lasagna presence level 3 of PHercParis4: 8 native voxels per voxel).
Written polylines are sampled every --output-step presence voxels.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from vesuvius.neural_tracing.fiber_follow.data.fiber_snapping import SnapConfig, snap_afv, snap_json
from vesuvius.neural_tracing.fiber_follow.data.volume import ChunkedArray


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--presence', required=True, help='Presence zarr array (a level directory, uint8, decoded)')
    ap.add_argument('--native-per-voxel', type=float, required=True, help='Native base voxels per presence voxel')
    ap.add_argument('--offset', type=float, default=0., help='Presence-voxel offset subtracted after scaling')
    ap.add_argument('--output-step', type=float, default=1., help='Presence voxels between written points')
    ap.add_argument('--limit', type=int, default=0, help='AFV: only the first N fibers (testing)')
    ap.add_argument('--device', default=None)
    for field, value in SnapConfig().to_dict().items():
        ap.add_argument('--'+field.replace('_', '-'), type=type(value), default=value)
    args = ap.parse_args(argv)
    cfg = SnapConfig(**{f: getattr(args, f) for f in SnapConfig().to_dict()})
    presence = ChunkedArray(args.presence, 4 << 30)
    if presence.dtype != np.dtype('uint8'):
        raise ValueError('Presence must be a decoded uint8 array')
    provenance = dict(method='presence centroid snapping (data/fiber_snapping.py)', presence=str(Path(args.presence).resolve()),
                      native_per_voxel=args.native_per_voxel, offset=args.offset, output_step=args.output_step,
                      config=cfg.to_dict())
    began = time.monotonic()
    if args.input.suffix == '.afv':
        result = snap_afv(args.input, args.output, presence, args.native_per_voxel, args.offset, cfg, args.device,
                          args.output_step, args.limit, provenance)
    else:
        paths = sorted(args.input.glob('*.json')) if args.input.is_dir() else [args.input]
        paths = [p for p in paths if json.loads(p.read_text()).get('type') == 'vc3d_fiber']
        result = snap_json(paths, args.output, presence, args.native_per_voxel, args.offset, cfg, args.device,
                           args.output_step, provenance)
    print(json.dumps(dict(result, seconds=round(time.monotonic()-began, 1), output=str(args.output))))


if __name__ == '__main__':
    main()
