"""Strips (an infer_all or split.py run) as surface patches for the spiral fit.

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.fit_patches RUN PATCHES \\
        [--prefix strip-net-] [--min-nodes 500] [--stride 2] [--min-strided-nodes 16]

Every strip RUN/strips/BUCKET/fiber_UUID with at least --min-nodes kept nodes becomes the tifxyz
PATCHES/<prefix>UUID holding every --stride-th row and column (strips come at about 4x5 patch-voxel quads, far finer
than the fit needs), with ``expected_error.tif`` thinned beside it when the strip has one. Its trace.json is the
strip's (which every strip must have) with ``patch`` (the source strip and the stride). A thinned strip with fewer than --min-strided-nodes nodes,
or fewer than 2 rows or columns, is not written.

The patches carry no weights. The fit weighs them by name and place: ``patch_sampling_name_weights`` (e.g.
{"strip-net-": 1.0}), ``patch_sampling_core_radius`` and ``patch_sampling_core_weight`` (spiral-fitting).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from vc_reader import images, tifxyz

from vesuvius.neural_tracing.tracers.sheets.tracing.inference import write_surface


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=Path)
    ap.add_argument('patches', type=Path)
    ap.add_argument('--prefix', default='strip-net-')
    ap.add_argument('--min-nodes', type=int, default=500, help='skip strips with fewer kept nodes')
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--min-strided-nodes', type=int, default=16)
    a = ap.parse_args(argv)
    if a.stride < 1:
        raise SystemExit('--stride must be at least 1')
    if a.patches.exists():
        raise SystemExit(f'{a.patches} exists')
    strips = sorted(a.run.glob('strips/*/fiber_*'))
    if not strips:
        raise SystemExit(f'{a.run} has no strips')
    surfaces, errors, traces = [], [], []
    for strip in strips:  # read everything before writing anything: a malformed strip stops the run
        try:
            surfaces.append(tifxyz.read(strip))
        except Exception as error:
            raise SystemExit(f'{strip} is not a readable tifxyz: {error}')
        errors.append(images.read(strip/'expected_error.tif') if (strip/'expected_error.tif').is_file() else None)
        if errors[-1] is not None and errors[-1].shape != tuple(surfaces[-1].shape):
            raise SystemExit(f'{strip}: expected_error.tif is not on the strip grid')
        try:
            traces.append(json.loads((strip/'trace.json').read_text()))
        except (OSError, ValueError) as error:
            raise SystemExit(f'{strip} has no readable trace.json: {error}')
        if not isinstance(traces[-1], dict):
            raise SystemExit(f'{strip} has no readable trace.json: not an object')
    a.patches.mkdir(parents=True)
    s = a.stride
    written = short = dropped = 0
    for strip, surface, error, trace in zip(strips, surfaces, errors, traces):
        xyz, valid = np.asarray(surface.xyz), np.asarray(surface.valid)
        if int(valid.sum()) < a.min_nodes:
            short += 1
            continue
        keep = valid[::s, ::s]
        if int(keep.sum()) < a.min_strided_nodes or min(keep.shape) < 2:
            dropped += 1
            continue
        dest = a.patches/f'{a.prefix}{strip.name[len("fiber_"):]}'
        write_surface(dest, xyz[::s, ::s], keep, s/surface.scale[0], s/surface.scale[1],
                      dict(trace, patch=dict(source_strip=str(strip), stride=s)),
                      voxel_size_um=surface.extra.get('voxel_size_um'))
        if error is not None:
            images.write_tiff(dest/'expected_error.tif', np.ascontiguousarray(error[::s, ::s]))
        written += 1
    print(f'{a.patches}: {written} patches as {a.prefix}* (stride {s}); skipped {short} with < {a.min_nodes} nodes, '
          f'{dropped} with < {a.min_strided_nodes} strided nodes')


if __name__ == '__main__':
    main()
