"""Write the fibers of a tracing/volume.py run as an Automated Fiber Volume (.afv), for VC3D's fiber collection dock.

The geometry comes from the run's finished tiles (tiles/*.npz, trace-grid xyz), scaled to native L0 xyz (the run's
grid_scale); each fiber is stored exactly as traced. The format, schema and coordinate frame come from a template
.afv of the same scan (only its 'frame' and 'root' metadata are kept). Each fiber row:
- id: the run's fiber id; name 'fiber_follow NNNNNNNN'; family 'H'/'V' (the seed's);
- annotation: a VC3D fiber annotation without line_points ({"type": "vc3d_fiber", "version": 1}) carrying the
  run's record: seed (native xyz), seed plane, stop reasons (backward, forward), links (ids of fibers each end
  touches, -1 for none), tile.
Metadata adds a new uuid, source, and provenance (the run's settings from run.json and the tile count).

Runs alongside a live run too: it reads only finished tiles (each tile's npz is written atomically).

  python -m vesuvius.neural_tracing.tracers.fibers.tracing.volume_afv OUT [--afv OUT/fibers.afv] [--template T.afv]
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np

from vesuvius.neural_tracing.tracers.shared.annotations.afv import write_afv

PARIS4_TEMPLATE = (Path(__file__).resolve().parents[1]/'datasets'/'automated_fiber_volumes'/
                   'Paris4-central-5mm-hv-dedup-20261005-presence-snapped.afv')


def run_fibers(out, grid_scale):
    """(fibers row dict, native xyz) for every fiber of the run's finished tiles, in tile then id order."""
    for f in sorted((Path(out)/'tiles').glob('*.npz')):
        if f.name.endswith('.partial.npz'):
            continue
        d = np.load(f)
        o = d['offsets']
        for k, (a, b) in enumerate(zip(o[:-1], o[1:])):
            xyz = d['points'][a:b].astype(np.float64)*grid_scale
            if len(xyz) < 2:
                continue
            fid = int(d['id'][k])
            annotation = dict(type='vc3d_fiber', version=1, source='fiber_follow volume trace',
                              fiber_follow=dict(id=fid, tile=f.stem, seed=(d['seed'][k]*grid_scale).tolist(),
                                                seed_plane=str(d['plane'][k]),
                                                stop_reasons=[str(r) for r in d['reasons'][k]],
                                                links=[int(v) for v in d['links'][k]]))
            row = dict(id=fid, name=f'fiber_follow {fid:08d}', family=str(d['family'][k]),
                       annotation=json.dumps(annotation, separators=(',', ':')))
            yield row, xyz


def export_afv(out, destination=None, template=PARIS4_TEMPLATE):
    """Write OUT's finished fibers to ``destination`` (default OUT/fibers.afv, replaced); returns write_afv's counts."""
    out = Path(out)
    record = json.loads((out/'run.json').read_text())
    destination = Path(destination) if destination else out/'fibers.afv'
    tmp = destination.with_name(destination.stem+'.writing.afv')
    tmp.unlink(missing_ok=True)
    tiles = sum(1 for f in (out/'tiles').glob('*.npz') if not f.name.endswith('.partial.npz'))
    provenance = dict(run=str(out.resolve()), written=time.strftime('%Y-%m-%dT%H:%M:%S%z'), finished_tiles=tiles,
                      checkpoint=record['checkpoint'], box_grid_xyz=record['box'], grid_scale=record['grid_scale'],
                      operating_policy=record['operating_policy'],
                      settings={k: record[k] for k in ('threshold', 'plane_spacing', 'dedupe', 'stop_decisions',
                                                       'overlap', 'trim_run', 'min_length', 'max_len', 'confidence',
                                                       'precision') if k in record})
    counts = write_afv(template, tmp, run_fibers(out, float(record['grid_scale'])),
                       dict(uuid=str(uuid.uuid4()), source='fiber_follow volume trace (tracing/volume.py)',
                            provenance=provenance),
                       keep_metadata=('frame', 'root'))
    tmp.replace(destination)
    return dict(counts, path=str(destination))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('out', help='tracing/volume.py output folder')
    ap.add_argument('--afv', help='destination (default OUT/fibers.afv)')
    ap.add_argument('--template', default=str(PARIS4_TEMPLATE), help='.afv of the same scan (schema and frame)')
    args = ap.parse_args(argv)
    t = time.time()
    result = export_afv(args.out, args.afv, args.template)
    print(json.dumps(dict(result, seconds=round(time.time()-t, 1))))


if __name__ == '__main__':
    main()
