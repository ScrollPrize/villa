"""Write the fibers of a tracing/volume.py run as a VC object store (fibers.vcobj), for VC3D.

The geometry comes from the run's finished tiles (tiles/*.npz, trace-grid xyz), scaled to native L0 xyz (the run's
grid_scale); each fiber is stored exactly as traced, bound to the volume of the run's CT when it is a Vesuvius
open-data store (its volume id); for any other CT, a fiber store of the same scan gives the volume (--template).
Each fiber:
- uuid: uuid5 of the run's fiber namespace (run.json 'fiber_namespace', drawn when the run started) and the run's
  integer fiber id, so every export of a run gives a fiber the same UUID; name 'fiber_follow NNNNNNNN'; family
  'H'/'V' (the seed's);
- meta['fiber_follow']: the run's record: run (the fiber namespace, which identifies the run and its run.json),
  run_id (the run's fiber id), tile, seed (native xyz), seed plane, stop reasons (backward, forward), links (UUIDs of
  the fibers each end touches, null for none).

Runs alongside a live run too: it reads only finished tiles (each tile's npz is written atomically).

  python -m vesuvius.neural_tracing.tracers.fibers.tracing.volume_store OUT [--fibers OUT/fibers.vcobj]
      [--template T.vcobj] [--ct CT]   (CT: the run's CT store when run.json predates its record, e.g. s3://...)
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np

from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fiber_volume, write_fibers
from vesuvius.neural_tracing.tracers.shared.annotations.open_data import open_data_volume


def run_ct(record, ct=None):
    """The CT store a run traced: given, recorded (ct_zarr; --ct), else its checkpoint's."""
    if ct or record.get('ct_zarr') or record.get('ct'):
        return ct or record.get('ct_zarr') or record['ct']
    import torch
    from vesuvius.neural_tracing.tracers.fibers.tracing.runtime import resolve_checkpoint
    return torch.load(resolve_checkpoint(record['checkpoint']), map_location='cpu', weights_only=False)['vol_spec']['ct_zarr']


def fiber_uuid(namespace, fid):
    """The exported UUID of the run's fiber ``fid`` (an int), None for -1 (no fiber)."""
    return None if fid < 0 else str(uuid.uuid5(uuid.UUID(str(namespace)), str(int(fid))))


def run_fibers(out, grid_scale, namespace):
    """(write_fibers row dict, native xyz) for every fiber of the run's finished tiles, in tile then id order."""
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
            record = dict(run=str(namespace), run_id=fid, tile=f.stem, seed=(d['seed'][k]*grid_scale).tolist(),
                          seed_plane=str(d['plane'][k]), stop_reasons=[str(r) for r in d['reasons'][k]],
                          links=[fiber_uuid(namespace, int(v)) for v in d['links'][k]])
            row = dict(id=fiber_uuid(namespace, fid), name=f'fiber_follow {fid:08d}', family=str(d['family'][k]),
                       meta=dict(fiber_follow=record))
            yield row, xyz


def export_store(out, destination=None, template=None, ct=None):
    """Write OUT's finished fibers to ``destination`` (default OUT/fibers.vcobj, replaced); returns write_fibers'
    counts. The volume: ``template``'s (a fiber store of the same scan) when given, else the run's open-data CT's
    (``ct`` overrides it)."""
    out = Path(out)
    record = json.loads((out/'run.json').read_text())
    if not record.get('fiber_namespace'):
        raise SystemExit(f'{out}/run.json has no fiber_namespace (the UUID namespace of its exported fibers)')
    destination = Path(destination) if destination else out/'fibers.vcobj'
    if template:
        volume = fiber_volume(template)
    else:
        store = run_ct(record, ct)
        ct_volume = open_data_volume(store)
        if ct_volume is None:
            raise SystemExit(f'{store} is not a Vesuvius open-data store: pass a fiber store of the same scan as the '
                             'template of its volume (--template / --fibers-template)')
        volume = ct_volume[1]
    tmp = destination.with_name(destination.stem+'.writing.vcobj')
    tmp.unlink(missing_ok=True)
    counts = write_fibers(tmp, run_fibers(out, float(record['grid_scale']), record['fiber_namespace']), volume=volume)
    tmp.replace(destination)
    return dict(fibers=counts['fibers'], points=counts['points'], volume=volume, path=str(destination))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('out', help='tracing/volume.py output folder')
    ap.add_argument('--fibers', help='destination (default OUT/fibers.vcobj)')
    ap.add_argument('--template', help="fiber store of the same scan: its volume (default: the volume of the run's "
                                       'CT, when a Vesuvius open-data store)')
    ap.add_argument('--ct', help="the run's CT store (default: recorded in run.json, else the checkpoint's)")
    args = ap.parse_args(argv)
    t = time.time()
    result = export_store(args.out, args.fibers, args.template, args.ct)
    print(json.dumps(dict(result, seconds=round(time.time()-t, 1))))


if __name__ == '__main__':
    main()
