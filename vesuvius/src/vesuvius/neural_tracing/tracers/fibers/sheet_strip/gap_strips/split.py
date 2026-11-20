"""Cut strips traced on mirrored chains (mirror.py) back into one strip per piece.

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.split CHAIN_RUN CHAINS.vcobj PIECES.vcobj \\
        OUT_RUN [--max-offset 16]

CHAIN_RUN is infer_all's output for CHAINS.vcobj (or for a retry.py store). A strip is used only when its trace.json
names the chain version now in CHAINS.vcobj and PIECES.vcobj holds the piece versions that chain was made with, so
a strip never meets other geometry than it was traced on. Every strip
column along the fiber (trace.json ``fiber_along``; a 'rows' strip is handled transposed and written back as rows) is
placed on the chain by its centre vertex (the valid vertex nearest the middle row): the nearest point of the chain
polyline, in patch voxels (the chain's recorded ``level_scale``). A column belongs to a piece only when that point
lies on the piece itself, its end points included; columns on a joint step between pieces or on a mirrored extension
are dropped. The columns of each piece (cropped to their first..last column; other columns in between set invalid)
become the tifxyz OUT_RUN/strips/BUCKET/fiber_<piece UUID> (written as infer_all writes strips, area and bounds
measured on the piece strip), with ``expected_error.tif`` cropped beside it. Its trace.json is the chain strip's with
``fiber``, ``fibers`` and ``source_reference`` naming the piece in PIECES.vcobj, and ``split`` (the chain, the chain
version traced, the column range, the piece number). OUT_RUN/run.json is CHAIN_RUN's with ``fibers`` = PIECES.vcobj;
OUT_RUN/index/shard_0.jsonl and OUT_RUN/split_report.json list the piece strips. A run that stops leaves no OUT_RUN.

A strip whose centre vertices lie more than --max-offset patch voxels (median) from its chain stops the run: the
chain was traced at another scale than its ``level_scale``.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from vc_reader import images, tifxyz

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.spans import read_derived
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fiber_catalog
from vesuvius.neural_tracing.tracers.shared.geometry import exact_nearest
from vesuvius.neural_tracing.tracers.sheets.tracing.inference import write_surface

JOINT, EMPTY = -1, -2


def chain_positions(points, queries):
    """For every query point, its nearest point on the polyline ``points`` as a position in point-index units
    (segment index + fraction along the segment) and its distance from it."""
    distance, _, segment, fraction = exact_nearest(queries, points)
    return segment+fraction, distance


def column_labels(xyz, valid, points, pieces):
    """Per strip column (axis 1 of ``xyz`` (R, C, 3)), the index in ``pieces`` of the piece its centre vertex lies
    on (``pieces``: dicts with inclusive ``start``/``end`` point indices into the chain ``points``), JOINT when it
    lies on a joint step or an extension, EMPTY when the column has no valid vertex; and the median distance of the
    centre vertices from the chain."""
    nr, nc = valid.shape
    labels = np.full(nc, EMPTY)
    columns = np.flatnonzero(valid.any(0))
    if not len(columns):
        return labels, 0.
    rows = np.array([np.flatnonzero(valid[:, c]) for c in columns], dtype=object)
    centre = np.array([r[np.argmin(np.abs(r-(nr-1)/2))] for r in rows])
    position, distance = chain_positions(points, xyz[centre, columns].astype(np.float64))
    labels[columns] = JOINT
    for k, piece in enumerate(pieces):
        labels[columns[(position >= piece['start']) & (position <= piece['end'])]] = k
    return labels, float(np.median(distance))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('chain_run', type=Path)
    ap.add_argument('chains', type=Path)
    ap.add_argument('pieces', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--max-offset', type=float, default=16.,
                    help='patch voxels: largest median distance of a strip from its chain')
    a = ap.parse_args(argv)
    src, out = a.chain_run, a.out
    if out.exists():
        raise SystemExit(f'{out} exists')
    strips = sorted(src.glob('strips/*/fiber_*'))
    if not (src/'run.json').is_file() or not strips:
        raise SystemExit(f'{src} is not an infer_all run with strips')
    try:
        chains = read_derived(a.chains, 'chain')
    except ValueError as error:
        raise SystemExit(str(error))
    piece_rows = {row['id']: row for row in fiber_catalog(a.pieces)}
    (out/'index').mkdir(parents=True)
    try:
        split_run(a, src, out, strips, chains, piece_rows)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise


def split_run(a, src, out, strips, chains, piece_rows):
    run = json.loads((src/'run.json').read_text())
    run['fibers'] = str(a.pieces)
    run['split_from'] = dict(run_dir=str(src), fibers=str(a.chains))
    (out/'run.json').write_text(json.dumps(run, indent=1))
    if (src/'ct_normalization.json').exists():
        shutil.copy2(src/'ct_normalization.json', out/'ct_normalization.json')
    report = []
    with open(out/'index'/'shard_0.jsonl', 'w') as index:
        for sd in strips:
            cid = sd.name[len('fiber_'):]
            if cid not in chains:
                raise SystemExit(f'{sd}: no chain {cid} in {a.chains}')
            row, points = chains[cid]
            derived = row['gap_strips']
            trace = json.loads((sd/'trace.json').read_text())
            if trace.get('source_reference') != row['reference']:
                raise SystemExit(f'{sd} was not traced on the version of chain {cid} in {a.chains}')
            if trace.get('fiber_along') not in ('columns', 'rows'):
                raise SystemExit(f"{sd}: fiber_along {trace.get('fiber_along')!r}")
            changed = [s['id'] for s in derived['pieces']
                       if s['id'] not in piece_rows or piece_rows[s['id']]['reference'] != s['reference']]
            if changed:
                raise SystemExit(f'{a.pieces} does not hold piece {changed[0]} as chain {cid} was made with it')
            surface = tifxyz.read(sd)
            transposed = trace['fiber_along'] == 'rows'  # work along columns; transpose back on write
            turn = (lambda array: np.swapaxes(array, 0, 1)) if transposed else (lambda array: array)
            xyz, valid = turn(np.asarray(surface.xyz)), turn(np.asarray(surface.valid))
            expected = None
            if (sd/'expected_error.tif').is_file():
                expected = images.read(sd/'expected_error.tif')
                if expected.shape != tuple(surface.shape):
                    raise SystemExit(f'{sd}: expected_error.tif is not on the strip grid')
                expected = turn(expected)
            labels, offset = column_labels(xyz, valid, points/derived['level_scale'], derived['pieces'])
            if offset > a.max_offset:
                raise SystemExit(f"{sd} lies {offset:.1f} patch voxels from chain {cid}: it was not traced at "
                                 f"{derived['level_scale']:g} volume voxels per patch voxel")
            rec = dict(chain_strip=str(sd), chain=cid, name=row['name'], dropped_columns=int((labels == JOINT).sum()),
                       pieces=[])
            for k, s in enumerate(derived['pieces']):
                piece = piece_rows[s['id']]
                cols = np.flatnonzero(labels == k)
                if len(cols) == 0:
                    rec['pieces'].append(dict(id=s['id'], name=piece['name'], columns=0, nodes=0))
                    continue
                c0, c1 = int(cols.min()), int(cols.max())+1
                keep = np.zeros(valid.shape, bool)
                keep[:, cols] = True
                keep &= valid
                path = f"strips/{s['id'][:2]}/fiber_{s['id']}"
                provenance = dict(trace, fiber=s['id'], fibers=str(a.pieces), source_reference=piece['reference'],
                                  first_row_arc_patch_voxels=None,  # the chain's arc origin means nothing here
                                  split=dict(chain=cid, chain_reference=row['reference'], chains=str(a.chains),
                                             columns=[c0, c1], piece=s['piece']))
                write_surface(out/path, turn(xyz[:, c0:c1]), turn(keep[:, c0:c1]), 1./surface.scale[0],
                              1./surface.scale[1], provenance, voxel_size_um=surface.extra.get('voxel_size_um'),
                              mmap_layout=True)
                if expected is not None:
                    images.write_tiff(out/path/'expected_error.tif', np.ascontiguousarray(
                        turn(np.where(keep, expected, np.nan)[:, c0:c1]).astype(np.float32)))
                nodes = int(keep.sum())
                index.write(json.dumps(dict(fiber=s['id'], name=piece['name'], family=trace.get('family'),
                                            source_reference=piece['reference'], nodes=nodes, path=path,
                                            split_from=cid))+'\n')
                rec['pieces'].append(dict(id=s['id'], name=piece['name'], columns=int(len(cols)), nodes=nodes))
            report.append(rec)
    (out/'split_report.json').write_text(json.dumps(report, indent=1))
    got = [p for r in report for p in r['pieces']]
    print(f"{len(report)} chain strips -> {sum(1 for p in got if p['columns'])} piece strips; "
          f"{sum(r['dropped_columns'] for r in report)} joint or mirrored columns dropped; "
          f"{sum(1 for p in got if not p['columns'])} pieces with no columns")


if __name__ == '__main__':
    main()
