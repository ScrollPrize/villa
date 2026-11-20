"""Cut strips traced on mirrored chains (mirror.py) back into one strip per piece.

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.gap_strips.split CHAIN_RUN CHAINS.vcobj PIECES.vcobj \\
        OUT_RUN [--level-scale 4]

CHAIN_RUN is infer_all's output for CHAINS.vcobj. Every strip column along the fiber (trace.json ``fiber_along``; a
'rows' strip is handled transposed and written back as rows) is labelled by its centre vertex (the valid vertex
nearest the middle row): the nearest point of the chain line, sampled as the pieces' own points (label: the piece),
the straight joint steps between consecutive pieces every 0.25 patch voxels (label: joint) and the mirrored
extensions (label: joint). Joint columns are dropped. The columns of each piece (cropped to their first..last
column; other columns in between set invalid, -1) become OUT_RUN/strips/BUCKET/fiber_<piece UUID>, with
x/y/z/mask/expected_error cropped, meta.json bbox and area recomputed and the split recorded in trace.json.
OUT_RUN/run.json is CHAIN_RUN's with ``fibers`` = PIECES.vcobj, so the strips name the piece fibers
('<name>#piece<k>') exactly as strips traced per piece do. OUT_RUN/split_report.json lists the columns and nodes every piece got.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import read_polylines

LAYERS = ('x.tif', 'y.tif', 'z.tif', 'mask.tif', 'expected_error.tif')
JOINT = -1


def chain_labels(points, entry, joint_step=0.25):
    """Sample points of a chain line (patch voxels) and their labels: a piece's index in ``entry['pieces']`` for its
    own points, JOINT for the straight steps between consecutive pieces and for the mirrored extensions."""
    label_points, labels = [], []
    pieces = entry['pieces']
    for k, s in enumerate(pieces):
        label_points.append(points[s['start']:s['end'] + 1])
        labels += [k] * (s['end'] - s['start'] + 1)
        if k + 1 < len(pieces):
            a, b = points[s['end']], points[pieces[k + 1]['start']]
            n = max(2, int(np.ceil(np.linalg.norm(b - a) / joint_step)) + 1)
            seg = a + (b - a) * np.linspace(0, 1, n)[1:-1, None]
            label_points.append(seg)
            labels += [JOINT] * len(seg)
    for r0, r1 in entry.get('ext', []):
        label_points.append(points[r0:r1 + 1])
        labels += [JOINT] * (r1 - r0 + 1)
    return np.concatenate(label_points), np.array(labels)


def column_labels(x, y, z, label_points, labels):
    """Per strip column, the label of the chain point nearest its centre vertex (-2: the column has no valid
    vertex)."""
    valid = z != -1
    nr, nc = z.shape
    tree = cKDTree(label_points)
    out = np.full(nc, -2)
    for c in range(nc):
        rows = np.where(valid[:, c])[0]
        if len(rows) == 0:
            continue
        r = rows[np.argmin(np.abs(rows - (nr - 1) / 2))]
        out[c] = labels[tree.query([x[r, c], y[r, c], z[r, c]])[1]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('chain_run', type=Path)
    ap.add_argument('chains', type=Path)
    ap.add_argument('pieces', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--level-scale', type=float, default=4.0, help='volume voxels per patch voxel')
    a = ap.parse_args(argv)
    src, out = a.chain_run, a.out
    if out.exists():
        raise SystemExit(f'{out} exists')
    chain_log = {e['id']: e for e in json.load(open(str(a.chains) + '.joined.json'))}
    chain_points = {r['id']: p for r, p in read_polylines(a.chains)}
    piece_names = {e['id']: e['name'] for e in json.load(open(str(a.pieces) + '.pieces.json'))}
    (out / 'index').mkdir(parents=True)
    run = json.load(open(src / 'run.json'))
    run['fibers'] = str(a.pieces)
    run['split_from'] = dict(run_dir=str(src), fibers=str(a.chains))
    json.dump(run, open(out / 'run.json', 'w'), indent=1)
    if (src / 'ct_normalization.json').exists():
        shutil.copy2(src / 'ct_normalization.json', out / 'ct_normalization.json')
    report = []
    with open(out / 'index' / 'shard_0.jsonl', 'w') as index:
        for sd in sorted(src.glob('strips/*/fiber_*')):
            cid = sd.name[len('fiber_'):]
            e = chain_log[cid]
            trace = json.load(open(sd / 'trace.json'))
            if trace['fiber_along'] not in ('columns', 'rows'):
                raise SystemExit(f"{sd}: fiber_along {trace['fiber_along']!r}")
            transposed = trace['fiber_along'] == 'rows'  # work along columns; transpose back on write
            A = {f: tifffile.imread(sd / f) for f in LAYERS if (sd / f).exists()}
            if transposed:
                A = {f: arr.T for f, arr in A.items()}
            X, Y, Z = A['x.tif'], A['y.tif'], A['z.tif']
            nr, nc = Z.shape
            valid = Z != -1
            col_lab = column_labels(X, Y, Z, *chain_labels(chain_points[cid] / a.level_scale, e))
            rec = dict(chain_strip=str(sd), name=e['name'], joint_columns=int((col_lab == JOINT).sum()), pieces=[])
            for k, s in enumerate(e['pieces']):
                cols = np.where(col_lab == k)[0]
                if len(cols) == 0:
                    rec['pieces'].append(dict(id=s['id'], columns=0))
                    continue
                c0, c1 = cols.min(), cols.max() + 1
                keep = np.zeros((nr, nc), bool)
                keep[:, cols] = True
                keep &= valid
                path = f"strips/{s['id'][:2]}/fiber_{s['id']}"
                d = out / path
                d.mkdir(parents=True)
                for f, arr in A.items():
                    arr = arr.copy()
                    arr[~keep] = -1 if f in ('x.tif', 'y.tif', 'z.tif') else 0
                    tifffile.imwrite(d / f, np.ascontiguousarray(arr[:, c0:c1].T if transposed else arr[:, c0:c1]))
                meta = json.load(open(sd / 'meta.json'))
                k2 = keep[:, c0:c1]
                V = np.stack([X[:, c0:c1][k2], Y[:, c0:c1][k2], Z[:, c0:c1][k2]], 1)
                meta['bbox'] = [V.min(0).tolist(), V.max(0).tolist()]
                frac = keep.sum() / max(valid.sum(), 1)
                meta['area_vx2'] *= frac
                meta['area_cm2'] *= frac
                meta['uuid'] = meta['uuid'] + f'-p{k}'
                json.dump(meta, open(d / 'meta.json', 'w'), indent=1)
                piece_trace = dict(trace, fiber=s['id'], fibers=str(a.pieces),
                                   split=dict(chain=cid, chains=str(a.chains), columns=[int(c0), int(c1)],
                                              piece=s['piece']))
                piece_trace['first_row_arc_patch_voxels'] = None  # the chain row's arc origin means nothing for the piece
                json.dump(piece_trace, open(d / 'trace.json', 'w'), indent=1)
                index.write(json.dumps(dict(fiber=s['id'], name=piece_names[s['id']], family=trace.get('family'),
                                            nodes=int(keep.sum()), path=path, split_from=cid)) + '\n')
                rec['pieces'].append(dict(id=s['id'], name=piece_names[s['id']],
                                          columns=int(len(cols)), nodes=int(keep.sum())))
            report.append(rec)
    json.dump(report, open(out / 'split_report.json', 'w'), indent=1)
    got = [p for r in report for p in r['pieces']]
    print(f"{len(report)} chain strips -> {sum(1 for p in got if p['columns'])} piece strips; joint columns dropped "
          f"{sum(r['joint_columns'] for r in report)}; pieces with no columns {sum(1 for p in got if not p['columns'])}")


if __name__ == '__main__':
    main()
