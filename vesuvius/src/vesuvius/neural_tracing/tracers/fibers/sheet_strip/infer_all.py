"""Predict the sheet strip of every AFV fiber above a length, sharded over processes (sheet_strip/infer.py per fiber,
without scoring or renders).

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer_all --checkpoint RUN/last.pt --out OUT \\
        [--afv FIBERS.afv] [--min-length-mm 2.5] [--base-voxel-um 2.4] [--step 8] [--blend 16] [--trim 75] \\
        [--shard K N]

The fibers (AFV catalog order) are dealt to ``N`` shards; shard ``K`` predicts every ``N``-th one from the ``K``-th, so
shards can run as parallel processes. A fiber is a decision every ``--step`` trace voxels from its start (as
infer.py), stitched into a tifxyz at OUT/strips/BUCKET/fiber_ID (patch voxels; BUCKET: the catalog index // 1000;
uncompressed single-strip TIFFs, memory-mappable as written). Its nodes within ``--trim`` patch voxels of either fiber
end, or beyond it, are not kept (-1): the strip is least reliable there (the annotated fiber's ends, few decisions).
An H fiber runs along its columns and a V fiber along its rows, higher z at the top (``display_grid``; the layout is
in trace.json). ``expected_error.tif`` lies beside it
when the model predicts its error (``strip_uncertainty``: the stitched expected |error| per node, voxels; NaN off the
strip). Each finished fiber appends a line to OUT/index/shard_K.jsonl (fiber id, family, length, decisions, nodes,
checkpoint, the mean expected error, seconds); a fiber already in it is skipped, so a stopped shard resumes where it
left off, also with another checkpoint (run.json lists every checkpoint used). Lengths: the AFV catalog's, in base
voxels of ``--base-voxel-um``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import tifffile
import torch

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer import (
    MIN_HEAD_ARC, StripPredictor, orient_decisions, stitch, stitch_values)
from vesuvius.neural_tracing.tracers.shared.annotations.afv import AFVFibers
from vesuvius.neural_tracing.tracers.shared.io import atomic_write
from vesuvius.neural_tracing.tracers.sheets.tracing.inference import write_surface


def selected_fibers(afv, min_length_mm, base_voxel_um):
    """Catalog indices of the fibers longer than ``min_length_mm``, catalog order."""
    lengths = np.array([row[3] for row in afv.catalog], np.float64)*base_voxel_um/1000.
    return np.flatnonzero(lengths > min_length_mm)


def trim_ends(keep, first_arc, row_step, length, margin):
    """``keep`` (R, C; rows along the fiber, row r at arclength ``first_arc + r*row_step``) without the rows beyond
    ``margin`` inside either end of a fiber of ``length`` (one unit throughout)."""
    arc = first_arc+np.arange(len(keep))*row_step
    return keep & ((arc >= margin) & (arc <= length-margin))[:, None]


def strip_of_fiber(predictor, fiber, step, blend, trim=0.):
    """The stitched strip of a fiber (rows (R, Nv, 3) trace voxels, kept nodes, expected |error| per node in voxels
    or None) and its decision arclengths; nodes within ``trim`` trace voxels of the fiber's ends, or beyond them, are
    not kept."""
    arcs = np.arange(MIN_HEAD_ARC, fiber.length+1e-9, step)
    heights, scales, frames, positions = predictor.predict(fiber, arcs)
    oriented, oriented_frames, flipped = orient_decisions(heights, frames)
    rows, weight, _ = stitch(oriented, oriented_frames, positions, arcs, predictor.f, predictor.v, blend)
    f = predictor.f
    keep = trim_ends(weight > 0, arcs[0]+f[0], f[1]-f[0], fiber.length, trim)
    expected = None
    if scales is not None:
        error = np.exp(np.where(flipped[:, None, None], scales[:, :, ::-1], scales))/predictor.trace_per_patch
        expected = stitch_values(error, arcs, predictor.f, blend)
    return rows, keep, expected, arcs



def display_grid(rows, keep, expected, family):
    """A strip (rows (R, C, 3), keep (R, C), expected (R, C) or None; rows along the fiber) laid out as the fiber runs:
    an H fiber along the columns (transposed), a V fiber along the rows, each turned so that higher z is the top
    (row 0). Returns rows, keep, expected and the layout: ``fiber_along`` ('rows'/'columns'), ``fiber_reversed``
    (the fiber axis runs against arclength) and ``across_reversed`` (the in-sheet axis runs against v)."""
    z = rows[..., 2]
    r, c = np.nonzero(keep)
    rising = lambda index: np.sum((index-index.mean())*(z[r, c]-z[r, c].mean())) > 0  # z grows with the index
    along_columns = family == 'H'
    flip_rows = not along_columns and len(r) > 1 and rising(r)  # V: the fiber's start at the top only if it is highest
    flip_columns = along_columns and len(c) > 1 and rising(c)  # H: v's highest side becomes row 0
    index = (slice(None, None, -1 if flip_rows else 1), slice(None, None, -1 if flip_columns else 1))
    arrays = [a if a is None else a[index] for a in (rows, keep, expected)]
    if along_columns:
        arrays = [a if a is None else np.swapaxes(a, 0, 1) for a in arrays]
    layout = dict(fiber_along='columns' if along_columns else 'rows', fiber_reversed=bool(flip_rows),
                  across_reversed=bool(flip_columns))
    return (*arrays, layout)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--afv', help="default: the checkpoint run's")
    ap.add_argument('--min-length-mm', type=float, default=2.5)
    ap.add_argument('--base-voxel-um', type=float, default=2.4, help='voxel size of the AFV catalog lengths')
    ap.add_argument('--step', type=float, default=8., help='trace voxels between decisions (a multiple of the f spacing)')
    ap.add_argument('--blend', type=float, default=16., help='trace voxels: decisions weigh their rows within this of the head')
    ap.add_argument('--trim', type=float, default=75., help='patch voxels: strip nodes within this of either fiber end '
                    '(or beyond it) are not kept')
    ap.add_argument('--shard', type=int, nargs=2, default=(0, 1), metavar=('K', 'N'))
    ap.add_argument('--limit', type=int, help='at most this many fibers of the shard (benchmarks)')
    ap.add_argument('--umbilicus', help="for models reading the crop orientation (default: the checkpoint run's)")
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--crop-threads', type=int, default=1, help="threads sampling a batch's CT crops")
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--voxel-size-um', type=float, default=9.6, help='patch voxel size, for area_cm2 in meta.json')
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    k, n = args.shard
    predictor = StripPredictor(args.checkpoint, args.device, args.umbilicus, args.batch, args.crop_threads)
    f, v, tpp = predictor.f, predictor.v, predictor.trace_per_patch
    df = f[1]-f[0]
    if abs(args.step/df-round(args.step/df)) > 1e-6 or args.step <= 0:
        raise SystemExit(f'--step must be a positive multiple of the strip row spacing {df:g}')
    afv_path = args.afv or predictor.data['afv']
    afv = AFVFibers(afv_path, predictor.spec.grid_scale, split='all')
    chosen = selected_fibers(afv, args.min_length_mm, args.base_voxel_um)[k::n]
    if args.limit:
        chosen = chosen[:args.limit]
    (args.out/'index').mkdir(parents=True, exist_ok=True)
    log_path = args.out/'index'/f'shard_{k}.jsonl'
    done = set()
    if log_path.exists():
        done = {json.loads(line)['fiber'] for line in log_path.read_text().splitlines() if line.strip()}
    checkpoint = str(Path(args.checkpoint).resolve())
    # A run resumed with another checkpoint keeps the earlier ones (each fiber's index line names its own).
    earlier = json.loads((args.out/'run.json').read_text()) if (args.out/'run.json').exists() else {}
    checkpoints = [c for c in earlier.get('checkpoints', [earlier['checkpoint']] if 'checkpoint' in earlier else [])
                   if c != checkpoint]+[checkpoint]
    atomic_write(args.out/'run.json', (json.dumps(dict(
        checkpoint=checkpoint, step=predictor.ck.get('step'), checkpoints=checkpoints, afv=str(afv_path),
        min_length_mm=args.min_length_mm, base_voxel_um=args.base_voxel_um, decision_step=args.step,
        blend=args.blend, trim_patch_voxels=args.trim, shards=n, units='patch voxels', trace_per_patch=tpp),
        indent=1)+'\n').encode())
    row_step, column_step = df/tpp, (v[1]-v[0])/tpp
    started, decisions = time.time(), 0
    todo = [c for c in chosen if int(afv.catalog[c][0]) not in done]
    print(f'[shard {k}/{n}] {len(chosen)} fibers, {len(chosen)-len(todo)} done already', flush=True)
    with open(log_path, 'a') as log:
        for count, catalog_index in enumerate(todo, 1):
            tick = time.time()
            fiber_id, _, family, length = afv.catalog[catalog_index][:4]
            fiber = afv[catalog_index]
            rows, keep, expected, arcs = strip_of_fiber(predictor, fiber, args.step, args.blend, args.trim*tpp)
            out = args.out/'strips'/f'{catalog_index//1000:04d}'/f'fiber_{fiber_id}'
            nodes = int(keep.sum())
            rows, keep, expected, layout = display_grid(rows, keep, expected, family)
            provenance = dict(checkpoint=str(Path(args.checkpoint).resolve()), step=predictor.ck.get('step'),
                              afv=str(afv_path), fiber=int(fiber_id), family=family, decision_step=args.step,
                              blend=args.blend, trim_patch_voxels=args.trim, units='patch voxels', trace_per_patch=tpp,
                              first_row_arc_patch_voxels=float((arcs[0]+f[0])/tpp), **layout)
            steps = (column_step, row_step) if layout['fiber_along'] == 'columns' else (row_step, column_step)
            write_surface(out, rows/tpp, keep, *steps, provenance, overwrite=True, voxel_size_um=args.voxel_size_um,
                          mmap_layout=True)
            record = dict(fiber=int(fiber_id), catalog=int(catalog_index), family=family,
                          length_mm=float(length)*args.base_voxel_um/1000., decisions=len(arcs),
                          nodes=nodes, path=str(out.relative_to(args.out)), checkpoint=checkpoint)
            if expected is not None:
                tifffile.imwrite(out/'expected_error.tif', np.where(keep, expected, np.nan).astype(np.float32))
                record['expected_error_vox'] = float(np.nanmean(np.where(keep, expected, np.nan)))
            record['seconds'] = round(time.time()-tick, 2)
            log.write(json.dumps(record)+'\n')
            log.flush()
            decisions += len(arcs)
            if count % 50 == 0 or count == len(todo):
                rate = decisions/(time.time()-started)
                print(f'[shard {k}/{n}] {count}/{len(todo)} fibers, {decisions} decisions, {rate:.1f} decisions/s',
                      flush=True)


if __name__ == '__main__':
    main()
