"""Predict the sheet strip of every fiber of a fiber store above a length, sharded over processes (sheet_strip/infer.py per fiber,
without scoring or renders).

    python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer_all --fibers FIBERS.vcobj --out OUT \\
        [--checkpoint scrollprize/fiber_strip_small_v1] [--min-length-mm 2.5] [--base-voxel-um 2.4] [--step 8] \\
        [--blend 16] [--trim 75] [--shard K N] [--across-fibers] [--workers N [--compile MODE] --crops gpu]

``--checkpoint`` and the CT, frame-model and umbilicus options are infer.py's (a Hugging Face repo by default).

The fibers (catalog order: UUID order) are dealt to ``N`` shards; shard ``K`` predicts every ``N``-th one from the ``K``-th, so
shards can run as parallel processes. A fiber is a decision every ``--step`` trace voxels from its start (as
infer.py), stitched into a tifxyz at OUT/strips/BUCKET/fiber_UUID (patch voxels; BUCKET: the UUID's first two digits;
uncompressed single-strip TIFFs, memory-mappable as written). Its nodes within ``--trim`` patch voxels of either fiber
end, or beyond it, are not kept (-1): the strip is least reliable there (the annotated fiber's ends, few decisions).
An H fiber runs along its columns and a V fiber along its rows, higher z at the top (``display_grid``; the layout is
in trace.json). ``expected_error.tif`` lies beside it
when the model predicts its error (``strip_uncertainty``: the stitched expected |error| per node, voxels; NaN off the
strip). Each finished fiber appends a line to OUT/index/shard_K.jsonl (fiber UUID, family, length, decisions, nodes,
checkpoint, the complete input reference, the mean expected error, seconds); an unchanged fiber already in it is skipped, so a stopped shard resumes where it
left off, also with another checkpoint (run.json lists every checkpoint used). Lengths: the store catalog's, in base
voxels of ``--base-voxel-um``.

Fiber by fiber, this process prepares the next batch of decisions (crops, frames) while the device runs the current
one, and writes each strip as its fiber completes; a record's seconds is the time since the shard's previous record.
Each fiber's decisions run in batches of ``--batch`` of their own, so on the CPU the strips are those of infer.py.
``--across-fibers`` fills every strip-model batch with the next ``--batch`` decisions, from the next fibers when one
ends, so batches stay full (faster on a GPU; heights then move by the batch's rounding).

``--workers N`` (CUDA or the CPU) runs the shard through sheet_strip/batched.py instead of fiber by fiber: N CPU worker processes build
the decisions' geometry and write the strips, and this process runs the crops (``--crops gpu``: on the GPU) and both
models over the decisions of many fibers at once. Strips and index records are written as above, in the order fibers
finish; a record's seconds is the time since the shard's previous record, since fibers overlap. The strip model runs
each fiber's decisions in batches of ``--batch`` as the per-fiber loop does (on the CPU, the same heights), or with
``--across-fibers`` in batches of ``--batch`` decisions from any fibers (faster at a large batch; bf16 heights move by
the batch size's rounding). ``--compile MODE`` compiles the strip model (torch.compile, with ``--across-fibers``),
with inductor emulating eager's bf16 casts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from vc_reader import images
import torch

from vesuvius.neural_tracing.tracers.fibers.sheet_strip.batched import (
    FiberJob, StripEngine, StripGeometry, run, stitched_strip)
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer import (
    BLEND, MIN_HEAD_ARC, STEP, TRIM, StripPredictor, add_checkpoint_arguments, recorded_file,
    trim_ends)  # noqa: F401  trim_ends: imported from here by callers
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import FiberStore, fiber_volume
from vesuvius.neural_tracing.tracers.shared.io import atomic_write
from vesuvius.neural_tracing.tracers.sheets.tracing.inference import write_surface


def selected_fibers(fibers, min_length_mm, base_voxel_um):
    """Catalog indices of the fibers longer than ``min_length_mm``, catalog order."""
    lengths = np.array([row['length'] for row in fibers.catalog], np.float64)*base_voxel_um/1000.
    return np.flatnonzero(lengths > min_length_mm)


def pending_fibers(catalog, chosen, completed):
    """Resume only identical selected inputs; UUID-only legacy completions must be regenerated."""
    return [c for c in chosen if completed.get(catalog[c]['id']) != catalog[c]['reference']]


# a worker reads the CT only for its fibers' seed headings (the structure tensor around each seed)
WORKER_CT_CACHE = 1 << 28
VOLUME_OPTIONS = ('ct', 'ct_level', 'ct_grid_scale', 'grid_scale', 'cache_dir', 'frame_checkpoint')


def decision_arcs(fiber, step=STEP):
    """Arclengths (trace voxels) of a fiber's decisions: every ``step`` from the first head with enough trace."""
    return np.arange(MIN_HEAD_ARC, fiber.length+1e-9, step)


def strip_of_fiber(predictor, fiber, step=STEP, blend=BLEND, trim=0.):
    """The stitched strip of a fiber (rows (R, Nv, 3) trace voxels, kept nodes, expected |error| per node in voxels
    or None) and its decision arclengths; nodes within ``trim`` trace voxels of the fiber's ends, or beyond them, are
    not kept."""
    arcs = decision_arcs(fiber, step)
    heights, scales, frames, positions = predictor.predict(fiber, arcs)
    grid = (predictor.f, predictor.v, predictor.trace_per_patch)
    rows, keep, expected = stitched_strip(grid, fiber, arcs, heights, scales, frames, positions, blend, trim)
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


@dataclass(frozen=True)
class StripWriter:
    """Writes a fiber's stitched strip under ``out`` (tifxyz, trace.json provenance, expected_error.tif) and returns
    its index record without ``seconds``; the per-fiber loop and the batched workers both write through it."""
    out: Path
    checkpoint: str
    checkpoint_step: object
    fibers: str
    decision_step: float
    blend: float
    trim: float
    trace_per_patch: float
    first_row: float
    row_step: float
    column_step: float
    base_voxel_um: float
    voxel_size_um: float

    @classmethod
    def of(cls, args, predictor, fibers_path):
        f, v, tpp = predictor.f, predictor.v, predictor.trace_per_patch
        return cls(args.out, predictor.source, predictor.ck.get('step'), str(fibers_path), args.step, args.blend,
                   args.trim, tpp, float(f[0]), (f[1]-f[0])/tpp, (v[1]-v[0])/tpp, args.base_voxel_um,
                   args.voxel_size_um)

    def write(self, catalog_index, row, rows, keep, expected, arcs):
        """``rows``, ``keep``, ``expected`` and ``arcs`` as ``strip_of_fiber`` returns them for catalog ``row``."""
        tpp = self.trace_per_patch
        fiber_id, family = row['id'], row['family']
        out = self.out/'strips'/fiber_id[:2]/f'fiber_{fiber_id}'
        nodes = int(keep.sum())
        rows, keep, expected, layout = display_grid(rows, keep, expected, family)
        provenance = dict(checkpoint=self.checkpoint, step=self.checkpoint_step, source_reference=row['reference'],
                          fibers=self.fibers, fiber=fiber_id, family=family, decision_step=self.decision_step,
                          blend=self.blend, trim_patch_voxels=self.trim, units='patch voxels', trace_per_patch=tpp,
                          first_row_arc_patch_voxels=float((arcs[0]+self.first_row)/tpp), **layout)
        along_columns = layout['fiber_along'] == 'columns'
        steps = (self.column_step, self.row_step) if along_columns else (self.row_step, self.column_step)
        write_surface(out, rows/tpp, keep, *steps, provenance, overwrite=True, voxel_size_um=self.voxel_size_um,
                      mmap_layout=True)
        record = dict(fiber=fiber_id, catalog=int(catalog_index), family=family, source_reference=row['reference'],
                      length_mm=float(row['length'])*self.base_voxel_um/1000., decisions=len(arcs),
                      nodes=nodes, path=str(out.relative_to(self.out)), checkpoint=self.checkpoint)
        if expected is not None:
            images.write_tiff(out/'expected_error.tif', np.where(keep, expected, np.nan).astype(np.float32))
            record['expected_error_vox'] = float(np.nanmean(np.where(keep, expected, np.nan)))
        return record


class StoreJob(FiberJob):
    """The fibers of a store by catalog index for ``batched.run``, written by ``writer``. Picklable for the spawned
    workers: the store drops its handle when pickled, and each worker opens its own. ``predictor_args``: the
    StripPredictor arguments (checkpoint, umbilicus, batch, volume options) each worker loads its geometry from."""

    def __init__(self, fibers, writer, predictor_args):
        self.fibers, self.writer, self.predictor_args = fibers, writer, predictor_args

    def geometry(self):
        checkpoint, umbilicus, batch, volume = self.predictor_args
        # a CPU predictor for the checkpoint's configuration and CT reader; no crops or models run in a worker
        predictor = StripPredictor(checkpoint, 'cpu', umbilicus, batch, volume, memory_cache=WORKER_CT_CACHE)
        return StripGeometry.of(predictor)

    def load(self, key):
        fiber = self.fibers[key]
        return fiber, decision_arcs(fiber, self.writer.decision_step)

    def write(self, key, fiber, arcs, strip, raw):
        return self.writer.write(key, self.fibers.catalog[key], *strip, arcs)

    def failed(self, key, error):
        raise error  # a seed without a usable CT orientation stops the run, as it stops the per-fiber loop


def batched_engine(args, predictor):
    """The StripEngine of ``--workers`` runs; SystemExit for options it cannot run."""
    try:
        StripGeometry.of(predictor)
        # both models in batches of --batch, as predict runs them
        engine = StripEngine(predictor, batch=args.batch, frame_chunk=args.batch, across_fibers=args.across_fibers,
                             compile_mode=args.compile)
    except ValueError as error:
        raise SystemExit(str(error))
    # the frame model (the only fp32 convolutions) without TF32, as on the CPU and in the measured runs
    torch.backends.cudnn.allow_tf32 = False
    return engine


def run_batched(engine, job, todo, workers, append, report):
    """The catalog indices ``todo`` through ``batched.run``, each record appended as its fiber finishes with the time
    since the previous record as its seconds: fibers overlap, so the column sums to the shard's wall time, as the
    per-fiber loop's does. ``report`` sees the progress every 50 fibers and at the end."""
    if not todo:
        return
    writer = job.writer
    last, reported = [time.time()], [0]

    def on_record(key, record):
        now = time.time()
        append(record, now-last[0])
        last[0] = now

    def log(count, total, decisions, seconds):
        # run calls this every pass of its loop; a pass may finish several fibers
        if count > reported[0] and (count//50 > reported[0]//50 or count == total):
            reported[0] = count
            report(count, decisions, seconds)

    run(engine, job, todo, workers=workers, blend=writer.blend, trim=writer.trim*writer.trace_per_patch,
        on_record=on_record, log=log)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_checkpoint_arguments(ap)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--fibers', help="fiber store (default: the checkpoint run's, when on this machine)")
    ap.add_argument('--min-length-mm', type=float, default=2.5)
    ap.add_argument('--base-voxel-um', type=float, default=2.4, help='voxel size of the store catalog lengths')
    ap.add_argument('--step', type=float, default=STEP, help='trace voxels between decisions (a multiple of the f spacing)')
    ap.add_argument('--blend', type=float, default=BLEND, help='trace voxels: decisions weigh their rows within this of the head')
    ap.add_argument('--trim', type=float, default=TRIM,
                    help='patch voxels: strip nodes within this of either fiber end (or beyond it) are not kept')
    ap.add_argument('--shard', type=int, nargs=2, default=(0, 1), metavar=('K', 'N'))
    ap.add_argument('--limit', type=int, help='at most this many fibers of the shard (benchmarks)')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--voxel-size-um', type=float, default=9.6, help='patch voxel size, for area_cm2 in meta.json')
    ap.add_argument('--workers', type=int, default=0,
                    help='batched mode (sheet_strip/batched.py): CPU worker processes building the geometry while this '
                         'process runs crops and models over many fibers (default 0: fiber by fiber)')
    ap.add_argument('--across-fibers', action='store_true',
                    help="strip-model batches of --batch decisions from any fibers (default: each fiber's own)")
    ap.add_argument('--compile', metavar='MODE',
                    help='with --workers and --across-fibers on CUDA: torch.compile the strip model in this mode (e.g. '
                         'max-autotune-no-cudagraphs), with inductor emulating eager precision casts')
    args = ap.parse_args(argv)
    if args.workers < 0:
        raise SystemExit('--workers must be 0 (fiber by fiber) or more')
    if args.compile and not (args.workers and args.across_fibers):
        raise SystemExit('--compile needs --workers and --across-fibers (one batch shape)')
    if args.compile and torch.device(args.device).type != 'cuda':
        raise SystemExit('--compile needs a CUDA --device (inductor with emulated bf16 casts)')
    torch.set_num_threads(args.threads)
    k, n = args.shard
    try:
        predictor = StripPredictor(args.checkpoint, args.device, args.umbilicus, args.batch, args, crops=args.crops,
                                   gpu_chunks=args.gpu_chunks)
    except ValueError as error:
        raise SystemExit(str(error))
    engine = batched_engine(args, predictor) if args.workers else None
    f, tpp = predictor.f, predictor.trace_per_patch
    df = f[1]-f[0]
    if abs(args.step/df-round(args.step/df)) > 1e-6 or args.step <= 0:
        raise SystemExit(f'--step must be a positive multiple of the strip row spacing {df:g}')
    fibers_path = args.fibers or recorded_file(predictor.data.get('fibers'))
    if not fibers_path:
        raise SystemExit(f"Pass --fibers: the checkpoint's fiber store {predictor.data.get('fibers')} is not on this "
                         'machine')
    fibers = FiberStore(fibers_path, fiber_volume(fibers_path), predictor.spec.grid_scale, split='all')
    chosen = selected_fibers(fibers, args.min_length_mm, args.base_voxel_um)[k::n]
    if args.limit:
        chosen = chosen[:args.limit]
    (args.out/'index').mkdir(parents=True, exist_ok=True)
    log_path = args.out/'index'/f'shard_{k}.jsonl'
    done = {}
    if log_path.exists():
        done = {entry['fiber']: entry.get('source_reference') for line in log_path.read_text().splitlines()
                if line.strip() for entry in [json.loads(line)]}
    checkpoint = predictor.source
    # A run resumed with another checkpoint keeps the earlier ones (each fiber's index line names its own).
    earlier = json.loads((args.out/'run.json').read_text()) if (args.out/'run.json').exists() else {}
    checkpoints = [c for c in earlier.get('checkpoints', [earlier['checkpoint']] if 'checkpoint' in earlier else [])
                   if c != checkpoint]+[checkpoint]
    atomic_write(args.out/'run.json', (json.dumps(dict(
        checkpoint=checkpoint, step=predictor.ck.get('step'), checkpoints=checkpoints, fibers=str(fibers_path),
        min_length_mm=args.min_length_mm, base_voxel_um=args.base_voxel_um, decision_step=args.step,
        blend=args.blend, trim_patch_voxels=args.trim, shards=n, units='patch voxels', trace_per_patch=tpp),
        indent=1)+'\n').encode())
    writer = StripWriter.of(args, predictor, fibers_path)
    started, decisions = time.time(), 0
    todo = pending_fibers(fibers.catalog, chosen, done)
    print(f'[shard {k}/{n}] {len(chosen)} fibers, {len(chosen)-len(todo)} done already', flush=True)
    with open(log_path, 'a') as log:
        def append(record, seconds):
            log.write(json.dumps(dict(record, seconds=round(seconds, 2)))+'\n')
            log.flush()

        def report(count, decisions, seconds):
            print(f'[shard {k}/{n}] {count}/{len(todo)} fibers, {decisions} decisions, '
                  f'{decisions/max(seconds, 1e-9):.1f} decisions/s', flush=True)

        if engine is not None:
            volume = argparse.Namespace(**{key: getattr(args, key) for key in VOLUME_OPTIONS})
            volume.cache_dir = predictor.spec.cache_dir  # as resolved here, so workers neither resolve nor announce it
            job = StoreJob(fibers, writer, (args.checkpoint, args.umbilicus, args.batch, volume))
            run_batched(engine, job, todo, args.workers, append, report)
            return
        # predict_many keeps the device busy between fibers, preparing the next batch while one runs; with
        # --across-fibers its batches also take decisions from the next fibers, otherwise each fiber's are its own, as
        # predict batches them. Records are written as fibers complete; seconds is the time since the previous one
        jobs = ((index, fiber, decision_arcs(fiber, args.step)) for index in todo for fiber in (fibers[index],))
        grid = (predictor.f, predictor.v, tpp)
        tick = time.time()
        results = predictor.predict_many(jobs, across_fibers=args.across_fibers)
        for count, ((catalog_index, fiber, arcs), result) in enumerate(results, 1):
            rows, keep, expected = stitched_strip(grid, fiber, arcs, *result, args.blend, args.trim*tpp)
            record = writer.write(catalog_index, fibers.catalog[catalog_index], rows, keep, expected, arcs)
            now = time.time()
            append(record, now-tick)
            tick = now
            decisions += len(arcs)
            if count % 50 == 0 or count == len(todo):
                report(count, decisions, time.time()-started)


if __name__ == '__main__':
    main()
