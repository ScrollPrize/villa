"""Measure the real fiber trainer, including loader waits and optimizer updates.

Run with the Vesuvius environment's Python; see --help. Run outputs are isolated
from the input checkpoint. Diagnostics and new DAgger collection are disabled.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


class BenchmarkComplete(Exception):
    pass


def replicate_replay(source, destination):
    """Private directories/JSON with read-only references to large replay arrays."""
    for path in source.rglob('*'):
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.suffix == '.json':
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(path.read_text().replace(str(source), str(destination)))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('existing', 'vc-reader'), required=True)
    parser.add_argument('--heading', choices=('learned', 'tensor'), default='learned')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--grad-steps', type=int, default=1)
    parser.add_argument('--lookahead', type=int, default=0,
                        help='Future batch plans per source/worker (default: disabled)')
    parser.add_argument('--vc-site-packages', type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or args.grad_steps < 1:
        parser.error('steps and grad-steps must be positive; warmup must be nonnegative')
    if args.heading == 'tensor' and args.backend != 'vc-reader':
        parser.error('tensor heading benchmark requires --backend vc-reader')
    if args.lookahead < 0:
        parser.error('lookahead must be nonnegative')
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('AGENTS_AGENT_MODE', '1')
    os.environ.setdefault('NUMBA_CACHE_DIR', str(args.output / 'numba_cache'))
    import torch
    import numpy as np
    from vesuvius.neural_tracing.tracers.fibers.train import train

    if args.vc_site_packages:
        # Load just vc_reader's editable import finder; keep the trainer's torch.
        sys.path.append(str(args.vc_site_packages.resolve()))
        os.environ['FIBER_BENCH_VC_SITE'] = str(args.vc_site_packages.resolve())
        importlib.import_module('_editable_skbc_vc_reader')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    source = args.checkpoint.parent
    run = json.loads((source / 'run.json').read_text())
    run['name'], run['out_root'] = 'run', str(args.output)
    run['runtime']['remote_prefetch_lookahead'] = args.lookahead
    done = int(checkpoint['step'])
    # Keep the LR schedule common to both runs. Stop via instrumentation before
    # the trainer's final checkpoint/evaluation work.
    run['training'].update(steps=done + args.warmup + args.steps + 1,
                           grad_steps=args.grad_steps,
                           dagger_every=0, diag_every=0, batch_diag_every=0,
                           recovery_every=0, ckpt_every=1000000000)
    out = args.output / 'run'
    out.mkdir()
    for path in source.glob('validation_*.json'):
        shutil.copy2(path, out / path.name)
    replicate_replay(source / 'dagger', out / 'dagger')
    config = args.output / 'config.json'
    config.write_text(json.dumps(run, indent=2) + '\n')
    backend = None
    if args.backend == 'vc-reader':
        from fiber_cuda_pipeline import install
        backend = install()
    heading = None
    if args.heading == 'tensor':
        from fiber_tensor_heading import install
        heading = install(backend.pool)

    records = []
    coverage = {}
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.crops import tight_block
    from vesuvius.neural_tracing.tracers.shared.geometry import CropSpec
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.prefetch import chunk_keys
    crop = CropSpec(**checkpoint['crop'])

    def audit(batches):
        for batch in batches:
            index = int(batch['dataset_id'][0])
            source_config = run['dataset']['sources'][index]
            key = source_config['ct'] + '::' + str(source_config['ct_level'])
            meta = checkpoint['ct_normalization']['volumes'][key]
            state = coverage.setdefault(source_config['name'], dict(crops=0, positions=set(), chunks=set(), touches=0))
            scale = source_config['grid_scale'] / source_config['ct_grid_scale']
            for pos, frame in zip(batch['crop_pos'].numpy(), batch['crop_frame'].numpy()):
                start, size = tight_block(pos, frame, crop, scale)
                keys = list(chunk_keys(start, size, meta['shape'], meta['chunks']))
                state['crops'] += 1
                state['positions'].add(pos.tobytes())
                state['chunks'].update(keys)
                state['touches'] += len(keys)
    original_next = train.DecisionBatchPrefetch.__next__
    original_update = train.optimizer_update
    pending = {}

    def next_batches(prefetch):
        torch.cuda.synchronize()
        pending['started'] = time.perf_counter()
        batches = original_next(prefetch)
        pending['loader_seconds'] = time.perf_counter() - pending['started']
        if backend is not None:
            for batch in batches:
                backend.materialize(batch)
        torch.cuda.synchronize()
        pending['input_seconds'] = time.perf_counter() - pending['started']
        return batches

    def update(*pos, **kw):
        started = time.perf_counter()
        metrics = original_update(*pos, **kw)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        index = len(records)
        row = dict(index=index, measured=index >= args.warmup,
                   seconds=finished - pending['started'],
                   loader_seconds=pending['loader_seconds'],
                   input_seconds=pending['input_seconds'],
                   update_seconds=finished - started, loss=metrics['loss'],
                   error_mean=metrics['error_mean'],
                   dataset_counts=metrics.get('dataset_counts'),
                   observed_states=metrics['observed_states'])
        if not np.isfinite(row['loss']) or not np.isfinite(row['error_mean']):
            raise RuntimeError('Training produced nonfinite benchmark metrics')
        records.append(row)
        if index == args.warmup:
            pending['measured_started'] = pending['started']
        if row['measured']:
            audit(pos[3])
        with (args.output / 'steps.jsonl').open('a') as file:
            file.write(json.dumps(row) + '\n')
        if len(records) % 25 == 0 or len(records) == args.warmup:
            print(f'BENCH {args.backend} {len(records)}/{args.warmup + args.steps}: '
                  f'{row["seconds"]:.3f}s, loss={row["loss"]:.5f}', flush=True)
        if len(records) == args.warmup + args.steps:
            pending['measured_wall_seconds'] = finished - pending['measured_started']
            raise BenchmarkComplete
        return metrics

    train.DecisionBatchPrefetch.__next__ = next_batches
    train.optimizer_update = update
    # Plotting is ancillary work between updates and can perturb loader overlap.
    train.plot_run_curves = lambda *a, **k: None
    launched = time.perf_counter()
    try:
        train.main(['--config', str(config), '--resume', str(args.checkpoint)])
    except BenchmarkComplete:
        pass
    finally:
        if heading is not None:
            heading.close()
    measured = [r for r in records if r['measured']]
    if len(measured) != args.steps:
        raise RuntimeError('Trainer ended without completing the requested measurements')
    report = dict(backend=args.backend, heading=args.heading, checkpoint=str(args.checkpoint),
                  command=[sys.executable, *sys.argv],
                  warmup=args.warmup, steps=args.steps, batch=run['training']['batch'],
                  grad_steps=run['training']['grad_steps'],
                  workers=run['runtime']['workers'], threads=run['runtime']['threads'],
                  remote_prefetch_lookahead=run['runtime']['remote_prefetch_lookahead'],
                  model_autocast='bfloat16', compiled=True,
                  dataset_names=[s['name'] for s in run['dataset']['sources']],
                  torch=torch.__version__, python=sys.version,
                  gpu=torch.cuda.get_device_name(), elapsed_seconds=time.perf_counter()-launched,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    report['measured_wall_seconds'] = pending['measured_wall_seconds']
    report['coverage'] = {name: dict(crops=state['crops'], unique_positions=len(state['positions']),
                                    unique_crop_footprint_chunks=len(state['chunks']),
                                    crop_footprint_chunk_touches=state['touches'])
                          for name, state in coverage.items()}
    for key in ('seconds', 'loader_seconds', 'input_seconds', 'update_seconds'):
        values = np.array([r[key] for r in measured])
        report[key] = dict(total=float(values.sum()), mean=float(values.mean()),
                           p50=float(np.median(values)), p95=float(np.percentile(values, 95)),
                           min=float(values.min()), max=float(values.max()))
    report['decisions_per_second'] = sum(r['observed_states'] for r in measured)/report['seconds']['total']
    report['wall_decisions_per_second'] = sum(r['observed_states'] for r in measured)/report['measured_wall_seconds']
    if backend is not None:
        report['reader_stats'] = backend.stats()
    if heading is not None:
        report['heading_stats'] = heading.stats()
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
