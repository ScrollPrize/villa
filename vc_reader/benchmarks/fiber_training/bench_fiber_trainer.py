"""Measure the real fiber trainer, including loader waits and optimizer updates.

Run with the Vesuvius environment's Python; see --help. Run outputs are isolated
from the input checkpoint. Diagnostics and new DAgger collection are disabled.
"""
import argparse
import importlib
import hashlib
import subprocess
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
    parser.add_argument('--reader-device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--trainer-source', type=Path,
                        default=Path(__file__).resolve().parents[3]/'vesuvius'/'src',
                        help='Trainer source checkout; defaults to this branch')
    parser.add_argument('--location-weights-dir', type=Path,
                        help='Explicit sampling-weight cache shared by all compared routes')
    parser.add_argument('--heading', choices=('learned', 'tensor'), default='learned')
    parser.add_argument('--heading-stream', choices=('default', 'separate'), default='default',
                        help='Diagnostic CUDA stream for the tensor heading service')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--grad-steps', type=int, default=1)
    parser.add_argument('--lookahead', type=int, default=0,
                        help='Future batch plans per source/worker (default: disabled)')
    parser.add_argument('--cuda-augmentation', choices=('sample', 'batch'), default='batch')
    parser.add_argument('--cuda-cache', choices=('separate', 'shared'), default='shared')
    parser.add_argument('--geometry', choices=('existing', 'vc-reader'), default='existing')
    parser.add_argument('--geometry-placement', choices=('main', 'prefetch'), default='prefetch',
                        help='Prepare CUDA geometry in the existing batch producer or on the main thread')
    parser.add_argument('--validate-geometry', action='store_true',
                        help='Compare targets against the trainer during a separate correctness run')
    parser.add_argument('--timeline', action='store_true')
    parser.add_argument('--timing-sync', choices=('stream','device'), default='stream',
                        help='Wait for measured work on its stream; device also drains background work')
    parser.add_argument('--profile-workers', action='store_true',
                        help='Profile one geometry/label batch per worker (requires --timeline)')
    parser.add_argument('--gil-interval', type=float,
                        help='Diagnostic Python thread switch interval in seconds')
    parser.add_argument('--profile-batches', type=int, default=0,
                        help='Capture this many measured batches with an external CUDA profiler; -1 captures through process exit')
    parser.add_argument('--vc-site-packages', type=Path)
    args = parser.parse_args()
    cpu_reader = args.backend == 'vc-reader' and args.reader_device == 'cpu'
    if args.steps < 1 or args.warmup < 0 or args.grad_steps < 1:
        parser.error('steps and grad-steps must be positive; warmup must be nonnegative')
    if args.heading == 'tensor' and args.backend != 'vc-reader':
        parser.error('tensor heading benchmark requires --backend vc-reader')
    if args.geometry == 'vc-reader' and args.backend != 'vc-reader':
        parser.error('vc-reader geometry requires --backend vc-reader')
    if args.validate_geometry and args.geometry != 'vc-reader':
        parser.error('validate-geometry requires --geometry vc-reader')
    if args.profile_workers and not args.timeline:
        parser.error('profile-workers requires --timeline')
    if args.lookahead < 0:
        parser.error('lookahead must be nonnegative')
    if args.gil_interval is not None:
        if args.gil_interval <= 0:
            parser.error('gil-interval must be positive')
        sys.setswitchinterval(args.gil_interval)
    args.checkpoint = args.checkpoint.resolve()
    args.trainer_source = args.trainer_source.resolve()
    if not (args.trainer_source/'vesuvius').is_dir():
        parser.error('trainer-source must contain the vesuvius package')
    sys.path.insert(0, str(args.trainer_source))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['FIBER_BENCH_TIMELINE'] = '1' if args.timeline else '0'
    if args.profile_workers:
        os.environ['FIBER_BENCH_WORKER_PROFILE'] = str(args.output)
    os.environ.setdefault('AGENTS_AGENT_MODE', '1')
    os.environ.setdefault('NUMBA_CACHE_DIR', str(args.output / 'numba_cache'))
    import torch
    import numpy as np
    synchronize = torch.cuda.synchronize if args.timing_sync == 'device' else lambda: torch.cuda.current_stream().synchronize()
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
    location_weights = {}
    if args.location_weights_dir is not None:
        from vesuvius.neural_tracing.tracers.fibers.data.location_weights import cache_path
        weights_dir = args.location_weights_dir.resolve()
        run['training']['location_weights_dir'] = str(weights_dir)
        for source_config in run['dataset']['sources']:
            path = cache_path(source_config, weights_dir)
            location_weights[source_config['name']] = dict(path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
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
    if args.backend == 'vc-reader' and not cpu_reader:
        from fiber_cuda_pipeline import install
        backend = install(augmentation=args.cuda_augmentation, cache=args.cuda_cache)
    heading = None
    geometry = None
    if args.geometry == 'vc-reader':
        os.environ['FIBER_BENCH_GEOMETRY_FAILURE'] = str(args.output/'geometry_failure.pt')
        from fiber_geometry import install
        geometry = install(validate=args.validate_geometry,
                           placement='workers' if cpu_reader else args.geometry_placement,
                           backend=args.reader_device)
    if cpu_reader:
        from fiber_cpu_pipeline import install
        backend = install(tensor_heading=args.heading == 'tensor')
    if args.heading == 'tensor' and not cpu_reader:
        from fiber_tensor_heading import install
        heading = install(backend.pool, stream=args.heading_stream)

    extension_build = None
    if backend is not None:
        from vc_reader import _transforms, _reader
        binary = Path(_transforms.__file__).resolve()
        extension_build = dict(path=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                            revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                cwd=Path(__file__).resolve().parent, text=True).strip())
        reader_binary = Path(_reader.__file__).resolve()
        extension_build['reader'] = dict(path=str(reader_binary), sha256=hashlib.sha256(reader_binary.read_bytes()).hexdigest())
        extension_build['adapters'] = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ('bench_fiber_trainer.py', 'fiber_cuda_pipeline.py', 'fiber_cpu_pipeline.py',
                         'fiber_tensor_heading.py', 'fiber_geometry.py')}
        from vc_reader import cuda
        extension_build['python'] = {name: hashlib.sha256((Path(cuda.__file__).parent/name).read_bytes()).hexdigest()
                                 for name in ('cuda.py', 'geometry.py')}
    from fiber_timeline import span
    if args.timeline:
        from fiber_timeline import install_main
        install_main(train)
    if args.profile_workers and args.backend == 'existing':
        from fiber_timeline import ProfileDataLoader
        torch.utils.data.DataLoader = ProfileDataLoader
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
    optimizer_step_devices = {}

    def next_batches(prefetch):
        if args.profile_batches and len(records) == args.warmup:
            torch.cuda.profiler.start()
        sync_started = time.perf_counter()
        with span('before_batch.synchronize'):
            synchronize()
        pending['started'] = time.perf_counter()
        with span('loader.wait'):
            batches = original_next(prefetch)
        returned = time.perf_counter()
        pending['loader_seconds'] = returned - pending['started']
        pending['geometry_seconds'] = 0.
        pending['geometry_background_seconds'] = 0.
        if args.timeline:
            pending['trace'] = dict(sync_started=sync_started, requested=pending['started'], returned=returned,
                worker=[b.get('_bench_worker_timing') for b in batches],
                collect=[b.get('_bench_collect_timing') for b in batches])
        if backend is not None:
            for batch in batches:
                backend.materialize(batch)
                if geometry is not None and not cpu_reader and args.geometry_placement == 'main':
                    geometry_started = time.perf_counter()
                    with span('geometry.materialize'):
                        geometry.materialize(batch)
                    pending['geometry_seconds'] += time.perf_counter()-geometry_started
                elif geometry is not None and not cpu_reader:
                    batch['foreign'].record_stream(torch.cuda.current_stream())
                    timing = batch['_bench_geometry_timing']
                    pending['geometry_background_seconds'] += timing['ready']-timing['began']
        with span('images.synchronize'):
            synchronize()
        if args.timeline:
            pending['trace']['prepared'] = time.perf_counter()
        pending['input_seconds'] = time.perf_counter() - pending['started']
        return batches

    def update(*pos, **kw):
        if not records:
            for state in pos[2].state.values():
                if 'step' in state:
                    device = str(state['step'].device)
                    optimizer_step_devices[device] = optimizer_step_devices.get(device, 0) + 1
        started = time.perf_counter()
        with span('model.update'):
            metrics = original_update(*pos, **kw)
        metrics_done = time.perf_counter()
        with span('model.synchronize'):
            synchronize()
        finished = time.perf_counter()
        index = len(records)
        row = dict(index=index, measured=index >= args.warmup,
                   seconds=finished - pending['started'],
                   loader_seconds=pending['loader_seconds'],
                   geometry_seconds=pending['geometry_seconds'],
                   geometry_background_seconds=pending['geometry_background_seconds'],
                   input_seconds=pending['input_seconds'],
                   update_seconds=finished - started, loss=metrics['loss'],
                   error_mean=metrics['error_mean'],
                   dataset_counts=metrics.get('dataset_counts'),
                   observed_states=metrics['observed_states'])
        if not np.isfinite(row['loss']) or not np.isfinite(row['error_mean']):
            raise RuntimeError('Training produced nonfinite benchmark metrics')
        if args.timeline:
            row['trace'] = dict(pending['trace'], update_started=started, metrics_done=metrics_done, finished=finished)
        records.append(row)
        if args.profile_batches and len(records) == args.warmup + args.profile_batches:
            torch.cuda.profiler.stop()
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
    if args.timeline and heading is not None:
        (args.output / 'heading_timeline.json').write_text(json.dumps(heading.timeline) + '\n')
    measured = [r for r in records if r['measured']]
    if len(measured) != args.steps:
        raise RuntimeError('Trainer ended without completing the requested measurements')
    from vesuvius.neural_tracing.tracers.fibers.train import step as training_step
    trainer_source = Path(training_step.__file__).resolve()
    tracers = trainer_source.parents[2]
    trainer_hashes = {str(path.relative_to(tracers)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(tracers.rglob('*.py')) if 'tests' not in path.parts}
    report = dict(backend=args.backend, heading=args.heading, geometry=args.geometry,
                  reader_device=args.reader_device if backend is not None else None,
                  trainer_source=str(args.trainer_source),
                  location_weights=location_weights,
                  trainer_hashes=trainer_hashes,
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  trainer_step=dict(path=str(trainer_source),
                                    sha256=hashlib.sha256(trainer_source.read_bytes()).hexdigest()),
                  optimizer_step_devices=optimizer_step_devices,
                  geometry_placement=('workers' if cpu_reader else args.geometry_placement) if geometry is not None else None,
                  validate_geometry=args.validate_geometry, checkpoint=str(args.checkpoint),
                  command=[sys.executable, *sys.argv],
                  cuda_augmentation=args.cuda_augmentation if backend is not None and not cpu_reader else None,
                  cuda_cache=args.cuda_cache if backend is not None and not cpu_reader else None,
                  extension_build=extension_build, timeline=args.timeline, profile_batches=args.profile_batches,
                  timing_sync=args.timing_sync,
                  gil_interval=sys.getswitchinterval(),
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
    for key in ('seconds', 'loader_seconds', 'geometry_seconds', 'geometry_background_seconds',
                'input_seconds', 'update_seconds'):
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
    if geometry is not None:
        report['geometry_stats'] = (dict(backend='cpu', placement='workers',
            workers={str(k): v['geometry'] for k, v in backend.workers.items()})
            if cpu_reader else geometry.stats())
    if cpu_reader and args.heading == 'tensor':
        report['heading_stats'] = dict(backend='cpu', placement='workers',
            workers={str(k): v['heading'] for k, v in backend.workers.items()})
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
