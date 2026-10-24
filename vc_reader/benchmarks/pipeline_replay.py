"""Replay a reader report through CPU/CUDA normalization and transforms."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from extension_build import load_extensions


def restore_requests(source, names, minimum):
    """Restore exact recorded grids, checking surface geometry fingerprints."""
    from bench_pipeline import surface_workload

    if source.get('suite') != 'readers' or source.get('schema_version') != 3:
        raise ValueError('--requests-from requires a schema-3 reader report')
    missing = set(names) - source['workloads'].keys()
    if missing:
        raise ValueError(f'workloads absent from reader report: {sorted(missing)}')
    templates = None
    if 'surface' in names:
        config = SimpleNamespace(**source['config'])
        config.surface = Path(config.surface)
        templates, description = surface_workload(config, source['input']['shape'])
        expected = source['surface']
        if description['files_sha256'] != expected['files_sha256'] or json.dumps(description['tiles'], sort_keys=True) != json.dumps(expected['tiles'], sort_keys=True):
            raise ValueError('surface fixture/geometry differs from the recorded reader requests')
    restored = {}
    for name in names:
        row = source['workloads'][name]
        if row['coverage']['unique_chunks'] < minimum:
            raise ValueError(f'{name}: recorded coverage is below {minimum} chunks')
        if row['coverage']['requests'] != len(row['requests']):
            raise ValueError(f'{name}: inconsistent request count in reader report')
        if name == 'surface':
            restored[name] = [[(tuple(item['shape']), item['coordinate_origin'],
                                templates[item['template']][0][2])] for item in row['requests']]
        else:
            restored[name] = [[(tuple(shape), origin, np.asarray(axes, dtype=np.float32))
                               for shape, origin, axes in batch] for batch in row['requests']]
    return restored


def image_shape(name, shape):
    # Oblique planes have a singleton sampling-depth axis, but are 2D images.
    return (1, *shape[1:]) if name == 'plane' else (1, *shape)


def run_request(reader, transform, name, grids, seeds, args, *, device, host_outputs=None):
    if device and args.cuda_cache == 'clear':
        reader.clear_cache()
    kwargs = {'source': 'chunks'} if device else {}
    start = time.perf_counter_ns()
    if name == 'surface':
        images = [reader.sample(g[2], coordinate_origin=g[1], order=1, dtype=np.float32, **kwargs)
                  for g in grids]
    elif name in ('plane', 'rotated_box'):
        images = reader.read_affine_many(grids, order=1, dtype=np.float32, **kwargs)
    else:
        images = reader.read_regions([(tuple(int(v) for v in g[1]), g[0]) for g in grids],
                                     dtype=np.float32, **kwargs)
    read_done = time.perf_counter_ns()
    if device and args.cuda_transforms == 'batched':
        import torch

        tensors = [torch.from_dlpack(image).reshape(image_shape(name, grid[0]))
                   for image, grid in zip(images, grids)]
        batch = tensors[0].unsqueeze(0) if len(tensors) == 1 else torch.stack(tensors)
        outputs = transform.batch({'image': batch}, seeds=seeds)['image']
    else:
        outputs = [transform({'image': image.reshape(image_shape(name, grid[0]))}, seed=seed,
                             inplace=True)['image'] for image, grid, seed in zip(images, grids, seeds)]
    transformed = time.perf_counter_ns()
    if device and args.cuda_output == 'host':
        if host_outputs is not None:
            import torch

            shapes = tuple(tuple(value.shape) for value in outputs)
            destination = host_outputs.get(shapes)
            if destination is None:
                # Shape-changing recipes may need a new buffer on first use.
                # Subsequent requests and timed passes reuse the warmup buffers.
                destination = ([torch.empty(shape, dtype=torch.float32, pin_memory=True) for shape in shapes]
                               if len(set(shapes)) > 1 else
                               torch.empty((len(shapes), *shapes[0]), dtype=torch.float32, pin_memory=True))
                host_outputs[shapes] = destination
            if isinstance(outputs, torch.Tensor) and isinstance(destination, torch.Tensor):
                destination.copy_(outputs, non_blocking=True)
            else:
                for target, value in zip(destination, outputs):
                    tensor = value if isinstance(value, torch.Tensor) else torch.from_dlpack(value)
                    target.copy_(tensor, non_blocking=True)
            torch.cuda.current_stream(args.device).synchronize()
            outputs = ([value.numpy() for value in destination] if isinstance(destination, list)
                       else destination.numpy())
        elif args.cuda_transforms == 'batched':
            outputs = ([image.cpu().numpy() for image in outputs] if isinstance(outputs, list)
                       else outputs.cpu().numpy())
        else:
            outputs = [image.numpy() for image in outputs]
    done = time.perf_counter_ns()
    return outputs, dict(read_sample=(read_done - start) / 1e6,
                         normalization_transform=(transformed - read_done) / 1e6,
                         download=(done - transformed) / 1e6 if device and args.cuda_output == 'host' else 0.,
                         total=(done - start) / 1e6)


def validate_outputs(outputs, grids, name):
    """Bound temporary validation storage; return CPU/CUDA numerical differences."""
    from vc_reader import cuda

    arrays = {}
    for mode, values in outputs.items():
        if len(values) != len(grids):
            raise ValueError(f'{name}/{mode}: invalid transformed output count')
        arrays[mode] = [v.numpy() if isinstance(v, cuda.DeviceArray) else
                        v.detach().cpu().numpy() if hasattr(v, 'detach') else v for v in values]
        for value, grid in zip(arrays[mode], grids):
            if value.shape != image_shape(name, grid[0]) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f'{name}/{mode}: invalid transformed shape, dtype, or values')
    if 'cpu' not in arrays or 'cuda' not in arrays:
        return {}
    maximum, square_sum, count = 0., 0., 0
    for actual, expected in zip(arrays['cuda'], arrays['cpu']):
        a, b = actual.ravel(), expected.ravel()
        for start in range(0, a.size, 1 << 18):
            x, y = a[start:start + (1 << 18)], b[start:start + (1 << 18)]
            delta = np.abs(x - y)
            maximum = max(maximum, float(delta.max(initial=0)))
            square_sum += float(np.sum(delta * delta, dtype=np.float32))
            count += delta.size
            if not np.all(delta <= np.float32(5e-5) + np.float32(1e-5) * np.abs(y)):
                raise AssertionError(f'{name}: CPU/CUDA transforms differ (max_abs={maximum})')
    return dict(max_abs=maximum, rmse=math.sqrt(square_sum / count))


def benchmark_replay(args):
    from bench_pipeline import environment, heavy_transforms, summary

    if args.build_dir:
        load_extensions(args.build_dir.resolve())
    from vc_reader import cuda, transforms as A
    from vc_reader.reader import ZarrReader

    raw_source = args.requests_from.read_bytes()
    source = json.loads(raw_source)
    names = list(dict.fromkeys(args.workloads))
    workloads = restore_requests(source, names, args.min_unique_chunks)
    modes = ['cpu', 'cuda'] if args.backend == 'both' else [args.backend]
    if 'cuda' in modes and not cuda.is_available():
        raise ValueError('CUDA reader and transform build/device required')
    transforms = heavy_transforms() if args.transform == 'heavy' else [
        A.Flip(axes='x', p_axis=1), A.BrightnessMultiply(1.1), A.Gamma(.9), A.GaussianBlur(1)]
    transforms = [A.Normalize(args.normalization), *transforms]
    pipelines = {mode: A.Pipeline(transforms, backend=mode, compact=False) for mode in modes}
    location = args.array or source['input']['location']
    level = args.level if args.level is not None else source['input']['level']
    if location != source['input']['location'] or level != source['input']['level']:
        raise ValueError('replay must use the reader report\'s array and level')
    report = dict(schema_version=3, suite='pipeline', command=[sys.executable, *sys.argv],
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  environment=environment(args, transform=True), input=source['input'], workloads={},
                  transforms=[repr(t) for t in transforms],
                  request_replay=dict(path=str(args.requests_from), sha256=hashlib.sha256(raw_source).hexdigest(),
                                      geometry_sha256={name: hashlib.sha256(json.dumps(source['workloads'][name]['requests'],
                                                       sort_keys=True).encode()).hexdigest() for name in names}),
                  protocol=dict(included='same reader calls/grids, float32 output, normalization and transforms, '
                                         'allocation; final CUDA download when host output is selected',
                                excluded='setup, geometry reconstruction, validation/warmup, cache clearing, output disposal',
                                warmup='first complete warmup pass also validates every request; no separate full validation pass',
                                layout='channel-first; plane and surface are 2D; other crops are 3D',
                                batching=('batched reads; CUDA Pipeline.batch per reader request (8 crops for ml_batch, 1 otherwise), '
                                          'including device batch assembly; CPU sequential per-crop transforms'
                                          if args.cuda_transforms == 'batched' else
                                          'batched reads; independent sequential per-crop transforms on both backends'),
                                transforms='combined normalization + transform Pipeline, compact=False; per-crop calls use inplace=True on fresh reader buffers',
                                seeds='A.seed(seed, workload, request index, crop index), reused across backends/repetitions',
                                caches='warm OS cache; configured decoded host budget; CUDA cache policy applied before each request',
                                endpoints='CPU host; CUDA ' + args.cuda_output,
                                host_memory=('pinned output buffers populated during warmup and reused; '
                                             'returned host views consumed before the next request'
                                             if args.cuda_host_memory == 'pinned' else
                                             'fresh host allocations per request'),
                                throughput='float32 output bytes/voxels divided by summed timed request seconds; MB=10^6 bytes, Mvox=10^6 voxels',
                                order='backend order rotates each timed repetition',
                                coverage='exact saved reader geometry and nonzero-region coverage, unchanged by replay'))
    if 'surface' in names:
        report['surface'] = source['surface']

    def save():
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    with ExitStack() as stack:
        host = stack.enter_context(ZarrReader.open(location, level=level, threads=args.threads,
                                                   memory_cache=args.memory_cache_mib << 20))
        if list(host.shape) != source['input']['shape'] or list(host.chunks) != source['input']['chunks'] or str(host.dtype) != source['input']['dtype']:
            raise ValueError('input array metadata differs from reader report')
        readers = {'cpu': host}
        if 'cuda' in modes:
            readers['cuda'] = stack.enter_context(host.cuda(cache_bytes=args.cuda_cache_mib << 20, device=args.device))
        for name, requests in workloads.items():
            host_outputs = {} if args.cuda_host_memory == 'pinned' and args.cuda_output == 'host' else None
            original = source['workloads'][name]
            seeds = [[A.seed(args.seed, name, index, item) for item in range(len(batch))]
                     for index, batch in enumerate(requests)]
            row = dict(requests=original['requests'], coverage=original['coverage'], dtype='float32',
                       sampling=original['sampling'], results={},
                       validation=dict(atol=5e-5, rtol=1e-5, requests=[]))
            report['workloads'][name] = row
            print(f'{name}: replaying {len(requests)} requests, {row["coverage"]["unique_chunks"]} unique chunks', flush=True)
            progress = time.monotonic()
            for warmup in range(args.warmup):
                for index, grids in enumerate(requests):
                    outputs = {}
                    for mode in modes:
                        outputs[mode], _ = run_request(readers[mode], pipelines[mode], name, grids, seeds[index],
                                                      args, device=mode == 'cuda', host_outputs=host_outputs)
                    if warmup == 0:
                        comparison = validate_outputs(outputs, grids, name)
                        row['validation']['requests'].append(dict(request=index, **comparison))
                    del outputs
                    if time.monotonic() - progress >= 30:
                        print(f'{name}: warmup/validation {warmup + 1}/{args.warmup}, request {index + 1}/{len(requests)}', flush=True)
                        progress = time.monotonic()
            print(f'{name}: all requests validated and warmup complete', flush=True)
            before = dict(host=host.stats(), cuda=readers['cuda'].stats if 'cuda' in modes else None)
            records = {mode: [] for mode in modes}
            for rep in range(args.reps):
                order = modes[rep % len(modes):] + modes[:rep % len(modes)]
                for index, grids in enumerate(requests):
                    for mode in order:
                        outputs, times = run_request(readers[mode], pipelines[mode], name, grids, seeds[index],
                                                     args, device=mode == 'cuda', host_outputs=host_outputs)
                        records[mode].append(dict(repetition=rep, request=index, **times))
                        del outputs
                    if time.monotonic() - progress >= 30:
                        print(f'{name}: timed pass {rep + 1}/{args.reps}, request {index + 1}/{len(requests)}', flush=True)
                        progress = time.monotonic()
                voxels = sum(math.prod(g[0]) for batch in requests for g in batch) * (rep + 1)
                crops = sum(len(batch) for batch in requests) * (rep + 1)
                for mode, samples in records.items():
                    stages = ('read_sample', 'normalization_transform', 'download', 'total')
                    seconds = sum(s['total'] for s in samples) / 1000
                    row['results'][mode] = dict(timings={stage: summary([s[stage] for s in samples]) for stage in stages},
                                               raw_ms=samples, samples=len(samples), total_seconds=seconds,
                                               requests_per_second=len(samples)/seconds, crops_per_second=crops/seconds,
                                               mvoxels_per_second=voxels/seconds/1e6,
                                               output_mbytes_per_second=voxels*4/seconds/1e6)
                row['completed_repetitions'] = rep + 1
                row['cache_stats'] = dict(before=before, after=dict(host=host.stats(), cuda=readers['cuda'].stats if 'cuda' in modes else None))
                row['speedups'] = ({'cpu_over_cuda': row['results']['cpu']['timings']['total']['p50_ms'] /
                                                    row['results']['cuda']['timings']['total']['p50_ms']}
                                   if args.backend == 'both' else {})
                save()
                print(f'{name}: timed pass {rep + 1}/{args.reps} complete; ' + ', '.join(
                    f"{mode}={row['results'][mode]['total_seconds']:.3f} s, "
                    f"{row['results'][mode]['output_mbytes_per_second']:.2f} MB/s, "
                    f"{row['results'][mode]['mvoxels_per_second']:.2f} Mvox/s" for mode in modes), flush=True)
    return report
