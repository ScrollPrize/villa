"""Compare resident CUDA batches with per-crop execution on saved reader requests.

Both routes preserve inputs and copy into the same reused pinned host buffer.
Reading/upload and buffer preparation are excluded; public transform overhead,
output wrapping, host copies, and completion are timed separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from extension_build import load_extensions
from bench_pipeline import heavy_transforms, positive, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--requests-from', type=Path, default=Path('docs/benchmarks/readers-5000.json'))
    parser.add_argument('--requests', type=positive, default=4)
    parser.add_argument('--crop-size', type=positive, default=128,
                        help='center subcrop edge for transform scaling experiments; must fit the saved crops')
    parser.add_argument('--reps', type=positive, default=10)
    parser.add_argument('--warmup', type=positive, default=2)
    parser.add_argument('--stages', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    load_extensions(args.build_dir)
    import torch
    from vc_reader import cuda, transforms as A
    from vc_reader.reader import ZarrReader

    if not cuda.is_available():
        parser.error('CUDA reader and transform build/device required')
    torch.set_num_threads(1)
    source_bytes = args.requests_from.read_bytes()
    source = json.loads(source_bytes)
    requests = source['workloads']['ml_batch']['requests'][:args.requests]
    inputs, seeds = [], []
    with ZarrReader.open(source['input']['location'], level=source['input']['level'],
                         threads=8, memory_cache=0) as reader:
        with reader.cuda(cache_bytes=512 << 20) as gpu:
            for index, grids in enumerate(requests):
                images = gpu.read_regions([(tuple(int(v) for v in origin), tuple(shape))
                                           for shape, origin, axes in grids], dtype=np.float32, source='chunks')
                tensors = []
                for image in images:
                    if any(n < args.crop_size for n in image.shape):
                        parser.error('--crop-size must fit every saved crop')
                    slices = tuple(slice((n-args.crop_size)//2, (n-args.crop_size)//2+args.crop_size)
                                   for n in image.shape)
                    tensors.append(torch.from_dlpack(image)[slices].unsqueeze(0))
                inputs.append(torch.stack(tensors) / 255.)
                seeds.append([A.seed(source['config']['seed'], 'ml_batch', index, i) for i in range(len(grids))])
    views = [[cuda.wrap(v) for v in batch] for batch in inputs]
    torch.cuda.synchronize()
    recipe = [A.Normalize('minmax'), *heavy_transforms()]
    cases = {'full': recipe}
    if args.stages:
        cases.update({repr(t).split('(', 1)[0]: [t] for t in recipe})
    report = dict(source_sha256=hashlib.sha256(source_bytes).hexdigest(), requests=requests,
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  extension_sha256={name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                    for name, module in (('reader', cuda._reader), ('transforms', A._transforms))},
                  protocol='GPU-resident float32 inputs /255; same per-crop seeds; preserve inputs on both routes; '
                           'same reused pinned host destination; rotated execution order; synchronized public calls; '
                           'reads/upload/validation/output disposal excluded', results={})
    for name, transforms in cases.items():
        pipeline = A.Pipeline(transforms, backend='cuda', compact=False)
        host = torch.empty(inputs[0].shape, dtype=torch.float32, pin_memory=True)
        host.zero_()
        records = {mode: [] for mode in ('batched', 'unbatched')}
        maximum = 0.
        for rep in range(-args.warmup, args.reps):
            order = list(records)
            if rep % 2: order.reverse()
            for index, batch in enumerate(inputs):
                reference = None
                for mode in order:
                    torch.cuda.nvtx.range_push(f'{name}/{mode}')
                    start = time.perf_counter_ns()
                    if mode == 'batched':
                        output = pipeline.batch({'image': batch}, seeds=seeds[index])['image']
                    else:
                        output = [torch.from_dlpack(pipeline({'image': v}, seed=seed)['image'])
                                  for v, seed in zip(views[index], seeds[index])]
                    transformed = time.perf_counter_ns()
                    if isinstance(output, list):
                        for target, value in zip(host, output):
                            target.copy_(value, non_blocking=True)
                    else:
                        host.copy_(output, non_blocking=True)
                    torch.cuda.current_stream().synchronize()
                    done = time.perf_counter_ns()
                    torch.cuda.nvtx.range_pop()
                    if rep >= 0:
                        records[mode].append(dict(request=index, repetition=rep,
                            transform=(transformed-start)/1e6, download=(done-transformed)/1e6, total=(done-start)/1e6))
                    elif rep == -args.warmup:
                        actual = host.numpy()
                        if not np.isfinite(actual).all(): raise AssertionError(f'{name}: nonfinite output')
                        if reference is None: reference = actual.copy()
                        else:
                            maximum = max(maximum, float(np.max(np.abs(actual-reference))))
                            np.testing.assert_allclose(actual, reference, atol=5e-5, rtol=1e-5)
                    del output
        row = dict(transforms=[repr(t) for t in transforms], max_abs=maximum, results={})
        voxels = sum(v.numel() for v in inputs) * args.reps
        for mode, samples in records.items():
            seconds = sum(s['total'] for s in samples)/1000
            row['results'][mode] = dict(raw_ms=samples, total_seconds=seconds,
                timings={stage: summary([s[stage] for s in samples]) for stage in ('transform','download','total')},
                mvoxels_per_second=voxels/seconds/1e6, output_mbytes_per_second=voxels*4/seconds/1e6)
        report['results'][name] = row
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(name, {mode: {k: round(v['mean_ms'],3) for k,v in row['results'][mode]['timings'].items()}
                     for mode in records}, flush=True)


if __name__ == '__main__':
    main()
