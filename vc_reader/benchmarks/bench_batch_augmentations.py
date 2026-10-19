"""Measure resident augmentation batches on a real CT crop; reading is untimed."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from bench_pipeline import load_extensions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--array', required=True)
    parser.add_argument('--level', default='2')
    parser.add_argument('--start', type=int, nargs=3, default=(9000, 3600, 3600))
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16])
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.size, args.reps, *args.batches) < 1 or args.warmup < 0:
        parser.error('sizes and repetitions must be positive')
    load_extensions(args.native_dir)
    import torch
    from vc_reader import augment as A
    from vc_reader.reader import ZarrReader
    with ZarrReader.open(args.array, level=args.level, threads=1) as reader:
        image = reader.read_region(args.start, (args.size,)*3, dtype=np.float32)[None]/np.float32(255)
    cases = {
        'brightness': [A.BrightnessAdd(.1)],
        'noise': [A.GaussianNoise(.03)],
        'blur': [A.GaussianBlur(.9)],
        'normalize': [A.Normalize('zscore')],
        'percentile': [A.Normalize('percentile')],
        'affine': [A.Affine(degrees=5)],
        'elastic': [A.Elastic(magnitude=2)],
        'equalize': [A.Equalize(bins=64)],
        'fft': [A.GibbsRinging()],
        'pipeline': [A.Flip(), A.BrightnessMultiply((.8, 1.2)), A.GaussianBlur(.9), A.GaussianNoise(.03)],
    }
    if args.cases:
        if set(args.cases)-cases.keys():
            parser.error('unknown case')
        cases = {k: cases[k] for k in args.cases}
    report = {'command': [sys.executable, *sys.argv], 'gpu': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'input': {'array': args.array, 'level': args.level,
              'start': args.start, 'shape': list(image.shape)}, 'results': {}}
    for count in args.batches:
        source = torch.from_numpy(np.repeat(image[None], count, axis=0)).cuda()
        seeds = list(range(100, 100+count))
        for name, transforms in cases.items():
            pipeline = A.Pipeline(transforms, backend='cuda')
            for _ in range(args.warmup):
                pipeline.batch({'image': source}, seeds=seeds)
            times = []
            for _ in range(args.reps):
                start = time.perf_counter_ns()
                output = pipeline.batch({'image': source}, seeds=seeds)['image']
                times.append((time.perf_counter_ns()-start)/1e6)
                del output
            row = {'batch': count, 'raw_ms': times, 'mean_ms': float(np.mean(times)),
                   'p50_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95)),
                   'min_ms': min(times), 'max_ms': max(times)}
            if args.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profile:
                    pipeline.batch({'image': source}, seeds=seeds)
                kernels = {}
                for event in profile.events():
                    if event.device_type == torch.autograd.DeviceType.CUDA and not event.name.startswith('Memcpy'):
                        kernels[event.name] = kernels.get(event.name, 0)+1
                row['gpu_events'] = kernels
                row['gpu_event_count'] = sum(kernels.values())
            report['results'][f'{name}/B{count}'] = row
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(f'{name}/B{count}: {row["p50_ms"]:.3f} ms', flush=True)


if __name__ == '__main__':
    main()
