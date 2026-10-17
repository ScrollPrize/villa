"""GPU chunk-cache miss latency on real CT, with warm filesystem caches.

Run via run_native.py to compare native builds without installing them. The
gather cases force many full chunk uploads with negligible sampling/output cost;
they are transfer stress cases, not recommended routes for sparse gathers.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from _common import environment
from bench_cuda_sample import stats
from bench_sample import DEFAULT_ARRAY, REGION
from vc_reader.reader import ZarrReader


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--array', default=DEFAULT_ARRAY)
    ap.add_argument('--reps', type=int, default=31)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--only', default='')
    ap.add_argument('--compare', type=Path)
    ap.add_argument('--build-dir', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if min(args.reps, args.warmup, args.threads) < 1:
        ap.error('reps, warmup and threads must be positive')
    env = environment()
    report = dict(environment=env, command=sys.argv, array=args.array,
                  reps=args.reps, warmup=args.warmup, threads=args.threads,
                  binary_sha256=hashlib.sha256(Path(env['native_extension']).read_bytes()).hexdigest(),
                  protocol='Warm filesystem, memory_cache=0; synchronous public calls including planning/allocation; '
                           'fresh clears GPU cache only (staging reused); resident preloads; '
                           'downloads/verification/clear/destruction excluded; seeded shuffled mode order.', rows=[])
    if args.build_dir:
        report['cmake_cache'] = (args.build_dir / 'CMakeCache.txt').read_text()
    previous = {} if not args.compare else {r['name']: r for r in json.loads(args.compare.read_text())['rows']}
    only = set(filter(None, args.only.split(',')))
    rng = np.random.default_rng(901)
    with ZarrReader.open(args.array, threads=args.threads, memory_cache=0) as reader, reader.cuda(cache_bytes=512 << 20) as gpu:
        origin = (np.asarray(REGION[0]) // reader.chunks + 1) * reader.chunks
        axes = np.array([[1, 0, 0], [0, np.cos(.35), -np.sin(.35)],
                         [0, np.sin(.35), np.cos(.35)]], np.float32)
        cases = {}
        for count in (1, 8, 32, 128):
            indices = np.array(list(np.ndindex(8, 4, 4))[:count])
            points = (origin + indices * reader.chunks + 17).astype(np.int64)
            request = gpu.prepare_points(points, integer=True)
            cases[f'chunks{count}'] = (lambda p=points: reader.read_points(p),
                                      lambda q=request: q.read(source='chunks'))
        for edge in (128, 256):
            cases[f'affine{edge}'] = (
                lambda e=edge: reader.read_affine((e,) * 3, origin + .25, axes),
                lambda e=edge: gpu.read_affine((e,) * 3, origin + .25, axes, source='chunks'))
        for name, (cpu, device) in cases.items():
            if only and name not in only:
                continue
            expected = cpu()
            # Warm all required full source chunks, including those only partially
            # touched by the CPU reference. No storage-cold claims are made.
            initial = device().numpy()
            np.testing.assert_allclose(initial, expected, atol=5e-5, rtol=1e-6)
            digest = hashlib.sha256(initial.tobytes()).hexdigest()
            if args.compare:
                assert previous[name]['sha256'] == digest, f'{name}: build output differs'
            times = {mode: [] for mode in ('fresh', 'resident')}
            uploaded = 0
            for rep in range(-args.warmup, args.reps):
                modes = list(times)
                rng.shuffle(modes)
                for mode in modes:
                    if mode == 'fresh':
                        gpu.clear_cache()
                    else:
                        device()
                    before = gpu.stats['uploaded_bytes']
                    start = time.perf_counter()
                    output = device()
                    elapsed = (time.perf_counter() - start) * 1000
                    transferred = gpu.stats['uploaded_bytes'] - before
                    if mode == 'resident':
                        assert transferred == 0
                    else:
                        uploaded = transferred
                    actual = output.numpy()
                    assert hashlib.sha256(actual.tobytes()).hexdigest() == digest
                    if rep >= 0:
                        times[mode].append(elapsed)
                    del output, actual
            row = dict(name=name, sha256=digest, uploaded_bytes=uploaded,
                       timings={mode: stats(t) for mode, t in times.items()})
            report['rows'].append(row)
            print(name + f' upload={uploaded / (1 << 20):.0f} MiB ' +
                  ' '.join(f'{mode}={np.median(t):.3f} ms' for mode, t in times.items()), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
