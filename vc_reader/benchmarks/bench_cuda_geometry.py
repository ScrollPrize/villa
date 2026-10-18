"""Resident flip/elastic timings and exact output/replay regression snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from bench_pipeline import environment, load_extensions, positive, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--shape', type=positive, nargs='+', default=(128, 128, 128))
    parser.add_argument('--reps', type=positive, default=40)
    parser.add_argument('--warmup', type=positive, default=5)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, help='save outputs and replay plans for exact comparison')
    parser.add_argument('--reference', type=Path, help='compare against a previous snapshot')
    parser.add_argument('--allow-differences', action='store_true', help='report numerical differences instead of requiring exact equality')
    args = parser.parse_args()
    args.backend = 'cuda'
    load_extensions(args.native_dir)
    from vc_reader import augment as A, cuda
    shape = tuple(args.shape)
    rng = np.random.default_rng(123)
    source = rng.random((1, *shape), dtype=np.float32)
    source.reshape(-1)[::11] = -0.0
    image = cuda.asarray(source)
    mask = cuda.asarray(rng.integers(0, 5, source.shape, dtype=np.int16))
    pipelines = {
        'flip_x': A.Pipeline([A.Flip(axes='x', p_axis=1)], backend='cuda'),
        'flip_all': A.Pipeline([A.Flip(p_axis=1)], backend='cuda'),
        'elastic': A.Pipeline([A.Elastic(sigma=12, magnitude=2, border=True)], backend='cuda'),
    }
    snapshots, validation = {}, {}
    # Outputs and serialized replay fields, for multiple fixed random seeds.
    for name, pipeline in pipelines.items():
        for seed in (0, 123, 456):
            result = pipeline({'image': image}, seed=seed)['image'].numpy()
            plan = pipeline.sample({'image': image}, seed=seed)
            replay = plan.apply({'image': image})['image'].numpy()
            assert result.tobytes() == replay.tobytes()
            key = f'{name}_{seed}'
            snapshots[key] = result
            snapshots[key+'_plan'] = np.frombuffer(plan._native.dump(), dtype=np.uint8)
            snapshots[key+'_mask'] = pipeline({'image': image, 'mask': mask}, seed=seed)['mask'].numpy()
            validation[key] = hashlib.sha256(result.tobytes()).hexdigest()
    differences = {}
    if args.reference:
        with np.load(args.reference) as reference:
            for key, value in snapshots.items():
                if reference[key].dtype != value.dtype or reference[key].shape != value.shape:
                    raise AssertionError(f'{key}: dtype/shape differs from reference')
                equal = reference[key].tobytes() == value.tobytes()
                differences[key] = dict(byte_identical=equal)
                if not key.endswith('_plan'):
                    delta = value.astype(np.float64) - reference[key].astype(np.float64)
                    differences[key].update(changed_values=int(np.count_nonzero(delta)),
                                            max_abs=float(np.abs(delta).max()), rmse=float(np.sqrt(np.mean(delta*delta))))
                if not equal and not args.allow_differences:
                    raise AssertionError(f'{key}: differs from reference')
            # Old plans retain their sampled field and must still replay exactly,
            # even when fresh sampling now uses different numerical precision.
            for name, pipeline in pipelines.items():
                for seed in (0, 123, 456):
                    key = f'{name}_{seed}'
                    plan = A.Plan(pipeline, pipeline._native.load_plan(reference[key+'_plan'].tobytes()))
                    replay = plan.apply({'image': image})['image'].numpy()
                    assert replay.tobytes() == reference[key].tobytes(), f'{key}: old replay changed'
    if args.snapshot:
        np.savez(args.snapshot, **snapshots)
    for pipeline in pipelines.values():
        for i in range(args.warmup):
            output = pipeline({'image': image}, seed=i)
            del output
    times = {name: [] for name in pipelines}
    names = list(pipelines)
    for rep in range(args.reps):
        for name in names[rep % len(names):] + names[:rep % len(names)]:
            start = time.perf_counter_ns()
            output = pipelines[name]({'image': image}, seed=123)
            times[name].append((time.perf_counter_ns()-start)/1e6)
            del output
    report = dict(command=[sys.executable, *sys.argv], shape=shape, reps=args.reps, warmup=args.warmup,
                  environment=environment(args, augmentation=True),
                  protocol='single resident float32 image; synchronous wall time; includes input preservation, validation, '
                           'field construction and transform; excludes upload, download and output destruction; rotated case order',
                  validation=validation, reference=str(args.reference) if args.reference else None, differences=differences,
                  results={name: dict(timing=summary(values), raw_ms=values) for name, values in times.items()})
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for name, result in report['results'].items():
        print(name, result['timing'], flush=True)


if __name__ == '__main__':
    main()
