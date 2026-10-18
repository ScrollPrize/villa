"""Time every README augmentation on the same real CT crop, without rendering.

Reuses the preview recipe/input helpers. CPU, CUDA uploaded and CUDA resident
use synchronous public calls; output validation and disposal are not timed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import subprocess
import time

import numpy as np

from bench_pipeline import environment, load_extensions, nonnegative, positive, summary

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from augment_examples import constructor, inputs, recipes, workload


def fingerprint(sample):
    return {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                      sha256=hashlib.sha256(value.tobytes()).hexdigest())
            for key, value in sample.items()}


def compare(actual, expected, *, atol=5e-5, rtol=1e-5):
    if actual.keys() != expected.keys():
        raise AssertionError('output keys differ')
    report = {}
    for key, reference in expected.items():
        value = actual[key]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise AssertionError(f'{key}: output shape/dtype differs')
        maximum, mismatched, changed = 0., 0, 0
        for start in range(0, value.size, 1 << 20):
            a = value.ravel()[start:start+(1 << 20)]
            b = reference.ravel()[start:start+(1 << 20)]
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise AssertionError(f'{key}: nonfinite output')
            delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
            maximum = max(maximum, float(delta.max(initial=0)))
            changed += int(np.count_nonzero(a != b))
            close = delta <= atol+rtol*np.abs(b) if value.dtype.kind == 'f' else a == b
            mismatched += int(np.count_nonzero(~close))
        report[key] = dict(max_abs=maximum, mismatched=mismatched, changed=changed,
                           shape=list(value.shape), dtype=str(value.dtype))
    return report


def update_readme(path, results):
    text = path.read_text()
    for name, row in results.items():
        times = [row['timings'][mode]['p50_ms'] for mode in ('cpu', 'cuda_host', 'cuda_resident')]
        cpu, host, device = times
        replacement = f'| {cpu:.3f} ms | {host:.3f} ms | {device:.3f} ms | {cpu/host:.1f}× | {cpu/device:.1f}× |'
        pattern = (rf'(<summary><strong>{re.escape(name)}</strong></summary>.*?'
                   r'\| CPU \| CUDA uploaded \| CUDA resident \| Uploaded speedup \| Resident speedup \|\n'
                   r'\|[^\n]+\|\n)\|[^\n]+\|')
        text, count = re.subn(pattern, lambda match: match[1]+replacement, text, count=1, flags=re.DOTALL)
        if count != 1:
            raise ValueError(f'{name}: missing README timing table')
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--array', required=True)
    parser.add_argument('--level', default='2')
    parser.add_argument('--start', type=int, nargs=3, default=(9000, 3600, 3600))
    parser.add_argument('--size', type=positive, default=128)
    parser.add_argument('--reps', type=positive, default=5)
    parser.add_argument('--warmup', type=positive, default=1)
    parser.add_argument('--seed', type=nonnegative, default=123)
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--readme', type=Path, default=ROOT / 'README.md')
    parser.add_argument('--update-readme', action='store_true', help='replace timing rows after validation')
    parser.add_argument('--allow-differences', action='store_true', help='retain and report numerical mismatches')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.update_readme and args.size != 128:
        parser.error('README timing rows describe 128-cubed inputs')
    args.backend = 'both'
    load_extensions(args.native_dir)
    from vc_reader import augment as A, cuda
    from vc_reader.reader import ZarrReader
    if not cuda.is_available():
        parser.error('CUDA reader and augmentation builds and a device are required')
    selected = recipes(args.readme)
    if args.only:
        if set(args.only)-selected.keys():
            parser.error('unknown augmentation name')
        selected = {name: recipe for name, recipe in selected.items() if name in args.only}
    shape = (args.size,)*3
    reference_start = (args.start[0]+384, *args.start[1:])
    with ZarrReader.open(args.array, level=args.level, threads=1) as reader:
        for origin in (args.start, reference_start):
            if len(reader.shape) != 3 or any(s < 0 or s+n > edge for s, n, edge in zip(origin, shape, reader.shape)):
                parser.error('input/reference crop must fit in the 3D volume')
        image = reader.read(args.start, shape, dtype=np.float32)[None]/np.float32(255)
        reference = reader.read(reference_start, shape, dtype=np.float32)[None]/np.float32(255)
    if not np.isfinite(image).all() or not np.isfinite(reference).all() or np.ptp(image) == 0:
        parser.error('input must be finite and contain visible CT signal')
    samples = inputs(image)
    report = dict(command=[sys.executable, *sys.argv],
                  recorded_at=datetime.now(timezone.utc).isoformat(),
                  git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  thread_env={key: os.environ.get(key) for key in
                              ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  environment=environment(args, augmentation=True),
                  protocol='one CPU thread; compact=True; synchronous owning public calls; '
                           'seed + repetition; deterministically shuffled case/backend order; '
                           'reads, preparation, validation, output disposal and rendering excluded; '
                           'CUDA uploaded includes transfers; resident excludes initial upload/final download; '
                           'public-call input preservation and validation included',
                  inputs={key: fingerprint(sample) for key, sample in samples.items()},
                  reference_start=reference_start, reference=fingerprint({'reference': reference}),
                  readme_sha256=hashlib.sha256(args.readme.read_bytes()).hexdigest(),
                  validation=dict(atol=5e-5, rtol=1e-5, integer='exact', mismatched_cases=[]), results={})
    names = list(selected)
    random.Random(21017).shuffle(names)
    for name in names:
        case = workload(name)
        source = samples[case.sample]
        transform = constructor(name, selected[name], reference, A)
        pipelines = {mode: A.Pipeline([transform], backend='cpu' if mode == 'cpu' else 'cuda',
                                     targets=dict(image=A.Image(), mask=A.Label(), surface=A.CoordinateField()))
                     for mode in ('cpu', 'cuda_host', 'cuda_resident')}
        device = {key: cuda.asarray(value) for key, value in source.items()}
        measured = {mode: [] for mode in pipelines}
        comparisons = []
        for rep in range(-args.warmup, args.reps):
            modes = list(pipelines)
            random.Random(291+rep).shuffle(modes)
            outputs = {}
            for mode in modes:
                start = time.perf_counter_ns()
                value = pipelines[mode](device if mode == 'cuda_resident' else source,
                                        seed=args.seed+max(0, rep))
                elapsed = (time.perf_counter_ns()-start)/1e6
                if rep >= 0:
                    measured[mode].append(elapsed)
                outputs[mode] = {key: array.numpy() for key, array in value.items()} if mode == 'cuda_resident' else value
                del value
            detail = {mode: compare(outputs[mode], outputs['cpu']) for mode in ('cuda_host', 'cuda_resident')}
            # Upload/resident routing should not change the CUDA transform.
            resident = compare(outputs['cuda_resident'], outputs['cuda_host'], atol=0, rtol=0)
            if any(v['mismatched'] for v in resident.values()):
                raise AssertionError(f'{name}: uploaded/resident CUDA outputs differ')
            if rep >= 0:
                comparisons.append(dict(seed=args.seed+rep, **detail))
            del outputs
        if fingerprint(source) != report['inputs'][case.sample]:
            raise AssertionError(f'{name}: source mutated')
        row = dict(recipe=selected[name], sample=case.sample,
                   timings={mode: summary(values) for mode, values in measured.items()},
                   raw_ms=measured, comparisons=comparisons)
        mismatch = any(v['mismatched'] for item in comparisons for mode in ('cuda_host', 'cuda_resident')
                       for v in item[mode].values())
        if mismatch:
            report['validation']['mismatched_cases'].append(name)
        report['results'][name] = row
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(name, {mode: round(t['p50_ms'], 3) for mode, t in row['timings'].items()},
              'CPU/CUDA differences' if mismatch else 'validated', flush=True)
        del pipelines, device
    if report['validation']['mismatched_cases'] and not args.allow_differences:
        raise SystemExit('Numerical differences recorded; inspect validation before publishing timings')
    if args.update_readme:
        update_readme(args.readme, report['results'])
    print(f'Completed {len(selected)} cases: {args.output}', flush=True)


if __name__ == '__main__':
    main()
