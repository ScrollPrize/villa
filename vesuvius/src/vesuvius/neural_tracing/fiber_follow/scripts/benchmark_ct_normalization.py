"""Reproducible CPU calibration/normalization check on the reviewed real CT crops.

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/benchmark_ct_normalization.py
"""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import numba

from vesuvius.neural_tracing.fiber_follow.shared.ct_normalization import normalize_ct, prepare_normalization
from vesuvius.neural_tracing.fiber_follow.regression.datasets import ct_source_spec, read_dataset_config


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root/'configs/mixed_ct_datasets.json')
    parser.add_argument('--review', type=Path, default=root/'output/ct_normalization_review/report.json')
    parser.add_argument('--out', type=Path, default=root/'output/ct_normalization_validation')
    parser.add_argument('--repeats', type=int, default=40)
    args = parser.parse_args()
    config, _ = read_dataset_config(args.config)
    specs = [ct_source_spec(s, config['cache_dir']) for s in config['sources']]
    started = time.perf_counter()
    document = prepare_normalization(args.out, specs)
    calibration_seconds = time.perf_counter()-started
    cases = json.loads(args.review.read_text())['cases']
    report = dict(calibration_seconds=calibration_seconds, repeats=args.repeats, warmup=3,
                  cpu=platform.processor(), machine=platform.machine(), numba=numba.__version__,
                  numpy=np.__version__, results=[],
                  scope='One CPU thread; warmed kernel; excludes I/O/interpolation and buffer allocation. '
                        'Raw copy baseline; normalization mutates an existing interpolated float32 buffer.')
    for source, spec in zip(config['sources'], specs):
        for case in [c for c in cases if c['source'] == source['name']]:
            raw = np.array(np.memmap(case['path'], dtype=np.uint8, mode='r', shape=(128,)*3))
            image = raw[:120,:104,:104].astype(np.float32)
            image = np.ascontiguousarray((.75*image+.25*np.roll(image, 1, axis=2))/255.)
            for shape in ((120,104,104), (8,65,65)):
                original = np.ascontiguousarray(image[:shape[0], :shape[1], :shape[2]])
                output = np.empty_like(original)
                for method in ('raw_copy', 'foreground_mad'):
                    samples = []
                    for i in range(args.repeats+3):
                        np.copyto(output, original)
                        started = time.perf_counter_ns()
                        if method == 'raw_copy':
                            np.copyto(output, original)
                        else:
                            normalize_ct(output, spec.ct_normalization)
                        elapsed = (time.perf_counter_ns()-started)/1e6
                        if i >= 3:
                            samples.append(elapsed)
                    assert np.isfinite(output).all()
                    report['results'].append(dict(source=source['name'], case=case['label'], shape=shape,
                        method=method, mean_ms=float(np.mean(samples)), median_ms=float(np.median(samples)),
                        p95_ms=float(np.percentile(samples, 95))))
    started = time.perf_counter()
    assert prepare_normalization(args.out, specs, resume=document) == document
    report['resume_seconds'] = time.perf_counter()-started
    (args.out/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
