"""Compare metadata queries on a fixed local catalog; never access asset storage."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc

from vc_reader import Catalog, SourceCollection


def measure(call, warmups, repetitions):
    for _ in range(warmups):
        call()
    times = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        result = call()
        times.append((time.perf_counter_ns() - start) / 1000)
    return {"count": len(result), "mean_us": statistics.mean(times),
            "p50_us": statistics.median(times),
            "p95_us": statistics.quantiles(times, n=20, method="inclusive")[18]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True, help='Local JSON or gzip catalog snapshot')
    parser.add_argument('--sample', default='PHercParis4')
    parser.add_argument('--volumes', nargs=6, help='Six explicit volume IDs; defaults to first six sorted IDs')
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path, help='Optional cProfile output, separate from timed queries')
    args = parser.parse_args()
    if args.repetitions < 2 or args.warmups < 0:
        parser.error('need at least two repetitions and nonnegative warmups')
    catalog = Catalog.from_file(args.metadata)
    sample = catalog.sample(args.sample)
    volumes = [sample.volume(v) for v in args.volumes] if args.volumes else sample.volume_list()[:6]
    if len(volumes) != 6:
        parser.error('sample must contain at least six volumes')
    start = time.perf_counter()
    collection = SourceCollection(catalog)
    initialization = time.perf_counter() - start
    rows = []
    for volume in volumes:
        expected = sorted((m.segment.id, m.labels.label_set, m.labels.target_volume_id) for m in volume.ink_labels())
        sources = collection.labels(volume=volume)
        actual = sorted((s.owner_id, s.label_set, s.registration.volume_id) for s in sources)
        if expected != actual:
            raise AssertionError(f'{volume.id}: discovery differs from specialized ink query')
        rows.append({
            'volume_id': volume.id, 'source_ids': [s.id for s in sources],
            'ink_labels': measure(volume.ink_labels, args.warmups, args.repetitions),
            'labels': measure(lambda: collection.labels(volume=volume), args.warmups, args.repetitions),
            'representations_level2': measure(lambda: collection.representations(volume=volume.level(2)),
                                              args.warmups, args.repetitions),
            'registered_volumes': measure(lambda: collection.registered_volumes(volume=volume),
                                          args.warmups, args.repetitions),
        })
    # Memory instrumentation runs separately so it does not affect timings.
    tracemalloc.start()
    measured = SourceCollection(catalog)
    for volume in volumes:
        measured.labels(volume=volume)
        measured.representations(volume=volume.level(2))
        measured.registered_volumes(volume=volume)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del measured
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        profiled = SourceCollection(catalog)
        for _ in range(args.repetitions):
            for volume in volumes:
                volume.ink_labels()
                profiled.labels(volume=volume)
                profiled.representations(volume=volume.level(2))
        profiler.disable()
        profiler.dump_stats(args.profile)
    report = {
        'command': sys.argv, 'python': sys.version, 'platform': platform.platform(),
        'build': 'stdlib-only; no native modules or CUDA used',
        'metadata': str(args.metadata.resolve()),
        'file_sha256': hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        'canonical_snapshot_sha256': collection.snapshot_sha256,
        'sample': sample.id, 'source_count': len(collection),
        'warmups': args.warmups, 'repetitions': args.repetitions,
        'initialization_seconds': initialization, 'python_retained_bytes': retained,
        'python_peak_bytes': peak, 'storage_requests': 0, 'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(collection)} sources; index {initialization:.3f}s; Python peak {peak / 2**20:.2f} MiB')
    for row in rows:
        before, after = row['ink_labels'], row['labels']
        print(f"{row['volume_id']}: ink_labels {before['mean_us']:.2f} / {before['p50_us']:.2f} / "
              f"{before['p95_us']:.2f} us; labels {after['mean_us']:.2f} / {after['p50_us']:.2f} / "
              f"{after['p95_us']:.2f} us")
    print(args.output)


if __name__ == '__main__':
    main()
