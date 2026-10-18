"""Profile metadata indexing and six direct scroll-volume navigation requests."""
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

from vc_reader import Catalog


def measure(call, warmups=2, repetitions=10):
    for _ in range(warmups):
        call()
    times = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        result = call()
        times.append((time.perf_counter_ns()-start)/1000)
    return dict(count=len(result), mean_us=statistics.mean(times), p50_us=statistics.median(times),
                p95_us=statistics.quantiles(times, n=20, method='inclusive')[18], samples_us=times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--sample', default='PHercParis4')
    parser.add_argument('--volumes', nargs=6)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    args = parser.parse_args()
    catalog = Catalog.from_file(args.metadata)
    def build_index():
        return Catalog(catalog.raw)._index.sources
    indexing = measure(build_index)
    scroll = catalog.samples[args.sample]
    volumes = args.volumes or [v.id for v in scroll.volumes.filter(include_superseded=True)][:6]
    if len(volumes) != 6:
        parser.error('six volume IDs are required')
    rows = []
    for identifier in volumes:
        def navigate():
            return list(scroll.on_volume(identifier).labels.ink)
        # A direct relationship scan is an independent metadata-only baseline.
        def scan():
            return [a for segment in scroll.record.segments.values() for a in segment.artifacts
                    if a.type == 'ink-labels' and a.target_volume_id == identifier]
        active = navigate()
        assert {a.artifact_id for a in active} <= {a.id for a in scan()}
        rows.append(dict(volume=identifier, source_ids=[a.id for a in active],
                         scan=measure(scan), navigation=measure(navigate)))
    tracemalloc.start()
    measured = Catalog(catalog.raw)
    len(measured._index.sources)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(build_index)
        profiler.dump_stats(args.profile)
    report = dict(command=sys.argv, platform=platform.platform(), python=sys.version,
                  build='stdlib only', metadata=str(args.metadata), sha256=hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
                  warmups=2, repetitions=10, requests=6, indexing=indexing, python_retained_bytes=retained,
                  python_peak_bytes=peak, array_requests=0, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(indexing=indexing['mean_us'], navigation_mean_us=statistics.mean(r['navigation']['mean_us'] for r in rows)), indent=2))


if __name__ == '__main__':
    main()
