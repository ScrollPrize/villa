"""Benchmark production snapshots of real patch directories; never edits sources."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import shutil
import statistics
import tempfile
import time

from service_editing import EditingWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('patches', type=Path)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--destination', type=Path, default=Path('/tmp'))
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if args.count < 1 or args.iterations < 1:
        parser.error('count and iterations must be positive')
    sources = sorted(p for p in args.patches.iterdir() if (p / 'meta.json').is_file())[:args.count]
    if not sources:
        parser.error('No patch directories found')
    size = sum(p.stat().st_size for source in sources for p in source.rglob('*') if p.is_file())
    profiler = cProfile.Profile()
    times = []
    with tempfile.TemporaryDirectory(prefix='spiral-snapshot-bench-', dir=args.destination) as temporary:
        root = Path(temporary)
        workspace = EditingWorkspace(args.patches, root / 'out', {}, lambda: None)
        for iteration in range(args.iterations):
            destination = root / str(iteration)
            start = time.perf_counter()
            if args.profile:
                profiler.enable()
            for index, source in enumerate(sources):
                workspace._copy(source, destination / str(index))
            if args.profile:
                profiler.disable()
            times.append(time.perf_counter() - start)
            shutil.rmtree(destination)
    if args.profile:
        pstats.Stats(profiler).sort_stats('cumulative').print_stats(12)
    print(json.dumps({'patches': [str(p) for p in sources], 'bytes': size,
                      'seconds': times, 'mean': statistics.mean(times),
                      'min': min(times), 'median': statistics.median(times),
                      'max': max(times)}, indent=2))


if __name__ == '__main__':
    main()
