"""Real CT/ink paired reads versus independently opened arrays, six fixed crops."""
import argparse
import cProfile
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

STARTS = ((18576, 7984), (19344, 14192), (19856, 15696), (20496, 13232), (22128, 11472), (23152, 7984))


def summary(times):
    return dict(mean_ms=statistics.mean(times), p50_ms=statistics.median(times),
                p95_ms=statistics.quantiles(times, n=20, method='inclusive')[18] if len(times)>1 else times[0], samples_ms=times)


def timed(call):
    start = time.perf_counter_ns()
    result = call()
    return result, (time.perf_counter_ns()-start)/1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    args = parser.parse_args()
    from bench_pipeline import load_extensions
    load_extensions(args.native_dir)
    import numpy as np
    from vc_reader import load, AmbiguousMatchError
    from vc_reader.cuda import asarray, is_available
    from vc_reader.reader import ZarrGroup, MemoryCache
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from paired_reading import discover
    if args.backend == 'cuda' and not is_available():
        raise RuntimeError('CUDA device is unavailable')
    catalog = load(args.metadata)
    label, render = discover(catalog)
    ambiguous = catalog.scrolls['PHerc0139'].on_volume('20260413113053').segments['20260302000001'].renders.zarr
    assert len(ambiguous) == 2
    try:
        ambiguous.one()
    except AmbiguousMatchError:
        pass
    else:
        raise AssertionError('ambiguous published renders were silently selected')
    resolutions = catalog.scrolls['PHercParis4'].volumes.at_resolution(8, rel_tol=.02)
    assert len(resolutions) == 2
    options = dict(cache_dir=str(args.cache_dir), offline=args.offline, memory_cache=64 << 20, threads=1)
    opened = []
    for name in ('paired', 'direct'):
        stack = ExitStack()
        try:
            def open_readers():
                if name == 'paired':
                    return stack.enter_context(label.open_paired(render=render, layers=slice(40, 65), backend=args.backend, **options))
                cache = MemoryCache(64 << 20)
                return {key: stack.enter_context(ZarrGroup.open(source.storage.location, **{**options, 'memory_cache': cache})).level(0)
                        for key, source in [('image', render.source), ('label', label.source), *[(c.name, c.source) for c in label.source.companions]]}
            reader, opening_ms = timed(open_readers)
            def request(start):
                y, x = (slice(n, n+128) for n in start)
                if name == 'paired':
                    patch = reader[y, x]
                    return {**patch.data, **patch.masks['label']}
                result = {k: (r[40:65, y, x] if k == 'image' else r[y, x])[None] for k, r in reader.items()}
                if args.backend == 'cuda':
                    result = {k: asarray(v) for k, v in result.items()}
                return result
            first, first_ms, hashes, nonzero = [], [], [], []
            for start in STARTS:
                result, ms = timed(lambda: request(start))
                values = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in result.items()}
                first_ms.append(ms)
                hashes.append({k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in values.items()})
                nonzero.append(int(np.count_nonzero(values['label'])))
            for _ in range(2):
                for start in STARTS:
                    request(start)
            before = reader.stats() if name == 'paired' else {k: r.stats() for k, r in reader.items()}
            times = []
            for _ in range(10):
                for start in STARTS:
                    _, ms = timed(lambda: request(start))
                    times.append(ms)
            after = reader.stats() if name == 'paired' else {k: r.stats() for k, r in reader.items()}
            if args.profile and name == 'paired':
                profiler = cProfile.Profile()
                profiler.enable()
                for _ in range(10):
                    for start in STARTS:
                        request(start)
                profiler.disable()
                profiler.dump_stats(args.profile)
            opened.append(dict(name=name, opening_ms=opening_ms, first_reads=summary(first_ms), warm=summary(times),
                               hashes=hashes, label_nonzero=nonzero, stats_before=before, stats_after=after))
        finally:
            stack.close()
    assert opened[0]['hashes'] == opened[1]['hashes']
    assert opened[0]['label_nonzero'][0] == 9926
    report = dict(command=sys.argv, platform=platform.platform(), python=sys.version, build='Release', backend=args.backend,
                  native_dir=str(args.native_dir), metadata_sha256=hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest(),
                  render=render.id, label=label.id, starts=STARTS, layers=[40,65], size=[128,128], warmups=2,
                  repetitions=10, requests=6, ambiguous_render_ids=[a.id for a in ambiguous],
                  resolution_candidates=[dict(volume=m.volume.id, level=m.level, estimate=m.estimated_resolution_um, verified=m.verified) for m in resolutions],
                  cache_protocol='64 MiB shared chunk cache per reader set; disk cache reused; OS caches not cleared; output validation/download outside timings',
                  rows=opened)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for row in opened:
        print(row['name'], 'open ms', row['opening_ms'], 'first mean ms', row['first_reads']['mean_ms'], 'warm', {k:v for k,v in row['warm'].items() if k!='samples_ms'})


if __name__ == '__main__':
    main()
