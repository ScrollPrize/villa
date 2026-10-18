"""Six fixed aligned crops from a real volume, with low-level read baselines.

Transforms are controlled benchmark placements, not claims of anatomical
registration. Optional multichannel sources must have explicit real provenance.
Run with existing Release builds; this script never installs dependencies.
"""
import argparse
import cProfile
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import platform
import pstats
import resource
import sys
import time
import tracemalloc

import numpy as np
from bench_source_readers import summary


def timed(call, requests):
    for _ in range(2):
        for request in requests:
            call(*request)
    times = []
    for _ in range(10):
        for request in requests:
            now = time.perf_counter()
            call(*request)
            times.append((time.perf_counter()-now)*1000)
    profiler = cProfile.Profile()
    profiler.enable()
    for request in requests:
        call(*request)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumtime').print_stats(15)
    tracemalloc.start()
    for request in requests:
        call(*request)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return dict(summary=summary(times), milliseconds=times, python_peak_bytes=peak, profile=stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--array', required=True, help='existing local real Zarr array')
    parser.add_argument('--start', type=int, nargs=3, required=True)
    parser.add_argument('--size', type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda', 'both'), default='both')
    parser.add_argument('--field-source', type=Path, help='serialized registered real multichannel Source')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from bench_pipeline import load_extensions
    load_extensions(args.native_dir)
    from vc_reader import Source, Storage, VolumeRegistration, Affine
    from vc_reader.sampling import open_sample
    from vc_reader.reader import ZarrReader
    from vc_reader.cuda import asarray

    reference = Source('benchmark-real-volume', Storage(args.array, 'zarr'), role='image', axes=('z', 'y', 'x'),
                       registration=VolumeRegistration('benchmark', 'volume', affine=Affine.identity()),
                       provenance={'kind': 'real-volume', 'path': args.array})
    # Shear plus a quarter voxel translation, anchored at the fixed crop origin.
    z, y, x = args.start
    pull = Affine.from_rows(((1, .125, 0, -.125*y+.25), (0, 1, .125, -.125*z+.25), (0, 0, 1, .25)))
    target = replace(reference, id='user:controlled-affine', role='scalar',
                     registration=replace(reference.registration, affine=pull.inverse()))
    requests = [(tuple(s+j for s in args.start), tuple(args.size)) for j in (0, 2, 4, 6, 8, 10)]
    matrix = np.array(pull.linear, dtype='float32')[::-1, ::-1].copy()
    field = Source.from_dict(json.loads(args.field_source.read_text())) if args.field_source else None
    report = dict(command=sys.argv, platform=platform.platform(), python=sys.version, numpy=np.__version__,
                  reference=reference.to_dict(), target=target.to_dict(), requests=requests,
                  native_sha256=hashlib.sha256(next(args.native_dir.glob('_native*.so')).read_bytes()).hexdigest(),
                  protocol=dict(warmups=2, repetitions=10, requests=6, os_cache_cleared=False,
                                cold='fresh reader and empty application caches; OS file cache unchanged'), runs={})
    for backend in ('cpu', 'cuda') if args.backend == 'both' else (args.backend,):
        for name, targets in [('direct', {}), ('affine', {'target': target})] + ([('field', {'target': field})] if field else []):
            now = time.perf_counter()
            with open_sample(reference, targets, backend=backend, sampling_cache_bytes=128 << 20,
                             memory_cache=128 << 20) as reader:
                opened = (time.perf_counter()-now)*1000
                cold = []
                hashes = []
                for request in requests:
                    now = time.perf_counter(); result = reader.read(*request); cold.append((time.perf_counter()-now)*1000)
                    data = result['image' if name == 'direct' else 'target'].data
                    data = data.numpy() if backend == 'cuda' else data
                    hashes.append(hashlib.sha256(data.tobytes()).hexdigest())
                initial_stats = reader.stats()
                run = timed(reader.read, requests)
                run.update(open_ms=opened, first_reads_ms=cold, initial_stats=initial_stats,
                           final_stats=reader.stats(), output_hashes=hashes)
                if name != 'field':
                    with ZarrReader.open(args.array, memory_cache=128 << 20, threads=1) as raw:
                        device_reader = raw.cuda(cache_bytes=128 << 20) if backend == 'cuda' else None
                        def baseline(start, size):
                            if backend == 'cuda':
                                image = device_reader.read(start, size, valid=True)
                                if name == 'direct':
                                    return image
                                origin = tuple(reversed(pull.apply(tuple(reversed(start)))))
                                return image, device_reader.read_affine(size, origin, matrix, valid=True, source='chunks')
                            image = raw.read(start, size)
                            if name == 'direct':
                                return image, np.ones(size, dtype=bool)
                            origin = tuple(reversed(pull.apply(tuple(reversed(start)))))
                            return image, raw.read_affine(size, origin, matrix)
                        # Baseline CPU affine timing excludes its unavailable
                        # validity output; report this explicitly, not as parity.
                        run['baseline'] = timed(baseline, requests)
                        run['baseline_note'] = ('data and masks' if backend == 'cuda' or name == 'direct'
                                                else 'image + affine data; low-level CPU API has no validity result')
                        for request in requests:
                            start, size = request
                            actual = reader.read(start, size)
                            expected = raw.read(start, size) if name == 'direct' else raw.read_affine(
                                size, tuple(reversed(pull.apply(tuple(reversed(start))))), matrix)
                            output = actual['image' if name == 'direct' else 'target'].data
                            output = output.numpy() if backend == 'cuda' else output
                            np.testing.assert_allclose(output[0], expected, rtol=2e-6, atol=2e-5)
                        if device_reader is not None:
                            device_reader.close()
                if backend == 'cuda':
                    # Time a transfer of an equally sized result independently.
                    host = np.ones((1, *args.size), dtype='uint8' if name == 'direct' else 'float32')
                    run['upload_only'] = timed(lambda *_: asarray(host), requests)
                report['runs'][backend+'_'+name] = run
    report['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {n: v[n] for n in ('summary', 'open_ms', 'first_reads_ms')} for k, v in report['runs'].items()}, indent=2))


if __name__ == '__main__':
    main()
