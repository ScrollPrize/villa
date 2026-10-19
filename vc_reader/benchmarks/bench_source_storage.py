"""Cold-cache and warm-crop acceptance for real 2D label assets.

A new temporary disk cache is used per run. OS caches are not cleared. The
baseline uses the existing Zarr reader or the format decoder directly. For
PNG/JPEG the baseline decodes the complete image per crop; TIFF maps a page
when possible and otherwise decodes it. No image is silently re-encoded here.
"""

import argparse
import cProfile
import hashlib
import io
import json
from pathlib import Path
import platform
import pstats
import resource
import tempfile
import time
import tracemalloc

import numpy as np

from bench_source_readers import summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source', type=Path, help='serialized Source.to_dict() JSON')
    inputs.add_argument('--input', type=Path)
    parser.add_argument('--format', choices=('png', 'jpeg', 'tiff'))
    parser.add_argument('--native-dir', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--crop', type=int, default=128)
    parser.add_argument('--requests', type=Path, help='fixed [[start, size], ...] JSON; six requests')
    args = parser.parse_args()
    if args.native_dir:
        from bench_pipeline import load_extensions
        load_extensions(args.native_dir)
    from vc_reader import Source, Storage
    from vc_reader.reader import ZarrReader
    from PIL import Image
    import PIL
    import tifffile
    source = (Source.from_dict(json.loads(args.source.read_text())) if args.source else
              Source('acceptance:' + args.input.name, Storage(str(args.input.resolve()), args.format)))
    with tempfile.TemporaryDirectory(prefix='vc-source-benchmark-') as directory:
        options = dict(cache_dir=directory, cache_max_bytes=64 << 20, memory_cache=16 << 20, threads=1)
        started = time.perf_counter()
        with source.open(**options) as reader:
            open_ms = (time.perf_counter()-started)*1000
            if reader.axes != ('y', 'x') or source.encoding is not None:
                raise ValueError('acceptance benchmark expects unencoded YX data')
            size = tuple(min(n, args.crop) for n in reader.shape)
            requests = [(tuple((n-k)*i//5 for n, k in zip(reader.shape, size)), size) for i in range(6)]
            if args.requests:
                requests = json.loads(args.requests.read_text())
                if len(requests) != 6:
                    raise ValueError('acceptance protocol requires six fixed requests')
            cold, expected = [], []
            for request in requests:
                started = time.perf_counter()
                expected.append(reader.read(*request))
                cold.append((time.perf_counter()-started)*1000)
            cold_stats = reader.stats()
            compression = None
            if reader._is_zarr:
                base = ZarrReader.open(reader.resolved.location, **options)
                baseline = base.read_region
                compression = base.meta.compressor
            else:
                base = None
                if reader._files[0].format == 'tiff':
                    with tifffile.TiffFile(source.storage.location) as tif:
                        compression = tif.pages[0].compression.name
                else:
                    compression = reader._files[0].format
                def baseline(start, size):
                    y, x = start
                    h, w = size
                    if reader._files[0].format == 'tiff':
                        with tifffile.TiffFile(source.storage.location) as tif:
                            page = tif.series[0].pages[0]
                            values = page.asarray(out='memmap' if page.is_memmappable else None, maxworkers=1)
                            try:
                                return values[y:y+h, x:x+w].copy()
                            finally:
                                if isinstance(values, np.memmap):
                                    values._mmap.close()
                    with Image.open(source.storage.location) as image:
                        return np.array(image)[y:y+h, x:x+w].copy()
            timings = {}
            for name, read in (('direct_reader', baseline), ('source_reader', reader.read)):
                samples = []
                for repetition in range(12):
                    for i, request in enumerate(requests):
                        started = time.perf_counter()
                        output = read(*request)
                        elapsed = (time.perf_counter()-started)*1000
                        if repetition >= 2:
                            samples.append(elapsed)
                        if repetition == 0:
                            np.testing.assert_array_equal(output, expected[i])
                timings[name] = summary(samples)
            if base is not None:
                base.close()
            companions = {}
            for name, child in reader.companions.items():
                values = child.read_many(requests)
                companions[name] = dict(shape=child.shape, dtype=str(child.dtype),
                    crop_hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in values], stats=child.stats())
            reader.decoded_cache.clear()
            if reader._is_zarr and reader._backend.reader.memory_cache is not None:
                reader._backend.reader.memory_cache.clear()
            tracemalloc.start()
            reader.read_many(requests)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            profile = cProfile.Profile()
            profile.runcall(reader.read_many, requests*10)
            buffer = io.StringIO()
            pstats.Stats(profile, stream=buffer).strip_dirs().sort_stats('cumtime').print_stats(12)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report = dict(source_id=source.id, location=source.storage.location,
                source_descriptor_sha256=hashlib.sha256(json.dumps(source.to_dict(), sort_keys=True).encode()).hexdigest(),
                shape=reader.shape, dtype=str(reader.dtype), compression=compression,
                requests=requests, warmups=2, repetitions=10, measured_crops=60,
                open_ms=open_ms, cold_reads_ms=cold, cold_stats=cold_stats, warm=timings,
                final_stats=reader.stats(), companions=companions,
                crop_hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in expected],
                crop_nonzero=[int(np.count_nonzero(a)) for a in expected],
                traced_decode_peak_bytes=peak,
                process_peak_rss_bytes=rss if platform.system() == 'Darwin' else rss*1024,
                physical_disk_bytes=None, platform=platform.platform(), python=platform.python_version(),
                numpy=np.__version__, pillow=PIL.__version__, tifffile=tifffile.__version__,
                native_dir=str(args.native_dir), build='Release' if args.native_dir else 'Python image decoder',
                cache_protocol='new empty disk cache and new reader; OS cache not cleared; 16 MiB Zarr memory cache',
                profile=buffer.getvalue())
            if args.input:
                report['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
                report['encoded_bytes'] = args.input.stat().st_size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('source_id', 'shape', 'compression', 'open_ms', 'cold_reads_ms', 'warm', 'cold_stats')}))


if __name__ == '__main__':
    main()
