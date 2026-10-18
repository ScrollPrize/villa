"""Measure local 2D TIFF source crops against equivalent tifffile page reads.

Default input is the bundled real PHercParis4 geometry mask. This reports
fresh-reader costs separately; it does not clear the OS file cache or claim
physical disk-byte measurements for memory-mapped access.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
import tracemalloc

import numpy as np
import tifffile

from vc_reader import Source, Storage


def summary(values):
    return dict(mean_ms=float(np.mean(values)), p50_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).parent / 'fixtures/fiber_132487/mask.tif')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.input.resolve()
    source = Source('benchmark-label', Storage(str(path), 'tiff'))
    started = time.perf_counter()
    reader = source.open()
    open_ms = (time.perf_counter() - started) * 1000
    if reader.axes != ('y', 'x'):
        raise ValueError('benchmark expects one YX TIFF series')
    h, w = reader.shape
    size = (min(h, 16), min(w, 512))
    requests = [((int((h-size[0])*i/5), int((w-size[1])*i/5)), size) for i in range(6)]
    with tifffile.TiffFile(path) as tif:
        compression = tif.pages[0].compression.name

    def baseline(start, size):
        with tifffile.TiffFile(path) as tif:
            page = tif.series[0].pages[0]
            array = page.asarray(out='memmap' if page.is_memmappable else None, maxworkers=1)
            try:
                y, x = start
                return array[y:y+size[0], x:x+size[1]].copy()
            finally:
                if isinstance(array, np.memmap):
                    array._mmap.close()

    started = time.perf_counter()
    first = reader.read(*requests[0])
    first_read_ms = (time.perf_counter() - started) * 1000
    np.testing.assert_array_equal(first, baseline(*requests[0]))
    timings = {}
    for name, read in (('tifffile', baseline), ('source_reader', reader.read)):
        values = []
        for iteration in range(12):
            for request in requests:
                started = time.perf_counter()
                output = read(*request)
                elapsed = (time.perf_counter() - started) * 1000
                if iteration >= 2:
                    values.append(elapsed)
                if iteration == 0:
                    np.testing.assert_array_equal(output, baseline(*request))
        timings[name] = summary(values)
    tracemalloc.start()
    for request in requests:
        reader.read(*request)
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = dict(input=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  source_id=source.id, shape=reader.shape, dtype=str(reader.dtype), compression=compression,
                  requests=requests, warmups=2, repetitions=10, measured_crops=60,
                  platform=platform.platform(), python=platform.python_version(),
                  numpy=np.__version__, tifffile=tifffile.__version__,
                  build='Python image reader; no native extension loaded',
                  cache_protocol='fresh reader first read; OS file cache not cleared; warm measurements follow',
                  open_ms=open_ms, first_read_ms=first_read_ms, warm=timings,
                  decoded_cache=reader.decoded_cache.stats(), python_traced_peak_bytes=traced_peak,
                  process_peak_rss_bytes=rss if platform.system() == 'Darwin' else rss*1024,
                  source_file_bytes=path.stat().st_size, physical_bytes_read=None,
                  requested_output_bytes_per_repetition=sum(np.prod(n).item() for _, n in requests)*reader.dtype.itemsize)
    reader.close()
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
