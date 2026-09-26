"""Decode zarr-v2 array levels in place, so readers memory-map raw chunks.

Decoding a 128³ CT chunk from the volpkg's rANS codec costs about 3.5 ms and
holds the GIL, which made chunk decoding the dominant cost of training data
loading. Rewriting a level uncompressed (``compressor: null``, same shape,
chunking, dtype and layout) lets ``ChunkedArray`` memory-map chunks, so every
loader worker, the tracer and the collector share decoded data through the
page cache. Values are exactly the decoded source values; ``--verify``
re-decodes a random sample of chunks and compares bytes before anything is
replaced.

    python scripts/decode_store.py /path/to/volume.zarr/0 /path/to/volume.zarr/1

Each chunk is first decoded to a ``<chunk>.raw`` sibling; only when every
chunk has one are the originals replaced and ``.zarray`` rewritten, so an
interrupted run leaves the array readable and resumes where it stopped. Stop
readers of the array before running: the compressed source is not kept.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import time
from pathlib import Path

import numpy as np

RAW_SUFFIX = '.raw'


def read_meta(path) -> dict:
    return json.loads(Path(path, '.zarray').read_text())


def chunk_keys(path, sep) -> list[tuple[int, int, int]]:
    """Existing chunk keys of a three-dimensional array (sibling files are ignored)."""
    keys = []
    if sep == '.':
        pattern = re.compile(r'^(\d+)\.(\d+)\.(\d+)$')
        for entry in os.scandir(path):
            match = pattern.match(entry.name)
            if match and entry.is_file():
                keys.append(tuple(int(v) for v in match.groups()))
    else:
        for z in os.scandir(path):
            if not z.is_dir() or not z.name.isdigit():
                continue
            for y in os.scandir(z.path):
                if not y.is_dir() or not y.name.isdigit():
                    continue
                for x in os.scandir(y.path):
                    if x.is_file() and x.name.isdigit():
                        keys.append((int(z.name), int(y.name), int(x.name)))
    return sorted(keys)


def chunk_file(root, sep, key) -> str:
    return os.path.join(root, sep.join(str(k) for k in key))


_WORK: dict = {}


def _init_worker(source, meta):
    import numcodecs
    from vesuvius.neural_tracing.fiber_follow.volume import _register_vcz1
    _register_vcz1()
    _WORK.update(source=source, sep=meta.get('dimension_separator', '.'),
                 codec=None if meta.get('compressor') is None else numcodecs.get_codec(meta['compressor']),
                 nbytes=int(np.prod(meta['chunks']))*np.dtype(meta['dtype']).itemsize)


def _decode(key) -> bytes:
    with open(chunk_file(_WORK['source'], _WORK['sep'], key), 'rb') as fh:
        raw = fh.read()
    codec = _WORK['codec']
    buf = memoryview(raw if codec is None else codec.decode(raw)).cast('B')
    if len(buf) != _WORK['nbytes']:
        raise ValueError(f'chunk {key} decoded to {len(buf)} bytes, expected {_WORK["nbytes"]}')
    return bytes(buf)


def _decode_sibling(key) -> int:
    """Write ``<chunk>.raw``; returns bytes written (0 when already present or already swapped)."""
    original = chunk_file(_WORK['source'], _WORK['sep'], key)
    target = original+RAW_SUFFIX
    nbytes = _WORK['nbytes']
    if os.path.isfile(target) and os.path.getsize(target) == nbytes:
        return 0
    if os.path.getsize(original) == nbytes:
        # An earlier run may have swapped this chunk already; a compressed
        # payload of exactly the raw size is only accepted if it decodes.
        try:
            data = _decode(key)
        except Exception:
            return 0
    else:
        data = _decode(key)
    temporary = f'{target}.partial.{os.getpid()}'
    with open(temporary, 'wb') as fh:
        fh.write(data)
    os.replace(temporary, target)
    return len(data)


def _verify_sibling(key) -> bool:
    with open(chunk_file(_WORK['source'], _WORK['sep'], key)+RAW_SUFFIX, 'rb') as fh:
        return fh.read() == _decode(key)


def _swap(key) -> int:
    original = chunk_file(_WORK['source'], _WORK['sep'], key)
    raw = original+RAW_SUFFIX
    if os.path.isfile(raw):
        os.replace(raw, original)
        return 1
    return 0


def _clean_partials(source, sep):
    """Remove partial sibling writes left by an interrupted run."""
    for root, _, files in os.walk(source):
        for name in files:
            if '.partial.' in name:
                os.remove(os.path.join(root, name))
        if sep == '.':
            break


def decode_in_place(source, workers=None, verify=64, seed=0, log=print) -> dict:
    """Rewrite every chunk of ``source`` uncompressed; resumable, readable until the final swap."""
    source = os.path.abspath(source)
    meta = read_meta(source)
    if len(meta['shape']) != 3:
        raise ValueError('Only three-dimensional arrays are decoded')
    sep = meta.get('dimension_separator', '.')
    keys = chunk_keys(source, sep)
    workers = workers or max(1, (os.cpu_count() or 2)-2)
    report = dict(source=source, chunks=len(keys), decoded_bytes=0, verified=0, swapped=0)
    _clean_partials(source, sep)
    context = multiprocessing.get_context('forkserver')
    started = time.monotonic()
    with context.Pool(workers, initializer=_init_worker, initargs=(source, meta)) as pool:
        if meta.get('compressor') is not None:
            done = 0
            for nbytes in pool.imap_unordered(_decode_sibling, keys, chunksize=8):
                report['decoded_bytes'] += nbytes
                done += 1
                if done % 5000 == 0 or done == len(keys):
                    elapsed = time.monotonic()-started
                    log(f'{source}: {done}/{len(keys)} chunks decoded, {report["decoded_bytes"]/2**30:.1f} GB, '
                        f'{report["decoded_bytes"]/2**30/max(elapsed, 1e-9):.2f} GB/s, {elapsed:.0f} s', flush=True)
            if verify and keys:
                rng = np.random.default_rng(seed)
                sample = [keys[i] for i in rng.choice(len(keys), min(verify, len(keys)), replace=False)]
                pending = [key for key in sample if os.path.isfile(chunk_file(source, sep, key)+RAW_SUFFIX)]
                for ok in pool.imap_unordered(_verify_sibling, pending):
                    if not ok:
                        raise ValueError(f'{source}: decoded chunk differs from the source')
                    report['verified'] += 1
        # Every chunk has its raw sibling (or was swapped before): replace the originals.
        for swapped in pool.imap_unordered(_swap, keys, chunksize=64):
            report['swapped'] += swapped
    if meta.get('compressor') is not None or meta.get('filters'):
        temporary = Path(source, '.zarray.partial')
        temporary.write_text(json.dumps(dict(meta, compressor=None, filters=None), indent=2))
        os.replace(temporary, Path(source, '.zarray'))
    log(f'{source}: uncompressed, {report["swapped"]} chunks swapped, {time.monotonic()-started:.0f} s', flush=True)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('arrays', nargs='+', help='zarr-v2 array directories (a pyramid level)')
    ap.add_argument('--workers', type=int)
    ap.add_argument('--verify', type=int, default=64, help='random chunks re-decoded and compared; 0 disables')
    args = ap.parse_args(argv)
    for array in args.arrays:
        print(json.dumps(decode_in_place(array, args.workers, args.verify)), flush=True)


if __name__ == '__main__':
    main()
