"""Build the next-coarser CT pyramid level in the local remote-CT cache from already cached chunks.

The remote OME pyramids store level L+1 as the 2x2x2 block mean of level L, rounded half to even. That rule is
bit-exact against downloaded PHerc0175A and PHerc1447 level-1 chunks, and ``--verify`` repeats the check.

A level-(L+1) chunk is written only when all eight level-L children are cached and lie fully inside the volume, so
every written chunk equals the remote one. Chunks that don't qualify are left for the remote prefetcher, and
existing chunks are kept. Writes are atomic and take the prefetcher's per-chunk lock, so this can run next to
training. Read level-L pages are dropped from the page cache where the OS supports it (Linux), keeping it for the
coarse data.

Usage (from fiber_follow/):
  python scripts/downsample_ct_cache.py --dataset-config configs/mixed_ct_datasets_paris50.json  # remote CT sources
  python scripts/downsample_ct_cache.py s3://bucket/volume.zarr --cache-dir /mnt/raid_nvme/volume_cache --level 0
"""
import argparse
from collections import Counter
import fcntl
import json
from multiprocessing import get_context
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vesuvius.neural_tracing.fiber_follow.data.volume import RemoteChunkedArray

REMOTE = ('s3://', 'http://', 'https://')


def chunk_path(root, key, separator):
    return Path(root).joinpath(*map(str, key)) if separator == '/' else Path(root)/'.'.join(map(str, key))


def cached_keys(root, separator):
    """Chunk keys present in a cached Zarr v2 level (atomic writes: a present file is complete)."""
    if separator != '/':
        return [tuple(map(int, e.name.split('.'))) for e in os.scandir(root)
                if e.is_file() and len(e.name.split('.')) == 3 and all(v.isdigit() for v in e.name.split('.'))]
    keys = []
    for z in os.scandir(root):
        if z.is_dir() and z.name.isdigit():
            for y in os.scandir(z.path):
                if y.is_dir() and y.name.isdigit():
                    keys.extend((int(z.name), int(y.name), int(x.name)) for x in os.scandir(y.path) if x.name.isdigit())
    return keys


def block_mean(fine):
    """2x2x2 means of a uint8 block, rounded half to even like the remote pyramid (== np.round(sum/8))."""
    n = [s//2 for s in fine.shape]
    total = fine.reshape(n[0], 2, n[1], 2, n[2], 2).sum((1, 3, 5), dtype=np.uint16)
    quotient, remainder = np.divmod(total, 8)
    return (quotient+(remainder > 4)+((remainder == 4) & (quotient & 1).astype(bool))).astype(np.uint8)


def read_chunk(path, chunks):
    with open(path, 'rb') as stream:
        data = np.fromfile(stream, np.uint8)
        if hasattr(os, 'posix_fadvise'):  # Linux; the coarse chunk is what training reads next
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return data.reshape(chunks)


def coarse_chunk(fine_root, separator, key, chunks):
    out = np.empty(chunks, np.uint8)
    half = [c//2 for c in chunks]
    for octant in np.ndindex(2, 2, 2):
        child = tuple(2*k+o for k, o in zip(key, octant))
        out[tuple(slice(o*h, (o+1)*h) for o, h in zip(octant, half))] = block_mean(
            read_chunk(chunk_path(fine_root, child, separator), chunks))
    return out


def build(task):
    fine_root, coarse_root, fine_sep, coarse_sep, key, chunks = task
    path = chunk_path(coarse_root, key, coarse_sep)
    if path.is_file():
        return 'existing'
    # The prefetcher's per-chunk lock: never race a download of the same chunk.
    lock_path = Path(coarse_root)/'.locks'/'.'.join(map(str, key))
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 'busy'
        if path.is_file():
            return 'existing'
        RemoteChunkedArray._atomic_write(path, coarse_chunk(fine_root, fine_sep, key, chunks).tobytes(order='C'))
    return 'built'


def verify(task):
    fine_root, coarse_root, fine_sep, coarse_sep, key, chunks = task
    expected = np.fromfile(chunk_path(coarse_root, key, coarse_sep), np.uint8).reshape(chunks)
    return bool(np.array_equal(coarse_chunk(fine_root, fine_sep, key, chunks), expected))


def plan(url, level, cache_dir):
    """Coarse keys whose eight children are cached and fully in bounds, plus the level descriptors."""
    fine_root = RemoteChunkedArray.cache_path(url, level, cache_dir)
    if not (fine_root/'.zarray').exists():
        raise FileNotFoundError(f'No cached level {level} for {url} under {cache_dir}')
    RemoteChunkedArray(url, level+1, cache_dir, 1 << 20)  # publishes the coarse level's remote metadata if missing
    coarse_root = RemoteChunkedArray.cache_path(url, level+1, cache_dir)
    fine, coarse = (json.loads((r/'.zarray').read_text()) for r in (fine_root, coarse_root))
    chunks = tuple(fine['chunks'])
    if (tuple(coarse['chunks']) != chunks or any(c % 2 for c in chunks) or fine['dtype'] != '|u1' or coarse['dtype'] != '|u1'
            or [-(-s//2) for s in fine['shape']] != coarse['shape']):
        raise ValueError(f'{url}: levels {level}/{level+1} are not a uint8 2x pyramid with equal chunks')
    fine_sep, coarse_sep = fine.get('dimension_separator', '.'), coarse.get('dimension_separator', '.')
    children = Counter(tuple(k//2 for k in key) for key in cached_keys(fine_root, fine_sep))
    inside = lambda key: all((2*k+2)*c <= s for k, c, s in zip(key, chunks, fine['shape']))
    complete = sorted(key for key, n in children.items() if n == 8 and inside(key))
    tasks = [(str(fine_root), str(coarse_root), fine_sep, coarse_sep, key, chunks) for key in complete]
    return tasks, len(children), coarse_root, coarse_sep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('urls', nargs='*', help='remote CT zarr roots (s3://...)')
    ap.add_argument('--dataset-config', help='follower dataset config: every remote CT source at its ct_level')
    ap.add_argument('--cache-dir', help='remote CT cache root (default: the dataset config cache_dir)')
    ap.add_argument('--level', type=int, default=0, help='finer level to downsample (with explicit URLs)')
    ap.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument('--verify', type=int, default=8, help='existing coarse chunks to recompute and compare first')
    args = ap.parse_args()
    volumes = [(url, args.level, args.cache_dir) for url in args.urls]
    if args.dataset_config:
        document = json.loads(Path(args.dataset_config).read_text())
        volumes += [(s['ct'], int(s.get('ct_level', 0)), args.cache_dir or document['cache_dir'])
                    for s in document['sources'] if str(s.get('ct', '')).startswith(REMOTE)]
    if not volumes or any(cache is None for *_, cache in volumes):
        ap.error('give remote CT URLs with --cache-dir, or --dataset-config')
    with get_context('spawn').Pool(args.workers) as pool:
        for url, level, cache_dir in volumes:
            tasks, touched, coarse_root, coarse_sep = plan(url, level, cache_dir)
            print(f'{url} level {level} -> {level+1}: {len(tasks)} of {touched} touched coarse chunks have all children '
                  f'cached', flush=True)
            existing = [t for t in tasks if chunk_path(coarse_root, t[4], coarse_sep).is_file()]
            if args.verify and existing:
                rng = np.random.default_rng(0)
                sample = [existing[i] for i in rng.choice(len(existing), min(args.verify, len(existing)), replace=False)]
                matches = sum(pool.map(verify, sample))
                print(f'  verify: {matches}/{len(sample)} existing (downloaded) coarse chunks reproduced exactly', flush=True)
                if matches != len(sample):
                    raise SystemExit('Local downsampling does not reproduce the remote pyramid; nothing written')
            status = Counter(tqdm(pool.imap_unordered(build, tasks, chunksize=8), total=len(tasks), unit='chunk',
                                  desc=f'level {level+1}', dynamic_ncols=True))
            print(f'  {dict(status)}', flush=True)


if __name__ == '__main__':
    main()
