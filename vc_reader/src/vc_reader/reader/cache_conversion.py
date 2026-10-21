"""Offline, lossless conversion of downloaded Zarr v2 chunks to Delta3D.

Readers and writers must be closed for the entire conversion. Each array is
staged beside its source, verified, and atomically exchanged before removing
the original. Missing chunks are never fetched or materialized.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import ctypes
import errno
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

from .disk_cache import atomic_write


def _exchange(left: Path, right: Path) -> None:
    """Swap complete directories without exposing mismatched chunk metadata."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith('linux'):
        function = getattr(libc, 'renameat2', None)
        if function is None:
            raise OSError(errno.ENOTSUP, 'atomic directory exchange requires renameat2')
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        args = (-100, os.fsencode(left), -100, os.fsencode(right), 2)  # RENAME_EXCHANGE
    elif sys.platform == 'darwin':
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = (os.fsencode(left), os.fsencode(right), 2)  # RENAME_SWAP
    else:
        raise OSError(errno.ENOTSUP, 'atomic directory exchange requires Linux or macOS')
    function.restype = ctypes.c_int
    if function(*args):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(left))


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _arrays(root: Path):
    for parent, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and not (Path(parent) / d).is_symlink())
        if '.zarray' in files:
            dirs.clear()
            yield Path(parent)


def _signature(path: Path):
    stat = path.stat(follow_symlinks=False)
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def _inventory(path: Path, meta: dict):
    """Record chunks and sidecars, excluding transient locks and temp files."""
    files, chunks = {}, []
    grid = tuple((s + c - 1) // c for s, c in zip(meta['shape'], meta['chunks']))
    sep = meta.get('dimension_separator', '.')
    if sep not in ('/', '.'):
        raise ValueError(f'unsupported dimension_separator: {path}')
    expected = math.prod(meta['chunks']) * np.dtype(meta['dtype']).itemsize
    for parent, dirs, names in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d != '.locks')
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                raise ValueError(f'cache array contains a directory symlink: {parent}/{name}')
        for name in sorted(names):
            if '.tmp.' in name or name.endswith('.lock'):
                continue
            file = Path(parent) / name
            if file.is_symlink() or not file.is_file():
                raise ValueError(f'cache array contains a nonregular file: {file}')
            relative = file.relative_to(path)
            parts = relative.parts if sep == '/' else relative.as_posix().split('.')
            signature = _signature(file)
            files[relative] = signature
            if len(parts) == 3 and all(p.isdecimal() for p in parts):
                if any(int(i) >= g for i, g in zip(parts, grid)):
                    raise ValueError(f'chunk coordinate outside array: {file}')
                if signature[0] != expected:
                    raise ValueError(f'chunk has {signature[0]} bytes; expected {expected}: {file}')
                chunks.append(relative)
    return files, chunks


def _validate_metadata(path: Path, meta: dict):
    dtype = np.dtype(meta['dtype'])
    if (meta.get('zarr_format') != 2 or len(meta['shape']) != 3 or len(meta['chunks']) != 3
            or any(not isinstance(n, int) or n <= 0 for n in (*meta['shape'], *meta['chunks']))
            or any(n > 2**31 - 1 for n in meta['chunks'])
            or dtype.kind != 'u' or dtype.itemsize not in (1, 2)
            or meta.get('filters') or meta.get('order', 'C') != 'C'):
        raise ValueError(f'conversion requires unfiltered C-order 3D uint8/uint16 Zarr v2: {path}')
    for parent in (path, *path.parents):
        if (parent / '.vc_reader_cache.sqlite3').exists():
            raise ValueError(f'conversion of a byte-budgeted cache is not supported: {parent}')
        if (parent / '.zmetadata').exists():
            raise ValueError(f'remove/rebuild consolidated Zarr metadata before conversion: {parent}')


def _convert_chunk(source, target, relative, signature, meta):
    from .. import _reader
    file = source / relative
    if _signature(file) != signature:
        raise RuntimeError(f'source changed during conversion: {file}')
    raw = file.read_bytes()
    dtype = np.dtype(meta['dtype'])
    encoded = _reader._encode_delta3d_verified(raw, meta['chunks'], dtype.itemsize)
    if _signature(file) != signature:
        raise RuntimeError(f'source changed during conversion: {file}')
    output = target / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if output.read_bytes() != encoded:
        raise RuntimeError(f'written chunk verification failed: {output}')
    return len(encoded), output.stat().st_blocks * 512


def _verify_reader(source, stage, chunks, meta):
    """Exercise the public reader on a stored chunk before publishing the array."""
    if not chunks:
        return
    from .zarr_reader import open_zarr
    relative = chunks[0]
    parts = relative.parts if meta.get('dimension_separator', '.') == '/' else relative.name.split('.')
    start = tuple(int(i) * c for i, c in zip(parts, meta['chunks']))
    size = tuple(min(c, s - i) for c, s, i in zip(meta['chunks'], meta['shape'], start))
    expected = np.frombuffer((source / relative).read_bytes(), dtype=meta['dtype']).reshape(meta['chunks'])
    expected = expected[tuple(slice(n) for n in size)]
    with open_zarr(str(stage), threads=1, memory_cache=0, cache_writes=False) as reader:
        actual = reader[tuple(slice(i, i + n) for i, n in zip(start, size))]
    np.testing.assert_array_equal(actual, expected)


def convert_array(path: Path, *, workers=8, dry_run=False, progress=print):
    path = Path(path).resolve()
    meta = json.loads((path / '.zarray').read_text())
    if meta.get('compressor') is not None:
        return None
    _validate_metadata(path, meta)
    files, chunks = _inventory(path, meta)
    before = sum(files[p][0] for p in chunks)
    result = dict(path=str(path), chunks=len(chunks), before_bytes=before, status='planned')
    if progress:
        progress(f'{path}: {len(chunks):,} chunks, {before / 1e9:.3f} GB uncompressed')
    if dry_run:
        return result
    if workers < 1:
        raise ValueError('workers must be positive')
    from .. import _reader
    if not hasattr(_reader, '_encode_delta3d_verified'):
        raise RuntimeError('rebuild the vc_reader extension before running compress-cache')
    if shutil.disk_usage(path.parent).free < before + (1 << 30):
        raise OSError(errno.ENOSPC, f'need room for a staged copy of {before} bytes plus 1 GiB: {path}')
    started = time.monotonic()
    stage = Path(tempfile.mkdtemp(prefix=f'.{path.name}.delta3d-', dir=path.parent))
    swapped = False
    try:
        # Fail before encoding if this filesystem cannot exchange directories.
        with tempfile.TemporaryDirectory(prefix='.vc-exchange-', dir=path.parent) as probe:
            a, b = Path(probe) / 'a', Path(probe) / 'b'
            a.mkdir(); b.mkdir(); _exchange(a, b)
        after = allocated = completed = 0
        last = time.monotonic()
        iterator = iter(chunks)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = set()
            def submit():
                relative = next(iterator, None)
                if relative is not None:
                    pending.add(pool.submit(_convert_chunk, path, stage, relative, files[relative], meta))
            for _ in range(workers * 2):
                submit()
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    n, blocks = future.result()
                    after += n; allocated += blocks; completed += 1
                    submit()
                if progress and time.monotonic() - last >= 10:
                    last = time.monotonic()
                    progress(f'  {completed:,}/{len(chunks):,} chunks verified; {after / 1e9:.3f} GB written')
        chunk_set = set(chunks)
        if progress:
            progress('  preserving metadata and missing-chunk markers')
        for relative in files:
            if relative not in chunk_set and relative != Path('.zarray'):
                output = stage / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                if files[relative][0] == 0:
                    # Empty markers have no payload to copy or flush. Persist the
                    # new links with their containing directories before exchange.
                    os.link(path / relative, output)
                else:
                    shutil.copy2(path / relative, output)
                    with output.open('rb') as stream:
                        os.fsync(stream.fileno())
        compressed = dict(meta, compressor={'id': 'vc-delta3d', 'quant': 1})
        atomic_write(stage / '.zarray', json.dumps(compressed, indent=2).encode() + b'\n')
        with (stage / '.zarray').open('rb') as stream:
            os.fsync(stream.fileno())
        _verify_reader(path, stage, chunks, meta)
        current, _ = _inventory(path, meta)
        if current != files:
            raise RuntimeError(f'source array changed; stop all cache readers/writers before conversion: {path}')
        shutil.copymode(path, stage)
        for parent, _, _ in os.walk(stage, topdown=False):
            _sync_directory(Path(parent))
        _exchange(path, stage)
        swapped = True
        if progress:
            progress('  verified array published; removing original chunks')
        _sync_directory(path.parent)
        result.update(after_bytes=after, allocated_after_bytes=allocated,
                      ratio=before / after if after else None, status='converted',
                      seconds=time.monotonic() - started)
        shutil.rmtree(stage)
        _sync_directory(path.parent)
    except BaseException:
        if not swapped:
            shutil.rmtree(stage)
        # After publication, leave the original at stage if cleanup fails.
        raise
    if progress:
        progress(f'  complete: {before / 1e9:.3f} -> {after / 1e9:.3f} GB; '
                 f'{before / after:.2f}x' if after else '  complete: empty array')
    return result


def compress_cache(root, *, workers=8, dry_run=False, report=None, progress=print):
    """Convert all raw arrays under root; leave compressed arrays untouched."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'not a directory: {root}')
    if workers < 1:
        raise ValueError('workers must be positive')
    rows = []
    try:
        for path in _arrays(root):
            row = convert_array(path, workers=workers, dry_run=dry_run, progress=progress)
            if row is not None:
                rows.append(row)
            if report:
                atomic_write(report, json.dumps(dict(root=str(root), dry_run=dry_run, arrays=rows), indent=2).encode())
    finally:
        if report:
            atomic_write(report, json.dumps(dict(root=str(root), dry_run=dry_run, arrays=rows), indent=2).encode())
    before = sum(r['before_bytes'] for r in rows)
    after = sum(r.get('after_bytes', r['before_bytes']) for r in rows)
    if progress:
        progress(f'Total: {len(rows)} arrays; {before / 1e9:.3f} -> {after / 1e9:.3f} GB'
                 + (' (dry run)' if dry_run else f'; saved {(before - after) / 1e9:.3f} GB'))
    return rows
