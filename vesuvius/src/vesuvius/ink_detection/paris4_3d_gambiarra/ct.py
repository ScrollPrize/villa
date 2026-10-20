"""uint8 CT reader for one zarr v2 array level (HTTP(S) or a local directory), with an in-memory chunk cache.

The PHercParis4 level-0 array is stored as uncompressed 128^3 uint8 chunks; an absent chunk (HTTP 404 or no
file) is zero-filled scroll exterior. Compressed arrays are decoded with numcodecs when it is installed.
"""
import hashlib
import http.client
import itertools
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PARIS4_LEVEL0 = ('https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/volumes/'
                 '20260411134726-2.400um-0.2m-78keV-masked.zarr/0/')


class Absent(KeyError):
    """A chunk the store does not have (zero-filled in the volume)."""


class ChunkStore:
    def __init__(self, source=PARIS4_LEVEL0, cache_gib=4., threads=16, retries=6):
        self.source = str(source).rstrip('/') + '/'
        self.remote = self.source.startswith(('http://', 'https://'))
        self.retries = retries
        self.stats = dict(reads=0, absent=0, bytes=0, retries=0)
        self.meta = json.loads(self._read('.zarray'))
        assert self.meta['zarr_format'] == 2 and self.meta['dtype'] == '|u1' and self.meta['order'] == 'C', self.meta
        assert not self.meta.get('filters'), 'zarr filters are not supported'
        self.shape = tuple(self.meta['shape'])
        self.chunks = tuple(self.meta['chunks'])
        self.separator = self.meta.get('dimension_separator', '.')
        self.fill = int(self.meta.get('fill_value') or 0)
        assert self.fill == 0, 'PHercParis4 inference requires zero-filled absent chunks'
        self.codec = None
        if self.meta['compressor'] is not None:
            import numcodecs
            self.codec = numcodecs.get_codec(self.meta['compressor'])
        self.cache, self.cache_bytes, self.limit = OrderedDict(), 0, int(cache_gib * (1 << 30))
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(threads)

    def _read(self, name):
        if not self.remote:
            path = Path(self.source) / name
            if not path.exists():
                raise Absent(name)
            return path.read_bytes()
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(self.source + name, timeout=120) as response:
                    return response.read()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise Absent(name) from None
                if e.code in (401, 403):
                    raise
                if attempt == self.retries - 1:
                    raise
            except (OSError, http.client.HTTPException):  # includes truncated reads
                if attempt == self.retries - 1:
                    raise
            self.stats['retries'] += 1
            time.sleep(min(60, 2 ** attempt))

    def _fetch(self, key):
        name = self.separator.join(map(str, key))
        try:
            data = self._read(name)
        except Absent:
            self.stats['absent'] += 1
            return None
        if self.codec is not None:
            data = self.codec.decode(data)
        chunk = np.frombuffer(data, np.uint8)
        assert chunk.size == np.prod(self.chunks), (name, chunk.size)
        self.stats['reads'] += 1
        self.stats['bytes'] += chunk.size
        return chunk.reshape(self.chunks)

    def exists(self, key):
        """Is chunk `key` stored? (HTTP HEAD, or a file test.)"""
        name = self.separator.join(map(str, key))
        if not self.remote:
            return (Path(self.source) / name).exists()
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.source + name, method='HEAD'), timeout=60):
                    return True
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return False
                if e.code in (401, 403):
                    raise
                if attempt == self.retries - 1:
                    raise
            except (OSError, http.client.HTTPException):
                if attempt == self.retries - 1:
                    raise
            time.sleep(min(60, 2 ** attempt))

    def identity(self):
        """The array a run reads: its source (absolute path when local) and .zarray metadata. Chunk bytes are not
        hashed; the source must be immutable."""
        source = self.source if self.remote else str(Path(self.source).resolve()) + '/'
        return dict(source=source, zarray=self.meta)

    def grid(self):
        return tuple(-(-s // c) for s, c in zip(self.shape, self.chunks))

    def probe(self, c0, c1):
        """bool presence of chunks [c0, c1) by one exists() call per chunk."""
        keys = list(itertools.product(*(range(int(a), int(b)) for a, b in zip(c0, c1))))
        found = list(self.pool.map(self.exists, keys))
        return np.array(found, bool).reshape(tuple(int(b) - int(a) for a, b in zip(c0, c1)))

    def presence(self, threads=32):
        """bool chunk presence of the whole array: a directory walk (local) or S3 ListObjectsV2 per first-axis
        index (an https://<bucket>.s3.amazonaws.com/<prefix> source). Other HTTP servers: use probe()."""
        present = np.zeros(self.grid(), bool)
        sep = self.separator
        pattern = re.compile(r'(\d+)' + re.escape(sep) + r'(\d+)' + re.escape(sep) + r'(\d+)')
        if not self.remote:
            root = Path(self.source)
            for dirpath, _, files in os.walk(root):
                for f in files:
                    m = pattern.fullmatch(os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, '/'))
                    if m:
                        present[tuple(int(v) for v in m.groups())] = True
            return present
        url = urllib.parse.urlsplit(self.source)
        assert url.netloc.endswith('amazonaws.com') and '.s3' in url.netloc, \
            f'chunk listing needs an S3 bucket URL ({self.source}); use probe()'
        base, prefix = f'{url.scheme}://{url.netloc}/', url.path.lstrip('/')
        key = re.compile(r'<Key>' + re.escape(prefix) + r'(\d+)' + re.escape(sep) + r'(\d+)' + re.escape(sep) + r'(\d+)</Key>')

        def list_z(z):
            keys, token = [], None
            while True:
                query = {'list-type': '2', 'prefix': f'{prefix}{z}{sep}', 'max-keys': '1000'}
                if token:
                    query['continuation-token'] = token
                for attempt in range(self.retries):
                    try:
                        with urllib.request.urlopen(base + '?' + urllib.parse.urlencode(query), timeout=120) as r:
                            body = r.read().decode()
                        break
                    except (OSError, http.client.HTTPException):
                        if attempt == self.retries - 1:
                            raise
                        time.sleep(min(60, 2 ** attempt))
                if '<ListBucketResult' not in body:
                    raise ValueError(f'not an S3 ListObjectsV2 response: {body[:200]}')
                keys += [tuple(int(v) for v in m.groups()) for m in key.finditer(body)]
                if '<IsTruncated>true</IsTruncated>' not in body:
                    return keys
                token = re.search(r'<NextContinuationToken>([^<]+)</NextContinuationToken>', body).group(1)

        with ThreadPoolExecutor(threads) as pool:
            for keys in pool.map(list_z, range(present.shape[0])):
                for k in keys:
                    present[k] = True
        return present

    def chunk(self, key):
        """The chunk at grid index `key` (read-only uint8 array), or None if absent."""
        key = tuple(int(v) for v in key)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        value = self._fetch(key)
        with self.lock:
            if key not in self.cache:
                self.cache[key] = value
                self.cache_bytes += 0 if value is None else value.nbytes
                while self.cache_bytes > self.limit and len(self.cache) > 1:
                    old = self.cache.popitem(last=False)[1]
                    self.cache_bytes -= 0 if old is None else old.nbytes
            return self.cache[key]

    def keys(self, origin, size):
        origin, end = np.asarray(origin, np.int64), np.asarray(origin, np.int64) + np.asarray(size, np.int64)
        return list(itertools.product(*(range(max(int(a), 0) // c, (min(int(b), s) - 1) // c + 1)
                                        for a, b, c, s in zip(origin, end, self.chunks, self.shape))))

    def box(self, origin, size, strict=True):
        """uint8 block [origin, origin + size). strict: every voxel inside the array and every chunk present
        (training crops); otherwise absent chunks and voxels outside the array read as the fill value."""
        origin = np.asarray(origin, np.int64)
        size = np.broadcast_to(np.asarray(size, np.int64), (3,))
        end = origin + size
        if strict:
            assert np.all(origin >= 0) and np.all(end <= np.array(self.shape)), (origin, end)
        keys = self.keys(origin, size)
        out = np.full(tuple(size), self.fill, np.uint8)
        for key, chunk in zip(keys, self.pool.map(self.chunk, keys)):
            if chunk is None:
                if strict:
                    raise Absent(f'chunk {key} is absent inside a training crop')
                continue
            start = np.array(key) * np.array(self.chunks)
            lo = np.maximum(start, origin)
            hi = np.minimum(np.minimum(start + np.array(self.chunks), end), self.shape)
            out[tuple(slice(int(a), int(b)) for a, b in zip(lo - origin, hi - origin))] = \
                chunk[tuple(slice(int(a), int(b)) for a, b in zip(lo - start, hi - start))]
        return out

    def close(self):
        self.pool.shutdown(wait=True)


def percentiles(raw):
    """(p1, p99) of a raw block, as Python floats: np.percentile of its float32 copy (linear interpolation)."""
    return tuple(float(np.percentile(raw.astype(np.float32), p)) for p in (1, 99))


def normalize(raw, lo, hi):
    """Clip to [lo, hi] and scale to [0, 1] in float32 (all zeros if hi - lo < 1e-6)."""
    x = np.array(raw, dtype=np.float32, copy=True)
    np.clip(x, lo, hi, out=x)
    if hi - lo < 1e-6:
        x.fill(0)
    else:
        x -= lo
        x /= (hi - lo)
    return x


def sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class Parents:
    """Raw 256^3 parent blocks and their p1/p99, with an LRU of decoded blocks (thread-safe).

    origins: callable index -> parent origin (ZYX). context(i) returns the 320^3 block at origin - 32 (teacher
    stage 1 rotations reach 32 voxels outside the parent); parent(i) the 256^3 block at origin.
    expected: optional dict index -> raw256 sha256 to verify (unlabelled pool)."""
    def __init__(self, store, origins, cache_gib=16., expected=None):
        self.store, self.origins, self.expected = store, origins, expected or {}
        self.cache, self.bytes, self.limit = OrderedDict(), 0, int(cache_gib * (1 << 30))
        self.lock = threading.Lock()
        self.records = {}

    def _get(self, key, make):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        value = make()
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            self.cache[key] = value
            self.bytes += value.nbytes
            while self.bytes > self.limit and len(self.cache) > 1:
                self.bytes -= self.cache.popitem(last=False)[1].nbytes
        return value

    def context(self, i):
        return self._get(('context', i), lambda: self.store.box(np.asarray(self.origins(i)) - 32, 320))

    def parent(self, i, from_context=False):
        if from_context:
            return self.context(i)[32:288, 32:288, 32:288]
        return self._get(('parent', i), lambda: self.store.box(self.origins(i), 256))

    def limits(self, i, raw=None):
        """(p1, p99) of parent i (computed once, then remembered)."""
        if i not in self.records:
            raw = self.parent(i) if raw is None else raw
            if i in self.expected:
                assert sha256(raw) == self.expected[i], f'parent {i}: CT bytes differ from the recorded sha256'
            self.records[i] = percentiles(raw)
        return self.records[i]
