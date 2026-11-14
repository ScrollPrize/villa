"""Chunk-level reader for the raw CT (a Zarr v2 uint8 array, local path or URL).

The CT is never copied to disk. Chunks the next z slab needs are fetched into
memory before the slab runs and the others are released; a chunk a profile
needs that the plan did not prefetch is fetched on demand and counted, so no
profile is lost at a slab edge.
"""

from __future__ import annotations

import itertools
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import remote_zarr


class ZarrChunkStore:
    """In-memory chunk cache over a Zarr v2 uint8 array (local or http)."""

    def __init__(self, location, fetch_threads=16):
        self.location = str(location).rstrip("/")
        self.fetch_threads = max(1, int(fetch_threads))
        self._local = threading.local()
        meta = remote_zarr.get_json(self._session(), f"{self.location}/.zarray")
        if int(meta.get("zarr_format", 2)) != 2:
            raise ValueError(f"{self.location}: only Zarr v2 arrays are supported")
        if len(meta["shape"]) != 3 or np.dtype(meta["dtype"]) != np.uint8:
            raise ValueError(f"{self.location}: expected a 3-D uint8 array, got "
                             f"shape {meta['shape']} dtype {meta['dtype']}")
        if meta.get("order", "C") != "C" or meta.get("filters"):
            raise ValueError(f"{self.location}: only C-order arrays without filters are supported")
        self.meta = meta
        self.shape = tuple(int(v) for v in meta["shape"])
        self.chunks = tuple(int(v) for v in meta["chunks"])
        self.grid = tuple(-(-s // c) for s, c in zip(self.shape, self.chunks))
        self.fill = int(meta.get("fill_value") or 0)
        self.codec = None
        if meta.get("compressor"):
            import numcodecs

            self.codec = numcodecs.get_codec(meta["compressor"])
        self.chunk_bytes = int(np.prod(self.chunks))
        self._resident = {}  # linear chunk key -> uint8 array, or None when absent
        self.stats = {"chunks_fetched": 0, "chunks_absent": 0, "bytes_fetched": 0, "on_demand_chunks": 0}

    # -- keys ---------------------------------------------------------------
    def linear(self, kz, ky, kx):
        return (np.asarray(kz, np.int64) * self.grid[1] + ky) * self.grid[2] + kx

    def unravel(self, key):
        key = int(key)
        kx = key % self.grid[2]
        ky = (key // self.grid[2]) % self.grid[1]
        return key // (self.grid[1] * self.grid[2]), ky, kx

    def keys_near(self, points, margin):
        """Linear keys of every chunk within ``margin`` voxels of the points (zyx)."""
        points = np.asarray(points, np.float64)
        points = points[np.isfinite(points).all(1)]
        keys = []
        for dz in (-margin, margin):
            for dy in (-margin, margin):
                for dx in (-margin, margin):
                    q = np.floor(points + (dz, dy, dx)).astype(np.int64)
                    k = q // np.asarray(self.chunks)
                    ok = ((k >= 0) & (k < np.asarray(self.grid))).all(1)
                    keys.append(self.linear(k[ok, 0], k[ok, 1], k[ok, 2]))
        return np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)

    # -- fetching -------------------------------------------------------------
    def _session(self):
        session = getattr(self._local, "session", None)
        if session is None or getattr(self._local, "pid", None) != os.getpid():
            import requests

            session = requests.Session()
            self._local.session = session
            self._local.pid = os.getpid()
        return session

    def _fetch(self, key):
        index = self.unravel(key)
        raw = remote_zarr.fetch(self._session(), remote_zarr.chunk_url(self.location, index, self.meta))
        if raw is None:
            return key, None
        if self.codec is not None:
            raw = self.codec.decode(raw)
        data = np.frombuffer(raw, np.uint8)
        if data.size != self.chunk_bytes:
            raise ValueError(f"chunk {index} of {self.location}: {data.size} bytes, expected {self.chunk_bytes}")
        return key, data.reshape(self.chunks)

    def _store(self, key, data):
        self._resident[int(key)] = data
        if data is None:
            self.stats["chunks_absent"] += 1
        else:
            self.stats["chunks_fetched"] += 1
            self.stats["bytes_fetched"] += self.chunk_bytes

    def retain(self, keys):
        """Keep exactly ``keys`` resident: release the others, fetch the missing."""
        want = {int(k) for k in keys}
        for key in list(self._resident):
            if key not in want:
                del self._resident[key]
        missing = sorted(want - set(self._resident))
        if not missing:
            return
        with ThreadPoolExecutor(self.fetch_threads) as pool:
            for key, data in pool.map(self._fetch, missing):
                self._store(key, data)

    def resident_bytes(self):
        return sum(self.chunk_bytes for v in self._resident.values() if v is not None)

    # -- sampling -------------------------------------------------------------
    def sample_int(self, zi, yi, xi):
        """Integer voxel lookups -> float32. A z outside the array's chunk rows
        gives NaN; y or x outside the array gives 0; an absent chunk gives the
        array's fill value."""
        zi = np.asarray(zi, np.int64)
        yi = np.asarray(yi, np.int64)
        xi = np.asarray(xi, np.int64)
        out = np.zeros(len(zi), np.float32)
        inz = (zi >= 0) & (zi < self.grid[0] * self.chunks[0])
        out[~inz] = np.nan
        ok = inz & (yi >= 0) & (yi < self.shape[1]) & (xi >= 0) & (xi < self.shape[2])
        sel = np.flatnonzero(ok)
        if not len(sel):
            return out
        cz, cy, cx = self.chunks
        key = self.linear(zi[sel] // cz, yi[sel] // cy, xi[sel] // cx)
        order = np.argsort(key, kind="stable")
        key = key[order]
        sel = sel[order]
        cuts = np.concatenate([[0], np.flatnonzero(np.diff(key)) + 1, [len(key)]])
        for a, b in itertools.pairwise(cuts):
            k = int(key[a])
            if k not in self._resident:
                self._store(k, self._fetch(k)[1])
                self.stats["on_demand_chunks"] += 1
            data = self._resident[k]
            s = sel[a:b]
            if data is None:
                out[s] = self.fill
            else:
                out[s] = data[zi[s] % cz, yi[s] % cy, xi[s] % cx]
        return out
