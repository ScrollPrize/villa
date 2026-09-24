"""Chunk-cached readers for the fiber prediction zarrs and the CT volume.

Arrays are indexed ``z, y, x``. Fiber predictions use the eight-base-voxel
trace grid. CT-only inputs can use a finer native grid; ``input_scale`` maps
trace coordinates into the selected CT array. Fiber JSON uses base-voxel xyz.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numcodecs
import numpy as np
import torch

_VCZ1_REGISTERED = False
_MISSING = object()


def _register_vcz1() -> None:
    global _VCZ1_REGISTERED
    if _VCZ1_REGISTERED:
        return
    try:
        from vc.compression import vcz1_numcodecs

        vcz1_numcodecs.register()
    except ImportError:
        pass
    _VCZ1_REGISTERED = True


class ChunkedArray:
    """Minimal zarr-v2 reader with an LRU cache of decoded chunks.

    Bypasses zarr-python so reads of small, arbitrary blocks are cheap and
    worker-process friendly (only the path is pickled).
    """

    def __init__(self, path: str | Path, cache_bytes: int = 2 << 30) -> None:
        self.path = str(path)
        meta = json.loads(Path(self.path, ".zarray").read_text())
        self.shape = tuple(int(v) for v in meta["shape"])
        self.chunks = tuple(int(v) for v in meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.fill = meta.get("fill_value") or 0
        self.sep = meta.get("dimension_separator", ".")
        comp = meta.get("compressor")
        if comp is not None and comp.get("id") == "vcz1":
            _register_vcz1()
        self.codec = None if comp is None else numcodecs.get_codec(comp)
        self.cache_bytes = int(cache_bytes)
        self._cache: OrderedDict[tuple[int, int, int], np.ndarray | None] = OrderedDict()
        self._cached = 0
        self._chunk_nbytes = int(np.prod(self.chunks)) * self.dtype.itemsize
        self._lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        state["_cached"] = 0
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _load(self, key: tuple[int, int, int]) -> np.ndarray | None:
        fn = os.path.join(self.path, self.sep.join(str(k) for k in key))
        try:
            with open(fn, "rb") as fh:
                raw = fh.read()
        except FileNotFoundError:
            return None
        buf = raw if self.codec is None else self.codec.decode(raw)
        return np.frombuffer(buf, dtype=self.dtype).reshape(self.chunks)

    def chunk(self, key: tuple[int, int, int]) -> np.ndarray | None:
        with self._lock:
            hit = self._cache.get(key, _MISSING)
            if hit is not _MISSING:
                self._cache.move_to_end(key)
                return hit
        arr = self._load(key)  # decode outside the lock (codecs release the GIL)
        with self._lock:
            if key not in self._cache:
                self._cache[key] = arr
                self._cached += self._chunk_nbytes if arr is not None else 64
            while self._cached > self.cache_bytes and self._cache:
                _, old = self._cache.popitem(last=False)
                self._cached -= self._chunk_nbytes if old is not None else 64
        return arr

    def sample_nearest(self, zyx: np.ndarray) -> np.ndarray:
        """Nearest-voxel values at float zyx points (fill outside), via the chunk cache."""
        idx = np.round(np.asarray(zyx)).astype(np.int64).reshape(-1, 3)
        out = np.full(len(idx), self.fill, dtype=self.dtype)
        ok = np.all((idx >= 0) & (idx < np.asarray(self.shape)), axis=1)
        ch = np.asarray(self.chunks)
        keys = idx // ch
        order = np.lexsort(keys[:, ::-1].T)
        ko = keys[order]
        bounds = np.nonzero(np.any(np.diff(ko, axis=0) != 0, axis=1))[0] + 1
        for grp in np.split(order, bounds):
            grp = grp[ok[grp]]
            if not len(grp):
                continue
            key = tuple(int(v) for v in keys[grp[0]])
            arr = self.chunk(key)
            if arr is None:
                continue
            r = idx[grp] - np.asarray(key) * ch
            out[grp] = arr[r[:, 0], r[:, 1], r[:, 2]]
        return out.reshape(np.shape(zyx)[:-1])

    def read(self, start, size) -> np.ndarray:
        """Read block ``[start, start+size)`` (zyx); out-of-bounds is fill."""
        start = np.asarray(start, dtype=np.int64)
        size = np.asarray(size, dtype=np.int64)
        out = np.full(tuple(size), self.fill, dtype=self.dtype)
        ch = np.asarray(self.chunks)
        lo = np.maximum(start, 0)
        hi = np.minimum(start + size, np.asarray(self.shape))
        if np.any(hi <= lo):
            return out
        c0 = lo // ch
        c1 = (hi - 1) // ch
        for cz in range(c0[0], c1[0] + 1):
            for cy in range(c0[1], c1[1] + 1):
                for cx in range(c0[2], c1[2] + 1):
                    key = (int(cz), int(cy), int(cx))
                    arr = self.chunk(key)
                    if arr is None:
                        continue
                    base = np.array(key) * ch
                    a = np.maximum(lo, base)
                    b = np.minimum(hi, base + ch)
                    out[tuple(slice(a[i] - start[i], b[i] - start[i]) for i in range(3))] = arr[
                        tuple(slice(a[i] - base[i], b[i] - base[i]) for i in range(3))
                    ]
        return out


@dataclass
class FiberVolumeSpec:
    fiber_zarr_dir: str
    ct_zarr: str | None = None
    fiber_level: int = 3
    ct_level: int = 1
    ct_grid_scale: float = 8.0  # base voxels per selected CT voxel; s1_ds2 level 0 = 4
    # base voxels per trace-grid voxel (fiber OME level 3 -> 8)
    grid_scale: float = 8.0
    # "ct+presence" samples each scalar field on its own grid, without axes.
    inputs: str = "fiber"

    @property
    def mode(self) -> str:
        return self.inputs

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _find_channel_zarr(root: str, channel: str) -> str:
    for name in sorted(os.listdir(root)):
        if name.endswith(f"_{channel}.ome.zarr"):
            return os.path.join(root, name)
    raise FileNotFoundError(f"no *_{channel}.ome.zarr under {root}")


class FiberVolume:
    """Model-image readers plus presence for seed initialization.

    CT and CT + presence open no predicted direction arrays and read CT at its
    native resolution. External positions and ``shape`` remain in trace units.
    """

    def __init__(self, spec: FiberVolumeSpec, cache_bytes: int = 3 << 30) -> None:
        self.spec = spec
        if spec.mode not in ('fiber', 'fiber+ct', 'ct', 'ct+presence') or min(spec.grid_scale, spec.ct_grid_scale) <= 0:
            raise ValueError('Invalid input mode or voxel scale')
        lvl = str(spec.fiber_level)
        per = cache_bytes // (4 if spec.ct_zarr else 3)
        self.presence = ChunkedArray(os.path.join(_find_channel_zarr(spec.fiber_zarr_dir, "presence"), lvl), per)
        self.nx = self.ny = None
        if spec.mode in ('fiber', 'fiber+ct'):
            self.nx = ChunkedArray(os.path.join(_find_channel_zarr(spec.fiber_zarr_dir, "nx"), lvl), per)
            self.ny = ChunkedArray(os.path.join(_find_channel_zarr(spec.fiber_zarr_dir, "ny"), lvl), per)
        self.ct = None
        self.input_scale = spec.grid_scale/spec.ct_grid_scale if spec.mode in ('ct', 'ct+presence') else 1.
        if not self.input_scale.is_integer():
            raise ValueError('CT sampling currently requires an integer number of CT voxels per trace voxel')
        if spec.mode in ('ct', 'ct+presence', 'fiber+ct') and not spec.ct_zarr:
            raise ValueError("CT input mode needs ct_zarr")
        if spec.ct_zarr:
            # Native CT is the larger field; presence keeps its own cache.
            ct = ChunkedArray(os.path.join(spec.ct_zarr, str(spec.ct_level)),
                              int(cache_bytes * 0.75) if spec.mode in ('ct', 'ct+presence') else per)
            expected = np.asarray(self.presence.shape)*spec.grid_scale/spec.ct_grid_scale
            if np.any(np.abs(np.asarray(ct.shape)-expected) > 1):
                raise ValueError(
                    f"CT shape {ct.shape} and voxel scale {spec.ct_grid_scale} do not align "
                    f"with presence shape {self.presence.shape} at scale {spec.grid_scale}"
                )
            if spec.mode == 'fiber+ct' and (ct.shape != self.presence.shape or spec.ct_grid_scale != spec.grid_scale):
                raise ValueError('fiber+ct requires CT on the fiber grid; use ct mode for native-resolution CT')
            if ct.dtype != np.dtype('uint8'):
                raise ValueError('CT intensity normalization currently requires uint8 data')
            self.ct = ct
        self.shape = self.presence.shape

    @property
    def raw_channels(self) -> int:
        if self.spec.mode in ('ct', 'ct+presence'):
            return 1
        return 4 if self.ct is not None else 3

    @property
    def channels(self) -> int:
        """Image input channels, including independently sampled presence."""
        if self.spec.mode in ('ct', 'ct+presence'):
            return 2 if self.spec.mode == 'ct+presence' else 1
        return self.raw_channels + 4

    def fiber_raw_block(self, start, size) -> np.ndarray:
        """uint8 (presence, nx, ny) block, zyx, regardless of model inputs."""
        if self.nx is None:
            raise ValueError('CT-only mode does not open predicted direction fields')
        return np.stack([self.presence.read(start, size), self.nx.read(start, size), self.ny.read(start, size)], 0)

    def raw_block(self, start, size) -> np.ndarray:
        """Model-input uint8 block in source-array coordinates (native CT in ct mode)."""
        if self.spec.mode in ('ct', 'ct+presence'):
            return self.ct.read(start, size)[None]
        chans = [self.presence.read(start, size), self.nx.read(start, size), self.ny.read(start, size)]
        if self.ct is not None:
            chans.append(self.ct.read(start, size))
        return np.stack(chans, 0)

    def sample_image_nearest(self, points_zyx):
        """Diagnostic intensities at trace-grid coordinates, with no extra fields."""
        array = self.ct if self.spec.mode in ('ct', 'ct+presence') else self.presence
        return array.sample_nearest(np.asarray(points_zyx)*self.input_scale)


def decode_raw(raw: torch.Tensor) -> torch.Tensor:
    """(B, 3|4, ...) uint8 -> (B, 7|8, ...) float:
    presence, axis tensor (xx, yy, zz, xy, xz, yz), [ct]."""
    r = raw.float()
    pres = r[:, 0] * (1.0 / 255.0)
    nx = (r[:, 1] - 128.0) * (1.0 / 127.0)
    ny = (r[:, 2] - 128.0) * (1.0 / 127.0)
    nz = torch.sqrt(torch.clamp(1.0 - nx * nx - ny * ny, min=0.0))
    inv = torch.rsqrt(torch.clamp(nx * nx + ny * ny + nz * nz, min=1e-12))
    nx, ny, nz = nx * inv, ny * inv, nz * inv
    chans = [pres, nx * nx, ny * ny, nz * nz, nx * ny, nx * nz, ny * nz]
    if r.shape[1] > 3:
        chans.append(r[:, 3] * (1.0 / 255.0))
    return torch.stack(chans, 1)
