"""Chunk-cached readers for the fiber prediction zarrs and the CT volume.

Arrays are indexed ``z, y, x``. Fiber predictions use the eight-base-voxel
trace grid. CT-only inputs can use a finer native grid; ``input_scale`` maps
trace coordinates into the selected CT array. Fiber JSON uses base-voxel xyz.
"""

from __future__ import annotations

import json
import mmap
import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numcodecs
import numpy as np

_VCZ1_REGISTERED = False
_MISSING = object()
_MADV_WILLNEED = getattr(mmap, 'MADV_WILLNEED', None)


def _willneed_rows(arr, lo, hi) -> None:
    """Queue reads of the pages that hold rows ``[lo[1], hi[1])`` of slices ``[lo[0], hi[0])`` of a mapped chunk.

    A fault on a file mapping otherwise reads the device's whole read-around window
    (3 MB on the md RAID, i.e. the full 2 MB chunk), one blocking fault at a time.
    The hint reads only the touched pages, all in flight before the copy faults on
    them. Advisory only: the copied values are unchanged.

    Each hint is a syscall, so when the rows cover at least a quarter of each slice
    one hint spans the whole slab (reading at most 4x the rows) instead of one per slice.
    """
    mm = getattr(arr, '_mmap', None)
    if mm is None or _MADV_WILLNEED is None or getattr(arr, 'offset', 0) != 0:
        return
    page = mmap.PAGESIZE
    plane, row = arr.strides[0], arr.strides[1]
    z0, z1, first = int(lo[0]), int(hi[0]), int(lo[1])*row
    need = (int(hi[1])-int(lo[1]))*row
    spans = ([(z0*plane+first, (z1-1)*plane+first+need)] if plane-need <= 3*need else
             [(z*plane+first, z*plane+first+need) for z in range(z0, z1)])
    try:
        for s, e in spans:
            s = s//page*page
            mm.madvise(_MADV_WILLNEED, s, -(-e//page)*page-s)
    except OSError:
        pass  # a hint the platform refuses only costs speed


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
    worker-process friendly (only the path is pickled). Uncompressed arrays
    (see ``decode_store.py``) are memory-mapped chunk by chunk, so processes
    share them through the page cache and the cache bounds live mappings
    rather than private memory. Each mapping holds a file descriptor, so the
    trainer raises its soft open-file limit at startup.
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
        if self.codec is None:
            try:
                return np.memmap(fn, dtype=self.dtype, mode="r", shape=self.chunks)
            except FileNotFoundError:
                return None
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
        pieces = []
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
                    pieces.append((arr, base, a, b))
        if self.codec is None:
            # Queue every mapped piece's pages before the first copy blocks on one.
            for arr, base, a, b in pieces:
                _willneed_rows(arr, a - base, b - base)
        for arr, base, a, b in pieces:
            out[tuple(slice(a[i] - start[i], b[i] - start[i]) for i in range(3))] = arr[
                tuple(slice(a[i] - base[i], b[i] - base[i]) for i in range(3))
            ]
        return out


class RemoteChunkedArray(ChunkedArray):
    """Fetch on demand into an uncompressed, memory-mapped local Zarr v2.

    Only decoded chunks are persisted. Atomic replacement permits independent
    loader/collector processes to share the cache. Cached reads need no network.
    """
    def __init__(self, path, level, cache_dir, cache_bytes, *, cache_only=False):
        if not cache_dir:
            raise ValueError('Remote CT requires a persistent cache_dir')
        self.remote_url, self.level = str(path).rstrip('/'), int(level)
        self.cache_dir = str(cache_dir)
        self.cache_only = cache_only
        local = self.cache_path(self.remote_url,self.level,cache_dir)
        self._array = None
        if not (local/'.zarray').exists():
            if cache_only:
                raise FileNotFoundError(f'Remote CT metadata is not prefetched: {local}')
            self._array = self._open()
            self.publish_metadata(local,self._array,self.remote_url,self.level)
        super().__init__(local,cache_bytes)

    @staticmethod
    def cache_path(url,level,cache_dir):
        source = urlsplit(str(url).rstrip('/'))
        return Path(cache_dir)/source.scheme/source.netloc/source.path.lstrip('/')/str(level)

    @classmethod
    def publish_metadata(cls,local,array,url,level):
        if len(array.shape) != 3:
            raise ValueError('Remote CT must be a three-dimensional ZYX array')
        metadata = dict(zarr_format=2,shape=list(array.shape),chunks=list(array.chunks),
            dtype=np.dtype(array.dtype).str,fill_value=np.asarray(getattr(array,'metadata',array).fill_value or 0).item(),
            compressor=None,filters=None,order='C',
            dimension_separator=getattr(getattr(array,'metadata',array),'dimension_separator','.'))
        cls._atomic_write(local/'.zattrs',json.dumps(dict(remote_url=url,level=level)).encode())
        cls._atomic_write(local/'.zarray',json.dumps(metadata).encode())

    @staticmethod
    def _atomic_write(path, data):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.partial-',delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            os.replace(temporary,path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _open(self):
        import zarr
        return zarr.open(remote_store(self.remote_url),path=str(self.level),mode='r')

    def __getstate__(self):
        return {**super().__getstate__(), '_array': None}

    def _load(self, key, *, blocking=True):
        cached = super()._load(key)
        if cached is not None:
            return cached
        if self.cache_only:
            raise FileNotFoundError(f'Remote CT chunk is not prefetched: {self.path}/{key}')
        # POSIX advisory locks work on both supported platforms (Linux/macOS).
        # Separate file handles serialize threads as well as loader processes.
        # Cached reads avoid the lock. Prefetch skips chunks already in flight;
        # a foreground reader waits for that exact chunk, without downloading it twice.
        import fcntl
        lock_path = Path(self.path)/'.locks'/'.'.join(map(str,key))
        lock_path.parent.mkdir(exist_ok=True)
        with lock_path.open('a+b') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                return None
            cached = super()._load(key)
            if cached is not None:
                return cached
            with self._lock:
                if self._array is None:
                    self._array = self._open()
            start = np.asarray(key)*self.chunks
            stop = np.minimum(start+self.chunks, self.shape)
            values = np.asarray(self._array[tuple(slice(int(a), int(b)) for a,b in zip(start,stop))])
            result = np.full(self.chunks, self.fill, self.dtype)
            result[tuple(slice(0,n) for n in values.shape)] = values
            self._atomic_write(Path(self.path)/self.sep.join(map(str,key)),result.tobytes(order='C'))
            return super()._load(key)


def remote_store(url, *, max_pool_connections=None):
    from vesuvius.neural_tracing.datasets.common import _make_remote_store
    from vesuvius.neural_tracing.s3_utils import s3_storage_options_for_path
    _register_vcz1()
    options = s3_storage_options_for_path(url) if url.startswith('s3://') else {}
    if url.startswith('s3://') and max_pool_connections is not None:
        options['config_kwargs'] = {'max_pool_connections': max_pool_connections}
    return _make_remote_store(url,options,missing_exceptions=(KeyError,FileNotFoundError))


async def open_remote_array(url,level, *, max_pool_connections=None):
    """Native async Zarr I/O; no synchronous array reads on the event loop."""
    from zarr.api.asynchronous import open_array
    return await open_array(store=remote_store(url,max_pool_connections=max_pool_connections),
                            path=str(level),mode='r')


@dataclass
class FiberVolumeSpec:
    fiber_zarr_dir: str
    ct_zarr: str | None = None
    fiber_level: int = 3
    ct_level: int = 1
    ct_grid_scale: float = 8.0  # base voxels per selected CT voxel; s1_ds2 level 0 = 4
    # base voxels per trace-grid voxel (fiber OME level 3 -> 8)
    grid_scale: float = 8.0
    # "ct" or "ct+presence" (each scalar field sampled on its own grid). The
    # retired "fiber" default is kept so older metadata compares unchanged.
    inputs: str = "fiber"
    load_presence: bool = True  # False: CT metadata alone defines tracing bounds.
    cache_dir: str | None = None  # Persistent uncompressed remote CT chunks.
    ct_normalization: dict | None = None  # Bound at startup, persisted in run JSON/checkpoints.

    @property
    def mode(self) -> str:
        return self.inputs

    def to_dict(self) -> dict:
        result = dict(self.__dict__)
        # Keep legacy volume metadata byte-for-byte comparable on old resumes.
        if self.cache_dir is None:
            result.pop('cache_dir')
        return result


def _find_channel_zarr(root: str, channel: str) -> str:
    for name in sorted(os.listdir(root)):
        if name.endswith(f"_{channel}.ome.zarr"):
            return os.path.join(root, name)
    raise FileNotFoundError(f"no *_{channel}.ome.zarr under {root}")


def _grid_metadata(array_path: Path):
    """Comparable OME grid declaration, when present (bare Zarr is also valid)."""
    attrs = array_path.parent / '.zattrs'
    if not attrs.exists():
        return None
    meta = json.loads(attrs.read_text()).get('multiscales', [])
    if not meta:
        return None
    scale = meta[0]
    axes = scale.get('axes', [])
    if [a.get('name') if isinstance(a, dict) else a for a in axes] != ['z', 'y', 'x']:
        raise ValueError(f'Expected z,y,x direction/presence axes: {array_path}')
    dataset = next((d for d in scale['datasets'] if d['path'] == array_path.name), None)
    if dataset is None:
        raise ValueError(f'OME metadata does not declare level: {array_path}')
    return axes, scale.get('coordinateTransformations'), dataset.get('coordinateTransformations')


class FiberVolume:
    """Model-image readers plus presence for seed initialization.

    CT is read at its native resolution. External positions and ``shape``
    remain in trace units.
    """

    def __init__(self, spec: FiberVolumeSpec, cache_bytes: int = 3 << 30, *, cache_only=False) -> None:
        self.spec = spec
        self._directions = None
        self._cache_bytes = int(cache_bytes)
        if spec.mode not in ('ct', 'ct+presence') or min(spec.grid_scale, spec.ct_grid_scale) <= 0:
            raise ValueError('Invalid input mode or voxel scale; direction-field inputs are no longer supported')
        lvl = str(spec.fiber_level)
        if not spec.load_presence and spec.mode != 'ct':
            raise ValueError('Presence-free volume reading requires CT-only inputs')
        self.presence = (ChunkedArray(os.path.join(_find_channel_zarr(spec.fiber_zarr_dir, "presence"), lvl), cache_bytes // 4)
                         if spec.load_presence else None)
        self._seed_presence = None
        self.input_scale = spec.grid_scale/spec.ct_grid_scale
        if not self.input_scale.is_integer():
            raise ValueError('CT sampling currently requires an integer number of CT voxels per trace voxel')
        if not spec.ct_zarr:
            raise ValueError("CT input mode needs ct_zarr")
        # Native CT is the larger field; presence keeps its own cache.
        ct = (RemoteChunkedArray(spec.ct_zarr, spec.ct_level, spec.cache_dir, int(cache_bytes * .75),cache_only=cache_only)
              if spec.ct_zarr.startswith(('s3://','http://','https://')) else
              ChunkedArray(os.path.join(spec.ct_zarr, str(spec.ct_level)), int(cache_bytes * 0.75)))
        expected = np.asarray(self.presence.shape)*spec.grid_scale/spec.ct_grid_scale if self.presence is not None else np.asarray(ct.shape)
        if np.any(np.abs(np.asarray(ct.shape)-expected) > 1):
            raise ValueError(
                f"CT shape {ct.shape} and voxel scale {spec.ct_grid_scale} do not align "
                f"with presence shape {self.presence.shape} at scale {spec.grid_scale}"
            )
        if ct.dtype != np.dtype('uint8'):
            raise ValueError('CT intensity normalization currently requires uint8 data')
        self.ct = ct
        if spec.ct_normalization is not None:
            from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import validate_record
            validate_record(spec.ct_normalization, spec)
        self.shape = (self.presence.shape if self.presence is not None else
                      tuple(np.ceil(np.asarray(self.ct.shape)/self.input_scale).astype(int)))

    def presence_for_seeding(self):
        """Explicit seed-selection dependency, never opened for CT-only model crops."""
        if self.presence is not None:
            return self.presence
        if self._seed_presence is None:
            self._seed_presence = ChunkedArray(os.path.join(
                _find_channel_zarr(self.spec.fiber_zarr_dir, 'presence'), str(self.spec.fiber_level)), 64 << 20)
        return self._seed_presence

    def direction_fields(self):
        """Lazily open nx/ny siblings of the selected presence store, on its grid.

        Legacy CT/presence callers never open these files. Reallocate one quarter
        of the reader budget from CT to the two direction caches when enabled.
        """
        if self._directions is None:
            if self.presence is None:
                raise ValueError('Direction inputs require a presence store')
            presence = Path(self.presence.path)
            suffix = '_presence.ome.zarr'
            if not presence.parent.name.endswith(suffix):
                raise ValueError('Cannot derive nx/ny siblings from presence store name')
            prefix = presence.parent.name[:-len(suffix)]
            grid = _grid_metadata(presence)
            fields = []
            for channel in ('nx', 'ny'):
                path = presence.parent.with_name(f'{prefix}_{channel}.ome.zarr') / presence.name
                field = ChunkedArray(path, self._cache_bytes // 8)
                if field.shape != self.presence.shape or field.dtype != np.dtype('uint8') or field.fill != 0:
                    raise ValueError(f'{channel}: expected uint8, zero fill and the presence grid shape')
                if _grid_metadata(path) != grid:
                    raise ValueError(f'{channel}: OME axes/scale/origin differ from the presence grid')
                fields.append(field)
            self.ct.cache_bytes = self._cache_bytes // 2
            self._directions = tuple(fields)
        return self._directions

    @property
    def channels(self) -> int:
        """Image input channels, including independently sampled presence."""
        return 2 if self.spec.mode == 'ct+presence' else 1

    def raw_block(self, start, size) -> np.ndarray:
        """Model-input uint8 (1, ...) native CT block in source-array coordinates."""
        return self.ct.read(start, size)[None]

    def sample_image_nearest(self, points_zyx):
        """Diagnostic CT intensities at trace-grid coordinates."""
        return self.ct.sample_nearest(np.asarray(points_zyx)*self.input_scale)
