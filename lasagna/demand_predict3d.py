"""Demand scheduling around shared GPU, accumulation and Zarr finalization code.

Coordinator owns metadata; chunk-owner workers add shared-RAM results in arrival
order. Only finalized Zarr chunks survive a session; scratch is not durable truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import replace
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import errno
import itertools
import heapq
import faulthandler
import json
import math
import mmap
import os
from pathlib import Path
import shutil
import tempfile
import multiprocessing as mp
import queue
import re
import threading
import time

import numpy as np
import torch

from lasagna import tiled_predict3d as tiled
from lasagna.inference_provenance import atomic_write
from lasagna.inference_profiling import stage_timing


# Ext4 with 4 KiB blocks rejects individual files at 16 TiB, and many
# concurrent publication threads cannot safely map multi-terabyte extents even
# when those files are sparse. Keep each temporary accumulator shard at most
# 128 GiB where whole-plane boundaries permit while retaining one logical
# full-volume array in the shared descriptor. An indivisible plane may exceed
# that target, but remains bounded well below ext4's 16-TiB limit.
MMAP_SHARD_TARGET_BYTES = 128 << 30
MAX_MMAP_FILE_BYTES = 8 << 40


def prepare_background_requests(stores, backgrounds, *, workers, check_cancelled=lambda: None):
    """Prepare uncapped startup requests in parallel before any pool is pumped.

    Workers only read caches/geometry. The coordinator installs records in
    request order; ordinary request() subsequently owns heap/publication updates.
    Bounded batches avoid one future per chunk and unlimited completed results.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("startup workers must be a positive integer")
    local = threading.local()

    def prepare(item):
        key, index = item
        store = stores[key]
        origin = store.geometry.origin(index)
        if origin in store.chunks:
            return key, origin, None
        if not hasattr(local, "arrays"):
            local.arrays = {}
        ready = store._cached_chunk_complete(origin, read_arrays=local.arrays)
        dependencies = None if ready else store.geometry.contributors(origin)
        return key, origin, (ready, dependencies)

    requests = ((key, index) for key, indices in backgrounds.items() for index in indices)
    count = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cache-startup") as pool:
        while batch := list(itertools.islice(requests, workers * 4)):
            check_cancelled()
            # Materialize before installing so workers never race dictionary writes.
            prepared = list(pool.map(prepare, batch))
            for key, origin, result in prepared:
                store = stores[key]
                if result is None or origin in store.chunks:
                    continue
                ready, dependencies = result
                if ready:
                    store.chunks[origin] = {"ready": True}
                    store.stats["warm_chunks"] += 1
                else:
                    store._register_missing(origin, dependencies)
            count += len(batch)
            if count % (workers * 40) == 0:
                print(f"cache-startup prepared={count} workers={workers} elapsed={time.monotonic()-started:.1f}s", flush=True)
    print(f"cache-startup complete prepared={count} workers={workers} elapsed={time.monotonic()-started:.1f}s", flush=True)


def cleanup_stale_live_tmp(channel_paths: Iterable[Path]) -> int:
    """Remove abandoned accumulator sessions from known channel roots.

    Only directories named ``session-*`` directly below a channel's
    ``.live_tmp`` directory are eligible. Symlinks and unrelated files are
    never followed or removed. The caller must hold the cache writer lock, so
    no live writer can own one of these sessions while the scan runs.
    """
    removed = 0
    for channel_path in channel_paths:
        live_tmp = Path(channel_path) / ".live_tmp"
        try:
            entries = tuple(os.scandir(live_tmp))
        except FileNotFoundError:
            continue
        for entry in entries:
            if not entry.name.startswith("session-") or entry.is_symlink():
                continue
            if not entry.is_dir(follow_symlinks=False):
                continue
            shutil.rmtree(entry.path)
            removed += 1
    return removed


@dataclass(frozen=True)
class _MmapHandle:
    paths: tuple[str, ...]
    shard_axis0_sizes: tuple[int, ...]
    dtype: str


def cache_identity(identity, geometry, accumulator_dtype, accumulator_layout, accumulator_prefault):
    """Persistent cache contract shared by store validation and cache discovery."""
    return json.loads(json.dumps({"schema": 2, "identity": identity, "geometry": asdict(geometry),
        "accumulator_dtype": str(np.dtype(accumulator_dtype)), "accumulation_order": "arrival",
        "accumulator_layout": accumulator_layout, "accumulator_prefault": accumulator_prefault},
        sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class DemandGeometry:
    shape: tuple[int, int, int]
    tile: int
    overlap: int
    border: int
    scale: int
    chunk: int

    def __post_init__(self):
        if (len(self.shape) != 3 or min(self.shape) < 1 or self.tile < 1 or self.chunk < 1
                or self.scale < 1 or self.scale & (self.scale - 1)
                or not 0 <= self.border * 2 < self.tile
                or not self.border * 2 <= self.overlap < self.tile
                or self.tile % self.scale or self.border % self.scale
                or (self.tile - self.overlap) % self.scale):
            raise ValueError("invalid demand inference geometry/alignment")

    @property
    def output_shape(self):
        return tuple((v + self.scale - 1) // self.scale for v in self.shape)

    def origin(self, index):
        index = tuple(int(v) for v in index)
        if len(index) != 3 or any(i < 0 or i * self.chunk >= n for i, n in zip(index, self.output_shape)):
            raise ValueError(f"invalid output chunk index: {index}")
        return tuple(i * self.chunk for i in index)

    def contributors(self, origin):
        axes = [tiled._canonical_tile_positions_for_output_region(
            volume_size=n, output_start=o, output_end=min(o + self.chunk, out),
            scaledown=self.scale, tile_size=self.tile, stride=self.tile - self.overlap,
            border=self.border, scaledown_multiple=self.scale,
        ) for n, o, out in zip(self.shape, origin, self.output_shape)]
        return tuple(itertools.product(*axes))

    def tile_origin(self, tile):
        return tuple(v // self.scale - self.border // self.scale for v in tile)

    def touched(self, tile):
        start = self.tile_origin(tile)
        axes = [range(max(0, o) // self.chunk * self.chunk,
                      min(n, o + self.tile // self.scale), self.chunk)
                for o, n in zip(start, self.output_shape)]
        return itertools.product(*axes)

    def intersection(self, origin, tile):
        start = self.tile_origin(tile)
        lo = tuple(max(o, t, 0) for o, t in zip(origin, start))
        hi = tuple(min(o + self.chunk, t + self.tile // self.scale, n)
                   for o, t, n in zip(origin, start, self.output_shape))
        if any(a >= b for a, b in zip(lo, hi)):
            raise ValueError("nonintersecting contribution")
        destination = tuple(slice(a, b) for a, b in zip(lo, hi))
        source = tuple(slice(a - t, b - t) for a, b, t in zip(lo, hi, start))
        return destination, source


class DemandStore:
    """RAM-only scheduling plus full-logical-shape mmap accumulation for a source.

    Each tile contributes once to every touched (including unrequested) chunk.
    Partial neighbors never schedule work. Only the coordinator changes metadata.
    """
    def __init__(self, root, identity, geometry, model_adapter, output_adapter,
                 *, accumulator_dtype="float16", accumulator_layout="flat", accumulator_prefault=False,
                 prepare_output=None):
        if accumulator_layout not in ("flat", "chunk-major"):
            raise ValueError("accumulator_layout must be flat or chunk-major")
        if not isinstance(accumulator_prefault, bool):
            raise ValueError("accumulator_prefault must be boolean")
        if accumulator_prefault:
            if accumulator_layout != "chunk-major":
                raise ValueError("writable prefault requires chunk-major layout")
            tiled._mmap_populate_write_advice()
        self.layout = accumulator_layout
        self.prefault = accumulator_prefault
        import fcntl  # Ubuntu/macOS; reject unsupported platforms explicitly.
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / "writer.lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise RuntimeError(f"live inference cache already has a writer: {self.root}")
        self.geometry = geometry
        self.model_adapter, self.output_adapter = model_adapter, output_adapter
        self.products = tuple(model_adapter.output_products)
        if not self.products or any(p.chunk_size != geometry.chunk or p.inference_scaledown != geometry.scale for p in self.products):
            self.lock.close()
            raise ValueError("demand products must share the configured storage lattice")
        # Clean legacy named scratch only after writer ownership is established.
        # New slot pools use anonymous/unlinked backing where supported, but old
        # killed sessions must still be reclaimed explicitly.
        cleanup_stale_live_tmp(
            Path(channel.relative_path) for product in self.products
            for channel in product.channels)
        self.dtype = np.dtype(accumulator_dtype)
        if self.dtype not in (np.dtype("float16"), np.dtype("float32")):
            self.lock.close()
            raise ValueError("accumulator dtype must match sequential float16/float32")
        self.scratch_dirs, self.accumulators = [], {}
        self.arena_key = "sync"
        self._sync_arena = self._sync_publication = self._sync_record_spec = None
        self.tiles, self.chunks, self.priorities = {}, {}, {}
        self._pending_priorities = []
        self.publications = deque()
        self.stats = {"inferred_tiles": 0, "published_chunks": 0, "warm_chunks": 0}
        try:
            encoded = cache_identity(identity, geometry, self.dtype, self.layout, self.prefault)
            identity_path = self.root / "cache_identity.json"
            if identity_path.is_symlink():
                raise ValueError("unsafe cache identity symlink")
            if not identity_path.exists() and (self.root / "live.sqlite").exists():
                raise ValueError("legacy database-only inference cache; use a fresh cache root (old files untouched)")
            if identity_path.exists() and json.loads(identity_path.read_text()) != encoded:
                raise ValueError("cache inference identity mismatch")
            for product in self.products:
                for channel in product.channels:
                    Path(channel.relative_path).resolve().relative_to(self.root.resolve())
                    if not identity_path.exists():
                        level = Path(channel.relative_path) / str(product.level)
                        # Never bless unidentified payloads with a new model's
                        # provenance. Inspect this level only, not the volume.
                        if level.is_symlink():
                            raise ValueError("unsafe unidentified output level")
                        if level.exists():
                            with os.scandir(level) as entries:
                                if any(e.name not in {".zarray", ".zattrs"} for e in entries):
                                    raise ValueError("unidentified cache payloads; use a fresh cache root")
            # Legacy full-volume `.live_tmp/session-*` scratch is deliberately
            # ignored. It is disposable and explicit offline GC must handle it;
            # resume must never synchronously walk/free multi-terabyte extents.
            if not identity_path.exists():
                # Static provenance only: no tile/chunk/scratch state is saved.
                atomic_write(identity_path, encoded)
            if prepare_output is not None:
                # Metadata creation may replace an empty group directory, so
                # it must run under writer ownership BEFORE creating mmap files.
                prepare_output()
            self._channel_metadata = tuple((channel, product,
                tiled._omezarr_dim_sep(str(channel.relative_path), product.level))
                for product in self.products for channel in product.channels)
            self._read_arrays = {}
            self.weight = None
        except BaseException:
            self.close()
            raise

    def _mmap(self, path, dtype):
        descriptor = tiled._MmapBandDescriptor((str(path),), self.geometry.output_shape, str(dtype),
            self.geometry.chunk if self.layout == "chunk-major" else 0, self.prefault)
        storage_shape = descriptor.storage_shape
        axis0_slab_bytes = math.prod(storage_shape[1:]) * dtype.itemsize
        if axis0_slab_bytes > MAX_MMAP_FILE_BYTES:
            raise ValueError("one accumulator storage-axis-0 slab exceeds the mmap shard ceiling")
        axis0_per_shard = max(1, MMAP_SHARD_TARGET_BYTES // axis0_slab_bytes)
        sizes = tuple(min(axis0_per_shard, storage_shape[0] - start)
                      for start in range(0, storage_shape[0], axis0_per_shard))
        paths = tuple(path if len(sizes) == 1 else
                      path.with_name(f"{path.stem}.part-{index:04d}{path.suffix}")
                      for index in range(len(sizes)))
        # Fresh ftruncate creates zero-filled holes; opening/mapping is deferred
        # until a worker actually touches the corresponding Z shard.
        for shard_path, axis0_size in zip(paths, sizes):
            with shard_path.open("xb") as file:
                file.truncate(axis0_size * axis0_slab_bytes)
        return _MmapHandle(tuple(str(member) for member in paths), tuple(int(v) for v in sizes), str(dtype))

    def _chunk_files(self, origin):
        for channel, product, sep in self._channel_metadata:
            index = tuple(v // product.chunk_size for v in origin)
            yield Path(tiled._zarr_chunk_path(str(Path(channel.relative_path) / str(product.level)), sep, *index))

    def _cached_chunk_complete(self, origin, *, read_arrays=None):
        # Existing shared completeness checks require actual files, including
        # explicit zero chunks. A missing sibling must never become Zarr fill.
        if not all(self.output_adapter.product_chunk_complete(product, chunk_origin_zyx=origin)
                   for product in self.products):
            return False
        import zarr
        read_arrays = self._read_arrays if read_arrays is None else read_arrays
        region = tuple(slice(o, min(o + self.geometry.chunk, n)) for o, n in zip(origin, self.geometry.output_shape))
        try:
            for channel, product, _ in self._channel_metadata:
                path = str(Path(channel.relative_path) / str(product.level))
                if path not in read_arrays:
                    read_arrays[path] = zarr.open_array(path, mode="r")
                array = read_arrays[path]
                if (tuple(array.shape) != self.geometry.output_shape or array.dtype != np.dtype(product.dtype)
                        or tuple(array.chunks) != (self.geometry.chunk,) * 3):
                    return False
                # Decodes only the requested chunk (including its padded edge
                # storage). No array scan or persistent checksum database.
                values = np.asarray(array[region])
                if values.shape != tuple(s.stop - s.start for s in region):
                    return False
        except MemoryError:
            raise
        except Exception:
            return False
        return True

    def _sync_published(self, origin):
        """Confirm atomic chunk publication; cache files are regenerable.

        Closing each temp file before os.replace makes process-level readers
        safe.  Deliberate fsync durability is omitted: forcing every channel
        file and parent directory to stable storage serialized this transient
        inference cache and provided no useful recovery guarantee.
        """
        for member in self._chunk_files(origin):
            if not member.is_file():
                return False
        return True

    def ready(self, origin):
        entry = self.chunks.get(origin)
        if entry is not None:
            return entry["ready"]
        if self._cached_chunk_complete(origin):
            self.chunks[origin] = {"ready": True}
            self.stats["warm_chunks"] += 1
            return True
        return False

    def _register(self, origin):
        if self.ready(origin):
            return
        if origin not in self.chunks:
            dependencies = self.geometry.contributors(origin)
            self._register_missing(origin, dependencies)

    def _register_missing(self, origin, dependencies):
        self.chunks[origin] = {"ready": False, "dependencies": dependencies,
                               "applied": set(), "pending": set(), "requested": False,
                               "publishing": False}

    def request(self, index, *, remaining=0.0, background=False):
        origin = self.geometry.origin(index)
        self._register(origin)
        entry = self.chunks[origin]
        if not entry["ready"]:
            entry["requested"] = True
            priority = (0 if background else 1, float(remaining))
            for tile in entry["dependencies"]:
                if tile not in self.tiles:
                    previous = self.priorities.get(tile)
                    if previous is None or priority > previous:
                        self.priorities[tile] = priority
                        heapq.heappush(self._pending_priorities, (-priority[0], -priority[1], tile))
            self._queue_publication(origin)
        return origin

    def pending_tile_key(self):
        """Best unadmitted tile; lazy promotion entries retain exact priority ties."""
        while self._pending_priorities:
            foreground, remaining, tile = self._pending_priorities[0]
            if tile not in self.tiles and self.priorities.get(tile) == (-foreground, -remaining):
                return foreground, remaining, tile
            heapq.heappop(self._pending_priorities)
        return None

    def next_tile(self, *, profile_samples=None):
        with stage_timing(profile_samples, "admit.store_priority_selection"):
            key = self.pending_tile_key()
            if key is None:
                return None
            _, _, tile = heapq.heappop(self._pending_priorities)
        with stage_timing(profile_samples, "admit.store_state_update"):
            self.tiles[tile] = {"state": "inflight"}
        return tile

    def _descriptor(self, arrays):
        if not arrays or any(array.dtype != arrays[0].dtype or
                array.shard_axis0_sizes != arrays[0].shard_axis0_sizes for array in arrays):
            raise ValueError("incompatible mmap channel handles")
        return tiled._MmapBandDescriptor(
            tuple(path for array in arrays for path in array.paths),
            self.geometry.output_shape, arrays[0].dtype,
            self.geometry.chunk if self.layout == "chunk-major" else 0,
            self.prefault, arrays[0].shard_axis0_sizes)

    def begin_complete(self, tile, result_spec, slot, sequence, *, weight_spec=None):
        """Plan chunk-owner tasks without copying or persisting result tensors."""
        if self.tiles.get(tile, {}).get("state") != "inflight":
            raise ValueError("duplicate/unrequested inference completion")
        tasks = []
        for origin in self.geometry.touched(tile):
            self._register(origin)
            entry = self.chunks[origin]
            if entry["ready"]:
                continue
            if tile not in entry["dependencies"] or tile in entry["pending"] or tile in entry["applied"]:
                raise ValueError("invalid/duplicate contribution")
            destination, source = self.geometry.intersection(origin, tile)
            destination = tuple(slice(a - o, b - o) for (a, b), o in
                                zip(((s.start, s.stop) for s in destination), origin))
            tasks.append(tiled._AccumulateProcessTask(
                -1, sequence, slot, result_spec, self.geometry.scale, origin,
                tuple(v for s in destination for v in (s.start, s.stop)),
                tuple(v for s in source for v in (s.start, s.stop)),
                tuple(p.name for p in self.products),
                (), None, weight_spec, (self.arena_key, origin)))
            entry["pending"].add(tile)
        self.tiles[tile] = {"state": "accumulating", "remaining": len(tasks)}
        self.stats["inferred_tiles"] += 1
        if not tasks:
            self._tile_consumed(tile)
        return tasks

    def contribution_done(self, tile, origin):
        entry = self.chunks[origin]
        if tile not in entry["pending"]:
            raise ValueError("duplicate/unexpected accumulation acknowledgement")
        entry["pending"].remove(tile)
        entry["applied"].add(tile)
        self.tiles[tile]["remaining"] -= 1
        if not self.tiles[tile]["remaining"]:
            self._tile_consumed(tile)
        self._queue_publication(origin)

    def _tile_consumed(self, tile):
        self.tiles[tile] = {"state": "consumed"}

    def _queue_publication(self, origin):
        entry = self.chunks[origin]
        if (not entry["ready"] and entry["requested"] and not entry["publishing"]
                and not entry["pending"] and len(entry["applied"]) == len(entry["dependencies"])):
            entry["publishing"] = True
            self.publications.append(origin)

    def publication_done(self, origin, published):
        # Called only by coordinator AFTER the background writer has finished.
        if not published:
            raise RuntimeError("inference finalization did not publish all coherent channel chunks")
        entry = self.chunks[origin]
        if entry["ready"] or not entry["publishing"]:
            raise ValueError("duplicate/unexpected publication acknowledgement")
        entry["ready"] = True
        self.stats["published_chunks"] += 1

    def complete(self, tile, arrays):
        """Synchronous small-test adapter using the SAME shared worker executor.

        Production already owns shared result slots and uses begin_complete().
        """
        required = {"weight": (self.geometry.tile // self.geometry.scale,) * 3}
        required.update({p.name: (p.raw_channel_count, *required["weight"]) for p in self.products})
        if set(arrays) != set(required) or any(
                arrays[n].shape != shape or arrays[n].dtype != np.float32 for n, shape in required.items()):
            raise ValueError("invalid shared-worker output")
        entries = [(f"weight:{self.geometry.scale}", arrays["weight"])]
        entries += [(f"product:{p.name}", arrays[p.name]) for p in self.products]
        layouts, size = tiled._packed_layouts((name, a.shape, a.dtype) for name, a in entries)
        shm, = tiled._create_shared_slots(1, size)
        from lasagna.chunk_accumulator_arena import ChunkAccumulatorArena
        if self._sync_arena is None:
            record_entries = [("weight", (self.geometry.chunk,) * 3, np.float32)]
            record_entries += [(f"product:{p.name}:{channel}", (self.geometry.chunk,) * 3, self.dtype)
                               for p in self.products for channel in range(p.raw_channel_count)]
            record_layouts, record_bytes = tiled._packed_layouts(record_entries)
            self._sync_record_spec = tiled._AccumulatorRecordSpec(record_bytes, record_layouts)
            self._sync_publication = tiled._create_shared_slots(1, record_bytes)[0]
            self._sync_arena = ChunkAccumulatorArena(
                self.root / ".accumulator_arena", record_bytes=record_bytes,
                hot_slots=max(8, len(tuple(self.geometry.touched(tile)))))
        record_spec, publication = self._sync_record_spec, self._sync_publication
        record_bytes = record_spec.nbytes
        publication_spec = tiled._SharedSlotSpec(publication.name, record_bytes,
            (tiled._SharedArrayLayout("record", 0, (record_bytes,), np.dtype(np.uint8).str),))
        mmaps = tiled._BoundedMmapCache()
        arena = self._sync_arena
        try:
            for layout, (_, a) in zip(layouts, entries):
                tiled._slot_array(shm, layout)[:] = a
            spec = tiled._SharedSlotSpec(shm.name, size, layouts)
            for task in self.begin_complete(tile, spec, 0, 0):
                tiled._execute_accumulate_process_task(task, {shm.name: shm}, mmaps, arena, record_spec)
                self.contribution_done(tile, task.origin)
            while self.publications:
                origin = self.publications.popleft()
                destination = np.ndarray((record_bytes,), np.uint8, buffer=publication.buf)
                arena.snapshot_and_release((self.arena_key, origin), destination)
                self.publication_done(origin, self._publish_record(origin, publication_spec, record_spec))
        finally:
            mmaps.close()
            shm.close()
            shm.unlink()

    def _publish(self, origin):
        # Only incomplete, non-ready groups are submitted here. Clear all old
        # siblings first so a process crash cannot leave a mixed-generation group
        # that appears complete on restart. No completion journal/marker.
        members = tuple(self._chunk_files(origin))
        for member in members:
            if member.is_symlink():
                raise ValueError("unsafe published chunk symlink")
            try:
                member.resolve().relative_to(self.root.resolve())
            except ValueError as error:
                raise ValueError("unsafe published chunk path outside cache") from error
        for member in members:
            member.unlink(missing_ok=True)
        task = tiled._FlushProcessTask(0, 0, self.geometry.scale, origin,
            tuple(p.name for p in self.products), True, 0, self.geometry.chunk,
            (0, 0, 0, *self.geometry.output_shape), self.products,
            tuple((p.name, self._descriptor(self.accumulators[p.name])) for p in self.products), self._descriptor((self.weight,)))
        # These are threads in the store's process, not flush subprocesses.
        # Give each publication a small private shard-map cache: sharing mutable
        # mmap/LRU state between publication threads is unsafe, while bounding
        # it avoids mapping a whole sparse logical volume at once.
        # Publication is admitted only after this chunk's writers have finished;
        # the shared flush helper copies the requested region before finalizing.
        # Finalization consumes weight/raw regions sequentially, so one mapping
        # is sufficient and bounds aggregate virtual address use across the
        # publication thread pool.
        cache = tiled._BoundedMmapCache(max_open=1)
        try:
            tiled._execute_flush_process_task(task, self.model_adapter, self.output_adapter, cache)
            return self._sync_published(origin)
        finally:
            cache.close()

    def _publish_record(self, origin, publication_spec, record_spec):
        """Finalize one immutable chunk-local arena snapshot."""
        members = tuple(self._chunk_files(origin))
        for member in members:
            if member.is_symlink():
                raise ValueError("unsafe published chunk symlink")
            member.resolve().relative_to(self.root.resolve())
        for member in members:
            member.unlink(missing_ok=True)
        shm = tiled._attach_shared_memory(publication_spec.shm_name)
        try:
            record = np.ndarray((record_spec.nbytes,), np.uint8, buffer=shm.buf)
            layouts = {layout.name: layout for layout in record_spec.layouts}
            shape = tuple(min(self.geometry.chunk, n - o)
                          for n, o in zip(self.geometry.output_shape, origin))
            region = tuple(slice(0, n) for n in shape)
            denom = np.asarray(tiled._record_array(record, layouts["weight"])[region],
                               dtype=np.float32).copy()
            if np.any(denom > 0.0):
                np.maximum(denom, 1.0e-7, out=denom)
                for product in self.products:
                    raw = np.stack([np.asarray(tiled._record_array(
                        record, layouts[f"product:{product.name}:{channel}"])[region],
                        dtype=np.float32) for channel in range(product.raw_channel_count)])
                    raw /= denom[None]
                    finalizer = getattr(self.model_adapter, "finalize_product_slab", None)
                    if finalizer is None:
                        persisted = {channel.name: np.clip(raw[i] * 255.0, 0.0, 255.0).astype(product.dtype)
                                     for i, channel in enumerate(product.channels)}
                    else:
                        persisted = finalizer(product, raw)
                    self.output_adapter.write_product_chunk(
                        product, chunk_origin_zyx=origin,
                        data={channel.name: persisted[channel.name] for channel in product.channels})
            return self._sync_published(origin)
        finally:
            shm.close()

    def close(self):
        if getattr(self, "_sync_arena", None) is not None:
            self._sync_arena.close()
            self._sync_arena = None
        if getattr(self, "_sync_publication", None) is not None:
            self._sync_publication.close()
            self._sync_publication.unlink()
            self._sync_publication = None
        for directory in getattr(self, "scratch_dirs", ()):
            # Only this instance's mkdtemp-created scratch directory is removed.
            shutil.rmtree(directory)
        self.scratch_dirs = []
        self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def format_queue_snapshot(row):
    return (f"chunks_pending={row['requested_chunks_pending']} tiles_queued={row['tiles_queued']} "
            f"reads(q/run/done)={row['read_queued']}/{row['read_running']}/{row['read_done']} "
            f"gpu_submitted={row['gpu_submitted_uncollected']} "
            f"input_free={row['input_slots_free']}/{row['input_slots_capacity']} "
            f"accumulating={row['tiles_accumulating']} "
            f"publish(q/submitted)={row['publications_queued']}/{row['publications_submitted']} "
            f"arena(hot/spill)={row.get('arena_hot_occupancy', 0)}/"
            f"{row.get('arena_spill_occupancy', 0)} "
            f"spill(slots/segments)={row.get('arena_broker_slot_high_water', 0)}/"
            f"{row.get('arena_broker_segments', 0)}")


def _pump_refill_budget(worker_count, max_batch_size):
    """Bound coordinator latency while refilling two batches per GPU."""
    return min(64, max(16, max(1, int(worker_count)) * int(max_batch_size) * 2))


def _gated_demand_worker(gate, arguments, vram_budget):
    # No CUDA context/model before the manager records this child's identity.
    # Native CUDA/shared-memory faults otherwise surface only as an opaque
    # negative process exit code in the coordinator.  Preserve the Python
    # stack for fatal signals such as SIGBUS without changing worker behavior.
    faulthandler.enable(all_threads=True)
    gate.wait()
    if arguments[1].startswith("GPU-"):
        os.environ["CUDA_VISIBLE_DEVICES"] = arguments[1]
        arguments = (arguments[0], "cuda:0", *arguments[2:])
    if vram_budget is not None:
        os.environ["FIBER_INFERENCE_VRAM_BUDGET_BYTES"] = str(vram_budget)
    tiled._multi_gpu_worker_main(*arguments)


class _SpillArenaBroker:
    """Own fixed preallocated spill segments and recycle slots globally.

    Workers receive a slot and the path of its immutable backing segment.  The
    parent keeps every O_TMPFILE descriptor alive, while workers use O_DIRECT
    where supported.  This avoids dirty page-cache growth and reuses both file
    extents and record slots across accumulator lifetimes.
    """

    def __init__(self, context, scratch_dir, *, record_bytes, segment_slots, workers):
        self.record_bytes = int(record_bytes)
        self.segment_slots = int(segment_slots)
        granularity = int(mmap.ALLOCATIONGRANULARITY)
        raw_stride = self.record_bytes * self.segment_slots
        self.segment_stride = ((raw_stride + granularity - 1) // granularity) * granularity
        self.directory = Path(scratch_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.requests = context.Queue(maxsize=max(64, int(workers) * 4))
        self.responses = [context.Queue(maxsize=4) for _ in range(int(workers))]
        self._free = []
        self._next = 0
        self._segment_fds = []
        self._segment_paths = []
        self._named_paths = []
        self._closed = False
        self.stats = {"slot_high_water": 0, "logical_bytes": 0,
                      "allocated_bytes": 0, "segments": 0,
                      "reuses": 0, "requests": 0, "releases": 0,
                      "preallocation_fallbacks": 0}
        self._thread = threading.Thread(target=self._serve, name="spill-arena-broker", daemon=True)
        self._thread.start()

    def seal(self):
        """Compatibility no-op: segment paths must remain openable until close."""

    def _prepare_segment(self, segment):
        if segment != len(self._segment_fds):
            raise RuntimeError("spill segments must be prepared sequentially")
        fd = None
        named_path = None
        if hasattr(os, "O_TMPFILE"):
            try:
                fd = os.open(self.directory, os.O_RDWR | os.O_TMPFILE, 0o600)
                path = f"/proc/{os.getpid()}/fd/{fd}"
            except OSError:
                fd = None
        if fd is None:
            fd, name = tempfile.mkstemp(
                prefix=f"spill-segment-{segment:06d}-", suffix=".bin",
                dir=self.directory)
            path = name
            named_path = Path(name)
        try:
            os.ftruncate(fd, self.segment_stride)
            if hasattr(os, "posix_fallocate"):
                try:
                    os.posix_fallocate(fd, 0, self.segment_stride)
                    self.stats["allocated_bytes"] += self.segment_stride
                except OSError as exc:
                    if exc.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                        raise
                    self.stats["preallocation_fallbacks"] += 1
            else:  # pragma: no cover - macOS lacks posix_fallocate in Python
                self.stats["preallocation_fallbacks"] += 1
        except BaseException:
            os.close(fd)
            if named_path is not None:
                named_path.unlink(missing_ok=True)
            raise
        self._segment_fds.append(fd)
        self._segment_paths.append(path)
        if named_path is not None:
            self._named_paths.append(named_path)
        self.stats["segments"] = len(self._segment_fds)
        self.stats["logical_bytes"] = len(self._segment_fds) * self.segment_stride

    def _prepare_new_slot(self, slot):
        segment = slot // self.segment_slots
        while segment >= len(self._segment_fds):
            self._prepare_segment(len(self._segment_fds))
        return self._segment_paths[segment]

    def _serve(self):
        while True:
            item = self.requests.get()
            if item[0] == "stop":
                return
            kind, worker, value = item
            if kind == "release":
                heapq.heappush(self._free, int(value))
                self.stats["releases"] += 1
                continue
            if kind != "acquire":
                self.responses[worker].put(("error", f"invalid spill broker request {kind!r}"))
                continue
            self.stats["requests"] += 1
            try:
                if self._free:
                    slot = heapq.heappop(self._free)
                    self.stats["reuses"] += 1
                    path = self._segment_paths[slot // self.segment_slots]
                else:
                    slot = self._next
                    path = self._prepare_new_slot(slot)
                    self._next += 1
                    self.stats["slot_high_water"] = self._next
                self.responses[worker].put(("slot", slot, path))
            except BaseException as exc:
                self.responses[worker].put(("error", f"{type(exc).__name__}: {exc}"))

    def close(self):
        if self._closed:
            return
        self.requests.put(("stop", -1, -1))
        self._thread.join()
        for channel in (self.requests, *self.responses):
            channel.cancel_join_thread()
            channel.close()
        for fd in self._segment_fds:
            os.close(fd)
        self._segment_fds.clear()
        self._segment_paths.clear()
        for path in self._named_paths:
            path.unlink(missing_ok=True)
        self._named_paths.clear()
        self._closed = True


class DemandWorkerPool:
    """Shared persistent model workers, bounded descriptor slots and fast readers.

    Call pump() on the coordinator thread. Each reader is a callable taking a
    canonical padded tile origin and returning unnormalized uint8/uint16. No
    geometry or tensor payload passes through the native control transport.
    """
    def __init__(self, stores, readers, devices, *, slots_per_gpu=2, input_workers=8, input_slots=None,
                 result_slots=None, accumulator_workers=None, flush_workers=4, profile_pipeline=False,
                 max_batch_size=1, deferred_gpu=False, accumulator_hot_slots=None,
                 accumulator_spill_segment_slots=256):
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if not stores or not devices or slots_per_gpu < 1 or input_workers < 1:
            raise ValueError("nonempty stores/devices and positive worker budgets required")
        self.stores, self.readers = stores, readers
        first = next(iter(stores.values()))
        self.adapter, self.geometry = first.model_adapter, first.geometry
        native_stride = int(getattr(self.adapter, "native_output_stride", 1))
        if native_stride < 1 or self.geometry.scale % native_stride:
            raise ValueError("output scaledown must be divisible by the model native output stride")
        signature = lambda store: (store.geometry.tile, store.geometry.scale, store.geometry.border, store.geometry.overlap,
                                   tuple((p.name, p.raw_channel_count) for p in store.products))
        if any(signature(s) != signature(first) for s in stores.values()):
            raise ValueError("one worker pool requires common model output/tile geometry")
        for store in stores.values():
            adapter = store.model_adapter
            if (type(adapter) is not type(self.adapter)
                    or getattr(adapter, "checkpoint", None) != getattr(self.adapter, "checkpoint", None)
                    or getattr(adapter, "config", None) != getattr(self.adapter, "config", None)
                    or getattr(adapter, "native_output_stride", 1) != getattr(self.adapter, "native_output_stride", 1)):
                raise ValueError("one worker pool cannot mix different models/preprocessing")
        count = len(devices) * int(slots_per_gpu) if input_slots is None else input_slots
        if type(count) is not int or count < 1:
            raise ValueError("input_slots must be a positive integer")
        result_count = count if result_slots is None else int(result_slots)
        if result_count < 1:
            raise ValueError("result_slots must be positive")
        accumulator_workers = len(devices) if accumulator_workers is None else int(accumulator_workers)
        if accumulator_workers < 1 or flush_workers < 1:
            raise ValueError("positive accumulation/publication worker budgets required")
        for key, store in stores.items():
            store.arena_key = str(key)
        tile_shape = (self.geometry.tile,) * 3
        output_shape = (self.geometry.tile // self.geometry.scale,) * 3
        self.input_layouts, input_bytes = tiled._packed_layouts((("input", tile_shape, np.uint16),))
        # Result slots carry only model outputs.  Blend weights are immutable
        # for a geometry, so keep one shared template instead of copying a
        # 64 MiB full-resolution weight volume through every result slot.
        entries = [(f"product:{p.name}", (p.raw_channel_count, *output_shape), np.float32)
                   for p in first.products]
        self.result_layouts, result_bytes = tiled._packed_layouts(entries)
        self.weight_layouts, weight_bytes = tiled._packed_layouts(
            ((f"weight:{self.geometry.scale}", output_shape, np.float32),))
        record_entries = [("weight", (self.geometry.chunk,) * 3, np.float32)]
        record_entries += [(f"product:{p.name}:{channel}", (self.geometry.chunk,) * 3, first.dtype)
                           for p in first.products for channel in range(p.raw_channel_count)]
        record_layouts, record_bytes = tiled._packed_layouts(record_entries)
        self.record_spec = tiled._AccumulatorRecordSpec(record_bytes, record_layouts)
        hot_total = accumulator_workers * 8 if accumulator_hot_slots is None else int(accumulator_hot_slots)
        if hot_total < accumulator_workers or int(accumulator_spill_segment_slots) < 1:
            raise ValueError("hot slots must cover every accumulator worker and spill segments must be positive")
        hot_counts = tuple(hot_total // accumulator_workers + (i < hot_total % accumulator_workers)
                           for i in range(accumulator_workers))
        self.inputs, self.results, self.weight_templates = [], [], []
        self.result_slot_pool = None
        self.publication_slots, self.processes = [], []
        self.active, self.loading, self.free = {}, {}, deque(range(result_count))
        self.free_inputs = deque(range(count))
        self.input_owners = {}
        self.accumulating, self.accumulation_tasks, self.publishing = {}, {}, {}
        self.publication_pending = {}
        self.free_publication_slots = deque(range(int(flush_workers)))
        self.publication_generations = [0] * int(flush_workers)
        self.pending_accumulation = [deque() for _ in range(accumulator_workers)]
        self.accumulator_processes = []
        self.accumulator_counts = [0] * accumulator_workers
        self.accumulator_queues = []
        self.spill_broker = None
        self.task_sequence = 0
        self.flush_workers = int(flush_workers)
        self.sequence = 0
        self.profile_pipeline = profile_pipeline
        self.gpu_authorized = not deferred_gpu
        self.gpu_gate = None
        self._gpu_stopping = None
        self._closed = False
        self.last_gpu_activity = time.monotonic()
        self.stage_samples = [] if profile_pipeline else None
        self.input_workers = int(input_workers)
        self.profile_samples = []
        self.accumulator_samples = []
        self.accumulator_queue_samples = []
        self.accumulator_submitted = {}
        self.accumulator_backends = {}
        self.arena_worker_stats = {}
        self.worker_ids = set()
        self.stats = {"tiles": 0, "worker_seconds": 0.0, "load_seconds": 0.0,
                      "max_batch_size": max_batch_size, "batches": 0,
                      "batch_size_counts": {}, "batch_seconds": [], "gpu_memory_peaks": {},
                      "input_copy_seconds": 0.0, "pump_admit_seconds": 0.0,
                      "pump_dispatch_seconds": 0.0, "pump_events_seconds": 0.0,
                      "pump_background_seconds": 0.0,
                      "input_slots": count, "input_workers": input_workers, "result_slots": result_count,
                      "input_slot_bytes": input_bytes, "result_slot_bytes": result_bytes,
                      "result_slot_backing": "file-mmap",
                      "weight_template_bytes": weight_bytes,
                      "accumulator_record_bytes": record_bytes, "accumulator_hot_slots": hot_total,
                      "accumulator_hot_capacity_bytes": hot_total * record_bytes,
                      "shared_slot_bytes": count * input_bytes + result_count * result_bytes + weight_bytes
                          + int(flush_workers) * record_bytes}
        self.executor = ThreadPoolExecutor(max_workers=input_workers, thread_name_prefix="inference-input")
        self.publisher = ThreadPoolExecutor(max_workers=self.flush_workers, thread_name_prefix="inference-publish")
        context = mp.get_context("spawn")
        self.work = context.Queue(maxsize=count)
        self.events = context.Queue(maxsize=count * 2 + len(devices))
        self.accumulator_events = context.Queue(maxsize=128 * accumulator_workers)
        try:
            self.inputs = tiled._create_shared_slots(count, input_bytes)
            # Result tiles dominate POSIX-shm usage at full resolution. A
            # paused training process can retain most of the host's shmem and
            # make otherwise valid result writes fail with SIGBUS. Keep this
            # bounded ring in one sparse regular-file mmap under the cache;
            # pages remain cacheable and are overwritten without explicit sync.
            self.result_slot_pool = tiled._FileMappedSlotPool(
                first.root / ".accumulator_arena", count=result_count,
                size=result_bytes, prefix="result-slots-")
            self.results = self.result_slot_pool.handles()
            self.weight_templates = tiled._create_shared_slots(1, weight_bytes)
            self.publication_slots = tiled._create_shared_slots(int(flush_workers), record_bytes)
            self.spill_broker = _SpillArenaBroker(
                context, first.root / ".accumulator_arena", record_bytes=record_bytes,
                segment_slots=int(accumulator_spill_segment_slots), workers=accumulator_workers)
            input_specs = tuple(tiled._SharedSlotSpec(s.name, input_bytes, self.input_layouts) for s in self.inputs)
            result_specs = self.result_slot_pool.specs(self.result_layouts)
            self.result_specs = result_specs
            self.weight_spec = tiled._SharedSlotSpec(
                self.weight_templates[0].name, weight_bytes, self.weight_layouts)
            self.publication_specs = tuple(tiled._SharedSlotSpec(
                slot.name, record_bytes, (tiled._SharedArrayLayout(
                    "record", 0, (record_bytes,), np.dtype(np.uint8).str),))
                for slot in self.publication_slots)
            blend_ramp = tiled.blend_ramp(self.geometry.tile, self.geometry.border, self.geometry.overlap)
            weights = tiled._blend_weights_by_scale(
                blend_ramp, native_stride, (self.geometry.scale,), device=torch.device("cpu"),
            )[self.geometry.scale]
            tiled._copy_tensor_to_shared_array(
                tiled._slot_array(self.weight_templates[0], self.weight_layouts[0]), weights,
            )
            for i in range(accumulator_workers):
                work = context.Queue(maxsize=32)
                self.accumulator_queues.append(work)
                process = context.Process(target=tiled._accumulate_process_main,
                    args=(i, work, self.accumulator_events, self.record_spec, hot_counts[i],
                          int(accumulator_spill_segment_slots), str(first.root / ".accumulator_arena"),
                          self.spill_broker.requests, self.spill_broker.responses[i],
                          self.spill_broker.segment_stride),
                    name=f"demand-accumulate-{i}")
                process.start()
                self.accumulator_processes.append(process)
            attached = set()
            deadline = time.monotonic() + 60
            while len(attached) < accumulator_workers:
                for process in self.accumulator_processes:
                    if process.exitcode is not None:
                        raise RuntimeError(
                            f"spill arena worker {process.name} failed to attach: {process.exitcode}")
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    raise TimeoutError("timed out attaching accumulator workers to spill arena")
                try:
                    event = self.accumulator_events.get(timeout=min(1, timeout))
                except queue.Empty:
                    # Spawn and module import can legitimately take longer than
                    # one polling interval on a busy host.  The one-second get
                    # is only for checking child exit status; the attachment
                    # deadline above is the actual startup timeout.
                    continue
                if event[0] != "arena_attached" or event[1] in attached:
                    raise RuntimeError(f"invalid spill arena attachment event: {event}")
                attached.add(event[1])
            scales = {p.name: self.geometry.scale for p in first.products}
            self.gpu_arguments = (
                    self.adapter, None, self.geometry.tile,
                    blend_ramp, scales,
                    int(getattr(self.adapter, "native_output_stride", 1)), input_specs, result_specs,
                    self.work, self.events, profile_pipeline, max_batch_size, True)
            if not deferred_gpu:
                self.start_gpu(devices)
                self.authorize_gpu()
        except BaseException:
            self.close()
            raise

    def start_gpu(self, devices, *, vram_budget=None):
        if self.processes or self.active or self.input_owners:
            raise RuntimeError("GPU restart before submitted tiles drained")
        context = mp.get_context("spawn")
        self.gpu_gate = context.Event()
        self.gpu_authorized = False
        self.last_gpu_activity = time.monotonic()
        try:
            for i, device in enumerate(devices):
                process = context.Process(target=_gated_demand_worker, args=(self.gpu_gate, (
                    i, str(device), *self.gpu_arguments), vram_budget),
                    name=f"demand-gpu-{i}")
                process.start()
                self.processes.append(process)
        except BaseException:
            self.stop_gpu()
            raise

    def authorize_gpu(self):
        self.gpu_gate.set()
        self.gpu_authorized = True

    def stop_gpu(self):
        """Only at a drained boundary; shared CPU/result buffers remain alive."""
        while not self.poll_stop_gpu():
            time.sleep(.01)

    def poll_stop_gpu(self):
        """Nonblocking join state machine for manager/heartbeat event loops."""
        if self.active or self.input_owners:
            raise RuntimeError("cannot release GPUs with submitted tiles")
        self.gpu_authorized = False
        if self._gpu_stopping is None:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            self._gpu_stopping = (time.monotonic() + 10, False)
        for process in self.processes:
            process.join(timeout=0)
        if any(p.is_alive() for p in self.processes):
            deadline, killed = self._gpu_stopping
            if time.monotonic() >= deadline:
                if killed:
                    raise RuntimeError("GPU worker survived shutdown; retain reservation")
                for process in self.processes:
                    if process.is_alive():
                        process.kill()
                self._gpu_stopping = (time.monotonic() + 10, True)
            return False
        self.processes = []
        self.gpu_gate = None
        self._gpu_stopping = None
        # A terminated idle Queue reader may own its reader lock. Never reuse
        # those transports for a replacement worker generation.
        for channel in (self.work, self.events):
            channel.cancel_join_thread()
            channel.close()
        context = mp.get_context("spawn")
        self.work = context.Queue(maxsize=self.stats["input_slots"])
        self.events = context.Queue(maxsize=self.stats["input_slots"] * 2 + 8)
        self.gpu_arguments = (*self.gpu_arguments[:8], self.work, self.events,
                              *self.gpu_arguments[10:])
        return True

    @property
    def busy(self):
        return bool(self.active or self.loading or self.accumulating or self.publishing
                    or self.publication_pending or any(
            s.publications or s.pending_tile_key() is not None for s in self.stores.values()))

    def queue_snapshot(self):
        """Coordinator snapshot, not device utilization or native I/O request counts.

        Futures can finish while sampled. Each is classified exactly once;
        done reads still hold input slots until the coordinator dispatches them.
        GPU submitted includes queued, executing and uncollected completed work.
        """
        reads = {"read_queued": 0, "read_running": 0, "read_done": 0}
        for _, _, _, future, _ in self.loading.values():
            key = "read_done" if future.done() else "read_running" if future.running() else "read_queued"
            reads[key] += 1
        queued = [priority[0] > 0 for store in self.stores.values()
                  for tile, priority in store.priorities.items() if tile not in store.tiles]
        chunks = [entry for store in self.stores.values() for entry in store.chunks.values()
                  if not entry["ready"]]
        requested = sum(bool(entry.get("requested")) for entry in chunks)
        arena_totals = {key: sum(int(row.get(key, 0)) for row in getattr(self, "arena_worker_stats", {}).values())
                        for key in ("hot_occupancy", "hot_high_water", "spill_occupancy",
                                    "spill_high_water", "spill_capacity", "evictions",
                                    "promotions", "spill_bytes_written", "spill_bytes_read")}
        broker = getattr(self, "spill_broker", None)
        broker_stats = {} if broker is None else {
            f"arena_broker_{key}": int(value) for key, value in broker.stats.items()}
        return {**reads, "reads_submitted": len(self.loading),
                "tiles_queued": len(queued), "tiles_queued_foreground": sum(queued),
                "tiles_queued_background": len(queued)-sum(queued),
                "requested_chunks_pending": requested, "incidental_chunks_pending": len(chunks)-requested,
                "gpu_submitted_uncollected": len(self.active), "input_slots_owned": len(self.input_owners),
                "input_slots_free": len(self.free_inputs), "input_slots_capacity": self.stats["input_slots"],
                "result_slots_free": len(self.free), "result_slots_capacity": self.stats["result_slots"],
                "tiles_accumulating": len(self.accumulating),
                "accumulation_tasks_outstanding": len(self.accumulation_tasks),
                "accumulation_tasks_not_submitted": sum(map(len, self.pending_accumulation)),
                "publications_submitted": len(self.publishing),
                "publications_queued": sum(len(s.publications) for s in self.stores.values()),
                "publication_snapshots_pending": len(getattr(self, "publication_pending", {})),
                **{f"arena_{key}": value for key, value in arena_totals.items()}, **broker_stats}

    def _release_result(self, sequence):
        _, _, slot, _ = self.accumulating.pop(sequence)
        self.free.append(slot)
        self.stats["tiles"] += 1

    def _pump_accumulators(self):
        while True:
            try:
                event = self.accumulator_events.get_nowait()
            except queue.Empty:
                break
            if event[0] == "publish_snapshot":
                name, origin, slot, generation = self.publication_pending.pop(event[1])
                if slot != event[2] or generation != event[3] or self.publication_generations[slot] != generation:
                    raise RuntimeError("publication snapshot slot mismatch")
                future = self.publisher.submit(
                    self.stores[name]._publish_record, origin,
                    self.publication_specs[slot], self.record_spec)
                self.publishing[name, origin] = (future, slot)
                self.arena_worker_stats[event[4]] = event[7]
                continue
            if event[0] == "batch_error":
                raise RuntimeError(f"shared accumulator batch error: {event}")
            if event[0] != "batch_result":
                raise RuntimeError(f"shared accumulator error: {event}")
            name, tile, origins, sequence = self.accumulation_tasks.pop(event[1])
            if event[2] != sequence:
                raise RuntimeError("accumulator result sequence mismatch")
            if event[6] != len(origins):
                raise RuntimeError("accumulator batch contribution count mismatch")
            for origin in origins:
                self.stores[name].contribution_done(tile, origin)
            self.accumulator_counts[event[3]] += len(origins)
            if len(event) > 7:
                self.arena_worker_stats[event[3]] = event[7]
            self.accumulating[sequence][3] -= len(origins)
            if not self.accumulating[sequence][3]:
                self._release_result(sequence)
            if self.profile_pipeline:
                self.accumulator_samples.append(event[5])
                self.accumulator_queue_samples.append(max(0, time.perf_counter() - self.accumulator_submitted.pop(event[1]) - event[5]))
                self.accumulator_backends[event[4]] = self.accumulator_backends.get(event[4], 0) + len(origins)
        for pending, work in zip(self.pending_accumulation, self.accumulator_queues):
            while pending:
                try:
                    work.put_nowait(pending[0])
                except queue.Full:
                    break
                if self.profile_pipeline and hasattr(pending[0], "batch_id"):
                    self.accumulator_submitted[pending[0].batch_id] = time.perf_counter()
                pending.popleft()

    def _pump_publications(self):
        for key, (future, slot) in list(self.publishing.items()):
            if future.done():
                name, origin = key
                self.stores[name].publication_done(origin, future.result())
                del self.publishing[key]
                self.free_publication_slots.append(slot)
        for name, store in self.stores.items():
            while store.publications and self.free_publication_slots:
                origin = store.publications.popleft()
                slot = self.free_publication_slots.popleft()
                self.publication_generations[slot] += 1
                generation = self.publication_generations[slot]
                request_id = self.task_sequence
                self.task_sequence += 1
                chunk = store.geometry.chunk
                grid = tuple(o // chunk for o in origin)
                owner = tiled._stable_accumulator_owner(store.geometry.scale, grid,
                                                         len(self.accumulator_queues))
                task = tiled._ArenaPublishTask(
                    request_id, (store.arena_key, origin), slot, generation,
                    self.publication_specs[slot])
                self.publication_pending[request_id] = (name, origin, slot, generation)
                self.pending_accumulation[owner].append(task)

    def _read_into_slot(self, name, tile, slot):
        # The input slot was reserved by the coordinator and remains owned
        # until H2D completes. Populate it in the reader thread, not in the
        # control loop: many large copies/page faults must not stop GPU dispatch
        # or accumulation acknowledgements. Return metadata, not a second array.
        raw = np.asarray(self.readers[name](tile))
        if raw.shape != self.input_layouts[0].shape or raw.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise ValueError("inference input reader returned invalid raw tile")
        started = time.perf_counter()
        layout = replace(self.input_layouts[0], dtype=raw.dtype.str)
        with stage_timing(self.stage_samples, "reader.shared_slot_copy"):
            tiled._slot_array(self.inputs[slot], layout)[:] = raw
        return raw.dtype.str, time.perf_counter() - started

    def _next_admission(self):
        choices = []
        for name, store in self.stores.items():
            key = store.pending_tile_key()
            if key is not None:
                choices.append((key[0], key[1], name, key[2]))
        return min(choices, default=None)

    def pump(self):
        for process in ([] if self._gpu_stopping is not None else self.processes) + self.accumulator_processes:
            if process.exitcode is not None:
                raise RuntimeError(f"inference/accumulator worker {process.name} exited unexpectedly: {process.exitcode}")
        phase_started = time.perf_counter()
        self._pump_accumulators()
        self._pump_publications()
        self.stats["pump_background_seconds"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        # Bound coordinator work: a deep input window must not prevent RPC
        # priority updates or completed GPU/accumulation events being drained.
        # Refill at least two dynamic batches per GPU in one coordinator pass.
        # The old fixed 16/8 limits were adequate for batch-1 but visibly
        # starved eight batch-4 workers after a completion wave.
        pump_budget = _pump_refill_budget(len(self.processes), self.stats["max_batch_size"])
        admitted = 0
        # Keep the executor's admitted window at most reader-sized. The deeper
        # priority queue stays in stores, where new foreground demands can
        # overtake background work not yet submitted to a non-preemptible read.
        while self.free_inputs and admitted < pump_budget and len(self.loading) < self.input_workers:
            with stage_timing(self.stage_samples, "admit.global_priority_selection"):
                choice = self._next_admission()
                if choice is None:
                    break
                _, _, name, _ = choice
            tile = self.stores[name].next_tile(profile_samples=self.stage_samples)
            with stage_timing(self.stage_samples, "admit.reserve_input_slot"):
                slot, sequence = self.free_inputs.popleft(), self.sequence
                self.sequence += 1
                started = time.perf_counter()
            with stage_timing(self.stage_samples, "admit.executor_submit") as details:
                if details is not None:
                    details["executor_threads_before"] = len(getattr(self.executor, "_threads", ()))
                future = self.executor.submit(self._read_into_slot, name, tile, slot)
                if details is not None:
                    details["executor_threads_after"] = len(getattr(self.executor, "_threads", ()))
            with stage_timing(self.stage_samples, "admit.register_future"):
                self.loading[sequence] = (name, tile, slot, future, started)
            admitted += 1
        self.stats["pump_admit_seconds"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        dispatched = 0
        # CPU failures must surface even while GPU dispatch is paused.
        for _, _, _, future, _ in self.loading.values():
            if future.done():
                error = future.exception()
                if error is not None:
                    raise error
        for sequence, (name, tile, slot, future, started) in list(self.loading.items()):
            if not self.gpu_authorized or dispatched >= pump_budget:
                break
            if not future.done() or not self.free:
                continue
            raw_dtype, copy_seconds = future.result()
            self.stats["input_copy_seconds"] += copy_seconds
            names = tuple(p.name for p in self.stores[name].products)
            result_slot = self.free.popleft()
            self.work.put_nowait((sequence, tile, slot, result_slot, names, raw_dtype))
            self.input_owners[sequence] = slot
            self.active[sequence] = (name, tile, result_slot)
            del self.loading[sequence]
            self.stats["load_seconds"] += time.perf_counter() - started
            dispatched += 1
            self.last_gpu_activity = time.monotonic()
        self.stats["pump_dispatch_seconds"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event[0] == "input_released":
                if self.input_owners.pop(event[1]) != event[2]:
                    raise RuntimeError("input slot ownership mismatch")
                self.free_inputs.append(event[2])
                continue
            if event[0] == "error":
                raise RuntimeError(f"shared inference worker error: {event[4:]}")
            if event[0] == "batch":
                worker, sample = str(event[1]), event[2]
                size = str(sample["size"])
                self.stats["batches"] += 1
                counts = self.stats["batch_size_counts"]
                counts[size] = counts.get(size, 0) + 1
                self.stats["batch_seconds"].append(sample["seconds"])
                peaks = self.stats["gpu_memory_peaks"].setdefault(worker, {})
                for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                    peaks[key] = max(peaks.get(key, 0), sample[key])
                continue
            if event[0] != "result":
                raise RuntimeError("unknown shared worker event")
            self.worker_ids.add(event[5])
            if len(event) == 9:
                self.profile_samples.append({**event[7], "worker_index": event[5],
                    "coordinator_receive_delay_s": time.perf_counter() - event[8]})
            name, tile, slot = self.active.pop(event[1])
            self.last_gpu_activity = time.monotonic()
            tasks = self.stores[name].begin_complete(
                tile, self.result_specs[slot], slot, event[1], weight_spec=self.weight_spec,
            )
            self.accumulating[event[1]] = [name, tile, slot, len(tasks)]
            owner_tasks = [[] for _ in self.accumulator_queues]
            for task in tasks:
                # Hash chunk indices, not voxel origins: origins divisible by
                # chunk32 would otherwise all hash to the SAME worker modulo8.
                chunk = self.stores[name].geometry.chunk
                grid = tuple(o // chunk for o in task.origin)
                owner = tiled._stable_accumulator_owner(task.sd, grid, len(self.accumulator_queues))
                owner_tasks[owner].append(task)
            for owner, owned in enumerate(owner_tasks):
                if not owned:
                    continue
                batch_id = self.task_sequence
                self.task_sequence += 1
                owned = tuple(replace(task, task_id=batch_id) for task in owned)
                batch = tiled._AccumulateProcessBatch(batch_id, event[1], owned)
                self.accumulation_tasks[batch_id] = (
                    name, tile, tuple(task.origin for task in owned), event[1])
                self.pending_accumulation[owner].append(batch)
            if not tasks:
                self._release_result(event[1])
            self.stats["worker_seconds"] += event[6]
        self.stats["pump_events_seconds"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        self._pump_accumulators()
        self._pump_publications()
        self.stats["pump_background_seconds"] += time.perf_counter() - phase_started

    def close(self):
        if self._closed:
            return
        # Cancel work on failure/exit; never block on a full data queue to send
        # a shutdown sentinel. No raw output from a terminated worker is ready.
        for process in self.processes + self.accumulator_processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes + self.accumulator_processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        self.processes = []
        self.accumulator_processes = []
        if self.spill_broker is not None:
            self.stats["spill_broker"] = dict(self.spill_broker.stats)
            self.spill_broker.close()
            self.spill_broker = None
        # No publisher may retain a view of scratch after the store is closed.
        self.publisher.shutdown(wait=True, cancel_futures=True)
        self.executor.shutdown(wait=True, cancel_futures=True)
        for array in (self.inputs + self.results + self.weight_templates
                      + self.publication_slots):
            array.close()
            unlink = getattr(array, "unlink", None)
            if unlink is not None:
                unlink()
        if self.result_slot_pool is not None:
            self.result_slot_pool.close()
            self.result_slot_pool = None
        self.inputs, self.results, self.weight_templates = [], [], []
        for channel in (self.work, self.events, self.accumulator_events, *self.accumulator_queues):
            channel.cancel_join_thread()
            channel.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
