"""Demand scheduling around shared GPU, accumulation and Zarr finalization code.

Coordinator owns metadata; chunk-owner workers add shared-RAM results in arrival
order. Only finalized Zarr chunks survive a session; scratch is not durable truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import replace
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import itertools
import heapq
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import multiprocessing as mp
import queue
import time

import numpy as np

from lasagna import tiled_predict3d as tiled
from lasagna.inference_provenance import atomic_write
from lasagna.inference_profiling import stage_timing


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
        self.dtype = np.dtype(accumulator_dtype)
        if self.dtype not in (np.dtype("float16"), np.dtype("float32")):
            self.lock.close()
            raise ValueError("accumulator dtype must match sequential float16/float32")
        self.scratch_dirs, self.arrays, self.accumulators = [], [], {}
        self.tiles, self.chunks, self.priorities = {}, {}, {}
        self._pending_priorities = []
        self.publications = deque()
        self.stats = {"inferred_tiles": 0, "published_chunks": 0, "warm_chunks": 0}
        try:
            encoded = json.loads(json.dumps({"schema": 2, "identity": identity, "geometry": asdict(geometry),
                                  "accumulator_dtype": str(self.dtype), "accumulation_order": "arrival",
                                  "accumulator_layout": self.layout, "accumulator_prefault": self.prefault},
                                 sort_keys=True, allow_nan=False))
            identity_path = self.root / "cache_identity.json"
            if identity_path.is_symlink():
                raise ValueError("unsafe cache identity symlink")
            if not identity_path.exists() and (self.root / "live.sqlite").exists():
                raise ValueError("legacy database-only inference cache; use a fresh cache root (old files untouched)")
            if identity_path.exists() and json.loads(identity_path.read_text()) != encoded:
                raise ValueError("cache inference identity mismatch")
            allowed = set()
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
                parent = Path(product.channels[0].relative_path) / ".live_tmp"
                if parent.is_symlink():
                    raise ValueError("unsafe scratch parent symlink")
                allowed.add(parent)
            stale_dirs = []
            # Validate every candidate before deleting any. Only reserved,
            # immediately contained session directories are scratch, not outputs.
            for parent in allowed:
                if not parent.exists():
                    continue
                for stale in parent.iterdir():
                    if not stale.name.startswith("session-"):
                        continue
                    if stale.is_symlink() or not stale.is_dir():
                        raise ValueError("unsafe stale scratch directory")
                    for member in stale.iterdir():
                        raw = member.name.removeprefix("raw-").removesuffix(".mmap")
                        known = member.name == "weight.mmap" or (member.name == f"raw-{raw}.mmap" and raw.isdigit())
                        if member.is_symlink() or not member.is_file() or not known:
                            raise ValueError("unsafe stale scratch contents")
                    stale_dirs.append(stale)
            for stale in stale_dirs:
                shutil.rmtree(stale)
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
            for product in self.products:
                parent = Path(product.channels[0].relative_path) / ".live_tmp"
                parent.mkdir(parents=True, exist_ok=True)
                directory = Path(tempfile.mkdtemp(prefix="session-", dir=parent))
                self.scratch_dirs.append(directory)
                self.accumulators[product.name] = tuple(self._mmap(directory / f"raw-{c}.mmap", self.dtype)
                                                        for c in range(product.raw_channel_count))
            self.weight = self._mmap(self.scratch_dirs[0] / "weight.mmap", np.dtype("float32"))
        except BaseException:
            self.close()
            raise

    def _mmap(self, path, dtype):
        descriptor = tiled._MmapBandDescriptor((str(path),), self.geometry.output_shape, str(dtype),
            self.geometry.chunk if self.layout == "chunk-major" else 0, self.prefault)
        size = math.prod(descriptor.storage_shape) * dtype.itemsize
        # Fresh ftruncate allocates zero-filled holes, not a whole-volume clear.
        with path.open("xb") as file:
            file.truncate(size)
        array = np.memmap(path, mode="r+", dtype=dtype, shape=descriptor.storage_shape)
        self.arrays.append(array)
        return array

    def _chunk_files(self, origin):
        for channel, product, sep in self._channel_metadata:
            index = tuple(v // product.chunk_size for v in origin)
            yield Path(tiled._zarr_chunk_path(str(Path(channel.relative_path) / str(product.level)), sep, *index))

    def _cached_chunk_complete(self, origin):
        # Existing shared completeness checks require actual files, including
        # explicit zero chunks. A missing sibling must never become Zarr fill.
        if not all(self.output_adapter.product_chunk_complete(product, chunk_origin_zyx=origin)
                   for product in self.products):
            return False
        import zarr
        region = tuple(slice(o, min(o + self.geometry.chunk, n)) for o, n in zip(origin, self.geometry.output_shape))
        try:
            for channel, product, _ in self._channel_metadata:
                path = str(Path(channel.relative_path) / str(product.level))
                if path not in self._read_arrays:
                    self._read_arrays[path] = zarr.open_array(path, mode="r")
                array = self._read_arrays[path]
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
        for member in self._chunk_files(origin):
            if not member.is_file():
                return False
            with member.open("rb") as file:
                os.fsync(file.fileno())
            descriptor = os.open(member.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
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
        return tiled._MmapBandDescriptor(tuple(str(a.filename) for a in arrays),
                                         self.geometry.output_shape, str(arrays[0].dtype),
                                         self.geometry.chunk if self.layout == "chunk-major" else 0, self.prefault)

    def begin_complete(self, tile, result_spec, slot, sequence):
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
            tasks.append(tiled._AccumulateProcessTask(
                -1, sequence, slot, result_spec, self.geometry.scale, origin,
                tuple(v for s in destination for v in (s.start, s.stop)),
                tuple(v for s in source for v in (s.start, s.stop)),
                tuple(p.name for p in self.products),
                tuple((p.name, self._descriptor(self.accumulators[p.name])) for p in self.products),
                self._descriptor((self.weight,))))
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
        mmaps = {}
        try:
            for layout, (_, a) in zip(layouts, entries):
                tiled._slot_array(shm, layout)[:] = a
            spec = tiled._SharedSlotSpec(shm.name, size, layouts)
            for task in self.begin_complete(tile, spec, 0, 0):
                tiled._execute_accumulate_process_task(task, {shm.name: shm}, mmaps)
                self.contribution_done(tile, task.origin)
            while self.publications:
                origin = self.publications.popleft()
                self.publication_done(origin, self._publish(origin))
        finally:
            for a in mmaps.values():
                a._mmap.close()
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
        cache = {}
        try:
            tiled._execute_flush_process_task(task, self.model_adapter, self.output_adapter, cache)
        finally:
            for value in cache.values():
                value._mmap.close()
        return self._sync_published(origin)

    def close(self):
        for array in getattr(self, "arrays", ()):
            array._mmap.close()
        self.arrays = []
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
            f"publish(q/submitted)={row['publications_queued']}/{row['publications_submitted']}")


class DemandWorkerPool:
    """Shared persistent model workers, bounded descriptor slots and fast readers.

    Call pump() on the coordinator thread. Each reader is a callable taking a
    canonical padded tile origin and returning unnormalized uint8/uint16. No
    geometry or tensor payload passes through the native control transport.
    """
    def __init__(self, stores, readers, devices, *, slots_per_gpu=2, input_workers=8, input_slots=None,
                 result_slots=None, accumulator_workers=None, flush_workers=4, profile_pipeline=False,
                 max_batch_size=1):
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
        tile_shape = (self.geometry.tile,) * 3
        output_shape = (self.geometry.tile // self.geometry.scale,) * 3
        self.input_layouts, input_bytes = tiled._packed_layouts((("input", tile_shape, np.uint16),))
        entries = [(f"weight:{self.geometry.scale}", output_shape, np.float32)]
        entries += [(f"product:{p.name}", (p.raw_channel_count, *output_shape), np.float32) for p in first.products]
        self.result_layouts, result_bytes = tiled._packed_layouts(entries)
        self.inputs, self.results, self.processes = [], [], []
        self.active, self.loading, self.free = {}, {}, deque(range(result_count))
        self.free_inputs = deque(range(count))
        self.input_owners = {}
        self.accumulating, self.accumulation_tasks, self.publishing = {}, {}, {}
        self.pending_accumulation = [deque() for _ in range(accumulator_workers)]
        self.accumulator_processes = []
        self.accumulator_counts = [0] * accumulator_workers
        self.accumulator_queues = []
        self.task_sequence = 0
        self.flush_workers = int(flush_workers)
        self.sequence = 0
        self.profile_pipeline = profile_pipeline
        self.stage_samples = [] if profile_pipeline else None
        self.input_workers = int(input_workers)
        self.profile_samples = []
        self.accumulator_samples = []
        self.accumulator_queue_samples = []
        self.accumulator_submitted = {}
        self.accumulator_backends = {}
        self.worker_ids = set()
        self.stats = {"tiles": 0, "worker_seconds": 0.0, "load_seconds": 0.0,
                      "max_batch_size": max_batch_size, "batches": 0,
                      "batch_size_counts": {}, "batch_seconds": [], "gpu_memory_peaks": {},
                      "input_copy_seconds": 0.0, "pump_admit_seconds": 0.0,
                      "pump_dispatch_seconds": 0.0, "pump_events_seconds": 0.0,
                      "pump_background_seconds": 0.0,
                      "input_slots": count, "input_workers": input_workers, "result_slots": result_count,
                      "input_slot_bytes": input_bytes, "result_slot_bytes": result_bytes,
                      "shared_slot_bytes": count * input_bytes + result_count * result_bytes}
        self.executor = ThreadPoolExecutor(max_workers=input_workers, thread_name_prefix="inference-input")
        self.publisher = ThreadPoolExecutor(max_workers=self.flush_workers, thread_name_prefix="inference-publish")
        context = mp.get_context("spawn")
        self.work = context.Queue(maxsize=count)
        self.events = context.Queue(maxsize=count * 2 + len(devices))
        self.accumulator_events = context.Queue(maxsize=128 * accumulator_workers)
        try:
            self.inputs = tiled._create_shared_slots(count, input_bytes)
            self.results = tiled._create_shared_slots(result_count, result_bytes)
            input_specs = tuple(tiled._SharedSlotSpec(s.name, input_bytes, self.input_layouts) for s in self.inputs)
            result_specs = tuple(tiled._SharedSlotSpec(s.name, result_bytes, self.result_layouts) for s in self.results)
            self.result_specs = result_specs
            for i in range(accumulator_workers):
                work = context.Queue(maxsize=32)
                self.accumulator_queues.append(work)
                process = context.Process(target=tiled._accumulate_process_main,
                    args=(i, work, self.accumulator_events), name=f"demand-accumulate-{i}")
                process.start()
                self.accumulator_processes.append(process)
            scales = {p.name: self.geometry.scale for p in first.products}
            for i, device in enumerate(devices):
                process = context.Process(target=tiled._multi_gpu_worker_main, args=(
                    i, str(device), self.adapter, None, self.geometry.tile,
                    tiled.blend_ramp(self.geometry.tile, self.geometry.border, self.geometry.overlap), scales,
                    int(getattr(self.adapter, "native_output_stride", 1)), input_specs, result_specs,
                    self.work, self.events, profile_pipeline, max_batch_size, True))
                process.start()
                self.processes.append(process)
        except BaseException:
            self.close()
            raise

    @property
    def busy(self):
        return bool(self.active or self.loading or self.accumulating or self.publishing or any(
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
                "publications_queued": sum(len(s.publications) for s in self.stores.values())}

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
            if event[0] != "result":
                raise RuntimeError(f"shared accumulator error: {event}")
            name, tile, origin, sequence = self.accumulation_tasks.pop(event[1])
            if event[2] != sequence:
                raise RuntimeError("accumulator result sequence mismatch")
            self.stores[name].contribution_done(tile, origin)
            self.accumulator_counts[event[3]] += 1
            self.accumulating[sequence][3] -= 1
            if not self.accumulating[sequence][3]:
                self._release_result(sequence)
            if self.profile_pipeline:
                self.accumulator_samples.append(event[5])
                self.accumulator_queue_samples.append(max(0, time.perf_counter() - self.accumulator_submitted.pop(event[1]) - event[5]))
                self.accumulator_backends[event[4]] = self.accumulator_backends.get(event[4], 0) + 1
        for pending, work in zip(self.pending_accumulation, self.accumulator_queues):
            while pending:
                try:
                    work.put_nowait(pending[0])
                except queue.Full:
                    break
                if self.profile_pipeline:
                    self.accumulator_submitted[pending[0].task_id] = time.perf_counter()
                pending.popleft()

    def _pump_publications(self):
        for key, future in list(self.publishing.items()):
            if future.done():
                name, origin = key
                self.stores[name].publication_done(origin, future.result())
                del self.publishing[key]
        for name, store in self.stores.items():
            while store.publications and len(self.publishing) < self.flush_workers:
                origin = store.publications.popleft()
                self.publishing[name, origin] = self.publisher.submit(store._publish, origin)

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
        for process in self.processes + self.accumulator_processes:
            if process.exitcode is not None:
                raise RuntimeError(f"inference/accumulator worker {process.name} exited unexpectedly: {process.exitcode}")
        phase_started = time.perf_counter()
        self._pump_accumulators()
        self._pump_publications()
        self.stats["pump_background_seconds"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        # Bound coordinator work: a deep input window must not prevent RPC
        # priority updates or completed GPU/accumulation events being drained.
        admitted = 0
        # Keep the executor's admitted window at most reader-sized. The deeper
        # priority queue stays in stores, where new foreground demands can
        # overtake background work not yet submitted to a non-preemptible read.
        while self.free_inputs and admitted < 16 and len(self.loading) < self.input_workers:
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
        for sequence, (name, tile, slot, future, started) in list(self.loading.items()):
            if dispatched >= 8:
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
            tasks = self.stores[name].begin_complete(tile, self.result_specs[slot], slot, event[1])
            self.accumulating[event[1]] = [name, tile, slot, len(tasks)]
            for task in tasks:
                task = replace(task, task_id=self.task_sequence)
                self.task_sequence += 1
                self.accumulation_tasks[task.task_id] = (name, tile, task.origin, event[1])
                # Hash chunk indices, not voxel origins: origins divisible by
                # chunk32 would otherwise all hash to the SAME worker modulo8.
                chunk = self.stores[name].geometry.chunk
                grid = tuple(o // chunk for o in task.origin)
                owner = tiled._stable_accumulator_owner(task.sd, grid, len(self.accumulator_queues))
                self.pending_accumulation[owner].append(task)
            if not tasks:
                self._release_result(event[1])
            self.stats["worker_seconds"] += event[6]
        self.stats["pump_events_seconds"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        self._pump_accumulators()
        self._pump_publications()
        self.stats["pump_background_seconds"] += time.perf_counter() - phase_started

    def close(self):
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
        # No publisher may retain a view of scratch after the store is closed.
        self.publisher.shutdown(wait=True, cancel_futures=True)
        self.executor.shutdown(wait=True, cancel_futures=True)
        for array in self.inputs + self.results:
            array.close()
            array.unlink()
        self.inputs, self.results = [], []
        for channel in (self.work, self.events, self.accumulator_events, *self.accumulator_queues):
            channel.cancel_join_thread()
            channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
