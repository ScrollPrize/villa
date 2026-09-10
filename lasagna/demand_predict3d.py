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
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import multiprocessing as mp
import queue
import time

import numpy as np

from lasagna import tiled_predict3d as tiled
from lasagna.inference_provenance import sha256_file


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


def _key(coords):
    return ",".join(map(str, coords))


class DemandStore:
    """Sparse ledger plus full-logical-shape mmap accumulation for one source.

    Each tile contributes once to every touched (including unrequested) chunk.
    Partial neighbors never schedule work. Only the coordinator changes metadata.
    """
    def __init__(self, root, identity, geometry, model_adapter, output_adapter,
                 *, accumulator_dtype="float16", accumulator_layout="flat", accumulator_prefault=False,
                 scratch_budget_bytes=64 << 30, prepare_output=None):
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
        self.budget = int(scratch_budget_bytes)
        if self.budget <= 0:
            self.lock.close()
            raise ValueError("scratch budget must be positive")
        self.scratch_dirs, self.arrays, self.accumulators = [], [], {}
        self.tiles, self.chunks, self.priorities = {}, {}, {}
        self.scratch_bytes = 0
        self.publications = deque()
        self.stats = {"inferred_tiles": 0, "published_chunks": 0, "warm_chunks": 0}
        try:
            self.db = sqlite3.connect(self.root / "live.sqlite")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (key TEXT PRIMARY KEY, state TEXT NOT NULL, fingerprints TEXT);
                CREATE TABLE IF NOT EXISTS tiles (key TEXT PRIMARY KEY, state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS scratch (path TEXT PRIMARY KEY);
            """)
            encoded = json.dumps({"schema": 1, "identity": identity, "geometry": asdict(geometry),
                                  "accumulator_dtype": str(self.dtype), "accumulation_order": "arrival",
                                  "accumulator_layout": self.layout, "accumulator_prefault": self.prefault},
                                 sort_keys=True, separators=(",", ":"), allow_nan=False)
            existing = self.db.execute("SELECT value FROM identity WHERE id=1").fetchone()
            if existing and existing[0] != encoded:
                raise ValueError("cache inference identity mismatch")
            self.db.execute("INSERT OR IGNORE INTO identity VALUES (1,?)", (encoded,))
            allowed = set()
            for product in self.products:
                for channel in product.channels:
                    Path(channel.relative_path).resolve().relative_to(self.root.resolve())
                allowed.add((Path(product.channels[0].relative_path) / ".live_tmp").resolve())
            for (stored,) in self.db.execute("SELECT path FROM scratch").fetchall():
                stale = Path(stored)
                if stale.is_symlink() or stale.parent.resolve() not in allowed or not stale.name.startswith("session-"):
                    raise ValueError("unsafe stale scratch path in live ledger")
                if stale.exists():
                    shutil.rmtree(stale)
            self.db.execute("DELETE FROM scratch")
            # Scratch from a previous process is not resumable truth. Durable
            # output rows are validated lazily on request, without array scans.
            self.db.execute("DELETE FROM tiles")
            self.db.execute("DELETE FROM chunks WHERE state != 'ready'")
            self.db.commit()
            if prepare_output is not None:
                # Metadata creation may replace an empty group directory, so
                # it must run under writer ownership BEFORE creating mmap files.
                prepare_output()
            for product in self.products:
                parent = Path(product.channels[0].relative_path) / ".live_tmp"
                parent.mkdir(parents=True, exist_ok=True)
                directory = Path(tempfile.mkdtemp(prefix="session-", dir=parent))
                self.scratch_dirs.append(directory)
                self.db.execute("INSERT INTO scratch VALUES (?)", (str(directory.resolve()),))
                self.db.commit()
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

    def _fingerprints(self, origin, *, sync=False):
        files = []
        for product in self.products:
            for channel in product.channels:
                path = str(channel.relative_path)
                sep = tiled._omezarr_dim_sep(path, product.level)
                index = tuple(v // product.chunk_size for v in origin)
                member = Path(tiled._zarr_chunk_path(str(Path(path) / str(product.level)), sep, *index))
                if not member.is_file():
                    return None
                if sync:
                    with member.open("rb") as file:
                        os.fsync(file.fileno())
                    descriptor = os.open(member.parent, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                files.append((str(member.relative_to(self.root)), member.stat().st_size, sha256_file(member)))
        return json.dumps(files, separators=(",", ":"))

    def ready(self, origin):
        entry = self.chunks.get(origin)
        if entry is not None:
            return entry["ready"]
        row = self.db.execute("SELECT fingerprints FROM chunks WHERE key=? AND state='ready'", (_key(origin),)).fetchone()
        if row and row[0] == self._fingerprints(origin):
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
                                   "allocated": False, "publishing": False}
            self.db.execute("INSERT OR REPLACE INTO chunks VALUES (?, 'partial', NULL)", (_key(origin),))

    def request(self, index, *, remaining=0.0, background=False):
        origin = self.geometry.origin(index)
        self._register(origin)
        entry = self.chunks[origin]
        if not entry["ready"]:
            entry["requested"] = True
            priority = (0 if background else 1, float(remaining))
            for tile in entry["dependencies"]:
                if tile not in self.tiles:
                    self.priorities[tile] = max(priority, self.priorities.get(tile, (-1, 0)))
                    self.db.execute("INSERT OR IGNORE INTO tiles VALUES (?, 'queued')", (_key(tile),))
            self._queue_publication(origin)
        return origin

    def next_tile(self):
        candidates = [tile for tile in self.priorities if tile not in self.tiles]
        if not candidates:
            return None
        tile = min(candidates, key=lambda t: (-self.priorities[t][0], -self.priorities[t][1], t))
        self.tiles[tile] = {"state": "inflight"}
        self.db.execute("UPDATE tiles SET state='inflight' WHERE key=?", (_key(tile),))
        return tile

    def _descriptor(self, arrays):
        return tiled._MmapBandDescriptor(tuple(str(a.filename) for a in arrays),
                                         self.geometry.output_shape, str(arrays[0].dtype),
                                         self.geometry.chunk if self.layout == "chunk-major" else 0, self.prefault)

    def begin_complete(self, tile, result_spec, slot, sequence):
        """Plan chunk-owner tasks without copying or persisting result tensors."""
        if self.tiles.get(tile, {}).get("state") != "inflight":
            raise ValueError("duplicate/unrequested inference completion")
        self._check_disk_budget()
        tasks = []
        for origin in self.geometry.touched(tile):
            self._register(origin)
            entry = self.chunks[origin]
            if entry["ready"]:
                continue
            if tile not in entry["dependencies"] or tile in entry["pending"] or tile in entry["applied"]:
                raise ValueError("invalid/duplicate contribution")
            if not entry["allocated"]:
                size = (self.geometry.chunk ** 3 if self.layout == "chunk-major" else
                        math.prod(min(self.geometry.chunk, n - o)
                                  for n, o in zip(self.geometry.output_shape, origin)))
                size *= sum(a.dtype.itemsize for a in self.arrays)
                if self.scratch_bytes + size > self.budget:
                    raise RuntimeError("live inference scratch budget exhausted")
                self.scratch_bytes += size
                entry["allocated"] = True
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
        self.db.execute("UPDATE tiles SET state='accumulating' WHERE key=?", (_key(tile),))
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
        self.db.execute("UPDATE tiles SET state='consumed' WHERE key=?", (_key(tile),))

    def _queue_publication(self, origin):
        entry = self.chunks[origin]
        if (not entry["ready"] and entry["requested"] and not entry["publishing"]
                and not entry["pending"] and len(entry["applied"]) == len(entry["dependencies"])):
            entry["publishing"] = True
            self.publications.append(origin)

    def publication_done(self, origin, fingerprints):
        # Called only by coordinator AFTER the background writer has finished.
        if not fingerprints:
            raise RuntimeError("inference finalization did not publish all coherent channel chunks")
        entry = self.chunks[origin]
        if entry["ready"] or not entry["publishing"]:
            raise ValueError("duplicate/unexpected publication acknowledgement")
        self.db.execute("UPDATE chunks SET state='ready', fingerprints=? WHERE key=?", (fingerprints, _key(origin)))
        self.db.commit()
        entry["ready"] = True
        self.stats["published_chunks"] += 1

    def _check_disk_budget(self):
        # Advisory only: no preallocation, zero writes, or scratch fsync. Another
        # process can still exhaust disk after this check; worker failure aborts.
        allocated = sum(Path(a.filename).stat().st_blocks * 512 for a in self.arrays)
        if max(allocated, self.scratch_bytes) > self.budget or shutil.disk_usage(self.root).free < (64 << 20):
            raise RuntimeError("live inference scratch budget/free disk exhausted")

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
        return self._fingerprints(origin, sync=True)

    def close(self):
        for array in getattr(self, "arrays", ()):
            array._mmap.close()
        self.arrays = []
        for directory in getattr(self, "scratch_dirs", ()):
            # Only this instance's mkdtemp-created scratch directory is removed.
            shutil.rmtree(directory)
            if hasattr(self, "db"):
                self.db.execute("DELETE FROM scratch WHERE path=?", (str(directory.resolve()),))
        self.scratch_dirs = []
        if hasattr(self, "db"):
            self.db.commit()
            self.db.close()
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
    def __init__(self, stores, readers, devices, *, slots_per_gpu=2, input_workers=8,
                 result_slots=None, accumulator_workers=None, flush_workers=4, profile_pipeline=False):
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
        count = len(devices) * int(slots_per_gpu)
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
        self.profile_samples = []
        self.accumulator_samples = []
        self.accumulator_queue_samples = []
        self.accumulator_submitted = {}
        self.accumulator_backends = {}
        self.worker_ids = set()
        self.stats = {"tiles": 0, "worker_seconds": 0.0, "load_seconds": 0.0,
                      "input_slots": count, "result_slots": result_count,
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
                    self.work, self.events, profile_pipeline))
                process.start()
                self.processes.append(process)
        except BaseException:
            self.close()
            raise

    @property
    def busy(self):
        return bool(self.active or self.loading or self.accumulating or self.publishing or any(
            s.publications or any(tile not in s.tiles for tile in s.priorities) for s in self.stores.values()))

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

    def pump(self):
        for process in self.processes + self.accumulator_processes:
            if process.exitcode is not None:
                raise RuntimeError(f"inference/accumulator worker {process.name} exited unexpectedly: {process.exitcode}")
        self._pump_accumulators()
        self._pump_publications()
        while self.free_inputs:
            choices = []
            for name, store in self.stores.items():
                for tile, priority in store.priorities.items():
                    if tile not in store.tiles:
                        choices.append((-priority[0], -priority[1], name, tile))
            if not choices:
                break
            _, _, name, _ = min(choices)
            tile = self.stores[name].next_tile()
            slot, sequence = self.free_inputs.popleft(), self.sequence
            self.sequence += 1
            started = time.perf_counter()
            future = self.executor.submit(self.readers[name], tile)
            self.loading[sequence] = (name, tile, slot, future, started)
        for sequence, (name, tile, slot, future, started) in list(self.loading.items()):
            if not future.done() or not self.free:
                continue
            raw = np.asarray(future.result())
            if raw.shape != self.input_layouts[0].shape or raw.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
                raise ValueError("inference input reader returned invalid raw tile")
            layout = replace(self.input_layouts[0], dtype=raw.dtype.str)
            tiled._slot_array(self.inputs[slot], layout)[:] = raw
            names = tuple(p.name for p in self.stores[name].products)
            result_slot = self.free.popleft()
            self.work.put_nowait((sequence, tile, slot, result_slot, names, raw.dtype.str))
            self.input_owners[sequence] = slot
            self.active[sequence] = (name, tile, result_slot)
            del self.loading[sequence]
            self.stats["load_seconds"] += time.perf_counter() - started
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
        self._pump_accumulators()
        self._pump_publications()

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
