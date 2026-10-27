"""Chunk prefetch with bounded scheduling and replaceable lookahead windows.

The same scheduler runs locally or behind a spawn-based multiprocessing client.
Only descriptors cross the service boundary for speculation. Demand can return
one owning chunk when eviction (or a small disk budget) prevents cache handoff.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass, field
import hashlib
import itertools
import math
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import operator
import os
import pickle
import threading
import time
import uuid
import weakref


def _positive(name, value):
    value = operator.index(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _keys(reader, batches, limit):
    """Bound expansion before materializing a large region's chunk footprint."""
    keys = {}
    for regions in batches:
        for start, size in regions:
            start, size = tuple(map(operator.index, start)), tuple(map(operator.index, size))
            if len(start) != reader.ndim or len(size) != reader.ndim or any(n < 0 for n in size):
                raise ValueError(f"regions need {reader.ndim} start/size entries and nonnegative sizes")
            lo = [max(s, 0) for s in start]
            hi = [min(s + n, d) for s, n, d in zip(start, size, reader.shape)]
            if any(h <= l for l, h in zip(lo, hi)):
                continue
            for key in itertools.product(*(range(l // c, (h - 1) // c + 1)
                                           for l, h, c in zip(lo, hi, reader.chunks))):
                keys[key] = None
                if len(keys) > limit:
                    raise ValueError("prefetch plan exceeds max_plan_chunks")
    return tuple(keys)


@dataclass
class _Plan:
    keys: set
    remaining: set
    todo: deque
    window: bool
    errors: dict = field(default_factory=dict)
    cancelled: bool = False
    forgotten: bool = False


@dataclass
class _Job:
    target: str
    index: tuple
    running: bool = False
    demand: Future | None = None


class _Scheduler:
    """One condition protects plans, jobs and admission; I/O never holds it."""

    def __init__(self, connections=8, max_pending_chunks=512, max_inflight_bytes=256 << 20,
                 max_plan_chunks=65536, max_requests=1024):
        self.connections = _positive("connections", connections)
        self.max_pending_chunks = _positive("max_pending_chunks", max_pending_chunks)
        self.max_inflight_bytes = _positive("max_inflight_bytes", max_inflight_bytes)
        self.max_plan_chunks = _positive("max_plan_chunks", max_plan_chunks)
        self.max_requests = _positive("max_requests", max_requests)
        self.cv = threading.Condition()
        self.targets = {}
        self.plans = OrderedDict()
        self.jobs = OrderedDict()
        self.active = self.active_bytes = self.speculative = 0
        self.stopped = False
        self.counts = dict(completed=0, errors=0, promoted=0, deduplicated=0, cancelled=0)
        self.threads = [threading.Thread(target=self._worker, name="vc-reader-prefetch", daemon=True)
                        for _ in range(self.connections)]
        for thread in self.threads:
            thread.start()

    def register(self, name, load, nbytes):
        with self.cv:
            self._check()
            if nbytes > self.max_inflight_bytes:
                raise ValueError("one decoded chunk exceeds max_inflight_bytes")
            self.targets[name] = (load, nbytes)

    def _check(self):
        if self.stopped:
            raise RuntimeError("prefetch scheduler is closed")

    def update(self, owner, target, indices, window=False):
        keys = tuple((target, tuple(index)) for index in indices)
        keyset = set(keys)
        with self.cv:
            self._check()
            if target not in self.targets:
                raise ValueError("unknown prefetch array")
            old = self.plans.get(owner)
            if old is None and len(self.plans) >= self.max_requests:
                raise ValueError("too many live prefetch requests/windows; release completed requests")
            total = sum(len(p.keys) for p in self.plans.values()) - (len(old.keys) if old else 0)
            if total + len(keyset) > self.max_plan_chunks:
                raise ValueError("live prefetch plans exceed max_plan_chunks")
            done = old.keys - old.remaining if old and window else set()
            remaining = keyset - done
            errors = {k: e for k, e in old.errors.items() if k in keyset} if old and window else {}
            self.plans[owner] = _Plan(keyset, remaining, deque(k for k in keys if k in remaining),
                                       window, errors)
            self._prune()
            self.cv.notify_all()

    def _wanted(self, key):
        return any(key in p.remaining for p in self.plans.values())

    def _prune(self):
        for key, job in list(self.jobs.items()):
            if not job.running and job.demand is None and not self._wanted(key):
                del self.jobs[key]
                self.counts["cancelled"] += 1

    def _fill(self):
        # Round-robin windows, preserving each window's near-to-far order.
        visits = 0
        while self.plans and len(self.jobs) < self.max_pending_chunks:
            owner, plan = self.plans.popitem(last=False)
            self.plans[owner] = plan
            while plan.todo and plan.todo[0] not in plan.remaining:
                plan.todo.popleft()
            if plan.todo:
                key = plan.todo.popleft()
                if key in self.jobs:
                    self.counts["deduplicated"] += 1
                else:
                    self.jobs[key] = _Job(*key)
                visits = 0
            else:
                visits += 1
                if visits >= len(self.plans):
                    break

    def demand(self, target, index, timeout=None):
        key = target, tuple(index)
        with self.cv:
            self._check()
            while key not in self.jobs and len(self.jobs) >= self.max_pending_chunks:
                victim = next((k for k, j in self.jobs.items() if not j.running and j.demand is None), None)
                if victim is not None:
                    del self.jobs[victim]
                    for plan in self.plans.values():
                        if victim in plan.remaining:
                            plan.todo.appendleft(victim)
                    break
                self.cv.wait()
                self._check()
            job = self.jobs.setdefault(key, _Job(*key))
            if job.demand is None:
                job.demand = Future()
                if job.running or self._wanted(key):
                    self.counts["promoted"] += 1
            else:
                self.counts["deduplicated"] += 1
            future = job.demand
            self.cv.notify_all()
        return future.result(timeout)

    def _select(self):
        self._fill()
        largest = max((size for _, size in self.targets.values()), default=0)
        reserve = largest if self.connections > 1 and self.max_inflight_bytes >= 2 * largest else 0
        for demand in (True, False):
            for key, job in self.jobs.items():
                if job.running or (job.demand is not None) != demand:
                    continue
                size = self.targets[job.target][1]
                limit = self.max_inflight_bytes - (0 if demand else reserve)
                if self.active_bytes + size > limit:
                    continue
                if not demand and self.speculative >= max(1, self.connections - 1):
                    continue
                job.running = True
                self.active += 1
                self.active_bytes += size
                self.speculative += not demand
                return key, job, size, not demand
        return None

    def _worker(self):
        while True:
            with self.cv:
                while not self.stopped and (selected := self._select()) is None:
                    self.cv.wait()
                if self.stopped:
                    return
                key, job, size, speculative = selected
                load = self.targets[job.target][0]
            value, error, published = None, None, False
            try:
                value, published = load(job.index)
            except Exception as exc:
                error = exc
            with self.cv:
                self.active -= 1
                self.active_bytes -= size
                self.speculative -= speculative
                del self.jobs[key]
                publication_error = error
                if publication_error is None and not published:
                    publication_error = RuntimeError(
                        "prefetched chunk could not be published to the disk cache (budget or write failure); "
                        "demand reads can still consume it")
                failed = error is not None or (publication_error is not None and self._wanted(key))
                self.counts["errors" if failed else "completed"] += 1
                for owner, plan in list(self.plans.items()):
                    if key in plan.remaining:
                        plan.remaining.remove(key)
                        if publication_error is not None:
                            plan.errors[key] = publication_error
                        if not plan.remaining:
                            plan.todo.clear()
                            if plan.forgotten:
                                del self.plans[owner]
                            elif not plan.window:
                                plan.keys.clear()
                if job.demand is not None:
                    if error is not None:
                        job.demand.set_exception(error)
                    else:
                        job.demand.set_result(value)
                self.cv.notify_all()
            # Futures belong only to active demand callers; speculation retains
            # status, never decoded payloads outside the configured caches.
            del value, job, selected

    def status(self, owner):
        with self.cv:
            plan = self.plans[owner]
            return dict(done=not plan.remaining, cancelled=plan.cancelled,
                        error=next(iter(plan.errors.values()), None))

    def wait(self, owner, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.cv:
            while self.plans[owner].remaining:
                self._check()
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("prefetch did not finish before timeout")
                self.cv.wait(remaining)
            state = self.status(owner)
        if state["cancelled"]:
            raise CancelledError()
        if state["error"] is not None:
            raise state["error"]

    def cancel(self, owner):
        with self.cv:
            plan = self.plans[owner]
            if not plan.remaining:
                return False
            plan.cancelled = True
            plan.remaining.clear()
            plan.todo.clear()
            plan.keys.clear()
            self._prune()
            self.cv.notify_all()
            return True

    def release(self, owner, cancel=False):
        with self.cv:
            plan = self.plans.get(owner)
            if plan is None:
                return
            if cancel or not plan.remaining:
                del self.plans[owner]
                self._prune()
            else:
                plan.forgotten = True
            self.cv.notify_all()

    def stats(self):
        with self.cv:
            return dict(self.counts, active=self.active, active_bytes=self.active_bytes,
                        queued=sum(not j.running for j in self.jobs.values()),
                        plans=len(self.plans), planned_chunks=sum(len(p.remaining) for p in self.plans.values()))

    def close(self, wait=True):
        with self.cv:
            self.stopped = True
            for key, job in list(self.jobs.items()):
                if not job.running:
                    if job.demand is not None:
                        job.demand.set_exception(RuntimeError("prefetch scheduler is closed"))
                    del self.jobs[key]
            self.cv.notify_all()
        if wait:
            for thread in self.threads:
                thread.join()
            self.targets.clear()


def _release(backend, owner, cancel, pid):
    if os.getpid() == pid:
        try:
            backend.release(owner, cancel)
        except (EOFError, OSError, KeyError):
            pass  # The owning service may already have shut down.


class PrefetchRequest:
    """Completion of one prefetch submission, without retaining chunk payloads.

    ``wait`` raises loading/publication errors; completion is not a residency
    lease. ``cancel`` withdraws interest in queued work, not other callers' work.
    """
    def __init__(self, backend, owner):
        self._backend, self._owner, self._pid = backend, owner, os.getpid()
        self._finalizer = weakref.finalize(self, _release, backend, owner, False, self._pid)

    def _check(self):
        if os.getpid() != self._pid:
            raise RuntimeError("create prefetch requests/windows inside each worker")

    def __getstate__(self):
        raise TypeError("pass the reader or service.client to workers; create prefetch requests/windows there")

    def done(self):
        self._check()
        return self._backend.status(self._owner)["done"]

    def exception(self):
        self._check()
        return self._backend.status(self._owner)["error"]

    def wait(self, timeout=None):
        self._check()
        self._backend.wait(self._owner, timeout)

    def cancel(self):
        self._check()
        return self._backend.cancel(self._owner)


class PrefetchWindow(PrefetchRequest):
    """An ordered, replaceable set of future region batches owned by one caller."""
    def __init__(self, reader, backend, target, limit):
        owner = uuid.uuid4().hex
        backend.update(owner, target, (), True)
        super().__init__(backend, owner)
        self._finalizer.detach()
        self._finalizer = weakref.finalize(self, _release, backend, owner, True, self._pid)
        self._reader, self._target, self._limit = reader, target, limit
        self._closed = False

    def update(self, batches):
        self._check()
        if self._closed:
            raise ValueError("prefetch window is closed")
        self._reader._ready()
        self._backend.update(self._owner, self._target, _keys(self._reader, batches, self._limit), True)

    def close(self):
        self._check()
        self._closed = True
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _local(reader):
    """Lazily opt a reader into scheduling; readers without prefetch are untouched."""
    scheduler = _Scheduler(connections=reader._spec["connections"])
    array = reader._array
    from .auth import is_remote
    remote = is_remote(reader.location)
    nbytes = reader.dtype.itemsize * math.prod(reader.chunks)
    memory = reader.memory_cache is not None and reader.memory_cache.max_bytes >= nbytes

    def load(index):
        chunk = array.load_chunk(index)
        # Locally, an existing MemoryCache can be the sole prefetch destination.
        published = array.publish_chunk(index) if remote else True
        published = published or memory
        return chunk, published

    try:
        if remote and not reader.cache_layers and not memory:
            raise ValueError("remote prefetch needs a disk cache or memory_cache large enough for one chunk")
        scheduler.register("array", load, nbytes)
    except BaseException:
        scheduler.close()
        raise
    return scheduler, "array"


class _ServiceBackend:
    """Manager-owned scheduler and C++ readers; no training-specific state."""
    def __init__(self, options):
        self.scheduler = _Scheduler(**options)
        self.readers = {}
        self.lock = threading.Lock()

    def register(self, spec):
        from .zarr_reader import ZarrReader
        spec = dict(spec, prefetcher=None, threads=1, connections=self.scheduler.connections)
        # Reader identity includes cache/fidelity/auth policy; compatible workers
        # share a reader even when their local thread counts differ.
        identity = dict(spec, meta=None, attrs=None, access=spec["access"].auth_tuple(),
                        resolved_url=spec["access"].url(spec["location"]))
        name = hashlib.sha256(pickle.dumps(identity)).hexdigest()
        with self.lock:
            if name not in self.readers:
                if len(self.readers) >= 128:
                    raise ValueError("prefetch service supports at most 128 distinct array configurations")
                reader = ZarrReader(spec, None)

                def load(index):
                    reader._ready()
                    chunk = reader._array.load_chunk(index)
                    return chunk, reader._array.publish_chunk(index)

                try:
                    import math
                    self.scheduler.register(name, load, math.prod(reader.chunks) * reader.dtype.itemsize)
                except BaseException:
                    reader.close()
                    raise
                self.readers[name] = reader
            return name, dict(self.readers[name].meta.raw)

    def demand(self, target, index):
        return self.scheduler.demand(target, index).to_bytes()

    def update(self, *args):
        return self.scheduler.update(*args)

    def status(self, *args):
        return self.scheduler.status(*args)

    def wait(self, *args):
        return self.scheduler.wait(*args)

    def cancel(self, *args):
        return self.scheduler.cancel(*args)

    def release(self, *args):
        return self.scheduler.release(*args)

    def stats(self):
        return self.scheduler.stats()

    def stop(self):
        self.scheduler.close(wait=False)


class _Manager(BaseManager):
    pass


_Manager.register("Backend", _ServiceBackend)


class PrefetchClient:
    """Picklable endpoint passed to ``ZarrReader.open(prefetcher=...)``."""
    def __init__(self, backend, max_plan_chunks):
        self._backend, self.max_plan_chunks = backend, max_plan_chunks

    def register(self, spec):
        return self._backend.register(spec)

    def demand(self, target, index):
        from .. import _reader
        try:
            return _reader.ChunkHandle.from_bytes(self._backend.demand(target, index))
        except (EOFError, BrokenPipeError, ConnectionError, FileNotFoundError) as exc:
            raise RuntimeError("prefetch service stopped; no independent download attempted") from exc


class PrefetchService:
    """One explicitly owned process coordinating readers across loader workers.

    Create in the parent under the usual ``if __name__ == '__main__'`` guard;
    pass ``client`` to workers. Requires writable disk caches for remote arrays.
    Connections bound concurrent chunk loads; ``max_inflight_bytes`` bounds
    their decoded sizes, not total RSS (HTTP, IPC and output buffers also exist).
    One slot and one chunk's byte capacity are reserved for demand when possible.
    """
    def __init__(self, connections=8, max_pending_chunks=512, max_inflight_bytes=256 << 20,
                 max_plan_chunks=65536, max_requests=1024):
        options = {name: _positive(name, value) for name, value in dict(
            connections=connections, max_pending_chunks=max_pending_chunks,
            max_inflight_bytes=max_inflight_bytes, max_plan_chunks=max_plan_chunks,
            max_requests=max_requests).items()}
        self._pid = os.getpid()
        self._manager = _Manager(ctx=mp.get_context("spawn"))
        self._manager.start()
        try:
            backend = self._manager.Backend(options)
        except BaseException:
            self._manager.shutdown()
            raise
        self.client = PrefetchClient(backend, max_plan_chunks)
        self._closed = False

    def stats(self):
        return self.client._backend.stats()

    def close(self):
        if os.getpid() != self._pid:
            raise RuntimeError("only the parent may close the prefetch service")
        if not self._closed:
            self._closed = True
            try:
                self.client._backend.stop()
            finally:
                self._manager.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
