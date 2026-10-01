"""Async CT prefetch in a separate process, with preemptive demand priority.

Two bounded IPC queues feed a priority heap. Loader workers only memory-map
ready chunks; all remote metadata/chunk I/O uses Zarr's async API here.
"""
import asyncio
import heapq
from itertools import product
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import uuid

import numpy as np

from .volume import RemoteChunkedArray, open_remote_array

DEMAND, AHEAD = 0, 10
COUNTERS = ('submitted', 'deferred', 'cached', 'completed', 'completed_bytes',
            'deduplicated', 'promoted', 'preempted', 'errors', 'active')


def _add(counters, name, amount=1):
    with counters.get_lock():
        counters[COUNTERS.index(name)] += amount


def chunk_keys(start, size, shape, chunks):
    """Same clipped chunk footprint as ChunkedArray.read (ZYX)."""
    start, size, shape, chunks = map(np.asarray, (start, size, shape, chunks))
    lo, hi = np.maximum(start, 0), np.minimum(start+size, shape)
    if np.any(hi <= lo):
        return
    yield from product(*(range(int(a), int(b)+1) for a,b in zip(lo//chunks, (hi-1)//chunks)))


def _paths(descriptor, key, session):
    root = RemoteChunkedArray.cache_path(*descriptor)
    name = '.zarray' if key is None else '.'.join(map(str,key))
    return root/name, root/'.prefetch_errors'/session/(name+'.json')


class PrefetchClient:
    """Picklable endpoint for regular workers; submission never waits for space."""
    def __init__(self, urgent, ahead, stopped, heartbeat, counters, session, timeout):
        self.urgent, self.ahead, self.stopped = urgent, ahead, stopped
        self.heartbeat, self.counters = heartbeat, counters
        self.session, self.timeout, self._pid = session, timeout, None

    def _submit(self, descriptor, keys, priority):
        if self.stopped.is_set():
            return
        if self._pid != os.getpid():
            self.urgent.cancel_join_thread()
            self.ahead.cancel_join_thread()
            self._pid = os.getpid()
        destination = self.urgent if priority == DEMAND else self.ahead
        for key in keys:
            path,_ = _paths(descriptor,key,self.session)
            if path.is_file():
                _add(self.counters,'cached')
                continue
            try:
                destination.put_nowait((descriptor,key))
                _add(self.counters,'submitted')
            except queue.Full:
                # The readiness gate resubmits every still-missing demand.
                _add(self.counters,'deferred')

    @staticmethod
    def _keys(reader, bounds):
        return sorted({key for start,size in bounds for key in chunk_keys(start,size,reader.shape,reader.chunks)})

    def submit(self, reader, bounds):
        if isinstance(reader,RemoteChunkedArray):
            self._submit((reader.remote_url,reader.level,reader.cache_dir),self._keys(reader,bounds),AHEAD)

    def _ensure(self, descriptor, keys):
        pending = list(keys)
        deadline, next_submit = time.monotonic()+self.timeout, 0.
        while pending:
            pending = [key for key in pending if not _paths(descriptor,key,self.session)[0].is_file()]
            if not pending:
                return
            for key in pending:
                _,error_path = _paths(descriptor,key,self.session)
                if error_path.exists():
                    raise RuntimeError(f'Remote CT prefetch failed: {error_path.read_text()}')
            now = time.monotonic()
            if self.stopped.is_set() or now-self.heartbeat.value > 30:
                raise RuntimeError('Remote CT prefetch service stopped or became unresponsive')
            if now >= deadline:
                raise TimeoutError(f'Timed out waiting for {len(pending)} prefetched CT chunks; no foreground download attempted')
            if now >= next_submit:
                self._submit(descriptor,pending,DEMAND)
                next_submit = now+.25
            # Waiting for readiness is distinct from executing blocking Zarr I/O.
            # Other loader workers and async requests continue independently.
            self.stopped.wait(.01)

    def ensure_metadata(self, spec):
        self._ensure((str(spec.ct_zarr).rstrip('/'),int(spec.ct_level),str(spec.cache_dir)),[None])

    def ensure(self, reader, bounds):
        if isinstance(reader,RemoteChunkedArray):
            self._ensure((reader.remote_url,reader.level,reader.cache_dir),self._keys(reader,bounds))


class AsyncFetcher:
    def __init__(self, session):
        self.session, self.opening = session, {}

    async def array(self, descriptor):
        if descriptor not in self.opening:
            self.opening[descriptor] = asyncio.create_task(open_remote_array(*descriptor[:2]))
        task = self.opening[descriptor]
        try:
            # Preempting one chunk must not cancel shared metadata needed by others.
            return await asyncio.shield(task)
        except Exception:
            if self.opening.get(descriptor) is task:
                self.opening.pop(descriptor)
            raise

    async def fetch(self, descriptor, key):
        import fcntl
        path,_ = _paths(descriptor,key,self.session)
        if path.is_file():
            return 'cached',0
        if key is None:
            array = await self.array(descriptor)
            await asyncio.to_thread(RemoteChunkedArray.publish_metadata,path.parent,array,*descriptor[:2])
            return 'completed',0
        lock_path = path.parent/'.locks'/path.name
        lock_path.parent.mkdir(parents=True,exist_ok=True)
        with lock_path.open('a+b') as lock:
            while True:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    # Never block the event loop behind another process's chunk.
                    await asyncio.sleep(.01)
            if path.is_file():
                return 'cached',0
            array = await self.array(descriptor)
            start = np.asarray(key)*array.chunks
            stop = np.minimum(start+array.chunks,array.shape)
            values = await array.getitem(tuple(slice(int(a),int(b)) for a,b in zip(start,stop)))
            result = np.full(array.chunks,array.fill_value or 0,array.dtype)
            result[tuple(slice(0,n) for n in values.shape)] = values
            # Disk writes run off the network event loop. Finish an atomic write
            # before releasing its chunk lock, even when the request is preempted.
            write = asyncio.create_task(asyncio.to_thread(RemoteChunkedArray._atomic_write,path,result.tobytes(order='C')))
            try:
                await asyncio.shield(write)
            except asyncio.CancelledError:
                await write
                raise
            return 'completed',result.nbytes

    async def request(self, descriptor, key):
        for attempt in range(3):
            try:
                return await asyncio.wait_for(self.fetch(descriptor,key),timeout=30)
            except Exception as error:
                if attempt < 2:
                    await asyncio.sleep(.25*2**attempt)
                    continue
                _,error_path = _paths(descriptor,key,self.session)
                await asyncio.to_thread(RemoteChunkedArray._atomic_write,error_path,
                    json.dumps(dict(source=descriptor[0],chunk=key,error=str(error))).encode())
                return 'errors',0

    async def close(self):
        for task in self.opening.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.opening.values(),return_exceptions=True)


async def _run(urgent,ahead,stopped,heartbeat,counters,session,connections,queue_size):
    fetcher = AsyncFetcher(session)
    heap, waiting, active = [], {}, {}
    serial = 0

    def enqueue(request,priority):
        nonlocal serial
        if request in active:
            task,old = active[request]
            if priority < old:
                active[request] = (task,priority)
                _add(counters,'promoted')
            else:
                _add(counters,'deduplicated')
            return
        old = waiting.get(request)
        if old is not None and old <= priority:
            _add(counters,'deduplicated')
            return
        if old is not None:
            _add(counters,'promoted')
        if old is None and len(waiting) >= queue_size:
            # Reserve space for demand by evicting a speculative hint.
            victim = next((r for r,p in waiting.items() if p > priority),None)
            if victim is None:
                _add(counters,'deferred')
                return
            del waiting[victim]
        waiting[request] = priority
        serial += 1
        heapq.heappush(heap,(priority,serial,request))

    try:
        while not stopped.is_set():
            heartbeat.value = time.monotonic()
            for request,(task,priority) in list(active.items()):
                if not task.done():
                    continue
                del active[request]
                _add(counters,'active',-1)
                if task.cancelled():
                    enqueue(request,priority)
                else:
                    status,byte_count = task.result()
                    _add(counters,status)
                    _add(counters,'completed_bytes',byte_count)
            # Always ingest demand first, even when every connection is busy.
            for _ in range(queue_size):
                try:
                    enqueue(urgent.get_nowait(),DEMAND)
                except queue.Empty:
                    break
            need = sum(p == DEMAND for p in waiting.values())-(connections-len(active))
            if need > 0:
                for request,(task,priority) in list(active.items()):
                    if need <= 0:
                        break
                    if priority > DEMAND and not task.cancelling():
                        task.cancel()
                        _add(counters,'preempted')
                        need -= 1
            for _ in range(max(0,queue_size-len(waiting))):
                try:
                    enqueue(ahead.get_nowait(),AHEAD)
                except queue.Empty:
                    break
            while heap and len(active) < connections:
                priority,_,request = heapq.heappop(heap)
                if waiting.get(request) != priority:
                    continue
                del waiting[request]
                active[request] = (asyncio.create_task(fetcher.request(*request)),priority)
                _add(counters,'active')
            # Promotions/evictions leave stale heap entries; keep memory bounded.
            if len(heap) > 2*queue_size:
                heap = [row for row in heap if waiting.get(row[2]) == row[0]]
                heapq.heapify(heap)
            await asyncio.sleep(.01)
    finally:
        tasks = [task for task,_ in active.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        await fetcher.close()
        _add(counters,'active',-len(active))


def _serve(*args):
    try:
        asyncio.run(_run(*args))
    finally:
        args[2].set()


class RemotePrefetcher:
    """Trainer-owned lifetime; DataLoader daemons never create child processes."""
    def __init__(self, connections=8, queue_size=512, timeout=120):
        if connections < 1 or queue_size < 1 or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('Prefetch connections, queue size and timeout must be positive')
        context = mp.get_context('spawn')
        self.urgent, self.ahead = (context.Queue(maxsize=queue_size) for _ in range(2))
        self.stopped = context.Event()
        self.heartbeat = context.Value('d',time.monotonic())
        self.counters = context.Array('q',len(COUNTERS))
        session = uuid.uuid4().hex
        self.client = PrefetchClient(self.urgent,self.ahead,self.stopped,self.heartbeat,self.counters,session,timeout)
        self.process = context.Process(target=_serve,
            args=(self.urgent,self.ahead,self.stopped,self.heartbeat,self.counters,session,connections,queue_size),
            name='remote-ct-prefetch',daemon=True)
        self.process.start()

    def snapshot(self):
        with self.counters.get_lock():
            result = dict(zip(COUNTERS,self.counters[:]))
        return dict(result,alive=self.process.is_alive())

    def close(self):
        self.stopped.set()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        for requests in (self.urgent,self.ahead):
            requests.cancel_join_thread()
            requests.close()

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()
