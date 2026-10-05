"""Async CT prefetch in a separate process, with preemptive demand priority.

Two bounded IPC queues feed a priority heap. Loader workers only memory-map
ready chunks; all remote metadata/chunk I/O uses Zarr's async API here.
"""
import asyncio
from collections import deque, OrderedDict
import heapq
from functools import lru_cache
from itertools import product
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import uuid

import numpy as np

from vesuvius.neural_tracing.fiber_follow.data.volume import RemoteChunkedArray, open_remote_array

DEMAND, AHEAD = 0, 10
COUNTERS = ('submitted', 'deferred', 'cached', 'completed', 'completed_bytes',
            'deduplicated', 'promoted', 'preempted', 'errors', 'active',
            'lookahead_windows', 'lookahead_chunks')


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


@lru_cache(maxsize=128)
def _chunk_separator(root):
    # Metadata is published before chunk requests and stays fixed for this array.
    return json.loads((root/'.zarray').read_text()).get('dimension_separator','.')


def _paths(descriptor, key, session):
    root = RemoteChunkedArray.cache_path(*descriptor)
    name = '.zarray' if key is None else '.'.join(map(str,key))
    chunk_path = name if key is None else _chunk_separator(root).join(map(str,key))
    return root/chunk_path, root/'.prefetch_errors'/session/(name+'.json')


class PrefetchClient:
    """Picklable endpoint for regular workers; submission never waits for space."""
    def __init__(self, urgent, ahead, stopped, heartbeat, counters, session, timeout, windows=None):
        self.urgent, self.ahead, self.stopped = urgent, ahead, stopped
        self.heartbeat, self.counters = heartbeat, counters
        self.session, self.timeout, self._pid = session, timeout, None
        self.windows = windows
        self._pending_windows = {}

    def _connect(self):
        if self._pid != os.getpid():
            for requests in (self.urgent,self.ahead,self.windows):
                if requests is not None:
                    requests.cancel_join_thread()
            self._pid = os.getpid()

    def _publish_windows(self):
        self._connect()
        for (scope,descriptor),keys in list(self._pending_windows.items()):
            try:
                self.windows.put_nowait(((os.getpid(),scope),descriptor,keys))
            except queue.Full:
                _add(self.counters,'deferred')
                break
            del self._pending_windows[(scope,descriptor)]

    def lookahead(self, reader, batches, *, scope=None):
        """Publish the bounded plan window; chunk-queue capacity does not trim it."""
        if not isinstance(reader,RemoteChunkedArray) or self.windows is None:
            return
        descriptor = (reader.remote_url,reader.level,reader.cache_dir)
        # Preserve batch order while downloading each overlapping chunk once.
        keys = tuple(dict.fromkeys(key for bounds in batches for key in self._keys(reader,bounds)))
        self._pending_windows[(scope,descriptor)] = keys
        self._publish_windows()

    def _submit(self, descriptor, keys, priority):
        if self.stopped.is_set():
            return
        self._connect()
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
            if self._pending_windows:
                self._publish_windows()
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
        if not spec.cache_dir:
            raise ValueError('Remote CT prefetch requires a persistent cache_dir')
        self._ensure((str(spec.ct_zarr).rstrip('/'),int(spec.ct_level),str(spec.cache_dir)),[None])
        if getattr(spec, 'crop_ct_level', None) is not None:  # model crops read another level of the same store
            self._ensure((str(spec.ct_zarr).rstrip('/'),int(spec.crop_ct_level),str(spec.cache_dir)),[None])

    def ensure(self, reader, bounds):
        if isinstance(reader,RemoteChunkedArray):
            self._ensure((reader.remote_url,reader.level,reader.cache_dir),self._keys(reader,bounds))


class AsyncFetcher:
    def __init__(self, session, connections):
        self.session, self.opening = session, {}
        self.connections = connections

    async def array(self, descriptor):
        if descriptor not in self.opening:
            self.opening[descriptor] = asyncio.create_task(open_remote_array(
                *descriptor[:2],max_pool_connections=self.connections))
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
        lock_path = RemoteChunkedArray.cache_path(*descriptor)/'.locks'/'.'.join(map(str,key))
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
            result = np.full(array.chunks,getattr(array,'metadata',array).fill_value or 0,array.dtype)
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


async def _run(urgent,ahead,stopped,heartbeat,counters,session,connections,queue_size,
               window_queue=None,lookahead_slots=0):
    fetcher = AsyncFetcher(session,connections)
    heap, waiting, active = [], {}, {}
    windows = OrderedDict()
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
            if window_queue is not None:
                # Drain plan updates even with a full chunk heap. Each source
                # and worker replaces its old window; only keys are retained.
                for _ in range(2*lookahead_slots):
                    try:
                        owner,descriptor,keys = window_queue.get_nowait()
                    except queue.Empty:
                        break
                    slot = (owner,descriptor)
                    if not keys:
                        windows.pop(slot,None)
                    elif slot in windows or len(windows) < lookahead_slots:
                        windows[slot] = deque(keys)
                    else:
                        _add(counters,'deferred')
                # Round-robin sources/workers; nearby batches lead each window.
                # Keep in-flight/evicted keys until the file exists, so demand
                # preemption and a small chunk queue cannot lose future work.
                visits = 0
                remaining = sum(map(len,windows.values()))
                while windows and len(waiting) < queue_size and visits < min(remaining,queue_size):
                    slot,keys = windows.popitem(last=False)
                    descriptor = slot[1]
                    key = keys.popleft()
                    path,error = _paths(descriptor,key,session)
                    if not path.is_file() and not error.is_file():
                        request = (descriptor,key)
                        if request not in waiting and request not in active:
                            enqueue(request,AHEAD)
                        keys.append(key)
                    if keys:
                        windows[slot] = keys
                    visits += 1
                with counters.get_lock():
                    counters[COUNTERS.index('lookahead_windows')] = len(windows)
                    counters[COUNTERS.index('lookahead_chunks')] = sum(map(len,windows.values()))
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
        with counters.get_lock():
            counters[COUNTERS.index('lookahead_windows')] = 0
            counters[COUNTERS.index('lookahead_chunks')] = 0


def _serve(*args):
    try:
        asyncio.run(_run(*args))
    finally:
        args[2].set()


def attach_remote_prefetch(datasets, connections, queue_size, timeout, lookahead, workers):
    """Start the async CT prefetch process and hand its client to every remote-CT dataset.

    Datasets are FollowDataset-like (``vol_spec``, ``remote_prefetch``, ``remote_prefetch_lookahead``).
    Returns (prefetcher or None, remote datasets). The caller owns the prefetcher's lifetime.
    """
    remote = [d for d in datasets if str(d.vol_spec.ct_zarr).startswith(('s3://', 'http://', 'https://'))]
    if not connections or not remote:
        return None, remote
    prefetcher = RemotePrefetcher(connections, queue_size, timeout,
                                  lookahead_slots=max(1, workers)*len(remote) if lookahead else 0)
    for dataset in remote:
        dataset.remote_prefetch = prefetcher.client
        dataset.remote_prefetch_lookahead = lookahead
    return prefetcher, remote


class RemotePrefetcher:
    """Trainer-owned lifetime; DataLoader daemons never create child processes."""
    def __init__(self, connections=8, queue_size=512, timeout=120, *, lookahead_slots=0):
        if connections < 1 or queue_size < 1 or lookahead_slots < 0 or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('Prefetch connections, queue size and timeout must be positive')
        context = mp.get_context('spawn')
        self.urgent, self.ahead = (context.Queue(maxsize=queue_size) for _ in range(2))
        self.windows = context.Queue(maxsize=2*lookahead_slots) if lookahead_slots else None
        self.stopped = context.Event()
        self.heartbeat = context.Value('d',time.monotonic())
        self.counters = context.Array('q',len(COUNTERS))
        session = uuid.uuid4().hex
        self.client = PrefetchClient(self.urgent,self.ahead,self.stopped,self.heartbeat,self.counters,session,timeout,self.windows)
        self.process = context.Process(target=_serve,
            args=(self.urgent,self.ahead,self.stopped,self.heartbeat,self.counters,session,connections,queue_size,
                  self.windows,lookahead_slots),
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
        for requests in (self.urgent,self.ahead,self.windows):
            if requests is not None:
                requests.cancel_join_thread()
                requests.close()

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()
