"""One-step-ahead prefetch of per-step CPU batch assembly.

The training loop's per-step CPU work (numpy patch-strip sampling, the track
point gather, and their host->device uploads) otherwise runs serially while
the GPU is idle: the track gather even forces a device sync. Prefetching runs
the *next* step's sampling on a worker thread while the GPU chews on the
current step.

Determinism: jobs draw randomness from dedicated per-key RNG streams (numpy
Generator / CUDA torch.Generator) seeded once from the global streams at
first use, so results do not depend on thread scheduling. This changes the
RNG stream relative to the non-prefetch path (different sequences, not just
reordered); the non-prefetch path is preserved and is the default
(FIT_SPIRAL_PREFETCH unset/0). Caveat when opting in: the per-key generator
states are not saved in checkpoints, so a resumed run re-seeds them and its
sampling sequence diverges from the uninterrupted run from that point on.

GPU work inside jobs runs on a dedicated side stream; consumers make their
stream wait on the job's recorded event before using the returned tensors.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


def prefetch_enabled():
    # Default OFF: on the GB10 the worker thread's numpy/gather work contends
    # for the GIL exactly while the main thread dispatches the GPU-bound
    # backward, and measured net -4 ms/step at 1000-slice / -44 ms/step at
    # 3000-slice scale. Worth retrying on CPU-bound machines (or free-threaded
    # Python) with FIT_SPIRAL_PREFETCH=1.
    return os.environ.get('FIT_SPIRAL_PREFETCH', '0') == '1'


class _GeneratorShim:
    # np.random.Generator with the legacy np.random module's method names, so
    # sampling code can run against either source.

    def __init__(self, gen):
        self._g = gen

    def random(self, size=None):
        return self._g.random(size)

    def randint(self, low, high=None, size=None):
        return self._g.integers(low, high, size=size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._g.uniform(low, high, size)

    def choice(self, a, size=None, replace=True, p=None):
        return self._g.choice(a, size=size, replace=replace, p=p)


class LegacyNumpyRandom:
    # Adapter with the same surface as _GeneratorShim but backed by the global
    # np.random stream (used by the non-prefetch path, keeping it bitwise
    # identical to the historical behaviour).

    random = staticmethod(np.random.random)
    randint = staticmethod(np.random.randint)
    uniform = staticmethod(np.random.uniform)
    choice = staticmethod(np.random.choice)


class StepPrefetcher:

    def __init__(self):
        self._executor = None
        self._pending = {}
        self._np_rngs = {}
        self._torch_rngs = {}
        self._stream = None

    def _ensure_executor(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix='spiral-prefetch')
        return self._executor

    def np_rng(self, key):
        rng = self._np_rngs.get(key)
        if rng is None:
            # Seeded once from the global stream: deterministic given the
            # deterministic first-use order, then independent of it.
            seed = int(np.random.randint(0, 2 ** 63 - 1))
            rng = _GeneratorShim(np.random.Generator(np.random.PCG64(seed)))
            self._np_rngs[key] = rng
        return rng

    def torch_rng(self, key, device):
        rng = self._torch_rngs.get(key)
        if rng is None:
            rng = torch.Generator(device=device)
            rng.manual_seed(int(torch.randint(2 ** 62, (1,)).item()))
            self._torch_rngs[key] = rng
        return rng

    def stream(self):
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        return self._stream

    def run_job(self, job):
        # Execute `job` under the side stream and record an event after it.
        with torch.cuda.stream(self.stream()):
            result = job()
            evt = torch.cuda.Event()
            evt.record(self.stream())
        return result, evt

    def pop_or_run(self, key, job):
        """Return this step's batch for `key` and schedule the next one.

        `job` must be self-contained (draw randomness only from the
        prefetcher's own RNGs) and returns arbitrary tensors; device tensors
        it creates must be allocated/copied on self.stream(). The returned
        event has been waited on the caller's current stream, and returned
        CUDA tensors are safe to use on it.
        """
        # Batches whose geometry changed (key includes the batch-shape params)
        # would otherwise linger forever; keys are (name, *params) tuples.
        for stale in [k for k in self._pending
                      if k[0] == key[0] and k != key]:
            self.drop(stale)
        pending = self._pending.pop(key, None)
        if pending is not None:
            result, evt = pending.result()
        else:
            result, evt = self.run_job(job)
        self._pending[key] = self._ensure_executor().submit(self.run_job, job)
        torch.cuda.current_stream().wait_event(evt)
        for t in result if isinstance(result, tuple) else (result,):
            if isinstance(t, torch.Tensor) and t.is_cuda:
                t.record_stream(torch.cuda.current_stream())
        return result

    def drop(self, key):
        # Discard a stale pending batch (e.g. when the batch geometry changes).
        pending = self._pending.pop(key, None)
        if pending is not None:
            pending.result()


_PREFETCHER = None


def get_prefetcher():
    global _PREFETCHER
    if _PREFETCHER is None:
        _PREFETCHER = StepPrefetcher()
    return _PREFETCHER
