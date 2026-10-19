"""Optional producer/consumer timestamps and NVTX ranges for the real trainer."""
from contextlib import contextmanager
from functools import wraps
import os
import threading
import time

import torch

ENABLED = os.environ.get('FIBER_BENCH_TIMELINE') == '1'
LOCAL = threading.local()


@contextmanager
def span(name):
    if ENABLED:
        torch.cuda.nvtx.range_push(name)
        began, cpu = time.perf_counter(), time.thread_time()
    try:
        yield
    finally:
        if ENABLED:
            stages = getattr(LOCAL, 'stages', None)
            if stages is not None and name in ('heading.read', 'heading.tensor'):
                bucket = stages.setdefault(name, dict(seconds=0., cpu_seconds=0., calls=0))
                bucket['seconds'] += time.perf_counter()-began
                bucket['cpu_seconds'] += time.thread_time()-cpu
                bucket['calls'] += 1
            torch.cuda.nvtx.range_pop()


def wrap(name, function):
    @wraps(function)
    def call(*args, **kwargs):
        with span(name):
            return function(*args, **kwargs)
    return call


def install_worker(client):
    from vesuvius.neural_tracing.tracers.fibers.data.datasets import WeightedDatasets
    original = WeightedDatasets.__iter__

    def iterate(dataset):
        stream = original(dataset)
        worker = torch.utils.data.get_worker_info().id
        index = 0
        while True:
            client.trace_context = (worker, index)
            client.trace_seconds = 0.
            client.trace_requests = 0
            began, cpu = time.perf_counter(), time.thread_time()
            profile_path = os.environ.get('FIBER_BENCH_WORKER_PROFILE')
            if profile_path and index == 10:
                import cProfile
                profile = cProfile.Profile()
                batch = profile.runcall(next, stream)
                profile.dump_stats(f'{profile_path}/worker_{worker}.pstats')
            else:
                batch = next(stream)
            batch['_bench_worker_timing'] = dict(worker=worker, index=index, began=began,
                ready=time.perf_counter(), cpu_seconds=time.thread_time()-cpu,
                heading_seconds=client.trace_seconds, heading_requests=client.trace_requests)
            index += 1
            yield batch
    WeightedDatasets.__iter__ = iterate


def init_profile_worker(worker_id=None):
    from types import SimpleNamespace
    install_worker(SimpleNamespace())


class ProfileDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        if kwargs.get('worker_init_fn') is not None:
            raise ValueError('Profiling loader requires ownership of worker initialization')
        super().__init__(*args, worker_init_fn=init_profile_worker, **kwargs)


def install_main(train):
    from vc_reader import transforms
    from vc_reader.reader.cuda import CudaReader
    import vc_reader.sheet_normal as normal
    original = train.DecisionBatchPrefetch.collect

    def collect(prefetch):
        began = time.perf_counter()
        batches = original(prefetch)
        ready = time.perf_counter()
        for batch in batches:
            batch['_bench_collect_timing'] = dict(began=began, ready=ready)
        return batches
    train.DecisionBatchPrefetch.collect = collect
    transforms.Pipeline.batch = wrap('augmentation', transforms.Pipeline.batch)
    normal.sheet_normals = wrap('heading.tensor', normal.sheet_normals)
    original_read = CudaReader.read_affine_batch

    @wraps(original_read)
    def read(*args, **kwargs):
        name = 'heading.read' if threading.current_thread().name == 'cuda-sheet-heading' else 'images.read'
        with span(name):
            return original_read(*args, **kwargs)
    CudaReader.read_affine_batch = read
