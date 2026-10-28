"""Worker-side vc_reader CPU inputs, headings and geometry for the fiber trainer."""
import os
import time
from functools import partial

import numpy as np
import torch

from fiber_cuda_pipeline import (
    DeferredObservationBuilder, augmentation_params, augmentation_pipeline,
    crop_affines, install_worker_reader, reader_spec,
)
from fiber_tensor_heading import Client, HeadingCalculator, install_heading

STATE = None


class CpuReaderPool:
    def __init__(self, cache_bytes=384 << 20):
        self.cache_bytes = cache_bytes
        self.readers = {}

    def get(self, spec, consumer):
        from vc_reader.reader import ZarrReader
        key = tuple(sorted(spec.items()))
        if key not in self.readers:
            self.readers[key] = ZarrReader.open(**spec, threads=1, memory_cache=self.cache_bytes)
        return self.readers[key]


class LocalHeadingClient(Client):
    def __init__(self, calculator):
        super().__init__(None)
        self.calculator = calculator

    def exchange(self, request):
        began = time.perf_counter()
        result = self.calculator.calculate(request)
        counters = self.calculator.counters
        counters['requests'] += 1
        counters['queries'] += len(result)
        counters['seed_queries' if request['mode'] == 'seed' else 'frame_queries'] += len(result)
        counters['seconds'] += time.perf_counter()-began
        return True, result


class CpuInputs:
    def __init__(self):
        self.pid = os.getpid()
        self.pool = CpuReaderPool()
        self.heading = HeadingCalculator(self.pool, device='cpu')
        self.pipeline = augmentation_pipeline('cpu')
        self.sampling = dict(method='adaptive dense/direct')
        self.geometry = None
        self.stage_seconds = dict(read_sample=0., augmentation=0.)
        if os.environ.get('FIBER_BENCH_GEOMETRY') == '1':
            from fiber_geometry import Geometry
            self.geometry = Geometry(backend='cpu')
            self.geometry.placement = 'workers'

    def materialize(self, batch):
        from vesuvius.neural_tracing.tracers.shared.crop_sampling.crops import empty_image_batch
        request = batch.pop('_vc_crop')
        crop = request['crop']
        shape = (crop.depth, crop.height or crop.width, crop.width)
        origins, axes = crop_affines(batch['crop_pos'].numpy(), batch['crop_frame'].numpy(),
                                    crop, request['scale'])
        reader = self.pool.get(request['reader'], 'images')
        images = empty_image_batch((len(origins), 1, *shape))
        arrays = images.numpy()
        began = time.perf_counter()
        grids = [(shape, origin, axis) for origin, axis in zip(origins, axes)]
        reader.read_affine_many(grids, out=[row[0] for row in arrays])
        self.stage_seconds['read_sample'] += time.perf_counter()-began
        params = augmentation_params(request)
        began = time.perf_counter()
        for i, row in enumerate(arrays):
            result = self.pipeline({'image': row}, seed=request['seeds'][i], inplace=True,
                                   params={k: v[i] for k, v in params.items()})['image']
            if not np.shares_memory(result, row):
                np.copyto(row, result)
        self.stage_seconds['augmentation'] += time.perf_counter()-began
        batch['x']['fine'] = images
        if self.geometry is not None:
            self.geometry.materialize(batch)
        batch['_vc_cpu_stats'] = dict(pid=self.pid, image_seconds=dict(self.stage_seconds),
            sampling=self.sampling,
            heading=dict(self.heading.counters), geometry=self.geometry.stats() if self.geometry else None,
            readers={str(key): reader.stats() for key, reader in self.pool.readers.items()})


def state():
    global STATE
    if STATE is None or STATE.pid != os.getpid():
        STATE = CpuInputs()
    return STATE


def worker_chunk_reader(ct):
    return state().pool.get(reader_spec(ct), 'context')


def init_worker(worker_id=None, *, tensor_heading=True):
    from fiber_cuda_pipeline import volume
    install_worker_reader(worker_id)
    # Context, headings and final images share each worker's decoded chunks.
    volume.ChunkedArray.read = lambda ct, start, size: worker_chunk_reader(ct).read_region(start, size)
    volume.ChunkedArray.sample_nearest = lambda ct, points: worker_chunk_reader(ct).read_points(
        np.rint(points).astype(np.int64))
    if tensor_heading:
        install_heading(client=LocalHeadingClient(state().heading))
    if worker_id is not None and os.environ.get('FIBER_BENCH_WORKER_PROFILE'):
        from fiber_timeline import init_profile_worker
        init_profile_worker(worker_id)


class CpuObservationBuilder(DeferredObservationBuilder):
    def __call__(self, items, vol):
        batch = super().__call__(items, vol)
        state().materialize(batch)
        return batch


class CpuDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        if kwargs.get('worker_init_fn') is not None:
            raise ValueError('Benchmark requires ownership of worker initialization')
        super().__init__(*args, worker_init_fn=partial(init_worker,
            tensor_heading=os.environ.get('FIBER_BENCH_CPU_HEADING') == '1'), **kwargs)


class CpuBackend:
    def __init__(self):
        self.workers = {}

    def materialize(self, batch):
        if batch['x']['fine'].is_cuda or batch['foreign'].is_cuda:
            raise RuntimeError('CPU worker pipeline returned CUDA data')
        stats = batch.pop('_vc_cpu_stats')
        self.workers[stats['pid']] = stats

    def stats(self):
        return dict(backend='cpu', placement='workers', workers=self.workers,
                    image_output_device='cpu', decoded_cache_bytes_per_worker_volume=384 << 20,
                    augmentation='per-sample C++ CPU pipeline',
                    heading='C++ CPU structure tensor', geometry='C++ CPU geometry')


def install(*, tensor_heading=True):
    from vesuvius.neural_tracing.tracers.fibers.train import train
    from vesuvius.neural_tracing.tracers.fibers.data import datasets
    os.environ['FIBER_BENCH_CPU_HEADING'] = '1' if tensor_heading else '0'
    init_worker(tensor_heading=tensor_heading)
    train.IdentityObservationBuilder = datasets.IdentityObservationBuilder = CpuObservationBuilder
    torch.utils.data.DataLoader = CpuDataLoader
    return CpuBackend()
