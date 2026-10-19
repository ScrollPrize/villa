"""vc_reader input pipeline for the fiber-trainer benchmark.

Workers retain frame estimation, target construction and live-chain feedback.
They send final crop geometry instead of dense CT images. The training process
reads source chunks, interpolates and augments on CUDA before the model runs.
The trainer's shared remote prefetch service supplies the same disk cache.
"""
import importlib
import os
import sys
import threading

import numpy as np
import torch

from vesuvius.neural_tracing.tracers.fibers.data import observations
from vesuvius.neural_tracing.tracers.shared.crop_sampling import volume


def load_reader_package():
    site = os.environ.get('FIBER_BENCH_VC_SITE')
    if site:
        if site not in sys.path:
            sys.path.append(site)
        importlib.import_module('_editable_skbc_vc_reader')


def reader_spec(ct):
    if isinstance(ct, volume.RemoteChunkedArray):
        return dict(location=ct.remote_url, level=ct.level, cache_dir=ct.cache_dir,
                    offline=ct.cache_only, cache_writes=not ct.cache_only)
    return dict(location=ct.path)


def chunk_reader(ct):
    reader = getattr(ct, '_vc_reader', None)
    if reader is None:
        from vc_reader.reader import ZarrReader
        reader = ct._vc_reader = ZarrReader.open(**reader_spec(ct), threads=1,
                                               memory_cache=ct.cache_bytes)
    return reader


def read_block(ct, start, size):
    return chunk_reader(ct).read_region(start, size)


def read_nearest(ct, points):
    # Preserve the trainer's ties-to-even nearest-voxel rule.
    points = np.rint(points).astype(np.int64)
    return chunk_reader(ct).read_points(points)


def deferred_images(items, vol, crop, pool=None):
    return torch.empty((len(items), 1, 0, 0, 0), dtype=torch.float32)


def deferred_augmentation(*args, **kwargs):
    pass


def install_worker_reader(worker_id=None):
    load_reader_package()
    if os.environ.get('FIBER_BENCH_GEOMETRY') == '1':
        from fiber_geometry import install_worker
        install_worker()
    volume.ChunkedArray.read = read_block
    volume.ChunkedArray.sample_nearest = read_nearest
    observations.augment_image_pair = deferred_augmentation


class DeferredObservationBuilder(observations.IdentityObservationBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_images = deferred_images

    def __call__(self, items, vol):
        batch = super().__call__(items, vol)
        spec = reader_spec(vol.ct)
        # The worker's readiness gate covers final crops and frame context.
        # Offline mode raises on absent cache chunks instead of zero-filling.
        if 'cache_dir' in spec:
            spec.update(offline=True, cache_writes=False)
        batch['_vc_crop'] = dict(reader=spec, scale=vol.input_scale,
                                 crop=self.cfg.fine,
                                 photometric=[i.get('photometric') if self.augment else None for i in items],
                                 blur=[i.get('blur_sigma', 0.) for i in items],
                                 seeds=[int(i.get('identity_seed', 0))+1 for i in items])
        return batch


class ReaderDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        if kwargs.get('worker_init_fn') is not None:
            raise ValueError('Benchmark requires ownership of worker initialization')
        super().__init__(*args, worker_init_fn=install_worker_reader, **kwargs)


def crop_affines(positions, frames, crop, scale):
    # Keep incoming world coordinates in their original float64 precision until
    # vc_reader splits an integer origin from local float32 coordinates. Casting
    # global CT coordinates first would lose subvoxel precision on large scrolls.
    offset = np.array([crop.lateral_coords[0], crop.vertical_coords[0],
                       crop.forward_coords[0]])
    origins = (positions + np.einsum('bij,j->bi', frames, offset)) * scale
    axes = frames[:, ::-1, ::-1] * (crop.spacing * scale)
    return origins[:, ::-1], axes.astype(np.float32)


def augmentation_pipeline(backend='cuda'):
    from vc_reader import augment as a
    # Per-sample values are resolved by the common C++ batch executor.
    return a.Pipeline([
        a.Normalize(mode='fixed', scale=255.),
        a.Normalize(mode='zscore', nonzero=False, eps=1e-6),
        a.GaussianBlur(sigma=a.Param('blur'), truncate=4., full_width=False, boundary='reflect'),
        a.Contrast(factor=a.Param('contrast'), preserve_range=False),
        a.BrightnessAdd(shift=a.Param('brightness')),
        a.GaussianNoise(std=a.Param('noise')),
    ], targets={'image': a.Image()}, backend=backend)


def augmentation_params(request):
    params = dict(blur=[], contrast=[], brightness=[], noise=[])
    for photometric, sigma in zip(request['photometric'], request['blur']):
        contrast, brightness, noise = photometric if photometric is not None else (1., 0., 0.)
        params['blur'].append(sigma if photometric is not None else 0.)
        params['contrast'].append(contrast)
        params['brightness'].append(brightness * 8.)
        params['noise'].append(noise * 8.)
    return params


class CudaReaderPool:
    """Compare shared and consumer-specific caches at equal total capacity."""
    def __init__(self, *, shared=True):
        self.shared = shared
        self.readers = {}
        self.consumers = {}
        self.startup_stats = {}
        self.lock = threading.Lock()

    def get(self, spec, consumer):
        from vc_reader.reader import ZarrReader
        partition = 'shared' if self.shared else consumer
        key = (partition, tuple(sorted(spec.items())))
        with self.lock:
            if key not in self.readers:
                # Equal combined training budgets: 768 MiB shared, or 512 MiB
                # images plus 256 MiB headings. Startup is identical in all runs.
                budget = (768 if self.shared else 512 if consumer == 'images' else 256) << 20
                if not spec.get('offline', False):
                    budget = 256 << 20
                reader = ZarrReader.open(**spec, threads=4, memory_cache=budget)
                self.readers[key] = (reader, reader.cuda(cache_bytes=budget))
                self.consumers[key] = set()
            self.consumers[key].add(consumer)
            return self.readers[key][1]

    def finish_startup(self):
        # Manifest queries have completed before workers start. Release their
        # online readers; all training reads must obey the offline readiness gate.
        with self.lock:
            for key in list(self.readers):
                spec = dict(key[1])
                if 'cache_dir' in spec and not spec.get('offline', False):
                    cpu, gpu = self.readers.pop(key)
                    self.startup_stats[self.label(key)] = gpu.stats
                    self.consumers.pop(key)
                    gpu.close()
                    cpu.close()

    @staticmethod
    def label(key):
        return f'{key[0]}: {dict(key[1])}'

    def stats(self):
        with self.lock:
            readers = {self.label(key): gpu.stats for key, (_, gpu) in self.readers.items()}
            if any(s['cpu_routes'] for s in [*readers.values(), *self.startup_stats.values()]):
                raise RuntimeError('CUDA sampling unexpectedly routed through CPU')
            return dict(readers=readers, startup_readers=self.startup_stats,
                        consumers={self.label(k): sorted(v) for k, v in self.consumers.items()},
                        layout='shared' if self.shared else 'separate',
                        combined_cache_bytes_per_volume=768 << 20)


class CudaInputs:
    def __init__(self, *, augmentation="batch", cache="shared"):
        from vc_reader.cuda import is_available
        if not is_available():
            raise RuntimeError('The benchmark requires a CUDA-enabled vc_reader build')
        if augmentation not in ('sample', 'batch') or cache not in ('separate', 'shared'):
            raise ValueError('Invalid CUDA benchmark configuration')
        self.augmentation = augmentation
        self.pool = CudaReaderPool(shared=cache == 'shared')
        self.pipeline = augmentation_pipeline()
        self.stage_seconds = dict(read_sample=0., augmentation=0.)
        self.batch_calls = 0

    def materialize(self, batch):
        import time
        request = batch.pop('_vc_crop')
        reader = self.pool.get(request['reader'], 'images')
        crop = request['crop']
        shape = (crop.depth, crop.height or crop.width, crop.width)
        origins, axes = crop_affines(batch['crop_pos'].numpy(), batch['crop_frame'].numpy(),
                                    crop, request['scale'])
        images = torch.empty((len(origins), 1, *shape), device='cuda', dtype=torch.float32)
        began = time.perf_counter()
        # Bound the resident chunk working set; no CPU sampling fallback.
        for start in range(0, len(origins), 4):
            end = min(start + 4, len(origins))
            reader.read_affine_batch(shape, origins[start:end], axes[start:end], source='chunks',
                                     out=images[start:end, 0])
        self.stage_seconds['read_sample'] += time.perf_counter() - began
        began = time.perf_counter()
        params = augmentation_params(request)
        if self.augmentation == 'batch':
            output = self.pipeline.batch({'image': images}, seeds=request['seeds'], params=params)['image']
            self.batch_calls += 1
        else:
            output = torch.empty_like(images)
            for i in range(len(images)):
                result = self.pipeline.batch({'image': images[i:i+1]}, seeds=request['seeds'][i:i+1],
                                             params={k: v[i:i+1] for k, v in params.items()})['image']
                output[i:i+1].copy_(result)
                self.batch_calls += 1
        if not output.is_cuda:
            raise RuntimeError('Augmentation returned a host image')
        self.stage_seconds['augmentation'] += time.perf_counter() - began
        batch['x']['fine'] = output

    def stats(self):
        return dict(chunk_cache=self.pool.stats(), augmentation=self.augmentation,
                    image_output_device='cuda', sampling_source='chunks',
                    augmentation_batch_calls_including_warmup=self.batch_calls,
                    stage_seconds_including_warmup=self.stage_seconds)


def install(*, augmentation="batch", cache="shared"):
    from vesuvius.neural_tracing.tracers.fibers.train import train
    from vesuvius.neural_tracing.tracers.fibers.data import datasets
    install_worker_reader()
    train.IdentityObservationBuilder = DeferredObservationBuilder
    datasets.IdentityObservationBuilder = DeferredObservationBuilder
    torch.utils.data.DataLoader = ReaderDataLoader
    return CudaInputs(augmentation=augmentation, cache=cache)
