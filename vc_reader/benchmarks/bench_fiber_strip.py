"""Replay and loader benchmarks using the real sheet-strip trainer. See fiber_strip_results.md.

No training artifacts or datasets are modified. The capture uses the original
StripBuilder for sampling, frames, targets and augmentation parameters. Only its
large CT image is deferred, allowing identical requests to exercise each reader.
"""
from __future__ import annotations

import argparse
import hashlib
from functools import partial
import json
import multiprocessing as mp
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch

from vesuvius.neural_tracing.tracers.fibers.data.observations import augment_ct
from vesuvius.neural_tracing.tracers.fibers.frames.learned import frame_predictor
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.data import StripBuilder
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.model import SheetStripConfig
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.train import (
    RunConfig, DataOptions, TrainOptions, RuntimeOptions, build_data,
)
from vesuvius.neural_tracing.tracers.shared.crop_sampling.crops import empty_image_batch, scalar_crops
from vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization import normalize_ct, volume_key
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolume, CTVolumeSpec
from vc_reader.reader import ZarrReader
from vc_reader import augment as A
from vc_reader.cuda import asarray
from vesuvius.neural_tracing.tracers.shared.system import raise_open_file_limit


DEFAULT_CHECKPOINT = '/home/sean/.cache/huggingface/hub/models--scrollprize--fiber_strip_small_v1/snapshots/7bd163a22c839ff312879ae1916195d0c9729c0e/last.pt'
DEFAULT_ARRAY = '/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2'


def stats(values):
    a = np.asarray(values)
    return dict(mean=float(a.mean()), p50=float(np.median(a)), p95=float(np.percentile(a, 95)),
                min=float(a.min()), max=float(a.max()), n=len(a))


def request(items):
    return {k: np.array([i[k] for i in items]) for k in
            ('pos', 'frame', 'photometric', 'blur_sigma', 'identity_seed')}


def unpack(req):
    arrays = {k: np.asarray(v) for k, v in req.items()}
    return [{k: v[j] for k, v in arrays.items()} for j in range(len(arrays['pos']))]


def grids_for(items, crop, scale):
    """Convert trainer XYZ local axes (u,v,f) to reader ZYX output axes (f,v,u)."""
    first = np.array([crop.lateral_coords[0], crop.vertical_coords[0], crop.forward_coords[0]])
    shape = (crop.depth, crop.height or crop.width, crop.width)
    return [(shape, (np.asarray(i['pos'])*scale + np.asarray(i['frame'])*scale @ first)[::-1],
             (np.asarray(i['frame'])*scale*crop.spacing)[::-1, ::-1]) for i in items]


def pipeline(item, backend, noise=True, augment=True):
    transforms = [A.Normalize('zscore')]
    if not augment:
        return A.Pipeline(transforms, targets={'image': A.Image()}, compact=False, backend=backend)
    contrast, brightness, sigma = item['photometric']
    if item['blur_sigma'] > 0:
        sigma_blur = float(item['blur_sigma'])
        # SciPy truncate=4 specifies radius; native truncate specifies full width.
        radius = int(4.*sigma_blur+.5)
        transforms.append(A.GaussianBlur(sigma=sigma_blur, boundary='reflect',
                                         truncate=(2*radius+.5)/sigma_blur))
    transforms += [A.Contrast(factor=float(contrast), preserve_range=False),
                   A.BrightnessAdd(shift=float(brightness)*8.)]
    if noise and sigma > 0:
        transforms.append(A.GaussianNoise(std=float(sigma)*8.))
    return A.Pipeline(transforms, targets={'image': A.Image()}, compact=False, backend=backend)


class Images:
    def __init__(self, mode, spec, crop, threads=4, cache_gb=4., source='chunks'):
        self.mode, self.spec, self.crop, self.source = mode, spec, crop, source
        self.vol = CTVolume(spec, cache_bytes=512 << 20)
        self.reader = self.gpu = None
        if mode != 'legacy':
            self.reader = ZarrReader.open(Path(spec.ct_zarr)/str(spec.ct_level), threads=threads,
                                          memory_cache=384 << 20)
            if mode == 'cuda':
                self.gpu = self.reader.cuda(cache_bytes=int(cache_gb*(1 << 30)))

    def __call__(self, req, *, augment=True, noise=True):
        items, c = unpack(req), self.crop
        t0 = time.perf_counter()
        if self.mode == 'legacy':
            x = empty_image_batch((len(items), 1, c.depth, c.height or c.width, c.width))
            scalar_crops(items, self.vol, c, out=x.numpy(), normalize=False)
        elif self.mode == 'cpu':
            x = empty_image_batch((len(items), 1, c.depth, c.height or c.width, c.width))
            self.reader.read_affine_many(grids_for(items, c, self.vol.input_scale),
                                        out=[row[0].numpy() for row in x])
        else:
            arrays = self.gpu.read_affine_many(grids_for(items, c, self.vol.input_scale), source=self.source)
        t1 = time.perf_counter()
        device_results = []
        for j, item in enumerate(items):
            if self.mode == 'legacy':
                normalize_ct(x[j, 0].numpy(), self.spec.ct_normalization)
                if augment:
                    params = np.array(item['photometric'], copy=True)
                    if not noise:
                        params[2] = 0.
                    augment_ct(x[j, 0].numpy(), params, np.random.default_rng(int(item['identity_seed'])+1),
                               blur_sigma=float(item['blur_sigma']))
            else:
                p = pipeline(item, 'cuda' if self.mode == 'cuda' else 'cpu', noise, augment)
                a = arrays[j].reshape(1, *arrays[j].shape) if self.mode == 'cuda' else x[j]
                result = p({'image': a}, seed=int(item['identity_seed'])+1, inplace=True)['image']
                if self.mode == 'cuda':
                    device_results.append(torch.from_dlpack(result))
                else:
                    x[j].copy_(result)
        if self.mode == 'cuda':
            x = torch.stack(device_results)
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        self.last_ms = dict(read=(t1-t0)*1000, normalize_augment=(t2-t1)*1000)
        return x

    def close(self):
        if self.gpu is not None:
            self.gpu.close()
        if self.reader is not None:
            self.reader.close()


class DeferredBuilder(StripBuilder):
    """Reuse the entire trainer builder; defer only image work to the selected backend.

    Original augmentation executes on a tiny disposable placeholder. This keeps
    geometry/label logic and its RNG consumption intact without copying that code.
    """
    execution = 'capture'
    reader_threads = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_images = self.capture
        self.images_backend = None

    def capture(self, items, vol, crop, pool=None):
        self.captured = request(items)
        return torch.zeros((len(items), 1, 1, 1, 1), dtype=torch.float32)

    def __call__(self, items, vol):
        batch = super().__call__(items, vol)
        batch['_request'] = self.captured
        if self.execution == 'cpu':
            if self.images_backend is None:
                self.images_backend = Images('cpu', vol.spec, self.cfg.fine, self.reader_threads)
            batch['x']['fine'] = self.images_backend(self.captured)
        return batch


def setup(args):
    # PatchStore uses tifffile's writable default despite only reading patches.
    # Keep benchmark dataset access read-only without changing the trainer.
    import tifffile
    tifffile.memmap = partial(tifffile.memmap, mode='r')
    raise_open_file_limit()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    values = ck['run_config']
    run = RunConfig(**{k: v for k, v in values.items() if k not in ('data', 'training', 'runtime')},
                    data=DataOptions(**values['data']), training=TrainOptions(**values['training']),
                    runtime=RuntimeOptions(**values['runtime']))
    if args.index:
        run.data.index = args.index
    if args.afv:
        run.data.afv = args.afv
    cfg = SheetStripConfig(**ck['model_cfg'])
    spec = CTVolumeSpec.from_dict(ck['vol_spec'])
    spec.ct_zarr, spec.ct_level = str(Path(args.array).parent), int(Path(args.array).name)
    spec.ct_normalization = dict(spec.ct_normalization, volume=volume_key(spec))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    return ck, run, cfg, spec


def make_loader(args, ck, run, cfg, spec, mode):
    frame_predictor(cfg)
    DeferredBuilder.execution = 'cpu' if mode == 'cpu' else 'capture'
    DeferredBuilder.reader_threads = 1
    loader, index = build_data(run, cfg, spec, ck, 'train', augment=True, seed=args.seed,
                               workers=args.workers, builder_type=StripBuilder if mode == 'legacy' else DeferredBuilder)
    if args.workers:
        # Explicit fork inherits run_native's selected extensions. No worker uses CUDA.
        if 'fork' not in mp.get_all_start_methods():
            raise RuntimeError('This explicit-native-build worker benchmark requires fork support')
        loader.multiprocessing_context = 'fork'
    return loader


def capture(args, ck, run, cfg, spec):
    loader = make_loader(args, ck, run, cfg, spec, 'capture')
    batches = []
    it = iter(loader)
    start = time.perf_counter()
    for j in range(args.batches):
        batch = next(it)
        batches.append({k: np.asarray(v) for k, v in batch['_request'].items()})
        print(f'captured {j+1}/{args.batches} batches ({time.perf_counter()-start:.1f}s)', flush=True)
    torch.save(dict(requests=batches, crop=cfg.fine, spec=spec.to_dict(), run=run.to_dict(), seed=args.seed), args.output)


def validate(args, cfg, spec, requests):
    # Full-size crops, same geometry/photometric draws. Noise is disabled for paired
    # comparisons because NumPy and native generators deliberately differ.
    req = {k: v[:4] for k, v in requests[0].items()}
    assert np.any(req['blur_sigma'] > 0) and np.any(req['blur_sigma'] == 0), 'capture needs both blur cases'
    outputs, report = {}, {}
    # Explicitly exercise native zero-preserving population z-score semantics.
    values = np.array([0., 2., 4., 6.], np.float32).reshape(1, 1, 2, 2)
    expected_normalized = np.array([0., -np.sqrt(1.5), 0., np.sqrt(1.5)], np.float32).reshape(values.shape)
    for device in ('cpu', 'cuda'):
        norm = A.Pipeline([A.Normalize('zscore')], targets={'image': A.Image()}, backend=device, compact=False)
        for data, expected_result in ((values, expected_normalized), (np.zeros_like(values), np.zeros_like(values)),
                                      (np.full_like(values, 5.), np.zeros_like(values))):
            result = norm({'image': asarray(data) if device == 'cuda' else data}, seed=0)['image']
            np.testing.assert_allclose(result.numpy() if device == 'cuda' else result, expected_result, atol=1e-6)
    for mode in ('legacy', 'cpu', 'cuda'):
        backend = Images(mode, spec, cfg.fine, args.threads, args.cache_gb, args.source)
        outputs[mode] = backend(req, noise=False).cpu().numpy().copy()
        noisy = backend(req).cpu().numpy()
        residual = noisy-outputs[mode]
        means = residual.mean(axis=(1, 2, 3, 4))
        stds = residual.std(axis=(1, 2, 3, 4))
        expected = np.asarray(req['photometric'])[:, 2]*8.
        np.testing.assert_allclose(stds, expected, rtol=.01, atol=1e-4)
        np.testing.assert_allclose(means, 0., atol=.002)
        backend.close()
        assert np.isfinite(outputs[mode]).all()
        report[mode] = dict(noise_mean=means.tolist(), noise_std=stds.tolist(), noise_expected_std=expected.tolist())
        if mode != 'legacy':
            delta = np.abs(outputs[mode]-outputs['legacy'])
            report[mode].update(mean_abs=float(delta.mean()), max_abs=float(delta.max()),
                                rmse=float(np.sqrt(np.mean(delta**2))))
            # Native normalization intentionally differs from the trainer on zero
            # voxels and constant channels. Report that difference; compare the
            # native CPU and CUDA implementations directly for correctness.
            if mode == 'cuda':
                np.testing.assert_allclose(outputs[mode], outputs['cpu'], atol=.001, rtol=.001)
    return report


def replay(args, cfg, spec):
    capture_data = torch.load(args.capture, weights_only=False)
    assert capture_data['crop'] == cfg.fine and capture_data['spec'] == spec.to_dict()
    requests = capture_data['requests']
    report = dict(validation=validate(args, cfg, spec, requests) if args.validate else None, rows={})
    for mode in args.modes.split(','):
        backend = Images(mode, spec, cfg.fine, args.threads, args.cache_gb, args.source)
        timings, stages, uploaded = [], [], []
        for j in range(-args.warmup, args.batches):
            req = requests[j % len(requests)]
            if backend.gpu is not None and args.clear_gpu:
                backend.gpu.clear_cache()
            before = backend.gpu.stats['uploaded_bytes'] if backend.gpu is not None else 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = backend(req)
            t1 = time.perf_counter()
            if mode != 'cuda':
                x = x.pin_memory().to('cuda', non_blocking=True)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter()-t0)*1000
            if j >= 0:
                timings.append(elapsed)
                stages.append(dict(backend.last_ms, pin_upload=(time.perf_counter()-t1)*1000))
                uploaded.append(backend.gpu.stats['uploaded_bytes']-before if backend.gpu is not None else x.numel()*4)
            del x
            print(f'{mode} batch {j}: {elapsed:.1f} ms', flush=True)
        report['rows'][mode] = dict(ms=stats(timings), crops_s=len(requests[0]['pos'])*1000/np.mean(timings),
                                    stages_ms={k: stats([r[k] for r in stages]) for k in stages[0]},
                                    upload_bytes=stats(uploaded), samples_ms=timings)
        backend.close()
    return report


def loader_bench(args, ck, run, cfg, spec):
    mode = args.modes
    assert mode in ('legacy', 'cpu', 'cuda'), 'one loader mode per fresh process'
    loader = make_loader(args, ck, run, cfg, spec, mode)
    it = iter(loader)  # start CPU workers before creating CUDA reader/context
    backend = Images(mode, spec, cfg.fine, args.threads, args.cache_gb, args.source) if mode == 'cuda' else None
    timings, waiting, requests_digest = [], [], hashlib.sha256()
    first = time.perf_counter()
    for j in range(-args.warmup, args.batches):
        t0 = time.perf_counter()
        batch = next(it)
        t1 = time.perf_counter()
        if backend is not None:
            batch['x']['fine'] = backend(batch['_request'])
        fine = batch['x']['fine'].to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter()-t0)*1000
        if j >= 0:
            timings.append(elapsed)
            waiting.append((t1-t0)*1000)
            # Frame/position digest works for original and deferred builders alike.
            for k in ('crop_pos', 'crop_frame'):
                requests_digest.update(batch[k].numpy().tobytes())
        del batch, fine
        print(f'{mode} batch {j}: {elapsed:.1f} ms (wait {(t1-t0)*1000:.1f})', flush=True)
    result = dict(mode=mode, ms=stats(timings), wait_ms=stats(waiting),
                  crops_s=run.training.batch*1000/np.mean(timings), samples_ms=timings,
                  total_s=time.perf_counter()-first, geometry_sha256=requests_digest.hexdigest())
    if backend is not None:
        result['cuda_stats'] = backend.gpu.stats
        backend.close()
    if args.workers:
        it._shutdown_workers()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('capture', 'replay', 'loader'))
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    ap.add_argument('--array', default=DEFAULT_ARRAY)
    ap.add_argument('--index')
    ap.add_argument('--afv')
    ap.add_argument('--capture', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--modes', default='legacy,cpu,cuda')
    ap.add_argument('--batches', type=int, default=28)
    ap.add_argument('--warmup', type=int, default=14)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cache-gb', type=float, default=4.)
    ap.add_argument('--source', choices=('chunks', 'cpu', 'auto'), default='chunks')
    ap.add_argument('--clear-gpu', action='store_true')
    ap.add_argument('--validate', action='store_true')
    args = ap.parse_args()
    if args.batches < 1 or args.warmup < 0 or args.workers < 0 or args.threads < 1:
        ap.error('invalid counts')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ck, run, cfg, spec = setup(args)
    if args.action == 'capture':
        capture(args, ck, run, cfg, spec)
        return
    if args.action == 'replay' and args.capture is None:
        ap.error('replay requires --capture')
    result = replay(args, cfg, spec) if args.action == 'replay' else loader_bench(args, ck, run, cfg, spec)
    from vc_reader import _native, _augment
    report = dict(result=result, command=sys.argv, pid=os.getpid(), torch=torch.__version__,
                  normalization={'legacy': 'trainer_all_values_zscore', 'cpu': 'vc_reader_zero_preserving_zscore',
                                 'cuda': 'vc_reader_zero_preserving_zscore'},
                  python=sys.version, platform=platform.platform(),
                  gpu=torch.cuda.get_device_name(), native=_native.__file__, augment=_augment.__file__,
                  binary_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in (_native.__file__, _augment.__file__)},
                  run=run.to_dict(), spec=spec.to_dict(), crop=cfg.fine.__dict__, options=vars(args))
    args.output.write_text(json.dumps(report, indent=2, default=str)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
