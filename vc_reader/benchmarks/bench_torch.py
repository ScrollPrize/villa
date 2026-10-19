"""GPU-ready PyTorch loader batches: serial, public batch and prefetch.

No installs or downloads. --array selects real data; otherwise a seeded local
Zarr is generated. See benchmarks/README.md for commands and timing scope.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from extension_build import load_extensions
from bench_pipeline import environment, heavy_transforms, positive, summary, synthetic_array

# Spawned DataLoader workers must import the same extensions as the parent.
if os.environ.get("VC_READER_BUILD_DIR"):
    load_extensions(Path(os.environ["VC_READER_BUILD_DIR"]))


def pipeline():
    from vc_reader import transforms as A
    return A.Pipeline([A.Normalize('minmax'), *heavy_transforms()],
                      backend='cuda', compact=False)


class Crops(Dataset):
    def __init__(self, args, location, serial=False):
        self.args, self.location, self.serial = args, location, serial
        self.reader = self.transform = None

    def __len__(self):
        return self.args.batches * self.args.batch_size

    def __getitem__(self, index):
        from vc_reader import transforms as A, cuda
        from vc_reader.reader import ZarrReader
        if self.reader is None:
            self.reader = ZarrReader.open(self.location, level=self.args.level, threads=self.args.reader_threads,
                                          memory_cache=256 << 20)
        shape = self.args.shape
        center = tuple((s - n) // 2 for s, n in zip(self.reader.shape, shape))
        origin = tuple(c + ((index * (a * 10 + 7)) % 65 - 32) for a, c in enumerate(center))
        image = self.reader.read_region(origin, shape).reshape(1, *shape)
        seed = A.seed(self.args.seed, index)
        if self.serial:
            if self.transform is None:
                self.transform = pipeline()
            output = self.transform({'image': cuda.asarray(image).astype(np.float32)}, seed=seed)['image']
            image = torch.from_dlpack(output)
        return {'image': image, 'aug_seed': np.asarray(seed, dtype=np.uint64)}


class PublicBatch:
    def __init__(self, loader, transform):
        self.loader, self.transform = loader, transform

    def __iter__(self):
        for batch in self.loader:
            yield self.transform.batch(batch, seeds=batch['aug_seed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--array')
    parser.add_argument('--level')
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--modes', nargs='+', choices=['serial', 'batch', 'prefetch'],
                        default=['serial', 'batch', 'prefetch'])
    parser.add_argument('--shape', type=positive, nargs=3, default=(128, 128, 128))
    parser.add_argument('--batches', type=positive, default=64)
    parser.add_argument('--batch-size', type=positive, default=4)
    parser.add_argument('--workers', type=positive, default=4)
    parser.add_argument('--reader-threads', type=positive, default=4)
    parser.add_argument('--reps', type=positive, default=3)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.build_dir:
        args.build_dir = args.build_dir.resolve()
        os.environ['VC_READER_BUILD_DIR'] = str(args.build_dir)
        load_extensions(args.build_dir)
    from vc_reader.torch import CudaLoader
    torch.set_num_threads(1)
    torch.cuda.init()
    # Reuse the existing benchmark environment/provenance recorder.
    args.backend = 'cuda'
    report = dict(command=[sys.executable, *sys.argv], config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  environment=environment(args, transform=True), torch=torch.__version__, results={},
                  protocol='one validation/warmup epoch per mode; rotated mode order per repetition; warm caches; '
                           'GPU-ready batches; includes reads, transfers, casts, transform, collation and output disposal; '
                           'excludes startup/validation and model execution')
    with ExitStack() as stack:
        location = args.array
        if location is None:
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='vc-torch-')))
            location = synthetic_array(directory / 'input.zarr', args.shape, args.seed)
        report['input'] = location if args.array else 'temporary seeded uniform uint8 Zarr'
        cpu = DataLoader(Crops(args, location), batch_size=args.batch_size, num_workers=args.workers,
                         multiprocessing_context='spawn', persistent_workers=True, pin_memory=True)
        transform = pipeline()
        report['transforms'] = [repr(t) for t in transform.transforms]
        loaders = dict(serial=DataLoader(Crops(args, location, serial=True), batch_size=args.batch_size),
                       batch=PublicBatch(cpu, transform),
                       prefetch=CudaLoader(cpu, transform, seed_key='aug_seed'))
        hashes = {}
        for mode in args.modes:
            hashes[mode] = []
            for batch in loaders[mode]:
                values = batch['image'].cpu().numpy()
                assert values.shape == (args.batch_size, 1, *args.shape) and np.isfinite(values).all()
                hashes[mode].extend(hashlib.sha256(x.tobytes()).hexdigest() for x in values)
            if hashes[mode] != hashes[args.modes[0]]:
                raise AssertionError(f'{mode} outputs differ from {args.modes[0]}')
            report['results'][mode] = dict(raw_batch_ms=[], epoch_seconds=[])
        report['validation'] = dict(crops=len(hashes[args.modes[0]]), comparison='byte-identical across selected modes',
                                    hashes=hashes[args.modes[0]])
        for rep in range(args.reps):
            order = args.modes[rep % len(args.modes):] + args.modes[:rep % len(args.modes)]
            for mode in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                iterator = iter(loaders[mode])
                for _ in range(args.batches):
                    t0 = time.perf_counter()
                    batch = next(iterator)
                    torch.cuda.current_stream().synchronize()
                    report['results'][mode]['raw_batch_ms'].append((time.perf_counter()-t0)*1000)
                    del batch
                try:
                    next(iterator)
                    raise AssertionError('extra batch')
                except StopIteration:
                    pass
                torch.cuda.synchronize()
                report['results'][mode]['epoch_seconds'].append(time.perf_counter()-start)
        for mode, result in report['results'].items():
            result['batch_ms'] = summary(result['raw_batch_ms'])
            result['crops_per_second'] = args.batches*args.batch_size*args.reps / sum(result['epoch_seconds'])
            print(f"{mode}: {result['batch_ms']}, {result['crops_per_second']:.2f} crops/s", flush=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
