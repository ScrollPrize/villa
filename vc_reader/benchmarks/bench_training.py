"""Training integration: six requests, two warmups, ten measured repetitions.

Reader, dict selection, collation, pinning, upload, resident augmentation and
persistent-worker epochs are timed separately. No timed validation/downloads.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import tempfile

from training_support import (TrainingDataset, augment_cpu_sample, cpu_batches, data_loader, load_native,
                                pipeline, select_arrays)
from training_inputs import build_workflow
from bench_paired import summary, timed


def measure(call, indices, synchronize=lambda: None):
    samples = []
    for repetition in range(12):
        for index in indices:
            synchronize()
            def run():
                output = call(index)
                synchronize()
                return output
            output, ms = timed(run)
            if repetition >= 2:
                samples.append(ms)
    return summary(samples)


def independent_pair(spec, requests, options):
    """No pairing/sampling helper participates in these requests."""
    import numpy as np
    from vc_reader.reader import MemoryCache
    options = {k: v for k, v in options.items() if k not in ('sampling_cache_bytes', 'plan_cache_bytes')}
    options['memory_cache'] = MemoryCache(64 << 20)
    stack = ExitStack()
    try:
        image = stack.enter_context(spec.image.open(**options))
        label = stack.enter_context(spec.reference.open(**options))
        masks = {c.name: stack.enter_context(c.source.open(**options)) for c in spec.reference.source.companions}
        def read(index):
            _, start, size = requests[index]
            region = tuple(slice(a, a+n) for a, n in zip(start, size))
            values = image[(slice(None), spec.layers, *region)] if len(image.shape) == 4 else image[(spec.layers, *region)][None]
            result = dict(image=values, label=label[region][None])
            result.update({'label_' + name: reader[region][None] for name, reader in masks.items()})
            return result
        yield read
    finally:
        stack.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow', choices=('paired', 'custom', 'ct', 'registered', 'transfer', 'render', 'project', 'local-render', 'local-project', 'published-target', 'published-transfer'))
    parser.add_argument('--metadata')
    parser.add_argument('--native-dir', required=True)
    parser.add_argument('--cache-dir', default='/tmp/vc-reader-paired-cache')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.workflow in ('project', 'published-transfer') and args.batch_size != 1:
        parser.error('projection workflows require --batch-size 1 for geometry-dependent crop shapes')
    load_native(args.native_dir)
    import numpy as np
    import torch
    torch.set_num_threads(args.torch_threads)
    from torch.utils.data import default_collate
    from vc_reader.reader import MemoryCache
    report = dict(command=sys.argv, platform=platform.platform(), python=sys.version,
                  protocol=dict(requests=6, warmups=2, repetitions=10, disk_cache='reused', os_cache_cleared=False),
                  metadata_sha256=hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest() if args.metadata else None,
                  workers=args.workers, batch_size=args.batch_size, threads_per_reader=1, torch_threads=args.torch_threads,
                  chunk_budget_per_worker=64 << 20, timings={})
    with tempfile.TemporaryDirectory(prefix='vc-training-bench-') as directory:
        built, discovery_ms = timed(lambda: build_workflow(args.workflow, Path(directory), args.metadata))
        specs, requests, arrays, dimensions, provenance = built
        report.update(inputs=provenance, requests=requests, discovery_ms=discovery_ms)
        dataset = TrainingDataset(specs, requests, arrays, seeds=range(100, 106), cache_dir=args.cache_dir,
                                  offline=args.offline, native_dir=args.native_dir)
        times = report['timings']
        with dataset, ExitStack() as stack:
            cache = MemoryCache(64 << 20)
            reader, report['open_ms'] = timed(lambda: stack.enter_context(specs[0].open(memory_cache=cache, **dataset.options)))
            def read(index):
                _, start, size = requests[index]
                return specs[0].read(reader, (start, size))
            first, first_ms = [], []
            for i in range(6):
                sample, ms = timed(lambda: read(i))
                first.append(sample)
                first_ms.append(ms)
            report['first_read_ms'] = first_ms
            report['sample_metadata'] = [sample.metadata for sample in first]
            report['hashes'] = [{key: hashlib.sha256(t.data.tobytes()).hexdigest() for key, t in sample.targets.items()} for sample in first]
            report['coverage'] = [{key: int(t.valid.sum()) for key, t in sample.targets.items()} for sample in first]
            report['reader_stats_before_warm'] = reader.stats()
            times['reader'] = measure(read, range(6))
            report['reader_stats_after_warm'] = reader.stats()
            samples = [dataset[i] for i in range(6)]
            for i in range(6):
                for name, value in select_arrays(first[i], arrays).items():
                    np.testing.assert_array_equal(samples[i][name], value)
            report['dataset_matches_reader'] = True
            if args.workflow in ('paired', 'custom'):
                from contextlib import contextmanager
                with contextmanager(independent_pair)(specs[0], requests, dataset.options) as direct:
                    for i in range(6):
                        for key, value in direct(i).items():
                            actual = first[i].image.data if key == 'image' else samples[i][key]
                            np.testing.assert_array_equal(actual, value)
                    times['independent_raw_arrays'] = measure(direct, range(6))
                report['independent_comparison'] = 'all six image/label/companion arrays equal independent Source.open reads'
                if args.workflow == 'paired':
                    report['first_ink_nonzero'] = int(np.count_nonzero(samples[0]['label']))
                    assert report['first_ink_nonzero'] == 9926
            elif args.workflow in ('ct', 'published-target'):
                raw = stack.enter_context(specs[0].reference.open(level=specs[0].level, cache_dir=args.cache_dir, offline=args.offline))
                for i, (_, start, size) in enumerate(requests):
                    np.testing.assert_array_equal(first[i].image.data[0], raw.read(start, size))
                if args.workflow == 'published-target':
                    target = specs[0].targets['features']
                    target_raw = stack.enter_context(target.open(cache_dir=args.cache_dir, offline=args.offline))
                    for i, (_, start, size) in enumerate(requests):
                        np.testing.assert_array_equal(first[i]['features'].data[0], target_raw.read(start, size))
                report['independent_comparison'] = 'all six arrays equal independent Source/Volume.open reads'
            elif args.workflow in ('render', 'project', 'published-transfer'):
                from training_reference import verify_surface_images
                report['independent_comparison'] = verify_surface_images(specs[0], requests, first,
                    cache_dir=args.cache_dir, offline=args.offline)
                if specs[0].mode == 'project':
                    from training_reference import verify_projected_labels
                    report['independent_labels'] = verify_projected_labels(specs[0], requests, reader,
                        cache_dir=args.cache_dir, offline=args.offline)
                if args.workflow == 'published-transfer':
                    from training_reference import verify_scan_transfer
                    report['independent_scan_transfer'] = verify_scan_transfer(specs[0], first,
                        cache_dir=args.cache_dir, offline=args.offline)
                if args.cuda:
                    device_reader = stack.enter_context(specs[0].open(backend='cuda', **dataset.options))
                    for i, (_, start, size) in enumerate(requests):
                        device_sample = specs[0].read(device_reader, (start, size))
                        for key, target in device_sample.targets.items():
                            expected = first[i][key]
                            actual = target.data.numpy()
                            if actual.dtype.kind in 'bui':
                                np.testing.assert_array_equal(actual, expected.data)
                            else:
                                np.testing.assert_allclose(actual, expected.data, atol=.03, rtol=2e-5)
                            np.testing.assert_array_equal(target.valid.numpy(), expected.valid)
                            for name, mask in target.masks.items():
                                np.testing.assert_array_equal(mask.numpy(), expected.masks[name])
                    report['cpu_cuda_readers'] = 'six actual geometry reads; labels, validity and companion masks exact'
            elif args.workflow in ('registered', 'transfer', 'local-render', 'local-project'):
                from training_reference import verify_analytic
                report['independent_comparison'] = verify_analytic(args.workflow, requests, first)
            times['dataset_read_and_selection'] = measure(dataset.__getitem__, range(6))
            times['selection_only'] = measure(lambda i: select_arrays(first[i], arrays), range(6))
            groups = [samples[i:i+args.batch_size] for i in range(0, 6, args.batch_size)]
            times['default_collation'] = measure(lambda i: default_collate(groups[i]), range(len(groups)))
            batches = [default_collate(group) for group in groups]
            cpu = pipeline(arrays, spatial_dims=dimensions, backend='cpu')
            targets = [{key: sample[key] for key in arrays} for sample in samples]
            times['cpu_augmentation_per_sample'] = measure(lambda i: augment_cpu_sample(cpu, targets[i], seed=dataset.seeds[i]), range(6))
            if args.cuda:
                from vc_reader.torch import CudaLoader
                gpu = pipeline(arrays, spatial_dims=dimensions, backend='cuda')
                times['pin_memory'] = measure(lambda i: {key: value.pin_memory() if key in arrays else value for key, value in batches[i].items()}, range(len(batches)))
                pinned = [{key: value.pin_memory() if key in arrays else value for key, value in batch.items()} for batch in batches]
                def upload(i):
                    return {key: value.to('cuda', non_blocking=True) if key in arrays else value for key, value in pinned[i].items()}
                sync = torch.cuda.synchronize
                times['host_to_device'] = measure(upload, range(len(batches)), sync)
                resident = [upload(i) for i in range(len(batches))]
                sync()
                times['cuda_augmentation_resident'] = measure(lambda i: gpu.batch(resident[i], seeds=batches[i]['seed']), range(len(batches)), sync)
                report['upload_bytes_per_six_requests'] = sum(batch[key].numel()*batch[key].element_size() for batch in batches for key in arrays)
                # CPU/CUDA values and categories are compared outside timing.
                for batch_index, batch in enumerate(resident):
                    result = gpu.batch(batch, seeds=batches[batch_index]['seed'])
                    for j, index in enumerate(batches[batch_index]['index'].tolist()):
                        expected = augment_cpu_sample(cpu, targets[index], seed=dataset.seeds[index])
                        for key in arrays:
                            actual = result[key][j].cpu().numpy()
                            if arrays[key].role in ('label', 'valid'):
                                np.testing.assert_array_equal(actual, expected[key])
                            else:
                                np.testing.assert_allclose(actual, expected[key], rtol=2e-5, atol=2e-5)
                report['cpu_cuda_comparison'] = 'all arrays; categorical/valid exact, continuous atol=rtol=2e-5'
            report['reader_stats'] = reader.stats()
            report['retained_chunk_bytes'] = cache.nbytes
            with data_loader(dataset, batch_size=args.batch_size, workers=args.workers, pin_memory=args.cuda) as loader:
                def drain(iterable):
                    for batch in iterable:
                        pass
                times['dataloader_first_epoch'] = timed(lambda: drain(loader))[1]
                times['dataloader_epoch'] = measure(lambda _: drain(loader), [0])
                augmented = CudaLoader(loader, gpu, seed_key='seed') if args.cuda else None
                times['dataloader_and_augmentation_epoch'] = measure(
                    lambda _: drain(augmented if augmented is not None else cpu_batches(loader, cpu)), [0])
        report['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({key: value['p50_ms'] if isinstance(value, dict) else value for key, value in times.items()}, indent=2))


if __name__ == '__main__':
    main()
