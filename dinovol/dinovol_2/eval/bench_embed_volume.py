"""Benchmark a small real embedding region without modifying a production output.

Run with --help. Uses the volume CLI's model, normalization and storage settings,
but creates disposable output groups beneath --work-dir. Compilation warmup is
excluded from timings. Results and optional CPU/CUDA profiles remain in --report.
Each slab group is flushed before the next; inference_s includes those flushes,
while writer_drain_s measures only the final close. total_s covers the full run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from dinovol_2.eval import embed_volume as current


def summary(values):
    return {"mean": float(np.mean(values)), "p50": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--volume', required=True)
    parser.add_argument('--level', type=int, default=2)
    parser.add_argument('--normalization-stats', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, default=Path('/tmp'))
    parser.add_argument('--module-file', type=Path, help='benchmark a saved baseline module')
    parser.add_argument('--sampler-file', type=Path, help='pair a saved sampler with a baseline module')
    parser.add_argument('--z-start', type=int, default=1536)
    parser.add_argument('--depth', type=int, default=192)
    parser.add_argument('--slab-depth', type=int, help='benchmark separate slabs within --depth')
    parser.add_argument('--slab-group-size', type=int, default=1)
    parser.add_argument('--column-block-size', type=int, default=16)
    parser.add_argument('--y-start', type=int, default=1920)
    parser.add_argument('--x-start', type=int, default=1920)
    parser.add_argument('--cells', type=int, default=1)
    parser.add_argument('--x-cells', type=int)
    parser.add_argument('--y-cells', type=int)
    parser.add_argument('--torch-threads', type=int, default=24)
    parser.add_argument('--reader-threads', type=int, default=16)
    parser.add_argument('--writers', type=int, default=8)
    parser.add_argument('--point-batch', type=int, default=32768)
    parser.add_argument('--point-sampler', choices=('auto', 'torch'), default='auto')
    parser.add_argument('--async-output', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--strip-device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--read', choices=('band', 'column'), default='band')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--transformer-precision', choices=('bf16', 'fp8'), default='bf16')
    parser.add_argument('--keep-last', action='store_true', help='keep last output for numeric comparison')
    args = parser.parse_args()
    ev = current
    if args.module_file:
        spec = importlib.util.spec_from_file_location('embed_volume_baseline', args.module_file)
        ev = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ev
        spec.loader.exec_module(ev)
    if args.sampler_file:
        spec = importlib.util.spec_from_file_location('sampler_baseline', args.sampler_file)
        sampler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sampler)
        ev.sample_grid_points = sampler.sample_grid_points
    torch.set_num_threads(args.torch_threads)
    torch.backends.cudnn.benchmark = False
    args.report.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    saved = json.loads(Path(args.normalization_stats).read_text())
    statistics = saved.get('normalization', saved).get('statistics', saved)
    source, cache = ev.resolve_volume_source(args.volume, args.level, None)
    reader = ev.open_volume(source, level=args.level, cache_dir=cache, cache_writes=False,
                            offline=True, threads=args.reader_threads)
    loaded = ev.load_backbone_from_checkpoint(args.checkpoint)
    tile, patch = loaded.global_input_size[0], loaded.patch_size[0]
    overlap = max(patch, tile // 6 // patch * patch)
    lattice = [ev.AxisLattice(tile, overlap, overlap)] * 3
    sampler_options = {} if args.module_file else dict(point_sampler=args.point_sampler,
                                                      transformer_precision=args.transformer_precision,
                                                      column_block_size=args.column_block_size)
    embedder = ev.RegionEmbedder(reader, loaded,
                                 transforms=ev.global_normalization(loaded.normalization_scheme, statistics,
                                                                   loaded.intensity_properties),
                                 outputs=ev.OUTPUT_NAMES, dense_stride=1, lattice=lattice,
                                 dtype='int8', dtypes=ev.CLI_DTYPES['int8'], codec='vc-delta3d',
                                 compile_model=args.compile, strip_device=args.strip_device,
                                 point_chunk_size=args.point_batch, **sampler_options)
    z_range = (args.z_start, args.z_start + args.depth)
    y_range = (args.y_start, args.y_start + lattice[1].stride * (args.y_cells or args.cells))
    x_range = (args.x_start, args.x_start + lattice[2].stride * (args.x_cells or args.cells))
    slab_depth = args.slab_depth or args.depth
    if slab_depth <= 0 or slab_depth % lattice[0].stride or args.slab_group_size < 1:
        parser.error('slab depth must be a positive lattice-stride multiple; slab group size must be positive')
    slabs = current.plan_slabs(z_range, slab_depth)
    report = dict(command=[sys.executable, *sys.argv], torch=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(), settings=vars(args).copy(), warmup=[], runs=[])
    report['settings'] = {k: str(v) if isinstance(v, Path) else v for k, v in report['settings'].items()}

    def run(path):
        writer_options = {} if args.module_file else dict(async_transfers=args.async_output)
        sink = ev.ZarrGroupWriter(path, attrs={}, workers=args.writers, **writer_options)
        embedder.create_arrays(sink, z_range, z_cell=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            stats = {}
            for group in current.pending_slab_groups(slabs, {}, args.slab_group_size):
                grouped_range = (slabs[group[0]][0], slabs[group[-1]][1])
                row = embedder.run(sink, z_range=grouped_range, array_z_range=z_range,
                                   y_range=y_range, x_range=x_range, progress=False, read=args.read)
                sink.flush()
                for key, value in row.items():
                    stats[key] = stats.get(key, 0) + value
            torch.cuda.synchronize()
            inference_done = time.perf_counter()
        finally:
            sink.close()
        end = time.perf_counter()
        return dict(total_s=end-start, inference_s=inference_done-start,
                    writer_drain_s=end-inference_done, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                    **stats)

    try:
        for i in range(args.warmup + args.repeats):
            phase = 'warmup' if i < args.warmup else 'runs'
            with tempfile.TemporaryDirectory(prefix='dinovol-bench-', dir=args.work_dir) as tmp:
                output = Path(tmp) / 'out.zarr'
                row = run(output)
                if args.keep_last and i == args.warmup + args.repeats - 1:
                    # Copy outside the timed interval so all repeats use the same filesystem.
                    shutil.copytree(output, args.report / 'last.zarr')
            report[phase].append(row)
            print(phase, i, json.dumps(row), flush=True)
            (args.report / 'results.json').write_text(json.dumps(report, indent=2))
        report['summary'] = {key: summary([r[key] for r in report['runs']])
                             for key in ('total_s', 'inference_s', 'writer_drain_s')}
        (args.report / 'results.json').write_text(json.dumps(report, indent=2))
        if args.profile:
            with tempfile.TemporaryDirectory(prefix='dinovol-bench-', dir=args.work_dir) as tmp:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    run(Path(tmp) / 'out.zarr')
            prof.export_chrome_trace(str(args.report / 'trace.json'))
            (args.report / 'cpu.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=45))
            (args.report / 'gpu.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=45))
        print('SUMMARY', json.dumps(report['summary']), flush=True)
    finally:
        reader.close()


if __name__ == '__main__':
    main()
