"""Synthetic reproducible Stage 2b benchmark; no data downloads or training.

AGENTS_AGENT_MODE=1 .venv/bin/python tests/benchmark_pins_stage2b.py --iterations 20
Add --device cuda for synchronized CUDA timings and peak allocated VRAM.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_pins_stage2b import scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = scene(args.device)
    model.cfg['model_pin_rebin_interval'] = 8
    theta = torch.linspace(.3, 4., 16, device=args.device)
    z = torch.linspace(30., 120., 16, device=args.device)
    ids = torch.arange(16, device=args.device).repeat_interleave(32)
    radii = torch.linspace(20., 150., 512, device=args.device)
    points = torch.stack([z[ids], theta.sin()[ids] * radii, theta.cos()[ids] * radii], dim=-1)
    report = {'device': args.device, 'torch': torch.__version__, 'dtype': 'float32',
              'seed': 42, 'threads': 1, 'registry_pins': model.pin_registry.num_pins,
              'rays': 16, 'samples': 512, 'iterations': args.iterations, 'warmup': 3}

    def measure(name, fn):
        values = []
        for i in range(args.iterations + 3):
            model.zero_grad(set_to_none=True)
            if args.device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            fn()
            if args.device == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if i >= 3:
                values.append(elapsed)
        report[name] = {'mean_ms': statistics.mean(values), 'median_ms': statistics.median(values),
                        'min_ms': min(values), 'max_ms': max(values)}
        if args.device == 'cuda':
            report[name]['peak_allocated_mb'] = torch.cuda.max_memory_allocated() / 2**20

    def step(mode):
        if mode == 'unpinned':
            tr = model.get_unpinned_slice_to_spiral_transform()
            out = tr.inv(points)
        else:
            tr = model.get_slice_to_spiral_transform()
            out = tr.inv(points)
        out.square().mean().backward()

    for mode in ('unpinned', 'generic'):
        measure(mode, lambda mode=mode: step(mode))
    for count in (40, 100, 0):
        model.cfg['sample_count_pins'] = count
        measure(f'compute_pins_{count or "full"}', lambda: model.compute_pins(full=False).square().mean().backward())
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        step('generic')
    print(json.dumps(report, indent=2))
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=10))


if __name__ == '__main__':
    main()
