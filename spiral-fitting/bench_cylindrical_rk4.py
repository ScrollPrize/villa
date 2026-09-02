#!/usr/bin/env python3
"""Fresh-process RTX 5090 benchmark for stationary RK4 flow integration.

Examples:
  python bench_cylindrical_rk4.py
  python bench_cylindrical_rk4.py --variants cylindrical-fused --blocks 64 128 256
  ncu --set basic -o cyl-rk4 python bench_cylindrical_rk4.py --worker \
      --variant cylindrical-fused --warmups 1 --iterations 1
"""
import argparse
import json
import os
import statistics
import subprocess
import sys


VARIANTS = ('cylindrical-eager', 'cartesian-direct', 'cylindrical-fused')


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--resolution', nargs=3, type=int, default=(832, 400, 400))
    parser.add_argument('--scale-factor', type=int, default=6)
    parser.add_argument('--points', type=int, default=2_400_000)
    parser.add_argument('--reduced-eager-points', type=int, default=200_000)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--block', type=int, choices=(64, 128, 256), default=128)
    parser.add_argument('--blocks', nargs='+', type=int, choices=(64, 128, 256))
    parser.add_argument('--allow-busy', action='store_true')
    return parser.parse_args()


def _summary(values):
    ordered = sorted(values)
    percentile = lambda q: ordered[round(q * (len(ordered) - 1))]
    return {
        'mean_ms': statistics.fmean(values),
        'p50_ms': percentile(0.50),
        'p95_ms': percentile(0.95),
    }


def _worker(args):
    os.environ['AGENTS_AGENT_MODE'] = '1'
    os.environ['FIT_SPIRAL_CYL_BLOCK'] = str(args.block)
    os.environ['FIT_SPIRAL_TRITON'] = (
        '0' if args.variant == 'cylindrical-eager' else '1')
    if args.variant == 'cartesian-direct':
        os.environ['FIT_SPIRAL_DIRECT_LR'] = '1'

    import torch
    import flow_triton
    from flow_fields import CartesianFlowField, CylindricalFlowField

    torch.manual_seed(1701)
    device = torch.device('cuda')
    resolution = tuple(args.resolution)
    if args.variant == 'cartesian-direct':
        flow = CartesianFlowField(
            resolution, args.scale_factor, direct_lr=True).to(device)
    else:
        flow = CylindricalFlowField(resolution, args.scale_factor).to(device)
    with torch.no_grad():
        flow.flows[0].uniform_(-2.0e-4, 2.0e-4)
        flow.flows[1].uniform_(-5.0e-5, 5.0e-5)

    generator = torch.Generator(device=device).manual_seed(314159)
    z = torch.rand(args.points, generator=generator, device=device)
    theta = torch.rand(args.points, generator=generator, device=device) * (2 * torch.pi)
    radius = torch.sqrt(torch.rand(args.points, generator=generator, device=device))
    points = torch.stack(
        [z, 0.5 + 0.5 * radius * torch.sin(theta),
         0.5 + 0.5 * radius * torch.cos(theta)], dim=-1).requires_grad_(True)
    upstream = torch.full_like(points, 0.03125)
    del z, theta, radius
    h = 1.0 / args.steps

    phases = ('setup', 'forward', 'backward', 'handoff', 'total')

    def one_iteration(record):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        end = torch.cuda.Event(enable_timing=True)
        events[0].record()
        flow.zero_grad(set_to_none=True)
        points.grad = None
        integrate = flow.get_time_invariant_integrator()
        events[1].record()
        output = integrate(points, h, args.steps)
        events[2].record()
        output.backward(upstream)
        events[3].record()
        flow.apply_accumulated_field_grad()
        events[4].record()
        end.record()
        torch.cuda.synchronize()
        if not record:
            return None
        boundaries = [event.elapsed_time(events[index + 1])
                      for index, event in enumerate(events[:-1])]
        return dict(zip(phases, boundaries + [events[0].elapsed_time(end)]))

    try:
        for _ in range(args.warmups):
            one_iteration(False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = [one_iteration(True) for _ in range(args.iterations)]
        result = {
            'status': 'ok',
            'variant': args.variant,
            'resolution': resolution,
            'scale_factor': args.scale_factor,
            'points': args.points,
            'steps': args.steps,
            'warmups': args.warmups,
            'iterations': args.iterations,
            'block': args.block if args.variant == 'cylindrical-fused' else None,
            'device': torch.cuda.get_device_name(),
            'capability': torch.cuda.get_device_capability(),
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'triton': flow_triton.triton.__version__,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'phases': {phase: _summary([sample[phase] for sample in samples])
                       for phase in phases},
        }
    except torch.OutOfMemoryError as error:
        result = {
            'status': 'oom', 'variant': args.variant, 'points': args.points,
            'error': str(error),
        }
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'ok' else 2


def _gpu_preflight(allow_busy):
    command = [
        'nvidia-smi',
        '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits',
    ]
    line = subprocess.check_output(command, text=True).strip().splitlines()[0]
    name, driver, total, used, utilization = [item.strip() for item in line.split(',')]
    result = {
        'name': name, 'driver': driver, 'memory_total_mib': int(total),
        'memory_used_mib': int(used), 'utilization_percent': int(utilization),
    }
    if not allow_busy and (int(used) > 3000 or int(utilization) > 10):
        raise SystemExit(f'GPU is not idle: {json.dumps(result)}')
    return result


def _controller(args):
    preflight = _gpu_preflight(args.allow_busy)
    results = []
    blocks = args.blocks or [args.block]
    for variant in args.variants:
        variant_blocks = blocks if variant == 'cylindrical-fused' else [args.block]
        for block in variant_blocks:
            for repetition in range(args.repetitions):
                command = [
                    sys.executable, os.path.abspath(__file__), '--worker',
                    '--variant', variant, '--resolution', *map(str, args.resolution),
                    '--scale-factor', str(args.scale_factor), '--points', str(args.points),
                    '--steps', str(args.steps), '--warmups', str(args.warmups),
                    '--iterations', str(args.iterations), '--block', str(block),
                ]
                completed = subprocess.run(command, text=True, capture_output=True)
                output = completed.stdout.strip().splitlines()
                if not output:
                    raise RuntimeError(completed.stderr)
                result = json.loads(output[-1])
                result['repetition'] = repetition
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
                if (variant == 'cylindrical-eager' and result['status'] == 'oom'
                        and args.points != args.reduced_eager_points):
                    reduced = command.copy()
                    reduced[reduced.index('--points') + 1] = str(args.reduced_eager_points)
                    diagnostic = subprocess.run(reduced, text=True, capture_output=True)
                    reduced_result = json.loads(diagnostic.stdout.strip().splitlines()[-1])
                    reduced_result['diagnostic_for_full_oom'] = True
                    reduced_result['repetition'] = repetition
                    results.append(reduced_result)
                    print(json.dumps(reduced_result, sort_keys=True), flush=True)
                    break
    report = {'preflight': preflight, 'results': results}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    parsed = _arguments()
    raise SystemExit(_worker(parsed) if parsed.worker else _controller(parsed))
