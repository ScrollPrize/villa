#!/usr/bin/env python3
"""MPS benchmark of the fused Metal kernels against the eager paths they replace.

Modelled on villa's spiral-fitting/bench_cylindrical_rk4.py (fresh process per
variant, the same setup/forward/backward/handoff phases), on Apple's MPS:

  flow-eager   CartesianFlowField(direct_lr=True) with FIT_SPIRAL_METAL=0: the
               MPS fallback (upsampled LR field + eager RK4 per slab)
  flow-metal   the same field with the Metal direct-LR kernels
  gap-eager    GapExpandingTransform _call + _inverse, eager
  gap-metal    the same with the Metal gap kernels

Each worker sets the FIT_SPIRAL_METAL its variant names and clears an
inherited FIT_SPIRAL_DIRECT_LR (which would otherwise turn flow-metal into an
eager run, or make flow-eager refuse to run), then checks that the dispatch
picks the labelled path and reports it as "path" (metal or eager).

Run from spiral-fitting/ on a Mac (exact numbers need an otherwise idle GPU):
  .venv/bin/python bench_metal_kernels.py
  .venv/bin/python bench_metal_kernels.py --variants flow-eager flow-metal --points 1000000 --repetitions 1
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

VARIANTS = ('flow-eager', 'flow-metal', 'gap-eager', 'gap-metal')


def _arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--variant', choices=VARIANTS)
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--resolution', nargs=3, type=int, default=(64, 384, 384))
    p.add_argument('--scale-factor', type=int, default=6)
    p.add_argument('--points', type=int, default=200_000)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--windings', type=int, default=130)
    p.add_argument('--warmups', type=int, default=5)
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--repetitions', type=int, default=2)
    return p.parse_args()


def _summary(values):
    ordered = sorted(values)
    pct = lambda q: ordered[round(q * (len(ordered) - 1))]
    return {'mean_ms': statistics.fmean(values), 'p50_ms': pct(0.5), 'p95_ms': pct(0.95)}


def _worker(args):
    os.environ['AGENTS_AGENT_MODE'] = '1'
    # The variant alone decides the path.
    os.environ['FIT_SPIRAL_METAL'] = '1' if args.variant.endswith('metal') else '0'
    os.environ.pop('FIT_SPIRAL_DIRECT_LR', None)
    wanted = 'metal' if args.variant.endswith('metal') else 'eager'
    import torch
    sync = torch.mps.synchronize
    torch.manual_seed(1701)
    dev = torch.device('mps')
    gen = torch.Generator(device='cpu').manual_seed(314159)

    if args.variant.startswith('flow'):
        from flow_fields import CartesianFlowField
        flow = CartesianFlowField(tuple(args.resolution), args.scale_factor,
                                  direct_lr=True).to(dev)
        with torch.no_grad():
            flow.flows[0].uniform_(-2.0e-4, 2.0e-4)
            flow.flows[1].uniform_(-5.0e-5, 5.0e-5)
        import flow_metal
        interpolant, backend = flow._resolve_interpolant(log=False)
        path = 'metal' if backend is flow_metal else 'eager'
        if path == 'eager' and interpolant != 'upsampled':
            path = f'eager {interpolant}'
        z = torch.rand(args.points, generator=gen)
        theta = torch.rand(args.points, generator=gen) * (2 * math.pi)
        radius = torch.sqrt(torch.rand(args.points, generator=gen))
        points = torch.stack([z, 0.5 + 0.5 * radius * torch.sin(theta),
                              0.5 + 0.5 * radius * torch.cos(theta)], -1).to(dev)
        points.requires_grad_(True)
        upstream = torch.full_like(points, 0.03125)
        h = 1.0 / args.steps

        def iteration():
            t = [time.perf_counter()]
            flow.zero_grad(set_to_none=True)
            points.grad = None
            integrate = flow.get_integrator()
            sync(); t.append(time.perf_counter())
            out = integrate(points, h, args.steps)
            sync(); t.append(time.perf_counter())
            out.backward(upstream)
            sync(); t.append(time.perf_counter())
            flow.apply_accumulated_field_grad()
            sync(); t.append(time.perf_counter())
            return t
        phases = ('setup', 'forward', 'backward', 'handoff')
        shape = {'resolution': list(args.resolution), 'scale_factor': args.scale_factor,
                 'steps': args.steps}
    else:
        from transforms import GapExpanderParams, GapExpandingTransform
        dr_nominal, zmax = 16.0, float(args.resolution[0] * 16)
        params = GapExpanderParams(resolution=24, min_z=0.0, max_z=zmax,
                                   num_windings=args.windings,
                                   dr_per_winding=dr_nominal).to(dev)
        with torch.no_grad():
            params.logits.copy_(torch.randn(params.logits.shape, generator=gen) * 0.01)
        dr = torch.tensor(dr_nominal, device=dev, requires_grad=True)
        theta = torch.rand(args.points, generator=gen) * (2 * math.pi)
        wind = torch.rand(args.points, generator=gen) * (args.windings - 4) + 1
        radius = (wind + theta / (2 * math.pi)) * dr_nominal
        z = torch.rand(args.points, generator=gen) * zmax
        points = torch.stack([z, radius * torch.sin(theta), radius * torch.cos(theta)], -1).to(dev)
        weights = torch.randn(points.shape, generator=gen).to(dev)
        import gap_metal
        probe = GapExpandingTransform(params, dr, 0.0, zmax, gap_expander_lr_scale=0.3,
                                      min_gap=1.0, softplus_bias=4.0)
        path = 'metal' if probe._fused_gap(points, dr) is gap_metal else 'eager'

        def iteration():
            t = [time.perf_counter()]
            params.zero_grad(set_to_none=True)
            dr.grad = None
            tr = GapExpandingTransform(params, dr, 0.0, zmax, gap_expander_lr_scale=0.3,
                                       min_gap=1.0, softplus_bias=4.0)
            x = points.detach().requires_grad_(True)
            sync(); t.append(time.perf_counter())
            loss = (tr._call(x) * weights).sum() + (tr._inverse(x) * weights).sum()
            sync(); t.append(time.perf_counter())
            loss.backward()
            sync(); t.append(time.perf_counter())
            return t
        phases = ('setup', 'forward', 'backward')
        shape = {'windings': args.windings, 'lattice': list(params.logits.shape)}

    if path != wanted:
        raise SystemExit(f'{args.variant} would time the {path} path, not the {wanted} one '
                         f'(FIT_SPIRAL_METAL={os.environ["FIT_SPIRAL_METAL"]}, device {dev})')
    for _ in range(args.warmups):
        iteration()
    samples = []
    for _ in range(args.iterations):
        t = iteration()
        samples.append({ph: (t[i + 1] - t[i]) * 1e3 for i, ph in enumerate(phases)}
                       | {'total': (t[-1] - t[0]) * 1e3})
    result = {
        'status': 'ok', 'variant': args.variant, 'path': path, 'points': args.points, **shape,
        'iterations': args.iterations, 'torch': torch.__version__,
        'driver_allocated_gib': torch.mps.driver_allocated_memory() / 2**30,
        'phases': {ph: _summary([s[ph] for s in samples]) for ph in (*phases, 'total')},
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _controller(args):
    inherited = sorted(k for k in ('FIT_SPIRAL_METAL', 'FIT_SPIRAL_DIRECT_LR') if k in os.environ)
    if inherited:
        print(f'note: ignoring {", ".join(inherited)} from the environment; each variant sets its own',
              flush=True)
    results = []
    for rep in range(args.repetitions):
        for variant in args.variants:   # interleaved arms
            cmd = [sys.executable, os.path.abspath(__file__), '--worker', '--variant', variant,
                   '--resolution', *map(str, args.resolution), '--scale-factor', str(args.scale_factor),
                   '--points', str(args.points), '--steps', str(args.steps),
                   '--windings', str(args.windings), '--warmups', str(args.warmups),
                   '--iterations', str(args.iterations)]
            done = subprocess.run(cmd, text=True, capture_output=True)
            lines = done.stdout.strip().splitlines()
            if not lines:
                raise RuntimeError(done.stderr[-3000:])
            r = json.loads(lines[-1])
            r['repetition'] = rep
            results.append(r)
            print(f"{variant:11s} rep{rep} ({r['path']}): total p50 {r['phases']['total']['p50_ms']:9.2f} ms "
                  f"(fwd {r['phases']['forward']['p50_ms']:.2f}, bwd {r['phases']['backward']['p50_ms']:.2f}) "
                  f"driver {r['driver_allocated_gib']:.2f} GiB", flush=True)
    print(json.dumps({'results': results}, indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    a = _arguments()
    raise SystemExit(_worker(a) if a.worker else _controller(a))
