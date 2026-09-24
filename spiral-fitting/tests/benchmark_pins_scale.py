"""Production-scale synthetic Stage 2b benchmark (no data, no training).

A production-sized model (130 windings, 3200-voxel flow box, two flow stages)
with a synthetic registry of patch-grid pins on the ideal spiral, in
thousands of components. Times one step's worth of transform evaluations,
forward + backward, with the free and the pinned gap expander.

AGENTS_AGENT_MODE=1 .venv/bin/python tests/benchmark_pins_scale.py --device cuda
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pins
from config import Config
from transforms import SpiralAndTransform

TWO_PI = 2 * math.pi


def build(device, num_components, pins_per_patch, sample_count_pins, rebin_interval, seed=42):
    torch.manual_seed(seed)
    cfg = Config().as_dict()
    cfg['sample_count_pins'] = sample_count_pins
    cfg['model_pin_rebin_interval'] = rebin_interval
    z0, z1, radius = 10000, 11000, cfg['model_flow_bounds_radius']
    margin = cfg['model_flow_bounds_z_margin']
    lo = torch.tensor([z0 - margin, -radius, -radius], dtype=torch.int64, device=device)
    hi = torch.tensor([z1 + margin, radius, radius], dtype=torch.int64, device=device)
    centre = torch.tensor([4000., 4000.])
    umbilicus = torch.zeros([9, 3])
    umbilicus[:, 0] = torch.linspace(z0 - margin, z1 + margin, 9)
    umbilicus[:, 1:] = centre
    model = SpiralAndTransform(
        flow_integration_steps=cfg['model_num_flow_integration_steps'],
        flow_integration_solver=cfg['model_flow_integration_solver'],
        flow_min_corner_zyx=lo, flow_max_corner_zyx=hi, umbilicus_zyx=umbilicus.to(device),
        config=cfg).to(device)
    dr = float(model.get_dr_per_winding())
    # Synthetic registry: one patch grid per component, sitting exactly on the
    # ideal spiral; grids stay clear of the theta = 0 seam and within one turn.
    side = max(int(round(math.sqrt(pins_per_patch))), 2)
    winding = torch.randint(15, 125, [num_components]).float()
    theta_lo = torch.rand([num_components]) * (TWO_PI - 1.2) + 0.1
    z_lo = torch.rand([num_components]) * (z1 - z0 - 200) + z0
    r_mid = dr * (winding + 0.5)
    dz, dtheta = 20.0, 20.0 / r_mid
    gi = torch.arange(side).float()
    theta = theta_lo[:, None, None] + dtheta[:, None, None] * gi[None, :, None]
    z = z_lo[:, None, None] + dz * gi[None, None, :]
    theta, z = torch.broadcast_tensors(theta, z)
    r = dr * (winding[:, None, None] + theta / TWO_PI)
    zyx = torch.stack([z, centre[0] + r * torch.sin(theta), centre[1] + r * torch.cos(theta)], dim=-1).reshape(-1, 3)
    n = zyx.shape[0]
    component = torch.arange(num_components).repeat_interleave(side * side)
    rule = pins.FootprintRule(spacing_factor=cfg['model_pin_kernel_spacing_factor'],
                              min_arc_voxels=cfg['model_pin_kernel_min_arc_voxels'],
                              min_z_voxels=cfg['model_pin_kernel_min_z_voxels'],
                              max_theta_radians=cfg['model_pin_kernel_max_theta_radians'],
                              max_z_voxels=cfg['model_pin_kernel_max_z_voxels'])
    eps_theta, eps_z = rule.apply(dtheta[component], torch.full([n], dz), r.reshape(-1))
    registry = pins.PinRegistry(
        zyx=zyx.float(), component=component, n0=torch.zeros([n], dtype=torch.int32),
        theta0=theta.reshape(-1).float(), eps_theta=eps_theta.float(), eps_z=eps_z.float(),
        kind=torch.full([n], pins.PIN_KIND_PATCH, dtype=torch.int64), num_components=num_components,
        fixed_T=torch.zeros([num_components], dtype=torch.bool), fixed_T_value=torch.zeros([num_components]),
        initial_T=winding, fingerprint='synthetic', consistency_report={},
        local_gap=torch.full([n], dr), patch_component=torch.zeros([0], dtype=torch.int64),
        patch_offset=torch.zeros([0], dtype=torch.int64))
    model.set_pin_registry(registry.to(device))
    model.pins_active = True
    return model, cfg, (z0, z1, centre.to(device), dr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--components', type=int, default=9000)
    parser.add_argument('--pins-per-patch', type=int, default=28)
    parser.add_argument('--sample-count-pins', type=int, default=100000)
    parser.add_argument('--rebin-interval', type=int, default=1)
    parser.add_argument('--points', type=int, default=300000)
    parser.add_argument('--profile', default='')
    args = parser.parse_args()
    device = args.device
    torch._C._jit_override_can_fuse_on_gpu(False)
    model, cfg, (z0, z1, centre, dr) = build(device, args.components, args.pins_per_patch,
                                             args.sample_count_pins, args.rebin_interval)
    torch.manual_seed(1)
    # Loss-like sample sets: spiral-space points across the windings and
    # scroll-space points near the sheets.
    n = args.points
    theta_s = torch.rand([n], device=device) * TWO_PI
    rad_s = dr * (torch.rand([n], device=device) * 110 + 15)
    z_s = torch.rand([n], device=device) * (z1 - z0) + z0
    spiral_pts = torch.stack([z_s, rad_s * theta_s.sin(), rad_s * theta_s.cos()], dim=-1)
    scroll_pts = torch.stack([z_s, centre[0] + rad_s * theta_s.sin(), centre[1] + rad_s * theta_s.cos()], dim=-1)
    report = {'device': device, 'torch': torch.__version__, 'registry_pins': model.pin_registry.num_pins,
              'components': args.components, 'sample_count_pins': args.sample_count_pins,
              'rebin_interval': args.rebin_interval, 'points': n,
              'iterations': args.iterations, 'warmup': 2}

    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()

    def measure(name, fn):
        values = []
        for i in range(args.iterations + 2):
            model.zero_grad(set_to_none=True)
            sync()
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            fn()
            sync()
            if i >= 2:
                values.append((time.perf_counter() - start) * 1000)
        report[name] = {'mean_ms': round(statistics.mean(values), 2), 'median_ms': round(statistics.median(values), 2),
                        'min_ms': round(min(values), 2), 'max_ms': round(max(values), 2)}
        if device == 'cuda':
            report[name]['peak_allocated_mb'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)

    def shared_transform(pinned):
        model.pins_active = pinned
        shared = tuple(x.detach().requires_grad_(True) for x in model.get_shared_transform_tensors())
        return shared, model.get_slice_to_spiral_transform(shared=shared)

    def run(pinned, part):
        shared, tr = shared_transform(pinned)
        if part == 'inverse':
            out = tr.inv(spiral_pts)
        elif part == 'forward':
            out = tr(scroll_pts)
        elif part == 'build':
            return
        out.square().mean().backward()
        if pinned:
            # Propagate the pins leaf once, like the training step does.
            torch.autograd.backward(shared[3], shared[3].grad)

    for part in ('build', 'inverse', 'forward'):
        for pinned in (False, True):
            measure(f'{"pinned" if pinned else "unpinned"}_{part}', lambda pinned=pinned, part=part: run(pinned, part))
    print(json.dumps(report, indent=2))
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            run(True, args.profile)
            sync()
        print(prof.key_averages().table(sort_by='cuda_time_total' if device == 'cuda' else 'self_cpu_time_total', row_limit=25))


if __name__ == '__main__':
    main()
