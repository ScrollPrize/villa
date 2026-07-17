"""Offline phase-bundle measurement/tuning on a saved Spiral transform.

This runner never creates an optimizer and never mutates or saves the model.
It reconstructs a checkpoint, prepares the production SDT/normal stores, and
repeats the no-grad phase-registration and crossing-count measurements with
private generators, writing all per-pass and aggregate metrics to JSON. It
calls the phase measurement directly - there is no training-time shadow mode.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch

import fit_spiral as fs
from checkpoint_io import load_checkpoint_cpu
from lasagna_data import prepare_lasagna_volume, prepare_surf_sdt_volume
from sdt_losses import get_min_spacing_loss, iter_phase_bundle_losses


def _build_model(checkpoint, dataset, cfg, outward_sense):
    device = torch.device('cuda')
    model_z_begin = int(checkpoint['z_begin'])
    model_z_end = int(checkpoint['z_end'])
    umbilicus_fn = fs.json_umbilicus_z_to_yx(
        str(dataset / 'umbilicus.json'), coordinate_scale=1.0)
    all_z = np.arange(model_z_begin, model_z_end)
    umbilicus_zyx = torch.from_numpy(np.concatenate([
        all_z[:, None], umbilicus_fn(all_z),
    ], axis=-1).astype(np.float32)).to(device)
    radius = int(cfg['flow_bounds_radius'])
    flow_min = torch.tensor([
        model_z_begin - int(cfg['flow_bounds_z_margin']), -radius, -radius,
    ], dtype=torch.int64, device=device)
    flow_max = torch.tensor([
        model_z_end + int(cfg['flow_bounds_z_margin']), radius, radius,
    ], dtype=torch.int64, device=device)
    model = fs.SpiralAndTransform(
        flow_integration_steps=int(cfg['num_flow_integration_steps']),
        flow_integration_solver=cfg['flow_integration_solver'],
        flow_min_corner_zyx=flow_min,
        flow_max_corner_zyx=flow_max,
        umbilicus_zyx=umbilicus_zyx,
        config=cfg,
        spiral_outward_sense=outward_sense,
    ).to(device)
    model.load_state_dict(checkpoint['spiral_and_transform'])
    model.eval()
    model.flow_field.flow_scales[1] = cfg[
        'flow_field_high_res_lr_scale_final']
    return model


def _aggregate(passes):
    keys = sorted(set().union(*(item.keys() for item in passes)))
    summary = {}
    for key in keys:
        values = [
            float(item[key]) for item in passes
            if key in item and isinstance(item[key], (int, float))
        ]
        if not values:
            continue
        summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'num_passes': len(values),
        }
    return summary


def _barrier_gradient_probe(model, outer_winding, cfg, z_begin, z_end, seed):
    model.zero_grad(set_to_none=True)
    generator = torch.Generator(device=model.device).manual_seed(seed)
    loss, metrics = get_min_spacing_loss(
        model, outer_winding, cfg, z_begin, z_end, generator=generator)
    loss.backward()
    grad = model.gap_expander_params.logits.grad
    nonzero = grad.detach().abs()[grad.detach() != 0]
    result = dict(metrics)
    result.update({
        'loss': float(loss.detach().item()),
        'gap_logit_gradient_l2': float(grad.detach().norm().item()),
        'gap_logit_gradient_nonzero_count': int(nonzero.numel()),
        'dr_per_winding_gradient_is_none': model.dr_per_winding_logit.grad is None,
    })
    if nonzero.numel():
        q = torch.quantile(
            nonzero, torch.tensor([0.5, 0.9, 0.99], device=nonzero.device))
        result.update({
            'gap_logit_gradient_abs_p50_nonzero': float(q[0].item()),
            'gap_logit_gradient_abs_p90_nonzero': float(q[1].item()),
            'gap_logit_gradient_abs_p99_nonzero': float(q[2].item()),
        })
    model.zero_grad(set_to_none=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--z-begin', type=int, required=True)
    parser.add_argument('--z-end', type=int, required=True)
    parser.add_argument('--passes', type=int, default=8)
    parser.add_argument('--pairs', type=int, default=2000)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.z_begin >= args.z_end:
        raise ValueError('--z-begin must be less than --z-end')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the production phase measurement')

    dataset = args.dataset.resolve()
    checkpoint_path = (
        args.checkpoint.resolve() if args.checkpoint
        else dataset / 'checkpoint_fitted.ckpt')
    checkpoint = load_checkpoint_cpu(str(checkpoint_path))
    cfg = dict(fs.default_config)
    cfg.update({
        key: value for key, value in checkpoint.get('cfg', {}).items()
        if key in cfg
    })
    cfg.update({
        'dense_spacing_mode': 'phase',
        'dense_spacing_num_pairs': int(args.pairs),
        # Measure phase registration and count only; the barrier gradient is
        # probed separately and attachment is not part of this measurement.
        'loss_weight_min_spacing': 0.0,
        'loss_weight_dense_attachment': 0.0,
    })
    outward_sense = checkpoint.get('spiral_outward_sense') or 'CW'
    lasagna_group = str(checkpoint.get('lasagna_group') or '4')
    lasagna_scale = int(checkpoint.get('lasagna_scale') or 4)
    model_z = (int(checkpoint['z_begin']), int(checkpoint['z_end']))
    if not (model_z[0] <= args.z_begin < args.z_end <= model_z[1]):
        raise ValueError(
            f'requested z range [{args.z_begin}, {args.z_end}) is outside '
            f'checkpoint model range [{model_z[0]}, {model_z[1]})')

    args.cache.mkdir(parents=True, exist_ok=True)
    print(f'loading checkpoint {checkpoint_path}')
    model_started = time.perf_counter()
    model = _build_model(checkpoint, dataset, cfg, outward_sense)
    model_prepare_seconds = time.perf_counter() - model_started
    transform = model.get_slice_to_spiral_transform()
    dr_per_winding = model.get_dr_per_winding()
    outer_winding = int(cfg['shell_outer_winding_idx'])

    print(f'preparing normal store for z [{args.z_begin}, {args.z_end})')
    normal_started = time.perf_counter()
    normals = prepare_lasagna_volume(
        None,
        use_normals=True,
        use_spacing=False,
        normal_nx_zarr_path=str(
            dataset / 'lasagna_inputs/las_008_nx.ome.zarr'),
        normal_ny_zarr_path=str(
            dataset / 'lasagna_inputs/las_008_ny.ome.zarr'),
        grad_mag_zarr_path=None,
        normal_zarr_group=lasagna_group,
        z_begin=args.z_begin,
        z_end=args.z_end,
        lasagna_scale=lasagna_scale,
        storage_backend='mmap',
        cache_directory=str(args.cache),
    )
    normal_prepare_seconds = time.perf_counter() - normal_started
    print(f'preparing SDT store for z [{args.z_begin}, {args.z_end})')
    sdt_started = time.perf_counter()
    sdt = prepare_surf_sdt_volume(
        str(dataset / 'lasagna_inputs/las_008_surf_sdt.ome.zarr'),
        '1',
        z_begin=args.z_begin,
        z_end=args.z_end,
        cache_directory=str(args.cache),
        storage_backend='mmap',
    )
    sdt_prepare_seconds = time.perf_counter() - sdt_started

    pass_metrics = []
    try:
        for pass_index in range(args.passes):
            generator = torch.Generator(device=model.device).manual_seed(
                0x5EED5EED + pass_index * 1_000_003)
            torch.cuda.synchronize()
            baseline_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            pass_started = time.perf_counter()
            metrics = {}
            with torch.no_grad():
                for _name, _loss, component_metrics in iter_phase_bundle_losses(
                        model, transform, dr_per_winding, sdt, normals,
                        outer_winding, cfg, args.z_begin, args.z_end,
                        generator=generator):
                    metrics.update({
                        key: value
                        for key, value in component_metrics.items()
                        if not key.startswith('_')
                    })
            torch.cuda.synchronize()
            metrics.update({
                'dense_spacing_phase_pass_wall_seconds': (
                    time.perf_counter() - pass_started),
                'dense_spacing_phase_cuda_baseline_allocated_bytes': float(
                    baseline_allocated),
                'dense_spacing_phase_cuda_peak_allocated_bytes': float(
                    torch.cuda.max_memory_allocated()),
                'dense_spacing_phase_cuda_peak_incremental_bytes': float(
                    max(0, torch.cuda.max_memory_allocated() - baseline_allocated)),
                'dense_spacing_phase_cuda_peak_reserved_bytes': float(
                    torch.cuda.max_memory_reserved()),
            })
            pass_metrics.append(metrics)
            print(
                f'pass {pass_index + 1}/{args.passes}: '
                f'valid={metrics.get("dense_spacing_phase_valid_fraction", 0):.3f}, '
                f'|rho|={metrics.get("dense_spacing_phase_residual_mean_abs", float("nan")):.3f}, '
                f'match_mass={metrics.get("dense_spacing_phase_match_mass", 0):.1f}, '
                f'missing/ray={metrics.get("dense_spacing_phase_missing_per_ray", 0):.3f}, '
                f'extra/ray={metrics.get("dense_spacing_phase_extra_per_ray", 0):.3f}, '
                f'entropy={metrics.get("dense_spacing_phase_alignment_entropy_mean", 0):.3f}')
        barrier_probe = _barrier_gradient_probe(
            model, outer_winding, cfg, args.z_begin, args.z_end,
            0x6A09E667)
    finally:
        normals['store'].close()
        sdt['store'].close()

    summary = _aggregate(pass_metrics)
    report = {
        'schema_version': 2,
        'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'dataset': str(dataset),
        'checkpoint': str(checkpoint_path),
        'checkpoint_model_z_range': list(model_z),
        'measurement_z_range': [args.z_begin, args.z_end],
        'passes': args.passes,
        'pairs_per_pass': args.pairs,
        'preparation_seconds': {
            'model': model_prepare_seconds,
            'normal_store': normal_prepare_seconds,
            'sdt_store': sdt_prepare_seconds,
        },
        'configuration': {
            key: cfg[key] for key in (
                'dense_spacing_phase_huber_delta',
                'dense_spacing_phase_extension_windings',
                'dense_spacing_phase_min_center_gap_wv',
                'dense_spacing_phase_graze_dot',
                'dense_spacing_phase_graze_depth_wv',
                'dense_spacing_phase_window_windings',
                'dense_spacing_phase_missing_cost',
                'dense_spacing_phase_missing_extend_cost',
                'dense_spacing_phase_extra_cost',
                'dense_spacing_phase_extra_extend_cost',
                'dense_spacing_phase_temperature',
                'dense_spacing_phase_band_confidence_cost',
                'dense_spacing_phase_top2_margin',
                'dense_spacing_phase_min_matched_windings',
                'dense_spacing_phase_min_matched_mass',
                'min_spacing_d_min_wv',
                'min_spacing_independent_samples',
            )
        },
        'summary': summary,
        'barrier_gradient_probe': barrier_probe,
        'per_pass': pass_metrics,
    }
    output = args.output
    if output is None:
        stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        output = dataset / 'spiral_output' / (
            f'phase_tuning_z{args.z_begin}-{args.z_end}_{stamp}.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
