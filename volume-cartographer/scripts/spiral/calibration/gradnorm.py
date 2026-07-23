"""Per-component weighted-loss gradient norms for the phase bundle.

Measures, for each bundle component at its configured weight, the L2 gradient
norm on the flow-field, gap-expander, and remaining (linear/dr) parameter
groups. Attachment is measured at weight 1.0 so a default can be chosen from
the ratio. Run on a given checkpoint/z-window.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import fit_spiral as fs
from checkpoint_io import load_checkpoint_cpu
from lasagna_data import prepare_lasagna_volume, prepare_surf_sdt_volume
from sdt_losses import iter_phase_bundle_losses, phase_bundle_component_weights
from phase_tuning import _build_model


def group_norms(model, groups):
    out = {}
    for gname, params in groups.items():
        sq = 0.0
        for p in params:
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum().item())
        out[gname] = sq ** 0.5
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', default='/home/sean/Desktop/spiral_dataset/to_hf')
    parser.add_argument('--z-begin', type=int, default=8000)
    parser.add_argument('--z-end', type=int, default=9250)
    parser.add_argument('--passes', type=int, default=4)
    parser.add_argument('--pairs', type=int, default=2000)
    parser.add_argument('--cache', default='/home/sean/Desktop/spiral_dataset/to_hf/.spiral-cache')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    checkpoint = load_checkpoint_cpu(args.checkpoint)
    cfg = dict(fs.default_config)
    cfg.update({k: v for k, v in checkpoint.get('cfg', {}).items() if k in cfg})
    cfg.update({
        'dense_spacing_mode': 'phase',
        'dense_spacing_num_pairs': int(args.pairs),
        # measure attachment at unit weight; keep repo-default 12/8/2 for the rest
        'loss_weight_dense_spacing': 12.0,
        'loss_weight_dense_spacing_count': 8.0,
        'loss_weight_min_spacing': 2.0,
        'loss_weight_dense_attachment': 1.0,
    })
    outward = checkpoint.get('spiral_outward_sense') or 'CW'
    lasagna_group = str(checkpoint.get('lasagna_group') or '4')
    lasagna_scale = int(checkpoint.get('lasagna_scale') or 4)
    model = _build_model(checkpoint, dataset, cfg, outward)
    model.train()
    transform = model.get_slice_to_spiral_transform()
    dr = model.get_dr_per_winding()
    outer = int(cfg['shell_outer_winding_idx'])

    flow_params = list(model.flow_field.parameters())
    gap_params = list(model.gap_expander_params.parameters())
    ids = {id(p) for p in flow_params + gap_params}
    other_params = [p for p in model.parameters() if id(p) not in ids]
    groups = {'flow_field': flow_params, 'gap_expander': gap_params,
              'other': other_params}

    normals = prepare_lasagna_volume(
        None, use_normals=True, use_spacing=True,
        normal_nx_zarr_path=str(dataset / 'lasagna_inputs/las_008_nx.ome.zarr'),
        normal_ny_zarr_path=str(dataset / 'lasagna_inputs/las_008_ny.ome.zarr'),
        grad_mag_zarr_path=str(
            dataset / 'lasagna_inputs/las_008_grad_mag.ome.zarr'),
        normal_zarr_group=lasagna_group,
        z_begin=args.z_begin, z_end=args.z_end, lasagna_scale=lasagna_scale,
        storage_backend='auto', cache_directory=args.cache)
    sdt = prepare_surf_sdt_volume(
        str(dataset / 'lasagna_inputs/las_008_surf_sdt.ome.zarr'), '1',
        z_begin=args.z_begin, z_end=args.z_end,
        cache_directory=args.cache, storage_backend='auto')

    weights = phase_bundle_component_weights(cfg, attachment_ramp=1.0)
    records = {}
    losses = {}
    try:
        component_names = ['dense_spacing_count', 'dense_spacing_phase',
                           'dense_spacing_density', 'min_spacing',
                           'dense_attachment']
        # grad_mag-mode references, measured through iter_lasagna_losses at
        # their production weights: 'dense_spacing' is the legacy spacing
        # objective the bundle replaced, 'dense_normals' the ever-present
        # stabilizer — the magnitude yardsticks for the bundle terms.
        import losses as losses_module
        from losses import iter_lasagna_losses
        losses_module.configure_losses(cfg, args.z_begin, args.z_end)
        lasagna_weights = {
            'dense_spacing': float(cfg['loss_weight_dense_spacing']),
            'dense_normals': float(cfg['loss_weight_dense_normals']),
        }
        for p in range(args.passes):
            for target_name in ('dense_spacing', 'dense_normals'):
                torch.manual_seed(0xCA11B4A + p * 7_777_777)
                transform = model.get_slice_to_spiral_transform()
                dr = model.get_dr_per_winding()
                model.zero_grad(set_to_none=True)
                for name, loss in iter_lasagna_losses(
                        transform, dr, normals, outer,
                        cfg['dense_normals_num_points'],
                        compute_spacing=True):
                    if name != target_name:
                        continue
                    weighted = loss * lasagna_weights[name]
                    if weighted.requires_grad:
                        weighted.backward()
                    apply_acc = getattr(model.flow_field,
                                        'apply_accumulated_field_grad', None)
                    if apply_acc is not None:
                        apply_acc()
                    norms = group_norms(model, groups)
                    records.setdefault(name, []).append(norms)
                    losses.setdefault(name, []).append(float(loss.item()))
            for target_name in component_names:
                # Fresh transform + identical seed per component: the RK4
                # field-gradient accumulator is armed per transform, so each
                # component must own its graph to be measured in isolation.
                generator = torch.Generator(device=model.device).manual_seed(
                    0xCA11B4A + p * 7_777_777)
                transform = model.get_slice_to_spiral_transform()
                dr = model.get_dr_per_winding()
                model.zero_grad(set_to_none=True)
                # a component may yield several times (density: shared batch
                # + supplement chunks) and the graphs share the transform —
                # accumulate and backward ONCE or the second yield hits a
                # freed graph
                acc = None
                acc_loss = 0.0
                for name, loss, metrics in iter_phase_bundle_losses(
                        model, transform, dr, sdt, normals, outer, cfg,
                        args.z_begin, args.z_end, generator=generator):
                    if name != target_name:
                        continue
                    term = loss * weights[name]
                    acc = term if acc is None else acc + term
                    acc_loss += float(loss.item())
                if acc is None or not acc.requires_grad:
                    continue
                acc.backward()
                apply_acc = getattr(model.flow_field,
                                    'apply_accumulated_field_grad', None)
                if apply_acc is not None:
                    apply_acc()
                norms = group_norms(model, groups)
                records.setdefault(target_name, []).append(norms)
                losses.setdefault(target_name, []).append(acc_loss)
            print(f'pass {p + 1}/{args.passes} done')
    finally:
        for volume in (normals, sdt):
            if volume['backend'] == 'mmap':
                volume['store'].close()

    summary = {}
    for name, rows in records.items():
        summary[name] = {
            'weight': weights.get(name, {
                'dense_spacing': float(cfg['loss_weight_dense_spacing']),
                'dense_normals': float(cfg['loss_weight_dense_normals']),
            }.get(name, 0.0)),
            'loss_mean': float(np.mean(losses[name])),
            'grad_norms': {
                g: {'mean': float(np.mean([r[g] for r in rows])),
                    'std': float(np.std([r[g] for r in rows]))}
                for g in rows[0]
            },
        }
    Path(args.output).write_text(json.dumps({
        'checkpoint': args.checkpoint,
        'z_range': [args.z_begin, args.z_end],
        'passes': args.passes, 'pairs': args.pairs,
        'weights_used': weights,
        'summary': summary,
    }, indent=2) + '\n')
    for name, s in summary.items():
        gn = s['grad_norms']
        print(f"{name} (w={s['weight']}): loss={s['loss_mean']:.4f} "
              + ' '.join(f"{g}={gn[g]['mean']:.3e}" for g in gn))
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
