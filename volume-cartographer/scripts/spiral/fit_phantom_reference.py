"""CPU reference fitter for phantom datasets: the recovery experiment.

Fits a fresh SpiralAndTransform (the production model class) to a dataset
exported by export_phantom_dataset.py, using only the exported patches'
absolute winding annotations -- then the result can be scored against the
phantom's known truth with winding_error.py. This closes the phantom ->
dataset -> fit -> truth-referenced-score loop end to end on any machine.

This is deliberately NOT fit_spiral.py: the production fitter is CUDA-only
(its patch atlas and training loop hard-code 'cuda') and optimises a large
family of losses (patch DT, tracks, dense normals, spacing, ...). This
reference fitter optimises the same *model* with one loss -- Huber on
(predicted winding - annotated winding - gauge), the same quantity
get_patch_abs_winding_loss anchors (shifted_radius == winding *
dr_per_winding) -- so its scores are a baseline, a lower bound on what the
production fitter should achieve on the same dataset, not a reimplementation
of it. Run the production fitter on the exported dataset on a CUDA host for
the real number.

What the fitter is allowed to see: the exported patches, the umbilicus curve
(human-provided in production too), and volume/lattice geometry from
dataset_meta.json (operator-chosen in production). It never reads the
phantom's sampled deformation parameters.

Winding-annotation caveat: a winding field jumps by 1 across the canonical
theta=0 ray, so samples whose *current* predicted theta sits within
--seam-margin of that ray are dropped from each step's loss (mirroring the
production loss's L-strip seam unwrapping, at reference-fitter fidelity).

Output: a phantom-style checkpoint (carries its umbilicus, so winding_error
consumes it directly via --candidate-checkpoint).
"""

import json
import os

import click
import numpy as np
import torch
from tqdm import tqdm

from sample_spiral import get_theta_and_radii
from synth_phantom import build_model, make_phantom_config
from tifxyz import load_tifxyz


def load_dataset(dataset_dir):
    meta = json.load(open(os.path.join(dataset_dir, 'dataset_meta.json')))
    if not meta['winding_annotations']:
        raise click.UsageError(
            'this reference fitter needs winding annotations; re-export without '
            '--no-winding-annotations')
    umbilicus = json.load(open(os.path.join(dataset_dir, 'umbilicus.json')))
    umbilicus_zyx = torch.tensor(
        [[p['z'], p['y'], p['x']] for p in
         sorted(umbilicus['control_points'], key=lambda p: p['z'])],
        dtype=torch.float32)

    patches_dir = os.path.join(dataset_dir, 'verified_patches')
    zyxs, windings = [], []
    for entry in sorted(os.listdir(patches_dir)):
        patch = load_tifxyz(os.path.join(patches_dir, entry))
        if not isinstance(patch.winding, torch.Tensor):
            raise click.UsageError(f'{entry} has no per-vertex winding annotation')
        # Sparse grids mark missing cells with the -1 sentinel; training on
        # those would anchor the fit to garbage coordinates (phantom exports
        # are fully valid, so this only bites on real traced data).
        valid = patch.valid_vertex_mask.reshape(-1)
        zyxs.append(patch.zyxs.reshape(-1, 3)[valid])
        windings.append(patch.winding.reshape(-1)[valid])
    return meta, umbilicus_zyx, torch.cat(zyxs), torch.cat(windings)


@click.command()
@click.option('--dataset', required=True, type=click.Path(exists=True, file_okay=False),
              help='export_phantom_dataset.py output directory.')
@click.option('--out', required=True, type=click.Path(dir_okay=False),
              help='Fitted checkpoint output path (.pt).')
@click.option('--steps', default=600, type=int)
@click.option('--batch', default=8192, type=int)
@click.option('--lr', default=3e-4, type=float,
              help='Adam learning rate (production base rate is 3e-5 over many '
                   'more iterations; the reference fit runs hotter and shorter).')
@click.option('--huber-delta', default=0.25, type=float,
              help='Huber transition point, in windings.')
@click.option('--seam-margin', default=0.15, type=float,
              help='Drop samples within this angle (radians) of the theta=0 ray.')
@click.option('--reg-gap', default=0., type=float,
              help='L2 weight on effective log winding gaps: a prior toward '
                   'uniform spacing between annotated sheets. The annotation '
                   'loss alone leaves inter-sheet interpolation unconstrained '
                   '(a stand-in for the production DT/density losses).')
@click.option('--reg-flow', default=0., type=float,
              help='L2 weight on the flow velocity field (prior toward the '
                   'identity deformation off the annotated surfaces).')
@click.option('--seed', default=0, type=int)
@click.option('--device', default='cpu')
def main(dataset, out, steps, batch, lr, huber_delta, seam_margin, reg_gap,
         reg_flow, seed, device):
    torch.manual_seed(seed)
    device = torch.device(device)
    meta, umbilicus_zyx, zyxs, windings = load_dataset(dataset)
    pm = meta['phantom_meta']
    click.echo(f"{len(zyxs)} annotated vertices from {meta['num_patches']} patches")

    z_margin = max(8, pm['z_size'] // 4)
    cfg = make_phantom_config(pm['z_size'], pm['yx_size'], pm['dr_per_winding'],
                              num_table_windings=pm['last_winding'] + 8,
                              z_margin=z_margin)
    model = build_model(cfg, 0, pm['z_size'], umbilicus_zyx, 'CW', device)
    model.train().requires_grad_(True)
    # Initialise the gauge at the median initial residual: annotations may
    # carry a large constant winding offset (any theta-origin / numbering
    # convention is valid), and at lr * steps the gauge scalar can only travel
    # a fraction of a winding during the fit -- a mis-initialised gauge
    # otherwise leaves the whole fit stuck at |offset| windings of loss.
    with torch.no_grad():
        transform0 = model.get_slice_to_spiral_transform()
        dr0 = model.get_dr_per_winding()
        _, _, shifted0 = get_theta_and_radii(
            transform0(zyxs.to(device))[..., 1:], dr0)
        gauge0 = (shifted0 / dr0 - windings.to(device)).median()
    gauge = gauge0.clone().requires_grad_(True)
    click.echo(f'gauge initialised to {float(gauge0):+.3f} windings')
    optimiser = torch.optim.Adam([*model.parameters(), gauge], lr=lr)

    zyxs = zyxs.to(device)
    windings = windings.to(device)
    losses = []
    for step in tqdm(range(steps), desc='fit'):
        idx = torch.randint(0, len(zyxs), [min(batch, len(zyxs))], device=device)
        transform = model.get_slice_to_spiral_transform()
        spiral = transform(zyxs[idx])
        dr = model.get_dr_per_winding()
        theta, _, shifted = get_theta_and_radii(spiral[..., 1:], dr)
        residual = shifted / dr - windings[idx] - gauge
        keep = torch.minimum(theta, 2 * torch.pi - theta) > seam_margin
        loss = F_huber(residual[keep], huber_delta)
        if reg_gap > 0:
            # Effective log gap = logits * lr_scale * 2e2 (GapExpandingTransform).
            eff = model.gap_expander_params.logits * (
                cfg['model_gap_expander_lr_scale'] * 2.e2)
            loss = loss + reg_gap * eff.pow(2).mean()
        if reg_flow > 0:
            loss = loss + reg_flow * sum(
                flow.pow(2).mean() for field in model.flow_fields
                for flow in field.flows)
        optimiser.zero_grad()
        loss.backward()
        # The flow field routes its gradients through a shared accumulator
        # (see CartesianFlowField); flush it into .grad like the production
        # loop does before stepping.
        for field in model.flow_fields:
            field.apply_accumulated_field_grad()
        optimiser.step()
        losses.append(float(loss))
        if (step + 1) % 100 == 0:
            tqdm.write(f'step {step + 1}: loss {np.mean(losses[-100:]):.5f} '
                       f'windings, dr {float(dr):.3f}, gauge {float(gauge):+.3f}')

    model.eval().requires_grad_(False)
    torch.save({
        'schema_version': 2,
        'phantom_schema_version': 1,  # phantom-style: carries its umbilicus
        'spiral_and_transform': model.state_dict(),
        'cfg': cfg,
        'z_begin': 0,
        'z_end': pm['z_size'],
        'spiral_outward_sense': 'CW',
        'umbilicus_zyx': umbilicus_zyx.cpu(),
        'reference_fit': {'dataset': os.path.abspath(dataset), 'steps': steps,
                          'lr': lr, 'batch': batch, 'seed': seed,
                          'reg_gap': reg_gap, 'reg_flow': reg_flow,
                          'final_loss': float(np.mean(losses[-50:])),
                          'gauge': float(gauge)},
    }, out)
    click.echo(f'fitted checkpoint -> {out} '
               f'(final loss {np.mean(losses[-50:]):.5f} windings)')


def F_huber(residual, delta):
    absr = residual.abs()
    quadratic = 0.5 * residual ** 2 / delta
    return torch.where(absr <= delta, quadratic, absr - 0.5 * delta).mean()


if __name__ == '__main__':
    main()
