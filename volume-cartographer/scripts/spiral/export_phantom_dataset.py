"""Export a synth_phantom phantom as a fit_spiral-consumable dataset.

Writes the minimal input layout the fitter's loaders read:

  <out>/umbilicus.json           {"control_points": [{"z","y","x"}, ...]},
                                 from the phantom's own (known) curve
  <out>/verified_patches/<id>/   tifxyz patch grids (written by tifxyz.
                                 save_tifxyz, so the format always matches the
                                 loader) sampled from TRUE winding surfaces,
                                 plus winding.tif carrying the absolute
                                 winding annotation in the fit's convention
                                 (shifted_radius == winding * dr_per_winding)
  <out>/dataset_meta.json        provenance + the knob values

Because patches are derived from truth, their quality is a *dial* rather than
an accident of annotation effort -- which turns the open-problems doc's "label
quality is now one of the main unwrapping bottlenecks" from a belief into a
measurable curve:

  --coverage        fraction of each winding's arc covered by patches
  --winding-stride  annotate every Nth winding only
  --position-noise  iid Gaussian vertex corruption, in voxels (label
                    imprecision -- labels that "wiggle" off the true surface)
  --no-winding-annotations  drop winding.tif (geometry-only patches)

Scoring a fit trained on each exported variant with winding_error.py measures
how much label quantity/quality that fitter needs. Scope caveat: dial curves
produced with fit_phantom_reference.py (a single-loss baseline) characterise
the BASELINE's sensitivity, not the production fitter's -- fit_spiral carries
DT/track/normal losses that may absorb label defects very differently. Treat
reference-fitter curves as demonstrations of the dials until the production
fitter is run on the same exported datasets (CUDA host required).

The production fitter is CUDA-only (fit_spiral.py builds its patch atlas on
'cuda'); this exporter is device-free, so datasets can be produced anywhere
and fitted on a GPU host. fit_phantom_reference.py is the CPU path.
"""

import json
import os

import click
import numpy as np
import tifffile
import torch

from checkpoint_io import load_checkpoint_cpu
from synth_phantom import build_model_from_phantom_checkpoint
from tifxyz import save_tifxyz


def export_umbilicus(umbilicus_zyx, path):
    points = [{'z': float(z), 'y': float(y), 'x': float(x)}
              for z, y, x in umbilicus_zyx.tolist()]
    with open(path, 'w') as f:
        json.dump({'control_points': points}, f)


@torch.no_grad()
def winding_surface_grid(model, winding, theta0, theta1, z0, z1, step, device):
    """A [rows=z, cols=theta] grid on true winding `winding`, arc [theta0,
    theta1), mapped into slice space through the phantom's spiral->slice map.

    Returns (zyxs [H, W, 3] float32 numpy, winding_annotation [H, W] float32).
    Column spacing targets `step` voxels of arc length at the winding's mean
    radius, matching the row spacing so the tifxyz scale is square.
    """
    dr = float(model.get_dr_per_winding())
    mean_radius = (winding + 0.5) * dr
    num_cols = max(2, int(round((theta1 - theta0) * mean_radius / step)))
    num_rows = max(2, int(round((z1 - z0) / step)))
    theta = torch.linspace(theta0, theta1, num_cols)
    z = torch.linspace(z0, z1, num_rows)
    zz, tt = torch.meshgrid(z, theta, indexing='ij')
    radius = (winding + tt / (2 * torch.pi)) * dr
    spiral = torch.stack([zz, torch.sin(tt) * radius, torch.cos(tt) * radius], dim=-1)
    slice_to_spiral = model.get_slice_to_spiral_transform()
    zyxs = slice_to_spiral.inv(spiral.reshape(-1, 3).to(device))
    # Absolute winding in the fit's convention (get_patch_abs_winding_loss:
    # shifted_radius == winding * dr_per_winding): constant along the sheet --
    # the winding INDEX, not the radius parameter k + theta/2pi. Annotating the
    # latter injects a screw-dislocation error with mean +0.5 windings.
    annotation = torch.full_like(tt, float(winding))
    return (zyxs.reshape(num_rows, num_cols, 3).cpu().numpy().astype(np.float32),
            annotation.numpy().astype(np.float32))


@click.command()
@click.option('--phantom', required=True, type=click.Path(exists=True, file_okay=False),
              help='synth_phantom output directory.')
@click.option('--out', required=True, type=click.Path(file_okay=False),
              help='Dataset output directory.')
@click.option('--coverage', default=0.6, type=click.FloatRange(0., 1.),
              help='Fraction of each winding arc covered by patches.')
@click.option('--patch-arc-deg', default=90., type=float,
              help='Arc length of each individual patch, degrees.')
@click.option('--winding-stride', default=1, type=int,
              help='Export patches on every Nth winding.')
@click.option('--step', default=4., type=float,
              help='Patch grid spacing in voxels (both axes).')
@click.option('--position-noise', default=0., type=float,
              help='Gaussian vertex position noise std, voxels (label imprecision).')
@click.option('--winding-annotations/--no-winding-annotations', default=True,
              help='Write per-vertex absolute winding annotations (winding.tif).')
@click.option('--z-margin', default=2., type=float,
              help='Patch z inset from the volume faces, voxels.')
@click.option('--seed', default=0, type=int)
@click.option('--device', default='cpu')
def main(phantom, out, coverage, patch_arc_deg, winding_stride, step,
         position_noise, winding_annotations, z_margin, seed, device):
    rng = np.random.default_rng(seed)
    device = torch.device(device)
    checkpoint = load_checkpoint_cpu(os.path.join(phantom, 'phantom_checkpoint.pt'))
    model = build_model_from_phantom_checkpoint(checkpoint, device)
    meta = json.load(open(os.path.join(phantom, 'meta.json')))

    patches_dir = os.path.join(out, 'verified_patches')
    os.makedirs(patches_dir, exist_ok=True)
    export_umbilicus(checkpoint['umbilicus_zyx'], os.path.join(out, 'umbilicus.json'))

    z0, z1 = z_margin, meta['z_size'] - 1 - z_margin
    patch_arc = np.deg2rad(patch_arc_deg)
    num_patches = 0
    for winding in range(meta['first_winding'], meta['last_winding'] + 1, winding_stride):
        # Random patch starts along the arc until the coverage budget is spent;
        # gaps between patches are the realistic case (tracing never covers a
        # winding completely).
        budget = coverage * 2 * np.pi
        starts = []
        while budget > 0:
            arc = min(patch_arc, budget)
            starts.append((float(rng.uniform(0, 2 * np.pi - arc)), arc))
            budget -= patch_arc
        for theta0, arc in starts:
            zyxs, annotation = winding_surface_grid(
                model, winding, theta0, theta0 + arc, z0, z1, step, device)
            if position_noise > 0:
                zyxs = zyxs + rng.standard_normal(zyxs.shape).astype(np.float32) * position_noise
            uuid = f'phantom_w{winding:03d}_t{int(np.rad2deg(theta0)):03d}'
            save_tifxyz(zyxs, patches_dir, uuid, step_size=int(round(step)),
                        voxel_size_um=1.0, source='synth_phantom')
            if winding_annotations:
                tifffile.imwrite(os.path.join(patches_dir, uuid, 'winding.tif'),
                                 annotation)
            num_patches += 1

    with open(os.path.join(out, 'dataset_meta.json'), 'w') as f:
        json.dump({
            'phantom_dir': os.path.abspath(phantom),
            'phantom_meta': meta,
            'coverage': coverage, 'patch_arc_deg': patch_arc_deg,
            'winding_stride': winding_stride, 'step': step,
            'position_noise': position_noise,
            'winding_annotations': winding_annotations,
            'seed': seed, 'num_patches': num_patches,
        }, f, indent=2)
    click.echo(f'{num_patches} patches -> {patches_dir}')


if __name__ == '__main__':
    main()
