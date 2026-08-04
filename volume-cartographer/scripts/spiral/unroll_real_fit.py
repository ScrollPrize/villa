"""Render a fitted real-scroll reroll: winding-curve overlay on the mid
slice, and the unrolled strips (fitted vs unfitted-circles baseline). Part 3
of the reproducible demo sequence documented in trace_real_windings.py.

The slab fetch parameters must match the trace run (they are read from the
dataset's dataset_meta.json when --dataset is given, which is the reliable
way). If the trace mirrored x, unrolled strips are re-flipped for display.
"""

import json
import os

import click
import numpy as np
import scipy.ndimage
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from checkpoint_io import load_checkpoint_cpu
from synth_phantom import build_model, build_model_from_phantom_checkpoint
from trace_real_windings import fetch_slab


@click.command()
@click.option('--checkpoint', required=True, type=click.Path(exists=True, dir_okay=False),
              help='fit_phantom_reference output for the traced dataset.')
@click.option('--dataset', required=True, type=click.Path(exists=True, file_okay=False),
              help='trace_real_windings output dir (slab source + mirror flag).')
@click.option('--slab-cache', default=None, type=click.Path(dir_okay=False))
@click.option('--out-prefix', required=True,
              help='Writes <prefix>_overlay.png and <prefix>_unrolled.png.')
@click.option('--k-lo', default=8, type=int)
@click.option('--k-hi', default=36, type=int)
@click.option('--k-step', default=4, type=int)
def main(checkpoint, dataset, slab_cache, out_prefix, k_lo, k_hi, k_step):
    meta = json.load(open(os.path.join(dataset, 'dataset_meta.json')))
    src = meta['source']
    slab = fetch_slab(src['volume_url'], src['level'], src['z0'],
                      src['z_size'], slab_cache)
    if src['mirror_x']:
        slab = slab[:, :, ::-1].copy()
    Z = slab.shape[0]

    ckpt = load_checkpoint_cpu(checkpoint)
    fitted = build_model_from_phantom_checkpoint(ckpt, torch.device('cpu'))
    init = build_model(ckpt['cfg'], 0, Z, ckpt['umbilicus_zyx'],
                       ckpt['spiral_outward_sense'], torch.device('cpu'))
    gauge = float(ckpt['reference_fit']['gauge'])
    dr = float(fitted.get_dr_per_winding())
    click.echo(f'fitted dr {dr:.3f}, gauge {gauge:+.3f}')

    def unroll(model, gauge_off):
        tt = torch.tensor(np.deg2rad(np.arange(0, 360, 0.5)), dtype=torch.float32)
        spiral_to_slice = model.get_slice_to_spiral_transform().inv
        strips = []
        for k in range(k_lo, k_hi + 1, k_step):
            radius = (k + gauge_off + tt / (2 * np.pi)) * dr
            rows = []
            for z in range(Z):
                pts = torch.stack([torch.full_like(tt, float(z)),
                                   torch.sin(tt) * radius,
                                   torch.cos(tt) * radius], dim=-1)
                with torch.no_grad():
                    sp = spiral_to_slice(pts).numpy()
                rows.append(scipy.ndimage.map_coordinates(
                    slab, [sp[:, 0], sp[:, 1], sp[:, 2]], order=1, cval=0.))
            strips.append(np.array(rows))
        return strips

    fitted_strips = unroll(fitted, gauge)
    init_strips = unroll(init, 0.)
    sep = np.full((4, fitted_strips[0].shape[1]), 255.)

    def montage(strips):
        rows = []
        for s in strips:
            rows += [s, sep]
        return np.concatenate(rows[:-1], axis=0)

    flip = -1 if src['mirror_x'] else 1
    fig, axes = plt.subplots(2, 1, figsize=(20, 14))
    axes[0].imshow(montage(init_strips)[:, ::flip], cmap='gray', aspect='auto')
    axes[0].set_title(f'UNFITTED baseline (concentric circles): windings {k_lo}..{k_hi} step {k_step}')
    axes[1].imshow(montage(fitted_strips)[:, ::flip], cmap='gray', aspect='auto')
    axes[1].set_title('FITTED reroll: same windings')
    for a in axes:
        a.axis('off')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_unrolled.png', dpi=110)

    fig2, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(slab[Z // 2], cmap='gray')
    tt = torch.tensor(np.deg2rad(np.arange(0, 360, 1.)), dtype=torch.float32)
    spiral_to_slice = fitted.get_slice_to_spiral_transform().inv
    for k in range(k_lo, k_hi + 1, 2):
        radius = (k + gauge + tt / (2 * np.pi)) * dr
        pts = torch.stack([torch.full_like(tt, float(Z // 2)),
                           torch.sin(tt) * radius, torch.cos(tt) * radius], dim=-1)
        with torch.no_grad():
            sp = spiral_to_slice(pts).numpy()
        ax.plot(sp[:, 2], sp[:, 1], lw=0.6)
    ax.set_title('fitted winding surfaces over mid slice (QC; mirrored frame)')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_overlay.png', dpi=110)
    click.echo(f'wrote {out_prefix}_unrolled.png, {out_prefix}_overlay.png')


if __name__ == '__main__':
    main()
