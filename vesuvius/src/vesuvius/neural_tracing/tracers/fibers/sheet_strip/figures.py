"""Sheet-strip figures: decisions (CT slices with the patch sheet, the predicted sheet, the tube and the path) and the
run's curves.

One row per decision:
  * the slice along the fiber through ``v = 0`` (crop axis f across, sheet normal u up);
  * cross-sections at three crop-axis positions (in-sheet v across, u up);
  * the absolute height error over the strip grid (f across, v up; blank where unsupervised).
Green: the patch sheet (supervised points); red: the prediction; the tube channel is tinted orange; cyan: the selected
path (planes ahead of the head). Units on the axes are voxels of the crop (9.6 um for Paris4).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from vesuvius.neural_tracing.tracers.shared.tensors import move_batch

CROSS_SECTIONS = (-24., 0., 24.)  # crop units along the axis


@torch.no_grad()
def plot_decisions(model, cpu_batch, run, path, step, rows=6):
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip.train import REFINE_ALL, with_tube
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    rows = min(rows, len(cpu_batch['hist']))
    batch = move_batch({k: (v[:rows] if torch.is_tensor(v) else {kk: vv[:rows] for kk, vv in v.items()})
                        for k, v in cpu_batch.items() if not k.startswith('_') and (torch.is_tensor(v) or isinstance(v, dict))}, device)
    x = with_tube(model, batch)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        output = model.training_forward(x, batch['hist'], batch['hmask'], batch['hist'].new_full((), REFINE_ALL))
    output = {k: v.float() if torch.is_floating_point(v) else v for k, v in output.items()}
    selected = model.select_prediction(output, run.training.confidence, run.training.n_commit)
    model.train(was_training)
    cfg, crop = model.cfg, model.cfg.fine
    image = x['fine'].float().cpu().numpy()
    heights = output['strip_heights'][:, -1].cpu().numpy()
    target, mask = batch['strip_target'].cpu().numpy(), batch['strip_mask'].cpu().numpy()
    path_points = selected['points'].cpu().numpy()
    f, v = model.strip_f.cpu().numpy(), model.strip_v.cpu().numpy()
    vox = lambda value: value/crop.spacing  # crop units -> samples (voxels)
    centre = (crop.width-1)/2
    fig, axes = plt.subplots(rows, 5, figsize=(22, 3.6*rows), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.5, 1, 1, 1, 1.1]))
    middle = len(v)//2
    for r in range(rows):
        ct, tube = image[r, 0], image[r, 1] if image.shape[1] > 1 else None
        full = bool(batch['tube_full'][r])
        # Along the fiber, v = 0: the two middle sample rows of the cross-section.
        h = (crop.width-1)/2
        along = ct[:, int(np.floor(h)), :]*.5+ct[:, int(np.ceil(h)), :]*.5  # (D, W): f, u
        ax = axes[r, 0]
        extent = (-crop.behind, crop.depth-1-crop.behind, -centre, centre)
        ax.imshow(along.T, cmap='gray', origin='lower', extent=extent, aspect='auto')
        if tube is not None:
            t = .5*tube[:, int(np.floor(h)), :]+.5*tube[:, int(np.ceil(h)), :]
            ax.imshow(np.ma.masked_less(t.T, .05), cmap='Oranges', origin='lower', extent=extent, aspect='auto',
                      alpha=.6, vmin=0, vmax=1)
        m = mask[r, :, middle]
        ax.plot(vox(f)[m], vox(target[r, :, middle])[m], 'g.-', lw=1.2, ms=3, label='patch')
        ax.plot(vox(f), vox(heights[r, :, middle]), 'r-', lw=1.2, label='predicted')
        if not full:
            ax.plot(vox(path_points[r, :, 2]), vox(path_points[r, :, 0]), 'c-', lw=1, label='path')
        ax.set_title(f'#{r} {"full fiber" if full else "observed trace"}, {int(batch["strip_patches"][r])} patches: '
                     f'along the fiber (v=0)', fontsize=9)
        ax.set_xlabel('f (vox)')
        ax.set_ylabel('u (vox)')
        if r == 0:
            ax.legend(fontsize=7, loc='upper right')
        for c, position in enumerate(CROSS_SECTIONS):
            k = int(np.argmin(np.abs(f-position)))
            d = int(round(vox(f[k])+crop.behind))
            ax = axes[r, 1+c]
            ext = (-centre, centre, -centre, centre)
            ax.imshow(ct[d].T, cmap='gray', origin='lower', extent=ext)  # (H, W) = v, u
            if tube is not None:
                ax.imshow(np.ma.masked_less(tube[d].T, .05), cmap='Oranges', origin='lower', extent=ext, alpha=.6,
                          vmin=0, vmax=1)
            m = mask[r, k]
            ax.plot(vox(v)[m], vox(target[r, k])[m], 'g.-', lw=1.2, ms=3)
            ax.plot(vox(v), vox(heights[r, k]), 'r-', lw=1.2)
            ax.set_title(f'cross-section f={vox(f[k]):+.0f}', fontsize=9)
            ax.set_xlabel('v (vox)')
        ax = axes[r, 4]
        error = np.where(mask[r], np.abs(heights[r]-target[r])/crop.spacing, np.nan)
        shown = ax.imshow(error.T, origin='lower', cmap='magma', vmin=0, vmax=6, aspect='auto',
                          extent=(vox(f[0]), vox(f[-1]), vox(v[0]), vox(v[-1])))
        plt.colorbar(shown, ax=ax, fraction=.04)
        supervised = mask[r].any()
        mae = np.nanmean(error) if supervised else np.nan
        plane = np.abs(target[r][mask[r]]).mean()/crop.spacing if supervised else np.nan
        ax.set_title(f'|error| vox: mean {mae:.2f} (plane {plane:.2f}), {mask[r].mean():.0%} supervised', fontsize=9)
        ax.set_xlabel('f (vox)')
        ax.set_ylabel('v (vox)')
    fig.suptitle(f'{run.name} update {step}: {path.stem.split("_")[0]} decisions', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def plot_curves(log_path, path):
    records = [json.loads(line) for line in open(log_path)]
    train = [r for r in records if r['kind'] == 'train']
    val = [r for r in records if r['kind'] == 'val']
    panels = (('strip MAE (vox)', ('strip_mae_vox', 'plane_mae_vox', 'full_strip_mae_vox', 'observed_strip_mae_vox')),
              ('within 2 vox', ('within2',)), ('loss', ('loss', 'strip', 'geometry', 'confidence')),
              ('path error (crop units)', ('path_error',)))
    fig, axes = plt.subplots(1, len(panels), figsize=(5*len(panels), 3.8))
    for ax, (title, keys) in zip(axes, panels):
        for key in keys:
            for kind, rows, style in (('train', train, '-'), ('val', val, 'o-')):
                points = [(r['step'], r[key]) for r in rows if r.get(key) is not None]
                if points:
                    ax.plot(*zip(*points), style, ms=3, lw=1, label=f'{kind} {key}')
        ax.set_title(title)
        ax.set_xlabel('update')
        ax.legend(fontsize=6)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
