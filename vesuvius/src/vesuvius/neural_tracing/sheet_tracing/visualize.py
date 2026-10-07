"""Diagnostic figures: one rollout per episode figure, and per-step metric curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DIST_MAX = 8.


def _ct(ax, img, extent, title):
    h, w = extent
    ax.imshow(img, cmap='gray', extent=(-w/2, w/2, h/2, -h/2), interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-w/2, w/2)
    ax.set_ylim(h/2, -h/2)


def episode_figure(record: dict, cfg, path: Path) -> None:
    """Sheet map of the rollout (GT patch u/v), CT along the advance, CT along the final front."""
    spec = record['spec']
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1., 1.2, 1.2]))
    ax = axes[0]
    valid = record['valid']
    u_hi = spec['u_lo']+spec['res']*(valid.shape[0]-1)
    v_hi = spec['v_lo']+spec['res']*(valid.shape[1]-1)
    ax.imshow(valid, cmap='Greys_r', vmin=-1, vmax=1.5, origin='lower', aspect='equal',
              extent=(spec['v_lo'], v_hi, spec['u_lo'], u_hi))
    to_vx = lambda uv: (spec['v_lo']+spec['res']*uv[..., 1], spec['u_lo']+spec['res']*uv[..., 0])
    for row in record['uv_seed']:
        v, u = to_vx(row)
        ax.plot(v, u, '-', color='tab:cyan', lw=1)
    thr = cfg.trace.confidence_threshold
    for t, row in enumerate(record['uv_rows']):
        v, u = to_vx(row)
        ax.plot(v, u, '-', color='0.6', lw=.6)
        conf = record['conf'][t] > thr
        dist = np.where(record['ok'][t], record['dist'][t], np.nan)
        sc = ax.scatter(v[conf], u[conf], c=dist[conf], cmap='plasma', vmin=0, vmax=DIST_MAX, s=14, marker='o')
        ax.scatter(v[~conf], u[~conf], c=dist[~conf], cmap='plasma', vmin=0, vmax=DIST_MAX, s=22, marker='x')
    fig.colorbar(sc, ax=ax, fraction=.046, label='on-sheet distance (vx)')
    meta = record['meta']
    ax.set_title(f"{meta['patch'][:40]}\nGT u/v (o confident, x stopped)", fontsize=8)
    ax.set_xlabel('v along front (vx)')
    ax.set_ylabel('u advance (vx)')

    _ct(axes[1], record['adv_img'], record['extents']['adv'], 'CT along advance (middle point): normal vs advance')
    g = record['adv_gt']
    axes[1].plot(g[:, 1], g[:, 0], '-', color='lime', lw=1, label='GT sheet')
    s = record['adv_seed']
    axes[1].plot(s[:, 1], s[:, 0], 'o', color='tab:cyan', ms=4, label='seed')
    tr = record['adv_trace']
    mid = cfg.data.front_points//2
    conf = record['conf'][:, mid] > thr
    axes[1].plot(tr[:, 1], tr[:, 0], '-', color='red', lw=.8)
    axes[1].plot(tr[conf, 1], tr[conf, 0], 'o', color='red', ms=4, label='trace (confident)')
    axes[1].plot(tr[~conf, 1], tr[~conf, 0], 'x', color='orange', ms=6, label='trace (stopped)')
    axes[1].legend(fontsize=7, loc='lower right')

    _ct(axes[2], record['front_img'], record['extents']['front'], 'CT across the final front: normal vs tangent')
    g = record['front_gt']
    axes[2].plot(g[:, 1], g[:, 0], '-', color='lime', lw=1)
    tr = record['front_trace']
    conf = record['front_conf'] > thr
    c = record['front_curve']
    axes[2].plot(c[:, 1], c[:, 0], '-', color='red', lw=.8)
    axes[2].plot(tr[conf, 1], tr[conf, 0], 'o', color='red', ms=4)
    axes[2].plot(tr[~conf, 1], tr[~conf, 0], 'x', color='orange', ms=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def metric_figure(runs: dict, cfg, path: Path) -> None:
    """Per-step curves for one or more named rollouts (e.g. two checkpoints on the same episodes)."""
    names = ('coverage', 'same_sheet', 'curve_same_sheet', 'confident', 'precision', 'caught', 'rejoined', 'dist_p90')
    fig, axes = plt.subplots(1, len(names), figsize=(3.2*len(names), 3))
    for name, per_step in runs.items():
        steps = np.arange(1, len(per_step['coverage'])+1)
        for ax, metric in zip(axes, names):
            ax.plot(steps, per_step[metric], '-o', ms=3, label=name)
    for ax, metric in zip(axes, names):
        ax.set_title(metric, fontsize=9)
        ax.set_xlabel('step')
        ax.grid(alpha=.3)
        if not metric.startswith('dist'):
            ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def training_figure(rows: list[dict], path: Path) -> None:
    """Training/validation curves from log rows."""
    train = [r for r in rows if r['kind'] == 'train']
    val = [r for r in rows if r['kind'] == 'val']
    keys = ('loss', 'dist', 'same_sheet', 'curve_same_sheet', 'confident', 'conf_precision', 'off_caught')
    fig, axes = plt.subplots(1, len(keys)+1, figsize=(3.2*(len(keys)+1), 3))
    for ax, k in zip(axes, keys):
        ax.plot([r['iteration'] for r in train], [r[k] for r in train], lw=.8)
        ax.set_title('train '+k, fontsize=9)
        ax.grid(alpha=.3)
    if val:
        for k in ('score_mean', 'coverage_mean', 'same_sheet_mean', 'caught_mean', 'kept_off_mean'):
            axes[-1].plot([r['iteration'] for r in val], [r[k] for r in val], '-o', ms=3, label=k)
        axes[-1].legend(fontsize=7)
        axes[-1].set_title('val', fontsize=9)
        axes[-1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
