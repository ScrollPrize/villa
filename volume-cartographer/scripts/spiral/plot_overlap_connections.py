#!/usr/bin/env python3
"""Debug plots for connect_overlapping_patches output.

For each synthesised connecting point-collection in ``--collection-output``,
draw XY / YZ / XZ scatter projections of both named patches' valid vertices
together with the connecting pcl points (numbered, in path order, with start/end
marked). One PNG per collection is written to ``--out-dir`` (default: a folder
``overlap_debug`` next to the collection JSON), so the creation can be eyeballed
for whether the points really lie on the two named patches.
"""

import json
import os

import click
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tifxyz import load_tifxyz


def patch_xyz(patch):
    """Valid vertices of a patch as (N, 3) world xyz (Patch stores zyx)."""
    return patch.valid_zyxs.detach().cpu().numpy()[:, ::-1].copy()


def parse_patch_ids(name):
    """Extract (a_id, b_id) from a ``between_patches__{a}__{b}`` collection name."""
    core = name.split('between_patches__', 1)[-1]  # strip the prefix if present
    return tuple(core.split('__', 1)) if '__' in core else (core, '')


def _panel(ax, h, v, A, B, P, labels):
    """Scatter one projection: A, B patch points (faint) + pcl points (path)."""
    ax.scatter(A[:, h], A[:, v], s=2, c='tab:blue', alpha=0.25, linewidths=0, label='patch A')
    ax.scatter(B[:, h], B[:, v], s=2, c='tab:orange', alpha=0.25, linewidths=0, label='patch B')
    if len(P):
        ax.plot(P[:, h], P[:, v], '-', c='red', lw=0.8, alpha=0.7)
        ax.scatter(P[:, h], P[:, v], s=22, c='red', edgecolors='k', linewidths=0.4, zorder=5, label='pcl points')
        ax.scatter(P[0, h], P[0, v], s=90, marker='*', c='lime', edgecolors='k', zorder=6, label='start')
        ax.scatter(P[-1, h], P[-1, v], s=90, marker='*', c='magenta', edgecolors='k', zorder=6, label='end')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, ls=':', alpha=0.3)


def plot_collection(cid, name, A, B, P, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _panel(axes[0], 0, 1, A, B, P, ('x', 'y'))  # XY
    _panel(axes[1], 1, 2, A, B, P, ('y', 'z'))  # YZ
    _panel(axes[2], 0, 2, A, B, P, ('x', 'z'))  # XZ
    a_id, b_id = parse_patch_ids(name)
    fig.suptitle(f'collection {cid}   ({len(P)} pcl pts)\n'
                 f'A (blue) = {a_id}\nB (orange) = {b_id}', fontsize=10)
    axes[0].legend(loc='best', fontsize=7, framealpha=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


@click.command(help=__doc__)
@click.argument('patches_path', type=click.Path(exists=True, file_okay=False))
@click.option('--collection-output', '-c', type=click.Path(dir_okay=False, exists=True),
              default='overlap_connections.json', show_default=True)
@click.option('--pairs-output', '-o', type=click.Path(dir_okay=False, exists=True),
              default='overlap_pairs.jsonl', show_default=True)
@click.option('--out-dir', type=click.Path(file_okay=False), default=None,
              help='where to write PNGs (default: overlap_debug next to the collection JSON)')
@click.option('--limit', type=int, default=None, help='plot at most this many collections')
def main(patches_path, collection_output, pairs_output, out_dir, limit):
    coll = json.load(open(collection_output))['collections']
    rows = [json.loads(l) for l in open(pairs_output)]
    by_cid = {r['collection_id']: r for r in rows if r.get('collection_id') is not None}

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(collection_output)), 'overlap_debug')
    os.makedirs(out_dir, exist_ok=True)

    patch_cache = {}

    def get_xyz(pid):
        if pid not in patch_cache:
            patch_cache[pid] = patch_xyz(load_tifxyz(os.path.join(patches_path, pid)))
        return patch_cache[pid]

    items = sorted(coll.items(), key=lambda kv: int(kv[0]))
    if limit is not None:
        items = items[:limit]

    for cid, c in items:
        r = by_cid.get(int(cid))
        a_id, b_id = (r['a'], r['b']) if r else parse_patch_ids(c['name'])
        A, B = get_xyz(a_id), get_xyz(b_id)
        P = np.array([p['p'] for p in c['points'].values()], dtype=np.float32).reshape(-1, 3)
        out_path = os.path.join(out_dir, f'c{int(cid):03d}.png')
        plot_collection(cid, c['name'], A, B, P, out_path)
        print(f'wrote {out_path}  (A={len(A)} B={len(B)} pcl={len(P)})')

    print(f'\nwrote {len(items)} debug plots to {out_dir}')


if __name__ == '__main__':
    main()
