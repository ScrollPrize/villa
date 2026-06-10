#!/usr/bin/env python3
"""Render ink predictions for the `_spliced` winding meshes written by fit_spiral's
save_mesh, then tile their max-composites into horizontal strips.

Mesh folders are named `wXXX[_suffix]` (e.g. `w020_spliced_erode_r2`). Only the
`_spliced` variants are processed. For each, this runs vc_render_tifxyz to render
--num-slices layer tifs into a per-mesh `ink/` subfolder and max-composites them.
The per-winding composites are then concatenated horizontally (whole tifs, never
chopped) into chunks of about --chunk-width px, renormalised to their
95th-percentile intensity, and written as 8-bit jpgs to an `ink/` folder
alongside the meshes, named after their winding range, e.g. `w010-022.jpg`.
"""

import os
import re
import json
import glob
import subprocess

import click
import numpy as np
from PIL import Image


def is_tifxyz(path):
    meta_path = os.path.join(path, 'meta.json')
    if not os.path.isdir(path) or not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, 'r') as f:
            return json.load(f).get('format') == 'tifxyz'
    except Exception:
        return False


def winding_idx(name):
    """Parse the leading wXXX winding index from a mesh folder name, or None."""
    m = re.match(r'^w(\d+)', name)
    return int(m.group(1)) if m else None


def max_composite(tif_paths):
    composite = None
    for tif_path in tif_paths:
        layer = np.asarray(Image.open(tif_path))
        if composite is None:
            composite = layer
        else:
            composite = np.maximum(composite, layer)
    return composite


@click.command(help=__doc__)
@click.argument('meshes_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--volume', required=True, help='Ink volume zarr path')
@click.option('--vc-render-bin', default='vc_render_tifxyz', show_default=True, help='Path to the vc_render_tifxyz binary')
@click.option('--scale', type=float, default=0.25, show_default=True)
@click.option('--group-idx', type=int, default=1, show_default=True)
@click.option('--num-slices', type=int, default=5, show_default=True)
@click.option('--chunk-width', type=int, default=4000, show_default=True, help='Target width (px) per horizontal strip')
def main(meshes_dir, volume, vc_render_bin, scale, group_idx, num_slices, chunk_width):
    meshes = sorted(
        (winding_idx(name), name)
        for name in os.listdir(meshes_dir)
        if '_spliced' in name
        and winding_idx(name) is not None
        and is_tifxyz(os.path.join(meshes_dir, name))
    )
    if not meshes:
        raise click.ClickException(f'No _spliced tifxyz meshes found in {meshes_dir}')

    collect_dir = os.path.join(meshes_dir, 'ink')
    os.makedirs(collect_dir, exist_ok=True)

    print(f'Found {len(meshes)} _spliced mesh(es) in {meshes_dir}')
    composites = []  # (winding_idx, composite array)
    for i, (widx, name) in enumerate(meshes):
        mesh_path = os.path.join(meshes_dir, name)
        per_mesh_ink = os.path.join(mesh_path, 'ink')

        print(f'[{i + 1}/{len(meshes)}] {name}: rendering ink')
        os.makedirs(per_mesh_ink, exist_ok=True)
        subprocess.run([
            vc_render_bin,
            '--segmentation', mesh_path,
            '--scale', str(scale),
            '--group-idx', str(group_idx),
            '--volume', volume,
            '--tif-output', per_mesh_ink,
            '--num-slices', str(num_slices),
        ], check=True)
        tif_paths = sorted(glob.glob(os.path.join(per_mesh_ink, '*.tif')))

        if not tif_paths:
            print(f'[{i + 1}/{len(meshes)}] {name}: WARNING no tifs produced, skipping')
            continue
        composites.append((widx, max_composite(tif_paths)))

    if not composites:
        raise click.ClickException('No composites produced')

    # Greedily pack whole per-winding composites into horizontal strips of ~chunk_width px.
    chunk = []  # (winding_idx, composite array)
    chunk_w = 0

    def flush():
        if not chunk:
            return
        lo, hi = chunk[0][0], chunk[-1][0]
        out_path = os.path.join(collect_dir, f'w{lo:03d}-{hi:03d}.jpg')
        strip = np.concatenate([c for _, c in chunk], axis=1).astype(np.float32)
        p95 = np.percentile(strip, 95)
        strip = np.clip(strip / p95, 0, 1) * 255 if p95 > 0 else strip
        Image.fromarray(strip.astype(np.uint8)).save(out_path, quality=95)
        print(f'wrote {out_path} ({len(chunk)} windings, {chunk_w}px wide, p95={p95:.1f})')

    for widx, comp in composites:
        w = comp.shape[1]
        if chunk and chunk_w + w > chunk_width:
            flush()
            chunk, chunk_w = [], 0
        chunk.append((widx, comp))
        chunk_w += w
    flush()

    print(f'Done. Strips in {collect_dir}')


if __name__ == '__main__':
    main()
