#!/usr/bin/env python3
"""Render ink predictions for the `_spliced` winding meshes written by fit_spiral's
save_mesh, then tile their max-composites into horizontal strips.

Mesh folders are named `wXXX[_suffix]` (e.g. `w020_spliced_erode_r2`). Only the
`_spliced` variants are processed. For each, this runs vc_render_tifxyz to render
--num-slices layer tifs into a per-mesh `ink/` subfolder and max-composites them.
The per-winding composites are then concatenated horizontally (whole tifs, never
chopped) into images, renormalised to their 95th-percentile intensity, and written
as 8-bit jpgs to an `ink/` folder alongside the meshes, named after their winding
range, e.g. `w010-027.jpg`.

How many windings land in each image is a fixed function of the absolute winding
number (WINDINGS_PER_IMAGE below), not the rendered image widths, so the tiling is
reproducible run-to-run.
"""

import os
import re
import json
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# Number of windings packed into each successive image, starting from winding 0.
# After this fixed schedule is exhausted, every further winding gets its own image.
WINDINGS_PER_IMAGE = (18, 10, 8, 7, 6) + (5,) * 2 + (4,) * 5 + (3,) * 9 + (2,) * 20


def image_bin(widx, base):
    """Map a winding number to an image-tile index using WINDINGS_PER_IMAGE (then 1
    winding per image thereafter), counting from `base` (the first winding present
    in the data). Keying off the winding number keeps the tiling reproducible
    regardless of per-winding image widths."""
    offset = widx - base
    start = 0
    for bin_idx, size in enumerate(WINDINGS_PER_IMAGE):
        if offset < start + size:
            return bin_idx
        start += size
    return len(WINDINGS_PER_IMAGE) + (offset - start)


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
@click.option('--num-processes', '-j', type=int, default=1, show_default=True, help='Number of meshes to render concurrently')
def main(meshes_dir, volume, vc_render_bin, scale, group_idx, num_slices, num_processes):
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

    def render(widx, name):
        mesh_path = os.path.join(meshes_dir, name)
        per_mesh_ink = os.path.join(mesh_path, 'ink')
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
            return None
        return max_composite(tif_paths)

    composites = []  # (winding_idx, composite array)
    with ThreadPoolExecutor(max_workers=max(1, num_processes)) as pool:
        futures = {pool.submit(render, widx, name): (widx, name) for widx, name in meshes}
        for n, future in enumerate(as_completed(futures)):
            widx, name = futures[future]
            comp = future.result()
            if comp is None:
                print(f'[{n + 1}/{len(meshes)}] {name}: WARNING no tifs produced, skipping')
                continue
            composites.append((widx, comp))
            print(f'[{n + 1}/{len(meshes)}] {name}: rendered ink')

    if not composites:
        raise click.ClickException('No composites produced')

    # Group whole per-winding composites into images by their fixed schedule bin,
    # counted from the first winding present, so the tiling is reproducible.
    base = min(widx for widx, _ in composites)
    bins = {}  # image_bin -> [(winding_idx, composite array)]
    for widx, comp in composites:
        bins.setdefault(image_bin(widx, base), []).append((widx, comp))

    for bin_idx in sorted(bins):
        chunk = sorted(bins[bin_idx], key=lambda wc: wc[0])
        lo, hi = chunk[0][0], chunk[-1][0]
        out_path = os.path.join(collect_dir, f'w{lo:03d}-{hi:03d}.jpg')
        strip = np.concatenate([c for _, c in chunk], axis=1).astype(np.float32)
        p95 = np.percentile(strip, 95)
        strip = np.clip(strip / p95, 0, 1) * 255 if p95 > 0 else strip
        Image.fromarray(strip.astype(np.uint8)).save(out_path, quality=95)
        print(f'wrote {out_path} ({len(chunk)} windings, {strip.shape[1]}px wide, p95={p95:.1f})')

    print(f'Done. Strips in {collect_dir}')


if __name__ == '__main__':
    main()
