#!/usr/bin/env python3
"""Concatenate the `_spliced` winding meshes written by fit_spiral's save_mesh into
winding-range chunks, then render each chunk's ink prediction into a horizontal strip.

Mesh folders are named `wXXX[_suffix]` (e.g. `w020_spliced_erode_r2`). Only the
`_spliced` variants are processed. They are first grouped into winding-range chunks by
a fixed schedule (WINDINGS_PER_IMAGE below) and each chunk's meshes are concatenated
(along the theta/width axis, whole meshes, never chopped) into a single tifxyz mesh,
written to a `concat/` folder alongside the meshes and named after its winding range,
e.g. `w010-027`. These let you load the geometry behind each ink strip as a single mesh.

Each concatenated mesh is then rendered with vc_render_tifxyz, which writes --num-slices
layer tifs into the chunk mesh's `ink/` subfolder. Those are max-composited, renormalised
to their 95th-percentile intensity, and written as an 8-bit jpg to an `ink/` folder
alongside the meshes, named after the same winding range, e.g. `w010-027.jpg`.

How many windings land in each chunk is a fixed function of the absolute winding number
(WINDINGS_PER_IMAGE below), not the rendered image widths, so the tiling is reproducible
run-to-run.
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

from tifxyz import load_tifxyz, save_tifxyz


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


def read_step_and_voxel(meta_path):
    """Recover the step_size and voxel_size_um that save_tifxyz encoded into a mesh's
    meta.json, so a concatenated mesh can be written with matching metadata."""
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    step_size = 1.0 / meta['scale'][0]
    voxel_size_um = (meta['area_cm2'] / meta['area_vx2'] * 1.e8) ** 0.5
    return step_size, voxel_size_um


def concat_meshes(mesh_paths):
    """Load the given tifxyz meshes and concatenate their zyxs grids along the
    theta/width axis (axis=1), padding shorter grids with the -1 invalid sentinel so
    heights match. Returns the combined zyxs array in zyx order."""
    grids = [load_tifxyz(p).zyxs.cpu().numpy() for p in mesh_paths]
    max_h = max(g.shape[0] for g in grids)
    padded = [
        g if g.shape[0] == max_h
        else np.pad(g, ((0, max_h - g.shape[0]), (0, 0), (0, 0)), constant_values=-1.0)
        for g in grids
    ]
    return np.concatenate(padded, axis=1).astype(np.float32)


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

    concat_dir = os.path.join(meshes_dir, 'concat')
    os.makedirs(concat_dir, exist_ok=True)
    step_size, voxel_size_um = read_step_and_voxel(
        os.path.join(meshes_dir, meshes[0][1], 'meta.json')
    )

    print(f'Found {len(meshes)} _spliced mesh(es) in {meshes_dir}')

    # Group the meshes into winding-range chunks by their fixed schedule bin, counted
    # from the first winding present, so the tiling is reproducible run-to-run.
    base = min(widx for widx, _ in meshes)
    bins = {}  # image_bin -> [(winding_idx, name)]
    for widx, name in meshes:
        bins.setdefault(image_bin(widx, base), []).append((widx, name))

    # Concatenate the _spliced meshes in each chunk into a single tifxyz mesh covering
    # that winding range, written to concat/ and named after the range.
    chunks = []  # (name, concat_mesh_path)
    for bin_idx in sorted(bins):
        chunk = sorted(bins[bin_idx], key=lambda wc: wc[0])
        lo, hi = chunk[0][0], chunk[-1][0]
        name = f'w{lo:03d}-{hi:03d}'
        mesh_paths = [os.path.join(meshes_dir, nm) for _, nm in chunk]
        concat_zyxs = concat_meshes(mesh_paths)
        save_tifxyz(
            concat_zyxs,
            concat_dir,
            uuid=name,
            step_size=step_size,
            voxel_size_um=voxel_size_um,
            source=f'render_ink concat {os.path.basename(meshes_dir.rstrip("/"))}',
        )
        concat_path = os.path.join(concat_dir, name)
        chunks.append((name, concat_path))
        print(f'wrote {concat_path} '
              f'({len(chunk)} windings, {concat_zyxs.shape[1]}px wide tifxyz mesh)')

    # Render ink from each concatenated mesh and max-composite it into a strip jpg.
    def render(name, concat_path):
        per_mesh_ink = os.path.join(concat_path, 'ink')
        os.makedirs(per_mesh_ink, exist_ok=True)
        subprocess.run([
            vc_render_bin,
            '--segmentation', concat_path,
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

    with ThreadPoolExecutor(max_workers=max(1, num_processes)) as pool:
        futures = {pool.submit(render, name, cp): name for name, cp in chunks}
        for n, future in enumerate(as_completed(futures)):
            name = futures[future]
            comp = future.result()
            if comp is None:
                print(f'[{n + 1}/{len(chunks)}] {name}: WARNING no tifs produced, skipping')
                continue
            out_path = os.path.join(collect_dir, f'{name}.jpg')
            strip = comp.astype(np.float32)
            p95 = np.percentile(strip, 95)
            strip = np.clip(strip / p95, 0, 1) * 255 if p95 > 0 else strip
            Image.fromarray(strip.astype(np.uint8)).save(out_path, quality=95)
            print(f'[{n + 1}/{len(chunks)}] wrote {out_path} '
                  f'({strip.shape[1]}px wide, p95={p95:.1f})')

    print(f'Done. Strips in {collect_dir}')


if __name__ == '__main__':
    main()
