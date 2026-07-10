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
import shutil
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


def flatten_mesh(concat_path, tifxyz2obj_bin, flatboi_bin, obj2tifxyz_bin, uv_lift_bin,
                 iters, energy, tol, inpaint, keep, flatboi_env):
    """SLIM-flatten a concatenated tifxyz mesh and return the path to a new
    flattened tifxyz mesh (a sibling `<name>_flat` folder).

    Mirrors VC3D's SLIM-flatten pipeline (SegmentationCommandHandler). With keep >= 100
    (no decimation) the mesh is flattened at full resolution:
      1. vc_tifxyz2obj <mesh> <obj>             -> OBJ with grid UVs
      2. flatboi <obj> <iters> <energy> [--tol] -> writes <stem>_flatboi.obj alongside
      3. vc_obj2tifxyz <flatObj> <out> --tifxyz-source=<mesh>

    With keep < 100, flatboi runs on a decimated (coarse) mesh — smaller meshes are
    less likely to diverge (NaN energy) — and its UVs are lifted back onto the full-res
    mesh before conversion, matching VC3D's decimating branch:
      1. vc_tifxyz2obj <mesh> <coarse.obj> --keep=<p>  -> coarse OBJ with grid UVs
      2. vc_tifxyz2obj <mesh> <fine.obj>               -> full-res OBJ (lift target)
      3. flatboi <coarse.obj> ...                      -> <coarse>_flatboi.obj
      4. vc_obj_uv_lift <coarse.obj> <coarse_flatboi.obj> <fine.obj> <lifted.obj>
      5. vc_obj2tifxyz <lifted.obj> <out> --tifxyz-source=<mesh>

    flatboi_env is a dict of env overrides (PASTIX/OPENBLAS threads, coretype) that
    is layered onto the current environment for the flatboi step only.
    """
    concat_dir = os.path.dirname(concat_path)
    name = os.path.basename(concat_path)
    decimating = keep < 100.0
    obj_path = os.path.join(concat_dir, name + ('_coarse.obj' if decimating else '.obj'))
    flat_obj = os.path.join(concat_dir, name + ('_coarse_flatboi.obj' if decimating else '_flatboi.obj'))
    flat_path = os.path.join(concat_dir, name + '_flat')

    # 1. tifxyz -> obj (grid UVs). --inpaint fills isolated invalid cells so the
    #    concat's padding/holes don't break flatboi's boundary loop. --keep decimates
    #    the mesh flatboi sees (a smaller mesh is less prone to SLIM divergence).
    to_obj_args = [tifxyz2obj_bin, concat_path, obj_path]
    if decimating:
        to_obj_args.append(f'--keep={keep:.4f}')
    if inpaint:
        to_obj_args.append('--inpaint')
    subprocess.run(to_obj_args, check=True, capture_output=True, text=True)

    # 1b. When decimating, also emit the full-res OBJ that the flattened coarse UVs
    #     get lifted onto (so the flattened tifxyz keeps full input resolution).
    fine_obj = os.path.join(concat_dir, name + '.obj')
    if decimating:
        fine_args = [tifxyz2obj_bin, concat_path, fine_obj]
        if inpaint:
            fine_args.append('--inpaint')
        subprocess.run(fine_args, check=True, capture_output=True, text=True)

    # 2. flatboi (SLIM). Writes <stem>_flatboi.obj next to the input obj.
    flatboi_args = [flatboi_bin, obj_path, str(iters), energy]
    if tol and tol > 0:
        flatboi_args.append(f'--tol={tol:g}')
    env = os.environ.copy()
    env.update(flatboi_env)
    subprocess.run(flatboi_args, check=True, capture_output=True, text=True, env=env)

    # 3. When decimating, lift the flattened coarse UVs onto the full-res mesh
    #    (grid-space lift via the grid UVs vc_tifxyz2obj wrote into each OBJ).
    if decimating:
        lifted_obj = os.path.join(concat_dir, name + '_lifted.obj')
        subprocess.run(
            [uv_lift_bin, obj_path, flat_obj, fine_obj, lifted_obj],
            check=True, capture_output=True, text=True,
        )
        src_obj = lifted_obj
    else:
        src_obj = flat_obj

    # 4. flattened obj -> tifxyz. vc_obj2tifxyz requires the target NOT to exist;
    #    --tifxyz-source sizes the output grid to the input sampling density.
    if os.path.exists(flat_path):
        shutil.rmtree(flat_path)
    subprocess.run([
        obj2tifxyz_bin, src_obj, flat_path,
        f'--tifxyz-source={concat_path}',
    ], check=True, capture_output=True, text=True)

    return flat_path


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
@click.option('--num-processes', '-j', type=int, default=1, show_default=True, help='Number of meshes to render (and flatten) concurrently')
@click.option('--flatten/--no-flatten', default=True, show_default=True, help='SLIM-flatten each concatenated mesh before rendering')
@click.option('--flatboi-bin', default='flatboi', show_default=True, help='Path to the flatboi binary')
@click.option('--tifxyz2obj-bin', default='vc_tifxyz2obj', show_default=True, help='Path to the vc_tifxyz2obj binary')
@click.option('--obj2tifxyz-bin', default='vc_obj2tifxyz', show_default=True, help='Path to the vc_obj2tifxyz binary')
@click.option('--uv-lift-bin', default='vc_obj_uv_lift', show_default=True, help='Path to the vc_obj_uv_lift binary (used only when --flatten-keep < 100)')
@click.option('--flatten-keep', type=float, default=100.0, show_default=True, help='Percent of points to keep when decimating the mesh flatboi flattens; <100 decimates (smaller mesh is less prone to SLIM divergence). Quantized to a per-axis stride, so kept fraction ~= 1/round(1/sqrt(keep/100))^2 (e.g. any keep<100 => stride>=2 => <=25%)')
@click.option('--flatten-iters', type=int, default=50, show_default=True, help='flatboi SLIM iterations')
@click.option('--flatten-energy', default='symmetric_dirichlet', show_default=True, help='flatboi energy (symmetric_dirichlet or conformal)')
@click.option('--flatten-tol', type=float, default=0.0, show_default=True, help='flatboi relative-energy early-stop tolerance (0 disables)')
@click.option('--flatten-inpaint/--no-flatten-inpaint', default=True, show_default=True, help='Inpaint invalid cells during tifxyz->obj so holes/padding do not break flattening')
@click.option('--flatboi-threads', type=int, default=32, show_default=True, help='Shared value for PASTIX_NUM_THREADS and OPENBLAS_NUM_THREADS passed to flatboi')
@click.option('--openblas-coretype', default='Haswell', show_default=True, help='Value for OPENBLAS_CORETYPE passed to flatboi')
def main(meshes_dir, volume, vc_render_bin, scale, group_idx, num_slices, num_processes,
         flatten, flatboi_bin, tifxyz2obj_bin, obj2tifxyz_bin, uv_lift_bin, flatten_keep,
         flatten_iters, flatten_energy, flatten_tol, flatten_inpaint, flatboi_threads,
         openblas_coretype):
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

    # Optionally SLIM-flatten each concatenated mesh (concurrently) and render the
    # flattened mesh instead. A chunk whose flattening fails falls back to rendering
    # its unflattened concat so the run still completes.
    if flatten:
        flatboi_env = {}
        if flatboi_threads is not None:
            flatboi_env['PASTIX_NUM_THREADS'] = str(flatboi_threads)
            flatboi_env['OPENBLAS_NUM_THREADS'] = str(flatboi_threads)
        if openblas_coretype:
            flatboi_env['OPENBLAS_CORETYPE'] = openblas_coretype

        def flatten_one(name, concat_path):
            return name, flatten_mesh(
                concat_path, tifxyz2obj_bin, flatboi_bin, obj2tifxyz_bin, uv_lift_bin,
                flatten_iters, flatten_energy, flatten_tol, flatten_inpaint, flatten_keep,
                flatboi_env,
            )

        print(f'Flattening {len(chunks)} mesh(es) with flatboi (iters={flatten_iters}, '
              f'energy={flatten_energy}, j={num_processes})')
        cp_by_name = dict(chunks)
        flattened = {}
        with ThreadPoolExecutor(max_workers=max(1, num_processes)) as pool:
            futures = {pool.submit(flatten_one, name, cp): name for name, cp in chunks}
            for n, future in enumerate(as_completed(futures)):
                name = futures[future]
                try:
                    _, flat_path = future.result()
                    flattened[name] = flat_path
                    print(f'[{n + 1}/{len(chunks)}] flattened {name} -> {flat_path}')
                except subprocess.CalledProcessError as e:
                    print(f'[{n + 1}/{len(chunks)}] {name}: WARNING flatten failed '
                          f'({e.cmd[0]} exit {e.returncode}), rendering unflattened concat')
                    if e.stderr:
                        print(e.stderr.strip())
        # Swap in flattened paths where available; keep concat for any that failed.
        chunks = [(name, flattened.get(name, cp_by_name[name])) for name, _ in chunks]

    # Render ink from each (flattened, if enabled) mesh and max-composite it into a strip jpg.
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
