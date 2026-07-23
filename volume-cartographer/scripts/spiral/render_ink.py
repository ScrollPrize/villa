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

The winding-range strip chunks described above are only built and rendered when --strips
is passed; by default they are skipped.

By default (unless --no-full-scroll), *all* windings are concatenated into a single
full-scroll tifxyz (concat/wLLL-HHH), which is then flattened with the lasagna forward
flattener (../../../lasagna/fit.py, config flatten_fast_nofilter.json plus a generated
overlay that points external_surfaces at the concat) rather than flatboi. The flattened
tifxyz is written as a top-level mesh at concat/wLLL-HHH_flat (meta.json + x/y/z.tif),
trimmed to its valid-cell bounding box with vc_tifxyz_trim (dropping the flatten's invalid
output-margin border, unless --no-full-scroll-trim), and ink-rendered. The rendered ink
strip is chopped (no downsampling) into fixed-width jpg tiles ink/wLLL-HHH_flat.NNN.jpg of
at most --max-strip-width px each.
"""

import os
import re
import sys
import json
import glob
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import numpy as np
from PIL import Image

# Full-scroll strips are legitimately enormous (hundreds of megapixels); disable PIL's
# decompression-bomb guard so max_composite can open them.
Image.MAX_IMAGE_PIXELS = None

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


def set_tifxyz_id(tifxyz_path, new_id):
    """Rewrite a tifxyz meta.json's uuid/name to new_id. lasagna's fit2tifxyz stamps them
    with the output folder name (`flatten.tifxyz`); we want the winding-range id instead."""
    meta_path = os.path.join(tifxyz_path, 'meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    meta['uuid'] = new_id
    meta['name'] = new_id
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def build_full_concat(meshes_dir, meshes, concat_dir, step_size, voxel_size_um):
    """Concatenate *every* _spliced winding mesh (in ascending winding order) into a
    single full-scroll tifxyz written to concat/, named `wLLL-HHH`. Returns
    (name, concat_path, width_px)."""
    ordered = sorted(meshes, key=lambda wc: wc[0])
    lo, hi = ordered[0][0], ordered[-1][0]
    name = f'w{lo:03d}-{hi:03d}'
    mesh_paths = [os.path.join(meshes_dir, nm) for _, nm in ordered]
    concat_zyxs = concat_meshes(mesh_paths)
    save_tifxyz(
        concat_zyxs,
        concat_dir,
        uuid=name,
        step_size=step_size,
        voxel_size_um=voxel_size_um,
        source=f'render_ink full-scroll concat {os.path.basename(meshes_dir.rstrip("/"))}',
    )
    return name, os.path.join(concat_dir, name), int(concat_zyxs.shape[1])


def lasagna_flatten(concat_path, flat_path, lasagna_dir, lasagna_config, lasagna_fit_script, device):
    """Flatten a (large, all-windings) concat tifxyz with the lasagna forward flattener and
    write the result as a top-level tifxyz at `flat_path` (meta.json + x/y/z.tif directly).

    Writes an overlay JSON alongside the concat that points `external_surfaces` at the
    concat, then runs `<lasagna_fit_script> <lasagna_config> <overlay> --out-dir <scratch>`
    from `lasagna_dir` (so fit.py's flat local imports resolve). fit.py's
    model-init=flatten path writes `<scratch>/tifxyz/flatten.tifxyz` (plus model
    checkpoints); we move that tifxyz up to `flat_path`, rewrite its meta id to the folder
    name, and drop the scratch dir. Returns `flat_path`.

    Config paths are passed as absolute .json positionals (cli_json treats any non-flag
    *.json arg as a config, merged left-to-right, so the overlay's external_surfaces is
    layered onto flatten_fast_nofilter.json). The external_surfaces path must be absolute
    because fit runs with cwd=lasagna_dir.
    """
    concat_abs = os.path.abspath(concat_path)
    flat_abs = os.path.abspath(flat_path)
    scratch = concat_abs + '_lasagna'
    overlay_path = concat_abs + '_lasagna_overlay.json'
    overlay = {'external_surfaces': [{'path': concat_abs}]}
    with open(overlay_path, 'w') as f:
        json.dump(overlay, f, indent=2)

    # fit2tifxyz writes into <scratch>/tifxyz/flatten.tifxyz; clear any stale run first.
    if os.path.exists(scratch):
        shutil.rmtree(scratch)

    # expandable_segments reduces CUDA fragmentation/reserved-but-unallocated waste, which
    # matters for the large full-scroll flatten on a memory-contended GPU.
    env = os.environ.copy()
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    subprocess.run(
        [sys.executable, lasagna_fit_script,
         os.path.abspath(lasagna_config), overlay_path,
         '--out-dir', scratch, '--device', device],
        check=True, cwd=lasagna_dir, env=env,
    )
    produced = os.path.join(scratch, 'tifxyz', 'flatten.tifxyz')
    if not is_tifxyz(produced):
        raise RuntimeError(f'lasagna flatten did not produce a tifxyz at {produced}')

    # Promote the flattened tifxyz to the top-level flat_path folder, stamp its id with the
    # folder name, and discard lasagna's scratch dir (model checkpoints, snapshots, overlay).
    if os.path.exists(flat_abs):
        shutil.rmtree(flat_abs)
    shutil.move(produced, flat_abs)
    set_tifxyz_id(flat_abs, os.path.basename(flat_abs))
    shutil.rmtree(scratch, ignore_errors=True)
    if os.path.exists(overlay_path):
        os.remove(overlay_path)
    return flat_abs


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
@click.option('--flatten-keep', type=float, default=6.25, show_default=True, help='Percent of points to keep when decimating the mesh flatboi flattens; <100 decimates (smaller mesh is less prone to SLIM divergence). Quantized to a per-axis stride, so kept fraction ~= 1/round(1/sqrt(keep/100))^2 (e.g. any keep<100 => stride>=2 => <=25%)')
@click.option('--flatten-iters', type=int, default=50, show_default=True, help='flatboi SLIM iterations')
@click.option('--flatten-energy', default='symmetric_dirichlet', show_default=True, help='flatboi energy (symmetric_dirichlet or conformal)')
@click.option('--flatten-tol', type=float, default=0.0, show_default=True, help='flatboi relative-energy early-stop tolerance (0 disables)')
@click.option('--flatten-inpaint/--no-flatten-inpaint', default=True, show_default=True, help='Inpaint invalid cells during tifxyz->obj so holes/padding do not break flattening')
@click.option('--flatboi-threads', type=int, default=32, show_default=True, help='Shared value for PASTIX_NUM_THREADS and OPENBLAS_NUM_THREADS passed to flatboi')
@click.option('--openblas-coretype', default='Haswell', show_default=True, help='Value for OPENBLAS_CORETYPE passed to flatboi')
@click.option('--strips/--no-strips', default=False, show_default=True, help='Build the per-winding-range strip concats (concat/wLLL-HHH), flatboi-flatten them (per --flatten), and ink-render them. Off by default: only the full-scroll concat is rendered')
@click.option('--full-scroll/--no-full-scroll', default=True, show_default=True, help='Concatenate ALL windings into one full-scroll mesh, flatten it with the lasagna forward flattener (not flatboi), and ink-render it')
@click.option('--max-strip-width', type=int, default=16384, show_default=True, help='Max width (px) of each saved ink jpg. Strips wider than this are chopped into <name>.NNN.jpg tiles of this width (no downsampling); narrower strips stay a single <name>.jpg')
@click.option('--full-scroll-trim/--no-full-scroll-trim', default=True, show_default=True, help='After the lasagna flatten, trim the flattened tifxyz to its valid-cell bounding box (removes the flatten output-margin border that renders as black bands) with vc_tifxyz_trim')
@click.option('--tifxyz-trim-bin', default='vc_tifxyz_trim', show_default=True, help='Path to the vc_tifxyz_trim binary (crops a tifxyz to its valid-cell bbox in place)')
@click.option('--lasagna-dir', default='', help='Path to the lasagna repo dir (holds fit.py). Default: <this script>/../../../lasagna')
@click.option('--lasagna-config', default='', help='Base lasagna flatten config json. Default: <lasagna-dir>/configs/flatten_fast_nofilter.json')
@click.option('--lasagna-fit-script', default='', help='Lasagna fit entrypoint run for the full-scroll flatten. Default: _run_flatten_threaded.py if present, else fit.py')
@click.option('--lasagna-device', default='cuda', show_default=True, help='--device passed to the lasagna flattener for the full-scroll flatten')
def main(meshes_dir, volume, vc_render_bin, scale, group_idx, num_slices, num_processes,
         flatten, flatboi_bin, tifxyz2obj_bin, obj2tifxyz_bin, uv_lift_bin, flatten_keep,
         flatten_iters, flatten_energy, flatten_tol, flatten_inpaint, flatboi_threads,
         openblas_coretype, strips, full_scroll, max_strip_width, full_scroll_trim,
         tifxyz_trim_bin, lasagna_dir, lasagna_config, lasagna_fit_script, lasagna_device):
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
    # from the first winding present, so the tiling is reproducible run-to-run. Skipped
    # unless --strips: by default only the full-scroll concat below is rendered.
    chunks = []  # (name, concat_mesh_path)
    if strips:
        base = min(widx for widx, _ in meshes)
        bins = {}  # image_bin -> [(winding_idx, name)]
        for widx, name in meshes:
            bins.setdefault(image_bin(widx, base), []).append((widx, name))

        # Concatenate the _spliced meshes in each chunk into a single tifxyz mesh covering
        # that winding range, written to concat/ and named after the range.
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

    # Optionally SLIM-flatten each strip concat (concurrently) and render the flattened
    # mesh instead. A chunk whose flattening fails falls back to rendering its unflattened
    # concat so the run still completes. Only relevant when --strips built chunks.
    if flatten and chunks:
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

    # Optionally build a single full-scroll concat over ALL windings and flatten it with
    # the lasagna forward flattener (a large mesh that flatboi does not handle well), then
    # render it alongside the per-chunk strips. Runs serially (one big GPU job) before the
    # render pool. If the flatten fails, the full-scroll render is skipped (not rendered
    # unflattened).
    full_items = []  # (name, mesh_path) rendered in addition to `chunks`
    if full_scroll:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        l_dir = os.path.abspath(lasagna_dir) if lasagna_dir else os.path.abspath(
            os.path.join(script_dir, '..', '..', '..', 'lasagna'))
        l_cfg = lasagna_config or os.path.join(l_dir, 'configs', 'flatten_fast_nofilter.json')
        if lasagna_fit_script:
            l_fit = lasagna_fit_script
        else:
            threaded = os.path.join(l_dir, '_run_flatten_threaded.py')
            l_fit = threaded if os.path.exists(threaded) else os.path.join(l_dir, 'fit.py')

        full_name, full_concat, width = build_full_concat(
            meshes_dir, meshes, concat_dir, step_size, voxel_size_um)
        print(f'wrote {full_concat} full-scroll concat '
              f'({len(meshes)} windings, {width}px wide tifxyz mesh)')
        print(f'Lasagna-flattening full-scroll concat via {l_fit} (config {l_cfg})')
        flat_name = full_name + '_flat'
        flat_dest = os.path.join(concat_dir, flat_name)
        # If the trim bin wasn't given explicitly, look for it next to --vc-render-bin
        # (both live in build/bin), so callers passing a full vc-render path get it free.
        trim_bin = tifxyz_trim_bin
        if trim_bin == 'vc_tifxyz_trim' and os.sep in vc_render_bin:
            sibling = os.path.join(os.path.dirname(vc_render_bin), 'vc_tifxyz_trim')
            if os.path.exists(sibling):
                trim_bin = sibling
        try:
            flat_path = lasagna_flatten(full_concat, flat_dest, l_dir, l_cfg, l_fit, lasagna_device)
            print(f'lasagna-flattened {full_name} -> {flat_path}')
        except subprocess.CalledProcessError as e:
            print(f'{full_name}: WARNING lasagna flatten failed ({e.cmd[0]} exit '
                  f'{e.returncode}), skipping full-scroll render')
            flat_path = None
        if flat_path is not None:
            # The flatten pads its output with an invalid-cell margin that renders as black
            # bands; trim the flattened tifxyz to its valid-cell bbox in place (VC3D tool).
            # A trim failure is non-fatal: fall back to rendering the untrimmed flatten.
            if full_scroll_trim:
                try:
                    subprocess.run([trim_bin, flat_path], check=True)
                    print(f'trimmed {flat_name} to valid bbox')
                except subprocess.CalledProcessError as e:
                    print(f'{flat_name}: WARNING vc_tifxyz_trim failed ({e.cmd[0]} exit '
                          f'{e.returncode}), rendering untrimmed')
            full_items.append((flat_name, flat_path))

    render_items = chunks + full_items

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
        futures = {pool.submit(render, name, cp): name for name, cp in render_items}
        for n, future in enumerate(as_completed(futures)):
            name = futures[future]
            comp = future.result()
            if comp is None:
                print(f'[{n + 1}/{len(render_items)}] {name}: WARNING no tifs produced, skipping')
                continue
            strip = comp.astype(np.float32)
            p95 = np.percentile(strip, 95)
            strip = np.clip(strip / p95, 0, 1) * 255 if p95 > 0 else strip
            strip8 = strip.astype(np.uint8)
            width = strip8.shape[1]
            # Chop strips wider than --max-strip-width into fixed-width jpg tiles
            # (<name>.NNN.jpg) without downsampling; narrower strips stay one <name>.jpg.
            if width <= max_strip_width:
                out_path = os.path.join(collect_dir, f'{name}.jpg')
                Image.fromarray(strip8).save(out_path, quality=95)
                print(f'[{n + 1}/{len(render_items)}] wrote {out_path} '
                      f'({width}px wide, p95={p95:.1f})')
            else:
                n_tiles = (width + max_strip_width - 1) // max_strip_width
                for t in range(n_tiles):
                    x0 = t * max_strip_width
                    x1 = min(width, x0 + max_strip_width)
                    tile_path = os.path.join(collect_dir, f'{name}.{t:03d}.jpg')
                    Image.fromarray(strip8[:, x0:x1]).save(tile_path, quality=95)
                print(f'[{n + 1}/{len(render_items)}] wrote {n_tiles} tiles '
                      f'{name}.000-{n_tiles - 1:03d}.jpg ({width}px wide total, '
                      f'p95={p95:.1f})')

    print(f'Done. Strips in {collect_dir}')


if __name__ == '__main__':
    main()
