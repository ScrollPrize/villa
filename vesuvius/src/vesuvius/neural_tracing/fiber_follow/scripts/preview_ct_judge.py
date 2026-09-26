"""Reproducible real-CT examples for direct/CT_SLICE_JUDGE_PLAN.md.

This is a plan illustration, not a trained judge. Reuses the production fiber
parser, frames, block bounds, CT reader and trilinear sampler. Downloads, when
explicitly requested, are limited to interpolation-support chunks for the views.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vesuvius.neural_tracing.fiber_follow.data import load_fibers, tight_block
from vesuvius.neural_tracing.fiber_follow.fast_sample import sample_crop
from vesuvius.neural_tracing.fiber_follow.geometry import (
    CropSpec, crop_local_grid, frame_from_heading, interp_at,
)
from vesuvius.neural_tracing.fiber_follow.volume import ChunkedArray

LOCAL_CT = '/mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr/0'
FIBER = '/mnt/raid_nvme/spiral_dataset_working/fibers/anon_20260815T014816946_000002.json'


def views(fiber, start, count, step, pixels, spacing):
    """Frame at each position depends only on the preceding four path voxels."""
    arcs = start + np.arange(count)*step
    if start < 4 or arcs[-1] > fiber.length:
        raise ValueError('Requested sequence must lie inside the annotated path')
    centers = interp_at(fiber.points, fiber.s, arcs)
    from vesuvius.neural_tracing.fiber_follow.direct.judge_slices import transported_frames
    seed = interp_at(fiber.points, fiber.s, np.array([0., min(1., fiber.length)]))
    frames = transported_frames(fiber.points, frame_from_heading(seed[1]-seed[0]), arcs)
    crop = CropSpec(depth=1, width=pixels, behind=0, spacing=spacing)
    grid = crop_local_grid(crop).reshape(-1, 3)
    records = []
    for arc, center, frame in zip(arcs, centers, frames):
        u, v, f = frame.T
        for name, axes in [('uv', (u, v, f)), ('uf', (u, f, -v)), ('vf', (v, f, u))]:
            records.append(dict(arc=float(arc), center=center, frame=np.stack(axes, axis=1), view=name))
    return records, crop, grid


def needed_keys(record, grid, scale, meta):
    """All chunks that contain trilinear support, conservatively including ties."""
    xyz = (record['center'] + grid @ record['frame'].T)*scale
    low = np.floor(xyz[:, ::-1]).astype(np.int64)
    shape, chunks = np.asarray(meta['shape']), np.asarray(meta['chunks'])
    keys = set()
    for offset in itertools.product((0, 1), repeat=3):
        idx = low + offset
        valid = ((idx >= 0) & (idx < shape)).all(1)
        keys.update(map(tuple, np.unique(idx[valid]//chunks, axis=0).tolist()))
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fiber-json', default=FIBER)
    ap.add_argument('--ct-array', help='Local CT array (default: follower s1_ds2.zarr/0)')
    ap.add_argument('--ct-grid-scale', type=float, default=4.)
    ap.add_argument('--source', help='Explicit remote zarr root, only used with --download-native')
    ap.add_argument('--download-native', action='store_true')
    ap.add_argument('--out', default='direct/ct_slice_judge_examples')
    ap.add_argument('--start', type=float, default=200.)
    ap.add_argument('--count', type=int, default=9)
    ap.add_argument('--step', type=float, default=4.)
    ap.add_argument('--pixels', type=int, default=257)
    ap.add_argument('--spacing', type=float, help='Default: one source CT voxel per pixel')
    args = ap.parse_args()
    if args.download_native and (not args.source or not args.ct_array):
        ap.error('--download-native requires an explicit --source and --ct-array cache destination')
    args.ct_array = args.ct_array or LOCAL_CT
    from vesuvius.neural_tracing.fiber_follow.direct.judge_slices import SliceConfig, sample_planes
    cfg = SliceConfig(source=args.ct_array, grid_scale=args.ct_grid_scale, pixels=args.pixels, spacing=args.spacing)
    args.spacing = cfg.spacing
    if args.count < 1 or args.step <= 0 or args.pixels < 3 or args.pixels % 2 != 1 or args.spacing <= 0:
        ap.error('Use positive sampling settings and an odd image width >= 3')
    with tempfile.TemporaryDirectory(prefix='ct-judge-fiber-') as folder:
        shutil.copy2(args.fiber_json, folder)
        fiber, = load_fibers(folder)
    records, crop, grid = views(fiber, args.start, args.count, args.step, args.pixels, args.spacing)
    # Coordinates in fiber JSON are base-grid xyz; one trace voxel = 8 base voxels.
    scale = cfg.trace_scale/cfg.grid_scale
    root = Path(args.ct_array)
    if args.download_native:
        root.mkdir(parents=True, exist_ok=True)
        metadata = urllib.request.urlopen(args.source.rstrip('/')+'/0/.zarray', timeout=30).read()
        (root/'.zarray').write_bytes(metadata)
    meta = json.loads((root/'.zarray').read_text())
    if len(meta['shape']) != 3 or meta['dtype'] != '|u1':
        raise ValueError('Preview expects an aligned three-dimensional uint8 CT source')
    keys_by_view = [needed_keys(r, grid, scale, meta) for r in records]
    keys = sorted(set().union(*keys_by_view))
    sep = meta.get('dimension_separator', '.')
    names = [sep.join(map(str, k)) for k in keys]
    if len(names) > 160:
        raise ValueError(f'Request needs {len(names)} chunks; preview download cap is 160')
    missing = [name for name in names if not (root/name).is_file()]
    print(f'{len(records)} views; {len(names)} source chunks; {len(missing)} missing', flush=True)
    if missing and not args.download_native:
        raise ValueError('Source coverage incomplete; use --download-native to fetch required chunks')
    def fetch(name):
        raw = urllib.request.urlopen(args.source.rstrip('/')+'/0/'+name, timeout=60).read()
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return len(raw)
    if missing:
        with ThreadPoolExecutor(max_workers=6) as pool:
            sizes = list(pool.map(fetch, missing))
        print(f'Downloaded {sum(sizes)/2**20:.1f} MiB into {root}', flush=True)
    reader = ChunkedArray(root, cache_bytes=256 << 20)
    images = []
    for r in records[::3]:
        planes = sample_planes(reader, r['center'], r['frame'], cfg)
        if not planes[:, 2].all():
            raise ValueError('Missing source support')
        images.extend(planes[:, 0])
    images = np.asarray(images).reshape(args.count, 3, args.pixels, args.pixels)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:args.pixels, :args.pixels] - args.pixels//2
    sigma = 2/(scale*args.spacing)
    marker = np.exp(-(xx*xx+yy*yy)/(2*sigma**2)).astype(np.float32)
    # Training-ready raw CT, separate center marker and explicit support, no colored overlays.
    np.savez_compressed(out/'slices.npz', ct=images, marker=marker,
                        support=np.ones_like(images, dtype=np.uint8),
                        arc=np.array([r['arc'] for r in records]).reshape(args.count, 3)[:, 0],
                        centers=np.array([r['center'] for r in records]).reshape(args.count, 3, 3)[:, 0],
                        frames=np.array([r['frame'] for r in records]).reshape(args.count, 3, 3, 3))
    lo, hi = map(float, np.quantile(images, [.01, .995]))
    if hi <= lo:
        raise ValueError('CT example has no display contrast')
    fig, axes = plt.subplots(3, args.count, figsize=(2.65*args.count, 8.4), squeeze=False)
    extent = (args.pixels-1)*args.spacing/2
    for i in range(args.count):
        for j, name in enumerate(('uv', 'uf', 'vf')):
            arr = images[i, j]
            Image.fromarray(np.flipud(np.rint(arr*255).astype(np.uint8))).save(out/f'{i:02}_{name}_raw.png')
            display = np.rint(np.clip((arr-lo)/(hi-lo), 0, 1)*255).astype(np.uint8)
            Image.fromarray(np.flipud(display)).save(out/f'{i:02}_{name}_display.png')
            ax = axes[j, i]
            ax.imshow(arr, cmap='gray', vmin=lo, vmax=hi, origin='lower',
                      extent=(-extent, extent, -extent, extent), interpolation='nearest')
            ax.scatter([0], [0], s=65, facecolors='none', edgecolors='#00e5ff', linewidths=.9)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel({'uv':'Transverse u-v', 'uf':'Longitudinal u-f', 'vf':'Longitudinal v-f'}[name])
            if j == 0:
                ax.set_title(f"s = {args.start+i*args.step:g}  (+{i*args.step:g})", fontsize=10)
    width = (args.pixels-1)*args.spacing
    fig.suptitle('Native CT along one annotated training fiber | cyan ring = sampled path location\n'
                 f'Three orthogonal planes; {args.step:g} trace voxels between positions; '
                 f'{width:g} × {width:g} trace-voxel field of view', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out/'contact_sheet.png', dpi=140)
    plt.close(fig)
    manifest = dict(kind='annotated positive illustration, not a departure or accuracy test',
                    fiber=fiber.name, fiber_source_sha256=fiber.source_hash,
                    source_url=args.source, ct_array=str(root), ct_shape=meta['shape'],
                    source_metadata_sha256=hashlib.sha256((root/'.zarray').read_bytes()).hexdigest(),
                    source_chunk_count=len(keys), source_chunks=names,
                    source_chunk_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
                    trace_grid_scale=cfg.trace_scale, ct_grid_scale=cfg.grid_scale, native_voxels_per_pixel=args.spacing*scale,
                    slice_spacing_trace=args.spacing, pixels=args.pixels,
                    path_step_trace=args.step, start_arc_trace=args.start, count=args.count,
                    view_names=['uv','uf','vf'], input_normalization='uint8 / 255',
                    display_window_normalized=[lo,hi], display_window_scope='all views, whole example sequence',
                    display_vertical_axis='increasing upward; exported PNG rows flip the numerical array',
                    full_interpolation_support=True, record=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()} for r in records])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'index.html').write_text('''<!doctype html><html><meta charset="utf-8">
<title>CT slice judge — real fiber example</title>
<style>body{background:#17191c;color:#eee;font:17px system-ui;margin:28px} .views{display:flex;gap:14px;flex-wrap:wrap}img{width:min(29vw,514px);image-rendering:pixelated} .image{position:relative}.ring{position:absolute;left:50%;top:50%;width:14px;height:14px;border:1px solid cyan;border-radius:50%;transform:translate(-50%,-50%);pointer-events:none}label{margin-right:24px}a{color:#77d5ff}</style>
<h1>CT slices along an annotated training fiber</h1>
<p>Each image spans '''+f'{width:g}'+''' trace voxels ('''+f'{width*scale:g}'+''' native CT intervals). Samples are '''+f'{args.step:g}'+''' trace voxels apart.
The cyan ring marks the path location. This is a valid annotated path, not a detected switch.</p>
<label>Position <input id="pos" type="range" min="0" max="'''+str(args.count-1)+'''" value="0"> <span id="value"></span></label>
<label><input id="raw" type="checkbox">Raw intensity / 255</label>
<label><input id="marker" type="checkbox" checked>Show path marker</label>
<div class="views">'''+''.join(f'<div><h3>{title}</h3><div class="image"><img id="{name}"><span class="ring"></span></div></div>' for name,title in [('uv','Transverse u–v'),('uf','Longitudinal u–f'),('vf','Longitudinal v–f')])+'''</div>
<p>Default display uses one shared contrast window across all images. Training data in
<a href="slices.npz">slices.npz</a> remains CT / 255 with separate marker and support arrays.
<a href="contact_sheet.png">All positions</a> · <a href="manifest.json">Sampling provenance</a></p>
<script>const p=document.querySelector('#pos'),r=document.querySelector('#raw'),m=document.querySelector('#marker');
function update(){document.querySelector('#value').textContent='s = '+('''+str(args.start)+'''+Number(p.value)*'''+str(args.step)+''')+' trace voxels';
for(const v of ['uv','uf','vf']) document.getElementById(v).src=String(p.value).padStart(2,'0')+'_'+v+'_'+(r.checked?'raw':'display')+'.png';
for(const e of document.querySelectorAll('.ring'))e.style.display=m.checked?'block':'none';}
p.oninput=r.onchange=m.onchange=update;update();</script></html>''')
    print(f'Wrote {out}/contact_sheet.png, index.html, slices.npz and manifest.json', flush=True)


if __name__ == '__main__':
    main()
