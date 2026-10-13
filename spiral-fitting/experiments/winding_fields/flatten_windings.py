"""Lasagna-flatten the per-winding tifxyz exactly as fit_spiral publication / flatten_spiral_checkpoint do:
save_combined_tifxyz (export layout, 3-cell erosion + largest component), then
flatten_spiral_checkpoint._flatten with Lasagna's flatten_fast_nofilter.json."""
import glob, json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np, tifffile
sys.path.insert(0, '/ephemeral/sean/villa/spiral-fitting')
_here = os.path.dirname(os.path.abspath(__file__))
_sf = os.path.normpath(os.path.join(_here, '..', '..'))                       # spiral-fitting/ (tifxyz: export layout, outermost first)
sys.path.insert(0, _sf if os.path.exists(os.path.join(_sf, 'tifxyz.py')) else os.path.join(_here, 'vendor_sf'))
from tifxyz import save_combined_tifxyz
import flatten_spiral_checkpoint as fsc

src, output = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
W = '/ephemeral/sean/experiments/winding_fields'
scroll = json.load(open(f'{W}/data/spiral-scroll.json'))
step = 18.0
S = {}
for d in sorted(glob.glob(f'{src}/winding_*')):
    S[int(d.rsplit('_', 1)[1])] = np.stack([tifffile.imread(f'{d}/{c}.tif') for c in 'zyx'], -1)
# align rows by z (each per-winding file is trimmed to its valid rows)
zall = np.concatenate([a[..., 0][a[..., 0] > 0] for a in S.values()]); z0 = zall.min()
nrow = int(round((zall.max() - z0) / step)) + 1
grids = {}
for k, a in S.items():
    rz = np.array([np.median(r[r > 0]) if (r > 0).any() else np.nan for r in a[..., 0]])
    i0 = np.nonzero(np.isfinite(rz))[0][0]; off = int(round((rz[i0] - z0) / step)) - i0
    g = np.full((nrow, a.shape[1], 3), -1.0, np.float32); g[off:off + len(a)] = a
    grids[k] = g
first = min(grids)
work = Path(tempfile.mkdtemp(prefix=f'.{output.name}.flatten-', dir=output.parent))
try:
    man = save_combined_tifxyz(grids, work / 'source', 'winding-fields-joint.tifxyz', step, float(scroll['voxel_size_um']),
                               'winding_fields joint sheet-graph (combine for lasagna flatten)',
                               z_direction_is_top_to_bottom=scroll.get('z_direction_is_top_to_bottom'),
                               first_winding=first, cleanup_erosion_cells=3, base_shape_zyx=scroll.get('base_shape_zyx'))
    surface = Path(man['surface_path']).resolve()
    m = json.load(open(surface / 'meta.json')); x = tifffile.imread(surface / 'x.tif')
    print(f'source surface {x.shape}, valid after cleanup {np.mean(x != -1):.3f}; cleanup meta: '
          f'{ {k: v for k, v in m.items() if "cleanup" in k} }', flush=True)
    service, cfg = fsc._resolve_lasagna(None)
    print(f'lasagna {service} config {cfg}', flush=True)
    keep = output.parent / (output.name.replace('.tifxyz', '') + '-source.tifxyz')
    if not keep.exists():
        shutil.copytree(surface, keep)                       # keep the un-flattened input next to the output
    fsc._flatten(surface, output, service, cfg, work, voxel_size_um=float(scroll['voxel_size_um']))
    print('FLATTEN_DONE', output, flush=True)
finally:
    shutil.rmtree(work, ignore_errors=True)
