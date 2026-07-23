"""Build a capped signed-distance (surf-SDT) OME-Zarr volume from a raw
surface-prediction zarr, per docs/spiral_surf_sdt_generation.md.

The output is a uint8 volume encoding a capped signed euclidean distance to
the binarized surface prediction:

    value 0        no data (reserved; not emitted by this build)
    value 1..255   sd = (value - offset) * unit_working_voxels,
                   positive outside the mask, negative inside,
                   clipped to +/- cap_working_voxels

The distance is exact under the cap: each tile is computed on a window with a
halo sized so that any in-cap nearest surface voxel is inside the window, so
the windowed EDT equals the global EDT for every value that survives the
clip. Windows with no mask voxel at all write uniform saturation (edt returns
inf, which clips to the cap).

Nothing volume-specific is hardcoded: geometry (grid scale, voxel size) is
read from the source store's metadata with flags as overrides, the
binarization threshold is a required per-volume argument, and all resolved
parameters are recorded in the output attributes so the store is
interpretable without this script.

Resumable: finished tile ids are appended to a sidecar done_tiles.json next
to the output store (rewritten atomically); --resume skips them after
verifying the sidecar was written by an identical parameterization (the
working-z restriction may differ between runs, enabling ROI-first builds).
A "complete": true attribute is stamped only once every tile of every group
has been built.

Empty-chunk skipping: with --ct, a prepass on a downsampled group of the
paired CT volume (same working grid) marks the output chunks whose extent
holds no CT content (intensity >= --ct-threshold, default 1 = any nonzero
voxel); those chunks are never computed or written, so they read back as the
zarr fill value 0 = no-data, which the fitter's validity policy already
handles. The occupancy is dilated by --ct-dilate CT voxels (default 1) so a
chunk is only skipped when its whole neighbourhood is empty.

Prediction conditioning, applied per tile before the EDT: --erode (default
0 = off) binary-erodes the binarized prediction by N source voxels (cross-shaped
structuring element, N iterations; the tile halo is enlarged by N so the
windowed result still equals the global one under the cap), and --ct-zero
(on by default when --ct is given) clears the prediction wherever the paired
CT volume is exactly 0 = unscanned fill, so garbage predictions outside the
scan produce no surface. The exact-zero test samples the CT group whose
scale is nearest the source grid (nearest-neighbour voxel mapping); regions
beyond the CT extent also count as zero.

Chunks are uncompressed by default (--codec none): at the scales in use the
store is small enough that compression is not worth the read-speed penalty.
--codec vcz1-rans (delta + rANS lossless, quant bin width 1) and blosc-zstd
remain available; vcz1 consumers must have the numcodecs wrapper registered
(import vc.compression.vcz1_numcodecs) to decode.
"""

import importlib.metadata
import itertools
import json
import math
import os
import shlex
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import click
import numpy as np
import zarr
from numcodecs import Blosc
from scipy import ndimage
from tqdm import tqdm

try:
    from vc.compression.vcz1_numcodecs import Vcz1  # also registers the codec
except ImportError:
    Vcz1 = None

ENCODING_OFFSET = 128
MIN_EDT_VERSION = (3, 1, 2)
DILATE_SLAB = 256


def _require_edt_version():
    # 3.1.1 (the version kimimaro's constraint resolves to) mishandles
    # scanlines with no boundary: inf poisons the Felzenszwalb passes and
    # corrupts exactly the long air runs this SDT exists to measure.
    ver = importlib.metadata.version('edt')
    parts = tuple(int(p) for p in ver.split('.')[:3])
    if parts < MIN_EDT_VERSION:
        sys.exit(f'edt {ver} is installed, but >= {".".join(map(str, MIN_EDT_VERSION))} is required '
                 f'(3.1.1 computes wrong distances on boundary-free scanlines); run uv sync')


def _resolve_source_scale(src_root, group_name):
    """Per-axis working-voxels-per-source-grid-voxel from the source OME
    multiscales (e.g. 2.0 for a working/2 group)."""
    ms = src_root.attrs.get('multiscales')
    if not ms:
        return None
    for ds in ms[0].get('datasets', []):
        if ds.get('path') == group_name:
            for tr in ds.get('coordinateTransformations', []):
                if tr.get('type') == 'scale':
                    return tuple(float(s) for s in tr['scale'])
    return None


def _resolve_working_voxel_um(surf_path):
    """Volpkg convention: meta.json next to the volume zarr carries the
    stored (group-0 = working) voxel size."""
    meta_path = os.path.join(surf_path, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if 'voxelsize' in meta:
            return float(meta['voxelsize'])
    return None


def _scalar_or_list(values):
    values = list(values)
    return values[0] if all(v == values[0] for v in values) else values


def _tile_starts(extent, tile):
    return list(range(0, extent, tile))


def _grid_z_bounds(z_range_working, scale_z, extent):
    """Convert a working-voxel z range to this grid's voxel range, clamped."""
    if z_range_working is None:
        return 0, extent
    lo, hi = z_range_working
    return max(0, math.floor(lo / scale_z)), min(extent, math.ceil(hi / scale_z))


def _nearest_scale_group(root, target_scale):
    """(group, scale) of the multiscales dataset whose scale is nearest
    `target_scale` (log-space distance, summed over axes)."""
    ms = root.attrs.get('multiscales')
    best = None
    for ds in (ms[0].get('datasets', []) if ms else []):
        for tr in ds.get('coordinateTransformations', []):
            if tr.get('type') == 'scale':
                sc = tuple(float(s) for s in tr['scale'])
                err = sum(abs(math.log(a / b)) for a, b in zip(sc, target_scale))
                if best is None or err < best[0]:
                    best = (err, ds['path'], sc)
    return (best[1], best[2]) if best else None


def _coarsest_group(root):
    ms = root.attrs.get('multiscales')
    if not ms:
        return None
    datasets = ms[0].get('datasets', [])
    if not datasets:
        return None
    def scale_of(ds):
        for tr in ds.get('coordinateTransformations', []):
            if tr.get('type') == 'scale':
                return max(tr['scale'])
        return 0
    return max(datasets, key=scale_of)['path']


def _dilate_occupancy(occ, radius, bar):
    """Cube maximum filter, z-chunked with a `radius` halo (so slab results
    equal the global filter) and threaded (ndimage releases the GIL)."""
    out = np.empty_like(occ)

    def run(z0):
        z1 = min(z0 + DILATE_SLAB, occ.shape[0])
        lo = max(0, z0 - radius)
        hi = min(occ.shape[0], z1 + radius)
        win = ndimage.maximum_filter(occ[lo:hi], size=2 * radius + 1)
        out[z0:z1] = win[z0 - lo:z1 - lo]
        bar.update(1)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        list(pool.map(run, range(0, occ.shape[0], DILATE_SLAB)))
    return out


def _load_ct_occupancy(ct_path, ct_group, ct_threshold, ct_dilate):
    """Boolean CT-content occupancy on the (downsampled) CT grid, plus that
    grid's per-axis working-voxel pitch."""
    root = zarr.open_group(ct_path, mode='r')
    if ct_group is None:
        ct_group = _coarsest_group(root)
        if ct_group is None:
            sys.exit('no multiscales metadata in the CT volume; pass --ct-group')
    if ct_group not in root:
        sys.exit(f'CT group {ct_group!r} not found in {ct_path}')
    ct_scale = _resolve_source_scale(root, ct_group)
    if ct_scale is None:
        sys.exit(f'no scale for CT group {ct_group!r} in the CT OME multiscales metadata')
    arr = root[ct_group]
    slab = max(1, arr.chunks[0])
    starts = range(0, arr.shape[0], slab)
    dilate_steps = len(range(0, arr.shape[0], DILATE_SLAB)) if ct_dilate > 0 else 0
    occ = np.empty(arr.shape, bool)
    with tqdm(total=len(starts) + dilate_steps, desc=f'ct prepass (group {ct_group})',
              unit='step', disable=None) as bar:
        for z0 in starts:
            z1 = min(z0 + slab, arr.shape[0])
            occ[z0:z1] = np.asarray(arr[z0:z1]) >= ct_threshold
            bar.update(1)
        if ct_dilate > 0:
            # cube footprint: conservative in the diagonal directions too
            occ = _dilate_occupancy(occ, ct_dilate, bar)
    return occ, ct_scale, ct_group


def _axis_max_over_chunks(a, axis, n_out, chunk_ext_wv, ct_pitch_wv):
    """Reduce axis `axis` of `a` (CT-grid occupancy) to per-output-chunk max.
    Chunk c covers working [c, c+1) * chunk_ext_wv; CT voxel i covers working
    [i, i+1) * ct_pitch_wv; overlapping CT voxels contribute (conservative)."""
    n = a.shape[axis]
    empty_shape = list(a.shape)
    empty_shape[axis] = 1
    segs = []
    for c in range(n_out):
        lo = min(math.floor(c * chunk_ext_wv / ct_pitch_wv), n)
        hi = min(max(math.ceil((c + 1) * chunk_ext_wv / ct_pitch_wv), lo), n)
        if lo >= hi:  # chunk lies beyond the CT volume's extent: no data
            segs.append(np.zeros(empty_shape, bool))
        else:
            sl = [slice(None)] * 3
            sl[axis] = slice(lo, hi)
            segs.append(a[tuple(sl)].max(axis=axis, keepdims=True))
    return np.concatenate(segs, axis=axis)


def _chunk_occupancy(occ_ct, ct_scale, shape, grid_scale, chunk):
    """Per-output-chunk CT occupancy for a group of `shape` at `grid_scale`
    (grid voxels per working voxel, per axis)."""
    a = occ_ct
    for axis in range(3):
        n_out = -(-shape[axis] // chunk)
        a = _axis_max_over_chunks(a, axis, n_out, chunk * grid_scale[axis], ct_scale[axis])
    return a.astype(bool)


def _mean_pool2_encoded(block):
    """2x mean pooling of encoded uint8 SDT values, excluding no-data (0).

    The encoding is affine, so mean-pooling encoded values (over valid voxels
    only) is exactly mean-pooling the decoded distances. Blocks that are
    entirely no-data pool to 0.
    """
    valid = block != 0
    vals = block.astype(np.float32)
    pad = [(0, s % 2) for s in block.shape]
    if any(p for _, p in pad):
        vals = np.pad(vals, pad, mode='edge')
        valid = np.pad(valid, pad, mode='edge')
    out_shape = tuple(s // 2 for s in vals.shape)
    vsum = np.zeros(out_shape, np.float32)
    n = np.zeros(out_shape, np.float32)
    for dz, dy, dx in itertools.product((0, 1), repeat=3):
        sub = (slice(dz, None, 2), slice(dy, None, 2), slice(dx, None, 2))
        v = valid[sub]
        vsum += vals[sub] * v
        n += v
    out = np.zeros(out_shape, np.uint8)
    nz = n > 0
    out[nz] = np.rint(vsum[nz] / n[nz]).astype(np.uint8)
    return out


# --- worker pool -------------------------------------------------------------

_G = {}


def _init_worker(surf_path, source_group, out_path, threshold, scale_zyx, unit_wv, qcap,
                 edt_threads, erode, ct_zero):
    _G['src'] = zarr.open_group(surf_path, mode='r')[source_group]
    _G['out_root'] = zarr.open_group(out_path, mode='r+')
    _G['threshold'] = threshold
    _G['scale'] = tuple(scale_zyx)
    _G['unit'] = unit_wv
    _G['qcap'] = qcap
    _G['edt_threads'] = edt_threads
    _G['erode'] = erode
    if ct_zero is not None:
        _G['ct_arr'] = zarr.open_group(ct_zero['path'], mode='r')[ct_zero['group']]
        _G['ct_ratio'] = ct_zero['ratio']  # source-grid pitch / CT-grid pitch, per axis
    else:
        _G['ct_arr'] = None


def _ct_zero_window(lo, hi):
    """Boolean window (source-grid bounds lo..hi): True where the paired CT is
    exactly 0 (nearest-neighbour sample of the matched CT group) or the
    location lies beyond the CT extent."""
    arr = _G['ct_arr']
    idxs, invalid, read = [], [], []
    for ax in range(3):
        idx = np.floor((np.arange(lo[ax], hi[ax]) + 0.5) * _G['ct_ratio'][ax]).astype(np.int64)
        ok = (idx >= 0) & (idx < arr.shape[ax])
        if not ok.any():
            return np.ones(tuple(h - l for l, h in zip(lo, hi)), bool)
        c0, c1 = int(idx[ok].min()), int(idx[ok].max()) + 1
        idxs.append(np.clip(idx, c0, c1 - 1) - c0)
        shape = [1, 1, 1]
        shape[ax] = ok.size
        invalid.append(np.logical_not(ok).reshape(shape))
        read.append(slice(c0, c1))
    zero = np.asarray(arr[tuple(read)])[np.ix_(*idxs)] == 0
    for inv in invalid:
        zero |= inv
    return zero


def _run_job(job):
    if job['kind'] == 'base':
        _build_base_tile(job)
    else:
        _build_pyramid_tile(job)
    return job['id']


def _build_base_tile(job):
    import edt

    src = _G['src']
    dst = _G['out_root'][job['group']]
    core = job['bounds']  # ((z0, z1), (y0, y1), (x0, x1)) in source voxels
    halo = job['halo']
    lo = [max(0, c0 - h) for (c0, _), h in zip(core, halo)]
    hi = [min(s, c1 + h) for (_, c1), s, h in zip(core, src.shape, halo)]
    window = np.ascontiguousarray(src[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    mask = window >= _G['threshold']
    if _G['erode'] > 0 and mask.any():
        # equals eroding the grayscale prediction then thresholding
        mask = ndimage.binary_erosion(mask, iterations=_G['erode'])
    if _G['ct_arr'] is not None:
        mask &= np.logical_not(_ct_zero_window(lo, hi))

    # anisotropy = grid pitch in working voxels, so distances come out in
    # working voxels directly; float32 output.
    d_out = edt.edt(np.logical_not(mask).view(np.uint8), anisotropy=_G['scale'],
                    black_border=False, parallel=_G['edt_threads'])
    d_in = edt.edt(mask.view(np.uint8), anisotropy=_G['scale'],
                   black_border=False, parallel=_G['edt_threads'])
    sd = d_out - d_in
    # inf (window with no mask / no air at all) clips to the cap, which is the
    # intended uniform "far outside/inside" value.
    q = np.clip(np.rint(sd / _G['unit']), -_G['qcap'], _G['qcap'])
    v = (q + ENCODING_OFFSET).astype(np.uint8)

    for box in job['write_boxes']:
        sel = tuple(slice(b0 - w0, b1 - w0) for (b0, b1), w0 in zip(box, lo))
        dst[tuple(slice(b0, b1) for b0, b1 in box)] = v[sel]


def _build_pyramid_tile(job):
    root = _G['out_root']
    src = root[job['src_group']]
    dst = root[job['group']]
    core = job['bounds']  # destination-grid core bounds
    src_sel = tuple(slice(2 * c0, min(2 * c1, s)) for (c0, c1), s in zip(core, src.shape))
    pooled = _mean_pool2_encoded(np.asarray(src[src_sel]))
    for box in job['write_boxes']:
        sel = tuple(slice(b0 - c0, b1 - c0) for (b0, b1), (c0, _) in zip(box, core))
        sub = pooled[sel]
        if sub.any():  # an all-no-data chunk stays unwritten (reads as fill 0)
            dst[tuple(slice(b0, b1) for b0, b1 in box)] = sub


# --- sidecar -----------------------------------------------------------------

def _sidecar_path(out_path):
    return os.path.abspath(out_path).rstrip('/') + '.done_tiles.json'


def _write_sidecar(path, params, done):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'params': params, 'done': sorted(done)}, f)
    os.replace(tmp, path)


def _load_sidecar(path, params):
    with open(path) as f:
        sidecar = json.load(f)
    if sidecar['params'] != params:
        diffs = {k: (sidecar['params'].get(k), params.get(k))
                 for k in set(sidecar['params']) | set(params)
                 if sidecar['params'].get(k) != params.get(k)}
        sys.exit(f'--resume refused: sidecar {path} was written by a different '
                 f'parameterization: {diffs}')
    return set(sidecar['done'])


def _merge_working_ranges(ranges):
    merged = []
    for lo, hi in sorted((float(a), float(b)) for a, b in ranges):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def _git_commit():
    try:
        return subprocess.run(['git', '-C', os.path.dirname(os.path.abspath(__file__)),
                               'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def _human_bytes(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f'{n:.1f} {unit}'
        n /= 1024


# --- main --------------------------------------------------------------------

@click.command(help='Build a capped signed-distance (surf-SDT) OME-Zarr from a '
                    'surface-prediction zarr. See docs/spiral_surf_sdt_generation.md.')
@click.option('--surf', 'surf_path', required=True,
              type=click.Path(exists=True, file_okay=False), help='source surface-prediction zarr')
@click.option('--group', 'group_name', required=True, help='source group to read (e.g. 1)')
@click.option('--out', 'out_path', required=True, type=click.Path(), help='output ome-zarr path')
@click.option('--threshold', required=True, type=int,
              help='binarization threshold (per-volume choice from the histogram plateau)')
@click.option('--scale', default=None,
              help='working voxels per source-grid voxel, float or "z,y,x" '
                   '(2.0 for a working/2 group); '
                   'default: read from the source OME multiscales metadata')
@click.option('--working-voxel-um', default=None, type=float,
              help='working voxel size in um; default: read from the source meta.json')
@click.option('--unit-wv', default=1.0, type=float, show_default=True,
              help='working voxels per encoded unit')
@click.option('--cap-wv', default=127.0, type=float, show_default=True,
              help='distance cap in working voxels')
@click.option('--z-range', default=None,
              help='optional working-z restriction "lo:hi" (fit ROI builds)')
@click.option('--tile', default=256, type=int, show_default=True,
              help='core tile edge in source voxels; must be a multiple of --chunk')
@click.option('--halo', default=None, type=int,
              help='halo in source voxels; default: derived as ceil(cap-wv / scale)')
@click.option('--chunk', default=128, type=int, show_default=True, help='output chunk edge')
@click.option('--ct', 'ct_path', default=None, type=click.Path(exists=True, file_okay=False),
              help='paired CT volume zarr (same working grid); output chunks whose extent holds '
                   'no CT content are skipped entirely and read back as fill 0 = no-data')
@click.option('--ct-group', default=None,
              help='CT group for the occupancy prepass; default: coarsest group in its multiscales')
@click.option('--ct-threshold', default=1, type=int, show_default=True,
              help='CT intensity >= this counts as content (default 1 = any nonzero voxel)')
@click.option('--ct-dilate', default=1, type=int, show_default=True,
              help='dilate the CT occupancy by this many CT-grid voxels before the chunk check, '
                   'so only chunks with an empty neighbourhood are skipped')
@click.option('--ct-zero/--no-ct-zero', default=True, show_default=True,
              help='clear the prediction wherever the paired CT volume is exactly 0 = unscanned '
                   'fill (sampled from the CT group nearest the source scale); needs --ct')
@click.option('--erode', default=0, type=int, show_default=True,
              help='binary-erode the binarized prediction by this many source voxels before the '
                   'EDT (the halo is enlarged to keep windowed distances exact)')
@click.option('--codec', default='none', show_default=True,
              type=click.Choice(['none', 'vcz1-rans', 'blosc-zstd']),
              help='chunk compression; none = uncompressed (fastest reads), vcz1-rans is the '
                   'VCZ1 delta + rANS lossless configuration')
@click.option('--pyramid-levels', default=2, type=int, show_default=True,
              help='number of 2x mean-pooled pyramid groups after the base group')
@click.option('--workers', default=4, type=int, show_default=True, help='worker processes')
@click.option('--edt-threads', default=None, type=int,
              help='edt threads per worker (default: cores / workers)')
@click.option('--resume', is_flag=True, help='skip tiles already marked complete in the sidecar')
def main(surf_path, group_name, out_path, threshold, scale, working_voxel_um, unit_wv, cap_wv,
         z_range, tile, halo, chunk, ct_path, ct_group, ct_threshold, ct_dilate, ct_zero, erode,
         codec, pyramid_levels, workers, edt_threads, resume):
    _require_edt_version()
    command_line = shlex.join(sys.argv)

    ct_zero = ct_zero and ct_path is not None
    if codec == 'vcz1-rans':
        if Vcz1 is None:
            sys.exit('the vcz1-rans codec needs the volume-cartographer python bindings '
                     '(vc.compression.vcz1); install them or pass another --codec')
        compressor = Vcz1(codec='rans', quant=1)  # delta + rANS, quant 1 = lossless
    elif codec == 'blosc-zstd':
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
    else:
        compressor = None

    surf_path = os.path.abspath(surf_path)
    src_root = zarr.open_group(surf_path, mode='r')
    if group_name not in src_root:
        sys.exit(f'source group {group_name!r} not found in {surf_path}')
    src = src_root[group_name]
    if src.ndim != 3:
        sys.exit(f'source group is {src.ndim}-d, expected 3-d (z, y, x)')

    if scale is not None:
        parts = [float(p) for p in scale.split(',')]
        scale_zyx = tuple(parts) if len(parts) == 3 else (parts[0],) * 3
    else:
        scale_zyx = _resolve_source_scale(src_root, group_name)
        if scale_zyx is None:
            sys.exit('no scale in the source OME multiscales metadata; pass --scale')

    if working_voxel_um is None:
        working_voxel_um = _resolve_working_voxel_um(surf_path)
        if working_voxel_um is None:
            sys.exit('no voxelsize in the source meta.json; pass --working-voxel-um')

    qcap = int(round(cap_wv / unit_wv))
    if not 1 <= qcap <= ENCODING_OFFSET - 1:
        sys.exit(f'cap-wv/unit-wv = {cap_wv / unit_wv:g} encoded units; must be in '
                 f'1..{ENCODING_OFFSET - 1} to fit the uint8 encoding')

    if halo is None:
        halo_zyx = tuple(math.ceil(cap_wv / s) + erode for s in scale_zyx)
    else:
        halo_zyx = (halo,) * 3
        short = [f'({h} - erode {erode}) * {s:g} = {(h - erode) * s:g} < {cap_wv:g}'
                 for h, s in zip(halo_zyx, scale_zyx) if (h - erode) * s < cap_wv]
        if short:
            sys.exit('--halo too small for the cap (need (halo - erode) * scale >= cap-wv): '
                     f'{short[0]}')

    if tile % chunk != 0:
        sys.exit(f'--tile {tile} must be a multiple of the output chunk edge {chunk} '
                 '(tile writes must be chunk-aligned for safe parallel/resumable writes)')

    z_range_working = None
    if z_range is not None:
        lo, hi = (float(p) for p in z_range.split(':'))
        if hi <= lo:
            sys.exit(f'--z-range {z_range}: hi must be > lo')
        z_range_working = (lo, hi)

    if pyramid_levels > 0 and not group_name.isdigit():
        sys.exit(f'pyramid group naming needs a numeric source group name, got {group_name!r}; '
                 'pass --pyramid-levels 0')
    group_names = [group_name] + [str(int(group_name) + k) for k in range(1, pyramid_levels + 1)]
    shapes = [tuple(src.shape)]
    for _ in range(pyramid_levels):
        shapes.append(tuple((s + 1) // 2 for s in shapes[-1]))

    if edt_threads is None:
        edt_threads = max(1, (os.cpu_count() or workers) // workers)

    # --- CT-zero pred masking: which CT group matches the source grid ---
    ct_zero_info = None
    if ct_zero:
        ct_path = os.path.abspath(ct_path)
        picked = _nearest_scale_group(zarr.open_group(ct_path, mode='r'), scale_zyx)
        if picked is None:
            sys.exit('no multiscales scales in the CT volume; --ct-zero needs them to '
                     'match a CT group to the source grid')
        zgroup, zscale = picked
        ct_zero_info = {'path': ct_path, 'group': zgroup,
                        'ratio': tuple(a / b for a, b in zip(scale_zyx, zscale))}

    # --- CT occupancy prepass: which output chunks hold any content ---
    chunk_occ = None
    if ct_path is not None:
        ct_path = os.path.abspath(ct_path)
        occ_ct, ct_scale, ct_group = _load_ct_occupancy(ct_path, ct_group, ct_threshold, ct_dilate)
        src_ext = [s * sc for s, sc in zip(src.shape, scale_zyx)]
        ct_ext = [s * sc for s, sc in zip(occ_ct.shape, ct_scale)]
        if any(abs(a - b) > 2 * max(sa, sb)
               for a, b, sa, sb in zip(src_ext, ct_ext, scale_zyx, ct_scale)):
            print(f'WARNING: CT working extent {ct_ext} differs from source working extent '
                  f'{src_ext}; is {ct_path} really the paired volume?')
        chunk_occ = [
            _chunk_occupancy(occ_ct, ct_scale, shape, [s * 2 ** lvl for s in scale_zyx], chunk)
            for lvl, shape in enumerate(shapes)
        ]

    # --- plan tiles ---
    # A pyramid tile is "provisional" when its parent source region is not
    # fully inside the parent's built z-range for this (z-restricted) run: it
    # is still built (no-data-aware pooling keeps it correct where the parent
    # exists) but not marked done, so a later wider resume rebuilds it.
    passes = []  # (group, src_group_or_None, [(tile_id, core_bounds, provisional, write_boxes)])
    parent_z = None
    for level, (name, shape) in enumerate(zip(group_names, shapes)):
        scale_z = scale_zyx[0] * 2 ** level
        z_lo, z_hi = _grid_z_bounds(z_range_working, scale_z, shape[0])
        occ = chunk_occ[level] if chunk_occ is not None else None
        tiles = []
        for tz in _tile_starts(shape[0], tile):
            core_z = (tz, min(tz + tile, shape[0]))
            if core_z[1] <= z_lo or core_z[0] >= z_hi:
                continue
            provisional = level > 0 and (
                2 * core_z[0] < parent_z[0]
                or min(2 * core_z[1], shapes[level - 1][0]) > parent_z[1])
            for ty in _tile_starts(shape[1], tile):
                for tx in _tile_starts(shape[2], tile):
                    bounds = tuple((c, min(c + tile, s)) for c, s in zip((tz, ty, tx), shape))
                    if occ is None:
                        boxes = [bounds]
                    else:
                        boxes = [
                            tuple((ci * chunk, min((ci + 1) * chunk, s))
                                  for ci, s in zip(cidx, shape))
                            for cidx in itertools.product(
                                *(range(b0 // chunk, -(-b1 // chunk)) for b0, b1 in bounds))
                            if occ[cidx]
                        ]
                        if not boxes:
                            continue  # every chunk of this tile is CT-empty
                    tiles.append((f'{name}:{tz // tile},{ty // tile},{tx // tile}',
                                  bounds, provisional, boxes))
        passes.append((name, group_names[level - 1] if level else None, tiles))
        parent_z = (z_lo, z_hi)

    total_tiles = sum(len(t) for _, _, t in passes)
    grid_voxel_um = [working_voxel_um * s for s in scale_zyx]
    if chunk_occ is not None:
        est_bytes = sum(int(o.sum()) * chunk ** 3 for o in chunk_occ)
    else:
        est_bytes = sum(math.prod(s) for s in shapes)

    print(f'source:   {surf_path} group {group_name}, shape {tuple(src.shape)}, dtype {src.dtype}')
    print(f'output:   {out_path} groups {group_names}, chunks ({chunk},)*3')
    print(f'geometry: scale vs working {_scalar_or_list(scale_zyx)}, '
          f'working voxel {working_voxel_um:g} um, grid voxel {_scalar_or_list(grid_voxel_um)} um')
    print(f'encoding: threshold {threshold}, unit {unit_wv:g} wv, cap {cap_wv:g} wv '
          f'(+/-{qcap} encoded units), offset {ENCODING_OFFSET}')
    if erode > 0 or ct_zero_info is not None:
        parts = []
        if erode > 0:
            parts.append(f'erode {erode} source voxels')
        if ct_zero_info is not None:
            parts.append(f'zero pred where CT group {ct_zero_info["group"]} == 0 '
                         f'(source/CT grid ratio {_scalar_or_list(ct_zero_info["ratio"])})')
        print(f'pred:     {"; ".join(parts)}')
    print(f'tiling:   tile {tile}, halo {_scalar_or_list(halo_zyx)} '
          f'(window {tuple(min(t, s) for t, s in zip((tile + 2 * h for h in halo_zyx), src.shape))}), '
          f'{" + ".join(str(len(t)) for _, _, t in passes)} = {total_tiles} tiles'
          + (f', working-z {z_range_working[0]:g}:{z_range_working[1]:g}' if z_range_working else ''))
    if chunk_occ is not None:
        kept = sum(int(o.sum()) for o in chunk_occ)
        total = sum(o.size for o in chunk_occ)
        print(f'ct mask:  {ct_path} group {ct_group}, threshold {ct_threshold}, '
              f'dilate {ct_dilate}; keeping {kept}/{total} chunks '
              f'({100 * (1 - kept / total):.1f}% skipped as CT-empty)')
    print(f'codec:    {codec}; workers {workers} x {edt_threads} edt threads; '
          f'estimated uncompressed size {_human_bytes(est_bytes)}')

    # --- create / reopen the store ---
    store_exists = os.path.exists(out_path)
    if store_exists and not resume:
        sys.exit(f'{out_path} already exists; pass --resume to continue it or remove it first')
    if resume and not store_exists:
        sys.exit(f'--resume given but {out_path} does not exist')

    sidecar_params = {
        'source': surf_path,
        'source_group': group_name,
        'threshold': threshold,
        'unit_working_voxels': unit_wv,
        'cap_working_voxels': cap_wv,
        'offset': ENCODING_OFFSET,
        'scale_vs_working': list(scale_zyx),
        'tile': tile,
        'halo': list(halo_zyx),
        'chunk': chunk,
        'groups': group_names,
        'codec': codec,
        'ct': ct_path,
        'ct_group': ct_group,
        'ct_threshold': ct_threshold if ct_path is not None else None,
        'ct_dilate': ct_dilate if ct_path is not None else None,
        'erode': erode,
        'ct_zero_group': ct_zero_info['group'] if ct_zero_info is not None else None,
    }
    sidecar = _sidecar_path(out_path)
    done = _load_sidecar(sidecar, sidecar_params) if resume else set()

    root = zarr.open_group(out_path, mode='a', zarr_format=2)
    for name, shape in zip(group_names, shapes):
        if name not in root:
            root.create_array(name, shape=shape, chunks=(chunk,) * 3, dtype='uint8',
                              fill_value=0, compressors=compressor,
                              chunk_key_encoding={'name': 'v2', 'separator': '/'})
    if not store_exists:
        attrs = {
            'kind': 'surf_sdt',
            'source': surf_path,
            'source_group': group_name,
            'source_shape_zyx': list(src.shape),
            'threshold': threshold,
            'unit_working_voxels': unit_wv,
            'offset': ENCODING_OFFSET,
            'cap_working_voxels': cap_wv,
            'sign': 'positive_outside',
            'working_voxel_um': working_voxel_um,
            'grid_voxel_um': _scalar_or_list(grid_voxel_um),
            'scale_vs_working': _scalar_or_list(scale_zyx),
            'tile': tile,
            'halo': _scalar_or_list(halo_zyx),
            'erode_source_voxels': erode,
            'created': datetime.now(timezone.utc).isoformat(),
            'git_commit': _git_commit(),
            'command_line': command_line,
            'multiscales': [{
                'version': '0.4',
                'name': 'surf_sdt',
                'axes': [{'name': a, 'type': 'space', 'unit': 'pixel'} for a in 'zyx'],
                'datasets': [{
                    'path': name,
                    'coordinateTransformations': [{
                        'type': 'scale',
                        'scale': [s * 2 ** level for s in scale_zyx],
                    }],
                } for level, name in enumerate(group_names)],
            }],
            'lasagna_pyramid_downsample': 'mean_pool2x',
        }
        if z_range_working is not None:
            attrs['z_range_working'] = list(z_range_working)
        if chunk_occ is not None:
            attrs['ct_mask'] = {
                'source': ct_path,
                'group': ct_group,
                'threshold': ct_threshold,
                'dilate_ct_voxels': ct_dilate,
                'kept_chunks': [int(o.sum()) for o in chunk_occ],
                'total_chunks': [int(o.size) for o in chunk_occ],
            }
            attrs['nodata_value'] = 0  # CT-empty chunks are unwritten; fill 0 = no data
        if ct_zero_info is not None:
            attrs['ct_zero'] = {
                'source': ct_zero_info['path'],
                'group': ct_zero_info['group'],
                'grid_ratio': _scalar_or_list(ct_zero_info['ratio']),
            }
        root.attrs.update(attrs)
        _write_sidecar(sidecar, sidecar_params, done)

    # --- run the passes ---
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker,
        initargs=(surf_path, group_name, os.path.abspath(out_path), threshold,
                  scale_zyx, unit_wv, qcap, edt_threads, erode, ct_zero_info),
    ) as pool:
        for name, src_group, tiles in passes:
            pending, provisional_ids = [], set()
            for tile_id, bounds, provisional, boxes in tiles:
                if provisional:
                    provisional_ids.add(tile_id)
                if tile_id in done:
                    continue
                job = {'id': tile_id, 'group': name, 'bounds': bounds, 'write_boxes': boxes}
                if src_group is None:
                    job.update(kind='base', halo=halo_zyx)
                else:
                    job.update(kind='pyramid', src_group=src_group)
                pending.append(job)
            desc = f'group {name}' + ('' if src_group is None else ' (pyramid)')
            with tqdm(total=len(tiles), initial=len(tiles) - len(pending),
                      desc=desc, unit='tile', disable=None) as bar:
                futures = [pool.submit(_run_job, job) for job in pending]
                for fut in as_completed(futures):
                    tile_id = fut.result()
                    if tile_id not in provisional_ids:
                        done.add(tile_id)
                        _write_sidecar(sidecar, sidecar_params, done)
                    bar.update(1)

    all_tiles = {tile_id for _, _, tiles in passes for tile_id, _, _, _ in tiles}

    # Embedded coverage provenance: the fitter accepts a partial store only via
    # attributes travelling with the zarr (never the external sidecar), so each
    # invocation that finishes every base-group tile of its z-range records
    # that range, and command history accumulates across --resume runs.
    base_tile_ids = {tile_id for tile_id, _, _, _ in passes[0][2]}
    if base_tile_ids and base_tile_ids <= done:
        built = (list(z_range_working) if z_range_working is not None
                 else [0.0, src.shape[0] * scale_zyx[0]])
        root.attrs['built_z_ranges_working'] = _merge_working_ranges(
            list(root.attrs.get('built_z_ranges_working', [])) + [built])
    root.attrs['command_history'] = list(root.attrs.get('command_history', [])) + [command_line]

    if z_range_working is None and all_tiles <= done:
        root.attrs['complete'] = True
        # A completing resume of an ROI-first store must not keep advertising
        # the stale creation-time restriction.
        try:
            del root.attrs['z_range_working']
        except KeyError:
            pass
        print('store complete; stamped "complete": true')
    else:
        missing = 'restricted to a working-z range' if z_range_working is not None \
            else f'{len(all_tiles - done)} tiles missing'
        print(f'build finished for this invocation, but the store is partial ({missing}); '
              'not stamping "complete"')


if __name__ == '__main__':
    main()
