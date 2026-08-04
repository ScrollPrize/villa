"""Classical (no-ML) winding tracing of a real scroll slab -> a
fit_phantom_reference-compatible dataset. Defaults reproduce the PHerc0172
reroll demo end to end; the full sequence is:

  python trace_real_windings.py --out /tmp/p172_dataset
  python fit_phantom_reference.py --dataset /tmp/p172_dataset \
      --out /tmp/p172_fit.pt --steps 1200 \
      --reg-gap 0.03 --reg-flow 1.0 --reg-flow-smooth 100
  python unroll_real_fit.py --checkpoint /tmp/p172_fit.pt \
      --out-prefix /tmp/p172

Method: stream a z-slab from an open-data OME-Zarr volume (cached locally),
cast rays from the per-slice umbilicus (mask centroid), find intensity peaks
(sheet crossings), link peaks across neighbouring rays by radius continuity
into fragments, and assign each fragment an absolute winding index via its
spiral phase coordinate with a shared per-theta radial-offset correction
(estimated once, on the mid slice -- per-slice corrections can shift a whole
slice's indices by one and shred the cross-z assembly). Fragments of each
index jointly fill a [z, theta] surface grid, written as tifxyz + winding.tif.

Honest limitations, measured on PHerc0172 (see the branch history):
integer mis-indexing noise survives in the output (about 40% of holdout
residuals sit near nonzero integer shifts), and the resulting sparse
constraints underdetermine the deformation between them -- classical tracing
is a demo-grade constraint source, not a replacement for dense surface
predictions. PHerc0172 winds anticlockwise, so the default --mirror-x flips
the slab to make the whole chain CW; unrolled images must be re-flipped for
display (unroll_real_fit.py does).
"""

import json
import os

import click
import numpy as np
import scipy.ndimage
import scipy.signal
import tifffile
import zarr

from tifxyz import save_tifxyz

DEFAULT_VOLUME = ('https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
                  'PHerc0172/volumes/20241024131839-7.910um-53keV-masked.zarr')


def fetch_slab(volume_url, level, z0, z_size, cache_path):
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path).astype(np.float32)
    group = zarr.open_group(volume_url, mode='r')
    path = next(d['path'] for d in group.attrs['multiscales'][0]['datasets']
                if d['path'] == str(level))
    slab = group[path][z0:z0 + z_size]
    if cache_path:
        np.save(cache_path, slab)
    return slab.astype(np.float32)


def trace_slice(img, cy, cx, thetas, radii_samples, match_tol, max_ray_gap,
                min_arc, peak_distance, peak_prominence):
    """Track peaks over rays; returns fragments as {theta_idx: radius}."""
    H, W = img.shape
    smoothed = scipy.ndimage.gaussian_filter(img, 1.0)
    active, done = [], []
    for ti, t in enumerate(thetas):
        ys = cy + radii_samples * np.sin(t)
        xs = cx + radii_samples * np.cos(t)
        inside = (ys >= 0) & (ys < H - 1) & (xs >= 0) & (xs < W - 1)
        prof = np.zeros(len(radii_samples))
        prof[inside] = scipy.ndimage.map_coordinates(
            smoothed, [ys[inside], xs[inside]], order=1)
        peaks, _ = scipy.signal.find_peaks(
            prof, distance=peak_distance, prominence=peak_prominence)
        radii = radii_samples[peaks]
        used = np.zeros(len(radii), dtype=bool)
        for tr in active:
            if len(radii) == 0:
                continue
            d = np.abs(radii - tr['last_r'])
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] < match_tol:
                tr['pts'][ti] = radii[j]
                tr['last_r'] = radii[j]
                tr['last_ti'] = ti
                used[j] = True
        for j in np.nonzero(~used)[0]:
            active.append({'pts': {ti: radii[j]}, 'last_r': radii[j], 'last_ti': ti})
        still = []
        for tr in active:
            (done if ti - tr['last_ti'] > max_ray_gap else still).append(tr)
        active = still
    done += active
    return [t['pts'] for t in done if len(t['pts']) >= min_arc]


def index_fragments(tracks, dr, thetas, num_angles, delta=None):
    """Absolute winding index per fragment via the spiral phase coordinate;
    pass a shared per-theta offset field `delta` so all slices agree."""
    def phases(d):
        return np.array([float(np.median(
            [(t[ti] - thetas[ti] / (2 * np.pi) * dr - d[ti]) / dr for ti in t]))
            for t in tracks])

    if delta is None:
        k1 = np.round(phases(np.zeros(num_angles)))
        resid = [[] for _ in range(num_angles)]
        for t, k in zip(tracks, k1):
            for ti in t:
                resid[ti].append(t[ti] - (thetas[ti] / (2 * np.pi) + k) * dr)
        delta = np.array([np.median(r) if r else 0. for r in resid])
        delta = scipy.ndimage.uniform_filter1d(delta, 31, mode='wrap')
        delta -= delta.mean()  # keep annotations near the raw spiral phase
    return [(int(k), t) for k, t in zip(np.round(phases(delta)), tracks)], delta


@click.command()
@click.option('--volume-url', default=DEFAULT_VOLUME, show_default=False,
              help='OME-Zarr volume (default: PHerc0172 7.91um masked).')
@click.option('--level', default=3, type=int, help='Resolution level (ds 2^level).')
@click.option('--z0', default=1280, type=int, help='Slab start z at that level.')
@click.option('--z-size', default=32, type=int)
@click.option('--slab-cache', default=None, type=click.Path(dir_okay=False),
              help='Cache the fetched slab as .npy (reused if present).')
@click.option('--mirror-x/--no-mirror-x', default=True,
              help='Flip x so an ACW-wound scroll becomes CW (PHerc0172 needs it).')
@click.option('--out', required=True, type=click.Path(file_okay=False))
@click.option('--r-min', default=60., type=float)
@click.option('--r-max', default=470., type=float)
@click.option('--z-step', default=2, type=int, help='Trace every Nth slice.')
@click.option('--theta-step', default=3, type=int, help='Grid column spacing, degrees.')
@click.option('--min-coverage', default=0.15, type=float,
              help='Drop winding grids with fewer valid cells than this fraction.')
def main(volume_url, level, z0, z_size, slab_cache, mirror_x, out, r_min, r_max,
         z_step, theta_step, min_coverage):
    slab = fetch_slab(volume_url, level, z0, z_size, slab_cache)
    if mirror_x:
        slab = slab[:, :, ::-1].copy()
    Z, H, W = slab.shape
    click.echo(f'slab {slab.shape}{" (x-mirrored -> CW)" if mirror_x else ""}')

    centroids = np.array([[c.mean() for c in np.nonzero(slab[z] > 0)] for z in range(Z)])
    centroids = scipy.ndimage.median_filter(centroids, size=(5, 1))
    num_angles = 360
    thetas = np.deg2rad(np.arange(num_angles))
    radii_samples = np.arange(r_min, r_max, 0.5)
    trace_kwargs = dict(thetas=thetas, radii_samples=radii_samples, match_tol=2.5,
                        max_ray_gap=6, min_arc=40, peak_distance=8, peak_prominence=10)

    # dr: median peak-to-peak spacing with the SAME detector the tracker uses
    # (autocorrelation latches onto the coarser pack-bunching scale instead).
    z_list = list(range(0, Z, z_step))
    mid_z = z_list[len(z_list) // 2]
    mid_img = scipy.ndimage.gaussian_filter(slab[mid_z], 1.0)
    spac = []
    for tdeg in range(0, 360, 15):
        t = thetas[tdeg]
        ys, xs = centroids[mid_z][0] + radii_samples * np.sin(t), centroids[mid_z][1] + radii_samples * np.cos(t)
        inside = (ys >= 0) & (ys < H - 1) & (xs >= 0) & (xs < W - 1)
        prof = np.zeros(len(radii_samples))
        prof[inside] = scipy.ndimage.map_coordinates(mid_img, [ys[inside], xs[inside]], order=1)
        peaks, _ = scipy.signal.find_peaks(prof, distance=8, prominence=10)
        spac.extend(np.diff(radii_samples[peaks]))
    dr = float(np.median(spac))
    click.echo(f'dr per winding estimate: {dr:.2f} vox')

    _, shared_delta = index_fragments(
        trace_slice(slab[mid_z], *centroids[mid_z], **trace_kwargs),
        dr, thetas, num_angles)
    per_slice = {}
    for z in z_list:
        fragments = trace_slice(slab[z], *centroids[z], **trace_kwargs)
        per_slice[z], _ = index_fragments(fragments, dr, thetas, num_angles, shared_delta)

    theta_cols = list(range(0, num_angles, theta_step))
    all_k = sorted({k for frs in per_slice.values() for k, _ in frs})
    surfaces = {}
    for k in range(max(0, min(all_k)), max(all_k) + 1):
        grid = np.full((len(z_list), len(theta_cols), 3), -1., dtype=np.float32)
        for zi, z in enumerate(z_list):
            cy, cx = centroids[z]
            for kk, t in per_slice[z]:
                if kk != k:
                    continue
                for ci, tdeg in enumerate(theta_cols):
                    if tdeg in t:
                        r, th = t[tdeg], thetas[tdeg]
                        grid[zi, ci] = (z, cy + r * np.sin(th), cx + r * np.cos(th))
        if (grid[..., 0] >= 0).mean() > min_coverage:
            surfaces[k] = grid

    patches_dir = os.path.join(out, 'verified_patches')
    os.makedirs(patches_dir, exist_ok=True)
    with open(os.path.join(out, 'umbilicus.json'), 'w') as f:
        json.dump({'control_points': [
            {'z': float(z), 'y': float(centroids[z][0]), 'x': float(centroids[z][1])}
            for z in range(Z)]}, f)
    for k, grid in surfaces.items():
        uuid = f'traced_w{k:03d}'
        save_tifxyz(grid, patches_dir, uuid, step_size=4,
                    voxel_size_um=7.91 * 2 ** level, source='trace_real_windings')
        tifffile.imwrite(os.path.join(patches_dir, uuid, 'winding.tif'),
                         np.full(grid.shape[:2], float(k), dtype=np.float32))

    coverage = float(np.mean([(g[..., 0] >= 0).mean() for g in surfaces.values()]))
    with open(os.path.join(out, 'dataset_meta.json'), 'w') as f:
        json.dump({
            'winding_annotations': True, 'num_patches': len(surfaces),
            'position_noise': 0.0, 'coverage': coverage,
            'source': {'volume_url': volume_url, 'level': level, 'z0': z0,
                       'z_size': z_size, 'mirror_x': mirror_x},
            'phantom_meta': {'z_size': int(Z), 'yx_size': int(max(H, W)),
                             'dr_per_winding': dr, 'first_winding': 0,
                             'last_winding': int(max(surfaces))},
        }, f, indent=2)
    click.echo(f'{len(surfaces)} winding surfaces -> {out} '
               f'(mean coverage {coverage:.2f}, dr {dr:.2f})')


if __name__ == '__main__':
    main()
