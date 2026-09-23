"""Render review artifacts from prototype_fiber_merge; never modify its surfaces.

Uses the existing local Zarr chunk decoder for optional uint8 CT previews.
Example commands and limitations: ../volume-cartographer/apps/VC3D/test/
prototype_fiber_merge.md.
"""
from __future__ import annotations

import argparse
import base64
from functools import lru_cache
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import label

from zarr_chunk_io import _chunk_path, _make_chunk_reader


def load_surface(root, info):
    if info.get('components'):
        surfaces = [load_surface(root, component) for component in info['components']]
        spacing = surfaces[0][2]
        low = np.minimum.reduce([uv for _, uv, _ in surfaces])
        high = np.maximum.reduce([uv + np.array([xyz.shape[1]-1, xyz.shape[0]-1])*step
                                  for xyz, uv, step in surfaces])
        wh = np.rint((high-low)/spacing).astype(int)+1
        merged = np.full((wh[1], wh[0], 3), -1., dtype=np.float32)
        for xyz, uv, step in surfaces:
            if step != spacing:
                raise ValueError('component spacing differs')
            placed = place_on_grid(xyz, uv, low, merged.shape[:2], spacing)
            valid = np.isfinite(placed[..., 0])
            merged[valid] = placed[valid]
        return merged, low, spacing
    xyz = np.stack([tifffile.imread(root / info['path'] / f'{c}.tif') for c in 'xyz'], axis=-1)
    return xyz, np.asarray(info['origin'], dtype=float), float(info['spacing'])


def upsample_surface(xyz, factor):
    """Bilinear preview samples only inside complete source quads."""
    if factor < 1 or int(factor) != factor:
        raise ValueError('upsampling factor must be a positive integer')
    height, width = xyz.shape[:2]
    yy, xx = np.mgrid[:(height-1)*factor, :(width-1)*factor].astype(float) / factor
    y, x = yy.astype(int), xx.astype(int)
    a, b, c, d = xyz[y, x], xyz[y, x+1], xyz[y+1, x], xyz[y+1, x+1]
    valid = np.all(np.isfinite(np.stack([a, b, c, d])), axis=(0, 3))
    valid &= (a[..., 0] != -1) & (b[..., 0] != -1) & (c[..., 0] != -1) & (d[..., 0] != -1)
    u, v = (xx-x)[..., None], (yy-y)[..., None]
    out = (a*(1-u)+b*u)*(1-v) + (c*(1-u)+d*u)*v
    out[~valid] = -1
    return out, valid


class CTReader:
    def __init__(self, path, scale):
        self.path = Path(path)
        meta = json.loads((self.path / '.zarray').read_text())
        if meta['dtype'] != '|u1' or meta['order'] != 'C' or meta.get('filters'):
            raise ValueError('preview CT requires unfiltered, C-order uint8 Zarr v2')
        self.shape = np.asarray(meta['shape'])
        self.chunks = np.asarray(meta['chunks'])
        self.separator = meta.get('dimension_separator', '.')
        self.scale = scale
        self.read_chunk = _make_chunk_reader(meta.get('compressor'), int(self.chunks.prod()))
        self.missing = set()

    @lru_cache(maxsize=32)
    def chunk(self, key):
        path = _chunk_path(self.path, key, self.separator)
        if not path.exists():
            self.missing.add(key)
            return None
        return self.read_chunk(path).reshape(tuple(self.chunks))

    def sample(self, xyz):
        shape = xyz.shape[:-1]
        flat = xyz.reshape(-1, 3)
        index = np.rint(flat[:, ::-1] * self.scale).astype(np.int64)
        valid = (flat[:, 0] != -1) & np.all(np.isfinite(flat), axis=1)
        valid &= np.all((index >= 0) & (index < self.shape), axis=1)
        out = np.full(len(flat), np.nan, dtype=np.float32)
        selected = np.flatnonzero(valid)
        keys, inverse = np.unique(index[selected] // self.chunks, axis=0, return_inverse=True)
        order = np.argsort(inverse, kind='stable')
        boundaries = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(keys)))]
        for k, key in enumerate(keys):
            chunk = self.chunk(tuple(int(x) for x in key))
            if chunk is None:
                continue
            positions = selected[order[boundaries[k]:boundaries[k+1]]]
            local = index[positions] % self.chunks
            out[positions] = chunk[local[:, 0], local[:, 1], local[:, 2]]
        return out.reshape(shape)


def extent(xyz, origin, spacing):
    return [origin[0], origin[0]+(xyz.shape[1]-1)*spacing,
            origin[1], origin[1]+(xyz.shape[0]-1)*spacing]


def place_on_grid(xyz, origin, common_origin, shape, spacing):
    offset = (origin-common_origin) / spacing
    if not np.allclose(offset, np.rint(offset), atol=1e-6):
        raise ValueError('surfaces do not share the export lattice')
    x, y = np.rint(offset).astype(int)
    out = np.full((*shape, 3), np.nan, dtype=np.float64)
    values = xyz.astype(float).copy()
    values[xyz[..., 0] == -1] = np.nan
    out[y:y+len(xyz), x:x+xyz.shape[1]] = values
    return out


def source_rms(stack):
    count = np.isfinite(stack[..., 0]).sum(axis=0)
    mean = np.nansum(stack, axis=0)/np.maximum(count[..., None], 1)
    rms = np.sqrt(np.nansum((stack-mean)**2, axis=(0, 3))/np.maximum(count, 1))
    return rms, count, mean


def surface_metrics(xyz):
    valid = (xyz[..., 0] != -1) & np.isfinite(xyz).all(axis=-1)
    quads = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    labels, count = label(quads)
    sizes = np.bincount(labels.ravel())[1:]
    a, b, c, d = xyz[:-1, :-1].astype(float), xyz[:-1, 1:].astype(float), xyz[1:, :-1].astype(float), xyz[1:, 1:].astype(float)
    dot = (np.cross(b-a, c-a)*np.cross(c-d, b-d)).sum(axis=-1)
    return {'valid_quads': int(quads.sum()), 'quad_connected_regions': int(count),
            'single_quad_regions': int((sizes == 1).sum()),
            'opposing_triangle_quads': int((dot[quads] <= 0).sum())}


def validate_exports(root):
    report = json.loads((root / 'report.json').read_text())
    exports = [(info, False) for name in ('layers', 'initial_layers') for info in report.get(name, {}).values()]
    exports += [(group[phase], True) for group in report.get('groups', []) for phase in ('initial', 'refined') if phase in group]
    exports += [(info, True) for info in report.get('continuous', {}).values()]
    checked = []
    for info, merged in exports:
        if not info.get('valid'):
            continue
        for component in info.get('components', [info]):
            xyz, _, _ = load_surface(root, component)
            metrics = surface_metrics(xyz)
            if metrics['quad_connected_regions'] != 1 or metrics['valid_quads'] < 2:
                raise ValueError(f"{component['path']}: disconnected or single-quad export: {metrics}")
            if merged and metrics['opposing_triangle_quads']:
                raise ValueError(f"{component['path']}: twisted merged quads: {metrics}")
            checked.append({'path': component['path'], **metrics})
    (root / 'validation.json').write_text(json.dumps({'surfaces': checked}, indent=2))
    return checked


def preview(root, ct_path=None, xyz_scale=1):
    report = json.loads((root / 'report.json').read_text())
    if 'fiber_axis' not in report:
        raise ValueError('Merge report is incomplete; wait for prototype_fiber_merge to finish')
    vertical = report['fiber_axis'] == 1
    def read_surface(info):
        xyz, origin, spacing = load_surface(root, info)
        return (xyz.transpose(1, 0, 2), origin[::-1], spacing) if vertical else (xyz, origin, spacing)
    layers = {key: read_surface(value) for key, value in report['layers'].items() if value['valid']}
    ct = CTReader(ct_path, xyz_scale) if ct_path else None
    images, summaries = [], []
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'figure.facecolor': '#fafafa'})
    fiber_uv = np.asarray(report['fiber_uv'])
    if vertical:
        fiber_uv = fiber_uv[:, ::-1]
    for group in report['groups']:
        if not all(key in layers for key in group['patches']):
            summaries.append({'group': group['id'], 'status': group['status']})
            continue
        ids = sorted(group['patches'])
        sources = [layers[key] for key in ids]
        fused = group.get('refined', {}).get('valid', False)
        merged, origin, spacing = read_surface(group['refined']) if fused else sources[0]
        before = read_surface(group['initial']) if group.get('initial', {}).get('valid') else None
        low = np.minimum.reduce([s[1] for s in sources] + [origin])
        high = np.maximum.reduce([s[1]+np.array([s[0].shape[1]-1, s[0].shape[0]-1])*spacing for s in sources] +
                                 [origin+np.array([merged.shape[1]-1, merged.shape[0]-1])*spacing])
        wh = np.rint((high-low)/spacing).astype(int)+1
        stack = np.stack([place_on_grid(xyz, uv, low, (wh[1], wh[0]), spacing) for xyz, uv, _ in sources])
        rms, count, mean = source_rms(stack)
        rms[count < 2] = np.nan
        pair_values = rms[np.isfinite(rms)]
        common_extent = [low[0], high[0], low[1], high[1]]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        fig.suptitle(f"Fiber {report['fiber'].split('_')[-1].removesuffix('.json')} · group {group['id']} · {len(ids)} patches", fontsize=17)
        colors = plt.get_cmap('tab20')(np.linspace(0, 1, max(2, len(ids))))
        for n, (xyz, uv, step) in enumerate(sources):
            valid = xyz[..., 0] != -1
            rgba = np.zeros((*valid.shape, 4)); rgba[valid] = colors[n]; rgba[..., 3] *= .36
            axes[0, 0].imshow(rgba, origin='lower', extent=extent(xyz, uv, step), interpolation='nearest')
            xs = uv[0]+np.arange(xyz.shape[1])*step
            ys = uv[1]+np.arange(xyz.shape[0])*step
            for ax in (axes[0, 0], axes[1, 0]):
                ax.contour(xs, ys, valid, levels=[.5], colors=[colors[n]], linewidths=.6, zorder=3)
            if valid.any():
                yy, xx = np.argwhere(valid)[len(np.argwhere(valid))//2]
                axes[0, 0].text(xs[xx], ys[yy], str(n+1), fontsize=9, weight='bold')
        axes[0, 0].set_title('Review-fitted source coverage · colored patch boundaries')
        for ax, surface, title in [(axes[0, 1], before, 'First review pass · shared surface'),
                                    (axes[1, 0], (merged, origin, spacing) if fused else None, 'After neighbor refit · shared surface')]:
            ax.set_title(title + (' · CT' if ct else ' · XYZ height'))
            if surface is None:
                phase = 'initial' if ax is axes[0, 1] else 'refined'
                reason = group.get(phase, {}).get('message', group['status'])
                ax.text(.5, .5, f'{reason}\nGroup retained; no merged surface exported',
                        transform=ax.transAxes, ha='center', bbox={'facecolor': 'white', 'alpha': .8}, zorder=8)
                # Diagnostic mosaic only: independent fitted sources in stable
                # ID order. Never write this display composite as a surface.
                if ct and ax is axes[1, 0]:
                    for xyz, uv, step in sources:
                        dense, _ = upsample_surface(xyz, max(1, int(round(step/2))))
                        ax.imshow(np.ma.masked_invalid(ct.sample(dense)), origin='lower',
                                  extent=extent(xyz, uv, step), cmap='gray', vmin=40, vmax=180)
                    ax.set_title('Independent fitted layers · diagnostic CT mosaic only')
                continue
            xyz, uv, step = surface
            if ct:
                dense, valid = upsample_surface(xyz, max(1, int(round(step/2))))
                pixels = ct.sample(dense)
                ax.imshow(np.ma.masked_invalid(pixels), origin='lower', extent=extent(xyz, uv, step), cmap='gray', vmin=40, vmax=180)
            else:
                pixels = np.where(xyz[..., 0] != -1, xyz[..., 2], np.nan)
                ax.imshow(np.ma.masked_invalid(pixels), origin='lower', extent=extent(xyz, uv, step), cmap='gray')
        vmax = max(1., float(np.percentile(pair_values, 95))) if len(pair_values) else 1.
        im = axes[1, 1].imshow(np.ma.masked_invalid(rms), origin='lower', extent=common_extent, cmap='magma', vmin=0, vmax=vmax)
        axes[1, 1].set_title('Source XYZ disagreement at common UV · overlap only')
        fig.colorbar(im, ax=axes[1, 1], label='RMS distance to source mean (working voxels)', shrink=.7)
        for ax in axes.flat:
            ax.set_facecolor('#e5e7eb')
            ax.plot(fiber_uv[:, 0], fiber_uv[:, 1], color='#18afbf', linewidth=.9, zorder=5)
            ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1]); ax.set_aspect('equal')
            ax.set_xlabel(f"Fiber chart {'V' if vertical else 'U'} (working voxels)")
            ax.set_ylabel('U' if vertical else 'V')
        path = root / f"group_{group['id']}_preview.png"
        fig.savefig(path, dpi=150); plt.close(fig)
        images.append(path)
        summary = {'group': group['id'], 'patch_count': len(ids), 'status': group['status'],
                   'overlap_grid_samples': len(pair_values), 'patch_legend': {str(i+1): key for i, key in enumerate(ids)}}
        summary['exported_components'] = group.get('refined', {}).get('component_count')
        summary['dropped_single_quad_components'] = group.get('refined', {}).get('dropped_single_quad_components')
        if len(pair_values):
            summary['source_disagreement_median'] = float(np.median(pair_values))
            summary['source_disagreement_p95'] = float(np.percentile(pair_values, 95))
        if fused:
            summary.update(surface_metrics(merged))
            output_grid = place_on_grid(merged, origin, low, (wh[1], wh[0]), spacing)
            available = np.all(np.isfinite(output_grid), axis=-1) & (count > 0)
            distance = np.linalg.norm(output_grid[available]-mean[available], axis=-1)
            summary['fusion_shift_from_source_mean_p95'] = float(np.percentile(distance, 95)) if len(distance) else None
        if all(report.get('initial_layers', {}).get(key, {}).get('valid') for key in ids):
            old_sources = [read_surface(report['initial_layers'][key]) for key in ids]
            old_low = np.minimum.reduce([s[1] for s in old_sources])
            old_high = np.maximum.reduce([s[1]+np.array([s[0].shape[1]-1,s[0].shape[0]-1])*spacing for s in old_sources])
            old_wh = np.rint((old_high-old_low)/spacing).astype(int)+1
            old_stack = np.stack([place_on_grid(xyz, uv, old_low, (old_wh[1],old_wh[0]), spacing) for xyz, uv, _ in old_sources])
            old_rms, old_count, _ = source_rms(old_stack)
            old_values = old_rms[old_count>1]
            if len(old_values):
                summary['initial_source_disagreement_median'] = float(np.median(old_values))
                summary['initial_source_disagreement_p95'] = float(np.percentile(old_values,95))
                summary['initial_overlap_grid_samples'] = int(len(old_values))
            paired_low = np.minimum(low, old_low)
            paired_high = np.maximum(high, old_high)
            paired_wh = np.rint((paired_high-paired_low)/spacing).astype(int)+1
            paired_shape = (paired_wh[1], paired_wh[0])
            old_paired = np.stack([place_on_grid(xyz, uv, paired_low, paired_shape, spacing) for xyz, uv, _ in old_sources])
            new_paired = np.stack([place_on_grid(xyz, uv, paired_low, paired_shape, spacing) for xyz, uv, _ in sources])
            old_rms, old_count, _ = source_rms(old_paired)
            new_rms, new_count, _ = source_rms(new_paired)
            same_sources = np.all(np.isfinite(old_paired[..., 0]) == np.isfinite(new_paired[..., 0]), axis=0)
            paired = same_sources & (old_count > 1) & (new_count > 1)
            summary['paired_overlap_samples'] = int(paired.sum())
            if paired.any():
                summary['paired_source_disagreement_median_before_after'] = [float(np.median(a[paired])) for a in (old_rms, new_rms)]
                summary['paired_source_disagreement_p95_before_after'] = [float(np.percentile(a[paired], 95)) for a in (old_rms, new_rms)]
        summaries.append(summary)
        print(path, flush=True)
    if 'continuous' in report:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        fig.suptitle('Patches + generated fiber strip · detached islands removed', fontsize=16)
        for ax, phase in zip(axes, ('initial', 'refined')):
            info = report['continuous'].get(phase, {})
            ax.set_title(f'{phase} · one continuous component')
            if not info.get('valid'):
                ax.text(.5, .5, info.get('message', 'No export'), transform=ax.transAxes, ha='center')
                continue
            xyz, uv, step = read_surface(info)
            if ct:
                dense, _ = upsample_surface(xyz, max(1, int(round(step/2))))
                pixels = ct.sample(dense)
            else:
                pixels = np.where(xyz[..., 0] != -1, xyz[..., 2], np.nan)
            ax.imshow(np.ma.masked_invalid(pixels), origin='lower', extent=extent(xyz, uv, step), cmap='gray',
                      **({'vmin': 40, 'vmax': 180} if ct else {}))
            ax.plot(fiber_uv[:, 0], fiber_uv[:, 1], color='#18afbf', linewidth=.9)
            ax.set_facecolor('#e5e7eb'); ax.set_aspect('equal')
            summaries.append({'export': f'continuous/{phase}', **surface_metrics(xyz),
                              'dropped_single_quad_components': info.get('dropped_single_quad_components'),
                              'dropped_other_components': info.get('dropped_other_components')})
        path = root / 'continuous_preview.png'
        fig.savefig(path, dpi=150); plt.close(fig); images.append(path)
        print(path, flush=True)
    notes = ('Experimental outputs. Cyan: fixed fiber. Gray background: unsupported surface. '
             'Source disagreement is measured in 3D at common fitted UV, not ground-truth accuracy. '
             'Patch-only exports split disconnected quad regions and omit single-quad islands. '
             'The optional continuous export includes the generated fiber strip and retains only its main component. '
             'Source files are unchanged. CT previews use nearest voxel samples on bilinearly interpolated '
             'exported XYZ grids; they are inspection images, not final renders. '
             'An extra refit can reduce constraint residuals while increasing 3D source disagreement; '
             'both first-pass and refit exports are retained. Paired metrics use identical source '
             'membership at the same UV positions across passes.')
    body = '<!doctype html><meta charset="utf-8"><title>Fiber merge prototype</title>'
    body += '<style>body{font:16px system-ui;margin:30px auto;max-width:1500px;background:#fafafa}img{width:100%}pre{white-space:pre-wrap}</style>'
    body += f'<h1>{html.escape(report["fiber"])}</h1><p>{notes}</p>'
    for path in images:
        data = base64.b64encode(path.read_bytes()).decode()
        body += f'<img src="data:image/png;base64,{data}" alt="{html.escape(path.name)}">'
    body += '<h2>Measurements and patch legend</h2><pre>' + html.escape(json.dumps(summaries, indent=2)) + '</pre>'
    missing = len(ct.missing) if ct else 0
    body += f'<p>Missing CT chunks: {missing}. Missing samples remain blank.</p>'
    (root / 'preview.html').write_text(body)
    (root / 'preview_metrics.json').write_text(json.dumps({'groups': summaries, 'missing_ct_chunks': missing,
        'ct_array': str(ct_path) if ct_path else None, 'xyz_to_ct_scale': xyz_scale}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--ct-array', type=Path, help='Zarr v2 array directory containing .zarray')
    parser.add_argument('--xyz-scale', type=float, default=1, help='export XYZ to chosen CT array scale')
    parser.add_argument('--validate-only', action='store_true', help='check every exported TIFXYZ without rendering previews')
    args = parser.parse_args()
    if not np.isfinite(args.xyz_scale) or args.xyz_scale <= 0:
        parser.error('--xyz-scale must be finite and positive')
    checked = validate_exports(args.output)
    print(f'Validated {len(checked)} TIFXYZ components', flush=True)
    if not args.validate_only:
        preview(args.output, args.ct_array, args.xyz_scale)
