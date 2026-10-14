"""Matched surface CT and fitted-attachment render; native VC samples CT."""
import colorsys
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tifffile

DEFAULT_PIXEL_SIZE = 8
DEFAULT_LINE_WIDTH = 4

from vesuvius.tifxyz_label_transfer.io import load_surface
from vesuvius.tifxyz_label_transfer.self_render_tifxyz import _sample_grid_field


def canvas(surface, pixel_size):
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError('pixel_size must be positive')
    # Vertex-aligned pixels; last pixel never extends beyond the last quad.
    step_yx = 1 / np.asarray(surface.scale_yx, dtype=float)
    factor_yx = step_yx / pixel_size
    shape = tuple(int(v) for v in np.floor((np.array(surface.shape)-1)*factor_yx)+1)
    return shape, factor_yx


def band_geometry(surface, shape, factor_yx, start, end):
    yy, xx = np.meshgrid(np.arange(start, end)/factor_yx[0],
                         np.arange(shape[1])/factor_yx[1], indexing='ij')
    valid = np.ones(yy.shape, bool)
    fields = []
    for field in (surface.x, surface.y, surface.z):
        value, ok = _sample_grid_field(field, surface.valid, yy, xx)
        fields.append(value)
        valid &= ok
    return np.ascontiguousarray(np.stack(fields, -1), dtype=np.float32), valid


def inspection_crop(surface, attachments, columns):
    """Crop only the exported view, never the fitted graph or its connectivity."""
    from vesuvius.tifxyz_label_transfer.core import Surface
    a, b = map(int, columns)
    if not 0 <= a < b <= surface.shape[1] or b-a < 2:
        raise ValueError('Expected valid half-open surface column range')
    view = Surface(x=surface.x[:, a:b], y=surface.y[:, a:b], z=surface.z[:, a:b],
                   valid=surface.valid[:, a:b], scale_yx=surface.scale_yx)
    shifted = dict(attachments)
    for key in ('initial_grid_uv', 'final_grid_uv'):
        if key in shifted:
            shifted[key] = shifted[key]-np.array([a, 0])
    return view, shifted


def fiber_overlay(attachments, shape, factor_yx, width=2):
    image = Image.new('RGBA', (shape[1], shape[0]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    index = []
    for piece in np.unique(attachments['piece']):
        ids = np.flatnonzero(attachments['piece'] == piece)
        ids = ids[np.argsort(attachments['arc'][ids], kind='stable')]
        source = int(attachments['source'][ids[0]])
        if not np.all(attachments['source'][ids] == source):
            raise ValueError('A piece contains multiple source fibers')
        color = tuple(round(v*255) for v in colorsys.hsv_to_rgb((source*.61803398875)%1, .85, 1))
        xy = attachments['final_grid_uv'][ids]*factor_yx[::-1]
        if not np.isfinite(xy).all():
            raise ValueError('Nonfinite attachment UV')
        if len(xy) > 1:
            draw.line([tuple(p) for p in xy], fill=color+(220,), width=width)
        index.append(dict(piece=int(piece), source=source, color_rgb=color,
                          family=('H' if int(attachments['family'][ids[0]]) == 0 else 'V') if 'family' in attachments else 'H',
                          samples=len(ids), arc_range=[float(attachments['arc'][ids].min()),
                                                       float(attachments['arc'][ids].max())]))
    return np.asarray(image).copy(), index


def write_overlays(out, attachments, shape, factors, ct, mask, width):
    overlay, index = fiber_overlay(attachments, shape, factors, width)
    overlay[mask == 0, 3] = 0
    tifffile.imwrite(out/'fibers.tif', overlay, photometric='rgb', extrasamples=['unassalpha'],
                     compression='lzw', metadata=None)
    combined = np.repeat(ct[..., None], 3, axis=2)
    for row in range(0, shape[0], 128):
        alpha = overlay[row:row+128, :, 3:4].astype(np.float32)/255
        combined[row:row+128] = np.rint(combined[row:row+128]*(1-alpha)+overlay[row:row+128, :, :3]*alpha).astype(np.uint8)
    tifffile.imwrite(out/'ct_fibers.tif', combined, photometric='rgb', compression='lzw', metadata=None)
    (out/'fibers.json').write_text(json.dumps(index, indent=2)+'\n')
    return index


def cached_render(settings):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import sha256_file
    out = Path(settings['output_root'])
    original = json.loads((out/'config.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    for path, digest in summary['sha256'].items():
        if path != str(Path(__file__)) and sha256_file(path) != digest:
            raise ValueError(f'Render input changed: {path}')
    with np.load(original['attachments']) as f:
        attachments = {k: f[k] for k in f.files}
    surface = load_surface(original['surface'])
    if original.get('column_range') is not None:
        surface, attachments = inspection_crop(surface, attachments, original['column_range'])
    ct = (tifffile.imread(out/'ct.tif') if (out/'ct.tif').exists()
          else np.asarray(Image.open(out/'ct.jpg')))
    mask = tifffile.imread(out/'valid.tif')
    shape = tuple(summary['shape'])
    if ct.shape != shape or mask.shape != shape:
        raise ValueError('Cached CT/mask dimensions do not match render')
    return out, original, summary, surface, attachments, ct, mask


def fold_pixels(surface, shape, factors):
    """Nearest containing quad, not interpolated vertex labels; CT registration."""
    from fiber_surface import geometry_masks
    degenerate, folded = geometry_masks(np.stack([surface.x, surface.y, surface.z], -1))
    valid = (surface.valid[:-1,:-1] & surface.valid[1:,:-1] &
             surface.valid[:-1,1:] & surface.valid[1:,1:])
    folded &= valid
    degenerate &= valid
    rows = np.minimum(np.floor(np.arange(shape[0])/factors[0]).astype(int), folded.shape[0]-1)
    cols = np.minimum(np.floor(np.arange(shape[1])/factors[1]).astype(int), folded.shape[1]-1)
    return folded[rows[:,None],cols[None,:]], folded, degenerate


def write_fold_maps(out, surface, shape, factors, ct, mask, column_offset=0):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import sha256_file
    pixels, folded, degenerate = fold_pixels(surface, shape, factors)
    pixels &= mask != 0
    tifffile.imwrite(out/'folded_quads.tif', pixels.astype(np.uint8)*255,
                     compression='lzw', metadata=None)
    tifffile.imwrite(out/'folded_quad_grid.tif', folded.astype(np.uint8)*255,
                     compression='lzw', metadata=None)
    combined = np.repeat(ct[...,None],3,axis=2)
    # Preserve the CT texture beneath a conspicuous red tint.
    combined[pixels] = np.rint(.35*combined[pixels]+.65*np.array([255,0,0])).astype(np.uint8)
    tifffile.imwrite(out/'ct_folds.tif', combined, photometric='rgb', compression='lzw', metadata=None)
    Image.fromarray(combined).save(out/'ct_folds.jpg', quality=95)
    cells = np.argwhere(folded)
    global_cells = cells+np.array([0,column_offset])
    info = dict(folded_quads=int(folded.sum()), degenerate_quads=int(degenerate.sum()),
                shape=list(shape), folded_pixels_percent=float(100*pixels.mean()),
                local_quad_row_col=cells.tolist(), source_quad_row_col=global_cells.tolist(),
                column_offset=int(column_offset), color='red = quad triangle normal dot <= 0',
                definition='Existing geometry_masks: dot of the two quad triangle normals <= 0; not global intersections',
                code_sha256=sha256_file(Path(__file__)))
    (out/'folds.json').write_text(json.dumps(info,indent=2)+'\n')
    return {k:v for k,v in info.items() if not k.endswith('row_col')}


def refresh_fold_maps(settings):
    start = time.monotonic()
    out, original, summary, surface, _, ct, mask = cached_render(settings)
    info = write_fold_maps(out, surface, tuple(summary['shape']),
        np.array(summary['grid_to_pixel_xy'])[::-1], ct, mask,
        (original.get('column_range') or [0])[0])
    print(json.dumps(dict(**info, reused_ct=True, seconds=time.monotonic()-start)), flush=True)
    return info


def refresh_overlays(settings):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import sha256_file
    start = time.monotonic()
    out, original, summary, surface, attachments, ct, mask = cached_render(settings)
    shape = tuple(summary['shape'])
    width = int(settings['line_width'])
    if width < 1:
        raise ValueError('Line width must be positive')
    index = write_overlays(out, attachments, shape,
        np.array(summary['grid_to_pixel_xy'])[::-1], ct, mask, width)
    result = dict(line_width=width, pieces=len(index), seconds=time.monotonic()-start,
                  reused_ct=True, code_sha256=sha256_file(Path(__file__)))
    (out/'overlay_refresh.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result


def run(settings):
    if settings.get('folds_only'):
        return refresh_fold_maps(settings)
    if settings.get('overlay_only'):
        return refresh_overlays(settings)
    import vc.volume as vc
    start = time.monotonic()
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import sha256_file
    files = [Path(settings['attachments']), Path(settings['surface'])/'meta.json',
             *[Path(settings['surface'])/(c+'.tif') for c in 'xyz'], Path(__file__)]
    hashes = {str(p):sha256_file(p) for p in files}
    out = Path(settings['output_root'])
    if out.exists() and any(p.name != 'config.json' for p in out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(settings, indent=2)+'\n')
    surface = load_surface(settings['surface'])
    with np.load(settings['attachments']) as f:
        attachments = {k: f[k] for k in f.files}
    if settings.get('column_range') is not None:
        surface, attachments = inspection_crop(surface, attachments, settings['column_range'])
        from vesuvius.tifxyz.types import Tifxyz
        from vesuvius.tifxyz.writer import write_tifxyz
        write_tifxyz(out/'inspection.tifxyz', Tifxyz(surface.x, surface.y, surface.z,
            uuid=out.name+'-inspection', _scale=surface.scale_yx, _mask=surface.valid,
            extra=dict(inspection_only=True, source_surface=settings['surface'],
                       source_column_range=settings['column_range'])), compression='lzw')
    shape, factors = canvas(surface, settings.get('pixel_size', DEFAULT_PIXEL_SIZE))
    vc.initialize_chunk_cache(16*1024**3, 64, 8, 32)
    # Older installed bindings take an explicit cache root; current bindings
    # resolve the shared root internally. Never let the older default use cwd.
    try:
        volume = vc.Volume.open_url(settings['volume_url'], cache_root=vc.global_cache_root())
    except TypeError:
        volume = vc.Volume.open_url(settings['volume_url'])
    level = settings.get('level', 2)
    if not volume.has_scale_level(level) or volume.dtype != 'uint8':
        raise ValueError(f'Expected uint8 CT with level {level}; got {volume.dtype}')
    ct = np.zeros(shape, np.uint8)
    mask = np.zeros(shape, np.uint8)
    bands = []
    print(f'Rendering {shape[1]}x{shape[0]}, CT level {level}, cache {volume.remote_cache_path}', flush=True)
    for row in range(0, shape[0], settings.get('band_rows', 64)):
        end = min(row+settings.get('band_rows', 64), shape[0])
        t = time.monotonic()
        xyz, valid = band_geometry(surface, shape, factors, row, end)
        values, covered, stats = volume.sample_coords(xyz, valid, level=level,
            sampling='trilinear', blocking=True, sampler_threads=settings.get('threads', 64))
        if stats.get('error_chunks', 0):
            raise RuntimeError(f'CT fetch/decode errors in band {row}: {stats}')
        ct[row:end] = np.asarray(values).reshape(end-row, shape[1])
        mask[row:end] = np.asarray(covered).reshape(end-row, shape[1])*255
        bands.append(dict(row=row, end=end, seconds=time.monotonic()-t, stats=dict(stats)))
        print(f'CT rows {end}/{shape[0]} ({end/shape[0]:.1%}), {bands[-1]["seconds"]:.2f}s/band', flush=True)
    ct[mask == 0] = 0
    Image.fromarray(ct).save(out/'ct.jpg', quality=95)
    tifffile.imwrite(out/'valid.tif', mask, compression='lzw', metadata=None)
    index = write_overlays(out, attachments, shape, factors, ct, mask, settings.get('line_width', DEFAULT_LINE_WIDTH))
    folds = write_fold_maps(out, surface, shape, factors, ct, mask,
                           (settings.get('column_range') or [0])[0])
    summary = dict(shape=list(shape), pixel_size_uv_base=settings.get('pixel_size', DEFAULT_PIXEL_SIZE),
        grid_to_pixel_xy=factors[::-1].tolist(), origin='pixel (0,0) = surface vertex (0,0)',
        valid_percent=float(np.count_nonzero(mask)/mask.size*100),
        seconds=time.monotonic()-start, pieces=len(index), sources=len(np.unique(attachments['source'])),
        sha256=hashes, bands=bands, remote_cache_path=str(volume.remote_cache_path), folds=folds,
        note='H/V attachment polylines at saved final UV; not nearest projection. V is offset supervision, not on-surface data. Surface folds remain. Zero-offset single CT slice, no slab projection.')
    (out/'fibers.json').write_text(json.dumps(index, indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('bands','sha256')}), flush=True)
    return summary
