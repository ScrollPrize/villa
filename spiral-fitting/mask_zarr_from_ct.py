r"""Build smooth CT envelopes in Z batches and mask local Zarr v2 copies.

Example (the CT and normals here use different pyramid level numbers)::

    python mask_zarr_from_ct.py /data/ct.zarr/2 \
        /data/nx.ome.zarr /data/ny.ome.zarr --output-dir /data/masked

All OME pyramid levels are processed. Sources are always read-only. See
docs/ct_region_masking.md for grid mapping, memory, and partial test outputs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numcodecs
import numpy as np

from ct_region_mask import MaskSettings, make_ct_mask
from zarr_chunk_io import _chunk_path, _make_chunk_reader


class LocalArray:
    """Small local v2 adapter; avoids a second I/O thread pool in zarr-python."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.meta = json.loads((self.path / '.zarray').read_text())
        m = self.meta
        self.shape = tuple(m['shape'])
        self.chunks = tuple(m['chunks'])
        self.dtype = np.dtype(m['dtype'])
        self.order = m.get('order', 'C')
        self.separator = m.get('dimension_separator', '.')
        if (m.get('zarr_format') != 2 or len(self.shape) != 3
                or len(self.chunks) != 3 or min(*self.shape, *self.chunks) <= 0
                or self.dtype.kind not in 'buif' or self.order not in ('C', 'F')
                or self.separator not in ('.', '/') or m.get('filters')):
            raise ValueError(f'{path}: requires numeric 3D ZYX Zarr v2, without filters')
        fill = m.get('fill_value', 0)
        self.fill = np.asarray(0 if fill is None else fill, dtype=self.dtype).item()
        self.reader = _make_chunk_reader(
            m.get('compressor'), math.prod(self.chunks) * self.dtype.itemsize)

    def chunk_path(self, key):
        return _chunk_path(self.path, key, self.separator)

    def read_chunk(self, key):
        path = self.chunk_path(key)
        if not path.exists():
            return np.full(self.chunks, self.fill, dtype=self.dtype, order=self.order)
        return self.reader(path).view(self.dtype).reshape(self.chunks, order=self.order)

    def encoder(self):
        config = self.meta.get('compressor')
        if config and (config['id'] in ('vcz1', 'vc-delta3d')
                       or config.get('quantize') is not None):
            raise ValueError(f'{self.path}: target codec {config["id"]} is not supported; '
                             'use lossless Blosc, gzip, zstd, or uncompressed targets')
        return None if config is None else numcodecs.get_codec(config)


def bounded_map(pool, function, items, limit):
    """At most limit queued tasks/results; never submit an entire volume."""
    iterator = iter(items)
    pending = set()
    exhausted = False
    while pending or not exhausted:
        while not exhausted and len(pending) < limit:
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending.add(pool.submit(function, item))
        if pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()


def read_ct_batch(array, z0, z1, pool, workers):
    out = np.full((z1 - z0, *array.shape[1:]), array.fill, dtype=array.dtype)
    c = array.chunks
    keys = itertools.product(range(z0 // c[0], (z1 - 1) // c[0] + 1),
                             range(math.ceil(array.shape[1] / c[1])),
                             range(math.ceil(array.shape[2] / c[2])))

    def read(key):
        origin = tuple(k * n for k, n in zip(key, c))
        a = (max(z0, origin[0]), origin[1], origin[2])
        b = tuple(min(end, start + size) for end, start, size in
                  zip((z1, *array.shape[1:]), origin, c))
        data = array.read_chunk(key)
        out[a[0] - z0:b[0] - z0, a[1]:b[1], a[2]:b[2]] = data[
            tuple(slice(lo - start, hi - start) for lo, hi, start in zip(a, b, origin))]

    for _ in bounded_map(pool, read, keys, 2 * workers):
        pass
    return out


def _json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _level_scale(dataset):
    scale = [Fraction(1)] * 3
    for transform in dataset.get('coordinateTransformations', []):
        if transform['type'] == 'scale':
            values = transform['scale']
            if len(values) != 3:
                raise ValueError('pyramid scales must contain three ZYX values')
            scale = [a * Fraction(str(b)) for a, b in zip(scale, values)]
        elif transform['type'] == 'translation':
            if any(float(v) != 0 for v in transform['translation']):
                raise ValueError('translated target grids are not supported')
        else:
            raise ValueError(f'unsupported coordinate transform: {transform}')
    if any(s <= 0 for s in scale):
        raise ValueError('pyramid scales must be positive')
    return tuple(scale)


def discover_levels(root):
    """Return array paths and relative scales; only declared OME levels are used."""
    if (root / '.zarray').is_file():
        return [(Path('.'), (Fraction(1),) * 3)]
    attrs = _json(root / '.zattrs', {})
    multiscales = attrs.get('multiscales', [])
    if len(multiscales) != 1:
        raise ValueError(f'{root}: pass a v2 array or a single OME multiscale group')
    ms = multiscales[0]
    axes = [a if isinstance(a, str) else a['name'] for a in ms.get('axes', [])]
    if axes != ['z', 'y', 'x']:
        raise ValueError(f'{root}: expected OME axes [z,y,x], got {axes}')
    _level_scale(ms)  # Reject translated/non-supported group transforms too.
    levels = []
    for ds in ms['datasets']:
        relative = Path(ds['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('OME dataset paths must stay within the input root')
        levels.append((relative, _level_scale(ds)))
    if not levels or len({p for p, _ in levels}) != len(levels):
        raise ValueError('OME pyramid needs unique dataset paths')
    levels.sort(key=lambda item: math.prod(item[1]))
    base = levels[0][1]
    return [(path, tuple(a / b for a, b in zip(scale, base))) for path, scale in levels]


@dataclass
class Target:
    source: LocalArray
    output: Path
    scale: tuple[Fraction, Fraction, Fraction]
    codec: object


def _mapped_indices(start, stop, scale):
    return np.arange(start, stop, dtype=np.int64) * scale.numerator // scale.denominator


def plan_targets(ct, roots, output_dir, target_scale):
    targets = []
    output_roots = []
    for root in roots:
        dest = output_dir / root.name
        if dest.exists() or dest in output_roots:
            raise ValueError(f'output already exists or has a duplicate name: {dest}')
        output_roots.append(dest)
        levels = discover_levels(root)
        base = LocalArray(root / levels[0][0])
        if target_scale is None and base.shape != ct.shape:
            raise ValueError(f'{root}: finest shape {base.shape} != CT {ct.shape}; '
                             'specify --target-scale Z Y X (CT voxels per finest target voxel)')
        base_scale = target_scale or (Fraction(1),) * 3
        for relative, relative_scale in levels:
            arr = base if relative == levels[0][0] else LocalArray(root / relative)
            scale = tuple(a * b for a, b in zip(base_scale, relative_scale))
            for axis, (n, size, s) in enumerate(zip(arr.shape, ct.shape, scale)):
                last = ((n - 1) * s.numerator) // s.denominator
                if s <= 0 or last >= size or size - 1 - last >= max(1, math.ceil(s)):
                    raise ValueError(f'{arr.path}: axis {axis} does not cover CT with scale {s}')
            mapped_depth = arr.chunks[0] * scale[0]
            if mapped_depth.denominator != 1:
                raise ValueError('target chunk depth times Z scale must be an integer')
            targets.append(Target(arr, dest / relative, scale, arr.encoder()))
    return targets, output_roots


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def create_outputs(roots, output_roots, targets, provenance):
    for root, dest in zip(roots, output_roots):
        dest.mkdir(parents=True, exist_ok=False)
        if (root / '.zgroup').exists():
            shutil.copy2(root / '.zgroup', dest / '.zgroup')
        attrs = _json(root / '.zattrs', {})
        attrs['ct_region_mask'] = provenance
        write_json(dest / '.zattrs', attrs)
    for target in targets:
        dest, source = target.output, target.source
        dest.mkdir(parents=True, exist_ok=True)
        # Preserve nested OME group metadata, not just the array metadata.
        for root, out in zip(roots, output_roots):
            if source.path.is_relative_to(root):
                parent = source.path.parent
                while parent != root and parent.is_relative_to(root):
                    out_parent = out / parent.relative_to(root)
                    out_parent.mkdir(parents=True, exist_ok=True)
                    for name in ('.zgroup', '.zattrs'):
                        if (parent / name).exists():
                            shutil.copy2(parent / name, out_parent / name)
                    parent = parent.parent
                break
        meta = dict(source.meta, fill_value=0)
        write_json(dest / '.zarray', meta)
        if dest not in output_roots and (source.path / '.zattrs').exists():
            shutil.copy2(source.path / '.zattrs', dest / '.zattrs')


def apply_chunk(item, masks, z0, z1):
    """Each worker owns one spatial chunk across targets sharing its grid."""
    group, key = item
    first = group[0]
    shape, chunks, scale = first.source.shape, first.source.chunks, first.scale
    starts = tuple(k * c for k, c in zip(key, chunks))
    stops = tuple(min(s + c, n) for s, c, n in zip(starts, chunks, shape))
    indices = [_mapped_indices(a, b, s) for a, b, s in zip(starts, stops, scale)]
    selected = (indices[0] >= z0) & (indices[0] < z1)
    keep = np.zeros(tuple(b - a for a, b in zip(starts, stops)), dtype=bool)
    keep[selected] = masks[np.ix_(indices[0][selected] - z0, indices[1], indices[2])]
    counts = dict(copied=0, masked=0, skipped=0, boundary_zeroed=0)
    for target in group:
        source = target.source
        src = source.chunk_path(key)
        if not keep.any() or (not src.exists() and source.fill == 0):
            counts['skipped'] += 1
            continue
        dest = _chunk_path(target.output, key, source.separator)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_name(dest.name + f'.tmp-{threading.get_ident()}')
        if keep.all() and src.exists():
            shutil.copyfile(src, temporary)
            counts['copied'] += 1
        else:
            data = source.read_chunk(key).copy(order=source.order)
            actual = tuple(slice(0, b - a) for a, b in zip(starts, stops))
            view = data[actual]
            counts['boundary_zeroed'] += int(np.count_nonzero(view[~keep]))
            view[~keep] = 0
            if not np.any(view):
                counts['skipped'] += 1
                continue
            encoded = (data.tobytes(order=source.order) if target.codec is None
                       else target.codec.encode(data.ravel(order=source.order)))
            temporary.write_bytes(encoded)
            counts['masked'] += 1
        os.replace(temporary, dest)
    return counts


def run(ct_path, target_paths, output_dir, *, settings=MaskSettings(), workers=2,
        mask_workers=4, batch_depth=None, target_scale=None, z_range=None):
    started = time.perf_counter()
    if workers < 1 or mask_workers < 1:
        raise ValueError('worker counts must be positive')
    cv2.setNumThreads(1)
    numcodecs.blosc.set_nthreads(1)
    ct = LocalArray(ct_path)
    roots = [Path(p).resolve() for p in target_paths]
    out = Path(output_dir).resolve()
    if not roots or len(set(roots)) != len(roots):
        raise ValueError('provide one or more distinct targets')
    for root in [ct.path, *roots]:
        if out == root or out.is_relative_to(root):
            raise ValueError('output directory must be outside all source stores')
    if target_scale is not None:
        target_scale = tuple(Fraction(str(s)) for s in target_scale)
        if len(target_scale) != 3 or min(target_scale) <= 0:
            raise ValueError('target scale needs three positive ZYX values')
    targets, output_roots = plan_targets(ct, roots, out, target_scale)
    alignment = math.lcm(*(int(t.source.chunks[0] * t.scale[0]) for t in targets))
    depth = math.lcm(ct.chunks[0], alignment) if batch_depth is None else batch_depth
    if depth < 1 or depth % alignment:
        raise ValueError(f'--batch-depth must be a positive multiple of {alignment}')
    start, stop = (0, ct.shape[0]) if z_range is None else z_range
    if not 0 <= start < stop <= ct.shape[0]:
        raise ValueError(f'Z range must satisfy 0 <= start < stop <= {ct.shape[0]}')
    provenance = dict(version=1, complete=False, ct=str(ct.path), settings=asdict(settings),
                      include_internal_cavities=True,
                      z_range=[start, stop], partial=(start, stop) != (0, ct.shape[0]),
                      mapping='CT index = floor(target index * scale), zero origin',
                      levels=[dict(source=str(t.source.path), output=str(t.output),
                                   scale_zyx=[str(s) for s in t.scale]) for t in targets])
    # Validate the whole plan before creating any output.
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'mask_report.json'
    if report_path.exists():
        raise ValueError(f'output report already exists: {report_path}')
    create_outputs(roots, output_roots, targets, provenance)
    report = dict(provenance, workers=workers, mask_workers=mask_workers,
                  batch_depth=depth, batches=[], totals=dict(copied=0, masked=0, skipped=0,
                                                          boundary_zeroed=0))
    write_json(report_path, report)
    groups = {}
    for target in targets:
        key = (target.source.shape, target.source.chunks, target.scale)
        groups.setdefault(key, []).append(target)
    resident_mib = depth * math.prod(ct.shape[1:]) * (ct.dtype.itemsize + 1) / 2**20
    print(f'CT {ct.shape}; {len(targets)} target levels; batch depth {depth}; '
          f'CT + mask buffers up to {resident_mib:.0f} MiB (plus worker scratch)', flush=True)
    try:
        with ThreadPoolExecutor(workers) as io_pool, ThreadPoolExecutor(mask_workers) as mask_pool:
            z0 = start
            while z0 < stop:
                z1 = min(stop, (z0 // depth + 1) * depth)
                t0 = time.perf_counter()
                ct_batch = read_ct_batch(ct, z0, z1, io_pool, workers)
                t1 = time.perf_counter()
                masks = np.empty(ct_batch.shape, dtype=bool)

                def make_slice(index):
                    masks[index] = make_ct_mask(ct_batch[index], settings)

                for _ in bounded_map(mask_pool, make_slice, range(z1 - z0), mask_workers):
                    pass
                del ct_batch
                t2 = time.perf_counter()

                def jobs():
                    for group in groups.values():
                        target = group[0]
                        c, shape, s = target.source.chunks, target.source.shape, target.scale[0]
                        a = math.ceil(Fraction(z0) / s)
                        b = min(shape[0], math.ceil(Fraction(z1) / s))
                        if a >= b:
                            continue
                        for key in itertools.product(range(a // c[0], (b - 1) // c[0] + 1),
                                                     range(math.ceil(shape[1] / c[1])),
                                                     range(math.ceil(shape[2] / c[2]))):
                            yield group, key

                counts = dict.fromkeys(report['totals'], 0)
                for result in bounded_map(io_pool, lambda job: apply_chunk(job, masks, z0, z1),
                                          jobs(), 2 * workers):
                    for name, value in result.items():
                        counts[name] += value
                t3 = time.perf_counter()
                batch = dict(z_range=[z0, z1], read_seconds=t1 - t0,
                             mask_seconds=t2 - t1, apply_seconds=t3 - t2, **counts)
                report['batches'].append(batch)
                for name in counts:
                    report['totals'][name] += counts[name]
                write_json(report_path, report)
                print(f'Z {z0}:{z1}: read {t1-t0:.2f}s, mask {t2-t1:.2f}s, '
                      f'apply {t3-t2:.2f}s; {counts}', flush=True)
                del masks
                z0 = z1
        provenance['complete'] = True
        for dest in output_roots:
            attrs = _json(dest / '.zattrs', {})
            attrs['ct_region_mask'] = provenance
            write_json(dest / '.zattrs', attrs)
        report['complete'] = True
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ct', help='path to a specific 3D CT Zarr v2 array (including its level)')
    parser.add_argument('targets', nargs='+', help='target arrays or OME Zarr roots; all declared levels are masked')
    parser.add_argument('--output-dir', required=True, help='new copies are created here, named after target roots')
    parser.add_argument('--workers', type=int, default=2, help='parallel chunk I/O workers (default: 2)')
    parser.add_argument('--mask-workers', type=int, default=4, help='parallel CT slice workers (default: 4)')
    parser.add_argument('--batch-depth', type=int, help='CT Z slices per batch; default aligns all chunk grids')
    parser.add_argument('--target-scale', nargs=3, metavar=('Z', 'Y', 'X'),
                        help='CT voxels per finest target voxel; omit only for identical finest-grid shapes')
    parser.add_argument('--z-range', type=int, nargs=2, metavar=('START', 'STOP'),
                        help='partial test in CT coordinates; output outside this range is zero')
    parser.add_argument('--threshold', type=float, default=65)
    parser.add_argument('--ct-sigma', type=float, default=1)
    parser.add_argument('--seed-radius', type=int, default=3)
    parser.add_argument('--closing-radius', type=int, default=12)
    parser.add_argument('--boundary-sigma', type=float, default=4)
    args = parser.parse_args()
    try:
        settings = MaskSettings(args.threshold, args.ct_sigma, args.seed_radius,
                                args.closing_radius, args.boundary_sigma)
        report = run(args.ct, args.targets, args.output_dir, settings=settings,
                     workers=args.workers, mask_workers=args.mask_workers,
                     batch_depth=args.batch_depth, target_scale=args.target_scale,
                     z_range=args.z_range)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f'Finished in {report["elapsed_seconds"]:.2f}s. Report: '
          f'{Path(args.output_dir) / "mask_report.json"}')


if __name__ == '__main__':
    main()
