"""Real-CT CPU elastic experiments; reads only fully cached raw Zarr v2 chunks.

Run with the optional CMake extension on PYTHONPATH. See the comparison README.
No dataset writes, downloads, or installations. Timing excludes I/O and plots.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import types

import numpy as np
import torch
import _elastic_lab as lab

ROOT = Path(__file__).resolve().parents[3]
ARRAY = Path('/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2')
CHUNKS = {'parallel': (70, 35, 25), 'curved': (40, 30, 30), 'compact': (40, 22, 40)}
METHODS = ['dense', 'coarse', 'bspline', 'normal_bspline']
TITLES = {'dense': 'Dense Gaussian', 'coarse': 'Coarse Gaussian (4 voxel grid)',
          'bspline': 'Cubic B-spline', 'normal_bspline': 'Normal-guided B-spline',
          'python': 'Python Gaussian'}
SCALES, STRENGTHS = [8, 16, 32], [.5, 1.5, 4., 8.]
EDGE, HALO, BOUND = 128, 64, .8
EMPTY = np.empty((0, 3), dtype=np.float64)


def read_crop(array, chunk):
    metadata = json.loads((array / '.zarray').read_text())
    assert metadata['chunks'] == [128]*3 and metadata['dtype'] == '|u1'
    assert metadata['compressor'] is None and metadata['filters'] is None
    assert metadata['order'] == 'C' and metadata['dimension_separator'] == '/'
    origin = np.array(chunk)*EDGE-HALO
    crop = np.empty((EDGE+2*HALO,)*3, dtype=np.uint8)
    for index in itertools.product(*(range(int(s//128), int((s+255)//128)+1) for s in origin)):
        path = array.joinpath(*map(str, index))
        # A missing cache file is an error, never silently treated as background.
        block = np.fromfile(path, dtype=np.uint8).reshape((128,)*3)
        start = np.array(index)*128
        lo, hi = np.maximum(start, origin), np.minimum(start+128, origin+256)
        crop[tuple(slice(int(a), int(b)) for a, b in zip(lo-origin, hi-origin))] = block[
            tuple(slice(int(a), int(b)) for a, b in zip(lo-start, hi-start))]
    return crop, origin


def normal_lattice(raw, origin, scale):
    spacing = 2*scale
    n = (EDGE-1)//spacing+4
    axis = np.clip((np.arange(n)-1)*spacing, 0, EDGE-1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    points = np.ascontiguousarray(points+origin+HALO, dtype=np.float64)
    result = lab.normal_points(raw, tuple(origin), points)
    # Reject nearly degenerate tensors; preserve isotropic noise at these sites.
    result['normals'][result['gaps'] < .2] = 0
    result['valid_fraction'] = float(np.mean(np.linalg.norm(result['normals'], axis=1) > .5))
    result['unique_queries'] = len(np.unique(points, axis=0))
    return result


def reset(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def python_transform(scale):
    if 'vesuvius' not in sys.modules:
        package = types.ModuleType('vesuvius')
        package.__path__ = [str(ROOT/'vesuvius/src/vesuvius')]
        sys.modules['vesuvius'] = package
    cls = getattr(importlib.import_module('vesuvius.models.augmentation.transforms.spatial.spatial'), 'SpatialTransform')
    return cls(patch_size=(EDGE,)*3, patch_center_dist_from_border=0, random_crop=False,
               p_elastic_deform=1, elastic_deform_scale=scale/EDGE, elastic_deform_magnitude=4.,
               p_synchronize_def_scale_across_axes=1, mode_seg='nearest')


def python_run(transform, image, rms=None):
    start = time.perf_counter()
    params = transform.get_parameters(image=image)
    if rms is not None:
        field = lab.normalize(np.ascontiguousarray(params['elastic_offsets'].numpy(), dtype=np.float64), rms, BOUND)
        params['elastic_offsets'] = torch.from_numpy(field.astype(np.float32))
    field_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    result = transform.apply({'image': image}, **params)['image'][0].numpy()
    warp_ms = (time.perf_counter()-start)*1000
    if rms is None:
        field = None  # No diagnostic conversion in the unmodified Python benchmark.
    return dict(image=result, field=field, field_ms=field_ms, warp_ms=warp_ms)


def stats(samples):
    v = np.asarray(samples)
    return dict(mean_ms=float(v.mean()), p50_ms=float(np.median(v)), p95_ms=float(np.percentile(v, 95)),
                min_ms=float(v.min()), max_ms=float(v.max()), samples_ms=v.tolist())


def metrics(field, requested):
    rms = float(np.sqrt(np.mean(np.sum(field**2, axis=-1))))
    maxima = np.stack([np.abs(np.diff(field, axis=a)).max(axis=(0, 1, 2)) for a in range(3)])
    small = field[::2, ::2, ::2]
    jac = np.empty((*small.shape[:3], 3, 3))
    for c in range(3):
        for axis, derivative in enumerate(np.gradient(small[..., c], 2., edge_order=2)):
            jac[..., c, axis] = derivative + (c == axis)
    det = np.linalg.det(jac)
    outside = np.zeros(field.shape[:3], dtype=bool)
    for axis in range(3):
        shape = [1]*3; shape[axis] = EDGE
        coord = np.arange(EDGE).reshape(shape)+HALO+field[..., axis]
        outside |= (coord < 0) | (coord >= EDGE+2*HALO-1)
    correlations = {}
    for lag in (8, 16, 32):
        dots, energies = 0., 0.
        for a in range(3):
            left, right = [slice(None)]*4, [slice(None)]*4
            left[a], right[a] = slice(None, -lag), slice(lag, None)
            u, v = field[tuple(left)], field[tuple(right)]
            dots += np.sum(u*v); energies += np.sqrt(np.sum(u*u)*np.sum(v*v))
        correlations[str(lag)] = float(dots/energies) if energies else 1.
    return dict(rms=rms, requested_rms=requested, cap_applied=rms < requested*.999,
                lipschitz_bound=float(maxima.sum(axis=0).max()),
                sampled_det_percentiles=np.percentile(det, [0, 1, 50, 99, 100]).tolist(),
                outside_fraction=float(outside.mean()), correlation=correlations)


def planes(volume):
    return [volume[64], volume[:, 64], volume[:, :, 64]]


def panel(ax, image, title):
    ax.imshow(image, cmap='gray', vmin=0, vmax=190/255, interpolation='nearest')
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def render(output, name, original, visuals):
    import matplotlib.pyplot as plt
    columns = ['original']+METHODS+['python']
    fig, axes = plt.subplots(3, len(columns), figsize=(16, 8), layout='constrained')
    for col, method in enumerate(columns):
        record = None if method == 'original' else visuals[(method, 16, 4.)]
        volume = original if record is None else record['image']
        for row, (plane, orientation) in enumerate(zip(planes(volume), ['YX', 'ZX', 'ZY'])):
            title = 'Original' if record is None else TITLES[method]
            strength = '' if record is None else f' | RMS {record["metrics"]["rms"]:.2f}'
            panel(axes[row, col], plane, title+f'\n{orientation}'+strength)
    fig.suptitle(f'Paris 4: {name} appearance | scale 16, requested RMS 4 voxels | seed 123\n'
                 '128 voxel output; same location, slices and intensity window')
    fig.savefig(output/f'{name}_overview.png', dpi=160); plt.close(fig)
    for method in METHODS+['python']:
        fig, axes = plt.subplots(3, 1+len(STRENGTHS), figsize=(15, 9), layout='constrained')
        for row, scale in enumerate(SCALES):
            panel(axes[row, 0], original[64], f'Original | scale {scale}')
            for col, strength in enumerate(STRENGTHS, 1):
                record = visuals[(method, scale, strength)]
                m = record['metrics']
                cap = ' (limited)' if m['cap_applied'] else ''
                panel(axes[row, col], record['image'][64], f'RMS {strength:g} → {m["rms"]:.2f}{cap}')
        fig.suptitle(f'{name} | {TITLES[method]} | YX slice | seed 123\n'
                     'Rows: nominal scale 8 / 16 / 32 voxels; columns: requested RMS 0.5 / 1.5 / 4 / 8 voxels')
        fig.savefig(output/f'{name}_{method}_parameters.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, len(METHODS), figsize=(14, 7), layout='constrained')
    for col, method in enumerate(METHODS):
        f = visuals[(method, 16, 4.)]['field'][64]
        yy, xx = np.indices((EDGE, EDGE), dtype=float)
        # Show the inverse sampling map; z displacement is omitted in this 2D projection.
        y, x = yy+f[..., 1], xx+f[..., 2]
        ax = axes[0, col]
        for i in range(0, EDGE, 8):
            ax.plot(x[i], y[i], color='steelblue', linewidth=.7)
            ax.plot(x[:, i], y[:, i], color='steelblue', linewidth=.7)
        ax.set_aspect('equal'); ax.invert_yaxis(); ax.set_title(TITLES[method], fontsize=10)
        ax.set_xlim(-12, 140); ax.set_ylim(140, -12)
        axes[1, col].imshow(np.linalg.norm(f, axis=-1), vmin=0, vmax=12, cmap='magma')
        axes[1, col].set_title('Displacement magnitude (0–12 voxels)', fontsize=9)
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
    fig.suptitle(f'{name} | deformation diagnostics | scale 16, requested RMS 4 | seed 123')
    fig.savefig(output/f'{name}_deformation.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--array', type=Path, default=ARRAY)
    parser.add_argument('--output', type=Path, default=ROOT/'vc_reader/elastic_comparison')
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--chunks', nargs='+', choices=list(CHUNKS), default=list(CHUNKS))
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--visuals-only', action='store_true', help='Regenerate images and metrics, preserving existing timing samples')
    args = parser.parse_args()
    if args.reps < 1 or args.warmup < 0:
        parser.error('reps must be positive and warmup nonnegative')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((args.output/'results.json').read_text()) if args.visuals_only else None
    if previous and set(args.chunks) != {v['name'] for v in previous['inputs']}:
        parser.error('--visuals-only must regenerate every previously measured chunk')
    report = dict(array=str(args.array), edge=EDGE, halo=HALO, bound=BOUND, seed=123,
                  scales=SCALES, strengths=STRENGTHS, reps=args.reps, warmup=args.warmup,
                  python=sys.version, numpy=np.__version__, torch=torch.__version__, platform=platform.platform(),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                  extension_sha256=hashlib.sha256(Path(lab.__file__).read_bytes()).hexdigest(),
                  timings=[], tensors=[], cases=[], inputs=[])
    if previous:
        for key in ('extension_sha256', 'array', 'edge', 'halo', 'bound', 'seed', 'scales'):
            assert previous[key] == report[key], f'{key} changed since timing run'
    for name in args.chunks:
        print(f'Reading {name}: chunk {CHUNKS[name]}', flush=True)
        raw, origin = read_crop(args.array, CHUNKS[name])
        image = raw.astype(np.float32)/255
        image_before = image.copy()
        tensor = torch.from_numpy(image[None])
        original = image[HALO:HALO+EDGE, HALO:HALO+EDGE, HALO:HALO+EDGE]
        report['inputs'].append(dict(name=name, chunk=list(CHUNKS[name]), origin=origin.tolist(),
                                    raw_sha256=hashlib.sha256(raw).hexdigest()))
        if previous:
            prior = next(v for v in previous['inputs'] if v['name'] == name)
            assert prior == report['inputs'][-1], 'CT input changed since timing run'
        visuals = {}
        for scale in SCALES:
            start = time.perf_counter()
            normals = normal_lattice(raw, origin, scale)
            wall_ms = (time.perf_counter()-start)*1000
            report['tensors'].append(dict(chunk=name, scale=scale, wall_ms=wall_ms,
                                          kernel_ms=normals['ms'], valid_fraction=normals['valid_fraction'],
                                          unique_queries=normals['unique_queries']))
            print(f'  scale {scale}: normals {wall_ms:.1f} ms, accepted {normals["valid_fraction"]:.1%}', flush=True)
            transform = python_transform(scale)
            functions = {method: (lambda method=method: lab.run(image, method, EDGE, float(HALO), float(scale),
                        4., BOUND, 123, normals['normals'] if method == 'normal_bspline' else EMPTY)) for method in METHODS}
            functions['python_original'] = lambda: python_run(transform, tensor)
            functions['python_matched'] = lambda: python_run(transform, tensor, 4.)
            samples = {method: {'total': [], 'field': [], 'warp': []} for method in functions}
            # Interleave methods within repetitions, rotating order to reduce ordering bias.
            for rep in ([] if args.visuals_only else range(-args.warmup, args.reps)):
                order = list(functions); shift = rep % len(order); order = order[shift:]+order[:shift]
                for method in order:
                    reset(123)
                    start = time.perf_counter(); record = functions[method](); elapsed = (time.perf_counter()-start)*1000
                    if rep >= 0:
                        for key, value in [('total', elapsed), ('field', record['field_ms']), ('warp', record['warp_ms'])]:
                            samples[method][key].append(value)
                    del record
            for method, values in samples.items():
                if args.visuals_only: continue
                row = dict(chunk=name, scale=scale, method=method, **{k: stats(v) for k, v in values.items()})
                row['compute'] = stats(np.array(values['field'])+values['warp'])
                report['timings'].append(row)
                print(f'    {method:17s} {row["total"]["p50_ms"]:8.2f} ms', flush=True)
            # Generate once per method, then adjust the same field for the parameter sweep.
            # This avoids spending Gaussian generation time again on every strength.
            for method in METHODS+['python']:
                reset(123)
                if method == 'python':
                    params = transform.get_parameters(image=tensor)
                    base = np.ascontiguousarray(params['elastic_offsets'].numpy(), dtype=np.float64)
                else:
                    # Low strength does not cap the field; later normalizations recover requested RMS.
                    base = lab.run(image, method, EDGE, float(HALO), float(scale), .01, BOUND, 123,
                                   normals['normals'] if method == 'normal_bspline' else EMPTY)['field']
                for strength in STRENGTHS:
                    field = lab.normalize(base, strength, BOUND)
                    if method == 'python':
                        params['elastic_offsets'] = torch.from_numpy(field.astype(np.float32))
                        result = transform.apply({'image': tensor}, **params)['image'][0].numpy()
                        if strength == 4.:
                            # Matched coordinates verify the CPU resampler against actual Python grid_sample.
                            delta = np.abs(result-lab.warp(image, field, float(HALO)))
                            report.setdefault('python_warp_validation', []).append(dict(chunk=name, scale=scale,
                                max_abs=float(delta.max()), mean_abs=float(delta.mean())))
                            assert delta.max() < 1e-4
                    else:
                        result = lab.warp(image, field, float(HALO))
                    m = metrics(field, strength)
                    assert m['lipschitz_bound'] < BOUND+1e-12 and m['outside_fraction'] == 0
                    report['cases'].append(dict(chunk=name, method=method, scale=scale, strength=strength, **m,
                        image_sha256=hashlib.sha256(result).hexdigest()))
                    # Only retain full fields for the diagnostics page, never hundreds of dense volumes.
                    record = dict(image=result.copy(), metrics=m)
                    if scale == 16 and strength == 4.: record['field'] = field
                    visuals[(method, scale, strength)] = record
            if not args.visuals_only:
                (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        np.testing.assert_array_equal(image, image_before)
        if not args.no_plots: render(args.output, name, original, visuals)
        print(f'Finished {name}', flush=True)
    if previous:
        assert previous['inputs'] == report['inputs'], 'CT inputs changed since timing run'
        assert previous['extension_sha256'] == report['extension_sha256'], 'Native build changed since timing run'
        previous.update({key: report[key] for key in ('strengths', 'cases', 'python_warp_validation')})
        report = previous
    # Compare the same timed stages, excluding diagnostic return packaging.
    # Keep external wall time separately (older runs also materialized a Python diagnostic field).
    for row in report['timings']:
        row['compute'] = stats(np.array(row['field']['samples_ms'])+row['warp']['samples_ms'])
    report['timing_definition'] = 'compute = field generation/normalization + image resampling; total = external diagnostic-call wall time'
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
