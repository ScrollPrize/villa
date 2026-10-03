"""Paired Lasagna-only tensor scale sweep on Paris 4 and AFV 1447.

Uses the 32 saved spatial regions per volume from compare_sheet_normals.py, with
seven locations per region. No crossing-derived normals enter the comparison.
First 16 regions select a setting; last 16 report a held-out comparison.
python evaluation/sweep_lasagna_tensor.py --fetch
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import html
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from vesuvius.neural_tracing.fiber_follow.data.data import _grid_flat
from vesuvius.neural_tracing.fiber_follow.data.datasets import read_dataset_config
from vesuvius.neural_tracing.fiber_follow.data.volume import ChunkedArray, RemoteChunkedArray
from vesuvius.neural_tracing.fiber_follow.evaluation.compare_sheet_normals import BUCKET, MANIFESTS, angle, normal_at, stats
from vesuvius.neural_tracing.fiber_follow.shared.fast_sample import sample_scalar_crop
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.tracing.heading import ct_structure_tensor


@dataclass(frozen=True)
class Variant:
    spacing: float
    derivative: float
    integration: float
    radius: float
    native: bool = False

    @property
    def key(self):
        return ('native' if self.native else f's{self.spacing:g}')+f'_d{self.derivative:g}_i{self.integration:g}_r{self.radius:g}'


def variants():
    out = [Variant(s, d, i, 4*i+3*d) for s in (2.5, 1.25, 1.) for d in (1., 2.) for i in (2., 4., 6., 8.)]
    out += [Variant(2.5, 1., 4., 32.), Variant(1., 1., 4., 32., native=True)]
    return out


BASELINE = 's2.5_d1_i4_r19'


def estimate(block, origin, pos, frame, variant):
    """Same interpolation, grid phase and tensor as shared training labels; explicit CT-unit sigmas."""
    began = time.perf_counter()
    if variant.native:
        lo = np.floor(pos[::-1]-origin).astype(int)-32
        raw = block[tuple(slice(a, a+65) for a in lo)]
        center = pos[::-1]-origin-lo
        rotation = np.eye(3)
    else:
        r = int(np.ceil(variant.radius/variant.spacing))
        crop = CropSpec(depth=2*r+1, width=2*r+2, behind=r, spacing=variant.spacing)
        grid = _grid_flat(crop)
        # Fail rather than silently sample outside the cached CT context.
        corners = np.array(np.meshgrid(*[(v.min(), v.max()) for v in grid.T], indexing='ij')).reshape(3, -1).T
        world = ((corners @ frame.T+pos)[:, ::-1]-origin)
        if np.any(world < 0) or np.any(world >= np.array(block.shape)-1):
            raise ValueError('Insufficient CT context for oriented stencil')
        raw = np.empty((crop.depth, crop.width, crop.width), np.float32)
        sample_scalar_crop(block[None], origin, pos, frame, grid, raw)
        center = [r, r+.5, r+.5]
        rotation = frame
    sampled = time.perf_counter()
    tensor = ct_structure_tensor(raw, center, sample_spacing=variant.spacing,
                                 derivative_sigma=variant.derivative, integration_sigma=variant.integration)
    values, vectors = np.linalg.eigh(tensor)
    gap = float((values[-1]-values[-2])/values[-1]) if values[-1] > 1e-12 else 0.
    n = rotation @ vectors[:, -1] if values[-1] > 1e-12 else np.full(3, np.nan)
    return n, gap, (sampled-began)*1000, (time.perf_counter()-sampled)*1000


def prepare(source, old, dest, fetch):
    """Cache 161-cubes and independently decoded Lasagna references; no training files changed."""
    target = dest/source['name']; target.mkdir(exist_ok=True)
    manifest = json.loads((old/source['name']/'lasagna.json').read_text())
    url = BUCKET+MANIFESTS[source['name']]
    arrays = {}
    for channel in ('nx', 'ny', 'grad_mag'):
        root, level = manifest['groups'][channel]['zarr'].rsplit('/', 1)
        arrays[channel] = RemoteChunkedArray(url.rsplit('/', 1)[0]+'/'+root, int(level), str(old/'cache'),
                                             64 << 20, cache_only=not fetch)
    ct = (ChunkedArray(Path(source['ct'])/'0', 128 << 20) if source['kind'] == 'paris4' else
          RemoteChunkedArray(source['ct'], source['ct_level'], str(old/'cache'), 128 << 20, cache_only=not fetch))
    frames = np.load(old.parent/'heading_patch_normals_20261002'/(source['name']+'.npz'))['frames']
    offsets = np.r_[np.zeros((1, 3)), np.eye(3)*8, -np.eye(3)*8]
    files = sorted((old/source['name']).glob('chunk_*.npz'))

    def one(pair):
        i, path = pair
        output = target/path.name
        if output.exists():
            return
        with np.load(path) as data:
            row = json.loads(str(data['row_json']))
        pos = np.array(row['position_ct_xyz'])
        origin = np.floor(pos[::-1]).astype(int)-80
        block = ct.read(origin, [161]*3)
        positions = pos+offsets
        reference = [normal_at(arrays, manifest, source['ct_grid_scale'], p) for p in positions]
        np.savez_compressed(output, ct=block, origin=origin, positions=positions,
                            frames=np.stack([frames[(i*7+j) % len(frames)] for j in range(7)]),
                            normals=np.stack([r[0] for r in reference]), support=[r[1] for r in reference],
                            coherence=[r[2] for r in reference])
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, enumerate(files)))
    print('Prepared', source['name'], len(files), 'regions', flush=True)


def paired(error, base, mask):
    valid = mask & np.isfinite(error) & np.isfinite(base)
    delta = np.where(valid, error-base, np.nan)
    regions = np.array([np.mean(row[np.isfinite(row)]) for row in delta if np.isfinite(row).any()])
    if not len(regions):
        return dict(n=0)
    rng = np.random.default_rng(1937)
    bootstrap = rng.choice(regions, (3000, len(regions)), replace=True).mean(axis=1)
    return dict(n=int(valid.sum()), mean_change=float(np.mean(delta[valid])),
                improved_fraction=float(np.mean(delta[valid] < 0)),
                region_bootstrap_mean_change_ci95=np.quantile(bootstrap, [.025, .975]).tolist())


def summarize(dest, settings, sources):
    result = {}
    all_errors = {}
    for source in sources:
        name = source['name']
        data = np.load(dest/name/'results.npz')
        errors = data['errors']  # variants x regions x seven points
        all_errors[name] = errors
        baseline = errors[[v.key for v in settings].index(BASELINE)]
        row = {}
        for i, variant in enumerate(settings):
            e = errors[i]
            row[variant.key] = dict(parameters=asdict(variant), all=stats(e.ravel()),
                tuning=stats(e[:16].ravel()), held_out=stats(e[16:].ravel()),
                paired=paired(e, baseline, np.ones_like(e, bool)),
                paired_held_out=paired(e, baseline, np.indices(e.shape)[0] >= 16),
                reliable_fraction=float(np.mean(data['gaps'][i] >= .05)),
                sampling_ms_p50_p95=np.quantile(data['timings'][i, :, :, 0], [.5, .95]).tolist(),
                tensor_ms_p50_p95=np.quantile(data['timings'][i, :, :, 1], [.5, .95]).tolist())
        result[name] = row
        keys = list(data['variant_keys'])
        context_delta = angle(data['normals'][keys.index(BASELINE)], data['normals'][keys.index('s2.5_d1_i4_r32')])
        row['s2.5_d1_i4_r32']['change_in_normal_from_extra_context'] = stats(context_delta.ravel())
    # Equal weight per volume; choose solely on first 16 regions from each volume.
    winner = min(settings, key=lambda v: np.mean([result[s['name']][v.key]['tuning']['mean'] for s in sources])).key
    report = dict(baseline=BASELINE, selected_on_tuning=winner, sources=result)
    (dest/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    for name, rows in result.items():
        print(name, 'best tuning:', min(rows, key=lambda k: rows[k]['tuning']['mean']), flush=True)
        for key in (BASELINE, 's2.5_d1_i4_r32', 's2.5_d1_i6_r27', 's2.5_d1_i8_r35', 's1_d1_i4_r19', winner):
            print(key, json.dumps(rows[key]['all']), 'held-out', rows[key]['held_out'], flush=True)
    make_plot(dest, report)
    write_report(dest, report)
    return report


def make_plot(dest, report):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fiber_follow_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for column, (name, rows) in enumerate(report['sources'].items()):
        for r, d in enumerate((1, 2)):
            ax = axes[r, column]
            for spacing in (2.5, 1.25, 1):
                keys = [Variant(spacing, d, i, 4*i+3*d).key for i in (2, 4, 6, 8)]
                values = [rows[k]['held_out'] for k in keys]
                line, = ax.plot((2, 4, 6, 8), [v['median'] for v in values], marker='o', label=f'{spacing:g} CT voxels/sample')
                ax.plot((2, 4, 6, 8), [v['p90'] for v in values], ls=':', color=line.get_color(), alpha=.7)
            ax.set(title=f'{name}, derivative sigma {d}', xlabel='Integration sigma (selected-CT voxels)',
                   ylabel='Unsigned difference from Lasagna (degrees)', xticks=[2, 4, 6, 8])
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Held-out regions: solid = median, dotted = p90\n16 regions x 7 positions per volume; same positions and orientations for every setting')
    fig.tight_layout()
    fig.savefig(dest/'comparison.png', dpi=160)
    plt.close(fig)


def write_report(dest, report):
    names = list(report['sources'])
    selected = report['selected_on_tuning']
    def metric(value):
        return f"{value['median']:.2f} / {value['p90']:.2f}°" if value['n'] else 'unavailable'
    rows = []
    for key, first in report['sources'][names[0]].items():
        p = first['parameters']
        cells = [html.escape(key), str(p['spacing']), str(p['derivative']), str(p['integration']), str(p['radius'])]
        for name in names:
            row = report['sources'][name][key]
            cells += [metric(row['held_out']), f"{sum(row[k][0] for k in ('sampling_ms_p50_p95', 'tensor_ms_p50_p95')):.2f}"]
        style = ' class="selected"' if key == selected else (' class="baseline"' if key == BASELINE else '')
        rows.append(f'<tr{style}>'+''.join(f'<td>{v}</td>' for v in cells)+'</tr>')
    changes = []
    for name in names:
        row = report['sources'][name][selected]
        change = row['paired_held_out']
        lo, hi = change['region_bootstrap_mean_change_ci95']
        baseline = report['sources'][name][BASELINE]
        context = report['sources'][name]['s2.5_d1_i4_r32']['change_in_normal_from_extra_context']
        changes.append(f'<li><b>{html.escape(name)}</b>: baseline {metric(baseline["held_out"])} → '
            f'{metric(row["held_out"])} (median / p90, held-out). Mean paired error change '
            f'{change["mean_change"]:.2f}°, region-bootstrap 95% interval [{lo:.2f}, {hi:.2f}]°. '
            f'{change["improved_fraction"]:.0%} of held-out points improve. Enlarging only context '
            f'changes the normal by median {context["median"]:.4f}°, p90 {context["p90"]:.4f}°.</li>')
    image = base64.b64encode((dest/'comparison.png').read_bytes()).decode()
    page = f'''<!doctype html><html><meta charset="utf-8"><title>Lasagna versus tensor scale sweep</title>
<style>body{{font:16px/1.5 system-ui;max-width:1450px;margin:32px auto;padding:0 24px;color:#182332}}
table{{border-collapse:collapse;font-size:14px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}
.selected{{background:#dcf4e7}}.baseline{{background:#fff2ca}}img{{max-width:100%}}code{{background:#eee;padding:2px 5px}}</style>
<h1>Lasagna versus structure-tensor normals</h1>
<p>Paris 4 and AFV 1447 only. 32 spatially separated regions per volume, seven nearby points per region;
448 paired locations total. Every setting uses identical locations and crop orientations. No crossing-derived
normal is used. The first 16 regions per volume select the setting; the last 16 are held out.</p>
<p><b>Selected on tuning regions:</b> <code>{selected}</code>. Current training labels:
<code>{BASELINE}</code>. All distances and Gaussian sigmas below are in voxels of the selected CT volume.</p>
<ul>{''.join(changes)}</ul>
<img src="data:image/png;base64,{image}">
<h2>All settings: held-out comparison</h2>
<p>Errors are unsigned angular differences, median / p90; lower means closer to Lasagna. Green is the
tuning-selected setting, yellow the current training-label setting. Approximate CPU milliseconds combine
the median stencil-sampling and tensor times. These exclude CT I/O, network, heading input preparation and GPU
training, so they are not estimates of full training speed.</p>
<table><tr><th>Setting</th><th>Sample spacing</th><th>Derivative sigma</th><th>Integration sigma</th><th>Context radius</th>
{''.join(f'<th>{html.escape(n)} error</th><th>CPU ms</th>' for n in names)}</tr>{''.join(rows)}</table>
<h2>Protocol and limits</h2><ul>
<li>Native CT is trilinearly sampled using the trainer's fused sampler. The oriented stencil uses the same
grid phase as training: the head is on a depth sample and halfway between samples on both lateral axes.
Recorded randomized heading-model frames supply the orientations. The native65 control instead uses raw
world-aligned CT. Context radius is four integration sigmas plus three derivative sigmas, rounded outward.</li>
<li>The context-only control increases radius from 19 to 32 CT voxels, keeping resolution and smoothing unchanged.
The tensor uses the same shared helper as training, with explicit experimental sigmas; defaults remain unchanged.
There is no intensity threshold. Normal directions are the largest-eigenvalue axes, compared with acos(|dot|).</li>
<li>Lasagna normals are decoded with the repository decoder and interpolated through outer products to avoid sign seams.
All interpolated support must have grad_mag &gt; 0. Reference manifests are the saved public Paris 4 and PHerc1447
products; their normal-grid spacing is four selected-CT voxels. Local/public Paris CT alignment was verified
in the preceding experiment.</li>
<li>Region centers were originally selected by nearby H/V fiber geometry, without examining normal agreement.
Only their locations are reused here; crossing-derived normal estimates are excluded. Offsets are the center
and ±8 CT voxels along each world axis. Nearby points are correlated; confidence intervals resample regions,
not individual points. This exploratory dataset does not establish accuracy throughout either volume.</li>
<li>Lasagna is a model prediction, not ground truth. Better agreement can favor its smoothness. Broad smoothing
may lose local sheet curvature. No training configuration, checkpoint, or dataset was changed.</li>
</ul><h2>Reproduce</h2><pre>OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python evaluation/sweep_lasagna_tensor.py --fetch</pre>
<p>Omit --fetch once cached. Protocol, raw per-point errors/normals/timings, CT contexts, and summary.json are
stored beside this report. Each timing measures one warmed stencil per location on one CPU thread; aggregate
medians and p95 are in summary.json. Region-bootstrap seed: 1937.</p></html>'''
    (dest/'report.html').write_text(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old', type=Path, default=Path('output/sheet_normals_20261002'))
    ap.add_argument('--out', type=Path, default=Path('output/lasagna_tensor_sweep_20261002'))
    ap.add_argument('--fetch', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config, _ = read_dataset_config('configs/mixed_ct_datasets_paris50.json')
    sources = [s for s in config['sources'] if s['name'] in ('paris4', '1447_5mm_v1')]
    settings = variants()
    (args.out/'protocol.json').write_text(json.dumps(dict(command=sys.argv, platform=platform.platform(),
        settings=[dict(key=v.key, **asdict(v)) for v in settings], seed=1937,
        locations='32 previously geometry-selected regions per source; center and +/-8 CT voxels on each world axis',
        orientations='Recorded randomized heading-model frames, identical across settings; native65 control has world-aligned axes',
        selection='First 16 regions per volume select by equal-volume mean error; last 16 regions held out',
        reference='Public Lasagna; unsigned outer-product interpolation; all support grad_mag>0; no crossing normal comparison',
        timing='One CPU thread; each location/variant measured once after JIT warmup; excludes CT reads and GPU; does not predict trainer throughput',
        caveat='Agreement is not ground-truth accuracy; nearby points correlated; bootstrap resamples regions; exploratory dataset'), indent=2)+'\n')
    for source in sources:
        name = source['name']
        prepare(source, args.old, args.out, args.fetch)
        output = args.out/name/'results.npz'
        if output.exists():
            continue
        blocks = sorted((args.out/name).glob('chunk_*.npz'))
        errors = np.full((len(settings), len(blocks), 7), np.nan)
        gaps = np.zeros_like(errors)
        normals = np.zeros((*errors.shape, 3))
        timings = np.zeros((*errors.shape, 2))
        for b, path in enumerate(blocks):
            with np.load(path) as data:
                block, origin, positions, frames, references = (data[k] for k in ('ct', 'origin', 'positions', 'frames', 'normals'))
                if b == 0:
                    estimate(block, origin, positions[0], frames[0], settings[0])
                for j, (pos, frame, reference) in enumerate(zip(positions, frames, references)):
                    for i, variant in enumerate(settings):
                        n, gap, sample_ms, tensor_ms = estimate(block, origin, pos, frame, variant)
                        normals[i, b, j], gaps[i, b, j] = n, gap
                        errors[i, b, j], timings[i, b, j] = angle(n, reference), [sample_ms, tensor_ms]
            if (b+1) % 4 == 0:
                print(name, f'{b+1}/{len(blocks)} regions evaluated', flush=True)
        np.savez_compressed(output, errors=errors, gaps=gaps, normals=normals, timings=timings,
                            variant_keys=[v.key for v in settings])
    summarize(args.out, settings, sources)


if __name__ == '__main__':
    main()
