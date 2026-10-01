"""Offline CT normalization review; reads local chunks, never changes the loader.

Run with the project Python. Writes a self-contained slice viewer, PNG contact
sheets, numerical robustness checks and warm CPU benchmarks into --out.
"""
import argparse
import base64
import json
import math
from pathlib import Path
import platform
import time
from urllib.parse import urlsplit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numba
import numpy as np
from vesuvius.neural_tracing.fiber_follow.shared.ct_normalization import chunk_paths


METHODS = ('raw', 'zscore', 'percentile', 'mad', 'huber', 'asinh', 'bounded_mad', 'volume_mad')
LABELS = ('Raw / 255', 'Mean / std', 'Percentile-clipped z', 'Median / MAD',
          'Huber z', 'MAD + soft tails', 'Bounded sampled MAD', 'Volume median / MAD')
FLOOR = 2.  # Native uint8 intensity units, explicitly reported in every artifact.
LIMIT = 4.
VALUES = np.arange(256, dtype=np.float64)


def window_histogram(histogram, zero_below):
    """Match zeroing uint8 values strictly below the threshold, including fill."""
    h = histogram.copy()
    if zero_below:
        removed = h[:zero_below].sum()
        h[:zero_below] = 0
        h[0] = removed
    return h


def quantile(histogram, q):
    return float(np.searchsorted(np.cumsum(histogram), q*histogram.sum(), side='left'))


def statistics(histogram, method, masked=False):
    """Robust estimates on 256 bins; no large-array sorting or percentile pass."""
    h = np.asarray(histogram, dtype=np.float64).copy()
    if masked:
        h[0] = 0  # Exact fill zeros only, never treat nonzero background as tissue.
    count = float(h.sum())
    if count == 0:
        return dict(center=0., scale=FLOOR, lower=0., upper=255., empty=True)
    lower, upper = 0., 255.
    if method == 'zscore':
        center = float(h @ VALUES/count)
        scale = math.sqrt(float(h @ ((VALUES-center)**2)/count))
    elif method == 'percentile':
        lower, upper = quantile(h, .005), quantile(h, .995)
        clipped = np.clip(VALUES, lower, upper)
        center = float(h @ clipped/count)
        scale = math.sqrt(float(h @ ((clipped-center)**2)/count))
    else:
        center = quantile(h, .5)
        deviations = np.bincount(np.abs(VALUES-center).astype(int), weights=h, minlength=256)
        scale = 1.4826*quantile(deviations, .5)
        scale = max(scale, FLOOR)
        if method == 'huber':
            # Huber location plus bounded squared residuals (normal-consistent).
            cutoff = 2.5
            tail = .5*math.erfc(cutoff/math.sqrt(2))
            phi = math.exp(-cutoff**2/2)/math.sqrt(2*math.pi)
            correction = 1-2*(cutoff*phi+tail)+2*cutoff**2*tail
            for _ in range(4):
                residual = VALUES-center
                weights = np.minimum(1., cutoff*scale/np.maximum(np.abs(residual), 1e-12))
                center += float((h*weights) @ residual/(h @ weights))
                scale = max(FLOOR, math.sqrt(float(h @ np.minimum((VALUES-center)**2,
                    (cutoff*scale)**2)/count/correction)))
    return dict(center=float(center), scale=float(max(scale, FLOOR)), lower=lower,
                upper=upper, empty=False)


def transform(raw, stats, method, masked):
    x = raw.astype(np.float32)
    if method == 'raw':
        return x/255.
    if method == 'percentile':
        x = np.clip(x, stats['lower'], stats['upper'])
    z = (x-stats['center'])/stats['scale']
    if method == 'asinh':
        z = 2*np.arcsinh(z/2)  # Unit slope near zero, softer compression than hard clipping.
    np.clip(z, -LIMIT, LIMIT, out=z)
    if masked:
        z[raw == 0] = -LIMIT
    return z


def estimate(hist, method, masked, volume_stats):
    if method == 'volume_mad':
        return dict(volume_stats)
    stats = statistics(hist, method, masked)
    if method == 'bounded_mad':
        stats['scale'] = max(stats['scale'], .5*volume_stats['scale'])
    return stats


@numba.njit(cache=True)
def histogram_float(image, step):
    """Quantize only statistics to uint8 bins; the output remains continuous."""
    hist = np.zeros(256, np.int64)
    for z in range(0, image.shape[0], step):
        for y in range(0, image.shape[1], step):
            for x in range(0, image.shape[2], step):
                v = image[z, y, x]
                if np.isfinite(v):
                    hist[max(0, min(255, int(v*255+.5)))] += 1
    return hist


@numba.njit(cache=True)
def apply_float(image, output, center, scale, lower, upper, masked, soft):
    inverse = 1./scale
    for i in range(image.size):
        value = image[i]
        if not np.isfinite(value) or (masked and value == 0):
            output[i] = -LIMIT
        else:
            raw = min(upper, max(lower, value*255.))
            z = (raw-center)*inverse
            if soft:
                z = 2*math.asinh(z/2)
            output[i] = min(LIMIT, max(-LIMIT, z))


def normalize_float(image, method, masked, volume_stats, output):
    if method == 'raw':
        np.copyto(output, image)
        return
    if method == 'volume_mad':
        stats = volume_stats
    else:
        hist = histogram_float(image, 4 if method in ('sampled_mad', 'bounded_mad') else 1)
        stats = estimate(hist, method, masked, volume_stats)
    apply_float(image.reshape(-1), output.reshape(-1), stats['center'], stats['scale'], stats['lower'], stats['upper'],
                masked, method == 'asinh')


def source_candidates(source, cache, seed, count):
    uri = urlsplit(source['ct'])
    root = ((Path(cache)/uri.scheme/uri.netloc/uri.path.lstrip('/')) if uri.scheme else Path(source['ct']))/str(source['ct_level'])
    meta = json.loads((root/'.zarray').read_text())
    if meta['dtype'] != '|u1' or meta['compressor'] is not None:
        raise ValueError('Review expects decoded uint8 chunks in the existing local cache')
    paths = chunk_paths(root, meta.get('dimension_separator', '.'))
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(paths), min(count, len(paths)), replace=False)
    masked = '-masked.zarr' in source['ct']
    rows, calibration = [], np.zeros(256, np.int64)
    for index in selected:
        path = paths[index]
        if path.stat().st_size != math.prod(meta['chunks']):
            continue  # Incomplete in-flight cache writes are not samples.
        raw = np.memmap(path, dtype=np.uint8, mode='r', shape=tuple(meta['chunks']))
        h = np.bincount(raw[::4, ::4, ::4].ravel(), minlength=256)
        calibration += h
        if h[1:].sum() < 128:
            continue
        valid_h = h.copy()
        if masked:
            valid_h[0] = 0
        stats = statistics(h, 'mad', masked)
        rows.append(dict(path=str(path), shape=meta['chunks'], zero_fraction=float(h[0]/h.sum()),
            center=stats['center'], scale=stats['scale'],
            tail=(quantile(valid_h, .999)-stats['center'])/stats['scale']))
    if len(rows) < 3:
        raise ValueError(f'Insufficient populated chunks in {root}')
    dense = [r for r in rows if r['zero_fraction'] < .05 and r['scale'] > FLOOR] or rows
    typical = sorted(dense, key=lambda r: r['scale'])[len(dense)//2]
    tail = max((r for r in rows if r != typical), key=lambda r: r['tail'])
    boundary = [r for r in rows if .05 < r['zero_fraction'] < .9 and r not in (typical, tail)]
    if boundary:
        third, label = min(boundary, key=lambda r: abs(r['zero_fraction']-.5)), 'mask boundary'
    else:
        third, label = min((r for r in dense if r not in (typical, tail)), key=lambda r: r['scale']), 'low contrast'
    picked = [(typical, 'typical'), (tail, 'bright tail'), (third, label)]
    # Keep exploratory volume calibration independent of the displayed chunks.
    for row, _ in picked:
        raw = np.memmap(row['path'], dtype=np.uint8, mode='r', shape=tuple(row['shape']))
        calibration -= np.bincount(raw[::4, ::4, ::4].ravel(), minlength=256)
    return picked, statistics(calibration, 'mad', masked), dict(root=str(root), masked=masked,
        available_chunks=len(paths), screened_chunks=len(selected), eligible_chunks=len(rows),
        calibration_samples=int(calibration.sum()), calibration_histogram=calibration.tolist())


def timing(image, method, masked, volume_stats, repeats):
    output = np.empty_like(image)
    for _ in range(3):
        normalize_float(image, method, masked, volume_stats, output)
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        normalize_float(image, method, masked, volume_stats, output)
        times.append((time.perf_counter_ns()-start)/1e6)
    return dict(mean_ms=float(np.mean(times)), median_ms=float(np.median(times)),
                p95_ms=float(np.percentile(times, 95)))


def save_sheet(cases, path):
    fig, axes = plt.subplots(len(cases), len(METHODS), figsize=(24, 3.6*len(cases)), squeeze=False)
    for row, case in enumerate(cases):
        raw = case['_raw'][case['_raw'].shape[0]//2]
        for column, (method, label) in enumerate(zip(METHODS, LABELS)):
            stats = case['stats'][method]
            values = transform(raw, stats, method, case['masked'])
            ax = axes[row, column]
            ax.imshow(values, cmap='gray', vmin=0 if method == 'raw' else -3,
                      vmax=1 if method == 'raw' else 3, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(label if row == 0 else '', fontsize=11)
            if column == 0:
                ax.set_ylabel(case['label'], fontsize=11)
            if method != 'raw':
                ax.set_xlabel(f"center {stats['center']:.1f} / scale {stats['scale']:.1f}", fontsize=9)
    window = f" | values < {cases[0]['zero_below']} set to zero" if cases[0]['zero_below'] else ''
    fig.suptitle(cases[0]['source']+window+' | normalized display [-3, 3]; raw [0, 1]', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_viewer(cases, report, path):
    payload = []
    for case in cases:
        value = {k: v for k, v in case.items() if k != '_raw'}
        value['voxels'] = base64.b64encode(case['_raw'].tobytes()).decode()
        payload.append(value)
    page = '''<!doctype html><meta charset="utf-8"><title>CT normalization review</title>
<style>body{font:16px system-ui;background:#171a20;color:#e6e9ef;margin:24px}h1{font-size:25px}
label{margin-right:20px}select,input,button{font:inherit}#panels{display:grid;grid-template-columns:repeat(4,minmax(230px,1fr));gap:16px}
canvas{width:100%;image-rendering:pixelated;background:black}article{background:#252a33;padding:12px;border-radius:8px}
small{color:#c4cbd5}p{max-width:1100px}table{border-collapse:collapse}td,th{padding:6px 16px;text-align:left;border-bottom:1px solid #555}
@media(max-width:900px){#panels{grid-template-columns:repeat(2,1fr)}}</style>
<h1>CT normalization: real cached chunks</h1>
<p>__WINDOW_NOTE__</p>
<p>All methods show the same voxels. No per-image display autoscaling. Raw uses [0,1];
normalized methods share the window below. Statistics use the whole 3D chunk, except bounded sampled MAD
(every fourth sample on each axis) and volume MAD (a separate pooled calibration sample).</p>
<p>Remaining nonzero background noise is retained in the statistics. For windowed data and volumes explicitly named “masked”,
exact zeros are excluded from estimation and mapped to −4 (black). This is an assumption,
not tissue segmentation. Scale floor: 2 uint8 levels; normalized outputs bounded at ±4.
Bounded sampled MAD additionally floors the local scale at half the volume MAD scale to limit noise amplification.</p>
<p><label>Chunk <select id="case"></select></label><label>View <select id="axis"><option value="0">Z slice</option><option value="1">Y slice</option><option value="2">X slice</option></select></label></p>
<p><label>Slice <input id="slice" type="range" min="0" max="127" value="64"><span id="sliceLabel"></span></label>
<label>Display half-window ±<input id="window" type="range" min="0.5" max="4" step="0.1" value="3"><span id="windowLabel"></span></label></p>
<p id="meta"></p><div id="panels"></div><h2>Warm CPU cost per production-sized crop</h2>
<p>120×104×104 float32 input in [0,1], one thread. Includes statistics + application, excludes I/O,
interpolation, thresholding, allocation, and JIT compilation. Raw is a copy baseline. Volume MAD excludes its one-time calibration.</p>
<div id="timings"></div><p>Review only: training loader and volume cache are unchanged.</p>
<script>const cases=__CASES__, methods=__METHODS__, labels=__LABELS__, report=__REPORT__;
const byId=id=>document.getElementById(id); let current=-1, data;
cases.forEach((c,i)=>byId('case').add(new Option(c.source+' — '+c.label,i)));
byId('panels').innerHTML=methods.map((m,i)=>'<article><strong>'+labels[i]+'</strong><canvas id="canvas'+i+'" width="128" height="128"></canvas><small id="stats'+i+'"></small></article>').join('');
function redraw(){let n=+byId('case').value,c=cases[n];if(n!==current){data=Uint8Array.from(atob(c.voxels),x=>x.charCodeAt(0));current=n;}
let axis=+byId('axis').value,k=+byId('slice').value,w=+byId('window').value;byId('sliceLabel').textContent=k;byId('windowLabel').textContent=w;
byId('meta').textContent='Chunk '+c.chunk_key+' | zero fraction '+(100*c.zero_fraction).toFixed(2)+'% | '+c.path;
methods.forEach((method,i)=>{let ctx=byId('canvas'+i).getContext('2d'),image=ctx.createImageData(128,128),s=c.stats[method];
for(let y=0;y<128;y++)for(let x=0;x<128;x++){let index=axis===0?(k*128+y)*128+x:axis===1?(y*128+k)*128+x:(y*128+x)*128+k;
let raw=data[index],v=raw/255;if(method!=='raw'){let a=method==='percentile'?Math.min(s.upper,Math.max(s.lower,raw)):raw;
v=(a-s.center)/s.scale;if(method==='asinh')v=2*Math.asinh(v/2);v=Math.min(4,Math.max(-4,v));if(c.masked&&raw===0)v=-4;v=(v+w)/(2*w);}
let p=4*(y*128+x),g=Math.round(255*Math.min(1,Math.max(0,v)));image.data[p]=image.data[p+1]=image.data[p+2]=g;image.data[p+3]=255;}
ctx.putImageData(image,0,0);byId('stats'+i).textContent=method==='raw'?'Native uint8 / 255':'center '+s.center.toFixed(2)+'; scale '+s.scale.toFixed(2)+' native levels';});}
['case','axis','slice','window'].forEach(id=>byId(id).addEventListener('input',redraw));redraw();
byId('timings').innerHTML='<table><tr><th>Method</th><th>Median ms</th><th>p95 ms</th></tr>'+methods.map((m,i)=>'<tr><td>'+labels[i]+'</td><td>'+report.timings[m].median_ms.toFixed(3)+'</td><td>'+report.timings[m].p95_ms.toFixed(3)+'</td></tr>').join('')+'</table>';
</script>'''
    for key, value in [('__CASES__', payload), ('__METHODS__', METHODS), ('__LABELS__', LABELS), ('__REPORT__', report)]:
        page = page.replace(key, json.dumps(value))
    page = page.replace('__WINDOW_NOTE__', report['window_note'])
    path.write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--config', type=Path, default=root/'configs/mixed_ct_datasets.json')
    parser.add_argument('--out', type=Path, default=root/'output/ct_normalization_review')
    parser.add_argument('--screen-chunks', type=int, default=192)
    parser.add_argument('--repeats', type=int, default=40)
    parser.add_argument('--seed', type=int, default=7349)
    parser.add_argument('--zero-below', type=int, default=0,
                        help='Set native uint8 values strictly below this to zero; exclude zeros from statistics.')
    args = parser.parse_args()
    if not 0 <= args.zero_below <= 255:
        parser.error('--zero-below must be in [0, 255]')
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    cases, sources = [], {}
    for index, source in enumerate(config['sources']):
        picked, calibration, provenance = source_candidates(source, config['cache_dir'], args.seed+index, args.screen_chunks)
        # Select on the original data so both reviews contain identical chunks.
        if args.zero_below:
            provenance['masked'] = True
            h = window_histogram(np.asarray(provenance['calibration_histogram']), args.zero_below)
            provenance['calibration_histogram'] = h.tolist()
            calibration = statistics(h, 'mad', masked=True)
        sources[source['name']] = dict(provenance, volume_stats=calibration)
        print(source['name'], 'screened', provenance['screened_chunks'], 'chunks', flush=True)
        group = []
        for row, label in picked:
            raw = np.array(np.memmap(row['path'], dtype=np.uint8, mode='r', shape=tuple(row['shape'])))
            if raw.shape != (128, 128, 128):
                raise ValueError('Viewer expects the configured 128-cubed chunks')
            raw[raw < args.zero_below] = 0
            h = np.bincount(raw.ravel(), minlength=256)
            sample_h = np.bincount(raw[::4, ::4, ::4].ravel(), minlength=256)
            stats = {method: estimate(sample_h if method == 'bounded_mad' else h, method,
                                      provenance['masked'], calibration) for method in METHODS}
            case = dict(row, label=label, source=source['name'], masked=provenance['masked'], stats=stats,
                        chunk_key=str(Path(row['path']).relative_to(provenance['root'])), _raw=raw,
                        zero_below=args.zero_below)
            case['zero_fraction'] = float(h[0]/h.sum())
            group.append(case); cases.append(case)
        save_sheet(group, args.out/(source['name']+'.png'))
    # Benchmark actual-sized input with interpolation-like fractional values.
    raw = np.memmap(cases[0]['path'], dtype=np.uint8, mode='r', shape=(128, 128, 128))[:120, :104, :104].astype(np.float32)
    image = np.ascontiguousarray((.75*raw+.25*np.roll(raw, 1, axis=2))/255.)
    image[image < args.zero_below/255.] = 0
    timings = {method: timing(image, method, cases[0]['masked'], cases[0]['stats']['volume_mad'], args.repeats)
               for method in METHODS}
    # Same sample, controlled contamination. No subjective image ranking implied.
    robustness = []
    rng = np.random.default_rng(args.seed)
    for case in cases:
        for fraction in (0., .001, .01, .05):
            raw = case['_raw'].copy()
            eligible = np.flatnonzero(raw.ravel() > 0) if case['masked'] else np.arange(raw.size)
            if fraction:
                raw.ravel()[rng.choice(eligible, int(fraction*len(eligible)), replace=False)] = 255
            h = np.bincount(raw.ravel(), minlength=256)
            sample_h = np.bincount(raw[::4, ::4, ::4].ravel(), minlength=256)
            for method in ('zscore', 'percentile', 'mad', 'huber', 'bounded_mad'):
                stats = estimate(sample_h if method == 'bounded_mad' else h, method, case['masked'], case['stats']['volume_mad'])
                original = case['stats'][method]
                robustness.append(dict(source=case['source'], case=case['label'], spike_fraction=fraction,
                    method=method, center_shift=stats['center']-original['center'], scale_ratio=stats['scale']/original['scale']))
    window_note = (f'Values below {args.zero_below} are set to zero before normalization; values at or above '
                   f'{args.zero_below} are unchanged. Zeros are excluded from statistics and displayed black. '
                   'Chunk selection matches the unwindowed review.' if args.zero_below else 'No intensity threshold applied.')
    report = dict(seed=args.seed, zero_below=args.zero_below, window_note=window_note,
        sources=sources, methods=dict(zip(METHODS, LABELS)),
        scale_floor_uint8=FLOOR, output_limit=LIMIT, timings=timings, robustness=robustness,
        benchmark=dict(shape=list(image.shape), repeats=args.repeats, warmup=3, threads=1,
                       source=cases[0]['source'], chunk_key=cases[0]['chunk_key'], excludes_thresholding=True,
                       cpu=platform.processor(), machine=platform.machine(), numba=numba.__version__, numpy=np.__version__),
        notes=['Only local available chunks were sampled; remote cache coverage is not a uniform volume sample.',
               'Remaining nonzero noise is included; zeros in windowed or explicitly masked volumes are excluded.',
               'Volume MAD is an exploratory calibration, not production training statistics.',
               'Statistics for float crops are quantized to 256 bins; output values remain continuous.',
               'All normalized methods use a common display window; raw uses 0..1.',
               'Normalization runs after interpolation in the benchmark, never separately on storage chunks.'],
        cases=[{k: v for k, v in c.items() if k != '_raw'} for c in cases])
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    save_viewer(cases, report, args.out/'index.html')
    print(json.dumps(timings, indent=2), flush=True)
    print('Viewer:', args.out/'index.html', flush=True)


if __name__ == '__main__':
    main()
