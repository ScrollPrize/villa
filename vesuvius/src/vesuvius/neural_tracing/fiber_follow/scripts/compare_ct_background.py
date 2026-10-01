"""Estimate candidate CT background from quiet 8-cubed regions; review material masks.

Run with the project Python and OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1. Reads local caches only. Does not change training.
"""
import argparse
import base64
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy import ndimage as ndi

from vesuvius.neural_tracing.fiber_follow.scripts.compare_ct_normalization import statistics
from vesuvius.neural_tracing.fiber_follow.shared.ct_normalization import chunk_paths, background_from_blocks


MASKS = ('cut30', 'pixel', 'spatial2', 'spatial1')
LABELS = ('Threshold 30', 'Background + 3 noise scales', 'Spatial: grow above +2 scales', 'Spatial: grow above +1 scale')


def calibrate(root, excluded, seed, count):
    meta = json.loads((root/'.zarray').read_text())
    paths = [p for p in chunk_paths(root, meta.get('dimension_separator', '.')) if str(p) not in excluded]
    rng = np.random.default_rng(seed)
    rows, used = [], []
    for index in rng.choice(len(paths), min(count, len(paths)), replace=False):
        path = paths[index]
        if path.stat().st_size != 128**3:
            continue
        raw = np.memmap(path, dtype=np.uint8, mode='r', shape=(128,)*3)
        used.append(str(path))
        for z, y, x in rng.integers(0, 121, (64, 3)):
            block = np.array(raw[z:z+8, y:y+8, x:x+8], dtype=np.float32).ravel()
            if np.count_nonzero(block) < .98*block.size:
                continue  # Avoid zero-fill edges masquerading as quiet background.
            center = np.median(block)
            rows.append((center, 1.4826*np.median(np.abs(block-center))))
    rows = np.asarray(rows)
    if len(rows) < 100:
        raise ValueError(f'Too few nonempty calibration blocks in {root}')
    result = dict(background_from_blocks(rows), calibration_chunks=used)
    return result, rows


@numba.njit(cache=True)
def coarse_mean(raw, cap):
    """Capped 2-cubed means: outliers cannot dominate mask evidence."""
    out = np.empty((raw.shape[0]//2, raw.shape[1]//2, raw.shape[2]//2), np.float32)
    for z in range(out.shape[0]):
        for y in range(out.shape[1]):
            for x in range(out.shape[2]):
                total = 0.
                for dz in range(2):
                    for dy in range(2):
                        for dx in range(2):
                            total += min(float(raw[2*z+dz, 2*y+dy, 2*x+dx]), cap)
                out[z, y, x] = total/8.
    return out


def spatial_mask(raw, background, grow):
    if any(s % 2 for s in raw.shape):
        raise ValueError('Spatial preview requires even crop dimensions')
    center, noise = background['center'], background['noise']
    coarse = coarse_mean(raw, center+8*noise)
    smooth = ndi.uniform_filter(coarse, size=3, mode='reflect')
    # Hysteresis retains weaker regions only when connected to strong evidence.
    strong = smooth > center+4*noise
    allowed = smooth > center+grow*noise
    keep = ndi.binary_propagation(strong, mask=allowed)
    return keep.repeat(2, 0).repeat(2, 1).repeat(2, 2) & (raw > 0)


def make_mask(raw, background, method):
    if method == 'cut30':
        return raw >= 30
    if method == 'pixel':
        return raw > background['center']+3*background['noise']
    return spatial_mask(raw, background, 2 if method == 'spatial2' else 1)


def material_stats(raw, mask, background):
    # Match fast sampled statistics; an empty mask produces an all-black preview.
    sampled, selected = raw[::4, ::4, ::4], mask[::4, ::4, ::4]
    hist = np.bincount(sampled[selected], minlength=256)
    if mask.any() and not selected.any():
        hist = np.bincount(raw[mask], minlength=256)
    stats = statistics(hist, 'mad')
    stats['scale'] = max(stats['scale'], 2*background['noise'])
    return stats


def display(raw, mask, stats):
    # Same [-3, 3] foreground display for every method; background explicitly black.
    return np.where(mask, np.clip(((raw.astype(np.float32)-stats['center'])/stats['scale']+3)/6, 0, 1), 0)


def overlay(raw, mask):
    gray = np.repeat((raw.astype(np.float32)/255)[..., None], 3, axis=-1)
    gray[mask] = .55*gray[mask]+.45*np.array([0., 1., .3])
    return gray


def sheet(cases, path):
    fig, axes = plt.subplots(len(cases), 6, figsize=(20, 3.5*len(cases)))
    for row, case in enumerate(cases):
        raw, masks = case['_raw'], case['_masks']
        # Use identical central slices to earlier reviews unless completely empty.
        z = 64 if np.any(raw[64]) else int(np.argmax(np.count_nonzero(raw, axis=(1, 2))))
        views = [raw[z]/255]+[display(raw[z], masks[m][z], case['stats'][m]) for m in MASKS]
        views += [overlay(raw[z], masks['spatial1'][z])]
        for col, view in enumerate(views):
            axes[row, col].imshow(view, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(('Raw', *LABELS, 'Kept region in green')[col], fontsize=10)
            if col == 0:
                axes[row, col].set_ylabel(case['label']+f' / z={z}')
            elif col <= 4:
                axes[row, col].set_xlabel(f"Retained {100*case['fractions'][MASKS[col-1]]:.1f}% of nonzero voxels")
    bg = cases[0]['background']
    fig.suptitle(f"{cases[0]['source']} | candidate background {bg['center']:.1f}, noise scale {bg['noise']:.1f} | foreground median/MAD; black outside mask")
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(path, dpi=130); plt.close(fig)


def viewer(cases, report, path):
    payload = []
    for case in cases:
        c = {k: v for k, v in case.items() if not k.startswith('_')}
        c['voxels'] = base64.b64encode(case['_raw'].tobytes()).decode()
        c['masks'] = {m: base64.b64encode(np.packbits(mask.ravel(), bitorder='little').tobytes()).decode()
                      for m, mask in case['_masks'].items()}
        payload.append(c)
    html = '''<!doctype html><meta charset="utf-8"><title>CT background and material review</title>
<style>body{font:16px system-ui;background:#171a20;color:#eee;margin:24px}p{max-width:1150px}label{margin-right:20px}select,input{font:inherit}#panels{display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));gap:12px}article{background:#252a33;padding:12px}canvas{width:100%;image-rendering:pixelated}small{display:block}@media(max-width:900px){#panels{grid-template-columns:repeat(2,1fr)}}</style>
<h1>Candidate material masks</h1><p>Background is estimated separately per volume from the dominant intensity mode among quiet 8×8×8 blocks. The calibration chunks exclude the displayed chunks. These are candidate masks, not ground-truth material segmentation.</p>
<p>Spatial masks cap bright values only for mask calculation, average on a half-resolution grid, then grow from strong evidence (+4 noise scales). Weak growth thresholds are +2 or +1 scales. Original intensities are retained inside the mask. Green overlays show what survives.</p>
<p>Normalization uses foreground-only sampled median/MAD, with scale ≥ twice the estimated background noise. Shared display window [-3,3]; masked background is explicitly black. Noise scales describe voxel variation, not significance levels or standard errors of the smoothed means. Training and cache unchanged.</p>
<label>Chunk <select id="case"></select></label><label>Axis <select id="axis"><option value="0">Z</option><option value="1">Y</option><option value="2">X</option></select></label><label>Slice <input id="slice" type="range" min="0" max="127" value="64"><span id="sl"></span></label><p id="meta"></p><div id="panels"></div>
<script>const cases=__CASES__, methods=__METHODS__, labels=__LABELS__;
const byId=id=>document.getElementById(id),decode=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));let current=-1,data,masks;
const panels=[['raw','Raw'],...methods.map((m,i)=>[m,labels[i]]),['overlay2','Spatial +2: kept in green'],['overlay1','Spatial +1: kept in green']];
cases.forEach((c,i)=>byId('case').add(new Option(c.source+' — '+c.label,i)));byId('panels').innerHTML=panels.map((p,i)=>'<article><strong>'+p[1]+'</strong><canvas id="c'+i+'" width="128" height="128"></canvas><small id="s'+i+'"></small></article>').join('');
function redraw(){let n=+byId('case').value,c=cases[n];if(n!==current){data=decode(c.voxels);masks=Object.fromEntries(methods.map(m=>[m,decode(c.masks[m])]));current=n;}let axis=+byId('axis').value,k=+byId('slice').value;byId('sl').textContent=k;byId('meta').textContent='Background '+c.background.center.toFixed(1)+'; noise scale '+c.background.noise.toFixed(2)+' native uint8 levels | '+c.path;
panels.forEach(([m,title],i)=>{let ctx=byId('c'+i).getContext('2d'),im=ctx.createImageData(128,128),ov=m.startsWith('overlay'),method=ov?(m==='overlay2'?'spatial2':'spatial1'):m,s=c.stats[method];
for(let y=0;y<128;y++)for(let x=0;x<128;x++){let index=axis===0?(k*128+y)*128+x:axis===1?(y*128+k)*128+x:(y*128+x)*128+k,raw=data[index],v=raw/255,keep=m==='raw'?true:((masks[method][index>>3]>>(index&7))&1);if(m!=='raw'&&!ov)v=keep?Math.max(0,Math.min(1,((raw-s.center)/s.scale+3)/6)):0;
let p=4*(y*128+x);im.data[p]=255*(ov&&keep?.55*v:v);im.data[p+1]=255*(ov&&keep?.55*v+.45:v);im.data[p+2]=255*(ov&&keep?.55*v+.135:v);im.data[p+3]=255;}
ctx.putImageData(im,0,0);byId('s'+i).textContent=m==='raw'?'[0,255]':(100*c.fractions[method]).toFixed(1)+'% of nonzero voxels retained';});}
['case','axis','slice'].forEach(id=>byId(id).addEventListener('input',redraw));redraw();</script>'''
    for key, value in [('__CASES__', payload), ('__METHODS__', MASKS), ('__LABELS__', LABELS)]:
        html = html.replace(key, json.dumps(value))
    path.write_text(html)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, default=root/'output/ct_normalization_review/report.json')
    parser.add_argument('--out', type=Path, default=root/'output/ct_background_review')
    parser.add_argument('--seed', type=int, default=532)
    parser.add_argument('--calibration-chunks', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    original = json.loads(args.review.read_text()); excluded = {c['path'] for c in original['cases']}
    backgrounds, allcases, timings = {}, [], {}
    fig, axes = plt.subplots(1, len(original['sources']), figsize=(15, 4))
    for ax, (source, info) in zip(axes, original['sources'].items()):
        bg, rows = calibrate(Path(info['root']), excluded, args.seed, args.calibration_chunks)
        backgrounds[source] = bg
        np.savez_compressed(args.out/(source+'_calibration.npz'), block_median_mad=rows)
        ax.hist(rows[:, 0], bins=np.arange(257), histtype='step', label='All block medians')
        quiet = rows[rows[:, 1] <= bg['quiet_mad_cutoff']]
        ax.hist(quiet[:, 0], bins=np.arange(257), label='Quietest quarter', alpha=.7)
        ax.axvline(bg['center'], color='red', label=f"Background {bg['center']:.0f}")
        ax.set_title(source); ax.set_xlabel('Block median intensity'); ax.legend(fontsize=8)
        print(source, 'candidate background', bg['center'], 'noise', bg['noise'], flush=True)
        group = []
        for old in [c for c in original['cases'] if c['source'] == source]:
            raw = np.array(np.memmap(old['path'], dtype=np.uint8, mode='r', shape=(128,)*3))
            masks = {m: make_mask(raw, bg, m) for m in MASKS}
            stats = {m: material_stats(raw, masks[m], bg) for m in MASKS}
            case = dict(source=source, label=old['label'], path=old['path'], background=bg,
                        stats=stats, fractions={m: float(mask.sum()/max(1, np.count_nonzero(raw))) for m, mask in masks.items()},
                        _raw=raw, _masks=masks)
            group.append(case); allcases.append(case)
        sheet(group, args.out/(source+'.png'))
        raw = np.ascontiguousarray(group[0]['_raw'][:120, :104, :104], dtype=np.float32)
        raw = .75*raw+.25*np.roll(raw, 1, axis=2)  # Fractional interpolated input, native units.
        timings[source] = {}
        for method in MASKS:
            for _ in range(3):
                make_mask(raw, bg, method)
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter_ns(); make_mask(raw, bg, method)
                samples.append((time.perf_counter_ns()-start)/1e6)
            timings[source][method] = dict(mean_ms=float(np.mean(samples)), median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)))
    fig.tight_layout(); fig.savefig(args.out/'background_estimates.png', dpi=140); plt.close(fig)
    report = dict(backgrounds=backgrounds, mask_timings=timings, seed=args.seed,
                  benchmark=dict(shape=[120, 104, 104], input='float32, native intensity units', repeats=args.repeats, warmup=3,
                                 scope='mask only; includes allocation, excludes I/O, calibration, normalization and JIT'),
                  cases=[{k: v for k, v in c.items() if not k.startswith('_')} for c in allcases],
                  limitations=['Unlabeled local cache samples, not a uniform or training-only volume sample.',
                               'Quiet material can contaminate background estimation; faint material can be removed.',
                               'Spatial masks use six-voxel smoothing support and can miss thin isolated material.',
                               'Per-volume background may vary spatially; native-unit thresholds are exploratory.',
                               'Foreground output is a display preview; no production normalization was enabled.'])
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    viewer(allcases, report, args.out/'index.html')
    print(json.dumps(timings, indent=2), flush=True)
    print(args.out/'index.html', flush=True)


if __name__ == '__main__':
    main()
