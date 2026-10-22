"""Read-only posterior-weighted kollesis candidate maps; not a sheet detector."""
import csv
import io
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import softmax

from . import fiber_uv_layout as uv


def conditional_abs(samples, weights, targets):
    """E|sample-target| per row/target, O(K log K), without KxK storage."""
    order = np.argsort(samples, axis=1, kind='stable')
    x = np.take_along_axis(samples, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    sw = np.pad(np.cumsum(w, axis=1), ((0, 0), (1, 0)))
    sx = np.pad(np.cumsum(w*x, axis=1), ((0, 0), (1, 0)))
    low = np.zeros(targets.shape, np.int64)
    high = np.full(targets.shape, x.shape[1], np.int64)
    rows = np.arange(len(x))[:, None]
    for _ in range(x.shape[1].bit_length()):
        mid = (low+high)//2
        go_right = (mid < x.shape[1]) & (x[rows, mid.clip(max=x.shape[1]-1)] < targets)
        low = np.where(go_right, mid+1, low)
        high = np.where(go_right, high, mid)
    return targets*(2*sw[rows, low]-sw[:, -1, None])+sx[:, -1, None]-2*sx[rows, low]


def conditional_discrepancy(samples, weights, centers, measured):
    """E ||sample-center|-measured|; measured must be nonnegative."""
    if np.any(measured < 0):
        raise ValueError('Negative winding distance')
    m = measured[:, None]
    # Identity holds for m>=0, including non-normalized weights.
    result = (conditional_abs(samples, weights, centers-m)
              + conditional_abs(samples, weights, centers+m)
              - conditional_abs(samples, weights, centers)-m*weights.sum(1)[:, None])
    return np.maximum(result, 0)


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float),
                     where=denominator > 0)


class Maps:
    def __init__(self, bounds, step):
        self.step = float(step)
        self.origin = np.floor(np.asarray(bounds[:2])/step)*step
        self.shape = tuple(int(n) for n in (np.floor((np.asarray(bounds[2:])-self.origin)/step).astype(int)+1)[::-1])
        self.values = {}

    def add(self, name, x, y, weight):
        ix = np.floor((x-self.origin[0])/self.step).astype(np.int64)
        iy = np.floor((y-self.origin[1])/self.step).astype(np.int64)
        if np.any((ix < 0) | (ix >= self.shape[1]) | (iy < 0) | (iy >= self.shape[0])):
            raise ValueError('Deposition outside UV grid')
        values = np.bincount((iy*self.shape[1]+ix).ravel(), weights=np.broadcast_to(weight, x.shape).ravel(),
                             minlength=int(np.prod(self.shape))).reshape(self.shape)
        if name not in self.values:
            self.values[name] = values
        else:
            self.values[name] += values


def verify_signature(cfg):
    signature = json.loads(Path(cfg['tensor_cache']).with_suffix('.json').read_text())['signature']
    for key in ('database', 'stage1'):
        if uv.sha256_file(cfg[key]) != signature[key]:
            raise ValueError(f'{key} differs from frozen tensor input')
    return signature


def event_geometry(cfg, d, root):
    """Single read-only streaming scan; shared between CW and CCW by event IDs."""
    if np.any(np.diff(d['event_ids']) <= 0):
        raise ValueError('Event IDs must be strictly increasing')
    signature = verify_signature(cfg)
    target = root/'event_geometry.npz'
    if target.exists():
        with np.load(target) as src:
            if str(src['database_hash']) != signature['database'] or not np.array_equal(src['event_ids'], d['event_ids']):
                raise ValueError('Geometry cache provenance mismatch')
            return src['geometry'].copy()
    geom = np.full((len(d['event_ids']), 3), np.nan)
    found = np.zeros(len(geom), bool)
    fields = ['id', 'ax', 'ay', 'az', 'bx', 'by', 'bz']
    for name in ('normal_a', 'normal_b'):
        fields.extend(f"json_extract({name},'$.xyz[{axis}]')" for axis in range(3))
    started = time.monotonic()
    with uv._ro(cfg['database']) as db:
        cursor = db.execute('SELECT '+','.join(fields)+' FROM constraints ORDER BY id')
        while rows := cursor.fetchmany(65536):
            a = np.asarray(rows, dtype=float)
            pos = np.searchsorted(d['event_ids'], a[:, 0].astype(np.int64))
            good = pos < len(geom)
            good &= d['event_ids'][pos.clip(max=len(geom)-1)] == a[:, 0]
            a = a[good]; pos = pos[good]
            c = a[:, 4:7]-a[:, 1:4]
            na, nb = a[:, 7:10], a[:, 10:13]
            la, lb = np.linalg.norm(na, axis=1), np.linalg.norm(nb, axis=1)
            na = na/np.maximum(la[:, None], 1e-12); nb = nb/np.maximum(lb[:, None], 1e-12)
            pa = (c*na).sum(1); pb = (c*nb).sum(1)
            normal = (abs(pa)+abs(pb))/2
            transverse = np.sqrt(np.maximum(0, (c*c).sum(1)-(pa*pa+pb*pb)/2))
            agreement = abs((na*nb).sum(1))
            geom[pos] = np.column_stack((normal, transverse, agreement))
            geom[pos[(la < 1e-8) | (lb < 1e-8)]] = np.nan
            found[pos] = True
            if int(found.sum())//1000000 != int((found.sum()-len(pos)))//1000000:
                print(f'kollesis geometry {found.sum():,}/{len(geom):,} elapsed={time.monotonic()-started:.1f}s', flush=True)
    if not found.all():
        raise ValueError('Missing original event geometry')
    np.savez(target, geometry=geom, event_ids=d['event_ids'], database_hash=signature['database'])
    return geom


def all_coordinates(d, state, widths):
    """Reuse solver coordinates, on CPU in bounded batches."""
    import torch
    torch.set_num_threads(4)
    b = np.r_[0., np.cumsum(widths, dtype=float)]
    u = np.empty((len(state), len(widths))); v = np.empty_like(u)
    w = torch.as_tensor(widths, dtype=torch.float64); bounds = torch.as_tensor(b)
    for start in range(0, len(state), 8192):
        s = slice(start, start+8192)
        with torch.no_grad():
            x, y, _ = uv.coordinates(torch.as_tensor(state[s], dtype=torch.float64),
                torch.as_tensor(d['phase'][s]), torch.as_tensor(d['z'][s]), torch.as_tensor(d['family'][s]),
                torch.zeros(len(state[s]), dtype=torch.float64), w, bounds)
        u[s], v[s] = x.numpy(), y.numpy()
    return u, v, b


def segment_cells(maps, d, u, v, hypothesis, profile=False):
    """Exact axis-aligned UV segment/cell overlap, one row per piece/cell.

    profile=True collapses V analytically: each V piece counts only once per U
    bucket. Endpoints alone do not create zero-length counts in adjacent cells.
    Length weights are rescaled to the original source arc length to remove
    float32 signed-offset roundoff. No original 3D curve is straightened here:
    this uses the solver's existing axis-aligned UV representation.
    """
    for family in (0, 1):
        ids = np.flatnonzero(d['family'] == family)
        if profile and family == 1:
            yield ids, u[ids, hypothesis], np.zeros(len(ids)), d['lengths'][ids]
            continue
        coordinate = (u if family == 0 else v)[ids, hypothesis]
        lo = coordinate+d['node_ds'][ids][:, [0, 2]].min(1)
        hi = coordinate+d['node_ds'][ids][:, [0, 2]].max(1)
        if np.any(hi <= lo):
            raise ValueError('Zero/nonpositive UV piece span')
        origin = maps.origin[family]
        first = np.floor((lo-origin)/maps.step).astype(np.int64)
        last = np.ceil((hi-origin)/maps.step).astype(np.int64)-1
        counts = last-first+1
        owners = np.repeat(np.arange(len(ids)), counts)
        bins = np.repeat(first, counts)+np.arange(len(owners))-np.repeat(np.cumsum(counts)-counts, counts)
        low = np.maximum(lo[owners], origin+bins*maps.step)
        high = np.minimum(hi[owners], origin+(bins+1)*maps.step)
        # Subtraction at nonzero grid origins can round onto an adjacent edge.
        # Exclude zero-length touches explicitly: they must never count a piece.
        positive = high > low
        owners, low, high = owners[positive], low[positive], high[positive]
        chosen = ids[owners]
        length = (high-low)*d['lengths'][chosen]/(hi-lo)[owners]
        pos = (low+high)/2
        x = pos if family == 0 else u[chosen, hypothesis]
        y = v[chosen, hypothesis] if family == 0 else pos
        if profile:
            y = np.zeros(len(chosen))
        yield chosen, x, y, length


def length_maps(maps, d, u, v, probability, sample_step):
    keep = probability[:, :-1].sum(1)
    conditional = probability[:, :-1]/keep[:, None]
    entropy = -(conditional*np.log(np.maximum(conditional, 1e-300))).sum(1)/np.log(conditional.shape[1])
    intersections = 0
    for k in range(conditional.shape[1]):
        for ids, x, y, length in segment_cells(maps, d, u, v, k):
            intersections += len(ids)
            weight = length*conditional[ids, k]
            maps.add('length_support', x, y, weight)
            maps.add('rejection_sum', x, y, weight*probability[ids, -1])
            maps.add('retained_length', x, y, weight*keep[ids])
            maps.add('entropy_sum', x, y, weight*keep[ids]*entropy[ids])
    return dict(segment_cell_intersections=intersections, deposition='exact axis-aligned segment overlap', legacy_sample_step_ignored=sample_step)


def edge_maps(maps, d, u, v, probability, widths, boundaries, geom, config):
    keep = probability[:, :-1].sum(1)
    cp = probability[:, :-1]/keep[:, None]
    bands = config.get('normal_bands', [0, 8, 16, 32, 64, 128])
    started = time.monotonic(); size = int(config.get('event_batch', 2048))
    aggregate = maps
    def batch(first):
        maps = Maps.__new__(Maps)
        maps.step, maps.origin, maps.shape, maps.values = aggregate.step, aggregate.origin, aggregate.shape, {}
        s = slice(first, first+size); i, j = d['i'][s], d['j'][s]
        xa = u[i]+((d['family'][i] == 0)*d['ds_a'][s])[:, None]
        ya = v[i]+((d['family'][i] == 1)*d['ds_a'][s])[:, None]
        xb = u[j]+((d['family'][j] == 0)*d['ds_b'][s])[:, None]
        yb = v[j]+((d['family'][j] == 1)*d['ds_b'][s])[:, None]
        def q(x):
            idx = (np.searchsorted(boundaries, x, side='right')-1).clip(0, len(widths)-1)
            return idx+(x-boundaries[idx])/widths[idx]
        qa, qb = q(xa), q(xb)
        shift = d['phase_delta'][s]-.5*(d['family'][j]-d['family'][i])
        ca = conditional_discrepancy(qb, cp[j], qa+shift[:, None], d['distance'][s])
        cb = conditional_discrepancy(qa, cp[i], qb-shift[:, None], d['distance'][s])
        g = geom[s]
        eligible = (d['parallel'][s] & (d['family'][i] == d['family'][j])
            & (d['source_id'][i] != d['source_id'][j]) & np.isfinite(g).all(1)
            & (g[:, 2] >= config.get('normal_agreement', .9))
            & (g[:, 1] <= config.get('transverse_max', 32)) & (g[:, 0] <= bands[-1]))
        for x, y, own, other, cost in ((xa, ya, i, j, ca), (xb, yb, j, i, cb)):
            weight = .5*probability[own, :-1]*keep[other, None]
            maps.add('event_support', x, y, weight)
            maps.add('discrepancy_sum', x, y, weight*cost)
            for name, selected in (('parallel', d['parallel'][s]), ('perpendicular', ~d['parallel'][s])):
                maps.add(name+'_support', x, y, weight*selected[:, None])
                maps.add(name+'_discrepancy_sum', x, y, weight*cost*selected[:, None])
            if eligible.any():
                xx, yy, ww = x[eligible], y[eligible], weight[eligible]
                maps.add('crowding_support', xx, yy, ww)
                for n, (lo, hi) in enumerate(zip(bands[:-1], bands[1:])):
                    selected = (g[eligible, 0] >= lo) & ((g[eligible, 0] < hi) if n < len(bands)-2 else (g[eligible, 0] <= hi))
                    maps.add(f'normal_{lo}_{hi}', xx, yy, ww*selected[:, None])
        return maps.values, int(eligible.sum())

    eligible_total = 0
    # Bounded work-ahead; read-only shared arrays, private histogram reductions.
    # Reduction order remains fixed, regardless of thread completion order.
    workers = int(config.get('event_workers', 8))
    starts = iter(range(0, len(d['i']), size))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(workers):
            first = next(starts, None)
            if first is not None:
                pending.append((first, pool.submit(batch, first)))
        while pending:
            first, future = pending.popleft()
            values, count = future.result()
            eligible_total += count
            for name, value in values.items():
                if name not in aggregate.values:
                    aggregate.values[name] = value
                else:
                    aggregate.values[name] += value
            del values
            next_start = next(starts, None)
            if next_start is not None:
                pending.append((next_start, pool.submit(batch, next_start)))
            if first != 0 and (first//size+1) % 100 != 0:
                continue
            elapsed = time.monotonic()-started
            done = min(first+size, len(d['i']))
            print(f'kollesis events {done:,}/{len(d["i"]):,} elapsed={elapsed:.1f}s eta={elapsed*(len(d["i"])-done)/done:.1f}s', flush=True)
    return dict(eligible_crowding_encounters=eligible_total, event_seconds=time.monotonic()-started)


def metric_definitions(bands):
    definitions = {
        'rejection': ('rejection_sum', 'length_support', 'Rejected probability (%)', 100.),
        'uncertainty': ('entropy_sum', 'retained_length', 'Conditional winding entropy (%)', 100.),
        'discrepancy': ('discrepancy_sum', 'event_support', 'Posterior mean |absolute layer gap - Lasagna| (turns)', 1.),
        'parallel_discrepancy': ('parallel_discrepancy_sum', 'parallel_support', 'Parallel distance discrepancy (turns)', 1.),
        'perpendicular_discrepancy': ('perpendicular_discrepancy_sum', 'perpendicular_support', 'Perpendicular distance discrepancy (turns)', 1.),
        'close_encounter_rate': ('close_pair_sum', 'retained_length', 'Close (0-32vx) encounters / 1000 retained fiber vx', 1000.),
    }
    for lo, hi in zip(bands[:-1], bands[1:]):
        definitions[f'normal_{lo}_{hi}'] = (f'normal_{lo}_{hi}', 'crowding_support', f'Eligible encounters: normal separation {lo}-{hi} vx (%)', 100.)
    return definitions


def write_plots(root, maps, config, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    root.mkdir(parents=True, exist_ok=True)
    definitions = metric_definitions(config.get('normal_bands', [0, 8, 16, 32, 64, 128]))
    maps.values['close_pair_sum'] = sum((maps.values.get(f'normal_{a}_{b}', np.zeros(maps.shape))
        for a, b in zip(config.get('normal_bands', [0, 8, 16, 32, 64, 128])[:-1],
                        config.get('normal_bands', [0, 8, 16, 32, 64, 128])[1:]) if b <= 32), np.zeros(maps.shape))
    x = maps.origin[0]+(np.arange(maps.shape[1])+.5)*maps.step
    metrics = {}; profiles = {}; stats = {}
    for name, (numerator, denominator, title, scale) in definitions.items():
        n = maps.values.get(numerator, np.zeros(maps.shape))
        den = maps.values.get(denominator, np.zeros(maps.shape))
        raw = scale*ratio(n, den)
        # Posterior tails have nonzero support everywhere; use disclosed practical support.
        minimum = config.get('min_length_support', 1000.) if denominator in ('length_support', 'retained_length') else config.get('min_event_support', 5.)
        supported = den >= minimum
        shown = np.where(supported, raw, np.nan)
        profile_den = den.sum(0)
        profile = scale*ratio(n.sum(0), profile_den)
        profile[profile_den < minimum] = np.nan
        metrics[name] = shown; profiles[name] = profile
        vals = raw[supported]; weights = den[supported]
        if len(vals):
            order = np.argsort(vals); cs = np.cumsum(weights[order]); threshold = vals[order[np.searchsorted(cs, cs[-1]*.75)]]
        else:
            threshold = float('nan')
        elevated = ratio(((raw > threshold) & supported).sum(0).astype(float), supported.sum(0).astype(float))*100
        stats[name] = dict(global_mean=float(scale*n.sum()/den.sum()) if den.sum() else None,
            support_total=float(den.sum()), supported_cells=int(supported.sum()),
            minimum_cell_support=minimum, elevation_threshold=float(threshold),
            top_u_buckets=[dict(u=float(x[k]), value=float(profile[k]), support=float(profile_den[k]))
                for k in np.argsort(np.nan_to_num(profile, nan=-np.inf))[-10:][::-1] if np.isfinite(profile[k])])
        profiles[name+'_support'] = profile_den
        profiles[name+'_supported_v_percent'] = 100*supported.mean(0)
        profiles[name+'_elevated_supported_v_percent'] = elevated
        good = np.isfinite(profile) & (supported.mean(0) >= .25)
        peaks = []
        for k in np.argsort(np.where(good, profile, -np.inf))[::-1]:
            if not good[k] or len(peaks) == 10:
                break
            if any(abs(x[k]-peak['u']) < 10000 for peak in peaks):
                continue
            peaks.append(dict(u=float(x[k]), value=float(profile[k]), support=float(profile_den[k]),
                supported_v_percent=float(100*supported[:, k].mean()), elevated_supported_v_percent=float(elevated[k])))
        stats[name]['peaks_with_25percent_v_support'] = peaks
    from scipy.stats import spearmanr
    selected = ('rejection', 'uncertainty', 'discrepancy', 'close_encounter_rate')
    correlations = []
    for a, name in enumerate(selected):
        for other in selected[a+1:]:
            good = (np.isfinite(profiles[name]) & np.isfinite(profiles[other])
                & (profiles[name+'_supported_v_percent'] >= 25) & (profiles[other+'_supported_v_percent'] >= 25))
            variable = good.sum() >= 3 and np.ptp(profiles[name][good]) > 0 and np.ptp(profiles[other][good]) > 0
            correlations.append(dict(a=name, b=other, buckets=int(good.sum()),
                spearman=float(spearmanr(profiles[name][good], profiles[other][good]).statistic) if variable else None))
    uv.atomic_write(root/'profile_correlations.json', dict(minimum_supported_v_percent=25, correlations=correlations), durable=False)
    def save(fig, stem):
        if config.get('plot_names') and stem not in config['plot_names']:
            plt.close(fig)
            return
        fig.savefig(root/(stem+'.svg'))
        if config.get('inspection_previews'):
            preview = Path(config['inspection_previews'])/label.lower()
            preview.mkdir(parents=True, exist_ok=True)
            fig.savefig(preview/(stem+'.png'), dpi=85)
        memory = io.BytesIO(); fig.savefig(memory, format='png', dpi=125); memory.seek(0)
        with Image.open(memory) as img:
            img.convert('RGB').save(root/(stem+'.tif'), compression='tiff_lzw')
        plt.close(fig)
    fig, axes = plt.subplots(7, 1, figsize=(24, 15), sharex=True, constrained_layout=True)
    short_labels = dict(rejection='Rejection\n(%)', uncertainty='Winding entropy\n(%)', discrepancy='Distance discrepancy\n(turns)')
    for ax, name in zip(axes[:3], ('rejection', 'uncertainty', 'discrepancy')):
        ax.plot(x, profiles[name], lw=.7); ax.set_ylabel(short_labels[name]); ax.grid(alpha=.2)
    for name in ('parallel_discrepancy', 'perpendicular_discrepancy'):
        axes[3].plot(x, profiles[name], lw=.7, label=name)
    axes[3].legend(); axes[3].set_ylabel('Distance discrepancy (turns)')
    for name in definitions:
        if name.startswith('normal_'):
            axes[4].plot(x, profiles[name], lw=.7, label=name.removeprefix('normal_')+' vx')
    axes[4].legend(ncol=5); axes[4].set_ylabel('Normal-distance bands (%)')
    for name in ('event_support', 'crowding_support'):
        den = maps.values.get(name, np.zeros(maps.shape)).sum(0)
        axes[5].plot(x, den, lw=.7, label=name)
    axes[5].set_yscale('log'); axes[5].legend(); axes[5].set_ylabel('Weighted encounters')
    axes[6].plot(x, profiles['close_encounter_rate'], lw=.7); axes[6].set_ylabel('Close encounters / 1000vx'); axes[6].set_xlabel('U (base voxels)')
    fig.suptitle(label+' — all-height profiles; 1000-vx buckets; saved20; no endpoint metric')
    save(fig, 'profiles')
    for name, z in metrics.items():
        # Split U into strips; each panel uses equal physical U/V scale.
        panels = int(config.get('heatmap_panels', 8))
        edges = np.linspace(0, maps.shape[1], panels+1).astype(int)
        fig, axes = plt.subplots(panels, 1, figsize=(22, 2*panels), constrained_layout=True)
        values = z[np.isfinite(z)]
        vmax = 100 if definitions[name][3] == 100 else (max(.01, np.quantile(values, .99)) if len(values) else 1)
        for ax, a, b in zip(axes, edges[:-1], edges[1:]):
            im = ax.imshow(z[:, a:b], origin='lower', interpolation='nearest', aspect='equal', cmap='viridis', vmin=0, vmax=vmax,
                extent=(maps.origin[0]+a*maps.step, maps.origin[0]+b*maps.step,
                        maps.origin[1], maps.origin[1]+maps.shape[0]*maps.step))
            ax.set_ylabel('V (vx)'); ax.set_facecolor('#ddd')
        axes[-1].set_xlabel('U (base voxels)')
        fig.colorbar(im, ax=list(axes), label=definitions[name][2], shrink=.7)
        fig.suptitle(label+' — '+definitions[name][2]+'; gray = insufficient support')
        save(fig, name+'_heatmap')
    # Matching support maps, not just confidence-filtered metric maps.
    for name in ('length_support', 'retained_length', 'event_support', 'crowding_support'):
        panels = int(config.get('heatmap_panels', 8))
        edges = np.linspace(0, maps.shape[1], panels+1).astype(int)
        fig, axes = plt.subplots(panels, 1, figsize=(22, 2*panels), constrained_layout=True)
        den = maps.values.get(name, np.zeros(maps.shape))
        z = np.where(den > 0, np.log10(np.maximum(den, 1e-8)), np.nan)
        for ax, a, b in zip(axes, edges[:-1], edges[1:]):
            im = ax.imshow(z[:, a:b], origin='lower', aspect='equal', interpolation='nearest', vmin=-2,
                vmax=max(0., float(np.log10(max(1., den.max())))),
                extent=(maps.origin[0]+a*maps.step, maps.origin[0]+b*maps.step,
                        maps.origin[1], maps.origin[1]+maps.shape[0]*maps.step))
            ax.set_ylabel('V (vx)'); ax.set_facecolor('#ddd')
        axes[-1].set_xlabel('U (vx)'); fig.suptitle(label+' '+name+' — log10; equal physical U/V scale')
        fig.colorbar(im, ax=list(axes), shrink=.7); save(fig, name+'_support')
    fig, axes = plt.subplots(2, 1, figsize=(24, 7), sharex=True, constrained_layout=True)
    for name in ('rejection', 'uncertainty', 'discrepancy', 'close_encounter_rate', 'normal_0_8', 'normal_8_16', 'normal_16_32'):
        axes[0].plot(x, profiles[name+'_elevated_supported_v_percent'], lw=.7, label=name)
        axes[1].plot(x, profiles[name+'_supported_v_percent'], lw=.7, label=name)
    axes[0].set_ylabel('Elevated / supported V bins (%)'); axes[0].legend(ncol=3)
    axes[1].set_ylabel('Supported / all V bins (%)'); axes[1].set_xlabel('U (base voxels)')
    fig.suptitle(label+' — V coherence; metric-specific weighted global 75th percentile')
    save(fig, 'v_coherence')
    keys = list(profiles)
    with (root/'profiles.csv').open('w') as f:
        writer = csv.writer(f); writer.writerow(['u_center', *keys])
        writer.writerows(zip(x, *(profiles[k] for k in keys)))
    np.savez_compressed(root/'maps.npz', origin=maps.origin, step=maps.step,
                        **{'raw_'+k: v for k, v in maps.values.items()}, **metrics)
    uv.atomic_write(root/'metrics.json', stats, durable=False)
    return stats


def run(config):
    root = Path(config['output_root']); root.mkdir(parents=True, exist_ok=True)
    reports = []
    for entry in config['solutions']:
        started = time.monotonic(); run_root = Path(entry['run_root'])
        cfg = json.loads((run_root/'summary.json').read_text())['config']
        out = root/entry['label'].lower(); out.mkdir(parents=True, exist_ok=True)
        if config.get('plots_only'):
            evidence = json.loads((out/'accumulation.json').read_text())
            if evidence['solution_sha256'] != uv.sha256_file(run_root/'solution.npz') or evidence['tensor_sha256'] != uv.sha256_file(cfg['tensor_cache']):
                raise ValueError('Replot source mismatch')
            for key in ('bucket_size', 'sample_step', 'normal_bands', 'normal_agreement', 'transverse_max'):
                if config.get(key) != evidence['config'].get(key):
                    raise ValueError(f'Cannot change accumulated metric setting {key}')
            with np.load(out/'accumulation.npz') as src:
                maps = Maps.__new__(Maps)
                maps.origin, maps.step, maps.shape = src['origin'], float(src['step']), tuple(int(n) for n in src['shape'])
                maps.values = {k: src[k] for k in src.files if k not in ('origin', 'step', 'shape')}
            stats = write_plots(out, maps, config, entry['label'])
            reports.append(dict(label=entry['label'], metrics=stats))
            print(f'kollesis {entry["label"]}: replotted existing evidence', flush=True)
            continue
        print(f'kollesis {entry["label"]}: loading saved solution', flush=True)
        with np.load(cfg['tensor_cache']) as src:
            d = {key: src[key] for key in src.files}
        with np.load(run_root/'solution.npz') as src:
            state, widths = src['state'].copy(), src['widths'].astype(float)
            for key in ('source_id', 'start', 'end', 'family'):
                if not np.array_equal(d[key], src[key]):
                    raise ValueError('Saved piece identities mismatch')
        geom = event_geometry(cfg, d, root)
        u, v, boundaries = all_coordinates(d, state, widths)
        probability = softmax(state[:, :len(widths)+1].astype(float), axis=1)
        if np.any(probability[:, :-1].sum(1) <= 0):
            raise ValueError('Undefined conditional winding distribution')
        dx = (d['family'] == 0)[:, None]*d['node_ds'][:, [0, 2]]
        dy = (d['family'] == 1)[:, None]*d['node_ds'][:, [0, 2]]
        bounds = (float((u.min(1)+dx.min(1)).min()), float((v.min(1)+dy.min(1)).min()),
                  float((u.max(1)+dx.max(1)).max()), float((v.max(1)+dy.max(1)).max()))
        maps = Maps(bounds, config.get('bucket_size', 1000.))
        print(f'kollesis {entry["label"]}: grid={maps.shape}, fiber length deposition', flush=True)
        length = length_maps(maps, d, u, v, probability, config.get('sample_step', 250.))
        edges = edge_maps(maps, d, u, v, probability, widths, boundaries, geom, config)
        print(f'kollesis {entry["label"]}: writing plots', flush=True)
        # Persist expensive accumulation before plotting; permit validated replot.
        np.savez_compressed(out/'accumulation.npz', origin=maps.origin, step=maps.step, shape=maps.shape, **maps.values)
        uv.atomic_write(out/'accumulation.json', dict(solution_sha256=uv.sha256_file(run_root/'solution.npz'),
            tensor_sha256=uv.sha256_file(cfg['tensor_cache']), config=config, length=length, edges=edges), durable=False)
        stats = write_plots(out, maps, config, entry['label'])
        keep = probability[:, :-1].sum(1)
        report = dict(label=entry['label'], run_root=str(run_root), grid_shape=maps.shape,
            solution_sha256=uv.sha256_file(run_root/'solution.npz'), tensor_sha256=uv.sha256_file(cfg['tensor_cache']),
            config=config, length=length, edges=edges, metrics=stats, seconds=time.monotonic()-started,
            expected_total_length=float(d['lengths'].sum()), deposited_total_length=float(maps.values['length_support'].sum()),
            expected_event_support=float((keep[d['i']]*keep[d['j']]).sum()),
            deposited_event_support=float(maps.values['event_support'].sum()))
        uv.atomic_write(out/'summary.json', report, durable=False); reports.append(report)
        print(f'kollesis {entry["label"]}: finished {report["seconds"]:.1f}s', flush=True)
        del d, state, probability, u, v, maps, geom
    if not config.get('plots_only'):
        uv.atomic_write(root/'results.json', reports, durable=False)
    return reports
