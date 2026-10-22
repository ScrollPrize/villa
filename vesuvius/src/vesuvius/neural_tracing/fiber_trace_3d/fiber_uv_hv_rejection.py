"""H/V rejection diagnostics with exact length and distinct piece-cell counts."""
import csv
import io
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax

from . import fiber_uv_layout as uv
from .fiber_uv_kollesis import Maps, all_coordinates, ratio, segment_cells


METRICS = {
    'disabled_length': ('disabled_length', 'length', 100., 'Disabled length (%)'),
    'disabled_count': ('disabled_count', None, 1., 'Disabled piece count'),
    'disabled_piece_fraction': ('disabled_count', 'count', 100., 'Disabled pieces (%)'),
    'probability_piece': ('prob_count', 'count', 100., 'Mean P(broken), piece-weighted (%)'),
    'probability_length': ('prob_length', 'length', 100., 'Mean P(broken), length-weighted (%)'),
}


def deposit(d, u, v, p, heat_step=1000., profile_step=10000.):
    cp = p[:, :-1]/p[:, :-1].sum(1)[:, None]
    pick = cp.argmax(1); broken = p[:, -1] > .5
    dx = (d['family'] == 0)[:, None]*d['node_ds'][:, [0, 2]]
    dy = (d['family'] == 1)[:, None]*d['node_ds'][:, [0, 2]]
    bounds = (float((u.min(1)+dx.min(1)).min()), float((v.min(1)+dy.min(1)).min()),
              float((u.max(1)+dx.max(1)).max()), float((v.max(1)+dy.max(1)).max()))
    grids = {mode: {'heat': Maps(bounds, heat_step), 'profile': Maps((bounds[0], 0., bounds[2], 0.), profile_step)}
             for mode in ('posterior', 'map')}
    started = time.monotonic()
    for k in range(cp.shape[1]):
        for target in ('heat', 'profile'):
            template = grids['posterior'][target]
            for ids, x, y, length in segment_cells(template, d, u, v, k, profile=target == 'profile'):
                if not len(ids):
                    continue
                family = 'h' if d['family'][ids[0]] == 0 else 'v'
                for mode in ('posterior', 'map'):
                    chosen = np.ones(len(ids), bool) if mode == 'posterior' else pick[ids] == k
                    if not chosen.any():
                        continue
                    ix, xx, yy, ll = ids[chosen], x[chosen], y[chosen], length[chosen]
                    weight = cp[ix, k] if mode == 'posterior' else np.ones(len(ix))
                    values = dict(count=weight, length=weight*ll, disabled_count=weight*broken[ix],
                        disabled_length=weight*ll*broken[ix], prob_count=weight*p[ix, -1], prob_length=weight*ll*p[ix, -1])
                    for name, val in values.items():
                        grids[mode][target].add(family+'_'+name, xx, yy, val)
        if (k+1) % 10 == 0:
            print(f'H/V rejection {k+1}/{cp.shape[1]} hypotheses, {time.monotonic()-started:.1f}s', flush=True)
    return grids


def comparisons(grid, name, cfg):
    numerator, denominator, scale, _ = METRICS[name]
    values, supports = {}, {}
    for family in ('h', 'v'):
        get = lambda key: grid.values.get(family+'_'+key, np.zeros(grid.shape))
        counts, lengths = get('count'), get('length')
        good = counts >= cfg.get('minimum_count', 5.)
        if denominator == 'length':
            good &= lengths >= cfg.get('minimum_length', 1000.)
        val = get(numerator) if denominator is None else scale*ratio(get(numerator), get(denominator))
        values[family] = np.where(good, val, np.nan)
        supports[family] = good
    diff = values['v']-values['h']
    minimum_h = cfg.get('ratio_minimum_h_count', 3.) if denominator is None else 100*cfg.get('ratio_minimum_h_fraction', .01)
    divided = ratio(values['v'], values['h'])
    divided[values['h'] < minimum_h] = np.nan
    return dict(h=values['h'], v=values['v'], difference=diff, relative=divided)


def save_figure(fig, root, name, cfg, label):
    import matplotlib.pyplot as plt
    from PIL import Image
    fig.savefig(root/(name+'.svg'))
    memory = io.BytesIO(); fig.savefig(memory, format='png', dpi=125); memory.seek(0)
    with Image.open(memory) as im:
        im.convert('RGB').save(root/(name+'.tif'), compression='tiff_lzw')
    if cfg.get('inspection_previews'):
        path = Path(cfg['inspection_previews'])/label
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path/(name+'.png'), dpi=75)
    plt.close(fig)


def plots(grids, root, cfg, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root.mkdir(parents=True, exist_ok=True)
    stats = {}
    for mode, pair in grids.items():
        dest = root/mode; dest.mkdir(exist_ok=True)
        grid = pair['profile']; x = grid.origin[0]+(np.arange(grid.shape[1])+.5)*grid.step
        fig, axes = plt.subplots(len(METRICS), 3, figsize=(25, 14), sharex=True, constrained_layout=True)
        columns = {'u_center': x}; stats[mode] = {}
        for row, (name, (_, denominator, _, title)) in enumerate(METRICS.items()):
            if name == 'disabled_count' and mode == 'posterior':
                title = 'Disabled pieces\n(weighted count)'
            arrays = comparisons(grid, name, cfg)
            axes[row, 0].plot(x, arrays['h'][0], color='tab:blue', label='H', lw=.8)
            axes[row, 0].plot(x, arrays['v'][0], color='tab:orange', label='V', lw=.8)
            axes[row, 0].set_ylabel(title.replace(', ', '\n')); axes[row, 0].legend()
            axes[row, 1].plot(x, arrays['difference'][0], lw=.8)
            axes[row, 1].axhline(0, color='gray', lw=.6)
            axes[row, 2].plot(x, arrays['relative'][0], lw=.8)
            axes[row, 2].axhline(1, color='gray', lw=.6)
            for kind, val in arrays.items():
                columns[name+'_'+kind] = val[0]
            for ax in axes[row]:
                ax.grid(alpha=.2)
            good = np.isfinite(arrays['relative'][0])
            ids = np.flatnonzero(good)
            strongest = ids[np.argsort(arrays['relative'][0, ids])[-10:][::-1]]
            stats[mode][name] = dict(valid_ratio_buckets=int(good.sum()), strongest_ratios=[dict(
                u=float(x[k]), h=float(arrays['h'][0, k]), v=float(arrays['v'][0, k]), ratio=float(arrays['relative'][0, k]),
                h_count=float(grid.values['h_count'][0, k]), v_count=float(grid.values['v_count'][0, k])) for k in strongest])
        for ax, title in zip(axes[0], ('H and V', 'V − H (percentage points or counts)', 'V / H (fold ratio; weak H masked)')):
            ax.set_title(title)
        for ax in axes[-1]:
            ax.set_xlabel('U (base voxels)')
        fig.suptitle(f'{label} — {mode} placement; {grid.step:g}-vx U buckets; P(broken)>0.5 disabled')
        save_figure(fig, dest, 'profiles', cfg, label.lower()+'/'+mode)
        # Show absolute support and count comparisons on their own, not just percentages.
        fig, axes = plt.subplots(2, 1, figsize=(22, 6), sharex=True, constrained_layout=True)
        for row, name in enumerate(('count', 'length')):
            for family in ('h', 'v'):
                val = grid.values[family+'_'+name][0]
                axes[row].plot(x, val, lw=.8, label=family.upper())
                columns[family+'_'+name] = val
            axes[row].set_ylabel('Intersecting pieces' if name == 'count' else 'Fiber length (vx)')
            axes[row].set_yscale('log'); axes[row].legend()
        axes[-1].set_xlabel('U (base voxels)'); fig.suptitle(f'{label} — {mode} support; counts once per piece/U bucket')
        save_figure(fig, dest, 'support_profiles', cfg, label.lower()+'/'+mode)
        with (dest/'profiles.csv').open('w') as f:
            writer = csv.writer(f); writer.writerow(columns); writer.writerows(zip(*columns.values()))
        for target, g in pair.items():
            np.savez_compressed(dest/(target+'_raw.npz'), origin=g.origin, step=g.step, shape=g.shape, **g.values)
        heat = pair['heat']; panels = int(cfg.get('heatmap_panels', 8))
        edges = np.linspace(0, heat.shape[1], panels+1).astype(int)
        for name, (_, denominator, scale, title) in METRICS.items():
            if name == 'disabled_count' and mode == 'posterior':
                title = 'Disabled pieces (weighted count)'
            arrays = comparisons(heat, name, cfg)
            finite = np.r_[arrays['h'][np.isfinite(arrays['h'])], arrays['v'][np.isfinite(arrays['v'])]]
            upper = 100 if scale == 100 else max(1, np.quantile(finite, .99) if len(finite) else 1)
            delta = abs(arrays['difference'][np.isfinite(arrays['difference'])])
            limit = max(.01, np.quantile(delta, .99) if len(delta) else 1)
            rel = arrays['relative'][np.isfinite(arrays['relative'])]
            rlimit = max(2., np.quantile(rel, .99) if len(rel) else 2.)
            fig, axes = plt.subplots(panels, 4, figsize=(30, 2*panels), constrained_layout=True)
            for col, (kind, title_col, low, high, cmap) in enumerate((
                ('h', 'H', 0, upper, 'viridis'), ('v', 'V', 0, upper, 'viridis'),
                ('difference', 'V − H', -limit, limit, 'coolwarm'), ('relative', 'V / H', 0, rlimit, 'viridis'))):
                for row, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                    ax = axes[row, col]
                    im = ax.imshow(arrays[kind][:, a:b], origin='lower', interpolation='nearest', aspect='equal',
                        vmin=low, vmax=high, cmap=cmap, extent=(heat.origin[0]+a*heat.step, heat.origin[0]+b*heat.step,
                        heat.origin[1], heat.origin[1]+heat.shape[0]*heat.step))
                    ax.set_facecolor('#ddd')
                    if col == 0:
                        ax.set_ylabel('V (vx)')
                    if row == 0:
                        ax.set_title(title_col)
                    if row == panels-1:
                        ax.set_xlabel('U (vx)')
                fig.colorbar(im, ax=list(axes[:, col]), shrink=.65, label='fold ratio' if kind == 'relative' else ('pp' if kind == 'difference' and scale == 100 else title))
            fig.suptitle(f'{label} — {mode}: {title}; {heat.step:g}vx cells; gray = insufficient support; color extremes clipped only for display')
            save_figure(fig, dest, name+'_heatmaps', cfg, label.lower()+'/'+mode)
    uv.atomic_write(root/'metric_peaks.json', stats, durable=False)


def run(cfg):
    root = Path(cfg['output_root']); root.mkdir(parents=True, exist_ok=True)
    reports = []
    for entry in cfg['solutions']:
        start = time.monotonic(); run_root = Path(entry['run_root'])
        source_cfg = json.loads((run_root/'summary.json').read_text())['config']
        with np.load(source_cfg['tensor_cache']) as src:
            d = {k: src[k] for k in ('family', 'phase', 'z', 'node_ds', 'lengths', 'source_id', 'start', 'end')}
        with np.load(run_root/'solution.npz') as src:
            state, widths = src['state'].copy(), src['widths'].astype(float)
            for key in ('source_id', 'start', 'end', 'family'):
                if not np.array_equal(src[key], d[key]):
                    raise ValueError('Solution piece identity mismatch')
        print(f'H/V rejection {entry["label"]}: coordinates', flush=True)
        u, v, _ = all_coordinates(d, state, widths)
        p = softmax(state[:, :len(widths)+1].astype(float), axis=1)
        grids = deposit(d, u, v, p, cfg.get('heatmap_step', 1000.), cfg.get('profile_step', 10000.))
        out = root/entry['label'].lower(); out.mkdir(parents=True, exist_ok=True)
        summary = dict(label=entry['label'], run_root=str(run_root), config=cfg,
            solution_sha256=uv.sha256_file(run_root/'solution.npz'), tensor_sha256=uv.sha256_file(source_cfg['tensor_cache']),
            analysis_sha256=uv.sha256_file(__file__), families={})
        for family, label in ((0, 'h'), (1, 'v')):
            select = d['family'] == family; length = d['lengths'][select]; prob = p[select, -1]; disabled = prob > .5
            summary['families'][label] = dict(pieces=int(select.sum()), disabled_pieces=int(disabled.sum()),
                source_fibers=int(len(np.unique(d['source_id'][select]))), length=float(length.sum()),
                disabled_length=float(length[disabled].sum()), disabled_length_percent=float(100*length[disabled].sum()/length.sum()),
                disabled_piece_percent=float(100*disabled.mean()), mean_probability_piece_percent=float(100*prob.mean()),
                mean_probability_length_percent=float(100*(prob*length).sum()/length.sum()))
            for mode, pair in grids.items():
                for target, grid in pair.items():
                    for name, total in (('length', length.sum()), ('disabled_length', length[disabled].sum()), ('prob_length', (prob*length).sum())):
                        if not np.isclose(grid.values[label+'_'+name].sum(), total, rtol=1e-10, atol=1e-6):
                            raise ValueError('Length mass not conserved')
                    if mode == 'map' and not np.all(grid.values[label+'_count'] == np.floor(grid.values[label+'_count'])):
                        raise ValueError('MAP counts must be integer')
        summary['calculation_seconds'] = time.monotonic()-start
        uv.atomic_write(out/'summary.json', summary, durable=False)
        print(f'H/V rejection {entry["label"]}: writing plots', flush=True)
        plots(grids, out, cfg, entry['label'])
        summary['seconds'] = time.monotonic()-start; uv.atomic_write(out/'summary.json', summary, durable=False)
        reports.append(summary)
        print(f'H/V rejection {entry["label"]}: finished {summary["seconds"]:.1f}s', flush=True)
        del d, state, u, v, p, grids
    uv.atomic_write(root/'results.json', reports, durable=False)
    return reports
