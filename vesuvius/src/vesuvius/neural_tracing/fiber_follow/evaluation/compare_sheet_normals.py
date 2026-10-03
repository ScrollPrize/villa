"""Compare CT tensors, public Lasagna, and crossing H/V fiber normals.

Run with the existing vesuvius Python environment, from fiber_follow/:
  python evaluation/compare_sheet_normals.py --out output/sheet_normals --select-only
  python evaluation/compare_sheet_normals.py --out output/sheet_normals

Selection uses only fiber geometry, never agreement with either normal field.
Network reads are cached under --out/cache; source datasets are read-only.
"""
import argparse
import base64
import csv
import html
import json
import os
from pathlib import Path
import sys
from urllib.request import urlopen

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[4]))  # sibling lasagna package in the monorepo
from vesuvius.neural_tracing.fiber_follow.data.afv import AFVFibers
from vesuvius.neural_tracing.fiber_follow.data.data import load_fibers, TracedFiber
from vesuvius.neural_tracing.fiber_follow.data.neighbor_mining import exact_nearest
from vesuvius.neural_tracing.fiber_follow.data.volume import ChunkedArray, RemoteChunkedArray
from vesuvius.neural_tracing.fiber_follow.shared.geometry import interp_at, normalize, arclength
from vesuvius.neural_tracing.fiber_follow.tracing.heading import ct_structure_tensor
from vesuvius.neural_tracing.fiber_trace.geometry import decode_lasagna_normals_xyz

BUCKET = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
MANIFESTS = {
    'paris4': 'PHercParis4/representations/predictions/lasagna/20260411134726-lasagna-20260419180421-L2/PHercParis4-20260411134726-las-sd2-5b17ff6c.lasagna.json',
    '0175A_5mm_v1': None,
    '1447_5mm_v1': 'PHerc1447/representations/predictions/lasagna/20250521151220-lasagna-20260419180421/PHerc1447.lasagna.json',
}


def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(normalize(a)*normalize(b), axis=-1)), 0, 1)))


def fit_tangent(fiber, point, half):
    _, _, segment, fraction = exact_nearest(np.asarray(point)[None], fiber.points)
    t = fiber.s[segment[0]] + fraction[0]*(fiber.s[segment[0]+1]-fiber.s[segment[0]])
    if t < half or fiber.length-t < half:
        return None
    offsets = np.linspace(-half, half, 25)
    points = interp_at(fiber.points, fiber.s, t+offsets)
    return normalize(offsets @ (points-points.mean(0)))


def crossing(h, v, center, max_gap=2.):
    hp = h.points[np.linalg.norm(h.points-center, axis=1) < 64]
    if len(hp) < 2:
        return None
    distance, q, _, _ = exact_nearest(hp, v.points)
    p, q = hp[distance.argmin()], q[distance.argmin()]
    for _ in range(12):
        p = exact_nearest(q[None], h.points)[1][0]
        q = exact_nearest(p[None], v.points)[1][0]
    gap = np.linalg.norm(p-q)
    if gap > max_gap:
        return None
    normals, tangents = {}, {}
    for half in (6, 12, 24):
        a, b = fit_tangent(h, p, half), fit_tangent(v, q, half)
        if a is None or b is None or angle(a, b) < 30:
            return None
        normals[str(half)] = normalize(np.cross(a, b)).tolist()
        tangents[str(half)] = [a.tolist(), b.tolist()]
    return dict(h=h.name, v=v.name, position_ct_xyz=((p+q)/2).tolist(),
                h_position=p.tolist(), v_position=q.tolist(), gap_ct=float(gap),
                crossing_angle=float(angle(*np.asarray(tangents['12']))),
                normals=normals, tangents=tangents,
                span_sensitivity_deg=float(angle(np.array(normals['6']), np.array(normals['24']))))


def select(source, count, rng, out, max_gap):
    print('Selecting crossings:', source['name'], flush=True)
    if source['kind'] == 'paris4':
        cached = out/'paris_fibers.npz'
        if cached.exists():
            points = np.load(cached)
            info = json.loads((out/'paris_fibers.json').read_text())
            fibers = [TracedFiber(f['name'], points[str(i)], arclength(points[str(i)]), f['tag']) for i, f in enumerate(info)]
        else:
            fibers = load_fibers(source['fibers'], grid_scale=source['ct_grid_scale'], spacing=1.)
        hids = [i for i, f in enumerate(fibers) if f.tag.upper() == 'H']
        vids = [i for i, f in enumerate(fibers) if f.tag.upper() == 'V']
        vp = np.concatenate([fibers[i].points[::2] for i in vids])
        vi = np.concatenate([np.full(len(fibers[i].points[::2]), i) for i in vids])
        tree = cKDTree(vp)
        def neighbors(center, hid):
            return sorted(set(vi[tree.query_ball_point(center, 64)].tolist()))
    else:
        fibers = AFVFibers(ROOT/source['path'], grid_scale=1., split='all')
        hids = [i for i, row in enumerate(fibers.catalog) if row[2].upper() == 'H']
        def neighbors(center, hid):
            ids = fibers.nearby_fiber_ids(center, 64, fibers.catalog[hid][0])
            return [fibers.id_to_index[i] for i in ids if fibers.catalog[fibers.id_to_index[i]][2].upper() == 'V']
    print('H fibers:', len(hids), flush=True)
    out = []
    for hid in rng.permutation(hids):
        h = fibers[int(hid)]
        if h.length < 60:
            continue
        center = interp_at(h.points, h.s, [rng.uniform(25, h.length-25)])[0]
        if source['kind'] == 'paris4':
            probes = h.points[::4]
            distance, _ = tree.query(probes)
            possible = np.flatnonzero(distance < max_gap+1)
            if not len(possible):
                continue
            center = probes[rng.choice(possible)]
        for vid in rng.permutation(neighbors(center, int(hid))):
            result = crossing(h, fibers[int(vid)], center, max_gap)
            if result is None:
                continue
            pos = np.array(result['position_ct_xyz'])
            if any(np.linalg.norm(pos-np.array(r['position_ct_xyz'])) < 128 for r in out):
                continue
            out.append(result)
            print(source['name'], len(out), 'gap', round(result['gap_ct'], 3), flush=True)
            break
        if len(out) >= count:
            break
    if len(out) < count:
        raise RuntimeError(f'Only {len(out)} crossings for {source["name"]}; requested {count}')
    return out


def open_remote(url, level, cache):
    return RemoteChunkedArray(url, level, str(cache), 64 << 20)


def normal_at(arrays, manifest, ct_scale, xyz):
    """Interpolate unsigned normals through their outer products, avoiding sign seams."""
    if manifest is None:
        return np.full(3, np.nan), 0., 0.
    factor = 2**manifest['groups']['nx']['scaledown']*manifest['source_to_base']/ct_scale
    coord = np.asarray(xyz)[::-1]/factor
    start = np.floor(coord).astype(int)
    fraction = coord-start
    nx, ny = [arrays[k].read(start, [2]*3) for k in ('nx', 'ny')]
    normals, valid = decode_lasagna_normals_xyz(nx, ny)
    mag = arrays['grad_mag'].read(start, [2]*3)
    valid &= mag > 0
    axes = np.indices((2, 2, 2)).transpose(1, 2, 3, 0)
    weights = np.prod(np.where(axes, fraction, 1-fraction), axis=-1)
    support = float(weights[valid].sum())
    if support < .99:
        return np.full(3, np.nan), support, 0.
    moment = np.einsum('ijk,ijka,ijkb->ab', weights*valid, normals, normals)
    values, vectors = np.linalg.eigh(moment)
    return vectors[:, -1], support, float((values[-1]-values[-2])/max(values[-1], 1e-12))


def tensor_at(block, origin, xyz, threshold=None):
    pos = np.asarray(xyz)[::-1]-origin
    start = np.floor(pos).astype(int)-32
    cube = block[tuple(slice(int(s), int(s+65)) for s in start)]
    if cube.shape != (65, 65, 65):
        raise ValueError('Insufficient CT halo')
    if threshold is not None:
        cube = np.where(cube < threshold, 0, cube)
    values, vectors = np.linalg.eigh(ct_structure_tensor(cube, pos-start))
    gap = float((values[-1]-values[-2])/max(values[-1], 1e-12))
    normal = vectors[:, -1] if values[-1] > 1e-12 and gap > 1e-6 else np.full(3, np.nan)
    return normal, gap, float(values[-1])


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0)
    return dict(n=len(values), median=float(np.median(values)), p90=float(np.percentile(values, 90)),
                mean=float(values.mean()), under10=float(np.mean(values < 10)), under20=float(np.mean(values < 20)))


def add_threshold_comparison(saved, arrays, manifest, scale):
    row = json.loads(str(saved['row_json']))
    if 'tensor60_normal' in row:
        return saved
    block, origin = saved['ct'], saved['origin_zyx']
    pos = np.array(row['position_ct_xyz'])
    normal, gap, energy = tensor_at(block, origin, pos, threshold=60)
    row.update(tensor60_normal=normal.tolist(), tensor60_gap=gap, tensor60_energy=energy,
        tensor60_lasagna=float(angle(normal, row['lasagna_normal'])),
        fiber_tensor60=float(angle(normal, row['normals']['12'])),
        fiber24_tensor60=float(angle(normal, row['normals']['24'])),
        tensor60_original=float(angle(normal, row['tensor_normal'])),
        fiber6_tensor=float(angle(row['normals']['6'], row['tensor_normal'])),
        fiber24_tensor=float(angle(row['normals']['24'], row['tensor_normal'])),
        masked_fraction=float(np.mean(block < 60)))
    grid = []
    for p in saved['grid_positions']:
        n, g, e = tensor_at(block, origin, p, threshold=60)
        las, support, _ = normal_at(arrays, manifest, scale, p)
        grid.append([float(angle(n, las)), g, e, support])
    print('Threshold 60:', row['index'], 'Lasagna', round(row['tensor_lasagna'], 2), '->', round(row['tensor60_lasagna'], 2),
          'H×V', round(row['fiber_tensor'], 2), '->', round(row['fiber_tensor60'], 2), flush=True)
    return dict(saved, row_json=json.dumps(row), grid60_angles=np.array(grid))


def paired_change(original, thresholded):
    a, b = np.asarray(original), np.asarray(thresholded)
    valid = np.isfinite(a) & np.isfinite(b)
    return dict(n=int(valid.sum()), improved_fraction=float(np.mean(b[valid] < a[valid])) if valid.any() else None,
                median_change=float(np.median(b[valid]-a[valid])) if valid.any() else None,
                original_invalid=int(np.sum(~np.isfinite(a))), thresholded_invalid=int(np.sum(~np.isfinite(b))))


def evaluate(source, selections, out):
    name = source['name']
    dest = out/name
    dest.mkdir(exist_ok=True)
    cache = out/'cache'
    url = BUCKET+MANIFESTS[name] if MANIFESTS[name] else None
    manifest = None
    arrays = {}
    if url:
        mf = dest/'lasagna.json'
        if not mf.exists():
            mf.write_bytes(urlopen(url, timeout=30).read())
        manifest = json.loads(mf.read_text())
        for channel in ('nx', 'ny', 'grad_mag'):
            g = manifest['groups'][channel]
            root, level = g['zarr'].rsplit('/', 1)
            arrays[channel] = open_remote(url.rsplit('/', 1)[0]+'/'+root, int(level), cache)
        assert len({tuple(arrays[k].shape) for k in arrays}) == 1
        assert len({manifest['groups'][k]['scaledown'] for k in arrays}) == 1
    scale = source['ct_grid_scale']
    if source['kind'] == 'paris4':
        ct = ChunkedArray(Path(source['ct'])/'0', 128 << 20)
        public = open_remote(BUCKET+'PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr', 2, cache)
        checks = []
        for r in selections[:3]:
            start = np.round(np.array(r['position_ct_xyz'])[::-1]).astype(int)-16
            a, b = ct.read(start, [32]*3), public.read(start, [32]*3)
            correlation = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
            checks.append(dict(start=start.tolist(), correlation=correlation,
                               exact_fraction=float(np.mean(a == b)), mean_abs_difference=float(np.abs(a.astype(float)-b).mean())))
            if not correlation > .98:
                raise RuntimeError(f'Paris4 coordinate/intensity check failed: {checks[-1]}')
        (dest/'alignment.json').write_text(json.dumps(checks, indent=2))
        print('Paris4 registration:', checks, flush=True)
    else:
        ct = open_remote(source['ct'], source['ct_level'], cache)
    rows, grids, grids60 = [], [], []
    for i, r in enumerate(selections):
        fn = dest/f'chunk_{i:03d}.npz'
        if fn.exists():
            with np.load(fn) as archive:
                saved = {k: archive[k] for k in archive.files}
            augmented = add_threshold_comparison(saved, arrays, manifest, scale)
            if augmented is not saved:
                np.savez_compressed(fn, **augmented)
            saved = augmented
            rows.append(json.loads(str(saved['row_json'])))
            grids.extend(saved['grid_angles'].tolist())
            grids60.extend(saved['grid60_angles'].tolist())
            continue
        pos = np.array(r['position_ct_xyz'])
        origin = np.floor(pos[::-1]).astype(int)-50
        block = ct.read(origin, [101]*3)
        st, gap, energy = tensor_at(block, origin, pos)
        las, support, coherence = normal_at(arrays, manifest, scale, pos)
        fib = np.array(r['normals']['12'])
        grid_pos = pos+np.stack(np.meshgrid(*[[-16, 0, 16]]*3, indexing='ij'), -1).reshape(-1, 3)
        angles = []
        for p in grid_pos:
            a, g, e = tensor_at(block, origin, p)
            b, valid, _ = normal_at(arrays, manifest, scale, p)
            angles.append([float(angle(a, b)), g, e, valid])
        row = dict(r, index=i, tensor_normal=st.tolist(), lasagna_normal=las.tolist(),
                   tensor_gap=gap, tensor_energy=energy, lasagna_support=support, lasagna_coherence=coherence,
                   tensor_lasagna=float(angle(st, las)), fiber_lasagna=float(angle(fib, las)),
                   fiber_tensor=float(angle(fib, st)),
                   fiber6_lasagna=float(angle(np.array(r['normals']['6']), las)),
                   fiber24_lasagna=float(angle(np.array(r['normals']['24']), las)))
        saved = add_threshold_comparison(dict(ct=block, origin_zyx=origin, row_json=json.dumps(row),
                            grid_positions=grid_pos, grid_angles=np.array(angles)), arrays, manifest, scale)
        np.savez_compressed(fn, **saved)
        row = json.loads(saved['row_json'])
        rows.append(row)
        grids.extend(angles)
        grids60.extend(saved['grid60_angles'].tolist())
        print(name, i+1, 'ST/Las, fiber/Las, fiber/ST:',
              [round(row[k], 2) for k in ('tensor_lasagna', 'fiber_lasagna', 'fiber_tensor')], flush=True)
    columns = ['index', 'h', 'v', 'gap_ct', 'crossing_angle', 'span_sensitivity_deg', 'tensor_gap',
               'lasagna_support', 'lasagna_coherence', 'tensor_lasagna', 'fiber_lasagna', 'fiber_tensor',
               'fiber6_lasagna', 'fiber24_lasagna', 'fiber6_tensor', 'fiber24_tensor',
               'tensor60_lasagna', 'fiber_tensor60', 'fiber24_tensor60', 'tensor60_original', 'masked_fraction']
    with (dest/'crossings.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    grid = np.asarray(grids)
    grid60 = np.asarray(grids60)
    summary = dict(manifest=url, ct=source['ct'], ct_grid_scale=scale,
        crossings={k: stats([r[k] for r in rows]) for k in ('tensor_lasagna', 'fiber_lasagna', 'fiber_tensor')},
        stable_crossings={k: stats([r[k] for r in rows if r['span_sensitivity_deg'] < 10 and r['gap_ct'] < 6])
                          for k in ('tensor_lasagna', 'fiber_lasagna', 'fiber_tensor')},
        tangent_span={k: stats([r[k] for r in rows]) for k in ('fiber6_lasagna', 'fiber24_lasagna', 'fiber6_tensor', 'fiber24_tensor', 'span_sensitivity_deg')},
        threshold60={k: stats([r[k] for r in rows]) for k in ('tensor60_lasagna', 'fiber_tensor60', 'fiber24_tensor60', 'tensor60_original')},
        threshold60_change={key: paired_change([r[old] for r in rows], [r[new] for r in rows])
                            for key, old, new in [('lasagna', 'tensor_lasagna', 'tensor60_lasagna'),
                                                ('fibers', 'fiber_tensor', 'fiber_tensor60')]},
        grid_tensor60_lasagna=stats(grid60[:, 0]),
        grid_threshold60_change=paired_change(grid[:, 0], grid60[:, 0]),
        grid_tensor_lasagna=stats(grid[:, 0]),
        grid_reliable_tensor_lasagna=stats(grid[(grid[:, 1] >= .2) & (grid[:, 2] > 1e-12), 0]),
        invalid_grid_points=int(np.sum(~np.isfinite(grid[:, 0]))),
        gap_ct=stats([r['gap_ct'] for r in rows]), crossing_angle=stats([r['crossing_angle'] for r in rows]))
    (dest/'summary.json').write_text(json.dumps(summary, indent=2))
    return rows, grid, summary


def plots(results, out):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fiber_follow_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(results), figsize=(6*len(results), 4), squeeze=False)
    for ax, (name, (rows, grid, _)) in zip(axs[0], results.items()):
        for key, label in [('tensor_lasagna', 'Tensor vs Lasagna'), ('fiber_lasagna', 'H×V vs Lasagna'),
                           ('fiber_tensor', 'H×V vs tensor')]:
            values = np.sort([r[key] for r in rows if np.isfinite(r[key])])
            if not len(values):
                continue
            ax.plot(values, np.arange(1, len(values)+1)/len(values), label=label)
        ax.set(xlim=(0, 90), ylim=(0, 1), xlabel='Unsigned normal difference (degrees)', ylabel='Fraction of crossings', title=name)
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(2, len(results), figsize=(5*len(results), 8), squeeze=False)
    for column, (name, (rows, _, _)) in enumerate(results.items()):
        for ax, old, new, label in [(axs[0, column], 'tensor_lasagna', 'tensor60_lasagna', 'Lasagna'),
                                    (axs[1, column], 'fiber_tensor', 'fiber_tensor60', 'H×V')]:
            a, b = np.array([[r[old], r[new]] for r in rows]).T
            good = np.isfinite(a) & np.isfinite(b)
            ax.plot([0, 90], [0, 90], color='gray', ls='--', lw=1)
            ax.scatter(a[good], b[good], s=24, alpha=.8)
            ax.set(xlim=(0, 45), ylim=(0, 45), aspect='equal', xlabel='Original tensor error (degrees)',
                   ylabel='Threshold-60 tensor error (degrees)', title=f'{name}: vs {label}')
            ax.grid(alpha=.2)
            if not good.any():
                ax.text(.5, .5, 'No public Lasagna normals', transform=ax.transAxes, ha='center')
    fig.suptitle('CT < 60 → 0 before derivatives; points below the diagonal improve agreement')
    fig.tight_layout()
    fig.savefig(out/'threshold60.png', dpi=160)
    plt.close(fig)
    for name, (rows, _, _) in results.items():
        ordered = sorted(rows, key=lambda r: r['tensor_lasagna'] if np.isfinite(r['tensor_lasagna']) else r['fiber_tensor'])
        choices = [ordered[j] for j in np.linspace(0, len(rows)-1, 4).astype(int)]
        fig, axs = plt.subplots(4, 3, figsize=(12, 14))
        for axes, r in zip(axs, choices):
            saved = np.load(out/name/f'chunk_{r["index"]:03d}.npz')
            block, origin = saved['ct'], saved['origin_zyx']
            pos = np.array(r['position_ct_xyz'])-origin[::-1]
            for ax, (a, b, sliced), label in zip(axes, [(0, 1, 2), (0, 2, 1), (1, 2, 0)], ['XY', 'XZ', 'YZ']):
                image = np.take(block, int(round(pos[sliced])), axis=2-sliced)
                lo, hi = np.percentile(block, [2, 98])
                ax.imshow(image, cmap='gray', origin='lower', vmin=lo, vmax=hi)
                for key, color, text in [('tensor_normal', '#00ffff', 'tensor'), ('tensor60_normal', '#ff5555', 'tensor ≥60'), ('lasagna_normal', '#ff55ff', 'Lasagna'),
                                         ('normals', '#ffff00', 'H×V')]:
                    n = np.array(r[key]['12'] if key == 'normals' else r[key])
                    if not np.isfinite(n).all():
                        continue
                    ax.plot(pos[a]+np.array([-15, 15])*n[a], pos[b]+np.array([-15, 15])*n[b], color=color, lw=2, label=text)
                for point, tangent, color in zip([r['h_position'], r['v_position']], r['tangents']['12'], ['lime', 'orange']):
                    point = np.array(point)-origin[::-1]
                    tangent = np.array(tangent)
                    ax.plot(point[a]+np.array([-12, 12])*tangent[a], point[b]+np.array([-12, 12])*tangent[b], color=color, ls='--', lw=1)
                text = (f'ST/L {r["tensor_lasagna"]:.1f}°, H×V/L {r["fiber_lasagna"]:.1f}°'
                        if np.isfinite(r['tensor_lasagna']) else f'H×V/ST {r["fiber_tensor"]:.1f}°')
                ax.set_title(f'{label}, #{r["index"]}: {text}')
                ax.set_xlim(15, 85)
                ax.set_ylim(15, 85)
            axes[0].legend(fontsize=8)
        fig.suptitle(name+' — crossings across tensor/Lasagna agreement range; dashed H/V tangents')
        fig.tight_layout()
        fig.savefig(out/name/'examples.png', dpi=140)
        plt.close(fig)


def write_report(results, out):
    def metric(value):
        return f'{value["median"]:.1f}° / {value["p90"]:.1f}°' if value['n'] else 'Unavailable'
    table, mask_table, span_table = [], [], []
    for name, (_, _, s) in results.items():
        table.append('<tr><th>'+html.escape(name)+'</th>'+''.join('<td>'+metric(s['crossings'][k])+'</td>'
            for k in ('tensor_lasagna', 'fiber_lasagna', 'fiber_tensor'))+'</tr>')
        for ref, old, new in [('Lasagna', 'tensor_lasagna', 'tensor60_lasagna'), ('H×V', 'fiber_tensor', 'fiber_tensor60')]:
            change = s['threshold60_change']['lasagna' if ref == 'Lasagna' else 'fibers']
            fraction = f'{change["improved_fraction"]:.0%}' if change['n'] else '—'
            mask_table.append(f'<tr><th>{name}</th><td>{ref}</td><td>{metric(s["crossings"][old])}</td>'
                              f'<td>{metric(s["threshold60"][new])}</td><td>{fraction}</td></tr>')
        span_table.append(f'<tr><th>{name}</th><td>{metric(s["tangent_span"]["fiber24_lasagna"])}</td>'
                          f'<td>{metric(s["tangent_span"]["fiber24_tensor"])}</td></tr>')
    def img(path):
        return '<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'
    examples = ''.join(f'<details><summary>{html.escape(name)} CT examples</summary>{img(out/name/"examples.png")}</details>' for name in results)
    report = f'''<!doctype html><html><meta charset="utf-8"><title>Sheet normal comparison</title>
<style>body{{font:16px/1.5 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#17202a}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}summary{{cursor:pointer;font-weight:600;padding:16px 0}}code{{background:#eee;padding:2px 5px}}</style>
<h1>Sheet normals: CT tensor, Lasagna, and crossing fibers</h1>
<p>32 spatially separated H/V close approaches per volume, 96 total. Every location uses a saved 101³ CT block.
For Paris4 and PHerc1447, tensor–Lasagna agreement also uses 27 nearby points per block (864 per volume).
Numbers below are median / 90th-percentile unsigned angular differences; lower means closer agreement.</p>
<table><tr><th>Source</th><th>Tensor vs Lasagna</th><th>H×V vs Lasagna</th><th>H×V vs tensor</th></tr>{''.join(table)}</table>
{img(out/'comparison.png')}
<h2>Zeroing CT values below 60</h2>
<p>The threshold operates on raw uint8 CT, before the Gaussian derivatives and tensor integration. Values equal to 60 remain unchanged.
All comparisons use exactly the same locations and reference normals.</p>
<table><tr><th>Source</th><th>Reference</th><th>Original tensor</th><th>Threshold 60</th><th>Locations improved</th></tr>{''.join(mask_table)}</table>
{img(out/'threshold60.png')}
<h2>Tangent-fit sensitivity</h2><p>The primary H×V estimate fits each tangent over ±12 CT voxels. These are the results for ±24 voxels (48 total):</p>
<table><tr><th>Source</th><th>H×V, wider fit vs Lasagna</th><th>H×V, wider fit vs original tensor</th></tr>{''.join(span_table)}</table>
<h2>Method and limits</h2><ul>
<li>Locations were selected from fiber geometry without examining normal agreement: H/V separation ≤8 CT voxels, acute tangent angle ≥30°, enough support for ±24-voxel fits, and centers ≥128 CT voxels apart. Saved selection.json records exact coordinates and fiber identities.</li>
<li>Fiber normals are normalized cross products of centered, equal-arclength least-squares tangents. Primary fits span 24 CT voxels; 12- and 48-voxel spans test sensitivity. Comparisons occur at the midpoint of each H/V close approach.</li>
<li>The original tensor calls the follower's ct_structure_tensor unchanged: derivative sigma 1, integration sigma 4, 65³ CT context. Its largest-eigenvalue eigenvector is the sheet normal. The threshold variant changes only the CT intensities supplied to that function.</li>
<li>Lasagna nx/ny use the repository's hemisphere decoder. Trilinear interpolation averages unsigned normal outer products, then takes the principal eigenvector. All contributing support must have grad_mag &gt; 0. Angles are acos(|a·b|), so signs do not matter.</li>
<li>Paris4 local/public CT alignment was checked on three 32³ blocks (correlation &gt;0.9998). Normal spacing is four selected-CT voxels for both Paris4 and PHerc1447. PHerc0175A has no matching public Lasagna product, so it is compared to CT tensors only.</li>
<li>These are nearby H/V fibers, not guaranteed exact intersections or verified same-sheet pairs. Agreement is not ground-truth accuracy. AFV fibers are automated annotations; their agreement is not proof of independent supervision. Adjacent grid samples are correlated, and 32 regions per volume are exploratory.</li>
</ul><h2>Visual checks</h2><p>Cyan: original tensor; red: threshold-60 tensor; magenta: Lasagna; yellow: H×V. Dashed green/orange lines show fitted H/V tangents. Normal lines are projected into each displayed CT plane.</p>{examples}
<h2>Reproduce</h2><p>From fiber_follow/, using the existing vesuvius environment:</p>
<pre>../../../../.venv/bin/python evaluation/compare_sheet_normals.py --out output/sheet_normals_20261002</pre>
<p>The saved selection, per-source crossings.csv and summary.json, and chunk_*.npz files contain the measurements, CT blocks, coordinates, and normals. Training code and datasets were not changed.</p></html>'''
    (out/'report.html').write_text(report)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--count', type=int, default=32)
    ap.add_argument('--seed', type=int, default=7349)
    ap.add_argument('--max-gap', type=float, default=8.)
    ap.add_argument('--select-only', action='store_true')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT/'configs/mixed_ct_datasets_paris50.json').read_text())
    # Config paths are relative to configs/, not this script.
    for s in cfg['sources']:
        if 'path' in s:
            s['path'] = str((ROOT/'configs'/s['path']).resolve())
    sources = [s for s in cfg['sources'] if s['name'] in MANIFESTS]
    selection = out/'selection.json'
    if not selection.exists():
        rng = np.random.default_rng(args.seed)
        selected = {s['name']: select(s, args.count, rng, out, args.max_gap) for s in sources}
        selection.write_text(json.dumps(dict(seed=args.seed, count=args.count, max_gap=args.max_gap, sources=selected), indent=2))
    selected = json.loads(selection.read_text())
    for i, source in enumerate(sources):
        if source['name'] not in selected['sources']:
            selected['sources'][source['name']] = select(source, args.count, np.random.default_rng(args.seed+i), out, args.max_gap)
    selection.write_text(json.dumps(selected, indent=2))
    if args.select_only:
        return
    results = {s['name']: evaluate(s, selected['sources'][s['name']], out) for s in sources}
    plots(results, out)
    write_report(results, out)
    summary = {name: value[2] for name, value in results.items()}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
