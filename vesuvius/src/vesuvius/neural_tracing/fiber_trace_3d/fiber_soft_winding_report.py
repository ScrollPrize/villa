"""Read-only H→V constraint match audit of a saved continuous soft layout."""
import json
from pathlib import Path

import numpy as np
import torch
from lasagna.inference_provenance import atomic_write, sha256_file
from . import fiber_soft_winding as soft
from . import fiber_uv_layout as uv


def stats(values):
    values = np.asarray(values)
    if not len(values): return None
    absolute = abs(values)
    return dict(mean=float(values.mean()), l1=float(absolute.mean()),
                median_abs=float(np.median(absolute)), p95_abs=float(np.quantile(absolute, .95)),
                max_abs=float(absolute.max()))


def bin_mask(values, index):
    """20 half-open decimal bins; include +1 only in the final bin."""
    lo, hi = (index - 10) / 10, (index - 9) / 10
    return (values >= lo) & ((values <= hi) if index == 19 else (values < hi))


def run(settings):
    root = Path(settings['solve_root']); output = Path(settings['output_root'])
    cfg = json.loads((root / 'config.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    if summary['model'] != 'continuous_soft_winding_v1': raise ValueError('Not a soft solve')
    with np.load(cfg['tensor_cache']) as src: d = {k: src[k] for k in src.files}
    with np.load(root / 'solution.npz') as src: s = {k: src[k] for k in src.files}
    for key in ('source_id', 'start', 'end', 'family'):
        if not np.array_equal(s[key], d[key]): raise ValueError('Changed piece identity: ' + key)
    if cfg.get('tensor_sha256') and sha256_file(cfg['tensor_cache']) != cfg['tensor_sha256']:
        raise ValueError('Changed input tensor cache')
    state = torch.as_tensor(s['state'], dtype=torch.float64)
    widths = torch.as_tensor(s['widths'], dtype=torch.float64)
    i, j = d['i'], d['j']; count = len(i)
    ta = torch.as_tensor(d['family'][i]); tb = torch.as_tensor(d['family'][j])
    da = torch.as_tensor(d['ds_a'], dtype=torch.float64); db = torch.as_tensor(d['ds_b'], dtype=torch.float64)
    with torch.no_grad():
        qa, ua, va = soft.points(state[i], ta, da, widths)
        qb, ub, vb = soft.points(state[j], tb, db, widths)
        orient = torch.where(ta == 0, 1., -1.)
        uhv = orient * (ub - ua)
        phase = soft.circular(uhv).numpy()
        aligned = (orient * (state[j, 0] - state[i, 0]) + torch.floor(uhv + .5)).numpy()
        qdelta = (orient * (qb - qa)).numpy()
        vertical = (orient * (vb - va)).numpy()
        wind_error = soft.pair_errors(state[i], state[j], ta, tb, da, db, widths,
            torch.as_tensor(d['sign']), torch.as_tensor(d['distance']), torch.as_tensor(d['parallel']),
            cfg.get('perpendicular_offset', .5))[0].numpy()
        probability = state[:, 3].sigmoid().numpy()
    selected = (d['family'][i] != d['family'][j]) & (probability[i] >= .5) & (probability[j] >= .5)
    measured = np.where(d['family'][i] == 0, 1., -1.) * d['sign'] * d['distance']
    gate = uv.constraint_class_weights(d['parallel'], dict(cfg, constraint_class_balance=True)) * probability[i] * probability[j]
    losses = dict(u=cfg.get('phase_weight', 1.) * gate * phase**2,
                  v=cfg.get('metric_weight', .02) * gate * (vertical / cfg.get('v_pair_scale', 1000.))**2,
                  winding=cfg.get('distance_weight', .25) * gate * (wind_error / cfg.get('distance_sigma', .5))**2)

    def summarize(mask):
        n = int(mask.sum())
        return dict(count=n, classified_parallel=int((mask & d['parallel']).sum()),
            u_windings=stats(phase[mask]), v_base_voxels=stats(vertical[mask]),
            full_u_separation_windings=stats(qdelta[mask]),
            winding_evidence_residual=stats(wind_error[mask]),
            losses={key: dict(conditional_mean=float(x[mask].mean()) if n else None,
                              full_objective_contribution=float(x[mask].sum() / count)) for key, x in losses.items()})

    tables = {}
    for name, values, eligible in [('fitted_aligned', aligned, selected),
        ('fitted_continuous_q', qdelta, selected), ('measured_lasagna', measured, selected & (d['sign'] != 0))]:
        tables[name] = dict(total=summarize(eligible), below=summarize(eligible & (values < -1)),
            in_range=summarize(eligible & (values >= -1) & (values <= 1)),
            above=summarize(eligible & (values > 1)), bins=[dict(lo=(b-10)/10, hi=(b-9)/10,
                **summarize(eligible & bin_mask(values, b))) for b in range(20)])
    result = dict(solve_root=str(root), total_events=count, enabled_hv=int(selected.sum()),
        unknown_order_enabled_hv=int((selected & (d['sign'] == 0)).sum()),
        config=cfg, tables=tables, hashes={str(p): sha256_file(p) for p in
            (root/'solution.npz', root/'config.json', Path(cfg['tensor_cache']))})
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output/'results.json', result, durable=False)
    lines = ['# Soft-winding H→V match audit', '',
        'Enabled endpoints only (P≥0.5), opposite fixed H/V families. U is circular phase',
        'mismatch in windings; V is signed V−H mismatch in base voxels. L1=mean absolute',
        'error; max=maximum absolute error. Bins are half-open except the last includes +1.',
        'Fitted aligned bins omit the H/V layer calibration; measured Lasagna bins include',
        'physical layer spacing. Unknown normal order is excluded only from measured bins.', '',
        f'Enabled H/V events: {result["enabled_hv"]}; unknown measured order: {result["unknown_order_enabled_hv"]}.', '']
    def fmt(x): return '—' if x is None else f'{x:.5f}'
    for name, table in tables.items():
        lines += [f'## {name}', '', f'Below −1: {table["below"]["count"]}; above +1: {table["above"]["count"]}.', '',
            '| Winding bin | Count | U mean | U L1 | U max | V mean | V L1 | V max |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in table['bins']:
            vals = [row[key].get(stat) if row[key] else None for key in ('u_windings', 'v_base_voxels') for stat in ('mean','l1','max_abs')]
            lines.append(f'| {row["lo"]:+.1f} to {row["hi"]:+.1f} | {row["count"]} | ' + ' | '.join(map(fmt,vals)) + ' |')
        lines += ['', 'Actual P/class/coefficient-weighted loss contributions to the full event-mean objective:', '',
            '| Bin | U loss | V loss | Winding evidence loss |', '|---|---:|---:|---:|']
        for row in table['bins']:
            lines.append(f'| {row["lo"]:+.1f} to {row["hi"]:+.1f} | ' + ' | '.join(
                f"{row['losses'][k]['full_objective_contribution']:.6g}" for k in ('u','v','winding')) + ' |')
        lines += ['']
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output), enabled_hv=result['enabled_hv'],
                         fitted_total=tables['fitted_aligned']['total'])), flush=True)
    return result
