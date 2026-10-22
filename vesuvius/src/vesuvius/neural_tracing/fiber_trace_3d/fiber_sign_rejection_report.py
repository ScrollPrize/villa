"""Retention-aware comparison of saved sign/rejection sweeps; no optimization."""
import json
from pathlib import Path
import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from .fiber_density_sign_report import gt_ordering


def cohort_metrics(events, order, mask):
    residual = np.array([e['residual'] for e in events])[mask]
    n = len(residual)
    return dict(events=n, signs=int((order[mask] < 0).sum()),
                l1=float(np.abs(residual).mean()) if n else None,
                correct_percent=float(100 * (np.floor(residual + .5) == 0).mean()) if n else None)


def markdown_report(rows):
    """Keep signs, rejection and fit together; never rank by L1 alone."""
    def internal(r, kind, cohort='both_enabled'):
        return next(x for x in r['internal'] if x['kind'] == kind and x['cohort'] == cohort)

    def signs(n, total):
        return f'{n}/{total} ({100*n/total:.3f}%)' if total else '0/0 (n/a)'

    text = ['# Sign weight versus disabling cost', '',
            'Fresh radial initialization; seed0; 1000updates; cavity cap8; sign threshold0.',
            'Only sign weight and disabling cost differ. Input/init arrays and solver hashes checked.',
            'All sign counts are MAP ordering violations, not Gaussian-tail probability.',
            'Each run uses its saved global GT offset; common-cohort scores do not refit it.',
            'The dataset is GT-curated, not an independent held-out set. Objectives across weights are not directly comparable.', '']
    for title, selected in [('Matched-factor grid (0.1x/1x/10x)',
                             [r for r in rows if r['weight'] in (3,30,300) and r['rejection'] in (2,20,200)]),
                            ('Original grid', [r for r in rows if r['weight'] in (30,100,300) and r['rejection'] in (2,10,20)])]:
        selected = sorted(selected, key=lambda r: (r['weight'], r['rejection']))
        text += ['## '+title, '',
                 '| Sign | Disable cost | Disabled fibers | Length retained % | Internal parallel signs | Internal perp signs | GT parallel signs | GT perp signs |',
                 '|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in selected:
            p,q = internal(r,'parallel'), internal(r,'perpendicular')
            gp,gq = r['cohorts']['parallel']['enabled'],r['cohorts']['perpendicular']['enabled']
            text.append(f"|{r['weight']}|{r['rejection']}|{r['retention']['total']-r['retention']['fibers']}|{r['retention']['length_percent']:.2f}|{signs(p['violations'],p['signed_events'])}|{signs(q['violations'],q['signed_events'])}|{signs(gp['signs'],gp['events'])}|{signs(gq['signs'],gq['events'])}|")
        text += ['', '| Sign | Disable cost | GT parallel coverage % | GT perp coverage % | Parallel L1 | Perp L1 | Parallel correct % | Perp correct % | Objective | Width/GT | Offset | Solve seconds |',
                 '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in selected:
            p,q = r['cohorts']['parallel']['enabled'],r['cohorts']['perpendicular']['enabled']
            coverage=r['score']['coverage_by_type']
            def number(v):return 'n/a' if v is None else f'{v:.6f}'
            text.append('|'+ '|'.join(map(str,[r['weight'],r['rejection'],number(coverage['parallel']['percent']),number(coverage['perpendicular']['percent']),number(p['l1']),number(q['l1']),number(p['correct_percent']),number(q['correct_percent']),number(r['objective']),number(r['width_factor']),r['offset'],number(r['seconds'])]))+'|')
        text += ['']
    text += ['## All-fiber versus enabled signs', '',
             'All-fiber statistics include discarded fibers, whose positions are weakly constrained after rejection; they are not retained-solution errors.', '',
             '| Sign | Disable cost | Internal all / enabled violations | GT all / enabled violations |',
             '|---:|---:|---:|---:|']
    for r in rows:
        all_gt=sum(r['cohorts'][k]['all']['signs'] for k in ('parallel','perpendicular'))
        enabled_gt=sum(r['cohorts'][k]['enabled']['signs'] for k in ('parallel','perpendicular'))
        text.append(f"|{r['weight']}|{r['rejection']}|{internal(r,'all','all')['violations']} / {internal(r,'all')['violations']}|{all_gt} / {enabled_gt}|")
    text += ['', '## Shared retained GT comparison', '',
             'Baseline and trial evaluated on exactly the same retained GT events, using their respective saved offsets.', '',
             '| Sign | Disable cost | Type | Events | Baseline signs | Trial signs | Baseline L1 | Trial L1 |',
             '|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        for kind in ('parallel','perpendicular'):
            a,b=r['cohorts'][kind]['baseline_common'],r['cohorts'][kind]['trial_common']
            text.append('|'+ '|'.join(map(str,[r['weight'],r['rejection'],kind,a['events'],a['signs'],b['signs'],a['l1'],b['l1']]))+'|')
    return '\n'.join(text)+'\n'


def compare(roots, baseline, audit_root, output):
    baseline = Path(baseline)
    be = json.loads((baseline / 'gt_events.json').read_text())
    bc = json.loads((baseline / 'solve/config.json').read_text())
    bo = gt_ordering(be, bc)
    identity = lambda events: [{k: v for k, v in e.items() if k not in ('residual', 'enabled')} for e in events]
    rows = []
    for root in map(Path, roots):
        cfg = json.loads((root / 'solve/config.json').read_text())
        result = json.loads((root / 'results.json').read_text())
        if result['solution_sha256'] != sha256_file(root / 'solve/solution.npz'):
            raise ValueError('Result solution changed')
        events = json.loads((root / 'gt_events.json').read_text())
        if identity(events) != identity(be):
            raise ValueError('GT event identities/order differ')
        for filename in ('init/solution.npz', 'init/samples.npz', 'tensors.npz'):
            with np.load(baseline / filename) as a, np.load(root / filename) as b:
                if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                    raise ValueError('Initialization/input arrays differ: ' + filename)
        allowed = {'sign_violation_weight', 'rejection_cost', 'devices', 'gpu_manager',
                   'tensor_cache', 'output_root', 'initial_solution', 'soft_samples', 'code_provenance'}
        differing = [k for k in set(cfg) | set(bc) if cfg.get(k) != bc.get(k)
                     and k not in allowed and not k.endswith('_sha256') and not k.startswith('executed_')]
        if differing:
            raise ValueError('Unexpected config difference: ' + str(differing))
        for key in ('shared_solver_sha256', 'cavity_solver_sha256'):
            if cfg[key] != bc[key]:
                raise ValueError('Solver implementation changed: ' + key)
        if cfg['code_provenance']['optimizer_sha256'] != bc['code_provenance']['optimizer_sha256']:
            raise ValueError('Optimizer implementation changed')
        with np.load(root / 'solve/solution.npz') as s:
            keep = s['state'][:, 3] >= 0
            lengths = s['end'] - s['start']
            retention = dict(fibers=int(keep.sum()), total=len(keep),
                             length_percent=float(100 * lengths[keep].sum() / lengths.sum()))
        order = gt_ordering(events, cfg)
        enabled = np.array([e['enabled'] for e in events])
        common = enabled & np.array([e['enabled'] for e in be])
        par = np.array([e['parallel'] for e in events])
        cohorts = {}
        for kind, select in [('parallel', par), ('perpendicular', ~par)]:
            cohorts[kind] = {name: cohort_metrics(events, order, select & mask)
                            for name, mask in [('all', np.ones(len(events), bool)), ('enabled', enabled)]}
            cohorts[kind]['baseline_common'] = cohort_metrics(be, bo, select & common)
            cohorts[kind]['trial_common'] = cohort_metrics(events, order, select & common)
        audit_path = Path(audit_root) / root.name.removesuffix('_v1') / 'summary.json'
        if root == baseline:
            audit_path = baseline.parent / 'sign_tuning_audit_v1/w30_t00/summary.json'
        internal = json.loads(audit_path.read_text())
        for path in (root / 'solve/solution.npz', root / 'solve/config.json', Path(cfg['tensor_cache'])):
            if internal['hashes'][str(path)] != sha256_file(path):
                raise ValueError('Stale internal sign audit: ' + str(path))
        rows.append(dict(root=str(root), weight=cfg['sign_violation_weight'], rejection=cfg['rejection_cost'],
                         retention=retention, cohorts=cohorts, internal=internal['rows'],
                         offset=result['score']['gt_global_offset'], objective=result['final']['loss'],
                         width_factor=result['score']['gt_solve_widths']['mean_normalized_factor'],
                         score=result['score'], seconds=result['final']['seconds']))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output / 'comparison.json', rows, durable=False)
    (output / 'report.md').write_text(markdown_report(rows))
    return rows
