"""Saved-data-only integration matrix and full tracing comparisons."""
import json
import math
from collections import Counter
from pathlib import Path
from lasagna.inference_provenance import atomic_write, sha256_file
from .replay_metrics import complete_report, split_report


def validate_coverage(rows):
    if not rows:raise ValueError('no completed comparisons')
    first=rows[0]['metrics']
    for row in rows:
        m=row['metrics']
        for key in ('reference_selection_policy','metric_definition','mean_definition'):
            if m[key]!=first[key]:raise ValueError('incompatible comparison '+key)
        if m['scrolls'].keys()!=first['scrolls'].keys():raise ValueError('different scroll coverage')
        for name,stats in [('all',m['global_result']),*m['scrolls'].items()]:
            ref=first['global_result'] if name=='all' else first['scrolls'][name]
            if stats['directions']!=ref['directions'] or not math.isclose(stats['directed_reference_mm'],ref['directed_reference_mm'],rel_tol=1e-10):
                raise ValueError('different reference coverage: '+row['name']+'/'+name)


def report(settings):
    root=Path(settings['integration_output'])
    data=json.loads((Path(settings['output'])/'dataset.json').read_text())
    matrix=json.loads((root/'matrix.json').read_text())
    parity=json.loads((root/'parity.json').read_text())
    base=Path(settings['evaluation']).parents[1]
    runs=[('Classic','classic_thickness40_grace160_20260914_v2'),
          ('u11800 embeddings off','eligible_ema999_final_vis_20260914_v1'),
          ('Original EMA .25/512','mixed_current_ema_only_20260915_v1'),
          ('Previous fitted EMA','fixed_path_ema_linear_20260915_v1')]
    runs += [(family,'embedding_integration_'+family+'_20260915_v1') for family in matrix['winners']]
    result=dict(matrix=matrix,parity=parity,rows=[],pending=[],split=data['split'])
    for label,name in runs:
        path=base/name/'full/summary.json'
        if not path.exists():result['pending'].append(label);continue
        summary=json.loads(path.read_text())
        if not summary.get('completed'):result['pending'].append(label);continue
        inference=path.with_name('inference_summary.json')
        result['rows'].append(dict(name=label,path=str(path),summary_sha256=sha256_file(path),
            metrics=complete_report(summary),splits=split_report(summary,data['split']),
            native_wall_seconds=summary.get('wall_seconds'),
            coordinator_wall_seconds=json.loads(inference.read_text()).get('wall_seconds') if inference.exists() else None,
            effective_trace=summary.get('effective_trace')))
    validate_coverage(result['rows'])
    rejected=Counter()
    for case in data['cases']:rejected.update(case['rejected'])
    result['excluded']=dict(rejected)
    result['accepted_windows']=sum(len(c['windows']) for c in data['cases'])
    lines=['# Embedding integration experiments — 2026-09-15','',
        'Frozen u11800 presence/direction + paired current u42000 embeddings. No anchors. No default promotion.',
        '50 fibers / five scrolls / 180 eligible run-directions. Split:30 fit,9 selection,11 calibration-heldout fibers.',
        'Mean = combined directed reference distance / (counted restarts +1). Median/P05/P95 include terminal episodes. All distances below are mm.','']
    def table(title,rows):
        lines.extend(['## '+title,'','| Method | Restarts | Mean | Median | P05 | P95 |','|---|---:|---:|---:|---:|---:|'])
        for name,m in rows:
            lines.append(f"| {name} | {m['failures']} | {m['mean_segment_length_mm']:.3f} | {m['median_segment_length_mm']:.3f} | {m['p05_segment_length_mm']:.3f} | {m['p95_segment_length_mm']:.3f} |")
        lines.append('')
    table('Full cohort',[(r['name'],r['metrics']['global_result']) for r in result['rows']])
    for scroll in sorted(result['rows'][0]['metrics']['scrolls']):
        table(scroll,[(r['name'],r['metrics']['scrolls'][scroll]) for r in result['rows']])
    for split in ('fit','selection','test'):
        table('Trace subset: '+split,[(r['name'],r['splits'][split]) for r in result['rows']])
    lines += ['## All fitted candidates','','Parameter vectors: EMA=[weight,half-life]; window=[weight]; gate=[weight,half-life,bias,presence coefficient,distance coefficient]; angular=[weight,half-life,cap degrees per8basevoxels]; dual=[weight,short half-life,long-minus-short half-life,mix bias,mix slope].',
        'Spacing/range are base voxels. Asterisk marks each selection-set winner. Heldout values never select candidates.','',
        '| Family | Spacing | Window | Start | Fitted parameters | Iteration | Selection loss | Heldout loss |',
        '|---|---:|---:|---|---|---:|---:|---:|']
    for c in matrix['candidates']:
        win=matrix['winners'][c['family']]['index']==c['index']
        fmt=lambda xs:', '.join(f'{x:.6g}' for x in xs)
        lines.append(f"| {c['family']}{'*' if win else ''} | {c['variant']['spacing']} | {c['variant'].get('window','—')} | {fmt(c['variant']['start'])} | {fmt(c['best']['parameters'])} | {c['best']['iteration']} | {c['best']['selection_loss']:.7f} | {c['metrics']['test']['ranking_loss']:.7f} |")
    lines += ['','## Runtime (not a controlled speed benchmark)','','| Family | Native seconds | Coordinator seconds |','|---|---:|---:|']
    for row in result['rows']:
        if row['name'] in matrix['winners']:
            seconds=lambda x:'—' if x is None else f'{x:.3f}'
            lines.append(f"| {row['name']} | {seconds(row['native_wall_seconds'])} | {seconds(row['coordinator_wall_seconds'])} |")
    lines += ['','RelWithDebInfo GCC13, one full180direction evaluation per family; exclusive64CPU/all8GPU manager allocations. Same cached source/inference data, with additional inference for newly visited off-route chunks where needed. Timings include differing cache misses and are not decoder microbenchmarks.',
        '','## Verification and limitations','',
        f"Native/PyTorch parity: {parity['paths']} real paths, max absolute weighted-score error {parity['max_abs']:.9g}.",
        f"Accepted windows: {result['accepted_windows']}; exclusions: `{dict(rejected)}`.",
        'GPU fitting:150 Adam updates per family,45 candidates,LR.03; candidates within each family are vectorized. Weight and half-life are jointly fitted (finite window fits only weight). Common warm-up up to4096basevoxels is recomputed and detached; branches are32steps/~256basevoxels.',
        'Spacing controls admission of selected observations; every candidate is still scored. Gate rejection consumes the admission interval. Finite history expires at the candidate position. Angular cap scales with elapsed physical distance. Dual continuously updates two EMAs and adaptively mixes distances, not anchors.',
        'Fixed-path surrogate is not beam-search rollout: connected GT candidates need not lie on the discrete beam grid, finite warm-up resets ancestry, and truncated backprop ignores warm-up gradients. This cohort was used for earlier tuning; heldout is not pristine unseen data. Small full-eval count changes are not evidence of significance.',
        'Opt-in flags leave old default behavior unchanged. No rendering requested.','',
        '## Full evaluation artifacts','']
    lines += [f"- {r['name']}: `{r['path']}`" for r in result['rows']]
    runner='/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py'
    configs=Path(__file__).parent/'configs'
    commands=[f'{runner} {mode} --config {configs}/eval_embedding_integration_fit_20260915_v1.json'
              for mode in ('fixed-path-extract','integration-fit','integration-audit','integration-report')]
    commands += [f'{runner} {mode} --config {configs}/eval_embedding_integration_{family}_20260915_v1.json'
                 for family in matrix['winners'] for mode in ('prepare','validate-full','full')]
    result['commands']=commands
    lines += ['','## Exact commands','','```bash',*commands,'```']
    if not result['pending']:
        old=next(r for r in result['rows'] if r['name']=='Previous fitted EMA')
        experiments=[r for r in result['rows'] if r['name'] in matrix['winners']]
        best=min(r['metrics']['global_result']['failures'] for r in experiments)
        previous=old['metrics']['global_result']['failures']
        names=', '.join(r['name'] for r in experiments if r['metrics']['global_result']['failures']==best)
        lines += ['','## Outcome','',
            f'Best new full-cohort count: {best} restarts ({names}); previous fitted EMA: {previous}.',
            'No new family improves the previous full-cohort restart count.' if best>=previous else 'The lower full-cohort count needs independent confirmation; this is not a significance claim.',
            'Compare per-scroll and heldout tables before interpreting an aggregate tie. No automatic default change.']
    lines += ['',f"Pending: {result['pending']}",'']
    atomic_write(root/'comparison.json',result)
    (root/'comparison.md').write_text('\n'.join(lines))
    print(str(root/'comparison.md'),flush=True)
    return result
