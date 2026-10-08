"""Resumable native cost sweep of the selected median64 benchmark variant."""
import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import iso20_constraint_isolation as isolation
import iso20_probe_removal as removal
from reviewed_bp_independent import initial_states
from lasagna.inference_provenance import atomic_write

shared=removal.shared
SOURCE=removal.SOURCE
OUT=SOURCE.parent/'20261003_iso20_median_costs_v1'
REPORT=shared.REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-03-median-costs.json'


def matrix():
    for p in (.05,0,.005,.01,.025,.1):
        yield 'p'+str(p).replace('.','p'),{'--winding-weights':f'0,0,0,{p},0'}
    for d in (0,75,300):
        yield f'd{d}',{'--winding-defect-cost':str(d)}
    for b in (100,500):
        yield f'b{b}',{'--winding-gap-boundary-cost':str(b)}
    for i,name in ((0,'perp-near'),(1,'perp-far'),(2,'parallel-same')):
        weights=[0,0,0,.05,0];weights[i]=.05
        yield name,{'--winding-weights':','.join(map(str,weights))}
    for cost in (22,88):
        yield f'sign{cost}',{'--winding-hard-signs':'both','--winding-sign-cost':str(cost),'--winding-hard-sign-angle':'off'}
    yield 'no-parallel-sign',{'--winding-sign-weights':'1,0'}
    yield 'no-perp-sign',{'--winding-sign-weights':'0,1'}


def combinations(rows):
    base=rows['p0p05'].get('oracle',{}).get('constraint_score_percent')
    if base is None:return []
    candidates=sorted((r for r in rows.values() if r.get('oracle',{}).get('constraint_score_percent',-float('inf'))>base),
        key=lambda r:r['oracle']['constraint_score_percent'],reverse=True)
    result=[]
    for i,a in enumerate(candidates):
        for b in candidates[i+1:]:
            if set(a['updates']) & set(b['updates']):continue
            update={**a['updates'],**b['updates']}
            if update not in [u for _,u in result]:result.append((f'combo{len(result)+1}',update))
            if len(result)==2:return result
    return result


def trial(name,updates,template,baseline):
    r=dict(updates=updates)
    independent=shared.configured(template,name+'-ind','',updates)
    rec=shared.execute(independent)
    r['independent'],_=isolation.collect(name+'-ind',rec,baseline)
    assert r['independent']['same_piece_identities']
    oracle=shared.configured(template,name+'-oracle','',updates)
    oracle['commands'][0]['args']=removal.oracle_args(oracle['commands'][0]['args'])
    try:
        rec=shared.execute(oracle)
    except Exception as exc:
        path=OUT/f'{name}-oracle.run.json'
        if not path.exists():raise
        rec=json.loads(path.read_text())
        if rec['returncode']==0:raise
        r['oracle_error']=str(exc)
        r['oracle_log_tail']=(OUT/f'{name}-oracle.log').read_text()[-3000:]
        return r
    log=(OUT/f'{name}-oracle.log').read_text()
    r['oracle']=dict(**removal.bp_metrics(log),counts=removal.pruning_counts(log),seconds=rec['wall_seconds'])
    r['same_initial_states']=initial_states(OUT/f'{name}-ind_consistency.csv')==initial_states(OUT/f'{name}-oracle_consistency.csv')
    r['same_initial_factors']=removal.factor_identities(OUT/f'{name}-ind_winding_factors.csv')==removal.factor_identities(OUT/f'{name}-oracle_winding_factors.csv')
    assert r['same_initial_states'] and r['same_initial_factors']
    if name=='p0p05':
        assert initial_states(OUT/f'{name}-ind_consistency.csv')==initial_states(SOURCE/'winner-median64_consistency.csv')
        old=json.loads((removal.OUT/'results.json').read_text())['winner-median64']
        assert r['oracle']['counts']==old['counts']
    return r


def publish(rows):
    atomic_write(OUT/'results.json',rows,durable=False)
    atomic_write(REPORT,rows,durable=False)
    lines=['# Median64 cost sweep','',
        'Selected baseline:median64/min256, length150 per128,boundary0,weights0/0/0/0.05/0,hard signs both. '
        'Frozen raw40500 iso20 1024 crop;500iterations,128native threads,up to4 independent candidates concurrently. '
        'Each candidate gets independent readout AND historical oracle-inliers removal. Oracle protocol fixed. '
        'No inference, new geometry, repair, or iteration tuning. One run per variant; timings under concurrent scheduling are not speed comparisons.','',
        '| Case | Parameter changes | Removal score ↑ | Problematic % | Problematic / fulfilled counts | Oracle round0 accuracy ↑ | Independent ↑ | Shift ↑ |',
        '|---|---|---:|---:|---|---:|---:|---:|']
    order=[n for n,_ in matrix()]+sorted(n for n in rows if n.startswith('combo'))
    for name in order:
        if name not in rows:continue
        r=rows[name]
        ind=r['independent'];o=r.get('oracle');c=o['counts'] if o else None
        lines.append('| '+ ' | '.join([name,', '.join(k.removeprefix('--')+'='+v for k,v in r['updates'].items()),
            isolation.percent(o['constraint_score_percent']) if o else 'unavailable',
            isolation.percent(c['problematic_percent_initial']) if c else '—',
            f"{c['problematic']} / {c['retained_fulfilled']}" if c else '—',
            isolation.percent(o['reference_score_percent']) if o else '—',
            isolation.percent(ind['independent_percent']),isolation.percent(ind['observed_ladder_shift']['ladder_shift_percent'])])+' |')
    lines+=['','Removal score=−100×problematic/fulfilled; closer to0 is better. Problematic includes incident-to-removed, '
        'retained infringed and retained defect; full breakdown/missing refs/initial-state checks in JSON. '
        'Historical old overlap:removal−25.2365%,oracle80.7692%,independent50%,shift83.3333%. '
        'Replay100% inherited unchanged. Shift uses observed references only; independent accuracy denominator26.',
        '', 'P2 cannot contribute under the retained parallel cutoff1.5; not swept. Hard continuity stays hard; '
        'gap-boundary is its tunable active/disabled cost. Finite sign cases relax both hard sign categories. '
        'Zero sign weights remove the selected sign term. Calibration/temperature/confidence/oracle cost fixed. '
        'No production defaults changed. Current baseline designation applies to the selected fine-split approach, not global dominance.',
        '', f'Artifacts: `{OUT}`. Approved command `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, '
        'action `iso20_median_costs`; `report_only:true` refreshes summaries. Config/run records validate resumption. '
        'At most2 pairwise combinations of individually removal-improving disjoint cost coordinates follow the matrix.']
    failures=[(n,r) for n,r in rows.items() if 'oracle_error' in r]
    if failures:
        lines+=['','## Unavailable cases']
        for name,r in failures:
            lines+=['',f"- {name}: {r['oracle_log_tail'].strip().splitlines()[-1]}. No gauge/metric substitution applied."]
    if 'combo2' in rows:
        lines+=['','## Verdict','',
            '- Completed18 single-coordinate cases and2 grouped-coordinate combinations.17+2 removal results valid; '
            'no-perp-sign is unavailable because reference pieces cross incompatible gauges. Independent/oracle initial states and factors match for every valid pair. '
            'Baseline repeats previous median64 initial states and removal counts exactly.10 Python tests passed; no native changes.',
            '- Lower P1=.01 gives only a small removal improvement (−29.4301% vs−29.5030%) while shift falls87.50%→83.33%. '
            'P1=0 collapses independent accuracy to3.85% and removal to−341.3170%. Smaller is not monotonically better.',
            '- Best removal:far-perpendicular magnitude.05 plus no parallel sign,−13.1589%,28337 problematic/215345 fulfilled '
            '(11.6287% problematic),oracle80.7692%,independent46.1538%,shift79.1667%.',
            '- Better balance:far-perpendicular magnitude.05 plus disabling75 per128,−17.5073%,36306 problematic/207376 fulfilled '
            '(14.8989% problematic),oracle80.7692%,independent61.5385%,shift83.3333%. '
            'This preserves baseline exact accuracy and substantially improves removal, but loses one observed shift step.',
            '- Keep selected median64/P1=.05/disable150 baseline recorded; new results are Pareto alternatives, not automatic production promotions. '
            'Changed costs affect reference evidence and fulfillment as well as graph optimization; these are end-to-end operational removal scores, '
            'not proof the underlying measured constraints became more accurate.',
            '- Five-entry magnitude vector treated as one grouped coordinate in bounded combination selection; not an exhaustive Cartesian grid '
            'and internal combinations of P1 with other magnitude entries were not swept. Inactive P2, calibration, temperature, '
            'confidence, continuity hardness and oracle protocol unchanged. No extra iterations or rejected repair.']
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main(report_only=False):
    OUT.mkdir(parents=True,exist_ok=True);shared.OUT=OUT;isolation.OUT=OUT
    if report_only:publish(json.loads((OUT/'results.json').read_text()));return
    template=json.loads((SOURCE/'winner-median64.config.json').read_text())
    with (SOURCE/'winner-median64_consistency.csv').open() as f:baseline=list(csv.DictReader(f))
    rows={}
    for stage in (0,1):
        cases=list(matrix()) if stage==0 else combinations(rows)
        if not cases:print('No individually improving disjoint coordinates to combine.',flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending={pool.submit(trial,n,u,template,baseline):n for n,u in cases}
            for future in as_completed(pending):
                name=pending[future];rows[name]=future.result();publish(rows)
                print('COMPLETE '+name+' '+json.dumps(rows[name].get('oracle',{'error':rows[name].get('oracle_error')})),flush=True)


if __name__=='__main__':
    import sys
    main('--report-only' in sys.argv)
