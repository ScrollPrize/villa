"""Accuracy-first coordinate sweeps; all inference/solving uses existing native runner."""
import csv
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import iso20_constraint_isolation as isolation
import iso20_probe_removal as removal
from reviewed_bp_independent import initial_states
from lasagna.inference_provenance import atomic_write

shared=removal.shared
SOURCE=removal.SOURCE
OUT=SOURCE.parent/'20261003_iso20_iterative_v1'
REPORT=shared.REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-03-iterative-accuracy.json'
GRIDS=dict(minimum=[128,192,256,384,512,768,1024],boundary=[0,1,5,10,25,50],
    defect=[75,100,125,150,175,200,250,300],parallel=[.005,.01,.025,.0375,.05,.075,.1],
    span=[16,32,64,128,256],blend=[0,.5,1],scale=[.7,.76,.8,.822,.85,.9,.95])
BASE=dict(minimum=256,boundary=0,defect=150,parallel=.05,span=64,blend=1,scale=.822)


def case_id(params):
    assert set(params)==set(GRIDS)
    return 'c'+hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:12]


def canonical(params):
    return tuple(params[k] for k in GRIDS)


def neighbours(params):
    found={case_id(params):dict(params)}
    for key,values in GRIDS.items():
        for value in values:
            p={**params,key:value};found[case_id(p)]=p
    return sorted(found.items(),key=lambda item:canonical(item[1]))


def updates(p):
    return {'--local-min-piece-length':p['minimum'],'--winding-gap-boundary-cost':p['boundary'],
        '--winding-defect-cost':p['defect'],'--winding-weights':f"0,0,0,{p['parallel']},0",
        '--orthogonal-probe-span':p['span'],'--orthogonal-probe-blend':p['blend'],
        '--winding-fixed-scale':p['scale'],'--winding-fixed-phase':.5}


def primary(row):
    ind=row.get('independent',{})
    if ind.get('independent_percent') is None:return Fraction(-1),Fraction(-1)
    s=ind['observed_ladder_shift']
    return Fraction(ind['correct_references'],26),Fraction(s['references']-s['ladder_shifts'],s['references'])


def rank(row):
    o=row.get('oracle')
    last=(True,Fraction(-o['problematic_unique_constraints'],o['retained_fulfilled_unique_constraints'])) if o else (False,Fraction(0))
    return (*primary(row),last)


def finalists(names,rows):
    best=max(primary(rows[n]) for n in names)
    return sorted((n for n in names if primary(rows[n])==best),key=lambda n:canonical(rows[n]['params']))


def choose(names,rows,incumbent):
    best=max(rank(rows[n]) for n in names)
    if incumbent in names and rank(rows[incumbent])==best:return incumbent
    return min((n for n in names if rank(rows[n])==best),key=lambda n:canonical(rows[n]['params']))


def signature(record):
    return {k:record[k] for k in ('binary_sha256','shared_libraries','inputs','references')}


def run_stage(name,p,stage,template,baseline,expected=None,source_name=None,parameter_updates=None):
    config=shared.configured(template,name+'-'+stage,'',updates(p) if parameter_updates is None else parameter_updates)
    if stage=='oracle':config['commands'][0]['args']=removal.oracle_args(config['commands'][0]['args'])
    try:
        rec=shared.execute(config)
    except subprocess.CalledProcessError as exc:
        rec=json.loads((OUT/f'{name}-{stage}.run.json').read_text())
        if expected is not None:assert signature(rec)==expected,'provenance changed'
        log=(OUT/f'{name}-{stage}.log').read_text()
        return {stage+'_error':log.strip().splitlines()[-1],stage+'_returncode':exc.returncode}
    if expected is not None:assert signature(rec)==expected,'provenance changed'
    if stage=='ind':
        r,_=isolation.collect(name+'-ind',rec,baseline)
        return dict(independent=r,provenance=signature(rec))
    log=(OUT/f'{name}-oracle.log').read_text()
    o=dict(**removal.bp_metrics(log),counts=removal.pruning_counts(log),seconds=rec['wall_seconds'])
    original=source_name or name
    same=initial_states(OUT/f'{name}-oracle_consistency.csv')==initial_states(OUT/f'{original}-ind_consistency.csv')
    factors=removal.factor_identities(OUT/f'{name}-oracle_winding_factors.csv')==removal.factor_identities(OUT/f'{original}-ind_winding_factors.csv')
    assert same and factors,'independent/oracle initial state or measurement mismatch'
    return dict(oracle=o,initial_states_identical=same,initial_factors_identical=factors)


def publish(state):
    atomic_write(OUT/'state.json',state,durable=False)
    atomic_write(REPORT,state,durable=False)
    lines=['# Iterative accuracy-first sweep','',
        'Strict rank: exact non-oracle accuracy /26, then observed shift score, then constraint-disable score. '
        '500 solver iterations, 128 native threads, up to 4 concurrent candidates. No inference or rejected repair. '
        'Same raw40500 iso20 crop; piece layout and fixed measurement scale are explicit tuning coordinates. '
        'Scale applies consistently to graph and reference readout; no claim of graph-only calibration gain.','',
        f"Status: {state['stage']}; incumbent `{state['incumbent']}`. Independent rows: {sum('independent' in r for r in state['rows'].values())}.",
        '', '| Round | Previous | Selected | Accuracy ↑ | Shift ↑ | Removal ↑ |', '|---:|---|---|---:|---:|---:|']
    for h in state['rounds']:
        r=state['rows'][h['selected']];i=r['independent'];o=r.get('oracle',{})
        lines.append(f"| {h['round']} | {h['previous']} | {h['selected']} | {isolation.percent(i['independent_percent'])} | {isolation.percent(i['observed_ladder_shift']['ladder_shift_percent'])} | {isolation.percent(o.get('constraint_score_percent'))} |")
    lines+=['','| Candidate | Min | Boundary | Disable | P1 | Probe ± | Blend | Scale | Accuracy ↑ | Shift ↑ (N) | Removal ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for name,r in sorted(state['rows'].items(),key=lambda item:canonical(item[1]['params'])):
        p=r['params'];i=r.get('independent',{});s=i.get('observed_ladder_shift',{});o=r.get('oracle',{})
        end=isolation.percent(o.get('constraint_score_percent')) if o else ('unavailable' if 'oracle_error' in r else 'not run')
        lines.append('| '+' | '.join([name,*[str(p[k]) for k in GRIDS],isolation.percent(i.get('independent_percent')),
            f"{isolation.percent(s.get('ladder_shift_percent'))} ({s.get('references','—')})",end])+' |')
    lines+=['','Removal solves only for candidates tied at the best first-two scores in a round; others explicitly not run. '
        'Missing-reference, tie, gauge and full oracle counts are in JSON. Exact integer ratios drive decisions, not rounded percentages. '
        'Oracle round0 accuracy is recorded but never selects a winner. Current incumbent retained on exact ties. '
        'Historical overlap:50%accuracy/83.3333%shift/−25.2365%removal. Starting median64:61.5385%/87.5%/−29.5030%.',
        '', 'One-coordinate grid-local search, not a global optimum or exhaustive Cartesian sweep. '
        'Unknown removal cannot establish superiority on the third tie-break; failure does not erase a strict accuracy/shift gain. '
        'Results are tuning-crop results, not held-out validation. Production defaults unchanged.',
        '',f'Artifacts: `{OUT}`. Approved command `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, '
        'action `iso20_iterative`; optional `report_only:true`. Restart validates baseline provenance; per-case config/run records are reused.']
    if state.get('repeat'):lines+=['','Final repeat: `'+json.dumps(state['repeat'],sort_keys=True)+'`.']
    if state['stage']=='complete':
        winner=state['rows'][state['incumbent']];base=state['rows'][case_id(BASE)]
        summary=['## Outcome','',
            f"Completed {len(state['rounds'])} rounds / {len(state['rows'])} unique configurations; "
            f"{sum('oracle' in r for r in state['rows'].values())} removal solves plus separate winner verification. " +
            ('No accuracy or shift improvement; improvement only in the third-priority metric.'
             if primary(winner)==primary(base) else 'Improved primary metric(s); see comparison below.'),
            '', '| Setting | Non-oracle accuracy ↑ | Observed shift ↑ | Constraint-disable score ↑ | Problematic / initial ↓ |',
            '|---|---:|---:|---:|---:|']
        for label,r in [('Starting median ±64',base),('Selected combination',winner)]:
            i=r['independent'];o=r['oracle']
            summary.append(f"| {label} | {i['independent_percent']:.4f}% | {i['observed_ladder_shift']['ladder_shift_percent']:.4f}% | {o['constraint_score_percent']:.4f}% | {o['counts']['problematic_percent_initial']:.4f}% |")
        summary+=['',f"Selected parameters: `{json.dumps(winner['params'],sort_keys=True)}`.",
            '', 'Accuracy uses all 26 reference labels. Observed shift uses 24 covered labels, omitting missing windings 4 and 12; '
            'it is not a complete-ladder score. Removal score is −100 × problematic / retained fulfilled; higher (less negative) is better. '
            'Removal solves are reference-conditioned, separate from the non-oracle ranking.',
            '', 'Fixed input: raw40500 iso20 traced crop, XYZ [10240,22016,6144]–[11264,23040,7168], '
            '1024 base voxels per side. Native RelWithDebInfo build, one evaluation per unique configuration plus one independent winner repeat. '
            'Exact commands, binary/library/input/reference hashes, run timings and logs are in the artifact directory.', '']
        lines[2:2]=summary
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main(report_only=False):
    OUT.mkdir(parents=True,exist_ok=True);shared.OUT=OUT;isolation.OUT=OUT
    path=OUT/'state.json'
    state=json.loads(path.read_text()) if path.exists() else dict(schema=1,grids=GRIDS,stage='baseline',incumbent=case_id(BASE),rows={},rounds=[])
    assert state['grids']==GRIDS
    if report_only:publish(state);return
    template=json.loads((SOURCE/'winner-median64.config.json').read_text())
    with (SOURCE/'winner-median64_consistency.csv').open() as f:baseline=list(csv.DictReader(f))
    b=case_id(BASE)
    initial=run_stage(b,BASE,'ind',template,baseline,state.get('provenance'))
    assert 'independent' in initial
    state['provenance']=initial['provenance']
    state['rows'].setdefault(b,{'params':BASE}).update(initial)
    assert initial_states(OUT/f'{b}-ind_consistency.csv')==initial_states(SOURCE/'winner-median64_consistency.csv')
    if 'oracle' not in state['rows'][b]:state['rows'][b].update(run_stage(b,BASE,'oracle',template,baseline,state['provenance']))
    assert state['rows'][b]['oracle']['counts']==json.loads((removal.OUT/'results.json').read_text())['winner-median64']['counts']
    publish(state)
    while state['stage'] not in ('plateau','complete'):
        incumbent=state['incumbent'];around=neighbours(state['rows'][incumbent]['params'])
        for n,p in around:
            assert n not in state['rows'] or state['rows'][n]['params']==p
            state['rows'].setdefault(n,{'params':p})
        state['stage']='independent round '+str(len(state['rounds'])+1);publish(state)
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs={pool.submit(run_stage,n,p,'ind',template,baseline,state['provenance']):n for n,p in around if 'independent' not in state['rows'][n] and 'ind_error' not in state['rows'][n]}
            for f in as_completed(jobs):
                n=jobs[f];state['rows'][n].update(f.result());publish(state)
                print('INDEPENDENT '+n+' '+str(primary(state['rows'][n])),flush=True)
        names=[n for n,_ in around];top=finalists(names,state['rows'])
        state['stage']='removal round '+str(len(state['rounds'])+1);publish(state)
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs={pool.submit(run_stage,n,state['rows'][n]['params'],'oracle',template,baseline,state['provenance']):n for n in top if 'oracle' not in state['rows'][n] and 'oracle_error' not in state['rows'][n]}
            for f in as_completed(jobs):
                n=jobs[f];state['rows'][n].update(f.result());publish(state)
                print('REMOVAL '+n+' '+str(state['rows'][n].get('oracle',{}).get('constraint_score_percent')),flush=True)
        selected=choose(top,state['rows'],incumbent)
        assert rank(state['rows'][selected])>=rank(state['rows'][incumbent])
        h=dict(round=len(state['rounds'])+1,previous=incumbent,selected=selected,neighbours=names,finalists=top,
            failed_neighbours=[n for n in names if 'ind_error' in state['rows'][n]],
            unresolved_removal=[n for n in top if 'oracle' not in state['rows'][n]])
        state['rounds'].append(h);state['incumbent']=selected
        state['stage']='plateau' if selected==incumbent else 'next round'
        publish(state);print('ROUND '+json.dumps(h),flush=True)
    if state['stage']=='plateau':
        winner=state['incumbent'];p=state['rows'][winner]['params'];repeat=winner+'-repeat'
        r=run_stage(repeat,p,'ind',template,baseline,state['provenance'])
        assert initial_states(OUT/f'{repeat}-ind_consistency.csv')==initial_states(OUT/f'{winner}-ind_consistency.csv')
        assert removal.factor_identities(OUT/f'{repeat}-ind_winding_factors.csv')==removal.factor_identities(OUT/f'{winner}-ind_winding_factors.csv')
        assert r['independent']['ladder']==state['rows'][winner]['independent']['ladder']
        state['repeat']={'independent_states_identical':True,'independent_factors_identical':True,'ladder_identical':True}
        if 'oracle' in state['rows'][winner]:
            r.update(run_stage(repeat,p,'oracle',template,baseline,state['provenance']))
            assert r['oracle']['counts']==state['rows'][winner]['oracle']['counts']
            state['repeat']['oracle_counts_identical']=True
        else:state['repeat']['oracle']='unavailable for winner; not repeated'
        state['stage']='complete';publish(state)


if __name__=='__main__':
    import sys
    main('--report-only' in sys.argv)
