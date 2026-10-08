"""Cost-only retuning of all frozen zero-overlap selections; native shared solves."""
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import iso20_iterative as search
from reviewed_bp_independent import initial_states
from lasagna.inference_provenance import atomic_write

shared=search.shared
SOURCE=shared.OUT.parent/'20261003_iso20_post_overlap_low_v1'
OUT=SOURCE.parent/'20261003_iso20_overlap_costs_v1'
REPORT=shared.REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-03-overlap-costs.json'
VARIANTS=[f'{rank}-{n}x{t}-f0' for n,t in ((10,20),(20,40),(10,10),(20,20)) for rank in ('length','cost','presence')]
GRIDS=dict(disable=[0,25,50,100,200,400],boundary=[0,10,50,200],
           p1=[0,.005,.015,.0375,.1,.3],near=[0,.01,.05,.2],far=[0,.01,.05,.2],
           p0=[0,.01,.05,.2],sign=['11','44','176','hard'])
BASE=dict(disable=100,boundary=10,p1=.0375,near=0,far=0,p0=0,sign='hard')


def updates(p):
    return {'--winding-defect-cost':p['disable'],'--winding-gap-boundary-cost':p['boundary'],
            '--winding-weights':f"{p['near']},{p['far']},{p['p0']},{p['p1']},0",
            '--winding-sign-cost':p['sign'],'--winding-hard-signs':'both',
            '--winding-sign-weights':'1,1','--winding-hard-sign-angle':'off'}


def identity(variant,p):
    return variant+'-'+hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:12]


def neighbours(p):
    result={json.dumps(p,sort_keys=True):dict(p)}
    for k,values in GRIDS.items():
        for v in values:
            q={**p,k:v};result[json.dumps(q,sort_keys=True)]=q
    return [result[k] for k in sorted(result)]


def choose(names,rows,incumbent):
    best=max(search.rank(rows[n]) for n in names)
    if incumbent in names and search.rank(rows[incumbent])==best:return incumbent
    return min(n for n in names if search.rank(rows[n])==best)


def remaining_rounds(item):
    plateau=bool(item['rounds'] and item['rounds'][-1]['winner']==item['rounds'][-1]['previous'])
    return range(2 if plateau or item['stage']=='complete' else len(item['rounds']),2)


def publish(state):
    atomic_write(OUT/'state.json',state,durable=False)
    atomic_write(REPORT,state,durable=False)
    old=json.loads((SOURCE/'results.json').read_text())
    def metrics(r):
        i=r.get('independent',{});o=r.get('oracle',{})
        return [i.get('independent_percent'),i.get('observed_ladder_shift',{}).get('ladder_shift_percent'),o.get('constraint_score_percent')]
    def values(r):return ' | '.join('unavailable' if x is None else f'{x:.4f}%' for x in metrics(r))
    lines=['# Zero-overlap cost tuning','',
        'Frozen raw40500 iso20 beam8 1024 crop, all twelve 0% overlap selections. '
        'Cost-only, at most two coordinate-search rounds per selection; not exhaustive. '
        'Rank: non-oracle accuracy, observed shift, then reference-conditioned removal. '
        'Min256, median±64, fixed scale .822/phase .5,500 messages unchanged. '
        'Four concurrent candidates,128 native threads each. No inference or retracing.','',
        f"Status: {state['stage']}. Trials: {len(state['rows'])}.",
        '', '| Variant | Status | Before accuracy | Before shift | Before removal | Tuned accuracy | Tuned shift | Tuned removal |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
        '| No-dedup baseline | historical matched control | '+values(old['control'])+' | '+values(old['control'])+' |']
    for v in VARIANTS:
        item=state['variants'].get(v,{})
        r=state['rows'].get(item.get('winner'),{})
        lines.append(f"| {v} | {item.get('stage','pending')} | "+values(old[v])+' | '+values(r)+' |')
    if state['stage']=='complete':
        best=max(VARIANTS,key=lambda v:search.rank(state['rows'][state['variants'][v]['winner']]))
        winner=state['rows'][state['variants'][best]['winner']]
        failures=sum('ind_error' in r or 'oracle_error' in r for r in state['rows'].values())
        repeat_ind=sum('independent' in item.get('repeat',{}) for item in state['variants'].values())
        repeat_oracle=sum('oracle' in item.get('repeat',{}) for item in state['variants'].values())
        removal=metrics(winner)[2]
        removal_text='unavailable' if removal is None else f'{removal:.4f}%'
        lines+=['','## Outcome','',
            f"All12 selections complete, {len(state['rows'])} unique weight configurations, "
            f"{sum('oracle' in r for r in state['rows'].values())} reference-conditioned removal solves "
            f"plus {repeat_ind} independent and {repeat_oracle} oracle winner repeats. Failed trials: {failures}.",
            '',f"Best accuracy-first selection: `{best}`; accuracy {metrics(winner)[0]:.4f}%, "
            f"observed shift {metrics(winner)[1]:.4f}%, removal {removal_text}.",
            '', 'All repeated ladders/states and oracle counts match. This is bounded '
            'two-round tuning, not a proof of a global optimum. No held-out crop test '
            'or automatic default promotion was performed.']
    lines+=['','## Selected weights','',
        '| Variant | Disable /128 | Boundary | Perp near | Perp far | Parallel same | Parallel one | Sign cost |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v,item in state['variants'].items():
        p=state['rows'][item['winner']]['params']
        lines.append('| '+v+' | '+' | '.join(str(p[k]) for k in ('disable','boundary','near','far','p0','p1','sign'))+' |')
    lines+=['','## Winner coverage/readout diagnostics','',
        '| Variant | Correct /26 | Missing references | Shift references | Reference-conditioned accuracy |',
        '|---|---:|---|---:|---:|']
    for v,item in state['variants'].items():
        r=state['rows'][item['winner']];i=r.get('independent',{});o=r.get('oracle',{})
        oracle_accuracy=o.get('reference_score_percent')
        oracle_text='unavailable' if oracle_accuracy is None else f'{oracle_accuracy:.4f}%'
        lines.append(f"| {v} | {i.get('correct_references','unavailable')} | {i.get('missing_references',i.get('independent_missing','unavailable'))} | {i.get('observed_ladder_shift',{}).get('references','unavailable')} | {oracle_text} |")
    lines+=['','## Protocol and reproducibility','',
        'Grids: `'+json.dumps(GRIDS,sort_keys=True)+'`. The second round combines changes by '
        'searching around the first winner. A first-round plateau stops that selection. '
        'Inactive P2 stays0 (parallel cutoff1.5); continuity remains hard. '
        'Finite sign penalties permit sign violations; hard does not. Zero magnitude '
        'does not remove the separately configured sign term. '
        'Removal is evaluated only for tied accuracy/shift finalists plus controls/winners; '
        'other removal scores are not run, not zero. Failed trials/gauge failures stay explicit in JSON. '
        'All baselines are rerun and checked against the saved untuned ladder; winners repeated. '
        'Every valid independent/oracle pair must have identical initial states and factors.',
        '', 'Accuracy denominator26; observed shift excludes missing references. '
        'Removal is −100×problematic/retained fulfilled (higher is better), not an accuracy. '
        'These are tuning-set results, not held-out validation. No production default changes. '
        'Retained trace geometry is fixed per variant; artifact replay not measured.',
        '', f'Artifacts/configs/logs/provenance: `{OUT}`. Native RelWithDebInfo. '
        'Run: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, control '
        '`{"action":"iso20_overlap_costs"}`; add `"report_only":true` to refresh. '
        'One solve per configuration plus final repeats; concurrent timings are not latency comparisons.',
        '', '## All trials','',
        '| Variant/trial | Weights | Accuracy | Shift | Removal | Error |',
        '|---|---|---:|---:|---:|---|']
    for n,r in sorted(state['rows'].items()):
        lines.append('| '+n+' | `'+json.dumps(r['params'],sort_keys=True)+'` | '+values(r)+' | '+str(r.get('ind_error',r.get('oracle_error',''))).replace('|','/')+' |')
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main(report_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    shared.OUT=OUT;search.OUT=OUT;search.isolation.OUT=OUT
    path=OUT/'state.json'
    state=json.loads(path.read_text()) if path.exists() else dict(grids=GRIDS,stage='starting',rows={},variants={})
    assert state['grids']==GRIDS
    if report_only:publish(state);return
    old=json.loads((SOURCE/'results.json').read_text())
    for v in VARIANTS:
        template=json.loads((SOURCE/f'{v}-ind.config.json').read_text())
        with (SOURCE/f'{v}-ind_consistency.csv').open() as f:baseline=list(csv.DictReader(f))
        def trial(n,p,stage,expected=None):
            return search.run_stage(n,p,stage,template,baseline,expected,parameter_updates=updates(p))
        b=identity(v,BASE)
        saved=state['variants'].get(v,{})
        saved_expected=saved.get('provenance',state['rows'].get(b,{}).get('provenance'))
        r=trial(b,BASE,'ind',saved_expected)
        assert 'independent' in r
        assert r['independent']['ladder']==old[v]['independent']['ladder'],'untuned ladder changed'
        assert initial_states(OUT/f'{b}-ind_consistency.csv')==initial_states(SOURCE/f'{v}-ind_consistency.csv')
        expected=r['provenance']
        state['rows'].setdefault(b,{'params':BASE}).update(r)
        state['rows'][b].update(trial(b,BASE,'oracle',expected))
        assert state['rows'][b]['oracle']['counts']==old[v]['oracle']['counts']
        item=state['variants'].setdefault(v,dict(winner=b,rounds=[],stage='baseline'))
        item['provenance']=expected
        for name,row in state['rows'].items():
            if name.startswith(v+'-') and 'provenance' in row:
                assert row['provenance']==expected,'resumed candidate provenance changed'
        for round_index in remaining_rounds(item):
            incumbent=item['winner'];around=neighbours(state['rows'][incumbent]['params'])
            names=[]
            for p in around:
                n=identity(v,p);names.append(n);state['rows'].setdefault(n,dict(params=p))
            item['stage']=f'round {round_index+1}';state['stage']=v+' '+item['stage'];publish(state)
            for stage in ('ind','oracle'):
                if stage=='ind':todo=[n for n in names if 'independent' not in state['rows'][n] and 'ind_error' not in state['rows'][n]]
                else:
                    best=max(search.primary(state['rows'][n]) for n in names)
                    todo=[n for n in names if search.primary(state['rows'][n])==best and 'oracle' not in state['rows'][n] and 'oracle_error' not in state['rows'][n]]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    jobs={pool.submit(trial,n,state['rows'][n]['params'],stage,expected):n for n in todo}
                    for future in as_completed(jobs):
                        n=jobs[future];state['rows'][n].update(future.result());publish(state)
                        print('TRIAL '+n+' '+stage+' '+str(search.primary(state['rows'][n])),flush=True)
            item['winner']=choose(names,state['rows'],incumbent)
            item['rounds'].append(dict(previous=incumbent,winner=item['winner'],trials=names))
            publish(state)
            if item['winner']==incumbent:break
        n=item['winner'];p=state['rows'][n]['params'];repeat=n+'-repeat'
        check=trial(repeat,p,'ind',expected)
        assert check['independent']['ladder']==state['rows'][n]['independent']['ladder']
        assert initial_states(OUT/f'{repeat}-ind_consistency.csv')==initial_states(OUT/f'{n}-ind_consistency.csv')
        if 'oracle' in state['rows'][n]:
            check.update(trial(repeat,p,'oracle',expected))
            assert check['oracle']['counts']==state['rows'][n]['oracle']['counts']
        item['repeat']=check;item['stage']='complete';publish(state)
        print('WINNER '+v+' '+json.dumps(state['rows'][n]),flush=True)
    state['stage']='complete';publish(state)


if __name__=='__main__':
    import sys
    main('--report-only' in sys.argv)
