"""Later bounded stages; native solver, frozen evidence, per-stage checkpoints."""
import csv
import json
from iso20_stable_gap import OUT as SCREEN, blend_parameters
from iso20_gap_tuning import collect_trial, rank, MIN_SOURCE
import strict_reverse_crop as shared
from reviewed_bp_independent import initial_states
from lasagna.inference_provenance import atomic_write

OUT=SCREEN.parent/'20261002_iso20_stable_followups_v1'
REPORT=shared.REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-02-iso20-stable-followups.md'


def publish(rows,stages):
    doc=dict(rows=rows,stages=stages)
    atomic_write(OUT/'results.json',doc,durable=False)
    atomic_write(REPORT.with_suffix('.json'),doc,durable=False)
    lines=['# Stable gap-aware BP follow-ups','',
        'Frozen raw40500 iso20, CPU128, RelWithDebInfo; no retracing. Replay100% inherited; '
        'local-event oracle and constraint-removal metrics unavailable due existing calibration failure. '
        'Accuracy/26; shift observed-only (N and missing shown), deterministic rank1 for ties/unbounded. '
        'Overlapping baseline50.00% accuracy/83.33% shift; prior length300/128+boundary1000 at500:42.31%/62.50%. '
        'Full parameters, solver objective/residual, repair stats and ladders in adjacent JSON. '
        'Trial key: a=length blend, d=winding update fraction, b=boundary cost, c=length cost per128basevx, '
        'i=iteration cap, m=minimum piece length, best=checkpoint retention, r=run-opening cost, s=gap-repair sweeps. '
        'Objectives are comparable only within identical cost settings, not across grid rows.','',
        '| Trial | Accuracy % | Shift % (N) | Disabled % | Interior/endpoint/whole gaps | Boundaries | Components | Projected defects | Ties/unbounded | Missing | Selected iter | Objective / final | Seconds |',
        '|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---|---:|']
    for name,r in rows.items():
        s=r['observed_ladder_shift'];d=r['decode_diagnostics'].get('decode_objective',{})
        lines.append(f"| {name} | {r['independent_percent']:.2f}% | {s['ladder_shift_percent']:.2f}% ({s['references']}) | "
            f"{r['disabled_percent']:.3f}% | {r['interior_gaps']}/{r['endpoint_gaps']}/{r['whole_disabled_traces']} | "
            f"{r['boundaries']} | {r['active_components']} | {r['projected_defects']} | {r['tied_rows']}/{r['unbounded_rows']} | "
            f"{r['missing_references']} | {d.get('selected_iteration','—')} | {d.get('normalized','—')} / {d.get('final','—')} | {r['seconds']:.2f} |")
    lines+=['','## Stage comparison (best measured configuration per stage)','',
        '| Method | Replay % | Constraint score % | Oracle accuracy % | Non-oracle accuracy % | Shift % (N) | Disabled length % |',
        '|---|---:|---:|---:|---:|---|---:|',
        '| Overlapping iso20 baseline | 100.00% | −25.24% | 80.77% | 50.00% | 83.33% (24) | — |']
    filters=[('Prior length300/boundary1000',lambda r:r['minimum']==64 and r['alpha']==1 and r['boundary_cost']==1000 and r['length_cost']==300 and not r['retain'] and not r['opening'] and not r['repair']),
        ('Checkpoint selection',lambda r:r['retain'] and not r['opening'] and not r['repair']),
        ('Refined cost grid',lambda r:r['minimum']==64 and not r['opening'] and not r['repair']),
        ('Run-opening prior',lambda r:r['minimum']==64 and r['opening']>0 and not r['repair']),
        ('Native gap repair (with/without run prior)',lambda r:r['minimum']==64 and r['repair']>0)]
    for label,predicate in filters:
        group=[r for r in rows.values() if predicate(r)]
        if not group:continue
        r=max(group,key=rank);s=r['observed_ladder_shift']
        lines.append(f"| {label} | 100.00% | Unavailable | Unavailable | {r['independent_percent']:.2f}% | "
            f"{s['ladder_shift_percent']:.2f}% ({s['references']}) | {r['disabled_percent']:.3f}% |")
    lines+=['','## Interval repair details','',
        '| Trial | Proposals | Accepted moves | Restored base vx | Disabled base vx | Objective before → after | Repair seconds | Cap hits / unvisited pieces |',
        '|---|---:|---:|---:|---:|---|---:|---|']
    for name,r in rows.items():
        if not r['repair']:continue
        d=r['decode_diagnostics']['gap_repair']
        lines.append(f"| {name} | {d['proposals']} | {d['accepted']} | {d['restored_length']} | {d['disabled_length']} | "
            f"{d['before']} → {d['after']} | {d['seconds']} | {d['cap_hits']} / {d['unvisited_pieces']} |")
    lines+=['','## Completed stages','',*[f'- {s}' for s in stages],'',
        f'Artifacts:`{OUT}`. Approved runner action `iso20_stable_followups`; control key `stage` selects2–5. '
        'Stage0 refreshes this report only, without running a solver. '
        'One run per setting; runtime is observational, not isolated microbenchmark.']
    if all(s in stages for s in (2,3,4,5)):
        lines+=['','## Conclusion','',
            'None of these stages beats the preceding42.31% accuracy/62.50% shift result, '
            'and the overlapping-piece baseline50.00%/83.33% remains better. '
            'Checkpoint selection chose the terminal state in both controls. '
            'The cost grid tied the previous scores with slightly fewer gaps. '
            'All positive run-opening penalties hurt these benchmark scores. '
            'Local repair reduces the optimized objective and boundaries but not winding errors. '
            'No experimental feature or parameter is promoted to a production default. '
            'The prior blend/damping screen is in2026-10-02-iso20-stable-gap.md. '
            'After the explicit user correction, all subsequent runs use500iterations; no more budget sweeps.']
    REPORT.write_text('\n'.join(lines)+'\n')


def main(stage):
    OUT.mkdir(parents=True,exist_ok=True);shared.OUT=OUT
    saved=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else dict(rows={},stages=[])
    rows=saved['rows'];stages=saved['stages']
    if stage==0:
        publish(rows,stages)
        return
    screen=json.loads((SCREEN/'results.json').read_text())
    best=max((r for r in screen.values() if r['messages']==500 and not r['control']),key=rank)
    def trial(alpha=1.,damping=.5,boundary=1000,cost=300,messages=500,minimum=64,
              retain=False,opening=0,repair=0):
        if messages!=500:raise ValueError('This experiment is fixed at500 iterations; no budget sweeps')
        name=f'a{alpha:g}-d{damping:g}-b{boundary}-c{cost}-i{messages}-m{minimum}-best{int(retain)}-r{opening}-s{repair}'.replace('.','p')
        updates=blend_parameters(alpha,cost)
        updates.update({'--winding-damping':damping,'--winding-gap-boundary-cost':boundary,
            '--winding-message-iterations':messages,'--winding-run-opening':opening,'--winding-gap-repair':repair})
        template=json.loads((MIN_SOURCE/f'independent-min{minimum}.config.json').read_text())
        config=shared.configured(template,name,'',updates,['--winding-best-feasible'] if retain else [])
        rec=shared.execute(config);r=collect_trial(OUT,name,rec)
        with (MIN_SOURCE/f'independent-min{minimum}_consistency.csv').open() as f:
            expected=[p['winding_prepass_class'] for p in csv.DictReader(f)]
        assert r.pop('prepass_classes')==expected
        if minimum==64 and boundary==1000 and cost==300 and not retain and opening==0 and repair==0:
            source=SCREEN/(f'a{alpha:g}-damp{damping:g}-i500'.replace('.','p')+'_consistency.csv')
            assert initial_states(OUT/f'{name}_consistency.csv')==initial_states(source), 'rebuilt control changed'
        r.update(alpha=alpha,damping=damping,boundary_cost=boundary,length_cost=cost,messages=messages,
            minimum=minimum,retain=retain,opening=opening,repair=repair,prepass_identical=True)
        d=r['decode_diagnostics']['decode_objective']
        if retain:assert float(d['normalized'])<=float(d['final'])+1e-4
        rows[name]=r;publish(rows,stages)
        print('COMPLETE '+name+f" accuracy={r['independent_percent']:.4f}% shift={r['observed_ladder_shift']['ladder_shift_percent']:.4f}%",flush=True)
        return r
    def args(r):
        return dict(alpha=r['alpha'],damping=r['damping'],boundary=r['boundary_cost'],cost=r['length_cost'],retain=r['retain'])
    if stage==2:
        settings=list(dict.fromkeys([(best['alpha'],best['damping']),(1.,.5),(0.,.5)]))
        for alpha,damping in settings:
            for messages in (500,):
                for retain in (False,True):trial(alpha,damping,messages=messages,retain=retain)
    elif stage==3:
        chosen=max((r for r in rows.values() if r['messages']==500 and r['minimum']==64),key=rank)
        alphas={chosen['alpha'],1.}
        if chosen['alpha']==0:
            alphas.add(max((r for r in rows.values() if r['messages']==500 and r['alpha']>0),key=rank)['alpha'])
        candidates=[]
        for alpha in sorted(alphas):
            for boundary in (500,1000,2000):
                for cost in ((300,) if alpha==0 else (150,300,600)):
                    candidates.append(trial(alpha,chosen['damping'],boundary,cost,retain=chosen['retain']))
        for r in sorted(candidates,key=rank,reverse=True)[:2]:
            trial(**args(r),minimum=256)
    elif stage==4:
        chosen=max((r for r in rows.values() if r['messages']==500 and r['minimum']==64),key=rank)
        candidates=[]
        for alpha in sorted({chosen['alpha'],1.}):
            for opening in (0,250,1000,4000):
                p=args(chosen);p['alpha']=alpha
                candidates.append(trial(**p,opening=opening))
        r=max(candidates,key=rank);trial(**args(r),opening=r['opening'],minimum=256)
    elif stage==5:
        chosen=max((r for r in rows.values() if r['messages']==500 and r['minimum']==64),key=rank)
        positive=max((r for r in rows.values() if r['messages']==500 and r['minimum']==64 and r['opening']>0),key=rank)
        for opening in (0,positive['opening']):
            for sweeps in (0,1,3,10):trial(**args(chosen),opening=opening,repair=sweeps)
    else:raise ValueError(stage)
    if stage not in stages:stages.append(stage)
    publish(rows,stages)


if __name__=='__main__':
    import sys
    main(int(sys.argv[1]))
