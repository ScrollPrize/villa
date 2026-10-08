"""Staged gap-boundary/disable-cost tuning on frozen iso20 event graphs."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import strict_reverse_crop as shared
from iso20_minimum_sweep import OUT as MIN_SOURCE
from reviewed_bp_independent import initial_states,readout_metrics
from lasagna.inference_provenance import atomic_write

OUT=MIN_SOURCE.parent/'20261002_iso20_gap_tuning_v1'
REPORT='docs/fiber_benchmark_runs/2026-10-02-iso20-gap-tuning.md'
MINIMUMS=(64,256)
BOUNDARIES=(0,10,100,1000,10000)


def fragmentation(pieces):
    by_trace=defaultdict(list)
    for p in pieces:by_trace[p['original_trace']].append(p)
    total=disabled=0.
    gaps=boundaries=whole=endpoint=0
    for group in by_trace.values():
        group.sort(key=lambda p:float(p['begin_arc_base']))
        states=[]
        for i,p in enumerate(group):
            length=float(p['end_arc_base'])-float(p['begin_arc_base'])
            assert length>=0
            if i:assert abs(float(p['begin_arc_base'])-float(group[i-1]['end_arc_base']))<1e-6
            dead=p['winding_final_class'] not in ('h','v')
            states.append(dead);total+=length;disabled+=length*dead
        boundaries+=sum(a!=b for a,b in zip(states,states[1:]))
        gaps+=sum(dead and (i==0 or not states[i-1]) for i,dead in enumerate(states))
        whole+=all(states)
        if not all(states):endpoint+=states[0]+states[-1]
    return dict(total_length=total,disabled_length=disabled,disabled_percent=100*disabled/total,
        gap_runs=gaps,boundaries=boundaries,whole_disabled_traces=whole,endpoint_gaps=endpoint,
        interior_gaps=gaps-whole-endpoint)


def rank(r):
    return (r['independent_percent'],r['observed_ladder_shift']['ladder_shift_percent'],
        -r['gap_runs'],-r['boundary_cost'])


def collect_trial(out, name, rec):
    """Shared native-output metrics for cost sweeps; no solver work in Python."""
    log=(out/f'{name}.log').read_text()
    r=readout_metrics(log)
    with (out/f'{name}_consistency.csv').open() as f:pieces=list(csv.DictReader(f))
    r.update(fragmentation(pieces))
    block=log.split('BP hard-order connectivity\n')[1].split('BP final states')[0]
    r['active_components']=int(next(line.split()[3] for line in block.splitlines() if line.startswith('active ')))
    calibration=log.split('fiber winding calibration\n')[1].splitlines()
    fields=dict(zip(calibration[0].split(),calibration[1].split()))
    r['projected_defects']=int(fields['hard_sign_projected_defects'])
    r['converged']=fields['converged']=='true'
    r['solver_fields']=fields
    r['decode_diagnostics']={}
    for line in log.splitlines():
        if line.startswith(('decode_objective ','gap_repair ','decode_repair ')):
            r['decode_diagnostics'][line.split()[0]]=dict(
                item.split('=',1) for item in line.split()[1:] if '=' in item)
    r['seconds']=rec['wall_seconds']
    r['diagnostics']=[line for line in log.splitlines() if line.startswith('local constraint events:') or
        'missing_owner' in line or line.startswith('message_limit ') or line.startswith('converged ')]
    r['prepass_classes']=[p['winding_prepass_class'] for p in pieces]
    assert abs(r['total_length']-757036.997991017)<1e-5
    return r


def report(rows,notes):
    path=shared.REPO/'volume-cartographer'/REPORT
    atomic_write(OUT/'results.json',dict(rows=rows,notes=notes),durable=False)
    atomic_write(path.with_suffix('.json'),dict(rows=rows,notes=notes),durable=False)
    lines=['# Isotropic20 gap-boundary and disabling-cost tuning','',
        'Frozen raw40500 iso20/full-prefix512/no-history traces. Orthogonal-only clean pieces, minima64/256. '
        'Constraint extraction, evidence, prepass and hard active-active continuity unchanged. New opt-in '
        'boundary prior charges once per adjacent active/Defect edge: interior disabledrun two boundaries, '
        'endpoint run one, wholly disabledtrace zero. No charge inside a contiguous disabledrun. '
        'Defect cost remains per weighted incident mass unless explicitly labeled length-based. '
        'Post-decode hard-sign projection is unchanged and may add defects without optimizing this prior. '
        'Gap counts include wholly disabled source traces; interior gaps are reported separately. '
        'Only retained continuation edges participate in the optimized boundary prior: prepass-disconnected '
        'pieces and post-decode projection mean geometric boundary counts need not equal optimized penalties.','',
        'Overlapping iso20 baseline: non-oracle50%,observedshift83.3333%,oracle80.7692%,constraint−25.2365%. '
        'Replay100% inherited for every row:0ordinary failures,6filteredstops,93.7463%retained. '
        'Oracle/constraint scores for event mode remain unavailable due to the existing calibration issue; '
        'this sweep does not alter or retry that evaluator. Accuracy/26; shift uses the observed-reference '
        'count N shown per row (usually24; some settings also lose reference8 and have23). '
        'Tied/unbounded readouts use deterministic rank1, not uniquely identified ladders.','',
        '| Case | Min | Boundary cost | Defect cost | Unary unit | Messages | Non-oracle % | Shift % (N) | Disabled length % | Gaps | Interior gaps | Whole disabled traces | Boundaries | Active components | Projected defects | Ties/unbounded | Seconds |',
        '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        unit=f"{r['length_unit']} base vx" if r.get('length_unit') else 'incident mass'
        lines.append(f"| {name} | {r['minimum']} | {r['boundary_cost']} | {r['defect_cost']} | {unit} | {r['messages']} | "
            f"{r['independent_percent']:.2f}% | {r['observed_ladder_shift']['ladder_shift_percent']:.2f}% ({r['observed_ladder_shift']['references']}) | "
            f"{r['disabled_percent']:.3f}% | {r['gap_runs']} | {r['interior_gaps']} | {r['whole_disabled_traces']} | {r['boundaries']} | {r['active_components']} | "
            f"{r['projected_defects']} | {r['tied_rows']}/{r['unbounded_rows']} | {r['seconds']:.2f} |")
    lines+=['','## Matched comparison (500-message tuning budget)','',
        'Higher percentages are better except disabled length. Replay is inherited, not rerun. '
        '“Unavailable” is the existing local-event oracle calibration failure, not zero. '
        'Length winners are selected across the tested boundary settings.','',
        '| Method | Replay % | Constraint score % | Oracle accuracy % | Non-oracle accuracy % | Observed shift score % | Disabled length % |',
        '|---|---:|---:|---:|---:|---:|---:|',
        '| Overlapping iso20 baseline | 100.00% | −25.24% | 80.77% | 50.00% | 83.33% | — |']
    for minimum in MINIMUMS:
        control=rows.get(f'm{minimum}-b0-d100-i500')
        groups=[('untuned', [control] if control else []),
            ('incident-cost tuned',[r for r in rows.values() if r['minimum']==minimum and r['messages']==500 and not r.get('length_unit')]),
            ('length-cost tuned',[r for r in rows.values() if r['minimum']==minimum and r['messages']==500 and r.get('length_unit')])]
        for label,group in groups:
            if not group:continue
            r=max(group,key=rank)
            lines.append(f"| Minimum{minimum}, {label} | 100.00% | Unavailable | Unavailable | "
                f"{r['independent_percent']:.2f}% | {r['observed_ladder_shift']['ladder_shift_percent']:.2f}% | {r['disabled_percent']:.3f}% |")
    lines+=['','## Stage notes','',*notes,'',f'Artifacts:`{OUT}`. Approved no-argument runner '
        '`bash /home/hendrik/villa3/scripts/run_training_failure_mining`, actions `iso20_gap_tuning` '
        'and `iso20_gap_length_tuning` in scripts/training_failure_mining_control.json. '
        'CPU128threads,RelWithDebInfo,one repetition; no inference/retracing. Parameters selected on this crop, '
        'not a held-out validation. Full ladders/convergence/geometry checks in adjacent JSON.']
    path.write_text('\n'.join(lines)+'\n')


def main(length_stage=False, length_boundary=None):
    OUT.mkdir(parents=True,exist_ok=True);shared.OUT=OUT
    saved=json.loads((OUT/'results.json').read_text()) if length_stage else dict(rows={},notes=[])
    rows=saved['rows'];notes=saved['notes']
    def trial(minimum,boundary,defect=100,messages=500,length_unit=None,suffix=''):
        name=f'm{minimum}-b{boundary}-d{defect}-i{messages}'
        if length_unit is not None:name+=f'-length{length_unit}'
        name+=suffix
        template=json.loads((MIN_SOURCE/f'independent-min{minimum}.config.json').read_text())
        updates={'--winding-gap-boundary-cost':boundary,
            '--winding-defect-cost':defect,'--winding-message-iterations':messages}
        if length_unit is not None:updates['--winding-defect-length-unit']=length_unit
        config=shared.configured(template,name,'',updates)
        rec=shared.execute(config)
        r=collect_trial(OUT,name,rec)
        r.update(minimum=minimum,boundary_cost=boundary,defect_cost=defect,messages=messages,seconds=rec['wall_seconds'])
        r['length_unit']=length_unit
        if boundary==0 and defect==100 and messages==500 and length_unit is None:
            assert initial_states(OUT/f'{name}_consistency.csv')==initial_states(MIN_SOURCE/f'independent-min{minimum}_consistency.csv')
            r['zero_control_identical']=True
        assert abs(r['total_length']-757036.997991017)<1e-5
        rows[name]=r;report(rows,notes)
        print('COMPLETE '+json.dumps({k:v for k,v in r.items() if k!='ladder'}),flush=True)
        return r
    if length_stage:
        candidates=[r for r in rows.values() if r['messages']==500 and not r.get('length_unit')]
        boundary=max(candidates,key=rank)['boundary_cost'] if length_boundary is None else length_boundary
        stage_note=('StageC: physical-length unary via existing native length model; '
            f'common boundarycost{boundary}, disablecost300/1200/4800 per128basevoxels at both minima. '
            'Central1200 approximately matches12.14 signed incidences per128voxels at cost100 in the min64 control; '
            'not exact evidence-mass matching. Prepass unchanged. Rebuilt zero controls checked against historical states.')
        if stage_note not in notes:notes.append(stage_note)
        for minimum in MINIMUMS:
            trial(minimum,0,suffix='-rebuilt-control')
            for cost in (300,1200,4800):trial(minimum,boundary,cost,length_unit=128)
        for minimum in MINIMUMS:
            chosen=max((r for r in rows.values() if r['minimum']==minimum and r.get('length_unit')==128 and r['messages']==500 and r['boundary_cost']==boundary),key=rank)
            previous=max((r for r in candidates if r['minimum']==minimum),key=rank)
            if length_boundary is None or rank(chosen)[:2]>rank(previous)[:2]:
                for budget in (1000,5000):trial(minimum,boundary,chosen['defect_cost'],budget,length_unit=128)
        notes[:]=[n for n in notes if n!='Length-normalized stage pending assessment.']
        completion_note=(f'Length stage at boundarycost{boundary} complete. Primary length sweep has1000/5000budget checks; '
            'strong-boundary interaction receives those checks only if it beats the same-minimum incident-cost winner. '
            'No production defaults changed.')
        if completion_note not in notes:notes.append(completion_note)
        report(rows,notes)
        return
    for minimum in MINIMUMS:
        for boundary in BOUNDARIES:trial(minimum,boundary)
    notes.append('StageA complete: boundarycost0/10/100/1000/10000, defect100; zero controls exactly match historical initial states.')
    for minimum in MINIMUMS:
        best=max((r for r in rows.values() if r['minimum']==minimum),key=rank)
        for defect in (25,400):trial(minimum,best['boundary_cost'],defect)
    notes.append('StageB complete: defect25/100/400 at bestboundary perminimum; selection accuracy then shift then fewergaps.')
    winners={m:max((r for r in rows.values() if r['minimum']==m),key=rank) for m in MINIMUMS}
    improved=any(rank(winners[m])[:2]>rank(rows[f'm{m}-b0-d100-i500'])[:2] for m in MINIMUMS)
    notes.append('Length-normalized stage pending assessment.' if improved else
        'Length-normalized stage not triggered: boundary/disable tuning did not improve accuracy/shift over same-minimum controls.')
    report(rows,notes)
    for minimum,r in winners.items():
        for messages in (1000,5000):trial(minimum,r['boundary_cost'],r['defect_cost'],messages)
    notes.append('Higher-budget1000/5000 checks completed for each minimum winner. No production defaults changed.')
    report(rows,notes)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--length-stage',action='store_true')
    parser.add_argument('--length-boundary',type=int)
    args=parser.parse_args()
    if args.length_boundary is not None and (not args.length_stage or args.length_boundary<0):
        parser.error('--length-boundary requires --length-stage and a nonnegative value')
    main(args.length_stage,args.length_boundary)
