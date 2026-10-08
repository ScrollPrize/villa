"""Controlled native BP follow-ups on frozen iso20 traces (approved runner)."""
import csv
import json
from iso20_gap_tuning import collect_trial, rank, OUT as PREVIOUS, MIN_SOURCE
import strict_reverse_crop as shared
from reviewed_bp_independent import initial_states
from lasagna.inference_provenance import atomic_write

OUT=PREVIOUS.parent/'20261002_iso20_stable_gap_v1'
REPORT=shared.REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-02-iso20-stable-gap.md'


def blend_parameters(alpha, length_cost):
    # Existing native cost = base*((1-alpha)*M + alpha*L/unit).
    # Fix base100; unit=128*100/cL implements the explicitly dimensioned blend.
    if not 0 <= alpha <= 1 or length_cost <= 0:
        raise ValueError('invalid blend/cost')
    return {'--winding-defect-cost':100, '--winding-defect-length-unit':12800/length_cost,
            '--winding-defect-length-blend':alpha}


def report(rows):
    atomic_write(OUT/'results.json',rows,durable=False)
    atomic_write(REPORT.with_suffix('.json'),rows,durable=False)
    lines=['# Stable gap-aware BP experiments','',
        'Frozen raw40500 iso20 traces, CPU128, RelWithDebInfo. No inference/retracing. '
        'Replay100% inherited; local-event oracle/constraint metrics unavailable (existing calibration issue). '
        'Accuracy uses all26 references; shift uses observed N, missing omitted. Tied/unbounded readouts use deterministic rank1. '
        'Selection on this crop is tuning, not held-out validation.','',
        'Controls: overlapping baseline accuracy50.00%, shift83.33%; prior minimum64 incident tuned30.77%/54.17%; '
        'pure length tuned42.31%/62.50% at500,30.77%/58.33% at5000.','',
        'Cost=(1-alpha)*100*M + alpha*cL*L/128. Winding damping is the new-message fraction; prepass remains unchanged.','',
        '| Trial | Accuracy % | Shift % (N) | Disabled length % | Gaps interior/endpoint/whole | Boundaries | Components | Projection defects | Ties/unbounded | Missing | Seconds |',
        '|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|']
    for name,r in rows.items():
        s=r['observed_ladder_shift']
        lines.append(f"| {name} | {r['independent_percent']:.2f}% | {s['ladder_shift_percent']:.2f}% ({s['references']}) | "
            f"{r['disabled_percent']:.3f}% | {r['interior_gaps']}/{r['endpoint_gaps']}/{r['whole_disabled_traces']} | "
            f"{r['boundaries']} | {r['active_components']} | {r['projected_defects']} | {r['tied_rows']}/{r['unbounded_rows']} | "
            f"{r['missing_references']} | {r['seconds']:.2f} |")
    lines+=['','Full solver fields, convergence diagnostics, ladders and parameter settings in adjacent JSON.',
        f'Artifacts: `{OUT}`. Approved runner: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, action `iso20_stable_gap`.']
    REPORT.write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True);shared.OUT=OUT
    rows={}
    with (MIN_SOURCE/'independent-min64_consistency.csv').open() as f:
        expected_prepass=[p['winding_prepass_class'] for p in csv.DictReader(f)]
    def trial(alpha,damping,messages=500,control=False):
        if messages!=500:raise ValueError('This experiment is fixed at500 iterations; no budget sweeps')
        name=f'a{alpha:g}-damp{damping:g}-i{messages}' if not control else 'default-control'
        name=name.replace('.','p')
        template=json.loads((MIN_SOURCE/'independent-min64.config.json').read_text())
        updates={'--winding-gap-boundary-cost':1000,'--winding-message-iterations':messages}
        if not control:
            updates.update(blend_parameters(alpha,300));updates['--winding-damping']=damping
        config=shared.configured(template,name,'',updates)
        rec=shared.execute(config);r=collect_trial(OUT,name,rec)
        assert r.pop('prepass_classes')==expected_prepass, 'orientation prepass changed'
        r.update(alpha=alpha,damping=damping,messages=messages,length_cost=300,boundary_cost=1000,minimum=64,
                 prepass_identical=True,control=control)
        if control:
            assert initial_states(OUT/f'{name}_consistency.csv')==initial_states(PREVIOUS/'m64-b1000-d100-i500_consistency.csv')
        if alpha==0 and damping==.5 and not control:
            assert initial_states(OUT/f'{name}_consistency.csv')==initial_states(OUT/'default-control_consistency.csv')
        rows[name]=r;report(rows)
        print('COMPLETE '+name+' '+json.dumps({k:r[k] for k in ('independent_percent','observed_ladder_shift','disabled_percent','seconds')}),flush=True)
        return r
    trial(0,.5,control=True)
    for alpha in (0,.25,.5,.75,1):
        for damping in (.5,.25,.1):trial(alpha,damping)
    # User constraint: keep all experiments at500; no higher-iteration retries.


if __name__=='__main__':main()
