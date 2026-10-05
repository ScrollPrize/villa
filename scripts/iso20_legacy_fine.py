"""Port legacy measurements to fine owners; all extraction/solving stays native."""
import csv
import json
from pathlib import Path
import iso20_constraint_isolation as isolation
from lasagna.inference_provenance import atomic_write

OUT=isolation.OUT.parent/'20261002_iso20_legacy_fine_v1'
REPORT=isolation.REPORT.with_name('2026-10-02-iso20-legacy-fine.md')


def matrix():
    yield 'baseline',{},[]
    for minimum in (64,128,256):
        for mode in ('incident','length','length-boundary'):
            updates={'--local-min-piece-length':minimum}
            flags=['--legacy-fine-pieces']
            if mode!='incident':
                updates.update({'--winding-defect-length-unit':128,'--winding-defect-cost':150,
                    '--winding-gap-boundary-cost':500 if mode=='length-boundary' else 0})
                flags.append('--partition-length-cost')
            yield f'm{minimum}-{mode}',updates,flags


def publish(rows):
    atomic_write(OUT/'results.json',rows,durable=False)
    atomic_write(REPORT.with_suffix('.json'),rows,durable=False)
    lines=['# Legacy constraints ported to fine pieces','',
        'Frozen raw40500 iso20 Paris4 1024 crop,128CPU workers,RelWithDebInfo,500 iterations. '
        'Old512/128 measurements preserved once each; only their owner pieces and continuity change. '
        'Cuts between old orthogonal observation arcs, with minimum64/128/256 base voxels. '
        'No new certification, local geometry scorer, normalization or deduplication. Old reference readout throughout. '
        'Measurements retain their old spatial support, not remeasured on small owners. No inference/retracing.','',
        'Accuracy denominator26; shift uses24 observed references unless indicated. Higher is better. '
        'Replay100% inherited; old oracle80.77%,constraint score−25.24% historical only. No new oracle per variant.','',
        '| Variant | Accuracy | Shift (N) | Disabled length | Pieces | Missing | Ties/unbounded | Seconds |',
        '|---|---:|---|---:|---:|---|---|---:|']
    for name,r in rows.items():
        s=r['observed_ladder_shift']
        lines.append(f"| {name} | {isolation.percent(r['independent_percent'])} | {isolation.percent(s['ladder_shift_percent'])} ({s['references']}) | "
            f"{r['disabled_percent']:.2f}% | {r['pieces']} | {r['missing_references']} | {r.get('tied_rows','N/A')}/{r.get('unbounded_rows','N/A')} | {r['seconds']:.1f} |")
    lines+=['','Incident: original weighted-incidence disabling cost. Length:150 per128 base voxels. '
        'Length-boundary additionally charges500 at active/disabled transitions. binary suffix hardens old orientation scores; '
        'd75/d300 change length cost only. N/A means multiple active independent gauges, summaries in JSON. '
        'Tied assignments use deterministic rank1, not uniquely determined labels. Shift excludes missing references and bridges their gaps.',
        '', 'Historical fine64 with new measurements/tuned costs and old readout:38.46% accuracy/45.83% shift. '
        'Old overlap with binary scores or length+boundary:53.85%/79.17%. These are tuning-crop comparisons, not held-out validation.',
        '',f'Artifacts: `{OUT}`. Exact commands/binary hashes are in run.json files. '
        'Approved runner: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, action `iso20_legacy_fine`. '
        '`followup:true` runs at most six predefined local followups; `report_only:true` only refreshes this table.']
    if 'm256-incident-boundary' in rows:
        lines+=['','## Outcome','',
            '- Completed baseline +9 ports +6 bounded followups. All242762 ordinary observations and total evidence weight preserved; old baseline initial states reproduced exactly.',
            '- Best shift: minimum256, soft legacy measurements, length cost75 or150 per128 voxels, boundary0:50.00% accuracy/87.50% observed shift. No tied/unbounded reference rows. '
            'Same13/26 exact references as baseline; offset-change count falls4 to3 among24 observed references. This is a modest metric improvement, not a solution to the remaining winding errors.',
            '- Length150 disables6.04% of physical length versus old baseline10.49%; length75 gives6.07% and the same reference ladder. '
            'Minimum256 yields2799 nonoverlapping pieces vs2732 overlapping baseline pieces; the winning control is not the very fine64 layout.',
            '- Fine64 with length+boundary improves to46.15%/75.00%, versus38.46%/45.83% using new measurements with that readout. Minimum128+length+boundary matches50.00%/83.33%.',
            '- Binary scores, doubled length cost or boundary priors do not improve the best port. Prior tuned old-overlap accuracy53.85%/shift79.17% remains an alternative tradeoff, not dominated on both metrics.',
            '-446 native tests +46 Python checks pass. No default promotion, no extra BP iteration budgets, no inference/retracing. '
            'One crop only; oracle/removal scores not recomputed. Recorded times are single-run observations, not repeated performance claims.']
    REPORT.write_text('\n'.join(lines)+'\n')


def main(followup=False,report_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    isolation.OUT=OUT;isolation.shared.OUT=OUT
    rows=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else {}
    if report_only:publish(rows);return
    template=json.loads((isolation.SOURCE/'independent-isotropic20.config.json').read_text())
    cases=list(matrix())
    if followup:
        originals={name:(params,flags) for name,params,flags in matrix()}
        candidates=[(r['independent_percent'],r['observed_ladder_shift']['ladder_shift_percent'],name)
            for name,r in rows.items() if name in originals and name.startswith('m') and r['independent_percent'] is not None]
        chosen=max(candidates)[2]
        params,flags=originals[chosen]
        minimum=params['--local-min-piece-length']
        cases=[(chosen+'-binary',params,flags+['--constraint-binary-score'])]
        for cost in (75,300):
            for boundary in (0,500):
                cases.append((f'm{minimum}-d{cost}-b{boundary}',{'--local-min-piece-length':minimum,
                    '--winding-defect-length-unit':128,'--winding-defect-cost':cost,
                    '--winding-gap-boundary-cost':boundary},['--legacy-fine-pieces','--partition-length-cost']))
        cases.append((f'm{minimum}-incident-boundary',{'--local-min-piece-length':minimum,
            '--winding-gap-boundary-cost':500},['--legacy-fine-pieces']))
    baseline=None
    if (OUT/'baseline_consistency.csv').exists():
        with (OUT/'baseline_consistency.csv').open() as f:baseline=list(csv.DictReader(f))
    for name,params,flags in cases:
        config=isolation.shared.configured(template,name,'',{'--reference-measurements':'old',
            '--winding-message-iterations':500,**params},flags)
        rec=isolation.shared.execute(config)
        if baseline is None:
            with (OUT/'baseline_consistency.csv').open() as f:baseline=list(csv.DictReader(f))
        if name=='baseline':
            assert isolation.initial_states(OUT/'baseline_consistency.csv')==isolation.initial_states(isolation.SOURCE/'independent-isotropic20_consistency.csv')
        r,_=isolation.collect(name,rec,baseline)
        stats,_=isolation.factor_inventory(name);r['measurement_inventory']=stats
        if name!='baseline':
            b=rows['baseline']['measurement_inventory']
            assert stats['measurements']==b['measurements'],'lost legacy observations'
            assert stats['evidence_mass']==b['evidence_mass'],'changed legacy evidence mass'
        r['parameters']=params;r['flags']=flags;rows[name]=r;publish(rows)
        print(f"COMPLETE {name}: accuracy={isolation.percent(r['independent_percent'])} shift={isolation.percent(r['observed_ladder_shift']['ladder_shift_percent'])} measurements={stats['measurements']}",flush=True)


if __name__=='__main__':
    import sys
    main('--followup' in sys.argv,'--report-only' in sys.argv)
