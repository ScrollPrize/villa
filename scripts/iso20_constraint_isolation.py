"""One-axis ablations on frozen old-overlap BP; optimization remains native."""
import csv
import io
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import strict_reverse_crop as shared
from reviewed_bp_independent import initial_states, readout_metrics
from lasagna.inference_provenance import atomic_write

SOURCE=Path('/ephemeral/hendrik/fiber_crop_bench/20261002_strict_reverse_crop_v1')
OUT=SOURCE.parent/'20261002_iso20_constraint_isolation_v1'
REPORT=shared.REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-02-iso20-constraint-isolation.md'


def factor_inventory(name):
    with (OUT/f'{name}_consistency.csv').open() as f:
        sources={p['piece']:p['original_trace'] for p in csv.DictReader(f)}
    pairs={};count=0;mass=0.
    with (OUT/f'{name}_winding_factors.csv').open() as f:
        for c in csv.DictReader(f):
            if sources[c['piece_a']]==sources[c['piece_b']]:continue
            count+=1;w=float(c['evidence_weight']);mass+=w
            p=pairs.setdefault((c['piece_a'],c['piece_b']),[0.,0.,0.,0.,0,0])
            parallel=float(c['parallel'])>float(c['perpendicular'])
            p[0 if parallel else 1]+=w
            field='effective_signed_parallel_delta' if parallel else 'effective_perpendicular_signed_delta'
            if c[field]!='NA':
                d=float(c[field]);p[2]+=w*d;p[3]+=w
                hard=c['hard_parallel_sign' if parallel else 'hard_perpendicular_sign']=='1'
                if hard and d:p[4 if d>0 else 5]+=1
    return dict(measurements=count,evidence_mass=mass,pairs=len(pairs),
        contradictory_hard_sign_pairs=sum(p[4]>0 and p[5]>0 for p in pairs.values())),pairs


_BASE_FACTORS=None
def init_factor_analysis():
    global _BASE_FACTORS
    _BASE_FACTORS=factor_inventory('baseline')[1]


def analyze_factors(name):
    stats,pairs=factor_inventory(name);base=_BASE_FACTORS
    common=pairs.keys() & base.keys()
    stats.update(common_pairs=len(common),lost_pairs=len(base.keys()-pairs.keys()),new_pairs=len(pairs.keys()-base.keys()),
        dominant_class_changes=sum((pairs[k][0]>pairs[k][1])!=(base[k][0]>base[k][1]) for k in common),
        signed_mean_target_changes_gt_quarter=sum(abs(pairs[k][2]/pairs[k][3]-base[k][2]/base[k][3])>.25
            for k in common if pairs[k][3] and base[k][3]))
    return name,stats


def matrix():
    yield 'baseline',{},[],'Old overlapping baseline, explicit old reference extraction'
    yield 'reference-events',{'--reference-measurements':'events'},[],'Readout-only: certified source reference events'
    for name,updates in [
        ('closest',{'--constraint-parallel-estimator':'closest'}),
        ('certify',{'--constraint-certification':'on'}),
        ('closest-certify',{'--constraint-parallel-estimator':'closest','--constraint-certification':'on'})]:
        yield name,updates,[],'Old candidates/scorer; distance estimator and/or winding probe check'
    for estimator in ('median','closest'):
        for certify in ('off','on'):
            yield f'local-scorer-{estimator}-{certify}',{'--constraint-parallel-estimator':estimator,
                '--constraint-certification':certify},['--constraint-local-scorer'],'Old piece-pair candidates; local geometric scorer'
    events=['--local-constraint-events','--local-overlap-pieces']
    yield 'events-overlap',{},events,'Full new source measurements on exact old overlap layout'
    yield 'events-unit',{},events+['--local-unit-evidence'],'Same as events-overlap, unit evidence'
    yield 'events-unit-one',{},events+['--local-unit-evidence','--local-one-per-pair'],'Unit evidence, nearest event per owner pair'
    yield 'events-normalized-one',{},events+['--local-one-per-pair'],'Normalized evidence, nearest event per owner pair'
    yield 'events-refine0',{'--local-refinement-rounds':0},events,'Full source events without transition refinement'
    yield 'events-median',{'--constraint-parallel-estimator':'median'},events,'Full source events with median estimator'
    yield 'events-no-winding-cert',{'--constraint-certification':'off'},events,'Full source events without winding consistency rejection'
    yield 'events-replicated',{},events+['--local-replicate-overlap'],'All overlapping owners; conserved total event mass'
    for suffix,freeze in [('classes',['--freeze-piece-classes']),('mass',['--freeze-piece-mass']),
                          ('classes-mass',['--freeze-piece-classes','--freeze-piece-mass'])]:
        yield 'events-frozen-'+suffix,{'--frozen-piece-prior':str(OUT/'baseline-prior.json')},events+freeze,'Full new events, frozen baseline '+suffix
    yield 'old-boundary500',{'--winding-gap-boundary-cost':500},[],'Old graph, boundary prior only'
    for boundary in (0,500):
        yield f'old-length150-boundary{boundary}',{'--winding-gap-boundary-cost':boundary,
            '--winding-defect-length-unit':128,'--winding-defect-cost':150},['--partition-length-cost'],'Old graph, coverage-partitioned physical-length unary'
    yield 'fine64-old-readout',{'--local-min-piece-length':64},['--local-constraint-events'],'Fine64 original costs, old reference extraction'
    yield 'fine64-tuned-old-readout',{'--local-min-piece-length':64,'--winding-gap-boundary-cost':500,
        '--winding-defect-length-unit':12800/150,'--winding-defect-cost':100,'--winding-defect-length-blend':1},[
        '--local-constraint-events'],'Fine64 tuned costs, old reference extraction'


def key(row):
    return (int(row['original_trace']),float(row['begin_arc_base']),float(row['end_arc_base']))


def collect(name,rec,baseline):
    log=(OUT/f'{name}.log').read_text()
    try:
        result=readout_metrics(log)
    except ValueError as exc:
        if 'multiple independent gauges' not in str(exc):raise
        summaries=list(csv.DictReader(io.StringIO(log.split('independent_reference_summary\n')[1].split('independent_reference_candidates')[0])))
        result=dict(independent_percent=None,observed_ladder_shift=dict(ladder_shift_percent=None,references=None),
            missing_references=sorted({r['reference'] for r in summaries}-{r['reference'] for r in summaries if int(r['evaluated'])>0},key=float),
            readout_unavailable=str(exc),gauge_summaries=summaries)
    with (OUT/f'{name}_consistency.csv').open() as f:pieces=list(csv.DictReader(f))
    classes={key(p):p['winding_prepass_class'] for p in pieces}
    baseclasses={key(p):p['winding_prepass_class'] for p in baseline}
    common=classes.keys() & baseclasses.keys()
    result.update(seconds=rec['wall_seconds'],pieces=len(pieces),same_piece_identities=classes.keys()==baseclasses.keys(),
        prepass_changes=sum(classes[k]!=baseclasses[k] for k in common),common_pieces=len(common),
        prepass_counts={c:sum(v==c for v in classes.values()) for c in sorted(set(classes.values()))},
        prior_mass=sum(float(p['winding_disabling_measure']) for p in pieces),
        ordinary_graph=[s for s in log.splitlines() if s.startswith(('fiber BP component selection','constraint_ablation '))],
        readout_diagnostics=[s for s in log.splitlines() if s.startswith('reference cross source events:')],
        state_sha256=hashlib.sha256(json.dumps(initial_states(OUT/f'{name}_consistency.csv')).encode()).hexdigest())
    # Coverage-normalized retained length: integrate intervals once, including overlap.
    total=dead=0.
    by_source={}
    for p in pieces:by_source.setdefault(p['original_trace'],[]).append(p)
    for group in by_source.values():
        cuts=sorted({float(p[k]) for p in group for k in ('begin_arc_base','end_arc_base')})
        for a,b in zip(cuts,cuts[1:]):
            cover=[p for p in group if float(p['begin_arc_base'])<=a and float(p['end_arc_base'])>=b]
            if cover:
                total+=b-a
                dead+=(b-a)*sum(p['winding_final_class'] not in ('h','v') for p in cover)/len(cover)
    result['disabled_percent']=100*dead/total
    return result,pieces


def publish(rows):
    for r in rows.values():
        if 'gauge_summaries' in r:
            summaries=r['gauge_summaries']
            r['missing_references']=sorted({s['reference'] for s in summaries}-{s['reference'] for s in summaries if int(s['evaluated'])>0},key=float)
    atomic_write(OUT/'results.json',rows,durable=False);atomic_write(REPORT.with_suffix('.json'),rows,durable=False)
    lines=['# Constraint isolation on old overlap512/128','',
        'Frozen raw40500 iso20 Paris4 1024³ crop [10240,22016,6144]–[11264,23040,7168], '
        'hendrik_crop1,128CPU workers, RelWithDebInfo,500iterations. No inference/retracing or repair. '
        'Replay100% inherited for all rows. Old oracle80.77%, constraint score−25.24% retained as historical '
        'baseline; this first isolation matrix measures independent readout, not new oracle/removal runs. '
        'Accuracy/26; observed shift excludes missing refs and reports N; tied rank1 is not unique.','',
        '| Variant | Non-oracle % | Shift % (N) | Disabled length % | Pieces | Changed prepass / common | Missing refs | Seconds |',
        '|---|---:|---|---:|---:|---|---|---:|']
    for name,r in rows.items():
        s=r['observed_ladder_shift'];lines.append(f"| {name} | {percent(r['independent_percent'])} | "
            f"{percent(s['ladder_shift_percent'])} ({s['references']}) | {r['disabled_percent']:.2f}% | "
            f"{r['pieces']} | {r['prepass_changes']}/{r['common_pieces']} | {r['missing_references']} | {r['seconds']:.1f} |")
    lines+=['','## Row definitions','',*[f'- **{name}:** {r["description"]}.' for name,r in rows.items()],
        '','## Controls and caveats','',
        '- All graph rows force the legacy reference extraction independently of graph options. Reference-events is the deliberate readout-only exception; initial states must exactly match baseline.',
        '- Local scorer bundles short one-sided tangents, reciprocal projections and geometric probe checks. Winding consistency and closest-vs-median are separate switches. Refined points outside their old owning piece are rejected.',
        '- Event-overlap layout exactly matches old intervals before any graph selection. Unique owner is last-start covering interval; replicated case conserves mass across all covering owner pairs, not historical full-mass duplication.',
        '- Event multiplicity/weights affect both factors and incidence-based disabling cost. Frozen-mass/classes controls explicitly separate these coupled effects. Frozen identities must match original trace and exact source arc endpoints.',
        '- Disabled length splits overlapping arc coverage equally, not double-counting physical length. This is a coverage-normalized fraction, not union of all disabled intervals.',
        '- Reference observations use the old method but necessarily depend on current surviving pieces. Changed reference coverage is reported; this is not identical cross-evidence across different layouts.',
        '- N/A means multiple independent gauges prevented a single comparable reference ladder; all gauge summaries are retained in JSON, not silently combined.',
        '- One run per variant, wall times observational. No production-default changes. Full command, binary/shared-library and input hashes live in each run.json.',
        '',f'Artifacts: `{OUT}`. Approved runner action `iso20_constraint_isolation`; report_only refresh supported.']
    if any('factor_inventory' in r for r in rows.values()):
        lines+=['','## Prepared ordinary measurement inventory','',
            'Excludes same-source continuation. Includes prepared observations subsequently pruned from the active fixed-orientation grid; these are not active solver-factor counts. '
            'Pair comparisons require identical piece identities, not fine layouts. Mean targets are evidence-weighted quantized signed targets.','',
            '| Variant | Measurements | Evidence mass | Pairs | Lost baseline pairs | Changed pair class | Target changes >0.25 | Contradictory hard-sign pairs |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,r in rows.items():
            if 'factor_inventory' not in r:continue
            f=r['factor_inventory'];lines.append(f"| {name} | {f['measurements']} | {f['evidence_mass']:.2f} | "
                f"{f['pairs']} | {f['lost_pairs']} | {f['dominant_class_changes']} | {f['signed_mean_target_changes_gt_quarter']} | {f['contradictory_hard_sign_pairs']} |")
    if 'binary-frozen' in rows and 'factor_inventory' in rows['baseline']:
        base=rows['baseline']['factor_inventory'];local=rows['local-scorer-median-off']['factor_inventory']
        cert=rows['certify']['factor_inventory'];events=rows['events-overlap']['factor_inventory']
        lines+=['','## Findings','',
            '- Old baseline, rebuilt baseline, frozen-prior baseline, and readout-only initial states match exactly.',
            '- New reference readout alone changes50.00% to46.15% accuracy; shift remains83.33%. It is not the main regression.',
            f'- Local scorer on old candidates loses {100*local["lost_pairs"]/base["pairs"]:.2f}% of baseline piece-pair links; '
            'accuracy19.23%/shift16.67%. Freezing old classes gives identical scores; freezing old disabling mass gives19.23%/20.83%.',
            f'- Winding consistency alone removes {100*cert["lost_pairs"]/base["pairs"]:.2f}% of baseline links; '
            '26.92%/62.50%, or30.77%/70.83% with frozen costs. Closest estimator also regresses46.15%/70.83% even with frozen costs.',
            f'- Full source events have {events["measurements"]:,} measurements but only {events["pairs"]:,} unique owner pairs '
            f'versus {base["pairs"]:,} baseline pairs: repetition is not equivalent to coverage. '
            'Frozen disabling mass helps34.62%/29.17% to38.46%/58.33%, but does not recover baseline.',
            '- Binary orientation scores alone are not harmful here:53.85%/79.17%, also with frozen classes/costs. '
            'Old graph + physical-length150/128 + boundary500 gives the same aggregate scores (not necessarily the same ladder). '
            'Each gains one exact reference but loses one shift-score step versus baseline; neither dominates both metrics.',
            '- Fine64 tuned with old reference readout remains38.46%/45.83%. The new measurement rules regress before changing the layout, '
            'so these results do not establish that fine splitting itself is inherently worse.',
            '- Scope limitation: local tangent, reciprocal projection and geometric probes remain one bundled scorer; '
            'this matrix localizes its regression but does not isolate every internal geometric test. '
            'New oracle/removal metrics were not run. Two frozen-both controls have multiple active gauges and no single comparable ladder.',
            '- No production defaults changed. Tested on one tuning crop, not held-out validation. '
            '445 native cases and45 Python checks passed. Native solve times are individual runs, not repeated speed benchmarks.',
            '', 'Reproduction: set scripts/training_failure_mining_control.json to action iso20_constraint_isolation '
            '(supplement:true for supplemental controls; analyze:true for saved CSV analysis; report_only:true for report). '
            'Run `bash /home/hendrik/villa3/scripts/run_training_failure_mining`. Build/test uses the same runner with action build.']
    REPORT.write_text('\n'.join(lines)+'\n')


def percent(value):
    return 'N/A' if value is None else f'{value:.2f}%'


def main(report_only=False,supplement=False,analyze=False):
    OUT.mkdir(parents=True,exist_ok=True);shared.OUT=OUT
    if analyze:
        rows=json.loads((OUT/'results.json').read_text())
        names=[name for name,r in rows.items() if r['same_piece_identities']]
        with ProcessPoolExecutor(max_workers=4,initializer=init_factor_analysis) as pool:
            for name,stats in pool.map(analyze_factors,names):rows[name]['factor_inventory']=stats
        publish(rows);return
    if report_only:publish(json.loads((OUT/'results.json').read_text()));return
    rows=json.loads((OUT/'results.json').read_text()) if supplement else {}
    baseline=None
    if supplement:
        with (OUT/'baseline_consistency.csv').open() as f:baseline=list(csv.DictReader(f))
    template=json.loads((SOURCE/'independent-isotropic20.config.json').read_text())
    cases=matrix() if not supplement else [
        ('baseline-rebuilt',{},[],'Rebuilt supplemental binary, old baseline'),
        ('baseline-frozen',{'--frozen-piece-prior':str(OUT/'baseline-prior.json')},
            ['--freeze-piece-classes','--freeze-piece-mass'],'Frozen-prior zero-change control'),
        ('binary-score',{},['--constraint-binary-score'],'Old measured geometry/distances, binary rather than soft orientation scores'),
        ('binary-frozen',{'--frozen-piece-prior':str(OUT/'baseline-prior.json')},
            ['--constraint-binary-score','--freeze-piece-classes','--freeze-piece-mass'],'Binary scores, fixed baseline classes and disabling masses'),
        ('certify-frozen-mass',{'--constraint-certification':'on','--frozen-piece-prior':str(OUT/'baseline-prior.json')},
            ['--freeze-piece-mass'],'Consistency rejection only, original disabling costs'),
        ('closest-frozen-mass',{'--constraint-parallel-estimator':'closest','--frozen-piece-prior':str(OUT/'baseline-prior.json')},
            ['--freeze-piece-mass'],'Closest estimator only, original disabling costs')]
    if supplement:
        for suffix,flags in [('classes',['--freeze-piece-classes']),('mass',['--freeze-piece-mass']),
                             ('both',['--freeze-piece-classes','--freeze-piece-mass'])]:
            cases.append(('local-scorer-frozen-'+suffix,{'--constraint-parallel-estimator':'median',
                '--constraint-certification':'off','--frozen-piece-prior':str(OUT/'baseline-prior.json')},
                ['--constraint-local-scorer',*flags],'Local geometric scorer only, frozen baseline '+suffix))
    for name,updates,extra,description in cases:
        params={'--winding-message-iterations':500,'--reference-measurements':'old',**updates}
        config=shared.configured(template,name,'',params,extra)
        if '--frozen-piece-prior' in params:config['input_files'].append(str(OUT/'baseline-prior.json'))
        rec=shared.execute(config)
        if baseline is None:
            with (OUT/'baseline_consistency.csv').open() as f:baseline=list(csv.DictReader(f))
            assert initial_states(OUT/'baseline_consistency.csv')==initial_states(SOURCE/'independent-isotropic20_consistency.csv'),'old baseline changed'
            atomic_write(OUT/'baseline-prior.json',dict(version=1,trace_artifact=template['commands'][0]['args'][1],
                pieces=[dict(original_trace=int(p['original_trace']),begin_arc_base=float(p['begin_arc_base']),
                    end_arc_base=float(p['end_arc_base']),mass=float(p['winding_disabling_measure']),
                    **{'class':p['winding_prepass_class']}) for p in baseline]),durable=False)
        r,pieces=collect(name,rec,baseline);r['description']=description
        if name=='reference-events':
            assert initial_states(OUT/f'{name}_consistency.csv')==initial_states(OUT/'baseline_consistency.csv')
        if name in ('baseline-rebuilt','baseline-frozen'):
            assert initial_states(OUT/f'{name}_consistency.csv')==initial_states(OUT/'baseline_consistency.csv')
        if '--freeze-piece-classes' in extra:assert r['prepass_changes']==0
        if '--freeze-piece-mass' in extra:
            base={key(p):float(p['winding_disabling_measure']) for p in baseline}
            assert all(float(p['winding_disabling_measure'])==base[key(p)] for p in pieces)
        if name.startswith('events-') or name.startswith('local-scorer') or name in ('closest','certify','closest-certify'):
            assert r['same_piece_identities'],'unexpected piece-layout change'
        rows[name]=r;publish(rows)
        print(f"COMPLETE {name}: accuracy={percent(r['independent_percent'])} shift={percent(r['observed_ladder_shift']['ladder_shift_percent'])} prepass_changes={r['prepass_changes']}",flush=True)


if __name__=='__main__':
    import sys
    main('--report-only' in sys.argv[1:],'--supplement' in sys.argv[1:],'--analyze' in sys.argv[1:])
