"""Conservative two-sided evidence trials using shared native benchmark machinery."""
import csv
import json
import iso20_constraint_isolation as shared
from lasagna.inference_provenance import atomic_write

BASE=shared.OUT.parent/'20261002_iso20_legacy_fine_v1'
OUT=BASE.parent/'20261002_iso20_orthogonal_probes_v1'
REPORT=shared.REPORT.with_name('2026-10-02-iso20-orthogonal-probes.md')


def matrix():
    yield 'baseline',{},[]
    for span in (16,32,64):
        for rule in ('median','consensus'):
            yield f'{rule}-{span}',{'--orthogonal-probe-span':span},(['--orthogonal-probe-consensus'] if rule=='consensus' else [])
    for span in (32,64):
        yield f'half-median-{span}',{'--orthogonal-probe-span':span,'--orthogonal-probe-blend':.5},[]


def followup_matrix():
    for w in (0,.05,.2):
        yield 'parallel-'+str(float(w)).replace('.','p'),{'--winding-weights':f'0,0,0,{w},0'},[]
    yield 'normal-cosine',{'--winding-normal-confidence':'cosine'},[]


def inventory(name,base):
    changed=sign_changed=0;kept={}
    fields=('piece_a','piece_b','parallel','perpendicular','original_signed_parallel_delta',
        'effective_parallel_winding_distance','evidence_weight','arc_a','arc_b')
    with (OUT/f'{name}_winding_factors.csv').open() as f:
        for r in csv.DictReader(f):
            i=r['constraint'];kept[i]=tuple(r[k] for k in fields)+(r['original_signed_delta'],)
            if base is not None:
                assert kept[i][:-1]==base[i][:-1],f'non-target measurement changed: {i}'
                a,b=base[i][-1],r['original_signed_delta']
                changed+=a!=b
                if a!='NA' and b!='NA':sign_changed+=(float(a)>0)-(float(a)<0)!=(float(b)>0)-(float(b)<0)
    if base is not None:assert kept.keys()==base.keys()
    return dict(changed_targets=changed,changed_signs=sign_changed,measurements=len(kept)),kept


def publish(rows):
    atomic_write(OUT/'results.json',rows,durable=False);atomic_write(REPORT.with_suffix('.json'),rows,durable=False)
    lines=['# Conservative two-sided orthogonal evidence','',
        'Frozen raw40500 iso20 Paris4 1024 crop; legacy-to-fine256, length150/128, boundary0. '
        '128CPU workers, RelWithDebInfo,500 iterations, old reference readout. No inference/retracing. '
        'Four exact ±span probes: move A against centerB and move B against centerA. '
        'Require all four finite probes in the center normal component, otherwise keep original. '
        'Median includes center; consensus additionally requires all five signed half-winding quantizations agree. '
        'Adapt after legacy acceptance/cutoff. No rejection or parallel-estimator changes; same cuts, scores, weights and owners.','',
        '| Variant | Exact accuracy ↑ | Shift ↑ (N) | Disabled length | Changed targets | Changed signs | Seconds |',
        '|---|---:|---|---:|---:|---:|---:|']
    for name,r in rows.items():
        s=r['observed_ladder_shift'];d=r['target_inventory']
        lines.append(f"| {name} | {shared.percent(r['independent_percent'])} | {shared.percent(s['ladder_shift_percent'])} ({s['references']}) | "
            f"{r['disabled_percent']:.2f}% | {d['changed_targets']} | {d['changed_signs']} | {r['seconds']:.1f} |")
    lines+=['','Accuracy denominator26; observed shift skips missing references. Full ladders/ties/missing in JSON. '
        'Replay100% inherited; old overlap50%/83.33%, oracle80.77%,constraint−25.24% historical only. '
        'No new oracle/removal runs. One tuning crop, no default change.','',
        f'Artifacts: `{OUT}`. Approved runner action `iso20_orthogonal_probes`; exact commands/hashes in run.json. '
        '`report_only:true` refreshes saved results. `followup:true` tests finite parallel magnitude weights0/0.05/0.2 '
        '(baseline0.1) and normal-alignment cosine confidence, on the unmodified port without probes. '
        'Hard ordering signs stay enabled. These objective settings also apply to independent reference ranking; '
        'they are end-to-end objective trials, not readout-isolated graph experiments.']
    if 'winner-consensus32' in rows:
        lines+=['','## Findings','',
            '- All eight probe-only variants retain50.00% exact accuracy/87.50% observed shift. They change3271–20724 scalar targets but only1–7 signs; most solves have different piece states, so this is not an inactive flag. '
            'They add no benchmark benefit, even when combined with the better objective.',
            '- Soft parallel one-winding distance weight0.05 (was0.1) gives61.54% accuracy/87.50% shift:16/26 exact references, three more than baseline. '
            'Exact repeat reproduces all solved piece states and reference ladder. Missing references4/12 unchanged, no ties or unbounded rows.',
            '- Weight0 gives3.85%/20.83%; weight0.2 and cosine alignment confidence give50.00%/83.33%. Retain a weak magnitude term; hard signs alone are insufficient here.',
            '- Combining winner with median64 or consensus32 probes keeps61.54%/87.50%; prefer simpler no-probe winner. Disabled length6.09%, vs6.04% prior port and10.49% old overlap.',
            '- Current leading tested independent-score candidate: legacy-fine minimum256, length150/128, boundary0, soft original measurements, '
            '`--winding-weights 0,0,0,0.05,0`, hard signs both, old reference readout,500 iterations. '
            'The weight also changes reference ranking costs, so this gain is not proven solely attributable to the initial graph solve.',
            '-16 valid reported runs including repeat.447 native cases +48 Python checks passed. Opt-in only, no production defaults changed. '
            'One tuning crop and no new oracle/removal scores; not held-out validation.',
            '', 'Reproduce with approved runner action iso20_orthogonal_probes: initial matrix, followup:true, confirmation:true, or report_only:true. '
            'Build/tests via the same runner action build. Per-run command and binary/input hashes are retained.']
    REPORT.write_text('\n'.join(lines)+'\n')


def main(report_only=False,followup=False,confirmation=False):
    OUT.mkdir(parents=True,exist_ok=True);shared.OUT=OUT;shared.shared.OUT=OUT
    if report_only:publish(json.loads((OUT/'results.json').read_text()));return
    template=json.loads((BASE/'m256-length.config.json').read_text())
    rows=json.loads((OUT/'results.json').read_text()) if followup or confirmation else {}
    baseline=None;base_factors=None
    cases=list(matrix())
    if followup or confirmation:
        with (OUT/'baseline_consistency.csv').open() as f:baseline=list(csv.DictReader(f))
        _,base_factors=inventory('baseline',None)
        # Native output paths use stem(): decimal dots would alias output files.
        rows.pop('parallel-0',None)
        cases=list(followup_matrix())
        if confirmation:
            weight={'--winding-weights':'0,0,0,0.05,0'}
            cases=[('winner-repeat',weight,[]),
                ('winner-median64',{**weight,'--orthogonal-probe-span':64},[]),
                ('winner-consensus32',{**weight,'--orthogonal-probe-span':32},['--orthogonal-probe-consensus'])]
    for name,params,flags in cases:
        config=shared.shared.configured(template,name,'',params,flags)
        rec=shared.shared.execute(config)
        if baseline is None:
            with (OUT/'baseline_consistency.csv').open() as f:baseline=list(csv.DictReader(f))
            assert shared.initial_states(OUT/'baseline_consistency.csv')==shared.initial_states(BASE/'m256-length_consistency.csv')
        r,_=shared.collect(name,rec,baseline)
        assert r['same_piece_identities'] and r['prepass_changes']==0
        stats,factors=inventory(name,base_factors)
        if base_factors is None:base_factors=factors
        r['target_inventory']=stats;rows[name]=r;publish(rows)
        if name=='winner-repeat':
            assert shared.initial_states(OUT/f'{name}_consistency.csv')==shared.initial_states(OUT/'parallel-0p05_consistency.csv')
            assert r['ladder']==rows['parallel-0p05']['ladder']
        print(f"COMPLETE {name}: {shared.percent(r['independent_percent'])}/{shared.percent(r['observed_ladder_shift']['ladder_shift_percent'])} changed={stats}",flush=True)


if __name__=='__main__':
    import sys
    main('--report-only' in sys.argv,'--followup' in sys.argv,'--confirmation' in sys.argv)
