"""Bounded native source-memory experiments on the frozen iso20 graph."""
import csv
import json
from iso20_gap_tuning import collect_trial, MIN_SOURCE
from iso20_stable_followups import OUT as PREVIOUS
from iso20_stable_gap import blend_parameters
from reviewed_bp_independent import initial_states
import strict_reverse_crop as shared
from lasagna.inference_provenance import atomic_write

OUT=MIN_SOURCE.parent/'20261002_iso20_source_memory_v1'
REPORT=shared.REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-02-iso20-source-memory.md'


def publish(rows):
    atomic_write(OUT/'results.json',rows,durable=False)
    atomic_write(REPORT.with_suffix('.json'),rows,durable=False)
    lines=['# Coarse/fine source-memory BP','',
        'Paris4 1024³ crop, baseXYZ half-open [10240,22016,6144]–[11264,23040,7168]; '
        'hendrik_crop1 references from /home/hendrik/test_datasets/2026-09-01_fiber_stack2/. '
        'Frozen source: /ephemeral/hendrik/fiber_crop_bench/20261002_strict_reverse_crop_v1/beam-isotropic20.zarr. '
        'Raw40500 model SHA256 9d3bb1c0cd5b0ee58d16d9b2bdcec931faec675421e6e2c707f7fd10ce059f7e.','',
        'Frozen raw40500 iso20, fine orthogonal-event minimum64; RelWithDebInfo,128 CPU workers, '
        '500 iterations. No inference/retracing, no interval repair. All rows inherit100% replay '
        '(0 ordinary failures,6 filtered stops,93.7463% retained). Accuracy/26, shift observed-only; '
        'ties use deterministic rank1. Local-event constraint-removal and oracle metrics remain '
        'unavailable due the previously documented evaluator calibration issue.','',
        '| Method | Replay % | Constraint score % | Oracle % | Non-oracle % | Shift % (N) | Disabled length % | Gaps | Boundaries | Memory-projected pieces | Seconds |',
        '|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|',
        '| Historical overlap512/128 | 100.00% | −25.24% | 80.77% | 50.00% | 83.33% (24) | — | — | — | — | — |',
        '| Prior fine64 tuned | 100.00% | Unavailable | Unavailable | 42.31% | 62.50% (24) | 16.054% | 561 | 61 | 0 | — |']
    for name,r in rows.items():
        shift=r['observed_ladder_shift']
        lines.append(f"| {name} | 100.00% | Unavailable | Unavailable | {r['independent_percent']:.2f}% | "
            f"{shift['ladder_shift_percent']:.2f}% ({shift['references']}) | {r['disabled_percent']:.3f}% | "
            f"{r['gap_runs']} | {r['boundaries']} | {r['memory'].get('projected_defects',0)} | {r['seconds']:.2f} |")
    lines+=['','## Formulation and limitations','',
        '- Each fine piece owns its evidence and physical-length disabling cost once. Windows repeat only prior coupling.',
        '- Window membership uses piece arc midpoint in half-open windows, anchored at each contiguous compatible source-run start; window512, stride512 or384. Long fine pieces are not forcibly split.',
        '- All retained labels within a hard window must agree even across disabled pieces. Overlapping windows can transitively couple longer spans when their shared pieces survive.',
        '- Chain DP remembers the last active class/winding through disabled spans. A different next retained label pays the break cost; each disabled run pays opening once. No latent-label multiplicity, including all-disabled chains.',
        '- Original hard adjacent active/active continuity remains: a winding break requires a disabled gap, not two differently labeled adjacent retained pieces.',
        '- Chains split at geometry gaps and different effective integer gauges; prepass-Defect nodes do not introduce gauge links. No new measurements, prepass changes or gauge merges.',
        '- Constant-window decoding monotonically disables conflicting labels, preserving gauges/fixed states. These additional projection defects are reported, not hidden as successful joint optimization.',
        '- Exact conditional sequence messages do not make the overall loopy BP globally exact. The old overlapping extraction has a different prepass/aggregation; these are not claimed to be identical formulations.',
        '- Objective includes memory factors. Reported active_components in JSON describes ordinary retained factors, not additional memory connectivity.',
        '- Pilot implementation uses O(N*K²) native sequence messages, parallel across factors; effective overlapping-coupling span distributions are not yet instrumented.',
        '- One run per setting; wall time is observational, not an isolated speed benchmark. Full ladders, missing/tied refs and provenance in adjacent JSON.',
        '',f'Artifacts: `{OUT}`. Approved runner action `iso20_source_memory`.']
    if len(rows)==7:
        lines+=['','## Result','',
            'No improvement. The rebuilt control exactly reproduces the prior fine-piece states. '
            'Every memory variant loses reference8 in addition to already missing4/12, increases '
            'disabled length, and worsens both benchmark scores. Extra source-memory feasibility '
            'projection disabled0pieces in every run, so the regression is not caused by that new '
            'final projection. All prepass assignments remain identical. Opening1000 worsens '
            'the chains further. Stronger correlation alone does not recover the old overlapping '
            'model behavior on this fine-piece graph. This does not prove all hierarchical models '
            'fail: coarse extraction/prepass is different and was not replaced in this trial. '
            'No production defaults promoted; these remain opt-in experiments.',
            '', 'All six new variants took35.1–36.6seconds end-to-end versus32.7seconds for control '
            '(one run/configuration). No larger iteration budgets or rejected repair were used.']
    REPORT.write_text('\n'.join(lines)+'\n')


def main(report_only=False):
    OUT.mkdir(parents=True,exist_ok=True);shared.OUT=OUT
    if report_only:
        publish(json.loads((OUT/'results.json').read_text()))
        return
    rows={}
    template=json.loads((MIN_SOURCE/'independent-min64.config.json').read_text())
    with (MIN_SOURCE/'independent-min64_consistency.csv').open() as f:
        expected=[p['winding_prepass_class'] for p in csv.DictReader(f)]
    matrix=[('control',0,0,None,0),('window512',512,0,None,0),('overlap512-128',512,128,None,0)]
    matrix += [(f'chain-break{b}-open{o}',-1,0,b,o) for b in (500,2000) for o in (0,1000)]
    for name,window,overlap,brk,opening in matrix:
        updates={'--winding-message-iterations':500,'--winding-gap-boundary-cost':500,
            '--winding-damping':.5,'--winding-source-window':window,
            '--winding-source-overlap':overlap,'--winding-source-opening':opening}
        updates.update(blend_parameters(1.,150))
        if brk is not None:updates['--winding-source-break']=brk
        config=shared.configured(template,name,'',updates)
        rec=shared.execute(config);r=collect_trial(OUT,name,rec)
        assert r.pop('prepass_classes')==expected
        if name=='control':
            assert initial_states(OUT/f'{name}_consistency.csv')==initial_states(
                PREVIOUS/'a1-d0p5-b500-c150-i500-m64-best0-r0-s0_consistency.csv')
        log=(OUT/f'{name}.log').read_text()
        r['memory']={}
        for line in log.splitlines():
            if line.startswith('source_memory '):
                r['memory']=dict(x.split('=',1) for x in line.split()[1:])
        r.update(window=window,overlap=overlap,break_cost=brk,opening=opening,
            prepass_identical=True,iterations=500)
        rows[name]=r;publish(rows)
        print(f"COMPLETE {name}: accuracy={r['independent_percent']:.4f}% "
              f"shift={r['observed_ladder_shift']['ladder_shift_percent']:.4f}% "
              f"disabled={r['disabled_percent']:.4f}% seconds={r['seconds']:.1f}",flush=True)


if __name__=='__main__':
    import sys
    main('--report-only' in sys.argv[1:])
