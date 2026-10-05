"""Frozen post-fw/bw whole-piece suppression, native geometry and existing BP."""
import copy
import json
import re
from pathlib import Path
import strict_reverse_crop as shared
from reviewed_bp_independent import readout_metrics
from iso20_probe_removal import oracle_args, pruning_counts
from vesuvius.neural_tracing.fiber_trace_3d.crop_benchmark import bp_metrics
from lasagna.inference_provenance import atomic_write

SOURCE=shared.OUT
WIN=SOURCE.parent/'20261003_iso20_iterative_v1'
OUT=SOURCE.parent/'20261003_iso20_post_overlap_v1'
REPORT=shared.REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-03-post-overlap'
VARIANTS=[(f'{rank}-{n}x{t}-f{int(f*100)}',rank,n,t,f)
          for n,t in ((10,20),(20,40)) for f in (.8,.5) for rank in ('length','cost','presence')]
LOW=False
HISTORY=OUT


def setup_low():
    global OUT,REPORT,VARIANTS,LOW
    LOW=True
    OUT=SOURCE.parent/'20261003_iso20_post_overlap_low_v1'
    REPORT=shared.REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-03-post-overlap-low'
    VARIANTS=[(f'{rank}-{n}x{t}-f{int(f*100)}',rank,n,t,f)
              for n,t in ((10,20),(20,40),(10,10),(20,20))
              for f in (0.,.1,.2) for rank in ('length','cost','presence')]


def filter_config(template,name,rank,n,t,f,source_trace=None):
    source_trace=Path(source_trace) if source_trace is not None else SOURCE/'beam-isotropic20.zarr'
    config=copy.deepcopy(template)
    args=template['commands'][0]['args']
    normal=args[args.index('--normal-manifest')+1]
    fiber=args[args.index('--fiber-manifest')+1]
    config['output_root']=str(OUT)
    config['commands']=[{'name':name,'binary':'vc_fiber_trace_chunk','args':[
        'deduplicate',str(source_trace),'--output',str(OUT/f'{name}.zarr'),
        '--normal-manifest',normal,'--fiber-manifest',fiber,'--threads','128',
        '--overlap-rank',rank,'--overlap-normal',str(n),'--overlap-tangential',str(t),
        '--overlap-fraction',str(f)]}]
    config['input_files']=[normal,fiber,str(source_trace/'.zattrs')]
    if LOW:
        config['commands'][0]['args']+=['--overlap-cache',str(OUT/f'overlap-{n}x{t}.json')]
    return config


def bp_config(template,name,trace):
    config=shared.configured(template,name+'-ind','',{})
    args=config['commands'][0]['args'];old=Path(args[1]);args[1]=str(trace)
    config['input_files']=[str(trace/'.zattrs') if p==str(old/'.zattrs') else p for p in config['input_files']]
    return config


def publish(rows):
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_write(OUT/'results.json',rows,durable=False)
    atomic_write(REPORT.with_suffix('.json'),rows,durable=False)
    lines=['# Post-fw/bw piece overlap suppression —1024 pilot','',
        'Frozen1879 raw40500 beam8/full-prefix512/iso20 pieces. No early stopping or retracing. '
        'Every retained reverse-filter split is independent. Global quality-ranked whole-piece suppression; '
        'ties use length then source ordinal. Suppressed pieces cannot suppress others.',
        '', ('Collision:0%,10%,20% of the SHORTER piece;0% requires at least one positive sampled match, never disjoint pairs. '
             'Ellipses10/20,20/40 first; then isotropic10/10,20/20. ' if LOW else
             'Collision: at least50% or80% of the SHORTER piece has matching segments of ONE other piece. ')+
        'Arc-weighted midpoint samples at most4base voxels apart, shared normal ellipse, unoriented angle45°. '
        'Presence is length-weighted shared prediction sampling (max valid option), not seed presence. '
        'Cost is saved tracer loss/path length, lower wins. No GT is used in selection.',
        '', '| Variant | Pieces | Retained length | Postprocess s | BP pieces | Constraints | Non-oracle ↑ | Observed shift ↑ | Removal ↑ | Missing refs |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    base=rows.get('control',{}).get('summary',{}).get('retained_length_base',757036.997991)
    for name,r in rows.items():
        s=r.get('summary',{});i=r.get('independent',{});o=r.get('oracle',{})
        def pct(v):return 'pending' if v is None else f'{v:.4f}%'
        lines.append(f"| {name} | {r.get('traces','—')} | {pct(100*s.get('retained_length_base',0)/base)} | {s.get('postprocess_seconds',0):.2f} | {r.get('pieces','—')} | {r.get('constraints','—')} | {pct(i.get('independent_percent'))} | {pct(i.get('observed_ladder_shift',{}).get('ladder_shift_percent'))} | {pct(o.get('constraint_score_percent'))} | {i.get('independent_missing','pending')} |")
    if len(rows)==len(VARIANTS)+1 and all('oracle' in r and 'independent' in r for r in rows.values()):
        def score(r):
            return (r['independent']['independent_percent'],r['independent']['observed_ladder_shift']['ladder_shift_percent'],r['oracle']['constraint_score_percent'])
        best=max((n for n in rows if n!='control'),key=lambda n:score(rows[n]))
        verdict='No improvement: keep disabled.' if score(rows[best])<=score(rows['control']) else 'Promising pilot; defaults remain unchanged pending review.'
        lines+=['','## Verdict','',verdict+' Best filtered accuracy-first setting: `'+best+'`. '
            'This tests postprocessing the completed candidate pool, not online replacement/reseeding. '
            'Deleting overlapping evidence changes BP connectivity and weights and may lose unique tails; '
            'these mechanisms were not individually isolated. No2048 test or iteration increase.',
            '', 'Oracle reference accuracy (separate from non-oracle readout): '+', '.join(
                f"{n}: {r['oracle']['reference_score_percent']:.4f}%" for n,r in rows.items())+'.',
            '', 'Validation: native synthetic overlap tests, full native regression suites and Python config tests. '
            'Per-run binary/library hashes and source patches '
            'record the exact benchmark version. No new model inference was run.']
    lines+=['', 'Same selected median±64/min256 BP parameters,500messages,128threads,RelWithDebInfo; '
        'one trial per setting, cached inference only. Baseline BP rerun on the identical frozen input. '
        'Retained length is total geometry, not reference coverage. Removing whole pieces may lose unique tails. '
        'Observed shift excludes missing references; non-oracle denominator26. Removal is the separate reference-conditioned score. '
        'Artifact replay not measured: do not substitute direct reference-seeded replay, which does not test this filter.',
        '', 'Opt-in native mode: `vc_fiber_trace_chunk deduplicate SOURCE --output TARGET --normal-manifest NORMAL '
        '--overlap-rank RANK --overlap-normal N --overlap-tangential T --overlap-fraction F --threads 128` '
        '(RANK is length, cost or presence; presence additionally requires `--fiber-manifest`). '
        'Each output records original indices, suppressor IDs, measured overlap and full source processing settings.',
        '', 'Approved runner: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, control action '
        '`iso20_post_overlap`'+(' with `low:true`' if LOW else '')+' (`report_only:true` to regenerate report). Defaults unchanged.',
        '',f'All commands/provenance/artifacts: `{OUT}`.']
    if LOW:
        history=json.loads((HISTORY/'results.json').read_text())
        atomic_write(OUT/'comparison.json',{'current':rows,'historical':history},durable=False)
        lines+=['','## Best ranking per corridor and threshold','',
                'Selected by non-oracle accuracy, then observed shift, then removal score. '
                'All individual trials remain in the table above; this is not a new tuned run.',
                '', '| Corridor | Overlap | Keep by | Pieces | Non-oracle ↑ | Shift ↑ | Removal ↑ |',
                '|---|---:|---|---:|---:|---:|---:|']
        for n,t in ((10,20),(20,40),(10,10),(20,20)):
            for fraction in (0,10,20):
                names=[f'{rank}-{n}x{t}-f{fraction}' for rank in ('length','cost','presence')]
                if not all(name in rows and 'oracle' in rows[name] for name in names):continue
                best=max(names,key=lambda name:(rows[name]['independent']['independent_percent'],
                    rows[name]['independent']['observed_ladder_shift']['ladder_shift_percent'],rows[name]['oracle']['constraint_score_percent']))
                r=rows[best]
                lines.append(f"| {n}/{t} | {fraction}% | {best.split('-')[0]} | {r['traces']} | {r['independent']['independent_percent']:.4f}% | {r['independent']['observed_ladder_shift']['ladder_shift_percent']:.4f}% | {r['oracle']['constraint_score_percent']:.4f}% |")
        lines+=['','## Historical50/80% controls (saved, not rerun)','',
                '| Variant | Pieces | Non-oracle ↑ | Observed shift ↑ | Removal ↑ |',
                '|---|---:|---:|---:|---:|']
        for name,r in history.items():
            lines.append(f"| {name} | {r['traces']} | {r['independent']['independent_percent']:.4f}% | {r['independent']['observed_ladder_shift']['ladder_shift_percent']:.4f}% | {r['oracle']['constraint_score_percent']:.4f}% |")
        lines+=['','Overlap graph cache stores all positive pair fractions and exact ordered input geometry. '
                'Key includes normal identity/contents, scale1, radii, angle, sample spacing and contract version. '
                'Normal chunks are assumed immutable as in the existing benchmark caches. Ranking and threshold '
                'do not alter measured geometry. Each new variant still performs its own selection and BP solves. '
                'An80% length control checks exact survivor IDs against the prior experiment before the sweep. '
                'Cached-filter timings exclude the first measurement; inspect overlap_cache_hit in JSON.']
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main(report_only=False,low=False):
    if low:setup_low()
    OUT.mkdir(parents=True,exist_ok=True);shared.OUT=OUT
    rows=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else {}
    if report_only:publish(rows);return
    tt=json.loads((SOURCE/'beam-isotropic20.config.json').read_text())
    bt=json.loads((WIN/'c1681f71b4163-ind.config.json').read_text())
    if LOW:
        shared.execute(filter_config(tt,'audit80','length',10,20,.8))
        def selection(path):
            return json.loads((path/'.zattrs').read_text())['processing']['preprocessing']['post_overlap']['retained_source_indices']
        assert selection(OUT/'audit80.zarr')==selection(HISTORY/'length-10x20-f80.zarr'),'measurement refactor changed80%selection'
        shared.execute(filter_config(tt,'audit80cached','length',10,20,.8))
        assert selection(OUT/'audit80cached.zarr')==selection(OUT/'audit80.zarr'),'cached measurement changed80%selection'
        print('AUDIT80 exact survivor IDs match historical selection',flush=True)
    for name,rank,n,t,f in [('control',None,0,0,0),*VARIANTS]:
        trace=SOURCE/'beam-isotropic20.zarr' if rank is None else OUT/f'{name}.zarr'
        if rank is not None:shared.execute(filter_config(tt,name,rank,n,t,f))
        art=json.loads((trace/'.zattrs').read_text())['processing']['artifact']
        r=rows.setdefault(name,{})
        r.update(traces=art['trace_count'],summary={k:v for k,v in art['trace_summary'].items() if k!='reverse_piece_provenance'})
        publish(rows)
        bc=bp_config(bt,name,trace)
        for stage in ('ind','oracle'):
            config=bc if stage=='ind' else shared.configured(bc,name+'-oracle','',{})
            if stage=='oracle':config['commands'][0]['args']=oracle_args(config['commands'][0]['args'])
            rec=shared.execute(config)
            log=(OUT/f'{name}-{stage}.log').read_text()
            if stage=='ind':
                match=re.search(r'BP-only cohort .*pieces=(\d+) selected_constraints=(\d+)',log)
                r.update(pieces=int(match[1]),constraints=int(match[2]),independent=readout_metrics(log))
            else:r['oracle']=dict(bp_metrics(log),counts=pruning_counts(log))
            r[stage+'_seconds']=rec['wall_seconds'];publish(rows)
        print('COMPLETE '+name+' '+json.dumps(dict(traces=r['traces'],accuracy=r['independent']['independent_percent'],
              shift=r['independent']['observed_ladder_shift']['ladder_shift_percent'],removal=r['oracle']['constraint_score_percent'])),flush=True)


if __name__=='__main__':
    import sys
    main('--report-only' in sys.argv,'--low' in sys.argv)
