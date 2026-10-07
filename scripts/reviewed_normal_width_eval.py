"""Frozen normal/width/HV checkpoints through the existing reviewed FW/BW runner."""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

from reviewed_fiber_evaluation import ROOT, REPO, freeze_selection
from reviewed_reverse_prefix import workload, evaluate_config
from lasagna.inference_provenance import atomic_write

RUNS = Path('/ephemeral/hendrik/fiber3d_train2/runs')
PREFIX = 'reviewed_normal_width_hv_fullres_128_sd1_bs128_ema_20261006_'
SELECTIONS = (
    ('normalwidth_083922_besttrace_fw512_20261006_v1', '083922', 'best_reviewed_trace.pt', 'reviewed_trace/restarts_combined'),
    ('normalwidth_105948_bestloss_fw512_20261006_v1', '105948', 'best.pt', 'test/loss_total'),
    ('normalwidth_105948_current_fw512_20261006_v1', '105948', 'current.pt', None),
)
BASELINE = ROOT/'reverse_prefix_40500_v1/prefix512_terminal'
REPORT = ROOT/'normalwidth_fw512_20261006_comparison.json'
WIDTH_NAME = 'normalwidth_u11000_width20_fw512_20261006_v1'


def measured_result(folder):
    from reviewed_fiber_evaluation import validate_result
    from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import consistency_report
    summary=json.loads((folder/'full/summary.json').read_text())
    validate_result(summary)
    if summary['cases']!=182 or summary['eligible_runs']!=91:
        raise ValueError('reviewed cohort changed')
    result=consistency_report(summary)
    scales={s['source_id']:s['summary']['base_voxel_size_um']/1000 for s in summary['sources']}
    spans={s:[] for s in scales}
    def read(path):
        document=json.loads(path.read_text())
        source=document['provenance']['source_id']
        lengths=[max(0.,s['end_reference_arc_base']-s['matches'][0]['search_begin_arc_base'])*scales[source]
                 if s['matches'] else 0. for s in document['replay']['segments']]
        conversions=sum('filtered_distance_failure_xyz_arc' in s for s in document['replay']['segments'])
        return source,lengths,conversions
    paths=list((folder/'full/cases').glob('*/trace.json'))
    if len(paths)!=182: raise ValueError('missing case artifacts')
    conversions=0
    with ThreadPoolExecutor(16) as pool:
        for source,lengths,count in pool.map(read,paths):
            spans[source].extend(lengths)
            conversions+=count
    def update(row,values,stops):
        values=sorted(values)
        def quantile(q):
            i=(len(values)-1)*q
            lo=int(i)
            return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(i-lo)
        if not math.isclose(sum(values),row['retained_mm'],rel_tol=1e-8,abs_tol=1e-7):
            raise ValueError('retained span lengths do not match native summary')
        row.update(actual_stops=stops,mean_mm=row['retained_mm']/(stops+1),
            median_mm=quantile(.5),p05_mm=quantile(.05),p95_mm=quantile(.95))
    all_stops=0
    for source in summary['sources']:
        stops=sum(len(c['failures'])+len(c.get('filtered_stops',[])) for c in source['cases'])
        all_stops+=stops
        update(result['scrolls'][source['source_id']],spans[source['source_id']],stops)
    update(result['combined'],[x for v in spans.values() for x in v],all_stops)
    result.update(summary_path=str(folder/'full/summary.json'),
        mean_definition='retained-GT-distance-per-all-actual-stops-plus-one',
        terminal_distance_failures_filtered=conversions)
    return result


def report():
    results=json.loads(REPORT.read_text()) if REPORT.exists() else {}
    for name,folder in [('Beam40500 FW/BW512',BASELINE)]+[(n,ROOT/n) for n,_,_,_ in SELECTIONS]:
        path=folder/'full/summary.json'
        if name not in results and path.exists() and json.loads(path.read_text()).get('completed'):
            results[name]=measured_result(folder)
            meta=folder/'snapshot_metadata.json'
            if meta.exists(): results[name]['snapshot']=json.loads(meta.read_text())
    atomic_write(REPORT,results,durable=False)
    return results


def document_results(results):
    """Audit matched inputs and persist the complete comparison in the repo."""
    baseline=json.loads((BASELINE/'full.json').read_text())
    def cohort(config):
        return [(s['id'],s['coordinate_space'],s['base_voxel_size_um'],s['fibers'])
                for s in config['sources']]
    for name,_,_,_ in SELECTIONS:
        if name not in results:
            continue
        folder=ROOT/name
        config=json.loads((folder/'full.json').read_text())
        if cohort(config)!=cohort(baseline):
            raise ValueError('reference cohort or coordinates differ from baseline')
        args=config['trace_args']
        for flag,value in [('--replay-reverse-interval-base','512'),
                           ('--replay-reverse-normal-base','20'),
                           ('--replay-reverse-tangential-base','40')]:
            if args[args.index(flag)+1]!=value:
                raise ValueError(f'incorrect effective {flag}')
        if not {'--replay-reverse-to-start','--replay-reverse-all-endpoints'}<=set(args):
            raise ValueError('missing full-prefix/terminal consistency checks')
        results[name]['wall_seconds']=json.loads((folder/'timing.json').read_text())['wall_seconds']
    target=REPO/'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks/reviewed_normal_width_fw512_20261006'
    atomic_write(target.with_suffix('.json'),results,durable=False)
    labels=['Previous raw u40500','083922 best-restarts raw u9800',
            '105948 best-loss raw u10800','105948 current raw u11000']
    keys=['Beam40500 FW/BW512']+[x[0] for x in SELECTIONS]
    lines=['# Normal/width/HV models with full-prefix FW/BW consistency', '',
        '91 reviewed fibers, 182 directions, five scrolls. Identical frozen cohort and geometry to the previous raw u40500 baseline. '
        'Beam8/lookahead2; forward ellipse40/80 base voxels, excursion grace160. '
        'Reverse full prefix every512 base voxels and at endpoints; reverse ellipse20/40, no reversed history, no embedding. '
        'New auxiliary model heads are loaded but do not change the existing presence/direction tracing inputs.', '',
        'Error restarts are geometric episodes grouped within80 base voxels. Filtered stops are ungrouped. '
        'Actual stops counts individual geometric failures plus filtered stops, before episode grouping. '
        'Mean = retained GT mm / (actual stops +1), treating the cohort as one connected line as requested. '
        'Median and P05/P95 describe actual retained spans, including terminal and zero-length spans. '
        'GT still controls ordinary failure endpoints/reseeding; terminal failure→filter conversions are oracle-timed diagnostics, not GT-free detection.', '',
        '| Model | Error restarts | Filtered stops | Actual stops | Retained % | Mean mm | Median mm | P05 mm | P95 mm |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def row(label,c):
        return f'| {label} | {c["ordinary_restarts"]} | {c["filtered_stops"]} | {c["actual_stops"]} | {c["retained_percent"]:.4f} | {c["mean_mm"]:.4f} | {c["median_mm"]:.4f} | {c["p05_mm"]:.4f} | {c["p95_mm"]:.4f} |'
    lines.extend(row(label,results[key]['combined']) for label,key in zip(labels,keys) if key in results)
    lines+=['','## Per scroll','',
        '| Scroll / model | Error restarts | Filtered stops | Actual stops | Retained % | Mean mm | Median mm | P05 mm | P95 mm |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scroll in results[keys[0]]['scrolls']:
        lines.extend(row(scroll+' / '+label,results[key]['scrolls'][scroll])
                     for label,key in zip(labels,keys) if key in results)
    lines+=['','## Snapshots and reproduction','',
        'Code: fiber-exp5 after fast-forward merge of fiber-normal-width-hv at5d76a43ad. '
        'Neural follower integration was committed separately atba86eea12 on fiber-exp6. '
        'All requested raw checkpoints were frozen before evaluation; later training changes do not alter these selections.', '',
        'Use control action `reviewed_normal_width` in `scripts/training_failure_mining_control.json`, then:', '',
        '```bash','bash /home/hendrik/villa3/scripts/run_training_failure_mining','```','',
        'GPUs0–5, existing manager-exclusive runner. Existing BF16 inference/float16 accumulation retained for matched baseline protocol. '
        'Wall time includes preparation, manager/resource waits and new inference; it is not isolated trace speed. '
        'Full per-case artifacts and immutable workloads: `'+str(ROOT)+'`. Machine-readable companion JSON contains source paths, hashes and full metrics.', '',
        '| Selection | SHA256 | Wall s | Terminal failure→filtered |','|---|---|---:|---:|']
    for label,key in zip(labels,keys):
        if key not in results: continue
        r=results[key]
        lines.append(f'| {label} | {r.get("snapshot",{}).get("sha256","historical baseline")} | {r.get("wall_seconds","—")} | {r["terminal_distance_failures_filtered"]} |')
    lines+=['','Validation:498 native cases,4 queue regressions,5 configuration/report/reverse-prefix tests passed. '
        'Completed results additionally validate182cases/91eligible runs, exact source fiber/coordinate identities, effective reverse flags, and retained-span sums. '
        'The u9800 training-time selection was22restarts on32pieces/64directions; that is not the full benchmark score.']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def validate_snapshot(saved, metric):
    if saved.get('snapshot_kind') not in (None, 'raw') or saved.get('ema_metadata'):
        raise ValueError('requested raw snapshot, not EMA')
    if metric is not None and saved.get('metric_name') != metric:
        raise ValueError(f'expected {metric}, got {saved.get("metric_name")}')
    if metric == 'reviewed_trace/restarts_combined' and not saved.get('reviewed_trace_selection'):
        raise ValueError('best-by-restarts snapshot lacks selection provenance')


def configuration(base, name, snapshot, digest):
    config = workload(base, name, 512, True, experiment=ROOT)
    config.update(checkpoint=str(snapshot), checkpoint_sha256=digest,
        native_binary=str(REPO/'volume-cartographer/build/fiber-evaluation/bin/vc_fiber_trace_metric'),
        devices='0,1,2,3,4,5', pilot_devices='0,1,2,3,4,5')
    config['evaluation']['inference_cache_namespace'] = name
    # Make the already-selected reverse acceptance policy explicit.
    config['evaluation']['trace_args'] += ['--replay-reverse-normal-base', '20',
                                         '--replay-reverse-tangential-base', '40']
    return config


def main(control):
    if control.get('width_no_reverse'):
        return run_width_no_reverse(control)
    if control.get('width_sweep'):
        return run_width_sweep(control)
    if control.get('width_gate'):
        return run_width_experiment(control)
    if control.get('report_only'):
        document_results(report())
        return
    if control.get('tests'):
        import subprocess
        subprocess.run([sys.executable,'-m','pytest','--noconftest','-q','scripts/test_reviewed_normal_width_eval.py',
            'scripts/test_reviewed_reverse_prefix.py'],cwd=REPO,check=True)
        return
    import torch
    base = json.loads((ROOT/'halfsigma_sideband_best_raw_20261002/workload.json').read_text())
    configs=[]
    # Capture all mutable selections before any long inference/evaluation.
    for name, run, filename, metric in SELECTIONS:
        source=RUNS/(PREFIX+run)/'snapshots'/filename
        snapshot,digest=freeze_selection(name,source)
        saved=torch.load(snapshot,map_location='cpu',weights_only=False,mmap=True)
        validate_snapshot(saved,metric)
        metadata=dict(source=str(source),sha256=digest,
            **{k:saved.get(k) for k in ('step','samples_seen','metric','metric_name','snapshot_kind',
                'reviewed_trace_selection','reviewed_trace_metrics')})
        atomic_write(ROOT/name/'snapshot_metadata.json',metadata,durable=False)
        print(f'SNAPSHOT {name}: {json.dumps(metadata)}',flush=True)
        del saved
        configs.append((name,configuration(base,name,snapshot,digest)))
    if control.get('freeze_only'):
        return
    for name,config in configs:
        evaluate_config(config,name)
        print(f'FWBW RESULT {name}: {json.dumps(report()[name])}',flush=True)
    document_results(report())


def run_width_experiment(control):
    from lasagna.inference_provenance import sha256_file
    baseline=ROOT/SELECTIONS[-1][0]
    base=json.loads((baseline/'workload.json').read_text())
    if sha256_file(base['checkpoint'])!=base['checkpoint_sha256']:
        raise ValueError('frozen u11000 checkpoint changed')
    config=json.loads(json.dumps(base))
    config['output_root']=str(ROOT/WIDTH_NAME)
    config['evaluation']['gpu_manager']['label']='reviewed-u11000-width20-fw512'
    config['evaluation']['inference']['emit_width']=True
    config['evaluation']['inference_cache_namespace']=WIDTH_NAME
    config['evaluation']['trace_args']+=['--width-relative-tolerance','0.2']
    atomic_write(ROOT/WIDTH_NAME/'snapshot_metadata.json',
                 json.loads((baseline/'snapshot_metadata.json').read_text()),durable=False)
    if not control.get('report_only'):
        evaluate_config(config,WIDTH_NAME)
    folders=[('Previous u40500',BASELINE),('u11000 unweighted',baseline),('u11000 width20',ROOT/WIDTH_NAME)]
    results={name:measured_result(folder) for name,folder in folders}
    effective=json.loads((ROOT/WIDTH_NAME/'full.json').read_text())
    prior=json.loads((baseline/'full.json').read_text())
    assert effective['sources']==prior['sources']
    assert effective['trace_args']==prior['trace_args']+['--width-relative-tolerance','0.2']
    results['u11000 width20']['snapshot']=json.loads((baseline/'snapshot_metadata.json').read_text())
    results['u11000 width20']['wall_seconds']=json.loads((ROOT/WIDTH_NAME/'timing.json').read_text())['wall_seconds']
    target=REPO/'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks/reviewed_u11000_width20_fw512_20261006'
    atomic_write(ROOT/WIDTH_NAME/'comparison.json',results,durable=False)
    atomic_write(target.with_suffix('.json'),results,durable=False)
    lines=['# Cumulative-width presence weighting — frozen raw u11000', '',
        'Same reviewed91fibers/182directions and prefix512 terminal FW/BW20/40 protocol. '
        'Presence multiplier=max(0,1-abs(sample_width/mean_width-1)/0.20). '
        'Mean initialized from seed and updated by each newly irrevocable forward point, before its observer. '
        'Speculative beam samples never update it; shared-prefix commits can lag the frontier. '
        'Each reverse check freezes the accumulated forward mean, including its endpoint, with no reseeding/updating. '
        'Restart initializes a new mean. Invalid candidate widths suppress presence; invalid seed/committed widths raise an error.', '',
        'Width is the shared annotation softplus output in network-input voxel units, explicitly published as float32 '
        '`fiber_width.ome.zarr` / manifest channel `width`. Same units used for both sides of each ratio. '
        'Existing BF16 inference and float16 accumulation retained; no scheduling or worker settings changed. '
        'Width-containing cache identity differs from legacy caches; those remain untouched. Auxiliary normal/HV heads not used.', '',
        'Error restarts are80-base-voxel grouped episodes, filtered stops ungrouped. '
        'Actual stops counts all individual error+filtered events. Mean=retained GT mm/(actual stops+1). '
        'Median/P05/P95 use actual retained spans including terminal/zero-length spans. '
        'GT determines ordinary failure endpoints/reseeding, so terminal failure→filter conversions remain oracle-timed.', '',
        '| Model | Error restarts | Filtered stops | Actual stops | Retained % | Mean mm | Median mm | P05 mm | P95 mm |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def row(label,c):
        return f'| {label} | {c["ordinary_restarts"]} | {c["filtered_stops"]} | {c["actual_stops"]} | {c["retained_percent"]:.4f} | {c["mean_mm"]:.4f} | {c["median_mm"]:.4f} | {c["p05_mm"]:.4f} | {c["p95_mm"]:.4f} |'
    lines.extend(row(n,r['combined']) for n,r in results.items())
    lines+=['','## Per scroll','',
        '| Scroll / model | Error restarts | Filtered stops | Actual stops | Retained % | Mean mm | Median mm | P05 mm | P95 mm |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scroll in results['Previous u40500']['scrolls']:
        lines.extend(row(scroll+' / '+n,r['scrolls'][scroll]) for n,r in results.items())
    lines+=['','## Reproduce','',
        'Control: `{"action":"reviewed_normal_width","width_gate":true}` in scripts/training_failure_mining_control.json.', '',
        '```bash','bash /home/hendrik/villa3/scripts/run_training_failure_mining','```','',
        f'Artifacts: `{ROOT/WIDTH_NAME}`. GPUs0–5. Companion JSON records exact frozen model/hash and per-case artifact paths. '
        'The runner reuses completed results; add `report_only:true` to regenerate tables without evaluation.', '',
        'Validation: width export tests preserve original outputs; native arithmetic mean, committed-only state, frozen reverse, '
        'generic/fused scoring and scalar sampling regressions; full native build/test suite. '
        'All requested directions complete, effective sources/trace flags checked against unweighted u11000 and retained sums audited.']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({n:r['combined'] for n,r in results.items()},indent=2),flush=True)


def without_reverse_width_config(base, name, *, cosine, tolerance=.1):
    """Keep the published inference identity; change only trace scoring/filtering."""
    from vesuvius.neural_tracing.fiber_trace_3d.beam_parameter_sweep import apply_trace_args
    config=json.loads(json.dumps(base))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('width tolerance must be positive and finite')
    values={'--replay-reverse-interval-base','--replay-reverse-normal-base',
            '--replay-reverse-tangential-base','--replay-reverse-history-steps'}
    switches={'--replay-reverse-all-endpoints','--replay-reverse-to-start'}
    original=config['evaluation']['trace_args']
    args=[]
    index=0
    while index<len(original):
        flag=original[index]
        if flag in values:
            if index+1>=len(original): raise ValueError('missing reverse option value')
            index+=2
        elif flag in switches:
            index+=1
        elif flag.startswith('--replay-reverse'):
            raise ValueError(f'unhandled reverse option: {flag}')
        else:
            args.append(flag)
            index+=1
    config['evaluation']['trace_args']=apply_trace_args(args,{
        'width-relative-tolerance':tolerance if cosine else 0,'width-weighting':'cosine' if cosine else 'linear'})
    config['output_root']=str(ROOT/name)
    config['evaluation']['gpu_manager']['label']=name
    return config


def width_metric_table(results):
    lines=['| Variant | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        lines.append(f'| {name} | {r["ordinary_restarts"]} | {r["filtered_stops"]} | {r["actual_stops"]} | {r["retained_percent"]:.4f} | {r["mean_mm"]:.4f} | {r["median_mm"]:.4f} | {r["p05_mm"]:.4f} | {r["p95_mm"]:.4f} |')
    return lines


def run_width_no_reverse(control):
    from lasagna.inference_provenance import sha256_file
    base=json.loads((ROOT/WIDTH_NAME/'workload.json').read_text())
    expected='2408ece6ffa2251655709cf3987357f1892a573828929934266145f09f9d2eba'
    if base['checkpoint_sha256']!=expected or sha256_file(base['checkpoint'])!=expected:
        raise ValueError('expected frozen raw u11000')
    results={}
    for label,cosine,tolerance in [('off',False,.1),('cosine10',True,.1),('cosine40',True,.4)]:
        name=f'normalwidth_u11000_{label}_no_reverse_20261006_v1'
        config=without_reverse_width_config(base,name,cosine=cosine,tolerance=tolerance)
        if not control.get('report_only'):
            evaluate_config(config,name)
        folder=ROOT/name
        result=measured_result(folder)
        if result['combined']['reverse_checks'] or result['combined']['filtered_stops']:
            raise ValueError('reverse consistency was not disabled')
        effective=json.loads((folder/'full.json').read_text())
        prior=json.loads((ROOT/WIDTH_NAME/'full.json').read_text())
        if effective['sources']!=prior['sources'] or effective['trace_args']!=config['evaluation']['trace_args']:
            raise ValueError('no-reverse cohort or trace settings mismatch')
        summary=json.loads((folder/'full/summary.json').read_text())
        trace=summary['effective_trace']
        if trace['replay_reverse_interval_base']!=0 or trace['replay_reverse_all_endpoints']:
            raise ValueError('native reverse options not disabled')
        if trace['width_relative_tolerance']!=(tolerance if cosine else 0) or trace['width_weighting']!=('cosine' if cosine else 'linear'):
            raise ValueError('native width options mismatch')
        result.update(checkpoint_sha256=expected,trace_args=effective['trace_args'],
                      wall_seconds=json.loads((folder/'timing.json').read_text())['wall_seconds'])
        inv=json.loads((folder/'full/inference_summary.json').read_text())
        result['inferred_tiles']=sum(s['inferred_tiles'] for s in inv['sources'].values())
        results[label]=result
        print(f'NO REVERSE {label}: '+json.dumps(result['combined']),flush=True)
        target=REPO/'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks/reviewed_u11000_width_no_reverse_20261006'
        atomic_write(target.with_suffix('.json'),results,durable=False)
        lines=['# Width presence modulation without FW/BW — raw u11000','',
            'Full91-fiber/182-direction cohort. Same published width/presence/direction predictions, '
            'checkpoint '+expected+'. FW/BW disabled: no periodic or endpoint reverse checks. '
            'All other tracing, inference precision and worker settings unchanged. Missing cache data uses existing GPU0–5 demand inference.', '',
            'Cosine multiplies sampled presence by cos(pi*d/2) for d<1, else0, '
            'where d=abs(width/committed_mean-1)/tolerance (0.10 or0.40). This is additional to direction alignment scoring and smoothness, '
            'not a separate stop criterion. Mean is cumulative over seed+committed points and resets at ordinary restart.', '',
            'Errors are80-base-voxel grouped restart episodes. Actual stops includes all individual failures; '
            'mean=retained GT mm/(actual stops+1), virtually concatenating all references. '
            'Median/quantiles include terminal and zero-length spans. GT drives error detection/restarts.', '']
        lines+=width_metric_table({n:r['combined'] for n,r in results.items()})
        for scroll in result['scrolls']:
            lines+=['',f'## {scroll}','']+width_metric_table({n:r['scrolls'][scroll] for n,r in results.items()})
        previous=REPO/'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks/reviewed_u11000_width_sweep_20261006.json'
        old=json.loads(previous.read_text())
        lines+=['','## Prior FW/BW512 context','']+width_metric_table({n:old[n]['combined'] for n in results})
        lines+=['','## Runtime and reproduction','','| Variant | Wall seconds | Additional tiles |','|---|---:|---:|']
        lines += [f'| {n} | {r["wall_seconds"]:.2f} | {r["inferred_tiles"]} |' for n,r in results.items()]
        lines+=['','One run each; times include startup/cache checks and any inference. '
                'Control: `{"action":"reviewed_normal_width","width_no_reverse":true}`.', '',
                '```bash','bash /home/hendrik/villa3/scripts/run_training_failure_mining','```','',
                'Companion JSON records exact summary paths, checkpoint and trace flags. '
                'Add report_only:true to regenerate without rerunning. Completed runs are skipped on normal resume.']
        target.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def width_sweep_variants():
    variants=[{'name':'linear20','parameters':{}},
              {'name':'off','parameters':{'width-relative-tolerance':0}}]
    for shape in ('linear','cosine'):
        for percent in (10,20,40,80):
            if shape=='linear' and percent==20:
                continue
            variants.append({'name':f'{shape}{percent}', 'parameters':{
                'width-relative-tolerance':percent/100, 'width-weighting':shape}})
    return variants


def run_width_sweep(control):
    from vesuvius.neural_tracing.fiber_trace_3d.beam_parameter_sweep import run_sweep
    from lasagna.inference_provenance import sha256_file
    base=json.loads((ROOT/WIDTH_NAME/'workload.json').read_text())
    expected='2408ece6ffa2251655709cf3987357f1892a573828929934266145f09f9d2eba'
    if base['checkpoint_sha256']!=expected or sha256_file(base['checkpoint'])!=expected:
        raise ValueError('width sweep must use frozen raw u11000')
    config=json.loads(json.dumps(base))
    demand=bool(control.get('fill_missing'))
    folder=ROOT/('normalwidth_u11000_gate_sweep_demand_20261006_v1' if demand
                 else 'normalwidth_u11000_gate_sweep_20261006_v1')
    config.update(output_root=str(folder),cached_evaluation=str(ROOT/WIDTH_NAME/'full'))
    config['beam_sweep']={'variants':width_sweep_variants(),'inference':'demand' if demand else 'cached'}
    # Existing cached beam sweep audits source hashes/manifests/cache identity,
    # retains native parallelism and fails loudly if a new path needs missing data.
    if not control.get('report_only'):
        run_sweep(config)
    results={}
    for variant in width_sweep_variants():
        saved=json.loads((folder/variant['name']/'result.json').read_text())
        if saved['row']['status']!='completed':
            results[variant['name']]=saved['row']
            continue
        result=measured_result(Path(saved['attempt_root']))
        result.update(parameters=variant['parameters'],native_sha256=saved['native_sha256'],
                      attempt_root=saved['attempt_root'],wall_seconds=saved['row']['wall_seconds'])
        inventory=json.loads((Path(saved['attempt_root'])/'full/inference_summary.json').read_text())
        result['inferred_tiles']=sum(s['inferred_tiles'] for s in inventory['sources'].values())
        result['published_chunks']=sum(s['published_chunks'] for s in inventory['sources'].values())
        results[variant['name']]=result
    target=REPO/'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks/reviewed_u11000_width_sweep_20261006'
    atomic_write(target.with_suffix('.json'),results,durable=False)
    lines=['# Width limit and cosine sweep — raw u11000', '',
        'Same frozen checkpoint as width20 benchmark; same91fibers/182directions, full-prefix512 '
        'FW/BW20/40. Same published width/presence/direction cache for ALL rows, including off. '+
        ('Existing on-demand runner fills only missing chunks on GPUs0–5. ' if demand else 'No GPU inference. ')+
        'Cumulative committed arithmetic mean; reverse mean frozen; restart resets. '
        'd=abs(width/mean-1)/limit. Linear=max(0,1-d); cosine=cos(pi*d/2) for d<1, else0.', '',
        'Mean=retained GT mm/(all actual stops+1). Median/P05/P95 are retained spans including terminal/zero-length spans. '
        'Errors are80-base-voxel grouped episodes; filtered stops ungrouped. '
        'GT failure endpoints/reseeding remain part of this benchmark, not fully GT-free filtering.', '']
    header=['| Gate | Errors | Filtered | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def row(name,r):
        return f'| {name} | {r["ordinary_restarts"]} | {r["filtered_stops"]} | {r["actual_stops"]} | {r["retained_percent"]:.4f} | {r["mean_mm"]:.4f} | {r["median_mm"]:.4f} | {r["p05_mm"]:.4f} | {r["p95_mm"]:.4f} |'
    lines+=header
    for name,r in results.items():
        if 'combined' in r:
            lines.append(row(name,r['combined']))
        else:
            lines+=['',f'FAILED {name}: {r.get("error")}', '']
    for scroll in next(r for r in results.values() if 'combined' in r)['scrolls']:
        lines+=['',f'## {scroll}','']+header
        lines += [row(name,r['scrolls'][scroll]) for name,r in results.items() if 'combined' in r]
    lines+=['','## Runtime and cache extension','',
            '| Gate | Wall seconds | Additional tiles | Published chunks |',
            '|---|---:|---:|---:|']
    lines += [f'| {n} | {r["wall_seconds"]:.2f} | {r["inferred_tiles"]} | {r["published_chunks"]} |'
              for n,r in results.items() if 'combined' in r]
    lines+=['','One run per configuration; times include cache validation, setup and any demand GPU inference. '
            'All published predictions are reused; new paths can require additional chunks. '
            'Earlier strict cache-only attempts are preserved separately, not counted as trace failures.',
        '', '## Reproduce','',
        'Control: `'+json.dumps({'action':'reviewed_normal_width','width_sweep':True,'fill_missing':demand})+'`.', '',
        '```bash','bash /home/hendrik/villa3/scripts/run_training_failure_mining','```','',
        f'Artifacts and audited timing/provenance: `{folder}`. Failed missing-cache trials are not scored. '
        'Add report_only:true to regenerate this report. Native linear20 repeat is the unchanged-code control.']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({n:r.get('combined',r) for n,r in results.items()},indent=2),flush=True)


if __name__=='__main__':
    main(json.loads((REPO/'scripts/training_failure_mining_control.json').read_text()))
