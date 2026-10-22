"""Resumable, final-metric coordinate search using the unchanged joint solver."""
import hashlib
import json
import math
from pathlib import Path


def stop(settings):
    """Stop one verified search controller through the approved runner (Linux)."""
    import os
    import signal
    import sys
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Verified process control requires Linux /proc; stop manually on this OS')
    pid=int(settings['pid']);path=Path(settings['config']).resolve()
    request=json.loads(path.read_text())
    config=request.get('fiber_uv_layout',{})
    if not config.get('loss_weight_search') and 'fiber_selected_weight_search' not in request:
        raise ValueError('Not a weight search configuration')
    args=Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    if args[1:]!=['-m','vesuvius.neural_tracing.fiber_trace_3d.fiber_workflows','--config',str(path)]:
        raise ValueError('PID does not match the exact requested benchmark search')
    if settings.get('terminate_tree',False):
        # Only descendants of the exact, verified controller are in scope.
        # Recheck start times before signalling to detect reused PIDs.
        parents={};starts={}
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():continue
            try:
                fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
                parents[int(entry.name)]=int(fields[1])
                starts[int(entry.name)]=fields[19]
            except (OSError,ValueError,IndexError):continue
        targets=[pid]
        for parent in targets:
            targets.extend(child for child,ppid in parents.items() if ppid==parent)
        for target in targets:
            try:
                current=Path(f'/proc/{target}/stat').read_text().rsplit(')',1)[1].split()
                if current[19]==starts[target]:os.kill(target,signal.SIGTERM)
            except (FileNotFoundError,ProcessLookupError):pass
        print(json.dumps(dict(terminated_tree=targets,completed_trials_preserved=True)),flush=True)
        return
    os.kill(pid,signal.SIGINT)
    print(json.dumps(dict(interrupted_pid=pid,config=str(path),completed_trials_preserved=True)),flush=True)


def configuration_key(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]


def coordinate_search(baseline, evaluate, grids, tolerance=1e-4, accepted=None, metric='ref_l1',secondary_metric=None, evaluate_many=None):
    """Restart every coordinate after improvement; finish at a grid-local minimum.

    evaluate must cache complete configurations. Resuming by replay is deliberate:
    cached decisions reproduce the same path, including interrupted acceptances.
    """
    current = dict(baseline)
    best = evaluate(current)
    for grid in grids:
        while True:
            improved = False
            for name, values in grid.items():
                if evaluate_many is not None:
                    evaluate_many([dict(current, **{name: value}) for value in values
                                   if value != current[name]])
                for value in values:
                    if value == current[name]:
                        continue
                    candidate = dict(current, **{name: value})
                    result = evaluate(candidate)
                    value=result.get(metric)
                    secondary_ok=(secondary_metric is None or
                        (result.get(secondary_metric) is not None and math.isfinite(result[secondary_metric])
                         and result[secondary_metric]<best[secondary_metric]))
                    if value is not None and math.isfinite(value) and value < best[metric] - tolerance and secondary_ok:
                        old = best
                        current, best = candidate, result
                        if accepted:
                            accepted(name, current, best, old)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    return current, best


def metrics(summary):
    final = summary['final']
    reference = final['reference']
    l1 = reference['mean_absolute_winding_error']
    if l1 is None or not math.isfinite(l1) or reference['matched_fibers'] == 0 or not math.isfinite(final['loss']):
        raise ValueError('No finite final reference L1; trial cannot be ranked')
    return dict(ref_l1=l1, loss=final['loss'],
        within_two_percent=reference['within_two_windings_percent'],
        retained_length_percent=final['retained_arc_percent'],
        matched_fibers=reference['matched_fibers'],
        width_factor=reference['widths']['mean_normalized_factor'],
        sign_percent=final['sign_violation_percent'],
        fit_range=reference['fitted_winding_q10_q90'],
        reference_range=reference['reference_winding_q10_q90'],
        solve_seconds=summary['solve_wall_seconds'])


def run_density(settings):
    """Managed independent density trials; same resumable coordinate algorithm."""
    import concurrent.futures
    import subprocess
    import sys
    import queue
    import threading
    import numpy as np
    from lasagna.inference_provenance import atomic_write, sha256_file
    from .fiber_gt_constraint_score import FrozenGTScorer
    from .fiber_implicit_match_report import run as match_report

    root = Path(settings['output_root']); root.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(settings['base_config']).read_text())['fiber_implicit_winding']
    if base.get('learned_winding_sigma') or base.get('v_offset_match_weight') or base.get('u_offset_match_weight'):
        raise ValueError('Density search requires fixed sigma and no endpoint offsets')
    initial = {name: float(base[name]) for name in settings['weights']}
    if not all(math.isfinite(v) and v > 0 for v in initial.values()):
        raise ValueError('Positive finite baseline weights required')
    grids = [{name: [value*f for f in factors] for name,value in initial.items()}
             for factors in settings.get('factor_grids', [[.25,1.,4.],[.25,.5,1.,2.,4.]])]
    scoring = dict(base, soft_winding_model=True, implicit_winding_map=True, unified_winding_loss=True)
    scorer = FrozenGTScorer(settings['gt_score'], scoring)
    paths = [Path(base[k]) for k in ('initial_solution','tensor_cache')]
    prepared_path=Path(base['initial_solution']).parent/'config.json'
    prepared=json.loads(prepared_path.read_text())
    paths += [prepared_path,Path(prepared['soft_samples'])]
    paths += [Path(__file__), Path(__file__).with_name('fiber_implicit_winding.py'),
              Path(__file__).with_name('fiber_soft_winding.py'),
              Path(__file__).with_name('fiber_gt_constraint_score.py'),
              Path(__file__).with_name('fiber_implicit_match_report.py')]
    paths += sorted(Path(__file__).parent.glob('fiber_uv*.py'))
    paths += [Path(base[k]) for k in ('umbilicus','features','stage1','hv_solution',
              'h_direction_solution','v_direction_solution','reference_subset') if k in base]
    manifest = dict(settings=settings,base=base,grids=grids,
                    hashes={str(p):sha256_file(p) for p in paths},gt_hashes=scorer.hashes)
    path = root/'search_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Search inputs changed; use a new output root')
    atomic_write(path,manifest,durable=False)
    devices = settings['devices']
    if not devices or len(set(devices)) != len(devices): raise ValueError('Unique GPU lanes required')
    lanes = queue.Queue()
    for device in devices: lanes.put(str(device))
    results = {}
    for path in sorted((root/'trials').glob('*/result.json')):
        row=json.loads(path.read_text()); trial=path.parent
        if row['key']!=configuration_key(row['weights']) or row['key']!=trial.name:
            raise ValueError('Cached trial key mismatch')
        summary=json.loads((trial/'summary.json').read_text())
        expected=dict(base,**row['weights'],output_root=str(trial),log_every=1000)
        for name,value in expected.items():
            if name not in ('devices','gpu_manager') and summary['config'].get(name)!=value:
                raise ValueError('Cached trial configuration mismatch: '+name)
        if summary['final']['iteration']!=base['steps']:
            raise ValueError('Cached trial incomplete')
        if sha256_file(trial/'solution.npz') != row['solution_sha256']:
            raise ValueError('Cached solution changed')
        results[row['key']]=row
    acceptances=[]
    report_lock=threading.Lock()  # Reporter configures global PyTorch CPU thread count.

    def evaluate(values):
        key=configuration_key(values)
        if key in results: return results[key]
        device=lanes.get()
        try:
            trial=root/'trials'/key; trial.mkdir(parents=True,exist_ok=True)
            child=dict(base,**values,devices=device,output_root=str(trial),log_every=1000)
            child['gpu_manager']=dict(base['gpu_manager'],exclusive=False,label='density-search-'+key)
            config_path=trial/'input.json'
            atomic_write(config_path,{'fiber_implicit_winding':child},durable=False)
            if not (trial/'summary.json').exists():
                with (trial/'run.log').open('w') as log:
                    subprocess.run([sys.executable,'-m','vesuvius.neural_tracing.fiber_trace_3d.fiber_workflows','--config',str(config_path)],
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
            summary=json.loads((trial/'summary.json').read_text())
            for name,value in child.items():
                if name not in ('devices','gpu_manager') and summary['config'].get(name)!=value:
                    raise ValueError('Saved trial configuration mismatch: '+name)
            if summary['final']['iteration'] != base['steps']:
                raise ValueError('Incomplete trial cannot be ranked')
            with np.load(trial/'solution.npz') as saved:
                if not np.isfinite(saved['state']).all() or not np.isfinite(saved['widths']).all() or np.any(saved['widths']<=0):
                    raise ValueError('Invalid solution')
            score=scorer.score(trial/'solution.npz',dict(scoring,**values))
            with report_lock:
                meeting=match_report(dict(output_root=str(trial/'matches'),cutoffs=[.5,.7],solve_roots=[str(trial)]))
            row=dict(key=key,weights=values,output_root=str(trial),**score,
                     loss=summary['final']['loss'],final=summary['final'],
                     solve_seconds=summary['solve_wall_seconds'],matches=meeting,
                     solution_sha256=sha256_file(trial/'solution.npz'))
            if not math.isfinite(row['loss']) or not math.isfinite(row['gt_enabled_l1']):
                raise ValueError('Nonfinite ranking metric')
            atomic_write(trial/'result.json',row,durable=False)
            results[key]=row
            return row
        finally: lanes.put(device)

    def publish():
        atomic_write(root/'results.json',list(results.values()),durable=False)

    def accept(name,values,best,old):
        row=dict(best,changed_weight=name,previous_gt_l1=old['gt_enabled_l1'])
        acceptances.append(row)
        atomic_write(root/'acceptances.json',acceptances,durable=False)
        atomic_write(root/'best.json',best,durable=False)
        publish()
        print('DENSITY SEARCH IMPROVED '+json.dumps(dict(weight=name,value=values[name],
            gt_l1=best['gt_enabled_l1'],correct=best['gt_enabled_correct_percent'],
            trials=len(results))),flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        def evaluate_many(candidates):
            pending=[v for v in candidates if configuration_key(v) not in results]
            atomic_write(root/'active.json',dict(candidates=pending,completed=len(results)),durable=False)
            list(pool.map(evaluate,pending))
            publish()
        first=evaluate(initial)
        atomic_write(root/'best.json',first,durable=False)
        values,best=coordinate_search(initial,evaluate,grids,metric='gt_enabled_l1',
            accepted=accept,evaluate_many=evaluate_many)
    certificate=[]
    for name,alternatives in grids[-1].items():
        for value in alternatives:
            neighbor=dict(values,**{name:value}); row=results[configuration_key(neighbor)]
            if row['gt_enabled_l1'] < best['gt_enabled_l1']-1e-4:
                raise AssertionError('Final neighborhood not converged')
            certificate.append(dict(weight=name,value=value,key=row['key'],gt_l1=row['gt_enabled_l1']))
    final=dict(best=best,weights=values,trials=len(results),acceptances=acceptances,
               termination='All final fine-grid coordinate alternatives checked; no improvement >1e-4',
               certificate=certificate)
    publish();atomic_write(root/'best.json',best,durable=False)
    atomic_write(root/'summary.json',final,durable=False)
    print('DENSITY SEARCH COMPLETE '+json.dumps(dict(trials=len(results),gt_l1=best['gt_enabled_l1'])),flush=True)
    return final


def selected_retry_device(trial, devices):
    """The evaluator's original stamp, not a rewritten child config, owns the lane."""
    stamp=Path(trial)/'eval/inputs.json'
    if not stamp.exists():return None
    device=str(json.loads(stamp.read_text())['settings']['devices'])
    if device not in devices:raise ValueError('Recorded trial GPU not in search lanes')
    return device


def validate_selected_retry(trial, child, settings):
    stamp=Path(trial)/'eval/inputs.json'
    if not stamp.exists():return
    recorded=json.loads(stamp.read_text())
    if recorded['settings']!=settings:raise ValueError('Changed retry evaluation settings')
    payload=json.dumps({'fiber_implicit_winding':child},indent=2,sort_keys=True)+'\n'
    expected=recorded['inputs'][str(Path(trial)/'base.json')]
    if hashlib.sha256(payload.encode()).hexdigest()!=expected:
        raise ValueError('Changed retry solver configuration')


def regression_guard(reports,limits):
    """No GT labels enter optimization; reject bad signs/geometry after solving."""
    signs=0;reversals=0;excess=0.
    for report in reports.values():
        row=report['final']
        signs+=row['perp_sign_bad']+row['parallel_sign_bad']
        for family in row['segments'].values():
            reversals+=family['reversals']
            for metric,limit in [('length_error_percent',limits['length_max_percent']),
                                 ('angle_degrees',limits['angle_max_degrees'])]:
                value=family[metric]['max']
                if value is None or not math.isfinite(value):raise ValueError('Missing GT geometry')
                excess+=max(0.,value-limit)
    return dict(passed=not(signs or reversals or excess),signs=signs,reversals=reversals,
                geometry_excess=excess)


def needs_gt_validation(perp_l1, results, tolerance):
    """No skipped candidate can outrank an already validated incumbent."""
    passing=[r['perp_l1'] for r in results if (r.get('gt_guard') or {}).get('passed',False)]
    return not passing or perp_l1 < min(passing)-tolerance


def run_selected(settings):
    """Reuse coordinate search with the selected-dataset fresh-init evaluator."""
    import concurrent.futures
    import queue
    import subprocess
    import sys
    import threading
    from lasagna.inference_provenance import atomic_write, sha256_file
    if settings.get('wait_for_controller_pid'):
        import time
        pid=int(settings['wait_for_controller_pid'])
        expected=str(Path(settings['wait_for_controller_config']).resolve())
        proc=Path(f'/proc/{pid}/cmdline')
        print(f'Waiting for controller {pid} to finish its current candidate; no duplicate solve.',flush=True)
        while proc.exists():
            try:args=proc.read_bytes().rstrip(b'\0').decode().split('\0')
            except FileNotFoundError:break
            if not args or args==['']:break
            if args[-2:]!=['--config',expected]:raise ValueError('Wait PID no longer matches old controller')
            time.sleep(2)
    root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path(settings['base_config']).read_text())['fiber_implicit_winding']
    initial={k:float(base[k]) for k in settings['weights']}
    grids=[{k:[v*f for f in factors] for k,v in initial.items()}
           for factors in settings['factor_grids']]
    manifest=dict(settings=settings,base=base,hashes={str(p):sha256_file(p)
        for p in sorted(Path(__file__).parent.glob('fiber_*.py'))})
    stamp=root/'manifest.json'
    if stamp.exists() and json.loads(stamp.read_text())!=manifest:
        raise ValueError('Search code/config changed; use a new output directory')
    atomic_write(stamp,manifest,durable=False)
    if not settings['devices'] or len(set(settings['devices']))!=len(settings['devices']):
        raise ValueError('Unique GPU lanes required')
    devices=[str(device) for device in settings['devices']]
    available=set(devices);lane_condition=threading.Condition()
    results={};lock=threading.Lock();acceptances=[]
    # Explicit scheduling-only continuation: retain immutable original trials
    # and provenance, rather than rewriting their manifests or re-solving them.
    for origin in settings.get('reuse_completed_from',[]):
        old=Path(origin);previous=json.loads((old/'manifest.json').read_text())
        if previous['base']!=base:raise ValueError('Reuse requires identical base scientific configuration')
        for key in ('dataset','gt_root','weights','gt_parameters','gt_limits','gt_regressions'):
            if previous['settings'].get(key)!=settings.get(key):raise ValueError('Reuse settings differ: '+key)
        for cached in sorted((old/'trials').glob('*/rank.json')):
            row=json.loads(cached.read_text())
            if row['sha256']!=sha256_file(Path(row['output_root'])/'solve/solution.npz'):
                raise ValueError('Reused solution changed')
            results[row['key']]=dict(row,reused_from=str(cached))
    if results:atomic_write(root/'results.json',list(results.values()),durable=False)
    def evaluate(values):
        key=configuration_key(values)
        if key in results:return results[key]
        trial=root/'trials'/key;trial.mkdir(parents=True,exist_ok=True)
        cached=trial/'rank.json'
        if cached.exists():
            row=json.loads(cached.read_text())
            if row['weights']!=values or row['sha256']!=sha256_file(trial/'eval/solve/solution.npz'):
                raise ValueError('Changed cached trial')
            with lock:results[key]=row
            return row
        preferred=selected_retry_device(trial,devices)
        with lane_condition:
            lane_condition.wait_for(lambda: preferred in available if preferred is not None else bool(available))
            device=preferred if preferred is not None else next(d for d in devices if d in available)
            available.remove(device)
        try:
            child=dict(base,**values,devices=device,output_root=str(trial/'eval'),
                       save_diagnostic_checkpoints=False)
            child['gpu_manager']=dict(base['gpu_manager'],exclusive=False,label='affine-weight-search')
            cfg=trial/'base.json';run=trial/'run.json'
            evaluation=dict(dataset=settings['dataset'],gt_root=settings['gt_root'],base_config=str(cfg),
                            output_root=str(trial/'eval'),devices=device)
            validate_selected_retry(trial,child,evaluation)
            atomic_write(cfg,{'fiber_implicit_winding':child},durable=False)
            atomic_write(run,{'fiber_selected_dataset_eval':evaluation},durable=False)
            with (trial/'run.log').open('a') as log:
                subprocess.run([sys.executable,'-m','vesuvius.neural_tracing.fiber_trace_3d.fiber_workflows','--config',str(run)],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            r=json.loads((trial/'eval/results.json').read_text())
            sign=trial/'sign.json'
            atomic_write(sign,{'fiber_density_sign_report':dict(gt_evaluations=[str(trial/'eval')],
                output_root=str(trial/'sign'))},durable=False)
            with (trial/'sign.log').open('w') as log:
                subprocess.run([sys.executable,'-m','vesuvius.neural_tracing.fiber_trace_3d.fiber_workflows','--config',str(sign)],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            signs=json.loads((trial/'sign/summary.json').read_text())['results'][0]['rows']
            row=dict(key=key,weights=values,perp_l1=r['score']['gt_perpendicular']['l1'],
                score=r['score'],final=r['final'],signs=signs,seconds=r['wall_seconds'],
                output_root=str(trial/'eval'),sha256=sha256_file(trial/'eval/solve/solution.npz'))
            if settings.get('gt_regressions'):
                from .fiber_gt_constraint_score import arrays
                from .fiber_shared_curve import saved_segment_metrics
                solved=json.loads((trial/'eval/solve/config.json').read_text())
                row['segments']=saved_segment_metrics(arrays(solved['tensor_cache']),arrays(trial/'eval/solve/solution.npz'))
                with lock:
                    validate=needs_gt_validation(row['perp_l1'],results.values(),settings.get('tolerance',1e-4))
                row['gt_regressions']={}
                def check_gt(item):
                    name,recipe=item
                    gt=trial/'gt'/name;gt.mkdir(parents=True,exist_ok=True)
                    params=dict(settings.get('gt_parameters',{}),**values)
                    request={'fiber_solver_trial':dict(recipe=recipe,output_root=str(gt),parameters=params)}
                    atomic_write(gt/'run.json',request,durable=False)
                    if not (gt/'joint/report.json').exists():
                        with (gt/'run.log').open('w') as log:
                            subprocess.run([sys.executable,'-m','vesuvius.neural_tracing.fiber_trace_3d.fiber_workflows','--config',str(gt/'run.json')],
                                stdout=log,stderr=subprocess.STDOUT,check=True)
                    report=json.loads((gt/'joint/report.json').read_text())
                    return name,dict(final=report['final'],seconds=report['seconds'],optimizer=report['optimizer'])
                if validate:
                    workers=int(settings.get('gt_workers',4))
                    if workers<1:raise ValueError('gt_workers must be positive')
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as checks:
                        row['gt_regressions']=dict(checks.map(check_gt,settings['gt_regressions'].items()))
                    guard=regression_guard(row['gt_regressions'],settings['gt_limits'])
                    row['gt_guard']=guard
                    row['gt_validation']='completed'
                    row['guarded_score']=row['perp_l1']+(0 if guard['passed'] else
                        1e6+1000*guard['signs']+10000*guard['reversals']+guard['geometry_excess'])
                else:
                    row['gt_guard']=None
                    row['gt_validation']='skipped: cannot improve validated incumbent'
                    row['guarded_score']=1e100
            if not math.isfinite(row['perp_l1']) or not math.isfinite(row['final']['loss']):
                raise ValueError('Nonfinite trial')
            atomic_write(cached,row,durable=False)
            with lock:
                results[key]=row
                atomic_write(root/'results.json',list(results.values()),durable=False)
            return row
        finally:
            with lane_condition:
                available.add(device)
                lane_condition.notify_all()
    def accept(name,values,best,old):
        acceptances.append(dict(parameter=name,value=values[name],previous=old['perp_l1'],
                                perp_l1=best['perp_l1'],key=best['key'],gt_guard=best.get('gt_guard')))
        atomic_write(root/'acceptances.json',acceptances,durable=False)
        atomic_write(root/'best.json',best,durable=False)
        print('IMPROVED '+json.dumps(acceptances[-1]),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(settings['devices'])) as pool:
        def many(candidates):
            atomic_write(root/'active.json',dict(candidates=candidates,completed=len(results)),durable=False)
            list(pool.map(evaluate,candidates))
        first=evaluate(initial);atomic_write(root/'best.json',first,durable=False)
        values,best=coordinate_search(initial,evaluate,grids,metric='guarded_score' if settings.get('gt_regressions') else 'perp_l1',
            tolerance=settings.get('tolerance',1e-4),accepted=accept,evaluate_many=many)
    atomic_write(root/'summary.json',dict(best=best,acceptances=acceptances,trials=len(results),
        termination='No improvement in final finite coordinate grid; not a global optimum'),durable=False)
    return best


def run(config):
    from lasagna.inference_provenance import atomic_write
    from .fiber_uv_layout import solve
    import gc
    import torch
    import numpy as np
    from .fiber_normal_sign import sha256_file

    if not config['gpu_manager'].get('exclusive'):
        raise ValueError('Weight search requires an exclusive managed lease')
    spec = config['loss_weight_search']
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    base = {k: v for k, v in config.items() if k != 'loss_weight_search'}
    if base.get('warm_start') or base.get('reference_initialization'):
        raise ValueError('This search requires fresh initialization')
    names = spec['weights']
    initial = {name: float(base[name]) for name in names}
    cohort=None
    baseline_root=spec.get('baseline_output_root')
    if spec.get('inlier_metrics'):
        from .fiber_uv_inlier_metrics import score_saved
        if set(names)!={'event_offset_longitudinal_weight','event_offset_transverse_weight','event_offset_anchor_weight'}:
            raise ValueError('Offset tuning may only vary the three regularizer weights')
        if baseline_root is None:raise ValueError('Frozen inlier cohort requires saved baseline')
        frozen=json.loads((Path(baseline_root)/'config.json').read_text())
        ignored={'output_root','code_provenance'}
        if {k:v for k,v in frozen.items() if k not in ignored}!={k:v for k,v in base.items() if k not in ignored}:
            raise ValueError('Saved offset baseline configuration mismatch')
        _,cohort=score_saved(frozen)
    if not all(math.isfinite(v) and v > 0 for v in initial.values()):
        raise ValueError('Search baseline weights must be finite and positive')
    grids = [{name: [initial[name] * factor for factor in factors] for name in names}
             for factors in spec.get('factor_grids',([.25, 1., 4.], [.25, .5, 1., 2., 4.]))]
    objective=spec.get('objective','ref_l1')
    scorer=None
    if objective=='gt_enabled_l1':
        from .fiber_gt_constraint_score import FrozenGTScorer
        scorer=FrozenGTScorer(spec['gt_score'],base)
    elif objective!='ref_l1':
        raise ValueError('Unknown search objective: '+objective)
    paths = [Path(base['tensor_cache']), Path(base['tensor_cache']).with_suffix('.json'),
             Path(base['reference_projection']), Path(base['reference_widths']['meta'])]
    paths += sorted(Path(__file__).parent.glob('fiber_uv*.py'))
    paths += [Path(__file__).with_name('fiber_reference_comparison.py'),
              Path(base['reference_subset'])]
    if baseline_root:
        paths += [Path(baseline_root)/name for name in ('config.json','solution.npz','summary.json')]
    signature = Path(base['reference_projection']).with_name('projection_signature.json')
    if signature.exists():
        paths.append(signature)
    manifest = dict(base=base, grids=grids, tolerance=1e-4,search_spec=spec,
                    hashes={str(p): sha256_file(p) for p in paths})
    if scorer:
        manifest.update(objective=objective,gt_score_hashes=scorer.hashes)
        path=Path(__file__).with_name('fiber_gt_constraint_score.py')
        manifest['hashes'][str(path)]=sha256_file(path)
    manifest_path = root / 'search_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Search configuration changed; use a new output root')
    atomic_write(manifest_path, manifest, durable=False)
    if cohort is not None:np.save(root/'inlier_cohort.npy',cohort)
    results = []
    acceptances = []
    announced_path = root / 'announced.json'
    announced = set(json.loads(announced_path.read_text())) if announced_path.exists() else set()

    def evaluate(values):
        key = configuration_key(values)
        trial_root = root / 'trials' / key
        reused=baseline_root is not None and values==initial
        if reused:trial_root=Path(baseline_root)
        child = dict(base, **values, output_root=str(trial_root))
        trial_root.mkdir(parents=True, exist_ok=True)
        saved_config = trial_root / 'config.json'
        if not reused and saved_config.exists() and json.loads(saved_config.read_text()) != child:
            raise ValueError('Trial config mismatch: ' + str(trial_root))
        if not reused:atomic_write(saved_config, child, durable=False)
        summary_path = trial_root / 'summary.json'
        if not summary_path.exists():
            atomic_write(root / 'active_trial.json', dict(key=key, weights=values), durable=False)
            print('WEIGHT SEARCH TRIAL ' + json.dumps(dict(key=key, weights=values)), flush=True)
            solve(child)
            gc.collect()
            torch.cuda.empty_cache()
        summary = json.loads(summary_path.read_text())
        if summary['final']['optimizer_updates'] != child['steps']:
            raise ValueError('Incomplete trial cannot be ranked')
        with np.load(trial_root / 'solution.npz') as solution:
            if not np.isfinite(solution['state']).all() or not np.isfinite(solution['widths']).all() or np.any(solution['widths'] <= 100):
                raise ValueError('Nonfinite final solution')
            with np.load(base['tensor_cache']) as data:
                for field in ('source_id', 'start', 'end', 'family'):
                    if not np.array_equal(solution[field], data[field]):
                        raise ValueError('Changed solution identity: ' + field)
        result = dict(metrics(summary), key=key, weights=values, output_root=str(trial_root))
        if cohort is not None:
            result['inlier'],_=score_saved(child,cohort)
            result['inlier_fine_loss']=result['inlier']['fixed']['balanced_fine_loss']
        if scorer:
            try:
                result.update(scorer.score(trial_root/'solution.npz',child))
            except ValueError as exc:
                if str(exc)!='No finite enabled GT constraints':raise
                result.update(gt_enabled_l1=None,invalid_reason=str(exc))
            result.update(objective_name=objective,objective_value=result[objective])
        if not results and result.get(objective) is None:
            raise ValueError('Baseline has no valid objective')
        if not any(row['key'] == key for row in results):
            results.append(result)
            atomic_write(root / 'results.json', results, durable=False)
            print('WEIGHT SEARCH RESULT '+json.dumps(result),flush=True)
        if len(results)==1:
            atomic_write(root / 'best.json',result,durable=False)
        return result

    def accept(name, values, best, old):
        row = dict(best, changed_weight=name, previous_ref_l1=old['ref_l1'])
        if scorer:row.update(previous_objective=old[objective])
        acceptances.append(row)
        atomic_write(root / 'acceptances.json', acceptances, durable=False)
        atomic_write(root / 'best.json', best, durable=False)
        if best['key'] not in announced:
            print('WEIGHT SEARCH IMPROVED ' + json.dumps(row), flush=True)
            announced.add(best['key'])
            atomic_write(announced_path, sorted(announced), durable=False)

    values, best = coordinate_search(initial, evaluate, grids, accepted=accept,metric=objective,
        secondary_metric='inlier_fine_loss' if cohort is not None else None)
    final = dict(best=best, weights=values, trials=len(results), acceptances=acceptances,
        termination='Every grid alternative tested at final incumbent; no accepted L1 improvement >1e-4'+
            (' also reducing fixed-cohort fine loss' if cohort is not None else ''),
        caveat='Reference is tuning data; retained cohort can change. Finite-grid coordinate optimum only.')
    atomic_write(root / 'best.json', best, durable=False)
    atomic_write(root / 'summary.json', final, durable=False)
    print('WEIGHT SEARCH COMPLETE ' + json.dumps(final), flush=True)
    return final
