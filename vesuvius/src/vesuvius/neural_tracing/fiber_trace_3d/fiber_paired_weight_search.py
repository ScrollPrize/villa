"""Shared-parameter, seven-problem search ranked by GT perpendicular ordering.

Reuse existing solvers, frozen GT scorers, configuration hashes and GT guard.
No model code or scientific defaults are altered by this controller.
"""
import concurrent.futures
import json
import math
import queue
import subprocess
import sys
from pathlib import Path
from lasagna.inference_provenance import atomic_write, sha256_file
from .fiber_uv_weight_search import configuration_key, regression_guard


def numeric_key(values):
    return configuration_key({k:float(v) for k,v in values.items()})


def rank_pair(reports, guard):
    count=0; l1=0.
    for name in ('free','joint'):
        r=reports[name]
        count+=r['signs']['perpendicular']['violations']
        l1+=r['score']['gt_perpendicular']['l1']
        if not math.isfinite(l1) or not math.isfinite(r['final']['loss']):
            raise ValueError('Nonfinite large-run metrics')
    return dict(passed=guard['passed'],perp_sign_sum=count,perp_l1_sum=l1)


def better(candidate, incumbent, tolerance=1e-6):
    if not candidate['passed']:return False
    if not incumbent['passed']:return True
    a,b=candidate['perp_sign_sum'],incumbent['perp_sign_sum']
    return a<b or (a==b and candidate['perp_l1_sum']<incumbent['perp_l1_sum']-tolerance)


def refine(initial,evaluate,parameters,factors,bounds,accepted,tolerance=1e-6,absolute_values=None):
    """Recenter and restart every parameter/grid after each accepted candidate."""
    current=dict(initial);best=evaluate(current)
    while True:
        improved=False
        for grid in factors:
            for name in parameters:
                lo,hi=bounds[name]
                values=[float(f'{current[name]*factor:.12g}') for factor in grid]
                if grid==factors[0]:values+=list((absolute_values or {}).get(name,[]))
                for value in dict.fromkeys(values):
                    if value==current[name] or not lo<=value<=hi:continue
                    candidate=dict(current,**{name:value});result=evaluate(candidate)
                    if better(result['ranking'],best['ranking'],tolerance):
                        old=best;current,best=candidate,result
                        accepted(name,current,best,old)
                        improved=True;break
                if improved:break
            if improved:break
        if not improved:return best


DEFAULT_FACTOR_GRIDS = ((.5, 2.), (.75, 1.5), (.9, 1.1), (.97, 1.03), (.99, 1.01))


def run(settings):
    settings = dict(settings)
    settings.setdefault('factor_grids', [list(pair) for pair in DEFAULT_FACTOR_GRIDS])
    if 'width_prior' in settings['weights']:
        raise ValueError('width_prior was removed and must not be tuned')
    root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path(settings['base_config']).read_text())['fiber_implicit_winding']
    base.update(settings.get('base_overrides',{}))
    canonical=json.loads(Path(settings['canonical_baseline']).read_text())['parameters']
    names=settings['weights'];initial={k:float(base[k]) for k in names}
    initial.update(settings.get('initial_parameters',{}))
    devices=list(map(str,settings['devices']))
    screened=bool(settings.get('screen_no_gt',False))
    if len(set(devices))!=len(devices) or (len(devices)<2 if screened else len(devices)!=4):
        raise ValueError('Distinct GPU lanes required (four for legacy mode)')
    paths=[Path(settings['base_config']),Path(settings['canonical_baseline'])]
    paths+=list(map(Path,settings['gt_regressions'].values()))
    paths+=list(Path(__file__).parent.glob('fiber_*.py'))
    if screened:paths.append(Path(__file__).with_name('paired_search_schedule.py'))
    if settings.get('asynchronous_queue'):paths.append(Path(__file__).with_name('paired_async_schedule.py'))
    manifest=dict(settings=settings,base=base,hashes={str(p):sha256_file(p) for p in sorted(paths)})
    stamp=root/'manifest.json'
    if stamp.exists() and json.loads(stamp.read_text())!=manifest:
        raise ValueError('Search code or configuration changed; use a new root')
    atomic_write(stamp,manifest,durable=False)
    results={};acceptances=[];imported_keys={}
    def key_for(values):
        # JSON 1 and 1.0 are identical parameters, not different experiments.
        fingerprint=numeric_key(values)
        return imported_keys.get(fingerprint,fingerprint) if screened else configuration_key(values)

    def execute(path,request,done):
        path.parent.mkdir(parents=True,exist_ok=True)
        def scientific(value):
            if isinstance(value,dict):return {k:scientific(v) for k,v in value.items() if k!='devices'}
            if isinstance(value,list):return [scientific(v) for v in value]
            return value
        if path.exists() and scientific(json.loads(path.read_text()))!=scientific(request):
            raise ValueError('Changed cached job '+str(path))
        atomic_write(path,request,durable=False)
        if not done.exists():
            with path.with_suffix('.log').open('a') as log:
                subprocess.run([sys.executable,'-m','vesuvius.neural_tracing.fiber_trace_3d.fiber_workflows','--config',str(path)],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        return json.loads(done.read_text())

    def large(trial,values,joint,device=None):
        name='joint' if joint else 'free';out=trial/name
        cfg=dict(base,**values,save_diagnostic_checkpoints=False)
        cfgpath=trial/(name+'_base.json')
        atomic_write(cfgpath,dict(fiber_implicit_winding=cfg),durable=False)
        args=dict(base_config=str(cfgpath),gt_root=settings['gt_root'],output_root=str(out),
                  devices=device if device is not None else devices[1 if joint else 0])
        if joint:
            args['baseline_root']=settings['baseline_root']
            execute(trial/'joint_run.json',dict(fiber_joint_gt_eval=args),out/'results.json')
            r=execute(trial/'joint_report.json',dict(fiber_joint_gt_eval=dict(args,report_only=True)),out/'comparison.json')
        else:
            args['dataset']=settings['dataset']
            if (out/'solve/summary.json').exists() and not (out/'results.json').exists():
                # Reporting can be retried independently after a completed solve.
                r=execute(trial/'free_score.json',dict(fiber_selected_score=args),out/'results.json')
            else:
                r=execute(trial/'free_run.json',dict(fiber_selected_dataset_eval=args),out/'results.json')
            sign=execute(trial/'free_sign.json',dict(fiber_density_sign_report=dict(
                gt_evaluations=[str(out)],output_root=str(out/'gt_signs'))),out/'gt_signs/summary.json')
            r=dict(r,signs={row['kind']:dict(violations=row['violations'],events=row['events'],percent=row['violation_percent'])
                for row in sign['results'][0]['rows'] if row['cohort']=='enabled'})
        return dict(score=r['score'],signs=r['signs'],final=r['final'],matches=r['matches'],
                    output_root=str(out),solution_sha256=sha256_file(out/'solve/solution.npz'))

    def small_lane(trial,values,lane,device=None,only=None):
        reports={}
        for index,(name,recipe) in enumerate(settings['gt_regressions'].items()):
            if only is not None:
                if name!=only:continue
            elif index%2!=lane:continue
            wrapper=json.loads(Path(recipe).read_text())['fiber_solver_trial']
            original=json.loads(Path(wrapper['recipe']).read_text())['fiber_gt_convergence_probe']
            # Same scientific values on every test, except the explicitly
            # defined test mode, width initialization and sparse-prior control.
            params=dict(wrapper['parameters'],**canonical)
            params.update({k:base[k] for k in settings['shared_parameters']})
            params.update(values)
            if name=='sign_sparse_control':params['arc_prior_spacing_vx']=None
            out=trial/'gt'/name
            request=dict(original,output_root=str(out),devices=device if device is not None else devices[2+lane],
                parameters=dict(original.get('parameters',{}),**params),gpu_manager=dict(
                    exclusive=False,cpu_cores=4,vram_bytes_per_gpu=4*1024**3,label='paired-GT-'+name))
            report=execute(trial/('gt_'+name+'.json'),dict(fiber_gt_convergence_probe=request),out/'report.json')['joint']
            reports[name]=dict(final=report['final'],seconds=report['seconds'],optimizer=report['optimizer'],
                               output_root=str(out),solution_sha256=sha256_file(out/'joint/solution.npz'))
        return reports

    def evaluate(values):
        key=key_for(values)
        if key in results:return results[key]
        trial=(Path(settings['reuse_initial_trial']) if values==initial and settings.get('reuse_initial_trial')
               else root/'trials'/key)
        trial.mkdir(parents=True,exist_ok=True)
        cached=trial/'rank.json'
        if cached.exists():
            result=json.loads(cached.read_text())
            if result['parameters']!=values:raise ValueError('Changed trial parameters')
            for report in result['large'].values():
                if report['solution_sha256']!=sha256_file(Path(report['output_root'])/'solve/solution.npz'):
                    raise ValueError('Cached solution changed')
        else:
            atomic_write(root/'active.json',dict(key=key,parameters=values,completed=len(results)),durable=False)
            print('TRIAL '+json.dumps(dict(key=key,parameters=values)),flush=True)
            # Only one candidate owns the four fixed lanes at a time; all
            # seven problems are checked, regardless of large-run ranking.
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                free=pool.submit(large,trial,values,False)
                joint=pool.submit(large,trial,values,True)
                a=pool.submit(small_lane,trial,values,0)
                b=pool.submit(small_lane,trial,values,1)
                reports=dict(free=free.result(),joint=joint.result())
                gt=dict(a.result(),**b.result())
            guard=regression_guard(gt,settings['gt_limits'])
            result=dict(key=key,parameters=values,large=reports,gt=gt,gt_guard=guard,
                        ranking=rank_pair(reports,guard))
            atomic_write(cached,result,durable=False)
            print('RESULT '+json.dumps(dict(key=key,ranking=result['ranking'],guard=guard)),flush=True)
        results[key]=result
        atomic_write(root/'results.json',list(results.values()),durable=False)
        if not (root/'best.json').exists():atomic_write(root/'best.json',result,durable=False)
        return result

    def accepted(name,values,best,old):
        row=dict(parameter=name,value=values[name],key=best['key'],previous=old['ranking'],ranking=best['ranking'])
        acceptances.append(row)
        atomic_write(root/'acceptances.json',acceptances,durable=False)
        atomic_write(root/'best.json',best,durable=False)
        print('IMPROVED '+json.dumps(row),flush=True)

    if screened:
        from .paired_search_schedule import can_improve,optimistic_rank,cyclic_refine
        # Carry forward completed evaluations without rerunning the search.
        # Only scheduling code may differ from an imported scientific manifest.
        imported_free={}
        for old_root in map(Path,settings.get('reuse_roots',[])):
            old_manifest=json.loads((old_root/'manifest.json').read_text())
            for filename,digest in old_manifest['hashes'].items():
                if Path(filename).name=='fiber_paired_weight_search.py':continue
                if sha256_file(Path(filename))!=digest:
                    raise ValueError('Imported scientific code/config changed: '+filename)
            for item in json.loads((old_root/'results.json').read_text()):
                for report in item['large'].values():
                    if report['solution_sha256']!=sha256_file(Path(report['output_root'])/'solve/solution.npz'):
                        raise ValueError('Imported solution changed')
                results[item['key']]=item
                imported_keys[numeric_key(item['parameters'])]=item['key']
            # A completed no-GT screen can survive interruption of the other lanes.
            for trial in (old_root/'trials').iterdir():
                result_file=trial/'free/results.json';sign_file=trial/'free/gt_signs/summary.json'
                if not result_file.exists() or not sign_file.exists():continue
                r=json.loads(result_file.read_text());sign=json.loads(sign_file.read_text())
                old_config=json.loads((trial/'free/solve/config.json').read_text())
                imported_keys[numeric_key({k:old_config[k] for k in names})]=trial.name
                if r['solution_sha256']!=sha256_file(trial/'free/solve/solution.npz'):
                    raise ValueError('Imported no-GT solution changed')
                imported_free[trial.name]=dict(score=r['score'],final=r['final'],matches=r['matches'],
                    output_root=str(trial/'free'),solution_sha256=r['solution_sha256'],
                    signs={x['kind']:dict(violations=x['violations'],events=x['events'],percent=x['violation_percent'])
                        for x in sign['results'][0]['rows'] if x['cohort']=='enabled'})
        atomic_write(root/'results.json',list(results.values()),durable=False)
        slots=queue.Queue()
        for device in devices:slots.put(device)
        def scheduled(function,*args,**kwargs):
            device=slots.get()
            try:return function(*args,device=device,**kwargs)
            finally:slots.put(device)

        def evaluate_many(candidates,incumbent):
            pending={};parts={};trials={};values_by_key={}
            tolerance=settings.get('l1_tie_tolerance',1e-6)
            def save(key,result):
                results[key]=result
                atomic_write(trials[key]/'rank.json',result,durable=False)
                atomic_write(root/'results.json',list(results.values()),durable=False)
                print('SCREEN_RESULT '+json.dumps(dict(key=key,ranking=result['ranking'],
                    screened_out=result.get('screened_out',False))),flush=True)
            def continue_candidate(key,free,pool):
                if not can_improve(free,incumbent,tolerance):
                    save(key,dict(key=key,parameters=values_by_key[key],large=dict(free=free),
                        screened_out=True,optimistic_ranking=optimistic_rank(free),
                        ranking=dict(optimistic_rank(free),passed=False),
                        screening_incumbent=incumbent,
                        skip_reason='Cannot beat incumbent even with zero GT-assisted sign errors and L1'))
                    return
                parts[key]=dict(free=free,gt={})
                pending[pool.submit(scheduled,large,trials[key],values_by_key[key],True)]=(key,'joint')
                for name in settings['gt_regressions']:
                    pending[pool.submit(scheduled,small_lane,trials[key],values_by_key[key],0,only=name)]=(key,name)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
                for values in candidates:
                    key=key_for(values)
                    if key in results:continue
                    trial=root/'trials'/key;trial.mkdir(parents=True,exist_ok=True)
                    trials[key]=trial;values_by_key[key]=values
                    cached=trial/'rank.json'
                    if cached.exists():
                        item=json.loads(cached.read_text())
                        if item['parameters']!=values:raise ValueError('Cached parameters changed')
                        results[key]=item;continue
                    if (trial/'free_report.json').exists():
                        imported_free[key]=json.loads((trial/'free_report.json').read_text())
                    if key in imported_free:
                        parts[key]=dict(cached_free=imported_free[key])
                    else:
                        pending[pool.submit(scheduled,large,trial,values,False)]=(key,'free')
                # Queue continuations only after all new no-GT screens have been submitted.
                for key in list(parts):
                    if 'cached_free' in parts[key]:continue_candidate(key,parts.pop(key)['cached_free'],pool)
                while pending:
                    done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        key,stage=pending.pop(future);report=future.result()
                        if stage=='free':
                            atomic_write(trials[key]/'free_report.json',report,durable=False)
                            continue_candidate(key,report,pool)
                        else:
                            if stage=='joint':parts[key]['joint']=report
                            else:parts[key]['gt'].update(report)
                            part=parts[key]
                            if 'joint' in part and len(part['gt'])==len(settings['gt_regressions']):
                                reports=dict(free=part['free'],joint=part['joint'])
                                guard=regression_guard(part['gt'],settings['gt_limits'])
                                save(key,dict(key=key,parameters=values_by_key[key],large=reports,
                                    gt=part['gt'],gt_guard=guard,ranking=rank_pair(reports,guard)))
                    atomic_write(root/'running_jobs.json',[dict(key=k,stage=s) for k,s in pending.values()],durable=False)
            return [results[key_for(v)] for v in candidates]

        def progress(sweep,name,candidates,best):
            atomic_write(root/'active.json',dict(sweep=sweep,parameter=name,
                completed=len(results),best_key=best['key'],
                candidates=[dict(key=key_for(v),value=v[name]) for v in candidates]),durable=False)
        if key_for(initial) in results:
            atomic_write(root/'best.json',results[key_for(initial)],durable=False)
        if settings.get('asynchronous_queue'):
            from .paired_async_schedule import asynchronous_refine
            stage_cache={(key,'free'):report for key,report in imported_free.items()}
            def async_work(values,stage,device):
                trial=root/'trials'/key_for(values);trial.mkdir(parents=True,exist_ok=True)
                path=trial/'stage_reports'/(stage+'.json')
                if path.exists():return json.loads(path.read_text())
                if stage in ('free','joint'):report=large(trial,values,stage=='joint',device=device)
                else:report=small_lane(trial,values,0,device=device,only=stage)
                path.parent.mkdir(parents=True,exist_ok=True)
                atomic_write(path,report,durable=False)
                return report
            def async_finish(key,values,parts):
                gt={}
                for name in settings['gt_regressions']:gt.update(parts[name])
                guard=regression_guard(gt,settings['gt_limits'])
                reports=dict(free=parts['free'],joint=parts['joint'])
                return dict(key=key,parameters=values,large=reports,gt=gt,gt_guard=guard,
                            ranking=rank_pair(reports,guard))
            def async_save(result):
                trial=root/'trials'/result['key'];trial.mkdir(parents=True,exist_ok=True)
                atomic_write(trial/'rank.json',result,durable=False)
                atomic_write(root/'results.json',list(results.values()),durable=False)
                print('ASYNC_RESULT '+json.dumps(dict(key=result['key'],ranking=result['ranking'],
                    screened_out=result.get('screened_out',False))),flush=True)
            def async_progress(status):
                atomic_write(root/'active.json',status,durable=False)
                atomic_write(root/'running_jobs.json',status['running'],durable=False)
            best=asynchronous_refine(evaluate(initial),names,settings['factor_grids'],settings['bounds'],
                devices,list(settings['gt_regressions']),key_for,results,stage_cache,async_work,async_finish,
                async_save,accepted,async_progress,settings.get('l1_tie_tolerance',1e-6),settings.get('absolute_values'))
        else:
            best=cyclic_refine(initial,evaluate,evaluate_many,names,settings['factor_grids'],settings['bounds'],accepted,
                settings.get('l1_tie_tolerance',1e-6),settings.get('absolute_values'),len(devices),progress)
    else:
        best=refine(initial,evaluate,names,settings['factor_grids'],settings['bounds'],accepted,
                    settings.get('l1_tie_tolerance',1e-6),settings.get('absolute_values'))
    summary=dict(best=best,trials=len(results),acceptances=acceptances,
        termination='No improvement in all recentered factor grids within configured bounds; not a global optimum')
    atomic_write(root/'summary.json',summary,durable=False)
    return summary
