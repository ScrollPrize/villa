"""Three-arm asynchronous piece-cost refinement, with persistent trial records."""
import concurrent.futures as futures
import json
import itertools
import math
from pathlib import Path
import subprocess
import sys
import time
from lasagna.inference_provenance import atomic_write,sha256_file
from .fiber_paired_weight_search import numeric_key,DEFAULT_FACTOR_GRIDS


def hold_existing(settings):
    """Interrupt only the coordinator; its executor drains running evaluations."""
    import os
    import signal
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Verified process control requires Linux /proc')
    pid=int(settings['pid'])
    command=Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    expected=str(Path(settings['config']).resolve()).encode()
    if len(command)!=5 or command[1:4]!=[b'-m',b'vesuvius.neural_tracing.fiber_trace_3d.fiber_workflows',b'--config'] or command[4]!=expected:
        raise ValueError('Coordinator PID does not match the exact requested config')
    children=[]
    if settings.get('stop_active',False):
        import psutil
        children=psutil.Process(pid).children(recursive=True)
    os.kill(pid,signal.SIGINT)
    for process in reversed(children):
        try:process.terminate()
        except psutil.NoSuchProcess:pass
    result=dict(pid=pid,acceptance='interrupted',running_evaluations='terminated' if children else 'allowed to drain',
                terminated_descendants=[p.pid for p in children],reason=settings['reason'])
    atomic_write(Path(settings['output_root'])/'acceptance_hold.json',result,durable=False)
    return result


def ranking(reports):
    gt=reports['selectable']['gt_enablement']
    total=sum(x['length_vx'] for x in gt)
    retained=sum(x['length_vx'] for x in gt if x['enabled_probability']>=.5)
    signs=sum(r['signs']['perpendicular']['violations'] for r in reports.values())
    rate=sum(r['signs']['perpendicular']['percent'] for r in reports.values())
    l1=sum(r['score']['gt_perpendicular']['l1'] for r in reports.values())
    if not all(map(math.isfinite,(total,retained,rate,l1))) or total<=0:raise ValueError('Invalid search metrics')
    return dict(gt_retained_vx=retained,gt_total_vx=total,gt_retained_percent=100*retained/total,
                perp_sign_sum=signs,perp_sign_percent_sum=rate,perp_l1_sum=l1,
                count_per_kept_fraction=signs*total/retained if retained>0 else None)


def better(a,b):
    if a['count_per_kept_fraction'] is None:return False
    if b['count_per_kept_fraction'] is None:return True
    delta=a['count_per_kept_fraction']-b['count_per_kept_fraction']
    if abs(delta)>1e-9:return delta<0
    return a['perp_l1_sum']<b['perp_l1_sum']-1e-6


def acceptance_limit_reached(target, count):
    """A reporting milestone is not a search termination condition."""
    return target is not None and count >= target


def focused_proposals(settings, best, grid):
    """Joint moves without changing fixed model parameters or the objective."""
    if grid != 0:
        return []
    result=[]
    for group in settings.get('proposal_groups', []):
        axes=group['axes']
        for choices in itertools.product(*(axis['values'] for axis in axes)):
            values=dict(best)
            for axis,choice in zip(axes,choices):
                for name in axis['parameters']:
                    if name not in settings['weights']:
                        raise ValueError('Grouped proposal changes fixed parameter: '+name)
                    value=best[name]*choice if axis['mode']=='factor' else choice
                    values[name]=int(round(value)) if name=='split_min_perpendicular' else float(f'{value:.12g}')
            if values==best:continue
            if all(settings['bounds'][name][0]<=values[name]<=settings['bounds'][name][1]
                   for name in settings['weights']):
                result.append((values,group['name']))
    return result


def rescore_completed(settings):
    """Re-rank complete triples only, without changing historical records."""
    candidates={}
    for previous in settings.get('reuse_roots',[]):
        for trial in sorted((Path(previous)/'trials').iterdir()):
            configs=sorted(trial.glob('*_base.json'))
            if not configs:continue
            cfg=json.loads(configs[0].read_text())['fiber_implicit_winding']
            values={k:cfg[k] for k in settings['initial_parameters']}
            key=numeric_key(values)
            candidate=candidates.setdefault(key,dict(key=key,parameters=values,reports={}))
            for name in ('free','joint','selectable'):
                report=trial/name/'search_report.json'
                if report.exists() and name not in candidate['reports']:
                    candidate['reports'][name]=json.loads(report.read_text())
    complete=[]
    for candidate in candidates.values():
        if len(candidate['reports'])!=3:continue
        candidate['ranking']=ranking(candidate['reports'])
        complete.append(candidate)
    return sorted(complete,key=lambda c:(
        c['ranking']['count_per_kept_fraction'] if c['ranking']['count_per_kept_fraction'] is not None else math.inf,
        c['ranking']['perp_l1_sum'],c['key']))


def historical_trial_key(previous,values):
    """A formerly fixed parameter may become tunable without invalidating a solve.

    The caller must still check the complete saved model config and checkpoint.
    """
    manifest=json.loads((Path(previous)/'manifest.json').read_text())
    keys=manifest['settings']['initial_parameters']
    return numeric_key({k:values[k] for k in keys})


def arm(settings,values,key,name,device):
    root=Path(settings['output_root'])/'trials'/key;out=root/name;out.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path(settings['base_config']).read_text())['fiber_implicit_winding']
    base.update(values);base['save_diagnostic_checkpoints']=False
    base['gpu_manager']=dict(base['gpu_manager'],exclusive=False)
    cfgpath=root/(name+'_base.json');atomic_write(cfgpath,dict(fiber_implicit_winding=base),durable=False)
    for previous in settings.get('reuse_roots',[]):
        old=Path(previous)/'trials'/historical_trial_key(previous,values)
        path=old/name/'search_report.json'
        if not path.exists():continue
        if json.loads((old/(name+'_base.json')).read_text())['fiber_implicit_winding']!=base:
            continue  # Formerly fixed parameter differs: this is a new solve.
        report=json.loads(path.read_text())
        if sha256_file(Path(report['output_root'])/'solve/solution.npz')!=report['solution_sha256']:
            raise ValueError('Cached solution changed')
        atomic_write(out/'search_report.json',report,durable=False)
        print('REUSE '+json.dumps(dict(key=key,arm=name,source=str(path))),flush=True)
        return report
    args=dict(base_config=str(cfgpath),gt_root=settings['gt_root'],output_root=str(out),devices=str(device))
    def execute(label,request,done):
        path=root/(name+'_'+label+'.json');atomic_write(path,request,durable=False)
        if not done.exists():
            with (root/(name+'_'+label+'.log')).open('a') as log:
                subprocess.run([sys.executable,'-m','vesuvius.neural_tracing.fiber_trace_3d.fiber_workflows','--config',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True)
        return json.loads(done.read_text())
    if name=='free':
        args['dataset']=settings['dataset']
        task='fiber_selected_score' if (out/'solve/summary.json').exists() and not (out/'results.json').exists() else 'fiber_selected_dataset_eval'
        r=execute('run',{task:args},out/'results.json')
        audit=execute('sign',dict(fiber_density_sign_report=dict(gt_evaluations=[str(out)],output_root=str(out/'gt_signs'))),out/'gt_signs/summary.json')
        signs={row['kind']:dict(violations=row['violations'],events=row['events'],percent=row['violation_percent']) for row in audit['results'][0]['rows'] if row['cohort']=='enabled'}
        gt=[]
    else:
        args.update(baseline_root=settings['baseline_root'],force_gt_enabled=name=='joint')
        result=execute('run',dict(fiber_joint_gt_eval=args),out/'results.json')
        r=execute('report',dict(fiber_joint_gt_eval=dict(args,report_only=True)),out/'comparison.json')
        signs=r['signs'];gt=result['gt_enablement']
    report=dict(score=r['score'],signs=signs,final=r['final'],matches=r['matches'],gt_enablement=gt,
                output_root=str(out),solution_sha256=sha256_file(out/'solve/solution.npz'))
    atomic_write(out/'search_report.json',report,durable=False)
    return report


def run(settings):
    root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    if settings.get('background',False):
        child=dict(settings,background=False)
        path=root/'background_config.json'
        atomic_write(path,dict(fiber_split_weight_search=child),durable=False)
        with (root/'coordinator.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-m','vesuvius.neural_tracing.fiber_trace_3d.fiber_workflows','--config',str(path)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result=dict(pid=process.pid,config=str(path),log=str(root/'coordinator.log'))
        atomic_write(root/'background_process.json',result,durable=False)
        print(json.dumps(result),flush=True)
        return result
    waiting=list(map(Path,settings.get('wait_for_reports',[])))
    deadline=time.monotonic()+settings.get('wait_for_reports_timeout_seconds',3600)
    while any(not p.exists() for p in waiting):
        atomic_write(root/'status.json',dict(status='waiting_for_reusable_baseline',
            missing=[str(p) for p in waiting if not p.exists()]),durable=False)
        if time.monotonic()>deadline:raise TimeoutError('Baseline reports did not finish')
        time.sleep(5)
    manifest=dict(settings=settings,code={str(p):sha256_file(p) for p in sorted(Path(__file__).parent.glob('fiber_*.py'))})
    reuse_values=[]
    for previous in settings.get('reuse_roots',[]):
        old=json.loads((Path(previous)/'manifest.json').read_text())
        for k in ('base_config','dataset','baseline_root','gt_root'):
            if old['settings'][k]!=settings[k]:raise ValueError('Reuse dataset differs: '+k)
        for path,digest in old['code'].items():
            if Path(path).name==Path(__file__).name:continue
            if sha256_file(path)!=digest:raise ValueError('Reuse model/scoring implementation changed: '+path)
        for trial in sorted((Path(previous)/'trials').iterdir()):
            configs=sorted(trial.glob('*_base.json'))
            if not configs:continue
            cfg=json.loads(configs[0].read_text())['fiber_implicit_winding']
            reuse_values.append({k:cfg[k] for k in settings['initial_parameters']})
    stamp=root/'manifest.json'
    if stamp.exists() and json.loads(stamp.read_text())!=manifest:raise ValueError('Search inputs/code changed; use a new root')
    atomic_write(stamp,manifest,durable=False)
    accepted=json.loads((root/'accepted.json').read_text()) if (root/'accepted.json').exists() else []
    initial=dict(settings['initial_parameters']);best=None;trials={};pending=[];active={}
    if settings.get('start_from_best_completed',False):
        rescored=rescore_completed(settings)
        atomic_write(root/'rescored_completed.json',rescored,durable=False)
        if not rescored or rescored[0]['ranking']['count_per_kept_fraction'] is None:
            raise ValueError('No valid completed trial available for restart')
        initial=dict(rescored[0]['parameters'])
        atomic_write(root/'restart_seed.json',rescored[0],durable=False)
        print('RESTART_SEED '+json.dumps(dict(key=rescored[0]['key'],parameters=initial,ranking=rescored[0]['ranking'])),flush=True)
    target=settings.get('accepted_improvements')
    if target is not None:
        target=int(target)
        if target<1:raise ValueError('accepted_improvements must be positive or null')
    devices=list(map(str,settings['devices']));idle=devices.copy()
    if len(set(devices))!=len(devices):raise ValueError('Distinct GPU lanes required')
    if not settings.get('replay_reused_candidates',True):reuse_values=[]
    start=time.monotonic();seen=set();generation=0;candidate_queue=[];cursor=0;grid=0;group_generation=-1
    def add(values,name=None):
        key=numeric_key(values)
        if key in seen:return
        seen.add(key);trial=dict(key=key,parameters=values,parameter=name,reports={},generation=generation)
        trials[key]=trial
        # Selectable gives retained fraction and a nonnegative lower bound
        # on the sum of three violation counts; retention alone cannot screen.
        pending.extend((key,arm_name) for arm_name in (('selectable','free','joint') if best is None else ('selectable',)))
    def proposals():
        nonlocal candidate_queue,cursor,grid,group_generation
        if group_generation!=generation:
            group_generation=generation
            candidate_queue.extend((values,name) for values,name in focused_proposals(settings,best['parameters'],grid)
                                   if numeric_key(values) not in seen)
            if candidate_queue:return
        while reuse_values:
            values=reuse_values.pop(0)
            if numeric_key(values) not in seen:
                candidate_queue.append((values,'rescored_existing'));return
        weights=settings['weights'];factors=settings.get('factor_grids',DEFAULT_FACTOR_GRIDS)
        while not candidate_queue and grid<len(factors):
            name=weights[cursor];value=best['parameters'][name]
            candidates=[value*f for f in factors[grid]]
            if grid==0:candidates+=settings.get('absolute_values',{}).get(name,[])
            if name in settings.get('discrete_values',{}):
                candidates=settings['discrete_values'][name] if grid==0 else []
            lo,hi=settings['bounds'][name]
            for v in candidates:
                v=int(round(v)) if name=='split_min_perpendicular' else float(f'{v:.12g}')
                values=dict(best['parameters'],**{name:v})
                if lo<=v<=hi and v!=value and numeric_key(values) not in seen:candidate_queue.append((values,name))
            cursor+=1
            if cursor==len(weights):cursor=0;grid+=1
    def publish(status):
        atomic_write(root/'status.json',dict(status=status,accepted=len(accepted),target=target,best=None if best is None else dict(key=best['key'],parameters=best['parameters'],ranking=best['ranking']),
            active=[dict(key=k,arm=n,gpu=g) for k,n,g in active.values()],queued=len(pending),trials=len(trials),seconds=time.monotonic()-start),durable=False)
    if accepted:
        best=json.loads((root/'trials'/accepted[-1]['key']/'result.json').read_text())
    else:add(initial)
    with futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        while not acceptance_limit_reached(target,len(accepted)):
            if best is not None:
                while len(pending)<len(idle):
                    proposals()
                    if not candidate_queue:break
                    values,name=candidate_queue.pop(0);add(values,name)
            while idle and pending:
                key,name=pending.pop(0);gpu=idle.pop(0);trial=trials[key]
                future=pool.submit(arm,settings,trial['parameters'],key,name,gpu);active[future]=(key,name,gpu)
                print('START '+json.dumps(dict(key=key,arm=name,gpu=gpu,parameters=trial['parameters'])),flush=True)
            publish('running')
            if not active:break
            done,_=futures.wait(active,timeout=30,return_when=futures.FIRST_COMPLETED)
            for future in done:
                key,name,gpu=active.pop(future);idle.append(gpu);trial=trials[key]
                try:report=future.result()
                except Exception as error:
                    atomic_write(root/'error.json',dict(key=key,arm=name,error=repr(error)),durable=False)
                    raise
                trial['reports'][name]=report
                print('DONE '+json.dumps(dict(key=key,arm=name,loss=report['final']['loss'])),flush=True)
                if name=='selectable' and best is not None and len(trial['reports'])==1:
                    retained=sum(x['length_vx'] for x in report['gt_enablement'] if x['enabled_probability']>=.5)
                    total=sum(x['length_vx'] for x in report['gt_enablement'])
                    lower_bound=report['signs']['perpendicular']['violations']*total/retained if retained>0 else None
                    incumbent=best['ranking']['count_per_kept_fraction']
                    if lower_bound is None or (incumbent is not None and lower_bound>incumbent+1e-9):
                        atomic_write(root/'trials'/key/'screened.json',dict(optimistic_ratio=lower_bound,invalid_zero_retention=retained<=0,incumbent=best['key']),durable=False)
                    else:pending[0:0]=[(key,'free'),(key,'joint')]
                if len(trial['reports'])==3:
                    trial['ranking']=ranking(trial['reports'])
                    atomic_write(root/'trials'/key/'result.json',trial,durable=False)
                    if best is None:best=trial;atomic_write(root/'baseline.json',trial,durable=False)
                    elif better(trial['ranking'],best['ranking']):
                        accepted.append(dict(key=key,parameters=trial['parameters'],ranking=trial['ranking'],previous=best['key']))
                        best=trial;generation+=1;candidate_queue=[];grid=0
                        # Drop unstarted old proposals; running solves remain useful
                        # evidence and may still win against the new incumbent.
                        pending=[job for job in pending if trials[job[0]]['reports']]
                        atomic_write(root/'accepted.json',accepted,durable=False)
                        atomic_write(root/'best.json',best,durable=False)
                        print('ACCEPT '+json.dumps(accepted[-1]),flush=True)
                if acceptance_limit_reached(target,len(accepted)):break
        # Only an explicit acceptance limit or exhausted proposals stops search.
        # Drain already-running arms cleanly to release their manager grants.
        publish('draining' if active else 'finished')
        for future in futures.as_completed(list(active)):
            key,name,gpu=active.pop(future);trial=trials[key]
            trial['reports'][name]=future.result()
            if len(trial['reports'])==3:
                trial['ranking']=ranking(trial['reports']);atomic_write(root/'trials'/key/'result.json',trial,durable=False)
        publish('acceptance_limit_reached' if acceptance_limit_reached(target,len(accepted)) else 'search_exhausted')
    return dict(best=best,accepted=accepted)
