"""Whole-fiber sign/broken optimization; CPU preparation, manager-owned CUDA solve."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import subprocess

import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from .fiber_graph_store import FiberGraph, store_digest, write_results


def sign_alignment_terms(sign_logits, prior, i, j, dot, keep=None):
    """Shared binary vector-sign agreement with fixed evidence normalization."""
    s = 2 * sign_logits.sigmoid() - 1
    weight = 1 if keep is None else keep[i]*keep[j]
    pair = (weight*(dot.abs()-dot*s[i]*s[j])*.5).sum() / dot.abs().sum().clamp_min(1e-12)
    prior_loss = ((1-prior*s)*.5).mean()
    return pair,prior_loss


def losses(sign_logits, broken_logits, prior, i, j, dot, prior_weight, broken_penalty):
    """Fixed denominator; disabling everything never removes the broken penalty."""
    b = broken_logits.sigmoid()
    pair,outward = sign_alignment_terms(sign_logits,prior,i,j,dot,1-b)
    broken = b.mean()
    return pair + prior_weight*outward + broken_penalty*broken, pair, outward, broken


STRICT_MAX_ANGLE_DEGREES = 30.   # internal evaluation: nearly parallel normal axes ...
STRICT_MAX_WINDING_DISTANCE = 1.  # ... on crossings less than one winding apart


def winding_distance_factor(distance, mode):
    """Per-event weight factor from the measured winding distance: ``inverse_square`` gives
    min(1, 1/d^2) (1 within one winding, 1/4 at two, 1/9 at three); an unmeasured (NaN)
    distance keeps 1. ``None`` disables the factor."""
    import torch
    if mode is None:
        return torch.ones_like(distance)
    if mode != 'inverse_square':
        raise ValueError('winding_distance_weight must be inverse_square or none')
    d = distance.abs()
    factor = torch.where(torch.isfinite(d) & (d > 1.), 1./(d*d).clamp_min(1e-12), torch.ones_like(d))
    return factor


def metrics(u, v, a, i, j, dot, distance=None, factor=None):
    """Event-weighted agreement (weight |dot| times the winding-distance factor), plus the
    unweighted internal evaluation over nearly parallel normals within one winding."""
    import torch, math
    with torch.no_grad():
        s = torch.where(u >= 0, 1., -1.)
        kept = (v[i] < 0) & (v[j] < 0)
        wrong = dot*s[i]*s[j] < 0
        weight = dot.abs() if factor is None else dot.abs()*factor
        denominator = weight.sum().clamp_min(1e-12)
        kept_weight = weight[kept].sum()
        retained_error = 100*weight[kept & wrong].sum()/kept_weight if kept_weight > 0 else None
        strict = {}
        if distance is not None:
            close = dot.abs() > math.cos(math.radians(STRICT_MAX_ANGLE_DEGREES))
            near = torch.isfinite(distance) & (distance.abs() < STRICT_MAX_WINDING_DISTANCE)
            sel = close & near
            selected = int(sel.sum())
            strict = dict(strict_events=selected,
                          strict_disagreement_percent=float(100*(wrong & sel).sum()/max(selected, 1)),
                          strict_retained_disagreement_percent=float(100*(wrong & sel & kept).sum()/max(int((sel & kept).sum()), 1)))
        return dict(broken_fibers_percent=float(100*(v>=0).float().mean()),
                    expected_broken_percent=float(100*v.sigmoid().mean()),
                    retained_events_percent=float(100*kept.float().mean()) if kept.numel() else None,
                    weighted_disagreement_all_percent=float(100*weight[wrong].sum()/denominator),
                    weighted_disagreement_retained_percent=float(retained_error) if retained_error is not None else None,
                    retained_disagreement_original_weight_percent=float(100*weight[kept & wrong].sum()/denominator),
                    outward_agreement=float((a*s).mean()),
                    uncertain_signs_percent=float(100*((u.sigmoid()>.1)&(u.sigmoid()<.9)).float().mean()),
                    **strict)


CROSSING_FILTER_SUFFIX = '.crossing-filter.bin'


def crossing_filter_path(config, database):
    """The extraction's multi-crossing filter (``vc_fiber_constraints`` with ``crossing_filter_output``),
    or None when the stage runs without one. ``crossing_filter: auto`` names the file the extract stage
    writes next to its store; a missing file is an error, never a silent unfiltered solve."""
    selected = config.get('crossing_filter')
    if selected is None or selected is False:
        return None
    database = Path(database)
    path = database.with_name(database.stem+CROSSING_FILTER_SUFFIX) if selected in (True, 'auto') else Path(selected)
    if not path.exists() or not Path(str(path)+'.json').exists():
        raise FileNotFoundError(f'No crossing filter {path}; run the extract stage with crossing_filter, or set crossing_filter to false')
    return path


CROSSING_FINDINGS = dict(nothing_left=4, mixed=8, mixed_near=16, several_left=32)
# Without signs a crossing can only be dropped. Nothing left: the partner passes the normal surface and
# returns. Mixed: the encounter holds strong crossings of both relations, so it decides no sign.
CROSSING_FIRST_PASS = ('nothing_left', 'mixed')


def read_crossing_filter(path, graph):
    """Per event of the graph: the finding bits and the |dot| of the nearest crossing stating the
    other relation (0 where there is none or the event is not in the filter)."""
    statement = json.loads(Path(str(path)+'.json').read_text())
    if statement.get('format') != 'fiber_crossing_filter' or statement.get('version') != 2:
        raise ValueError(f'{path} is not a version 2 crossing filter')
    if str(statement['store']) != str(graph.source['store_uuid']) or int(statement['events']) != len(graph.event_ids):
        raise ValueError('Crossing filter was computed for another extraction store')
    records = np.fromfile(path, dtype=np.dtype([('event', '<i8'), ('findings', '<i8'), ('alternative', '<f8')]))
    if len(records) != int(statement['sign_events']):
        raise ValueError('Crossing filter file and statement differ')
    at = np.clip(np.searchsorted(records['event'], graph.event_ids), 0, max(len(records)-1, 0))
    known = records['event'][at] == graph.event_ids if len(records) else np.zeros(len(graph.event_ids), bool)
    findings = np.where(known, records['findings'][at], 0) if len(records) else np.zeros(len(graph.event_ids), np.int64)
    alternative = np.where(known, records['alternative'][at], 0.) if len(records) else np.zeros(len(graph.event_ids))
    return findings, alternative


def prepare(config, *, graph=None, return_arrays=False, prior=None, signs=None):
    """Convert fixed evidence once; no geometry or model evaluation in optimization.

    With a crossing filter the evidence depends on the pass. Without ``signs`` (first pass) the events
    with a finding of ``crossing_filter_first_pass`` are dropped. With the whole-fiber ``signs`` of a
    first pass (per graph fiber, +1/-1, 0 unknown) only the nothing-left events are dropped, and an
    event that disagrees under those signs takes the dot of the closest crossing that agrees, where
    its encounter has one."""
    started = time.monotonic()
    db_path, prior_path = Path(config['database']), Path(config['priors'])
    filter_path = crossing_filter_path(config, db_path)
    signature = dict(version=5, database_sha256=store_digest(db_path), priors_sha256=sha256_file(prior_path),
                     crossing_filter_sha256=sha256_file(filter_path) if filter_path else None)
    cache = Path(config.get('tensor_cache',Path(config.get('output_root','.'))/'tensors.npz'))
    stamp = cache.with_suffix('.json')
    if signs is not None and not return_arrays:
        raise ValueError('A signed pass is prepared in memory')
    if not return_arrays and cache.exists() and stamp.exists() and json.loads(stamp.read_text())['signature'] == signature:
        return cache
    prior = json.loads(prior_path.read_text()) if prior is None else prior
    graph = FiberGraph(db_path) if graph is None else graph
    from .fiber_graph_store import snapshot_digest
    if snapshot_digest(graph.source) != signature['database_sha256']:
        raise ValueError('Prepared graph belongs to another snapshot')
    if prior['format'] != 'fiber-outward-priors' or prior.get('source_store') != graph.source:
        raise ValueError('Priors were generated from another store snapshot')
    ids = graph.ids
    if not np.array_equal(ids, np.asarray(prior['source_ids'], dtype=str)):
        raise ValueError('Prior/source fiber IDs differ')
    a = np.asarray(prior['agreement'], dtype=np.float32)
    if len(a) != len(ids) or not np.isfinite(a).all() or np.any(abs(a) > 1.000001):
        raise ValueError('Invalid radial priors')
    eligible = ~graph.parallel
    valid = eligible & graph.valid_a & graph.valid_b
    total = int(eligible.sum())
    filtered = replaced = 0
    alternative = None
    if signs is not None and not filter_path:
        raise ValueError('A signed pass needs the crossing filter')
    if filter_path:
        findings, alternative = read_crossing_filter(filter_path, graph)
        rules = ('nothing_left',) if signs is not None else tuple(config.get('crossing_filter_first_pass', CROSSING_FIRST_PASS))
        if set(rules)-set(CROSSING_FINDINGS):
            raise ValueError('Unknown crossing filter finding in crossing_filter_first_pass')
        listed = (findings & sum(CROSSING_FINDINGS[rule] for rule in rules)) != 0
        filtered = int((valid & listed).sum())
        valid = valid & ~listed
    event_ids = graph.event_ids[valid]
    indices = np.column_stack((graph.i[valid], graph.j[valid]))
    dot = np.sum(graph.normal_a[valid]*graph.normal_b[valid], axis=1)
    if signs is not None:
        signs = np.asarray(signs, dtype=np.float64)
        if len(signs) != len(graph.ids):
            raise ValueError('First-pass signs do not cover the graph fibers')
        other = alternative[valid]
        swap = (dot*signs[indices[:,0]]*signs[indices[:,1]] < 0) & (other > 0)
        dot = np.where(swap, -np.sign(dot)*other, dot)
        replaced = int(swap.sum())
    dot=dot.astype(np.float32)
    if not np.isfinite(dot).all() or np.any(abs(dot)>1.000001):
        raise ValueError('Invalid normal dot products')
    dot=np.clip(dot,-1,1)
    # Measured winding distance per event (NaN when the extractor measured none);
    # the solve can weight events by it and the strict metric selects on it.
    distance=np.asarray(graph.distance[valid],dtype=np.float32)
    data=dict(ids=ids,lengths=graph.lengths.copy() if return_arrays else graph.lengths,
              prior=a,i=indices[:,0],j=indices[:,1],dot=dot,distance=distance,event_ids=event_ids)
    degree=np.bincount(indices.ravel(),minlength=len(ids))
    stats=dict(signature=signature,fibers=len(ids),perpendicular_events=total,
               used_events=len(dot),invalid_events_skipped=total-len(dot)-filtered,
               crossing_filter_dropped=filtered,crossing_filter_replaced=replaced,
               events_with_winding_distance=int(np.isfinite(distance).sum()),
               isolated_fibers=int((degree==0).sum()),tensor_bytes=0,
               seconds=time.monotonic()-started)
    if return_arrays:return data,stats
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache.with_suffix('.partial.npz');np.savez(temporary,**data);temporary.replace(cache)
    stats['tensor_bytes']=cache.stat().st_size
    atomic_write(stamp,stats,durable=False)
    print('tensor preparation '+json.dumps(stats),flush=True)
    return cache


def solve(config):
    if config.get('optimizer')=='wrapped_irls':
        from .fiber_wrapped_stages import solve_stage
        return solve_stage(config,'normal_sign')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for production sign solve; no CPU fallback')
    torch.set_num_threads(4)
    torch.manual_seed(config.get('seed',0))
    torch.use_deterministic_algorithms(True)
    data=np.load(config['tensor_cache'],allow_pickle=False)
    device=torch.device('cuda:0')
    a=torch.from_numpy(data['prior']).to(device)
    i=torch.from_numpy(data['i']).to(device)
    j=torch.from_numpy(data['j']).to(device)
    dot=torch.from_numpy(data['dot']).to(device)
    if not len(a): raise ValueError('No fibers')
    initial=torch.where(a==0,torch.full_like(a,.001),2*a)
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists(): raise FileExistsError('Refusing to overwrite completed sign solve')
    steps=int(config.get('steps',2000));every=int(config.get('log_every',100))
    lr=float(config.get('lr',.05));weak=float(config.get('prior_weight',.001))
    penalties=config.get('broken_penalties',[.01,.1,1.])
    if steps<=0 or every<=0 or lr<=0 or weak<0 or any(p<=0 for p in penalties): raise ValueError('Invalid optimization settings')
    summaries=[]
    baseline_v=torch.full_like(a,-30.)
    baselines={name:metrics(u,baseline_v,a,i,j,dot) for name,u in
               [('baked_plus',torch.ones_like(a)),('radial',initial)]}
    for penalty in penalties:
        label=f'{penalty:g}'
        u=torch.nn.Parameter(initial.clone())
        v=torch.nn.Parameter(torch.full_like(a,float(np.log(.01/.99))))
        method=config.get('optimizer','adam')
        if method not in ('adam','sgd'): raise ValueError('optimizer must be adam or sgd')
        optimizer=(torch.optim.Adam if method=='adam' else torch.optim.SGD)([u,v],lr=lr)
        history=[];started=time.monotonic();torch.cuda.synchronize()
        for step in range(steps+1):
            total,pair,outward,broken=losses(u,v,a,i,j,dot,weak,penalty)
            if step%every==0 or step==steps:
                torch.cuda.synchronize()
                record=dict(step=step,seconds=time.monotonic()-started,loss=float(total.detach()),
                            pair_loss=float(pair.detach()),prior_loss=float(outward.detach()),
                            **metrics(u,v,a,i,j,dot))
                if not np.isfinite(record['loss']): raise RuntimeError('Nonfinite loss')
                history.append(record)
                print('sign solve penalty='+label+' '+json.dumps(record),flush=True)
                atomic_write(root/f'history_{label}.json',history,durable=False)
            if step==steps: break
            optimizer.zero_grad(set_to_none=True)
            total.backward();optimizer.step()
        with torch.no_grad():
            from .fiber_normal_convention import inward_solution
            saved=inward_solution(dict(ids=data['ids'],
                     sign_probability=u.sigmoid().cpu().numpy(),broken_probability=v.sigmoid().cpu().numpy(),
                     normal_sign=torch.where(u>=0,1,-1).cpu().numpy().astype(np.int8),
                     broken=(v>=0).cpu().numpy(),sign_logit=u.cpu().numpy(),broken_logit=v.cpu().numpy()))
            np.savez(root/f'solution_{label}.npz',**saved)
        if config.get('write_result_store',True):
            write_results(root/f'solution_{label}.vcobj',[config['database']], 'normal_sign', method,
                {'fibers':[dict(fiber_uuid=str(fid),normal_sign=int(sign),enabled=int(not broken),
                                sign_probability=float(p),p_broken=float(pb))
                    for fid,sign,broken,p,pb in zip(saved['ids'],saved['normal_sign'],saved['broken'],
                                                   saved['sign_probability'],saved['broken_probability'])]},
                settings=dict(broken_penalty=float(penalty),steps=steps,lr=float(lr)))
        summaries.append(dict(broken_penalty=penalty,initial=history[0],final=history[-1],
                              all_broken_objective=penalty+weak*(1-float(a.abs().mean()))/2))
    summary=dict(config=config,baselines=baselines,runs=summaries,
                 input=json.loads(Path(config['tensor_cache']).with_suffix('.json').read_text()),
                 torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),deterministic_algorithms=True,
                 hard_readout='Saved solutions use inward normals (negated internal outward gauge); history uses outward gauge. Broken for logit>=0.',
                 interpretation='event-weighted agreement, not anatomical accuracy; no winding solve')
    atomic_write(root/'summary.json',summary,durable=False)
    return summary


def run(config):
    """Approved benchmark runner entry. Manager owns CUDA child, never this loader."""
    prepare(config)
    return launch_tensor_solve(config,__name__,Path(__file__))






def tensor_matrix_jobs(config,visible):
    """Validate isolated lanes before starting any CUDA child."""
    lanes=config['experiment_matrix']
    if not config['gpu_manager'].get('exclusive'):
        raise ValueError('Experiment matrix requires exclusive parent lease')
    if len(lanes)>len(visible) or not lanes:raise ValueError('Insufficient matrix GPUs')
    jobs=[];outputs=set()
    for lane,items in enumerate(lanes):
        if not items:raise ValueError('Empty experiment lane')
        for item in items:
            child=json.loads(Path(item).read_text())['fiber_uv_layout']
            if 'experiment_matrix' in child:raise ValueError('Nested experiment matrix')
            root=Path(child['output_root']).resolve()
            if str(root) in outputs:raise ValueError('Duplicate matrix output')
            if (root/'summary.json').exists():raise FileExistsError('Completed matrix child exists')
            outputs.add(str(root));jobs.append((lane,visible[lane],child))
    return jobs


def solve_tensor_matrix(config,module):
    """One exclusive manager lease, independent solver processes per GPU lane."""
    from concurrent.futures import ThreadPoolExecutor,as_completed
    import threading
    visible=[v for v in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if v]
    jobs=tensor_matrix_jobs(config,visible)
    active=[];lock=threading.Lock();stop=threading.Event();results=[]
    def lane_run(lane):
        completed=[]
        for _,gpu,child in [j for j in jobs if j[0]==lane]:
            root=Path(child['output_root']);root.mkdir(parents=True,exist_ok=True)
            child=dict(child,code_provenance=config.get('code_provenance'),matrix_gpu=gpu,
                       matrix_lane=lane,matrix_parent=config['output_root'],
                       matrix_launcher_sha256=sha256_file(Path(__file__)))
            frozen=root/'config.json';atomic_write(frozen,child,durable=False)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
            with (root/'runner.log').open('w') as log:
                with lock:
                    if stop.is_set():return completed
                    process=subprocess.Popen([sys.executable,'-m',module,'--config',str(frozen),'--unmanaged','--devices','0'],
                        env=env,stdout=log,stderr=subprocess.STDOUT)
                    active.append(process)
                rc=process.wait()
                with lock:active.remove(process)
            record=dict(output_root=str(root),gpu=gpu,exit_code=rc)
            atomic_write(root/'matrix_job.json',record,durable=False)
            print('MATRIX '+json.dumps(record),flush=True)
            if rc:raise RuntimeError(f'Matrix child failed: {root}, exit {rc}')
            completed.append(record)
        return completed
    pool=ThreadPoolExecutor(max_workers=len(config['experiment_matrix']))
    try:
        futures=[pool.submit(lane_run,lane) for lane in range(len(config['experiment_matrix']))]
        for future in as_completed(futures):results.extend(future.result())
    finally:
        with lock:
            stop.set();running=list(active)
            for process in running:
                if process.poll() is None:process.terminate()
        for process in running:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        pool.shutdown(wait=True,cancel_futures=True)
    summary=dict(config=config,jobs=results)
    atomic_write(Path(config['output_root'])/'summary.json',summary,durable=False)
    return summary


def freeze_solver_config(config,source):
    """Record code provenance and freeze the stage configuration under output_root.

    Shared by the managed CUDA launcher and in-process CPU solves so both
    leave the same ``config.json``/``summary.json`` artifacts.
    """
    repo=Path(__file__).resolve().parents[5]
    config=dict(config,code_provenance=dict(
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        optimizer_sha256=sha256_file(source)))
    if config.get('optimizer')=='wrapped_irls':
        for name in ('fiber_wrapped_irls','fiber_wrapped_stages'):
            config['code_provenance'][name+'_sha256']=sha256_file(Path(__file__).with_name(name+'.py'))
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    frozen=root/'config.json'
    if (root/'summary.json').exists(): raise FileExistsError('Completed output exists')
    atomic_write(frozen,config,durable=False)
    return config,frozen


def launch_tensor_solve(config,module,source):
    """Shared managed CUDA launch for fixed-evidence optimization stages."""
    from .managed_evaluation import launch_managed
    config,frozen=freeze_solver_config(config,source)
    root=Path(config['output_root'])
    command=[sys.executable,'-m',module,'--config',str(frozen)]
    rc=launch_managed(command,devices=config.get('devices','0'),settings=config['gpu_manager'])
    if rc: raise RuntimeError(f'Managed sign solver exited {rc}')
    return json.loads((root/'summary.json').read_text())


def run_priors(config):
    """Fingerprint immutable native inputs before and after existing CPU runner."""
    from .constraint_extraction import run_native_constraints
    native=json.loads(Path(config['native_config']).read_text())
    source=Path(native['orient_database']);umbilicus=Path(native['umbilicus'])
    native['source_database_sha256']=store_digest(source)
    native['umbilicus_sha256']=sha256_file(umbilicus)
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    frozen=root/'native_config.json';atomic_write(frozen,native,durable=False)
    run_native_constraints(frozen, config)
    if store_digest(source)!=native['source_database_sha256'] or sha256_file(umbilicus)!=native['umbilicus_sha256']:
        artifact=Path(native['normal_prior_output'])
        artifact.rename(artifact.with_name(artifact.name+'.invalid'))
        raise RuntimeError('Inputs changed during prior export; output must not be used')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--unmanaged',action='store_true')
    p.add_argument('--devices')
    args=p.parse_args()
    if not args.unmanaged:
        config = json.loads(args.config.read_text())
        if args.devices:
            config['devices'] = args.devices
        run(config)
        raise SystemExit(0)
    solve(json.loads(args.config.read_text()))
