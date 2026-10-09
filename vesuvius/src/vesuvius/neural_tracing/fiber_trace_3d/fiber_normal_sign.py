"""Whole-fiber sign/broken optimization; CPU preparation, manager-owned CUDA solve."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import subprocess

import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file


def losses(sign_logits, broken_logits, prior, i, j, dot, prior_weight, broken_penalty):
    """Fixed denominator; disabling everything never removes the broken penalty."""
    import torch
    s = 2 * sign_logits.sigmoid() - 1
    b = broken_logits.sigmoid()
    pair = ((1-b[i])*(1-b[j])*(dot.abs()-dot*s[i]*s[j])*.5).sum() / dot.abs().sum().clamp_min(1e-12)
    outward = ((1-prior*s)*.5).mean()
    broken = b.mean()
    return pair + prior_weight*outward + broken_penalty*broken, pair, outward, broken


def metrics(u, v, a, i, j, dot):
    import torch
    with torch.no_grad():
        s = torch.where(u >= 0, 1., -1.)
        kept = (v[i] < 0) & (v[j] < 0)
        wrong = dot*s[i]*s[j] < 0
        weight = dot.abs()
        denominator = weight.sum().clamp_min(1e-12)
        kept_weight = weight[kept].sum()
        retained_error = 100*weight[kept & wrong].sum()/kept_weight if kept_weight > 0 else None
        return dict(broken_fibers_percent=float(100*(v>=0).float().mean()),
                    expected_broken_percent=float(100*v.sigmoid().mean()),
                    retained_events_percent=float(100*kept.float().mean()) if kept.numel() else None,
                    weighted_disagreement_all_percent=float(100*weight[wrong].sum()/denominator),
                    weighted_disagreement_retained_percent=float(retained_error) if retained_error is not None else None,
                    retained_disagreement_original_weight_percent=float(100*weight[kept & wrong].sum()/denominator),
                    outward_agreement=float((a*s).mean()),
                    uncertain_signs_percent=float(100*((u.sigmoid()>.1)&(u.sigmoid()<.9)).float().mean()))


def prepare(config):
    """Convert fixed evidence once; no geometry or model evaluation in optimization."""
    started = time.monotonic()
    db_path, prior_path = Path(config['database']), Path(config['priors'])
    signature = dict(version=2, database_sha256=sha256_file(db_path), priors_sha256=sha256_file(prior_path))
    cache = Path(config['tensor_cache'])
    stamp = cache.with_suffix('.json')
    if cache.exists() and stamp.exists() and json.loads(stamp.read_text())['signature'] == signature:
        return cache
    prior = json.loads(prior_path.read_text())
    if prior['format'] != 'fiber-outward-priors' or Path(prior['source_database']).resolve()!=db_path.resolve():
        raise ValueError('Prior database identity differs')
    if prior.get('source_database_sha256')!=signature['database_sha256']:
        raise ValueError('Priors were generated from different database content (or lack a fingerprint)')
    db = sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro', uri=True)
    meta = {k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
    if meta['schema_version']!=2 or not meta['complete']:
        raise ValueError('Signed schema2 database required')
    fibers = np.asarray(db.execute('SELECT id,normal_sign FROM fibers ORDER BY source_index').fetchall(), dtype=np.int64)
    ids = fibers[:,0]
    if not np.all(fibers[:,1]==1) or not np.all(np.diff(ids)>0):
        raise ValueError('Expected original +1 gauge multipliers and ascending source IDs')
    if not np.array_equal(ids,np.asarray(prior['source_ids'],dtype=np.int64)):
        raise ValueError('Prior/source fiber IDs differ')
    a = np.asarray(prior['agreement'],dtype=np.float32)
    if len(a)!=len(ids) or not np.isfinite(a).all() or np.any(abs(a)>1.000001):
        raise ValueError('Invalid radial priors')
    if prior['statistics']['sample_spacing_base']!=meta['normal_orientation']['sample_spacing_base']:
        raise ValueError('Prior orientation sampling differs')
    terms = [f"json_extract(normal_a,'$.xyz[{k}]')*json_extract(normal_b,'$.xyz[{k}]')" for k in range(3)]
    eligible = "perpendicular_score>parallel_score"
    valid = "json_extract(normal_a,'$.valid') AND json_extract(normal_b,'$.valid')"
    total = db.execute(f'SELECT count(*) FROM constraints WHERE {eligible}').fetchone()[0]
    cursor = db.execute(f'SELECT id,fiber_a,fiber_b,{"+".join(terms)} FROM constraints WHERE {eligible} AND {valid} ORDER BY id')
    chunks=[]
    while rows:=cursor.fetchmany(65536):
        chunks.append(np.asarray(rows,dtype=np.float64))
    edges=np.concatenate(chunks) if chunks else np.empty((0,4),dtype=np.float64)
    event_ids=edges[:,0].astype(np.int64)
    endpoints=edges[:,1:3].astype(np.int64)
    indices=np.searchsorted(ids,endpoints)
    if np.any(indices>=len(ids)) or not np.array_equal(ids[indices],endpoints):
        raise ValueError('Unknown constraint fiber')
    dot=edges[:,3].astype(np.float32)
    if not np.isfinite(dot).all() or np.any(abs(dot)>1.000001):
        raise ValueError('Invalid normal dot products')
    dot=np.clip(dot,-1,1)
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache.with_suffix('.partial.npz')
    np.savez(temporary,ids=ids,prior=a,i=indices[:,0],j=indices[:,1],dot=dot,event_ids=event_ids)
    temporary.replace(cache)
    degree=np.bincount(indices.ravel(),minlength=len(ids))
    stats=dict(signature=signature,fibers=len(ids),perpendicular_events=total,
               used_events=len(dot),invalid_events_skipped=total-len(dot),
               isolated_fibers=int((degree==0).sum()),tensor_bytes=cache.stat().st_size,
               seconds=time.monotonic()-started)
    atomic_write(stamp,stats,durable=False)
    print('tensor preparation '+json.dumps(stats),flush=True)
    return cache


def solve(config):
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
            np.savez(root/f'solution_{label}.npz',**inward_solution(dict(ids=data['ids'],
                     sign_probability=u.sigmoid().cpu().numpy(),broken_probability=v.sigmoid().cpu().numpy(),
                     normal_sign=torch.where(u>=0,1,-1).cpu().numpy().astype(np.int8),
                     broken=(v>=0).cpu().numpy(),sign_logit=u.cpu().numpy(),broken_logit=v.cpu().numpy())))
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


def launch_tensor_solve(config,module,source):
    """Shared managed CUDA launch for fixed-evidence optimization stages."""
    from .managed_evaluation import launch_managed
    repo=Path(__file__).resolve().parents[5]
    config=dict(config,code_provenance=dict(
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        optimizer_sha256=sha256_file(source)))
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    frozen=root/'config.json'
    if (root/'summary.json').exists(): raise FileExistsError('Completed output exists')
    atomic_write(frozen,config,durable=False)
    command=[sys.executable,'-m',module,'--config',str(frozen)]
    rc=launch_managed(command,devices=config.get('devices','0'),settings=config['gpu_manager'])
    if rc: raise RuntimeError(f'Managed sign solver exited {rc}')
    return json.loads((root/'summary.json').read_text())


def run_priors(config):
    """Fingerprint immutable native inputs before and after existing CPU runner."""
    from .crop_benchmark import run as run_crop
    native=json.loads(Path(config['native_config']).read_text())
    source=Path(native['orient_database']);umbilicus=Path(native['umbilicus'])
    native['source_database_sha256']=sha256_file(source)
    native['umbilicus_sha256']=sha256_file(umbilicus)
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    frozen=root/'native_config.json';atomic_write(frozen,native,durable=False)
    run_crop(dict(output_root=str(root),reference_dir=str(umbilicus.parent),
                  input_files=[str(frozen),str(umbilicus)],commands=[
                      dict(name='tests',binary='test_fiber_trace_winding_bp',args=[]),
                      dict(name='priors',binary='vc_fiber_trace_chunk',args=['afv-constraints',str(frozen)])]))
    if sha256_file(source)!=native['source_database_sha256'] or sha256_file(umbilicus)!=native['umbilicus_sha256']:
        artifact=Path(native['normal_prior_output'])
        artifact.rename(artifact.with_name(artifact.name+'.invalid'))
        raise RuntimeError('Inputs changed during prior export; output must not be used')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--unmanaged',action='store_true')
    p.add_argument('--devices')
    args=p.parse_args()
    if not args.unmanaged: raise RuntimeError('Use approved evaluation runner for managed GPU execution')
    solve(json.loads(args.config.read_text()))
