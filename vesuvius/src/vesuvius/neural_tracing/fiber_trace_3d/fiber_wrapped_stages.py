"""Optional wrapped-IRLS backend for the existing sign and H/V stage caches."""
import json
from pathlib import Path

import numpy as np
import torch

from lasagna.inference_provenance import atomic_write
from .fiber_graph_store import write_results
from .fiber_wrapped_irls import class_score, solve


def _logit(p):
    return torch.logit(p.clamp(1e-7,1-1e-7))


def solve_stage(config, stage):
    from . import fiber_hv_stage as hv
    from . import fiber_normal_sign as normal
    from .fiber_normal_convention import inward_solution
    if stage not in ('normal_sign','classification'):
        raise ValueError('unknown wrapped graph stage')
    if not torch.cuda.is_available():
        raise RuntimeError('Production wrapped-IRLS stage requires CUDA')
    if stage=='classification' and any(float(v)!=0 for v in config.get('entropy_weights',[0.,.01])):
        raise ValueError('wrapped_irls currently requires explicit entropy_weights:[0]; Adam is unchanged')
    torch.set_num_threads(4)
    torch.manual_seed(config.get('seed',0))
    torch.use_deterministic_algorithms(True)
    with np.load(config['tensor_cache']) as cache:
        d={k:cache[k] for k in cache.files}
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():
        raise FileExistsError('Completed stage output exists')
    stats=json.loads(Path(config['tensor_cache']).with_suffix('.json').read_text())
    tensor=lambda x:torch.as_tensor(x,device='cuda')
    a=tensor(d['prior' if stage=='normal_sign' else 'a'])
    i,j=tensor(d['i']),tensor(d['j'])
    n=a.numel()
    initial=torch.where(a==0,torch.full_like(a,.001),2*a).sigmoid()
    phase=1-initial if stage=='normal_sign' else initial
    weak=float(config.get('prior_weight',.001))
    unary_weight=weak*a.abs()/n
    unary_constant=weak*(1-a.abs())/(2*n)
    prior_target=((a<0) if stage=='normal_sign' else (a>=0)).to(a.dtype)
    if stage=='normal_sign':
        evidence=tensor(d['dot'])
        target=(evidence<0).to(a.dtype)
        weight=evidence.abs()/evidence.abs().sum().clamp_min(1e-12)
        modes=['uniform']
    else:
        evidence=tensor(d['relation'])
        target=(evidence<0).to(a.dtype)
        weight=torch.ones_like(evidence)/max(1,len(evidence))
        modes=config.get('cost_modes',['uniform','degree_mix'])
    options=dict(epsilon=float(config.get('irls_epsilon',.05)),
                 iterations=int(config.get('irls_iterations',100)),
                 max_cg=int(config.get('irls_cg_iterations',100)),
                 cg_tolerance=float(config.get('irls_cg_tolerance',1e-5)),
                 tolerance=float(config.get('irls_tolerance',1e-7)))
    runs=[]
    for mode in modes:
        if mode not in ('uniform','degree_mix'):
            raise ValueError('unknown broken cost mode')
        degree=tensor(d['degree']) if mode=='degree_mix' else torch.ones_like(a)
        cost_weight=.5+.5*degree/degree.mean().clamp_min(1e-12) if mode=='degree_mix' else degree
        for penalty in config.get('broken_penalties',[.1]):
            if float(penalty)<=0:
                raise ValueError('broken penalty must be positive')
            cost=cost_weight*float(penalty)
            name=f'{penalty:g}' if stage=='normal_sign' else f'{mode}_b{penalty:g}_e0'

            def measure(step,x,k):
                p=class_score(x)
                b=_logit(1-k)
                if stage=='normal_sign':
                    h=_logit(1-p)
                    m=normal.metrics(h,b,a,i,j,evidence)
                    old=normal.losses(h,b,a,i,j,evidence,weak,float(penalty))[0]
                    m['retained_arc_percent']=float(100*(tensor(d['lengths'])*(k>.5)).sum()/tensor(d['lengths']).sum()) if 'lengths' in d else None
                else:
                    h=_logit(p)
                    probs=torch.stack((k*(1-p),k*p,1-k),-1)
                    m=hv.readout_metrics(probs.cpu().numpy(),d,stats)
                    old=hv.losses(h,b,a,i,j,evidence,cost,weak,0.)[0]
                m['original_expected_objective']=float(old)
                return m

            def log(row):
                if row['step']%int(config.get('irls_log_every',10))==0:
                    print(stage+' wrapped_irls '+name+' '+json.dumps(row),flush=True)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            fixed_keep=bool(config.get('fixed_keep',False))
            if fixed_keep and stage!='normal_sign':
                raise ValueError('fixed_keep is only supported for downstream normal-sign alignment')
            initial_keep=tensor(d['keep']) if fixed_keep else torch.full_like(a,.99)
            result=solve(phase,initial_keep,i,j,target,weight,
                         prior_target,unary_weight,unary_constant,cost/n,
                         mask_prior=stage=='classification',fixed_keep=fixed_keep,callback=measure,log=log,**options)
            x,k=result['phase'],result['keep'];p=class_score(x)
            history=result['history']
            atomic_write(root/f'history_{name}.json',history,durable=False)
            print(stage+' wrapped_irls FINAL '+name+' '+json.dumps(history[-1]),flush=True)
            if stage=='normal_sign':
                h=_logit(1-p);b=_logit(1-k)
                saved=inward_solution(dict(ids=d['ids'],sign_probability=(1-p).cpu().numpy(),
                    broken_probability=(1-k).cpu().numpy(),normal_sign=torch.where(h>=0,1,-1).cpu().numpy().astype(np.int8),
                    broken=(k<=.5).cpu().numpy(),sign_logit=h.cpu().numpy(),broken_logit=b.cpu().numpy()))
                saved['wrapped_phase']=torch.remainder(x+1,2).cpu().numpy()  # Saved normal gauge is inward.
                if config.get('write_result_store',True):
                    rows=[dict(fiber_uuid=str(fid),normal_sign=int(s),enabled=int(not b),sign_probability=float(p),p_broken=float(pb))
                          for fid,s,b,p,pb in zip(saved['ids'],saved['normal_sign'],saved['broken'],saved['sign_probability'],saved['broken_probability'])]
                    if fixed_keep:
                        for row in rows:
                            del row['enabled'];del row['p_broken']
                runs.append(dict(broken_penalty=float(penalty),initial=history[0],final=history[-1]))
            else:
                probs=torch.stack((k*(1-p),k*p,1-k),-1).cpu().numpy()
                full=np.zeros((len(d['ids']),3),np.float32);full[:,2]=1;full[d['active']]=probs
                labels=full.argmax(1).astype(np.uint8)
                saved=dict(ids=d['ids'],probabilities=full,labels=labels,inherited_broken=d['inherited'],
                    hv_logit=_logit(p).cpu().numpy(),broken_logit=_logit(1-k).cpu().numpy(),active=d['active'],wrapped_phase=x.cpu().numpy())
                if config.get('write_result_store',True):
                    rows=[dict(fiber_uuid=str(fid),family=int(label),enabled=int(label!=2),p_h=float(p[0]),p_v=float(p[1]),p_broken=float(p[2]))
                          for fid,label,p in zip(d['ids'],labels,full)]
                runs.append(dict(name=name,cost_mode=mode,penalty=float(penalty),entropy_max=0.,initial=history[0],final=history[-1]))
            runs[-1].update(iterations=result['iterations'],cg_iterations=result['total_cg_iterations'],converged=result['converged'],
                            peak_allocated_bytes=torch.cuda.max_memory_allocated())
            np.savez(root/f'solution_{name}.npz',**saved)
            if config.get('write_result_store',True):
                write_results(root/f'solution_{name}.vcobj',[config['database']],stage,'wrapped_irls',{'fibers':rows},
                              settings=dict(options,broken_penalty=float(penalty),prior_weight=weak,cost_mode=mode))
    summary=dict(config=config,input=stats,runs=runs,gpu=torch.cuda.get_device_name(),
        backend='wrapped_irls',peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in runs),
        interpretation='Periodic Huber-L1 relaxation; class scores are not calibrated probabilities; original_expected_objective is the unchanged SGD objective at this state.')
    if stage=='classification':
        summary['baseline']=hv.readout_metrics(torch.stack((1-initial,initial,torch.zeros_like(initial)),-1).cpu().numpy(),d,stats)
        summary['labels']=['H','V','broken']
    atomic_write(root/'summary.json',summary,durable=False)
    return summary
