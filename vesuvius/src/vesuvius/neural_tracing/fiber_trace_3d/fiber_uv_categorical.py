"""Assignment-only, full-neighborhood cost-target experiment for global U/V.

Continuous geometry is frozen deliberately. Uses the same geometric losses,
input provenance, manager and hard readout as fiber_uv_layout, not new evidence.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import fiber_uv_layout as uv


def accumulate_pair_costs(out, i, j, cost, probabilities, scale):
    """Both conditional expectations; broken neighbor contributes zero."""
    k=cost.shape[1]
    a=(cost*probabilities[j,None,:k]).sum(2)*scale
    b=(cost*probabilities[i,:k,None]).sum(1)*scale
    out[:,:k].index_add_(0,i,a.to(out.dtype))
    out[:,:k].index_add_(0,j,b.to(out.dtype))
    return (a*probabilities[i,:k]).sum()/len(out)


def categorical_target(cost, temperature):
    if temperature<=0:raise ValueError('Target temperature must be positive')
    return (-cost.detach()/temperature).softmax(1)


def ce_logit_step(logits,target,step):
    """Exact per-node cross-entropy SGD gradient, without a 1/N step shrink."""
    if not 0<step<=4:raise ValueError('Categorical step must be in (0,4]')
    return logits-step*(logits.softmax(1)-target.detach())


@torch.no_grad()
def conditional_costs(state,widths,data,table,config,with_energy=False):
    n=len(state);k=len(widths);e=len(data['i']);batch=config.get('batch_size',1024)
    probs=state[:,:k+1].softmax(1)
    costs=torch.zeros_like(probs)
    degree=data['degree'];mean_degree=degree.mean();degree_cost=.5+.5*degree/mean_degree.clamp_min(1.)
    costs[:,k]=config.get('rejection_cost',10.)*degree_cost
    energy=(costs[:,k]*probs[:,k]).mean()
    bounds=torch.cat((widths.new_zeros(1),widths.cumsum(0)))
    scale=n/e
    def coords(ids,ds):
        return uv.coordinates(state[ids].double(),data['phase'][ids],data['z'][ids],
                              data['family'][ids],ds,widths,bounds)
    for start in range(0,e,batch):
        sl=slice(start,min(start+batch,e));i=data['i'][sl];j=data['j'][sl]
        _,va,qa=coords(i,data['ds_a'][sl]);_,vb,qb=coords(j,data['ds_b'][sl])
        cost,ordered,_,_=uv.pair_cost_components(qa,qb,va,vb,data['phase_delta'][sl],
            data['family'][j]-data['family'][i],data['sign'][sl],data['distance'][sl],
            data['target_v'][sl],table,config)
        charge=uv.calibrated_sign_charge(degree[i],degree[j],mean_degree,e,n,
            config.get('rejection_cost',10.),config.get('sign_charge_multiplier',4.))
        cost=cost+charge[:,None,None]*(ordered<0)
        energy+=accumulate_pair_costs(costs,i,j,cost,probs,scale)
        for ids,q,key in ((i,qa,'phase_a'),(j,qb,'phase_b')):
            angle=q-data[key][sl,None];angle-=angle.round()
            unary=(scale*.5*config.get('prior_weight',.01)*angle.square()).float()
            costs[:,:k].index_add_(0,ids,unary)
            energy+=(unary*probs[ids,:k]).sum()/n
    links=data['links']
    for start in range(0,len(links),batch):
        link=links[start:start+batch];i=link[:,0].long();j=link[:,1].long()
        ua,va,_=coords(i,link[:,2]);ub,vb,_=coords(j,link[:,3])
        cost=(((ua[:,:,None]-ub[:,None,:])/config.get('continuation_scale',128.)).square()+
              ((va[:,:,None]-vb[:,None,:])/config.get('continuation_scale',128.)).square())
        energy+=accumulate_pair_costs(costs,i,j,cost,probs,scale*config.get('continuation_weight',10.))
    for start in range(0,n,batch):
        ids=torch.arange(start,min(start+batch,n),device=state.device)
        for point in range(3):
            _,v,q=coords(ids,data['node_ds'][ids,point])
            angle=q-data['node_phase'][ids,point,None];angle-=angle.round()
            prior=angle.square()+((v-data['node_z'][ids,point,None])/1000).square()
            prior+=torch.relu(-q).square()+torch.relu(q-100).square()
            unary=(config.get('prior_weight',.01)*prior/3).float()
            costs[ids,:k]+=unary
            energy+=(unary*probs[ids,:k]).sum()/n
    return (costs,energy) if with_energy else costs


def solve_categorical(config):
    config=dict(config,executed_modules={str(Path(p).name):uv.sha256_file(p)
                for p in (__file__,uv.__file__)})
    if not torch.cuda.is_available():raise RuntimeError('Production categorical solve requires CUDA')
    if not uv.anchored(config) or not config.get('warm_start'):raise ValueError('Requires shared-anchor warm start')
    torch.set_num_threads(4);torch.manual_seed(config.get('seed',0));torch.use_deterministic_algorithms(True)
    with np.load(config['tensor_cache']) as src:d={key:src[key] for key in src.files}
    state,widths=uv.load_warm_start(config,d,(len(d['family']),303))
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():raise FileExistsError('Completed categorical output exists')
    tensor=torch.tensor(state,device='cuda');w=torch.tensor(widths,device='cuda',dtype=torch.float64)
    keys=('family','phase','z','degree','node_ds','node_phase','node_z','i','j','ds_a','ds_b',
          'phase_a','phase_b','phase_delta','target_v','distance','sign','links')
    data={key:torch.as_tensor(d[key],device='cuda',dtype=torch.long if key in ('family','i','j') else torch.float64) for key in keys}
    table=torch.tensor(uv.make_distance_table(float(d['distance'].max()),config.get('distance_sigma',.5)),device='cuda')
    baseline=uv.audit(d,state,widths,config);history=[];started=time.monotonic()
    torch.cuda.reset_peak_memory_stats();sweeps=config.get('categorical_sweeps',30)
    for sweep in range(sweeps+1):
        snapshot=tensor.cpu().numpy()
        metrics=uv.audit(d,snapshot,widths,config)
        record=dict(step=sweep,seconds=time.monotonic()-started,equivalent_event_epochs=sweep,
                    geometry_frozen=True,**metrics)
        if sweep:record.update(target_broken_mean=float(target[:,-1].mean()),
            target_max_residual=float((tensor[:,:101].softmax(1)-target).abs().max()),
            aggregation_seconds=aggregation_seconds,pre_update_energy=float(energy))
        history.append(record);uv.atomic_write(root/'history.json',history,durable=False)
        print('UV categorical '+json.dumps(record),flush=True)
        if sweep==sweeps:break
        tick=time.monotonic();cost,energy=conditional_costs(tensor,w,data,table,config,with_energy=True)
        target=categorical_target(cost,config.get('target_temperature',.25))
        for _ in range(config.get('categorical_inner_steps',8)):
            tensor[:,:101]=ce_logit_step(tensor[:,:101],target,config.get('categorical_step',4.))
        torch.cuda.synchronize();aggregation_seconds=time.monotonic()-tick
    np.savez(root/'solution.npz',state=snapshot,widths=widths,
        **{key:d[key] for key in ('source_id','start','end','family')})
    if not np.array_equal(snapshot[:,101:],state[:,101:]):raise RuntimeError('Frozen geometry changed')
    summary=dict(config=config,input=json.loads(Path(config['tensor_cache']).with_suffix('.json').read_text()),
        baseline=baseline,final=history[-1],gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        interpretation='Assignment-only full-neighborhood categorical targets; frozen geometry; expected pair energy, not mixture likelihood; no posthoc rejection')
    uv.atomic_write(root/'summary.json',summary,durable=False)
