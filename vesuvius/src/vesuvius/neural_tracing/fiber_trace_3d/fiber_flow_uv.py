"""Conditional fixed-flow-label geometry fit using the SGD geometry/optimizer.

No probability, rejection, reclassification, or integer-label optimization.
The selected integer separation at each event is immutable. Geometry uses the
existing straight fixed-length SharedCurve and learned monotone winding widths.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .fiber_min_flow_experiment import arrays, write
from .fiber_geometry_cache import width_boundaries


def saved_assignment_inputs(data, solution, component_selection='largest'):
    """Adapt the saved graph solver's largest component, preserving its labels.

    Input endpoint arcs may use aligned signs; the flow UV staging format uses
    unsigned arc length from the original start and aligns them during prepare.
    """
    from .fiber_min_flow_benchmark import subset_minimal
    np.testing.assert_array_equal(data['source_id'],solution['source_id'])
    if len(data['links']) or np.any(data['start']!=0):
        raise ValueError('Fixed-label UV expects whole unsplit fibers')
    comp=solution['component']
    if component_selection not in ('all','largest'):
        raise ValueError('component_selection must be all or largest')
    keep=np.ones(len(comp),bool) if component_selection=='all' else comp==np.bincount(comp).argmax()
    d=subset_minimal(data,keep)
    direction=np.sign(d['node_ds'][:,-1])
    if np.any(direction==0):raise ValueError('Missing endpoint arc direction')
    for side,key in [('a','i'),('b','j')]:d['ds_'+side]*=direction[d[key]]
    d['node_ds']*=direction[:,None]
    labels=solution['winding'][keep]
    if not np.isfinite(labels).all() or np.any(labels!=np.rint(labels)):
        raise ValueError('Use saved integer assignments, not continuous potentials')
    return d,labels


def run_saved(config):
    """Fixed-label UV fitting from an existing IRLS/LEMON checkpoint; no re-solve."""
    from .fiber_min_flow_experiment import digest
    from .fiber_graph_store import workflow_volume
    root=Path(config['output_root']);stage=root/'fixed_input';stage.mkdir(parents=True,exist_ok=True)
    binding=dict(request=config,sha256={str(p):digest(p) for p in
        (config['tensor_cache'],config['solution'])})
    manifest=stage/'binding.json'
    if manifest.exists() and json.loads(manifest.read_text())!=binding:
        raise ValueError('Saved assignment inputs changed; choose a new output root')
    d,labels=saved_assignment_inputs(arrays(config['tensor_cache']),arrays(config['solution']),
                                   config.get('component_selection','largest'))
    np.savez(stage/'selected_tensors.npz',**d)
    np.savez(stage/'selected_solution.npz',winding=labels)
    fibers=stage/'selected_with_gt.vcobj'
    if not fibers.exists():fibers.symlink_to(Path(config['fibers']).resolve())
    write(manifest,binding)
    fit=dict(config['parameters'],flow_root=str(stage),output_root=str(root),
        working_volume=workflow_volume(config),
        umbilicus=config['umbilicus'],devices=config.get('devices','0'),gpu_manager=config['gpu_manager'])
    return run(fit)


def supported(d):
    """Induce the graph on perpendicular-supported fibers (no fallback geometry)."""
    degree=np.bincount(np.r_[d['i'][~d['parallel']],d['j'][~d['parallel']]],minlength=len(d['source_id']))
    keep=degree>0
    edge=keep[d['i']]&keep[d['j']]
    from .fiber_min_flow_benchmark import subset_minimal
    result=subset_minimal(d,keep)
    return result,keep,edge


def prepare(config):
    from .fiber_min_flow_benchmark import Paths
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    parent=Path(config['flow_root']);original=arrays(parent/'selected_tensors.npz')
    sol=arrays(parent/'selected_solution.npz');d,keep,edge=supported(original)
    if not keep.any():raise ValueError('No perpendicular-supported fibers')
    if config.get('packed_geometry'):
        from .fiber_min_flow_full import PackedPaths
        from .fiber_packed_priors import geometry_priors
        paths=PackedPaths(config['packed_geometry'],d)
        prior=geometry_priors(paths,d['source_id'],d['family'],config.get('prior_spacing',1000))
        directions,phase,z,radii=(prior[k] for k in ('directions','phase','z','radii'))
        sample_owner,sample_arc,sample_phase,sample_z=(prior[k] for k in
            ('sample_owner','sample_arc','sample_phase','sample_z'))
        minimum_turn,maximum_turn=prior['minimum_turn'],prior['maximum_turn']
    else:
        paths=Paths(dict(fibers=str(parent/'selected_with_gt.vcobj'),umbilicus=config['umbilicus'],
                         working_volume=config.get('working_volume')),d)
    if not config.get('packed_geometry'):
        directions=[];phase=[];z=[];radii=[];sample_owner=[];sample_arc=[];sample_phase=[];sample_z=[]
    for i,sid in enumerate(d['source_id'] if not config.get('packed_geometry') else []):
        arc,xyz,transport=paths.paths[str(sid)];rel=paths.seam.relative(xyz)
        p=((np.arctan2(rel[:,1],rel[:,0])-paths.seam.angle)%(2*np.pi))/(2*np.pi)
        direction=np.sign((transport[-1]+p[-1]-p[0]) if d['family'][i]==0 else xyz[-1,2]-xyz[0,2])
        if not direction:raise ValueError('Unidentifiable fiber direction: '+str(sid))
        directions.append(direction);phase.append(p[0]);z.append(xyz[0,2]);radii.append(np.median(np.linalg.norm(rel,axis=1)))
        at=np.linspace(0,arc[-1],max(2,int(np.ceil(arc[-1]/config.get('prior_spacing',1000)))+1))
        pos,_=paths.at_many(np.full(len(at),sid),at);rel=paths.seam.relative(pos)
        sample_owner.extend([i]*len(at));sample_arc.extend(at)
        sample_phase.extend(((np.arctan2(rel[:,1],rel[:,0])-paths.seam.angle)%(2*np.pi))/(2*np.pi));sample_z.extend(pos[:,2])
    if not config.get('packed_geometry'):
        minimum_turn=np.array([paths.paths[str(s)][2].min() for s in d['source_id']])
        maximum_turn=np.array([paths.paths[str(s)][2].max() for s in d['source_id']])
    # All targets are read from the flow solution, never from rounded SGD values.
    labels=sol['winding'][keep].copy()
    shift=2-int(np.min(labels+minimum_turn))
    labels+=shift
    if config.get('packed_geometry'):
        from .fiber_packed_priors import fixed_targets
        target=fixed_targets(paths,d,labels,int(config.get('preparation_batch_size',65536)))
    else:
        a,ta=paths.ends('a');b,tb=paths.ends('b');connector=paths.seam.segments(a,b)
        target=labels[d['j']]-labels[d['i']]+tb-ta-connector
    if not ((target==0)&~d['parallel']).any():raise ValueError('No fixed same-sheet perpendicular meetings')
    maximum=np.max(labels+maximum_turn+1)
    bins=np.arange(int(maximum)+3)
    # Radius supplies INITIALIZATION ONLY; there is no radius/width prior loss.
    known=np.unique(labels);width=np.array([np.median(np.asarray(radii)[labels==k])*2*np.pi for k in known])
    widths=np.maximum.accumulate(np.interp(bins,known,width))
    directions=np.asarray(directions)
    d['node_ds']*=directions[:,None]
    d['ds_a']*=directions[d['i']];d['ds_b']*=directions[d['j']]
    d.update(fixed_winding=labels,fixed_target=target,phase=np.array(phase),z=np.array(z),
        width_initial=widths,direction=directions,sample_owner=np.array(sample_owner,np.int64),
        sample_ds=np.asarray(sample_arc)*directions[np.asarray(sample_owner)],
        sample_phase=np.array(sample_phase),sample_z=np.array(sample_z))
    np.savez(root/'tensors.npz',**d)
    write(root/'selection.json',dict(input_fibers=len(keep),retained=int(keep.sum()),
        removed_source_ids=original['source_id'][~keep].tolist(),events=len(target),global_integer_shift=shift,
        same_sheet_perpendicular=int(((target==0)&~d['parallel']).sum())))
    print(json.dumps(dict(stage='prepared',fibers=int(keep.sum()),events=len(target))),flush=True)


def solve(config):
    import torch
    from .fiber_shared_curve import SharedCurve
    from .fiber_scaled_lbfgs import uncertainty_joint
    from . import fiber_soft_winding as soft
    torch.set_num_threads(4)
    device='cuda';dtype=torch.float64
    root=Path(config['output_root']);d=arrays(root/'tensors.npz');started=time.monotonic()
    def tensor(x,kind=dtype):return torch.as_tensor(x,device=device,dtype=kind)
    n=len(d['source_id'])
    # One node per fiber is sufficient for the EXACT straight model; it is not
    # a fallback for unsupported fibers, which were explicitly removed above.
    owner=np.arange(n);arc=np.zeros(n)
    curve=SharedCurve(d,owner,arc,dict(config,straight_fixed_length_geometry=True,shared_curve_integer_geometry=True),device)
    extension=torch.nn.Module();extension.add_module('curve',curve)
    state=torch.nn.Parameter(tensor(np.column_stack((d['fixed_winding'],d['phase'],d['z'],np.full(n,100.)))))
    initial_width=tensor(d['width_initial']);raw=torch.nn.Parameter(soft.encode_widths(initial_width,config))
    curve.initialize(state,initial_width)
    if config.get('initial_solution'):
        saved=arrays(config['initial_solution'])
        for key in ('source_id','start','end','family','direction','fixed_winding','fixed_target'):
            if not np.array_equal(saved[key],d[key]):raise ValueError('Continuation identity differs: '+key)
        with torch.no_grad():
            state.copy_(tensor(saved['state']))
            raw.copy_(soft.encode_widths(tensor(saved['widths']),config))
            curve.u.copy_(tensor(saved['shared_u']));curve.v.copy_(tensor(saved['shared_v']))
    from .fiber_flow_uv_objective import FixedFlowObjective
    objective=FixedFlowObjective(curve,state,raw,d,initial_width,config)
    def stats(x):
        x=abs(x)
        return dict(mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(x.max()))
    history=[]
    def save(name):
        np.savez(root/name,state=state.detach().cpu().numpy(),widths=soft.decode_widths(raw,config).detach().cpu().numpy(),
            shared_u=curve.u.detach().cpu().numpy(),shared_v=curve.v.detach().cpu().numpy(),
            **{k:d[k] for k in ('source_id','start','end','family','direction','fixed_winding','fixed_target')})
    def record(step):
        with torch.no_grad():
            loss,terms=objective()
            same_u,same_v,same_q,all_q=objective.errors()
            row=dict(iteration=step,seconds=time.monotonic()-started,loss=float(loss),terms={k:float(v) for k,v in terms.items()},
                same_sheet_u_vx=stats(same_u),same_sheet_v_vx=stats(same_v),
                same_sheet_winding=stats(same_q),all_winding_residual=stats(all_q),
                winding_assignments_changed=int((state[:,0]!=tensor(d['fixed_winding'])).sum()))
        history.append(row);write(root/'history.json',history);print(json.dumps(row),flush=True)
    record(0);save('initial.npz')
    result=uncertainty_joint(objective,state,raw,extension,config.get('steps',1000),{},config,
        lambda step:record(step) if step%config.get('log_every',100)==0 else None,
        freeze_winding=True,optimize_rejection=False,objective_parts=objective.parts)
    if history[-1]['iteration']!=result['iterations']:record(result['iterations'])
    save('solution.npz')
    assert np.array_equal(state[:,0].detach().cpu().numpy(),d['fixed_winding'])
    from .fiber_graph_store import write_results
    # Export the evaluated curve, not raw trainable offsets. Both coordinates
    # use the same fixed hypothesis and width map as the optimized objective.
    with torch.no_grad():
        widths=soft.decode_widths(raw,config)
        q,x,v=objective.samples()
        positions=[];node_index=np.zeros(n,np.int64)
        for k,owner in enumerate(d['sample_owner']):
            positions.append(dict(fiber_uuid=str(d['source_id'][owner]),layer=3,
                hypothesis=int(d['fixed_winding'][owner]),node_index=int(node_index[owner]),
                arc=float(d['sample_ds'][k]*d['direction'][owner]),
                u_turns=float(q[k]-d['fixed_winding'][owner]),u_vx=float(x[k]),v_vx=float(v[k])))
            node_index[owner]+=1
        run_uuid=write_results(config.get('result_store',root/'solution.vcobj'),[Path(config['flow_root'])/'selected_with_gt.vcobj'],
            'fixed_winding_uv',config.get('optimizer','torch_lbfgs'),
            dict(positions=positions,
                 widths=[dict(winding=i,width_vx=float(w)) for i,w in enumerate(widths.cpu().numpy())],
                 fibers=[dict(fiber_uuid=str(fid),enabled=1,family=int(d['family'][i]),
                              selected_winding=int(d['fixed_winding'][i])) for i,fid in enumerate(d['source_id'])]),
            input_runs=[config.get('input_run_uuid')],
            settings=dict(global_integer_shift=json.loads((root/'selection.json').read_text())['global_integer_shift'],
                          batch_size=objective.batch,assignments_fixed=True))
    summary=dict(run_uuid=run_uuid,optimizer=result,initial=history[0],final=history[-1],seconds=time.monotonic()-started,
        retained=n,assignments_fixed=True,disablement_possible=False,peak_vram_bytes=torch.cuda.max_memory_allocated())
    write(root/'summary.json',summary)
    return summary


def visualize(config):
    import torch
    from .fiber_soft_winding import points
    from .fiber_uv_layout import q_to_u
    from .fiber_uv_visualization import TiledRaster,VectorPDF,DrawingSinks
    root=Path(config['output_root']);out=root/'visualization';out.mkdir(exist_ok=True)
    d=arrays(root/'tensors.npz');curves={};endpoints={}
    same=(d['fixed_target']==0)&~d['parallel']
    for name in ('initial','solution'):
        s=arrays(config['visual_initial'] if name=='initial' and config.get('visual_initial') else root/(name+'.npz'))
        state=s['state'].copy();state[:,1]=s['shared_u'];state[:,2]=s['shared_v']
        def query(ids,ds):
            q,_,v=points(torch.tensor(state[ids]),torch.tensor(d['family'][ids]),torch.tensor(ds),torch.tensor(s['widths']))
            w=torch.tensor(s['widths'])
            return np.column_stack((q_to_u(q,w,width_boundaries(w)).numpy(),v.numpy()))
        ids=np.repeat(np.arange(len(state)),2);ds=np.column_stack((np.zeros(len(state)),d['end']*d['direction'])).ravel()
        curves[name]=query(ids,ds).reshape(-1,2,2)
        endpoints[name]=(query(d['i'][same],d['ds_a'][same]),query(d['j'][same],d['ds_b'][same]))
    both=np.concatenate(list(curves.values()));lo=both.min((0,1));hi=both.max((0,1))
    height=config.get('image_height',2000);scale=(height-1)/max(1,hi[1]-lo[1]);width=int(np.ceil((hi[0]-lo[0])*scale))+1
    def xy(p):return np.stack(((p[...,0]-lo[0])*scale,(hi[1]-p[...,1])*scale),-1)
    for name in curves:
        sinks=[TiledRaster(width,height),VectorPDF(width,height)];draw=DrawingSinks(*sinks)
        saved=arrays(config['visual_initial'] if name=='initial' and config.get('visual_initial') else root/(name+'.npz'))
        for boundary in np.r_[0,np.cumsum(saved['widths'])]:
            x=(boundary-lo[0])*scale
            if 0<=x<width:draw.line((x,0,x,height-1),fill=(215,215,215,255),width=1)
        for a,b in zip(*endpoints[name]):draw.line(tuple(xy(np.array([a,b])).ravel()),fill=(30,30,30,120),width=1)
        for k,line in enumerate(curves[name]):
            color=(27,108,168,255) if d['family'][k]==0 else (208,105,22,255)
            draw.line(tuple(xy(line).ravel()),fill=color,width=2)
        sinks[0].save(out/(name+'.tif'));sinks[1].save(out/(name+'.pdf'))
    write(out/'layout.json',dict(width=width,height=height,scale=scale,minimum=lo.tolist(),maximum=hi.tolist(),
          h_color='blue',v_color='orange',connections='fixed same-sheet perpendicular',axis_scale='equal'))


def run(config):
    from .fiber_normal_sign import launch_tensor_solve
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import read_polylines,write_initial_fibers
    from .fiber_graph_store import workflow_volume
    root=Path(config['output_root'])
    if (root/'summary.json').exists():
        frozen=json.loads((root/'config.json').read_text());frozen.pop('code_provenance',None)
        if frozen!=config:raise ValueError('Completed fit configuration differs')
        result=json.loads((root/'summary.json').read_text())
    else:
        prepare(config)
        result=launch_tensor_solve(config,__name__,Path(__file__))
    if config.get('visualize',True):visualize(config)
    if config.get('write_geometry_store',True) and not (root/'fibers.vcobj').exists():
        source=Path(config['flow_root'])/'selected_with_gt.vcobj'
        d=arrays(root/'tensors.npz')
        volume=workflow_volume(dict(fibers=str(source),working_volume=config.get('working_volume')))
        write_initial_fibers(root/'fibers.vcobj',read_polylines(source,d['source_id'],volume=volume),
                             volume=volume,packed=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--unmanaged',action='store_true');parser.add_argument('--devices')
    args=parser.parse_args()
    if not args.unmanaged:raise RuntimeError('Use the managed workflow')
    solve(json.loads(Path(args.config).read_text()))
