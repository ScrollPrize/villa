"""Conditional fixed-flow-label geometry fit with shared graph CG or L-BFGS.

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
    """Fit existing signed checkpoint inputs without restaging their arrays."""
    from .fiber_graph_store import workflow_volume
    fit=dict(config.get('parameters',{}),output_root=config['output_root'],
        input_store=config['fibers'],checkpoint_tensor_cache=config['tensor_cache'],
        checkpoint_solution=config['solution'],checkpoint_arcs_unsigned=False,
        component_selection=config.get('component_selection','largest'),
        working_volume=workflow_volume(config),umbilicus=config['umbilicus'],
        devices=config.get('devices','0'),gpu_manager=config['gpu_manager'])
    # Paths/settings stay scalar provisioning. The arrays are loaded once,
    # inside the managed child, never serialized to another input archive.
    for key in ('packed_geometry','workers','result_store','input_run_uuid','input_source'):
        if key in config:fit[key]=config[key]
    return run(fit)


def supported(d):
    """Induce the graph on perpendicular-supported fibers (no fallback geometry)."""
    degree=np.bincount(np.r_[d['i'][~d['parallel']],d['j'][~d['parallel']]],minlength=len(d['source_id']))
    keep=degree>0
    edge=keep[d['i']]&keep[d['j']]
    from .fiber_min_flow_benchmark import subset_minimal
    result=subset_minimal(d,keep)
    return result,keep,edge


def _checkpoint_binding(config, *, create):
    """Pin explicit legacy input content; never substitute size/mtime checks."""
    from .fiber_min_flow_experiment import digest
    from vc_object_store import CollectionReader
    source=config.get('input_source')
    if source is None:
        with CollectionReader(config['input_store']) as reader:source=reader.source
    binding=dict(tensor_sha256=digest(config['checkpoint_tensor_cache']),
                 solution_sha256=digest(config['checkpoint_solution']),input_source=source)
    manifest=Path(config['output_root'])/'checkpoint-inputs.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=binding:
            raise ValueError('Saved assignment inputs changed; choose a new output root')
    elif create:
        manifest.parent.mkdir(parents=True,exist_ok=True)
        write(manifest,binding)
    else:
        raise ValueError('Completed checkpoint fit lacks its input content binding')
    return binding


def prepare(config):
    """Read explicit checkpoint inputs once and return native-prepared arrays.

    This is an input adapter, not a cache writer. Store-to-store production
    workflows enter prepare_data directly and do not use checkpoint files.
    """
    original=arrays(config['checkpoint_tensor_cache'])
    solution=arrays(config['checkpoint_solution'])
    if 'source_id' not in solution:solution=dict(solution,source_id=original['source_id'])
    if 'component' not in solution:
        solution=dict(solution,component=np.zeros(len(original['source_id']),dtype=np.int64))
    if config.get('checkpoint_arcs_unsigned',False):
        # The historical flow_root boundary stores unsigned measured arcs.
        # Reuse the signed-checkpoint adapter with a neutral direction column;
        # this never changes the physical input endpoint positions.
        original=dict(original,node_ds=np.column_stack((np.zeros(len(original['source_id'])),
                                                       np.asarray(original['end']))))
    selected,labels=saved_assignment_inputs(original,solution,config.get('component_selection','all'))
    return prepare_data(config,selected,labels)


def prepare_data(config, data, labels, paths=None, *, supported_input=False):
    """Prepare owned native columns in memory, without staging solver files.

    Event arcs are unsigned measured arclength on entry. Native assembly can
    already remove unsupported fibers; the saved-input adapter retains its
    explicit selection path for existing checkpoints.
    """
    from .fiber_packed_priors import geometry_priors, fixed_target_columns, width_initial
    from .fiber_min_flow_full import PackedPaths
    started=time.monotonic()
    if supported_input:
        d=dict(data);keep=np.ones(len(labels),bool)
    else:
        from .fiber_uv_native_prepare import select_inputs
        d,chosen=select_inputs(data,workers=int(config.get('workers',32)))
        keep=chosen
    labels=np.asarray(labels)[keep].copy()
    if not len(labels):raise ValueError('No perpendicular-supported fibers')
    if not np.isfinite(labels).all() or np.any(labels!=np.rint(labels)):
        raise ValueError('Use saved integer assignments, not continuous potentials')
    if paths is None:
        paths=PackedPaths(config['packed_geometry']) if config.get('packed_geometry') else PackedPaths.from_store(config,d['source_id'])
    prior=geometry_priors(paths,d['source_id'],d['family'],config.get('prior_spacing',1000.),
                          workers=int(config.get('workers',32)))
    phase_done=time.monotonic()
    print(json.dumps(dict(stage='uv_geometry_priors',seconds=phase_done-started,
        fibers=len(labels),samples=len(prior['sample_owner']))),flush=True)
    shift=2-int(np.min(labels+prior['minimum_turn']));labels+=shift
    directions=prior['directions']
    target_columns=fixed_target_columns(paths,d,labels,directions,workers=int(config.get('workers',32)))
    target=target_columns['target']
    if not ((target==0)&~d['parallel']).any():raise ValueError('No fixed same-sheet perpendicular meetings')
    print(json.dumps(dict(stage='uv_fixed_targets',seconds=time.monotonic()-phase_done,
        events=len(target))),flush=True)
    bins=np.arange(int(np.max(labels+prior['maximum_turn']+1))+3)
    widths=width_initial(labels,prior['radii'],bins,workers=int(config.get('workers',32)))
    owner=prior['sample_owner']
    # These are final solver columns; no input dictionary is mutated, and no
    # unused SGD initializer/node arrays are constructed.
    events=d
    d={k:events[k] for k in ('source_id','family','end','i','j','parallel')}
    d.update(start=np.zeros(len(labels)),links=np.empty((0,4)),
        node_ds=np.column_stack((np.zeros(len(labels)),events['end']*directions)),
        ds_a=target_columns['ds_a'],ds_b=target_columns['ds_b'],
        fixed_winding=labels,fixed_target=target,phase=prior['phase'],z=prior['z'],
        width_initial=widths,direction=directions,sample_owner=owner,
        sample_ds=prior['sample_arc']*directions[owner],
        sample_phase=prior['sample_phase'],sample_z=prior['sample_z'])
    selection=dict(input_fibers=len(data['source_id']),retained=len(labels),events=len(target),
        global_integer_shift=shift,same_sheet_perpendicular=int(((target==0)&~d['parallel']).sum()))
    return d,selection


def solve(config, data=None, selection=None, *, snapshots=None):
    import torch
    from .fiber_shared_curve import SharedCurve
    from .fiber_scaled_lbfgs import uncertainty_joint
    from . import fiber_soft_winding as soft
    torch.set_num_threads(4)
    device='cuda';dtype=torch.float64
    root=Path(config['output_root']);d=arrays(root/'tensors.npz') if data is None else data
    started=time.monotonic()
    from vc_object_store import CollectionReader
    input_store=config.get('input_store',str(Path(config.get('flow_root',root))/'selected_with_gt.vcobj'))
    if config.get('input_source') is None:
        with CollectionReader(input_store) as source:input_source=source.source
        config=dict(config,input_store=input_store,input_source=input_source)
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
        checkpoint=config.get('write_solver_checkpoints',True)
        if not checkpoint and snapshots is None:return
        host_copy=lambda value:value.detach().to(device='cpu',copy=True).numpy()
        values=dict(state=host_copy(state),widths=host_copy(soft.decode_widths(raw,config)),
                    shared_u=host_copy(curve.u),shared_v=host_copy(curve.v))
        if snapshots is not None:snapshots[Path(name).stem]=values
        if checkpoint:
            np.savez(root/name,**values,
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
    optimizer=config.get('uv_optimizer','gauss_newton_cg')
    callback=lambda step:record(step) if step%config.get('log_every',100)==0 else None
    if optimizer=='gauss_newton_cg':
        from .fiber_flow_uv_cg import solve as solve_cg
        result=solve_cg(objective,iterations=config.get('steps',1000),callback=callback)
    elif optimizer=='torch_lbfgs':
        result=uncertainty_joint(objective,state,raw,extension,config.get('steps',1000),{},config,
            callback,freeze_winding=True,optimize_rejection=False,objective_parts=objective.parts)
    else:
        raise ValueError('UV optimizer must be gauss_newton_cg or torch_lbfgs')
    if history[-1]['iteration']!=result['iterations']:record(result['iterations'])
    save('solution.npz')
    assert np.array_equal(state[:,0].detach().cpu().numpy(),d['fixed_winding'])
    # Export the evaluated curve, not raw trainable offsets. Both coordinates
    # use the same fixed hypothesis and width map as the optimized objective.
    with torch.no_grad():
        widths=soft.decode_widths(raw,config)
        q,x,v=objective.samples()
        from .fiber_uv_native_prepare import write_uv_results
        shift=(selection if selection is not None else json.loads((root/'selection.json').read_text()))['global_integer_shift']
        run_uuid=write_uv_results(dict(config,result_store=str(config.get('result_store',root/'solution.vcobj'))),
            d,q,x,v,widths.cpu().numpy(),optimizer=optimizer,global_integer_shift=shift)
    summary=dict(run_uuid=run_uuid,optimizer=result,initial=history[0],final=history[-1],seconds=time.monotonic()-started,
        retained=n,assignments_fixed=True,disablement_possible=False,peak_vram_bytes=torch.cuda.max_memory_allocated())
    write(root/'summary.json',summary)
    return summary


def visualize(config, data=None, *, snapshots=None):
    import torch
    from .fiber_soft_winding import points
    from .fiber_uv_layout import q_to_u
    from .fiber_uv_visualization import TiledRaster,VectorPDF,DrawingSinks
    root=Path(config['output_root']);out=root/'visualization';out.mkdir(exist_ok=True)
    d=arrays(root/'tensors.npz') if data is None else data
    curves={};endpoints={}
    same=(d['fixed_target']==0)&~d['parallel']
    def saved_for(name):
        if name=='initial' and config.get('visual_initial'):return arrays(config['visual_initial'])
        if snapshots is not None:return snapshots[name]
        return arrays(root/(name+'.npz'))
    for name in ('initial','solution'):
        s=saved_for(name)
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
        saved=saved_for(name)
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


def _solve_checkpoint(config):
    """Managed child: prepare and solve without another process/file handoff."""
    from vc_object_store import CollectionReader
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import read_polylines,write_initial_fibers
    from .fiber_graph_store import workflow_volume
    root=Path(config['output_root'])
    root.mkdir(parents=True,exist_ok=True)
    with CollectionReader(config['input_store']) as source:input_source=source.source
    if config.get('input_source') is not None and config['input_source']!=input_source:
        raise ValueError('UV checkpoint source snapshot changed')
    config=dict(config,input_source=input_source)
    _checkpoint_binding(config,create=True)
    data,selection=prepare(config)
    write(root/'selection.json',selection)
    if config.get('visualize',True):
        snapshots={}
        result=solve(config,data,selection,snapshots=snapshots)
        visualize(config,data,snapshots=snapshots)
    else:
        result=solve(config,data,selection)
    if config.get('write_geometry_store',True) and not (root/'fibers.vcobj').exists():
        # Explicit compatibility export only. Production store-to-store UV
        # sets this false and retains immutable source geometry by reference.
        volume=workflow_volume(dict(fibers=config['input_store'],working_volume=config.get('working_volume')))
        write_initial_fibers(root/'fibers.vcobj',read_polylines(config['input_store'],data['source_id'],volume=volume),
                             volume=volume,packed=True)
    return result


def run(config):
    """Run native preparation and UV fitting in one managed process.

    ``flow_root`` remains an explicit legacy input boundary; it is normalized
    to the two existing checkpoint filenames, never copied to a new archive.
    """
    from .fiber_normal_sign import launch_tensor_solve
    config=dict(config,uv_checkpoint=True)
    if 'checkpoint_tensor_cache' not in config:
        parent=Path(config['flow_root'])
        config.update(checkpoint_tensor_cache=str(parent/'selected_tensors.npz'),
                      checkpoint_solution=str(parent/'selected_solution.npz'),
                      input_store=str(parent/'selected_with_gt.vcobj'),checkpoint_arcs_unsigned=True)
        config.setdefault('component_selection','all')
    root=Path(config['output_root'])
    if (root/'summary.json').exists():
        frozen=json.loads((root/'config.json').read_text());frozen.pop('code_provenance',None)
        if frozen!=config:raise ValueError('Completed fit configuration differs')
        _checkpoint_binding(config,create=False)
        return json.loads((root/'summary.json').read_text())
    return launch_tensor_solve(config,__name__,Path(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--unmanaged',action='store_true');parser.add_argument('--devices')
    args=parser.parse_args()
    if not args.unmanaged:raise RuntimeError('Use the managed workflow')
    config=json.loads(Path(args.config).read_text())
    if config.get('uv_checkpoint'):_solve_checkpoint(config)
    else:solve(config)
