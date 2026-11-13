import numpy as np
import torch
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv import supported
from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv import saved_assignment_inputs
from vesuvius.neural_tracing.fiber_trace_3d.fiber_scaled_lbfgs import scaled_joint
from .fiber_graph_fixture import uid


def _cg_fixture(device='cpu'):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_shared_curve import SharedCurve
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv_objective import FixedFlowObjective
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
    d=dict(family=np.array([0,1,0]),start=np.zeros(3),end=np.full(3,20.),
        node_ds=np.tile([0.,20.],(3,1)),i=np.array([0,1,0,2,1]),j=np.array([1,2,2,0,2]),
        ds_a=np.array([1.,2.,4.,5.,3.]),ds_b=np.array([2.,4.,5.,7.,6.]),
        parallel=np.array([False,False,True,True,False]),fixed_target=np.array([0.,1.,0.,-1.,0.]),
        sample_owner=np.array([0,0,1,1,1,2,2]),sample_ds=np.array([0.,20.,0.,10.,20.,0.,20.]),
        sample_phase=np.array([.1,.15,.2,.2,.2,.3,.35]),sample_z=np.arange(7.)*2.)
    config=dict(straight_fixed_length_geometry=True,shared_curve_integer_geometry=True,
                width_parameterization='monotone_relative',minimum_winding_width=100.,batch_size=3,
                soft_minimum_width=True,uv_cg_tolerance=1e-10)
    state=torch.nn.Parameter(torch.tensor([[2.,.1,10.,100.],[2.,.2,20.,100.],[3.,.3,30.,100.]],dtype=torch.float64,device=device))
    widths=torch.tensor([100.,110.,130.,160.,200.,250.],dtype=torch.float64,device=device)
    raw=torch.nn.Parameter(soft.encode_widths(widths,config))
    curve=SharedCurve(d,np.arange(3),np.zeros(3),config,device);curve.initialize(state,widths)
    return FixedFlowObjective(curve,state,raw,d,widths,config)


def test_uv_gn_uses_same_objective_and_exact_linearization():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv_cg import StraightSystem
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_cache import width_boundaries
    obj=_cg_fixture();system=StraightSystem(obj)
    width=soft.decode_widths(obj.raw,obj.config)
    x=soft.uv.q_to_u(obj.state[:,0]+obj.curve.u,width,width_boundaries(width)).detach().requires_grad_()
    value,terms=obj()
    horizontal=system.horizontal_loss(x,obj.raw)
    expected=terms['winding']+terms['phase_prior']+terms['width_minimum']+terms['width_smooth']
    qa,qb,va,vb,xa,xb=obj.edge_values(slice(None),width)
    expected+=(system.c_meeting*(xa-xb).square()).sum()
    torch.testing.assert_close(horizontal,expected,rtol=1e-12,atol=1e-14)
    gx,gr=torch.autograd.grad(horizontal,(x,obj.raw))
    operator,gradient,diagonal=system.linearize(x.detach(),obj.raw.detach())
    torch.testing.assert_close(gradient,torch.cat((gx,gr))*.5,rtol=1e-10,atol=1e-12)
    a=torch.linspace(.1,.9,len(gradient),dtype=gradient.dtype)
    b=a.flip(0)
    torch.testing.assert_close(torch.dot(a,operator(b)),torch.dot(b,operator(a)),rtol=1e-10,atol=1e-12)
    assert torch.dot(a,operator(a))>=0 and (diagonal>=0).all()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_uv_width_reduction_matches_direct_atomic_with_tail(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv_cg import StraightSystem
    system=StraightSystem(_cg_fixture(device));k=system.objective.raw.numel()
    generator=torch.Generator(device=device);generator.manual_seed(683)
    for count in (31,32*k,32*k+19,100_003):
        values=torch.randn(count,generator=generator,dtype=torch.float64,device=device)
        bins=torch.randint(k,(count,),generator=generator,device=device)
        expected=values.new_zeros(k).index_add_(0,bins,values)
        torch.testing.assert_close(system.scatter(values,bins,k),expected,rtol=1e-11,atol=1e-11)


def test_uv_gn_large_reduction_preserves_objective_gradient_and_operator():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv_cg import StraightSystem
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_cache import width_boundaries
    objective=_cg_fixture()
    # Enough repeated events to exercise grouped reductions in the complete
    # matrix-free normal equations, not merely the scalar histogram helper.
    for key in ('i','j','ds_a','ds_b','parallel','fixed_target'):
        objective.data[key]=np.tile(objective.data[key],1000)
    objective=type(objective)(objective.curve,objective.state,objective.raw,objective.data,
                              objective.initial_width,objective.config)
    system=StraightSystem(objective);width=soft.decode_widths(objective.raw,objective.config)
    x=soft.uv.q_to_u(objective.state[:,0]+objective.curve.u,width,width_boundaries(width)).detach().requires_grad_()
    loss=system.horizontal_loss(x,objective.raw)
    grad=torch.cat(torch.autograd.grad(loss,(x,objective.raw)))*.5
    operator,actual,_=system.linearize(x.detach(),objective.raw.detach())
    torch.testing.assert_close(actual,grad,rtol=1e-10,atol=1e-12)
    a=torch.linspace(-.4,.8,len(grad),dtype=grad.dtype);b=a.flip(0)
    torch.testing.assert_close(torch.dot(a,operator(b)),torch.dot(b,operator(a)),rtol=1e-10,atol=1e-12)


@pytest.mark.parametrize('mode',['softplus','monotone_additive','monotone_relative'])
def test_uv_gn_operator_matches_explicit_residual_jacobian(mode):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv_cg import StraightSystem
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_cache import width_boundaries
    objective=_cg_fixture();objective.config['width_parameterization']=mode
    objective.config['soft_minimum_width']=mode=='monotone_relative'
    raw=soft.encode_widths(objective.initial_width,objective.config)
    system=StraightSystem(objective)
    # Both extrapolation tails are part of the model; no outer wrapping/carry.
    x=torch.tensor([-30.,50.,2000.],dtype=torch.float64)
    packed=torch.cat((x,raw))
    def residual(values):
        x,raw=values[:system.n],values[system.n:]
        widths=soft.decode_widths(raw,objective.config);bounds=width_boundaries(widths)
        a=system.plan(x,widths,bounds,system.i,system.dua)
        b=system.plan(x,widths,bounds,system.j,system.dub)
        p=system.plan(x,widths,bounds,system.owner,system.du)
        return torch.cat((system.c_winding.sqrt()*(system.q(b)-system.q(a)-system.target),
                          system.c_meeting.sqrt()*(x[system.j]+system.dub-x[system.i]-system.dua),
                          system.c_phase.sqrt()*soft.circular(system.q(p)-system.phase),
                          system.regularizer(raw)))
    jacobian=torch.func.jacrev(residual)(packed)
    operator,gradient,_,block=system.linearize(x,raw,width_block=True)
    direction=torch.linspace(-.7,.9,len(packed),dtype=packed.dtype)
    torch.testing.assert_close(operator(direction),jacobian.T@(jacobian@direction),rtol=2e-11,atol=1e-12)
    torch.testing.assert_close(gradient,jacobian.T@residual(packed),rtol=2e-11,atol=1e-12)
    torch.testing.assert_close(block,jacobian[:,system.n:].T@jacobian[:,system.n:],rtol=2e-11,atol=1e-12)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_uv_gn_reduces_full_objective_without_changing_labels_or_lengths(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv_cg import solve
    obj=_cg_fixture(device);labels=obj.state[:,0].detach().clone()
    initial=float(obj()[0].detach());history=[]
    result=solve(obj,iterations=20,callback=lambda _:history.append(float(obj()[0])))
    assert result['iterations']>0 and result['total_cg_iterations']>0
    assert history[-1]<initial*.5
    assert result['history'][-1]['loss']==pytest.approx(history[-1],rel=1e-10,abs=1e-12)
    assert np.all(np.diff(history)<=1e-10)
    torch.testing.assert_close(labels,obj.state[:,0],rtol=0,atol=0)
    width=__import__('vesuvius.neural_tracing.fiber_trace_3d.fiber_soft_winding',fromlist=['decode_widths']).decode_widths(obj.raw,obj.config)
    for owner,family in enumerate(obj.data['family']):
        ids=np.array([owner,owner]);arc=np.array([0.,20.])
        q,_,v=obj.query(ids,arc,width)
        from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import q_to_u
        from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_cache import width_boundaries
        u=q_to_u(q,width,width_boundaries(width))
        delta=torch.stack((u[1]-u[0],v[1]-v[0]))
        torch.testing.assert_close(delta,delta.new_tensor([20.,0.] if family==0 else [0.,20.]),rtol=1e-10,atol=1e-10)


def test_supported_preserves_indices_and_perp_support():
    d=dict(source_id=np.arange(4),family=np.array([0,1,0,0]),start=np.zeros(4),end=np.ones(4),
           node_ds=np.ones((4,2)),i=np.array([0,1,2]),j=np.array([1,2,3]),parallel=np.array([False,False,True]))
    for key in ('ds_a','ds_b','distance','sign'):d[key]=np.ones(3)
    result,keep,edges=supported(d)
    assert keep.tolist()==[True,True,True,False]
    assert edges.tolist()==[True,True,False]
    assert result['source_id'].tolist()==[0,1,2]


def _packed_uv_inputs(tmp_path,reverse_h=False,reverse_v=False):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import Seam
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_full import PackedPaths,KEY_DTYPE,PACKED_GEOMETRY_FORMAT
    axis=np.array([[0.,0.,-20.],[0.,0.,20.]])
    curves=[np.array([[5.,-np.sqrt(75.),0.],[10.,0.,0.],[5.,np.sqrt(75.),0.]]),
            np.array([[10.,0.,-2.],[10.,0.,0.],[10.,0.,2.]]),
            np.array([[10.,-np.sqrt(300.),5.],[20.,0.,5.],[10.,np.sqrt(300.),5.]])]
    if reverse_h:curves[0]=curves[0][::-1].copy()
    if reverse_v:curves[1]=curves[1][::-1].copy()
    ids=np.array(list(map(uid,[10,20,30])))
    offsets=np.r_[0,np.cumsum(list(map(len,curves)))];seam=Seam(axis)
    keys=np.empty(offsets[-1],KEY_DTYPE);turns=np.empty(offsets[-1],np.int64);lengths=[]
    for owner,points in enumerate(curves):
        sl=slice(offsets[owner],offsets[owner+1]);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        keys['source'][sl]=owner;keys['arc'][sl]=arc;turns[sl]=seam.polyline(points);lengths.append(arc[-1])
    root=tmp_path/'packed';root.mkdir()
    np.savez(root/'catalog.npz',format_version=PACKED_GEOMETRY_FORMAT,volume=uid(999),
             source_id=ids,point_offset=offsets,axis=axis)
    np.save(root/'keys.npy',keys);np.save(root/'xyz.npy',np.concatenate(curves));np.save(root/'turns.npy',turns)
    data=dict(source_id=ids,family=np.array([0,1,0]),end=np.asarray(lengths),
              i=np.array([0,0]),j=np.array([1,2]),ds_a=np.array([10.,5.]),ds_b=np.array([2.,0.]),
              parallel=np.array([False,True]))
    return PackedPaths(root),data,np.array([-4.,-4.,8.]),curves


@pytest.mark.parametrize('reverse_h,reverse_v',[(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('workers',[1,4])
def test_prepare_uv_in_memory_has_no_npz_roundtrip_and_preserves_inputs(tmp_path,monkeypatch,reverse_h,reverse_v,workers):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_uv as uv
    paths,data,labels,curves=_packed_uv_inputs(tmp_path,reverse_h,reverse_v)
    original={key:value.copy() for key,value in data.items()};original_labels=labels.copy()
    def forbidden(*args,**kwargs):raise AssertionError('In-memory UV preparation must not serialize or reread NPZ')
    monkeypatch.setattr(np,'savez',forbidden);monkeypatch.setattr(np,'savez_compressed',forbidden)
    monkeypatch.setattr(np,'load',forbidden)
    output=tmp_path/'must-not-be-created'
    prepared,selection=uv.prepare_data(dict(output_root=str(output),prior_spacing=8.,workers=workers),data,labels,paths)
    assert not output.exists()
    for key,value in original.items():np.testing.assert_array_equal(data[key],value)
    np.testing.assert_array_equal(labels,original_labels)
    np.testing.assert_array_equal(prepared['source_id'],original['source_id'][:2])
    np.testing.assert_array_equal(prepared['family'],[0,1])
    np.testing.assert_array_equal(prepared['i'],[0]);np.testing.assert_array_equal(prepared['j'],[1])
    np.testing.assert_array_equal(prepared['parallel'],[False])
    direction=np.array([-1. if reverse_h else 1.,-1. if reverse_v else 1.])
    np.testing.assert_array_equal(prepared['direction'],direction)
    np.testing.assert_array_equal(prepared['ds_a'],[10.*direction[0]])
    np.testing.assert_array_equal(prepared['ds_b'],[2.*direction[1]])
    np.testing.assert_array_equal(prepared['fixed_winding'],[2.,2.])
    np.testing.assert_array_equal(prepared['fixed_target'],[0.])
    np.testing.assert_array_equal(prepared['start'],[0.,0.])
    np.testing.assert_array_equal(prepared['end'],original['end'][:2])
    assert prepared['node_ds'].shape==(2,2)
    np.testing.assert_array_equal(prepared['node_ds'][:,0],[0.,0.])
    np.testing.assert_allclose(prepared['node_ds'][:,-1],original['end'][:2]*direction,rtol=0,atol=1e-14)
    np.testing.assert_allclose(prepared['width_initial'],np.full(6,20*np.pi),rtol=1e-14,atol=1e-14)
    for owner,points in enumerate(curves[:2]):
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        at=np.linspace(0.,arc[-1],max(2,int(np.ceil(arc[-1]/8.))+1))
        xyz=np.column_stack([np.interp(at,arc,points[:,dim]) for dim in range(3)])
        phase=((np.arctan2(xyz[:,1],xyz[:,0])+np.pi/2)%(2*np.pi))/(2*np.pi)
        chosen=prepared['sample_owner']==owner
        np.testing.assert_allclose(prepared['sample_ds'][chosen],at*direction[owner],rtol=0,atol=2e-14)
        np.testing.assert_allclose(prepared['sample_phase'][chosen],phase,rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(prepared['sample_z'][chosen],xyz[:,2],rtol=1e-14,atol=1e-14)
        assert prepared['phase'][owner]==pytest.approx(phase[0],abs=1e-14)
        assert prepared['z'][owner]==points[0,2]
    expected_selection=dict(input_fibers=3,retained=2,events=1,
                            global_integer_shift=6,same_sheet_perpendicular=1)
    for key,value in expected_selection.items():assert selection[key]==value
    assert 'removed_source_ids' not in selection


def test_prepare_uv_rejects_fractional_fixed_labels_without_writes(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_uv as uv
    paths,data,labels,_=_packed_uv_inputs(tmp_path)
    labels[1]+=.25
    def forbidden(*args,**kwargs):raise AssertionError('Rejected labels must not produce work artifacts')
    monkeypatch.setattr(np,'savez',forbidden)
    with pytest.raises(ValueError,match='integer'):
        uv.prepare_data(dict(prior_spacing=8.),data,labels,paths)


def test_freeze_winding_keeps_labels_but_fits_geometry():
    state=torch.nn.Parameter(torch.tensor([[3.,0,0,100.]],dtype=torch.float64))
    raw=torch.nn.Parameter(torch.ones(2,dtype=torch.float64))
    ext=torch.nn.Module();ext.register_parameter('u',torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64)))
    def obj():return state[:,0].square().sum()+(ext.u-5).square().sum()+raw.sum()*0,{}
    scaled_joint(obj,state,raw,ext,100,freeze_winding=True,freeze_widths=True)
    assert state[0,0]==3
    assert abs(float(ext.u.detach())-5)<1e-5


def test_saved_assignment_uses_main_component_and_unsigned_arcs():
    d=dict(source_id=np.array(list(map(uid,[7,8,9]))),family=np.array([0,1,0]),start=np.zeros(3),
        end=np.ones(3)*10,node_ds=np.array([[0.,-10],[0,10],[0,10]]),links=np.empty((0,4)),
        i=np.array([0]),j=np.array([1]),ds_a=np.array([-3.]),ds_b=np.array([4.]),
        distance=np.array([.5]),sign=np.ones(1),parallel=np.zeros(1,bool))
    s=dict(source_id=d['source_id'],component=np.array([0,0,1]),winding=np.array([-4,-3,99]))
    out,labels=saved_assignment_inputs(d,s)
    np.testing.assert_array_equal(out['source_id'],list(map(uid,[7,8])))
    np.testing.assert_array_equal(labels,[-4,-3])
    np.testing.assert_array_equal(out['ds_a'],[3])
    np.testing.assert_array_equal(out['node_ds'],[[0,10],[0,10]])
    np.testing.assert_array_equal(d['ds_a'],[-3])


def test_saved_assignment_passes_explicit_inputs_without_restaging_or_resolving_windings(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_uv as uv
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_initial_fibers
    ids=np.array([uid(7),uid(8)])
    d=dict(source_id=ids,family=np.array([0,1]),start=np.zeros(2),end=np.ones(2)*10,
        node_ds=np.array([[0.,-10.],[0.,10.]]),links=np.empty((0,4)),i=np.array([0]),j=np.array([1]),
        ds_a=np.array([-3.]),ds_b=np.array([4.]),distance=np.array([.5]),sign=np.ones(1),parallel=np.zeros(1,bool))
    source=tmp_path/'fibers.vcobj'
    write_initial_fibers(source,[(dict(id=s),np.array([[0.,0,0],[10.,0,0]])) for s in ids],
                         volume=uid(999),workers=2,packed=True)
    tensor=tmp_path/'tensors.npz';np.savez(tensor,**d)
    solution=tmp_path/'solution.npz';np.savez(solution,source_id=ids,component=[0,0],winding=[-4,-3])
    def fit(config):
        assert config['input_store']==str(source)
        assert config['checkpoint_tensor_cache']==str(tensor)
        assert config['checkpoint_solution']==str(solution)
        assert config['checkpoint_arcs_unsigned'] is False
        assert config['component_selection']=='largest'
        assert not (tmp_path/'fit').exists()
        return {'checked':True}
    monkeypatch.setattr(uv,'run',fit)
    def forbidden(*args,**kwargs):raise AssertionError('Parent must not read or serialize checkpoint arrays')
    monkeypatch.setattr(uv,'arrays',forbidden);monkeypatch.setattr(np,'savez',forbidden)
    result=uv.run_saved(dict(fibers=str(source),tensor_cache=str(tensor),solution=str(solution),
        output_root=str(tmp_path/'fit'),parameters={},umbilicus='axis.json',gpu_manager={}))
    assert result=={'checked':True}


@pytest.mark.parametrize('signed_input',[False,True])
def test_checkpoint_prepare_reads_existing_inputs_once_and_calls_native_preparation(tmp_path,monkeypatch,signed_input):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_uv as uv
    ids=np.array([uid(7),uid(8)])
    original=dict(source_id=ids,family=np.array([0,1]),start=np.zeros(2),end=np.ones(2)*10,
        node_ds=np.array([[0.,-10.],[0.,10.]]),links=np.empty((0,4)),i=np.array([0]),j=np.array([1]),
        ds_a=np.array([-3.] if signed_input else [3.]),ds_b=np.array([4.]),
        distance=np.array([.5]),sign=np.ones(1),parallel=np.zeros(1,bool))
    # Historical flow_root checkpoints only stored winding; full solver
    # checkpoints additionally identify source fibers and components.
    solution=dict(winding=np.array([-4,-3]))
    if signed_input:solution.update(source_id=ids,component=np.array([0,0]))
    calls=[]
    def read(path):
        calls.append(path)
        return original if path=='existing-input.npz' else solution
    prepared=dict(marker=np.array([99.]));selection=dict(retained=2)
    def native(config,data,labels,paths=None,**kwargs):
        np.testing.assert_array_equal(data['ds_a'],[3.])
        np.testing.assert_array_equal(data['ds_b'],[4.])
        np.testing.assert_array_equal(labels,[-4,-3])
        assert paths is None and kwargs=={}
        return prepared,selection
    def forbidden(*args,**kwargs):raise AssertionError('Checkpoint preparation must not restage arrays')
    monkeypatch.setattr(uv,'arrays',read);monkeypatch.setattr(uv,'prepare_data',native)
    monkeypatch.setattr(np,'savez',forbidden)
    result=uv.prepare(dict(checkpoint_tensor_cache='existing-input.npz',checkpoint_solution='existing-solution.npz',
                          checkpoint_arcs_unsigned=not signed_input,output_root=str(tmp_path/'unused')))
    assert result==(prepared,selection)
    assert calls==['existing-input.npz','existing-solution.npz']
    np.testing.assert_array_equal(original['ds_a'],[-3.] if signed_input else [3.])
    assert not (tmp_path/'unused').exists()


def test_checkpoint_runner_launches_before_loading_or_preparing_arrays(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_uv as uv
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_normal_sign as normal
    root=tmp_path/'output';flow=tmp_path/'input'
    def forbidden(*args,**kwargs):raise AssertionError('Parent may only configure managed child, not import solver arrays')
    monkeypatch.setattr(uv,'prepare',forbidden);monkeypatch.setattr(uv,'arrays',forbidden)
    monkeypatch.setattr(np,'savez',forbidden)
    def managed(config,module,source):
        assert config['uv_checkpoint'] is True
        assert config['checkpoint_arcs_unsigned'] is True
        assert config['checkpoint_tensor_cache']==str(flow/'selected_tensors.npz')
        assert config['checkpoint_solution']==str(flow/'selected_solution.npz')
        assert config['input_store']==str(flow/'selected_with_gt.vcobj')
        assert config['component_selection']=='all'
        assert module==uv.__name__ and str(source)==uv.__file__
        assert not root.exists()
        return dict(run_uuid='result-run')
    monkeypatch.setattr(normal,'launch_tensor_solve',managed)
    result=uv.run(dict(flow_root=str(flow),output_root=str(root),gpu_manager={}))
    assert result==dict(run_uuid='result-run')


def test_checkpoint_child_keeps_prepared_buffers_for_the_solver(tmp_path,monkeypatch):
    import vc_object_store
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_uv as uv
    source_snapshot=dict(store_uuid='store',generation='generation',commit_sequence=4)
    data=dict(source_id=np.array([uid(7),uid(8)]));selection=dict(global_integer_shift=3)
    config=dict(input_store='existing.vcobj',output_root=str(tmp_path/'fit'),
                visualize=False,write_geometry_store=False)
    class Reader:
        def __init__(self,path):assert path=='existing.vcobj'
        def __enter__(self):return self
        def __exit__(self,*args):return False
        source=source_snapshot
    def prepare(value):
        assert value['input_source'] is source_snapshot
        return data,selection
    def solve(value,actual_data,actual_selection):
        assert actual_data is data and actual_selection is selection
        assert value['input_source'] is source_snapshot
        return dict(run_uuid='result-run')
    def forbidden(*args,**kwargs):raise AssertionError('Child must not write intermediate NPZ tensor archives')
    def binding(value,*,create):
        assert value['input_source'] is source_snapshot
        assert create is True
    monkeypatch.setattr(vc_object_store,'CollectionReader',Reader)
    monkeypatch.setattr(uv,'_checkpoint_binding',binding)
    monkeypatch.setattr(uv,'prepare',prepare);monkeypatch.setattr(uv,'solve',solve)
    monkeypatch.setattr(np,'savez',forbidden)
    assert uv._solve_checkpoint(config)==dict(run_uuid='result-run')
    assert not list((tmp_path/'fit').glob('*.npz'))


def test_checkpoint_binding_detects_same_size_input_changes(tmp_path):
    import hashlib
    import json
    import os
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_uv as uv
    tensor=tmp_path/'existing-input.npz';solution=tmp_path/'existing-solution.npz'
    tensor.write_bytes(b'original');solution.write_bytes(b'solution')
    stamp=tensor.stat()
    config=dict(checkpoint_tensor_cache=str(tensor),checkpoint_solution=str(solution),
                input_source=dict(store_uuid='source',generation='generation',commit_sequence=7),
                output_root=str(tmp_path/'output'))
    before=uv._checkpoint_binding(config,create=True)
    assert before['tensor_sha256']==hashlib.sha256(b'original').hexdigest()
    assert uv._checkpoint_binding(config,create=False)==before
    manifest=tmp_path/'output/checkpoint-inputs.json'
    assert json.loads(manifest.read_text())==before
    # Content changes must not pass merely because file length and timestamps
    # are unchanged, nor overwrite the original binding on a retry.
    tensor.write_bytes(b'changed!')
    os.utime(tensor,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    assert tensor.stat().st_size==stamp.st_size
    for create in (False,True):
        with pytest.raises(ValueError,match='inputs changed'):
            uv._checkpoint_binding(config,create=create)
    assert json.loads(manifest.read_text())==before


def test_solve_summary_identifies_persisted_result_run(tmp_path,monkeypatch):
    import json
    import uuid
    from vc_object_store import CollectionReader
    from .fiber_graph_fixture import measured_store
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_uv as uv
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_shared_curve as shared,fiber_scaled_lbfgs as scaled
    source=measured_store(tmp_path,[(0,10,20,False,.5)])
    root=tmp_path/'fit';root.mkdir();flow=tmp_path/'flow';flow.mkdir()
    (flow/'selected_with_gt.vcobj').symlink_to(source)
    np.savez(root/'tensors.npz',source_id=np.array([uid(10),uid(20)]),
        family=[0,1],start=[0.,0.],end=[10.,10.],node_ds=[[0.,10.],[0.,10.]],direction=[1.,1.],
        i=[0],j=[1],ds_a=[5.],ds_b=[5.],parallel=[False],fixed_target=[0.],
        fixed_winding=[2.,2.],phase=[.1,.1],z=[10.,10.],width_initial=np.arange(1000.,1600.,100.),
        sample_owner=[0,0,1,1],sample_ds=[0.,10.,0.,10.],sample_phase=[.1,.11,.1,.1],sample_z=[10.,10.,10.,20.])
    (root/'selection.json').write_text(json.dumps(dict(global_integer_shift=3)))
    # Exercise the real geometry/readout/store writer on CPU; optimizer and CUDA
    # placement are irrelevant to this result-identity regression.
    original_tensor=torch.as_tensor;original_curve=shared.SharedCurve
    def cpu_tensor(value,*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return original_tensor(value,*args,**kwargs)
    monkeypatch.setattr(torch,'as_tensor',cpu_tensor)
    monkeypatch.setattr(shared,'SharedCurve',lambda d,owner,arc,config,device:original_curve(d,owner,arc,config,'cpu'))
    monkeypatch.setattr(scaled,'uncertainty_joint',lambda *args,**kwargs:dict(iterations=0))
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    output=tmp_path/'uv.vcobj'
    summary=uv.solve(dict(output_root=str(root),flow_root=str(flow),result_store=str(output),uv_optimizer='torch_lbfgs'))
    assert uuid.UUID(summary['run_uuid'])
    assert json.loads((root/'summary.json').read_text())['run_uuid']==summary['run_uuid']
    with CollectionReader(str(output)) as store:
        runs=store.extension_rows('ext_fiber_graph_solve_runs')
        positions=store.extension_rows('ext_fiber_graph_solve_positions')
    assert len(runs)==1 and runs[0][0]==uuid.UUID(summary['run_uuid']).bytes
    assert all(row[2]==3 and row[3]==2 for row in positions)
    assert abs(positions[0][-3]-.1)<1e-12  # U excludes the selected k=2.


def test_direct_solve_uses_owned_arrays_without_input_npz_and_pins_source_store(tmp_path,monkeypatch):
    import json
    import uuid
    from pathlib import Path
    from vc_object_store import CollectionReader
    from .fiber_graph_fixture import measured_store
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_uv as uv
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_shared_curve as shared,fiber_scaled_lbfgs as scaled
    source=measured_store(tmp_path,[(0,10,20,False,.5)])
    root=tmp_path/'fit';root.mkdir()
    data={key:np.asarray(value) for key,value in dict(
        source_id=[uid(10),uid(20)],family=[0,1],start=[0.,0.],end=[10.,10.],
        node_ds=[[0.,10.],[0.,10.]],direction=[1.,1.],i=[0],j=[1],ds_a=[5.],ds_b=[5.],
        parallel=[False],fixed_target=[0.],fixed_winding=[2.,2.],phase=[.1,.1],z=[10.,10.],
        width_initial=np.arange(1000.,1600.,100.),sample_owner=[0,0,1,1],sample_ds=[0.,10.,0.,10.],
        sample_phase=[.1,.11,.1,.1],sample_z=[10.,10.,10.,20.]).items()}
    before={key:value.copy() for key,value in data.items()}
    original_tensor=torch.as_tensor;original_curve=shared.SharedCurve;original_savez=np.savez
    def cpu_tensor(value,*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return original_tensor(value,*args,**kwargs)
    def forbidden_read(*args,**kwargs):raise AssertionError('Direct solve must not load imported tensor files')
    def result_checkpoint(path,*args,**kwargs):
        assert Path(path).name in ('initial.npz','solution.npz'), 'Only explicit result checkpoints may be NPZ'
        return original_savez(path,*args,**kwargs)
    monkeypatch.setattr(torch,'as_tensor',cpu_tensor)
    monkeypatch.setattr(shared,'SharedCurve',lambda d,owner,arc,config,device:original_curve(d,owner,arc,config,'cpu'))
    monkeypatch.setattr(scaled,'uncertainty_joint',lambda *args,**kwargs:dict(iterations=0))
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(uv,'arrays',forbidden_read);monkeypatch.setattr(np,'load',forbidden_read)
    monkeypatch.setattr(np,'savez',result_checkpoint)
    output=tmp_path/'uv.vcobj'
    summary=uv.solve(dict(output_root=str(root),input_store=str(source),result_store=str(output),
                          uv_optimizer='torch_lbfgs'),data,dict(global_integer_shift=3))
    assert uuid.UUID(summary['run_uuid'])
    assert json.loads((root/'summary.json').read_text())['run_uuid']==summary['run_uuid']
    assert not (root/'tensors.npz').exists() and not (root/'selected_tensors.npz').exists()
    assert not (root/'fixed_input').exists()
    for key,value in before.items():np.testing.assert_array_equal(data[key],value)
    with CollectionReader(str(output)) as store:
        assert len(store.objects('fiber'))==2
        positions=store.extension_rows('ext_fiber_graph_solve_positions')
        runs=store.extension_rows('ext_fiber_graph_solve_runs')
        inputs=store.extension_rows('ext_fiber_graph_solve_inputs')
    assert len(runs)==1 and len(inputs)==1
    assert all(row[2]==3 and row[3]==2 for row in positions)
    assert abs(positions[0][-3]-.1)<1e-12


def test_fixed_flow_objective_batches_preserve_full_loss_and_gradients():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_shared_curve import SharedCurve
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv_objective import FixedFlowObjective
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
    d=dict(family=np.array([0,1,0]),start=np.zeros(3),end=np.full(3,20.),
        node_ds=np.tile([0.,20.],(3,1)),i=np.array([0,1,0,2,1]),j=np.array([1,2,2,0,2]),
        ds_a=np.array([1.,2.,4.,5.,3.]),ds_b=np.array([2.,4.,5.,7.,6.]),
        parallel=np.array([False,False,True,True,False]),fixed_target=np.array([0.,1.,0.,-1.,0.]),
        sample_owner=np.array([0,0,1,1,1,2,2]),sample_ds=np.array([0.,20.,0.,10.,20.,0.,20.]),
        sample_phase=np.array([.1,.15,.2,.2,.2,.3,.35]),sample_z=np.arange(7.)*2.)
    config=dict(straight_fixed_length_geometry=True,shared_curve_integer_geometry=True,
                width_parameterization='monotone_relative',minimum_winding_width=100.,batch_size=99)
    state=torch.nn.Parameter(torch.tensor([[2.,.1,10.,100.],[2.,.2,20.,100.],[3.,.3,30.,100.]],dtype=torch.float64))
    widths=torch.tensor([100.,110.,130.,160.,200.,250.],dtype=torch.float64)
    raw=torch.nn.Parameter(soft.encode_widths(widths,config))
    curve=SharedCurve(d,np.arange(3),np.zeros(3),config,'cpu');curve.initialize(state,widths)
    targets=(state,raw,*curve.parameters())
    full=FixedFlowObjective(curve,state,raw,d,widths,config)
    value,terms=full();gradient=torch.autograd.grad(value,targets,allow_unused=True)
    # Independently retain the original all-at-once reduction as the oracle.
    decoded=soft.decode_widths(raw,config)
    qa,qb,va,vb,xa,xb=full.edge_values(slice(None),decoded)
    t=lambda x:torch.as_tensor(x,dtype=torch.float64)
    parallel=torch.as_tensor(d['parallel']);family=torch.as_tensor(d['family'][d['i']])
    same=torch.as_tensor(full.same);weight=t(full.weights)
    _,phase,z=full.query(d['sample_owner'],d['sample_ds'],decoded)
    sample_weight=1/t(np.bincount(d['sample_owner'])[d['sample_owner']])
    reference=dict(
        winding=(weight*((~parallel)|(family==1))*(qb-qa-t(d['fixed_target'])).square()).mean(),
        vertical=(weight*((~parallel)|(family==0))*((vb-va)/1000.).square()).mean(),
        meeting=((xa[same]-xb[same]).square()+(va[same]-vb[same]).square()).mean()/128.**2,
        phase_prior=(soft.circular(phase-t(d['sample_phase'])).square()*sample_weight).sum()/3*.000625,
        z_prior=(((z-t(d['sample_z']))/1000.).square()*sample_weight).sum()/3*.01)
    reference.update(soft.width_regularization(raw,decoded,widths,config))
    for key,expected in reference.items():torch.testing.assert_close(terms[key],expected,atol=1e-12,rtol=1e-12)
    reference_gradient=torch.autograd.grad(sum(reference.values()),targets,allow_unused=True)
    for expected,actual in zip(reference_gradient,gradient):
        if expected is not None:torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    for batch in (1,2,3):
        objective=FixedFlowObjective(curve,state,raw,d,widths,dict(config,batch_size=batch))
        total=0.;accumulated=[torch.zeros_like(p) for p in targets]
        for part,_ in objective.parts():
            total+=float(part.detach())
            for target,grad in zip(accumulated,torch.autograd.grad(part,targets,allow_unused=True)):
                if grad is not None:target.add_(grad)
        np.testing.assert_allclose(total,float(value.detach()),rtol=1e-12)
        for expected,actual in zip(gradient,accumulated):
            if expected is not None:torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
        for expected,actual in zip(full.errors(),objective.errors()):np.testing.assert_array_equal(actual,expected)
        for expected,actual in zip(full.samples(),objective.samples()):np.testing.assert_array_equal(actual,expected)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA requires managed host validation')
def test_fixed_flow_objective_cuda_microbenchmark():
    """Bounded synthetic GPU closure smoke; report, not a flaky speed assertion."""
    import json
    import time
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_shared_curve import SharedCurve
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv_objective import FixedFlowObjective
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
    n,m=8192,131072;rng=np.random.default_rng(42)
    d=dict(family=np.arange(n)%2,start=np.zeros(n),end=np.full(n,20.),
        node_ds=np.tile([0.,20.],(n,1)),i=rng.integers(n,size=m),j=rng.integers(n,size=m),
        ds_a=rng.uniform(0,20,m),ds_b=rng.uniform(0,20,m),parallel=np.arange(m)%3==0,
        fixed_target=np.zeros(m),sample_owner=np.repeat(np.arange(n),2),sample_ds=np.tile([0.,20.],n),
        sample_phase=np.full(n*2,.2),sample_z=np.zeros(n*2))
    config=dict(straight_fixed_length_geometry=True,shared_curve_integer_geometry=True,
                width_parameterization='monotone_relative',batch_size=32768)
    state=torch.nn.Parameter(torch.tensor(np.column_stack((np.full(n,2.),np.full(n,.2),np.zeros(n),np.full(n,100.))),
                                          device='cuda',dtype=torch.float64))
    widths=torch.tensor([1000.,1100.,1300.,1600.,2000.,2500.],device='cuda',dtype=torch.float64)
    raw=torch.nn.Parameter(soft.encode_widths(widths,config))
    curve=SharedCurve(d,np.arange(n),np.zeros(n),config,'cuda');curve.initialize(state,widths)
    objective=FixedFlowObjective(curve,state,raw,d,widths,config);targets=(state,raw,*curve.parameters())
    torch.cuda.reset_peak_memory_stats();times=[]
    for repeat in range(3):
        gradients=[torch.zeros_like(p) for p in targets]
        torch.cuda.synchronize();started=time.perf_counter()
        for part,_ in objective.parts():
            assert torch.isfinite(part)
            for total,gradient in zip(gradients,torch.autograd.grad(part,targets,allow_unused=True)):
                if gradient is not None:total.add_(gradient)
        torch.cuda.synchronize();times.append(time.perf_counter()-started)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
    print(json.dumps(dict(benchmark='fixed_flow_uv_closure',fibers=n,edges=m,batch_size=32768,
        warmup_seconds=times[0],closure_seconds=times[1:],
        peak_vram_bytes=torch.cuda.max_memory_allocated(),
        full_135m_edge_equivalent_seconds=135000000/m*np.mean(times[1:]))),flush=True)
