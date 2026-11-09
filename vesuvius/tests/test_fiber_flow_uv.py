import numpy as np
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv import supported
from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_uv import saved_assignment_inputs
from vesuvius.neural_tracing.fiber_trace_3d.fiber_scaled_lbfgs import scaled_joint
from .fiber_graph_fixture import uid


def test_supported_preserves_indices_and_perp_support():
    d=dict(source_id=np.arange(4),family=np.array([0,1,0,0]),start=np.zeros(4),end=np.ones(4),
           node_ds=np.ones((4,2)),i=np.array([0,1,2]),j=np.array([1,2,3]),parallel=np.array([False,False,True]))
    for key in ('ds_a','ds_b','distance','sign'):d[key]=np.ones(3)
    result,keep,edges=supported(d)
    assert keep.tolist()==[True,True,True,False]
    assert edges.tolist()==[True,True,False]
    assert result['source_id'].tolist()==[0,1,2]


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


def test_saved_assignment_stages_native_store_without_resolving_windings(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_uv as uv
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_initial_fibers, read_polylines
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
        stage=tmp_path/'fit/fixed_input'
        assert config['flow_root']==str(stage)
        assert (stage/'selected_with_gt.vcobj').resolve()==source
        assert [r['id'] for r,_ in read_polylines(stage/'selected_with_gt.vcobj')]==ids.tolist()
        np.testing.assert_array_equal(np.load(stage/'selected_solution.npz')['winding'],[-4,-3])
        np.testing.assert_array_equal(np.load(stage/'selected_tensors.npz')['ds_a'],[3.])
        return {'checked':True}
    monkeypatch.setattr(uv,'run',fit)
    result=uv.run_saved(dict(fibers=str(source),tensor_cache=str(tensor),solution=str(solution),
        output_root=str(tmp_path/'fit'),parameters={},umbilicus='axis.json',gpu_manager={}))
    assert result=={'checked':True}
