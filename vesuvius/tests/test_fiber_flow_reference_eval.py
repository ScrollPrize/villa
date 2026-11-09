import numpy as np
import json
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import Seam
from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_reference_eval import sample_arcs,continuous_winding,error_summary,analyze
from .fiber_graph_fixture import uid


def test_solution_component_maps_to_original_geometry_indices():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_reference_eval import solution_catalog_indices
    catalog=dict(source_id=np.array([uid(k) for k in (0,3,7,11)]),family=np.array([0,1,0,1]))
    solution=dict(source_id=np.array([uid(k) for k in (0,7,11)]),family=np.array([0,0,1]))
    np.testing.assert_array_equal(solution_catalog_indices(catalog,solution),[0,2,3])
    with pytest.raises(ValueError,match='classification'):
        solution_catalog_indices(catalog,dict(solution,family=np.array([1,0,1])))
    with pytest.raises(ValueError,match='Unknown'):
        solution_catalog_indices(catalog,dict(source_id=np.array([uid(2)]),family=np.array([0])))
    with pytest.raises(ValueError,match='Invalid'):
        solution_catalog_indices(catalog,dict(source_id=np.array([uid(12)]),family=np.array([0])))
    with pytest.raises(ValueError,match='Invalid'):
        solution_catalog_indices(catalog,dict(source_id=np.array([uid(3),uid(0)]),family=np.array([1,0])))


def test_fresh_projection_of_noncontiguous_component(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_reference_eval as mod
    seam=Seam(np.array([[0,0,-1],[0,0,100]]))
    catalog=dict(source_id=np.array([uid(k) for k in (0,3,7,11)]),family=np.array([0,1,0,1]))
    def at_many(ids,arc):
        return np.column_stack([np.ones(len(ids)),np.zeros(len(ids)),arc]),np.zeros(len(ids))
    paths=SimpleNamespace(catalog=catalog,ids=catalog['source_id'],seam=seam,
        offset=np.array([0,2,4,6,8]),keys={'arc':np.array([0,10,0,20,0,30,0,40])},at_many=at_many)
    monkeypatch.setattr(mod,'PackedPaths',lambda _:paths)
    monkeypatch.setattr(mod,'Surface',lambda *a:SimpleNamespace(x=None,q=None,
        project=lambda xyz:(np.zeros((len(xyz),2)),np.zeros(len(xyz)))))
    monkeypatch.setattr(mod,'winding_at',lambda x,*a:np.ones(len(x)))
    solution=tmp_path/'solution.npz'
    np.savez(solution,source_id=np.array([uid(3),uid(11)]),family=np.array([1,1]),
        winding=np.zeros(2),component=np.zeros(2,dtype=int))
    config=dict(geometry='unused',solution=str(solution),output_root=str(tmp_path/'result'),
        allow_solution_subset=True,spiral=dict(reference='unused',sample_spacing=10,fiber_batch=1))
    result=mod.run(config)
    assert result['fibers']==2
    assert result['spiral']['samples']==8  # lengths20 and40, not the first two catalog entries
    assert result['spiral']['all']['l1']==0
    with np.load(tmp_path/'result/fiber_reference.npz') as saved:
        np.testing.assert_array_equal(saved['source_id'],[uid(3),uid(11)])
        np.testing.assert_allclose(saved['coverage'],1)


def test_continuous_gt_scoring_matches_uuid_sources(monkeypatch):
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_reference_eval as mod
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_irls_benchmark as irls
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_arc_phase as arc
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as layout
    seam=Seam(np.array([[0,0,-1],[0,0,100]]))
    queried=[]
    def at_many(ids,arcs):
        queried.extend(ids.tolist())
        return np.column_stack((np.ones(len(ids)),np.zeros(len(ids)),arcs)),np.zeros(len(ids))
    paths=SimpleNamespace(seam=seam,at_many=at_many)
    monkeypatch.setattr(mod,'arrays',lambda _:dict(state={},widths=np.ones(2)))
    monkeypatch.setattr(layout,'hard_assignment',lambda *a,**k:(None,None,None,np.zeros(1,int)))
    monkeypatch.setattr(arc,'event_arc_points',lambda *a:np.zeros((len(a[-1]),2)))
    monkeypatch.setattr(irls,'metrics',lambda q,t,rows,keep:dict(events=len(rows)))
    event=dict(test_source=uid(11),test_arc=2.,gt_source=uid(40),gt_piece=0,gt_arc=1.)
    solution=dict(source_id=np.array([uid(3),uid(11)]),winding=np.array([2,4]))
    report=mod.continuous_gt_metrics(paths,solution,dict(config='unused/config.json'),{},
        dict(source_id=np.array([uid(40)])),[event,dict(event,test_source=uid(99))])
    assert report['events']==1 and queried==[uid(11)]


def test_sampling_preserves_lengths_and_endpoints():
    owner,arc,weight,offset=sample_arcs(np.array([12.,51.,99.]),20.)
    np.testing.assert_allclose(np.add.reduceat(weight,offset[:-1]),[12,51,99])
    np.testing.assert_allclose(arc[offset[:-1]],0)
    np.testing.assert_allclose(arc[offset[1:]-1],[12,51,99])
    assert np.all(np.bincount(owner)==np.diff(offset))


def test_continuous_winding_across_seam():
    seam=Seam(np.array([[0,0,-1],[0,0,1]]))
    angle=np.linspace(-np.pi/2-.2,3*np.pi/2+.2,101)
    xyz=np.column_stack([np.cos(angle),np.sin(angle),np.zeros(len(angle))])
    q=continuous_winding(seam,xyz,seam.polyline(xyz),np.zeros(len(angle)))
    np.testing.assert_allclose(q-q[0],(angle-angle[0])/(2*np.pi),atol=1e-12)


def test_offset_diagnostic_and_histogram():
    result=error_summary(np.array([9.,10.,11.]))
    assert result['l1']==10
    assert result['signed_median']==10
    assert result['local_centered_l1']==2/3
    assert sum(row['count'] for row in result['histogram'])==3
    assert result['within_percent']['2']==0


def test_per_winding_shift_sign_and_reference_rounding(tmp_path):
    source=tmp_path/'input';source.mkdir()
    ref=np.array([20.1,20.2,20.6,20.9,22.1,22.2,23.1,23.2])
    error=np.array([2.,4.,-1.,-3.,0.,2.,5.,7.])
    np.savez(source/'fiber_reference.npz',reference=ref,fitted=ref+error-5,
        reference_minus_fit=5-error,coverage=np.ones(8),source_id=np.array([uid(k) for k in range(8)]),component=np.zeros(8,int))
    (source/'summary.json').write_text(json.dumps({'spiral':{'all':{'global_offset':5.}}}))
    result=analyze(dict(evaluation_root=str(source),output_root=str(tmp_path/'output')))
    rows={r['winding']:r for r in result['per_reference_winding']['main_component']}
    assert rows[20]['additional_shift']==-3
    assert rows[20]['total_shift']==2
    assert rows[20]['l1_after']==1
    assert rows[21]['additional_shift']==2
    assert (tmp_path/'output/per_winding_main_component.csv').is_file()


def test_cached_median_residual_rebase_for_constant_label_change():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_comparison import grouped_weighted_median
    residual=np.array([.1,.3,1.2,4.,5.]);weights=np.array([1.,2.,1.,1.,1.]);offset=np.array([0,3,5])
    change=np.array([7.,-3.])
    before=grouped_weighted_median(residual,weights,offset)
    after=grouped_weighted_median(residual-np.repeat(change,np.diff(offset)),weights,offset)
    np.testing.assert_allclose(after,before-change,atol=1e-12)
