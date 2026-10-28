import numpy as np
import json
from vesuvius.neural_tracing.fiber_trace_3d.fiber_selected_solver_comparison import subset_minimal
from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_sgd_full import measured_geometry


def test_minimal_induction_removes_reference_edges():
    d=dict(source_id=np.array([1,2,999]),family=np.array([0,1,0]),start=np.zeros(3),
        end=np.ones(3),node_ds=np.array([[0,1]]*3),i=np.array([0,1]),j=np.array([1,2]),
        ds_a=np.array([.2,.3]),ds_b=np.array([.4,.5]),distance=np.array([.1,.2]),
        sign=np.ones(2),parallel=np.zeros(2,bool))
    out=subset_minimal(d,np.array([True,True,False]))
    np.testing.assert_array_equal(out['source_id'],[1,2])
    np.testing.assert_array_equal(out['i'],[0]);np.testing.assert_array_equal(out['j'],[1])
    assert len(out['distance'])==1 and out['links'].shape==(0,4)


def test_measured_geometry_aligns_signed_arcs():
    class Seam:
        angle=-np.pi/2
        def relative(self,xyz):return xyz[:,:2]
    class Paths:
        seam=Seam()
        def at_many(self,ids,arc):
            # H runs clockwise; V runs upward. Neither uses fitted windings.
            theta=np.where(ids==1,-arc/100,0.)
            xyz=np.column_stack((100*np.cos(theta),100*np.sin(theta),np.where(ids==2,arc,0)))
            return xyz,np.zeros(len(ids))
    d=dict(i=np.array([0]),j=np.array([1]),ds_a=np.array([10.]),ds_b=np.array([20.]))
    out=measured_geometry(d,Paths(),np.array([1,2]),np.array([0,1]),np.array([50.,50.]))
    np.testing.assert_array_equal(out['node_ds'][:,-1],[-50,50])
    np.testing.assert_array_equal(out['ds_a'],[-10]);np.testing.assert_array_equal(out['ds_b'],[20])


def test_graph_only_run_does_not_launch_or_load_sgd(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import (
        fiber_selected_solver_comparison as workflow, fiber_full_slab,
        fiber_implicit_winding, fiber_irls_benchmark)
    base=tmp_path/'base.json';base.write_text('{}')
    root=tmp_path/'output';root.mkdir()
    (root/'prepared.json').write_text('{"complete":true}')
    monkeypatch.setattr(fiber_full_slab,'clean_config',lambda _:dict(
        perpendicular_relative_weight=1.25,perpendicular_offset=.5,lasagna_winding_scale=1))
    def forbidden(_):raise AssertionError('Independent winding SGD must not run')
    monkeypatch.setattr(fiber_implicit_winding,'run',forbidden)
    def compare(config):
        assert 'sgd_solution' not in config
        assert config['exact_lp'] is False
        (root/'comparison').mkdir()
        (root/'comparison/summary.json').write_text('{"results":{}}')
    monkeypatch.setattr(fiber_irls_benchmark,'run',compare)
    result=workflow.run(dict(output_root=str(root),selection_root='selection',
        base_config=str(base),umbilicus='axis.json',gt_root='gt',gpu_manager={},
        solver_source='solver.py',solver_sha256='hash',include_sgd=False,
        exact_lp=False,uv_methods=[]))
    assert result=={'results':{}}
    assert not (root/'solve_config.json').exists()
