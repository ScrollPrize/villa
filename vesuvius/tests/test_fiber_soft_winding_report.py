import numpy as np
import json
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_soft_winding_report import bin_mask, stats


def test_partition_and_boundaries():
    x = np.r_[np.arange(-10,11)/10, -1.01, 1.01]
    masks = np.array([bin_mask(x,b) for b in range(20)])
    np.testing.assert_array_equal(masks.sum(0), np.r_[np.ones(21),0,0])
    assert bin_mask(np.array([-.1,0.,.1]),10).tolist()==[False,True,False]


def test_stats_signed_vs_absolute():
    s=stats(np.array([-2.,4.]))
    assert s['mean']==1 and s['l1']==3 and s['max_abs']==4
    assert stats([]) is None


def test_saved_report_orientation_gauge_and_losses(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_soft_winding_report import run
    root=tmp_path/'solve';root.mkdir()
    identity=dict(source_id=np.array([1,2]),start=np.zeros(2),end=np.ones(2),family=np.array([0,1]))
    d=dict(identity,i=np.array([0,1]),j=np.array([1,0]),ds_a=np.zeros(2),ds_b=np.zeros(2),
           sign=np.array([1.,-1.]),distance=np.array([.5,.5]),parallel=np.zeros(2,dtype=bool))
    np.savez(root/'tensors.npz',**d)
    cfg=dict(tensor_cache=str(root/'tensors.npz'))
    (root/'config.json').write_text(json.dumps(cfg))
    (root/'summary.json').write_text(json.dumps(dict(model='continuous_soft_winding_v1')))
    state=np.array([[0.,.2,0.,2.],[0.,.23,10.,2.]])
    def report():
        np.savez(root/'solution.npz',state=state,widths=np.array([1000.,1200.]),**identity)
        return run(dict(solve_root=str(root),output_root=str(tmp_path/'report')))
    first=report()['tables']['fitted_aligned']
    total=first['total'];p=1/(1+np.exp(-2.))
    assert total['count']==2
    assert total['u_windings']['mean']==pytest.approx(.03)
    assert total['v_base_voxels']['mean']==pytest.approx(10.)
    assert total['losses']['u']['full_objective_contribution']==pytest.approx(p*p*.03**2)
    assert total['losses']['v']['full_objective_contribution']==pytest.approx(p*p*.02*(10/1000)**2)
    assert total['losses']['winding']['full_objective_contribution']==pytest.approx(p*p*.03**2)
    assert sum(b['count'] for b in first['bins'])==2
    state[1,0]+=2;state[1,1]-=2
    second=report()['tables']['fitted_aligned']
    for key in ('u_windings','v_base_voxels','full_u_separation_windings'):
        assert second['total'][key]==pytest.approx(total[key],abs=1e-12)
    for a,b in zip(first['bins'],second['bins']): assert a['count']==b['count']
