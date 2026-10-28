import numpy as np
import json
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import Seam
from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_reference_eval import sample_arcs,continuous_winding,error_summary,analyze


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
        reference_minus_fit=5-error,coverage=np.ones(8),source_id=np.arange(8),component=np.zeros(8,int))
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
