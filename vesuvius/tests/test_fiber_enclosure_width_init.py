import numpy as np
import pytest


def fixture():
    d=dict(i=np.array([0,0]),j=np.array([1,1]),parallel=np.array([False,False]),
           sign=np.array([1.,-1.]),family=np.array([0,1]),ds_a=np.array([0.,100.]),ds_b=np.zeros(2))
    samples=dict(owner=np.array([0,0,1,1]),ds=np.array([0.,100.,0.,100.]),phase=np.array([0.,1.,0.,0.]))
    state=np.array([[20.,0.,0.,100.],[5.,0.,0.,100.]])
    return d,samples,state


def test_enclosure_width_initializer_recovers_physical_turn_without_reference():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_enclosure_width_init import fit_enclosure_widths
    d,s,state=fixture()
    d['ds_a']*=10;s['ds']*=10
    d.update(source_id=np.array([101,102]),phase_a=np.array([0.,1.]),phase_b=np.zeros(2))
    widths,anchors,qv,brackets=fit_enclosure_widths(d,s,np.array([2,2]),np.full(6,1500.),1,np.arange(1,5))
    assert widths[2]==pytest.approx(1000.,abs=1e-7)
    assert qv[1]==2
    assert brackets[0]['turn_arc_vx']==1000
    assert anchors[0]==pytest.approx(widths[1],abs=1e-7)


def test_enclosure_width_initializer_requires_geometric_enclosure():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_enclosure_width_init import fit_enclosure_widths
    d,s,state=fixture();d['sign'][:]=1
    d.update(source_id=np.array([101,102]),phase_a=np.array([0.,1.]),phase_b=np.zeros(2))
    with pytest.raises(ValueError,match='bracket'):
        fit_enclosure_widths(d,s,np.array([2,2]),np.full(6,1500.),1,np.arange(1,5))
