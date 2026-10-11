import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import reference_initial_centers,validate_joint_rejection


def test_reference_centers_correct_anchor_phase_and_arc():
    d=dict(phase=np.array([.8,.2,.2]),family=np.array([0,1,0]),
           node_ds=np.array([[0,40,80],[0,40,80],[0,40,80]]))
    centers=reference_initial_centers(d,np.full(20,100.),np.array([10.,10.,np.nan]),np.array([3.,4.,5.]))
    np.testing.assert_equal(centers,[9,10,5])
    physical=centers+d['phase']+(d['family']==0)*d['node_ds'][:,1]/100
    np.testing.assert_equal(np.floor(physical[:2]+.5),[10,10])


def test_invalid_reference_and_conflicting_checkpoint_fail():
    d=dict(phase=np.zeros(1),family=np.zeros(1),node_ds=np.zeros((1,3)))
    with pytest.raises(ValueError):reference_initial_centers(d,np.ones(20),np.array([21.]),np.zeros(1))
    with pytest.raises(ValueError,match='mutually exclusive'):
        validate_joint_rejection({'reference_initialization':'x','warm_start':'y'})
