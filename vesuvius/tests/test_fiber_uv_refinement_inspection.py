import numpy as np

from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_refinement_inspection import stats, winding_positions


def test_identity_movement():
    result=stats(np.zeros(20))
    assert all(value==0 for value in result.values())


def test_movement_not_coordinate_rms():
    delta=np.array([[3.,4.],[0.,0.]])
    result=stats(np.linalg.norm(delta,axis=1))
    assert result['maximum']==5
    assert result['mean']==2.5
    assert result['rms']==np.sqrt(12.5)


def test_winding_positions_cross_variable_width_boundary():
    points=np.array([[50.,0.],[100.,0.],[200.,0.],[300.,0.]])
    np.testing.assert_allclose(winding_positions(points,np.array([100.,200.])),[.5,1.,1.5,2.])
