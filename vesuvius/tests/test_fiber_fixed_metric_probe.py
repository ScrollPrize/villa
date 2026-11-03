import numpy as np
from scipy.optimize import check_grad
from vesuvius.neural_tracing.fiber_trace_3d.fiber_fixed_metric_probe import Equalities, StraightEqualities


def test_equality_jacobian_and_exact_geometry():
    eq=Equalities(4,[0,2],[1,3],[2.,3.],[1],[2])
    x=np.array([[0.,0.],[2.,0.],[2.,0.],[2.,3.]]).ravel()
    np.testing.assert_allclose(eq.fun(x),0,atol=1e-14)
    for i in range(4):
        assert check_grad(lambda z:eq.fun(z)[i],lambda z:eq.jac(z)[i],x)<1e-6


def test_straight_meetings_free_length_and_signed_progress():
    eq=StraightEqualities(4,[0,2],[1,3],[0,1],[1,-1],[1,1],[2,2])
    x=np.array([[0.,0.],[5.,0.],[5.,0.],[5.,-7.]]).ravel()
    np.testing.assert_allclose(eq.fun(x),0,atol=1e-14)
    np.testing.assert_allclose(eq.forward@x,[5.,7.])
    assert np.linalg.matrix_rank(eq.matrix)==len(eq.matrix)
    for i in range(len(eq.matrix)):
        assert check_grad(lambda z:eq.fun(z)[i],lambda z:eq.jac(z)[i],x)<1e-6
    x[-1]=7.
    assert (eq.forward@x)[-1]<0
