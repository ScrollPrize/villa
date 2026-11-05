import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_visualization import segments


A,B='00000000-0000-4000-8000-000000000007','00000000-0000-4000-8000-000000000008'


def test_deformation_curves_reverse_arc_and_source_rates():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_visualization import deformation_geometry
    d=dict(family=np.array([0,1]),source_id=np.array([A,B]),start=np.array([0.,0.]),end=np.array([100.,100.]),
        node_ds=np.array([[0.,50.,100.],[0.,-50.,-100.]]))
    saved=dict(state=np.array([[1.,.5,10.,8.],[1.,.5,30.,8.]]),widths=np.ones(4)*100,
        retained=np.ones(2,dtype=bool),v_offset_source=np.array([A,A,A,B,B]),v_offset_arc=np.array([0.,50.,100.,0.,100.]),
        v_offset_left=np.array([0,1,3]),v_offset_right=np.array([1,2,4]),
        u_offsets=np.array([0.,40.,0.,0.,0.]),v_offsets=np.array([0.,0.,0.,0.,20.]))
    a,b,r=deformation_geometry(d,saved,dict(winding_sigma=.5))
    np.testing.assert_allclose(b[1][0]-a[1][0],[0,40,0])
    np.testing.assert_allclose(b[2][1],[30,-50])
    assert r[A]==(.8,0) and r[B]==(0,.2)


def test_density_segments_and_integer_gauge():
    d=dict(family=np.array([0,1,0]),node_ds=np.array([[0.,50.,100.],[0.,-10.,-20.],[0.,1.,2.]]))
    state=np.array([[1.2,.5,10.,4.],[2.1,.25,30.,4.],[0.,0.,0.,-4.]])
    w=np.array([100.,200.,300.,400.])
    cfg=dict(saved_model='implicit_discrete_winding_v1',winding_sigma=.5)
    a=segments(d,state,w,cfg)
    np.testing.assert_array_equal(a[0],[0,1])
    np.testing.assert_allclose(a[1],[[200.,300.],[375.,375.]])
    np.testing.assert_allclose(a[2],[[10.,10.],[30.,10.]])
    np.testing.assert_allclose(a[3]/a[4],1/(1+np.exp(-4)))
    state[:,0]+=[1,-1,2];state[:,1]-=[1,-1,2]
    b=segments(d,state,w,cfg)
    for i in range(5):np.testing.assert_allclose(a[i],b[i])
