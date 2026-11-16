"""Native winding transport/priors preserve measured float64 geometry."""
import numpy as np
import pytest

from vc_object_store import NativeWindingSeam, PackedWindingGeometry, winding_grouped_medians, pack_winding_geometry


def _transport(axis, a, b, angle=-np.pi/2):
    """Small independent scalar oracle, including axis-knot crossings."""
    axis=axis[np.argsort(axis[:,2])]
    cuts=np.array([0.,1.])
    if b[2]!=a[2]:
        t=(axis[:,2]-a[2])/(b[2]-a[2])
        cuts=np.sort(np.r_[cuts,t[(t>0)&(t<1)]])
    points=a[None]+cuts[:,None]*(b-a)
    relative=points[:,:2]-np.column_stack([np.interp(points[:,2],axis[:,2],axis[:,k]) for k in (0,1)])
    ray=np.array([np.cos(angle),np.sin(angle)]);ray[abs(ray)<1e-15]=0
    u=relative@ray;v=ray[0]*relative[:,1]-ray[1]*relative[:,0]
    up=(v[:-1]<0)&(v[1:]>=0);down=(v[:-1]>=0)&(v[1:]<0)
    chosen=up|down
    t=-v[:-1][chosen]/np.diff(v)[chosen]
    return int(np.sum(np.where(up[chosen],1,-1)*(u[:-1][chosen]+t*np.diff(u)[chosen]>0)))


@pytest.mark.parametrize('workers',[1,4])
def test_native_seam_matches_exact_piecewise_axis_transport(workers):
    axis=np.array([[1.,0.,-5.],[-2.,1.,0.],[2.,-1.,4.],[0.,0.,10.]])
    seam=NativeWindingSeam(axis)
    rng=np.random.default_rng(31)
    a=rng.normal(size=(200,3))*[20,20,12]
    b=rng.normal(size=(200,3))*[20,20,12]
    expected=np.array([_transport(axis,x,y) for x,y in zip(a,b)])
    np.testing.assert_array_equal(seam.segments(a,b,workers),expected)
    np.testing.assert_array_equal(seam.segments(b,a,workers),-expected)
    expected_relative=a[:,:2]-np.column_stack([np.interp(a[:,2],axis[:,2],axis[:,k]) for k in (0,1)])
    np.testing.assert_allclose(seam.relative(a,workers),expected_relative,rtol=0,atol=2e-15)


def test_native_seam_rejects_axis_intersection_and_invalid_arrays():
    seam=NativeWindingSeam(np.array([[0.,0.,-5.],[0.,0.,5.]]))
    with pytest.raises(ValueError,match='intersects umbilicus'):
        seam.segments(np.array([[-1.,0.,0.]]),np.array([[1.,0.,0.]]))
    with pytest.raises(ValueError,match='matching XYZ'):
        seam.segments(np.zeros((1,3)),np.zeros((2,3)))


@pytest.mark.parametrize('workers',[1,4])
def test_native_geometry_priors_targets_own_buffers_and_preserve_unsigned_inputs(workers):
    offsets=np.array([0,3,6],np.int64)
    arcs=np.array([0.,10.,20.,0.,2.,4.])
    xyz=np.array([[5.,-np.sqrt(75.),0.],[10.,0.,0.],[5.,np.sqrt(75.),0.],
                  [10.,0.,-2.],[10.,0.,0.],[10.,0.,2.]])
    turns=np.zeros(6,np.int64);axis=np.array([[0.,0.,-10.],[0.,0.,10.]])
    geometry=PackedWindingGeometry(offsets,arcs,xyz,turns,axis,-np.pi/2)
    owner=np.array([0,1],np.int64);family=np.array([0,1],np.int64)
    prior=geometry.geometry_priors(owner,family,8.,workers)
    np.testing.assert_array_equal(prior['directions'],[1.,1.])
    np.testing.assert_allclose(prior['radii'],[10.,10.],rtol=1e-15,atol=0)
    np.testing.assert_array_equal(prior['sample_owner'],[0,0,0,0,1,1])
    np.testing.assert_allclose(prior['sample_arc'],[0,20/3,40/3,20,0,4],rtol=1e-15)
    a=np.array([0],np.int64);b=np.array([1],np.int64);sa=np.array([10.]);sb=np.array([2.])
    saved_a=sa.copy();saved_b=sb.copy()
    result=geometry.fixed_target_columns(owner,np.array([2,2],np.int64),a,b,sa,sb,np.array([-1.,1.]),workers)
    del geometry
    np.testing.assert_array_equal(result['target'],[0]);np.testing.assert_array_equal(result['ds_a'],[-10.])
    np.testing.assert_array_equal(result['ds_b'],[2.]);np.testing.assert_array_equal(sa,saved_a);np.testing.assert_array_equal(sb,saved_b)
    assert not result['target'].flags.writeable
    assert not prior['phase'].flags.writeable


def test_native_grouped_medians_agree_with_numpy_and_validate_values():
    groups=np.array([3,1,3,-1,1,3,-1],np.int64);values=np.array([8.,9.,1.,5.,3.,7.,6.])
    result=winding_grouped_medians(groups,values,4)
    np.testing.assert_array_equal(result['groups'],[-1,1,3])
    np.testing.assert_array_equal(result['medians'],[5.5,6.,7.])
    with pytest.raises(ValueError,match='nonfinite'):
        winding_grouped_medians(np.array([1],np.int64),np.array([np.nan]))


@pytest.mark.parametrize('workers',[1,4])
def test_native_geometry_pack_matches_per_fiber_arc_and_transport(workers):
    rng=np.random.default_rng(19)
    curves=[rng.normal(size=(count,3))*[20.,20.,10.] for count in (2,19,7,40)]
    offsets=np.r_[0,np.cumsum([len(p) for p in curves])].astype(np.int64)
    xyz=np.concatenate(curves);original=xyz.copy()
    axis=np.array([[1.,0.,-5.],[-2.,1.,0.],[2.,-1.,4.],[0.,0.,10.]])
    columns=pack_winding_geometry(offsets,xyz,axis,first_owner=27,workers=workers)
    for owner,points in enumerate(curves):
        sl=slice(offsets[owner],offsets[owner+1])
        expected_arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        expected_turns=np.r_[0,np.cumsum([_transport(axis,a,b) for a,b in zip(points[:-1],points[1:])])]
        np.testing.assert_array_equal(columns['source'][sl],np.full(len(points),27+owner))
        np.testing.assert_array_equal(columns['arc'][sl],expected_arc)
        np.testing.assert_array_equal(columns['turns'][sl],expected_turns)
    np.testing.assert_array_equal(xyz,original)
    assert not columns['arc'].flags.writeable
    with pytest.raises(ValueError,match='at least two'):
        pack_winding_geometry(np.array([0,1,2],np.int64),xyz[:2],axis)


def _brute_force(nodes, a, b, target, weight, span=4):
    """Exact minimum of sum w|k_b-k_a-t| over small integer windings (node 0 fixed)."""
    import itertools
    best=np.inf
    for rest in itertools.product(range(-span,span+1),repeat=nodes-1):
        k=np.r_[0,rest]
        best=min(best,float(np.sum(weight*np.abs(k[b]-k[a]-target))))
    return best


@pytest.mark.parametrize('backend',['network_simplex','cost_scaling'])
def test_native_min_cost_flow_matches_brute_force_and_reports_quantization(backend):
    from vc_object_store import solve_winding_min_cost_flow, winding_min_cost_flow_available
    if not winding_min_cost_flow_available():
        pytest.skip('vc_object_store built without LEMON')
    rng=np.random.default_rng(7)
    for _ in range(12):
        a=rng.integers(0,5,8);b=(a+rng.integers(1,5,8))%5
        target=rng.integers(-2,3,8);weight=rng.integers(1,4,8).astype(float)
        result=solve_winding_min_cost_flow(5,a,b,target,weight,backend=backend,capacity_scale=1)
        expected=_brute_force(5,a,b,target,weight)
        assert result['objective']==expected and result['quantized_objective']==expected
        assert result['backend']==backend and result['winding'].dtype==np.int64
        assert np.all(np.abs(result['flow'])<=result['capacity'])
    # Unsorted arcs, an isolated node, and fractional weights quantized explicitly.
    result=solve_winding_min_cost_flow(4,np.array([2,0,1]),np.array([0,1,2]),np.array([-5,2,3]),np.array([.49,.5,.51]),
                                       backend=backend,capacity_scale=1)
    assert result['capacity'].tolist()==[0,1,1] and result['winding'][3]==0
    q=result['quantization']
    assert q['dropped_arcs']==1 and q['dropped_original_weight']==.49 and q['retained_arcs']==2
    fine=solve_winding_min_cost_flow(4,np.array([2,0,1]),np.array([0,1,2]),np.array([-5,2,3]),np.array([.49,.5,.51]),
                                     backend=backend,capacity_scale=1000)
    assert fine['capacity'].tolist()==[490,500,510] and fine['quantization']['capacity_absolute_error']==0
    k=fine['winding'];assert k[1]-k[0]==2 and k[2]-k[1]==3 and k[0]-k[2]==-5
    with pytest.raises(ValueError):
        solve_winding_min_cost_flow(2,np.array([0]),np.array([3]),np.array([0]),np.array([1.]),backend=backend)
    with pytest.raises(ValueError):
        solve_winding_min_cost_flow(2,np.array([0]),np.array([1]),np.array([0]),np.array([1.]),backend='simplex')


def test_seam_handedness_negates_transport_and_rejects_other_values():
    axis=np.array([[0.,0.,-50.],[0.,0.,50.]])
    start=np.array([[1.,-10.,0.]]);end=np.array([[-1.,-10.,0.]])   # crosses the -Y seam ray once
    forward=NativeWindingSeam(axis,-np.pi/2,1).segments(start,end,1)
    backward=NativeWindingSeam(axis,-np.pi/2,-1).segments(start,end,1)
    assert abs(int(forward[0]))==1 and int(backward[0])==-int(forward[0])
    assert NativeWindingSeam(axis,-np.pi/2,-1).handedness==-1
    with pytest.raises(ValueError):
        NativeWindingSeam(axis,-np.pi/2,0)
    theta=np.linspace(0,4*np.pi,65);xyz=np.column_stack((10*np.cos(theta),10*np.sin(theta),np.zeros(65)))
    offsets=np.array([0,65],np.int64)
    plus=pack_winding_geometry(offsets,xyz,axis,-np.pi/2,1,workers=1)['turns']
    minus=pack_winding_geometry(offsets,xyz,axis,-np.pi/2,-1,workers=1)['turns']
    np.testing.assert_array_equal(np.asarray(minus),-np.asarray(plus))
    assert int(np.asarray(plus)[-1])==2

