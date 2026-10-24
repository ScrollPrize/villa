"""Independent references and CPU/CUDA agreement for general geometry queries."""
import gc
import os
from pathlib import Path
import sys

import numpy as np
import pytest

if os.environ.get('VC_READER_BUILD_DIR'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from extension_build import load_extensions
    load_extensions(Path(os.environ['VC_READER_BUILD_DIR']))

from vc_reader import cuda, _reader
from vc_reader.geometry import PolylineSet, VoxelSet, transform_points


def host(x):
    if isinstance(x, dict):
        return {k: host(v) for k,v in x.items()}
    return x.numpy() if isinstance(x, cuda.DeviceArray) else x


@pytest.fixture(params=['cpu','cuda'])
def backend(request):
    if request.param == 'cuda' and not cuda.is_available():
        pytest.skip('CUDA build/device required')
    return request.param


def upload(x, backend):
    x = np.asarray(x)
    return cuda.asarray(x) if backend == 'cuda' else x


def test_sorted_voxel_keys(backend):
    p = np.array([[1, 2, 3], [1, 2, 3], [0, 0, 0], [np.nan, 0, 0]], np.float32)
    voxels = VoxelSet(upload(p, backend), (3, 4, 5))
    result = host(voxels.sorted_keys(capacity=8))
    expected = [0, 33, 33] if backend == 'cuda' else [0, 33]
    np.testing.assert_array_equal(result, expected + [np.iinfo(np.int64).max]*(8-len(expected)))
    assert np.all(result[:-1] <= result[1:])
    with pytest.raises(ValueError, match='capacity'):
        voxels.sorted_keys(capacity=1)
    empty = VoxelSet(upload(np.empty((0, 3), np.float32), backend), (3, 4, 5))
    assert host(empty.sorted_keys()).shape == (0,)
    np.testing.assert_array_equal(host(empty.sorted_keys(capacity=3)), [np.iinfo(np.int64).max]*3)


def curves(points, offsets, backend, **kw):
    c = PolylineSet(points, offsets, **kw)
    return c.cuda() if backend == 'cuda' else c


def assert_results(a,b,atol=1e-5):
    for key,x in host(a).items():
        if x.dtype.kind == 'f':
            np.testing.assert_allclose(x,host(b)[key],atol=atol,rtol=2e-6,equal_nan=True)
        else:
            np.testing.assert_array_equal(x,host(b)[key])


def test_interpolate_endpoints_duplicates_and_masks(backend):
    p = np.array([[0,0,0],[1,0,0],[1,0,0],[1,3,0],[9,8,7]],np.float32)
    c = curves(p,[0,4,5],backend)
    ids = np.array([0,0,0,0,0,0,1,-1,2,0],np.int64)
    s = np.array([-1,0,.5,1,2.5,8,0,0,0,np.nan],np.float32)
    result = host(c.interpolate(upload(ids,backend),upload(s,backend)))
    np.testing.assert_allclose(result['points'][:7],[[0,0,0],[0,0,0],[.5,0,0],[1,0,0],[1,1.5,0],[1,3,0],[9,8,7]])
    np.testing.assert_array_equal(result['valid'],[False,True,True,True,True,False,True,False,False,False])
    assert np.isnan(result['points'][7:]).all()


def test_custom_arcs_and_empty(backend):
    c = curves([[0,0,0],[4,0,0]],[0,2],backend,arclengths=[10,12])
    r = host(c.interpolate(upload(np.array([0],np.int64),backend),upload(np.array([11],np.float32),backend)))
    np.testing.assert_array_equal(r['points'],[[2,0,0]])
    empty = curves(np.empty((0,3),np.float32),[0],backend)
    r = host(empty.nearest(upload(np.zeros((2,3),np.float32),backend)))
    assert not r['valid'].any()
    assert np.isinf(r['distances']).all()
    assert c.interpolate(upload(np.empty(0,np.int64),backend),upload(np.empty(0,np.float32),backend))['points'].shape == (0,3)


def test_plane_first_crossing_direction_window(backend):
    p = np.array([[0,0,0],[1,0,2],[2,0,0],[3,0,2]],np.float32)
    c = curves(p,[0,4],backend,arclengths=[0,1,2,3])
    ids = upload(np.zeros(5,np.int64),backend)
    planes = upload(np.tile(np.array([0,0,1,1],np.float32),(5,1)),backend)
    intervals = upload(np.array([[0,3],[1,3],[3,0],[.5,1],[1,2]],np.float32),backend)
    pos = host(c.intersect_planes(ids,planes,intervals=intervals,direction=1))
    np.testing.assert_array_equal(pos['indices'],[0,2,1,-1,-1])
    np.testing.assert_allclose(pos['arclengths'][:3],[.5,2.5,1.5])
    neg = host(c.intersect_planes(ids,planes,intervals=intervals,direction=-1))
    np.testing.assert_array_equal(neg['indices'],[1,1,2,-1,1])
    either = host(c.intersect_planes(ids,planes,intervals=intervals,direction=0))
    np.testing.assert_allclose(either['arclengths'][:3],[.5,1.5,2.5])
    assert not either['valid'][3]


def brute_nearest(p, offsets, q, ids, vertices):
    out = []
    for x,curve in zip(q,ids):
        candidates = []
        for c,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
            if curve >= 0 and c != curve:
                continue
            for i in range(lo,hi if vertices or hi-lo==1 else hi-1):
                a,b = p[i],p[min(i+1,hi-1)]
                d = b-a
                t = 0. if vertices or d@d==0 else np.clip((x-a)@d/(d@d),0,1)
                point = a+t*d
                candidates.append((float(np.sum((x-point)**2)),i,c,point))
        out.append(min(candidates,key=lambda c:c[:2]))
    return out


@pytest.mark.parametrize('vertices',[False,True])
def test_indexed_nearest_matches_brute_force(backend,vertices):
    rng = np.random.default_rng(741)
    p = np.cumsum(rng.normal(size=(151,3)),axis=0).astype(np.float32)
    p[21] = p[20]
    offsets = [0,49,150,151]
    q = rng.uniform(-12,12,(81,3)).astype(np.float32)
    ids = rng.integers(-1,3,len(q),dtype=np.int64)
    c = curves(p,offsets,backend)
    r = host(c.nearest(upload(q,backend),curve_ids=upload(ids,backend),vertices=vertices))
    ref = brute_nearest(p,offsets,q,ids,vertices)
    np.testing.assert_array_equal(r['indices'],[v[1] for v in ref])
    np.testing.assert_array_equal(r['curve_ids'],[v[2] for v in ref])
    np.testing.assert_allclose(r['points'],[v[3] for v in ref],atol=3e-6)
    np.testing.assert_allclose(r['distances'],np.sqrt([v[0] for v in ref]),rtol=2e-6)
    assert r['valid'].all()
    cpu = PolylineSet(p,offsets).nearest(q,curve_ids=ids,vertices=vertices)
    assert_results(r,cpu)


def test_nearest_ties_limits_and_invalid(backend):
    c = curves([[-1,0,0],[1,0,0],[-1,0,0],[1,0,0]],[0,2,4],backend)
    q = upload(np.array([[0,1,0],[0,2,0],[np.nan,0,0]],np.float32),backend)
    r = host(c.nearest(q,max_distance=1))
    np.testing.assert_array_equal(r['valid'],[True,False,False])
    assert r['indices'][0] == 0
    assert r['distances'][0] == 1
    assert not host(c.nearest(q,curve_ids=upload(np.array([3,-2,0],np.int64),backend)))['valid'].any()


def test_boxes_candidates_capacity_and_invalid(backend):
    p = np.array([[i,0,0] for i in range(30)],np.float32)
    c = curves(p,[0,15,30],backend)
    lo = np.array([[1,-1,-1],[13,-1,-1],[3,0,0],[np.nan,0,0]],np.float32)
    hi = np.array([[12,1,1],[17,1,1],[2,1,1],[1,1,1]],np.float32)
    r = host(c.query_boxes(upload(lo,backend),upload(hi,backend),capacity=3))
    np.testing.assert_array_equal(r['counts'],[13,5,0,0])
    np.testing.assert_array_equal(r['valid'],[True,True,False,False])
    assert len(set(r['indices'][0]))==3
    assert_results(r,PolylineSet(p,[0,15,30]).query_boxes(lo,hi,capacity=3))
    full = host(c.query_boxes(upload(lo[:2],backend),upload(hi[:2],backend),capacity=32,vertices=True))
    np.testing.assert_array_equal(np.sort(full['indices'][0,:12]),np.arange(1,13))
    assert full['counts'][0]==12


def test_transforms(backend):
    p = np.array([[1,2,3],[2,3,4]],np.float32)
    mat = np.array([[0,-1,0,10],[1,0,0,20],[0,0,2,30]],np.float32)
    actual = host(transform_points(upload(p,backend),upload(mat,backend)))
    np.testing.assert_array_equal(actual,p@mat[:,:3].T+mat[:,3])
    actual = host(transform_points(upload(p,backend),upload(np.stack([mat,mat*2]),backend),vectors=True))
    np.testing.assert_array_equal(actual,np.stack([p[0]@mat[:,:3].T,p[1]@mat[:,:3].T*2]))


def test_sparse_voxels_rounding_batches_and_dense(backend):
    p = np.array([[0,0,0],[0,0,0],[1.5,2.5,3.5],[-.5,0,0],[-.51,0,0],[3.5,0,0],[np.nan,0,0],[2,2,2]],np.float32)
    ids = np.array([0,1,1,0,0,0,0,2],np.int64)
    v = VoxelSet(upload(p,backend),(4,4,5),batch_ids=upload(ids,backend),batches=2)
    dense = host(v.to_dense())
    expected = np.zeros((2,4,4,5),np.uint8)
    expected[0,0,0,0]=expected[1,0,0,0]=expected[1,2,2,4]=1
    np.testing.assert_array_equal(dense,expected)
    r = host(v.contains(upload(p,backend),batch_ids=upload(ids,backend)))
    np.testing.assert_array_equal(r,[True,True,True,True,False,False,False,False])
    if backend=='cuda':
        other = VoxelSet(p,(4,4,5),batch_ids=ids,batches=2).cuda()
        np.testing.assert_array_equal(host(other.to_dense()),expected)
    assert v.nbytes <= len(p)*8
    empty = VoxelSet(upload(np.empty((0,3),np.float32),backend),(2,3,4))
    assert not host(empty.to_dense()).any()


def test_output_validation_and_input_ownership():
    p = np.array([[0,0,0],[2,0,0]],np.float32)
    c = PolylineSet(p,[0,2]);p[:]=100
    ids,s = np.array([0],np.int64),np.array([1],np.float32)
    out = dict(points=np.empty((1,3),np.float32),valid=np.empty(1,bool))
    assert c.interpolate(ids,s,out=out) is out
    np.testing.assert_array_equal(out['points'],[[1,0,0]])
    out['points']=np.empty((1,6),np.float32)[:,::2]
    with pytest.raises(ValueError): c.interpolate(ids,s,out=out)
    with pytest.raises(ValueError,match='overlap'):
        c.interpolate(ids,s,out=dict(points=np.broadcast_to(s,(1,3)).copy(),valid=s.view(bool)[:1]))
    with pytest.raises(ValueError): PolylineSet([[0,0,0]],[0,0,1])
    with pytest.raises(TypeError): PolylineSet([[0,0,0]],[0.,1.])
    with pytest.raises(ValueError): PolylineSet([[0,0,0],[1,0,0]],[0,2],arclengths=[1,0])
    with pytest.raises(ValueError): VoxelSet(np.zeros((1,3),np.float32),(2**24+1,2,2))
    with pytest.raises(ValueError): VoxelSet(np.zeros((1,3),np.float32),(2**24,2**24,2**24))
    with pytest.raises(ValueError): c.nearest(np.zeros((1,3),np.float32),max_distance=-1)
    with pytest.raises(ValueError): transform_points(p,np.eye(3,dtype=np.float32))
    if not cuda.is_available():
        with pytest.raises(RuntimeError,match='CUDA geometry'): c.cuda()
        with pytest.raises(RuntimeError,match='CUDA geometry'): VoxelSet(p,(4,4,4)).cuda()


def test_cuda_torch_owned_outputs_and_other_stream():
    torch = pytest.importorskip('torch')
    if not cuda.is_available() or not torch.cuda.is_available():
        pytest.skip('CUDA required')
    c = PolylineSet([[0,0,0],[2,0,0]],[0,2]).cuda()
    with pytest.raises(TypeError,match='upload explicitly'):
        c.nearest(np.zeros((1,3),np.float32))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ids=torch.zeros(64,device='cuda',dtype=torch.int64)
        s=torch.ones(64,device='cuda')
        out=dict(points=torch.empty((64,3),device='cuda'),valid=torch.empty(64,device='cuda',dtype=torch.bool))
        stream.synchronize()
        _reader._set_stream(stream.cuda_stream)
        try:
            assert c.interpolate(ids,s,out=out) is out
            value=out['points'].clone()
            bad=dict(out,points=torch.empty((64,6),device='cuda')[:,::2])
            with pytest.raises((TypeError,RuntimeError)): c.interpolate(ids,s,out=bad)
        finally:
            stream.synchronize();_reader._set_stream(0)
    np.testing.assert_array_equal(value.cpu(),np.tile([1,0,0],(64,1)))
    # Neither query validation nor destruction should drain an unrelated stream.
    done=torch.cuda.Event()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(500_000_000);done.record()
    try:
        c.interpolate(ids,s,out=out)
        vox=VoxelSet(cuda.asarray(np.array([[0,0,0]],np.float32)),(4,4,4))
        assert vox.contains(cuda.asarray(np.array([[0,0,0]],np.float32))).shape == (1,)
        keys = torch.empty(8, device='cuda', dtype=torch.int64)
        assert vox.sorted_keys(capacity=8, out=keys) is keys
        del vox;gc.collect()
        assert not done.query()
    finally: stream.synchronize()
    np.testing.assert_array_equal(keys.cpu(), [0]+[np.iinfo(np.int64).max]*7)


def test_cuda_wrap_does_not_fence_default_stream():
    torch = pytest.importorskip('torch')
    if not cuda.is_available() or not torch.cuda.is_available():
        pytest.skip('CUDA required')
    producer = torch.cuda.Stream()
    done = torch.cuda.Event()
    torch.cuda.default_stream().synchronize()
    try:
        with torch.cuda.stream(producer):
            tensor = torch.empty(16, device='cuda')
            torch.cuda._sleep(500_000_000)
            done.record()
            _reader._set_stream(producer.cuda_stream)
            wrapped = cuda.wrap(tensor)
            assert wrapped.shape == (16,)
        # Importing storage must not make the unrelated default stream wait
        # for the producer. The consumer is the producer stream itself.
        torch.cuda.default_stream().synchronize()
        assert not done.query()
    finally:
        producer.synchronize()
        _reader._set_stream(0)


def test_many_curves_and_zero_length_segments(backend):
    count=513
    centers=np.stack([np.zeros(count),np.arange(count)*10,np.zeros(count)],axis=1).astype(np.float32)
    p=np.stack([centers-1,centers,centers,centers+1],axis=1).reshape(-1,3)
    c=curves(p,np.arange(count+1)*4,backend)
    q=centers+np.array([.25,0,-.25],np.float32)
    r=host(c.nearest(upload(q,backend)))
    np.testing.assert_array_equal(r['curve_ids'],np.arange(count))
    np.testing.assert_array_equal(r['indices'],np.arange(count)*4)
    np.testing.assert_allclose(r['points'],centers,atol=1e-5)
    np.testing.assert_allclose(r['distances'],np.sqrt(.125),rtol=1e-6)


def test_plane_scaled_normals_and_degenerate_curves(backend):
    c=curves([[0,0,0],[0,0,0],[0,0,2]],[0,3],backend)
    scales=np.array([1e-30,1,1e30,0],np.float32)
    planes=np.zeros((4,4),np.float32);planes[:,2]=scales;planes[:,3]=scales
    r=host(c.intersect_planes(upload(np.zeros(4,np.int64),backend),upload(planes,backend)))
    np.testing.assert_array_equal(r['valid'],[True,True,True,False])
    np.testing.assert_allclose(r['points'][:3],np.tile([0,0,1],(3,1)))
    np.testing.assert_array_equal(r['indices'][:3],[1,1,1])


def test_plane_crossings_match_independent_reference(backend):
    rng=np.random.default_rng(352)
    p=np.cumsum(rng.normal(size=(83,3)),axis=0).astype(np.float32)
    arcs=np.r_[np.float32(0),np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))].astype(np.float32)
    planes=rng.normal(size=(200,4)).astype(np.float32)
    c=curves(p,[0,len(p)],backend,arclengths=arcs)
    r=host(c.intersect_planes(upload(np.zeros(len(planes),np.int64),backend),upload(planes,backend)))
    for j,plane in enumerate(planes):
        # Float64 reference provides a separate arithmetic path for correctness.
        signed=p.astype(np.float64)@plane[:3].astype(np.float64)-float(plane[3])
        hits=np.flatnonzero((signed[:-1]<0)&(signed[1:]>=0))
        assert r['valid'][j]==bool(len(hits))
        if len(hits):
            i=hits[0];t=signed[i]/(signed[i]-signed[i+1])
            np.testing.assert_allclose(r['points'][j],p[i]*(1-t)+p[i+1]*t,atol=2e-5)
            assert r['indices'][j]==i


def test_float32_arclength_overflow_rejected():
    with pytest.raises(ValueError,match='finite segment spans'):
        PolylineSet([[0,0,0],[1,0,0]],[0,2],arclengths=[-3e38,3e38])


def test_cpu_index_concurrent_readers():
    from concurrent.futures import ThreadPoolExecutor
    c=PolylineSet([[0,0,0],[1,0,0],[1,1,0]],[0,3])
    q=np.random.default_rng(72).normal(size=(1024,3)).astype(np.float32)
    expected=c.nearest(q)
    with ThreadPoolExecutor(4) as pool:
        for result in pool.map(lambda _:c.nearest(q),range(12)):
            assert_results(result,expected)


def test_gather_candidates_without_host_roundtrip(backend):
    c=curves([[0,0,0],[1,0,0],[1,2,0],[5,5,5]],[0,3,4],backend)
    ids=upload(np.array([0,1,2,3,-1,4],np.int64),backend)
    vertices=host(c.gather(ids))
    np.testing.assert_array_equal(vertices['valid'],[True,True,True,True,False,False])
    np.testing.assert_array_equal(vertices['arclengths'][:4],[0,1,3,0])
    edges=host(c.gather(ids,segments=True))
    np.testing.assert_array_equal(edges['valid'],[True,True,False,True,False,False])
    np.testing.assert_array_equal(edges['points'][1],[[1,0,0],[1,2,0]])
    np.testing.assert_array_equal(edges['points'][3],[[5,5,5],[5,5,5]])
    candidates=c.query_boxes(upload(np.array([[-1,-1,-1]],np.float32),backend),
        upload(np.array([[2,3,1]],np.float32),backend),capacity=8)
    gathered=host(c.gather(candidates['indices'].reshape(-1),segments=True))
    assert gathered['valid'].sum()==2
    np.testing.assert_array_equal(gathered['curve_ids'][:2],[0,0])
