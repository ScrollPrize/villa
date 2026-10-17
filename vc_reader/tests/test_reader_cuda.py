"""Public CUDA reader, residency and device-to-device pipeline contracts."""
import gc
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest

from vc_reader.reader import ZarrReader
from vc_reader import cuda

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA reader unavailable")

@pytest.fixture
def reader(tmp_path):
    shape=(9,10,11); chunks=(4,5,3)
    (tmp_path/'.zarray').write_text(json.dumps(dict(zarr_format=2,shape=shape,chunks=chunks,dtype='|u1',compressor=None,fill_value=37,order='C',filters=None)))
    rng=np.random.default_rng(7)
    for idx in np.ndindex(3,2,4):
        if idx==(1,1,1): continue
        (tmp_path/'.'.join(map(str,idx))).write_bytes(rng.integers(0,256,chunks,dtype=np.uint8).tobytes())
    with ZarrReader.open(tmp_path,threads=2) as r: yield r

@pytest.mark.parametrize('source',['auto','chunks','cpu'])
@pytest.mark.parametrize('dtype',[np.uint8,np.float32])
def test_boxes(reader,source,dtype):
    regions=[((-2,1,2),(4,6,7)),((7,8,9),(4,5,6)),((20,0,0),(2,3,4)),((0,0,0),(0,2,3))]
    with reader.cuda(cache_bytes=1<<20) as r:
        actual=r.read_many(regions,dtype=dtype,source=source)
        for a,(start,shape) in zip(actual,regions):
            np.testing.assert_array_equal(a.numpy(),reader.read(start,shape,dtype=dtype))
        assert r.read_many([])==[]
        # Results survive cache eviction, closure and owner collection.
        r.clear_cache()
    gc.collect()
    np.testing.assert_array_equal(actual[0].numpy(),reader.read(*regions[0],dtype=dtype))

@pytest.mark.parametrize('source',['chunks','cpu'])
@pytest.mark.parametrize('order',[0,1])
def test_affine_and_points(reader,source,order):
    grids=[((4,5,6),[2.25,-.5,3.75],[[.8,.1,0],[.1,.9,.2],[0,.2,-.8]]),
           ((1,8,9),[8.5,1.25,2.5],np.eye(3)),((0,3,4),[0,0,0],np.eye(3)),
           ((2,3,4),[30,0,0],np.eye(3))]
    pts=np.random.default_rng(2).uniform(-2,12,(1000,3))
    pts=np.concatenate([pts,[[.5,1.5,2.5],[np.nan,0,0],[0,np.inf,0],[1e30,0,0]]])
    with reader.cuda() as r:
        for a,e in zip(r.read_affine_many(grids,order=order,source=source),reader.read_affine_many(grids,order=order)):
            np.testing.assert_allclose(a.numpy(),e,atol=5e-5,rtol=1e-6)
        np.testing.assert_allclose(r.sample(pts,order=order,source=source).numpy(),reader.sample(pts,order=order),atol=5e-5,rtol=1e-6,equal_nan=True)
        local=np.array([[.25,.5,.75],[1,2,3]],np.float32)
        np.testing.assert_allclose(r.sample(local,coordinate_origin=(3,4,5),order=order).numpy(),reader.sample(local,coordinate_origin=(3,4,5),order=order),atol=5e-5,rtol=1e-6)
        assert r.sample(np.empty((0,3))).shape==(0,)
        assert r.sample([1,2,3]).shape==()

@pytest.mark.parametrize('dtype',[np.int32,np.int64])
@pytest.mark.parametrize('source',['chunks','cpu'])
def test_integer(reader,dtype,source):
    bounds=np.iinfo(dtype)
    pts=np.array([[0,0,0],[8,9,10],[4,5,3],[-1,0,0],[9,0,0],[bounds.max,0,0],[bounds.min,0,0]],dtype=dtype)
    with reader.cuda() as r:
        np.testing.assert_array_equal(r.read_points(pts,source=source).numpy(),reader.read_points(pts))
        assert r.read_points(np.empty((0,3),dtype)).shape==(0,)


def test_cache_budget_routing_and_concurrency(reader):
    chunk_bytes=int(np.prod(reader.chunks))
    with reader.cuda(cache_bytes=chunk_bytes) as r:
        a=r.read((0,0,0),reader.chunks,source='chunks')
        assert r.stats['chunk_uploads']==1
        r.read((1,1,1),(2,2,2),source='chunks')
        assert r.stats['chunk_uploads']==1
        r.read((4,0,0),reader.chunks,source='chunks')
        assert r.stats['cache_bytes']==chunk_bytes
        with pytest.raises(ValueError,match='budget'): r.read((0,0,0),reader.shape,source='chunks')
        np.testing.assert_array_equal(r.read((0,0,0),reader.shape).numpy(),reader[:])
        assert r.stats['cpu_routes']==1
        def run(i): return r.read((i,0,0),(1,2,3),dtype=np.float32).numpy()
        with ThreadPoolExecutor(4) as pool:
            for i,out in enumerate(pool.map(run,range(8))): np.testing.assert_array_equal(out,reader.read((i,0,0),(1,2,3),dtype=np.float32))
        np.testing.assert_array_equal(a.numpy(),reader.read((0,0,0),reader.chunks))
        with pytest.raises(TypeError): pickle.dumps(r)
    with pytest.raises(ValueError,match='closed'): r.read((0,0,0),(1,1,1))


def test_device_array(reader):
    host=np.arange(24,dtype=np.uint8).reshape(2,3,4)
    a=cuda.asarray(host)
    np.testing.assert_array_equal(a.numpy(),host)
    b=a.astype(np.float32).reshape(1,2,3,4)
    del a; gc.collect()
    np.testing.assert_array_equal(b.numpy(),host[None])
    with pytest.raises(TypeError): np.asarray(b)
    with pytest.raises(TypeError): pickle.dumps(b)
    with pytest.raises(ValueError): b.reshape(25)
    assert b.__dlpack_device__()==(2,0)
    # Capsule ownership is exercised by importing another DLPack view.
    from vc_reader import _native
    view=_native._cuda_view(b,(24,),0)
    del b; gc.collect()
    out=np.empty(24,np.float32)
    _native._cuda_download(view,out.view(np.uint8))
    np.testing.assert_array_equal(out,host.ravel())


def test_resident_pipeline(reader):
    from vc_reader import augment as A
    from vc_reader.augment.cuda import is_available
    if not is_available(): pytest.skip('CUDA augmentation unavailable')
    with reader.cuda() as r:
        crop=r.read_affine((5,6,7),(1.25,1.5,1.75),np.eye(3)).reshape(1,5,6,7)
    host=crop.numpy()
    transforms=[A.Normalize('minmax'),A.BrightnessMultiply(1.1),A.Flip('x',p_axis=1),A.GaussianNoise(.01)]
    pipeline=A.Pipeline(transforms,backend='cuda')
    expected=pipeline({'image':host},seed=12)['image']
    output=pipeline({'image':crop,'id':'test'},seed=12)
    assert isinstance(output['image'],cuda.DeviceArray)
    assert output['id']=='test'
    np.testing.assert_allclose(output['image'].numpy(),expected,atol=3e-7,rtol=2e-6)
    np.testing.assert_array_equal(crop.numpy(),host)
    plan=pipeline.sample({'image':crop},seed=12)
    replay=pipeline.apply({'image':crop},plan)['image']
    np.testing.assert_array_equal(replay.numpy(),output['image'].numpy())
    inplace=pipeline({'image':crop},seed=12,inplace=True)['image']
    del crop; gc.collect()
    np.testing.assert_array_equal(inplace.numpy(),output['image'].numpy())
    with pytest.raises(ValueError,match='CUDA pipeline'): A.Pipeline([])({'image':output['image']},seed=1)
    invalid=cuda.asarray(np.full((1,2,3,4),np.nan,np.float32))
    with pytest.raises(ValueError,match='nonfinite'): pipeline({'image':invalid},seed=1)
    badmask=cuda.asarray(np.full((1,5,6,7),2,np.uint8))
    with pytest.raises(ValueError,match='zero or one'): pipeline({'image':output['image'],'padding_mask':badmask},seed=1)

@pytest.mark.parametrize('integer',[False,True])
def test_prepared_points(reader,integer):
    points=np.array([[0,1,2],[4,5,3],[8,9,10]],np.int64 if integer else np.float32)
    with reader.cuda(cache_bytes=1<<20) as r:
        request=r.prepare_points(points,integer=integer)
        expected=reader.read_points(points) if integer else reader.sample(points)
        points[:]=-1
        for source in ('chunks','auto','cpu'):
            np.testing.assert_array_equal(request.read(source=source).numpy(),expected)
        before=r.stats['chunk_uploads']
        request.read(source='chunks')
        assert r.stats['chunk_uploads']==before
        r.clear_cache()
        np.testing.assert_array_equal(request.read().numpy(),expected)
        with pytest.raises(TypeError): pickle.dumps(request)
    with pytest.raises(ValueError,match='closed'): request.read()


def test_validation_and_large_origins(reader,tmp_path):
    with reader.cuda() as r:
        for method,args in [(r.sample,([[1,2]],)),(r.read_affine,((2,3,4),[0,0],np.eye(3))),
                            (r.read,((0,0,0),(-1,2,3)))]:
            with pytest.raises(ValueError): method(*args)
        with pytest.raises(TypeError): r.sample([[0,0,0]],dtype=np.float64)
        with pytest.raises(ValueError): r.read((0,0,0),(1,2,3),source='bad')
        with pytest.raises(TypeError): r.read_points([[1.5,2,3]])
        with pytest.raises(ValueError): r.sample([[1,2,3]],coordinate_origin=(-1,0,0))
        with pytest.raises(ValueError): r.read_affine((2,2,2),[0,0,0],np.full((3,3),np.inf))
    from test_sampling_float32 import make_array
    shift=np.array([1_000_000,2_000_000,3_000_000])
    make_array(tmp_path/'large',shift)
    with ZarrReader.open(tmp_path/'large') as large, large.cuda() as gpu:
        local=np.array([[0,.5,1.5],[3.5,4.5,5.5],[4,5,3]],np.float32)
        for order in (0,1):
            np.testing.assert_allclose(gpu.sample(local,coordinate_origin=shift,order=order).numpy(),
                                       large.sample(local,coordinate_origin=shift,order=order),atol=5e-5,rtol=1e-6)
            grid=((4,5,6),[.25,.5,.75],np.diag([.8,.9,1]))
            np.testing.assert_allclose(gpu.read_affine(*grid,coordinate_origin=shift,order=order).numpy(),
                                       large.read_affine(*grid,coordinate_origin=shift,order=order),atol=5e-5,rtol=1e-6)


def test_resident_labels_and_shape_changes(reader):
    from vc_reader import augment as A
    from vc_reader.augment.cuda import is_available
    if not is_available(): pytest.skip('CUDA augmentation unavailable')
    host=np.random.default_rng(9).random((1,5,6,7),dtype=np.float32)
    mask=(host>.5).astype(np.int16)
    valid=np.ones_like(mask,np.uint8); valid[...,0]=0
    transforms=[A.Normalize('zscore'),A.Transpose('zyx'),A.Rotate90(),A.LowResolution(.7)]
    p=A.Pipeline(transforms,backend='cuda')
    sample={'image':host,'mask':mask,'padding_mask':valid}
    expected=p(sample,seed=4)
    got=p({k:cuda.asarray(v) for k,v in sample.items()},seed=4)
    for key in sample:
        np.testing.assert_allclose(got[key].numpy(),expected[key],atol=3e-7,rtol=2e-6)


def test_fork_rejects_device_objects(reader):
    import os
    if not hasattr(os,'fork'): pytest.skip('requires fork')
    r=reader.cuda(); a=r.read((0,0,0),(2,3,4))
    pid=os.fork()
    if pid==0:
        try:
            for call in (lambda:r.read((0,0,0),(1,1,1)),a.numpy):
                try: call()
                except RuntimeError: pass
                else: os._exit(1)
            os._exit(0)
        except BaseException: os._exit(2)
    _,status=os.waitpid(pid,0)
    assert os.waitstatus_to_exitcode(status)==0
    np.testing.assert_array_equal(a.numpy(),reader.read((0,0,0),(2,3,4)))
    r.close()


def test_dlpack_torch_stream_lifetime(reader):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available(): pytest.skip('CUDA Torch unavailable')
    with reader.cuda() as r:
        array=r.read((0,0,0),(5,6,7),dtype=np.float32)
    expected=array.numpy()
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensor=torch.from_dlpack(array)
        result=tensor*2
        del tensor,array
    stream.synchronize()
    np.testing.assert_array_equal(result.cpu().numpy(),expected*2)


def test_cuda_remote_compressed(tmp_path):
    from test_zarr_reader import _write_zarr, _serve, BLOSC
    host=np.random.default_rng(23).integers(0,256,(9,10,11),dtype=np.uint8)
    _write_zarr(str(tmp_path/'a.zarr'),host,(4,5,3),compressor=BLOSC,missing={(1,1,1)})
    server,url=_serve(tmp_path)
    try:
        with ZarrReader.open(url+'/a.zarr',cache_dir=tmp_path/'cache') as reader, reader.cuda() as gpu:
            grid=((7,8,9),(.25,.5,.75),np.eye(3))
            expected=reader.read_affine(*grid)
            np.testing.assert_allclose(gpu.read_affine(*grid,source='chunks').numpy(),expected,atol=5e-5,rtol=1e-6)
            before=gpu.stats['chunk_uploads']
            gpu.read_affine(*grid,source='chunks')
            assert gpu.stats['chunk_uploads']==before
    finally:
        server.shutdown()


@pytest.mark.parametrize('read_mode', ['mmap', 'slab'])
def test_upload_staging_reuse_and_read_failure(tmp_path, read_mode):
    # 82 MiB crosses both 32 MiB staging slots and reuses the first while
    # earlier transfers may still be in flight. A later read error must drain
    # queued uploads before the caller can retry, clear, or close the reader.
    chunks = (128, 128, 128)
    count = 41
    shape = (count * 128 - 3, 128, 128)
    (tmp_path / '.zarray').write_text(json.dumps(dict(
        zarr_format=2, shape=shape, chunks=chunks, dtype='|u1',
        compressor=None, fill_value=37, order='C', filters=None)))
    data = np.random.default_rng(95).integers(0, 256, chunks, dtype=np.uint8)
    for i in range(count):
        if i != 18:  # Missing chunk in the second batch must retain fill semantics.
            (tmp_path / f'{i}.0.0').write_bytes((data ^ np.uint8(i)).tobytes())
    broken = tmp_path / '20.0.0'
    if read_mode == 'slab':
        broken.write_bytes(b'short')
    points = np.array([(i * 128 + z, y, x) for i in range(count)
                       for z, y, x in ((0, 0, 0), (63, 89, 75), (127, 127, 127))], np.int64)
    with ZarrReader.open(tmp_path, threads=4, memory_cache=0, read_mode=read_mode) as reader, reader.cuda(cache_bytes=96 << 20) as gpu:
        if read_mode == 'slab':  # mmap intentionally retains immutable source mappings.
            with pytest.raises(RuntimeError):
                gpu.read_points(points, source='chunks')
            gpu.clear_cache()
            broken.write_bytes((data ^ np.uint8(20)).tobytes())
        expected = reader.read_points(points)
        for subset in (points[:3], points, points[:24], points):
            gpu.clear_cache()
            output = gpu.read_points(subset, source='chunks')
            np.testing.assert_array_equal(output.numpy(), expected[:len(subset)])
            before = gpu.stats['chunk_uploads']
            np.testing.assert_array_equal(gpu.read_points(subset, source='chunks').numpy(), expected[:len(subset)])
            assert gpu.stats['chunk_uploads'] == before
        gpu.clear_cache()
    np.testing.assert_array_equal(output.numpy(), expected)
