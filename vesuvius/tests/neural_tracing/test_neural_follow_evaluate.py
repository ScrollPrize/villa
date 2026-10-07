import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.neural_follow_batch import ForwardQueue, merge_rows
from vesuvius.neural_tracing.fiber_trace_3d.neural_follow_evaluate import family_of


def test_family_manual_precedes_automatic():
    assert family_of({'hv_classification':{'manual_tag':'V','automatic_tag':'H'}})=='V'
    with pytest.raises(ValueError): family_of({})


def test_injected_ct_reader_and_grid_scale():
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume, FiberVolumeSpec
    array=SimpleNamespace(dtype=np.dtype('uint8'),shape=(100,80,60))
    levels=[]
    def factory(level):
        levels.append(level)
        return array
    volume=FiberVolume(FiberVolumeSpec(ct_zarr='/not/opened',ct_level=2,ct_grid_scale=4.,grid_scale=8.),array_factory=factory)
    assert levels==[2] and volume.ct is array and volume.input_scale==2
    assert volume.shape==(50,40,30)
    base=np.array([17868.9,10896.8,47583.0])
    np.testing.assert_allclose(base/8*volume.input_scale,base/4)


def test_commit_observer_receives_only_committed_path(monkeypatch):
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_follow.tracing import trace as module
    monkeypatch.setattr(module,'ct_frame',lambda *a,**kw:np.eye(3))
    tracer=module.ModelTracer.__new__(module.ModelTracer)
    tracer.model=SimpleNamespace(cfg=SimpleNamespace(n_future=1,gate_plane=None,max_recovery_distance=6.))
    tracer.vol=SimpleNamespace(shape=(128,128,128))
    tracer.n_history=4
    tracer.p=module.TraceParams(n_commit=1,confidence=.5,gate='full',max_len=10,seed=0)
    tracer.device='cpu'
    tracer.build_inputs=lambda *args:{}
    tracer.forward=lambda *args:dict(points=torch.tensor([[[0.,0.,1.]]]),confidence=torch.ones(1,1))
    observed=[]
    def stop(index,path):
        observed.append(path.copy())
        return True
    paths,reasons=tracer._trace(np.array([[10.,10.,10.]]),np.array([[0.,0.,1.]]),None,on_commit=stop)
    assert reasons==['observer_stop'] and len(observed)==1
    np.testing.assert_array_equal(paths[0],[[10,10,10],[10,10,11]])
    np.testing.assert_array_equal(observed[0],paths[0])


def test_forward_queue_keeps_rows_and_propagates_errors():
    def forward(x,h,m,s): return {'points':x['v']+h, 'memory_entry':m*2}
    queue=ForwardQueue(forward,4)
    try:
        with ThreadPoolExecutor(8) as pool:
            def call(i):
                t=torch.tensor([[float(i)]])
                return queue({'v':t},t,t,{})
            results=list(pool.map(call,range(32)))
        for i,result in enumerate(results):
            assert result['points'].item()==i*2
            assert result['memory_entry'].item()==i*2
        assert sum(queue.batch_sizes)==32
    finally: queue.close()
    broken=ForwardQueue(lambda *args: (_ for _ in ()).throw(ValueError('bad')))
    try:
        with pytest.raises(ValueError,match='bad'): broken({},torch.ones(1,1),torch.ones(1,1),{})
    finally: broken.close()


def test_merge_rejects_policy_mixing():
    with pytest.raises(ValueError): merge_rows([{'confidence':.4},{'confidence':.5}])


def normal_fixture(tmp_path):
    import zarr
    groups={}
    for channel,value in [('nx',128),('ny',128),('grad_mag',255)]:
        path=tmp_path/f'{channel}.zarr'
        array=zarr.open_array(str(path),mode='w',shape=(32,32,32),chunks=(8,8,8),dtype='u1',
                             zarr_format=2,compressor=None)
        array[:]=value
        groups[channel]={'zarr':path.name,'scaledown':0,'channels':[channel]}
    path=tmp_path/'normal.json'
    path.write_text(json.dumps(dict(version=2,source_to_base=1,grad_mag_encode_scale=255.,
        grad_mag_factor=1.,base_shape_zyx=[32,32,32],groups=groups)))
    from vesuvius.neural_tracing.fiber_trace_3d.neural_follow_native import load_native
    NormalSampler=load_native().NormalSampler
    return NormalSampler(str(path))


def test_native_callback_replay_success_and_expired_observer(tmp_path):
    normals=normal_fixture(tmp_path)
    escaped=[]
    def trace(start,tangent,remaining,max_steps,observe):
        escaped.append(observe)
        points=np.array([np.asarray(start)+np.asarray(tangent)*s for s in np.arange(1,remaining+2)],np.float64)
        observe(points,1)
        return 'done'
    result=json.loads(normals.replay(np.array([[8.,16.,16.],[24.,16.,16.]]),trace,1.,2.,0.,2.,1.,3.))
    assert result['case']['evaluation_complete']
    assert not result['case']['failures']
    with pytest.raises(RuntimeError,match='expired'):
        escaped[0](np.array([[1.,2.,3.]]),1)


def test_native_callback_stops_inside_multi_point_commit_and_restarts(tmp_path):
    normals=normal_fixture(tmp_path)
    starts=[]
    def trace(start,tangent,remaining,max_steps,observe):
        starts.append(start)
        assert not observe(np.array([np.array(start)+[0,0,5], [10000,10000,10000]],np.float64),1)
        return 'abort'
    result=json.loads(normals.replay(np.array([[8.,16.,16.],[24.,16.,16.]]),trace,1.,2.,0.,2.,1.,3.))
    assert result['case']['evaluation_complete']
    assert len(starts)>1
    assert result['case']['failures']
    assert '10000' not in json.dumps(result)
    assert all(b[0]>a[0] for a,b in zip(starts,starts[1:]))


def test_native_callback_validation_and_exception(tmp_path):
    normals=normal_fixture(tmp_path)
    line=np.array([[8.,16.,16.],[24.,16.,16.]])
    with pytest.raises(ValueError): normals.replay(line,lambda *args:'',0.)
    def fail(*args): raise ValueError('callback failed')
    with pytest.raises(ValueError,match='callback failed'): normals.replay(line,fail,1.)


def test_native_immediate_confidence_stop_advances_restart(tmp_path):
    normals=normal_fixture(tmp_path)
    starts=[]
    def trace(start,*args):
        starts.append(start)
        return 'confidence'
    result=json.loads(normals.replay(np.array([[8.,16.,16.],[24.,16.,16.]]),trace,1.,2.,0.,2.,1.,3.))
    assert result['case']['evaluation_complete']
    assert result['case']['failures']
    assert all(b[0]>a[0] for a,b in zip(starts,starts[1:]))


def test_native_reader_preserves_bytes_and_fill(tmp_path):
    import zarr
    from vesuvius.neural_tracing.fiber_trace_3d.neural_follow_native import load_native
    from vesuvius.neural_tracing.fiber_trace_3d.neural_follow_reader import NativeArray
    path=tmp_path/'ct.zarr'
    array=zarr.open_array(str(path/'0'),mode='w',shape=(8,8,8),chunks=(4,4,4),dtype='u1',
                         zarr_format=2,compressor=None,fill_value=17)
    values=np.arange(64,dtype=np.uint8).reshape(4,4,4)
    array[:4,:4,:4]=values
    native=NativeArray(load_native().Volume.open(str(path)),0)
    assert np.array_equal(native.read((0,0,0),(8,8,8)),array[:])
    padded=native.read((-1,-1,-1),(6,6,6))
    assert np.all(padded[0]==17)
    assert np.array_equal(padded[1:5,1:5,1:5],values)
