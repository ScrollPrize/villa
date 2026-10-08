import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.neural_follow_batch import ForwardQueue, merge_rows
from vesuvius.neural_tracing.fiber_trace_3d.neural_follow_evaluate import family_of


def test_historical_benchmark_preserves_saved_settings(tmp_path,monkeypatch):
    import neural_follow_benchmark as driver
    original=dict(checkpoint='old.pt',heading_checkpoint='heading.pt',output='old',
        native_binary='old-bin',case_workers_per_gpu=16,max_batch_size=8,
        input_workers_per_gpu=24,io_threads_per_gpu=24,precision='bf16',
        policy_overrides={},replay=dict(nominal_step_base=8))
    folder=tmp_path/'flow-full'
    folder.mkdir()
    (folder/'config.json').write_text(json.dumps(original))
    monkeypatch.setattr(driver,'HISTORICAL_ROOT',tmp_path)
    config=driver.historical_config('flow',tmp_path/'new')
    for key in ('checkpoint','heading_checkpoint','case_workers_per_gpu','max_batch_size',
                'input_workers_per_gpu','io_threads_per_gpu','precision','policy_overrides','replay'):
        assert config[key]==original[key]
    assert config['backend']=='historical'
    assert config['forward_length_mode']=='reference_remaining_x3'
    assert config['follower_runtime_commit']==driver.HISTORICAL_COMMIT
    assert json.loads((folder/'config.json').read_text())==original


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
    def forbidden(*args): raise AssertionError('disabled reverse must not run')
    disabled=json.loads(normals.replay(np.array([[8.,16.,16.],[24.,16.,16.]]),trace,1.,2.,0.,2.,1.,3.,
        reverse_runner=forbidden,reverse_interval_base=0.))
    assert disabled==result
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


@pytest.mark.parametrize('mode',['correct','lateral','wrong_way','confidence','duplicate'])
def test_native_reverse_full_prefix(tmp_path,mode):
    normals=normal_fixture(tmp_path)
    line=np.array([[8.,16.,16.],[24.,16.,16.]])
    starts=[]
    reverse_starts=[]
    def forward(start,tangent,remaining,max_steps,observe):
        starts.append(start[0])
        observe(np.array([np.asarray(start)+np.asarray(tangent)*s
            for s in np.arange(1,remaining+1)],np.float64),1)
        return 'done'
    def reverse(start,tangent,remaining,max_steps,observe):
        reverse_starts.append((start[0],remaining))
        if mode=='confidence': return 'confidence'
        if mode=='duplicate':
            observe(np.array([start,start],np.float64),1)
            return 'length'
        direction=np.asarray(tangent) if mode!='wrong_way' else -np.asarray(tangent)
        points=np.array([np.asarray(start)+direction*s for s in np.arange(1,remaining+1)],np.float64)
        if mode=='lateral': points[:,2]+=5
        observe(points,1)
        return 'done'
    result=json.loads(normals.replay(line,forward,1.,2.,0.,2.,1.,3.,
        reverse_runner=reverse,reverse_interval_base=6.,reverse_normal_radius_base=2.,reverse_tangent_radius_base=4.))
    assert result['case']['evaluation_complete']
    assert reverse_starts
    reasons=[s['termination_reason'] for s in result['replay']['segments']]
    if mode=='correct':
        assert reasons==['reference_end']
        assert reverse_starts==[(14.,6.),(20.,12.),(24.,16.)]
    else:
        assert 'reverse_consistency_filtered' in reasons
        assert all(b>a for a,b in zip(starts,starts[1:]))


def test_native_reverse_rollback_and_terminal_error_check(tmp_path):
    normals=normal_fixture(tmp_path)
    line=np.array([[8.,16.,16.],[24.,16.,16.]])
    starts=[]
    saved=[]
    def forward(start,tangent,remaining,max_steps,observe):
        starts.append(start[0])
        observe(np.array([np.asarray(start)+np.asarray(tangent)*s
            for s in np.arange(1,remaining+1)],np.float64),1)
        return 'confidence'
    def reverse(start,tangent,remaining,max_steps,observe):
        saved.append(observe)
        if start[0]==20: return 'confidence'
        observe(np.array([np.asarray(start)+np.asarray(tangent)*s
            for s in np.arange(1,remaining+1)],np.float64),1)
        return 'done'
    result=json.loads(normals.replay(line,forward,1.,2.,0.,2.,1.,3.,
        reverse_runner=reverse,reverse_interval_base=6.))
    segments=result['replay']['segments']
    assert starts==[8.,21.]
    assert segments[0]['termination_reason']=='reverse_consistency_filtered'
    assert segments[0]['end_reference_arc_base']==6.
    assert segments[0]['trace_points_base_xyz'][-1]==[14.,16.,16.]
    assert segments[1]['termination_reason']=='reference_end'
    with pytest.raises(RuntimeError,match='expired'):
        saved[0](np.array([[8.,16.,16.]]),1)

    def bad_forward(start,tangent,remaining,max_steps,observe):
        observe(np.array([np.asarray(start)+[1,0,5]],np.float64),1)
        return 'observer_stop'
    checks=[]
    def short_reverse(*args):
        checks.append(args[0])
        return 'confidence'
    result=json.loads(normals.replay(line,bad_forward,1.,2.,0.,2.,1.,3.,
        reverse_runner=short_reverse,reverse_interval_base=100.))
    assert checks  # GT-error endpoints checked despite not reaching interval.
    assert all(s['termination_reason']=='reverse_consistency_filtered'
        for s in result['replay']['segments'])
