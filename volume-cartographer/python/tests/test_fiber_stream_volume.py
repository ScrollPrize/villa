"""Native continuous observer and seed API contracts."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from test_fiber_chunk_trace import inputs, handles, chunk_config, trace_config
from vc import fiber_trace as ft
from vc.fiber_stream_volume import NativeCT, uniform_path, save_page, read_page

def test_cooperative_scheduler_cold_ct_and_resume(inputs):
    import json
    from test_fiber_trace import _write_u8_zarr
    field,normals=handles(inputs)
    trace=trace_config();trace.width_relative_tolerance=.3;trace.width_hard_relative_tolerance=.2
    ct=inputs/'ct'
    _write_u8_zarr(ct,np.full((32,32,256),17,dtype=np.uint8))
    options=dict(workers=4,active_fibers=40,seed_workers=2,max_open=4,
                 policy_grid_base=4,decision_base=64,maximum_base=256,
                 box_base_xyz=[[0,0,0],[256,32,32]],ct_locator=str(ct),ct_level=0)
    output=inputs/'stream'
    keys=[[i,0,0] for i in range(4)]
    result=json.loads(ft.run_stream_volume(field,normals,trace,chunk_config(),json.dumps(options),keys,str(output)))
    assert result['complete'] and result['finished_tiles']==4
    assert result['started_pairs']==result['finished_pairs']>0
    assert result['active_fibers']==0 and result['peak_active_fibers']<=40
    assert result['collision_checks']>0 and result['pieces']>0
    pages=sorted((output/'tiles').glob('*.msgpack.zst'))
    before={p.name:(p.stat().st_mtime_ns,p.read_bytes()) for p in pages}
    assert len(pages)==4
    resumed=json.loads(ft.run_stream_volume(field,normals,trace,chunk_config(),json.dumps(options),keys,str(output)))
    assert resumed['complete'] and resumed['started_pairs']==0
    assert resumed['pieces']==result['pieces']
    np.testing.assert_allclose(resumed['kept_base'],result['kept_base'],rtol=1e-12)
    assert before=={p.name:(p.stat().st_mtime_ns,p.read_bytes()) for p in pages}
    # A native exception must survive future.get(), not be masked by cleanup.
    import pytest
    options['ct_level']=9
    with pytest.raises(ValueError,match='CT level unavailable'):
        ft.run_stream_volume(field,normals,trace,chunk_config(),json.dumps(options),keys,str(output))

def test_native_coverage_matches_shared_policy():
    from vesuvius.neural_tracing.tracers.fibers.tracing.volume_policy import Owned,LivePaths,unit,tangents
    rng=np.random.default_rng(271)
    old=Owned(32,merge=50);native=ft.VolumeCoverage(32,50)
    for fid in range(20):
        points=np.cumsum(rng.normal(size=(30,3)),axis=0)+rng.uniform(-40,40,3)
        width=float(rng.uniform(40,100));old.add(points,fid,width);native.add_owned(points,fid,width)
        queries=points+rng.normal(scale=.3,size=points.shape);dirs=tangents(points)
        for gate in (0.,.2):
            expected=old.owner(queries,dirs,2,widths=width,tolerance=gate)
            actual=[native.owner(p,d,2,width,gate) for p,d in zip(queries,dirs)]
            np.testing.assert_array_equal(actual,expected)
    live=LivePaths(2);a=rng.normal(size=(8,3));ids=np.arange(8)
    for step in range(70):
        b=a+rng.normal(size=a.shape);dirs=unit(b-a);width=np.linspace(40,100,len(ids))
        expected=live.hits(b,dirs,ids,step,width,.2)
        actual=[native.live_hit(p,d,2,int(i),step,float(w),.2) for p,d,i,w in zip(b,dirs,ids,width)]
        np.testing.assert_array_equal(actual,expected)
        live.add(a,b,ids,step,width)
        for p,q,i,w in zip(a,b,ids,width):native.add_live(p,q,int(i),step,float(w))
        if step==20:
            live.drop([1,2]);native.drop_live(1);native.drop_live(2)
        a=b
        if step==20:
            keep=~np.isin(ids,[1,2]);a=a[keep];ids=ids[keep]

def test_stream_width_callbacks_and_concurrency(inputs):
    field,normals=handles(inputs)
    cfg=trace_config();cfg.width_relative_tolerance=.3;cfg.width_hard_relative_tolerance=.2
    cfg.sampling_window_voxels=64
    def one(_):
        seen=[]
        def observe(p,d,w,step):
            seen.append((p,d,w,step));return step<4
        result=ft.trace_stream(field,normals,[32,16,16],[1,0,0],160,cfg,2,observe,20)
        assert result['reason']=='observer_stop'
        assert [v[3] for v in seen]==[2,4]
        assert len(result['points'])==5
        np.testing.assert_allclose(result['widths'],20)
        np.testing.assert_allclose([v[2] for v in seen],20)
        return np.asarray(result['points'])
    with ThreadPoolExecutor(8) as pool:
        outputs=list(pool.map(one,range(16)))
    for result in outputs[1:]:np.testing.assert_array_equal(result,outputs[0])

def test_native_queue_and_cancellation(inputs):
    field,normals=handles(inputs)
    cfg=trace_config();cfg.width_relative_tolerance=.3
    control=ft.StreamControl()
    with ThreadPoolExecutor(8) as pool:
        jobs=[pool.submit(ft.trace_stream,field,normals,[32,16,16],[1,0,0],
                         160,cfg,2,control,20,i) for i in range(8)]
        seen=set()
        while len(seen)<8:
            events=control.poll(100)
            for e in events:
                assert e['step']==0 and e['width']==20
                seen.add(e['row'])
            # Intentionally leave all eight native requests waiting.
        control.cancel()
        for job in jobs:
            result=job.result(timeout=5)
            assert result['reason']=='observer_stop'
            assert len(result['points'])==1

def test_seeds_are_core_owned_and_have_width(inputs):
    field,normals=handles(inputs);cfg=chunk_config()
    seeds=ft.chunk_seeds(field,normals,[0,0,0],cfg)
    assert len(seeds['points_base'])>0
    assert np.all(seeds['points_base']>=0)
    assert np.all(seeds['points_base']<cfg.core_base)
    np.testing.assert_allclose(seeds['widths_base'],20)

def test_ct_sampling_does_not_double_scale():
    class Fake:
        def sample_coords(self,p,mask,**kwargs):
            np.testing.assert_array_equal(p,[[[100,200,300]]])
            assert kwargs['level']==2 and kwargs['sampling']=='nearest'
            return np.array([[17]],np.uint8),None,None
    ct=NativeCT.__new__(NativeCT);ct.volume=Fake();ct.level=2;ct.scale=1.
    np.testing.assert_array_equal(ct.outside([[100,200,300]]),[False])

def test_storage_and_policy_sampling(tmp_path):
    points=np.array([[0,0,0],[16,0,0],[32,0,0]])
    p,w,c=uniform_path(points,[20,20,20],[0,1,2])
    np.testing.assert_array_equal(p[:,0],[0,8,16,24,32])
    page=tmp_path/'a.msgpack.zst'
    pieces=[dict(id=7,points_base=p.tolist(),width_history_base=w.tolist(),cost=c.tolist())]
    save_page(page,pieces)
    assert read_page(page)['pieces']==pieces
