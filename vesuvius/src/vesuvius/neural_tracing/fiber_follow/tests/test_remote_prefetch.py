"""Async concurrency, demand preemption, local-only reads and queue backpressure."""
import asyncio
import multiprocessing as mp
from pathlib import Path
import queue
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.shared import remote_prefetch as module
from vesuvius.neural_tracing.fiber_follow.shared.volume import RemoteChunkedArray


@pytest.mark.parametrize('connections',[3,32])
def test_prefetch_connection_limit_reaches_s3_pool(tmp_path,monkeypatch,connections):
    from zarr.api import asynchronous
    from vesuvius.neural_tracing.fiber_follow.shared.volume import remote_store
    opened=[]
    async def open_array(*,store,path,mode):
        # Use the real Zarr/fsspec store and S3 filesystem without remote I/O.
        assert store.fs.config_kwargs['max_pool_connections']==connections
        assert store.fs.anon and path=='0' and mode=='r'
        opened.append(store)
        return store
    monkeypatch.setattr(asynchronous,'open_array',open_array)
    async def scenario():
        fetcher=module.AsyncFetcher('test',connections)
        try:
            for scan in ('first','second'):
                descriptor=(f's3://vesuvius-challenge-open-data/{scan}.zarr',0,str(tmp_path))
                first=await fetcher.array(descriptor)
                assert await fetcher.array(descriptor) is first
            assert len(opened)==2
            # Synchronous users keep the backend default.
            assert 'max_pool_connections' not in remote_store(descriptor[0]).fs.config_kwargs
        finally:
            await fetcher.close()
    asyncio.run(scenario())


def source_at(tmp_path,separator='.'):
    import zarr
    from numcodecs import Blosc
    values=np.arange(9*10*11,dtype=np.uint8).reshape(9,10,11)
    root=tmp_path/'source.zarr'
    array=zarr.open_array(str(root/'0'),mode='w',zarr_format=2,shape=values.shape,
        chunks=(4,4,4),dtype=values.dtype,compressor=Blosc(cname='zstd'),dimension_separator=separator)
    array[:]=values
    spec=SimpleNamespace(ct_zarr=root.as_uri(),ct_level=0,cache_dir=str(tmp_path/'cache'))
    return spec,values


@pytest.mark.parametrize('separator',['.','/'])
def test_async_process_prefetches_compressed_source_and_never_downloads_in_reader(tmp_path,monkeypatch,separator):
    import json
    import zarr
    spec,values=source_at(tmp_path,separator)
    def forbidden(self):raise AssertionError('Foreground remote I/O is forbidden')
    monkeypatch.setattr(RemoteChunkedArray,'_open',forbidden)
    with pytest.raises(FileNotFoundError,match='metadata'):
        RemoteChunkedArray(spec.ct_zarr,0,spec.cache_dir,1024,cache_only=True)
    with module.RemotePrefetcher(connections=4,queue_size=2,timeout=30) as service:
        service.client.ensure_metadata(spec)
        reader=RemoteChunkedArray(spec.ct_zarr,0,spec.cache_dir,1024,cache_only=True)
        assert Path(reader.path)==Path(spec.cache_dir)/'file'/str(tmp_path/'source.zarr').lstrip('/')/'0'
        bounds=[(np.zeros(3,dtype=int),values.shape)]
        with pytest.raises(FileNotFoundError,match='chunk'):
            reader.read([0,0,0],[1,1,1])
        service.client.submit(reader,bounds)
        service.client.ensure(reader,bounds)
        np.testing.assert_array_equal(reader.read([0,0,0],values.shape),values)
        # More required chunks than either queue can hold: none may be lost.
        pattern='[0-9]*' if separator=='.' else '[0-9]*/[0-9]*/[0-9]*'
        assert len(list(Path(reader.path).glob(pattern)))==27
        assert (Path(reader.path)/separator.join(['2','2','2'])).stat().st_size==64
        metadata=json.loads((Path(reader.path)/'.zarray').read_text())
        assert metadata['compressor'] is None and metadata['filters'] is None
        assert metadata['dimension_separator']==separator
        np.testing.assert_array_equal(zarr.open_array(reader.path,mode='r')[:],values)
        assert service.snapshot()['deferred']>0
    assert not service.snapshot()['alive']
    np.testing.assert_array_equal(reader.read([7,8,9],[2,2,2]),values[7:9,8:10,9:11])


@pytest.mark.parametrize('scopes',[1,2])
def test_retained_lookahead_downloads_past_chunk_queue_while_loader_is_idle(tmp_path,scopes):
    spec,values=source_at(tmp_path,'/')
    with module.RemotePrefetcher(connections=4,queue_size=2,lookahead_slots=scopes) as service:
        service.client.ensure_metadata(spec)
        reader=RemoteChunkedArray(spec.ct_zarr,0,spec.cache_dir,0,cache_only=True)
        # Publish once, then stop consuming/producing batches like compilation.
        # The 27-chunk window is much larger than either two-entry chunk queue.
        if scopes==1:
            service.client.lookahead(reader,[[(np.zeros(3,dtype=int),values.shape)]])
        else:
            # Different fiber collections may share one CT volume; neither
            # collection's plans may replace the other's window.
            service.client.lookahead(reader,[[([0,0,0],[4,10,11])]],scope=0)
            service.client.lookahead(reader,[[([4,0,0],[5,10,11])]],scope=1)
        deadline=time.monotonic()+15
        while service.snapshot()['completed_bytes']<27*64:
            assert time.monotonic()<deadline, service.snapshot()
            time.sleep(.02)
        np.testing.assert_array_equal(reader.read([0,0,0],values.shape),values)
        assert service.snapshot()['completed_bytes']==27*64


class WorkerProbe(torch.utils.data.IterableDataset):
    def __init__(self,spec,client):self.spec,self.client=spec,client
    def __iter__(self):
        self.client.ensure_metadata(self.spec)
        reader=RemoteChunkedArray(self.spec.ct_zarr,0,self.spec.cache_dir,1024,cache_only=True)
        worker=torch.utils.data.get_worker_info().id
        start,size=[worker*4,0,0],[4,10,11]
        self.client.submit(reader,[(start,size)])
        self.client.ensure(reader,[(start,size)])
        yield dict(worker=worker,data=reader.read(start,size))


@pytest.mark.parametrize('start_method',sorted({'spawn',mp.get_start_method()}))
def test_multiple_spawned_loader_workers_share_one_prefetch_process(tmp_path,start_method):
    spec,values=source_at(tmp_path)
    with module.RemotePrefetcher(connections=4,queue_size=8,timeout=30) as service:
        loader=torch.utils.data.DataLoader(WorkerProbe(spec,service.client),batch_size=None,
            num_workers=2,multiprocessing_context=start_method)
        batches=list(loader)
        assert len(batches)==2
        for batch in batches:
            start=4*batch['worker']
            np.testing.assert_array_equal(batch['data'].numpy(),values[start:start+4])


@pytest.mark.parametrize('windowed',[False,True])
def test_priority_promotes_matching_fetch_and_preempts_unrelated_inflight_work(monkeypatch,tmp_path,windowed):
    async def scenario():
        context=mp.get_context('spawn')
        stopped=context.Event();heartbeat=context.Value('d',time.monotonic())
        counters=context.Array('q',len(module.COUNTERS))
        urgent,ahead=queue.Queue(),queue.Queue()
        windows=queue.Queue()
        monkeypatch.setattr(module,'_paths',lambda descriptor,key,session:
            (tmp_path/str(key),tmp_path/(str(key)+'.error')))
        began={key:asyncio.Event() for key in ('far_a','far_b','urgent')}
        release=asyncio.Event();done=asyncio.Event();cancelled=[];finished=[]
        class Fetcher:
            def __init__(self,session,connections):assert connections==2
            async def request(self,descriptor,key):
                began[key].set()
                if key=='urgent':
                    (tmp_path/key).touch()
                    finished.append(key);done.set();return 'completed',1
                try:
                    await release.wait()
                    (tmp_path/key).touch()
                    finished.append(key);return 'completed',1
                except asyncio.CancelledError:
                    cancelled.append(key);raise
            async def close(self):pass
        monkeypatch.setattr(module,'AsyncFetcher',Fetcher)
        if windowed:windows.put((0,'source',('far_a','far_b')))
        else:
            ahead.put(('source','far_a'));ahead.put(('source','far_b'))
        task=asyncio.create_task(module._run(urgent,ahead,stopped,heartbeat,counters,'test',2,8,
                                           windows,1))
        try:
            await asyncio.wait_for(asyncio.gather(began['far_a'].wait(),began['far_b'].wait()),2)
            # Both connections are occupied. A current batch needs far_a and a
            # different urgent chunk: promote a in place, preempt b immediately.
            urgent.put(('source','far_a'));urgent.put(('source','urgent'))
            await asyncio.wait_for(done.wait(),2)
            assert cancelled==['far_b'] and finished==['urgent']
            assert counters[module.COUNTERS.index('promoted')]==1
            assert counters[module.COUNTERS.index('preempted')]==1
            release.set()
            for _ in range(100):
                if 'far_b' in finished and 'far_a' in finished:break
                await asyncio.sleep(.01)
            assert 'far_a' in finished and 'far_b' in finished
        finally:
            stopped.set();release.set();await task
    asyncio.run(scenario())


def test_fetch_uses_concurrent_async_zarr_getitem_and_atomic_raw_writes(tmp_path,monkeypatch):
    async def scenario():
        active=peak=0
        both=asyncio.Event()
        class Array:
            shape=(8,4,4);chunks=(4,4,4);dtype=np.dtype('u1');fill_value=0
            async def getitem(self,selection):
                nonlocal active,peak
                active+=1;peak=max(peak,active)
                if active==2:both.set()
                await asyncio.wait_for(both.wait(),1)
                await asyncio.sleep(.01)
                active-=1
                return np.full((4,4,4),selection[0].start,np.uint8)
        async def open_array(*args,**kwargs):return Array()
        monkeypatch.setattr(module,'open_remote_array',open_array)
        fetcher=module.AsyncFetcher('test',2)
        descriptor=('s3://fixture/scan',0,str(tmp_path))
        try:
            await fetcher.fetch(descriptor,None)
            results=await asyncio.gather(fetcher.fetch(descriptor,(0,0,0)),fetcher.fetch(descriptor,(1,0,0)))
            assert peak==2 and results==[('completed',64),('completed',64)]
            reader=RemoteChunkedArray(*descriptor,cache_bytes=0,cache_only=True)
            assert (reader.read([4,0,0],[4,4,4])==4).all()
        finally:await fetcher.close()
    asyncio.run(scenario())


def test_prefetch_failure_propagates_without_foreground_fallback(tmp_path,monkeypatch):
    async def fail(*args,**kwargs):raise OSError('remote source unavailable')
    monkeypatch.setattr(module,'open_remote_array',fail)
    descriptor=('s3://fixture/broken',0,str(tmp_path));session='test'
    async def attempt():
        fetcher=module.AsyncFetcher(session,2)
        try:assert await fetcher.request(descriptor,None)==('errors',0)
        finally:await fetcher.close()
    asyncio.run(attempt())
    context=mp.get_context('spawn')
    counters=context.Array('q',len(module.COUNTERS))
    client=module.PrefetchClient(None,None,context.Event(),context.Value('d',time.monotonic()),counters,session,1)
    spec=SimpleNamespace(ct_zarr=descriptor[0],ct_level=0,cache_dir=descriptor[2])
    with pytest.raises(RuntimeError,match='remote source unavailable'):client.ensure_metadata(spec)


def test_main_and_history_prefetch_covers_ct_normals_and_all_crop_rolls():
    from test_history_slabs import observation
    from slab_fixtures import cfg
    from vesuvius.neural_tracing.fiber_follow.regression.data import ObservationBuilder
    from vesuvius.neural_tracing.fiber_follow.regression.history_slabs import slab_layout,SLAB
    from vesuvius.neural_tracing.fiber_follow.shared.data import tight_block
    model=cfg();builder=ObservationBuilder(model)
    item=observation([[100,100,0],[100,100,256]])
    bounds=list(builder.prefetch_bounds(item,SimpleNamespace(input_scale=2.)))
    observations=[(item,model.fine)]+[(s,SLAB) for s in slab_layout(item)]
    assert len(bounds)==18
    for j,(observation,crop) in enumerate(observations):
        lo,size=bounds[2*j]
        for angle in np.linspace(0,2*np.pi,33):
            c,s=np.cos(angle),np.sin(angle)
            rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            start,extent=tight_block(observation['pos'],observation['frame']@rotation,crop,2.)
            assert np.all(start>=lo) and np.all(start+extent<=lo+size)
        start,extent=bounds[2*j+1]
        center=observation['pos'][::-1]*2.
        assert np.all(center-start>=32) and np.all(start+extent-center>32)


def test_dataset_gates_reads_without_changing_samples(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.shared import data as data_module
    events=[]
    monkeypatch.setattr(data_module,'FiberVolume',lambda *a,**kw: events.append(('volume',kw.get('cache_only',False))) or SimpleNamespace(ct='ct'))
    monkeypatch.setattr(data_module,'crop_local_grid',lambda crop:np.zeros((1,3)))
    monkeypatch.setattr(data_module,'make_sample',lambda f,t,rev,cfg,rng,**options:dict(at=t,reverse=rev))
    class Builder:
        def prefetch_bounds(self,item,vol):return [(item['at'],item['reverse'])]
        def __call__(self,items,vol):
            events.append(('read',len(items)))
            return items
    class Client:
        def ensure_metadata(self,spec):events.append(('metadata',True))
        def submit(self,reader,bounds):events.append(('ahead',len(bounds)))
        def ensure(self,reader,bounds):events.append(('demand',len(bounds)))
    def sample(prefetch):
        ds=data_module.FollowDataset([SimpleNamespace(length=100.)],None,
            SimpleNamespace(crop=None,future_s=[16.]),None,chunk=4,seed=17,batch_builder=Builder(),
            budget=data_module.TaskBudget.parse(['fresh=1','live=0','dagger_pre_excursion=0','dagger_recoverable=0',
                'dagger_terminal=0','dagger_premature_stop=0','dagger_ordinary=0','synthetic_terminal=0']))
        ds.state_allowed=lambda item:True
        ds.remote_prefetch=Client() if prefetch else None
        stream=iter(ds)
        return [next(stream) for _ in range(3)]
    baseline=sample(False);events.clear()
    assert sample(True)==baseline
    assert events[:2]==[('metadata',True),('volume',True)]
    assert events[2:]==([('ahead',1)]*4+[('demand',4),('read',4)])*3
