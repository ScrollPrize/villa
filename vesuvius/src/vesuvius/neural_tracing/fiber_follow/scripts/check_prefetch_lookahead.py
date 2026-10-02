"""Measure fetching during an idle-loader pause with deterministic simulated I/O.

Uses the real priority scheduler, async fetcher and raw Zarr cache with a fixed
per-chunk delay instead of S3. CPU only; temporary caches; no training changes.
This measures lookahead scheduling, not real network or training throughput.
"""
import argparse
import asyncio
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import statistics
import tempfile
import time
from unittest.mock import patch

import numpy as np

from vesuvius.neural_tracing.fiber_follow.data import remote_prefetch as module
from vesuvius.neural_tracing.fiber_follow.data.volume import RemoteChunkedArray


async def measure(folder,lookahead,args):
    class Array:
        shape=((args.chunks+1)*16,16,16);chunks=(16,16,16)
        dtype=np.dtype('u1');fill_value=0
        async def getitem(self,selection):
            await asyncio.sleep(args.latency)
            return np.full(self.chunks,selection[0].start//16,np.uint8)
    async def open_array(*a,**kw):return Array()
    context=mp.get_context('spawn')
    stopped=context.Event();heartbeat=context.Value('d',time.monotonic())
    counters=context.Array('q',len(module.COUNTERS))
    urgent,ahead,windows=queue.Queue(),queue.Queue(),queue.Queue()
    descriptor=('s3://benchmark/ct',0,str(folder));session='benchmark'
    def ready(key):return module._paths(descriptor,key,session)[0].is_file()
    async def demand(keys):
        deadline=time.monotonic()+30
        while missing:=[key for key in keys if not ready(key)]:
            if time.monotonic()>deadline:raise TimeoutError('Benchmark cache readiness')
            for key in missing:urgent.put((descriptor,key))
            await asyncio.sleep(.02)
    with patch.object(module,'open_remote_array',open_array):
        task=asyncio.create_task(module._run(urgent,ahead,stopped,heartbeat,counters,
                                           session,args.connections,2,windows,1))
        try:
            await demand([None]);await demand([(0,0,0)])
            keys=[(i,0,0) for i in range(1,args.chunks+1)]
            if lookahead:windows.put((0,descriptor,tuple(keys)))
            await asyncio.sleep(args.pause)
            cached=sum(ready(key) for key in keys)
            began=time.monotonic()
            await demand(keys)
            wait=time.monotonic()-began
            reader=RemoteChunkedArray(*descriptor,cache_bytes=0,cache_only=True)
            values=reader.read([0,0,0],Array.shape)
            expected=np.broadcast_to(np.repeat(np.arange(args.chunks+1,dtype=np.uint8),16)[:,None,None],Array.shape)
            np.testing.assert_array_equal(values,expected)
            return dict(lookahead=lookahead,cached_after_pause=cached,demand_wait_seconds=wait,
                        voxel_sha256=hashlib.sha256(values.tobytes()).hexdigest())
        finally:
            stopped.set();await task


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--chunks',type=int,default=16)
    ap.add_argument('--connections',type=int,default=4)
    ap.add_argument('--latency',type=float,default=.05)
    ap.add_argument('--pause',type=float,default=.5)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    if min(args.repeats,args.chunks,args.connections,args.latency,args.pause)<=0 or args.chunks>255:
        raise ValueError('Positive benchmark settings required, chunks <= 255')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='fiber-lookahead-') as folder:
        for repeat in range(args.repeats):
            for enabled in ((False,True) if repeat%2==0 else (True,False)):
                row=asyncio.run(measure(Path(folder)/f'{repeat}_{enabled}',enabled,args))
                rows.append(dict(repeat=repeat,**row))
    assert len({row['voxel_sha256'] for row in rows})==1
    summary={}
    for enabled in (False,True):
        selected=[row for row in rows if row['lookahead']==enabled]
        timings=[row['demand_wait_seconds'] for row in selected]
        summary[str(enabled)]=dict(mean_wait_seconds=statistics.mean(timings),
            p50_wait_seconds=float(np.percentile(timings,50)),p95_wait_seconds=float(np.percentile(timings,95)),
            cached_after_pause=[row['cached_after_pause'] for row in selected])
    report=dict(scope=__doc__,settings=vars(args),rows=rows,summary=summary,identical_voxels=True)
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
