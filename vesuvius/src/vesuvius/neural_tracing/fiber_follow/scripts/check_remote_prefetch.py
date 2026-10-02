"""Compare cold-cache async CT fetching with one and multiple connections.

Uses 8 native chunks from each recorded scan (32 MiB total at 128^3 uint8).
Creates temporary independent caches, checks identical voxels, and leaves the
training cache and running model alone. Timings exclude process/metadata startup.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from vesuvius.neural_tracing.fiber_follow.data.remote_prefetch import RemotePrefetcher
from vesuvius.neural_tracing.fiber_follow.data.volume import RemoteChunkedArray


def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--connections',type=int,default=8)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--out',default=str(root/'datasets/automated_fiber_volumes/prefetch_benchmark.json'))
    args=ap.parse_args()
    if args.connections<2 or args.repeats<1:raise ValueError('Need at least two connections and one repeat')
    sources=json.loads((root/'datasets/automated_fiber_volumes/cache_verification.json').read_text())
    rows=[];expected={}
    with tempfile.TemporaryDirectory(prefix='fiber-async-ct-') as folder:
        for repeat in range(args.repeats):
            # Alternate order to reduce warm-up/order bias in the remote service.
            for connections in ((1,args.connections) if repeat%2==0 else (args.connections,1)):
                cache=Path(folder)/f'{repeat}_{connections}'
                with RemotePrefetcher(connections=connections,queue_size=128) as service:
                    readers=[]
                    startup=time.monotonic()
                    for source in sources:
                        spec=SimpleNamespace(ct_zarr=source['source'],ct_level=0,cache_dir=str(cache))
                        service.client.ensure_metadata(spec)
                        reader=RemoteChunkedArray(spec.ct_zarr,0,str(cache),0,cache_only=True)
                        start=(np.asarray(source['patch_origin_zyx'])//reader.chunks)*reader.chunks
                        size=np.asarray(reader.chunks)*2
                        readers.append((reader,start,size))
                    startup=time.monotonic()-startup
                    began=time.monotonic()
                    for reader,start,size in readers:
                        service.client.ensure(reader,[(start,size)])
                    elapsed=time.monotonic()-began
                    hashes={reader.remote_url:hashlib.sha256(reader.read(start,size).tobytes()).hexdigest()
                            for reader,start,size in readers}
                    if not expected:expected=hashes
                    assert hashes==expected
                    count=sum(int(np.prod(size))*reader.dtype.itemsize for reader,_,size in readers)
                    row=dict(repeat=repeat,connections=connections,fetch_seconds=elapsed,
                        startup_seconds=startup,bytes=count,mib_per_second=count/2**20/elapsed,
                        voxel_sha256=hashes,stats=service.snapshot())
                    print(json.dumps({k:v for k,v in row.items() if k not in ('voxel_sha256','stats')}),flush=True)
                    rows.append(row)
    summary={}
    for connections in (1,args.connections):
        times=[r['fetch_seconds'] for r in rows if r['connections']==connections]
        summary[str(connections)]=dict(mean_seconds=statistics.mean(times),
            p50_seconds=float(np.percentile(times,50)),p95_seconds=float(np.percentile(times,95)))
    report=dict(sources=[s['source'] for s in sources],rows=rows,summary=summary,
        identical_voxels=True,scope='Cold temporary disk caches; excludes process and metadata startup; not training throughput')
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
