"""Resident CPU/CUDA geometry batches; setup/uploads and correctness checks are separate."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--queries',type=int,default=8192)
    parser.add_argument('--reps',type=int,default=30)
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args()
    if args.queries<1 or args.reps<1:parser.error('queries and reps must be positive')
    from bench_pipeline import load_extensions
    load_extensions(args.native_dir)
    from vc_reader import cuda, _native
    from vc_reader.geometry import PolylineSet,VoxelSet,transform_points
    rng=np.random.default_rng(1907)
    count,length=512,128
    base=rng.uniform(-1000,1000,(count,1,3)).astype(np.float32)
    delta=rng.normal(0,.3,(count,length,3)).astype(np.float32)
    delta[:,:,2]=1
    points=(base+np.cumsum(delta,axis=1)).reshape(-1,3)
    ids=rng.integers(0,count,args.queries,dtype=np.int64)
    queries=points[ids*length+rng.integers(0,length,args.queries)]+rng.normal(0,.4,(args.queries,3)).astype(np.float32)
    arcs=rng.uniform(0,125,args.queries).astype(np.float32)
    planes=np.zeros((args.queries,4),np.float32);planes[:,2]=1;planes[:,3]=points[ids*length,2]+arcs
    matrix=np.array([[0,-1,0,2],[1,0,0,3],[0,0,1,4]],np.float32)
    shape=(144,104,104)
    voxels=rng.uniform(0,np.array(shape)-1,(max(200000,args.queries),3)).astype(np.float32)
    batches=rng.integers(0,32,len(voxels),dtype=np.int64)
    lookup=voxels[:args.queries].copy();lookup[::2]+=.7
    report=dict(queries=args.queries,reps=args.reps,curves=count,vertices=len(points),voxel_points=len(voxels),cases={})
    report.update(command=sys.argv,python=platform.python_version(),native_path=_native.__file__,
                  native_sha256=hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest())
    before=time.perf_counter();cpu=PolylineSet(points,np.arange(count+1,dtype=np.int64)*length);report['cpu_index_seconds']=time.perf_counter()-before
    before=time.perf_counter();gpu=cpu.cuda();report['cuda_setup_upload_seconds']=time.perf_counter()-before
    report['curve_index_bytes']=cpu.nbytes
    up=lambda a:cuda.asarray(a)
    d_ids,d_queries,d_arcs,d_planes,d_matrix,d_voxels,d_batches,d_lookup=map(up,[ids,queries,arcs,planes,matrix,voxels,batches,lookup])
    cv=VoxelSet(voxels,shape,batch_ids=batches,batches=32)
    gv=VoxelSet(d_voxels,shape,batch_ids=d_batches,batches=32)
    report['sparse_bytes']=dict(cpu=cv.nbytes,cuda=gv.nbytes,dense=32*int(np.prod(shape)))
    def measure(fn):
        for _ in range(3):fn()
        times=[]
        for _ in range(args.reps):
            start=time.perf_counter();fn();times.append((time.perf_counter()-start)*1000)
        return dict(median_ms=statistics.median(times),mean_ms=statistics.mean(times),p95_ms=float(np.percentile(times,95)),times_ms=times)
    gpu_cases=[]
    def case(name,cpu_fn,gpu_fn):
        a,b=cpu_fn(),gpu_fn()
        aa,bb=(a,b) if isinstance(a,dict) else ({'value':a},{'value':b})
        for key in aa:
            x,y=aa[key],bb[key].numpy()
            if x.dtype.kind=='f':np.testing.assert_allclose(x,y,atol=3e-4,rtol=2e-5,equal_nan=True)
            else:np.testing.assert_array_equal(x,y)
        c=measure(lambda:cpu_fn(a));g=measure(lambda:gpu_fn(b))
        gpu_cases.append(lambda:gpu_fn(b))
        report['cases'][name]=dict(cpu=c,cuda=g,speedup=c['median_ms']/g['median_ms'])
        print(name,round(c['median_ms'],3),round(g['median_ms'],3),round(c['median_ms']/g['median_ms'],2),flush=True)
    case('interpolate',lambda out=None:cpu.interpolate(ids,arcs,out=out),lambda out=None:gpu.interpolate(d_ids,d_arcs,out=out))
    case('intersect_planes',lambda out=None:cpu.intersect_planes(ids,planes,out=out),lambda out=None:gpu.intersect_planes(d_ids,d_planes,out=out))
    for vertex in [False,True]:
        case('nearest_vertices' if vertex else 'nearest_segments',lambda out=None:cpu.nearest(queries,vertices=vertex,out=out),lambda out=None:gpu.nearest(d_queries,vertices=vertex,out=out))
    lower,upper=queries-2,queries+2
    d_lower,d_upper=up(lower),up(upper)
    case('query_boxes',lambda out=None:cpu.query_boxes(lower,upper,capacity=32,out=out),lambda out=None:gpu.query_boxes(d_lower,d_upper,capacity=32,out=out))
    selected=rng.integers(0,len(points),args.queries,dtype=np.int64);d_selected=up(selected)
    case('gather_segments',lambda out=None:cpu.gather(selected,segments=True,out=out),lambda out=None:gpu.gather(d_selected,segments=True,out=out))
    case('transform',lambda out=None:transform_points(queries,matrix,out=out),lambda out=None:transform_points(d_queries,d_matrix,out=out))
    lookup_ids=batches[:args.queries];d_lookup_ids=up(lookup_ids)
    case('contains',lambda out=None:cv.contains(lookup,batch_ids=lookup_ids,out=out),lambda out=None:gv.contains(d_lookup,batch_ids=d_lookup_ids,out=out))
    case('to_dense',lambda out=None:cv.to_dense(out=out),lambda out=None:gv.to_dense(out=out))
    c=measure(lambda:VoxelSet(voxels,shape,batch_ids=batches,batches=32))
    g=measure(lambda:VoxelSet(d_voxels,shape,batch_ids=d_batches,batches=32))
    report['cases']['voxelize']=dict(cpu=c,cuda=g,speedup=c['median_ms']/g['median_ms'])
    print('voxelize',round(c['median_ms'],3),round(g['median_ms'],3),round(c['median_ms']/g['median_ms'],2),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.profile:
        import torch
        report.update(gpu=torch.cuda.get_device_name(),torch=torch.__version__)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            with torch.profiler.record_function('geometry.resident'):
                for fn in gpu_cases:fn()
                VoxelSet(d_voxels,shape,batch_ids=d_batches,batches=32)
        trace_path=args.output.with_suffix('.trace.json')
        profile.export_chrome_trace(str(trace_path))
        events=json.loads(trace_path.read_text())['traceEvents']
        region=next(e for e in events if e.get('name')=='geometry.resident')
        transfers=[e for e in events if e.get('cat')=='gpu_memcpy']
        barriers=[e for e in events if e.get('name','').startswith('cudaDeviceSynchronize') and region['ts']<=e.get('ts',-1)<=region['ts']+region['dur']]
        report['profile']=dict(download_bytes=sum(e.get('args',{}).get('bytes',0) for e in transfers if 'DtoH' in e.get('name','')),
            upload_bytes=sum(e.get('args',{}).get('bytes',0) for e in transfers if 'HtoD' in e.get('name','')),device_barriers=len(barriers))
        assert report['profile']==dict(download_bytes=0,upload_bytes=0,device_barriers=0),report['profile']
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
