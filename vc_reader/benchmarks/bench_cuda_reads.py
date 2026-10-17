"""Integer gathers, axis-aligned crops and batched affine CUDA experiments.

Uses the same optional library and native reader as bench_cuda_sample.py.
All performance inputs are real CT; --validate-only uses a small fixture.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from bench_cuda_sample import (CUDA, bounds, compare, cpu_float32, exact, pack_chunks,
                              pointer, rebase, stats, validate, write_validation_array)
from bench_sample import DEFAULT_ARRAY, REGION, point_sets
from _common import digest, environment
from vc_reader.reader import ZarrReader


class ReadsCUDA(CUDA):
    def setup(self, case, local_grids):
        signatures = {
            "sample_integer_init": [ct.c_void_p],
            "sample_integer_run": [ct.c_void_p]*3+[ct.c_int]*2+[ct.POINTER(ct.c_float)],
            "sample_batch_init": [ct.c_void_p,ct.c_int]+[ct.c_void_p]*4,
            "sample_batch_run": [ct.c_void_p]*2+[ct.c_int]*4+[ct.POINTER(ct.c_float)],
            "sample_upload_output": [ct.c_void_p]*2+[ct.c_size_t]*2,
            "sample_upload_cast": [ct.c_void_p]*2,
        }
        for name,signature in signatures.items():
            getattr(self.lib,name).argtypes=signature
            getattr(self.lib,name).restype=ct.c_int
        self.kind=case["kind"]
        self.dtype=np.dtype(case["dtype"])
        if self.kind=="integer":
            self.check(self.lib.sample_integer_init(self.ctx))
        else:
            shapes=np.ascontiguousarray([g[0] for g in local_grids],np.int32)
            starts=None if self.kind=="affine" else np.ascontiguousarray([g[1] for g in case["grids"]],np.int32)
            origins=np.ascontiguousarray([g[1] for g in local_grids],np.float32)
            axes=np.ascontiguousarray([g[2] for g in local_grids],np.float32)
            self.check(self.lib.sample_batch_init(self.ctx,len(local_grids),pointer(shapes),
                       pointer(starts),pointer(origins),pointer(axes)))

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.sample_error().decode())

    def execute(self, source, points=None, upload=False, serial=False, upload_points=True):
        ms=ct.c_float()
        if self.kind=="integer":
            self.check(self.lib.sample_integer_run(self.ctx,pointer(source),pointer(points),
                       points.dtype.itemsize*8,int(upload)+2*int(upload_points),ct.byref(ms)))
        else:
            self.check(self.lib.sample_batch_run(self.ctx,pointer(source),self.kind=="affine",
                       self.dtype==np.float32,upload,serial,ct.byref(ms)))
        return ms.value

    def upload_result(self, output, cast=False):
        if cast:
            self.check(self.lib.sample_upload_cast(self.ctx,pointer(output)))
        else:
            self.check(self.lib.sample_upload_output(self.ctx,pointer(output),output.nbytes,0))

    def result(self):
        return self.download(0 if self.dtype==np.uint8 else 1,self.dtype)


def prepare(reader, case):
    """Known request -> conservative chunk union; no point enumeration for grids."""
    if case["kind"]=="integer":
        pts=case["points"]
        inside=np.all((pts>=0)&(pts<np.asarray(reader.shape)),axis=1)
        indices=np.unique(pts[inside]//np.asarray(reader.chunks),axis=0)
    else:
        indices=set()
        for grid in case["grids"]:
            if np.prod(grid[0])==0:
                continue
            if case["kind"]=="box":
                start,shape=grid[1],grid[0]
            else:
                start,shape=bounds(None,grid,reader.shape)
            indices.update(reader.chunk_indices(start,shape))
    # An entirely outside request can use a single fill-only lookup entry.
    if not len(indices):
        indices=[(0,0,0)]
    start,shape,mapping,read,nchunks=pack_chunks(reader,indices)
    grids=[rebase(None,g,start)[1] for g in case.get("grids",[])]
    return start,shape,mapping,read,nchunks,grids


def count(case):
    return len(case["points"]) if case["kind"]=="integer" else sum(int(np.prod(g[0])) for g in case["grids"])


def cpu(reader,case,offset,grids,raw=False):
    if case["kind"]=="integer":
        return reader.read_points(case["points"])
    dtype=np.uint8 if raw else np.dtype(case["dtype"])
    output=np.empty(count(case),dtype)
    views=[]
    begin=0
    for shape,_,_ in case["grids"]:
        n=int(np.prod(shape)); views.append(output[begin:begin+n].reshape(tuple(shape))); begin+=n
    if case["kind"]=="box":
        reader.read_many([(g[1],g[0]) for g in case["grids"]],out=views,dtype=dtype)
    else:
        for grid,out in zip(grids,views):
            cpu_float32(reader,None,grid,offset,1,out=out.reshape(-1))
    return output


def grid_cases():
    base=np.asarray(REGION[0])+128
    identity=np.eye(3,dtype=np.float32)
    angle=.35
    axes=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]],np.float32)
    for edge in (128,256):
        for dtype in ("uint8","float32"):
            for alignment in ("aligned","unaligned") if edge==128 else ("unaligned",):
                origin=base//128*128
                if alignment=="unaligned": origin=origin+[17,33,49]
                yield dict(name=f"box{edge}_{alignment}_{dtype}",kind="box",dtype=dtype,
                           grids=[(np.full(3,edge,np.int32),origin,identity)])
    for kind in ("box","affine"):
        for edge,batch in ((32,32),(64,16),(128,8)):
            for overlap in (True,False):
                grids=[]
                for i in range(batch):
                    offset=np.array([i//16,i//4%4,i%4])*(4 if overlap else 192)
                    origin=base+offset+[17,33,49]
                    if kind=="affine": origin=origin+.25
                    grids.append((np.full(3,edge,np.int32),origin,axes if kind=="affine" else identity))
                yield dict(name=f"{kind}{batch}x{edge}_{'overlap' if overlap else 'spread'}",
                           kind=kind,dtype="float32",grids=grids)


def cases(only):
    if not only or any(n.startswith("integer_") for n in only):
        for name,pts in point_sets(*REGION,1.).items():
            label="integer_"+name
            if not only or label in only:
                yield dict(name=label,kind="integer",dtype="uint8",points=np.rint(pts).astype(np.int64))
    for case in grid_cases():
        if not only or case["name"] in only: yield case


def validate_reads(binary):
    with tempfile.TemporaryDirectory(prefix="vc-read-check-") as tmp:
        write_validation_array(Path(tmp))
        with ZarrReader.open(tmp,threads=2) as reader:
            points=np.array([[0,0,0],[3,4,2],[4,5,3],[8,9,10],[-1,0,0],[9,0,0],
                             [2**60,0,0],[-2**60,0,0]],np.int64)
            grids=[(np.array([3,4,5]),np.array([-1,0,1]),np.eye(3)),
                   (np.array([2,3,4]),np.array([7,8,9]),np.eye(3)),
                   (np.array([0,2,3]),np.array([0,0,0]),np.eye(3)),
                   (np.array([2,2,2]),np.array([30,40,50]),np.eye(3))]
            tests=[dict(name="integer",kind="integer",dtype="uint8",points=points)]
            tests += [dict(name=dtype,kind="box",dtype=dtype,grids=grids) for dtype in ("uint8","float32")]
            tests.append(dict(name="affine",kind="affine",dtype="float32",grids=[
                (np.array([4,5,6]),np.array([6.5,-.5,2]),np.array([[-.8,0,.2],[.1,1,0],[0,.2,.9]])),*grids]))
            for case in tests:
                start,shape,mapping,read,nchunks,local=prepare(reader,case)
                source=read(); expected=cpu(reader,case,start,local)
                cuda=ReadsCUDA(binary,shape,start,reader.shape,count(case),int(reader.fill_value))
                try:
                    cuda.chunks(reader.chunks,mapping,nchunks); cuda.setup(case,local)
                    for serial in (False,True):
                        cuda.execute(source,case.get("points"),upload=True,serial=serial)
                        compare(cuda.result(),expected,True)
                        cuda.execute(source,case.get("points"),serial=serial)
                        compare(cuda.result(),expected,True)
                    cuda.upload_result(expected); exact(cuda.result(),expected)
                    if case["kind"]=="box" and case["dtype"]=="float32":
                        cuda.upload_result(cpu(reader,case,start,local,raw=True),cast=True)
                        exact(cuda.result(),expected)
                    if case["kind"]=="integer":
                        small=points.copy(); small[6:,0]=[-1,20]
                        expected=reader.read_points(small)
                        cuda.execute(source,small.astype(np.int32),upload=True)
                        exact(cuda.result(),expected)
                finally: cuda.close()
    print("read validation: integer64/32, boxes, mixed/empty/outside grids, missing chunks, float32 batches and serial replay",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("binary",type=Path)
    ap.add_argument("--array",default=DEFAULT_ARRAY)
    ap.add_argument("--only",default="")
    ap.add_argument("--reps",type=int,default=11)
    ap.add_argument("--warmup",type=int,default=3)
    ap.add_argument("--threads",type=int,default=16)
    ap.add_argument("--validate-only",action="store_true")
    ap.add_argument("--output",type=Path,default=Path("/tmp/cuda-reads.json"))
    args=ap.parse_args()
    if args.reps<1 or args.warmup<1: ap.error("reps/warmup must be positive")
    only=set(filter(None,args.only.split(",")))
    known={c["name"] for c in grid_cases()}|{"integer_random","integer_patch","integer_local"}
    if only-known: ap.error(f"unknown cases: {only-known}")
    validate(args.binary,True); validate_reads(args.binary)
    if args.validate_only: return
    from vc_reader import _native
    report=dict(command=[sys.executable,*sys.argv],environment=environment(),array=args.array,
        threads=args.threads,reps=args.reps,warmup=args.warmup,
        native_sha256=hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],text=True).strip(),
        protocol="warm mmap; seeded shuffled modes; uint8 source; float32 affine math; synchronous completion; source plan/rebase and device allocations excluded; GPU outputs stay on device, verification download excluded; per-batch descriptor upload included; integer coordinates uploaded each call except all_resident; CPU_device includes final output upload; CPU_raw_device includes uint8 crop read/upload and GPU cast",
        cpu_affine="shared native float32 sampler, one call per grid; output allocated contiguously for whole batch",
        rows=[])
    rng=np.random.default_rng(901)
    with ZarrReader.open(args.array,threads=args.threads,memory_cache=0) as reader:
        if reader.dtype!=np.uint8 or reader.ndim!=3: ap.error("3-D uint8 CT required")
        for case in cases(only):
            t=time.perf_counter(); start,shape,mapping,read,nchunks,local=prepare(reader,case)
            plan_ms=(time.perf_counter()-t)*1000
            source=read(); source_hash=digest(source)
            expected=cpu(reader,case,start,local)
            cuda=ReadsCUDA(args.binary,shape,start,reader.shape,count(case),int(reader.fill_value))
            try:
                cuda.chunks(reader.chunks,mapping,nchunks); cuda.setup(case,local)
                points=case.get("points")
                points32=None if points is None else points.astype(np.int32)
                if points is not None and not np.array_equal(points,points32): raise ValueError("int32 workload overflow")
                cuda.execute(source,points,upload=True)
                modes=["cpu","cpu_device","gpu_upload","gpu_read_upload","gpu_resident"]
                if case["kind"]=="integer": modes += ["gpu_resident_i32","gpu_all_resident"]
                else: modes += ["gpu_serial_resident"]
                if case["kind"]=="box" and case["dtype"]=="float32": modes += ["cpu_raw_device"]
                times={m:[] for m in modes}; kernels={m:[] for m in modes if m.startswith("gpu_")}
                errors=dict(max_abs=0.,changed=0)
                for rep in range(-args.warmup,args.reps):
                    order=modes.copy(); rng.shuffle(order)
                    for mode in order:
                        # The int32 mode shares storage with int64; restore resident
                        # int64 coordinates outside timing for the no-upload scenario.
                        if mode=="gpu_all_resident": cuda.execute(source,points)
                        t=time.perf_counter()
                        if mode.startswith("cpu"):
                            out=cpu(reader,case,start,local,raw=mode=="cpu_raw_device")
                            if mode!="cpu": cuda.upload_result(out,cast=mode=="cpu_raw_device")
                        else:
                            data=read() if mode=="gpu_read_upload" else source
                            ms=cuda.execute(data,points32 if mode=="gpu_resident_i32" else points,
                                upload=mode in ("gpu_upload","gpu_read_upload"),
                                serial=mode=="gpu_serial_resident",upload_points=mode!="gpu_all_resident")
                        elapsed=(time.perf_counter()-t)*1000
                        got=out if mode=="cpu" else cuda.result()
                        diff=compare(got,expected,True)
                        for key in errors: errors[key]=max(errors[key],diff[key])
                        if rep>=0:
                            times[mode].append(elapsed)
                            if mode in kernels: kernels[mode].append(ms)
                        del got
                        if mode.startswith("cpu"): del out
                        else: del data
                assert digest(source)==source_hash
                row=dict(name=case["name"],kind=case["kind"],dtype=case["dtype"],values=count(case),
                    batch=len(case.get("grids",[])),chunks=nchunks,source_bytes=source.nbytes,
                    output_bytes=expected.nbytes,point_bytes=0 if points is None else points.nbytes,
                    plan_ms=plan_ms,source_hash=source_hash,output_hash=digest(expected),differences=errors,
                    timings={m:stats(v) for m,v in times.items()},kernels={m:stats(v) for m,v in kernels.items()})
                report["rows"].append(row)
                print(case["name"]+f" chunks={nchunks} src={source.nbytes/2**20:.1f}MiB "+
                      " ".join(f"{m}={np.median(v):.3f}" for m,v in times.items()),flush=True)
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(report,indent=2)+"\n")
            finally: cuda.close()
    print(f"wrote {args.output}")


if __name__=="__main__": main()
