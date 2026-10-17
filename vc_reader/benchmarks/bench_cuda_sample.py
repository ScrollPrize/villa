"""CUDA CT sampling feasibility: public CPU API vs source boxes/packed chunks.

Requires an existing native reader, NumPy, CUDA toolkit and GPU. No installs.
See cuda_sampling_results.md for the protocol, limitations and reproduction.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import itertools
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from _common import digest, environment
from bench_sample import DEFAULT_ARRAY, REGION, patch_geometry, point_sets
from vc_reader.reader import ZarrReader


def pointer(a):
    return None if a is None else a.ctypes.data


class CUDA:
    """Private benchmark ABI; callers supply contiguous uint8/float32 buffers."""
    def __init__(self, binary, shape, start, volume, count, fill=0):
        self.lib = ct.CDLL(str(Path(binary).resolve()))
        self.lib.sample_error.restype = ct.c_char_p
        self.lib.sample_create.argtypes = [ct.c_void_p]*3 + [ct.c_size_t, ct.c_ubyte]
        self.lib.sample_create.restype = ct.c_void_p
        self.lib.sample_destroy.argtypes = [ct.c_void_p]
        self.lib.sample_destroy.restype = None
        self.lib.sample_run.argtypes = [ct.c_void_p]*7 + [ct.c_int]*3 + [ct.POINTER(ct.c_float)]
        self.lib.sample_run.restype = ct.c_int
        self.lib.sample_download.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
        self.lib.sample_download.restype = ct.c_int
        arrays = [np.asarray(v, np.int32) for v in (shape, start, volume)]
        self.ctx = self.lib.sample_create(*(pointer(a) for a in arrays), count, fill)
        if not self.ctx:
            raise RuntimeError(self.lib.sample_error().decode())
        self.count = count

    def close(self):
        if self.ctx:
            self.lib.sample_destroy(self.ctx)
            self.ctx = None

    def chunks(self, chunks, mapping, count):
        self.lib.sample_chunks.argtypes = [ct.c_void_p]*4 + [ct.c_size_t]
        self.lib.sample_chunks.restype = ct.c_int
        shape = np.asarray(mapping.shape,np.int32)
        chunks = np.asarray(chunks,np.int32)
        if self.lib.sample_chunks(self.ctx,pointer(chunks),pointer(shape),pointer(mapping),count):
            raise RuntimeError(self.lib.sample_error().decode())

    def download(self, order, dtype):
        out = np.empty(self.count, dtype=np.uint8 if order == 0 else dtype)
        if self.lib.sample_download(self.ctx, pointer(out), out.nbytes):
            raise RuntimeError(self.lib.sample_error().decode())
        return out

    def run(self, source, points, grid, order, dtype, uploads, device_output=False):
        out = None if device_output else np.empty(self.count, dtype=np.uint8 if order == 0 else dtype)
        shape, origin, axes = (None, None, None) if grid is None else grid
        ms = ct.c_float()
        if self.lib.sample_run(self.ctx, pointer(source), pointer(points), pointer(out),
                               pointer(shape), pointer(origin), pointer(axes), order,
                               np.dtype(dtype) == np.float64, uploads, ct.byref(ms)):
            raise RuntimeError(self.lib.sample_error().decode())
        return out, ms.value


def rebase(points, grid, offset):
    """One-time input conversion; all subsequent sampler arithmetic is float32."""
    if grid is None:
        with np.errstate(over="ignore"):
            return np.ascontiguousarray(points-offset,dtype=np.float32),None
    shape,origin,axes = grid
    return None,(shape,np.ascontiguousarray(origin-offset,dtype=np.float32),
                 np.ascontiguousarray(axes,dtype=np.float32))


def cpu_float32(reader, points, grid, offset, order, out=None):
    count = len(points) if grid is None else int(np.prod(grid[0]))
    if out is None:
        out = np.empty(count,np.uint8 if order==0 else np.float32)
    offset = list(map(int,offset))
    if grid is None:
        reader._array._sample_u8_float32(points,offset,out.view(np.uint8),order)
    else:
        shape,origin,axes = grid
        reader._array._affine_u8_float32(list(map(int,shape)),origin.tolist(),axes.ravel().tolist(),
                                        offset,out.view(np.uint8),order)
    return out


def precision_error(actual, reference):
    a,b = actual.reshape(-1),reference.reshape(-1)
    finite = np.isfinite(b)
    delta = np.abs(a[finite].astype(float)-b[finite])
    return dict(max_abs=float(np.max(delta,initial=0)),mean_abs=float(np.mean(delta)),
                changed=int(np.count_nonzero(delta)),values=len(delta))


def bounds(points, grid, volume):
    if grid is not None:
        shape, origin, axes = grid
        corners = np.asarray(list(itertools.product(*[(0, int(n)-1) for n in shape])))
        points = origin + corners[:, :1]*axes[:, 0] + corners[:, 1:2]*axes[:, 1] + corners[:, 2:]*axes[:, 2]
    lo = np.maximum(0, np.floor(points.min(axis=0)).astype(np.int64))
    hi = np.minimum(volume, np.ceil(points.max(axis=0)).astype(np.int64)+1)
    return lo, hi-lo


def chunk_source(reader, points, grid):
    """Conservative touched-chunk plan. Python planning is reported separately.

    A production path should share the reader's native planner/cache, not use
    this NumPy enumeration. The experiment isolates transfer/sampling economics.
    """
    chunks = np.asarray(reader.chunks,np.int64)
    volume = np.asarray(reader.shape,np.int64)
    lo,shape = bounds(points,grid,volume)
    start = lo//chunks*chunks
    end = np.minimum(volume,(lo+shape+chunks-1)//chunks*chunks)
    map_shape = (end-start+chunks-1)//chunks
    touched = set()
    count = len(points) if grid is None else int(np.prod(grid[0]))
    for begin in range(0,count,262144):
        if grid is None:
            c = points[begin:begin+262144]
        else:
            output,origin,axes = grid
            p = np.arange(begin,min(begin+262144,count),dtype=np.int64)
            z,y,x = p//(int(output[1])*output[2]), p//output[2]%output[1], p%output[2]
            c = origin+z[:,None]*axes[:,0]+y[:,None]*axes[:,1]+x[:,None]*axes[:,2]
        lower = np.floor(np.nextafter(c,-np.inf)).astype(np.int64)
        upper = np.ceil(np.nextafter(c,np.inf)).astype(np.int64)
        lower = (np.clip(lower,start,end-1)-start)//chunks
        upper = (np.clip(upper,start,end-1)-start)//chunks
        for corner in itertools.product((0,1),repeat=3):
            ijk = [upper[:,a] if corner[a] else lower[:,a] for a in range(3)]
            keys = (ijk[0]*map_shape[1]+ijk[1])*map_shape[2]+ijk[2]
            touched.update(np.unique(keys).tolist())
    keys = np.asarray(sorted(touched),np.int64)
    indices = np.array(np.unravel_index(keys,tuple(map_shape))).T+start//chunks
    return pack_chunks(reader,indices)


def pack_chunks(reader, indices):
    """Common deterministic packed-chunk source and lookup table for experiments."""
    indices = np.asarray(sorted(set(map(tuple,indices))),np.int64)
    if not len(indices):
        raise ValueError("source plan has no in-volume chunks")
    chunks = np.asarray(reader.chunks,np.int64)
    first = indices.min(axis=0)
    start = first*chunks
    map_shape = indices.max(axis=0)-first+1
    end = np.minimum(reader.shape,start+map_shape*chunks)
    mapping = np.full(tuple(map_shape),-1,np.int32)
    mapping[tuple((indices-first).T)] = np.arange(len(indices),dtype=np.int32)
    origins = indices*chunks
    regions = [(tuple(origin),tuple(chunks)) for origin in origins]

    def read():
        data = np.empty((len(indices),*chunks),np.uint8)
        reader.read_many(regions,out=list(data))
        return data

    return start,end-start,mapping,read,len(indices)


def stats(values):
    return dict(mean_ms=float(np.mean(values)), p50_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values,95)), min_ms=min(values), max_ms=max(values), ms=values)


def exact(actual, expected):
    # Enforce bitwise identity for finite values, including signed zero; NaNs may
    # have a different payload, but must occur at exactly the same positions.
    actual, expected = actual.reshape(-1), expected.reshape(-1)
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError("output dtype/shape differs")
    nan = np.isnan(expected)
    if not np.array_equal(nan, np.isnan(actual)) or actual[~nan].tobytes() != expected[~nan].tobytes():
        delta = np.abs(actual.astype(float)-expected.astype(float))
        raise AssertionError(f"CPU/CUDA output differs: max abs {np.nanmax(delta)}")


def compare(actual, expected, allow_fma):
    actual, expected = actual.reshape(-1), expected.reshape(-1)
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError("output dtype/shape differs")
    if not allow_fma or expected.dtype == np.uint8:
        exact(actual,expected)
        return dict(max_abs=0.,changed=0)
    else:
        # uint8 CT is on a 0..255 scale. No lowered precision or fast math.
        atol,rtol = (3.1e-5,1e-6) if expected.dtype == np.float32 else (2e-9,1e-12)
        delta = np.abs(actual.astype(float)-expected)
        maximum = float(np.max(delta,initial=0))
        if not np.isfinite(maximum) or maximum > atol:
            np.testing.assert_allclose(actual,expected,atol=atol,rtol=rtol,equal_nan=True)
        finite = np.isfinite(delta)
        return dict(max_abs=float(np.max(delta,where=finite,initial=0)),
                    changed=int(np.count_nonzero((delta!=0)&finite)))


def write_validation_array(path):
    shape, chunks = (9, 10, 11), (4, 5, 3)
    (path/".zarray").write_text(json.dumps(dict(zarr_format=2, shape=shape, chunks=chunks,
        dtype="|u1", compressor=None, fill_value=37, order="C", filters=None)))
    rng = np.random.default_rng(12)
    for idx in itertools.product(range(3), range(2), range(4)):
        if idx == (1, 1, 1):  # missing chunk, nonzero fill
            continue
        (path/".".join(map(str,idx))).write_bytes(rng.integers(0,256,chunks,dtype=np.uint8).tobytes())
    return shape,chunks


def validate(binary, allow_fma=False):
    """Untimed contract checks; performance runs below always use real CT."""
    with tempfile.TemporaryDirectory(prefix="vc-sample-check-") as tmp:
        path = Path(tmp)
        shape,chunks = write_validation_array(path)
        rng = np.random.default_rng(12)
        with ZarrReader.open(tmp, threads=2) as reader:
            source = reader.read((0,0,0), shape)
            points = np.concatenate([rng.uniform(-1,12,(3000,3)),
                np.asarray(list(itertools.product([-1.,-.5,0.,.5,1.,3.,3.5,4.,8.5,10.,11.],repeat=3))),
                [[np.nan,1,2], [1,np.inf,2], [1,2,-np.inf], [1e300,1,2]]])
            grid = (np.array([7,8,9],np.int32), np.array([8.5,-.5,2.]),
                    np.array([[-.8,0,.2],[.1,1,0],[0,.2,.9]]))
            for pts, g in [(points,None), (None,grid)]:
                pts,g = rebase(pts,g,np.zeros(3,dtype=np.int64))
                count = len(pts) if g is None else int(np.prod(g[0]))
                cuda = CUDA(binary,shape,(0,0,0),shape,count,37)
                try:
                    for order, dtype in [(0,np.uint8),(1,np.float32)]:
                        expected = cpu_float32(reader,pts,g,(0,0,0),order)
                        actual,_ = cuda.run(source,pts,g,order,dtype,3)
                        compare(actual,expected,allow_fma)
                        repeated,_ = cuda.run(source,pts,g,order,dtype,0)
                        exact(repeated,actual)
                        cuda.run(source,pts,g,order,dtype,0,True)
                        exact(cuda.download(order,dtype),actual)
                    # Exercise chunk crossings and an absent lookup-table entry.
                    indices = [idx for idx in itertools.product(range(3),range(2),range(4)) if idx != (1,1,1)]
                    mapping = np.full((3,2,4),-1,np.int32)
                    for i,idx in enumerate(indices): mapping[idx] = i
                    packed = np.stack(reader.read_many([(tuple(np.array(idx)*chunks),chunks) for idx in indices]))
                    cuda.chunks(chunks,mapping,len(indices))
                    for order,dtype in [(0,np.uint8),(1,np.float32)]:
                        expected = cpu_float32(reader,pts,g,(0,0,0),order)
                        actual,_ = cuda.run(packed,pts,g,order,dtype,3)
                        compare(actual,expected,allow_fma)
                finally:
                    cuda.close()
    print(f"validation: float32 nearest/linear, boundaries, ties, missing chunks, nonfinite, affine, exact replay; allow_fma={allow_fma}",flush=True)


def workloads(start, edge, scale, only):
    # Reuse the established sampler workload definitions.
    for name, points in point_sets(start,edge,scale).items():
        if not only or name in only:
            yield name, np.ascontiguousarray(points), None
    for size in (64,128,256):
        name = f"affine{size}"
        if not only or name in only:
            angle = .35
            axes = np.array([[1.,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
            origin = np.array(start,dtype=float)+edge/2-axes @ np.full(3,size/2)+.25
            yield name,None,(np.full(3,size,np.int32),origin,axes)
    name = "plane"
    if not only or name in only:
        side,origin,u,v,w = patch_geometry(start,edge,scale)
        axes = np.ascontiguousarray(np.column_stack((.5*w,.73*v,.73*u)))
        yield name,None,(np.array([16,side,side],np.int32),origin,axes)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("binary",type=Path)
    ap.add_argument("--array",default=DEFAULT_ARRAY)
    ap.add_argument("--start",type=int,nargs=3,default=REGION[0])
    ap.add_argument("--edge",type=int,default=REGION[1])
    ap.add_argument("--scale",type=float,default=1.)
    ap.add_argument("--only",default="")
    ap.add_argument("--reps",type=int,default=11)
    ap.add_argument("--warmup",type=int,default=3)
    ap.add_argument("--threads",type=int,default=16)
    ap.add_argument("--validate-only",action="store_true")
    ap.add_argument("--allow-fma",action="store_true",help="validate with documented float tolerances instead of exact bits")
    ap.add_argument("--layout",choices=["box","chunks"],default="box")
    ap.add_argument("--modes",default="cpu,upload,resident,read_upload,upload_device,resident_device,read_upload_device")
    ap.add_argument("--outputs",default="nearest,float32")
    ap.add_argument("--output",type=Path,default=Path("/tmp/cuda-sample.json"))
    args = ap.parse_args(argv)
    if args.reps < 1 or args.warmup < 1 or args.scale <= 0:
        ap.error("reps, warmup and scale must be positive")
    only = set(filter(None,args.only.split(",")))
    if only - {"random","patch","local","plane","affine64","affine128","affine256"}:
        ap.error("unknown workload")
    modes_selected = args.modes.split(",")
    if not modes_selected or set(modes_selected)-{"cpu","cpu64","upload","resident","read_upload","upload_device","resident_device","read_upload_device"}:
        ap.error("unknown timing mode")
    output_types = {"nearest":(0,np.uint8),"float32":(1,np.float32)}
    if not args.outputs or set(args.outputs.split(","))-output_types.keys():
        ap.error("outputs must be nearest,float32")
    validate(args.binary,args.allow_fma)
    if args.validate_only:
        return
    from vc_reader import _native
    report = dict(command=[sys.executable,*sys.argv], environment=environment(),
        array=args.array,start=args.start,edge=args.edge,threads=args.threads,
        reps=args.reps,warmup=args.warmup,scale=args.scale,
        native=str(_native.__file__),native_sha256=hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        binary=str(args.binary.resolve()),binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],text=True).strip(),
        allow_fma=args.allow_fma,layout=args.layout,measured_modes=modes_selected,
        protocol="warm mmap; fixed seed shuffled mode order; synchronized outputs; reusable GPU allocations; uint8 source, rebased float32 geometry/interpolation; every timed result checked; no normalization/augmentation",
        modes=dict(cpu="float32 native reader planner; experimental entry", cpu64="unchanged public float64 sampler arithmetic, cast to same output dtype", upload="host source box + points upload, kernel, D2H",
                   resident="reused GPU source, points upload, kernel, D2H", read_upload="reader.read of known box + upload + kernel + D2H",
                   kernel="CUDA events, excludes transfers/allocation/reader",
                   device_suffix="same work but output stays on GPU; synchronization timed, verification D2H outside timing"),
        exclusions="source-bound computation (separately measured), CUDA context/allocation, cold/remote IO, GPU eviction/chunk management, downstream model; resident source upload amortized, points uploaded every call",
        rows=[])
    shuffle = np.random.default_rng(912)
    with ZarrReader.open(args.array,threads=args.threads,memory_cache=0) as reader:
        if reader.dtype != np.dtype(np.uint8) or reader.ndim != 3:
            ap.error("prototype supports only 3-D uint8 CT")
        if any(n > np.iinfo(np.int32).max for n in reader.shape):
            ap.error("prototype dimensions must fit int32")
        for name, points, grid in workloads(args.start,args.edge,args.scale,only):
            t = time.perf_counter()
            lo,shape = bounds(points,grid,reader.shape)
            mapping,chunk_count = None,None
            read = lambda: reader.read(lo,shape)
            if args.layout == "chunks":
                lo,shape,mapping,read,chunk_count = chunk_source(reader,points,grid)
            planning_ms = (time.perf_counter()-t)*1000
            source = read()
            source_hash = digest(source)
            count = len(points) if grid is None else int(np.prod(grid[0]))
            t = time.perf_counter()
            local_points,local_grid = rebase(points,grid,lo)
            conversion_ms = (time.perf_counter()-t)*1000
            cuda = CUDA(args.binary,shape,lo,reader.shape,count,int(reader.fill_value))
            try:
                if mapping is not None: cuda.chunks(reader.chunks,mapping,chunk_count)
                for order,dtype in [output_types[key] for key in args.outputs.split(",")]:
                    def cpu():
                        return cpu_float32(reader,local_points,local_grid,lo,order)
                    def cpu64():
                        return reader.sample(points,order=order,dtype=dtype) if grid is None else reader.read_affine(*grid,order=order,dtype=dtype)
                    expected = cpu()
                    old_reference = cpu64()
                    precision = precision_error(expected,old_reference)
                    cuda.run(source,local_points,local_grid,order,dtype,3)
                    times = {mode:[] for mode in modes_selected}
                    kernels = {mode:[] for mode in times if mode not in ("cpu","cpu64")}
                    differences = dict(max_abs=0.,changed=0)
                    for rep in range(-args.warmup,args.reps):
                        modes = list(times); shuffle.shuffle(modes)
                        for mode in modes:
                            t = time.perf_counter()
                            if mode == "cpu":
                                out = cpu()
                            elif mode == "cpu64":
                                out = cpu64()
                            else:
                                data = read() if mode.startswith("read_upload") else source
                                out,ms = cuda.run(data,local_points,local_grid,order,dtype,2 if mode.startswith("resident") else 3,
                                                  mode.endswith("_device"))
                            elapsed = (time.perf_counter()-t)*1000
                            if mode.endswith("_device"):
                                out = cuda.download(order,dtype)
                            diff = compare(out,old_reference if mode=="cpu64" else expected,args.allow_fma)
                            for key in differences: differences[key] = max(differences[key],diff[key])
                            if rep >= 0:
                                times[mode].append(elapsed)
                                if mode not in ("cpu","cpu64"): kernels[mode].append(ms)
                            del out
                            if mode not in ("cpu","cpu64"): del data
                    row = dict(workload=name,order=order,dtype=np.dtype(dtype).name,points=count,
                        source_start=lo.tolist(),source_shape=shape.tolist(),source_bytes=source.nbytes,
                        point_bytes=0 if local_points is None else local_points.nbytes,bounds_ms=planning_ms,
                        conversion_ms=conversion_ms,precision_vs_float64=precision,
                        chunks=chunk_count,table_bytes=0 if mapping is None else mapping.nbytes,
                        source_hash=source_hash,output_hash=digest(expected),differences=differences,
                        timings={mode:stats(v) for mode,v in times.items()},
                        kernels={mode:stats(v) for mode,v in kernels.items()})
                    report["rows"].append(row)
                    print(f"{name:10} {order} {np.dtype(dtype).name:7} source {source.nbytes/2**20:7.2f} MiB " +
                          " ".join(f"{k}={np.median(v):.3f}" for k,v in times.items()),flush=True)
                    args.output.parent.mkdir(parents=True,exist_ok=True)
                    args.output.write_text(json.dumps(report,indent=2)+"\n")
                assert digest(source) == source_hash
            finally:
                cuda.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
