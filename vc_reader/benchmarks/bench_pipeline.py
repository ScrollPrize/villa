"""Benchmark volume reading/sampling, or the full CPU/CUDA training pipeline.

The readers suite compares vc_reader with TensorStore and zarr-python + SciPy.
The pipeline suite adds normalization and transform using vc_reader only.
Run --help for workload, real-data, synthetic-input, cache, and output options.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

from extension_build import load_extensions


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("readers", "pipeline"), default="readers")
    parser.add_argument("--baselines", nargs="*", choices=("tensorstore", "zarr"), default=["tensorstore", "zarr"],
                        help="readers suite only; pass without names for vc_reader only")
    parser.add_argument("--workloads", nargs="+", choices=("ml_batch", "plane", "surface", "slab", "rotated_box"),
                        default=["ml_batch", "plane", "slab", "rotated_box"])
    parser.add_argument("--surface", type=Path, default=Path(__file__).resolve().parent / "fixtures/fiber_132487",
                        help="tifxyz directory for the surface workload (default: bundled PHercParis4 strip)")
    parser.add_argument("--surface-size", type=positive, nargs=2, default=(96, 512), metavar=("H", "W"),
                        help="surface tile in full-resolution UV pixels, upsampled according to meta.json scale")
    parser.add_argument("--surface-coordinate-scale", type=float, default=1.,
                        help="multiply stored XYZ to map to array voxels; bundled fixture uses level-2 voxels")
    parser.add_argument("--batch-size", type=positive, default=8, help="crops per ML request")
    parser.add_argument("--plane-size", type=positive, nargs=2, default=(256, 256), metavar=("H", "W"))
    parser.add_argument("--slab-depth", type=positive, default=16)
    parser.add_argument("--transform", choices=("light", "heavy"), default="heavy", help="pipeline suite only")
    parser.add_argument("--pipeline-read", choices=("affine", "box"), default="affine",
                        help="pipeline suite: affine sampling or integer axis-aligned box reads")
    parser.add_argument("--include-hybrid", action="store_true",
                        help="pipeline suite: also read/sample on CPU, upload, then normalize/augment on CUDA")
    parser.add_argument("--array", help="local/HTTP/S3 Zarr v2 array; omitted: temporary seeded uint8 input")
    parser.add_argument("--level", help="array level within a Zarr group")
    parser.add_argument("--backend", choices=("cpu", "cuda", "both"), default="cpu")
    parser.add_argument("--shape", type=positive, nargs=3, default=(128, 128, 128), metavar=("Z", "Y", "X"))
    parser.add_argument("--start", type=float, nargs=3, help="first crop origin in source ZYX coordinates; default: centered")
    parser.add_argument("--crops", type=positive, default=4, help="seeded requests per repetition (each ML request is a batch)")
    parser.add_argument("--jitter", type=float, default=32, help="origin jitter in voxels, clipped to in-bounds origins")
    parser.add_argument("--angle", type=float, default=10, help="rotation degrees: two coordinate planes in readers suite, YX in pipeline suite")
    parser.add_argument("--normalization", choices=("minmax", "zscore", "percentile", "robust"), default="minmax")
    parser.add_argument("--reps", type=positive, default=10)
    parser.add_argument("--warmup", type=positive, default=2, help="untimed passes over all crops, after validation")
    parser.add_argument("--threads", type=positive, default=1, help="reader worker threads")
    parser.add_argument("--seed", type=nonnegative, default=123)
    parser.add_argument("--memory-cache-mib", type=nonnegative, default=256)
    parser.add_argument("--cuda-cache-mib", type=positive, default=512)
    parser.add_argument("--cuda-cache", choices=("warm", "clear"), default="warm",
                        help="reuse device chunks or clear before each crop; host/OS caches stay warm")
    parser.add_argument("--cuda-output", choices=("host", "device"), default="host",
                        help="host includes final download; device leaves output ready for a GPU consumer")
    parser.add_argument("--device", type=nonnegative, default=0)
    parser.add_argument("--build-dir", type=Path, help="use both extensions from an existing CMake build, without installing")
    parser.add_argument("--output", type=Path, help="write complete JSON report, including raw measurements")
    args = parser.parse_args()
    if not np.isfinite(args.jitter) or args.jitter < 0 or not np.isfinite(args.angle):
        parser.error("jitter must be finite and nonnegative; angle must be finite")
    if args.start is not None and not np.isfinite(args.start).all():
        parser.error("start must be finite")
    if args.level is not None and args.array is None:
        parser.error("--level requires --array")
    if (args.include_hybrid or args.pipeline_read != "affine") and args.suite != "pipeline":
        parser.error("--include-hybrid and --pipeline-read apply to --suite pipeline")
    if args.include_hybrid and args.backend == "cpu":
        parser.error("--include-hybrid requires --backend cuda or both")
    if not np.isfinite(args.surface_coordinate_scale) or args.surface_coordinate_scale <= 0:
        parser.error("--surface-coordinate-scale must be finite and positive")
    if args.suite == "readers" and "surface" in args.workloads and args.array is None:
        parser.error("the surface workload requires --array matching the tifxyz coordinates")
    return args


def synthetic_array(path, shape, seed):
    """Write raw Zarr chunks with NumPy/stdlib only; generation is not timed."""
    path.mkdir()
    chunk = 32
    edge = ((2 * max(shape) + chunk - 1) // chunk + 1) * chunk
    metadata = dict(zarr_format=2, shape=[edge] * 3, chunks=[chunk] * 3,
                    dtype="|u1", compressor=None, fill_value=0, order="C", filters=None)
    (path / ".zarray").write_text(json.dumps(metadata))
    rng = np.random.default_rng(seed)
    for index in itertools.product(range(edge // chunk), repeat=3):
        rng.integers(0, 256, (chunk,) * 3, dtype=np.uint8).tofile(path / ".".join(map(str, index)))
    return str(path)


def grids(args, source_shape, *, axis_aligned=False):
    angle = 0. if axis_aligned else np.deg2rad(args.angle)
    axes = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]], dtype=np.float32)
    corners = np.array(list(itertools.product(*[(0, n - 1) for n in args.shape]))) @ axes.T
    lower = -corners.min(axis=0)
    upper = np.asarray(source_shape) - (1 if axis_aligned else 1.001) - corners.max(axis=0)
    if np.any(upper < lower):
        raise ValueError("crop does not fit inside the input array")
    base = (lower + upper) / 2 if args.start is None else np.asarray(args.start)
    if np.any(base < lower) or np.any(base > upper):
        raise ValueError("--start places the crop outside the input array")
    rng = np.random.default_rng(args.seed)
    origins = [base.tolist()]
    origins.extend(rng.uniform(np.maximum(lower, base - args.jitter),
                               np.minimum(upper, base + args.jitter), (args.crops - 1, 3)).tolist())
    if axis_aligned:
        origins = np.floor(origins).astype(np.int64).tolist()
    return origins, axes


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean_ms=float(values.mean()), p50_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), min_ms=float(values.min()), max_ms=float(values.max()))


def environment(args, *, transform=False):
    from vc_reader import _reader

    result = dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
                  processor=platform.processor(), cpu_count=os.cpu_count(),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  extensions={})
    modules = {"reader": _reader}
    if transform:
        from vc_reader import _transforms
        modules["transforms"] = _transforms
    for name, module in modules.items():
        path = Path(module.__file__)
        result["extensions"][name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    cache = args.build_dir / "CMakeCache.txt" if args.build_dir else None
    result["build"] = {"type": "unknown (use --build-dir to record CMake settings)"}
    if cache and cache.is_file():
        keys = ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_ARCHITECTURES", "VC_READER_ENABLE_CUDA")
        result["build"] = {line.split(":", 1)[0]: line.split("=", 1)[1]
                           for line in cache.read_text().splitlines()
                           if "=" in line and line.split(":", 1)[0] in keys}
    if args.backend in ("cuda", "both"):
        try:
            info = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=5, check=True)
            result["gpus"] = info.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            result["gpus"] = "nvidia-smi unavailable"
    return result


def reading_workloads(args, volume_shape):
    """Requests contain the same geometry for every reader; no data is preloaded."""
    angle = np.deg2rad(args.angle)
    c, s = np.cos(angle), np.sin(angle)
    rotation = (np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @
                np.array([[1, 0, 0], [0, c, -s], [0, s, c]])).astype(np.float32)
    result = {}
    for ordinal, name in enumerate(("ml_batch", "plane", "slab", "rotated_box")):
        if name not in args.workloads:
            continue
        rng = np.random.default_rng(args.seed + ordinal)
        requests = []
        for request in range(args.crops):
            batch = []
            for item in range(args.batch_size if name == "ml_batch" else 1):
                if name == "plane":
                    shape = (1, *args.plane_size)
                    axes = np.roll(rotation, request % 3, axis=0)
                elif name == "slab":
                    shape = tuple(np.roll((args.slab_depth, *args.plane_size), request % 3))
                    axes = np.eye(3, dtype=np.float32)
                else:
                    shape = tuple(args.shape)
                    axes = rotation if name == "rotated_box" else np.eye(3, dtype=np.float32)
                corners = np.array(list(itertools.product(*[(0, n - 1) for n in shape]))) @ axes.T
                sampled = name in ("plane", "rotated_box")
                lower = -corners.min(axis=0)
                upper = np.asarray(volume_shape) - (1.001 if sampled else 1) - corners.max(axis=0)
                if np.any(lower > upper):
                    raise ValueError(f"{name} does not fit in the input; reduce --shape/--plane-size/--slab-depth")
                base = (lower + upper) / 2 if args.start is None else np.asarray(args.start)
                if np.any(base < lower) or np.any(base > upper):
                    raise ValueError(f"--start places {name} outside the volume")
                origin = base if request == item == 0 else rng.uniform(np.maximum(lower, base - args.jitter),
                                                                     np.minimum(upper, base + args.jitter))
                if not sampled:
                    origin = np.floor(origin).astype(np.int64)
                batch.append((tuple(int(n) for n in shape), origin.tolist(), axes))
            requests.append(batch)
        result[name] = requests
    return result


def sampling_box(grid):
    """Baseline ROI and coordinates using the C++ reader's float32 geometry.

    SciPy performs interpolation; coordinates are generated here to avoid
    comparing float64 affine geometry against vc_reader's float32 contract.
    Planning/coordinate allocation are included in the baseline timing.
    """
    from vc_reader.reader._sampling import affine32

    shape, base, local, axes = affine32(*grid)
    z = np.arange(shape[0], dtype=np.float32)[:, None, None]
    y = np.arange(shape[1], dtype=np.float32)[None, :, None]
    x = np.arange(shape[2], dtype=np.float32)[None, None, :]
    coords = np.empty((3, *shape), dtype=np.float32)
    for axis in range(3):
        coords[axis] = ((local[axis] + axes[axis, 0] * z) + axes[axis, 1] * y) + axes[axis, 2] * x
    return coordinate_box(coords, base)


def coordinate_box(coords, base):
    """Enclosing read and local SciPy coordinates, preserving float32 fractions."""
    lo = np.floor(coords.reshape(3, -1).min(axis=1)).astype(np.int64)
    hi = np.ceil(coords.reshape(3, -1).max(axis=1)).astype(np.int64) + 1
    # Translating float32 coordinates in float64 preserves the fractions exactly.
    relative = coords.astype(np.float64) - lo.reshape((3,) + (1,) * (coords.ndim - 1))
    return tuple(np.asarray(base) + lo), tuple(hi - lo), relative


def surface_workload(args, volume_shape):
    """Select real curved UV patches using the library's bounded geometry reader."""
    from vc_reader import Affine, Storage, Surface
    from vc_reader.surface import SurfaceGeometry

    path = args.surface
    meta = json.loads((path / "meta.json").read_text())
    surface = Surface(str(path), Storage(str(path), "tifxyz"), "PHercParis4", "20260411134726",
                      axes=("y", "x"), component_order=("x", "y", "z"),
                      xyz_to_volume=Affine.scale(4), spacing_um=(9.6,)*3)
    scale = np.asarray(meta["scale"], dtype=np.float32)
    with SurfaceGeometry(surface) as geometry:
        # tifxyz scale is XY; public patch dimensions and starts are YX.
        extent = np.floor((np.asarray(geometry.geometry.shape)-1)/scale[::-1]).astype(int)+1
        size = np.asarray(args.surface_size)
        if np.any(size > extent):
            raise ValueError(f"--surface-size does not fit supported UV extent {extent.tolist()}")
        row = int((extent[0]-size[0])//2)
        candidates = np.unique(np.rint(np.linspace(0, extent[1]-size[1], max(64,args.crops*8))).astype(int))
        tiles = []
        for col in candidates:
            _, valid, _ = geometry.read_points((row,int(col)),tuple(size))
            if valid.all():
                tiles.append(int(col))
        if len(tiles) < args.crops:
            raise ValueError(f"only {len(tiles)} valid surface tiles found; reduce --surface-size/--crops")
        selected = np.rint(np.linspace(0,len(tiles)-1,args.crops)).astype(int)
        requests, descriptions = [], []
        for index in selected:
            col = tiles[index]
            points, _, xyz_base = geometry.read_points((row,col),tuple(size))
            # Keep coordinates local before applying a requested volume-level scale.
            from vc_reader.surface import _transform
            local, base = _transform(points[0], Affine.scale(args.surface_coordinate_scale), xyz_base)
            local = np.ascontiguousarray(local[...,::-1]); base = tuple(reversed(base))
            shape = tuple(size)
            lower = local.reshape(-1,3).min(axis=0)
            upper = local.reshape(-1,3).max(axis=0)
            if any(lo+b<0 or hi+b>n-1 for lo,hi,b,n in zip(lower,upper,base,volume_shape)):
                raise ValueError("surface coordinates fall outside the selected volume")
            requests.append([(shape,base,local)])
            descriptions.append(dict(uv_start=[row,col],shape=list(shape),coordinate_origin=base,
                local_coordinates_sha256=hashlib.sha256(local.tobytes()).hexdigest(),
                bounds_zyx=[[float(v)+b for v,b in zip(lower,base)],[float(v)+b for v,b in zip(upper,base)]]))
        stored_shape = geometry.geometry.shape
    filenames = ["meta.json","x.tif","y.tif","z.tif"] + (["mask.tif"] if (path/"mask.tif").exists() else [])
    return requests, dict(path=str(path),uuid=meta.get("uuid"),stored_shape=stored_shape,
        scale_xy=scale.tolist(),voxel_size_um=meta.get("voxel_size_um"),
        coordinate_scale=args.surface_coordinate_scale,layers=1,
        preparation="shared bounded geometry reader; excludes geometry preparation from volume sampling timings",
        files_sha256={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in filenames},tiles=descriptions)


class BaselineReader:
    """Async TensorStore reads or concurrent zarr-python reads, then shared SciPy sampling."""
    def __init__(self, name, location, args, stack):
        self.name = name
        self.pool = None
        if name == "tensorstore":
            import tensorstore as ts

            if location.startswith("s3://"):
                bucket, _, key = location[5:].partition("/")
                kvstore = dict(driver="s3", bucket=bucket, path=key.rstrip("/") + "/")
            elif location.startswith(("http://", "https://")):
                kvstore = dict(driver="http", base_url=location.rstrip("/") + "/")
            else:
                kvstore = dict(driver="file", path=str(Path(location).resolve()) + "/")
            self.array = ts.open(dict(driver="zarr", kvstore=kvstore,
                                      context={"cache_pool": {"total_bytes_limit": args.memory_cache_mib << 20},
                                               "data_copy_concurrency": {"limit": args.threads},
                                               "file_io_concurrency": {"limit": args.threads}},
                                      recheck_cached_data=False), read=True).result()
        else:
            import zarr

            stack.enter_context(zarr.config.set({"async.concurrency": args.threads, "threading.max_workers": args.threads}))
            self.array = zarr.open_array(location, mode="r", zarr_format=2)
            stack.callback(self.array.store.close)
            self.pool = stack.enter_context(ThreadPoolExecutor(max_workers=args.threads))

    def read(self, grids, sampling):
        from scipy.ndimage import map_coordinates

        start = time.perf_counter_ns()
        plans = [(coordinate_box(np.moveaxis(g[2], -1, 0), g[1]) if sampling == "surface" else sampling_box(g))
                 if sampling else (g[1], g[0], None) for g in grids]
        slices = [tuple(slice(int(s), int(s + n)) for s, n in zip(origin, shape)) for origin, shape, _ in plans]
        planned = time.perf_counter_ns()
        if self.name == "tensorstore":
            futures = [self.array[index].read() for index in slices]
            blocks = [future.result() for future in futures]
        else:
            blocks = list(self.pool.map(self.array.__getitem__, slices))
        loaded = time.perf_counter_ns()
        outputs = [map_coordinates(block, plan[2], output=np.float32, order=1, prefilter=False, mode="nearest")
                   for block, plan in zip(blocks, plans)] if sampling else blocks
        done = time.perf_counter_ns()
        return outputs, dict(planning=(planned-start)/1e6, read=(loaded-planned)/1e6,
                             sampling=(done-loaded)/1e6, read_sample=(done-start)/1e6,
                             download=0., total=(done-start)/1e6)


def reading_benchmark(args):
    if args.build_dir:
        load_extensions(args.build_dir.resolve())
    from vc_reader import cuda
    from vc_reader.reader import ZarrReader

    modes = (["cpu", "cuda"] if args.backend == "both" else [args.backend]) + list(dict.fromkeys(args.baselines))
    if "cuda" in modes and not cuda.is_available():
        raise ValueError("CUDA reader unavailable; use a CUDA build and accessible device")
    report = dict(schema_version=2, suite="readers", command=[sys.executable, *sys.argv],
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  environment=environment(args), workloads={},
                  protocol=dict(included="request planning, allocation, reads, decode, interpolation, final CUDA download when selected",
                                excluded="normalization, transform, setup, warmup, validation, destruction, cache clearing",
                                sampling="TensorStore/zarr read an enclosing ROI then SciPy order=1 samples matched float32 coordinates; vc_reader samples chunks directly",
                                batching="ML: one vc_reader read_many call, concurrent TensorStore futures, zarr thread-pool reads; no pre-stacking",
                                order="cyclic rotation of backend order each repetition",
                                caches="warm OS cache; vc_reader/TensorStore use requested decoded-cache budgets; zarr has no added decoded cache",
                                endpoints="all outputs on host except when --cuda-output=device",
                                throughput="output voxels/bytes, not bytes fetched from storage",
                                speedup="baseline p50 / vc_reader p50; >1 means vc_reader is faster"))
    for package in ("tensorstore", "zarr", "scipy", "numcodecs", "tifffile"):
        try:
            report["environment"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["environment"][package] = None
    if args.baselines:
        # Fail before data generation, rather than silently omit a baseline.
        import scipy.ndimage
    with ExitStack() as stack:
        location = args.array
        if location is None:
            work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="vc-readers-")))
            size = max(*args.shape, *args.plane_size, args.slab_depth)
            location = synthetic_array(work / "input.zarr", (size,) * 3, args.seed)
        location = location.rstrip("/") + "/" + args.level if args.level is not None else location
        host = stack.enter_context(ZarrReader.open(location, threads=args.threads, memory_cache=args.memory_cache_mib << 20))
        if host.ndim != 3 or host.dtype != np.uint8:
            raise ValueError("reader comparison requires a 3-D uint8 Zarr array")
        readers = {"cpu": host}
        if "cuda" in modes:
            readers["cuda"] = stack.enter_context(host.cuda(cache_bytes=args.cuda_cache_mib << 20, device=args.device))
        for mode in args.baselines:
            readers[mode] = BaselineReader(mode, location, args, stack)
            if tuple(readers[mode].array.shape) != tuple(host.shape):
                raise ValueError(f"{mode} opened a different array shape")
        workloads = reading_workloads(args, host.shape)
        if "surface" in args.workloads:
            workloads["surface"], report["surface"] = surface_workload(args, host.shape)
        workloads = {name: workloads[name] for name in dict.fromkeys(args.workloads)}
        report["input"] = dict(location=args.array or "synthetic seeded uniform uint8", level=args.level,
                               shape=list(host.shape), chunks=list(host.chunks), dtype=str(host.dtype),
                               compressor=host.meta.compressor)

        def run(mode, grids, sampling):
            if mode in args.baselines:
                return readers[mode].read(grids, sampling)
            reader = readers[mode]
            if mode == "cuda" and args.cuda_cache == "clear":
                reader.clear_cache()
            kwargs = {"source": "chunks"} if mode == "cuda" else {}
            start = time.perf_counter_ns()
            if sampling == "surface":
                out = [reader.sample(g[2], coordinate_origin=g[1], order=1, dtype=np.float32, **kwargs) for g in grids]
            elif sampling:
                out = reader.read_affine_many(grids, order=1, dtype=np.float32, **kwargs)
            else:
                out = reader.read_regions([(tuple(int(v) for v in g[1]), g[0]) for g in grids], **kwargs)
            ready = time.perf_counter_ns()
            if mode == "cuda" and args.cuda_output == "host":
                out = [a.numpy() for a in out]
            done = time.perf_counter_ns()
            return out, dict(read_sample=(ready-start)/1e6, download=(done-ready)/1e6 if mode == "cuda" and args.cuda_output == "host" else 0.,
                             total=(done-start)/1e6)

        for name, requests in workloads.items():
            sampling = "surface" if name == "surface" else "affine" if name in ("plane", "rotated_box") else None
            geometry = report["surface"]["tiles"] if sampling == "surface" else [[(shape, origin, axes.tolist()) for shape, origin, axes in batch] for batch in requests]
            row = dict(requests=geometry, sampling=sampling,
                       dtype="float32" if sampling else "uint8", validation=dict(atol=5e-5, rtol=1e-5, requests=[]), results={})
            for index, grids in enumerate(requests):
                outputs = {}
                for mode in modes:
                    arrays, _ = run(mode, grids, sampling)
                    outputs[mode] = [a.numpy() if isinstance(a, cuda.DeviceArray) else a for a in arrays]
                    for a, grid in zip(outputs[mode], grids):
                        if a.shape != grid[0] or a.dtype != np.dtype(row["dtype"]) or not np.isfinite(a).all():
                            raise ValueError(f"invalid output: {name}/{mode}")
                reference = modes[0]
                validation = dict(request=index, reference=reference, comparisons={})
                if sampling == "surface":
                    validation["reference_range"] = [float(outputs[reference][0].min()), float(outputs[reference][0].max())]
                for mode in modes[1:]:
                    errors = []
                    for a, b in zip(outputs[mode], outputs[reference]):
                        if sampling:
                            np.testing.assert_allclose(a, b, atol=5e-5, rtol=1e-5)
                        else:
                            np.testing.assert_array_equal(a, b)
                        errors.append(float(np.abs(a.astype(np.float64) - b).max()))
                    validation["comparisons"][mode] = dict(max_abs=max(errors))
                row["validation"]["requests"].append(validation)
            for _ in range(args.warmup):
                for grids in requests:
                    for mode in modes:
                        out, _ = run(mode, grids, sampling)
                        del out
            records = {mode: [] for mode in modes}
            before = dict(host=host.stats(), cuda=readers["cuda"].stats if "cuda" in modes else None)
            for rep in range(args.reps):
                order = modes[rep % len(modes):] + modes[:rep % len(modes)]
                for index, grids in enumerate(requests):
                    for mode in order:
                        out, times = run(mode, grids, sampling)
                        records[mode].append(dict(repetition=rep, request=index, **times))
                        del out
            row["cache_stats"] = dict(before=before, after=dict(host=host.stats(), cuda=readers["cuda"].stats if "cuda" in modes else None))
            voxels = sum(int(np.prod(g[0])) for batch in requests for g in batch) * args.reps
            crops = sum(len(batch) for batch in requests) * args.reps
            for mode, samples in records.items():
                stages = [key for key in samples[0] if key not in ("repetition", "request")]
                seconds = sum(s["total"] for s in samples) / 1000
                row["results"][mode] = dict(timings={stage: summary([s[stage] for s in samples]) for stage in stages},
                                           raw_ms=samples, samples=len(samples), requests_per_second=len(samples)/seconds,
                                           crops_per_second=crops/seconds, mvoxels_per_second=voxels/seconds/1e6,
                                           output_gib_per_second=voxels*np.dtype(row["dtype"]).itemsize/seconds/(1<<30))
            row["speedups"] = {f"{baseline}_over_{backend}": row["results"][baseline]["timings"]["total"]["p50_ms"] /
                                row["results"][backend]["timings"]["total"]["p50_ms"]
                                for backend in ("cpu", "cuda") if backend in modes
                                for baseline in modes if baseline != backend}
            report["workloads"][name] = row
            print(f"{name}: validation passed; " + ", ".join(f"{mode}={row['results'][mode]['timings']['total']['p50_ms']:.3f} ms" for mode in modes), flush=True)
    return report


def run_crop(reader, normalization, transform, origin, axes, args, seed, cuda=False, source="chunks"):
    # Cache clearing and output destruction are outside the timed interval.
    if cuda and args.cuda_cache == "clear":
        reader.clear_cache()
    start = time.perf_counter_ns()
    kwargs = {"source": source} if cuda else {}
    if args.pipeline_read == "box":
        # source='cpu' reads uint8 crops, uploads only those bytes, then casts
        # to float32 on-device inside the public CUDA reader.
        image = reader.read_region(tuple(origin), args.shape, dtype=np.float32, **kwargs)
    else:
        image = reader.read_affine(args.shape, origin, axes, order=1, dtype=np.float32, **kwargs)
    image = image.reshape(1, *args.shape)
    sampled = time.perf_counter_ns()
    normalized = normalization({"image": image}, seed=seed)["image"]
    norm_done = time.perf_counter_ns()
    output = transform({"image": normalized}, seed=seed)["image"]
    augmented = time.perf_counter_ns()
    if cuda and args.cuda_output == "host":
        output = output.numpy()
    done = time.perf_counter_ns()
    times = dict(read_sample=(sampled - start) / 1e6, normalization=(norm_done - sampled) / 1e6,
                 transform=(augmented - norm_done) / 1e6, download=(done - augmented) / 1e6 if cuda and args.cuda_output == "host" else 0.,
                 total=(done - start) / 1e6)
    return output, times


def heavy_transforms():
    """Shared 12-transform workload for single-crop and PyTorch benchmarks."""
    from vc_reader import transforms as A
    return [A.Elastic(sigma=12, magnitude=2, border=True), A.MedianFilter(3),
            A.Flip(axes="x", p_axis=1), A.GaussianBlur(.7), A.BrightnessMultiply(1.1),
            A.BrightnessAdd(.03), A.Contrast(1.15), A.Gamma(.9),
            A.LocalContrast(12, loc=.5, factor=1.1),
            A.LocalSmoothing(12, loc=.5, strength=.3, sigma=1),
            A.Sharpen(.1, p_clamp=0), A.LowResolution(.85)]


def benchmark(args):
    if args.build_dir:
        load_extensions(args.build_dir.resolve())
    from vc_reader import transforms as A, cuda
    from vc_reader.reader import ZarrReader

    modes = ["cpu", "cuda"] if args.backend == "both" else [args.backend]
    if "cuda" in modes and not cuda.is_available():
        raise ValueError("CUDA reader unavailable; use a CUDA build and accessible device (no CPU fallback)")
    if args.include_hybrid:
        modes.append("hybrid")
    transforms = [A.Flip(axes="x", p_axis=1), A.BrightnessMultiply(1.1), A.Gamma(.9), A.GaussianBlur(1)]
    if args.transform == "heavy":
        transforms = heavy_transforms()
    pipelines = {mode: (A.Pipeline([A.Normalize(args.normalization)], backend="cpu" if mode == "cpu" else "cuda", compact=False),
                        A.Pipeline(transforms, backend="cpu" if mode == "cpu" else "cuda", compact=False)) for mode in modes}
    report = dict(schema_version=2, suite="pipeline", command=[sys.executable, *sys.argv],
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  environment=environment(args, transform=True),
                  transforms=[repr(t) for t in transforms],
                  protocol=dict(timing="synchronous wall time; fused read/sample, separate normalization and transform calls",
                                included="chunk I/O/decode, planning, allocations, sampling, normalization, transform; CUDA upload on cache misses",
                                excluded="input generation, reader/pipeline construction, validation, warmup, cache clearing, output destruction",
                                caches="host and OS caches are not flushed; warm CUDA cache may evict if its budget is too small",
                                order="backend order rotates each repetition; identical crops and seeds",
                                routes="cpu: CPU read/normalize/augment; cuda: device chunks; hybrid: CPU read/sample + upload, CUDA normalize/augment",
                                hybrid_upload="box: uint8 crop uploaded then cast on GPU; affine: CPU-sampled float32 crop uploaded",
                                throughput="output voxels and float32 output bytes, not storage or PCIe bandwidth",
                                speedup="CPU / CUDA p50 latency; >1 means CUDA is faster",
                                endpoints="CPU host; CUDA " + args.cuda_output),
                  validation=dict(atol=5e-5, rtol=1e-5, crops=[]), results={})
    with ExitStack() as stack:
        location = args.array
        if location is None:
            work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="vc-pipeline-")))
            location = synthetic_array(work / "input.zarr", args.shape, args.seed)
        host = stack.enter_context(ZarrReader.open(location, level=args.level, threads=args.threads,
                                                  memory_cache=args.memory_cache_mib << 20))
        if host.ndim != 3 or host.dtype != np.uint8:
            raise ValueError("this CPU/CUDA benchmark requires a 3-D uint8 array")
        readers = {"cpu": host}
        if "cuda" in modes:
            readers["cuda"] = stack.enter_context(host.cuda(cache_bytes=args.cuda_cache_mib << 20, device=args.device))
            if args.include_hybrid:
                readers["hybrid"] = readers["cuda"]
        origins, axes = grids(args, host.shape, axis_aligned=args.pipeline_read == "box")
        report["input"] = dict(location=args.array or "synthetic seeded uniform uint8 (temporary raw Zarr v2)",
                               shape=list(host.shape), chunks=list(host.chunks), dtype=str(host.dtype),
                               origins=origins, axes=axes.tolist())

        def run(mode, index):
            return run_crop(readers[mode], *pipelines[mode], origins[index], axes, args,
                            args.seed + index, cuda=mode != "cpu", source="cpu" if mode == "hybrid" else "chunks")

        # Validate before warmup; CUDA-only runs do not run a CPU reference pipeline.
        for index in range(args.crops):
            outputs = {}
            for mode in modes:
                output, _ = run(mode, index)
                actual = output.numpy() if isinstance(output, cuda.DeviceArray) else output
                if actual.shape != (1, *args.shape) or actual.dtype != np.float32 or not np.isfinite(actual).all():
                    raise ValueError(f"invalid {mode} output for crop {index}")
                outputs[mode] = actual
            row = dict(crop=index, sha256={mode: hashlib.sha256(a.tobytes()).hexdigest() for mode, a in outputs.items()})
            row["reference"] = modes[0]
            row["comparisons"] = {}
            for mode in modes[1:]:
                delta = outputs[mode].astype(np.float64) - outputs[modes[0]]
                row["comparisons"][mode] = dict(max_abs=float(np.abs(delta).max()), rmse=float(np.sqrt(np.mean(delta * delta))))
                np.testing.assert_allclose(outputs[mode], outputs[modes[0]], atol=5e-5, rtol=1e-5)
            if "cpu" in outputs and "cuda" in outputs:
                row.update(row["comparisons"]["cuda"])
            report["validation"]["crops"].append(row)
        report["validation"]["comparison"] = "/".join(modes) + " allclose" if len(modes) > 1 else "shape, dtype, finite values (single backend)"
        for _ in range(args.warmup):
            for index in range(args.crops):
                for mode in modes:
                    output, _ = run(mode, index)
                    del output
        before = dict(host=host.stats())
        if "cuda" in modes:
            before["cuda"] = readers["cuda"].stats
        records = {mode: [] for mode in modes}
        for rep in range(args.reps):
            order = modes[rep % len(modes):] + modes[:rep % len(modes)]
            for index in range(args.crops):
                for mode in order:
                    output, times = run(mode, index)
                    records[mode].append(dict(repetition=rep, crop=index, **times))
                    del output
        after = dict(host=host.stats())
        if "cuda" in modes:
            after["cuda"] = readers["cuda"].stats
        report["cache_stats"] = dict(before_measurement=before, after_measurement=after)
    voxels = int(np.prod(args.shape))
    for mode, rows in records.items():
        timings = {stage: summary([row[stage] for row in rows]) for stage in ("read_sample", "normalization", "transform", "download", "total")}
        seconds = sum(row["total"] for row in rows) / 1000
        report["results"][mode] = dict(timings=timings, samples=len(rows), raw_ms=rows,
                                       crops_per_second=len(rows) / seconds,
                                       mvoxels_per_second=len(rows) * voxels / seconds / 1e6,
                                       output_gib_per_second=len(rows) * voxels * 4 / seconds / (1 << 30))
    if "cpu" in modes and "cuda" in modes:
        report["speedup_cpu_over_cuda"] = {stage: report["results"]["cpu"]["timings"][stage]["p50_ms"] /
                                           report["results"]["cuda"]["timings"][stage]["p50_ms"]
                                           for stage in ("read_sample", "normalization", "transform", "total")}
    if "hybrid" in modes:
        report["speedup_over_hybrid"] = {mode: report["results"][mode]["timings"]["total"]["p50_ms"] /
                                         report["results"]["hybrid"]["timings"]["total"]["p50_ms"]
                                         for mode in modes if mode != "hybrid"}
    return report


def main():
    args = arguments()
    try:
        report = reading_benchmark(args) if args.suite == "readers" else benchmark(args)
    except (ImportError, AttributeError, RuntimeError, ValueError, AssertionError) as exc:
        raise SystemExit(f"Benchmark failed: {exc}\nUse matching reader/transform extensions; --build-dir can select an existing build.") from exc
    if args.suite == "readers":
        print("Workload     Backend       mean ms     p50 ms     p95 ms      req/s     Mvox/s")
        for name, workload in report["workloads"].items():
            for mode, result in workload["results"].items():
                t = result["timings"]["total"]
                print(f"{name:12} {mode:12} {t['mean_ms']:10.3f} {t['p50_ms']:10.3f} {t['p95_ms']:10.3f} "
                      f"{result['requests_per_second']:10.2f} {result['mvoxels_per_second']:10.2f}")
            print("Speedups (baseline/vc_reader): " + ", ".join(f"{key}={value:.2f}x" for key, value in workload["speedups"].items()))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"JSON: {args.output}")
        return
    print(f"Input: {report['input']['location']} | shape={args.shape}, crops={args.crops}, reps={args.reps}, warmup={args.warmup}, threads={args.threads}")
    print(f"Warm host/OS caches; CUDA cache={args.cuda_cache}, output={args.cuda_output}. Reading and sampling are fused.")
    print("Backend  Stage             mean ms     p50 ms     p95 ms     min ms     max ms")
    for mode, result in report["results"].items():
        for stage, values in result["timings"].items():
            print(f"{mode:8} {stage:16}" + "".join(f"{values[key]:11.3f}" for key in ("mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms")))
        print(f"{mode}: {result['crops_per_second']:.2f} crops/s, {result['mvoxels_per_second']:.2f} Mvox/s, {result['output_gib_per_second']:.3f} output GiB/s")
    if "speedup_cpu_over_cuda" in report:
        print("Speedup (CPU/CUDA p50): " + ", ".join(f"{stage}={ratio:.2f}x" for stage, ratio in report["speedup_cpu_over_cuda"].items()))
    if "speedup_over_hybrid" in report:
        print("Hybrid speedup (other/hybrid p50): " + ", ".join(f"{mode}={ratio:.2f}x" for mode, ratio in report["speedup_over_hybrid"].items()))
    print("Validation: " + report["validation"]["comparison"] + " passed")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
