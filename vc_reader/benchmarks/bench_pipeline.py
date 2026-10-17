"""Benchmark reading/sampling -> normalization -> augmentation on CPU and CUDA.

Requires vc-reader[reader,augment]; no benchmark-only dependencies. Run --help
for real-data, synthetic-input, cache, and output options. CUDA public calls are
synchronous. Reading and affine sampling are fused and timed together.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np


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
    parser.add_argument("--array", help="local/HTTP/S3 Zarr v2 array; omitted: temporary seeded uint8 input")
    parser.add_argument("--level", help="array level within a Zarr group")
    parser.add_argument("--backend", choices=("cpu", "cuda", "both"), default="cpu")
    parser.add_argument("--shape", type=positive, nargs=3, default=(128, 128, 128), metavar=("Z", "Y", "X"))
    parser.add_argument("--start", type=float, nargs=3, help="first affine origin in source ZYX coordinates; default: centered")
    parser.add_argument("--crops", type=positive, default=4, help="distinct seeded crop origins per repetition")
    parser.add_argument("--jitter", type=float, default=32, help="origin jitter in voxels, clipped to in-bounds origins")
    parser.add_argument("--angle", type=float, default=10, help="YX rotation in degrees for trilinear sampling")
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
    parser.add_argument("--native-dir", type=Path, help="use both extensions from an existing CMake build, without installing")
    parser.add_argument("--output", type=Path, help="write complete JSON report, including raw measurements")
    args = parser.parse_args()
    if not np.isfinite(args.jitter) or args.jitter < 0 or not np.isfinite(args.angle):
        parser.error("jitter must be finite and nonnegative; angle must be finite")
    if args.start is not None and not np.isfinite(args.start).all():
        parser.error("start must be finite")
    if args.level is not None and args.array is None:
        parser.error("--level requires --array")
    return args


def load_extensions(directory):
    # Keep this optional override in this file rather than retaining a runner.
    import importlib.machinery

    for name in ("_native", "_augment"):
        candidates = [directory / (name + suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise ValueError(f"no {name} extension in {directory}")
        spec = importlib.util.spec_from_file_location(f"vc_reader.{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[spec.name] = module


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


def grids(args, source_shape):
    angle = np.deg2rad(args.angle)
    axes = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]], dtype=np.float32)
    corners = np.array(list(itertools.product(*[(0, n - 1) for n in args.shape]))) @ axes.T
    lower = -corners.min(axis=0)
    upper = np.asarray(source_shape) - 1.001 - corners.max(axis=0)
    if np.any(upper < lower):
        raise ValueError("rotated crop does not fit inside the input array")
    base = (lower + upper) / 2 if args.start is None else np.asarray(args.start)
    if np.any(base < lower) or np.any(base > upper):
        raise ValueError("--start places the rotated crop outside the input array")
    rng = np.random.default_rng(args.seed)
    origins = [base.tolist()]
    origins.extend(rng.uniform(np.maximum(lower, base - args.jitter),
                               np.minimum(upper, base + args.jitter), (args.crops - 1, 3)).tolist())
    return origins, axes


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean_ms=float(values.mean()), p50_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), min_ms=float(values.min()), max_ms=float(values.max()))


def run_crop(reader, normalization, augmentation, origin, axes, args, seed, cuda=False):
    # Cache clearing and output destruction are outside the timed interval.
    if cuda and args.cuda_cache == "clear":
        reader.clear_cache()
    start = time.perf_counter_ns()
    kwargs = {"source": "chunks"} if cuda else {}
    image = reader.read_affine(args.shape, origin, axes, order=1, dtype=np.float32, **kwargs)
    image = image.reshape(1, *args.shape)
    sampled = time.perf_counter_ns()
    normalized = normalization({"image": image}, seed=seed)["image"]
    norm_done = time.perf_counter_ns()
    output = augmentation({"image": normalized}, seed=seed)["image"]
    augmented = time.perf_counter_ns()
    if cuda and args.cuda_output == "host":
        output = output.numpy()
    done = time.perf_counter_ns()
    times = dict(read_sample=(sampled - start) / 1e6, normalization=(norm_done - sampled) / 1e6,
                 augmentation=(augmented - norm_done) / 1e6, download=(done - augmented) / 1e6 if cuda and args.cuda_output == "host" else 0.,
                 total=(done - start) / 1e6)
    return output, times


def benchmark(args):
    if args.native_dir:
        load_extensions(args.native_dir.resolve())
    from vc_reader import augment as A, cuda, _native, _augment
    from vc_reader.reader import ZarrReader

    modes = ["cpu", "cuda"] if args.backend == "both" else [args.backend]
    if "cuda" in modes and not cuda.is_available():
        raise ValueError("CUDA reader unavailable; use a CUDA build and accessible device (no CPU fallback)")
    transforms = [A.Flip(axes="x", p_axis=1), A.BrightnessMultiply(1.1), A.Gamma(.9), A.GaussianBlur(1)]
    pipelines = {mode: (A.Pipeline([A.Normalize(args.normalization)], backend=mode, compact=False),
                        A.Pipeline(transforms, backend=mode, compact=False)) for mode in modes}
    report = dict(schema_version=1, command=[sys.executable, *sys.argv],
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
                                   processor=platform.processor(), extensions={}),
                  transforms=[repr(t) for t in transforms],
                  protocol=dict(timing="synchronous wall time; fused read/sample, separate normalization and augmentation calls",
                                included="chunk I/O/decode, planning, allocations, sampling, normalization, augmentation; CUDA upload on cache misses",
                                excluded="input generation, reader/pipeline construction, validation, warmup, cache clearing, output destruction",
                                caches="host and OS caches are not flushed; warm CUDA cache may evict if its budget is too small",
                                order="CPU/CUDA order alternates each repetition; identical crops and seeds",
                                throughput="output voxels and float32 output bytes, not storage or PCIe bandwidth",
                                speedup="CPU / CUDA p50 latency; >1 means CUDA is faster",
                                endpoints="CPU host; CUDA " + args.cuda_output),
                  validation=dict(atol=5e-5, rtol=1e-5, crops=[]), results={})
    for name, module in (("reader", _native), ("augment", _augment)):
        path = Path(module.__file__)
        report["environment"]["extensions"][name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
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
        origins, axes = grids(args, host.shape)
        report["input"] = dict(location=args.array or "synthetic seeded uniform uint8 (temporary raw Zarr v2)",
                               shape=list(host.shape), chunks=list(host.chunks), dtype=str(host.dtype),
                               origins=origins, axes=axes.tolist())

        def run(mode, index):
            return run_crop(readers[mode], *pipelines[mode], origins[index], axes, args,
                            args.seed + index, cuda=mode == "cuda")

        # Validate before warmup. A CUDA-only run never silently executes CPU work.
        for index in range(args.crops):
            outputs = {}
            for mode in modes:
                output, _ = run(mode, index)
                actual = output.numpy() if isinstance(output, cuda.DeviceArray) else output
                if actual.shape != (1, *args.shape) or actual.dtype != np.float32 or not np.isfinite(actual).all():
                    raise ValueError(f"invalid {mode} output for crop {index}")
                outputs[mode] = actual
            row = dict(crop=index, sha256={mode: hashlib.sha256(a.tobytes()).hexdigest() for mode, a in outputs.items()})
            if len(outputs) == 2:
                delta = outputs["cuda"].astype(np.float64) - outputs["cpu"]
                row.update(max_abs=float(np.abs(delta).max()), rmse=float(np.sqrt(np.mean(delta * delta))))
                np.testing.assert_allclose(outputs["cuda"], outputs["cpu"], atol=5e-5, rtol=1e-5)
            report["validation"]["crops"].append(row)
        report["validation"]["comparison"] = "CPU/CUDA allclose" if len(modes) == 2 else "shape, dtype, finite values (single backend)"
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
            for index in range(args.crops):
                for mode in (modes if rep % 2 == 0 else modes[::-1]):
                    output, times = run(mode, index)
                    records[mode].append(dict(repetition=rep, crop=index, **times))
                    del output
        after = dict(host=host.stats())
        if "cuda" in modes:
            after["cuda"] = readers["cuda"].stats
        report["cache_stats"] = dict(before_measurement=before, after_measurement=after)
    voxels = int(np.prod(args.shape))
    for mode, rows in records.items():
        timings = {stage: summary([row[stage] for row in rows]) for stage in ("read_sample", "normalization", "augmentation", "download", "total")}
        seconds = sum(row["total"] for row in rows) / 1000
        report["results"][mode] = dict(timings=timings, samples=len(rows), raw_ms=rows,
                                       crops_per_second=len(rows) / seconds,
                                       mvoxels_per_second=len(rows) * voxels / seconds / 1e6,
                                       output_gib_per_second=len(rows) * voxels * 4 / seconds / (1 << 30))
    if len(modes) == 2:
        report["speedup_cpu_over_cuda"] = {stage: report["results"]["cpu"]["timings"][stage]["p50_ms"] /
                                           report["results"]["cuda"]["timings"][stage]["p50_ms"]
                                           for stage in ("read_sample", "normalization", "augmentation", "total")}
    return report


def main():
    args = arguments()
    try:
        report = benchmark(args)
    except (ImportError, AttributeError, RuntimeError, ValueError, AssertionError) as exc:
        raise SystemExit(f"Benchmark failed: {exc}\nUse matching reader/augmentation extensions; --native-dir can select an existing build.") from exc
    print(f"Input: {report['input']['location']} | shape={args.shape}, crops={args.crops}, reps={args.reps}, warmup={args.warmup}, threads={args.threads}")
    print(f"Warm host/OS caches; CUDA cache={args.cuda_cache}, output={args.cuda_output}. Reading and sampling are fused.")
    print("Backend  Stage             mean ms     p50 ms     p95 ms     min ms     max ms")
    for mode, result in report["results"].items():
        for stage, values in result["timings"].items():
            print(f"{mode:8} {stage:16}" + "".join(f"{values[key]:11.3f}" for key in ("mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms")))
        print(f"{mode}: {result['crops_per_second']:.2f} crops/s, {result['mvoxels_per_second']:.2f} Mvox/s, {result['output_gib_per_second']:.3f} output GiB/s")
    if "speedup_cpu_over_cuda" in report:
        print("Speedup (CPU/CUDA p50): " + ", ".join(f"{stage}={ratio:.2f}x" for stage, ratio in report["speedup_cpu_over_cuda"].items()))
    print("Validation: " + report["validation"]["comparison"] + " passed")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
