"""Native-only runner for bench_augment.py; shares its transform configurations.

Invoke bench_augment.py --native-only, not this module directly.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np

from vc_reader import augment as A
from vc_reader import _augment
from vc_reader.reader import ZarrReader


# Preserve the original three crops and add three with denser threshold masks.
DEFAULT_CROPS_128 = (
    (8960, 4480, 3200), (5120, 3840, 3840), (5120, 2816, 5120),
    (4608, 5760, 5888), (4480, 5760, 5248), (4224, 5888, 5120),
)


def crop_starts(args):
    if args.start is not None:
        return [tuple(args.start), *map(tuple, args.additional_start)]
    if args.additional_start:
        raise SystemExit("--additional-start requires an explicit --start")
    # Keep the same centers when changing patch size (e.g. 128³ -> 64³).
    return [tuple(origin + (128 - size) // 2 for origin, size in zip(start, args.shape))
            for start in DEFAULT_CROPS_128]


def arrays(sample):
    """Flatten array outputs, including deep supervision, excluding metadata."""
    for key, value in sorted(sample.items()):
        if isinstance(value, np.ndarray):
            yield key, value
        elif isinstance(value, (list, tuple)):
            for i, level in enumerate(value):
                if isinstance(level, np.ndarray):
                    yield f"{key}/{i}", level


def fingerprint(sample):
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                  "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
            for key, value in arrays(sample)}


def prepare(image):
    labels = (image > .5).astype(np.int16)
    return {"image": {"image": image}, "both": {"image": image, "segmentation": labels},
            "binary": {"image": labels.astype(np.float32)}}


def run(args, cases, summary):
    starts = crop_starts(args)
    selected = set(args.only.split(",")) if args.only else {case.name for case in cases}
    unknown = selected - {case.name for case in cases}
    if unknown:
        raise SystemExit(f"Unknown cases: {', '.join(sorted(unknown))}")
    cases = [case for case in cases if case.name in selected]
    profiled = set(args.profile.split(",")) if args.profile else set()
    if profiled - selected - {"all"}:
        raise SystemExit("--profile must name selected cases, or 'all'")
    pipelines = {case.name: A.Pipeline(case.native, targets={"image": A.Image(), "segmentation": A.Label()})
                 for case in cases}
    root = Path(__file__).resolve().parents[2]
    cpu_info = Path("/proc/cpuinfo")
    cpu = next((line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                if line.startswith("model name")), platform.processor()) if cpu_info.exists() else platform.processor()
    flags = args.build_dir / "CMakeFiles/_augment.dir/flags.make"
    cache = args.build_dir / "CMakeCache.txt"
    binary_hash = hashlib.sha256(Path(_augment.__file__).read_bytes()).hexdigest()
    report = dict(command=sys.argv, mode="native-only", python=sys.version,
                  numpy=np.__version__, platform=platform.platform(), cpu=cpu,
                  logical_cpus=os.cpu_count(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  thread_environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                  native_extension=_augment.__file__, native_sha256=binary_hash,
                  native_build_flags=flags.read_text() if flags.exists() else None,
                  native_build_type=next((line.split("=", 1)[1] for line in cache.read_text().splitlines()
                                         if line.startswith("CMAKE_BUILD_TYPE:")), None) if cache.exists() else None,
                  native_matches_build=any(hashlib.sha256(p.read_bytes()).hexdigest() == binary_hash
                                           for pattern in ("_augment*.so", "_augment*.pyd")
                                           for p in args.build_dir.glob(pattern)),
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  array=args.array, shape=args.shape, channels=1, dtype="float32",
                  labels="CT > 0.5 after uint8 / 255, int16; derived masks, not annotations",
                  warmup=args.warmup, reps=args.reps, seeds="123 + repetition; same seed across crops and transforms",
                  ownership="default owning outputs, original input preserved on every call",
                  timed="public NumPy API: adaptation, native validation/sampling/kernels and output allocation; excludes reads, normalization, hashing, destruction",
                  order="deterministically shuffled cases each repetition (Python Random(20261016))",
                  input_noise_std=args.input_noise_std, crops=[], cases=[],
                  excluded=["Data preparation, label encoding/cleanup, target generation, copy controls and composite pipelines",
                            "RemoveComponents, ClipOutliers, Affine, SheetCompression: excluded by request",
                            "StripGeometry, SurfacePerturb, translate_crop_box: require tracing/surface inputs not provided by CT crops",
                            "OneOf and standalone probability wrappers; parameters chosen to exercise transforms, not training application frequencies"])
    inputs = []
    with ZarrReader.open(args.array, threads=1) as reader:
        if np.dtype(reader.dtype) != np.dtype("uint8"):
            raise ValueError("this workload expects uint8 CT (normalized by 255)")
        for start in starts:
            image = reader.read(tuple(start), tuple(args.shape)).astype(np.float32)[None] / np.float32(255)
            if image.max() == image.min():
                raise ValueError(f"crop {start} contains no CT contrast")
            if args.input_noise_std:
                image += np.random.default_rng(123).normal(0, args.input_noise_std, image.shape).astype(np.float32)
            sample = prepare(image)
            inputs.append(sample)
            report["crops"].append(dict(start=list(start), image=fingerprint(sample["image"]),
                                        nonzero_fraction=float(np.mean(image != 0)),
                                        foreground_fraction=float(np.mean(image > .5))))
    rows = {case.name: dict(name=case.name, input=case.inputs, native=[repr(t) for t in case.native],
                            crops=[], timings=[[] for _ in inputs]) for case in cases}
    print(f"Native only: {cpu}; shape={args.shape}; {len(inputs)} crops; {args.reps} repetitions", flush=True)
    # Validate all cases before timing: exact repeatability, finite outputs and
    # input preservation. Profiling is a separate, explicitly untimed call.
    for case in cases:
        pipeline = pipelines[case.name]
        for crop_index, source in enumerate(inputs):
            sample = source[case.inputs]
            before = fingerprint(sample)
            first = pipeline(sample, seed=123)
            if not all(np.isfinite(value).all() for _, value in arrays(first)):
                raise AssertionError(f"nonfinite output: {case.name}")
            output_hash = fingerprint(first)
            if output_hash != fingerprint(pipeline(sample, seed=123)):
                raise AssertionError(f"seeded repeatability failed: {case.name}")
            if before != fingerprint(sample):
                raise AssertionError(f"input modified: {case.name}")
            detail = dict(start=report["crops"][crop_index]["start"], output=output_hash)
            del first
            if "all" in profiled or case.name in profiled:
                timings = {}
                for i in range(5):
                    output = pipeline(sample, seed=123 + i, profile=True)
                    for index, elapsed in output["_aug_perf"].items():
                        timings.setdefault(index, []).append(elapsed)
                    del output
                detail["native_steps"] = {index: summary(values) for index, values in timings.items()}
            rows[case.name]["crops"].append(detail)
            for i in range(args.warmup):
                pipeline(sample, seed=123 + i)
        print(f"  validated/warmed {case.name}", flush=True)
    order_rng = random.Random(20261016)
    for repetition in range(args.reps):
        order = list(cases)
        order_rng.shuffle(order)
        for case in order:
            for crop_index, source in enumerate(inputs):
                sample = source[case.inputs]
                pipeline = pipelines[case.name]
                seed = 123 + repetition
                start = time.perf_counter_ns()
                output = pipeline(sample, seed=seed)
                elapsed = (time.perf_counter_ns() - start) * 1e-9
                rows[case.name]["timings"][crop_index].append(elapsed)
                del output
        print(f"  repetition {repetition + 1}/{args.reps}", flush=True)
    for case in cases:
        row = rows[case.name]
        timings = row.pop("timings")
        row["latency"] = summary([t for crop in timings for t in crop])
        for crop, values in zip(row["crops"], timings):
            crop["latency"] = summary(values)
        report["cases"].append(row)
    report["cases"].sort(key=lambda row: row["latency"]["p50_ms"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'transform':24} {'mean ms':>10} {'p50 ms':>10} {'p95 ms':>10}")
    for row in report["cases"]:
        stats = row["latency"]
        print(f"{row['name']:24} {stats['mean_ms']:10.3f} {stats['p50_ms']:10.3f} {stats['p95_ms']:10.3f}")
    print(f"Saved {args.output}", flush=True)
