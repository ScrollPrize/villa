"""Paired public CPU/CUDA API benchmark on the native ranking's real CT crops.

No installation or downloads. Inputs, sampling, transfers, allocations and
synchronization use the public Pipeline API. Every timed pair is checked;
GPU replay/repeatability and CPU dense-reference checks are outside timing.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from vc_reader import _augment, augment as A
from vc_reader.augment.cuda import is_available
from vc_reader.reader import ZarrReader
from bench_augment import DEFAULT_ARRAY, cases, summary
from bench_augment_native import crop_starts, fingerprint, prepare


def comparison(actual, expected):
    if actual.keys() != expected.keys():
        raise AssertionError("output keys differ")
    report = {}
    for key, ref in expected.items():
        out = actual[key]
        if out.shape != ref.shape or out.dtype != ref.dtype or not np.isfinite(out).all():
            raise AssertionError(f"invalid CUDA output: {key}")
        error = np.abs(out.astype(np.float64) - ref.astype(np.float64))
        report[key] = dict(max_abs=float(error.max()), unequal=int(np.count_nonzero(out != ref)),
                           byte_equal=out.tobytes() == ref.tobytes())
        if ref.dtype.kind == "f":
            np.testing.assert_allclose(out, ref, atol=3e-7, rtol=2e-6, err_msg=key)
        else:
            np.testing.assert_array_equal(out, ref, err_msg=key)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array", default=DEFAULT_ARRAY)
    parser.add_argument("--shape", nargs=3, type=int, default=(128,128,128))
    parser.add_argument("--start", nargs=3, type=int)
    parser.add_argument("--additional-start", nargs=3, type=int, action="append", default=[])
    parser.add_argument("--only", default="")
    parser.add_argument("--chains", action="store_true", help="also benchmark composed spatial and intensity pipelines")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not is_available():
        parser.error("a CUDA-enabled vc_reader build and visible GPU are required")
    if args.reps < 1 or args.warmup < 0 or min(args.shape) < 1:
        parser.error("shape/reps must be positive; warmup must be nonnegative")
    workload = cases(tuple(args.shape))
    median = next(c for c in workload if c.name == "MedianFilter")
    workload.append(replace(median,name="MedianFilterContinuous",inputs="continuous"))
    if args.chains:
        by_name = {c.name:c for c in workload}
        for name, inputs, steps in (
            ("SpatialChain", "both", ("Elastic", "MedianFilter", "LocalSmoothing", "Gamma", "Contrast")),
            ("IntensityChain", "image", ("BrightnessMultiply", "GaussianNoise", "BrightnessGradient",
                                         "LocalContrast", "LocalGamma", "GaussianBlur", "Sharpen")),
        ):
            workload.append(replace(median, name=name, inputs=inputs,
                                    native=[t for step in steps for t in by_name[step].native]))
    selected = set(args.only.split(",")) if args.only else {c.name for c in workload}
    unknown = selected - {c.name for c in workload}
    if unknown:
        parser.error(f"unknown cases: {sorted(unknown)}")
    workload = [c for c in workload if c.name in selected]
    pipelines = {c.name: {mode: A.Pipeline(c.native, targets={"image":A.Image(),"segmentation":A.Label()},
                                          backend="cuda" if mode == "cuda" else "cpu", compact=mode=="cpu")
                          for mode in ("cpu","cuda","dense_reference")} for c in workload}
    build = args.build_dir
    binary = Path(_augment.__file__)
    flags = build / "CMakeFiles/_augment.dir/flags.make"
    cache = build / "CMakeCache.txt"
    report = dict(command=sys.argv, array=args.array, shape=args.shape, reps=args.reps, warmup=args.warmup,
                  native_extension=str(binary), native_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  build_flags=flags.read_text() if flags.exists() else None,
                  build_cache=cache.read_text() if cache.exists() else None,
                  gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],text=True).strip(),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
                  protocol="single CPU thread; seeded shuffled case/backend order; seeds 123+repetition; synchronized public host-to-host API; includes sampling, validation, allocations and all transfers; excludes reads, normalization, output comparison/destruction",
                  reference="CPU default compact=True timed; dense compact=False also checked outside timing. CUDA uses dense buffers. Labels exact, images atol=3e-7 rtol=2e-6; no fast math or reduced precision.",
                  inputs="uint8 CT/255 float32; derived labels CT>.5 int16; continuous median adds fixed normal noise std .01 seed 123",
                  crops=[], results={c.name:dict(cpu=[],cuda=[],comparisons=[]) for c in workload})
    order = random.Random(20261016)
    with ZarrReader.open(args.array,threads=1) as reader:
        if np.dtype(reader.dtype) != np.dtype("uint8"):
            raise ValueError("expected uint8 CT")
        for start in crop_starts(args):
            image = reader.read(start,tuple(args.shape)).astype(np.float32)[None]/np.float32(255)
            if image.min() == image.max():
                raise ValueError(f"crop {start} has no contrast")
            samples = prepare(image)
            samples["continuous"] = dict(image=image+np.random.default_rng(123).normal(0,.01,image.shape).astype(np.float32))
            before = {key:fingerprint(value) for key,value in samples.items()}
            report["crops"].append(dict(start=start,inputs=before,outputs={}))
            for case in workload:
                sample, modes = samples[case.inputs], pipelines[case.name]
                first = modes["cuda"](sample,seed=123)
                comparison(first,modes["dense_reference"](sample,seed=123))
                expected_hash = fingerprint(first)
                assert fingerprint(modes["cuda"].sample(sample,seed=123).apply(sample)) == expected_hash
                assert fingerprint(modes["cuda"](sample,seed=123)) == expected_hash
                report["crops"][-1]["outputs"][case.name] = expected_hash
                for _ in range(args.warmup):
                    for mode in ("cpu","cuda"):
                        modes[mode](sample,seed=123)
            for rep in range(args.reps):
                sequence = list(workload); order.shuffle(sequence)
                for case in sequence:
                    sample = samples[case.inputs]
                    modes = ["cpu","cuda"]; order.shuffle(modes)
                    outputs = {}
                    for mode in modes:
                        begin = time.perf_counter_ns()
                        out = pipelines[case.name][mode](sample,seed=123+rep)
                        elapsed = (time.perf_counter_ns()-begin)*1e-9
                        outputs[mode] = out
                        report["results"][case.name][mode].append(elapsed)
                    detail = comparison(outputs["cuda"],outputs["cpu"])
                    report["results"][case.name]["comparisons"].append(dict(start=start,seed=123+rep,**detail))
                    del out, outputs
            assert before == {key:fingerprint(value) for key,value in samples.items()}
            print("validated crop",start,flush=True)
    for name,row in report["results"].items():
        for mode in ("cpu","cuda"):
            row[mode] = summary(row[mode])
        row["speedup"] = row["cpu"]["p50_ms"]/row["cuda"]["p50_ms"]
        print(f"{name:24} CPU {row['cpu']['p50_ms']:9.3f} ms CUDA {row['cuda']['p50_ms']:9.3f} ms {row['speedup']:6.2f}x",flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
