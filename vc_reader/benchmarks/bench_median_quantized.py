"""Compare exact/256-level median after native Elastic on real CT crops.

Run through run_native.py to select a built extension; see benchmarks/README.md.
Reads, normalization, deformation, validation, and output destruction are untimed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from vc_reader import augment as A, _augment
from vc_reader.reader import ZarrReader
from bench_augment import DEFAULT_ARRAY, summary
from bench_augment_native import DEFAULT_CROPS_128, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array", default=DEFAULT_ARRAY)
    parser.add_argument("--edge", type=int, default=128)
    parser.add_argument("--crops", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 5, 8])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.edge < 1 or args.reps < 1:
        parser.error("--edge and --reps must be positive")
    deformation = A.Pipeline([A.Elastic(sigma=3, magnitude=1)])
    report = dict(array=args.array, edge=args.edge, reps=args.reps,
                  dtype="one float32 image channel, CT / 255", deformation="Elastic(sigma=3, magnitude=1), seed=123",
                  extension=_augment.__file__, sha256=hashlib.sha256(Path(_augment.__file__).read_bytes()).hexdigest(),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  timed="median public API only, owning outputs; includes profiling clocks",
                  warmup="one correctness/reference call per case/crop", crops=[], results=[])
    inputs = []
    with ZarrReader.open(args.array, threads=1) as reader:
        for start in DEFAULT_CROPS_128[:args.crops]:
            start = tuple(v + (128 - args.edge) // 2 for v in start)
            image = reader.read(start, (args.edge,) * 3).astype(np.float32)[None] / np.float32(255)
            sample = deformation({"image": image}, seed=123)
            inputs.append(sample)
            report["crops"].append(dict(start=start, deformed=fingerprint(sample)))
    for size in args.sizes:
        pipelines = {name: A.Pipeline([A.MedianFilter(size, quantize=quantize)])
                     for name, quantize in (("exact", False), ("quantized", True))}
        timings = {name: [] for name in pipelines}
        profiles = {name: [] for name in pipelines}
        errors = []
        for sample in inputs:
            before = fingerprint(sample)
            reference = pipelines["exact"](sample, seed=123)["image"]
            approximate = pipelines["quantized"](sample, seed=123)["image"]
            error = approximate.astype(np.float64) - reference.astype(np.float64)
            bound = (float(sample["image"].max()) - float(sample["image"].min())) / 510
            rounding = float(np.finfo(np.float32).eps) * float(np.abs(sample["image"]).max()) * 2
            assert np.isfinite(approximate).all() and np.abs(error).max() <= bound + rounding
            assert fingerprint(sample) == before
            errors.append(dict(max_absolute=float(np.abs(error).max()), rmse=float(np.sqrt(np.mean(error * error))),
                               half_bin=bound, output_rounding_allowance=rounding))
            del reference, approximate, error
        for repetition in range(args.reps):
            for sample in inputs:
                for name in (list(pipelines) if repetition % 2 == 0 else list(reversed(pipelines))):
                    start = time.perf_counter()
                    output = pipelines[name](sample, seed=123, profile=True)
                    timings[name].append(time.perf_counter() - start)
                    profiles[name].append(output["_aug_perf"]["0"])
                    del output
        row = dict(size=size, latency={name: summary(values) for name, values in timings.items()},
                   native_step={name: summary(values) for name, values in profiles.items()}, errors=errors)
        report["results"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(size, {name: {k: round(v, 3) for k, v in summary(values).items() if k != "samples_ms"}
                     for name, values in timings.items()}, flush=True)


if __name__ == "__main__":
    main()
