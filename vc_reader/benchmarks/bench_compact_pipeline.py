"""Measure the public compact pipeline, including float32-to-palette encoding.

Run through run_native.py; see README.md. No uint8 input API is assumed.
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


def cases():
    statistics = [A.Contrast(1.2), A.Gamma(1.3, p_invert=1, p_retain_stats=1),
                  A.Gamma(.8, p_retain_stats=1), A.Invert()]
    permutations = [A.Rotate90(planes=("yx",), turns=(1,)), A.Transpose(order="xyz"),
                    A.Flip(axes="z", p_axis=1)]
    return {
        "Gamma": (False, [A.Gamma(.8, p_retain_stats=1)]),
        "Statistics": (False, statistics),
        "Median3": (False, [A.MedianFilter(3)]),
        "Median3_permutations_intensity": (False, [A.MedianFilter(3), *permutations,
                                                     A.BrightnessMultiply(1.1), A.BrightnessAdd(.05), *statistics]),
        "QuantizedMedian5_deformed": (True, [A.MedianFilter(5, quantize=True), *statistics]),
        "Gamma_continuous": (True, [A.Gamma(.8, p_retain_stats=1)]),
        "GaussianBlur": (False, [A.GaussianBlur(1)]),
        "Median3_blur_gamma": (False, [A.MedianFilter(3), A.GaussianBlur(1), A.Gamma(.8)]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array", default=DEFAULT_ARRAY)
    parser.add_argument("--edge", type=int, default=128)
    parser.add_argument("--crops", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--cases", nargs="+", choices=cases())
    parser.add_argument("--legacy-constructor", action="store_true",
                        help="time only the ordinary path in a pre-compact native build")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.edge < 1 or args.reps < 1:
        parser.error("edge and reps must be positive")
    if args.legacy_constructor:
        constructor = _augment.Pipeline
        _augment.Pipeline = lambda transforms, targets, compact=False: constructor(transforms, targets)
    report = dict(array=args.array, edge=args.edge, reps=args.reps,
                  dtype="float32 CT / 255, one image channel", inputs=[], results=[],
                  comparison="legacy baseline only" if args.legacy_constructor else "ordinary versus compact",
                  extension=_augment.__file__, sha256=hashlib.sha256(Path(_augment.__file__).read_bytes()).hexdigest(),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  methodology="one validation/warmup per mode/crop; alternating mode order; public call including validation, sampling, encoding and decode; excludes I/O, deformation preparation, output destruction")
    inputs = []
    deformation = A.Pipeline([A.Elastic(sigma=3, magnitude=1)], compact=False)
    with ZarrReader.open(args.array, threads=1) as reader:
        for start in DEFAULT_CROPS_128[:args.crops]:
            start = tuple(v + (128 - args.edge) // 2 for v in start)
            raw = {"image": reader.read(start, (args.edge,) * 3).astype(np.float32)[None] / np.float32(255)}
            deformed = deformation(raw, seed=123)
            inputs.append((raw, deformed))
            report["inputs"].append(dict(start=start, raw=fingerprint(raw), deformed=fingerprint(deformed)))
    for name, (deformed, transforms) in cases().items():
        if args.cases and name not in args.cases:
            continue
        modes = ("ordinary",) if args.legacy_constructor else ("ordinary", "compact")
        pipelines = {mode: A.Pipeline(transforms, compact=mode == "compact") for mode in modes}
        timings = {mode: [] for mode in pipelines}
        profiles = {mode: [] for mode in pipelines}
        maximum = squared = count = 0
        bitwise = True
        for raw, warped in inputs:
            sample = warped if deformed else raw
            before = fingerprint(sample)
            outputs = {mode: pipeline(sample, seed=123, profile=True) for mode, pipeline in pipelines.items()}
            a = outputs["ordinary"]["image"]
            b = outputs.get("compact", outputs["ordinary"])["image"]
            delta = b.astype(np.float64) - a
            maximum = max(maximum, float(np.abs(delta).max()))
            squared += float(np.sum(delta * delta)); count += delta.size
            bitwise &= a.tobytes() == b.tobytes()
            np.testing.assert_allclose(b, a, atol=2e-6, rtol=1e-6)
            assert fingerprint(sample) == before
            for mode in pipelines:
                profiles[mode].append(outputs[mode]["_aug_perf"])
        for repetition in range(args.reps):
            for raw, warped in inputs:
                sample = warped if deformed else raw
                for mode in (list(pipelines) if repetition % 2 == 0 else list(reversed(pipelines))):
                    start = time.perf_counter()
                    output = pipelines[mode](sample, seed=123)
                    timings[mode].append(time.perf_counter() - start)
                    del output
        row = dict(name=name, transforms=[repr(t) for t in transforms],
                   latency={mode: summary(t) for mode, t in timings.items()},
                   native_profile=profiles, max_absolute=maximum, rmse=(squared / count) ** .5, bitwise_equal=bitwise)
        report["results"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, {mode: round(summary(t)["mean_ms"], 3) for mode, t in timings.items()},
              "max error", maximum, flush=True)


if __name__ == "__main__":
    main()
