"""Compare existing Elastic label dtypes on real CT-derived binary masks.

Run through run_native.py; casting and I/O are outside timing. No compression
or implicit dtype conversion is performed by the augmentation pipeline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
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
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.edge < 1 or args.reps < 1:
        parser.error("edge and reps must be positive")
    dtypes = ("bool", "uint8", "int8", "int16", "int32", "int64")
    pipeline = A.Pipeline([A.Elastic(sigma=3, magnitude=1)],
                          targets={"image": A.Image(), "mask": A.Label()})
    timings = {(mode, dtype): [] for mode in ("labels", "image_and_labels") for dtype in dtypes}
    report = dict(array=args.array, edge=args.edge, reps=args.reps, inputs=[],
                  extension=_augment.__file__, sha256=hashlib.sha256(Path(_augment.__file__).read_bytes()).hexdigest(),
                  methodology="six CT crops; one validation/warmup per case/crop; deterministic shuffled case order; seeds 123 + repetition; public API; excludes I/O, casting, hashing, output destruction")
    rng = random.Random(20261016)
    with ZarrReader.open(args.array, threads=1) as reader:
        for origin in DEFAULT_CROPS_128:
            start = tuple(x + (128 - args.edge) // 2 for x in origin)
            image = reader.read(start, (args.edge,) * 3).astype(np.float32)[None] / np.float32(255)
            masks = {dtype: (image > .5).astype(dtype) for dtype in dtypes}
            samples = {(mode, dtype): dict(mask=mask, **({"image": image} if mode == "image_and_labels" else {}))
                       for mode in ("labels", "image_and_labels") for dtype, mask in masks.items()}
            report["inputs"].append(dict(start=start, arrays=fingerprint(dict(image=image, mask=masks["int16"]))))
            before = {case: fingerprint(sample) for case, sample in samples.items()}
            reference = pipeline(samples["image_and_labels", "int16"], seed=123)
            for (mode, dtype), sample in samples.items():
                out = pipeline(sample, seed=123)
                assert out["mask"].dtype == np.dtype(dtype)
                np.testing.assert_array_equal(out["mask"], reference["mask"])
                if "image" in out:
                    np.testing.assert_array_equal(out["image"], reference["image"])
            for rep in range(args.reps):
                order = list(samples); rng.shuffle(order)
                for case in order:
                    begin = time.perf_counter_ns()
                    out = pipeline(samples[case], seed=123 + rep)
                    timings[case].append((time.perf_counter_ns() - begin) * 1e-9)
                    del out
            assert all(fingerprint(sample) == before[case] for case, sample in samples.items())
    report["results"] = [dict(mode=mode, dtype=dtype, mask_bytes=args.edge**3 * np.dtype(dtype).itemsize,
                              latency=summary(values)) for (mode, dtype), values in timings.items()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["results"]:
        print(row["mode"], row["dtype"],
              {key: round(row["latency"][key], 3) for key in ("mean_ms", "p50_ms", "p95_ms")}, flush=True)


if __name__ == "__main__":
    main()
