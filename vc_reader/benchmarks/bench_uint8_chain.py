"""Read real CT crops and run the experimental native uint8 storage benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np

from vc_reader.reader import ZarrReader
from bench_augment import DEFAULT_ARRAY
from bench_augment_native import DEFAULT_CROPS_128


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--array", default=DEFAULT_ARRAY)
    parser.add_argument("--edge", type=int, default=128)
    parser.add_argument("--crops", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--median-size", type=int, default=3)
    parser.add_argument("--suite", choices=["basic", "extended"], default="basic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.edge < 1 or args.reps < 1:
        parser.error("--edge and --reps must be positive")
    crops = []
    with tempfile.TemporaryDirectory(prefix="vc-uint8-") as work:
        paths = []
        with ZarrReader.open(args.array, threads=1) as reader:
            for index, start in enumerate(DEFAULT_CROPS_128[:args.crops]):
                start = tuple(v + (128 - args.edge) // 2 for v in start)
                image = reader.read(start, (args.edge,) * 3)
                if image.dtype != np.uint8:
                    raise ValueError("experiment requires uint8 source CT")
                path = Path(work) / f"crop-{index}.raw"
                image.tofile(path)
                paths.append(str(path))
                crops.append(dict(start=start, sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        command = [str(args.binary.resolve()), str(args.edge), str(args.reps), str(args.median_size)]
        if args.suite == "extended":
            command.append("--extended")
        command.extend(paths)
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stderr, end="")
        result.check_returncode()
        report = json.loads(result.stdout)
    report.update(array=args.array, inputs=crops, platform=platform.platform(), binary=str(args.binary.resolve()),
                  binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  geometry="Rotate90(yx,1), Transpose(xyz), Flip(z)",
                  intensity="cumulative BrightnessMultiply(1.1), BrightnessAdd(.05), Gamma(.8, retain_stats=False)",
                  quantization="uint8 clips/rounds to [0,1] after each intensity stage; palette keeps float32 values",
                  timing="includes initial copy/upcast, kernel construction/application, and final decode; excludes reads and output destruction",
                  validation="palette output must be bit-exact to float32; uint8 median/geometry must be bit-exact; per-prefix errors on all voxels",
                  warmup="one validation call per crop/case/mode; timing uses all six mode orders every six repetitions")
    if args.suite == "extended":
        report.update(geometry="CropPad +/-8 per axis; ThickSlice(scale=.5, axis=z)",
                      intensity="Contrast(1.2), Invert, Gamma(.8, retain_stats=True), Gamma(1.3, invert=True, retain_stats=True); see named chains in palette_extended.hpp",
                      quantization="only SliceIllumination rounds to 256 levels over the slice tables' global min/max; no clipping",
                      slice_parameters="native sampled defects=(2,5), width=(25,50), reduction=(.3,1.5), base_p=(.2,.4), base_reduction=(.5,.9)",
                      timing="includes initial copy/upcast, replay kernels and final decode; excludes reads, plan sampling, validation comparisons and output destruction",
                      validation="matched native replay plans; per-voxel and per-z-slice mean errors; non-approximate cases require max error <=2e-6",
                      warmup="one validation call per crop/case/mode; two timing orders alternate; fixed seed=123+crop index")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["results"]:
        print(row.get("name", (row.get("median"), row.get("intensity_stages"))), row["mode"],
              "mean_ms", round(row["latency"]["mean_ms"], 3),
              "max_error", row["max_absolute"], "rmse", row["rmse"])


if __name__ == "__main__":
    main()
