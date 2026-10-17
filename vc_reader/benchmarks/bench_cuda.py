"""Optional CUDA feasibility benchmark; build bench_cuda.cu, then use run_native.py.

This is deliberately narrower than the public API: one finite float32 image,
cubic 3D grids, int16 labels, fixed transform parameters. No automatic dispatch.
Reports synchronized kernel time separately from CPU sampling + pageable host
transfers + host output allocation. Reuses device allocations across calls.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from vc_reader import augment as A, _augment
from vc_reader.reader import ZarrReader
from bench_augment import DEFAULT_ARRAY, summary
from bench_augment_native import DEFAULT_CROPS_128, fingerprint


class CUDA:
    def __init__(self, path, edge):
        self.lib = C.CDLL(str(path.resolve()))
        specs = {
            "error": (C.c_char_p, []),
            "create": (C.c_void_p, [C.c_int]),
            "destroy": (None, [C.c_void_p]),
            "field": (C.c_void_p, [C.c_int, C.c_uint64, C.c_int, C.POINTER(C.c_void_p)]),
            "free_field": (None, [C.c_void_p]),
            "copy_field": (C.c_int, [C.c_void_p, C.c_void_p]),
            "run": (C.c_int, [C.c_void_p, C.c_int] + [C.c_void_p]*5 + [C.POINTER(C.c_float)]),
        }
        for name, (result, arguments) in specs.items():
            function = getattr(self.lib, "bench_cuda_" + name)
            function.restype, function.argtypes = result, arguments
            setattr(self, name, function)
        self.edge = edge
        self.ctx = self.create(edge)
        if not self.ctx:
            raise RuntimeError(self.error().decode())

    def apply(self, sample, op, seed):
        field_handle = None
        try:
            output = {key: np.empty_like(value) for key, value in sample.items()}
            field, kernel_ms = C.c_void_p(), C.c_float()
            if op != 0:
                field_handle = self.field(self.edge, seed, op, C.byref(field))
                if not field_handle:
                    raise RuntimeError(self.error().decode())
            status = self.run(self.ctx, op, sample["image"].ctypes.data,
                              sample["mask"].ctypes.data if op != 0 else None, field,
                              output["image"].ctypes.data,
                              output["mask"].ctypes.data if op != 0 else None,
                              C.byref(kernel_ms))
            if status:
                raise RuntimeError(self.error().decode())
            return output, kernel_ms.value * .001
        finally:
            if field_handle:
                self.free_field(field_handle)

    def close(self):
        self.destroy(self.ctx)

    def field_error(self, seed):
        """Untimed field comparison; reductions differ only in experimental op 2."""
        actual = np.empty((self.edge**3*3,), dtype=np.float64)
        if self.copy_field(self.ctx, actual.ctypes.data):
            raise RuntimeError(self.error().decode())
        ptr = C.c_void_p()
        handle = self.field(self.edge, seed, 1, C.byref(ptr))
        if not handle:
            raise RuntimeError(self.error().decode())
        try:
            expected = np.ctypeslib.as_array(C.cast(ptr, C.POINTER(C.c_double)), shape=actual.shape)
            error = float(np.max(np.abs(actual-expected)))
            if error > 5e-12:
                raise AssertionError(f"GPU field error {error}")
            return error, hashlib.sha256(actual.tobytes()).hexdigest()
        finally:
            self.free_field(handle)


def check(actual, expected, gpu_field=False):
    result = {}
    for key in expected:
        exact = actual[key].tobytes() == expected[key].tobytes()
        error = np.abs(actual[key].astype(float)-expected[key])
        result[key] = dict(byte_equal=exact, max_abs=float(error.max()),
                           unequal=int(np.count_nonzero(actual[key] != expected[key])))
        if actual[key].dtype != expected[key].dtype or not np.isfinite(actual[key]).all():
            raise AssertionError(f"{key}: invalid CUDA output")
        if not exact and not (gpu_field and key == "image" and error.max() <= 2e-7):
            raise AssertionError(f"{key}: CUDA differs from CPU: {result[key]}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--array", default=DEFAULT_ARRAY)
    parser.add_argument("--edge", type=int, default=128)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--gpu-field", action="store_true",
                        help="also test GPU field expansion/normalization; reports reduction-rounding error")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.edge <= 256 or args.reps < 1:
        parser.error("edge must be 1..256 and reps positive")
    cuda = CUDA(args.binary, args.edge)
    pipelines = {
        "MedianFilter_continuous": A.Pipeline([A.MedianFilter(3)]),
        "Elastic": A.Pipeline([A.Elastic(sigma=3, magnitude=1)],
                               targets={"image": A.Image(), "mask": A.Label()}),
    }
    operations = {"MedianFilter_continuous": 0, "Elastic": 1}
    if args.gpu_field:
        pipelines["Elastic_gpu_field"] = pipelines["Elastic"]
        operations["Elastic_gpu_field"] = 2
    times = {name: {mode: [] for mode in ("cpu_public", "cuda_host", "cuda_kernel")}
             for name in pipelines}
    report = dict(command=sys.argv, edge=args.edge, reps=args.reps, array=args.array, crops=[], comparisons=[],
                  gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip(),
                  binary=str(args.binary), binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  native_extension=_augment.__file__, native_sha256=hashlib.sha256(Path(_augment.__file__).read_bytes()).hexdigest(),
                  protocol="one warmup per case/crop; seeded shuffled case and backend order; CPU 0, one thread; seeds 123 + repetition; comparisons after every timed pair",
                  cuda_host="CPU field (op 1) or control-lattice draws (op 2), host output allocation, pageable H2D/D2H transfers, kernels and synchronization; device buffers/context reused; excludes public schema/finite validation",
                  cuda_kernel="CUDA events enclosing kernels after H2D, including field expansion/normalization for op 2; excludes CPU sampling and all transfers",
                  gpu_field="op 2 only: CPU draws small random control lattice using production RNG; GPU expands, centers and normalizes dense field then resamples; deterministic double reduction trees change summation order; field error <=5e-12, image error <=2e-7, labels must be exact; no production behavior change",
                  inputs="float32 CT /255; continuous median adds fixed Gaussian noise std .01 seed 123; Elastic labels CT > .5 int16")
    order_rng = random.Random(20261016)
    try:
        # Synthetic smoke checks cover thin grids in separate --edge runs,
        # ties/plateaus, boundary stencils, signed labels and random floats.
        shape = (1, args.edge, args.edge, args.edge)
        rng = np.random.default_rng(4321)
        synthetic = rng.normal(size=shape).astype(np.float32)
        for values in (synthetic, np.zeros(shape, np.float32), (synthetic > 0).astype(np.float32)):
            sample = dict(image=values)
            check(cuda.apply(sample, 0, 123)[0], pipelines["MedianFilter_continuous"](sample, seed=123))
        sample = dict(image=synthetic, mask=rng.integers(-10, 10, size=shape, dtype=np.int16))
        check(cuda.apply(sample, 1, 123)[0], pipelines["Elastic"](sample, seed=123))
        if args.gpu_field:
            check(cuda.apply(sample, 2, 123)[0], pipelines["Elastic"](sample, seed=123), True)
            cuda.field_error(123)
        with ZarrReader.open(args.array, threads=1) as reader:
            for origin in DEFAULT_CROPS_128:
                start = tuple(x + (128 - args.edge)//2 for x in origin)
                image = reader.read(start, (args.edge,)*3).astype(np.float32)[None] / np.float32(255)
                noisy = image + np.random.default_rng(123).normal(0, .01, image.shape).astype(np.float32)
                samples = {"MedianFilter_continuous": dict(image=noisy),
                           "Elastic": dict(image=image, mask=(image > .5).astype(np.int16))}
                if args.gpu_field: samples["Elastic_gpu_field"] = samples["Elastic"]
                before = {name: fingerprint(sample) for name, sample in samples.items()}
                report["crops"].append(dict(start=start, inputs=before, outputs={}))
                for name, sample in samples.items():
                    check(cuda.apply(sample, operations[name], 123)[0], pipelines[name](sample, seed=123), operations[name] == 2)
                    if operations[name] == 2:
                        field_hash = cuda.field_error(123)[1]
                        cuda.apply(sample, 2, 123)
                        assert cuda.field_error(123)[1] == field_hash
                for rep in range(args.reps):
                    names = list(samples); order_rng.shuffle(names)
                    for name in names:
                        sample = samples[name]
                        modes = ["cpu_public", "cuda_host"]; order_rng.shuffle(modes)
                        outputs = {}
                        for mode in modes:
                            begin = time.perf_counter_ns()
                            if mode == "cpu_public":
                                out = pipelines[name](sample, seed=123+rep)
                            else:
                                out, kernel = cuda.apply(sample, operations[name], 123+rep)
                            elapsed = (time.perf_counter_ns()-begin)*1e-9
                            outputs[mode] = out
                            times[name][mode].append(elapsed)
                            if mode == "cuda_host": times[name]["cuda_kernel"].append(kernel)
                        comparison = check(outputs["cuda_host"], outputs["cpu_public"], operations[name] == 2)
                        if operations[name] == 2:
                            comparison["field_max_abs"] = cuda.field_error(123+rep)[0]
                        report["comparisons"].append(dict(case=name, start=start, seed=123+rep, **comparison))
                        if rep == 0:
                            report["crops"][-1]["outputs"][name] = fingerprint(out)
                        del out, outputs
                assert before == {name: fingerprint(sample) for name, sample in samples.items()}
                print("validated crop", start, flush=True)
    finally:
        cuda.close()
    report["results"] = {name: {mode: summary(values) for mode, values in modes.items()}
                         for name, modes in times.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, modes in report["results"].items():
        print(name, {mode: round(value["p50_ms"], 3) for mode, value in modes.items()}, flush=True)


if __name__ == "__main__":
    main()
