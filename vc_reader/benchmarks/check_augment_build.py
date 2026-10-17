"""Capture/compare native build outputs on the benchmark's six real CT crops.

Run with run_native.py to select each build. Integer outputs must match exactly;
floating outputs are compared with explicit tolerances. This does not time calls.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from vc_reader import _augment, augment as A
from vc_reader.reader import ZarrReader
from bench_augment import DEFAULT_ARRAY, cases
from bench_augment_native import arrays, crop_starts, fingerprint, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--array", default=DEFAULT_ARRAY)
    parser.add_argument("--shape", type=int, nargs=3, default=[128] * 3)
    parser.add_argument("--only", default="Elastic")
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 124, 125])
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selected = set(args.only.split(","))
    configurations = [c for c in cases(args.shape) if c.name in selected]
    if selected != {c.name for c in configurations}:
        parser.error("--only contains unknown benchmark cases")
    args.start, args.additional_start = None, []
    starts = crop_starts(args)
    workload = dict(array=args.array, shape=args.shape, starts=starts, seeds=args.seeds,
                    cases={c.name: [repr(t) for t in c.native] for c in configurations})
    # Canonical JSON representation also converts tuples to lists.
    workload = json.loads(json.dumps(workload))
    manifest_path = args.reference / "manifest.json"
    if args.capture:
        args.reference.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            parser.error("reference already exists; choose a new directory")
        manifest = dict(workload=workload, inputs=[])
    else:
        manifest = json.loads(manifest_path.read_text())
        if manifest["workload"] != workload:
            parser.error("workload differs from the reference")
    binary_hash = hashlib.sha256(Path(_augment.__file__).read_bytes()).hexdigest()
    report = dict(native_sha256=binary_hash, workload=workload, atol=args.atol,
                  rtol=args.rtol, comparisons=[])
    pipelines = {c.name: A.Pipeline(c.native, targets={"image": A.Image(), "segmentation": A.Label()})
                 for c in configurations}
    with ZarrReader.open(args.array, threads=1) as reader:
        if np.dtype(reader.dtype) != np.dtype("uint8"):
            raise ValueError("this workload expects uint8 CT")
        for crop_index, start in enumerate(starts):
            image = reader.read(tuple(start), tuple(args.shape)).astype(np.float32)[None] / np.float32(255)
            inputs = prepare(image)
            input_hash = fingerprint(inputs["both"])
            if args.capture:
                manifest["inputs"].append(input_hash)
            else:
                assert input_hash == manifest["inputs"][crop_index], "input changed"
            for case in configurations:
                for seed in args.seeds:
                    actual = dict(arrays(pipelines[case.name](inputs[case.inputs], seed=seed)))
                    path = args.reference / f"{case.name}-{crop_index}-{seed}.npz"
                    if args.capture:
                        np.savez(path, **actual)
                        continue
                    with np.load(path, allow_pickle=False) as expected:
                        assert set(actual) == set(expected.files), "output fields changed"
                        for key, value in actual.items():
                            reference = expected[key]
                            assert value.shape == reference.shape and value.dtype == reference.dtype
                            equal = value == reference
                            max_error = 0.
                            if np.issubdtype(value.dtype, np.floating):
                                assert np.isfinite(value).all() and np.isfinite(reference).all(), "nonfinite output"
                                np.testing.assert_allclose(value, reference, atol=args.atol, rtol=args.rtol)
                                max_error = float(np.max(np.abs(value.astype(np.float64) - reference)))
                            else:
                                np.testing.assert_array_equal(value, reference)
                            report["comparisons"].append(dict(case=case.name, crop=crop_index, seed=seed,
                                field=key, changed=int(np.count_nonzero(~equal)), elements=value.size,
                                max_abs_error=max_error))
            print(f"{'Captured' if args.capture else 'Checked'} crop {crop_index + 1}/{len(starts)}", flush=True)
    if args.capture:
        manifest["native_sha256"] = binary_hash
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        report["reference_sha256"] = manifest["native_sha256"]
        print(f"Maximum absolute error: {max(c['max_abs_error'] for c in report['comparisons']):.9g}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
