"""Reproduce Copy bbox timings using public PHerc0841 mesh data.

Manual benchmark, not part of the pytest test suite. Uses *real* official
ScrollPrize tifxyz coordinate TIFFs, with their published SHA256 checked.
Runs the unmodified generator from the pinned upstream Git revision and the patched
generator in the checkout. No checkpoint/GPU/network volume needed.

From the villa checkout:
    python vesuvius/tests/neural_tracing/bench_bbox_pherc0841.py --repeats 4
    python vesuvius/tests/neural_tracing/bench_bbox_pherc0841.py --repeats 4 --crop-size 128 384 384
Dependencies: numpy, tifffile, tqdm
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import types
import urllib.request
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR_PATH = "vesuvius/src/vesuvius/neural_tracing/inference/generate_segment_cover_bboxes.py"
PATCHED = REPO_ROOT / GENERATOR_PATH

SOURCE_PREFIX = (
    "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
    "PHerc0841/segments/20260221022814-auto_grown_20260220174252405/"
    "mesh/20260221022814-on-20250821151531-9.366um.tifxyz/"
)
FILE_HASHES = {
    "x.tif": "912895f58e1572f2b3ee43704e71f1b2e055af7b70f2d904168182a7e7b68e97",
    "y.tif": "49db4ef0e1fd68b107f12b940840afddbeb367fde1bffdbce99488fd84d86df0",
    "z.tif": "5fcf84a80ba590b3ab218da5aeb7780e4da8763cf45f4c9f7dee73bbfc6a1cb2",
}
WINDOWS = {
    "A": (36, 74, 40, 86),
    "B": (106, 144, 120, 166),
}
CROP = (128, 128, 128)
# Exact upstream commit used for the published before/after measurements.
# This remains stable even if later commits are added to the PR branch.
DEFAULT_BASELINE = "25f3c93c65b492be3b90e402213a1ba6b5fd624c"


def load_generator(source: str, label: str):
    # This geometry function does not call read_tifxyz when given points.
    # Stub only the import, to run without unrelated native volume-cartographer
    # bindings. Restore imports afterward.
    old_pkg = sys.modules.get("vesuvius")
    old_sub = sys.modules.get("vesuvius.tifxyz")
    stub_pkg = types.ModuleType("vesuvius")
    stub_pkg.__path__ = []
    stub_tif = types.ModuleType("vesuvius.tifxyz")

    def forbidden_read(*_args, **_kwargs):
        raise RuntimeError("Expected in-memory points from public TIFF data")

    stub_tif.read_tifxyz = forbidden_read
    sys.modules["vesuvius"] = stub_pkg
    sys.modules["vesuvius.tifxyz"] = stub_tif
    try:
        module = types.ModuleType(label)
        module.__file__ = str(PATCHED)
        exec(compile(source, str(PATCHED), "exec"), module.__dict__)
    finally:
        if old_pkg is None:
            sys.modules.pop("vesuvius", None)
        else:
            sys.modules["vesuvius"] = old_pkg
        if old_sub is None:
            sys.modules.pop("vesuvius.tifxyz", None)
        else:
            sys.modules["vesuvius.tifxyz"] = old_sub
    return module


def load_real_points(data_dir: Path):
    arrays = {}
    for name, sha256 in FILE_HASHES.items():
        path = data_dir / name
        if not path.is_file():
            req = urllib.request.Request(SOURCE_PREFIX + name, headers={
                "User-Agent": "ScrollPrize-bbox-benchmark/1.0"
            })
            with urllib.request.urlopen(req, timeout=45) as response:
                data = response.read(400_000)
            if len(data) >= 400_000:
                raise ValueError("Unexpected large TIFF file; refusing download")
            if hashlib.sha256(data).hexdigest() != sha256:
                raise ValueError("Official file checksum mismatch: " + name)
            path.write_bytes(data)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError("Local TIFF checksum mismatch: " + name)
        arrays[name[0]] = tifffile.imread(path)

    if any(a.shape != arrays["x"].shape for a in arrays.values()):
        raise ValueError("Coordinate grid shapes disagree")
    return arrays


def points_from_window(arrays, window):
    r0, r1, c0, c1 = window
    v = {axis: values[r0:r1, c0:c1] for axis, values in arrays.items()}
    valid = np.isfinite(v["x"]) & np.isfinite(v["y"]) & np.isfinite(v["z"])
    valid &= (v["x"] >= 0) & (v["y"] >= 0) & (v["z"] >= 0)
    return np.stack([v["z"][valid], v["y"][valid], v["x"][valid]], axis=1).astype(np.float64)


def record_signature(result):
    # Retain all relevant fields and ordering, not just bounding box counts.
    return [
        {
            "bbox": list(record["bbox"]),
            "grid_index": list(record["grid_index"]),
            "shift_axis": record["shift_axis"],
            "shift_voxels": int(record["shift_voxels"]),
            "point_indices": record["point_indices"].tolist(),
        }
        for record in result["final_records"]
    ]


def timed(module, points, overlap, crop_size):
    start = time.perf_counter()
    result = module._generate_segment_cover_records(
        points, crop_size, overlap=overlap, prune_bboxes=True, band_workers=1,
        show_progress=False,
    )
    return result, time.perf_counter() - start


def summarize_seconds(samples):
    """Calculate summary statistics from unrounded individual timings."""
    return {
        "mean": round(statistics.mean(samples), 4),
        "min": round(min(samples), 4),
        "median": round(statistics.median(samples), 4),
        "max": round(max(samples), 4),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=4,
                        help="Even number of paired trials per case (default: 4)")
    parser.add_argument("--window", choices=("A", "B", "both"), default="both")
    parser.add_argument("--overlap", choices=("0.0", "0.5", "both"), default="both")
    parser.add_argument("--crop-size", type=int, nargs=3, default=CROP, metavar=("Z", "Y", "X"),
                        help="3D input crop. Official Copy checkpoint currently specifies 128 384 384")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE,
                        help="Git ref with unmodified generator (default: pinned base commit)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON output path outside the repo")
    args = parser.parse_args(argv)
    if not 2 <= args.repeats <= 10 or args.repeats % 2:
        parser.error("--repeats must be even and between 2 and 10 to counterbalance run order")
    if any(int(d) <= 0 for d in args.crop_size):
        parser.error("--crop-size needs three positive integers")
    baseline = subprocess.run(
        ["git", "show", f"{args.baseline}:{GENERATOR_PATH}"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    patched = PATCHED.read_text(encoding="utf-8")
    before = load_generator(baseline, "bbox_baseline")
    after = load_generator(patched, "bbox_optimized")

    with tempfile.TemporaryDirectory(prefix="pherc0841-bbox-") as folder:
        arrays = load_real_points(Path(folder))

    chosen = ["A", "B"] if args.window == "both" else [args.window]
    report = {
        "scroll": "PHerc0841",
        "segment": "20260221022814 (s022814)",
        "real_mesh_url": SOURCE_PREFIX,
        "file_sha256": FILE_HASHES,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_build": list(platform.python_build()),
        "machine_arch": platform.machine(),
        "execution": "Python source, NumPy/tifffile wheels; no villa native build",
        "measurement": "perf_counter wall time; background CPU tasks may affect samples",
        "crop_size": list(args.crop_size),
        "prune_bboxes": True,
        "band_workers": 1,
        "baseline_ref": args.baseline,
        "baseline_sha256": hashlib.sha256(baseline.encode()).hexdigest(),
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "repeats": args.repeats,
        "cases": [],
    }
    for label in chosen:
        points = points_from_window(arrays, WINDOWS[label])
        if len(points) != 1748:
            raise AssertionError("Unexpected real mesh point count")
        overlap_values = (0.0, 0.5) if args.overlap == "both" else (float(args.overlap),)
        for overlap in overlap_values:
            before_times, after_times = [], []
            for i in range(args.repeats):
                # Alternate run order with an EVEN trial count so each
                # implementation runs first equally often.
                if i % 2 == 0:
                    old, t_old = timed(before, points, overlap, args.crop_size)
                    new, t_new = timed(after, points, overlap, args.crop_size)
                else:
                    new, t_new = timed(after, points, overlap, args.crop_size)
                    old, t_old = timed(before, points, overlap, args.crop_size)
                old_sig = record_signature(old)
                new_sig = record_signature(new)
                if old_sig != new_sig:
                    raise AssertionError(f"Output differs: {label}, overlap {overlap}")
                # Store full-resolution measurements; round only for reporting.
                before_times.append(t_old)
                after_times.append(t_new)

            record = {
                "window": label,
                "grid_window": list(WINDOWS[label]),
                "real_points": len(points),
                "overlap": overlap,
                "bbox_count": len(old_sig),
                "exact_record_output": True,
                "bbox_record_sha256": hashlib.sha256(
                    json.dumps(old_sig, sort_keys=True).encode()
                ).hexdigest(),
                "before_seconds": [round(value, 4) for value in before_times],
                "after_seconds": [round(value, 4) for value in after_times],
                "original_stats_seconds": summarize_seconds(before_times),
                "patched_stats_seconds": summarize_seconds(after_times),
                "median_before_seconds": round(statistics.median(before_times), 4),
                "median_after_seconds": round(statistics.median(after_times), 4),
            }
            record["median_speedup"] = round(
                statistics.median(before_times) / statistics.median(after_times), 3
            )
            report["cases"].append(record)
            print(json.dumps(record), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("REPORT", args.output, flush=True)
    print("PASS: all real-scroll bbox outputs match", flush=True)


if __name__ == "__main__":
    main()
