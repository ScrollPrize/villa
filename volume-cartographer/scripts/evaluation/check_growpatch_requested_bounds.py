#!/usr/bin/env python3
"""Opt-in real-data regression for vc_grow_seg_from_seed resume growth limits.

Requires numpy and Pillow, an already built official CLI, and local real inputs.
No downloads or builds. Use a small, fully valid rectangular tifxyz resume mesh.
The fixture-root form accepts the archived VC-GROW-2104 portable-data/fixtures
layout and verifies its portable-data manifest before use. Example:

  python check_growpatch_requested_bounds.py --binary /path/vc_grow_seg_from_seed \
    --fixture-root /path/attack_20261020 --output /new/path/results

Or supply --volume /real/prediction.zarr --resume /real/mesh --params /params.json.
Exit 1 means a completed run violated the contract; exit 2 means invalid input,
native execution/timeout failure, or unreadable output (not a regression result).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def inventory(paths):
    result = {}
    for path in paths:
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for file in files:
            if file.is_file():
                result[str(file)] = {"bytes": file.stat().st_size, "sha256": digest(file)}
    return result


def read_surface(path):
    bands = {}
    for name in ("x", "y", "z", "generations"):
        with Image.open(path / (name + ".tif")) as image:
            bands[name] = np.array(image)
    shape = bands["x"].shape
    if len(shape) != 2 or any(band.shape != shape for band in bands.values()):
        raise ValueError("Expected matching 2-D xyz and generations TIFFs")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != "tifxyz":
        raise ValueError("Expected tifxyz metadata")
    valid = bands["x"] != -1
    if not valid.any() or any(not np.isfinite(bands[name][valid]).all() for name in "xyz"):
        raise ValueError("Empty or non-finite surface")
    return bands, meta


def inspect_output(path, original, start_gen, profile):
    bands, meta = read_surface(path)
    offset = meta.get("grid_offset")
    if not isinstance(offset, list) or len(offset) != 2 or any(type(v) is not int for v in offset):
        raise ValueError("Missing integer grid_offset [column, row] in output metadata")
    rows, cols = original["x"].shape
    rr, cc = np.nonzero(bands["x"] != -1)
    rr = rr - offset[1]
    cc = cc - offset[0]
    extra_rows, extra_cols = {"fill-zero": (0, 0), "right-one": (0, 1), "wide-control": (4, 4)}[profile]
    allowed = ((rr >= -extra_rows) & (rr < rows + extra_rows) &
               (cc >= -extra_cols) & (cc < cols + extra_cols))
    outside = (rr < 0) | (rr >= rows) | (cc < 0) | (cc >= cols)
    violations = []
    if not allowed.all():
        violations.append("valid points outside requested grid")
    if profile == "right-one" and (np.any(cc < 0) or not np.any(cc == cols)):
        violations.append("right-one must grow into exactly the first new right column")
    if profile == "wide-control" and not outside.any():
        violations.append("positive control produced no growth outside original mesh")
    r0, c0 = offset[1], offset[0]
    if r0 < 0 or c0 < 0 or r0 + rows > bands["x"].shape[0] or c0 + cols > bands["x"].shape[1]:
        violations.append("output lost the original mesh rectangle")
    else:
        region = np.s_[r0:r0 + rows, c0:c0 + cols]
        if np.any(bands["x"][region] == -1):
            violations.append("output lost original valid points")
        if profile == "fill-zero":
            if len(rr) != rows * cols or any(
                    bands[name][region].dtype != original[name].dtype or
                    bands[name][region].tobytes() != original[name].tobytes()
                    for name in "xyz"):
                violations.append("zero expansion must preserve all original coordinates exactly")
    if np.any(bands["generations"][bands["x"] != -1] == 0):
        violations.append("valid point has zero generation")
    if profile != "fill-zero" and outside.any():
        new_generations = bands["generations"][bands["x"] != -1][outside]
        if np.any(new_generations <= start_gen):
            violations.append("new points do not carry a later generation")
    return {"profile": profile, "valid_points": len(rr), "new_points": int(outside.sum()),
            "outside_requested_grid": int((~allowed).sum()), "grid_offset": offset,
            "original_grid_extent_rc": [int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())],
            "violations": violations, "passed": not violations}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--volume", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--path-prepend", action="append", default=[])
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    manifest = None
    if args.fixture_root:
        if any((args.volume, args.resume, args.params)):
            parser.error("Use either --fixture-root or all three --volume/--resume/--params")
        root = args.fixture_root.resolve(strict=True)
        args.volume = root / "portable-data/prediction-roi.zarr"
        args.resume = root / "fixtures/development"
        args.params = root / "fixtures/development-fill-zero.json"
        manifest = root / "portable-data/manifest.json"
        entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        for name, expected in entries.items():
            file = (manifest.parent / name).resolve(strict=True)
            if not file.is_relative_to(manifest.parent) or file.stat().st_size != expected["bytes"] or digest(file) != expected["sha256"]:
                raise ValueError("Fixture manifest mismatch: " + name)
    elif not all((args.volume, args.resume, args.params)):
        parser.error("Supply --fixture-root or all three --volume/--resume/--params")
    inputs = [path.resolve(strict=True) for path in (args.volume, args.resume, args.params)]
    volume, resume, params_path = inputs
    if manifest:
        inputs.append(manifest)
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    if any(output == path or output.is_relative_to(path) or path.is_relative_to(output) for path in inputs):
        raise ValueError("Output must be separate from every input")
    if output.exists():
        raise ValueError("Output directory must not already exist")
    original, _ = read_surface(resume)
    if np.any(original["x"] == -1) or np.any(original["generations"] == 0):
        raise ValueError("Use a fully valid rectangular resume with nonzero generations")
    start_gen = int(original["generations"].max())
    if start_gen > 65530:
        raise ValueError("Generation counter too close to uint16 limit")
    base = json.loads(params_path.read_text(encoding="utf-8"))
    before = inventory(inputs)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "inputs-before.json", before)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VC_GROWPATCH_RNG_SEED="2104")
    env["PATH"] = os.pathsep.join(args.path_prepend + [env.get("PATH", "")])
    results = []
    try:
        for profile in ("fill-zero", "right-one", "wide-control"):
            run = output / profile
            run.mkdir()
            params = dict(base)
            extra_rows, extra_cols = {"fill-zero": (0, 0), "right-one": (0, 1), "wide-control": (4, 4)}[profile]
            params.update(mode="resume", growth_scale=0, inpaint=False, rewind_gen=-1, resume_opt="skip",
                          cell_reopt_mode=False, thread_limit=1, generations=start_gen + (4 if profile == "right-one" else 3),
                          disable_grid_expansion=(profile == "fill-zero"), grow_extra_rows=extra_rows,
                          grow_extra_cols=extra_cols, grow_max_extra_rows=extra_rows, grow_max_extra_cols=extra_cols,
                          growth_directions=["right"] if profile == "right-one" else ["all"])
            params["snapshot-interval"] = 0
            save(run / "params.json", params)
            cache = run / "cache"
            cache.mkdir()
            child_env = dict(env, TEMP=str(cache), TMP=str(cache), TMPDIR=str(cache))
            argv = [str(binary), "--volume", str(volume), "--target-dir", str(run / "surface"),
                    "--params", str(run / "params.json"), "--resume", str(resume), "--resume-opt", "skip",
                    "--segment-name", "requested-bounds-regression", "--rng-seed", "2104"]
            record = {"argv": argv, "binary_sha256": digest(binary), "params_sha256": digest(run / "params.json"), "status": "STARTING"}
            save(run / "process.json", record)
            started = time.monotonic()
            with (run / "stdout.txt").open("wb") as out, (run / "stderr.txt").open("wb") as err:
                process = subprocess.Popen(argv, cwd=run, env=child_env, stdout=out, stderr=err)
                record.update(status="RUNNING", pid=process.pid)
                save(run / "process.json", record)
                try:
                    code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = process.wait()
                    record["timed_out_killed_and_reaped"] = True
                except BaseException:
                    process.kill()
                    code = process.wait()
                    record.update(status="INTERRUPTED", exit_code=code,
                                  wall_seconds=time.monotonic() - started)
                    save(run / "process.json", record)
                    raise
            record.update(status="COMPLETED", exit_code=code, wall_seconds=time.monotonic() - started)
            save(run / "process.json", record)
            if code != 0 or record.get("timed_out_killed_and_reaped"):
                raise RuntimeError("Native run failed/timed out; inspect " + str(run / "process.json"))
            result = inspect_output(run / "surface", original, start_gen, profile)
            result["wall_seconds"] = record["wall_seconds"]
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        after = inventory(inputs)
        save(output / "inputs-after.json", after)
        save(output / "results.json", {"results": results, "input_bytes_unchanged": before == after,
             "scope": "Requested-grid contract on caller-supplied real input; growth_scale=0, no inpaint. No accuracy/generalization claim."})
        if before != after:
            raise RuntimeError("Input bytes changed during run")
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print("CHECK ERROR: " + str(error), file=sys.stderr)
        sys.exit(2)
