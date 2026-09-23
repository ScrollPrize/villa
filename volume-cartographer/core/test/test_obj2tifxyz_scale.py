#!/usr/bin/env python3
"""CLI regression for vc_obj2tifxyz scale semantics (#1319)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


BIN = Path(sys.argv[1]).resolve()


def write_plane(path: Path) -> None:
    path.write_text(
        """v 0 0 10
v 100 0 10
v 100 100 10
v 0 100 10
vt 0 0
vt 1 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3
f 1/1 3/3 4/4
"""
    )


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BIN), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def load_scale(path: Path) -> tuple[float, float]:
    meta = json.loads((path / "meta.json").read_text())
    return float(meta["scale"][0]), float(meta["scale"][1])


with tempfile.TemporaryDirectory(prefix="obj2tifxyz-scale-") as tmp:
    root = Path(tmp)
    obj = root / "plane.obj"
    write_plane(obj)

    # 20 intervals span 100 volume units, so the final grid spacing is 5
    # units/cell and tifxyz scale must be its reciprocal: 0.2 cells/unit.
    out = root / "standalone"
    proc = run(str(obj), str(out), "20", "1.0")
    if proc.returncode != 0:
        raise SystemExit(
            "standalone conversion failed\nSTDOUT:\n"
            + proc.stdout
            + "\nSTDERR:\n"
            + proc.stderr
        )

    scale = load_scale(out)
    expected = 0.2
    if not all(math.isclose(v, expected, rel_tol=1e-4, abs_tol=1e-6) for v in scale):
        raise SystemExit(
            f"standalone scale {scale} != measured grid density {(expected, expected)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    # Source-aware mode has an explicit scale contract: preserve the source
    # density verbatim even if a different stretch factor is requested.
    sourced = root / "source-aware"
    proc2 = run(
        str(obj),
        str(sourced),
        "50",
        "1.0",
        f"--tifxyz-source={out}",
    )
    if proc2.returncode != 0:
        raise SystemExit(
            "source-aware conversion failed\nSTDOUT:\n"
            + proc2.stdout
            + "\nSTDERR:\n"
            + proc2.stderr
        )

    sourced_scale = load_scale(sourced)
    if not all(math.isclose(v, expected, rel_tol=1e-6, abs_tol=1e-7) for v in sourced_scale):
        raise SystemExit(
            f"source-aware scale {sourced_scale} did not preserve {(expected, expected)}"
        )

    print(f"standalone scale: {scale}")
    print(f"source-aware scale: {sourced_scale}")
    print("obj2tifxyz scale regression: PASS")
