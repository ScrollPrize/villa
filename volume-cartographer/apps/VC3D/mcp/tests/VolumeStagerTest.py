#!/usr/bin/env python3
"""Focused regression tests for bounded Zarr staging and coordinate metadata."""
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np


def load_stager(path: Path):
    spec = importlib.util.spec_from_file_location("volume_stager", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stager", required=True, type=Path)
    arguments = parser.parse_args()
    stager = load_stager(arguments.stager)

    assert stager.allowed_remote_uri("s3://vesuvius-challenge-open-data/example.zarr")
    assert stager.allowed_remote_uri("https://dl.ash2txt.org/example.zarr/")
    assert not stager.allowed_remote_uri("https://dl.ash2txt.org.evil.example/example.zarr")
    assert not stager.allowed_remote_uri("https://dl.ash2txt.org/a/%2e%2e/secrets")
    assert not stager.allowed_remote_uri("https://dl.ash2txt.org/example.zarr?token=secret")

    with tempfile.TemporaryDirectory(prefix="vc-volume-stager-") as temporary:
        root = Path(temporary)
        zarr = root / "input.zarr"
        (zarr / "0").mkdir(parents=True)
        (zarr / ".zgroup").write_text('{"zarr_format":2}\n')
        (zarr / "0" / ".zarray").write_text(
            json.dumps(
                {
                    "zarr_format": 2,
                    "shape": [8, 9, 10],
                    "chunks": [8, 9, 10],
                    "dtype": "<u2",
                    "compressor": None,
                    "fill_value": 0,
                    "filters": None,
                    "order": "C",
                    "dimension_separator": ".",
                }
            )
        )
        values = np.arange(8 * 9 * 10, dtype="<u2").reshape(8, 9, 10)
        (zarr / "0" / "0.0.0").write_bytes(values.tobytes(order="C"))
        request = {
            "source": {
                "kind": "local_zarr",
                "path": str(zarr),
                "array_path": "0",
                "scale": 2,
                "voxel_spacing": [2.0, 3.0, 4.0],
                "origin_xyz": [100.0, 200.0, 300.0],
            },
            "region": {
                "x": 2,
                "y": 3,
                "z": 4,
                "width": 5,
                "height": 4,
                "depth": 3,
                "space": "ct_l0_xyz",
            },
        }
        output = root / "output"
        manifest = stager.stage(request, output)
        staged = np.load(output / "staged-volume.npy", allow_pickle=False)
        np.testing.assert_array_equal(staged, values[4:7, 3:7, 2:7])
        assert manifest["submitted_region_xyz"] == request["region"]
        assert manifest["array_slices_zyx"] == {
            "z": {"start": 4, "stop": 7},
            "y": {"start": 3, "stop": 7},
            "x": {"start": 2, "stop": 7},
        }
        assert manifest["scale"] == 2
        assert manifest["array_path"] == "0"
        assert manifest["voxel_spacing"] == [2.0, 3.0, 4.0]
        assert manifest["origin_xyz"] == [104.0, 209.0, 316.0]

    print("VolumeStagerTest passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
