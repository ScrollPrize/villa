"""fill_inner_outer_labels runs end to end on every zarr the package accepts (pyproject: zarr>=2.18.7,<4).

create_output_zarr called Group.create_dataset, which zarr 3 removed, so the script crashed under
zarr 3 before processing any slice. It now creates the output level through create_level_dataset.
"""
import json
import sys

import numpy as np
import pytest
import zarr
from numcodecs import Blosc

from vesuvius.image_proc.run.zarr_tasks.utils import create_level_dataset
from vesuvius.scripts import fill_inner_outer_labels as fill


def _ring(size):
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    return ((r > size * 0.3) & (r < size * 0.4)).astype(np.uint8) * 255


def _write_input(path):
    # Level 1 holds a ring per slice, level 0 is the same ring at twice the resolution.
    levels = {"0": np.stack([_ring(64)] * 8), "1": np.stack([_ring(32)] * 4)}
    for name, data in levels.items():
        arr = create_level_dataset(str(path), name, data.shape, (4, 16, 16), data.dtype, Blosc(cname="zstd", clevel=1))
        arr[...] = data
    return levels


def _run(tmp_path, monkeypatch, compute_level, output_level):
    out = tmp_path / "out.zarr"
    monkeypatch.setattr(sys, "argv", [
        "fill_inner_outer_labels", str(tmp_path / "in.zarr"), str(out), "--skip-outer", "--workers", "1",
        "--compute-level", str(compute_level), "--output-level", str(output_level),
    ])
    fill.main()
    return out


@pytest.mark.parametrize("compute_level, output_level", [(1, 1), (1, 0)])
def test_fills_the_inner_region(tmp_path, monkeypatch, compute_level, output_level):
    levels = _write_input(tmp_path / "in.zarr")
    out = _run(tmp_path, monkeypatch, compute_level, output_level)
    result = zarr.open(str(out / str(output_level)), mode="r")[...]
    target = levels[str(output_level)]
    expected = np.zeros_like(target)
    for z in range(target.shape[0]):
        inner = fill.detect_inner_region(levels[str(compute_level)][z // 2 ** (compute_level - output_level)])
        expected[z][fill.upscale_mask(inner, target.shape[1:])] = 255
    assert expected.any()
    np.testing.assert_array_equal(result, expected)


def test_output_is_zarr_v2_with_input_attrs(tmp_path, monkeypatch):
    _write_input(tmp_path / "in.zarr")
    zarr.open_group(str(tmp_path / "in.zarr"), mode="r+").attrs["note"] = "kept"
    out = _run(tmp_path, monkeypatch, 1, 1)
    assert json.loads((out / ".zgroup").read_text()) == {"zarr_format": 2}
    assert not (out / "zarr.json").exists()
    assert (out / "1" / ".zarray").exists()
    assert json.loads((out / ".zattrs").read_text())["note"] == "kept"
