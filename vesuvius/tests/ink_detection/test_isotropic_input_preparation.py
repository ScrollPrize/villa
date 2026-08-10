from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import zarr

from vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input import (
    FORMAT_TAG,
    centered_slice,
    main,
    open_source_array,
    prepare_isotropic_input,
)


def _write_source(path: Path, data_ZYX: np.ndarray, *, level: str = "2") -> None:
    kwargs = {"mode": "w"}
    if int(zarr.__version__.split(".", 1)[0]) >= 3:
        kwargs["zarr_format"] = 2
    group = zarr.open_group(path, **kwargs)
    chunks = (21, *(2 for _ in data_ZYX.shape[1:]))
    if int(zarr.__version__.split(".", 1)[0]) >= 3:
        group.create_array(level, data=data_ZYX, chunks=chunks)
    else:
        group.create_dataset(level, data=data_ZYX, chunks=chunks)


def test_centered_slice_places_an_odd_margin_before_the_slice():
    assert centered_slice(84, 84) == (0, 84)
    assert centered_slice(85, 84) == (1, 85)
    assert centered_slice(86, 84) == (1, 85)
    with pytest.raises(ValueError, match="Cannot take 84 centered planes from 83"):
        centered_slice(83, 84)


def test_source_opening_accepts_bare_array_and_named_level(tmp_path):
    data_ZYX = np.zeros((84, 2, 3), dtype=np.uint8)
    group_path = tmp_path / "group.zarr"
    _write_source(group_path, data_ZYX)
    assert open_source_array(str(group_path), "2").shape == data_ZYX.shape
    with pytest.raises(KeyError, match=r"available: \['2'\]"):
        open_source_array(str(group_path), "4")

    array_path = tmp_path / "array.zarr"
    zarr.open_array(
        array_path,
        mode="w",
        shape=data_ZYX.shape,
        chunks=(21, 2, 3),
        dtype=np.uint8,
    )[:] = data_ZYX
    assert open_source_array(str(array_path), "ignored").shape == data_ZYX.shape


def test_preparation_writes_exact_float32_rounded_means_and_metadata(tmp_path):
    source_ZYX = np.zeros((86, 2, 3), dtype=np.uint8)
    centered_ZYX = np.arange(84 * 2 * 3, dtype=np.uint16).reshape(84, 2, 3) % 256
    source_ZYX[1:85] = centered_ZYX.astype(np.uint8)
    input_path = tmp_path / "source.zarr"
    output_path = tmp_path / "prepared.zarr"
    _write_source(input_path, source_ZYX)

    assert prepare_isotropic_input(
        str(input_path), output_path, level="2", workers=2
    ) == output_path
    assert output_path.exists()
    assert not output_path.with_name(output_path.name + ".partial").exists()
    expected_ZYX = np.rint(
        centered_ZYX.astype(np.float32).reshape(21, 4, 2, 3).mean(axis=1)
    ).astype(np.uint8)
    group = zarr.open_group(output_path, mode="r")
    np.testing.assert_array_equal(group["0"][:], expected_ZYX)
    assert dict(group.attrs) == {
        "format": FORMAT_TAG,
        "source": str(input_path),
        "source_level": "2",
        "source_shape_zyx": [86, 2, 3],
        "source_z_slice": [1, 85],
        "z_pool": "rounded mean of 4 centered source planes",
    }
    assert json.loads((output_path / ".zgroup").read_text()) == {"zarr_format": 2}


def test_preparation_uses_bankers_rounding_at_half_values(tmp_path):
    source_ZYX = np.zeros((84, 1, 2), dtype=np.uint8)
    source_ZYX[0:4, 0, 0] = [0, 1, 0, 1]
    source_ZYX[4:8, 0, 0] = [1, 2, 1, 2]
    source_ZYX[0:4, 0, 1] = [2, 3, 2, 3]
    input_path = tmp_path / "half-values.zarr"
    output_path = tmp_path / "prepared.zarr"
    _write_source(input_path, source_ZYX)

    prepare_isotropic_input(str(input_path), output_path, workers=1)
    output_ZYX = zarr.open_group(output_path, mode="r")["0"][:]
    assert output_ZYX[0, 0].tolist() == [0, 2]
    assert int(output_ZYX[1, 0, 0]) == 2


def test_preparation_refuses_destination_and_partial_on_rerun(tmp_path):
    source_ZYX = np.zeros((84, 1, 1), dtype=np.uint8)
    input_path = tmp_path / "source.zarr"
    _write_source(input_path, source_ZYX)
    output_path = tmp_path / "prepared.zarr"
    prepare_isotropic_input(str(input_path), output_path, workers=1)
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        prepare_isotropic_input(str(input_path), output_path, workers=1)

    other_output = tmp_path / "other.zarr"
    other_output.with_name(other_output.name + ".partial").mkdir()
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        prepare_isotropic_input(str(input_path), other_output, workers=1)


def test_preparation_rejects_wrong_source_contract(tmp_path):
    wrong_dtype = tmp_path / "wrong-dtype.zarr"
    _write_source(wrong_dtype, np.zeros((84, 1, 1), dtype=np.uint16))
    with pytest.raises(ValueError, match="Expected 3D uint8 source"):
        prepare_isotropic_input(str(wrong_dtype), tmp_path / "out.zarr")

    wrong_shape = tmp_path / "wrong-shape.zarr"
    _write_source(wrong_shape, np.zeros((84, 1, 1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="Expected 3D uint8 source"):
        prepare_isotropic_input(str(wrong_shape), tmp_path / "out-2.zarr")


def test_preparation_command_and_cli_module_help(tmp_path):
    input_path = tmp_path / "source.zarr"
    output_path = tmp_path / "prepared.zarr"
    _write_source(input_path, np.arange(84, dtype=np.uint8).reshape(84, 1, 1))
    assert main(
        [str(input_path), str(output_path), "--level", "2", "--workers", "1"]
    ) == 0
    assert zarr.open_group(output_path, mode="r")["0"].shape == (21, 1, 1)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input",
            "-h",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--workers" in completed.stdout
