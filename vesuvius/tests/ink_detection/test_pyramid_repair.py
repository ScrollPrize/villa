"""Truncated mean pyramids rebuild to a full-depth successive 2x mean."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from vesuvius.ink_detection.pyramid_repair import (
    PyramidRepairError,
    repair_pyramid,
    rounded_mean_downsample,
    successive_rounded_mean,
)


def _open_group(path: Path, *, mode: str):
    kwargs: dict[str, object] = {"mode": mode}
    if mode == "w" and int(zarr.__version__.split(".", 1)[0]) >= 3:
        kwargs["zarr_format"] = 2
    return zarr.open_group(path, **kwargs)


def _create(group, name: str, data: np.ndarray):
    kwargs = {
        "shape": data.shape,
        "chunks": data.shape,
        "dtype": data.dtype,
        "fill_value": 0,
    }
    array = (
        group.create_array(name, **kwargs)
        if hasattr(group, "create_array")
        else group.create_dataset(name, **kwargs)
    )
    array[:] = data
    return array


def _scales(level_scales: dict[str, list[float]]) -> dict:
    return {
        "multiscales": [
            {
                "version": "0.4",
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "path": name,
                        "coordinateTransformations": [
                            {"type": "scale", "scale": scale},
                            {"type": "translation", "translation": [0.0, 0.0, 0.0]},
                        ],
                    }
                    for name, scale in level_scales.items()
                ],
                "metadata": {"downsampling_method": "mean"},
            }
        ]
    }


def _xy_mean(array: np.ndarray) -> np.ndarray:
    pooled = np.empty(array.shape[:1] + tuple((n + 1) // 2 for n in array.shape[1:]), dtype=array.dtype)
    for index in range(array.shape[0]):
        pooled[index] = rounded_mean_downsample(array[index])
    return pooled


def test_two_by_two_block_uses_one_rounded_mean():
    block = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    # 0+1+2+3+4+5+6+7 = 28; (28 + 4) // 8 = 4
    assert rounded_mean_downsample(block).tolist() == [[[4]]]


def test_odd_edge_keeps_the_leftover_voxel():
    block = np.array([[[1, 2, 9], [3, 4, 9], [9, 9, 8]]], dtype=np.uint8)
    pooled = rounded_mean_downsample(block[0])
    # 2x2 mean is (1+2+3+4+2)//4 = 3; leftover row and column stay 9 and 8.
    assert pooled.tolist() == [[3, 9], [9, 8]]


def test_repair_replaces_a_depth_prefix_and_keeps_level_zero(tmp_path: Path):
    level0 = np.arange(65 * 8 * 8, dtype=np.uint16).reshape(65, 8, 8)
    xy = _xy_mean(_xy_mean(level0))
    prefix = xy[: xy.shape[0] // 4 + (xy.shape[0] % 4 > 0)]
    # Two XY passes on 8 produce 2. Crop Z from 65 to the pooled depth, 17.
    assert prefix.shape == (17, 2, 2)
    source = tmp_path / "source.zarr"
    group = _open_group(source, mode="w")
    _create(group, "0", level0)
    _create(group, "2", prefix)
    group.attrs.update(_scales({"0": [1.0, 1.0, 1.0], "2": [1.0, 4.0, 4.0]}))

    destination = tmp_path / "repaired.zarr"
    rebuilt = repair_pyramid(source, destination)

    assert rebuilt == ("2",)
    repaired = _open_group(destination, mode="r")
    assert np.array_equal(np.asarray(repaired["0"]), level0)
    expected = successive_rounded_mean(level0, 2)
    assert expected.shape == (17, 2, 2)
    assert np.array_equal(np.asarray(repaired["2"]), expected)
    assert not np.array_equal(expected, prefix)
    transforms = repaired.attrs["multiscales"][0]["datasets"][1]["coordinateTransformations"]
    assert transforms[0]["scale"] == [4.0, 4.0, 4.0]
    assert transforms[1]["translation"] == [0.0, 0.0, 0.0]
    assert repaired.attrs["multiscales"][0]["metadata"]["rebuilt_levels"] == ["2"]


def test_coarser_level_pools_the_repaired_parent(tmp_path: Path):
    level0 = np.arange(8 * 8 * 8, dtype=np.uint8).reshape(8, 8, 8)
    first = _xy_mean(level0)[:4]
    second = _xy_mean(first)[:2]
    source = tmp_path / "chain.zarr"
    group = _open_group(source, mode="w")
    _create(group, "0", level0)
    _create(group, "1", first)
    _create(group, "2", second)
    group.attrs.update(
        _scales(
            {
                "0": [1.0, 1.0, 1.0],
                "1": [1.0, 2.0, 2.0],
                "2": [1.0, 4.0, 4.0],
            }
        )
    )

    repair_pyramid(source, tmp_path / "chain-out.zarr")
    repaired = _open_group(tmp_path / "chain-out.zarr", mode="r")
    level1 = np.asarray(repaired["1"])
    level2 = np.asarray(repaired["2"])
    assert np.array_equal(level1, successive_rounded_mean(level0, 1))
    assert np.array_equal(level2, rounded_mean_downsample(level1))
    assert np.array_equal(level2, successive_rounded_mean(level0, 2))


def test_declared_depth_pool_is_left_unwritten(tmp_path: Path):
    source = tmp_path / "good.zarr"
    group = _open_group(source, mode="w")
    level0 = np.zeros((65, 8, 8), dtype=np.uint8)
    pooled = successive_rounded_mean(level0, 2)
    _create(group, "0", level0)
    _create(group, "2", pooled)
    group.attrs.update(_scales({"0": [1.0, 1.0, 1.0], "2": [4.0, 4.0, 4.0]}))

    with pytest.raises(PyramidRepairError, match="no truncated mean level"):
        repair_pyramid(source, tmp_path / "unused.zarr")
    assert not (tmp_path / "unused.zarr").exists()


def test_existing_destination_is_refused(tmp_path: Path):
    source = tmp_path / "source.zarr"
    group = _open_group(source, mode="w")
    level0 = np.ones((4, 4, 4), dtype=np.uint8)
    _create(group, "0", level0)
    _create(group, "1", level0[:2, :2, :2])
    group.attrs.update(_scales({"0": [1.0, 1.0, 1.0], "1": [1.0, 2.0, 2.0]}))
    destination = tmp_path / "already"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        repair_pyramid(source, destination)
