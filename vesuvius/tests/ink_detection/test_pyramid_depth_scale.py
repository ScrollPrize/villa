"""A pyramid level must keep the depth its Z scale declares."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from vc_reader.reader import create_group

from vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input import (
    prepare_isotropic_input,
)
from vesuvius.ink_detection.volume_io import (
    PyramidDepthScaleError,
    open_volume,
    open_volume_root,
    select_volume_level,
)


def _scales(*levels: tuple[str, tuple[int, int, int], tuple[float, float, float]]) -> dict:
    return {
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "path": name,
                        "coordinateTransformations": [{"type": "scale", "scale": list(scale)}],
                    }
                    for name, _shape, scale in levels
                ],
            }
        ]
    }


def _group(path: Path, *levels: tuple[str, tuple[int, ...], tuple[float, ...]], axes=None) -> str:
    attrs = _scales(*((name, shape, scale) for name, shape, scale in levels))
    if axes is not None:
        attrs["multiscales"][0]["axes"] = axes
    group = create_group(path, attrs=attrs, overwrite=True)
    for name, shape, _scale in levels:
        group.create_array(name, shape=shape, chunks=shape, dtype=np.uint8)
    return str(path)


def test_missing_metadata_and_level_zero_are_unchanged(tmp_path):
    bare = create_group(tmp_path / "bare.zarr", overwrite=True)
    bare.create_array("2", shape=(17, 4, 4), chunks=(17, 4, 4), dtype=np.uint8)
    assert open_volume(str(tmp_path / "bare.zarr"), 2).shape == (17, 4, 4)

    kept = _group(
        tmp_path / "full.zarr",
        ("0", (65, 8, 8), (1.0, 1.0, 1.0)),
        ("2", (17, 2, 2), (1.0, 4.0, 4.0)),
    )
    assert open_volume(kept, 0).shape == (65, 8, 8)


def test_unchanged_z_scale_rejects_a_shortened_level(tmp_path):
    path = _group(
        tmp_path / "truncated.zarr",
        ("0", (65, 8, 8), (1.0, 1.0, 1.0)),
        ("2", (17, 2, 2), (1.0, 4.0, 4.0)),
    )
    with pytest.raises(PyramidDepthScaleError, match="depth 17") as opened:
        open_volume(path, 2)
    message = str(opened.value)
    assert message.endswith("Rebuild this level from the full stack.")
    assert "set its Z scale" not in message
    root = open_volume_root(path)
    with pytest.raises(PyramidDepthScaleError, match="ratio near 1"):
        select_volume_level(root, 2, source=path)


def test_a_short_level_with_the_wrong_z_scale_is_rejected(tmp_path):
    path = _group(
        tmp_path / "short.zarr",
        ("0", (65, 8, 8), (1.0, 1.0, 1.0)),
        ("3", (9, 1, 1), (4.0, 8.0, 8.0)),
    )
    with pytest.raises(PyramidDepthScaleError, match="expects about 16.25"):
        open_volume(path, 3)


def test_declared_depth_pooling_is_accepted(tmp_path):
    path = _group(
        tmp_path / "pooled.zarr",
        ("0", (65, 64, 64), (1.0, 1.0, 1.0)),
        ("2", (17, 16, 16), (4.0, 4.0, 4.0)),
        ("5", (3, 2, 2), (32.0, 32.0, 32.0)),
    )
    assert open_volume(path, 2).shape[0] == 17
    assert open_volume(path, 5).shape[0] == 3


def test_z_axis_is_found_when_it_is_not_first(tmp_path):
    path = _group(
        tmp_path / "xyz.zarr",
        ("0", (8, 8, 65), (1.0, 1.0, 1.0)),
        ("2", (2, 2, 17), (4.0, 4.0, 1.0)),
        axes=[
            {"name": "x", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "z", "type": "space"},
        ],
    )
    with pytest.raises(PyramidDepthScaleError, match="depth 17"):
        open_volume(path, 2)


def test_physical_scale_with_every_slice_kept_is_accepted(tmp_path):
    path = _group(
        tmp_path / "physical.zarr",
        ("0", (65, 8, 8), (2.4, 2.4, 2.4)),
        ("2", (65, 2, 2), (2.4, 9.6, 9.6)),
    )
    assert open_volume(path, 2).shape == (65, 2, 2)


def test_prepare_refuses_a_prefix_long_enough_to_pool(tmp_path):
    path = _group(
        tmp_path / "prefix.zarr",
        ("0", (200, 2, 2), (1.0, 1.0, 1.0)),
        ("2", (84, 2, 2), (1.0, 4.0, 4.0)),
    )
    with pytest.raises(PyramidDepthScaleError, match="depth 84"):
        prepare_isotropic_input(path, tmp_path / "out.zarr", level="2", workers=1)
    assert not (tmp_path / "out.zarr").exists()
    assert not (tmp_path / "out.zarr.partial").exists()
