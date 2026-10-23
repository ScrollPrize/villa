"""Depth must agree with the declared Z scale before a pyramid level is read."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input import (
    prepare_isotropic_input,
)
from vesuvius.ink_detection.volume_io import (
    PyramidDepthScaleError,
    require_pyramid_depth_scale,
)


def _write_pyramid(
    path: Path,
    *,
    shapes: dict[str, tuple[int, int, int]],
    scales: dict[str, list[float]] | None,
) -> None:
    kwargs: dict[str, object] = {"mode": "w"}
    if int(zarr.__version__.split(".", 1)[0]) >= 3:
        kwargs["zarr_format"] = 2
    group = zarr.open_group(path, **kwargs)
    create = group.create_array if "zarr_format" in kwargs else group.create_dataset
    for name, shape in shapes.items():
        create(name, shape=shape, chunks=shape, dtype="u1", fill_value=0)
    if scales is not None:
        group.attrs["multiscales"] = [
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
                            {"type": "scale", "scale": scale}
                        ],
                    }
                    for name, scale in scales.items()
                ],
            }
        ]


def test_missing_multiscale_metadata_is_left_alone(tmp_path):
    path = tmp_path / "bare.zarr"
    _write_pyramid(path, shapes={"2": (17, 4, 4)}, scales=None)
    root = zarr.open_group(path, mode="r")

    require_pyramid_depth_scale(root, "2", source=str(path))


def test_declared_full_depth_rejects_the_published_prefix_layout(tmp_path):
    # Public ink/841/w00/w00.zarr on 2026-10-22: level 0 is 65 deep with
    # scale [1, 1, 1]; level 2 is 17 deep with scale [1, 4, 4].
    path = tmp_path / "truncated.zarr"
    _write_pyramid(
        path,
        shapes={"0": (65, 8, 8), "2": (17, 2, 2)},
        scales={"0": [1.0, 1.0, 1.0], "2": [1.0, 4.0, 4.0]},
    )
    root = zarr.open_group(path, mode="r")

    with pytest.raises(PyramidDepthScaleError, match="depth 17"):
        require_pyramid_depth_scale(root, "2", source=str(path))
    require_pyramid_depth_scale(root, "0", source=str(path))


def test_declared_depth_pooling_accepts_ceil_halved_shapes(tmp_path):
    # Public ink/0139/w035 keeps the same 65-to-17 depth and sets Z scale to 4.
    path = tmp_path / "pooled.zarr"
    _write_pyramid(
        path,
        shapes={"0": (65, 8, 8), "2": (17, 2, 2), "5": (3, 1, 1)},
        scales={
            "0": [1.0, 1.0, 1.0],
            "2": [4.0, 4.0, 4.0],
            "5": [32.0, 32.0, 32.0],
        },
    )
    root = zarr.open_group(path, mode="r")

    require_pyramid_depth_scale(root, "2", source=str(path))
    require_pyramid_depth_scale(root, "5", source=str(path))


def test_depth_axis_follows_the_named_z_axis(tmp_path):
    path = tmp_path / "xyz.zarr"
    kwargs: dict[str, object] = {"mode": "w"}
    if int(zarr.__version__.split(".", 1)[0]) >= 3:
        kwargs["zarr_format"] = 2
    group = zarr.open_group(path, **kwargs)
    create = group.create_array if "zarr_format" in kwargs else group.create_dataset
    create("0", shape=(8, 8, 65), chunks=(8, 8, 65), dtype="u1", fill_value=0)
    create("2", shape=(2, 2, 17), chunks=(2, 2, 17), dtype="u1", fill_value=0)
    group.attrs["multiscales"] = [
        {
            "axes": [
                {"name": "x", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "z", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                    ],
                },
                {
                    "path": "2",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [4.0, 4.0, 1.0]}
                    ],
                },
            ],
        }
    ]

    with pytest.raises(PyramidDepthScaleError, match="depth 17"):
        require_pyramid_depth_scale(zarr.open_group(path, mode="r"), "2")


def test_retained_depth_accepts_an_unchanged_z_scale(tmp_path):
    path = tmp_path / "retained.zarr"
    _write_pyramid(
        path,
        shapes={"0": (65, 8, 8), "2": (65, 2, 2)},
        scales={"0": [2.4, 2.4, 2.4], "2": [2.4, 9.6, 9.6]},
    )

    require_pyramid_depth_scale(zarr.open_group(path, mode="r"), "2")


def test_preparation_refuses_a_long_enough_prefix_before_pooling(tmp_path):
    path = tmp_path / "prefix.zarr"
    _write_pyramid(
        path,
        shapes={"0": (200, 2, 2), "2": (84, 2, 2)},
        scales={"0": [1.0, 1.0, 1.0], "2": [1.0, 4.0, 4.0]},
    )
    level2 = zarr.open_group(path, mode="r+")["2"]
    level2[:] = np.arange(84 * 2 * 2, dtype=np.uint8).reshape(84, 2, 2)

    with pytest.raises(PyramidDepthScaleError, match="depth 84"):
        prepare_isotropic_input(str(path), tmp_path / "out.zarr", workers=1)
