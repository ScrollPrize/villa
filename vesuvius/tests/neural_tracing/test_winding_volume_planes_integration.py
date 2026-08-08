from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vesuvius.neural_tracing.winding_models.volume_plane_extractor import (
    VolumePlaneExtractor,
)

CONFIG_PATH = (
    Path(__file__).parents[2] / "src/vesuvius/neural_tracing/winding_models/config.json"
)


def test_real_configured_volume_intersecting_planes_match_coordinate_sampler() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    volume_cfg = cfg["datasets"][0]
    volume_path = VolumePlaneExtractor.scaled_volume_path(
        Path(volume_cfg["volume_path"]), int(volume_cfg["volume_scale"])
    )
    if not volume_path.exists():
        pytest.skip(f"configured winding volume is unavailable: {volume_path}")

    shape = (int(cfg["plane_height"]), int(cfg["ray_length"]))
    segment_to_volume = VolumePlaneExtractor.load_segment_to_volume_transform(
        Path(volume_cfg["volume_path"]),
        int(volume_cfg["volume_scale"]),
        segment_downscale=int(volume_cfg["segment_downscale"]),
        use_registration=(
            volume_cfg.get("segment_volume_id") != volume_cfg.get("volume_id")
        ),
    )
    extractor = VolumePlaneExtractor(
        [volume_path],
        shape=shape,
        spacing=float(cfg["plane_spacing"]),
        sampling=cfg["plane_sampling"],
        tile_size=int(cfg["plane_tile_size"]),
        cache_bytes=512 * 1024 * 1024,
        segment_to_volume_xyz=[segment_to_volume],
    )
    volume = extractor._volume(0)
    volume_shape_xyz = np.asarray(volume.shape_xyz, dtype=np.float64)
    volume_to_segment = np.linalg.inv(segment_to_volume)
    segment_center = (
        0.5 * volume_shape_xyz @ volume_to_segment[:3, :3].T + volume_to_segment[:3, 3]
    )
    direction = np.asarray([0.31, -0.23, 0.922], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    ray_origin = segment_center - 0.5 * (shape[1] - 1) * extractor.spacing * direction
    origins, x_steps, y_steps = extractor.intersecting_geometry(direction, ray_origin)
    sample_origins, sample_x_steps, sample_y_steps = extractor.sampling_geometry(
        0, (origins, x_steps, y_steps)
    )

    images, valid, _ = volume.sample_planes(
        sample_origins,
        sample_x_steps,
        sample_y_steps,
        shape,
        sampling=extractor.sampling,
        tile_size=extractor.tile_size,
    )

    height, width = shape
    coords = np.empty((2, height, width, 3), dtype=np.float32)
    for plane in range(2):
        for y in range(height):
            row_origin = sample_origins[plane] + np.float32(y) * sample_y_steps[plane]
            for x in range(width):
                coords[plane, y, x] = row_origin + np.float32(x) * sample_x_steps[plane]
    flat_coords = coords.reshape(2 * height, width, 3)
    input_valid = np.ones(flat_coords.shape[:2], dtype=bool)
    expected, expected_valid, _ = volume.sample_coords(
        flat_coords,
        input_valid,
        sampling=extractor.sampling,
        tile_size=extractor.tile_size,
    )

    # The native affine loop and NumPy can differ by one float32 coordinate ULP;
    # after trilinear interpolation that can move an integer output by one.
    np.testing.assert_allclose(images, expected.reshape(images.shape), atol=1, rtol=0)
    np.testing.assert_array_equal(valid, expected_valid.reshape(valid.shape))
    assert valid.all()
    assert np.count_nonzero(images) > images.size // 2
