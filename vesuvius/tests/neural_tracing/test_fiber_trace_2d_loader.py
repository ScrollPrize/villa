from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

from vesuvius.neural_tracing.fiber_trace_2d.augmentation import (
    FiberStripAugmentConfig,
    FiberStripAugmentParams,
    apply_value_augmentation,
    limit_augmentation_rows,
    random_combined_augmentation,
    smooth_offset_field,
    source_coordinate_grid_for_output,
    transformed_centerline_coords,
    transformed_source_point_coords,
)
from vesuvius.neural_tracing.fiber_trace_2d.direction import (
    build_direction_supervision,
    cp_neighborhood_yx,
    encode_lasagna_direction_xy,
)
from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import load_vc3d_fiber
from vesuvius.neural_tracing.fiber_trace_2d.loader import (
    FiberStrip2DLoader,
    ZarrChunkRequest,
    load_config,
    strip_z_offsets_from_count_step,
)
from vesuvius.neural_tracing.fiber_trace_2d.train import (
    _draw_predicted_cp_direction,
    _should_print_training_step,
    _test_loader_config_from_raw,
    _training_config_from_raw,
    prefetch_training,
    run_training,
)
from vesuvius.neural_tracing.fiber_trace_2d.runner import _export_augment_contact_sheet
from vesuvius.neural_tracing.fiber_trace_2d.sampling import NumpyZarrCoordinateSampler
from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import (
    build_side_strip_patch_grid,
    build_side_strip_patch_grid_from_line_window,
    build_side_strip_patch_grid_from_line_window_torch,
    side_strip_line_window,
)


def _write_fiber(
    path: Path,
    points: list[list[float]] | None = None,
    control_points: list[list[float]] | None = None,
) -> Path:
    points = points or [[0.0, 20.0, 20.0], [10.0, 20.0, 20.0], [20.0, 20.0, 20.0]]
    control_points = control_points or points
    obj = {
        "type": "vc3d_fiber",
        "version": 1,
        "line_points": points,
        "control_points": control_points,
        "generation": 1,
    }
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _write_zarr(path: Path) -> Path:
    root = zarr.open_group(str(path), mode="w")
    arr = root.create_dataset(
        "0",
        shape=(48, 48, 48),
        chunks=(16, 16, 16),
        dtype="float32",
        compressor=None,
    )
    z, y, x = np.indices(arr.shape, dtype=np.float32)
    arr[:] = z * 10000.0 + y * 100.0 + x
    return path


def _write_lasagna_manifest(tmp_path: Path) -> Path:
    normals_path = tmp_path / "normals.zarr"
    root = zarr.open_group(str(normals_path), mode="w")
    arr = root.create_dataset(
        "0",
        shape=(3, 48, 48, 48),
        chunks=(1, 16, 16, 16),
        dtype="uint8",
        compressor=None,
    )
    arr[0, :, :, :] = 1
    arr[1, :, :, :] = 128
    arr[2, :, :, :] = 255
    manifest = {
        "version": 2,
        "source_to_base": 1.0,
        "base_shape_zyx": [48, 48, 48],
        "grad_mag_encode_scale": 1000.0,
        "grad_mag_factor": 1.0,
        "umbilicus_json": "umbilicus.json",
        "groups": {
            "normals": {
                "zarr": "normals.zarr",
                "scaledown": 0,
                "channels": ["grad_mag", "nx", "ny"],
            }
        },
    }
    (tmp_path / "umbilicus.json").write_text("{}", encoding="utf-8")
    path = tmp_path / "test.lasagna.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_config(tmp_path: Path, *, batch_size: int = 2) -> Path:
    fiber = _write_fiber(tmp_path / "fiber.json")
    volume = _write_zarr(tmp_path / "vol.zarr")
    manifest = _write_lasagna_manifest(tmp_path)
    config = {
        "batch_size": batch_size,
        "patch_shape_hw": [3, 3],
        "strip_z_offset_count": 3,
        "strip_z_offset_step": 1.0,
        "augment_device": "cpu",
        "seed": 123,
        "datasets": [
            {
                "fiber_paths": [str(fiber)],
                "base_volume_path": str(volume),
                "base_volume_scale": 0,
                "lasagna_manifest_path": str(manifest),
            }
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _test_sampler_factory(**kwargs):
    return NumpyZarrCoordinateSampler(
        kwargs["array"],
        level_spacing_base=kwargs["level_spacing_base"],
    )


def _make_loader(config) -> FiberStrip2DLoader:
    return FiberStrip2DLoader(config, sampler_factory=_test_sampler_factory)


def test_fiber_parser_reuse(tmp_path: Path) -> None:
    path = _write_fiber(tmp_path / "fiber.json")
    fiber = load_vc3d_fiber(path)
    assert fiber.control_points_xyz.shape == (3, 3)
    assert np.allclose(fiber.control_points_zyx[0], [20.0, 20.0, 0.0])


def test_side_strip_grid_matches_vc3d_simple_line_boundaries(tmp_path: Path) -> None:
    fiber = load_vc3d_fiber(_write_fiber(tmp_path / "fiber.json"))
    grid = build_side_strip_patch_grid(
        fiber,
        control_point_index=1,
        patch_shape_hw=(3, 3),
        strip_z_offset=0.0,
        sampled_normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    )

    assert np.allclose(grid.coords_xyz[0, 1], [10.0, 20.0, 19.0])
    assert np.allclose(grid.coords_xyz[1, 1], [10.0, 20.0, 20.0])
    assert np.allclose(grid.coords_xyz[2, 1], [10.0, 20.0, 21.0])
    assert np.all(grid.valid_mask)


def test_strip_pixel_spacing_is_in_base_voxels(tmp_path: Path) -> None:
    fiber = load_vc3d_fiber(_write_fiber(tmp_path / "fiber.json"))
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    strip = build_side_strip_patch_grid(
        fiber,
        control_point_index=1,
        patch_shape_hw=(3, 3),
        strip_z_offset=0.0,
        sampled_normal=normal,
        pixel_spacing_base=2.0,
    )

    assert np.allclose(strip.coords_xyz[1, :, 0], [8.0, 10.0, 12.0])
    assert np.allclose(strip.coords_xyz[:, 1, 2], [18.0, 20.0, 22.0])


def test_torch_side_strip_grid_matches_numpy_path(tmp_path: Path) -> None:
    fiber = load_vc3d_fiber(
        _write_fiber(
            tmp_path / "fiber.json",
            points=[
                [0.0, 20.0, 20.0],
                [8.0, 22.0, 20.0],
                [16.0, 20.0, 20.0],
                [24.0, 18.0, 20.0],
            ],
            control_points=[
                [8.0, 22.0, 20.0],
                [16.0, 20.0, 20.0],
            ],
        )
    )
    window = side_strip_line_window(
        fiber,
        control_point_index=1,
        patch_shape_hw=(5, 7),
        pixel_spacing_base=2.0,
        interpolation_point_margin=0,
    )
    normals = np.repeat(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), window.line_points_xyz.shape[0], axis=0)

    numpy_grid = build_side_strip_patch_grid_from_line_window(
        window,
        patch_shape_hw=(5, 7),
        strip_z_offset=0.25,
        sampled_normals=normals,
        pixel_spacing_base=2.0,
    )
    torch_grid = build_side_strip_patch_grid_from_line_window_torch(
        window,
        patch_shape_hw=(5, 7),
        strip_z_offset=0.25,
        sampled_normals=normals,
        pixel_spacing_base=2.0,
        device=torch.device("cpu"),
    )

    assert np.array_equal(torch_grid.valid_mask, numpy_grid.valid_mask)
    assert np.allclose(torch_grid.coords_xyz, numpy_grid.coords_xyz, atol=1.0e-5)
    assert np.allclose(torch_grid.coords_zyx, numpy_grid.coords_zyx, atol=1.0e-5)


def test_side_strip_uses_fiber_line_points_not_sparse_control_chord(tmp_path: Path) -> None:
    fiber = load_vc3d_fiber(
        _write_fiber(
            tmp_path / "fiber.json",
            points=[
                [0.0, 20.0, 20.0],
                [10.0, 30.0, 20.0],
                [20.0, 20.0, 20.0],
            ],
            control_points=[
                [0.0, 20.0, 20.0],
                [10.0, 30.0, 20.0],
                [20.0, 20.0, 20.0],
            ],
        )
    )
    grid = build_side_strip_patch_grid(
        fiber,
        control_point_index=1,
        patch_shape_hw=(1, 3),
        strip_z_offset=0.0,
        sampled_normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        pixel_spacing_base=2.0,
    )

    assert grid.coords_xyz[0, 1, 1] > 29.0
    assert grid.coords_xyz[0, 0, 0] < grid.coords_xyz[0, 2, 0]


def test_side_strip_rejects_control_point_not_in_line_points(tmp_path: Path) -> None:
    fiber = load_vc3d_fiber(
        _write_fiber(
            tmp_path / "fiber.json",
            points=[
                [0.0, 20.0, 20.0],
                [10.0, 30.0, 20.0],
                [20.0, 20.0, 20.0],
            ],
            control_points=[
                [10.0, 20.0, 20.0],
            ],
        )
    )

    with pytest.raises(ValueError, match="not an exact member of line_points"):
        build_side_strip_patch_grid(
            fiber,
            control_point_index=0,
            patch_shape_hw=(1, 3),
            strip_z_offset=0.0,
            sampled_normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            pixel_spacing_base=2.0,
        )


def test_strip_z_offsets_generated_from_count_and_step() -> None:
    assert strip_z_offsets_from_count_step(16, 1.0) == tuple(float(v) for v in range(-7, 9))
    assert strip_z_offsets_from_count_step(3, 2.0) == (-2.0, 0.0, 2.0)


def test_config_rejects_literal_strip_z_offsets(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["strip_z_offsets"] = [-1, 0, 1]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="strip_z_offsets was removed"):
        load_config(config_path)


def test_config_and_loader_batch_shape(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, batch_size=2))
    loader = _make_loader(config)
    batch = loader.load_batch(0)

    assert batch.images.shape == (2, 3, 1, 3, 3)
    assert batch.coords_zyx.shape == (2, 3, 3, 3, 3)
    assert batch.valid_mask.shape == (2, 3, 3, 3)
    assert not hasattr(batch, "planar_images")
    assert batch.strip_z_offsets.tolist() == [-1.0, 0.0, 1.0]
    assert batch.control_point_indices.shape == (2,)


def test_loader_skips_whole_fiber_with_out_of_volume_control_point(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    bad_fiber = _write_fiber(
        tmp_path / "bad_fiber.json",
        points=[[0.0, 20.0, 20.0], [10.0, 20.0, 20.0], [20.0, 20.0, 60.0]],
        control_points=[[0.0, 20.0, 20.0], [20.0, 20.0, 60.0]],
    )
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["datasets"][0]["fiber_paths"].append(str(bad_fiber))
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    loader = _make_loader(load_config(config_path))

    assert loader.sample_count == 3
    assert len(loader.records) == 1
    assert Path(loader.records[0].fiber.path).name == "fiber.json"


def test_loader_does_not_sample_normals_for_remote_line_endpoints(tmp_path: Path) -> None:
    points = [[0.0, 20.0, -10.0]]
    points.extend([[float(x), 20.0, 20.0] for x in range(5, 50, 5)])
    fiber = _write_fiber(
        tmp_path / "fiber.json",
        points=points,
        control_points=[[30.0, 20.0, 20.0]],
    )
    volume = _write_zarr(tmp_path / "vol.zarr")
    manifest = _write_lasagna_manifest(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "batch_size": 1,
                "patch_shape_hw": [3, 3],
                "strip_z_offset_count": 1,
                "strip_z_offset_step": 1.0,
                "seed": 123,
                "datasets": [
                    {
                        "fiber_paths": [str(fiber)],
                        "base_volume_path": str(volume),
                        "base_volume_scale": 0,
                        "lasagna_manifest_path": str(manifest),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    loader = _make_loader(load_config(config_path))

    batch = loader.load_batch(0)

    assert batch.images.shape == (1, 1, 1, 3, 3)
    assert batch.control_point_indices.tolist() == [0]


def test_loader_samples_only_cp_local_lasagna_normals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    points = [[float(x), 20.0, 20.0] for x in range(0, 90, 10)]
    fiber = _write_fiber(
        tmp_path / "fiber.json",
        points=points,
        control_points=[[40.0, 20.0, 20.0]],
    )
    volume = _write_zarr(tmp_path / "vol.zarr")
    manifest = _write_lasagna_manifest(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "batch_size": 1,
                "patch_shape_hw": [3, 3],
                "strip_z_offset_count": 1,
                "strip_z_offset_step": 1.0,
                "seed": 123,
                "datasets": [
                    {
                        "fiber_paths": [str(fiber)],
                        "base_volume_path": str(volume),
                        "base_volume_scale": 0,
                        "lasagna_manifest_path": str(manifest),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    loader = _make_loader(load_config(config_path))
    sampled_indices: list[int | None] = []

    def fake_normal(record, point_zyx_base, *, line_point_index=None, control_point_index=None):
        sampled_indices.append(line_point_index)
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(loader, "_lasagna_normal_at_zyx", fake_normal)

    loader.build_center_strip_patch(0)

    assert sampled_indices == [1, 2, 3, 4, 5, 6, 7]
    assert len(sampled_indices) < len(points)


def test_augmentation_config_defaults_and_overrides(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["augment_enabled"] = True
    raw["augment_seed"] = 999
    raw["augment_shift_x"] = 2.5
    raw["augment_brightness"] = 0.25
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    config = load_config(config_path)

    assert config.augment.enabled is True
    assert config.augment.seed == 999
    assert config.augment.device == "cpu"
    assert config.augment.shift_x == 2.5
    assert config.augment.shift_y == 0.75
    assert config.augment.brightness == 0.25
    assert config.augment.rotation_degrees == 180.0
    assert config.augment.shear_x == 1.0
    assert config.augment.shear_y == 1.0
    assert config.augment.scale_min == math.sqrt(0.5)
    assert config.augment.scale_max == math.sqrt(2.0)
    assert config.augment.smooth_offset == 8.0
    assert config.augment.smooth_offset_stride == 16.0
    assert config.augment.contrast_min == 0.5
    assert config.augment.contrast_max == 2.0
    assert config.augment.gamma_min == 0.5
    assert config.augment.gamma_max == 2.0
    assert config.augment.noise_std == 0.125
    assert config.augment.blur_sigma == 2.0


def test_combined_augmentation_is_deterministic() -> None:
    config = FiberStripAugmentConfig(enabled=True, seed=42, device="cpu")

    first = random_combined_augmentation(config, sample_index=7, variant_index=3)
    second = random_combined_augmentation(config, sample_index=7, variant_index=3)
    other = random_combined_augmentation(config, sample_index=7, variant_index=4)

    assert first == second
    assert first != other


def test_coordinate_scale_and_smooth_offset_are_deterministic() -> None:
    device = torch.device("cpu")
    base = source_coordinate_grid_for_output(
        5,
        5,
        9,
        9,
        FiberStripAugmentParams(),
        device=device,
    )
    scaled = source_coordinate_grid_for_output(
        5,
        5,
        9,
        9,
        FiberStripAugmentParams(scale=2.0),
        device=device,
    )
    smooth_a = smooth_offset_field(5, 5, amplitude=3.0, stride=2.0, seed=17, device=device)
    smooth_b = smooth_offset_field(5, 5, amplitude=3.0, stride=2.0, seed=17, device=device)

    assert torch.allclose(smooth_a, smooth_b)
    assert float(torch.abs(smooth_a).max().item()) <= 4.5
    assert torch.allclose(base[2, 2], scaled[2, 2])
    assert abs(float(scaled[2, 3, 0] - scaled[2, 2, 0])) < abs(float(base[2, 3, 0] - base[2, 2, 0]))


def test_shift_is_applied_after_scale_in_output_space() -> None:
    device = torch.device("cpu")
    params = FiberStripAugmentParams(shift_x=1.0, shift_y=-2.0, scale=2.0)

    point = transformed_source_point_coords(
        (9, 9),
        (9, 9),
        params,
        (5.0, 4.0),
        device=device,
    )
    grid = source_coordinate_grid_for_output(9, 9, 9, 9, params, device=device)

    assert np.allclose(point, np.asarray([7.0, 2.0], dtype=np.float32))
    assert torch.allclose(grid[int(round(point[1])), int(round(point[0]))], torch.tensor([5.0, 4.0]))


def test_line_augmentation_returns_coordinates_not_mask() -> None:
    line = transformed_centerline_coords(
        (5, 5),
        (9, 9),
        FiberStripAugmentParams(shift_x=2.0),
        device=torch.device("cpu"),
    )

    assert line.ndim == 2
    assert line.shape[1] == 2
    assert line.dtype == np.float32


def test_line_augmentation_shift_is_vectorized_direct_mapping() -> None:
    line = transformed_centerline_coords(
        (5, 5),
        (5, 5),
        FiberStripAugmentParams(shift_x=1.0),
        device=torch.device("cpu"),
    )

    assert np.allclose(line[:, 0], np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert np.allclose(line[:, 1], np.full((4,), 2.0, dtype=np.float32))


def test_value_augmentation_noise_and_blur_are_torch_based() -> None:
    image = np.zeros((9, 9), dtype=np.float32)
    image[4, 4] = 255.0
    valid = np.ones((9, 9), dtype=bool)

    blurred, blurred_valid = apply_value_augmentation(
        image,
        valid,
        FiberStripAugmentParams(blur_sigma=1.0),
        device=torch.device("cpu"),
    )
    noisy, _ = apply_value_augmentation(
        image,
        valid,
        FiberStripAugmentParams(noise_std=0.25, noise_seed=1),
        device=torch.device("cpu"),
    )

    assert blurred.device.type == "cpu"
    assert bool(blurred_valid.all().item())
    assert 0.0 < float(blurred[4, 4].item()) < 255.0
    assert float(blurred[4, 3].item()) > 0.0
    assert not torch.allclose(noisy, torch.as_tensor(image))


def test_gamma_augmentation_is_range_based() -> None:
    image = np.linspace(10.0, 110.0, 9, dtype=np.float32).reshape(3, 3)
    valid = np.ones((3, 3), dtype=bool)

    gamma, gamma_valid = apply_value_augmentation(
        image,
        valid,
        FiberStripAugmentParams(gamma=2.0),
        device=torch.device("cpu"),
    )

    assert bool(gamma_valid.all().item())
    assert torch.isclose(gamma[0, 0], torch.tensor(10.0))
    assert torch.isclose(gamma[-1, -1], torch.tensor(110.0))
    assert not torch.allclose(gamma, torch.as_tensor(image))


def test_augmented_loader_preserves_shape_and_changes_values(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["augment_enabled"] = True
    raw["augment_brightness"] = 1.0
    raw["augment_contrast"] = 0.0
    raw["augment_noise_std"] = 0.0
    raw["augment_blur_sigma"] = 0.0
    raw["augment_shift_x"] = 0.0
    raw["augment_shift_y"] = 0.0
    raw["augment_rotation_degrees"] = 0.0
    raw["augment_shear_x"] = 0.0
    raw["augment_shear_y"] = 0.0
    raw["augment_scale_min"] = 1.0
    raw["augment_scale_max"] = 1.0
    raw["augment_smooth_offset"] = 0.0
    raw["augment_gamma_min"] = 1.0
    raw["augment_gamma_max"] = 1.0
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base_config = load_config(_write_config(base_dir, batch_size=1))
    aug_config = load_config(config_path)
    base = _make_loader(base_config).load_batch(0)
    augmented = _make_loader(aug_config).load_batch(0)

    assert augmented.images.shape == base.images.shape
    assert augmented.valid_mask.shape == base.valid_mask.shape
    assert not np.allclose(augmented.images, base.images)


def test_augment_center_patch_loads_one_zarr_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = load_config(_write_config(tmp_path, batch_size=1))
    loader = _make_loader(config)
    calls = 0
    original = loader.records[0].sampler.sample_coords

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(loader.records[0].sampler, "sample_coords", counted)

    sample, image, valid = loader.build_center_strip_patch(0)

    assert calls == 1
    assert image.shape == (3, 3)
    assert valid.shape == (3, 3)
    assert sample.strip_z_offset == 0.0


def test_augmented_center_patch_loads_one_zarr_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["augment_enabled"] = True
    raw["augment_device"] = "cpu"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))
    params = random_combined_augmentation(loader.config.augment, sample_index=0, variant_index=0)
    calls = 0
    original = loader.records[0].sampler.sample_coords

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(loader.records[0].sampler, "sample_coords", counted)

    sample, image, valid, line = loader.build_augmented_center_strip_patch(
        0,
        params,
        device=torch.device("cpu"),
    )

    assert calls == 1
    assert image.shape == (3, 3)
    assert valid.shape == (3, 3)
    assert line.ndim == 2
    assert line.shape[1] == 2
    assert sample.strip_z_offset == 0.0


def test_identity_equivalent_augment_vis_entries_match(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["augment_enabled"] = True
    raw["augment_device"] = "cpu"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))
    lower_entries, _ = limit_augmentation_rows(loader.config.augment, sample_index=0)
    params_by_name = dict(lower_entries)

    outputs = {}
    for name in ("unaugmented", "noise_min", "blur_min"):
        _, image, valid, line = loader.build_augmented_center_strip_patch(
            0,
            params_by_name[name],
            device=torch.device("cpu"),
        )
        outputs[name] = (image, valid, line)

    base_image, base_valid, base_line = outputs["unaugmented"]
    for name in ("noise_min", "blur_min"):
        image, valid, line = outputs[name]
        assert np.allclose(image, base_image), name
        assert np.array_equal(valid, base_valid), name
        assert np.allclose(line, base_line), name


def test_augment_contact_sheet_export_writes_jpg(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["augment_enabled"] = True
    raw["augment_device"] = "cpu"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))
    out = tmp_path / "aug_out"

    _export_augment_contact_sheet(loader, 0, out)

    assert (out / "augment_contact_sheet.jpg").is_file()
    assert (out / "augment_summary.txt").is_file()
    from PIL import Image

    with Image.open(out / "augment_contact_sheet.jpg") as image:
        assert image.size[0] == 3 * 15
        assert image.size[1] > 3 * 3
        arr = np.asarray(image)
        assert arr[:2, :8].max() > 0
        cyan_dominant = (arr[:, :, 1] > arr[:, :, 0] + 40) & (arr[:, :, 2] > arr[:, :, 0] + 40)
        assert np.count_nonzero(cyan_dominant) > 0
    summary = (out / "augment_summary.txt").read_text(encoding="utf-8")
    assert "layout=row 1: lower limits; row 2: upper limits; row 3: random combined" in summary
    assert "cp_xy=" in summary
    assert "rotate_min" in summary
    assert "rotate_max" in summary
    assert "blur_max" in summary
    assert "scale_max" in summary
    assert "smooth_max" in summary
    assert "gamma_max" in summary
    assert "combined_00" in summary


def test_deterministic_sample_index_independent_of_batch_size(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    config_a = load_config(config_path)
    config_b = load_config(config_path)
    loader_a = _make_loader(config_a)
    loader_b = _make_loader(config_b)

    a = [loader_a.descriptor_for_sample_index(i)[2] for i in range(8)]
    b = [loader_b.descriptor_for_sample_index(i)[2] for i in range(8)]
    assert a == b


def test_prefetch_uses_dependency_chunks_and_cache_markers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = load_config(_write_config(tmp_path, batch_size=1))
    loader = _make_loader(config)

    class Store:
        def __init__(self, root: Path) -> None:
            self.root = root
            self.calls: list[str] = []

    store = Store(tmp_path / "cache")
    store.root.mkdir(parents=True)
    (store.root / "hit.bin").write_bytes(b"cached")
    (store.root / "known_missing.empty").write_bytes(b"")

    def downloader(request: ZarrChunkRequest) -> bytes | None:
        store.calls.append(request.key)
        if request.key == "new_missing":
            return None
        return b"\x01\x02"

    def request(key: str) -> ZarrChunkRequest:
        return ZarrChunkRequest(
            store=store,
            store_identity="test",
            key=key,
            cache_path=store.root / f"{key}.bin",
            empty_path=store.root / f"{key}.empty",
            remote_url=f"https://example.invalid/{key}",
            cache_payload_format="source_bytes",
            downloader=downloader,
        )

    def fake_chunk_requests(coords, valid):
        del coords, valid
        return [
            request("hit"),
            request("known_missing"),
            request("download"),
            request("new_missing"),
            request("download"),
        ]

    def forbidden_prefetch(coords, valid):
        del coords, valid
        raise AssertionError("loader prefetch must not call sampler.prefetch_coords")

    monkeypatch.setattr(loader.records[0].sampler, "chunk_requests_for_coords", fake_chunk_requests)
    monkeypatch.setattr(loader.records[0].sampler, "prefetch_coords", forbidden_prefetch)

    summary = loader.prefetch(0, 1, workers=8)

    assert sorted(store.calls) == ["download", "new_missing"]
    assert summary["patches"] == 3
    assert summary["generated"] == 4
    assert summary["cache_hits"] == 1
    assert summary["known_missing"] == 1
    assert summary["newly_missing"] == 1
    assert summary["missing"] == 2
    assert summary["downloaded"] == 1
    assert summary["queued_for_download"] == 2
    assert summary["download_done"] == 2
    assert summary["bytes"] == 2
    assert summary["errors"] == 0
    assert (store.root / "new_missing.empty").is_file()
    assert (store.root / "new_missing.empty").read_bytes() == b""
    assert (store.root / "download.bin").read_bytes() == b"\x01\x02"
    output = capsys.readouterr().out
    assert "samples[" in output
    assert "downloads[" in output


def test_prefetch_skips_invalid_sample_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    loader = _make_loader(load_config(_write_config(tmp_path, batch_size=2)))
    original_build_strip_source = loader.build_strip_source

    def build_strip_source(sample_index: int, **kwargs):
        if int(sample_index) == 0:
            raise ValueError("Lasagna grad_mag sample is zero at fiber line point")
        return original_build_strip_source(sample_index, **kwargs)

    monkeypatch.setattr(loader, "build_strip_source", build_strip_source)
    monkeypatch.setattr(loader.records[0].sampler, "chunk_requests_for_coords", lambda coords, valid: [])

    summary = loader.prefetch(0, 2, workers=2)

    assert summary["samples"] == 2
    assert summary["skipped_samples"] == 1
    assert summary["patches"] == 3
    assert "grad_mag sample is zero" in summary["first_sample_skip"]
    output = capsys.readouterr().out
    assert "skipped=1" in output
    assert "first_skip" not in output


def test_load_batch_skips_invalid_training_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loader = _make_loader(load_config(_write_config(tmp_path, batch_size=2)))
    original_build_sample = loader.build_sample
    attempted: list[int] = []

    def build_sample(sample_index: int):
        attempted.append(int(sample_index))
        if int(sample_index) == 0:
            raise ValueError("Lasagna grad_mag sample is zero at fiber line point")
        return original_build_sample(sample_index)

    monkeypatch.setattr(loader, "build_sample", build_sample)

    batch = loader.load_batch(0, batch_size=2)

    assert attempted[:3] == [0, 1, 2]
    assert batch.images.shape[0] == 2
    assert batch.coords_zyx.shape[0] == 2
    assert loader._load_batch_skipped_samples == 1


class _RecordingCoordinateSampler(NumpyZarrCoordinateSampler):
    def __init__(self, array, *, level_spacing_base: float) -> None:
        super().__init__(array, level_spacing_base=level_spacing_base)
        self.sample_coords_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.chunk_request_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.prefetch_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def sample_coords(self, coords_zyx_base: np.ndarray, valid_mask: np.ndarray):
        self.sample_coords_calls.append((np.array(coords_zyx_base, copy=True), np.array(valid_mask, copy=True)))
        return super().sample_coords(coords_zyx_base, valid_mask)

    def chunk_requests_for_coords(self, coords_zyx_base: np.ndarray, valid_mask: np.ndarray):
        self.chunk_request_calls.append((np.array(coords_zyx_base, copy=True), np.array(valid_mask, copy=True)))
        coords = np.asarray(coords_zyx_base, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(coords).all(axis=-1)
        if not bool(valid.any()):
            return []
        chunk_idx = np.floor(coords[valid] / 16.0).astype(np.int64)
        unique = np.unique(chunk_idx, axis=0)
        return [
            ZarrChunkRequest(store={}, store_identity="recording", key=f"{z}.{y}.{x}")
            for z, y, x in unique.tolist()
        ]

    def prefetch_coords(self, coords_zyx_base: np.ndarray, valid_mask: np.ndarray):
        self.prefetch_calls.append((np.array(coords_zyx_base, copy=True), np.array(valid_mask, copy=True)))
        return super().prefetch_coords(coords_zyx_base, valid_mask)


def _request_keys(requests: list[ZarrChunkRequest]) -> set[tuple[str, str]]:
    return {(request.store_identity, request.key) for request in requests}


def test_prefetch_envelope_covers_augmented_loading_dependencies(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    fiber = _write_fiber(
        tmp_path / "long_fiber.json",
        points=[[float(x), 24.0, 24.0] for x in range(8, 41, 4)],
        control_points=[[24.0, 24.0, 24.0]],
    )
    raw["datasets"][0]["fiber_paths"] = [str(fiber)]
    raw["patch_shape_hw"] = [5, 5]
    raw["strip_z_offset_count"] = 2
    raw["augment_enabled"] = True
    raw["augment_device"] = "cpu"
    raw["augment_shift_x"] = 1.0
    raw["augment_shift_y"] = 1.0
    raw["augment_rotation_degrees"] = 12.0
    raw["augment_shear_x"] = 0.2
    raw["augment_shear_y"] = 0.15
    raw["augment_scale_min"] = 0.95
    raw["augment_scale_max"] = 1.05
    raw["augment_smooth_offset"] = 0.0
    raw["augment_brightness"] = 0.0
    raw["augment_contrast_min"] = 1.0
    raw["augment_contrast_max"] = 1.0
    raw["augment_gamma_min"] = 1.0
    raw["augment_gamma_max"] = 1.0
    raw["augment_noise_std"] = 0.0
    raw["augment_blur_sigma"] = 0.0
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    samplers: list[_RecordingCoordinateSampler] = []

    def factory(**kwargs):
        sampler = _RecordingCoordinateSampler(
            kwargs["array"],
            level_spacing_base=kwargs["level_spacing_base"],
        )
        samplers.append(sampler)
        return sampler

    loader = FiberStrip2DLoader(load_config(config_path), sampler_factory=factory)
    sample_index = 4

    prefetch_requests = loader.chunk_requests_for_sample_index(sample_index)
    loader.build_sample(sample_index)

    sampler = samplers[0]
    assert len(sampler.chunk_request_calls) == 2
    assert len(sampler.sample_coords_calls) == 2
    prefetch_keys = _request_keys(prefetch_requests)
    assert prefetch_keys
    for load_coords, load_valid in sampler.sample_coords_calls:
        load_keys = _request_keys(sampler.chunk_requests_for_coords(load_coords, load_valid))
        assert load_keys <= prefetch_keys


def test_training_load_does_not_call_numpy_strip_builder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import vesuvius.neural_tracing.fiber_trace_2d.loader as loader_module
    import vesuvius.neural_tracing.fiber_trace_2d.strip_geometry as strip_geometry_module

    config = load_config(_write_config(tmp_path, batch_size=1))
    loader = _make_loader(config)

    def forbidden(*args, **kwargs):
        raise AssertionError("training loader must use shared torch source path")

    assert not hasattr(loader_module, "build_side_strip_patch_grid_from_line_window")
    monkeypatch.setattr(strip_geometry_module, "build_side_strip_patch_grid_from_line_window", forbidden)

    batch = loader.load_batch(0)

    assert batch.images.shape == (1, 3, 1, 3, 3)


def test_training_prefetch_maps_steps_to_sample_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    run_path = tmp_path / "runs"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["training"] = {
        "run_path": str(run_path),
        "run_name": "prefetch",
        "max_steps": 7,
        "control_points_per_step": 3,
        "tensorboard_enabled": False,
    }
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    calls: list[tuple[int, int]] = []

    def fake_prefetch(self, start_sample_index, sample_count, *, workers=None):
        del self, workers
        calls.append((int(start_sample_index), int(sample_count)))
        return {
            "generated": 11,
            "missing": 5,
            "downloaded": 5,
            "bytes": 123,
            "errors": 0,
            "workers": 1,
        }

    monkeypatch.setattr(FiberStrip2DLoader, "prefetch", fake_prefetch)

    summary = prefetch_training(config_path, prefetch_steps=3, sampler_factory=_test_sampler_factory)

    assert calls == [(0, 9)]
    assert summary["generated"] == 11
    assert not run_path.exists()

    calls.clear()
    prefetch_training(
        config_path,
        prefetch_steps=0,
        prefetch_start_step=2,
        sampler_factory=_test_sampler_factory,
    )

    assert calls == [(3, 21)]
    assert not run_path.exists()


def test_training_prefetch_rejects_negative_steps(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)

    with pytest.raises(ValueError, match="--prefetch-steps must be >= 0"):
        prefetch_training(config_path, prefetch_steps=-1, sampler_factory=_test_sampler_factory)


def test_training_config_parses_test_settings() -> None:
    config = _training_config_from_raw(
        {
            "training": {
                "test_interval": 17,
                "test_control_points": 3,
                "test_start_sample_index": 11,
            }
        }
    )

    assert config.test_interval == 17
    assert config.test_control_points == 3
    assert config.test_start_sample_index == 11


def test_test_datasets_replace_training_datasets_for_loader_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    train_fiber = raw["datasets"][0]["fiber_paths"][0]
    test_fiber = _write_fiber(tmp_path / "test_fiber.json", points=[[5.0, 22.0, 22.0], [15.0, 22.0, 22.0]])
    raw["test_datasets"] = [
        {
            **raw["datasets"][0],
            "fiber_paths": [str(test_fiber)],
        }
    ]
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    loader_config = load_config(config_path)
    test_config = _test_loader_config_from_raw(raw, loader_config)

    assert test_config is not None
    assert loader_config.datasets[0]["fiber_paths"] == [train_fiber]
    assert test_config.datasets[0]["fiber_paths"] == [str(test_fiber)]


def test_lasagna_direction_encoding_is_forward_backward_ambiguous() -> None:
    directions = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [math.sqrt(0.5), math.sqrt(0.5)],
        ],
        dtype=torch.float32,
    )

    encoded = encode_lasagna_direction_xy(directions)

    assert torch.allclose(encoded[0], encoded[1])
    assert torch.allclose(encoded[0], torch.tensor([1.0, 0.5 + 0.5 / math.sqrt(2.0)]))
    assert torch.allclose(encoded[2], torch.tensor([0.0, 0.5 - 0.5 / math.sqrt(2.0)]), atol=1.0e-6)
    assert torch.allclose(encoded[3], torch.tensor([0.5, 0.5 - 0.5 / math.sqrt(2.0)]), atol=1.0e-6)


def test_cp_local_supervision_uses_eight_samples_and_augmented_line() -> None:
    valid = np.ones((1, 5, 5), dtype=bool)
    sample = SimpleNamespace(
        line_xy=np.asarray(
            [
                [0.0, 1.0],
                [1.0, 1.5],
                [2.0, 2.0],
                [3.0, 2.5],
                [4.0, 3.0],
            ],
            dtype=np.float32,
        )
    )

    supervision = build_direction_supervision([sample], valid, device=torch.device("cpu"))

    assert supervision.target.shape == (8, 2)
    assert sorted(zip(supervision.y.tolist(), supervision.x.tolist())) == sorted(
        tuple(item) for item in cp_neighborhood_yx(np.asarray([2.0, 2.0], dtype=np.float32), (5, 5)).tolist()
    )
    expected = encode_lasagna_direction_xy(torch.tensor([1.0, 0.5]))
    assert torch.allclose(supervision.target[0], expected, atol=1.0e-6)


def test_training_batch_assembly_default_patch_count_is_64(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["patch_shape_hw"] = [5, 5]
    raw["strip_z_offset_count"] = 16
    raw["training"] = {"control_points_per_step": 4}
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))

    batch = loader.load_batch(0, batch_size=4)

    assert batch.images.shape == (4, 16, 1, 5, 5)
    assert len(batch.samples) == 64


def test_training_prints_first_100_samples_then_interval() -> None:
    assert _should_print_training_step(
        1,
        scalar_log_interval=50,
        start_sample_index=0,
        sample_count=4,
    )
    assert _should_print_training_step(
        25,
        scalar_log_interval=50,
        start_sample_index=96,
        sample_count=4,
    )
    assert not _should_print_training_step(
        26,
        scalar_log_interval=50,
        start_sample_index=100,
        sample_count=4,
    )
    assert _should_print_training_step(
        50,
        scalar_log_interval=50,
        start_sample_index=196,
        sample_count=4,
    )


def test_training_visualization_draws_only_predicted_cp_direction() -> None:
    rgb = np.zeros((17, 17, 3), dtype=np.uint8)
    out = _draw_predicted_cp_direction(
        rgb,
        cp_xy=np.asarray([8.0, 8.0], dtype=np.float32),
        prediction_xy=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    green = (out[:, :, 1] > out[:, :, 0] + 40) & (out[:, :, 1] > out[:, :, 2] + 40)
    yellow = (out[:, :, 0] > 180) & (out[:, :, 1] > 180) & (out[:, :, 2] < 80)
    cyan = (out[:, :, 1] > 180) & (out[:, :, 2] > 180) & (out[:, :, 0] < 80)

    assert np.count_nonzero(green) > 0
    assert np.count_nonzero(yellow) == 0
    assert np.count_nonzero(cyan) == 0


def test_one_step_training_smoke_writes_checkpoint(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["patch_shape_hw"] = [5, 5]
    raw["training"] = {
        "run_path": str(tmp_path / "runs"),
        "run_name": "smoke",
        "max_steps": 1,
        "control_points_per_step": 1,
        "tensorboard_enabled": False,
        "model_hidden_channels": 4,
        "model_depth": 2,
    }
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    run_dir = run_training(config_path, sampler_factory=_test_sampler_factory)

    assert (run_dir / "snapshots" / "current.pt").is_file()
    assert (run_dir / "snapshots" / "best.pt").is_file()


def test_training_with_test_dataset_uses_test_interval_for_snapshots(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    test_fiber = _write_fiber(tmp_path / "heldout_fiber.json", points=[[8.0, 24.0, 24.0], [18.0, 24.0, 24.0]])
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["patch_shape_hw"] = [5, 5]
    raw["test_datasets"] = [
        {
            **raw["datasets"][0],
            "fiber_paths": [str(test_fiber)],
        }
    ]
    raw["training"] = {
        "run_path": str(tmp_path / "runs"),
        "run_name": "test_interval",
        "max_steps": 2,
        "control_points_per_step": 1,
        "test_interval": 2,
        "test_control_points": 1,
        "test_start_sample_index": 0,
        "tensorboard_enabled": False,
        "model_hidden_channels": 4,
        "model_depth": 2,
    }
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    run_dir = run_training(config_path, sampler_factory=_test_sampler_factory)

    current = torch.load(run_dir / "snapshots" / "current.pt", map_location="cpu", weights_only=False)
    best = torch.load(run_dir / "snapshots" / "best.pt", map_location="cpu", weights_only=False)
    assert current["step"] == 2
    assert current["metric_name"] == "test/loss_direction"
    assert best["metric_name"] == "test/loss_direction"
