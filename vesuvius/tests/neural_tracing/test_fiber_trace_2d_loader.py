from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from vesuvius.neural_tracing.fiber_trace_2d.augmentation import (
    FiberStripAugmentConfig,
    FiberStripAugmentParams,
    apply_value_augmentation,
    apply_strip_augmentation,
    random_combined_augmentation,
    smooth_offset_field,
    source_coordinate_grid_for_output,
)
from vesuvius.neural_tracing.fiber_trace_2d import loader as loader_module
from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import load_vc3d_fiber
from vesuvius.neural_tracing.fiber_trace_2d.loader import (
    FiberStrip2DLoader,
    ZarrChunkRequest,
    load_config,
    strip_z_offsets_from_count_step,
)
from vesuvius.neural_tracing.fiber_trace_2d.runner import _export_augment_contact_sheet
from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import (
    build_planar_side_strip_patch_grid,
    build_side_strip_patch_grid,
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


def test_strip_and_planar_pixel_spacing_is_in_base_voxels(tmp_path: Path) -> None:
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
    planar = build_planar_side_strip_patch_grid(
        fiber,
        control_point_index=1,
        patch_shape_hw=(3, 3),
        strip_z_offset=0.0,
        sampled_normal=normal,
        pixel_spacing_base=2.0,
    )

    assert np.allclose(strip.coords_xyz[1, :, 0], [8.0, 10.0, 12.0])
    assert np.allclose(strip.coords_xyz[:, 1, 2], [18.0, 20.0, 22.0])
    assert np.allclose(planar.coords_xyz[1, :, 0], [8.0, 10.0, 12.0])
    assert np.allclose(planar.coords_xyz[:, 1, 2], [18.0, 20.0, 22.0])


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
    loader = FiberStrip2DLoader(config)
    batch = loader.load_batch(0)

    assert batch.images.shape == (2, 3, 1, 3, 3)
    assert batch.coords_zyx.shape == (2, 3, 3, 3, 3)
    assert batch.valid_mask.shape == (2, 3, 3, 3)
    assert batch.strip_z_offsets.tolist() == [-1.0, 0.0, 1.0]
    assert batch.control_point_indices.shape == (2,)


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


def test_torch_augmentation_marks_out_of_bounds_invalid() -> None:
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    valid = np.ones((5, 5), dtype=bool)
    params = FiberStripAugmentParams(shift_x=10.0)

    augmented, augmented_valid, line = apply_strip_augmentation(
        image,
        valid,
        params,
        device=torch.device("cpu"),
        return_line_mask=True,
    )

    assert augmented.device.type == "cpu"
    assert augmented_valid.sum().item() == 0
    assert float(augmented.abs().sum().item()) == 0.0
    assert float(line.sum().item()) == 0.0


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
    base = FiberStrip2DLoader(base_config).load_batch(0)
    augmented = FiberStrip2DLoader(aug_config).load_batch(0)

    assert augmented.images.shape == base.images.shape
    assert augmented.valid_mask.shape == base.valid_mask.shape
    assert not np.allclose(augmented.images, base.images)


def test_augment_center_patch_loads_one_zarr_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = load_config(_write_config(tmp_path, batch_size=1))
    loader = FiberStrip2DLoader(config)
    calls = 0
    original = loader_module._sample_array_trilinear

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(loader_module, "_sample_array_trilinear", counted)

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
    loader = FiberStrip2DLoader(load_config(config_path))
    params = random_combined_augmentation(loader.config.augment, sample_index=0, variant_index=0)
    calls = 0
    original = loader_module._sample_array_trilinear

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(loader_module, "_sample_array_trilinear", counted)

    sample, image, valid, line = loader.build_augmented_center_strip_patch(
        0,
        params,
        device=torch.device("cpu"),
    )

    assert calls == 1
    assert image.shape == (3, 3)
    assert valid.shape == (3, 3)
    assert line.shape == (3, 3)
    assert sample.strip_z_offset == 0.0


def test_augment_contact_sheet_export_writes_jpg(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["augment_enabled"] = True
    raw["augment_device"] = "cpu"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = FiberStrip2DLoader(load_config(config_path))
    out = tmp_path / "aug_out"

    _export_augment_contact_sheet(loader, 0, out)

    assert (out / "augment_contact_sheet.jpg").is_file()
    assert (out / "augment_summary.txt").is_file()
    from PIL import Image

    with Image.open(out / "augment_contact_sheet.jpg") as image:
        assert image.size == (3 * 15, 3 * 3)
        arr = np.asarray(image)
        assert arr[:2, :8].max() > 0
    summary = (out / "augment_summary.txt").read_text(encoding="utf-8")
    assert "layout=row 1: lower limits; row 2: upper limits; row 3: random combined" in summary
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
    loader_a = FiberStrip2DLoader(config_a)
    loader_b = FiberStrip2DLoader(config_b)

    a = [loader_a.descriptor_for_sample_index(i)[2] for i in range(8)]
    b = [loader_b.descriptor_for_sample_index(i)[2] for i in range(8)]
    assert a == b


class _FakeStore(dict):
    _url = "s3://bucket/vol.zarr"
    _cache_dir = "/tmp/fake-cache"
    _NEGATIVE_MARKER_SUFFIX = ".__notfound__"


def test_prefetch_deduplicates_and_skips_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, batch_size=1))
    loader = FiberStrip2DLoader(config)
    store = _FakeStore({"0.0.0": b"abc", "0.0.1": b"def"})
    seen = [
        ZarrChunkRequest(store=store, store_identity="same", key="0.0.0"),
        ZarrChunkRequest(store=store, store_identity="same", key="0.0.0"),
        ZarrChunkRequest(store=store, store_identity="same", key="0.0.1"),
    ]
    monkeypatch.setattr(loader, "chunk_requests_for_samples", lambda start, count: seen)
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(store, "_cache_dir", str(cache_dir), raising=False)
    (cache_dir / "0.0.0").parent.mkdir(parents=True, exist_ok=True)
    (cache_dir / "0.0.0").write_bytes(b"cached")

    summary = loader.prefetch(0, 1, workers=1)

    assert summary["generated"] == 2
    assert summary["missing"] == 1
    assert summary["downloaded"] == 1
    assert summary["errors"] == 0
