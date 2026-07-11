from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

import vesuvius.neural_tracing.fiber_trace_2d.augmentation as augment_module
import vesuvius.neural_tracing.fiber_trace_2d.loader as loader_module
from vesuvius.neural_tracing.fiber_trace_2d.augmentation import (
    FiberStripAugmentConfig,
    FiberStripAugmentParams,
    apply_value_augmentation,
    apply_value_augmentation_batch,
    limit_augmentation_rows,
    random_combined_augmentation,
    sample_xy_maps_bilinear,
    smooth_offset_field,
    source_coordinate_grid_for_output,
    strip_augment_transform,
    transformed_centerline_coords,
    transformed_source_point_coords,
)
from vesuvius.neural_tracing.fiber_trace_2d.direction import (
    build_direction_supervision,
    cp_neighborhood_yx,
    direction_angle_error_degrees,
    encode_lasagna_direction_xy,
)
from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import load_vc3d_fiber
from vesuvius.neural_tracing.fiber_trace_2d.loader import (
    FiberStrip2DLoader,
    ZarrChunkRequest,
    load_config,
    strip_z_offsets_from_count_step,
)
from vesuvius.neural_tracing.fiber_trace_2d.model import (
    FiberStripDirectionModelConfig,
    FiberStripDirectionNet,
)
from vesuvius.neural_tracing.fiber_trace_2d.train import (
    _benchmark_stage_totals,
    _draw_predicted_cp_direction,
    _print_profile_header,
    _select_visualization_patch_indices,
    _should_print_training_step,
    _TrainingBatchPipeline,
    _test_loader_config_from_raw,
    _training_config_from_raw,
    _validate_training_batch_config,
    FiberStripTrainingConfig,
    prefetch_training,
    run_benchmark,
    run_training,
)
from vesuvius.neural_tracing.fiber_trace_2d.runner import (
    _TtaDirectionField,
    _bilinear_direction_sample,
    _direction_field_overlay_rgb,
    _export_augment_contact_sheet,
    _line_trace_tta_entries,
    _nearest_tta_point_for_reference,
    _print_timing_table,
    _reference_direction_to_source_grid_direction,
    _score_trace2cp,
    _source_grid_direction_to_reference,
    _transform_points_xy,
    _trace2cp_tta_params,
    _trace_direction_line,
    _trace_direction_line_to_target,
    _trace_median_tta_direction_line_to_target,
    _trace_median_tta_direction_line,
)
from vesuvius.neural_tracing.fiber_trace_2d.sampling import NumpyZarrCoordinateSampler
from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import (
    build_side_strip_patch_grid,
    build_side_strip_patch_grid_from_line_window,
    build_side_strip_patch_grid_from_line_window_torch,
    build_side_strip_patch_grid_tensor_from_line_window,
    control_point_line_index,
    side_strip_segment_line_window,
    side_strip_line_window,
    source_point_xy_for_line_index,
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


def test_augment_vis_timing_table_reports_warm_path_average(capsys: pytest.CaptureFixture[str]) -> None:
    rows = [
        ("unaugmented", {"total": 100.0, "loader_total": 90.0, "volume_sample": 80.0}),
        ("shift_x_min", {"total": 10.0, "loader_total": 8.0, "volume_sample": 6.0}),
        ("shift_x_max", {"total": 14.0, "loader_total": 10.0, "volume_sample": 8.0}),
    ]
    totals = {"total": 124.0, "loader_total": 108.0, "volume_sample": 94.0}

    _print_timing_table(rows, totals)

    output = capsys.readouterr().out
    assert "total/no-first" in output
    assert "avg/no-first" in output
    assert "         24.0" in output
    assert "         12.0" in output


def test_line_trace_bilinear_direction_sample_normalizes() -> None:
    field = np.zeros((3, 3, 2), dtype=np.float32)
    field[:, :, 0] = 2.0

    sampled = _bilinear_direction_sample(field, np.asarray([1.25, 1.5], dtype=np.float32))

    assert sampled is not None
    assert np.allclose(sampled, [1.0, 0.0])


def test_line_trace_horizontal_stops_at_margin() -> None:
    field = np.zeros((9, 9, 2), dtype=np.float32)
    field[:, :, 0] = 1.0

    line = _trace_direction_line(
        field,
        np.asarray([4.0, 4.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        step_px=1.0,
        rf_margin_px=2.0,
    )

    assert np.allclose(line[:, 1], 4.0)
    assert np.isclose(line[0, 0], 2.0)
    assert np.isclose(line[-1, 0], 6.0)


def test_line_trace_keeps_ambiguous_direction_continuous() -> None:
    field = np.zeros((9, 9, 2), dtype=np.float32)
    field[:, :, 0] = 1.0
    field[:, :4, 0] = -1.0

    line = _trace_direction_line(
        field,
        np.asarray([4.0, 4.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        step_px=1.0,
        rf_margin_px=1.0,
    )

    assert np.all(np.diff(line[:, 0]) > 0.0)
    assert np.isclose(line[0, 0], 1.0)
    assert np.isclose(line[-1, 0], 7.0)


def test_trace2cp_scoring_interpolates_target_column() -> None:
    trace = np.asarray([[1.0, 4.0], [3.0, 6.0]], dtype=np.float32)
    result = _score_trace2cp(
        trace,
        np.asarray([2.0, 5.0], dtype=np.float32),
        shape_hw=(11, 11),
        rf_margin_px=1.0,
        termination_reason="target_column",
    )

    assert result.reached_target_column
    assert result.score == pytest.approx(0.0)
    assert result.raw_y_error_px == pytest.approx(0.0)
    assert result.trace_y_at_target_x == pytest.approx(5.0)


def test_trace2cp_scoring_failure_is_edge_score() -> None:
    trace = np.asarray([[1.0, 4.0], [1.5, 4.0]], dtype=np.float32)
    result = _score_trace2cp(
        trace,
        np.asarray([8.0, 5.0], dtype=np.float32),
        shape_hw=(11, 11),
        rf_margin_px=1.0,
        termination_reason="invalid_direction",
    )

    assert not result.reached_target_column
    assert result.score == pytest.approx(1.0)
    assert result.raw_y_error_px == pytest.approx(4.0)


def test_trace2cp_one_way_trace_stops_at_target_column() -> None:
    field = np.zeros((9, 9, 2), dtype=np.float32)
    field[:, :, 0] = 1.0

    line, reason = _trace_direction_line_to_target(
        field,
        np.asarray([2.0, 4.0], dtype=np.float32),
        np.asarray([6.0, 4.0], dtype=np.float32),
        step_px=1.0,
        rf_margin_px=1.0,
    )

    assert reason == "target_column"
    assert np.allclose(line[:, 1], 4.0)
    assert np.isclose(line[0, 0], 2.0)
    assert np.isclose(line[-1, 0], 6.0)


def test_trace2cp_target_column_wins_over_next_rf_margin() -> None:
    field = np.zeros((9, 9, 2), dtype=np.float32)
    field[:, :, 0] = 1.0

    line, reason = _trace_direction_line_to_target(
        field,
        np.asarray([4.0, 4.0], dtype=np.float32),
        np.asarray([6.0, 4.0], dtype=np.float32),
        step_px=2.0,
        rf_margin_px=3.0,
    )

    assert reason == "target_column"
    assert np.isclose(line[-1, 0], 6.0)


def test_trace2cp_tta_params_drop_y_shift_and_scale() -> None:
    params = FiberStripAugmentParams(
        shift_x=3.0,
        shift_y=4.0,
        rotation_degrees=17.0,
        shear_x=0.25,
        shear_y=-0.5,
        scale=1.8,
        smooth_offset=2.5,
        smooth_offset_stride=8.0,
        smooth_offset_seed=123,
        flip_x=True,
        flip_y=True,
        brightness=0.5,
        contrast=2.0,
        gamma=0.25,
        noise_std=0.75,
        blur_sigma=1.5,
    )

    filtered = _trace2cp_tta_params(params)

    assert filtered.shift_x == pytest.approx(3.0)
    assert filtered.shift_y == pytest.approx(0.0)
    assert filtered.scale == pytest.approx(1.0)
    assert filtered.rotation_degrees == pytest.approx(17.0)
    assert filtered.shear_x == pytest.approx(0.25)
    assert filtered.shear_y == pytest.approx(-0.5)
    assert filtered.smooth_offset == pytest.approx(2.5)
    assert filtered.flip_x is True
    assert filtered.flip_y is True
    assert filtered.brightness == pytest.approx(0.0)
    assert filtered.contrast == pytest.approx(1.0)
    assert filtered.gamma == pytest.approx(1.0)
    assert filtered.noise_std == pytest.approx(0.0)
    assert filtered.blur_sigma == pytest.approx(0.0)


def test_trace2cp_median_tta_trace_stops_at_target_column() -> None:
    field = np.zeros((9, 9, 2), dtype=np.float32)
    field[:, :, 0] = 1.0
    identity = np.eye(3, dtype=np.float32)
    fields = [
        _TtaDirectionField(
            name="reference",
            direction_xy=field,
            valid_mask=np.ones((9, 9), dtype=bool),
            matrix_ref_to_tta=identity,
            matrix_tta_to_ref=identity,
        )
    ]

    line, reason = _trace_median_tta_direction_line_to_target(
        fields,
        np.asarray([2.0, 4.0], dtype=np.float32),
        np.asarray([6.0, 4.0], dtype=np.float32),
        shape_hw=(9, 9),
        step_px=1.0,
        rf_margin_px=1.0,
    )

    assert reason == "target_column"
    assert np.allclose(line[:, 1], 4.0)
    assert np.isclose(line[-1, 0], 6.0)


def test_trace2cp_segment_window_maps_start_and_target_cp(tmp_path: Path) -> None:
    fiber = load_vc3d_fiber(
        _write_fiber(
            tmp_path / "fiber_trace_2d_trace2cp_window.json",
            points=[[0.0, 20.0, 20.0], [10.0, 20.0, 20.0], [20.0, 20.0, 20.0]],
            control_points=[[0.0, 20.0, 20.0], [10.0, 20.0, 20.0], [20.0, 20.0, 20.0]],
        )
    )
    window = side_strip_segment_line_window(
        fiber,
        start_control_point_index=0,
        target_control_point_index=1,
        margin_px=2.0,
        pixel_spacing_base=1.0,
    )
    start_line_index = control_point_line_index(fiber, 0)
    target_line_index = control_point_line_index(fiber, 1)
    start_xy = source_point_xy_for_line_index(
        window,
        original_line_index=start_line_index,
        patch_shape_hw=(9, 16),
        anchor_column_px=2.0,
        pixel_spacing_base=1.0,
    )
    target_xy = source_point_xy_for_line_index(
        window,
        original_line_index=target_line_index,
        patch_shape_hw=(9, 16),
        anchor_column_px=2.0,
        pixel_spacing_base=1.0,
    )

    assert start_xy.tolist() == pytest.approx([2.0, 4.0])
    assert target_xy.tolist() == pytest.approx([12.0, 4.0])


def test_trace2cp_loader_rejects_same_cp(tmp_path: Path) -> None:
    loader = _make_loader(load_config(_write_config(tmp_path, batch_size=1)))

    with pytest.raises(ValueError, match="must differ"):
        loader.build_trace2cp_segment_patch(
            0,
            target_control_point_index=0,
            rf_margin_px=0.0,
            sample_mode="flat",
        )


def test_trace2cp_segment_patch_uses_double_configured_height(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["patch_shape_hw"] = [5, 7]
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))

    sample, image, valid = loader.build_trace2cp_segment_patch(
        0,
        target_control_point_index=1,
        rf_margin_px=0.0,
        sample_mode="flat",
    )

    assert image.shape[0] == 10
    assert valid.shape[0] == 10
    assert sample.coords_zyx.shape[0] == 10


def test_line_trace_tta_point_transforms_round_trip() -> None:
    loader = SimpleNamespace(
        config=SimpleNamespace(
            augment=SimpleNamespace(
                shift_x=3.0,
                shift_y=2.0,
            )
        )
    )
    points = np.asarray([[1.0, 2.0], [4.5, 6.25], [8.0, 3.0]], dtype=np.float32)

    for _, matrix in _line_trace_tta_entries((11, 13), loader):
        warped = _transform_points_xy(points, matrix)
        restored = _transform_points_xy(warped, np.linalg.inv(matrix))
        assert np.allclose(restored, points, atol=1.0e-5)


def test_line_trace_random_tta_source_grid_maps_points_and_directions() -> None:
    source_xy = np.zeros((9, 9, 2), dtype=np.float32)
    yy, xx = np.indices((9, 9), dtype=np.float32)
    source_xy[..., 0] = xx - 1.0
    source_xy[..., 1] = yy

    tta_point = _nearest_tta_point_for_reference(source_xy, np.asarray([4.0, 4.0], dtype=np.float32))
    assert tta_point is not None
    assert np.allclose(tta_point, [5.0, 4.0])

    tta_direction = _reference_direction_to_source_grid_direction(
        source_xy,
        tta_point,
        np.asarray([1.0, 0.0], dtype=np.float32),
    )
    assert tta_direction is not None
    assert np.allclose(tta_direction, [1.0, 0.0])

    reference_direction = _source_grid_direction_to_reference(source_xy, tta_point, tta_direction)
    assert reference_direction is not None
    assert np.allclose(reference_direction, [1.0, 0.0])


def test_median_tta_trace_uses_reference_space_directions() -> None:
    identity = np.eye(3, dtype=np.float32)
    shift = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    shifted_inverse = np.linalg.inv(shift).astype(np.float32)
    ref = np.zeros((9, 9, 2), dtype=np.float32)
    ref[:, :, 0] = 1.0
    up = np.zeros_like(ref)
    up[:, :, 1] = 1.0
    shifted_opposite = np.zeros_like(ref)
    shifted_opposite[:, :, 0] = -1.0
    valid = np.ones((9, 9), dtype=bool)
    fields = [
        _TtaDirectionField("ref", ref, valid, identity, identity),
        _TtaDirectionField("up", up, valid, identity, identity),
        _TtaDirectionField("shifted_opposite", shifted_opposite, valid, shift, shifted_inverse),
    ]

    line = _trace_median_tta_direction_line(
        fields,
        np.asarray([4.0, 4.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        shape_hw=(9, 9),
        step_px=1.0,
        rf_margin_px=2.0,
    )

    assert np.allclose(line[:, 1], 4.0)
    assert np.isclose(line[0, 0], 2.0)
    assert np.isclose(line[-1, 0], 6.0)


def test_dir_vis_overlay_scales_and_strides() -> None:
    image = np.full((9, 9), 64, dtype=np.uint8)
    valid = np.zeros((9, 9), dtype=bool)
    valid[0::4, 0::4] = True
    valid[4, 4] = False
    direction = np.zeros((9, 9, 2), dtype=np.float32)
    direction[:, :, 0] = 1.0

    overlay, drawn = _direction_field_overlay_rgb(image, valid, direction, scale=2, stride=4)

    assert overlay.shape == (18, 18, 3)
    assert drawn == 8
    scaled = np.repeat(np.repeat(np.repeat(image[..., None], 3, axis=2), 2, axis=0), 2, axis=1)
    assert not np.array_equal(overlay, scaled)


def test_direction_model_defaults_to_10_block_64_channel_resnet() -> None:
    config = FiberStripDirectionModelConfig()
    model = FiberStripDirectionNet(config)

    assert config.hidden_channels == 64
    assert config.depth == 10
    assert len(model.blocks) == 10
    assert model.input[0].out_channels == 64
    assert model.input[1].num_groups == 8
    assert model.blocks[0].norm1.num_groups == 8


def test_direction_model_forward_shape_and_range() -> None:
    model = FiberStripDirectionNet(FiberStripDirectionModelConfig(hidden_channels=4, depth=2))
    image = torch.zeros((3, 1, 8, 9), dtype=torch.float32)

    output = model(image)

    assert model.input[1].num_groups == 4
    assert output.shape == (3, 2, 8, 9)
    assert bool(torch.all(output >= 0.0))
    assert bool(torch.all(output <= 1.0))


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
    tensor_grid = build_side_strip_patch_grid_tensor_from_line_window(
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
    assert isinstance(tensor_grid.coords_zyx, torch.Tensor)
    assert tensor_grid.coords_zyx.device.type == "cpu"
    assert tensor_grid.coords_zyx.dtype == torch.float32
    assert np.allclose(tensor_grid.to_numpy().coords_zyx, numpy_grid.coords_zyx, atol=1.0e-5)


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


def test_config_prefetch_workers_is_not_capped_at_16(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["prefetch_workers"] = 64
    config_path.write_text(json.dumps(config), encoding="utf-8")

    parsed = load_config(config_path)

    assert parsed.prefetch_workers == 64


def test_config_prefetch_sampler_workers_is_separate_from_download_workers(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["prefetch_workers"] = 64
    config["prefetch_sampler_workers"] = 3
    config_path.write_text(json.dumps(config), encoding="utf-8")

    parsed = load_config(config_path)

    assert parsed.prefetch_workers == 64
    assert parsed.prefetch_sampler_workers == 3


def test_config_loader_workers_defaults_to_logical_cpu_count(tmp_path: Path) -> None:
    parsed = load_config(_write_config(tmp_path, batch_size=1))

    assert parsed.loader_workers == max(1, loader_module.os.cpu_count() or 1)


def test_config_loader_workers_accepts_single_worker_debug_mode(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["loader_workers"] = 1
    config_path.write_text(json.dumps(config), encoding="utf-8")

    parsed = load_config(config_path)

    assert parsed.loader_workers == 1


def test_numpy_sampler_batch_loading_matches_repeated_patch_loading(tmp_path: Path) -> None:
    array = zarr.open(str(_write_zarr(tmp_path / "vol.zarr") / "0"), mode="r")
    sampler = NumpyZarrCoordinateSampler(array, level_spacing_base=1.0)
    coords = np.zeros((2, 3, 3, 3), dtype=np.float32)
    valid = np.ones((2, 3, 3), dtype=bool)
    coords[0, ..., 0] = 20.0
    coords[0, ..., 1] = np.arange(3, dtype=np.float32).reshape(3, 1) + 20.0
    coords[0, ..., 2] = np.arange(3, dtype=np.float32).reshape(1, 3) + 20.0
    coords[1] = coords[0] + np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    batch = sampler.sample_coord_batch(coords, valid)
    repeated = [sampler.sample_coords(coords[index], valid[index]) for index in range(2)]

    assert batch.image.shape == (2, 3, 3)
    assert batch.valid_mask.shape == (2, 3, 3)
    np.testing.assert_allclose(batch.image, np.stack([item.image for item in repeated], axis=0))
    np.testing.assert_array_equal(batch.valid_mask, np.stack([item.valid_mask for item in repeated], axis=0))
    assert batch.stats["batch_patches"] == 2
    assert batch.stats["batch_mode_flattened"] == 1


def test_config_parses_strip_coord_cache_dir(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["strip_coord_cache_dir"] = "coord-cache"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    parsed = load_config(config_path)

    assert parsed.strip_coord_cache_dir == str((tmp_path / "coord-cache").resolve())


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


def test_loader_batch_profile_collects_stage_timings(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["augment_enabled"] = True
    raw["augment_device"] = "cpu"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))
    profile: dict[str, float] = {}

    batch = loader.load_batch(0, profile=profile)

    assert batch.images.shape == (1, 3, 1, 3, 3)
    for key in (
        "descriptor",
        "line_window",
        "lasagna_normals",
        "strip_coords",
        "coord_augmentation",
        "line_coords",
        "volume_sample",
        "value_augmentation",
        "load_batch_wall",
        "load_batch_worker",
    ):
        assert key in profile
        assert profile[key] >= 0.0


def test_training_profile_splits_coord_cache_and_line(capsys: pytest.CaptureFixture[str]) -> None:
    stages = _benchmark_stage_totals(
        {
            "descriptor": 1.0,
            "strip_coord_cache": 2.0,
            "line_window": 3.0,
            "lasagna_normals": 4.0,
            "strip_coords": 5.0,
            "line_coords": 6.0,
            "coord_augmentation": 7.0,
            "volume_sample": 8.0,
            "load_batch_wall": 10.0,
            "load_batch_worker": 25.0,
        },
        {},
    )

    _print_profile_header()
    output = capsys.readouterr().out

    assert stages["descriptor"] == 1.0
    assert stages["coord_cache"] == 2.0
    assert stages["source_geom"] == 12.0
    assert stages["line"] == 6.0
    assert stages["coord_gen"] == 21.0
    assert stages["loader_wall"] == 10.0
    assert stages["loader_worker"] == 25.0
    assert stages["loader_thread_factor"] == 2.5
    assert "cache" in output
    assert "source" in output
    assert "line" in output
    assert "tf" in output


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


def test_strip_augment_transform_affine_round_trip() -> None:
    device = torch.device("cpu")
    params = FiberStripAugmentParams(
        shift_x=3.0,
        shift_y=-2.0,
        rotation_degrees=17.0,
        shear_x=0.15,
        shear_y=-0.08,
        scale=1.2,
        flip_x=True,
    )
    transform = strip_augment_transform((19, 23), (31, 37), params, device=device)
    source_points = torch.tensor(
        [[18.0, 15.0], [12.5, 9.25], [24.0, 17.5]],
        dtype=torch.float32,
        device=device,
    )

    output_points = transform.source_to_output_points(source_points)
    round_tripped = transform.output_to_source_points(output_points)

    assert torch.allclose(round_tripped, source_points, atol=1.0e-4)


def test_strip_augment_transform_smooth_round_trip_is_subpixel_with_cached_maps() -> None:
    device = torch.device("cpu")
    params = FiberStripAugmentParams(
        shift_x=2.0,
        shift_y=-1.0,
        rotation_degrees=8.0,
        shear_x=0.05,
        shear_y=-0.03,
        scale=1.1,
        smooth_offset=2.5,
        smooth_offset_stride=4.0,
        smooth_offset_seed=19,
    )
    transform = strip_augment_transform((21, 21), (31, 31), params, device=device)
    source_points = torch.tensor(
        [[15.0, 15.0], [12.0, 13.0], [18.0, 17.0]],
        dtype=torch.float32,
        device=device,
    )

    output_points = transform.source_to_output_points(source_points)
    round_tripped = transform.output_to_source_points(output_points)

    assert torch.allclose(round_tripped, source_points, atol=0.1)


def test_strip_augment_transform_caches_smooth_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = augment_module._smooth_offset_controls

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(augment_module, "_smooth_offset_controls", counted)
    transform = strip_augment_transform(
        (21, 21),
        (31, 31),
        FiberStripAugmentParams(smooth_offset=2.0, smooth_offset_stride=4.0, smooth_offset_seed=5),
        device=torch.device("cpu"),
    )
    source_points = torch.tensor([[15.0, 15.0], [12.0, 13.0]], dtype=torch.float32)

    transform.source_to_output_points(source_points)
    transform.source_to_output_points(source_points)
    transform.output_to_source_points(torch.tensor([[10.0, 10.0]], dtype=torch.float32))

    assert calls == 1


def test_strip_augment_transform_point_mapping_uses_cached_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    params = FiberStripAugmentParams(smooth_offset=2.0, smooth_offset_stride=4.0, smooth_offset_seed=5)
    transform = strip_augment_transform((21, 21), (31, 31), params, device=torch.device("cpu"))

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("point mapping must use cached map tensors")

    monkeypatch.setattr(augment_module, "_cubic_interp_1d", forbidden)

    source_points = torch.tensor([[15.0, 15.0], [12.0, 13.0]], dtype=torch.float32)
    output_points = torch.tensor([[10.0, 10.0], [11.0, 12.0]], dtype=torch.float32)

    assert torch.isfinite(transform.source_to_output_points(source_points)).all()
    assert torch.isfinite(transform.output_to_source_points(output_points)).all()


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


def test_smooth_line_and_cp_mapping_do_not_use_nearest_grid_search(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("smooth line mapping should not use dense nearest-grid search")

    monkeypatch.setattr(augment_module, "_nearest_output_pixels_for_source_points", forbidden)
    params = FiberStripAugmentParams(smooth_offset=2.0, smooth_offset_stride=3.0, smooth_offset_seed=7)

    line = transformed_centerline_coords(
        (9, 9),
        (13, 13),
        params,
        device=torch.device("cpu"),
    )
    point = transformed_source_point_coords(
        (9, 9),
        (13, 13),
        params,
        (6.0, 6.0),
        device=torch.device("cpu"),
    )

    assert line.shape[1] == 2
    assert np.isfinite(line).all()
    assert np.isfinite(point).all()


def test_line_augmentation_shift_is_vectorized_direct_mapping() -> None:
    line = transformed_centerline_coords(
        (5, 5),
        (5, 5),
        FiberStripAugmentParams(shift_x=1.0),
        device=torch.device("cpu"),
    )

    assert np.allclose(line[:, 0], np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert np.allclose(line[:, 1], np.full((4,), 2.0, dtype=np.float32))


def test_batched_xy_map_bilinear_matches_single_transform_lookup() -> None:
    device = torch.device("cpu")
    params = [
        FiberStripAugmentParams(shift_x=1.0, shift_y=-0.5, scale=1.1),
        FiberStripAugmentParams(rotation_degrees=8.0, smooth_offset=1.5, smooth_offset_seed=4),
    ]
    transforms = [strip_augment_transform((9, 9), (13, 13), param, device=device) for param in params]
    maps = torch.stack([transform.forward_map_xy for transform in transforms], dim=0)
    points = torch.tensor(
        [
            [[6.0, 6.0], [3.5, 7.25], [10.0, 9.0]],
            [[6.0, 6.0], [3.5, 7.25], [10.0, 9.0]],
        ],
        dtype=torch.float32,
    )

    batched, valid = sample_xy_maps_bilinear(maps, points)

    assert bool(valid.all().item())
    for index, transform in enumerate(transforms):
        assert torch.allclose(batched[index], transform.source_to_output_points(points[index]), atol=1.0e-6)


def test_batched_xy_map_bilinear_marks_oob_points() -> None:
    maps = torch.zeros((1, 5, 5, 2), dtype=torch.float32)
    points = torch.tensor([[[-1.0, 2.0], [2.0, 2.0], [5.0, 1.0]]], dtype=torch.float32)

    sampled, valid = sample_xy_maps_bilinear(maps, points)

    assert valid.tolist() == [[False, True, False]]
    assert not torch.isfinite(sampled[0, 0]).all()
    assert torch.isfinite(sampled[0, 1]).all()
    assert not torch.isfinite(sampled[0, 2]).all()


def test_batched_coordinate_augmentation_matches_single_patch_path() -> None:
    device = torch.device("cpu")
    yy, xx = torch.meshgrid(torch.arange(7, dtype=torch.float32), torch.arange(7, dtype=torch.float32), indexing="ij")
    base = torch.stack([yy, xx, yy + xx], dim=-1)
    coords = torch.stack([base, base + 10.0], dim=0)
    valid = torch.ones((2, 7, 7), dtype=torch.bool)
    params = [
        FiberStripAugmentParams(shift_x=1.0, shift_y=-0.5, scale=1.1),
        FiberStripAugmentParams(rotation_degrees=5.0, shear_x=0.1),
    ]
    transforms = [strip_augment_transform((5, 5), (7, 7), param, device=device) for param in params]
    maps = torch.stack([transform.backward_map_xy for transform in transforms], dim=0)

    batch_coords, batch_valid = loader_module._resample_coord_tensor_batch_like_augmentation(coords, valid, maps)
    singles = [
        loader_module._resample_coord_tensors_like_augmentation(
            coords[index],
            valid[index],
            params[index],
            output_shape_hw=(5, 5),
            device=device,
            transform=transforms[index],
        )
        for index in range(2)
    ]

    assert torch.allclose(batch_coords, torch.stack([item[0] for item in singles], dim=0), atol=1.0e-6)
    assert torch.equal(batch_valid, torch.stack([item[1] for item in singles], dim=0))


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


def test_batched_value_augmentation_matches_single_patch_path() -> None:
    images = np.stack(
        [
            np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4),
            np.linspace(1.0, 2.0, 16, dtype=np.float32).reshape(4, 4),
        ],
        axis=0,
    )
    valid = np.ones((2, 4, 4), dtype=bool)
    params = [
        FiberStripAugmentParams(brightness=0.1, contrast=1.2, gamma=1.0, noise_std=0.0, blur_sigma=0.0),
        FiberStripAugmentParams(brightness=-0.1, contrast=0.8, gamma=2.0, noise_std=0.0, blur_sigma=0.0),
    ]

    batched, batched_valid = apply_value_augmentation_batch(images, valid, params, device=torch.device("cpu"))
    singles = [
        apply_value_augmentation(images[index], valid[index], params[index], device=torch.device("cpu"))[0]
        for index in range(2)
    ]

    assert bool(batched_valid.all().item())
    assert torch.allclose(batched, torch.stack(singles, dim=0), atol=1.0e-6)


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


def test_strip_coord_cache_serves_fresh_loader_without_resampling_normals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["strip_coord_cache_dir"] = str(tmp_path / "coord-cache")
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_config(config_path)
    first_loader = _make_loader(config)

    cold_source = first_loader.build_strip_source(0, device=torch.device("cpu"))

    cache_files = sorted((tmp_path / "coord-cache").glob("*/*.npz"))
    assert len(cache_files) == 1

    warm_loader = _make_loader(config)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Lasagna normals should not be sampled on strip coordinate cache hit")

    monkeypatch.setattr(warm_loader, "_lasagna_normals_for_line_window", forbidden)

    warm_source = warm_loader.build_strip_source(0, device=torch.device("cpu"))

    assert np.allclose(
        _as_test_numpy(cold_source.grid.coords_zyx),
        _as_test_numpy(warm_source.grid.coords_zyx),
    )
    assert np.array_equal(
        _as_test_numpy(cold_source.grid.valid_mask),
        _as_test_numpy(warm_source.grid.valid_mask),
    )
    assert np.allclose(
        _as_test_numpy(cold_source.source_line_xy),
        _as_test_numpy(warm_source.source_line_xy),
    )
    assert np.allclose(
        _as_test_numpy(cold_source.source_control_point_xy),
        _as_test_numpy(warm_source.source_control_point_xy),
    )


def _as_test_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def test_strip_coord_cache_larger_source_satisfies_smaller_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["strip_coord_cache_dir"] = str(tmp_path / "coord-cache")
    raw["patch_shape_hw"] = [7, 7]
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    large_config = load_config(config_path)
    large_loader = _make_loader(large_config)
    large_source = large_loader.build_strip_source(0, device=torch.device("cpu"))
    assert large_source.grid.coords_zyx.shape[:2] == (7, 7)

    raw["patch_shape_hw"] = [3, 3]
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    small_config = load_config(config_path)
    small_loader = _make_loader(small_config)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("smaller strip source should be served from larger cached entry")

    monkeypatch.setattr(small_loader, "_lasagna_normals_for_line_window", forbidden)

    small_source = small_loader.build_strip_source(0, device=torch.device("cpu"))

    assert small_source.grid.coords_zyx.shape[:2] == (3, 3)
    assert np.allclose(
        _as_test_numpy(small_source.grid.coords_zyx),
        _as_test_numpy(large_source.grid.coords_zyx)[2:5, 2:5],
    )
    assert np.allclose(
        _as_test_numpy(small_source.source_line_xy),
        np.asarray([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float32),
    )
    assert np.allclose(
        _as_test_numpy(small_source.source_control_point_xy),
        np.asarray([1.0, 1.0], dtype=np.float32),
    )


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


def test_augmented_patch_reuses_one_transform_for_coords_and_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["augment_enabled"] = True
    raw["augment_device"] = "cpu"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))
    params = random_combined_augmentation(loader.config.augment, sample_index=0, variant_index=0)
    original_factory = loader_module.strip_augment_transform
    calls = 0

    def counted_factory(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_factory(*args, **kwargs)

    monkeypatch.setattr(loader_module, "strip_augment_transform", counted_factory)

    loader.build_augmented_center_strip_patch(
        0,
        params,
        device=torch.device("cpu"),
    )

    assert calls == 1


def test_loader_maps_augmented_line_and_cp_in_one_batched_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["augment_enabled"] = True
    raw["augment_device"] = "cpu"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))
    source = loader.build_strip_source(0, device=torch.device("cpu"), use_augmentation_envelope=True)
    params = random_combined_augmentation(loader.config.augment, sample_index=0, variant_index=0)
    original_factory = loader_module.strip_augment_transform
    call_shapes: list[tuple[int, ...]] = []

    class CountingTransform:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def source_to_output_points(self, points):
            call_shapes.append(tuple(points.shape))
            return self._wrapped.source_to_output_points(points)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def counted_factory(*args, **kwargs):
        return CountingTransform(original_factory(*args, **kwargs))

    monkeypatch.setattr(loader_module, "strip_augment_transform", counted_factory)

    line_xy, cp_xy = loader._line_and_cp_xy_for_params(
        source,
        params,
        device=torch.device("cpu"),
    )

    assert call_shapes == [(int(source.source_line_xy.shape[0]) + 1, 2)]
    assert line_xy.ndim == 2
    assert line_xy.shape[1] == 2
    assert cp_xy.shape == (2,)


def test_build_sample_uses_batched_line_lookup_for_augmented_offsets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["augment_enabled"] = True
    raw["augment_device"] = "cpu"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))
    original = loader_module.sample_xy_maps_bilinear
    call_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def counted(*args, **kwargs):
        call_shapes.append((tuple(args[0].shape), tuple(args[1].shape)))
        return original(*args, **kwargs)

    monkeypatch.setattr(loader_module, "sample_xy_maps_bilinear", counted)

    samples, images, coords, valids = loader.build_sample(0, sample_mode="flat")

    assert call_shapes
    assert any(shape[0][0] == len(loader.strip_z_offsets) for shape in call_shapes)
    assert len(samples) == len(loader.strip_z_offsets)
    assert images.shape[0] == len(loader.strip_z_offsets)
    assert coords.shape[0] == len(loader.strip_z_offsets)
    assert valids.shape[0] == len(loader.strip_z_offsets)


def test_deferred_batch_image_augmentation_matches_inline_batch(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["augment_enabled"] = True
    raw["augment_device"] = "cpu"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    loader = _make_loader(load_config(config_path))
    inline = loader.load_batch(0, batch_size=1, sample_mode="flat", apply_image_augmentation=True)
    deferred_base = loader.load_batch(0, batch_size=1, sample_mode="flat", apply_image_augmentation=False)
    deferred = loader.apply_batch_image_augmentation(deferred_base)

    assert len(deferred_base.augmentation_params) == deferred_base.images.shape[0] * deferred_base.images.shape[1]
    np.testing.assert_allclose(deferred.images, inline.images)
    np.testing.assert_array_equal(deferred.valid_mask, inline.valid_mask)


def test_batched_value_blur_matches_single_patch_path() -> None:
    images = np.arange(2 * 7 * 7, dtype=np.float32).reshape(2, 7, 7)
    valid = np.ones((2, 7, 7), dtype=bool)
    params = (
        FiberStripAugmentParams(blur_sigma=0.5),
        FiberStripAugmentParams(blur_sigma=1.25),
    )

    batch_images, batch_valid = apply_value_augmentation_batch(
        images,
        valid,
        params,
        device=torch.device("cpu"),
    )
    single_images = []
    for index, param in enumerate(params):
        image, single_valid = apply_value_augmentation(
            images[index],
            valid[index],
            param,
            device=torch.device("cpu"),
        )
        single_images.append(image)
        assert torch.equal(single_valid, batch_valid[index])

    torch.testing.assert_close(batch_images, torch.stack(single_images), rtol=1.0e-5, atol=1.0e-5)


def test_training_rejects_batch_size_control_point_mismatch(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=2)
    loader_config = load_config(config_path)
    training = FiberStripTrainingConfig(train_control_points_per_step=3)

    with pytest.raises(ValueError, match="control_points_per_step.*batch_size"):
        _validate_training_batch_config(training, loader_config)


def test_training_batch_pipeline_preserves_step_sample_order() -> None:
    class FakeLoader:
        def __init__(self) -> None:
            self.starts: list[int] = []

        def load_batch(self, start_sample_index: int, **kwargs):
            self.starts.append(int(start_sample_index))
            return SimpleNamespace(start=int(start_sample_index))

    training = FiberStripTrainingConfig(
        train_control_points_per_step=2,
        max_sample_index=0,
        pipeline_depth=3,
    )
    loader = FakeLoader()
    with _TrainingBatchPipeline(
        loader,  # type: ignore[arg-type]
        training,
        sample_mode="random",
        start_step=1,
        max_step=3,
        profile_enabled=False,
        apply_image_augmentation=False,
    ) as pipeline:
        first, _ = pipeline.next()
        second, _ = pipeline.next()
        third, _ = pipeline.next()

    assert [first.raw_start_sample_index, second.raw_start_sample_index, third.raw_start_sample_index] == [0, 2, 4]
    assert [first.batch.start, second.batch.start, third.batch.start] == [0, 2, 4]
    assert loader.starts == [0, 2, 4]


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


def test_augment_contact_sheet_export_writes_jpg(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["augment_enabled"] = True
    raw["augment_device"] = "cpu"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))
    out = tmp_path / "aug_out"

    _export_augment_contact_sheet(loader, 0, out)

    output = capsys.readouterr().out
    assert "augment-vis timings" not in output
    assert "output timings" not in output
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


def test_augment_contact_sheet_profile_prints_cold_and_warm_tables(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["augment_enabled"] = True
    raw["augment_device"] = "cpu"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))
    out = tmp_path / "aug_profile_out"

    _export_augment_contact_sheet(loader, 0, out, profile=True)

    output = capsys.readouterr().out
    assert "fiber_trace_2d augment-vis timings pass=1 cold-ish in ms" in output
    assert "fiber_trace_2d augment-vis timings pass=2 warm in ms" in output
    assert "total/no-first" in output
    assert "avg/no-first" in output
    assert (out / "augment_contact_sheet.jpg").is_file()


def test_deterministic_sample_index_independent_of_batch_size(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    config_a = load_config(config_path)
    config_b = load_config(config_path)
    loader_a = _make_loader(config_a)
    loader_b = _make_loader(config_b)

    a = [loader_a.descriptor_for_sample_index(i)[2] for i in range(8)]
    b = [loader_b.descriptor_for_sample_index(i)[2] for i in range(8)]
    assert a == b


def test_random_sample_mode_covers_all_control_points_before_repeating(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, batch_size=1))
    loader = _make_loader(config)

    first_pass = [loader._random_flat_index(i) for i in range(loader.sample_count)]
    second_pass = [
        loader._random_flat_index(i)
        for i in range(loader.sample_count, loader.sample_count * 2)
    ]
    again = [loader._random_flat_index(i) for i in range(loader.sample_count)]

    assert sorted(first_pass) == list(range(loader.sample_count))
    assert sorted(second_pass) == list(range(loader.sample_count))
    assert first_pass == again


def test_flat_sample_mode_iterates_control_points_sequentially_for_debug(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, batch_size=1))
    loader = _make_loader(config)

    indices = [loader.descriptor_for_sample_index(i, sample_mode="flat")[2] for i in range(5)]

    assert indices == [0, 1, 2, 0, 1]


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
    assert summary["workers"] == 8
    assert summary["producer_workers"] == 1
    assert (store.root / "new_missing.empty").is_file()
    assert (store.root / "new_missing.empty").read_bytes() == b""
    assert (store.root / "download.bin").read_bytes() == b"\x01\x02"
    output = capsys.readouterr().out
    assert "samples[" in output
    assert "downloads[" in output
    assert "idx=1" in output
    assert "samplers=1" in output


def test_prefetch_idx_requires_cache_complete_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["prefetch_sampler_workers"] = 1
    raw["volume_cache_retry_seconds"] = 0.001
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))

    class Store:
        def __init__(self, root: Path) -> None:
            self.root = root

    store = Store(tmp_path / "cache")
    store.root.mkdir(parents=True)
    (store.root / "hit.bin").write_bytes(b"cached")
    call_count = 0

    def failing_downloader(request: ZarrChunkRequest) -> bytes:
        del request
        raise RuntimeError("network failed")

    def request(key: str) -> ZarrChunkRequest:
        return ZarrChunkRequest(
            store=store,
            store_identity="idx-test",
            key=key,
            cache_path=store.root / f"{key}.bin",
            empty_path=store.root / f"{key}.empty",
            remote_url=f"https://example.invalid/{key}",
            cache_payload_format="source_bytes",
            downloader=failing_downloader,
        )

    def fake_chunk_requests(coords, valid):
        nonlocal call_count
        del coords, valid
        call_count += 1
        return [request("error" if call_count <= len(loader.strip_z_offsets) else "hit")]

    monkeypatch.setattr(loader.records[0].sampler, "chunk_requests_for_coords", fake_chunk_requests)

    summary = loader.prefetch(0, 2, workers=1)

    assert summary["errors"] == 1
    assert summary["cache_hits"] == 1
    assert summary["max_exclusive_sample_index"] == 0
    output = capsys.readouterr().out
    assert "idx=0" in output
    assert "idx=1" not in output


def test_prefetch_sampler_workers_limits_dependency_producers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["prefetch_workers"] = 8
    raw["prefetch_sampler_workers"] = 2
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))
    monkeypatch.setattr(loader.records[0].sampler, "chunk_requests_for_coords", lambda coords, valid: [])

    summary = loader.prefetch(0, 3)

    assert summary["workers"] == 8
    assert summary["producer_workers"] == 2


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
    config_path = _write_config(tmp_path, batch_size=2)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["loader_workers"] = 1
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))
    original_prepare_sample = loader._prepare_sample
    attempted: list[int] = []

    def prepare_sample(sample_index: int, **kwargs):
        attempted.append(int(sample_index))
        if int(sample_index) == 0:
            raise ValueError("Lasagna grad_mag sample is zero at fiber line point")
        return original_prepare_sample(sample_index, **kwargs)

    monkeypatch.setattr(loader, "_prepare_sample", prepare_sample)

    batch = loader.load_batch(0, batch_size=2)

    assert attempted[:3] == [0, 1, 2]
    assert batch.images.shape[0] == 2
    assert batch.coords_zyx.shape[0] == 2
    assert loader._load_batch_skipped_samples == 1


def test_load_batch_wraps_through_sample_index_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["loader_workers"] = 1
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))
    attempted: list[int] = []

    original_prepare_sample = loader._prepare_sample

    def prepare_sample(
        sample_index: int,
        *,
        sample_mode: str = "random",
        profile=None,
    ):
        attempted.append(int(sample_index))
        return original_prepare_sample(sample_index, sample_mode=sample_mode, profile=profile)

    monkeypatch.setattr(loader, "_prepare_sample", prepare_sample)

    batch = loader.load_batch(1, batch_size=4, sample_index_limit=2)

    assert attempted == [1, 0, 1, 0]
    assert batch.images.shape == (4, len(loader.strip_z_offsets), 1, 3, 3)


def test_parallel_load_batch_preserves_output_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_config(tmp_path, batch_size=3)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["loader_workers"] = 4
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))
    original_prepare_sample = loader._prepare_sample

    def prepare_sample(
        sample_index: int,
        *,
        sample_mode: str = "random",
        profile=None,
    ):
        if int(sample_index) == 0:
            loader_module.time.sleep(0.02)
        return original_prepare_sample(sample_index, sample_mode=sample_mode, profile=profile)

    monkeypatch.setattr(loader, "_prepare_sample", prepare_sample)

    batch = loader.load_batch(0, batch_size=3)

    assert batch.control_point_indices.tolist() == [0, 1, 2]


def test_parallel_load_batch_reuses_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_config(tmp_path, batch_size=2)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["loader_workers"] = 2
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))

    def build_sample(
        sample_index: int,
        *,
        sample_mode: str = "random",
        profile=None,
        apply_image_augmentation: bool = True,
    ):
        del sample_mode
        del profile
        del apply_image_augmentation
        sample = SimpleNamespace(
            record_index=0,
            control_point_index=int(sample_index),
            fiber_path=f"fiber_{sample_index}.json",
        )
        image = np.full((len(loader.strip_z_offsets), 3, 3), float(sample_index), dtype=np.float32)
        coords = np.zeros((len(loader.strip_z_offsets), 3, 3, 3), dtype=np.float32)
        valid = np.ones((len(loader.strip_z_offsets), 3, 3), dtype=bool)
        return [sample], image, coords, valid

    monkeypatch.setattr(loader, "build_sample", build_sample)

    loader.load_batch(0, batch_size=2)
    first_executor = loader._loader_executor
    loader.load_batch(2, batch_size=2)

    assert first_executor is not None
    assert loader._loader_executor is first_executor
    loader.close()
    assert loader._loader_executor is None


def test_random_pass_order_is_cached_before_parallel_worker_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, batch_size=2)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["loader_workers"] = 2
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    loader = _make_loader(load_config(config_path))
    original_ensure = loader._ensure_random_pass_order
    calls: list[int] = []

    def ensure_random_pass_order(pass_index: int):
        calls.append(int(pass_index))
        return original_ensure(pass_index)

    monkeypatch.setattr(loader, "_ensure_random_pass_order", ensure_random_pass_order)

    def build_sample(
        sample_index: int,
        *,
        sample_mode: str = "random",
        profile=None,
        apply_image_augmentation: bool = True,
    ):
        del profile
        del apply_image_augmentation
        record, record_index, control_index = loader.descriptor_for_sample_index(
            sample_index, sample_mode=sample_mode
        )
        del record
        sample = SimpleNamespace(
            record_index=record_index,
            control_point_index=control_index,
            fiber_path=f"fiber_{control_index}.json",
        )
        image = np.full((len(loader.strip_z_offsets), 3, 3), float(control_index), dtype=np.float32)
        coords = np.zeros((len(loader.strip_z_offsets), 3, 3, 3), dtype=np.float32)
        valid = np.ones((len(loader.strip_z_offsets), 3, 3), dtype=bool)
        return [sample], image, coords, valid

    monkeypatch.setattr(loader, "build_sample", build_sample)

    loader.load_batch(0, batch_size=2)

    assert calls
    assert 0 in loader._random_pass_cache
    assert loader._random_pass_cache[0].shape[0] == loader.sample_count


class _RecordingCoordinateSampler(NumpyZarrCoordinateSampler):
    def __init__(self, array, *, level_spacing_base: float) -> None:
        super().__init__(array, level_spacing_base=level_spacing_base)
        self.sample_coords_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.sample_coord_batch_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.chunk_request_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.prefetch_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def sample_coords(self, coords_zyx_base: np.ndarray, valid_mask: np.ndarray):
        self.sample_coords_calls.append((np.array(coords_zyx_base, copy=True), np.array(valid_mask, copy=True)))
        return super().sample_coords(coords_zyx_base, valid_mask)

    def sample_coord_batch(self, coords_zyx_base: np.ndarray, valid_mask: np.ndarray):
        self.sample_coord_batch_calls.append(
            (np.array(coords_zyx_base, copy=True), np.array(valid_mask, copy=True))
        )
        return super().sample_coord_batch(coords_zyx_base, valid_mask)

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
    assert len(sampler.sample_coord_batch_calls) == 1
    assert len(sampler.sample_coords_calls) == 1
    prefetch_keys = _request_keys(prefetch_requests)
    assert prefetch_keys
    for load_coords, load_valid in sampler.sample_coord_batch_calls:
        assert load_coords.shape[:3] == (2, 5, 5)
        assert load_valid.shape == (2, 5, 5)
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
    calls: list[tuple[int, int, str, int]] = []

    def fake_prefetch(
        self,
        start_sample_index,
        sample_count,
        *,
        workers=None,
        sample_mode="random",
        sample_index_limit=None,
    ):
        del self, workers
        calls.append((int(start_sample_index), int(sample_count), str(sample_mode), int(sample_index_limit or 0)))
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

    assert calls == [(0, 9, "random", 0)]
    assert summary["generated"] == 11
    assert not run_path.exists()

    calls.clear()
    prefetch_training(
        config_path,
        prefetch_steps=None,
        prefetch_start_step=2,
        sampler_factory=_test_sampler_factory,
    )

    assert calls == [(3, 21, "random", 0)]
    assert not run_path.exists()

    calls.clear()
    prefetch_training(
        config_path,
        prefetch_steps=0,
        prefetch_start_step=2,
        sampler_factory=_test_sampler_factory,
    )

    assert calls == [(0, 3, "random", 0)]


def test_training_prefetch_respects_max_sample_index_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["training"] = {
        "max_steps": 7,
        "max_sample_index": 2,
        "control_points_per_step": 3,
        "tensorboard_enabled": False,
    }
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    calls: list[tuple[int, int, str, int]] = []

    def fake_prefetch(
        self,
        start_sample_index,
        sample_count,
        *,
        workers=None,
        sample_mode="random",
        sample_index_limit=None,
    ):
        del self, workers
        calls.append((int(start_sample_index), int(sample_count), str(sample_mode), int(sample_index_limit or 0)))
        return {"generated": 0, "missing": 0, "downloaded": 0, "bytes": 0, "errors": 0, "workers": 1}

    monkeypatch.setattr(FiberStrip2DLoader, "prefetch", fake_prefetch)

    prefetch_training(config_path, prefetch_steps=0, sampler_factory=_test_sampler_factory)

    assert calls == [(0, 2, "random", 2)]


def test_training_prefetch_max_steps_zero_prefetches_dataset_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["training"] = {
        "max_steps": 0,
        "control_points_per_step": 2,
        "tensorboard_enabled": False,
    }
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    calls: list[tuple[int, int, str]] = []

    def fake_prefetch(
        self,
        start_sample_index,
        sample_count,
        *,
        workers=None,
        sample_mode="random",
        sample_index_limit=None,
    ):
        del workers
        calls.append((int(start_sample_index), int(sample_count), str(sample_mode)))
        return {
            "generated": 3,
            "missing": 0,
            "downloaded": 0,
            "bytes": 0,
            "errors": 0,
            "workers": 1,
            "samples": int(sample_count),
            "loader_sample_count": int(self.sample_count),
        }

    monkeypatch.setattr(FiberStrip2DLoader, "prefetch", fake_prefetch)

    summary = prefetch_training(config_path, prefetch_steps=0, sampler_factory=_test_sampler_factory)

    assert calls == [(0, 3, "random")]
    assert summary["samples"] == 3
    assert summary["loader_sample_count"] == 3


def test_training_prefetch_max_steps_zero_prefetches_test_datasets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    test_fiber = _write_fiber(tmp_path / "heldout_fiber.json", points=[[8.0, 24.0, 24.0], [18.0, 24.0, 24.0]])
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["test_datasets"] = [
        {
            **raw["datasets"][0],
            "fiber_paths": [str(test_fiber)],
        }
    ]
    raw["training"] = {
        "max_steps": 0,
        "control_points_per_step": 2,
        "tensorboard_enabled": False,
    }
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    calls: list[tuple[int, int, str, int]] = []

    def fake_prefetch(
        self,
        start_sample_index,
        sample_count,
        *,
        workers=None,
        sample_mode="random",
        sample_index_limit=None,
    ):
        del workers
        calls.append((int(start_sample_index), int(sample_count), str(sample_mode), int(self.sample_count)))
        return {
            "generated": int(sample_count),
            "missing": 0,
            "downloaded": 0,
            "bytes": 0,
            "errors": 0,
            "workers": 1,
            "samples": int(sample_count),
        }

    monkeypatch.setattr(FiberStrip2DLoader, "prefetch", fake_prefetch)

    summary = prefetch_training(config_path, prefetch_steps=0, sampler_factory=_test_sampler_factory)

    assert calls == [(0, 3, "random", 3), (0, 2, "random", 2)]
    assert summary["samples"] == 5


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


def test_training_config_allows_zero_max_steps() -> None:
    config = _training_config_from_raw({"training": {"max_steps": 0}})

    assert config.max_steps == 0


def test_training_config_parses_max_sample_index() -> None:
    config = _training_config_from_raw({"training": {"max_sample_index": 123}})

    assert config.max_sample_index == 123


def test_training_config_rejects_negative_max_sample_index() -> None:
    with pytest.raises(ValueError, match="training.max_sample_index must be >= 0"):
        _training_config_from_raw({"training": {"max_sample_index": -1}})


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


def test_direction_angle_error_reports_degrees() -> None:
    valid = np.ones((1, 5, 5), dtype=bool)
    sample = SimpleNamespace(
        line_xy=np.asarray(
            [
                [0.0, 2.0],
                [1.0, 2.0],
                [2.0, 2.0],
                [3.0, 2.0],
                [4.0, 2.0],
            ],
            dtype=np.float32,
        ),
        control_point_xy=np.asarray([2.0, 2.0], dtype=np.float32),
    )
    supervision = build_direction_supervision([sample], valid, device=torch.device("cpu"))
    perfect = torch.zeros((1, 2, 5, 5), dtype=torch.float32)
    perpendicular = torch.zeros((1, 2, 5, 5), dtype=torch.float32)
    perfect_value = encode_lasagna_direction_xy(torch.tensor([1.0, 0.0], dtype=torch.float32))
    perpendicular_value = encode_lasagna_direction_xy(torch.tensor([0.0, 1.0], dtype=torch.float32))
    perfect[:, :, :, :] = perfect_value.view(1, 2, 1, 1)
    perpendicular[:, :, :, :] = perpendicular_value.view(1, 2, 1, 1)

    perfect_error = direction_angle_error_degrees(perfect, supervision)
    perpendicular_error = direction_angle_error_degrees(perpendicular, supervision)

    assert float(perfect_error.max()) < 0.5
    assert float(perpendicular_error.min()) > 89.0


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


def test_training_visualization_selects_different_control_points_first() -> None:
    batch = SimpleNamespace(
        images=np.zeros((4, 16, 1, 5, 5), dtype=np.float32),
        strip_z_offsets=np.asarray(
            [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            dtype=np.float32,
        ),
    )

    indices = _select_visualization_patch_indices(batch, max_patches=8)

    assert indices[:4] == [7, 23, 39, 55]
    assert [index // 16 for index in indices[:4]] == [0, 1, 2, 3]


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


def test_training_benchmark_reports_patch_throughput_without_run_dir(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["patch_shape_hw"] = [5, 5]
    raw["training"] = {
        "run_path": str(tmp_path / "runs"),
        "run_name": "benchmark",
        "max_steps": 1,
        "control_points_per_step": 1,
        "tensorboard_enabled": False,
        "model_hidden_channels": 4,
        "model_depth": 2,
    }
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    summary = run_benchmark(config_path, sampler_factory=_test_sampler_factory, batches=2, profile=True)

    assert summary.batches == 2
    assert summary.patches == 6
    assert summary.patches_per_second > 0.0
    assert "coord_gen" in summary.stage_ms_per_patch
    assert "loader_wall" in summary.stage_ms_per_patch
    assert "loader_worker" in summary.stage_ms_per_patch
    assert not (tmp_path / "runs").exists()


def test_training_load_only_benchmark_skips_image_aug_and_model_work(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, batch_size=1)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["patch_shape_hw"] = [5, 5]
    raw["augment_enabled"] = True
    raw["augment_device"] = "cpu"
    raw["training"] = {
        "run_path": str(tmp_path / "runs"),
        "run_name": "load_only",
        "max_steps": 1,
        "control_points_per_step": 1,
        "tensorboard_enabled": False,
        "model_hidden_channels": 4,
        "model_depth": 2,
    }
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    summary = run_benchmark(
        config_path,
        sampler_factory=_test_sampler_factory,
        batches=2,
        profile=True,
        load_only=True,
    )

    assert summary.batches == 2
    assert summary.patches == 6
    assert summary.stage_ms_per_patch["loading"] >= 0.0
    assert summary.stage_ms_per_patch["image_aug"] == 0.0
    assert summary.stage_ms_per_patch["fw"] == 0.0
    assert summary.stage_ms_per_patch["bw_step"] == 0.0
    assert not (tmp_path / "runs").exists()


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
