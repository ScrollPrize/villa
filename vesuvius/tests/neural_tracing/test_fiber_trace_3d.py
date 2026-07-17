from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from vesuvius.neural_tracing.fiber_trace.fiber_json import Vc3dFiber
from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import FiberStripFrame
from vesuvius.neural_tracing.fiber_trace_2d.loader_support import ZarrChunkRequest
from vesuvius.neural_tracing.fiber_trace_2d.sampling import (
    CoordinateSampleResult,
    NumpyZarrCoordinateSampler,
    Vc3dCoordinateSampler,
)
from vesuvius.neural_tracing.fiber_trace_3d import loader as loader3d_module
from vesuvius.neural_tracing.fiber_trace_3d.direction import (
    decode_lasagna_direction_3x2_analytic,
    encode_lasagna_direction_2d,
    encode_lasagna_direction_3x2,
)
from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    DEFAULT_VOLUME_CACHE_MEMORY_MIB,
    FiberTrace3DBatch,
    FiberTrace3DLoader,
    _TARGET_MODE_CP_ONLY,
    _TARGET_MODE_DENSE_LINE,
    _anisotropic_blur_3d,
    config_from_mapping,
    load_config,
)
from vesuvius.neural_tracing.fiber_trace_3d.model import (
    FiberTrace3DModelConfig,
    FiberTrace3DNet,
    direction_output,
    direction_outputs,
    presence_output,
    presence_outputs,
)
from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets
from vesuvius.neural_tracing.fiber_trace_3d.projection import (
    project_3d_direction_to_2d_frame,
)
from vesuvius.neural_tracing.fiber_trace_3d.trace2cp_bridge import (
    project_3d_output_to_trace2cp_fields,
    score_trace2cp_projected_fields,
)
from vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool import (
    NativeTrace2CpConfig,
    NativeTraceFieldCache,
    NativeTraceResult,
    _adaptive_trace2cp_cross_strip_height,
    _build_native_whole_fiber_span_source,
    _fiber_line_tangent_zyx_toward_target,
    _image_to_u8,
    _interpolate_plane_crossing,
    _NativeLasagnaNormalSampler,
    _native_cumulative_tangent_smoothness_loss_torch,
    _native_trace2cp_whole_fiber_mode,
    _native_whole_fiber_visual_spans,
    _native_smoothness_loss_torch,
    _parse_args,
    _native_trace_geometry_normal_record,
    _project_trace_to_initial_strip,
    _resolve_native_trace2cp_selection,
    _sample_presence_on_strip,
    _sample_trace_start_direction_aligned,
    _sample_trace_point_aligned,
    _score_candidate_batch,
    _score_candidate_loss_tensors_batched,
    _target_plane_in_plane_error_voxels,
    _trace_candidate_directions_torch,
    _volume_trace_to_source_trace_xyz,
    fuse_forward_reverse_traces,
    generate_cone_candidates_by_angle_step,
    generate_cone_candidates,
    trace_native_3d_one_way,
    trace_native_3d_pair,
    trace_native_3d_whole_fiber,
)
from vesuvius.neural_tracing.fiber_trace_3d.train import compute_losses
from vesuvius.neural_tracing.fiber_trace_3d.train import (
    _FiberTrace3DBatchDataset,
    _Trace2Cp3DConfig,
    _branch_presence_views_from_sampled_output,
    _evaluate_trace2cp_metric_fixed_set_3d,
    _draw_panel_line_aa,
    _draw_projected_cp_direction,
    _identity_batch_collate,
    _make_batch_dataloader,
    _make_test_loader_raw_config,
    _make_train_sample_3d_contact_sheet,
    _make_train_sample_3d_sheet,
    _oblique_line_presence_for_display,
    _resolve_dense_test_selection,
    _resolve_prefetch_sample_count,
    _training_sample_index_limit,
    _write_3d_sample_sheet,
)


def _straight_fiber() -> Vc3dFiber:
    x = np.arange(20, 45, dtype=np.float32)
    y = np.full_like(x, 32.0)
    z = np.full_like(x, 32.0)
    points = np.stack([x, y, z], axis=1)
    return Vc3dFiber(
        path=None,
        version=1,
        line_points_xyz=points,
        control_points_xyz=points[::4].copy(),
        generation=1,
        metadata={},
    )


def _long_segment_fiber(*, source_format: str | None = None) -> Vc3dFiber:
    points = np.asarray(
        [
            [0.0, 32.0, 32.0],
            [32.0, 32.0, 32.0],
            [80.0, 32.0, 32.0],
        ],
        dtype=np.float32,
    )
    return Vc3dFiber(
        path=None,
        version=1,
        line_points_xyz=points,
        control_points_xyz=points[1:2].copy(),
        generation=1,
        metadata={} if source_format is None else {"source_format": source_format},
    )


def _loader(*, augment_enabled: bool = False) -> FiberTrace3DLoader:
    z, y, x = np.meshgrid(
        np.arange(64, dtype=np.float32),
        np.arange(64, dtype=np.float32),
        np.arange(64, dtype=np.float32),
        indexing="ij",
    )
    volume = z * 0.01 + y * 0.1 + x
    config = config_from_mapping(
        {
            "batch_size": 2,
            "patch_shape_zyx": [16, 16, 16],
            "seed": 7,
            "cp_margin_voxels": 3,
            "presence_radius_voxels": 1.5,
            "image_normalization": "none",
            "augment_enabled": augment_enabled,
            "augment_shift_zyx": [3, 3, 3],
            "augment_rotation_degrees": 10.0,
            "augment_scale_min": 0.95,
            "augment_scale_max": 1.05,
            "round_source_to_chunk_boundaries": False,
            "_array_records": [
                {
                    "volume": volume,
                    "fiber": _straight_fiber(),
                    "volume_path": "synthetic",
                }
            ],
        }
    )
    return FiberTrace3DLoader(config)


def _native_trace_record(
    volume: np.ndarray,
    *,
    spacing_base: float = 1.0,
    sampler: object | None = None,
    base_shape_zyx: tuple[int, int, int] | None = None,
) -> SimpleNamespace:
    volume_arr = np.asarray(volume, dtype=np.float32)
    spacing = float(spacing_base)
    if base_shape_zyx is None:
        base_shape_zyx = tuple(
            int(round(float(size) * spacing)) for size in volume_arr.shape
        )
    if sampler is None:
        sampler = NumpyZarrCoordinateSampler(
            volume_arr,
            level_spacing_base=spacing,
        )
    return SimpleNamespace(
        volume=volume_arr,
        sampler=sampler,
        volume_spacing_base=spacing,
        base_shape_zyx=tuple(int(v) for v in base_shape_zyx),
    )


def _long_segment_loader(
    *,
    source_format: str | None = None,
    presence_negative_edge_margin_voxels: int | None = None,
) -> FiberTrace3DLoader:
    z, y, x = np.meshgrid(
        np.arange(96, dtype=np.float32),
        np.arange(96, dtype=np.float32),
        np.arange(96, dtype=np.float32),
        indexing="ij",
    )
    volume = z * 0.01 + y * 0.1 + x
    raw = {
        "batch_size": 1,
        "patch_shape_zyx": [16, 16, 16],
        "seed": 11,
        "cp_margin_voxels": 3,
        "presence_radius_voxels": 1.5,
        "image_normalization": "none",
        "augment_enabled": False,
        "round_source_to_chunk_boundaries": False,
        "_array_records": [
            {
                "volume": volume,
                "fiber": _long_segment_fiber(source_format=source_format),
                "volume_path": "synthetic",
            }
        ],
    }
    if presence_negative_edge_margin_voxels is not None:
        raw["presence_negative_edge_margin_voxels"] = int(
            presence_negative_edge_margin_voxels
        )
    config = config_from_mapping(raw)
    return FiberTrace3DLoader(config)


def _loader_with_mapping(raw: dict) -> FiberTrace3DLoader:
    z, y, x = np.meshgrid(
        np.arange(64, dtype=np.float32),
        np.arange(64, dtype=np.float32),
        np.arange(64, dtype=np.float32),
        indexing="ij",
    )
    volume = z * 0.01 + y * 0.1 + x
    base = {
        "batch_size": 1,
        "patch_shape_zyx": [16, 16, 16],
        "seed": 13,
        "cp_margin_voxels": 3,
        "presence_radius_voxels": 1.5,
        "image_normalization": "none",
        "augment_enabled": True,
        "augment_shift_zyx": [3, 3, 3],
        "augment_rotation_degrees": 10.0,
        "augment_scale_min": 0.95,
        "augment_scale_max": 1.05,
        "round_source_to_chunk_boundaries": False,
        "_array_records": [
            {
                "volume": volume,
                "fiber": _straight_fiber(),
                "volume_path": "synthetic",
            }
        ],
    }
    base.update(raw)
    return FiberTrace3DLoader(config_from_mapping(base))


def test_3d_config_defaults_volume_cache_to_512_mib() -> None:
    for extra in ({}, {"volume_cache_memory_mib": None}):
        raw = {
            "batch_size": 1,
            "patch_shape_zyx": [16, 16, 16],
            "seed": 3,
            "_array_records": [
                {
                    "volume": np.zeros((16, 16, 16), dtype=np.float32),
                    "fiber": _straight_fiber(),
                    "volume_path": "synthetic",
                }
            ],
        }
        raw.update(extra)
        config = config_from_mapping(raw)

        assert config.volume_cache_memory_mib == DEFAULT_VOLUME_CACHE_MEMORY_MIB
        assert config.volume_cache_memory_bytes == 512 * 1024 * 1024


def test_3d_config_preserves_explicit_volume_cache_budget() -> None:
    config = config_from_mapping(
        {
            "batch_size": 1,
            "patch_shape_zyx": [16, 16, 16],
            "volume_cache_memory_mib": 128,
            "_array_records": [
                {
                    "volume": np.zeros((16, 16, 16), dtype=np.float32),
                    "fiber": _straight_fiber(),
                    "volume_path": "synthetic",
                }
            ],
        }
    )

    assert config.volume_cache_memory_mib == 128.0
    assert config.volume_cache_memory_bytes == 128 * 1024 * 1024


def _load_materialized_batch(
    loader: FiberTrace3DLoader,
    start_sample_index: int,
    *,
    sample_mode: str = "random",
) -> object:
    batch = loader.load_batch(start_sample_index, sample_mode=sample_mode)
    return materialize_targets(batch, loader.config, profile=True)


def _has_sparse_direction_at(batch: object, sample: int, z: int, y: int, x: int) -> bool:
    indices = batch.direction_indices_bzyx
    assert indices is not None
    target = torch.tensor([sample, z, y, x], dtype=indices.dtype, device=indices.device)
    return bool(torch.any(torch.all(indices == target.view(1, 4), dim=1)))


def test_lasagna_3x2_encoding_matches_projection_formula() -> None:
    encoded_2d = encode_lasagna_direction_2d(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    )
    assert np.allclose(encoded_2d[0], [1.0, 0.5 + 0.5 / np.sqrt(2.0)])
    assert np.allclose(encoded_2d[1], [0.0, 0.5 - 0.5 / np.sqrt(2.0)])

    encoded_3d = encode_lasagna_direction_3x2(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    )[0]
    assert np.allclose(encoded_3d[:2], encoded_2d[0])
    assert np.allclose(encoded_3d[2:4], encoded_2d[0])
    assert np.allclose(encoded_3d[4:], [0.5, 0.5])


def test_lasagna_3x2_analytic_decode_round_trips_ambiguous_axes() -> None:
    axes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [-0.2, 0.4, 0.9],
            [1.0e-4, 1.0, 2.0e-4],
        ],
        dtype=torch.float32,
    )
    axes = axes / torch.linalg.vector_norm(axes, dim=1, keepdim=True)
    encoded = encode_lasagna_direction_3x2(axes)
    decoded = decode_lasagna_direction_3x2_analytic(encoded)
    agreement = torch.abs(torch.sum(decoded * axes, dim=1))
    assert torch.all(agreement > 0.999)


def test_loader_builds_deterministic_cp_centered_3d_batch() -> None:
    loader = _loader(augment_enabled=False)
    raw_batch = loader.load_batch(0)
    assert raw_batch.direction_target is None
    assert raw_batch.direction_indices_bzyx is None
    assert raw_batch.presence_target is None
    assert raw_batch.target_modes.shape == (2,)
    assert raw_batch.target_tangent_zyx.shape == (2, 3)
    batch_a = _load_materialized_batch(loader, 0)
    batch_b = _load_materialized_batch(loader, 0)
    assert batch_a.volume.shape == (2, 1, 16, 16, 16)
    assert batch_a.direction_target is None
    assert batch_a.direction_indices_bzyx.shape[1] == 4
    assert batch_a.direction_target_sparse.shape[1] == 6
    assert batch_a.presence_target.shape == (2, 1, 16, 16, 16)
    assert torch.equal(batch_a.sample_indices, batch_b.sample_indices)
    assert torch.allclose(batch_a.volume, batch_b.volume)
    assert int(batch_a.direction_indices_bzyx.shape[0]) > 0
    assert bool((batch_a.presence_target > 0.5).any())
    assert bool((batch_a.presence_target < 0.5).any())


def test_3d_batch_dataset_preserves_whole_batch_items() -> None:
    config = _loader(augment_enabled=False).config
    serial_dataset = _FiberTrace3DBatchDataset(
        config,
        start_batch_index=0,
        batch_count=3,
        sample_mode="random",
        worker_device="cpu",
    )
    serial_batches = [serial_dataset[index] for index in range(3)]
    dataloader_dataset = _FiberTrace3DBatchDataset(
        config,
        start_batch_index=0,
        batch_count=3,
        sample_mode="random",
        worker_device="cpu",
    )
    assert (
        _make_batch_dataloader(
            config,
            raw_config={"training": {"loader_workers": 0}},
            start_batch_index=0,
            batch_count=3,
            sample_mode="random",
        )
        is None
    )
    dataloader = DataLoader(
        dataloader_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        collate_fn=_identity_batch_collate,
    )
    dataloader_batches = list(dataloader)
    assert len(dataloader_batches) == len(serial_batches)
    for expected, actual in zip(serial_batches, dataloader_batches, strict=True):
        expected = materialize_targets(expected, config, profile=True)
        actual = materialize_targets(actual, config, profile=True)
        assert type(actual).__name__ == "FiberTrace3DBatch"
        assert torch.equal(actual.sample_indices, expected.sample_indices)
        assert torch.equal(actual.record_indices, expected.record_indices)
        assert torch.equal(actual.control_point_indices, expected.control_point_indices)
        assert actual.fiber_paths == expected.fiber_paths
        assert torch.allclose(actual.volume, expected.volume)
        assert torch.allclose(actual.presence_target, expected.presence_target)
        assert torch.equal(actual.direction_indices_bzyx, expected.direction_indices_bzyx)
        assert torch.allclose(actual.direction_target_sparse, expected.direction_target_sparse)


def test_3d_batch_dataset_applies_sample_index_limit_to_data_only() -> None:
    config = _loader(augment_enabled=True).config
    dataset = _FiberTrace3DBatchDataset(
        config,
        start_batch_index=0,
        batch_count=2,
        sample_index_limit=1,
        sample_mode="random",
        worker_device="cpu",
    )

    first = dataset[0]
    second = dataset[1]

    assert torch.equal(first.sample_indices, torch.zeros_like(first.sample_indices))
    assert torch.equal(second.sample_indices, torch.zeros_like(second.sample_indices))
    assert torch.equal(first.control_point_indices, second.control_point_indices)
    assert not torch.allclose(first.cp_local_zyx, second.cp_local_zyx)


def test_loader_augmented_batch_is_finite() -> None:
    loader = _loader(augment_enabled=True)
    batch = _load_materialized_batch(loader, 1)
    assert torch.isfinite(batch.volume).all()
    assert torch.isfinite(batch.direction_target_sparse).all()
    assert batch.volume.shape == (2, 1, 16, 16, 16)


def test_3d_augmentation_index_is_separate_from_data_index() -> None:
    loader = _loader(augment_enabled=True)
    record, _record_index, cp_index = loader.descriptor_for_sample_index(0, sample_mode="flat")

    params_a = loader._sample_augment_params(record, cp_index, 0)
    params_a_repeat = loader._sample_augment_params(record, cp_index, 0)
    params_b = loader._sample_augment_params(record, cp_index, 1000)

    assert np.allclose(params_a.cp_volume_zyx, params_b.cp_volume_zyx)
    assert np.allclose(params_a.cp_local_zyx, params_a_repeat.cp_local_zyx)
    assert np.allclose(params_a.source_to_output_zyx, params_a_repeat.source_to_output_zyx)
    assert not np.allclose(params_a.source_to_output_zyx, params_b.source_to_output_zyx)


def test_3d_bounded_data_index_reuses_cp_but_not_augmentation() -> None:
    loader = _loader(augment_enabled=True)

    batch_a = loader.load_batch(
        0,
        sample_mode="random",
        sample_index_limit=1,
    )
    batch_b = loader.load_batch(
        2,
        sample_mode="random",
        sample_index_limit=1,
    )

    assert torch.equal(batch_a.sample_indices, torch.zeros_like(batch_a.sample_indices))
    assert torch.equal(batch_b.sample_indices, torch.zeros_like(batch_b.sample_indices))
    assert torch.equal(batch_a.record_indices, batch_b.record_indices)
    assert torch.equal(batch_a.control_point_indices, batch_b.control_point_indices)
    assert not torch.allclose(batch_a.cp_local_zyx, batch_b.cp_local_zyx)


def test_loader_clips_long_label_segments_to_patch_domain() -> None:
    loader = _long_segment_loader(source_format="nml")
    raw_batch = loader.load_batch(0, sample_mode="flat")
    assert int(raw_batch.target_modes[0]) == _TARGET_MODE_DENSE_LINE
    assert int(raw_batch.target_segment_counts[0]) > 0
    batch = _load_materialized_batch(loader, 0, sample_mode="flat")
    assert batch.volume.shape == (1, 1, 16, 16, 16)
    assert int(batch.direction_indices_bzyx.shape[0]) > 0
    assert bool((batch.presence_target > 0.5).any())
    cp = torch.round(batch.cp_local_zyx[0]).to(dtype=torch.long)
    assert batch.presence_target[0, 0, int(cp[0]), int(cp[1]), int(cp[2])] > 0.5
    far_on_segment_x = min(int(cp[2]) + 4, 15)
    assert batch.presence_target[0, 0, int(cp[0]), int(cp[1]), far_on_segment_x] > 0.5
    off_line_y = min(int(cp[1]) + 1, 15)
    assert batch.presence_target[0, 0, int(cp[0]), off_line_y, far_on_segment_x] == 0.0


def test_nml_dense_line_presence_mask_covers_patch_edges() -> None:
    loader = _long_segment_loader(
        source_format="nml",
        presence_negative_edge_margin_voxels=4,
    )
    batch = _load_materialized_batch(loader, 0, sample_mode="flat")

    assert int(batch.target_modes[0]) == _TARGET_MODE_DENSE_LINE
    cp = torch.round(batch.cp_local_zyx[0]).to(dtype=torch.long)
    z = int(cp[0])
    y = int(cp[1])
    assert bool(batch.presence_mask[0, 0, z, y, 0])
    assert batch.presence_target[0, 0, z, y, 0] > 0.5
    assert bool(batch.presence_mask[0, 0, 0, 0, 0])
    assert batch.presence_target[0, 0, 0, 0, 0] == 0.0


def test_nml_dense_line_batch_keeps_cp_tangent_for_oblique_visuals() -> None:
    loader = _long_segment_loader(source_format="nml")
    raw_batch = loader.load_batch(0, sample_mode="flat")

    assert int(raw_batch.target_modes[0]) == _TARGET_MODE_DENSE_LINE
    assert torch.allclose(
        raw_batch.target_tangent_zyx[0],
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        atol=1.0e-5,
    )


def test_cp_only_presence_mask_keeps_edge_exclusion() -> None:
    loader = _long_segment_loader(
        source_format=None,
        presence_negative_edge_margin_voxels=4,
    )
    batch = _load_materialized_batch(loader, 0, sample_mode="flat")

    assert int(batch.target_modes[0]) == _TARGET_MODE_CP_ONLY
    assert not bool(batch.presence_mask[0, 0, 0, 0, 0])
    cp = torch.round(batch.cp_local_zyx[0]).to(dtype=torch.long)
    assert bool(batch.presence_mask[0, 0, int(cp[0]), int(cp[1]), int(cp[2])])


def test_loader_uses_cp_only_supervision_for_non_nml_fibers() -> None:
    loader = _long_segment_loader(source_format=None)
    raw_batch = loader.load_batch(0, sample_mode="flat")
    assert int(raw_batch.target_modes[0]) == _TARGET_MODE_CP_ONLY
    assert int(raw_batch.target_segment_counts[0]) > 0
    batch = _load_materialized_batch(loader, 0, sample_mode="flat")
    cp = torch.round(batch.cp_local_zyx[0]).to(dtype=torch.long)
    far_on_segment_x = min(int(cp[2]) + 4, 15)
    assert batch.presence_target[0, 0, int(cp[0]), int(cp[1]), int(cp[2])] > 0.5
    assert batch.presence_target[0, 0, int(cp[0]), int(cp[1]), far_on_segment_x] == 0.0
    assert not _has_sparse_direction_at(batch, 0, int(cp[0]), int(cp[1]), far_on_segment_x)


def test_3d_sheet_visualizes_line_context_for_cp_only_presence() -> None:
    loader = _long_segment_loader(source_format=None)
    batch = _load_materialized_batch(loader, 0, sample_mode="flat")
    cp = torch.round(batch.cp_local_zyx[0]).to(dtype=torch.long)
    far_on_segment_x = min(int(cp[2]) + 4, 15)
    assert batch.presence_target[0, 0, int(cp[0]), int(cp[1]), far_on_segment_x] == 0.0

    output = torch.zeros(
        (
            int(batch.volume.shape[0]),
            7,
            *tuple(int(v) for v in batch.volume.shape[-3:]),
        ),
        dtype=batch.volume.dtype,
    )
    sheet = _make_train_sample_3d_sheet(batch, output)
    patch = int(batch.volume.shape[-1])
    gap = 4
    target_panel = sheet[:patch, patch + gap : patch * 2 + gap]

    assert target_panel[int(cp[1]), far_on_segment_x, 0] > 0


def test_train_sample_3d_sheet_contains_principal_slice_panels() -> None:
    loader = _loader(augment_enabled=False)
    batch = _load_materialized_batch(loader, 0)
    model = FiberTrace3DNet(
        FiberTrace3DModelConfig(
            input_channels=1,
            output_channels=7,
            features_per_stage=(4, 8),
            strides=((1, 1, 1), (2, 2, 2)),
            decoder_upsample_mode="pixelshuffle",
        )
    )
    with torch.no_grad():
        output = model(batch.volume[:1])
    sheet = _make_train_sample_3d_sheet(batch, output)
    assert sheet.ndim == 3
    assert sheet.shape[2] == 3
    assert sheet.dtype == np.uint8
    assert sheet.shape[0] > 16
    assert sheet.shape[1] == 16 * 7 + 6 * 4
    first_image_panel = sheet[:16, :16]
    colored = (
        (first_image_panel[..., 0] != first_image_panel[..., 1])
        | (first_image_panel[..., 1] != first_image_panel[..., 2])
    )
    assert bool(np.any(colored))


def test_train_sample_3d_sheet_maxpools_target_presence_for_display() -> None:
    patch = 8
    cp = torch.tensor([[4.0, 4.0, 4.0]], dtype=torch.float32)
    target = torch.zeros((1, 1, patch, patch, patch), dtype=torch.float32)
    target[0, 0, 5, 5, 4] = 1.0
    batch = FiberTrace3DBatch(
        volume=torch.zeros((1, 1, patch, patch, patch), dtype=torch.float32),
        valid_mask=torch.ones((1, 1, patch, patch, patch), dtype=torch.bool),
        cp_local_zyx=cp,
        crop_origin_zyx=torch.zeros((1, 3), dtype=torch.float32),
        sample_indices=torch.zeros((1,), dtype=torch.long),
        record_indices=torch.zeros((1,), dtype=torch.long),
        control_point_indices=torch.zeros((1,), dtype=torch.long),
        fiber_paths=("synthetic.json",),
        target_modes=torch.ones((1,), dtype=torch.long),
        target_segment_offsets=torch.zeros((1,), dtype=torch.long),
        target_segment_counts=torch.zeros((1,), dtype=torch.long),
        target_segment_starts_zyx=torch.zeros((0, 3), dtype=torch.float32),
        target_segment_ends_zyx=torch.zeros((0, 3), dtype=torch.float32),
        target_segment_bbox_lo_zyx=torch.zeros((0, 3), dtype=torch.long),
        target_segment_bbox_hi_zyx=torch.zeros((0, 3), dtype=torch.long),
        target_tangent_zyx=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        presence_target=target,
        presence_mask=torch.ones((1, 1, patch, patch, patch), dtype=torch.bool),
    )
    output = torch.full((1, 7, patch, patch, patch), 0.5, dtype=torch.float32)

    sheet = _make_train_sample_3d_sheet(batch, output)
    target_panel = sheet[:patch, patch + 4 : patch * 2 + 4]
    assert target_panel[5, 4, 0] == 255
    assert target_panel[5, 4, 1] == 255
    assert target_panel[5, 4, 2] == 255


def test_affine_only_forward_map_matches_matrix_transform() -> None:
    loader = _loader(augment_enabled=True)
    record, _record_index, cp_index = loader.descriptor_for_sample_index(0)
    params = loader._sample_augment_params(record, cp_index, 0)
    map_start, map_end = loader._source_bbox_for_maps(record, params)
    geometry = loader._build_geometry_maps(
        params,
        forward_start_zyx=map_start,
        forward_end_zyx=map_end,
        device=torch.device("cpu"),
    )
    points = params.cp_volume_zyx.reshape(1, 3) + np.asarray(
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    mapped, valid = loader._source_points_to_output_np(points, geometry)
    expected = params.cp_local_zyx.reshape(1, 3) + (
        points - params.cp_volume_zyx.reshape(1, 3)
    ) @ params.source_to_output_zyx.T
    assert bool(valid.all())
    assert np.allclose(mapped, expected, atol=1.0e-4)


def test_smooth_displacement_modes_are_deterministic_and_finite() -> None:
    for mode in ("1d", "2d", "3d"):
        loader = _loader_with_mapping(
            {
                "augment_smooth_displacement_mode": mode,
                "augment_smooth_displacement_amplitude_zyx": [1.0, 0.75, 0.5],
                "augment_smooth_displacement_control_spacing_zyx": [8.0, 8.0, 8.0],
                "augment_smooth_displacement_probability": 1.0,
            }
        )
        batch_a = _load_materialized_batch(loader, 2)
        batch_b = _load_materialized_batch(loader, 2)
        assert torch.isfinite(batch_a.volume).all()
        assert torch.isfinite(batch_a.direction_target_sparse).all()
        assert torch.allclose(batch_a.volume, batch_b.volume)
        assert torch.equal(batch_a.direction_indices_bzyx, batch_b.direction_indices_bzyx)
        assert torch.allclose(batch_a.direction_target_sparse, batch_b.direction_target_sparse)
        assert int(batch_a.direction_indices_bzyx.shape[0]) > 0


def test_anisotropic_blur_spreads_along_configured_axis() -> None:
    volume = torch.zeros((9, 9, 9), dtype=torch.float32)
    volume[4, 4, 4] = 1.0
    blurred = _anisotropic_blur_3d(
        volume,
        axis_zyx=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        sigma_along=1.5,
        sigma_across=0.0,
    )
    assert blurred[3, 4, 4] > 0.0
    assert blurred[5, 4, 4] > 0.0
    assert blurred[4, 4, 3] == 0.0


def test_nonzero_unsupported_3d_shear_and_ringing_raise(tmp_path) -> None:
    config_path = tmp_path / "bad.json"
    raw = {
        "datasets": [{"base_volume_path": "synthetic", "lasagna_manifest_path": "synthetic"}],
        "augment_shear_x": 1.0,
    }
    config_path.write_text(__import__("json").dumps(raw), encoding="utf-8")
    try:
        load_config(config_path)
    except ValueError as exc:
        assert "augment_shear_x" in str(exc)
    else:
        raise AssertionError("non-zero augment_shear_x should fail")


def test_local_prefetch_has_no_remote_chunks() -> None:
    loader = _loader(augment_enabled=False)
    summary = loader.prefetch(0, 2)
    assert summary["chunks"] == 0
    assert summary["errors"] == 0


def test_3d_training_sample_index_limit_validation() -> None:
    assert _training_sample_index_limit({}, 10) == 0
    assert _training_sample_index_limit({"max_sample_index": 4}, 10) == 4
    try:
        _training_sample_index_limit({"max_sample_index": -1}, 10)
    except ValueError as exc:
        assert "max_sample_index" in str(exc)
    else:
        raise AssertionError("negative max_sample_index should fail")
    try:
        _training_sample_index_limit({"max_sample_index": 11}, 10)
    except ValueError as exc:
        assert "<= configured sample count" in str(exc)
    else:
        raise AssertionError("too-large max_sample_index should fail")


def test_3d_prefetch_step_count_resolution_matches_2d_sentinels() -> None:
    assert (
        _resolve_prefetch_sample_count(
            training={"max_steps": 3},
            loader_sample_count=100,
            batch_size=8,
            prefetch_steps=None,
        )
        == 24
    )
    assert (
        _resolve_prefetch_sample_count(
            training={"max_steps": 3},
            loader_sample_count=100,
            batch_size=8,
            prefetch_steps=2,
        )
        == 16
    )
    assert (
        _resolve_prefetch_sample_count(
            training={"max_steps": 3, "max_sample_index": 12},
            loader_sample_count=100,
            batch_size=8,
            prefetch_steps=2,
        )
        == 12
    )
    assert (
        _resolve_prefetch_sample_count(
            training={"max_steps": 0},
            loader_sample_count=100,
            batch_size=8,
            prefetch_steps=None,
        )
        == 100
    )
    assert (
        _resolve_prefetch_sample_count(
            training={"max_steps": 3, "max_sample_index": 12},
            loader_sample_count=100,
            batch_size=8,
            prefetch_steps=0,
        )
        == 12
    )
    try:
        _resolve_prefetch_sample_count(
            training={"max_steps": 3},
            loader_sample_count=100,
            batch_size=8,
            prefetch_steps=-1,
        )
    except ValueError as exc:
        assert "--prefetch-steps" in str(exc)
    else:
        raise AssertionError("negative prefetch steps should fail")


def test_3d_test_loader_raw_config_disables_augmentation_by_default() -> None:
    raw = {
        "datasets": [{"name": "train"}],
        "test_datasets": [{"name": "test"}],
        "augment_enabled": True,
    }

    default_test = _make_test_loader_raw_config(raw, {})
    augmented_test = _make_test_loader_raw_config(raw, {"test_augment_enabled": True})

    assert default_test["datasets"] == [{"name": "test"}]
    assert default_test["augment_enabled"] is False
    assert "test_datasets" not in default_test
    assert augmented_test["augment_enabled"] is True


def test_3d_dense_test_control_points_zero_uses_random_full_set() -> None:
    assert _resolve_dense_test_selection(
        {"test_control_points": 0, "test_start_sample_index": 99},
        loader_sample_count=17,
        default_count=4,
    ) == (17, 0, "random")
    assert _resolve_dense_test_selection(
        {"test_start_sample_index": 99},
        loader_sample_count=17,
        default_count=0,
    ) == (17, 0, "random")
    assert _resolve_dense_test_selection(
        {"test_control_points": 5, "test_start_sample_index": 9},
        loader_sample_count=17,
        default_count=4,
    ) == (5, 9, "random")


def test_train_sample_3d_sheet_has_branch_presence_columns_and_line_overlay() -> None:
    loader = _loader(augment_enabled=False)
    batch = _load_materialized_batch(loader, 0)
    output = torch.zeros(
        (
            int(batch.volume.shape[0]),
            7,
            *tuple(int(v) for v in batch.volume.shape[-3:]),
        ),
        dtype=batch.volume.dtype,
    )
    sheet = _make_train_sample_3d_sheet(batch, output)

    patch = int(batch.volume.shape[-1])
    gap = 4
    assert sheet.shape[1] == patch * 7 + 6 * gap
    first_image_panel = sheet[:patch, :patch]
    colored = (
        (first_image_panel[..., 0] != first_image_panel[..., 1])
        | (first_image_panel[..., 1] != first_image_panel[..., 2])
    )
    assert bool(np.any(colored))


def test_branch_presence_view_selects_output_closer_to_slice_normal() -> None:
    z_dir = encode_lasagna_direction_3x2(
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    )[0]
    x_dir = encode_lasagna_direction_3x2(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    )[0]
    sampled = torch.zeros((2, 3, 14), dtype=torch.float32)
    sampled[..., 0:6] = z_dir
    sampled[..., 6] = 0.25
    sampled[..., 7:13] = x_dir
    sampled[..., 13] = 0.75

    close, other, max_p, min_p, avg_p = _branch_presence_views_from_sampled_output(
        sampled,
        normal_zyx=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )

    assert np.allclose(close, 0.25)
    assert np.allclose(other, 0.75)
    assert np.allclose(max_p, 0.75)
    assert np.allclose(min_p, 0.25)
    assert np.allclose(avg_p, 0.5)


def test_oblique_line_presence_uses_slice_frame_axes() -> None:
    center = np.asarray([8.0, 8.0, 8.0], dtype=np.float32)
    segment_start = np.asarray([[8.0, 8.0, 0.0]], dtype=np.float32)
    segment_end = np.asarray([[8.0, 8.0, 16.0]], dtype=np.float32)

    cross = _oblique_line_presence_for_display(
        height=17,
        width=17,
        segments_start_zyx=segment_start,
        segments_end_zyx=segment_end,
        center_zyx=center,
        row_axis_zyx=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        col_axis_zyx=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        normal_zyx=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        threshold_voxels=1.0,
    )
    tangent = _oblique_line_presence_for_display(
        height=17,
        width=17,
        segments_start_zyx=segment_start,
        segments_end_zyx=segment_end,
        center_zyx=center,
        row_axis_zyx=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        col_axis_zyx=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        normal_zyx=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        threshold_voxels=1.0,
    )

    cross_cols = np.nonzero(cross > 0.0)[1]
    tangent_cols = np.nonzero(tangent > 0.0)[1]
    assert int(cross_cols.max() - cross_cols.min()) <= 4
    assert int(tangent_cols.max() - tangent_cols.min()) >= 14


def test_panel_line_aa_draws_fractional_thin_line() -> None:
    panel = np.zeros((16, 16, 3), dtype=np.uint8)

    _draw_panel_line_aa(
        panel,
        np.asarray([2.25, 1.5], dtype=np.float32),
        np.asarray([13.25, 10.5], dtype=np.float32),
        (255, 80, 0),
    )

    red = panel[..., 0]
    assert int(np.count_nonzero(red)) > 0
    assert bool(np.any((red > 0) & (red < 255)))
    assert int(np.count_nonzero(red)) < 16 * 4


def test_projected_cp_direction_shortens_out_of_slice_direction() -> None:
    full_panel = np.zeros((40, 40, 3), dtype=np.uint8)
    shortened_panel = np.zeros((40, 40, 3), dtype=np.uint8)

    _draw_projected_cp_direction(
        full_panel,
        cp_row=20,
        cp_col=20,
        direction_zyx=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        row_axis=1,
        col_axis=2,
    )
    _draw_projected_cp_direction(
        shortened_panel,
        cp_row=20,
        cp_col=20,
        direction_zyx=np.asarray([np.sqrt(3.0), 0.0, 1.0], dtype=np.float32),
        row_axis=1,
        col_axis=2,
    )

    full_cols = np.nonzero(full_panel[..., 0] > 0)[1]
    shortened_cols = np.nonzero(shortened_panel[..., 0] > 0)[1]
    full_extent = int(full_cols.max() - full_cols.min() + 1)
    shortened_extent = int(shortened_cols.max() - shortened_cols.min() + 1)
    assert shortened_extent < full_extent
    assert shortened_extent <= int(round(full_extent * 0.75))


def test_write_3d_sample_sheet_adds_image() -> None:
    class _ZeroModel(torch.nn.Module):
        def forward(self, volume: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                (int(volume.shape[0]), 7, *tuple(int(v) for v in volume.shape[-3:])),
                dtype=volume.dtype,
                device=volume.device,
            )

    class _Writer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, np.ndarray, int, str]] = []

        def add_image(self, tag: str, image: np.ndarray, step: int, *, dataformats: str) -> None:
            self.calls.append((tag, image, step, dataformats))

    loader = _loader(augment_enabled=False)
    batch = _load_materialized_batch(loader, 0)
    writer = _Writer()

    _write_3d_sample_sheet(writer, "test_sample_3d/principal_slices", _ZeroModel(), batch, 7)

    assert len(writer.calls) == 1
    tag, image, step, dataformats = writer.calls[0]
    assert tag == "test_sample_3d/principal_slices"
    assert step == 7
    assert dataformats == "HWC"
    assert image.ndim == 3


def test_train_sample_3d_contact_sheet_stacks_multiple_samples() -> None:
    loader = _loader(augment_enabled=False)
    batch = _load_materialized_batch(loader, 0)
    output = torch.zeros(
        (
            int(batch.volume.shape[0]),
            7,
            *tuple(int(v) for v in batch.volume.shape[-3:]),
        ),
        dtype=batch.volume.dtype,
    )

    single = _make_train_sample_3d_contact_sheet(batch, output, sample_count=1)
    multi = _make_train_sample_3d_contact_sheet(batch, output, sample_count=2)

    assert multi.shape[0] == single.shape[0]
    assert multi.shape[1] > single.shape[1]


def test_3d_prefetch_streams_producers_but_downloads_in_sample_order(
    tmp_path,
    monkeypatch,
) -> None:
    loader = FiberTrace3DLoader.__new__(FiberTrace3DLoader)
    loader.config = SimpleNamespace(
        prefetch_workers=1,
        prefetch_sampler_workers=2,
        volume_cache_retry_seconds=0.0,
    )

    store = object()

    def request(key: str) -> ZarrChunkRequest:
        return ZarrChunkRequest(
            store=store,
            store_identity="fake-store",
            key=key,
            cache_path=tmp_path / f"{key}.bin",
            empty_path=tmp_path / f"{key}.empty",
            remote_url=f"https://example.invalid/{key}",
            remote_chunk_key=key,
            cache_payload_format="source_bytes",
        )

    samples = {
        0: [request("a")],
        1: [request("b")],
        2: [request("b"), request("c")],
    }

    def fake_requests(
        sample_index: int,
        *,
        sample_mode: str = "random",
        sample_index_limit: int | None = None,
    ) -> tuple[int, list[ZarrChunkRequest]]:
        del sample_mode, sample_index_limit
        if int(sample_index) == 0:
            time.sleep(0.05)
        return 8, list(samples[int(sample_index)])

    loader._chunk_requests_and_valid_voxels_for_sample_index = fake_requests
    download_order: list[str] = []

    def fake_download(request: ZarrChunkRequest, *, retry_seconds: float) -> dict[str, object]:
        del retry_seconds
        download_order.append(request.key)
        return {"status": "downloaded", "bytes": 4}

    monkeypatch.setattr(loader3d_module, "_download_prefetch_request", fake_download)
    summary = loader.prefetch(0, 3)

    assert download_order == ["a", "b", "c"]
    assert summary["samples"] == 3
    assert summary["chunks"] == 3
    assert summary["downloaded"] == 3
    assert summary["download_done"] == 3
    assert summary["max_exclusive_sample_index"] == 3
    assert summary["valid_voxels"] == 24


def test_3d_prefetch_classifies_cache_hit_empty_and_skip(tmp_path, monkeypatch) -> None:
    loader = FiberTrace3DLoader.__new__(FiberTrace3DLoader)
    loader.config = SimpleNamespace(
        prefetch_workers=2,
        prefetch_sampler_workers=2,
        volume_cache_retry_seconds=0.0,
    )

    store = object()
    hit_path = tmp_path / "hit.bin"
    hit_path.write_bytes(b"cached")
    empty_path = tmp_path / "missing.empty"
    empty_path.write_bytes(b"")

    def request(key: str, *, cache_path: object | None, empty_path_value: object | None) -> ZarrChunkRequest:
        return ZarrChunkRequest(
            store=store,
            store_identity="fake-store",
            key=key,
            cache_path=None if cache_path is None else Path(cache_path),
            empty_path=None if empty_path_value is None else Path(empty_path_value),
            remote_url=f"https://example.invalid/{key}",
            remote_chunk_key=key,
            cache_payload_format="source_bytes",
        )

    def fake_requests(
        sample_index: int,
        *,
        sample_mode: str = "random",
        sample_index_limit: int | None = None,
    ) -> tuple[int, list[ZarrChunkRequest]]:
        del sample_mode, sample_index_limit
        if int(sample_index) == 1:
            raise ValueError("invalid synthetic sample")
        if int(sample_index) == 0:
            return 7, [
                request("hit", cache_path=hit_path, empty_path_value=tmp_path / "hit.empty"),
                request("missing", cache_path=tmp_path / "missing.bin", empty_path_value=empty_path),
            ]
        return 5, [
            request("download", cache_path=tmp_path / "download.bin", empty_path_value=tmp_path / "download.empty")
        ]

    loader._chunk_requests_and_valid_voxels_for_sample_index = fake_requests
    downloaded_keys: list[str] = []

    def fake_download(request: ZarrChunkRequest, *, retry_seconds: float) -> dict[str, object]:
        del retry_seconds
        downloaded_keys.append(request.key)
        return {"status": "downloaded", "bytes": 3}

    monkeypatch.setattr(loader3d_module, "_download_prefetch_request", fake_download)
    summary = loader.prefetch(0, 3)

    assert downloaded_keys == ["download"]
    assert summary["samples"] == 3
    assert summary["skipped_samples"] == 1
    assert summary["chunks"] == 3
    assert summary["cache_hits"] == 1
    assert summary["known_missing"] == 1
    assert summary["downloaded"] == 1
    assert summary["errors"] == 0
    assert summary["max_exclusive_sample_index"] == 3
    assert "invalid synthetic sample" in summary["first_sample_skip"]


def test_vc3d_dependency_prefetch_flattens_3d_coords_like_training_sampling() -> None:
    class _FakeVolume:
        remote_url = "s3://example/volume.zarr"

        def __init__(self) -> None:
            self.calls: list[tuple[int, ...]] = []

        def collect_coords_dependencies(
            self,
            coords_xyz: np.ndarray,
            valid_mask: np.ndarray,
            level: int,
            sampling: str,
            workers: int,
        ) -> list[dict[str, object]]:
            if coords_xyz.ndim != 3 or coords_xyz.shape[-1] != 3:
                raise ValueError("coords_xyz must have shape [H, W, 3]")
            if valid_mask.shape != coords_xyz.shape[:2]:
                raise ValueError("valid_mask must have shape [H, W]")
            call_index = len(self.calls)
            self.calls.append(tuple(int(v) for v in coords_xyz.shape))
            assert level == 2
            assert sampling == "trilinear"
            assert workers == 32
            return [
                {
                    "key": f"2/0/0/{call_index}",
                    "cache_path": f"/tmp/cache/{call_index}.bin",
                    "empty_path": f"/tmp/cache/{call_index}.empty",
                    "remote_url": "https://example.invalid/volume.zarr",
                    "remote_chunk_key": f"2/0/0/{call_index}",
                    "cache_payload_format": "direct",
                    "persistent_extension": ".bin",
                },
                {
                    "key": "2/shared",
                    "cache_path": "/tmp/cache/shared.bin",
                    "empty_path": "/tmp/cache/shared.empty",
                    "remote_url": "https://example.invalid/volume.zarr",
                    "remote_chunk_key": "2/shared",
                    "cache_payload_format": "direct",
                    "persistent_extension": ".bin",
                },
            ]

    fake = _FakeVolume()
    sampler = Vc3dCoordinateSampler.__new__(Vc3dCoordinateSampler)
    sampler.volume = fake
    sampler.level = 2
    sampler.sampling = "trilinear"
    sampler.blocking = True

    coords = np.zeros((3, 4, 5, 3), dtype=np.float32)
    valid = np.ones((3, 4, 5), dtype=bool)
    valid[1, :, :] = False
    requests = sampler.chunk_requests_for_coords(coords, valid)

    assert fake.calls == [(12, 5, 3)]
    assert [request.key for request in requests] == ["2/0/0/0", "2/shared"]
    assert requests[0].cache_path is not None


def test_3d_model_and_losses_smoke() -> None:
    loader = _loader(augment_enabled=False)
    batch = _load_materialized_batch(loader, 0)
    model = FiberTrace3DNet(
        FiberTrace3DModelConfig(
            input_channels=1,
            output_channels=7,
            features_per_stage=(4, 8),
            strides=((1, 1, 1), (2, 2, 2)),
            decoder_upsample_mode="pixelshuffle",
        )
    )
    output = model(batch.volume)
    assert direction_output(output).shape == (2, 6, 16, 16, 16)
    assert presence_output(output).shape == (2, 1, 16, 16, 16)
    losses = compute_losses(output, batch, direction_weight=1.0, presence_weight=1.0)
    assert losses["total"].ndim == 0
    assert torch.isfinite(losses["total"])
    assert losses["angle_mean_deg"].ndim == 0
    assert torch.isfinite(losses["angle_mean_deg"])


def test_3d_model_branch_helpers_preserve_branch_zero_layout() -> None:
    model = FiberTrace3DNet(
        FiberTrace3DModelConfig(
            input_channels=1,
            direction_branch_count=2,
            output_channels=14,
            features_per_stage=(4, 8),
            strides=((1, 1, 1), (2, 2, 2)),
            decoder_upsample_mode="pixelshuffle",
        )
    )
    with torch.no_grad():
        model_output = model(torch.zeros((2, 1, 8, 8, 8), dtype=torch.float32))
    assert direction_outputs(model_output).shape == (2, 2, 6, 8, 8, 8)
    assert presence_outputs(model_output).shape == (2, 2, 1, 8, 8, 8)

    output = torch.zeros((2, 14, 3, 4, 5), dtype=torch.float32)
    output[:, 0:6] = 0.1
    output[:, 6:7] = 0.2
    output[:, 7:13] = 0.3
    output[:, 13:14] = 0.4

    dirs = direction_outputs(output)
    presences = presence_outputs(output)

    assert dirs.shape == (2, 2, 6, 3, 4, 5)
    assert presences.shape == (2, 2, 1, 3, 4, 5)
    assert torch.allclose(direction_output(output), dirs[:, 0])
    assert torch.allclose(presence_output(output), presences[:, 0])
    assert torch.allclose(dirs[:, 1], torch.full_like(dirs[:, 1], 0.3))
    assert torch.allclose(presences[:, 1], torch.full_like(presences[:, 1], 0.4))


def test_3d_two_branch_positive_supervision_routes_to_best_branch() -> None:
    target_dir = encode_lasagna_direction_3x2(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    )[0]
    bad_dir = encode_lasagna_direction_3x2(
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    )[0]
    better_dir = encode_lasagna_direction_3x2(
        torch.tensor([[0.9, 0.2, 0.0]], dtype=torch.float32)
    )[0]
    output = torch.zeros((1, 14, 1, 1, 2), dtype=torch.float32)
    output[0, 0:6, 0, 0, 0] = bad_dir
    output[0, 7:13, 0, 0, 0] = better_dir
    output[0, 6, 0, 0, 0] = 0.95
    output[0, 13, 0, 0, 0] = 0.55
    output[0, 6, 0, 0, 1] = 0.40
    output[0, 13, 0, 0, 1] = 0.30
    output.requires_grad_()
    batch = FiberTrace3DBatch(
        volume=torch.zeros((1, 1, 1, 1, 2), dtype=torch.float32),
        valid_mask=torch.ones((1, 1, 1, 1, 2), dtype=torch.bool),
        cp_local_zyx=torch.zeros((1, 3), dtype=torch.float32),
        crop_origin_zyx=torch.zeros((1, 3), dtype=torch.float32),
        sample_indices=torch.zeros((1,), dtype=torch.long),
        record_indices=torch.zeros((1,), dtype=torch.long),
        control_point_indices=torch.zeros((1,), dtype=torch.long),
        fiber_paths=("synthetic.json",),
        target_modes=torch.zeros((1,), dtype=torch.long),
        target_segment_offsets=torch.zeros((1,), dtype=torch.long),
        target_segment_counts=torch.zeros((1,), dtype=torch.long),
        target_segment_starts_zyx=torch.zeros((0, 3), dtype=torch.float32),
        target_segment_ends_zyx=torch.zeros((0, 3), dtype=torch.float32),
        target_segment_bbox_lo_zyx=torch.zeros((0, 3), dtype=torch.long),
        target_segment_bbox_hi_zyx=torch.zeros((0, 3), dtype=torch.long),
        target_tangent_zyx=torch.zeros((1, 3), dtype=torch.float32),
        direction_indices_bzyx=torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
        direction_target_sparse=target_dir.view(1, 6),
        direction_weight_sparse=torch.ones((1, 6), dtype=torch.float32),
        presence_target=torch.tensor([1.0, 0.0], dtype=torch.float32).view(1, 1, 1, 1, 2),
        presence_mask=torch.ones((1, 1, 1, 1, 2), dtype=torch.bool),
    )

    eval_losses = compute_losses(output, batch, direction_weight=1.0, presence_weight=1.0)
    assert torch.allclose(eval_losses["branch0_fraction"], torch.tensor(0.0))
    assert torch.allclose(eval_losses["branch1_fraction"], torch.tensor(1.0))

    losses = compute_losses(
        output,
        batch,
        direction_weight=1.0,
        presence_weight=1.0,
        enforce_branch_min_fraction=True,
    )
    losses["total"].backward()

    assert torch.allclose(losses["branch0_fraction"], torch.tensor(0.0))
    assert torch.allclose(losses["branch1_fraction"], torch.tensor(1.0))
    assert torch.count_nonzero(output.grad[0, 0:6, 0, 0, 0]) == 0
    assert torch.count_nonzero(output.grad[0, 7:13, 0, 0, 0]) > 0
    assert output.grad[0, 6, 0, 0, 0] == 0.0
    assert output.grad[0, 13, 0, 0, 0] != 0.0
    assert output.grad[0, 6, 0, 0, 1] != 0.0
    assert output.grad[0, 13, 0, 0, 1] != 0.0


def test_3d_two_branch_positive_supervision_enforces_minority_floor() -> None:
    target_dir = encode_lasagna_direction_3x2(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    )[0]
    weaker_dir = encode_lasagna_direction_3x2(
        torch.tensor([[0.95, 0.25, 0.0]], dtype=torch.float32)
    )[0]
    output = torch.zeros((1, 14, 1, 1, 12), dtype=torch.float32)
    output[0, 0:6, 0, 0, :] = target_dir.view(6, 1)
    output[0, 7:13, 0, 0, :] = weaker_dir.view(6, 1)
    output[0, 6, 0, 0, :] = 0.95
    output[0, 13, 0, 0, :] = torch.tensor(
        [0.10, 0.10, 0.10, 0.10, 0.70, 0.60, 0.50, 0.40, 0.80, 0.10, 0.10, 0.10],
        dtype=torch.float32,
    )
    output.requires_grad_()
    batch = FiberTrace3DBatch(
        volume=torch.zeros((1, 1, 1, 1, 12), dtype=torch.float32),
        valid_mask=torch.ones((1, 1, 1, 1, 12), dtype=torch.bool),
        cp_local_zyx=torch.zeros((1, 3), dtype=torch.float32),
        crop_origin_zyx=torch.zeros((1, 3), dtype=torch.float32),
        sample_indices=torch.zeros((1,), dtype=torch.long),
        record_indices=torch.zeros((1,), dtype=torch.long),
        control_point_indices=torch.zeros((1,), dtype=torch.long),
        fiber_paths=("synthetic.json",),
        target_modes=torch.zeros((1,), dtype=torch.long),
        target_segment_offsets=torch.zeros((1,), dtype=torch.long),
        target_segment_counts=torch.zeros((1,), dtype=torch.long),
        target_segment_starts_zyx=torch.zeros((0, 3), dtype=torch.float32),
        target_segment_ends_zyx=torch.zeros((0, 3), dtype=torch.float32),
        target_segment_bbox_lo_zyx=torch.zeros((0, 3), dtype=torch.long),
        target_segment_bbox_hi_zyx=torch.zeros((0, 3), dtype=torch.long),
        target_tangent_zyx=torch.zeros((1, 3), dtype=torch.float32),
        direction_indices_bzyx=torch.stack(
            [
                torch.zeros((12,), dtype=torch.long),
                torch.zeros((12,), dtype=torch.long),
                torch.zeros((12,), dtype=torch.long),
                torch.arange(12, dtype=torch.long),
            ],
            dim=1,
        ),
        direction_target_sparse=target_dir.view(1, 6).expand(12, 6),
        direction_weight_sparse=torch.ones((12, 6), dtype=torch.float32),
        presence_target=torch.ones((1, 1, 1, 1, 12), dtype=torch.float32),
        presence_mask=torch.ones((1, 1, 1, 1, 12), dtype=torch.bool),
    )

    eval_losses = compute_losses(output, batch, direction_weight=1.0, presence_weight=1.0)
    assert torch.allclose(eval_losses["branch0_fraction"], torch.tensor(1.0))
    assert torch.allclose(eval_losses["branch1_fraction"], torch.tensor(0.0))

    losses = compute_losses(
        output,
        batch,
        direction_weight=1.0,
        presence_weight=1.0,
        enforce_branch_min_fraction=True,
    )
    losses["total"].backward()

    assert torch.allclose(losses["branch0_fraction"], torch.tensor(8.0 / 12.0))
    assert torch.allclose(losses["branch1_fraction"], torch.tensor(4.0 / 12.0))
    for x in range(4, 8):
        assert torch.count_nonzero(output.grad[0, 7:13, 0, 0, x]) > 0
        assert output.grad[0, 13, 0, 0, x] != 0.0
        assert torch.count_nonzero(output.grad[0, 0:6, 0, 0, x]) == 0
        assert output.grad[0, 6, 0, 0, x] == 0.0
    assert torch.count_nonzero(output.grad[0, 7:13, 0, 0, 8]) == 0
    assert output.grad[0, 13, 0, 0, 8] == 0.0
    assert torch.count_nonzero(output.grad[0, 0:6, 0, 0, 8]) == 0
    assert output.grad[0, 6, 0, 0, 8] != 0.0


def test_3d_positive_presence_ignores_masked_sparse_points() -> None:
    target_dir = encode_lasagna_direction_3x2(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    )[0]
    output = torch.zeros((1, 14, 1, 1, 2), dtype=torch.float32)
    output[0, 0:6, 0, 0, 0] = target_dir
    output[0, 7:13, 0, 0, 0] = target_dir
    output[0, 6, 0, 0, :] = torch.tensor([0.8, 0.4], dtype=torch.float32)
    output[0, 13, 0, 0, :] = torch.tensor([0.9, 0.3], dtype=torch.float32)
    output.requires_grad_()
    batch = FiberTrace3DBatch(
        volume=torch.zeros((1, 1, 1, 1, 2), dtype=torch.float32),
        valid_mask=torch.ones((1, 1, 1, 1, 2), dtype=torch.bool),
        cp_local_zyx=torch.zeros((1, 3), dtype=torch.float32),
        crop_origin_zyx=torch.zeros((1, 3), dtype=torch.float32),
        sample_indices=torch.zeros((1,), dtype=torch.long),
        record_indices=torch.zeros((1,), dtype=torch.long),
        control_point_indices=torch.zeros((1,), dtype=torch.long),
        fiber_paths=("synthetic.json",),
        target_modes=torch.zeros((1,), dtype=torch.long),
        target_segment_offsets=torch.zeros((1,), dtype=torch.long),
        target_segment_counts=torch.zeros((1,), dtype=torch.long),
        target_segment_starts_zyx=torch.zeros((0, 3), dtype=torch.float32),
        target_segment_ends_zyx=torch.zeros((0, 3), dtype=torch.float32),
        target_segment_bbox_lo_zyx=torch.zeros((0, 3), dtype=torch.long),
        target_segment_bbox_hi_zyx=torch.zeros((0, 3), dtype=torch.long),
        target_tangent_zyx=torch.zeros((1, 3), dtype=torch.float32),
        direction_indices_bzyx=torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
        direction_target_sparse=target_dir.view(1, 6),
        direction_weight_sparse=torch.ones((1, 6), dtype=torch.float32),
        presence_target=torch.tensor([1.0, 0.0], dtype=torch.float32).view(1, 1, 1, 1, 2),
        presence_mask=torch.tensor([False, True], dtype=torch.bool).view(1, 1, 1, 1, 2),
    )

    losses = compute_losses(output, batch, direction_weight=0.0, presence_weight=1.0)
    losses["total"].backward()

    assert output.grad[0, 6, 0, 0, 0] == 0.0
    assert output.grad[0, 13, 0, 0, 0] == 0.0
    assert output.grad[0, 6, 0, 0, 1] != 0.0
    assert output.grad[0, 13, 0, 0, 1] != 0.0


def test_3d_fiber_model_uses_batchnorm_by_default() -> None:
    model = FiberTrace3DNet(
        FiberTrace3DModelConfig(
            input_channels=1,
            output_channels=7,
            features_per_stage=(4, 8),
            strides=((1, 1, 1), (2, 2, 2)),
            decoder_upsample_mode="pixelshuffle",
        )
    )
    assert any(isinstance(module, torch.nn.BatchNorm3d) for module in model.modules())
    assert not any(
        isinstance(module, (torch.nn.GroupNorm, torch.nn.InstanceNorm3d))
        for module in model.modules()
    )


def test_3d_fiber_model_can_disable_normalization_explicitly() -> None:
    model = FiberTrace3DNet(
        FiberTrace3DModelConfig(
            input_channels=1,
            output_channels=7,
            features_per_stage=(4, 8),
            strides=((1, 1, 1), (2, 2, 2)),
            decoder_upsample_mode="pixelshuffle",
            normalization="none",
        )
    )
    assert not any(
        isinstance(
            module,
            (torch.nn.BatchNorm3d, torch.nn.GroupNorm, torch.nn.InstanceNorm3d),
        )
        for module in model.modules()
    )


def test_3d_presence_loss_balances_positive_and_negative_regions() -> None:
    presence_values = torch.tensor([0.9, 0.2, 0.8, 0.4], dtype=torch.float32)
    output = torch.zeros((1, 7, 1, 1, 4), dtype=torch.float32)
    output[0, 0:6, 0, 0, 0] = encode_lasagna_direction_3x2(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    )[0]
    output[0, 6, 0, 0, :] = presence_values
    presence_target = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32).view(1, 1, 1, 1, 4)
    batch = FiberTrace3DBatch(
        volume=torch.zeros((1, 1, 1, 1, 4), dtype=torch.float32),
        valid_mask=torch.ones((1, 1, 1, 1, 4), dtype=torch.bool),
        cp_local_zyx=torch.zeros((1, 3), dtype=torch.float32),
        crop_origin_zyx=torch.zeros((1, 3), dtype=torch.float32),
        sample_indices=torch.zeros((1,), dtype=torch.long),
        record_indices=torch.zeros((1,), dtype=torch.long),
        control_point_indices=torch.zeros((1,), dtype=torch.long),
        fiber_paths=("synthetic.json",),
        target_modes=torch.zeros((1,), dtype=torch.long),
        target_segment_offsets=torch.zeros((1,), dtype=torch.long),
        target_segment_counts=torch.zeros((1,), dtype=torch.long),
        target_segment_starts_zyx=torch.zeros((0, 3), dtype=torch.float32),
        target_segment_ends_zyx=torch.zeros((0, 3), dtype=torch.float32),
        target_segment_bbox_lo_zyx=torch.zeros((0, 3), dtype=torch.long),
        target_segment_bbox_hi_zyx=torch.zeros((0, 3), dtype=torch.long),
        target_tangent_zyx=torch.zeros((1, 3), dtype=torch.float32),
        direction_indices_bzyx=torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
        direction_target_sparse=encode_lasagna_direction_3x2(
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        ),
        direction_weight_sparse=torch.zeros((1, 6), dtype=torch.float32),
        presence_target=presence_target,
        presence_mask=torch.ones((1, 1, 1, 1, 4), dtype=torch.bool),
    )

    losses = compute_losses(output, batch, direction_weight=0.0, presence_weight=1.0)
    pos_expected = -torch.log(torch.tensor(0.9))
    neg_expected = -torch.log(1.0 - torch.tensor([0.2, 0.8, 0.4])).mean()
    expected = pos_expected + neg_expected
    assert torch.allclose(losses["presence"], expected)


def test_3d_negative_presence_loss_normalizes_per_patch_and_branch() -> None:
    output = torch.zeros((2, 14, 1, 1, 3), dtype=torch.float32)
    output[0, 6, 0, 0, :] = torch.tensor([0.2, 0.8, 0.4], dtype=torch.float32)
    output[0, 13, 0, 0, :] = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
    output[1, 6, 0, 0, :] = torch.tensor([0.7, 0.0, 0.0], dtype=torch.float32)
    output[1, 13, 0, 0, :] = torch.tensor([0.6, 0.0, 0.0], dtype=torch.float32)
    presence_mask = torch.tensor(
        [
            [True, True, True],
            [True, False, False],
        ],
        dtype=torch.bool,
    ).view(2, 1, 1, 1, 3)
    batch = FiberTrace3DBatch(
        volume=torch.zeros((2, 1, 1, 1, 3), dtype=torch.float32),
        valid_mask=torch.ones((2, 1, 1, 1, 3), dtype=torch.bool),
        cp_local_zyx=torch.zeros((2, 3), dtype=torch.float32),
        crop_origin_zyx=torch.zeros((2, 3), dtype=torch.float32),
        sample_indices=torch.zeros((2,), dtype=torch.long),
        record_indices=torch.zeros((2,), dtype=torch.long),
        control_point_indices=torch.zeros((2,), dtype=torch.long),
        fiber_paths=("synthetic0.json", "synthetic1.json"),
        target_modes=torch.zeros((2,), dtype=torch.long),
        target_segment_offsets=torch.zeros((2,), dtype=torch.long),
        target_segment_counts=torch.zeros((2,), dtype=torch.long),
        target_segment_starts_zyx=torch.zeros((0, 3), dtype=torch.float32),
        target_segment_ends_zyx=torch.zeros((0, 3), dtype=torch.float32),
        target_segment_bbox_lo_zyx=torch.zeros((0, 3), dtype=torch.long),
        target_segment_bbox_hi_zyx=torch.zeros((0, 3), dtype=torch.long),
        target_tangent_zyx=torch.zeros((2, 3), dtype=torch.float32),
        direction_indices_bzyx=torch.zeros((0, 4), dtype=torch.long),
        direction_target_sparse=torch.zeros((0, 6), dtype=torch.float32),
        direction_weight_sparse=torch.zeros((0, 6), dtype=torch.float32),
        presence_target=torch.zeros((2, 1, 1, 1, 3), dtype=torch.float32),
        presence_mask=presence_mask,
    )

    losses = compute_losses(output, batch, direction_weight=0.0, presence_weight=1.0)
    expected = torch.stack(
        [
            -torch.log(1.0 - torch.tensor([0.2, 0.8, 0.4])).mean(),
            -torch.log(1.0 - torch.tensor([0.1, 0.1, 0.1])).mean(),
            -torch.log(1.0 - torch.tensor(0.7)),
            -torch.log(1.0 - torch.tensor(0.6)),
        ]
    ).mean()
    global_mean = -torch.log(
        1.0 - torch.tensor([0.2, 0.8, 0.4, 0.1, 0.1, 0.1, 0.7, 0.6])
    ).mean()
    assert torch.allclose(losses["presence"], expected)
    assert not torch.allclose(losses["presence"], global_mean)


def test_3d_projection_bridge_recovers_straight_xy_direction() -> None:
    encoded = encode_lasagna_direction_3x2(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    )
    projected = project_3d_direction_to_2d_frame(
        encoded,
        frame_x_xyz=torch.tensor([1.0, 0.0, 0.0]),
        frame_y_xyz=torch.tensor([0.0, 1.0, 0.0]),
    )
    assert torch.abs(projected[0, 0]) > 0.95
    assert torch.abs(projected[0, 1]) < 0.15


def test_trace2cp_bridge_scores_projected_3d_field() -> None:
    encoded = encode_lasagna_direction_3x2(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    )[0]
    output = torch.zeros((7, 3, 8, 12), dtype=torch.float32)
    output[:6] = encoded.view(6, 1, 1, 1)
    output[6] = 1.0
    yy, xx = np.meshgrid(
        np.arange(8, dtype=np.float32),
        np.arange(12, dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([np.ones_like(xx), yy, xx], axis=-1)
    fields = project_3d_output_to_trace2cp_fields(
        output,
        coords,
        np.ones((8, 12), dtype=bool),
        frame_x_xyz=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        frame_y_xyz=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    score = score_trace2cp_projected_fields(
        fields,
        start_xy=np.asarray([1.0, 4.0], dtype=np.float32),
        target_xy=np.asarray([10.0, 4.0], dtype=np.float32),
        step_px=1.0,
        rf_margin_px=0.0,
    )
    assert score.trace2cp_error < 0.05
    assert score.reached_target_columns


def test_3d_trace2cp_fixed_set_evaluator_scores_synthetic_source() -> None:
    encoded = encode_lasagna_direction_3x2(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    )[0]

    class ConstantTrace2CpModel(torch.nn.Module):
        def forward(self, volume: torch.Tensor) -> torch.Tensor:
            b, _c, d, h, w = volume.shape
            out = torch.zeros((b, 7, d, h, w), dtype=torch.float32, device=volume.device)
            out[:, :6] = encoded.to(volume.device).view(1, 6, 1, 1, 1)
            out[:, 6] = 1.0
            return out

    yy, xx = np.meshgrid(
        np.arange(8, dtype=np.float32),
        np.arange(12, dtype=np.float32),
        indexing="ij",
    )
    coords_xyz = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    valid = np.ones((8, 12), dtype=bool)
    grid = SimpleNamespace(
        coords_xyz=torch.as_tensor(coords_xyz, dtype=torch.float32),
        valid_mask=torch.as_tensor(valid, dtype=torch.bool),
        frame=FiberStripFrame(
            tangent_xyz=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            side_xyz=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            mesh_normal_xyz=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        ),
        offset_axis_xyz=torch.as_tensor(
            np.broadcast_to(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), (8, 12, 3)).copy()
        ),
    )
    source = SimpleNamespace(
        grid=grid,
        record=SimpleNamespace(
            volume=np.ones((3, 8, 12), dtype=np.float32),
            volume_spacing_base=1.0,
        ),
        start_control_point_xy=np.asarray([1.0, 4.0], dtype=np.float32),
        target_control_point_xy=np.asarray([10.0, 4.0], dtype=np.float32),
    )

    class FakeGeometryLoader:
        sample_count = 1

        def build_trace2cp_segment_source(self, *_args, **_kwargs):
            return source

    cfg = _Trace2Cp3DConfig(
        enabled=True,
        control_points=1,
        start_sample_index=0,
        sample_mode="flat",
        step_px=1.0,
        rf_margin_px=0.0,
        presence_enabled=True,
        patch_shape_hw=(8, 12),
        strip_z_offset_count=1,
        strip_z_offset_step=1.0,
        tile_shape_hw=(8, 12),
        block_context_voxels=0,
        loader_config_path=None,
    )
    result = _evaluate_trace2cp_metric_fixed_set_3d(
        ConstantTrace2CpModel(),
        FakeGeometryLoader(),
        image_normalization="none",
        cfg=cfg,
        device=torch.device("cpu"),
    )
    assert result.error_mean < 0.05
    assert result.segments == 1
    assert result.skipped_segments == 0


def test_native_3d_trace2cp_cone_candidates_are_deterministic_and_bounded() -> None:
    axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    candidates_a = generate_cone_candidates(
        axis,
        max_angle_degrees=25.0,
        grid_size=5,
    )
    candidates_b = generate_cone_candidates(
        axis,
        max_angle_degrees=25.0,
        grid_size=5,
    )

    assert np.allclose(candidates_a, candidates_b)
    assert candidates_a.shape == (5 * 5, 3)
    assert np.allclose(candidates_a[0], axis)
    assert np.allclose(np.linalg.norm(candidates_a, axis=1), 1.0, atol=1.0e-5)
    angles = np.rad2deg(np.arccos(np.clip(candidates_a @ axis, -1.0, 1.0)))
    assert float(np.max(angles)) <= 25.0 + 1.0e-4


def test_native_3d_trace2cp_angle_step_candidates_are_deterministic_and_bounded() -> None:
    axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    candidates_a = generate_cone_candidates_by_angle_step(
        axis,
        max_angle_degrees=25.0,
        angle_step_degrees=5.0,
    )
    candidates_b = generate_cone_candidates_by_angle_step(
        axis,
        max_angle_degrees=25.0,
        angle_step_degrees=5.0,
    )

    assert np.allclose(candidates_a, candidates_b)
    assert candidates_a.shape == (81, 3)
    assert np.allclose(candidates_a[0], axis)
    assert np.allclose(np.linalg.norm(candidates_a, axis=1), 1.0, atol=1.0e-5)
    angles = np.rad2deg(np.arccos(np.clip(candidates_a @ axis, -1.0, 1.0)))
    assert float(np.max(angles)) <= 25.0 + 1.0e-4


def test_native_3d_trace2cp_torch_candidate_generation_matches_numpy() -> None:
    axes = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    cfg = NativeTrace2CpConfig(cone_angle_degrees=25.0, cone_angle_step_degrees=5.0)

    candidates = _trace_candidate_directions_torch(axes, cfg).detach().cpu().numpy()

    assert candidates.shape == (2, 81, 3)
    assert np.allclose(
        candidates[0],
        generate_cone_candidates_by_angle_step(
            axes[0].numpy(),
            max_angle_degrees=25.0,
            angle_step_degrees=5.0,
        ),
        atol=1.0e-6,
    )
    assert np.allclose(
        candidates[1],
        generate_cone_candidates_by_angle_step(
            axes[1].numpy(),
            max_angle_degrees=25.0,
            angle_step_degrees=5.0,
        ),
        atol=1.0e-6,
    )


def test_native_3d_trace2cp_defaults_to_training_patch_size() -> None:
    assert NativeTrace2CpConfig().inference_patch_shape_zyx == (64, 64, 64)
    assert NativeTrace2CpConfig().cone_grid_size == 25
    assert NativeTrace2CpConfig().cone_angle_step_degrees == 5.0
    assert NativeTrace2CpConfig().beam_width == 8
    assert NativeTrace2CpConfig().beam_prune_distance_voxels == 1.0
    assert NativeTrace2CpConfig().beam_lookahead_steps == 1
    assert NativeTrace2CpConfig().candidate_substeps == 1
    assert NativeTrace2CpConfig().max_step_factor == 3.0
    assert NativeTrace2CpConfig().max_steps is None
    assert NativeTrace2CpConfig().smoothness_weight == 2.0
    assert NativeTrace2CpConfig().smoothness_tangent_weight == 10.0
    assert NativeTrace2CpConfig().smoothness_normal_weight == 0.1
    assert NativeTrace2CpConfig().smoothness_free_angle_degrees == 0.0
    assert NativeTrace2CpConfig().cumulative_smoothness_steps == 4
    assert NativeTrace2CpConfig().cumulative_smoothness_tangent_weight == 2.0
    assert NativeTrace2CpConfig().all_pairs_direction_product is True
    assert NativeTrace2CpConfig().core_margin_voxels == 20
    assert NativeTrace2CpConfig().whole_fiber_error_threshold_voxels == 100.0


def test_native_3d_trace2cp_cli_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "trace2cp_tool",
            "config.json",
            "--checkpoint",
            "checkpoint.pt",
            "--export-dir",
            "out",
        ],
    )

    args = _parse_args()

    assert args.sample_index is None
    assert args.beam_width == 8
    assert args.beam_lookahead_steps == 1
    assert args.smoothness_normal_weight == 0.1
    assert args.smoothness_tangent_weight == 10.0
    assert args.core_margin_voxels == 20


def test_native_3d_trace2cp_mode_routes_only_unselected_fiber_json_to_whole_fiber() -> None:
    assert _native_trace2cp_whole_fiber_mode(
        fiber_json=Path("fiber.json"),
        sample_index=None,
        start_cp_index=None,
        target_cp_index=None,
    )
    assert not _native_trace2cp_whole_fiber_mode(
        fiber_json=Path("fiber.json"),
        sample_index=0,
        start_cp_index=None,
        target_cp_index=None,
    )
    assert not _native_trace2cp_whole_fiber_mode(
        fiber_json=Path("fiber.json"),
        sample_index=None,
        start_cp_index=0,
        target_cp_index=1,
    )
    assert not _native_trace2cp_whole_fiber_mode(
        fiber_json=None,
        sample_index=None,
        start_cp_index=None,
        target_cp_index=None,
    )
    with pytest.raises(ValueError, match="provided together"):
        _native_trace2cp_whole_fiber_mode(
            fiber_json=Path("fiber.json"),
            sample_index=None,
            start_cp_index=0,
            target_cp_index=None,
        )


def test_native_3d_trace2cp_explicit_segment_selection_uses_fiber_cp_indices() -> None:
    record = SimpleNamespace(
        fiber=SimpleNamespace(
            control_points_zyx=np.zeros((5, 3), dtype=np.float32),
        ),
    )
    loader = SimpleNamespace(records=[record])

    selection = _resolve_native_trace2cp_selection(
        loader,
        sample_index=99,
        fiber_json=Path("fiber.json"),
        start_cp_index=3,
        target_cp_index=1,
        target_offset=1,
        sample_mode=None,
    )

    assert selection.record is record
    assert selection.sample_index == 3
    assert selection.sample_mode == "flat"
    assert selection.start_cp_index == 3
    assert selection.target_cp_index == 1
    assert selection.explicit_segment is True


def test_native_3d_trace2cp_explicit_segment_selection_requires_fiber_json() -> None:
    loader = SimpleNamespace(
        records=[
            SimpleNamespace(
                fiber=SimpleNamespace(control_points_zyx=np.zeros((3, 3), dtype=np.float32))
            )
        ]
    )

    with pytest.raises(ValueError, match="require --fiber-json"):
        _resolve_native_trace2cp_selection(
            loader,
            sample_index=0,
            fiber_json=None,
            start_cp_index=0,
            target_cp_index=1,
            target_offset=1,
            sample_mode=None,
        )


def test_native_3d_trace2cp_sample_index_selection_stays_unchanged() -> None:
    record = SimpleNamespace(
        fiber=SimpleNamespace(
            control_points_zyx=np.zeros((5, 3), dtype=np.float32),
        ),
    )

    class FakeLoader:
        def descriptor_for_sample_index(self, sample_index: int, *, sample_mode: str):
            assert sample_index == 11
            assert sample_mode == "random"
            return record, 7, 2

    selection = _resolve_native_trace2cp_selection(
        FakeLoader(),
        sample_index=11,
        fiber_json=None,
        start_cp_index=None,
        target_cp_index=None,
        target_offset=2,
        sample_mode=None,
    )

    assert selection.record is record
    assert selection.record_index == 7
    assert selection.sample_index == 11
    assert selection.sample_mode == "random"
    assert selection.start_cp_index == 2
    assert selection.target_cp_index == 4
    assert selection.explicit_segment is False


def test_native_3d_trace2cp_image_to_u8_uses_inference_zscore_values() -> None:
    image = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [np.nan, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    valid = np.asarray(
        [
            [True, True, True],
            [True, False, False],
        ],
        dtype=bool,
    )

    rendered = _image_to_u8(image, valid, normalization="zscore")

    assert rendered[0, 1] == 128
    assert rendered[0, 0] < rendered[0, 1] < rendered[0, 2]
    assert rendered[1, 0] == 0
    assert rendered[1, 1] == 0


def test_native_3d_trace2cp_strip_presence_samples_cache_in_one_batch() -> None:
    class FakeCache:
        def __init__(self) -> None:
            self.points_zyx: np.ndarray | None = None

        def sample_points_torch(
            self,
            points_zyx: np.ndarray,
            *,
            progress_label: str | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            assert progress_label is None
            self.points_zyx = np.asarray(points_zyx, dtype=np.float32).copy()
            count = int(points_zyx.shape[0])
            directions = torch.zeros((count, 3), dtype=torch.float32)
            presence = torch.arange(count, dtype=torch.float32) / max(1, count - 1)
            valid = torch.ones((count,), dtype=torch.bool)
            return directions, presence, valid

    coords_xyz = np.asarray(
        [
            [[2.0, 4.0, 6.0], [4.0, 6.0, 8.0]],
            [[6.0, 8.0, 10.0], [8.0, 10.0, 12.0]],
        ],
        dtype=np.float32,
    )
    grid_valid = np.asarray([[True, False], [True, True]], dtype=bool)
    cache = FakeCache()

    presence, valid = _sample_presence_on_strip(
        cache,
        coords_xyz,
        grid_valid,
        spacing_base=2.0,
    )

    assert cache.points_zyx is not None
    expected_points = coords_xyz[grid_valid][:, [2, 1, 0]] / np.float32(2.0)
    assert np.allclose(cache.points_zyx, expected_points)
    assert presence.shape == (2, 2)
    assert valid.tolist() == [[True, False], [True, True]]
    assert presence[0, 0] == pytest.approx(0.0)
    assert presence[1, 1] == pytest.approx(1.0)


def test_native_3d_trace2cp_projects_trace_to_initial_strip_axis() -> None:
    line_points_xyz = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    line_xy = np.asarray([[0.0, 5.0], [10.0, 5.0]], dtype=np.float32)
    axes = np.zeros((16, 16, 3), dtype=np.float32)
    axes[..., 1] = 1.0
    source = SimpleNamespace(
        record=SimpleNamespace(volume_spacing_base=1.0),
        line_window=SimpleNamespace(line_points_xyz=line_points_xyz),
        line_xy=line_xy,
        grid=SimpleNamespace(
            offset_axis_xyz=axes,
            valid_mask=np.ones((16, 16), dtype=bool),
        ),
    )
    trace_xyz = np.asarray([[2.0, 3.0, 0.0], [8.0, -2.0, 0.0]], dtype=np.float32)

    projected = _project_trace_to_initial_strip(
        source,
        trace_xyz,
        axis_name="offset_axis_xyz",
    )

    assert np.allclose(projected, [[2.0, 8.0], [8.0, 3.0]], atol=1.0e-5)


def test_native_3d_trace2cp_adaptive_cross_height_expands_and_caps() -> None:
    centered = (np.asarray([[1.0, 9.5], [4.0, 9.5]], dtype=np.float32), (0, 0, 0, 255))
    assert _adaptive_trace2cp_cross_strip_height(20, ((centered,),)) == 5

    shifted = (np.asarray([[1.0, 12.5], [4.0, 26.5]], dtype=np.float32), (0, 0, 0, 255))
    assert _adaptive_trace2cp_cross_strip_height(40, ((shifted,),)) == 27

    far = (np.asarray([[1.0, -100.0], [4.0, 100.0]], dtype=np.float32), (0, 0, 0, 255))
    assert _adaptive_trace2cp_cross_strip_height(20, ((far,),)) == 19


def test_native_3d_trace2cp_converts_volume_trace_to_source_xyz_offsets() -> None:
    line_points_xyz = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    line_xy = np.asarray([[0.0, 5.0], [10.0, 5.0]], dtype=np.float32)
    row_axes = np.zeros((16, 16, 3), dtype=np.float32)
    row_axes[..., 1] = 1.0
    side_axes = np.zeros((16, 16, 3), dtype=np.float32)
    side_axes[..., 2] = 1.0
    source = SimpleNamespace(
        record=SimpleNamespace(volume_spacing_base=2.0),
        line_window=SimpleNamespace(line_points_xyz=line_points_xyz),
        line_xy=line_xy,
        grid=SimpleNamespace(
            offset_axis_xyz=row_axes,
            side_axis_xyz=side_axes,
            valid_mask=np.ones((16, 16), dtype=bool),
        ),
    )
    trace_xyz = np.asarray(
        [
            [2.0, 4.0, 6.0],
            [8.0, -2.0, -4.0],
        ],
        dtype=np.float32,
    )

    source_trace = _volume_trace_to_source_trace_xyz(source, trace_xyz)

    assert np.allclose(source_trace, [[2.0, 7.0, 3.0], [8.0, 4.0, -2.0]], atol=1.0e-5)


def test_native_3d_trace2cp_fusion_uses_pairwise_arc_length_score() -> None:
    start = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    target = np.asarray([10.0, 0.0, 0.0], dtype=np.float32)
    forward = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    reverse = np.asarray(
        [
            [10.0, 2.0, 0.0],
            [5.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    fusion = fuse_forward_reverse_traces(
        forward,
        reverse,
        start_zyx=start,
        target_zyx=target,
        step_voxels=1.0,
    )

    assert fusion.reached_overlap
    assert fusion.reason == "pairwise_arc_length_meeting"
    assert fusion.closest_progress == pytest.approx(0.45, abs=1.0e-6)
    assert np.allclose(fusion.closest_forward_zyx, [4.0, 0.0, 0.0], atol=1.0e-6)
    assert np.allclose(fusion.closest_reverse_zyx, [5.0, 2.0, 0.0], atol=1.0e-6)
    assert np.allclose(fusion.closest_midpoint_zyx, [4.5, 1.0, 0.0], atol=1.0e-6)
    assert fusion.raw_gap_voxels == pytest.approx(math.sqrt(5.0), abs=1.0e-6)
    assert fusion.considered_gap_voxels == pytest.approx(9.0 + 2.0 * math.sqrt(5.0), abs=1.0e-6)
    assert fusion.center_penalty == pytest.approx(1.0, abs=1.0e-6)
    assert np.allclose(fusion.fused_zyx[0], start, atol=1.0e-6)
    assert np.allclose(fusion.fused_zyx[-1], target, atol=1.0e-6)
    progress = fusion.fused_zyx[:, 0] / 10.0
    assert np.all(np.diff(progress) >= -1.0e-5)


def test_native_3d_trace2cp_fusion_tie_prefers_balanced_meeting() -> None:
    start = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    target = np.asarray([10.0, 0.0, 0.0], dtype=np.float32)
    forward = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    reverse = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    fusion = fuse_forward_reverse_traces(
        forward,
        reverse,
        start_zyx=start,
        target_zyx=target,
        step_voxels=1.0,
    )

    assert fusion.reached_overlap
    assert fusion.reason == "pairwise_arc_length_meeting"
    assert fusion.closest_progress == pytest.approx(0.5, abs=1.0e-6)
    assert fusion.raw_gap_voxels == pytest.approx(0.0, abs=1.0e-6)
    assert fusion.considered_gap_voxels == pytest.approx(10.0, abs=1.0e-6)
    assert np.allclose(fusion.closest_forward_zyx, [5.0, 0.0, 0.0], atol=1.0e-6)
    assert np.allclose(fusion.closest_reverse_zyx, [5.0, 0.0, 0.0], atol=1.0e-6)
    assert np.allclose(fusion.closest_midpoint_zyx, [5.0, 0.0, 0.0], atol=1.0e-6)
    assert np.allclose(fusion.fused_zyx[0], start, atol=1.0e-6)
    assert np.allclose(fusion.fused_zyx[-1], target, atol=1.0e-6)


def test_native_3d_trace2cp_fusion_preserves_wrong_direction_trace_order() -> None:
    start = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    target = np.asarray([10.0, 0.0, 0.0], dtype=np.float32)
    forward = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    reverse = np.asarray(
        [
            [10.0, 10.0, 0.0],
            [12.0, 10.0, 0.0],
            [14.0, 10.0, 0.0],
        ],
        dtype=np.float32,
    )

    fusion = fuse_forward_reverse_traces(
        forward,
        reverse,
        start_zyx=start,
        target_zyx=target,
        step_voxels=1.0,
    )

    assert fusion.reached_overlap
    assert fusion.reason == "pairwise_arc_length_meeting"
    assert fusion.closest_progress == pytest.approx(0.7, abs=1.0e-6)
    assert np.allclose(fusion.closest_forward_zyx, [4.0, 0.0, 0.0], atol=1.0e-5)
    assert np.allclose(fusion.closest_reverse_zyx, [10.0, 10.0, 0.0], atol=1.0e-5)
    assert np.allclose(fusion.closest_midpoint_zyx, [7.0, 5.0, 0.0], atol=1.0e-5)
    assert np.allclose(fusion.fused_zyx[0], start, atol=1.0e-6)
    assert np.allclose(fusion.fused_zyx[-1], target, atol=1.0e-6)


def test_native_3d_trace2cp_plane_crossing_interpolates() -> None:
    crossing = _interpolate_plane_crossing(
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 10.0], dtype=np.float32),
        plane_point_zyx=np.asarray([0.0, 0.0, 4.0], dtype=np.float32),
        plane_normal_zyx=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )

    assert crossing is not None
    assert np.allclose(crossing, [0.0, 0.0, 4.0])


def test_native_3d_trace2cp_target_plane_error_uses_in_plane_distance() -> None:
    error = _target_plane_in_plane_error_voxels(
        np.asarray([3.0, 4.0, 12.0], dtype=np.float32),
        target_zyx=np.asarray([0.0, 0.0, 10.0], dtype=np.float32),
        plane_normal_zyx=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )

    assert error == pytest.approx(5.0, abs=1.0e-6)


def _whole_native_trace_record() -> SimpleNamespace:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    fiber = Vc3dFiber(
        path=Path("fiber.json"),
        version=1,
        line_points_xyz=points,
        control_points_xyz=points.copy(),
        generation=1,
        metadata={},
    )
    return SimpleNamespace(fiber=fiber, volume_spacing_base=1.0)


def test_native_3d_whole_fiber_trace_all_success_has_zero_restart_rate() -> None:
    record = _whole_native_trace_record()
    cache = SimpleNamespace(_blocks={})

    def fake_trace(_cache, *, start_zyx, target_zyx, **_kwargs):
        return NativeTraceResult(
            trace_zyx=np.stack([start_zyx, target_zyx], axis=0).astype(np.float32),
            reached_target_plane=True,
            reason="target_plane",
            steps=(),
        )

    result = trace_native_3d_whole_fiber(
        cache,
        record=record,
        cfg=NativeTrace2CpConfig(step_voxels=1.0, cone_grid_size=1),
        error_threshold_voxels=1.0,
        trace_segment_fn=fake_trace,
    )

    assert result.segment_count == 2
    assert result.restart_count == 0
    assert result.restart_rate == pytest.approx(0.0)
    assert [segment.success for segment in result.segments] == [True, True]
    assert np.allclose(result.stitched_trace_zyx[[0, -1]], [[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]])


def test_native_3d_whole_fiber_trace_restarts_after_plane_miss() -> None:
    record = _whole_native_trace_record()
    cache = SimpleNamespace(_blocks={})
    starts: list[np.ndarray] = []

    def fake_trace(_cache, *, start_zyx, target_zyx, **_kwargs):
        starts.append(np.asarray(start_zyx, dtype=np.float32).copy())
        if len(starts) == 1:
            return NativeTraceResult(
                trace_zyx=np.stack(
                    [
                        start_zyx,
                        np.asarray(start_zyx, dtype=np.float32) + np.asarray([0.0, 0.0, 2.0], dtype=np.float32),
                    ],
                    axis=0,
                ).astype(np.float32),
                reached_target_plane=False,
                reason="max_step_factor",
                steps=(),
            )
        return NativeTraceResult(
            trace_zyx=np.stack([start_zyx, target_zyx], axis=0).astype(np.float32),
            reached_target_plane=True,
            reason="target_plane",
            steps=(),
        )

    result = trace_native_3d_whole_fiber(
        cache,
        record=record,
        cfg=NativeTrace2CpConfig(step_voxels=1.0, cone_grid_size=1),
        error_threshold_voxels=1.0,
        trace_segment_fn=fake_trace,
    )

    assert result.restart_count == 1
    assert result.restart_rate == pytest.approx(0.5)
    assert result.segments[0].reason == "max_step_factor"
    assert result.segments[0].restart
    assert np.allclose(starts[1], [0.0, 0.0, 10.0])


def test_native_3d_whole_fiber_trace_restarts_after_large_in_plane_error() -> None:
    record = _whole_native_trace_record()
    cache = SimpleNamespace(_blocks={})

    def fake_trace(_cache, *, start_zyx, target_zyx, **_kwargs):
        if np.allclose(target_zyx, [0.0, 0.0, 10.0]):
            crossing = np.asarray(target_zyx, dtype=np.float32) + np.asarray([0.0, 5.0, 0.0], dtype=np.float32)
            return NativeTraceResult(
                trace_zyx=np.stack([start_zyx, crossing], axis=0).astype(np.float32),
                reached_target_plane=True,
                reason="target_plane",
                steps=(),
            )
        return NativeTraceResult(
            trace_zyx=np.stack([start_zyx, target_zyx], axis=0).astype(np.float32),
            reached_target_plane=True,
            reason="target_plane",
            steps=(),
        )

    result = trace_native_3d_whole_fiber(
        cache,
        record=record,
        cfg=NativeTrace2CpConfig(step_voxels=1.0, cone_grid_size=1),
        error_threshold_voxels=1.0,
        trace_segment_fn=fake_trace,
    )

    assert result.restart_count == 1
    assert result.segments[0].reached_target_plane
    assert result.segments[0].reason == "in_plane_error"
    assert result.segments[0].in_plane_error_voxels == pytest.approx(5.0, abs=1.0e-6)


def test_native_3d_whole_fiber_trace_last_segment_failure_finishes() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    record = SimpleNamespace(
        fiber=Vc3dFiber(
            path=Path("fiber.json"),
            version=1,
            line_points_xyz=points,
            control_points_xyz=points.copy(),
            generation=1,
            metadata={},
        ),
        volume_spacing_base=1.0,
    )
    cache = SimpleNamespace(_blocks={})

    def fake_trace(_cache, *, start_zyx, **_kwargs):
        return NativeTraceResult(
            trace_zyx=np.stack(
                [start_zyx, np.asarray(start_zyx, dtype=np.float32) + np.asarray([0.0, 0.0, 1.0], dtype=np.float32)],
                axis=0,
            ).astype(np.float32),
            reached_target_plane=False,
            reason="trace_step_limit",
            steps=(),
        )

    result = trace_native_3d_whole_fiber(
        cache,
        record=record,
        cfg=NativeTrace2CpConfig(step_voxels=1.0, cone_grid_size=1),
        error_threshold_voxels=1.0,
        trace_segment_fn=fake_trace,
    )

    assert result.segment_count == 1
    assert result.restart_count == 1
    assert result.segments[0].reason == "trace_step_limit"


def _native_whole_fiber_segment(
    start_cp: int,
    target_cp: int,
    *,
    success: bool,
) -> object:
    return SimpleNamespace(
        start_cp_index=int(start_cp),
        target_cp_index=int(target_cp),
        trace_zyx=np.asarray(
            [[0.0, 0.0, float(start_cp)], [0.0, 0.0, float(target_cp)]],
            dtype=np.float32,
        ),
        start_zyx=np.asarray([0.0, 0.0, float(start_cp)], dtype=np.float32),
        target_zyx=np.asarray([0.0, 0.0, float(target_cp)], dtype=np.float32),
        reached_target_plane=bool(success),
        success=bool(success),
        restart=not bool(success),
        reason="target_plane" if success else "trace_step_limit",
        in_plane_error_voxels=0.0 if success else float("inf"),
        reference_arc_distance_voxels=float(target_cp),
        step_count=1,
    )


def test_native_3d_whole_fiber_visual_spans_are_restart_delimited() -> None:
    segments = [
        _native_whole_fiber_segment(0, 1, success=True),
        _native_whole_fiber_segment(1, 2, success=False),
        _native_whole_fiber_segment(2, 3, success=True),
        _native_whole_fiber_segment(3, 4, success=True),
    ]

    spans = _native_whole_fiber_visual_spans(segments)

    assert [(span.start_cp_index, span.end_cp_index, span.restart_after) for span in spans] == [
        (0, 2, True),
        (2, 4, False),
    ]
    assert [len(span.segments) for span in spans] == [2, 2]


def test_native_3d_whole_fiber_span_source_uses_fixed_cross_width() -> None:
    class FakeGeometryLoader:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def build_trace2cp_segment_source(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "source"

    loader = FakeGeometryLoader()

    source = _build_native_whole_fiber_span_source(
        loader,
        start_cp_index=3,
        end_cp_index=9,
        trace2cp_rf_margin_px=17.5,
    )

    assert source == "source"
    assert len(loader.calls) == 1
    args, kwargs = loader.calls[0]
    assert args == (3,)
    assert kwargs["target_control_point_index"] == 9
    assert kwargs["rf_margin_px"] == pytest.approx(17.5)
    assert kwargs["cross_strip_height_px"] == 64
    assert kwargs["sample_mode"] == "flat"


def test_native_3d_trace2cp_block_router_uses_trusted_core() -> None:
    encoded = encode_lasagna_direction_3x2(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    )[0]

    class ConstantNativeModel(torch.nn.Module):
        def forward(self, volume: torch.Tensor) -> torch.Tensor:
            b, _c, d, h, w = volume.shape
            out = torch.zeros((b, 7, d, h, w), dtype=torch.float32, device=volume.device)
            out[:, :6] = encoded.to(volume.device).view(1, 6, 1, 1, 1)
            out[:, 6] = 1.0
            return out

    record = _native_trace_record(np.ones((32, 32, 32), dtype=np.float32))
    cache = NativeTraceFieldCache(
        record=record,
        model=ConstantNativeModel(),
        config=_loader(augment_enabled=False).config,
        image_normalization="none",
        patch_shape_zyx=(16, 16, 16),
        core_margin_voxels=4,
        device=torch.device("cpu"),
    )

    first = cache.block_for_point(np.asarray([8.0, 8.0, 8.0], dtype=np.float32))
    second = cache.block_for_point(np.asarray([12.25, 8.0, 8.0], dtype=np.float32))

    assert np.allclose(first.origin_zyx, [0, 0, 0])
    assert np.allclose(second.origin_zyx, [8, 0, 0])


def test_native_3d_trace2cp_field_cache_uses_sampler_and_base_scale() -> None:
    class UnsliceableVolume:
        def __getitem__(self, _key):
            raise AssertionError("native trace field cache must not slice volume directly")

    class RecordingSampler:
        blocking = True

        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray]] = []

        def sample_coord_batch(
            self,
            coords_zyx_base: np.ndarray,
            valid_mask: np.ndarray,
        ) -> CoordinateSampleResult:
            coords = np.asarray(coords_zyx_base, dtype=np.float32)
            valid = np.asarray(valid_mask, dtype=bool)
            self.calls.append((coords.copy(), valid.copy()))
            return CoordinateSampleResult(
                image=coords[..., 0].astype(np.float32, copy=False),
                valid_mask=valid,
                stats={"sample_coord_batch": 1},
            )

    class RecordingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_volume: torch.Tensor | None = None

        def forward(self, volume: torch.Tensor) -> torch.Tensor:
            self.input_volume = volume.detach().cpu()
            b, _c, d, h, w = volume.shape
            return torch.zeros((b, 7, d, h, w), dtype=torch.float32, device=volume.device)

    sampler = RecordingSampler()
    model = RecordingModel()
    record = SimpleNamespace(
        volume=UnsliceableVolume(),
        sampler=sampler,
        volume_spacing_base=4.0,
        base_shape_zyx=(128, 128, 128),
    )
    cache = NativeTraceFieldCache(
        record=record,
        model=model,
        config=_loader(augment_enabled=False).config,
        image_normalization="none",
        patch_shape_zyx=(4, 4, 4),
        core_margin_voxels=1,
        device=torch.device("cpu"),
    )

    block = cache._infer_block(np.asarray([1, 2, 3], dtype=np.int64))

    assert block.output_czyx.shape == (7, 4, 4, 4)
    assert block.output_czyx.device.type == "cpu"
    assert block.valid_mask_zyx.device.type == "cpu"
    assert len(sampler.calls) == 1
    coords, valid = sampler.calls[0]
    assert coords.shape == (4, 4, 4, 3)
    assert valid.shape == (4, 4, 4)
    assert bool(valid.all())
    assert np.allclose(coords[0, 0, 0], [4.0, 8.0, 12.0])
    assert np.allclose(coords[0, 0, 1], [4.0, 8.0, 16.0])
    assert model.input_volume is not None
    assert float(model.input_volume[0, 0, 0, 0, 0]) == 4.0


def test_native_3d_trace2cp_candidate_score_maximizes_dot_product_presence() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_points_torch(self, points_zyx: np.ndarray):
            count = int(np.asarray(points_zyx).shape[0])
            assert count == 2
            directions = torch.tensor(
                [
                    [-1.0, 0.0, 0.0],  # sign-ambiguous match to candidate 0
                    [1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
            presence = torch.tensor([0.51, 1.0], dtype=torch.float32)
            valid = torch.ones((count,), dtype=torch.bool)
            return directions, presence, valid

    candidate_b = np.asarray(
        [0.7, math.sqrt(1.0 - 0.7 * 0.7), 0.0],
        dtype=np.float32,
    )
    best_index, total_loss, direction_loss, presence_loss, smoothness_loss, rejected = (
        _score_candidate_batch(
            FakeCache(),
            current_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            previous_step_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            candidate_directions=np.stack(
                [
                    np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                    candidate_b,
                ],
                axis=0,
            ),
            next_points=np.zeros((2, 3), dtype=np.float32),
        )
    )

    assert best_index == 0
    assert rejected == 0
    assert total_loss == pytest.approx(0.49, abs=1.0e-6)
    assert direction_loss == pytest.approx(0.0, abs=1.0e-6)
    assert presence_loss == pytest.approx(0.49, abs=1.0e-6)
    assert smoothness_loss == pytest.approx(0.0, abs=1.0e-6)


def test_native_3d_trace2cp_candidate_score_evaluates_all_branches() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_point_choices_torch(self, points_zyx: np.ndarray):
            count = int(np.asarray(points_zyx).shape[0])
            assert count == 2
            directions = torch.tensor(
                [
                    [
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ],
                    [
                        [0.7, 0.71414286, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                ],
                dtype=torch.float32,
            )
            presence = torch.tensor(
                [
                    [0.10, 0.95],
                    [1.00, 0.20],
                ],
                dtype=torch.float32,
            )
            valid = torch.ones((count, 2), dtype=torch.bool)
            return directions, presence, valid

    candidate_b = np.asarray(
        [0.7, math.sqrt(1.0 - 0.7 * 0.7), 0.0],
        dtype=np.float32,
    )
    best_index, total_loss, direction_loss, presence_loss, smoothness_loss, rejected = (
        _score_candidate_batch(
            FakeCache(),
            current_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            previous_step_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            candidate_directions=np.stack(
                [
                    np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                    candidate_b,
                ],
                axis=0,
            ),
            next_points=np.zeros((2, 3), dtype=np.float32),
        )
    )

    assert best_index == 0
    assert rejected == 0
    assert total_loss == pytest.approx(0.05, abs=1.0e-6)
    assert direction_loss == pytest.approx(0.0, abs=1.0e-6)
    assert presence_loss == pytest.approx(0.05, abs=1.0e-6)
    assert smoothness_loss == pytest.approx(0.0, abs=1.0e-6)


def test_native_3d_trace2cp_batched_candidate_score_couples_branches() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_point_choices_torch(self, points_zyx: np.ndarray):
            count = int(np.asarray(points_zyx).shape[0])
            assert count == 4
            directions = torch.tensor(
                [
                    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ],
                dtype=torch.float32,
            )
            presence = torch.tensor(
                [
                    [0.1, 0.9],
                    [0.4, 1.0],
                    [0.2, 0.8],
                    [0.2, 1.0],
                ],
                dtype=torch.float32,
            )
            valid = torch.ones((count, 2), dtype=torch.bool)
            return directions, presence, valid

    total_loss, direction_loss, presence_loss, smoothness_loss, candidate_valid, rejected = (
        _score_candidate_loss_tensors_batched(
            FakeCache(),
            current_directions=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            ),
            previous_step_directions=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            ),
            candidate_directions=torch.tensor(
                [
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ],
                dtype=torch.float32,
            ),
            next_points=torch.zeros((2, 2, 3), dtype=torch.float32),
            smoothness_weight=0.0,
        )
    )

    best_loss, best_branch = torch.min(total_loss, dim=2)
    assert candidate_valid.tolist() == [[True, True], [True, True]]
    assert rejected.tolist() == [0, 0]
    assert int(torch.argmin(best_loss[0])) == 0
    assert int(best_branch[0, 0]) == 1
    assert int(torch.argmin(best_loss[1])) == 0
    assert int(best_branch[1, 0]) == 1
    assert float(direction_loss[0, 0, 1]) == pytest.approx(0.0, abs=1.0e-6)
    assert float(presence_loss[0, 0, 1]) == pytest.approx(0.1, abs=1.0e-6)
    assert int(torch.count_nonzero(smoothness_loss).item()) == 0


def test_native_3d_trace2cp_all_pairs_direction_product_penalizes_previous_outlier() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_point_choices_torch(self, points_zyx: np.ndarray):
            points = np.asarray(points_zyx, dtype=np.float32)
            directions = torch.as_tensor(points[:, None, :], dtype=torch.float32)
            presence = torch.tensor([[1.0], [0.95]], dtype=torch.float32)
            valid = torch.ones((points.shape[0], 1), dtype=torch.bool)
            return directions, presence, valid

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    candidates = torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    kwargs = dict(
        cache=FakeCache(),
        current_directions=torch.tensor([[inv_sqrt2, inv_sqrt2, 0.0]], dtype=torch.float32),
        previous_step_directions=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        candidate_directions=candidates,
        next_points=candidates.clone(),
        smoothness_weight=0.0,
        cumulative_smoothness_tangent_weight=0.0,
    )

    legacy_total, *_ = _score_candidate_loss_tensors_batched(
        **kwargs,
        all_pairs_direction_product=False,
    )
    all_pairs_total, *_ = _score_candidate_loss_tensors_batched(
        **kwargs,
        all_pairs_direction_product=True,
    )

    assert int(torch.argmin(legacy_total[0, :, 0])) == 0
    assert int(torch.argmin(all_pairs_total[0, :, 0])) == 1


def test_native_3d_trace2cp_all_pairs_first_step_uses_regular_direction_gate() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_point_choices_torch(self, points_zyx: np.ndarray):
            points = np.asarray(points_zyx, dtype=np.float32)
            directions = torch.as_tensor(points[:, None, :], dtype=torch.float32)
            presence = torch.ones((points.shape[0], 1), dtype=torch.float32)
            valid = torch.ones((points.shape[0], 1), dtype=torch.bool)
            return directions, presence, valid

    candidates = torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    total_loss, *_ = _score_candidate_loss_tensors_batched(
        FakeCache(),
        current_directions=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        previous_step_directions=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        candidate_directions=candidates,
        next_points=candidates.clone(),
        smoothness_weight=0.0,
        cumulative_smoothness_tangent_weight=0.0,
        candidate_normals=torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ),
        candidate_normals_valid=torch.ones((1, 2), dtype=torch.bool),
        all_pairs_direction_product=True,
    )

    assert float(total_loss[0, 0, 0]) == pytest.approx(1.0, abs=1.0e-6)
    assert float(total_loss[0, 1, 0]) == pytest.approx(0.0, abs=1.0e-6)


def test_native_3d_trace2cp_candidate_substeps_reject_invalid_midpoint() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.call_sizes: list[int] = []

        def sample_point_choices_torch(self, points_zyx: np.ndarray):
            points = np.asarray(points_zyx, dtype=np.float32)
            self.call_sizes.append(int(points.shape[0]))
            directions: list[list[list[float]]] = []
            presence: list[list[float]] = []
            valid: list[list[bool]] = []
            for point in points:
                if abs(float(point[0])) > abs(float(point[1])):
                    directions.append([[1.0, 0.0, 0.0]])
                    presence.append([1.0])
                    valid.append([abs(float(point[0]) - 1.0) > 1.0e-4])
                else:
                    directions.append([[0.0, 1.0, 0.0]])
                    presence.append([0.7])
                    valid.append([True])
            return (
                torch.tensor(directions, dtype=torch.float32),
                torch.tensor(presence, dtype=torch.float32),
                torch.tensor(valid, dtype=torch.bool),
            )

    current = np.zeros((3,), dtype=np.float32)
    candidates = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    next_points = current[None, :] + candidates * np.float32(2.0)

    endpoint_cache = FakeCache()
    best_endpoint, endpoint_loss, *_endpoint_rest = _score_candidate_batch(
        endpoint_cache,
        current_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        previous_step_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        candidate_directions=candidates,
        next_points=next_points,
        smoothness_weight=0.0,
    )

    substep_cache = FakeCache()
    best_substep, substep_loss, _dir_loss, _presence_loss, _smoothness_loss, rejected = (
        _score_candidate_batch(
            substep_cache,
            current_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            previous_step_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            candidate_directions=candidates,
            next_points=next_points,
            current_point=current,
            step_voxels=2.0,
            candidate_substeps=2,
            smoothness_weight=0.0,
        )
    )

    assert endpoint_cache.call_sizes == [2]
    assert best_endpoint == 0
    assert endpoint_loss == pytest.approx(0.0, abs=1.0e-6)
    assert substep_cache.call_sizes == [4]
    assert best_substep == 1
    assert rejected == 1
    assert substep_loss == pytest.approx(1.0, abs=1.0e-6)


def test_native_3d_trace2cp_normal_aware_smoothness_splits_turn_components() -> None:
    previous = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    tangent_turn = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float32)
    normal_tilt = torch.tensor(
        [[[1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0)]]],
        dtype=torch.float32,
    )
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)

    tangent_only = _native_smoothness_loss_torch(
        previous,
        tangent_turn,
        candidate_normals=normal,
        candidate_normals_valid=valid,
        smoothness_weight=0.0,
        smoothness_tangent_weight=1.0,
        smoothness_normal_weight=0.0,
        smoothness_free_angle_degrees=0.0,
    )
    tangent_ignored = _native_smoothness_loss_torch(
        previous,
        tangent_turn,
        candidate_normals=normal,
        candidate_normals_valid=valid,
        smoothness_weight=0.0,
        smoothness_tangent_weight=0.0,
        smoothness_normal_weight=1.0,
        smoothness_free_angle_degrees=0.0,
    )
    normal_only = _native_smoothness_loss_torch(
        previous,
        normal_tilt,
        candidate_normals=normal,
        candidate_normals_valid=valid,
        smoothness_weight=0.0,
        smoothness_tangent_weight=0.0,
        smoothness_normal_weight=1.0,
        smoothness_free_angle_degrees=0.0,
    )
    normal_ignored = _native_smoothness_loss_torch(
        previous,
        normal_tilt,
        candidate_normals=normal,
        candidate_normals_valid=valid,
        smoothness_weight=0.0,
        smoothness_tangent_weight=1.0,
        smoothness_normal_weight=0.0,
        smoothness_free_angle_degrees=0.0,
    )

    assert float(tangent_only[0, 0]) == pytest.approx((math.pi / 2.0) ** 2, abs=1.0e-6)
    assert float(tangent_ignored[0, 0]) == pytest.approx(0.0, abs=1.0e-6)
    assert float(normal_only[0, 0]) == pytest.approx((math.pi / 4.0) ** 2, abs=1.0e-6)
    assert float(normal_ignored[0, 0]) == pytest.approx(0.0, abs=1.0e-6)


def test_native_3d_trace2cp_normal_aware_smoothness_is_normal_sign_ambiguous() -> None:
    previous = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    candidate = torch.tensor(
        [[[0.5, 0.5, math.sqrt(0.5)]]],
        dtype=torch.float32,
    )
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)

    positive = _native_smoothness_loss_torch(
        previous,
        candidate,
        candidate_normals=normal,
        candidate_normals_valid=valid,
        smoothness_weight=0.0,
        smoothness_tangent_weight=1.0,
        smoothness_normal_weight=1.0,
        smoothness_free_angle_degrees=0.0,
    )
    negative = _native_smoothness_loss_torch(
        previous,
        candidate,
        candidate_normals=-normal,
        candidate_normals_valid=valid,
        smoothness_weight=0.0,
        smoothness_tangent_weight=1.0,
        smoothness_normal_weight=1.0,
        smoothness_free_angle_degrees=0.0,
    )

    assert float(positive[0, 0]) == pytest.approx(float(negative[0, 0]), abs=1.0e-6)


def test_native_3d_trace2cp_start_direction_uses_best_alignment_not_presence() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_point_choices_torch(self, points_zyx: np.ndarray):
            count = int(np.asarray(points_zyx, dtype=np.float32).shape[0])
            directions = torch.tensor(
                [[[0.0, 0.0, 1.0], [0.0, 0.6, 0.8]]],
                dtype=torch.float32,
            ).expand(count, 2, 3)
            presence = torch.tensor([[0.05, 1.0]], dtype=torch.float32).expand(count, 2)
            valid = torch.ones((count, 2), dtype=torch.bool)
            return directions, presence, valid

    direction, presence, valid = _sample_trace_start_direction_aligned(
        FakeCache(),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        cp_reference_direction_zyx=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )

    assert valid
    assert presence == pytest.approx(0.05, abs=1.0e-6)
    assert np.allclose(direction, [0.0, 0.0, 1.0], atol=1.0e-6)


def test_native_3d_trace2cp_first_step_applies_regular_smoothness() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_point_choices_torch(self, points_zyx: np.ndarray):
            points = np.asarray(points_zyx, dtype=np.float32)
            directions = torch.as_tensor(points[:, None, :], dtype=torch.float32)
            presence = torch.ones((points.shape[0], 1), dtype=torch.float32)
            valid = torch.ones((points.shape[0], 1), dtype=torch.bool)
            return directions, presence, valid

    current = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    previous = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    candidates = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    normals = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid_normals = torch.ones((1, 2), dtype=torch.bool)

    regular = _score_candidate_loss_tensors_batched(
        FakeCache(),
        current_directions=current,
        previous_step_directions=previous,
        candidate_directions=candidates,
        next_points=candidates.clone(),
        smoothness_weight=0.0,
        smoothness_tangent_weight=10.0,
        smoothness_normal_weight=0.0,
        smoothness_free_angle_degrees=0.0,
        candidate_normals=normals,
        candidate_normals_valid=valid_normals,
    )

    regular_total, _regular_direction, _regular_presence, regular_smooth, *_ = regular
    assert float(regular_total[0, 0, 0]) == pytest.approx(0.0, abs=1.0e-6)
    assert math.isfinite(float(regular_total[0, 1, 0]))
    assert float(regular_total[0, 1, 0]) > 1.0
    assert float(regular_smooth[0, 1]) > 1.0


def test_native_3d_trace2cp_first_step_candidate_direction_still_matters() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_point_choices_torch(self, points_zyx: np.ndarray):
            points = np.asarray(points_zyx, dtype=np.float32)
            directions = torch.as_tensor(points[:, None, :], dtype=torch.float32)
            presence = torch.ones((points.shape[0], 1), dtype=torch.float32)
            valid = torch.ones((points.shape[0], 1), dtype=torch.bool)
            return directions, presence, valid

    tilted = np.asarray([math.sqrt(0.5), 0.0, math.sqrt(0.5)], dtype=np.float32)
    candidates = torch.tensor([[[1.0, 0.0, 0.0], tilted.tolist()]], dtype=torch.float32)
    total_loss, direction_loss, _presence_loss, smoothness_loss, *_ = (
        _score_candidate_loss_tensors_batched(
            FakeCache(),
            current_directions=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            previous_step_directions=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            candidate_directions=candidates,
            next_points=candidates.clone(),
            smoothness_weight=0.0,
            smoothness_tangent_weight=10.0,
            smoothness_normal_weight=10.0,
            candidate_normals=torch.tensor(
                [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]],
                dtype=torch.float32,
            ),
            candidate_normals_valid=torch.ones((1, 2), dtype=torch.bool),
        )
    )

    assert float(total_loss[0, 0, 0]) == pytest.approx(0.0, abs=1.0e-6)
    assert float(total_loss[0, 1, 0]) > 1.0
    assert float(direction_loss[0, 1, 0]) == pytest.approx(
        0.5 * (1.0 - math.sqrt(0.5)),
        abs=1.0e-6,
    )
    assert float(smoothness_loss[0, 1]) > 1.0


def test_native_3d_trace2cp_cumulative_tangent_smoothness_is_tangent_only() -> None:
    history = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    candidates = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [math.sqrt(0.5), 0.0, math.sqrt(0.5)]]],
        dtype=torch.float32,
    )
    normals = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )

    loss = _native_cumulative_tangent_smoothness_loss_torch(
        history,
        candidates,
        candidate_normals=normals,
        candidate_normals_valid=torch.ones((1, 3), dtype=torch.bool),
        cumulative_smoothness_tangent_weight=2.0,
        smoothness_free_angle_degrees=0.0,
    )

    assert float(loss[0, 0]) == pytest.approx(0.0, abs=1.0e-6)
    assert float(loss[0, 1]) == pytest.approx(2.0 * (math.pi / 2.0) ** 2, abs=1.0e-6)
    assert float(loss[0, 2]) == pytest.approx(0.0, abs=1.0e-6)


def test_native_3d_trace2cp_cumulative_tangent_smoothness_skips_invalid_normals() -> None:
    history = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    candidates = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float32)
    normals = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)

    loss = _native_cumulative_tangent_smoothness_loss_torch(
        history,
        candidates,
        candidate_normals=normals,
        candidate_normals_valid=torch.zeros((1, 1), dtype=torch.bool),
        cumulative_smoothness_tangent_weight=2.0,
        smoothness_free_angle_degrees=0.0,
    )

    assert float(loss[0, 0]) == pytest.approx(0.0, abs=1.0e-6)


def test_native_3d_trace2cp_cumulative_tangent_smoothness_applies_to_first_step() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_point_choices_torch(self, points_zyx: np.ndarray):
            points = np.asarray(points_zyx, dtype=np.float32)
            directions = torch.as_tensor(points[:, None, :], dtype=torch.float32)
            presence = torch.ones((points.shape[0], 1), dtype=torch.float32)
            valid = torch.ones((points.shape[0], 1), dtype=torch.bool)
            return directions, presence, valid

    candidates = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float32)
    total_loss, _direction_loss, _presence_loss, smoothness_loss, *_ = (
        _score_candidate_loss_tensors_batched(
            FakeCache(),
            current_directions=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            previous_step_directions=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            history_directions=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            candidate_directions=candidates,
            next_points=candidates.clone(),
            smoothness_weight=0.0,
            smoothness_tangent_weight=0.0,
            smoothness_normal_weight=0.0,
            cumulative_smoothness_tangent_weight=10.0,
            candidate_normals=torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32),
            candidate_normals_valid=torch.ones((1, 1), dtype=torch.bool),
        )
    )

    assert float(total_loss[0, 0, 0]) > 1.0
    assert float(smoothness_loss[0, 0]) > 1.0


def test_native_3d_trace2cp_normal_aware_smoothness_changes_candidate_choice() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_point_choices_torch(self, points_zyx: np.ndarray):
            count = int(np.asarray(points_zyx).shape[0])
            assert count == 2
            directions = torch.tensor(
                [
                    [[0.0, 1.0, 0.0]],
                    [[1.0, 0.0, 0.0]],
                ],
                dtype=torch.float32,
            )
            presence = torch.ones((count, 1), dtype=torch.float32)
            valid = torch.ones((count, 1), dtype=torch.bool)
            return directions, presence, valid

    candidates = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    best_without, *_ = _score_candidate_batch(
        FakeCache(),
        current_direction=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        previous_step_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        candidate_directions=candidates,
        next_points=np.zeros((2, 3), dtype=np.float32),
        smoothness_weight=0.0,
    )
    best_with, total_loss, _direction_loss, _presence_loss, smoothness_loss, _rejected = (
        _score_candidate_batch(
            FakeCache(),
            current_direction=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            previous_step_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            candidate_directions=candidates,
            next_points=np.zeros((2, 3), dtype=np.float32),
            smoothness_weight=0.0,
            smoothness_tangent_weight=2.0,
            smoothness_normal_weight=0.0,
            smoothness_free_angle_degrees=0.0,
            candidate_normals=np.asarray(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            candidate_normals_valid=np.asarray([True, True]),
        )
    )

    assert best_without == 0
    assert best_with == 1
    assert total_loss == pytest.approx(1.0, abs=1.0e-6)
    assert smoothness_loss == pytest.approx(0.0, abs=1.0e-6)


def test_native_3d_trace2cp_lasagna_normal_sampler_uses_geometry_record() -> None:
    fiber = Vc3dFiber(
        path=Path("same.json"),
        version=1,
        line_points_xyz=np.asarray(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        control_points_xyz=np.asarray(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        generation=1,
        metadata={},
    )
    trace_record = SimpleNamespace(
        fiber=fiber,
        volume_path="vol.zarr",
        volume_scale=2,
        volume_spacing_base=4.0,
    )
    normal_record = SimpleNamespace(
        fiber=fiber,
        volume_path="vol.zarr",
        volume_scale=2,
        volume_spacing_base=4.0,
        grad_mag=object(),
    )

    class FakeGeometryLoader:
        records = [normal_record]

        def __init__(self) -> None:
            self.used_record = None
            self.points_base = None

        def _lasagna_normals_at_zyx_batch(self, record, points_zyx_base, *, line_indices):
            self.used_record = record
            self.points_base = np.asarray(points_zyx_base, dtype=np.float64).copy()
            assert hasattr(record, "grad_mag")
            assert line_indices.tolist() == [0]
            return (
                np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                np.asarray([True]),
                {},
            )

    geometry_loader = FakeGeometryLoader()
    resolved = _native_trace_geometry_normal_record(geometry_loader, trace_record)
    sampler = _NativeLasagnaNormalSampler(
        geometry_loader=geometry_loader,
        trace_record=trace_record,
        normal_record=resolved,
    )

    normals_zyx, valid = sampler(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))

    assert resolved is normal_record
    assert geometry_loader.used_record is normal_record
    assert np.allclose(geometry_loader.points_base, [[4.0, 8.0, 12.0]])
    assert valid.tolist() == [True]
    assert torch.allclose(normals_zyx, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32))


def test_native_3d_trace2cp_current_point_direction_uses_best_branch() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_point_choices_torch(self, points_zyx: np.ndarray):
            assert int(np.asarray(points_zyx).shape[0]) == 1
            directions = torch.tensor(
                [
                    [
                        [0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            )
            presence = torch.tensor([[1.0, 0.75]], dtype=torch.float32)
            valid = torch.ones((1, 2), dtype=torch.bool)
            return directions, presence, valid

    direction, presence, valid = _sample_trace_point_aligned(
        FakeCache(),
        np.zeros((3,), dtype=np.float32),
        reference_direction_zyx=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )

    assert valid
    assert presence == pytest.approx(0.75)
    assert np.allclose(direction, [1.0, 0.0, 0.0])


def test_native_3d_trace2cp_candidate_smoothness_can_reject_branch_switch() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_points_torch(self, points_zyx: np.ndarray):
            count = int(np.asarray(points_zyx).shape[0])
            assert count == 2
            directions = torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
            presence = torch.ones((count,), dtype=torch.float32)
            valid = torch.ones((count,), dtype=torch.bool)
            return directions, presence, valid

    branch_direction = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    previous_direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    candidates = np.stack([branch_direction, previous_direction], axis=0)

    best_index_without_smoothness, *_ = _score_candidate_batch(
        FakeCache(),
        current_direction=branch_direction,
        previous_step_direction=previous_direction,
        candidate_directions=candidates,
        next_points=np.zeros((2, 3), dtype=np.float32),
        smoothness_weight=0.0,
    )
    best_index_with_smoothness, total_loss, *_rest = _score_candidate_batch(
        FakeCache(),
        current_direction=branch_direction,
        previous_step_direction=previous_direction,
        candidate_directions=candidates,
        next_points=np.zeros((2, 3), dtype=np.float32),
        smoothness_weight=2.0,
        smoothness_free_angle_degrees=0.0,
    )

    assert best_index_without_smoothness == 0
    assert best_index_with_smoothness == 1
    assert total_loss == pytest.approx(1.0, abs=1.0e-6)


def test_native_3d_trace2cp_fiber_line_tangent_uses_adjacent_segment_not_chord() -> None:
    fiber = Vc3dFiber(
        path=None,
        version=1,
        line_points_xyz=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 5.0],
            ],
            dtype=np.float32,
        ),
        control_points_xyz=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 5.0],
            ],
            dtype=np.float32,
        ),
        generation=1,
        metadata={},
    )
    record = SimpleNamespace(fiber=fiber, volume_spacing_base=2.0)

    forward = _fiber_line_tangent_zyx_toward_target(
        record,
        start_control_point_index=0,
        target_control_point_index=1,
    )
    reverse = _fiber_line_tangent_zyx_toward_target(
        record,
        start_control_point_index=1,
        target_control_point_index=0,
    )

    assert np.allclose(forward, [0.0, 0.0, 1.0])
    assert np.allclose(reverse, [-1.0, 0.0, 0.0])


def test_native_3d_trace2cp_first_step_uses_sampled_direction_aligned_to_line_tangent() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_point(self, point_zyx: np.ndarray):
            point = np.asarray(point_zyx, dtype=np.float32)
            if float(np.linalg.norm(point)) <= 1.0e-6:
                return (
                    np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                    1.0,
                    True,
                )
            return (
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                1.0,
                True,
            )

        def sample_points_torch(self, points_zyx: np.ndarray):
            points = np.asarray(points_zyx, dtype=np.float32)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            directions = points / np.maximum(norms, np.float32(1.0e-6))
            return (
                torch.as_tensor(directions, dtype=torch.float32),
                torch.ones((points.shape[0],), dtype=torch.float32),
                torch.ones((points.shape[0],), dtype=torch.bool),
            )

    result = trace_native_3d_one_way(
        FakeCache(),
        start_zyx=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        target_zyx=np.asarray([10.0, 0.0, 10.0], dtype=np.float32),
        initial_direction_zyx=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        cfg=NativeTrace2CpConfig(
            step_voxels=2.0,
            cone_angle_degrees=0.0,
            cone_grid_size=1,
            trace_step_limit=1,
        ),
    )

    assert result.reason == "trace_step_limit"
    assert np.allclose(result.trace_zyx[1], [0.0, 2.0, 0.0])


def test_native_3d_trace2cp_trace_paths_sample_candidate_normals() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_point_choices_torch(self, points_zyx: np.ndarray):
            count = int(np.asarray(points_zyx).shape[0])
            directions = torch.zeros((count, 1, 3), dtype=torch.float32)
            directions[:, 0, 2] = 1.0
            presence = torch.ones((count, 1), dtype=torch.float32)
            valid = torch.ones((count, 1), dtype=torch.bool)
            return directions, presence, valid

    def run_with_beam_width(width: int) -> list[np.ndarray]:
        calls: list[np.ndarray] = []

        def normal_sampler(points_zyx):
            points = np.asarray(points_zyx, dtype=np.float32)
            calls.append(points.copy())
            normals = np.zeros((points.shape[0], 3), dtype=np.float32)
            normals[:, 0] = 1.0
            valid = np.ones((points.shape[0],), dtype=bool)
            return normals, valid

        result = trace_native_3d_one_way(
            FakeCache(),
            start_zyx=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            target_zyx=np.asarray([0.0, 0.0, 4.0], dtype=np.float32),
            initial_direction_zyx=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            cfg=NativeTrace2CpConfig(
                step_voxels=1.0,
                cone_angle_degrees=0.0,
                cone_angle_step_degrees=5.0,
                beam_width=width,
                trace_step_limit=1,
                smoothness_weight=0.5,
            ),
            normal_sampler=normal_sampler,
        )
        assert result.reason == "trace_step_limit"
        return calls

    greedy_calls = run_with_beam_width(1)
    beam_calls = run_with_beam_width(2)

    assert len(greedy_calls) == 1
    assert greedy_calls[0].shape == (1, 3)
    assert np.allclose(greedy_calls[0][0], [0.0, 0.0, 1.0])
    assert len(beam_calls) == 1
    assert beam_calls[0].shape == (1, 3)
    assert np.allclose(beam_calls[0][0], [0.0, 0.0, 1.0])


def test_native_3d_trace2cp_trace_paths_apply_first_step_tangent_gate() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_point_choices_torch(self, points_zyx: np.ndarray):
            points = np.asarray(points_zyx, dtype=np.float32)
            directions = np.zeros((points.shape[0], 1, 3), dtype=np.float32)
            norms = np.linalg.norm(points, axis=1)
            nonzero = norms > 1.0e-6
            directions[nonzero, 0, :] = points[nonzero] / norms[nonzero, None]
            directions[~nonzero, 0, 0] = 1.0
            presence = np.full((points.shape[0], 1), 0.1, dtype=np.float32)
            presence[np.abs(points[:, 1] - 1.0) < 1.0e-4, 0] = 1.0
            valid = np.ones((points.shape[0], 1), dtype=bool)
            return (
                torch.as_tensor(directions, dtype=torch.float32),
                torch.as_tensor(presence, dtype=torch.float32),
                torch.as_tensor(valid, dtype=torch.bool),
            )

    def normal_sampler(points_zyx):
        points = np.asarray(points_zyx, dtype=np.float32)
        normals = np.zeros((points.shape[0], 3), dtype=np.float32)
        normals[:, 2] = 1.0
        valid = np.ones((points.shape[0],), dtype=bool)
        return normals, valid

    for beam_width in (1, 2):
        result = trace_native_3d_one_way(
            FakeCache(),
            start_zyx=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            target_zyx=np.asarray([0.0, 10.0, 0.0], dtype=np.float32),
            initial_direction_zyx=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            cfg=NativeTrace2CpConfig(
                step_voxels=1.0,
                cone_angle_degrees=90.0,
                cone_angle_step_degrees=90.0,
                beam_width=beam_width,
                beam_lookahead_steps=1,
                trace_step_limit=1,
                smoothness_weight=0.0,
                smoothness_tangent_weight=100.0,
                smoothness_normal_weight=0.0,
            ),
            normal_sampler=normal_sampler,
        )

        assert not result.reached_target_plane
        assert result.reason == "trace_step_limit"
        assert np.allclose(result.trace_zyx[-1], [1.0, 0.0, 0.0], atol=1.0e-5)


def test_native_3d_trace2cp_beam_lookahead_keeps_initially_worse_valid_path() -> None:
    class FakeCache:
        device = torch.device("cpu")

        def sample_point(self, point_zyx: np.ndarray):
            point = np.asarray(point_zyx, dtype=np.float32)
            radial_offset = float(np.linalg.norm(point[:2]))
            if float(point[2]) > 0.5 and (radial_offset < 0.1 or float(point[1]) < -0.1):
                return (
                    np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
                    0.0,
                    False,
                )
            return (
                np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
                1.0,
                True,
            )

        def sample_points_torch(self, points_zyx: np.ndarray):
            points = np.asarray(points_zyx, dtype=np.float32)
            count = int(points.shape[0])
            directions = np.zeros((count, 3), dtype=np.float32)
            directions[:, 2] = 1.0
            radial_offset = np.linalg.norm(points[:, :2], axis=1)
            presence = np.full((count,), 0.6, dtype=np.float32)
            presence = np.where(points[:, 1] > 0.1, 0.7, presence)
            presence = np.where(points[:, 1] < -0.1, 0.95, presence)
            presence = np.where(radial_offset < 0.1, 1.0, presence).astype(np.float32)
            valid = np.ones((count,), dtype=bool)
            return (
                torch.as_tensor(directions, dtype=torch.float32),
                torch.as_tensor(presence, dtype=torch.float32),
                torch.as_tensor(valid, dtype=torch.bool),
            )

    kwargs = dict(
        cache=FakeCache(),
        start_zyx=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        target_zyx=np.asarray([0.0, 0.0, 3.0], dtype=np.float32),
        initial_direction_zyx=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    greedy = trace_native_3d_one_way(
        **kwargs,
        cfg=NativeTrace2CpConfig(
            step_voxels=1.0,
            cone_angle_degrees=25.0,
            cone_angle_step_degrees=25.0,
            beam_width=1,
            trace_step_limit=6,
            smoothness_weight=0.0,
        ),
    )
    one_step_beam = trace_native_3d_one_way(
        **kwargs,
        cfg=NativeTrace2CpConfig(
            step_voxels=1.0,
            cone_angle_degrees=25.0,
            cone_angle_step_degrees=25.0,
            beam_width=2,
            beam_prune_distance_voxels=0.0,
            beam_lookahead_steps=1,
            trace_step_limit=6,
            smoothness_weight=0.0,
        ),
    )
    lookahead_beam = trace_native_3d_one_way(
        **kwargs,
        cfg=NativeTrace2CpConfig(
            step_voxels=1.0,
            cone_angle_degrees=25.0,
            cone_angle_step_degrees=25.0,
            beam_width=2,
            beam_prune_distance_voxels=0.0,
            beam_lookahead_steps=3,
            trace_step_limit=6,
            smoothness_weight=0.0,
        ),
    )

    assert not greedy.reached_target_plane
    assert greedy.reason == "invalid_current_point"
    assert not one_step_beam.reached_target_plane
    assert one_step_beam.reason == "all_candidates_invalid"
    assert lookahead_beam.reached_target_plane
    assert lookahead_beam.reason == "target_plane"
    assert np.max(np.linalg.norm(lookahead_beam.trace_zyx[:, :2], axis=1)) > 0.1


def test_native_3d_trace2cp_field_cache_rejects_nonblocking_sampler() -> None:
    class NonBlockingSampler:
        blocking = False

        def sample_coord_batch(
            self,
            coords_zyx_base: np.ndarray,
            valid_mask: np.ndarray,
        ) -> CoordinateSampleResult:
            del coords_zyx_base, valid_mask
            raise AssertionError("non-blocking sampler should be rejected before sampling")

    record = SimpleNamespace(
        volume=object(),
        sampler=NonBlockingSampler(),
        volume_spacing_base=1.0,
        base_shape_zyx=(32, 32, 32),
    )
    cache = NativeTraceFieldCache(
        record=record,
        model=torch.nn.Identity(),
        config=_loader(augment_enabled=False).config,
        image_normalization="none",
        patch_shape_zyx=(4, 4, 4),
        core_margin_voxels=1,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="blocking coordinate sampling"):
        cache._infer_block(np.asarray([0, 0, 0], dtype=np.int64))


def test_native_3d_trace2cp_constant_field_reaches_target_plane() -> None:
    encoded = encode_lasagna_direction_3x2(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    )[0]

    class ConstantNativeModel(torch.nn.Module):
        def forward(self, volume: torch.Tensor) -> torch.Tensor:
            b, _c, d, h, w = volume.shape
            out = torch.zeros((b, 7, d, h, w), dtype=torch.float32, device=volume.device)
            out[:, :6] = encoded.to(volume.device).view(1, 6, 1, 1, 1)
            out[:, 6] = 1.0
            return out

    record = _native_trace_record(np.ones((32, 32, 32), dtype=np.float32))
    cache = NativeTraceFieldCache(
        record=record,
        model=ConstantNativeModel(),
        config=_loader(augment_enabled=False).config,
        image_normalization="none",
        patch_shape_zyx=(16, 16, 16),
        core_margin_voxels=2,
        device=torch.device("cpu"),
    )
    result = trace_native_3d_pair(
        cache,
        start_zyx=np.asarray([8.0, 8.0, 4.0], dtype=np.float32),
        target_zyx=np.asarray([8.0, 8.0, 20.0], dtype=np.float32),
        forward_initial_direction_zyx=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        reverse_initial_direction_zyx=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        cfg=NativeTrace2CpConfig(
            step_voxels=2.0,
            cone_angle_degrees=25.0,
            cone_grid_size=5,
            beam_width=1,
            max_steps=20,
            inference_patch_shape_zyx=(16, 16, 16),
            core_margin_voxels=2,
        ),
    )

    assert result.forward.reached_target_plane
    assert result.reverse.reached_target_plane
    assert result.plane_error < 1.0e-6
    assert result.closest_target_error < 1.0e-6


def test_native_3d_trace2cp_trace_step_limit_stops_partial_trace() -> None:
    encoded = encode_lasagna_direction_3x2(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    )[0]

    class ConstantNativeModel(torch.nn.Module):
        def forward(self, volume: torch.Tensor) -> torch.Tensor:
            b, _c, d, h, w = volume.shape
            out = torch.zeros((b, 7, d, h, w), dtype=torch.float32, device=volume.device)
            out[:, :6] = encoded.to(volume.device).view(1, 6, 1, 1, 1)
            out[:, 6] = 1.0
            return out

    record = _native_trace_record(np.ones((32, 32, 32), dtype=np.float32))
    cache = NativeTraceFieldCache(
        record=record,
        model=ConstantNativeModel(),
        config=_loader(augment_enabled=False).config,
        image_normalization="none",
        patch_shape_zyx=(16, 16, 16),
        core_margin_voxels=2,
        device=torch.device("cpu"),
    )

    result = trace_native_3d_pair(
        cache,
        start_zyx=np.asarray([8.0, 8.0, 4.0], dtype=np.float32),
        target_zyx=np.asarray([8.0, 8.0, 20.0], dtype=np.float32),
        forward_initial_direction_zyx=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        reverse_initial_direction_zyx=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        cfg=NativeTrace2CpConfig(
            step_voxels=2.0,
            cone_angle_degrees=25.0,
            cone_grid_size=5,
            beam_width=1,
            max_steps=20,
            trace_step_limit=3,
            inference_patch_shape_zyx=(16, 16, 16),
            core_margin_voxels=2,
        ),
    )

    assert not result.forward.reached_target_plane
    assert result.forward.reason == "trace_step_limit"
    assert len(result.forward.steps) == 3
    assert not result.reverse.reached_target_plane
    assert result.reverse.reason == "trace_step_limit"
    assert len(result.reverse.steps) == 3


def test_native_3d_trace2cp_max_step_factor_limits_trace() -> None:
    encoded = encode_lasagna_direction_3x2(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    )[0]

    class ConstantNativeModel(torch.nn.Module):
        def forward(self, volume: torch.Tensor) -> torch.Tensor:
            b, _c, d, h, w = volume.shape
            out = torch.zeros((b, 7, d, h, w), dtype=torch.float32, device=volume.device)
            out[:, :6] = encoded.to(volume.device).view(1, 6, 1, 1, 1)
            out[:, 6] = 1.0
            return out

    record = _native_trace_record(np.ones((32, 32, 32), dtype=np.float32))
    cache = NativeTraceFieldCache(
        record=record,
        model=ConstantNativeModel(),
        config=_loader(augment_enabled=False).config,
        image_normalization="none",
        patch_shape_zyx=(16, 16, 16),
        core_margin_voxels=2,
        device=torch.device("cpu"),
    )

    result = trace_native_3d_pair(
        cache,
        start_zyx=np.asarray([8.0, 8.0, 4.0], dtype=np.float32),
        target_zyx=np.asarray([8.0, 8.0, 20.0], dtype=np.float32),
        forward_initial_direction_zyx=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        reverse_initial_direction_zyx=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        cfg=NativeTrace2CpConfig(
            step_voxels=2.0,
            cone_angle_degrees=25.0,
            cone_grid_size=5,
            beam_width=1,
            max_step_factor=0.25,
            inference_patch_shape_zyx=(16, 16, 16),
            core_margin_voxels=2,
        ),
    )

    assert not result.forward.reached_target_plane
    assert result.forward.reason == "max_step_factor"
    assert len(result.forward.steps) == 2
    assert not result.reverse.reached_target_plane
    assert result.reverse.reason == "max_step_factor"
    assert len(result.reverse.steps) == 2
