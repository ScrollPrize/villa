from collections import Counter
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_trace.fiber_json import Vc3dFiber
from vesuvius.neural_tracing.fiber_trace_3d.arc_sampling import FiberLocation, ScrollArcSampler
from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    FiberTrace3DLoader, config_from_mapping, load_config,
    _TARGET_MODE_CP_ONLY, _TARGET_MODE_DENSE_LINE,
)
from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import MacroBatchTargetSource
from vesuvius.neural_tracing.fiber_trace_3d.train import (
    compute_losses,
    _make_test_loader_raw_config, _resolve_dense_test_selection,
)


def fiber(points, cps=None, source=None):
    points = np.asarray(points, dtype=np.float32)
    cps = points[:1] if cps is None else np.asarray(cps, dtype=np.float32)
    return Vc3dFiber(None, 1, points, cps, (None,) * len(cps), 0,
                     {} if source is None else {"source_format": source})


def record(scroll, points, cps=None):
    return SimpleNamespace(dataset_config={"scroll_id": scroll}, fiber=fiber(points, cps))


def test_scroll_and_arc_length_probabilities_independent_of_vertex_and_cp_density():
    records = [record("A", [[0, 0, 0], [10, 0, 0]]),
               record("A", [[0, 1, 0], [30, 1, 0]]),
               record("B", [[i, 2, 0] for i in range(11)], [[i, 2, 0] for i in range(11)]),
               record("B", [[0, 3, 0], [10, 3, 0]])]
    sampler = ScrollArcSampler(records)
    rng = np.random.default_rng(71)
    counts, fractions = Counter(), []
    for _ in range(24000):
        _, index, location = sampler.sample(rng)
        counts[index] += 1
        fractions.append(location.fraction)
    assert abs((counts[0] + counts[1]) / 24000 - 0.5) < 0.015
    assert abs(counts[1] / (counts[0] + counts[1]) - 0.75) < 0.02
    assert abs(counts[2] / (counts[2] + counts[3]) - 0.5) < 0.02
    assert abs(np.mean(fractions) - 0.5) < 0.02
    assert all(0 <= f < 1 for f in fractions)


def test_duplicate_vertices_have_no_sampling_mass():
    sampler = ScrollArcSampler([record("A", [[0, 0, 0], [0, 0, 0], [10, 0, 0], [10, 0, 0]])])
    rng = np.random.default_rng(4)
    for _ in range(100):
        _, _, location = sampler.sample(rng)
        assert location.segment_index == 1
        assert location.xyz[0] == pytest.approx(10 * location.fraction)


def test_group_weights_balance_paris_json_nml_without_length_bias():
    records = [record(scroll, [[0, 0, 0], [10, 0, 0]]) for scroll in ('A','B','C','D')]
    # Two JSON directories belong to ONE group; radically different NML length.
    for group, length in [('json', 10), ('json', 30), ('nml', 1000)]:
        row = record('Paris4', [[0, 0, 0], [length, 0, 0]])
        row.dataset_config.update(sampling_group=group, sampling_group_weight=1)
        records.append(row)
    sampler = ScrollArcSampler(records)
    rng = np.random.default_rng(42)
    counts, fractions = Counter(), []
    for _ in range(50000):
        _, i, location = sampler.sample(rng)
        counts[i] += 1
        fractions.append(location.fraction)
    for i in range(4):
        assert counts[i] / 50000 == pytest.approx(.2, abs=.01)
    assert (counts[4] + counts[5]) / 50000 == pytest.approx(.1, abs=.01)
    assert counts[6] / 50000 == pytest.approx(.1, abs=.01)
    assert counts[5] / (counts[4] + counts[5]) == pytest.approx(.75, abs=.025)
    assert np.mean(fractions) == pytest.approx(.5, abs=.01)


@pytest.mark.parametrize('bad', [None, '', 0])
def test_grouped_scroll_rejects_missing_group(bad):
    rows = [record('A', [[0,0,0], [10,0,0]]) for _ in range(2)]
    rows[0].dataset_config.update(sampling_group='json', sampling_group_weight=1)
    rows[1].dataset_config.update(sampling_group=bad, sampling_group_weight=1)
    with pytest.raises(ValueError, match='sampling_group'):
        ScrollArcSampler(rows)


@pytest.mark.parametrize('weight', [None, 0, -1, float('nan'), float('inf'), True, '1'])
def test_group_weight_validation(weight):
    row = record('A', [[0,0,0], [10,0,0]])
    row.dataset_config.update(sampling_group='json', sampling_group_weight=weight)
    with pytest.raises(ValueError, match='positive and finite'):
        ScrollArcSampler([row])


def test_default_sampling_keeps_original_random_draws_and_distances():
    rows = [record('A', [[0,0,0], [10,0,0]]), record('A', [[0,1,0], [30,1,0]]),
            record('B', [[0,2,0], [5,2,0]])]
    sampler = ScrollArcSampler(rows)
    old_rng, new_rng = np.random.default_rng(72), np.random.default_rng(72)
    for _ in range(1000):
        scroll = int(old_rng.integers(2))
        distance = old_rng.random() * (40 if scroll == 0 else 5)
        index = (0 if distance < 10 else 1) if scroll == 0 else 2
        distance -= 10 if index == 1 else 0
        _, actual, location = sampler.sample(new_rng)
        assert actual == index
        assert location.fraction == distance / (10,30,5)[index]
    assert old_rng.random() == new_rng.random()


def test_same_group_disagrees_on_weight():
    rows = [record('A', [[0,0,0], [10,0,0]]) for _ in range(2)]
    for row, weight in zip(rows, [1,2]):
        row.dataset_config.update(sampling_group='json', sampling_group_weight=weight)
    with pytest.raises(ValueError, match='agree'):
        ScrollArcSampler(rows)


@pytest.mark.parametrize("points", [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [float('nan'), 0, 0]]])
def test_invalid_lengths_fail(points):
    with pytest.raises(ValueError, match="positive finite"):
        ScrollArcSampler([record("A", points)])


def make_loader(*, dense=True, balanced=True, extra=None, source=None, points=None):
    line = [[16, 32, 32], [32, 32, 32], [48, 32, 32]] if points is None else points
    records = [{"volume": np.zeros((64, 64, 64), np.uint8),
                "fiber": fiber(line, [[32, 32, 32]], source),
                "scroll_id": "A", "dense_supervision": dense}]
    if extra:
        records.extend(extra)
    return FiberTrace3DLoader(config_from_mapping({
        "_array_records": records, "sampling_strategy": "scroll_balanced_arclength" if balanced else "control_points",
        "seed": 11, "batch_size": 1, "patch_shape_zyx": [16, 16, 16],
        "cp_margin_voxels": 3, "presence_radius_voxels": 1,
        "image_normalization": "none", "augment_enabled": False,
        "round_source_to_chunk_boundaries": False, "geometry_backend": "eager",
        "value_augmentation_backend": "torch",
    }))


def test_global_index_replay_and_flat_compatibility():
    loader = make_loader()
    expected = [loader.descriptor_for_sample_index(i)[2] for i in range(100)]
    for i in np.random.default_rng(1).permutation(100):
        assert loader.descriptor_for_sample_index(int(i))[2] == expected[i]
    other = make_loader()
    assert [other.descriptor_for_sample_index(i)[2] for i in range(100)] == expected
    assert loader.descriptor_for_sample_index(0, sample_mode="flat")[2] == 0
    assert any(16 < p.xyz[0] < 48 and abs(p.xyz[0] - 32) > 1 for p in expected)


def test_continuous_location_drives_loading_and_prefetch():
    loader = make_loader()
    rec, _, location = loader.descriptor_for_sample_index(3)
    params = loader._sample_augment_params(rec, location, 3)
    assert np.allclose(params.cp_volume_zyx, location.xyz[::-1])
    lo, hi = loader._prefetch_source_bbox(rec, location)
    actual_lo, actual_hi = loader._actual_source_bbox(rec, params)
    assert np.all(lo <= actual_lo) and np.all(hi >= actual_hi)
    sample = loader.load_sample(3)
    assert sample.control_point_index == -1
    assert torch.allclose(sample.crop_origin_zyx, torch.tensor(np.floor(params.cp_volume_zyx - params.cp_local_zyx), dtype=torch.float32))


def test_bounded_location_preserves_unbounded_augmentation_stream():
    loader = make_loader()
    loader.config = replace(loader.config, augment_enabled=True, augment_brightness=0.2)
    first = loader.load_raw_sample(0, sample_index_limit=1)
    later = loader.load_raw_sample(5, sample_index_limit=1)
    assert first.sample.data_index == later.sample.data_index == 0
    assert first.sample.stream_index == 0 and later.sample.stream_index == 5
    assert np.array_equal(first.augmentation.cp_volume_zyx, later.augmentation.cp_volume_zyx)
    assert first.augmentation.brightness != later.augmentation.brightness


def test_length_probabilities_do_not_depend_on_volume_level():
    records = [record("A", [[0, 0, 0], [10, 0, 0]]), record("A", [[0, 0, 0], [10, 0, 0]])]
    records[0].volume_spacing_base = 1
    records[1].volume_spacing_base = 8
    sampler = ScrollArcSampler(records)
    assert np.array_equal(sampler.fiber_cdfs[0], [10, 20])


@pytest.mark.parametrize("stride", [1, 4])
def test_dense_json_supervises_only_selected_line_and_matches_nml(stride):
    extra = {"volume": np.zeros((64, 64, 64), np.uint8),
             "fiber": fiber([[16, 38, 32], [48, 38, 32]]),
             "scroll_id": "A", "dense_supervision": True}
    loader = make_loader(extra=[extra])
    raw = loader.load_batch(0, sample_mode="flat")
    assert raw.target_modes.item() == _TARGET_MODE_DENSE_LINE
    batch = materialize_targets(raw, loader.config, output_stride=stride)
    nml = make_loader(dense=False, balanced=False, source="nml")
    reference = materialize_targets(nml.load_batch(0, sample_mode="flat"), nml.config, output_stride=stride)
    assert torch.equal(batch.presence_target, reference.presence_target)
    assert torch.equal(batch.direction_indices_bzyx, reference.direction_indices_bzyx)
    assert batch.presence_target.sum() >= 16 // stride
    assert batch.presence_target.sum() < (16 // stride) ** 2


def _gaussian_line_batch(*, stride=1):
    loader = make_loader()
    raw = loader.load_batch(0, sample_mode="flat")
    shift = torch.round(raw.cp_local_zyx[0]) - raw.cp_local_zyx[0]
    normals = torch.zeros_like(raw.target_segment_starts_zyx)
    normals[:, 0] = 1.0  # Surface-normal thickness is output Z.
    shape = torch.tensor(loader.config.patch_shape_zyx)
    raw = replace(
        raw,
        cp_local_zyx=raw.cp_local_zyx + shift,
        target_segment_starts_zyx=raw.target_segment_starts_zyx + shift,
        target_segment_ends_zyx=raw.target_segment_ends_zyx + shift,
        target_segment_normals_zyx=normals,
        target_segment_bbox_lo_zyx=torch.zeros_like(raw.target_segment_bbox_lo_zyx),
        target_segment_bbox_hi_zyx=shape.view(1, 3).expand_as(raw.target_segment_bbox_hi_zyx),
        presence_target_mode="anisotropic_gaussian",
    )
    config = replace(
        loader.config,
        presence_target_mode="anisotropic_gaussian",
        presence_radius_voxels=1.0,
        presence_gaussian_width_ratio=4.0,
        presence_gaussian_truncate_sigma=3.0,
    )
    return raw, config, materialize_targets(raw, config, output_stride=stride)


def test_anisotropic_gaussian_presence_has_four_to_one_local_sigma():
    stride = 1
    raw, config, batch = _gaussian_line_batch(stride=stride)
    target = batch.presence_target[0, 0]
    center = torch.round(raw.cp_local_zyx[0] / stride).long()
    assert target[tuple(center)] == pytest.approx(1.0)
    thickness_step = max(1, round(config.presence_radius_voxels / stride))
    width_step = max(1, round(4 * config.presence_radius_voxels / stride))
    z = center.clone(); z[0] += thickness_step
    y = center.clone(); y[1] += width_step
    assert target[tuple(z)] == pytest.approx(target[tuple(y)], abs=2.0e-5)
    assert 0.4 < float(target[tuple(z)]) < 0.8
    support_indices = torch.nonzero(target > 0.0, as_tuple=False)
    assert torch.equal(batch.direction_indices_bzyx[:, 1:], support_indices)
    assert int(batch.direction_indices_bzyx.shape[0]) > int(
        materialize_targets(
            raw, replace(config, presence_target_mode="binary"), output_stride=stride
        ).direction_indices_bzyx.shape[0]
    )
    center_row = torch.nonzero(
        (batch.direction_indices_bzyx[:, 1:] == center).all(dim=1), as_tuple=False
    ).item()
    z_row = torch.nonzero(
        (batch.direction_indices_bzyx[:, 1:] == z).all(dim=1), as_tuple=False
    ).item()
    center_weight = batch.direction_weight_sparse[center_row].sum()
    z_weight = batch.direction_weight_sparse[z_row].sum()
    assert z_weight / center_weight == pytest.approx(float(target[tuple(z)]), abs=2.0e-5)


def test_anisotropic_gaussian_uses_input_voxel_units_at_stride_four():
    raw, config, batch = _gaussian_line_batch(stride=4)
    target = batch.presence_target[0, 0]
    center = torch.round(raw.cp_local_zyx[0] / 4).long()
    width = center.clone(); width[1] += 1
    thickness = center.clone(); thickness[0] += 1
    assert target[tuple(center)] == pytest.approx(1.0)
    assert target[tuple(width)] == pytest.approx(math.exp(-0.5), abs=2.0e-5)
    assert target[tuple(thickness)] == 0.0


def test_anisotropic_gaussian_tail_is_soft_positive_and_cutoff_is_zero():
    _raw, _config, batch = _gaussian_line_batch()
    center = torch.round(batch.cp_local_zyx[0]).long()
    tail = center.clone(); tail[0] += 2
    outside = center.clone(); outside[0] += 4
    assert 0.0 < float(batch.presence_target[(0, 0, *tail)]) < 0.5
    assert batch.presence_target[(0, 0, *outside)] == 0.0
    output = torch.full((1, 7, *batch.presence_target.shape[-3:]), 0.5, requires_grad=True)
    output.data[(0, 6, *tail)] = 0.01
    loss = compute_losses(output, batch, direction_weight=0, presence_weight=1)["total"]
    loss.backward()
    assert output.grad[(0, 6, *tail)] < 0  # Soft tail pulls upward, not toward hard zero.


def test_anisotropic_gaussian_normal_sign_is_irrelevant():
    raw, config, positive = _gaussian_line_batch()
    negative = materialize_targets(
        replace(raw, target_segment_normals_zyx=-raw.target_segment_normals_zyx),
        config,
    )
    torch.testing.assert_close(positive.presence_target, negative.presence_target, rtol=0, atol=0)


def test_anisotropic_gaussian_target_prepass_counts_match_loss_groups():
    raw, config, batch = _gaussian_line_batch()
    source = MacroBatchTargetSource(
        None,
        config=config,
        device=torch.device("cpu"),
        output_stride=1,
        batch_sizes=[1],
    )
    counts = source._counts(raw)
    output = torch.full((1, 7, *batch.presence_target.shape[-3:]), 0.5)
    statistics = compute_losses(
        output,
        batch,
        direction_weight=1,
        presence_weight=1,
        _return_statistics=True,
    )
    actual = torch.stack((
        statistics.direction_count,
        statistics.positive_presence_count,
        statistics.negative_presence_count,
        statistics.distribution_presence_count,
    )).cpu()
    torch.testing.assert_close(counts, actual, rtol=0, atol=0)


def test_anisotropic_gaussian_sidebands_cover_both_sides_outside_support():
    raw, config, _batch = _gaussian_line_batch()
    raw = replace(raw, volume_spacing_base=torch.tensor([2.0]))
    config = replace(
        config,
        presence_gaussian_width_ratio=1.0,
        presence_sideband_weight=0.1,
        presence_sideband_half_width_base=12.0,
        presence_sideband_half_height_base=8.0,
    )
    batch = materialize_targets(raw, config)
    assert batch.presence_sideband_mask is not None
    left, right = batch.presence_sideband_mask[0]
    assert left.any() and right.any()
    assert not (left & right).any()
    support = batch.presence_target[0, 0] > 0.0
    assert not (left & support).any()
    assert not (right & support).any()

    center = torch.round(raw.cp_local_zyx[0]).long()
    # The synthetic line's in-sheet transverse direction is output Y. At
    # scale 1, 12 base voxels span six input voxels on either side.
    centerline_left = torch.nonzero(
        left[center[0], :, center[2]], as_tuple=False
    ).flatten()
    centerline_right = torch.nonzero(
        right[center[0], :, center[2]], as_tuple=False
    ).flatten()
    assert int(centerline_left.min()) == int(center[1] - 6)
    assert int(centerline_right.max()) == int(center[1] + 6)
    assert int(centerline_left.max()) < int(center[1] - 3)
    assert int(centerline_right.min()) > int(center[1] + 3)


def test_sideband_loss_is_separately_normalized_per_side():
    raw, config, _batch = _gaussian_line_batch()
    config = replace(
        config,
        presence_gaussian_width_ratio=1.0,
        presence_sideband_weight=0.1,
        presence_sideband_half_width_base=6.0,
        presence_sideband_half_height_base=4.0,
    )
    batch = materialize_targets(raw, config)
    assert batch.presence_sideband_mask is not None
    output = torch.full((1, 7, *batch.presence_target.shape[-3:]), 0.5)
    output[:, 6][batch.presence_sideband_mask[:, 0]] = 0.8
    output[:, 6][batch.presence_sideband_mask[:, 1]] = 0.2
    losses = compute_losses(output, batch, direction_weight=0, presence_weight=1)
    assert losses["presence_sideband_left"] == pytest.approx(-math.log(0.2), abs=1e-5)
    assert losses["presence_sideband_right"] == pytest.approx(-math.log(0.8), abs=1e-5)

    source = MacroBatchTargetSource(
        None, config=config, device=torch.device("cpu"), output_stride=1, batch_sizes=[1]
    )
    counts = source._counts(raw)
    statistics = compute_losses(
        output, batch, direction_weight=1, presence_weight=1, _return_statistics=True
    )
    actual = torch.stack((
        statistics.direction_count,
        statistics.positive_presence_count,
        statistics.negative_presence_count,
        statistics.distribution_presence_count,
        statistics.sideband_left_presence_count,
        statistics.sideband_right_presence_count,
    )).cpu()
    torch.testing.assert_close(counts, actual, rtol=0, atol=0)


def test_whole_selected_line_reentering_patch_remains_positive():
    line = [[16, 32, 32], [32, 32, 32], [1000, 32, 32], [1000, 36, 32], [16, 36, 32]]
    loader = make_loader(points=line, balanced=False)
    batch = materialize_targets(loader.load_batch(0, sample_mode="flat"), loader.config)
    cp = torch.round(batch.cp_local_zyx[0]).long()
    assert batch.presence_target[0, 0, cp[0], cp[1] + 4, cp[2]] == 1


def test_legacy_json_default_and_dense_switch_validation():
    loader = make_loader(dense=False, balanced=False)
    assert loader.load_batch(0, sample_mode="flat").target_modes.item() == _TARGET_MODE_CP_ONLY
    with pytest.raises(ValueError, match="requires dense"):
        make_loader(dense=False)
    with pytest.raises(ValueError, match="boolean"):
        make_loader(dense="yes")
    with pytest.raises(ValueError, match="scroll_id"):
        ScrollArcSampler([record("", [[0, 0, 0], [1, 0, 0]])])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_dense_collection_forward_backward(dtype):
    if os.environ.get("FIBER_COOPERATIVE_CUDA_TEST") != "1":
        pytest.skip("requires manager-authorized CUDA test launcher")
    from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
    from vesuvius.neural_tracing.fiber_trace_3d.train import compute_losses

    torch.cuda.set_per_process_memory_fraction(
        4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.manual_seed(11)
    raw_config = {"model": {"features_per_stage": [4, 8, 16],
                             "normalization": "group", "output_stride": 4}}
    loader = make_loader()
    loader.config = replace(loader.config, patch_shape_zyx=(128, 128, 128))
    raw = loader.load_batch(3)
    assert raw.control_point_indices.item() == -1
    raw.volume.copy_(torch.randn_like(raw.volume))
    cpu = materialize_targets(raw, loader.config, output_stride=4)
    gpu = materialize_targets(raw.to("cuda:0"), loader.config, output_stride=4)
    assert torch.equal(cpu.presence_target, gpu.presence_target.cpu())
    assert torch.equal(cpu.direction_indices_bzyx, gpu.direction_indices_bzyx.cpu())
    torch.testing.assert_close(cpu.direction_target_sparse, gpu.direction_target_sparse.cpu())
    assert gpu.direction_indices_bzyx.shape[0] > 1
    model = build_fiber_trace_3d_model(raw_config).cuda()
    before = next(model.parameters()).detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        prediction = model(gpu.volume)
        losses = compute_losses(prediction, gpu, direction_weight=10, presence_weight=1,
                                branch_selection_mode="train_offset_grid_min_fraction")
    assert all(torch.isfinite(v).all() for v in losses.values())
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g).item() for g in grads)
    optimizer.step()
    assert not torch.equal(before, next(model.parameters()).detach())
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        evaluation = compute_losses(model(gpu.volume), gpu, direction_weight=10, presence_weight=1)
        repeated = compute_losses(model(gpu.volume), gpu, direction_weight=10, presence_weight=1)
    assert all(torch.isfinite(v).all() for v in evaluation.values())
    assert all(not v.requires_grad for v in evaluation.values())
    for key in evaluation:
        torch.testing.assert_close(evaluation[key], repeated[key], rtol=0, atol=0)
    print(f"DENSE_CUDA_CHECK device=1GPU batch=1 patch=128 stride=4 dtype={dtype} "
          f"train_loss={losses['total'].item():.6f} eval_loss={evaluation['total'].item():.6f}")
