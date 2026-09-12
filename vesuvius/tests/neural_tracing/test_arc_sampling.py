from collections import Counter
from dataclasses import replace
import json
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
from vesuvius.neural_tracing.fiber_trace_3d.train import (
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


def test_collection_train_test_config_and_fixed_test_selection():
    path = Path(__file__).resolve().parents[2] / "src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_collection_20260909_5scroll_16x8.json"
    raw = json.loads(path.read_text())
    assert load_config(path).sampling_strategy == "scroll_balanced_arclength"
    for cfg in (raw, _make_test_loader_raw_config(raw, raw["training"])):
        parsed = config_from_mapping(cfg)
        assert parsed.sampling_strategy == "scroll_balanced_arclength"
        assert {d["scroll_id"] for d in parsed.datasets} == {"PHerc0139", "PHerc0332", "PHerc1299", "PHerc1451", "PHercParis4"}
        assert all(d["dense_supervision"] for d in parsed.datasets)
    assert not _make_test_loader_raw_config(raw, raw["training"])["augment_enabled"]
    assert _resolve_dense_test_selection(raw["training"], loader_sample_count=1942, default_count=512) == (1942, 0, "random")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_dense_collection_forward_backward(dtype):
    if os.environ.get("FIBER_COOPERATIVE_CUDA_TEST") != "1":
        pytest.skip("requires manager-authorized CUDA test launcher")
    from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
    from vesuvius.neural_tracing.fiber_trace_3d.train import compute_losses

    torch.cuda.set_per_process_memory_fraction(
        4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.manual_seed(11)
    path = Path(__file__).resolve().parents[2] / "src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_collection_20260909_5scroll_16x8.json"
    raw_config = json.loads(path.read_text())
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
