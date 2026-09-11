"""CPU-only embedding contracts, including the effective-batch training path."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.embedding import (
    EmbeddingConfig, EmbeddingGeometry, build_eligibility, embedding_statistics,
    nearest_polyline_points, outside_normal_ellipse, sample_trace_points,
)
from vesuvius.neural_tracing.fiber_trace_3d.gradient_accumulation import ComponentGradientAccumulator
from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    FiberTrace3DSharedTensorSlots, collate_fiber_trace_3d_samples,
)
from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model, presence_outputs
from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets
from vesuvius.neural_tracing.fiber_trace_3d.train import (
    _finalize_loss_statistics, _forward_loss_statistics, _forward_loss_tensors, _slice_batch,
)
from test_fiber_trace_3d import _loader_with_mapping


def test_pair_budget_is_linear_and_balanced():
    cfg = EmbeddingConfig(enabled=True)
    assert cfg.pairs((128, 128, 128)) == 65536
    assert cfg.pairs((256, 256, 256)) == 131072
    assert cfg.pairs((32, 32, 32)) == 16384
    assert cfg.pairs((129, 129, 129)) % 2 == 0


@pytest.mark.parametrize("values", [{"enabled": 1}, {"channels": 0}, {"pairs_at_128": 3},
                                   {"weight": float("nan")}, {"unknown": True}])
def test_invalid_config_rejected(values):
    with pytest.raises(ValueError):
        EmbeddingConfig.from_mapping({"embedding": values})


def test_ellipse_boundary_sign_and_tangent_plane():
    delta = np.array([[20, 0, 0], [20.01, 0, 0], [0, 80, 0], [0, 0, 80], [0, 0, 80.01],
                      [12, 64, 0], [12, 64.01, 0]])
    normal = np.tile([1, 0, 0], (len(delta), 1))
    expected = [False, True, False, False, True, False, True]
    assert outside_normal_ellipse(delta, normal).tolist() == expected
    assert outside_normal_ellipse(delta, -normal * 3).tolist() == expected
    with pytest.raises(ValueError, match="nonzero"):
        outside_normal_ellipse(delta, normal * 0)


def test_continuous_nearest_segment_not_nearest_vertex():
    nearest, distances = nearest_polyline_points([[50, 3, 0]], [[0, 0, 0], [100, 0, 0]])
    np.testing.assert_allclose(nearest, [[50, 0, 0]])
    np.testing.assert_allclose(distances, [9])


def test_midpoint_index_matches_scan_for_folded_and_long_segments():
    from vesuvius.neural_tracing.fiber_trace_3d.embedding import _nearest_by_scan
    rng = np.random.default_rng(99)
    for line in (rng.uniform(0, 30, (100, 3)), rng.uniform(0, 30, (7, 3))):
        points = rng.uniform(0, 30, (300, 3))
        expected, d = _nearest_by_scan(points, line)
        actual, actual_d = nearest_polyline_points(points, line)
        np.testing.assert_allclose(actual_d, d, atol=1e-10)
        np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_base_coordinate_exclusion_respects_scale_rotation_and_warp(stride):
    shape = (16, 16, 16)
    # A nonlinear inverse map, with axis swap and anisotropic scaling. This
    # test's scalar oracle evaluates the exact same source-space coordinates.
    def to_base(p):
        return np.column_stack([50 + 3 * p[:, 1], 50 + p[:, 0] * 4,
                                50 + 2 * p[:, 2] + np.sin(p[:, 0])])
    line = np.array([[50, 50, 0], [50, 50, 150]])
    def normals(p):
        assert np.all(p[:, 0] == 50)
        assert np.all(p[:, 1] == 50)
        return np.tile([1, 0, 0], (len(p), 1))
    actual = build_eligibility(shape, stride, to_base, line, normals, (200, 200, 200))
    for idx in np.ndindex(actual.shape):
        q = to_base(np.array([idx]) * stride)[0]
        expected = (q[0] - 50) ** 2 + (q[1] - 50) ** 2 / 16 > 400
        assert bool(actual[idx]) == expected


def test_arc_length_sampling_not_segment_count():
    start = torch.tensor([[0., 0, 0], [0, 0, 1]])
    end = torch.tensor([[0., 0, 1], [0, 0, 10]])
    points = sample_trace_points(start, end, 20000, torch.Generator().manual_seed(1), (2, 2, 11))
    assert float((points[:, 2] < 1).float().mean()) == pytest.approx(0.1, abs=0.01)
    with pytest.raises(ValueError, match="nonzero GT"):
        sample_trace_points(start, start, 1, torch.Generator(), (2, 2, 11))


def pair_batch(presence_shape=(2, 2, 3), pairs=65536):
    return SimpleNamespace(
        embedding_geometry=(EmbeddingGeometry(torch.ones(presence_shape, dtype=torch.bool), 57, pairs, 1, 1.),),
        target_segment_offsets=torch.tensor([0]), target_segment_counts=torch.tensor([1]),
        target_segment_starts_zyx=torch.tensor([[0., 0, 0]]),
        target_segment_ends_zyx=torch.tensor([[0., 0, 1]]))


def test_presence_sampling_linear_detached_and_zero_never_chosen(monkeypatch):
    batch = pair_batch()
    embeddings = torch.randn(1, 4, 2, 2, 3, requires_grad=True)
    presence = torch.zeros(1, 1, 2, 2, 3, requires_grad=True)
    with torch.no_grad():
        presence[0, 0, 1, 1, 1] = 0.1
        presence[0, 0, 1, 1, 2] = 1
    calls = []
    original = torch.multinomial
    def capture(weights, *args, **kwargs):
        result = original(weights, *args, **kwargs)
        calls.append((weights.detach().clone(), result))
        return result
    monkeypatch.setattr(torch, "multinomial", capture)
    pos, pc, neg, nc = embedding_statistics(embeddings, presence, batch)
    weights, chosen = calls[-1]
    assert weights[-1] / weights[-2] == 10
    assert set(chosen.tolist()) == {10, 11}
    assert (chosen == 11).sum() / (chosen == 10).sum() == pytest.approx(10, rel=0.06)
    assert pc == nc == 32768
    (pos / pc + neg / nc).backward()
    assert presence.grad is None
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()


@pytest.mark.parametrize("stream_index", [0, 1, 127])
def test_loader_seed_fits_torch_and_preserves_deterministic_pairs(stream_index):
    from vesuvius.neural_tracing.fiber_trace_3d.loader import _stable_seed
    seed = _stable_seed(0, stream_index, "embedding")
    assert seed > 2**64 - 1  # Exercise the production failure, not a small test seed.
    batch = pair_batch(pairs=128)
    original_geometry = batch.embedding_geometry[0]
    source = torch.arange(48, dtype=torch.float32).reshape(1, 4, 2, 2, 3) / 48
    presence = torch.full((1, 1, 2, 2, 3), 0.5)
    results = []
    for candidate in (seed, seed, seed % (2**63 - 1)):
        batch.embedding_geometry = (replace(original_geometry, seed=candidate),)
        embeddings = source.clone().requires_grad_()
        stats = embedding_statistics(embeddings, presence, batch)
        (stats[0] / stats[1] + stats[2] / stats[3]).backward()
        results.append((torch.stack(stats).detach(), embeddings.grad))
    for actual in results[1:]:
        for value, expected in zip(actual, results[0]):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_zero_negative_support_retains_positive_term():
    embedding = torch.randn(1, 4, 2, 2, 3, requires_grad=True)
    batch = pair_batch(pairs=128)
    a = embedding_statistics(embedding, torch.zeros(1, 1, 2, 2, 3), batch)
    b = embedding_statistics(embedding, torch.ones(1, 1, 2, 2, 3), batch)
    assert a[3] == 0 and a[2] == 0 and a[1] == 64
    torch.testing.assert_close(a[0], b[0])


def tiny_config():
    return {"patch_shape_zyx": [16, 16, 16], "embedding": {"enabled": True, "pairs_at_128": 512},
            "model_3d": {"features_per_stage": [8, 16], "normalization": "group", "output_stride": 2}}


def test_model_and_embedded_checkpoint_keep_legacy_output_contract(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.inference_adapter import _config_from_checkpoint
    config = tiny_config()
    model = build_fiber_trace_3d_model(config).eval()
    x = torch.randn(1, 1, 16, 16, 16)
    result = model(x, return_embeddings=True)
    assert result["prediction"].shape == (1, 7, 8, 8, 8)
    assert result["embeddings"].shape == (1, 16, 8, 8, 8)
    torch.testing.assert_close(model(x), result["prediction"])
    assert presence_outputs(model(x)).shape == (1, 1, 1, 8, 8, 8)
    path = tmp_path / "snapshot.pt"
    torch.save({"config": config, "model": model.state_dict()}, path)
    effective = _config_from_checkpoint({}, path)
    restored = build_fiber_trace_3d_model(effective).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x), model(x))


def training_batch():
    loader = _loader_with_mapping({"augment_enabled": False, "batch_size": 2})
    samples = [loader.load_sample(i) for i in range(2)]
    for i, sample in enumerate(samples):
        eligible = torch.ones(8, 8, 8, dtype=torch.bool)
        if i == 1:
            eligible.zero_()  # unequal support must not alter normalization
        samples[i] = replace(sample, embedding_geometry=EmbeddingGeometry(eligible, i + 11, 64, 2, 1.))
    batch = collate_fiber_trace_3d_samples(samples).to("cpu")
    return materialize_targets(batch, loader.config, output_stride=2)


def test_regular_and_bounded_embedding_gradients_match():
    torch.manual_seed(5)
    batch = training_batch()
    expected = build_fiber_trace_3d_model(tiny_config())
    actual = copy.deepcopy(expected)
    losses = _forward_loss_tensors(expected, batch, direction_weight=1.2, presence_weight=0.7, training_loss=True)
    losses["total"].backward()
    accumulator = ComponentGradientAccumulator(actual)
    assert len(accumulator.banks) == 5
    for i in range(2):
        statistics = _forward_loss_statistics(actual, _slice_batch(batch, i, i + 1))
        accumulator.add(statistics)
    total = accumulator.reduce_and_assign(direction_weight=1.2, presence_weight=0.7)
    reduced = _finalize_loss_statistics(total, direction_weight=1.2, presence_weight=0.7)
    for key in losses:
        torch.testing.assert_close(reduced[key], losses[key], atol=2e-5, rtol=2e-5)
    for a, b in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-5, rtol=3e-4)
    assert reduced["embedding_negative_pairs"] == 32
    legacy = build_fiber_trace_3d_model({**tiny_config(), "embedding": {"enabled": False}})
    assert len(ComponentGradientAccumulator(legacy).banks) == 3


def test_normals_loader_and_shared_transport(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import loader as module
    def channels(*args, **kwargs):
        shape = (64, 64, 64)
        return shape, np.ones(shape), 1., np.full(shape, 128), np.full(shape, 128), 1., 1.
    monkeypatch.setattr(module, "_open_manifest_channels", channels)
    loader = _loader_with_mapping({"augment_enabled": False, "embedding": {"enabled": True},
                                   "model_3d": {"output_stride": 4},
                                   "datasets": [{"lasagna_manifest_path": "synthetic.json"}]})
    raw = loader.load_raw_sample(2)
    assert raw.sample.embedding_geometry.eligible.shape == (4, 4, 4)
    slots = FiberTrace3DSharedTensorSlots(start_index=2, slot_count=1, patch_shape_zyx=(16, 16, 16), embedding_stride=4)
    payload = slots.encode(2, raw, final=False)
    assert payload.value.sample.embedding_geometry.eligible is None
    restored = slots.decode(2, payload, final=False)
    assert torch.equal(restored.sample.embedding_geometry.eligible, raw.sample.embedding_geometry.eligible)
    assert restored.sample.embedding_geometry.seed == raw.sample.embedding_geometry.seed


def test_enabled_loader_requires_normals():
    with pytest.raises(ValueError, match="lasagna_manifest_path"):
        _loader_with_mapping({"embedding": {"enabled": True}})


def test_invalid_normals_fail_instead_of_spherical_fallback(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import loader as module
    monkeypatch.setattr(module, "_open_manifest_channels", lambda *a, **kw: (
        (64, 64, 64), np.zeros((64, 64, 64)), 1.,
        np.full((64, 64, 64), 128), np.full((64, 64, 64), 128), 1., 1.))
    loader = _loader_with_mapping({"embedding": {"enabled": True}, "model_3d": {"output_stride": 4},
                                   "datasets": [{"lasagna_manifest_path": "synthetic.json"}]})
    with pytest.raises(ValueError, match="embedding exclusion.*grad_mag"):
        loader.load_raw_sample(0)


def test_real_manifest_scales_and_actual_augmentation(tmp_path, monkeypatch):
    import json
    from tests.zarr_utils import create_v2_array, create_v2_group_array
    from test_fiber_trace_2d_loader import _write_fiber
    from vesuvius.neural_tracing.fiber_trace_3d import loader as module
    from vesuvius.neural_tracing.fiber_trace_3d.embedding import _nearest_by_scan
    from vesuvius.neural_tracing.fiber_trace_2d.sampling import NumpyZarrCoordinateSampler
    volume = tmp_path / "volume.zarr"
    create_v2_group_array(volume, "0", shape=(64, 64, 64), chunks=(16, 16, 16), dtype="uint8")[:] = 80
    create_v2_array(volume / "1", shape=(32, 32, 32), chunks=(16, 16, 16), dtype="uint8")[:] = 80
    normals = create_v2_group_array(tmp_path / "normals.zarr", "2", shape=(3, 16, 16, 16),
                                    chunks=(1, 8, 8, 8), dtype="uint8")
    normals[0] = 1
    normals[1] = 128
    normals[2] = 128  # surface normal +Z in base coordinates
    manifest = tmp_path / "normal.lasagna.json"
    manifest.write_text(json.dumps({"version": 2, "source_to_base": 1., "base_shape_zyx": [64, 64, 64],
        "grad_mag_encode_scale": 1000., "grad_mag_factor": 1., "groups": {"normals": {
            "zarr": "normals.zarr", "scaledown": 2, "channels": ["grad_mag", "nx", "ny"]}}}))
    # Input annotations use a translation transform, distinct from augmentation.
    fiber = _write_fiber(tmp_path / "fiber.json", points=[[x, 27, 29] for x in range(10, 51, 2)])
    transform = np.eye(4)
    transform[:3, 3] = [0, 5, 3]
    monkeypatch.setattr(module, "make_coordinate_sampler", lambda **kw: NumpyZarrCoordinateSampler(
        kw["array"], level_spacing_base=kw["level_spacing_base"]))
    raw = {"patch_shape_zyx": [24, 24, 24], "embedding": {"enabled": True},
           "model_3d": {"output_stride": 2}, "geometry_backend": "eager", "seed": 73,
           "augment_enabled": True, "augment_rotation_degrees": 35,
           "augment_scale_min": 0.7, "augment_scale_max": 0.8,
           "augment_smooth_displacement_mode": "3d", "augment_smooth_displacement_probability": 1.,
           "augment_smooth_displacement_amplitude_zyx": [2., 2., 2.],
           "datasets": [{"base_volume_path": str(volume), "base_volume_scale": 1,
                         "lasagna_manifest_path": str(manifest), "fiber_paths": [str(fiber)],
                         "fiber_transform": transform.tolist()}]}
    loader = module.FiberTrace3DLoader(module.config_from_mapping(raw, config_dir=tmp_path))
    loaded = loader.load_raw_sample(0)
    record, record_index, cp = loader.descriptor_for_sample_index(0)
    assert record.volume_spacing_base == 2
    assert loader._embedding_normal_records[record_index].nx_spacing_base == 4
    np.testing.assert_allclose(record.fiber.line_points_zyx[:, :2], 32)
    params = loader._sample_augment_params(record, cp, 0)
    indices = np.indices((12, 12, 12)).reshape(3, -1).T
    source = loader._output_points_to_source_volume_np(indices * 2, params) * 2
    matched, _ = _nearest_by_scan(source, record.fiber.line_points_zyx)
    delta = source - matched
    expected = delta[:, 0] ** 2 + (delta[:, 1] ** 2 + delta[:, 2] ** 2) / 16 > 400
    expected &= ((source >= 0) & (source <= 63)).all(axis=1)
    assert expected.any() and not expected.all()
    np.testing.assert_array_equal(loaded.sample.embedding_geometry.eligible.numpy().reshape(-1), expected)


def test_state_only_checkpoint_does_not_infer_embedding_as_branches(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.inference_adapter import _config_from_checkpoint
    config = tiny_config()
    config["embedding"]["channels"] = 7
    model = build_fiber_trace_3d_model(config)
    path = tmp_path / "state.pt"
    torch.save(model.state_dict(), path)
    restored_config = _config_from_checkpoint(config, path)
    assert restored_config["model_3d"]["direction_branch_count"] == 1
    restored = build_fiber_trace_3d_model(restored_config)
    restored.load_state_dict(model.state_dict())


def _distributed_embedding_worker(rank, directory):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{directory}/init", rank=rank, world_size=3)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(8)
        model = build_fiber_trace_3d_model(tiny_config())
        oracle = copy.deepcopy(model)
        batch = training_batch()
        expected = _forward_loss_tensors(oracle, batch, direction_weight=1., presence_weight=1., training_loss=True)
        expected["total"].backward()
        accumulator = ComponentGradientAccumulator(model)
        # Rank 2 deliberately has no samples; rank 1 has no negative pairs.
        if rank < 2:
            accumulator.add(_forward_loss_statistics(model, _slice_batch(batch, rank, rank + 1)))
        statistics = accumulator.reduce_and_assign(direction_weight=1., presence_weight=1., distributed=True)
        for a, b in zip(model.parameters(), oracle.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=3e-5, rtol=3e-4)
        assert statistics.positive_embedding_count == 64
        assert statistics.negative_embedding_count == 32
    finally:
        dist.destroy_process_group()


def test_gloo_embedding_unequal_and_zero_work_ranks(tmp_path):
    torch.multiprocessing.spawn(_distributed_embedding_worker, args=(str(tmp_path),), nprocs=3)


def test_embedding_cpu_cost(capsys):
    """Small repeatable CPU benchmark, no GPUs and no performance assertion."""
    import json
    import time
    torch.set_num_threads(1)
    shape = (32, 32, 32)  # native quarter-resolution head for 128 input
    batch = pair_batch(shape)
    batch.target_segment_ends_zyx[0, 2] = 31
    embeddings = torch.randn(1, 16, *shape, requires_grad=True)
    presence = torch.rand(1, 1, *shape)
    times = []
    for i in range(4):
        embeddings.grad = None
        begin = time.perf_counter()
        pos, pc, neg, nc = embedding_statistics(embeddings, presence, batch)
        (0.5 * (pos / pc + neg / nc)).backward()
        if i:
            times.append(time.perf_counter() - begin)
    line = np.column_stack([np.full(129, 128), np.full(129, 128), np.arange(129) + 64])
    geometry_times = []
    for i in range(3):
        begin = time.perf_counter()
        build_eligibility((128, 128, 128), 4, lambda p: p + 64, line,
                          lambda p: np.tile([1., 0, 0], (len(p), 1)), (256, 256, 256))
        geometry_times.append(time.perf_counter() - begin)
    report = {"processing_units": "one CPU thread; no GPUs", "pairs": 65536,
              "embedding_channels": 16, "input_patch": 128, "output_stride": 4,
              "pair_forward_backward_seconds_min_median_max": [min(times), float(np.median(times)), max(times)],
              "exclusion_seconds_min_median_max": [min(geometry_times), float(np.median(geometry_times)), max(geometry_times)],
              "note": "synthetic CPU tensors; exclusion uses constant normals, excludes disk I/O"}
    with capsys.disabled():
        print("EMBEDDING_CPU_BENCH " + json.dumps(report), flush=True)
