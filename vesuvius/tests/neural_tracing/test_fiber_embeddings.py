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
    InvalidNormalSampleError,
)
from vesuvius.neural_tracing.fiber_trace_3d.gradient_accumulation import NormalizedGradientAccumulator, validate_target_only_loss_config
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import MacroBatchTargetSource
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
    assert a[3] == 64 and a[2] == 0 and a[1] == 64
    torch.testing.assert_close(a[0], b[0])


def test_native_normal_interpolation_and_validity(tmp_path, capsys):
    import json
    from vc.volume import NormalSampler
    from tests.zarr_utils import create_v2_group_array
    groups = {}
    arrays = {}
    for name, value in (("grad_mag", 1), ("nx", 255), ("ny", 128)):
        array = create_v2_group_array(tmp_path / (name+".zarr"), "0", shape=(8,8,8),
                                     chunks=(4,4,4), dtype="uint8")
        array[:4,:4,:4] = value
        if name == "nx":
            array[:4,:4,2:4] = 1  # +X to -X encoding jump, SAME unoriented axis
        if name == "grad_mag":
            array[1,1,1] = 0  # stored zero is as invalid as absent far chunk
        groups[name] = {"zarr": name+".zarr/0", "scaledown":0, "channels":[name]}
        arrays[name] = array
    manifest = tmp_path / "normals.lasagna.json"
    manifest.write_text(json.dumps({"version":2, "source_to_base":1,
        "base_shape_zyx":[8,8,8], "grad_mag_encode_scale":1000,
        "grad_mag_factor":.25, "groups":groups}))
    sampler = NormalSampler(str(manifest), str(tmp_path / "cache"))
    points = np.array([[1.5,2,2], [1,1,1], [6,6,6], [2,2,2]], dtype=np.float32)
    normals, valid = sampler.sample(points)
    assert valid.tolist() == [True, False, False, True]
    np.testing.assert_allclose(np.abs(normals[valid]), [[1,0,0], [1,0,0]], atol=1e-5)
    for point, expected, ok in zip(points, normals, valid):
        one, one_valid = sampler.sample(point[None])
        assert one_valid[0] == ok
        np.testing.assert_allclose(one[0], expected)
    empty, empty_valid = sampler.sample(np.empty((0,3), np.float32))
    assert empty.shape == (0,3) and empty_valid.shape == (0,)
    with pytest.raises(ValueError):
        sampler.sample(np.zeros((2,2), np.float32))
    with pytest.raises(Exception):
        NormalSampler(str(tmp_path / "nonexistent.json"))
    # Same warm-cache workload against the former Python/Zarr normal path.
    import time
    from vesuvius.neural_tracing.fiber_trace_2d.loader import FiberStrip2DLoader
    record = SimpleNamespace(**arrays, grad_mag_spacing_base=1., nx_spacing_base=1., ny_spacing_base=1.)
    queries = np.random.default_rng(11).uniform(.1, 2.9, (32768,3)).astype(np.float32)
    rows = {}
    for backend in ("native", "python_zarr"):
        durations = []
        for iteration in range(4):
            start = time.perf_counter()
            if backend == "native":
                axis, ok = sampler.sample(queries, threads=1)
            else:
                axis, ok, _ = FiberStrip2DLoader.sample_lasagna_normals_at_zyx_batch(
                    record, queries[:, ::-1], line_indices=np.arange(len(queries)))
            durations.append(time.perf_counter()-start)
            assert ok.all()
            np.testing.assert_allclose(np.abs(axis[:,0]), 1, atol=1e-5)
        measured = durations[1:]
        rows[backend] = {"seconds_min_median_max": [min(measured), float(np.median(measured)), max(measured)]}
    with capsys.disabled():
        print("NORMAL_SAMPLING_BENCH " + json.dumps({"points":32768, "cpu_sampling_threads":1,
            "dataset":"synthetic 8-cube; warm cache; sign-jump normals", "results":rows}), flush=True)


@pytest.mark.parametrize("mode", ["flat", "random"])
def test_whole_sample_replacement_deterministic_and_bounded(monkeypatch, mode):
    loader = _loader_with_mapping({"augment_enabled": False})
    calls = []
    def load(index, **kwargs):
        calls.append(kwargs["augmentation_index"])
        if len(calls) == 1:
            raise InvalidNormalSampleError("missing")
        return SimpleNamespace(sample=SimpleNamespace(stream_index=index),
                               descriptor=kwargs["descriptor"], seed=kwargs["augmentation_index"])
    monkeypatch.setattr(loader, "_load_raw_sample_once", load)
    first = loader.load_raw_sample(7, sample_mode=mode)
    calls.clear()
    second = loader.load_raw_sample(7, sample_mode=mode)
    assert first.sample.stream_index == second.sample.stream_index == 7
    assert first.descriptor == second.descriptor and first.seed == second.seed
    def invalid(*a, **kw): raise ValueError("present but invalid")
    monkeypatch.setattr(loader, "_load_raw_sample_once", invalid)
    with pytest.raises(ValueError, match="present but invalid"):
        loader.load_raw_sample(7)
    def io_error(*a, **kw): raise TimeoutError("network timeout")
    monkeypatch.setattr(loader, "_load_raw_sample_once", io_error)
    with pytest.raises(TimeoutError): loader.load_raw_sample(7)
    def missing(*a, **kw): raise InvalidNormalSampleError("missing")
    monkeypatch.setattr(loader, "_load_raw_sample_once", missing)
    with pytest.raises(RuntimeError, match="128 replacements"):
        loader.load_raw_sample(7)


def test_replacement_stays_in_scroll_and_group():
    from vesuvius.neural_tracing.fiber_trace_3d.arc_sampling import ScrollArcSampler
    loader = _loader_with_mapping({"augment_enabled": False})
    base = loader.records[0]
    loader.records = [replace(base, dataset_config={**base.dataset_config, "scroll_id":s,
        "sampling_group":g, "sampling_group_weight":1})
        for s,g in [("Paris4","json"),("Paris4","json"),("Paris4","nml"),("Other","json")]]
    loader._arc_sampler = ScrollArcSampler(loader.records)
    for mode in ("flat","random"):
        chosen = [loader._replacement_descriptor((loader.records[0],0,0), 0, i, mode)[1] for i in range(100)]
        assert set(chosen) == {0,1}


def test_embedding_prepass_owns_eligibility_after_shared_slot_release():
    batch = training_batch()
    parts = [_slice_batch(batch, i, i + 1) for i in range(2)]
    class Source:
        def __next__(self):
            self.current = parts.pop(0)
            return self.current
        def release_current(self):
            self.current.embedding_geometry[0].eligible.logical_not_()
        def close(self): pass
    config = replace(_loader_with_mapping({"augment_enabled": False}).config,
                     embedding=EmbeddingConfig(enabled=True))
    source = MacroBatchTargetSource(Source(), config=config, device=torch.device("cpu"),
                                   output_stride=2, batch_sizes=[1, 1])
    first, second = next(source), next(source)
    assert first.embedding_geometry[0].eligible.all()
    assert not second.embedding_geometry[0].eligible.any()
    torch.testing.assert_close(first.normalization_counts[-2:], torch.tensor([64.,64.]))


def test_combined_embedding_real_data_preflight():
    """Explicit host-only integration check; never fetch training data in CI."""
    import os
    import glob
    import json
    from pathlib import Path
    if os.environ.get("FIBER_EMBEDDING_DATA_TEST") != "1":
        pytest.skip("requires approved host-data test runner")
    from vesuvius.neural_tracing.fiber_trace_3d.loader import FiberTrace3DLoader, config_from_mapping
    path = Path(__file__).parents[2] / "src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_combined_5scroll_s1a_embedding_ema_16x8.json"
    raw = json.loads(path.read_text())
    assert raw["training"]["learning_rate"] == .01 and "resume" not in raw["training"]
    assert raw["training"]["max_steps"] == 2000000
    assert raw["sampling_strategy"] == "scroll_balanced_arclength"
    assert len({d["scroll_id"] for d in raw["datasets"]}) == 5
    paris = [d for d in raw["datasets"] if d["scroll_id"] == "PHercParis4"]
    assert {(d["sampling_group"], d["sampling_group_weight"]) for d in paris} == {("json",1),("nml",1)}
    torch.set_num_threads(1)
    for kind in ("datasets", "test_datasets", "historical_test_datasets"):
        for dataset in raw[kind]:
            files = sorted(glob.glob(dataset["fiber_glob"]))
            assert files, dataset
            entry = {k:v for k,v in dataset.items() if k != "fiber_glob"}
            entry["fiber_paths"] = files
            probe = {**raw, "datasets": [entry], "batch_size": 1,
                     "volume_io_threads": 1, "volume_decode_threads": 1, "volume_sample_threads": 1}
            if kind == "historical_test_datasets":
                probe["sampling_strategy"] = "control_points"
                probe["embedding"] = {"enabled": False}
            loader = FiberTrace3DLoader(config_from_mapping(probe, config_dir=path.parent))
            sample = loader.load_raw_sample(0, profile=True).sample
            if kind == "historical_test_datasets":
                assert sample.embedding_geometry is None
            else:
                assert sample.embedding_geometry.eligible.shape == (32,32,32)
            print(f"EMBEDDING_DATA_OK {kind} {dataset['scroll_id']} "
                  f"embedding_ms={sample.profile_timings_ms.get('embedding_geometry_ms')} "
                  f"replacements={loader._missing_normal_samples}", flush=True)
            if kind == "datasets" and dataset is raw[kind][0]:
                from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import (
                    FiberTrace3DSampleQueue, SampleQueueSource)
                import time
                queue = FiberTrace3DSampleQueue(sources={"train": SampleQueueSource(loader.config)},
                    workers=2, slot_count=2, initial_in_flight=2)
                try:
                    ticket = queue.submit("train", (0,1))
                    deadline = time.monotonic()+90
                    batch = None
                    while batch is None and time.monotonic() < deadline:
                        batch = queue.batch(ticket)
                        if batch is None: time.sleep(.02)
                    assert batch is not None, "queued embedding load timed out"
                    assert len(batch.embedding_geometry) == 2
                    assert all(g.eligible.shape == (32,32,32) for g in batch.embedding_geometry)
                    queue.release(ticket)
                    print("EMBEDDING_NATIVE_QUEUE_OK workers=2 samples=2", flush=True)
                finally:
                    queue.close()


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


def pair_prepass_counts(batch):
    loader = _loader_with_mapping({"augment_enabled": False, "batch_size": 2})
    config = replace(loader.config, embedding=EmbeddingConfig(enabled=True))
    source = MacroBatchTargetSource(None, config=config, device=batch.volume.device,
                                   output_stride=2, batch_sizes=[len(batch.volume)])
    return source._counts(batch)


@pytest.mark.parametrize('competition', [False, True])
def test_regular_and_bounded_embedding_gradients_match(monkeypatch, competition):
    torch.manual_seed(5)
    batch = training_batch()
    config = tiny_config()
    if competition:
        config['embedding']['objective'] = 'local_contrastive'
        batch = replace(batch, embedding_geometry=tuple(replace(g, objective='local_contrastive') for g in batch.embedding_geometry))
    expected = build_fiber_trace_3d_model(config)
    actual = copy.deepcopy(expected)
    losses = _forward_loss_tensors(expected, batch, direction_weight=1.2, presence_weight=0.7, training_loss=True)
    losses["total"].backward()
    validate_target_only_loss_config(tiny_config())
    accumulator = NormalizedGradientAccumulator(actual, pair_prepass_counts(batch),
                                               direction_weight=1.2, presence_weight=0.7)
    calls = []
    original = torch.Tensor.backward
    def backward(tensor, *args, **kwargs):
        calls.append(1)
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "backward", backward)
    for i in range(2):
        statistics = _forward_loss_statistics(actual, _slice_batch(batch, i, i + 1))
        accumulator.add(statistics)
    total = accumulator.reduce_and_assign()
    assert len(calls) == 2
    reduced = _finalize_loss_statistics(total, direction_weight=1.2, presence_weight=0.7)
    for key in losses:
        torch.testing.assert_close(reduced[key], losses[key], atol=2e-5, rtol=2e-5)
    for a, b in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-5, rtol=3e-4)
    assert reduced["embedding_negative_pairs"] == (0 if competition else 64)
    if competition:
        assert reduced['embedding_anchors'] == 4  # includes excluded patch
    legacy = build_fiber_trace_3d_model({**tiny_config(), "embedding": {"enabled": False}})
    assert len(NormalizedGradientAccumulator(legacy, torch.zeros(3),
               direction_weight=1, presence_weight=1).components) == 3


def test_normals_loader_and_shared_transport(monkeypatch):
    import vc.volume
    opened = []
    class Sampler:
        def __init__(self, *args): opened.append(args)
        def sample(self, xyz, threads):
            assert threads == 1
            return np.tile([0.,0.,1.], (len(xyz),1)), np.ones(len(xyz), bool)
    monkeypatch.setattr(vc.volume, "NormalSampler", Sampler)
    loader = _loader_with_mapping({"augment_enabled": False, "embedding": {"enabled": True},
                                   "model_3d": {"output_stride": 4},
                                   "datasets": [{"lasagna_manifest_path": "synthetic.json"}]})
    loader.records[0].dataset_config["lasagna_manifest_path"] = "synthetic.json"
    raw = loader.load_raw_sample(2)
    assert raw.sample.embedding_geometry.eligible.shape == (4, 4, 4)
    slots = FiberTrace3DSharedTensorSlots(start_index=2, slot_count=1, patch_shape_zyx=(16, 16, 16), embedding_stride=4)
    payload = slots.encode(2, raw, final=False)
    assert payload.value.sample.embedding_geometry.eligible is None
    restored = slots.decode(2, payload, final=False)
    assert torch.equal(restored.sample.embedding_geometry.eligible, raw.sample.embedding_geometry.eligible)
    assert restored.sample.embedding_geometry.seed == raw.sample.embedding_geometry.seed
    loader.load_raw_sample(3)
    assert len(opened) == 1  # manifest handle reused across samples


def test_enabled_loader_requires_normals():
    with pytest.raises(ValueError, match="lasagna_manifest_path"):
        _loader_with_mapping({"embedding": {"enabled": True}})


def test_invalid_normals_fail_instead_of_spherical_fallback(monkeypatch):
    import vc.volume
    class Sampler:
        def __init__(self, *args): pass
        def sample(self, xyz, threads):
            return np.zeros_like(xyz), np.zeros(len(xyz), bool)
    monkeypatch.setattr(vc.volume, "NormalSampler", Sampler)
    loader = _loader_with_mapping({"embedding": {"enabled": True}, "model_3d": {"output_stride": 4},
                                   "datasets": [{"lasagna_manifest_path": "synthetic.json"}]})
    loader.records[0].dataset_config["lasagna_manifest_path"] = "synthetic.json"
    with pytest.raises(RuntimeError, match="128 replacements"):
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
    groups = {}
    for channel, value in (("grad_mag", 1), ("nx", 128), ("ny", 128)):
        create_v2_group_array(tmp_path / (channel+".zarr"), "2", shape=(16,16,16),
                              chunks=(8,8,8), dtype="uint8")[:] = value
        groups[channel] = {"zarr": channel+".zarr/2", "scaledown":2, "channels":[channel]}
    manifest = tmp_path / "normal.lasagna.json"
    manifest.write_text(json.dumps({"version": 2, "source_to_base": 1., "base_shape_zyx": [64, 64, 64],
        "grad_mag_encode_scale": 1000., "grad_mag_factor": 1., "groups": groups}))
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
    assert str(manifest) in loader._embedding_normal_records
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


def _distributed_embedding_worker(rank, directory, backend="gloo", world_size=3, amp=False, competition=False):
    import torch.distributed as dist
    from contextlib import nullcontext
    from datetime import timedelta
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
        # Compare actual FP32 math, not shape-dependent TF32 convolution choices.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    dist.init_process_group(backend, init_method=f"file://{directory}/init", rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=60))
    try:
        torch.set_num_threads(1)
        torch.manual_seed(8)
        config = tiny_config()
        if competition:
            config['embedding']['objective'] = 'local_contrastive'
        model = build_fiber_trace_3d_model(config).to(device)
        oracle = copy.deepcopy(model)
        batch = training_batch().to(device)
        if competition:
            batch = replace(batch, embedding_geometry=tuple(replace(g, objective='local_contrastive') for g in batch.embedding_geometry))
        context = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
        with context():
            expected = _forward_loss_tensors(oracle, batch, direction_weight=1., presence_weight=1., training_loss=True)
        expected["total"].backward()
        counts = pair_prepass_counts(_slice_batch(batch, rank, rank + 1)) if rank < 2 else torch.zeros(6 if competition else 5)
        accumulator = NormalizedGradientAccumulator(model, counts, direction_weight=1.,
                                                    presence_weight=1., distributed=True)
        # Rank 2 has no samples; rank 1 has no eligible negatives but retains its budget.
        if rank < 2:
            with context():
                stats = _forward_loss_statistics(model, _slice_batch(batch, rank, rank + 1))
            accumulator.add(stats)
        statistics = accumulator.reduce_and_assign()
        for a, b in zip(model.parameters(), oracle.parameters()):
            if not amp:
                torch.testing.assert_close(a.grad, b.grad, atol=3e-5, rtol=3e-4)
        if amp:
            actual = torch.cat([p.grad.flatten() for p in model.parameters()])
            reference = torch.cat([p.grad.flatten() for p in oracle.parameters()])
            relative = (actual-reference).norm()/reference.norm()
            assert relative < .02, relative.item()
            if rank == 0:
                print(f"EMBEDDING_BF16_GRADIENT_RELATIVE_L2={relative.item():.6g}", flush=True)
        assert statistics.positive_embedding_count == (0 if competition else 64)
        assert statistics.negative_embedding_count == (0 if competition else 64)
        if competition:
            assert statistics.contrastive_embedding_count == 4
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('competition', [False, True])
def test_gloo_embedding_unequal_and_zero_work_ranks(tmp_path, competition):
    torch.multiprocessing.spawn(_distributed_embedding_worker, args=(str(tmp_path),'gloo',3,False,competition), nprocs=3)


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize('competition', [False, True])
def test_cuda_embedding_distributed_single_backward(tmp_path, amp, competition):
    import os
    if os.environ.get("FIBER_COOPERATIVE_CUDA_TEST") != "1":
        pytest.skip("requires manager-authorized CUDA test launcher")
    torch.multiprocessing.spawn(_distributed_embedding_worker,
        args=(str(tmp_path), "nccl", 2, amp, competition), nprocs=2)


def test_cuda_embedding_objective_cost():
    import os, json, time
    if os.environ.get('FIBER_COOPERATIVE_CUDA_TEST') != '1':
        pytest.skip('manager-authorized CUDA test required')
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(19)
    batch = pair_batch((32,32,32))
    batch.target_segment_ends_zyx[0,2] = 31
    batch.target_segment_offsets = batch.target_segment_offsets.cuda()
    batch.target_segment_counts = batch.target_segment_counts.cuda()
    batch.target_segment_starts_zyx = batch.target_segment_starts_zyx.cuda()
    batch.target_segment_ends_zyx = batch.target_segment_ends_zyx.cuda()
    batch.embedding_geometry = tuple(replace(g,eligible=g.eligible.cuda()) for g in batch.embedding_geometry)
    embedding = torch.randn(1,16,32,32,32,device='cuda',requires_grad=True)
    presence = torch.rand(1,1,32,32,32,device='cuda')
    report = {}
    for mode in ('pairwise','local_contrastive'):
        batch.embedding_geometry = tuple(replace(g,objective=mode) for g in batch.embedding_geometry)
        times=[]
        torch.cuda.reset_peak_memory_stats()
        for i in range(6):
            embedding.grad = None
            torch.cuda.synchronize()
            start=time.perf_counter()
            r = embedding_statistics(embedding,presence,batch)
            loss = (r[0]/r[1]+r[2]/r[3])*.5 if len(r)==4 else r[4]/r[5]
            loss.backward()
            torch.cuda.synchronize()
            if i: times.append(time.perf_counter()-start)
        diagnostic_times=[]
        for i in range(6):
            diag=[]
            torch.cuda.synchronize()
            start=time.perf_counter()
            with torch.no_grad(): embedding_statistics(embedding,presence,batch,diag)
            torch.cuda.synchronize()
            if i: diagnostic_times.append(time.perf_counter()-start)
        report[mode]=dict(forward_backward_seconds_min_median_max=[min(times),float(np.median(times)),max(times)],
            test_forward_with_diagnostics_seconds_min_median_max=[min(diagnostic_times),float(np.median(diagnostic_times)),max(diagnostic_times)],
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    print('EMBEDDING_GPU_COST '+json.dumps(dict(processing_units='one GPU, one patch;16 channels,32-cube output,65536 budget;FP32 loss',results=report)),flush=True)


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
