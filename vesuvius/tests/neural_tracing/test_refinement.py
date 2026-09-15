from dataclasses import replace
from types import SimpleNamespace
import json
import os
import time
import zipfile

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.refinement import (
    RefinementConfig, GeometryStore, RefinementObserver, RefinementObservation,
    arclength, resample, ellipse_metric, clamp_update, taper, semi_implicit_step,
    presence_force, refine_curve, refinement_session,
    refine_observations, BatchedPatchTransform, step_limits,
)


def line():
    points = np.c_[np.linspace(4, 59, 24), np.full(24, 24.), np.full(24, 32.)]
    return np.c_[points, arclength(points)]


def test_config_rejects_invalid():
    for key, value in [("iterations", 0), ("shape_weight", -1), ("max_step_fraction", 2),
                       ("border_fraction", .9), ("enabled", 1), ("unknown", 3), ("spacing_base", "bad")]:
        with pytest.raises(ValueError):
            RefinementConfig.parse({"refinement": {key: value}})
    assert not RefinementConfig.parse({}).enabled
    for key in ("model", "model_3d"):
        with pytest.raises(ValueError, match="unconditioned"):
            RefinementConfig.parse({"refinement": {"enabled": True}, key: {"conditioned_decoder_enabled": True}})


def test_ellipse_plane_step_and_border():
    cfg = RefinementConfig()
    p = line()[:, :3]
    normals = np.tile([0., 1., 0.], (len(p), 1))
    q = ellipse_metric(normals, cfg)
    delta = clamp_update(np.tile([20., 100., 100.], (len(p), 1)), p, q, np.full(len(p), .125))
    np.testing.assert_allclose(delta[:, 0], 0, atol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.einsum("ni,nij,nj->n", delta, q, delta)), .125)
    np.testing.assert_allclose(taper(np.array([[0, 32, 32], [4, 32, 32], [8, 32, 32], [63, 32, 32]]), (64,)*3, .125), [0, .5, 1, 0])


def test_shape_translation_and_tether():
    cfg = RefinementConfig(gt_weight=0)
    baseline = line()[:, :3]
    points = baseline + [0, 4, 0]
    q = ellipse_metric(np.tile([0, 1, 0], (len(points), 1)), cfg)
    np.testing.assert_allclose(semi_implicit_step(points, baseline, q, np.zeros_like(points), cfg), 0, atol=1e-12)
    cfg = replace(cfg, gt_weight=1)
    delta = semi_implicit_step(points, baseline, q, np.zeros_like(points), cfg)
    assert np.all(delta[:, 1] < 0)


def test_quadrature_matches_linear_field():
    p = line()[:, :3]
    presence = np.broadcast_to(np.arange(64, dtype=float)[None, :, None] / 63, (64,)*3).copy()
    force = presence_force(p, lambda x: x, presence, (64,)*3, RefinementConfig())
    np.testing.assert_allclose(force[:, 1], 1/63, atol=1e-12)
    np.testing.assert_allclose(force[:, [0, 2]], 0, atol=1e-12)


def test_native_stride_lattice_no_half_pixel_shift():
    p = line()[:, :3]
    presence = np.broadcast_to(np.exp(-((np.arange(16)[None,:,None] - 6)/2)**2), (16,)*3).copy()
    force = presence_force(p, lambda x:x, presence, (64,)*3, RefinementConfig(), output_stride=4)
    np.testing.assert_allclose(force[:, 1], 0, atol=1e-12)  # peak index6 == input y24


def make_refining_loader(monkeypatch, tmp_path):
    from tests.neural_tracing.test_fiber_trace_3d import _loader_with_mapping
    from vesuvius.neural_tracing.fiber_trace_3d.loader import FiberTrace3DLoader
    monkeypatch.setenv("VILLA_FIBER_REFINEMENT_STATE", str(tmp_path / "state"))
    original = _loader_with_mapping({"augment_enabled": False, "patch_shape_zyx": [32]*3})
    for record in original.config._array_records:
        record.update(scroll_id="A", dense_supervision=True, lasagna_manifest_path="mock-normals")
    cfg = replace(original.config, refinement=RefinementConfig(enabled=True), sampling_strategy="scroll_balanced_arclength",
                  datasets=({"lasagna_manifest_path": "mock-normals"},))
    object.__setattr__(cfg, "_array_records", original.config._array_records)
    monkeypatch.setattr(FiberTrace3DLoader, "_sample_native_normals", lambda self,r,p,t=None: np.tile([0.,1.,0.], (len(p),1)))
    return FiberTrace3DLoader(cfg)


def test_loader_reuses_updated_geometry_and_transports_observation(monkeypatch, tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.loader import FiberTrace3DSharedTensorSlots, collate_fiber_trace_3d_samples, FiberTrace3DLoader
    from vesuvius.neural_tracing.fiber_trace_3d.train import _slice_batch
    loader = make_refining_loader(monkeypatch, tmp_path)
    sample = loader.load_sample(0)
    assert sample.refinement_geometry is not None
    slots = FiberTrace3DSharedTensorSlots(start_index=0, slot_count=1, patch_shape_zyx=(32,)*3)
    payload = slots.encode(0, sample, final=True)
    copied = slots.decode(0, payload, final=True)
    np.testing.assert_array_equal(copied.refinement_geometry.current, sample.refinement_geometry.current)
    batch = collate_fiber_trace_3d_samples([copied]).to("cpu")
    assert _slice_batch(batch, 0, 1).refinement_geometry == batch.refinement_geometry
    key = loader._refinement_keys[0]
    before = loader._refinement_store.read(key)
    after = before.copy()
    after[:,1] += 2
    assert loader._refinement_store.commit(key, 0, before, after)
    again = loader.load_sample(0)
    assert again.refinement_geometry is not None
    np.testing.assert_allclose(again.refinement_geometry.current[:,1], sample.refinement_geometry.current[:,1]+2)
    # Baseline loader records remain immutable and keep sampling identities.
    np.testing.assert_array_equal(loader._refinement_store.baseline(key)[:,:3], loader.records[0].fiber.line_points_zyx)
    test_cfg = replace(loader.config, refinement_split="test")
    object.__setattr__(test_cfg, "_array_records", loader.config._array_records)
    test = FiberTrace3DLoader(test_cfg)
    assert test._refinement_keys[0] != key
    np.testing.assert_array_equal(test._refinement_store.read(test._refinement_keys[0]), before)


def test_fixed_test_crop_local_targets_and_resume(monkeypatch, tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.loader import FiberTrace3DLoader
    train = make_refining_loader(monkeypatch, tmp_path)
    cfg = replace(train.config, refinement_split="test")
    object.__setattr__(cfg, "_array_records", train.config._array_records)
    loader = FiberTrace3DLoader(cfg)
    first = loader.load_sample(0)
    other = loader.load_sample(1)
    obs = first.refinement_geometry
    assert obs is not None and other.refinement_geometry is not None
    assert obs.independent_piece and other.refinement_geometry.independent_piece
    assert obs.key != other.refinement_geometry.key
    store = loader._refinement_store
    source = store.read(loader._refinement_keys[0])
    moved = obs.current.copy()
    moved[1:-1, 1] += 2
    assert store.commit(obs.key, 0, obs.current, moved)
    again = loader.load_sample(0)
    np.testing.assert_array_equal(again.refinement_geometry.current, moved)
    np.testing.assert_array_equal(again.refinement_geometry.params.cp_volume_zyx, obs.params.cp_volume_zyx)
    np.testing.assert_array_equal(again.refinement_geometry.params.source_to_output_zyx, obs.params.source_to_output_zyx)
    torch.testing.assert_close(first.volume, again.volume, rtol=0, atol=0)
    torch.testing.assert_close(first.crop_origin_zyx, again.crop_origin_zyx, rtol=0, atol=0)
    assert not torch.equal(first.target_segment_starts_zyx, again.target_segment_starts_zyx)
    np.testing.assert_array_equal(loader.load_sample(1).refinement_geometry.current, other.refinement_geometry.current)
    np.testing.assert_array_equal(store.read(loader._refinement_keys[0]), source)
    # Legacy whole-test updates never move the crop or seed its local curve.
    legacy = source.copy()
    legacy[:, 1] += 10
    assert store.commit(loader._refinement_keys[0], 0, source, legacy)
    np.testing.assert_array_equal(loader.load_sample(0).refinement_geometry.current, moved)
    archive = tmp_path / "geometry.zip"
    saved = store.save(archive)
    with zipfile.ZipFile(archive) as z:
        source_json = json.loads(z.read(loader._refinement_keys[0] + "/fiber.json"))
        assert source_json["refinement"]["evaluation_role"] == "immutable_source_only"
    restored = GeometryStore(tmp_path / "restored")
    restored.restore(archive, saved["sha256"])
    monkeypatch.setenv("VILLA_FIBER_REFINEMENT_STATE", str(restored.root))
    resumed = FiberTrace3DLoader(cfg)
    np.testing.assert_array_equal(resumed.load_sample(0).refinement_geometry.current, moved)
    assert resumed.load_sample(2).refinement_geometry is not None  # lazy new crop after restore


def test_independent_piece_uniform_cap_and_free_endpoints():
    from vesuvius.neural_tracing.fiber_trace_3d.loader import _Augment3DParams
    cfg = RefinementConfig(iterations=1, gt_weight=0., shape_weight=0., learning_rate=10.)
    points = np.c_[np.linspace(0, 63, 9), np.ones(9), np.full(9, 32.)]
    current = np.c_[points, arclength(points)]
    normals = np.tile([0.,1.,0.], (len(points),1))
    presence = np.broadcast_to(np.arange(64)[None,:,None]/63., (64,)*3).copy()
    params = _Augment3DParams(0, np.zeros(3), np.zeros(3), np.eye(3), np.eye(3),
                            0,1,1,0,0,None,0,0,None)
    train = RefinementObservation("a",0,current,current,normals,params,1.,(64,)*3,cfg)
    test = replace(train, independent_piece=True)
    a,b = refine_observations([train,test], [presence,presence])
    expected = refine_curve(current,points,current[:,3],normals,presence,lambda x:x,
                            (64,)*3,cfg,independent_piece=True)
    np.testing.assert_allclose(b,expected,atol=1e-10)
    np.testing.assert_allclose(b[:,1]-points[:,1], cfg.max_step_fraction*cfg.normal_radius_base)
    np.testing.assert_array_equal(a[[0,-1]], current[[0,-1]])
    limits = step_limits(points,(64,)*3,cfg,True,np.array([0,8]))
    np.testing.assert_array_equal(limits,np.full(9,.125))


def test_independent_piece_omits_stitching_segments(monkeypatch, tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.arc_sampling import FiberLocation
    loader = make_refining_loader(monkeypatch, tmp_path)
    record, _, cp = loader.descriptor_for_sample_index(0)
    params = loader._sample_augment_params(record, cp, 0)
    points = np.c_[np.linspace(4,28,5), np.full(5,16.), np.full(5,16.)]
    source = loader._output_points_to_source_volume_np(points, params) * record.volume_spacing_base
    record = replace(record, fiber=replace(record.fiber,line_points_xyz=source[:,::-1]))
    loc = FiberLocation(2,.5,tuple(source[2,::-1]))
    all_segments = loader._line_segment_spec(record,loc,params,patch_shape=(32,)*3,radius=1.)
    separate = loader._line_segment_spec(replace(record,label_breaks=(1,3)),loc,params,
                                         patch_shape=(32,)*3,radius=1.)
    for all_values, actual in zip(all_segments,separate):
        np.testing.assert_array_equal(actual,all_values[[0,2]])


def test_raw_and_ema_metric_tags_match():
    from vesuvius.neural_tracing.fiber_trace_3d.train import _log_test_scalars
    values = dict(total=1., direction=.1, presence=.9, angle_mean_deg=3.,
                  branch0_fraction=1., branch1_fraction=0., selected_score_mean=.8)
    calls = []
    writer = SimpleNamespace(add_scalar=lambda *args: calls.append(args))
    for prefix in ("test", "test_ema_0p99", "test_ema_0p999"):
        _log_test_scalars(writer, prefix, values, SimpleNamespace(embedding_channels=0), 100)
    tags = [{tag.split("/", 1)[1] for tag, _, _ in calls if tag.startswith(p + "/")}
            for p in ("test", "test_ema_0p99", "test_ema_0p999")]
    assert tags[0] == tags[1] == tags[2]
    assert {"loss_presence", "loss_direction", "loss_total"} <= tags[0]


def test_refine_peak_and_material_reference():
    p = line()
    normals = np.tile([0., 1., 0.], (len(p), 1))
    presence = np.broadcast_to(np.exp(-((np.arange(64)[None, :, None] - 28) / 5)**2), (64,)*3).copy()
    out = refine_curve(p, p[:, :3], p[:, 3], normals, presence, lambda x: x, (64,)*3, RefinementConfig())
    assert out[12, 1] > p[12, 1]
    np.testing.assert_array_equal(out[[0, -1]], p[[0, -1]])
    assert np.all(np.diff(out[:, 3]) > 0)
    # Uniform distances are measured along the input polyline, not straight
    # chords across its corners. The two are deliberately not identical.
    corner = np.array([[0., 0, 0], [3, 0, 0], [3, 3, 0]])
    sampled = resample(corner, arclength(corner), 5)
    np.testing.assert_allclose(sampled[:, 3], [0, 1.5, 3, 4.5, 6])
    np.testing.assert_allclose(sampled[:, :2], [[0,0],[1.5,0],[3,0],[3,1.5],[3,3]])
    flat = refine_curve(p, p[:, :3], p[:, 3], normals, np.zeros((64,)*3), lambda x: x, (64,)*3, RefinementConfig())
    np.testing.assert_allclose(flat, p, atol=1e-12)
    flipped = normals.copy()
    flipped[::2] *= -1
    equivalent = refine_curve(p, p[:,:3], p[:,3], flipped, presence, lambda x:x, (64,)*3, RefinementConfig())
    np.testing.assert_array_equal(equivalent, out)


def test_store_snapshot_restore_and_stale(tmp_path):
    cfg = RefinementConfig()
    store = GeometryStore(tmp_path / "live")
    key = "a"*64
    p = line()[:, :3]
    store.register(key, p, {"split": "train", "source": "fiber.json"}, cfg)
    before = store.read(key)
    after = before.copy()
    after[2, 1] += 1
    assert store.commit(key, 0, before, after)
    assert not store.commit(key, 0, before, after)
    saved = store.save(tmp_path / "fibers.zip")
    restored = GeometryStore(tmp_path / "restored")
    restored.restore(tmp_path / saved["file"], saved["sha256"])
    np.testing.assert_array_equal(restored.read(key), after)
    np.testing.assert_array_equal(restored.baseline(key)[:, :3], p)
    from vesuvius.neural_tracing.fiber_trace.fiber_json import parse_vc3d_fiber
    with zipfile.ZipFile(tmp_path / "fibers.zip") as z:
        exported = parse_vc3d_fiber(json.loads(z.read(key + "/fiber.json")))
    np.testing.assert_allclose(exported.line_points_zyx, after[:, :3])


def test_session_restores_and_cleans_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("VILLA_FIBER_REFINEMENT_STATE", raising=False)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"refinement": {"enabled": True}}))
    @refinement_session
    def run(path):
        root = os.environ["VILLA_FIBER_REFINEMENT_STATE"]
        assert os.path.isdir(root)
        return root
    root = run(cfg)
    assert not os.path.exists(root)
    assert "VILLA_FIBER_REFINEMENT_STATE" not in os.environ


def _distributed_refinement_worker(rank, world, rendezvous, root):
    import torch.distributed as dist
    from datetime import timedelta
    from vesuvius.neural_tracing.fiber_trace_3d.loader import _Augment3DParams
    os.environ["VILLA_FIBER_REFINEMENT_STATE"] = root
    dist.init_process_group("gloo", init_method="file://"+rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(seconds=30))
    try:
        cfg = RefinementConfig(enabled=True)
        store = GeometryStore(root)
        key = "a"*64
        current = store.read(key)
        params = _Augment3DParams(rank, np.zeros(3), np.zeros(3), np.eye(3), np.eye(3),
            0, 1, 1, 0, 0, None, 0, 0, None)
        obs = RefinementObservation(key, 0, current, store.baseline(key).copy(),
            np.tile([0,1,0], (len(current),1)), params, 1., (64,)*3, cfg, 4)
        batch = SimpleNamespace(stream_indices=torch.tensor([rank]), refinement_geometry=(obs,))
        prediction = torch.zeros(1,7,16,16,16)
        prediction[:,6] = torch.arange(16)[None,:,None] / 16
        observer = RefinementObserver(cfg)
        observer(batch, prediction)
        observer.flush(SimpleNamespace(enabled=True, is_main=rank==0, world_size=world))
        if rank == 0:
            assert observer.accepted == 1
            assert observer.stale == world-1
    finally:
        dist.destroy_process_group()


def test_distributed_updates_survive_cohort_change(tmp_path):
    import torch.multiprocessing as mp
    store = GeometryStore(tmp_path / "store")
    store.register("a"*64, line()[:,:3], {}, RefinementConfig())
    original = store.read("a"*64)
    for world in (2, 1):
        mp.spawn(_distributed_refinement_worker, args=(world, str(tmp_path/f"rdzv{world}"), str(store.root)),
                 nprocs=world, join=True)
    assert not np.array_equal(original, store.read("a"*64))


def test_restore_rejects_changed_dataset(tmp_path):
    store = GeometryStore(tmp_path / "one")
    cfg = RefinementConfig()
    store.register("a"*64, line()[:,:3], {"split":"train"}, cfg)
    saved = store.save(tmp_path / "snapshot.zip")
    restored = GeometryStore(tmp_path / "two")
    restored.restore(tmp_path / saved["file"], saved["sha256"])
    with pytest.raises(ValueError, match="sources differ"):
        restored.register("b"*64, line()[:,:3], {"split":"train"}, cfg)
    with pytest.raises(ValueError, match="membership"):
        restored.validate_split([], "train")
    restored.validate_split(["a"*64], "train")


def test_augmented_mapping_quadrature_and_cost():
    from vesuvius.neural_tracing.fiber_trace_3d.loader import FiberTrace3DLoader, _Augment3DParams
    from tests.neural_tracing.test_fiber_trace_3d import _loader_with_mapping
    loader = _loader_with_mapping({"patch_shape_zyx": [64]*3, "augment_enabled": True,
        "augment_rotation_degrees": 45, "augment_scale_min": .9, "augment_scale_max": 1.1,
        "augment_smooth_displacement_probability": 1., "augment_smooth_displacement_mode": "3d",
        "augment_smooth_displacement_amplitude_zyx": [2]*3})
    record, _, cp = loader.descriptor_for_sample_index(0)
    params = loader._sample_augment_params(record, cp, 0)
    out = line()[:, :3]
    base = loader._output_points_to_source_volume_np(out, params)
    mapped, _ = loader._source_points_to_output_np(base, params)
    np.testing.assert_allclose(mapped, out, atol=.02)
    current = np.c_[base, arclength(base)]
    presence = np.broadcast_to(np.arange(64)[None, :, None]/63., (64,)*3).copy()
    fn = lambda x: loader._source_points_to_output_np(x, params, array_backend=True)[0]
    np.testing.assert_allclose(fn(base), mapped, atol=2e-5)
    start = time.perf_counter()
    for _ in range(10):
        refined = refine_curve(current, base, current[:, 3], np.tile([0, 1, 0], (len(base), 1)),
                               presence, fn, (64,)*3, RefinementConfig())
    elapsed = time.perf_counter()-start
    assert np.isfinite(refined).all()
    print(f"refinement CPU one-core solver: {elapsed/10*1000:.3f} ms/patch, {10/elapsed:.1f} patches/s; 24 nodes,4 iterations,affine+smooth")
    observations = []
    for i in range(16):
        rec, _, cp = loader.descriptor_for_sample_index(i)
        parameters = loader._sample_augment_params(rec, cp, i)
        source = loader._output_points_to_source_volume_np(out, parameters)
        c = np.c_[source, arclength(source)]
        observations.append(RefinementObservation(str(i), 0, c, c.copy(),
            np.tile([0,1,0],(len(c),1)), parameters, 1., (64,)*3, RefinementConfig()))
    mapper = BatchedPatchTransform(observations)
    for i,o in enumerate(observations):
        expected = loader._source_points_to_output_np(o.current[:,:3], o.params, array_backend=True)[0]
        np.testing.assert_allclose(mapper(o.current[:,:3], np.full(len(o.current),i)), expected, atol=1e-10)
    batch_result = refine_observations(observations, [presence]*16)
    for o,result in zip(observations, batch_result):
        expected = refine_curve(o.current,o.baseline[:,:3],o.baseline[:,3],o.normals,presence,
            lambda p: loader._source_points_to_output_np(p,o.params,array_backend=True)[0],o.shape,o.config)
        np.testing.assert_allclose(result,expected,atol=1e-8)
    durations = []
    for _ in range(10):
        start = time.perf_counter()
        refine_observations(observations, [presence]*16)
        durations.append(time.perf_counter()-start)
    print(f"refinement batched CPU one-core solver: 16 patches,24 nodes each,4 iterations,affine+smooth; "
          f"ms/batch min/median/max={np.array([min(durations),np.median(durations),max(durations)])*1000}; "
          f"patches/s={16/np.median(durations):.1f}")


@pytest.mark.skipif(os.environ.get("FIBER_COOPERATIVE_CUDA_TEST") != "1", reason="manager-gated CUDA only")
def test_cuda_fixed_test_crop_feedback(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d.loader import FiberTrace3DLoader, collate_fiber_trace_3d_samples
    train = make_refining_loader(monkeypatch, tmp_path)
    cfg = replace(train.config, refinement_split="test")
    object.__setattr__(cfg, "_array_records", train.config._array_records)
    loader = FiberTrace3DLoader(cfg)
    sample = loader.load_sample(0)
    batch = collate_fiber_trace_3d_samples([sample]).to("cuda")
    stride = sample.refinement_geometry.output_stride
    output = torch.rand(1, 7, *[32 // stride]*3, device="cuda")
    observer = RefinementObserver(cfg.refinement)
    observer(batch, output)
    observer.flush(SimpleNamespace(enabled=False, is_main=True))
    assert observer.accepted == 1
    again = loader.load_sample(0)
    torch.testing.assert_close(sample.volume, again.volume, rtol=0, atol=0)
    np.testing.assert_array_equal(again.refinement_geometry.current,
                                 loader._refinement_store.read(sample.refinement_geometry.key))


@pytest.mark.skipif(os.environ.get("FIBER_COOPERATIVE_CUDA_TEST") != "1", reason="manager-gated CUDA only")
def test_cuda_observer_detaches_and_preserves_backward(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d.loader import _Augment3DParams
    cfg = RefinementConfig(enabled=True)
    monkeypatch.setenv("VILLA_FIBER_REFINEMENT_STATE", str(tmp_path / "store"))
    store = GeometryStore(tmp_path / "store")
    key = "b"*64
    store.register(key, line()[:, :3], {}, cfg)
    p = store.read(key)
    params = _Augment3DParams(0, np.zeros(3), np.zeros(3), np.eye(3), np.eye(3),
        0, 1, 1, 0, 0, None, 0, 0, None)
    observation = RefinementObservation(key, 0, p, store.baseline(key).copy(),
        np.tile([0,1,0], (len(p),1)), params, 1., (64,)*3, cfg, 4)
    batch = SimpleNamespace(stream_indices=torch.tensor([0], device="cuda"), refinement_geometry=(observation,))
    output = torch.rand(1,7,16,16,16,device="cuda",requires_grad=True)
    observer = RefinementObserver(cfg)
    observer(batch, output)
    assert not observer.pending[0][2].requires_grad
    output.square().mean().backward()
    assert torch.isfinite(output.grad).all()
    observer.flush(SimpleNamespace(enabled=False, is_main=True))
    assert observer.accepted == 1
    assert np.isfinite(store.read(key)).all()
