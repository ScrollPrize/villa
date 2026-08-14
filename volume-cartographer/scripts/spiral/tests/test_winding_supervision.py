"""Store-level tests for the winding-model crossing supervision."""

import json

import numpy as np
import torch

import winding_supervision as ws


def _array_entry(directory, name, value):
    path = directory / f"{name}.npy"
    np.save(path, np.ascontiguousarray(value), allow_pickle=False)
    return {
        "file": path.name,
        "shape": list(value.shape),
        "dtype": np.dtype(value.dtype).str,
        "sha256": ws._sha256(path),
    }


def build_store(root, rays):
    """Write a minimal single-shard compact store from per-ray dicts."""
    shard_dir = root / "shard0"
    shard_dir.mkdir(parents=True)
    origins = np.asarray([r["origin"] for r in rays], np.float32)
    steps = np.asarray([r["step"] for r in rays], np.float32)
    crossing_t = np.concatenate(
        [np.asarray(r["t"], np.float32) for r in rays])
    crossing_level = np.concatenate(
        [np.asarray(r["level"], np.int16) for r in rays])
    offsets = np.cumsum([0] + [len(r["t"]) for r in rays]).astype(np.int64)
    arrays = {
        "ray_origin_zyx": _array_entry(shard_dir, "ray_origin_zyx", origins),
        "ray_step_zyx": _array_entry(shard_dir, "ray_step_zyx", steps),
        "crossing_t": _array_entry(shard_dir, "crossing_t", crossing_t),
        "crossing_level": _array_entry(
            shard_dir, "crossing_level", crossing_level),
        "crossing_offsets": _array_entry(
            shard_dir, "crossing_offsets", offsets),
        "seed_winding": _array_entry(
            shard_dir, "seed_winding", np.zeros(len(rays), np.int16)),
    }
    manifest = {
        "artifact_type": ws.ARTIFACT_TYPE,
        "format_version": ws.FORMAT_VERSION,
        "coordinate_order": "zyx",
        "num_rays": len(rays),
        "num_crossings": int(len(crossing_t)),
        "shards": [{"name": "shard0", "arrays": arrays}],
    }
    manifest["fingerprint"] = ws._canonical_digest(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))


def _test_rays():
    # Ray A: 10 consecutive crossings along +z from z=100 (z span [100, 109]).
    # Ray B: 3 consecutive crossings along +z from z=500 (z span [500, 502]).
    # Ray C: 2 in-plane crossings at z=50 with a level gap (5 then 7), so its
    #        adjacent-pair target is 2, exposing index/level confusion.
    return [
        {"origin": [100.0, 10.0, 20.0], "step": [1.0, 0.0, 0.0],
         "t": np.arange(10.0), "level": np.arange(1, 11)},
        {"origin": [500.0, 10.0, 20.0], "step": [1.0, 0.0, 0.0],
         "t": np.arange(3.0), "level": np.arange(1, 4)},
        {"origin": [50.0, 10.0, 20.0], "step": [0.0, 1.0, 0.0],
         "t": np.array([0.0, 4.0]), "level": np.array([5, 7])},
    ]


def _generator():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    return generator


def test_adjacent_pairs_target_level_difference(tmp_path):
    build_store(tmp_path, _test_rays())
    store = ws.load_winding_inference_store(tmp_path, "cpu")
    assert sorted(store.density_rays.tolist()) == [0, 1, 2]

    samples = store.sample_adjacent(512, generator=_generator())
    assert samples["points"].shape == (512, 2, 3)
    # Rays A and B have consecutive levels (target 1); ray C skips a level
    # (target 2). Targets follow levels, not index separation.
    assert set(samples["target"].tolist()) == {1.0, 2.0}
    on_ray_c = samples["points"][:, 0, 0] == 50.0
    assert on_ray_c.any()
    assert (samples["target"][on_ray_c] == 2.0).all()
    assert (samples["target"][~on_ray_c] == 1.0).all()


def test_relative_pairs_respect_delta_bounds_and_geometry(tmp_path):
    build_store(tmp_path, _test_rays())
    store = ws.load_winding_inference_store(tmp_path, "cpu")

    samples = store.sample_relative(512, 2, 5, generator=_generator())
    # Only rays A (10 crossings) and B (3 crossings) are eligible; both have
    # consecutive levels, so targets equal the index separation: [2, 5] on A,
    # exactly 2 on B (clamped by its length).
    targets = samples["target"]
    assert ((targets >= 2.0) & (targets <= 5.0)).all()
    on_ray_b = samples["points"][:, 0, 0] >= 500.0
    assert (targets[on_ray_b] == 2.0).all()
    # Points lie on the originating ray: origin + t * step keeps y/x fixed
    # for the +z rays used here.
    assert (samples["points"][:, :, 1] == 10.0).all()
    assert (samples["points"][:, :, 2] == 20.0).all()


def test_z_range_prefilter_drops_unreachable_rays(tmp_path):
    build_store(tmp_path, _test_rays())
    store = ws.load_winding_inference_store(
        tmp_path, "cpu", z_range=(0.0, 200.0))
    assert store.num_z_eligible_rays == 2
    # Ray B (z span [500, 502]) can never yield a valid pair in [0, 200).
    assert sorted(store.density_rays.tolist()) == [0, 2]

    relative = store.sample_relative(256, 2, 5, generator=_generator())
    assert len(relative["target"]) == 256
    assert (relative["points"][..., 0] < 200.0).all()

    adjacent = store.sample_adjacent(256, generator=_generator())
    assert (adjacent["points"][..., 0] < 200.0).all()


def test_fully_filtered_store_returns_empty_samples(tmp_path):
    build_store(tmp_path, _test_rays())
    store = ws.load_winding_inference_store(
        tmp_path, "cpu", z_range=(1000.0, 2000.0))
    assert store.num_z_eligible_rays == 0
    assert len(store.sample_adjacent(64, generator=_generator())["target"]) == 0
    assert len(store.sample_relative(
        64, 2, 5, generator=_generator())["target"]) == 0


def test_manifest_fingerprint_is_enforced(tmp_path):
    build_store(tmp_path, _test_rays())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["num_rays"] = 999
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    try:
        ws.load_winding_inference_store(tmp_path, "cpu")
    except ValueError as error:
        assert "fingerprint" in str(error)
    else:
        raise AssertionError("tampered manifest was accepted")
