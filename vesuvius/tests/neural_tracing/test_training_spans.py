"""CPU-only eligibility checks; no volume downloads or GPU work."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import glob
import json

import numpy as np
import pytest

from vc3d_fiber_format import legacy_lasagna_segments
from vesuvius.neural_tracing.fiber_trace.fiber_json import Vc3dFiber
from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import load_fiber_file
from vesuvius.neural_tracing.fiber_trace_3d.training_spans import training_fiber_spans
from vesuvius.neural_tracing.fiber_trace_3d.arc_sampling import ScrollArcSampler
from vesuvius.neural_tracing.fiber_trace_3d import loader as module
from vesuvius.neural_tracing.fiber_trace_3d.train import _make_test_loader_raw_config


def example(modes=("cspline", "trace", "lasagna", "cspline")):
    points = np.array([[x, 16, 16] for x in range(0, 61)], np.float32)
    cps = points[[10, 20, 30, 40, 50]]
    legacy = legacy_lasagna_segments(len(cps))
    segments = tuple(replace(legacy[i], interp_mode=mode) for i, mode in enumerate(modes)) + (None,)
    return Vc3dFiber(Path("fiber.json"), 3, points, cps, segments, 1, {})


def test_filter_joins_modes_splits_gaps_and_trims_both_tails():
    original = example()
    a, b = training_fiber_spans(original)
    assert a.line_points_xyz[:, 0].tolist() == list(range(10, 31))
    assert b.line_points_xyz[:, 0].tolist() == list(range(40, 51))
    assert [m.interp_mode for m in a.control_point_segments[:-1]] == ["cspline", "trace"]
    assert a.control_point_segments[-1] is None
    assert b.metadata["training_control_span"] == (3, 4)
    assert len(original.line_points_xyz) == 61  # no mutation of shared parsed references


def test_no_length_cutoff():
    f = example(("cspline",) * 4)
    f = replace(f, line_points_xyz=f.line_points_xyz * 100, control_points_xyz=f.control_points_xyz * 100)
    assert len(training_fiber_spans(f)) == 1


def test_missing_and_legacy_json_modes_are_not_accepted_but_nml_edges_are():
    f = example(("lasagna",) * 4)
    assert training_fiber_spans(f) == []
    assert training_fiber_spans(replace(f, control_point_segments=(None,) * 5)) == []
    nml = replace(f, metadata={"source_format": "nml"})
    parts = training_fiber_spans(nml)
    assert len(parts) == 1
    assert parts[0].line_points_xyz[[0, -1], 0].tolist() == [10, 50]


def test_invalid_control_order_is_not_silently_bridged():
    f = example()
    cps = f.control_points_xyz.copy()
    cps[1] = cps[0]
    with pytest.raises(ValueError, match="line order"):
        training_fiber_spans(replace(f, control_points_xyz=cps))


def test_repeated_line_vertices_are_allowed_without_sampling_mass():
    f = example()
    f = replace(f, line_points_xyz=np.insert(f.line_points_xyz, 15, f.line_points_xyz[14], axis=0))
    parts = training_fiber_spans(f)
    sampler = ScrollArcSampler([SimpleNamespace(fiber=p, dataset_config={"scroll_id": "A"}) for p in parts])
    rng = np.random.default_rng(8)
    counts = [0, 0]
    for _ in range(6000):
        _, index, pos = sampler.sample(rng)
        counts[index] += 1
        assert 10 <= pos.xyz[0] <= 30 or 40 <= pos.xyz[0] <= 50
    assert abs(counts[0] / sum(counts) - 2/3) < .025


def test_file_backed_train_and_test_use_identical_filtered_geometry(monkeypatch):
    volume = np.zeros((64, 64, 64), np.uint8)
    monkeypatch.setattr(module, "_open_dataset_volume", lambda *a, **k: volume)
    monkeypatch.setattr(module, "_validate_dataset_manifest", lambda *a, **k: volume.shape)
    monkeypatch.setattr(module, "_resolve_fiber_paths", lambda *a, **k: [Path("fiber.json")])
    monkeypatch.setattr(module, "load_fiber_file", lambda *a, **k: [example()])
    monkeypatch.setattr(module, "make_coordinate_sampler", lambda **k: module.NumpyZarrCoordinateSampler(volume, level_spacing_base=1))
    dataset = {"base_volume_path": "unused", "fiber_glob": "unused", "scroll_id": "A", "dense_supervision": True}
    raw = {"datasets": [dataset], "test_datasets": [dataset], "sampling_strategy": "scroll_balanced_arclength",
           "patch_shape_zyx": [16]*3, "cp_margin_voxels": 3, "augment_enabled": False}
    configs = [raw, _make_test_loader_raw_config(raw, {})]
    loaders = [module.FiberTrace3DLoader(module.config_from_mapping(c)) for c in configs]
    for loader in loaders:
        assert len(loader.records) == 2
        assert len({r.fiber_identity for r in loader.records}) == 2
        for i in range(20):
            rec, _, location = loader.descriptor_for_sample_index(i)
            params = loader._sample_augment_params(rec, location, i)
            lines, _ = loader._line_window_for_labels(rec, location, params)
            assert lines[:, 2].min() >= 10 and lines[:, 2].max() <= 50
            assert not (lines[:, 2].min() < 30 and lines[:, 2].max() > 40)
    for a, b in zip(loaders[0].records, loaders[1].records):
        np.testing.assert_array_equal(a.fiber.line_points_xyz, b.fiber.line_points_xyz)
    monkeypatch.setattr(module, "load_fiber_file", lambda *a, **k: [example(("lasagna",)*4)])
    with pytest.raises(ValueError, match="no eligible training spans"):
        module.FiberTrace3DLoader(module.config_from_mapping(raw))


CONFIG = Path(__file__).resolve().parents[2] / "src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_5scroll_json_eligible_ema_16x8.json"


def test_clean_json_only_training_config():
    raw = json.loads(CONFIG.read_text())
    assert not module.load_config(CONFIG).embedding.enabled
    assert raw["training"]["learning_rate"] == .01
    assert raw["training"]["max_steps"] == 2000000
    assert raw["training"]["optimizer_global_batch_size"] == 128
    assert "historical_test_datasets" not in raw
    assert not any("resume" in k for k in raw["training"])
    for split in ("datasets", "test_datasets"):
        assert len({d["scroll_id"] for d in raw[split]}) == 5
        assert all(d["fiber_glob"].endswith("*.json") for d in raw[split])


def test_local_corpus_eligible_coverage():
    raw = json.loads(CONFIG.read_text())
    if not Path(raw["datasets"][0]["fiber_glob"]).parent.exists():
        pytest.skip("local collection unavailable")
    for split in ("datasets", "test_datasets"):
        counts = {}
        for d in raw[split]:
            row = counts.setdefault(d["scroll_id"], [0, 0, 0.0])
            for path in sorted(glob.glob(d["fiber_glob"])):
                for fiber in load_fiber_file(path):
                    row[0] += 1
                    for part in training_fiber_spans(fiber):
                        row[1] += 1
                        row[2] += float(np.linalg.norm(np.diff(part.line_points_xyz.astype(float), axis=0), axis=1).sum())
                        assert all(m.interp_mode in {"cspline", "trace"} for m in part.control_point_segments[:-1])
                        np.testing.assert_array_equal(part.line_points_xyz[[0, -1]], part.control_points_xyz[[0, -1]])
        assert len(counts) == 5 and all(r[1] > 0 and r[2] > 0 for r in counts.values())
        print("eligible corpus", split, json.dumps(counts, sort_keys=True))


def test_paired_contrastive_config_preserves_regular_settings():
    path = CONFIG.with_name("train_5scroll_json_eligible_paired_contrastive_ema_16x8.json")
    raw = json.loads(path.read_text())
    base = json.loads(CONFIG.read_text())
    old = json.loads(CONFIG.with_name("train_combined_5scroll_s1a_paired_contrastive_ema_16x8.json").read_text())
    assert raw.pop("embedding") == old["embedding"]
    base.pop("embedding")
    assert module.load_config(path).embedding.paired_patches
    for split in ("datasets", "test_datasets"):
        for dataset in raw[split]:
            normal = Path(dataset.pop("lasagna_manifest_path"))
            if not normal.is_absolute():
                assert (CONFIG.parent / normal).is_file()
    raw["training"].pop("run_name")
    base["training"].pop("run_name")
    assert raw == base
