import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


def test_default_classic_workload_uses_full_set_validated_batching():
    configs = Path(__file__).resolve().parents[2] / "src/vesuvius/neural_tracing/fiber_trace_3d/configs"
    default = json.loads((configs / "eval_classic_collection.json").read_text())
    measured = json.loads((configs / "eval_classic_full_dynamic256_b8_heap.json").read_text())
    assert default["evaluation"]["inference"] == measured["evaluation"]["inference"]
    assert default["sources"] == measured["sources"]
    assert sum(s["expected_test_count"] for s in default["sources"]) == 50
    assert all("test_fibers" not in s for s in default["sources"])
    assert default["evaluation"]["trace_args"] == measured["evaluation"]["trace_args"]
    assert default["checkpoint_sha256"] == measured["checkpoint_sha256"]
    assert default["evaluation"]["gpu_manager"] == measured["evaluation"]["gpu_manager"]
    assert default["output_root"] != measured["output_root"]
    assert default["evaluation"]["inference_cache_namespace"] != measured["evaluation"]["inference_cache_namespace"]


def test_explicit_test_subset_is_validated_and_preserves_original_order(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/run_classic_fiber_evaluation.py"
    spec = importlib.util.spec_from_file_location("classic_eval", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = [tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"]
    assert module.select_test_files(files) == files
    assert module.select_test_files(files, ["c.json", "a.json"]) == [files[0], files[2]]
    for invalid in ([], "a.json", [1], ["a.json", "a.json"], ["missing.json"], ["../a.json"], [str(files[0])]):
        with pytest.raises(ValueError):
            module.select_test_files(files, invalid)


def test_workload_config_owns_collection_mapping(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/run_classic_fiber_evaluation.py"
    spec = importlib.util.spec_from_file_location("classic_eval", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "workload.json"
    config = {"version": 1, "pilot_source_id": "collection-label", "sources": [{
        "id": "collection-label", "sample_id": "different-scroll", "volume_id": "scan-id",
        "input_volume": "s3://example/ct.zarr", "normal": {"kind": "manager-run", "selector": "normal-run"}}]}
    path.write_text(json.dumps(config))
    assert module.load_workload(path) == config
    config["sources"].append(config["sources"][0])
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="duplicate"):
        module.load_workload(path)


def test_configured_local_normals_validate_exact_source(monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/run_classic_fiber_evaluation.py"
    spec = importlib.util.spec_from_file_location("classic_eval", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from lasagna.manager import fiberlets
    result = {"source": {"sample_id": "PHercParis4", "volume_id": "20260411134726"},
              "manifest_path": Path("/tmp/test-normal.json"),
              "manifest": {"base_shape_zyx": [100, 100, 100]}}
    def completed(manager, selector, *, expected_kind):
        assert selector == "configured-normal-run"
        assert expected_kind == "lasagna"
        return result
    monkeypatch.setattr(fiberlets, "_completed_dependency", completed)
    binding = {"sample_id": "PHercParis4", "volume_id": "20260411134726",
               "normal": {"kind": "manager-run", "selector": "configured-normal-run"}}
    normal = module.resolve_normal_binding(None, binding, [])
    assert normal["base_shape_zyx"] == [100, 100, 100]
    assert normal["manifest_url"] == "/tmp/test-normal.json"
    with pytest.raises(ValueError, match="different source"):
        module.resolve_normal_binding(None, {**binding, "volume_id": "wrong-scan"}, [])


def test_locked_membership_rejects_changed_content_and_membership(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/run_classic_fiber_evaluation.py"
    spec = importlib.util.spec_from_file_location("classic_eval", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / "test/source"
    directory.mkdir(parents=True)
    annotation = directory / "fiber.json"
    annotation.write_text("{}")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": [{
        "source": "source", "split": "test", "path": "source/fiber.json",
        "sha256": hashlib.sha256(b"{}").hexdigest()}]}))
    module.validate_locked_membership(tmp_path, "source")
    annotation.write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.validate_locked_membership(tmp_path, "source")
    annotation.write_text("{}")
    annotation.rename(directory / "swapped.json")
    with pytest.raises(ValueError, match="membership changed"):
        module.validate_locked_membership(tmp_path, "source")
