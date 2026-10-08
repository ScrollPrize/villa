"""End-to-end native replay/export/strip tests; no network or model inference.

Set VC_REPLAY_TEST_BINARY to the rebuilt vc_fiber_trace_metric executable.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import socket
import threading
import sys

import numpy as np
import pytest
import zarr

ZARR_V2 = {"zarr_format": 2} if int(zarr.__version__.split(".")[0]) >= 3 else {"zarr_version": 2}


def test_live_remote_metadata_does_not_call_local_only_api(monkeypatch):
    import io
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from vesuvius.neural_tracing.fiber_trace_3d import live_evaluate as live
    class Remote:
        remote_url = "https://example.invalid/ct.zarr/"
        @property
        def root_attrs(self):
            raise RuntimeError("local only")
    calls = []
    def fetch(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(b'{"multiscales": []}')
    monkeypatch.setattr(live, "urlopen", fetch)
    assert live.input_root_attributes(Remote()) == {"multiscales": []}
    assert calls == [("https://example.invalid/ct.zarr/.zattrs", 30)]
    assert live.input_root_attributes(SimpleNamespace(root_attrs={"local": True})) == {"local": True}
    monkeypatch.setattr(live, "urlopen", lambda *a, **kw: io.BytesIO(b'[]'))
    with pytest.raises(ValueError, match="must be an object"):
        live.input_root_attributes(Remote())


def test_live_input_lattice_rejects_affine_mismatch():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import validate_input_lattice
    metadata = {"multiscales": [{"axes": ["z", "y", "x"], "datasets": [
        {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [2.4] * 3}]},
        {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [4.8] * 3}]}]}]}
    validate_input_lattice(metadata, 1)
    transform = metadata["multiscales"][0]["datasets"][1]["coordinateTransformations"][0]
    transform["scale"] = [4.8, 4.8, 7.2]
    with pytest.raises(ValueError, match="power-of-two"):
        validate_input_lattice(metadata, 1)
    transform.update(type="translation", translation=[1, 0, 0])
    with pytest.raises(ValueError, match="translated"):
        validate_input_lattice(metadata, 1)


class SyntheticFiberAdapter:
    """Spawn-picklable identity model with a fixed +X synthetic fiber field."""
    native_output_stride = 1

    def __init__(self, config, *, checkpoint, level, scaledown, inference_scaledown,
                 chunk_size, product_prefix, zarr_path_prefix, emit_embeddings=False):
        assert not emit_embeddings
        from lasagna.tiled_predict3d import OutputProductSpec, OutputChannelSpec
        self.config, self.checkpoint = config, checkpoint
        self.output_products = tuple(OutputProductSpec(name=f"fiber_option_{i:03d}", level=level,
            scaledown=scaledown, inference_scaledown=inference_scaledown, chunk_size=chunk_size,
            accumulator_channel_count=1,
            channels=tuple(OutputChannelSpec(c, relative_path=f"{zarr_path_prefix}_option_{i:03d}_{c}.ome.zarr")
                           for c in ("presence", "nx", "ny"))) for i in range(2))

    def load_model(self, *, device):
        import torch
        return torch.nn.Identity().to(device)

    def run_tile_inference(self, model, tile, *, device):
        return model(tile)

    def product_tensors_from_output(self, value):
        return {p.name: value for p in self.output_products}

    def finalize_product_slab(self, product, raw):
        return {c: np.full(raw.shape[1:], v, np.uint8) for c, v in (("presence", 255), ("nx", 255), ("ny", 128))}


@pytest.fixture
def replay_fixture(tmp_path):
    binary = os.environ.get("VC_REPLAY_TEST_BINARY")
    if not binary:
        pytest.skip("requires a rebuilt native vc_fiber_trace_metric")

    def write(path, value):
        path.write_text(json.dumps(value))

    def volume(name, channels):
        groups = {}
        for channel, value in channels.items():
            path = tmp_path / f"{name}_{channel}.zarr"
            array = zarr.open_array(str(path), mode="w", shape=(32, 32, 32),
                                    chunks=(8, 8, 8), dtype="u1", compressor=None, **ZARR_V2)
            array[:] = value
            groups[channel] = {"zarr": path.name, "scaledown": 0, "channels": [channel]}
        write(tmp_path / f"{name}.json", {"version": 2, "source_to_base": 1,
              "grad_mag_encode_scale": 255.0, "grad_mag_factor": 1.0,
              "base_shape_zyx": [32, 32, 32], "groups": groups})

    volume("fiber", {"presence": 255, "nx": 255, "ny": 128})
    volume("normal", {"grad_mag": 255, "nx": 128, "ny": 128})
    ct = zarr.open_group(str(tmp_path / "ct.zarr"), mode="w", **ZARR_V2)
    array = zarr.open_array(str(tmp_path / "ct.zarr" / "0"), mode="w",
                            shape=(32, 32, 32), chunks=(8, 8, 8), dtype="u1",
                            compressor=None, **ZARR_V2)
    array[:] = np.indices(array.shape)[1].astype(np.uint8) * 7
    ct.attrs["multiscales"] = [{"axes": [{"name": n, "type": "space"} for n in ("z", "y", "x")],
                                "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]}]}]
    points = [[8, 16, 16], [24, 16, 16]]
    write(tmp_path / "annotation.json", {"type": "vc3d_fiber", "version": 1,
          "optimization_mode": "native_fiber_trace3d",
          "line_points": points, "control_points": points})
    config = {"version": 1, "output": "evaluation", "workers": 2,
              "decoded_cache_bytes": 1024 * 1024,
              "trace_args": ["--cone-angle-degrees", "0", "--step-voxels", "2"],
              "sources": [{"id": "synthetic", "fiber_manifest": "fiber.json", "normal_manifest": "normal.json",
                           "ct_volume": "ct.zarr/0", "base_voxel_size_um": 1,
                           "inference_scaledown_power": 0, "fibers": ["annotation.json"]}]}
    write(tmp_path / "evaluation.json", config)
    render = {"normal_manifest": "normal.json", "ct_volume": "ct.zarr/0", "normal_working_to_base": 1,
              "threads": 1, "output": "rendered", "decoded_cache_bytes": 1024 * 1024}
    write(tmp_path / "render.json", render)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def run(*arguments, check=True):
        result = subprocess.run([binary, *map(str, arguments)], cwd="/tmp", env=env,
                                text=True, capture_output=True, timeout=120)
        if check:
            assert result.returncode == 0, result.stdout + result.stderr
        return result

    return tmp_path, config, run


@pytest.mark.parametrize("fit", [None, {"method":"weighted","iterations":4,"presence_exponent":2,"direction_exponent":2}])
def test_anchor_only_coverage_does_not_construct_fiberlets(replay_fixture, fit):
    root, config, run = replay_fixture
    config.update(backend="anchors", fiberlet={"storage_chunk_side_base": 16, "construction_workers": 8})
    if fit is not None:
        config["fiberlet"]["anchor_fit"] = fit
    config["decoded_cache_bytes"] = 256 * 1024 * 1024
    presence = zarr.open_array(str(root / "fiber_presence.zarr"), mode="r+")
    presence[:] = 0
    presence[15:18, 15:18, :] = 255
    config["sources"][0]["fiberlet_cache_root"] = str(root / "anchors-cache")
    (root / "evaluation.json").write_text(json.dumps(config))
    first = run("replay-set", "--config", root / "evaluation.json")
    assert "stage=anchors" in first.stdout
    assert "stage=fiberlets" not in first.stdout
    result = json.loads((root / "evaluation/summary.json").read_text())
    assert result["completed"] and result["cases"] == 1  # placement is not directional
    case = result["anchor_cases"][0]
    assert case["accepted_anchors"]
    assert case["cell_side_base"] == 4
    assert all(a["direction_error_degrees"] is not None for a in case["accepted_anchors"])
    if fit is not None:
        cells = [tuple(a["id"]["cell_zyx"]) for a in case["accepted_anchors"]]
        assert len(cells) == len(set(cells))
    assert not (root / "evaluation/cases/0/graph.json").exists()
    config["output"] = "anchors-warm"
    (root / "evaluation.json").write_text(json.dumps(config))
    second = run("replay-set", "--config", root / "evaluation.json")
    assert "stage=fiberlets" not in second.stdout
    warm = json.loads((root / "anchors-warm/summary.json").read_text())
    assert result["anchor_cases"] == warm["anchor_cases"]


@pytest.mark.parametrize('direction', ['forward', 'reverse'])
@pytest.mark.parametrize('broken', [False, True])
def test_global_failure_trial_original_arcs_and_endpoint_guard(replay_fixture, direction, broken):
    from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import failure_trial_descriptor
    root, config, run = replay_fixture
    # Original annotation is x=8..24. Eligible segment starts four voxels in.
    candidate = dict(id='nonzero-offset', piece_key=[str(root/'annotation.json')],
        source_id='synthetic', target=8, run=dict(original_arcs_base=[4, 14]))
    trial = failure_trial_descriptor(candidate, direction, 2,
        np.array([[12., 16., 16.], [22., 16., 16.]]), np.array([0., 10.]))
    if broken:
        trial['original_fiber_arc_start_base'] -= 4
        trial['original_fiber_arc_end_base'] -= 4
    config.update(backend='beam', failure_search_trials_only=True, failure_search_trials=[trial])
    (root/'evaluation.json').write_text(json.dumps(config))
    result = run('replay-set', '--config', root/'evaluation.json', check=not broken)
    if broken:
        assert result.returncode != 0
        assert 'endpoint mismatch' in result.stderr + result.stdout
    else:
        doc = json.loads((root/'evaluation/cases/0/trace.json').read_text())
        np.testing.assert_allclose(doc['reference_points_base_xyz'][0], trial['expected_start_base_xyz'], atol=1e-6)
        np.testing.assert_allclose(doc['reference_points_base_xyz'][-1], trial['expected_end_base_xyz'], atol=1e-6)


def test_fiberlet_backend_canonical_cache_resume(replay_fixture):
    import hashlib
    root, config, run = replay_fixture
    config.update(backend="fiberlet", fiberlet={"storage_chunk_side_base": 16,
        "construction_workers": 8, "lookahead_base_voxels": 16})
    config["decoded_cache_bytes"] = 256 * 1024 * 1024
    presence = zarr.open_array(str(root / "fiber_presence.zarr"), mode="r+")
    presence[:] = 0
    presence[15:18, 15:18, :] = 255
    config["sources"][0]["fiberlet_cache_root"] = str(root / "fiberlet-cache")
    (root / "evaluation.json").write_text(json.dumps(config))
    cold = run("replay-set", "--config", root / "evaluation.json")
    # Prefix and route misses for one owner must share generation, including
    # empty owners encountered in the surrounding graph halo.
    completed = [line.split(' seconds=')[0] for line in cold.stdout.splitlines()
                 if line.startswith('fiberlet-preprocess ') and ' stage=fiberlets ' in line]
    assert completed and len(completed) == len(set(completed))
    first = json.loads((root / "evaluation/summary.json").read_text())
    assert first["completed"] and first["cases"] == 2
    assert first["sources"][0]["tracer"] == "fiberlet"
    cache = root / "fiberlet-cache"
    def payloads():
        return {str(p.relative_to(cache)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in cache.rglob("*") if p.is_file() and p.name != ".lock"}
    before = payloads()
    assert before
    config["output"] = "warm"
    config["render"] = True
    config["excursion_grace_base_voxels"] = 160
    config["tangential_threshold_factor"] = 2
    (root / "evaluation.json").write_text(json.dumps(config))
    run("replay-set", "--config", root / "evaluation.json")
    second = json.loads((root / "warm/summary.json").read_text())
    assert second["completed"] and second["cases"] == 2
    assert payloads() == before  # replay-only settings do not alter construction
    assert (root / "warm/cases/0/graph.json").is_file()
    assert (root / "warm/cases/0/rendered/render.json").is_file()

    config["output"] = "path-fine"
    config["render"] = False
    config["fiberlet"]["path"] = {
        "longitudinal_step_prediction": 1.0,
        "transverse_step_prediction": 1.0,
    }
    (root / "evaluation.json").write_text(json.dumps(config))
    run("replay-set", "--config", root / "evaluation.json")
    after = payloads()
    assert before.items() <= after.items()
    assert len(after) > len(before)  # construction settings create a new fingerprint root
    metadata = [json.loads(path.read_text()) for path in cache.rglob(".zattrs")]
    assert any(value.get("processing", {}).get("paths", {}).get(
        "longitudinal_step_prediction") == 1.0 for value in metadata)

    config["output"] = "invalid-path"
    config["fiberlet"]["path"]["parallel_threads"] = 8
    (root / "evaluation.json").write_text(json.dumps(config))
    rejected = run("replay-set", "--config", root / "evaluation.json", check=False)
    assert rejected.returncode != 0
    assert "unknown fiberlet.path parameter: parallel_threads" in rejected.stderr


def test_fiberlet_backend_empty_seed_reports_untraced_coverage(replay_fixture):
    root, config, run = replay_fixture
    zarr.open_array(str(root / "fiber_presence.zarr"), mode="r+")[:] = 0
    config.update(backend="fiberlet", render=True, fiberlet={
        "storage_chunk_side_base": 16, "workers_per_source": 2})
    config["sources"][0]["fiberlet_cache_root"] = str(root / "empty-cache")
    (root / "evaluation.json").write_text(json.dumps(config))
    run("replay-set", "--config", root / "evaluation.json")
    result = json.loads((root / "evaluation/summary.json").read_text())
    assert result["completed"]
    assert result["sources"][0]["summary"]["traced_length_base_voxels"] == 0
    assert result["sources"][0]["summary"]["length_weighted_success_percent"] == 0
    for case in (0, 1):
        artifact = json.loads((root / f"evaluation/cases/{case}/trace.json").read_text())
        assert artifact["replay"]["geometry_backend"] == "fiberlet"
        assert artifact["replay"]["failures"]
        assert (root / f"evaluation/cases/{case}/rendered/render.json").is_file()


def test_eligible_spans_trim_extrapolation_and_skip_lasagna(replay_fixture):
    root, config, run = replay_fixture
    def metadata(mode):
        return dict(optimizer="native_fiber_trace3d", metadata_version=3, tracer_version=2,
            interp_goal="trace", interp_mode=mode, metric=None, msg="test",
            normal_manifest="", fiber_manifest="", trace_to_base_scale=1,
            meeting_error_base_voxels=None, meeting_error_ratio=None, meeting_source="",
            failure_code="", failure_detail="", lasagna_failure_code="", lasagna_failure_detail="",
            config=dict(step_voxels=4, cone_angle_degrees=25, cone_angle_step_degrees=5,
                cone_grid_size=25, beam_width=8, beam_prune_distance_voxels=1, beam_lookahead_steps=2,
                smoothness_weight=2, smoothness_normal_weight=.1, smoothness_tangent_weight=10,
                smoothness_free_angle_degrees=0, cumulative_smoothness_steps=4,
                cumulative_smoothness_tangent_weight=2, initial_free_angle_degrees=0,
                max_step_factor=3, meeting_accept_max_error_ratio=.1, endpoint_accept_threshold_base_voxels=20))
    points = [[x, 16, 16] for x in range(4, 29, 2)]
    controls = [{"position": points[i], "segment_to_next": metadata(mode)}
                for i, mode in [(1, "cspline"), (3, "cspline"), (5, "lasagna"), (8, "cspline")]]
    controls.append({"position": points[10]})
    annotation = dict(type="vc3d_fiber", version=3, optimization_mode="native_fiber_trace3d",
                      line_points=points, control_points=controls)
    (root / "annotation.json").write_text(json.dumps(annotation))
    plan = json.loads(run("replay-references", "--config", root / "evaluation.json").stdout)
    selected = plan["sources"][0]["fibers"][0]
    assert selected["original_length_base_voxels"] == 24
    assert selected["eligible_length_base_voxels"] == 12
    assert [r["line_indices"] for r in selected["runs"]] == [[1, 5], [8, 10]]
    config["reference_plan"] = plan
    config["render"] = True
    (root / "evaluation.json").write_text(json.dumps(config))
    run("replay-set", "--config", root / "evaluation.json")
    summary = json.loads((root / "evaluation/summary.json").read_text())
    assert summary["cases"] == 4 and summary["failures"] == 0
    assert summary["directed_reference_length_mm"] == pytest.approx(.024)
    assert summary["mean_segment_length_mm"] == pytest.approx(.024)
    for case in range(4):
        document = json.loads((root / f"evaluation/cases/{case}/trace.json").read_text())
        span = document["provenance"]["reference_run"]
        metadata_json = json.loads((root / f"evaluation/cases/{case}/rendered/render.json").read_text())
        for section in metadata_json["cross_sections"]:
            lo, hi = span["original_arcs_base"]
            expected = hi - section["directional_arc_base"] if document["provenance"]["reverse"] else lo + section["directional_arc_base"]
            assert section["original_fiber_arc_base"] == pytest.approx(expected)
    for cp in controls[:-1]:
        cp["segment_to_next"]["interp_mode"] = "lasagna"
    (root / "annotation.json").write_text(json.dumps(annotation))
    changed = run("replay-set", "--config", root / "evaluation.json", check=False)
    assert changed.returncode != 0 and "selection changed" in changed.stderr
    config.pop("reference_plan")
    config["output"] = "excluded"
    (root / "excluded.json").write_text(json.dumps(config))
    run("replay-set", "--config", root / "excluded.json")
    empty = json.loads((root / "excluded/summary.json").read_text())
    assert empty["cases"] == 0 and empty["failures"] == 0
    assert empty["mean_segment_length_mm"] is None


@pytest.mark.parametrize("directory_alias", [False, True])
def test_reference_plan_resolves_symlink_identity(replay_fixture, directory_alias):
    root, config, run = replay_fixture
    target = root / "annotation.json"
    if directory_alias:
        (root / "test").symlink_to(root, target_is_directory=True)
        alias = root / "test/annotation.json"
    else:
        alias = root / "test.json"
        alias.symlink_to("annotation.json")
    config["sources"][0]["fibers"] = [str(alias)]
    (root / "alias.json").write_text(json.dumps(config))
    plan = json.loads(run("replay-references", "--config", root / "alias.json").stdout)
    assert plan["sources"][0]["fibers"][0]["fiber_path"] == str(target.resolve())
    config["reference_plan"] = plan
    config["sources"][0]["fibers"] = [str(alias.resolve())]
    (root / "resolved.json").write_text(json.dumps(config))
    run("replay-set", "--config", root / "resolved.json")
    assert json.loads((root / "evaluation/summary.json").read_text())["cases"] == 2
    document = json.loads(target.read_text())
    document["control_points"][1][0] -= 1
    document["line_points"][1][0] -= 1
    target.write_text(json.dumps(document))
    changed = run("replay-set", "--config", root / "resolved.json", check=False)
    assert changed.returncode != 0 and "selection changed" in changed.stderr
    target.unlink()
    assert run("replay-references", "--config", root / "alias.json", check=False).returncode != 0


def test_bidirectional_export_and_posthoc_strips(replay_fixture):
    root, config, run = replay_fixture
    run("replay-set", "--config", root / "evaluation.json")
    summary = json.loads((root / "evaluation/summary.json").read_text())
    assert summary["completed"] and summary["cases"] == 2
    assert summary["config"]["decoded_cache_bytes"] == 1024 * 1024
    assert summary["failures"] == 0
    assert summary["directed_reference_length_mm"] == pytest.approx(0.032)
    assert summary["version"] == 3
    assert summary["metric_definition"] == "corpus-failures-plus-one-v1"
    assert summary["reliability_segments"] == 1
    assert summary["mean_segment_length_mm"] == pytest.approx(0.032)
    assert all(source["version"] == 6 for source in summary["sources"])
    for index in range(2):
        artifact = root / "evaluation/cases" / str(index)
        data = json.loads((artifact / "trace.json").read_text())
        assert data["provenance"]["reverse"] == bool(index)
        assert data["replay"]["segments"]
        assert (artifact / "trace.obj").is_file()
        assert (artifact / "reference.obj").is_file()
    # Remove predictions: rendering must use saved geometry, not sample them.
    (root / "fiber.json").rename(root / "fiber-not-available.json")
    run("render-replay", root / "evaluation/cases/0/trace.json", "--config", root / "render.json")
    from PIL import Image
    files = list((root / "rendered").glob("strips_*.jpg"))
    assert files
    for path in files:
        with Image.open(path) as image:
            assert image.width > 0 and image.height > 0
    assert run("replay-set", "--config", root / "evaluation.json", check=False).returncode != 0


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_single_direction_annotation(replay_fixture, direction):
    root, config, run = replay_fixture
    path = root / "annotation.json"
    annotation = json.loads(path.read_text())
    annotation["evaluation_direction"] = direction
    path.write_text(json.dumps(annotation))
    run("replay-set", "--config", root / "evaluation.json")
    summary = json.loads((root / "evaluation/summary.json").read_text())
    assert summary["cases"] == 1 and summary["eligible_runs"] == 1
    artifact = json.loads((root / "evaluation/cases/0/trace.json").read_text())
    assert artifact["provenance"]["reverse"] == (direction == "reverse")


def test_exported_short_piece_remains_eligible(replay_fixture):
    root, config, run = replay_fixture
    path = root / "annotation.json"
    original = json.loads(path.read_text())
    points = original["line_points"]
    path.write_text(json.dumps(dict(type="vc3d_fiber", version=1,
        line_points=points, control_points=[points[0],points[-1]],
        optimization_mode="native_fiber_trace3d", evaluation_direction="forward")))
    plan = json.loads(run("replay-references", "--config", root / "evaluation.json").stdout)
    selected = plan["sources"][0]["fibers"][0]
    assert len(selected["runs"]) == 1
    assert selected["eligible_length_base_voxels"] == pytest.approx(selected["original_length_base_voxels"])


@pytest.mark.parametrize("folded", [False, True])
def test_indexed_self_proximity(replay_fixture, folded):
    root, config, run = replay_fixture
    config['self_proximity'] = dict(minimum_arc_separation_base=20, sample_spacing_base=1)
    config['error_threshold_base_voxels'] = 1
    config['tangential_threshold_factor'] = 2
    (root/'evaluation.json').write_text(json.dumps(config))
    points = [[4,16,16],[28,16,16]]
    if folded: points += [[28,20,16],[4,20,16],[4,16,16],[6,16,16]]
    annotation=dict(type='vc3d_fiber',version=1,optimization_mode='native_fiber_trace3d',
                    line_points=points,control_points=[points[0],points[-1]])
    (root/'annotation.json').write_text(json.dumps(annotation))
    plan=json.loads(run('replay-references','--config',root/'evaluation.json').stdout)
    proximity=plan['sources'][0]['fibers'][0]['self_proximity']
    assert proximity['excluded'] == folded
    if folded:
        assert abs(proximity['original_arcs_base'][1]-proximity['original_arcs_base'][0]) >=20-1e-7
        assert proximity['threshold_ratio'] <=1


def test_annotation_binding_mismatch_fails_before_output(replay_fixture):
    root, config, run = replay_fixture
    annotation = json.loads((root / "annotation.json").read_text())
    annotation["vc_open_data_coordinate_space"] = "different-scroll/volume@L0"
    (root / "annotation.json").write_text(json.dumps(annotation))
    result = run("replay-set", "--config", root / "evaluation.json", check=False)
    assert result.returncode != 0
    assert "coordinate-space binding mismatch" in result.stderr
    assert not (root / "evaluation").exists()


def test_batch_render_flag_matches_posthoc_and_preserves_traces(replay_fixture):
    root, config, run = replay_fixture
    run("replay-set", "--config", root / "evaluation.json")
    assert not list((root / "evaluation").rglob("strips_*.jpg"))
    config["output"] = "inline"
    (root / "inline.json").write_text(json.dumps(config))
    run("replay-set", "--render", "--config", root / "inline.json")
    assert json.loads((root / "inline/summary.json").read_text())["config"]["render"] is True
    for index in range(2):
        before = json.loads((root / f"evaluation/cases/{index}/trace.json").read_text())
        after = json.loads((root / f"inline/cases/{index}/trace.json").read_text())
        assert before["replay"] == after["replay"]
        assert list((root / f"inline/cases/{index}/rendered").glob("strips_*.jpg"))
        metadata = json.loads((root / f"inline/cases/{index}/rendered/render.json").read_text())
        assert metadata["modalities"] == ["ct", "presence"]
        assert metadata["render_scale"] == 1
        assert metadata["base_samples_per_pixel"] == 2
        assert metadata["cross_sections"] and metadata["rows"]
        for failure in after["replay"]["failures"]:
            sections = [s for s in metadata["cross_sections"] if s.get("restart_index") == failure["index"]]
            for role, offset, point_index in [("breakoff", 0, -1), ("restart_start", 1, 0)]:
                segment_index = failure["segment_index"] + offset
                if segment_index >= len(after["replay"]["segments"]):
                    continue
                points = after["replay"]["segments"][segment_index]["trace_points_base_xyz"]
                if not points:
                    continue
                matching = [s for s in sections if s["role"] == role]
                assert len(matching) == 1
                assert matching[0]["center_base_xyz"] == points[point_index]
        for section in metadata["cross_sections"]:
            assert section["page"] == section["strip_page"]
            left, right = section["columns"]
            assert (left + right - 1) / 2 == section["strip_center_column"]
        assert not (root / f"inline/cases/{index}/rendered_presence").exists()
    run("render-replay", root / "evaluation/cases/0/trace.json", "--config", root / "render.json")
    assert list((root / "rendered").glob("strips_*.jpg"))  # Legacy CT-only posthoc remains available.


def test_visualization_resume_preserves_complete_galleries_and_trace_artifacts(replay_fixture):
    root, config, run = replay_fixture
    config.update(output="inline", render=True)
    path = root / "inline.json"
    path.write_text(json.dumps(config))
    run("replay-set", "--config", path)
    trace_before = {i: (root / f"inline/cases/{i}/trace.json").read_bytes() for i in range(2)}
    summary_before = (root / "inline/summary.json").read_bytes()
    preserved = {p.name: p.read_bytes() for p in (root / "inline/cases/0/rendered").iterdir()}
    shutil.rmtree(root / "inline/cases/1/rendered")
    config.update(resume_visualization=True, resume_render_limit=1,
                  resume_render_cases=[1])
    path.write_text(json.dumps(config))
    run("replay-set", "--config", path)
    assert trace_before == {i: (root / f"inline/cases/{i}/trace.json").read_bytes() for i in range(2)}
    assert (root / "inline/summary.json").read_bytes() == summary_before
    assert preserved == {p.name: p.read_bytes() for p in (root / "inline/cases/0/rendered").iterdir()}
    status = json.loads((root / "inline/visualization_resume.json").read_text())
    assert status["batch_selected_cases"] == [1]
    assert status["remaining_cases"] == []
    regenerated = {p.name: p.read_bytes() for p in (root / "inline/cases/1/rendered").iterdir()}
    del config["resume_render_cases"]
    path.write_text(json.dumps(config))
    run("replay-set", "--config", path)
    assert regenerated == {p.name: p.read_bytes() for p in (root / "inline/cases/1/rendered").iterdir()}
    assert json.loads((root / "inline/visualization_resume.json").read_text())["batch_selected_cases"] == []

    artifact = json.loads((root / "inline/cases/1/trace.json").read_text())
    artifact["provenance"]["source_id"] = "changed"
    (root / "inline/cases/1/trace.json").write_text(json.dumps(artifact))
    failed = run("replay-set", "--config", path, check=False)
    assert failed.returncode != 0
    assert "trace/provenance mismatch" in failed.stderr


def test_visualization_resume_accepts_render_only_and_saved_derived_provenance(replay_fixture):
    root, config, run = replay_fixture
    config.update(output="resume-derived", render=False)
    config["sources"][0].pop("ct_volume")
    path = root / "resume-derived.json"
    path.write_text(json.dumps(config))
    run("replay-set", "--config", path)
    summary_before = (root / "resume-derived/summary.json").read_bytes()
    traces = {}
    for index in range(2):
        artifact_path = root / f"resume-derived/cases/{index}/trace.json"
        artifact = json.loads(artifact_path.read_text())
        artifact["provenance"]["fiberlet_processing"] = {"derived": True}
        artifact_path.write_text(json.dumps(artifact))
        traces[index] = artifact_path.read_bytes()

    config.update(render=True, resume_visualization=True, resume_render_limit=2,
                  resume_render_cases=[0, 1])
    config["sources"][0].update(ct_volume="ct.zarr/0")
    path.write_text(json.dumps(config))
    run("replay-set", "--config", path)

    assert traces == {
        index: (root / f"resume-derived/cases/{index}/trace.json").read_bytes()
        for index in range(2)
    }
    assert (root / "resume-derived/summary.json").read_bytes() == summary_before
    for index in range(2):
        rendered = root / f"resume-derived/cases/{index}/rendered"
        assert list(rendered.glob("strips_*.jpg"))
        provenance = json.loads((rendered / "render.json").read_text())["provenance"]
        assert provenance["fiberlet_processing"] == {"derived": True}
        assert provenance["ct_volume"] == str(root / "ct.zarr/0")


def test_batch_render_base_pyramid(replay_fixture):
    root, config, run = replay_fixture
    config["sources"][0].update(ct_volume="ct.zarr", ct_base_volume=True)
    (root / "base.json").write_text(json.dumps(config))
    run("replay-set", "--render", "--config", root / "base.json")
    for index in range(2):
        metadata = json.loads((root / f"evaluation/cases/{index}/rendered/render.json").read_text())
        assert metadata["texture_source"]["scale_from_base_xyz"] == [1, 1, 1]
        assert metadata["texture_source"]["shape_zyx"] == [32, 32, 32]


def test_batch_render_remote_base_pyramid(replay_fixture):
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    root, config, run = replay_fixture
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config["sources"][0].update(
            ct_volume=f"http://127.0.0.1:{server.server_port}/ct.zarr", ct_base_volume=True)
        (root / "remote.json").write_text(json.dumps(config))
        run("replay-set", "--render", "--config", root / "remote.json")
        assert list((root / "evaluation/cases/0/rendered").glob("strips_*.jpg"))
        assert json.loads((root / "evaluation/cases/0/rendered/render.json").read_text())["modalities"] == ["ct", "presence"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("ct", [None, "missing.zarr/0", "https://example.invalid/ct.zarr/0"])
def test_batch_render_validates_ct_before_tracing(replay_fixture, ct):
    root, config, run = replay_fixture
    if ct is None:
        del config["sources"][0]["ct_volume"]
    else:
        config["sources"][0]["ct_volume"] = ct
    config["render"] = True
    (root / "evaluation.json").write_text(json.dumps(config))
    result = run("replay-set", "--config", root / "evaluation.json", check=False)
    assert result.returncode != 0
    assert not (root / "evaluation").exists()
    # CT is not a requirement or accessed with rendering disabled.
    config["render"] = False
    (root / "evaluation.json").write_text(json.dumps(config))
    run("replay-set", "--config", root / "evaluation.json")
    assert not list((root / "evaluation").rglob("strips_*.jpg"))


def test_cache_root_rejects_unwritable_before_saving(tmp_path):
    binary = os.environ.get("VC_CACHE_ROOT_BINARY")
    if not binary:
        replay = os.environ.get("VC_REPLAY_TEST_BINARY")
        binary = str(Path(replay).with_name("vc_cache_root")) if replay else None
    if not binary or not Path(binary).is_file():
        pytest.skip("requires rebuilt vc_cache_root")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root can write permission-restricted directories")
    env = dict(os.environ, VC3D_CONFIG_DIR=str(tmp_path / "config"))
    root = tmp_path / "readonly"
    root.mkdir()
    root.chmod(0o555)
    try:
        result = subprocess.run([binary, "--set", str(root)], env=env, capture_output=True, timeout=10)
        assert result.returncode != 0
        assert not (tmp_path / "config/cache.json").exists()
    finally:
        root.chmod(0o755)
    result = subprocess.run([binary, "--set", str(root)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(root)
    assert not list(root.iterdir())  # successful write probe cleans up its own files


def test_batched_tracer_one_and_four_workers_are_identical(replay_fixture):
    root, config, run = replay_fixture
    presence = zarr.open_array(str(root / "fiber_presence.zarr"), mode="r+")
    z, y, x = np.indices(presence.shape)
    presence[:] = (180 + (3 * x + 2 * y + z) % 70).astype(np.uint8)
    results = []
    for threads in (1, 4):
        config["output"] = "batch-" + str(threads)
        config["profile_tracing"] = True
        config["trace_args"] = ["--threads", str(threads), "--cone-angle-degrees", "10",
                                "--step-voxels", "2", "--beam-width", "4", "--beam-lookahead-steps", "2"]
        (root / "batch.json").write_text(json.dumps(config))
        run("replay-set", "--config", root / "batch.json")
        output = root / config["output"]
        results.append([json.loads((output / f"cases/{i}/trace.json").read_text())["replay"] for i in range(2)])
        for i in range(2):
            profile = json.loads((output / f"cases/{i}/profile.json").read_text())
            assert profile["trace"]["corner_points"] > 0
    assert results[0] == results[1]


@pytest.mark.parametrize("beam_width", [1, 4])
@pytest.mark.parametrize("readiness_entries", [0, 4096])
def test_live_gate_matches_offline_with_all_cases_waiting(replay_fixture, beam_width, readiness_entries):
    root, config, run = replay_fixture
    config["workers"] = 1
    config["trace_args"] += ["--beam-width", str(beam_width), "--beam-lookahead-steps", "2"]
    # Different normal/prediction grids force the non-fused batch route.
    normal = json.loads((root / "normal.json").read_text())
    normal["source_to_base"] = 2
    (root / "normal.json").write_text(json.dumps(normal))
    (root / "evaluation.json").write_text(json.dumps(config))
    run("replay-set", "--config", root / "evaluation.json")
    # Remove every prediction data chunk. Only the coordinator may publish it.
    chunks = {}
    for array in root.glob("fiber_*.zarr"):
        for path in array.iterdir():
            if not path.name.startswith("."):
                chunks[path] = path.read_bytes()
                path.unlink()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(15)
    config["output"] = "live-evaluation"
    # Rendering requests additional off-center interpolation chunks while CPU
    # permits are released; one worker/two cases also exercises render-lock waits.
    config["render"] = True
    config["profile_tracing"] = True
    config["readiness_cache_entries"] = readiness_entries
    config["live_endpoint"] = {"port": listener.getsockname()[1], "token": "synthetic-secret", "timeout_seconds": 15}
    (root / "live.json").write_text(json.dumps(config))
    errors, requests = [], []

    def serve():
        try:
            with listener, listener.accept()[0] as connection:
                connection.settimeout(15)
                with connection.makefile("rwb") as wire:
                    hello = json.loads(wire.readline())
                    assert hello == {"version": 1, "type": "hello", "token": "synthetic-secret"}

                    def publish(message):
                        assert message["type"] == "demand"
                        assert message["remaining_base_voxels"] >= 0
                        for item in message["chunks"]:
                            path = Path(item["array"]) / ".".join(map(str, item["index_zyx"]))
                            assert path in chunks
                            if not path.exists():
                                temporary = path.with_name(path.name + ".publishing")
                                temporary.write_bytes(chunks[path])
                                temporary.replace(path)
                        wire.write((json.dumps({"version": 1, "type": "ready", "request_id": message["request_id"]}) + "\n").encode())
                        wire.flush()

                    # Neither can progress until both demand requests arrive:
                    # this detects holding a CPU permit while waiting.
                    for _ in range(2):
                        requests.append(json.loads(wire.readline()))
                    assert len({item["case"] for item in requests}) == 2
                    for message in requests:
                        publish(message)
                    while line := wire.readline():
                        message = json.loads(line)
                        requests.append(message)
                        publish(message)
        except BaseException as error:
            errors.append(error)

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    result = run("replay-set", "--config", root / "live.json", check=False)
    server.join(20)
    assert not server.is_alive()
    assert not errors, errors
    assert result.returncode == 0, result.stdout + result.stderr
    assert requests
    for i in range(2):
        offline = json.loads((root / f"evaluation/cases/{i}/trace.json").read_text())
        live = json.loads((root / f"live-evaluation/cases/{i}/trace.json").read_text())
        assert live["replay"] == offline["replay"]
        measured = json.loads((root / f"live-evaluation/cases/{i}/profile.json").read_text())
        ready = measured["readiness"]
        assert ready["bridge_calls"] > 0
        assert ready["calls"] == ready["bridge_calls"] + ready["local_hits"]
        assert (ready["local_hits"] > 0) == (readiness_entries > 0)
        assert measured["trace"]["candidates"] > 0
        assert measured["trace"]["readiness_plan_points"] == measured["trace"]["candidates"]
        assert ready["bridge_lock_seconds"] >= 0
    assert "synthetic-secret" not in (root / "live-evaluation/summary.json").read_text()


def test_live_coordinator_error_is_not_a_geometric_failure(replay_fixture):
    root, config, run = replay_fixture
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(10)
    config["workers"] = 1
    config["live_endpoint"] = {"port": listener.getsockname()[1], "token": "failure-secret", "timeout_seconds": 10}
    (root / "evaluation.json").write_text(json.dumps(config))
    errors = []

    def serve():
        try:
            with listener, listener.accept()[0] as connection:
                connection.settimeout(10)
                with connection.makefile("rwb") as wire:
                    assert json.loads(wire.readline())["type"] == "hello"
                    for _ in range(2):
                        assert json.loads(wire.readline())["type"] == "demand"
                    wire.write(b'{"version":1,"type":"error","message":"synthetic inference worker failed"}\n')
                    wire.flush()
        except BaseException as error:
            errors.append(error)

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    result = run("replay-set", "--config", root / "evaluation.json", check=False)
    server.join(15)
    assert not server.is_alive()
    assert not errors, errors
    assert result.returncode != 0
    incomplete = (root / "evaluation/incomplete.json").read_text()
    assert json.loads(incomplete)["completed_cases"] == 0
    assert "failure-secret" not in incomplete
    assert not (root / "evaluation/summary.json").exists()


@pytest.mark.parametrize("resident", [False, True])
def test_complete_runner_cold_and_warm_with_identity_model(replay_fixture, monkeypatch, resident):
    root, config, run_native = replay_fixture
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from vesuvius.neural_tracing.fiber_trace_3d import live_evaluate as runner
    import torch
    torch.set_num_threads(1)
    monkeypatch.setattr(runner.inference_adapter, "FiberTrace3DPredictAdapter", SyntheticFiberAdapter)
    monkeypatch.setattr(runner, "_resolve_inference_precision", lambda *a, **kw: ("off", "test"))
    real_pool = runner.demand.DemandWorkerPool
    monkeypatch.setattr(runner.demand, "DemandWorkerPool",
                        lambda stores, readers, devices, **kw: real_pool(stores, readers, ["cpu"], **kw))
    original_output = subprocess.check_output

    def output(command, **kwargs):
        if Path(command[0]).name == "vc_cache_root":
            return str(root / "cache") + "\n"
        return original_output(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", output)
    checkpoint = root / "snapshot.pt"
    torch.save({"config": {}}, checkpoint)
    config["sources"][0]["input_volume"] = "ct.zarr"
    config["workers"] = 1
    config["inference"] = {"tile_size": 8, "overlap": 2, "border": 1, "scaledown_power": 0,
                           "chunk_size": 4, "input_workers": 2, "slots_per_gpu": 2, "compressor": "none"}
    borrowed_threads = []
    borrowed_connections = []
    owned_pids = []
    def borrow(arguments, count):
        import multiprocessing
        from vesuvius.neural_tracing.fiber_trace_3d.resident_trace import ResidentWorkers, serve_resident
        assert count == 1
        model = torch.nn.Conv3d(1, 1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        model.train()
        parent, child = multiprocessing.get_context("spawn").Pipe()
        borrowed_connections.extend((parent, child))
        def serve():
            assert child.recv() == "trace"
            descriptor = child.recv()
            serve_resident(child, model, 0, descriptor)
            assert model.training
        thread = threading.Thread(target=serve)
        borrowed_threads.append(thread)
        return ResidentWorkers([parent], arguments, thread.start, threading.Event())
    if resident:
        monkeypatch.setattr(SyntheticFiberAdapter, "load_model",
                            lambda *a, **k: pytest.fail("reloaded resident model"))
    for name in ("cold", "warm"):
        config["output"] = name
        if name == "warm":
            config["gpu_manager"] = {"enabled": True,
                "vram_bytes_per_gpu": 4 << 30, "cpu_cores": 1}
            from vesuvius.neural_tracing.fiber_trace_3d import demand_gpu
            monkeypatch.setattr(demand_gpu, "DemandGpuLease",
                lambda *a, **k: pytest.fail("warm cache acquired a manager lease"))
        (root / "live-run.json").write_text(json.dumps(config))
        runner.run(root / "live-run.json", checkpoint, "0",
                   os.environ["VC_REPLAY_TEST_BINARY"], render=True,
                   unmanaged=name == "cold",
                   worker_factory=borrow if resident else None,
                   on_process_start=owned_pids.extend if resident else None)
        summary = json.loads((root / name / "summary.json").read_text())
        assert summary["completed"] and summary["cases"] == 2 and summary["failures"] == 0
        assert summary["config"]["render"] is True
        for i in range(2):
            assert list((root / name / f"cases/{i}/rendered").glob("strips_*.jpg"))
            assert json.loads((root / name / f"cases/{i}/rendered/render.json").read_text())["format"] == "vc_fiber_inspection"
    cold = json.loads((root / "cold/inference_summary.json").read_text())
    warm = json.loads((root / "warm/inference_summary.json").read_text())
    assert cold["worker_pool"]["tiles"] > 0
    assert warm["worker_pool"] is None  # no model worker is even started on a full warm cache
    if resident:
        assert len(borrowed_threads) == 1
        assert len(set(owned_pids)) >= 4  # reference, accumulator, and native workers
        import psutil
        assert all(not psutil.pid_exists(pid) for pid in owned_pids)
        for thread in borrowed_threads:
            thread.join(2)
            assert not thread.is_alive()
        for connection in borrowed_connections:
            connection.close()
    for i in range(2):
        a = json.loads((root / f"cold/cases/{i}/trace.json").read_text())
        b = json.loads((root / f"warm/cases/{i}/trace.json").read_text())
        assert a["replay"] == b["replay"]
        cold_images = {p.name: p.read_bytes() for p in (root / f"cold/cases/{i}/rendered").glob("strips_*.jpg")}
        warm_images = {p.name: p.read_bytes() for p in (root / f"warm/cases/{i}/rendered").glob("strips_*.jpg")}
        assert cold_images == warm_images
