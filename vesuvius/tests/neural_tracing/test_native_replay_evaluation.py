"""End-to-end native replay/export/strip tests; no network or model inference.

Set VC_REPLAY_TEST_BINARY to the rebuilt vc_fiber_trace_metric executable.
"""
import json
import os
from pathlib import Path
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
    assert summary["mean_segment_length_mm"] == pytest.approx(.006)
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
    assert summary["version"] == 2
    assert summary["metric_definition"] == "complete-reference-partition-v1"
    assert summary["reliability_segments"] == 2
    assert summary["mean_segment_length_mm"] == pytest.approx(0.016)
    assert all(source["version"] == 5 for source in summary["sources"])
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


def test_complete_runner_cold_and_warm_with_identity_model(replay_fixture, monkeypatch):
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
    for name in ("cold", "warm"):
        config["output"] = name
        (root / "live-run.json").write_text(json.dumps(config))
        runner.run(root / "live-run.json", checkpoint, "0", os.environ["VC_REPLAY_TEST_BINARY"], render=True)
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
    for i in range(2):
        a = json.loads((root / f"cold/cases/{i}/trace.json").read_text())
        b = json.loads((root / f"warm/cases/{i}/trace.json").read_text())
        assert a["replay"] == b["replay"]
        cold_images = {p.name: p.read_bytes() for p in (root / f"cold/cases/{i}/rendered").glob("strips_*.jpg")}
        warm_images = {p.name: p.read_bytes() for p in (root / f"warm/cases/{i}/rendered").glob("strips_*.jpg")}
        assert cold_images == warm_images
