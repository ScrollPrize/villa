"""Tests for the Spiral service protocol.

Covers bearer authentication, API-key auto-generation, launch-time dataset
ownership, the artifact registry and its HTTP endpoints, session input
uploads with ephemeral staging, and dataset commits. The resident fitter is
faked; these tests exercise the service plumbing only.
"""

import argparse
import io
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spiral_service
from spiral_service import (ApiError, ArtifactRegistry, ExclusiveFileLock,
                            FileLockUnavailable, ServiceLogBuffer, ServiceState,
                            SpiralServer, _mapped_winding_ids,
                            _prepare_cleaned_lasagna_surface,
                            _raw_run_diff_rgba, _sample_rgba_through_map,
                            _validate_tifxyz_output_step,
                            load_or_create_api_key, parse_gpu_ids,
                            parse_session_name)
from fit_session import API_VERSION, SpiralInputPaths, resolve_dataset_root
from config import Config


class FakeSession:
    def __init__(self):
        self.state = "Paused"
        self.run_calls = []
        self.run_config = {
            "sample_count_patches_per_step": 360,
            "loss_weight_patch_radius": 8.0,
            "loss_start_patch_dt": 25_000,
            "loss_start_track_dt": 10_000,
            "output_save_png_visualizations": False,
            "track_length_bin_weights": None,
            "track_max_track_crossing_per_step": 0,
            "track_min_sample_spacing": 20.0,
            "track_max_sample_spacing": 60.0,
            "track_min_walk_steps_per_track": 24,
            "track_max_walk_steps_per_track": 256,
            "track_min_walks_per_track": 2,
            "track_max_walks_per_track": 4,
        }
        self.default_advanced_config = {
            "optimizer_learning_rate": 3e-5,
            "sample_count_patches_per_step": 360,
            "loss_weight_patch_radius": 8.0,
            "track_crossing_precompute_max": 8,
            "track_crossing_mode": "track_walk",
            "track_walk_minimum_cycle_travel": 20.0,
        }
        self.saved = []
        self.closed = False
        self.path_change_calls = []

    def status(self):
        return {
            "state": self.state, "phase": self.state, "current_iteration": 5,
            "target_iteration": 5, "latest_metrics": {}, "warnings": [],
            "error": None, "preview_manifest_path": None, "preview_generation": 0,
            "supports_input_incorporation": True,
            "run_config": dict(self.run_config),
            "run_config_limits": {"track_max_track_crossing_per_step": 8},
            "default_advanced_config": dict(self.default_advanced_config),
        }

    def run(self, count, pending_inputs=None, mark_incorporated=None,
            influence_config=None, run_config=None, path_changes=None):
        self.run_calls.append((count, list(pending_inputs or []), mark_incorporated,
                               dict(influence_config or {}), dict(run_config or {})))
        self.path_change_calls.append(dict(path_changes or {}))
        self.run_config.update(run_config or {})
        return 5 + count

    def save_checkpoint(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"PK\x03\x04checkpoint")
        self.saved.append(path)
        return path

    def close(self):
        self.closed = True


def _attach_fake_session(state, output_directory, dataset_root=""):
    state.session = FakeSession()
    state.session_generation += 1
    state.session_id = f"spiral-test-{state.session_generation}"
    state.session_paths = SpiralInputPaths.from_mapping({
        "dataset_root": str(dataset_root),
        "output_directory": str(output_directory),
        "verified_patches": str(Path(dataset_root) / "verified_patches") if dataset_root else "",
        "fibers": str(Path(dataset_root) / "fibers") if dataset_root else "",
    })
    state.session_request = {
        "paths": state.session_paths.manifest(),
        "run": {"config": Config().as_dict()},
    }
    state.session_revision += 1
    return state.session


def _planned_run(state, request):
    request = dict(request)
    configuration = Config(request.pop("run_config", {})).as_dict()
    plan = state.plan_run({
        "configuration": configuration,
        "iterations": request.pop("iterations"),
        "influence": request.pop("influence_config", {}),
        "expected_session_revision": state.session_revision,
    })
    return state.run({"plan_token": plan["plan_token"]})


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _upload_input(state, kind, input_id, files, role=None):
    request = {
        "kind": kind, "id": input_id,
        "files": [{"name": name, "size": len(data), "sha256": _digest(data)}
                  for name, data in files.items()],
    }
    if role:
        request["role"] = role
    upload_id = state.begin_upload(request)["upload_id"]
    for name, data in files.items():
        state.receive_upload_file(upload_id, name, io.BytesIO(data), len(data))
    return upload_id


def _commit_pcl_process(dataset, output, input_id, ready, start, result):
    try:
        state = ServiceState()
        _attach_fake_session(state, output, dataset)
        upload_id = _upload_input(
            state, "pcl", input_id, PCL_FILES, role="patch_overlap")
        state.finalize_upload(upload_id)
        ready.put(input_id)
        if not start.wait(10):
            raise RuntimeError("timed out waiting for concurrent commit start")
        state.commit_inputs()
        result.put((input_id, "ok"))
    except BaseException as exc:
        result.put((input_id, f"{type(exc).__name__}: {exc}"))


PATCH_FILES = {
    "meta.json": json.dumps({"format": "tifxyz"}).encode(),
    "x.tif": b"x-raster", "y.tif": b"y-raster", "z.tif": b"z-raster",
}
FIBER_FILES = {"fiber.json": json.dumps(
    {"type": "vc3d_fiber", "control_points": [[0, 0, 0], [4, 4, 4]]}).encode()}
PCL_FILES = {"pcls.json": json.dumps({
    "vc_pointcollections_json_version": "1",
    "collections": {"0": {"name": "c", "points": {}}},
}).encode()}


class ApiKeyTests(unittest.TestCase):
    def test_key_is_generated_with_owner_only_mode_and_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sub" / "spiral_api_key"
            key, created = load_or_create_api_key(path)
            self.assertTrue(created)
            self.assertGreaterEqual(len(key), 32)
            mode = stat.S_IMODE(path.stat().st_mode)
            self.assertEqual(mode, stat.S_IRUSR | stat.S_IWUSR)
            again, created_again = load_or_create_api_key(path)
            self.assertFalse(created_again)
            self.assertEqual(key, again)


class GpuSelectionTests(unittest.TestCase):
    def test_single_gpu_zero_is_the_default_service_state(self):
        self.assertEqual(ServiceState().health()["gpus"], [0])

    def test_comma_separated_gpu_ids_are_parsed_in_order(self):
        self.assertEqual(parse_gpu_ids("0,1,2,3"), (0, 1, 2, 3))
        self.assertEqual(parse_gpu_ids(" 3, 1 "), (3, 1))

    def test_invalid_gpu_lists_are_rejected(self):
        for value in ("", "0,", "-1", "gpu0", "1,1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_gpu_ids(value)


class NamedSessionTests(unittest.TestCase):
    def test_safe_session_names_are_accepted(self):
        for value in ("alice", "gpu-1", "team_one", "run.2026"):
            with self.subTest(value=value):
                self.assertEqual(parse_session_name(value), value)

    def test_unsafe_session_names_are_rejected(self):
        for value in ("", ".", "..", "../alice", "alice/bob", "a b", "_alice",
                      "a" * 65):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_session_name(value)

    def test_named_resolution_isolates_output_but_shares_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "umbilicus.json").write_text("{}")
            (root / "verified_patches").mkdir()
            unnamed = resolve_dataset_root(root)
            alice = resolve_dataset_root(root, session_name="alice")
            bob = resolve_dataset_root(root, session_name="bob")
            self.assertEqual(unnamed.resolved["output_directory"],
                             str(root / "spiral_output"))
            self.assertEqual(alice.resolved["output_directory"],
                             str(root / "spiral_output" / "alice"))
            self.assertEqual(bob.resolved["output_directory"],
                             str(root / "spiral_output" / "bob"))
            self.assertEqual(alice.resolved["cache_directory"],
                             bob.resolved["cache_directory"])

    def test_exclusive_lock_rejects_duplicate_live_owner_and_releases(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session" / ".spiral-service.lock"
            first = ExclusiveFileLock(path).acquire()
            second = ExclusiveFileLock(path)
            with self.assertRaises(FileLockUnavailable):
                second.acquire()
            first.release()
            second.acquire()
            second.release()


class HttpServiceFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state = ServiceState()
        SpiralServer.allow_reuse_address = False
        self.server = SpiralServer(("127.0.0.1", 0), ["secret-key"], self.state)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        self.temporary.cleanup()

    def request(self, method, path, *, token="secret-key", body=None, headers=None,
                nonce_header=False):
        request = urllib.request.Request(self.base + path, method=method)
        if token is not None:
            if nonce_header:
                request.add_header("X-Spiral-Nonce", token)
            else:
                request.add_header("Authorization", f"Bearer {token}")
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        data = json.dumps(body).encode() if body is not None else None
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, data=data, timeout=10) as response:
                return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as error:
            return error.code, error.read(), dict(error.headers)


class AuthenticationTests(HttpServiceFixture):
    def test_missing_malformed_wrong_and_correct_credentials(self):
        status, _, _ = self.request("GET", "/health", token=None)
        self.assertEqual(status, 401)
        status, _, _ = self.request("GET", "/health", token="")
        self.assertEqual(status, 401)
        status, _, _ = self.request("GET", "/health", token="wrong-key")
        self.assertEqual(status, 401)
        status, payload, _ = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(payload)["ready"])

    def test_nonce_header_remains_a_compatibility_alias(self):
        status, _, _ = self.request("GET", "/health", token="secret-key",
                                    nonce_header=True)
        self.assertEqual(status, 200)

    def test_health_carries_service_identity(self):
        _, payload, _ = self.request("GET", "/health")
        health = json.loads(payload)
        self.assertEqual(health["api_version"], API_VERSION)
        self.assertIn("service_version", health)
        self.assertIn("service_name", health)
        self.assertIn("session_name", health)
        self.assertIn("session_generation", health)
        self.assertIn("service_generation", health)


class RestartTests(HttpServiceFixture):
    def test_restart_requires_authentication_and_command_id(self):
        status, _, _ = self.request(
            "POST", "/service/restart", token=None,
            body={"command_id": "restart-1"})
        self.assertEqual(status, 401)
        self.assertFalse(self.server.restart_requested.is_set())

        status, payload, _ = self.request("POST", "/service/restart", body={})
        self.assertEqual(status, 400)
        self.assertIn("command_id", json.loads(payload)["error"])
        self.assertFalse(self.server.restart_requested.is_set())

    def test_restart_is_acknowledged_and_duplicate_requests_are_idempotent(self):
        body = {"command_id": "restart-1"}
        status, payload, _ = self.request("POST", "/service/restart", body=body)
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(payload)["restarting"])

        status, duplicate, _ = self.request("POST", "/service/restart", body=body)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(duplicate), json.loads(payload))

        self.assertTrue(self.server._restart_scheduled)
        self.assertTrue(self.server.restart_requested.wait(2))


class LogStreamingTests(HttpServiceFixture):
    def test_authenticated_log_reads_are_incremental(self):
        self.state.logs.write("stdout", "loading inputs\n")
        self.state.logs.write("stderr", "iteration 1\niteration 2\n")

        status, payload, _ = self.request("GET", "/logs?after=0")
        self.assertEqual(status, 200)
        first = json.loads(payload)
        self.assertEqual(
            [(entry["stream"], entry["text"]) for entry in first["entries"]],
            [("stdout", "loading inputs"),
             ("stderr", "iteration 1"),
             ("stderr", "iteration 2")])
        self.assertEqual(first["next_sequence"], 3)

        self.state.logs.write("stdout", "iteration 3\n")
        status, payload, _ = self.request(
            "GET", f"/logs?after={first['next_sequence']}")
        self.assertEqual(status, 200)
        second = json.loads(payload)
        self.assertEqual([entry["text"] for entry in second["entries"]],
                         ["iteration 3"])
        self.assertEqual(second["next_sequence"], 4)

    def test_log_cursor_validation_and_authentication(self):
        status, _, _ = self.request("GET", "/logs?after=not-a-number")
        self.assertEqual(status, 400)
        status, _, _ = self.request("GET", "/logs?after=-1")
        self.assertEqual(status, 400)
        status, _, _ = self.request("GET", "/logs?after=0", token=None)
        self.assertEqual(status, 401)

    def test_bounded_log_buffer_reports_dropped_entries_and_cursor_reset(self):
        logs = ServiceLogBuffer(max_entries=2)
        logs.write("stdout", "one\ntwo\nthree\n")
        result = logs.read_after(0)
        self.assertEqual([entry["text"] for entry in result["entries"]],
                         ["two", "three"])
        self.assertEqual(result["dropped"], 1)

        restarted_client = logs.read_after(100)
        self.assertTrue(restarted_client["cursor_reset"])
        self.assertEqual([entry["text"] for entry in restarted_client["entries"]],
                         ["two", "three"])

    def test_routine_poll_access_lines_do_not_fill_the_log_buffer(self):
        logs = ServiceLogBuffer()
        logs.write("stderr", 'SPIRAL_HTTP "GET /session/status HTTP/1.1" 200 -\n')
        logs.write("stderr", 'SPIRAL_HTTP "GET /logs?after=4 HTTP/1.1" 200 -\n')
        logs.write("stderr", "useful fitter warning\n")
        self.assertEqual([entry["text"] for entry in logs.read_after(0)["entries"]],
                         ["useful fitter warning"])

    def test_carriage_return_progress_redraws_are_relayed_as_complete_lines(self):
        logs = ServiceLogBuffer()
        logs.write("stderr", "\rloading patches:  25%|██▌       | 1/4")
        logs.write("stderr", "\rloading patches: 100%|██████████| 4/4\n")
        logs.write("stderr", "\r 40%|████      | 400/1000")
        logs.write("stderr", "\r100%|██████████| 1000/1000\n")

        self.assertEqual(
            [entry["text"] for entry in logs.read_after(0)["entries"]],
            [
                "loading patches:  25%|██▌       | 1/4",
                "loading patches: 100%|██████████| 4/4",
                " 40%|████      | 400/1000",
                "100%|██████████| 1000/1000",
            ])


class ArtifactHttpTests(HttpServiceFixture):
    def _register_artifact(self, contents=b"0123456789abcdef"):
        artifact_root = self.root / "artifact"
        artifact_root.mkdir()
        (artifact_root / "manifest.json").write_text("{}")
        (artifact_root / "payload.bin").write_bytes(contents)
        return self.state.artifacts.register_directory(
            "spiral-preview", "session-1", 1, artifact_root, "manifest.json",
            delete_root_on_prune=True), artifact_root

    def test_manifest_exposes_only_registered_files(self):
        ref, _ = self._register_artifact()
        status, payload, _ = self.request("GET", f"/artifacts/{ref['id']}/manifest")
        self.assertEqual(status, 200)
        manifest = json.loads(payload)
        names = {entry["name"] for entry in manifest["files"]}
        self.assertEqual(names, {"manifest.json", "payload.bin"})
        self.assertEqual(manifest["entry_point"], "manifest.json")
        for entry in manifest["files"]:
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")

    def test_unknown_artifact_is_not_found_and_pruned_is_gone(self):
        ref, _ = self._register_artifact()
        status, _, _ = self.request("GET", "/artifacts/nonexistent/manifest")
        self.assertEqual(status, 404)
        self.state.artifacts.prune("spiral-preview", "session-1", 0)
        status, _, _ = self.request("GET", f"/artifacts/{ref['id']}/manifest")
        self.assertEqual(status, 410)

    def test_file_download_with_range_resume(self):
        ref, _ = self._register_artifact(b"0123456789abcdef")
        status, payload, headers = self.request(
            "GET", f"/artifacts/{ref['id']}/files/payload.bin")
        self.assertEqual(status, 200)
        self.assertEqual(payload, b"0123456789abcdef")
        self.assertEqual(headers.get("Accept-Ranges"), "bytes")
        status, payload, headers = self.request(
            "GET", f"/artifacts/{ref['id']}/files/payload.bin",
            headers={"Range": "bytes=10-"})
        self.assertEqual(status, 206)
        self.assertEqual(payload, b"abcdef")
        self.assertEqual(headers.get("Content-Range"), "bytes 10-15/16")

    def test_traversal_and_absolute_paths_are_rejected(self):
        ref, artifact_root = self._register_artifact()
        secret = self.root / "secret.txt"
        secret.write_text("secret")
        for name in ("../secret.txt", "%2e%2e/secret.txt", "..%2fsecret.txt",
                     "/etc/passwd", "a/../../secret.txt"):
            status, _, _ = self.request(
                "GET", f"/artifacts/{ref['id']}/files/{name}")
            self.assertIn(status, (403, 404), name)

    def test_symlink_escape_is_rejected(self):
        artifact_root = self.root / "artifact-symlink"
        artifact_root.mkdir()
        (artifact_root / "manifest.json").write_text("{}")
        secret = self.root / "outside.txt"
        secret.write_text("outside")
        (artifact_root / "link.txt").symlink_to(secret)
        ref = self.state.artifacts.register_directory(
            "spiral-preview", "session-1", 2, artifact_root, "manifest.json")
        status, payload, _ = self.request(
            "GET", f"/artifacts/{ref['id']}/manifest")
        names = {entry["name"] for entry in json.loads(payload)["files"]}
        self.assertNotIn("link.txt", names)
        status, _, _ = self.request(
            "GET", f"/artifacts/{ref['id']}/files/link.txt")
        self.assertIn(status, (403, 404))

    def test_inflight_download_defers_pruning_deletion(self):
        ref, artifact_root = self._register_artifact()
        artifact, path, info = self.state.artifacts.acquire_file(
            ref["id"], "payload.bin")
        self.state.artifacts.prune("spiral-preview", "session-1", 0)
        self.assertTrue(artifact_root.exists(),
                        "artifact deleted while a download held a reference")
        self.assertEqual(path.read_bytes(), b"0123456789abcdef")
        self.state.artifacts.release(artifact)
        self.assertFalse(artifact_root.exists())


class DatasetOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "umbilicus.json").write_text("{}")
        (self.root / "verified_patches").mkdir()
        (self.root / "spiral_output").mkdir()
        self.resolution = resolve_dataset_root(self.root)
        self.assertTrue(self.resolution.ok)
        self.state = ServiceState(dataset_root=str(self.root),
                                  dataset_resolution=self.resolution)

    def tearDown(self):
        self.temporary.cleanup()

    def test_dataset_endpoint_advertises_resolution(self):
        response = self.state.dataset()
        self.assertEqual(response["root"], str(self.root))
        self.assertIn("umbilicus", response["resolved"])

    def test_dataset_endpoint_advertises_named_session(self):
        resolution = resolve_dataset_root(self.root, session_name="alice")
        state = ServiceState(
            dataset_root=str(self.root), dataset_resolution=resolution,
            session_name="alice")
        response = state.dataset()
        self.assertEqual(response["session_name"], "alice")
        self.assertEqual(
            response["resolved"]["output_directory"],
            str(self.root / "spiral_output" / "alice"))

    def test_dataset_resolution_discovers_drawn_control_points(self):
        drawn = self.root / "drawn_control_points.json"
        drawn.write_text(json.dumps({
            "vc_pointcollections_json_version": "1",
            "collections": {"0": {"name": "drawn", "points": {}}},
        }))
        resolution = resolve_dataset_root(self.root)
        entries = [entry for entry in resolution.pcl_inputs
                   if entry["role"] == "drawn_control_points"]
        self.assertEqual(entries, [{
            "path": str(drawn), "role": "drawn_control_points", "required": False,
        }])

    def test_dataset_resolution_discovers_same_winding_collections(self):
        same_winding = self.root / "same_windings.json"
        same_winding.write_text(json.dumps({
            "vc_pointcollections_json_version": "1",
            "collections": {"0": {"name": "same_winding_0001", "points": {}}},
        }))
        resolution = resolve_dataset_root(self.root)
        entries = [entry for entry in resolution.pcl_inputs
                   if entry["role"] == "same_winding"]
        self.assertEqual(entries, [{
            "path": str(same_winding), "role": "same_winding", "required": False,
        }])

    def test_arbitrary_root_resolution_is_refused(self):
        with self.assertRaises(ApiError) as caught:
            self.state.resolve("/somewhere/else")
        self.assertEqual(caught.exception.status, 403)
        # The launch dataset itself (or an empty root) returns the resolution.
        self.assertEqual(self.state.resolve("")["root"], str(self.root))
        self.assertEqual(self.state.resolve(str(self.root))["root"], str(self.root))

    def test_load_rejects_client_base_input_paths(self):
        with self.assertRaises(ApiError) as caught:
            self.state.load({"paths": {"umbilicus": "/attacker/umbilicus.json"},
                             "run": {"z_begin": 0, "z_end": 10}})
        self.assertEqual(caught.exception.status, 400)
        fields = {detail["field"] for detail in caught.exception.details}
        self.assertEqual(fields, {"umbilicus"})

    def test_load_rejects_unadvertised_checkpoint(self):
        with self.assertRaises(ApiError) as caught:
            self.state.load({"paths": {"checkpoint": "/attacker/model.ckpt"},
                             "run": {"z_begin": 0, "z_end": 10}})
        self.assertEqual(caught.exception.status, 400)

    def test_dataset_request_uses_resolved_paths_without_input_toggles(self):
        request = self.state._dataset_session_request({
            "run": {"z_begin": 0, "z_end": 10},
        })
        self.assertEqual(
            request["paths"]["verified_patches"],
            str(self.root / "verified_patches"))
        self.assertEqual(request["paths"]["unverified_patches"], "")

    def test_status_advertises_canonical_active_session_request(self):
        config = {
            "dense_spacing_mode": "grad_mag",
            "loss_weight_dense_spacing": 0,
            "loss_weight_dense_normals": 0,
            "loss_weight_shell_outer": 0,
            "loss_weight_shell_patch_radius": 0,
            "loss_weight_patch_radius": 7.5,
        }
        request = {
            "run": {
                "z_begin": 100,
                "z_end": 900,
                "scroll_name": "attached-scroll",
                "config": config,
            },
            "preview": {"first_winding": 12, "variant": "raw"},
        }
        with mock.patch("spiral_runtime.create_session",
                        return_value=FakeSession()):
            response = self.state.load(request)

        attached = response["session_request"]
        self.assertEqual(attached, self.state.status()["session_request"])
        self.assertEqual(attached["paths"]["dataset_root"], str(self.root))
        self.assertEqual(attached["paths"]["verified_patches"],
                         str(self.root / "verified_patches"))
        self.assertEqual(attached["paths"]["fibers"], "")
        self.assertEqual(attached["run"]["z_begin"], 100)
        self.assertEqual(attached["run"]["z_end"], 900)
        self.assertEqual(attached["run"]["scroll_name"], "attached-scroll")
        self.assertEqual(attached["run"]["config"], config)
        self.assertEqual(attached["preview"],
                         {"first_winding": 12, "variant": "raw"})

        self.state.delete()
        self.assertIsNone(self.state.status()["session_request"])

    def test_failed_load_does_not_advertise_a_session_request(self):
        request = {
            "run": {
                "z_begin": 0,
                "z_end": 10,
                "config": {
                    "dense_spacing_mode": "grad_mag",
                    "loss_weight_dense_spacing": 0,
                    "loss_weight_dense_normals": 0,
                    "loss_weight_shell_outer": 0,
                    "loss_weight_shell_patch_radius": 0,
                },
            },
        }
        with mock.patch("spiral_runtime.create_session",
                        side_effect=RuntimeError("startup failed")):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                self.state.load(request)
        status = self.state.status()
        self.assertIsNone(status["session_id"])
        self.assertIsNone(status["session_request"])

    def test_save_checkpoint_is_constrained_to_output_directory(self):
        _attach_fake_session(self.state, self.root / "spiral_output", self.root)
        with self.assertRaises(ApiError) as caught:
            self.state.save_checkpoint({"path": str(self.root / "elsewhere.ckpt")})
        self.assertEqual(caught.exception.status, 400)
        response = self.state.save_checkpoint(
            {"path": str(self.root / "spiral_output" / "manual.ckpt")})
        self.assertTrue(response["checkpoint_path"].endswith("manual.ckpt"))


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "verified_patches").mkdir(parents=True)
        (self.dataset / "fibers").mkdir()
        self.output = self.root / "output"
        self.output.mkdir()
        self.state = ServiceState()

    def tearDown(self):
        self.temporary.cleanup()

    def _session(self):
        return _attach_fake_session(self.state, self.output, self.dataset)

    def test_upload_requires_an_active_session(self):
        with self.assertRaises(ApiError) as caught:
            self.state.begin_upload({"kind": "patch", "id": "p1",
                                     "files": [{"name": "meta.json", "size": 1,
                                                "sha256": "0" * 64}]})
        self.assertEqual(caught.exception.status, 409)

    def test_unsafe_identifiers_and_names_are_rejected(self):
        self._session()
        for bad_id in ("../p", "a/b", ".hidden", "", "a" * 200):
            with self.assertRaises(ApiError, msg=bad_id):
                self.state.begin_upload({"kind": "patch", "id": bad_id,
                                         "files": [{"name": "meta.json", "size": 1,
                                                    "sha256": "0" * 64}]})
        for bad_name in ("../x", "/abs", "a//b", "a/../b", "..", "a\\b"):
            with self.assertRaises(ApiError, msg=bad_name):
                self.state.begin_upload({"kind": "patch", "id": "p1",
                                         "files": [{"name": bad_name, "size": 1,
                                                    "sha256": "0" * 64}]})

    def test_finalize_verifies_content_and_publishes_atomically(self):
        self._session()
        upload_id = _upload_input(self.state, "patch", "patch-1", PATCH_FILES)
        response = self.state.finalize_upload(upload_id)
        record = response["input"]
        self.assertEqual(record["state"], "pending")
        published = Path(record["path"])
        self.assertTrue((published / "meta.json").is_file())
        self.assertIn(".spiral-ephemeral", str(published))
        status = self.state.status()
        self.assertEqual(status["ephemeral_inputs"][0]["id"], "patch-1")
        self.assertEqual(status["default_advanced_config"]["optimizer_learning_rate"], 3e-5)
        self.assertNotEqual(status["default_advanced_config"], status["run_config"])
        # Finalize is idempotent.
        self.assertEqual(self.state.finalize_upload(upload_id)["input"]["id"],
                         "patch-1")

    def test_finalize_rejects_missing_files_and_digest_mismatch(self):
        self._session()
        data = PATCH_FILES["meta.json"]
        request = {"kind": "patch", "id": "p2", "files": [
            {"name": "meta.json", "size": len(data), "sha256": _digest(data)},
            {"name": "x.tif", "size": 4, "sha256": _digest(b"xxxx")},
        ]}
        upload_id = self.state.begin_upload(request)["upload_id"]
        self.state.receive_upload_file(upload_id, "meta.json", io.BytesIO(data), len(data))
        with self.assertRaisesRegex(ApiError, "missing declared files"):
            self.state.finalize_upload(upload_id)
        with self.assertRaisesRegex(ApiError, "SHA-256"):
            self.state.receive_upload_file(upload_id, "x.tif", io.BytesIO(b"yyyy"), 4)
        ephemeral = self.output / ".spiral-ephemeral"
        self.assertFalse(any(ephemeral.rglob("*")) if ephemeral.exists() else False,
                         "nothing may be published before finalize succeeds")

    def test_finalize_rejects_invalid_patch_and_untyped_json(self):
        self._session()
        bad_patch = dict(PATCH_FILES)
        bad_patch["meta.json"] = json.dumps({"format": "not-tifxyz"}).encode()
        upload_id = _upload_input(self.state, "patch", "bad-patch", bad_patch)
        with self.assertRaisesRegex(ApiError, "tifxyz"):
            self.state.finalize_upload(upload_id)
        untyped = {"fiber.json": json.dumps({"control_points": []}).encode()}
        upload_id = _upload_input(self.state, "fiber", "bad-fiber", untyped)
        with self.assertRaisesRegex(ApiError, "vc3d_fiber"):
            self.state.finalize_upload(upload_id)
        bad_pcl = {"pcl.json": json.dumps({"some": "json"}).encode()}
        upload_id = _upload_input(self.state, "pcl", "bad-pcl", bad_pcl,
                                  role="patch_overlap")
        with self.assertRaisesRegex(ApiError, "vc_pointcollections"):
            self.state.finalize_upload(upload_id)

    def test_pcl_uploads_require_a_role(self):
        self._session()
        with self.assertRaisesRegex(ApiError, "role"):
            self.state.begin_upload({"kind": "pcl", "id": "roleless", "files": [
                {"name": "pcl.json", "size": 1, "sha256": "0" * 64}]})

    def test_drawn_control_points_are_forwarded_to_the_next_run(self):
        session = self._session()
        upload_id = _upload_input(self.state, "pcl", "drawn-1", PCL_FILES,
                                  role="drawn_control_points")
        self.state.finalize_upload(upload_id)
        _planned_run(self.state, {"iterations": 2})
        _, pending, _, _, _ = session.run_calls[-1]
        self.assertEqual([(record["id"], record["role"]) for record in pending],
                         [("drawn-1", "drawn_control_points")])

    def test_quota_is_enforced(self):
        self._session()
        original = spiral_service.EPHEMERAL_QUOTA_BYTES
        spiral_service.EPHEMERAL_QUOTA_BYTES = 10
        try:
            with self.assertRaisesRegex(ApiError, "quota"):
                _upload_input(self.state, "patch", "big", PATCH_FILES)
        finally:
            spiral_service.EPHEMERAL_QUOTA_BYTES = original

    def test_deleted_and_aborted_uploads_leave_no_partial_data(self):
        self._session()
        upload_id = _upload_input(self.state, "patch", "aborted", PATCH_FILES)
        staging = self.output / ".spiral-upload-staging" / upload_id
        self.assertTrue(staging.exists())
        self.state.delete_upload(upload_id)
        self.assertFalse(staging.exists())

    def test_expired_uploads_are_garbage_collected(self):
        self._session()
        upload_id = _upload_input(self.state, "patch", "stale", PATCH_FILES)
        self.state.uploads[upload_id].created -= spiral_service.UPLOAD_GC_SECONDS + 1
        self.state.gc_uploads()
        self.assertNotIn(upload_id, self.state.uploads)
        self.assertFalse((self.output / ".spiral-upload-staging" / upload_id).exists())

    def test_run_passes_pending_inputs_and_marks_incorporated(self):
        session = self._session()
        upload_id = _upload_input(self.state, "fiber", "fiber-1", FIBER_FILES)
        self.state.finalize_upload(upload_id)
        _planned_run(self.state, {"iterations": 10})
        count, pending, mark, influence, _ = session.run_calls[-1]
        self.assertEqual(count, 10)
        self.assertEqual([record["id"] for record in pending], ["fiber-1"])
        self.assertEqual(influence, {})
        mark(pending)
        self.assertEqual(self.state.status()["ephemeral_inputs"][0]["state"],
                         "incorporated")
        # A later run does not re-incorporate.
        _planned_run(self.state, {"iterations": 5})
        self.assertEqual(session.run_calls[-1][1], [])

    def test_run_passes_and_validates_transient_influence_config(self):
        session = self._session()
        influence = {
            "influence_enabled": True,
            "influence_z": 1200,
            "influence_windings": 2.5,
            "influence_theta_frac": 0.2,
            "influence_disable_dt_frac": 0.4,
            "influence_sigma": 0.25,
            "sample_count_influence_footprint_points": 512,
            "sample_count_influence_anchor_lattice_points": 2000,
            "sample_count_influence_anchor_geometry_points": 1000,
            "sample_count_influence_anchor_samples_per_step": 128,
            "influence_anchor_ramp_power": 3.0,
            "loss_weight_anchor": 15.0,
        }
        _planned_run(self.state, {"iterations": 10, "influence_config": influence})
        self.assertEqual(session.run_calls[-1][3], influence)

        with self.assertRaises(ApiError) as caught:
            _planned_run(self.state, {"iterations": 10, "influence_config": {
                "influence_theta_frac": 1.5,
            }})
        self.assertEqual(caught.exception.status, 400)

    def test_run_passes_and_validates_mutable_training_config(self):
        session = self._session()
        config = {
            "sample_count_patches_per_step": 240,
            "loss_weight_patch_radius": 3.5,
            "loss_start_track_dt": None,
            "output_save_png_visualizations": True,
            "track_length_bin_weights": [0.2, 0.3, 0.5],
            "track_max_track_crossing_per_step": 3,
            "track_min_sample_spacing": 12.0,
            "track_max_sample_spacing": 32.0,
            "track_min_walk_steps_per_track": 18,
            "track_max_walk_steps_per_track": 96,
            "track_max_walks_per_track": 5,
        }

        response = _planned_run(self.state, {"iterations": 10, "run_config": config})

        self.assertEqual(session.run_calls[-1][4], config)
        self.assertEqual(response["run_config"]["sample_count_patches_per_step"], 240)

        with self.assertRaisesRegex(ApiError, "Start New Fit"):
            _planned_run(self.state, {"iterations": 10, "run_config": {
                "model_num_flow_stages": 2,
            }})
        with self.assertRaisesRegex(ValueError, "Invalid value"):
            _planned_run(self.state, {"iterations": 10, "run_config": {
                "output_save_png_visualizations": 1,
            }})
        with self.assertRaisesRegex(ValueError, "vector length"):
            _planned_run(self.state, {"iterations": 10, "run_config": {
                "track_length_bin_weights": [1, 2],
            }})

    def test_run_accepts_advertised_zero_count_for_disabled_input(self):
        session = self._session()
        session.run_config["sample_count_dense_attachment_points"] = 0

        response = _planned_run(self.state, {"iterations": 10, "run_config": {
            "sample_count_dense_attachment_points": 0,
        }})

        self.assertEqual(session.run_calls[-1][4], {
            "sample_count_dense_attachment_points": 0,
        })
        self.assertEqual(response["run_config"]["sample_count_dense_attachment_points"], 0)

    def test_outer_shell_path_is_a_shell_only_run_change(self):
        session = self._session()
        shell = self.dataset / "outer_shell_v2"
        shell.mkdir()
        inputs = self.state.session_paths.manifest()
        inputs["outer_shell"] = str(shell)
        plan = self.state.plan_run({
            "configuration": Config().as_dict(),
            "iterations": 3,
            "inputs": inputs,
            "expected_session_revision": self.state.session_revision,
        })

        self.assertFalse(plan["session_reload_required"])
        self.assertEqual(plan["affected_prepared_inputs"], ["shell"])
        self.assertEqual(plan["input_changes"][0]["runtime_impact"],
                         "shell_reload")
        self.state.run({"plan_token": plan["plan_token"]})
        self.assertEqual(session.path_change_calls[-1], {
            "outer_shell": str(shell),
        })

    def test_non_shell_path_and_patch_erosion_still_require_reload(self):
        self._session()
        inputs = self.state.session_paths.manifest()
        inputs["verified_patches"] = str(self.dataset / "other-patches")
        configuration = Config({
            "patch_erode_patches": (
                Config().patch_erode_patches + 1),
        }).as_dict()
        plan = self.state.plan_run({
            "configuration": configuration,
            "iterations": 3,
            "inputs": inputs,
            "expected_session_revision": self.state.session_revision,
        })

        self.assertTrue(plan["session_reload_required"])
        with self.assertRaisesRegex(ApiError, "reloading fit inputs"):
            self.state.run({"plan_token": plan["plan_token"]})

    def test_new_session_does_not_see_previous_ephemeral_inputs(self):
        self._session()
        upload_id = _upload_input(self.state, "fiber", "fiber-1", FIBER_FILES)
        self.state.finalize_upload(upload_id)
        ephemeral_dir = self.state._session_ephemeral_dir()
        self.assertTrue(ephemeral_dir.exists())
        self.state.delete()
        self.assertEqual(self.state.ephemeral_records, [])
        self.assertFalse(ephemeral_dir.exists())


def _zip_checkpoint_bytes(payload=b"payload"):
    import io as _io
    import zipfile
    buffer = _io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.pkl", payload)
    return buffer.getvalue()


class CheckpointUploadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "umbilicus.json").write_text("{}")
        (self.root / "verified_patches").mkdir()
        (self.root / "spiral_output").mkdir()
        self.resolution = resolve_dataset_root(self.root)
        self.assertTrue(self.resolution.ok)
        self.state = ServiceState(dataset_root=str(self.root),
                                  dataset_resolution=self.resolution)

    def tearDown(self):
        self.temporary.cleanup()

    def _upload_checkpoint(self, name, data=None):
        data = data if data is not None else _zip_checkpoint_bytes()
        request = {
            "kind": "checkpoint",
            "id": name,
            "files": [{
                "name": name,
                "size": len(data),
                "sha256": _digest(data),
            }],
        }
        begin = self.state.begin_upload(request)
        if begin.get("deduplicated"):
            return begin["input"]
        upload_id = begin["upload_id"]
        self.state.receive_upload_file(
            upload_id, name, io.BytesIO(data), len(data))
        return self.state.finalize_upload(upload_id)["input"]

    def test_checkpoint_upload_works_without_a_session_in_dataset_mode(self):
        record = self._upload_checkpoint("resume.ckpt")
        published = Path(record["path"])
        self.assertTrue(published.is_file())
        self.assertIn("uploaded-checkpoints", str(published))
        # The published path lies under the output directory, so the
        # dataset-mode load validation accepts it as a resume checkpoint.
        request = self.state._dataset_session_request(
            {"paths": {"checkpoint": str(published)},
             "run": {"z_begin": 0, "z_end": 10}})
        self.assertEqual(request["paths"]["checkpoint"], str(published))
        # Checkpoint uploads are not session inputs: nothing ephemeral listed.
        self.assertEqual(self.state.status()["ephemeral_inputs"], [])

    def test_checkpoint_upload_requires_output_root(self):
        bare = ServiceState()
        with self.assertRaises(ApiError) as caught:
            bare.begin_upload({"kind": "checkpoint", "id": "resume.ckpt",
                               "files": [{"name": "resume.ckpt", "size": 1,
                                          "sha256": "0" * 64}]})
        self.assertEqual(caught.exception.status, 409)

    def test_invalid_container_is_rejected(self):
        upload_id = _upload_input(self.state, "checkpoint", "bad.ckpt",
                                  {"bad.ckpt": b"not a torch archive"})
        with self.assertRaisesRegex(ApiError, "Invalid checkpoint"):
            self.state.finalize_upload(upload_id)
        self.assertFalse(any((self.root / "spiral_output" / "uploaded-checkpoints").glob("*"))
                         if (self.root / "spiral_output" / "uploaded-checkpoints").exists()
                         else False)

    def test_identical_checkpoint_is_reused_without_an_upload(self):
        first = self._upload_checkpoint("resume.ckpt")
        data = _zip_checkpoint_bytes()
        begin = self.state.begin_upload({
            "kind": "checkpoint",
            "id": "renamed.ckpt",
            "files": [{
                "name": "renamed.ckpt",
                "size": len(data),
                "sha256": _digest(data),
            }],
        })
        self.assertTrue(begin["deduplicated"])
        self.assertNotIn("upload_id", begin)
        second = begin["input"]
        self.assertEqual(first["path"], second["path"])
        self.assertEqual(second["id"], "renamed.ckpt")
        self.assertTrue(Path(first["path"]).is_file())

    def test_same_name_with_different_content_is_not_reused(self):
        first = self._upload_checkpoint("resume.ckpt", _zip_checkpoint_bytes(b"first"))
        second = self._upload_checkpoint("resume.ckpt", _zip_checkpoint_bytes(b"second"))
        self.assertNotEqual(first["path"], second["path"])
        self.assertTrue(Path(first["path"]).is_file())
        self.assertTrue(Path(second["path"]).is_file())

    def test_checkpoint_deduplication_survives_service_restart(self):
        first = self._upload_checkpoint("resume.ckpt")
        restarted = ServiceState(dataset_root=str(self.root),
                                 dataset_resolution=self.resolution)
        data = _zip_checkpoint_bytes()
        begin = restarted.begin_upload({
            "kind": "checkpoint",
            "id": "after-restart.ckpt",
            "files": [{
                "name": "after-restart.ckpt",
                "size": len(data),
                "sha256": _digest(data),
            }],
        })
        self.assertTrue(begin["deduplicated"])
        self.assertEqual(begin["input"]["path"], first["path"])

    def test_legacy_named_checkpoint_is_found_by_digest(self):
        root = self.root / "spiral_output" / "uploaded-checkpoints"
        root.mkdir()
        data = _zip_checkpoint_bytes()
        legacy = root / "resume-before-v7.ckpt"
        legacy.write_bytes(data)
        begin = self.state.begin_upload({
            "kind": "checkpoint",
            "id": "resume.ckpt",
            "files": [{
                "name": "resume.ckpt",
                "size": len(data),
                "sha256": _digest(data),
            }],
        })
        self.assertTrue(begin["deduplicated"])
        self.assertEqual(Path(begin["input"]["path"]), legacy)

    def test_concurrent_identical_uploads_converge_on_one_file(self):
        data = _zip_checkpoint_bytes()
        request = {
            "kind": "checkpoint",
            "id": "resume.ckpt",
            "files": [{
                "name": "resume.ckpt",
                "size": len(data),
                "sha256": _digest(data),
            }],
        }
        first_id = self.state.begin_upload(request)["upload_id"]
        second_id = self.state.begin_upload(request)["upload_id"]
        for upload_id in (first_id, second_id):
            self.state.receive_upload_file(
                upload_id, "resume.ckpt", io.BytesIO(data), len(data))
        first = self.state.finalize_upload(first_id)["input"]
        second = self.state.finalize_upload(second_id)["input"]
        self.assertEqual(first["path"], second["path"])
        root = self.root / "spiral_output" / "uploaded-checkpoints"
        self.assertEqual([Path(first["path"])], list(root.iterdir()))

    def test_reusing_a_checkpoint_refreshes_retention_recency(self):
        payloads = [str(i).encode() for i in range(
            spiral_service.UPLOADED_CHECKPOINTS_KEPT)]
        published = [
            Path(self._upload_checkpoint(
                f"resume-{i}.ckpt", _zip_checkpoint_bytes(payload))["path"])
            for i, payload in enumerate(payloads)
        ]
        base_time = time.time() - 100
        for age, path in enumerate(published):
            os.utime(path, (base_time + age, base_time + age))

        reused_data = _zip_checkpoint_bytes(payloads[0])
        begin = self.state.begin_upload({
            "kind": "checkpoint",
            "id": "reuse.ckpt",
            "files": [{
                "name": "reuse.ckpt",
                "size": len(reused_data),
                "sha256": _digest(reused_data),
            }],
        })
        self.assertTrue(begin["deduplicated"])
        newest = Path(self._upload_checkpoint(
            "new.ckpt", _zip_checkpoint_bytes(b"new"))["path"])
        self.assertTrue(published[0].exists())
        self.assertFalse(published[1].exists())
        self.assertTrue(newest.exists())

    def test_retention_prunes_old_uploads(self):
        published = [Path(self._upload_checkpoint(
            f"resume-{i}.ckpt", _zip_checkpoint_bytes(str(i).encode()))["path"])
                     for i in range(spiral_service.UPLOADED_CHECKPOINTS_KEPT + 2)]
        for old_age, path in enumerate(published):
            if path.exists():
                # Ensure distinguishable mtimes for deterministic pruning.
                os.utime(path, (time.time() + old_age, time.time() + old_age))
        surviving = [path for path in published if path.exists()]
        self.assertLessEqual(len(surviving), spiral_service.UPLOADED_CHECKPOINTS_KEPT)
        self.assertTrue(published[-1].exists(), "the newest upload must survive")

    def test_checkpoint_uploads_are_exempt_from_ephemeral_quota(self):
        original = spiral_service.EPHEMERAL_QUOTA_BYTES
        spiral_service.EPHEMERAL_QUOTA_BYTES = 1
        try:
            record = self._upload_checkpoint("big.ckpt")
            self.assertTrue(Path(record["path"]).is_file())
        finally:
            spiral_service.EPHEMERAL_QUOTA_BYTES = original


class CommitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "verified_patches").mkdir(parents=True)
        (self.dataset / "fibers").mkdir()
        # The ephemeral folder lives under the output directory, which may be a
        # different filesystem; the copy-then-rename move must still work.
        self.output = self.root / "output"
        self.output.mkdir()
        self.state = ServiceState()
        self.session = _attach_fake_session(self.state, self.output, self.dataset)

    def tearDown(self):
        os.chmod(self.dataset, 0o755)
        self.temporary.cleanup()

    def _finalize(self, kind, input_id, files, role=None):
        upload_id = _upload_input(self.state, kind, input_id, files, role=role)
        return self.state.finalize_upload(upload_id)["input"]

    def test_commit_publishes_patches_fibers_and_merges_role_pcls(self):
        self._finalize("patch", "patch-9", PATCH_FILES)
        self._finalize("fiber", "fiber-9", FIBER_FILES)
        existing = {"vc_pointcollections_json_version": "1",
                    "collections": {"3": {"name": "old", "points": {}}}}
        target = self.dataset / "patch-overlap-pcls.json"
        target.write_text(json.dumps(existing))
        self._finalize("pcl", "pcl-9", PCL_FILES, role="patch_overlap")
        response = self.state.commit_inputs()
        self.assertEqual(sorted(response["committed"]),
                         ["fiber-9", "patch-9", "pcl-9"])
        self.assertTrue((self.dataset / "verified_patches" / "patch-9" / "meta.json").is_file())
        self.assertTrue((self.dataset / "fibers" / "fiber-9.json").is_file())
        merged = json.loads(target.read_text())
        self.assertEqual(len(merged["collections"]), 2)
        backups = list(self.dataset.glob("patch-overlap-pcls.json.*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text()), existing)
        # Still-pending inputs stay queued for the next run after a commit.
        inputs = self.state.status()["ephemeral_inputs"]
        self.assertEqual({record["id"] for record in inputs},
                         {"patch-9", "fiber-9", "pcl-9"})
        self.assertTrue(all(record["committed"] and record["state"] == "pending"
                            for record in inputs))

    def test_concurrent_service_commits_serialize_pcl_merges(self):
        output_b = self.root / "output-b"
        output_b.mkdir()
        state_b = ServiceState()
        _attach_fake_session(state_b, output_b, self.dataset)
        upload_a = _upload_input(
            self.state, "pcl", "pcl-a", PCL_FILES, role="patch_overlap")
        upload_b = _upload_input(
            state_b, "pcl", "pcl-b", PCL_FILES, role="patch_overlap")
        self.state.finalize_upload(upload_a)
        state_b.finalize_upload(upload_b)
        target = self.dataset / "patch-overlap-pcls.json"
        target.write_text(json.dumps({
            "vc_pointcollections_json_version": "1", "collections": {},
        }))

        active = 0
        max_active = 0
        activity_lock = threading.Lock()
        original_merge = spiral_service._merge_pcl_documents

        def slow_merge(existing, incoming):
            nonlocal active, max_active
            with activity_lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.1)
                return original_merge(existing, incoming)
            finally:
                with activity_lock:
                    active -= 1

        errors = []

        def commit(state):
            try:
                state.commit_inputs()
            except BaseException as exc:
                errors.append(exc)

        with mock.patch.object(
                spiral_service, "_merge_pcl_documents", side_effect=slow_merge):
            threads = [
                threading.Thread(target=commit, args=(self.state,)),
                threading.Thread(target=commit, args=(state_b,)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
        self.assertFalse(errors)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(max_active, 1)
        merged = json.loads(target.read_text())
        self.assertEqual(len(merged["collections"]), 2)

    def test_independent_processes_preserve_both_pcl_commits(self):
        target = self.dataset / "patch-overlap-pcls.json"
        target.write_text(json.dumps({
            "vc_pointcollections_json_version": "1", "collections": {},
        }))
        context = multiprocessing.get_context("spawn")
        ready = context.Queue()
        start = context.Event()
        result = context.Queue()
        processes = []
        for name in ("process-a", "process-b"):
            output = self.root / name
            output.mkdir()
            processes.append(context.Process(
                target=_commit_pcl_process,
                args=(str(self.dataset), str(output), name, ready, start, result)))
        for process in processes:
            process.start()
        self.assertEqual({ready.get(timeout=20), ready.get(timeout=20)},
                         {"process-a", "process-b"})
        start.set()
        outcomes = dict(result.get(timeout=30) for _ in processes)
        for process in processes:
            process.join(30)
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(outcomes, {"process-a": "ok", "process-b": "ok"})
        merged = json.loads(target.read_text())
        self.assertEqual(len(merged["collections"]), 2)

    def test_drawn_control_points_commit_preserves_line_and_point_order(self):
        existing = {
            "vc_pointcollections_json_version": "1",
            "collections": {"4": {"name": "existing", "points": {}}},
        }
        target = self.dataset / "drawn_control_points.json"
        target.write_text(json.dumps(existing))
        incoming = {"drawn.json": json.dumps({
            "vc_pointcollections_json_version": "1",
            "collections": {
                "0": {"name": "first", "points": {
                    "0": {"p": [0, 0, 0]}, "1": {"p": [30, 0, 0]}}},
                "1": {"name": "second", "points": {
                    "0": {"p": [0, 1, 0]}, "1": {"p": [30, 1, 0]}}},
            },
        }).encode()}
        self._finalize("pcl", "drawn-1", incoming, role="drawn_control_points")
        self.state.commit_inputs()
        merged = json.loads(target.read_text())
        self.assertEqual([collection["name"] for collection in merged["collections"].values()],
                         ["existing", "first", "second"])
        self.assertEqual(list(merged["collections"]["5"]["points"]), ["0", "1"])
        self.assertEqual(len(list(self.dataset.glob(
            "drawn_control_points.json.*.bak"))), 1)

    def test_same_winding_commit_preserves_collection_and_point_order(self):
        existing = {
            "vc_pointcollections_json_version": "1",
            "collections": {"2": {"name": "existing", "points": {}}},
        }
        target = self.dataset / "same_windings.json"
        target.write_text(json.dumps(existing))
        incoming = {"same.json": json.dumps({
            "vc_pointcollections_json_version": "1",
            "collections": {
                "0": {"name": "same_winding_0001", "points": {
                    "0": {"p": [1, 2, 3]}, "1": {"p": [4, 5, 6]}}},
                "1": {"name": "same_winding_0002", "points": {
                    "0": {"p": [7, 8, 9]}, "1": {"p": [10, 11, 12]}}},
            },
        }).encode()}
        self._finalize("pcl", "same-1", incoming, role="same_winding")
        self.state.commit_inputs()
        merged = json.loads(target.read_text())
        self.assertEqual([collection["name"] for collection in merged["collections"].values()],
                         ["existing", "same_winding_0001", "same_winding_0002"])
        self.assertEqual(list(merged["collections"]["3"]["points"]), ["0", "1"])
        self.assertEqual(len(list(self.dataset.glob(
            "same_windings.json.*.bak"))), 1)

    def test_commit_keeps_pending_inputs_queued_and_incorporation_retires_them(self):
        record = self._finalize("patch", "patch-9", PATCH_FILES)
        staged = Path(record["path"])
        self.state.commit_inputs()
        # The staged copy remains the incorporation source for the next run.
        self.assertTrue(staged.exists())
        with self.assertRaisesRegex(ApiError, "already committed"):
            self.state.commit_inputs()
        _planned_run(self.state, {"iterations": 3})
        _, pending, mark, _, _ = self.session.run_calls[-1]
        self.assertEqual([entry["id"] for entry in pending], ["patch-9"])
        # Once incorporated, a committed record is done and leaves the list.
        mark(pending)
        self.assertEqual(self.state.status()["ephemeral_inputs"], [])

    def test_remove_pending_input_deletes_the_staged_copy(self):
        record = self._finalize("fiber", "fiber-9", FIBER_FILES)
        staged = Path(record["path"])
        response = self.state.remove_input({"kind": "fiber", "id": "fiber-9"})
        self.assertEqual(response["removed"], "fiber-9")
        self.assertEqual(self.state.status()["ephemeral_inputs"], [])
        self.assertFalse(staged.exists())
        _planned_run(self.state, {"iterations": 1})
        self.assertEqual(self.session.run_calls[-1][1], [])

    def test_remove_incorporated_input_is_rejected(self):
        self._finalize("patch", "patch-9", PATCH_FILES)
        _planned_run(self.state, {"iterations": 1})
        _, pending, mark, _, _ = self.session.run_calls[-1]
        mark(pending)
        with self.assertRaises(ApiError) as caught:
            self.state.remove_input({"kind": "patch", "id": "patch-9"})
        self.assertEqual(caught.exception.status, 409)

    def test_remove_committed_pending_input_keeps_the_dataset_copy(self):
        self._finalize("patch", "patch-9", PATCH_FILES)
        self.state.commit_inputs()
        self.state.remove_input({"kind": "patch", "id": "patch-9"})
        self.assertEqual(self.state.status()["ephemeral_inputs"], [])
        self.assertTrue((self.dataset / "verified_patches" / "patch-9" / "meta.json").is_file())

    def test_patch_identifier_collision_is_rejected_without_overwrite(self):
        existing = self.dataset / "verified_patches" / "patch-1"
        existing.mkdir()
        (existing / "meta.json").write_text("original")
        self._finalize("patch", "patch-1", PATCH_FILES)
        with self.assertRaises(ApiError) as caught:
            self.state.commit_inputs()
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual((existing / "meta.json").read_text(), "original")
        # The ephemeral input is untouched and still usable.
        self.assertEqual(self.state.status()["ephemeral_inputs"][0]["state"], "pending")

    def test_commit_on_read_only_dataset_is_reported_unavailable(self):
        self._finalize("patch", "patch-2", PATCH_FILES)
        os.chmod(self.dataset, 0o555)
        status = self.state.status()
        self.assertFalse(status["commit_available"])
        self.assertIn("read-only", status["commit_unavailable_reason"])
        with self.assertRaisesRegex(ApiError, "read-only"):
            self.state.commit_inputs()


class MappedPreviewArtifactTests(unittest.TestCase):
    def test_lasagna_output_scale_must_match_requested_step(self):
        self.assertEqual(
            _validate_tifxyz_output_step(
                {"scale": [0.05, 0.05]}, 20.0),
            [0.05, 0.05])
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _validate_tifxyz_output_step(
                {"scale": [0.04, 0.04]}, 20.0)

    def test_failed_flatten_keeps_previous_preview_and_discards_raw_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = root / "previous"
            current = root / "current"
            previous.mkdir()
            current.mkdir()
            previous_manifest = previous / "manifest.json"
            current_manifest = current / "manifest.json"
            previous_manifest.write_text("{}")
            current_manifest.write_text("{}")
            state = ServiceState()
            state.session_id = "session"
            state._previous_raw_preview_manifest = str(previous_manifest)
            state._preview_artifact = {"id": "previous-preview"}

            with mock.patch.object(
                    state, "_publish_flattened_preview",
                    side_effect=RuntimeError("flatten failed")):
                state._maybe_register_artifacts({
                    "preview_generation": 1,
                    "preview_manifest_path": str(current_manifest),
                })

            self.assertEqual(
                state._preview_artifact, {"id": "previous-preview"})
            self.assertTrue(previous.exists())
            self.assertFalse(current.exists())
            self.assertIn("flatten failed", state._preview_publish_error)

    def test_winding_membership_uses_flatten_correspondence(self):
        manifest = {
            "winding_ids": [7, 9],
            "winding_column_ranges": [[0, 2], [2, 4]],
        }
        source_yx = np.asarray([
            [[0.0, 3.0], [1.0, 0.0], [1.0, 2.0]],
            [[9.0, 9.0], [0.0, 1.0], [0.0, 2.0]],
        ], dtype=np.float32)
        output_valid = np.asarray([
            [True, True, True],
            [True, False, True],
        ])

        winding_ids, bounds = _mapped_winding_ids(
            manifest, (2, 4), source_yx, output_valid)

        np.testing.assert_array_equal(winding_ids, np.asarray([
            [9, 7, 9],
            [-1, -1, 9],
        ], dtype=np.int32))
        self.assertEqual(bounds, [
            {
                "winding": 7, "row_begin": 0, "row_end": 1,
                "column_begin": 1, "column_end": 2,
            },
            {
                "winding": 9, "row_begin": 0, "row_end": 2,
                "column_begin": 0, "column_end": 3,
            },
        ])

    def test_loss_overlay_is_bilinearly_warped_with_alpha(self):
        source = np.zeros((2, 2, 4), dtype=np.uint8)
        source[0, 0] = [200, 0, 0, 255]
        source[0, 1] = [0, 0, 200, 255]
        source_yx = np.asarray([[
            [0.0, 0.0], [0.0, 0.5], [0.0, 1.0],
        ]], dtype=np.float32)

        mapped = _sample_rgba_through_map(
            source, source_yx, np.asarray([[True, True, False]]))

        np.testing.assert_array_equal(mapped[0, 0], [200, 0, 0, 255])
        np.testing.assert_allclose(mapped[0, 1], [100, 0, 100, 255], atol=1)
        np.testing.assert_array_equal(mapped[0, 2], [0, 0, 0, 0])

    @staticmethod
    def _write_surface(path, xyz):
        path.mkdir()
        for axis, values in zip("xyz", np.moveaxis(xyz, -1, 0)):
            Image.fromarray(values.astype(np.float32)).save(
                path / f"{axis}.tif")

    def test_run_diff_matches_windings_before_flatten_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = np.zeros((2, 4, 3), dtype=np.float32)
            current = previous.copy()
            current[:, 2:, 2] = 3.0
            self._write_surface(root / "previous", previous)
            self._write_surface(root / "current", current)
            common = {
                "winding_ids": [1, 2],
                "winding_column_ranges": [[0, 2], [2, 4]],
            }

            rgba, changed = _raw_run_diff_rgba(
                {**common, "surface_path": str(root / "previous")},
                {**common, "surface_path": str(root / "current")})

            self.assertEqual(changed, 4)
            self.assertTrue(np.all(rgba[:, :2, 3] == 0))
            self.assertTrue(np.all(rgba[:, 2:, 3] > 0))


class LasagnaSurfaceCleanupTests(unittest.TestCase):
    @staticmethod
    def _write_surface(path, valid):
        path.mkdir()
        rows, columns = np.indices(valid.shape)
        xyz = [
            columns.astype(np.float32),
            rows.astype(np.float32),
            np.full(valid.shape, 10.0, dtype=np.float32),
        ]
        for coordinate in xyz:
            coordinate[~valid] = -1.0
        for name, coordinate in zip(("x.tif", "y.tif", "z.tif"), xyz):
            Image.fromarray(coordinate).save(path / name)
        (path / "meta.json").write_text(json.dumps({
            "format": "tifxyz",
            "type": "seg",
            "uuid": "preview",
            "scale": [1.0, 1.0],
            "bbox": [[0.0, 0.0, 0.0], [99.0, 99.0, 99.0]],
            "area_vx2": 100.0,
            "area_cm2": 2.0,
            "winding_column_ranges": [[0, valid.shape[1]]],
        }))

    def test_erodes_and_keeps_only_largest_component_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "cleaned"
            valid = np.zeros((16, 28), dtype=bool)
            valid[1:12, 1:12] = True
            valid[4:11, 18:25] = True
            self._write_surface(source, valid)
            original_files = {
                name: (source / name).read_bytes()
                for name in ("meta.json", "x.tif", "y.tif", "z.tif")
            }

            result = _prepare_cleaned_lasagna_surface(source, destination)

            self.assertEqual(result, destination)
            for name, contents in original_files.items():
                self.assertEqual((source / name).read_bytes(), contents)
            cleaned_coordinates = []
            for axis in "xyz":
                with Image.open(destination / f"{axis}.tif") as image:
                    cleaned_coordinates.append(np.asarray(image).copy())
            cleaned_xyz = np.stack(cleaned_coordinates, axis=-1)
            cleaned_valid = np.any(cleaned_xyz != -1.0, axis=-1)
            expected = np.zeros_like(valid)
            expected[4:9, 4:9] = True
            np.testing.assert_array_equal(cleaned_valid, expected)
            self.assertTrue(np.all(cleaned_xyz[~expected] == -1.0))

            metadata = json.loads(
                (destination / "meta.json").read_text())
            self.assertEqual(
                metadata["bbox"],
                [[4.0, 4.0, 10.0], [8.0, 8.0, 10.0]])
            self.assertEqual(metadata["area_vx2"], 16.0)
            self.assertAlmostEqual(metadata["area_cm2"], 0.32)
            self.assertEqual(metadata["winding_column_ranges"], [[0, 28]])
            self.assertEqual(metadata["lasagna_input_cleanup"], {
                "erosion_cells": 3,
                "component_connectivity": 4,
                "components_after_erosion": 2,
            })

    def test_fails_if_erosion_removes_every_valid_vertex(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            valid = np.ones((6, 6), dtype=bool)
            self._write_surface(source, valid)

            with self.assertRaisesRegex(
                    RuntimeError, "removed every valid TIFXYZ vertex"):
                _prepare_cleaned_lasagna_surface(
                    source, root / "cleaned")
            self.assertFalse((root / "cleaned").exists())


class ServiceProcessTests(unittest.TestCase):
    """End-to-end launch of the real service process (no torch import)."""

    def _launch(self, arguments, temporary):
        script = Path(__file__).resolve().parents[1] / "spiral_service.py"
        return subprocess.Popen(
            [sys.executable, str(script)] + arguments,
            cwd=str(script.parent),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def _read_until_ready(self, process, deadline=30.0):
        lines = []
        end = time.time() + deadline
        while time.time() < end:
            line = process.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip())
            if line.startswith("SPIRAL_SERVICE_READY"):
                return lines
        raise AssertionError(f"service never became ready: {lines}")

    def test_ready_line_is_version_agnostic_and_health_negotiates(self):
        with tempfile.TemporaryDirectory() as temporary:
            key_file = Path(temporary) / "key"
            process = self._launch(["--port", "0", "--api-key-file", str(key_file)],
                                   temporary)
            try:
                lines = self._read_until_ready(process)
                ready = [line for line in lines if line.startswith("SPIRAL_SERVICE_READY")][0]
                self.assertNotIn("api_version", ready)
                port = int(ready.split("port=")[1].split()[0])
                key = key_file.read_text().strip()
                # The API key must not appear in the ready line.
                self.assertNotIn(key, ready)
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/health",
                    headers={"Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    health = json.loads(response.read())
                self.assertEqual(health["api_version"], API_VERSION)
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/logs?after=0",
                    headers={"Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    logs = json.loads(response.read())
                self.assertIn(
                    ready, [entry["text"] for entry in logs["entries"]])
            finally:
                process.terminate()
                process.wait(10)
                process.stdout.close()

    def test_restart_reexecs_in_place_and_reuses_the_fixed_port(self):
        with tempfile.TemporaryDirectory() as temporary:
            key_file = Path(temporary) / "key"
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            process = self._launch(
                ["--port", str(port), "--api-key-file", str(key_file)],
                temporary)
            try:
                self._read_until_ready(process)
                original_pid = process.pid
                key = key_file.read_text().strip()
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/service/restart",
                    method="POST",
                    data=json.dumps({"command_id": "restart-integration"}).encode(),
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    })
                with urllib.request.urlopen(request, timeout=10) as response:
                    acknowledgement = json.loads(response.read())
                self.assertTrue(acknowledgement["restarting"])

                self._read_until_ready(process)
                self.assertEqual(process.pid, original_pid)
                health_request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/health",
                    headers={"Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(health_request, timeout=10) as response:
                    self.assertTrue(json.loads(response.read())["ready"])
            finally:
                process.terminate()
                process.wait(10)
                process.stdout.close()

    def test_selected_gpus_are_reported_by_health(self):
        with tempfile.TemporaryDirectory() as temporary:
            key_file = Path(temporary) / "key"
            process = self._launch([
                "--port", "0", "--api-key-file", str(key_file),
                "--gpus", "3,1",
            ], temporary)
            try:
                lines = self._read_until_ready(process)
                ready = next(line for line in lines
                             if line.startswith("SPIRAL_SERVICE_READY"))
                port = int(ready.split("port=")[1].split()[0])
                key = key_file.read_text().strip()
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/health",
                    headers={"Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    health = json.loads(response.read())
                self.assertEqual(health["gpus"], [3, 1])
                self.assertIn("Spiral CUDA devices: 3,1", lines)
            finally:
                process.terminate()
                process.wait(10)

    def test_explicit_port_can_be_rebound_immediately(self):
        with tempfile.TemporaryDirectory() as temporary:
            key_file = Path(temporary) / "key"
            process = self._launch(["--port", "0", "--api-key-file", str(key_file)],
                                   temporary)
            try:
                ready = [line for line in self._read_until_ready(process)
                         if line.startswith("SPIRAL_SERVICE_READY")][0]
                port = int(ready.split("port=")[1].split()[0])
            finally:
                process.kill()
                process.wait(10)
            # Restart on the same, now-explicit port straight away.
            process = self._launch(["--port", str(port),
                                    "--api-key-file", str(key_file)], temporary)
            try:
                self._read_until_ready(process)
            finally:
                process.terminate()
                process.wait(10)

    def test_dataset_service_refuses_to_start_when_entries_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            key_file = Path(temporary) / "key"
            empty = Path(temporary) / "empty-dataset"
            empty.mkdir()
            process = self._launch(["--port", "0", "--api-key-file", str(key_file),
                                    "--dataset", str(empty)], temporary)
            output, _ = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 2)
            self.assertIn("missing required", output)

    def test_named_dataset_service_has_exclusive_restartable_ownership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "umbilicus.json").write_text("{}")
            (dataset / "verified_patches").mkdir()
            key_file = root / "key"
            arguments = [
                "--port", "0", "--api-key-file", str(key_file),
                "--dataset", str(dataset), "--session-name", "alice",
            ]
            first = self._launch(arguments, temporary)
            try:
                lines = self._read_until_ready(first)
                self.assertIn("Spiral session name: alice", lines)
                ready = next(line for line in lines
                             if line.startswith("SPIRAL_SERVICE_READY"))
                port = int(ready.split("port=")[1].split()[0])
                key = key_file.read_text().strip()
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/health",
                    headers={"Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    health = json.loads(response.read())
                self.assertEqual(health["session_name"], "alice")

                duplicate = self._launch(arguments, temporary)
                output, _ = duplicate.communicate(timeout=30)
                self.assertEqual(duplicate.returncode, 2)
                self.assertIn("already owned", output)
            finally:
                first.terminate()
                first.wait(10)
                first.stdout.close()

            restarted = self._launch(arguments, temporary)
            try:
                self._read_until_ready(restarted)
            finally:
                restarted.terminate()
                restarted.wait(10)
                restarted.stdout.close()

    def test_non_loopback_bind_requires_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            key_file = Path(temporary) / "key"
            process = self._launch(["--bind", "0.0.0.0", "--port", "0",
                                    "--api-key-file", str(key_file)], temporary)
            output, _ = process.communicate(timeout=30)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("--dataset", output)


if __name__ == "__main__":
    unittest.main()
