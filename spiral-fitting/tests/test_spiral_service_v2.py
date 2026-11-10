"""Tests for the Spiral service protocol.

Covers bearer authentication, API-key auto-generation, launch-time host
path bindings, the artifact registry and its HTTP endpoints, session input
uploads with immutable workspace staging, and checkpoint transfers. The resident fitter is
faked; these tests exercise the service plumbing only.
"""

import argparse
import copy
import dataclasses
from concurrent.futures import ThreadPoolExecutor
import io
import hashlib
import json
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
import unittest.mock
import urllib.error
import urllib.request
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spiral_service
from spiral_service import (ApiError, ArtifactRegistry,
                            ExclusiveFileLock,
                            FileLockUnavailable, ServiceLogBuffer, ServiceState,
                            SpiralServer, _mapped_winding_ids,
                            _load_flatten_correspondence,
                            _prepare_cleaned_lasagna_surface,
                            _raw_run_diff_rgba, _sample_rgba_through_map,
                            _validate_tifxyz_output_step,
                            load_or_create_api_key, parse_gpu_ids,
                            parse_session_name)
from service_uploads import UPLOADED_CHECKPOINTS_KEPT
from lasagna_publish import (PreviewPublication, PublishedPreview,
                             _flatten_request_body)
from fit_session import (API_VERSION, AUTOSAVE_CHECKPOINT_NAME, AUTOSAVE_METADATA_NAME,
                         AUTOSAVE_METADATA_SCHEMA, SCROLL_SPEC_OWNED_RUN_KEYS,
                         AutosaveError, SessionState, SpiralInputPaths,
                         SpiralProject, select_startup_autosave,
                         validate_autosave, write_autosave_metadata)
from config import Config
from service_project import UMBILICUS, VOLUME, project, with_project


class FakeSession:
    def __init__(self):
        self.state = SessionState.Idle
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
        }
        # The resolved configuration the fit is running, as a real session
        # publishes it once it has one; a checkpoint refusal is analysed
        # against this.
        self.applied_config = None
        # The object-store inputs' summary, as a real session reports it.
        self.input_summary = None
        self.default_advanced_config = {
            "optimizer_learning_rate": 3e-5,
            "sample_count_patches_per_step": 360,
            "loss_weight_patch_radius": 8.0,
            "track_crossing_precompute_max": 8,
        }
        self.saved = []
        self.autosave_calls = []
        self.previews = 0
        self.preview_diagnostics = []
        self.preview_gate = None
        self.preview_failure = None
        self.loaded = []
        self.load_refusal = None
        self.closed = False
        self.path_change_calls = []
        self.model_rebuilds = []
        self.progress = None
        self.preview_schedules = []
        self.dt_loss_schedules = []

    def status(self):
        applied = ({"applied_config": dict(self.applied_config)}
                   if self.applied_config is not None else {})
        if self.input_summary is not None:
            applied["input_summary"] = dict(self.input_summary)
        return {
            **applied,
            "state": self.state, "phase": str(self.state),
            "current_iteration": 5,
            "target_iteration": 5, "latest_metrics": {}, "warnings": [],
            "error": None, "preview_manifest_path": None, "preview_generation": 0,
            "supports_input_incorporation": True,
            "run_config": dict(self.run_config),
            "run_config_limits": {"track_max_track_crossing_per_step": 8},
            "default_advanced_config": dict(self.default_advanced_config),
            "progress": self.progress,
        }

    def run(self, count, run_config=None, path_changes=None,
            autosave_on_pause=True, preview_schedule=None, dt_loss_schedule=None):
        self.run_calls.append((count, dict(run_config or {})))
        self.path_change_calls.append(dict(path_changes or {}))
        self.autosave_calls.append(autosave_on_pause)
        self.preview_schedules.append(copy.deepcopy(preview_schedule))
        self.dt_loss_schedules.append(copy.deepcopy(dt_loss_schedule))
        self.run_config.update(run_config or {})
        return 5 + count

    def save_checkpoint(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"PK\x03\x04checkpoint")
        self.saved.append(path)
        return path

    def load_checkpoint(self, path, timeout=600.0):
        if self.load_refusal is not None:
            raise RuntimeError(self.load_refusal)
        self.loaded.append(path)
        return {"completed_iterations": 4200, "config_revision": 1,
                "path": path}

    def export_preview(self, timeout=600.0, diagnostics=False):
        # The real export blocks its caller for minutes; these let a test
        # hold it open, or fail it, the way a real one can.
        if self.preview_gate is not None:
            self.preview_gate.wait(5)
        if self.preview_failure is not None:
            raise RuntimeError(self.preview_failure)
        self.previews += 1
        self.preview_diagnostics.append(bool(diagnostics))
        return {"preview_generation": self.previews,
                "preview_manifest_path": f"/preview/{self.previews}",
                "preview_diagnostics": bool(diagnostics)}

    def rebuild_model(self, paths, run, timeout=1800.0):
        self.model_rebuilds.append((paths, run))
        return {"config_revision": 1, "current_iteration": 0}

    def close(self):
        self.closed = True


# The test project carries no Lasagna dataset, so the launch
# defaults these fixtures start from zero-weight the dense losses that would
# require them.
_NO_DENSE_LOSSES = {
    "dense_spacing_mode": "grad_mag",
    "loss_weight_dense_spacing": 0,
    "loss_weight_dense_normals": 0,
    "loss_weight_shell_outer": 0,
}


def _await_build(state, timeout=10.0):
    """Wait for the asynchronous session build to settle, and report how."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state.session is not None or state._session_state == SessionState.Error:
            return state.status()
        time.sleep(0.01)
    raise AssertionError("the session build did not settle")


def _attach_fake_session(state, output_directory, volume_uuid=""):
    state.session = FakeSession()
    state.session_generation += 1
    state.session_id = f"spiral-test-{state.session_generation}"
    state.session_paths = SpiralInputPaths.from_mapping({
        "volume_uuid": str(volume_uuid),
        "output_directory": str(output_directory),
    })
    state.session_request = {
        "paths": state.session_paths.manifest(),
        "run": {"config": Config().as_dict()},
    }
    state._session_project = SpiralProject.from_mapping(project())
    state.session_revision += 1
    return state.session


_PROJECT = SpiralProject.from_mapping(project())


def _publish_request(state, request):
    """Make ``request`` the live session request, as a build publishes it:
    prepared, then resolved for the project's volume."""
    paths, run, preview, scroll = state._prepare_session_request(with_project(request))
    paths, spec = state._resolve_session_scroll(
        scroll, VOLUME, paths, state._effective_config(paths, run))
    state.session_paths = paths
    state.session_request = {
        "paths": paths.manifest(), "run": run.manifest(),
        "preview": preview.manifest(), "project": scroll.project.record(VOLUME),
        "scroll": spec.manifest()}
    state._session_project = scroll.project


class ProgressStatusTests(unittest.TestCase):
    def test_status_propagates_structured_fit_progress(self):
        state = spiral_service.ServiceState()
        session = _attach_fake_session(state, "/tmp/spiral-progress-test")
        session.state = "Loading"
        session.progress = {
            "operation": "loading",
            "stage_name": "Loading tracks",
            "detail": "12,000 tracks retained",
            "step": 3,
            "total_steps": 10,
            "unit": "DB keys",
            "elapsed_seconds": 4.5,
            "eta_seconds": 10.5,
        }

        status = state.status()

        # The snapshot carries the raw progress facts; eta_seconds is a
        # presentation value the client derives, never served.
        expected = {key: value for key, value in session.progress.items()
                    if key != "eta_seconds"}
        self.assertEqual(status["progress"], expected)
        self.assertEqual(status["progress"]["stage_name"], "Loading tracks")

    def test_status_never_synthesizes_eta(self):
        state = spiral_service.ServiceState()
        session = _attach_fake_session(state, "/tmp/spiral-progress-test")
        session.progress = {
            "operation": "optimizing", "stage_name": "Optimizing",
            "detail": None, "step": 5, "total_steps": 10,
            "unit": "iterations", "elapsed_seconds": 30.0,
            "eta_seconds": 30.0,
        }
        self.assertNotIn("eta_seconds", state.status()["progress"])

        # The preview-publication progress block is also served raw.
        state._preview.claim("spiral-test-1", 2)
        state._preview.record_progress(2, {
            "state": "running",
            "stage_name": "Flattening preview surface",
            "step": 5, "total_steps": 10,
        })
        state._preview.stage_started = None
        progress = state.status()["progress"]
        self.assertEqual(progress["operation"], "publishing_preview")
        self.assertNotIn("eta_seconds", progress)

    def test_empty_status_has_explicit_null_progress(self):
        self.assertIsNone(spiral_service.ServiceState().status()["progress"])


def _planned_run(state, request):
    request = dict(request)
    configuration = dict(Config.catalog()["defaults"])
    configuration.update(request.pop("run_config", {}))
    return state.run({
        "configuration": configuration,
        "iterations": request.pop("iterations"),
        "dt_loss_schedule": request.pop("dt_loss_schedule", {
            "enabled": False, "last_fraction": 0.25}),
        "expected_session_revision": state.session_revision,
        **request,
    })


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _upload_input(state, kind, input_id, files, role=None, base_revision=None,
                  operation=None, target_collection_id=None,
                  base_source_revision=None):
    request = {
        "upload_id": spiral_service.secrets.token_hex(16),
        "kind": kind, "id": input_id,
        "files": [{"name": name, "size": len(data), "sha256": _digest(data)}
                  for name, data in files.items()],
    }
    if role:
        request["role"] = role
    if base_revision is not None:
        request["base_revision"] = base_revision
    if operation is not None:
        request.update({
            "operation": operation,
            "target_collection_id": target_collection_id,
            "base_source_revision": base_source_revision,
        })
    upload_id = state.begin_upload(request)["upload_id"]
    for name, data in files.items():
        state.receive_upload_file(upload_id, name, io.BytesIO(data), len(data))
    return upload_id


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

    def test_bindings_are_the_launch_host_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output" / "alice"
            cache = Path(temporary) / "cache"
            bindings = spiral_service.bind_service_inputs(output, cache)
            self.assertEqual(bindings.advertisement(), {
                "output_directory": str(output), "cache_directory": str(cache),
                "winding_inference": {
                    "bound": False, "path": None, "available": False,
                    "error": "the service was launched without --winding-inference"}})

    def test_default_user_cache_dir_honours_xdg_and_is_stable(self):
        from fit_session import default_user_cache_dir
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": temporary}):
                self.assertEqual(default_user_cache_dir(),
                                 str(Path(temporary).resolve() / "vc3d" / "spiral"))
            with mock.patch.dict(os.environ, clear=True):
                os.environ["HOME"] = temporary
                self.assertEqual(
                    default_user_cache_dir(),
                    str(Path(temporary).resolve() / ".cache" / "vc3d" / "spiral"))

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
        # Process identity is process_id; there is no service_generation
        # counter (it was the constant 1 and nothing read it).
        self.assertIn("process_id", health)
        self.assertNotIn("service_generation", health)
        self.assertNotIn("command_generation", health)


class RouteTableTests(HttpServiceFixture):
    def test_routes_resolve_from_the_declarative_table(self):
        route, args = spiral_service.resolve_route("GET", "/session/status")
        self.assertEqual((route.operation, args), ("session_status", ()))
        self.assertEqual(route.idempotency, spiral_service.Idempotency.NONE)

        route, args = spiral_service.resolve_route("POST", "/session/stop")
        self.assertEqual((route.operation, args), ("session_stop", ()))
        self.assertEqual(route.idempotency,
                         spiral_service.Idempotency.COMMAND_ID)
        self.assertTrue(route.reads_body)

        # Logical mutations: command-ID idempotent, so a retried load or
        # preview export replays instead of acting twice.
        for path, operation in (("/session/load-checkpoint", "load_checkpoint"),
                                ("/session/export-preview", "export_preview")):
            route, args = spiral_service.resolve_route("POST", path)
            self.assertEqual((route.operation, args), (operation, ()))
            self.assertEqual(route.idempotency,
                             spiral_service.Idempotency.COMMAND_ID)
            self.assertTrue(route.reads_body)

        upload_id = "a" * 32
        route, args = spiral_service.resolve_route(
            "PUT", f"/session/inputs/{upload_id}/files/meta.json")
        self.assertEqual((route.operation, args),
                         ("upload_file", (upload_id, "meta.json")))
        self.assertEqual(route.idempotency,
                         spiral_service.Idempotency.CONTENT)

        route, _ = spiral_service.resolve_route(
            "POST", f"/session/inputs/{upload_id}/finalize")
        self.assertEqual(route.idempotency,
                         spiral_service.Idempotency.UPLOAD_ID)

        # A path is only reachable through the method that declares it.
        self.assertEqual(spiral_service.resolve_route("GET", "/session/stop"),
                         (None, ()))
        self.assertEqual(spiral_service.resolve_route("GET", "/nope"),
                         (None, ()))
        for route in spiral_service.ROUTES:
            self.assertIn(route.idempotency,
                          {spiral_service.Idempotency.NONE,
                           spiral_service.Idempotency.COMMAND_ID,
                           spiral_service.Idempotency.CONTENT,
                           spiral_service.Idempotency.UPLOAD_ID},
                          route.operation)

    def test_dispatch_serves_read_and_mutating_routes_unchanged(self):
        status, payload, _ = self.request("GET", "/session/status")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["api_version"], API_VERSION)

        # A command-ID route still refuses an unstamped request, and reaches
        # the operation (whose session is not initialized) when stamped.
        status, payload, _ = self.request("POST", "/session/stop", body={})
        self.assertEqual(status, 400)
        self.assertIn("command_id", json.loads(payload)["error"])
        status, payload, _ = self.request("POST", "/session/stop",
                                          body={"command_id": "stop-1"})
        self.assertEqual(status, 409)
        self.assertIn("not been initialized", json.loads(payload)["error"])

        status, _, _ = self.request("GET", "/nonexistent")
        self.assertEqual(status, 404)

    def test_session_deletion_is_not_part_of_the_surface(self):
        # There is no verb that removes a session once initialized.
        self.assertEqual(spiral_service.resolve_route("DELETE", "/session"),
                         (None, ()))
        self.assertIsNone(
            spiral_service.resolve_route("POST", "/session/load")[0])
        status, _, _ = self.request("DELETE", "/session",
                                    body={"command_id": "delete-1"})
        self.assertEqual(status, 404)
        route, _ = spiral_service.resolve_route("POST", "/session/rebuild")
        self.assertEqual(route.operation, "session_rebuild")
        self.assertEqual(route.idempotency,
                         spiral_service.Idempotency.COMMAND_ID)
        route, _ = spiral_service.resolve_route("POST", "/session/initialize")
        self.assertEqual(route.operation, "session_initialize")
        self.assertEqual(route.idempotency,
                         spiral_service.Idempotency.COMMAND_ID)

    def test_malformed_post_body_is_reported_before_the_route_is_known(self):
        request = urllib.request.Request(self.base + "/nonexistent",
                                         method="POST")
        request.add_header("Authorization", "Bearer secret-key")
        request.add_header("Content-Type", "application/json")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, data=b"{not json", timeout=10)
        self.assertEqual(caught.exception.code, 400)


class CounterSurfaceTests(unittest.TestCase):
    """What each surviving counter is for, and what no longer exists."""

    def test_only_the_three_load_bearing_counters_are_published(self):
        state = ServiceState()
        base = state.status()
        # Session identity, configuration/input revision, status ordering.
        for key in ("session_generation", "session_revision", "generation"):
            self.assertIn(key, base)
        # Removed: nothing read them. Process identity is process_id.
        for key in ("service_generation", "command_generation",
                    "session_replacement_in_progress",
                    "replacement_old_session_released"):
            self.assertNotIn(key, base)
        self.assertIn("process_id", state.health())

    def test_command_replay_no_longer_stamps_a_counter(self):
        state = ServiceState()
        calls = []

        def operation():
            calls.append(1)
            return {"ok": True}

        first = state.replay_command("session_stop", "cmd-1", operation)
        second = state.replay_command("session_stop", "cmd-1", operation)
        # Deduplication is keyed by (operation, command ID); the counter it
        # used to publish alongside the response was never a key.
        self.assertEqual(len(calls), 1)
        self.assertEqual(first, second)
        self.assertNotIn("command_generation", first)

    def test_preview_publication_state_lives_in_one_record(self):
        state = ServiceState()
        for legacy in ("_registered_preview_generation",
                       "_processed_preview_generation",
                       "_publishing_preview_generation",
                       "_preview_publish", "_preview_publish_error",
                       "_preview_progress_started", "_preview_process",
                       "_preview_artifact", "_previous_raw_preview_manifest"):
            self.assertFalse(hasattr(state, legacy), legacy)

        publication = state._preview
        self.assertTrue(publication.claim("session-a", 4))
        # One generation is published once: neither a replay of the same
        # generation nor an older one claims it again.
        self.assertFalse(publication.claim("session-a", 4))
        self.assertFalse(publication.claim("session-a", 3))
        self.assertEqual(state.status()["preview_publish"], None)
        publication.record_progress(4, {"stage_name": "Flattening",
                                        "step": 1, "total_steps": 4})
        self.assertEqual(state.status()["preview_publish"]["generation"], 4)
        self.assertEqual(state.status()["progress"]["operation"],
                         "publishing_preview")
        # A stale generation cannot write into the live record.
        self.assertIsNone(publication.record_progress(3, {"step": 9}))

        publication.finish(4)
        self.assertEqual(publication.generation, 0)
        self.assertEqual(publication.completed_generation, 4)
        self.assertFalse(publication.claim("session-a", 4))
        self.assertIsNone(state.status()["preview_publish"])


class CommandReplayTests(unittest.TestCase):
    def test_replay_is_namespaced_per_operation(self):
        state = ServiceState()
        calls = []

        def action(name):
            def run():
                calls.append(name)
                return {"accepted": True, "operation": name}
            return run

        first = state.replay_command("session_stop", "shared-id",
                                     action("stop"))
        repeat = state.replay_command("session_stop", "shared-id",
                                      action("stop-again"))
        self.assertEqual(repeat, first)
        self.assertEqual(calls, ["stop"])

        # A different operation carrying the same command ID is its own
        # command; it must not replay the first operation's response.
        other = state.replay_command("session_load", "shared-id",
                                     action("load"))
        self.assertEqual(other["operation"], "load")
        self.assertEqual(calls, ["stop", "load"])

    def test_command_id_is_required(self):
        state = ServiceState()
        for command_id in (None, "", "   ", 7):
            with self.assertRaisesRegex(ApiError, "command_id"):
                state.replay_command("session_stop", command_id,
                                     lambda: {"accepted": True})


class ConsoleRelayTests(HttpServiceFixture):
    """The tee's whole client surface is the log-kind records in /events.

    There is no /logs relay any more: it retained every console line a
    second time for a client that never read it.
    """

    def relayed(self, buffer=None):
        events = (buffer or self.state.events).read_after(0)["events"]
        return [(record["source"], record["text"]) for record in events
                if record["kind"] == "log"]

    def test_console_lines_are_relayed_as_events_and_logs_is_gone(self):
        self.state.logs.write("stdout", "loading inputs\n")
        self.state.logs.write("stderr", "iteration 1\niteration 2\n")

        self.assertEqual(self.relayed(),
                         [("stdout", "loading inputs"),
                          ("stderr", "iteration 1"),
                          ("stderr", "iteration 2")])
        status, _, _ = self.request("GET", "/logs?after=0")
        self.assertEqual(status, 404)

    def test_successful_polling_requests_are_suppressed_at_the_source(self):
        # The handler's log_request override keeps successful status and
        # event polls out of the terminal (and therefore out of the relay);
        # failed polls and other requests still log.
        handler = spiral_service.SpiralHandler.__new__(
            spiral_service.SpiralHandler)
        logged = []
        handler.log_message = lambda fmt, *args: logged.append(fmt % args)
        handler.requestline = "GET /session/status HTTP/1.1"

        def probe(command, path, code):
            handler.command = command
            handler.path = path
            handler.log_request(code)

        probe("GET", "/session/status", 200)
        probe("GET", "/events?cursor=9", 200)
        self.assertEqual(logged, [])
        probe("GET", "/session/status", 401)
        probe("GET", "/health", 200)
        probe("POST", "/session/run", 200)
        self.assertEqual(len(logged), 3)

    def test_relay_no_longer_string_filters_access_lines(self):
        events = spiral_service.ServiceEventBuffer()
        logs = ServiceLogBuffer(events=events)
        logs.write("stderr", 'SPIRAL_HTTP "GET /session/status HTTP/1.1" 401 -\n')
        logs.write("stderr", "useful fitter warning\n")
        self.assertEqual(
            self.relayed(events),
            [("stderr", 'SPIRAL_HTTP "GET /session/status HTTP/1.1" 401 -'),
             ("stderr", "useful fitter warning")])

    def test_carriage_return_progress_redraws_are_relayed_as_complete_lines(self):
        events = spiral_service.ServiceEventBuffer()
        logs = ServiceLogBuffer(events=events)
        logs.write("stderr", "\rloading patches:  25%|██▌       | 1/4")
        logs.write("stderr", "\rloading patches: 100%|██████████| 4/4\n")
        logs.write("stderr", "\r 40%|████      | 400/1000")
        logs.write("stderr", "\r100%|██████████| 1000/1000\n")

        self.assertEqual(
            [text for _, text in self.relayed(events)],
            [
                "loading patches:  25%|██▌       | 1/4",
                "loading patches: 100%|██████████| 4/4",
                " 40%|████      | 400/1000",
                "100%|██████████| 1000/1000",
            ])

    def test_oversized_lines_are_truncated_before_the_ring(self):
        events = spiral_service.ServiceEventBuffer()
        logs = ServiceLogBuffer(events=events)
        logs.write("stdout", "x" * (spiral_service.MAX_LOG_ENTRY_CHARS + 50)
                   + "\n")
        text = self.relayed(events)[0][1]
        self.assertTrue(text.endswith(" … [truncated]"))
        self.assertEqual(text.count("x"), spiral_service.MAX_LOG_ENTRY_CHARS)


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

    def test_registration_reports_each_hashed_file(self):
        artifact_root = self.root / "artifact-progress"
        artifact_root.mkdir()
        (artifact_root / "manifest.json").write_text("{}")
        (artifact_root / "first.bin").write_bytes(b"first")
        (artifact_root / "second.bin").write_bytes(b"second")
        progress = []

        self.state.artifacts.register_directory(
            "spiral-preview", "session-1", 2, artifact_root, "manifest.json",
            progress=lambda current, total, relative: progress.append(
                (current, total, relative)),
            hash_workers=2)

        self.assertEqual(
            progress,
            [
                (1, 3, "first.bin"),
                (2, 3, "manifest.json"),
                (3, 3, "second.bin"),
            ])

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


class BindingOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        # Somewhere outside the service's bindings.
        self.root = base / "elsewhere"
        self.root.mkdir()
        self.output = base / "output"
        self.output.mkdir()
        self.cache = base / "cache"
        self.state = ServiceState(bindings=spiral_service.bind_service_inputs(
                                      self.output, self.cache),
                                  startup_run={"z_begin": 0, "z_end": 10,
                                               "config": _NO_DENSE_LOSSES})

    def tearDown(self):
        self.temporary.cleanup()

    def test_bindings_endpoint_advertises_the_host_paths(self):
        response = self.state.binding_advertisement()
        # The project supplies the scroll and its inputs; there is no dataset.
        self.assertFalse({"root", "resolved", "scroll_spec", "detected_checkpoints",
                          "missing_required", "ok"} & set(response))
        # The startup-bound deployment roots are the advertisement.
        self.assertEqual(response["bindings"]["output_directory"],
                         str(self.output))
        self.assertEqual(response["bindings"]["cache_directory"],
                         str(self.cache))
        self.assertFalse(response["bindings"]["winding_inference"]["bound"])
        self.assertFalse(response["capabilities"]["winding_inference"])
        self.assertNotIn("dataset_root", self.state.health())

    def test_bindings_endpoint_advertises_named_session(self):
        state = ServiceState(
            bindings=spiral_service.bind_service_inputs(
                self.output / "alice", self.cache),
            session_name="alice")
        response = state.binding_advertisement()
        self.assertEqual(response["session_name"], "alice")
        self.assertEqual(
            response["bindings"]["output_directory"],
            str(self.output / "alice"))

    def test_winding_counts_come_from_the_configuration_alone(self):
        # No host scroll specification seeds the catalog's winding defaults.
        catalog = self.state.configuration_catalog()
        generic = Config().as_dict()
        for key in ("shell_outer_winding_idx", "model_gap_expander_num_windings",
                    "model_gap_expander_capacity_windings"):
            self.assertEqual(catalog["defaults"][key], generic[key])

    def test_generated_state_roots_live_under_the_output_root(self):
        # Every generated-state family hangs off the bound --output root.
        self.assertEqual(self.state._output_root(), self.output)
        self.assertEqual(self.state._staging_root(),
                         self.output / ".spiral-upload-staging")
        self.assertEqual(self.state._checkpoint_upload_root(),
                         self.output / "uploaded-checkpoints")

    def test_initialize_rejects_client_base_input_paths(self):
        with self.assertRaises(ApiError) as caught:
            self.state.initialize(with_project({"paths": {"umbilicus": "/attacker/umbilicus.json"},
                                   "run": {"z_begin": 0, "z_end": 10}}))
        self.assertEqual(caught.exception.status, 400)
        fields = {detail["field"] for detail in caught.exception.details}
        self.assertEqual(fields, {"paths.umbilicus"})

    def test_initialize_rejects_unadvertised_checkpoint(self):
        with self.assertRaises(ApiError) as caught:
            self.state.initialize(with_project({"paths": {"checkpoint": "/attacker/model.ckpt"},
                                   "run": {"z_begin": 0, "z_end": 10}}))
        self.assertEqual(caught.exception.status, 400)

    def _attach_session_for_request(self, config=None):
        """Attach a fake session whose canonical request the service produced."""
        session = _attach_fake_session(self.state, self.output, VOLUME)
        _publish_request(self.state, {"run": {
            "z_begin": 0, "z_end": 10,
            "config": {**_NO_DENSE_LOSSES, **(config or {})}}})
        return session

    def _stage_for(self, request):
        paths, run, preview, scroll = self.state._prepare_session_request(
            with_project(request))
        with self.state.lock:
            return self.state._rebuild_stage_locked(
                paths, run, preview, scroll, SessionState.Idle)

    def _base_request(self, config=None, **run):
        return {"run": {"z_begin": 0, "z_end": 10,
                        "config": {**_NO_DENSE_LOSSES, **(config or {})},
                        **run}}

    def test_only_allowlisted_configuration_keeps_the_loaded_inputs(self):
        self._attach_session_for_request()
        self.assertEqual(self._stage_for(self._base_request()), "model")
        self.assertEqual(
            self._stage_for(self._base_request(
                {"model_linear_z_resolution": 24})),
            "model")
        # An unaudited key alongside an allowlisted one is still the whole
        # build, and so is anything outside run.config.
        self.assertEqual(
            self._stage_for(self._base_request({
                "model_linear_z_resolution": 24,
                "optimizer_random_seed": 7})),
            "all")
        # A run-boundary key is applied through apply_config by the model
        # rebuild, so it never widens the stage on its own.
        self.assertEqual(
            self._stage_for(self._base_request({"loss_weight_patch_radius": 1.0})),
            "model")
        self.assertEqual(
            self._stage_for(self._base_request(run_tag="second")), "all")
        self.assertEqual(
            self._stage_for({"run": {"z_begin": 0, "z_end": 20,
                                     "config": dict(_NO_DENSE_LOSSES)}}),
            "all")

    def test_shell_atlas_settings_rebuild_everything_once_tracks_were_shell_filtered(self):
        # Without a shell-filtered track pool the atlas settings are ordinary
        # run-boundary knobs the model rebuild applies through apply_config.
        session = self._attach_session_for_request()
        self.assertEqual(
            self._stage_for(self._base_request({"shell_num_theta_bins": 360})),
            "model")
        # A session that read both tracks and the store's outer shell
        # filtered the tracks against the shell; apply_config refuses the
        # atlas settings there, so the request has to rebuild the host
        # inputs.
        session.input_summary = {"outer_shell": "6d1f0e4e-6d7a-4d55-9d0c-2b8f9a0c1e12",
                                 "selection": {"track": 3}}
        live = self.state.session_paths
        run = self._base_request({"shell_num_theta_bins": 360})["run"]
        _, run_config, preview, scroll = self.state._prepare_session_request(with_project(
            {"run": run}))
        # As the request validates them: the project supplies the umbilicus
        # and the build the fit volume.
        requested = dataclasses.replace(live, umbilicus="", volume_uuid="")
        with self.state.lock:
            self.assertEqual(
                self.state._rebuild_stage_locked(
                    requested, run_config, preview, scroll, SessionState.Idle),
                "all")
            # Other shell loss settings still keep the loaded inputs.
            _, other_run, _, _ = self.state._prepare_session_request(with_project(
                {"run": self._base_request(
                    {"shell_huber_delta": 8.0})["run"]}))
            self.assertEqual(
                self.state._rebuild_stage_locked(
                    requested, other_run, preview, scroll, SessionState.Idle),
                "model")

    def test_a_session_that_is_not_idle_has_nothing_to_rebuild_around(self):
        self._attach_session_for_request()
        paths, run, preview, scroll = self.state._prepare_session_request(with_project(
            self._base_request({"model_linear_z_resolution": 24})))
        with self.state.lock:
            self.assertEqual(
                self.state._rebuild_stage_locked(
                    paths, run, preview, scroll, SessionState.Error),
                "all")

    def test_a_model_stage_rebuild_keeps_the_session_and_its_inputs(self):
        session = self._attach_session_for_request()
        generation = self.state.session_generation
        response = self.state.rebuild(with_project(
            self._base_request({"model_linear_z_resolution": 24})))
        self.assertEqual(response["stage"], "model")
        deadline = time.monotonic() + 5.0
        while self.state._building and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.state._building)
        # The same session object, not a replacement: its host inputs, and
        # everything incorporated into them, are still resident.
        self.assertIs(self.state.session, session)
        self.assertFalse(session.closed)
        self.assertEqual(self.state.session_generation, generation)
        self.assertEqual(len(session.model_rebuilds), 1)
        rebuilt_paths, rebuilt_run = session.model_rebuilds[0]
        self.assertEqual(rebuilt_run.config["model_linear_z_resolution"], 24)
        self.assertEqual(rebuilt_paths.manifest(),
                         self.state.session_request["paths"])
        self.assertEqual(
            self.state.session_request["run"]["config"]
            ["model_linear_z_resolution"], 24)

    def test_bound_request_filters_paths_with_input_toggles(self):
        request = self.state._bound_session_request({
            "run": {"z_begin": 0, "z_end": 10},
        }, _PROJECT)
        # Patches, fibers and PCLs come from the object store, never from
        # paths.
        self.assertFalse({"verified_patches", "fibers", "pcls"} & set(request["paths"]))
        self.assertNotIn("dataset_root", request["paths"])

        self.assertNotIn("tracks_dbm", request["paths"])
        with self.assertRaises(ApiError):
            # Tracks come from the store too: a request cannot name a database.
            self.state._bound_session_request({
                "paths": {"tracks_dbm": "/a/store.dbm"},
                "run": {"z_begin": 0, "z_end": 10},
            }, _PROJECT)

    def _bind_winding_inference(self):
        winding = self.root / "winding_inference"
        winding.mkdir()
        (winding / "manifest.json").write_text(json.dumps({
            "artifact_type": "winding_inference_crossings",
            "format_version": 1,
        }))
        self.state.bindings = spiral_service.bind_service_inputs(
            self.output, self.cache, winding)
        return winding

    def test_omitted_spacing_mode_preserves_default_winding_input(self):
        winding = self._bind_winding_inference()
        config = {**_NO_DENSE_LOSSES,
                  "input_use_winding_inference": True,
                  "input_use_outer_shell": True}
        config.pop("dense_spacing_mode")
        self.assertNotIn("dense_spacing_mode", config)
        request = {"run": {"z_begin": 0, "z_end": 10, "config": config}}

        paths, run, _, _ = self.state._prepare_session_request(with_project(request))

        self.assertEqual(paths.winding_inference, str(winding))
        self.assertEqual(run.config, config)

        # Missing default-mode inputs must fail preflight, before GPU work,
        # saying how the service gets them.
        self.state.bindings = spiral_service.bind_service_inputs(
            self.output, self.cache)
        with self.assertRaises(ApiError) as caught:
            self.state._prepare_session_request(with_project(request))
        [detail] = [detail for detail in caught.exception.details
                    if detail["field"] == "winding_inference"]
        self.assertIn("--winding-inference PATH", detail["message"])

    def test_checkpoint_config_selects_winding_model_inputs(self):
        winding = self._bind_winding_inference()

        checkpoint_config = Config({
            **_NO_DENSE_LOSSES,
            "dense_spacing_mode": "winding_model",
            "input_use_winding_inference": True,
            "input_use_outer_shell": True,
        }).as_dict()
        checkpoint = self._write_checkpoint(
            "winding-model.ckpt", checkpoint_config)

        request, input_config = self.state._bound_session_request({
            "paths": {"checkpoint": str(checkpoint)},
            "run": {"z_begin": 0, "z_end": 10, "config": {}},
        }, _PROJECT, include_input_config=True)

        self.assertEqual(request["paths"]["winding_inference"], str(winding))
        self.assertEqual(input_config["dense_spacing_mode"], "winding_model")

        # Preflight validates against the same checkpoint-derived selection
        # config while preserving an empty override object for the runtime.
        paths, run, _, _ = self.state._prepare_session_request(with_project({
            "paths": {"checkpoint": str(checkpoint)},
            "run": {"z_begin": 0, "z_end": 10, "config": {}},
        }))
        self.assertEqual(paths.winding_inference, str(winding))
        self.assertEqual(run.config, {})
        self.assertEqual((run.z_begin, run.z_end), (0, 10))

        # Loading the checkpoint seeds the canonical run range (and therefore
        # the VC3D dock), even though ordinary request preparation leaves later
        # explicit dock edits alone.
        with mock.patch("spiral_runtime.create_session",
                        return_value=FakeSession()):
            response = self.state.initialize(with_project({
                "paths": {"checkpoint": str(checkpoint)},
                "run": {"z_begin": 0, "z_end": 10, "config": {}},
            }))
            _await_build(self.state)
        attached_run = response["session_request"]["run"]
        self.assertEqual(
            (attached_run["z_begin"], attached_run["z_end"]),
            (checkpoint_config["z_begin"], checkpoint_config["z_end"]))

    def test_status_advertises_canonical_active_session_request(self):
        config = {
            "dense_spacing_mode": "grad_mag",
            "loss_weight_dense_spacing": 0,
            "loss_weight_dense_normals": 0,
            "loss_weight_shell_outer": 0,
            "loss_weight_patch_radius": 7.5,
        }
        request = {
            "run": {
                "z_begin": 100,
                "z_end": 900,
                "config": config,
            },
            "preview": {"first_winding": 12, "variant": "raw"},
        }
        with mock.patch("spiral_runtime.create_session",
                        return_value=FakeSession()):
            # Initialization is accepted immediately; the runtime is constructed
            # off the request thread (the fake finishes instantly, so the
            # accepted response may already report Idle rather than Loading).
            response = self.state.initialize(with_project(request))
            self.assertTrue(response["initializing"])
            self.assertIn(response["state"],
                          {SessionState.Loading, SessionState.Idle})
            _await_build(self.state)

        attached = response["session_request"]
        self.assertEqual(attached, self.state.status()["session_request"])
        self.assertNotIn("dataset_root", attached["paths"])
        self.assertEqual(attached["paths"]["volume_uuid"], VOLUME)
        self.assertEqual(attached["paths"]["provenance"]["project"], "s1")
        self.assertFalse({"verified_patches", "fibers", "pcls"} & set(attached["paths"]))
        self.assertEqual(attached["run"]["z_begin"], 100)
        self.assertEqual(attached["run"]["z_end"], 900)
        for key in SCROLL_SPEC_OWNED_RUN_KEYS:
            self.assertNotIn(key, attached["run"])
        self.assertEqual(attached["run"]["config"], config)
        self.assertEqual(attached["preview"],
                         {"first_winding": 12, "variant": "raw"})
        # The project's umbilicus, written where the fit reads it, and the
        # scroll: the project's name and facts.
        self.assertEqual(attached["project"]["volume"], VOLUME)
        self.assertEqual(attached["project"]["umbilicus"]["version"],
                         UMBILICUS["version"])
        umbilicus = json.loads(Path(attached["paths"]["umbilicus"]).read_text())
        self.assertEqual([[p["z"], p["y"], p["x"]] for p in umbilicus["control_points"]],
                         UMBILICUS["points_zyx"])
        self.assertEqual(attached["scroll"]["name"], "s1")
        self.assertEqual(attached["scroll"]["voxel_size_um"], 9.6)
        self.assertEqual(attached["scroll"]["umbilicus_coordinate_scale"], 1.0)

    def test_initialization_refuses_run_keys_the_scroll_spec_owns(self):
        for key, value in (("scroll_name", "renamed"), ("voxel_size_um", 4.0),
                           ("lasagna_group", "8"), ("lasagna_scale", 2)):
            with self.subTest(key=key):
                with self.assertRaises(ApiError) as caught:
                    self.state.initialize(with_project({"run": {"z_begin": 100, "z_end": 900,
                                                   key: value}}))
                self.assertEqual(caught.exception.status, 400)
                self.assertEqual([detail["field"] for detail
                                  in caught.exception.details],
                                 [f"run.{key}"])

    def test_initialization_refuses_z_range_in_advanced_config(self):
        with self.assertRaises(ApiError) as caught:
            self.state.initialize(with_project({
                "run": {"z_begin": 100, "z_end": 900,
                        "config": {"z_begin": 200, "z_end": 800}},
            }))
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(
            [detail["field"] for detail in caught.exception.details],
            ["run.config.z_begin", "run.config.z_end"])

    def test_a_failed_build_reports_error_and_a_rebuild_recovers(self):
        request = {
            "run": {
                "z_begin": 0,
                "z_end": 10,
                "config": {
                    "dense_spacing_mode": "grad_mag",
                    "loss_weight_dense_spacing": 0,
                    "loss_weight_dense_normals": 0,
                    "loss_weight_shell_outer": 0,
                },
            },
        }
        with mock.patch("spiral_runtime.create_session",
                        side_effect=RuntimeError("startup failed")):
            self.state.initialize(with_project(request))
            status = _await_build(self.state)
        # A build that fails is Error with the cause, not a service that has
        # quietly stopped having a session.
        self.assertEqual(status["state"], SessionState.Error)
        self.assertIn("startup failed", status["error"])
        with self.assertRaises(ApiError) as caught:
            self.state.stop()
        self.assertIn("startup failed", caught.exception.message)

        # Rebuild with defaults is the documented recovery.
        with mock.patch("spiral_runtime.create_session",
                        return_value=FakeSession()):
            self.state.rebuild(with_project({"defaults": True}))
            status = _await_build(self.state)
        self.assertEqual(status["state"], SessionState.Idle)
        self.assertIsNone(status["error"])

    def test_save_checkpoint_names_a_file_the_service_places(self):
        """The client names the checkpoint; the service owns the location."""
        _attach_fake_session(self.state, self.output, VOLUME)
        for name in ("", "   ", "../escape.ckpt", "sub/dir.ckpt", "..",
                     "/absolute.ckpt"):
            with self.assertRaises(ApiError) as caught:
                self.state.save_checkpoint({"name": name})
            self.assertEqual(caught.exception.status, 400, name)

        response = self.state.save_checkpoint({"name": "manual.ckpt"})
        expected = self.output / "checkpoints" / "manual.ckpt"
        self.assertEqual(response["checkpoint_path"], str(expected))
        self.assertTrue(expected.parent.is_dir())

        # A bare name gains the conventional suffix.
        response = self.state.save_checkpoint({"name": "second"})
        self.assertEqual(response["checkpoint_path"],
                         str(self.output / "checkpoints" / "second.ckpt"))

    def test_load_checkpoint_reads_only_service_owned_checkpoints(self):
        session = _attach_fake_session(self.state, self.output, VOLUME)
        outside = self.root / "elsewhere.ckpt"
        outside.write_bytes(b"PK\x03\x04checkpoint")
        uploaded = self.output / "uploaded-checkpoints" / "a.ckpt"
        uploaded.parent.mkdir(parents=True, exist_ok=True)
        uploaded.write_bytes(b"PK\x03\x04checkpoint")
        # A host checkpoint must be one this service advertises, and an
        # uploaded one must live in the upload store; neither field takes an
        # arbitrary path, and a load names exactly one of them.
        for request in ({"host_checkpoint": str(outside)},
                        {"host_checkpoint": str(self.output / "absent.ckpt")},
                        # The upload store is not advertised, so the two
                        # sources name disjoint sets.
                        {"host_checkpoint": str(uploaded)},
                        {"uploaded_checkpoint": str(outside)},
                        {"uploaded_checkpoint": str(self.output / "a.ckpt")},
                        {"host_checkpoint": str(uploaded),
                         "uploaded_checkpoint": str(uploaded)},
                        {}):
            with self.subTest(request=request):
                with self.assertRaises(ApiError) as caught:
                    self.state.load_checkpoint(request)
                self.assertEqual(caught.exception.status, 400)
        self.assertEqual(session.loaded, [])

    def test_bindings_advertise_the_checkpoints_a_load_may_name(self):
        _attach_fake_session(self.state, self.output, VOLUME)
        saved = self.state.save_checkpoint({"name": "manual"})["checkpoint_path"]
        staging = self.output / ".spiral-artifacts" / "checkpoint-abc"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "checkpoint.ckpt").write_bytes(b"PK\x03\x04checkpoint")

        listed = self.state.binding_advertisement()["session_checkpoints"]

        self.assertIn(saved, listed)
        # Artifact staging is transfer plumbing, not a user choice.
        self.assertNotIn(str(staging / "checkpoint.ckpt"), listed)

    def test_load_checkpoint_reports_the_restored_iteration(self):
        session = _attach_fake_session(self.state, self.output, VOLUME)
        checkpoint = self.output / "uploaded-checkpoints" / "a.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"PK\x03\x04checkpoint")
        revision = self.state.session_revision

        response = self.state.load_checkpoint(
            {"uploaded_checkpoint": str(checkpoint)})

        self.assertTrue(response["loaded"])
        self.assertEqual(response["restored_iteration"], 4200)
        self.assertEqual(session.loaded, [str(checkpoint)])
        self.assertEqual(self.state.session_paths.checkpoint, str(checkpoint))
        self.assertEqual(
            self.state.session_request["paths"]["checkpoint"], str(checkpoint))
        self.assertEqual(self.state.session_revision, revision + 1)

    def test_a_refused_checkpoint_is_a_conflict_and_changes_nothing(self):
        session = _attach_fake_session(self.state, self.output, VOLUME)
        session.load_refusal = "checkpoint model z-domain [0, 10) is not..."
        checkpoint = self.output / "a.ckpt"
        checkpoint.write_bytes(b"PK\x03\x04checkpoint")
        revision = self.state.session_revision

        with self.assertRaises(ApiError) as caught:
            self.state.load_checkpoint({"host_checkpoint": str(checkpoint)})

        self.assertEqual(caught.exception.status, 409)
        self.assertIn("z-domain", caught.exception.message)
        self.assertEqual(self.state.session_revision, revision)
        self.assertEqual(self.state.session_paths.checkpoint, "")

    def test_autosave_on_pause_is_a_run_request_flag_defaulting_on(self):
        session = _attach_fake_session(self.state, self.output, VOLUME)

        _planned_run(self.state, {"iterations": 4})
        self.assertEqual(session.autosave_calls, [True])

        _planned_run(self.state, {"iterations": 4, "autosave_on_pause": False})
        self.assertEqual(session.autosave_calls, [True, False])

        with self.assertRaises(ApiError) as caught:
            _planned_run(self.state,
                         {"iterations": 4, "autosave_on_pause": "no"})
        self.assertEqual(caught.exception.status, 400)

    def test_preview_schedule_is_optional_validated_and_service_owned(self):
        session = _attach_fake_session(self.state, self.output, VOLUME)
        _planned_run(self.state, {
            "iterations": 4,
            "preview_schedule": {
                "cadence_iterations": 100, "diagnostics": True},
        })
        self.assertEqual(session.preview_schedules, [{
            "cadence_iterations": 100, "diagnostics": True}])
        status = self.state.status()
        self.assertEqual(status["preview_schedule"], {
            "cadence_iterations": 100, "diagnostics": True})
        self.assertEqual(status["next_preview_iteration"], 105)

        for invalid in ({"cadence_iterations": 0, "diagnostics": False},
                        {"cadence_iterations": 10, "diagnostics": "yes"}):
            session.state = SessionState.Idle
            with self.assertRaises(ApiError) as caught:
                _planned_run(self.state, {
                    "iterations": 4, "preview_schedule": invalid})
            self.assertEqual(caught.exception.status, 400)

    def test_export_preview_accepts_and_runs_off_the_request_thread(self):
        """A preview costs minutes; the verb accepts it and returns.

        Holding the request open outlived every client transfer timeout, so
        a preview that succeeded was reported as a failure. The client
        follows preview_exporting in the status it already polls.
        """
        session = _attach_fake_session(self.state, self.output, VOLUME)
        release = threading.Event()
        session.preview_gate = release

        response = self.state.export_preview()
        self.assertTrue(response["accepted"])
        self.assertNotIn("exported", response)
        self.assertTrue(response["preview_exporting"])

        # Single-flight while the export is still running.
        with self.assertRaises(ApiError) as caught:
            self.state.export_preview()
        self.assertEqual(caught.exception.status, 409)

        release.set()
        deadline = time.monotonic() + 5.0
        while self.state.status()["preview_exporting"] \
                and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.state.status()["preview_exporting"])
        self.assertEqual(session.previews, 1)

        session.state = SessionState.Running
        response = self.state.export_preview()
        self.assertTrue(response["accepted"])
        deadline = time.monotonic() + 5.0
        while self.state.status()["preview_exporting"] \
                and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(session.previews, 2)

    def test_a_failed_preview_export_is_reported_through_status(self):
        session = _attach_fake_session(self.state, self.output, VOLUME)
        session.preview_failure = "OSError: no space left on device"

        self.state.export_preview()
        deadline = time.monotonic() + 5.0
        while self.state.status()["preview_exporting"] \
                and time.monotonic() < deadline:
            time.sleep(0.01)
        status = self.state.status()
        self.assertFalse(status["preview_exporting"])
        self.assertIn("no space left", status["preview_publish_error"])

    def _write_checkpoint(self, name, cfg, volume_uuid=None):
        """A checkpoint carrying the durable cfg a refusal is analysed from."""
        import torch

        path = self.output / name
        torch.save({
            # Checkpoints store the full configuration schema, and the
            # refusal analysis compares key sets against exactly that schema.
            "schema_version": 2, "cfg": dict(cfg),
            "input_manifest": {
                "volume_uuid": VOLUME if volume_uuid is None else volume_uuid,
                # Object-store inputs of some fit: the fake session has no
                # manifest to compare them with, so the cfg alone decides.
                "object_store": {"objects": [], "relationships": []}},
        }, path)
        return path

    def _refuse_load(self, session, checkpoint, reason="model mismatch"):
        session.load_refusal = reason
        with self.assertRaises(ApiError) as caught:
            self.state.load_checkpoint({"host_checkpoint": str(checkpoint)})
        return caught.exception

    def test_a_refusal_reports_the_rebuild_that_would_accept_the_checkpoint(self):
        session = _attach_fake_session(self.state, self.output, VOLUME)
        live = Config().as_dict()
        session.applied_config = dict(live)

        # A checkpoint differing only in allowlisted model configuration is a
        # model-stage rebuild away from being loadable.
        error = self._refuse_load(session, self._write_checkpoint(
            "model.ckpt", {**live, "model_num_flow_stages": 3}))
        self.assertEqual(error.status, 409)
        self.assertEqual(error.payload["stage"], "model")
        # The verdict's own text, unmodified.
        self.assertEqual(error.payload["reasons"], ["model mismatch"])
        self.assertNotIn("refused", error.payload)

        # The stage comes from the whole cfg diff, so a checkpoint differing
        # in a host-affecting key needs the whole build even when the
        # preflight only complained about the model.
        error = self._refuse_load(session, self._write_checkpoint(
            "host.ckpt", {**live, "model_num_flow_stages": 3,
                          "patch_erode_patches": 3}))
        self.assertEqual(error.payload["stage"], "all")
        # And a model z-domain mismatch reaches "all" through z_begin/z_end
        # rather than through a rule of its own.
        error = self._refuse_load(session, self._write_checkpoint(
            "domain.ckpt", {**live, "z_end": live["z_end"] + 1000}))
        self.assertEqual(error.payload["stage"], "all")

        # Stored configurations are loaded tolerantly: a key the checkpoint
        # predates takes its default and a key the schema no longer has is
        # dropped, so neither turns a rebuildable model mismatch into a
        # permanent refusal.
        partial = {key: value for key, value in live.items()
                   if key != "input_use_tracks"}
        partial["model_num_flow_stages"] = 3
        partial["a_setting_that_no_longer_exists"] = 1
        error = self._refuse_load(session, self._write_checkpoint(
            "tolerated.ckpt", partial))
        self.assertEqual(error.payload["stage"], "model")
        self.assertNotIn("refused", error.payload)
        # The default the rebuild would resume with is what the stage
        # calculation diffs, so a live setting that differs from it promotes
        # the rebuild to the full host-input stage.
        session.applied_config["input_use_tracks"] = not live["input_use_tracks"]
        error = self._refuse_load(session, self._write_checkpoint(
            "tolerated-live-differs.ckpt", partial))
        self.assertEqual(error.payload["stage"], "all")

    def test_a_refusal_no_rebuild_can_fix_offers_nothing(self):
        session = _attach_fake_session(self.state, self.output, VOLUME)
        live = Config().as_dict()
        session.applied_config = dict(live)

        # Fitted in another volume entirely.
        error = self._refuse_load(session, self._write_checkpoint(
            "foreign.ckpt", dict(live), volume_uuid="another-volume"))
        self.assertTrue(error.payload["refused"])
        self.assertNotIn("stage", error.payload)

        # A stored value the schema cannot interpret.
        error = self._refuse_load(session, self._write_checkpoint(
            "invalid.ckpt", {**live, "dense_spacing_mode": "bogus"}))
        self.assertTrue(error.payload["refused"])
        self.assertNotIn("stage", error.payload)

        # A file that will not load at all.
        unreadable = self.output / "unreadable.ckpt"
        unreadable.write_bytes(b"PK\x03\x04not-a-checkpoint")
        error = self._refuse_load(session, unreadable)
        self.assertTrue(error.payload["refused"])

    def test_allow_rebuild_rebuilds_onto_the_checkpoint_without_overrides(self):
        session = _attach_fake_session(self.state, self.output, VOLUME)
        live = Config().as_dict()
        session.applied_config = dict(live)
        # A live session request carrying an advanced profile, as a client
        # that had been editing configuration would leave it.
        _publish_request(self.state, {
            "run": {"z_begin": 0, "z_end": 10,
                    "config": {**_NO_DENSE_LOSSES, "model_num_flow_stages": 9}}})
        checkpoint = self._write_checkpoint("resume.ckpt", dict(live))

        rebuilds = []
        self.state.rebuild = lambda request: rebuilds.append(request) or {}
        response = self.state.load_checkpoint({
            "host_checkpoint": str(checkpoint), "allow_rebuild": True})

        self.assertEqual(len(rebuilds), 1)
        self.assertEqual(response["checkpoint_path"], str(checkpoint))
        self.assertEqual(rebuilds[0]["paths"]["checkpoint"], str(checkpoint))
        self.assertNotIn("dataset_root", rebuilds[0]["paths"])
        self.assertNotIn("verified_patches", rebuilds[0]["paths"])
        self.assertNotIn("output_directory", rebuilds[0]["paths"])
        # No advanced overrides: the runtime layers run.config on top of the
        # checkpoint's own cfg, so resending the profile would re-impose the
        # very keys the preflight just refused.
        self.assertEqual(rebuilds[0]["run"]["config"], {})
        self.assertEqual(rebuilds[0]["run"]["z_begin"], live["z_begin"])
        self.assertEqual(rebuilds[0]["run"]["z_end"], live["z_end"])

    def test_allow_rebuild_must_be_a_boolean(self):
        _attach_fake_session(self.state, self.output, VOLUME)
        checkpoint = self.output / "a.ckpt"
        checkpoint.write_bytes(b"PK\x03\x04checkpoint")
        with self.assertRaises(ApiError) as caught:
            self.state.load_checkpoint({"host_checkpoint": str(checkpoint),
                                        "allow_rebuild": "yes"})
        self.assertEqual(caught.exception.status, 400)

    def test_a_rebuild_refuses_overrides_its_checkpoint_contradicts(self):
        _attach_fake_session(self.state, self.output, VOLUME)
        live = Config().as_dict()
        checkpoint = self._write_checkpoint("resume.ckpt", dict(live))
        request = {
            "paths": {"checkpoint": str(checkpoint)},
            "run": {"z_begin": 0, "z_end": 10,
                    "config": {**_NO_DENSE_LOSSES,
                               "model_flow_bounds_radius": 128}},
        }
        with self.assertRaises(ApiError) as caught:
            self.state.rebuild(with_project(request))
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual([detail["field"] for detail in caught.exception.details],
                         ["run.config.model_flow_bounds_radius"])
        # Everything else stays a legitimate change to make while resuming.
        request["run"]["config"] = {**_NO_DENSE_LOSSES,
                                    "loss_weight_patch_radius": 1.0}
        self.state._reject_overrides_the_checkpoint_contradicts(
            str(checkpoint), request["run"]["config"])

    def test_load_checkpoint_requires_an_idle_session(self):
        session = _attach_fake_session(self.state, self.output, VOLUME)
        session.state = SessionState.Running
        checkpoint = self.output / "a.ckpt"
        checkpoint.write_bytes(b"PK\x03\x04checkpoint")
        with self.assertRaises(ApiError) as caught:
            self.state.load_checkpoint({"host_checkpoint": str(checkpoint)})
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(session.loaded, [])

    def test_error_session_can_rebuild_from_a_different_checkpoint(self):
        _attach_fake_session(self.state, self.output, VOLUME)
        # A failed build has a canonical request and identity to recover, but
        # no usable resident object on which an in-place load could operate.
        self.state.session = None
        self.state._session_state = SessionState.Error
        self.state._session_error = "bad checkpoint"
        checkpoint = self.output / "uploaded-checkpoints" / "recovery.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save({
            "schema_version": 2,
            "cfg": {**Config().as_dict(), **_NO_DENSE_LOSSES},
            "input_manifest": {"volume_uuid": VOLUME},
        }, checkpoint)

        with self.assertRaises(ApiError) as caught:
            self.state.load_checkpoint(
                {"uploaded_checkpoint": str(checkpoint)})
        self.assertEqual(caught.exception.status, 409)

        # Exercise the real rebuild request parser.  The stored session
        # request contains all canonical service-owned paths, but retrying a
        # checkpoint must feed only client-selectable paths back through the
        # binding boundary.
        with mock.patch.object(self.state, "_begin_build") as begin_build:
            response = self.state.load_checkpoint({
                "uploaded_checkpoint": str(checkpoint),
                "allow_rebuild": True,
            })

        begin_build.assert_called_once()
        rebuilt_paths, rebuilt_run = begin_build.call_args.args[:2]
        self.assertEqual(rebuilt_paths.checkpoint, str(checkpoint))
        self.assertFalse(hasattr(rebuilt_paths, "dataset_root"))
        self.assertEqual(rebuilt_paths.output_directory, str(self.output))
        self.assertEqual(rebuilt_run.config, {})
        self.assertTrue(response["rebuilding"])
        self.assertEqual(response["checkpoint_path"], str(checkpoint))


def _write_autosave(directory, *, iterations, namespace, volume_uuid,
                    payload=b"payload", corrupt=False):
    """Write one autosave plus the metadata that makes it selectable."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / AUTOSAVE_CHECKPOINT_NAME
    checkpoint.write_bytes(_zip_checkpoint_bytes(payload))
    write_autosave_metadata(checkpoint, session_namespace=namespace,
                            volume_uuid=volume_uuid,
                            completed_iterations=iterations)
    if corrupt:
        # Truncate the archive after the sidecar recorded it: the metadata
        # still claims a valid container of a known size and digest.
        checkpoint.write_bytes(b"PK\x03\x04truncated")
    return checkpoint


class StartupAutosaveSelectionTests(unittest.TestCase):
    """The startup autosave is chosen from metadata, never from filenames."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name) / "output" / "session-a"
        self.output.mkdir(parents=True)
        self.volume = VOLUME

    def tearDown(self):
        self.temporary.cleanup()

    def _select(self):
        return select_startup_autosave(
            self.output, session_namespace=self.output,
            volume_uuid=self.volume)

    def test_the_furthest_autosave_wins_regardless_of_filename_order(self):
        # "aaa-run" sorts first and "zzz-run" sorts last; neither ordering
        # decides anything. The completed iteration count does.
        _write_autosave(self.output / "zzz-run", iterations=40,
                        namespace=self.output, volume_uuid=self.volume)
        winner = _write_autosave(self.output / "aaa-run", iterations=900,
                                 namespace=self.output,
                                 volume_uuid=self.volume)
        selection = self._select()
        self.assertEqual(selection.selected.checkpoint, str(winner))
        self.assertEqual(selection.selected.completed_iterations, 900)
        self.assertEqual(
            [reason for _, reason in selection.rejected],
            [f"superseded by {winner} at 900 iterations"])

    def test_a_checkpoint_without_metadata_is_never_selected(self):
        # A bare autosave file is inert: nothing says which session
        # namespace or volume it belongs to, so it is not a candidate.
        run = self.output / "run"
        run.mkdir()
        (run / AUTOSAVE_CHECKPOINT_NAME).write_bytes(_zip_checkpoint_bytes())
        self.assertIsNone(self._select().selected)

    def test_another_namespace_or_volume_is_rejected_not_selected(self):
        _write_autosave(self.output / "other-namespace", iterations=10,
                        namespace=Path(self.temporary.name) / "output" / "b",
                        volume_uuid=self.volume)
        _write_autosave(self.output / "other-volume", iterations=20,
                        namespace=self.output,
                        volume_uuid="another-volume")
        selection = self._select()
        self.assertIsNone(selection.selected)
        reasons = sorted(reason for _, reason in selection.rejected)
        self.assertEqual(len(reasons), 2)
        self.assertIn("belongs to session namespace", reasons[0])
        self.assertIn("was fitted in the volume another-volume", reasons[1])

    def test_unreadable_metadata_is_rejected_with_its_reason(self):
        run = self.output / "run"
        run.mkdir()
        (run / AUTOSAVE_METADATA_NAME).write_text("{not json")
        stale = self.output / "stale"
        stale.mkdir()
        (stale / AUTOSAVE_METADATA_NAME).write_text(
            json.dumps({"schema": "spiral-autosave/0"}))
        selection = self._select()
        self.assertIsNone(selection.selected)
        self.assertEqual(len(selection.rejected), 2)
        for _, reason in selection.rejected:
            self.assertIn("unreadable metadata", reason)

    def test_a_selected_autosave_must_match_its_recorded_identity(self):
        _write_autosave(self.output / "run", iterations=10,
                        namespace=self.output, volume_uuid=self.volume,
                        corrupt=True)
        selected = self._select().selected
        self.assertIsNotNone(selected)
        with self.assertRaises(AutosaveError) as caught:
            validate_autosave(selected)
        self.assertIn("bytes", str(caught.exception))

        # Same size, different content: the digest still catches it.
        checkpoint = _write_autosave(
            self.output / "run2", iterations=20, namespace=self.output,
            volume_uuid=self.volume, payload=b"payload")
        checkpoint.write_bytes(_zip_checkpoint_bytes(b"payloaX"))
        with self.assertRaises(AutosaveError) as caught:
            validate_autosave(self._select().selected)
        self.assertIn("digest", str(caught.exception))

    def test_a_valid_autosave_passes_identity_validation(self):
        _write_autosave(self.output / "run", iterations=7,
                        namespace=self.output, volume_uuid=self.volume)
        validate_autosave(self._select().selected)


class ExplicitInitializationTests(HttpServiceFixture):
    """The service is useful before a client explicitly creates a fit."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.volume = VOLUME
        self.output = base / "output" / "session-a"
        self.output.mkdir(parents=True)
        self.cache = base / "cache"
        self.state = ServiceState(
            bindings=spiral_service.bind_service_inputs(self.output, self.cache),
            startup_run={"z_begin": 0, "z_end": 10,
                         "config": _NO_DENSE_LOSSES})
        SpiralServer.allow_reuse_address = False
        self.server = SpiralServer(("127.0.0.1", 0), ["secret-key"], self.state)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.state.lease().claim("test-owner", "claim")

    def request(self, method, path, **kwargs):
        kwargs['headers'] = {'X-Spiral-Session-Token': 'test-owner', **kwargs.get('headers', {})}
        return super().request(method, path, **kwargs)

    def tearDown(self):
        self.state.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        self.temporary.cleanup()

    def _assert_service_answers(self, expected_state):
        status, payload, _ = self.request("GET", "/session/status")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["state"], expected_state)
        for path in ("/health", "/bindings", "/configuration"):
            status, payload, _ = self.request("GET", path)
            self.assertEqual(status, 200, path)
            self.assertEqual(json.loads(payload)["api_version"], API_VERSION)
        status, payload, _ = self.request("GET", "/events")
        self.assertEqual(status, 200)
        self.assertIn("events", json.loads(payload))
        # Checkpoint upload and the recovery command stay reachable too.
        status, _, _ = self.request(
            "POST", "/session/inputs",
            body={"kind": "checkpoint", "id": "resume.ckpt",
                  "files": [{"name": "resume.ckpt", "size": 1,
                             "sha256": "0" * 64}]})
        self.assertNotIn(status, (404, 405, 500))

    def test_every_read_endpoint_answers_before_initialization(self):
        with mock.patch("spiral_runtime.create_session") as create:
            self._assert_service_answers(SessionState.Uninitialized)
        create.assert_not_called()
        status = self.state.status()
        self.assertIsNone(status["session_id"])
        self.assertIsNone(status["session_request"])
        health = json.loads(self.request("GET", "/health")[1])
        self.assertIs(health["cuda_ready"], False)

    def test_every_read_endpoint_answers_while_initialization_is_loading(self):
        blocked = threading.Event()
        release = threading.Event()

        def slow_create(*_args, **_kwargs):
            blocked.set()
            release.wait(10)
            return FakeSession()

        with mock.patch("spiral_runtime.create_session", slow_create):
            self.state.initialize(with_project(self.state.startup_session_request()))
            self.assertTrue(blocked.wait(10))
            # The HTTP surface is fully responsive while CUDA and the model
            # are being built.
            self._assert_service_answers(SessionState.Loading)
            health = json.loads(self.request("GET", "/health")[1])
            self.assertIsNone(health["cuda_ready"])
            release.set()
            _await_build(self.state)
        self.assertEqual(self.state.status()["state"], SessionState.Idle)

    def test_every_read_endpoint_answers_while_the_session_is_in_error(self):
        with mock.patch("spiral_runtime.create_session",
                        side_effect=RuntimeError("no CUDA device")):
            self.state.initialize(with_project(self.state.startup_session_request()))
            status = _await_build(self.state)
        self.assertEqual(status["state"], SessionState.Error)
        self.assertIn("no CUDA device", status["error"])
        self._assert_service_answers(SessionState.Error)
        health = json.loads(self.request("GET", "/health")[1])
        self.assertIs(health["cuda_ready"], False)

    def test_initialization_does_not_select_an_autosave(self):
        _write_autosave(self.output / "old-run", iterations=5,
                        namespace=self.output, volume_uuid=self.volume)
        _write_autosave(self.output / "new-run", iterations=500,
                        namespace=self.output, volume_uuid=self.volume)
        with mock.patch("spiral_runtime.create_session",
                        return_value=FakeSession()) as create:
            self.state.initialize(with_project(self.state.startup_session_request()))
            _await_build(self.state)
        self.assertEqual(create.call_args.args[0].checkpoint, "")
        self.assertNotIn("autosave_selection", self.state.status())

    def test_a_corrupt_autosave_is_ignored_unless_selected(self):
        _write_autosave(self.output / "run", iterations=50,
                        namespace=self.output, volume_uuid=self.volume,
                        corrupt=True)
        with mock.patch("spiral_runtime.create_session",
                        return_value=FakeSession()) as create:
            self.state.initialize(with_project(self.state.startup_session_request()))
            status = _await_build(self.state)
        self.assertEqual(status["state"], SessionState.Idle)
        self.assertEqual(create.call_count, 1)
        self.assertEqual(create.call_args.args[0].checkpoint, "")

    def test_initialize_route_creates_the_first_session_once(self):
        with mock.patch("spiral_runtime.create_session",
                        return_value=FakeSession()):
            status, payload, _ = self.request(
                "POST", "/session/initialize",
                body={"command_id": "initialize-1",
                      **with_project(self.state.startup_session_request())})
            self.assertEqual(status, 200, payload)
            _await_build(self.state)
        status = json.loads(self.request("GET", "/session/status")[1])
        self.assertEqual(status["state"], SessionState.Idle)
        self.assertTrue(status["session_id"])
        duplicate, payload, _ = self.request(
            "POST", "/session/initialize",
            body={"command_id": "initialize-2"})
        self.assertEqual(duplicate, 409, payload)

    def test_initialize_route_rejects_non_mapping_advanced_config(self):
        status, payload, _ = self.request(
            "POST", "/session/initialize",
            body={"command_id": "initialize-malformed",
                  "run": {"z_begin": 0, "z_end": 10, "config": 1}})
        self.assertEqual(status, 400, payload)
        self.assertIn("Malformed session request",
                      json.loads(payload)["error"])

    def test_rebuild_before_initialization_is_rejected(self):
        status, payload, _ = self.request(
            "POST", "/session/rebuild",
            body={"command_id": "rebuild-1"})
        self.assertEqual(status, 409, payload)
        self.assertIn("initialize it first", json.loads(payload)["error"])

    def test_only_a_rebuild_may_replace_the_model_domain(self):
        with mock.patch("spiral_runtime.create_session",
                        return_value=FakeSession()):
            self.state.initialize(with_project(self.state.startup_session_request()))
            _await_build(self.state)

            # A run that changes a new-fit configuration key is refused: the
            # resident model keeps its domain.
            configuration = {
                **Config.catalog()["defaults"],
                **_NO_DENSE_LOSSES,
                "model_num_flow_stages": 3,
            }
            with self.assertRaisesRegex(ApiError, "requires rebuilding"):
                self.state.run({
                    "configuration": configuration,
                    "iterations": 3,
                    "dt_loss_schedule": {
                        "enabled": False, "last_fraction": 0.25},
                    "expected_session_revision": self.state.session_revision,
                })

            # The same change through a rebuild is accepted; it is the one
            # path that tears the model down and builds a new domain.
            self.state.rebuild(with_project({"run": {"z_begin": 0, "z_end": 10,
                                        "config": configuration}}))
            _await_build(self.state)
        applied = self.state.status()["session_request"]["run"]["config"]
        self.assertEqual(applied["model_num_flow_stages"], 3)


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.state = ServiceState()

    def tearDown(self):
        self.state.close()
        self.temporary.cleanup()

    def _session(self):
        # A bound service gates mutations on the session lease.
        self.state.bindings = spiral_service.bind_service_inputs(
            self.output, self.root / "cache")
        session = _attach_fake_session(self.state, self.output, VOLUME)
        self.state.lease().claim('owner', 'claim')
        return session

    def test_fit_inputs_are_never_uploaded(self):
        # Patches, PCLs and fibers come from the bound object store; an
        # upload is never how they change, with or without a session.
        for session in (False, True):
            if session:
                self._session()
            for kind in ("patch", "fiber", "pcl"):
                with self.subTest(kind=kind, session=session), \
                        self.assertRaises(ApiError) as caught:
                    self.state.begin_upload({
                        "upload_id": spiral_service.secrets.token_hex(16),
                        "kind": kind, "id": "p1",
                        "files": [{"name": "meta.json", "size": 1,
                                   "sha256": "0" * 64}]})
                self.assertEqual(caught.exception.status, 400)
                self.assertEqual(caught.exception.payload["code"],
                                 "inputs_in_object_store")

    def test_run_requires_validates_and_propagates_dt_loss_schedule(self):
        session = self._session()
        for schedule in (
                {"enabled": False, "last_fraction": 0.25},
                {"enabled": True, "last_fraction": 0},
                {"enabled": True, "last_fraction": 0.25},
                {"enabled": True, "last_fraction": 1}):
            _planned_run(self.state, {
                "iterations": 10, "dt_loss_schedule": schedule})
            self.assertEqual(session.dt_loss_schedules[-1], {
                "enabled": schedule["enabled"],
                "last_fraction": float(schedule["last_fraction"]),
            })

        base = {
            "configuration": dict(Config.catalog()["defaults"]),
            "iterations": 10,
            "expected_session_revision": self.state.session_revision,
        }
        invalid = (
            None,
            {},
            {"enabled": True},
            {"enabled": True, "last_fraction": 0.25, "extra": 1},
            {"enabled": 1, "last_fraction": 0.25},
            {"enabled": True, "last_fraction": True},
            {"enabled": True, "last_fraction": "0.25"},
            {"enabled": True, "last_fraction": -0.01},
            {"enabled": True, "last_fraction": 1.01},
            {"enabled": True, "last_fraction": float("inf")},
            {"enabled": True, "last_fraction": float("nan")},
        )
        for schedule in invalid:
            request = dict(base)
            if schedule is not None:
                request["dt_loss_schedule"] = schedule
            with self.subTest(schedule=schedule), self.assertRaises(ApiError) as caught:
                self.state.run(request)
            self.assertEqual(caught.exception.status, 400)

    def test_run_configuration_ignores_and_reports_unknown_fields(self):
        session = self._session()
        configuration = dict(Config.catalog()["defaults"])
        configuration["influence_disable_dt_frac"] = 0.75
        configuration["future_unknown_setting"] = 7
        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            result = self.state.run({
                "configuration": configuration,
                "iterations": 10,
                "dt_loss_schedule": {
                    "enabled": False, "last_fraction": 0.25},
                "expected_session_revision": self.state.session_revision,
            })
        self.assertTrue(result["accepted"])
        self.assertNotIn("influence_disable_dt_frac", session.run_calls[-1][1])
        self.assertNotIn("future_unknown_setting", session.run_calls[-1][1])
        warning = ("Ignoring unknown run configuration keys: "
                   "['future_unknown_setting', 'influence_disable_dt_frac']")
        self.assertIn(warning, output.getvalue())
        self.assertTrue(any(
            event["text"] == warning and event["severity"] == "warning"
            for event in self.state.events.read_after(0)["events"]))

        # Extra obsolete fields must not conceal missing current fields.
        del configuration["optimizer_learning_rate"]
        with self.assertRaisesRegex(ApiError, "complete configuration"):
            self.state.run({
                "configuration": configuration,
                "expected_session_revision": self.state.session_revision,
            })

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
        }

        response = _planned_run(self.state, {"iterations": 10, "run_config": config})

        self.assertEqual(session.run_calls[-1][1], config)
        self.assertEqual(response["run_config"]["sample_count_patches_per_step"], 240)

        with self.assertRaisesRegex(ApiError, "requires rebuilding"):
            _planned_run(self.state, {"iterations": 10, "run_config": {
                "model_num_flow_stages": 3,
            }})
        with self.assertRaisesRegex(ApiError, "Invalid value"):
            _planned_run(self.state, {"iterations": 10, "run_config": {
                "output_save_png_visualizations": 1,
            }})
        with self.assertRaisesRegex(ApiError, "vector length"):
            _planned_run(self.state, {"iterations": 10, "run_config": {
                "track_length_bin_weights": [1, 2],
            }})

    def test_run_accepts_advertised_zero_count_for_disabled_input(self):
        session = self._session()
        session.run_config["sample_count_fiber_direction_points"] = 0

        response = _planned_run(self.state, {"iterations": 10, "run_config": {
            "sample_count_fiber_direction_points": 0,
        }})

        self.assertEqual(session.run_calls[-1][1], {
            "sample_count_fiber_direction_points": 0,
        })
        self.assertEqual(response["run_config"]["sample_count_fiber_direction_points"], 0)

    def test_outer_shell_path_change_requires_session_reload(self):
        self._session()
        shell = self.root / "outer_shell_v2"
        shell.mkdir()
        inputs = self.state.session_paths.manifest()
        inputs["outer_shell"] = str(shell)
        with self.assertRaisesRegex(ApiError, "Static fit inputs") as caught:
            self.state.run({
                "configuration": dict(Config.catalog()["defaults"]),
                "iterations": 3,
                "dt_loss_schedule": {
                    "enabled": False, "last_fraction": 0.25},
                "inputs": inputs,
                "expected_session_revision": self.state.session_revision,
            })
        self.assertEqual(caught.exception.status, 409)

    def test_any_other_static_path_change_is_also_rejected(self):
        self._session()
        inputs = self.state.session_paths.manifest()
        inputs["verified_patches"] = str(self.root / "other-patches")
        with self.assertRaisesRegex(ApiError, "Static fit inputs") as caught:
            self.state.run({
                "configuration": dict(Config.catalog()["defaults"]),
                "iterations": 3,
                "dt_loss_schedule": {
                    "enabled": False, "last_fraction": 0.25},
                "inputs": inputs,
                "expected_session_revision": self.state.session_revision,
            })
        self.assertEqual(caught.exception.status, 409)


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
        base = Path(self.temporary.name)
        self.output = base / "output"
        self.output.mkdir()
        self.bindings = spiral_service.bind_service_inputs(self.output, base / "cache")
        self.state = ServiceState(bindings=self.bindings,
                                  startup_run={"z_begin": 0, "z_end": 10})

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

    def test_checkpoint_upload_works_without_a_session(self):
        record = self._upload_checkpoint("resume.ckpt")
        published = Path(record["path"])
        self.assertTrue(published.is_file())
        self.assertIn("uploaded-checkpoints", str(published))
        # The published path lies under the output directory, so the
        # bound load validation accepts it as a resume checkpoint.
        request = self.state._bound_session_request(
            {"paths": {"checkpoint": str(published)},
             "run": {"z_begin": 0, "z_end": 10}}, _PROJECT)
        self.assertEqual(request["paths"]["checkpoint"], str(published))
        # Checkpoint uploads are not session inputs: nothing ephemeral listed.
        self.assertNotIn("ephemeral_inputs", self.state.status())

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
        self.assertFalse(any((self.output / "uploaded-checkpoints").glob("*"))
                         if (self.output / "uploaded-checkpoints").exists()
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
        restarted = ServiceState(bindings=self.bindings,
                                 startup_run={"z_begin": 0, "z_end": 10})
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
        root = self.output / "uploaded-checkpoints"
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
        root = self.output / "uploaded-checkpoints"
        self.assertEqual([Path(first["path"])], list(root.iterdir()))

    def test_reusing_a_checkpoint_refreshes_retention_recency(self):
        payloads = [str(i).encode() for i in range(
            UPLOADED_CHECKPOINTS_KEPT)]
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
                     for i in range(UPLOADED_CHECKPOINTS_KEPT + 2)]
        for old_age, path in enumerate(published):
            if path.exists():
                # Ensure distinguishable mtimes for deterministic pruning.
                os.utime(path, (time.time() + old_age, time.time() + old_age))
        surviving = [path for path in published if path.exists()]
        self.assertLessEqual(len(surviving), UPLOADED_CHECKPOINTS_KEPT)
        self.assertTrue(published[-1].exists(), "the newest upload must survive")


class MappedPreviewArtifactTests(unittest.TestCase):
    @staticmethod
    def _wait_finished(state, generation=1):
        deadline = time.monotonic() + 5.0
        while state._preview.completed_generation < generation \
                and time.monotonic() < deadline:
            time.sleep(0.005)
        if state._preview.completed_generation < generation:
            raise AssertionError("background preview publication did not finish")

    def test_lasagna_output_scale_must_match_requested_step(self):
        self.assertEqual(
            _validate_tifxyz_output_step(
                {"scale": [0.05, 0.05]}, 20.0),
            [0.05, 0.05])
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _validate_tifxyz_output_step(
                {"scale": [0.04, 0.04]}, 20.0)

    def test_every_generation_is_flattened_from_scratch(self):
        profile = {
            "args": {"model-init": "flatten"},
            "stages": [{"name": "cold", "steps": 4500, "lr": 0.01},
                       {"name": "cold-refine", "steps": 1000, "lr": 0.001}],
        }
        body = _flatten_request_body(
            profile, {"kind": "surface"}, surface_id="s-lasagna.tifxyz",
            publish_root=Path("/publish"), model_output=Path("/publish/m.pt"))
        # The request carries no warm-start state from an earlier generation
        # and leaves the configured profile's schedule untouched.
        self.assertNotIn("export_flatten_uv", body)
        self.assertNotIn("flatten_initial_uv", body["config"]["args"])
        self.assertEqual(body["config"]["stages"], profile["stages"])
        self.assertEqual(body["job_spec"]["config"], body["config"])
        self.assertEqual(body["config"]["external_surfaces"],
                         [{"kind": "surface"}])
        self.assertNotIn("external_surfaces", profile)
        self.assertTrue(body["export_flatten_map"])

    def test_active_publication_coalesces_to_only_the_newest_pending_raw(self):
        publication = PreviewPublication()
        self.assertTrue(publication.claim(
            "session", 1, manifest="/raw/1/manifest.json",
            source_fit_iteration=100))
        self.assertFalse(publication.claim(
            "session", 2, manifest="/raw/2/manifest.json",
            source_fit_iteration=200))
        self.assertFalse(publication.claim(
            "session", 3, manifest="/raw/3/manifest.json",
            source_fit_iteration=300))
        self.assertEqual(publication.pending_generation, 3)
        publication.finish(1)
        pending = publication.take_pending()
        self.assertEqual(pending["preview_generation"], 3)
        self.assertEqual(pending["current_iteration"], 300)

    def test_published_active_and_pending_iterations_are_not_conflated(self):
        publication = PreviewPublication()
        publication.source_fit_iteration = 100
        self.assertTrue(publication.claim(
            "session", 2, manifest="/raw/2/manifest.json",
            source_fit_iteration=200))
        self.assertFalse(publication.claim(
            "session", 3, manifest="/raw/3/manifest.json",
            source_fit_iteration=300))

        self.assertEqual(publication.source_fit_iteration, 100)
        self.assertEqual(publication.active_source_fit_iteration, 200)
        self.assertEqual(publication.pending_source_fit_iteration, 300)

    def test_session_reset_invalidates_every_retained_raw_generation(self):
        publication = PreviewPublication()
        publication.previous_raw_manifest = "/raw/previous/manifest.json"
        self.assertTrue(publication.claim(
            "session", 2, manifest="/raw/active/manifest.json"))
        self.assertFalse(publication.claim(
            "session", 3, manifest="/raw/pending/manifest.json"))

        self.assertEqual(set(publication.reset_session_scope()), {
            "/raw/previous/manifest.json",
            "/raw/active/manifest.json",
            "/raw/pending/manifest.json",
        })
        self.assertEqual(publication.generation, 0)
        self.assertIsNone(publication.manifest)
        self.assertEqual(publication.pending_generation, 0)

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
            state._preview.previous_raw_manifest = str(previous_manifest)
            state._preview.artifact = {"id": "previous-preview"}

            with mock.patch.object(
                    state, "_publish_flattened_preview",
                    side_effect=RuntimeError("flatten failed")):
                state._maybe_register_artifacts({
                    "preview_generation": 1,
                    "preview_manifest_path": str(current_manifest),
                })
                self._wait_finished(state)

            self.assertEqual(
                state._preview.artifact, {"id": "previous-preview"})
            self.assertTrue(previous.exists())
            self.assertFalse(current.exists())
            self.assertIn("flatten failed", state._preview.error)
            # One record holds the whole outcome: the failed generation is
            # finished (never retried), nothing is in flight, and the
            # previous successful raw generation is still the overlay base.
            self.assertEqual(state._preview.generation, 0)
            self.assertEqual(state._preview.completed_generation, 1)
            self.assertEqual(state._preview.previous_raw_manifest,
                             str(previous_manifest))

    def _published(self, root, generation=1):
        """A finished surface wave, as the publisher hands one over."""
        surface = root / "published"
        surface.mkdir()
        (surface / "manifest.json").write_text("{}")
        return PublishedPreview(
            manifest_path=surface / "manifest.json",
            surface_id="surface-1", generation=generation,
            raw_manifest={}, raw_manifest_path=root / "raw" / "manifest.json",
            publish_parent=root, correspondence=None, flattened_valid=None)

    def test_surface_is_announced_before_the_diagnostics_wave_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raw").mkdir()
            (root / "raw" / "manifest.json").write_text("{}")
            diagnostics = root / "diagnostics"
            diagnostics.mkdir()
            (diagnostics / "manifest.json").write_text("{}")
            state = ServiceState()
            state.session_id = "session"
            announced = []
            publisher = mock.Mock()
            # The surface must already be announced by the time the overlays
            # are asked for; that is the whole point of the second wave.
            publisher.publish_diagnostics.side_effect = (
                lambda published: (
                    announced.append(dict(state._preview.artifact or {})),
                    diagnostics / "manifest.json")[1])

            with mock.patch.object(
                    state, "_publish_flattened_preview",
                    return_value=(publisher, self._published(root))):
                state._maybe_register_artifacts({
                    "preview_generation": 1,
                    "preview_manifest_path": str(root / "raw" / "manifest.json"),
                    "preview_diagnostics": True,
                })
                self._wait_finished(state)

            self.assertEqual(len(announced), 1)
            self.assertEqual(announced[0].get("kind"), "spiral-preview")
            self.assertEqual(state._preview.artifact["kind"], "spiral-preview")
            self.assertEqual(state._preview.diagnostics_artifact["kind"],
                             "spiral-preview-diagnostics")
            self.assertIsNone(state._preview.error)
            self.assertIn("preview_diagnostics_artifact", state.status())

    def test_a_preview_without_diagnostics_publishes_no_second_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raw").mkdir()
            (root / "raw" / "manifest.json").write_text("{}")
            state = ServiceState()
            state.session_id = "session"
            publisher = mock.Mock()

            with mock.patch.object(
                    state, "_publish_flattened_preview",
                    return_value=(publisher, self._published(root))):
                state._maybe_register_artifacts({
                    "preview_generation": 1,
                    "preview_manifest_path": str(root / "raw" / "manifest.json"),
                })
                self._wait_finished(state)

            publisher.publish_diagnostics.assert_not_called()
            self.assertEqual(state._preview.artifact["kind"], "spiral-preview")
            self.assertIsNone(state._preview.diagnostics_artifact)

    def test_failed_overlays_do_not_fail_the_published_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raw").mkdir()
            (root / "raw" / "manifest.json").write_text("{}")
            state = ServiceState()
            state.session_id = "session"
            publisher = mock.Mock()
            publisher.publish_diagnostics.side_effect = RuntimeError(
                "overlay remap failed")

            with mock.patch.object(
                    state, "_publish_flattened_preview",
                    return_value=(publisher, self._published(root))):
                state._maybe_register_artifacts({
                    "preview_generation": 1,
                    "preview_manifest_path": str(root / "raw" / "manifest.json"),
                    "preview_diagnostics": True,
                })
                self._wait_finished(state)

            self.assertEqual(state._preview.artifact["kind"], "spiral-preview")
            self.assertIsNone(state._preview.diagnostics_artifact)
            self.assertIsNone(state._preview.error)
            # The raw generation is the next run-difference base; a failed
            # overlay must not discard it.
            self.assertTrue((root / "raw").exists())

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

    def test_threaded_loss_overlay_matches_sequential_bytes(self):
        rng = np.random.default_rng(20260730)
        source = rng.integers(0, 256, size=(17, 23, 4), dtype=np.uint8)
        source_yx = np.stack([
            rng.uniform(-1.0, 17.5, size=(13, 19)),
            rng.uniform(-1.0, 23.5, size=(13, 19)),
        ], axis=-1).astype(np.float32)
        output_valid = rng.random((13, 19)) > 0.2

        sequential = _sample_rgba_through_map(
            source, source_yx, output_valid)
        with ThreadPoolExecutor(max_workers=4) as executor:
            threaded = _sample_rgba_through_map(
                source, source_yx, output_valid, executor=executor)

        np.testing.assert_array_equal(threaded, sequential)

    def test_winding_bounds_union_duplicate_disjoint_ranges(self):
        manifest = {
            "winding_ids": [7, 9, 7],
            "winding_column_ranges": [[0, 2], [2, 4], [4, 6]],
        }
        source_yx = np.asarray([[
            [0, 0], [0, 2], [0, 5], [0, 3],
        ]], dtype=np.float32)
        result, bounds = _mapped_winding_ids(
            manifest, (1, 6), source_yx, np.ones((1, 4), dtype=bool))

        np.testing.assert_array_equal(result, [[7, 9, 7, 9]])
        self.assertEqual(bounds, [
            {
                "winding": 7, "row_begin": 0, "row_end": 1,
                "column_begin": 0, "column_end": 3,
            },
            {
                "winding": 9, "row_begin": 0, "row_end": 1,
                "column_begin": 1, "column_end": 4,
            },
        ])

    def test_flatten_correspondence_prefers_npy_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            map_path = Path(temporary) / "flatten-map.npy"
            expected = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
            np.save(map_path, expected, allow_pickle=False)
            actual = _load_flatten_correspondence(
                Path(temporary) / "missing.pt", map_path)
            np.testing.assert_array_equal(actual, expected)

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

    def _launch(self, arguments, temporary, env=None):
        script = Path(__file__).resolve().parents[1] / "spiral_service.py"
        merged_env = None
        if env is not None:
            merged_env = dict(os.environ)
            merged_env.update(env)
        return subprocess.Popen(
            [sys.executable, str(script)] + arguments,
            cwd=str(script.parent), env=merged_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    @staticmethod
    def _make_output(temporary):
        output = Path(temporary) / "output"
        output.mkdir(exist_ok=True)
        return output

    def _launch_arguments(self, temporary):
        return ["--output", str(self._make_output(temporary)),
                "--object-store", self._make_store(temporary)]

    @staticmethod
    def _make_store(temporary):
        """An empty object store: the launch binding every service needs."""
        try:
            from vc_object_store import ObjectStore
        except ImportError:
            raise unittest.SkipTest("the service launch needs vc_object_store")
        path = Path(temporary) / "inputs.vcobj"
        if not path.exists():
            with ObjectStore.create(str(path), author="test", hostname="host"):
                pass
        return str(path)

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
            process = self._launch(["--port", "0", "--api-key-file", str(key_file)]
                                   + self._launch_arguments(temporary),
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
                    f"http://127.0.0.1:{port}/events?cursor=0",
                    headers={"Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    events = json.loads(response.read())
                self.assertIn(ready, [record["text"]
                                      for record in events["events"]
                                      if record["kind"] == "log"])
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
            ] + self._launch_arguments(temporary), temporary)
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
            arguments = self._launch_arguments(temporary)
            process = self._launch(["--port", "0", "--api-key-file", str(key_file)]
                                   + arguments, temporary)
            try:
                ready = [line for line in self._read_until_ready(process)
                         if line.startswith("SPIRAL_SERVICE_READY")][0]
                port = int(ready.split("port=")[1].split()[0])
            finally:
                process.kill()
                process.wait(10)
            # Restart on the same, now-explicit port straight away.
            process = self._launch(["--port", str(port),
                                    "--api-key-file", str(key_file)]
                                   + arguments, temporary)
            try:
                self._read_until_ready(process)
            finally:
                process.terminate()
                process.wait(10)

    def test_output_and_object_store_are_required_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            process = self._launch(["--port", "0"], temporary)
            output, _ = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 2)
            self.assertIn("--output", output)
            self.assertIn("--object-store", output)
            self.assertNotIn("--dataset", output)

    def test_a_missing_winding_inference_binding_refuses_to_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "winding_inference"
            process = self._launch(["--port", "0", "--winding-inference", str(missing)]
                                   + self._launch_arguments(temporary), temporary)
            output, _ = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 2)
            self.assertIn(f"--winding-inference {missing} is not a directory", output)

    def test_bindings_advertise_bound_output_default_cache_and_winding(self):
        with tempfile.TemporaryDirectory() as temporary:
            key_file = Path(temporary) / "key"
            output_root = self._make_output(temporary)
            winding = Path(temporary) / "winding_inference"
            winding.mkdir()
            (winding / "manifest.json").write_text(json.dumps({
                "artifact_type": "winding_inference_crossings",
                "format_version": 1}))
            xdg_cache = Path(temporary) / "xdg-cache"
            process = self._launch(
                ["--port", "0", "--api-key-file", str(key_file),
                 "--output", str(output_root),
                 "--object-store", self._make_store(temporary),
                 "--winding-inference", str(winding)],
                temporary, env={"XDG_CACHE_HOME": str(xdg_cache)})
            try:
                lines = self._read_until_ready(process)
                self.assertIn(f"Spiral winding inference: {winding.resolve()}", lines)
                ready = next(line for line in lines
                             if line.startswith("SPIRAL_SERVICE_READY"))
                port = int(ready.split("port=")[1].split()[0])
                key = key_file.read_text().strip()
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/bindings",
                    headers={"Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    advertised = json.loads(response.read())
                bindings = advertised["bindings"]
                self.assertEqual(bindings["output_directory"],
                                 str(output_root.resolve()))
                self.assertEqual(
                    bindings["cache_directory"],
                    str(xdg_cache.resolve() / "vc3d" / "spiral"))
                self.assertEqual(bindings["winding_inference"], {
                    "bound": True, "path": str(winding.resolve()),
                    "available": True, "error": None})
                self.assertTrue(advertised["capabilities"]["winding_inference"])
                # The dataset endpoint is gone.
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/dataset",
                    headers={"Authorization": f"Bearer {key}"})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request, timeout=10)
                self.assertEqual(caught.exception.code, 404)
                caught.exception.close()
            finally:
                process.terminate()
                process.wait(10)
                process.stdout.close()

    def test_named_service_has_exclusive_restartable_ownership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = self._make_output(temporary)
            key_file = root / "key"
            arguments = [
                "--port", "0", "--api-key-file", str(key_file),
                "--output", str(output_root),
                "--object-store", self._make_store(temporary),
                "--session-name", "alice",
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
                # The exclusive lease lives under the named output namespace.
                self.assertTrue(
                    (output_root / "alice" / ".spiral-service.lock").is_file())

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


if __name__ == "__main__":
    unittest.main()
