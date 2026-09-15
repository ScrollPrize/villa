
"""Tests for the Spiral service protocol.

Covers bearer authentication, API-key auto-generation, launch-time dataset
ownership, the artifact registry and its HTTP endpoints, session input
uploads with immutable workspace staging, and checkpoint transfers. The resident fitter is
faked; these tests exercise the service plumbing only.
"""

from service_fixtures import (
    FakeSession,
    _NO_DENSE_LOSSES,
    _await_build,
    _write_scroll_spec,
    _attach_fake_session,
    HttpServiceFixture,
)
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import hashlib
import json
import os
from pathlib import Path
import shutil
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
import spiral_service
from spiral_service import (
    ApiError,
    ServiceState,
    SpiralServer,
    _mapped_winding_ids,
    _prepare_cleaned_lasagna_surface,
    _raw_run_diff_rgba,
    _sample_rgba_through_map,
    parse_gpu_ids,
    parse_session_name,
)
from service_uploads import UPLOADED_CHECKPOINTS_KEPT
from lasagna_publish import PreviewPublication, PublishedPreview
from fit_session import (
    API_VERSION,
    AUTOSAVE_CHECKPOINT_NAME,
    AutosaveError,
    PclRole,
    SessionState,
    SpiralInputPaths,
    resolve_dataset_root,
    select_startup_autosave,
    validate_autosave,
    write_autosave_metadata,
)
from config import Config, durable_config


def _planned_run(state, request):
    request = dict(request)
    configuration = dict(Config.catalog()["defaults"])
    configuration.update(request.pop("run_config", {}))
    return state.run({
        "configuration": configuration,
        "iterations": request.pop("iterations"),
        "influence": request.pop("influence_config", {}),
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


class GpuSelectionTests(unittest.TestCase):
    def test_gpu_selection_preserves_order_and_rejects_invalid_lists(self):
        self.assertEqual(parse_gpu_ids(" 3, 1 "), (3, 1))
        for value in ("", "0,", "-1", "gpu0", "1,1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_gpu_ids(value)


class NamedSessionTests(unittest.TestCase):
    def test_unsafe_session_names_are_rejected(self):
        for value in ("", ".", "..", "../alice", "alice/bob", "a b", "_alice",
                      "a" * 65):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_session_name(value)


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


class ArtifactHttpTests(HttpServiceFixture):
    def _register_artifact(self, contents=b"0123456789abcdef"):
        artifact_root = self.root / "artifact"
        artifact_root.mkdir()
        (artifact_root / "manifest.json").write_text("{}")
        (artifact_root / "payload.bin").write_bytes(contents)
        return self.state.artifacts.register_directory(
            "spiral-preview", "session-1", 1, artifact_root, "manifest.json",
            delete_root_on_prune=True), artifact_root

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
        base = Path(self.temporary.name)
        self.root = base / "dataset"
        self.root.mkdir()
        _write_scroll_spec(self.root)
        (self.root / "umbilicus.json").write_text("{}")
        (self.root / "verified_patches").mkdir()
        self.output = base / "output"
        self.output.mkdir()
        self.cache = base / "cache"
        self.resolution = spiral_service.bind_service_paths(
            resolve_dataset_root(self.root), self.output, self.cache)
        self.assertTrue(self.resolution.ok)
        self.state = ServiceState(dataset_root=str(self.root),
                                  dataset_resolution=self.resolution,
                                  startup_run={"z_begin": 0, "z_end": 10,
                                               "config": _NO_DENSE_LOSSES})

    def tearDown(self):
        self.temporary.cleanup()

    def test_initialize_rejects_client_base_input_paths(self):
        with self.assertRaises(ApiError) as caught:
            self.state.initialize({"paths": {"umbilicus": "/attacker/umbilicus.json"},
                                   "run": {"z_begin": 0, "z_end": 10}})
        self.assertEqual(caught.exception.status, 400)
        fields = {detail["field"] for detail in caught.exception.details}
        self.assertEqual(fields, {"umbilicus"})

    def _attach_session_for_request(self, config=None):
        """Attach a fake session whose canonical request the service produced."""
        session = _attach_fake_session(self.state, self.output, self.root)
        request = {"run": {"z_begin": 0, "z_end": 10,
                           "config": {**_NO_DENSE_LOSSES, **(config or {})}}}
        paths, run, preview, _ = self.state._prepare_session_request(request)
        self.state.session_paths = paths
        self.state.session_request = {
            "paths": paths.manifest(), "run": run.manifest(),
            "preview": preview.manifest()}
        return session

    def _stage_for(self, request):
        paths, run, preview, _ = self.state._prepare_session_request(request)
        with self.state.lock:
            return self.state._rebuild_stage_locked(
                paths, run, preview, SessionState.Idle)

    def _base_request(self, config=None, **run):
        return {"run": {"z_begin": 0, "z_end": 10,
                        "config": {**_NO_DENSE_LOSSES, **(config or {})},
                        **run}}

    def test_shell_atlas_settings_rebuild_everything_once_tracks_were_shell_filtered(self):
        # Without a shell-filtered track pool the atlas settings are ordinary
        # run-boundary knobs the model rebuild applies through apply_config.
        self._attach_session_for_request()
        self.assertEqual(
            self._stage_for(self._base_request({"shell_num_theta_bins": 360})),
            "model")
        # A session that loaded both a tracks store and an outer shell
        # filtered the tracks against the shell; apply_config refuses the
        # atlas settings there, so the request has to rebuild the host inputs.
        live = SpiralInputPaths.from_mapping({
            **self.state.session_paths.manifest(),
            "tracks_dbm": str(self.root / "tracks.dbm"),
            "outer_shell": str(self.root / "outer_shell"),
        })
        self.state.session_paths = live
        self.state.session_request["paths"] = live.manifest()
        run = self._base_request({"shell_num_theta_bins": 360})["run"]
        _, run_config, preview, _ = self.state._prepare_session_request(
            {"run": run})
        with self.state.lock:
            self.assertEqual(
                self.state._rebuild_stage_locked(
                    live, run_config, preview, SessionState.Idle),
                "all")
            # Other shell loss settings still keep the loaded inputs.
            _, other_run, _, _ = self.state._prepare_session_request(
                {"run": self._base_request(
                    {"shell_huber_delta": 8.0})["run"]})
            self.assertEqual(
                self.state._rebuild_stage_locked(
                    live, other_run, preview, SessionState.Idle),
                "model")

    def test_a_model_stage_rebuild_keeps_the_session_and_its_inputs(self):
        session = self._attach_session_for_request()
        generation = self.state.session_generation
        response = self.state.rebuild(
            self._base_request({"model_linear_z_resolution": 24}))
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
                    "loss_weight_shell_patch_radius": 0,
                },
            },
        }
        with mock.patch("spiral_runtime.create_session",
                        side_effect=RuntimeError("startup failed")):
            self.state.initialize(request)
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
            self.state.rebuild({"defaults": True})
            status = _await_build(self.state)
        self.assertEqual(status["state"], SessionState.Idle)
        self.assertIsNone(status["error"])

    def test_save_checkpoint_names_a_file_the_service_places(self):
        """The client names the checkpoint; the service owns the location."""
        _attach_fake_session(self.state, self.output, self.root)
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
        session = _attach_fake_session(self.state, self.output, self.root)
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

    def test_a_refused_checkpoint_is_a_conflict_and_changes_nothing(self):
        session = _attach_fake_session(self.state, self.output, self.root)
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

    def test_export_preview_accepts_and_runs_off_the_request_thread(self):
        """A preview costs minutes; the verb accepts it and returns.

        Holding the request open outlived every client transfer timeout, so
        a preview that succeeded was reported as a failure. The client
        follows preview_exporting in the status it already polls.
        """
        session = _attach_fake_session(self.state, self.output, self.root)
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
        session = _attach_fake_session(self.state, self.output, self.root)
        session.preview_failure = "OSError: no space left on device"

        self.state.export_preview()
        deadline = time.monotonic() + 5.0
        while self.state.status()["preview_exporting"] \
                and time.monotonic() < deadline:
            time.sleep(0.01)
        status = self.state.status()
        self.assertFalse(status["preview_exporting"])
        self.assertIn("no space left", status["preview_publish_error"])

    def _write_checkpoint(self, name, cfg, dataset_root=None):
        """A checkpoint carrying the durable cfg a refusal is analysed from."""
        import torch

        path = self.output / name
        torch.save({
            # Checkpoints store the durable subset of the schema, and the
            # refusal analysis compares against exactly that subset.
            "schema_version": 2, "cfg": durable_config(cfg),
            "input_manifest": {"dataset_root": str(
                self.root if dataset_root is None else dataset_root)},
        }, path)
        return path

    def test_allow_rebuild_rebuilds_onto_the_checkpoint_without_overrides(self):
        session = _attach_fake_session(self.state, self.output, self.root)
        live = Config().as_dict()
        session.applied_config = dict(live)
        # A live session request carrying an advanced profile, as a client
        # that had been editing configuration would leave it.
        paths, run, preview, _ = self.state._prepare_session_request({
            "run": {"z_begin": 0, "z_end": 10,
                    "config": {**_NO_DENSE_LOSSES, "model_num_flow_stages": 9}}})
        self.state.session_paths = paths
        self.state.session_request = {
            "paths": paths.manifest(), "run": run.manifest(),
            "preview": preview.manifest()}
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

    def test_a_rebuild_refuses_overrides_its_checkpoint_contradicts(self):
        _attach_fake_session(self.state, self.output, self.root)
        live = Config().as_dict()
        checkpoint = self._write_checkpoint("resume.ckpt", dict(live))
        request = {
            "paths": {"checkpoint": str(checkpoint)},
            "run": {"z_begin": 0, "z_end": 10,
                    "config": {**_NO_DENSE_LOSSES,
                               "model_flow_bounds_radius": 128}},
        }
        with self.assertRaises(ApiError) as caught:
            self.state.rebuild(request)
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual([detail["field"] for detail in caught.exception.details],
                         ["run.config.model_flow_bounds_radius"])
        # Everything else stays a legitimate change to make while resuming.
        request["run"]["config"] = {**_NO_DENSE_LOSSES,
                                    "loss_weight_patch_radius": 1.0}
        self.state._reject_overrides_the_checkpoint_contradicts(
            str(checkpoint), request["run"]["config"])


def _write_autosave(directory, *, iterations, namespace, dataset_root,
                    payload=b"payload", corrupt=False):
    """Write one autosave plus the metadata that makes it selectable."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / AUTOSAVE_CHECKPOINT_NAME
    checkpoint.write_bytes(_zip_checkpoint_bytes(payload))
    write_autosave_metadata(checkpoint, session_namespace=namespace,
                            dataset_root=dataset_root,
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
        self.dataset = str(Path(self.temporary.name) / "dataset")

    def tearDown(self):
        self.temporary.cleanup()

    def _select(self):
        return select_startup_autosave(
            self.output, session_namespace=self.output,
            dataset_root=self.dataset)

    def test_the_furthest_autosave_wins_regardless_of_filename_order(self):
        # "aaa-run" sorts first and "zzz-run" sorts last; neither ordering
        # decides anything. The completed iteration count does.
        _write_autosave(self.output / "zzz-run", iterations=40,
                        namespace=self.output, dataset_root=self.dataset)
        winner = _write_autosave(self.output / "aaa-run", iterations=900,
                                 namespace=self.output,
                                 dataset_root=self.dataset)
        selection = self._select()
        self.assertEqual(selection.selected.checkpoint, str(winner))
        self.assertEqual(selection.selected.completed_iterations, 900)
        self.assertEqual(
            [reason for _, reason in selection.rejected],
            [f"superseded by {winner} at 900 iterations"])

    def test_a_selected_autosave_must_match_its_recorded_identity(self):
        _write_autosave(self.output / "run", iterations=10,
                        namespace=self.output, dataset_root=self.dataset,
                        corrupt=True)
        selected = self._select().selected
        self.assertIsNotNone(selected)
        with self.assertRaises(AutosaveError) as caught:
            validate_autosave(selected)
        self.assertIn("bytes", str(caught.exception))

        # Same size, different content: the digest still catches it.
        checkpoint = _write_autosave(
            self.output / "run2", iterations=20, namespace=self.output,
            dataset_root=self.dataset, payload=b"payload")
        checkpoint.write_bytes(_zip_checkpoint_bytes(b"payloaX"))
        with self.assertRaises(AutosaveError) as caught:
            validate_autosave(self._select().selected)
        self.assertIn("digest", str(caught.exception))


class ExplicitInitializationTests(HttpServiceFixture):
    """The service is useful before a client explicitly creates a fit."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.dataset = base / "dataset"
        self.dataset.mkdir()
        _write_scroll_spec(self.dataset)
        (self.dataset / "umbilicus.json").write_text("{}")
        (self.dataset / "verified_patches").mkdir()
        self.output = base / "output" / "session-a"
        self.output.mkdir(parents=True)
        self.cache = base / "cache"
        self.resolution = spiral_service.bind_service_paths(
            resolve_dataset_root(self.dataset), self.output, self.cache)
        self.state = ServiceState(
            dataset_root=str(self.dataset),
            dataset_resolution=self.resolution,
            startup_run={"z_begin": 0, "z_end": 10,
                         "config": _NO_DENSE_LOSSES})
        SpiralServer.allow_reuse_address = False
        self.server = SpiralServer(("127.0.0.1", 0), ["secret-key"], self.state)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.state.editing().claim("test-owner", "claim")

    def request(self, method, path, **kwargs):
        kwargs['headers'] = {'X-Spiral-Workspace-Token': 'test-owner', **kwargs.get('headers', {})}
        return super().request(method, path, **kwargs)

    def tearDown(self):
        self.state.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        self.temporary.cleanup()

    def test_initialize_route_creates_the_first_session_once(self):
        with mock.patch("spiral_runtime.create_session",
                        return_value=FakeSession()):
            status, payload, _ = self.request(
                "POST", "/session/initialize",
                body={"command_id": "initialize-1",
                      **self.state.startup_session_request()})
            self.assertEqual(status, 200, payload)
            _await_build(self.state)
        status = json.loads(self.request("GET", "/session/status")[1])
        self.assertEqual(status["state"], SessionState.Idle)
        self.assertTrue(status["session_id"])
        duplicate, payload, _ = self.request(
            "POST", "/session/initialize",
            body={"command_id": "initialize-2"})
        self.assertEqual(duplicate, 409, payload)


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
        self.state.close()
        self.temporary.cleanup()

    def _session(self):
        self.state.dataset_root = str(self.dataset)
        session = _attach_fake_session(self.state, self.output, self.dataset)
        self.state.editing().claim('owner', 'claim')
        return session

    def test_upload_put_retries_are_content_addressed(self):
        self._session()
        data = PATCH_FILES["meta.json"]
        upload_id = self.state.begin_upload({"upload_id": spiral_service.secrets.token_hex(16),
            "kind": "patch", "id": "retried", "files": [
                {"name": name, "size": len(payload),
                 "sha256": _digest(payload)}
                for name, payload in PATCH_FILES.items()],
        })["upload_id"]
        staging = self.state.editing().uploads.staging_root() / upload_id

        # An upload PUT carries no command ID: the manifest's size and digest
        # decide the outcome, so repeating one is safe and converges.
        for _ in range(3):
            response = self.state.receive_upload_file(
                upload_id, "meta.json", io.BytesIO(data), len(data))
            self.assertEqual(response["received"], "meta.json")
        self.assertEqual((staging / "meta.json").read_bytes(), data)

        # A retry that delivers different bytes is refused and never replaces
        # the staged copy that already matched the manifest.
        corrupted = b"!" * len(data)
        with self.assertRaisesRegex(ApiError, "SHA-256"):
            self.state.receive_upload_file(
                upload_id, "meta.json", io.BytesIO(corrupted), len(corrupted))
        self.assertEqual((staging / "meta.json").read_bytes(), data)
        self.assertEqual([path.name for path in staging.iterdir()],
                         ["meta.json"])

        for name in ("x.tif", "y.tif", "z.tif"):
            payload = PATCH_FILES[name]
            self.state.receive_upload_file(
                upload_id, name, io.BytesIO(payload), len(payload))
        record = self.state.finalize_upload(upload_id)["input"]
        self.assertEqual(record["id"], "retried")
        published = Path(record["path"])
        self.assertTrue(published.is_relative_to(self.state.editing().root))
        for name, payload in PATCH_FILES.items():
            self.assertEqual((published / name).read_bytes(), payload)
        self.assertEqual(self.state.finalize_upload(upload_id)["input"], record)
        self.assertEqual(self.state.editing().catalog.entries(), ())

    def test_finalize_rejects_missing_files_and_digest_mismatch(self):
        self._session()
        data = PATCH_FILES["meta.json"]
        request = {"upload_id": spiral_service.secrets.token_hex(16), "kind": "patch", "id": "p2", "files": [
            {"name": "meta.json", "size": len(data), "sha256": _digest(data)},
            {"name": "x.tif", "size": 4, "sha256": _digest(b"xxxx")},
        ]}
        upload_id = self.state.begin_upload(request)["upload_id"]
        self.state.receive_upload_file(upload_id, "meta.json", io.BytesIO(data), len(data))
        with self.assertRaisesRegex(ApiError, "missing declared files"):
            self.state.finalize_upload(upload_id)
        with self.assertRaisesRegex(ApiError, "SHA-256"):
            self.state.receive_upload_file(upload_id, "x.tif", io.BytesIO(b"yyyy"), 4)
        self.assertIsNone(self.state.editing().uploads.get(upload_id).record)
        self.assertEqual(self.state.editing().catalog.entries(), ())

    def test_finalize_rejects_incomplete_v3_fibers_without_repair(self):
        self._session()
        incomplete = {
            "type": "vc3d_fiber",
            "version": 3,
            "line_points": [[0, 0, 0], [1, 0, 0]],
            "control_points": [{"position": [0, 0, 0]}, {"position": [1, 0, 0]}],
        }
        for suffix, mutation, message in (
            ("mode", {}, "missing optimization_mode"),
            ("segment", {"optimization_mode": "lasagna"}, "missing segment_to_next"),
        ):
            payload = {**incomplete, **mutation}
            upload_id = _upload_input(
                self.state,
                "fiber",
                f"bad-v3-{suffix}",
                {"fiber.json": json.dumps(payload).encode()},
            )
            with self.assertRaisesRegex(ApiError, message):
                self.state.finalize_upload(upload_id)

    def test_concurrent_pcl_publications_keep_the_advertised_artifact(self):
        self._session()
        target = self.dataset / "same_windings.json"
        target.write_text('{"collections": {}}')
        self.state.dataset_resolution = mock.Mock(
            scroll_spec={"base_shape_zyx": [10, 20, 30]})
        registered = threading.Event()
        release = threading.Event()
        second_started = threading.Event()
        second_registered = threading.Event()
        real_register = self.state.artifacts.register_directory

        def register(*args, **kwargs):
            ref = real_register(*args, **kwargs)
            if not registered.is_set():
                registered.set()
                self.assertTrue(release.wait(5))
            else:
                second_registered.set()
            return ref

        def second_publish():
            second_started.set()
            return self.state._publish_pcl_artifact(PclRole.SAME_WINDING, target)

        with mock.patch.object(self.state.artifacts, "register_directory", register):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(self.state._publish_pcl_artifact,
                                    PclRole.SAME_WINDING, target)
                try:
                    self.assertTrue(registered.wait(5))
                    second = pool.submit(second_publish)
                    self.assertTrue(second_started.wait(5))
                    self.assertFalse(second_registered.wait(0.1))
                finally:
                    release.set()
                first.result(timeout=5)
                latest = second.result(timeout=5)
        self.assertEqual(self.state.pcl_artifacts["same_winding"], latest)
        artifact, snapshot, _ = self.state.artifacts.acquire_file(
            latest["id"], "same_windings.json")
        try:
            self.assertEqual(snapshot.read_text(), target.read_text())
        finally:
            self.state.artifacts.release(artifact)
        manifests = list((self.output / ".spiral-artifacts").glob(
            "same-winding-*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        self.assertEqual(json.loads(manifests[0].read_text())["source_revision"],
                         latest["source_revision"])

    def test_pcl_artifact_revision_describes_the_snapshot_not_the_live_source(self):
        # Publishing runs without the commit lock. A write landing between
        # the copy and the hash must not advertise the newer document's
        # revision for the older bytes, or a client could edit the stale
        # snapshot and still pass the revision check.
        self._session()
        target = self.dataset / "same_windings.json"
        original = json.dumps({
            "vc_pointcollections_json_version": "1",
            "collections": {"0": {"name": "zero", "points": {}}},
        })
        target.write_text(original)
        replaced = json.dumps({
            "vc_pointcollections_json_version": "1",
            "collections": {"0": {"name": "changed", "points": {}}},
        })

        class Resolution:
            scroll_spec = {"base_shape_zyx": [10, 20, 30]}

        self.state.dataset_resolution = Resolution()
        real_copy2 = shutil.copy2

        def racing_copy2(src, dst, *args, **kwargs):
            result = real_copy2(src, dst, *args, **kwargs)
            Path(src).write_text(replaced)
            return result

        with unittest.mock.patch.object(spiral_service.shutil, "copy2",
                                        racing_copy2):
            ref = self.state._publish_pcl_artifact(PclRole.SAME_WINDING, target)
        snapshot_revision = hashlib.sha256(original.encode("utf-8")).hexdigest()
        self.assertEqual(ref["source_revision"], snapshot_revision)
        self.assertNotEqual(ref["source_revision"],
                            self.state._file_sha256(target))
        manifests = list((self.output / ".spiral-artifacts").glob(
            "same-winding-*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        descriptor = json.loads(manifests[0].read_text())
        self.assertEqual(descriptor["source_revision"], snapshot_revision)
        self.assertEqual(
            (manifests[0].parent / descriptor["pcl_file"]).read_text(), original)

    def test_run_passes_and_validates_transient_influence_config(self):
        session = self._session()
        influence = {
            "influence_enabled": True,
            "influence_z": 1200,
            "influence_windings": 2.5,
            "influence_theta_frac": 0.2,
            "influence_sigma": 0.25,
            "sample_count_influence_footprint_points": 512,
            "sample_count_influence_anchor_lattice_points": 2000,
            "sample_count_influence_anchor_geometry_points": 1000,
            "sample_count_influence_anchor_samples_per_step": 128,
            "influence_anchor_ramp_power": 3.0,
            "loss_weight_anchor": 15.0,
        }
        _planned_run(self.state, {"iterations": 10, "influence_config": influence})
        self.assertEqual(session.run_calls[-1][1], influence)

        with self.assertRaises(ApiError) as caught:
            _planned_run(self.state, {"iterations": 10, "influence_config": {
                "influence_theta_frac": 1.5,
            }})
        self.assertEqual(caught.exception.status, 400)

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
            "influence": {},
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

        self.assertEqual(session.run_calls[-1][2], config)
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

    def test_any_other_static_path_change_is_also_rejected(self):
        self._session()
        inputs = self.state.session_paths.manifest()
        inputs["verified_patches"] = str(self.dataset / "other-patches")
        with self.assertRaisesRegex(ApiError, "Static dataset inputs") as caught:
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
        self.root = base / "dataset"
        self.root.mkdir()
        _write_scroll_spec(self.root)
        (self.root / "umbilicus.json").write_text("{}")
        (self.root / "verified_patches").mkdir()
        self.output = base / "output"
        self.output.mkdir()
        self.resolution = spiral_service.bind_service_paths(
            resolve_dataset_root(self.root), self.output, base / "cache")
        self.assertTrue(self.resolution.ok)
        self.state = ServiceState(dataset_root=str(self.root),
                                  dataset_resolution=self.resolution,
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
        self.assertNotIn("ephemeral_inputs", self.state.status())

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
    def _make_dataset(temporary):
        """A minimal valid dataset plus a sibling output root."""
        dataset = Path(temporary) / "dataset"
        dataset.mkdir(exist_ok=True)
        _write_scroll_spec(dataset)
        (dataset / "umbilicus.json").write_text("{}")
        (dataset / "verified_patches").mkdir(exist_ok=True)
        output = Path(temporary) / "output"
        output.mkdir(exist_ok=True)
        return dataset, output

    def _dataset_arguments(self, temporary):
        dataset, output = self._make_dataset(temporary)
        return ["--dataset", str(dataset), "--output", str(output)]

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

    def test_output_inside_the_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, _ = self._make_dataset(temporary)
            for inside in (dataset, dataset / "spiral_output"):
                process = self._launch(["--port", "0",
                                        "--dataset", str(dataset),
                                        "--output", str(inside)], temporary)
                output, _ = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 2)
                self.assertIn("--output must resolve outside the dataset root",
                              output)

    def test_cache_inside_the_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, output_root = self._make_dataset(temporary)
            process = self._launch(["--port", "0",
                                    "--dataset", str(dataset),
                                    "--output", str(output_root),
                                    "--cache", str(dataset / ".spiral-cache")],
                                   temporary)
            output, _ = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 2)
            self.assertIn("--cache must resolve outside the dataset root",
                          output)

    def test_named_dataset_service_has_exclusive_restartable_ownership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, output_root = self._make_dataset(temporary)
            key_file = root / "key"
            arguments = [
                "--port", "0", "--api-key-file", str(key_file),
                "--dataset", str(dataset), "--output", str(output_root),
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
                self.assertGreaterEqual(len(key), 32)
                self.assertEqual(stat.S_IMODE(key_file.stat().st_mode), 0o600)
                self.assertNotIn(key, ready)
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/health",
                    headers={"Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    health = json.loads(response.read())
                self.assertEqual(health["session_name"], "alice")
                self.assertEqual(health["api_version"], API_VERSION)
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/events?cursor=0",
                    headers={"Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    events = json.loads(response.read())
                self.assertIn(ready, [record["text"]
                                     for record in events["events"]
                                     if record["kind"] == "log"])
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
                self.assertEqual(key_file.read_text().strip(), key)
            finally:
                restarted.terminate()
                restarted.wait(10)
                restarted.stdout.close()
