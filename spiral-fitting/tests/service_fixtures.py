"""Shared fixtures for the focused spiral tests."""

import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import unittest.mock
import urllib.error
import urllib.request
from spiral_service import ServiceState, SpiralServer
from fit_session import SessionState, SpiralInputPaths
from config import Config


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
            "track_min_walk_steps_per_track": 24,
            "track_max_walk_steps_per_track": 256,
            "track_min_walks_per_track": 2,
            "track_max_walks_per_track": 4,
        }
        # The resolved configuration the fit is running, as a real session
        # publishes it once it has one; a checkpoint refusal is analysed
        # against this.
        self.applied_config = None
        self.default_advanced_config = {
            "optimizer_learning_rate": 3e-5,
            "sample_count_patches_per_step": 360,
            "loss_weight_patch_radius": 8.0,
            "track_crossing_precompute_max": 8,
            "track_crossing_mode": "track_walk",
            "track_walk_minimum_cycle_travel": 20.0,
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

    def run(self, count, influence_config=None, run_config=None, path_changes=None,
            autosave_on_pause=True, preview_schedule=None, dt_loss_schedule=None):
        self.run_calls.append((count, dict(influence_config or {}), dict(run_config or {})))
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

    def close(self):
        self.closed = True

    def rebuild_model(self, paths, run, timeout=1800.0):
        self.model_rebuilds.append((paths, run))
        return {"config_revision": 1, "current_iteration": 0}

_NO_DENSE_LOSSES = {
    "dense_spacing_mode": "grad_mag",
    "loss_weight_dense_spacing": 0,
    "loss_weight_dense_normals": 0,
    "loss_weight_shell_outer": 0,
    "loss_weight_shell_patch_radius": 0,
}


def _await_build(state, timeout=10.0):
    """Wait for the asynchronous session build to settle, and report how."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state.session is not None or state._session_state == SessionState.Error:
            return state.status()
        time.sleep(0.01)
    raise AssertionError("the session build did not settle")


def _write_scroll_spec(root):
    (Path(root) / "spiral-scroll.json").write_text(json.dumps({
        "schema_version": 1,
        "name": "s1",
        "voxel_size_um": 9.6,
        "spiral_outward_sense": "CW",
    }))


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
