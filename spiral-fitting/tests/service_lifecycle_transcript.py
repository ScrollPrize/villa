#!/usr/bin/env python3
"""Record the Spiral service's input-agnostic lifecycle over HTTP.

Drives initialize -> run -> stop -> preview -> save -> reconnect -> load
checkpoint against an in-process service whose resident runtime is faked
(as in test_spiral_service_v2.py), and writes a transcript: each request's
HTTP status, response keys and lifecycle values, and the event kinds the
service published. Run it at two checkouts and diff the transcripts to
compare protocol behaviour across a port:

    PYTHONPATH=<checkout>/spiral-fitting python service_lifecycle_transcript.py \\
        --out transcript.json

It adapts to the ownership protocol it finds: the editing-workspace claim
of the file-input service or the session lease that replaced it.
"""
import argparse
import inspect
import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import spiral_runtime
import spiral_service
from config import Config
from fit_session import SessionState, resolve_dataset_root


class FakeRuntime:
    """Just enough of InteractiveFitSession for the service's lifecycle."""

    def __init__(self, *args, **kwargs):
        self.state = SessionState.Idle
        self.completed = 0
        self.target = 0
        self.lock = threading.Lock()

    def status(self):
        with self.lock:
            return {
                "state": self.state, "phase": str(self.state),
                "current_iteration": self.completed,
                "target_iteration": self.target, "latest_metrics": {},
                "warnings": [], "error": None, "preview_manifest_path": None,
                "preview_generation": 0, "progress": None,
                "applied_config": Config().as_dict(),
                "run_config": {}, "default_advanced_config": {},
            }

    def run(self, count, **kwargs):
        with self.lock:
            self.state = SessionState.Running
            self.target = self.completed + count

        def finish():
            time.sleep(0.05)
            with self.lock:
                self.completed = self.target
                self.state = SessionState.Idle
        threading.Thread(target=finish, daemon=True).start()
        return self.target

    def stop(self, barrier=None):
        with self.lock:
            self.state = SessionState.Idle
            self.target = self.completed

    def save_checkpoint(self, path, timeout=120.0):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"PK\x03\x04checkpoint")
        return path

    def load_checkpoint(self, path, timeout=600.0):
        return {"completed_iterations": self.completed, "config_revision": 1}

    def export_preview(self, timeout=600.0, diagnostics=False):
        return {}

    def disable_automatic_previews(self, error=None):
        pass

    def close(self, timeout=15.0):
        with self.lock:
            self.state = SessionState.Closing


# Response fields that vary between runs rather than between protocols.
_VOLATILE = {"session_id", "service_name", "timestamp", "process_id",
             "dataset_root", "workspace_id", "checkpoint_path",
             "output_directory", "cache_directory", "service_version"}
_RECORDED = ("state", "phase", "current_iteration", "target_iteration",
             "session_generation", "session_revision", "accepted",
             "initializing", "rebuilding", "stage", "loaded",
             "preview_exporting", "api_version")


class Client:
    def __init__(self, port, token):
        self.base = f"http://127.0.0.1:{port}"
        self.token = token
        self.transcript = []
        self.commands = 0

    def call(self, method, path, body=None, *, label=None):
        request = urllib.request.Request(self.base + path, method=method)
        request.add_header("Authorization", "Bearer secret")
        if self.token is not None:
            for header in ("X-Spiral-Workspace-Token", "X-Spiral-Session-Token"):
                request.add_header(header, self.token)
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, data=data, timeout=30) as response:
                status, payload = response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            status, payload = error.code, json.loads(error.read() or b"{}")
        self.transcript.append({
            "request": label or f"{method} {path}", "http_status": status,
            "keys": sorted(key for key in payload if key not in _VOLATILE),
            **{key: payload[key] for key in _RECORDED if key in payload},
            **({"error": payload["error"], "details": payload.get("details")}
               if "error" in payload else {}),
        })
        return status, payload

    def command(self, path, body=None):
        self.commands += 1
        return self.call("POST", path, {**(body or {}), "command_id": f"c{self.commands}"})

    def wait_state(self, wanted, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.get_status()
            if status.get("state") == wanted and not status.get("preview_exporting"):
                return status
            time.sleep(0.02)
        raise RuntimeError(f"state {wanted} not reached")

    def get_status(self):
        request = urllib.request.Request(self.base + "/session/status")
        request.add_header("Authorization", "Bearer secret")
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())


# The dataset has no Lasagna stores or shell, so the run turns off the losses
# that need them.
_RUN = {"z_begin": 0, "z_end": 10, "config": {
    "dense_spacing_mode": "grad_mag", "loss_weight_dense_spacing": 0,
    "loss_weight_dense_normals": 0, "loss_weight_shell_outer": 0}}


def _claim_route():
    paths = {route.path for route in spiral_service.ROUTES if route.path}
    if "/session/lease/claim" in paths:
        return "/session/lease/claim"
    return "/session/editing/claim"


def _service_state(root):
    dataset = root / "dataset"
    dataset.mkdir()
    (dataset / "spiral-scroll.json").write_text(json.dumps({
        "schema_version": 1, "name": "s1", "voxel_size_um": 9.6,
        "spiral_outward_sense": "CW"}))
    (dataset / "umbilicus.json").write_text("{}")
    (dataset / "verified_patches").mkdir()
    output, cache = root / "output", root / "cache"
    output.mkdir()
    resolution = spiral_service.bind_service_paths(
        resolve_dataset_root(dataset), output, cache)
    kwargs = dict(dataset_root=str(dataset), dataset_resolution=resolution,
                  startup_run=dict(_RUN))
    if "object_store" in inspect.signature(spiral_service.ServiceState).parameters:
        kwargs["object_store"] = spiral_service.bind_object_store(
            _empty_store(root / "inputs.vcobj"))
    return spiral_service.ServiceState(**kwargs)


def _empty_store(path):
    from vc_object_store import ObjectStore
    with ObjectStore.create(str(path), author="transcript", hostname="host"):
        pass
    return str(path)


def record(out):
    spiral_runtime.create_session = FakeRuntime
    with tempfile.TemporaryDirectory() as directory:
        state = _service_state(Path(directory))
        server = spiral_service.SpiralServer(("127.0.0.1", 0), ["secret"], state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = Client(server.server_port, "owner-token")
            client.call("GET", "/health")
            client.call("GET", "/dataset")
            client.call("POST", _claim_route(), {"command_id": "claim-1"},
                        label="POST claim")
            client.command("/session/initialize", {"run": _RUN})
            client.wait_state(SessionState.Idle)
            client.call("GET", "/session/status")
            revision = client.get_status()["session_revision"]
            configuration = Config().as_dict()
            client.command("/session/run", {
                "iterations": 3, "expected_session_revision": revision,
                "configuration": configuration,
                "dt_loss_schedule": {"enabled": False, "last_fraction": 0.0}})
            client.wait_state(SessionState.Idle)
            client.command("/session/stop")
            client.command("/session/export-preview")
            client.wait_state(SessionState.Idle)
            _, saved = client.command("/session/save-checkpoint", {"name": "baseline"})
            # Reconnect: a second client reads status first, then events,
            # then reclaims ownership with the same token.
            again = Client(server.server_port, "owner-token")
            again.commands = 100
            again.call("GET", "/session/status")
            _, events = again.call("GET", "/events?cursor=0")
            again.call("POST", _claim_route(), {"command_id": "claim-2"},
                       label="POST claim")
            again.command("/session/load-checkpoint",
                          {"host_checkpoint": saved.get("checkpoint_path", "")})
            stranger = Client(server.server_port, "another-token")
            stranger.command("/session/stop")
            transcript = {
                "requests": client.transcript + again.transcript + stranger.transcript,
                "event_kinds": sorted({(event["kind"], event["source"])
                                       for event in events["events"]}),
            }
        finally:
            server.shutdown()
            server.server_close()
            state.close()
    transcript["event_kinds"] = [list(kind) for kind in transcript["event_kinds"]]
    with open(out, "w") as stream:
        json.dump(transcript, stream, indent=1, sort_keys=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    record(parser.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
