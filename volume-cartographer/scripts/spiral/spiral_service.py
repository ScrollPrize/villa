#!/usr/bin/env python3
"""Loopback-only HTTP service for a persistent interactive Spiral fit."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import secrets
import signal
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from fit_session import API_VERSION, parse_session_request, resolve_dataset_root, validate_session_request


MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_DEDUPLICATED_COMMANDS = 256


class ApiError(Exception):
    def __init__(self, status, message, details=None):
        super().__init__(message)
        self.status = int(status)
        self.message = message
        self.details = details


class ServiceState:
    def __init__(self):
        self.lock = threading.RLock()
        self.session = None
        self.session_id = None
        self.service_generation = 1
        self.session_generation = 0
        self.command_generation = 0
        self.status_generation = 0
        self.commands = OrderedDict()
        self.inflight_commands = set()
        self.command_condition = threading.Condition(self.lock)
        self.replacing = False
        self.replacement_old_session_released = False

    def _base(self):
        return {
            "api_version": API_VERSION,
            "session_id": self.session_id,
            "service_generation": self.service_generation,
            "session_generation": self.session_generation,
            "command_generation": self.command_generation,
            "generation": self.status_generation,
            "session_replacement_in_progress": self.replacing,
            "replacement_old_session_released": self.replacement_old_session_released,
        }

    def status(self):
        with self.lock:
            response = self._base()
            response.update(self.session.status() if self.session else {
                "state": "Empty", "phase": "No session", "current_iteration": 0,
                "target_iteration": 0, "latest_metrics": {}, "warnings": [],
                "error": None, "preview_manifest_path": None, "preview_generation": 0,
            })
            return response

    def health(self):
        response = self._base()
        response.update({
            "ready": True,
            "process_id": os.getpid(),
            "cuda_ready": None if not self.session else self.session.status()["state"] != "Error",
        })
        return response

    def load(self, request):
        paths, run, preview = parse_session_request(request)
        errors = validate_session_request(paths, run)
        if errors:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Session validation failed", errors)
        with self.lock:
            if self.replacing:
                raise ApiError(HTTPStatus.CONFLICT, "A session replacement is already in progress")
            if self.session and self.session.status()["state"] in {
                "Loading", "Running", "Saving", "ExportingPreview"
            }:
                raise ApiError(HTTPStatus.CONFLICT, "The current session is active")
            previous = self.session
            self.replacing = True
            self.replacement_old_session_released = False
        try:
            if previous:
                previous.close()
                with self.lock:
                    # Validation happened before replacement.  Once teardown has
                    # succeeded, report honestly that the previous resident CUDA
                    # session is no longer available even if new loading fails.
                    if self.session is previous:
                        self.session = None
                        self.session_id = None
                    self.replacement_old_session_released = True
                    self.status_generation += 1
            from spiral_runtime import create_session
            with self.lock:
                self.session_generation += 1
                self.session_id = f"spiral-{self.session_generation}-{secrets.token_hex(5)}"
                self.session = create_session(paths, run, preview, self._status_changed)
                self.status_generation += 1
                response = self.status()
                response["accepted"] = True
                return response
        finally:
            with self.lock:
                self.replacing = False

    def _status_changed(self, _status):
        with self.lock:
            self.status_generation += 1

    def run(self, request):
        session = self._require_session()
        target = session.run(int(request.get("iterations", 0)))
        with self.lock:
            self.status_generation += 1
        return {**self.status(), "accepted": True, "target_iteration": target}

    def stop(self):
        self._require_session().stop()
        with self.lock:
            self.status_generation += 1
        return {**self.status(), "accepted": True}

    def save_checkpoint(self, request):
        session = self._require_session()
        path = request.get("path")
        if not path:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Checkpoint path is required")
        saved = session.save_checkpoint(str(Path(path).expanduser().resolve(strict=False)))
        return {**self.status(), "checkpoint_path": saved}

    def delete(self):
        with self.lock:
            if not self.session:
                return {**self.status(), "deleted": False}
            if self.session.status()["state"] in {"Loading", "Running", "Saving", "ExportingPreview"}:
                raise ApiError(HTTPStatus.CONFLICT, "Stop and wait for the session to settle before deleting it")
            session = self.session
            self.session = None
            self.session_id = None
            self.session_generation += 1
            self.status_generation += 1
        session.close()
        return {**self.status(), "deleted": True}

    def _require_session(self):
        with self.lock:
            if self.session is None:
                raise ApiError(HTTPStatus.CONFLICT, "No fit session is loaded")
            return self.session

    def deduplicated(self, command_id, operation):
        if not isinstance(command_id, str) or not command_id.strip():
            raise ApiError(HTTPStatus.BAD_REQUEST, "A non-empty command_id is required")
        with self.lock:
            while command_id in self.inflight_commands:
                self.command_condition.wait()
            if command_id in self.commands:
                cached = self.commands[command_id]
                self.commands.move_to_end(command_id)
                return cached
            self.inflight_commands.add(command_id)
        try:
            response = operation()
            with self.lock:
                self.command_generation += 1
                response["command_generation"] = self.command_generation
                self.commands[command_id] = response
                while len(self.commands) > MAX_DEDUPLICATED_COMMANDS:
                    self.commands.popitem(last=False)
            return response
        finally:
            with self.lock:
                self.inflight_commands.discard(command_id)
                self.command_condition.notify_all()

    def close(self):
        with self.lock:
            session = self.session
            self.session = None
        if session:
            session.close()


class SpiralServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, nonce, state):
        super().__init__(address, SpiralHandler)
        self.nonce = nonce
        self.state = state


class SpiralHandler(BaseHTTPRequestHandler):
    server_version = "VC3D-Spiral/1"

    def log_message(self, fmt, *args):
        print("SPIRAL_HTTP " + (fmt % args), file=sys.stderr, flush=True)

    def _authorise(self):
        if not secrets.compare_digest(self.headers.get("X-Spiral-Nonce", ""), self.server.nonce):
            raise ApiError(HTTPStatus.UNAUTHORIZED, "Invalid service nonce")

    def _body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
        if length < 0 or length > MAX_BODY_BYTES:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body is too large")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {exc}")

    def _send(self, status, value):
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def _dispatch(self):
        self._authorise()
        path = urlparse(self.path).path.rstrip("/") or "/"
        state = self.server.state
        if self.command == "GET" and path == "/health":
            return state.health()
        if self.command == "GET" and path == "/session/status":
            return state.status()
        if self.command == "POST" and path == "/dataset/resolve":
            body = self._body()
            return {**state._base(), **resolve_dataset_root(body.get("dataset_root", "")).to_dict()}
        if self.command == "DELETE" and path == "/session":
            body = self._body()
            return state.deduplicated(body.get("command_id"), state.delete)
        if self.command == "POST":
            body = self._body()
            command_id = body.get("command_id")
            if path == "/session/load":
                return state.deduplicated(command_id, lambda: state.load(body))
            if path == "/session/run":
                return state.deduplicated(command_id, lambda: state.run(body))
            if path == "/session/stop":
                return state.deduplicated(command_id, state.stop)
            if path == "/session/save-checkpoint":
                return state.deduplicated(command_id, lambda: state.save_checkpoint(body))
            if path == "/session/export-full":
                raise ApiError(HTTPStatus.NOT_IMPLEMENTED, "Full diagnostic export is not implemented by the interactive service")
        raise ApiError(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _handle(self):
        try:
            self._send(HTTPStatus.OK, self._dispatch())
        except ApiError as exc:
            payload = self.server.state._base()
            payload.update({"error": exc.message, "details": exc.details})
            self._send(exc.status, payload)
        except Exception as exc:
            payload = self.server.state._base()
            payload.update({"error": f"{type(exc).__name__}: {exc}"})
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, payload)

    do_GET = _handle
    do_POST = _handle
    do_DELETE = _handle


def _install_parent_watch(parent_pid, shutdown):
    if not parent_pid:
        return
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL(None)
            libc.prctl(1, signal.SIGTERM)
        except Exception:
            pass

    def watch():
        while not shutdown.is_set():
            try:
                os.kill(parent_pid, 0)
            except OSError:
                shutdown.set()
                return
            shutdown.wait(2.0)
    threading.Thread(target=watch, name="spiral-parent-watch", daemon=True).start()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.nonce:
        parser.error("--nonce must not be empty")

    state = ServiceState()
    server = SpiralServer(("127.0.0.1", args.port), args.nonce, state)
    shutdown = threading.Event()
    _install_parent_watch(args.parent_pid, shutdown)

    def request_shutdown(_signum=None, _frame=None):
        shutdown.set()
    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    print(f"SPIRAL_SERVICE_READY port={server.server_port} api_version={API_VERSION}", flush=True)
    server.timeout = 0.5
    try:
        while not shutdown.is_set():
            server.handle_request()
    finally:
        server.server_close()
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
