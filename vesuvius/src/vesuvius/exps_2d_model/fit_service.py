"""HTTP service wrapping the 2D fit optimizer for use by VC3D.

Start with:
    python fit_service.py [--port PORT] [--data-dir PATH]

Endpoints:
    GET  /health          -> {"status": "ok"}
    GET  /status          -> current job state
    GET  /datasets        -> available .zarr datasets from --data-dir
    POST /optimize        -> start an optimization job (JSON body)
    POST /stop            -> request cancellation of the running job
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

_data_dir: str | None = None  # Set via --data-dir CLI flag


# ---------------------------------------------------------------------------
# Service announcement (file-based discovery)
# ---------------------------------------------------------------------------

_ANNOUNCE_DIR = Path.home() / ".fit_services"
_announce_file: Path | None = None


def _list_datasets() -> list[dict[str, str]]:
    """Return available .zarr directories from _data_dir."""
    if not _data_dir:
        return []
    data_path = Path(_data_dir)
    if not data_path.is_dir():
        return []
    datasets = []
    for entry in sorted(data_path.iterdir()):
        if entry.is_dir() and entry.name.endswith(".zarr"):
            datasets.append({"name": entry.name, "path": str(entry.resolve())})
    return datasets


def _clean_stale_announcements() -> None:
    """Remove announcement files whose PIDs are no longer alive."""
    if not _ANNOUNCE_DIR.is_dir():
        return
    for f in _ANNOUNCE_DIR.glob("*.json"):
        try:
            info = json.loads(f.read_text())
            pid = info.get("pid", -1)
            # Check if process is alive
            os.kill(pid, 0)
        except (OSError, json.JSONDecodeError, TypeError):
            try:
                f.unlink()
            except OSError:
                pass


def _write_announcement(host: str, port: int) -> None:
    """Write a service announcement file for discovery."""
    global _announce_file
    _ANNOUNCE_DIR.mkdir(parents=True, exist_ok=True)
    _clean_stale_announcements()

    pid = os.getpid()
    datasets = _list_datasets()
    info = {
        "host": host,
        "port": port,
        "pid": pid,
        "data_dir": _data_dir or "",
        "datasets": [d["name"] for d in datasets],
    }
    _announce_file = _ANNOUNCE_DIR / f"{pid}.json"
    _announce_file.write_text(json.dumps(info, indent=2))


def _remove_announcement() -> None:
    """Remove the announcement file on shutdown."""
    global _announce_file
    if _announce_file is not None:
        try:
            _announce_file.unlink(missing_ok=True)
        except OSError:
            pass
        _announce_file = None


# ---------------------------------------------------------------------------
# mDNS announcement via avahi-publish-service
# ---------------------------------------------------------------------------

_avahi_proc: subprocess.Popen | None = None


def _start_avahi_publish(port: int) -> None:
    """Publish service via avahi-publish-service (auto-unregisters on exit)."""
    global _avahi_proc

    txt_records: list[str] = []
    if _data_dir:
        txt_records.append(f"data_dir={_data_dir}")
    datasets = _list_datasets()
    if datasets:
        txt_records.append("datasets=" + ",".join(d["name"] for d in datasets))

    cmd = [
        "avahi-publish-service",
        f"Fit Optimizer (pid {os.getpid()})",
        "_fitoptimizer._tcp",
        str(port),
    ] + txt_records

    try:
        _avahi_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[fit-service] mDNS: registered as _fitoptimizer._tcp on port {port}",
              flush=True)
    except FileNotFoundError:
        print("[fit-service] avahi-publish-service not found, skipping mDNS",
              flush=True)


def _stop_avahi_publish() -> None:
    """Stop the avahi-publish-service subprocess."""
    global _avahi_proc
    if _avahi_proc is not None:
        _avahi_proc.terminate()
        try:
            _avahi_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _avahi_proc.kill()
        _avahi_proc = None


# ---------------------------------------------------------------------------
# Global job state (one job at a time)
# ---------------------------------------------------------------------------

class _JobState:
    """Thread-safe mutable job state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "idle"
        self._stage = ""
        self._step = 0
        self._total_steps = 0
        self._loss = 0.0
        self._error: str | None = None
        self._cancel = False
        self._output_dir: str | None = None
        self._results_tmp: str | None = None  # temp dir to clean up after download

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "stage": self._stage,
                "step": self._step,
                "total_steps": self._total_steps,
                "loss": self._loss,
                "error": self._error,
                "output_dir": self._output_dir,
            }

    def set_running(self, stage: str, step: int, total: int, loss: float) -> None:
        with self._lock:
            self._state = "running"
            self._stage = stage
            self._step = step
            self._total_steps = total
            self._loss = loss

    def set_finished(self, output_dir: str, results_tmp: str | None = None) -> None:
        with self._lock:
            self._state = "finished"
            self._output_dir = output_dir
            self._results_tmp = results_tmp

    def set_error(self, msg: str) -> None:
        with self._lock:
            self._state = "error"
            self._error = msg

    def set_idle(self) -> None:
        with self._lock:
            # Clean up any leftover results temp dir from previous run
            if self._results_tmp:
                import shutil
                shutil.rmtree(self._results_tmp, ignore_errors=True)
            self._state = "idle"
            self._error = None
            self._cancel = False
            self._output_dir = None
            self._results_tmp = None

    @property
    def results_tmp(self) -> str | None:
        with self._lock:
            return self._results_tmp

    def clear_results(self) -> None:
        """Clean up results temp dir after download."""
        with self._lock:
            if self._results_tmp:
                import shutil
                shutil.rmtree(self._results_tmp, ignore_errors=True)
                self._results_tmp = None

    def request_cancel(self) -> None:
        with self._lock:
            self._cancel = True

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancel

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._state == "running"


_job = _JobState()
_job_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Optimization runner (called in background thread)
# ---------------------------------------------------------------------------

def _run_optimization(body: dict[str, Any]) -> None:
    """Run fit.py then fit2tifxyz.py based on the request body.

    Supports two modes:
      - Local (internal): model_input and output_dir are local paths.
      - Remote (external): model_data contains base64-encoded model bytes,
        output goes to a temp dir, caller downloads results via GET /results.
    """
    global _job
    import base64
    import tempfile

    model_input = body.get("model_input")
    model_data = body.get("model_data")  # base64-encoded model bytes
    model_output = body.get("model_output")
    data_input = body.get("data_input")
    output_dir = body.get("output_dir")
    config = body.get("config", {})

    if not model_input and not model_data:
        _job.set_error("missing 'model_input' or 'model_data'")
        return
    if not data_input:
        _job.set_error("missing 'data_input'")
        return

    try:
        # Use a temp directory for all intermediate files (config json,
        # model output, etc.) so nothing leaks into the volpkg paths dir.
        tmp_dir = tempfile.mkdtemp(prefix="fit_reopt_")

        # Handle model_data (external/remote mode): decode and save to temp
        if model_data:
            model_bytes = base64.b64decode(model_data)
            model_input = str(Path(tmp_dir) / "model_input.pt")
            Path(model_input).write_bytes(model_bytes)
            print(f"[fit-service] received model data ({len(model_bytes)} bytes)", flush=True)

        # If no output_dir, create a temp dir for results (external mode)
        results_tmp = None
        if not output_dir:
            results_tmp = tempfile.mkdtemp(prefix="fit_results_")
            output_dir = results_tmp

        # model_output goes into temp dir
        if not model_output:
            model_output = str(Path(tmp_dir) / "model_reopt.pt")

        # Build argv for fit.py from the config dict.
        cfg = dict(config)
        args_section = dict(cfg.get("args", {}))
        args_section["input"] = str(data_input)
        args_section["model-input"] = str(model_input)
        args_section["model-output"] = str(model_output)
        # Only set out-dir if explicitly requested (enables debug vis output).
        # By default we skip vis output for speed.
        if body.get("out_dir"):
            args_section["out-dir"] = str(body["out_dir"])
        cfg["args"] = args_section
        cfg_path = str(Path(tmp_dir) / "fit_config.json")
        Path(cfg_path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        # Monkey-patch the optimizer to report progress & check cancellation.
        import optimizer as opt_mod

        _orig_optimize = opt_mod.optimize

        def _patched_optimize(**kwargs: Any) -> Any:
            orig_snapshot = kwargs.get("snapshot_fn")
            orig_progress = kwargs.get("progress_fn")

            def _wrapped_snapshot(*, stage: str, step: int, loss: float, **kw: Any) -> None:
                if _job.cancelled:
                    raise KeyboardInterrupt("cancelled by user")
                if orig_snapshot is not None:
                    orig_snapshot(stage=stage, step=step, loss=loss, **kw)

            def _wrapped_progress(*, step: int, total: int, loss: float) -> None:
                _job.set_running("optimizing", step, total, loss)
                if _job.cancelled:
                    raise KeyboardInterrupt("cancelled by user")
                if orig_progress is not None:
                    orig_progress(step=step, total=total, loss=loss)

            kwargs["snapshot_fn"] = _wrapped_snapshot
            kwargs["progress_fn"] = _wrapped_progress
            return _orig_optimize(**kwargs)

        opt_mod.optimize = _patched_optimize
        try:
            import fit as fit_mod
            _job.set_running("loading", 0, 0, 0.0)
            fit_mod.main([cfg_path])
        finally:
            opt_mod.optimize = _orig_optimize

        if _job.cancelled:
            _job.set_error("cancelled")
            return

        # Export to tifxyz
        _job.set_running("exporting", 0, 0, 0.0)
        import fit2tifxyz
        export_argv = ["--input", str(model_output), "--output", str(output_dir)]
        if body.get("single_segment"):
            export_argv.append("--single-segment")
        if body.get("copy_model"):
            export_argv.append("--copy-model")
        output_name = body.get("output_name")
        if output_name:
            export_argv.extend(["--output-name", str(output_name)])
        fit2tifxyz.main(export_argv)

        # Clean up intermediate files (but keep results_tmp for download)
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        _job.set_finished(str(output_dir), results_tmp=results_tmp)
        print(f"[fit-service] optimization finished, output: {output_dir}", flush=True)

    except KeyboardInterrupt:
        _job.set_error("cancelled")
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[fit-service] error: {tb}", file=sys.stderr, flush=True)
        _job.set_error(str(exc))


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):

    def _send_json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/status":
            self._send_json(_job.snapshot())
        elif self.path == "/datasets":
            self._send_json({"datasets": _list_datasets()})
        elif self.path == "/results":
            self._handle_results()
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_results(self) -> None:
        """Package finished optimization results as tar.gz and send them."""
        import tarfile
        import io

        snap = _job.snapshot()
        if snap["state"] != "finished":
            self._send_json({"error": "no finished results available"}, 404)
            return

        output_dir = snap["output_dir"]
        if not output_dir or not Path(output_dir).is_dir():
            self._send_json({"error": "output directory not found"}, 404)
            return

        # Create tar.gz in memory.  Archive paths are relative to output_dir
        # so the tar contains e.g. "winding_combined_v004.tifxyz/meta.json".
        # Extracting in the local paths dir recreates the tifxyz subdirectory.
        buf = io.BytesIO()
        out_path = Path(output_dir)
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for child in sorted(out_path.iterdir()):
                tar.add(str(child), arcname=child.name)

        data = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/gzip")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

        print(f"[fit-service] results downloaded ({len(data)} bytes)", flush=True)
        _job.clear_results()

    def do_POST(self) -> None:  # noqa: N802
        global _job_thread

        if self.path == "/optimize":
            if _job.is_busy:
                self._send_json({"error": "optimization already running"}, 409)
                return
            try:
                body = self._read_json()
            except Exception as exc:
                self._send_json({"error": f"bad json: {exc}"}, 400)
                return
            _job.set_idle()
            _job.set_running("starting", 0, 0, 0.0)
            _job_thread = threading.Thread(target=_run_optimization, args=(body,), daemon=True)
            _job_thread.start()
            self._send_json({"status": "started"})

        elif self.path == "/stop":
            if _job.is_busy:
                _job.request_cancel()
                self._send_json({"status": "stopping"})
            else:
                self._send_json({"status": "not running"})

        else:
            self._send_json({"error": "not found"}, 404)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Prefix HTTP logs
        print(f"[fit-service] {fmt % args}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _data_dir

    p = argparse.ArgumentParser(description="Fit optimizer HTTP service for VC3D")
    p.add_argument("--port", type=int, default=0, help="Port (0 = auto-select)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--data-dir", default=None,
                   help="Directory containing .zarr datasets")
    args = p.parse_args()

    if args.data_dir:
        _data_dir = str(Path(args.data_dir).resolve())

    server = HTTPServer((args.host, args.port), _Handler)
    actual_port = server.server_address[1]

    # Write service announcement for discovery (file-based + mDNS)
    _write_announcement(args.host, actual_port)
    _start_avahi_publish(actual_port)
    atexit.register(_remove_announcement)
    atexit.register(_stop_avahi_publish)

    # This exact format is parsed by FitServiceManager on the C++ side
    print(f"listening on http://{args.host}:{actual_port}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop_avahi_publish()
        _remove_announcement()
        server.server_close()


if __name__ == "__main__":
    main()
