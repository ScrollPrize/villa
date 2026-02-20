"""HTTP service wrapping the 2D fit optimizer for use by VC3D.

Start with:
    python fit_service.py [--port PORT]

Endpoints:
    GET  /health          -> {"status": "ok"}
    GET  /status          -> current job state
    POST /optimize        -> start an optimization job (JSON body)
    POST /stop            -> request cancellation of the running job
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any


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

    def set_finished(self, output_dir: str) -> None:
        with self._lock:
            self._state = "finished"
            self._output_dir = output_dir

    def set_error(self, msg: str) -> None:
        with self._lock:
            self._state = "error"
            self._error = msg

    def set_idle(self) -> None:
        with self._lock:
            self._state = "idle"
            self._error = None
            self._cancel = False
            self._output_dir = None

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
    """Run fit.py then fit2tifxyz.py based on the request body."""
    global _job

    model_input = body.get("model_input")
    model_output = body.get("model_output")
    data_input = body.get("data_input")
    output_dir = body.get("output_dir")
    config = body.get("config", {})

    if not model_input:
        _job.set_error("missing 'model_input'")
        return
    if not data_input:
        _job.set_error("missing 'data_input'")
        return
    if not output_dir:
        _job.set_error("missing 'output_dir'")
        return

    try:
        import tempfile
        # Use a temp directory for all intermediate files (config json,
        # model output, etc.) so nothing leaks into the volpkg paths dir.
        tmp_dir = tempfile.mkdtemp(prefix="fit_reopt_")

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

        # Clean up temp directory (config json, intermediate model, etc.)
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        _job.set_finished(str(output_dir))
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
        else:
            self._send_json({"error": "not found"}, 404)

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
    p = argparse.ArgumentParser(description="Fit optimizer HTTP service for VC3D")
    p.add_argument("--port", type=int, default=0, help="Port (0 = auto-select)")
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    server = HTTPServer((args.host, args.port), _Handler)
    actual_port = server.server_address[1]

    # This exact format is parsed by FitServiceManager on the C++ side
    print(f"listening on http://{args.host}:{actual_port}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
