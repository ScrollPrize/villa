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
    output_dir = body.get("output_dir")
    config = body.get("config", {})

    if not model_input:
        _job.set_error("missing 'model_input'")
        return
    if not output_dir:
        _job.set_error("missing 'output_dir'")
        return

    # model_output defaults to overwriting model_input
    if not model_output:
        model_output = model_input

    try:
        # Build argv for fit.py from the config dict.
        # The config is the full JSON config (base, stages, args) just like
        # direct_from_zarr.json.  We write it to a temp file and pass as
        # a positional json config arg.
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=output_dir
        ) as f:
            # Inject model_input / model_output into the args section
            cfg = dict(config)
            args_section = dict(cfg.get("args", {}))
            args_section["model-input"] = str(model_input)
            args_section["model-output"] = str(model_output)
            args_section["out-dir"] = str(output_dir)
            cfg["args"] = args_section
            json.dump(cfg, f, indent=2)
            cfg_path = f.name

        # Monkey-patch the optimizer to report progress & check cancellation.
        import optimizer as opt_mod

        _orig_optimize = opt_mod.optimize

        def _patched_optimize(**kwargs: Any) -> Any:
            orig_snapshot = kwargs.get("snapshot_fn")

            def _wrapped_snapshot(*, stage: str, step: int, loss: float, **kw: Any) -> None:
                _job.set_running(stage, step, 0, loss)
                if _job.cancelled:
                    raise KeyboardInterrupt("cancelled by user")
                if orig_snapshot is not None:
                    orig_snapshot(stage=stage, step=step, loss=loss, **kw)

            kwargs["snapshot_fn"] = _wrapped_snapshot
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
        fit2tifxyz.main(["--input", str(model_output), "--output", str(output_dir)])

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
