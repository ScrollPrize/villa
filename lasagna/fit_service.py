"""HTTP service wrapping the 3D fit optimizer for use by VC3D.

Start with:
    python fit_service.py [--port PORT] [--data-dir PATH]

Endpoints:
    GET  /health          -> {"status": "ok"}
    GET  /status          -> current job state
    GET  /datasets        -> available .lasagna.json datasets from --data-dir
    POST /optimize        -> start an optimization job (JSON body)
    POST /stop            -> request cancellation of the running job
    POST /export_vis      -> export multi-layer OBJ visualization (JSON body)
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
_gpu_pause_enabled: bool = True  # Set via --no-gpu-pause CLI flag
_sparse_prefetch_backend: str = "tensorstore"  # Set via --sparse-prefetch-backend

# One debug switch for sparse coverage and coordinate sanity guards. The service
# enables it by default so optimizer jobs fail before CUDA indexing asserts.
os.environ.setdefault("LASAGNA_CHECK_SPARSE_CACHE", "1")


def _truthy_config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _approval_inpaint_enabled(args_section: dict[str, Any]) -> bool:
    return _truthy_config_bool(args_section.get("approval-inpaint", False))


def _config_enables_pred_dt_flow_gate(cfg: dict[str, Any]) -> bool:
    stages = cfg.get("stages")
    if not isinstance(stages, list):
        return False
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        opt_cfg = stage.get("global_opt")
        if not isinstance(opt_cfg, dict):
            opt_cfg = stage
        args = opt_cfg.get("args")
        if not isinstance(args, dict):
            continue
        gate = args.get("pred_dt_flow_gate")
        if isinstance(gate, dict) and bool(gate.get("enabled", False)):
            return True
    return False


def _set_pred_dt_flow_gate_debug_out_dir(cfg: dict[str, Any], out_dir: str) -> None:
    stages = cfg.get("stages")
    if not isinstance(stages, list):
        return
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        opt_cfg = stage.get("global_opt")
        if not isinstance(opt_cfg, dict):
            opt_cfg = stage
        args = opt_cfg.get("args")
        if not isinstance(args, dict):
            continue
        gate = args.get("pred_dt_flow_gate")
        if isinstance(gate, dict) and bool(gate.get("enabled", False)):
            gate.setdefault("debug_out_dir", out_dir)


def _decode_tifxyz_for_request(
    *,
    body: dict[str, Any],
    cfg: dict[str, Any],
    args_section: dict[str, Any],
    tmp_dir: str,
    model_init: str,
    ext_offset_enabled: bool,
) -> str | None:
    """Decode generic request tifxyz and attach it to configured consumers."""
    import base64

    approval_enabled = _approval_inpaint_enabled(args_section)
    if approval_enabled and model_init != "seed":
        raise ValueError("args.approval-inpaint is only valid with args.model-init=seed")

    tifxyz_payload = body.get("tifxyz")
    if "tifxyz" in body and not isinstance(tifxyz_payload, dict):
        raise ValueError("request tifxyz must be an object mapping filenames to base64 data")

    tifxyz_dir: str | None = None
    if isinstance(tifxyz_payload, dict):
        tifxyz_dir = str(Path(tmp_dir) / "tifxyz_input")
        Path(tifxyz_dir).mkdir(parents=True, exist_ok=True)
        for fname, b64 in tifxyz_payload.items():
            (Path(tifxyz_dir) / fname).write_bytes(base64.b64decode(b64))
        print(f"[fit-service] decoded tifxyz ({len(tifxyz_payload)} files) to {tifxyz_dir}", flush=True)
        if model_init == "ext":
            args_section["tifxyz-init"] = tifxyz_dir
        if approval_enabled:
            args_section["approval-inpaint-tifxyz"] = tifxyz_dir
        if ext_offset_enabled:
            offset_val = float(cfg.pop("offset_value", 1.0))
            cfg["external_surfaces"] = [{"path": tifxyz_dir, "offset": offset_val}]
    elif model_init == "ext":
        raise ValueError("model-init=ext requires request tifxyz")
    elif ext_offset_enabled:
        raise ValueError("ext_offset is enabled but request has no tifxyz")
    elif approval_enabled:
        raise ValueError("approval-inpaint requires request tifxyz")

    return tifxyz_dir


def _config_effective_ext_offset_enabled(cfg: dict[str, Any]) -> bool:
    base_cfg = cfg.get("base")
    base_ext = 0.0
    if isinstance(base_cfg, dict):
        try:
            base_ext = float(base_cfg.get("ext_offset", 0.0))
        except (TypeError, ValueError):
            base_ext = 0.0
    if abs(base_ext) > 0.0:
        return True

    stages = cfg.get("stages")
    if not isinstance(stages, list):
        return False
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        opt_cfg = stage.get("global_opt")
        if not isinstance(opt_cfg, dict):
            opt_cfg = stage
        try:
            steps = int(opt_cfg.get("steps", 0))
        except (TypeError, ValueError):
            steps = 0
        if steps <= 0:
            continue

        eff = base_ext
        w_fac = opt_cfg.get("w_fac")
        has_ext_wfac = isinstance(w_fac, dict) and "ext_offset" in w_fac
        default_mul = opt_cfg.get("default_mul")
        if default_mul is not None and not has_ext_wfac:
            try:
                eff = base_ext * float(default_mul)
            except (TypeError, ValueError):
                eff = 0.0
        if has_ext_wfac and w_fac.get("ext_offset") is not None:
            try:
                eff = base_ext * float(w_fac.get("ext_offset"))
            except (TypeError, ValueError):
                eff = 0.0
        if abs(eff) > 0.0:
            return True
    return False


# ---------------------------------------------------------------------------
# Service announcement (file-based discovery)
# ---------------------------------------------------------------------------

_ANNOUNCE_DIR = Path.home() / ".fit_services"
_announce_file: Path | None = None


def _list_datasets() -> list[dict[str, str]]:
    """Return available .lasagna.json datasets from _data_dir."""
    if not _data_dir:
        return []
    data_path = Path(_data_dir)
    if not data_path.is_dir():
        return []
    datasets = []
    for entry in sorted(data_path.iterdir()):
        if entry.is_file() and entry.name.endswith(".lasagna.json"):
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
        self._stage_progress = 0.0
        self._overall_progress = 0.0
        self._stage_name = ""
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
                "stage_progress": self._stage_progress,
                "overall_progress": self._overall_progress,
                "stage_name": self._stage_name,
                "error": self._error,
                "output_dir": self._output_dir,
            }

    def set_running(self, stage: str, step: int, total: int, loss: float,
                    stage_progress: float = 0.0, overall_progress: float = 0.0,
                    stage_name: str = "") -> None:
        with self._lock:
            self._state = "running"
            self._stage = stage
            self._step = step
            self._total_steps = total
            self._loss = loss
            self._stage_progress = stage_progress
            self._overall_progress = overall_progress
            self._stage_name = stage_name

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
    is_new_model = not model_input and not model_data

    if not data_input:
        _job.set_error("missing 'data_input'")
        return

    try:
        # Use a temp directory for all intermediate files (config json,
        # model output, etc.) so nothing leaks into the volpkg paths dir.
        tmp_dir = tempfile.mkdtemp(prefix="fit_reopt_")
        service_workdir = Path.cwd()
        print(f"[fit-service] cwd: {service_workdir}", flush=True)

        # Handle model_data (external/remote mode): decode and save to temp
        if model_data:
            model_bytes = base64.b64decode(model_data)
            model_input = str(Path(tmp_dir) / "model_input.pt")
            Path(model_input).write_bytes(model_bytes)
            print(f"[fit-service] received model data ({len(model_bytes)} bytes)", flush=True)
        elif is_new_model:
            print("[fit-service] new model mode (no model_input)", flush=True)

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
        args_section_pre = cfg.get("args", {})
        if not isinstance(args_section_pre, dict):
            args_section_pre = {}
        model_init = str(args_section_pre.get("model-init", args_section_pre.get("model_init", "seed"))).strip().lower()
        if model_init not in {"seed", "ext", "model"}:
            raise ValueError(f"invalid args.model-init '{model_init}' (expected seed, ext, or model)")
        args_section_pre.pop("model_init", None)
        args_section_pre["model-init"] = model_init
        if model_init != "ext" and args_section_pre.get("tifxyz-init"):
            raise ValueError("args.tifxyz-init is only valid with args.model-init=ext")
        if model_init != "model" and args_section_pre.get("model-input"):
            raise ValueError("args.model-input is only valid with args.model-init=model")
        if model_init != "model" and model_input:
            raise ValueError("request model_data/model_input is only valid with args.model-init=model")
        cfg["args"] = args_section_pre
        ext_offset_enabled = _config_effective_ext_offset_enabled(cfg)

        tifxyz_dir = _decode_tifxyz_for_request(
            body=body,
            cfg=cfg,
            args_section=args_section_pre,
            tmp_dir=tmp_dir,
            model_init=model_init,
            ext_offset_enabled=ext_offset_enabled,
        )

        if model_init == "model" and not model_input:
            raise ValueError("model-init=model requires request model_data or model_input")

        if ext_offset_enabled and tifxyz_dir is not None and "external_surfaces" not in cfg:
            offset_val = float(cfg.pop("offset_value", 1.0))
            cfg["external_surfaces"] = [{"path": tifxyz_dir, "offset": offset_val}]

        args_section = dict(cfg.get("args", {}))
        args_section["input"] = str(data_input)
        args_section.setdefault("sparse-prefetch-backend", _sparse_prefetch_backend)
        if model_input:
            args_section["model-input"] = str(model_input)
        args_section["model-output"] = str(model_output)
        # Only set fit.py out-dir if explicitly requested. pred_dt_flow_gate
        # debug slices use their own debug_out_dir so enabling them does not
        # make fit.py export model_final/tifxyz into the service cwd.
        if body.get("out_dir"):
            args_section["out-dir"] = str(body["out_dir"])
        elif _config_enables_pred_dt_flow_gate(cfg):
            _set_pred_dt_flow_gate_debug_out_dir(cfg, str(service_workdir))
        cfg["args"] = args_section
        cfg_path = str(Path(tmp_dir) / "fit_config.json")
        has_corr = "corr_points" in cfg
        n_corr_cols = len(cfg["corr_points"].get("collections", {})) if has_corr and isinstance(cfg.get("corr_points"), dict) else 0
        print(f"[fit-service] writing config: corr_points={has_corr} ({n_corr_cols} collections)", flush=True)
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

            def _wrapped_progress(*, step: int, total: int, loss: float, **kw: Any) -> None:
                _job.set_running(
                    "optimizing", step, total, loss,
                    stage_progress=float(kw.get("stage_progress", 0.0)),
                    overall_progress=float(kw.get("overall_progress", 0.0)),
                    stage_name=str(kw.get("stage_name", "")),
                )
                if _job.cancelled:
                    raise KeyboardInterrupt("cancelled by user")
                if orig_progress is not None:
                    orig_progress(step=step, total=total, loss=loss, **kw)

            kwargs["snapshot_fn"] = _wrapped_snapshot
            kwargs["progress_fn"] = _wrapped_progress
            return _orig_optimize(**kwargs)

        from contextlib import nullcontext
        from gpu_pause import gpu_pause_context

        opt_mod.optimize = _patched_optimize
        with (gpu_pause_context() if _gpu_pause_enabled else nullcontext()):
            try:
                import fit as fit_mod
                _job.set_running("loading", 0, 0, 0.0)
                fit_mod.main([cfg_path])
            finally:
                opt_mod.optimize = _orig_optimize

            if _job.cancelled:
                _job.set_error("cancelled")
                return

            # Export to tifxyz — skip if windowed mode already exported.
            # Windowed mode exports .tifxyz dirs into the parent of
            # model_output (= tmp_dir). Move them to output_dir.
            _window_tifxyz = [p for p in Path(tmp_dir).iterdir()
                              if p.name.endswith(".tifxyz") and p.is_dir()]
            if _window_tifxyz:
                import shutil as _shutil
                _win_base = body.get("output_name", "")
                if _win_base.endswith(".tifxyz"):
                    _win_base = _win_base[:-len(".tifxyz")]
                for i, p in enumerate(sorted(_window_tifxyz, key=lambda x: x.name)):
                    dst_name = f"{_win_base}_w{i}.tifxyz" if _win_base else p.name
                    dst = Path(output_dir) / dst_name
                    if dst.exists():
                        _shutil.rmtree(dst)
                    _shutil.move(str(p), str(dst))
                    # Update UUID in meta.json to match the new directory name
                    _meta_path = dst / "meta.json"
                    if _meta_path.exists():
                        import json as _json
                        _meta = _json.loads(_meta_path.read_text(encoding="utf-8"))
                        _meta["uuid"] = dst_name
                        _meta_path.write_text(_json.dumps(_meta, indent=2) + "\n", encoding="utf-8")
                print(f"[fit-service] windowed mode: moved {len(_window_tifxyz)} "
                      f"window tifxyz to {output_dir}", flush=True)
            else:
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
                voxel_size_um = config.get("voxel_size_um")
                if voxel_size_um is not None:
                    export_argv.extend(["--voxel-size-um", str(float(voxel_size_um))])
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

        elif self.path == "/export_vis":
            try:
                body = self._read_json()
            except Exception as exc:
                self._send_json({"error": f"bad json: {exc}"}, 400)
                return
            self._handle_export_vis(body)

        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_export_vis(self, body: dict[str, Any]) -> None:
        """Synchronously export multi-layer OBJ visualization.

        Returns the exported files as a tar.gz binary response.
        """
        import base64
        import io
        import shutil
        import tarfile
        import tempfile

        model_input = body.get("model_input")
        model_data = body.get("model_data")
        data_input = body.get("data_input")

        tmp_model = None
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="fit_vis_")

            if model_data:
                model_bytes = base64.b64decode(model_data)
                tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
                tmp.write(model_bytes)
                tmp.close()
                model_input = tmp.name
                tmp_model = tmp.name
            elif not model_input:
                self._send_json({"error": "missing model_input or model_data"}, 400)
                return

            # If data_input not provided, extract from checkpoint's _fit_config_
            if not data_input:
                import torch
                st = torch.load(str(model_input), map_location="cpu", weights_only=False)
                fit_cfg = st.get("_fit_config_", {}) or {}
                fit_args = fit_cfg.get("args", {}) or {}
                data_input = fit_args.get("input")
                if not data_input:
                    self._send_json({"error": "missing 'data_input' and checkpoint has no _fit_config_.args.input"}, 400)
                    return
                # Resolve relative paths against _data_dir if available
                if _data_dir and not Path(data_input).is_absolute():
                    candidate = Path(_data_dir) / data_input
                    if candidate.exists():
                        data_input = str(candidate)

            import lasagna_analyze
            from contextlib import nullcontext
            from gpu_pause import gpu_pause_context
            with (gpu_pause_context() if _gpu_pause_enabled else nullcontext()):
                lasagna_analyze.export_vis_obj(
                    model_path=str(model_input),
                    data_path=str(data_input),
                    output_dir=tmp_dir,
                    slices=body.get("slices", []),
                    channels=body.get("channels", []),
                    losses=body.get("losses", []),
                    include_mesh=bool(body.get("include_mesh", True)),
                    include_connections=bool(body.get("include_connections", True)),
                    device=body.get("device", "cuda"),
                )

            # Package as tar.gz
            buf = io.BytesIO()
            out_path = Path(tmp_dir)
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for child in sorted(out_path.iterdir()):
                    tar.add(str(child), arcname=child.name)

            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/gzip")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

            print(f"[fit-service] export_vis done ({len(data)} bytes)", flush=True)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[fit-service] export_vis error: {tb}", file=sys.stderr, flush=True)
            self._send_json({"error": str(exc)}, 500)
        finally:
            if tmp_model:
                try:
                    os.unlink(tmp_model)
                except OSError:
                    pass
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def log_message(self, fmt: str, *args: Any) -> None:
        msg = fmt % args
        if "/status" in msg:
            return
        print(f"[fit-service] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _data_dir, _gpu_pause_enabled, _sparse_prefetch_backend

    p = argparse.ArgumentParser(description="Fit optimizer HTTP service for VC3D")
    p.add_argument("--port", type=int, default=9999, help="Port (default 9999)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--data-dir", default=None,
                   help="Directory containing .lasagna.json datasets")
    p.add_argument("--no-gpu-pause", action="store_true", default=False,
                   help="Disable automatic GPU pause/resume of training")
    p.add_argument("--sparse-prefetch-backend",
                   choices=("tensorstore", "python-zarr"),
                   default="tensorstore",
                   help="Sparse streaming prefetch backend for fit jobs")
    args = p.parse_args()

    if args.data_dir:
        _data_dir = str(Path(args.data_dir).resolve())
    if args.no_gpu_pause:
        _gpu_pause_enabled = False
    _sparse_prefetch_backend = str(args.sparse_prefetch_backend)

    datasets = _list_datasets()
    if not datasets:
        data_dir_msg = _data_dir if _data_dir else "<not set>"
        print(
            f"[fit-service] error: no .lasagna.json datasets found in --data-dir {data_dir_msg}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)

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
