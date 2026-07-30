"""Resident adapter around the legacy fitter's optimizer loop.

Only the fitter thread calls Torch/CUDA. Other threads communicate through a
condition variable and consume copied status snapshots.
"""

from __future__ import annotations

import copy
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
from typing import Mapping
import uuid

from fit_session import (SpiralInputPaths, SpiralPreviewConfig, SpiralRunConfig,
                         run_mutable_config)
from config import Config
from spiral_progress import NullProgressReporter, ProgressReporter


_NO_PROGRESS = NullProgressReporter()


class _SessionShutdown(BaseException):
    pass


class InteractiveFitSession:
    def __init__(self, paths: SpiralInputPaths, run: SpiralRunConfig,
                 preview: SpiralPreviewConfig, status_callback=None,
                 publishes_outputs=True) -> None:
        self.paths = paths
        self.run_config = run
        self.preview_config = preview
        self.input_manifest = paths.manifest()
        self.requested_config = dict(run.config)
        self._applied_config = None
        self._run_config = None
        self._run_config_limits = None
        self._default_advanced_config = None
        self._status_callback = status_callback
        self.publishes_outputs = publishes_outputs
        self._condition = threading.Condition()
        self._state = "Loading"
        self._phase = "Importing fitter"
        self._error = None
        self._warnings = []
        self._completed = 0
        self._target = 0
        self._pending = 0
        self._stop_requested = False
        self._shutdown = False
        self._latest_metrics = {}
        self._output_path = paths.output_directory
        self._preview_manifest = None
        self._preview_generation = 0
        self._preview_session_id = uuid.uuid4().hex
        self._save_checkpoint = None
        self._export_preview = None
        self._incorporate_inputs = None
        self._finish_run = None
        self._configure_run = None
        self._idle_actions = []
        self.progress = ProgressReporter(
            self._progress_changed,
            stream=sys.stdout,
        )
        self._run_start_completed = 0
        self._thread = threading.Thread(target=self._fit_main, name="spiral-fit-worker", daemon=True)
        self._thread.start()

    @property
    def completed_iterations(self):
        with self._condition:
            return self._completed

    def status(self):
        with self._condition:
            result = {
                "state": self._state, "phase": self._phase,
                "current_iteration": self._completed,
                "target_iteration": self._target,
                "session_horizon": None,
                "latest_metrics": copy.deepcopy(self._latest_metrics),
                "warnings": list(self._warnings), "error": self._error,
                "preview_manifest_path": self._preview_manifest,
                "preview_generation": self._preview_generation,
                "supports_input_incorporation": self._incorporate_inputs is not None,
                "input_manifest": copy.deepcopy(self.input_manifest),
                "progress": self._progress_reporter().snapshot(),
            }
            if self._run_config is not None:
                result["run_config"] = copy.deepcopy(self._run_config)
            if self._run_config_limits is not None:
                result["run_config_limits"] = copy.deepcopy(
                    self._run_config_limits)
            if self._default_advanced_config is not None:
                result["default_advanced_config"] = copy.deepcopy(
                    self._default_advanced_config)
            if getattr(self, "_applied_config", None) is not None:
                result["applied_config"] = copy.deepcopy(self._applied_config)
            return result

    def _publish_status(self):
        if self._status_callback is not None:
            self._status_callback(self.status())

    def _progress_reporter(self):
        return getattr(self, "progress", _NO_PROGRESS)

    def _progress_changed(self, snapshot):
        if snapshot is not None:
            with self._condition:
                self._phase = str(snapshot.get("stage_name") or self._phase)
        self._publish_status()

    def _set_state(self, state, phase=""):
        with self._condition:
            self._state = state
            self._phase = phase or state
            self._condition.notify_all()
        self._publish_status()

    def _fit_main(self):
        fitter = None
        wandb = None
        distributed_initialized = False
        try:
            self._progress_reporter().begin("loading", "Importing Torch and fitter")
            self._set_state("Loading", "Importing Torch and fitter")
            import wandb
            import fit_spiral as fitter
            from ddp_helpers import (maybe_destroy_distributed,
                                     maybe_init_distributed,
                                     split_counts_across_ranks)
            from spiral_helpers import (
                SAMPLING_COUNT_FLOORS, scale_counts_for_z_range)

            maybe_init_distributed()
            distributed_initialized = True

            config = Config().as_dict()
            checkpoint_profile_config = None
            if self.paths.checkpoint:
                self._progress_reporter().begin(
                    "loading", "Reading checkpoint configuration",
                    detail=Path(self.paths.checkpoint).name)
                from checkpoint_io import load_checkpoint_cpu
                checkpoint_config = load_checkpoint_cpu(self.paths.checkpoint)
                try:
                    if not isinstance(checkpoint_config, dict) or not isinstance(
                            checkpoint_config.get('cfg'), Mapping):
                        raise ValueError("Checkpoint has no current Spiral configuration")
                    durable = dict(checkpoint_config['cfg'])
                    if set(durable) != set(config):
                        raise ValueError(
                            "Checkpoint configuration does not match the current schema")
                    durable = Config(durable).as_dict()
                    config.update(durable)
                    # The session-scoped profile initially reproduces the
                    # checkpoint without applying scaling twice.
                    checkpoint_profile_config = copy.deepcopy(durable)
                finally:
                    # This first load exists only to resolve configuration.  Do
                    # not retain a complete model + optimiser checkpoint for the
                    # lifetime of the resident fitter thread.
                    del checkpoint_config
            unknown = sorted(set(self.run_config.config) - set(config))
            if unknown:
                raise ValueError(f"Unknown advanced config keys: {unknown}")
            if checkpoint_profile_config is not None:
                default_advanced_config = checkpoint_profile_config
            else:
                # Without a checkpoint, Default is the Python baseline.
                default_advanced_config = copy.deepcopy(config)
            # Explicit sample-count overrides are literal active counts. This
            # lets VC3D round-trip the host's post-scaling values through a
            # reload without applying the z-range/DDP transforms twice.
            # Checkpoint cfg values are resolved fitter values too, so give
            # their counts the same treatment.
            explicit_sampling_counts = {
                key: value for key, value in (checkpoint_profile_config or {}).items()
                if key.startswith("sample_count_")
            }
            explicit_sampling_counts.update({
                key: value for key, value in self.run_config.config.items()
                if key.startswith("sample_count_")
            })
            config.update(self.run_config.config)
            fields = Config.catalog()["schema"]["fields"]
            count_keys = tuple(
                key for key, spec in fields.items() if spec.get("scale_with_z"))
            scale_counts_for_z_range(
                config, self.run_config.z_begin, self.run_config.z_end,
                9500, count_keys, floors=SAMPLING_COUNT_FLOORS)
            split_counts_across_ranks(config, count_keys)
            if checkpoint_profile_config is None:
                scale_counts_for_z_range(
                    default_advanced_config, self.run_config.z_begin,
                    self.run_config.z_end, 9500, count_keys,
                    floors=SAMPLING_COUNT_FLOORS)
                split_counts_across_ranks(default_advanced_config, count_keys)
            config.update(explicit_sampling_counts)
            self.requested_config = dict(config)
            with self._condition:
                self._applied_config = copy.deepcopy(config)
                self._run_config = run_mutable_config(config)
                self._run_config_limits = {
                    'track_max_track_crossing_per_step': max(
                        int(config.get('track_crossing_precompute_max', 0)),
                        int(config.get('track_max_track_crossing_per_step', 0))),
                }
                self._default_advanced_config = default_advanced_config
            self._publish_status()

            fitter.dataset_path = self.paths.dataset_root
            fitter.normal_nx_zarr_path = self.paths.normal_x or None
            fitter.normal_ny_zarr_path = self.paths.normal_y or None
            fitter.grad_mag_zarr_path = self.paths.gradient_magnitude or None
            fitter.surf_sdt_zarr_path = self.paths.surf_sdt or None
            fitter.normal_zarr_group = self.run_config.lasagna_group
            fitter.pcl_input_specs = [(spec.path, spec.role.value) for spec in self.paths.pcls]
            fitter.pcl_json_paths = [spec.path for spec in self.paths.pcls]
            fitter.fibers_path = self.paths.fibers or None
            fitter.verified_patches_path = self.paths.verified_patches or None
            fitter.unverified_patches_path = self.paths.unverified_patches or None
            fitter.shell_path = self.paths.outer_shell or None
            fitter.tracks_dbm_path = self.paths.tracks_dbm or None
            fitter.scroll_zarr_path = self.paths.scroll_zarr or None
            fitter.spiral_outward_sense = self.run_config.outward_sense
            fitter.scroll_name = self.run_config.scroll_name
            fitter.z_begin = self.run_config.z_begin
            fitter.z_end = self.run_config.z_end
            fitter.voxel_size_um = self.run_config.voxel_size_um
            fitter.lasagna_scale = self.run_config.lasagna_scale
            fitter.lasagna_storage_backend = self.run_config.storage_backend
            fitter.cache_path = self.paths.cache_directory
            fitter.render_volume_scale = self.run_config.render_volume_scale
            fitter.run_tag = self.run_config.run_tag or None
            fitter.umbilicus_z_to_yx = lambda: fitter.json_umbilicus_z_to_yx(
                self.paths.umbilicus, coordinate_scale=1.0)
            os.environ['FIT_SPIRAL_OUT_DIR'] = self.paths.output_directory
            if self.paths.checkpoint:
                os.environ['FIT_SPIRAL_RESUME_PATH'] = self.paths.checkpoint
                os.environ['FIT_SPIRAL_RESUME_STEP'] = str(self.run_config.legacy_checkpoint_step)
            else:
                os.environ.pop('FIT_SPIRAL_RESUME_PATH', None)
                os.environ.pop('FIT_SPIRAL_RESUME_STEP', None)

            wandb.init(project='scrolls', config=config, mode='disabled')
            fitter.cfg = wandb.config
            fitter.configure_losses(fitter.cfg, fitter.z_begin, fitter.z_end)
            self._progress_reporter().begin("loading", "Loading fit inputs and model")
            self._set_state("Loading", "Loading fit inputs and model")
            fitter.main(interactive_driver=self, progress=self.progress)
        except BaseException as exc:
            with self._condition:
                if self._shutdown and isinstance(exc, _SessionShutdown):
                    self._state, self._phase = "Empty", "Stopped"
                else:
                    self._state, self._phase = "Error", "Error"
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._warnings.append(traceback.format_exc(limit=12))
                self._condition.notify_all()
            self._publish_status()
        finally:
            if fitter is not None:
                fitter.release_interactive_resources()
            if wandb is not None:
                wandb.finish(quiet=True)
            if distributed_initialized:
                maybe_destroy_distributed()
            self._progress_reporter().close()

    # Fitter-thread callbacks.
    def on_ready(self, *, completed_iterations, output_path,
                 save_checkpoint, export_preview,
                 incorporate_inputs=None, finish_run=None, configure_run=None):
        with self._condition:
            self._completed = self._target = completed_iterations
            self._output_path = output_path
            self._save_checkpoint = save_checkpoint
            self._export_preview = export_preview
            self._incorporate_inputs = incorporate_inputs
            self._finish_run = finish_run
            self._configure_run = configure_run
        if self.paths.checkpoint and getattr(self, "publishes_outputs", True):
            self._set_state("ExportingPreview", "Exporting restored checkpoint preview")
            self._progress_reporter().begin(
                "exporting_preview", "Exporting restored checkpoint preview")
            self._publish_preview()
        self._progress_reporter().clear()
        self._set_state("Ready", "Ready")

    def wait_for_iteration(self, iteration):
        while True:
            with self._condition:
                if self._shutdown:
                    raise _SessionShutdown()
                # Idle actions are drained before the pending check so inputs
                # queued by run() are incorporated before the next step begins.
                action = self._idle_actions.pop(0) if self._idle_actions else None
                if action is None:
                    if self._pending > 0:
                        return True
                    self._condition.wait()
                    continue
            if action[0] == "incorporate":
                _, records, mark_incorporated, influence_config = action
                self._run_incorporation(records, mark_incorporated, influence_config)
                continue
            if action[0] == "configure":
                self._run_configuration(action[1], action[2], action[3])
                continue
            kind, path, done, result = action
            try:
                if kind == "save":
                    with self._condition:
                        previous_state = self._state
                        self._state, self._phase = "Saving", "Saving checkpoint"
                    self._progress_reporter().begin(
                        "saving_checkpoint", "Saving checkpoint",
                        detail=Path(path).name)
                    result["path"] = self._save_checkpoint(path, self._completed)
                    self._progress_reporter().finish()
                else:
                    result["error"] = f"Unknown idle action {kind}"
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if kind == "save":
                    self._progress_reporter().clear()
                    with self._condition:
                        self._state = previous_state
                        self._phase = previous_state
                    self._publish_status()
                done.set()

    def _run_incorporation(self, records, mark_incorporated, influence_config):
        """Append newly uploaded ephemeral inputs to the resident fit.

        Runs on the fitter thread at the pause boundary. A failure cancels the
        queued run and surfaces a warning instead of tearing down the session.
        """
        try:
            if self._incorporate_inputs is None:
                raise RuntimeError(
                    "The resident fitter does not support adding inputs to a running session")
            with self._condition:
                self._phase = "Incorporating new session inputs"
            self._progress_reporter().begin(
                "incorporating_inputs", "Incorporating new session inputs",
                step=0, total_steps=len(records), unit="inputs")
            self._publish_status()
            self._incorporate_inputs(records, influence_config)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._progress_reporter().clear()
            with self._condition:
                self._pending = 0
                self._target = self._completed
                self._warnings.append(f"Input incorporation failed: {error}")
                self._state, self._phase = "Paused", "Paused"
                self._condition.notify_all()
            if mark_incorporated is not None:
                mark_incorporated(records, error=error)
            self._publish_status()
        else:
            if mark_incorporated is not None:
                mark_incorporated(records)
            with self._condition:
                if self._state == "Running":
                    self._phase = "Optimizing"
            if getattr(self, "_state", None) == "Running":
                self._begin_optimization_progress()
            else:
                self._progress_reporter().clear()
            self._publish_status()

    def _run_configuration(
            self, config, path_changes=None, previous_run_config=None):
        """Apply validated Run-scoped settings on the fitter thread."""
        try:
            if self._configure_run is None:
                raise RuntimeError(
                    "The resident fitter does not support Run configuration changes")
            path_changes = dict(path_changes or {})
            self._progress_reporter().begin(
                "configuring", "Applying run configuration",
                detail=(
                    f"{len(config)} settings, {len(path_changes)} path changes"
                ))
            if path_changes:
                self._configure_run(dict(config), dict(path_changes))
            else:
                self._configure_run(dict(config))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._progress_reporter().clear()
            with self._condition:
                self._pending = 0
                self._target = self._completed
                # Leave newly uploaded inputs pending for a later valid Run.
                self._idle_actions = [
                    action for action in self._idle_actions
                    if action[0] != "incorporate"
                ]
                self._warnings.append(f"Run configuration failed: {error}")
                if previous_run_config is not None:
                    self._run_config.update(previous_run_config)
                self._state, self._phase = "Paused", "Paused"
                self._condition.notify_all()
            self._publish_status()
        else:
            if getattr(self, "_applied_config", None) is not None:
                with self._condition:
                    self._applied_config.update(config)
                    self._run_config.update(config)
                    self.requested_config.update(config)
                    self.input_manifest.update(path_changes)
            if getattr(self, "_state", None) == "Running":
                self._begin_optimization_progress()
            else:
                self._progress_reporter().clear()

    def _begin_optimization_progress(self):
        with self._condition:
            run_start = getattr(
                self, "_run_start_completed", self._completed)
            step = max(0, self._completed - run_start)
            total = max(0, self._target - run_start)
        self._progress_reporter().begin(
            "optimizing", "Optimizing",
            step=step, total_steps=total, unit="iterations")

    def iteration_completed(self, *, completed_iterations, total_loss, losses, learning_rate, metrics=None):
        with self._condition:
            self._completed = completed_iterations
            self._latest_metrics = {"total_loss": total_loss, "losses": dict(losses),
                                    "learning_rate": learning_rate, **dict(metrics or {})}
            self._pending = max(0, self._pending - 1)
            if self._stop_requested:
                self._pending = 0
                self._stop_requested = False
            pause = self._pending == 0
            run_start = getattr(
                self, "_run_start_completed",
                self._completed - max(0, self._target - self._completed))
            run_step = max(0, self._completed - run_start)
        self._progress_reporter().update(run_step)
        self._publish_status()
        if pause:
            if self._finish_run is not None:
                self._finish_run()
            if not getattr(self, "publishes_outputs", True):
                self._progress_reporter().clear()
                self._set_state("Paused", "Paused")
                return
            self._set_state("Saving", "Autosaving checkpoint")
            self._progress_reporter().begin(
                "saving_checkpoint", "Autosaving checkpoint",
                detail="checkpoint_autosave.ckpt")
            autosave = str(Path(self._output_path) / "checkpoint_autosave.ckpt")
            self._save_checkpoint(autosave, self._completed)
            self._set_state("ExportingPreview", "Exporting preview")
            self._progress_reporter().begin(
                "exporting_preview", "Exporting preview")
            self._publish_preview()
            self._progress_reporter().clear()
            self._set_state("Paused", "Paused")

    def _publish_preview(self):
        with self._condition:
            generation = self._preview_generation + 1
        generation_path = (Path(self.paths.output_directory) / ".spiral-preview" /
                           self._preview_session_id / f"generation-{generation}")
        surface_id = f"spiral-output-generation-{generation}"
        manifest = self._export_preview(str(generation_path), surface_id)
        with self._condition:
            self._preview_generation = generation
            self._preview_manifest = str(manifest["manifest_path"])
        # Publish while the session is still in ExportingPreview.  The host
        # service synchronously Lasagna-flattens and packages this generation
        # from the status callback, so clients cannot start another Run while
        # the downloadable preview is still being prepared.
        self._publish_status()

    def session_finished(self):
        raise RuntimeError("Interactive optimizer loop ended unexpectedly")

    # Coordinator-thread commands.
    def run(self, count, pending_inputs=None, mark_incorporated=None,
            influence_config=None, run_config=None, path_changes=None):
        if count < 1:
            raise ValueError("iterations must be at least 1")
        with self._condition:
            if self._state not in {"Ready", "Paused"}:
                raise RuntimeError(f"Run is not allowed while session state is {self._state}")
            run_config = dict(run_config or {})
            path_changes = dict(path_changes or {})
            target = self._completed + count
            requested_config = dict(
                getattr(self, "requested_config", {}) or {})
            requested_config.update(run_config)
            configured_horizon = int(
                requested_config.get("optimizer_num_training_steps", 0) or 0)
            # Interactive sessions are allowed to continue beyond the original
            # headless horizon. Each Run extends the schedule by its requested
            # iterations, based on at least the durable completed step, so the
            # fitter can realign the exponential LR curve exactly.
            if getattr(self, "_configure_run", None) is not None:
                run_config["optimizer_num_training_steps"] = (
                    max(configured_horizon, self._completed) + count)
            if run_config or path_changes:
                if self._configure_run is None:
                    raise RuntimeError(
                        "The resident fitter does not support Run configuration changes")
                requested_config = dict(self.requested_config)
                requested_config.update(run_config)
                run_config = {
                    key: requested_config[key]
                    for key in run_config
                }
                previous_run_config = {
                    key: self._run_config.get(key)
                    for key in run_config
                }
                self._idle_actions.append(
                    ("configure", run_config, path_changes,
                     previous_run_config))
                self._run_config.update(run_config)
            if pending_inputs:
                if self._incorporate_inputs is None:
                    raise RuntimeError(
                        "The resident fitter does not support adding inputs to a running session")
                self._idle_actions.append(
                    ("incorporate", list(pending_inputs), mark_incorporated,
                     dict(influence_config or {})))
            self._pending = count
            self._run_start_completed = self._completed
            self._target = target
            self._state, self._phase = "Running", "Optimizing"
            self._begin_optimization_progress()
            self._condition.notify_all()
            return self._target

    def stop(self):
        with self._condition:
            if self._state != "Running":
                raise RuntimeError("Session is not running")
            self._stop_requested = True

    def save_checkpoint(self, path, timeout=120.0):
        with self._condition:
            if self._state not in {"Ready", "Paused"}:
                raise RuntimeError(f"Checkpoint save is not allowed in {self._state}")
            done = threading.Event()
            result = {}
            self._idle_actions.append(("save", path, done, result))
            self._condition.notify_all()
        if not done.wait(timeout):
            raise TimeoutError("Checkpoint save timed out")
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["path"]

    def close(self, timeout=15.0):
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("Spiral fitter did not stop at a safe boundary")


def _free_loopback_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _distributed_session_worker(rank, world_size, gpu_id, master_port,
                                paths, run, preview, commands, events):
    """Own one CUDA rank and adapt queue commands to InteractiveFitSession."""
    os.environ.update({
        # Give each rank a one-device CUDA namespace. This prevents checkpoint
        # RNG snapshots and other process-global CUDA helpers from opening
        # contexts on GPUs owned by sibling ranks.
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(master_port),
        "WORLD_SIZE": str(world_size),
        "RANK": str(rank),
        "LOCAL_RANK": "0",
    })

    def publish_status(status):
        events.put(("status", rank, status))

    session = None
    closed = False
    try:
        session = InteractiveFitSession(
            paths, run, preview, publish_status, publishes_outputs=(rank == 0))
        while True:
            command_id, name, arguments = commands.get()
            try:
                if name == "run":
                    mark_incorporated = None
                    if rank == 0 and arguments.get("pending_inputs"):
                        def mark_incorporated(records, error=None, cid=command_id):
                            events.put(("incorporated", cid, error))
                    result = session.run(
                        arguments["count"],
                        pending_inputs=arguments.get("pending_inputs"),
                        mark_incorporated=mark_incorporated,
                        influence_config=arguments.get("influence_config"),
                        run_config=arguments.get("run_config"),
                        path_changes=arguments.get("path_changes"),
                    )
                elif name == "stop":
                    result = session.stop()
                elif name == "save_checkpoint":
                    result = session.save_checkpoint(
                        arguments["path"], arguments.get("timeout", 120.0))
                elif name == "close":
                    session.close(arguments.get("timeout", 15.0))
                    closed = True
                    result = None
                else:
                    raise ValueError(f"Unknown distributed session command {name}")
            except BaseException as exc:
                events.put(("ack", command_id, rank, False,
                            f"{type(exc).__name__}: {exc}"))
            else:
                events.put(("ack", command_id, rank, True, result))
            if name == "close":
                return
    except BaseException as exc:
        events.put(("worker_error", rank,
                    f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=12)))
    finally:
        if session is not None and not closed:
            try:
                session.close()
            except BaseException:
                pass


class DistributedInteractiveFitSession:
    """Parent-process proxy for one resident fitter process per selected GPU."""

    def __init__(self, paths, run, preview, gpu_ids, status_callback=None):
        self._gpu_ids = tuple(gpu_ids)
        self._status_callback = status_callback
        self._condition = threading.Condition()
        self._status = {
            "state": "Loading", "phase": "Starting GPU workers",
            "current_iteration": 0, "target_iteration": 0,
            "session_horizon": None, "latest_metrics": {}, "warnings": [],
            "error": None, "preview_manifest_path": None,
            "preview_generation": 0,
            "supports_input_incorporation": False,
            "progress": {
                "operation": "loading",
                "stage_name": "Starting GPU workers",
                "detail": None,
                "step": 0,
                "total_steps": len(self._gpu_ids),
                "unit": "workers",
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
            },
        }
        self._acks = {}
        self._incorporation_callbacks = {}
        self._rank_statuses = {}
        self._failed_error = None
        self._closed = False
        context = multiprocessing.get_context("spawn")
        self._events = context.Queue()
        self._commands = [context.Queue() for _ in self._gpu_ids]
        master_port = _free_loopback_port()
        self._processes = [
            context.Process(
                target=_distributed_session_worker,
                args=(rank, len(self._gpu_ids), gpu_id, master_port,
                      paths, run, preview, self._commands[rank], self._events),
                name=f"spiral-gpu-{gpu_id}",
            )
            for rank, gpu_id in enumerate(self._gpu_ids)
        ]
        self._listener = threading.Thread(
            target=self._listen, name="spiral-gpu-coordinator", daemon=True)
        self._listener.start()
        started = []
        try:
            for process in self._processes:
                process.start()
                started.append(process)
        except BaseException:
            for process in started:
                process.terminate()
            for process in started:
                process.join(5.0)
            self._events.put(None)
            self._listener.join(5.0)
            raise

    @property
    def completed_iterations(self):
        return self.status()["current_iteration"]

    def status(self):
        with self._condition:
            return copy.deepcopy(self._status)

    def _listen(self):
        while True:
            event = self._events.get()
            if event is None:
                return
            kind = event[0]
            callback = None
            snapshot = None
            if kind == "status":
                _, rank, status = event
                with self._condition:
                    self._rank_statuses[rank] = status
                    if self._failed_error is not None and status.get("state") != "Error":
                        continue
                    if status.get("state") == "Error":
                        if rank == 0:
                            self._status = status
                        else:
                            warnings = list(self._status.get("warnings", []))
                            warnings.extend(status.get("warnings", []))
                            self._status.update({
                                "state": "Error", "phase": "Error",
                                "error": f"GPU worker rank {rank}: {status.get('error')}",
                                "warnings": warnings,
                            })
                        self._failed_error = self._status.get("error") or \
                            f"GPU worker rank {rank} failed"
                        self._condition.notify_all()
                    else:
                        if rank == 0:
                            self._status = status
                        elif 0 not in self._rank_statuses:
                            continue

                        rank_zero = self._rank_statuses.get(0, {})
                        ready_states = {"Ready", "Paused"}
                        all_ranks_ready = (
                            len(self._rank_statuses) == len(self._gpu_ids)
                            and all(item.get("state") in ready_states
                                    for item in self._rank_statuses.values())
                        )
                        if rank_zero.get("state") in ready_states and not all_ranks_ready:
                            unfinished = next(
                                ((worker_rank, item)
                                 for worker_rank, item
                                 in sorted(self._rank_statuses.items())
                                 if item.get("state") not in ready_states),
                                None)
                            self._status = copy.deepcopy(rank_zero)
                            self._status.update({
                                "state": "Loading",
                                "phase": "Waiting for all GPU workers",
                            })
                            if unfinished is not None:
                                worker_rank, worker_status = unfinished
                                worker_progress = copy.deepcopy(
                                    worker_status.get("progress"))
                                if worker_progress:
                                    detail = worker_progress.get("detail")
                                    worker_progress["detail"] = (
                                        f"GPU worker {worker_rank + 1}/"
                                        f"{len(self._gpu_ids)}"
                                        + (f" — {detail}" if detail else ""))
                                    self._status["progress"] = worker_progress
                                    self._status["phase"] = str(
                                        worker_progress.get("stage_name")
                                        or self._status["phase"])
                        elif all_ranks_ready:
                            # Rank zero owns user-facing metrics and artifacts.
                            # A later secondary Ready event completes startup.
                            self._status = copy.deepcopy(rank_zero)
                        elif rank != 0:
                            continue
                    snapshot = copy.deepcopy(self._status)
                callback = self._status_callback
            elif kind == "ack":
                _, command_id, rank, ok, result = event
                with self._condition:
                    self._acks.setdefault(command_id, {})[rank] = (ok, result)
                    self._condition.notify_all()
            elif kind == "incorporated":
                _, command_id, error = event
                with self._condition:
                    pending_callback = self._incorporation_callbacks.pop(command_id, None)
                if pending_callback is not None:
                    callback, records = pending_callback
                    callback(records, error=error) if error else callback(records)
                continue
            elif kind == "worker_error":
                _, rank, error, trace = event
                with self._condition:
                    warnings = list(self._status.get("warnings", []))
                    warnings.append(f"GPU worker rank {rank} failed:\n{trace}")
                    self._status.update({
                        "state": "Error", "phase": "Error", "error": error,
                        "warnings": warnings,
                    })
                    self._failed_error = error
                    snapshot = copy.deepcopy(self._status)
                    self._condition.notify_all()
                callback = self._status_callback
            if callback is not None:
                callback(snapshot)

    def _call(self, name, arguments=None, ranks=None, timeout=30.0,
              incorporation_callback=None):
        if self._closed and name != "close":
            raise RuntimeError("Spiral fit session is closed")
        if self._failed_error is not None and name != "close":
            raise RuntimeError(self._failed_error)
        ranks = tuple(range(len(self._processes))) if ranks is None else tuple(ranks)
        command_id = uuid.uuid4().hex
        if incorporation_callback is not None:
            with self._condition:
                records = list((arguments or {}).get("pending_inputs", []))
                self._incorporation_callbacks[command_id] = (
                    incorporation_callback, records)
        for rank in ranks:
            self._commands[rank].put((command_id, name, dict(arguments or {})))
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self._acks.get(command_id, {})) < len(ranks):
                if self._failed_error is not None:
                    self._incorporation_callbacks.pop(command_id, None)
                    raise RuntimeError(self._failed_error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._incorporation_callbacks.pop(command_id, None)
                    raise TimeoutError(f"Timed out waiting for GPU workers to {name}")
                self._condition.wait(remaining)
            responses = self._acks.pop(command_id)
            failures = [f"rank {rank}: {responses[rank][1]}" for rank in ranks
                        if not responses[rank][0]]
            if failures:
                self._incorporation_callbacks.pop(command_id, None)
                raise RuntimeError("; ".join(failures))
            return responses[ranks[0]][1]

    def run(self, count, pending_inputs=None, mark_incorporated=None,
            influence_config=None, run_config=None, path_changes=None):
        state = self.status()["state"]
        if state not in {"Ready", "Paused"}:
            raise RuntimeError(f"Run is not allowed while session state is {state}")
        arguments = {
            "count": count,
            "pending_inputs": list(pending_inputs or []),
            "influence_config": dict(influence_config or {}),
            "run_config": dict(run_config or {}),
            "path_changes": dict(path_changes or {}),
        }
        return self._call("run", arguments, timeout=30.0,
                          incorporation_callback=mark_incorporated)

    def stop(self):
        state = self.status()["state"]
        if state != "Running":
            raise RuntimeError(f"Session is not running (state is {state})")
        return self._call("stop")

    def save_checkpoint(self, path, timeout=120.0):
        state = self.status()["state"]
        if state not in {"Ready", "Paused"}:
            raise RuntimeError(f"Checkpoint save is not allowed in {state}")
        return self._call("save_checkpoint", {"path": path, "timeout": timeout},
                          ranks=(0,), timeout=timeout + 5.0)

    def close(self, timeout=15.0):
        if self._closed:
            return
        if self._failed_error is not None:
            self._closed = True
            for process in self._processes:
                if process.is_alive():
                    process.terminate()
            for process in self._processes:
                process.join(5.0)
            self._events.put(None)
            self._listener.join(5.0)
            return
        try:
            self._call("close", {"timeout": timeout}, timeout=timeout + 5.0)
        finally:
            self._closed = True
            deadline = time.monotonic() + timeout
            for process in self._processes:
                process.join(max(0.0, deadline - time.monotonic()))
            alive = [process for process in self._processes if process.is_alive()]
            if alive:
                for process in alive:
                    process.terminate()
                for process in alive:
                    process.join(5.0)
                raise TimeoutError("Spiral GPU workers did not stop at a safe boundary")
            self._events.put(None)
            self._listener.join(5.0)


def create_session(paths, run, preview, status_callback=None, gpu_ids=(0,)):
    gpu_ids = tuple(gpu_ids)
    if len(gpu_ids) == 1:
        return InteractiveFitSession(paths, run, preview, status_callback)
    return DistributedInteractiveFitSession(
        paths, run, preview, gpu_ids, status_callback)
