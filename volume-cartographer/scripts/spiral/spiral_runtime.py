"""Resident adapter around the legacy fitter's optimizer loop.

Only the fitter thread calls Torch/CUDA. Other threads communicate through a
condition variable and consume copied status snapshots.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import threading
import traceback
from typing import Any, Callable, Mapping
import uuid

from fit_session import SpiralInputPaths, SpiralPreviewConfig, SpiralRunConfig


class _SessionShutdown(BaseException):
    pass


class InteractiveFitSession:
    def __init__(self, paths: SpiralInputPaths, run: SpiralRunConfig,
                 preview: SpiralPreviewConfig, status_callback=None) -> None:
        self.paths = paths
        self.run_config = run
        self.preview_config = preview
        self.input_manifest = paths.manifest()
        self.requested_config = dict(run.config)
        self._status_callback = status_callback
        self._condition = threading.Condition()
        self._state = "Loading"
        self._phase = "Importing fitter"
        self._error = None
        self._warnings = []
        self._completed = 0
        self._target = 0
        self._pending = 0
        self._horizon = 0
        self._stop_requested = False
        self._shutdown = False
        self._latest_metrics = {}
        self._output_path = paths.output_directory
        self._preview_manifest = None
        self._preview_generation = 0
        self._preview_session_id = uuid.uuid4().hex
        self._geometry_snapshot_manifest = None
        self._save_checkpoint = None
        self._export_preview = None
        self._idle_actions = []
        self._thread = threading.Thread(target=self._fit_main, name="spiral-fit-worker", daemon=True)
        self._thread.start()

    @property
    def completed_iterations(self):
        with self._condition:
            return self._completed

    def status(self):
        with self._condition:
            return {
                "state": self._state, "phase": self._phase,
                "current_iteration": self._completed,
                "target_iteration": self._target,
                "session_horizon": self._horizon,
                "latest_metrics": copy.deepcopy(self._latest_metrics),
                "warnings": list(self._warnings), "error": self._error,
                "preview_manifest_path": self._preview_manifest,
                "preview_generation": self._preview_generation,
                "geometry_snapshot_manifest_path": self._geometry_snapshot_manifest,
            }

    def _publish_status(self):
        if self._status_callback is not None:
            self._status_callback(self.status())

    def _set_state(self, state, phase=""):
        with self._condition:
            self._state = state
            self._phase = phase or state
            self._condition.notify_all()
        self._publish_status()

    def _fit_main(self):
        fitter = None
        wandb = None
        try:
            self._set_state("Loading", "Importing Torch and fitter")
            import wandb
            import fit_spiral as fitter
            from ddp_helpers import split_counts_across_ranks
            from spiral_helpers import scale_counts_for_z_range

            config = dict(fitter.default_config)
            if self.paths.checkpoint:
                import torch
                checkpoint_config = torch.load(self.paths.checkpoint, map_location='cpu', weights_only=False)
                if isinstance(checkpoint_config, dict) and isinstance(checkpoint_config.get('cfg'), Mapping):
                    config.update(dict(checkpoint_config['cfg']))
            unknown = sorted(set(self.run_config.config) - set(config))
            if unknown:
                raise ValueError(f"Unknown advanced config keys: {unknown}")
            config.update(self.run_config.config)
            self.requested_config = dict(config)
            count_keys = (
                'num_patches_per_step', 'num_patches_per_step_for_dt',
                'unverified_num_patches_per_step', 'unverified_num_patches_per_step_for_dt',
                'rel_winding_num_pcls', 'abs_winding_num_pcls',
                'unattached_pcl_num_per_step', 'track_num_per_step',
                'dense_normals_num_points', 'regularisation_num_points', 'shell_num_samples',
            )
            scale_counts_for_z_range(config, self.run_config.z_begin,
                                     self.run_config.z_end, 9500, count_keys)
            split_counts_across_ranks(config, count_keys)

            fitter.dataset_path = self.paths.dataset_root
            fitter.normal_nx_zarr_path = self.paths.normal_x or None
            fitter.normal_ny_zarr_path = self.paths.normal_y or None
            fitter.grad_mag_zarr_path = self.paths.gradient_magnitude or None
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
            self._set_state("Loading", "Loading fit inputs and model")
            fitter.main(interactive_driver=self)
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

    # Fitter-thread callbacks.
    def on_ready(self, *, completed_iterations, session_horizon, output_path,
                 save_checkpoint, export_preview, geometry_snapshot_manifest=None):
        with self._condition:
            self._completed = self._target = completed_iterations
            self._horizon = session_horizon
            self._output_path = output_path
            self._save_checkpoint = save_checkpoint
            self._export_preview = export_preview
            self._geometry_snapshot_manifest = geometry_snapshot_manifest
        if self.paths.checkpoint:
            self._set_state("ExportingPreview", "Exporting restored checkpoint preview")
            self._publish_preview()
        self._set_state("Ready", "Ready")

    def wait_for_iteration(self, iteration):
        while True:
            with self._condition:
                if self._shutdown:
                    raise _SessionShutdown()
                if self._pending > 0:
                    return True
                action = self._idle_actions.pop(0) if self._idle_actions else None
                if action is None:
                    self._condition.wait()
                    continue
            kind, path, done, result = action
            try:
                if kind == "save":
                    result["path"] = self._save_checkpoint(path, self._completed)
                else:
                    result["error"] = f"Unknown idle action {kind}"
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                done.set()

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
        self._publish_status()
        if pause:
            self._set_state("Saving", "Autosaving checkpoint")
            autosave = str(Path(self._output_path) / "checkpoint_autosave.ckpt")
            self._save_checkpoint(autosave, self._completed)
            self._set_state("ExportingPreview", "Exporting preview")
            self._publish_preview()
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

    def session_finished(self):
        self._set_state("Paused", "Session horizon reached")
        while True:
            self.wait_for_iteration(self._completed)
            raise RuntimeError("Cannot run beyond the configured session horizon")

    # Coordinator-thread commands.
    def run(self, count):
        if count < 1:
            raise ValueError("iterations must be at least 1")
        with self._condition:
            if self._state not in {"Ready", "Paused"}:
                raise RuntimeError(f"Run is not allowed while session state is {self._state}")
            if self._completed + count > self._horizon:
                raise ValueError(f"Run exceeds fixed session horizon {self._horizon}")
            self._pending = count
            self._target = self._completed + count
            self._state, self._phase = "Running", "Optimizing"
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


def create_session(paths, run, preview, status_callback=None):
    return InteractiveFitSession(paths, run, preview, status_callback)
