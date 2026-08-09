"""Resident owner of one FitContext per rank and its optimizer loop.

The fitter thread constructs the context, drives its load/build/step phases,
and closes it; only that thread calls Torch/CUDA. Other threads communicate
through a condition variable and consume copied status snapshots.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
from typing import Any, ClassVar, Mapping
import uuid

from fit_session import (ScrollSpec, SessionState, SpiralInputPaths,
                         SpiralPreviewConfig, SpiralRunConfig,
                         run_mutable_config)
from config import Config, FitConfig, durable_config
from spiral_progress import NullProgressReporter, ProgressReporter


_NO_PROGRESS = NullProgressReporter()


# The transitions the resident session is allowed to make, keyed by the state
# it is leaving. Error and Closing are reachable from every state and are
# therefore not repeated here; Closing is terminal. Anything outside this
# table is a programming error in the runtime, not a runtime condition.
_LEGAL_SESSION_TRANSITIONS = {
    # A phase-only update repeats the current state, so every state may
    # transition to itself.
    SessionState.Loading: {SessionState.Loading, SessionState.Idle,
                           SessionState.ExportingPreview},
    SessionState.Idle: {SessionState.Idle, SessionState.Running,
                        SessionState.Saving},
    SessionState.Running: {SessionState.Running, SessionState.Idle,
                           SessionState.Saving},
    SessionState.Saving: {SessionState.Saving, SessionState.Idle,
                          SessionState.ExportingPreview},
    SessionState.ExportingPreview: {SessionState.ExportingPreview,
                                    SessionState.Idle},
    SessionState.Error: {SessionState.Error},
    SessionState.Closing: {SessionState.Closing},
}


def idle_phase(completed_iterations):
    """Human-facing phase for an idle session.

    Ready and Paused are the same lifecycle state; only the iteration count
    tells "has never produced work" from "stopped after N iterations". The
    label is presentation, so clients derive the same words from
    ``state == Idle`` plus ``current_iteration``.
    """
    return "Paused" if int(completed_iterations or 0) > 0 else "Ready"


# Provisional ordering, least-advanced first, for the distributed proxy's
# "minimum state across ranks" aggregation. This models a collective state as
# the least-advanced rank, which is only an approximation: it cannot tell "one
# rank is still loading" from "the ranks disagree about which command they are
# executing". PR 3 commit 2 replaces it with an explicit command epoch that
# every rank validates, so collective states become visible only when all
# ranks report the same epoch. Until then this keeps the existing behaviour,
# expressed over the enum instead of over ad-hoc string sets.
_PROVISIONAL_RANK_STATE_ORDER = (
    SessionState.Loading, SessionState.Saving, SessionState.ExportingPreview,
    SessionState.Running, SessionState.Idle,
)


def _provisional_aggregate_state(states):
    ordered = [state for state in states
               if state in _PROVISIONAL_RANK_STATE_ORDER]
    if not ordered:
        return SessionState.Loading
    return min(ordered, key=_PROVISIONAL_RANK_STATE_ORDER.index)


class _SessionShutdown(BaseException):
    pass


@dataclasses.dataclass
class SessionCommand:
    """One command queued for the fitter thread's pause boundary.

    Commands are created by coordinator threads and executed by the fitter
    thread while the session is idle between steps. Each one carries the
    facts needed to decide, at execution time, whether it is still the
    command the caller asked for: the session generation it was queued
    against and the iteration/configuration revision it was computed from.
    A mismatch cancels the command instead of applying stale work.
    """

    kind: ClassVar[str] = "command"

    command_id: str = dataclasses.field(
        default_factory=lambda: uuid.uuid4().hex)
    session_generation: int = 0
    expected_iteration: int | None = None
    expected_config_revision: int | None = None
    result: dict[str, Any] = dataclasses.field(default_factory=dict)
    error: str | None = None
    cancelled: bool = False
    done: threading.Event = dataclasses.field(
        default_factory=threading.Event, repr=False)

    def stale_reason(self, *, session_generation, iteration, config_revision):
        if self.session_generation != session_generation:
            return (f"{self.kind} command {self.command_id} was queued "
                    f"against session generation {self.session_generation}, "
                    f"which is no longer current ({session_generation})")
        if (self.expected_iteration is not None
                and self.expected_iteration != iteration):
            return (f"{self.kind} command {self.command_id} expected "
                    f"iteration {self.expected_iteration}, found {iteration}")
        if (self.expected_config_revision is not None
                and self.expected_config_revision != config_revision):
            return (f"{self.kind} command {self.command_id} expected "
                    f"configuration revision {self.expected_config_revision}, "
                    f"found {config_revision}")
        return None

    def complete(self, **result):
        self.result.update(result)
        self.done.set()

    def fail(self, error):
        self.error = str(error)
        self.done.set()

    def cancel(self, reason):
        self.cancelled = True
        self.error = str(reason)
        self.done.set()

    def wait(self, timeout):
        return self.done.wait(timeout)


@dataclasses.dataclass
class ConfigureCommand(SessionCommand):
    """Apply Run-scoped configuration and input path changes."""

    kind: ClassVar[str] = "configure"

    config: dict[str, Any] = dataclasses.field(default_factory=dict)
    path_changes: dict[str, Any] = dataclasses.field(default_factory=dict)
    previous_run_config: dict[str, Any] | None = None


@dataclasses.dataclass
class IncorporateCommand(SessionCommand):
    """Append newly uploaded ephemeral inputs to the resident fit."""

    kind: ClassVar[str] = "incorporate"

    records: list = dataclasses.field(default_factory=list)
    mark_incorporated: Any = None
    influence_config: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class SaveCheckpointCommand(SessionCommand):
    """Write a checkpoint from the fitter thread."""

    kind: ClassVar[str] = "save"

    path: str = ""


class InteractiveFitSession:
    def __init__(self, paths: SpiralInputPaths, run: SpiralRunConfig,
                 preview: SpiralPreviewConfig, scroll: ScrollSpec,
                 status_callback=None, publishes_outputs=True,
                 event_callback=None) -> None:
        self.paths = paths
        self.run_config = run
        self.scroll = scroll
        self.preview_config = preview
        self.input_manifest = paths.manifest()
        self.requested_config = dict(run.config)
        self._applied_config = None
        self._run_config = None
        self._run_config_limits = None
        self._default_advanced_config = None
        self._status_callback = status_callback
        # Receives (rank, status) pairs; the in-process session is rank 0.
        self._event_callback = event_callback
        self.publishes_outputs = publishes_outputs
        self._condition = threading.Condition()
        self._state = SessionState.Loading
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
        # The FitContext this session owns, set on the fitter thread once the
        # device state is built. It doubles as the capability flag for the
        # fitter operations (save/preview/incorporate/configure).
        self._context = None
        # Commands queued for the fitter thread's pause boundary, oldest
        # first. Both the generation and the configuration revision fence
        # commands that were computed against a session state the fitter no
        # longer has.
        self._commands = []
        self.session_generation = 0
        self._config_revision = 0
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
                "supports_input_incorporation": self._context is not None,
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
        event_callback = getattr(self, "_event_callback", None)
        if self._status_callback is None and event_callback is None:
            return
        status = self.status()
        if event_callback is not None:
            try:
                event_callback(0, status)
            except Exception:
                traceback.print_exc(limit=4)
        if self._status_callback is not None:
            self._status_callback(status)

    def _progress_reporter(self):
        return getattr(self, "progress", _NO_PROGRESS)

    def _progress_changed(self, snapshot):
        if snapshot is not None:
            with self._condition:
                self._phase = str(snapshot.get("stage_name") or self._phase)
        self._publish_status()

    def _transition_locked(self, new_state, phase="", *, reason=None):
        """Authoritative lifecycle transition; call with the lock held.

        Every state change in the session goes through here. An illegal
        transition is a programming error in the runtime — the fitter thread
        is the only writer and the legal set describes the whole loop — so it
        raises instead of silently correcting itself.
        """
        new_state = SessionState(new_state)
        legal = _LEGAL_SESSION_TRANSITIONS[self._state]
        if self._state is not SessionState.Closing:
            legal = legal | {SessionState.Error, SessionState.Closing}
        if new_state not in legal:
            raise RuntimeError(
                f"Illegal session transition {self._state.name} -> "
                f"{new_state.name}"
                + (f" ({reason})" if reason else ""))
        self._state = new_state
        self._phase = phase or new_state.value
        self._condition.notify_all()

    def _transition(self, new_state, phase="", *, reason=None):
        """Transition and publish the resulting status snapshot."""
        with self._condition:
            self._transition_locked(new_state, phase, reason=reason)
        self._publish_status()

    def _fit_main(self):
        context = None
        distributed_initialized = False
        try:
            self._progress_reporter().begin("loading", "Importing Torch and fitter")
            self._transition(SessionState.Loading, "Importing Torch and fitter")
            import fit_spiral as fitter
            from ddp_helpers import (maybe_destroy_distributed,
                                     maybe_init_distributed)
            from spiral_helpers import scale_and_split_counts

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
                    # Checkpoints store the durable subset of the schema
                    # (see config.durable_config), so key sets compare
                    # against that subset. z_begin/z_end joined the schema
                    # after many checkpoints were written: a stored cfg
                    # lacking exactly those keys is accepted with defaults
                    # from the session request; every other key-set
                    # mismatch stays a strict error.
                    durable_schema = set(durable_config(config))
                    missing = durable_schema - set(durable)
                    if set(durable) - durable_schema or missing - {"z_begin", "z_end"}:
                        raise ValueError(
                            "Checkpoint configuration does not match the current schema")
                    if missing:
                        assumed = {
                            "z_begin": int(self.run_config.z_begin),
                            "z_end": int(self.run_config.z_end),
                        }
                        durable.update(
                            {key: assumed[key] for key in missing})
                        warning = (
                            f"Checkpoint {self.paths.checkpoint} predates "
                            "z_begin/z_end in the stored configuration; "
                            "assuming "
                            + ", ".join(f"{key}={assumed[key]}"
                                        for key in sorted(missing))
                            + " from the session request")
                        print(warning)
                        with self._condition:
                            self._warnings.append(warning)
                    durable = Config(durable).as_dict()
                    # The optimisation z window is owned by the session
                    # request; the checkpoint's stored range only documents
                    # what it trained with.
                    durable["z_begin"] = int(self.run_config.z_begin)
                    durable["z_end"] = int(self.run_config.z_end)
                    config.update(durable)
                    # The session-scoped profile initially reproduces the
                    # checkpoint without applying scaling twice.
                    checkpoint_profile_config = copy.deepcopy(durable)
                finally:
                    # This first load exists only to resolve configuration.  Do
                    # not retain a complete model + optimiser checkpoint for the
                    # lifetime of the resident fitter thread.
                    del checkpoint_config
            # The session request's z window is authoritative for this fit,
            # both for the applied configuration and the Default profile.
            config["z_begin"] = int(self.run_config.z_begin)
            config["z_end"] = int(self.run_config.z_end)
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
            config["z_begin"] = int(self.run_config.z_begin)
            config["z_end"] = int(self.run_config.z_end)
            fields = Config.catalog()["schema"]["fields"]
            count_keys = tuple(
                key for key, spec in fields.items() if spec.get("scale_with_z"))
            scale_and_split_counts(
                config, self.run_config.z_begin, self.run_config.z_end,
                count_keys)
            if checkpoint_profile_config is None:
                scale_and_split_counts(
                    default_advanced_config, self.run_config.z_begin,
                    self.run_config.z_end, count_keys)
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

            self._progress_reporter().begin("loading", "Loading fit inputs and model")
            self._transition(SessionState.Loading, "Loading fit inputs and model")
            # The scroll specification supplies the physical facts (including
            # the outward sense, which is not part of the load request). The
            # session request may still override the request-carried scroll
            # values it historically owned.
            scroll = dataclasses.replace(
                self.scroll,
                name=self.run_config.scroll_name,
                voxel_size_um=self.run_config.voxel_size_um,
                normal_zarr_group=self.run_config.lasagna_group,
                lasagna_scale=self.run_config.lasagna_scale,
            )
            # The runtime is the execution owner of the context: it constructs
            # it, drives every phase on this fitter thread, and closes it.
            # Configuration, the scroll facts, the resolved input paths, and
            # the fit controls (resume, output directory, run tag) are passed
            # explicitly; fit_spiral holds no module-global dataset state.
            context = fitter.FitContext(
                FitConfig(config),
                scroll=scroll,
                paths=self.paths,
                interactive_driver=self,
                progress=self.progress,
                resume_path=self.paths.checkpoint or None,
                resume_step=(self.run_config.legacy_checkpoint_step
                             if self.paths.checkpoint else 0),
                out_base_dir=self.paths.output_directory,
                run_tag=self.run_config.run_tag or None,
                cache_dir=self.paths.cache_directory,
                storage_backend=self.run_config.storage_backend,
                render_volume_scale=self.run_config.render_volume_scale)
            context.load_host_inputs()
            context.resolve_output_path()
            context.build_device_state()
            context.release_setup_only_tracks()
            self._session_ready(context)
            self._optimize(context)
        except BaseException as exc:
            with self._condition:
                if self._shutdown and isinstance(exc, _SessionShutdown):
                    self._transition_locked(SessionState.Closing, "Stopped")
                else:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._warnings.append(traceback.format_exc(limit=12))
                    self._transition_locked(SessionState.Error, "Error")
            self._publish_status()
        finally:
            if context is not None:
                # Resource release runs here, on the owning fitter thread.
                context.close()
            if distributed_initialized:
                maybe_destroy_distributed()
            self._progress_reporter().close()

    # Fitter-thread session driver.
    def _session_ready(self, context):
        """Adopt the built context and publish Idle.

        Runs on the fitter thread once build_device_state() has returned.
        When the session was constructed from a checkpoint, the restored
        model's preview is exported and published before the session goes
        idle, exactly as the legacy on_ready handoff did.
        """
        with self._condition:
            self._context = context
            self._completed = self._target = context.start_iteration
            self._output_path = context.out_path
        if self.paths.checkpoint and getattr(self, "publishes_outputs", True):
            self._transition(SessionState.ExportingPreview,
                             "Exporting restored checkpoint preview")
            self._progress_reporter().begin(
                "exporting_preview", "Exporting restored checkpoint preview")
            self._publish_preview()
        self._progress_reporter().clear()
        self._transition(SessionState.Idle, idle_phase(self._completed))

    def _optimize(self, context):
        """Drive the resident optimizer loop on the fitter thread.

        A resident session has no natural end: the configured horizon defines
        the learning-rate schedule but never caps how long the user may keep
        optimizing. The loop exits only through the shutdown exception raised
        at the wait_for_iteration pause boundary.
        """
        for iteration in itertools.count(context.start_iteration):
            self.wait_for_iteration(iteration)
            loss, losses, log_metrics, shell_metrics = context.step(iteration)
            self.iteration_completed(
                completed_iterations=iteration + 1,
                total_loss=float(loss.detach().item()),
                losses={name: float(value.detach().item())
                        for name, value in losses.items()},
                learning_rate=float(context.optimiser.param_groups[0]['lr']),
                metrics={name: float(value)
                         for name, value in log_metrics.items()},
            )
            context.log_step_metrics(
                iteration, loss, losses, log_metrics, shell_metrics)

    def wait_for_iteration(self, iteration):
        while True:
            with self._condition:
                if self._shutdown:
                    raise _SessionShutdown()
                # Commands are drained before the pending check so inputs
                # queued by run() are incorporated before the next step begins.
                command = self._commands.pop(0) if self._commands else None
                if command is None:
                    if self._pending > 0:
                        return
                    self._condition.wait()
                    continue
                stale = command.stale_reason(
                    session_generation=self.session_generation,
                    iteration=self._completed,
                    config_revision=self._config_revision)
            if stale is not None:
                command.cancel(stale)
                continue
            if isinstance(command, IncorporateCommand):
                self._run_incorporation(command)
                continue
            if isinstance(command, ConfigureCommand):
                self._run_configuration(command)
                continue
            if not isinstance(command, SaveCheckpointCommand):
                command.fail(f"Unknown session command {command.kind}")
                continue
            self._run_checkpoint_save(command)

    def _run_checkpoint_save(self, command):
        """Write one requested checkpoint at the pause boundary."""
        with self._condition:
            previous_state = self._state
            previous_phase = self._phase
            self._transition_locked(
                SessionState.Saving, "Saving checkpoint",
                reason=f"save command {command.command_id}")
        path = None
        error = None
        try:
            self._progress_reporter().begin(
                "saving_checkpoint", "Saving checkpoint",
                detail=Path(command.path).name)
            path = self._context.save_checkpoint(
                command.path, self._completed)
            self._progress_reporter().finish()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            self._progress_reporter().clear()
            with self._condition:
                self._transition_locked(previous_state, previous_phase)
            self._publish_status()
            # The waiter is released only once the session is back in the
            # state it will be observed in.
            if error is None:
                command.complete(path=path)
            else:
                command.fail(error)

    def _run_incorporation(self, command):
        """Append newly uploaded ephemeral inputs to the resident fit.

        Runs on the fitter thread at the pause boundary. A failure cancels the
        queued run and surfaces a warning instead of tearing down the session.
        """
        records = command.records
        mark_incorporated = command.mark_incorporated
        influence_config = command.influence_config
        try:
            if self._context is None:
                raise RuntimeError(
                    "The resident fitter does not support adding inputs to a running session")
            with self._condition:
                self._phase = "Incorporating new session inputs"
                # run() set the pause-boundary target alongside the queued
                # inputs; the context sizes its DT-free window from it.
                current_iteration = self._completed
                target_iteration = self._target
            self._progress_reporter().begin(
                "incorporating_inputs", "Incorporating new session inputs",
                step=0, total_steps=len(records), unit="inputs")
            self._publish_status()
            self._context.incorporate_interactive_inputs(
                records, influence_config,
                current_iteration=current_iteration,
                target_iteration=target_iteration)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._progress_reporter().clear()
            with self._condition:
                self._pending = 0
                self._target = self._completed
                self._warnings.append(f"Input incorporation failed: {error}")
                self._transition_locked(
                    SessionState.Idle, idle_phase(self._completed),
                    reason="input incorporation failed")
            if mark_incorporated is not None:
                mark_incorporated(records, error=error)
            command.fail(error)
            self._publish_status()
        else:
            if mark_incorporated is not None:
                mark_incorporated(records)
            with self._condition:
                if self._state is SessionState.Running:
                    self._phase = "Optimizing"
            command.complete(incorporated=len(records))
            if getattr(self, "_state", None) is SessionState.Running:
                self._begin_optimization_progress()
            else:
                self._progress_reporter().clear()
            self._publish_status()

    def _run_configuration(self, command):
        """Apply validated Run-scoped settings on the fitter thread."""
        config = command.config
        path_changes = command.path_changes
        previous_run_config = command.previous_run_config
        try:
            if self._context is None:
                raise RuntimeError(
                    "The resident fitter does not support Run configuration changes")
            path_changes = dict(path_changes or {})
            self._progress_reporter().begin(
                "configuring", "Applying run configuration",
                detail=(
                    f"{len(config)} settings, {len(path_changes)} path changes"
                ))
            # An LR-schedule change realigns at the durable completed step.
            self._context.apply_config(
                dict(config), path_changes,
                current_iteration=self.completed_iterations)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._progress_reporter().clear()
            with self._condition:
                self._pending = 0
                self._target = self._completed
                # Leave newly uploaded inputs pending for a later valid Run.
                abandoned = [queued for queued in self._commands
                             if isinstance(queued, IncorporateCommand)]
                self._commands = [queued for queued in self._commands
                                  if not isinstance(queued, IncorporateCommand)]
                self._warnings.append(f"Run configuration failed: {error}")
                if previous_run_config is not None:
                    self._run_config.update(previous_run_config)
                self._transition_locked(
                    SessionState.Idle, idle_phase(self._completed),
                    reason="run configuration failed")
            for queued in abandoned:
                queued.cancel(
                    f"cancelled by failed configure command "
                    f"{command.command_id}")
            command.fail(error)
            self._publish_status()
        else:
            if getattr(self, "_applied_config", None) is not None:
                with self._condition:
                    self._applied_config.update(config)
                    self._run_config.update(config)
                    self.requested_config.update(config)
                    self.input_manifest.update(path_changes)
            with self._condition:
                self._config_revision += 1
            command.complete(config_revision=self._config_revision)
            if getattr(self, "_state", None) is SessionState.Running:
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
            if self._context is not None:
                self._context.clear_interactive_influence()
            if not getattr(self, "publishes_outputs", True):
                self._progress_reporter().clear()
                self._transition(SessionState.Idle, idle_phase(self._completed))
                return
            self._transition(SessionState.Saving, "Autosaving checkpoint")
            self._progress_reporter().begin(
                "saving_checkpoint", "Autosaving checkpoint",
                detail="checkpoint_autosave.ckpt")
            autosave = str(Path(self._output_path) / "checkpoint_autosave.ckpt")
            self._context.save_checkpoint(autosave, self._completed)
            self._transition(SessionState.ExportingPreview, "Exporting preview")
            self._progress_reporter().begin(
                "exporting_preview", "Exporting preview")
            self._publish_preview()
            self._progress_reporter().clear()
            self._transition(SessionState.Idle, idle_phase(self._completed))

    def _publish_preview(self):
        with self._condition:
            generation = self._preview_generation + 1
        generation_path = (Path(self.paths.output_directory) / ".spiral-preview" /
                           self._preview_session_id / f"generation-{generation}")
        surface_id = f"spiral-output-generation-{generation}"
        manifest = self._context.export_preview(str(generation_path), surface_id)
        with self._condition:
            self._preview_generation = generation
            self._preview_manifest = str(manifest["manifest_path"])
        # Publish while the session is still in ExportingPreview.  The host
        # service synchronously Lasagna-flattens and packages this generation
        # from the status callback, so clients cannot start another Run while
        # the downloadable preview is still being prepared.
        self._publish_status()

    # Coordinator-thread commands.
    def run(self, count, pending_inputs=None, mark_incorporated=None,
            influence_config=None, run_config=None, path_changes=None):
        if count < 1:
            raise ValueError("iterations must be at least 1")
        with self._condition:
            if self._state is not SessionState.Idle:
                raise RuntimeError(
                    f"Run is not allowed while session state is "
                    f"{self._state.name}")
            run_config = dict(run_config or {})
            path_changes = dict(path_changes or {})
            target = self._completed + count
            requested_config = dict(
                getattr(self, "requested_config", {}) or {})
            requested_config.update(run_config)
            configured_horizon = int(
                requested_config.get("optimizer_num_training_steps", 0) or 0)
            # Interactive sessions are allowed to continue beyond the original
            # headless horizon. Preserve the current LR curve while the whole
            # requested run fits within it. When a run would cross the horizon,
            # extend the horizon by the requested count and realign the
            # exponential curve at the durable completed step.
            if (getattr(self, "_context", None) is not None
                    and target > configured_horizon):
                run_config["optimizer_num_training_steps"] = (
                    max(configured_horizon, self._completed) + count)
            if run_config or path_changes:
                if self._context is None:
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
                # Configuration is queued ahead of incorporation: new inputs
                # must be incorporated under the settings this run asked for.
                self._commands.append(ConfigureCommand(
                    session_generation=self.session_generation,
                    expected_iteration=self._completed,
                    expected_config_revision=self._config_revision,
                    config=run_config, path_changes=path_changes,
                    previous_run_config=previous_run_config))
                self._run_config.update(run_config)
            if pending_inputs:
                if self._context is None:
                    raise RuntimeError(
                        "The resident fitter does not support adding inputs to a running session")
                self._commands.append(IncorporateCommand(
                    session_generation=self.session_generation,
                    expected_iteration=self._completed,
                    records=list(pending_inputs),
                    mark_incorporated=mark_incorporated,
                    influence_config=dict(influence_config or {})))
            self._pending = count
            self._run_start_completed = self._completed
            self._target = target
            self._transition_locked(SessionState.Running, "Optimizing")
            self._begin_optimization_progress()
            self._condition.notify_all()
            return self._target

    def stop(self):
        with self._condition:
            if self._state is not SessionState.Running:
                raise RuntimeError("Session is not running")
            self._stop_requested = True

    def save_checkpoint(self, path, timeout=120.0):
        with self._condition:
            if self._state is not SessionState.Idle:
                raise RuntimeError(
                    f"Checkpoint save is not allowed in {self._state.name}")
            command = SaveCheckpointCommand(
                session_generation=self.session_generation,
                expected_iteration=self._completed,
                path=path)
            self._commands.append(command)
            self._condition.notify_all()
        if not command.wait(timeout):
            raise TimeoutError("Checkpoint save timed out")
        if command.error is not None:
            raise RuntimeError(command.error)
        return command.result["path"]

    def close(self, timeout=15.0):
        with self._condition:
            self._shutdown = True
            # Commands queued against the session being torn down belong to a
            # generation that no longer exists; release their waiters instead
            # of letting them time out.
            self.session_generation += 1
            abandoned, self._commands = self._commands, []
            self._condition.notify_all()
        for command in abandoned:
            command.cancel("The fit session was closed")
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("Spiral fitter did not stop at a safe boundary")


def _free_loopback_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _distributed_session_worker(rank, world_size, gpu_id, master_port,
                                paths, run, preview, scroll, commands, events):
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
            paths, run, preview, scroll, publish_status,
            publishes_outputs=(rank == 0))
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

    def __init__(self, paths, run, preview, scroll, gpu_ids,
                 status_callback=None, event_callback=None):
        self._gpu_ids = tuple(gpu_ids)
        self._status_callback = status_callback
        # Receives (rank, status) pairs. Child ranks publish their status
        # snapshots through the parent event queue; this routes them onward
        # tagged with the originating rank.
        self._event_callback = event_callback
        self._condition = threading.Condition()
        self._status = {
            "state": SessionState.Loading, "phase": "Starting GPU workers",
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
                      paths, run, preview, scroll,
                      self._commands[rank], self._events),
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

    def _publish_rank_event(self, rank, status):
        callback = getattr(self, "_event_callback", None)
        if callback is None:
            return
        try:
            callback(rank, status)
        except Exception:
            traceback.print_exc(limit=4)

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
                self._publish_rank_event(rank, status)
                with self._condition:
                    self._rank_statuses[rank] = status
                    if (self._failed_error is not None
                            and status.get("state") != SessionState.Error):
                        continue
                    if status.get("state") == SessionState.Error:
                        if rank == 0:
                            self._status = status
                        else:
                            warnings = list(self._status.get("warnings", []))
                            warnings.extend(status.get("warnings", []))
                            self._status.update({
                                "state": SessionState.Error, "phase": "Error",
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
                        all_ranks_ready = (
                            len(self._rank_statuses) == len(self._gpu_ids)
                            and all(item.get("state") == SessionState.Idle
                                    for item in self._rank_statuses.values())
                        )
                        if (rank_zero.get("state") == SessionState.Idle
                                and not all_ranks_ready):
                            unfinished = next(
                                ((worker_rank, item)
                                 for worker_rank, item
                                 in sorted(self._rank_statuses.items())
                                 if item.get("state") != SessionState.Idle),
                                None)
                            self._status = copy.deepcopy(rank_zero)
                            self._status.update({
                                "state": _provisional_aggregate_state(
                                    item.get("state")
                                    for item in self._rank_statuses.values()),
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
                self._publish_rank_event(rank, {
                    "state": SessionState.Error, "error": error,
                    "warnings": [trace]})
                with self._condition:
                    warnings = list(self._status.get("warnings", []))
                    warnings.append(f"GPU worker rank {rank} failed:\n{trace}")
                    self._status.update({
                        "state": SessionState.Error, "phase": "Error",
                        "error": error,
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
        if state != SessionState.Idle:
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
        if state != SessionState.Running:
            raise RuntimeError(f"Session is not running (state is {state})")
        return self._call("stop")

    def save_checkpoint(self, path, timeout=120.0):
        state = self.status()["state"]
        if state != SessionState.Idle:
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


def create_session(paths, run, preview, scroll, status_callback=None,
                   gpu_ids=(0,), event_callback=None):
    gpu_ids = tuple(gpu_ids)
    if len(gpu_ids) == 1:
        return InteractiveFitSession(paths, run, preview, scroll,
                                     status_callback,
                                     event_callback=event_callback)
    return DistributedInteractiveFitSession(
        paths, run, preview, scroll, gpu_ids, status_callback,
        event_callback=event_callback)
