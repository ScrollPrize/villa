"""Resident owner of one FitContext per rank and its optimizer loop.

The fitter thread constructs the context, drives its load/build/step phases,
and closes it; only that thread calls Torch/CUDA. Other threads communicate
through a condition variable and consume copied status snapshots.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, ClassVar, Mapping
import uuid

from fit_session import (AUTOSAVE_CHECKPOINT_NAME, AUTOSAVE_INTERVAL_ITERATIONS,
                         ScrollSpec, SessionState,
                         SpiralInputPaths, SpiralPreviewConfig,
                         SpiralRunConfig, run_mutable_config,
                         write_autosave_metadata)
from config import (BACKFILLABLE_CONFIG_DEFAULTS, RETIRED_CONFIG_KEYS,
                    Config, FitConfig,
                    durable_config, filter_known_config_keys)
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
    # Idle -> Loading is the in-session checkpoint load: inspecting and
    # applying a checkpoint takes the session out of Idle for the whole
    # two-phase operation, so nothing can be admitted against the model while
    # it is being replaced.
    SessionState.Idle: {SessionState.Idle, SessionState.Running,
                        SessionState.Saving, SessionState.Loading,
                        SessionState.ExportingPreview},
    SessionState.Running: {SessionState.Running, SessionState.Idle,
                           SessionState.Saving},
    SessionState.Saving: {SessionState.Saving, SessionState.Idle,
                          SessionState.ExportingPreview},
    SessionState.ExportingPreview: {SessionState.ExportingPreview,
                                    SessionState.Idle},
    SessionState.Error: {SessionState.Error},
    SessionState.Closing: {SessionState.Closing},
}


# An idle session reports one phase. "Ready" and "Paused" used to be
# separate lifecycle states, and briefly survived as labels derived from the
# completed iteration count, but the distinction is not one a user acts on:
# both mean the session is loaded, doing nothing, and ready to run. Clients
# that want to say how much work has happened read current_iteration.
IDLE_PHASE = "Idle"


# Bounded fail-stop budget for the distributed proxy.
#
# The rule these constants serve: every wait a client can be parked behind is
# bounded by a value chosen here, never by the process group's collective
# timeout. A rank that dies inside a collective leaves its siblings blocked in
# NCCL for FIT_SPIRAL_DDP_TIMEOUT_MIN (60 minutes by default), which is not an
# answer any interactive client can use, so the parent detects the death and
# takes the job down itself.
#
# Worst-case time from "a worker died" to "the client has a diagnosable
# error": WATCHDOG_POLL_INTERVAL_S + WORKER_TERMINATE_GRACE_S +
# WORKER_KILL_GRACE_S, i.e. about 11 seconds.

# How often the parent samples worker liveness. Small enough that a crash is
# reported in about the time a human notices a stall, large enough that
# polling a handful of processes costs nothing.
WATCHDOG_POLL_INTERVAL_S = 0.5

# Grace for a worker to exit after SIGTERM. Covers the process's own atexit /
# CUDA teardown; after it the worker is SIGKILLed.
WORKER_TERMINATE_GRACE_S = 5.0

# Grace for a SIGKILLed worker to be reaped. A worker stuck in an
# uninterruptible driver call can outlive this; the parent gives up waiting
# rather than blocking the client on it.
WORKER_KILL_GRACE_S = 5.0

# How long an all-rank command may take to be acknowledged by every rank.
# Admission commands (run/stop) only move a rank's own queue, so this is
# generous by two orders of magnitude; exceeding it means a rank is wedged.
COMMAND_ACK_TIMEOUT_S = 30.0

# Extra slack on top of a caller-supplied operation timeout (checkpoint save,
# close) before the parent calls the worker wedged.
COMMAND_ACK_GRACE_S = 5.0

# Input application rebuilds resident catalogs, indices, fiber views, and GPU
# influence masks.  Its acknowledgement is the result of that whole operation,
# not the quick queue-admission acknowledgement covered above.
INPUT_BATCH_TIMEOUT_S = 1200.0

# How long one requested preview export may take. The window covers the whole
# operation, not just the render: the host service Lasagna-flattens and hashes
# the generation synchronously from the status callback, inside the session's
# ExportingPreview window. Exceeding it reports a failure for an export that
# usually goes on to publish anyway, so it is sized for the largest fits.
PREVIEW_EXPORT_TIMEOUT_S = 1200.0


# All-rank commands that only a rank with nothing outstanding may enter.
_QUIESCENT_COMMANDS = frozenset({
    "run", "preflight_checkpoint", "apply_checkpoint", "rebuild_model",
})


class CommandBarrierViolation(RuntimeError):
    """A rank refused a command because the barrier did not match its state.

    Fail-stop, never a silent skip: a rank whose view of the epoch, the
    command kind, the configuration revision, or the pending count disagrees
    with the coordinator cannot enter another training step without diverging
    from its siblings.
    """


@dataclasses.dataclass(frozen=True)
class CommandBarrier:
    """The coordinator's description of one all-rank command.

    The same barrier is delivered to every participating rank, and every rank
    validates it against its own state before acting on the command. The
    epoch is monotonically increasing and owned by the coordinator, so a
    repeated, skipped, or reordered barrier is detectable locally by each
    rank without a collective.

    ``config_revision`` is the revision the coordinator believes every rank
    holds; it is None when the coordinator has not yet observed a unanimous
    revision (during startup), in which case the field is not checked.
    ``pending`` is the number of iterations that must still be outstanding at
    the boundary: 0 for commands that require a quiescent rank (a run may
    only be admitted by an idle rank), None for commands that are by nature
    asynchronous with respect to the step loop (stop, close).
    """

    epoch: int
    kind: str
    config_revision: int | None = None
    pending: int | None = None

    def violation(self, *, rank, expected_epoch, kind, config_revision,
                  pending):
        """Describe why this rank must refuse the barrier, or None."""
        if self.epoch != expected_epoch:
            return (f"rank {rank} expected command epoch {expected_epoch}, "
                    f"received epoch {self.epoch} for a {self.kind} command")
        if self.kind != kind:
            return (f"rank {rank} is executing a {kind} command under "
                    f"epoch {self.epoch}, which the coordinator issued as "
                    f"{self.kind}")
        if (self.config_revision is not None
                and self.config_revision != config_revision):
            return (f"rank {rank} holds configuration revision "
                    f"{config_revision}, but epoch {self.epoch} "
                    f"({self.kind}) was issued against revision "
                    f"{self.config_revision}")
        if self.pending is not None and self.pending != pending:
            return (f"rank {rank} has {pending} iterations pending, but "
                    f"epoch {self.epoch} ({self.kind}) requires "
                    f"{self.pending}")
        return None


# States rank 0 may occupy alone, as a coordinator sub-operation, while the
# other ranks are idle: only rank 0 publishes outputs, so only it saves the
# autosave/requested checkpoint and exports previews. These are named as
# coordinator work in the published phase instead of being dressed up as a
# collective state the other ranks are not in.
_COORDINATOR_OPERATIONS = {
    SessionState.Saving: "saving",
    SessionState.ExportingPreview: "exporting preview",
}


@dataclasses.dataclass(frozen=True)
class CollectiveView:
    """What the parent is entitled to publish from the per-rank snapshots.

    ``visible`` is True only when every participating rank reports the same
    command epoch and the same state: that is the one situation in which a
    state is a fact about the whole job. Otherwise the last agreed collective
    state stands, and the phase says what is actually happening — either a
    named coordinator sub-operation, or that the job is waiting for a
    specific rank.
    """

    state: SessionState
    visible: bool
    coordinator_operation: str | None = None
    laggard: int | None = None


def _laggard_rank(states):
    """The rank whose report is holding the collective state back.

    The coordinator's own state is the reference: the first other rank that
    disagrees with it is what the job is waiting for. With no coordinator
    report yet, the lowest reporting rank stands in.
    """
    if not states:
        return None
    if 0 not in states:
        return min(states)
    reference = states[0]
    return next((rank for rank in sorted(states)
                 if rank != 0 and states[rank] != reference), None)


def _rank_state(status):
    try:
        return SessionState(status.get("state"))
    except ValueError:
        return None


def collective_view(rank_statuses, world_size, last_state):
    """Decide the collective state from per-rank status snapshots."""
    last_state = SessionState(last_state)
    states = {rank: _rank_state(status)
              for rank, status in rank_statuses.items()}
    laggard = _laggard_rank(states)
    if len(rank_statuses) < world_size or None in states.values():
        return CollectiveView(last_state, False, laggard=laggard)
    epochs = {int(status.get("command_epoch", 0) or 0)
              for status in rank_statuses.values()}
    if len(epochs) != 1:
        # The ranks are executing different commands. Nothing about their
        # states is a collective fact, whatever they happen to say.
        epoch_laggard = _laggard_rank(
            {rank: int(status.get("command_epoch", 0) or 0)
             for rank, status in rank_statuses.items()})
        return CollectiveView(last_state, False,
                              laggard=(laggard if laggard is not None
                                       else epoch_laggard))
    distinct = set(states.values())
    if len(distinct) == 1:
        return CollectiveView(distinct.pop(), True)
    coordinator = states.get(0)
    others = {state for rank, state in states.items() if rank != 0}
    if (coordinator in _COORDINATOR_OPERATIONS
            and others == {SessionState.Idle}):
        return CollectiveView(
            coordinator, False,
            coordinator_operation=_COORDINATOR_OPERATIONS[coordinator])
    return CollectiveView(last_state, False, laggard=laggard)


class FileStoreRendezvous:
    """A rendezvous whose endpoint the service owns for the session lifetime.

    The previous scheme had rank 0 bind port 0, read the assigned port, close
    the socket, and hand the number to the workers: between the close and the
    workers' bind, any process on the host may take that port, and the
    failure surfaces much later as a rendezvous that never completes. A c10d
    file store has no such window. The endpoint is a path inside a directory
    this object creates under the service's output root and holds until
    :meth:`close`, so nothing else can claim it and nothing is left behind.

    Only the (picklable) :class:`RendezvousEndpoint` crosses into the worker
    processes; the directory stays owned by the parent.
    """

    def __init__(self, root=None):
        if root is not None:
            Path(root).mkdir(parents=True, exist_ok=True)
        self._directory = tempfile.mkdtemp(
            prefix="spiral-rendezvous-", dir=(str(root) if root else None))
        self._endpoint = RendezvousEndpoint(
            store_path=str(Path(self._directory) / "c10d-store"))

    @property
    def directory(self):
        return self._directory

    @property
    def endpoint(self):
        return self._endpoint

    def close(self):
        shutil.rmtree(self._directory, ignore_errors=True)


@dataclasses.dataclass(frozen=True)
class RendezvousEndpoint:
    """The picklable half of a rendezvous: where the ranks meet."""

    store_path: str


def _apply_input_changes_at_boundary(session, batch_id, records, influence_config, *,
                                     timeout, distributed):
    """Use the same prepare/decision protocol for local and distributed fits.

    Coordinator receipts survive caller timeouts. Device/worker timeouts still
    use the distributed runtime's existing fail-stop policy.
    """
    payload = json.dumps([records, influence_config], sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    with session._condition:
        if not hasattr(session, "_revision_batch_lock"):
            session._revision_batch_lock = threading.Lock()
            session._revision_batch_receipts = {}
        lock = session._revision_batch_lock

    def call(name, arguments):
        if distributed:
            return session._call(name, arguments, collective=False, return_all=True,
                                 timeout=timeout + COMMAND_ACK_GRACE_S)
        methods = {"reserve_input_boundary": session.reserve_input_boundary,
                   "cancel_input_boundary": session.cancel_input_boundary,
                   "prepare_input_batch": session.prepare_input_batch,
                   "finish_input_batch": session.finish_input_batch}
        return {0: methods[name](**arguments)}

    with lock:
        receipts = session._revision_batch_receipts
        receipt = receipts.get(batch_id)
        if receipt is not None and receipt["payload"] != payload:
            raise ValueError("Input batch ID reused with different content")
        if receipt is None:
            if any("result" not in previous for previous in receipts.values()):
                raise ValueError("Reconcile the outstanding input batch first")
            with session._condition:
                session._live_reservation_epoch += 1
                epoch = session._live_reservation_epoch
                status = session.status()
                target = int(status.get("current_iteration", 0))
            for _ in range(32):
                reservations = call("reserve_input_boundary", {
                    "target_iteration": target, "reservation_epoch": epoch,
                    "include_idle": True})
                if all(result.get("reserved") for result in reservations.values()):
                    break
                call("cancel_input_boundary", {"reservation_epoch": epoch})
                suggestions = [int(result["next_iteration"])
                               for result in reservations.values()
                               if not result.get("reserved")]
                next_target = max(suggestions)
                if next_target == target:
                    raise ValueError("Resident fit cannot accept inputs in its current state")
                target = next_target
            else:
                raise ValueError("Could not reserve a shared input boundary")
            receipt = receipts[batch_id] = {
                "payload": payload, "target": target, "epoch": epoch}
        if "result" in receipt:
            return copy.deepcopy(receipt["result"])
        if "preparations" not in receipt:
            captured_records, captured_config = json.loads(payload)
            receipt["preparations"] = call("prepare_input_batch", {
                "batch_id": batch_id, "records": captured_records,
                "influence_config": captured_config,
                "target_iteration": receipt["target"],
                "reservation_epoch": receipt["epoch"], "timeout": timeout})
        preparations = receipt["preparations"]
        canonical = preparations[min(preparations)]
        succeeded = all(result.get("prepared") for result in preparations.values())
        agreed = all(result.get("membership") == canonical.get("membership")
                     for result in preparations.values())
        install = succeeded and agreed
        outcomes = call("finish_input_batch", {
            "batch_id": batch_id, "install": install, "timeout": timeout})
        if install:
            result = {"applied": True, "iteration": receipt["target"],
                      "warnings": outcomes[min(outcomes)].get("warnings", []),
                      "prepare_seconds": max(item.get("prepare_seconds", 0) for item in preparations.values()),
                      "install_seconds": max(item.get("install_seconds", 0) for item in outcomes.values())}
        else:
            result = {"applied": False, "errors": {
                str(rank): result.get("error", "Ranks disagreed on prepared input membership")
                for rank, result in preparations.items()
                if not result.get("prepared") or not agreed}}
        receipt["result"] = result
        return copy.deepcopy(result)


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

    ``epoch`` fences commands that belong to an all-rank command epoch: they
    may only execute in the epoch that queued them, and a mismatch is a
    fail-stop rather than a cancellation, because a rank that ran the work of
    one epoch inside another has already diverged from its siblings. It is
    None for coordinator sub-operations (a rank-0-only checkpoint save),
    which do not participate in the epoch sequence at all.
    """

    kind: ClassVar[str] = "command"

    command_id: str = dataclasses.field(
        default_factory=lambda: uuid.uuid4().hex)
    session_generation: int = 0
    epoch: int | None = None
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
class RebuildModelCommand(SessionCommand):
    """Replace the model stage without reloading the session's inputs.

    Carries the request the rebuilt session is to be resolved from, because
    the whole point is that the configuration changes: the fitter thread
    installs these and re-derives its applied configuration from them.
    """

    kind: ClassVar[str] = "rebuild_model"

    paths: Any = None
    run: Any = None


@dataclasses.dataclass
class InputBatchCommand(SessionCommand):
    """Prepare or install an immutable batch while holding a worker boundary."""

    kind: ClassVar[str] = "input_batch"
    batch_id: str = ""
    action: str = "prepare"
    records: list = dataclasses.field(default_factory=list)
    influence_config: dict = dataclasses.field(default_factory=dict)
    reservation_epoch: int | None = None
    candidate: Any = dataclasses.field(default=None, repr=False)


@dataclasses.dataclass
class DtLossScheduleCommand(SessionCommand):
    """Install one Run's transient DT-loss window on the fitter thread."""

    kind: ClassVar[str] = "dt_loss_schedule"

    run_start: int = 0
    requested_iterations: int = 0
    schedule: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class SaveCheckpointCommand(SessionCommand):
    """Write a checkpoint from the fitter thread."""

    kind: ClassVar[str] = "save"

    path: str = ""


@dataclasses.dataclass
class ExportPreviewCommand(SessionCommand):
    """Export and publish one preview generation, on request.

    Previews are no longer a side effect of pausing or of resuming from a
    checkpoint: a client that wants one asks for one. Like a checkpoint save
    it is a coordinator sub-operation - only the publishing rank exports -
    so it carries no command epoch.
    """

    kind: ClassVar[str] = "export_preview"

    #: Compute the loss overlays alongside the surface. Off by default: they
    #: cost about as much as the surface and only the client knows whether
    #: anyone is looking at them.
    diagnostics: bool = False
    automatic: bool = False


@dataclasses.dataclass
class PreflightCheckpointCommand(SessionCommand):
    """Phase 1 of an in-session load: inspect a checkpoint on the CPU.

    Reads the file and validates it against this rank's live model; it never
    mutates resident state, so a refusal (on this or any sibling rank) leaves
    the session exactly as it was. An accepted preflight retains the loaded
    CPU payload for the apply command that follows it.
    """

    kind: ClassVar[str] = "preflight_checkpoint"

    path: str = ""


@dataclasses.dataclass
class ApplyCheckpointCommand(SessionCommand):
    """Phase 2 of an in-session load: move the preflighted payload in.

    Issued only after every rank accepted the same checkpoint. A failure here
    is fatal to the session rather than a refusal: the model and optimiser
    are partially written by then, and a partially loaded optimiser is not a
    state any rank may keep training from.
    """

    kind: ClassVar[str] = "apply_checkpoint"

    path: str = ""


@dataclasses.dataclass
class DiscardCheckpointCommand(SessionCommand):
    """Release a preflighted payload no rank will apply."""

    kind: ClassVar[str] = "discard_checkpoint"


class InteractiveFitSession:
    def __init__(self, paths: SpiralInputPaths, run: SpiralRunConfig,
                 preview: SpiralPreviewConfig, scroll: ScrollSpec,
                 status_callback=None, publishes_outputs=True,
                 event_callback=None, rank=0, world_size=1, local_rank=0,
                 rendezvous=None) -> None:
        # Who this session is in the job, passed in explicitly by whoever
        # spawned it. Nothing here reads RANK/WORLD_SIZE from the
        # environment; the DistributedContext is built from these values on
        # the fitter thread, where torch is imported.
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)
        self._rendezvous = rendezvous
        # The world size the fitter thread actually joined, which is what
        # per-rank count splitting divides by. Set once the process group
        # exists; until then nothing resolves a configuration.
        self._dist_world_size = self.world_size
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
        self._preview_diagnostics = False
        self._preview_source_iteration = None
        self._preview_schedule = None
        self._next_preview_iteration = None
        self._automatic_previews_disabled = False
        self._preview_session_id = uuid.uuid4().hex
        # The checkpoint the resident model is byte-identical to, or None
        # once a step has moved the parameters away from every saved file.
        # Set by a save, a load and a resume; the host pairs it with the
        # preview it published for the same model state.
        self._checkpoint_state = None
        # Set by every run; the default matters only for the interval before
        # the first one.
        self._autosave_on_pause = True
        # The FitContext this session owns, set on the fitter thread once the
        # device state is built. It doubles as the capability flag for the
        # fitter operations (save/preview/apply/configure).
        self._context = None
        # Commands queued for the fitter thread's pause boundary, oldest
        # first. Both the generation and the configuration revision fence
        # commands that were computed against a session state the fitter no
        # longer has.
        self._commands = []
        # The CPU-side checkpoint an accepted preflight is holding for the
        # apply command that follows it, as (path, payload). Only the fitter
        # thread ever reads or writes it.
        self._pending_checkpoint = None
        self.session_generation = 0
        self._config_revision = 0
        # The all-rank command epoch this rank has entered. It advances only
        # at an all-rank command barrier, identically on every rank, so two
        # ranks reporting the same epoch are executing the same command.
        self._command_epoch = 0
        # The epoch and configuration revision the current run was admitted
        # under. Re-checked at every pause boundary before another step.
        self._step_epoch = 0
        self._step_config_revision = 0
        # A input-batch reservation prevents this rank from crossing
        # the agreed DDP boundary while the coordinator completes the second
        # phase.  The iteration in progress is tracked separately so a claim
        # never targets a step that has already begun.
        self._live_reservation_iteration = None
        self._live_reservation_epoch = 0
        self._input_batches = {}
        self._iteration_in_progress = None
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
                "rank": getattr(self, "rank", 0),
                "world_size": getattr(self, "world_size", 1),
                # Published so the parent can tell "the ranks agree" from
                # "the ranks are in different commands" without a collective.
                "command_epoch": getattr(self, "_command_epoch", 0),
                "config_revision": getattr(self, "_config_revision", 0),
                "current_iteration": self._completed,
                "target_iteration": self._target,
                "session_horizon": None,
                "latest_metrics": copy.deepcopy(self._latest_metrics),
                "warnings": list(self._warnings), "error": self._error,
                "preview_manifest_path": self._preview_manifest,
                "preview_generation": self._preview_generation,
                "preview_source_iteration": getattr(
                    self, "_preview_source_iteration", None),
                "preview_schedule": copy.deepcopy(getattr(
                    self, "_preview_schedule", None)),
                "next_preview_iteration": getattr(
                    self, "_next_preview_iteration", None),
                "automatic_previews_disabled": bool(getattr(
                    self, "_automatic_previews_disabled", False)),
                # Whether this generation carries loss overlays, so the host
                # knows whether a diagnostics publication follows the surface.
                "preview_diagnostics": getattr(
                    self, "_preview_diagnostics", False),
                "checkpoint_state": copy.deepcopy(
                    getattr(self, "_checkpoint_state", None)),
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

    def _enter_epoch_locked(self, kind, barrier):
        """Advance this rank's command epoch at an all-rank command boundary.

        With a barrier (a rank inside a distributed job) the coordinator's
        epoch, command kind, configuration revision, and pending count are
        all validated against this rank's own state before the command is
        allowed to proceed; a mismatch raises, which fail-stops the rank
        rather than letting it run work its siblings are not running.

        Without one (a single-rank session, which is its own coordinator) the
        session issues the next epoch itself, so the epoch sequence and the
        checks that read it work identically in both shapes.
        """
        expected = self._command_epoch + 1
        if barrier is None:
            self._command_epoch = expected
            return expected
        violation = barrier.violation(
            rank=self.rank, expected_epoch=expected, kind=kind,
            config_revision=self._config_revision, pending=self._pending)
        if violation is not None:
            raise CommandBarrierViolation(violation)
        self._command_epoch = barrier.epoch
        return barrier.epoch

    def _check_step_barrier_locked(self):
        """Validate this rank's step preconditions at the pause boundary.

        Called immediately before the fitter is released into another
        training step. The epoch that admitted the run must still be the
        epoch this rank is in, and the configuration revision must be the one
        the run's queued configuration command established: a step taken
        under any other epoch or revision is a step the other ranks are not
        taking.
        """
        if self._command_epoch != self._step_epoch:
            raise CommandBarrierViolation(
                f"rank {self.rank} reached a step boundary in command epoch "
                f"{self._command_epoch}, but its run was admitted in epoch "
                f"{self._step_epoch}")
        if self._config_revision != self._step_config_revision:
            raise CommandBarrierViolation(
                f"rank {self.rank} reached a step boundary at configuration "
                f"revision {self._config_revision}, but its run was admitted "
                f"against revision {self._step_config_revision}")

    def _fit_main(self):
        context = None
        dist_context = None
        try:
            self._progress_reporter().begin("loading", "Importing Torch and fitter")
            self._transition(SessionState.Loading, "Importing Torch and fitter")
            import fit_spiral as fitter
            from ddp_helpers import (DistributedContext,
                                     maybe_destroy_distributed,
                                     maybe_init_distributed)

            # Explicit identity, and an endpoint the parent owns for the whole
            # session: no environment lookup and no free-port race.
            dist_context = maybe_init_distributed(
                DistributedContext(rank=self.rank,
                                   world_size=self.world_size,
                                   local_rank=self.local_rank),
                rendezvous=self._rendezvous)

            self._dist_world_size = dist_context.world_size
            config = self._resolve_session_config(self._dist_world_size)

            self._progress_reporter().begin("loading", "Loading fit inputs and model")
            self._transition(SessionState.Loading, "Loading fit inputs and model")
            # The scroll specification is used exactly as the dataset root
            # states it: its physical facts (name, resolution, outward sense)
            # and its Lasagna store layout. A session request may not restate
            # any of them, so there is nothing left here to override.
            scroll = self.scroll
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
                render_volume_scale=self.run_config.render_volume_scale,
                dist_context=dist_context)
            context.check_cuda_ready()
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
            with self._condition:
                # Prepared candidates own device tensors too. Release them on
                # this worker and resolve waiters when recovery is impossible.
                for batch in self._input_batches.values():
                    batch["prepare"].candidate = None
                for command in self._commands:
                    command.fail(self._error or "Resident fitter closed")
                self._commands.clear()
                self._prepared_input_batch_id = None
                self._live_reservation_iteration = None
                self._condition.notify_all()
            if context is not None:
                # Resource release runs here, on the owning fitter thread.
                context.close()
            if dist_context is not None:
                maybe_destroy_distributed(dist_context)
            self._progress_reporter().close()

    def _resolve_session_config(self, world_size):
        """Resolve this session's fitter configuration and publish it.

        The one place a session's applied configuration is derived, so a
        model-stage rebuild reproduces exactly the chain the initial build
        ran: the Python defaults, the resume checkpoint's stored ``cfg``, the
        request's advanced overrides, z-range/DDP count scaling, and finally
        the explicit sample counts that must not be scaled twice. It reads
        ``self.paths`` and ``self.run_config``, so replacing either and
        calling it again is all a rebuild needs.
        """
        from spiral_helpers import scale_and_split_counts

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
                for key in RETIRED_CONFIG_KEYS:
                    durable.pop(key, None)
                # Checkpoints store the durable subset of the schema
                # (see config.durable_config), so key sets compare
                # against that subset. A small explicit allowlist records
                # fields whose historical default is unambiguous; every
                # other key-set mismatch stays a strict error.
                durable_schema = set(durable_config(config))
                missing = durable_schema - set(durable)
                backfillable = set(BACKFILLABLE_CONFIG_DEFAULTS) | {
                    "z_begin", "z_end"}
                if set(durable) - durable_schema or missing - backfillable:
                    raise ValueError(
                        "Checkpoint configuration does not match the current schema")
                if missing:
                    assumed = {
                        **BACKFILLABLE_CONFIG_DEFAULTS,
                        "z_begin": int(self.run_config.z_begin),
                        "z_end": int(self.run_config.z_end),
                    }
                    durable.update(
                        {key: assumed[key] for key in missing})
                    warning = (
                        f"Checkpoint {self.paths.checkpoint} predates "
                        "defaultable fields in the stored configuration; "
                        "assuming "
                        + ", ".join(f"{key}={assumed[key]}"
                                    for key in sorted(missing))
                        + " from the session request")
                    print(warning)
                    with self._condition:
                        self._warnings.append(warning)
                durable = Config(durable).as_dict()
                # Keep the durable configuration aligned with the canonical
                # run window. The service restores a newly loaded checkpoint's
                # range into this run block; later dock edits remain free to
                # choose a narrower or otherwise compatible window.
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
        # The canonical run window (the request normally, or the restored
        # checkpoint window above) owns the applied configuration and Default
        # profile.
        config["z_begin"] = int(self.run_config.z_begin)
        config["z_end"] = int(self.run_config.z_end)
        # Saved workspace overrides may outlive any setting in the schema.
        # Report and remove unknown keys before applying overrides or counts.
        def warn(warning):
            print(warning)
            with self._condition:
                if warning not in self._warnings:
                    self._warnings.append(warning)
        advanced_config = filter_known_config_keys(
            self.run_config.config, config, label="advanced config", warn=warn)
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
            key: value for key, value in advanced_config.items()
            if key.startswith("sample_count_")
        })
        config.update(advanced_config)
        config["z_begin"] = int(self.run_config.z_begin)
        config["z_end"] = int(self.run_config.z_end)
        fields = Config.catalog()["schema"]["fields"]
        count_keys = tuple(
            key for key, spec in fields.items() if spec.get("scale_with_z"))
        scale_and_split_counts(
            config, self.run_config.z_begin, self.run_config.z_end,
            count_keys, world_size=world_size)
        if checkpoint_profile_config is None:
            scale_and_split_counts(
                default_advanced_config, self.run_config.z_begin,
                self.run_config.z_end, count_keys,
                world_size=world_size)
        config.update(explicit_sampling_counts)
        for key in ('z_begin', 'z_end'):
            default_advanced_config.pop(key, None)
        self.requested_config = dict(config)
        with self._condition:
            self._applied_config = copy.deepcopy(config)
            self._run_config = run_mutable_config(config)
            self._run_config_limits = self._run_config_limits_for(config)
            self._default_advanced_config = default_advanced_config
        self._publish_status()
        return config

    # Fitter-thread session driver.
    def _session_ready(self, context):
        """Adopt the built context and publish Idle.

        Runs on the fitter thread once build_device_state() has returned.
        Resuming from a checkpoint no longer exports a preview on the way to
        Idle: a preview is minutes of work a client may not want, and
        inspecting a checkpoint should not require it. A client that wants to
        see the restored model asks for one.
        """
        with self._condition:
            self._context = context
            self._completed = self._target = context.start_iteration
            self._output_path = context.out_path
        self._record_checkpoint_state(
            getattr(context, "resume_path", None), context.start_iteration)
        self._progress_reporter().clear()
        self._transition(SessionState.Idle, IDLE_PHASE)

    def _record_checkpoint_state(self, path, completed_iterations):
        """Name the checkpoint the resident model now equals, if any.

        The digest is taken from the live model rather than read out of the
        file: it is then right for checkpoints written before the field
        existed, and for a resume that migrated the payload on the way in.
        """
        state = None
        digest = getattr(self._context, "model_state_digest", None)
        if path and digest is not None:
            try:
                state = {
                    "path": str(path),
                    "model_state_sha256": digest(),
                    "completed_iterations": int(completed_iterations),
                }
            except Exception as exc:
                print(f"WARNING: could not digest the resident model state: "
                      f"{type(exc).__name__}: {exc}", flush=True)
        with self._condition:
            self._checkpoint_state = state
        return state

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

    def _next_command_locked(self):
        """Dequeue the next command this boundary may execute.

        Live commands hold a reserved optimizer boundary that may still lie
        ahead of this rank: with uneven GPU progress the coordinator reserves
        the boundary the fastest rank suggested. Such a command stays queued
        until this rank reaches that boundary rather than being cancelled as
        stale here. Outside Running, nothing is deferred: the reserved
        boundary can no longer be reached and the command resolves through
        its own no-future-step path or the stale check.
        """
        if not self._commands:
            return None
        held_batch = getattr(self, "_prepared_input_batch_id", None)
        if held_batch is not None:
            for index, queued in enumerate(self._commands):
                if (isinstance(queued, InputBatchCommand)
                        and queued.batch_id == held_batch and queued.action != "prepare"):
                    return self._commands.pop(index)
            return None
        if self._state is SessionState.Running:
            for index, queued in enumerate(self._commands):
                if (getattr(queued, "reservation_epoch", None) is not None
                        and queued.expected_iteration is not None
                        and queued.expected_iteration > self._completed):
                    continue
                return self._commands.pop(index)
            return None
        return self._commands.pop(0)

    def _release_live_reservation_locked(self, command):
        """Drop the boundary a live command holds, if it is still current."""
        reservation_epoch = getattr(command, "reservation_epoch", None)
        if (reservation_epoch is not None
                and self._live_reservation_iteration is not None
                and self._live_reservation_epoch == reservation_epoch):
            self._live_reservation_iteration = None
            self._condition.notify_all()

    def _live_expected_config_revision_locked(self):
        """The revision a live command will find when it executes.

        A Run's queued ConfigureCommand bumps the revision as it is applied,
        ahead of every live command queued behind it; a live command queued
        during that window must expect the post-configure revision.
        """
        return self._config_revision + sum(
            isinstance(queued, ConfigureCommand) for queued in self._commands)

    def wait_for_iteration(self, iteration):
        while True:
            with self._condition:
                if self._shutdown:
                    raise _SessionShutdown()
                # Commands are drained before the pending check so inputs
                # queued by run() are incorporated before the next step begins.
                command = self._next_command_locked()
                if command is None:
                    if self._pending > 0:
                        reserved_iteration = getattr(
                            self, "_live_reservation_iteration", None)
                        if (reserved_iteration is not None
                                and iteration >= reserved_iteration):
                            self._condition.wait()
                            continue
                        # Last gate before another training step: this rank
                        # only steps in the epoch and configuration revision
                        # its run was admitted under.
                        self._check_step_barrier_locked()
                        self._iteration_in_progress = iteration
                        return
                    self._condition.wait()
                    continue
                if (command.epoch is not None
                        and command.epoch != self._command_epoch):
                    raise CommandBarrierViolation(
                        f"rank {self.rank} dequeued {command.kind} command "
                        f"{command.command_id} from epoch {command.epoch} "
                        f"while in epoch {self._command_epoch}")
                stale = command.stale_reason(
                    session_generation=self.session_generation,
                    iteration=self._completed,
                    config_revision=self._config_revision)
                if stale is not None:
                    # A cancelled live command must not leave its boundary
                    # held: the step loop would otherwise park at it forever.
                    self._release_live_reservation_locked(command)
            if stale is not None:
                if isinstance(command, InputBatchCommand) and command.action == "prepare":
                    command.complete(prepared=False, error=stale)
                else:
                    command.cancel(stale)
                continue
            if isinstance(command, InputBatchCommand):
                self._run_input_batch(command)
                continue
            if isinstance(command, DtLossScheduleCommand):
                self._run_dt_loss_schedule(command)
                continue
            if isinstance(command, ConfigureCommand):
                self._run_configuration(command)
                continue
            if isinstance(command, RebuildModelCommand):
                # Not guarded past the point of release, for the same reason
                # apply_checkpoint is not: see _run_model_rebuild.
                self._run_model_rebuild(command)
                continue
            if isinstance(command, PreflightCheckpointCommand):
                self._run_checkpoint_preflight(command)
                continue
            if isinstance(command, ApplyCheckpointCommand):
                # Not guarded: a failure here is deliberately allowed to
                # unwind the fitter thread and fail the session.
                self._run_checkpoint_apply(command)
                continue
            if isinstance(command, DiscardCheckpointCommand):
                self._run_checkpoint_discard(command)
                continue
            if isinstance(command, ExportPreviewCommand):
                self._run_export_preview(command)
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
        checkpoint_state = None
        try:
            self._progress_reporter().begin(
                "saving_checkpoint", "Saving checkpoint",
                detail=Path(command.path).name)
            path = self._context.save_checkpoint(
                command.path, self._completed)
            checkpoint_state = self._record_checkpoint_state(
                path, self._completed)
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
                command.complete(
                    path=path,
                    model_state_sha256=(checkpoint_state or {}).get(
                        "model_state_sha256"))
            else:
                command.fail(error)

    def _release_pending_checkpoint(self):
        """Drop any retained CPU payload. Fitter thread only."""
        pending, self._pending_checkpoint = self._pending_checkpoint, None
        del pending

    def _leave_checkpoint_load(self, phase=None):
        """Return this rank to Idle after a load ended without applying."""
        self._progress_reporter().clear()
        with self._condition:
            self._transition_locked(
                SessionState.Idle, phase or IDLE_PHASE)
        self._publish_status()

    def _run_checkpoint_preflight(self, command):
        """Inspect a checkpoint on the CPU without touching resident state.

        The session leaves Idle for the whole two-phase load: an in-flight
        load must not race a run admitted against the model it is replacing.
        Nothing here writes model, optimiser, scheduler or RNG state, so any
        refusal - this rank's or, through the coordinator, a sibling's -
        leaves the live session exactly as it was.
        """
        checkpoint = None
        try:
            if self._context is None:
                raise RuntimeError(
                    "The resident fitter does not support loading a checkpoint")
            self._release_pending_checkpoint()
            with self._condition:
                self._transition_locked(
                    SessionState.Loading, "Inspecting checkpoint",
                    reason=f"checkpoint preflight {command.command_id}")
            self._publish_status()
            self._progress_reporter().begin(
                "loading", "Inspecting checkpoint",
                detail=Path(command.path).name)
            from checkpoint_io import load_checkpoint_cpu
            checkpoint = load_checkpoint_cpu(command.path)
            verdict = self._context.inspect_checkpoint(
                checkpoint, source=command.path)
        except Exception as exc:
            del checkpoint
            self._release_pending_checkpoint()
            self._leave_checkpoint_load()
            command.fail(f"{type(exc).__name__}: {exc}")
            return
        if not verdict.accepted:
            del checkpoint
            self._leave_checkpoint_load()
            command.fail(verdict.message())
            return
        # Retained for the apply command: re-reading the file between the two
        # phases would mean applying bytes no rank inspected.
        self._pending_checkpoint = (command.path, checkpoint)
        with self._condition:
            self._transition_locked(
                SessionState.Loading, "Checkpoint accepted")
        self._publish_status()
        command.complete(**verdict.to_dict())

    def _run_checkpoint_apply(self, command):
        """Move the preflighted checkpoint into the live fit.

        Reached only once every rank accepted this exact path. From the first
        ``load_state_dict`` there is no unchanged session left to return to,
        so a failure is re-raised: it unwinds the fitter thread, fails this
        session, and (under DDP) the parent watchdog takes the siblings down
        rather than leaving a rank training from a partially loaded optimiser.
        """
        pending = self._pending_checkpoint
        if pending is None or pending[0] != command.path:
            self._release_pending_checkpoint()
            self._leave_checkpoint_load()
            command.fail(
                f"No inspected checkpoint is pending for {command.path}")
            return
        self._pending_checkpoint = None
        path, checkpoint = pending
        try:
            self._progress_reporter().begin(
                "loading", "Restoring model and optimizer",
                detail=Path(path).name)
            completed = self._context.apply_checkpoint(
                checkpoint, realign_lr=True)
        except BaseException as exc:
            error = (f"Applying checkpoint {path} failed after every rank "
                     f"accepted it; the resident model and optimiser state "
                     f"are partial: {type(exc).__name__}: {exc}")
            command.fail(error)
            raise RuntimeError(error) from exc
        finally:
            del checkpoint, pending
        self._progress_reporter().clear()
        checkpoint_state = self._record_checkpoint_state(path, completed)
        with self._condition:
            # The durable step the checkpoint reached is the session's
            # iteration now; the LR schedule was realigned to it rather than
            # reset to zero.
            self._completed = self._target = completed
            self._run_start_completed = completed
            self._latest_metrics = {}
            self.input_manifest["checkpoint"] = path
            self._config_revision += 1
            revision = self._config_revision
            self._transition_locked(
                SessionState.Idle, IDLE_PHASE)
        self._publish_status()
        command.complete(path=path, completed_iterations=completed,
                         config_revision=revision,
                         model_state_sha256=(checkpoint_state or {}).get(
                             "model_state_sha256"))

    def _run_checkpoint_discard(self, command):
        """Release a payload no rank will apply and return to Idle."""
        self._release_pending_checkpoint()
        with self._condition:
            loading = self._state is SessionState.Loading
        if loading:
            self._leave_checkpoint_load()
        command.complete(discarded=True)

    def _preview_snapshot_barrier(self):
        """Synchronize every fitter rank around rank 0's raw snapshot copy."""
        dist_context = getattr(getattr(self, "_context", None), "dist", None)
        if not bool(getattr(dist_context, "is_distributed", False)):
            return
        import torch.distributed as dist
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[int(dist_context.local_rank)])
        else:
            dist.barrier()

    def _share_preview_capture_error(self, error):
        """Broadcast rank 0's capture outcome so cadence stays rank-consistent."""
        dist_context = getattr(getattr(self, "_context", None), "dist", None)
        if not bool(getattr(dist_context, "is_distributed", False)):
            return error
        import torch
        import torch.distributed as dist
        payload = [error if bool(getattr(
            dist_context, "is_main_process", False)) else None]
        device = None
        if dist.get_backend() == "nccl":
            device = torch.device("cuda", int(dist_context.local_rank))
        dist.broadcast_object_list(payload, src=0, device=device)
        return payload[0]

    def _run_export_preview(self, command):
        """Export and publish one preview generation at the pause boundary."""
        with self._condition:
            previous_state = self._state
            previous_phase = self._phase
            if previous_state is SessionState.Running:
                self._phase = "Capturing preview snapshot"
            else:
                self._transition_locked(
                    SessionState.ExportingPreview, "Exporting preview",
                    reason=f"preview command {command.command_id}")
        error = None
        collective_failure = None
        try:
            self._preview_snapshot_barrier()
            try:
                if getattr(self, "publishes_outputs", True):
                    self._progress_reporter().begin(
                        "exporting_preview", "Exporting preview")
                    self._publish_preview(diagnostics=command.diagnostics)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            error = self._share_preview_capture_error(error)
            self._preview_snapshot_barrier()
        except BaseException as exc:
            collective_failure = exc
        finally:
            self._progress_reporter().clear()
            with self._condition:
                if self._state is previous_state:
                    self._phase = previous_phase
                    self._condition.notify_all()
                else:
                    self._transition_locked(previous_state, previous_phase)
                manifest = self._preview_manifest
                generation = self._preview_generation
                source_iteration = getattr(
                    self, "_preview_source_iteration", None)
            self._publish_status()
            if collective_failure is not None:
                command.fail(
                    f"Preview snapshot synchronization failed: "
                    f"{type(collective_failure).__name__}: {collective_failure}")
            elif error is None:
                result = {
                    "preview_manifest_path": manifest,
                    "preview_generation": generation,
                }
                if source_iteration is not None:
                    result["source_fit_iteration"] = source_iteration
                command.complete(**result)
            else:
                if command.automatic:
                    with self._condition:
                        self._automatic_previews_disabled = True
                        self._warnings.append(
                            f"Automatic previews disabled for this Run: {error}")
                command.fail(error)
        if collective_failure is not None:
            raise collective_failure

    def _run_input_batch(self, command):
        started = time.perf_counter()
        if command.action == "prepare":
            self._prepared_input_batch_id = command.batch_id
            try:
                command.candidate = self._context.prepare_input_changes(
                    command.records, command.influence_config,
                    current_iteration=self._completed, target_iteration=self._target)
            except (ValueError, OSError) as exc:
                # Content/domain preparation failures leave the active context
                # intact. Hold this rank until every sibling is discarded.
                command.complete(prepared=False, error=f"{type(exc).__name__}: {exc}")
                return
            except BaseException as exc:
                command.fail(f"{type(exc).__name__}: {exc}")
                raise  # Unexpected worker/device failures remain fail-stop.
            candidate = command.candidate
            command.complete(prepared=True, prepare_seconds=time.perf_counter() - started, membership={
                "inputs": candidate._workspace_membership,
                "verified_patches": list(candidate.verified_patches),
            })
            return
        prepared = self._input_batches[command.batch_id]["prepare"]
        if command.action == "install":
            if prepared.candidate is None:
                command.fail("Input batch has no prepared candidate")
                return
            try:
                warnings = self._context.install_input_changes(prepared.candidate)
            except BaseException as exc:
                command.fail(f"{type(exc).__name__}: {exc}")
                raise  # Installation must never resume partially changed ranks.
            result = {"applied": True, "warnings": warnings,
                      "install_seconds": time.perf_counter() - started}
        else:
            result = {"discarded": True}
        prepared.candidate = None
        with self._condition:
            self._prepared_input_batch_id = None
            self._release_live_reservation_locked(command)
            running = self._state is SessionState.Running
        # Preparation shares the fitter's progress reporter. Restore its
        # owner after both install and discard, before training can resume.
        if running:
            self._begin_optimization_progress()
        else:
            self._progress_reporter().clear()
            with self._condition:
                self._transition_locked(SessionState.Idle, IDLE_PHASE)
        self._publish_status()
        command.complete(**result)

    def prepare_input_batch(self, batch_id, records, influence_config=None, *,
                            target_iteration, reservation_epoch,
                            timeout=INPUT_BATCH_TIMEOUT_S):
        payload = json.dumps([records, influence_config or {}], sort_keys=True,
                             allow_nan=False, separators=(",", ":"))
        with self._condition:
            if self._state not in {SessionState.Idle, SessionState.Running}:
                raise RuntimeError(getattr(self, "_error", None) or "Resident fitter is unavailable")
            batches = self._input_batches
            batch = batches.get(batch_id)
            if batch is not None:
                if batch["payload"] != payload:
                    raise ValueError("Input batch ID reused with different content")
                command = batch["prepare"]
            else:
                if (self._live_reservation_iteration != target_iteration
                        or self._live_reservation_epoch != reservation_epoch):
                    raise ValueError("Input batch does not own its worker boundary")
                captured_records, captured_config = json.loads(payload)
                command = InputBatchCommand(
                    batch_id=batch_id, session_generation=self.session_generation,
                    expected_iteration=target_iteration,
                    expected_config_revision=self._live_expected_config_revision_locked(),
                    reservation_epoch=reservation_epoch, records=captured_records,
                    influence_config=captured_config)
                batches[batch_id] = {"payload": payload, "prepare": command}
                self._commands.append(command)
                self._condition.notify_all()
        return self._wait_input_batch(command, timeout)

    def _wait_input_batch(self, command, timeout):
        # A timeout is an unknown outcome. Keep the immutable command and its
        # boundary; the same batch ID can reconcile without executing twice.
        deadline = time.monotonic() + timeout
        while not command.wait(min(.1, max(0, deadline - time.monotonic()))):
            with self._condition:
                if self._state in {SessionState.Error, SessionState.Closing}:
                    raise RuntimeError(getattr(self, "_error", None) or "Resident fitter closed")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Input batch {command.batch_id} outcome is unknown")
        if command.error is not None:
            raise RuntimeError(command.error)
        return dict(command.result)

    def finish_input_batch(self, batch_id, *, install, timeout=INPUT_BATCH_TIMEOUT_S):
        with self._condition:
            batch = self._input_batches[batch_id]
            prepared = batch["prepare"]
            if not prepared.done.is_set():
                raise ValueError("Input preparation is still running")
            if install and not prepared.result.get("prepared"):
                raise ValueError("Input preparation did not succeed")
            action = "install" if install else "discard"
            command = batch.get("finish")
            if command is not None:
                if command.action != action:
                    raise ValueError("Input batch already has a different decision")
            else:
                command = InputBatchCommand(
                    batch_id=batch_id, action=action,
                    session_generation=self.session_generation,
                    expected_iteration=prepared.expected_iteration if install else None,
                    reservation_epoch=prepared.reservation_epoch)
                batch["finish"] = command
                self._commands.append(command)
                self._condition.notify_all()
        return self._wait_input_batch(command, timeout)

    @staticmethod
    def _run_config_limits_for(config):
        """Per-Run ceilings the resident prepared pools impose."""
        return {
            'track_max_track_crossing_per_step': max(
                int(config.get('track_crossing_precompute_max', 0)),
                int(config.get('track_max_track_crossing_per_step', 0))),
        }

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
                abandoned = [queued for queued in self._commands
                             if isinstance(queued, DtLossScheduleCommand)]
                self._commands = [queued for queued in self._commands
                                  if queued not in abandoned]
                # No step follows a failed configure; a live boundary held
                # for this Run has nothing left to wait for.
                self._live_reservation_iteration = None
                self._warnings.append(f"Run configuration failed: {error}")
                if previous_run_config is not None:
                    self._run_config.update(previous_run_config)
                self._transition_locked(
                    SessionState.Idle, IDLE_PHASE,
                    reason="run configuration failed")
            self._clear_context_run_state()
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
                    # A role toggle loads or drops point-collection
                    # documents; the manifest names what is resident.
                    context_paths = getattr(self._context, 'paths', None)
                    if context_paths is not None and 'pcls' in self.input_manifest:
                        self.input_manifest['pcls'] = \
                            context_paths.manifest()['pcls']
                    # A crossing-ceiling change re-prepared the track pool
                    # with the new ceiling; advertise it.
                    self._run_config_limits = self._run_config_limits_for(
                        self._applied_config)
            with self._condition:
                self._config_revision += 1
                if command.epoch is not None:
                    self._step_config_revision = self._config_revision
            command.complete(config_revision=self._config_revision)
            if getattr(self, "_state", None) is SessionState.Running:
                self._begin_optimization_progress()
            else:
                self._progress_reporter().clear()

    def _clear_context_run_state(self):
        """Clear all transient fitter state installed for the current Run."""
        if self._context is None:
            return
        clear = getattr(self._context, "clear_interactive_run_state", None)
        if clear is not None:
            clear()
            return
        # Source compatibility for small in-process test/embedder contexts.
        clear = getattr(self._context, "clear_interactive_influence", None)
        if clear is not None:
            clear()
        clear = getattr(self._context, "clear_dt_loss_schedule", None)
        if clear is not None:
            clear()

    def _run_dt_loss_schedule(self, command):
        """Apply the Run DT schedule after all other Run setup commands."""
        try:
            if self._context is None:
                raise RuntimeError(
                    "The resident fitter does not support DT-loss schedules")
            self._context.configure_dt_loss_schedule(
                command.run_start, command.requested_iterations,
                dict(command.schedule))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            with self._condition:
                self._pending = 0
                self._target = self._completed
                self._warnings.append(f"Run DT-loss scheduling failed: {error}")
                if self._state is SessionState.Running:
                    self._transition_locked(
                        SessionState.Idle, IDLE_PHASE,
                        reason="DT-loss scheduling failed")
            self._clear_context_run_state()
            command.fail(error)
            self._progress_reporter().clear()
            self._publish_status()
        else:
            command.complete(
                resume_iteration=getattr(
                    self._context, "run_dt_resume_iteration", None))

    def _run_model_rebuild(self, command):
        """Rebuild the model stage in place, on the fitter thread.

        The session object, its thread, its output directory, its loaded host
        inputs and its brick pools all survive; the model, optimiser,
        scheduler and everything constructed after them are replaced from the
        request this command carries, and the session's iteration returns to
        whatever the rebuilt model resumed from — the state a full rebuild
        with the same request would have left.

        Resolving the new configuration can fail, and does so harmlessly: the
        live model is still the one the session had. From the moment
        rebuild_model_state() releases it there is no unchanged session left
        to refuse into, so a failure there is re-raised and takes the fitter
        thread (and, under DDP, its siblings) down rather than leaving a rank
        without a model.
        """
        try:
            if self._context is None:
                raise RuntimeError(
                    "The resident fitter does not support rebuilding the model")
            with self._condition:
                self._transition_locked(
                    SessionState.Loading, "Rebuilding the model",
                    reason=f"model rebuild {command.command_id}")
            self._publish_status()
            self._progress_reporter().begin("loading", "Rebuilding the model")
            paths, run = command.paths, command.run
            previous_paths, previous_run = self.paths, self.run_config
            self.paths, self.run_config = paths, run
            try:
                config = self._resolve_session_config(self._dist_world_size)
            except BaseException:
                self.paths, self.run_config = previous_paths, previous_run
                self._resolve_session_config(self._dist_world_size)
                raise
        except Exception as exc:
            self._progress_reporter().clear()
            with self._condition:
                self._warnings.append(f"Model rebuild failed: {exc}")
                self._transition_locked(
                    SessionState.Idle, IDLE_PHASE,
                    reason="model rebuild refused")
            command.fail(f"{type(exc).__name__}: {exc}")
            self._publish_status()
            return
        context = self._context
        # Run-boundary settings that changed alongside the model-stage ones
        # are applied through the same path a Run would use, while the live
        # model still exists: a refusal there is harmless, and the host-side
        # effects (strata, strips, track pool) survive the model release
        # below. rebuild_stage() ignored these keys when it chose "model".
        run_changes = {
            key: value for key, value in run_mutable_config(config).items()
            if context.config.get(key) != value
        }
        if run_changes:
            try:
                context.apply_config(
                    run_changes, current_iteration=self._completed)
            except Exception as exc:
                self._progress_reporter().clear()
                self.paths, self.run_config = previous_paths, previous_run
                self._resolve_session_config(self._dist_world_size)
                with self._condition:
                    self._warnings.append(f"Model rebuild failed: {exc}")
                    self._transition_locked(
                        SessionState.Idle, IDLE_PHASE,
                        reason="model rebuild refused")
                command.fail(f"{type(exc).__name__}: {exc}")
                self._publish_status()
                return
        context.config.update(config)
        context.paths = paths
        context.resume_path = paths.checkpoint or None
        context.resume_step = (run.legacy_checkpoint_step
                               if paths.checkpoint else 0)
        try:
            context.rebuild_model_state()
        except BaseException as exc:
            error = (f"Rebuilding the model failed after the previous one was "
                     f"released; this session has no model: "
                     f"{type(exc).__name__}: {exc}")
            command.fail(error)
            raise RuntimeError(error) from exc
        self._progress_reporter().clear()
        self._record_checkpoint_state(
            paths.checkpoint or None, context.start_iteration)
        with self._condition:
            self.input_manifest = paths.manifest()
            self._completed = self._target = context.start_iteration
            self._pending = 0
            self._config_revision += 1
            if command.epoch is not None:
                self._step_config_revision = self._config_revision
            self._transition_locked(SessionState.Idle, IDLE_PHASE)
        command.complete(config_revision=self._config_revision,
                         current_iteration=context.start_iteration)
        self._publish_status()

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
            self._iteration_in_progress = None
            self._completed = completed_iterations
            self._checkpoint_state = None
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
            schedule = getattr(self, "_preview_schedule", None)
            schedule_due = (
                schedule is not None
                and not getattr(self, "_automatic_previews_disabled", False)
                and self._next_preview_iteration is not None
                and self._completed >= self._next_preview_iteration)
            final_due = (
                schedule is not None
                and not getattr(self, "_automatic_previews_disabled", False)
                and pause
                and getattr(self, "_preview_source_iteration", None)
                    != self._completed)
            if schedule_due or final_due:
                self._commands.append(ExportPreviewCommand(
                    session_generation=self.session_generation,
                    expected_iteration=self._completed,
                    diagnostics=bool(schedule.get("diagnostics", False)),
                    automatic=True))
                if schedule_due:
                    cadence = int(schedule["cadence_iterations"])
                    while self._next_preview_iteration <= self._completed:
                        self._next_preview_iteration += cadence
                self._condition.notify_all()
        self._progress_reporter().update(run_step)
        self._publish_status()
        autosave_wanted = (getattr(self, "publishes_outputs", True)
                           and getattr(self, "_autosave_on_pause", True))
        if pause:
            self._clear_context_run_state()
            # Pausing writes the durable autosave when the run asked for it,
            # and does nothing else. Exporting a preview is a request of its
            # own: it costs minutes, and a client that wants one after a
            # pause asks for one.
            if autosave_wanted:
                self._transition(SessionState.Saving, "Autosaving checkpoint")
                self._write_autosave()
            self._progress_reporter().clear()
            self._transition(SessionState.Idle, IDLE_PHASE)
        elif (autosave_wanted
              and self._completed % AUTOSAVE_INTERVAL_ITERATIONS == 0):
            # A long run refreshes the same autosave at a fixed cadence so a
            # crash mid-run loses at most one interval. The session stays
            # Running; only the phase says what the fitter thread is doing.
            self._transition(SessionState.Running, "Autosaving checkpoint")
            try:
                self._write_autosave()
            except Exception as exc:
                # The previous autosave is intact (the write is a temp file
                # plus an atomic replace), so a failed refresh is a warning
                # on the status rather than the end of the run.
                with self._condition:
                    self._warnings.append(
                        f"Autosave at iteration {self._completed} failed: "
                        f"{type(exc).__name__}: {exc}")
            self._progress_reporter().clear()
            self._begin_optimization_progress()
            self._transition(SessionState.Running, "Optimizing")

    def _write_autosave(self):
        """Refresh ``checkpoint_autosave.ckpt`` and its sidecar in place.

        The previous autosave is only ever replaced by ``os.replace`` after
        the new archive is fully written and fsynced (see
        ``FitContext.save_checkpoint``), so an interrupted write leaves the
        last good checkpoint untouched. The sidecar follows the same rule.
        """
        self._progress_reporter().begin(
            "saving_checkpoint", "Autosaving checkpoint",
            detail=AUTOSAVE_CHECKPOINT_NAME)
        autosave = str(Path(self._output_path) / AUTOSAVE_CHECKPOINT_NAME)
        self._context.save_checkpoint(autosave, self._completed)
        checkpoint_state = self._record_checkpoint_state(autosave, self._completed)
        # Name the file beside itself. An always-loaded service picks its
        # startup autosave from these sidecars alone: the output root it
        # belongs to, the dataset it was fit against, and how far it got.
        # Without one the checkpoint is inert.
        write_autosave_metadata(
            autosave,
            session_namespace=self.paths.output_directory,
            dataset_root=self.paths.dataset_root,
            completed_iterations=self._completed,
            model_state_sha256=(checkpoint_state or {}).get("model_state_sha256"))
        return autosave

    def _publish_preview(self, diagnostics=False):
        with self._condition:
            generation = self._preview_generation + 1
            source_iteration = getattr(self, "_completed", 0)
        generation_path = (Path(self.paths.output_directory) / ".spiral-preview" /
                           self._preview_session_id / f"generation-{generation}")
        surface_id = f"spiral-output-generation-{generation}"
        manifest = self._context.export_preview(
            str(generation_path), surface_id, diagnostics=diagnostics)
        manifest = dict(manifest)
        manifest["source_fit_iteration"] = int(source_iteration)
        manifest_path = Path(str(manifest["manifest_path"]))
        if manifest_path.is_file():
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with self._condition:
            self._preview_generation = generation
            self._preview_manifest = str(manifest_path)
            self._preview_diagnostics = bool(diagnostics)
            self._preview_source_iteration = int(source_iteration)
        # The host callback only claims this immutable raw generation.
        # Flattening, mapping, indexing, and transfer continue independently.
        self._publish_status()

    # Coordinator-thread commands.
    def run(self, count, influence_config=None, run_config=None, path_changes=None,
            autosave_on_pause=True, preview_schedule=None,
            dt_loss_schedule=None, barrier=None):
        if count < 1:
            raise ValueError("iterations must be at least 1")
        with self._condition:
            if self._state is not SessionState.Idle:
                raise RuntimeError(
                    f"Run is not allowed while session state is "
                    f"{self._state.name}")
            # The safe boundary: this rank is idle, and it validates the
            # coordinator's epoch, kind, configuration revision, and pending
            # count before anything is queued for the step loop.
            epoch = self._enter_epoch_locked("run", barrier)
            # Whether this run's pause writes the durable autosave: a
            # property of the run that is pausing, decided at admission
            # rather than read from configuration at the boundary.
            self._autosave_on_pause = bool(autosave_on_pause)
            self._preview_schedule = copy.deepcopy(preview_schedule)
            self._next_preview_iteration = (
                self._completed + int(preview_schedule["cadence_iterations"])
                if preview_schedule else None)
            self._automatic_previews_disabled = False
            dt_loss_schedule = dict(dt_loss_schedule or {
                "enabled": False, "last_fraction": 0.25})
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
            setup_config_revision = self._config_revision
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
                # Apply the requested configuration before taking any steps.
                self._commands.append(ConfigureCommand(
                    session_generation=self.session_generation,
                    epoch=epoch,
                    expected_iteration=self._completed,
                    expected_config_revision=self._config_revision,
                    config=run_config, path_changes=path_changes,
                    previous_run_config=previous_run_config))
                self._run_config.update(run_config)
                setup_config_revision += 1
            # Every Run installs its schedule on the fitter thread, including
            # runs with no configuration changes. It is
            # deliberately last so the first step sees all setup atomically.
            self._commands.append(DtLossScheduleCommand(
                session_generation=self.session_generation,
                epoch=epoch,
                expected_iteration=self._completed,
                expected_config_revision=setup_config_revision,
                run_start=self._completed,
                requested_iterations=count,
                schedule=dt_loss_schedule))
            self._pending = count
            self._run_start_completed = self._completed
            self._target = target
            self._step_epoch = epoch
            # The revision this run's steps must run under. A configuration
            # command queued above bumps it as it is applied, before the
            # first step, and moves this expectation with it.
            self._step_config_revision = self._config_revision
            self._transition_locked(SessionState.Running, "Optimizing")
            self._begin_optimization_progress()
            self._condition.notify_all()
            return self._target

    def reserve_input_boundary(self, target_iteration, reservation_epoch, *,
                                   include_idle=False):
        """Phase one: hold one still-future optimizer boundary."""
        with self._condition:
            if (self._state is not SessionState.Running
                    and not (include_idle and self._state is SessionState.Idle)):
                return {"reserved": False, "no_future_step": True,
                        "next_iteration": self._completed}
            target_iteration = int(target_iteration)
            in_progress = self._iteration_in_progress
            if (target_iteration < self._completed
                    or target_iteration > self._target
                    or (target_iteration == self._target and not include_idle)
                    or (in_progress is not None
                        and target_iteration <= in_progress)):
                suggestion = (in_progress + 1
                              if in_progress is not None else self._completed)
                return {"reserved": False,
                        "no_future_step": suggestion >= self._target,
                        "next_iteration": suggestion}
            if (self._live_reservation_iteration is not None
                    and (self._live_reservation_iteration != target_iteration
                         or self._live_reservation_epoch != reservation_epoch)):
                return {"reserved": False, "no_future_step": True,
                        "next_iteration": (in_progress + 1
                                           if in_progress is not None
                                           else self._completed)}
            self._live_reservation_iteration = target_iteration
            self._live_reservation_epoch = int(reservation_epoch)
            self._condition.notify_all()
            return {"reserved": True, "target_iteration": target_iteration,
                    "reservation_epoch": int(reservation_epoch)}

    def cancel_input_boundary(self, reservation_epoch):
        with self._condition:
            if self._live_reservation_epoch == int(reservation_epoch):
                self._live_reservation_iteration = None
                self._condition.notify_all()
        return {"cancelled": True}

    def apply_input_changes(self, batch_id, records, influence_config=None, *,
                            timeout=INPUT_BATCH_TIMEOUT_S):
        return _apply_input_changes_at_boundary(
            self, batch_id, records, influence_config or {}, timeout=timeout,
            distributed=False)

    def stop(self, barrier=None):
        with self._condition:
            if self._state is not SessionState.Running:
                raise RuntimeError("Session is not running")
            self._enter_epoch_locked("stop", barrier)
            self._stop_requested = True
            self._live_reservation_iteration = None
            # If stop landed exactly between steps, permit the one boundary
            # needed to observe _stop_requested under the stop epoch.
            self._step_epoch = self._command_epoch
            self._condition.notify_all()

    def disable_automatic_previews(self, error=None):
        """Stop this Run's cadence after a capture/publication failure."""
        with self._condition:
            self._automatic_previews_disabled = True
            abandoned = [command for command in self._commands
                         if isinstance(command, ExportPreviewCommand)
                         and command.automatic]
            self._commands = [command for command in self._commands
                              if command not in abandoned]
            if error:
                warning = f"Automatic previews disabled for this Run: {error}"
                if warning not in self._warnings:
                    self._warnings.append(warning)
            self._condition.notify_all()
        for command in abandoned:
            command.cancel("automatic previews disabled")
        self._publish_status()
        return {"disabled": True}

    def save_checkpoint(self, path, timeout=120.0):
        with self._condition:
            if self._state is not SessionState.Idle:
                raise RuntimeError(
                    f"Checkpoint save is not allowed in {self._state.name}")
            # A checkpoint save is a coordinator sub-operation, not an
            # all-rank command: it runs on the publishing rank only and
            # therefore carries no epoch.
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

    def _queue_command(self, command, timeout):
        with self._condition:
            self._commands.append(command)
            self._condition.notify_all()
        if not command.wait(timeout):
            with self._condition:
                if not command.done.is_set():
                    if command in self._commands:
                        self._commands.remove(command)
                        command.cancel(f"{command.kind} command timed out after {timeout:g}s")
                        self._condition.notify_all()
                    raise TimeoutError(f"{command.kind} command timed out after {timeout:g}s")
        if command.error is not None:
            raise RuntimeError(command.error)
        return dict(command.result)

    def rebuild_model(self, paths, run, timeout=1800.0, barrier=None):
        """Rebuild the model stage of this rank against retained inputs.

        Valid only in Idle, like every other command that replaces the
        resident model.
        """
        with self._condition:
            if self._state is not SessionState.Idle:
                raise RuntimeError(
                    f"Rebuilding the model is not allowed while session state "
                    f"is {self._state.name}")
            epoch = self._enter_epoch_locked("rebuild_model", barrier)
            command = RebuildModelCommand(
                session_generation=self.session_generation, epoch=epoch,
                expected_iteration=self._completed,
                expected_config_revision=self._config_revision,
                paths=paths, run=run)
        return self._queue_command(command, timeout)

    def preflight_checkpoint(self, path, timeout=600.0, barrier=None):
        """Phase 1 of an in-session load, on this rank.

        Valid only in Idle: a checkpoint replaces the resident model, so the
        session must have nothing outstanding against it.
        """
        with self._condition:
            if self._state is not SessionState.Idle:
                raise RuntimeError(
                    f"Loading a checkpoint is not allowed while session state "
                    f"is {self._state.name}")
            epoch = self._enter_epoch_locked("preflight_checkpoint", barrier)
            command = PreflightCheckpointCommand(
                session_generation=self.session_generation, epoch=epoch,
                expected_iteration=self._completed,
                expected_config_revision=self._config_revision, path=path)
        return self._queue_command(command, timeout)

    def apply_checkpoint(self, path, timeout=600.0, barrier=None):
        """Phase 2 of an in-session load, on this rank."""
        with self._condition:
            if self._state is not SessionState.Loading:
                raise RuntimeError(
                    f"No checkpoint load is in progress (session state is "
                    f"{self._state.name})")
            epoch = self._enter_epoch_locked("apply_checkpoint", barrier)
            command = ApplyCheckpointCommand(
                session_generation=self.session_generation, epoch=epoch,
                expected_iteration=self._completed, path=path)
        return self._queue_command(command, timeout)

    def discard_checkpoint(self, timeout=30.0, barrier=None):
        """Release a preflighted payload after a refusal on any rank."""
        with self._condition:
            epoch = self._enter_epoch_locked("discard_checkpoint", barrier)
            command = DiscardCheckpointCommand(
                session_generation=self.session_generation, epoch=epoch)
        return self._queue_command(command, timeout)

    def export_preview(self, timeout=PREVIEW_EXPORT_TIMEOUT_S,
                       diagnostics=False, barrier=None):
        """Export and publish one preview generation, on request.

        A coordinator sub-operation: only the publishing rank exports, so it
        carries no command epoch, exactly like a checkpoint save.
        """
        with self._condition:
            if self._state not in {SessionState.Idle, SessionState.Running}:
                raise RuntimeError(
                    f"Preview export is not allowed in {self._state.name}")
            epoch = (
                None if self._state is SessionState.Idle and barrier is None
                else self._enter_epoch_locked("export_preview", barrier))
            if self._state is SessionState.Running:
                self._step_epoch = epoch
            command = ExportPreviewCommand(
                session_generation=self.session_generation,
                epoch=epoch,
                expected_iteration=self._completed,
                diagnostics=bool(diagnostics))
        return self._queue_command(command, timeout)

    def load_checkpoint(self, path, timeout=600.0):
        """Load a checkpoint into the live session, strictly and atomically.

        A single-rank session is its own coordinator, so it runs the same two
        phases the distributed proxy runs across its ranks: inspect first and
        refuse without touching anything, then apply.
        """
        try:
            verdict = self.preflight_checkpoint(path, timeout)
        except BaseException:
            try:
                self.discard_checkpoint()
            except BaseException:
                traceback.print_exc(limit=4)
            raise
        result = self.apply_checkpoint(path, timeout)
        return {**verdict, **result}

    def close(self, timeout=15.0, barrier=None):
        with self._condition:
            self._enter_epoch_locked("close", barrier)
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


def _distributed_session_worker(context, gpu_id, rendezvous, paths, run,
                                preview, scroll, commands, events):
    """Own one CUDA rank and adapt queue commands to InteractiveFitSession.

    ``context`` is the (rank, world_size, local_rank) identity assigned by the
    parent, passed explicitly rather than published through the environment.
    ``rendezvous`` is the endpoint the parent owns for the whole session.
    """
    rank = context.rank
    # The one environment variable this worker still sets: it gives the rank a
    # one-device CUDA namespace, so checkpoint RNG snapshots and other
    # process-global CUDA helpers cannot open contexts on GPUs owned by
    # sibling ranks. Rank and world size travel as values, not as env.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    def publish_status(status):
        events.put(("status", rank, status))

    session = None
    closed = False
    try:
        session = InteractiveFitSession(
            paths, run, preview, scroll, publish_status,
            publishes_outputs=(rank == 0),
            rank=rank, world_size=context.world_size,
            local_rank=context.local_rank, rendezvous=rendezvous)
        while True:
            barrier, command_id, name, arguments = commands.get()
            try:
                if name == "run":
                    result = session.run(
                        arguments["count"],
                        influence_config=arguments.get("influence_config"),
                        run_config=arguments.get("run_config"),
                        path_changes=arguments.get("path_changes"),
                        autosave_on_pause=arguments.get(
                            "autosave_on_pause", True),
                        preview_schedule=arguments.get("preview_schedule"),
                        dt_loss_schedule=arguments.get("dt_loss_schedule"),
                        barrier=barrier,
                    )
                elif name == "stop":
                    result = session.stop(barrier=barrier)
                elif name == "disable_automatic_previews":
                    result = session.disable_automatic_previews(
                        arguments.get("error"))
                elif name == "reserve_input_boundary":
                    result = session.reserve_input_boundary(
                        arguments["target_iteration"],
                        arguments["reservation_epoch"],
                        include_idle=arguments.get("include_idle", False))
                elif name == "prepare_input_batch":
                    result = session.prepare_input_batch(**arguments)
                elif name == "finish_input_batch":
                    result = session.finish_input_batch(**arguments)
                elif name == "cancel_input_boundary":
                    result = session.cancel_input_boundary(
                        arguments["reservation_epoch"])
                elif name == "export_preview":
                    result = session.export_preview(
                        arguments.get("timeout", PREVIEW_EXPORT_TIMEOUT_S),
                        diagnostics=arguments.get("diagnostics", False),
                        barrier=barrier)
                elif name == "rebuild_model":
                    result = session.rebuild_model(
                        arguments["paths"], arguments["run"],
                        arguments.get("timeout", 1800.0), barrier=barrier)
                elif name == "preflight_checkpoint":
                    result = session.preflight_checkpoint(
                        arguments["path"], arguments.get("timeout", 600.0),
                        barrier=barrier)
                elif name == "apply_checkpoint":
                    result = session.apply_checkpoint(
                        arguments["path"], arguments.get("timeout", 600.0),
                        barrier=barrier)
                elif name == "discard_checkpoint":
                    result = session.discard_checkpoint(
                        arguments.get("timeout", 30.0), barrier=barrier)
                elif name == "save_checkpoint":
                    # Coordinator sub-operation: only the publishing rank is
                    # asked, and it carries no barrier.
                    result = session.save_checkpoint(
                        arguments["path"], arguments.get("timeout", 120.0))
                elif name == "close":
                    session.close(arguments.get("timeout", 15.0),
                                  barrier=barrier)
                    closed = True
                    result = None
                else:
                    raise ValueError(f"Unknown distributed session command {name}")
            except CommandBarrierViolation as exc:
                # Fail-stop. A rank that cannot agree with the coordinator
                # about which command it is in must not continue: report and
                # exit, and let the parent watchdog take the siblings down.
                events.put(("worker_error", rank, f"{type(exc).__name__}: {exc}",
                            traceback.format_exc(limit=12)))
                return
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
    """Parent-process proxy for one resident fitter process per selected GPU.

    The parent is the coordinator: it owns the monotonically increasing
    command epoch stamped on every all-rank command, it owns the rendezvous
    the ranks meet at, and it owns the watchdog that turns any worker error,
    command timeout, or unexpected exit into a bounded-time session failure
    instead of a wait on the process group's collective timeout.
    """

    def __init__(self, paths, run, preview, scroll, gpu_ids,
                 status_callback=None, event_callback=None):
        from ddp_helpers import DistributedContext

        self._init_coordinator_state(gpu_ids, status_callback, event_callback)
        context = multiprocessing.get_context("spawn")
        self._events = context.Queue()
        self._commands = [context.Queue() for _ in self._gpu_ids]
        # Held open for the whole session and removed in close(): the ranks'
        # meeting point cannot be taken by anything else while they use it.
        self._rendezvous = FileStoreRendezvous(
            Path(paths.output_directory) / ".spiral-rendezvous")
        self._processes = [
            context.Process(
                target=_distributed_session_worker,
                args=(DistributedContext(rank=rank,
                                         world_size=len(self._gpu_ids),
                                         # One visible device per worker.
                                         local_rank=0),
                      gpu_id, self._rendezvous.endpoint,
                      paths, run, preview, scroll,
                      self._commands[rank], self._events),
                name=f"spiral-gpu-{gpu_id}",
            )
            for rank, gpu_id in enumerate(self._gpu_ids)
        ]
        self._start_coordinator_threads()
        started = []
        try:
            for process in self._processes:
                process.start()
                started.append(process)
        except BaseException:
            for process in started:
                process.terminate()
            for process in started:
                process.join(WORKER_TERMINATE_GRACE_S)
            self._stop_coordinator_threads()
            self._close_rendezvous()
            raise

    def _init_coordinator_state(self, gpu_ids, status_callback,
                                event_callback):
        """Set up everything that does not depend on live worker processes."""
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
        self._rank_statuses = {}
        self._failed_error = None
        self._closed = False
        # True from the moment an orderly close begins, so the watchdog does
        # not report an expected worker exit as a failure.
        self._closing = False
        self._aborted = False
        self._abort_lock = threading.Lock()
        # The coordinator's command epoch. Every all-rank command carries the
        # next value; ranks refuse anything else.
        self._command_epoch = 0
        self._live_reservation_epoch = 0
        # The last state every rank agreed on. Nothing else is publishable as
        # a collective state.
        self._collective_state = SessionState.Loading
        self._rendezvous = None
        self._events = None
        self._commands = []
        self._processes = []
        self._stop_watchdog = threading.Event()
        self._listener = None
        self._watchdog_thread = None

    def _start_coordinator_threads(self):
        self._listener = threading.Thread(
            target=self._listen, name="spiral-gpu-coordinator", daemon=True)
        self._listener.start()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog, name="spiral-gpu-watchdog", daemon=True)
        self._watchdog_thread.start()

    def _close_rendezvous(self):
        if self._rendezvous is not None:
            self._rendezvous.close()
            self._rendezvous = None

    def _stop_coordinator_threads(self):
        self._stop_watchdog.set()
        if self._events is not None:
            self._events.put(None)
        for thread in (self._listener, self._watchdog_thread):
            if thread is not None:
                thread.join(WORKER_TERMINATE_GRACE_S)

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

    # Coordinator: command epochs and barriers.
    def _issue_barrier(self, kind):
        """Stamp the next all-rank command with the epoch every rank checks.

        Call with the lock held. The configuration revision travels with the
        barrier only when every rank has reported the same one; before that
        (during startup) there is nothing to assert and the field is left
        unset rather than guessed.
        """
        self._command_epoch += 1
        revisions = {status.get("config_revision")
                     for status in self._rank_statuses.values()}
        config_revision = None
        if (len(self._rank_statuses) == len(self._gpu_ids)
                and len(revisions) == 1):
            config_revision = revisions.pop()
        return CommandBarrier(
            epoch=self._command_epoch, kind=kind,
            config_revision=config_revision,
            # A run, and either phase of a checkpoint load, is admitted only
            # by a quiescent rank. stop and close are asynchronous with
            # respect to the step loop by construction, so they assert
            # nothing about the pending count.
            pending=(0 if kind in _QUIESCENT_COMMANDS else None))

    # Coordinator: fail-stop.
    def _fail_session(self, cause, *, rank=None, detail=None):
        """Mark the session failed, take the workers down, return the message.

        The only path to Error for the whole job. It is idempotent: the first
        cause wins, so the diagnosable message is the original failure rather
        than whatever timed out behind it.
        """
        with self._condition:
            if self._failed_error is not None:
                return self._failed_error
            message = (f"GPU worker rank {rank}: {cause}"
                       if rank is not None else str(cause))
            warnings = list(self._status.get("warnings", []))
            if detail:
                warnings.append(detail)
            self._status.update({
                "state": SessionState.Error, "phase": "Error",
                "error": message, "warnings": warnings,
            })
            self._failed_error = message
            snapshot = copy.deepcopy(self._status)
            self._condition.notify_all()
        self._abort_workers()
        if self._status_callback is not None:
            self._status_callback(snapshot)
        return message

    def _abort_workers(self):
        """Terminate every worker, and with it the process group.

        Killing the ranks is what aborts the group: no rank is left sitting
        in a collective, so nothing waits out the NCCL timeout. Bounded by
        WORKER_TERMINATE_GRACE_S + WORKER_KILL_GRACE_S.
        """
        with self._abort_lock:
            if self._aborted:
                return
            self._aborted = True
        processes = list(self._processes)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(WORKER_TERMINATE_GRACE_S)
        survivors = [process for process in processes if process.is_alive()]
        for process in survivors:
            kill = getattr(process, "kill", process.terminate)
            kill()
        for process in survivors:
            process.join(WORKER_KILL_GRACE_S)

    def _watchdog(self):
        """Report any unexpected worker exit as a session failure.

        Generation and epoch checks happen at command boundaries, and a rank
        that dies inside a step never reaches one. This loop is what bounds
        the time to a diagnosable error in that case.
        """
        while not self._stop_watchdog.wait(WATCHDOG_POLL_INTERVAL_S):
            with self._condition:
                if self._closing or self._failed_error is not None:
                    continue
            for rank, process in enumerate(self._processes):
                if process.is_alive():
                    continue
                exit_code = process.exitcode
                if exit_code is None:
                    # Not started yet.
                    continue
                self._fail_session(
                    f"exited unexpectedly with exit code {exit_code} before "
                    f"the session was closed", rank=rank)
                break

    def _collective_status_locked(self, rank_zero_status):
        """Build the status the parent may publish from the rank snapshots."""
        view = collective_view(self._rank_statuses, len(self._gpu_ids),
                               self._collective_state)
        status = copy.deepcopy(rank_zero_status)
        if view.visible:
            self._collective_state = view.state
            return status
        status["state"] = view.state
        if view.coordinator_operation is not None:
            # Not a collective state: rank 0 alone is doing publication work
            # while its siblings idle. Say so.
            status["coordinator_operation"] = view.coordinator_operation
            status["phase"] = (f"Coordinator {view.coordinator_operation} "
                               f"(rank 0): {status.get('phase') or ''}".strip())
            progress = status.get("progress")
            if progress:
                detail = progress.get("detail")
                progress["detail"] = (
                    f"coordinator rank 1/{len(self._gpu_ids)}"
                    + (f" — {detail}" if detail else ""))
            return status
        status["phase"] = "Waiting for all GPU workers"
        if view.laggard is not None:
            worker_status = self._rank_statuses.get(view.laggard, {})
            worker_progress = copy.deepcopy(worker_status.get("progress"))
            if worker_progress:
                detail = worker_progress.get("detail")
                worker_progress["detail"] = (
                    f"GPU worker {view.laggard + 1}/{len(self._gpu_ids)}"
                    + (f" — {detail}" if detail else ""))
                status["progress"] = worker_progress
                status["phase"] = str(worker_progress.get("stage_name")
                                      or status["phase"])
        return status

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
                if status.get("state") == SessionState.Error:
                    self._fail_session(
                        status.get("error") or "reported an error",
                        rank=rank,
                        detail="\n".join(status.get("warnings", [])) or None)
                    continue
                with self._condition:
                    self._rank_statuses[rank] = status
                    if self._failed_error is not None:
                        continue
                    if 0 not in self._rank_statuses:
                        # Nothing user-facing exists until the coordinator
                        # rank has reported: it owns metrics and artifacts.
                        continue
                    self._status = self._collective_status_locked(
                        self._rank_statuses[0])
                    snapshot = copy.deepcopy(self._status)
                callback = self._status_callback
            elif kind == "ack":
                _, command_id, rank, ok, result = event
                with self._condition:
                    self._acks.setdefault(command_id, {})[rank] = (ok, result)
                    self._condition.notify_all()
            elif kind == "worker_error":
                _, rank, error, trace = event
                self._publish_rank_event(rank, {
                    "state": SessionState.Error, "error": error,
                    "warnings": [trace]})
                self._fail_session(
                    error, rank=rank,
                    detail=f"GPU worker rank {rank} failed:\n{trace}")
                continue
            if callback is not None:
                callback(snapshot)

    def _call(self, name, arguments=None, ranks=None,
              timeout=COMMAND_ACK_TIMEOUT_S,
              collective=True, return_all=False):
        """Send one command to the participating ranks and await every ack.

        ``collective`` commands carry a barrier: the same epoch, kind,
        expected configuration revision, and pending count reach every rank,
        and each validates it against its own state before acting. A command
        addressed to rank 0 alone is a coordinator sub-operation and carries
        no barrier, so it does not move the epoch the other ranks are in.
        """
        if self._closed and name != "close":
            raise RuntimeError("Spiral fit session is closed")
        if self._failed_error is not None and name != "close":
            raise RuntimeError(self._failed_error)
        ranks = tuple(range(len(self._processes))) if ranks is None else tuple(ranks)
        command_id = uuid.uuid4().hex
        with self._condition:
            barrier = self._issue_barrier(name) if collective else None
        for rank in ranks:
            self._commands[rank].put(
                (barrier, command_id, name, dict(arguments or {})))
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self._acks.get(command_id, {})) < len(ranks):
                if self._failed_error is not None:
                    raise RuntimeError(self._failed_error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    silent = [rank for rank in ranks
                              if rank not in self._acks.get(command_id, {})]
                    break
                self._condition.wait(remaining)
            else:
                responses = self._acks.pop(command_id)
                failures = [f"rank {rank}: {responses[rank][1]}" for rank in ranks
                            if not responses[rank][0]]
                if failures:
                    raise RuntimeError("; ".join(failures))
                if return_all:
                    return {rank: responses[rank][1] for rank in ranks}
                return responses[ranks[0]][1]
        # A command that is not acknowledged in bounded time means a rank is
        # wedged; the session fails now rather than when a collective
        # eventually times out.
        raise TimeoutError(self._fail_session(
            f"Timed out after {timeout:.0f}s waiting for GPU worker ranks "
            f"{silent} to {name}"))

    def run(self, count, influence_config=None, run_config=None, path_changes=None,
            autosave_on_pause=True, preview_schedule=None,
            dt_loss_schedule=None):
        state = self.status()["state"]
        if state != SessionState.Idle:
            raise RuntimeError(f"Run is not allowed while session state is {state}")
        arguments = {
            "count": count,
            "influence_config": dict(influence_config or {}),
            "run_config": dict(run_config or {}),
            "path_changes": dict(path_changes or {}),
            "autosave_on_pause": bool(autosave_on_pause),
            "preview_schedule": copy.deepcopy(preview_schedule),
            "dt_loss_schedule": copy.deepcopy(dt_loss_schedule or {
                "enabled": False, "last_fraction": 0.25}),
        }
        return self._call("run", arguments, timeout=COMMAND_ACK_TIMEOUT_S)

    def apply_input_changes(self, batch_id, records, influence_config=None, *,
                            timeout=INPUT_BATCH_TIMEOUT_S):
        return _apply_input_changes_at_boundary(
            self, batch_id, records, influence_config or {}, timeout=timeout,
            distributed=True)

    def stop(self):
        state = self.status()["state"]
        if state != SessionState.Running:
            raise RuntimeError(f"Session is not running (state is {state})")
        return self._call("stop")

    def disable_automatic_previews(self, error=None):
        return self._call(
            "disable_automatic_previews", {"error": error},
            collective=False)

    def export_preview(self, timeout=PREVIEW_EXPORT_TIMEOUT_S,
                       diagnostics=False):
        state = self.status()["state"]
        if state not in {SessionState.Idle, SessionState.Running}:
            raise RuntimeError(f"Preview export is not allowed in {state}")
        # Every rank enters the same short source-snapshot boundary. Rank 0
        # writes the raw TIFXYZ; siblings wait at the command boundary and all
        # resume optimization together once capture finishes.
        return self._call("export_preview",
                          {"timeout": timeout,
                           "diagnostics": bool(diagnostics)},
                          timeout=timeout + COMMAND_ACK_GRACE_S,
                          collective=True)

    def rebuild_model(self, paths, run, timeout=1800.0):
        """Rebuild the model stage on every rank against retained inputs."""
        state = self.status()["state"]
        if state != SessionState.Idle:
            raise RuntimeError(
                f"Rebuilding the model is not allowed in {state}")
        return self._call("rebuild_model",
                          {"paths": paths, "run": run, "timeout": timeout},
                          timeout=timeout + COMMAND_ACK_GRACE_S)

    def load_checkpoint(self, path, timeout=600.0):
        """Coordinate a strict two-phase checkpoint load across every rank.

        Phase 1 asks every rank to inspect the same file on its CPU. A
        refusal by any rank fails the whole load with every rank's reasons,
        and the payloads the accepting ranks retained are discarded, so the
        live session is untouched. Only when every rank accepted does phase 2
        apply it everywhere.
        """
        state = self.status()["state"]
        if state != SessionState.Idle:
            raise RuntimeError(
                f"Loading a checkpoint is not allowed in {state}")
        arguments = {"path": path, "timeout": timeout}
        try:
            verdict = self._call("preflight_checkpoint", arguments,
                                 timeout=timeout + COMMAND_ACK_GRACE_S)
        except BaseException:
            try:
                self._call("discard_checkpoint", {},
                           timeout=COMMAND_ACK_TIMEOUT_S)
            except BaseException:
                traceback.print_exc(limit=4)
            raise
        result = self._call("apply_checkpoint", arguments,
                            timeout=timeout + COMMAND_ACK_GRACE_S)
        return {**(verdict or {}), **(result or {})}

    def save_checkpoint(self, path, timeout=120.0):
        state = self.status()["state"]
        if state != SessionState.Idle:
            raise RuntimeError(f"Checkpoint save is not allowed in {state}")
        # Explicitly a coordinator sub-operation: rank 0 publishes outputs,
        # so rank 0 alone writes the checkpoint, outside the epoch sequence.
        return self._call("save_checkpoint", {"path": path, "timeout": timeout},
                          ranks=(0,), timeout=timeout + COMMAND_ACK_GRACE_S,
                          collective=False)

    def _require_stopped(self):
        # A timed-out close sets _closed before cleanup. Retrying it must not
        # authorize deleting inputs while a process or listener is still alive.
        if any(process.is_alive() for process in self._processes) or any(
                thread is not None and thread.is_alive()
                for thread in (self._listener, self._watchdog_thread)):
            raise TimeoutError("Spiral GPU workers or listeners are still using session files")

    def close(self, timeout=15.0):
        if self._closed:
            self._require_stopped()
            return
        with self._condition:
            self._closing = True
        if self._failed_error is not None:
            self._closed = True
            self._abort_workers()
            self._stop_coordinator_threads()
            self._close_rendezvous()
            self._require_stopped()
            return
        try:
            self._call("close", {"timeout": timeout},
                       timeout=timeout + COMMAND_ACK_GRACE_S)
        finally:
            self._closed = True
            deadline = time.monotonic() + timeout
            for process in self._processes:
                process.join(max(0.0, deadline - time.monotonic()))
            alive = [process for process in self._processes if process.is_alive()]
            if alive:
                self._abort_workers()
                self._stop_coordinator_threads()
                self._close_rendezvous()
                raise TimeoutError("Spiral GPU workers did not stop at a safe boundary")
            self._stop_coordinator_threads()
            self._close_rendezvous()
            self._require_stopped()


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
