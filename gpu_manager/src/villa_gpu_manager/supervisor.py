"""CPU-only lifecycle coordinator for full-release cooperative clients.

The cohort adapter owns application work and processes. The supervisor never
imports a compute framework. ``tick`` is called by the application's event loop;
neither initialization nor draining may block inside the cohort adapter.
"""

from copy import deepcopy
import time

from .policy import empty
from .protocol import ProtocolError


class Supervisor:
    """Authorize bounded work units only while a confirmed grant is current.

    A cohort implements ``poll()``, ``start_unit()``, ``drain()``, ``abort()``,
    ``authorize()``, and properties ``authorized``, ``ready``, ``idle``, ``closed``, ``done``, ``state``,
    ``processes``. ``closed`` means *joined*, not merely a stop message sent.
    ``poll`` must surface worker errors. ``drain`` is idempotent and stops at
    the current unit boundary; ``abort`` joins owned children on fatal errors.
    The factory must return promptly, with asynchronous initialization.
    """

    def __init__(self, client, *, label, request, cohort_factory, state=None,
                 poll_interval=1.0, clock=time.monotonic, dispose_state=None):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.client = client
        self.label, self.request = label, deepcopy(request)
        self.factory, self.state = cohort_factory, state
        self.dispose_state = dispose_state
        self.poll_interval, self.clock = float(poll_interval), clock
        self.next_poll = 0.0
        self.cohort = None
        self.grant = empty()
        self.generation = None
        self.epoch = None
        self.connected = False
        self.stopping = False
        self.finished = False
        self.outcome = "completed"
        self.error = None
        self.status = "connecting"
        self._draining = False
        self._released_generation = None
        self._released_epoch = None

    def stop(self, outcome="completed"):
        """Request clean shutdown, including draining CPU compute resources."""
        self.stopping, self.outcome = True, outcome
        self.next_poll = 0
        self._drain()

    def fail(self, error):
        """Abort owned workers after an application/CPU-service failure.

        Keep ticking afterwards to report the joined state to the manager.
        """
        self._fail(error)

    def _drain(self):
        if self.cohort is not None and not self._draining:
            self._draining = True
            self.cohort.drain()

    def _lost_connection(self):
        self.client.close()
        self.connected = False
        self.status = "manager_unavailable_draining"
        self._drain()

    def _fail(self, exc):
        self.error = exc
        self.stopping, self.outcome = True, f"failed: {type(exc).__name__}: {exc}"[:4096]
        # Never acknowledge release if abort/join itself fails.
        if self.cohort is not None:
            self.cohort.abort()
            if not self.cohort.closed:
                raise RuntimeError("cohort abort returned without joining workers") from exc
        self.next_poll = 0

    def _collect_closed(self):
        if self.cohort is None or not self.cohort.closed:
            return
        if self.error is None:
            if self.state is not None and self.state is not self.cohort.state and self.dispose_state is not None:
                self.dispose_state(self.state)
            self.state = self.cohort.state
        self._released_generation = self.generation
        self._released_epoch = self.epoch
        self.cohort = None
        self.grant = empty()
        self._draining = False
        self.next_poll = 0

    def tick(self):
        if self.finished:
            return
        if self.cohort is not None:
            try:
                self.cohort.poll()
                if self.cohort.done:
                    self.stop()
            except Exception as exc:
                self._fail(exc)
        self._collect_closed()
        now = self.clock()
        # Poll at every work boundary, not just once a second. The interval is
        # for long-running units/startup/pause; it must not cap training to 1Hz.
        if self.cohort is not None and self.cohort.ready and self.cohort.idle and not self._draining:
            self.next_poll = 0
        if now < self.next_poll:
            return
        self.next_poll = now + self.poll_interval
        try:
            if not self.connected:
                # Reconnect only after local work has stopped. No old permission
                # can authorize another unit after a connection/epoch loss.
                if self.cohort is not None:
                    return
                self.client.connect()
                task = self.client.register(self.label, self.request)
                self.connected = True
            else:
                task = self.client.poll()
            if self.cohort is not None and self.client.epoch != self.epoch:
                self._drain()
                self.status = "epoch_changed_draining"
                return
            if self.stopping and not task["paused"]["self"]:
                task = self.client.pause(True)
            phase = task["phase"]
            if phase == "release" or self.stopping or self._draining:
                self._drain()
                self.status = "draining" if self.cohort is not None else "released"
                if self.cohort is not None:
                    return
                if task["granted"] != empty():
                    task = self.client.report(
                        "failed" if self.error is not None else "released", empty(),
                        detail=self.outcome if self.error is not None else "workers joined",
                    )
                if self.stopping:
                    self.client.finish(self.outcome)
                    self.client.close()
                    self.finished, self.status = True, "finished"
                return
            # A release response may be lost. A *new* generation can be
            # initialized, but a stale applied/initializing grant must first be
            # released even if the manager never learned that we drained.
            if self.cohort is None and task["granted"] != empty() and (
                phase == "applied" or (
                    self._released_generation == task["generation"]
                    and self._released_epoch == self.client.epoch
                )
            ):
                self.client.report("released", empty(), detail="workers joined after disconnect")
                return
            if phase == "waiting":
                if self.cohort is not None:
                    raise RuntimeError("manager withdrew ownership without a release directive")
                self.status = "paused" if any(task["paused"].values()) else "waiting"
                return
            if self.cohort is None:
                if phase != "initialize" or task["granted"] == empty():
                    raise RuntimeError(f"cannot initialize from manager phase {phase}")
                self.grant = deepcopy(task["granted"])
                self.generation, self.epoch = task["generation"], self.client.epoch
                self.cohort = self.factory(deepcopy(self.grant), self.state)
                self.status = "initializing"
                return
            if task["granted"] != self.grant or task["generation"] != self.generation:
                raise RuntimeError("manager changed a live grant without release")
            if phase == "initialize":
                task = self.client.report("applied", self.grant, processes=self.cohort.processes)
                # Applying a superseded grant can immediately return release.
                if task["phase"] == "release":
                    self._drain()
                    return
            if task["phase"] != "applied":
                raise RuntimeError("work requires an applied grant")
            if not self.cohort.authorized:
                # Manager now knows the child identities *before* they can
                # initialize CUDA. Losing this reply leaves children gated.
                self.cohort.authorize()
            if not self.cohort.ready:
                self.status = "initializing"
                return
            self.status = "running"
            if self.cohort.idle:
                self.cohort.start_unit()
        except OSError:
            self._lost_connection()
        except ProtocolError as exc:
            if exc.code in ("stale_epoch", "unavailable"):
                self._lost_connection()
            elif exc.code == "terminal" and self.stopping and self.cohort is None:
                # finish may have committed before its response was lost. Never
                # try to resurrect the task or turn this into a retry loop.
                rows = self.client.status()["tasks"]
                task_id = self.client.task_id
                if any(row["task_id"] == task_id and row["terminal"] for row in rows):
                    self.client.close()
                    self.finished, self.status = True, "finished"
                else:
                    raise RuntimeError("terminal task cannot be reconciled") from exc
            else:
                self._fail(exc)
                # Authentication/schema errors cannot be repaired by retrying
                # the same request forever. Our own workers are already joined.
                raise
        except Exception as exc:
            self._fail(exc)
