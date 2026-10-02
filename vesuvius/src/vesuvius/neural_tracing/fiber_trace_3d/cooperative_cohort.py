"""Join the shared CPU broker to a replaceable GPU worker cohort.

Internal integration adapter; the manager server remains framework-independent.
CPU batches backed by shared slots cross the internal multiprocessing queues. Volume slot
ownership stays with the broker until a complete optimizer update is reported.
"""

import multiprocessing as mp
import json
from contextlib import ExitStack
import os
from pathlib import Path
import queue
import tempfile
import time

from .cooperative_resources import EffectiveBatchSampleIndex
from .cooperative_worker import FiberWorkerFactory
from vesuvius.neural_tracing.ordered_sample_pipeline import native_thread_environment


class FiberCohort:
    def __init__(self, *, config_path, grant, state, broker, calibration,
                 queue_depth=2, writer_factory=None, resume_checkpoint=None,
                 drain_timeout=300.0, training_overrides=None, calibration_observations=(),
                 startup_microbatch_size=None, startup_reserve_bytes=4 * 1024**3,
                 on_startup_observation=None, on_process_start=None,
                 maintenance_mailbox=None):
        from villa_gpu_manager.cohort import ProcessCohort
        if type(queue_depth) is not int or queue_depth < 1:
            raise ValueError("GPU input queue depth must be positive")
        self.broker = broker
        self.grant = grant
        self._draining = False
        self._maintenance_draining = False
        self._maintenance_ready = False
        self._queues_closed = False
        self._sent, self._drained = [], []
        self._pending = {}
        self.maintenance_mailbox = maintenance_mailbox
        if drain_timeout <= 0:
            raise ValueError("CPU drain timeout must be positive")
        self._drain_timeout = float(drain_timeout)
        self._drain_started = None
        self._publication_errors = queue.SimpleQueue()
        self._memory_monitor = None
        self._on_startup_observation = on_startup_observation
        self._driver_peaks = {gpu: 0 for gpu in grant["gpus"]}
        self.gpu = None
        self._authorized_at = None
        self._gpu_pids = set()
        self._temp = tempfile.TemporaryDirectory(prefix="fiber-cohort-")
        context = mp.get_context("spawn")
        devices = sorted(grant["gpus"])
        raw_config = json.loads(Path(config_path).read_text())
        group_size = 2 if raw_config.get("embedding", {}).get("paired_patches", False) else 1
        if startup_microbatch_size is not None:
            if type(startup_microbatch_size) is not int or startup_microbatch_size < 1 or calibration is not None:
                raise ValueError("startup requires a positive physical batch and no invented calibration")
            sizes = (startup_microbatch_size,) * len(devices)
            reserve = startup_reserve_bytes
        else:
            sizes = tuple(calibration.microbatch_for(grant["gpus"][gpu]) for gpu in devices)
            reserve = calibration.reserve_bytes
        sizes = tuple(s // group_size * group_size for s in sizes)
        if not sizes or min(sizes) < 1:
            self._temp.cleanup()
            raise ValueError("grant cannot fit a calibrated microbatch")
        self._queues, self._received, self._maps = [], [], []
        self._test_connections, self._test_children, self._test_pending = [], [], {}
        self._test_boundaries = {}
        self._ordinals = [0] * len(sizes)
        self._samples = [0] * len(sizes)
        start_step = broker.frontier // broker.effective_batch_size
        try:
            for rank in range(len(sizes)):
                parent_test, child_test = context.Pipe(duplex=True)
                self._test_connections.append(parent_test)
                self._test_children.append(child_test)
                self._queues.append(context.Queue(maxsize=queue_depth))
                # Queue.put returns before the feeder serializes the batch.
                # Surface its actual exception instead of treating admission
                # as successful delivery and waiting forever during release.
                self._queues[-1]._on_queue_feeder_error = self._publication_failed
                self._received.append(context.Value("q", 0))
                self._sent.append(0)
                self._drained.append(0)
                self._maps.append(EffectiveBatchSampleIndex(broker.effective_batch_size, sizes, rank, start_step, group_size))
            factory = FiberWorkerFactory(
                str(config_path), sizes, tuple(self._queues), tuple(self._received), os.getpid(),
                (Path(self._temp.name) / "rendezvous").as_uri(), start_step, broker.effective_batch_size,
                writer_factory=writer_factory, resume_checkpoint=resume_checkpoint,
                training_overrides=training_overrides,
                reserve_bytes=reserve,
                calibrated_gpu_names=tuple(sorted({row["gpu_name"] for row in calibration_observations})),
                calibrated_torch_versions=tuple(sorted({row["torch_version"] for row in calibration_observations})),
                calibrated_cuda_versions=tuple(sorted({row["cuda_version"] for row in calibration_observations})),
                test_connections=tuple(self._test_children),
                measure_startup_memory=startup_microbatch_size is not None,
            )
            self.gpu = ProcessCohort(factory, grant, state, environment=native_thread_environment(),
                                     on_unit=self._completed_unit, on_process_start=on_process_start)
            self._gpu_pids = {row["pid"] for row in self.gpu.processes}
            if calibration_observations or startup_microbatch_size is not None:
                from villa_gpu_manager.processes import NvidiaMonitor
                self._memory_monitor = NvidiaMonitor(grant, 0, interval=1)
        except BaseException:
            # The supervisor cannot own a constructor that never returned.
            # Reap even gated children before discarding their IPC resources.
            with ExitStack() as cleanup:
                cleanup.callback(self._temp.cleanup)
                cleanup.callback(self._close_queues, abort=True)
                if self._memory_monitor is not None:
                    cleanup.callback(self._memory_monitor.close)
                if self.gpu is not None:
                    cleanup.callback(self.gpu.abort)
            raise

    def _completed_unit(self, rows):
        progress = [{key: value for key, value in row.items() if key != "memory"} for row in rows]
        if not rows or any(row != progress[0] for row in progress[1:]):
            raise RuntimeError("GPU ranks disagree about completed training progress")
        step = rows[0]["step"]
        frontier = self.broker.frontier // self.broker.effective_batch_size
        if step != frontier:
            self.broker.commit(step)
        elif rows[0].get("phase") not in ("test", "update", "reviewed_trace"):
            raise RuntimeError("invalid repeated training boundary")
        if rows[0].get("phase") == "reviewed_trace":
            samples_seen = rows[0].get("samples_seen")
            if type(samples_seen) is not int or samples_seen < 0:
                raise RuntimeError("reviewed trace boundary lacks cumulative sample progress")
            if self.maintenance_mailbox is None:
                raise RuntimeError("reviewed trace boundary has no maintenance mailbox")
            self.maintenance_mailbox.publish({
                "kind": "reviewed_trace", "step": int(step),
                "samples_seen": samples_seen,
            })
        if self._on_startup_observation is not None:
            observations = []
            for row in rows:
                memory = dict(row["memory"])
                memory["driver_peak_bytes"] = self._driver_peaks[memory["gpu_uuid"]]
                observations.append(memory)
            self._on_startup_observation(observations, progress[0])
        return rows[0]["done"]

    @property
    def authorized(self):
        return self.gpu.authorized

    @property
    def ready(self):
        return self.gpu.ready

    @property
    def idle(self):
        # Hold the cohort at the completed trace boundary until the CPU
        # supervisor has requested a pause/drain.  It must not admit the next
        # optimizer unit in the same manager tick.
        return self.gpu.idle and not (
            self.maintenance_mailbox is not None and self.maintenance_mailbox.pending
        )

    @property
    def done(self):
        return self.gpu.done

    @property
    def state(self):
        return self.gpu.state

    @property
    def processes(self):
        from villa_gpu_manager.processes import process_identity
        return self.gpu.processes + [process_identity(p.pid) for p in self.broker.pipeline.processes if p.is_alive()]

    @property
    def closed(self):
        maintenance_blocks_close = (
            self.maintenance_mailbox is not None
            and self.maintenance_mailbox.pending
            and (not self._draining or self.maintenance_mailbox.attempt_active)
        )
        return not maintenance_blocks_close and self.gpu.closed and self._queues_closed and (
            self.broker.pipeline.closed or self.broker.workers_within_limit())

    def authorize(self):
        # GPU-host work is mostly launch/IO; reserve one core per GPU where
        # possible, leaving at least one actual loading/augmentation worker.
        budget = max(1, self.grant["cpu_cores"] - len(self._queues))
        self.broker.set_worker_limit(min(budget, self.broker.pipeline.shared_workers))
        self._authorized_at = time.monotonic()
        self.gpu.authorize()

    def start_unit(self):
        self.gpu.start_unit()

    def drain(self):
        self._draining = True
        self.gpu.drain()
        # Do NOT park loaders yet: a running unit may still need its CPU inputs.

    def begin_maintenance(self):
        """Begin a pending handoff when possible; otherwise leave it queued."""
        if self.maintenance_mailbox is None or not self.maintenance_mailbox.pending:
            raise RuntimeError("cohort has no pending maintenance request")
        if self._draining or self._maintenance_draining:
            return False
        if not self.authorized or not self.ready:
            return False
        self._maintenance_draining = True
        self.gpu.drain()
        return True

    @property
    def maintenance_ready(self):
        return self._maintenance_ready and not self._draining

    def maintenance_state(self):
        if not self._maintenance_ready:
            raise RuntimeError("maintenance handoff is not ready")
        return self.gpu.state

    def poll(self):
        self._check_publication()
        self.broker.poll()
        self.gpu.poll()
        if self._maintenance_draining:
            if not self.gpu.closed:
                return
            if self._memory_monitor is not None:
                self._memory_monitor.close()
                self._memory_monitor = None
            self.broker.set_worker_limit(0)
            self._drain_queues()
            if self._queues_closed:
                self._temp.cleanup()
                self._maintenance_ready = True
            return
        if self.gpu.closed:
            if self._memory_monitor is not None:
                self._memory_monitor.close()
            if self._drain_started is None:
                self._drain_started = time.monotonic()
            self.broker.set_worker_limit(0)
            self._drain_queues()
            if not self.closed and time.monotonic() - self._drain_started > self._drain_timeout:
                raise RuntimeError("CPU input queues/workers failed to drain after GPU cohort joined")
            if self._queues_closed:
                self._temp.cleanup()
            return
        if not self.authorized:
            return
        self._check_driver_memory()
        self._poll_test_requests()
        for rank, mapping in enumerate(self._maps):
            if not mapping.batch_sizes:
                continue
            take = mapping.batch_sizes[self._ordinals[rank] % len(mapping.batch_sizes)]
            indices = tuple(mapping(self._samples[rank] + i) for i in range(take))
            if indices[-1] >= self.broker.end_index:
                continue
            if self._draining and indices[0] >= self.broker.frontier + self.broker.effective_batch_size:
                continue
            batch = self._pending.get(rank)
            if batch is None:
                batch = self.broker.batch(indices)
            if batch is None:
                continue
            try:
                self._queues[rank].put_nowait(batch)
            except queue.Full:
                self._pending[rank] = batch
                continue
            self._pending.pop(rank, None)
            self._ordinals[rank] += 1
            self._samples[rank] += take
            self._sent[rank] += 1

    def _poll_test_requests(self):
        for rank, connection in enumerate(self._test_connections):
            while connection.poll():
                kind, step, indices = connection.recv()
                if kind == "release-test":
                    self.broker.release_test(step, indices)
                elif kind in ("begin-test", "end-test", "select-test") and rank not in self._test_boundaries:
                    self._test_boundaries[rank] = (kind, step, indices)
                elif kind == "test" and rank not in self._test_pending:
                    self._test_pending[rank] = (kind, step, indices)
                else:
                    raise RuntimeError("invalid managed evaluation sample request")
            if rank in self._test_pending:
                kind, step, indices = self._test_pending[rank]
                batch = self.broker.test_batch(step, indices)
                if batch is not None:
                    connection.send(batch)
                    del self._test_pending[rank]
        if len(self._test_boundaries) == len(self._test_connections):
            boundaries = list(self._test_boundaries.values())
            if any(row != boundaries[0] for row in boundaries):
                raise RuntimeError("GPU ranks disagree on sample phase boundary")
            kind, step, details = boundaries[0]
            if kind == "begin-test":
                self.broker.begin_test(step)
            elif kind == "select-test":
                self.broker.select_test_source(step, *details)
            else:
                self.broker.end_test(step)
            for connection in self._test_connections:
                connection.send((kind, step))
            self._test_boundaries.clear()

    def _publication_failed(self, error, _item):
        self._publication_errors.put(f"{type(error).__name__}: {error}")

    def _check_driver_memory(self):
        monitor = self._memory_monitor
        if monitor is None:
            return
        monitor.poll(self._gpu_pids)
        if time.monotonic() - (monitor.last_snapshot_at or self._authorized_at) > 20:
            raise RuntimeError("cannot verify task VRAM usage from the driver")
        if monitor.last_snapshot is None:
            return
        for gpu, budget in self.grant["gpus"].items():
            if gpu not in monitor.last_snapshot:
                raise RuntimeError(f"allocated GPU {gpu} disappeared from driver inventory")
            used = sum(size for pid, size in monitor.last_snapshot[gpu]["processes"].items()
                       if pid in self._gpu_pids)
            self._driver_peaks[gpu] = max(self._driver_peaks[gpu], used)
            if used > budget:
                raise RuntimeError(f"task exceeded VRAM grant on {gpu}: {used} > {budget} bytes")

    def _check_publication(self):
        try:
            error = self._publication_errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError(f"CPU batch queue publication failed: {error}")

    def _drain_queues(self):
        if self._queues_closed:
            return
        for rank, batches in enumerate(self._queues):
            remaining = self._sent[rank] - self._received[rank].value - self._drained[rank]
            for _ in range(remaining):
                try:
                    batches.get_nowait()
                except queue.Empty:
                    break
                self._drained[rank] += 1
        if all(self._sent[i] == self._received[i].value + self._drained[i] for i in range(len(self._queues))):
            self._close_queues()

    def _close_queues(self, *, abort=False):
        if self._queues_closed:
            return
        for connection in self._test_connections + self._test_children:
            connection.close()
        for batches in self._queues:
            if abort:
                batches.cancel_join_thread()
            batches.close()
            if not abort:
                batches.join_thread()
        self._pending.clear()
        self._queues_closed = True

    def abort(self):
        self.gpu.abort()
        if self._memory_monitor is not None:
            self._memory_monitor.close()
        self.broker.close()
        self._close_queues(abort=True)
        self._temp.cleanup()
