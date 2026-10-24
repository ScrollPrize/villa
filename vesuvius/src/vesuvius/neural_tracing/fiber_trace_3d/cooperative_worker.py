"""Fiber adapter for the reusable process-cohort lifecycle.

Imported in an authorized GPU worker, not in the resource-manager server.
Uses the ordinary trainer's unit iterator; no second implementation of losses,
optimization, tests or snapshots lives here.
"""

from dataclasses import dataclass
import os
import queue
import threading


class QueueBatchSource:
    """Cancellable input for the existing asynchronous CUDA batch preparer.

    CPU volume slots remain owned by the persistent broker through global
    update completion. Closing this view never closes the shared loading pool.
    """

    def __init__(self, batches, received, owner_pid, index_map, test_connection=None):
        self.batches, self.received, self.owner_pid = batches, received, owner_pid
        self.stopped = threading.Event()
        self.index_map = index_map
        self.ordinal = self.samples = 0
        self.test_connection = test_connection

    def test_batches(self, *, step, global_batch_indices, batch_size, start_sample_index=0,
                     sample_count=None):
        if self.test_connection is None:
            raise RuntimeError("managed evaluation requires the shared sample broker")
        return _QueueTestBatches(self, step, global_batch_indices, batch_size, start_sample_index, sample_count)

    def _test_boundary(self, kind, step, details=()):
        connection = self.test_connection
        if connection is None:
            raise RuntimeError("managed tests require shared sample phase coordination")
        connection.send((kind, step, details))
        while not connection.poll(0.1):
            if self.stopped.is_set():
                raise RuntimeError("sample source closed during test phase transition")
            if os.getppid() != self.owner_pid:
                raise RuntimeError("sample supervisor exited during test phase transition")
        if connection.recv() != (kind, step):
            raise RuntimeError("invalid sample phase acknowledgement")

    def begin_test(self, step):
        self._test_boundary("begin-test", step)

    def end_test(self, step):
        self._test_boundary("end-test", step)

    def select_test_source(self, step, source):
        self._test_boundary("select-test", step, (source,))

    def __iter__(self):
        return self

    def __next__(self):
        while not self.stopped.is_set():
            if os.getppid() != self.owner_pid:
                raise RuntimeError("training supervisor exited while awaiting CPU samples")
            try:
                batch = self.batches.get(timeout=0.1)
            except queue.Empty:
                continue
            with self.received.get_lock():
                self.received.value += 1
            sizes = self.index_map.batch_sizes
            if not sizes:
                raise RuntimeError("zero-work rank received a training sample")
            size = sizes[self.ordinal % len(sizes)]
            expected = [self.index_map(self.samples + i) for i in range(size)]
            if batch.stream_indices.tolist() != expected:
                raise RuntimeError("CPU input differs from the canonical training sample sequence")
            self.ordinal += 1
            self.samples += size
            return batch
        raise StopIteration

    def release_current(self):
        # Physical H2D completion is not effective-batch consumption. The broker
        # retains these CPU slots so a cohort change can discard GPU lookahead.
        pass

    def close(self):
        self.stopped.set()


class _QueueTestBatches:
    def __init__(self, source, step, groups, batch_size, offset, sample_count):
        self.source, self.step = source, step
        self.groups = iter(groups)
        self.batch_size, self.offset = batch_size, offset
        self.current = None
        self.sample_count = sample_count

    def __iter__(self):
        return self

    def __next__(self):
        if self.current is not None:
            raise RuntimeError("previous test batch has not been released")
        group = next(self.groups)
        stop = (group + 1) * self.batch_size
        if self.sample_count is not None:
            stop = min(stop, self.sample_count)
        indices = tuple(range(self.offset + group * self.batch_size, self.offset + stop))
        connection = self.source.test_connection
        connection.send(("test", self.step, indices))
        while not connection.poll(0.1):
            if self.source.stopped.is_set():
                raise StopIteration
            if os.getppid() != self.source.owner_pid:
                raise RuntimeError("sample supervisor exited during evaluation loading")
        batch = connection.recv()
        if batch.stream_indices.tolist() != list(indices):
            raise RuntimeError("test data differs from canonical sample sequence")
        self.current = indices
        return batch

    def release_current(self):
        if self.current is not None:
            self.source.test_connection.send(("release-test", self.step, self.current))
            self.current = None

    def close(self):
        self.release_current()


@dataclass(frozen=True)
class FiberWorkerFactory:
    config_path: str
    microbatch_sizes: tuple[int, ...]
    queues: tuple
    received: tuple
    owner_pid: int
    rendezvous: str
    start_step: int
    effective_batch_size: int
    writer_factory: object = None
    resume_checkpoint: str | None = None
    training_overrides: object = None
    reserve_bytes: int = 0
    calibrated_gpu_names: tuple = ()
    calibrated_torch_versions: tuple = ()
    calibrated_cuda_versions: tuple = ()
    test_connections: tuple = ()
    trace_connections: tuple = ()
    measure_startup_memory: bool = False
    dataset_metadata: object = None

    def __call__(self, rank, grant, state):
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(len(self.microbatch_sizes)), LOCAL_RANK="0")
        return FiberTrainingWorker(self, rank, grant, state)


class FiberTrainingWorker:
    def __init__(self, factory, rank, grant, state):
        import torch
        from .train import _load_raw_config
        raw = _load_raw_config(factory.config_path)
        self.factory, self.rank = factory, rank
        self.gpu_uuid = sorted(grant["gpus"])[rank]
        self._measurement_initialization = 0
        self._saw_test = False
        self._completed_initial_test = False
        self._require_initial_test = state is None
        self.allowance = grant["gpus"][sorted(grant["gpus"])[rank]]
        self.reserve = factory.reserve_bytes
        self.cuda = str(raw.get("training", {}).get("device", "auto")).split(":")[0] in ("auto", "cuda")
        if self.cuda:
            from .cooperative_resources import configure_cuda_memory_budget
            properties = configure_cuda_memory_budget(self.allowance, self.reserve)
            if factory.calibrated_gpu_names and properties.name not in factory.calibrated_gpu_names:
                raise ValueError("allocated GPU model has no matching memory calibration")
            if factory.calibrated_torch_versions and str(torch.__version__) not in factory.calibrated_torch_versions:
                raise ValueError("PyTorch version differs from the measured memory calibration")
            if factory.calibrated_cuda_versions and torch.version.cuda not in factory.calibrated_cuda_versions:
                raise ValueError("CUDA version differs from the measured memory calibration")
            from .cooperative_resources import initialize_worker_volume_cache
            if raw.get("datasets"):
                initialize_worker_volume_cache(raw)
            if factory.measure_startup_memory:
                torch.cuda.reset_peak_memory_stats(0)
        from .train import training_units
        from .cooperative_resources import EffectiveBatchSampleIndex
        index_map = EffectiveBatchSampleIndex(factory.effective_batch_size, factory.microbatch_sizes,
                                              rank, factory.start_step,
                                              2 if raw.get("embedding", {}).get("paired_patches", False) else 1)
        self.source = QueueBatchSource(factory.queues[rank], factory.received[rank], factory.owner_pid, index_map,
                                       factory.test_connections[rank] if factory.test_connections else None)
        self.units = training_units(
            factory.config_path, resume_checkpoint=factory.resume_checkpoint,
            handoff=state, physical_microbatch_sizes=factory.microbatch_sizes,
            batch_source=self.source, summary_writer_factory=factory.writer_factory,
            dataset_metadata=factory.dataset_metadata,
            distributed_init_method=factory.rendezvous,
            training_overrides=factory.training_overrides,
            yield_initialized=True,
            enable_reviewed_trace=not factory.measure_startup_memory,
            on_model_ready=lambda model: setattr(self, "model", model),
            on_failure=lambda detail: self._error_reporter(detail) if hasattr(self, "_error_reporter") else None,
        )
        self.boundary = next(self.units)
        if self.boundary.phase != "initialized":
            self.units.close()
            raise RuntimeError("training must initialize without executing scheduled work")
        if factory.measure_startup_memory:
            torch.cuda.synchronize(0)
            self._measurement_initialization = int(torch.cuda.memory_reserved(0))
            self._measurement_test_peak = 0
            from .training_state import config_fingerprint
            self._measurement_fingerprint = config_fingerprint(raw)
        self._check_memory()

    def _check_memory(self):
        if self.cuda and self.factory.measure_startup_memory:
            import torch
            if torch.cuda.memory_reserved(0) + self.reserve > self.allowance:
                raise RuntimeError("training exceeded its calibrated VRAM allowance; recalibrate before retry")

    def set_error_reporter(self, reporter):
        self._error_reporter = reporter

    def run_unit(self):
        import faulthandler
        dump_after = float(os.environ.get("FIBER_UNIT_STACK_SECONDS", "0"))
        if dump_after > 0:
            faulthandler.dump_traceback_later(dump_after, repeat=True)
        try:
            return self._run_unit()
        finally:
            if dump_after > 0:
                faulthandler.cancel_dump_traceback_later()

    def _run_unit(self):
        connection = self.factory.trace_connections[self.rank] if self.factory.trace_connections else None
        if connection is not None and connection.poll():
            from .resident_trace import serve_resident
            kind = connection.recv()
            if kind != "trace":
                raise RuntimeError(f"invalid resident trace command: {kind}")
            arguments = connection.recv()
            serve_resident(connection, self.model, self.rank, arguments)
            self._check_memory()
            return {"step": self.boundary.step, "done": False,
                    "phase": "reviewed_trace_complete", "samples_seen": self.boundary.samples_seen}
        if not self.boundary.done:
            try:
                self.boundary = next(self.units)
            except Exception as error:
                if self.factory.measure_startup_memory:
                    import torch
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        raise RuntimeError(
                            f"real startup calibration exhausted VRAM at microbatch "
                            f"{self.factory.microbatch_sizes[self.rank]}; last completed "
                            f"boundary step={self.boundary.step} phase={self.boundary.phase}. "
                            "No update was retried or discarded; the run is stopping.") from error
                raise
        self._check_memory()
        row = {"step": self.boundary.step, "done": self.boundary.done, "phase": self.boundary.phase}
        if self.boundary.samples_seen is not None:
            row["samples_seen"] = self.boundary.samples_seen
        if self.factory.measure_startup_memory:
            import torch
            torch.cuda.synchronize(0)
            self._saw_test |= self.boundary.phase == "test"
            if self._require_initial_test and self.boundary.phase == "update" and not self._saw_test:
                raise RuntimeError("startup calibration produced no initial test work; cannot publish a complete memory profile")
            if self.boundary.phase == "test":
                self._measurement_test_peak = max(self._measurement_test_peak, int(torch.cuda.max_memory_reserved(0)))
            self._completed_initial_test |= self._saw_test and self.boundary.phase == "update"
            row["memory"] = {
                "microbatch_size": self.factory.microbatch_sizes[self.rank],
                "initialization_reserved_bytes": self._measurement_initialization,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "gpu_name": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda, "config_fingerprint": self._measurement_fingerprint,
                "world_size": len(self.factory.microbatch_sizes),
                "gpu_uuid": self.gpu_uuid,
                "optimizer_steps": self.boundary.step - self.factory.start_step,
                "initial_test_completed": self._completed_initial_test,
                "initial_test_peak_bytes": self._measurement_test_peak,
            }
        return row

    def capture(self):
        from .training_state import replica_signature
        state = self.boundary.capture()
        return replica_signature(state), state

    def close(self):
        # Wake a preparation thread blocked waiting for a future CPU batch
        # before the shared trainer joins it and destroys its process group.
        self.source.close()
        self.units.close()
