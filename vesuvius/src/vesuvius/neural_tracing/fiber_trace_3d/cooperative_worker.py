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

    def __init__(self, batches, received, owner_pid, index_map):
        self.batches, self.received, self.owner_pid = batches, received, owner_pid
        self.stopped = threading.Event()
        self.index_map = index_map
        self.ordinal = self.samples = 0

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

    def __call__(self, rank, grant, state):
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(len(self.microbatch_sizes)), LOCAL_RANK="0")
        return FiberTrainingWorker(self, rank, grant, state)


class FiberTrainingWorker:
    def __init__(self, factory, rank, grant, state):
        import torch
        from .train import _load_raw_config
        raw = _load_raw_config(factory.config_path)
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
        from .train import training_units
        from .cooperative_resources import EffectiveBatchSampleIndex
        index_map = EffectiveBatchSampleIndex(factory.effective_batch_size, factory.microbatch_sizes,
                                              rank, factory.start_step)
        self.source = QueueBatchSource(factory.queues[rank], factory.received[rank], factory.owner_pid, index_map)
        self.units = training_units(
            factory.config_path, resume_checkpoint=factory.resume_checkpoint,
            handoff=state, physical_microbatch_sizes=factory.microbatch_sizes,
            batch_source=self.source, summary_writer_factory=factory.writer_factory,
            distributed_init_method=factory.rendezvous,
            training_overrides=factory.training_overrides,
        )
        self.boundary = next(self.units)
        self._check_memory()

    def _check_memory(self):
        if self.cuda:
            import torch
            if torch.cuda.memory_reserved(0) + self.reserve > self.allowance:
                raise RuntimeError("training exceeded its calibrated VRAM allowance; recalibrate before retry")

    def run_unit(self):
        if not self.boundary.done:
            self.boundary = next(self.units)
        self._check_memory()
        return {"step": self.boundary.step, "done": self.boundary.done, "phase": self.boundary.phase}

    def capture(self):
        from .training_state import replica_signature
        state = self.boundary.capture()
        return replica_signature(state), state

    def close(self):
        # Wake a preparation thread blocked waiting for a future CPU batch
        # before the shared trainer joins it and destroys its process group.
        self.source.close()
        self.units.close()
