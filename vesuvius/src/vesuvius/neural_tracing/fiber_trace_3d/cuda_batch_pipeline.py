"""Bounded asynchronous transfer and target preparation, extracted from Fiber3D."""
from __future__ import annotations

from dataclasses import dataclass, replace
import queue
import threading
import time
from typing import Any

import torch

from .loader import FiberTrace3DBatch
from .sample_pipeline import FiberTrace3DTrainingBatchPipeline, apply_deferred_batch_augmentation
from .targets import materialize_targets


@dataclass(frozen=True)
class _CudaPreparedBatch:
    batch: FiberTrace3DBatch
    pinned_batch: FiberTrace3DBatch
    transfer_started: torch.cuda.Event
    transfer_finished: torch.cuda.Event
    augmentation_finished: torch.cuda.Event
    preparation_finished: torch.cuda.Event
    queued_at: float
    source_wait_ms: float
    pin_memory_ms: float
    pinned_bytes: int


@dataclass(frozen=True)
class _PreparationFailure:
    error: BaseException


_PREPARATION_END = object()


def _batch_tensor_bytes(batch: FiberTrace3DBatch) -> int:
    return sum(
        int(value.numel()) * int(value.element_size())
        for value in vars(batch).values()
        if isinstance(value, torch.Tensor)
    )


class FiberTrace3DCudaBatchPipeline:
    """Prepare two future training batches on a dedicated CUDA stream."""

    def __init__(
        self,
        source: FiberTrace3DTrainingBatchPipeline,
        *,
        config: Any,
        device: torch.device,
        output_stride: int,
        depth: int = 2,
        value_augmentation_suffix: str | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA batch preparation requires a CUDA device")
        if int(depth) <= 0:
            raise ValueError("CUDA batch preparation depth must be positive")
        self.source = source
        self.config = config
        # CUDA current-device state is thread-local; resolve it in the caller.
        self.device = (torch.device("cuda", torch.cuda.current_device())
                       if device.index is None else device)
        self.output_stride = int(output_stride)
        self.depth = int(depth)
        self.value_augmentation_suffix = value_augmentation_suffix
        self.stream = torch.cuda.Stream(device=self.device)
        self._results: queue.Queue[object] = queue.Queue()
        self._available = threading.Semaphore(self.depth)
        self._stop = threading.Event()
        self._pause_condition = threading.Condition()
        self._paused = False
        self._submit_lock = threading.Lock()
        self._closed = False
        self.max_pinned_bytes = 0
        self._thread = threading.Thread(
            target=self._run,
            name="fiber3d-cuda-batch-preparation",
            daemon=False,
        )
        self._thread.start()

    def __iter__(self) -> "FiberTrace3DCudaBatchPipeline":
        return self

    def _wait_until_running(self) -> bool:
        with self._pause_condition:
            self._pause_condition.wait_for(
                lambda: self._stop.is_set() or not self._paused
            )
            return not self._stop.is_set()

    def _submit_cuda(self, pinned: FiberTrace3DBatch) -> _CudaPreparedBatch | None:
        while self._wait_until_running():
            with self._submit_lock:
                with self._pause_condition:
                    if self._paused:
                        continue
                with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                    transfer_started = torch.cuda.Event(enable_timing=True)
                    transfer_finished = torch.cuda.Event(enable_timing=True)
                    augmentation_finished = torch.cuda.Event(enable_timing=True)
                    preparation_finished = torch.cuda.Event(enable_timing=True)
                    transfer_started.record(self.stream)
                    device_batch = pinned.to(self.device, non_blocking=True)
                    transfer_finished.record(self.stream)
                    with torch.autocast(device_type="cuda", enabled=False):
                        device_batch = apply_deferred_batch_augmentation(
                            device_batch, self.config, mode=self.value_augmentation_suffix)
                    augmentation_finished.record(self.stream)
                    device_batch = materialize_targets(
                        device_batch,
                        self.config,
                        profile=False,
                        output_stride=self.output_stride,
                    )
                    preparation_finished.record(self.stream)
                return _CudaPreparedBatch(
                    batch=device_batch,
                    pinned_batch=pinned,
                    transfer_started=transfer_started,
                    transfer_finished=transfer_finished,
                    augmentation_finished=augmentation_finished,
                    preparation_finished=preparation_finished,
                    queued_at=time.perf_counter(),
                    source_wait_ms=0.0,
                    pin_memory_ms=0.0,
                    pinned_bytes=_batch_tensor_bytes(pinned),
                )
        return None

    def _run(self) -> None:
        try:
            torch.cuda.set_device(self.device)
            while not self._stop.is_set():
                self._available.acquire()
                if self._stop.is_set():
                    self._available.release()
                    break
                if not self._wait_until_running():
                    self._available.release()
                    break
                source_started = time.perf_counter()
                try:
                    cpu_batch = next(self.source)
                except StopIteration:
                    self._results.put(_PREPARATION_END)
                    return
                source_wait_ms = (time.perf_counter() - source_started) * 1000.0
                pin_started = time.perf_counter()
                try:
                    pinned = cpu_batch.pin_memory()
                finally:
                    self.source.release_current()
                pin_memory_ms = (time.perf_counter() - pin_started) * 1000.0
                prepared = self._submit_cuda(pinned)
                if prepared is None:
                    self._available.release()
                    break
                prepared = replace(
                    prepared,
                    source_wait_ms=source_wait_ms,
                    pin_memory_ms=pin_memory_ms,
                )
                self.max_pinned_bytes = max(
                    self.max_pinned_bytes,
                    prepared.pinned_bytes * (self.depth + 1),
                )
                self._results.put(prepared)
        except BaseException as error:
            self._results.put(_PreparationFailure(error))

    def __next__(self) -> FiberTrace3DBatch:
        if self._closed:
            raise RuntimeError("CUDA batch preparation pipeline is closed")
        wait_started = time.perf_counter()
        result = self._results.get()
        self._available.release()
        queue_wait_ms = (time.perf_counter() - wait_started) * 1000.0
        if result is _PREPARATION_END:
            raise StopIteration
        if isinstance(result, _PreparationFailure):
            raise RuntimeError("Fiber3D CUDA batch preparation failed") from result.error
        if not isinstance(result, _CudaPreparedBatch):
            raise RuntimeError("Fiber3D CUDA batch preparation returned invalid state")
        ready_wait_started = time.perf_counter()
        result.preparation_finished.synchronize()
        ready_wait_ms = (time.perf_counter() - ready_wait_started) * 1000.0
        transfer_ms = result.transfer_started.elapsed_time(result.transfer_finished)
        augmentation_ms = result.transfer_finished.elapsed_time(
            result.augmentation_finished
        )
        target_ms = result.augmentation_finished.elapsed_time(result.preparation_finished)
        current_stream = torch.cuda.current_stream(self.device)
        result.batch.record_stream(current_stream)
        profile_timings = dict(result.batch.profile_timings_ms or {})
        profile_timings.update(
            cuda_pipeline_source_wait_ms=float(result.source_wait_ms),
            cuda_pipeline_pin_memory_ms=float(result.pin_memory_ms),
            cuda_pipeline_queue_residence_ms=max(
                0.0, (time.perf_counter() - result.queued_at) * 1000.0
            ),
            cuda_pipeline_exposed_queue_wait_ms=float(queue_wait_ms),
            cuda_pipeline_exposed_event_wait_ms=float(ready_wait_ms),
            cuda_pipeline_transfer_ms=float(transfer_ms),
            cuda_pipeline_value_augmentation_ms=float(augmentation_ms),
            cuda_pipeline_target_ms=float(target_ms),
            cuda_pipeline_pinned_bytes=float(result.pinned_bytes),
        )
        return replace(result.batch, profile_timings_ms=profile_timings)

    def pause(self) -> None:
        if self._closed:
            return
        with self._pause_condition:
            self._paused = True
        with self._submit_lock:
            self.stream.synchronize()

    def resume(self) -> None:
        if self._closed:
            return
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()
        for _ in range(self.depth + 1):
            self._available.release()
        self.source.close()
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("Fiber3D CUDA batch preparation thread did not stop")
        while True:
            try:
                self._results.get_nowait()
            except queue.Empty:
                break
