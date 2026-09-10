"""Reusable Fiber3D operations for the ordered per-sample CPU pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import queue
import threading
import time
import warnings
from typing import Any, Callable

import torch

from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    FiberTrace3DBatch,
    FiberTrace3DLoader,
    FiberTrace3DRawSample,
    FiberTrace3DSample,
    FiberTrace3DSharedTensorSlots,
    augment_fiber_trace_3d_raw_sample,
    apply_value_augmentation_batch_3d,
    apply_value_augmentation_suffix_batch_3d,
    collate_fiber_trace_3d_samples,
    load_config,
    prepare_fiber_trace_3d_cuda_suffix,
)
from vesuvius.neural_tracing.ordered_sample_pipeline import (
    OrderedTwoStageSamplePipeline,
)
from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets


@dataclass(frozen=True)
class IdentitySampleIndex:
    def __call__(self, index: int) -> int:
        return int(index)


@dataclass(frozen=True)
class DistributedSampleIndex:
    """Map a rank-local consecutive stream onto the canonical DDP sample stream."""

    batch_size: int
    world_size: int
    rank: int
    start_step: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.world_size <= 0:
            raise ValueError("batch_size and world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.start_step < 0:
            raise ValueError("start_step must be nonnegative")

    def __call__(self, index: int) -> int:
        local_batch, offset = divmod(int(index), int(self.batch_size))
        global_batch = (
            (int(self.start_step) + local_batch) * int(self.world_size) + int(self.rank)
        )
        return global_batch * int(self.batch_size) + offset


@dataclass(frozen=True)
class VirtualBatchSampleIndex:
    """Map physical-rank work onto full logical rank-local batches."""

    batch_size: int
    physical_world_size: int
    rank: int
    virtual_world_size: int
    start_step: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.physical_world_size <= 0:
            raise ValueError("batch_size and physical_world_size must be positive")
        if self.virtual_world_size < self.physical_world_size:
            raise ValueError(
                "virtual_world_size must be at least physical_world_size"
            )
        if not 0 <= self.rank < self.physical_world_size:
            raise ValueError("rank must be in [0, physical_world_size)")
        if self.start_step < 0:
            raise ValueError("start_step must be nonnegative")

    @property
    def local_batches_per_step(self) -> int:
        return (
            int(self.virtual_world_size)
            - int(self.rank)
            + int(self.physical_world_size)
            - 1
        ) // int(self.physical_world_size)

    def __call__(self, index: int) -> int:
        local_batch, offset = divmod(int(index), int(self.batch_size))
        optimizer_step, local_batch_in_step = divmod(
            local_batch,
            self.local_batches_per_step,
        )
        virtual_rank = (
            int(self.rank)
            + local_batch_in_step * int(self.physical_world_size)
        )
        if virtual_rank >= int(self.virtual_world_size):
            raise RuntimeError("virtual batch mapping produced an invalid rank")
        global_batch = (
            (int(self.start_step) + optimizer_step) * int(self.virtual_world_size)
            + virtual_rank
        )
        return global_batch * int(self.batch_size) + offset


@dataclass(frozen=True)
class FiberTrace3DLoadOperation:
    loader: FiberTrace3DLoader
    sample_index_limit: int | None
    sample_mode: str
    index_map: Callable[[int], int]
    profile: bool

    def __call__(self, index: int) -> FiberTrace3DRawSample:
        return self.loader.load_raw_sample(
            self.index_map(index),
            sample_index_limit=self.sample_index_limit,
            sample_mode=self.sample_mode,
            device="cpu",
            profile=self.profile,
        )


@dataclass(frozen=True)
class FiberTrace3DLoadFactory:
    config_path: str
    sample_index_limit: int | None
    sample_mode: str
    index_map: Callable[[int], int] = IdentitySampleIndex()
    profile: bool = True
    single_core: bool = False

    def __call__(self) -> FiberTrace3DLoadOperation:
        warnings.filterwarnings(
            "ignore",
            message=r"skipping NML fiber component with branch node:.*",
            category=RuntimeWarning,
        )
        config = load_config(self.config_path)
        if self.single_core:
            from .cooperative_resources import initialize_worker_volume_cache
            initialize_worker_volume_cache({"volume_cache_memory_mib": config.volume_cache_memory_mib})
            config = replace(config, volume_io_threads=1, volume_decode_threads=1, volume_sample_threads=1)
        return FiberTrace3DLoadOperation(
            FiberTrace3DLoader(config),
            self.sample_index_limit,
            self.sample_mode,
            self.index_map,
            self.profile,
        )


@dataclass(frozen=True)
class FiberTrace3DFusedLoadOperation:
    loader: FiberTrace3DLoader
    sample_index_limit: int | None
    sample_mode: str
    index_map: Callable[[int], int]
    profile: bool

    def __call__(self, index: int) -> FiberTrace3DSample:
        return self.loader.load_sample(
            self.index_map(index),
            sample_index_limit=self.sample_index_limit,
            sample_mode=self.sample_mode,
            device="cpu",
            profile=self.profile,
        )


@dataclass(frozen=True)
class FiberTrace3DFusedLoadFactory:
    config_path: str
    sample_index_limit: int | None
    sample_mode: str
    index_map: Callable[[int], int] = IdentitySampleIndex()
    profile: bool = True

    def __call__(self) -> FiberTrace3DFusedLoadOperation:
        warnings.filterwarnings(
            "ignore",
            message=r"skipping NML fiber component with branch node:.*",
            category=RuntimeWarning,
        )
        return FiberTrace3DFusedLoadOperation(
            FiberTrace3DLoader(load_config(self.config_path)),
            self.sample_index_limit,
            self.sample_mode,
            self.index_map,
            self.profile,
        )


@dataclass(frozen=True)
class FiberTrace3DAugmentOperation:
    config: object
    profile: bool
    defer_cuda_suffix: str | None = None

    def __call__(self, raw: FiberTrace3DRawSample) -> FiberTrace3DSample | FiberTrace3DRawSample:
        if self.defer_cuda_suffix:
            return prepare_fiber_trace_3d_cuda_suffix(
                self.config,
                raw,
                profile=self.profile,
                mode=self.defer_cuda_suffix,
            )
        return augment_fiber_trace_3d_raw_sample(
            self.config,
            raw,
            profile=self.profile,
        )


@dataclass(frozen=True)
class FiberTrace3DAugmentFactory:
    config_path: str
    profile: bool = True
    defer_cuda_suffix: str | None = None

    def __call__(self) -> FiberTrace3DAugmentOperation:
        return FiberTrace3DAugmentOperation(
            load_config(self.config_path),
            self.profile,
            self.defer_cuda_suffix,
        )


def collate_pipeline_results(results, volumes, *, defer_cuda_value_augmentation=None,
                             profile=False, total_start=None, cpu_start=None, wall_start_ns=None):
    """Shared collation for rank-local and persistent canonical sample owners."""
    samples = []
    deferred_kernels, deferred_params = [], []
    for result in results:
        value = result.value
        if defer_cuda_value_augmentation is not None:
            if not isinstance(value, FiberTrace3DRawSample):
                raise RuntimeError("training sample pipeline lost deferred augmentation")
            sample = value.sample
            deferred_kernels.append(value.blur_kernels)
            deferred_params.append(value.augmentation)
        else:
            if not isinstance(value, FiberTrace3DSample):
                raise RuntimeError("training sample pipeline returned an invalid sample")
            sample = value
        if profile:
            timings = dict(sample.profile_timings_ms or {})
            timings.update(
                pipeline_stage1_service_ms=float(result.stage1_service_ms),
                pipeline_stage1_cpu_ms=float(result.stage1_cpu_ms),
                pipeline_raw_queue_ms=float(result.raw_queue_ms),
                pipeline_stage2_service_ms=float(result.stage2_service_ms),
                pipeline_stage2_cpu_ms=float(result.stage2_cpu_ms),
                pipeline_result_queue_ms=float(result.result_queue_ms),
                pipeline_reorder_wait_ms=float(result.reorder_wait_ms),
            )
            sample = replace(sample, profile_timings_ms=timings)
        samples.append(sample)
    return collate_fiber_trace_3d_samples(
        samples, volume_batch=volumes,
        deferred_augmentation_params=tuple(deferred_params) if defer_cuda_value_augmentation is not None else None,
        deferred_blur_kernels=tuple(deferred_kernels) if defer_cuda_value_augmentation is not None else None,
        profile=profile, total_start=total_start, cpu_total_start=cpu_start, wall_start_ns=wall_start_ns,
    )


class FiberTrace3DTrainingBatchPipeline:
    """Present deterministic batches backed by the bounded per-sample pipeline."""

    def __init__(
        self,
        *,
        config_path: str,
        index_map: Callable[[int], int],
        batch_size: int,
        patch_shape_zyx: tuple[int, int, int],
        batch_count: int,
        workers: int,
        load_affinity_workers: int,
        slot_count: int,
        initial_in_flight: int,
        sample_index_limit: int | None,
        sample_mode: str,
        profile: bool,
        defer_cuda_value_augmentation: str | None = None,
        batch_sizes: tuple[int, ...] | None = None,
        single_core_load: bool = False,
    ) -> None:
        if int(batch_count) <= 0:
            raise ValueError("batch_count must be positive")
        if int(slot_count) % int(batch_size) != 0:
            raise ValueError("slot_count must be a multiple of batch_size")
        if int(initial_in_flight) < int(batch_size):
            raise ValueError("initial_in_flight must be at least one batch")
        self.batch_size = int(batch_size)
        self.batch_sizes = (self.batch_size,) if batch_sizes is None else tuple(batch_sizes)
        if not self.batch_sizes or any(type(size) is not int or not 0 < size <= self.batch_size for size in self.batch_sizes):
            raise ValueError("batch_sizes must be a nonempty pattern within batch_size")
        if int(batch_count) % len(self.batch_sizes):
            raise ValueError("batch_count must cover whole batch-size patterns")
        self._batch_count, self._batch_ordinal = int(batch_count), 0
        self._variable_batches = batch_sizes is not None
        self.profile = bool(profile)
        self.defer_cuda_value_augmentation = defer_cuda_value_augmentation
        self._leased: list[int] = []
        self.transport = FiberTrace3DSharedTensorSlots(
            start_index=0,
            slot_count=int(slot_count),
            patch_shape_zyx=patch_shape_zyx,
            stream_index_map=index_map,
            volume_dtype=(
                torch.uint8
                if self.defer_cuda_value_augmentation == "all"
                else torch.float32
            ),
        )
        self.pipeline = OrderedTwoStageSamplePipeline(
            stage1_factory=FiberTrace3DLoadFactory(
                config_path,
                sample_index_limit,
                sample_mode,
                index_map=index_map,
                profile=profile,
                single_core=single_core_load,
            ),
            stage2_factory=FiberTrace3DAugmentFactory(
                config_path,
                profile=profile,
                defer_cuda_suffix=self.defer_cuda_value_augmentation,
            ),
            start_index=0,
            item_count=(int(batch_count) // len(self.batch_sizes)) * sum(self.batch_sizes),
            stage1_workers=0,
            stage2_workers=0,
            raw_queue_capacity=max(self.batch_size, min(int(slot_count), int(workers))),
            result_queue_capacity=max(self.batch_size, min(int(slot_count), int(workers))),
            max_in_flight=int(slot_count),
            multiprocessing_context="forkserver",
            transport=self.transport,
            shared_workers=int(workers),
            initial_in_flight=int(initial_in_flight),
            shared_stage1_affinity_workers=int(load_affinity_workers),
        )

    def __iter__(self) -> "FiberTrace3DTrainingBatchPipeline":
        return self

    def __next__(self) -> FiberTrace3DBatch:
        if self._leased:
            raise RuntimeError("previous Fiber3D pipeline batch was not released")
        if self._batch_ordinal >= self._batch_count:
            raise StopIteration
        take = self.batch_sizes[self._batch_ordinal % len(self.batch_sizes)]
        total_start = time.perf_counter()
        cpu_start = time.process_time()
        wall_start_ns = time.time_ns()
        results = [
            self.pipeline.next_ordered(release_credit=False)
            for _ in range(take)
        ]
        indices = [int(result.index) for result in results]
        volumes = self.transport.batch_tensors(indices, allow_wrap=self._variable_batches)
        batch = collate_pipeline_results(
            results, volumes, defer_cuda_value_augmentation=self.defer_cuda_value_augmentation,
            profile=self.profile, total_start=total_start, cpu_start=cpu_start, wall_start_ns=wall_start_ns,
        )
        self._leased = indices
        self._batch_ordinal += 1
        return batch

    def release_current(self) -> None:
        for index in self._leased:
            self.pipeline.release(index)
        self._leased.clear()

    def close(self) -> None:
        self.release_current()
        self.pipeline.close()


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
        # Resolve the caller's current device before starting the preparation
        # thread: CUDA's current device is thread-local, and set_device rejects
        # an index-less torch.device('cuda').
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
                        if device_batch.deferred_blur_kernels is not None:
                            if self.value_augmentation_suffix == "all":
                                if device_batch.deferred_augmentation_params is None:
                                    raise RuntimeError("CUDA augmentation parameters are missing")
                                augmented = apply_value_augmentation_batch_3d(
                                    self.config,
                                    device_batch.volume,
                                    device_batch.deferred_augmentation_params,
                                    device_batch.deferred_blur_kernels,
                                )
                            else:
                                augmented = apply_value_augmentation_suffix_batch_3d(
                                    device_batch.volume,
                                    device_batch.deferred_blur_kernels,
                                )
                            device_batch = replace(
                                device_batch,
                                volume=augmented,
                                deferred_augmentation_params=None,
                                deferred_blur_kernels=None,
                            )
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
